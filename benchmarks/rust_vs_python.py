"""Time every ported Rust kernel against the Python implementation it replaces (RUST-011).

``tests/parity_registry.py`` is the canonical list of ported kernel surfaces, so
this harness takes the entries from it rather than naming kernels again: for each
entry it times the same call twice, once through the Rust extension behind the
entry's dispatch module and once with every dispatch module switched to the
pure-Python path, and reports the best of N wall-clock timings after a warm-up
call.

Four measurements beyond that steady-state table:

- the extension's import cost and each entry's first call in a fresh
  interpreter (cold), against the warm numbers;
- a cell-count sweep of the spatial-block entries, whose least-squares intercept
  is the fixed per-call cost (Python orchestration, validation, the binding
  crossing, and the copy in) and whose slope is the per-cell kernel cost -- the
  small-input case where the boundary dominates;
- thread scaling of four disjoint cell blocks, with the backend on and off, for the
  question RUST-011's Rayon decision rests on: the kernels release the GIL, so an
  outer thread pool already parallelizes them;
- one gridded Dask SPI call per scheduler and error policy, plus a separate probe
  of a default spawned worker's dispatch guard inputs. Thread workers receive the
  requested policy explicitly; process rows set only the caller's policy.

Rust-labeled measurements ignore NumPy floating-point errors and clear warning
filters locally. Context-aware warnings disable native dispatch and are rejected.

``--netcdf`` replaces the routine measurements with the occasional full-grid run:
a subset of the registry's entries, computed from every land cell of a real
precipitation grid (nClimGrid-Monthly, prepared with
``benchmarks/cli_multiprocessing.py prepare``) and the matching mean temperature
grid, each as one eager call and across a thread pool at several thread counts,
through each backend, with no warm-up call. Rust rows record the share of wall
clock spent inside extension calls and whether every kernel the entry registers
ran. It takes tens of minutes and several GB of memory, so it is never the
default.

Run from the repository root, with the extension built (``uv run maturin develop
--release``)::

    uv run benchmarks/rust_vs_python.py
    uv run benchmarks/rust_vs_python.py --repeat 7 --write
    uv run benchmarks/rust_vs_python.py --skip-sweep --skip-threads --skip-dask
    uv run benchmarks/rust_vs_python.py --netcdf nclimgrid_prcp_1981_2024.nc --tavg nclimgrid_tavg.nc --repeat 2 --write
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import subprocess  # nosec B404 # only re-runs this interpreter with fixed arguments
import sys
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parallel_scaling  # noqa: E402  (a sibling script: benchmarks/ is on sys.path when it runs)
import pytest  # noqa: E402  (pytest.MonkeyPatch is the tests' backend switch, used directly)

from climate_indices import compute, eto, indices, palmer  # noqa: E402
from tests import conftest, parity_registry  # noqa: E402
from tests.parity_registry import ENTRIES, SAMPLES, Entry  # noqa: E402

# entries swept over the number of cells in the block, and the cell counts swept
SWEEP_ENTRIES = ("spi_gamma_spatial_block", "eddi_spatial_block")
SWEEP_CELLS = (1, 2, 4, 8, 16, 32)

# the synthetic block behind the thread-scaling measurement: enough cells per thread that the
# kernel time, not the per-call cost, is what a thread count changes
THREAD_CELLS = 4096
THREAD_YEARS = 40
THREAD_WORKERS = (1, 2, 4)

# the gridded Dask measurement: the docs/performance.md example grid, one compute per scheduler
DASK_SIDE = 25
DASK_YEARS = 40
DASK_SCALE = 3

# entries whose first call the cold measurement times, one per dispatch module
COLD_ENTRIES = ("spi_gamma", "thornthwaite", "fire_kbdi", "flood_api")

# the full-grid run: the recursion family, the fit family, and the entries slowest relative to Python
GRID_ENTRIES = ("spi_gamma", "spi_pearson", "spei_gamma", "eddi", "thornthwaite", "palmer_pdsi")
GRID_WORKERS = (1, 2, 4, 8)
GRID_CALIBRATION = (1991, 2020)
# a stand-in: nClimGrid's soil constants are not published at the 5 km grid's resolution
GRID_AWC_INCHES = 6.0
MM_PER_INCH = 25.4

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "rust_vs_python.txt"
DEFAULT_GRID_OUTPUT = DEFAULT_OUTPUT.with_name("rust_vs_python_nclimgrid.txt")


@dataclass(frozen=True)
class Timings:
    """One entry's best-of-N wall-clock seconds through each backend."""

    name: str
    family: str
    rust_seconds: float
    python_seconds: float

    @property
    def ratio(self) -> float:
        """Python seconds per Rust second; above 1 means the Rust kernel is faster."""
        return self.python_seconds / self.rust_seconds

    @property
    def rust_faster(self) -> bool:
        """Whether the Rust path beat the Python path on this run."""
        return self.rust_seconds < self.python_seconds


@dataclass(frozen=True)
class Comparison:
    """One configuration's timed samples through each backend, and what its Rust run reached."""

    name: str
    workers: int | None  # None: one eager call; otherwise a pool of this many threads over fixed blocks
    rust_samples: tuple[float, ...]
    python_samples: tuple[float, ...]
    rust_kernels_reached: bool
    extension_seconds: float  # wall clock inside extension calls, summed over the Rust samples

    @property
    def rust_seconds(self) -> float:
        """Best Rust sample."""
        return min(self.rust_samples)

    @property
    def python_seconds(self) -> float:
        """Best Python sample."""
        return min(self.python_samples)

    @property
    def extension_share(self) -> float:
        """Fraction of the Rust wall clock spent inside extension calls; meaningful for one thread only."""
        return self.extension_seconds / sum(self.rust_samples)


class KernelTimer(conftest.NativeRecorder):
    """The tests' recorder, also summing the wall clock spent inside the extension's calls."""

    def __init__(self, module: ModuleType) -> None:
        super().__init__(module)
        self.seconds = 0.0

    def __getattr__(self, name: str) -> Any:
        attribute = super().__getattr__(name)
        if isinstance(attribute, type):
            return attribute

        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return attribute(*args, **kwargs)
            finally:
                self.seconds += time.perf_counter() - start

        return timed


@dataclass(frozen=True)
class ThreadTiming:
    """One thread count's wall-clock seconds through each backend."""

    workers: int
    rust_seconds: float
    python_seconds: float


@dataclass(frozen=True)
class SchedulerTiming:
    """One gridded call's seconds and whether it reached the Rust kernels."""

    scheduler: str
    error_policy: str
    seconds: float
    rust_kernels_reached: bool | None


@contextmanager
def native_policy() -> Iterator[None]:
    """Establish the native guard's reporting policy without changing caller settings."""
    if getattr(sys.flags, "context_aware_warnings", False):
        raise RuntimeError("Rust measurements require -X context_aware_warnings=0")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.resetwarnings()  # the guard rejects any error filter, even one overridden by ignore
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def measure(run: Callable[[], Any], repeats: int, error_policy: str = "ignore") -> float:
    """Return the best of ``repeats`` wall-clock timings of ``run``, after a warm-up call.

    Args:
        run: the call to time
        repeats: number of timed repetitions, the minimum of which is returned
        error_policy: ``ignore`` establishes the native NumPy and warning policy;
            ``default`` leaves the caller's policy alone

    Returns:
        the best wall-clock duration in seconds
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    context = native_policy() if error_policy == "ignore" else nullcontext()
    with context:
        run()  # warm-up: first-call allocation and import costs are the cold measurement's subject
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            best = min(best, time.perf_counter() - start)
    return best


@contextmanager
def python_backend() -> Iterator[None]:
    """Switch every dispatch module to the pure-Python path for the duration of the block."""
    patch = pytest.MonkeyPatch()
    conftest.disable_native(patch)
    try:
        yield
    finally:
        patch.undo()


def backend_events(
    rust_run: Callable[[], Any],
    python_run: Callable[[], Any],
    repeats: int,
    warm_up: bool = True,
) -> Iterator[tuple[str, float]]:
    """Yield each timed call's backend and wall-clock seconds, swapping which backend runs first.

    Timing one backend's repetitions before the other's lets clock, cache, and thermal drift
    favour one side, so each repetition swaps which backend runs first. With ``warm_up``, the
    first repetition is each backend's untimed call -- it sets the opening order but is not
    yielded, because first-call allocation and import costs are the cold measurement's subject.

    Args:
        rust_run: the call to time through the Rust kernels
        python_run: the call to time on the pure-Python path
        repeats: number of timed repetitions per backend
        warm_up: make one untimed call per backend first

    Yields:
        the backend (``rust`` or ``python``) and its seconds, in call order
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    with native_policy():
        for repetition in range(repeats + 1 if warm_up else repeats):
            for backend in ("rust", "python") if repetition % 2 else ("python", "rust"):
                with python_backend() if backend == "python" else nullcontext():
                    run = python_run if backend == "python" else rust_run
                    start = time.perf_counter()
                    run()
                    elapsed = time.perf_counter() - start
                if warm_up and not repetition:
                    continue
                yield backend, elapsed


def measure_backends(rust_run: Callable[[], Any], python_run: Callable[[], Any], repeats: int) -> tuple[float, float]:
    """Return the best Rust and Python seconds of ``repeats`` interleaved repetitions, after a warm-up of each.

    Args:
        rust_run: the call to time through the Rust kernels
        python_run: the call to time on the pure-Python path
        repeats: number of timed repetitions per backend

    Returns:
        the best Rust seconds and the best Python seconds
    """
    best = {"rust": float("inf"), "python": float("inf")}
    for backend, elapsed in backend_events(rust_run, python_run, repeats):
        best[backend] = min(best[backend], elapsed)
    return best["rust"], best["python"]


def time_entry(entry: Entry, values: np.ndarray | None = None, repeats: int = 3) -> Timings:
    """Time one registry entry through the Rust kernels and on the Python path.

    Args:
        entry: the registry entry to time
        values: the input to draw the entry's call from; the family's fixed sample when omitted
        repeats: number of timed repetitions per backend

    Returns:
        the entry's best-of-``repeats`` seconds per backend
    """
    drawn = SAMPLES[entry.family] if values is None else values
    rust_seconds, python_seconds = measure_backends(entry.run(drawn), entry.run(drawn), repeats)
    return Timings(entry.name, entry.family, rust_seconds, python_seconds)


def time_entries(entries: Sequence[Entry] = ENTRIES, repeats: int = 3) -> list[Timings]:
    """Time every entry in ``entries``."""
    return [time_entry(entry, repeats=repeats) for entry in entries]


def cold_call_seconds(entry_name: str) -> float:
    """Time one entry's first call, in the fresh interpreter that ``--cold-entry`` runs in."""
    entry = parity_registry.ENTRIES_BY_NAME[entry_name]
    run = entry.run(SAMPLES[entry.family])
    with native_policy():
        start = time.perf_counter()
        run()
        return time.perf_counter() - start


def measure_cold(entry_name: str) -> float:
    """Run :func:`cold_call_seconds` in a fresh interpreter and return its reported seconds."""
    try:
        completed = subprocess.run(  # nosec B603 # this interpreter and this file; no shell
            [sys.executable, str(Path(__file__).resolve()), "--cold-entry", entry_name],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"cold entry {entry_name} failed: {error.stderr}") from error
    for line in reversed(completed.stdout.splitlines()):
        try:
            return float(json.loads(line)["seconds"])
        except (ValueError, KeyError, TypeError):
            continue
    raise RuntimeError(f"cold entry {entry_name} returned no timing: {completed.stdout}\n{completed.stderr}")


def measure_import_seconds() -> float:
    """Return the wall clock of ``import climate_indices`` in a fresh interpreter, interpreter start-up removed."""

    def interpreter(statement: str) -> float:
        start = time.perf_counter()
        subprocess.run([sys.executable, "-c", statement], check=True, capture_output=True)  # nosec B603 # literal statements
        return time.perf_counter() - start

    return interpreter("import climate_indices") - interpreter("pass")


def sweep_entry(entry: Entry, cells: Sequence[int] = SWEEP_CELLS, repeats: int = 3) -> list[tuple[int, Timings]]:
    """Time one spatial-block entry at each cell count, retaining its measured count."""
    years = SAMPLES[entry.family].shape[0] // 12
    return [
        (
            count,
            time_entry(
                entry, values=parity_registry._block_values(years, count, 20261012, 0.15, 0.02), repeats=repeats
            ),
        )
        for count in cells
    ]


def fit_fixed_and_per_cell(sweep: Sequence[tuple[int, Timings]], backend: str) -> tuple[float, float]:
    """Least-squares intercept and slope of one sweep's time against cell count, in seconds.

    Args:
        sweep: measured cell counts paired with their timings
        backend: ``rust`` or ``python``

    Returns:
        the fixed per-call seconds and the seconds added per cell
    """
    cells = np.array([count for count, _ in sweep], dtype=float)
    seconds = np.array([getattr(timing, f"{backend}_seconds") for _, timing in sweep], dtype=float)
    slope, intercept = np.polyfit(cells, seconds, 1)
    return float(intercept), float(slope)


def thread_scaling(
    workers: Sequence[int] = THREAD_WORKERS,
    cells: int = THREAD_CELLS,
    years: int = THREAD_YEARS,
    repeats: int = 3,
) -> list[ThreadTiming]:
    """Time disjoint cell blocks across ``workers`` threads, with the backend on and off.

    Args:
        workers: thread counts to measure
        cells: number of cells in the block, split evenly across the widest thread count
        years: record length of the synthetic block
        repeats: number of timed repetitions per thread count and backend

    Returns:
        one :class:`ThreadTiming` per thread count
    """
    values = parity_registry._block_values(years, cells, 20261013, 0.15, 0.02)
    blocks = np.array_split(values, max(workers), axis=1)

    def compute(thread_count: int) -> Callable[[], Any]:
        def run() -> Any:
            # every thread count computes the same blocks, so a row's seconds cover the same work
            with ThreadPoolExecutor(max_workers=thread_count, initializer=np.seterr, initargs=("ignore",)) as pool:
                return list(pool.map(lambda block: parity_registry._spi(block, indices.Distribution.gamma)(), blocks))

        return run

    return [ThreadTiming(count, *measure_backends(compute(count), compute(count), repeats)) for count in workers]


# builds one call over a selection of cells: ``slice(None)`` for all of them, or an index array for a block
Build = Callable[[np.ndarray | slice], Callable[[], Any]]


def pooled(build: Build, cells: int, blocks: int, workers: int) -> Callable[[], Any]:
    """A call that computes ``blocks`` disjoint cell blocks across a pool of ``workers`` threads."""
    runs = [build(index) for index in np.array_split(np.arange(cells), blocks)]

    def task(run: Callable[[], Any]) -> Any:
        # NumPy's error state is a context variable, and a pool thread starts from the default policy,
        # not the caller's: without this, every task takes the Python path
        with native_policy():
            return run()

    def run() -> Any:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(task, runs))

    return run


def compare(
    entry: Entry, build: Build, cells: int, repeats: int, workers: int | None = None, blocks: int = 1
) -> Comparison:
    """Time ``entry``'s full-grid call through each backend, timing the extension calls the Rust run makes.

    Args:
        entry: the registry entry whose dispatch module and kernels the run exercises
        build: builds the entry's call over a selection of the input's cells
        cells: number of cells in the input
        repeats: number of timed repetitions per backend
        workers: thread count of the pool to spread ``blocks`` across; one eager call when omitted
        blocks: number of cell blocks the input is split into for the pool

    Returns:
        the samples, whether the Rust run called every kernel the entry registers, and its extension time
    """
    run = build(slice(None)) if workers is None else pooled(build, cells, blocks, workers)
    timer = KernelTimer(require_native())
    patch = pytest.MonkeyPatch()
    patch.setattr(entry.dispatch, "_native", timer)
    try:
        events = list(backend_events(run, run, repeats, warm_up=False))
    finally:
        patch.undo()
    rust_samples = tuple(elapsed for backend, elapsed in events if backend == "rust")
    python_samples = tuple(elapsed for backend, elapsed in events if backend == "python")
    return Comparison(entry.name, workers, rust_samples, python_samples, entry.kernels <= timer.calls, timer.seconds)


def _worker_dispatch_probe() -> dict[str, Any]:
    """What a spawned Dask worker reports about its own dispatch guard inputs."""
    return {
        "extension_imported": compute_module()._native is not None,
        "float_error_policy": sorted({str(policy) for policy in np.geterr().values()}),
    }


def require_native() -> ModuleType:
    """Return the extension, or exit with the build command when it is not installed."""
    if importlib.util.find_spec("climate_indices._native") is None:
        raise SystemExit("climate_indices._native is not built; build it with `uv run maturin develop --release`")
    return importlib.import_module("climate_indices._native")


def compute_module() -> Any:
    """The dispatch module behind the standardized indices, imported in whichever process calls this."""
    from climate_indices import compute

    return compute


def dask_timings(side: int = DASK_SIDE, repeats: int = 3) -> tuple[list[SchedulerTiming], dict[str, Any]]:
    """Time gridded SPI with explicit thread-worker policies and default process workers.

    NumPy error state is thread-local: thread pools initialize the requested policy.
    Process rows change only the caller's policy, with no worker routing observation.
    The separate probe describes only its own default spawned worker.

    Args:
        side: latitude and longitude cell count of the synthetic grid
        repeats: number of timed repetitions per scheduler and error policy

    Returns:
        the timed combinations, and the spawned worker probe's report
    """
    import pandas as pd
    import xarray as xr
    from dask.base import compute
    from dask.delayed import delayed

    from climate_indices import spi

    time_index = pd.date_range("1981-01-01", periods=DASK_YEARS * 12, freq="MS")
    shape = (time_index.size, side, side)
    grid = xr.DataArray(
        np.random.default_rng(42).gamma(shape=2.0, scale=15.0, size=shape),
        coords={"time": time_index, "lat": np.linspace(25.0, 49.0, side), "lon": np.linspace(-125.0, -101.0, side)},
        dims=["time", "lat", "lon"],
        attrs={"units": "mm"},
    ).chunk({"time": -1, "lat": max(1, side // 2), "lon": max(1, side // 2)})
    lazy = spi(
        values=grid,
        scale=DASK_SCALE,
        distribution=indices.Distribution.gamma,
        calibration_year_initial=1981,
        calibration_year_final=2010,
    )

    timings = []
    for scheduler in ("threads", "processes"):
        for policy in ("default", "ignore"):
            recorder = conftest.NativeRecorder(require_native()) if scheduler == "threads" else None
            pool_context = (
                ThreadPoolExecutor(
                    initializer=lambda policy=policy: np.seterr(
                        all="ignore" if policy == "ignore" else "warn", under="ignore"
                    )
                )
                if scheduler == "threads"
                else nullcontext(None)
            )
            patch = pytest.MonkeyPatch()
            if recorder is not None:
                patch.setattr(compute_module(), "_native", recorder)
            try:
                with pool_context as pool, warnings.catch_warnings():
                    warnings.resetwarnings()
                    warnings.simplefilter("ignore")
                    seconds = measure(
                        lambda scheduler=scheduler: compute(lazy, scheduler=scheduler, pool=pool),
                        repeats,
                        error_policy=policy,
                    )
            finally:
                patch.undo()
            reached = bool(recorder.calls) if recorder is not None else None
            timings.append(SchedulerTiming(scheduler, policy, seconds, reached))
    probe = compute(delayed(_worker_dispatch_probe)(), scheduler="processes")[0]
    return timings, probe


def render_entries(timings: Sequence[Timings]) -> str:
    """Render the steady-state Rust/Python table as Markdown."""
    lines = ["| entry | family | Rust | Python | Python/Rust |", "|---|---|---|---|---|"]
    for timing in timings:
        lines.append(
            f"| `{timing.name}` | {timing.family} | {timing.rust_seconds * 1e3:.3f} ms | "
            f"{timing.python_seconds * 1e3:.3f} ms | {timing.ratio:.2f} |"
        )
    faster = sum(1 for timing in timings if timing.rust_faster)
    return "\n".join([*lines, "", f"Rust faster in {faster} of {len(timings)} entries."])


def render_cold(import_seconds: float, cold: dict[str, float]) -> str:
    """Render the import and first-call comparison as Markdown."""
    lines = [
        "| measurement | seconds |",
        "|---|---|",
        f"| `import climate_indices`, fresh interpreter (interpreter start-up removed) | {import_seconds:.3f} |",
    ]
    lines += [f"| first call `{name}`, fresh interpreter | {seconds:.6f} |" for name, seconds in cold.items()]
    return "\n".join(lines)


def render_sweep(name: str, sweep: Sequence[tuple[int, Timings]]) -> str:
    """Render one entry's cell-count sweep and its fixed/per-cell fit as Markdown."""
    fixed_rust, per_cell_rust = fit_fixed_and_per_cell(sweep, "rust")
    fixed_python, per_cell_python = fit_fixed_and_per_cell(sweep, "python")
    lines = [
        f"cell sweep, `{name}`",
        "",
        "| cells | Rust | Python | Python/Rust |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {count} | {timing.rust_seconds * 1e6:.1f} µs | {timing.python_seconds * 1e6:.1f} µs | {timing.ratio:.2f} |"
        for count, timing in sweep
    ]
    lines += [
        "",
        f"fixed per call: `{fixed_rust * 1e3:.3f}` ms Rust, `{fixed_python * 1e3:.3f}` ms Python; "
        f"per cell: `{per_cell_rust * 1e6:.3f}` µs Rust, `{per_cell_python * 1e6:.3f}` µs Python",
    ]
    return "\n".join(lines)


def render_threads(
    timings: Sequence[ThreadTiming], cells: int = THREAD_CELLS, blocks: int = max(THREAD_WORKERS)
) -> str:
    """Render the thread-scaling table as Markdown."""
    lines = [
        f"thread scaling, {blocks} blocks of {cells // blocks} cells, every row the same work",
        "",
        "| threads | Rust | Python | Python/Rust |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {timing.workers} | {timing.rust_seconds:.3f} s | {timing.python_seconds:.3f} s | "
        f"{timing.python_seconds / timing.rust_seconds:.2f} |"
        for timing in timings
    ]
    return "\n".join(lines)


def _format_samples(seconds: Sequence[float]) -> str:
    """Render every sample of one configuration, so the spread stays visible."""
    return ", ".join(f"{sample:.2f}" for sample in seconds)


@dataclass(frozen=True)
class GridInputs:
    """A prepared grid's land cells, as time-major blocks, and each cell's latitude."""

    precipitation: np.ndarray  # (time, cells, 1), mm
    temperature: np.ndarray  # (time, cells, 1), degrees Celsius
    latitude: np.ndarray  # (cells, 1), degrees north
    data_start_year: int
    description: str


def load_grid(
    precipitation_path: Path, precipitation_var: str, temperature_path: Path, temperature_var: str
) -> GridInputs:
    """Read a prepared precipitation grid's land cells and the matching temperature cells.

    Precipitation is prepared the way ``parallel_scaling.load_netcdf_grid`` prepares it for
    every other nClimGrid benchmark (land mask from the first time step, zeros as 0.01 mm);
    temperature is label-selected on the loader's rolled coordinates, so the two grids line
    up cell for cell. Only land cells are kept, so every cell is a series both backends compute.

    Args:
        precipitation_path: grid with ``time``, ``lat`` and ``lon`` dims, whole years of consecutive monthly mm
        precipitation_var: precipitation variable of ``precipitation_path``
        temperature_path: grid of monthly mean temperature in degrees Celsius on the same coordinates
        temperature_var: temperature variable of ``temperature_path``

    Returns:
        the land cells' blocks and latitudes, and a line describing the grid
    """
    grid = parallel_scaling.load_netcdf_grid(str(precipitation_path), precipitation_var)
    precip, land = grid.precip, grid.valid_cells
    assert land is not None
    months, lat, lon = precip.shape
    years = precip["time"].dt.year.values
    month_numbers = precip["time"].dt.month.values
    if (
        months % 12
        or month_numbers[0] != 1
        or np.any(np.diff(years * 12 + month_numbers) != 1)
        or years[0] > GRID_CALIBRATION[0]
        or years[-1] < GRID_CALIBRATION[1]
    ):
        raise SystemExit(
            f"{precipitation_path}: need whole years of consecutive monthly values starting in January and covering "
            f"{GRID_CALIBRATION[0]}-"
            f"{GRID_CALIBRATION[1]}, the calibration period; got {months} months from {years[0]} to {years[-1]}"
        )
    with xr.open_dataset(temperature_path) as dataset:
        # label selection lines temperature up with the loader's rolled precipitation, cell for cell
        temperature = (
            dataset[temperature_var]
            .transpose("time", "lat", "lon")
            .sel(time=precip["time"].values, lat=precip["lat"].values, lon=precip["lon"].values)
            .values.astype(np.float64)
        )
    precipitation = np.ascontiguousarray(precip.values[:, land])[:, :, np.newaxis]
    description = (
        f"grid: {precipitation_path.name} ({precipitation_var}), {temperature_path.name} ({temperature_var}); "
        f"time={months} ({years[0]}-{years[-1]}), lat={lat}, lon={lon}; {precipitation.shape[1]:,} land cells "
        f"of {lat * lon:,}; {np.isfinite(precipitation).sum():,} finite precipitation values"
    )
    return GridInputs(
        precipitation=precipitation,
        temperature=np.ascontiguousarray(temperature[:, land])[:, :, np.newaxis],
        latitude=np.broadcast_to(precip["lat"].values[:, np.newaxis], land.shape)[land].astype(np.float64)[:, None],
        data_start_year=int(years[0]),
        description=description,
    )


def grid_builds(inputs: GridInputs) -> dict[str, Build]:
    """The full-grid call of each :data:`GRID_ENTRIES` entry, from the grid's own inputs.

    PET is Thornthwaite's from the grid's temperature, computed while the call is
    built, so a timed call is the index alone. PDSI reads inches.
    """
    start, (first, last) = inputs.data_start_year, GRID_CALIBRATION
    monthly = compute.Periodicity.monthly

    def cells(block: np.ndarray, index: np.ndarray | slice) -> np.ndarray:
        return np.ascontiguousarray(block[:, index])

    def pet(index: np.ndarray | slice) -> np.ndarray:
        with native_policy():
            return eto.eto_thornthwaite(
                cells(inputs.temperature, index), inputs.latitude[index], start, spatial_time_major=True
            )

    def spi(distribution: indices.Distribution) -> Build:
        def build(index: np.ndarray | slice) -> Callable[[], Any]:
            values = cells(inputs.precipitation, index)
            return lambda: indices.spi(values, 6, distribution, start, first, last, monthly, spatial_time_major=True)

        return build

    def spei(index: np.ndarray | slice) -> Callable[[], Any]:
        values, evapotranspiration = cells(inputs.precipitation, index), pet(index)
        return lambda: indices.spei(
            values,
            evapotranspiration,
            6,
            indices.Distribution.gamma,
            monthly,
            start,
            first,
            last,
            spatial_time_major=True,
        )

    def eddi(index: np.ndarray | slice) -> Callable[[], Any]:
        evapotranspiration = pet(index)
        return lambda: indices.eddi(evapotranspiration, 6, start, first, last, monthly, spatial_time_major=True)

    def thornthwaite(index: np.ndarray | slice) -> Callable[[], Any]:
        temperature, latitude = cells(inputs.temperature, index), inputs.latitude[index]
        return lambda: eto.eto_thornthwaite(temperature, latitude, start, spatial_time_major=True)

    def pdsi(index: np.ndarray | slice) -> Callable[[], Any]:
        values, evapotranspiration = cells(inputs.precipitation, index) / MM_PER_INCH, pet(index) / MM_PER_INCH
        return lambda: palmer.pdsi(
            values, evapotranspiration, GRID_AWC_INCHES, start, first, last, spatial_time_major=True
        )

    return {
        "spi_gamma": spi(indices.Distribution.gamma),
        "spi_pearson": spi(indices.Distribution.pearson),
        "spei_gamma": spei,
        "eddi": eddi,
        "thornthwaite": thornthwaite,
        "palmer_pdsi": pdsi,
    }


def grid_comparisons(
    entries: Sequence[Entry], inputs: GridInputs, workers: Sequence[int], repeats: int
) -> list[Comparison]:
    """Time each entry over every land cell: one eager call, then a pool at each thread count.

    A full-grid call takes seconds to minutes, so there is no warm-up call: a first
    call's one-off costs are milliseconds (the cold measurement), not a visible share
    of a sample.
    """
    builds, cells = grid_builds(inputs), inputs.precipitation.shape[1]
    comparisons = []
    for entry in entries:
        build = builds[entry.name]
        comparisons.append(compare(entry, build, cells, repeats))
        comparisons += [compare(entry, build, cells, repeats, count, max(workers)) for count in workers]
    return comparisons


def render_grid(comparisons: Sequence[Comparison], workers: Sequence[int]) -> str:
    """Render the full-grid eager and thread-pool tables, with every sample, as Markdown."""
    eager = [comparison for comparison in comparisons if comparison.workers is None]
    lines = [
        "eager, one call on the whole block",
        "",
        "| entry | Rust | Python | Python/Rust | inside extension calls | Rust kernels reached |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| `{timing.name}` | {timing.rust_seconds:.2f} s | {timing.python_seconds:.2f} s | "
        f"{timing.python_seconds / timing.rust_seconds:.2f} | {timing.extension_share:.0%} | "
        f"{timing.rust_kernels_reached} |"
        for timing in eager
    ]
    lines += [
        "",
        f"thread pool, {max(workers)} blocks of cells, every row the same work",
        "",
        "| entry | threads | Rust | Python | Python/Rust | Rust kernels reached |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| `{timing.name}` | {timing.workers} | {timing.rust_seconds:.2f} s | {timing.python_seconds:.2f} s | "
        f"{timing.python_seconds / timing.rust_seconds:.2f} | {timing.rust_kernels_reached} |"
        for timing in comparisons
        if timing.workers is not None
    ]
    lines += ["", "samples, seconds in call order", ""]
    lines += [
        f"- `{timing.name}`, {'eager' if timing.workers is None else f'threads {timing.workers}'}: "
        f"Rust {_format_samples(timing.rust_samples)}; Python {_format_samples(timing.python_samples)}"
        for timing in comparisons
    ]
    return "\n".join(lines)


def run_grid(args: argparse.Namespace) -> str:
    """Run the full-grid measurement and return its report."""
    entries = [parity_registry.ENTRIES_BY_NAME[name] for name in args.grid_entries.split(",")]
    unsupported = sorted({entry.name for entry in entries} - set(GRID_ENTRIES))
    if unsupported:
        raise SystemExit(f"no full-grid call for {unsupported}; choose from {list(GRID_ENTRIES)}")
    workers = tuple(int(count) for count in args.threads.split(","))
    inputs = load_grid(args.netcdf, args.var, args.tavg, args.tavg_var)
    with warnings.catch_warnings():
        # the masked cells and the calibration fits warn on every call; the report is the output
        warnings.simplefilter("ignore")
        comparisons = grid_comparisons(entries, inputs, workers, args.repeat)
    return "\n".join(
        [
            parallel_scaling._environment(),
            f"revision: {parallel_scaling._revision()}",
            inputs.description,
            f"precipitation sha256: {parallel_scaling._hash_file(str(args.netcdf))}",
            f"temperature sha256: {parallel_scaling._hash_file(str(args.tavg))}",
            f"calibration {GRID_CALIBRATION[0]}-{GRID_CALIBRATION[1]}; scale 6; PET Thornthwaite from the "
            f"temperature; PDSI available water capacity {GRID_AWC_INCHES} in at every cell",
            f"repetitions: {args.repeat} per backend and configuration, no warm-up call; best sample reported",
            "",
            render_grid(comparisons, workers),
        ]
    )


def render_dask(timings: Sequence[SchedulerTiming], probe: dict[str, Any], side: int = DASK_SIDE) -> str:
    """Render the gridded Dask table and the worker probe as Markdown."""
    lines = [
        f"gridded Dask, {DASK_YEARS} years x {side} x {side} cells, chunked spatially",
        "",
        "| scheduler | NumPy error policy (threads: worker; processes: caller) | seconds | "
        "Rust kernels reached (recorder, this process) |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {timing.scheduler} | {timing.error_policy} | {timing.seconds:.3f} | "
        f"{timing.rust_kernels_reached if timing.rust_kernels_reached is not None else 'n/a'} |"
        for timing in timings
    ]
    lines += ["", f"spawned worker probe: `{json.dumps(probe, sort_keys=True)}`"]
    return "\n".join(lines)


def run_small(args: argparse.Namespace) -> str:
    """Run the routine measurements and return the report."""
    cold = {name: measure_cold(name) for name in filter(None, args.cold_entries.split(","))}
    report = [
        f"python {platform.python_version()}; {platform.platform()}; {platform.processor() or platform.machine()}",
        f"repetitions: {args.repeat}; best of the repetitions, after a warm-up call; "
        "Rust and Python alternate which runs first each repetition",
        "",
        render_entries(time_entries(repeats=args.repeat)),
        "",
        render_cold(measure_import_seconds(), cold),
    ]

    if not args.skip_sweep:
        report += [""]
        for name in SWEEP_ENTRIES:
            report += [render_sweep(name, sweep_entry(parity_registry.ENTRIES_BY_NAME[name], repeats=args.repeat)), ""]

    if not args.skip_threads:
        report += [render_threads(thread_scaling(repeats=args.repeat)), ""]

    if not args.skip_dask:
        timings, probe = dask_timings(repeats=args.repeat)
        report += [render_dask(timings, probe), ""]

    return "\n".join(report).rstrip()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested measurements and print the report as Markdown."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=3, help="timed repetitions per measurement (default: 3)")
    parser.add_argument("--output", type=Path, help="write the report here")
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write the report to {DEFAULT_OUTPUT}, or with --netcdf to {DEFAULT_GRID_OUTPUT}",
    )
    parser.add_argument("--netcdf", type=Path, help="run the full-grid measurement on this prepared grid instead")
    parser.add_argument("--var", default="prcp", help="precipitation variable of --netcdf (default: prcp)")
    parser.add_argument("--tavg", type=Path, help="monthly mean temperature on the --netcdf grid's coordinates")
    parser.add_argument("--tavg-var", default="tavg", help="temperature variable of --tavg (default: tavg)")
    parser.add_argument("--grid-entries", default=",".join(GRID_ENTRIES), help="entries the full-grid run times")
    parser.add_argument(
        "--threads", default=",".join(map(str, GRID_WORKERS)), help="thread counts of the full-grid thread pool"
    )
    parser.add_argument("--cold-entries", default=",".join(COLD_ENTRIES), help="entries whose first call is timed")
    parser.add_argument("--skip-sweep", action="store_true", help="skip the cell-count sweep")
    parser.add_argument("--skip-threads", action="store_true", help="skip the thread-scaling measurement")
    parser.add_argument("--skip-dask", action="store_true", help="skip the gridded Dask measurement")
    parser.add_argument("--cold-entry", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.netcdf and not args.tavg:
        parser.error("--netcdf needs --tavg: the PET-based entries compute from the grid's own temperature")

    if args.cold_entry:
        print(json.dumps({"seconds": cold_call_seconds(args.cold_entry)}))
        return 0

    require_native()  # every measurement needs the built extension
    text = run_grid(args) if args.netcdf else run_small(args)
    print(text)
    if args.write or args.output:
        destination = args.output or (DEFAULT_GRID_OUTPUT if args.netcdf else DEFAULT_OUTPUT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        print(f"written to {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

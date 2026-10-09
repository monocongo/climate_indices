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

Run from the repository root, with the extension built (``uv run maturin develop
--release``)::

    uv run benchmarks/rust_vs_python.py
    uv run benchmarks/rust_vs_python.py --repeat 7 --write
    uv run benchmarks/rust_vs_python.py --skip-sweep --skip-threads --skip-dask
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import subprocess
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402  (pytest.MonkeyPatch is the tests' backend switch, used directly)

from climate_indices import indices  # noqa: E402
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

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "rust_vs_python.txt"


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
    rust_run = entry.run(drawn)
    python_run = entry.run(drawn)
    rust_seconds = measure(rust_run, repeats)
    with python_backend():
        python_seconds = measure(python_run, repeats)
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
        completed = subprocess.run(
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
        subprocess.run([sys.executable, "-c", statement], check=True, capture_output=True)
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

    timings = []
    for count in workers:
        rust_seconds = measure(compute(count), repeats)
        with python_backend():
            python_seconds = measure(compute(count), repeats)
        timings.append(ThreadTiming(count, rust_seconds, python_seconds))
    return timings


def _worker_dispatch_probe() -> dict[str, Any]:
    """What a spawned Dask worker reports about its own dispatch guard inputs."""
    return {
        "extension_imported": compute_module()._native is not None,
        "float_error_policy": sorted(set(np.geterr().values())),
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
    import dask
    import pandas as pd
    import xarray as xr

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
                        lambda scheduler=scheduler: dask.compute(lazy, scheduler=scheduler, pool=pool),
                        repeats,
                        error_policy=policy,
                    )
            finally:
                patch.undo()
            reached = bool(recorder.calls) if recorder is not None else None
            timings.append(SchedulerTiming(scheduler, policy, seconds, reached))
    probe = dask.compute(dask.delayed(_worker_dispatch_probe)(), scheduler="processes")[0]
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested measurements and print the report as Markdown."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=3, help="timed repetitions per measurement (default: 3)")
    parser.add_argument("--output", type=Path, help="write the report here")
    parser.add_argument("--write", action="store_true", help=f"write the report to {DEFAULT_OUTPUT}")
    parser.add_argument("--cold-entries", default=",".join(COLD_ENTRIES), help="entries whose first call is timed")
    parser.add_argument("--skip-sweep", action="store_true", help="skip the cell-count sweep")
    parser.add_argument("--skip-threads", action="store_true", help="skip the thread-scaling measurement")
    parser.add_argument("--skip-dask", action="store_true", help="skip the gridded Dask measurement")
    parser.add_argument("--cold-entry", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeat <= 0:
        parser.error("--repeat must be positive")

    if args.cold_entry:
        print(json.dumps({"seconds": cold_call_seconds(args.cold_entry)}))
        return 0

    require_native()  # every measurement needs the built extension
    cold = {name: measure_cold(name) for name in filter(None, args.cold_entries.split(","))}
    report = [
        f"python {platform.python_version()}; {platform.platform()}; {platform.processor() or platform.machine()}",
        f"repetitions: {args.repeat}; best of the repetitions, after a warm-up call",
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

    text = "\n".join(report).rstrip()
    print(text)
    if args.write or args.output:
        destination = args.output or DEFAULT_OUTPUT
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        print(f"written to {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

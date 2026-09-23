"""Benchmark gridded index scaling from 1 to N Dask workers.

Runs the #893 reference grid (38x87 cells, 40 years of monthly precipitation)
through the public xarray API on a Dask-backed input, once per requested worker
count with the ``processes`` scheduler, and prints the wall clock, the speedup
against the first worker count, and the parallel efficiency. Before the worker
counts it times the in-memory, single-process call once per index on the same
grid, at the default INFO log level and again with logging quiet, which is the
serial reference the parallel numbers and the epic's speedup criterion are
measured against.

Spatial chunking follows the worker count so every worker gets blocks of
similar size, while ``time`` stays a single chunk (ADR-0003). The grid size is
fixed across worker counts, so the numbers are strong scaling at the reference
size.

PET is synthetic: a fixed fraction of the precipitation for SPEI, and a
latitude-gradient seasonal cycle for the Thornthwaite index. Goodness-of-fit
warnings are filtered for every run, and the Dask sweep additionally runs with
logging quiet, because per-cell log volume costs more than the computation at
this grid size.

``--netcdf`` switches to the real-grid mode: a NetCDF precipitation file with
``time``, ``lat`` and ``lon`` dimensions (the Morocco CHIRPS v3 case from
Ouranosinc/xclim#2091) replaces the synthetic grid, and only SPI runs, because
SPEI, PET and EDDI need inputs the precipitation file does not carry. The real-grid
mode records the file SHA-256 and the checkout revision, treats the finite cells of
the first time step as the land mask, replaces zeros with ``0.01`` mm for the gamma
fit, times every sample of every configuration, and separates the NetCDF read and
write from the compute.

``--scheduler ADDRESS`` (with ``--netcdf``) replaces the local ``processes``
scheduler with a running ``dask.distributed`` cluster: ``--cores`` then counts
worker *processes* on that cluster rather than local cores, packed one node at
a time (#1127). Inputs are persisted onto the selected workers before timing
starts, reported separately as the distribute time, and every distributed
result must match the eager serial run's output bit for bit or the benchmark
raises rather than reporting a number.

Run from the repository root::

    uv run benchmarks/parallel_scaling.py
    uv run benchmarks/parallel_scaling.py --cores 1,2,3,4 --indices spi,spei --repeat 5
    uv run benchmarks/parallel_scaling.py --indices spi,spei,pet,eddi --serial-only
    uv run benchmarks/parallel_scaling.py --netcdf mar_cli_chirps3.nc --cores 1,2,4,8 --scale 6
    uv run benchmarks/parallel_scaling.py --netcdf mar_cli_chirps3.nc --scale 6 --distribution pearson
    uv run benchmarks/parallel_scaling.py --netcdf nclimgrid_prcp.nc --var-name prcp --scale 6 \\
        --scheduler tcp://10.0.0.1:8786 --cores 16,32,64
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import platform
import subprocess
import time
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import dask
import numpy as np
import scipy
import xarray as xr
from profile_gridded_spi import (
    CALIBRATION_PERIOD,
    DATA_START_YEAR,
    REFERENCE_LAT,
    REFERENCE_LON,
    REFERENCE_YEARS,
    SCALE,
    SEED,
    build_grid,
    run_spi,
)

from climate_indices import eddi, pet_thornthwaite, spei, spi
from climate_indices.compute import Periodicity
from climate_indices.exceptions import GoodnessOfFitWarning
from climate_indices.indices import Distribution
from climate_indices.logging_config import ENV_LOG_LEVEL

PET_FRACTION = 0.6
TEMPERATURE_SEED = SEED + 1


class _Grid(NamedTuple):
    """Reference-grid inputs shared by every index runner.

    ``pet`` and ``temperature`` are None in the real-grid mode, which only SPI
    uses; ``valid_cells`` is the ``(lat, lon)`` land mask when the grid has one.
    """

    precip: xr.DataArray
    pet: xr.DataArray | None = None
    temperature: xr.DataArray | None = None
    valid_cells: np.ndarray | None = None


_Runner = Callable[[_Grid], xr.DataArray]


def build_pet(precip: xr.DataArray) -> xr.DataArray:
    """Build synthetic monthly PET on the precipitation grid.

    A fixed fraction of the precipitation keeps SPEI's (precipitation - PET)
    series positive, so the gamma fit sees a valid sample.

    Args:
        precip: precipitation grid with ``time``, ``lat`` and ``lon`` dimensions

    Returns:
        DataArray with the same coordinates, units "mm"
    """
    pet = (precip * PET_FRACTION).rename("pet")
    pet.attrs["units"] = "mm"
    return pet


def build_temperature(precip: xr.DataArray) -> xr.DataArray:
    """Build synthetic monthly mean temperature on the precipitation grid.

    A latitude gradient plus an annual cycle keeps the Thornthwaite heat index
    varying across cells and seasons, so the day-length term is exercised rather
    than constant.

    Args:
        precip: precipitation grid with ``time``, ``lat`` and ``lon`` dimensions

    Returns:
        DataArray with the same coordinates, units "degrees_celsius"
    """
    rng = np.random.default_rng(TEMPERATURE_SEED)
    months = precip["time"].dt.month.values
    latitude = precip["lat"].values
    seasonal = np.cos((months - 7) / 12 * 2 * np.pi)[:, None, None]
    values = 18.0 - 0.25 * latitude[None, :, None] + 10.0 * seasonal
    values = values + rng.normal(0.0, 1.0, size=precip.shape)
    temperature = xr.DataArray(
        values,
        coords=precip.coords,
        dims=precip.dims,
        attrs={"units": "degrees_celsius"},
    )
    return temperature.rename("temperature")


def build_inputs() -> _Grid:
    """Build the reference grid's precipitation, PET and temperature inputs."""
    precip = build_grid(lat=REFERENCE_LAT, lon=REFERENCE_LON, years=REFERENCE_YEARS)
    return _Grid(precip=precip, pet=build_pet(precip), temperature=build_temperature(precip))


def run_spei(grid: _Grid) -> xr.DataArray:
    """Run SPEI with the reference grid's parameters."""
    return spei(
        precips_mm=grid.precip,
        pet_mm=grid.pet,
        scale=SCALE,
        distribution=Distribution.gamma,
        data_start_year=DATA_START_YEAR,
        calibration_year_initial=CALIBRATION_PERIOD[0],
        calibration_year_final=CALIBRATION_PERIOD[1],
        periodicity=Periodicity.monthly,
    )


def run_pet(grid: _Grid) -> xr.DataArray:
    """Run Thornthwaite PET on the reference grid.

    Latitude is a ``(lat,)`` coordinate so each cell gets its own latitude for
    the day-length term; a scalar would apply one latitude to the whole grid.
    """
    return pet_thornthwaite(
        grid.temperature,
        grid.temperature["lat"],
        data_start_year=DATA_START_YEAR,
    )


def run_eddi(grid: _Grid) -> xr.DataArray:
    """Run EDDI on the reference grid's synthetic PET."""
    return eddi(
        grid.pet,
        scale=SCALE,
        data_start_year=DATA_START_YEAR,
        calibration_year_initial=CALIBRATION_PERIOD[0],
        calibration_year_final=CALIBRATION_PERIOD[1],
        periodicity=Periodicity.monthly,
    )


def _run_spi(grid: _Grid) -> xr.DataArray:
    """Run SPI with the reference grid's parameters (PET is unused)."""
    return run_spi(grid.precip)


class _Index(NamedTuple):
    """A benchmarkable index, and how many leading time steps it pads with NaN."""

    run: _Runner
    leading_pad: int


_RUNNERS: dict[str, _Index] = {
    "spi": _Index(_run_spi, SCALE - 1),
    "spei": _Index(run_spei, SCALE - 1),
    "pet": _Index(run_pet, 0),
    "eddi": _Index(run_eddi, SCALE - 1),
}


def _split_cells(cells: int, parts: int) -> tuple[int, ...]:
    """Split ``cells`` into ``parts`` contiguous chunks of near-equal size."""
    base, extra = divmod(cells, parts)
    return tuple(base + 1 if index < extra else base for index in range(parts))


def _chunk_for_workers(array: xr.DataArray, workers: int) -> xr.DataArray:
    """Chunk the spatial dimensions so every worker gets at least one block.

    The time dimension stays a single chunk, as ADR-0003 requires. The chunk
    sizes are explicit: rounding a uniform integer size can undershoot the
    requested part count (38 cells over 12 parts is 10 chunks at size 4), which
    would leave workers idle while their count still scales the reported
    efficiency.
    """
    lat_cells, lon_cells = array.sizes["lat"], array.sizes["lon"]
    lat_parts = min(lat_cells, max(1, round(math.sqrt(workers * lat_cells / lon_cells))))
    lon_parts = min(lon_cells, max(1, math.ceil(workers / lat_parts)))
    return array.chunk(
        {
            "time": -1,
            "lat": _split_cells(lat_cells, lat_parts),
            "lon": _split_cells(lon_cells, lon_parts),
        }
    )


def _require_finite_tail(values: np.ndarray, leading_pad: int, valid_cells: np.ndarray | None = None) -> None:
    """Reject output that degenerated past the index's leading NaN padding.

    SPI, SPEI and EDDI pad the first ``scale - 1`` time steps with NaN, so the
    check starts there; PET has no padding and is checked from the first step.
    Anything else non-finite means the fit degenerated and the timing above
    measures nothing useful.

    On a masked grid only the land cells may be finite, and the masked cells must
    stay NaN -- a run that filled the ocean would still time, but would not be
    the workload the fixture describes.

    Args:
        values: index output on the reference grid
        leading_pad: number of leading time steps allowed to be NaN
        valid_cells: ``(lat, lon)`` land mask, or None when the grid is fully populated
    """
    tail = values[leading_pad:]
    if valid_cells is None:
        if not np.isfinite(tail).all():
            raise RuntimeError(f"non-finite output beyond the leading {leading_pad} padded time steps")
        return
    if not np.isfinite(tail[:, valid_cells]).all():
        raise RuntimeError(f"non-finite land-cell output beyond the leading {leading_pad} padded time steps")
    if np.isfinite(tail[:, ~valid_cells]).any():
        raise RuntimeError("finite output on cells the input mask marked as missing")


def _time_serial(grid: _Grid, index: _Index) -> float:
    """Run ``index`` on the in-memory grid once and return the elapsed seconds."""
    start = time.perf_counter()
    values = index.run(grid).values
    elapsed = time.perf_counter() - start
    _require_finite_tail(values, index.leading_pad, grid.valid_cells)
    return elapsed


def _serial_timings(grid: _Grid, index: _Index, repeat: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Time the serial in-memory call at the default INFO level, then quiet.

    Both samples filter goodness-of-fit warnings, as the Dask runs do, so the
    INFO/quiet delta isolates per-cell log rendering. INFO therefore measures the
    library's default log level, not the full cost an unconfigured,
    warning-visible caller pays.

    Args:
        grid: reference-grid inputs
        index: index to benchmark
        repeat: timed runs per level; every observation is returned, not just the fastest

    Returns:
        INFO samples and WARNING samples, both in run order
    """
    logging.getLogger().setLevel(logging.INFO)
    info = tuple(_time_serial(grid, index) for _ in range(repeat))
    logging.getLogger().setLevel(logging.WARNING)
    quiet = tuple(_time_serial(grid, index) for _ in range(repeat))
    return info, quiet


def _measure(index: _Index, grid: _Grid, workers: int, repeat: int) -> tuple[float, ...]:
    """Return every timed run on ``workers`` Dask processors, in run order.

    One extra warm-up run keeps parent-side first-call imports out of the timed
    samples. Every ``compute()`` call creates a fresh process pool, so pool
    start-up stays inside every measurement, as it does for any caller of the
    ``processes`` scheduler. The pool initializer installs the goodness-of-fit
    filter in each worker before any task runs.

    The full spread is returned rather than the minimum alone: the run-to-run
    variance is part of the reported evidence (#1097), and a fastest-only table
    hides it.
    """
    inputs = _Grid(
        precip=_chunk_for_workers(grid.precip, workers),
        pet=_chunk_for_workers(grid.pet, workers) if grid.pet is not None else None,
        temperature=_chunk_for_workers(grid.temperature, workers) if grid.temperature is not None else None,
        valid_cells=grid.valid_cells,
    )
    timings = []
    for _ in range(repeat + 1):
        start = time.perf_counter()
        # chunksize=1: the default batches up to six ready tasks per submission, which
        # runs a whole six-block batch sequentially on one worker
        result = index.run(inputs).compute(
            scheduler="processes",
            num_workers=workers,
            chunksize=1,
            initializer=_quiet_worker,
        )
        timings.append(time.perf_counter() - start)
    _require_finite_tail(result.values, index.leading_pad, grid.valid_cells)
    return tuple(timings[1:])


def _select_workers(workers_info: dict[str, dict[str, Any]], count: int) -> tuple[tuple[str, ...], int]:
    """Pick ``count`` worker addresses from a ``Client.scheduler_info()["workers"]`` mapping.

    Sorted by ``(host, address)`` and taken in order, so selection packs one node
    at a time: with N processes per node, ``--cores N,2N,3N`` means 1, 2 and 3
    nodes. Returns the chosen addresses and the number of distinct hosts among
    them, so the results table can show node count alongside worker count.
    """
    ordered = sorted(workers_info.items(), key=lambda item: (item[1].get("host", item[0]), item[0]))
    if count > len(ordered):
        raise ValueError(f"cluster has {len(ordered)} worker processes, requested {count}")
    selected = ordered[:count]
    addresses = tuple(address for address, _ in selected)
    hosts = len({info.get("host", address) for address, info in selected})
    return addresses, hosts


def _measure_distributed(
    index: _Index, grid: _Grid, client: Any, addresses: tuple[str, ...], repeat: int, serial_digest: str
) -> tuple[float, tuple[float, ...], str]:
    """Persist chunked inputs onto ``addresses``, then time ``repeat`` distributed computes.

    Mirrors ``_measure``: one warm-up run plus ``repeat`` timed runs, every
    sample returned. Persisting the input onto the selected workers is untimed
    and reported separately as the distribute time, since sending a multi-GB
    grid over the network is not part of what the compute-only figure claims to
    measure. Every run, including warm-up, must match the eager serial result
    before its timing can be retained.

    Returns:
        distribute seconds, every timed compute sample, and the SHA-256 of the
        gathered result; each compute is checked against the eager serial digest
        before any speedup can be printed.
    """
    import distributed

    workers = len(addresses)
    inputs = _Grid(
        precip=_chunk_for_workers(grid.precip, workers),
        pet=_chunk_for_workers(grid.pet, workers) if grid.pet is not None else None,
        temperature=_chunk_for_workers(grid.temperature, workers) if grid.temperature is not None else None,
        valid_cells=grid.valid_cells,
    )
    distribute_start = time.perf_counter()
    persisted = _Grid(
        precip=client.persist(inputs.precip, workers=addresses),
        pet=client.persist(inputs.pet, workers=addresses) if inputs.pet is not None else None,
        temperature=client.persist(inputs.temperature, workers=addresses) if inputs.temperature is not None else None,
        valid_cells=inputs.valid_cells,
    )
    futures = []
    for value in (persisted.precip, persisted.pet, persisted.temperature):
        if value is not None:
            futures.extend(distributed.futures_of(value))
    distributed.wait(futures)
    distribute_seconds = time.perf_counter() - distribute_start

    timings = []
    for _ in range(repeat + 1):
        start = time.perf_counter()
        result = index.run(persisted).compute(workers=list(addresses), allow_other_workers=False)
        elapsed = time.perf_counter() - start
        _require_finite_tail(result.values, index.leading_pad, grid.valid_cells)
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(result.values))).hexdigest()
        if digest != serial_digest:
            raise RuntimeError(f"distributed result at {workers} workers does not match the eager serial run")
        timings.append(elapsed)
    return distribute_seconds, tuple(timings[1:]), digest


def _spatial_blocks(array: xr.DataArray, workers: int) -> int:
    """Number of spatial blocks ``workers`` produces for ``array``."""
    chunked = _chunk_for_workers(array, workers)
    return math.prod(len(axis_chunks) for axis_chunks in chunked.chunks[1:])


def _default_workers(cells: int = REFERENCE_LAT * REFERENCE_LON, cap: int | None = None) -> tuple[int, ...]:
    """Powers of two from 1 up to ``cap`` (default: the CPU count), never above the grid's cells."""
    limit = min(cap if cap is not None else (os.cpu_count() or 1), cells)
    counts = [1]
    while counts[-1] * 2 <= limit:
        counts.append(counts[-1] * 2)
    return tuple(counts)


def _worker_counts(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive worker counts."""
    counts = tuple(int(count) for count in value.split(","))
    if any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("worker counts must be at least 1")
    return counts


def _validate_worker_counts(counts: tuple[int, ...], cells: int) -> None:
    """Reject worker counts above the grid's cells, which cannot produce a block each."""
    if any(count > cells for count in counts):
        raise argparse.ArgumentTypeError(f"worker counts must not exceed the {cells} spatial cells")


def _require_worker_counts(counts: tuple[int, ...], cells: int) -> None:
    """Exit with the message instead of a traceback when ``counts`` cannot fit ``cells``."""
    try:
        _validate_worker_counts(counts, cells)
    except argparse.ArgumentTypeError as error:
        raise SystemExit(str(error)) from error


def _format_samples(samples: tuple[float, ...]) -> str:
    """Render every observation of a configuration, so the spread stays in the artifact."""
    return ", ".join(f"{seconds:.3f}" for seconds in samples)


def _hash_file(path: str) -> str:
    """SHA-256 of ``path``, read in chunks so a large fixture never sits in memory twice."""
    digest = hashlib.sha256()
    # the path is this benchmark's own --netcdf argument: a local, read-only input,
    # not an untrusted path boundary
    with open(path, "rb") as stream:  # NOSONAR
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_model() -> str:
    """CPU model name; ``platform.processor()`` is empty on some arm64 hosts."""
    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            )
            return completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def _revision() -> str:
    """The checkout's commit SHA, or "unknown" outside a git checkout."""
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _environment() -> str:
    """Environment header retained with every results artifact."""
    return (
        f"environment: python {platform.python_version()}; {platform.platform()}; "
        f"numpy {np.__version__}; scipy {scipy.__version__}; xarray {xr.__version__}; dask {dask.__version__}; "
        f"cpu {_cpu_model()}; {os.cpu_count()} CPUs"
    )


def load_netcdf_grid(path: str, var_name: str) -> _Grid:
    """Load a real precipitation grid for the SPI workload.

    The finite cells of the first time step define the land mask: every time step
    outside it is forced to NaN, so the fit never sees an ocean cell. Zeros become
    0.01 mm, because a gamma has no support at zero. The spatial axes are then
    cyclically rolled so the first cell is a land cell: the adapter's calibration
    preflight samples the first spatial point only, and an all-NaN first cell is a
    hard error there. Rolling values and coordinates together keeps every cell
    paired with its own labels, so the workload and the output are unchanged.

    Args:
        path: NetCDF file whose precipitation variable has ``time``, ``lat`` and ``lon`` dims
        var_name: variable holding monthly precipitation in mm

    Returns:
        Grid with the prepared precipitation field; PET and temperature stay None
    """
    with xr.open_dataset(path) as dataset:
        precip = dataset[var_name].transpose("time", "lat", "lon")
        values = precip.values.astype(np.float64)
        time = precip["time"].values
        latitude = precip["lat"].values
        longitude = precip["lon"].values

    valid_cells = np.isfinite(values[0])
    if not valid_cells.any():
        raise ValueError(f"{path}: {var_name} has no finite cell in the first time step")
    values[:, ~valid_cells] = np.nan
    values[values == 0] = 0.01

    first_lat, first_lon = np.argwhere(valid_cells)[0]
    roll_lat, roll_lon = -int(first_lat), -int(first_lon)
    values = np.roll(values, (roll_lat, roll_lon), axis=(1, 2))
    valid_cells = np.roll(valid_cells, (roll_lat, roll_lon), axis=(0, 1))

    grid = xr.DataArray(
        values,
        coords={
            "time": time,
            "lat": np.roll(latitude, roll_lat),
            "lon": np.roll(longitude, roll_lon),
        },
        dims=["time", "lat", "lon"],
        attrs={
            "units": "mm",
            "roll_lat": roll_lat,
            "roll_lon": roll_lon,
            "land_cells": int(valid_cells.sum()),
        },
    )
    return _Grid(precip=grid, valid_cells=valid_cells)


def _require_cluster_capacity(counts: tuple[int, ...], available: int) -> None:
    """Exit with a message if any worker count exceeds the connected cluster's worker processes."""
    exceeding = [count for count in counts if count > available]
    if exceeding:
        raise SystemExit(f"worker counts {exceeding} exceed the {available} connected worker processes")


def _connect_cluster(address: str, revision: str) -> Any:
    """Connect to a running ``dask.distributed`` scheduler and verify every worker matches this run.

    Refuses to proceed if any worker is not logging at WARNING or was not
    started from this checkout's commit: either mismatch would make the
    timings incomparable to the single-machine baseline without saying so.
    ``client.get_versions(check=True)`` covers the Python/library versions.
    """
    import distributed

    client = distributed.Client(address)
    client.get_versions(check=True)
    client.run(_quiet_worker)
    levels = client.run(lambda: os.environ.get(ENV_LOG_LEVEL))
    bad_levels = {worker: level for worker, level in levels.items() if level != "WARNING"}
    if bad_levels:
        raise SystemExit(f"workers not running with {ENV_LOG_LEVEL}=WARNING: {bad_levels}")
    revisions = client.run(_revision)
    bad_revisions = {worker: rev for worker, rev in revisions.items() if rev != revision}
    if bad_revisions:
        raise SystemExit(f"workers not on checkout revision {revision}: {bad_revisions}")
    return client


def _run_distributed_sweep(
    index: _Index,
    grid: _Grid,
    client: Any,
    workers: tuple[int, ...],
    repeat: int,
    serial_digest: str,
    address: str,
    read_seconds: float,
    write_seconds: float,
) -> None:
    """Sweep worker counts on a connected ``dask.distributed`` cluster, printing each row.

    Split out of ``_run_real_grid`` to keep that function's branching flat. The
    eager serial digest is the equivalence gate: a mismatch raises rather than
    printing a number, so no distributed speedup is ever reported for a result
    that differs from the serial one.
    """
    print(
        f"\nDask worker counts: {','.join(str(count) for count in workers)}; scheduler=distributed "
        f"({address}); time=-1 (ADR-0003); every result gated against the eager serial digest; "
        "compute-only seconds exclude distribute; total adds read + distribute + compute + the eager serial write"
    )
    print(
        f"{'workers':>8} {'hosts':>6} {'blocks':>7} {'distribute':>10} {'compute':>9} {'speedup':>8} "
        f"{'total':>9}  samples"
    )
    baseline = None
    for worker_count in workers:
        addresses, hosts = _select_workers(client.scheduler_info()["workers"], worker_count)
        distribute_seconds, samples, _ = _measure_distributed(index, grid, client, addresses, repeat, serial_digest)
        if baseline is None:
            baseline = min(samples)
        speedup = baseline / min(samples)
        blocks = _spatial_blocks(grid.precip, worker_count)
        total = read_seconds + distribute_seconds + min(samples) + write_seconds
        print(
            f"{worker_count:>8} {hosts:>6} {blocks:>7} {distribute_seconds:>10.3f} {min(samples):>9.3f} "
            f"{speedup:>7.2f}x {total:>9.3f}  [{_format_samples(samples)}]"
        )


def _run_real_grid(args: argparse.Namespace) -> None:
    """Benchmark SPI on a real NetCDF grid: read, eager serial, then Dask workers.

    The read is timed on its own, and the Dask table reports compute-only
    seconds plus a total that adds the read and, when requested, the write of
    the eager serial result, so the NetCDF I/O never hides inside a compute
    figure. ``--scheduler`` switches the worker sweep from the local
    ``processes`` scheduler to a ``dask.distributed`` cluster (#1127); that
    branch adds a distribute-time column and gates every result against the
    eager serial digest before reporting it.
    """
    scale = args.scale or 6
    distribution = Distribution[args.distribution or "gamma"]
    calibration_initial = args.calibration_start or 1991
    calibration_final = args.calibration_end or 2020

    read_start = time.perf_counter()
    grid = load_netcdf_grid(args.netcdf, args.var_name)
    read_seconds = time.perf_counter() - read_start

    lat_cells, lon_cells = grid.precip.sizes["lat"], grid.precip.sizes["lon"]
    cells = lat_cells * lon_cells

    data_start_year = int(grid.precip["time"].dt.year[0])

    def run(_grid: _Grid) -> xr.DataArray:
        """Run SPI-``scale`` with the real-grid mode's parameters."""
        return spi(
            values=_grid.precip,
            scale=scale,
            distribution=distribution,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_initial,
            calibration_year_final=calibration_final,
            periodicity=Periodicity.monthly,
        )

    index = _Index(run, scale - 1)
    valid = grid.valid_cells
    assert valid is not None

    revision = _revision()
    print(f"checkout revision: {revision}")
    print(f"fixture: {os.path.abspath(args.netcdf)} sha256={_hash_file(args.netcdf)}")
    print(f"grid: time={grid.precip.sizes['time']} lat={lat_cells} lon={lon_cells}")
    print(
        f"input: scale={scale}; calibration={calibration_initial}-{calibration_final}; "
        f"distribution={distribution.value}; zeros replaced with 0.01 mm; "
        f"land mask from the first time step ({int(valid.sum())} of {valid.size} cells); "
        f"spatial axes rolled by lat={grid.precip.attrs['roll_lat']} lon={grid.precip.attrs['roll_lon']} "
        "so the sampled preflight cell holds data"
    )
    print(_environment())

    client = None
    if args.scheduler:
        client = _connect_cluster(args.scheduler, revision)
        cluster_workers = client.scheduler_info()["workers"]
        workers = args.cores or _default_workers(cells, cap=len(cluster_workers))
        _require_worker_counts(workers, cells)
        _require_cluster_capacity(workers, len(cluster_workers))
        print(f"cluster: {len(cluster_workers)} worker processes at {args.scheduler}")
    else:
        workers = args.cores or _default_workers(cells)
        _require_worker_counts(workers, cells)

    print(f"process: read {read_seconds:.3f} s; {args.repeat} timed runs per configuration after a warm-up")

    _quiet_logging()
    index.run(grid)
    serial = tuple(_time_serial(grid, index) for _ in range(args.repeat))
    serial_digest = None
    if args.scheduler:
        # one extra untimed pass: the digest is the equivalence gate, not a timing
        serial_digest = hashlib.sha256(memoryview(np.ascontiguousarray(index.run(grid).values))).hexdigest()
    write_seconds = _write_output(index, grid, args.write_output)
    serial_total = read_seconds + min(serial) + write_seconds
    print(
        f"\nspi eager serial in-memory (quiet log): samples=[{_format_samples(serial)}] "
        f"min={min(serial):.3f} s; read={read_seconds:.3f} s; write={write_seconds:.3f} s; "
        f"total={serial_total:.3f} s"
    )
    if args.serial_only:
        return

    if client is not None:
        assert serial_digest is not None
        assert args.scheduler is not None
        _run_distributed_sweep(
            index,
            grid,
            client,
            workers,
            args.repeat,
            serial_digest,
            args.scheduler,
            read_seconds,
            write_seconds,
        )
        return

    print(
        f"\nDask worker counts: {','.join(str(count) for count in workers)}; scheduler=processes; chunksize=1; "
        "time=-1 (ADR-0003); compute-only seconds, total adds read + the eager serial write"
    )
    print(f"{'workers':>8} {'blocks':>7} {'compute':>9} {'speedup':>8} {'total':>9}  samples")
    baseline = None
    for worker_count in workers:
        samples = _measure(index, grid, worker_count, args.repeat)
        if baseline is None:
            baseline = min(samples)
        speedup = baseline / min(samples)
        blocks = _spatial_blocks(grid.precip, worker_count)
        total = read_seconds + min(samples) + write_seconds
        print(
            f"{worker_count:>8} {blocks:>7} {min(samples):>9.3f} {speedup:>7.2f}x {total:>9.3f}  "
            f"[{_format_samples(samples)}]"
        )


def _write_output(index: _Index, grid: _Grid, path: str | None) -> float:
    """Time writing the computed SPI result to NetCDF, or return 0.0 when no path was given.

    The compute and the ``float32`` cast happen before the timer, so the seconds
    are NetCDF encoding and disk I/O, not a second compute figure. ``grid`` may
    have been rolled to give the calibration preflight a land cell at ``[0, 0]``
    (see ``load_netcdf_grid``); the result is rolled back before writing so the
    file on disk carries the input's own monotonic lat/lon order rather than the
    wrapped one.
    """
    if not path:
        return 0.0
    payload = index.run(grid).astype("float32").rename("spi")
    roll_lat = grid.precip.attrs.get("roll_lat")
    roll_lon = grid.precip.attrs.get("roll_lon")
    if roll_lat is not None and roll_lon is not None:
        payload = payload.roll(lat=-roll_lat, lon=-roll_lon, roll_coords=True)
    # the loader's marker attrs describe the in-memory roll, not the file on disk
    for attr in ("roll_lat", "roll_lon", "land_cells"):
        payload.attrs.pop(attr, None)
    start = time.perf_counter()
    payload.to_netcdf(path)
    return time.perf_counter() - start


def _parse_args() -> argparse.Namespace:
    """Parse and validate the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cores",
        type=_worker_counts,
        help="comma-separated Dask worker counts (default: powers of two up to the CPU count)",
    )
    parser.add_argument(
        "--indices", help="comma-separated indices to benchmark (default: spi,spei; spi only with --netcdf)"
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="timed runs per configuration (default: 3); every sample is reported"
    )
    parser.add_argument(
        "--serial-only",
        action="store_true",
        help="time only the in-memory single-process path, skipping the Dask worker counts",
    )
    parser.add_argument(
        "--netcdf",
        help="NetCDF precipitation grid (dims time, lat, lon) to benchmark instead of the synthetic grid",
    )
    parser.add_argument("--var-name", help="precipitation variable in --netcdf (default: precip)")
    parser.add_argument("--scale", type=int, help="SPI timescale in --netcdf mode (default: 6)")
    parser.add_argument("--calibration-start", type=int, help="first calibration year in --netcdf mode (default: 1991)")
    parser.add_argument("--calibration-end", type=int, help="last calibration year in --netcdf mode (default: 2020)")
    parser.add_argument(
        "--distribution",
        choices=("gamma", "pearson"),
        help="SPI distribution in --netcdf mode (default: gamma); pearson degenerates on heavily masked grids (#1118)",
    )
    parser.add_argument("--write-output", help="write the computed SPI result to this NetCDF and report the write time")
    parser.add_argument(
        "--scheduler",
        help="dask.distributed scheduler address (e.g. tcp://host:8786) in --netcdf mode; --cores then "
        "counts worker processes on that cluster instead of local processes (#1127)",
    )
    args = parser.parse_args()
    args.indices = args.indices or ("spi" if args.netcdf else "spi,spei")
    unknown = sorted(set(args.indices.split(",")) - _RUNNERS.keys())
    if unknown:
        parser.error(f"unknown indices: {', '.join(unknown)}")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.netcdf and args.indices != "spi":
        parser.error("--netcdf benchmarks SPI only, because the fixture carries no PET or temperature")
    if not args.netcdf:
        netcdf_only = [
            flag
            for flag, value in (
                ("--var-name", args.var_name),
                ("--scale", args.scale),
                ("--calibration-start", args.calibration_start),
                ("--calibration-end", args.calibration_end),
                ("--distribution", args.distribution),
                ("--write-output", args.write_output),
                ("--scheduler", args.scheduler),
            )
            if value is not None
        ]
        if netcdf_only:
            parser.error(f"{', '.join(netcdf_only)} require --netcdf")
    elif args.scale is not None and args.scale < 1:
        parser.error("--scale must be at least 1")
    if args.var_name == "":
        parser.error("--var-name must not be empty")
    args.var_name = args.var_name or "precip"
    return args


def _quiet_logging() -> None:
    """Pin logging off for the Dask runs.

    Workers inherit the log level from the environment, and the pool
    initializer installs the goodness-of-fit filter in each worker, so
    unrelated warning categories stay visible on both sides of the pool.
    """
    os.environ[ENV_LOG_LEVEL] = "WARNING"
    logging.getLogger().setLevel(logging.WARNING)


def _quiet_worker() -> None:
    """Silence goodness-of-fit warnings, in the parent process and in Dask workers."""
    warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)


def main() -> None:
    """Benchmark every requested index: serial in-memory, then across Dask workers."""
    args = _parse_args()
    _quiet_worker()
    if args.netcdf:
        _run_real_grid(args)
        return
    workers = args.cores or _default_workers()
    _require_worker_counts(workers, REFERENCE_LAT * REFERENCE_LON)
    print(
        f"reference grid: {REFERENCE_LAT}x{REFERENCE_LON} cells, {REFERENCE_YEARS} years monthly; scale={SCALE}; "
        f"calibration={CALIBRATION_PERIOD[0]}-{CALIBRATION_PERIOD[1]}"
    )
    print(f"checkout revision: {_revision()}")
    print(_environment() + f"; {args.repeat} runs after a warm-up, every sample retained")

    grid = build_inputs()
    names = args.indices.split(",")
    for name in names:
        index = _RUNNERS[name]
        # one untimed run keeps first-call imports and caches out of the samples
        index.run(grid)
        info, quiet = _serial_timings(grid, index, args.repeat)
        print(
            f"\n{name}\nserial in-memory: {min(info):.3f} s (INFO, GoF warnings filtered) | "
            f"{min(quiet):.3f} s (quiet log); samples INFO=[{_format_samples(info)}] quiet=[{_format_samples(quiet)}]"
        )
    if args.serial_only:
        return

    _quiet_logging()
    print(f"\nDask worker counts: {','.join(str(count) for count in workers)}; scheduler=processes")
    for name in names:
        index = _RUNNERS[name]
        print(f"\n{name}\n{'workers':>8} {'blocks':>7} {'seconds':>9} {'speedup':>8} {'efficiency':>11}  samples")
        baseline = None
        for worker_count in workers:
            samples = _measure(index, grid, worker_count, args.repeat)
            seconds = min(samples)
            if baseline is None:
                baseline = seconds
            speedup = baseline / seconds
            blocks = _spatial_blocks(grid.precip, worker_count)
            # relative to the baseline's worker count, which need not be one
            efficiency = speedup * workers[0] / worker_count
            print(
                f"{worker_count:>8} {blocks:>7} {seconds:>9.3f} {speedup:>7.2f}x {efficiency:>10.0%}  "
                f"[{_format_samples(samples)}]"
            )


if __name__ == "__main__":
    main()

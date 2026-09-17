"""Benchmark gridded index scaling from 1 to N Dask workers.

Runs the #893 reference grid (38x87 cells, 40 years of monthly precipitation)
through the public xarray API on a Dask-backed input, once per requested worker
count with the ``processes`` scheduler, and prints the wall clock, the speedup
against the first worker count, and the parallel efficiency. Before each worker
count it also times the in-memory, single-process call on the same grid, at the
default INFO log level and with logging quiet, which is the serial reference the
parallel numbers and the epic's speedup criterion are measured against.

Spatial chunking follows the worker count so every worker gets blocks of
similar size, while ``time`` stays a single chunk (ADR-0003). The grid size is
fixed across worker counts, so the numbers are strong scaling at the reference
size.

PET for SPEI is synthetic (a fixed fraction of the precipitation), so the run
measures the fitting path rather than the PET kernels. Per-cell logging and
goodness-of-fit warnings are pinned off, because at this grid size they cost
more than the computation itself.

Run from the repository root::

    uv run benchmarks/parallel_scaling.py
    uv run benchmarks/parallel_scaling.py --cores 1,2,3,4 --indices spi,spei --repeat 5
    uv run benchmarks/parallel_scaling.py --indices spi,spei,pet,eddi --serial-only
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import platform
import time
import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
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

from climate_indices import eddi, pet_thornthwaite, spei
from climate_indices.compute import Periodicity
from climate_indices.exceptions import GoodnessOfFitWarning
from climate_indices.indices import Distribution
from climate_indices.logging_config import ENV_LOG_LEVEL

PET_FRACTION = 0.6
TEMPERATURE_SEED = SEED + 1


class _Grid(NamedTuple):
    """Reference-grid inputs shared by every index runner."""

    precip: xr.DataArray
    pet: xr.DataArray
    temperature: xr.DataArray


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

    Latitude is a ``(lat,)`` coordinate rather than a scalar so the spatial
    kernel receives it as a broadcast input (the scalar form stays on the
    per-cell path).
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


_RUNNERS: dict[str, _Runner] = {"spi": _run_spi, "spei": run_spei, "pet": run_pet, "eddi": run_eddi}


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


def _require_finite_tail(values: np.ndarray) -> None:
    """Reject output that degenerated past the leading scale-1 padding.

    Every benchmarked index pads the first ``scale - 1`` time steps with NaN;
    anything else non-finite means the fit degenerated and the timing above
    measures nothing useful.
    """
    if not np.isfinite(values[SCALE - 1 :]).all():
        raise RuntimeError(f"non-finite output beyond the leading {SCALE - 1} padded time steps")


def _time_serial(grid: _Grid, runner: _Runner) -> float:
    """Run ``runner`` on the in-memory grid once and return the elapsed seconds."""
    start = time.perf_counter()
    values = runner(grid).values
    elapsed = time.perf_counter() - start
    _require_finite_tail(values)
    return elapsed


def _serial_timings(grid: _Grid, runner: _Runner, repeat: int) -> tuple[float, float]:
    """Time the serial in-memory call at the default INFO level, then quiet.

    INFO is what an unconfigured caller pays; the WARNING run isolates the
    fitting path from the per-cell log volume the INFO run includes.

    Args:
        grid: reference-grid inputs
        runner: index to benchmark
        repeat: timed runs per level, the fastest of each is reported

    Returns:
        Fastest INFO seconds and fastest WARNING seconds
    """
    logging.getLogger().setLevel(logging.INFO)
    info = min(_time_serial(grid, runner) for _ in range(repeat))
    logging.getLogger().setLevel(logging.WARNING)
    quiet = min(_time_serial(grid, runner) for _ in range(repeat))
    return info, quiet


def _measure(runner: _Runner, grid: _Grid, workers: int, repeat: int) -> float:
    """Return the fastest of ``repeat`` runs on ``workers`` Dask processors.

    One extra warm-up run keeps parent-side first-call imports out of the timed
    samples. Every ``compute()`` call creates a fresh process pool, so pool
    start-up stays inside every measurement, as it does for any caller of the
    ``processes`` scheduler. The pool initializer installs the goodness-of-fit
    filter in each worker before any task runs.
    """
    inputs = _Grid(
        precip=_chunk_for_workers(grid.precip, workers),
        pet=_chunk_for_workers(grid.pet, workers),
        temperature=_chunk_for_workers(grid.temperature, workers),
    )
    timings = []
    for _ in range(repeat + 1):
        start = time.perf_counter()
        # chunksize=1: the default batches up to six ready tasks per submission, which
        # runs a whole six-block batch sequentially on one worker
        result = runner(inputs).compute(
            scheduler="processes",
            num_workers=workers,
            chunksize=1,
            initializer=_quiet_worker,
        )
        timings.append(time.perf_counter() - start)
    _require_finite_tail(result.values)
    return min(timings[1:])


def _spatial_blocks(array: xr.DataArray, workers: int) -> int:
    """Number of spatial blocks ``workers`` produces for ``array``."""
    chunked = _chunk_for_workers(array, workers)
    return math.prod(len(axis_chunks) for axis_chunks in chunked.chunks[1:])


def _default_workers() -> tuple[int, ...]:
    """Powers of two from 1 up to the CPU count, never above the grid's cells."""
    limit = min(os.cpu_count() or 1, REFERENCE_LAT * REFERENCE_LON)
    counts = [1]
    while counts[-1] * 2 <= limit:
        counts.append(counts[-1] * 2)
    return tuple(counts)


def _worker_counts(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of worker counts.

    Counts outside ``1..spatial cells`` cannot produce a block per worker, so
    they are rejected as argument errors rather than failing mid-benchmark.
    """
    counts = tuple(int(count) for count in value.split(","))
    if any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("worker counts must be at least 1")
    cells = REFERENCE_LAT * REFERENCE_LON
    if any(count > cells for count in counts):
        raise argparse.ArgumentTypeError(f"worker counts must not exceed the {cells} spatial cells")
    return counts


def _parse_args() -> argparse.Namespace:
    """Parse and validate the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cores",
        type=_worker_counts,
        help="comma-separated Dask worker counts (default: powers of two up to the CPU count)",
    )
    parser.add_argument(
        "--indices", default="spi,spei", help="comma-separated indices to benchmark (default: spi,spei)"
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="timed runs per worker count, fastest reported (default: 3)"
    )
    parser.add_argument(
        "--serial-only",
        action="store_true",
        help="time only the in-memory single-process path, skipping the Dask worker counts",
    )
    args = parser.parse_args()
    unknown = sorted(set(args.indices.split(",")) - _RUNNERS.keys())
    if unknown:
        parser.error(f"unknown indices: {', '.join(unknown)}")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def _quiet_gof_warnings() -> None:
    """Silence goodness-of-fit warnings, which cost more than the computation."""
    warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)


def _quiet_logging() -> None:
    """Pin logging off for the parallel runs.

    Workers inherit the log level from the environment, and the pool
    initializer installs the goodness-of-fit filter in each worker, so
    unrelated warning categories stay visible on both sides of the pool.
    """
    os.environ[ENV_LOG_LEVEL] = "WARNING"
    logging.getLogger().setLevel(logging.WARNING)


def _quiet_worker() -> None:
    """Silence goodness-of-fit warnings inside a Dask worker process."""
    warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)


def main() -> None:
    """Benchmark every requested index: serial in-memory, then across Dask workers."""
    args = _parse_args()
    _quiet_gof_warnings()
    workers = args.cores or _default_workers()
    print(
        f"reference grid: {REFERENCE_LAT}x{REFERENCE_LON} cells, {REFERENCE_YEARS} years monthly; scale={SCALE}; "
        f"calibration={CALIBRATION_PERIOD[0]}-{CALIBRATION_PERIOD[1]}"
    )
    print(
        f"environment: python {platform.python_version()}; {platform.platform()}; {os.cpu_count()} CPUs; "
        f"fastest of {args.repeat} runs after a warm-up"
    )

    grid = build_inputs()
    names = args.indices.split(",")
    for name in names:
        runner = _RUNNERS[name]
        # one untimed run keeps first-call imports and caches out of the samples
        runner(grid)
        info, quiet = _serial_timings(grid, runner, args.repeat)
        print(f"\n{name}\nserial in-memory: {info:.3f} s (INFO) | {quiet:.3f} s (quiet log)")
    if args.serial_only:
        return

    _quiet_logging()
    print(f"\nDask worker counts: {','.join(str(count) for count in workers)}; scheduler=processes")
    for name in names:
        runner = _RUNNERS[name]
        print(f"\n{name}\n{'workers':>8} {'blocks':>7} {'seconds':>9} {'speedup':>8} {'efficiency':>11}")
        baseline = None
        for worker_count in workers:
            seconds = _measure(runner, grid, worker_count, args.repeat)
            if baseline is None:
                baseline = seconds
            speedup = baseline / seconds
            blocks = _spatial_blocks(grid.precip, worker_count)
            # relative to the baseline's worker count, which need not be one
            efficiency = speedup * workers[0] / worker_count
            print(f"{worker_count:>8} {blocks:>7} {seconds:>9.3f} {speedup:>7.2f}x {efficiency:>10.0%}")


if __name__ == "__main__":
    main()

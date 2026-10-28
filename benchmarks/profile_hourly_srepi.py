"""Measure hourly SREPI on a realistic hourly grid (#1177).

Times ``energy.srepi`` in the 8784-slot hourly layout for a 30-year calibration
period, on a series and on a small gridded block, and reports peak allocation per
call. The shapes follow the C3S energy-indicator source: hourly values, not the
aggregated monthly product. Reported figures are evidence artifacts, not CI gates.

Run from the repository root::

    uv run benchmarks/profile_hourly_srepi.py --backend python
    uv run benchmarks/profile_hourly_srepi.py --backend native

A warm-up call runs first, so first-call imports and caches stay out of the
measurement window. Both backends ignore NumPy floating-point reporting, as native
execution requires. The native run requires the built extension; long series
convolutions still use NumPy. Peak traced allocation excludes Rust buffers; use an
external process-memory measurement as well (for example, macOS `/usr/bin/time -l`).
"""

from __future__ import annotations

import argparse
import platform
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from climate_indices import compute, energy

HOURS_PER_YEAR = compute.Periodicity.hourly.period_length
DATA_START_YEAR = 1980
CALIBRATION_PERIOD = (1981, 2010)
DEFAULT_YEARS = 41
DEFAULT_CELLS = 25


@dataclass(frozen=True)
class Measurement:
    """One timed call and the peak allocation it made."""

    name: str
    shape: tuple[int, ...]
    seconds: float
    peak_mebibytes: float


def _hourly_block(years: int, cells: int = 0) -> np.ndarray:
    """Build a deterministic hourly series, or a time-major block of ``cells`` cells.

    The block is three-dimensional, (time, cells, 1), because a two-dimensional
    (time, cells) input is read as one series in the legacy (years, periods) layout.
    """
    time_steps = years * HOURS_PER_YEAR
    hours_of_year = np.arange(time_steps) % HOURS_PER_YEAR
    diurnal = np.sin(2.0 * np.pi * hours_of_year / 24.0) + 2.0
    seasonal = np.cos(2.0 * np.pi * hours_of_year / HOURS_PER_YEAR)
    series = 1.5 + diurnal + seasonal
    if not cells:
        return series
    offsets = (np.arange(cells, dtype=float) / 10.0).reshape(1, cells, 1)
    return np.asarray(series[:, np.newaxis, np.newaxis] + offsets)


def _measure(name: str, values: np.ndarray, call: Callable[[], object]) -> Measurement:
    """Time one call and record its peak traced allocation."""
    tracemalloc.start()
    started = time.perf_counter()
    call()
    seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Measurement(name=name, shape=values.shape, seconds=seconds, peak_mebibytes=peak / (1024 * 1024))


def _run(years: int, cells: int) -> list[Measurement]:
    """Measure the series and gridded hourly calls for a span and grid width."""
    span = (DATA_START_YEAR, DATA_START_YEAR + years - 1)
    measurements: list[Measurement] = []
    for name, block_cells in (("series", 0), ("grid", cells)):
        values = _hourly_block(years, block_cells)
        call = lambda values=values: energy.srepi(  # noqa: E731 - keeps the timing window tight
            values,
            scale=24 * 30,
            data_start_year=DATA_START_YEAR,
            calibration_year_initial=CALIBRATION_PERIOD[0],
            calibration_year_final=CALIBRATION_PERIOD[1],
            periodicity=compute.Periodicity.hourly,
            aggregation="mean",
            spatial_time_major=values.ndim > 1,
        )
        call()  # warm-up
        measurements.append(_measure(name, values, call))
    print(f"span {span[0]}-{span[1]} ({years} years, scale 720 hours, cells {cells if cells else 0})")
    # A short monthly Timescale checks dispatch overhead outside the hourly case.
    monthly = np.random.default_rng(1485).uniform(0, 100, size=years * 12)
    for name, values in (
        ("monthly-series", monthly),
        ("monthly-grid", np.broadcast_to(monthly[:, None, None], (monthly.size, cells, 1)).copy()),
    ):

        def monthly_call(values: np.ndarray = values) -> np.ndarray:
            return compute.sum_to_scale(values, 3)

        warmed = monthly_call()
        if warmed.shape != values.shape:
            raise RuntimeError(f"{name}: aggregation returned an unexpected shape")
        if not np.isnan(warmed[:2]).all():
            raise RuntimeError(f"{name}: aggregation did not preserve leading NaNs")
        measurements.append(_measure(name, values, monthly_call))
    return measurements


def main() -> None:
    """Measure hourly SREPI and print the timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="years of hourly input")
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS, help="grid cells for the gridded call")
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    arguments = parser.parse_args()
    if arguments.backend == "native":
        if compute._native is None or not hasattr(compute._native, "trailing_sum"):
            raise RuntimeError("Build the extension with uv run maturin develop --release first")
    else:
        compute._native = None

    print(f"backend {arguments.backend}")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"period length {HOURS_PER_YEAR} hours, numpy {np.__version__}")
    with np.errstate(all="ignore"):
        measurements = _run(arguments.years, arguments.cells)
    for measurement in measurements:
        print(
            f"  {measurement.name:<7} input {str(measurement.shape):<18} "
            f"{measurement.seconds:10.6f} s  peak {measurement.peak_mebibytes:9.1f} MiB"
        )


if __name__ == "__main__":
    main()

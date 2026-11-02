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
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_indices import compute, energy  # noqa: E402
from tests import conftest  # noqa: E402

HOURS_PER_YEAR = compute.Periodicity.hourly.period_length
DATA_START_YEAR = 1980
CALIBRATION_PERIOD = (1981, 2010)
DEFAULT_YEARS = 41
DEFAULT_CELLS = 25


@dataclass(frozen=True)
class Measurement:
    """The fastest timed call and the peak allocation the call made."""

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
    """Time repeated calls without tracing, then record the peak traced allocation.

    An active tracer inflates the timed window and penalizes the backend with more
    tracked allocations, so the fastest of at least three calls, and at least 0.05 s of
    calls, is reported. The peak comes from a separate traced call.
    """
    deadline = time.perf_counter() + 0.05
    seconds = float("inf")
    repetitions = 0
    while repetitions < 3 or time.perf_counter() < deadline:
        started = time.perf_counter()
        call()
        seconds = min(seconds, time.perf_counter() - started)
        repetitions += 1
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Measurement(name=name, shape=values.shape, seconds=seconds, peak_mebibytes=peak / (1024 * 1024))


_Result = TypeVar("_Result")


def _warm_up(call: Callable[[], _Result], name: str, required: set[str], verify_native: bool) -> _Result:
    """Verify dispatch outside the timing window, without adding recorder overhead to timings."""
    if not verify_native:
        return call()
    selected = compute._native
    if selected is None:
        raise RuntimeError("Build the extension with uv run maturin develop --release first")
    recorder = conftest.NativeRecorder(selected)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compute, "_native", recorder)
        result = call()
    if not required <= recorder.calls:
        raise RuntimeError(f"{name}: native kernels did not run: {sorted(required - recorder.calls)}")
    print(f"  {name} verified kernels {sorted(recorder.calls)}")
    return result


def _run(years: int, cells: int, pool_window_days: int = 0, verify_native: bool = False) -> list[Measurement]:
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
            pool_window_days=pool_window_days,
        )
        # pooled ranking keeps the empirical transform in Python by design, so only
        # an unpooled run expects that kernel
        required = set() if pool_window_days else {"empirical_transform"}
        if block_cells:
            required.add("trailing_sum")
        _warm_up(call, name, required, verify_native)
        measurements.append(_measure(name, values, call))
    print(
        f"span {span[0]}-{span[1]} ({years} years, scale 720 hours, cells {cells if cells else 0}, "
        f"pool_window_days {pool_window_days})"
    )
    # A short monthly Timescale checks the one-dimensional dispatch cost outside the hourly case.
    monthly = np.random.default_rng(1485).uniform(0, 100, size=years * 12)
    for name, values in (
        ("monthly-series", monthly),
        ("monthly-grid", np.broadcast_to(monthly[:, None, None], (monthly.size, cells, 1)).copy()),
    ):

        def monthly_call(values: np.ndarray = values) -> np.ndarray:
            return compute.sum_to_scale(values, 3)

        warmed = _warm_up(monthly_call, name, {"trailing_sum"}, verify_native)
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
    parser.add_argument(
        "--pool-window-days",
        type=int,
        default=0,
        help="neighbouring days pooled into each hourly step; 0 keeps the hour x day-of-year strata",
    )
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    arguments = parser.parse_args()
    selected = compute._native
    try:
        if arguments.backend == "native":
            if selected is None or not hasattr(selected, "trailing_sum"):
                raise RuntimeError("Build the extension with uv run maturin develop --release first")
        else:
            compute._native = None

        print(f"backend {arguments.backend}")
        print(f"python {platform.python_version()} on {platform.platform()}")
        print(f"period length {HOURS_PER_YEAR} hours, numpy {np.__version__}")
        with np.errstate(all="ignore"):
            measurements = _run(
                arguments.years,
                arguments.cells,
                arguments.pool_window_days,
                verify_native=arguments.backend == "native",
            )
    finally:
        # a later run in the same process must see the imported extension again
        compute._native = selected
    for measurement in measurements:
        print(
            f"  {measurement.name:<7} input {str(measurement.shape):<18} "
            f"{measurement.seconds:10.6f} s  peak {measurement.peak_mebibytes:9.1f} MiB"
        )


if __name__ == "__main__":
    main()

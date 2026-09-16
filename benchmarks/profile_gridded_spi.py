"""Profile the canonical gridded SPI workflow on the reference grid.

Runs ``climate_indices.spi`` on a deterministic synthetic grid matching the
reference shape from the #893 performance epic (38x87 spatial cells, 40 years
of monthly precipitation) under ``cProfile`` and writes the raw report.

Run from the repository root::

    uv run benchmarks/profile_gridded_spi.py

The Numba kernels are compiled on a small warm-up grid first, so JIT
compilation does not dominate the profile.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import platform
import pstats
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import spi
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution

REFERENCE_LAT = 38
REFERENCE_LON = 87
REFERENCE_YEARS = 40
DATA_START_YEAR = 1980
CALIBRATION_PERIOD = (1981, 2010)
SCALE = 3
SEED = 42

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "profile_gridded_spi.txt"


def build_grid(lat: int, lon: int, years: int) -> xr.DataArray:
    """Build deterministic gamma-distributed monthly precipitation.

    Args:
        lat: number of latitude cells
        lon: number of longitude cells
        years: number of complete calendar years, starting in DATA_START_YEAR

    Returns:
        DataArray with dims (time, lat, lon), units "mm"
    """
    rng = np.random.default_rng(SEED)
    n_months = years * 12
    values = rng.gamma(shape=2.0, scale=50.0, size=(n_months, lat, lon))
    time = pd.date_range(f"{DATA_START_YEAR}-01-01", periods=n_months, freq="MS")
    return xr.DataArray(
        values,
        coords={
            "time": time,
            "lat": np.linspace(25.0, 50.0, lat),
            "lon": np.linspace(-125.0, -70.0, lon),
        },
        dims=["time", "lat", "lon"],
        attrs={"units": "mm"},
    )


def run_spi(precip: xr.DataArray) -> xr.DataArray:
    """Run the canonical xarray SPI path with the epic's reference parameters."""
    return spi(
        values=precip,
        scale=SCALE,
        distribution=Distribution.gamma,
        data_start_year=DATA_START_YEAR,
        calibration_year_initial=CALIBRATION_PERIOD[0],
        calibration_year_final=CALIBRATION_PERIOD[1],
        periodicity=Periodicity.monthly,
    )


def profile(top: int) -> float:
    """Profile SPI on the reference grid, writing the raw report to ``DEFAULT_OUTPUT``.

    Returns:
        Wall-clock seconds spent inside the profiled SPI call.
    """
    # compile the Numba kernels on a small grid first: JIT time is not part of
    # the steady-state bottleneck this script exists to measure
    warmup = build_grid(lat=2, lon=2, years=CALIBRATION_PERIOD[1] - DATA_START_YEAR + 1)
    run_spi(warmup)

    precip = build_grid(lat=REFERENCE_LAT, lon=REFERENCE_LON, years=REFERENCE_YEARS)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    run_spi(precip)
    profiler.disable()
    elapsed = time.perf_counter() - start

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    log_level = os.environ.get("CLIMATE_INDICES_LOG_LEVEL", "INFO")
    with DEFAULT_OUTPUT.open("w") as stream:
        print(
            f"grid: time={precip.sizes['time']} lat={precip.sizes['lat']} "
            f"lon={precip.sizes['lon']}; scale={SCALE}; "
            f"calibration={CALIBRATION_PERIOD[0]}-{CALIBRATION_PERIOD[1]}; "
            f"elapsed={elapsed:.1f}s",
            file=stream,
        )
        print(
            f"environment: python {platform.python_version()}; {platform.platform()}; "
            f"climate_indices log level {log_level}",
            file=stream,
        )
        for sort_key in ("cumulative", "tottime"):
            print(f"\n--- sorted by {sort_key} ---", file=stream)
            stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(sort_key)
            stats.print_stats(top)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=30, help="number of hottest entries per sort order")
    args = parser.parse_args()

    elapsed = profile(args.top)
    print(f"reference-grid SPI: {elapsed:.1f}s -> {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()

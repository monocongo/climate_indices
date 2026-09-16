"""Profile the gridded SPI workflow on the #893 reference grid.

Runs ``climate_indices.spi`` on a deterministic synthetic grid matching the
reference shape from the #893 performance epic (38x87 spatial cells, 40 years
of monthly precipitation) under ``cProfile`` and writes the raw report plus the
timings that contextualize it.

This exercises the numpy-backed (in-memory) adapter branch, the serial baseline
for gridded SPI. A Dask-backed input returns a lazy result from the same
per-cell ``apply_ufunc`` loop; Dask scheduling and multi-core scaling belong to
#927 and #928.

Run from the repository root::

    uv run benchmarks/profile_gridded_spi.py

The fitting/transform path is exercised on a small warm-up grid first, so
first-call imports and caches do not dominate the measurement window.
"""

from __future__ import annotations

import argparse
import cProfile
import logging
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
WARMUP_LAT = 2
WARMUP_LON = 2
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
    """Run the in-memory xarray SPI path with the epic's reference parameters."""
    return spi(
        values=precip,
        scale=SCALE,
        distribution=Distribution.gamma,
        data_start_year=DATA_START_YEAR,
        calibration_year_initial=CALIBRATION_PERIOD[0],
        calibration_year_final=CALIBRATION_PERIOD[1],
        periodicity=Periodicity.monthly,
    )


def _time_spi(precip: xr.DataArray) -> float:
    """Run SPI once and return the elapsed seconds."""
    start = time.perf_counter()
    run_spi(precip)
    return time.perf_counter() - start


def profile(top: int) -> tuple[float, float, float]:
    """Profile SPI on the reference grid, writing the raw report to ``DEFAULT_OUTPUT``.

    Returns:
        Baseline seconds at INFO, profiled seconds at INFO, baseline seconds at WARNING
    """
    # exercise imports and first-call caches before measuring
    run_spi(build_grid(lat=WARMUP_LAT, lon=WARMUP_LON, years=REFERENCE_YEARS))

    precip = build_grid(lat=REFERENCE_LAT, lon=REFERENCE_LON, years=REFERENCE_YEARS)
    baseline = _time_spi(precip)

    profiler = cProfile.Profile()
    profiler.enable()
    profiled = _time_spi(precip)
    profiler.disable()

    # record the level the library actually applied: configure_logging falls
    # back to INFO for unset or invalid CLIMATE_INDICES_LOG_LEVEL values
    log_level = logging.getLevelName(logging.getLogger().level)

    # repeat at WARNING: the difference isolates the per-cell logging volume
    # that the profile attributes to the adapter
    logging.getLogger().setLevel(logging.WARNING)
    quiet = _time_spi(precip)

    # write to a temporary file and replace it only once the full report is on
    # disk, so an interrupted write never truncates the existing evidence
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = DEFAULT_OUTPUT.with_suffix(DEFAULT_OUTPUT.suffix + ".tmp")
    with tmp_output.open("w") as stream:
        print(
            f"grid: time={precip.sizes['time']} lat={precip.sizes['lat']} "
            f"lon={precip.sizes['lon']}; scale={SCALE}; "
            f"calibration={CALIBRATION_PERIOD[0]}-{CALIBRATION_PERIOD[1]}",
            file=stream,
        )
        print(
            f"environment: python {platform.python_version()}; {platform.platform()}; "
            f"climate_indices log level {log_level} (WARNING for the quiet run)",
            file=stream,
        )
        print(
            f"timings: baseline={baseline:.1f}s profiled={profiled:.1f}s warning={quiet:.1f}s",
            file=stream,
        )
        for sort_key in ("cumulative", "tottime"):
            print(f"\n--- sorted by {sort_key} ---", file=stream)
            stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(sort_key)
            stats.print_stats(top)
    tmp_output.replace(DEFAULT_OUTPUT)
    return baseline, profiled, quiet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=30, help="number of hottest entries per sort order")
    args = parser.parse_args()

    baseline, profiled, quiet = profile(args.top)
    print(
        f"reference-grid SPI: baseline={baseline:.1f}s profiled={profiled:.1f}s warning={quiet:.1f}s -> {DEFAULT_OUTPUT}"
    )


if __name__ == "__main__":
    main()

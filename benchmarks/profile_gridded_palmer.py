"""Before/after timings for the gridded Palmer PDSI kernel on the #893 reference grid.

Runs ``climate_indices.palmer.pdsi`` on a deterministic synthetic grid matching
the reference shape from the #893 performance epic (38x87 spatial cells, 40
years of monthly precipitation and PET) two ways: the per-cell loop
``__main__._apply_along_axis_palmers`` used before #937 (one ``palmer.pdsi()``
call per grid cell), and the spatial block path #937 added
(``spatial_time_major=True``, one call for the whole grid). Reports the
before/after timings in the #921 "Findings" table style; #929, which would
otherwise own the general before/after table format, is unmerged with no
branch as of #937, so this script defines its own rather than waiting on it.

Run from the repository root::

    uv run benchmarks/profile_gridded_palmer.py

A small warm-up grid is exercised first, so first-call imports and caches do
not dominate the measurement window.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
from profile_gridded_spi import REFERENCE_LAT, REFERENCE_LON, REFERENCE_YEARS

from climate_indices import palmer

DATA_START_YEAR = 1980
CALIBRATION_PERIOD = (1981, 2010)
SEED = 42

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "profile_gridded_palmer.txt"


def build_inputs(lat: int, lon: int, years: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic precip/PET/AWC for a (lat, lon) grid, in inches.

    Args:
        lat: number of latitude cells
        lon: number of longitude cells
        years: number of complete calendar years, starting in DATA_START_YEAR

    Returns:
        precip, pet: (time, lat, lon) arrays, inches
        awc: (lat, lon) array, inches
    """
    rng = np.random.default_rng(SEED)
    n_months = years * 12
    precip = rng.gamma(shape=2.0, scale=2.0, size=(n_months, lat, lon))
    pet = rng.gamma(shape=2.0, scale=1.0, size=(n_months, lat, lon))
    awc = rng.uniform(1.0, 8.0, size=(lat, lon))
    return precip, pet, awc


def run_per_cell(precip: np.ndarray, pet: np.ndarray, awc: np.ndarray) -> None:
    """The pre-#937 CLI path: one palmer.pdsi() call per grid cell."""
    lat, lon = awc.shape
    for i in range(lat):
        for j in range(lon):
            palmer.pdsi(
                precip[:, i, j],
                pet[:, i, j],
                float(awc[i, j]),
                DATA_START_YEAR,
                CALIBRATION_PERIOD[0],
                CALIBRATION_PERIOD[1],
            )


def run_block(precip: np.ndarray, pet: np.ndarray, awc: np.ndarray) -> None:
    """The #937 path: one palmer.pdsi() call for the whole grid."""
    palmer.pdsi(
        precip,
        pet,
        awc,
        DATA_START_YEAR,
        CALIBRATION_PERIOD[0],
        CALIBRATION_PERIOD[1],
        spatial_time_major=True,
    )


def _time(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def profile() -> tuple[float, float]:
    """Time the per-cell and block paths on the reference grid.

    Returns:
        Fastest per-cell seconds, fastest block seconds (minimum of two runs each)
    """
    warmup_precip, warmup_pet, warmup_awc = build_inputs(lat=2, lon=2, years=REFERENCE_YEARS)
    run_per_cell(warmup_precip, warmup_pet, warmup_awc)
    run_block(warmup_precip, warmup_pet, warmup_awc)

    precip, pet, awc = build_inputs(lat=REFERENCE_LAT, lon=REFERENCE_LON, years=REFERENCE_YEARS)
    per_cell = min(_time(run_per_cell, precip, pet, awc), _time(run_per_cell, precip, pet, awc))
    block = min(_time(run_block, precip, pet, awc), _time(run_block, precip, pet, awc))

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = DEFAULT_OUTPUT.with_suffix(DEFAULT_OUTPUT.suffix + ".tmp")
    with tmp_output.open("w") as stream:
        print(
            f"grid: time={REFERENCE_YEARS * 12} lat={REFERENCE_LAT} lon={REFERENCE_LON}; "
            f"calibration={CALIBRATION_PERIOD[0]}-{CALIBRATION_PERIOD[1]}",
            file=stream,
        )
        print(f"environment: python {platform.python_version()}; {platform.platform()}", file=stream)
        print(
            f"timings: per_cell={per_cell:.1f}s block={block:.1f}s speedup={per_cell / block:.1f}x "
            "(minimum of two runs each)",
            file=stream,
        )
    tmp_output.replace(DEFAULT_OUTPUT)
    return per_cell, block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    per_cell, block = profile()
    print(
        f"reference-grid Palmer PDSI: per_cell={per_cell:.1f}s block={block:.1f}s "
        f"speedup={per_cell / block:.1f}x -> {DEFAULT_OUTPUT}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// pyproject
# [context]
# dependencies = [
#   "numpy",
# ]
# ///
"""Prepare NOAA NCEI climate-divisional SPI reference fixtures for validation tests.

Downloads NCEI's operational climdiv-spNNdv monthly SPI files for the 344 CONUS
climate divisions and prepares them as pytest fixtures for validating the
climate_indices SPI implementation (Pearson Type III path). This script must be
run manually when refreshing the reference data.

Usage:
    uv run scripts/prepare_ncei_spi_fixtures.py

The script will:
    1. Download the seven climdiv-spNNdv timescale files from NCEI
    2. Parse each division's monthly values for 1895-2022, matching the period
       already covered by tests/fixture/palmer/<division>/precips.npy
    3. Mask NCEI's -99.99 missing-value sentinel as NaN
    4. Save one (344, 1536) float32 array per timescale under
       tests/fixture/ncei_spi/
    5. Compute and record a SHA-256 checksum and provenance.json

Source:
    https://www.ncei.noaa.gov/pub/data/cirs/climdiv/
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixture"
OUTPUT_DIR = FIXTURE_DIR / "ncei_spi"
DIVISIONS_FILE = FIXTURE_DIR / "nclimdiv" / "divisions.json"

_NCEI_BASE_URL = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv"
_FILE_VERSION = "v1.0.0-20260904"
_SCALE_ELEMENT_CODES = {1: "01", 2: "02", 3: "03", 6: "06", 9: "09", 12: "12", 24: "24"}

_DATA_START_YEAR = 1895
_DATA_END_YEAR = 2022  # matches tests/fixture/palmer/<division>/precips.npy length
_MISSING_SENTINEL = -99.99

_APPROVED_ORIGIN = "https://www.ncei.noaa.gov/"
_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024  # files are ~6 MB; anything larger is malformed
_MIN_DIVISION_COVERAGE = 0.95  # fraction of months that must be non-NaN per division

# Agreement with climate_indices.indices.spi() (Pearson III, full-period-of-record
# calibration) measured across all 344 divisions (GitHub issue #777), and the
# loose characterization ceilings the tests assert against. Written into
# provenance.json, from which tests/test_ncei_spi_reference.py loads them.
_MEASURED_STATS = {
    1: {"median": 0.0136, "p90": 0.0342, "max": 1.2900},
    2: {"median": 0.0129, "p90": 0.0298, "max": 0.9576},
    3: {"median": 0.0126, "p90": 0.0287, "max": 0.8376},
    6: {"median": 0.0126, "p90": 0.0275, "max": 0.6563},
    9: {"median": 0.0123, "p90": 0.0260, "max": 0.4088},
    12: {"median": 0.0122, "p90": 0.0252, "max": 0.3644},
    24: {"median": 0.0304, "p90": 0.0827, "max": 1.1598},
}
_CEILINGS = {
    1: {"median": 0.03, "p90": 0.07, "max": 2.2},
    2: {"median": 0.03, "p90": 0.06, "max": 1.6},
    3: {"median": 0.03, "p90": 0.06, "max": 1.4},
    6: {"median": 0.03, "p90": 0.06, "max": 1.1},
    9: {"median": 0.03, "p90": 0.06, "max": 0.7},
    12: {"median": 0.03, "p90": 0.06, "max": 0.65},
    24: {"median": 0.06, "p90": 0.17, "max": 1.9},
}

_LINE_RE = re.compile(r"^(\d{4})(\d{2})(\d{4})(.{84})")


def _download(scale: int) -> str:
    """Fetch one climdiv-spNNdv file's raw text from NCEI."""
    filename = f"climdiv-sp{_SCALE_ELEMENT_CODES[scale]}dv-{_FILE_VERSION}"
    url = f"{_NCEI_BASE_URL}/{filename}"
    if not url.startswith(_APPROVED_ORIGIN):
        raise ValueError(f"NCEI URL must use {_APPROVED_ORIGIN}, got: {url}")
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 -- URL host validated above and below
        if not response.url.startswith(_APPROVED_ORIGIN):
            raise ValueError(f"download redirected off the approved NCEI origin: {response.url}")
        payload = response.read(_MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > _MAX_DOWNLOAD_BYTES:
        raise ValueError(f"NCEI response exceeds {_MAX_DOWNLOAD_BYTES} bytes, refusing to parse: {url}")
    return payload.decode("ascii")


def _parse(raw_text: str) -> dict[str, dict[int, np.ndarray]]:
    """Parse fixed-width NCEI records into {division: {year: 12 monthly values}}."""
    by_division: dict[str, dict[int, np.ndarray]] = {}
    for line in raw_text.splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        division, _element_code, year_text, monthly_text = match.groups()
        year = int(year_text)
        values = np.array(
            [float(monthly_text[i : i + 7]) for i in range(0, 84, 7)],
            dtype=np.float32,
        )
        by_division.setdefault(division, {})[year] = values
    return by_division


def _to_series(by_year: dict[int, np.ndarray]) -> np.ndarray:
    """Flatten a division's per-year records into one 1895-2022 monthly series."""
    rows = [
        by_year.get(year, np.full(12, np.nan, dtype=np.float32)) for year in range(_DATA_START_YEAR, _DATA_END_YEAR + 1)
    ]
    series = np.concatenate(rows)
    series[series <= _MISSING_SENTINEL + 0.005] = np.nan
    return series


def _compute_checksum(directory: Path) -> str:
    hasher = hashlib.sha256()
    for npy_file in sorted(directory.glob("*.npy")):
        hasher.update(npy_file.read_bytes())
    return hasher.hexdigest()


def _write_provenance(directory: Path, checksum: str) -> None:
    provenance = {
        "source": "NOAA National Centers for Environmental Information (NCEI)",
        "url": f"{_NCEI_BASE_URL}/",
        "download_date": dt.date.today().isoformat(),
        "subset_description": (
            "Operational nClimDiv monthly Standardized Precipitation Index (SPI) for the 344 "
            "US climate divisions present in tests/fixture/palmer and tests/fixture/nclimdiv, "
            "January 1895 through December 2022 (1536 months), for timescales 1, 2, 3, 6, 9, 12 "
            "and 24 months (element codes 71-77). Each <scale>.npy is a (344, 1536) float32 array "
            "whose row order matches tests/fixture/nclimdiv/divisions.json. Source files are "
            f"climdiv-sp{{01,02,03,06,09,12,24}}dv-{_FILE_VERSION}. NCEI's -99.99 missing "
            "sentinel is stored as NaN."
        ),
        "checksum_sha256": checksum,
        "fixture_version": "1.0.0",
        # the actual characterization criteria asserted by test_ncei_spi_reference.py:
        # per-scale median/p90/max absolute-difference ceilings
        "validation_tolerance": {
            f"sp{scale:02d}_{stat}": value for scale, stats in _CEILINGS.items() for stat, value in stats.items()
        },
        "measured_stats": {f"sp{scale:02d}": stats for scale, stats in _MEASURED_STATS.items()},
        "citation": (
            "McKee, T. B., Doesken, N. J., and Kleist, J., 1993: The relationship of drought "
            "frequency and duration to time scales. Proceedings of the 8th Conference on "
            "Applied Climatology, American Meteorological Society, 179-184. "
            "Data: NOAA NCEI Climate Divisional Database (nClimDiv)."
        ),
        "license": "U.S. Government work, public domain (17 U.S.C. Sec. 105).",
        "notes": (
            "External ground truth for climate_indices.indices.spi() with "
            "Distribution.pearson (Pearson Type III), matching NCEI's documented "
            "distribution choice for this product (drought-readme.txt). Input precipitation "
            "is the existing tests/fixture/palmer/<division>/precips.npy arrays, which are "
            "byte-identical (to 2 decimals) to NCEI's climdiv-pcpndv for the same divisions "
            "and period -- verified for division 0101 at fixture creation time. "
            "CALIBRATION FINDING (empirical, resolves GitHub issue #777): NCEI's "
            "drought-readme.txt states a fixed 1931-1990 calibration window, but reproducing "
            "SPI with that window disagrees badly with these files (median abs diff "
            "0.086-0.167 across timescales, sampled divisions). Baldwin & Chen (2020, JAMC, "
            "'Major Over- and Underestimation of Drought Found in NOAA's Climate Divisional "
            "SPI Dataset') report that NCEI's actual production behavior uses a full/expanding "
            "period-of-record window instead. Using calibration_year_initial=1895, "
            "calibration_year_final=2022 (the full input period) instead reproduces these "
            "files far more closely (see calibration_check below), confirming Baldwin & "
            "Chen's finding -- so validation tests MUST use full-period-of-record "
            "calibration, not the README's stated 1931-1990 window. "
            "INDEPENDENCE: this repository's own author separately maintains a fork of NOAA/"
            "NIDIS's gridded nClimGrid-monthly SPI/SPEI/PET Python tool (per drought.gov's "
            "'Source Code: Climate and Drought Indices in Python' page), so that gridded "
            "product is excluded as circular. This climate-divisional product is part of "
            "NCEI's older Climate Divisional Database lineage (documentation dated 2014, "
            "predating this Python package's 2017 copyright) and is plausibly computed by a "
            "separate, older Fortran-derived codebase (drought.gov separately lists 'Source "
            "Code: Drought Indices in Fortran (SPI, PDSI)' as distinct public-release code), "
            "but NOAA does not publish an explicit statement that the two are unrelated, so "
            "independence is plausible but not airtight -- see docs/research/spi-dataset-survey.md. "
            "MEASURED (full-period-of-record calibration, all 344 divisions): see measured_stats."
        ),
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    divisions = json.loads(DIVISIONS_FILE.read_text(encoding="utf-8"))
    row_by_division = {division: row for row, division in enumerate(divisions)}
    n_months = (_DATA_END_YEAR - _DATA_START_YEAR + 1) * 12

    arrays = {}
    for scale in _SCALE_ELEMENT_CODES:
        print(f"Downloading SPI-{scale} ...", file=sys.stderr)
        raw_text = _download(scale)
        by_division = _parse(raw_text)

        array = np.full((len(divisions), n_months), np.nan, dtype=np.float32)
        for division, by_year in by_division.items():
            if division not in row_by_division:
                continue  # state/regional/national aggregate or a division outside our 344
            array[row_by_division[division]] = _to_series(by_year)

        # refuse to commit fixtures with absent or sparse divisions
        missing = [division for division in divisions if division not in by_division]
        sparse = [
            division
            for division in divisions
            if division not in missing
            and np.count_nonzero(~np.isnan(array[row_by_division[division]])) < _MIN_DIVISION_COVERAGE * n_months
        ]
        if missing or sparse:
            raise RuntimeError(
                f"SPI-{scale}: incomplete NCEI coverage -- {len(missing)} divisions absent {missing[:5]}, "
                f"{len(sparse)} divisions below {_MIN_DIVISION_COVERAGE:.0%} non-NaN {sparse[:5]}"
            )
        arrays[scale] = array
        print(f"  parsed sp{scale:02d} {array.shape}", file=sys.stderr)

    # stage the complete generation in a sibling directory and swap it in, so an
    # interrupted refresh never leaves a mixture of old and new arrays live
    staging = Path(tempfile.mkdtemp(prefix=".ncei_spi-staging-", dir=FIXTURE_DIR))
    try:
        for scale, array in arrays.items():
            np.save(staging / f"sp{scale:02d}.npy", array)
        checksum = _compute_checksum(staging)
        _write_provenance(staging, checksum)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for staged_file in staging.iterdir():
            os.replace(staged_file, OUTPUT_DIR / staged_file.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"Done. checksum_sha256={checksum}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// pyproject
# [context]
# dependencies = [
#   "numpy",
#   "requests",
# ]
# ///
"""Prepare paired NOAA PSL EDDI reference fixtures.

Usage:
    uv run scripts/prepare_noaa_eddi_fixtures.py

The NOAA EDDI time-series tool returns a master table with monthly reference
ET and EDDI values at every supported timescale. This script saves its 1-, 3-,
and 6-month columns as the validation fixtures.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixture"

_NOAA_URL = "https://psl.noaa.gov"
_NOAA_TIMESERIES_URL = f"{_NOAA_URL}/eddi/#timeseries"
_NOAA_REQUEST_URL = f"{_NOAA_URL}/cgi-bin/eddi_int/eddi.ts.pl"
_EDDI_SCALES = (1, 3, 6)
_NOAA_MISSING_VALUE = -9999.0
_EXPECTED_CALIBRATION_YEAR_FINAL = 2023
_REQUEST_DATA = {
    "xlat1": "39.75",
    "xlat2": "39.875",
    "xlon1": "-105.0",
    "xlon2": "-104.875",
    "lonPair": "",
    "monavg": "1",
    "monend": "0",
    "title": "climate_indices validation",
    "Submit": "Generate Data & Plot",
}


def _compute_checksum(directory: Path) -> str:
    """Compute the SHA-256 checksum of sorted .npy files in a directory."""
    hasher = hashlib.sha256()
    for npy_file in sorted(directory.glob("*.npy")):
        hasher.update(npy_file.read_bytes())
    return hasher.hexdigest()


def _validated_noaa_url(url: str) -> str:
    """Return a URL only when it has the NOAA PSL HTTPS origin."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "psl.noaa.gov":
        raise ValueError(f"NOAA URL must use https://psl.noaa.gov: {url}")
    return url


def _download_master_table() -> str:
    """Fetch the master EDDI table generated for the fixed spatial subset."""
    response = requests.post(
        _validated_noaa_url(_NOAA_REQUEST_URL),
        data=_REQUEST_DATA,
        timeout=60,
        allow_redirects=False,
    )
    response.raise_for_status()
    iframe = re.search(r'<iframe[^>]+src=["\']([^"\']+)', response.text, re.IGNORECASE)
    if iframe is None:
        raise ValueError("NOAA EDDI response did not contain a result iframe")

    iframe_url = _validated_noaa_url(urljoin(_NOAA_URL, iframe.group(1)))
    result = requests.get(iframe_url, timeout=60, allow_redirects=False)
    result.raise_for_status()
    table_link = re.search(r'href=["\']([^"\']*master\.table)["\']', result.text, re.IGNORECASE)
    if table_link is None:
        raise ValueError("NOAA EDDI result did not link a master.table file")

    table_url = _validated_noaa_url(urljoin(_NOAA_URL, table_link.group(1)))
    table = requests.get(table_url, timeout=60, allow_redirects=False)
    table.raise_for_status()
    return table.text


def _parse_master_table(table: str) -> tuple[np.ndarray, list[str], int, int]:
    """Parse and validate the NOAA master table."""
    lines = table.splitlines()
    if len(lines) < 3:
        raise ValueError("NOAA master table has no data rows")

    columns = lines[1].split()
    required_columns = ["yyyy", "mm", "rET", *(f"eddi{scale}mo" for scale in _EDDI_SCALES)]
    missing_columns = set(required_columns).difference(columns)
    if missing_columns:
        raise ValueError(f"NOAA master table is missing columns: {sorted(missing_columns)}")

    try:
        values = np.array([[float(value) for value in line.split()] for line in lines[2:]])
    except ValueError as exc:
        raise ValueError("NOAA master table contains non-numeric data") from exc
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError("NOAA master table rows do not match its header")

    years = values[:, columns.index("yyyy")].astype(int)
    months = values[:, columns.index("mm")].astype(int)
    if np.any((months < 1) | (months > 12)) or not np.all(np.diff(years * 12 + months) == 1):
        raise ValueError("NOAA master table does not contain contiguous monthly rows")

    ret = values[:, columns.index("rET")]
    if np.any(ret == _NOAA_MISSING_VALUE):
        raise ValueError("NOAA master table contains missing reference ET values")

    for scale in _EDDI_SCALES:
        reference = values[:, columns.index(f"eddi{scale}mo")]
        expected_missing = np.arange(values.shape[0]) < scale - 1
        if not np.array_equal(reference == _NOAA_MISSING_VALUE, expected_missing):
            raise ValueError(f"NOAA EDDI {scale}-month missing values are not the expected leading values")

    complete_years = [year for year in np.unique(years) if np.array_equal(months[years == year], np.arange(1, 13))]
    if not complete_years:
        raise ValueError("NOAA master table contains no complete calendar year")
    return values, columns, int(years[0]), int(complete_years[-1])


def _write_fixture(
    scale: int,
    ret: np.ndarray,
    reference: np.ndarray,
    data_start_year: int,
    calibration_year_final: int,
    download_date: str,
) -> None:
    """Write one EDDI scale's arrays and provenance metadata."""
    output_dir = FIXTURE_DIR / f"noaa-eddi-{scale}month"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "pet_input.npy", ret)
    np.save(output_dir / "eddi_reference.npy", reference)

    metadata = {
        "data_start_year": data_start_year,
        "calibration_year_initial": data_start_year,
        "calibration_year_final": calibration_year_final,
        "spatial_subset": {
            "latitude_bounds": [39.75, 39.875],
            "longitude_bounds": [-105.0, -104.875],
        },
        "source": "NOAA PSL EDDI time-series master table",
        "source_url": _NOAA_TIMESERIES_URL,
        "table_baseline": f"{data_start_year}-{calibration_year_final}",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    provenance = {
        "source": "NOAA Physical Sciences Laboratory (PSL)",
        "url": _NOAA_TIMESERIES_URL,
        "download_date": download_date,
        "subset_description": (
            f"EDDI {scale}-month and monthly reference ET from the NOAA PSL EDDI time-series "
            "master table for latitude 39.75 to 39.875 and longitude -105.0 to -104.875."
        ),
        "checksum_sha256": _compute_checksum(output_dir),
        "fixture_version": "1.0.0",
        "validation_tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "citation": (
            "Hobbins, M. T., A. Wood, D. McEvoy, J. Huntington, C. Morton, M. Anderson, "
            "and C. Hain (2016), The Evaporative Demand Drought Index. Part I."
        ),
        "doi": "10.1175/JHM-D-15-0121.1",
        "license": "Public domain (U.S. Government work)",
        "notes": (
            f"Retrieved through the NOAA PSL time-series tool using monavg=1 and monend=0; "
            f"the table baseline is {data_start_year}-{calibration_year_final}."
        ),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> int:
    """Download and save the NOAA EDDI reference fixtures."""
    values, columns, data_start_year, calibration_year_final = _parse_master_table(_download_master_table())
    if calibration_year_final != _EXPECTED_CALIBRATION_YEAR_FINAL:
        raise ValueError(
            f"NOAA table baseline ended in {calibration_year_final}; expected {_EXPECTED_CALIBRATION_YEAR_FINAL}. "
            "Review and intentionally update the committed fixtures before changing this value."
        )

    ret = values[:, columns.index("rET")]
    download_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    for scale in _EDDI_SCALES:
        reference = values[:, columns.index(f"eddi{scale}mo")].copy()
        reference[reference == _NOAA_MISSING_VALUE] = np.nan
        _write_fixture(scale, ret, reference, data_start_year, calibration_year_final, download_date)

    print(f"Wrote NOAA EDDI fixtures for {data_start_year}-{calibration_year_final}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

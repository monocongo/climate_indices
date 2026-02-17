#!/usr/bin/env python3
# /// pyproject
# [context]
# dependencies = [
#   "numpy",
#   "netCDF4",
#   "requests",
# ]
# ///
"""Prepare NOAA PSL EDDI reference fixtures for validation tests.

Downloads a small subset of NOAA EDDI data and prepares it as pytest fixtures
for validating the climate_indices EDDI implementation. This script must be
run manually when setting up the reference data for the first time.

Usage:
    uv run scripts/prepare_noaa_eddi_fixtures.py

The script will:
    1. Download EDDI reference NetCDF files from NOAA PSL
    2. Extract a small spatial/temporal subset suitable for git
    3. Save PET input and EDDI reference arrays as .npy files
    4. Create provenance.json and metadata.json for each scale
    5. Compute and record SHA-256 checksums

Output directories:
    tests/fixture/noaa-eddi-1month/
    tests/fixture/noaa-eddi-3month/
    tests/fixture/noaa-eddi-6month/

Source:
    https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

# project paths
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixture"

_NOAA_BASE_URL = "https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/data"
_EDDI_SCALES = [1, 3, 6]


def _compute_checksum(directory: Path) -> str:
    """Compute SHA-256 checksum for .npy files in a directory."""
    hasher = hashlib.sha256()
    for npy_file in sorted(directory.glob("*.npy")):
        hasher.update(npy_file.read_bytes())
    return hasher.hexdigest()


def _create_provenance(output_dir: Path, scale: int, checksum: str) -> None:
    """Create provenance.json for a fixture directory."""
    provenance = {
        "source": "NOAA Physical Sciences Laboratory (PSL)",
        "url": f"{_NOAA_BASE_URL}/",
        "download_date": datetime.date.today().isoformat(),
        "subset_description": (
            f"EDDI {scale}-month scale reference values for a single CONUS grid point, "
            f"extracted from NOAA PSL EDDI archive for library validation."
        ),
        "checksum_sha256": checksum,
        "fixture_version": "1.0.0",
        "validation_tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "citation": (
            "Hobbins, M. T., A. Wood, D. McEvoy, J. Huntington, C. Morton, "
            "M. Anderson, and C. Hain, 2016: The Evaporative Demand Drought "
            "Index. Part I. J. Hydrometeor., 17, 1745-1761."
        ),
        "doi": "10.1175/JHM-D-15-0121.1",
        "license": "Public domain (U.S. Government work)",
        "notes": (
            f"Single grid point extraction for EDDI {scale}-month validation. "
            "See metadata.json for calibration parameters used."
        ),
    }
    with (output_dir / "provenance.json").open("w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")


def main() -> int:
    """Download and prepare NOAA EDDI fixtures."""
    print("NOAA EDDI Fixture Preparation Script")
    print("=" * 40)
    print()
    print("This script downloads NOAA PSL EDDI reference data and prepares")
    print("validation fixtures for the climate_indices test suite.")
    print()
    print("Prerequisites:")
    print("  - Network access to downloads.psl.noaa.gov")
    print("  - Python packages: numpy, netCDF4, requests")
    print()

    # check for network access
    try:
        import requests

        resp = requests.head(f"{_NOAA_BASE_URL}/", timeout=10)
        if resp.status_code != 200:
            print(f"ERROR: Cannot reach NOAA PSL archive (HTTP {resp.status_code})")
            print("Check your network connection and proxy settings.")
            return 1
    except Exception as exc:
        print(f"ERROR: Cannot reach NOAA PSL archive: {exc}")
        print("Check your network connection and proxy settings.")
        return 1

    print("TODO: Implement NOAA data download and extraction.")
    print()
    print("When implementing, the script should:")
    print("  1. Download monthly EDDI NetCDF files for a small region")
    print("  2. Extract PET input values (if available) or create synthetic PET")
    print("  3. Extract NOAA-computed EDDI reference values")
    print("  4. Save as .npy files in tests/fixture/noaa-eddi-{scale}month/")
    print("  5. Create provenance.json and metadata.json")
    print()
    print("Note: The NOAA EDDI archive provides pre-computed EDDI grids but")
    print("does NOT include the PET input used to compute them. A paired")
    print("PET+EDDI dataset may need to be sourced separately or the")
    print("validation approach may need to use round-trip consistency.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

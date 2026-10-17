#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "xclim==0.62.0", "windpowerlib==0.2.2", "numpy==2.5.4",
#   "xarray==2026.9.0", "pandas==3.0.6", "pint==0.26.1", "cf-xarray==0.11.3",
# ]
# ///
"""Generate independent wind-profile references; never import climate_indices.

Run manually: uv run --python 3.12 scripts/prepare_energy_wind_fixtures.py
Tests consume committed JSON without either oracle installed. Synthetic inputs
exercise neutral-profile arithmetic, not prediction of observed hub-height wind.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import os
import platform
import shutil
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixture" / "energy_wind"
SOURCES = {
    "xclim": {
        "version": "0.62.0",
        "commit": "6328035cce3661409acae5a11d7f00d45a1bc82f",
        "source_sha256": "f76c786c19b142438a227ebe84dce3dbad7273f8515bcc07bedeca33afb19a89",
    },
    "windpowerlib": {
        "version": "0.2.2",
        "commit": "255be722e0300326389b80f84d62e3992dcabfe0",
        "source_sha256": "7fa8ea74036038ba78d4b21d32202fa951525e26d3ca98fde28815e93b1607b5",
    },
}


def _validate_result(result: np.ndarray, speed: np.ndarray, identity: bool) -> None:
    """Check oracle output even when Python disables assert statements."""
    np.testing.assert_equal(result.shape, speed.shape)
    np.testing.assert_equal(result[0], 0.0)
    np.testing.assert_equal(np.isnan(result), np.isnan(speed))
    np.testing.assert_equal(np.isfinite(result[:-1]), True)
    if identity:
        np.testing.assert_array_equal(result, speed)


def _publish(staging: Path) -> None:
    """Publish a complete fixture pair; restore the previous pair if the swap fails."""
    backup = FIXTURE_DIR.with_name(f".{FIXTURE_DIR.name}-backup")
    if backup.exists() and not FIXTURE_DIR.exists():
        os.replace(backup, FIXTURE_DIR)
    try:
        if FIXTURE_DIR.exists():
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(FIXTURE_DIR, backup)
        os.replace(staging, FIXTURE_DIR)
    except BaseException:
        if backup.exists():
            os.replace(backup, FIXTURE_DIR)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _verify_source(package: str, source: str | None) -> None:
    """Verify the pinned package version and source-file checksum."""
    if source is None or version(package) != SOURCES[package]["version"]:
        raise RuntimeError(f"Expected pinned {package} source/version")
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != SOURCES[package]["source_sha256"]:
        raise RuntimeError(f"Unexpected {package} source checksum")


def main() -> None:
    """Verify pinned sources, generate vectors, then checksum the canonical JSON."""
    from windpowerlib.wind_speed import logarithmic_profile
    from xclim.indices import wind_profile

    for package, function in (("xclim", wind_profile), ("windpowerlib", logarithmic_profile)):
        source = inspect.getsourcefile(inspect.unwrap(function))
        _verify_source(package, source)

    speed = np.array([0.0, 1.0, 3.5, 12.0, np.nan])
    cases = []
    for method, reference_height, hub_height, parameter in (
        ("power", 10.0, 100.0, 1 / 7),
        ("power", 10.0, 150.0, 0.2),
        ("power", 100.0, 10.0, 0.3),
        ("power", 10.0, 100.0, 0.0),
        ("power", 10.0, 10.0, 0.2),
        ("log", 10.0, 100.0, 0.03),
        ("log", 10.0, 150.0, 0.1),
        ("log", 100.0, 10.0, 1.0),
        ("log", 10.0, 10.0, 0.1),
    ):
        kwargs = {"reference_height": reference_height, "hub_height": hub_height, "method": method}
        if method == "power":
            kwargs["alpha"] = parameter
            result = wind_profile(
                xr.DataArray(speed, dims="sample", attrs={"units": "m s-1"}),
                h=f"{hub_height} m",
                h_r=f"{reference_height} m",
                method="power_law",
                alpha=parameter,
            ).values
        else:
            kwargs["roughness_length"] = parameter
            result = logarithmic_profile(speed, reference_height, hub_height, parameter, obstacle_height=0.0)
        _validate_result(result, speed, reference_height == hub_height or (method == "power" and parameter == 0))
        cases.append(
            {"kwargs": kwargs, "expected_m_s": [None if np.isnan(value) else float(value) for value in result]}
        )

    data = {
        "sources": SOURCES,
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("numpy", "xarray", "pandas", "pint", "cf-xarray")},
        },
        "wind_speed_m_s": [None if np.isnan(value) else float(value) for value in speed],
        "cases": cases,
    }
    payload = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    provenance = {
        "source": "xclim 0.62.0 and windpowerlib 0.2.2, synthetic inputs",
        "url": "https://github.com/Ouranosinc/xclim/tree/v0.62.0",
        "download_date": dt.date.today().isoformat(),
        "subset_description": "Nine power/log wind-profile cases; speeds 0, 1, 3.5, 12 m s-1 and missing; heights 10-150 m.",
        "checksum_sha256": hashlib.sha256(payload).hexdigest(),
        "fixture_version": "1.0.0",
        "validation_tolerance": {"rtol": 0.0, "atol": 1e-12},
        "citation": "xclim.indices.wind_profile (power_law); windpowerlib.wind_speed.logarithmic_profile (obstacle_height=0), https://github.com/wind-python/windpowerlib/tree/v0.2.2.",
        "license": "Synthetic inputs and generated numerical outputs; reference software: xclim Apache-2.0, windpowerlib MIT. No implementation source copied.",
        "notes": "Generated offline by scripts/prepare_energy_wind_fixtures.py. reference.json records source commits, verified full source-file SHA-256 hashes and numerical dependency versions. Heights and roughness are metres; output is m s-1. Independent-implementation comparisons, not observed-wind validation. atol=1e-12 permits double-precision cross-platform arithmetic only; calm, identity and NaN masks are exact. No climate_indices comparison has run: #1165 is unimplemented. Two-level log-ratio recovery is consistency evidence only, not an independent oracle. Maintainer confirmation of generation/redistribution is required in the fixture PR.",
    }
    FIXTURE_DIR.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=FIXTURE_DIR.parent, prefix=f".{FIXTURE_DIR.name}-") as temporary:
        staging = Path(temporary) / FIXTURE_DIR.name
        if FIXTURE_DIR.exists():
            shutil.copytree(FIXTURE_DIR, staging)
        else:
            staging.mkdir()
        (staging / "reference.json").write_bytes(payload)
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        _publish(staging)
    print(f"Wrote {len(cases)} cases; SHA-256 {provenance['checksum_sha256']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
# ]
# ///
"""Prepare R SEI/SCI zero-placement reference fixtures (issue #1209).

Runs ``scripts/zero_handling_reference.R`` against a deterministic zero-heavy
monthly precipitation series and commits the R packages' gamma
standardized-index output as the independent-implementation reference for
ADR-0015's ``"classic"``, ``"center_of_mass"``, and ``"mean_zero"`` modes.

This script must be run manually when refreshing the reference data:

    uv run scripts/prepare_zero_handling_fixtures.py

It requires an R installation with the pinned ``SEI`` (0.2.0) and ``SCI``
(1.0-3) packages. The committed fixtures, not this script, are what the test
suite loads, so CI does not need R.

The reference is deliberately not an exact end-to-end oracle: ``SEI`` fits the
gamma to the positive values by maximum likelihood (``fitdistrplus``) while
``climate_indices`` uses Thom's method-of-moments approximation, so only the
zero-placement constants are expected to match exactly. The committed
``sei_shape``/``sei_rate`` arrays let the tests feed ``SEI``'s fitted
parameters into the ``climate_indices`` transform to isolate the placement.

Source:
    Allen, S. & Otero, N. (2024). Calculating Standardised Indices Using SEI.
    The R Journal 16(4), 102-122. https://doi.org/10.32614/RJ-2024-038
    Gudmundsson, L. & Stagge, J. H. (2014). SCI: Standardized Climate Indices.
    https://cran.r-project.org/package=SCI (Stagge et al., 2015,
    https://doi.org/10.1002/joc.4267)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixture" / "zero_handling"
R_SCRIPT = PROJECT_ROOT / "scripts" / "zero_handling_reference.R"

_DATA_START_YEAR = 1981
_N_YEARS = 100

# exact zeros per calendar month over the record; the calendar-month zero
# fraction is what the gamma transform uses as p0. The set spans near-zero
# (month 7, p0 == 0), a mid range, near-one (month 11, p0 == 0.95), and one
# all-zero step (month 6, p0 == 1) that no continuous distribution can fit.
_ZERO_COUNTS = np.array([2, 50, 88, 10, 75, 100, 0, 25, 60, 20, 88, 40], dtype=int)

_GENERATOR_SEED = 20260914
_GAMMA_SHAPE = 2.0
_GAMMA_SCALE_MM = 20.0


def _generate_input() -> np.ndarray:
    """A deterministic (years, 12) monthly precipitation series in millimetres."""
    rng = np.random.default_rng(_GENERATOR_SEED)
    values = np.empty((_N_YEARS, 12), dtype=np.float64)
    for month, zero_count in enumerate(_ZERO_COUNTS):
        values[:zero_count, month] = 0.0
        values[zero_count:, month] = rng.gamma(_GAMMA_SHAPE, _GAMMA_SCALE_MM, size=_N_YEARS - zero_count)
    return values


def _write_input_csv(values: np.ndarray, path: Path) -> None:
    rows = ["year,month,value"]
    for year_index in range(values.shape[0]):
        for month in range(values.shape[1]):
            rows.append(f"{_DATA_START_YEAR + year_index},{month + 1},{float(values[year_index, month]):.17g}")
    path.write_text("\n".join(rows) + "\n")


def _read_matrix(path: Path) -> np.ndarray:
    """Read an M1..M12 column CSV written by the R script into a (years, 12) array."""
    header = path.read_text().splitlines()[0].split(",")
    if header != [f"M{month}" for month in range(1, 13)]:
        raise ValueError(f"{path.name} has unexpected columns: {header}")
    matrix = np.loadtxt(path, delimiter=",", skiprows=1).reshape(_N_YEARS, 12)
    return np.asarray(matrix, dtype=np.float64)


def _compute_checksum(directory: Path) -> str:
    """SHA-256 over the sorted .npy fixture contents, per tests/fixture/README.md."""
    hasher = hashlib.sha256()
    for npy_file in sorted(directory.glob("*.npy")):
        hasher.update(npy_file.read_bytes())
    return hasher.hexdigest()


def main() -> int:
    if not R_SCRIPT.exists():
        raise FileNotFoundError(R_SCRIPT)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    input_values = _generate_input()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_csv = tmp_path / "input.csv"
        _write_input_csv(input_values, input_csv)
        subprocess.run(
            ["Rscript", str(R_SCRIPT), str(input_csv), str(tmp_path)],
            check=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "R_LIBS_USER": os.environ.get("R_LIBS_USER", str(Path.home() / ".R" / "library"))},
        )

        np.save(FIXTURE_DIR / "input_precipitation_mm.npy", input_values)
        np.save(FIXTURE_DIR / "sei_classic.npy", _read_matrix(tmp_path / "sei_none.csv"))
        np.save(FIXTURE_DIR / "sei_center_of_mass.npy", _read_matrix(tmp_path / "sei_prob.csv"))
        np.save(FIXTURE_DIR / "sei_mean_zero.npy", _read_matrix(tmp_path / "sei_normal.csv"))

        params = np.genfromtxt(tmp_path / "sei_params.csv", delimiter=",", names=True)
        np.save(FIXTURE_DIR / "sei_shape.npy", np.asarray(params["shape"], dtype=np.float64))
        np.save(FIXTURE_DIR / "sei_rate.npy", np.asarray(params["rate"], dtype=np.float64))

        sci = np.genfromtxt(tmp_path / "sci_p0_center_mass.csv", delimiter=",", names=True)
        np.save(
            FIXTURE_DIR / "sci_center_of_mass_probability.npy",
            np.asarray(sci["p0_center_mass"], dtype=np.float64),
        )

        versions_table = np.genfromtxt(
            tmp_path / "r_versions.csv", delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        versions = {str(row["name"]): str(row["version"]) for row in np.atleast_1d(versions_table)}

    provenance = {
        "source": "R SEI and SCI packages",
        "url": "https://cran.r-project.org/package=SEI",
        "download_date": dt.date.today().isoformat(),
        "subset_description": (
            f"Deterministic {_N_YEARS}-year zero-heavy monthly precipitation series and the R "
            "SEI (0.2.0) and SCI (1.0-3) gamma standardized-index reference for the ADR-0015 "
            "zero-placement modes."
        ),
        "checksum_sha256": _compute_checksum(FIXTURE_DIR),
        "fixture_version": "1.0.0",
        "validation_tolerance": {
            "zero_constant_atol": 1e-12,
            "parameter_matched_atol": 1e-6,
            "end_to_end_atol": 0.01,
            "sci_center_of_mass_atol": 0.2,
        },
        "citation": (
            "Allen, S. & Otero, N. (2024). Calculating Standardised Indices Using SEI. "
            "The R Journal 16(4), 102-122. https://doi.org/10.32614/RJ-2024-038"
        ),
        "doi": "10.32614/RJ-2024-038",
        "license": "R packages under their CRAN licenses (GPL-3)",
        "notes": (
            "Generated by scripts/prepare_zero_handling_fixtures.py via "
            "scripts/zero_handling_reference.R. "
            f"R {versions.get('R')}, SEI {versions.get('SEI')}, SCI {versions.get('SCI')}. "
            "SEI fits gamma by MLE (fitdistrplus) while climate_indices uses Thom's "
            "method-of-moments approximation, so the full series is compared against "
            "SEI's fitted shape/rate; only the zero-placement constants are exact. "
            "Month 6 is all zeros: SEI still assigns its censored constant while "
            "climate_indices treats the undefined mass as classic (ADR-0015 decision 5). "
            "SCI estimates p0 with the Weibull plotting position np/(n+1) and places "
            "zeros at (np+1)/(2(n+1)), so its centre-of-mass constant differs from "
            "climate_indices' np/(2n) by up to ~0.15 in z for p0 near 0; the "
            "sci_center_of_mass_atol records that estimator difference."
        ),
    }
    (FIXTURE_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {len(list(FIXTURE_DIR.glob('*.npy')))} arrays to {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

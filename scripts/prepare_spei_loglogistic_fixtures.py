#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
# ]
# ///
"""Prepare R SPEI log-logistic reference fixtures (issue #1195).

Runs ``scripts/spei_loglogistic_reference.R`` against the committed nClimDiv
inputs of three CONUS climate divisions and against a short, highly skewed
synthetic series, and commits the R ``SPEI`` package's log-logistic
standardized index (``distribution = "log-Logistic"``, ``fit = "ub-pwm"``) as
the independent-implementation reference for this library's
``indices.Distribution.loglogistic``.

This script must be run manually when refreshing the reference data:

    uv run scripts/prepare_spei_loglogistic_fixtures.py

It requires an R installation with the pinned ``SPEI`` (1.8.1) package. The
committed fixtures, not this script, are what the test suite loads, so CI does
not need R.

Both sides fit the identical scaled series: the script passes the R script the
``(P − PET) + 1000`` water balance -- the same ``+1000`` offset ``indices.spei``
applies internally -- so the comparison isolates the distribution fit from the
scaling and offsetting steps. Because the GLO is a location family, the offset
cancels in the standardized values; only the ``loc`` parameter carries it.

``SPEI`` fits each calendar step with unbiased-PWM L-moments
(``TLMoments::PWM`` + ``lmom::pelglo``), the same estimator this library's
``lmoments.fit_glo`` implements, so the fitted parameters and the standardized
series agree to machine precision rather than to a loose tolerance.

Source:
    Vicente-Serrano, S. M., Beguería, S. & López-Moreno, J. I. (2010).
    J. Climate 23, 1696-1718. https://doi.org/10.1175/2009JCLI2909.1
    Beguería, S., et al. (2014). SPEI revisited. Int. J. Climatol. 34,
    3001-3023. https://doi.org/10.1002/joc.3887
    https://cran.r-project.org/package=SPEI
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixture" / "spei_loglogistic"
R_SCRIPT = PROJECT_ROOT / "scripts" / "spei_loglogistic_reference.R"
PALMER_ROOT = PROJECT_ROOT / "tests" / "fixture" / "palmer"
DIVISIONS_JSON = PROJECT_ROOT / "tests" / "fixture" / "speibase" / "divisions.json"

# R reports these through packageVersion(); the committed fixtures assume them
_PINNED_R_PACKAGES = {
    "SPEI": "1.8.1",
    "TLMoments": "0.7.5.3",  # NOSONAR (S1313) a package version, not an IP address
    "lmom": "3.3",
}

_SCALES = (1, 3, 6, 12)
_SYNTHETIC_SCALES = (1, 6)

_PALMER_INPUT_START_YEAR = 1895  # tests/fixture/palmer/<division>/ inputs start here
_DATA_START_YEAR = 1901  # SPEIbase v2.11 starts here; matches tests/test_speibase_reference.py
_DATA_END_YEAR = 2022
_N_MONTHS = (_DATA_END_YEAR - _DATA_START_YEAR + 1) * 12
_INPUT_OFFSET = (_DATA_START_YEAR - _PALMER_INPUT_START_YEAR) * 12

# the +1000 mm offset indices.spei adds to the (P - PET) water balance before
# scaling; passing it to SPEI keeps the two implementations fitting the same series
_WATER_BALANCE_OFFSET_MM = 1000.0

# deterministic, highly skewed (gamma shape 0.5) short series with one constant
# calendar month, so the fixtures cover an extreme-skew fit, a short sample, and
# a calendar step that neither implementation can fit
_SYNTHETIC_SEED = 20260929
_SYNTHETIC_START_YEAR = 1980
_SYNTHETIC_YEARS = 12
_SYNTHETIC_GAMMA_SHAPE = 0.5
_SYNTHETIC_GAMMA_SCALE_MM = 60.0
_SYNTHETIC_SHIFT_MM = -20.0
_SYNTHETIC_CONSTANT_MONTH = 3  # 1-based; water balance forced to 5.0 for this month
_SYNTHETIC_CONSTANT_VALUE_MM = 5.0
_SYNTHETIC_PET_MM = 100.0


def _load_temps_fahrenheit(division_id: str) -> np.ndarray:
    values = np.load(PALMER_ROOT / division_id / "temps.npy", allow_pickle=True)
    return np.array([float(str(value).split()[0]) for value in values], dtype=float)


def _environmental_inputs(division: dict) -> tuple[np.ndarray, np.ndarray]:
    """Precipitation (mm) and Thornthwaite PET (mm) for 1901-2022."""
    from climate_indices import eto

    precip_inches = np.load(PALMER_ROOT / division["id"] / "precips.npy").astype(np.float64)[
        _INPUT_OFFSET : _INPUT_OFFSET + _N_MONTHS
    ]
    precip_mm = precip_inches * 25.4  # committed nClimDiv precipitation is inches
    temps_c = (_load_temps_fahrenheit(division["id"])[_INPUT_OFFSET : _INPUT_OFFSET + _N_MONTHS] - 32.0) * (5.0 / 9.0)
    pet_mm = eto.eto_thornthwaite(temps_c, division["latitude"], _DATA_START_YEAR)
    return precip_mm, pet_mm


def _synthetic_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Precipitation (mm) and constant PET (mm) for the synthetic edge series."""
    rng = np.random.default_rng(_SYNTHETIC_SEED)
    water_balance = (
        rng.gamma(_SYNTHETIC_GAMMA_SHAPE, _SYNTHETIC_GAMMA_SCALE_MM, _SYNTHETIC_YEARS * 12) + _SYNTHETIC_SHIFT_MM
    )
    water_balance[_SYNTHETIC_CONSTANT_MONTH - 1 :: 12] = _SYNTHETIC_CONSTANT_VALUE_MM
    pet_mm = np.full_like(water_balance, _SYNTHETIC_PET_MM)
    return water_balance + _SYNTHETIC_PET_MM, pet_mm


def _write_input_csv(series: dict[str, np.ndarray], start_year: int, path: Path) -> None:
    """Write a wide (year, month, <series>) water-balance CSV with the +1000 offset."""
    length = len(next(iter(series.values())))
    years = np.repeat(np.arange(start_year, start_year + length // 12), 12)
    months = np.tile(np.arange(1, 13), length // 12)
    names = list(series)
    rows = [",".join(("year", "month", *names))]
    for index, (year, month) in enumerate(zip(years, months, strict=True)):
        values = ",".join(f"{series[name][index]:.17g}" for name in names)
        rows.append(f"{year},{month},{values}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_fitted(path: Path, name: str, length: int, series_names: list[str]) -> np.ndarray:
    # pin every column's dtype: an all-NA column would otherwise be inferred as
    # boolean and np.genfromtxt would coerce its filling value to True (1.0)
    dtype = [("year", "i8"), ("month", "i8"), *((series_name, "f8") for series_name in series_names)]
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=dtype, missing_values="NA", filling_values=np.nan)
    values = np.asarray(data[name], dtype=np.float64)
    if values.shape != (length,):
        raise ValueError(f"{path.name}:{name} has shape {values.shape}, expected ({length},)")
    return values


def _read_params(path: Path, name: str) -> np.ndarray:
    """Return the (3, 12) loc/scale/shape matrix for one series."""
    dtype = [("series", "U32"), ("month", "i8"), ("loc", "f8"), ("scale", "f8"), ("shape", "f8")]
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=dtype, missing_values="NA", filling_values=np.nan)
    rows = data[data["series"] == name]
    if rows.shape != (12,):
        raise ValueError(f"{path.name}:{name} has {rows.shape[0]} rows, expected 12")
    return np.stack([np.asarray(rows[field], dtype=np.float64) for field in ("loc", "scale", "shape")])


def _compute_checksum(directory: Path) -> str:
    """SHA-256 over the sorted .npy fixture contents, per tests/fixture/README.md."""
    hasher = hashlib.sha256()
    for npy_file in sorted(directory.glob("*.npy")):
        hasher.update(npy_file.read_bytes())
    return hasher.hexdigest()


def _check_versions(output_dir: Path) -> dict[str, str]:
    """Refuse a refresh from unpinned package versions before any fixture is written."""
    table = np.genfromtxt(output_dir / "r_versions.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    versions = {str(row["name"]): str(row["version"]) for row in np.atleast_1d(table)}
    mismatched = {
        name: versions.get(name) for name, pinned in _PINNED_R_PACKAGES.items() if versions.get(name) != pinned
    }
    if mismatched:
        raise SystemExit(f"R packages differ from the pinned {_PINNED_R_PACKAGES}: found {mismatched}")
    return versions


def _publish(staging: Path) -> None:
    """Replace the published fixture directory with the staged one, as one transaction.

    Arrays and metadata must always come from a single generation: replacing the
    files individually leaves a mixed set behind when a later write fails. The
    previous directory is kept as a backup until the swap succeeds, and restored
    if it does not.
    """
    backup = FIXTURE_DIR.with_name(f".{FIXTURE_DIR.name}-backup")
    shutil.rmtree(backup, ignore_errors=True)
    if FIXTURE_DIR.exists():
        os.replace(FIXTURE_DIR, backup)
    try:
        os.replace(staging, FIXTURE_DIR)
    except BaseException:
        if backup.exists():
            os.replace(backup, FIXTURE_DIR)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def main() -> int:
    if not R_SCRIPT.exists():
        raise FileNotFoundError(R_SCRIPT)
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript not found; install R with the pinned SPEI 1.8.1 package")

    divisions = json.loads(DIVISIONS_JSON.read_text(encoding="utf-8"))
    division_series: dict[str, np.ndarray] = {}
    for division in divisions:
        precip_mm, pet_mm = _environmental_inputs(division)
        # mirror indices.spei's negative-precipitation clip before the water balance
        precip_mm = np.clip(precip_mm, 0.0, None)
        division_series[f"d{division['id']}"] = (precip_mm - pet_mm) + _WATER_BALANCE_OFFSET_MM
    synthetic_precip, synthetic_pet = _synthetic_inputs()
    synthetic_series = {"synthetic": (synthetic_precip - synthetic_pet) + _WATER_BALANCE_OFFSET_MM}

    division_names = list(division_series)
    staging = Path(tempfile.mkdtemp(prefix=f".{FIXTURE_DIR.name}-staging-", dir=FIXTURE_DIR.parent))

    # keep R's own user-library default unless this session installed the
    # packages into the conventional ~/.R/library instead
    env = dict(os.environ)
    if "R_LIBS_USER" not in env and (Path.home() / ".R" / "library").is_dir():
        env["R_LIBS_USER"] = str(Path.home() / ".R" / "library")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            versions: dict[str, str] = {}
            for label, series, start_year, scales, scale_out in (
                ("real", division_series, _DATA_START_YEAR, _SCALES, _SCALES),
                ("synthetic", synthetic_series, _SYNTHETIC_START_YEAR, _SYNTHETIC_SCALES, _SYNTHETIC_SCALES),
            ):
                output_dir = tmp_path / label
                input_csv = tmp_path / f"{label}.csv"
                _write_input_csv(series, start_year, input_csv)
                subprocess.run(
                    ["Rscript", str(R_SCRIPT), str(input_csv), str(output_dir), *(str(scale) for scale in scales)],
                    check=True,
                    cwd=PROJECT_ROOT,
                    env=env,
                )
                checked = _check_versions(output_dir)
                versions = versions or checked

                if label == "real":
                    for scale in scale_out:
                        fitted = np.stack(
                            [
                                _read_fitted(output_dir / f"fitted_{scale:02d}.csv", name, _N_MONTHS, division_names)
                                for name in series
                            ]
                        )
                        params = np.stack(
                            [_read_params(output_dir / f"params_{scale:02d}.csv", name) for name in series]
                        )
                        np.save(staging / f"r_spei_{scale:02d}.npy", fitted)
                        np.save(staging / f"r_params_{scale:02d}.npy", params)
                else:
                    for scale in scale_out:
                        np.save(
                            staging / f"r_synthetic_spei_{scale:02d}.npy",
                            _read_fitted(
                                output_dir / f"fitted_{scale:02d}.csv",
                                "synthetic",
                                _SYNTHETIC_YEARS * 12,
                                ["synthetic"],
                            ),
                        )
                        np.save(
                            staging / f"r_synthetic_params_{scale:02d}.npy",
                            _read_params(output_dir / f"params_{scale:02d}.csv", "synthetic"),
                        )

            np.save(staging / "synthetic_precip_mm.npy", synthetic_precip)
            np.save(staging / "synthetic_pet_mm.npy", synthetic_pet)

        provenance = {
            "source": "R SPEI package (CRAN)",
            "url": "https://cran.r-project.org/package=SPEI",
            "download_date": dt.date.today().isoformat(),
            "subset_description": (
                "Standardized log-logistic SPEI from the R SPEI package for three CONUS climate "
                "divisions (nClimDiv precip/temps, Thornthwaite PET, 1901-2022) at timescales "
                "1/3/6/12, plus a short highly skewed synthetic series at timescales 1/6."
            ),
            "checksum_sha256": _compute_checksum(staging),
            "fixture_version": "1.0.0",
            "validation_tolerance": {
                "series_atol": 1e-6,
                "parameter_atol": 1e-6,
                "parameter_rtol": 1e-8,
            },
            "citation": (
                "Vicente-Serrano, S. M., Beguería, S. & López-Moreno, J. I. (2010). J. Climate 23, "
                "1696-1718; Beguería, S., et al. (2014). Int. J. Climatol. 34, 3001-3023; "
                "https://cran.r-project.org/package=SPEI"
            ),
            "doi": "10.1175/2009JCLI2909.1",
            "license": "R packages under their CRAN licenses (GPL-3)",
            "notes": (
                "Generated by scripts/prepare_spei_loglogistic_fixtures.py via "
                "scripts/spei_loglogistic_reference.R. "
                f"R {versions.get('R')}, SPEI {versions.get('SPEI')}, TLMoments {versions.get('TLMoments')}, "
                f"lmom {versions.get('lmom')}. SPEI fits each calendar step with unbiased-PWM L-moments "
                "(TLMoments::PWM + lmom::pelglo), the estimator lmoments.fit_glo implements, so the fitted "
                "parameters and the standardized series agree to machine precision (series ~1e-12, "
                "parameters <=~1e-10); the recorded tolerances absorb only cross-platform numerical "
                "wobble. The R input is the (P - PET) + 1000 water balance indices.spei builds "
                "internally, so both sides fit the identical scaled series; the offset cancels in the "
                "standardized values. SPEI does not clip to the [-3.09, 3.09] range this library enforces, "
                "so the comparison covers finite values inside that range and asserts this library's clip "
                "on the out-of-range tail separately. The synthetic series has a constant calendar month "
                "that is unfittable at scale 1 (both implementations report it missing: SPEI as NA, this "
                "library as a zeroed scale -> NaN); at scale 6 the constant month enters non-constant "
                "multi-month sums and is fittable."
            ),
        }
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        _publish(staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"wrote {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

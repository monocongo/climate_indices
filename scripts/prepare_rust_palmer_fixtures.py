#!/usr/bin/env python3
"""Generate standalone Rust Palmer fixtures from the pure-Python implementations.

Run from the repository root with PYTHONPATH=src. This optional maintainer tool
requires the existing Python scientific dependencies, but not the Rust extension.
Cargo only reads the checked-in text fixture and never runs this script.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "crates/climate-indices/tests/fixtures/palmer_reference.txt"
SOURCES = (
    "palmer.py",
    "_palmer_common.py",
    "_palmer_duration.py",
    "_palmer_pdi.py",
    "_palmer_wells.py",
    "self_calibration.py",
    "_calibration_period.py",
    "utils.py",
    "compute.py",
)

# The Python revision the checked-in values were generated from. A change to any
# listed source file is a regeneration decision: review it, then update this.
SOURCE_REVISION = "f4cb938eb622a3ddf8b96c129337e0ebecd4121e"


# Block extension imports before importing any climate_indices module.
@patch.dict(sys.modules, {"climate_indices._native": None})
def generate() -> None:
    """Write deterministic golden values without importing the native extension."""
    # The guard must precede the imports: a changed checkout executes them on import.
    paths = [f"src/climate_indices/{name}" for name in SOURCES]
    subprocess.run(
        ["git", "diff", "--exit-code", SOURCE_REVISION, "--", *paths], cwd=ROOT, check=True, capture_output=True
    )

    from climate_indices import compute, palmer, self_calibration
    from climate_indices._palmer_duration import DurationFactors
    from climate_indices._palmer_pdi import PdiDurationFactors

    assert palmer._native is None and compute._native is None
    assert Path(palmer.__file__).resolve() == ROOT / "src/climate_indices/palmer.py"
    revision = SOURCE_REVISION
    sections: list[str] = []

    def save(name: str, values: np.ndarray, units: str) -> None:
        array = np.asarray(values, dtype=np.float64)
        rows = array.reshape(-1, array.shape[-1] if array.ndim > 1 else 1)
        text = "\n".join(" ".join(format(value, ".17g") for value in row) for row in rows)
        # Seventeen significant digits must preserve every stored value and NaN.
        restored = np.fromstring(text, sep=" ").reshape(array.shape)
        np.testing.assert_array_equal(restored, array)
        sections.append(f"# shape={array.shape}; units={units}; C-order (last axis varies fastest)\n[{name}]\n{text}\n")

    time: np.ndarray = np.arange(96, dtype=float)[:, None]
    cell: np.ndarray = np.arange(4, dtype=float)[None, :]
    # Rational arithmetic gives repeatable wet/dry regimes without random or trig inputs.
    precips = 0.15 + ((time * 7 + cell * 11) % 29) / 7
    precips[(time[:, 0] // 12) % 3 == 1] *= 0.2
    pet = 0.3 + ((time * 3 + cell * 5) % 19) / 8
    precips[90, 0] = np.nan  # Outside the Calibration Period.
    precips[:, 3] = np.nan  # A fully missing cell beside finite cells.
    awc = np.array([4.0, 0.5, 0.0, 6.0])
    prepared = palmer._initialize_prepared(
        precips.reshape(96, 1, 4), pet.reshape(96, 1, 4), awc, 2000, 2001, 2006, spatial_time_major=True
    )
    with np.errstate(all="ignore"):
        palmer._calc_water_balances(prepared)
        palmer._calc_cafec_coefficients(prepared)
        palmer._calc_zindex_factors(prepared)
        dbar, k_prime = palmer._calc_k_prime_and_dbar(prepared)
        palmer._calc_kfactors(prepared)
        standard_z = palmer._calc_raw_zindex(prepared)
        standard_ak = prepared.ak.copy()
        prepared.ak = k_prime
        unnormalized_z = palmer._calc_raw_zindex(prepared)
    save("water.awc", awc, "inches")
    for name in ("precips", "pet", *palmer._WATER_BALANCE_MONTHLY, *palmer._WATER_BALANCE_SUMS):
        save(f"water.{name}", getattr(prepared, name), "inches")
    for name in ("alpha", "beta", "gamma", "delta", "trat"):
        save(f"cafec.{name}", getattr(prepared, name), "dimensionless")
    save("cafec.dbar", dbar, "inches")
    save("cafec.k_prime", k_prime, "Palmer weighting factor (inch-based convention)")
    save("cafec.ak", standard_ak, "Palmer weighting factor (inch-based convention)")
    save("cafec.standard_z", standard_z, "dimensionless Palmer Z-Index")
    save("cafec.unnormalized_z", unnormalized_z, "dimensionless unnormalized Palmer Z-Index")

    # Three cells: repeated wet/dry transitions, a different history, and all NaN.
    time = np.arange(120, dtype=float)
    pattern = np.array([0.0, 0.0, 4.0, 3.0, 2.0, -0.2, -1.0, -4.0, -3.0, -2.0, 0.2, 1.0])
    prolonged_spells = np.where((time // 24) % 2 == 0, 2.0, -2.0) + ((time * 7) % 13 - 6) / 4
    z = np.column_stack((np.tile(pattern, 10), prolonged_spells, np.full(120, np.nan)))
    z[[15, 16, 61, 97], 0] = np.nan
    z[[0, 35, 72, 118], 1] = np.nan
    save("recursion.z", z, "dimensionless Palmer Z-Index; axes=(month, cell)")
    for label, factors in (
        ("default", DurationFactors.from_defaults()),
        ("asymmetric", DurationFactors.from_fitted(0.3, 2.6, 0.4, 2.5)),
    ):
        for name, value in asdict(factors).items():
            unit = "dimensionless (one-month recurrence)"
            save(f"factors.{label}.{name}", np.array([value]), unit)
        pdi = palmer._pdi_recursion(
            z.reshape(10, 12, 3), PdiDurationFactors.from_fitted(factors.wetm, factors.wetb, factors.drym, factors.dryb)
        )
        wells = [palmer._wells_recursion(z[:, i], factors) for i in range(3)]
        for index, name in enumerate(("pdsi", "phdi", "pmdi")):
            save(f"pdi.{label}.{name}", getattr(pdi, name).reshape(120, 3), "dimensionless; axes=(month, cell)")
            save(
                f"wells.{label}.{name}",
                np.column_stack([result[index] for result in wells]),
                "dimensionless; axes=(month, cell)",
            )

    for i in range(2):
        for label, sign in (("wet", self_calibration.WET_SIGN), ("dry", self_calibration.DRY_SIGN)):
            save(
                f"duration.{i}.{label}",
                np.array(self_calibration.duration_factors(z[:, i], sign)),
                "m: Z-Index / (severity * month); b: Z-Index / severity",
            )

    samples = np.array(
        [
            [5.0, np.nan, 1.0, 3.0, 2.0, 4.0, -2.0, 0.0, 5.0],
            [np.nan] * 9,
            [7.0] + [np.nan] * 8,
        ]
    )
    fractions = np.array([0.0, 0.02, 0.1, 0.5, 0.98, 1.0])
    save("percentile.samples", samples, "dimensionless synthetic values; axes=(case, sample)")
    save("percentile.fractions", fractions, "dimensionless fraction, not percent")
    save(
        "percentile.expected",
        np.array(
            [[self_calibration.nan_safe_percentile(sample, fraction) for fraction in fractions] for sample in samples]
        ),
        "dimensionless order statistic; axes=(case, fraction)",
    )

    header = f"""# Palmer standalone Rust regression fixtures.
# Classification: Python regression, NOT external or independent scientific validation.
# Source repository: https://github.com/monocongo/climate_indices
# Exact Python source revision: {revision}
# Source files: {", ".join(SOURCES)} (under src/climate_indices/).
# The generator refuses modifications to these source files relative to the recorded revision.
# Generator: scripts/prepare_rust_palmer_fixtures.py
# Generation command (repository root):
# PYTHONPATH=src uv run python scripts/prepare_rust_palmer_fixtures.py
# An existing local Python environment with project dependencies also works.
# Generation environment: Python {platform.python_version()}, NumPy {np.__version__}.
# The generator blocks climate_indices._native before package imports.
# No Rust output is used. Cargo never invokes Python or the generator.
#
# Contract: abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected).
# NaN masks match exactly; infinities match exactly, including their signs.
# Recursion output signs and exact-zero selections also match exactly.
# Numbers use 17 significant digits. Whitespace separates values; # starts comments.
# Each [section] is a C-order array with the shape and units stated above it.
#
# water.*: _initialize_prepared + _calc_water_balances; (8, 12, 4) monthly arrays,
# (12, 4) calibration sums; January 2000 through December 2007.
# Calibration Period: January 2001 through December 2006 (inclusive year rows 1..6).
# AWC cells: 4, 0.5, 0, 6 inches. Cell 3 precipitation is all missing.
# Cell 0 precipitation is missing in July 2007, outside the Calibration Period.
# The top layer remains one inch even for AWC below one inch (Python behavior).
# cafec.*: _calc_cafec_coefficients, _calc_zindex_factors, _calc_k_prime_and_dbar,
# _calc_kfactors, _calc_raw_zindex. standard_z uses normalized ak;
# unnormalized_z uses raw K-prime. The water-balance golden values supply operands.
# All water depths and monthly fluxes use inches, never millimeters.
#
# recursion.z: 120 months x 3 independent cells, with zeros, reversals, isolated
# and consecutive gaps, and a fully missing cell. Both recursions use the same Z.
# Cell 0 alternates rapidly; cell 1 has 24-month wet/dry regimes with rational noise.
# pdi.*: _palmer_pdi.calculate through _pdi_recursion, flattened from (10, 12, 3).
# wells.*: _palmer_wells.calculate through _wells_recursion, once per cell.
# factors.*: _palmer_duration.DurationFactors default and asymmetric coefficients.
# PDI uses b/(m+b); Wells uses the stored Python-derived complement coefficients.
# duration.*: self_calibration.duration_factors on each finite recursion.z cell,
# using all 120 periods, omitting NaN periods. Each pair is (m, b), wet and dry.
# percentile.*: self_calibration.nan_safe_percentile, truncated ranks, not interpolated.
# Padded NaN samples also represent all-missing and one-value cases.
"""
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(header + "\n" + "\n".join(sections), encoding="utf-8")
    print(f"Wrote {DESTINATION.relative_to(ROOT)} from Python revision {revision}")


if __name__ == "__main__":
    generate()

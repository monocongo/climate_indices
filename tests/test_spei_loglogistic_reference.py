"""Independent-implementation tests for log-logistic SPEI against R's SPEI package.

These tests compare ``indices.spei()`` with ``Distribution.loglogistic`` against
the R ``SPEI`` package's ``distribution = "log-Logistic"``, ``fit = "ub-pwm"``
output committed under ``tests/fixture/spei_loglogistic/`` (see that
directory's ``provenance.json``). The R package is the reference implementation
of the log-logistic SPEI and the distribution the CSIC SPEIbase is built from
(Vicente-Serrano et al., 2010; Beguería et al., 2014; issue #1195).

Both sides fit the identical scaled series: the fixture's R input is the
``(P − PET) + 1000`` water balance ``indices.spei`` builds internally, so the
comparison isolates the distribution fit. ``SPEI`` fits each calendar step with
unbiased-PWM L-moments (``TLMoments::PWM`` + ``lmom::pelglo``), the estimator
``lmoments.fit_glo`` implements, so the fitted parameters and the standardized
series agree to machine precision; the small tolerances in the fixture's
``provenance.json`` absorb only cross-platform numerical wobble.

The committed real-data fixtures cover three CONUS climate divisions spanning an
aridity gradient (humid Alabama 0101, subhumid Oklahoma 3405, arid southwest
Arizona 0205) at timescales 1/3/6/12, using the committed nClimDiv inputs and
Thornthwaite PET. The synthetic fixtures cover a short, highly skewed series
with one constant calendar month, so the comparison includes an extreme-skew
fit and a calendar step neither implementation can fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from climate_indices import compute, eto, indices

_FIXTURE_ROOT = Path(__file__).parent / "fixture"
_SPEI_ROOT = _FIXTURE_ROOT / "spei_loglogistic"
_PALMER_ROOT = _FIXTURE_ROOT / "palmer"
_DIVISIONS = json.loads((_FIXTURE_ROOT / "speibase" / "divisions.json").read_text(encoding="utf-8"))
_PROVENANCE = json.loads((_SPEI_ROOT / "provenance.json").read_text(encoding="utf-8"))

_PALMER_INPUT_START_YEAR = 1895
_DATA_START_YEAR = 1901
_DATA_END_YEAR = 2022
_N_MONTHS = (_DATA_END_YEAR - _DATA_START_YEAR + 1) * 12
_INPUT_OFFSET = (_DATA_START_YEAR - _PALMER_INPUT_START_YEAR) * 12
_REAL_SCALES = (1, 3, 6, 12)

_SYNTHETIC_START_YEAR = 1980
_SYNTHETIC_YEARS = 12
_SYNTHETIC_SCALES = (1, 6)
_SYNTHETIC_CONSTANT_MONTH = 3  # 1-based

# tolerances recorded by scripts/prepare_spei_loglogistic_fixtures.py
_SERIES_ATOL = _PROVENANCE["validation_tolerance"]["series_atol"]
_PARAMETER_ATOL = _PROVENANCE["validation_tolerance"]["parameter_atol"]
_PARAMETER_RTOL = _PROVENANCE["validation_tolerance"]["parameter_rtol"]

# SPEI does not clip to the range indices.spei enforces; comparisons exclude
# values at or beyond the clip so the clip contract is asserted separately.
_CLIP = indices._FITTED_INDEX_VALID_MAX


@pytest.fixture(scope="module", autouse=True)
def _verify_fixture_checksum():
    """Reject a mixed or edited fixture generation before any assertion reads it."""
    hasher = hashlib.sha256()
    for npy_file in sorted(_SPEI_ROOT.glob("*.npy")):
        hasher.update(npy_file.read_bytes())
    assert hasher.hexdigest() == _PROVENANCE["checksum_sha256"], (
        "tests/fixture/spei_loglogistic holds a mixed or edited fixture generation "
        f"(checksum {hasher.hexdigest()} != provenance {_PROVENANCE['checksum_sha256']}); "
        "rerun scripts/prepare_spei_loglogistic_fixtures.py"
    )


def _load_temps_fahrenheit(division_id: str) -> np.ndarray:
    values = np.load(_PALMER_ROOT / division_id / "temps.npy", allow_pickle=True)
    return np.array([float(str(value).split()[0]) for value in values], dtype=float)


def _environmental_inputs(division: dict) -> tuple[np.ndarray, np.ndarray]:
    """Precipitation (mm) and Thornthwaite PET (mm) for 1901-2022."""
    precip_inches = np.load(_PALMER_ROOT / division["id"] / "precips.npy").astype(np.float64)[
        _INPUT_OFFSET : _INPUT_OFFSET + _N_MONTHS
    ]
    precip_mm = precip_inches * 25.4
    temps_c = (_load_temps_fahrenheit(division["id"])[_INPUT_OFFSET : _INPUT_OFFSET + _N_MONTHS] - 32.0) * (5.0 / 9.0)
    pet_mm = eto.eto_thornthwaite(temps_c, division["latitude"], _DATA_START_YEAR)
    return precip_mm, pet_mm


def _spei(precip_mm: np.ndarray, pet_mm: np.ndarray, scale: int, start_year: int, end_year: int) -> np.ndarray:
    return indices.spei(
        precip_mm,
        pet_mm,
        scale,
        indices.Distribution.loglogistic,
        compute.Periodicity.monthly,
        start_year,
        start_year,
        end_year,
    )


def _parameters(precip_mm: np.ndarray, pet_mm: np.ndarray, scale: int, start_year: int, end_year: int) -> np.ndarray:
    """The library's (3, 12) loc/scale/shape parameters for one scaled series."""
    # mirror indices.spei's negative-precipitation clip before forming the water balance
    precip_mm = np.clip(precip_mm, 0.0, None)
    scaled = compute.prepare_scaled(
        (precip_mm - pet_mm) + 1000.0, scale, compute.Periodicity.monthly, clip_negatives=False
    )
    locs, scales, shapes = compute.loglogistic_parameters(
        scaled, start_year, start_year, end_year, compute.Periodicity.monthly
    )
    return np.stack([locs, scales, shapes])


@pytest.mark.validation
@pytest.mark.parametrize("scale", _REAL_SCALES)
def test_spei_loglogistic_matches_r_spei(scale: int) -> None:
    """``indices.spei`` reproduces R SPEI's log-logistic series and GLO parameters."""
    reference = np.load(_SPEI_ROOT / f"r_spei_{scale:02d}.npy")
    reference_params = np.load(_SPEI_ROOT / f"r_params_{scale:02d}.npy")
    assert reference.shape == (len(_DIVISIONS), _N_MONTHS)
    assert reference_params.shape == (len(_DIVISIONS), 3, 12)

    for row, division in enumerate(_DIVISIONS):
        precip_mm, pet_mm = _environmental_inputs(division)
        computed = _spei(precip_mm, pet_mm, scale, _DATA_START_YEAR, _DATA_END_YEAR)
        np.testing.assert_allclose(
            computed,
            reference[row],
            atol=_SERIES_ATOL,
            equal_nan=True,
            err_msg=f"{division['id']} SPEI-{scale} differs from R SPEI",
        )
        np.testing.assert_allclose(
            _parameters(precip_mm, pet_mm, scale, _DATA_START_YEAR, _DATA_END_YEAR),
            reference_params[row],
            rtol=_PARAMETER_RTOL,
            atol=_PARAMETER_ATOL,
            err_msg=f"{division['id']} SPEI-{scale} GLO parameters differ from R SPEI",
        )


@pytest.mark.validation
@pytest.mark.parametrize("scale", _SYNTHETIC_SCALES)
def test_synthetic_short_skewed_series_matches_r_spei(scale: int) -> None:
    """A short, highly skewed series matches R SPEI, including the clipped tail."""
    precip_mm = np.load(_SPEI_ROOT / "synthetic_precip_mm.npy")
    pet_mm = np.load(_SPEI_ROOT / "synthetic_pet_mm.npy")
    reference = np.load(_SPEI_ROOT / f"r_synthetic_spei_{scale:02d}.npy")
    end_year = _SYNTHETIC_START_YEAR + _SYNTHETIC_YEARS - 1

    computed = _spei(precip_mm, pet_mm, scale, _SYNTHETIC_START_YEAR, end_year)

    # the masks below skip non-finite reference values; pin where those may occur so
    # a refreshed fixture with unintended gaps cannot pass by comparing less
    expected_missing = np.zeros(reference.shape, dtype=bool)
    expected_missing[: scale - 1] = True  # leading months a longer scale cannot form
    if scale == 1:
        expected_missing[_SYNTHETIC_CONSTANT_MONTH - 1 :: 12] = True  # the unfittable step
    np.testing.assert_array_equal(np.isnan(reference), expected_missing)

    # R SPEI does not clip; compare where its output is finite and inside the clip,
    # then assert this library clips the out-of-range tail it reports.
    comparable = np.isfinite(reference) & (np.abs(reference) < _CLIP - 1e-9)
    np.testing.assert_allclose(
        computed[comparable],
        reference[comparable],
        atol=_SERIES_ATOL,
        err_msg=f"synthetic SPEI-{scale} differs from R SPEI",
    )
    # SPEI reports a value beyond the fitted support as +/-inf; this library clips it
    beyond = ~np.isnan(reference) & (np.abs(reference) >= _CLIP)
    if beyond.any():
        np.testing.assert_array_equal(computed[beyond], np.sign(reference[beyond]) * _CLIP)

    # R reports an unfittable step's parameters as NA; this library marks it with a
    # zeroed scale ("0" vs. NaN), so compare the steps both implementations fit and
    # leave the unfittable-step contract to test_synthetic_constant_step_is_missing_like_r_spei.
    computed_params = _parameters(precip_mm, pet_mm, scale, _SYNTHETIC_START_YEAR, end_year)
    reference_params = np.load(_SPEI_ROOT / f"r_synthetic_params_{scale:02d}.npy")
    expected_unfit = np.zeros(reference_params.shape, dtype=bool)
    if scale == 1:
        expected_unfit[:, _SYNTHETIC_CONSTANT_MONTH - 1] = True
    np.testing.assert_array_equal(np.isnan(reference_params), expected_unfit)
    valid = np.isfinite(reference_params)
    np.testing.assert_allclose(
        computed_params[valid],
        reference_params[valid],
        rtol=_PARAMETER_RTOL,
        atol=_PARAMETER_ATOL,
        err_msg=f"synthetic SPEI-{scale} GLO parameters differ from R SPEI",
    )


@pytest.mark.validation
def test_synthetic_constant_step_is_missing_like_r_spei() -> None:
    """A constant calendar step cannot be fitted by either implementation.

    ``SPEI`` reports the step's parameters and values as NA; this library marks
    the step's parameters with a zeroed scale and its values as NaN (ADR-0016).
    The other calendar steps remain finite and are compared above.
    """
    reference = np.load(_SPEI_ROOT / "r_synthetic_spei_01.npy")
    reference_params = np.load(_SPEI_ROOT / "r_synthetic_params_01.npy")
    precip_mm = np.load(_SPEI_ROOT / "synthetic_precip_mm.npy")
    pet_mm = np.load(_SPEI_ROOT / "synthetic_pet_mm.npy")

    computed = _spei(precip_mm, pet_mm, 1, _SYNTHETIC_START_YEAR, _SYNTHETIC_START_YEAR + _SYNTHETIC_YEARS - 1)
    constant_positions = slice(_SYNTHETIC_CONSTANT_MONTH - 1, None, 12)

    assert np.all(np.isnan(reference[constant_positions]))
    assert np.all(np.isnan(computed[constant_positions]))
    assert np.isfinite(computed).sum() == computed.size - computed[constant_positions].size

    month = _SYNTHETIC_CONSTANT_MONTH - 1
    assert np.all(np.isnan(reference_params[:, month]))
    locs, scales, shapes = _parameters(
        precip_mm, pet_mm, 1, _SYNTHETIC_START_YEAR, _SYNTHETIC_START_YEAR + _SYNTHETIC_YEARS - 1
    )
    assert scales[month] == 0.0
    assert locs[month] == 0.0
    assert shapes[month] == 0.0


def test_fixture_exercises_extreme_skew_and_a_negative_location() -> None:
    """The fixtures must actually span the edge cases the ticket names."""
    synthetic_params = np.load(_SPEI_ROOT / "r_synthetic_params_01.npy")
    assert np.nanmin(synthetic_params[2]) < -0.5, "synthetic fixture does not exercise extreme skew"

    # the +1000*scale offset this library adds cancels in standardization; the
    # underlying GLO location is negative for the arid divisions
    real_locations = np.load(_SPEI_ROOT / "r_params_01.npy")[:, 0, :] - 1000.0
    assert real_locations.min() < 0.0, "real fixture does not include a negative location parameter"

    # the synthetic scale-1 fit maps one value beyond the GLO support (SPEI reports it
    # as -inf), so the clip assertion in the parametrized test is actually exercised
    synthetic_reference = np.load(_SPEI_ROOT / "r_synthetic_spei_01.npy")
    assert np.any(~np.isnan(synthetic_reference) & (np.abs(synthetic_reference) >= _CLIP)), (
        "synthetic fixture does not reach this library's clip range"
    )

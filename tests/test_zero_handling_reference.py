"""Cross-implementation zero-placement validation against R SEI and SCI (issue #1209).

ADR-0015's zero-placement modes were pinned by closed-form and property
evidence in ``tests/test_zero_handling.py``; this module compares them to the
independent R implementations that define the same conventions:

- ``SEI`` (Allen & Otero, 2024): ``std_index(..., lower = 0, cens = "prob")``
  and ``cens = "normal"`` for the centre-of-mass and mean-zero constants.
- ``SCI`` (Gudmundsson & Stagge, 2014): ``p0.center.mass = TRUE`` for the
  Stagge et al. (2015) centre-of-mass placement.

The committed fixtures are produced by
``scripts/prepare_zero_handling_fixtures.py``; see
``tests/fixture/zero_handling/provenance.json`` for the pinned package versions
and tolerances. The tests are marked ``validation`` (run with ``-m validation``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

from climate_indices import compute, indices

pytestmark = pytest.mark.validation

_FIXTURE_DIR = Path(__file__).parent / "fixture" / "zero_handling"
_DATA_START_YEAR = 1981
_CALIBRATION_END_YEAR = 2080
_SPI_LIMIT = 3.09
_MODES = ("classic", "center_of_mass", "mean_zero")
_SEI_FILES = {
    "classic": "sei_classic.npy",
    "center_of_mass": "sei_center_of_mass.npy",
    "mean_zero": "sei_mean_zero.npy",
}


def _provenance() -> dict:
    with (_FIXTURE_DIR / "provenance.json").open() as handle:
        return json.load(handle)


def _load(name: str) -> np.ndarray:
    return np.load(_FIXTURE_DIR / name)


def _probabilities_of_zero(values: np.ndarray) -> np.ndarray:
    return np.count_nonzero(values == 0, axis=0) / values.shape[0]


def _spi(values: np.ndarray, zero_handling: str) -> np.ndarray:
    transformed = indices.spi(
        values.flatten(),
        1,
        indices.Distribution.gamma,
        _DATA_START_YEAR,
        _DATA_START_YEAR,
        _CALIBRATION_END_YEAR,
        compute.Periodicity.monthly,
        zero_handling=zero_handling,
    )
    return transformed.reshape(values.shape)


@pytest.mark.parametrize("zero_handling", _MODES)
def test_sei_zero_placement_constants_match_exactly(zero_handling: str) -> None:
    """Every moved zero takes SEI's censored-PIT constant for its calendar step."""
    values = _load("input_precipitation_mm.npy")
    sei = _load(_SEI_FILES[zero_handling])
    probabilities_of_zero = _probabilities_of_zero(values)

    computed = _spi(values, zero_handling)
    zeros = values == 0
    # The all-zero step (p0 == 1) has no defined zero mass in climate_indices
    # (ADR-0015 decision 5); it is covered by its own test below.
    movable = (probabilities_of_zero > 0.0) & (probabilities_of_zero < 1.0)

    np.testing.assert_allclose(
        computed[zeros & movable],
        sei[zeros & movable],
        rtol=0.0,
        atol=_provenance()["validation_tolerance"]["zero_constant_atol"],
    )


@pytest.mark.parametrize("zero_handling", _MODES)
def test_sei_parameter_matched_full_series_matches_sei(zero_handling: str) -> None:
    """With SEI's fitted shape/rate, the whole transformed series reproduces SEI's."""
    values = _load("input_precipitation_mm.npy")
    sei = _load(_SEI_FILES[zero_handling])
    shape = _load("sei_shape.npy")
    rate = _load("sei_rate.npy")
    probabilities_of_zero = _probabilities_of_zero(values)

    fitted = np.isfinite(shape) & np.isfinite(rate)
    computed = compute.transform_fitted_gamma(
        values,
        _DATA_START_YEAR,
        _DATA_START_YEAR,
        _CALIBRATION_END_YEAR,
        compute.Periodicity.monthly,
        alphas=shape,
        betas=1.0 / rate,
        probabilities_of_zero=probabilities_of_zero,
        zero_handling=zero_handling,
    )

    np.testing.assert_allclose(
        computed[:, fitted],
        sei[:, fitted],
        rtol=0.0,
        atol=_provenance()["validation_tolerance"]["parameter_matched_atol"],
    )


@pytest.mark.parametrize("zero_handling", _MODES)
def test_sei_end_to_end_index_agrees_within_documented_tolerance(zero_handling: str) -> None:
    """The full pipeline agrees with SEI despite its MLE gamma versus our method of moments."""
    values = _load("input_precipitation_mm.npy")
    sei = _load(_SEI_FILES[zero_handling])
    shape = _load("sei_shape.npy")
    fitted = np.isfinite(shape)

    computed = np.clip(_spi(values, zero_handling), -_SPI_LIMIT, _SPI_LIMIT)
    reference = np.clip(sei, -_SPI_LIMIT, _SPI_LIMIT)

    np.testing.assert_allclose(
        computed[:, fitted],
        reference[:, fitted],
        rtol=0.0,
        atol=_provenance()["validation_tolerance"]["end_to_end_atol"],
    )


def test_sci_center_of_mass_matches_within_its_p0_estimator_bias() -> None:
    """SCI's centre-of-mass placement agrees with our exercised mode up to its Weibull p0 estimator."""
    values = _load("input_precipitation_mm.npy")
    sci_p0 = _load("sci_center_of_mass_probability.npy")
    probabilities_of_zero = _probabilities_of_zero(values)
    sample_size = values.shape[0]
    movable = (probabilities_of_zero > 0.0) & (probabilities_of_zero < 1.0)

    # SCI estimates p0 with the Weibull plotting position np/(n+1) and places a
    # zero at the centre of [0, (np+1)/(n+1)]; climate_indices uses np/n and np/(2n).
    defined = np.isfinite(sci_p0)
    np.testing.assert_allclose(
        sci_p0[defined],
        probabilities_of_zero[defined] * sample_size / (sample_size + 1),
        rtol=0.0,
        atol=1e-12,
    )
    sci_placement = scipy.stats.norm.ppf((sci_p0 + 1.0 / (sample_size + 1)) / 2.0)

    # exercise the library's mode, not just the closed form, and pin it exactly
    computed = _spi(values, "center_of_mass")
    closed_form = scipy.stats.norm.ppf(probabilities_of_zero / 2.0)
    movable_zeros = (values == 0) & np.broadcast_to(movable, values.shape)
    np.testing.assert_allclose(
        computed[movable_zeros], np.broadcast_to(closed_form, values.shape)[movable_zeros], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        closed_form[movable],
        sci_placement[movable],
        rtol=0.0,
        atol=_provenance()["validation_tolerance"]["sci_center_of_mass_atol"],
    )


def test_sei_fitted_parameters_are_not_climate_indices_own_fit() -> None:
    """The SEI fixture carries an external MLE fit, not this library's method-of-moments output."""
    values = _load("input_precipitation_mm.npy")
    shape = _load("sei_shape.npy")
    rate = _load("sei_rate.npy")
    notes = _provenance()["notes"]
    assert "SEI 0.2.0" in notes
    assert "SCI 1.0.3" in notes

    alphas, betas = compute.gamma_parameters(
        values, _DATA_START_YEAR, _DATA_START_YEAR, _CALIBRATION_END_YEAR, compute.Periodicity.monthly
    )
    fittable = np.isfinite(shape) & np.isfinite(rate)
    # MLE (SEI) and Thom's method of moments differ by a few tenths of a percent;
    # a self-generated fixture would match exactly instead.
    assert np.max(np.abs((alphas[fittable] - shape[fittable]) / shape[fittable])) > 1e-3
    assert np.max(np.abs((betas[fittable] - 1.0 / rate[fittable]) / (1.0 / rate[fittable]))) > 1e-3


def test_all_zero_calibration_step_follows_adr_not_sei() -> None:
    """An all-zero step has no defined zero mass here; SEI still assigns a constant."""
    values = _load("input_precipitation_mm.npy")
    probabilities_of_zero = _probabilities_of_zero(values)
    all_zero = probabilities_of_zero >= 1.0
    zeros_all_zero_step = (values == 0) & all_zero

    for zero_handling in _MODES:
        computed = _spi(values, zero_handling)
        np.testing.assert_allclose(computed[zeros_all_zero_step], -_SPI_LIMIT, rtol=0.0)

    # SEI's censored modes place a constant (0 on the normal scale) even with no
    # fit; its classic mode gives +Inf instead.
    for zero_handling in ("center_of_mass", "mean_zero"):
        sei = _load(_SEI_FILES[zero_handling])
        np.testing.assert_allclose(sei[zeros_all_zero_step], 0.0, rtol=0.0)

"""Tests for the `_PalmerPrepared` and `_PalmerRecursion` structs behind one Palmer location."""

from typing import Any

import numpy as np

from climate_indices import palmer

_VALID_PARAMS: dict[str, Any] = {
    "alpha": [1.0] * 12,
    "beta": [2.0] * 12,
    "gamma": [3.0] * 12,
    "delta": [4.0] * 12,
}


def _initialize(fitting_params: dict[str, Any] | None = None) -> palmer._PalmerPrepared:
    return palmer._initialize_prepared(
        precips=np.zeros(12 * 2),
        pet=np.zeros(12 * 2),
        awc=1.0,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
        fitting_params=fitting_params,
    )


def test_initialize_recursion_defaults_the_recursion_state():
    """The recursion assigns these before reading them, and zero before it runs."""
    prepared = _initialize()
    state = palmer._initialize_recursion(prepared)

    assert prepared.calibrate is True
    assert (state.x1, state.x2, state.x3, state.v, state.pro) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert (state.pv, state.ze, state.ud, state.uw) == (0.0, 0.0, 0.0, 0.0)
    assert (state.k8, state.k8max, state.iass, state.year, state.month) == (0, 0, 0, 0, 0)
    assert state.pdsi.shape == (2, 12)
    assert state.indexj.shape == (palmer.K8_SIZE,)


def test_complete_fitting_params_skip_calibration():
    prepared = _initialize(_VALID_PARAMS)

    assert prepared.calibrate is False
    assert np.array_equal(prepared.alpha, np.full((12,), 1.0))
    assert np.array_equal(prepared.beta, np.full((12,), 2.0))
    assert np.array_equal(prepared.gamma, np.full((12,), 3.0))
    assert np.array_equal(prepared.delta, np.full((12,), 4.0))


def test_incomplete_or_malformed_fitting_params_fall_back_to_calibration():
    cases = {
        "missing": {"alpha": [1.0] * 12},
        "wrong length": {**_VALID_PARAMS, "gamma": [3.0] * 11},
        "wrong type": {**_VALID_PARAMS, "delta": "not-an-array"},
        "wrong container element": {**_VALID_PARAMS, "beta": np.zeros((2, 12))},
        "two-dimensional": {**_VALID_PARAMS, "beta": np.zeros((12, 2))},
        "non-numeric": {**_VALID_PARAMS, "gamma": ["a"] * 12},
    }
    for name, params in cases.items():
        prepared = _initialize(params)

        assert prepared.calibrate is True, name
        # no coefficient is adopted from a rejected parameter set
        assert np.isnan(prepared.alpha).all() and np.isnan(prepared.beta).all(), name
        assert np.isnan(prepared.gamma).all() and np.isnan(prepared.delta).all(), name

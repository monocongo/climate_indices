"""Tests for the `_PalmerData` struct that carries one location's Palmer state."""

from typing import Any

import numpy as np

from climate_indices import palmer

_VALID_PARAMS: dict[str, Any] = {
    "alpha": [1.0] * 12,
    "beta": [2.0] * 12,
    "gamma": [3.0] * 12,
    "delta": [4.0] * 12,
}


def _initialize(fitting_params: dict[str, Any] | None = None) -> palmer._PalmerData:
    return palmer._initialize_data(
        precips=np.zeros(12 * 2),
        pet=np.zeros(12 * 2),
        awc=1.0,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
        fitting_params=fitting_params,
    )


def test_initialize_data_defaults_the_recursion_state():
    """The recursion assigns these before reading them, and zero before it runs."""
    data = _initialize()

    assert data.calibrate is True
    assert (data.x1, data.x2, data.x3, data.v, data.pro) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert (data.pv, data.ze, data.ud, data.uw) == (0.0, 0.0, 0.0, 0.0)
    assert (data.k8, data.k8max, data.iass, data.year, data.month) == (0, 0, 0, 0, 0)
    assert data.pdsi.shape == (2, 12)
    assert data.indexj.shape == (palmer.K8_SIZE,)


def test_complete_fitting_params_skip_calibration():
    data = _initialize(_VALID_PARAMS)

    assert data.calibrate is False
    assert np.array_equal(data.alpha, np.full((12,), 1.0))
    assert np.array_equal(data.beta, np.full((12,), 2.0))
    assert np.array_equal(data.gamma, np.full((12,), 3.0))
    assert np.array_equal(data.delta, np.full((12,), 4.0))


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
        data = _initialize(params)

        assert data.calibrate is True, name
        # no coefficient is adopted from a rejected parameter set
        assert np.isnan(data.alpha).all() and np.isnan(data.beta).all(), name
        assert np.isnan(data.gamma).all() and np.isnan(data.delta).all(), name

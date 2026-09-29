"""Tests for the ``FittedDistribution`` seam introduced by issue #1216."""

import numpy as np
import pytest

from climate_indices import compute, indices


def _monthly_values() -> np.ndarray:
    return np.arange(1.0, 481.0).reshape(40, 12)


def test_resolve_from_data_transforms_like_the_public_transform() -> None:
    """A fit built from data transforms its values like the compatibility adapter."""
    values = _monthly_values()
    fitted = compute.FittedDistribution.resolve(
        values, indices.Distribution.gamma, 1981, 1981, 2010, compute.Periodicity.monthly
    )

    np.testing.assert_array_equal(
        fitted.transform(values),
        compute.transform_fitted_gamma(values, 1981, 1981, 2010, compute.Periodicity.monthly),
    )


def test_transform_and_diagnostics_share_one_seam() -> None:
    """The transform and diagnostics methods compose over the same resolved fit."""
    values = _monthly_values()
    fitted = compute.FittedDistribution.resolve(
        values, indices.Distribution.pearson, 1981, 1981, 2010, compute.Periodicity.monthly
    )

    np.testing.assert_array_equal(
        fitted.transform(values),
        compute.transform_fitted_pearson(values, 1981, 1981, 2010, compute.Periodicity.monthly),
    )
    ks, p_value, n_valid, parameters = fitted.diagnostics(values[:30])
    diagnostics = compute.fit_diagnostics(
        values, indices.Distribution.pearson, 1981, 1981, 2010, compute.Periodicity.monthly
    )
    np.testing.assert_array_equal(ks, diagnostics.ks_statistic)
    np.testing.assert_array_equal(p_value, diagnostics.ks_p_value)
    np.testing.assert_array_equal(n_valid, diagnostics.n_valid)
    assert set(parameters) == set(diagnostics.parameters)


def test_resolve_from_supplied_parameters_does_not_refit(monkeypatch) -> None:
    """A complete supplied parameter set is used as-is rather than fitted again."""
    values = _monthly_values()
    alphas = np.full(12, 4.0)
    betas = np.full(12, 8.0)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("gamma_parameters should not be called")

    monkeypatch.setattr(compute, "gamma_parameters", fail)
    fitted = compute.FittedDistribution.resolve(
        values,
        indices.Distribution.gamma,
        1981,
        1981,
        2010,
        compute.Periodicity.monthly,
        {"alpha": alphas, "beta": betas, "prob_zero": np.zeros(12)},
    )

    np.testing.assert_array_equal(fitted.parameters["alpha"], alphas)
    np.testing.assert_array_equal(fitted.parameters["beta"], betas)


def test_resolve_rejects_a_partial_pearson_set() -> None:
    """A partial Pearson set is one argument error across every surface."""
    values = _monthly_values()

    with pytest.raises(ValueError, match="either none or all"):
        compute.FittedDistribution.resolve(
            values,
            indices.Distribution.pearson,
            1981,
            1981,
            2010,
            compute.Periodicity.monthly,
            {"loc": np.ones(12)},
        )


def test_resolve_rejects_a_mis_shaped_pearson_set() -> None:
    """A Pearson parameter that does not carry the period axis is rejected before any fit."""
    values = _monthly_values()
    parameters = {
        "prob_zero": np.zeros(11),
        "loc": np.ones(11),
        "scale": np.ones(11),
        "skew": np.ones(11),
    }

    with pytest.raises(ValueError, match="must carry the period length"):
        compute.FittedDistribution.resolve(
            values, indices.Distribution.pearson, 1981, 1981, 2010, compute.Periodicity.monthly, parameters
        )

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
    for name, parameter in parameters.items():
        np.testing.assert_array_equal(parameter, diagnostics.parameters[name])


def test_pearson_fallback_decision_is_shared_by_index_and_diagnostics() -> None:
    """The fall-back decision agrees between ``fit_and_standardize`` and ``fit_diagnostics``.

    A zero-heavy block whose failed Pearson steps still place their sub-0.0005 values
    must not be reported as a gamma fall back when the index path keeps Pearson (#1216).
    """
    rng = np.random.default_rng(7)
    values = rng.gamma(2.0, 2.0, size=(40, 12))
    # seven calendar steps have only two non-zero calibration values, so their Pearson
    # fit fails and every value below 0.0005 is placed rather than lost
    values[:, :7] = 0.0
    values[::20, :7] = 1.0

    diagnostics = compute.fit_diagnostics(
        values,
        indices.Distribution.pearson,
        1981,
        1981,
        2010,
        compute.Periodicity.monthly,
        fallback_to_gamma=True,
    )
    standardized = compute.fit_and_standardize(
        values,
        indices.Distribution.pearson,
        1981,
        1981,
        2010,
        compute.Periodicity.monthly,
        fallback_to_gamma=True,
    )
    # compare against the Pearson transform: equality means no fall back happened
    pearson_only = compute.transform_fitted_pearson(values, 1981, 1981, 2010, compute.Periodicity.monthly)
    stayed_pearson = bool(np.array_equal(standardized, pearson_only, equal_nan=True))

    assert stayed_pearson
    assert diagnostics.fell_back_to_gamma is False


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

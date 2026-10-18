"""The beta distribution on ``standardized_index()`` (issue #1169, ADR-0024)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from climate_indices import compute, indices
from climate_indices.exceptions import InvalidArgumentError

_MONTHLY = compute.Periodicity.monthly
_DAILY = compute.Periodicity.daily

# the two calendar months of the helpers that carry a partial boundary mass
_ZERO_MONTH = 0
_ONE_MONTH = 1
_ZERO_FRACTION = 0.5
_ONE_FRACTION = 0.25


def _bounded_monthly(years: int = 60, seed: int = 1169) -> np.ndarray:
    """A (years, 12) bounded series: an interior beta sample, half zeros, and a quarter ones."""
    values = np.random.default_rng(seed).beta(2.0, 5.0, size=(years, 12))
    values[: years // 2, _ZERO_MONTH] = 0.0
    values[: years // 4, _ONE_MONTH] = 1.0
    return values


def _score(probability: float) -> float:
    """The standard-normal score of a cumulative probability."""
    return float(scipy.stats.norm.ppf(probability))


def test_beta_standardizes_a_bounded_series_to_a_standard_normal() -> None:
    """A long sample from a beta distribution standardizes to a standard normal per step."""
    values = _bounded_monthly(years=200)
    years, months = values.shape

    standardized = indices.standardized_index(
        values.ravel(),
        scale=1,
        distribution=indices.Distribution.beta,
        data_start_year=1900,
        calibration_year_initial=1900,
        calibration_year_final=1900 + years - 1,
        periodicity=_MONTHLY,
        aggregation="mean",
    ).reshape(years, months)

    for month in (2, 5, 11):
        column = standardized[:, month]
        assert np.nanmean(column) == pytest.approx(0.0, abs=0.15)
        assert np.nanstd(column) == pytest.approx(1.0, abs=0.15)
        assert scipy.stats.kstest(column, "norm").pvalue > 0.01


def test_beta_recovers_the_shape_parameters_of_a_synthetic_sample() -> None:
    """The method of moments recovers known shape parameters from a large sample."""
    alphas, betas = 2.0, 5.0
    values = np.random.default_rng(23).beta(alphas, betas, size=(20000, 12))

    fitted_alphas, fitted_betas, probabilities_of_zero, probabilities_of_one = compute.beta_parameters(
        values, 1900, 1900, 21900 + 1900 - 1, _MONTHLY
    )

    # a sample of 20000 per step leaves the method of moments within about 3 percent
    np.testing.assert_allclose(fitted_alphas, np.full(12, alphas), rtol=0.05, atol=0.0)
    np.testing.assert_allclose(fitted_betas, np.full(12, betas), rtol=0.05, atol=0.0)
    assert np.all(probabilities_of_zero == 0.0)
    assert np.all(probabilities_of_one == 0.0)


def test_supplied_parameters_reproduce_the_probability_integral_transform() -> None:
    """The transform is p0 + (1 - p0 - p1) * F(x), mapped to the standard normal."""
    values = _bounded_monthly(years=40)
    alphas, betas, probabilities_of_zero, probabilities_of_one = compute.beta_parameters(
        values, 1900, 1900, 1939, _MONTHLY
    )
    np.testing.assert_allclose(probabilities_of_zero[_ZERO_MONTH], _ZERO_FRACTION, rtol=0, atol=0.0)
    assert probabilities_of_one[_ONE_MONTH] == _ONE_FRACTION

    standardized = compute.transform_fitted_beta(
        values,
        1900,
        1900,
        1939,
        _MONTHLY,
        alphas,
        betas,
        probabilities_of_zero,
        probabilities_of_one,
    )

    # the beta CDF is exactly 0 at 0 and exactly 1 at 1, so one formula covers every value
    expected = scipy.stats.norm.ppf(
        probabilities_of_zero
        + (1.0 - probabilities_of_zero - probabilities_of_one)
        * scipy.stats.beta.cdf(np.clip(values, 0.0, 1.0), alphas, betas)
    )
    np.testing.assert_allclose(standardized, expected, rtol=0, atol=1e-12)

    # the boundary rows carry their own mass, not the interior CDF at that value
    np.testing.assert_allclose(standardized[:20, _ZERO_MONTH], _score(_ZERO_FRACTION), rtol=0, atol=1e-12)
    np.testing.assert_allclose(standardized[:10, _ONE_MONTH], _score(1.0 - _ONE_FRACTION), rtol=0, atol=1e-12)


def test_boundary_values_score_their_own_mass_at_both_ends() -> None:
    """A zero scores at the top of the lower mass and a one at the bottom of the upper."""
    ones = np.ones(12)
    values = np.empty((3, 12))
    values[0, :] = 0.0
    values[1, :] = 1.0
    values[2, :] = 0.5
    # equal masses at the two boundaries, so the mirror of a placement is its negative
    equal = ones * 0.25

    for handling in ("classic", "center_of_mass", "mean_zero"):
        standardized = compute.transform_fitted_beta(
            values, 1900, 1900, 1902, _MONTHLY, ones * 2.0, ones * 5.0, equal, equal, zero_handling=handling
        )
        np.testing.assert_allclose(standardized[0, :], -standardized[1, :], rtol=0, atol=1e-12)
        assert np.all(standardized[2, :] < standardized[1, :])
        assert np.all(standardized[2, :] > standardized[0, :])

    distributed = compute.transform_fitted_beta(
        values, 1900, 1900, 1902, _MONTHLY, ones * 2.0, ones * 5.0, ones * 0.3, ones * 0.2
    )
    np.testing.assert_allclose(distributed[0, :], _score(0.3), rtol=0, atol=1e-12)
    np.testing.assert_allclose(distributed[1, :], _score(0.8), rtol=0, atol=1e-12)

    probability_scale = compute.transform_fitted_beta(
        values,
        1900,
        1900,
        1902,
        _MONTHLY,
        ones * 2.0,
        ones * 5.0,
        ones * 0.5,
        ones * 0.5,
        output_scale="probability",
    )
    np.testing.assert_allclose(probability_scale[0, :], 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(probability_scale[1, :], 0.5, rtol=0, atol=1e-12)

    center = compute.transform_fitted_beta(
        values, 1900, 1900, 1902, _MONTHLY, ones * 2.0, ones * 5.0, equal, equal, zero_handling="center_of_mass"
    )
    np.testing.assert_allclose(center[0, :], _score(0.125), rtol=0, atol=1e-12)
    np.testing.assert_allclose(center[1, :], _score(0.875), rtol=0, atol=1e-12)

    mean_zero = compute.transform_fitted_beta(
        values, 1900, 1900, 1902, _MONTHLY, ones * 2.0, ones * 5.0, equal, equal, zero_handling="mean_zero"
    )
    tail_mean = float(scipy.stats.norm.pdf(scipy.stats.norm.ppf(0.25)) / 0.25)
    np.testing.assert_allclose(mean_zero[0, :], -tail_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mean_zero[1, :], tail_mean, rtol=0, atol=1e-12)


def test_all_zero_and_all_one_steps_score_their_own_extreme_end() -> None:
    """A step that never leaves one boundary scores that end, as gamma reads an all-zero step."""
    values = np.random.default_rng(9).beta(2.0, 5.0, size=(30, 12))
    values[:, _ZERO_MONTH] = 0.0
    values[:, _ONE_MONTH] = 1.0

    standardized = compute.transform_fitted_beta(values, 1900, 1900, 1929, _MONTHLY)

    assert np.all(np.isneginf(standardized[:, _ZERO_MONTH]))
    assert np.all(np.isposinf(standardized[:, _ONE_MONTH]))
    # the fitted shape parameters are missing for those steps, because the interior holds
    # no calibration value, while the boundary values still score from their own mass
    alphas, betas, _, _ = compute.beta_parameters(values, 1900, 1900, 1929, _MONTHLY)
    assert not np.isfinite(alphas[_ZERO_MONTH])
    assert not np.isfinite(betas[_ZERO_MONTH])

    clipped = indices.standardized_index(
        values.ravel(),
        scale=1,
        distribution=indices.Distribution.beta,
        data_start_year=1900,
        calibration_year_initial=1900,
        calibration_year_final=1929,
        periodicity=_MONTHLY,
        aggregation="mean",
    ).reshape(values.shape)
    assert np.all(clipped[:, _ZERO_MONTH] == indices._FITTED_INDEX_VALID_MIN)
    assert np.all(clipped[:, _ONE_MONTH] == indices._FITTED_INDEX_VALID_MAX)


def test_out_of_support_values_are_rejected_rather_than_clipped() -> None:
    """A value outside [0, 1] is a data error, and the support tolerance is narrow."""
    for value in (1.5, -0.2, 1.0 + 1e-9):
        with pytest.raises(InvalidArgumentError, match="only values in"):
            compute.transform_fitted_beta(np.full((2, 12), value), 1900, 1900, 1901, _MONTHLY)

    # a windowed mean over bounded values can round a few ulps outside, which is a one
    standardized = compute.transform_fitted_beta(
        np.full((2, 12), 1.0 + 1e-13),
        1900,
        1900,
        1901,
        _MONTHLY,
        np.full(12, 2.0),
        np.full(12, 5.0),
        np.zeros(12),
        np.ones(12),
    )
    assert np.all(np.isposinf(standardized))


def test_mass_sum_above_one_is_rejected() -> None:
    """Masses that leave the interior a negative probability are rejected."""
    ones = np.ones(12)
    with pytest.raises(ValueError, match="must not sum above 1"):
        compute.transform_fitted_beta(
            np.full((2, 12), 0.5), 1900, 1900, 1901, _MONTHLY, ones * 2.0, ones * 5.0, ones * 0.7, ones * 0.7
        )


def test_a_step_without_calibration_data_is_missing() -> None:
    """A calendar step with no calibration value has no defined mass, so it is NaN."""
    values = _bounded_monthly(years=40)
    values[10:, 3] = np.nan

    standardized = compute.transform_fitted_beta(values, 1900, 1900, 1939, _MONTHLY)

    assert np.all(np.isnan(standardized[10:, 3]))
    assert np.all(np.isfinite(standardized[:10, 3]))


def test_fitting_parameters_round_trip_through_the_pipeline() -> None:
    """A fitted parameter set reproduces the same values when passed back in."""
    values = _bounded_monthly(years=40)
    arguments = {
        "scale": 1,
        "distribution": indices.Distribution.beta,
        "data_start_year": 1900,
        "calibration_year_initial": 1900,
        "calibration_year_final": 1939,
        "periodicity": _MONTHLY,
        "aggregation": "mean",
    }
    fitted = indices.standardized_index(values.ravel(), **arguments)

    alphas, betas, probabilities_of_zero, probabilities_of_one = compute.beta_parameters(
        values, 1900, 1900, 1939, _MONTHLY
    )
    reused = indices.standardized_index(
        values.ravel(),
        fitting_params={
            "alpha": alphas,
            "beta": betas,
            "prob_zero": probabilities_of_zero,
            "prob_one": probabilities_of_one,
        },
        **arguments,
    )

    np.testing.assert_allclose(reused, fitted, rtol=0, atol=1e-12)


def test_beta_serves_daily_data_and_mean_aggregation() -> None:
    """The daily layout and a scale above one both run, and the fit covers every day."""
    values = np.random.default_rng(5).beta(2.0, 5.0, size=(10, 366))

    standardized = indices.standardized_index(
        values.ravel(),
        scale=7,
        distribution=indices.Distribution.beta,
        data_start_year=1900,
        calibration_year_initial=1900,
        calibration_year_final=1909,
        periodicity=_DAILY,
        aggregation="mean",
    ).reshape(values.shape)

    assert standardized.shape == values.shape
    assert np.count_nonzero(np.isnan(standardized)) == 6  # the leading scale - 1 pad
    assert np.nanmean(standardized) == pytest.approx(0.0, abs=0.15)


def test_beta_is_rejected_where_the_surface_does_not_support_it() -> None:
    """Beta is the bounded family of standardized_index() alone."""
    values = _bounded_monthly(years=40)
    arguments = {
        "scale": 1,
        "data_start_year": 1900,
        "calibration_year_initial": 1900,
        "calibration_year_final": 1939,
        "periodicity": _MONTHLY,
    }

    with pytest.raises(InvalidArgumentError, match="Unsupported distribution"):
        indices.spi(values.ravel() * 100.0, distribution=indices.Distribution.beta, **arguments)
    with pytest.raises(InvalidArgumentError, match="Unsupported distribution"):
        indices.spei(
            values.ravel() * 100.0,
            values.ravel() * 10.0,
            distribution=indices.Distribution.beta,
            **arguments,
        )
    with pytest.raises(InvalidArgumentError, match="Unsupported distribution"):
        indices.fit_diagnostics(values.ravel(), distribution=indices.Distribution.beta, **arguments)

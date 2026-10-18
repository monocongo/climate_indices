"""Contract tests for the empirical (rank-based) distribution option of the standardize seam (#1168)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import scipy.stats

from climate_indices import compute, indices
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.xarray_adapter import _supplies_complete_fit

_MONTHLY = compute.Periodicity.monthly
_EMPIRICAL = indices.Distribution.empirical


def _sei_probabilities(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """The SEI plotting position, (1 + count(calibration <= x)) / (n + 2), per calendar column."""
    expected = np.empty(values.shape)
    for column in range(values.shape[1]):
        reference = calibration[:, column]
        reference = reference[~np.isnan(reference)]
        counts = (reference[:, None] <= values[:, column]).sum(axis=0)
        expected[:, column] = (1 + counts) / (reference.size + 2)
    return expected


def _standardize(values: np.ndarray, **kwargs: Any) -> np.ndarray:
    kwargs.setdefault("distribution", _EMPIRICAL)
    return indices.standardized_index(
        values,
        kwargs.pop("scale", 1),
        kwargs.pop("distribution"),
        1981,
        kwargs.pop("calibration_year_initial", 1981),
        kwargs.pop("calibration_year_final", 2020),
        _MONTHLY,
        **kwargs,
    )


def test_probabilities_match_the_sei_plotting_position_per_calendar_step() -> None:
    """Each month ranks against its own Calibration Period values, not the whole record."""
    values = np.random.default_rng(1168).normal(size=(50, 12))

    computed = _standardize(
        values.flatten(),
        calibration_year_initial=1991,
        calibration_year_final=2020,
        clip_negatives=False,
        output_scale="probability",
    )

    expected = _sei_probabilities(values, values[10:40])
    np.testing.assert_allclose(computed, expected.flatten(), rtol=0, atol=1e-15)


def test_ties_take_the_maximum_rank_and_nothing_is_extrapolated() -> None:
    """Ties count every equal calibration value; p stays in [1/(n+2), (n+1)/(n+2)]."""
    calibration = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    evaluated = np.array([-5.0, 1.0, 8.0, 100.0])
    values = np.repeat(np.concatenate([calibration, evaluated])[:, None], 12, axis=1)

    probabilities = compute.transform_fitted_empirical(values, 2000, 2000, 2009, _MONTHLY, "probability")

    np.testing.assert_array_equal(probabilities[10:, 0], np.array([1.0, 4.0, 11.0, 11.0]) / 12.0)


def test_a_step_with_too_few_valid_calibration_values_is_missing() -> None:
    """Ten valid calibration values is the minimum; a missing value is not counted."""
    values = np.random.default_rng(7).normal(size=(10, 12))
    values[3, 0] = np.nan

    probabilities = compute.transform_fitted_empirical(values, 2000, 2000, 2009, _MONTHLY, "probability")

    assert compute.MIN_CALIBRATION_VALUES_FOR_EMPIRICAL == 10
    assert np.isnan(probabilities[:, 0]).all()
    assert np.isfinite(probabilities[:, 1:]).all()


def test_output_scales_share_one_probability() -> None:
    """normal is the unclipped inverse normal of p and bounded is 2p - 1."""
    values = np.random.default_rng(3).normal(size=(30, 12))

    def transform(output_scale: compute.OutputScale) -> np.ndarray:
        return compute.transform_fitted_empirical(values, 2000, 2000, 2029, _MONTHLY, output_scale)

    probabilities = transform("probability")
    np.testing.assert_allclose(transform("normal"), scipy.stats.norm.ppf(probabilities), rtol=0, atol=0)
    np.testing.assert_allclose(transform("bounded"), 2.0 * probabilities - 1.0, rtol=0, atol=0)


def test_signed_series_keeps_its_negatives_only_when_asked() -> None:
    """The default clips negatives, which ties them at zero; clip_negatives=False ranks them."""
    values = np.random.default_rng(11).normal(size=40 * 12)

    clipped = _standardize(values, output_scale="probability")
    signed = _standardize(values, clip_negatives=False, output_scale="probability")

    negative = values < 0.0
    assert np.unique(clipped.reshape(40, 12)[negative.reshape(40, 12)[:, 0], 0]).size == 1
    assert np.unique(signed.reshape(40, 12)[negative.reshape(40, 12)[:, 0], 0]).size > 1


def test_sum_and_mean_aggregation_rank_identically() -> None:
    """A sum is its mean times scale, so the ranks, and the index, do not change."""
    values = np.random.default_rng(5).normal(size=40 * 12)

    summed = _standardize(values, scale=3, clip_negatives=False)
    averaged = _standardize(values, scale=3, clip_negatives=False, aggregation="mean")

    np.testing.assert_array_equal(summed, averaged)


def test_time_major_block_matches_each_cell_alone() -> None:
    values = np.random.default_rng(13).normal(size=(40 * 12, 2, 3))

    block = _standardize(values, clip_negatives=False, spatial_time_major=True)

    for row, column in np.ndindex(values.shape[1:]):
        series = values[:, row, column]
        np.testing.assert_array_equal(block[:, row, column], _standardize(series, clip_negatives=False))


@pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
def test_parametric_fits_reject_negatives_they_are_told_to_keep(distribution: indices.Distribution) -> None:
    values = np.random.default_rng(17).normal(size=40 * 12)

    with pytest.raises(InvalidArgumentError, match="Distribution.empirical"):
        _standardize(values, distribution=distribution, clip_negatives=False)


def test_clip_negatives_false_leaves_a_non_negative_fit_unchanged() -> None:
    values = np.random.default_rng(19).gamma(2.0, size=40 * 12)

    default = _standardize(values, distribution=indices.Distribution.gamma)
    kept = _standardize(values, distribution=indices.Distribution.gamma, clip_negatives=False)

    np.testing.assert_array_equal(default, kept)


def test_empirical_rejects_fitting_params() -> None:
    values = np.random.default_rng(23).gamma(2.0, size=40 * 12)

    with pytest.raises(InvalidArgumentError, match="no fitting parameters"):
        _standardize(values, fitting_params={"alpha": np.ones(12), "beta": np.ones(12)})

    # an unrecognized key must still be rejected: alias normalization drops it, which
    # would otherwise read as a valid parameter-free call
    with pytest.raises(InvalidArgumentError, match="no fitting parameters"):
        _standardize(values, fitting_params={"unexpected": 1})


def test_seam_rejects_a_malformed_distribution_with_value_error() -> None:
    """A non-Distribution input gets the documented ValueError, not an AttributeError."""
    values = np.random.default_rng(31).gamma(2.0, size=40 * 12)

    for distribution in ("gamma", object()):
        with pytest.raises(ValueError, match="Unsupported distribution"):
            compute.fit_and_standardize(values, cast("indices.Distribution", distribution), 1981, 1981, 1990, _MONTHLY)


def test_other_surfaces_reject_empirical() -> None:
    """Only standardized_index() takes the empirical option; SPI, SPEI, and diagnostics fit."""
    values = np.random.default_rng(29).gamma(2.0, size=40 * 12)
    arguments = (1, _EMPIRICAL, 1981, 1981, 2020, _MONTHLY)

    with pytest.raises(InvalidArgumentError, match="Unsupported distribution"):
        indices.spi(values, *arguments)
    with pytest.raises(InvalidArgumentError, match="Unsupported distribution"):
        indices.spei(values, values, 1, _EMPIRICAL, _MONTHLY, 1981, 1981, 2020)
    with pytest.raises(InvalidArgumentError, match="Unsupported distribution"):
        indices.fit_diagnostics(values, *arguments)


def test_xarray_complete_fit_check_does_not_key_on_empirical() -> None:
    assert not _supplies_complete_fit({"distribution": _EMPIRICAL, "fitting_params": {"alpha": 1.0}})

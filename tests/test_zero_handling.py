"""Zero-placement modes in the gamma and Pearson Type III transforms (issue #1186, ADR-0015)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np
import pytest
import scipy.stats

from climate_indices import compute, indices

_MODES = ("classic", "center_of_mass", "mean_zero")
_DISTRIBUTIONS = (indices.Distribution.gamma, indices.Distribution.pearson)
_MONTHLY = compute.Periodicity.monthly


def _zero_score(probability_of_zero: float, zero_handling: str) -> float:
    """The closed-form normal-scale score ADR-0015 assigns a zero."""
    if zero_handling == "classic":
        return float(scipy.stats.norm.ppf(probability_of_zero))
    if zero_handling == "center_of_mass":
        return float(scipy.stats.norm.ppf(probability_of_zero / 2.0))
    return float(-scipy.stats.norm.pdf(scipy.stats.norm.ppf(probability_of_zero)) / probability_of_zero)


def _positive_monthly(years: int, seed: int) -> np.ndarray:
    """A (years, 12) gamma-distributed series with no zeros."""
    return np.random.default_rng(seed).gamma(2.0, 30.0, size=(years, 12)) + 1.0


def _half_zero_monthly(years: int = 40, seed: int = 11) -> np.ndarray:
    """A (years, 12) series whose every calendar month is exactly half zeros."""
    values = _positive_monthly(years, seed)
    values[::2, :] = 0.0
    return values


def _stepped_zero_monthly(years: int = 40, seed: int = 17) -> np.ndarray:
    """A (years, 12) series whose calendar month m holds 2 * (m + 1) zeros."""
    values = _positive_monthly(years, seed)
    for month in range(12):
        values[: 2 * (month + 1), month] = 0.0
    return values


# each month's zero fraction in a 40-year _stepped_zero_monthly series
_STEPPED_PROBABILITIES_OF_ZERO = np.array([2.0 * (month + 1) / 40.0 for month in range(12)])


def _assert_each_months_zeros(computed: np.ndarray, values: np.ndarray, expected_scores: np.ndarray) -> None:
    """Every zero in calendar month m holds that month's expected score."""
    for month in range(12):
        zeros = values[:, month] == 0
        np.testing.assert_allclose(computed[zeros, month], expected_scores[month], atol=1e-12)


def _spi(values: np.ndarray, distribution: indices.Distribution, **kwargs: Any) -> np.ndarray:
    """SPI-1 of a 1981-onward monthly series, calibrated on 1981-2020 unless told otherwise."""
    calibration = kwargs.pop("calibration", (1981, 2020))
    return indices.spi(
        values.flatten(),
        1,
        distribution,
        1981,
        calibration[0],
        calibration[1],
        _MONTHLY,
        **kwargs,
    )


def _whole_record_probabilities_of_zero(scaled_values: np.ndarray) -> np.ndarray:
    """The gamma probability of zero before ADR-0015: zeros over every year of the record."""
    return np.asarray(np.count_nonzero(scaled_values == 0, axis=0) / scaled_values.shape[0])


def _calibration_probabilities_of_zero(scaled_values: np.ndarray, first_year: int, last_year: int) -> np.ndarray:
    """Zeros over the non-missing values of a 1981-onward series' calibration years."""
    calibration = scaled_values[first_year - 1981 : last_year - 1981 + 1]
    return np.asarray(np.count_nonzero(calibration == 0, axis=0) / np.count_nonzero(~np.isnan(calibration), axis=0))


# ---------------------------------------------------------------------------
# the three modes


@pytest.mark.parametrize("zero_handling", _MODES)
def test_gamma_zero_scores_match_the_closed_forms(zero_handling: str) -> None:
    """At p0 = 0.5 the modes score a zero 0.00, -0.67, and -0.80, the ADR's worked example."""
    values = _half_zero_monthly()

    computed = _spi(values, indices.Distribution.gamma, zero_handling=zero_handling).reshape(values.shape)

    np.testing.assert_allclose(computed[values == 0], _zero_score(0.5, zero_handling), atol=1e-12)
    assert _zero_score(0.5, zero_handling) == pytest.approx(
        {"classic": 0.0, "center_of_mass": -0.6745, "mean_zero": -0.7979}[zero_handling], abs=1e-4
    )


@pytest.mark.parametrize("zero_handling", ["center_of_mass", "mean_zero"])
def test_pearson_zero_scores_match_the_closed_forms(zero_handling: str) -> None:
    """A Pearson Type III zero takes the mode's score for the calibration zero fraction."""
    values = _half_zero_monthly()

    computed = _spi(values, indices.Distribution.pearson, zero_handling=zero_handling).reshape(values.shape)

    np.testing.assert_allclose(computed[values == 0], _zero_score(0.5, zero_handling), atol=1e-12)


@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
@pytest.mark.parametrize("zero_handling", ["center_of_mass", "mean_zero"])
def test_non_zero_values_keep_their_classic_transform(distribution: indices.Distribution, zero_handling: str) -> None:
    """A mode moves the zeros and nothing else."""
    values = _half_zero_monthly()

    classic = _spi(values, distribution)
    moved = _spi(values, distribution, zero_handling=zero_handling)

    positive = values.flatten() > 0
    np.testing.assert_array_equal(moved[positive], classic[positive])
    assert np.all(moved[~positive] < classic[~positive])


@pytest.mark.parametrize("compute_index", [indices.spi, indices.standardized_index])
@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
@pytest.mark.parametrize("zero_handling", ["center_of_mass", "mean_zero"])
def test_each_steps_zeros_take_the_score_of_that_steps_zero_mass(
    compute_index: Callable[..., np.ndarray], distribution: indices.Distribution, zero_handling: str
) -> None:
    """Month m's zeros score the closed form for month m's own p0 = 2 * (m + 1) / 40."""
    values = _stepped_zero_monthly()

    computed = compute_index(
        values.flatten(), 1, distribution, 1981, 1981, 2020, _MONTHLY, zero_handling=zero_handling
    ).reshape(values.shape)

    expected = [_zero_score(probability, zero_handling) for probability in _STEPPED_PROBABILITIES_OF_ZERO]
    _assert_each_months_zeros(computed, values, np.array(expected))


# supplied Pearson Type III parameters whose lower bound, loc - 2 * scale / skew = -15,
# sits below zero, so no support-limit mask reaches the zeros
_PEARSON_PARAMETERS = {
    "prob_zero": _STEPPED_PROBABILITIES_OF_ZERO,
    "loc": np.full(12, 5.0),
    "scale": np.full(12, 10.0),
    "skew": np.full(12, 1.0),
}


@pytest.mark.parametrize("zero_handling", [None, "classic", "center_of_mass", "mean_zero"])
def test_the_transforms_and_fit_and_standardize_score_each_steps_zeros(zero_handling: str | None) -> None:
    """
    The compute transforms and fit_and_standardize, Pearson without the fall back
    included, place each step's zeros by the mode, and default to the classic
    Φ⁻¹(p0) when no mode is given.
    """
    values = _stepped_zero_monthly()
    mode: dict[str, Any] = {} if zero_handling is None else {"zero_handling": zero_handling}
    arguments = (1981, 1981, 2020, _MONTHLY)
    pearson = _PEARSON_PARAMETERS
    gamma = indices.Distribution.gamma

    results = (
        compute.transform_fitted_gamma(values, *arguments, **mode),
        compute.fit_and_standardize(values, gamma, *arguments, **mode),
        compute.transform_fitted_pearson(
            values, *arguments, pearson["prob_zero"], pearson["loc"], pearson["scale"], pearson["skew"], **mode
        ),
        compute.fit_and_standardize(values, indices.Distribution.pearson, *arguments, pearson, **mode),
    )

    expected = [_zero_score(probability, zero_handling or "classic") for probability in _STEPPED_PROBABILITIES_OF_ZERO]
    for result in results:
        _assert_each_months_zeros(result, values, np.array(expected))


@pytest.mark.parametrize("zero_handling", ["center_of_mass", "mean_zero"])
def test_pearson_moves_trace_values_with_the_exact_zeros(zero_handling: str) -> None:
    """
    Pearson Type III traces below 0.0005 share the zero score where p0 > 0, although p0
    counts only exact zeros (ADR-0015, decision 2).
    """
    values = _positive_monthly(40, seed=5)
    values[::4, :] = 0.0
    values[2::4, :] = 0.0003

    computed = _spi(values, indices.Distribution.pearson, zero_handling=zero_handling).reshape(values.shape)

    # 10 exact zeros in each month's 40 calibration values; the 10 traces are not zeros
    expected = _zero_score(0.25, zero_handling)
    np.testing.assert_allclose(computed[values == 0.0], expected, atol=1e-12)
    np.testing.assert_allclose(computed[values == 0.0003], expected, atol=1e-12)


def test_pearson_mode_overrides_the_support_limit_mask_at_zeros() -> None:
    """
    A classic zero below the fitted lower bound keeps the support-limit probability, while
    a non-classic mode gives it the mode's score; other values below the bound keep the mask.
    """
    values = np.tile(np.array([[0.0], [10.0], [60.0], [80.0]]), (1, 12))
    # loc - 2 * scale / skew puts the distribution's lower bound at 30
    parameters = {
        "probabilities_of_zero": np.full(12, 0.25),
        "locs": np.full(12, 50.0),
        "scales": np.full(12, 10.0),
        "skews": np.full(12, 1.0),
    }

    results = {
        zero_handling: compute.transform_fitted_pearson(
            values.copy(), 2000, 2000, 2003, _MONTHLY, **parameters, zero_handling=zero_handling
        )
        for zero_handling in _MODES
    }

    below_bound = float(scipy.stats.norm.ppf(0.25 + 0.75 * 0.0005))
    np.testing.assert_allclose(results["classic"][0], below_bound)
    np.testing.assert_allclose(results["center_of_mass"][0], _zero_score(0.25, "center_of_mass"))
    np.testing.assert_allclose(results["mean_zero"][0], _zero_score(0.25, "mean_zero"))
    for zero_handling in _MODES:
        np.testing.assert_array_equal(results[zero_handling][1:], results["classic"][1:])


# ---------------------------------------------------------------------------
# edge cases and clipping


@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
def test_every_mode_keeps_the_classic_result_where_the_calibration_has_no_zeros(
    distribution: indices.Distribution,
) -> None:
    """With p0 == 0, zeros outside the calibration period score as in classic (decision 5)."""
    values = _positive_monthly(40, seed=3)
    values[0, :] = 0.0

    results = [_spi(values, distribution, calibration=(1991, 2020), zero_handling=mode) for mode in _MODES]

    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    np.testing.assert_array_equal(results[0][:12], np.full(12, -3.09))


@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
def test_every_mode_keeps_the_classic_result_for_an_all_zero_calibration_step(
    distribution: indices.Distribution,
) -> None:
    """
    With every calibration value zero no distribution can be fitted, and the existing
    handling is kept (decision 5). Gamma resets that step's p0 to 0 and Pearson's
    minimum-non-zero guard zeroes its parameters, so the placement's own ``p0 == 1``
    guard is not reached here; a supplied Pearson ``prob_zero`` of 1 covers it below.
    """
    values = _positive_monthly(40, seed=4)
    values[:, 0] = 0.0

    results = [_spi(values, distribution, zero_handling=mode) for mode in _MODES]

    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    np.testing.assert_array_equal(results[0].reshape(values.shape)[:, 0], np.full(40, -3.09))


@pytest.mark.parametrize(("zero_handling", "unclipped"), [("center_of_mass", -3.29), ("mean_zero", -3.37)])
def test_a_moved_zero_is_clipped_like_every_other_value(zero_handling: str, unclipped: float) -> None:
    """Below p0 of about 0.002 a moved zero passes -3.09, and the clip still applies (decision 6)."""
    values = _positive_monthly(40, seed=6)
    values[0, :] = 0.0
    alphas, betas = compute.gamma_parameters(values, 1981, 1981, 2020, _MONTHLY)
    parameters = {"alpha": alphas, "beta": betas, "prob_zero": np.full(12, 0.001)}

    transformed = compute.transform_fitted_gamma(
        values, 1981, 1981, 2020, _MONTHLY, alphas, betas, parameters["prob_zero"], zero_handling=zero_handling
    )
    computed = _spi(values, indices.Distribution.gamma, fitting_params=parameters, zero_handling=zero_handling)

    np.testing.assert_allclose(transformed[0], _zero_score(0.001, zero_handling))
    np.testing.assert_allclose(transformed[0], unclipped, atol=0.005)
    np.testing.assert_array_equal(computed[:12], np.full(12, -3.09))


# ---------------------------------------------------------------------------
# the probability scales


@pytest.mark.parametrize("output_scale", ["probability", "bounded"])
@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
@pytest.mark.parametrize("zero_handling", _MODES)
def test_the_probability_scales_place_a_moved_zero_at_the_centre_of_the_zero_mass(
    zero_handling: str, distribution: indices.Distribution, output_scale: str
) -> None:
    """
    On the probability scales a zero maps to p0 under "classic" and to p0 / 2 under both
    other modes, "mean_zero" included (ADR-0015, decision 7); other values keep their
    classic probability.
    """
    values = _half_zero_monthly()

    classic = _spi(values, distribution, output_scale=output_scale)
    computed = _spi(values, distribution, output_scale=output_scale, zero_handling=zero_handling)

    zero_probability = 0.5 if zero_handling == "classic" else 0.25
    expected = zero_probability if output_scale == "probability" else (2.0 * zero_probability) - 1.0
    zeros = values.flatten() == 0
    np.testing.assert_allclose(computed[zeros], expected, atol=1e-12)
    np.testing.assert_array_equal(computed[~zeros], classic[~zeros])


# ---------------------------------------------------------------------------
# argument handling


_INVALID_MODE_CALLS: dict[str, Callable[[str], Any]] = {
    "spi": lambda mode: _spi(_half_zero_monthly(), indices.Distribution.gamma, zero_handling=mode),
    "standardized_index": lambda mode: indices.standardized_index(
        _half_zero_monthly().flatten(), 1, indices.Distribution.pearson, 1981, 1981, 2020, _MONTHLY, zero_handling=mode
    ),
    "fit_and_standardize": lambda mode: compute.fit_and_standardize(
        _half_zero_monthly(), indices.Distribution.gamma, 1981, 1981, 2020, _MONTHLY, zero_handling=mode
    ),
    "spi_all_missing": lambda mode: _spi(np.full((40, 12), np.nan), indices.Distribution.gamma, zero_handling=mode),
    "standardized_index_all_missing": lambda mode: indices.standardized_index(
        np.full(480, np.nan), 1, indices.Distribution.gamma, 1981, 1981, 2020, _MONTHLY, zero_handling=mode
    ),
    "transform_fitted_gamma": lambda mode: compute.transform_fitted_gamma(
        np.full((40, 12), np.nan), 1981, 1981, 2020, _MONTHLY, zero_handling=mode
    ),
    "transform_fitted_pearson": lambda mode: compute.transform_fitted_pearson(
        np.full((40, 12), np.nan), 1981, 1981, 2020, _MONTHLY, zero_handling=mode
    ),
}


@pytest.mark.parametrize("call", list(_INVALID_MODE_CALLS.values()), ids=list(_INVALID_MODE_CALLS))
@pytest.mark.parametrize("mode", ["mean", "Classic", None])
def test_an_unknown_mode_raises_value_error_naming_the_modes(call: Callable[[str], Any], mode: Any) -> None:
    """
    Any value other than the three modes raises. The all-missing cases check that the
    public entry points and the transforms validate the mode before their all-missing
    shortcut.
    """
    with pytest.raises(ValueError, match="'classic', 'center_of_mass', or 'mean_zero'"):
        call(mode)


@pytest.mark.parametrize("index", ["spei", "eddi"])
def test_indices_without_zero_mass_do_not_take_the_parameter(index: str) -> None:
    """SPEI and EDDI keep the existing unknown-keyword TypeError (decision 3)."""
    values = _positive_monthly(40, seed=1).flatten()
    arguments: dict[str, tuple[Any, ...]] = {
        "spei": (values, values / 2.0, 1, indices.Distribution.gamma, _MONTHLY, 1981, 1981, 2020),
        "eddi": (values, 1, 1981, 1981, 2020, _MONTHLY),
    }
    compute_index = getattr(indices, index)
    index_arguments = arguments[index]
    with pytest.raises(TypeError, match="zero_handling"):
        compute_index(*index_arguments, zero_handling="mean_zero")


# ---------------------------------------------------------------------------
# input layouts


@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
@pytest.mark.parametrize("zero_handling", _MODES)
def test_every_layout_gives_the_series_result(distribution: indices.Distribution, zero_handling: str) -> None:
    """1-D, legacy 2-D (years, periods), and time-major spatial input agree for every mode."""
    rng = np.random.default_rng(21)
    cells = np.stack([_positive_monthly(40, seed) for seed in range(6)], axis=-1)
    cells[rng.random(cells.shape) < 0.3] = 0.0
    block = cells.reshape(480, 2, 3)

    spatial = indices.spi(
        block, 1, distribution, 1981, 1991, 2020, _MONTHLY, spatial_time_major=True, zero_handling=zero_handling
    )
    for cell in range(6):
        series = cells[..., cell]
        expected = _spi(series, distribution, calibration=(1991, 2020), zero_handling=zero_handling)
        legacy = indices.spi(series, 1, distribution, 1981, 1991, 2020, _MONTHLY, zero_handling=zero_handling)
        np.testing.assert_array_equal(legacy, expected)
        np.testing.assert_allclose(spatial.reshape(480, 6)[:, cell], expected, atol=1e-8, rtol=1e-7)


def test_a_period_only_zero_mass_broadcasts_across_a_spatial_block() -> None:
    """A supplied (periods,) gamma prob_zero applies to every cell, as alpha and beta do."""
    cells = np.stack([_half_zero_monthly(seed=seed) for seed in range(4)], axis=-1)
    parameters = {"prob_zero": np.linspace(0.1, 0.32, 12)}

    spatial = indices.spi(
        cells.reshape(480, 2, 2),
        1,
        indices.Distribution.gamma,
        1981,
        1981,
        2020,
        _MONTHLY,
        parameters,
        spatial_time_major=True,
        zero_handling="mean_zero",
    ).reshape(480, 4)

    for cell in range(4):
        expected = _spi(
            cells[..., cell], indices.Distribution.gamma, fitting_params=parameters, zero_handling="mean_zero"
        )
        np.testing.assert_allclose(spatial[:, cell], expected, atol=1e-8, rtol=1e-7)
        _assert_each_months_zeros(
            spatial[:, cell].reshape(40, 12),
            cells[..., cell],
            np.array([_zero_score(probability, "mean_zero") for probability in parameters["prob_zero"]]),
        )


# ---------------------------------------------------------------------------
# the calibration-period gamma zero mass (ADR-0015, decision 4)


def _pre_adr_gamma_spi(values: np.ndarray, scale: int, first_year: int, last_year: int) -> np.ndarray:
    """Gamma SPI with the pre-ADR whole-record zero mass supplied through ``fitting_params``."""
    scaled = compute.prepare_scaled(values, scale, _MONTHLY)
    return indices.spi(
        values,
        scale,
        indices.Distribution.gamma,
        first_year,
        first_year,
        last_year,
        _MONTHLY,
        {"prob_zero": _whole_record_probabilities_of_zero(scaled)},
    )


@pytest.mark.parametrize("series", ["spi_fixture", "spi_fixture_with_zeros", "zero_inflated", "half_zero"])
def test_classic_gamma_is_unchanged_for_complete_years_at_scale_one(
    precips_mm_monthly, data_year_start_monthly, series: str
) -> None:
    """
    Classic gamma SPI and standardized_index are bit-identical to the whole-record zero
    mass when the calibration period is the full record of complete years, at scale 1,
    with no value missing and no ``prob_zero`` supplied.

    The zero mass is the only input of the transform that ADR-0015 changes, so the
    pre-ADR output is the transform given that whole-record zero mass. The tests
    below pin the cases outside those conditions, where the output moves.
    """
    scale = 1
    # the SPI fixture's final year is padded after February, so keep the complete years
    fixture = np.asarray(precips_mm_monthly)[:-1]
    first_year = data_year_start_monthly
    values = {
        "spi_fixture": fixture,
        "spi_fixture_with_zeros": np.where(np.arange(fixture.size).reshape(fixture.shape) % 7 == 0, 0.0, fixture),
        "zero_inflated": np.where(np.random.default_rng(2).random((60, 12)) < 0.4, 0.0, _positive_monthly(60, 9)),
        "half_zero": _half_zero_monthly(),
    }[series].flatten()
    if series in ("zero_inflated", "half_zero"):
        first_year = 1981
    last_year = first_year + values.size // 12 - 1

    expected = _pre_adr_gamma_spi(values, scale, first_year, last_year)
    for compute_index in (indices.spi, indices.standardized_index):
        computed = compute_index(
            values,
            scale,
            indices.Distribution.gamma,
            first_year,
            first_year,
            last_year,
            _MONTHLY,
            zero_handling="classic",
        )
        np.testing.assert_array_equal(computed, expected)


def test_classic_spi_matches_the_committed_fixtures(
    precips_mm_monthly, data_year_start_monthly, data_year_end_monthly, spi_1_month_gamma, spi_6_month_gamma
) -> None:
    """The committed full-record SPI fixtures still hold under an explicit classic mode."""
    for scale, fixture in ((1, spi_1_month_gamma), (6, spi_6_month_gamma)):
        computed = indices.spi(
            precips_mm_monthly.flatten(),
            scale,
            indices.Distribution.gamma,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
            _MONTHLY,
            zero_handling="classic",
        )
        np.testing.assert_allclose(computed, fixture, atol=0.001, equal_nan=True)


@pytest.mark.parametrize("compute_index", [indices.spi, indices.standardized_index])
def test_a_shorter_calibration_changes_classic_gamma_through_the_zero_mass(
    compute_index: Callable[..., np.ndarray],
) -> None:
    """
    Classic gamma output takes its zero mass from the calibration period's non-missing
    values, so a window whose zero fraction differs from the record's moves every value.
    """
    values = _positive_monthly(40, seed=8)
    values[:10:2, :] = 0.0  # five zeros per month before the 1991-2020 calibration period
    values[12::6, :] = 0.0  # five zeros per month inside it
    values[14, :] = np.nan  # a missing calibration year shrinks the denominator
    arguments = (values.flatten(), 1, indices.Distribution.gamma, 1981, 1991, 2020, _MONTHLY)

    computed = compute_index(*arguments)

    calibration = {"prob_zero": _calibration_probabilities_of_zero(values, 1991, 2020)}
    whole_record = {"prob_zero": _whole_record_probabilities_of_zero(values)}
    np.testing.assert_allclose(calibration["prob_zero"], np.full(12, 5.0 / 29.0))
    np.testing.assert_array_equal(computed, compute_index(*arguments, calibration))
    positive = values.flatten() > 0
    assert np.all(computed[positive] != compute_index(*arguments, whole_record)[positive])


def test_a_shorter_calibration_changes_classic_gamma_spei_through_an_exact_zero() -> None:
    """
    SPEI shares the gamma transform: an exact zero in its offset P - PET series outside
    the calibration period no longer carries a zero mass (ADR-0015, consequences).
    """
    precips = np.round(_positive_monthly(40, seed=12))
    pet = np.full(precips.shape, 30.0)
    # P - PET + 1000 is exactly zero in 1981, before the 1991-2020 calibration period
    pet[0, :] = precips[0, :] + 1000.0

    computed = indices.spei(precips.flatten(), pet.flatten(), 1, indices.Distribution.gamma, _MONTHLY, 1981, 1991, 2020)

    offset = precips - pet + 1000.0
    assert np.all(offset[0] == 0.0)
    expected = compute.transform_fitted_gamma(offset, 1981, 1991, 2020, _MONTHLY, probabilities_of_zero=np.zeros(12))
    np.testing.assert_array_equal(computed, np.clip(expected, -3.09, 3.09).flatten())
    np.testing.assert_array_equal(computed[:12], np.full(12, -3.09))

    # the pre-ADR whole-record mass of 1/40 scored these zeros at the top of that mass
    pre_adr = compute.transform_fitted_gamma(
        offset, 1981, 1991, 2020, _MONTHLY, probabilities_of_zero=np.full(12, 1.0 / 40.0)
    )
    np.testing.assert_allclose(pre_adr[0], scipy.stats.norm.ppf(1.0 / 40.0))


@pytest.mark.parametrize("zero_handling", _MODES)
def test_the_pearson_fall_back_uses_the_calibration_zero_mass_and_the_mode(zero_handling: str) -> None:
    """A failed Pearson Type III fit falls back to the gamma transform the gamma index runs."""
    values = _positive_monthly(40, seed=13)
    values[:10:2, :] = 0.0
    values[12::6, :] = 0.0
    failed_pearson = mock.patch(
        "climate_indices.compute.transform_fitted_pearson",
        side_effect=compute.DistributionFittingError("Pearson failed", distribution_name="pearson3"),
    )

    with failed_pearson:
        fallen_back = _spi(values, indices.Distribution.pearson, calibration=(1991, 2020), zero_handling=zero_handling)
    gamma = _spi(values, indices.Distribution.gamma, calibration=(1991, 2020), zero_handling=zero_handling)

    np.testing.assert_array_equal(fallen_back, gamma)
    np.testing.assert_allclose(
        fallen_back.reshape(values.shape)[values == 0], _zero_score(5.0 / 30.0, zero_handling), atol=1e-12
    )


def test_a_gamma_zero_mass_is_read_from_fitting_params_under_either_key() -> None:
    """A gamma ``prob_zero`` (or its deprecated alias) replaces the fitted zero mass."""
    values = _half_zero_monthly()

    canonical = _spi(values, indices.Distribution.gamma, fitting_params={"prob_zero": np.full(12, 0.3)})
    deprecated = _spi(values, indices.Distribution.gamma, fitting_params={"probabilities_of_zero": np.full(12, 0.3)})

    np.testing.assert_array_equal(canonical, deprecated)
    np.testing.assert_allclose(canonical[values.flatten() == 0], _zero_score(0.3, "classic"), atol=1e-12)


def test_the_leading_steps_a_scale_above_one_leaves_missing_leave_the_denominator() -> None:
    """An SPI-3 January zero mass counts the 39 January sums, not the 40 years."""
    values = _positive_monthly(40, seed=19)
    # nine January sums of November, December, and January are zero, in years 2 to 10
    for year in range(1, 10):
        values[year - 1, 10:] = 0.0
        values[year, 0] = 0.0

    computed = indices.spi(values.flatten(), 3, indices.Distribution.gamma, 1981, 1981, 2020, _MONTHLY)

    january = computed.reshape(values.shape)[:, 0]
    assert np.isnan(january[0])
    np.testing.assert_allclose(january[1:10], scipy.stats.norm.ppf(9.0 / 39.0), atol=1e-12)
    assert scipy.stats.norm.ppf(9.0 / 39.0) != pytest.approx(scipy.stats.norm.ppf(9.0 / 40.0))


def test_the_padded_final_year_leaves_the_denominator() -> None:
    """A record ending in June pads July with NaN, so July's zero mass counts 39 years."""
    values = _positive_monthly(40, seed=23)
    values[:9, 6] = 0.0
    series = values.flatten()[:-6]

    computed = indices.spi(series, 1, indices.Distribution.gamma, 1981, 1981, 2020, _MONTHLY)

    july = np.append(computed, np.full(6, np.nan)).reshape(values.shape)[:, 6]
    np.testing.assert_allclose(july[:9], scipy.stats.norm.ppf(9.0 / 39.0), atol=1e-12)


def test_a_gamma_prob_zero_key_that_was_ignored_now_sets_the_zero_mass() -> None:
    """A gamma fitting_params that already carried prob_zero now changes every value."""
    values = _half_zero_monthly()
    alphas, betas = compute.gamma_parameters(values, 1981, 1981, 2020, _MONTHLY)

    without_key = _spi(values, indices.Distribution.gamma, fitting_params={"alpha": alphas, "beta": betas})
    with_key = _spi(
        values,
        indices.Distribution.gamma,
        fitting_params={"alpha": alphas, "beta": betas, "prob_zero": np.full(12, 0.3)},
    )

    positive = values.flatten() > 0
    assert np.all(with_key[positive] != without_key[positive])


# ---------------------------------------------------------------------------
# a supplied gamma zero mass


@pytest.mark.parametrize("probability_of_zero", [np.full((12, 3), 0.2), np.full((12, 1, 1), 0.2), np.full(11, 0.2)])
def test_a_supplied_gamma_zero_mass_with_the_wrong_cells_is_rejected(probability_of_zero: np.ndarray) -> None:
    """The gamma prob_zero follows the Pearson rule for cell dimensions, rather than broadcasting."""
    block = np.stack([_half_zero_monthly(seed=seed).flatten() for seed in range(3)], axis=-1).reshape(480, 1, 3)
    parameters = {"prob_zero": probability_of_zero}

    with pytest.raises(ValueError, match="prob_zero"):
        indices.spi(
            block, 1, indices.Distribution.gamma, 1981, 1981, 2020, _MONTHLY, parameters, spatial_time_major=True
        )


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_a_supplied_gamma_shape_or_scale_with_the_wrong_cells_is_rejected(name: str) -> None:
    """A (periods, cells) alpha or beta that misses the block's cells raises ValueError, not IndexError."""
    block = np.stack([_positive_monthly(40, seed).flatten() for seed in range(3)], axis=-1).reshape(480, 1, 3)
    parameters = {"alpha": np.full((12, 1, 3), 2.0), "beta": np.full((12, 1, 3), 30.0)}
    parameters[name] = np.full((12, 3), parameters[name].flat[0])
    arguments = (1981, 1981, 2020, _MONTHLY)

    with pytest.raises(ValueError, match=f"'{name}' has shape \\(12, 3\\)"):
        indices.spi(block, 1, indices.Distribution.gamma, *arguments, parameters, spatial_time_major=True)
    with pytest.raises(ValueError, match=f"'{name}' has shape \\(12, 3\\)"):
        compute.fit_diagnostics(block.reshape(40, 12, 1, 3), indices.Distribution.gamma, *arguments, parameters)


@pytest.mark.parametrize("probability_of_zero", [1.2, -0.1])
def test_a_supplied_gamma_zero_mass_outside_the_unit_interval_is_rejected(probability_of_zero: float) -> None:
    """A probability outside [0, 1] is an argument error, not a silent NaN."""
    values = _half_zero_monthly()
    parameters = {"prob_zero": np.full(12, probability_of_zero)}

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _spi(values, indices.Distribution.gamma, fitting_params=parameters)


def test_only_a_supplied_gamma_zero_mass_of_exactly_one_is_reset() -> None:
    """A supplied 0.99999 is a zero mass; a supplied 1 means no gamma fit, as a computed 1 does."""
    values = _half_zero_monthly()
    arguments = (values, 1981, 1981, 2020, _MONTHLY)

    near_one = compute.transform_fitted_gamma(*arguments, probabilities_of_zero=np.full(12, 0.99999))
    one = compute.transform_fitted_gamma(*arguments, probabilities_of_zero=np.ones(12))

    np.testing.assert_allclose(near_one[values == 0], scipy.stats.norm.ppf(0.99999))
    np.testing.assert_array_equal(one[values == 0], -np.inf)


def test_an_undefined_zero_mass_leaves_its_zeros_missing() -> None:
    """
    A step without calibration data has no zero mass, so its zeros are NaN rather than
    an extreme drought, while its non-zero values keep supplied parameters' scores.
    A supplied NaN means the same, so fit_diagnostics' report round-trips.
    """
    complete = _positive_monthly(40, seed=29)
    alphas, betas = compute.gamma_parameters(complete, 1981, 1981, 2020, _MONTHLY)
    values = complete.copy()
    values[10:, 0] = np.nan  # no January value inside the 1991-2020 calibration period
    values[:5, 0] = 0.0
    arguments = (values, 1981, 1991, 2020, _MONTHLY)

    fitted = compute.transform_fitted_gamma(*arguments)
    supplied = compute.transform_fitted_gamma(*arguments, alphas, betas)
    diagnostics = compute.fit_diagnostics(values, indices.Distribution.gamma, 1981, 1991, 2020, _MONTHLY)
    round_trip = compute.transform_fitted_gamma(
        *arguments, diagnostics.parameters["alpha"], diagnostics.parameters["beta"], diagnostics.parameters["prob_zero"]
    )

    probability = compute.transform_fitted_gamma(*arguments, output_scale="probability")

    assert np.all(np.isnan(fitted[:5, 0]))
    assert np.all(np.isnan(probability[:5, 0]))
    assert np.all(np.isnan(supplied[:5, 0]))
    assert np.all(np.isfinite(supplied[5:10, 0]))
    assert np.isnan(diagnostics.prob_zero[0])
    np.testing.assert_array_equal(round_trip, fitted)


@pytest.mark.parametrize("zero_handling", ["center_of_mass", "mean_zero"])
def test_a_negative_value_passed_to_the_gamma_transform_is_placed_with_the_zeros(zero_handling: str) -> None:
    """The index functions clip negatives, and a direct caller's negative ranks with the zeros."""
    values = _half_zero_monthly()
    values[0, :] = -1.0

    transformed = compute.transform_fitted_gamma(values, 1981, 1981, 2020, _MONTHLY, zero_handling=zero_handling)

    # the negative row displaces a zero, leaving 19 zeros in each month's 40 values
    np.testing.assert_allclose(transformed[0], _zero_score(19.0 / 40.0, zero_handling), atol=1e-12)
    np.testing.assert_array_equal(transformed[0], transformed[2])


def test_a_masked_calibration_value_is_missing() -> None:
    """A masked zero is neither counted in the zero mass nor scored."""
    values = _half_zero_monthly()
    mask = np.zeros(values.shape, dtype=bool)
    mask[:4, :] = True  # two of each month's zeros, and two non-zero values

    masked = compute.transform_fitted_gamma(np.ma.array(values, mask=mask), 1981, 1981, 2020, _MONTHLY)

    assert np.all(np.isnan(masked[:4]))
    # 18 zeros among each month's 36 non-missing calibration values
    np.testing.assert_allclose(masked[4::2], scipy.stats.norm.ppf(18.0 / 36.0), atol=1e-12)


def test_a_calibration_period_outside_the_record_uses_the_full_record() -> None:
    """Calibration years beyond the data fall back to the record for the zero mass too."""
    values = _stepped_zero_monthly()

    outside = indices.spi(values.flatten(), 1, indices.Distribution.gamma, 1981, 1900, 2100, _MONTHLY)

    np.testing.assert_array_equal(outside, _spi(values, indices.Distribution.gamma))
    _assert_each_months_zeros(
        outside.reshape(values.shape), values, scipy.stats.norm.ppf(_STEPPED_PROBABILITIES_OF_ZERO)
    )


def test_every_mode_keeps_the_classic_result_for_a_supplied_pearson_zero_mass_of_one() -> None:
    """The p0 == 1 guard, reachable for Pearson only through supplied parameters (decision 5)."""
    values = _stepped_zero_monthly()
    parameters = dict(_PEARSON_PARAMETERS, prob_zero=np.where(np.arange(12) == 0, 1.0, _STEPPED_PROBABILITIES_OF_ZERO))

    results = [
        compute.fit_and_standardize(
            values, indices.Distribution.pearson, 1981, 1981, 2020, _MONTHLY, parameters, zero_handling=mode
        )
        for mode in _MODES
    ]

    for result in results[1:]:
        np.testing.assert_array_equal(result[:, 0], results[0][:, 0])


def test_fit_diagnostics_round_trips_a_supplied_gamma_zero_mass() -> None:
    """
    A supplied gamma prob_zero is reported and returned in parameters, so feeding the
    parameters back reproduces the fit; a Pearson fall back computes its own zero mass.
    """
    values = _stepped_zero_monthly(50)
    supplied = {"prob_zero": np.linspace(0.05, 0.3, 12)}
    arguments = (1981, 1981, 2010, _MONTHLY)

    diagnostics = compute.fit_diagnostics(values, indices.Distribution.gamma, *arguments, supplied)

    np.testing.assert_array_equal(diagnostics.prob_zero, supplied["prob_zero"])
    np.testing.assert_array_equal(
        compute.fit_and_standardize(values, indices.Distribution.gamma, *arguments, diagnostics.parameters),
        compute.fit_and_standardize(values, indices.Distribution.gamma, *arguments, supplied),
    )

    failed_pearson = mock.patch(
        "climate_indices.compute.transform_fitted_pearson",
        side_effect=compute.DistributionFittingError("Pearson failed", distribution_name="pearson3"),
    )
    with failed_pearson:
        fallen_back = compute.fit_diagnostics(
            values, indices.Distribution.pearson, *arguments, _PEARSON_PARAMETERS, fallback_to_gamma=True
        )

    assert fallen_back.fell_back_to_gamma
    np.testing.assert_array_equal(fallen_back.prob_zero, _calibration_probabilities_of_zero(values, 1981, 2010))
    np.testing.assert_array_equal(fallen_back.parameters["prob_zero"], fallen_back.prob_zero)

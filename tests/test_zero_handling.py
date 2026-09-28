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
    """With p0 == 1 no distribution can be fitted, and the existing handling is kept (decision 5)."""
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
# argument handling


_INVALID_MODE_CALLS: dict[str, Callable[[str], Any]] = {
    "spi": lambda mode: _spi(_half_zero_monthly(), indices.Distribution.gamma, zero_handling=mode),
    "standardized_index": lambda mode: indices.standardized_index(
        _half_zero_monthly().flatten(), 1, indices.Distribution.pearson, 1981, 1981, 2020, _MONTHLY, zero_handling=mode
    ),
    "fit_and_standardize": lambda mode: compute.fit_and_standardize(
        _half_zero_monthly(), indices.Distribution.gamma, 1981, 1981, 2020, _MONTHLY, zero_handling=mode
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
    """Any value other than the three modes raises, even for input with nothing to transform."""
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
    parameters = {"prob_zero": np.full(12, 0.2)}

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
    np.testing.assert_allclose(spatial[cells.reshape(480, 4) == 0], _zero_score(0.2, "mean_zero"), atol=1e-12)


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


@pytest.mark.parametrize(
    ("series", "scale"),
    [
        ("spi_fixture", 1),
        ("spi_fixture", 6),
        ("spi_fixture_with_zeros", 1),
        ("zero_inflated", 1),
        ("half_zero", 1),
    ],
)
def test_classic_gamma_is_unchanged_for_a_full_record_calibration_without_missing_values(
    precips_mm_monthly, data_year_start_monthly, series: str, scale: int
) -> None:
    """
    Classic gamma SPI and standardized_index are bit-identical to the whole-record zero
    mass when the calibration period is the full record and no value is missing.

    The zero mass is the only input of the transform that ADR-0015 changes, so the
    pre-ADR output is the transform given that whole-record zero mass.
    """
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

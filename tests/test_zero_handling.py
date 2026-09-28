"""Zero-placement modes for the gamma and Pearson transforms (ADR-0015, issue #1186)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from scipy import stats

from climate_indices import compute, indices
from climate_indices.compute import _replace_zeros_with_nan

_MODES = ("classic", "center_of_mass", "mean_zero")


def _expected_scores(probabilities_of_zero: np.ndarray, mode: str) -> np.ndarray:
    """The closed-form zero score per mode, clipped to the fitted-index range."""
    quantile = stats.norm.ppf(probabilities_of_zero)
    if mode == "classic":
        scores = quantile
    elif mode == "center_of_mass":
        scores = stats.norm.ppf(probabilities_of_zero / 2.0)
    else:
        scores = -stats.norm.pdf(quantile) / probabilities_of_zero
    return np.clip(scores, -3.09, 3.09)


def _legacy_probabilities_of_zero(values: np.ndarray) -> np.ndarray:
    """The pre-ADR gamma zero mass: zeros over every row of the record."""
    zero_mask, _ = _replace_zeros_with_nan(values)
    probabilities = zero_mask.sum(axis=0) / values.shape[0]
    probabilities[np.isclose(probabilities, 1.0)] = 0.0
    return probabilities


def _window_probabilities_of_zero(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
) -> np.ndarray:
    """The ADR-0015 zero mass: calibration-window zeros over the window's observations."""
    zero_mask, _ = _replace_zeros_with_nan(values)
    data_end_year = data_start_year + values.shape[0] - 1
    calibration_start_year = max(calibration_start_year, data_start_year)
    calibration_end_year = min(calibration_end_year, data_end_year)
    begin = max(calibration_start_year - data_start_year, 0)
    end = max((calibration_end_year - data_start_year) + 1, 0)
    window = values[begin:end, ...]
    zeros = zero_mask[begin:end, ...].sum(axis=0)
    non_missing = np.count_nonzero(~np.isnan(window), axis=0)
    probabilities = np.where(non_missing > 0, zeros / np.maximum(non_missing, 1), 0.0)
    probabilities[np.isclose(probabilities, 1.0)] = 0.0
    return probabilities


def _monthly_precip() -> np.ndarray:
    """40 years of monthly precipitation with a realistic zero mass."""
    rng = np.random.default_rng(20260928)
    values = rng.gamma(0.8, 7.0, size=(40, 12))
    values[values < 4.0] = 0.0
    return values


@pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
@pytest.mark.parametrize("mode", _MODES)
def test_zero_handling_modes_place_zeros_in_closed_form(distribution: indices.Distribution, mode: str) -> None:
    """Each mode lands every zero on its closed-form score and leaves the rest alone."""
    scaled = compute.prepare_scaled(_monthly_precip(), 1, compute.Periodicity.monthly)

    classic = compute.fit_and_standardize(scaled.copy(), distribution, 1980, 1980, 2019, compute.Periodicity.monthly)
    computed = compute.fit_and_standardize(
        scaled.copy(),
        distribution,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        zero_handling=mode,
    )

    if distribution is indices.Distribution.gamma:
        zero_mask, _ = _replace_zeros_with_nan(scaled)
        probabilities_of_zero = _window_probabilities_of_zero(scaled, 1980, 1980, 2019)
    else:
        probabilities_of_zero, _, _, _ = compute.pearson_parameters(
            scaled, 1980, 1980, 2019, compute.Periodicity.monthly
        )
        zero_mask = np.logical_and(scaled < 0.0005, probabilities_of_zero > 0.0)

    expected = np.broadcast_to(_expected_scores(probabilities_of_zero, mode), scaled.shape)
    np.testing.assert_allclose(computed[zero_mask], expected[zero_mask], rtol=1e-12)

    if mode == "classic":
        np.testing.assert_array_equal(computed, classic)
    else:
        # a mode moves the zeros, and nothing else
        np.testing.assert_array_equal(computed[~zero_mask], classic[~zero_mask])


def test_gamma_classic_is_legacy_for_a_full_record_calibration() -> None:
    """`classic` reproduces the pre-ADR transform when the window covers the record."""
    # scale 1 keeps the accumulated series free of the scaling pad, so the old and new
    # denominators count the same observations
    scaled = compute.prepare_scaled(_monthly_precip(), 1, compute.Periodicity.monthly)

    computed = compute.transform_fitted_gamma(scaled, 1980, 1980, 2019, compute.Periodicity.monthly)

    zero_mask, values_for_fitting = _replace_zeros_with_nan(scaled)
    probabilities_of_zero = _legacy_probabilities_of_zero(scaled)
    alphas, betas = compute.gamma_parameters(values_for_fitting, 1980, 1980, 2019, compute.Periodicity.monthly)
    gamma_probabilities = stats.gamma.cdf(values_for_fitting, a=alphas, scale=betas)
    gamma_probabilities[zero_mask] = 0.0
    legacy = stats.norm.ppf(probabilities_of_zero + (1 - probabilities_of_zero) * gamma_probabilities)

    np.testing.assert_array_equal(computed, legacy)


def test_gamma_zero_mass_uses_the_calibration_window() -> None:
    """Gamma's zero mass is counted over the calibration window (ADR-0015 decision 4)."""
    rng = np.random.default_rng(20260928)
    values = rng.gamma(0.8, 7.0, size=(40, 12)) + 1.0
    values[0:5, :] = 0.0  # zeros only in 1980-1984, outside the 1990-2019 window

    scaled = compute.prepare_scaled(values, 1, compute.Periodicity.monthly)
    legacy = _legacy_probabilities_of_zero(scaled)

    # a narrowed window with no zeros in it has no mass to place
    assert np.allclose(_window_probabilities_of_zero(scaled, 1980, 1990, 2019), 0.0)
    # the full record still counts the 1980s zeros, at the same fraction as before
    assert np.allclose(_window_probabilities_of_zero(scaled, 1980, 1980, 2019), legacy)

    # Hold zero mass fixed to isolate the fitted shape and scale. Nonzero values from
    # 1985-1989 remain in the full fit but not in the narrowed fit.
    shared_zero_mass = _window_probabilities_of_zero(scaled, 1980, 1990, 2019)
    narrowed = compute.transform_fitted_gamma(
        scaled, 1980, 1990, 2019, compute.Periodicity.monthly, probabilities_of_zero=shared_zero_mass
    )
    assert np.all(narrowed[0:5, :] == -np.inf)

    full = compute.transform_fitted_gamma(
        scaled, 1980, 1980, 2019, compute.Periodicity.monthly, probabilities_of_zero=shared_zero_mass
    )
    # Compare finite nonzero scores; -inf zero positions would make any mismatch trivial.
    nonzero = (scaled != 0.0) & np.isfinite(full) & np.isfinite(narrowed)
    assert nonzero.any()
    assert not np.allclose(narrowed[nonzero], full[nonzero])


def test_gamma_calibration_window_follows_the_full_record_fallback() -> None:
    """An out-of-range calibration window falls back to the full record for p0 too."""
    scaled = compute.prepare_scaled(_monthly_precip(), 1, compute.Periodicity.monthly)

    full = compute.transform_fitted_gamma(scaled, 1980, 1980, 2019, compute.Periodicity.monthly)
    for calibration in ((1990, 2050), (2025, 2030), (1500, 1990)):
        computed = compute.transform_fitted_gamma(
            scaled, 1980, calibration[0], calibration[1], compute.Periodicity.monthly
        )
        np.testing.assert_array_equal(computed, full)


def test_gamma_zero_mass_excludes_missing_calibration_values() -> None:
    """The zero mass divisor is the window's non-missing values per calendar step."""
    rng = np.random.default_rng(20260928)
    values = rng.gamma(0.8, 7.0, size=(40, 12)) + 1.0
    values[0:4, 1] = 0.0  # four zero Februaries
    values[2:4, 1] = np.nan  # two of them also missing, so two zeros remain

    scaled = compute.prepare_scaled(values, 1, compute.Periodicity.monthly)
    windowed = _window_probabilities_of_zero(scaled, 1980, 1980, 2019)
    legacy = _legacy_probabilities_of_zero(scaled)

    # zeros per step are unchanged, only the divisor loses the missing values
    assert np.isclose(windowed[1], 2 / 38)
    assert np.isclose(legacy[1], 2 / 40)
    # a calendar step with no missing values at all is untouched
    assert np.isclose(windowed[2], legacy[2])

    computed = compute.transform_fitted_gamma(scaled, 1980, 1980, 2019, compute.Periodicity.monthly)
    zero_mask, values_for_fitting = _replace_zeros_with_nan(scaled)
    alphas, betas = compute.gamma_parameters(values_for_fitting, 1980, 1980, 2019, compute.Periodicity.monthly)
    gamma_probabilities = stats.gamma.cdf(values_for_fitting, a=alphas, scale=betas)
    gamma_probabilities[zero_mask] = 0.0
    legacy_transform = stats.norm.ppf(legacy + (1 - legacy) * gamma_probabilities)

    # the changed mass moves the whole transform, not just the zeros
    assert not np.allclose(computed, legacy_transform)


def test_gamma_classic_changes_when_the_calibration_window_narrows() -> None:
    """A shorter window moves classic gamma output through the window's zero mass."""
    rng = np.random.default_rng(20260928)
    values = rng.gamma(0.8, 7.0, size=(40, 12)) + 1.0
    values[0:5, :] = 0.0  # zeros only in 1980-1984
    scaled = compute.prepare_scaled(values, 1, compute.Periodicity.monthly)

    narrowed = compute.transform_fitted_gamma(scaled, 1980, 1990, 2019, compute.Periodicity.monthly)
    # the 1990-2019 window has no zeros, so this step's classic zero score is -inf
    assert np.all(narrowed[0:5, :] == -np.inf)

    # with the whole-record mass the same positions carry a finite classic score instead
    probability_of_zero = _legacy_probabilities_of_zero(scaled)
    expected = compute.scipy.stats.norm.ppf(probability_of_zero)
    assert np.all(np.isfinite(expected))
    assert not np.allclose(narrowed[0:5, :], expected)


@pytest.mark.parametrize("mode", ("center_of_mass", "mean_zero"))
def test_gamma_modes_reach_the_pearson_fallback_transform(mode: str) -> None:
    """The mode survives the Pearson-to-gamma fall back."""
    scaled = compute.prepare_scaled(_monthly_precip(), 1, compute.Periodicity.monthly)
    with mock.patch(
        "climate_indices.compute.transform_fitted_pearson",
        side_effect=compute.DistributionFittingError("Pearson failed", distribution_name="pearson3"),
    ):
        fell_back = compute.fit_and_standardize(
            scaled.copy(),
            indices.Distribution.pearson,
            1980,
            1980,
            2019,
            compute.Periodicity.monthly,
            fallback_to_gamma=True,
            zero_handling=mode,
        )
    direct = compute.fit_and_standardize(
        scaled.copy(), indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly, zero_handling=mode
    )
    classic = compute.fit_and_standardize(
        scaled.copy(), indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly
    )
    np.testing.assert_array_equal(fell_back, direct)
    assert not np.allclose(fell_back, classic, equal_nan=True)


@pytest.mark.parametrize("mode", ("center_of_mass", "mean_zero"))
def test_zero_handling_reaches_the_spatial_block_path(mode: str) -> None:
    """A declared time-major block gets the same zero placement as a single series."""
    series = _monthly_precip()
    block = np.stack([series, series * 0.5, np.zeros_like(series)], axis=-1)  # (years, periods, cells)

    classic = indices.spi(
        block, 3, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly, spatial_time_major=True
    )
    computed = indices.spi(
        block,
        3,
        indices.Distribution.gamma,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        spatial_time_major=True,
        zero_handling=mode,
    )

    assert computed.shape == classic.shape
    assert not np.allclose(computed, classic)


def test_non_classic_modes_score_dry_steps_negative() -> None:
    """A dry January scores at or above zero under classic and below zero under both modes."""
    values = np.full((40, 12), 5.0)
    values[0:20, 0] = 0.0  # January is dry for half the record

    classic = indices.spi(values, 1, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly)
    centered = indices.spi(
        values,
        1,
        indices.Distribution.gamma,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        zero_handling="center_of_mass",
    )
    mean_zero = indices.spi(
        values, 1, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly, zero_handling="mean_zero"
    )

    january_zeros = np.zeros(40 * 12, dtype=bool)
    january_zeros[0 : 20 * 12 : 12] = True
    assert np.all(classic[january_zeros] >= 0.0)
    assert np.all(centered[january_zeros] < 0.0)
    assert np.all(mean_zero[january_zeros] < 0.0)
    assert np.all(mean_zero[january_zeros] < centered[january_zeros])


def test_zero_probability_of_zero_keeps_the_classic_placement() -> None:
    """Zeros outside the calibration window keep the classic result when p0 == 0."""
    rng = np.random.default_rng(20260928)
    values = rng.gamma(0.8, 7.0, size=(40, 12)) + 1.0  # strictly positive
    values[0:10, :] = 0.0  # zeros only in 1980-1989, outside the 1990-2019 window
    # scale 1 keeps the 1980s zeros out of the calibration window's accumulated values

    classic = indices.spi(values, 1, indices.Distribution.gamma, 1980, 1990, 2019, compute.Periodicity.monthly)
    for mode in ("center_of_mass", "mean_zero"):
        computed = indices.spi(
            values, 1, indices.Distribution.gamma, 1980, 1990, 2019, compute.Periodicity.monthly, zero_handling=mode
        )
        np.testing.assert_array_equal(computed, classic)
        assert np.all(computed[values.reshape(-1) == 0.0] == -3.09)


def test_all_zero_calibration_step_keeps_the_classic_placement() -> None:
    """An all-zero calibration step (p0 == 1) keeps the existing invalid-fit handling."""
    rng = np.random.default_rng(20260928)
    values = rng.gamma(0.8, 7.0, size=(40, 12)) + 1.0  # strictly positive
    values[:, 0] = 0.0

    classic = indices.spi(values, 1, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly)
    for mode in ("center_of_mass", "mean_zero"):
        computed = indices.spi(
            values, 1, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly, zero_handling=mode
        )
        np.testing.assert_array_equal(computed, classic)


@pytest.mark.parametrize("mode", ("centre_of_mass", "mean-zero", ""))
def test_unknown_zero_handling_raises_value_error(mode: str) -> None:
    """Only the three documented values are accepted."""
    values = _monthly_precip()
    with pytest.raises(ValueError, match="zero_handling"):
        indices.spi(
            values,
            3,
            indices.Distribution.gamma,
            1980,
            1980,
            2019,
            compute.Periodicity.monthly,
            zero_handling=mode,
        )


def test_gamma_reads_a_supplied_zero_mass_for_a_spatial_block() -> None:
    """A gamma `prob_zero` from `fitting_params` is used as-is, and broadcasts over cells."""
    values = _monthly_precip()
    block = np.stack([values, values * 0.5], axis=-1)
    zero_mask, values_for_fitting = _replace_zeros_with_nan(block)
    aim = np.where(
        np.isclose(zero_mask.sum(axis=0) / zero_mask.shape[0], 1.0), 0.0, zero_mask.sum(axis=0) / zero_mask.shape[0]
    )
    alphas, betas = compute.gamma_parameters(values_for_fitting, 1980, 1980, 2019, compute.Periodicity.monthly)

    supplied = compute.fit_and_standardize(
        block.copy(),
        indices.Distribution.gamma,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        {"alpha": alphas, "beta": betas, "prob_zero": aim},
    )
    fitted = compute.fit_and_standardize(
        block.copy(), indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly
    )
    assert supplied.shape == block.shape
    np.testing.assert_array_equal(supplied, fitted)

    # the supplied mass is read, not recomputed: halving it moves the output
    halved = compute.fit_and_standardize(
        block.copy(),
        indices.Distribution.gamma,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        {"alpha": alphas, "beta": betas, "prob_zero": aim / 2.0},
    )
    assert not np.allclose(halved, fitted)


@pytest.mark.parametrize("mode", _MODES)
def test_one_dimensional_input_matches_the_legacy_two_dimensional_layout(mode: str) -> None:
    """A 1-D series and the legacy (years, periods) layout take the same path."""
    values = _monthly_precip()
    flat = indices.spi(
        values.flatten(),
        1,
        indices.Distribution.gamma,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        zero_handling=mode,
    )
    two_d = indices.spi(
        values, 1, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly, zero_handling=mode
    )
    np.testing.assert_array_equal(flat, two_d)


@pytest.mark.parametrize("mode", ("center_of_mass", "mean_zero"))
def test_spatial_block_matches_one_call_per_cell(mode: str) -> None:
    """A time-major block gives each cell the result of its own single-series call."""
    series = _monthly_precip()
    flat = series.flatten()
    # (time, lat, lon), the layout the xarray adapter packs
    block = np.stack([flat, flat * 0.5, np.zeros_like(flat)], axis=-1)[:, None, :]

    computed = indices.spi(
        block,
        6,
        indices.Distribution.gamma,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        spatial_time_major=True,
        zero_handling=mode,
    )
    assert computed.shape == block.shape
    for lon in range(block.shape[-1]):
        single = indices.spi(
            block[:, 0, lon],
            6,
            indices.Distribution.gamma,
            1980,
            1980,
            2019,
            compute.Periodicity.monthly,
            zero_handling=mode,
        )
        np.testing.assert_array_equal(computed[:, 0, lon], single)


def test_all_zero_calibration_step_scores_extreme_drought() -> None:
    """An all-zero calibration step (p0 == 1) keeps the classic -3.09, never +3.09."""
    rng = np.random.default_rng(20260928)
    values = rng.gamma(0.8, 7.0, size=(40, 12)) + 1.0
    values[:, 0] = 0.0

    for mode in _MODES:
        computed = indices.spi(
            values, 1, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly, zero_handling=mode
        )
        january = computed.reshape(40, 12)[:, 0]
        np.testing.assert_array_equal(january, np.full(40, -3.09))


@pytest.mark.parametrize("entry", ("standardized_index", "gamma", "pearson"))
def test_unknown_zero_handling_raises_from_every_compute_entry_point(entry: str) -> None:
    """The mode is rejected whether it enters through the index API or the transforms."""
    values = _monthly_precip()
    if entry == "standardized_index":
        call = lambda: indices.standardized_index(  # noqa: E731
            values, 1, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly, zero_handling="bogus"
        )
    elif entry == "gamma":
        call = lambda: compute.transform_fitted_gamma(  # noqa: E731
            values, 1980, 1980, 2019, compute.Periodicity.monthly, zero_handling="bogus"
        )
    else:
        call = lambda: compute.transform_fitted_pearson(  # noqa: E731
            values, 1980, 1980, 2019, compute.Periodicity.monthly, zero_handling="bogus"
        )
    with pytest.raises(ValueError, match="zero_handling"):
        call()


def test_supplied_zero_mass_is_not_mutated_and_applies_under_a_mode() -> None:
    """A saved zero mass is read-only input and still drives non-classic placement."""
    scaled = compute.prepare_scaled(_monthly_precip(), 1, compute.Periodicity.monthly)
    zero_mask, values_for_fitting = _replace_zeros_with_nan(scaled)
    alphas, betas = compute.gamma_parameters(values_for_fitting, 1980, 1980, 2019, compute.Periodicity.monthly)
    supplied = np.where(
        np.isclose(zero_mask.sum(axis=0) / zero_mask.shape[0], 1.0), 0.0, zero_mask.sum(axis=0) / zero_mask.shape[0]
    )
    supplied[3] = 1.0  # an all-zero step, which the transform resets internally
    supplied[5] = 0.2  # a placed zero mass, to see the mode score
    untouched = supplied.copy()

    computed = compute.fit_and_standardize(
        scaled.copy(),
        indices.Distribution.gamma,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        {"alpha": alphas, "beta": betas, "prob_zero": supplied},
        zero_handling="mean_zero",
    )

    np.testing.assert_array_equal(supplied, untouched)
    expected = compute._zero_score(np.where(supplied == 1.0, 0.0, supplied), "mean_zero")
    assert np.isfinite(computed[:, 5][zero_mask[:, 5]]).all()
    np.testing.assert_allclose(computed[:, 5][zero_mask[:, 5]], expected[5])


def test_non_classic_modes_score_from_a_supplied_zero_mass() -> None:
    """A supplied gamma zero mass drives the mode score, not a recomputed one."""
    scaled = compute.prepare_scaled(_monthly_precip(), 1, compute.Periodicity.monthly)
    zero_mask, values_for_fitting = _replace_zeros_with_nan(scaled)
    alphas, betas = compute.gamma_parameters(values_for_fitting, 1980, 1980, 2019, compute.Periodicity.monthly)
    supplied = zero_mask.sum(axis=0) / zero_mask.shape[0]
    supplied[3] = 0.25  # deliberately different from the fitted mass

    computed = compute.fit_and_standardize(
        scaled.copy(),
        indices.Distribution.gamma,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        {"alpha": alphas, "beta": betas, "prob_zero": supplied},
        zero_handling="center_of_mass",
    )
    np.testing.assert_allclose(computed[:, 3][zero_mask[:, 3]], compute._zero_score(supplied, "center_of_mass")[3])


@pytest.mark.parametrize("mode", ("center_of_mass", "mean_zero"))
def test_non_classic_zero_scores_are_clipped(mode: str) -> None:
    """A zero mass small enough pushes a mode's zero score past the fitted-index bound."""
    values = np.full((200, 12), 5.0)
    values[0:1, 0] = 0.0  # one January zero in 200 years, p0 = 0.005

    computed = indices.spi(
        values, 1, indices.Distribution.gamma, 1980, 1980, 2179, compute.Periodicity.monthly, zero_handling=mode
    )
    # the mode's score for p0 = 0.005 is below -3.09 before the pipeline clips it
    assert np.all(computed[0] >= -3.09)
    assert np.all(computed[0] <= 3.09)


def test_gamma_spei_uses_the_calibration_window_zero_mass() -> None:
    """SPEI shares the gamma transform, so its zero mass follows the window too."""
    rng = np.random.default_rng(20260928)
    precips = rng.gamma(0.8, 7.0, size=(40, 12)) + 1.0
    precips[0:5, :] = 0.0
    pet = np.full((40, 12), 0.5)  # P - PET is exactly zero for the 1980s Januarys

    full = indices.spei(precips, pet, 1, indices.Distribution.gamma, compute.Periodicity.monthly, 1980, 1980, 2019)
    narrowed = indices.spei(precips, pet, 1, indices.Distribution.gamma, compute.Periodicity.monthly, 1980, 1990, 2019)
    assert not np.allclose(full, narrowed)


def test_pearson_trace_value_is_placed_by_a_mode() -> None:
    """A Pearson trace value (below the 0.0005 threshold, p0 > 0) is moved by a mode."""
    rng = np.random.default_rng(20260928)
    values = rng.gamma(2.0, 8.0, size=(40, 12)) + 0.001
    values[0:8, 0] = 0.0  # a zero mass for January
    values[8:12, 0] = 0.0001  # traces, below the threshold but not zero

    scaled = compute.prepare_scaled(values, 1, compute.Periodicity.monthly)
    zero_mask, values_for_fitting = _replace_zeros_with_nan(scaled)
    probabilities_of_zero, locs, scales, skews = compute.pearson_parameters(
        scaled, 1980, 1980, 2019, compute.Periodicity.monthly
    )
    trace_mask = (scaled < 0.0005) & (probabilities_of_zero > 0.0) & ~zero_mask
    assert trace_mask.any()

    classic = compute.transform_fitted_pearson(
        scaled, 1980, 1980, 2019, compute.Periodicity.monthly, probabilities_of_zero, locs, scales, skews
    )
    computed = compute.transform_fitted_pearson(
        scaled,
        1980,
        1980,
        2019,
        compute.Periodicity.monthly,
        probabilities_of_zero,
        locs,
        scales,
        skews,
        zero_handling="mean_zero",
    )
    expected = compute._zero_score(probabilities_of_zero, "mean_zero")
    np.testing.assert_allclose(computed[trace_mask], np.broadcast_to(expected, scaled.shape)[trace_mask])
    assert not np.allclose(computed[trace_mask], classic[trace_mask])

"""Pearson fits exclude the zero mass from the continuous component (#1562)."""

import warnings

import numpy as np
import pytest
from scipy import stats

from climate_indices import compute, indices


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("offset", [0.0001, 1.0, -7.0])
def test_time_step_fit_excludes_only_exact_zeros(dtype: str, offset: float) -> None:
    wet = np.arange(7, dtype=dtype) + offset
    sample = np.concatenate(([0.0, -0.0, np.nan], wet)).astype(dtype)
    # A strided input must retain its values and its storage layout.
    storage = np.repeat(sample, 2)
    sample = storage[::2]
    original = sample.copy()
    actual = compute.calculate_time_step_params(sample)
    # A symmetric seven-point sample has lambda_2 = 4/3 and zero L-skewness.
    expected = (2 / 9, np.mean(wet, dtype=np.float64), 4 / 3 * np.sqrt(np.pi), 0.0)
    tolerance = 1e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(sample, original)


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("periodicity", [compute.Periodicity.monthly, compute.Periodicity.daily])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_calibration_fit_separates_zero_mass(
    monkeypatch: pytest.MonkeyPatch,
    spatial: bool,
    periodicity: compute.Periodicity,
    dtype: str,
) -> None:
    monkeypatch.setattr(compute, "_native", None)
    periods = 12 if periodicity == compute.Periodicity.monthly else 366
    # The first and last years are outside the Calibration Period.
    sample = np.array([999, 0, -0.0, np.nan, 1, 2, 3, 4, 5, 6, 7, 999], dtype=dtype)
    values = np.tile(sample[:, None], (1, periods))
    if spatial:
        values = np.repeat(values[..., None], 2, axis=-1)
    original = values.copy()
    actual = compute.pearson_parameters(values, 2000, 2001, 2010, periodicity)
    for result, expected in zip(actual, (2 / 9, 4.0, 4 / 3 * np.sqrt(np.pi), 0.0), strict=True):
        # These integer samples have exact L-moments; allow f64 conversion roundoff.
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize(
    "sample",
    [
        [0, 0, 0, 0, 1, 2, 3, np.nan],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [np.nan] * 8,
        [0, 0, 2, 2, 2, 2, 2, np.nan],
    ],
)
def test_invalid_nonzero_sample_keeps_zeroed_parameters(
    monkeypatch: pytest.MonkeyPatch, spatial: bool, sample: list[float]
) -> None:
    monkeypatch.setattr(compute, "_native", None)
    values = np.tile(np.asarray(sample, dtype=float)[:, None], (1, 12))
    if spatial:
        values = values[..., None]
    actual = compute.pearson_parameters(values, 2000, 2000, 2007, compute.Periodicity.monthly)
    for result in actual:
        np.testing.assert_array_equal(result, 0.0)


def test_fitted_location_is_the_nonzero_sample_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute, "_native", None)
    rng = np.random.default_rng(31)
    values = rng.gamma(1.5, 40.0, (40, 12))
    # A different zero count per calendar month, so a whole-record mean cannot pass.
    zeros = rng.random(values.shape) < np.linspace(0.05, 0.5, 12)
    values[zeros] = 0.0
    probabilities_of_zero, locs, _, _ = compute.pearson_parameters(
        values, 2000, 2000, 2039, compute.Periodicity.monthly
    )
    # The first L-moment is the sample mean, so it identifies the fitted sample exactly.
    non_zero_mean = np.nansum(values, axis=0) / np.count_nonzero(values != 0, axis=0)
    np.testing.assert_allclose(locs, non_zero_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(probabilities_of_zero, np.count_nonzero(zeros, axis=0) / values.shape[0], rtol=0, atol=0)
    # Without the correction the location is the whole-record mean, which differs.
    assert not np.allclose(locs, np.nanmean(values, axis=0), rtol=1e-3, atol=1e-3)


def test_a_zero_skew_fit_has_an_unbounded_support() -> None:
    # A zero skew is the normal limit, so both support limits must stay disabled
    # without the 0/0 floating-point warnings the raw formula raises.
    skews = np.array([0.0, 0.5, -0.5, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        limits = compute._minimum_possible(skews, np.full(4, 2.5), np.ones(4))
    assert limits[0] == -np.inf
    assert np.isfinite(limits[1])
    assert np.isfinite(limits[2])
    assert np.isnan(limits[3])


def test_a_tiny_negative_skew_keeps_the_upper_support_mask_empty() -> None:
    # alpha = 4 / skew**2 overflows for |skew| < ~1e-154, so a tiny negative skew must
    # give +inf: -inf would make the (values >= limit) & (skew < 0) mask match everything.
    skews = np.array([-1e-160, 1e-160, -0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        limits = compute._minimum_possible(skews, np.full(3, 2.5), np.ones(3))
    np.testing.assert_array_equal(limits, [np.inf, -np.inf, -np.inf])


def test_a_symmetric_nonzero_component_transforms_without_warnings() -> None:
    # Every calendar month keeps the same symmetric non-zero component, so tau_3 = 0
    # and the fit is valid everywhere: the only support-limit edge in play is skew == 0.
    values = np.zeros((30, 12))
    values[:4, :] = np.array([1.0, 2.0, 3.0, 4.0])[:, None]
    probabilities_of_zero, locs, scales, skews = compute.pearson_parameters(
        values, 2000, 2000, 2029, compute.Periodicity.monthly
    )
    np.testing.assert_array_equal(skews, 0.0)
    np.testing.assert_allclose(locs, 2.5, rtol=1e-12, atol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pit = compute.transform_fitted_pearson(
            values,
            2000,
            2000,
            2029,
            compute.Periodicity.monthly,
            output_scale="probability",
            probabilities_of_zero=probabilities_of_zero,
            locs=locs,
            scales=scales,
            skews=skews,
        )
    # Rows 0-3 hold 1.0-4.0 and row 4 is a zero, which keeps the whole mass at p0.
    # The non-zero rows are the normal limit's CDF, mixed in above that mass.
    p0 = 26 / 30
    scale = 5 / 6 * np.sqrt(np.pi)
    np.testing.assert_allclose(pit[4, 0], p0, rtol=0, atol=1e-12)
    for row, value in ((0, 1.0), (3, 4.0)):
        expected = p0 + (1 - p0) * stats.norm.cdf((value - 2.5) / scale)
        np.testing.assert_allclose(pit[row, 0], expected, rtol=1e-12, atol=1e-12)


def test_spai_pooled_pearson_fit_keeps_exact_zeros() -> None:
    # SPAI has no probability of zero, so its pooled fit keeps every anomaly value.
    # This is the exception the zero-mass correction documents.
    pooled = np.array([[-2.0], [0.0], [0.0], [1.0], [2.0], [3.0]])
    params = indices._fit_pearson_or_normal(pooled)
    assert float(params["loc"][0]) == float(pooled.mean())
    assert float(params["loc"][0]) != float(pooled[pooled != 0].mean())


def test_series_fit_treats_masked_values_as_missing() -> None:
    rng = np.random.default_rng(5)
    values = rng.gamma(2.0, 30.0, (12, 12)) + 1.0
    mask = np.zeros(values.shape, dtype=bool)
    mask[0, 5] = True
    mask[3, 2] = True
    _, locs, _, _ = compute.pearson_parameters(
        np.ma.array(values, mask=mask), 2000, 2000, 2011, compute.Periodicity.monthly
    )
    # The first L-moment is the sample mean of the values the fit kept.
    kept = np.where(mask, np.nan, values)
    np.testing.assert_allclose(locs, np.nanmean(kept, axis=0), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("sentinel", [999.0, 0.0])
def test_spatial_fit_keeps_masked_values_out_of_the_sample(dtype: str, sentinel: float) -> None:
    # A masked value is missing in both counts, wherever it sits and whatever it holds:
    # the 0.0 sentinel is the only case where a mask changes the zero count.
    sample = np.array([0, 0, sentinel, 1, 2, 3, 4, 5, 6, 7], dtype=dtype)
    values = np.tile(sample[:, None, None], (1, 12, 2))
    mask = np.zeros(values.shape, dtype=bool)
    mask[2, :, :] = True
    masked = np.ma.array(values, mask=mask)
    probabilities_of_zero, locs, scales, skews, failed = compute._pearson_parameters_spatial(masked)
    assert failed == 0
    np.testing.assert_allclose(probabilities_of_zero, 2 / 9, rtol=0, atol=1e-12)
    for result, expected in zip((locs, scales, skews), (4.0, 4 / 3 * np.sqrt(np.pi), 0.0), strict=True):
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(masked.data, values)
    np.testing.assert_array_equal(np.ma.getmaskarray(masked), mask)


@pytest.mark.parametrize("spatial", [False, True])
def test_spi_uses_the_conditional_nonzero_cdf(monkeypatch: pytest.MonkeyPatch, spatial: bool) -> None:
    monkeypatch.setattr(compute, "_native", None)
    sample = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    values = np.tile(sample[:, None], (1, 12)).ravel()
    if spatial:
        values = np.repeat(values[:, None, None], 2, axis=-1)
    actual = indices.spi(
        values,
        1,
        indices.Distribution.pearson,
        2000,
        2000,
        2008,
        compute.Periodicity.monthly,
        output_scale="probability",
    )
    # The nonzero sample is symmetric, so Pearson III reduces to this normal CDF.
    expected = 2 / 9 + 7 / 9 * stats.norm.cdf((values - 4) / (4 / 3 * np.sqrt(np.pi)))
    expected[values == 0] = 2 / 9
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_spi_zero_inflated_gamma_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute, "_native", None)
    rng = np.random.default_rng(7)
    years, p0, shape, scale = 2000, 0.3, 1.5, 40.0
    zeros = rng.random(years * 12) < p0
    values = np.where(zeros, 0.0, rng.gamma(shape, scale, years * 12))
    actual = indices.spi(values, 1, indices.Distribution.pearson, 1, 1, years, compute.Periodicity.monthly)
    expected = stats.norm.ppf(p0 + (1 - p0) * stats.gamma.cdf(values[~zeros], shape, scale=scale))
    rmse = np.sqrt(np.mean((actual[~zeros] - expected) ** 2))
    # Sampling and L-moment estimation give about 0.04 RMSE, versus 0.34 before the fix.
    assert rmse < 0.06

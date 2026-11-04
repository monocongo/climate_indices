"""Pearson fits exclude the zero mass from the continuous component (#1562)."""

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


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_spatial_fit_keeps_masked_values_out_of_the_sample(dtype: str) -> None:
    sample = np.array([0, 0, 999, 1, 2, 3, 4, 5, 6, 7], dtype=dtype)
    values = np.tile(sample[:, None, None], (1, 12, 2))
    masked = np.ma.array(values, mask=values == 999)
    _, locs, scales, skews, failed = compute._pearson_parameters_spatial(masked)
    assert failed == 0
    for result, expected in zip((locs, scales, skews), (4.0, 4 / 3 * np.sqrt(np.pi), 0.0), strict=True):
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(masked.data, values)
    np.testing.assert_array_equal(masked.mask, values == 999)


@pytest.mark.parametrize("spatial", [False, True])
def test_spi_uses_the_conditional_nonzero_cdf(monkeypatch: pytest.MonkeyPatch, spatial: bool) -> None:
    monkeypatch.setattr(compute, "_native", None)
    sample = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    values = np.tile(sample[:, None], (1, 12)).ravel()
    if spatial:
        values = np.repeat(values[:, None, None], 2, axis=-1)
    # The support-limit formula divides by zero at the normal limit (skew == 0).
    with np.errstate(divide="ignore", invalid="ignore"):
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

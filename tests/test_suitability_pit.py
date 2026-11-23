"""Seasonal calibration of probability-scale index outputs (#1588)."""

import functools
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import xarray as xr

from climate_indices import compute, fit_diagnostics, provenance, spi, suitability
from climate_indices.drought import indices
from climate_indices.exceptions import BetaFeatureWarning, ClimateIndicesWarning, InvalidArgumentError
from climate_indices.vocabulary import Distribution

START = 1980
MONTHLY = compute.Periodicity.monthly
DAILY = compute.Periodicity.daily
pytestmark = pytest.mark.filterwarnings("ignore::climate_indices.exceptions.BetaFeatureWarning")


def _pit(values, scale=1, *, distribution=Distribution.normal, **kwargs):
    return suitability.pit_uniformity(
        values, scale, distribution=distribution, periodicity=MONTHLY, data_start_year=START, **kwargs
    )


@pytest.mark.parametrize(
    ("distribution", "zero_handling"),
    [
        (Distribution.normal, "classic"),
        (Distribution.gamma, "classic"),
        (Distribution.pearson, "classic"),
        (Distribution.pearson, "center_of_mass"),
    ],
)
def test_known_cdf_has_nominal_size(distribution, zero_handling):
    rng = np.random.default_rng(1588)
    sample = rng.uniform(size=(40 * 12, 300))
    kwargs = {}
    if distribution is not Distribution.normal:
        exact_zero = rng.random(sample.shape) < 0.4
        trace = np.zeros_like(exact_zero)
        mixed = np.where(exact_zero, 0.4, 0.4 + 0.6 * sample)
        kwargs["prob_zero"] = np.full((12, 300), 0.4)
        if distribution is Distribution.pearson:
            # Gamma(4, 0.001) is Pearson Type III with skew=1 and a small trace fraction.
            precipitation = np.where(exact_zero, 0, scipy.stats.gamma.ppf(sample, a=4, scale=0.001))
            trace = (precipitation < 0.0005) & ~exact_zero
            shape = (12, 300, 1)
            mixed = indices.spi(
                precipitation[..., None],
                1,
                distribution,
                START,
                START,
                START + 39,
                MONTHLY,
                fitting_params={
                    "prob_zero": np.full(shape, 0.4),
                    "loc": np.full(shape, 0.004),
                    "scale": np.full(shape, 0.002),
                    "skew": np.ones(shape),
                },
                output_scale="probability",
                zero_handling=zero_handling,
            )[..., 0]
            assert trace.any()
            kwargs["trace_mask"] = trace
        zero_mask = exact_zero | trace
        kwargs["zero_mask"] = zero_mask
    else:
        mixed = sample

    result = _pit(mixed, distribution=distribution, spatial_time_major=True, **kwargs)
    # The two-sided 99.9% binomial acceptance interval for 3,600 independent tests at alpha=0.05.
    lower, upper = scipy.stats.binom.interval(0.999, 3600, 0.05)
    assert lower <= np.count_nonzero(result.ks_p_value_naive <= 0.05) <= upper
    if kwargs:
        np.testing.assert_array_equal(result.n_valid, (~zero_mask).reshape(40, 12, 300).sum(axis=0))
        np.testing.assert_allclose(result.zero_share, exact_zero.reshape(40, 12, 300).mean(axis=0))
        np.testing.assert_allclose(result.trace_share, trace.reshape(40, 12, 300).mean(axis=0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_calibration_slice_and_statistics_match_independent_references(dtype):
    values = np.full((5, 12), 0.99, dtype=dtype)
    values[1:4, 0] = [0.2, 0.3, 0.9]
    values[1:4, 1] = [0, 1, np.nan]
    values[1:4, 2] = np.nan
    result = _pit(values, calibration_year_initial=START + 1, calibration_year_final=START + 3)
    january = values[1:4, 0].astype(float)
    expected = scipy.stats.kstest(january, "uniform")
    expected_ad = -3 - sum((2 * i - 1) * (np.log(january[i - 1]) + np.log(1 - january[-i])) for i in range(1, 4)) / 3
    # Three binary64 terms and exact K–S for this sample size.
    assert result.ks_statistic[0] == pytest.approx(expected.statistic, abs=1e-14)
    assert result.ks_p_value_naive[0] == pytest.approx(expected.pvalue, abs=1e-14)
    assert result.ad_statistic[0] == pytest.approx(expected_ad, abs=1e-14)
    np.testing.assert_array_equal(result.n_valid[:3], [3, 2, 0])
    assert np.isposinf(result.ad_statistic[1])
    assert np.isnan(result.ad_statistic[2])
    assert np.isnan(result.zero_share[2])


def test_small_continuous_probability_is_not_clipped():
    probability = 1e-100
    result = _pit(np.array([probability]))
    assert result.ad_statistic[0] == pytest.approx(-1 - np.log(probability) - np.log1p(-probability), rel=1e-14)


def test_masked_float32_uses_original_precision_at_the_zero_mass_boundary():
    values = np.ma.array(np.full(24, 0.7, dtype=np.float32), mask=np.arange(24) == 0)
    result = _pit(
        values, distribution=Distribution.gamma, prob_zero=np.full(12, 0.7), zero_mask=np.zeros(24, dtype=bool)
    )
    assert result.n_valid[0] == 1
    assert result.n_valid[1] == 2


def test_float32_zero_mass_preserves_the_upper_probability_endpoint():
    result = _pit(
        np.ones(24),
        distribution=Distribution.gamma,
        prob_zero=np.full(12, 0.1, dtype=np.float32),
        zero_mask=np.zeros(24, dtype=bool),
    )
    assert result.ks_statistic is not None
    assert result.ad_statistic is not None
    assert (result.ks_statistic == 1).all()
    assert np.isposinf(result.ad_statistic).all()


def test_seasonal_variance_in_spai_is_detected():
    rng = np.random.default_rng(410)
    precipitation = 200 + rng.normal(size=(120, 12)) * np.tile([1, 30], 6)
    probability = indices.spai(
        precipitation.ravel(), 1, Distribution.normal, START, START, START + 119, MONTHLY, output_scale="probability"
    )
    result = _pit(probability)
    assert np.all(result.ks_p_value_naive[::2] < 0.01)


def test_pearson_trace_mask_must_include_point_mass_and_reports_both_shares():
    precipitation = np.full((4, 12), 5.0)
    precipitation[:, 0] = [0.0, 0.0002, 3.0, np.nan]
    prob_zero = np.full(12, 0.4)
    values = indices.spi(
        precipitation.ravel(),
        1,
        Distribution.pearson,
        START,
        START,
        START + 3,
        MONTHLY,
        fitting_params={
            "prob_zero": prob_zero,
            "loc": np.full(12, 4.0),
            "scale": np.full(12, 2.0),
            "skew": np.ones(12),
        },
        zero_handling="center_of_mass",
        output_scale="probability",
    ).reshape(4, 12)
    exact_zero = precipitation == 0
    trace = (precipitation > 0) & (precipitation < 0.0005)
    np.testing.assert_array_equal(values[:2, 0], [0.2, 0.2])
    with pytest.raises(ValueError, match="zero_mask.*below prob_zero"):
        _pit(values, distribution=Distribution.pearson, prob_zero=prob_zero, zero_mask=exact_zero)
    result = _pit(
        values, distribution=Distribution.pearson, prob_zero=prob_zero, zero_mask=exact_zero | trace, trace_mask=trace
    )
    assert result.n_valid[0] == 1
    assert result.zero_share[0] == result.trace_share[0] == pytest.approx(1 / 3)
    # This Pearson Type III is a Gamma(4, 1) with a zero mass of 0.4.
    assert result.ks_statistic[0] == pytest.approx(1 - scipy.stats.gamma.cdf(3, a=4), abs=1e-14)


@pytest.mark.parametrize(
    ("distribution", "p0", "probability"),
    [(Distribution.gamma, 0.4, 0.9), (Distribution.pearson, 0.4, 0.9), (Distribution.gamma, 1.0, 0.5)],
)
@pytest.mark.parametrize("all_masked", [False, True])
def test_incompatible_masked_probabilities_are_rejected(distribution, p0, probability, all_masked):
    mask = np.full(24, all_masked, dtype=bool)
    mask[0] = True
    values = np.full(24, probability)
    prob_zero = np.full(12, p0)
    with pytest.raises(ValueError, match="zero_mask.*incompatible"):
        _pit(values, distribution=distribution, prob_zero=prob_zero, zero_mask=mask)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson])
@pytest.mark.parametrize("zero_handling", ["classic", "center_of_mass", "mean_zero"])
def test_produced_point_mass_placements_are_accepted(dtype, distribution, zero_handling):
    precipitation = np.full(24, 5.0)
    precipitation[:12] = 0
    parameters = {"prob_zero": np.full(12, 0.7)}
    if distribution is Distribution.gamma:
        parameters.update(alpha=np.ones(12), beta=np.ones(12))
    else:
        parameters.update(loc=np.full(12, 4.0), scale=np.full(12, 2.0), skew=np.ones(12))
    probabilities = indices.spi(
        precipitation,
        1,
        distribution,
        START,
        START,
        START + 1,
        MONTHLY,
        fitting_params=parameters,
        zero_handling=zero_handling,
        output_scale="probability",
    ).astype(dtype)
    result = _pit(
        probabilities, distribution=distribution, prob_zero=parameters["prob_zero"], zero_mask=precipitation == 0
    )
    assert (result.n_valid == 1).all()
    assert (result.zero_share == 0.5).all()


def test_pearson_upper_support_point_mass_is_accepted_but_gamma_rejects_it():
    precipitation = np.zeros(24)
    probabilities = indices.spi(
        precipitation,
        1,
        Distribution.pearson,
        START,
        START,
        START + 1,
        MONTHLY,
        fitting_params={
            "prob_zero": np.full(12, 0.4),
            "loc": np.full(12, -4.0),
            "scale": np.full(12, 2.0),
            "skew": -np.ones(12),
        },
        output_scale="probability",
    )
    assert (probabilities == 1).all()
    result = _pit(
        probabilities, distribution=Distribution.pearson, prob_zero=np.full(12, 0.4), zero_mask=precipitation == 0
    )
    assert (result.zero_share == 1).all()
    with pytest.raises(ValueError, match="zero_mask.*incompatible"):
        _pit(probabilities, distribution=Distribution.gamma, prob_zero=np.full(12, 0.4), zero_mask=precipitation == 0)


@pytest.mark.parametrize("lazy", [False, True])
def test_xarray_rejects_incompatible_point_mass_mask(lazy):
    times = pd.date_range("1980-01-01", periods=24, freq="MS")
    values = xr.DataArray(np.full(24, 0.9), dims="time", coords={"time": times})
    mask = xr.ones_like(values, dtype=bool)
    p0 = xr.DataArray(np.full(12, 0.4), dims="month", coords={"month": np.arange(1, 13)})
    if lazy:
        values = values.chunk({"time": -1})
    diagnose = functools.partial(
        suitability.pit_uniformity, values, 1, distribution=Distribution.gamma, prob_zero=p0, zero_mask=mask
    )
    if lazy:
        result = diagnose()
        assert isinstance(result, xr.Dataset)
        with pytest.raises(ValueError, match="zero_mask.*incompatible"):
            result.compute()
    else:
        with pytest.raises(ValueError, match="zero_mask.*incompatible"):
            diagnose()


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning:climate_indices.compute")
def test_gamma_spi_all_zero_month_has_share_but_no_statistics():
    times = pd.date_range(f"{START}-01-01", periods=60, freq="MS")
    values = np.repeat(np.arange(1.0, 6.0), 12)
    values[::12] = 0
    precipitation = xr.DataArray(values, dims="time", coords={"time": times}, attrs={"units": "mm"})
    fit = fit_diagnostics(precipitation, 1, Distribution.gamma)
    probabilities = spi(precipitation, 1, Distribution.gamma, output_scale="probability")
    result = suitability.pit_uniformity(
        probabilities, 1, distribution=Distribution.gamma, prob_zero=fit.prob_zero, zero_mask=precipitation == 0
    )

    assert isinstance(result, xr.Dataset)
    assert fit.prob_zero.sel(month=1).item() == 1
    assert (probabilities.isel(time=slice(None, None, 12)) == 0).all()
    january = result.sel(month=1)
    assert january.n_valid.item() == 0
    assert january.zero_share.item() == 1
    assert january.trace_share.item() == 0
    assert np.isnan(january.ks_statistic.item())
    assert np.isnan(january.ks_p_value_naive.item())
    assert np.isnan(january.ad_statistic.item())
    assert (result.n_valid.sel(month=slice(2, 12)) == 5).all()
    assert np.isfinite(result.ks_statistic.sel(month=slice(2, 12))).all()


def test_all_point_mass_has_share_but_no_statistics():
    probabilities = np.ones(24)
    result = _pit(
        probabilities, distribution=Distribution.gamma, prob_zero=np.ones(12), zero_mask=np.ones(24, dtype=bool)
    )
    assert (result.zero_share == 1).all()
    assert (result.trace_share == 0).all()
    assert (result.n_valid == 0).all()
    assert np.isnan(result.ks_statistic).all()


@pytest.mark.parametrize("distribution", ["empirical", "beta", Distribution.empirical, Distribution.beta])
def test_unsupported_distributions_are_rejected(distribution):
    with pytest.raises(InvalidArgumentError, match="distribution"):
        _pit(np.full(24, 0.5), distribution=distribution)


def test_zero_mass_inputs_require_a_mixed_distribution():
    with pytest.raises(ValueError, match="gamma or Pearson Type III"):
        _pit(np.full(24, 0.5), prob_zero=np.full(12, 0.5), zero_mask=np.ones(24, dtype=bool))


def test_ambiguous_spatial_shape_requires_an_explicit_layout():
    probabilities = np.full((24, 12, 2), 0.5)
    with pytest.raises(ValueError, match="spatial_time_major=True"):
        _pit(probabilities)
    assert (_pit(probabilities, spatial_time_major=True).n_valid == 2).all()


@pytest.mark.parametrize("value", [-0.1, 1.1, np.inf])
def test_invalid_probabilities_are_rejected(value):
    with pytest.raises(InvalidArgumentError, match="probabilities"):
        _pit(np.array([0.5, value]))


@pytest.mark.parametrize("lazy", [False, True])
def test_xarray_equals_numpy_with_mixed_cdf_and_transposed_inputs(lazy):
    rng = np.random.default_rng(171)
    raw = rng.uniform(size=(37, 3))
    prob_zero = rng.uniform(0, 0.5, size=(12, 3))
    p0 = prob_zero[np.arange(len(raw)) % 12]
    zero_mask = raw < p0
    probabilities = np.where(zero_mask, p0 / 2, p0 + (1 - p0) * raw)
    probabilities[:, 2] = np.nan
    times = pd.date_range(f"{START}-01-01", periods=len(raw), freq="MS")
    data = xr.DataArray(probabilities.T, dims=("cell", "date"), coords={"date": times, "cell": [10, 20, 30]})
    mask = data.copy(data=zero_mask.T).transpose()
    p0_data = xr.DataArray(prob_zero, dims=("month", "cell"), coords={"month": np.arange(1, 13), "cell": data.cell})
    if lazy:
        data = data.chunk({"cell": 1})
        mask = mask.chunk({"cell": 2})
    actual = suitability.pit_uniformity(
        data, 1, distribution=Distribution.gamma, time_dim="date", prob_zero=p0_data, zero_mask=mask
    )
    if lazy:
        assert actual.ks_statistic.chunks is not None
    expected = _pit(
        probabilities,
        distribution=Distribution.gamma,
        prob_zero=prob_zero,
        zero_mask=zero_mask,
        spatial_time_major=True,
    )
    actual = actual.compute()
    assert actual.ks_statistic.dims == ("month", "cell")
    assert actual.n_valid.dtype == np.int64
    for name in ("ks_statistic", "ks_p_value_naive", "ad_statistic", "n_valid", "zero_share", "trace_share"):
        np.testing.assert_array_equal(actual[name].values, getattr(expected, name))
    assert actual.attrs["calibration_year_final"] == START + 3


def test_xarray_refuses_misaligned_zero_mask():
    times = pd.date_range(f"{START}-01-01", periods=24, freq="MS")
    values = xr.DataArray(np.full(24, 0.4), dims="time", coords={"time": times})
    mask = xr.DataArray(np.zeros(24, dtype=bool), dims="time", coords={"time": times + pd.offsets.MonthBegin()})
    p0 = xr.DataArray(np.zeros(12), dims="month", coords={"month": np.arange(1, 13)})
    with pytest.raises(ValueError, match="join='exact'"):
        suitability.pit_uniformity(values, 1, distribution=Distribution.gamma, prob_zero=p0, zero_mask=mask)


@pytest.mark.parametrize("indexed", [False, True])
def test_xarray_refuses_mismatched_auxiliary_spatial_coordinates(indexed):
    times = pd.date_range(f"{START}-01-01", periods=24, freq="MS")
    values = xr.DataArray(
        np.full((24, 2), 0.4), dims=("time", "cell"), coords={"time": times, "latitude": ("cell", [5, 10])}
    )
    mask = values.copy(data=np.zeros((24, 2), dtype=bool)).assign_coords(latitude=("cell", [10, 5]))
    if indexed:
        values = values.set_xindex("latitude")
    p0 = xr.DataArray(np.zeros((12, 2)), dims=("month", "cell"), coords={"month": np.arange(1, 13)})
    with pytest.raises(ValueError, match="zero_mask.*latitude"):
        suitability.pit_uniformity(values, 1, distribution=Distribution.gamma, prob_zero=p0, zero_mask=mask)


def test_xarray_retains_primary_coordinates_and_records_diagnostic_metadata():
    times = pd.date_range(f"{START}-01-01", periods=24, freq="MS")
    values = xr.DataArray(
        np.full((24, 2), 0.4),
        dims=("time", "cell"),
        coords={"time": times, "latitude": ("cell", [5, 10])},
        attrs={"valid_max": 1, "long_name": "Source index probability scale"},
    )
    mask = values.copy(data=np.zeros((24, 2), dtype=bool)).assign_coords(n_valid=("cell", [0, 0]))
    p0 = xr.DataArray(np.zeros((12, 2)), dims=("month", "cell"), coords={"month": np.arange(1, 13)})
    result = suitability.pit_uniformity(values, 1, distribution=Distribution.gamma, prob_zero=p0, zero_mask=mask)
    assert result.latitude.identical(values.latitude)
    assert (result.n_valid == 2).all()
    assert "valid_max" not in result.n_valid.attrs
    assert "long_name" not in result.n_valid.attrs
    assert result.n_valid.attrs[provenance.PROVENANCE_ATTR] == result.attrs[provenance.PROVENANCE_ATTR]


def test_daily_calendar_scores_only_observed_dates():
    times = pd.date_range("1999-01-01", "2001-03-01", freq="D")
    values = np.full(len(times), 0.6)
    values[(times.month == 2) & (times.day == 29)] = 0.2
    data = xr.DataArray(values, dims="time", coords={"time": times})
    actual = suitability.pit_uniformity(data, 365, distribution=Distribution.normal)
    np.testing.assert_array_equal(actual.n_valid.sel(dayofyear=[59, 60, 61]), [3, 1, 3])
    assert actual.ks_statistic.sel(dayofyear=60).item() == pytest.approx(0.8)
    with pytest.warns(ClimateIndicesWarning, match="exceeds 365"):
        suitability.pit_uniformity(data, 366, distribution=Distribution.normal)


def test_statistic_selection_and_monthly_overlap_warning():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        result = _pit(np.full(24, 0.5), 12, statistics=("ad",))
    assert not any(type(w.message) is ClimateIndicesWarning for w in seen)
    assert any(isinstance(w.message, BetaFeatureWarning) for w in seen)
    assert result.ks_statistic is result.ks_p_value_naive is None
    assert np.isfinite(result.ad_statistic).all()
    with pytest.warns(ClimateIndicesWarning, match="exceeds 12"):
        _pit(np.full(24, 0.5), 13)

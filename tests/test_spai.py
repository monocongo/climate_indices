"""Contract tests for SPAI, the Standardized Precipitation Anomaly Index (#1480).

These are regression and property tests only: no independent SPAI reference dataset
is committed (see VALIDATION.md).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import scipy.stats
import xarray as xr

import climate_indices
from climate_indices import compute, indices, lmoments
from climate_indices.exceptions import BetaFeatureWarning, InsufficientDataError, InvalidArgumentError

pytestmark = pytest.mark.filterwarnings("ignore::climate_indices.exceptions.BetaFeatureWarning")

MONTHLY = compute.Periodicity.monthly
T = indices.Distribution.t_location_scale
PEARSON = indices.Distribution.pearson
NORMAL = indices.Distribution.normal
SPAI_DISTRIBUTIONS = (T, PEARSON, NORMAL)

START, CALIBRATION_START, CALIBRATION_END, YEARS = 1961, 1961, 2010, 60
CALIBRATION_STEPS = (CALIBRATION_END - CALIBRATION_START + 1) * 12

# a monsoon-like climatology: a June-September wet season whose spread is about twenty
# times the dry season's
SEASONAL_MEAN = np.array([10, 8, 12, 20, 60, 180, 260, 240, 170, 60, 20, 12], dtype=float)
SEASONAL_SPREAD = np.array([4, 3, 5, 8, 20, 60, 90, 80, 60, 20, 8, 5], dtype=float)


def _seasonal(seed: int = 0, years: int = YEARS) -> np.ndarray:
    """Monthly precipitation with heavy-tailed noise around a strongly seasonal climatology."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(8, size=(years, 12))
    return np.clip(SEASONAL_MEAN + SEASONAL_SPREAD * noise, 0.0, None).ravel()


def _spai(values: np.ndarray, distribution: indices.Distribution = T, scale: int = 1, **kwargs: Any) -> np.ndarray:
    return indices.spai(values, scale, distribution, START, CALIBRATION_START, CALIBRATION_END, MONTHLY, **kwargs)


@pytest.fixture
def seasonal() -> np.ndarray:
    return _seasonal()


@pytest.mark.parametrize("seed", range(3))
def test_reference_fit_is_near_standard_normal_over_the_calibration_period(seed: int) -> None:
    """The t location-scale reference fit: |mean| <= 0.05 and |sd - 1| <= 0.05.

    Five seeds of this series measured |mean| <= 0.015 and |sd - 1| <= 0.006. The bound
    holds for the reference fit only, and only over the pooled window. The pooled anomaly
    is a scale mixture, so the lighter-tailed Pearson and normal fits reach the +/-3.09
    clip on wet-season values and their clipped spread falls short of one; no calendar
    period is promised a unit variance either.
    """
    calibration = _spai(_seasonal(seed))[:CALIBRATION_STEPS]

    assert np.isfinite(calibration).all()
    assert abs(calibration.mean()) <= 0.05
    assert abs(calibration.std(ddof=1) - 1.0) <= 0.05


@pytest.mark.parametrize("distribution", SPAI_DISTRIBUTIONS, ids=lambda d: d.name)
def test_calibration_period_is_finite(distribution: indices.Distribution) -> None:
    calibration = _spai(_seasonal(), distribution)[:CALIBRATION_STEPS]

    assert np.isfinite(calibration).all()


@pytest.mark.parametrize("distribution", SPAI_DISTRIBUTIONS, ids=lambda d: d.name)
def test_monotone_in_precipitation_within_each_calendar_period(
    seasonal: np.ndarray, distribution: indices.Distribution
) -> None:
    result = _spai(seasonal, distribution).reshape(YEARS, 12)
    precipitation = seasonal.reshape(YEARS, 12)

    for month in range(12):
        order = np.argsort(precipitation[:, month], kind="stable")
        assert (np.diff(result[order, month]) >= 0.0).all(), f"month {month + 1}"


def test_wet_and_dry_seasons_share_one_pooled_anomaly(seasonal: np.ndarray) -> None:
    """The same anomaly value scores the same in the dry and the wet season."""
    precipitation = seasonal.reshape(YEARS, 12)
    calibration = precipitation[: CALIBRATION_END - CALIBRATION_START + 1]
    anomaly = precipitation - calibration.mean(axis=0)
    result = _spai(seasonal, NORMAL).reshape(YEARS, 12)

    # under the normal fit SPAI is one linear map of the anomaly for every month
    unclipped = np.abs(result) < 3.09
    slope, intercept = np.polyfit(anomaly[unclipped], result[unclipped], 1)
    np.testing.assert_allclose(result[unclipped], slope * anomaly[unclipped] + intercept, atol=1e-10)


def test_missing_value_propagates_through_its_windows(seasonal: np.ndarray) -> None:
    values = seasonal.copy()
    values[400] = np.nan

    result = _spai(values, scale=3)

    assert np.isnan(result[400:403]).all()
    assert np.isfinite(result[2:400]).all()
    assert np.isfinite(result[403:]).all()


def test_scale_three_leaves_the_first_two_values_missing(seasonal: np.ndarray) -> None:
    result = _spai(seasonal, scale=3)

    assert np.isnan(result[:2]).all()
    assert np.isfinite(result[2:]).all()


@pytest.mark.parametrize("distribution", SPAI_DISTRIBUTIONS, ids=lambda d: d.name)
def test_constant_calendar_period_is_a_constant_not_missing(
    seasonal: np.ndarray, distribution: indices.Distribution
) -> None:
    """A January that never rains has zero anomaly every year, so its SPAI is constant."""
    values = seasonal.copy().reshape(YEARS, 12)
    values[:, 0] = 0.0

    result = _spai(values.ravel(), distribution).reshape(YEARS, 12)

    assert np.isfinite(result).all()
    assert len(np.unique(result[:, 0])) == 1


@pytest.mark.parametrize("distribution", SPAI_DISTRIBUTIONS, ids=lambda d: d.name)
def test_cell_with_no_pooled_spread_is_missing_without_a_warning(
    distribution: indices.Distribution, caplog: pytest.LogCaptureFixture
) -> None:
    """A cell that never rains has no anomaly spread to fit, so it is missing, quietly."""
    caplog.set_level(logging.WARNING)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _spai(np.zeros(YEARS * 12), distribution)

    assert np.isnan(result).all()
    assert not [
        record.msg
        for record in caplog.records
        if isinstance(record.msg, dict) and record.msg.get("event") == "distribution_fallback"
    ]


def test_constant_cell_does_not_disturb_its_neighbours(seasonal: np.ndarray) -> None:
    block = np.stack([seasonal, np.zeros(YEARS * 12)], axis=1)[:, :, np.newaxis]

    result = _spai(block)

    assert np.isnan(result[:, 1, 0]).all()
    np.testing.assert_array_equal(result[:, 0, 0], _spai(seasonal))


def test_calendar_period_with_no_calibration_values_is_missing(seasonal: np.ndarray) -> None:
    """A January the Calibration Period never observes has no mean, so its SPAI is NaN."""
    values = seasonal.copy().reshape(YEARS, 12)
    values[: CALIBRATION_END - CALIBRATION_START + 1, 0] = np.nan

    result = _spai(values.ravel()).reshape(YEARS, 12)

    assert np.isnan(result[:, 0]).all()
    assert np.isfinite(result[:, 1:]).all()


@pytest.mark.parametrize("distribution", SPAI_DISTRIBUTIONS, ids=lambda d: d.name)
def test_data_outside_the_calibration_period_does_not_change_the_fit(
    seasonal: np.ndarray, distribution: indices.Distribution
) -> None:
    """Every output depends on the fit, so unchanged outputs mean unchanged parameters."""
    reference = _spai(seasonal, distribution)
    perturbed = seasonal.copy()
    perturbed[CALIBRATION_STEPS:] *= 3.0  # the ten years after the Calibration Period

    result = _spai(perturbed, distribution)

    np.testing.assert_array_equal(result[:CALIBRATION_STEPS], reference[:CALIBRATION_STEPS])
    assert not np.array_equal(result[CALIBRATION_STEPS:], reference[CALIBRATION_STEPS:])


def test_data_before_the_calibration_period_does_not_change_the_fit() -> None:
    values = _seasonal(years=YEARS)
    late_start = CALIBRATION_START + 5
    reference = indices.spai(values, 1, T, START, late_start, CALIBRATION_END, MONTHLY)
    perturbed = values.copy()
    perturbed[: 5 * 12] = 0.0

    result = indices.spai(perturbed, 1, T, START, late_start, CALIBRATION_END, MONTHLY)

    np.testing.assert_array_equal(result[5 * 12 :], reference[5 * 12 :])


def test_normal_reduces_to_the_pooled_anomaly() -> None:
    """An independent scipy computation on a small array."""
    rng = np.random.default_rng(7)
    precipitation = rng.gamma(2.0, 30.0, size=(6, 12))

    result = indices.spai(precipitation.ravel(), 1, NORMAL, 2000, 2000, 2005, MONTHLY)

    anomaly = precipitation - precipitation.mean(axis=0)
    loc, scale = scipy.stats.norm.fit(anomaly.ravel())
    expected = scipy.stats.norm.ppf(scipy.stats.norm.cdf(anomaly, loc, scale))
    np.testing.assert_allclose(result, np.clip(expected, -3.09, 3.09).ravel(), rtol=0, atol=1e-12)
    # with complete calibration data the pooled anomaly mean is exactly zero, and the
    # maximum-likelihood scale is its root-mean-square deviation
    assert loc == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(result, np.clip(anomaly / scale, -3.09, 3.09).ravel(), rtol=0, atol=1e-12)


def test_t_location_scale_matches_scipy_maximum_likelihood(seasonal: np.ndarray) -> None:
    """Within 1e-3 of scipy's optimizer, with a likelihood no lower (measured 3e-6)."""
    precipitation = seasonal.reshape(YEARS, 12)
    calibration = precipitation[: CALIBRATION_END - CALIBRATION_START + 1]
    anomaly = precipitation - calibration.mean(axis=0)
    pooled = anomaly[: CALIBRATION_END - CALIBRATION_START + 1].ravel()
    df, loc, scale = scipy.stats.t.fit(pooled)
    expected = np.clip(scipy.stats.norm.ppf(scipy.stats.t.cdf(anomaly, df, loc, scale)), -3.09, 3.09).ravel()

    fitted = indices._fit_t_location_scale(np.asfortranarray(pooled[:, np.newaxis]))
    result = _spai(seasonal, T)

    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-3)
    ours = scipy.stats.t.logpdf(pooled, fitted["df"][0], fitted["loc"][0], fitted["scale"][0]).sum()
    assert ours >= scipy.stats.t.logpdf(pooled, df, loc, scale).sum() - 1e-9


def test_probability_and_bounded_scales_share_the_fit(seasonal: np.ndarray) -> None:
    normal_scale = _spai(seasonal)
    probability = _spai(seasonal, output_scale="probability")
    bounded = _spai(seasonal, output_scale="bounded")

    assert ((probability >= 0.0) & (probability <= 1.0)).all()
    np.testing.assert_allclose(bounded, 2.0 * probability - 1.0, rtol=0, atol=1e-15)
    unclipped = np.abs(normal_scale) < 3.09
    np.testing.assert_allclose(scipy.stats.norm.ppf(probability[unclipped]), normal_scale[unclipped], atol=1e-12)


def test_single_year_calibration_period_raises_insufficient_data(seasonal: np.ndarray) -> None:
    with pytest.raises(InsufficientDataError) as error:
        indices.spai(seasonal, 1, T, START, 2000, 2000, MONTHLY)
    assert error.value.required_count == 2


def test_scale_longer_than_the_series_raises_insufficient_data() -> None:
    with pytest.raises(InsufficientDataError):
        indices.spai(np.ones(10), 12, T, START, START, START, MONTHLY)


def test_cell_with_too_few_pooled_calibration_values_is_missing(seasonal: np.ndarray) -> None:
    block = np.stack([seasonal, seasonal], axis=1)[:, :, np.newaxis]
    block[: CALIBRATION_STEPS - 1, 1, 0] = np.nan  # one calibration value left in the cell

    result = _spai(block)

    assert np.isnan(result[:, 1, 0]).all()
    np.testing.assert_array_equal(result[:, 0, 0], _spai(seasonal))


def test_all_missing_input_returns_missing() -> None:
    result = _spai(np.full(YEARS * 12, np.nan))

    assert result.shape == (YEARS * 12,)
    assert np.isnan(result).all()


def test_daily_periodicity(seasonal: np.ndarray) -> None:
    rng = np.random.default_rng(3)
    daily = rng.gamma(0.8, 4.0, size=40 * 366)

    result = indices.spai(daily, 30, T, 1981, 1981, 2010, compute.Periodicity.daily)

    assert result.shape == daily.shape
    assert np.isnan(result[:29]).all()
    assert np.isfinite(result[29:]).all()


@pytest.mark.parametrize(
    "distribution",
    [indices.Distribution.gamma, indices.Distribution.loglogistic, indices.Distribution.empirical],
    ids=lambda d: d.name,
)
def test_rejects_distributions_outside_the_pooled_anomaly_fit(
    seasonal: np.ndarray, distribution: indices.Distribution
) -> None:
    with pytest.raises(InvalidArgumentError) as error:
        _spai(seasonal, distribution)
    assert error.value.argument_name == "distribution"


def test_rejects_hourly_periodicity() -> None:
    with pytest.raises(InvalidArgumentError):
        indices.spai(np.ones(8784 * 2), 24, T, 2000, 2000, 2001, compute.Periodicity.hourly)


@pytest.mark.parametrize("distribution", [T, NORMAL], ids=lambda d: d.name)
def test_spi_spei_and_standardized_index_do_not_accept_the_spai_distributions(
    seasonal: np.ndarray, distribution: indices.Distribution
) -> None:
    calls: tuple[Callable[[], object], ...] = (
        lambda: indices.spi(seasonal, 1, distribution, START, CALIBRATION_START, CALIBRATION_END, MONTHLY),
        lambda: indices.spei(seasonal, seasonal, 1, distribution, MONTHLY, START, CALIBRATION_START, CALIBRATION_END),
        lambda: indices.standardized_index(
            seasonal, 1, distribution, START, CALIBRATION_START, CALIBRATION_END, MONTHLY
        ),
    )
    for call in calls:
        with pytest.raises(InvalidArgumentError, match="spai"):
            call()


def test_pearson_falls_back_to_normal_when_the_l_moments_are_invalid(
    seasonal: np.ndarray, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def invalid_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        zeros = np.zeros(values.shape[1:])
        return zeros, zeros, zeros, np.zeros(values.shape[1:], dtype=bool)

    monkeypatch.setattr(lmoments, "fit_spatial", invalid_fit)
    caplog.set_level(logging.WARNING)

    result = _spai(seasonal, PEARSON)

    np.testing.assert_allclose(result, _spai(seasonal, NORMAL), rtol=0, atol=1e-12)
    fallbacks = [
        record.msg
        for record in caplog.records
        if isinstance(record.msg, dict) and record.msg.get("event") == "distribution_fallback"
    ]
    assert len(fallbacks) == 1
    assert fallbacks[0]["from_distribution"] == "pearson"
    assert fallbacks[0]["to_distribution"] == "normal"


def test_fitting_params_reproduce_the_fit(seasonal: np.ndarray) -> None:
    precipitation = seasonal.reshape(YEARS, 12)
    calibration = precipitation[: CALIBRATION_END - CALIBRATION_START + 1]
    mean = calibration.mean(axis=0)
    loc, scale = scipy.stats.norm.fit((calibration - mean).ravel())

    reused = _spai(seasonal, NORMAL, fitting_params={"mean": mean, "loc": loc, "scale": scale})

    np.testing.assert_allclose(reused, _spai(seasonal, NORMAL), rtol=0, atol=1e-12)


def test_incomplete_fitting_params_raise(seasonal: np.ndarray) -> None:
    with pytest.raises(InvalidArgumentError, match="df") as error:
        _spai(seasonal, T, fitting_params={"mean": np.ones(12), "loc": 0.0, "scale": 1.0})
    assert error.value.argument_name == "fitting_params"


def test_each_public_call_emits_one_beta_warning(seasonal: np.ndarray) -> None:
    data = xr.DataArray(
        np.stack([seasonal] * 4, axis=1).reshape(-1, 2, 2),
        dims=("time", "lat", "lon"),
        coords={"time": xr.date_range(f"{START}-01-01", periods=YEARS * 12, freq="MS")},
    ).chunk({"time": -1, "lat": 1})
    calls: tuple[Callable[[], object], ...] = (
        lambda: indices.spai(seasonal, 1, T, START, CALIBRATION_START, CALIBRATION_END, MONTHLY),
        lambda: climate_indices.spai(seasonal, 1, T, START, CALIBRATION_START, CALIBRATION_END, MONTHLY),
        lambda: climate_indices.spai(data, 1, T).compute(),
    )
    for call in calls:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        assert [warning.category for warning in caught if warning.category is BetaFeatureWarning] == [
            BetaFeatureWarning
        ]


def _gridded(seasonal: np.ndarray) -> xr.DataArray:
    cells = np.stack([_seasonal(seed) for seed in range(6)], axis=1).reshape(-1, 2, 3)
    return xr.DataArray(
        cells,
        dims=("time", "lat", "lon"),
        coords={
            "time": xr.date_range(f"{START}-01-01", periods=YEARS * 12, freq="MS"),
            "lat": [10.0, 20.0],
            "lon": [70.0, 80.0, 90.0],
        },
    )


@pytest.mark.parametrize("distribution", SPAI_DISTRIBUTIONS, ids=lambda d: d.name)
@pytest.mark.parametrize("chunks", [None, {"time": -1, "lat": 1, "lon": 2}], ids=["eager", "dask"])
def test_numpy_and_xarray_paths_are_identical(
    seasonal: np.ndarray, distribution: indices.Distribution, chunks: dict[str, int] | None
) -> None:
    """Gridded, chunked, and per-cell computations give the same values exactly."""
    data = _gridded(seasonal)
    source = data if chunks is None else data.chunk(chunks)

    result = climate_indices.spai(
        source, 3, distribution, calibration_year_initial=CALIBRATION_START, calibration_year_final=CALIBRATION_END
    )

    expected = indices.spai(data.values, 3, distribution, START, CALIBRATION_START, CALIBRATION_END, MONTHLY)
    np.testing.assert_array_equal(result.values, expected)
    for lat in range(2):
        for lon in range(3):
            series = indices.spai(
                data.values[:, lat, lon], 3, distribution, START, CALIBRATION_START, CALIBRATION_END, MONTHLY
            )
            np.testing.assert_array_equal(expected[:, lat, lon], series)
    assert result.dims == data.dims
    assert result.attrs["long_name"] == "Standardized Precipitation Anomaly Index"
    assert result.attrs["distribution"] == distribution.name
    assert "Chanda" in result.attrs["references"]


def test_xarray_series_matches_numpy_and_carries_the_probability_metadata(seasonal: np.ndarray) -> None:
    data = xr.DataArray(
        seasonal, dims="time", coords={"time": xr.date_range(f"{START}-01-01", periods=YEARS * 12, freq="MS")}
    )

    result = climate_indices.spai(
        data,
        1,
        T,
        calibration_year_initial=CALIBRATION_START,
        calibration_year_final=CALIBRATION_END,
        output_scale="probability",
    )

    np.testing.assert_array_equal(result.values, _spai(seasonal, output_scale="probability"))
    assert result.attrs["long_name"] == "Standardized Precipitation Anomaly Index probability"
    assert result.attrs["climate_indices_variant"] == "probability"

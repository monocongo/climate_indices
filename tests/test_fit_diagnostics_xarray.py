"""Tests for the xarray/Dask fit-diagnostics surface.

The diagnostics are an audit view of a fitted distribution, so the tests here
pin parity with :func:`climate_indices.indices.fit_diagnostics`, the calendar
dimension naming, distribution fallback reporting, rejection paths, Dask
laziness, and a NetCDF round-trip.
"""

from __future__ import annotations

import unittest.mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, fit_diagnostics, indices, utils
from climate_indices.compute import FitDiagnostics, Periodicity
from climate_indices.exceptions import CoordinateValidationError, InsufficientDataError
from climate_indices.indices import Distribution

_CALIBRATION_START_YEAR = 1981
_CALIBRATION_END_YEAR = 2010
_MONTHLY_YEARS = 40
_MONTHLY_SERIES = np.arange(1.0, (_MONTHLY_YEARS * 12) + 1.0).reshape(_MONTHLY_YEARS, 12)
_GRID_SERIES = np.arange(1.0, (_MONTHLY_YEARS * 12 * 6) + 1.0).reshape(_MONTHLY_YEARS * 12, 2, 3)


def _monthly_series_da() -> xr.DataArray:
    """A single monthly series as a 1-D DataArray, positive so both fits succeed."""
    time = pd.date_range("1981-01-01", periods=_MONTHLY_SERIES.size, freq="MS")
    return xr.DataArray(
        _MONTHLY_SERIES.flatten(),
        coords={"time": time},
        dims=["time"],
        attrs={"units": "mm"},
    )


def _monthly_grid_da(values: np.ndarray | None = None) -> xr.DataArray:
    """A time-major monthly grid as a 3-D DataArray."""
    time = pd.date_range("1981-01-01", periods=_GRID_SERIES.shape[0], freq="MS")
    return xr.DataArray(
        _GRID_SERIES if values is None else values,
        coords={"time": time, "lat": [35.0, 40.0], "lon": [-100.0, -95.0, -90.0]},
        dims=["time", "lat", "lon"],
        attrs={"units": "mm"},
    )


def _expected_array(diagnostics: FitDiagnostics, name: str) -> np.ndarray:
    """Read a variable from a NumPy FitDiagnostics, parameter dict first."""
    if name in diagnostics.parameters:
        return diagnostics.parameters[name]
    return getattr(diagnostics, name)


def test_numpy_input_returns_fit_diagnostics() -> None:
    """The NumPy passthrough returns what ``indices.fit_diagnostics`` returns."""
    result = fit_diagnostics(
        _MONTHLY_SERIES,
        1,
        Distribution.gamma,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        Periodicity.monthly,
    )
    expected = indices.fit_diagnostics(
        _MONTHLY_SERIES,
        1,
        Distribution.gamma,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        Periodicity.monthly,
    )

    assert isinstance(result, FitDiagnostics)
    assert set(result.parameters) == {"alpha", "beta"}
    np.testing.assert_array_equal(result.parameters["alpha"], expected.parameters["alpha"])
    np.testing.assert_array_equal(result.ks_p_value, expected.ks_p_value)


@pytest.mark.parametrize("scale", [1, 3])
def test_monthly_series_matches_numpy(scale: int) -> None:
    """A 1-D xarray series yields the NumPy diagnostics over a ``month`` dimension."""
    result = fit_diagnostics(
        _monthly_series_da(),
        scale,
        Distribution.gamma,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )
    expected = indices.fit_diagnostics(
        _MONTHLY_SERIES,
        scale,
        Distribution.gamma,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        Periodicity.monthly,
    )

    assert set(result.data_vars) == {
        "alpha",
        "beta",
        "prob_zero",
        "n_valid",
        "ks_statistic",
        "ks_p_value",
        "distribution_used",
    }
    assert result["alpha"].dims == ("month",)
    np.testing.assert_array_equal(result["month"].values, np.arange(1, 13))
    for name in ("alpha", "beta", "prob_zero", "n_valid", "ks_statistic", "ks_p_value"):
        np.testing.assert_allclose(result[name].values, _expected_array(expected, name), equal_nan=True)
    assert result["distribution_used"].item() == "gamma"
    assert result["distribution_used"].dtype.kind in "US"
    assert result["distribution_used"].attrs["long_name"] == "Distribution used after any fall back"
    assert "units" not in result["distribution_used"].attrs
    assert result["alpha"].attrs["long_name"] == "Gamma shape parameter"
    assert result["alpha"].attrs["units"] == "1"
    assert result["alpha"].attrs["scale"] == scale
    assert result["alpha"].attrs["distribution"] == "gamma"
    assert result["alpha"].attrs["calibration_year_initial"] == _CALIBRATION_START_YEAR
    assert result["alpha"].attrs["references"]
    assert "history" in result["alpha"].attrs


def test_monthly_series_infers_temporal_parameters() -> None:
    """Omitting the temporal parameters infers them, matching the explicit call."""
    chain = _monthly_series_da()
    explicit = fit_diagnostics(
        chain,
        1,
        Distribution.gamma,
        calibration_year_initial=1981,
        calibration_year_final=2020,
    )
    inferred = fit_diagnostics(chain, 1, Distribution.gamma)

    for name in ("alpha", "beta", "prob_zero", "n_valid", "ks_statistic", "ks_p_value"):
        np.testing.assert_array_equal(inferred[name].values, explicit[name].values)
    assert inferred["alpha"].attrs["data_start_year"] == 1981
    assert inferred["alpha"].attrs["calibration_year_final"] == 2020


@pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson])
def test_grid_matches_numpy_block(distribution: Distribution) -> None:
    """A gridded xarray input matches the NumPy diagnostics on the folded block."""
    result = fit_diagnostics(
        _monthly_grid_da(),
        1,
        distribution,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )
    expected = indices.fit_diagnostics(
        _GRID_SERIES,
        1,
        distribution,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        Periodicity.monthly,
        spatial_time_major=True,
    )

    cell_dims = ("lat", "lon")
    parameter_names = ("alpha", "beta") if distribution is Distribution.gamma else ("prob_zero", "loc", "scale", "skew")
    for name in (*parameter_names, "n_valid", "ks_statistic", "ks_p_value"):
        variable = result[name]
        assert variable.dims == ("month", *cell_dims)
        np.testing.assert_allclose(variable.values, _expected_array(expected, name), equal_nan=True)
    if distribution is Distribution.pearson:
        # the gamma parameter slots stay in the schema but do not apply here
        assert result["alpha"].isnull().all()
        assert result["beta"].isnull().all()
    else:
        assert "loc" not in result
    assert result["distribution_used"].dims == cell_dims
    assert set(np.unique(result["distribution_used"].values)) == {distribution.value}


@pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson])
def test_two_dimensional_grid_matches_per_cell_numpy(distribution: Distribution) -> None:
    """A 2-D ``(time, cells)`` input fits each cell, matching per-cell NumPy."""
    values = np.arange(1.0, (_MONTHLY_SERIES.size * 2) + 1.0).reshape(_MONTHLY_SERIES.size, 2)
    time = pd.date_range("1981-01-01", periods=_MONTHLY_SERIES.size, freq="MS")
    result = fit_diagnostics(
        xr.DataArray(values, coords={"time": time, "station": ["a", "b"]}, dims=["time", "station"]),
        1,
        distribution,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )

    assert result["alpha"].dims == ("month", "station")
    for index in range(values.shape[1]):
        expected = indices.fit_diagnostics(
            values[:, index].copy(),
            1,
            distribution,
            _CALIBRATION_START_YEAR,
            _CALIBRATION_START_YEAR,
            _CALIBRATION_END_YEAR,
            Periodicity.monthly,
        )
        np.testing.assert_allclose(result["ks_statistic"].isel(station=index).values, expected.ks_statistic)
        assert result["distribution_used"].isel(station=index).item() == expected.distribution.value


def test_grid_with_one_degenerate_cell_matches_block_numpy() -> None:
    """A fit-failing cell keeps the block's distribution, matching the NumPy block."""
    values = _GRID_SERIES.copy()
    values[:, 0, 0] = 5.0
    result = fit_diagnostics(
        _monthly_grid_da(values),
        1,
        Distribution.pearson,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )
    expected = indices.fit_diagnostics(
        values,
        1,
        Distribution.pearson,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        Periodicity.monthly,
        spatial_time_major=True,
    )

    # the block-level fall back is shared, so the degenerate cell does not become gamma
    assert set(np.unique(result["distribution_used"].values)) == {"pearson"}
    np.testing.assert_allclose(result["loc"].values, expected.parameters["loc"], equal_nan=True)
    assert result["ks_statistic"].isel(lat=0, lon=0).isnull().all()


def test_pearson_fallback_reports_the_gamma_fit_and_values() -> None:
    """A failed Pearson fit falls back to gamma, and the gamma values are the fit's."""
    with unittest.mock.patch(
        "climate_indices.compute.pearson_parameters",
        side_effect=compute.DistributionFittingError("Pearson fit failed", distribution_name="pearson3"),
    ):
        result = fit_diagnostics(
            _monthly_grid_da(),
            1,
            Distribution.pearson,
            calibration_year_initial=_CALIBRATION_START_YEAR,
            calibration_year_final=_CALIBRATION_END_YEAR,
        )
    expected = indices.fit_diagnostics(
        _GRID_SERIES,
        1,
        Distribution.gamma,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        Periodicity.monthly,
        spatial_time_major=True,
    )

    assert {"alpha", "beta", "prob_zero", "n_valid", "ks_statistic", "ks_p_value", "distribution_used"} <= set(
        result.data_vars
    )
    assert result["alpha"].notnull().all()
    for name in ("alpha", "beta", "prob_zero", "n_valid", "ks_statistic", "ks_p_value"):
        np.testing.assert_allclose(result[name].values, _expected_array(expected, name), equal_nan=True)
    for name in ("loc", "scale", "skew"):
        assert result[name].isnull().all()
    assert set(np.unique(result["distribution_used"].values)) == {"gamma"}


def test_daily_input_uses_dayofyear_dimension() -> None:
    """Daily input gets a ``dayofyear`` dimension and the 366-day calendar positions."""
    total_years = 3
    days = pd.date_range("2001-01-01", periods=total_years * 365, freq="D")
    values = np.arange(1.0, len(days) + 1.0)
    result = fit_diagnostics(
        xr.DataArray(values, coords={"time": days}, dims=["time"]),
        1,
        Distribution.gamma,
        calibration_year_initial=2001,
        calibration_year_final=2003,
    )
    expected = indices.fit_diagnostics(
        utils.transform_to_366day(values, 2001, total_years),
        1,
        Distribution.gamma,
        2001,
        2001,
        2003,
        Periodicity.daily,
    )

    assert result["alpha"].dims == ("dayofyear",)
    np.testing.assert_array_equal(result["dayofyear"].values, np.arange(1, 367))
    np.testing.assert_allclose(result["alpha"].values, _expected_array(expected, "alpha"))


def test_partial_final_year_daily_matches_numpy() -> None:
    """A daily series with a partial final year uses the same padded calendar plan."""
    days = pd.date_range("1999-01-01", "2001-06-30", freq="D")
    values = np.arange(1.0, len(days) + 1.0)
    result = fit_diagnostics(
        xr.DataArray(values, coords={"time": days}, dims=["time"]),
        1,
        Distribution.gamma,
        calibration_year_initial=1999,
        calibration_year_final=2001,
    )
    expected = indices.fit_diagnostics(
        utils.transform_to_366day(values, 1999, 3),
        1,
        Distribution.gamma,
        1999,
        1999,
        2001,
        Periodicity.daily,
    )

    assert result["alpha"].dims == ("dayofyear",)
    np.testing.assert_allclose(result["alpha"].values, _expected_array(expected, "alpha"))


def test_dataset_variables_cover_the_numpy_diagnostics_contract() -> None:
    """Every NumPy diagnostics field and fitted-parameter key is present in the Dataset."""
    for distribution in (Distribution.gamma, Distribution.pearson):
        probe = indices.fit_diagnostics(
            _MONTHLY_SERIES,
            1,
            distribution,
            _CALIBRATION_START_YEAR,
            _CALIBRATION_START_YEAR,
            _CALIBRATION_END_YEAR,
            Periodicity.monthly,
        )
        result = fit_diagnostics(
            _monthly_series_da(),
            1,
            distribution,
            calibration_year_initial=_CALIBRATION_START_YEAR,
            calibration_year_final=_CALIBRATION_END_YEAR,
        )

        assert {"prob_zero", "n_valid", "ks_statistic", "ks_p_value"} <= set(result.data_vars)
        assert set(probe.parameters) <= set(result.data_vars)


def test_supplied_fitting_params_reproduce_the_fit() -> None:
    """A prior fit's parameters reproduce the same diagnostics and are recorded."""
    chain = _monthly_series_da()
    first = fit_diagnostics(
        chain,
        3,
        Distribution.gamma,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )
    supplied = {name: first[name].values for name in ("alpha", "beta")}
    second = fit_diagnostics(
        chain,
        3,
        Distribution.gamma,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
        fitting_params=supplied,
    )

    np.testing.assert_array_equal(second["alpha"].values, first["alpha"].values)
    np.testing.assert_array_equal(second["ks_p_value"].values, first["ks_p_value"].values)
    assert second["alpha"].attrs["fitting_params"] == "dict(keys=alpha,beta)"
    assert "fitting_params" not in first["alpha"].attrs


@pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson])
def test_all_missing_input_reports_no_valid_sample(distribution: Distribution) -> None:
    """An all-NaN series yields a Dataset of missing diagnostics rather than an error."""
    chain = _monthly_series_da()
    missing = chain.where(chain.time.dt.year < 1900)
    result = fit_diagnostics(
        missing,
        1,
        distribution,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )

    assert (result["n_valid"] == 0).all()
    assert result["ks_statistic"].isnull().all()
    assert result["ks_p_value"].isnull().all()
    assert result["distribution_used"].item() == distribution.value


def test_scale_longer_than_the_series_is_rejected() -> None:
    """The adapter rejects a scale the series cannot support, as the NumPy core does."""
    with pytest.raises(InsufficientDataError, match="Insufficient data for scale"):
        fit_diagnostics(
            _monthly_series_da(),
            1000,
            Distribution.gamma,
            calibration_year_initial=_CALIBRATION_START_YEAR,
            calibration_year_final=_CALIBRATION_END_YEAR,
        )


def test_rejects_monthly_input_that_does_not_begin_in_january() -> None:
    """Monthly input must begin in January, matching the index adapters."""
    time = pd.date_range("2000-03-01", periods=24, freq="MS")
    with pytest.raises(CoordinateValidationError, match="begin in January"):
        fit_diagnostics(
            xr.DataArray(np.arange(1.0, 25.0), coords={"time": time}, dims=["time"]),
            1,
            Distribution.gamma,
        )


def test_rejects_daily_input_that_does_not_begin_on_january_first() -> None:
    """Daily input must begin on January 1, matching the index adapters."""
    time = pd.date_range("2000-01-02", periods=365, freq="D")
    with pytest.raises(CoordinateValidationError, match="begin on January 1"):
        fit_diagnostics(
            xr.DataArray(np.arange(1.0, 366.0), coords={"time": time}, dims=["time"]),
            1,
            Distribution.gamma,
        )


def test_rejects_a_time_dimension_split_across_dask_chunks() -> None:
    """The time dimension must be a single chunk; rechunking is left to the caller."""
    with pytest.raises(CoordinateValidationError, match="single chunk"):
        fit_diagnostics(
            _monthly_series_da().chunk({"time": 120}),
            1,
            Distribution.gamma,
        )


def test_dask_grid_stays_lazy_and_matches_numpy() -> None:
    """A chunked grid returns a lazy Dataset whose values match the NumPy block."""
    chunked = _monthly_grid_da().chunk({"time": -1, "lat": 1, "lon": 2})
    result = fit_diagnostics(
        chunked,
        1,
        Distribution.gamma,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )
    expected = indices.fit_diagnostics(
        _GRID_SERIES,
        1,
        Distribution.gamma,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        Periodicity.monthly,
        spatial_time_major=True,
    )

    assert result["alpha"].chunks is not None
    np.testing.assert_allclose(result["alpha"].compute().values, _expected_array(expected, "alpha"))
    np.testing.assert_allclose(result["ks_p_value"].compute().values, expected.ks_p_value)


@pytest.mark.parametrize("distribution", [Distribution.gamma, Distribution.pearson])
def test_netcdf_round_trip(tmp_path, distribution: Distribution) -> None:
    """The Dataset, its calendar dimension, and its provenance survive a NetCDF write."""
    result = fit_diagnostics(
        _monthly_series_da(),
        1,
        distribution,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )
    path = tmp_path / "fit_diagnostics.nc"
    result.to_netcdf(path, engine="h5netcdf")

    with xr.open_dataset(path, engine="h5netcdf") as reopened:
        assert set(reopened.data_vars) == set(result.data_vars)
        np.testing.assert_array_equal(reopened["month"].values, np.arange(1, 13))
        for name in result.data_vars:
            if name == "distribution_used":
                np.testing.assert_array_equal(reopened[name].values, result[name].values)
            else:
                np.testing.assert_allclose(reopened[name].values, result[name].values, equal_nan=True)
        assert reopened["alpha"].attrs["long_name"] == "Gamma shape parameter"
        assert reopened["alpha"].attrs["units"] == "1"
        assert reopened["alpha"].attrs["distribution"] == distribution.value
        assert reopened["alpha"].attrs["calibration_year_initial"] == _CALIBRATION_START_YEAR
        assert "history" in reopened["alpha"].attrs

"""Tests for the xarray/Dask fit-diagnostics surface.

The diagnostics are an audit view of a fitted distribution, so the tests here
pin parity with :func:`climate_indices.indices.fit_diagnostics`, the calendar
dimension naming, distribution fallback reporting, Dask laziness, and a NetCDF
round-trip.
"""

from __future__ import annotations

import unittest.mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, fit_diagnostics, indices, utils
from climate_indices.compute import FitDiagnostics, Periodicity
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


def _monthly_grid_da() -> xr.DataArray:
    """A time-major monthly grid as a 3-D DataArray."""
    time = pd.date_range("1981-01-01", periods=_GRID_SERIES.shape[0], freq="MS")
    return xr.DataArray(
        _GRID_SERIES,
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

    assert isinstance(result, FitDiagnostics)
    np.testing.assert_array_equal(result.parameters["alpha"], _expected_array(result, "alpha"))


def test_monthly_series_matches_numpy() -> None:
    """A 1-D xarray series yields the NumPy diagnostics over a ``month`` dimension."""
    result = fit_diagnostics(
        _monthly_series_da(),
        1,
        Distribution.gamma,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
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
    assert result["alpha"].attrs["long_name"] == "Gamma shape parameter"
    assert result["alpha"].attrs["units"] == "1"
    assert result["alpha"].attrs["scale"] == 1
    assert result["alpha"].attrs["distribution"] == "gamma"
    assert result["alpha"].attrs["calibration_year_initial"] == _CALIBRATION_START_YEAR
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
    assert result["prob_zero"].dims == ("month", *cell_dims)
    assert result["distribution_used"].dims == cell_dims
    assert set(np.unique(result["distribution_used"].values)) == {distribution.value}


def test_pearson_fallback_reports_the_distribution_actually_used() -> None:
    """A failed Pearson fit falls back to gamma and says so in the Dataset."""
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

    assert {"alpha", "beta", "prob_zero", "n_valid", "ks_statistic", "ks_p_value", "distribution_used"} <= set(
        result.data_vars
    )
    assert result["alpha"].notnull().all()
    assert result["beta"].notnull().all()
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


def test_netcdf_round_trip(tmp_path) -> None:
    """The Dataset, its calendar dimension, and its provenance survive a NetCDF write."""
    result = fit_diagnostics(
        _monthly_series_da(),
        1,
        Distribution.gamma,
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
        assert reopened["alpha"].attrs["distribution"] == "gamma"
        assert reopened["alpha"].attrs["calibration_year_initial"] == _CALIBRATION_START_YEAR
        assert "history" in reopened["alpha"].attrs

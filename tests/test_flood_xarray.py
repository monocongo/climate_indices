"""NumPy-equivalent daily flood adapters, including Gregorian and Dask blocks."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import edi, flood
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import CoordinateValidationError, InvalidArgumentError
from climate_indices.utils import DailyCalendarPlan


def _rain_grid() -> xr.DataArray:
    dates = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    day = np.arange(len(dates))
    values = (1 + (day % 31) / 5 + (day // 366) * 2)[:, None, None] * np.ones((1, 2, 3))
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": dates, "lat": [10, 20], "lon": [0, 1, 2]},
        attrs={"units": "mm", "standard_name": "precipitation_amount", "history": "input"},
    )


def test_flood_xarray_matches_numpy_on_gregorian_grid_and_preserves_metadata() -> None:
    rain = _rain_grid()
    plan = DailyCalendarPlan.from_year_span(2000, 5, rain.sizes["time"])
    expected_pe = plan.to_gregorian(flood.effective_precipitation(plan.to_all_leap(rain.values), duration=30))
    pe = flood.effective_precipitation(rain, duration=30)
    xr.testing.assert_equal(pe.time, rain.time)
    np.testing.assert_allclose(pe.values, expected_pe, equal_nan=True)
    assert pe.attrs["units"] == "mm"
    assert pe.attrs["long_name"] == CF_METADATA["effective_precipitation"]["long_name"]
    assert "standard_name" not in pe.attrs
    assert "input" in pe.attrs["history"]
    assert "climate_indices_version" in pe.attrs

    assert flood.edi(pe).attrs["duration"] == 30  # PE provenance, not EDI's no-op default
    all_leap_pe = plan.to_all_leap(pe.values)
    for actual, expected in (
        (flood.edi(pe, duration=30), flood.edi(all_leap_pe, 2000, 2000, 2004, duration=30, spatial_time_major=True)),
        (
            flood.flood_index(pe, year_start_month=1),
            flood.flood_index(all_leap_pe, 2000, 2000, 2004, year_start_month=1, spatial_time_major=True),
        ),
    ):
        assert isinstance(actual, xr.DataArray)
        np.testing.assert_allclose(actual.values, plan.to_gregorian(expected), equal_nan=True)
        assert actual.dims == rain.dims
        assert actual.attrs["units"] == "dimensionless"
        assert "standard_name" not in actual.attrs
    assert edi is flood.edi


def test_flood_dask_keeps_spatial_chunks_and_requires_full_time_chunk() -> None:
    rain = _rain_grid().chunk({"time": -1, "lat": 1, "lon": 2})
    pe = flood.effective_precipitation(rain, duration=30)
    assert pe.chunks is not None
    assert pe.chunks[0] == (rain.sizes["time"],)
    assert pe.chunks[1:] == rain.chunks[1:]
    eager_pe = flood.effective_precipitation(_rain_grid(), duration=30)
    for result, expected in (
        (pe, eager_pe),
        (flood.edi(pe, duration=30), flood.edi(eager_pe, duration=30)),
        (flood.flood_index(pe, year_start_month=1), flood.flood_index(eager_pe, year_start_month=1)),
    ):
        np.testing.assert_allclose(result.compute().values, expected.values, equal_nan=True)
    split_rain = rain.chunk({"time": 100})
    split_pe = pe.chunk({"time": 100})
    with pytest.raises(CoordinateValidationError, match="chunk"):
        flood.effective_precipitation(split_rain)
    with pytest.raises(CoordinateValidationError, match="chunk"):
        flood.edi(split_pe, duration=30)


def test_api_xarray_spatial_chunks_units_and_resume() -> None:
    dates = pd.date_range("2001-01-01", periods=12)
    rain = xr.DataArray(
        np.arange(36, dtype=float).reshape(12, 3) / 25.4,
        dims=("time", "site"),
        coords={"time": dates, "site": [1, 2, 3]},
        attrs={"units": "inch", "standard_name": "precipitation_amount"},
    ).chunk({"time": -1, "site": 1})
    expected = flood.antecedent_precipitation_index((rain.values * 25.4)[..., None], 0.85, spatial_time_major=True)[
        ..., 0
    ]
    result = flood.antecedent_precipitation_index(rain, 0.85)
    assert isinstance(result, xr.DataArray)
    assert result.chunks == rain.chunks
    np.testing.assert_array_equal(result.compute().values, expected)
    assert result.attrs["units"] == "mm"
    assert "standard_name" not in result.attrs

    first = flood.antecedent_precipitation_index(rain.isel(time=slice(None, 6)), 0.85, return_state=True)
    assert isinstance(first, flood.APIResult)
    assert isinstance(first.values, xr.DataArray)
    assert isinstance(first.state.api, np.ndarray)
    second = flood.antecedent_precipitation_index(
        rain.isel(time=slice(6, None)), 0.85, initial_state=first.state, return_state=True
    )
    assert isinstance(second, flood.APIResult)
    np.testing.assert_array_equal(xr.concat([first.values, second.values], dim="time"), result.compute())
    np.testing.assert_array_equal(first.state.api, expected[5])
    np.testing.assert_array_equal(second.state.api, expected[-1])
    rechunked = rain.chunk({"time": 3})
    strided = rain.isel(time=slice(None, None, 2))
    with pytest.raises(CoordinateValidationError, match="Rechunk"):
        flood.antecedent_precipitation_index(rechunked, 0.85)
    with pytest.raises(CoordinateValidationError, match="daily"):
        flood.antecedent_precipitation_index(strided, 0.85)


def test_api_xarray_gap_state_and_time_last() -> None:
    rain = xr.DataArray(
        [[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]],
        dims=("time", "site"),
        coords={"time": pd.date_range("2001-01-01", periods=3)},
    ).transpose("site", "time")
    result = flood.antecedent_precipitation_index(
        rain, 0.9, nan_policy="bridge", max_gap_days=1, spin_up=1, return_state=True
    )
    assert isinstance(result, flood.APIResult)
    assert isinstance(result.values, xr.DataArray)
    assert result.values.dims == rain.dims
    xr.testing.assert_equal(result.values.time, rain.time.isel(time=slice(1, None)))
    expected = flood.antecedent_precipitation_index(
        rain.transpose("time", "site").values[..., None],
        0.9,
        nan_policy="bridge",
        max_gap_days=1,
        spin_up=1,
        return_state=True,
        spatial_time_major=True,
    )
    assert isinstance(expected, flood.APIResult)
    np.testing.assert_array_equal(result.values.transpose("time", "site"), expected.values[..., 0])
    np.testing.assert_array_equal(result.state.api, expected.state.api[..., 0])
    np.testing.assert_array_equal(result.state.trailing_gap_days, expected.state.trailing_gap_days[..., 0])


def test_api_xarray_time_only_accepts_scalar_state() -> None:
    rain = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": pd.date_range("2001-01-01", periods=4)})
    seeded = flood.antecedent_precipitation_index(np.ones((4, 1)), 0.5, return_state=True)
    assert isinstance(seeded, flood.APIResult)
    resumed = flood.antecedent_precipitation_index(rain, 0.5, initial_state=seeded.state, return_state=True)
    assert isinstance(resumed, flood.APIResult)
    expected = flood.antecedent_precipitation_index(rain.values, 0.5, initial_state=seeded.state)
    np.testing.assert_array_equal(resumed.values, expected)


def test_api_xarray_time_only_and_ambiguous_spatial_axis() -> None:
    dates = pd.date_range("2001-01-01", periods=4)
    rain = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": dates})
    result = flood.antecedent_precipitation_index(rain, 0.5, return_state=True)
    assert isinstance(result, flood.APIResult)
    np.testing.assert_array_equal(result.values, [1.0, 2.5, 4.25, 6.125])
    assert result.state.api.shape == ()
    grid = rain.expand_dims(site=range(12)).transpose("time", "site")
    gridded = flood.antecedent_precipitation_index(grid.chunk({"time": -1, "site": 3}), 0.5)
    np.testing.assert_array_equal(gridded.compute(), np.broadcast_to(result.values.values[:, None], (4, 12)))


def test_api_xarray_spin_up_keeps_time_coordinates_and_name() -> None:
    dates = pd.date_range("2001-01-01", periods=6)
    rain = xr.DataArray(
        np.arange(6.0),
        dims="time",
        name="precip",
        coords={
            "time": ("time", dates, {"long_name": "observation time"}),
            "month": ("time", dates.month.values, {"units": "1"}),
        },
        attrs={"units": "mm"},
    )
    eager = flood.antecedent_precipitation_index(rain, 0.9, spin_up=2)
    stateful = flood.antecedent_precipitation_index(rain, 0.9, spin_up=2, return_state=True)
    assert isinstance(eager, xr.DataArray)
    assert isinstance(stateful, flood.APIResult)
    for values in (eager, stateful.values):
        assert values.name == "precip"
        assert values.time.attrs == {"long_name": "observation time"}
        assert values.month.attrs == {"units": "1"}
        np.testing.assert_array_equal(values.month, rain.month.isel(time=slice(2, None)))
    unnamed = flood.antecedent_precipitation_index(rain.rename(None), 0.9, return_state=True)
    assert isinstance(unnamed, flood.APIResult)
    assert unnamed.values.name is None


def test_flood_infers_only_complete_calibration_periods() -> None:
    pe = flood.effective_precipitation(_rain_grid().isel(time=slice(None, -60)), duration=30)
    assert flood.edi(pe, duration=30).attrs["calibration_year_final"] == 2003
    assert flood.flood_index(pe, year_start_month=1).attrs["calibration_year_final"] == 2003
    assert flood.flood_index(pe, year_start_month=7).attrs["calibration_year_final"] == 2003
    earlier = pe.isel(time=slice(None, -300))
    assert flood.flood_index(earlier, year_start_month=7).attrs["calibration_year_final"] == 2002
    june_30 = pe.sel(time=slice(None, "2004-06-30"))
    inferred = flood.flood_index(june_30, year_start_month=7)
    assert inferred.attrs["calibration_year_final"] == 2003
    np.testing.assert_allclose(
        inferred, flood.flood_index(june_30, year_start_month=7, calibration_year_final=2003), equal_nan=True
    )


def test_flood_units_calendar_and_numpy_passthrough() -> None:
    rain = _rain_grid()
    inches = rain.assign_attrs(units="inch") / 25.4
    inches.attrs["units"] = "inch"
    np.testing.assert_allclose(
        flood.effective_precipitation(inches, duration=30),
        flood.effective_precipitation(rain, duration=30),
        equal_nan=True,
    )
    pe = flood.effective_precipitation(rain, duration=30)
    pe_inches = pe / 25.4
    pe_inches.attrs["units"] = "inch"
    np.testing.assert_allclose(flood.edi(pe_inches, duration=30), flood.edi(pe, duration=30), equal_nan=True)
    np.testing.assert_allclose(
        flood.flood_index(pe_inches, year_start_month=1),
        flood.flood_index(pe, year_start_month=1),
        equal_nan=True,
    )
    no_units = rain.copy()
    del no_units.attrs["units"]
    flood.effective_precipitation(no_units, duration=30)
    assert "units" not in no_units.attrs
    time_last = rain.transpose("lat", "lon", "time")
    time_last.time.attrs["axis"] = "T"
    result = flood.effective_precipitation(time_last, duration=30)
    result.time.attrs["axis"] = "other"
    assert time_last.time.attrs["axis"] == "T"
    invalid_units = rain.assign_attrs(units="bananas")
    missing_january_1 = rain.isel(time=slice(1, None))
    monthly_rain = rain.resample(time="MS").sum()
    with pytest.raises(InvalidArgumentError, match="units"):
        flood.effective_precipitation(invalid_units)
    with pytest.raises(CoordinateValidationError, match="January 1"):
        flood.effective_precipitation(missing_january_1)
    with pytest.raises(CoordinateValidationError, match="periodicity"):
        flood.effective_precipitation(monthly_rain)
    values = np.ones(60)
    np.testing.assert_allclose(flood.effective_precipitation(values, duration=30)[29:], 30.0)

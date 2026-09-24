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
    with pytest.raises(CoordinateValidationError, match="chunk"):
        flood.effective_precipitation(rain.chunk({"time": 100}))
    with pytest.raises(CoordinateValidationError, match="chunk"):
        flood.edi(pe.chunk({"time": 100}), duration=30)


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
    with pytest.raises(InvalidArgumentError, match="units"):
        flood.effective_precipitation(rain.assign_attrs(units="bananas"))
    with pytest.raises(CoordinateValidationError, match="January 1"):
        flood.effective_precipitation(rain.isel(time=slice(1, None)))
    with pytest.raises(CoordinateValidationError, match="periodicity"):
        flood.effective_precipitation(rain.resample(time="MS").sum())
    values = np.ones(60)
    np.testing.assert_allclose(flood.effective_precipitation(values, duration=30)[29:], 30.0)

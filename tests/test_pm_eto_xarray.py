"""Tests for the xarray Penman-Monteith PET adapter.

Values are checked against the NumPy path and, for the Uccle Example 18 case,
against the FAO-56 printed result. Metadata, dask laziness, and the shared
January-start daily calendar contract are checked alongside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import HumidityInputs, RadiationInputs, pet_penman_monteith
from climate_indices.exceptions import CoordinateValidationError
from climate_indices.xarray_adapter import pet_penman_monteith as pet_penman_monteith_impl


def _daily_coord(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=periods, freq="D")


def _example18_arrays(periods: int) -> dict[str, np.ndarray]:
    return {
        "tmin": np.full(periods, 12.3),
        "tmax": np.full(periods, 21.5),
        "wind": np.full(periods, 2.78),
        "rh_min": np.full(periods, 63.0),
        "rh_max": np.full(periods, 84.0),
        "sunshine": np.full(periods, 9.25),
    }


class TestPenmanMonteithXarrayEquivalence:
    def test_series_matches_numpy_path(self) -> None:
        periods = 366
        time = _daily_coord(periods)
        values = _example18_arrays(periods)

        coords = {"time": time}
        tmin = xr.DataArray(values["tmin"], coords=coords, dims=["time"])
        tmax = xr.DataArray(values["tmax"], coords=coords, dims=["time"])
        wind = xr.DataArray(values["wind"], coords=coords, dims=["time"])
        rh_min = xr.DataArray(values["rh_min"], coords=coords, dims=["time"])
        rh_max = xr.DataArray(values["rh_max"], coords=coords, dims=["time"])
        sunshine = xr.DataArray(values["sunshine"], coords=coords, dims=["time"])

        result = pet_penman_monteith(
            tmin,
            tmax,
            latitude=50.80,
            elevation_m=100.0,
            wind_speed_m_s=wind,
            wind_speed_height_m=10.0,
            humidity=HumidityInputs(rh_min=rh_min, rh_max=rh_max),
            radiation=RadiationInputs(sunshine_hours=sunshine),
        )

        expected = pet_penman_monteith(
            values["tmin"],
            values["tmax"],
            50.80,
            100.0,
            values["wind"],
            time.dayofyear.to_numpy(),
            10.0,
            humidity=HumidityInputs(rh_min=values["rh_min"], rh_max=values["rh_max"]),
            radiation=RadiationInputs(sunshine_hours=values["sunshine"]),
        )
        np.testing.assert_allclose(result.values, expected, rtol=1e-9, atol=1e-9)

        # FAO-56 Example 18 falls on 6 July, i.e. day 187 -> 3.88 mm/day
        assert float(result.isel(time=186)) == pytest.approx(3.88, abs=0.01)

    def test_gridded_matches_per_cell_numpy(self) -> None:
        periods = 5
        time = _daily_coord(periods)
        lats = [30.0, 40.0]
        lons = [-100.0, -90.0]
        shape = (periods, len(lats), len(lons))
        values = _example18_arrays(periods)

        def broadcast(array: np.ndarray) -> xr.DataArray:
            return xr.DataArray(
                np.broadcast_to(array[:, None, None], shape),
                coords={"time": time, "lat": lats, "lon": lons},
                dims=["time", "lat", "lon"],
            )

        latitude = xr.DataArray(lats, coords={"lat": lats}, dims=["lat"])
        result = pet_penman_monteith(
            broadcast(values["tmin"]),
            broadcast(values["tmax"]),
            latitude=latitude,
            elevation_m=100.0,
            wind_speed_m_s=broadcast(values["wind"]),
            wind_speed_height_m=10.0,
            humidity=HumidityInputs(rh_min=broadcast(values["rh_min"]), rh_max=broadcast(values["rh_max"])),
            radiation=RadiationInputs(sunshine_hours=broadcast(values["sunshine"])),
        )

        assert result.dims == ("time", "lat", "lon")
        expected = pet_penman_monteith_impl(
            values["tmin"],
            values["tmax"],
            30.0,
            100.0,
            values["wind"],
            time.dayofyear.to_numpy(),
            10.0,
            humidity=HumidityInputs(rh_min=values["rh_min"], rh_max=values["rh_max"]),
            radiation=RadiationInputs(sunshine_hours=values["sunshine"]),
        )
        np.testing.assert_allclose(result.isel(lat=0, lon=0).values, expected, rtol=1e-9, atol=1e-9)

    def test_dask_input_stays_lazy(self) -> None:
        periods = 20
        time = _daily_coord(periods)
        values = _example18_arrays(periods)
        tmin = xr.DataArray(values["tmin"], coords={"time": time}, dims=["time"]).chunk({"time": 5})
        tmax = xr.DataArray(values["tmax"], coords={"time": time}, dims=["time"]).chunk({"time": 5})

        result = pet_penman_monteith(
            tmin,
            tmax,
            latitude=50.80,
            elevation_m=100.0,
            wind_speed_m_s=2.78,
            humidity=HumidityInputs(rh_min=63.0, rh_max=84.0),
            radiation=RadiationInputs(sunshine_hours=9.25),
        )

        assert result.chunks is not None

    def test_metadata_and_history(self) -> None:
        periods = 3
        time = _daily_coord(periods)
        tmin = xr.DataArray(np.full(periods, 12.3), coords={"time": time}, dims=["time"])
        tmax = xr.DataArray(np.full(periods, 21.5), coords={"time": time}, dims=["time"])

        result = pet_penman_monteith(tmin, tmax, latitude=50.80, elevation_m=100.0, wind_speed_m_s=2.78)

        assert result.attrs["units"] == "mm/day"
        assert "Penman-Monteith" in result.attrs["long_name"]
        assert "climate_indices_version" in result.attrs
        assert "history" in result.attrs
        assert "latitude" in result.attrs

    def test_rejects_non_january_start(self) -> None:
        time = pd.date_range("2015-02-01", periods=10, freq="D")
        tmin = xr.DataArray(np.full(10, 12.3), coords={"time": time}, dims=["time"])
        tmax = xr.DataArray(np.full(10, 21.5), coords={"time": time}, dims=["time"])

        with pytest.raises(CoordinateValidationError):
            pet_penman_monteith(tmin, tmax, latitude=50.80, elevation_m=100.0, wind_speed_m_s=2.78)

    def test_requires_day_of_year_for_numpy(self) -> None:
        with pytest.raises(ValueError, match="day_of_year"):
            pet_penman_monteith(
                np.full(3, 12.3),
                np.full(3, 21.5),
                50.80,
                100.0,
                2.78,
            )

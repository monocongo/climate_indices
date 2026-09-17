"""Regression tests for the ERA5 fire-weather demo input preparation."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _prepare_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "prepare_fire_demo_inputs_test", SCRIPTS_DIR / "prepare_fire_demo_inputs.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _six_hourly(values, times, latitude, longitude, name):
    return xr.DataArray(
        values,
        coords={"time": times, "latitude": latitude, "longitude": longitude},
        dims=("time", "latitude", "longitude"),
        name=name,
    )


def test_daily_surface_aggregation_and_midday_timestamps(monkeypatch):
    """The long record sums precipitation and takes the day's extremes."""
    module = _prepare_module(monkeypatch)
    times = pd.date_range("2020-09-01", periods=8, freq="6h")
    latitude, longitude = [30.0], [-100.0]
    temperature = np.array([280.0, 290.0, 300.0, 295.0, 281.0, 291.0, 301.0, 296.0]).reshape(-1, 1, 1)
    precipitation = np.array([0.001, 0.002, 0.003, 0.004, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1, 1)
    wind = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]).reshape(-1, 1, 1)
    daily = module._to_daily_surface(
        xr.Dataset(
            {
                "2m_temperature": _six_hourly(temperature, times, latitude, longitude, "2m_temperature"),
                "total_precipitation_6hr": _six_hourly(precipitation, times, latitude, longitude, "precip"),
                "10m_wind_speed": _six_hourly(wind, times, latitude, longitude, "wind"),
            }
        )
    )

    assert list(daily.time.values) == [np.datetime64("2020-09-01T12:00"), np.datetime64("2020-09-02T12:00")]
    assert float(daily.tmean_c.isel(time=0, latitude=0, longitude=0).squeeze().item()) == pytest.approx(
        (6.85 + 16.85 + 26.85 + 21.85) / 4
    )
    assert float(daily.tmax_c.isel(time=0, latitude=0, longitude=0).squeeze().item()) == pytest.approx(26.85)
    assert float(daily.tmin_c.isel(time=0, latitude=0, longitude=0).squeeze().item()) == pytest.approx(6.85)
    assert float(daily.precip_mm.isel(time=0, latitude=0, longitude=0).squeeze().item()) == pytest.approx(10.0)
    assert float(daily.precip_mm.isel(time=1, latitude=0, longitude=0).squeeze().item()) == pytest.approx(0.0)
    assert float(daily.wind_speed_ms.isel(time=1, latitude=0, longitude=0).squeeze().item()) == pytest.approx(5.0)


def test_relative_humidity_saturates_and_clips(monkeypatch):
    """Saturated air reads 100 percent and dry air reads 0 percent."""
    module = _prepare_module(monkeypatch)
    temperature = xr.DataArray([0.0, 30.0], dims="level", coords={"level": [1000.0, 1000.0]})
    pressure = temperature["level"]
    saturation = 0.622 * 6.112 / (1000.0 - (1.0 - 0.622) * 6.112)
    humidity = xr.DataArray([saturation, 0.0], dims="level", coords={"level": [1000.0, 1000.0]})
    relative = module._relative_humidity(temperature, humidity, pressure)
    assert float(relative.isel(level=0)) == pytest.approx(100.0, abs=0.1)
    assert float(relative.isel(level=1)) == pytest.approx(0.0)


def test_lowest_above_ground_skips_extrapolated_levels(monkeypatch):
    """The surface humidity comes from the lowest level above the terrain."""
    module = _prepare_module(monkeypatch)
    levels = [1000.0, 850.0, 700.0]
    height = xr.DataArray(
        # a sea-level column with all levels above ground, and a high-terrain
        # column whose two deepest levels sit below the surface
        [[[10.0, -2000.0]], [[1500.0, -500.0]], [[3000.0, 900.0]]],
        coords={"level": levels, "latitude": [30.0], "longitude": [-100.0, -105.0]},
        dims=("level", "latitude", "longitude"),
    )
    values = xr.DataArray(
        [[[80.0, 20.0]], [[60.0, 40.0]], [[50.0, 70.0]]],
        coords={"level": levels, "latitude": [30.0], "longitude": [-100.0, -105.0]},
        dims=("level", "latitude", "longitude"),
    )
    surface = module._lowest_above_ground(values, height)
    assert float(surface.isel(latitude=0, longitude=0).squeeze().item()) == pytest.approx(80.0)
    assert float(surface.isel(latitude=0, longitude=1).squeeze().item()) == pytest.approx(70.0)


def test_daily_levels_derive_humidity_and_height(monkeypatch):
    """The level file carries the derived humidity, AGL heights, and surface humidity."""
    module = _prepare_module(monkeypatch)
    times = pd.date_range("2020-09-01", periods=4, freq="6h")
    levels = [1000.0, 850.0]
    coords = {"time": times, "level": levels, "latitude": [30.0], "longitude": [-100.0]}
    dims = ("time", "level", "latitude", "longitude")
    shape = (len(times), len(levels), 1, 1)
    parts = xr.Dataset(
        {
            "temperature": xr.DataArray(np.full(shape, 283.15), coords=coords, dims=dims),
            "specific_humidity": xr.DataArray(np.full(shape, 0.007), coords=coords, dims=dims),
            "wind_speed": xr.DataArray(np.full(shape, 5.0), coords=coords, dims=dims),
            "geopotential": xr.DataArray(np.array([[[[100.0]], [[1500.0]]]] * len(times)), coords=coords, dims=dims),
            "geopotential_at_surface": xr.DataArray(
                np.zeros((1, 1)),
                coords={"latitude": [30.0], "longitude": [-100.0]},
                dims=("latitude", "longitude"),
            ),
        }
    )

    daily = module._to_daily_levels(parts)

    assert list(daily.time.values) == [np.datetime64("2020-09-01T12:00")]
    assert float(daily.temperature_c.isel(level=0, latitude=0, longitude=0).squeeze().item()) == pytest.approx(10.0)
    assert float(daily.wind_speed_ms.isel(level=1, latitude=0, longitude=0).squeeze().item()) == pytest.approx(5.0)
    assert float(daily.height_agl_m.isel(level=0, latitude=0, longitude=0).squeeze().item()) == pytest.approx(
        (100.0 / module.GRAVITY) / 1.0, rel=1e-6
    )
    assert 0.0 <= float(daily.surface_relative_humidity_percent.isel(latitude=0, longitude=0).squeeze().item()) <= 100.0
    # the humid 10 C parcel at 1000 hPa is near saturation
    assert float(daily.relative_humidity_percent.isel(level=0, latitude=0, longitude=0).squeeze().item()) > 80.0

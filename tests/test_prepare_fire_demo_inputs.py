"""Regression tests for the ERA5 fire-weather demo input preparation."""

import hashlib
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


def _value(data: xr.DataArray) -> float:
    """Scalar float from a single-element DataArray."""
    return float(data.squeeze().item())


def _six_hourly(values, times, latitude, longitude, name):
    return xr.DataArray(
        values,
        coords={"time": times, "latitude": latitude, "longitude": longitude},
        dims=("time", "latitude", "longitude"),
        name=name,
    )


def test_daily_surface_aggregation_and_midday_timestamps(monkeypatch):
    """The long record sums accumulation-day precipitation and takes the day's extremes."""
    module = _prepare_module(monkeypatch)
    times = pd.date_range("2020-09-01", periods=8, freq="6h")
    latitude, longitude = [30.0], [-100.0]
    temperature = np.array([280.0, 290.0, 300.0, 295.0, 281.0, 291.0, 301.0, 296.0]).reshape(-1, 1, 1)
    # one accumulation is negative numerical noise: the sum must clip it to zero,
    # and the closed="right" binning sums the accumulations ending 24:00 on the
    # labelled day, i.e. indices 1-4 for 2020-09-01
    precipitation = np.array([0.0, 0.001, 0.002, -0.0005, 0.004, 0.0, 0.0, 0.0]).reshape(-1, 1, 1)
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
    assert _value(daily.tmean_c.isel(time=0, latitude=0, longitude=0)) == pytest.approx(
        (6.85 + 16.85 + 26.85 + 21.85) / 4
    )
    assert _value(daily.tmax_c.isel(time=0, latitude=0, longitude=0)) == pytest.approx(26.85)
    assert _value(daily.tmin_c.isel(time=0, latitude=0, longitude=0)) == pytest.approx(6.85)
    assert _value(daily.precip_mm.isel(time=0, latitude=0, longitude=0)) == pytest.approx(7.0)
    assert _value(daily.precip_mm.isel(time=1, latitude=0, longitude=0)) == pytest.approx(0.0)
    assert _value(daily.wind_speed_ms.isel(time=0, latitude=0, longitude=0)) == pytest.approx(2.5)
    assert _value(daily.wind_speed_ms.isel(time=1, latitude=0, longitude=0)) == pytest.approx(
        (2.0 + 4.0 + 6.0 + 8.0) / 4
    )


def test_relative_humidity_matches_saturation_and_clips(monkeypatch):
    """Saturated air reads 100 percent, half saturation reads 50, and overshoot clips."""
    module = _prepare_module(monkeypatch)
    temperature = xr.DataArray([0.0, 30.0, 0.0], dims="level", coords={"level": [1000.0, 1000.0, 1000.0]})
    pressure = temperature["level"]
    saturation = 0.622 * 6.112 / (1000.0 - (1.0 - 0.622) * 6.112)
    humidity = xr.DataArray([saturation, 0.0, saturation * 1.5], dims="level", coords=temperature.coords)
    relative = module._relative_humidity(temperature, humidity, pressure)
    assert _value(relative.isel(level=0)) == pytest.approx(100.0, abs=0.1)
    assert _value(relative.isel(level=1)) == pytest.approx(0.0)
    assert _value(relative.isel(level=2)) == pytest.approx(100.0)

    # an interior point is not clipped: it fails if the kPa-to-hPa or the
    # mixing-ratio denominator drifts
    half = xr.DataArray([saturation / 2.0, 0.0, 0.0], dims="level", coords=temperature.coords)
    assert _value(module._relative_humidity(temperature, half, pressure).isel(level=0)) == pytest.approx(50.0, abs=0.1)


def test_lowest_above_ground_skips_extrapolated_levels(monkeypatch):
    """The surface humidity comes from the lowest level above the terrain."""
    module = _prepare_module(monkeypatch)
    levels = [1000.0, 850.0, 700.0]
    coords = {"level": levels, "latitude": [30.0], "longitude": [-100.0, -105.0]}
    height = xr.DataArray(
        # a sea-level column with all levels above ground, and a high-terrain
        # column whose two deepest levels sit below the surface
        [[[10.0, -2000.0]], [[1500.0, -500.0]], [[3000.0, 900.0]]],
        coords=coords,
        dims=("level", "latitude", "longitude"),
    )
    values = xr.DataArray(
        [[[80.0, 20.0]], [[60.0, 40.0]], [[50.0, 70.0]]],
        coords=coords,
        dims=("level", "latitude", "longitude"),
    )
    surface = module._lowest_above_ground(values, height)
    assert _value(surface.isel(latitude=0, longitude=0)) == pytest.approx(80.0)
    assert _value(surface.isel(latitude=0, longitude=1)) == pytest.approx(70.0)


def test_daily_levels_derive_humidity_and_height(monkeypatch):
    """The level file carries the derived humidity, AGL heights, and surface humidity."""
    module = _prepare_module(monkeypatch)
    times = pd.date_range("2020-09-01", periods=4, freq="6h")
    levels = [1000.0, 850.0]
    coords = {"time": times, "level": levels, "latitude": [30.0], "longitude": [-100.0]}
    dims = ("time", "level", "latitude", "longitude")
    shape = (len(times), len(levels), 1, 1)
    surface_geopotential = 98.0665  # 10 m of geopotential height
    parts = xr.Dataset(
        {
            "temperature": xr.DataArray(np.full(shape, 283.15), coords=coords, dims=dims),
            "specific_humidity": xr.DataArray(np.full(shape, 0.007), coords=coords, dims=dims),
            "wind_speed": xr.DataArray(np.full(shape, 5.0), coords=coords, dims=dims),
            "geopotential": xr.DataArray(np.array([[[[100.0]], [[1500.0]]]] * len(times)), coords=coords, dims=dims),
            # the static field arrives without a time dimension
            "geopotential_at_surface": xr.DataArray(
                np.full((1, 1), surface_geopotential),
                coords={"latitude": [30.0], "longitude": [-100.0]},
                dims=("latitude", "longitude"),
            ),
        }
    )

    daily = module._to_daily_levels(parts)

    assert list(daily.time.values) == [np.datetime64("2020-09-01T12:00")]
    assert _value(daily.temperature_c.isel(level=0, latitude=0, longitude=0)) == pytest.approx(10.0)
    assert _value(daily.wind_speed_ms.isel(level=1, latitude=0, longitude=0)) == pytest.approx(5.0)
    assert _value(daily.height_agl_m.isel(level=0, latitude=0, longitude=0)) == pytest.approx(
        (100.0 - surface_geopotential) / module.GRAVITY
    )
    # the humid 10 C parcel at 1000 hPa is near saturation, and the selected
    # surface humidity is the 1000 hPa value (that level is above ground here)
    level_zero_humidity = _value(daily.relative_humidity_percent.isel(level=0, latitude=0, longitude=0))
    assert level_zero_humidity == pytest.approx(91.2, abs=0.5)
    assert _value(daily.surface_relative_humidity_percent.isel(latitude=0, longitude=0)) == pytest.approx(
        level_zero_humidity
    )


def test_daily_levels_accept_static_field_with_time_axis(monkeypatch):
    """A static field that carries a time axis is reduced to its first slice."""
    module = _prepare_module(monkeypatch)
    times = pd.date_range("2020-09-01", periods=2, freq="6h")
    coords = {"time": times, "level": [1000.0], "latitude": [30.0], "longitude": [-100.0]}
    shape = (len(times), 1, 1, 1)
    static_coords = {"time": [times[0], times[-1]], "latitude": [30.0], "longitude": [-100.0]}
    parts = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.full(shape, 283.15), coords=coords, dims=("time", "level", "latitude", "longitude")
            ),
            "specific_humidity": xr.DataArray(
                np.full(shape, 0.007), coords=coords, dims=("time", "level", "latitude", "longitude")
            ),
            "wind_speed": xr.DataArray(
                np.full(shape, 5.0), coords=coords, dims=("time", "level", "latitude", "longitude")
            ),
            "geopotential": xr.DataArray(
                np.full(shape, 100.0), coords=coords, dims=("time", "level", "latitude", "longitude")
            ),
            "geopotential_at_surface": xr.DataArray(
                np.array([[[98.0665]], [[0.0]]]),
                coords=static_coords,
                dims=("time", "latitude", "longitude"),
            ),
        }
    )

    daily = module._to_daily_levels(parts)

    assert _value(daily.height_agl_m.isel(level=0, latitude=0, longitude=0)) == pytest.approx(
        (100.0 - 98.0665) / module.GRAVITY
    )


def test_select_converts_longitude_and_keeps_static_fields_dimensionless(monkeypatch):
    """The 0-360 store longitude becomes -180-180 and static fields skip the time slice."""
    module = _prepare_module(monkeypatch)
    times = pd.date_range("2020-01-01", periods=4, freq="D")
    longitudes = np.array([235.5, 237.0, 270.0, 294.0])
    latitudes = np.array([25.5, 30.0, 49.5])
    values = np.arange(len(times) * len(latitudes) * len(longitudes), dtype=float).reshape(
        len(times), len(latitudes), len(longitudes)
    )
    dataset = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), values)},
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )
    selected = module._select(dataset, "temperature", "2020-01-02", "2020-01-03")
    assert list(selected.longitude.values) == [-124.5, -123.0, -90.0, -66.0]
    assert selected.sizes["time"] == 2

    static = xr.Dataset(
        {"geopotential_at_surface": (("latitude", "longitude"), np.zeros((3, 4)))},
        coords={"latitude": latitudes, "longitude": longitudes},
    )
    selected_static = module._select(static, "geopotential_at_surface", "2020-01-01", "2020-01-31")
    assert "time" not in selected_static.dims


def test_cache_builds_once_and_never_publishes_partial_files(tmp_path, monkeypatch):
    """A warm cache skips the builder and a failed build leaves nothing behind."""
    module = _prepare_module(monkeypatch)
    cache_dir = tmp_path / "cache"
    calls = []

    def builder():
        calls.append(1)
        return xr.DataArray([1.0, 2.0], dims="level", coords={"level": [1000.0, 850.0]}, name="temperature")

    first = module._cache(cache_dir, "temperature_2020", "temperature", builder)
    second = module._cache(cache_dir, "temperature_2020", "temperature", builder)
    assert len(calls) == 1
    xr.testing.assert_equal(first, second)
    assert [path.name for path in cache_dir.glob("*.nc")] == [f"{module._cache_fingerprint()}-temperature_2020.nc"]

    def failing():
        raise RuntimeError("source unavailable")

    with pytest.raises(RuntimeError, match="source unavailable"):
        module._cache(cache_dir, "2m_temperature_2020", "2m_temperature", failing)
    assert [path.name for path in cache_dir.glob("*.nc")] == [f"{module._cache_fingerprint()}-temperature_2020.nc"]
    assert not [path for path in cache_dir.iterdir() if path.is_dir()]


def test_publish_failure_preserves_previous_output(tmp_path, monkeypatch):
    """Only a fully written dataset may replace the published one."""
    module = _prepare_module(monkeypatch)
    path = tmp_path / "surface.nc"
    times = pd.date_range("2020-01-01", periods=2, freq="D") + pd.Timedelta(hours=12)
    dataset = xr.Dataset(
        {"tmean_c": (("time", "latitude", "longitude"), np.ones((2, 1, 1)))},
        coords={"time": times, "latitude": [30.0], "longitude": [-100.0]},
    )
    module._publish(dataset, path)
    published = path.read_bytes()

    class Exploding:
        def to_netcdf(self, target):
            raise RuntimeError("disk full")

    monkeypatch.setattr(module, "_add_units", lambda prepared: Exploding())
    with pytest.raises(RuntimeError, match="disk full"):
        module._publish(dataset, path)

    assert path.read_bytes() == published
    assert not list(tmp_path.glob(".surface.nc.*"))


def test_prepare_inputs_publishes_manifest_with_provenance(tmp_path, monkeypatch):
    """The manifest records the source, the requested and realized domains, and checksums."""
    module = _prepare_module(monkeypatch)
    times = pd.date_range("2020-01-01", periods=2, freq="D") + pd.Timedelta(hours=12)
    coordinates = {"time": times, "latitude": [30.0], "longitude": [-100.0]}
    surface = xr.Dataset(
        {"tmean_c": (("time", "latitude", "longitude"), np.ones((2, 1, 1)))},
        coords=coordinates,
    )
    levels = xr.Dataset(
        {"temperature_c": (("time", "level", "latitude", "longitude"), np.ones((2, 1, 1, 1)))},
        coords={**coordinates, "level": [1000.0]},
    )

    class Source:
        def close(self):
            pass

    monkeypatch.setattr(module, "_open_source", Source)
    monkeypatch.setattr(module, "_surface_inputs", lambda dataset, cache_dir: surface)
    monkeypatch.setattr(module, "_level_inputs", lambda dataset, cache_dir: levels)

    manifest = module.prepare_inputs(tmp_path)

    surface_path = tmp_path / module.OUTPUT_SURFACE
    assert surface_path.exists()
    assert (
        manifest["artifacts"][module.OUTPUT_SURFACE]["sha256"] == hashlib.sha256(surface_path.read_bytes()).hexdigest()
    )
    assert manifest["source_url"] == module.SOURCE_URL
    assert manifest["domain"]["requested_longitude"] == [-125.0, -65.0]
    assert manifest["domain"]["realized_longitude"] == [-100.0, -100.0]
    assert manifest["season"] == [module.SEASON_START, module.SEASON_END]
    assert manifest["generated_utc"]

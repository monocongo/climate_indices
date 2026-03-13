from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def bench_monthly_precip_np() -> np.ndarray:
    """40-year monthly precipitation as 2D numpy array (40, 12) for NumPy SPI path."""
    rng = np.random.default_rng(8888)
    return rng.gamma(shape=2.0, scale=50.0, size=(40, 12))


@pytest.fixture(scope="session")
def bench_monthly_precip_da() -> xr.DataArray:
    """40-year monthly precipitation as 1D xarray DataArray for xarray SPI path."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(8888)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))
    return xr.DataArray(values, coords={"time": time}, dims=["time"], attrs={"units": "mm"})


@pytest.fixture(scope="session")
def bench_monthly_pet_np() -> np.ndarray:
    """40-year monthly PET as 2D numpy array (40, 12) for NumPy SPEI path."""
    rng = np.random.default_rng(9999)
    return rng.gamma(shape=2.5, scale=60.0, size=(40, 12))


@pytest.fixture(scope="session")
def bench_monthly_pet_da() -> xr.DataArray:
    """40-year monthly PET as 1D xarray DataArray for xarray SPEI path."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(9999)
    values = rng.gamma(shape=2.5, scale=60.0, size=len(time))
    return xr.DataArray(values, coords={"time": time}, dims=["time"], attrs={"units": "mm"})


@pytest.fixture(scope="session")
def bench_monthly_temp_np() -> np.ndarray:
    """40-year monthly temperatures as 1D numpy array for NumPy PET Thornthwaite."""
    rng = np.random.default_rng(7777)
    return 15.0 + 10.0 * np.sin(np.linspace(0, 80 * np.pi, 480)) + rng.normal(0, 2, 480)


@pytest.fixture(scope="session")
def bench_monthly_temp_da() -> xr.DataArray:
    """40-year monthly temperatures as xarray DataArray for xarray PET Thornthwaite."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(7777)
    values = 15.0 + 10.0 * np.sin(np.linspace(0, 80 * np.pi, 480)) + rng.normal(0, 2, 480)
    return xr.DataArray(values, coords={"time": time}, dims=["time"], attrs={"units": "degC"})


@pytest.fixture(scope="session")
def bench_daily_tmin_np() -> np.ndarray:
    """5-year daily min temperature as 1D numpy for NumPy Hargreaves."""
    rng = np.random.default_rng(6666)
    days = 5 * 366
    day_of_year = np.tile(np.arange(1, 367), 5)[:days]
    return 10.0 + 5.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365) + rng.normal(0, 1, days)


@pytest.fixture(scope="session")
def bench_daily_tmax_np() -> np.ndarray:
    """5-year daily max temperature as 1D numpy for NumPy Hargreaves."""
    rng = np.random.default_rng(5555)
    days = 5 * 366
    day_of_year = np.tile(np.arange(1, 367), 5)[:days]
    return 22.5 + 7.5 * np.sin(2 * np.pi * (day_of_year - 105) / 365) + rng.normal(0, 1, days)


@pytest.fixture(scope="session")
def bench_daily_tmin_da(bench_daily_tmin_np: np.ndarray) -> xr.DataArray:
    """5-year daily min temp as xarray DataArray for xarray Hargreaves."""
    time = pd.date_range("2015-01-01", periods=len(bench_daily_tmin_np), freq="D")
    return xr.DataArray(bench_daily_tmin_np.copy(), coords={"time": time}, dims=["time"])


@pytest.fixture(scope="session")
def bench_daily_tmax_da(bench_daily_tmax_np: np.ndarray) -> xr.DataArray:
    """5-year daily max temp as xarray DataArray for xarray Hargreaves."""
    time = pd.date_range("2015-01-01", periods=len(bench_daily_tmax_np), freq="D")
    return xr.DataArray(bench_daily_tmax_np.copy(), coords={"time": time}, dims=["time"])


@pytest.fixture(scope="session")
def bench_gridded_precip_da() -> xr.DataArray:
    """Medium-grid monthly precip DataArray (480 time, 20 lat, 20 lon) for gridded benchmarks."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    lat = np.linspace(25.0, 50.0, 20)
    lon = np.linspace(-125.0, -70.0, 20)
    rng = np.random.default_rng(4444)
    values = rng.gamma(shape=2.0, scale=50.0, size=(len(time), len(lat), len(lon)))
    return xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        attrs={"units": "mm"},
    )

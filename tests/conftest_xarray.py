from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def make_dataarray():
    """Factory fixture for creating xarray DataArrays with sensible defaults.

    This factory reduces duplication by providing a configurable DataArray
    constructor. Use it when you need a test-specific DataArray instead of
    one of the pre-defined fixtures.

    Returns:
        Callable that creates DataArrays with customizable parameters

    Example:
        def test_something(make_dataarray):
            da = make_dataarray(freq="MS", periods=480, dims=1, seed=42)
            assert len(da.time) == 480
    """

    def _make(
        *,
        freq: str = "MS",
        periods: int = 480,
        dims: int = 1,
        seed: int = 42,
        fill_value: float | None = None,
        name: str = "test_data",
        start_date: str = "1980-01-01",
    ) -> xr.DataArray:
        """Create a DataArray with specified parameters.

        Args:
            freq: Pandas frequency string ("MS" for monthly, "D" for daily)
            periods: Number of time steps
            dims: Number of dimensions (1, 2, or 3)
            seed: Random seed for reproducibility
            fill_value: If provided, use constant values instead of random
            name: DataArray name
            start_date: Start date string

        Returns:
            xarray DataArray
        """
        rng = np.random.default_rng(seed)
        time = pd.date_range(start_date, periods=periods, freq=freq)

        if dims == 1:
            if fill_value is not None:
                values = np.full(periods, fill_value)
            else:
                values = rng.gamma(shape=2.0, scale=50.0, size=periods)
            return xr.DataArray(
                values,
                coords={"time": time},
                dims=["time"],
                attrs={"units": "mm"},
                name=name,
            )
        elif dims == 2:
            lat = np.linspace(30.0, 50.0, 5)
            if fill_value is not None:
                values = np.full((periods, len(lat)), fill_value)
            else:
                values = rng.gamma(shape=2.0, scale=50.0, size=(periods, len(lat)))
            return xr.DataArray(
                values,
                coords={"time": time, "lat": lat},
                dims=["time", "lat"],
                attrs={"units": "mm"},
                name=name,
            )
        elif dims == 3:
            lat = np.linspace(30.0, 50.0, 5)
            lon = np.linspace(-120.0, -100.0, 6)
            if fill_value is not None:
                values = np.full((periods, len(lat), len(lon)), fill_value)
            else:
                values = rng.gamma(shape=2.0, scale=50.0, size=(periods, len(lat), len(lon)))
            return xr.DataArray(
                values,
                coords={"time": time, "lat": lat, "lon": lon},
                dims=["time", "lat", "lon"],
                attrs={"units": "mm"},
                name=name,
            )
        else:
            raise ValueError(f"dims must be 1, 2, or 3, got {dims}")

    return _make


@pytest.fixture(scope="session")
def sample_monthly_precip_da() -> xr.DataArray:
    """Create a 1D monthly precipitation DataArray (40 years, 1980-2019)."""
    # 40 years * 12 months = 480 values
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    # generate random precipitation values
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
        },
    )


@pytest.fixture(scope="session")
def sample_daily_precip_da() -> xr.DataArray:
    """Create a 1D daily precipitation DataArray (5 years, 2015-2019)."""
    # 5 years of daily data
    time = pd.date_range("2015-01-01", "2019-12-31", freq="D")
    # generate random precipitation values
    rng = np.random.default_rng(123)
    values = rng.gamma(shape=2.0, scale=5.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Daily Precipitation",
        },
    )


@pytest.fixture(scope="session")
def sample_monthly_pet_da() -> xr.DataArray:
    """Create a 1D monthly PET DataArray matching precip fixture (40 years, 1980-2019)."""
    # 40 years * 12 months = 480 values, matching sample_monthly_precip_da
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    # generate random PET values (typically higher than precip)
    rng = np.random.default_rng(100)
    values = rng.gamma(shape=2.5, scale=60.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Potential Evapotranspiration",
        },
    )


@pytest.fixture(scope="session")
def sample_monthly_pet_offset_da() -> xr.DataArray:
    """
    1D monthly PET with time offset from standard precip (1985-2024).
    Used for testing coordinate alignment warnings.
    40 years, 480 months, realistic PET variation.
    """
    rng = np.random.default_rng(101)
    time = pd.date_range("1985-01-01", periods=480, freq="MS")
    data = 50.0 + rng.gamma(shape=3.0, scale=20.0, size=480)
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="pet",
        attrs={
            "units": "mm",
            "long_name": "Potential Evapotranspiration",
            "standard_name": "water_potential_evaporation_amount",
        },
    )


@pytest.fixture(scope="session")
def dask_monthly_precip_1d() -> xr.DataArray:
    """Create 1-D Dask-backed monthly precipitation DataArray (40 years, 1980-2019).

    Time dimension is a single chunk (required for SPI/SPEI).
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    da = xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation (Dask)",
        },
        name="precip_dask",
    )
    # chunk time as single chunk (required for climate indices)
    return da.chunk({"time": -1})


@pytest.fixture(scope="session")
def dask_monthly_precip_3d() -> xr.DataArray:
    """Create 3-D Dask-backed monthly precipitation DataArray (40 years, 5 lat, 6 lon).

    Time dimension is a single chunk, spatial dimensions are chunked.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    lat = np.linspace(30.0, 50.0, 5)
    lon = np.linspace(-120.0, -100.0, 6)
    rng = np.random.default_rng(99)
    values = rng.gamma(shape=2.0, scale=50.0, size=(len(time), len(lat), len(lon)))

    da = xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        attrs={
            "units": "mm",
            "long_name": "Gridded Precipitation (Dask)",
        },
        name="precip_grid_dask",
    )
    # chunk: single time chunk, spatial chunks
    return da.chunk({"time": -1, "lat": 2, "lon": 3})


@pytest.fixture(scope="session")
def gridded_monthly_precip_3d() -> xr.DataArray:
    """Create 3D gridded monthly precipitation DataArray (40 years, 5 lat, 6 lon).

    Non-Dask eager computation version for testing gridded processing.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    lat = np.linspace(30.0, 50.0, 5)
    lon = np.linspace(-120.0, -100.0, 6)
    rng = np.random.default_rng(99)
    values = rng.gamma(shape=2.0, scale=50.0, size=(len(time), len(lat), len(lon)))

    return xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        attrs={
            "units": "mm",
            "long_name": "Gridded Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def single_point_monthly_da() -> xr.DataArray:
    """Create 1D time-only monthly DataArray (40 years).

    Simple time series for single-point analysis.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Single Point Precipitation",
        },
    )


@pytest.fixture(scope="session")
def minimum_calibration_da() -> xr.DataArray:
    """Create DataArray with exactly 30 years (360 months) for boundary testing.

    Tests minimum calibration period requirement.
    """
    time = pd.date_range("1990-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Minimum Calibration Period Precipitation",
        },
    )


@pytest.fixture(scope="session")
def zero_inflated_precip_da() -> xr.DataArray:
    """Create precipitation DataArray with ~50% zeros (arid region pattern).

    Simulates arid/semi-arid climate with frequent zero-precipitation months.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    # generate base values
    values = rng.gamma(shape=2.0, scale=30.0, size=len(time))
    # randomly set ~50% to zero
    zero_mask = rng.random(len(time)) < 0.5
    values[zero_mask] = 0.0

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Zero-Inflated Precipitation (Arid Region)",
        },
    )


@pytest.fixture(scope="session")
def leading_nan_block_da() -> xr.DataArray:
    """Create DataArray with first 12 months all NaN.

    Tests handling of leading missing data.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))
    # set first year to NaN
    values[:12] = np.nan

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Precipitation with Leading NaN Block",
        },
    )


@pytest.fixture(scope="session")
def trailing_nan_block_da() -> xr.DataArray:
    """Create DataArray with last 12 months all NaN.

    Tests handling of trailing missing data.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))
    # set last year to NaN
    values[-12:] = np.nan

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Precipitation with Trailing NaN Block",
        },
    )


@pytest.fixture(scope="session")
def block_nan_pattern_da() -> xr.DataArray:
    """Create DataArray with consecutive year of NaN in middle of time series.

    Tests handling of contiguous missing data blocks.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))
    # set year 20 (months 240-251) to NaN
    values[240:252] = np.nan

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Precipitation with Mid-Series NaN Block",
        },
    )


@pytest.fixture(scope="session")
def sample_monthly_temp_da() -> xr.DataArray:
    """
    Standard 1D monthly temperature DataArray for testing.
    40 years (1980-2019), 480 months, sinusoidal seasonal pattern + noise.
    """
    rng = np.random.default_rng(300)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    # sinusoidal seasonal pattern (warm in summer, cold in winter)
    month_numbers = np.array([t.month for t in time])
    seasonal_pattern = 15.0 + 10.0 * np.sin(2 * np.pi * (month_numbers - 3) / 12)
    # add random noise
    noise = rng.normal(0, 2.0, 480)
    data = seasonal_pattern + noise
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="temperature",
        attrs={
            "units": "degC",
            "long_name": "Monthly Temperature",
            "standard_name": "air_temperature",
        },
    )


@pytest.fixture(scope="session")
def gridded_monthly_temp_3d() -> xr.DataArray:
    """
    Eager 3D monthly temperature (time, lat, lon).
    40 years, 5 lat points, 6 lon points. Latitude-dependent offset.
    """
    rng = np.random.default_rng(310)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    lat = np.linspace(-10.0, 10.0, 5)
    lon = np.linspace(-15.0, 15.0, 6)
    # create base seasonal pattern
    month_numbers = np.array([t.month for t in time])
    seasonal_pattern = 15.0 + 10.0 * np.sin(2 * np.pi * (month_numbers - 3) / 12)
    # broadcast and add latitude-dependent offset (warmer at equator)
    data = np.zeros((480, 5, 6))
    for i, latitude in enumerate(lat):
        lat_offset = 5.0 * (1.0 - abs(latitude) / 10.0)  # warmer near equator
        data[:, i, :] = seasonal_pattern[:, np.newaxis] + lat_offset
    # add random noise
    data += rng.normal(0, 2.0, size=(480, 5, 6))
    return xr.DataArray(
        data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        name="temperature",
        attrs={
            "units": "degC",
            "long_name": "Monthly Temperature",
            "standard_name": "air_temperature",
        },
    )


@pytest.fixture(scope="session")
def dask_monthly_temp_3d(gridded_monthly_temp_3d: xr.DataArray) -> xr.DataArray:
    """
    Dask-backed 3D monthly temperature (time, lat, lon).
    Chunked along time dimension (full chunk = -1).
    """
    return gridded_monthly_temp_3d.chunk({"time": -1, "lat": 5, "lon": 6})


@pytest.fixture(scope="session")
def coord_rich_1d_da() -> xr.DataArray:
    """
    1D DataArray with rich coordinate metadata.

    Features:
    - time dimension coord with multiple attrs (axis, calendar, long_name, bounds, standard_name)
    - month non-dimension auxiliary coord
    - station_id scalar coord with attrs
    """
    time = pd.date_range("2020-01-01", "2022-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    month = xr.DataArray(
        [t.month for t in time],
        dims=["time"],
        attrs={"long_name": "month of year", "units": "1"},
    )

    station_id = xr.DataArray(
        "GHCN-12345",
        attrs={"long_name": "station identifier", "cf_role": "timeseries_id"},
    )

    da = xr.DataArray(
        values,
        coords={
            "time": time,
            "month": month,
            "station_id": station_id,
        },
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
        },
        name="precip",
    )

    da.coords["time"].attrs = {
        "axis": "T",
        "calendar": "standard",
        "long_name": "time",
        "bounds": "time_bounds",
        "standard_name": "time",
    }

    return da


@pytest.fixture(scope="session")
def multi_coord_da() -> xr.DataArray:
    """
    2D DataArray with multiple dimension coords and rich metadata.

    Features:
    - time and lat dimension coords, each with rich attrs
    - month auxiliary coord (time-dependent)
    - ensemble scalar coord with attrs
    """
    time = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    lat = np.array([30.0, 35.0, 40.0, 45.0])
    rng = np.random.default_rng(99)
    values = rng.gamma(shape=2.0, scale=50.0, size=(len(time), len(lat)))

    month = xr.DataArray(
        [t.month for t in time],
        dims=["time"],
        attrs={"long_name": "month of year"},
    )

    ensemble = xr.DataArray(
        "ens01",
        attrs={"long_name": "ensemble member", "type": "hindcast"},
    )

    da = xr.DataArray(
        values,
        coords={
            "time": time,
            "lat": lat,
            "month": month,
            "ensemble": ensemble,
        },
        dims=["time", "lat"],
        attrs={
            "units": "mm",
            "long_name": "Gridded Precipitation",
        },
        name="precip_grid",
    )

    da.coords["time"].attrs = {
        "axis": "T",
        "standard_name": "time",
        "long_name": "time",
    }
    da.coords["lat"].attrs = {
        "axis": "Y",
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
    }

    return da


@pytest.fixture(scope="session")
def no_time_dim_da() -> xr.DataArray:
    """
    DataArray without a time dimension (spatial coordinates only).
    Used for testing validation that requires time dimension.
    """
    x = np.arange(10)
    lat = np.arange(5) * 10.0
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=(len(x), len(lat)))

    return xr.DataArray(
        values,
        coords={"x": x, "lat": lat},
        dims=["x", "lat"],
        attrs={"units": "mm"},
    )


@pytest.fixture(scope="session")
def non_monotonic_time_da() -> xr.DataArray:
    """
    DataArray with non-monotonic (shuffled) time coordinate.
    Used for testing time monotonicity validation.
    """
    time = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    shuffled_time = time[[0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10]]
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": shuffled_time},
        dims=["time"],
        attrs={"units": "mm"},
    )


@pytest.fixture(scope="session")
def short_monthly_da() -> xr.DataArray:
    """
    Short DataArray with only 3 months (below minimum calibration period).
    Used for testing insufficient data validation.
    """
    time = pd.date_range("2020-01-01", "2020-03-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={"units": "mm"},
    )


@pytest.fixture(scope="session")
def dask_multi_time_chunk() -> xr.DataArray:
    """
    3D Dask-backed DataArray with time split into multiple chunks (invalid).
    This fixture intentionally violates the chunking constraint to test validation.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    lat = np.linspace(30.0, 50.0, 5)
    lon = np.linspace(-120.0, -100.0, 6)
    rng = np.random.default_rng(99)
    values = rng.gamma(shape=2.0, scale=50.0, size=(len(time), len(lat), len(lon)))

    da = xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        attrs={"units": "mm"},
    )
    return da.chunk({"time": 120, "lat": 2, "lon": 3})


@pytest.fixture(scope="session")
def dask_monthly_pet_da() -> xr.DataArray:
    """
    1D Dask-backed monthly PET DataArray matching dask_monthly_precip_1d.
    For testing multi-input functions like SPEI with Dask arrays.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(100)
    values = rng.gamma(shape=2.5, scale=60.0, size=len(time))

    da = xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly PET (Dask)",
        },
        name="pet_dask",
    )
    return da.chunk({"time": -1})


@pytest.fixture(scope="session")
def monthly_precip_with_nan() -> xr.DataArray:
    """
    40-year monthly precipitation with ~10% NaN scattered throughout.
    Used for testing NaN handling in climate index calculations.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    nan_indices = rng.choice(len(time), size=int(len(time) * 0.1), replace=False)
    values[nan_indices] = np.nan

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation with NaN",
        },
    )


@pytest.fixture(scope="session")
def monthly_precip_heavy_nan() -> xr.DataArray:
    """
    40-year monthly precipitation with ~80% NaN (insufficient data).
    Used for testing validation that rejects insufficient non-NaN data.
    """
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(123)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    nan_indices = rng.choice(len(time), size=int(len(time) * 0.8), replace=False)
    values[nan_indices] = np.nan

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation with Heavy NaN",
        },
    )

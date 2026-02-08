import json
import os

import numpy as np
import pytest

# constants
# start and end year of the monthly precipitation, temperature, and PET datasets
_DATA_YEAR_START_MONTHLY = 1895
_DATA_YEAR_END_MONTHLY = 2017

# start and end year of the calibration periods
# used by SPI/SPEI Pearson calculations
_CALIBRATION_YEAR_START_MONTHLY = 1981
_CALIBRATION_YEAR_END_MONTHLY = 2010
_CALIBRATION_YEAR_START_PALMER = 1931
_CALIBRATION_YEAR_END_PALMER = 1990

# start and end years relevant to the daily datasets
_DATA_YEAR_START_DAILY = 1998
_CALIBRATION_YEAR_START_DAILY = 1998
_CALIBRATION_YEAR_END_DAILY = 2016

# latitude value used for computing the fixture datasets (PET, Palmers)
_LATITUDE_DEGREES = 25.2292

# available water capacity value used for computing the fixture datasets (Palmers)
_AWC_INCHES = 4.5


@pytest.fixture(scope="module")
def data_year_start_monthly():
    return _DATA_YEAR_START_MONTHLY


@pytest.fixture(scope="module")
def data_year_end_monthly():
    return _DATA_YEAR_END_MONTHLY


@pytest.fixture(scope="module")
def data_year_start_daily():
    return _DATA_YEAR_START_DAILY


@pytest.fixture(scope="module")
def data_year_start_palmer():
    return _DATA_YEAR_START_MONTHLY


@pytest.fixture(scope="module")
def calibration_year_start_monthly():
    return _CALIBRATION_YEAR_START_MONTHLY


@pytest.fixture(scope="module")
def calibration_year_end_monthly():
    return _CALIBRATION_YEAR_END_MONTHLY


@pytest.fixture(scope="module")
def calibration_year_start_daily():
    return _CALIBRATION_YEAR_START_DAILY


@pytest.fixture(scope="module")
def calibration_year_start_palmer():
    return _CALIBRATION_YEAR_START_PALMER


@pytest.fixture(scope="module")
def calibration_year_end_palmer():
    return _CALIBRATION_YEAR_END_PALMER


@pytest.fixture(scope="module")
def calibration_year_end_daily():
    return _CALIBRATION_YEAR_END_DAILY


@pytest.fixture(scope="module")
def latitude_degrees():
    return _LATITUDE_DEGREES


@pytest.fixture(scope="module")
def precips_mm_monthly():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "precips_mm_monthly.npy"))


@pytest.fixture(scope="module")
def precips_mm_daily():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "precips_mm_daily.npy"))


@pytest.fixture(scope="module")
def transformed_pearson3():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "pearson3_monthly.npy"))


@pytest.fixture(scope="module")
def transformed_pearson3_monthly_fullperiod():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "pearson3_monthly_full.npy"))


@pytest.fixture(scope="module")
def gamma_monthly():
    """
    Array of values corresponding to the gamma fitting parameters computed
    from the precips_mm_monthly array using an application of the gamma
    conversion/fitting algorithm provided in compute.py.

    :return:
    """
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "gamma_monthly.npy"))


@pytest.fixture(scope="module")
def gamma_daily():
    """
    Array of values corresponding to the gamma fitting parameters computed
    from the precips_mm_daily array using an application of the gamma
    conversion/fitting algorithm provided in compute.py.

    :return:
    """

    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "gamma_daily.npy"))


@pytest.fixture(scope="module")
def transformed_gamma_monthly():
    """
    Array of values corresponding to the precips_mm_monthly array, i.e. those
    values have been transformed/fitted using an application of the gamma
    conversion/fitting algorithm provided in compute.py.

    :return:
    """
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "transformed_gamma_monthly.npy"))


@pytest.fixture(scope="module")
def transformed_gamma_daily():
    """
    Array of values corresponding to the precips_mm_daily array, i.e. those
    values have been transformed/fitted using an application of the gamma
    conversion/fitting algorithm provided in compute.py.

    :return:
    """

    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "transformed_gamma_daily.npy"))


@pytest.fixture(scope="module")
def pet_thornthwaite_mm():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "pet_thornthwaite.npy"))


@pytest.fixture(scope="module")
def temps_celsius():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "temp_celsius.npy"))


@pytest.fixture(scope="module")
def pnp_6month():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "pnp_06.npy"))


@pytest.fixture(scope="module")
def spei_6_month_gamma():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "spei_06_gamma.npy"))


@pytest.fixture(scope="module")
def spei_6_month_pearson3():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "spei_06_pearson3.npy"))


@pytest.fixture(scope="module")
def spi_1_month_gamma():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "spi_01_gamma.npy"))


@pytest.fixture(scope="module")
def spi_6_month_gamma():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "spi_06_gamma.npy"))


@pytest.fixture(scope="module")
def spi_6_month_pearson3():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "spi_06_pearson3.npy"))


@pytest.fixture(scope="function")
def rain_mm() -> np.ndarray:
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "rain_mm.npy"))


@pytest.fixture(scope="function")
def rain_mm_365() -> np.ndarray:
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "rain_mm_365.npy"))


@pytest.fixture(scope="module")
def rain_mm_366():
    return np.load(os.path.join(os.path.split(__file__)[0], "fixture", "rain_mm_366.npy"))


@pytest.fixture(scope="module")
def palmer_awcs():
    with open(os.path.join(os.path.split(__file__)[0], "fixture", "palmer_awc.json")) as awcfile:
        return json.load(awcfile)


# Hargreaves fixtures for daily evapotranspiration calculations
# Start and end years for daily temperature data used in Hargreaves tests
_HARGREAVES_DATA_YEAR_START = 2000
_HARGREAVES_NUM_YEARS = 2


@pytest.fixture(scope="module")
def hargreaves_data_year_start():
    return _HARGREAVES_DATA_YEAR_START


@pytest.fixture(scope="module")
def hargreaves_daily_tmin_celsius():
    """
    Daily minimum temperature fixture for Hargreaves tests.
    Generates synthetic seasonal temperature data (2 years x 366 days).
    Temperature varies sinusoidally between 5 and 15 degrees C.
    """
    num_days = _HARGREAVES_NUM_YEARS * 366
    # create seasonal variation: cooler in winter, warmer in summer
    day_of_year = np.tile(np.arange(1, 367), _HARGREAVES_NUM_YEARS)
    # sinusoidal pattern: min around day 15 (Jan), max around day 196 (Jul)
    tmin = 10.0 + 5.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    return tmin[:num_days]


@pytest.fixture(scope="module")
def hargreaves_daily_tmax_celsius():
    """
    Daily maximum temperature fixture for Hargreaves tests.
    Generates synthetic seasonal temperature data (2 years x 366 days).
    Temperature varies sinusoidally between 15 and 30 degrees C.
    Always higher than tmin by design.
    """
    num_days = _HARGREAVES_NUM_YEARS * 366
    day_of_year = np.tile(np.arange(1, 367), _HARGREAVES_NUM_YEARS)
    # same phase as tmin but higher base and amplitude
    tmax = 22.5 + 7.5 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    return tmax[:num_days]


@pytest.fixture(scope="module")
def hargreaves_daily_tmean_celsius():
    """
    Daily mean temperature fixture for Hargreaves tests.
    Computed as midpoint between tmin and tmax fixtures.
    """
    num_days = _HARGREAVES_NUM_YEARS * 366
    day_of_year = np.tile(np.arange(1, 367), _HARGREAVES_NUM_YEARS)
    # mean is midpoint between tmin and tmax patterns
    tmin = 10.0 + 5.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    tmax = 22.5 + 7.5 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    tmean = (tmin + tmax) / 2.0
    return tmean[:num_days]


@pytest.fixture(scope="module")
def hargreaves_latitude_degrees():
    """Latitude for Hargreaves tests (mid-latitude location)."""
    return 35.0


# ==============================================================================
# xarray fixtures for testing xarray-aware climate index computations
# ==============================================================================

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402


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

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
# xarray-based fixtures for Story 4.x tests
# Session-scoped fixtures that generate synthetic DataArrays for edge case
# testing, metadata validation, and xarray integration tests.
# ==============================================================================

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402


@pytest.fixture(scope="session")
def sample_monthly_precip_da() -> xr.DataArray:
    """
    Standard 1D monthly precipitation DataArray for testing.
    40 years (1980-2019), 480 months, realistic variation (0-200mm).
    """
    rng = np.random.default_rng(42)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    data = rng.gamma(shape=2.0, scale=25.0, size=480)
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def sample_daily_precip_da() -> xr.DataArray:
    """
    1D daily precipitation DataArray for testing.
    5 years (2015-2019), ~1826 days, realistic variation.
    """
    rng = np.random.default_rng(123)
    time = pd.date_range("2015-01-01", periods=1826, freq="D")
    data = rng.gamma(shape=1.5, scale=3.0, size=1826)
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Daily Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def sample_monthly_pet_da() -> xr.DataArray:
    """
    Standard 1D monthly PET DataArray matching sample_monthly_precip_da time range.
    40 years (1980-2019), 480 months, realistic PET variation (50-150mm).
    """
    rng = np.random.default_rng(100)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
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
def dask_monthly_precip_1d(sample_monthly_precip_da: xr.DataArray) -> xr.DataArray:
    """
    Dask-backed 1D monthly precipitation.
    Chunked along time dimension (full chunk = -1 for simplicity).
    """
    return sample_monthly_precip_da.chunk({"time": -1})


@pytest.fixture(scope="session")
def dask_monthly_precip_3d() -> xr.DataArray:
    """
    Dask-backed 3D monthly precipitation (time, lat, lon).
    40 years, 5 lat points, 6 lon points. Chunked along time.
    """
    rng = np.random.default_rng(150)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    lat = np.linspace(-10.0, 10.0, 5)
    lon = np.linspace(-15.0, 15.0, 6)
    data = rng.gamma(shape=2.0, scale=25.0, size=(480, 5, 6))
    da = xr.DataArray(
        data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )
    return da.chunk({"time": -1, "lat": 5, "lon": 6})


@pytest.fixture(scope="session")
def gridded_monthly_precip_3d() -> xr.DataArray:
    """
    Eager 3D monthly precipitation (time, lat, lon).
    40 years, 5 lat points, 6 lon points. Not chunked.
    """
    rng = np.random.default_rng(160)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    lat = np.linspace(-10.0, 10.0, 5)
    lon = np.linspace(-15.0, 15.0, 6)
    data = rng.gamma(shape=2.0, scale=25.0, size=(480, 5, 6))
    return xr.DataArray(
        data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def gridded_monthly_pet_3d() -> xr.DataArray:
    """
    Eager 3D monthly PET (time, lat, lon) matching gridded_monthly_precip_3d dimensions.
    40 years, 5 lat points, 6 lon points. Not chunked.
    """
    rng = np.random.default_rng(200)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    lat = np.linspace(-10.0, 10.0, 5)
    lon = np.linspace(-15.0, 15.0, 6)
    data = 50.0 + rng.gamma(shape=3.0, scale=20.0, size=(480, 5, 6))
    return xr.DataArray(
        data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        name="pet",
        attrs={
            "units": "mm",
            "long_name": "Potential Evapotranspiration",
            "standard_name": "water_potential_evaporation_amount",
        },
    )


@pytest.fixture(scope="session")
def single_point_monthly_da() -> xr.DataArray:
    """
    Single-point monthly DataArray (1D time-only).
    40 years (1980-2019), 480 months.
    """
    rng = np.random.default_rng(170)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    data = rng.gamma(shape=2.0, scale=25.0, size=480)
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def minimum_calibration_da() -> xr.DataArray:
    """
    Exactly 30 years of monthly data (360 months).
    Minimum valid calibration period for SPI/SPEI (1990-2019).
    """
    rng = np.random.default_rng(180)
    time = pd.date_range("1990-01-01", periods=360, freq="MS")
    data = rng.gamma(shape=2.0, scale=25.0, size=360)
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def minimum_calibration_pet_da() -> xr.DataArray:
    """
    Exactly 30 years of monthly PET (360 months) matching minimum_calibration_da.
    Time range: 1990-2019.
    """
    rng = np.random.default_rng(210)
    time = pd.date_range("1990-01-01", periods=360, freq="MS")
    data = 50.0 + rng.gamma(shape=3.0, scale=20.0, size=360)
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
def zero_inflated_precip_da() -> xr.DataArray:
    """
    Zero-inflated precipitation (~50% zeros).
    40 years (1980-2019), 480 months.
    """
    rng = np.random.default_rng(190)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    data = rng.gamma(shape=2.0, scale=25.0, size=480)
    # set ~50% of values to zero
    zero_mask = rng.random(480) < 0.5
    data[zero_mask] = 0.0
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def leading_nan_block_da() -> xr.DataArray:
    """
    Precipitation with first 12 months as NaN.
    40 years (1980-2019), 480 months total.
    """
    rng = np.random.default_rng(195)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    data = rng.gamma(shape=2.0, scale=25.0, size=480)
    data[:12] = np.nan
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def trailing_nan_block_da() -> xr.DataArray:
    """
    Precipitation with last 12 months as NaN.
    40 years (1980-2019), 480 months total.
    """
    rng = np.random.default_rng(196)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    data = rng.gamma(shape=2.0, scale=25.0, size=480)
    data[-12:] = np.nan
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def block_nan_pattern_da() -> xr.DataArray:
    """
    Precipitation with contiguous NaN block (months 240-251, i.e., year 21).
    40 years (1980-2019), 480 months total.
    """
    rng = np.random.default_rng(197)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    data = rng.gamma(shape=2.0, scale=25.0, size=480)
    data[240:252] = np.nan
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )


@pytest.fixture(scope="session")
def random_nan_precip_da() -> xr.DataArray:
    """
    Precipitation with ~10% random scattered NaN values.
    40 years (1980-2019), 480 months.
    """
    rng = np.random.default_rng(55)
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    data = rng.gamma(shape=2.0, scale=25.0, size=480)
    # randomly set ~10% to NaN
    nan_mask = rng.random(480) < 0.1
    data[nan_mask] = np.nan
    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
            "standard_name": "precipitation_amount",
        },
    )

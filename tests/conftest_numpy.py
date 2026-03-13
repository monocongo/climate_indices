from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# get the tests directory path for fixture data loading
_TESTS_DIR = Path(__file__).parent
_FIXTURE_DIR = _TESTS_DIR / "fixture"


@pytest.fixture(scope="module")
def precips_mm_monthly() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "precips_mm_monthly.npy")


@pytest.fixture(scope="module")
def precips_mm_daily() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "precips_mm_daily.npy")


@pytest.fixture(scope="module")
def transformed_pearson3() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "pearson3_monthly.npy")


@pytest.fixture(scope="module")
def transformed_pearson3_monthly_fullperiod() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "pearson3_monthly_full.npy")


@pytest.fixture(scope="module")
def gamma_monthly() -> np.ndarray:
    """
    Array of values corresponding to the gamma fitting parameters computed
    from the precips_mm_monthly array using an application of the gamma
    conversion/fitting algorithm provided in compute.py.
    """
    return np.load(_FIXTURE_DIR / "gamma_monthly.npy")


@pytest.fixture(scope="module")
def gamma_daily() -> np.ndarray:
    """
    Array of values corresponding to the gamma fitting parameters computed
    from the precips_mm_daily array using an application of the gamma
    conversion/fitting algorithm provided in compute.py.
    """
    return np.load(_FIXTURE_DIR / "gamma_daily.npy")


@pytest.fixture(scope="module")
def transformed_gamma_monthly() -> np.ndarray:
    """
    Array of values corresponding to the precips_mm_monthly array, i.e. those
    values have been transformed/fitted using an application of the gamma
    conversion/fitting algorithm provided in compute.py.
    """
    return np.load(_FIXTURE_DIR / "transformed_gamma_monthly.npy")


@pytest.fixture(scope="module")
def transformed_gamma_daily() -> np.ndarray:
    """
    Array of values corresponding to the precips_mm_daily array, i.e. those
    values have been transformed/fitted using an application of the gamma
    conversion/fitting algorithm provided in compute.py.
    """
    return np.load(_FIXTURE_DIR / "transformed_gamma_daily.npy")


@pytest.fixture(scope="module")
def pet_thornthwaite_mm() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "pet_thornthwaite.npy")


@pytest.fixture(scope="module")
def temps_celsius() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "temp_celsius.npy")


@pytest.fixture(scope="module")
def pnp_6month() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "pnp_06.npy")


@pytest.fixture(scope="module")
def spei_6_month_gamma() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "spei_06_gamma.npy")


@pytest.fixture(scope="module")
def spei_6_month_pearson3() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "spei_06_pearson3.npy")


@pytest.fixture(scope="module")
def spi_1_month_gamma() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "spi_01_gamma.npy")


@pytest.fixture(scope="module")
def spi_6_month_gamma() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "spi_06_gamma.npy")


@pytest.fixture(scope="module")
def spi_6_month_pearson3() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "spi_06_pearson3.npy")


@pytest.fixture(scope="function")
def rain_mm() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "rain_mm.npy")


@pytest.fixture(scope="function")
def rain_mm_365() -> np.ndarray:
    return np.load(_FIXTURE_DIR / "rain_mm_365.npy")


@pytest.fixture(scope="function")
def rain_mm_366() -> np.ndarray:
    """Fixed scoping to match rain_mm and rain_mm_365 (function-scoped)."""
    return np.load(_FIXTURE_DIR / "rain_mm_366.npy")


@pytest.fixture(scope="module")
def palmer_awcs() -> dict:
    with open(_FIXTURE_DIR / "palmer_awc.json") as awcfile:
        return json.load(awcfile)


# ==============================================================================
# Hargreaves fixtures for daily evapotranspiration calculations
# ==============================================================================

# start and end years for daily temperature data used in Hargreaves tests
_HARGREAVES_DATA_YEAR_START = 2000
_HARGREAVES_NUM_YEARS = 2


@pytest.fixture(scope="module")
def hargreaves_data_year_start() -> int:
    return _HARGREAVES_DATA_YEAR_START


@pytest.fixture(scope="module")
def hargreaves_daily_tmin_celsius() -> np.ndarray:
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
def hargreaves_daily_tmax_celsius() -> np.ndarray:
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
def hargreaves_daily_tmean_celsius() -> np.ndarray:
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
def hargreaves_latitude_degrees() -> float:
    """Latitude for Hargreaves tests (mid-latitude location)."""
    return 35.0

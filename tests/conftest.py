from __future__ import annotations

import pytest

# register pytest plugin modules for fixture discovery
pytest_plugins = [
    "tests.conftest_numpy",
    "tests.conftest_xarray",
    "tests.conftest_benchmark",
    "tests.helpers.logging",
]

# ==============================================================================
# Shared constants and configuration
# ==============================================================================

# start and end year of the monthly precipitation, temperature, and PET datasets
_DATA_YEAR_START_MONTHLY = 1895
_DATA_YEAR_END_MONTHLY = 2017

# start and end year of the calibration periods used by SPI/SPEI Pearson calculations
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


# ==============================================================================
# Shared configuration fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def data_year_start_monthly() -> int:
    return _DATA_YEAR_START_MONTHLY


@pytest.fixture(scope="module")
def data_year_end_monthly() -> int:
    return _DATA_YEAR_END_MONTHLY


@pytest.fixture(scope="module")
def data_year_start_daily() -> int:
    return _DATA_YEAR_START_DAILY


@pytest.fixture(scope="module")
def data_year_start_palmer() -> int:
    return _DATA_YEAR_START_MONTHLY


@pytest.fixture(scope="module")
def calibration_year_start_monthly() -> int:
    return _CALIBRATION_YEAR_START_MONTHLY


@pytest.fixture(scope="module")
def calibration_year_end_monthly() -> int:
    return _CALIBRATION_YEAR_END_MONTHLY


@pytest.fixture(scope="module")
def calibration_year_start_daily() -> int:
    return _CALIBRATION_YEAR_START_DAILY


@pytest.fixture(scope="module")
def calibration_year_start_palmer() -> int:
    return _CALIBRATION_YEAR_START_PALMER


@pytest.fixture(scope="module")
def calibration_year_end_palmer() -> int:
    return _CALIBRATION_YEAR_END_PALMER


@pytest.fixture(scope="module")
def calibration_year_end_daily() -> int:
    return _CALIBRATION_YEAR_END_DAILY


@pytest.fixture(scope="module")
def latitude_degrees() -> float:
    return _LATITUDE_DEGREES

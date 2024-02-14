import os
import json

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
    with open(os.path.join(os.path.split(__file__)[0], "fixture", "palmer_awc.json"),"r") as awcfile:
        return json.load(awcfile)


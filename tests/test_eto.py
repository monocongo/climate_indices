import logging
import math

import numpy as np
import pytest

from climate_indices import eto

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
def test_eto_hargreaves():
    pass

    # # compute PET from the daily temperatures and latitude
    # computed_pet = eto.eto_hargreaves(temps_celsius,
    #                                   latitude_degrees,
    #                                   data_year_start_monthly)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "temps_celsius",
    "latitude_degrees",
    "data_year_start_monthly",
    "pet_thornthwaite_mm",
)
def test_eto_thornthwaite(temps_celsius, latitude_degrees, data_year_start_monthly, pet_thornthwaite_mm):
    # compute PET from the monthly temperatures, latitude, and initial years
    computed_pet = eto.eto_thornthwaite(temps_celsius, latitude_degrees, data_year_start_monthly)

    # make sure PET is being computed as expected
    np.testing.assert_allclose(
        computed_pet,
        pet_thornthwaite_mm.flatten(),
        atol=0.001,
        equal_nan=True,
        err_msg="PET (Thornthwaite) values not computed as expected",
    )

    # make sure that a 3-D array raises an error
    reshaped_temps = np.reshape(temps_celsius[0:1400], (123, 2, 6))
    pytest.raises(
        ValueError,
        eto.eto_thornthwaite,
        reshaped_temps,
        latitude_degrees,
        data_year_start_monthly,
    )

    # make sure that an invalid latitude value (lat > 90) raises an error
    pytest.raises(
        ValueError,
        eto.eto_thornthwaite,
        temps_celsius,
        91.0,  # latitude > 90 is invalid
        data_year_start_monthly,
    )

    # make sure that an invalid latitude value (lat < -90) raises an error
    pytest.raises(
        ValueError,
        eto.eto_thornthwaite,
        temps_celsius,
        -91.0,  # latitude < -90 is invalid
        data_year_start_monthly,
    )

    # make sure that an invalid latitude value (None) raises an error
    pytest.raises(TypeError, eto.eto_thornthwaite, temps_celsius, None, data_year_start_monthly)

    # make sure that an invalid latitude value (NaN) raises an error
    pytest.raises(ValueError, eto.eto_thornthwaite, temps_celsius, np.NaN, data_year_start_monthly)


# ------------------------------------------------------------------------------
def test_sunset_hour_angle():
    # make sure that an invalid latitude value raises an error
    pytest.raises(ValueError, eto._sunset_hour_angle, np.deg2rad(-100.0), np.deg2rad(0.0))
    pytest.raises(ValueError, eto._sunset_hour_angle, np.NaN, np.deg2rad(0.0))

    # make sure that an invalid solar declination angle raises an error
    pytest.raises(ValueError, eto._sunset_hour_angle, np.deg2rad(0.0), np.deg2rad(-75.0))
    pytest.raises(ValueError, eto._sunset_hour_angle, np.deg2rad(0.0), np.deg2rad(85.0))
    pytest.raises(ValueError, eto._sunset_hour_angle, np.deg2rad(0.0), np.NaN)

    expected_value = math.pi / 2
    computed_value = eto._sunset_hour_angle(0.0, np.deg2rad(0.0))
    np.testing.assert_allclose(
        computed_value,
        expected_value,
        atol=0.0001,
        err_msg="Sunset hour angle not computed as expected",
    )

    expected_value = 1.6018925422201806
    computed_value = eto._sunset_hour_angle(np.deg2rad(10.0), np.deg2rad(10.0))
    np.testing.assert_allclose(
        computed_value,
        expected_value,
        atol=0.0001,
        err_msg="Sunset hour angle not computed as expected",
    )


# ------------------------------------------------------------------------------
def test_solar_declination():
    # make sure invalid arguments raise an error
    pytest.raises(ValueError, eto._solar_declination, 0)
    pytest.raises(ValueError, eto._solar_declination, -1)
    pytest.raises(ValueError, eto._solar_declination, 367)
    pytest.raises(ValueError, eto._solar_declination, 5000)
    pytest.raises(ValueError, eto._solar_declination, np.NaN)

    expected_value = -0.313551072399921
    computed_value = eto._solar_declination(30)
    np.testing.assert_allclose(
        computed_value,
        expected_value,
        atol=0.0001,
        err_msg="Solar declination not computed as expected",
    )


# ------------------------------------------------------------------------------
def test_daylight_hours():
    # make sure invalid arguments raise an error
    pytest.raises(ValueError, eto._daylight_hours, math.pi + 1)
    pytest.raises(ValueError, eto._daylight_hours, -1.0)
    pytest.raises(ValueError, eto._daylight_hours, np.NaN)

    expected_value = 7.999999999999999
    computed_value = eto._daylight_hours(math.pi / 3)
    np.testing.assert_equal(
        computed_value,
        expected_value,
        err_msg="Daylight hours not computed as expected",
    )

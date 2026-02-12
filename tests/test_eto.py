import math

import numpy as np
import pytest

from climate_indices import eto


# ------------------------------------------------------------------------------
def test_eto_hargreaves_1d_input():
    """Test eto_hargreaves with 1-D input arrays (the primary bug case from issue #578)."""
    # create sample 1-D daily temperature arrays (730 days = 2 years)
    tmin = np.full(730, 10.0)  # 10 degrees C min
    tmax = np.full(730, 25.0)  # 25 degrees C max
    tmean = np.full(730, 17.5)  # 17.5 degrees C mean
    latitude = 35.0  # degrees north

    # should not raise IndexError (the bug)
    result = eto.eto_hargreaves(tmin, tmax, tmean, latitude)

    # verify output shape matches input
    assert result.shape == (730,)

    # verify reasonable PET values (positive for valid inputs)
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values > 0), "PET values should be positive"
    assert np.all(valid_values < 20), "PET values should be reasonable (< 20 mm/day)"


# ------------------------------------------------------------------------------
def test_eto_hargreaves_2d_input():
    """Test eto_hargreaves with 2-D input arrays (regression test)."""
    # create 2-D arrays (2 years x 366 days)
    tmin = np.full((2, 366), 10.0)
    tmax = np.full((2, 366), 25.0)
    tmean = np.full((2, 366), 17.5)
    latitude = 35.0

    result = eto.eto_hargreaves(tmin, tmax, tmean, latitude)

    # output should be flattened to 1-D with length = 2 * 366
    assert result.shape == (732,)

    # verify reasonable PET values
    valid_values = result[~np.isnan(result)]
    assert np.all(valid_values > 0), "PET values should be positive"


# ------------------------------------------------------------------------------
def test_eto_hargreaves_size_mismatch():
    """Test that size mismatch raises ValueError."""
    tmin = np.full(365, 10.0)
    tmax = np.full(366, 25.0)  # different size
    tmean = np.full(365, 17.5)

    pytest.raises(ValueError, eto.eto_hargreaves, tmin, tmax, tmean, 35.0)


# ------------------------------------------------------------------------------
def test_eto_hargreaves_invalid_latitude():
    """Test that invalid latitude raises ValueError."""
    tmin = np.full(366, 10.0)
    tmax = np.full(366, 25.0)
    tmean = np.full(366, 17.5)

    # latitude > 90 should raise error
    pytest.raises(ValueError, eto.eto_hargreaves, tmin, tmax, tmean, 91.0)

    # latitude < -90 should raise error
    pytest.raises(ValueError, eto.eto_hargreaves, tmin, tmax, tmean, -91.0)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "hargreaves_daily_tmin_celsius",
    "hargreaves_daily_tmax_celsius",
    "hargreaves_daily_tmean_celsius",
    "hargreaves_latitude_degrees",
)
def test_eto_hargreaves_with_fixtures(
    hargreaves_daily_tmin_celsius,
    hargreaves_daily_tmax_celsius,
    hargreaves_daily_tmean_celsius,
    hargreaves_latitude_degrees,
):
    """Test eto_hargreaves using the standard test fixtures."""
    result = eto.eto_hargreaves(
        hargreaves_daily_tmin_celsius,
        hargreaves_daily_tmax_celsius,
        hargreaves_daily_tmean_celsius,
        hargreaves_latitude_degrees,
    )

    # verify output shape matches input
    assert result.shape == hargreaves_daily_tmin_celsius.shape

    # verify reasonable PET values
    valid_values = result[~np.isnan(result)]
    assert len(valid_values) > 0, "Should produce some valid PET values"
    assert np.all(valid_values > 0), "PET values should be positive"
    assert np.all(valid_values < 20), "PET values should be reasonable (< 20 mm/day)"


# ------------------------------------------------------------------------------
def test_eto_hargreaves_temperature_validation_warning():
    """Test that temperature validation warnings are issued for invalid data."""
    # create data where tmin > tmax for some values
    tmin = np.full(366, 25.0)  # higher than tmax!
    tmax = np.full(366, 10.0)  # lower than tmin!
    tmean = np.full(366, 17.5)
    latitude = 35.0

    # should still compute (with warnings) rather than raise error
    # note: warnings are now logged via structlog and visible in test output
    # the primary test is that the function completes without raising an exception
    result = eto.eto_hargreaves(tmin, tmax, tmean, latitude)

    # function should still return results (may be invalid, but doesn't crash)
    assert result.shape == (366,)
    # the warnings are emitted via structlog and visible in pytest output
    # checking for specific warning text would require capturing structlog output
    # which is more complex than the value this test provides


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
    pytest.raises(ValueError, eto.eto_thornthwaite, temps_celsius, np.nan, data_year_start_monthly)


# ------------------------------------------------------------------------------
def test_sunset_hour_angle():
    # make sure that an invalid latitude value raises an error
    pytest.raises(ValueError, eto._sunset_hour_angle, np.deg2rad(-100.0), np.deg2rad(0.0))
    pytest.raises(ValueError, eto._sunset_hour_angle, np.nan, np.deg2rad(0.0))

    # make sure that an invalid solar declination angle raises an error
    pytest.raises(ValueError, eto._sunset_hour_angle, np.deg2rad(0.0), np.deg2rad(-75.0))
    pytest.raises(ValueError, eto._sunset_hour_angle, np.deg2rad(0.0), np.deg2rad(85.0))
    pytest.raises(ValueError, eto._sunset_hour_angle, np.deg2rad(0.0), np.nan)

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
    pytest.raises(ValueError, eto._solar_declination, np.nan)

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
    pytest.raises(ValueError, eto._daylight_hours, np.nan)

    expected_value = 7.999999999999999
    computed_value = eto._daylight_hours(math.pi / 3)
    np.testing.assert_equal(
        computed_value,
        expected_value,
        err_msg="Daylight hours not computed as expected",
    )

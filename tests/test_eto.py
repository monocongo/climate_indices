import logging
import math
import unittest

import numpy as np

from tests import fixtures
from climate_indices import eto

# ----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ----------------------------------------------------------------------------------------------------------------------
class EtoTestCase(fixtures.FixturesTestCase):
    """
    Tests for `eto.py`.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def test_eto_hargreaves(self):

        pass

        # # compute PET from the daily temperatures and latitude
        # computed_pet = eto.eto_hargreaves(self.fixture_temps_celsius,
        #                                   self.fixture_latitude_degrees,
        #                                   self.fixture_data_year_start_monthly)

    # ------------------------------------------------------------------------------------------------------------------
    def test_eto_thornthwaite(self):
        
        # compute PET from the monthly temperatures, latitude, and initial years
        computed_pet = eto.eto_thornthwaite(self.fixture_temps_celsius,
                                            self.fixture_latitude_degrees,
                                            self.fixture_data_year_start_monthly)
                                         
        # make sure PET is being computed as expected
        np.testing.assert_allclose(computed_pet, 
                                   self.fixture_pet_mm.flatten(),
                                   atol=0.001,
                                   equal_nan=True,
                                   err_msg='PET (Thornthwaite) values not computed as expected')

        # make sure that a 3-D array raises an error
        reshaped_temps = np.reshape(self.fixture_temps_celsius[0:1400], (123, 2, 6))
        self.assertRaises(ValueError, 
                          eto.eto_thornthwaite,
                          reshaped_temps, 
                          self.fixture_latitude_degrees, 
                          self.fixture_data_year_start_monthly)
        
        # make sure that an invalid latitude value (lat > 90) raises an error
        self.assertRaises(ValueError, 
                          eto.eto_thornthwaite,
                          self.fixture_temps_celsius, 
                          91.0,   # latitude > 90 is invalid 
                          self.fixture_data_year_start_monthly)

        # make sure that an invalid latitude value (lat < -90) raises an error
        self.assertRaises(ValueError, 
                          eto.eto_thornthwaite,
                          self.fixture_temps_celsius, 
                          -91.0,   # latitude < -90 is invalid 
                          self.fixture_data_year_start_monthly)

        # make sure that an invalid latitude value (None) raises an error
        self.assertRaises(TypeError, 
                          eto.eto_thornthwaite,
                          self.fixture_temps_celsius, 
                          None,
                          self.fixture_data_year_start_monthly)

        # make sure that an invalid latitude value (NaN) raises an error
        self.assertRaises(ValueError, 
                          eto.eto_thornthwaite,
                          self.fixture_temps_celsius, 
                          np.NaN,
                          self.fixture_data_year_start_monthly)

    # ------------------------------------------------------------------------------------------------------------------
    def test_sunset_hour_angle(self):

        # make sure that an invalid latitude value raises an error
        self.assertRaises(ValueError,
                          eto._sunset_hour_angle,
                          np.deg2rad(-100.0),
                          np.deg2rad(0.0))
        self.assertRaises(ValueError,
                          eto._sunset_hour_angle,
                          np.NaN,
                          np.deg2rad(0.0))

        # make sure that an invalid solar declination angle raises an error
        self.assertRaises(ValueError,
                          eto._sunset_hour_angle,
                          np.deg2rad(0.0),
                          np.deg2rad(-75.0))
        self.assertRaises(ValueError,
                          eto._sunset_hour_angle,
                          np.deg2rad(0.0),
                          np.deg2rad(85.0))
        self.assertRaises(ValueError,
                          eto._sunset_hour_angle,
                          np.deg2rad(0.0),
                          np.NaN)

        expected_value = math.pi / 2
        computed_value = eto._sunset_hour_angle(0.0, np.deg2rad(0.0))
        np.testing.assert_equal(computed_value,
                                expected_value,
                                err_msg='Sunset hour angle not computed as expected')

        expected_value = 1.6018925422201806
        computed_value = eto._sunset_hour_angle(np.deg2rad(10.0), np.deg2rad(10.0))
        np.testing.assert_equal(computed_value,
                                expected_value,
                                err_msg='Sunset hour angle not computed as expected')

    # ------------------------------------------------------------------------------------------------------------------
    def test_solar_declination(self):

        # make sure invalid arguments raise an error
        self.assertRaises(ValueError,
                          eto._solar_declination,
                          0)
        self.assertRaises(ValueError,
                          eto._solar_declination,
                          -1)
        self.assertRaises(ValueError,
                          eto._solar_declination,
                          367)
        self.assertRaises(ValueError,
                          eto._solar_declination,
                          5000)
        self.assertRaises(ValueError,
                          eto._solar_declination,
                          np.NaN)

        expected_value = -0.313551072399921
        computed_value = eto._solar_declination(30)
        np.testing.assert_equal(computed_value,
                                expected_value,
                                err_msg='Solar declination not computed as expected')

    # ------------------------------------------------------------------------------------------------------------------
    def test_daylight_hours(self):

        # make sure invalid arguments raise an error
        self.assertRaises(ValueError,
                          eto._daylight_hours,
                          math.pi + 1)
        self.assertRaises(ValueError,
                          eto._daylight_hours,
                          -1.0)
        self.assertRaises(ValueError,
                          eto._daylight_hours,
                          np.NaN)

        expected_value = 7.999999999999999
        computed_value = eto._daylight_hours(math.pi / 3)
        np.testing.assert_equal(computed_value,
                                expected_value,
                                err_msg='Daylight hours not computed as expected')


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    unittest.main()

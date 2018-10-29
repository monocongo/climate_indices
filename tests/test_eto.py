import logging
import numpy as np
import unittest

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

    def test_eto_thornthwaite(self):
        
        # compute PET from the monthly temperatures, latitude, and initial years above
        computed_pet = eto.eto_thornthwaite(self.fixture_temps_celsius,
                                            self.fixture_latitude_degrees,
                                            self.fixture_data_year_start_monthly)
                                         
        # make sure PET is being computed as expected
        np.testing.assert_allclose(computed_pet, 
                                   self.fixture_pet_mm.flatten(),
                                   atol=0.001,
                                   equal_nan=True,
                                   err_msg='PET values not computed as expected')

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


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    unittest.main()

import logging
import numpy as np
import unittest

from tests import fixtures
from climate_indices import thornthwaite

#-----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)

#-----------------------------------------------------------------------------------------------------------------------
class ThornthwaiteTestCase(fixtures.FixturesTestCase):
    '''
    Tests for `thornthwaite.py`.
    '''

    def test_potential_evapotranspiration(self):
        
        # compute PET from the monthly temperatures, latitude, and initial years above
        computed_pet = thornthwaite.potential_evapotranspiration(self.fixture_temps_celsius,
                                                                 self.fixture_latitude_degrees, 
                                                                 self.fixture_data_year_start_monthly)
                                         
#         # make sure that PET is being computed as expected
#         self.assertTrue(np.allclose(computed_pet, 
#                                     self.fixture_pet_mm.flatten(), 
#                                     equal_nan=True), 
#                         'PET values not computed as expected')
        # make sure PET is being computed as expected
        np.testing.assert_allclose(computed_pet, 
                                   self.fixture_pet_mm.flatten(),
                                   atol=0.001,
                                   equal_nan=True,
                                   err_msg='PET values not computed as expected')

        # make sure that a 3-D array raises an error
        reshaped_temps = np.reshape(self.fixture_temps_celsius[0:1400], (123, 2, 6))
        self.assertRaises(ValueError, 
                          thornthwaite.potential_evapotranspiration, 
                          reshaped_temps, 
                          self.fixture_latitude_degrees, 
                          self.fixture_data_year_start_monthly)
        
        # make sure that an invalid latitude value (lat > 90) raises an error
        self.assertRaises(ValueError, 
                          thornthwaite.potential_evapotranspiration, 
                          self.fixture_temps_celsius, 
                          91.0,   # latitude > 90 is invalid 
                          self.fixture_data_year_start_monthly)

        # make sure that an invalid latitude value (lat < -90) raises an error
        self.assertRaises(ValueError, 
                          thornthwaite.potential_evapotranspiration, 
                          self.fixture_temps_celsius, 
                          -91.0,   # latitude < -90 is invalid 
                          self.fixture_data_year_start_monthly)

        # make sure that an invalid latitude value (None) raises an error
        self.assertRaises(TypeError, 
                          thornthwaite.potential_evapotranspiration, 
                          self.fixture_temps_celsius, 
                          None,
                          self.fixture_data_year_start_monthly)

        # make sure that an invalid latitude value (NaN) raises an error
        self.assertRaises(ValueError, 
                          thornthwaite.potential_evapotranspiration, 
                          self.fixture_temps_celsius, 
                          np.NaN,
                          self.fixture_data_year_start_monthly)

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    unittest.main()
    
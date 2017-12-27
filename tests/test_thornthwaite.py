import logging
import numpy as np
import unittest

#FIXME uncomment below once absolute imports issue is worked out, below should be correct based on docs
# import thornthwaite
# from tests import fixtures

#-----------------------------------------------------------------------------------------------------------------------
# use a context to add the required modules into the path for the test
# from tests.context import thornthwaite
# from tests import fixtures
from tests.context import thornthwaite
from tests import fixtures


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
                                                                 self.fixture_initial_data_year)
                                         
        # make sure that PET is being computed as expected
        self.assertTrue(np.allclose(computed_pet, self.fixture_pet_mm), 
                        'PET values not computed as expected')
        
        # make sure that a 3-D array raises an error
        reshaped_temps = np.reshape(self.fixture_temps_celsius[0:1400], (14, 5, 20))
        self.assertRaises(ValueError, 
                          thornthwaite.potential_evapotranspiration, 
                          reshaped_temps, 
                          self.fixture_latitude_degrees, 
                          self.fixture_initial_data_year)
        
        # make sure that an invalid latitude value raises an error
        self.assertRaises(ValueError, 
                          thornthwaite.potential_evapotranspiration, 
                          self.fixture_temps_celsius, 
                          91.0,   # latitude > 90 is invalid 
                          self.fixture_initial_data_year)

        # make sure that an invalid latitude value raises an error
        self.assertRaises(ValueError, 
                          thornthwaite.potential_evapotranspiration, 
                          self.fixture_temps_celsius, 
                          -91.0,   # latitude < -90 is invalid 
                          self.fixture_initial_data_year)

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    unittest.main()
    
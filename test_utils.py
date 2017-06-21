import logging
import math
import numpy as np
import scipy
import unittest
import utils

# disable logging messages
logging.disable(logging.CRITICAL)

#-------------------------------------------------------------------------------------------------------------------------------------------
class UtilsTestCase(unittest.TestCase):
    '''
    Tests for `utils.py`.
    '''

    def test_reshape_to_years_months(self):
        '''
        Test for the utils.reshape_to_years_months() function
        '''
        
        # an array of monthly values
        values_1d = np.array([3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2, 1, 3, 5, 8, 5, 6])
        
        # the expected rearrangement of the above values from 1-D to 2-D
        values_2d_expected = np.array([[3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4], 
                                       [6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2], 
                                       [1, 3, 5, 8, 5, 6, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]])
        
        # exercise the function
        values_2d_computed = utils.reshape_to_years_months(values_1d)
        
        # verify that the function performed as expected
        np.testing.assert_equal(values_2d_computed, 
                                values_2d_expected, 
                                'Not rearranging the 1-D array months into 2-D year increments as expected')
        
        # a 2-D array that should be returned as-is
        values_2d = np.array([[3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4], 
                              [6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2], 
                              [1, 3, 5, 8, 5, 6, 3, 5, 1, 2, 8, 4]])
        
        # exercise the function
        values_2d_computed = utils.reshape_to_years_months(values_2d)
        
         # verify that the function performed as expected
        np.testing.assert_equal(values_2d_computed, 
                                values_2d, 
                                'Not returning a valid 2-D array as expected')
        
        # a 2-D array that's in an invalid shape for the function
        values_2d = np.array([[3, 4, 6, 2, 1, 3, 5, 3, 4], 
                              [6, 2, 1, 3, 5, 8, 5, 6, 2], 
                              [1, 3, 5, 8, 5, 6, 3, 8, 4]])
        
        # make sure that the function croaks with a ValueError
        np.testing.assert_raises(ValueError, utils.reshape_to_years_months, values_2d)
        
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
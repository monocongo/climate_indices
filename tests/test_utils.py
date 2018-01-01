import logging
import numpy as np
import unittest

from indices_python import utils

# disable logging messages
logging.disable(logging.CRITICAL)

#-----------------------------------------------------------------------------------------------------------------------
class UtilsTestCase(unittest.TestCase):
    """
    Tests for `utils.py`.
    """
    
    #----------------------------------------------------------------------------------------
    def test_is_data_valid(self):
        """
        Test for the utils.is_data_valid() function
        """
        
        valid_array = np.full((12,), 1.0)
        invalid_array = np.full((12,), np.NaN)
        self.assertTrue(utils.is_data_valid(valid_array))
        self.assertFalse(utils.is_data_valid(invalid_array))
        self.assertFalse(utils.is_data_valid(['bad', 'data']))
        
    #----------------------------------------------------------------------------------------
    def test_rmse(self):
        """
        Test for the utils.rmse() function
        """
        
        vals1 = np.array([32, 212, 100, 98.6, 150, -15])
        vals2 = np.array([35, 216, 90, 88.6, 153, -12])
        computed_rmse = utils.rmse(vals1, vals2)
        expected_rmse = 6.364
                
        # verify that the function performed as expected
        self.assertAlmostEqual(computed_rmse, 
                               expected_rmse, 
                               msg='Incorrect root mean square error (RMSE)',
                               delta=0.001)
        
    #----------------------------------------------------------------------------------------
    def test_f2c(self):
        """
        Test for the utils.f2c() function
        """
        
        fahrenheit = np.array([32, 212, 100, 98.6, 150, -15])
        computed_celsius = utils.f2c(fahrenheit)
        expected_celsius = np.array([0, 100, 37.78, 37, 65.56, -26.11])
                
        # verify that the function performed as expected
        np.testing.assert_allclose(computed_celsius, 
                                   expected_celsius, 
                                   atol=0.01, 
                                   equal_nan=True,
                                   err_msg='Incorrect Fahrenheit to Celsius conversion')
        
    #----------------------------------------------------------------------------------------
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
        
    #----------------------------------------------------------------------------------------
    def test_count_zeros_and_non_missings(self):
        '''
        Test for the utils.count_zeros_and_non_missings() function
        '''
        values = np.array([3, 4, 0, 2, 3.1, 5, np.NaN, 8, 5, 6, 0.0, np.NaN, 5.6, 2])
        zeros, non_missings = utils.count_zeros_and_non_missings(values)
        self.assertEqual(zeros, 2, 'Failed to correctly count zero values')
        self.assertEqual(non_missings, 12, 'Failed to correctly count non-missing values')
                
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
    
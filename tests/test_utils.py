import io
import logging
import numpy as np
import unittest.mock

from climate_indices import utils

# disable logging messages
logging.disable(logging.CRITICAL)

#-----------------------------------------------------------------------------------------------------------------------
class UtilsTestCase(unittest.TestCase):
    """
    Tests for `utils.py`.
    """
    
    #----------------------------------------------------------------------------------------
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)  # mocks stdout, will be used as first mock_* argument
    def _assert_print_equal(self, 
                            values, 
                            expected_output, 
                            mock_stdout):

        utils.print_years_months(values)
        self.assertEqual(mock_stdout.getvalue(), expected_output)

    #----------------------------------------------------------------------------------------
    def test_print_years_months(self):
        """
        Test for the utils.print_years_months() function
        """
        
        #TODO come up with fixture and expected results
        fixture = np.array([1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        expected = " 1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00, 10.00, 11.00, 12.00,  \\" + "\n" + \
                   "13.00, 14.00, 15.00, 16.00,   nan,   nan,   nan,   nan,   nan,   nan,   nan,   nan,  \\" + "\n"

        self._assert_print_equal(fixture, expected)
        
    #----------------------------------------------------------------------------------------
    def test_compute_days(self):
        """
        Test for the utils.compute_days() function
        """
        
        days_array = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        results = utils.compute_days(1800, 12, 1, 1800)
        np.testing.assert_allclose(days_array, results, err_msg='Fahrenheit to Celsius conversion failed', atol=0.01, equal_nan=True)
        
        days_array = np.array([18443, 18474, 18505, 18535, 18566, 18596, 18627, 18658, 18686, 18717, \
                               18747, 18778, 18808, 18839, 18870, 18900, 18931, 18961, 18992, 19023])
        results = utils.compute_days(1850, 20, 7, 1800)
        np.testing.assert_allclose(days_array, results, err_msg='Fahrenheit to Celsius conversion failed', atol=0.01, equal_nan=True)

    #----------------------------------------------------------------------------------------
    def test_count_zeros_and_non_missings(self):
        '''
        Test for the utils.count_zeros_and_non_missings() function
        '''
        
        # vanilla use case
        values_list = [3, 4, 0, 2, 3.1, 5, np.NaN, 8, 5, 6, 0.0, np.NaN, 5.6, 2]
        values = np.array(values_list)
        zeros, non_missings = utils.count_zeros_and_non_missings(values)
        self.assertEqual(zeros, 2, 'Failed to correctly count zero values')
        self.assertEqual(non_missings, 12, 'Failed to correctly count non-missing values')
        
        # test with lists
        values = values_list
        zeros, non_missings = utils.count_zeros_and_non_missings(values)
        self.assertEqual(zeros, 2, 'Failed to correctly count zero values')
        self.assertEqual(non_missings, 12, 'Failed to correctly count non-missing values')
        values = [[3, 4, 0, 2, 3.1, 5, np.NaN], [8, 5, 6, 0.0, np.NaN, 5.6, 2]]
        zeros, non_missings = utils.count_zeros_and_non_missings(values)
        self.assertEqual(zeros, 2, 'Failed to correctly count zero values')
        self.assertEqual(non_missings, 12, 'Failed to correctly count non-missing values')

        # using a list that can't be converted into an array should result in a TypeError 
        values = [1, 2, 3, 0, 'abcxyz']
        np.testing.assert_raises(TypeError, utils.count_zeros_and_non_missings, values)

    #----------------------------------------------------------------------------------------
    def test_f2c(self):
        """
        Test for the utils.f2c() function
        """
                
        # verify that the function performs as expected
        fahrenheit_array = np.array([np.NaN, 32, 212, 100, 72, 98.6, 150, -15])
        celsius_array = np.array([np.NaN, 0, 100, 37.78, 22.22, 37, 65.56, -26.11])
        np.testing.assert_allclose(celsius_array, 
                                   utils.f2c(fahrenheit_array), 
                                   atol=0.01, 
                                   equal_nan=True,
                                   err_msg='Incorrect Fahrenheit to Celsius conversion')
        
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
        self.assertTrue(utils.is_data_valid(np.ma.masked_array(valid_array)))
        
    #----------------------------------------------------------------------------------------
    def test_sign_change(self):
        """
        Test for the utils.sign_change() function
        """
        
        a = np.array([1., 2., 3., -4])
        b = np.array([1., -2., -3., -4])
        c = utils.sign_change(a, b)
        np.testing.assert_equal(c, np.array([False, True, True, False]), 'Sign changes not detected as expected')
        
        a = np.array([1., 2., 3., -4])
        b = np.array([[1., -2.], [-3., -4]])
        c = utils.sign_change(a, b)
        np.testing.assert_equal(c, np.array([False, True, True, False]), 'Sign changes not detected as expected')
        
        # make sure that the function croaks with a ValueError
        np.testing.assert_raises(ValueError, 
                                 utils.sign_change, 
                                 np.array([1., 2., 3., -4]), 
                                 np.array([1., 2., 3.]))
        np.testing.assert_raises(ValueError, 
                                 utils.sign_change, 
                                 np.array([1., 2., 3.]), 
                                 np.array([[1., 2.], [3., 4.]]))
        
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
        values_2d_reshaped = utils.reshape_to_years_months(values_1d)
        
        # verify that the function performed as expected
        np.testing.assert_equal(values_2d_expected, 
                                values_2d_reshaped, 
                                'Not rearranging the 1-D array months into 2-D year increments as expected')
        
        # a 3-D array that should be returned as-is
        values_3d = np.array([[3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4], 
                              [6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2], 
                              [1, 3, 5, 8, 5, 6, 3, 5, 1, 2, 8, 4]])
        
        # exercise the function
        values_3d_reshaped = utils.reshape_to_years_months(values_3d)
        
        # verify that the function performed as expected
        np.testing.assert_equal(values_3d, 
                                values_3d_reshaped, 
                                'Not returning a valid 2-D array as expected')
        
        # a 2-D array that's in an invalid shape for the function
        values_2d = np.array([[3, 4, 6, 2, 1, 3, 5, 3, 4], 
                              [6, 2, 1, 3, 5, 8, 5, 6, 2], 
                              [1, 3, 5, 8, 5, 6, 3, 8, 4]])
        
        # make sure that the function croaks with a ValueError when expected
        np.testing.assert_raises(ValueError, utils.reshape_to_years_months, values_2d)
        np.testing.assert_raises(ValueError, utils.reshape_to_years_months, np.reshape(values_2d, (3, 3, 3)))
        
    #----------------------------------------------------------------------------------------
    def test_reshape_to_divs_years_months(self):
        '''
        Test for the utils.reshape_to_divs_years_months() function
        '''
        
        # an array of monthly values
        values_1d = np.array([3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2, 1, 3, 5, 8, 5, 6])
        
        # verify that the function performed as expected
        np.testing.assert_raises(ValueError, utils.reshape_to_divs_years_months, values_1d)

        # array of values for a single division, as 2-D
        values_2d = np.array([[3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4], 
                              [6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2]])
        
        # the expected rearrangement of the above values from 2-D to 3-D
        values_3d_expected = np.array([[[3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4]], 
                                        [[6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2]]])
        
        # exercise the function
        values_3d_computed = utils.reshape_to_divs_years_months(values_2d)
        
        np.testing.assert_equal(values_3d_computed, 
                                values_3d_expected, 
                                'Not rearranging the 1-D array months into 2-D year increments as expected')
        
        # a 3-D array that should be returned as-is
        values_3d = np.array([[[3, 4, 6, 2, 1, 3, 5, 8, 5, 6, 3, 4], 
                               [6, 2, 1, 3, 5, 8, 5, 6, 3, 4, 6, 2], 
                               [1, 3, 5, 8, 5, 6, 3, 5, 1, 2, 8, 4]],
                              [[6, 2, 8, 3, 2, 1, 9, 6, 3, 4, 9, 8],
                               [3, 1, 6, 2, 7, 3, 5, 8, 5, 6, 3, 4], 
                               [4, 2, 1, 7, 2, 8, 5, 6, 3, 4, 7, 9]]])
         
        # exercise the function
        values_3d_reshaped = utils.reshape_to_divs_years_months(values_3d)
         
        # verify that the function performed as expected
        np.testing.assert_equal(values_3d, 
                                values_3d_reshaped, 
                                'Not returning a valid 2-D array as expected')
         
        # a 2-D array that's in an invalid shape for the function
        values_2d = np.array([[3, 4, 6, 2, 1, 3, 5, 3, 4], 
                              [6, 2, 1, 3, 5, 8, 5, 6, 2], 
                              [1, 3, 5, 8, 5, 6, 3, 8, 4]])
        
        # make sure that the function croaks with a ValueError whenever it gets a mis-shaped array
        np.testing.assert_raises(ValueError, utils.reshape_to_divs_years_months, values_1d)
        np.testing.assert_raises(ValueError, utils.reshape_to_divs_years_months, values_2d)
        np.testing.assert_raises(ValueError, utils.reshape_to_divs_years_months, np.reshape(values_2d, (3, 3, 3)))
        
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
    def test_transform_to_gregorian(self):
        '''
        Test for the utils.transform_to_gregorian() function
        '''
        #TODO
        
    #----------------------------------------------------------------------------------------
    def test_transform_to_366day(self):
        '''
        Test for the utils.transform_to_366day() function
        '''
        #TODO
        
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
    
import compute
import fixtures
import logging
import numpy as np
import unittest

#-----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)

#-------------------------------------------------------------------------------------------------------------------------------------------
class ComputeTestCase(fixtures.FixturesTestCase):
    '''
    Tests for `compute.py`.
    '''

    #----------------------------------------------------------------------------------------
    def test_transform_fitted_gamma(self):
        '''
        Test for the compute.transform_fitted_gamma() function
        '''
         
        # compute sigmas of transformed (normalized) values fitted to a gamma distribution
        computed_values = compute.transform_fitted_gamma(self.fixture_precips_mm)
                                          
        # make sure the values are being computed as expected
        np.testing.assert_allclose(computed_values, 
                                   self.fixture_transformed_gamma,
                                   err_msg='Transformed gamma fitted values not computed as expected')            
         
    #----------------------------------------------------------------------------------------
    def test_transform_fitted_pearson(self):
        '''
        Test for the compute.transform_fitted_pearson() function
        '''
         
        # compute sigmas of transformed (normalized) values fitted to a gamma distribution
        computed_values = compute.transform_fitted_pearson(self.fixture_precips_mm, 
                                                           1895,
                                                           1981,
                                                           2010)
                                         
        # make sure the values are being computed as expected
        np.testing.assert_allclose(computed_values, 
                                   self.fixture_transformed_pearson3,
                                   atol=0.01,
                                   err_msg='Transformed Pearson Type III fitted values not computed as expected')
        
    #----------------------------------------------------------------------------------------
    def test_sum_to_scale(self):
        '''
        Test for the compute.sum_to_scale() function
        '''

        # test an input array with no missing values    
        values = np.array([3, 4, 6, 2, 1, 3, 5, 8, 5])
        computed_values = compute.sum_to_scale(values, 3)
        expected_values = np.array([np.NaN, np.NaN, 13, 12, 9, 6, 9, 16, 18])
        np.testing.assert_allclose(computed_values, 
                                   expected_values, 
                                   err_msg='Sliding sums not computed as expected')            

        # test an input array with missing values on the end    
        values = np.array([3, 4, 6, 2, 1, 3, 5, 8, 5, np.NaN, np.NaN, np.NaN])
        computed_values = compute.sum_to_scale(values, 3)
        expected_values = np.array([np.NaN, np.NaN, 13, 12, 9, 6, 9, 16, 18, np.NaN, np.NaN, np.NaN])
        np.testing.assert_allclose(computed_values, 
                                   expected_values, 
                                   err_msg='Sliding sums not computed as expected when missing values appended to end of input array')            
    
        # test an input array with missing values within the array    
        values = np.array([3, 4, 6, 2, 1, 3, 5, np.NaN, 8, 5, 6])
        computed_values = compute.sum_to_scale(values, 3)
        expected_values = np.array([np.NaN, np.NaN, 13, 12, 9, 6, 9, np.NaN, np.NaN, np.NaN, 19])
        np.testing.assert_allclose(computed_values, 
                                   expected_values, 
                                   err_msg='Sliding sums not computed as expected when missing values appended to end of input array')            
    
#     #----------------------------------------------------------------------------------------
#     def test_error_function(self):
#         '''
#         Test for the compute._error_function() function
#         '''
#         value = 0.24
#         erf_compute = compute._error_function(value)
#         erf_scipy = scipy.special.erf(value)
#         erf_math = math.erf(value)
#         self.assertEqual(erf_compute, erf_scipy, 'Failed to match error function values')
        
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
    
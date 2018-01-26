import logging
import numpy as np
import unittest

from tests import fixtures
from indices_python import compute, utils

#-----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)

#-------------------------------------------------------------------------------------------------------------------------------------------
class ComputeTestCase(fixtures.FixturesTestCase):
    '''
    Tests for `compute.py`.
    '''

    #----------------------------------------------------------------------------------------
    def test_estimate_lmoments(self):
        """
        Test for the compute._estimate_lmoments() function
        """
        # provide some bogus inputs to at least make sure these raise expected errors
        np.testing.assert_raises(ValueError, compute._estimate_lmoments, [1.0, 0.0, 0.0])
        np.testing.assert_raises(ValueError, compute._estimate_lmoments, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
        np.testing.assert_raises(TypeError, compute._estimate_lmoments, None)
              
        values = [0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.8, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7, 0.6]
        lmoments_expected = [0.7, 0.0470588235294, -9.43689570931e-15]
        lmoments_computed = compute._estimate_lmoments(values)
        np.testing.assert_allclose(lmoments_expected, 
                                   lmoments_computed,
                                   atol=0.001, 
                                   err_msg='Failed to accurately estimate L-moments')
        
    #----------------------------------------------------------------------------------------
    def test_estimate_pearson3_parameters(self):
        """
        Test for the compute._estimate_pearson3_parameters() function
        """
        # provide some bogus inputs to at least make sure these raise expected errors
        np.testing.assert_raises(ValueError, compute._estimate_pearson3_parameters, [1.0, 0.0, 0.0])
        np.testing.assert_raises(ValueError, compute._estimate_pearson3_parameters, [1.0, 1.0, 5.0])
        np.testing.assert_raises(ValueError, compute._estimate_pearson3_parameters, [1.0, -1.0, 1.0])
        np.testing.assert_raises(ValueError, compute._estimate_pearson3_parameters, [1.0, -1.0, 1e-7])
        
    #----------------------------------------------------------------------------------------
    def test_pearson3_fitting_values(self):
        """
        Test for the compute._pearson3_fitting_values() function
        """
        # provide some bogus inputs to at least make sure these raise expected errors
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([1.0, 0.0, 0.0]), 1950, 1952, 1970)
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 1950, 1952, 1970)
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]), 1950, 1952, 1970)
        np.testing.assert_raises(TypeError, compute._pearson3_fitting_values, None)
            
        computed_values = compute._pearson3_fitting_values(utils.reshape_to_years_months(self.fixture_precips_mm), 1950, 1952, 1970)
        expected_values = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [48.539987664499996, 53.9852487665, 44.284745065842102, 62.583727384894736, 125.72157689160528, 182.03053042784214, 159.00575657926319, 170.92269736865791, 189.8925781252895, 155.13420024692104, 72.953125000026319, 43.31532689144737],
                                    [33.781507724523095, 43.572151699968387, 40.368173442404107, 44.05329691434887, 60.10621716019174, 59.343178125457186, 49.228795303727473, 66.775653341386999, 65.362977393206421, 94.467597091088265, 72.63706898364299, 34.250906049301463],
                                    [0.76530966976335302, 1.2461447518219784, 2.275517179222323, 0.8069305098698194, -0.6783037020197018, 1.022194696224529, 0.40876120732817578, 1.2372551346168916, 0.73881116931924118, 0.91911763257003465, 2.3846715887263725, 1.4700559294571962]])
        np.testing.assert_allclose(computed_values, 
                                   expected_values, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='Failed to accurately compute Pearson Type III fitting values')

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
    
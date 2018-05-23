import logging
import math
import numpy as np
import unittest

from tests import fixtures
from climate_indices import compute

#-----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)

#-------------------------------------------------------------------------------------------------------------------------------------------
class ComputeTestCase(fixtures.FixturesTestCase):
    '''
    Tests for `compute.py`.
    '''

    #----------------------------------------------------------------------------------------
    def test_error_function(self):
        """
        Test for the compute._error_function() function
        """

        self.assertEqual(compute._error_function(0.0), 
                         0.0, 
                         msg='Failed to accurately compute error function')
        
        self.assertEqual(compute._error_function(5.0), 
                         0.9999999999992313, 
                         msg='Failed to accurately compute error function')

        self.assertEqual(compute._error_function(-5.0), 
                         -0.9999999999992313, 
                         msg='Failed to accurately compute error function')

        self.assertEqual(compute._error_function(5.5), 
                         0.9999999999999941, 
                         msg='Failed to accurately compute error function')

        self.assertEqual(compute._error_function(0.17), 
                         0.5949962306009045, 
                         msg='Failed to accurately compute error function')

        self.assertEqual(compute._error_function(-0.07), 
                         -0.5394288598854453, 
                         msg='Failed to accurately compute error function')

        self.assertEqual(compute._error_function(50.0), 
                         1.0, 
                         msg='Failed to accurately compute error function')

        self.assertEqual(compute._error_function(-50.0), 
                         -1.0, 
                         msg='Failed to accurately compute error function')

        self.assertEqual(compute._error_function(-2.0), 
                         -0.9976611325094764, 
                         msg='Failed to accurately compute error function')

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
    def test_pearson3cdf(self):
        """
        Test for the compute._pearson3cdf() function
        """

        self.assertTrue(math.isnan(compute._pearson3cdf(5.0, [1.0, -1.0, 0.0])), 
                        msg='Failed to accurately compute Pearson Type III CDF')

        self.assertEqual(compute._pearson3cdf(5.0, [1.0, 1.0, 1e-7]), 
                         0.9999841643790834, 
                         msg='Failed to accurately compute Pearson Type III CDF')
         
        self.assertEqual(compute._pearson3cdf(7.7, [1.0, 501.0, 0.0]), 
                         0.752667498611228, 
                         msg='Failed to accurately compute Pearson Type III CDF')
         
        self.assertEqual(compute._pearson3cdf(7.7, [1.0, 501.0, -10.0]), 
                         0.10519432662999628, 
                         msg='Failed to accurately compute Pearson Type III CDF')
         
        self.assertEqual(compute._pearson3cdf(1e-6, [441.0, 501.0, 30.0]), 
                         0.0005,  # value corresponding to trace value
                         msg='Failed to accurately compute Pearson Type III CDF')

    #----------------------------------------------------------------------------------------
    def test_pearson_fit_ufunc(self):
        """
        Test for the compute._pearson_fit_ufunc() function
        """

        self.assertTrue(math.isnan(compute._pearson_fit_ufunc(np.NaN, 1.0, -1.0, 0.0, 0.0)), 
                        msg='Failed to accurately compute error function')

        self.assertTrue(math.isnan(compute._pearson_fit_ufunc(5.0, 1.0, -1.0, 0.0, 0.0)), 
                        msg='Failed to accurately compute error function')

        self.assertEqual(compute._pearson_fit_ufunc(7.7, 1.0, 501.0, 0.0, 0.07), 
                         0.7387835329883602, 
                         msg='Failed to accurately compute error function')

    #----------------------------------------------------------------------------------------
    def test_pearson3_fitting_values(self):
        """
        Test for the compute._pearson3_fitting_values() function
        """
        # provide some bogus inputs to make sure these raise expected errors
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0],
                                                                                         [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.7]]))
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]))
        np.testing.assert_raises(AttributeError, compute._pearson3_fitting_values, None)
            
        # try using a subset of the precipitation dataset (1897 - 1915, year indices 2 - 20)
        computed_values = compute._pearson3_fitting_values(self.fixture_precips_mm_monthly[2:21, :])
        expected_values = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [48.539987664499996, 53.9852487665, 44.284745065842102, 62.583727384894736, 125.72157689160528, 182.03053042784214, 159.00575657926319, 170.92269736865791, 189.8925781252895, 155.13420024692104, 72.953125000026319, 43.31532689144737],
                                    [33.781507724523095, 43.572151699968387, 40.368173442404107, 44.05329691434887, 60.10621716019174, 59.343178125457186, 49.228795303727473, 66.775653341386999, 65.362977393206421, 94.467597091088265, 72.63706898364299, 34.250906049301463],
                                    [0.76530966976335302, 1.2461447518219784, 2.275517179222323, 0.8069305098698194, -0.6783037020197018, 1.022194696224529, 0.40876120732817578, 1.2372551346168916, 0.73881116931924118, 0.91911763257003465, 2.3846715887263725, 1.4700559294571962]])
        np.testing.assert_allclose(computed_values, 
                                   expected_values, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='Failed to accurately compute Pearson Type III fitting values')

        # add some zeros in order to exercise the parts where it gets a percentage of zeros
        precips_mm = np.array(self.fixture_precips_mm_monthly, copy=True)
        precips_mm[0, 1] = 0.0
        precips_mm[3, 4] = 0.0
        precips_mm[14, 9] = 0.0
        precips_mm[2, 5] = 0.0
        precips_mm[8, 3] = 0.0
        precips_mm[7, 11] = 0.0
        precips_mm[3, 9] = 0.0
        precips_mm[11, 4] = 0.0
        precips_mm[13, 5] = 0.0
        computed_values = compute._pearson3_fitting_values(precips_mm)
        expected_values = np.array([[0.0, 0.008130081300813009, 0.0, 0.0081967213114754103, 0.016393442622950821, 0.016393442622950821, 0.0, 0.0, 0.0, 0.016393442622950821, 0.0, 0.0081967213114754103],
                                    [45.85372999240245, 46.347934133658548, 48.324170722364769, 67.635750192172154, 121.1711705943935, 184.12836193667619, 154.96859791263938, 170.28928662930332, 196.42544505660646, 153.52549468512296, 58.400078445204926, 38.858758644995909],
                                    [38.873650403487751, 35.333748953293423, 34.315010982762324, 50.257217545953182, 73.519095805475956, 100.17902892507252, 50.629474961599207, 63.070686393124326, 75.262836828223314, 93.674893334263402, 48.751881843917658, 33.011345617774751],
                                    [1.7567209830258725, 1.2512959828378094, 1.1665495317869126, 1.1928621421474375, 0.76407610195548825, 0.836464024048587, 0.18285005633387616, 0.99639419415939923, 0.83383974102177649, 1.162401322000489, 1.8477937169727758, 1.8064543865073583]])
        np.testing.assert_allclose(computed_values, 
                                   expected_values, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='Failed to accurately compute Pearson Type III fitting values')

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
    
    #----------------------------------------------------------------------------------------
    def test_transform_fitted_gamma(self):
        '''
        Test for the compute.transform_fitted_gamma() function
        '''
        
        # compute sigmas of transformed (normalized) values fitted to a gamma distribution,
        # using the full period of record as the calibration period
        computed_values = compute.transform_fitted_gamma(self.fixture_precips_mm_monthly, 
                                                         self.fixture_data_year_start_monthly,
                                                         self.fixture_data_year_start_monthly,
                                                         self.fixture_data_year_end_monthly,
                                                         'monthly')
        np.testing.assert_allclose(computed_values, 
                                   self.fixture_transformed_gamma_monthly,
                                   err_msg='Transformed gamma fitted monthly values not computed as expected')            
         
         
        # compute sigmas of transformed (normalized) values fitted to a gamma distribution,
        # using the full period of record as the calibration period
        computed_values = compute.transform_fitted_gamma(self.fixture_precips_mm_daily.flatten(),
                                                         self.fixture_data_year_start_daily,
                                                         self.fixture_calibration_year_start_daily,
                                                         self.fixture_calibration_year_end_daily,
                                                         'daily')
        np.testing.assert_allclose(computed_values,
                                   self.fixture_transformed_gamma_daily,
                                   atol=0.001,
                                   equal_nan=True,
                                   err_msg='Transformed gamma fitted daily values not computed as expected')            
         
        # confirm that we can call with a calibration period outside of valid range 
        # and as a result use the full period of record as the calibration period instead
        computed_values = compute.transform_fitted_gamma(self.fixture_precips_mm_monthly, 
                                                         self.fixture_data_year_start_monthly,
                                                         1500,
                                                         2500,
                                                         'monthly')
        np.testing.assert_allclose(computed_values, 
                                   self.fixture_transformed_gamma_monthly,
                                   atol=0.001,
                                   equal_nan=True,
                                   err_msg='Transformed Pearson Type III fitted values not computed as expected')

        # if we provide a 1-D array then we need to provide a corresponding time series type, 
        # confirm we can't use an invalid type
        flat_array = self.fixture_precips_mm_monthly.flatten()
        np.testing.assert_raises(ValueError, 
                                 compute.transform_fitted_gamma, 
                                 flat_array, 
                                 self.fixture_data_year_start_monthly,
                                 self.fixture_calibration_year_start_monthly,
                                 self.fixture_calibration_year_end_monthly,
                                 'invalid_value')
        np.testing.assert_raises(ValueError, 
                                 compute.transform_fitted_gamma, 
                                 flat_array, 
                                 self.fixture_data_year_start_monthly,
                                 self.fixture_calibration_year_start_monthly,
                                 self.fixture_calibration_year_end_monthly,
                                 None)

        # confirm that an input array which is not 1-D or 2-D will raise an error
        self.assertRaises(ValueError,
                          compute.transform_fitted_gamma,
                          np.zeros((9, 8, 7, 6), dtype=float),
                          self.fixture_data_year_start_daily,
                          self.fixture_calibration_year_start_daily,
                          self.fixture_calibration_year_end_daily,
                          'monthly')

    #----------------------------------------------------------------------------------------
    def test_transform_fitted_pearson(self):
        '''
        Test for the compute.transform_fitted_pearson() function
        '''
        
        # compute sigmas of transformed (normalized) values fitted to a Pearson Type III distribution
        computed_values = compute.transform_fitted_pearson(self.fixture_precips_mm_monthly, 
                                                           self.fixture_data_year_start_monthly,
                                                           self.fixture_calibration_year_start_monthly,
                                                           self.fixture_calibration_year_end_monthly,
                                                           'monthly')
        np.testing.assert_allclose(computed_values, 
                                   self.fixture_transformed_pearson3,
                                   atol=0.001,
                                   err_msg='Transformed Pearson Type III fitted values not computed as expected')
        
        # confirm that an input array of all NaNs will return the same array
        all_nans = np.full(self.fixture_precips_mm_monthly.shape, np.NaN)
        computed_values = compute.transform_fitted_pearson(all_nans, 
                                                           self.fixture_data_year_start_monthly,
                                                           self.fixture_calibration_year_start_monthly,
                                                           self.fixture_calibration_year_end_monthly,
                                                           'monthly')
        np.testing.assert_allclose(computed_values, 
                                   all_nans,
                                   equal_nan=True,
                                   err_msg='Transformed Pearson Type III fitted values not computed as expected')
        
        # confirm that we can call with a calibration period outside of valid range 
        # and as a result use the full period of record as the calibration period instead
        computed_values = compute.transform_fitted_pearson(self.fixture_precips_mm_monthly, 
                                                           self.fixture_data_year_start_monthly,
                                                           1500,
                                                           2500,
                                                           'monthly')
        np.testing.assert_allclose(computed_values.flatten(), 
                                   self.fixture_transformed_pearson3_monthly_fullperiod,
                                   atol=0.001,
                                   equal_nan=True,
                                   err_msg='Transformed Pearson Type III fitted values not computed as expected')
        
        # confirm that we can call with daily values and not raise an error
        compute.transform_fitted_pearson(self.fixture_precips_mm_daily, 
                                         self.fixture_data_year_start_daily,
                                         self.fixture_calibration_year_start_daily,
                                         self.fixture_calibration_year_end_daily,
                                         'daily')
                                         
        # confirm that we get expected errors when using invalid time series type arguments
        self.assertRaises(ValueError,
                          compute.transform_fitted_pearson,
                          self.fixture_precips_mm_monthly.flatten(), 
                          self.fixture_data_year_start_monthly,
                          self.fixture_calibration_year_start_monthly,
                          self.fixture_calibration_year_end_monthly,
                          None)
        self.assertRaises(ValueError,
                          compute.transform_fitted_pearson,
                          self.fixture_precips_mm_monthly.flatten(), 
                          self.fixture_data_year_start_monthly,
                          self.fixture_calibration_year_start_monthly,
                          self.fixture_calibration_year_end_monthly,
                          'unsupported_type')

        # confirm that an input array which is not 1-D or 2-D will raise an error
        self.assertRaises(ValueError,
                          compute.transform_fitted_pearson,
                          np.zeros((9, 8, 7, 6), dtype=float),
                          self.fixture_data_year_start_daily,
                          self.fixture_calibration_year_start_daily,
                          self.fixture_calibration_year_end_daily,
                          'monthly')

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
    
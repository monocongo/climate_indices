import logging
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

        np.testing.assert_allclose(compute._pearson3cdf(5.0, [1.0, -1.0, 0.0]), 
                                   np.NaN, 
                                   atol=0.01, 
                                   equal_nan=True, 
                                   err_msg='Failed to accurately compute Pearson Type III CDF')

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
    def test_pearson3_fitting_values(self):
        """
        Test for the compute._pearson3_fitting_values() function
        """
        # provide some bogus inputs to at least make sure these raise expected errors
#         np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([1.0, 0.0, 0.0]), 1950, 1952, 1970)
#         np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 1950, 1952, 1970)
#         np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0],
#                                                                                          [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.7]]), 1950, 1952, 1970)
#         np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]), 1950, 1952, 1970)
#         np.testing.assert_raises(TypeError, compute._pearson3_fitting_values, None)
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0],
                                                                                         [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.7]]))
        np.testing.assert_raises(ValueError, compute._pearson3_fitting_values, np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]))
        np.testing.assert_raises(AttributeError, compute._pearson3_fitting_values, None)
            
#         reshaped_values = utils.reshape_to_2d(self.fixture_precips_mm, 12)
#         computed_values = compute._pearson3_fitting_values(reshaped_values, 1950, 1952, 1970)
        computed_values = compute._pearson3_fitting_values(self.fixture_precips_mm[2:21, :])
#         computed_values = compute._pearson3_fitting_values(self.fixture_precips_mm, 1950, 1952, 1970)
        expected_values = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [48.539987664499996, 53.9852487665, 44.284745065842102, 62.583727384894736, 125.72157689160528, 182.03053042784214, 159.00575657926319, 170.92269736865791, 189.8925781252895, 155.13420024692104, 72.953125000026319, 43.31532689144737],
                                    [33.781507724523095, 43.572151699968387, 40.368173442404107, 44.05329691434887, 60.10621716019174, 59.343178125457186, 49.228795303727473, 66.775653341386999, 65.362977393206421, 94.467597091088265, 72.63706898364299, 34.250906049301463],
                                    [0.76530966976335302, 1.2461447518219784, 2.275517179222323, 0.8069305098698194, -0.6783037020197018, 1.022194696224529, 0.40876120732817578, 1.2372551346168916, 0.73881116931924118, 0.91911763257003465, 2.3846715887263725, 1.4700559294571962]])
        np.testing.assert_allclose(computed_values, 
                                   expected_values, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='Failed to accurately compute Pearson Type III fitting values')

#         # use some nonsense calibration years to make sure these are handled as expected
#         computed_values = compute._pearson3_fitting_values(self.fixture_precips_mm, 1950, 1945, 1970)
#         expected_values = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                     [45.85372999240245, 47.044683689044724, 48.324170722364769, 67.648293417069695, 122.99461289716399, 186.72172771536472, 154.96859791263938, 170.28928662930332, 196.42544505660646, 156.65434490285244, 58.400078445204926, 39.304991675221316],
#                                     [38.873650403487751, 35.293694637792619, 34.315010982762324, 50.246089899974869, 72.614093396123764, 97.561428781577163, 50.629474961599207, 63.070686393124326, 75.262836828223314, 92.461158114814808, 48.751881843917658, 32.829910098364323],
#                                     [1.7567209830258725, 1.236465572421074, 1.1665495317869126, 1.1961332793113155, 0.80348157450648583, 0.96098107449522363, 0.18285005633387616, 0.99639419415939923, 0.83383974102177649, 1.237596091853048, 1.8477937169727758, 1.7951017162633573]])
#         np.testing.assert_allclose(computed_values, 
#                                    expected_values, 
#                                    atol=0.001, 
#                                    equal_nan=True, 
#                                    err_msg='Failed to accurately compute Pearson Type III fitting values')
#         computed_values = compute._pearson3_fitting_values(self.fixture_precips_mm, 1950, 1954, 2200)
#         expected_values = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                     [45.85372999240245, 47.044683689044724, 48.324170722364769, 67.648293417069695, 122.99461289716399, 186.72172771536472, 154.96859791263938, 170.28928662930332, 196.42544505660646, 156.65434490285244, 58.400078445204926, 39.304991675221316],
#                                     [38.873650403487751, 35.293694637792619, 34.315010982762324, 50.246089899974869, 72.614093396123764, 97.561428781577163, 50.629474961599207, 63.070686393124326, 75.262836828223314, 92.461158114814808, 48.751881843917658, 32.829910098364323],
#                                     [1.7567209830258725, 1.236465572421074, 1.1665495317869126, 1.1961332793113155, 0.80348157450648583, 0.96098107449522363, 0.18285005633387616, 0.99639419415939923, 0.83383974102177649, 1.237596091853048, 1.8477937169727758, 1.7951017162633573]])
#         np.testing.assert_allclose(computed_values, 
#                                    expected_values, 
#                                    atol=0.001, 
#                                    equal_nan=True, 
#                                    err_msg='Failed to accurately compute Pearson Type III fitting values')

        # add some zeros in order to exercise the parts where it gets a percentage of zeros
        precips_mm = np.array(self.fixture_precips_mm, copy=True)
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
#         computed_values = compute._pearson3_fitting_values(precips_mm, 1950, 1954, 2200)
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
        computed_values = compute.transform_fitted_gamma(self.fixture_precips_mm, 
                                                         self.fixture_data_start_year,
                                                         self.fixture_data_start_year,
                                                         self.fixture_data_end_year,
                                                         'monthly')
                                          
        # make sure the values are being computed as expected
        np.testing.assert_allclose(computed_values, 
                                   self.fixture_transformed_gamma,
                                   err_msg='Transformed gamma fitted values not computed as expected')            
         
    #----------------------------------------------------------------------------------------
    def test_transform_fitted_pearson(self):
        '''
        Test for the compute.transform_fitted_pearson() function
        '''
        
#         # get array indices corresponding to a calibration period of 1981 - 2010
#         year_index_start = 1981 - self.fixture_data_start_year + 1
#         year_index_end = 2010 - self.fixture_data_start_year + 1
#         
#         values = self.fixture_precips_mm[year_index_start:year_index_end + 1, :]
        
        # compute sigmas of transformed (normalized) values fitted to a gamma distribution
        computed_values = compute.transform_fitted_pearson(self.fixture_precips_mm, 
                                                           1895,
                                                           1981,
                                                           2010,
                                                           'monthly')
                                         
        # make sure the values are being computed as expected
        np.testing.assert_allclose(computed_values, 
                                   self.fixture_transformed_pearson3,
                                   atol=0.01,
                                   err_msg='Transformed Pearson Type III fitted values not computed as expected')
        
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
    
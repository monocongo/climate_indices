import logging
import numpy as np
import unittest

from tests import fixtures
from climate_indices import indices

#-----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)

#-----------------------------------------------------------------------------------------------------------------------
class IndicesTestCase(fixtures.FixturesTestCase):
    '''
    Tests for `indices.py`.
    '''

    #----------------------------------------------------------------------------------------
    def test_pdsi(self):
        
        # the indices.pdsi() function is a wrapper for palmer.pdsi(), so we'll 
        # just confirm that this function can be called without raising an error and 
        # the compute.pdsi() function itself being tested within test_palmer.py
        indices.pdsi(self.fixture_precips_mm_monthly, 
                     self.fixture_pet_mm,
                     self.fixture_awc_inches, 
                     self.fixture_data_year_start_monthly, 
                     self.fixture_calibration_year_start_monthly, 
                     self.fixture_calibration_year_end_monthly)
        
    #----------------------------------------------------------------------------------------
    def test_scpdsi(self):
        
        # the indices.scpdsi() function is a wrapper for palmer.pdsi(), so we'll 
        # just confirm that this function can be called without raising an error and 
        # the compute.pdsi() function itself being tested within test_palmer.py
        indices.scpdsi(self.fixture_precips_mm_monthly, 
                       self.fixture_pet_mm,
                       self.fixture_awc_inches, 
                       self.fixture_data_year_start_monthly, 
                       self.fixture_calibration_year_start_monthly, 
                       self.fixture_calibration_year_end_monthly)
        
    #----------------------------------------------------------------------------------------
    def test_pet(self):
        
        # confirm that an input array of all NaNs for temperature results in the same array returned
        all_nan_temps = np.full(self.fixture_temps_celsius.shape, np.NaN)
        computed_pet = indices.pet(all_nan_temps,
                                   self.fixture_latitude_degrees, 
                                   self.fixture_data_year_start_monthly)
        np.testing.assert_equal(computed_pet, 
                                all_nan_temps,
                                'All-NaN input array does not result in the expected all-NaN result')
        
        # confirm that a masked input array of all NaNs for temperature results in the same masked array returned
        masked_all_nan_temps = np.ma.array(all_nan_temps)
        computed_pet = indices.pet(masked_all_nan_temps,
                                   self.fixture_latitude_degrees, 
                                   self.fixture_data_year_start_monthly)
        np.testing.assert_equal(computed_pet, 
                                masked_all_nan_temps,
                                'All-NaN masked input array does not result in the expected all-NaN masked result')
        
        # confirm that a missing/None latitude value raises an error
        np.testing.assert_raises(ValueError, 
                                 indices.pet, 
                                 self.fixture_temps_celsius, 
                                 None, 
                                 self.fixture_data_year_start_monthly)
        
        # confirm that a missing/None latitude value raises an error
        np.testing.assert_raises(ValueError, 
                                 indices.pet, 
                                 self.fixture_temps_celsius, 
                                 np.NaN, 
                                 self.fixture_data_year_start_monthly)
        
        # confirm that an invalid latitude value raises an error
        self.assertRaises(ValueError, 
                          indices.pet, 
                          self.fixture_temps_celsius, 
                          91.0,   # latitude > 90 is invalid 
                          self.fixture_data_year_start_monthly)

        # confirm that an invalid latitude value raises an error
        np.testing.assert_raises(ValueError, 
                                 indices.pet, 
                                 self.fixture_temps_celsius, 
                                 -91.0,   # latitude < -90 is invalid 
                                 self.fixture_data_year_start_monthly)

        # compute PET from the monthly temperatures, latitude, and initial years -- if this runs without 
        # error then this test passes, as the underlying method(s) being used to compute PET will be tested 
        # in the relevant test_compute.py or test_thornthwaite.py codes
        computed_pet = indices.pet(self.fixture_temps_celsius,
                                   self.fixture_latitude_degrees, 
                                   self.fixture_data_year_start_monthly)
                                         
                                         
    #----------------------------------------------------------------------------------------
    def test_pnp(self):
                
        # compute PNP from the daily precipitation array
        computed_pnp_6month = indices.percentage_of_normal(self.fixture_precips_mm_monthly.flatten(),
                                                           6, 
                                                           self.fixture_data_year_start_monthly,
                                                           self.fixture_calibration_year_start_monthly, 
                                                           self.fixture_calibration_year_end_monthly, 
                                                           'monthly')

        # confirm PNP is being computed as expected
        np.testing.assert_allclose(self.fixture_pnp_6month,
                                   computed_pnp_6month, 
                                   atol=0.001,
                                   equal_nan=True,
                                   err_msg='PNP values not computed as expected')
                              
        # confirm we can compute PNP from the daily values without raising an error 
        indices.percentage_of_normal(self.fixture_precips_mm_daily.flatten(),
                                     30, 
                                     self.fixture_data_year_start_daily,
                                     self.fixture_calibration_year_start_daily, 
                                     self.fixture_calibration_year_end_daily, 
                                     'daily')
                
        # invalid periodicity argument should raise an AttributeError
        np.testing.assert_raises(ValueError, 
                                 indices.percentage_of_normal,
                                 self.fixture_precips_mm_daily.flatten(),
                                 30, 
                                 self.fixture_data_year_start_daily,
                                 self.fixture_calibration_year_start_daily, 
                                 self.fixture_calibration_year_end_daily, 
                                 'unsupported_value')

    #----------------------------------------------------------------------------------------
    def test_spi(self):
        
        # compute SPI/gamma at 1-month scale
        computed_spi = indices.spi(self.fixture_precips_mm_monthly, 
                                   1,
                                   indices.Distribution.gamma,
                                   self.fixture_data_year_start_monthly, 
                                   self.fixture_data_year_start_monthly, 
                                   self.fixture_data_year_end_monthly, 
                                   'monthly')
                                         
        # confirm SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_1_month_gamma, 
                                   atol=0.001,
                                   err_msg='SPI/Gamma values for 1-month scale not computed as expected')
        
        computed_spi = indices.spi(self.fixture_precips_mm_monthly.flatten(), 
                                   6,
                                   indices.Distribution.gamma,
                                   self.fixture_data_year_start_monthly, 
                                   self.fixture_data_year_start_monthly, 
                                   self.fixture_data_year_end_monthly, 
                                   'monthly')
                                         
        # confirm SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_6_month_gamma, 
                                   atol=0.001,
                                   err_msg='SPI/Gamma values for 6-month scale not computed as expected')

        # confirm we can also call the function with daily data
        computed_spi = indices.spi(self.fixture_precips_mm_daily, 
                                   30,
                                   indices.Distribution.gamma,
                                   self.fixture_data_year_start_daily, 
                                   self.fixture_calibration_year_start_daily, 
                                   self.fixture_calibration_year_end_daily, 
                                   'daily')
                                
        # invalid periodicity argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spi,
                                 self.fixture_precips_mm_monthly.flatten(), 
                                 6,
                                 indices.Distribution.gamma,
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_data_year_end_monthly, 
                                 'unsupported_value')
        
        # input array argument that's neither 1-D nor 2-D should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spi,
                                 np.array(np.zeros((4, 4, 8))), 
                                 6,
                                 indices.Distribution.gamma,
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_data_year_end_monthly, 
                                 'daily')
        
        # compute SPI/Pearson at 6-month scale
        computed_spi = indices.spi(self.fixture_precips_mm_monthly.flatten(), 
                                   6, 
                                   indices.Distribution.pearson_type3,
                                   self.fixture_data_year_start_monthly, 
                                   self.fixture_calibration_year_start_monthly, 
                                   self.fixture_calibration_year_end_monthly,
                                   'monthly')
        
        # confirm we can compute from daily values without raising an error                                 
        indices.spi(self.fixture_precips_mm_daily.flatten(), 
                    60, 
                    indices.Distribution.pearson_type3,
                    self.fixture_data_year_start_daily, 
                    self.fixture_calibration_year_start_daily, 
                    self.fixture_calibration_year_end_daily,
                    'daily')

        # confirm SPI/Pearson is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_6_month_pearson3, 
                                   atol=0.01, 
                                   err_msg='SPI/Pearson values for 6-month scale not computed as expected')

        # invalid periodicity argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spi,
                                 self.fixture_precips_mm_monthly.flatten(), 
                                 6,
                                 indices.Distribution.pearson_type3,
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_calibration_year_start_monthly, 
                                 self.fixture_calibration_year_end_monthly,
                                 'unsupported_value')
        
    #----------------------------------------------------------------------------------------
    def test_spei(self):
        
        # compute SPEI/gamma at 6-month scale
        computed_spei = indices.spei(6,
                                     indices.Distribution.gamma,
                                     'monthly', 
                                     data_start_year=self.fixture_data_year_start_monthly,
                                     calibration_year_initial=self.fixture_data_year_start_monthly,
                                     calibration_year_final=self.fixture_data_year_end_monthly,
                                     precips_mm=self.fixture_precips_mm_monthly, 
                                     pet_mm=None, 
                                     temps_celsius=self.fixture_temps_celsius, 
                                     latitude_degrees=self.fixture_latitude_degrees)

        # confirm SPEI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spei, 
                                   self.fixture_spei_6_month_gamma, 
                                   atol=0.01,
                                   err_msg='SPEI/Gamma values for 6-month scale not computed as expected')
        
        # compute SPEI/Pearson at 6-month scale
        computed_spei = indices.spei(6,
                                     indices.Distribution.pearson_type3,
                                     'monthly', 
                                     data_start_year=self.fixture_data_year_start_monthly,
                                     calibration_year_initial=self.fixture_data_year_start_monthly,
                                     calibration_year_final=self.fixture_data_year_end_monthly,
                                     precips_mm=self.fixture_precips_mm_monthly, 
                                     pet_mm=None, 
                                     temps_celsius=self.fixture_temps_celsius, 
                                     latitude_degrees=self.fixture_latitude_degrees)

        # confirm SPEI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spei, 
                                   self.fixture_spei_6_month_pearson3, 
                                   atol=0.01,
                                   err_msg='SPEI/Pearson values for 6-month scale not computed as expected')
        
        # invalid periodicity argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spei, 
                                 6,
                                 indices.Distribution.pearson_type3,
                                 'unsupported_value', 
                                 data_start_year=self.fixture_data_year_start_monthly,
                                 calibration_year_initial=self.fixture_data_year_start_monthly,
                                 calibration_year_final=self.fixture_data_year_end_monthly,
                                 precips_mm=self.fixture_precips_mm_monthly, 
                                 pet_mm=None, 
                                 temps_celsius=self.fixture_temps_celsius, 
                                 latitude_degrees=self.fixture_latitude_degrees)
        
        # missing temperature and PET arguments should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spei, 
                                 6,
                                 indices.Distribution.pearson_type3,
                                 'monthly', 
                                 data_start_year=self.fixture_data_year_start_monthly,
                                 calibration_year_initial=self.fixture_data_year_start_monthly,
                                 calibration_year_final=self.fixture_data_year_end_monthly,
                                 precips_mm=self.fixture_precips_mm_monthly, 
                                 pet_mm=None, 
                                 temps_celsius=None, 
                                 latitude_degrees=self.fixture_latitude_degrees)
        
        # having both temperature and PET input array arguments should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spei, 
                                 6,
                                 indices.Distribution.pearson_type3,
                                 'monthly', 
                                 data_start_year=self.fixture_data_year_start_monthly,
                                 calibration_year_initial=self.fixture_data_year_start_monthly,
                                 calibration_year_final=self.fixture_data_year_end_monthly,
                                 precips_mm=self.fixture_precips_mm_monthly, 
                                 pet_mm=self.fixture_pet_mm, 
                                 temps_celsius=self.fixture_temps_celsius, 
                                 latitude_degrees=self.fixture_latitude_degrees)
        
        # having temperature without corresponding latitude argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spei, 
                                 6,
                                 indices.Distribution.pearson_type3,
                                 'monthly', 
                                 data_start_year=self.fixture_data_year_start_monthly,
                                 calibration_year_initial=self.fixture_data_year_start_monthly,
                                 calibration_year_final=self.fixture_data_year_end_monthly,
                                 precips_mm=self.fixture_precips_mm_monthly, 
                                 pet_mm=self.fixture_pet_mm, 
                                 temps_celsius=self.fixture_temps_celsius, 
                                 latitude_degrees=None)
        
        # having both precipitation and PET input array arguments with incongruent dimensions should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spei, 
                                 6,
                                 indices.Distribution.pearson_type3,
                                 'monthly', 
                                 data_start_year=self.fixture_data_year_start_monthly,
                                 calibration_year_initial=self.fixture_data_year_start_monthly,
                                 calibration_year_final=self.fixture_data_year_end_monthly,
                                 precips_mm=self.fixture_precips_mm_monthly, 
                                 pet_mm=np.array((200, 200), dtype=float))
        
        # having temperature without corresponding latitude argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spei, 
                                 6,
                                 indices.Distribution.pearson_type3,
                                 'monthly', 
                                 data_start_year=self.fixture_data_year_start_monthly,
                                 calibration_year_initial=self.fixture_data_year_start_monthly,
                                 calibration_year_final=self.fixture_data_year_end_monthly,
                                 precips_mm=self.fixture_precips_mm_monthly, 
                                 temps_celsius=self.fixture_temps_celsius, 
                                 latitude_degrees=None)
        
        # providing PET with a corresponding latitude argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spei, 
                                 6,
                                 indices.Distribution.pearson_type3,
                                 'monthly', 
                                 data_start_year=self.fixture_data_year_start_monthly,
                                 calibration_year_initial=self.fixture_data_year_start_monthly,
                                 calibration_year_final=self.fixture_data_year_end_monthly,
                                 precips_mm=self.fixture_precips_mm_monthly, 
                                 pet_mm=self.fixture_pet_mm, 
                                 temps_celsius=self.fixture_temps_celsius, 
                                 latitude_degrees=40.0)

#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
    
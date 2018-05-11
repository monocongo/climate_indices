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
    def test_pet(self):
        
        # compute PET from the monthly temperatures, latitude, and initial years above
        computed_pet = indices.pet(self.fixture_temps_celsius,
                                   self.fixture_latitude_degrees, 
                                   self.fixture_data_year_start_monthly)
                                         
        # make sure PET is being computed as expected
        np.testing.assert_allclose(computed_pet, 
                                   self.fixture_pet_mm.flatten(),
                                   atol=0.01,
                                   err_msg='PET values not computed as expected')
        
    #----------------------------------------------------------------------------------------
    def test_pnp(self):
                
        # compute PNP from the daily precipitation array
        computed_pnp_6month = indices.percentage_of_normal(self.fixture_precips_mm_monthly.flatten(),
                                                           6, 
                                                           self.fixture_data_year_start_monthly,
                                                           self.fixture_calibration_year_start_monthly, 
                                                           self.fixture_calibration_year_end_monthly, 
                                                           'monthly')

        # make sure PNP is being computed as expected
        np.testing.assert_allclose(self.fixture_pnp_6month,
                                   computed_pnp_6month, 
                                   atol=0.001,
                                   equal_nan=True,
                                   err_msg='PNP values not computed as expected')
                              
        # make sure we can compute PNP from the daily values without raising an error 
        indices.percentage_of_normal(self.fixture_precips_mm_daily.flatten(),
                                     30, 
                                     self.fixture_data_year_start_daily,
                                     self.fixture_calibration_year_start_daily, 
                                     self.fixture_calibration_year_end_daily, 
                                     'daily')
                
        # invalid time series type argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.percentage_of_normal,
                                 self.fixture_precips_mm_daily.flatten(),
                                 30, 
                                 self.fixture_data_year_start_daily,
                                 self.fixture_calibration_year_start_daily, 
                                 self.fixture_calibration_year_end_daily, 
                                 'unsupported_value')

    #----------------------------------------------------------------------------------------
    def test_spi_gamma(self):
        
        # compute SPI/gamma at 1-month scale
        computed_spi = indices.spi_gamma(self.fixture_precips_mm_monthly, 
                                         1,
                                         self.fixture_data_year_start_monthly, 
                                         self.fixture_data_year_start_monthly, 
                                         self.fixture_data_year_end_monthly, 
                                         'monthly')
                                         
        # make sure SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_1_month_gamma, 
                                   atol=0.001,
                                   err_msg='SPI/Gamma values for 1-month scale not computed as expected')
        
        computed_spi = indices.spi_gamma(self.fixture_precips_mm_monthly.flatten(), 
                                         6,
                                         self.fixture_data_year_start_monthly, 
                                         self.fixture_data_year_start_monthly, 
                                         self.fixture_data_year_end_monthly, 
                                         'monthly')
                                         
        # make sure SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_6_month_gamma, 
                                   atol=0.001,
                                   err_msg='SPI/Gamma values for 6-month scale not computed as expected')

        # make sure we can also call the function with daily data
        computed_spi = indices.spi_gamma(self.fixture_precips_mm_daily, 
                                         30,
                                         self.fixture_data_year_start_daily, 
                                         self.fixture_calibration_year_start_daily, 
                                         self.fixture_calibration_year_end_daily, 
                                         'daily')
                                
        # invalid time series type argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spi_gamma, 
                                 self.fixture_precips_mm_monthly.flatten(), 
                                 6,
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_data_year_end_monthly, 
                                 'unsupported_value')
        
        # input array argument that's neither 1-D nor 2-D should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spi_gamma, 
                                 np.array(np.zeros((4, 4, 8))), 
                                 6,
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_data_year_end_monthly, 
                                 'daily')
        
    #----------------------------------------------------------------------------------------
    def test_spi_pearson(self):
        
        # compute SPI/Pearson at 6-month scale
        computed_spi = indices.spi_pearson(self.fixture_precips_mm_monthly.flatten(), 
                                           6, 
                                           self.fixture_data_year_start_monthly, 
                                           self.fixture_calibration_year_start_monthly, 
                                           self.fixture_calibration_year_end_monthly,
                                           'monthly')
                                         
        # make sure SPI/Pearson is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_6_month_pearson3, 
                                   atol=0.01, 
                                   err_msg='SPI/Pearson values for 6-month scale not computed as expected')

        # invalid time series type argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spi_pearson, 
                                 self.fixture_precips_mm_monthly.flatten(), 
                                 6,
                                 self.fixture_data_year_start_monthly, 
                                 self.fixture_calibration_year_start_monthly, 
                                 self.fixture_calibration_year_end_monthly,
                                 'unsupported_value')
        
    #----------------------------------------------------------------------------------------
    def test_spei_gamma(self):
        
        # compute SPEI/gamma at 6-month scale
        computed_spei = indices.spei_gamma(6,
                                           'monthly', 
                                           data_start_year=self.fixture_data_year_start_monthly,
                                           calibration_year_initial=self.fixture_data_year_start_monthly,
                                           calibration_year_final=self.fixture_data_year_end_monthly,
                                           precips_mm=self.fixture_precips_mm_monthly, 
                                           pet_mm=None, 
                                           temps_celsius=self.fixture_temps_celsius, 
                                           latitude_degrees=self.fixture_latitude_degrees)

        # make sure SPEI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spei, 
                                   self.fixture_spei_6_month_gamma, 
                                   atol=0.01,
                                   err_msg='SPEI/Gamma values for 6-month scale not computed as expected')
        
        # invalid time series type argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spei_gamma, 
                                 6,
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
                                 indices.spei_gamma, 
                                 6,
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
                                 indices.spei_gamma, 
                                 6,
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
                                 indices.spei_gamma, 
                                 6,
                                 'monthly', 
                                 data_start_year=self.fixture_data_year_start_monthly,
                                 calibration_year_initial=self.fixture_data_year_start_monthly,
                                 calibration_year_final=self.fixture_data_year_end_monthly,
                                 precips_mm=self.fixture_precips_mm_monthly, 
                                 pet_mm=self.fixture_pet_mm, 
                                 temps_celsius=self.fixture_temps_celsius, 
                                 latitude_degrees=None)
        
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
    
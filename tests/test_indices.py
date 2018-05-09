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
                                   self.fixture_data_year_start)
                                         
        # make sure PET is being computed as expected
        np.testing.assert_allclose(computed_pet, 
                                   self.fixture_pet_mm.flatten(),
                                   atol=0.01,
                                   err_msg='PET values not computed as expected')
        
    #----------------------------------------------------------------------------------------
    def test_spi_gamma(self):
        
        # compute SPI/gamma at 1-month scale
        month_scale = 1
        computed_spi = indices.spi_gamma(self.fixture_precips_mm, 
                                         1,
                                         self.fixture_data_year_start, 
                                         self.fixture_data_year_start, 
                                         self.fixture_data_year_end, 
                                         'monthly')
                                         
        # make sure SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_1_month_gamma, 
                                   atol=0.01,
                                   err_msg='SPI/Gamma values for 1-month scale not computed as expected'.format(month_scale))
        
        computed_spi = indices.spi_gamma(self.fixture_precips_mm.flatten(), 
                                         6,
                                         self.fixture_data_year_start, 
                                         self.fixture_data_year_start, 
                                         self.fixture_data_year_end, 
                                         'monthly')
                                         
        # make sure SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_6_month_gamma, 
                                   atol=0.01,
                                   err_msg='SPI/Gamma values for 6-month scale not computed as expected'.format(month_scale))

        # invalid time series type argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spi_gamma, self.fixture_precips_mm.flatten(), 
                                 6,
                                 self.fixture_data_year_start, 
                                 self.fixture_data_year_start, 
                                 self.fixture_data_year_end, 
                                 'unsupported_value')
        
    #----------------------------------------------------------------------------------------
    def test_spi_pearson(self):
        
        # compute SPI/Pearson at 6-month scale
        computed_spi = indices.spi_pearson(self.fixture_precips_mm.flatten(), 
                                           6, 
                                           self.fixture_data_year_start, 
                                           self.fixture_calibration_year_start, 
                                           self.fixture_calibration_year_end,
                                           'monthly')
                                         
        # make sure SPI/Pearson is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_6_month_pearson3, 
                                   atol=0.01, 
                                   err_msg='SPI/Pearson values for 6-month scale not computed as expected')

        # invalid time series type argument should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spi_pearson, 
                                 self.fixture_precips_mm.flatten(), 
                                 6,
                                 self.fixture_data_year_start, 
                                 self.fixture_calibration_year_start, 
                                 self.fixture_calibration_year_end,
                                 'unsupported_value')
        
    #----------------------------------------------------------------------------------------
    def test_spei_gamma(self):
        
        # compute SPEI/gamma at 6-month scale
        computed_spei = indices.spei_gamma(6,
                                           'monthly', 
                                           data_start_year=self.fixture_data_year_start,
                                           calibration_year_initial=self.fixture_data_year_start,
                                           calibration_year_final=self.fixture_data_year_end,
                                           precips_mm=self.fixture_precips_mm, 
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
                                 data_start_year=self.fixture_data_year_start,
                                 calibration_year_initial=self.fixture_data_year_start,
                                 calibration_year_final=self.fixture_data_year_end,
                                 precips_mm=self.fixture_precips_mm, 
                                 pet_mm=None, 
                                 temps_celsius=self.fixture_temps_celsius, 
                                 latitude_degrees=self.fixture_latitude_degrees)
        
        # missing temperature and PET arguments should raise a ValueError
        np.testing.assert_raises(ValueError, 
                                 indices.spei_gamma, 
                                 6,
                                 'monthly', 
                                 data_start_year=self.fixture_data_year_start,
                                 calibration_year_initial=self.fixture_data_year_start,
                                 calibration_year_final=self.fixture_data_year_end,
                                 precips_mm=self.fixture_precips_mm, 
                                 pet_mm=None, 
                                 temps_celsius=None, 
                                 latitude_degrees=self.fixture_latitude_degrees)
        
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
    
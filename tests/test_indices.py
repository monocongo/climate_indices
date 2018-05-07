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
                                   self.fixture_data_start_year)
                                         
        # make sure PET is being computed as expected
        np.testing.assert_allclose(computed_pet, 
                                   self.fixture_pet_mm.flatten(),
                                   atol=0.01,
                                   err_msg='PET values not computed as expected')
        
    #----------------------------------------------------------------------------------------
    def test_spi_gamma_1month(self):
        
        # compute SPI/gamma at 1-month scale
        month_scale = 1
        computed_spi = indices.spi_gamma(self.fixture_precips_mm, 
                                         month_scale,
                                         self.fixture_data_start_year, 
                                         self.fixture_data_start_year, 
                                         self.fixture_data_end_year, 
                                         'monthly')
                                         
        # make sure SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_1_month_gamma, 
                                   atol=0.01,
                                   err_msg='SPI/Gamma values for {0}-month scale not computed as expected'.format(month_scale))
        
    #----------------------------------------------------------------------------------------
    def test_spi_gamma_6month(self):
        
        # compute SPI/gamma at 6-month scale
        month_scale = 6
        computed_spi = indices.spi_gamma(self.fixture_precips_mm.flatten(), 
                                         month_scale,
                                         self.fixture_data_start_year, 
                                         self.fixture_data_start_year, 
                                         self.fixture_data_end_year, 
                                         'monthly')
                                         
        # make sure SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_6_month_gamma, 
                                   atol=0.01,
                                   err_msg='SPI/Gamma values for {0}-month scale not computed as expected'.format(month_scale))

    #----------------------------------------------------------------------------------------
    def test_spi_pearson_6month(self):
        
        # compute SPI/Pearson at 6-month scale
        month_scale = 6
        computed_spi = indices.spi_pearson(self.fixture_precips_mm.flatten(), 
                                           month_scale, 
                                           1895, 
                                           1981, 
                                           2010,
                                           'monthly')
                                         
        # make sure SPI/Pearson is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_6_month_pearson3, 
                                   atol=0.01,
                                   err_msg='SPI/Pearson values for {0}-month scale not computed as expected'.format(month_scale))

    #----------------------------------------------------------------------------------------
    def test_spei_pearson_6month(self):
         
        # compute SPEI/Pearson at 6-month scale, using precipitation and temperatures as input 
        month_scale = 6
        computed_spei = indices.spei_pearson(month_scale,
                                             'monthly',
                                             data_start_year=1895,
                                             calibration_year_initial=1981,
                                             calibration_year_final=2010,
                                             precips_mm=self.fixture_precips_mm, 
                                             temps_celsius=self.fixture_temps_celsius,
                                             latitude_degrees=self.fixture_latitude_degrees)
        
        # make sure SPEI/Pearson is being computed as expected
        np.testing.assert_allclose(computed_spei, 
                                   self.fixture_spei_6_month_pearson3, 
                                   atol=0.01,
                                   err_msg='SPEI/Pearson values for {0}-month scale not computed as expected'.format(month_scale))
        
    #----------------------------------------------------------------------------------------
    def test_spei_gamma_6month(self):
        
        # compute SPEI/gamma at 6-month scale
        months_scale = 6
        computed_spei = indices.spei_gamma(months_scale,
                                           'monthly', 
                                           data_start_year=self.fixture_data_start_year,
                                           calibration_year_initial=self.fixture_data_start_year,
                                           calibration_year_final=self.fixture_data_end_year,
                                           precips_mm=self.fixture_precips_mm, 
                                           pet_mm=None, 
                                           temps_celsius=self.fixture_temps_celsius, 
                                           latitude_degrees=self.fixture_latitude_degrees)

        # make sure SPEI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spei, 
                                   self.fixture_spei_6_month_gamma, 
                                   atol=0.01,
                                   err_msg='SPEI/Gamma values for {0}-month scale not computed as expected'.format(months_scale))
        
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
    
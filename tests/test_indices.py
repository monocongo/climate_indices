import fixtures
import indices
import logging
import math
import numpy as np
import unittest

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
                                   self.fixture_initial_data_year)
                                         
        # make sure PET is being computed as expected
        np.testing.assert_allclose(computed_pet, 
                                   self.fixture_pet_mm,
                                   atol=0.01,
                                   err_msg='PET values not computed as expected')
        
    #----------------------------------------------------------------------------------------
    def test_spi_gamma_1month(self):
        
        # compute SPI/gamma at 1-month scale
        month_scale = 1
        computed_spi = indices.spi_gamma(self.fixture_precips_mm, month_scale)
                                         
        # make sure SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_1_month_gamma, 
                                   atol=0.01,
                                   err_msg='SPI/Gamma values for {0}-month scale not computed as expected'.format(month_scale))
        
    #----------------------------------------------------------------------------------------
    def test_spi_gamma_6month(self):
        
        # compute SPI/gamma at 6-month scale
        month_scale = 6
        computed_spi = indices.spi_gamma(self.fixture_precips_mm, month_scale)
                                         
        # make sure SPI/gamma is being computed as expected
        np.testing.assert_allclose(computed_spi, 
                                   self.fixture_spi_6_month_gamma, 
                                   atol=0.01,
                                   err_msg='SPI/Gamma values for {0}-month scale not computed as expected'.format(month_scale))

    #----------------------------------------------------------------------------------------
    def test_spi_pearson_6month(self):
        
        # compute SPI/Pearson at 6-month scale
        month_scale = 6
        computed_spi = indices.spi_pearson(self.fixture_precips_mm, month_scale, 1895, 1981, 2010)
                                         
        for i in range(computed_spi.size):
            if not math.isclose(computed_spi[i], self.fixture_spi_6_month_pearson3[i], abs_tol=0.01) and \
                not math.isnan(computed_spi[i]) and \
                not math.isnan(self.fixture_spi_6_month_pearson3[i]): 
                
                print('Month: {0}\n\tExpected:  {1}\n\tComputed:  {2}'.format(i, self.fixture_spi_6_month_pearson3[i], computed_spi[i]))

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
                                             1895,
                                             self.fixture_precips_mm, 
                                             temps_celsius=self.fixture_temps_celsius,
                                             latitude_degrees=self.fixture_latitude_degrees)
        
        #DEBUG ONLY -- REMOVE                                  
        for i in range(computed_spei.size):
            if not math.isnan(computed_spei[i]) and \
                not math.isnan(self.fixture_spei_6_month_pearson3[i]) and \
                not math.isclose(computed_spei[i], self.fixture_spei_6_month_pearson3[i], abs_tol=0.01): 
                
                print('Month: {0}\n\tExpected:  {1}\n\tComputed:  {2}'.format(i, self.fixture_spei_6_month_pearson3[i], computed_spei[i]))

        # make sure SPI/Gamma is being computed as expected
        np.testing.assert_allclose(computed_spei, 
                                   self.fixture_spei_6_month_pearson3, 
                                   atol=0.01,
                                   err_msg='SPEI/Pearson values for {0}-month scale not computed as expected'.format(month_scale))
        
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
    
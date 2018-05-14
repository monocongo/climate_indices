import logging
import numpy as np
import unittest

from tests import fixtures
from climate_indices import palmer

#-----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)

#-----------------------------------------------------------------------------------------------------------------------
class PalmerTestCase(fixtures.FixturesTestCase):
    '''
    Tests for `palmer.py`.
    '''

    #----------------------------------------------------------------------------------------
    def test_pdsi(self):
        
        pdsi, phdi, pmdi, zindex = palmer.pdsi(self.fixture_precips_mm_monthly, 
                           self.fixture_pet_mm,
                           self.fixture_awc_inches, 
                           self.fixture_data_year_start_monthly, 
                           self.fixture_calibration_year_start_monthly, 
                           self.fixture_calibration_year_end_monthly)
        
        np.testing.assert_allclose(pdsi, 
                                   self.fixture_palmer_pdsi_monthly, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='PDSI not computed as expected from monthly inputs')
    
        np.testing.assert_allclose(phdi, 
                                   self.fixture_palmer_phdi_monthly, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='PHDI not computed as expected from monthly inputs')
    
        np.testing.assert_allclose(pmdi, 
                                   self.fixture_palmer_pmdi_monthly, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='PMDI not computed as expected from monthly inputs')
    
        np.testing.assert_allclose(zindex, 
                                   self.fixture_palmer_zindex_monthly, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='Z-Index not computed as expected from monthly inputs')
    
    #----------------------------------------------------------------------------------------
    def test_pdsi_from_zindex(self):
        
        pdsi, phdi, pmdi = palmer._pdsi_from_zindex(self.fixture_palmer_zindex_monthly)
        
        np.testing.assert_allclose(pdsi, 
                                   self.fixture_palmer_pdsi_monthly, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='PDSI not computed as expected from monthly Z-Index fixture')
    
        np.testing.assert_allclose(phdi, 
                                   self.fixture_palmer_phdi_monthly, 
                                   atol=0.01, 
                                   equal_nan=True, 
                                   err_msg='PHDI not computed as expected from monthly Z-Index fixture')
    
        np.testing.assert_allclose(pmdi, 
                                   self.fixture_palmer_pmdi_monthly, 
                                   atol=0.01, 
                                   equal_nan=True, 
                                   err_msg='PMDI not computed as expected from monthly Z-Index fixture')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_z_index(self):
        '''
        Test for the palmer._z_index() function
        '''

        # call the _z_index() function                                                                                        
        Z = palmer._z_index(self.fixture_palmer_precip_AL01,
                            self.fixture_palmer_pet_AL01,
                            self.fixture_palmer_et_AL01,
                            self.fixture_palmer_pr_AL01,
                            self.fixture_palmer_r_AL01,
                            self.fixture_palmer_ro_AL01,
                            self.fixture_palmer_pro_AL01,
                            self.fixture_palmer_l_AL01,
                            self.fixture_palmer_pl_AL01,
                            self.fixture_palmer_data_begin_year,
                            self.fixture_palmer_calibration_begin_year,
                            self.fixture_palmer_calibration_end_year)
        
        # compare against expected results
        np.testing.assert_allclose(Z, 
                                   self.fixture_palmer_zindex_AL01, 
                                   atol=0.01,
                                   err_msg='Not computing the Z-Index as expected')        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_climatic_characteristic(self):
        '''
        Test for the palmer._climatic_characteristic() function
        '''
        
        # call the _cafec_coefficients() function                                                                                        
        palmer_K = palmer._climatic_characteristic(self.fixture_palmer_alpha_AL01,
                                                   self.fixture_palmer_beta_AL01,
                                                   self.fixture_palmer_gamma_AL01,
                                                   self.fixture_palmer_delta_AL01,
                                                   self.fixture_palmer_precip_AL01,
                                                   self.fixture_palmer_et_AL01,
                                                   self.fixture_palmer_pet_AL01,
                                                   self.fixture_palmer_r_AL01,
                                                   self.fixture_palmer_pr_AL01,
                                                   self.fixture_palmer_ro_AL01,
                                                   self.fixture_palmer_pro_AL01,
                                                   self.fixture_palmer_l_AL01,
                                                   self.fixture_palmer_pl_AL01,
                                                   self.fixture_palmer_data_begin_year,
                                                   self.fixture_palmer_calibration_begin_year,
                                                   self.fixture_palmer_calibration_end_year)
                    
        # compare against expected results
        np.testing.assert_allclose(palmer_K, 
                                   self.fixture_palmer_K_AL01, 
                                   atol=0.01,
                                   err_msg='Not computing the K as expected')        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_cafec_compute_X(self):
        '''
        Test for the palmer._compute_X() function
        '''
        
        # simulate computation of X at an initial step (with all zeros for intermediate value arrays)
        Z = self.fixture_palmer_zindex_monthly
        k = 0
        PPe = np.zeros(Z.shape)
        X1 = 0.0
        X2 = 0.0
        PX1 = np.zeros(Z.shape)
        PX2 = np.zeros(Z.shape)
        PX3 = np.zeros(Z.shape)
        X = np.zeros(Z.shape)
        BT = np.zeros(Z.shape)        
        PX1, PX2, PX3, X, BT = palmer._compute_X(Z, k, PPe, X1, X2, PX1, PX2, PX3, X, BT)
        self.assertEqual(PX1[0], 0.0, 'PX1 value not computed as expected at initial step')
        self.assertEqual(PX2[0], -0.34, 'PX2 value not computed as expected at initial step')
        self.assertEqual(PX3[0], 0.0, 'PX3 value not computed as expected at initial step')
        self.assertEqual(X[0], -0.34, 'X value not computed as expected at initial step')
        self.assertEqual(BT[0], 2, 'Backtrack value not computed as expected at initial step')
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_scpdsi(self):
        '''
        Test for the palmer.scpdsi() function
        '''
        
        scpdsi, pdsi, phdi, pmdi, zindex = palmer.scpdsi(self.fixture_precips_mm_monthly, 
                                                 self.fixture_pet_mm,
                                                 self.fixture_awc_inches, 
                                                 self.fixture_data_year_start_monthly, 
                                                 self.fixture_calibration_year_start_monthly, 
                                                 self.fixture_calibration_year_end_monthly)
        
        np.testing.assert_allclose(scpdsi, 
                                   self.fixture_palmer_scpdsi_monthly, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='PDSI not computed as expected from monthly inputs')
    
#         np.testing.assert_allclose(pdsi, 
#                                    self.fixture_palmer_pdsi_monthly, 
#                                    atol=0.001, 
#                                    equal_nan=True, 
#                                    err_msg='PDSI not computed as expected from monthly inputs')
    
        np.testing.assert_allclose(phdi, 
                                   self.fixture_palmer_scphdi_monthly, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='PHDI not computed as expected from monthly inputs')
    
        np.testing.assert_allclose(pmdi, 
                                   self.fixture_palmer_scpmdi_monthly, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='PMDI not computed as expected from monthly inputs')
    
        np.testing.assert_allclose(zindex, 
                                   self.fixture_palmer_sczindex_monthly, 
                                   atol=0.001, 
                                   equal_nan=True, 
                                   err_msg='Z-Index not computed as expected from monthly inputs')
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_cafec_coeff_ufunc(self):
        '''
        Test for the palmer._dry_spell_abatement() function
        '''
        
        self.assertEqual(palmer._cafec_coeff_ufunc(0, 0), 1)
        self.assertEqual(palmer._cafec_coeff_ufunc(5, 0), 0)
        self.assertEqual(palmer._cafec_coeff_ufunc(5, 10), 0.5)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_cafec_coefficients(self):
        '''
        Test for the palmer._cafec_coefficients() function
        '''
        
        # call the _cafec_coefficients() function                                                                                        
        alpha, beta, gamma, delta = palmer._cafec_coefficients(self.fixture_palmer_precip_AL01,
                                                               self.fixture_palmer_pet_AL01,
                                                               self.fixture_palmer_et_AL01,
                                                               self.fixture_palmer_pr_AL01,
                                                               self.fixture_palmer_r_AL01,
                                                               self.fixture_palmer_ro_AL01,
                                                               self.fixture_palmer_pro_AL01,
                                                               self.fixture_palmer_l_AL01,
                                                               self.fixture_palmer_pl_AL01,
                                                               self.fixture_palmer_data_begin_year,
                                                               self.fixture_palmer_calibration_begin_year,
                                                               self.fixture_palmer_calibration_end_year)
        
        # verify that the function performed as expected
        arys = [['Alpha', alpha, self.fixture_palmer_alpha_AL01], 
                ['Beta', beta, self.fixture_palmer_beta_AL01],
                ['Gamma', gamma, self.fixture_palmer_gamma_AL01],
                ['Delta', delta, self.fixture_palmer_delta_AL01]]
        
        for lst in arys:
            
            name = lst[0]
            actual = lst[1]
            expected = lst[2]
            
            # compare against expected results
            np.testing.assert_allclose(actual, 
                                       expected, 
                                       atol=0.01,
                                       err_msg='Not computing the {0} as expected'.format(name))        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_phdi_select_ufunc(self):
        '''
        Test for the palmer._phdi_select_ufunc() function
        '''
        
        self.assertEqual(palmer._phdi_select_ufunc(0, 5), 5)
        self.assertEqual(palmer._phdi_select_ufunc(8, 5), 8)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_water_balance(self):
        '''
        Test for the palmer._water_balance() function
        '''
        
        # call the water balance accounting function, providing AL-01 climate division input data
        palmer_ET, palmer_PR, palmer_R, palmer_RO, palmer_PRO, palmer_L, palmer_PL = \
                    palmer._water_balance(self.fixture_palmer_awc_AL01 + 1.0, self.fixture_palmer_pet_AL01, self.fixture_palmer_precip_AL01)
                    
        arys = [['ET', palmer_ET, self.fixture_palmer_et_AL01], 
                ['PR', palmer_PR, self.fixture_palmer_pr_AL01],
                ['R', palmer_R, self.fixture_palmer_r_AL01],
                ['RO', palmer_RO, self.fixture_palmer_ro_AL01],
                ['PRO', palmer_PRO, self.fixture_palmer_pro_AL01],
                ['L', palmer_L, self.fixture_palmer_l_AL01],
                ['PL', palmer_PL, self.fixture_palmer_pl_AL01]]
                  
        # verify that the function performed as expected
        for lst in arys:
            name = lst[0]
            actual = lst[1]
            expected = lst[2]
            
            np.testing.assert_allclose(actual, 
                                       expected, 
                                       atol=0.01,
                                       err_msg='Not computing the {0} as expected'.format(name))        

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()

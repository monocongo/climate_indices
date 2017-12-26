import fixtures
import logging
import numpy as np
import palmer
import unittest

#-----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)

#-----------------------------------------------------------------------------------------------------------------------
class PalmerTestCase(fixtures.FixturesTestCase):
    '''
    Tests for `palmer.py`.
    '''

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
            
            close_indices = np.isclose(actual, expected, atol=0.1, equal_nan=True)
            for i in range(actual.size):
                if not close_indices[i]:
                    print('Index:  {0}\n\tExpected:  {1}\n\tActual:  {2}'.format(i, expected[i], actual[i]))
                    
            np.testing.assert_allclose(actual, 
                                       expected, 
                                       atol=0.01,
                                       err_msg='Not computing the {0} as expected'.format(name))        

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()

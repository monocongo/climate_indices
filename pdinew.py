import logging
import math
import numpy as np
import utils
import warnings

# set up a basic, global logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------------------------------------------------
def _cafec_coefficients(P,
                        PET,
                        ET,
                        PR,
                        R,
                        RO,
                        PRO,
                        L,
                        PL,
                        SP,
                        data_start_year,
                        calibration_start_year,
                        calibration_end_year):
    '''
    This function calculates CAFEC coefficients used for computing Palmer's Z index using inputs from 
    the water balance function. Translated from Fortran pdinew.f
    
    :param P: 1-D numpy.ndarray of monthly precipitation observations, in inches, the number of array elements 
              (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param PET: 1-D numpy.ndarray of monthly potential evapotranspiration values, in inches, the number of array elements 
                (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param ET: 1-D numpy.ndarray of monthly evapotranspiration values, in inches, the number of array elements 
               (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param PR: 1-D numpy.ndarray of monthly potential recharge values, in inches, the number of array elements 
               (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param R: 1-D numpy.ndarray of monthly recharge values, in inches, the number of array elements 
              (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param RO: 1-D numpy.ndarray of monthly runoff values, in inches, the number of array elements 
               (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param PRO: 1-D numpy.ndarray of monthly potential runoff values, in inches, the number of array elements 
                (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param L: 1-D numpy.ndarray of monthly loss values, in inches, the number of array elements 
              (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param PL: 1-D numpy.ndarray of monthly potential loss values, in inches, the number of array elements 
              (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param SP: 1-D numpy.ndarray of monthly SP values, in inches, the number of array elements 
               (array size) should be a multiple of 12 (representing an ordinal number of full years)
    :param data_start_year: initial year of the input arrays, i.e. the first element of each of the input arrays 
                            is assumed to correspond to January of this initial year
    :param calibration_start_year: initial year of the calibration period, should be greater than or equal to the data_start_year
    :param calibration_end_year: final year of the calibration period
    :return 1-D numpy.ndarray of Z-Index values, with shape corresponding to the input arrays
    :rtype: numpy.ndarray of floats
    '''
    
    # the potential (PET, ET, PR, PL) and actual (R, RO, S, L, P) water balance arrays are reshaped as 2-D arrays  
    # (matrices) such that the rows of each matrix represent years and the columns represent calendar months
    PET = utils.reshape_to_years_months(PET)
    ET = utils.reshape_to_years_months(ET)
    PR = utils.reshape_to_years_months(PR)
    PL = utils.reshape_to_years_months(PL)
    R = utils.reshape_to_years_months(R)
    RO = utils.reshape_to_years_months(RO)
    PRO = utils.reshape_to_years_months(PRO)
    L = utils.reshape_to_years_months(L)
    P = utils.reshape_to_years_months(P)
    SP = utils.reshape_to_years_months(SP)
        
    # ALPHA, BETA, GAMMA, DELTA CALCULATIONS
    # A calibration period is used to calculate alpha, beta, gamma, and 
    # and delta, four coefficients dependent on the climate of the area being
    # examined. The NCDC and CPC use the calibration period January 1931
    # through December 1990 (cf. Karl, 1986; Journal of Climate and Applied 
    # Meteorology, Vol. 25, No. 1, January 1986).
    
    #!!!!!!!!!!!!!
    # TODO make sure calibration years range is valid, i.e. within actual data years range 
    
    # determine the array (year axis) indices for the calibration period
    total_data_years = int(P.shape[0] / 12)
    data_end_year = data_start_year + total_data_years - 1
    total_calibration_years = calibration_end_year - calibration_start_year + 1
    calibration_start_year_index = calibration_start_year - data_start_year
    calibration_end_year_index = calibration_end_year - data_start_year 
    
    # get calibration period arrays
    if (calibration_start_year > data_start_year) or (calibration_end_year < data_end_year):
        P_calibration = P[calibration_start_year_index:calibration_end_year_index + 1]
        ET_calibration = ET[calibration_start_year_index:calibration_end_year_index + 1]
        PET_calibration = PET[calibration_start_year_index:calibration_end_year_index + 1]
        R_calibration = R[calibration_start_year_index:calibration_end_year_index + 1]
        PR_calibration = PR[calibration_start_year_index:calibration_end_year_index + 1]
        L_calibration = L[calibration_start_year_index:calibration_end_year_index + 1]
        PL_calibration = PL[calibration_start_year_index:calibration_end_year_index + 1]
        RO_calibration = RO[calibration_start_year_index:calibration_end_year_index + 1]
        PRO_calibration = PRO[calibration_start_year_index:calibration_end_year_index + 1]
        SP_calibration = SP[calibration_start_year_index:calibration_end_year_index + 1]
    else:
        P_calibration = P
        ET_calibration = ET
        PET_calibration = PET
        R_calibration = R
        PR_calibration = PR
        L_calibration = L
        PL_calibration = PL
        RO_calibration = RO
        PRO_calibration = PRO
        SP_calibration = SP

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
#         # get averages for each calendar month (compute means over the year axis, giving an average for each calendar month over all years)
#         P_bar = np.nanmean(P_calibration, axis=0)
#         ET_bar = np.nanmean(ET_calibration, axis=0)
#         PET_bar = np.nanmean(PET_calibration, axis=0)
#         R_bar = np.nanmean(R_calibration, axis=0)
#         PR_bar = np.nanmean(PR_calibration, axis=0)
#         L_bar = np.nanmean(L_calibration, axis=0)
#         PL_bar = np.nanmean(PL_calibration, axis=0)
#         RO_bar = np.nanmean(RO_calibration, axis=0)
#         PRO_bar = np.nanmean(PRO_calibration, axis=0)
            
        # get sums for each calendar month (compute sums over the year axis, giving a sum for each calendar month over all years)
        P_sum = np.nansum(P_calibration, axis=0)
        ET_sum = np.nansum(ET_calibration, axis=0)
        PET_sum = np.nansum(PET_calibration, axis=0)
        R_sum = np.nansum(R_calibration, axis=0)
        PR_sum = np.nansum(PR_calibration, axis=0)
        L_sum = np.nansum(L_calibration, axis=0)
        PL_sum = np.nansum(PL_calibration, axis=0)
        RO_sum = np.nansum(RO_calibration, axis=0)
        SP_sum = np.nansum(SP_calibration, axis=0)
            
        # (calendar) monthly CAFEC coefficients
        alpha = np.empty((12,))
        beta = np.empty((12,))
        gamma = np.empty((12,))
        delta = np.empty((12,))
        t_ratio = np.empty((12,))
    
        # compute the alpha, beta, gamma, and delta coefficients for each calendar month
        for i in range(12):
            
            #     ALPHA CALCULATION 
            #     ----------------- 
            if PET_sum[i] != 0.0:   
                alpha[i] = ET_sum[i] / PET_sum[i]  
            else:  
                if ET_sum[i] == 0.0:  
                    alpha[i] = 1.0  
                else:  
                    alpha[i] = 0.0  
            
            #   
            #     BETA CALCULATION  
            #     ----------------  
            if PR_sum[i] != 0.0:  
                beta[i] = R_sum[i] / PR_sum[i] 
            else:  
                if R_sum[i] == 0.0:   
                    beta[i] = 1.0  
                else:  
                    beta[i] = 0.0  

            #   
            #     GAMMA CALCULATION 
            #     ----------------- 
            if SP_sum[i] != 0.0:  
                gamma[i] = RO_sum[i] / SP_sum[i]  
            else:  
                if RO_sum[i] == 0.0:  
                    gamma[i] = 1.   
                else:  
                    gamma[i] = 0.0  

            #   
            #     DELTA CALCULATION 
            #     ----------------- 
            if PL_sum[i] != 0.0:  
                delta[i] = L_sum[i] / PL_sum[i]  
            else:
                delta[i] = 0.0  

            # 'T' ratio of average moisture demand to the average moisture supply in the month
            t_ratio[i] = (PET_sum[i] + R_sum[i] + RO_sum[i]) / (P_sum[i] + L_sum[i])

    return alpha, beta, delta, gamma, t_ratio

#-----------------------------------------------------------------------------------------------------------------------
def _climatic_characteristic(alpha,
                             beta,
                             gamma,
                             delta,
                             Pdat,
                             PEdat,
                             PRdat,
                             SPdat,
                             PLdat,
                             t_ratio,
                             begin_year,
                             calibration_begin_year,
                             calibration_end_year):
    
    SABSD = np.zeros((12,))
    
    number_calibration_years = calibration_end_year - calibration_begin_year + 1
    
    # loop over the calibration years, get the sum of the absolute values of the moisture departure SABSD for each month
    for j in range(calibration_begin_year - begin_year, calibration_end_year - begin_year + 1):
        for m in range(12):
            
            #-----------------------------------------------------------------------
            #     REREAD MONTHLY PARAMETERS FOR CALCULATION OF 
            #     THE 'K' MONTHLY WEIGHTING FACTORS USED IN Z-INDEX CALCULATION 
            #-----------------------------------------------------------------------
            PHAT = (alpha[m] * PEdat[j, m]) + (beta[m] * PRdat[j, m]) + (gamma[m] * SPdat[j, m]) - (delta[m] * PLdat[j, m])  
            D = Pdat[j, m] - PHAT   
            SABSD[m] = SABSD[m] + abs(D) 

    # get the first approximation of K (weighting factor)
    SWTD = 0.0
    AKHAT = np.empty((12,))
    for m in range(12):
        DBAR = SABSD[m] / number_calibration_years 
        AKHAT[m] = 1.5 * math.log10((t_ratio[m] + 2.8) / DBAR) + 0.5
        SWTD = SWTD + (DBAR * AKHAT[m])  

    AK = np.empty((12,))
    for m in range(12):
        AK[m] = 17.67 * AKHAT[m] / SWTD 
 
    return AK

#-----------------------------------------------------------------------------------------------------------------------
def _zindex_pdsi(P,
                 PE,
                 PR,
                 SP,
                 PL,
                 PPR,
                 alpha,
                 beta,
                 gamma,
                 delta,
                 AK):

    P = utils.reshape_to_years_months(P)
    PPR = utils.reshape_to_years_months(PPR)
#     eff = np.full(P.shape, np.NaN)
    CP = np.full(P.shape, np.NaN)
    Z = np.full(P.shape, np.NaN)
    
    PDSI = np.full(P.shape, np.NaN)
    PHDI = np.full(P.shape, np.NaN)
    WPLM = np.full(P.shape, np.NaN)
    
    nbegyr = 1895
    nendyr = 2017

    PV = 0.0
    V   = 0.0 
    PRO = 0.0 
    X1  = 0.0 
    X2  = 0.0 
    X3  = 0.0 
    K8  = 0
    k8max = 0

    indexj = np.empty(P.shape)
    indexm = np.empty(P.shape)
    PX1 = np.zeros(P.shape)
    PX2 = np.zeros(P.shape)
    PX3 = np.zeros(P.shape)
    SX1 = np.zeros(P.shape)
    SX2 = np.zeros(P.shape)
    SX3 = np.zeros(P.shape)
    SX = np.full(P.shape, np.NaN)
    X = np.full(P.shape, np.NaN)
    
    for j in range(P.shape[0]):    
               
        for m in range(12):

            indexj[K8] = j
            indexm[K8] = m

            #-----------------------------------------------------------------------
            #     LOOP FROM 160 TO 230 REREADS data FOR CALCULATION OF   
            #     THE Z-INDEX (MOISTURE ANOMALY) AND PDSI (VARIABLE X). 
            #     THE FINAL OUTPUTS ARE THE VARIABLES PX3, X, AND Z  WRITTEN
            #     TO FILE 11.   
            #-----------------------------------------------------------------------
            ZE = 0.0 
            UD = 0.0 
            UW = 0.0 
            CET = alpha[m] * PE[j, m]
            CR = beta[m] * PR[j, m]
            CRO = gamma[m] * SP[j, m]
            CL = delta[m] * PL[j, m]
            CP[j, m]  = CET + CR + CRO - CL 
            CD = P[j, m] - CP[j, m]  
            Z[j, m] = AK[m] * CD 
            if PRO == 100.0 or PRO == 0.0:  
            #     ------------------------------------ NO ABATEMENT UNDERWAY
            #                                          WET OR DROUGHT WILL END IF   
            #                                             -0.5 =< X3 =< 0.5   
                if abs(X3) <= 0.5:
                #         ---------------------------------- END OF DROUGHT OR WET  
                    PV = 0.0 
                    PPR[j, m] = 0.0 
                    PX3[j, m] = 0.0 
                    #             ------------ BUT CHECK FOR NEW WET OR DROUGHT START FIRST
                    # GOTO 200 in pdinew.f
                    # compare to 
                    # PX1, PX2, PX3, X, BT = Main(Z, k, PV, PPe, X1, X2, PX1, PX2, PX3, X, BT)
                    # in pdsi_from_zindex()
                    X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8 = _compute_X(PX1,
                                                                          PX2,
                                                                          PX3,
                                                                          X,
                                                                          X1,
                                                                          X2,
                                                                          Z,
                                                                          j,
                                                                          m,
                                                                          K8,
                                                                          PPR,
                                                                          PDSI, 
                                                                          PHDI, 
                                                                          WPLM, 
                                                                          nendyr, 
                                                                          nbegyr, 
                                                                          SX1, 
                                                                          SX2, 
                                                                          SX3, 
                                                                          SX, 
                                                                          indexj, 
                                                                          indexm,
                                                                          PV)
                     
                elif X3 > 0.5:   
                    #         ----------------------- WE ARE IN A WET SPELL 
                    if Z[j, m] >= 0.15:   
                        #              ------------------ THE WET SPELL INTENSIFIES 
                        #GO TO 210 in pdinew.f
                        # compare to 
                        # PV, PX1, PX2, PX3, PPe, X, BT = Between0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)
                        # in pdsi_from_zindex()
                        X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8 = _between_0s(K8, 
                                                                               PPR, 
                                                                               PX1,
                                                                               PX2, 
                                                                               PX3, 
                                                                               X1,
                                                                               X2, 
                                                                               X3, 
                                                                               X, 
                                                                               PDSI, 
                                                                               PHDI, 
                                                                               WPLM, 
                                                                               j, 
                                                                               m, 
                                                                               nendyr, 
                                                                               nbegyr, 
                                                                               SX1, 
                                                                               SX2, 
                                                                               SX3, 
                                                                               SX, 
                                                                               indexj, 
                                                                               indexm,
                                                                               Z)
                        
                    else:
                        #             ------------------ THE WET STARTS TO ABATE (AND MAY END)  
                        #GO TO 170 in pdinew.f
                        # compare to
                        # Ud, Ze, Q, PV, PPe, PX1, PX2, PX3, X, BT = Function_Ud(k, Ud, Z, Ze, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                        # in pdsi_from_zindex()
                        X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8 = _wet_spell_abatement(Z, 
                                                                                        V, 
                                                                                        K8, 
                                                                                        PPR,
                                                                                        PRO, 
                                                                                        PX1,
                                                                                        PX2, 
                                                                                        PX3, 
                                                                                        X, 
                                                                                        X1,
                                                                                        X2,
                                                                                        X3,
                                                                                        PDSI, 
                                                                                        PHDI, 
                                                                                        WPLM, 
                                                                                        j, 
                                                                                        m, 
                                                                                        nendyr, 
                                                                                        nbegyr, 
                                                                                        SX1, 
                                                                                        SX2, 
                                                                                        SX3, 
                                                                                        SX, 
                                                                                        indexj, 
                                                                                        indexm)                        

                elif X3 < -0.5:  
                    #         ------------------------- WE ARE IN A DROUGHT 
                    if Z[j, m] <= -0.15:  
                        #              -------------------- THE DROUGHT INTENSIFIES 
                        #GO TO 210
                        # compare to 
                        # PV, PX1, PX2, PX3, PPe, X, BT = Between0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)
                        # in pdsi_from_zindex()
                        X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8 = _between_0s(K8, 
                                                                               PPR, 
                                                                               PX1,
                                                                               PX2, 
                                                                               PX3, 
                                                                               X1,
                                                                               X2, 
                                                                               X3, 
                                                                               X, 
                                                                               PDSI, 
                                                                               PHDI, 
                                                                               WPLM, 
                                                                               j, 
                                                                               m, 
                                                                               nendyr, 
                                                                               nbegyr, 
                                                                               SX1, 
                                                                               SX2, 
                                                                               SX3, 
                                                                               SX, 
                                                                               indexj, 
                                                                               indexm,
                                                                               Z) 
                    else:
                        #             ------------------ THE DROUGHT STARTS TO ABATE (AND MAY END)  
                        #GO TO 180
                        # compare to 
                        # Uw, Ze, Q, PV, PPe, PX1, PX2, PX3, X, BT = Function_Uw(k, Uw, Z, Ze, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                        # in pdsi_from_zindex()
                        X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8 = _dry_spell_abatement(K8, 
                                                                                        PPR, 
                                                                                        PX1,
                                                                                        PX2, 
                                                                                        PX3, 
                                                                                        X, 
                                                                                        PDSI, 
                                                                                        PHDI, 
                                                                                        WPLM, 
                                                                                        j, 
                                                                                        m, 
                                                                                        nendyr, 
                                                                                        nbegyr, 
                                                                                        SX1, 
                                                                                        SX2, 
                                                                                        SX3, 
                                                                                        SX, 
                                                                                        indexj, 
                                                                                        indexm,
                                                                                        Z,
                                                                                        PV,
                                                                                        V)
                         
                else:
                    #     ------------------------------------------ABATEMENT IS UNDERWAY   
                    if X3 > 0.0:
                        
                        #         ----------------------- WE ARE IN A WET SPELL 
                        #GO TO 170 in pdinew.f
                        # compare to
                        # Ud, Ze, Q, PV, PPe, PX1, PX2, PX3, X, BT = Function_Ud(k, Ud, Z, Ze, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                        # in pdsi_from_zindex()
                        X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8 = _wet_spell_abatement(Z, 
                                                                                        V, 
                                                                                        K8, 
                                                                                        PPR,
                                                                                        PRO,
                                                                                        PX1,
                                                                                        PX2, 
                                                                                        PX3, 
                                                                                        X, 
                                                                                        X1,
                                                                                        X2,
                                                                                        X3, 
                                                                                        PDSI, 
                                                                                        PHDI, 
                                                                                        WPLM, 
                                                                                        j, 
                                                                                        m, 
                                                                                        nendyr, 
                                                                                        nbegyr, 
                                                                                        SX1, 
                                                                                        SX2, 
                                                                                        SX3, 
                                                                                        SX, 
                                                                                        indexj, 
                                                                                        indexm)   
                    
                    else:  # if X3 <= 0.0:
                        
                        #         ----------------------- WE ARE IN A DROUGHT   
                        #GO TO 180
                        # compare to
                        # Uw, Ze, Q, PV, PPe, PX1, PX2, PX3, X, BT = Function_Uw(k, Uw, Z, Ze, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                        # in pdsi_from_zindex()
                        X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8 = _dry_spell_abatement(K8, 
                                                                                        PPR, 
                                                                                        PX1,
                                                                                        PX2, 
                                                                                        PX3, 
                                                                                        X, 
                                                                                        PDSI, 
                                                                                        PHDI, 
                                                                                        WPLM, 
                                                                                        j, 
                                                                                        m, 
                                                                                        nendyr, 
                                                                                        nbegyr, 
                                                                                        SX1, 
                                                                                        SX2, 
                                                                                        SX3, 
                                                                                        SX, 
                                                                                        indexj, 
                                                                                        indexm,
                                                                                        Z,
                                                                                        PV,
                                                                                        V)

    return PDSI, PHDI, WPLM, Z

#-----------------------------------------------------------------------------------------------------------------------
# compare to Function_Ud()
# from line 170 in pdinew.f
# NOTE careful not to confuse the PRO being returned (probability) with PRO array of potential run off values
def _wet_spell_abatement(Z, 
                        V, 
                        K8, 
                        PPR,
                        PRO,
                        PX1,
                        PX2, 
                        PX3, 
                        X,
                        X1,
                        X2,
                        X3, 
                        PDSI, 
                        PHDI, 
                        WPLM, 
                        j, 
                        m, 
                        nendyr, 
                        nbegyr, 
                        SX1, 
                        SX2, 
                        SX3, 
                        SX, 
                        indexj, 
                        indexm):
    
    #-----------------------------------------------------------------------
    #      WET SPELL ABATEMENT IS POSSIBLE  
    #-----------------------------------------------------------------------
    UD = Z[j, m] - 0.15  
    PV = UD + min(V, 0.0) 
    if PV >= 0:
        #PV, PX1, PX2, PX3, PPe, X, BT = Between0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)
        X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8 = _between_0s(K8, 
                                                              PPR, 
                                                              PX1,
                                                              PX2, 
                                                              PX3, 
                                                              X1,
                                                              X2, 
                                                              X3, 
                                                              X, 
                                                              PDSI, 
                                                              PHDI, 
                                                              WPLM, 
                                                              j, 
                                                              m, 
                                                              nendyr, 
                                                              nbegyr, 
                                                              SX1, 
                                                              SX2, 
                                                              SX3, 
                                                              SX, 
                                                              indexj, 
                                                              indexm,
                                                              Z)
    else:
        #     ---------------------- DURING A WET SPELL, PV => 0 IMPLIES
        #                            PROB(END) HAS RETURNED TO 0
        ZE = -2.691 * X3 + 1.5
    
        #-----------------------------------------------------------------------
        #     PROB(END) = 100 * (V/Q)  WHERE:   
        #                 V = SUM OF MOISTURE EXCESS OR DEFICIT (UD OR UW)  
        #                 DURING CURRENT ABATEMENT PERIOD   
        #             Q = TOTAL MOISTURE ANOMALY REQUIRED TO END THE
        #                 CURRENT DROUGHT OR WET SPELL  
        #-----------------------------------------------------------------------
        if PRO == 100.0: 
            #     --------------------- DROUGHT OR WET CONTINUES, CALCULATE 
            #                           PROB(END) - VARIABLE ZE 
            Q = ZE
        else:  
            Q = ZE + V
    
        PPR[j, m] = (PV / Q) * 100.0 
        if PPR[j, m] >= 100.0:
             
              PPR[j, m] = 100.0
              PX3[j, m] = 0.0  
        else:
              
              PX3[j, m] = 0.897 * X3 + Z[j, m] / 3.0


        # in new version the X values and backtracking is again computed here, skipped in the NCEI Fortran

        
    return X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8

#-----------------------------------------------------------------------------------------------------------------------
# compare to Function_Uw()
# from line 180 in pdinew.f
def _dry_spell_abatement(K8, 
                         PPR, 
                         PX1,
                         PX2, 
                         PX3, 
                         X, 
                         PDSI, 
                         PHDI, 
                         WPLM, 
                         j, 
                         m, 
                         nendyr, 
                         nbegyr, 
                         SX1, 
                         SX2, 
                         SX3, 
                         SX, 
                         indexj, 
                         indexm,
                         Z,
                         PV,
                         V):

        #-----------------------------------------------------------------------
        #      DROUGHT ABATEMENT IS POSSIBLE
        #-----------------------------------------------------------------------
        UW = Z[j, m] + 0.15  
        PV = UW + max(V, 0.0) 
        if (PV <= 0):
            # GOTO 210 in pdinew.f
            # compare to 
            # PV, PX1, PX2, PX3, PPe, X, BT = Between0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)
            # in pdsi_from_zindex()
            X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8 = _between_0s(K8, 
                                                                  PPR, 
                                                                  PX1,
                                                                  PX2, 
                                                                  PX3, 
                                                                  X1,
                                                                  X2, 
                                                                  X3, 
                                                                  X, 
                                                                  PDSI, 
                                                                  PHDI, 
                                                                  WPLM, 
                                                                  j, 
                                                                  m, 
                                                                  nendyr, 
                                                                  nbegyr, 
                                                                  SX1, 
                                                                  SX2, 
                                                                  SX3, 
                                                                  SX, 
                                                                  indexj, 
                                                                  indexm,
                                                                  Z)
                
        else:
            #     ---------------------- DURING A DROUGHT, PV =< 0 IMPLIES  
            #                            PROB(END) HAS RETURNED TO 0
            ZE = -2.691 * X3 - 1.5
            #-----------------------------------------------------------------------
            #     PROB(END) = 100 * (V/Q)  WHERE:   
            #                 V = SUM OF MOISTURE EXCESS OR DEFICIT (UD OR UW)  
            #                 DURING CURRENT ABATEMENT PERIOD   
            #             Q = TOTAL MOISTURE ANOMALY REQUIRED TO END THE
            #                 CURRENT DROUGHT OR WET SPELL  
            #-----------------------------------------------------------------------
            if PRO == 100.0: 
                #     --------------------- DROUGHT OR WET CONTINUES, CALCULATE 
                #                           PROB(END) - VARIABLE ZE 
                Q = ZE
                
            else:  
                
                Q = ZE + V

            PPR[j, m] = (PV / Q) * 100.0 
            if PPR[j, m] >= 100.0: 
                PPR[j, m] = 100.0
                PX3[j, m] = 0.0  
            else:
                PX3[j, m] = 0.897 * X3 + Z[j, m] / 3.0
              

            # in new version the X values and backtracking is again computed here, skipped in the NCEI Fortran


        return X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8
        
#-----------------------------------------------------------------------------------------------------------------------
# compare to 
# PX1, PX2, PX3, X, BT = Main(Z, k, PV, PPe, X1, X2, PX1, PX2, PX3, X, BT)
# in pdsi_from_zindex()
# from line 200 in pdinew.f
def _compute_X(PX1,
               PX2,
               PX3,
               X,
               X1,
               X2,
               Z,
               j,
               m,
               K8,
               PPR,
               PDSI, 
               PHDI, 
               WPLM, 
               nendyr, 
               nbegyr, 
               SX1, 
               SX2, 
               SX3, 
               SX, 
               indexj, 
               indexm,
               PV):
    
    # whether or not an appropriate value for X has been found
    found = False
    
    #-----------------------------------------------------------------------
    #     CONTINUE X1 AND X2 CALCULATIONS.  
    #     IF EITHER INDICATES THE START OF A NEW WET OR DROUGHT,
    #     AND IF THE LAST WET OR DROUGHT HAS ENDED, USE X1 OR X2
    #     AS THE NEW X3.
    #-----------------------------------------------------------------------
    PX1[j, m] = 0.897 * X1 + Z[j, m] / 3.0
    PX1[j, m] = max(PX1[j, m], 0.0)   
    if (PX1[j, m] >= 1.0):   
        
        if (PX3[j, m] == 0.0):   
            #         ------------------- IF NO EXISTING WET SPELL OR DROUGHT   
            #                             X1 BECOMES THE NEW X3 
            X[j, m]   = PX1[j, m] 
            PX3[j, m] = PX1[j, m] 
            PX1[j, m] = 0.0
            iass = 1
            _assign(iass, K8, PPR, PX1, PX2, PX3, X, PDSI, PHDI, WPLM, j, m, nendyr, nbegyr, SX1, SX2, SX3, SX, indexj, indexm)
            
            V = PV 
#             eff[j, m] = PV 
            PRO = PPR[j, m] 
            X1  = PX1[j, m] 
            X2  = PX2[j, m] 
            X3  = PX3[j, m] 

            found = True
            
    else:
        PX2[j, m] = 0.897 * X2 + Z[j, m] / 3.0
        PX2[j, m] = min(PX2[j, m], 0.0)   
        if PX2[j, m] <= -1.0:  
            
            if (PX3[j, m] == 0.0):   
                #         ------------------- IF NO EXISTING WET SPELL OR DROUGHT   
                #                             X2 BECOMES THE NEW X3 
                X[j, m]   = PX2[j, m] 
                PX3[j, m] = PX2[j, m] 
                PX2[j, m] = 0.0  
                iass = 2            
                _assign(iass, K8, PPR, PX1, PX2, PX3, X, PDSI, PHDI, WPLM, j, m, nendyr, nbegyr, SX1, SX2, SX3, SX, indexj, indexm)
    
                found = True
                
        elif PX3[j, m] == 0.0:   
            #    -------------------- NO ESTABLISHED DROUGHT (WET SPELL), BUT X3=0  
            #                         SO EITHER (NONZERO) X1 OR X2 MUST BE USED AS X3   
            if PX1[j, m] == 0.0:   
            
                X[j, m] = PX2[j, m]   
                iass = 2            
                _assign(iass, K8, PPR, PX1, PX2, PX3, X, PDSI, PHDI, WPLM, j, m, nendyr, nbegyr, SX1, SX2, SX3, SX, indexj, indexm)

                found = True

            elif PX2[j, m] == 0:
                
                X[j, m] = PX1[j, m]   
                iass = 1   
                _assign(iass, K8, PPR, PX1, PX2, PX3, X, PDSI, PHDI, WPLM, j, m, nendyr, nbegyr, SX1, SX2, SX3, SX, indexj, indexm)
    
                found = True

        #-----------------------------------------------------------------------
        #     AT THIS POINT THERE IS NO DETERMINED VALUE TO ASSIGN TO X,
        #     ALL VALUES OF X1, X2, AND X3 ARE SAVED IN FILE 8. AT A LATER  
        #     TIME X3 WILL REACH A VALUE WHERE IT IS THE VALUE OF X (PDSI). 
        #     AT THAT TIME, THE ASSIGN SUBROUTINE BACKTRACKS THROUGH FILE   
        #     8 CHOOSING THE APPROPRIATE X1 OR X2 TO BE THAT MONTHS X. 
        #-----------------------------------------------------------------------
        elif not found and (K8 > 40):  # STOP 'X STORE ARRAYS FULL'  
            
            SX1[K8] = PX1[j, m] 
            SX2[K8] = PX2[j, m] 
            SX3[K8] = PX3[j, m] 
            X[j, m]  = PX3[j, m] 
            K8 = K8 + 1
            k8max = K8  

    #-----------------------------------------------------------------------
    #     SAVE THIS MONTHS CALCULATED VARIABLES (V,PRO,X1,X2,X3) FOR   
    #     USE WITH NEXT MONTHS DATA 
    #-----------------------------------------------------------------------
    V = PV 
#     eff[j, m] = PV 
    PRO = PPR[j, m] 
    X1  = PX1[j, m] 
    X2  = PX2[j, m] 
    X3  = PX3[j, m] 

    return X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8
 
#-----------------------------------------------------------------------------------------------------------------------
# compare to Between0s()
# 
# typical usage:
#
# PV, PX1, PX2, PX3, PPe, X, BT = Between0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)
def _between_0s(K8, 
                PPR, 
                PX1,
                PX2, 
                PX3, 
                X1,
                X2, 
                X3, 
                X, 
                PDSI, 
                PHDI, 
                WPLM, 
                j, 
                m, 
                nendyr, 
                nbegyr, 
                SX1, 
                SX2, 
                SX3, 
                SX, 
                indexj, 
                indexm,
                Z):

    #-----------------------------------------------------------------------
    #     PROB(END) RETURNS TO 0.  A POSSIBLE ABATEMENT HAS FIZZLED OUT,
    #     SO WE ACCEPT ALL STORED VALUES OF X3  
    #-----------------------------------------------------------------------
    PV = 0.0 
    PX1[j, m] = 0.0 
    PX2[j, m] = 0.0 
    PPR[j, m] = 0.0 
    PX3[j, m] = 0.897 * X3 + Z[j, m] / 3.0
    X[j, m]   = PX3[j, m] 
    if (K8 == 1): 
        PDSI[j, m] = X[j, m]  
        PHDI[j, m] = PX3[j, m] 
        if PX3[j, m] ==  0.0:
            PHDI[j, m] = X[j, m]
        _case(PPR[j, m], PX1[j, m], PX2[j, m], PX3[j, m], WPLM[j, m]) 
    else:
        iass = 3   
        _assign(iass, K8, PPR, PX1, PX2, PX3, X, PDSI, PHDI, WPLM, j, m, nendyr, nbegyr, SX1, SX2, SX3, SX, indexj, indexm)

    #-----------------------------------------------------------------------
    #     SAVE THIS MONTHS CALCULATED VARIABLES (V,PRO,X1,X2,X3) FOR   
    #     USE WITH NEXT MONTHS DATA 
    #-----------------------------------------------------------------------
    V = PV 
#     eff[j, m] = PV 
    PRO = PPR[j, m] 
    X1  = PX1[j, m] 
    X2  = PX2[j, m] 
    X3  = PX3[j, m] 

    return X, X1, X2, X3, SX1, SX2, SX3, V, PRO, K8

#-----------------------------------------------------------------------------------------------------------------------
def _assign(iass,
            k8,
            PPR,
            X1,
            PX2,
            PX3,
            X,
            PDSI,
            PHDI,
            WPLM,
            j,
            m,
            nendyr,
            nbegyr,
            SX1,
            SX2,
            SX3,
            SX,
            indexj,
            indexm):

    '''
    :param iass:
    :param k8:  
    :param ppr: ndarray with shape (200, 12)
    :param px1: ndarray with shape (200, 12)
    :param px2: ndarray with shape (200, 12)
    :param px3: ndarray with shape (200, 12)
    :param x: ndarray with shape (200, 12)
    :param pdsi: ndarray with shape (200, 13)
    :param phdi: ndarray with shape (200, 13)
    :param wplm: ndarray with shape (200, 13)
    :param j:
    :param m:
    :param nendyr:
    :param nbegyr:    
    :param sx1: ndarray with 40 elements
    :param sx2: ndarray with 40 elements
    :param sx3: ndarray with 40 elements
    :param sx: ndarray with 96 elements
    :param indexj: ndarray with 40 elements
    :param indexm: ndarray with 40 elements
     '''
    #   
    #-----------------------------------------------------------------------
    #     FIRST FINISH OFF FILE 8 WITH LATEST VALUES OF PX3, Z,X
    #     X=PX1 FOR I=1, PX2 FOR I=2, PX3,  FOR I=3 
    #-----------------------------------------------------------------------
    SX[k8] = X[j, m] 
    ISAVE = iass
    if k8 == 1:
        PDSI[j, m] = X[j, m]  
        PHDI[j, m] = PX3[j, m] 
        if PX3[j, m] == 0.0:
            PHDI[j, m] = X[j, m]
        _case(PPR[j, m], PX1[j, m], PX2[j, m], PX3[j, m], WPLM[j, m]) 
        return

    if iass == 3:  
        #     ---------------- USE ALL X3 VALUES
        for Mm in range(k8):
    
            SX[Mm] = SX3[Mm]

    else: 
        #     -------------- BACKTRACK THRU ARRAYS, STORING ASSIGNED X1 (OR X2) 
        #                    IN SX UNTIL IT IS ZERO, THEN SWITCHING TO THE OTHER
        #                    UNTIL IT IS ZERO, ETC. 
        for Mm in range(k8, 0, -1):
            
            if SX1[Mm] == 0:
                ISAVE = 2 
                SX[Mm] = SX2[Mm]
            else:
                ISAVE = 1 
                SX[Mm] = SX1[Mm]

    #-----------------------------------------------------------------------
    #     PROPER ASSIGNMENTS TO ARRAY SX HAVE BEEN MADE,
    #     OUTPUT THE MESS   
    #-----------------------------------------------------------------------

    for n in range(k8):
        
        PDSI[indexj[n], indexm[n]] = SX[n] 
        PHDI[indexj[n], indexm[n]] = PX3[indexj[n], indexm[n]]
        
        if (PX3[indexj[n], indexm[n]] == 0.0):
            PHDI[indexj[n], indexm[n]] = SX[n]
            
        _case(PPR[indexj[n], indexm[n]],
             PX1[indexj[n], indexm[n]], 
             PX2[indexj[n], indexm[n]],
             PX3[indexj[n], indexm[n]],
             WPLM[indexj[n], indexm[n]])

    k8 = 1
    k8max = k8

    return

#-----------------------------------------------------------------------------------------------------------------------
def _case(PROB,
          X1,
          X2,
          X3):
    #   
    #     THIS SUBROUTINE SELECTS THE PRELIMINARY (OR NEAR-REAL TIME)   
    #     PALMER DROUGHT SEVERITY INDEX (PDSI) FROM THE GIVEN X VALUES  
    #     DEFINED BELOW AND THE PROBABILITY (PROB) OF ENDING EITHER A   
    #     DROUGHT OR WET SPELL. 
    #   
    #     X1   - INDEX FOR INCIPIENT WET SPELLS (ALWAYS POSITIVE)   
    #     X2   - INDEX FOR INCIPIENT DRY SPELLS (ALWAYS NEGATIVE)   
    #     X3   - SEVERITY INDEX FOR AN ESTABLISHED WET SPELL (POSITIVE) OR DROUGHT (NEGATIVE)  
    #     PALM - THE SELECTED PDSI (EITHER PRELIMINARY OR FINAL)
    #   
    #   This subroutine written and provided by CPC (Tom Heddinghaus & Paul Sabol).
    #   

    if X3 == 0.0: #) GO TO 10 in pdinew.f
        #     IF X3=0 THE INDEX IS NEAR NORMAL AND EITHER A DRY OR WET SPELL
        #     EXISTS.  CHOOSE THE LARGEST ABSOLUTE VALUE OF X1 OR X2.  
        PALM = X1
        if abs(X2) > abs(X1): 
            PALM = X2

    elif  PROB > 0.0 and PROB < 100.0: # GO TO 20 in pdinew.f

        # put the probability value into 0..1 range
        PRO = PROB / 100.0
        if X3 > 0.0: #) GO TO 30
            #     TAKE THE WEIGHTED SUM OF X3 AND X2
            PALM = (1.0 - PRO) * X3 + PRO * X2   
        else:  
            #     TAKE THE WEIGHTED SUM OF X3 AND X1
            PALM = (1.0 - PRO) * X3 + PRO * X1   

    else:
        #     A WEATHER SPELL IS ESTABLISHED AND PALM=X3 AND IS FINAL
        PALM = X3
 
    return PALM

import calendar
import logging
import math
import numpy as np
import pandas as pd
import utils
import warnings

# set up a basic, global logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def pdsi_from_climatology(precip_timeseries,
                          temp_timeseries,
                          awc,
                          latitude,
                          B,
                          H,
                          data_begin_year,
                          data_end_year,
                          calibration_begin_year,
                          calibration_end_year):
    
    # calculate the negative tangent of the latitude which is used as an argument to the water balance function
    neg_tan_lat = -1 * math.tan(math.radians(latitude))

    # compute water balance values using the function translated from the Fortran pdinew.f
    #NOTE keep this code in place in order to compute the PET used later, since the two have 
    # different PET algorithms and we want to compare PDSI using the same PET inputs 
    pdat, spdat, pedat, pldat, prdat, rdat, tldat, etdat, rodat, prodat, tdat, sssdat, ssudat = \
        _water_balance(temp_timeseries, precip_timeseries, awc, neg_tan_lat, B, H)
                 
    #NOTE we need to compute CAFEC coefficients for use later/below
    # compute PDSI etc. using translated functions from pdinew.f Fortran code
    alpha, beta, delta, gamma, t_ratio = _cafec_coefficients(precip_timeseries,
                                                             pedat,
                                                             etdat,
                                                             prdat,
                                                             rdat,
                                                             rodat,
                                                             prodat,
                                                             tldat,
                                                             pldat,
                                                             spdat,
                                                             data_begin_year,
                                                             calibration_begin_year,
                                                             calibration_end_year)
     
    # compute the weighting factor (climatic characteristic) using the version translated from pdinew.f
    K = _climatic_characteristic(alpha,
                                 beta,
                                 gamma,
                                 delta,
                                 pdat,
                                 pedat,
                                 prdat,
                                 spdat,
                                 pldat,
                                 t_ratio,
                                 data_begin_year,
                                 calibration_begin_year,
                                 calibration_end_year)

    # compute the indices now that we have calculated all required intermediate values    
    PDSI, PHDI, PMDI, Z = _zindex_pdsi(precip_timeseries,
                                              pedat,
                                              prdat,
                                              spdat,
                                              pldat,
                                              alpha,
                                              beta,
                                              gamma,
                                              delta,
                                              K, 
                                              data_begin_year, 
                                              data_end_year)
                
    return PDSI, PHDI, PMDI, Z

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
    :return 1-D numpy.ndarray of CAFEC coefficient values, alpha, beta, delta, gamma, and the T ratio,
            with shape of these arrays == (12,), corresponding to calendar months (12 elements)
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
def _water_balance(T,
                   P,
                   AWC,
                   TLA,
                   B, 
                   H,
                   begin_year=1895):

    '''
    Computes a water balance accounting for monthly time series. Translated from the Fortran code pdinew.f
    
    :param T: monthly average temperature values, starting in January of the initial year 
    :param P: monthly total precipitation values, starting in January of the initial year 
    :param AWC: available water capacity, below (not including) the top inch
    :param B: read from soil constants file 
    :param H: read from soil constants file
    :param begin_year: initial year of the dataset  
    :param TLA: negative tangent of the latitude
    :return: P, SP, PE, PL, PR, R, L, ET, RO, T, SSs, SSu
    :rtype: numpy arrays with shape (total_years, 12)
    '''
    
    # find the number of years from the input array, assume shape (months)
    
    # reshape the precipitation array from 1-D (assumed to be total months) to (years, 12) with the second  
    # dimension being calendar months, and the final/missing monthly values of the final year padded with NaNs
    T = utils.reshape_to_years_months(T)
    P = utils.reshape_to_years_months(P)
    total_years = P.shape[0]
    
    WCTOP = 1.0
    SS  = WCTOP
    SU = AWC
    WCTOT = AWC + WCTOP

    PHI = np.array([-0.3865982, -0.2316132, -0.0378180, 0.1715539, 0.3458803, 0.4308320, \
                     0.3916645, 0.2452467, 0.0535511, -0.15583436, -0.3340551, -0.4310691])
    
    # initialize the data arrays with NaNs    
    pdat = np.full((total_years, 12), np.NaN)
    spdat = np.full((total_years, 12), np.NaN)
    pedat = np.full((total_years, 12), np.NaN)
    pldat = np.full((total_years, 12), np.NaN)
    prdat = np.full((total_years, 12), np.NaN)
    rdat = np.full((total_years, 12), np.NaN)
    tldat = np.full((total_years, 12), np.NaN)
    etdat = np.full((total_years, 12), np.NaN)
    rodat = np.full((total_years, 12), np.NaN)
    prodat = np.full((total_years, 12), np.NaN)
    tdat = np.full((total_years, 12), np.NaN)
    sssdat = np.full((total_years, 12), np.NaN)
    ssudat = np.full((total_years, 12), np.NaN)

    #       loop on years and months
    end_year = begin_year + total_years
    years_range = range(begin_year, end_year)
    for year_index, year in enumerate(years_range):
    
        for month_index in range(12):
    
            temperature = T[year_index, month_index]
            precipitation = P[year_index, month_index]
            
            #-----------------------------------------------------------------------
            #     HERE START THE WATER BALANCE CALCULATIONS
            #-----------------------------------------------------------------------
            SP = SS + SU
            PR = AWC + WCTOP - SP

            # PRO is the potential runoff. According to Alley (1984),
            # PRO = AWC - PR = Ss + Su; here Ss and Su refer to those values at
            # the beginning of the month: Ss0 and Su0.
            PRO = SP

            #-----------------------------------------------------------------------
            #     1 - CALCULATE PE (POTENTIAL EVAPOTRANSPIRATION)   
            #-----------------------------------------------------------------------
            if temperature <= 32.0:
                PE   = 0.0
            else:  
                DUM = PHI[month_index] * TLA 
                DK = math.atan(math.sqrt(1.0 - (DUM * DUM)) / DUM)   
                if DK < 0.0:
                    DK = 3.141593 + DK  
                DK   = (DK + 0.0157) / 1.57  
                if temperature >= 80.0:
                    PE = (math.sin((temperature / 57.3) - 0.166) - 0.76) * DK
                else:  
                    DUM = math.log(temperature - 32.0)
                    PE = math.exp(-3.863233 + (B * 1.715598) - (B * math.log(H)) + (B * DUM)) * DK 
        
            #-----------------------------------------------------------------------
            #     CONVERT DAILY TO MONTHLY  
            #-----------------------------------------------------------------------
            PE = PE * calendar.monthrange(year, month_index + 1)[1]

            #-----------------------------------------------------------------------
            #     2 - PL  POTENTIAL LOSS
            #-----------------------------------------------------------------------
            if SS >= PE:
                PL  = PE  
            else:  
                PL = ((PE - SS) * SU) / (AWC + WCTOP) + SS   
                PL = min(PL, SP)   
        
            #-----------------------------------------------------------------------
            #     3 - CALCULATE RECHARGE, RUNOFF, RESIDUAL MOISTURE, LOSS TO BOTH   
            #         SURFACE AND UNDER LAYERS, DEPENDING ON STARTING MOISTURE  
            #         CONTENT AND VALUES OF PRECIPITATION AND EVAPORATION.  
            #-----------------------------------------------------------------------
            if precipitation >= PE:
                #     ----------------- PRECIP EXCEEDS POTENTIAL EVAPORATION
                ET = PE   
                TL = 0.0  
                if (precipitation - PE) > (WCTOP - SS):
                    #         ------------------------------ EXCESS PRECIP RECHARGES
                    #                                        UNDER LAYER AS WELL AS UPPER   
                    RS = WCTOP - SS  
                    SSS = WCTOP  
                    if (precipitation - PE - RS) < (AWC - SU):
                    #             ---------------------------------- BOTH LAYERS CAN TAKE   
                    #                                                THE ENTIRE EXCESS  
                        RU = precipitation - PE - RS  
                        RO = 0.0  
                    else:  
                        #             ---------------------------------- SOME RUNOFF OCCURS 
                        RU = AWC - SU   
                        RO = precipitation - PE - RS - RU 

                    SSU = SU + RU 
                    R   = RS + RU 
                else:  
                    #         ------------------------------ ONLY TOP LAYER RECHARGED   
                    R  = precipitation - PE  
                    SSS = SS + precipitation - PE 
                    SSU = SU  
                    RO  = 0.0 

            else:
                #     ----------------- EVAPORATION EXCEEDS PRECIPITATION   
                R  = 0.0  
                if SS >= (PE - precipitation):
                #         ----------------------- EVAP FROM SURFACE LAYER ONLY  
                    SL  = PE - precipitation  
                    SSS = SS - SL 
                    UL  = 0.0 
                    SSU = SU  
                else:
                    #         ----------------------- EVAP FROM BOTH LAYERS 
                    SL  = SS  
                    SSS = 0.0 
                    UL  = (PE - precipitation - SL) * SU / (WCTOT)  
                    UL  = min(UL, SU)
                    SSU = SU - UL 

                TL  = SL + UL 
                RO  = 0.0 
                ET  = precipitation  + SL + UL

            # set the climatology and water balance data array values for this year/month time step
            pdat[year_index, month_index] = precipitation
            spdat[year_index, month_index] = SP
            pedat[year_index, month_index] = PE
            pldat[year_index, month_index] = PL
            prdat[year_index, month_index] = PR
            rdat[year_index, month_index] = R
            tldat[year_index, month_index] = TL
            etdat[year_index, month_index] = ET
            rodat[year_index, month_index] = RO
            prodat[year_index, month_index] = PRO
            tdat[year_index, month_index] = temperature
            sssdat[year_index, month_index] = SSS
            ssudat[year_index, month_index] = SSU
      
            # reset the upper and lower soil moisture values
            SS = SSS
            SU = SSU

    return pdat, spdat, pedat, pldat, prdat, rdat, tldat, etdat, rodat, prodat, tdat, sssdat, ssudat
    
#-----------------------------------------------------------------------------------------------------------------------
def _zindex_pdsi(P,
                 PE,
                 PR,
                 SP,
                 PL,
                 alpha,
                 beta,
                 gamma,
                 delta,
                 AK,
                 nbegyr,#=1895,
                 nendyr):#=2017

    # reshape the precipitation and PPR to (years, 12)
    P = utils.reshape_to_years_months(P)
    
    # create empty/zero arrays for intermediate values
    PPR = np.full(P.shape, np.NaN)
    CP = np.full(P.shape, np.NaN)
    Z = np.full(P.shape, np.NaN)    
    PDSI = np.full(P.shape, np.NaN)
    PHDI = np.full(P.shape, np.NaN)
    WPLM = np.full(P.shape, np.NaN)
    SX = np.full(P.shape, np.NaN)
    X = np.full(P.shape, np.NaN)
    indexj = np.empty(P.shape, dtype=int)
    indexm = np.empty(P.shape, dtype=int)
    PX1 = np.zeros(P.shape)
    PX2 = np.zeros(P.shape)
    PX3 = np.zeros(P.shape)
    SX1 = np.zeros(P.shape)
    SX2 = np.zeros(P.shape)
    SX3 = np.zeros(P.shape)
    
    # intermediate values used in computations below 
    PV  = 0.0
    V   = 0.0 
    PRO = 0.0 
    X1  = 0.0 
    X2  = 0.0 
    X3  = 0.0 
    K8  = 0
    k8max = 0

    # create container for the arrays and values we'll use throughout the computation loop below
    array0 = np.reshape(P, (P.size, 1))
    df = pd.DataFrame(data=array0, 
                      index=range(0, array0.size), 
                      columns=['P'])

    # create a list of coluumn names that match to the intermediate work arrays
    column_names = ['PPR', 'CP', 'Z', 'PDSI', 'PHDI', 'WPLM', 'SX', 'SX1', 'SX2', 'SX3', 'X', 'indexj', 'indexm', 'PX1', 'PX2', 'PX3']
    for column_name in column_names:
        
        # get the array corresponding to the current column
        column_array = eval(column_name).flatten()
        
        # add the column to the DataFrame (as a Series)
        df[column_name] = pd.Series(column_array)

    # loop over all years and months of the time series
    for j in range(P.shape[0]):    
                
        for m in range(12):

            i = (j * 12) + m
            
            df.indexj[K8] = j
            df.indexm[K8] = m

            #-----------------------------------------------------------------------
            #     LOOP FROM 160 TO 230 REREADS data FOR CALCULATION OF   
            #     THE Z-INDEX (MOISTURE ANOMALY) AND PDSI (VARIABLE X). 
            #     THE FINAL OUTPUTS ARE THE VARIABLES PX3, X, AND Z  WRITTEN
            #     TO FILE 11.   
            #-----------------------------------------------------------------------
            ZE = 0.0 
            UD = 0.0 
            UW = 0.0 
            
            # compute the CAFEC precipitation (df.CP)
            CET = alpha[m] * PE[j, m]   # eq. 10, Palmer 1965
            CR = beta[m] * PR[j, m]     # eq. 11, Palmer 1965
            CRO = gamma[m] * SP[j, m]   # eq. 12, Palmer 1965
            CL = delta[m] * PL[j, m]    # eq. 13, Palmer 1965
            df.CP[i] = CET + CR + CRO - CL  # eq. 14, Palmer 1965
            
            # moisture departure, d
            CD = df.P[i] - df.CP[i]  
            
            # Z-index = K * d
            df.Z[i] = AK[m] * CD
            
            # now with the Z-Index value computed we compute X based on current values
             
            if PRO == 100.0 or PRO == 0.0:  
            #     ------------------------------------ NO ABATEMENT UNDERWAY
            #                                          WET OR DROUGHT WILL END IF   
            #                                             -0.5 =< X3 =< 0.5   
                if abs(X3) <= 0.5:
                #         ---------------------------------- END OF DROUGHT OR WET  
                    PV = 0.0 
                    df.PPR[i] = 0.0 
                    df.PX3[i] = 0.0 
                    #             ------------ BUT CHECK FOR NEW WET OR DROUGHT START FIRST
                    # GOTO 200 in pdinew.f
                    # compare to 
                    # PX1, PX2, PX3, X, BT = Main(Z, k, PV, PPe, X1, X2, PX1, PX2, PX3, X, BT)
                    # in palmer.pdsi_from_zindex()
                    df, X1, X2, X3, V, PRO, K8, k8max = _compute_X(df, X1, X2, j, m, K8, k8max, nendyr, nbegyr, PV)
                     
                elif X3 > 0.5:   
                    #         ----------------------- WE ARE IN A WET SPELL 
                    if df.Z[i] >= 0.15:   
                        #              ------------------ THE WET SPELL INTENSIFIES 
                        #GO TO 210 in pdinew.f
                        # compare to 
                        # PV, PX1, PX2, PX3, PPe, X, BT = Between0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)
                        # in palmer.pdsi_from_zindex()
                        df, X1, X2, X3, V, PRO, K8, k8max = _between_0s(df, K8, k8max, X1, X2, X3, j, m, nendyr, nbegyr)
                        
                    else:
                        #             ------------------ THE WET STARTS TO ABATE (AND MAY END)  
                        #GO TO 170 in pdinew.f
                        # compare to
                        # Ud, Ze, Q, PV, PPe, PX1, PX2, PX3, X, BT = Function_Ud(k, Ud, Z, Ze, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                        # in palmer.pdsi_from_zindex()
                        df, X1, X2, X3, V, PRO, K8, k8max = _wet_spell_abatement(df, V, K8, k8max, PRO, j, m, nendyr, nbegyr, X1, X2, X3)

                elif X3 < -0.5:  
                    #         ------------------------- WE ARE IN A DROUGHT 
                    if df.Z[i] <= -0.15:  
                        #              -------------------- THE DROUGHT INTENSIFIES 
                        #GO TO 210
                        # compare to 
                        # PV, PX1, PX2, PX3, PPe, X, BT = Between0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)
                        # in palmer.pdsi_from_zindex()
                        df, X1, X2, X3, V, PRO, K8, k8max = _between_0s(df, K8, k8max, X1, X2, X3, j, m, nendyr, nbegyr)

                    else:
                        #             ------------------ THE DROUGHT STARTS TO ABATE (AND MAY END)  
                        #GO TO 180
                        # compare to 
                        # Uw, Ze, Q, PV, PPe, PX1, PX2, PX3, X, BT = Function_Uw(k, Uw, Z, Ze, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                        # in palmer.pdsi_from_zindex()
                        df, X1, X2, X3, V, PRO, K8, k8max = _dry_spell_abatement(df, K8, k8max, j, m, nendyr, nbegyr, PV, V, X1, X2, X3, PRO)

            else:
                #     ------------------------------------------ABATEMENT IS UNDERWAY   
                if X3 > 0.0:
                    
                    #         ----------------------- WE ARE IN A WET SPELL 
                    #GO TO 170 in pdinew.f
                    # compare to
                    # Ud, Ze, Q, PV, PPe, PX1, PX2, PX3, X, BT = Function_Ud(k, Ud, Z, Ze, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                    # in palmer.pdsi_from_zindex()
                    df, X1, X2, X3, V, PRO, K8, k8max = _wet_spell_abatement(df, V, K8, k8max, PRO, j, m, nendyr, nbegyr, X1, X2, X3)
                
                else:  # if X3 <= 0.0:
                    
                    #         ----------------------- WE ARE IN A DROUGHT   
                    #GO TO 180
                    # compare to
                    # Uw, Ze, Q, PV, PPe, PX1, PX2, PX3, X, BT = Function_Uw(k, Uw, Z, Ze, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                    # in palmer.pdsi_from_zindex()
                    df, X1, X2, X3, V, PRO, K8, k8max = _dry_spell_abatement(df, K8, k8max, j, m, nendyr, nbegyr, PV, V, X1, X2, X3, PRO)

#     # assign X to the PDSI array?
#     df.PDSI = df.X
    
    for k8 in range(0, k8max):

        # years(j) and months(m) to series index ix
        ix = (df.indexj[k8] * 12) + df.indexm[k8]
        
        df.PDSI[ix] = df.X[ix] 
        df.PHDI[ix] = df.PX3[ix]
        
        if df.PX3[ix] == 0.0:
        
            df.PHDI[ix] = df.X[ix]
        
        df.WPLM[ix] = _case(df.PPR[ix], df.PX1[ix], df.PX2[ix], df.PX3[ix]) 
      
    return df.PDSI.values, df.PHDI.values, df.WPLM.values, df.Z.values

#-----------------------------------------------------------------------------------------------------------------------
# compare to Function_Ud()
# from line 170 in pdinew.f
# NOTE careful not to confuse the PRO being returned (probability) with PRO array of potential run off values
def _wet_spell_abatement(df,
                         V, 
                         K8, 
                         k8max,
                         PRO,
                         j, 
                         m, 
                         nendyr, 
                         nbegyr,
                         X1, 
                         X2, 
                         X3):
    
    # combine the years (j) and months (m) indices to series index ix
    ix = (df.indexj[K8] * 12) + df.indexm[K8]
        
    #-----------------------------------------------------------------------
    #      WET SPELL ABATEMENT IS POSSIBLE  
    #-----------------------------------------------------------------------
    UD = df.Z[ix] - 0.15  
    PV = UD + min(V, 0.0) 
    if PV >= 0.0:

        # GOTO 210 
        df, X1, X2, X3, V, PRO, K8, k8max = _between_0s(df, K8, k8max, X1, X2, X3, j, m, nendyr, nbegyr)

    else:
        #     ---------------------- DURING A WET SPELL, PV => 0 IMPLIES
        #                            PROB(END) HAS RETURNED TO 0
        ZE = -2.691 * X3 + 1.5
    
        #-----------------------------------------------------------------------
        #     PROB(END) = 100 * (V/Q)  WHERE:   
        #             V = SUM OF MOISTURE EXCESS OR DEFICIT (UD OR UW)  
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
    
        df.PPR[ix] = (PV / Q) * 100.0  # eq. 30 Palmer 1965, percentage probability that a drought or wet spell has ended
        if df.PPR[ix] >= 100.0:
             
              df.PPR[ix] = 100.0
              df.PX3[ix] = 0.0  
        else:
              # Wells et al (2003) eq. 4
              df.PX3[ix] = (0.897 * X3) + (df.Z[ix] / 3.0)


        # in new version the X values and backtracking is again computed here, skipped in the NCEI Fortran
        
    return _compute_X(df, X1, X2, j, m, K8, k8max, nendyr, nbegyr, PV)

#-----------------------------------------------------------------------------------------------------------------------
# compare to Function_Uw()
# from line 180 in pdinew.f
def _dry_spell_abatement(df,
                                K8,
                                k8max,
                                j, 
                                m, 
                                nendyr, 
                                nbegyr, 
                                PV,
                                V,
                                X1, 
                                X2,
                                X3,
                                PRO):

    # combine the years (j) and months (m) indices to series index ix
    ix = (df.indexj[K8] * 12) + df.indexm[K8]
        
    #-----------------------------------------------------------------------
    #      DROUGHT ABATEMENT IS POSSIBLE
    #-----------------------------------------------------------------------
    UW = df.Z[ix] + 0.15  
    PV = UW + max(V, 0.0) 
    if PV <= 0:
        # GOTO 210 in pdinew.f
        # compare to 
        # PV, PX1, PX2, PX3, PPe, X, BT = Between0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)
        # in pdsi_from_zindex()
        df, X1, X2, X3, V, PRO, K8, k8max = _between_0s(df, K8, k8max, X1, X2, X3, j, m, nendyr, nbegyr)
            
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

        # convert the 2-D index values j (years) and m (months) to a series index
        i = (j * 12) + m
        
        df.PPR[i] = (PV / Q) * 100.0 # eq. 30 Palmer 1965, percentage probability that a drought or wet spell has ended
        
        if df.PPR[i] >= 100.0: 
        
            df.PPR[i] = 100.0
            df.PX3[i] = 0.0  
        
        else:
        
            df.PX3[i] = (0.897 * X3) + (df.Z[i] / 3.0)
          

        #NOTE in new version the X values and backtracking is again computed here, not done in the NCEI Fortran

    return _compute_X(df, X1, X2, j, m, K8, k8max, nendyr, nbegyr, PV)

#-----------------------------------------------------------------------------------------------------------------------
# compare to 
# PX1, PX2, PX3, X, BT = Main(Z, k, PV, PPe, X1, X2, PX1, PX2, PX3, X, BT)
# in palmer.pdsi_from_zindex()
# from line 200 in pdinew.f
def _compute_X(df,
               X1,
               X2,
               j,
               m,
               K8,
               k8max,
               nendyr, 
               nbegyr, 
               PV):
    
    # the values within the DataFrame are in a series, so get the single index value assuming j is years and m is months
    i = (j * 12) + m


    # FIXME ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # We have an issue in this function whereby we sometimes happen to not fall into any of the conditionals below 
    # which result in a call to the _assign() function, resulting in df.* values of NaN for this index i.
    #
    # FIXME ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #-----------------------------------------------------------------------
    #     CONTINUE X1 AND X2 CALCULATIONS.  
    #     IF EITHER INDICATES THE START OF A NEW WET OR DROUGHT,
    #     AND IF THE LAST WET OR DROUGHT HAS ENDED, USE X1 OR X2
    #     AS THE NEW X3.
    #-----------------------------------------------------------------------
    df.PX1[i] = (0.897 * X1) + (df.Z[i] / 3.0)
    df.PX1[i] = max(df.PX1[i], 0.0)   
    if df.PX1[i] >= 1.0:   
        
        if df.PX3[i] == 0.0:   
            #         ------------------- IF NO EXISTING WET SPELL OR DROUGHT   
            #                             X1 BECOMES THE NEW X3 
            df.X[i]   = df.PX1[i] 
            df.PX3[i] = df.PX1[i] 
            df.PX1[i] = 0.0
            iass = 1
            df = _assign(df, iass, K8, j, m, nendyr, nbegyr)
            K8 = k8max = 0
            
            #GOTO 220 in pdinew.f
            V = PV 
            PRO = df.PPR[i] 
            X1  = df.PX1[i] 
            X2  = df.PX2[i] 
            X3  = df.PX3[i] 

            return df, X1, X2, X3, V, PRO, K8, k8max
            
    df.PX2[i] = (0.897 * X2) + (df.Z[i] / 3.0)
    df.PX2[i] = min(df.PX2[i], 0.0)   
    if df.PX2[i] <= -1.0:  
        
        if (df.PX3[i] == 0.0):   
            #         ------------------- IF NO EXISTING WET SPELL OR DROUGHT   
            #                             X2 BECOMES THE NEW X3 
            df.X[i]   = df.PX2[i] 
            df.PX3[i] = df.PX2[i] 
            df.PX2[i] = 0.0  
            iass = 2            
            df = _assign(df, iass, K8, j, m, nendyr, nbegyr)
            K8 = k8max = 0

            #GOTO 220 in pdinew.f
            V = PV 
            PRO = df.PPR[i] 
            X1  = df.PX1[i] 
            X2  = df.PX2[i] 
            X3  = df.PX3[i] 

            return df, X1, X2, X3, V, PRO, K8, k8max
            
    if df.PX3[i] == 0.0:   
        #    -------------------- NO ESTABLISHED DROUGHT (WET SPELL), BUT X3=0  
        #                         SO EITHER (NONZERO) X1 OR X2 MUST BE USED AS X3   
        if df.PX1[i] == 0.0:   
        
            df.X[i] = df.PX2[i]   
            iass = 2            
            df = _assign(df, iass, K8, j, m, nendyr, nbegyr)
            K8 = k8max = 0

            #GOTO 220 in pdinew.f
            V = PV 
            PRO = df.PPR[i] 
            X1  = df.PX1[i] 
            X2  = df.PX2[i] 
            X3  = df.PX3[i] 

            return df, X1, X2, X3, V, PRO, K8, k8max

        elif df.PX2[i] == 0:
            
            df.X[i] = df.PX1[i]   
            iass = 1   
            df = _assign(df, iass, K8, j, m, nendyr, nbegyr)
            K8 = k8max = 0

            #GOTO 220 in pdinew.f
            V = PV 
            PRO = df.PPR[i] 
            X1  = df.PX1[i] 
            X2  = df.PX2[i] 
            X3  = df.PX3[i] 

            return df, X1, X2, X3, V, PRO, K8, k8max

    #-----------------------------------------------------------------------
    #     AT THIS POINT THERE IS NO DETERMINED VALUE TO ASSIGN TO X,
    #     ALL VALUES OF X1, X2, AND X3 ARE SAVED IN SX* arrays. AT A LATER  
    #     TIME X3 WILL REACH A VALUE WHERE IT IS THE VALUE OF X (PDSI). 
    #     AT THAT TIME, THE ASSIGN SUBROUTINE BACKTRACKS THROUGH SX* arrays   
    #     CHOOSING THE APPROPRIATE X1 OR X2 TO BE THAT MONTH'S X. 
    #-----------------------------------------------------------------------
    
    df.SX1[K8] = df.PX1[i] 
    df.SX2[K8] = df.PX2[i] 
    df.SX3[K8] = df.PX3[i] 
    df.X[i]  = df.PX3[i] 
    K8 = K8 + 1
    k8max = K8  

    #-----------------------------------------------------------------------
    #     SAVE THIS MONTHS CALCULATED VARIABLES (V,PRO,X1,X2,X3) FOR   
    #     USE WITH NEXT MONTHS DATA 
    #-----------------------------------------------------------------------
    V = PV 
    PRO = df.PPR[i] 
    X1  = df.PX1[i] 
    X2  = df.PX2[i] 
    X3  = df.PX3[i] 

    return df, X1, X2, X3, V, PRO, K8, k8max
 
#-----------------------------------------------------------------------------------------------------------------------
# from 210 in pdinew.f
def _between_0s(df,
                K8,
                k8max, 
                X1,
                X2, 
                X3, 
                j, 
                m, 
                nendyr, 
                nbegyr):

    # convert years/months (j/m) indices to a series index
    ix = (j * 12) + m
    
    #-----------------------------------------------------------------------
    #     PROB(END) RETURNS TO 0.  A POSSIBLE ABATEMENT HAS FIZZLED OUT,
    #     SO WE ACCEPT ALL STORED VALUES OF X3  
    #-----------------------------------------------------------------------
    PV = 0.0 
    df.PPR[ix] = 0.0 
    df.PX1[ix] = 0.0 
    df.PX2[ix] = 0.0 
    df.PX3[ix] = (0.897 * X3) + (df.Z[ix] / 3.0)
    df.X[ix] = df.PX3[ix] 
    
    if K8 == 0: 
        
        df.PDSI[ix] = df.X[ix]  
        df.PHDI[ix] = df.PX3[ix] 
        
        if df.PX3[ix] == 0.0:
            
            df.PHDI[ix] = df.X[ix]
        
        df.WPLM[ix] = _case(df.PPR[ix], df.PX1[ix], df.PX2[ix], df.PX3[ix]) 
        
    else:

        iass = 3   
        df = _assign(df, iass, K8, j, m, nendyr, nbegyr)
        K8 = k8max = 0

    #-----------------------------------------------------------------------------------------------
    #     SAVE THIS MONTHS CALCULATED VARIABLES (V, PRO, X1, X2, X3) FOR USE WITH NEXT MONTHS DATA 
    #-----------------------------------------------------------------------------------------------
    V = PV 
    PRO = df.PPR[ix] 
    X1  = df.PX1[ix] 
    X2  = df.PX2[ix] 
    X3  = df.PX3[ix] 

    return df, X1, X2, X3, V, PRO, K8, k8max

#-----------------------------------------------------------------------------------------------------------------------
def _assign(df,
            iass,
            k8,
            j,
            m,
            nendyr,
            nbegyr):

    '''
    :df pandas DataFrame
    :param iass:
    :param k8:  
    :param j:
    :param m:
    :param nendyr:
    :param nbegyr:    
     '''

    # convert the 2-D time step indices (j, m) to a series index i
    i = (j * 12) + m
    
    #   
    #-----------------------------------------------------------------------
    #     FIRST FINISH OFF FILE 8 WITH LATEST VALUES OF PX3, Z,X
    #     X=PX1 FOR I=1, PX2 FOR I=2, PX3,  FOR I=3 
    #-----------------------------------------------------------------------
    df.SX[k8] = df.X[i] 
    if k8 == 0:  # no backtracking called for
        
        # set the PDSI to X
        df.PDSI[i] = df.X[i]  

        # the PHDI is X3 if not zero, otherwise use X
        df.PHDI[i] = df.PX3[i]         
        if df.PX3[i] == 0.0:
            df.PHDI[i] = df.X[i]
        
        # select the best fit for WPLM
        df.WPLM[i] = _case(df.PPR[i], df.PX1[i], df.PX2[i], df.PX3[i]) 
        
    else:
        
        if iass == 3:  
            #     ---------------- USE ALL X3 VALUES
            for Mm in range(k8):
        
                df.SX[Mm] = df.SX3[Mm]
    
        else: 
            #     -------------- BACKTRACK THRU ARRAYS, STORING ASSIGNED X1 (OR X2) 
            #                    IN SX UNTIL IT IS ZERO, THEN SWITCHING TO THE OTHER
            #                    UNTIL IT IS ZERO, ETC. 
            for Mm in range(k8, 0, -1):
                
                if df.SX1[Mm] == 0:
                    df.SX[Mm] = df.SX2[Mm]
                else:
                    df.SX[Mm] = df.SX1[Mm]
    
        #-----------------------------------------------------------------------
        #     PROPER ASSIGNMENTS TO ARRAY SX HAVE BEEN MADE,
        #     OUTPUT THE MESS   
        #-----------------------------------------------------------------------
    
        for n in range(k8):   # backtracking assignment of X
            
            # get the j/m index we need to start assigning to from the backtracking process
            ix = (df.indexj[n] * 12) + df.indexm[n]
            
            # pull from the current backtracking index
            df.PDSI[ix] = df.SX[n] 
            
            # the PHDI is X3 if not zero, otherwise use X
            df.PHDI[ix] = df.PX3[ix]
            if df.PX3[ix] == 0.0:
                df.PHDI[ix] = df.SX[n]
                
            # select the best fit for WPLM
            df.WPLM[ix] = _case(df.PPR[ix],
                                df.PX1[ix], 
                                df.PX2[ix],
                                df.PX3[ix])

        # set the PDSI to X
        df.PDSI[i] = df.X[i]  

        # the PHDI is X3 if not zero, otherwise use X
        df.PHDI[i] = df.PX3[i]         
        if df.PX3[i] == 0.0:
            df.PHDI[i] = df.X[i]
        
        # select the best fit for WPLM
        df.WPLM[i] = _case(df.PPR[i], df.PX1[i], df.PX2[i], df.PX3[i]) 

    return df

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
    #     PDSI - THE SELECTED PDSI (EITHER PRELIMINARY OR FINAL)
    #   
    #   This subroutine written and provided by CPC (Tom Heddinghaus & Paul Sabol).
    #   

    if X3 == 0.0: #) GO TO 10 in pdinew.f
        #     IF X3=0 THE INDEX IS NEAR NORMAL AND EITHER A DRY OR WET SPELL
        #     EXISTS.  CHOOSE THE LARGEST ABSOLUTE VALUE OF X1 OR X2.  
        PDSI = X1
        if abs(X2) > abs(X1): 
            PDSI = X2

    elif  PROB > 0.0 and PROB < 100.0: # GO TO 20 in pdinew.f

        # put the probability value into 0..1 range
        PRO = PROB / 100.0
        if X3 > 0.0: #) GO TO 30
            #     TAKE THE WEIGHTED SUM OF X3 AND X2
            PDSI = (1.0 - PRO) * X3 + PRO * X2   
        else:  
            #     TAKE THE WEIGHTED SUM OF X3 AND X1
            PDSI = (1.0 - PRO) * X3 + PRO * X1   

    else:
        #     A WEATHER SPELL IS ESTABLISHED AND PDSI=X3 AND IS FINAL
        PDSI = X3
 
    return PDSI

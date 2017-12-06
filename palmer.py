from collections import deque
import logging
import math
import pdinew
import numba
import numpy as np
import thornthwaite
import utils
import warnings

# set up a basic, global logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
_PDSI_MIN = -4.0
_PDSI_MAX = 4.0

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _water_balance(AWC,
                   PET,
                   P):
    """
    Performs a water balance accounting for a location which accounts for several monthly water balance variables, 
    calculated based on precipitation, potential evapotranspiration, and available water capacity of the soil.
     
    Input arrays are expected to be the same size, corresponding to the total number of months.
    
    :param AWC: available water capacity (total, including top/surface inch), in inches 
    :param PET: potential evapotranspiration, in inches 
    :param P: precipitation, in inches 
    :return: seven numpy arrays with values for evapotranspiration, potential recharge, recharge, runoff, 
             potential runoff, loss, and potential loss 
    """
    
    # P and PET should be in inches, flatten to a 1-D array
    PET = PET.flatten() 
    P = P.flatten()
    
    total_months = PET.shape[0]

    ET = np.zeros((total_months,))
    PR = np.zeros((total_months,))
    R = np.zeros((total_months,))
    Rs = np.zeros((total_months,))
    Ru = np.zeros((total_months,))
    RO = np.zeros((total_months,))
    PRO = np.zeros((total_months,))
    S = np.zeros((total_months,))
    Ss = np.zeros((total_months,))
    Su = np.zeros((total_months,))
    L = np.zeros((total_months,))
    Ls = np.zeros((total_months,))
    Lu = np.zeros((total_months,))
    PL = np.zeros((total_months,))
    PLs = np.zeros((total_months,))
    PLu = np.zeros((total_months,))
    
    # A is the difference between the soil moisture in the surface soil layer and the potential evapotranspiration.
    A = np.zeros((total_months,))
        
    # B is the difference between the precipitation and potential evapotranspiration, i.e. the excess precipitation
    B = np.zeros((total_months,))

    # C is the amount of room (in inches) in the surface soil layer that can be recharged with precipitation
    C = np.zeros((total_months,))

    # D is the amount of excess precipitation (in inches) that is left over after the surface soil layer is recharged
    D = np.zeros((total_months,))
    
    # E is the amount of room (in inches) in the underlying soil layer that is available to be recharged with excess precipitation
    E = np.zeros((total_months,))

    ## CONSTANTS
    
    # NOTE: SOIL MOISTURE STORAGE IS HANDLED BY DIVIDING THE SOIL INTO TWO
    # LAYERS AND ASSUMING THAT 1 INCH OF WATER CAN BE STORED IN THE SURFACE
    # LAYER. AWC IS THE COMBINED AVAILABLE MOISTURE CAPACITY IN BOTH SOIL
    # LAYERS. THE UNDERLYING LAYER HAS AN AVAILABLE CAPACITY THAT DEPENDS 
    # ON THE SOIL CHARACTERISTICS OF THE LOCATION. THE SOIL MOISTURE 
    # STORAGE WITHIN THE SURFACE LAYER (UNDERLYING LAYER) IS THE AMOUNT OF 
    # AVAILABLE MOISTURE STORED AT THE BEGINNING OF THE MONTH IN THE 
    # SURFACE (UNDERLYING) LAYER.
    
    # Ss_AWC is the available moisture capacity in the surface soil layer; it is a constant across all locations.
    Ss_AWC = 1 
    
    #!!!!!! VALIDATE !!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    # proposed fix for locations where the AWC is less than 1.0 inch
    #
    if AWC < 1.0:
        Ss_AWC = AWC
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
    

    # Su_AWC is the available moisture capacity in the underlying soil layer; it is a location-specific constant.
    Su_AWC = AWC - Ss_AWC
    
    ## INITIAL CONDITIONS
    
    # NOTE: AS THE FIRST STEP IN THE CALCULATION OF THE PALMER DROUGHT 
    # INDICES IS A WATER BALANCE, THE CALCULATION SHOULD BE INITIALIZED 
    # DURING A MONTH AND YEAR IN WHICH THE SOIL MOISTURE STORAGE CAN BE 
    # ASSUMED TO BE FULL.
    
    # S0 = AWC is the initial combined soil moisture storage 
    # in both soil layers. Within the following water balance
    # calculation loop, S0 is the soil moisture storage in
    # both soil layers at the beginning of each month.
    S0 = AWC 
    
    # Ss0 = 1 is the initial soil moisture storage in the surface 
    # soil layer. Within the following water balance calculation
    # loop, Ss0 is the soil moisture storage in the surface soil 
    # layer at the beginning of each month.
    Ss0 = 1 
    
    #!!!!!! VALIDATE !!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    # proposed fix for locations where the AWC is less than 1.0 inch
    #
    if AWC < 1.0:
        Ss0 = AWC
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    

    # Su0 = Su_AWC is the initial soil moisture storage in 
    # the underlying soil layer. Within the following 
    # water balance calculation loop, Su0 is the soil
    # moisture storage in the underlying soil layer at the 
    # beginning of each month.
    Su0 = Su_AWC
    
    ## CALCULATION OF THE WATER BALANCE
    
    # THE FIRST PART OF PALMER'S METHOD FOR CALCULATING THE PDSI INVOLVES 
    # THE CALCULATION OF  A WATER BALANCE USING HISTORIC RECORDS OF 
    # PRECIPITATION AND TEMPERATURE AND THORNTHWAITE'S METHOD.
    
    # k is the counter for each month of data on record
    for k in range(total_months):

        ## VARIABLE DEFINITIONS
        
        # P is the historical, monthly precipitation for the location.
        
        # Ss is the soil moisture storage in the surface layer at the end of the month.
        
        # Su is the soil moisture storage in the underlying layer at the end of the month.
        
        # S is the combined soil moisture storage in the combined surface 
        # and underlying soil moisture storage layers at the end of the month.
        
        # ET is the actual evapotranspiration from the combined surface and underlying soil moisture storage layers.
        
        # Ls is the actual soil moisture loss from the surface soil moisture storage layer.
        
        # Lu is the actual soil moisture loss from the underlying soil moisture storage layer.
        
        # L is the actual soil moisture loss from the combined surface and underlying soil moisture storage layers.
        
        # PLs is the potential soil moisture loss from the surface soil moisture storage layer.
        
        # PLu is the potential soil moisture loss from the underlying soil moisture storage layer.
        
        # PL is the potential soil moisture loss from the combined surface and underlying soil moisture storage layers.
        
        # Rs is the actual recharge to the surface soil moisture storage layer.
        
        # Ru is the actual recharge to the underlying soil moisture storage layer.
        
        # R is the actual recharge to the combined surface and underlying soil moisture storage layers.
        
        # PR is the potential recharge to the combined surface and underlying 
        # soil moisture storage layers at the beginning of the month.
        PR[k] = AWC - S0
        
        # RO is the actual runoff from the combined surface and underlying soil moisture storage layers.
        
        # PRO is the potential runoff. According to Alley (1984),
        # PRO = AWC - PR = Ss + Su; here Ss and Su refer to those values at
        # the beginning of the month: Ss0 and Su0.
        PRO[k] = AWC - PR[k]
        
        # A is the difference between the soil moisture in the surface soil layer and the potential evapotranspiration.
        A[k] = Ss0 - PET[k]
        
        # B is the difference between the precipitation and potential
        # evapotranspiration - it is the excess precipitation.
        B[k] = P[k] - PET[k]
        
        ## INTERNAL CALCULATIONS
        
        # A >= 0 indicates that there is sufficient moisture in the surface soil layer to satisfy the PET 
        # requirement for month k. Therefore, there is potential moisture loss from only the surface soil layer.
        if A[k] >= 0: 
            PLs[k] = PET[k]         
            PLu[k] = 0
            
        else: 
            # A < 0 indicates that there is not sufficient moisture in the surface soil layer to satisfy 
            # the PET requirement for month k. Therefore, there is potential moisture loss from both the surface
            # and underlying soil layers. The equation for PLu is given in Alley (1984).
            PLs[k] = Ss0
            PLu[k] = ((PET[k] - PLs[k]) * Su0) / AWC
            
            # Su0 >= PLu indicates that there is sufficient moisture in the underlying soil layer to (along with 
            # the moisture in the surface soil layer) satisfy the PET requirement for month k; therefore, PLu is
            # as calculated according to the equation given in Alley (1984).
            if Su0 >= PLu[k]: 
                PLu[k] = ((PET[k] - PLs[k]) * Su0) / AWC
            
            else:
                # Su0 < PLu indicates that there is not sufficient moisture in the underlying soil layer to (along with  
                # the moisture in the surface soil layer) satisfy the PET requirement for month k; therefore, PLu is 
                # equal to the moisture storage in the underlying soil layer at the beginning of the month.
                PLu[k] = Su0
        
        PL[k] = PLs[k] + PLu[k]
        
        if B[k] >= 0:
            # B >= 0 indicates that there is sufficient 
            # precipitation during month k to satisfy the PET 
            # requirement for month k - i.e., there is excess 
            # precipitation. Therefore, there is no moisture loss 
            # from either soil layer.
            
            # C is the amount of room (in inches) in the
            # surface soil layer that can be recharged with
            # precipitation. Here 1 refers to the
            # approximate number of inches of moisture 
            # allocated to the surface soil layer.
            C[k] = 1 - Ss0 
            
            if C[k] >= B[k]:
                # C >= B indicates that there is AT LEAST enough room in the surface soil layer for recharge than there 
                # is excess precipitation. Therefore, precipitation will recharge ONLY the surface soil layer, and there 
                # is NO runoff and NO soil moisture loss from either soil layer.
                Rs[k] = B[k]
                Ls[k] = 0
                Ss[k] = Ss0 + Rs[k]
                Ru[k] = 0
                Lu[k] = 0
                Su[k] = Su0
                RO[k] = 0

            else:
                # C < B indicates that there is more excess precipitation than there is room in the surface soil layer  
                # for recharge. Therefore, the excess precipitation will recharge BOTH the surface soil layer and 
                # the underlying soil layer, and there is NO soil moisture loss from either soil layer.
                Rs[k] = C[k]
                Ls[k] = 0 
                Ss[k] = 1   # the approximate number of inches of moisture allocated to the surface soil layer
                D[k] = B[k] - Rs[k] # amount of excess precipitation (in inches) left over after the surface soil layer is recharged
                E[k] = Su_AWC - Su0  # amount of room (in inches) in the underlying soil layer available to be recharged with excess precipitation
                if E[k] > D[k]: 
                    # E > D indicates that there is more room in the underlying soil layer than there is excess 
                    # precipitation available after recharge to the surface soil layer. Therefore, there is no runoff.
                    Ru[k] = D[k]
                    RO[k] = 0
            
                else: 
                    # E <= D indicates that there is AT MOST enough room in the underlying soil layer for the excess
                    # precipitation available after recharge to the surface soil layer. In the case that there is enough 
                    # room, there is no runoff. In the case that there is not enough room, runoff occurs.
                    Ru[k] = E[k]
                    RO[k] = D[k] - Ru[k]

                # Since there is more excess precipitation than there is room in the surface soil layer for recharge,
                # the soil moisture storage in the underlying soil layer at the end of the month is equal to the storage  
                # at the beginning of the month plus any recharge to the underlying soil layer.
                Lu[k] = 0
                Su[k] = Su0 + Ru[k] 

            # Since there is sufficient precipitation during month k to satisfy the PET
            # requirement for month k, the actual evapotranspiration is equal to PET.
            ET[k] = PET[k] 
            
        else: 
            # B < 0 indicates that there is not sufficient precipitation
            # during month k to satisfy the PET requirement for month k -
            # i.e., there is NO excess precipitation. Therefore, soil 
            # moisture loss occurs, and there is NO runoff and NO recharge 
            # to either soil layer.
            if Ss0 >= abs(B[k]):
                # Ss0 >= abs(B) indicates that there is AT LEAST sufficient moisture in the surface soil layer at 
                # the beginning of the month k to satisfy the PET requirement for month k. Therefore, soil moisture 
                # loss occurs from ONLY the surface soil layer, and the soil moisture storage in the surface soil layer 
                # at the end of the month is equal to the storage at the beginning of the month less any loss from
                # the surface soil layer.
                Ls[k] = abs(B[k])
                Rs[k] = 0
                Ss[k] = Ss0 - Ls[k]
                Lu[k] = 0
                Ru[k] = 0
                Su[k] = Su0
            else: 
                # Ss0 < abs(B) indicates that there is NOT sufficient moisture in the surface soil layer at 
                # the beginning of month k to satisfy the PET requirement for month k. Therefore, soil moisture loss 
                # occurs from BOTH the surface and underlying soil layers, and Lu is calculated according to 
                # the equation given in Alley (1984). The soil moisture storage in the underlying soil layer 
                # at the end of the month is equal to the storage at the beginning of the month less the loss from
                # the underlying soil layer.
                Ls[k] = Ss0
                Rs[k] = 0
                Ss[k] = 0
                Lu[k] = min(((abs(B[k]) - Ls[k]) * Su0) / AWC, Su0)
                #*
                #
                # Lu[k] = min((abs(B[k]) - Ls[k])*Su0/(AWC + 1),Su0);
                # NOTE: This equation above was used by the NCDC in their FORTRAN code (pdi.f)
                # prior to 2013. See Jacobi et al. (2013) for a full explanation. 
                #
                #*
                Ru[k] = 0
                Su[k] = Su0 - Lu[k]

            # Since there is NOT sufficient precipitation during month k to satisfy the PET requirement for month k, 
            # the actual evapotranspiration is equal to precipitation plus any soil moisture loss from BOTH the surface
            # and underlying soil layers.
            RO[k] = 0
            ET[k] = P[k] + Ls[k] + Lu[k] 
            
        R[k] = Rs[k] + Ru[k]
        L[k] = Ls[k] + Lu[k]
        S[k] = Ss[k] + Su[k]
        
        # S0, Ss0, and Su0 are reset to their end of the current month [k]
        # values - S, Ss, and Su0, respectively - such that they can be
        # used as the beginning of the month values for the next month 
        # (k + 1).
        S0 = S[k]
        Ss0 = Ss[k]
        Su0 = Su[k]
        
    return ET, PR, R, RO, PRO, L, PL 
          
#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit    # not working yet 
def _cafec_coefficients(P,
                        PET,
                        ET,
                        PR,
                        R,
                        RO,
                        PRO,
                        L,
                        PL,
                        data_start_year,
                        calibration_start_year,
                        calibration_end_year):
    '''
    This function calculates CAFEC coefficients used for computing Palmer's Z index using inputs from 
    the water balance function.
    
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
    :param data_start_year: initial year of the input arrays, i.e. the first element of each of the input arrays 
                            is assumed to correspond to January of this initial year
    :param calibration_start_year: initial year of the calibration period, should be >= data_start_year
    :param calibration_end_year: final year of the calibration period
    :return 1-D numpy.ndarray of Z-Index values, with shape corresponding to the input arrays
    :rtype: numpy.ndarray of floats
    '''
    
    # get only the data from within the calibration period
    [P, PET, ET, PR, R, PRO, RO, PL, L] = _calibrate_data([P, PET, ET, PR, R, PRO, RO, PL, L],
                                                          data_start_year,
                                                          calibration_start_year,
                                                          calibration_end_year)

    # ALPHA, BETA, GAMMA, DELTA CALCULATIONS
    # A calibration period is used to calculate alpha, beta, gamma, and delta, four coefficients dependent upon 
    # the climate of the area being examined. The NCDC and CPC use the calibration period January 1931
    # through December 1990 (cf. Karl, 1986; Journal of Climate and Applied Meteorology, Vol. 25, No. 1, January 1986).
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # get averages for each calendar month (compute means over the year axis, giving an average for each calendar month over all years)
        ET_bar = np.nanmean(ET, axis=0)
        PET_bar = np.nanmean(PET, axis=0)
        R_bar = np.nanmean(R, axis=0)
        PR_bar = np.nanmean(PR, axis=0)
        L_bar = np.nanmean(L, axis=0)
        PL_bar = np.nanmean(PL, axis=0)
        RO_bar = np.nanmean(RO, axis=0)
        PRO_bar = np.nanmean(PRO, axis=0)
            
        # (calendar) monthly CAFEC coefficients
        alpha = np.empty((12,))
        beta = np.empty((12,))
        gamma = np.empty((12,))
        delta = np.empty((12,))
    
        # compute the alpha, beta, gamma, and delta coefficients for each calendar month
        for i in range(12):
            
            # calculate alpha
            if PET_bar[i] == 0:
                if ET_bar[i] == 0:
                    alpha[i] = 1
                else:
                    alpha[i] = 0
                    #logger.warn('CHECK DATA: PET is less than ET.')
            else:
                alpha[i] = ET_bar[i] / PET_bar[i]
    
            # calculate beta
            if PR_bar[i] == 0:
                if R_bar[i] == 0:
                    beta[i] = 1
                else:
                    beta[i] = 0
                    #logger.warn('CHECK DATA: PR is less than R.')
            else:
                beta[i] = R_bar[i] / PR_bar[i]
    
            # calculate gamma
            if PRO_bar[i] == 0:
                if RO_bar[i] == 0:
                    gamma[i] = 1
                else:
                    gamma[i] = 0
                    #logger.warn('CHECK DATA: PRO is less than RO.')
            else:
                gamma[i] = RO_bar[i] / PRO_bar[i]
    
            # calculate delta
            if PL_bar[i] == 0:
                if L_bar[i] == 0:
                    delta[i] = 1
                else:
                    delta[i] = 0
                    #logger.warn('CHECK DATA: PL is less than L.')
            else:
                delta[i] = L_bar[i] / PL_bar[i]

    return alpha, beta, gamma, delta

#-----------------------------------------------------------------------------------------------------------------------    
@numba.jit  # this may not work well on Linux, needed to comment out this on climgrid-dev
def _calibrate_data(arrays,
                    data_start_year,
                    calibration_start_year,
                    calibration_end_year):

    #!!!!!!!!!!!!!
    # TODO make sure calibration years range is valid, i.e. within actual data years range 
    
    # determine the array (year axis) indices for the calibration period
    total_data_years = int(arrays[0].shape[0] / 12)
    data_end_year = data_start_year + total_data_years - 1
    calibration_start_year_index = calibration_start_year - data_start_year
    calibration_end_year_index = calibration_end_year - data_start_year 

    # for each array pull out the calibration period
    calibration_arrays = []
    for data_array in arrays:
        data_array = utils.reshape_to_years_months(data_array)
    
        # get calibration period arrays
        if (calibration_start_year > data_start_year) or (calibration_end_year < data_end_year):
            data_array = data_array[calibration_start_year_index:calibration_end_year_index + 1, :]
        
        # add to the list of calibrated arrays we'll return
        calibration_arrays.append(data_array)
        
    return calibration_arrays

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _climatic_characteristic(alpha,
                             beta,
                             gamma,
                             delta,
                             P,
                             ET,
                             PET,
                             R,
                             PR,
                             RO,
                             PRO,
                             L,
                             PL,
                             data_start_year,
                             calibration_start_year,
                             calibration_end_year):
    
    total_calibration_years = calibration_end_year - calibration_start_year + 1
    
    # get only the data from within the calibration period
    [P, PET, ET, PR, R, PRO, RO, PL, L] = _calibrate_data([P, PET, ET, PR, R, PRO, RO, PL, L],
                                                          data_start_year,
                                                          calibration_start_year,
                                                          calibration_end_year)
    
    # CALIBRATED CAFEC, K, AND d CALCULATION
    # NOTE: 
    # The Z index is calculated with a calibrated K (weighting factor) but a full record d (difference between actual 
    # precipitation and CAFEC precipitation). CAFEC precipitation is calculated analogously to a simple water balance,
    # where precipitation is equal to evaporation plus runoff (and ground water recharge) plus or minus any change 
    # in soil moisture storage. 
    d_hat = np.empty((total_calibration_years, 12)) 
    for k in range(total_calibration_years):
        for i in range(12):
            # CAFEC_hat is calculated for month i of year k of the calibration period.
            CAFEC_hat = (alpha[i] * PET[k, i]) + \
                        (beta[i] * PR[k, i]) + \
                        (gamma[i] * PRO[k, i]) - \
                        (delta[i] * PL[k, i])
                              
            # Calculate d_hat, the difference between actual precipitation
            # and CAFEC precipitation for month i of year k of the calibration period.
            d_hat[k, i] = P[k, i] - CAFEC_hat
    
    # NOTE: D_hat, T_hat, K_hat, and z_hat are all calibrated
    # variables - i.e., they are calculated only for the calibration period.
    D_hat = np.empty((12,)) 
    T_hat = np.empty((12,)) 
    K_hat = np.empty((12,)) 
    z_hat_m = np.empty((12,)) 
    P_bar = np.nanmean(P, axis=0)
    PET_bar = np.nanmean(PET, axis=0)
    R_bar = np.nanmean(R, axis=0)
    L_bar = np.nanmean(L, axis=0)
    RO_bar = np.nanmean(RO, axis=0)
            
    for i in range(12):
                    
        # Calculate D_hat, the average of the absolute values of d_hat for month i.
        D_hat[i] = np.nanmean(np.absolute(d_hat[:, i]))

        # Calculate T_hat, a measure of the ratio of "moisture demand" to "moisture supply" for month i
        #TODO if this value evaluates to a negative number less than -2.8 then the following equation for K_hat 
        # will result in a math domain error -- is it valid here to limit this value to -2.8 or greater? 
        T_hat[i] = (PET_bar[i] + R_bar[i] + RO_bar[i]) / (P_bar[i] + L_bar[i])
        
        # Calculate K_hat, the denominator of the K equation for month i.
        # from figure 3, Palmer 1965
        K_hat[i] = 1.5 * math.log10((T_hat[i] + 2.8) / D_hat[i]) + .50
        
        # Calculate z_hat, the numerator of the K equation for month i.
        z_hat_m[i] = D_hat[i] * K_hat[i]
    
    # sum the monthly Z-hat values
    z_hat = sum(z_hat_m)
    
    # Calculate the weighting factor, K, using the calibrated variables K_hat and z_hat. The purpose of
    # the weighting factors is to adjust the  departures from normal precipitation d (calculated below), 
    # such that they are comparable among different locations and for different months. The K tends to be
    # large in arid regions and small in humid regions (cf. Alley, 1984; Journal of Climate and Applied 
    # Meteorology, Vol. 23, No. 7, July 1984).
    K = np.empty((12,)) 
    for i in range(12):
        K[i] = (17.67 * K_hat[i]) / z_hat

    return K

#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit   # not working yet
def _z_index(P,
             PET,
             ET,
             PR,
             R,
             RO,
             PRO,
             L,
             PL,
             data_start_year,
             calibration_start_year,
             calibration_end_year):
    '''
    This function calculates Palmer's Z index using inputs from the water balance function.
    
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
    :param data_start_year: initial year of the input arrays, i.e. the first element of each of the input arrays 
                            is assumed to correspond to January of this initial year
    :param calibration_start_year: initial year of the calibration period, should be >= data_start_year
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
        
    # get the CAFEC coefficients
    alpha, beta, gamma, delta = _cafec_coefficients(P, 
                                                    PET, 
                                                    ET,
                                                    PR,
                                                    R,
                                                    RO,
                                                    PRO,
                                                    L, 
                                                    PL, 
                                                    data_start_year, 
                                                    calibration_start_year, 
                                                    calibration_end_year)
    # get the weighting factor K        
    K = _climatic_characteristic(alpha, 
                                 beta, 
                                 gamma, 
                                 delta, 
                                 P, 
                                 ET, 
                                 PET, 
                                 R, 
                                 PR, 
                                 RO, 
                                 PRO, 
                                 L, 
                                 PL, 
                                 data_start_year,
                                 calibration_start_year,
                                 calibration_end_year)
    
    # loop over the full period of record and compute the CAFEC precipitation, and use this to determine the moisture departure
    # FULL RECORD CAFEC AND d CALCULATION
    CAFEC = np.empty((P.shape[0], 12))
    z = np.empty((P.shape[0], 12))
    for n in range(P.shape[0]):
        for i in range(12):
            
            # calculate the CAFEC precipitation
            CAFEC[n, i] = (alpha[i] * PET[n, i]) + \
                          (beta[i] * PR[n, i]) + \
                          (gamma[i] * PRO[n, i]) - \
                          (delta[i] * PL[n, i])
            
            # Calculate d_hat, difference between actual precipitation and CAFEC precipitation
            departure = P[n, i] - CAFEC[n, i]
            
            # Calculate the Z-index (moisture anomaly index)
            z[n, i] = K[i] * departure

    # return the Z-Index values as a 1-D array
    return z.flatten()

#-----------------------------------------------------------------------------------------------------------------------
# previously Main()
@numba.jit
def _compute_X(Z, k, PV, PPe, X1, X2, PX1, PX2, PX3, X, BT):

    # This function calculates PX1 and PX2 and calls the backtracking loop.
    # If the absolute value of PX1 or PX2 goes over 1, that value becomes the new PX3. 
    
    # Calculate the current PX1 and PX2.
    PX1[k] = max(0.0, 0.897 * X1 + (Z[k] / 3))
    PX2[k] = min(0.0, 0.897 * X2 + (Z[k] / 3))
    
    if (PX1[k] >= 1) and (PX3[k] == 0):
        # When PX1 >= 1 the wet spell becomes established. X is assigned as PX1
        # and PX3 = PX1. PX1 is set to zero after PX3 is set to PX1. BT is set
        # to 1 and the _backtrack() function is called to begin backtracking up
        # PX1.
        X[k] = PX1[k]              
        PX3[k] = PX1[k]            
        PX1[k] = 0
        BT[k] = 1
        X, BT = _backtrack(k, PPe, PX1, PX2, PX3, X, BT)                                                             
    
    elif (PX2[k] <= -1) and (PX3[k] == 0):
        # When PX2 <= -1 the drought becomes established. X is assigned as PX2
        # and PX3 = PX2. PX2 is set to zero after PX3 is set to PX2. BT is set
        # to 2 and the _backtrack() function is called to begin backtracking up PX2.
        X[k] = PX2[k]
        PX3[k] = PX2[k]
        PX2[k] = 0
        BT[k] = 2
        X, BT = _backtrack(k, PPe, PX1, PX2, PX3, X, BT)                                                             
    
    elif PX3[k] == 0:
        # When PX3 is zero and both |PX1| and |PX2| are less than 1, there is
        # no established drought or wet spell. X is set to whatever PX1 or PX2
        # value is not equal to zero. BT is set to either 1 or 2 depending on
        # which PX1 or PX2 value equals zero. The _backtrack() function is called
        # to begin backtracking up either PX1 or PX2 depending on the BT value.
        if PX1[k] == 0:
        
            X[k] = PX2[k]
            BT[k] = 2
            X, BT = _backtrack(k, PPe, PX1, PX2, PX3, X, BT)                                                                 

        elif PX2[k] == 0:
            
            X[k] = PX1[k]
            BT[k] = 1
            X, BT = _backtrack(k, PPe, PX1, PX2, PX3, X, BT)                                                          
    else:
        # There is no determined value to assign to X when PX3 ~= 0, 
        # 0 <= PX1 < 1, and -1 < PX2 <= 0 so set X = PX3 for now.   
        X[k] = PX3[k]   # BT[k] stays zero, no update from its initial value, this will trigger backtracking to assign either X1 or X2 values depending upon the previous value of BT

    return PX1, PX2, PX3, X, BT

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _backtrack(k, 
               PPe, 
               PX1, 
               PX2, 
               PX3, 
               X, 
               BT):
    '''
    This function steps through stored index values computed for previous months and if/when . 
    
    Backtracking occurs in two instances: 
    
    (1) after the probability reaches 100 percent, and 
    (2) when the probability is zero. 
    In either case, the backtracking function works by backtracking through PX1 and PX2 until reaching a month 
    where PPe == 0. Either PX1 or PX2 is assigned to X as the backtracking progresses.
    '''
    # Backtracking occurs from either PPe[k] = 100 or PPe[k] = 0 to the first 
    # instance in the previous record where PPe = 0. This loop counts back
    # through previous PPe values to find the first instance where PPe = 0.
    # r is a variable used to mark the place of the last PPe = 0 before PPe = 100.
    r = 0
    for c in range(k, 0, -1):
        if PPe[c] == 0:
            r = c
            break
    
    # Backtrack from either PPe = 100 or PPe = 0 to the last instance of 
    # non-zero, non-one hundred probability.
    for count in range(k, r, -1):
        # When PPe[k] = 100 and |PX3| > 1 set X[k] = PX3[k].
        #                                                                       
        # Set the BT value of the previous month to either 1 or 2 based on the
        # sign of PX3[k]. If PX3[k] is negative, a BT = 2 begins backtracking 
        # up X2 and vice versa.
        if (PPe[count] == 100) and (abs(PX3[count]) > 1):
            X[count] = PX3[count]
            if PX3[count] < 0:
                BT[count - 1] = 2
            else:
                BT[count - 1] = 1
        
        # Everything below deals with months where PPe is not equal to 100. 
        # Based on the assigned BT value, start in either PX1 or PX2. If
        # that value is not 0, assign X and set the BT value for the preceding
        # month to 1 if X = PX1 or 2 if X = PX2. If BT = 1 and PX1 = 0, assign 
        # X to PX2 and set the BT value for the preceding month to 2 and vice
        # versa. Continue this process of backtracking up either PX1 or PX2
        # and switching when either PX1 or PX2 equals 0 or until the end of the
        # loop is reached.
        elif BT[count] == 2:
            if PX2[count] == 0:
                X[count] = PX1[count]
                BT[count] = 1
                BT[count - 1] = 1
            else:
                X[count] = PX2[count]
                BT[count - 1] = 2
        elif BT[count] == 1:
            if PX1[count] == 0:
                X[count] = PX2[count]
                BT[count] = 2
                BT[count - 1] = 2
            else:
                X[count] = PX1[count]
                BT[count - 1] = 1
    
    return X, BT

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _between_0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X):

    # This function is called when non-zero, non-one hundred PPe values occur
    # between values of PPe = 0. When this happens, a possible abatement
    # discontinues without ending the wet spell or drought. X should be
    # assigned to PX3 for all months between, and including, the two instances
    # of PPe = 0 (cf. Alley, 1984; Journal of Climate and Applied Meteorology, 
    # Vol. 23, No. 7). To do this, _backtrack() up to the first instance of 
    # PPe = 0 while setting X to PX3. 
    
    # Since the possible abatement has ended, the drought or wet spell
    # continues. Set PV, PX1, PX2, and PPe to 0. Calculate PX3 and set X = PX3.
    # Set BT=3 in preparation for backtracking.
    PV = 0                         
    PX1[k] = 0                     
    PX2[k] = 0                     
    PPe[k] = 0                     
    PX3[k] = 0.897 * X3 + (Z[k] / 3)
    X[k] = PX3[k]
    BT[k] = 3  # use the X3 value for this month if/when backtracking occurs
    
    # In order to set all values of X between the two instances of PPe = 0, the
    # first instance of PPe = 0 must be found. This "for" loop counts back 
    # through previous PPe values to find the first instance where PPe = 0.
    for count1 in range(k, 0, -1):
        if PPe[count1] == 0:
            r = count1
            break

    #FIXME r could be undefined at this point if the above loop never enters the PPe[count1] == 0 condition

    # Backtrack from the current month where PPe = 0 to the last month where PPe = 0.
    for count in range(k, r - 1, -1):
        # Set X = PX3, where the BT array indicates it for the month
        if BT[count] == 3:
            X[count] = PX3[count]
            # If the end of the backtracking loop hasn't been reached, set the
            # BT value for the preceding month to 3 to continue the backtracking.
            if count != r:
                BT[count - 1] = 3  # this makes it so that the previous if condition will be met for the previous month (the next backtracking step), so that it too will be assigned an X3 value; by this mechanism we enable the assignment of X3 values all the way back to index r

    return PV, PX1, PX2, PX3, PPe, X, BT

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _dry_spell_abatement(k, Z, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT):

    # In the case of an established drought, Palmer (1965) notes that a value of Z = -0.15 will maintain an
    # index of -0.50 from month to month. An established drought or wet spell is considered definitely over
    # when the index reaches the "near normal" category which lies between -0.50 and +0.50. Therefore, any
    # value of Z >= -0.15 will tend to end a drought. See Palmer 1965, pp. 29-30
    Uw = Z[k] + 0.15  # Palmer 1965, eq. 29
    
    PV = Uw + max(V, 0)  # add this month's effective wetness to the previously accumulated effective wetness (if it was positive), this value will eventually be used as the numerator in the equation for calculating percentage probability that an established weather spell has ended (Pe, eq. 30, Palmer 1965)
    if PV <= 0:
        # During a drought, PV <= 0 implies PPe = 0 (i.e., the 
        # probability that the drought has ended returns to zero).                                                             
        Q = 0 
        PV, PX1, PX2, PX3, PPe, X, BT = _between_0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)                                                                 
    
    else:
        Ze = (-2.691 * X3) - 1.5  # (Palmer eq. 28) Calculate the soil moisture anomaly (Z value) which corresponds to an amount of moisture that is sufficient to end the currently established drought in a single month. Once this is known then we can compare against the actual Z value, to see if we've pulled out of the established drought or not
        if Pe == 100: 
            Q = Ze  # Q is the total moisture anomaly required to end the current drought.
        else:
            Q = Ze + V

        PPe[k] = (PV / Q) * 100
    
        if PPe[k] >= 100:
            PPe[k] = 100
            PX3[k] = 0
        else:
            PX3[k] = 0.897 * X3 + (Z[k] / 3)

        PX1, PX2, PX3, X, BT = _compute_X(Z, k, PV, PPe, X1, X2, PX1, PX2, PX3, X, BT)                                                                 

    return PV, PPe, PX1, PX2, PX3, X, BT

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _wet_spell_abatement(k, Z, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT):

    # In the case of an established wet spell, Palmer (1965) notes that a value of Z = +0.15 will maintain an 
    # index of +0.50 from month to month. An established drought or wet spell is considered definitely over
    # when the index reaches the "near normal" category which lies between -0.50 and +0.50. Therefore, any 
    # value of Z <= +0.15 will tend to end a wet spell. See Palmer 1965, pp. 29-30
    Ud = Z[k] - 0.15  # Palmer 1965, eq. 32
    
    PV = Ud + min(V, 0)
    if PV >= 0: 
        # During a wet spell, PV >= 0 implies PPe = 0 (i.e., the 
        # probability that the wet spell has ended returns to zero).
        Q = 0 
        PV, PX1, PX2, PX3, PPe, X, BT = _between_0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)                                                           
    
    else:
        Ze = -2.691 * X3 + 1.5
        if Pe == 100:
            Q = Ze      # Q is the total moisture anomaly required to end the current wet spell.
        else:
            Q = Ze + V

        PPe[k] = (PV / Q) * 100
        if PPe[k] >= 100:
            PPe[k] = 100
            PX3[k] = 0
        else:
            PX3[k] = 0.897 * X3 + (Z[k] / 3)

        PX1, PX2, PX3, X, BT = _compute_X(Z, k, PV, PPe, X1, X2, PX1, PX2, PX3, X, BT)                                                                 

    return PV, PPe, PX1, PX2, PX3, X, BT

#-----------------------------------------------------------------------------------------------------------------------
# comparable to the case() subroutine in original NCDC pdi.f 
@numba.jit
def _pmdi(probability,
          X1, 
          X2, 
          X3):
    
    # the index is near normal and either a dry or wet spell exists, choose the largest absolute value of X1 or X2
    if X3 == 0:
        
        if abs(X2) > abs(X1):
            _pmdi = X2
        else:
            _pmdi = X1   
    
    else:
        if (probability > 0) and (probability < 100):
    
            PRO = probability / 100.0
            if X3 <= 0:
                # use the weighted sum of X3 and X1
                _pmdi = ((1.0 - PRO) * X3) + (PRO * X1)
            
            else:
                # use the weighted sum of X3 and X2
                _pmdi = ((1.0 - PRO) * X3) + (PRO * X2)
        else:
            # a weather spell is established
            _pmdi = X3

    return _pmdi

#------------------------------------------------------------------------------------------------------------------
#@numba.jit    # not working yet
def _pdsi_from_zindex(Z):

    ## INITIALIZE PDSI AND PHDI CALCULATIONS
    
    # V is the sum of the Uw (Ud) values for the current and previous months of an
    # established dry (wet) spell and is used in calculating the Pe value for a month.
    V = 0.0
    Pe = 0.0 # Pe is the probability that the current wet or dry spell has ended in a month.
    X1 = 0.0 # X1 is the severity index value for an incipient wet spell for a month.
    X2 = 0.0 # X2 is the severity index value for an incipient dry spell for a month.
    X3 = 0.0 # X3 is the severity index value of the current established wet or dry spell for a month.
    
    number_of_months = Z.shape[0]
    
    # BACTRACKING VARIABLES
    
    # BT is the backtracking variable, and is pre-allocated with zeros. Its value (1, 2, or 3) indicates which 
    # intermediate index (X1, X2, or X3) to backtrack up, selecting the associated term (X1, X2, or X3) for the PDSI. 
    # NOTE: BT may be operationally left equal to 0, as it cannot be known in real time when an existing drought or wet spell may or may not be over.
    BT = np.zeros((number_of_months,), dtype=np.int8) 
    
    ## CALCULATE PDSI AND PHDI
    PX1 = np.zeros((number_of_months,))
    PX2 = np.zeros((number_of_months,))
    PX3 = np.zeros((number_of_months,))
    PPe = np.zeros((number_of_months,))
    X = np.zeros((number_of_months,))
    PMDI = np.zeros((number_of_months,))
    
    # Palmer Hydrological Drought Index
    PHDI = np.zeros((number_of_months,))

    # loop over all months in the dataset, calculating PDSI and PHDI for each
    for k in range(number_of_months):
        
#         # DEBUGGING ONLY -- REMOVE
#         print('k: {0}'.format(k))
        
        if (Pe == 100) or (Pe == 0):   # no abatement underway
            
            if abs(X3) <= 0.5:   # drought or wet spell ends
                
                # PV is the preliminary V value and is used in operational calculations.
                PV = 0 
                
                # PPe is the preliminary Pe value and is used in operational calculations.
                PPe[k] = 0 
                
                # PX3 is the preliminary X3 value and is used in operational calculations.
                PX3[k] = 0 
                                
                PX1, PX2, PX3, X, BT = _compute_X(Z, k, PV, PPe, X1, X2, PX1, PX2, PX3, X, BT)
 
            elif X3 > 0.5: # Wet spell underway
                
                if Z[k] >= 0.15: # Wet spell intensifies
                
                    PV, PX1, PX2, PX3, PPe, X, BT = _between_0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)                                                  
                
                else: # Wet spell starts to abate, and it may end.
                
                    PV, PPe, PX1, PX2, PX3, X, BT = _wet_spell_abatement(k, Z, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                                                                                               
            elif X3 < -0.5: # Drought underway
                
                if Z[k] <= -0.15: # Drought intensifies 
                
                    PV, PX1, PX2, PX3, PPe, X, BT = _between_0s(k, Z, X3, PX1, PX2, PX3, PPe, BT, X)                                                 
                
                else: # Drought starts to abate, and it may end.
                
                    PV, PPe, PX1, PX2, PX3, X, BT = _dry_spell_abatement(k, Z, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
                                                                                          
        else: # Abatement underway
            
            if X3 > 0: # Wet spell underway
                
                PV, PPe, PX1, PX2, PX3, X, BT = _wet_spell_abatement(k, Z, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)                                                     
            
            else: # Drought underway
                
                PV, PPe, PX1, PX2, PX3, X, BT = _dry_spell_abatement(k, Z, V, Pe, PPe, PX1, PX2, PX3, X1, X2, X3, X, BT)
        
        # select the PMDI value
        PMDI[k] = _pmdi(Pe, X1, X2, X3)

        # FIXME PV could be undefined at this point if the above loop never assigns to the variable

        ## Assign V, Pe, X1, X2, and X3 for use with the next month
        V = PV
        Pe = PPe[k]
        X1 = PX1[k]
        X2 = PX2[k]
        X3 = PX3[k]
        
        ## ASSIGN X FOR CASES WHERE PX3 AND BT EQUAL ZERO
        # NOTE: This is a conflicting case that arises where X cannot be
        # assigned as X1, X2, or X3 in real time. Here 0 < PX1 < 1, 
        # -1 < PX2 < 0, and PX3 = 0, and it is not obvious which
        # intermediate index should be assigned to X. Therefore,
        # backtracking is used here, where BT is set equal to the next
        # month's BT value and X is assigned to the intermediate index
        # associated with that BT value.
        if k > 0:
            if (PX3[k - 1] == 0) and (BT[k - 1] == 0):
                r = 0
                for c in range(k - 1, 0, -1):  # here we loop over the BT array to look for the most previous month step where the value is not zero
                    if BT[c] != 0:
                        # Backtracking continues in a backstepping procedure up through the most previous month where BT is not equal to zero
                        r = c + 1    # r is the row number up through which backtracking continues.
                        break

                for count0 in range(k - 1, r - 1, -1):  # here we loop over the BT array from the previous month (k - 1) through the r index (most previous month with BT != 0), at each month assigning to X the value for the month called for in the BT array, unless that value is 0 in which case the BT value is switched and the corresponding X values are assigned (see _assign() in pdinew.f/pdinew.py)
                    BT[count0] = BT[count0 + 1] # Assign BT to next month's BT value.
                    if BT[count0] == 2:
                        if PX2[count0] == 0:  # If BT = 2, X = PX2 unless PX2 = 0, then X = PX1.
                            X[count0] = PX1[count0]
                            BT[count0] = 1  # flip the X we'll choose next step, from X2 to X1
                        else:
                            X[count0] = PX2[count0]
                    elif BT[count0] == 1:
                        if PX1[count0] == 0:  # If BT = 1, X = PX1 unless PX1 = 0, then X = PX2.
                            X[count0] = PX2[count0] 
                            BT[count0] = 2  # flip the X we'll choose next step, from X1 to X2
                        else:
                            X[count0] = PX1[count0]
                    
        # In instances where there is no established spell for the last monthly observation, X is initially 
        # assigned to 0. The code below sets X in the last month to greater of |PX1| or |PX2|. This prevents 
        # the PHDI from being inappropriately set to 0. 
        if k == (number_of_months - 1):
            if (PX3[k] == 0) and (X[k] == 0):
                if abs(PX1[k]) > abs(PX2[k]):
                    X[k] = PX1[k]
                else:
                    X[k] = PX2[k]

        # round values to four decimal places
        for values in [X1, X2, X3, Pe, V, X, PX1, PX2, PX3, PPe]:
            values = np.around(values, decimals=4)
        
    ## ASSIGN PDSI VALUES
    # NOTE: 
    # In Palmer's effort to create a meteorological drought index (PDSI),
    # Palmer expressed the beginning and ending of dry (or wet) periods in
    # terms of the probability that the spell has started or ended (Pe). A
    # drought (wet spell) is definitely over when the probability reaches
    # or exceeds 100%, but the drought (wet spell) is considered to have
    # ended the first month when the probability becomes greater than 0%
    # and then continues to remain greater than 0% until it reaches 100% 
    # (cf. Palmer, 1965; US Weather Bureau Research Paper 45).
    PDSI = X
    
    ## ASSIGN PHDI VALUES
    # NOTE:
    # There is a lag between the time that the drought-inducing
    # meteorological conditions end and the environment recovers from a
    # drought. Palmer made this distinction by computing a meteorological
    # drought index (described above) and a hydrological drought index. The
    # X3 term changes more slowly than the values of the incipient (X1 and
    # X2) terms. The X3 term is the index for the long-term hydrologic
    # moisture condition and is the PHDI.
    for s in range(len(PX3)):
        if PX3[s] == 0:
            # For calculation and program advancement purposes, the PX3 term is sometimes set equal to 0. 
            # In such instances, the PHDI is set equal to X (the PDSI), which accurately reflects the X3 value.
            PHDI[s] = X[s]
        else:
            PHDI[s] = PX3[s]
    
    # return the computed variables
    return PDSI, PHDI, PMDI

#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit  # not working yet
def _compute_scpdsi(established_index_values,
                    sczindex_values,
                    scpdsi_values,
                    pdsi_values,
                    wet_index_values,
                    dry_index_values,
                    wet_M,
                    wet_B,
                    dry_M,
                    dry_B,
                    calibration_complete,
                    tolerance=0.0):
    '''
    This function computes X values
    :param established_index_values
    :param sczindex_values
    :param scpdsi_values
    :param pdsi_values
    :param wet_index_values
    :param dry_index_values
    :param wet_M
    :param wet_B
    :param dry_M
    :param dry_B
    :param calibration_complete
    :param tolerance
     '''
    # empty all X lists
    wet_index_deque = deque([])
    dry_index_deque = deque([])

    # Initializes the book keeping indices used in finding the PDSI
    V = 0.0
    Q = 0.0

    period = 0
    previous_key = -1

    for period in range(established_index_values.size):
    
        # These variables represent the values for  corresponding variables for the current period.
        # They are kept separate because many calculations depend on last period's values.  
        new_X = 0
        new_X1 = 0
        new_X2 = 0
        new_X3 = 0
        previous_established_index_X3 = 0

#         # ZE is the Z value needed to end an established spell
#         ZE
#         m
#         b
#         c
# 
#         # wd is a sign changing flag.  It allows for use of the same equations during both a wet or dry spell by adjusting the appropriate signs.
#         wd

        if (previous_key >= 0) and not np.isnan(established_index_values[previous_key]):
        
            previous_established_index_X3 = established_index_values[previous_key]

        if previous_established_index_X3 >= 0:
        
            m = wet_M
            b = wet_B
        
        else:
        
            m = dry_M
            b = dry_B
        
        if not np.isnan(sczindex_values[period]) and ((m + b) != 0):

            c = 1 - (m / (m + b))

            # This sets the wd flag by looking at EstablishedIndex
            if previous_established_index_X3 >= 0:
                wd = 1
            else:
                wd = -1

            # If EstablishedIndex is 0 then there is no reason to calculate Q or ZE, V and Prob are reset to 0;
            if previous_established_index_X3 == 0:
            
                new_X3 = 0
                new_V = 0
                new_probability = 0
                new_X, new_X1, new_X2, new_X3 = _choose_X(pdsi_values,
                                                          established_index_values,
                                                          wet_index_values,
                                                          dry_index_values,
                                                          sczindex_values,
                                                          wet_index_deque,
                                                          dry_index_deque,
                                                          wet_M,
                                                          wet_B,
                                                          dry_M,
                                                          dry_B,
                                                          new_X, 
                                                          new_X1, 
                                                          new_X2, 
                                                          new_X3, 
                                                          period, 
                                                          previous_key)

            # Otherwise all calculations are needed.
            else:
                                
                new_X3 = (c * previous_established_index_X3 + sczindex_values[period] / (m + b))
                # ZE is the Z value needed to end an established spell
                ZE = (m + b) * (wd * 0.5 - c * previous_established_index_X3)
                Q = ZE + V
                new_V = sczindex_values[period] - wd * (m * 0.5) + wd * min(wd * V + tolerance, 0)

                if (wd * new_V) > 0:
                
                    new_V = 0
                    new_probability = 0
                    new_X1 = 0
                    new_X2 = 0
                    new_X = new_X3

                    wet_index_deque.clear()
                    dry_index_deque.clear()
                
                else:

                    new_probability = (new_V / Q) * 100
                    if new_probability >= (100 - tolerance):

                        new_X3 = 0
                        new_V = 0
                        new_probability = 100

                    # xValues should be a list of doubles
                    new_X, new_X1, new_X2, new_X3 = _choose_X(pdsi_values,
                                                          established_index_values,
                                                          wet_index_values,
                                                          dry_index_values,
                                                          sczindex_values,
                                                          wet_index_deque,
                                                          dry_index_deque,
                                                          wet_M,
                                                          wet_B,
                                                          dry_M,
                                                          dry_B,
                                                          new_X, 
                                                          new_X1, 
                                                          new_X2, 
                                                          new_X3, 
                                                          period, 
                                                          previous_key)

            wet_index_values[period] = new_X1
            dry_index_values[period] = new_X2
            established_index_values[period] = new_X3
            
            if calibration_complete:
                scpdsi_values[period] = new_X
            else:
                pdsi_values[period] = new_X
            
            # update variables for next month:
            V = new_V
        
        else:
        
            # This month's data is missing, so output MISSING as PDSI.  All variables used in calculating the PDSI are kept from the previous month.  
            # Only the linked lists are changed to make sure that if backtracking occurs, a MISSING value is kept as the PDSI for this month.
            pdsi_values[period] = np.NaN
            wet_index_values[period] = np.NaN
            dry_index_values[period] = np.NaN
            established_index_values[period] = np.NaN
            if calibration_complete:
                scpdsi_values[period] = np.NaN
            else:
                pdsi_values[period] = np.NaN

        previous_key = period
        period += 1

    return pdsi_values, scpdsi_values, wet_index_values, dry_index_values, established_index_values

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _choose_X(pdsi_values,
              established_index_values,
              wet_index_values,
              dry_index_values,
              sczindex_values,
              wet_index_deque,
              dry_index_deque,
              wet_M,
              wet_B,
              dry_M,
              dry_B,
              new_X,
              new_X1,
              new_X2,
              new_X3,
              month_index, 
              previous_key,
              tolerance=0.0):

    previous_wet_index_X1 = 0
    previous_dry_index_X2 = 0

    if (previous_key >= 0) and not np.isnan(established_index_values[previous_key]):
    
        previous_wet_index_X1 = wet_index_values[previous_key]
        previous_dry_index_X2 = dry_index_values[previous_key]
    
    wetc = 1 - (wet_M / (wet_M + wet_B))
    dryc = 1 - (dry_M / (dry_M + dry_B))

    zIndex = sczindex_values[month_index]
    new_X1 = (wetc * previous_wet_index_X1 + zIndex / (wet_M + wet_B))
    if new_X1 < 0:
    
        new_X1 = 0.0
    
    new_X2 = (dryc * previous_dry_index_X2 + zIndex / (dry_M + dry_B))
    if new_X2 > 0:
    
        new_X2 = 0.0

    if (new_X1 >= 0.5) and (new_X3 == 0):
    
        _backtrack_self_calibrated(pdsi_values,
                                   wet_index_deque,
                                   dry_index_deque,
                                   tolerance,
                                   new_X1,
                                   new_X2,
                                   month_index)
        new_X = new_X1
        new_X3 = new_X1
        new_X1 = 0.0
    
    else:
    
        new_X2 = dryc * previous_dry_index_X2 + zIndex / (dry_M + dry_B)
        if new_X2 > 0:
        
            new_X2 = 0.0
        
        if (new_X2 <= -0.5) and (new_X3 == 0):
        
            _backtrack_self_calibrated(pdsi_values,
                                       wet_index_deque,
                                       dry_index_deque,
                                       tolerance,
                                       new_X2,
                                       new_X1,
                                       month_index)
            new_X = new_X2
            new_X3 = new_X2
            new_X2 = 0.0
        
        elif new_X3 == 0:
        
            if new_X1 == 0:
            
                _backtrack_self_calibrated(pdsi_values,
                                           wet_index_deque,
                                           dry_index_deque,
                                           tolerance,
                                           new_X2,
                                           new_X1,
                                           month_index)
                new_X = new_X2
            
            elif new_X2 == 0:
            
                _backtrack_self_calibrated(pdsi_values,
                                           wet_index_deque,
                                           dry_index_deque,
                                           tolerance,
                                           new_X1,
                                           new_X2,
                                           month_index)
                new_X = new_X1
            
            else:
            
                wet_index_deque.appendleft(new_X1)
                dry_index_deque.appendleft(new_X2)
                new_X = new_X3
        
        else:
        
            # store WetIndex and DryIndex in their linked lists for possible use later
            wet_index_deque.appendleft(new_X1)
            dry_index_deque.appendleft(new_X2)
            new_X = new_X3
    
    return new_X, new_X1, new_X2, new_X3

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _backtrack_self_calibrated(pdsi_values,
                               wet_index_deque,
                               dry_index_deque,
                               tolerance,
                               X1, 
                               X2, 
                               month_index):
    '''
    :param pdsi_values
    :param wet_index_deque
    :param dry_index_deque
    :param tolerance
    :param X1
    :param X2
    :param month_index
    '''
    
    num1 = X1

    while (len(wet_index_deque) > 0) and (len(dry_index_deque) > 0):
    
        if num1 > 0:
        
            num1 = wet_index_deque.popleft()
            num2 = dry_index_deque.popleft()
        
        else:
        
            num1 = dry_index_deque.popleft()
            num2 = wet_index_deque.popleft()

        if ((-1.0 * tolerance) <= num1) and (num1 <= tolerance):
        
            num1 = num2
        
        pdsi_values[month_index] = num1

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _z_sum(interval, 
           wet_or_dry,
           sczindex_values,
           periods_per_year,
           calibration_start_year,
           calibration_end_year,
           input_start_year):

    z = 0.0
    z_temporary = deque()
    values_to_sum = deque()
    summed_values = deque()
    
    # get only non-NaN Z-index values
    for sczindex in sczindex_values:
    
        # we need to skip Z-index values from the list if they don't exist, this can result from empty months in the final year of the data set
        if not np.isnan(sczindex):
        
            z_temporary.append(sczindex)
            
    calibration_period_initial_index = (calibration_start_year - input_start_year) * periods_per_year
    i = 0
    while (i < calibration_period_initial_index) and (len(z_temporary) > 0):
        
        # remove periods before the start of the calibration interval
        z_temporary.pop()
        i += 1

    remaining_calibration_periods = (calibration_end_year - calibration_start_year + 1) * periods_per_year

    # get the first interval length of values from the end of the calibration period working backwards, creating the first sum of interval periods
    sum_value = 0.0
    for i in range(interval):
    
        if len(z_temporary) == 0:
           
            i = interval
            
        else:

            # pull a value off the end of the list
            z = z_temporary.pop()
            remaining_calibration_periods -= 1
            
            if not np.isnan(z):
            
                # add to the sum
                sum_value += z
                
                # add to the array of values we've used for the initial sum
                values_to_sum.appendleft(z)
            
            else:

                # reduce the loop counter so we don't skip a calibration interval period
                i -= 1
    
    # if we're dealing with wet conditions then we want to be using positive numbers, and if dry conditions  
    # then we need to be using negative numbers, so we introduce a sign variable to help with this 
    sign = 1
    if 'DRY' == wet_or_dry:

        sign = -1
    
    # for each remaining Z value, recalculate the sum of Z values
    largest_sum = sum_value
    summed_values.appendleft(sum_value)
    while (len(z_temporary) > 0) and (remaining_calibration_periods > 0):
    
        # take the next Z-index value off the end of the list 
        z = z_temporary.pop()

        # reduce by one period for each removal
        remaining_calibration_periods -= 1
    
        if not np.isnan(z):

            # come up with a new Z sum for this new group of Z values to sum
            
            # remove the last value from both the sum_value and the values to sum array
            sum_value -= values_to_sum.pop()
            
            # add to the Z sum, update the bookkeeping lists
            sum_value += z
            values_to_sum.append(z)
            summed_values.append(sum_value)
         
        # update the largest sum value
        if (sign * sum_value) > (sign * largest_sum):

            largest_sum = sum_value

    # Determine the highest or lowest reasonable value that isn't due to a freak anomaly in the data. 
    # A "freak anomaly" is defined as a value that is either
    #   1) 25% higher than the 98th percentile
    #   2) 25% lower than the 2nd percentile
    reasonable_percentile_index = 0
    if 'WET' == wet_or_dry:

        reasonable_percentile_index = int(len(summed_values) * 0.98)

    else:  # DRY
    
        reasonable_percentile_index = int(len(summed_values) * 0.02)
    
    # sort the list of sums into ascending order and get the sum_value value referenced by the safe percentile index
    summed_values = sorted(summed_values)
    sum_at_reasonable_percentile = summed_values[reasonable_percentile_index]
      
    # find the highest reasonable value out of the summed values
    highest_reasonable_value = 0.0
    reasonable_tolerance_ratio = 1.25
    while len(summed_values) > 0:

        sum_value = summed_values.pop()
        if (sign * sum_value) > 0:

            if (sum_value / sum_at_reasonable_percentile) < reasonable_tolerance_ratio:
            
                if (sign * sum_value) > (sign * highest_reasonable_value):
                
                    highest_reasonable_value = sum_value
    
    if 'WET' == wet_or_dry:
    
        return highest_reasonable_value
    
    else:  # DRY
    
        return largest_sum

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _least_squares(x, 
                   y, 
                   n, 
                   wetOrDry):
    
    correlation = 0.0
    c_tol = 0.85
    max_value = 0.0
    max_diff = 0.0
    max_i = 0
    sumX = 0.0
    sumY = 0.0
    sumX2 = 0.0
    sumY2 = 0.0
    sumXY = 0.0
    for i in range(n):
    
        this_x = x[i]
        this_y = y[i]
        sumX += this_x
        sumY += this_y
        sumX2 += this_x * this_x
        sumY2 += this_y * this_y
        sumXY += this_x * this_y
    
    SSX = sumX2 - (sumX * sumX) / n
    SSY = sumY2 - (sumY * sumY) / n
    SSXY = sumXY - (sumX * sumY) / n
    if (SSX > 0) and (SSY > 0):  # perform this check to avoid square root of negative(s)
        correlation = SSXY / (math.sqrt(SSX) * math.sqrt(SSY))
    
    i = n - 1
    
    # if we're dealing with wet conditions then we want to be using positive numbers, and for dry conditions  
    # then we want to be using negative numbers, so we introduce a sign variable to facilitate this 
    sign = 1
    if 'DRY' == wetOrDry: 
        sign = -1     
    
    while ((sign * correlation) < c_tol) and (i > 3):
    
        # when the correlation is off, it appears better to
        # take the earlier sums rather than the later ones.
        this_x = x[i]
        this_y = y[i]
        sumX -= this_x
        sumY -= this_y
        sumX2 -= this_x * this_x
        sumY2 -= this_y * this_y
        sumXY -= this_x * this_y
        SSX = sumX2 - (sumX * sumX) / i
        SSY = sumY2 - (sumY * sumY) / i
        SSXY = sumXY - (sumX * sumY) / i
        if (SSX > 0) and (SSY > 0):  # perform this check to avoid square root of negative(s)
            correlation = SSXY / (math.sqrt(SSX) * math.sqrt(SSY))
        i -= 1
    
    leastSquaresSlope = SSXY / SSX
    for j in range(i + 1):
    
        if (sign * (y[j] - leastSquaresSlope * x[j])) > (sign * max_diff):
        
            max_diff = y[j] - leastSquaresSlope * x[j]
            max_i = j
            max_value = y[j]
         
    leastSquaresIntercept = max_value - leastSquaresSlope * x[max_i]
    
    return leastSquaresSlope, leastSquaresIntercept

#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit      # working?
def pdsi_from_climatology(precip_time_series,
                          temp_time_series,
                          awc,
                          latitude,
                          data_start_year,
                          calibration_start_year,
                          calibration_end_year,
                          B=None,
                          H=None):

    '''
    This function computes the Palmer Drought Severity Index (PDSI), Palmer Hydrological Drought Index (PHDI), 
    and Palmer Z-Index.
    
    The PET values used in the calculation can be computed using Thornthwaite's method (default) or the original 
    method used historically by NCDC (enabled by providing both B and H arguments).
    
    :param precip_time_series: time series of monthly precipitation values, in inches
    :param temperature_time_series: time series of monthly temperature values, in degrees Fahrenheit
    :param awc: available water capacity (soil constant), in inches
    :param latitude: latitude, in degrees north 
    :param data_start_year: initial year of the input precipitation and temperature datasets, 
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period 
    :param calibration_end_year: final year of the calibration period 
    :param B: PET related constant read from the soil constants file for stations and/or climate divisions,
              if present along with the H argument then the original NCDC method for computing PET will be used 
    :param H: PET related constant read from the soil constants file for stations and/or climate divisions,
              if present along with the B argument then the original NCDC method for computing PET will be used 
    :return: four numpy arrays containing PDSI, PHDI, PMDI, and Z-Index values respectively 
    '''

    # convert monthly temperatures from Fahrenheit to Celsius
    monthly_temps_celsius = (temp_time_series - 32) * 5.0 / 9.0

    if B is not None and H is not None:

        # compute PET using method from original PDSI Fortran code pdinew.f
        pet_time_series = pdinew.potential_evapotranspiration(monthly_temps_celsius, 
                                                               latitude,
                                                               data_start_year,
                                                               B,
                                                               H)

    else:
        # compute PET using the Thornthwaite method
        pet_time_series = thornthwaite.potential_evapotranspiration(monthly_temps_celsius, 
                                                                    latitude, 
                                                                    data_start_year)

    return pdsi(precip_time_series,
                pet_time_series.flatten(),
                awc,
                data_start_year,
                calibration_start_year,
                calibration_end_year)

#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit          # working?
def scpdsi_from_climatology(precip_time_series,
                            temp_time_series,
                            awc,
                            latitude,
                            data_start_year,
                            calibration_start_year,
                            calibration_end_year):
    """
    This function computes the Self-calibrated Palmer Drought Severity Index (SCPDSI), Palmer Hydrological 
    Drought Index (PHDI), and Palmer Z-Index.
    
    :param precip_time_series: time series of monthly precipitation values, in inches
    :param temp_time_series: time series of monthly temperature values, in degrees Fahrenheit
    :param awc: available water capacity (soil constant), in inches
    :param latitude: latitude, in degrees north 
    :param data_start_year: initial year of the input precipitation and temperature datasets, 
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period 
    :param calibration_end_year: final year of the calibration period 
    :return: five numpy arrays of floats containing SCPDSI, PDSI, PHDI, PMDI, and Z-Index values respectively 
    """

    # convert monthly temperatures from Fahrenheit to Celsius
    monthly_temps_celsius = (temp_time_series - 32) * 5.0 / 9.0

    # compute PET
    pet_time_series = thornthwaite.potential_evapotranspiration(monthly_temps_celsius, 
                                                                latitude, 
                                                                data_start_year)
    return scpdsi(precip_time_series,
                  pet_time_series.flatten(),
                  awc,
                  data_start_year,
                  calibration_start_year,
                  calibration_end_year)

#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit    # not working yet
def _duration_factors(zindex_values,
                      calibration_start_year,
                      calibration_end_year,
                      data_start_year,
                      wet_or_dry):
    '''
    This functions calculates m and b, which are used to calculated X(i)
    based on the Z index.  These constants will determine the
    weight that the previous PDSI value and the current Z index
    will have on the current PDSI value.  This is done by finding
    several of the driest periods at this station and assuming that
    those periods represents an extreme drought.  Then a linear
    regression is done to determine the relationship between length
    of a dry (or wet) spell and the accumulated Z index during that
    same period.
    
    It appears that there needs to be a different weight given to
    negative and positive Z values, so the variable 'wet_or_dry' will
    determine whether the driest or wettest periods are looked at.

    :param pdsi_values:
    :param zindex_values:  
    :param calibration_start_year: 
    :param calibration_end_year:
    :param data_start_year:
    :param wet_or_dry: compute duration factors for either dry or wet spells. should be either 'WET' or 'DRY'
    :return: slope, intercept
    :rtype: two float values 
    '''
    month_scales = [3, 6, 9, 12, 18, 24, 30, 36, 42, 48]
    
    z_sums = np.zeros((len(month_scales),))
    for i in range(len(month_scales)):

        z_sums[i] = _z_sum(month_scales[i],
                           wet_or_dry, 
                           zindex_values,
                           12, 
                           calibration_start_year, 
                           calibration_end_year, 
                           data_start_year)
    
    slope, intercept = _least_squares(month_scales, z_sums, len(month_scales), wet_or_dry)
    
    # if we're dealing with wet conditions then we want to be using positive numbers, and if dry conditions then
    # we need to be using negative numbers, so we use a PDSI limit of 4 on the wet side and -4 on the dry side 
    pdsi_limit = _PDSI_MAX  # WET
    if 'DRY' == wet_or_dry:

        pdsi_limit = _PDSI_MIN
    
    # now divide slope and intercept by 4 or -4 because that line represents a PDSI of either 4.0 or -4.0
    slope = slope / pdsi_limit
    intercept = intercept / pdsi_limit            

    return slope, intercept

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _pdsi_at_percentile(pdsi_values,
                        percentile):

    pdsi_sorted = sorted(pdsi_values)
    return pdsi_sorted[int(len(pdsi_values) * percentile)]
    
#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit
def _self_calibrate(pdsi_values,
                    sczindex_values,
                    calibration_start_year,
                    calibration_end_year,
                    input_start_year):
    
    # remove periods before the end of the interval
    # calibrate using upper and lower 2% of values within the user-defined calibration interval
    # this is explained in equations (14) and (15) of Wells et al
    dry_ratio = _PDSI_MIN / _pdsi_at_percentile(pdsi_values, 0.02) 
    wet_ratio = _PDSI_MAX / _pdsi_at_percentile(pdsi_values, 0.98) 
        
    # adjust the self-calibrated Z-index values, using either the wet or dry ratio
    #TODO replace the below loop with a vectorized equivalent
    for time_step in range(sczindex_values.size):
    
        if not np.isnan(sczindex_values[time_step]):
        
            if sczindex_values[time_step] >= 0:
            
                adjustmentFactor = wet_ratio
            
            else:
            
                adjustmentFactor = dry_ratio

            sczindex_values[time_step] = sczindex_values[time_step] * adjustmentFactor

    # allocate arrays which will be populated in the following step
    established_index_values = np.full(pdsi_values.shape, np.NaN)
    scpdsi_values = np.full(pdsi_values.shape, np.NaN)
    wet_index_values = np.full(pdsi_values.shape, np.NaN)
    dry_index_values = np.full(pdsi_values.shape, np.NaN)
    
    wet_m, wet_b = _duration_factors(sczindex_values,
                                     calibration_start_year,
                                     calibration_end_year,
                                     input_start_year,
                                     'WET')
    dry_m, dry_b = _duration_factors(sczindex_values,
                                     calibration_start_year,
                                     calibration_end_year,
                                     input_start_year,
                                     'DRY')
    
#     logger.debug('wet_m: {0}   wet_b: {1}   dry_m: {2}   dry_b: {3}'.format(wet_m, wet_b, dry_m, dry_b))
    
    pdsi_values, scpdsi_values, wet_index_values, dry_index_values, established_index_values = \
        _compute_scpdsi(established_index_values,
                        sczindex_values,
                        scpdsi_values,
                        pdsi_values,
                        wet_index_values,
                        dry_index_values,
                        wet_m,
                        wet_b,
                        dry_m,
                        dry_b,
                        False)

    return sczindex_values, pdsi_values, scpdsi_values

#-----------------------------------------------------------------------------------------------------------------------
def scpdsi(precip_time_series,
           pet_time_series,
           awc,
           data_start_year,
           calibration_start_year,
           calibration_end_year):
    '''
    This function computes the Palmer Drought Severity Index (PDSI), Palmer Hydrological Drought Index (PHDI), 
    Modified Palmer Drought Index (PMDI), and Palmer Z-Index.
    
    :param precip_time_series: time series of monthly precipitation values, in inches
    :param pet_time_series: time series of monthly PET values, in inches
    :param awc: available water capacity (soil constant), in inches
    :param data_start_year: initial year of the input precipitation and PET datasets, 
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period 
    :param calibration_end_year: final year of the calibration period 
    :return: numpy arrays, respectively containing SCPDSI, PHDI, and Z-Index values  
    SCPDSI, PDSI, PHDI, PMDI, Z-Index, ET, PR, R, RO, PRO, L, PL
    '''
    try:
        # make sure we have matching precipitation and PET time series
        if precip_time_series.size != pet_time_series.size:
            message = 'Precipitation and PET time series do not match, unequal number or months'
            logger.error(message)
            raise ValueError(message)
                    
        # perform water balance accounting
        ET, PR, R, RO, PRO, L, PL = _water_balance(awc, pet_time_series, precip_time_series)
        
        # if we have input time series (precipitation and PET) with an incomplete 
        # final year then we pad all the time series arrays with NaN values
        pad_months = 12 - (precip_time_series.size % 12)
        if pad_months > 0:            
            precip_time_series = np.pad(precip_time_series, (0, pad_months), 'constant', constant_values=(np.nan))
            pet_time_series = np.pad(pet_time_series, (0, pad_months), 'constant', constant_values=(np.nan))
            ET = np.pad(ET, (0, pad_months), 'constant', constant_values=(np.nan))
            PR = np.pad(PR, (0, pad_months), 'constant', constant_values=(np.nan))
            R = np.pad(R, (0, pad_months), 'constant', constant_values=(np.nan))
            RO = np.pad(RO, (0, pad_months), 'constant', constant_values=(np.nan))
            PRO = np.pad(PRO, (0, pad_months), 'constant', constant_values=(np.nan))
            L = np.pad(L, (0, pad_months), 'constant', constant_values=(np.nan))
            PL = np.pad(PL, (0, pad_months), 'constant', constant_values=(np.nan))
                
        # compute Z-index values
        zindex = _z_index(precip_time_series, 
                          pet_time_series, 
                          ET, 
                          PR, 
                          R, 
                          RO, 
                          PRO, 
                          L, 
                          PL, 
                          data_start_year, 
                          calibration_start_year, 
                          calibration_end_year)
        
        # trim off the padded months from the Z-index array
        if pad_months > 0:
            zindex = zindex[0:-pad_months]
            ET = ET[0:-pad_months]
            PR = PR[0:-pad_months]
            R = R[0:-pad_months]
            RO = RO[0:-pad_months]
            PRO = PRO[0:-pad_months]
            L = L[0:-pad_months]
            PL = PL[0:-pad_months]
            
        # compute PDSI and other associated variables
        PDSI, PHDI, PMDI = _pdsi_from_zindex(zindex)

        # keep a copy of the originally computed PDSI for return
        final_PDSI = np.array(PDSI)
        
        # perform self-calibration        
        zindex, PDSI, SCPDSI = _self_calibrate(PDSI, 
                                               zindex,
                                               calibration_start_year,
                                               calibration_end_year,
                                               data_start_year)

        # recompute PDSI and other associated variables
        SCPDSI, PHDI, PMDI = _pdsi_from_zindex(zindex)
        
        return [SCPDSI, final_PDSI, PHDI, PMDI, zindex]

    except:
        # catch all exceptions, log rudimentary error information
        logger.error('Failed to complete', exc_info=True)
        raise

#-----------------------------------------------------------------------------------------------------------------------
def pdsi(precip_time_series,
         pet_time_series,
         awc,
         data_start_year,
         calibration_start_year=1931,
         calibration_end_year=1990):
    '''
    This function computes the Palmer Drought Severity Index (PDSI), Palmer Hydrological Drought Index (PHDI), 
    Palmer Modified Drought Index (PMDI), and Palmer Z-Index.
    
    :param precip_time_series: time series of monthly precipitation values, in inches
    :param pet_time_series: time series of monthly PET values, in inches
    :param awc: available water capacity (soil constant), in inches
    :param data_start_year: initial year of the input precipitation and PET datasets, 
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period 
    :param calibration_end_year: final year of the calibration period 
    :return: four numpy arrays containing PDSI, PHDI, PMDI, and Z-Index values respectively 
    '''
    try:
        # make sure we have matching precipitation and PET time series
        if precip_time_series.size != pet_time_series.size:
            message = 'Precipitation and PET time series do not match, unequal number or months'
            logger.error(message)
            raise ValueError(message)
                    
        # perform water balance accounting
        ET, PR, R, RO, PRO, L, PL = _water_balance(awc, pet_time_series, precip_time_series)
        
        # if we have input time series (precipitation and PET) with an incomplete 
        # final year then we pad all the time series arrays with NaN values
        pad_months = 12 - (precip_time_series.size % 12)
        if pad_months > 0:            
            precip_time_series = np.pad(precip_time_series, (0, pad_months), 'constant', constant_values=(np.nan))
            pet_time_series = np.pad(pet_time_series, (0, pad_months), 'constant', constant_values=(np.nan))
            ET = np.pad(ET, (0, pad_months), 'constant', constant_values=(np.nan))
            PR = np.pad(PR, (0, pad_months), 'constant', constant_values=(np.nan))
            R = np.pad(R, (0, pad_months), 'constant', constant_values=(np.nan))
            RO = np.pad(RO, (0, pad_months), 'constant', constant_values=(np.nan))
            PRO = np.pad(PRO, (0, pad_months), 'constant', constant_values=(np.nan))
            L = np.pad(L, (0, pad_months), 'constant', constant_values=(np.nan))
            PL = np.pad(PL, (0, pad_months), 'constant', constant_values=(np.nan))
                
        # compute Z-index values
        zindex = _z_index(precip_time_series, 
                          pet_time_series, 
                          ET, 
                          PR, 
                          R, 
                          RO, 
                          PRO, 
                          L, 
                          PL, 
                          data_start_year, 
                          calibration_start_year, 
                          calibration_end_year)
        
        # trim off the padded months from the Z-index array
        if pad_months > 0:
            zindex = zindex[0:-pad_months]
            
        # compute PDSI and other associated variables
        PDSI, PHDI, PMDI = _pdsi_from_zindex(zindex)
        
        return PDSI, PHDI, PMDI, zindex
    
    except:
        # catch all exceptions, log rudimentary error information
        logger.error('Failed to complete', exc_info=True)
        raise

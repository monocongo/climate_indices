import logging
import math
from math import exp, lgamma, log, pi, sqrt
from numba import float64, int32, jit
import numpy as np
import scipy.special
import scipy.stats
import utils

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger
logging.basicConfig(level=logging.WARN,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
@jit
def sum_to_scale(values,
                 scale):
    '''
    Compute a sliding sums array using 1-D convolution. The initial (scale - 1) elements 
    of the result array will be padded with np.NaN values. Missing values are not ignored, i.e. if a np.NaN
    (missing) value is part of the group of values to be summed then the sum will be np.NaN
    
    For example if the first array is [3, 4, 6, 2, 1, 3, 5, 8, 5] and the number of values to sum is 3 then the resulting
    array will be [np.NaN, np.NaN, 13, 12, 9, 6, 9, 16, 18].
    
    More generally:
    
    Y = f(X, n)
    
    Y[i] == np.NaN, where i < n
    Y[i] == sum(X[i - n + 1:i + 1]), where i >= n - 1 and X[i - n + 1:i + 1] contains no NaN values
    Y[i] == np.NaN, where i >= n - 1 and X[i - n + 1:i + 1] contains one or more NaN values
         
    :param values: the array of values over which we'll compute sliding sums
    :param scale: the number of values for which each sliding summation will encompass, for example if this value
                  is 3 then the first two elements of the output array will contain the pad value and the third 
                  element of the output array will contain the sum of the first three elements, and so on 
    :return: an array of sliding sums, equal in length to the input values array, left padded with NaN values  
    '''
    
    # don't bother if the number of values to sum is 1 (will result in duplicate array)
    if scale == 1:
        return values
    
    # get the valid sliding summations with 1D convolution
    sliding_sums = np.convolve(values, np.ones(scale), mode='valid')
    
    # pad the first (n - 1) elements of the array with NaN values
    return np.hstack(([np.NaN]*(scale - 1), sliding_sums))

#-----------------------------------------------------------------------------------------------------------------------
def _count_zeros_and_non_missings(values):
    
    # count the number of zeros and non-missing (non-NaN) values
    zeros = values.size - np.count_nonzero(values)
    non_missings = np.count_nonzero(~np.isnan(values))

    return zeros, non_missings

#-----------------------------------------------------------------------------------------------------------------------
def _estimate_pearson3_parameters(lmoments):    
    '''
    Estimate parameters via L-moments for the Pearson Type III distribution, based on Fortran code written 
    for inclusion in IBM Research Report RC20525, 'FORTRAN ROUTINES FOR USE WITH THE METHOD OF L-MOMENTS, VERSION 3' 
    by J. R. M. Hosking, IBM Research Division, T. J. Watson Research Center, Yorktown Heights, NY 10598
    This is a Python translation of the original Fortran subroutine named 'pearson3'.
    
    :param lmoments: 3-element, 1-D (flat) array containing the first three L-moments (lambda-1, lambda-2, and tau-3)
    :return the Pearson Type III parameters corresponding to the input L-moments
    :rtype: a 3-element, 1-D (flat) numpy array of floats
    '''
    
    C1 = 0.2906
    C2 = 0.1882
    C3 = 0.0442
    D1 = 0.36067
    D2 = -0.59567
    D3 = 0.25361
    D4 = -2.78861
    D5 = 2.56096
    D6 = -0.77045
    T3 = abs(lmoments[2])  # L-skewness?
    
    # ensure the validity of the L-moments
    if (lmoments[1] <= 0) or (T3 >= 1):
        message = 'Unable to calculate Pearson Type III parameters due to invalid L-moments'
        logger.error(message)
        raise ValueError(message)

    # initialize the output array    
    pearson3_parameters = np.zeros((3,))
    
    # the first Pearson Type III parameter is the same as the first L-moment
    pearson3_parameters[0] = lmoments[0]
    
    if (T3 <= 1e-6):
        # skewness is effectively zero
        pearson3_parameters[1] = lmoments[1] * sqrt(pi)

    else:
        if (T3 < 0.333333333):
            T = pi * 3 * T3 * T3
            alpha = (1.0 + (C1 * T)) / (T * (1.0 + (T * (C2 + (T * C3)))))
        else:
            T = 1.0 - T3
            alpha = T * (D1 + (T * (D2 + (T * D3)))) / (1.0 + (T * (D4 + (T * (D5 + (T * D6))))))
            
        alpha_root = sqrt(alpha)
        beta = sqrt(pi) * lmoments[1] * exp(lgamma(alpha) - lgamma(alpha + 0.5))
        pearson3_parameters[1] = beta * alpha_root
        
        # the sign of the third L-moment determines the sign of the third Pearson Type III parameter
        if (lmoments[2] < 0):
            pearson3_parameters[2] = -2.0 / alpha_root
        else:
            pearson3_parameters[2] = 2.0 / alpha_root

    return pearson3_parameters

#-----------------------------------------------------------------------------------------------------------------------    
@jit
def _estimate_lmoments(values):
    '''
    Estimate sample L-moments, based on Fortran code written for inclusion in IBM Research Report RC20525,
    'FORTRAN ROUTINES FOR USE WITH THE METHOD OF L-MOMENTS, VERSION 3' by J. R. M. Hosking, IBM Research Division,
    T. J. Watson Research Center, Yorktown Heights, NY 10598, Version 3 August 1996.
    
    Documentation on the original Fortran routines found here: https://rdrr.io/cran/nsRFA/man/HW.original.html
    
    This is a Python translation of the original Fortran subroutine SAMLMR() and which has been optimized 
    for calculating only the first three L-moments. 
    
    :param values: 1-D (flattened) array of float values
    :return: an estimate of the first three sample L-moments
    :rtype: 1-D numpy array of floats (the first three sample L-moments corresponding to the input values)
    '''
    
    # we need to have at least four values in order to make a sample L-moments estimation
    number_of_values = np.count_nonzero(~np.isnan(values))
    if (number_of_values < 4):
        message = 'Insufficient number of values to perform sample L-moments estimation'
        logger.warn(message)
        raise ValueError(message)
        
    # sort the values into ascending order
    values = np.sort(values)
    
    sums = np.zeros((3,))

    for i in range(1, number_of_values + 1):
        z = i
        term = values[i - 1]
        sums[0] = sums[0] + term
        for j in range(1, 3):
            z = z - 1
            term = term * z
            sums[j] = sums[j] + term
        
    y = float(number_of_values)
    z = float(number_of_values)
    sums[0] = sums[0] / z
    for j in range(1, 3):
        y = y - 1.0
        z = z * y
        sums[j] = sums[j] / z
    
    k = 3
    p0 = -1.0
    for _ in range(2):
        ak = float(k)
        p0 = -p0
        p = p0
        temp = p * sums[0]
        for i in range(1, k):
            ai = i
            p = -p * (ak + ai - 1.0) * (ak - ai) / (ai * ai)
            temp = temp + (p * sums[i])
        sums[k - 1] = temp
        k = k - 1
      
    lmoments = np.zeros((3,))  
    if (sums[1] != 0):
        lmoments[0] = sums[0]
        lmoments[1] = sums[1]
        lmoments[2] = sums[2] / sums[1]
        
    return lmoments
    
#-----------------------------------------------------------------------------------------------------------------------
@jit
def _pearson3_fitting_values(values, 
                             data_start_year, 
                             calibration_start_year,
                             calibration_end_year):
    '''
    This function computes the calendar monthly probability of zero and Pearson Type III distribution fitting parameters 
    corresponding to an array of monthly values.
    
    :param values: 2-D array of monthly values, with each row representing a year containing 12 values corresponding 
                   to the calendar months of that year, and assuming that the first value of the array is January of the initial year
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period 
    :param calibration_end_year: the final year to use for the calibration period 
    :return: a 2-D array of monthly fitting values for the Pearson Type III distribution, with shape (4, 12)
             returned_array[0] == probability of zero for each of the 12 calendar months 
             returned_array[1] == the first Pearson Type III distribution parameter for each of the 12 calendar months 
             returned_array[2] == the second Pearson Type III distribution parameter for each of the 12 calendar months 
             returned_array[3] == the third Pearson Type III distribution parameter for each of the 12 calendar months 
    '''
    
    #TODO validate that the values array has shape == (years x 12)
    
    # determine the end year of the values array
    data_end_year = data_start_year + values.shape[0]
    
    # make sure that we have data within the full calibration period, otherwise use the full period of record
    if (calibration_start_year < data_start_year) or (calibration_end_year > data_end_year):
        logger.info('Insufficient data for the specified calibration period ({0}-{1}), instead using the full period ' + 
                    'of record ({2}-{3})'.format(calibration_start_year, 
                                                 calibration_end_year, 
                                                 data_start_year, 
                                                 data_end_year))
        calibration_start_year = data_start_year
        calibration_end_year = data_end_year

    # get the year axis indices corresponding to the calibration start and end years
    calibration_begin_index = (calibration_start_year - data_start_year)
    calibration_end_index = (calibration_end_year - data_start_year) + 1
    
    # now we'll use these sums to come up with the probability of zero and Pearson parameters for each calendar month
    monthly_fitting_values = np.zeros((4, 12))
    #TODO vectorize the below loop?
    for month_index in range(12):
    
        # get the values for the current calendar month that fall within the calibration years period
        calibration_values = values[calibration_begin_index:calibration_end_index, month_index]

        # count the number of zeros and valid (non-missing/non-NaN) values
        number_of_zeros, number_of_non_missing = _count_zeros_and_non_missings(calibration_values)

        # make sure we have at least four values that are both non-missing (i.e. non-NaN)
        # and non-zero, otherwise use the entire period of record
        if (number_of_non_missing - number_of_zeros) < 4:
            
            # update the array of calibration values for the calendar month to include the full period of record
            calibration_values = values[:, month_index]
            
            # get new counts of the zeros and non-missing values
            number_of_zeros, number_of_non_missing = _count_zeros_and_non_missings(calibration_values)
            
        # calculate the probability of zero for the calendar month
        probability_of_zero = 0.0
        if number_of_zeros > 0:

            probability_of_zero = number_of_zeros / number_of_non_missing
            
        # get the estimated L-moments, if we have more than three non-missing/non-zero values
        if (number_of_non_missing - number_of_zeros) > 3:

            # estimate the L-moments of the calibration values
            lmoments = _estimate_lmoments(calibration_values)

            # if we have valid L-moments then we can proceed, otherwise the fitting values for the month will be all zeros
            if (lmoments[1] > 0.0) and (abs(lmoments[2]) < 1.0):
                
                # get the Pearson Type III parameters for the month, based on the L-moments
                pearson_parameters = _estimate_pearson3_parameters(lmoments)

                monthly_fitting_values[0, month_index] = probability_of_zero
                monthly_fitting_values[1, month_index] = pearson_parameters[0]
                monthly_fitting_values[2, month_index] = pearson_parameters[1]
                monthly_fitting_values[3, month_index] = pearson_parameters[2]

            else:
                logger.warn('Due to invalid L-moments the Pearson fitting values for month {0} are defaulting to zero'.format(month_index))

    return monthly_fitting_values

#----------------------------------------------------------------------------------------------------------------------
@jit
def _pearson3cdf(value,
                 pearson3_parameters):
    '''
    Compute the probability that a random variable along the Pearson Type III distribution described by a set 
    of parameters will be less than or equal to a value.
    
    :param value: 
    :param pearson3_parameters:
    :return 
    '''

    # it's only possible to make the calculation if the second Pearson parameter is above zero
    if pearson3_parameters[1] <= 0.0:
    
        logger.debug("The second Pearson parameter is less than or equal to zero, invalid for the CDF calculation")
        return np.NaN
    
    result = 0
    skew = pearson3_parameters[2]
    if abs(skew) <= 1e-6:
    
        z = (value - pearson3_parameters[0]) / pearson3_parameters[1]
        return 0.5 + (0.5 * _error_function(z * sqrt(0.5)))
    
    alpha = 4.0 / (skew * skew)
    x = ((2.0 * (value - pearson3_parameters[0])) / (pearson3_parameters[1] * skew)) + alpha
    if x > 0:
    
        result = scipy.special.gammainc(alpha, x)
        if skew < 0.0:
        
            result = 1.0 - result
        
    else:
    
        # calculate the lowest possible value that will fit the distribution (i.e. Z = 0)
        minimum_possible_value = pearson3_parameters[0] - ((alpha * pearson3_parameters[1] * skew) / 2.0)
        if value <= minimum_possible_value:
        
            result = 0.0005  # minimum probability (why this arbitrary value? Trevor/Richard? related to the trace precipitation value?)
        
        else:
        
            result = np.NaN

    return result

#----------------------------------------------------------------------------------------------------------------------
@jit(float64(float64))
def _error_function(value):
    '''
    TODO
    
    :param value:
    :return:  
    '''
    
    result = 0.0
    if value != 0.0:

        absValue = abs(value)

        if absValue > 6.25:
            if value < 0:
                result = -1.0
            else:
                result = 1.0
        else:
            exponential = exp(value * value * (-1))
            sqrtOfTwo = sqrt(2.0)
            zz = abs(value * sqrtOfTwo)
            if absValue > 5.0:
                # alternative error function calculation for when the input value is in the critical range
                result = exponential * (sqrtOfTwo / pi) / \
                                         (absValue + 1 / (zz + 2 / (zz + 3 / (zz + 4 / (zz + 0.65)))))

            else:
                # coefficients of rational-function approximation
                P0 = 220.2068679123761
                P1 = 221.2135961699311
                P2 = 112.0792914978709
                P3 = 33.91286607838300
                P4 = 6.373962203531650
                P5 = 0.7003830644436881
                P6 = 0.03526249659989109
                Q0 = 440.4137358247522
                Q1 = 793.8265125199484
                Q2 = 637.3336333788311
                Q3 = 296.5642487796737
                Q4 = 86.78073220294608
                Q5 = 16.06417757920695
                Q6 = 1.755667163182642
                Q7 = 0.08838834764831844

                # calculate the error function from the input value and constant values
                result = exponential * ((((((P6 * zz + P5) * zz + P4) * zz + P3) * zz + P2) * zz + P1) * zz + P0) /  \
                         (((((((Q7 * zz + Q6) * zz + Q5) * zz + Q4) * zz + Q3) * zz + Q2) * zz + Q1) * zz + Q0)

            if value > 0.0:
                result = 1 - result
            elif value < 0:
                result = result - 1.0

    return result

#-----------------------------------------------------------------------------------------------------------------------
@jit
def transform_fitted_pearson(monthly_values,
                             data_start_year,
                             calibration_start_year,
                             calibration_end_year):
    '''
    TODO explain this
    
    :param monthly_values: 2-D array of monthly values, with each row representing a year containing twelve calendar months
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period 
    :param calibration_end_year: the final year to use for the calibration period 
    :return: 2-D array of monthly values, corresponding in size and shape of the input array
    :rtype: numpy.ndarray of floats
    '''
    
    # if we're passed all missing values then we can't compute anything, return the same array of missing values
    if np.all(np.isnan(monthly_values)):
        logger.info('An array of all fill values was passed as the argument, no action taken, returning the same array')
        return monthly_values
        
    # validate (and possibly reshape) the input array
    if len(monthly_values.shape) == 1:
        
        # we've been passed a 1-D array with shape (months), reshape it to 2-D with shape (years, 12)
        monthly_values = utils.reshape_to_years_months(monthly_values)
    
    elif (len(monthly_values.shape) != 2) or monthly_values.shape[1] != 12:
     
        # neither a 1-D nor a 2-D array with valid shape was passed in
        message = 'Invalid input array with shape: {0}'.format(monthly_values.shape)
        logger.error(message)   
        raise ValueError(message)
    
    # compute the values we'll use to fit to the Pearson Type III distribution
    monthly_pearson_values = _pearson3_fitting_values(monthly_values, 
                                                     data_start_year,
                                                     calibration_start_year,
                                                     calibration_end_year)
    
    # allocate the array of values we'll return, with all values initialized to the fill value (NaN)
    fitted_values = np.full(monthly_values.shape, np.NaN)

    # compute Pearson CDF -> probability values -> fitted values for the entire period of record
    probability_of_zero = 0.0
    probability_value = 0.0
    pearson_parameters = np.zeros((3,))
    for year_index in range(monthly_values.shape[0]):
        for calendar_month_index in range(12):

            pearson_parameters[0] = monthly_pearson_values[1, calendar_month_index]   # first Pearson Type III parameter
            pearson_parameters[1] = monthly_pearson_values[2, calendar_month_index]   # second Pearson Type III parameter
            pearson_parameters[2] = monthly_pearson_values[3, calendar_month_index]   # third Pearson Type III parameter
            probability_of_zero = monthly_pearson_values[0, calendar_month_index]
    
            # only fit to the distribution if we don't have a fill value for the current month's scale sum
            if not math.isnan(monthly_values[year_index, calendar_month_index]):
    
                # get the Pearson Type III cumulative density function value
                pe3_cdf = 0.0
                
                #TODO questions for Trevor/Richard/Deke -- what is the significance of the value 0.0005 below? 
                # Is this a trace precip value or a floor probability value, etc.?
                
                # handle trace amounts as a special case
                if monthly_values[year_index, calendar_month_index] < 0.0005:
                
                    if probability_of_zero > 0.0:
                    
                        pe3_cdf = 0.0
                    
                    else:
                    
                        pe3_cdf = 0.0005  # minimum probability
                    
                else:
                
                    # calculate the CDF value corresponding to the current month's value
                    pe3_cdf = _pearson3cdf(monthly_values[year_index, calendar_month_index], pearson_parameters)
                                   
                if not math.isnan(pe3_cdf):
                
                    # calculate the probability value, clipped between 0 and 1
                    probability_value = np.clip((probability_of_zero + ((1.0 - probability_of_zero) * pe3_cdf)), 0.0, 1.0)
    
                    # the values we'll return are the values at which the probabilities of a normal distribution are less than or equal to
                    # the computed probabilities, as determined by the normal distribution's quantile (or inverse cumulative distribution) function  
                    fitted_values[year_index, calendar_month_index] = scipy.stats.norm.ppf(probability_value)
                
    return fitted_values

#-----------------------------------------------------------------------------------------------------------------------@jit

def transform_fitted_gamma(monthly_values):
    '''
    TODO explain this    

    :param monthly_values: an array of monthly values, either 1-D or 2-D with each row representing 
                           a year containing twelve columns representing the respective calendar months
    :return: 2-D array of monthly values, corresponding in size and shape of the input array if the input is 2-D, or if the input array  
             is 1-D then an equivalent 2-D array with NaN values used to fill the missing months of the final year, if any
    :rtype: numpy.ndarray of floats
    '''
    
    # if we're passed all missing values then we can't compute anything, return the same array of missing values
    if np.all(np.isnan(monthly_values)):
        logger.info('An array of all fill values was passed as the argument, no action taken, returning the same array')
        return monthly_values
        
    # validate (and possibly reshape) the input array
    if len(monthly_values.shape) == 1:
        
        # we've been passed a 1-D array with shape (months), reshape it to 2-D with shape (years, 12)
        monthly_values = utils.reshape_to_years_months(monthly_values)
    
    elif (len(monthly_values.shape) != 2) or monthly_values.shape[1] != 12:
     
        # neither a 1-D nor a 2-D array with valid shape was passed in
        message = 'Invalid input array with shape: {0}'.format(monthly_values.shape)
        logger.error(message)   
        raise ValueError(message)
    
    # find the percentage of zero values for each month
    zeros = (monthly_values == 0).sum(axis=0)
    probabilities_of_zero = zeros / monthly_values.shape[0]
    
    # replace zeros with NaNs
    monthly_values[monthly_values == 0] = np.NaN
    
    # compute the gamma distribution's shape and scale parameters, alpha and beta
    #TODO explain this better
    means = np.nanmean(monthly_values, axis=0)
    log_means = np.log(means)
    logs = np.log(monthly_values)
    mean_logs = np.nanmean(logs, axis=0)
    A = log_means - mean_logs
    alphas = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A)
    betas = means / alphas
    
    # find the gamma probability values using the gamma CDF
    gamma_probabilities = scipy.stats.gamma.cdf(monthly_values, a=alphas, scale=betas)

    #TODO explain this
    # (normalize including the probability of zero, putting into the range [0..1]?)    
    probabilities = probabilities_of_zero + ((1 - probabilities_of_zero) * gamma_probabilities)
    
    # the values we'll return are the values at which the probabilities of a normal distribution are less than or equal to
    # the computed probabilities, as determined by the normal distribution's quantile (or inverse cumulative distribution) function  
    return scipy.stats.norm.ppf(probabilities)
 
############################################################################################################################################   
#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------
# Below are new versions of the code to perform a transformed fitting to the Pearson Type III distribution (for SPI and SPEI) 
# using a) the SciPy statistics module for finding the fitting parameters instead of the existing method of estimating parameters
# using sample L-moments, and b) the SciPy statistics module for computing the cumulative distribution function. 
#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------
#@jit
def transform_fitted_pearson_new(monthly_values,
                                 data_start_year,
                                 calibration_start_year,
                                 calibration_end_year):
    '''
    TODO explain this
    
    :param monthly_values: 2-D array of monthly values, with each row representing a year containing twelve calendar months
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period 
    :param calibration_end_year: the final year to use for the calibration period 
    :return: 2-D array of monthly values, corresponding in size and shape of the input array
    :rtype: numpy.ndarray of floats
    '''
    
    # if we're passed all missing values then we can't compute anything, return the same array of missing values
    if np.all(np.isnan(monthly_values)):
        return monthly_values
        
    # compute the values we'll use to fit to the Pearson Type III distribution (parameters and probability of zero)
    monthly_pearson_values = pearson3_fitting_values_new(monthly_values, 
                                                         data_start_year,
                                                         calibration_start_year,
                                                         calibration_end_year)
    
    # allocate the array of values we'll return, with all values initialized to the fill value (NaN)
    fitted_values = np.full(monthly_values.shape, np.NaN)
    
    # compute Pearson CDF -> probability values -> fitted values for the entire period of record
    probability_of_zero = 0.0
    probability_value = 0.0
    pearson_parameters = np.zeros((3,))
    for year_index in range(monthly_values.shape[0]):
        for calendar_month_index in range(12):

            # only fit to the distribution if we don't have a fill value for the current month's scale sum
            if not math.isnan(monthly_values[year_index, calendar_month_index]):
    
                probability_of_zero = monthly_pearson_values[0, calendar_month_index]
            
                # get the Pearson Type III cumulative/probability density function values
                pe3_cdf = 0.0
                
                #TODO questions for Trevor/Richard/Deke -- what is the significance of the value 0.0005 below? 
                # Is this a trace precip value or a floor probability value, etc.?
                
                # handle trace amounts as a special case
                if monthly_values[year_index, calendar_month_index] < 0.0005:
                
                    if probability_of_zero > 0.0:
                    
                        pe3_cdf = 0.0
                    
                    else:
                    
                        pe3_cdf = 0.0005  # minimum probability
                    
                else:
                
                    # calculate the CDF value for the current month's value
                    pe3_shape = monthly_pearson_values[1, calendar_month_index]     # first Pearson Type III parameter, shape
                    pe3_location = monthly_pearson_values[2, calendar_month_index]  # second Pearson Type III parameter, location
                    pe3_scale = monthly_pearson_values[3, calendar_month_index]     # third Pearson Type III parameter, scale
                    pe3_cdf = scipy.stats.pearson3.cdf(monthly_values[year_index, calendar_month_index], 
                                                       skew=pe3_shape,
                                                       loc=pe3_location,
                                                       scale=pe3_scale)
                
                if not math.isnan(pe3_cdf):
                
                    # calculate the probability value, clipped between 0 and 1
                    probability_value = np.clip((probability_of_zero + ((1.0 - probability_of_zero) * pe3_cdf)), 0.0, 1.0)
    
                    # the values we'll return are the values at which the probabilities of a normal distribution are less than or equal to
                    # the computed probabilities, as determined by the normal distribution's quantile (or inverse cumulative distribution) function  
                    fitted_values[year_index, calendar_month_index] = scipy.stats.norm.ppf(probability_value)
                
    return fitted_values

#-----------------------------------------------------------------------------------------------------------------------
#@jit
def pearson3_fitting_values_new(values, 
                                data_start_year, 
                                calibration_start_year,
                                calibration_end_year):
    '''
    This function computes the calendar monthly probability of zero and skew corresponding to an array of monthly values.
    
    :param values: 2-D array of monthly values, with each row representing a year containing 12 values corresponding 
                   to the calendar months of that year, and assuming that the first value of the array is January of the initial year
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period 
    :param calibration_end_year: the final year to use for the calibration period 
    :return: a 2-D array of monthly fitting values for the Pearson Type III distribution, with shape (2, 12)
             returned_array[0] == probability of zero for each of the 12 calendar months 
             returned_array[1] == the skew for each of the 12 calendar months 
    '''
    
    #TODO validate that the values array has shape == (years x 12)
    
    # determine the end year of the values array
    data_end_year = data_start_year + values.shape[0]
    
    # make sure that we have data within the full calibration period, otherwise use the full period of record
    if (calibration_start_year < data_start_year) or (calibration_end_year > data_end_year):
        logger.info('Insufficient data for the specified calibration period ({1}-{2}), instead using the full period ' + 
                    'of record ({3}-{4})'.format(calibration_start_year, 
                                                 calibration_end_year, 
                                                 data_start_year, 
                                                 data_end_year))
        calibration_start_year = data_start_year
        calibration_end_year = data_end_year

    # get the year axis indices corresponding to the calibration start and end years
    calibration_begin_index = (calibration_start_year - data_start_year)
    calibration_end_index = (calibration_end_year - data_start_year) + 1
    
    # now we'll use these sums to come up with the probability of zero and Pearson parameters for each calendar month
    monthly_fitting_values = np.zeros((4, 12))
    #TODO vectorize the below loop?
    for month_index in range(12):
    
        # get the values for the current calendar month that fall within the calibration years period
        calibration_values = values[calibration_begin_index:calibration_end_index, month_index]

        # count the number of zeros and valid (non-missing/non-NaN) values
        number_of_zeros, number_of_non_missing = _count_zeros_and_non_missings(calibration_values)

        # make sure we have at least four values that are both non-missing (i.e. non-NaN)
        # and non-zero, otherwise use the entire period of record
        if (number_of_non_missing - number_of_zeros) < 4:
            
            # update the array of calibration values for the calendar month to include the full period of record
            calibration_values = values[:, month_index]
            
            # get new counts of the zeros and non-missing values
            number_of_zeros, number_of_non_missing = _count_zeros_and_non_missings(calibration_values)
            
        # calculate the probability of zero for the calendar month
        probability_of_zero = 0.0
        if number_of_zeros > 0:

            probability_of_zero = number_of_zeros / number_of_non_missing
            
        monthly_fitting_values[0, month_index] = probability_of_zero
        
        # get the Pearson Tyoe III parameters for this calendar month's values within the calibration period
        shape, location, scale = scipy.stats.pearson3.fit(calibration_values)
        monthly_fitting_values[1, month_index] = shape
        monthly_fitting_values[2, month_index] = location
        monthly_fitting_values[3, month_index] = scale

    return monthly_fitting_values

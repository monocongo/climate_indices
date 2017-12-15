from datetime import datetime
import logging
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def compute_days(initial_year, 
                 total_months):
    """
    This function computes a series (list) of day values to correspond with the first day of the month for each month 
    of a time series starting from an initial year.
    
    :param initial_year:
    :param total_months: total number of months in the time series
    :return: numpy array of integers corresponding to   
    """
    
    # the date from which the returned array of day values are since (i.e. when using "days since <start_date>" as our units for time)    
    start_date = datetime(initial_year, 1, 1)
    
    # initialize the list of day values we'll build
    days = np.empty(total_months, dtype=int)
    
    # loop over all time steps (months)
    for i in range(total_months):
        years = int(i / 12)  # the number of years since the initial year 
        months = int(i % 12) # the number of months since January
        
        # cook up a date for the current time step (month)
        current_date = datetime(initial_year + years, 1 + months, 1)
        
        # leverage the difference between dates operation available with datetime objects
        days[i] = (current_date - start_date).days
    
    return days

#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit
def reshape_to_years_months(monthly_values):
    '''
    :param monthly_values: an 1-D numpy.ndarray of monthly values, assumed to start at January
    :return: the original monthly values reshaped to 2-D, with each row representing a full year, with shape (years, 12)
    :rtype: 2-D numpy.ndarray of floats
    '''
    
    # if we've been passed a 2-D array with valid shape then let it pass through
    shape = monthly_values.shape
    if len(shape) == 2:
        if shape[1] == 12:
            # data is already in the shape we want, return it unaltered
            return monthly_values
        else:
            message = 'Values array has an invalid shape (2-D but second dimension not 12): {}'.format(shape)
            logger.error(message)
            raise ValueError(message)
    
    # otherwise make sure that we've been passed in a flat (1-D) array of values    
    elif len(shape) != 1:
        message = 'Values array has an invalid shape (not 1-D or 2-D): {}'.format(shape)
        logger.error(message)
        raise ValueError(message)

    # pad the final months of the final year, if necessary
    final_year_months = shape[0] % 12
    if final_year_months > 0:
        pad_months = 12 - final_year_months
        pad_values = np.full((pad_months,), np.NaN)
        monthly_values = np.append(monthly_values, pad_values)
        
    # we should have an ordinal number of years now (ordinally divisible by 12)
    total_years = int(monthly_values.shape[0] / 12)
    
    # reshape from (months) to (years, 12) in order to have one year of months per row
    return np.reshape(monthly_values, (total_years, 12))
            
#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit
def reshape_to_divs_years_months(monthly_values):
    '''
    :param monthly_values: an 2-D numpy.ndarray of monthly values, assumed to start at January of 
                           the first year for each division
    :return: the original monthly values reshaped to 3-D (divisions, years, 12), within each division each row 
             representing a full year, with shape (years, 12)
    :rtype: 3-D numpy.ndarray of floats
    '''
    
    # if we've been passed a 3-D array with valid shape then let it pass through
    shape = monthly_values.shape
    if len(shape) == 3:
        if shape[2] == 12:
            # data is already in the shape we want, return it unaltered
            return monthly_values
        else:
            message = 'Values array has an invalid shape (3-D but third dimension not 12): {}'.format(shape)
            logger.error(message)
            raise ValueError(message)
    
    # otherwise make sure that we've been passed in a 2-D array of values    
    elif len(shape) != 2:
        message = 'Values array has an invalid shape (not 2-D or 3-D): {}'.format(shape)
        logger.error(message)
        raise ValueError(message)

    # pad the final months of the final year, if necessary
    final_year_months = shape[1] % 12
    if final_year_months > 0:
        pad_months = 12 - final_year_months
#         pad_values = np.full((shape[1], pad_months,), np.NaN)
#         monthly_values = np.append(monthly_values, pad_values)
        
        monthly_values = np.pad(monthly_values, [(0, 0), (0, pad_months)], mode='constant', constant_values=np.NaN)
        
    # we should have an ordinal number of years now (ordinally divisible by 12)
    total_years = int(monthly_values.shape[1] / 12)
    
    # reshape from (months) to (years, 12) in order to have one year of months per row
    return np.reshape(monthly_values, (shape[0], total_years, 12))
            
#-----------------------------------------------------------------------------------------------------------------------
def count_zeros_and_non_missings(values):
    
    # count the number of zeros and non-missing (non-NaN) values
    zeros = values.size - np.count_nonzero(values)
    non_missings = np.count_nonzero(~np.isnan(values))

    return zeros, non_missings

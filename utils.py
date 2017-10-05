import logging
import numba
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

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
            

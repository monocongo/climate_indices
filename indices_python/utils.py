from datetime import datetime
import logging
import numba
import numpy as np
import pycurl

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def sign_change(a, b):
    """
    Given two same-sized arrays of floats return an array of booleans indicating if a sign change occurs at the 
    corresponding index.
    
    :param a: array of floats
    :param b: 
    :return: array of booleans of same size as input arrays
    """
    
    if a.size != b.size:
        
        raise ValueError('Mismatched input arrays')

    # use the shape of the first array as the shape of the array we'll return    
    original_shape = a.shape
    
    # get the sign value for each element
    sign_a = np.sign(a.flatten())
    sign_b = np.sign(b.flatten())
    
    # sign change between the two where values unequal
    sign_changes = (sign_a != sign_b)

    return np.reshape(sign_changes, original_shape)

#-----------------------------------------------------------------------------------------------------------------------
def is_data_valid(data):
    """
    Returns whether or not an array is valid, i.e. a supported array type (ndarray or MaskArray) which is not all-NaN.

    :param data: data object, expected as either numpy.ndarry or numpy.ma.MaskArray
    :return True if array is non-NaN for at least one element and is an array type valid for processing by other modules
    :rtype: boolean
    """

    # make sure we're not dealing with all NaN values
    if np.ma.isMaskedArray(data):

        valid_flag = bool(data.count())

    elif isinstance(data, np.ndarray):

        valid_flag = not np.all(np.isnan(data))

    else:
        _logger.warning('Invalid data type')
        valid_flag = False

    return valid_flag

#-----------------------------------------------------------------------------------------------------------------------
def retrieve_file(url,         # pragma: no cover
                  out_file):
    """
    Downloads and writes a file to a specified local file location.
    
    :param url: URL to the file we'll download, expected to be a binary file
    :param out_file: local file location where the file will be written once fetched from the URL  
    """
    
    with open(out_file, 'wb') as f:
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, f)
        c.perform()
        c.close()

#-----------------------------------------------------------------------------------------------------------------------
def rmse(predictions, targets):
    """
    Root mean square error
    
    :param predictions: np.ndarray
    :param targets: np.ndarray
    :return: np.ndarray
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

#-----------------------------------------------------------------------------------------------------------------------
@numba.vectorize([numba.float64(numba.float64),
                  numba.float32(numba.float32)])
def f2c(t):
    '''
    Converts a temperature value from Fahrenheit to Celsius
    
    :param t: temperature value, assumed to be in Fahrenheit
    :return: the Fahrenheit equivalent of the input Celsius value
    :rtype: scalar float (when used as a ufunc an array of floats is returned as a result of the call)
    '''
    return (t-32)*5.0/9

#-----------------------------------------------------------------------------------------------------------------------
def compute_days(initial_year,
                 total_months,
                 initial_month=1,
                 units_start_year=1800):
    '''
    Computes the "number of days" equivalent for regular, incremental monthly time steps given an initial year/month.
    Useful when using "days since <start_date>" as time units within a NetCDF dataset.
    
    :param initial_year: the initial year from which the day values should start, i.e. the first value in the output
                        array will correspond to the number of days between January of this initial year since January 
                        of the units start year
    :param initial_month: the month within the initial year from which the day values should start, with 1: January, 2: February, etc.
    :param total_months: the total number of monthly increments (time steps measured in days) to be computed
    :param units_start_year: the start year from which the monthly increments are computed, with time steps measured
                             in days since January of this starting year 
    :return: an array of time step increments, measured in days since midnight of January 1st of the units start year
    :rtype: ndarray of ints 
    '''

    # compute an offset from which the day values should begin 
    start_date = datetime(units_start_year, 1, 1)

    # initialize the list of day values we'll build
    days = np.empty(total_months, dtype=int)
    
    # loop over all time steps (months)
    for i in range(total_months):
        
        years = int((i + initial_month - 1) / 12)   # the number of years since the initial year 
        months = int((i + initial_month - 1) % 12)  # the number of months since January
        
        # cook up a datetime object for the current time step (month)
        current_date = datetime(initial_year + years, 1 + months, 1)
        
        # get the number of days since the initial date
        days[i] = (current_date - start_date).days
    
    return days

# #-----------------------------------------------------------------------------------------------------------------------
# def compute_days(initial_year, 
#                  total_months):
#     """
#     This function computes a series (list) of day values to correspond with the first day of the month for each month 
#     of a time series starting from an initial year.
#     
#     :param initial_year:
#     :param total_months: total number of months in the time series
#     :return: numpy array of integers corresponding to   
#     """
#     
#     # the date from which the returned array of day values are since (i.e. when using "days since <start_date>" as our units for time)    
#     start_date = datetime(initial_year, 1, 1)
#     
#     # initialize the list of day values we'll build
#     days = np.empty(total_months, dtype=int)
#     
#     # loop over all time steps (months)
#     for i in range(total_months):
#         years = int(i / 12)  # the number of years since the initial year 
#         months = int(i % 12) # the number of months since January
#         
#         # cook up a date for the current time step (month)
#         current_date = datetime(initial_year + years, 1 + months, 1)
#         
#         # leverage the difference between dates operation available with datetime objects
#         days[i] = (current_date - start_date).days
#     
#     return days
#
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
            _logger.error(message)
            raise ValueError(message)
    
    # otherwise make sure that we've been passed in a flat (1-D) array of values    
    elif len(shape) != 1:
        message = 'Values array has an invalid shape (not 1-D or 2-D): {}'.format(shape)
        _logger.error(message)
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
                           the first year for each division, with dimension 0: division, dimension 1: months (0 to total months - 1)
    :return: the original monthly values reshaped to 3-D (divisions, years, 12), within each division each row maps 
             to a year, with each column of the row matching to the corresponding calendar month
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
            _logger.error(message)
            raise ValueError(message)
    
    # otherwise make sure that we've been passed in a 2-D array of values    
    elif len(shape) != 2:
        message = 'Values array has an invalid shape (not 2-D or 3-D): {}'.format(shape)
        _logger.error(message)
        raise ValueError(message)

    # otherwise make sure that we've been passed in a 2-D array of values with the final dimension size == 12
    elif shape[1] != 12:
        message = 'Values array has an invalid shape (second/final dimension should be 12, but is not): {}'.format(shape)
        _logger.error(message)
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
    """
    Given an input array of values return a count of the zeros and non-missing values.
    Missing values assumed to be numpy.NaNs.
    
    :param values: array like object (numpy array, most likely)
    :return: two int scalars: 1) the count of zeros, and 2) the count of non-missing values  
    """
    
    # make sure we have a numpy array
    values = np.array(values)
    
    # count the number of zeros and non-missing (non-NaN) values
    zeros = values.size - np.count_nonzero(values)
    non_missings = np.count_nonzero(~np.isnan(values))

    return zeros, non_missings

#-----------------------------------------------------------------------------------------------------------------------
def print_years_months(values):
    """
    Takes an input array of value and prints it as if it were a 2-D array with (years, month) as dimensions, 
    with one year written per line and missing years listed as NaNs. Designed to accept an array of monthly values,
    with the initial value corresponding to January of the initial year.
    
    Useful for printing a timeseries of values when constructing a test fixture from running code that has results 
    we'd like to match in an unit test, etc.
    
    :param values: 
    """

    # reshape the array, go over the two dimensions and print
    values = reshape_to_years_months(values)
    for i in range(values.shape[0]):
        year_line = ''.join("%5.2f, " % (v) for v in values[i])
        print(year_line + ' \\')

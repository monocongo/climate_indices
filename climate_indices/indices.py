import logging
import numba
import numpy as np
from enum import Enum

from climate_indices import compute, palmer, thornthwaite, utils

#-------------------------------------------------------------------------------------------------------------------------------------------
class Distribution(Enum):
    """
    Enumeration type for distribution fittings used for SPI and SPEI.
    """
    pearson_type3 = 'Pearson Type III'
    gamma = 'gamma'

#-------------------------------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------------------------------------------------------------------
# valid upper and lower bounds for indices that are fitted/transformed to a distribution (SPI and SPEI)  
_FITTED_INDEX_VALID_MIN = -3.09
_FITTED_INDEX_VALID_MAX = 3.09

#-------------------------------------------------------------------------------------------------------------------------------------------
#@numba.jit
def spi(precips, 
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity):
    '''
    Computes SPI (Standardized Precipitation Index).
    
    :param precips: 1-D numpy array of precipitation values, in any units, first value assumed to correspond 
                    to January of the initial year if the periodicity is monthly, or January 1st of the initial
                    year if daily
    :param scale: number of time steps over which the values should be scaled before the index is computed
    :param distribution: distribution type to be used for the internal fitting/transform computation
    :param data_start_year: the initial year of the input precipitation dataset
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param periodicity: the periodicity of the time series represented by the input data, valid/supported values are 
                        'monthly' and 'daily'
                        'monthly' indicates an array of monthly values, assumed to span full years, i.e. the first 
                        value corresponds to January of the initial year and any missing final months of the final 
                        year filled with NaN values, with size == # of years * 12
                        'daily' indicates an array of full years of daily values with 366 days per year, as if each
                        year were a leap year and any missing final months of the final year filled with NaN values, 
                        with array size == (# years * 366)
    :return SPI values fitted to the gamma distribution at the specified time step scale, unitless
    :rtype: 1-D numpy.ndarray of floats of the same length as the input array of precipitation values
    '''

    # we expect to operate upon a 1-D array, so if we've been passed a 2-D array we flatten it, otherwise raise an error
    shape = precips.shape
    if len(shape) == 2:
        precips = precips.flatten()
    elif len(shape) != 1:
        message = 'Invalid shape of input array: {0} -- only 1-D and 2-D arrays are supported'.format(shape)
        _logger.error(message)
        raise ValueError(message)
        
    # if we're passed all missing values then we can't compute anything, return the same array of missing values
    if (np.ma.is_masked(precips) and precips.mask.all()) or np.all(np.isnan(precips)):
        return precips
        
    # remember the original length of the array, in order to facilitate returning an array of the same size
    original_length = precips.size
    
    # get a sliding sums array, with each time step's value scaled by the specified number of time steps
    scaled_precips = compute.sum_to_scale(precips, scale)

    # reshape precipitation values to (years, 12) for monthly, or to (years, 366) for daily
    if periodicity == 'monthly':
        
        scaled_precips = utils.reshape_to_2d(scaled_precips, 12)

    elif periodicity == 'daily':
        
        scaled_precips = utils.reshape_to_2d(scaled_precips, 366)
        
    else:
        
        raise ValueError('Invalid periodicity argument: %s' % periodicity)
    
    if distribution is Distribution.gamma:
        
        # fit the scaled values to a gamma distribution and transform to corresponding normalized sigmas 
        transformed_fitted_values = compute.transform_fitted_gamma(scaled_precips, 
                                                                   data_start_year,
                                                                   calibration_year_initial,
                                                                   calibration_year_final,
                                                                   periodicity)
    elif distribution is Distribution.pearson_type3:
        
        # fit the scaled values to a Pearson Type III distribution and transform to corresponding normalized sigmas 
        transformed_fitted_values = compute.transform_fitted_pearson(scaled_precips, 
                                                                     data_start_year,
                                                                     calibration_year_initial,
                                                                     calibration_year_final,
                                                                     periodicity)
        
    # clip values to within the valid range, reshape the array back to 1-D
    spi = np.clip(transformed_fitted_values, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX).flatten()
    
    # return the original size array 
    return spi[0:original_length]

#-------------------------------------------------------------------------------------------------------------------------------------------
#@numba.jit
def spei(scale,
         distribution,
         periodicity,
         data_start_year,
         calibration_year_initial,
         calibration_year_final,
         precips_mm,
         pet_mm=None,
         temps_celsius=None,
         latitude_degrees=None):
    '''
    Compute SPEI fitted to the gamma distribution.
    
    PET values are subtracted from the precipitation values to come up with an array of (P - PET) values, which is 
    then scaled to the specified months scale and finally fitted/transformed to SPEI values corresponding to the
    input precipitation time series.

    If an input array of temperature values is provided then PET values are computed internally using the input 
    temperature array, data start year, and latitude value (all three of which are required in combination). 
    In this case an input array of PET values should not be specified and if so will result in an error being 
    raised indicating invalid arguments.
    
    If an input array of PET values is provided then neither an input array of temperature values nor a latitude 
    should be specified, and if so will result in an error being raised indicating invalid arguments.
        
    :param scale: the number of months over which the values should be scaled before computing the indicator
    :param distribution: distribution type to be used for the internal fitting/transform computation
    :param periodicity: the periodicity of the time series represented by the input data, valid/supported values are 
                        'monthly' and 'daily'
                        'monthly' indicates an array of monthly values, assumed to span full years, i.e. the first 
                        value corresponds to January of the initial year and any missing final months of the final 
                        year filled with NaN values, with size == # of years * 12
                        'daily' indicates an array of full years of daily values with 366 days per year, as if each
                        year were a leap year and any missing final months of the final year filled with NaN values, 
                        with array size == (# years * 366)
    :param precips_mm: an array of monthly total precipitation values, in millimeters, should be of the same size 
                       (and shape?) as the input temperature array
    :param pet_mm: an array of monthly PET values, in millimeters, should be of the same size (and shape?) as the input 
                   precipitation array, must be unspecified or None if using an array of temperature values as input
    :param temps_celsius: an array of monthly average temperature values, in degrees Celsius, should be of the same size 
                          (and shape?) as the input precipitation array, must be unspecified or None if using an array 
                          of PET values as input
    :param data_start_year: the initial year of the input datasets (assumes that the two inputs cover the same period)
    :param latitude_degrees: the latitude of the location, in degrees north, must be unspecified or None if using 
                             an array of PET values as an input, and must be specified if using an array of temperatures 
                             as input, valid range is -90.0 to 90.0 (inclusive)
    :return: an array of SPEI values
    :rtype: numpy.ndarray of type float, of the same size and shape as the input temperature and precipitation arrays
    '''
                    
    # if we're passed all missing values then we can't compute anything, return the same array of missing values
    if (np.ma.is_masked(precips_mm) and precips_mm.mask.all()) or np.all(np.isnan(precips_mm)):
        return precips_mm

    # validate the function's argument combinations
    if temps_celsius is not None:
        
        # since we have temperature then it's expected that we'll compute PET internally, so we shouldn't have PET as an input
        if pet_mm is not None:
            message = 'Incompatible arguments: either temperature or PET arrays can be specified as arguments, but not both' 
            _logger.error(message)
            raise ValueError(message)
        
        # we'll need both the latitude and data start year in order to compute PET 
        elif (latitude_degrees is None) or (data_start_year is None):
            message = 'Missing arguments: since temperature is provided as an input then both latitude ' + \
                      'and the data start year must also be specified, and one or both is not'
            _logger.error(message)
            raise ValueError(message)

        # validate that the two input arrays are compatible
        elif precips_mm.size != temps_celsius.size:
            message = 'Incompatible precipitation and temperature arrays'
            _logger.error(message)
            raise ValueError(message)

        elif periodicity != 'monthly':
            # our PET currently uses a monthly version of Thornthwaite's equation and therefore's only valid for monthly 
            message = 'Unsupported periodicity: \'{0}\' '.format(periodicity) + \
                      '-- only monthly time series is supported when providing temperature and latitude inputs' 
            _logger.error(message)
            raise ValueError(message)

        # compute PET
        pet_mm = pet(temps_celsius, latitude_degrees, data_start_year)

    elif pet_mm is not None:
        
        # make sure there's no confusion by not allowing a user to specify unnecessary parameters 
        if latitude_degrees is not None:
            message = 'Invalid argument: since PET is provided as an input then latitude must be absent'
            _logger.error(message)
            raise ValueError(message)
            
        # validate that the two input arrays are compatible
        elif precips_mm.size != pet_mm.size:
            message = 'Incompatible precipitation and PET arrays'
            _logger.error(message)
            raise ValueError(message)

    else:
        
        message = 'Neither temperature nor PET array was specified, one or the other is required for SPEI'
        _logger.error(message)
        raise ValueError(message)

    # subtract the PET from precipitation, adding an offset to ensure that all values are positive
    p_minus_pet = (precips_mm.flatten() - pet_mm.flatten()) + 1000.0
        
    # remember the original length of the input array, in order to facilitate returning an array of the same size
    original_length = precips_mm.size
    
    # get a sliding sums array, with each element's value scaled by the specified number of time steps
    scaled_values = compute.sum_to_scale(p_minus_pet, scale)

    if distribution is Distribution.gamma:

        # fit the scaled values to a gamma distribution and transform to corresponding normalized sigmas 
        transformed_fitted_values = compute.transform_fitted_gamma(scaled_values,
                                                                   data_start_year, 
                                                                   calibration_year_initial,
                                                                   calibration_year_final,
                                                                   periodicity)
    
    elif distribution is Distribution.pearson_type3:
    
        # fit the scaled values to a Pearson Type III distribution and transform to corresponding normalized sigmas 
        transformed_fitted_values = compute.transform_fitted_pearson(scaled_values, 
                                                                     data_start_year,
                                                                     calibration_year_initial,
                                                                     calibration_year_final,
                                                                     periodicity)
        
    # clip values to within the valid range, reshape the array back to 1-D
    spei = np.clip(transformed_fitted_values, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX).flatten()
    
    # return the original size array 
    return spei[0:original_length]

#-------------------------------------------------------------------------------------------------------------------------------------------
@numba.jit
def scpdsi(precip_time_series,
           pet_time_series,
           awc,
           data_start_year,
           calibration_start_year,
           calibration_end_year):
    '''
    This function computes the self-calibrated Palmer Drought Severity Index (scPDSI), Palmer Drought Severity Index 
    (PDSI), Palmer Hydrological Drought Index (PHDI), Palmer Modified Drought Index (PMDI), and Palmer Z-Index.
    
    :param precip_time_series: time series of precipitation values, in inches
    :param pet_time_series: time series of PET values, in inches
    :param awc: available water capacity (soil constant), in inches
    :param data_start_year: initial year of the input precipitation and PET datasets, 
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period 
    :param calibration_end_year: final year of the calibration period 
    :return: four numpy arrays containing SCPDSI, PDSI, PHDI, and Z-Index values respectively 
    '''
    
    return palmer.scpdsi(precip_time_series,
                         pet_time_series,
                         awc,
                         data_start_year,
                         calibration_start_year,
                         calibration_end_year)
    
#-------------------------------------------------------------------------------------------------------------------------------------------
@numba.jit
def pdsi(precip_time_series,
         pet_time_series,
         awc,
         data_start_year,
         calibration_start_year,
         calibration_end_year):
    '''
    This function computes the Palmer Drought Severity Index (PDSI), Palmer Hydrological Drought Index (PHDI), 
    and Palmer Z-Index.
    
    :param precip_time_series: time series of monthly precipitation values, in inches
    :param pet_time_series: time series of monthly PET values, in inches
    :param awc: available water capacity (soil constant), in inches
    :param data_start_year: initial year of the input precipitation and PET datasets, 
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period 
    :param calibration_end_year: final year of the calibration period 
    :return: four numpy arrays containing PDSI, PHDI, PMDI, and Z-Index values respectively 
    '''
    
    return palmer.pdsi(precip_time_series,
                       pet_time_series,
                       awc,
                       data_start_year,
                       calibration_start_year,
                       calibration_end_year)
    
#-------------------------------------------------------------------------------------------------------------------------------------------
@numba.jit     
def percentage_of_normal(values, 
                         scale,
                         data_start_year,
                         calibration_start_year,
                         calibration_end_year,
                         periodicity):
    '''
    This function finds the percent of normal values (average of each calendar month or day over a specified 
    calibration period of years) for a specified time steps scale. The normal precipitation for each calendar time step 
    is computed for the specified time steps scale, and then each time step's scaled value is compared against the 
    corresponding calendar time step's average to determine the percentage of normal. The period that defines the 
    normal is described by the calibration start and end years arguments. The calibration period typically used  
    for US climate monitoring is 1981-2010. 
    
    :param values: 1-D numpy array of precipitation values, any length, initial value assumed to be January of the data 
                   start year (January 1st of the start year if daily periodicity), see the description of the 
                   *periodicity* argument below for further clarification
    :param scale: integer number of months over which the normal value is computed (eg 3-months, 6-months, etc.)
    :param data_start_year: the initial year of the input monthly values array
    :param calibration_start_year: the initial year of the calibration period over which the normal average for each  
                                   calendar time step is computed 
    :param calibration_start_year: the final year of the calibration period over which the normal average for each 
                                   calendar time step is computed 
    :param periodicity: the periodicity of the time series represented by the input data, valid/supported values are 
                        'monthly' and 'daily'
                        'monthly' indicates an array of monthly values, assumed to span full years, i.e. the first 
                        value corresponds to January of the initial year and any missing final months of the final 
                        year filled with NaN values, with size == # of years * 12
                        'daily' indicates an array of full years of daily values with 366 days per year, as if each
                        year were a leap year and any missing final months of the final year filled with NaN values, 
                        with array size == (# years * 366)
    :return: percent of normal precipitation values corresponding to the scaled precipitation values array   
    :rtype: numpy.ndarray of type float
    '''

    # if doing monthly then we'll use 12 periods, corresponding to calendar months, if daily assume years w/366 days
    if periodicity == 'monthly':
        periodicity = 12
    elif periodicity == 'daily':
        periodicity = 366
    else:
        message = 'Invalid periodicity argument: \'{0}\''.format(periodicity)
        _logger.error(message)
        raise ValueError(message)
    
    # bypass processing if all values are masked    
    if np.ma.is_masked(values) and values.mask.all():
        return values
    
    # make sure we've been provided with sane calibration limits
    if data_start_year > calibration_start_year:
        raise ValueError('Invalid start year arguments (data and/or calibration): calibration start year ' + \
                         'is before the data start year')
    elif ((calibration_end_year - calibration_start_year + 1) * 12) > values.size:
        raise ValueError('Invalid calibration period specified: total calibration years exceeds the actual ' + \
                         'number of years of data')
        
    # get an array containing a sliding sum on the specified time step scale -- i.e. if the scale is 3 then the first 
    # two elements will be np.NaN, since we need 3 elements to get a sum, and then from the third element to the end 
    # the values will equal the sum of the corresponding time step plus the values of the two previous time steps
    scale_sums = compute.sum_to_scale(values, scale)
    
    # extract the timesteps over which we'll compute the normal average for each time step of the year
    calibration_years = calibration_end_year - calibration_start_year + 1
    calibration_start_index = (calibration_start_year - data_start_year) * periodicity
    calibration_end_index = calibration_start_index + (calibration_years * periodicity)
    calibration_period_sums = scale_sums[calibration_start_index:calibration_end_index]
    
    # for each time step in the calibration period, get the average of the scale sum 
    # for that calendar time step (i.e. average all January sums, then all February sums, etc.) 
    averages = np.full((periodicity,), np.nan)
    for i in range(periodicity):
        averages[i] = np.nanmean(calibration_period_sums[i::periodicity])
    
    #TODO replace the below loop with a vectorized implementation
    # for each time step of the scale_sums array find its corresponding
    # percentage of the time steps scale average for its respective calendar time step
    percentages_of_normal = np.full(scale_sums.shape, np.nan)
    for i in range(scale_sums.size):

        # make sure we don't have a zero divisor
        divisor = averages[i % periodicity]
        if divisor > 0.0:
            
            percentages_of_normal[i] = scale_sums[i] / divisor
    
    return percentages_of_normal
    
#-------------------------------------------------------------------------------------------------------------------------------------------
@numba.jit
def pet(temperature_celsius,
        latitude_degrees,
        data_start_year):

    '''
    This function computes potential evapotranspiration (PET) using Thornthwaite's equation.
    
    :param temperature_celsius: an array of average temperature values, in degrees Celsius
    :param latitude_degrees: the latitude of the location, in degrees north, must be within range [-90.0 ... 90.0] (inclusive), otherwise 
                             a ValueError is raised
    :param data_start_year: the initial year of the input dataset
    :return: an array of PET values, of the same size and shape as the input temperature values array, in millimeters/time step
    :rtype: 1-D numpy.ndarray of floats
    '''
    
    # make sure we're not dealing with all NaN values
    if np.ma.isMaskedArray(temperature_celsius) and temperature_celsius.count() == 0:
        
        # we started with all NaNs for the temperature, so just return the same as PET
        return temperature_celsius

    else:
        
        # we were passed a vanilla Numpy array, look for indices where the value == NaN
        nan_indices = np.isnan(temperature_celsius)
        if np.all(nan_indices):
        
            # we started with all NaNs for the temperature, so just return the same
            return temperature_celsius
        
    # make sure we're not dealing with a NaN or out-of-range latitude value
    if latitude_degrees is not None and not np.isnan(latitude_degrees) and \
        (latitude_degrees < 90.0) and (latitude_degrees > -90.0):
        
        # compute and return the PET values using Thornthwaite's equation
        return thornthwaite.potential_evapotranspiration(temperature_celsius, latitude_degrees, data_start_year)
        
    else:
        message = 'Invalid latitude value: {0} (must be in degrees north, between -90.0 and 90.0 inclusive)'.format(latitude_degrees)
        _logger.error(message)
        raise ValueError(message)

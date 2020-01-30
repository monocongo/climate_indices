from enum import Enum
import logging
from typing import Dict

import numba
import numpy as np

from climate_indices import compute, eto, palmer, utils

# declare the names that should be included in the public API for this module
__all__ = ["pdsi", "percentage_of_normal", "pet", "scpdsi", "spei", "spi"]


# ------------------------------------------------------------------------------
class Distribution(Enum):
    """
    Enumeration type for distribution fittings used for SPI and SPEI.
    """

    pearson = "pearson"
    gamma = "gamma"


# ------------------------------------------------------------------------------
# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.DEBUG)

# ------------------------------------------------------------------------------
# valid upper and lower bounds for indices that are fitted/transformed to a distribution (SPI and SPEI)
_FITTED_INDEX_VALID_MIN = -3.09
_FITTED_INDEX_VALID_MAX = 3.09


# ------------------------------------------------------------------------------
@numba.jit
def spi(
        values: np.ndarray,
        scale: int,
        distribution: Distribution,
        data_start_year: int,
        calibration_year_initial: int,
        calibration_year_final: int,
        periodicity: compute.Periodicity,
        fitting_params: Dict = None,
) -> np.ndarray:
    """
    Computes SPI (Standardized Precipitation Index).

    :param values: 1-D numpy array of precipitation values, in any units,
        first value assumed to correspond to January of the initial year if
        the periodicity is monthly, or January 1st of the initial year if daily
    :param scale: number of time steps over which the values should be scaled
        before the index is computed
    :param distribution: distribution type to be used for the internal
        fitting/transform computation
    :param data_start_year: the initial year of the input precipitation dataset
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param periodicity: the periodicity of the time series represented by the
        input data, valid/supported values are 'monthly' and 'daily'
        'monthly' indicates an array of monthly values, assumed to span full
         years, i.e. the first value corresponds to January of the initial year
         and any missing final months of the final year filled with NaN values,
         with size == # of years * 12
         'daily' indicates an array of full years of daily values with 366 days
         per year, as if each year were a leap year and any missing final months
         of the final year filled with NaN values, with array size == (# years * 366)
    :param fitting_params: optional dictionary of pre-computed distribution
        fitting parameters, if the distribution is gamma then this dict should
        contain two arrays, keyed as "alphas" and "betas", and if the
        distribution is Pearson then this dict should contain four arrays keyed
        as "probabilities_of_zero", "locs", "scales", and "skews"
    :return SPI values fitted to the gamma distribution at the specified time
        step scale, unitless
    :rtype: 1-D numpy.ndarray of floats of the same length as the input array
        of precipitation values
    """

    # we expect to operate upon a 1-D array, so if we've been passed a 2-D array
    # then we flatten it, otherwise raise an error
    shape = values.shape
    if len(shape) == 2:
        values = values.flatten()
    elif len(shape) != 1:
        message = "Invalid shape of input array: {shape}".format(shape=shape) + \
                  " -- only 1-D and 2-D arrays are supported"
        _logger.error(message)
        raise ValueError(message)

    # if we're passed all missing values then we can't compute
    # anything, so we return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # clip any negative values to zero
    if np.amin(values) < 0.0:
        _logger.warn("Input contains negative values -- all negatives clipped to zero")
        values = np.clip(values, a_min=0.0, a_max=None)

    # remember the original length of the array, in order to facilitate
    # returning an array of the same size
    original_length = values.size

    # get a sliding sums array, with each time step's value scaled
    # by the specified number of time steps
    values = compute.sum_to_scale(values, scale)

    # reshape precipitation values to (years, 12) for monthly,
    # or to (years, 366) for daily
    if periodicity is compute.Periodicity.monthly:

        values = utils.reshape_to_2d(values, 12)

    elif periodicity is compute.Periodicity.daily:

        values = utils.reshape_to_2d(values, 366)

    else:

        raise ValueError("Invalid periodicity argument: %s" % periodicity)

    if distribution is Distribution.gamma:

        # get (optional) fitting parameters if provided
        if fitting_params is not None:
            alphas = fitting_params["alpha"]
            betas = fitting_params["beta"]
        else:
            alphas = None
            betas = None

        # fit the scaled values to a gamma distribution
        # and transform to corresponding normalized sigmas
        values = compute.transform_fitted_gamma(
            values,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            periodicity,
            alphas,
            betas,
        )
    elif distribution is Distribution.pearson:

        # get (optional) fitting parameters if provided
        if fitting_params is not None:
            probabilities_of_zero = fitting_params["prob_zero"]
            locs = fitting_params["loc"]
            scales = fitting_params["scale"]
            skews = fitting_params["skew"]
        else:
            probabilities_of_zero = None
            locs = None
            scales = None
            skews = None

        # fit the scaled values to a Pearson Type III distribution
        # and transform to corresponding normalized sigmas
        values = compute.transform_fitted_pearson(
            values,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            periodicity,
            probabilities_of_zero,
            locs,
            scales,
            skews,
        )

    else:

        message = "Unsupported distribution argument: " + \
                  "{dist}".format(dist=distribution)
        _logger.error(message)
        raise ValueError(message)

    # clip values to within the valid range, reshape the array back to 1-D
    values = np.clip(values, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX).flatten()

    # return the original size array
    return values[0:original_length]


# ------------------------------------------------------------------------------
@numba.jit
def spei(
        precips_mm: np.ndarray,
        pet_mm: np.ndarray,
        scale: int,
        distribution: Distribution,
        periodicity: compute.Periodicity,
        data_start_year: int,
        calibration_year_initial: int,
        calibration_year_final: int,
        fitting_params: dict = None,
) -> np.ndarray:
    """
    Compute SPEI fitted to the gamma distribution.

    PET values are subtracted from the precipitation values to come up with an array
    of (P - PET) values, which is then scaled to the specified months scale and
    finally fitted/transformed to SPEI values corresponding to the input
    precipitation time series.

    :param precips_mm: an array of monthly total precipitation values,
        in millimeters, should be of the same size (and shape?) as the input PET array
    :param pet_mm: an array of monthly PET values, in millimeters,
        should be of the same size (and shape?) as the input precipitation array
    :param scale: the number of months over which the values should be scaled
        before computing the indicator
    :param distribution: distribution type to be used for the internal
        fitting/transform computation
    :param periodicity: the periodicity of the time series represented by the
        input data, valid/supported values are 'monthly' and 'daily'
        'monthly' indicates an array of monthly values, assumed to span full
         years, i.e. the first value corresponds to January of the initial year
         and any missing final months of the final year filled with NaN values,
         with size == # of years * 12
         'daily' indicates an array of full years of daily values with 366 days
         per year, as if each year were a leap year and any missing final months
         of the final year filled with NaN values, with array size == (# years * 366)
    :param data_start_year: the initial year of the input datasets (assumes that
        the two inputs cover the same period)
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param fitting_params: optional dictionary of pre-computed distribution
        fitting parameters, if the distribution is gamma then this dict should
        contain two arrays, keyed as "alphas" and "betas", and if the
        distribution is Pearson then this dict should contain four arrays keyed
        as "probabilities_of_zero", "locs", "scales", and "skews"
    :return: an array of SPEI values
    :rtype: numpy.ndarray of type float, of the same size and shape as the input
        PET and precipitation arrays
    """

    # if we're passed all missing values then we can't compute anything,
    # so we return the same array of missing values
    if (np.ma.is_masked(precips_mm) and precips_mm.mask.all()) \
            or np.all(np.isnan(precips_mm)):
        return precips_mm

    # validate that the two input arrays are compatible
    if precips_mm.size != pet_mm.size:
        message = "Incompatible precipitation and PET arrays"
        _logger.error(message)
        raise ValueError(message)

    # clip any negative values to zero
    if np.amin(precips_mm) < 0.0:
        _logger.warn("Input contains negative values -- all negatives clipped to zero")
        precips_mm = np.clip(precips_mm, a_min=0.0, a_max=None)

    # subtract the PET from precipitation, adding an offset
    # to ensure that all values are positive
    p_minus_pet = (precips_mm.flatten() - pet_mm.flatten()) + 1000.0

    # remember the original length of the input array, in order to facilitate
    # returning an array of the same size
    original_length = precips_mm.size

    # get a sliding sums array, with each element's value
    # scaled by the specified number of time steps
    scaled_values = compute.sum_to_scale(p_minus_pet, scale)

    if distribution is Distribution.gamma:

        # get (optional) fitting parameters if provided
        if fitting_params is not None:
            alphas = fitting_params["alphas"]
            betas = fitting_params["betas"]
        else:
            alphas = None
            betas = None

        # fit the scaled values to a gamma distribution and
        # transform to corresponding normalized sigmas
        transformed_fitted_values = \
            compute.transform_fitted_gamma(
                scaled_values,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                periodicity,
                alphas,
                betas,
            )

    elif distribution is Distribution.pearson:

        # get (optional) fitting parameters if provided
        if fitting_params is not None:
            probabilities_of_zero = fitting_params["probabilities_of_zero"]
            locs = fitting_params["locs"]
            scales = fitting_params["scales"]
            skews = fitting_params["skews"]
        else:
            probabilities_of_zero = None
            locs = None
            scales = None
            skews = None

        # fit the scaled values to a Pearson Type III distribution
        # and transform to corresponding normalized sigmas
        transformed_fitted_values = \
            compute.transform_fitted_pearson(
                scaled_values,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                periodicity,
                probabilities_of_zero,
                locs,
                scales,
                skews,
            )

    else:
        message = "Unsupported distribution argument: " + \
                  "{dist}".format(dist=distribution)
        _logger.error(message)
        raise ValueError(message)

    # clip values to within the valid range, reshape the array back to 1-D
    values = \
        np.clip(transformed_fitted_values,
                _FITTED_INDEX_VALID_MIN,
                _FITTED_INDEX_VALID_MAX).flatten()

    # return the original size array
    return values[0:original_length]


# ------------------------------------------------------------------------------
@numba.jit
def scpdsi(precip_time_series: np.ndarray,
           pet_time_series: np.ndarray,
           awc: float,
           data_start_year: int,
           calibration_start_year: int,
           calibration_end_year: int):
    """
    This function computes the self-calibrated Palmer Drought Severity Index
    (scPDSI), Palmer Drought Severity Index (PDSI), Palmer Hydrological Drought
    Index (PHDI), Palmer Modified Drought Index (PMDI), and Palmer Z-Index.

    :param precip_time_series: time series of precipitation values, in inches
    :param pet_time_series: time series of PET values, in inches
    :param awc: available water capacity (soil constant), in inches
    :param data_start_year: initial year of the input precipitation and PET datasets,
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period
    :param calibration_end_year: final year of the calibration period
    :return: five numpy arrays containing SCPDSI, PDSI, PHDI, PMDI, and Z-Index values respectively
    """

    return palmer.scpdsi(precip_time_series,
                         pet_time_series,
                         awc,
                         data_start_year,
                         calibration_start_year,
                         calibration_end_year)


# ------------------------------------------------------------------------------
@numba.jit
def pdsi(precip_time_series: np.ndarray,
         pet_time_series: np.ndarray,
         awc: float,
         data_start_year: int,
         calibration_start_year: int,
         calibration_end_year: int):
    """
    This function computes the Palmer Drought Severity Index (PDSI), Palmer
    Hydrological Drought Index (PHDI), and Palmer Z-Index.

    :param precip_time_series: time series of monthly precipitation values, in inches
    :param pet_time_series: time series of monthly PET values, in inches
    :param awc: available water capacity (soil constant), in inches
    :param data_start_year: initial year of the input precipitation and PET datasets,
                            both of which are assumed to start in January of this year
    :param calibration_start_year: initial year of the calibration period
    :param calibration_end_year: final year of the calibration period
    :return: four numpy arrays containing PDSI, PHDI, PMDI, and Z-Index values respectively
    """

    return palmer.pdsi(precip_time_series,
                       pet_time_series,
                       awc,
                       data_start_year,
                       calibration_start_year,
                       calibration_end_year)


# ------------------------------------------------------------------------------
@numba.jit
def percentage_of_normal(values: np.ndarray,
                         scale: int,
                         data_start_year: int,
                         calibration_start_year: int,
                         calibration_end_year: int,
                         periodicity: compute.Periodicity) -> np.ndarray:
    """
    This function finds the percent of normal values (average of each calendar
    month or day over a specified calibration period of years) for a specified
    time steps scale. The normal precipitation for each calendar time step is
    computed for the specified time steps scale, and then each time step's
    scaled value is compared against the corresponding calendar time step's
    average to determine the percentage of normal. The period that defines the
    normal is described by the calibration start and end years arguments.
    The calibration period typically used for US climate monitoring is 1981-2010.

    :param values: 1-D numpy array of precipitation values, any length, initial
        value assumed to be January of the data start year (January 1st of the
        start year if daily periodicity), see the description of the
        *periodicity* argument below for further clarification
    :param scale: integer number of months over which the normal value is
        computed (eg 3-months, 6-months, etc.)
    :param data_start_year: the initial year of the input monthly values array
    :param calibration_start_year: the initial year of the calibration period
        over which the normal average for each calendar time step is computed
    :param calibration_end_year: the final year of the calibration period over
        which the normal average for each calendar time step is computed
    :param periodicity: the periodicity of the time series represented by the
        input data, valid/supported values are 'monthly' and 'daily'
        'monthly' indicates an array of monthly values, assumed to span full
         years, i.e. the first value corresponds to January of the initial year
         and any missing final months of the final year filled with NaN values,
         with size == # of years * 12
         'daily' indicates an array of full years of daily values with 366 days
         per year, as if each year were a leap year and any missing final months
         of the final year filled with NaN values, with array size == (# years * 366)
    :return: percent of normal precipitation values corresponding to the
        scaled precipitation values array
    :rtype: numpy.ndarray of type float
    """

    # validate the scale argument
    if (scale is None) or (scale < 1):
        message = "Invalid scale argument: '{0}'".format(scale)
        _logger.error(message)
        raise ValueError(message)

    # if doing monthly then we'll use 12 periods, corresponding to calendar
    # months, if daily assume years w/366 days
    if periodicity is compute.Periodicity.monthly:
        periodicity = 12
    elif periodicity is compute.Periodicity.daily:
        periodicity = 366
    else:
        message = "Invalid periodicity argument: '{0}'".format(periodicity)
        _logger.error(message)
        raise ValueError(message)

    # bypass processing if all values are masked
    if np.ma.is_masked(values) and values.mask.all():
        return values

    # make sure we've been provided with sane calibration limits
    if data_start_year > calibration_start_year:
        raise ValueError(
            "Invalid start year arguments (data and/or calibration): "
            "calibration start year is before the data start year",
        )
    elif ((calibration_end_year - calibration_start_year + 1) * 12) > values.size:
        raise ValueError(
            "Invalid calibration period specified: total calibration years "
            "exceeds the actual number of years of data",
        )

    # get an array containing a sliding sum on the specified time step
    # scale -- i.e. if the scale is 3 then the first two elements will be
    # np.NaN, since we need 3 elements to get a sum, and then from the third
    # element to the end the values will equal the sum of the corresponding
    # time step plus the values of the two previous time steps
    scale_sums = compute.sum_to_scale(values, scale)

    # extract the timesteps over which we'll compute the normal
    # average for each time step of the year
    calibration_years = calibration_end_year - calibration_start_year + 1
    calibration_start_index = (calibration_start_year - data_start_year) * periodicity
    calibration_end_index = calibration_start_index + (calibration_years * periodicity)
    calibration_period_sums = scale_sums[calibration_start_index:calibration_end_index]

    # for each time step in the calibration period, get the average of
    # the scale sum for that calendar time step (i.e. average all January sums,
    # then all February sums, etc.)
    averages = np.full((periodicity,), np.nan)
    for i in range(periodicity):
        averages[i] = np.nanmean(calibration_period_sums[i::periodicity])

    # TODO replace the below loop with a vectorized implementation
    # for each time step of the scale_sums array find its corresponding
    # percentage of the time steps scale average for its respective calendar time step
    percentages_of_normal = np.full(scale_sums.shape, np.nan)
    for i in range(scale_sums.size):

        # make sure we don't have a zero divisor
        divisor = averages[i % periodicity]
        if divisor > 0.0:

            percentages_of_normal[i] = scale_sums[i] / divisor

    return percentages_of_normal


# ------------------------------------------------------------------------------
@numba.jit
def pet(temperature_celsius: np.ndarray,
        latitude_degrees: float,
        data_start_year: int) -> np.ndarray:
    """
    This function computes potential evapotranspiration (PET) using
    Thornthwaite's equation.

    :param temperature_celsius: an array of average temperature values,
        in degrees Celsius
    :param latitude_degrees: the latitude of the location, in degrees north,
        must be within range [-90.0 ... 90.0] (inclusive), otherwise a
        ValueError is raised
    :param data_start_year: the initial year of the input dataset
    :return: an array of PET values, of the same size and shape as the input
        temperature values array, in millimeters/time step
    :rtype: 1-D numpy.ndarray of floats
    """

    # make sure we're not dealing with all NaN values
    if np.ma.isMaskedArray(temperature_celsius) and (temperature_celsius.count() == 0):

        # we started with all NaNs for the temperature, so just return the same as PET
        return temperature_celsius

    else:

        # we were passed a vanilla Numpy array, look for indices where the value == NaN
        if np.all(np.isnan(temperature_celsius)):

            # we started with all NaNs for the temperature, so just return the same
            return temperature_celsius

    # If we've been passed an array of latitude values then just use
    # the first one -- useful when applying this function with xarray.GroupBy
    # or numpy.apply_along_axis() where we've had to duplicate values in a 3-D
    # array of latitudes in order to correspond with a 3-D array of temperatures.
    if isinstance(latitude_degrees, np.ndarray) and (latitude_degrees.size > 1):
        latitude_degrees = latitude_degrees.flat[0]

    # make sure we're not dealing with a NaN or out-of-range latitude value
    if ((latitude_degrees is not None)
            and not np.isnan(latitude_degrees)
            and (latitude_degrees < 90.0)
            and (latitude_degrees > -90.0)):

        # compute and return the PET values using Thornthwaite's equation
        return eto.eto_thornthwaite(
            temperature_celsius,
            latitude_degrees,
            data_start_year,
        )

    else:
        message = ("Invalid latitude value: " + str(latitude_degrees) +
                   " (must be in degrees north, between -90.0 and " +
                   "90.0 inclusive)")
        _logger.error(message)
        raise ValueError(message)

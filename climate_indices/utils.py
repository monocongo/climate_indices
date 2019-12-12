import calendar
from datetime import datetime
import logging

import numba
import numpy as np


# ------------------------------------------------------------------------------
# set up a basic, global _logger
def get_logger(name, level):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d  %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# ------------------------------------------------------------------------------
def sign_change(
        a: np.ndarray,
        b: np.ndarray,
) -> np.ndarray:
    """
    Given two same-sized arrays of floats return an array of booleans indicating
    if a sign change occurs at the corresponding index.

    :param a: array of floats
    :param b: array of floats
    :return: array of booleans of same size as input arrays
    """

    if a.size != b.size:
        raise ValueError("Mismatched input arrays")

    # use the shape of the first array as the shape of the array we'll return
    original_shape = a.shape

    # get the sign value for each element
    sign_a = np.sign(a.flatten())
    sign_b = np.sign(b.flatten())

    # sign change between the two where values unequal
    sign_changes = sign_a != sign_b

    return np.reshape(sign_changes, original_shape)


# ------------------------------------------------------------------------------
def is_data_valid(
        data: np.ndarray,
) -> bool:
    """
    Returns whether or not an array is valid, i.e. a supported array type
    (ndarray or MaskArray) which is not all-NaN.

    :param data: data object, expected as either numpy.ndarry or numpy.ma.MaskArray
    :return True if array is non-NaN for at least one element
        and is an array type valid for processing by other modules
    :rtype: boolean
    """

    # make sure we're not dealing with all NaN values
    if np.ma.isMaskedArray(data):

        valid_flag = bool(data.count())

    elif isinstance(data, np.ndarray):

        valid_flag = not np.all(np.isnan(data))

    else:
        _logger.warning("Invalid data type")
        valid_flag = False

    return valid_flag


# ------------------------------------------------------------------------------
def rmse(
        predictions: np.ndarray,
        targets: np.ndarray,
) -> np.ndarray:
    """
    Root mean square error

    :param predictions: np.ndarray
    :param targets: np.ndarray
    :return: np.ndarray
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


# ------------------------------------------------------------------------------
def compute_days(
        initial_year: int,
        total_months: int,
        initial_month=1,
        units_start_year=1800,
) -> np.ndarray:
    """
    Computes the "number of days" equivalent for regular, incremental monthly
    time steps given an initial year/month. Useful when using "days since
    <start_date>" as time units within a NetCDF dataset.

    :param initial_year: the initial year from which the day values
        should start, i.e. the first value in the output array will correspond
        to the number of days between January of this initial year since January
        of the units start year
    :param initial_month: the month within the initial year from which the day
        values should start, with 1: January, 2: February, etc.
    :param total_months: the total number of monthly increments (time steps
        measured in days) to be computed
    :param units_start_year: the start year from which the monthly increments
        are computed, with time steps measured in days since January of this
        starting year
    :return: an array of time step increments, measured in days since midnight
        of January 1st of the units start year
    :rtype: ndarray of ints
    """

    # compute an offset from which the day values should begin
    start_date = datetime(units_start_year, 1, 1)

    # initialize the list of day values we'll build
    days = np.empty(total_months, dtype=int)

    # loop over all time steps (months)
    for i in range(total_months):

        years = int(
            (i + initial_month - 1) / 12
        )  # the number of years since the initial year
        months = int((i + initial_month - 1) % 12)  # the number of months since January

        # cook up a datetime object for the current time step (month)
        current_date = datetime(initial_year + years, 1 + months, 1)

        # get the number of days since the initial date
        days[i] = (current_date - start_date).days

    return days


# ------------------------------------------------------------------------------
@numba.jit
def reshape_to_2d(
        values: np.ndarray,
        second_axis_length: int,
) -> np.ndarray:
    """
    :param values: an 1-D numpy.ndarray of values
    :param second_axis_length:
    :return: the original values reshaped to 2-D, with shape
        [int(original length / second axis length), second axis length]
    :rtype: 2-D numpy.ndarray of floats
    """

    # if we've been passed a 2-D array with valid shape then let it pass through
    shape = values.shape
    if len(shape) == 2:
        if shape[1] == second_axis_length:
            # data is already in the shape we want, return it unaltered
            return values
        else:
            message = "Values array has an invalid shape (2-D but second " + \
                      "dimension not {dim}".format(dim=second_axis_length) + \
                      "): {shape}".format(shape=shape)
            _logger.error(message)
            raise ValueError(message)

    # otherwise make sure that we've been passed a flat (1-D) array of values
    elif len(shape) != 1:
        message = "Values array has an invalid shape (not 1-D " + \
                  "or 2-D): {shape}".format(shape=shape)
        _logger.error(message)
        raise ValueError(message)

    # pad the end of the original array in order
    # to have an ordinal increment, if necessary
    final_year_values = shape[0] % second_axis_length
    if final_year_values > 0:
        pads = second_axis_length - final_year_values
        values = np.pad(values,
                        pad_width=(0, pads),
                        mode="constant",
                        constant_values=np.NaN)

    # we should have an ordinal number of years now
    # (ordinally divisible by second_axis_length)
    first_axis_length = int(values.shape[0] / second_axis_length)

    # return the reshaped array
    return np.reshape(values, newshape=(first_axis_length, second_axis_length))


# ------------------------------------------------------------------------------
@numba.jit
def reshape_to_divs_years_months(
        monthly_values: np.ndarray,
) -> np.ndarray:
    """
    :param monthly_values: an 2-D numpy.ndarray of monthly values,
        assumed to start at January of the first year for each division,
        with dimension 0: division, dimension 1: months (0 to total months - 1)
    :return: the original monthly values reshaped to 3-D (divisions, years, 12),
        within each division each row maps to a year, with each column of
        the row matching to the corresponding calendar month
    :rtype: 3-D numpy.ndarray of floats
    """

    # if we've been passed a 3-D array with valid shape then let it pass through
    shape = monthly_values.shape
    if len(shape) == 3:
        if shape[2] == 12:
            # data is already in the shape we want, return it unaltered
            return monthly_values
        else:
            message = "Values array has an invalid shape (3-D but " + \
                      "third dimension is not 12): " + str(shape)
            _logger.error(message)
            raise ValueError(message)

    # otherwise make sure that we've been passed in a 2-D array of values
    elif len(shape) != 2:
        message = "Values array has an invalid shape (not 2-D or 3-D): " + str(shape)
        _logger.error(message)
        raise ValueError(message)

    # otherwise make sure that we've been passed in a 2-D array
    # of values with the final dimension size == 12
    elif shape[1] != 12:
        message = "Values array has an invalid shape (second dimension " + \
                  "should be 12, but is not): " + str(shape)
        _logger.error(message)
        raise ValueError(message)

    # we should have an ordinal number of years now (ordinally divisible by 12)
    total_years = int(monthly_values.shape[1] / 12)

    # reshape from (months) to (years, 12) in order
    # to have one year of months per row
    return np.reshape(monthly_values, (shape[0], total_years, 12))


# ------------------------------------------------------------------------------
@numba.jit
def gregorian_length_as_366day(
        length_gregorian: int,
        year_start: int,
) -> int:

    year = year_start
    remaining = length_gregorian
    length_366day = 0
    while remaining > 0:

        if calendar.isleap(year):
            days_in_current_year = 366
        else:
            days_in_current_year = 365

        if remaining >= days_in_current_year:
            length_366day += 366
        else:
            length_366day += remaining

        remaining -= days_in_current_year
        year += 1

    return length_366day


# ------------------------------------------------------------------------------
@numba.jit
def transform_to_366day(
        original: np.ndarray,
        year_start: int,
        total_years: int,
) -> np.ndarray:
    """
    Takes an array of daily values with only actual leap years represented
    as 366 day years (non-leap years with 365 days) and converts it to an array
    of daily values represented as containing full 366 day years as if each year
    is a leap year with computed/faux values for the Feb. 29th of each
    non-leap year.

    For example if provided an input array representing two years,
    we expect/assume that it will contain 730 elements if neither of the years
    represented are leap years (as indicated by the year start argument),
    or 731 elements if either of the two years is a leap year (i.e. a year with
    366 days). The resulting/transformed array will contain 732 elements -- 366
    for the leap year plus 366 for the non-leap year, with the element
    that corresponds to Feb. 29th in the non-leap year having a value that's an
    average of the Feb 28th and Mar. 1st values.

    :param original: 1-D array of daily values
    :param year_start: the year corresponding to the initial year of the input
        array, used to determine whether or not each increment of daily values
        represents an actual leap year
    :param total_years: the total number of years represented by the input array
    :return: 1-D array of values with size (total_years * 366)
    """
    # the original time series is assumed to be a one-dimensional
    # array of floats corresponding to a number of full years

    # validate the arguments
    if len(original.shape) > 1:
        message = "Invalid input array: only 1-D arrays are supported"
        _logger.error(message)
        raise ValueError(message)

    # allocate the new array for 366 daily values per year,
    # including a faux Feb 29 for non-leap years
    all_leap = np.full((total_years * 366,), np.NaN)

    # index of the first day of the year within the original and all_leap arrays
    original_index = 0
    all_leap_index = 0

    # loop over each year
    for year in range(year_start, year_start + total_years):

        if calendar.isleap(year):

            # write the next 366 days from the original time
            # series into the all_leap array
            all_leap[all_leap_index:(all_leap_index + 366)] = \
                original[original_index:(original_index + 366)]

            # increment the "start day of the current year" index for the original
            # so that the next iteration jumps ahead a full year
            original_index += 366

        else:

            # write the first 59 days (Jan 1 through Feb 28) from
            # the original time series into the all_leap array
            all_leap[all_leap_index:(all_leap_index + 59)] = \
                original[original_index:(original_index + 59)]

            # average the Feb 28th and March 1st values as the faux Feb 29th value
            all_leap[all_leap_index + 59] = \
                (original[original_index + 58] + original[original_index + 59]) / 2

            # write the remaining days of the year (Mar 1 through Dec 31)
            # from the original into the all_leap array
            original_year_end_index = original_index + 365
            if len(original) < original_year_end_index:
                # this should be the final year and we're just adding the remained days
                remainder = original[original_index + 59:]
                difference = len(all_leap[all_leap_index + 60:]) - len(remainder)
                if difference > 0:
                    final_days = np.pad(remainder, (0, difference,), mode='constant', constant_values=np.NaN)
                elif difference != 0:
                    raise ValueError("Incompatible shapes")
                else:
                    final_days = remainder
                all_leap[all_leap_index + 60:] = final_days
                continue
            else:
                all_leap[all_leap_index + 60:(all_leap_index + 366)] = \
                    original[original_index + 59:original_year_end_index]

            # increment the "start day of the current year" index for the original
            # so the next iteration jumps ahead a full year
            original_index += 365

        all_leap_index += 366

    return all_leap


# ------------------------------------------------------------------------------
@numba.jit
def transform_to_gregorian(
        original: np.ndarray,
        year_start: int,
) -> np.ndarray:
    """
    Takes an array of daily values represented as full 366 day years (as if each
    year is a leap year with fill/faux values for the Feb. 29th of each non-leap
    year) and converts it to an array of daily values with only actual leap
    years represented as 366 day years.

    For example if provided an input array representing two years,
    we expect/assume that it will contain 732 elements corresponding to two
    years with 366 days. Two possible transformation results are possible:

    1) If the start year or the following year is a leap year then
    the resulting/transformed array will contain 731 elements (366 for
    the leap year plus 365 for the non-leap year), with the element
    that corresponded to Feb. 29th in the non-leap year removed.

    2) If both years represented are non-leap years, as determined by
    the starting year argument, then the resulting/transformed array will
    contain 730 elements (365 days for both non-leap years), with the
    elements that corresponded to Feb. 29th removed.

    :param original: 1-D array of daily values, total size should be
        a multiple of 366
    :param year_start: the year corresponding to the initial year (first 366
        values) of the input array, used to determine whether or not each 366
        increment of daily values represents an actual leap year
    """
    # original time series is assumed to be a one-dimensional array of floats
    # corresponding to a number of full years, with each year containing
    # 366 days, as if each year is a leap year

    # validate the arguments
    if len(original.shape) > 1:
        message = "Invalid input array: only 1-D arrays are supported"
        _logger.error(message)
        raise ValueError(message)
    if original.size % 366 != 0:
        message = "Invalid input array: only 1-D arrays containing " + \
                  "multiples of 366 days are supported"
        _logger.error(message)
        raise ValueError(message)

    # find the total number of actual days between the start and end year
    total_years = int(original.size / 366)
    year_end = year_start + total_years - 1
    days_actual = (datetime(year_end, 12, 31) - datetime(year_start, 1, 1)).days + 1

    # allocate the new array we'll write daily values into,
    # including a faux Feb 29 for non-leap years
    gregorian = np.full((days_actual,), np.NaN)

    # index of the first day of the year within the original and gregorian arrays
    original_index = 0
    gregorian_index = 0

    # loop over each year
    for year in range(year_start, year_start + total_years):

        if calendar.isleap(year):

            # write the next 366 days from the original
            # time series into the gregorian array
            gregorian[gregorian_index:(gregorian_index + 366)] = \
                original[original_index:(original_index + 366)]

            # increment the "start day of the current year" index for the original
            # so the next iteration jumps ahead a full year
            gregorian_index += 366

        else:

            # write the first 59 days (Jan 1 through Feb 28) from the original
            # time series into the gregorian array
            gregorian[gregorian_index:(gregorian_index + 59)] = \
                original[original_index:(original_index + 59)]

            # write the remaining days of the year (Mar 1 through Dec 31)
            # from the original into the gregorian array
            gregorian[(gregorian_index + 59):(gregorian_index + 365)] = \
                original[(original_index + 60):(original_index + 366)]

            # increment the "start day of the current year" index for
            # the original so the next iteration jumps ahead a full year
            gregorian_index += 365

        original_index += 366

    return gregorian


# ------------------------------------------------------------------------------
def count_zeros_and_non_missings(
        values: np.ndarray,
) -> (int, int):
    """
    Given an input array of values return a count of the zeros
    and non-missing values. Missing values assumed to be numpy.NaNs.

    :param values: array like object (numpy array, most likely)
    :return: two int scalars: 1) the count of zeros, and
        2) the count of non-missing values
    """

    # make sure we have a numpy array
    values = np.array(values)

    # count the number of zeros and non-missing (non-NaN) values
    zeros = values.size - np.count_nonzero(values)
    non_missings = np.count_nonzero(~np.isnan(values))

    return zeros, non_missings


_logger = get_logger(__name__, logging.DEBUG)

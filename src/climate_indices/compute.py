"""
Common classes and functions used to compute the various climate indices.
"""

import logging
from enum import Enum

import numpy as np
import scipy.stats
import scipy.version
from packaging.version import Version

from climate_indices import lmoments, utils

# declare the function names that should be included in the public API for this module
__all__ = [
    "Periodicity",
    "scale_values",
    "sum_to_scale",
    "transform_fitted_gamma",
    "transform_fitted_pearson",
    "DistributionFittingError",
    "InsufficientDataError",
    "PearsonFittingError",
    "DistributionFallbackStrategy",
]

# depending on the version of scipy we may need to use a workaround due to a bug in some versions of scipy
_do_pearson3_workaround = Version(scipy.version.version) < Version("1.6.0")

# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.WARN)

# Configuration constants for distribution fitting and validation
# Minimum number of non-zero values required for Pearson Type III L-moments computation
MIN_NON_ZERO_VALUES_FOR_PEARSON = 4

# Maximum failure rate threshold before issuing high failure rate warnings
# Values above this percentage indicate systemic issues with the dataset
HIGH_FAILURE_RATE_THRESHOLD = 0.8  # 80%


# Custom exception classes for distribution fitting
class DistributionFittingError(Exception):
    """Base exception for distribution fitting failures."""

    pass


class InsufficientDataError(DistributionFittingError):
    """Raised when there is insufficient data for distribution fitting."""

    def __init__(self, message, non_zero_count=None, required_count=None):
        super().__init__(message)
        self.non_zero_count = non_zero_count
        self.required_count = required_count


class PearsonFittingError(DistributionFittingError):
    """Raised when Pearson Type III distribution fitting fails."""

    def __init__(self, message, underlying_error=None):
        super().__init__(message)
        self.underlying_error = underlying_error


class DistributionFallbackStrategy:
    """Strategy class for managing Pearson→Gamma distribution fallback logic."""

    def __init__(self, max_nan_percentage=0.5, high_failure_threshold=0.8):
        """
        Initialize the fallback strategy.

        :param max_nan_percentage: Maximum percentage of NaN values before triggering fallback
        :param high_failure_threshold: Failure rate threshold for issuing warnings
        """
        self.max_nan_percentage = max_nan_percentage
        self.high_failure_threshold = high_failure_threshold
        self._logger = utils.get_logger(self.__class__.__name__, logging.WARN)

    def should_fallback_from_excessive_nans(self, values: np.ndarray) -> bool:
        """Check if fallback is needed due to excessive NaN values."""
        if values.size == 0:
            return True
        nan_percentage = np.count_nonzero(np.isnan(values)) / values.size
        return nan_percentage > self.max_nan_percentage

    def should_warn_high_failure_rate(self, failure_count: int, total_count: int) -> bool:
        """Check if high failure rate warning should be issued."""
        if total_count == 0:
            return False
        failure_rate = failure_count / total_count
        return failure_rate > self.high_failure_threshold

    def log_fallback_warning(self, reason: str, context: str = ""):
        """Log a fallback warning with consistent formatting."""
        message = f"Pearson Type III distribution fitting failed ({reason}). "
        message += "Falling back to Gamma distribution for robust computation."
        if context:
            message += f" Context: {context}"
        self._logger.warning(message)

    def log_high_failure_rate(self, failure_count: int, total_count: int, context: str = ""):
        """Log high failure rate warning."""
        failure_rate = failure_count / total_count if total_count > 0 else 0
        message = (
            f"High failure rate for Pearson Type III distribution fitting: {failure_count}/{total_count} "
            f"time steps failed ({failure_rate:.1%} failure rate). This typically occurs with extensive zero "
            f"precipitation patterns that are better handled by Gamma distribution. "
            f"Results may contain many default parameter values."
        )
        if context:
            message += f" Context: {context}"
        self._logger.warning(message)


# Global fallback strategy instance
_default_fallback_strategy = DistributionFallbackStrategy()


class Periodicity(Enum):
    """
    Enumeration type for specifying dataset periodicity.

    'monthly' indicates an array of monthly values, assumed to span full years,
    i.e. the first value corresponds to January of the initial year and any
    missing final months of the final year filled with NaN values,
    with size == # of years * 12

    'daily' indicates an array of full years of daily values with 366 days per year,
    as if each year were a leap year and any missing final months of the final
    year filled with NaN values, with array size == (# years * 366)
    """

    monthly = 12
    daily = 366

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Periodicity[s]
        except KeyError:
            raise ValueError(f"No periodicity enumeration corresponding to {s}")

    def unit(self):
        if self.name == "monthly":
            unit = "month"
        elif self.name == "daily":
            unit = "day"
        else:
            raise ValueError(f"No periodicity unit corresponding to {self.name}")

        return unit


def _validate_array(
    values: np.ndarray,
    periodicity: Periodicity,
) -> np.ndarray:
    """
    Basic data cleaning and validation.

    :param values: array of values to be used as input
    :param periodicity: specifies whether data is monthly or daily
    :return: data array corresponding to the input array converted to
        the correct shape for the specified periodicity
    """

    # validate (and possibly reshape) the input array
    if len(values.shape) == 1:
        if periodicity is None:
            message = "1-D input array requires a corresponding periodicity argument, none provided"
            _logger.error(message)
            raise ValueError(message)

        elif periodicity is Periodicity.monthly:
            # we've been passed a 1-D array with shape (months),
            # reshape it to 2-D with shape (years, 12)
            values = utils.reshape_to_2d(values, 12)

        elif periodicity is Periodicity.daily:
            # we've been passed a 1-D array with shape (days),
            # reshape it to 2-D with shape (years, 366)
            values = utils.reshape_to_2d(values, 366)

        else:
            message = f"Unsupported periodicity argument: '{periodicity}'"
            _logger.error(message)
            raise ValueError(message)

    elif (len(values.shape) != 2) or (values.shape[1] not in (12, 366)):
        # ((values.shape[1] != 12) and (values.shape[1] != 366)):

        # neither a 1-D nor a 2-D array with valid shape was passed in
        message = f"Invalid input array with shape: {values.shape}"
        _logger.error(message)
        raise ValueError(message)

    return values


def sum_to_scale(
    values: np.ndarray,
    scale: int,
) -> np.ndarray:
    """
    Compute a sliding sums array using 1-D convolution. The initial
    (scale - 1) elements of the result array will be padded with np.nan values.
    Missing values are not ignored, i.e. if a np.nan
    (missing) value is part of the group of values to be summed then the sum
    will be np.nan

    For example if the first array is [3, 4, 6, 2, 1, 3, 5, 8, 5] and
    the number of values to sum is 3 then the resulting array
    will be [np.nan, np.nan, 13, 12, 9, 6, 9, 16, 18].

    More generally:

    Y = f(X, n)

    Y[i] == np.nan, where i < n
    Y[i] == sum(X[i - n + 1:i + 1]), where i >= n - 1 and X[i - n + 1:i + 1]
        contains no NaN values
    Y[i] == np.nan, where i >= n - 1 and X[i - n + 1:i + 1] contains
        one or more NaN values

    :param values: the array of values over which we'll compute sliding sums
    :param scale: the number of values for which each sliding summation will
        encompass, for example if this value is 3 then the first two elements of
        the output array will contain the pad value and the third element of the
        output array will contain the sum of the first three elements, and so on
    :return: an array of sliding sums, equal in length to the input values
        array, left padded with NaN values
    """

    # don't bother if the number of values to sum is 1
    if scale == 1:
        return values

    # get the valid sliding summations with 1D convolution
    sliding_sums = np.convolve(values, np.ones(scale), mode="valid")

    # pad the first (n - 1) elements of the array with NaN values
    return np.hstack(([np.nan] * (scale - 1), sliding_sums))

    # BELOW FOR dask/xarray DataArray integration
    # # pad the values array with (scale - 1) NaNs
    # values = pad(values, pad_width=(scale - 1, 0), mode='constant', constant_values=np.nan)
    #
    # start = 1
    # end = -(scale - 2)
    # return convolve(values, np.ones(scale), mode='reflect', cval=0.0, origin=0)[start: end]


def _log_and_raise_shape_error(shape: tuple[int]):
    message = f"Invalid shape of input data array: {shape}"
    _logger.error(message)
    raise ValueError(message)


def _probability_of_zero(
    values: np.ndarray,
) -> np.ndarray:
    """
    This function computes the probability of zero and Pearson Type III
    distribution parameters corresponding to an array of values.

    :param values: 2-D array of values, with each row representing a year
        containing either 12 values corresponding to the calendar months of
        that year, or 366 values corresponding to the days of the year
        (with Feb. 29th being an average of the Feb. 28th and Mar. 1st values for
        non-leap years) and assuming that the first value of the array is
        January of the initial year for an input array of monthly values or
        Jan. 1st of initial year for an input array daily values
    :return: a 1-D array of probability of zero values, with shape (12,) for
        monthly or (366,) for daily
    """
    # validate that the values array has shape: (years, 12) for monthly or (years, 366) for daily
    if len(values.shape) != 2:
        _log_and_raise_shape_error(shape=values.shape)

    else:
        # determine the number of time steps per year
        # (we expect 12 for monthly, 366 for daiy)
        time_steps_per_year = values.shape[1]
        if time_steps_per_year not in (12, 366):
            _log_and_raise_shape_error(shape=values.shape)

    # the values we'll compute and return
    probabilities_of_zero = np.zeros((time_steps_per_year,))

    # compute the probability of zero for each calendar time step
    # TODO vectorize the below loop? create a @numba.vectorize() ufunc
    #  for application over the second axis
    for time_step_index in range(time_steps_per_year):
        # get the values for the current calendar time step
        time_step_values = values[:, time_step_index]

        # count the number of zeros and valid (non-missing/non-NaN) values
        number_of_zeros, number_of_non_missing = utils.count_zeros_and_non_missings(time_step_values)

        # calculate the probability of zero for the calendar time step
        if (number_of_zeros > 0) and (number_of_non_missing > 0):
            probabilities_of_zero[time_step_index] = number_of_zeros / number_of_non_missing

        else:
            # fill with NaN
            probabilities_of_zero[time_step_index] = np.nan

    return probabilities_of_zero


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def reshape_values(values, periodicity):
    if periodicity is Periodicity.monthly:
        return utils.reshape_to_2d(values, 12)
    elif periodicity is Periodicity.daily:
        return utils.reshape_to_2d(values, 366)
    else:
        raise ValueError(f"Invalid periodicity argument: {periodicity}")


def validate_values_shape(values):
    if len(values.shape) != 2 or values.shape[1] not in (12, 366):
        _log_and_raise_shape_error(shape=values.shape)
    return values.shape[1]


def adjust_calibration_years(data_start_year, data_end_year, calibration_start_year, calibration_end_year):
    if (calibration_start_year < data_start_year) or (calibration_end_year > data_end_year):
        return data_start_year, data_end_year
    return calibration_start_year, calibration_end_year


def calculate_time_step_params(time_step_values):
    """
    Calculate Pearson Type III parameters for a time step's values.

    :param time_step_values: Array of values for a specific time step (e.g., all January values)
    :return: Tuple of (probability_of_zero, loc, scale, skew)
    :raises InsufficientDataError: When there are too few non-zero values
    :raises PearsonFittingError: When L-moments computation fails
    """
    number_of_zeros, number_of_non_missing = utils.count_zeros_and_non_missings(time_step_values)
    non_zero_count = number_of_non_missing - number_of_zeros

    if non_zero_count < MIN_NON_ZERO_VALUES_FOR_PEARSON:
        message = (
            f"Insufficient non-zero values for Pearson fitting: "
            f"{non_zero_count} values (minimum {MIN_NON_ZERO_VALUES_FOR_PEARSON} required). "
            f"Consider using Gamma distribution for areas with extensive zero precipitation."
        )
        raise InsufficientDataError(
            message=message, non_zero_count=non_zero_count, required_count=MIN_NON_ZERO_VALUES_FOR_PEARSON
        )

    probability_of_zero = number_of_zeros / number_of_non_missing if number_of_zeros > 0 else 0.0

    # At this point we know non_zero_count >= MIN_NON_ZERO_VALUES_FOR_PEARSON
    try:
        params = lmoments.fit(time_step_values)
        return probability_of_zero, params["loc"], params["scale"], params["skew"]
    except ValueError as e:
        message = f"L-moments fitting failed: {e}. Consider using Gamma distribution for this dataset."
        raise PearsonFittingError(message, underlying_error=e)


def pearson_parameters(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function computes the probability of zero and Pearson Type III
    distribution parameters corresponding to an array of values.

    :param values: 2-D array of values, with each row representing a year
        containing either 12 values corresponding to the calendar months of
        that year, or 366 values corresponding to the days of the year
        (with Feb. 29th being an average of the Feb. 28th and Mar. 1st values for
        non-leap years) and assuming that the first value of the array is
        January of the initial year for an input array of monthly values or
        Jan. 1st of initial year for an input array daily values
    :param data_start_year:
    :param calibration_start_year:
    :param calibration_end_year:
    :param periodicity: monthly or daily
    :return: four 1-D array of fitting values for the Pearson Type III
        distribution, with shape (12,) for monthly or (366,) for daily

        returned array 1: probability of zero
        returned array 2: first Pearson Type III distribution parameter (loc)
        returned array 3 :second Pearson Type III distribution parameter (scale)
        returned array 4: third Pearson Type III distribution parameter (skew)
    """
    values = reshape_values(values, periodicity)
    time_steps_per_year = validate_values_shape(values)
    data_end_year = data_start_year + values.shape[0]
    calibration_start_year, calibration_end_year = adjust_calibration_years(
        data_start_year, data_end_year, calibration_start_year, calibration_end_year
    )
    calibration_begin_index = calibration_start_year - data_start_year
    calibration_end_index = (calibration_end_year - data_start_year) + 1
    calibration_values = values[calibration_begin_index:calibration_end_index, :]
    probabilities_of_zero = np.zeros((time_steps_per_year,))
    locs = np.zeros((time_steps_per_year,))
    scales = np.zeros((time_steps_per_year,))
    skews = np.zeros((time_steps_per_year,))

    failed_fitting_count = 0

    for time_step_index in range(time_steps_per_year):
        time_step_values = calibration_values[:, time_step_index]
        try:
            prob, loc, scale, skew = calculate_time_step_params(time_step_values)
            probabilities_of_zero[time_step_index] = prob
            locs[time_step_index] = loc
            scales[time_step_index] = scale
            skews[time_step_index] = skew
        except DistributionFittingError:
            # Handle fitting failures by using default values
            failed_fitting_count += 1
            probabilities_of_zero[time_step_index] = 0.0
            locs[time_step_index] = 0.0
            scales[time_step_index] = 0.0
            skews[time_step_index] = 0.0

    # Check if we should warn about high failure rate using the fallback strategy
    if _default_fallback_strategy.should_warn_high_failure_rate(failed_fitting_count, time_steps_per_year):
        _default_fallback_strategy.log_high_failure_rate(
            failure_count=failed_fitting_count,
            total_count=time_steps_per_year,
            context="pearson_parameters computation",
        )

    return probabilities_of_zero, locs, scales, skews


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# def pearson_parameters_previous(
#     values: np.ndarray,
#     data_start_year: int,
#     calibration_start_year: int,
#     calibration_end_year: int,
#     periodicity: Periodicity,
# ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
#     """
#     This function computes the probability of zero and Pearson Type III
#     distribution parameters corresponding to an array of values.
#
#     :param values: 2-D array of values, with each row representing a year
#         containing either 12 values corresponding to the calendar months of
#         that year, or 366 values corresponding to the days of the year
#         (with Feb. 29th being an average of the Feb. 28th and Mar. 1st values for
#         non-leap years) and assuming that the first value of the array is
#         January of the initial year for an input array of monthly values or
#         Jan. 1st of initial year for an input array daily values
#     :param data_start_year:
#     :param calibration_start_year:
#     :param calibration_end_year:
#     :param periodicity: monthly or daily
#     :return: four 1-D array of fitting values for the Pearson Type III
#         distribution, with shape (12,) for monthly or (366,) for daily
#
#         returned array 1: probability of zero
#         returned array 2: first Pearson Type III distribution parameter (loc)
#         returned array 3 :second Pearson Type III distribution parameter (scale)
#         returned array 4: third Pearson Type III distribution parameter (skew)
#     """
#
#     # reshape precipitation values to (years, 12) for monthly,
#     # or to (years, 366) for daily
#     if periodicity is Periodicity.monthly:
#
#         values = utils.reshape_to_2d(values, 12)
#
#     elif periodicity is Periodicity.daily:
#
#         values = utils.reshape_to_2d(values, 366)
#
#     else:
#
#         raise ValueError("Invalid periodicity argument: %s" % periodicity)
#
#     # validate that the values array has shape: (years, 12) for monthly or (years, 366) for daily
#     if len(values.shape) != 2:
#         _log_and_raise_shape_error(shape=values.shape)
#
#     else:
#
#         time_steps_per_year = values.shape[1]
#         if time_steps_per_year not in (12, 366):
#             _log_and_raise_shape_error(shape=values.shape)
#
#     # determine the end year of the values array
#     data_end_year = data_start_year + values.shape[0]
#
#     # make sure that we have data within the full calibration period,
#     # otherwise use the full period of record
#     if (calibration_start_year < data_start_year) or \
#             (calibration_end_year > data_end_year):
#         calibration_start_year = data_start_year
#         calibration_end_year = data_end_year
#
#     # get the year axis indices corresponding to
#     # the calibration start and end years
#     calibration_begin_index = calibration_start_year - data_start_year
#     calibration_end_index = (calibration_end_year - data_start_year) + 1
#
#     # get the values for the current calendar time step
#     # that fall within the calibration years period
#     calibration_values = values[calibration_begin_index:calibration_end_index, :]
#
#     # the values we'll compute and return
#     probabilities_of_zero = np.zeros((time_steps_per_year,))
#     locs = np.zeros((time_steps_per_year,))
#     scales = np.zeros((time_steps_per_year,))
#     skews = np.zeros((time_steps_per_year,))
#
#     # compute the probability of zero and Pearson
#     # parameters for each calendar time step
#     # TODO vectorize the below loop? create a @numba.vectorize() ufunc
#     #  for application over the second axis
#     for time_step_index in range(time_steps_per_year):
#
#         # get the values for the current calendar time step
#         time_step_values = calibration_values[:, time_step_index]
#
#         # count the number of zeros and valid (non-missing/non-NaN) values
#         number_of_zeros, number_of_non_missing = \
#             utils.count_zeros_and_non_missings(time_step_values)
#
#         # make sure we have at least four values that are both non-missing (i.e. non-NaN)
#         # and non-zero, otherwise use the entire period of record
#         if (number_of_non_missing - number_of_zeros) < 4:
#
#             # we can't proceed, bail out using zeros
#             continue
#
#         # calculate the probability of zero for the calendar time step
#         probability_of_zero = 0.0
#         if number_of_zeros > 0:
#
#             probability_of_zero = number_of_zeros / number_of_non_missing
#
#         # get the estimated L-moments, if we have
#         # more than three non-missing/non-zero values
#         if (number_of_non_missing - number_of_zeros) > 3:
#
#             # get the Pearson Type III parameters for this time
#             # step's values within the calibration period
#             params = lmoments.fit(time_step_values)
#             probabilities_of_zero[time_step_index] = probability_of_zero
#             locs[time_step_index] = params["loc"]
#             scales[time_step_index] = params["scale"]
#             skews[time_step_index] = params["skew"]
#
#     return probabilities_of_zero, locs, scales, skews


def _minimum_possible(
    skew: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Compute the minimum possible value that can be fitted to a distribution
    described by a set of skew, loc, and scale parameters.

    :param skew:
    :param loc:
    :param scale:
    :return:
    """

    alpha = 4.0 / (skew * skew)

    # calculate the lowest possible value that will
    # fit the distribution (i.e. Z = 0)
    return loc - ((alpha * scale * skew) / 2.0)


def _pearson_fit(
    values: np.ndarray,
    probabilities_of_zero: np.ndarray,
    skew: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Perform fitting of an array of values to a Pearson Type III distribution
    as described by the Pearson Type III parameters and probability of zero arguments.

    :param values: an array of values to fit to the Pearson Type III
        distribution described by the skew, loc, and scale
    :param probabilities_of_zero: probability that the value is zero
    :param skew: first Pearson Type III parameter, the skew of the distribution
    :param loc: second Pearson Type III parameter, the loc of the distribution
    :param scale: third Pearson Type III parameter, the scale of the distribution
    """

    # only fit to the distribution if the values array is valid/not missing
    if not np.all(np.isnan(values)):
        # This is a misnomer of sorts. For positively skewed Pearson Type III
        # distributions, there is a hard lower limit. For negatively skewed
        # distributions, the limit is on the upper end.
        minimums_possible = _minimum_possible(skew, loc, scale)
        minimums_mask = (values <= minimums_possible) & (skew >= 0)
        maximums_mask = (values >= minimums_possible) & (skew < 0)

        # Not sure what the logic is here given that the inputs aren't
        # standardized values and Pearson III distributions could handle
        # these sorts of values just fine given the proper parameters.
        zero_mask = np.logical_and((values < 0.0005), (probabilities_of_zero > 0.0))
        trace_mask = np.logical_and((values < 0.0005), (probabilities_of_zero <= 0.0))

        # get the Pearson Type III cumulative density function value
        values = scipy.stats.pearson3.cdf(values, skew, loc, scale)

        # turn zero, trace, or minimum values either into either zero
        # or minimum value based on the probability of zero
        values[zero_mask] = 0.0
        values[trace_mask] = 0.0005

        # The original values were found to be outside the
        # range of the fitted distribution, so we will set
        # the probabilities to something just within the range.
        values[minimums_mask] = 0.0005
        values[maximums_mask] = 0.9995

        if not np.all(np.isnan(values)):
            # calculate the probability value, clipped between 0 and 1
            probabilities = np.clip(
                (probabilities_of_zero + ((1.0 - probabilities_of_zero) * values)),
                0.0,
                1.0,
            )

            # the values we'll return are the values at which the probabilities
            # of a normal distribution are less than or equal to the computed
            # probabilities, as determined by the normal distribution's
            # quantile (or inverse cumulative distribution) function
            fitted_values = scipy.stats.norm.ppf(probabilities)

        else:
            fitted_values = values

    else:
        fitted_values = values

    return fitted_values


def transform_fitted_pearson(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    probabilities_of_zero: np.ndarray = None,
    locs: np.ndarray = None,
    scales: np.ndarray = None,
    skews: np.ndarray = None,
) -> np.ndarray:
    """
    Fit values to a Pearson Type III distribution and transform the values
    to corresponding normalized sigmas.

    :param values: 2-D array of values, with each row representing a year containing
                   twelve columns representing the respective calendar months,
                   or 366 columns representing days as if all years were leap years
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period
    :param calibration_end_year: the final year to use for the calibration period
    :param periodicity: the periodicity of the time series represented by the input
                        data, valid/supported values are 'monthly' and 'daily'
                        'monthly' indicates an array of monthly values, assumed
                        to span full years, i.e. the first value corresponds
                        to January of the initial year and any missing final
                        months of the final year filled with NaN values,
                        with size == # of years * 12
                        'daily' indicates an array of full years of daily values
                        with 366 days per year, as if each year were a leap year
                        and any missing final months of the final year filled
                        with NaN values, with array size == (# years * 366)
    :param probabilities_of_zero: pre-computed probabilities of zero for each
        month or day of the year
    :param locs: pre-computed loc values for each month or day of the year
    :param scales: pre-computed scale values for each month or day of the year
    :param skews: pre-computed skew values for each month or day of the year
    :return: 2-D array of transformed/fitted values, corresponding in size
             and shape of the input array
    :rtype: numpy.ndarray of floats
    """

    # sanity check for the fitting parameters arguments
    pearson_param_args = [probabilities_of_zero, locs, scales, skews]
    if any(param_arg is None for param_arg in pearson_param_args):
        if pearson_param_args.count(None) < len(pearson_param_args):
            raise ValueError(
                "At least one but not all of the Pearson Type III fitting "
                "parameters are specified -- either none or all of "
                "these must be specified"
            )

    # if we're passed all missing values then we can't compute anything,
    # and we'll return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # validate (and possibly reshape) the input array
    values = _validate_array(values, periodicity)

    # compute the Pearson Type III fitting values if none were provided
    if any(param_arg is None for param_arg in pearson_param_args):
        # determine the end year of the values array
        data_end_year = data_start_year + values.shape[0]

        # make sure that we have data within the full calibration period,
        # otherwise use the full period of record
        if (calibration_start_year < data_start_year) or (calibration_end_year > data_end_year):
            calibration_start_year = data_start_year
            calibration_end_year = data_end_year

        # compute the values we'll use to fit to the Pearson Type III distribution
        probabilities_of_zero, locs, scales, skews = pearson_parameters(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
        )

    # fit each value to the Pearson Type III distribution
    values = _pearson_fit(values, probabilities_of_zero, skews, locs, scales)

    return values


def gamma_parameters(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
) -> (np.ndarray, np.ndarray):
    """
    Computes the gamma distribution parameters alpha and beta.

    :param values: 2-D array of values, with each row typically representing a year
                   containing twelve columns representing the respective calendar
                   months, or 366 days per column as if all years were leap years
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period
    :param calibration_end_year: the final year to use for the calibration period
    :param periodicity: the type of time series represented by the input data,
        valid values are 'monthly' or 'daily'
        'monthly': array of monthly values, assumed to span full years,
        i.e. the first value corresponds to January of the initial year and any
        missing final months of the final year filled with NaN values, with
        size == # of years * 12
        'daily': array of full years of daily values with 366 days per year,
        as if each year were a leap year and any missing final months of the final
        year filled with NaN values, with array size == (# years * 366)
    :return: two 2-D arrays of gamma fitting parameter values, corresponding in size
        and shape of the input array
    :rtype: tuple of two 2-D numpy.ndarrays of floats, alphas and betas
    """

    # if we're passed all missing values then we can't compute anything,
    # then we return an array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        if periodicity is Periodicity.monthly:
            shape = (12,)
        elif periodicity is Periodicity.daily:
            shape = (366,)
        else:
            raise ValueError(f"Unsupported periodicity: {periodicity}")
        alphas = np.full(shape=shape, fill_value=np.nan)
        betas = np.full(shape=shape, fill_value=np.nan)
        return alphas, betas

    # validate (and possibly reshape) the input array
    values = _validate_array(values, periodicity)

    # replace zeros with NaNs
    values[values == 0] = np.nan

    # determine the end year of the values array
    data_end_year = data_start_year + values.shape[0]

    # make sure that we have data within the full calibration period,
    # otherwise use the full period of record
    if (calibration_start_year < data_start_year) or (calibration_end_year > data_end_year):
        calibration_start_year = data_start_year
        calibration_end_year = data_end_year

    # get the year axis indices corresponding to
    # the calibration start and end years
    calibration_begin_index = calibration_start_year - data_start_year
    calibration_end_index = (calibration_end_year - data_start_year) + 1

    # get the values for the current calendar time step
    # that fall within the calibration years period
    calibration_values = values[calibration_begin_index:calibration_end_index, :]

    # compute the gamma distribution's shape and scale parameters, alpha and beta
    # using method of moments estimation
    means = np.nanmean(calibration_values, axis=0)
    log_means = np.log(means)
    logs = np.log(calibration_values)
    mean_logs = np.nanmean(logs, axis=0)
    a = log_means - mean_logs
    alphas = (1 + np.sqrt(1 + 4 * a / 3)) / (4 * a)
    betas = means / alphas

    return alphas, betas


def scale_values(
    values: np.ndarray,
    scale: int,
    periodicity: Periodicity,
):
    # we expect to operate upon a 1-D array, so if we've been passed a 2-D array
    # then we flatten it, otherwise raise an error
    shape = values.shape
    if len(shape) == 2:
        values = values.flatten()
    elif len(shape) != 1:
        # only 1-D and 2-D arrays are supported
        _log_and_raise_shape_error(shape=shape)

    # if we're passed all missing values then we can't compute
    # anything, so we return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # clip any negative values to zero
    if np.amin(values) < 0.0:
        _logger.warn("Input contains negative values -- all negatives clipped to zero")
        values = np.clip(values, a_min=0.0, a_max=None)

    # get a sliding sums array, with each time step's value scaled
    # by the specified number of time steps
    scaled_values = sum_to_scale(values, scale)

    # reshape precipitation values to (years, 12) for monthly,
    # or to (years, 366) for daily
    if periodicity is Periodicity.monthly:
        scaled_values = utils.reshape_to_2d(scaled_values, 12)

    elif periodicity is Periodicity.daily:
        scaled_values = utils.reshape_to_2d(scaled_values, 366)

    else:
        raise ValueError("Invalid periodicity argument: %s" % periodicity)

    return scaled_values


def transform_fitted_gamma(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    alphas: np.ndarray = None,
    betas: np.ndarray = None,
) -> np.ndarray:
    """
    Fit values to a gamma distribution and transform the values to corresponding
    normalized sigmas.

    :param values: 2-D array of values, with each row typically representing a year
                   containing twelve columns representing the respective calendar
                   months, or 366 days per column as if all years were leap years
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period
    :param calibration_end_year: the final year to use for the calibration period
    :param periodicity: the type of time series represented by the input data,
        valid values are 'monthly' or 'daily'
        'monthly': array of monthly values, assumed to span full years,
        i.e. the first value corresponds to January of the initial year and any
        missing final months of the final year filled with NaN values, with
        size == # of years * 12
        'daily': array of full years of daily values with 366 days per year,
        as if each year were a leap year and any missing final months of the final
        year filled with NaN values, with array size == (# years * 366)
    :param alphas: pre-computed gamma fitting parameters
    :param betas: pre-computed gamma fitting parameters
    :return: 2-D array of transformed/fitted values, corresponding in size
        and shape of the input array
    :rtype: numpy.ndarray of floats
    """

    # if we're passed all missing values then we can't compute anything,
    # then we return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # validate (and possibly reshape) the input array
    values = _validate_array(values, periodicity)

    # find the percentage of zero values for each time step
    zeros = (values == 0).sum(axis=0)
    probabilities_of_zero = zeros / values.shape[0]

    # replace zeros with NaNs
    values[values == 0] = np.nan

    # compute fitting parameters if none were provided
    if (alphas is None) or (betas is None):
        alphas, betas = gamma_parameters(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
        )

    # find the gamma probability values using the gamma CDF
    gamma_probabilities = scipy.stats.gamma.cdf(values, a=alphas, scale=betas)

    # TODO explain this better
    # (normalize including the probability of zero, putting into the range [0..1]?)
    probabilities = probabilities_of_zero + ((1 - probabilities_of_zero) * gamma_probabilities)

    # the values we'll return are the values at which the probabilities of
    # a normal distribution are less than or equal to the computed probabilities,
    # as determined by the normal distribution's quantile (or inverse
    # cumulative distribution) function
    return scipy.stats.norm.ppf(probabilities)

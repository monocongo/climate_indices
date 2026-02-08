"""
Common classes and functions used to compute the various climate indices.
"""

import warnings
from enum import Enum

import numpy as np
import scipy.stats
import scipy.version
from packaging.version import Version

from climate_indices import lmoments, utils
from climate_indices.exceptions import (
    DistributionFittingError,
    GoodnessOfFitWarning,
    InsufficientDataError,
    MissingDataWarning,
    PearsonFittingError,
    ShortCalibrationWarning,
)
from climate_indices.logging_config import get_logger

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

# module-level structlog logger
_logger = get_logger(__name__)

# Configuration constants for distribution fitting and validation
# Minimum number of non-zero values required for Pearson Type III L-moments computation
MIN_NON_ZERO_VALUES_FOR_PEARSON = 4

# Maximum failure rate threshold before issuing high failure rate warnings
# Values above this percentage indicate systemic issues with the dataset
HIGH_FAILURE_RATE_THRESHOLD = 0.8  # 80%

# Data quality warning thresholds
# Maximum acceptable proportion of missing data in calibration period
MISSING_DATA_THRESHOLD = 0.20  # 20%

# Minimum recommended calibration period length in years
MIN_CALIBRATION_YEARS = 30

# Kolmogorov-Smirnov test p-value threshold for goodness-of-fit warnings
GOODNESS_OF_FIT_P_VALUE_THRESHOLD = 0.05


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
        self._logger = get_logger(self.__class__.__name__)

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
        except KeyError as err:
            raise ValueError(f"No periodicity enumeration corresponding to {s}") from err

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
            _logger.error(
                "validation_error",
                operation="validate_array",
                reason="missing_periodicity",
                shape=str(values.shape),
            )
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
            _logger.error(
                "validation_error",
                operation="validate_array",
                reason="unsupported_periodicity",
                periodicity=str(periodicity),
            )
            raise ValueError(message)

    elif (len(values.shape) != 2) or (values.shape[1] not in (12, 366)):
        # ((values.shape[1] != 12) and (values.shape[1] != 366)):

        # neither a 1-D nor a 2-D array with valid shape was passed in
        message = f"Invalid input array with shape: {values.shape}"
        _logger.error(
            "validation_error",
            operation="validate_array",
            reason="invalid_shape",
            shape=str(values.shape),
        )
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
    _logger.error(
        "validation_error",
        operation="validate_shape",
        reason="invalid_shape",
        shape=str(shape),
    )
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


def _summarize_array(arr: np.ndarray | None, name: str = "array") -> str:
    """Summarize a numpy array for error messages.

    For small arrays (≤12 elements), returns the full array representation.
    For larger arrays, returns a summary with shape, min, max, mean, and nan count.

    Args:
        arr: The array to summarize, or None
        name: Name to use in the summary (e.g., "alphas", "values")

    Returns:
        A string representation suitable for error messages
    """
    if arr is None:
        return f"{name}=None"

    if arr.size <= 12:
        return f"{name}={arr}"

    nan_count = np.sum(np.isnan(arr))
    # use nanmin/nanmax/nanmean to avoid errors when all values are NaN
    min_val = np.nanmin(arr) if not np.all(np.isnan(arr)) else np.nan
    max_val = np.nanmax(arr) if not np.all(np.isnan(arr)) else np.nan
    mean_val = np.nanmean(arr) if not np.all(np.isnan(arr)) else np.nan

    return (
        f"{name}: shape={arr.shape}, "
        f"min={min_val:.4g}, max={max_val:.4g}, mean={mean_val:.4g}, "
        f"nan_count={nan_count}/{arr.size}"
    )


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
        raise PearsonFittingError(message, underlying_error=e) from e


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
    log = _logger.bind(
        operation="pearson_parameters",
        distribution="pearson3",
        periodicity=str(periodicity),
        calibration_period=f"{calibration_start_year}-{calibration_end_year}",
    )
    log.info("distribution_fitting_started")

    values = reshape_values(values, periodicity)
    time_steps_per_year = validate_values_shape(values)
    data_end_year = data_start_year + values.shape[0]
    calibration_start_year, calibration_end_year = adjust_calibration_years(
        data_start_year, data_end_year, calibration_start_year, calibration_end_year
    )
    calibration_begin_index = calibration_start_year - data_start_year
    calibration_end_index = (calibration_end_year - data_start_year) + 1
    calibration_values = values[calibration_begin_index:calibration_end_index, :]

    # check calibration data quality and emit warnings if needed
    _check_calibration_data_quality(calibration_values, calibration_start_year, calibration_end_year)

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

    # check goodness-of-fit and emit warning if poor
    _check_goodness_of_fit_pearson(calibration_values, probabilities_of_zero, locs, scales, skews)

    log.info("distribution_fitting_completed", output_shape=str(probabilities_of_zero.shape))
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
        try:
            values = scipy.stats.pearson3.cdf(values, skew, loc, scale)
        except (ValueError, RuntimeError, FloatingPointError) as e:
            raise DistributionFittingError(
                f"Pearson Type III distribution CDF computation failed: {e}",
                distribution_name="pearson3",
                input_shape=values.shape,
                parameters={
                    "skew": _summarize_array(skew, "skew"),
                    "loc": _summarize_array(loc, "loc"),
                    "scale": _summarize_array(scale, "scale"),
                    "values": _summarize_array(values, "values"),
                },
                suggestion="Try using gamma distribution instead",
                underlying_error=e,
            ) from e

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
            try:
                fitted_values = scipy.stats.norm.ppf(probabilities)
            except (ValueError, RuntimeError, FloatingPointError) as e:
                raise DistributionFittingError(
                    f"Normal distribution inverse CDF (ppf) computation failed during Pearson transformation: {e}",
                    distribution_name="pearson3",
                    input_shape=probabilities.shape,
                    parameters={
                        "probabilities": _summarize_array(probabilities, "probabilities"),
                        "skew": _summarize_array(skew, "skew"),
                        "loc": _summarize_array(loc, "loc"),
                        "scale": _summarize_array(scale, "scale"),
                    },
                    suggestion="Try using gamma distribution instead",
                    underlying_error=e,
                ) from e

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
    log = _logger.bind(
        operation="transform_fitted_pearson",
        distribution="pearson3",
        periodicity=str(periodicity),
        input_shape=str(values.shape),
    )
    log.info("distribution_transform_started")

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

    log.info("distribution_transform_completed", output_shape=str(values.shape))
    return values


def _check_calibration_data_quality(
    calibration_values: np.ndarray,
    calibration_start_year: int,
    calibration_end_year: int,
) -> None:
    """
    Check calibration period data quality and emit warnings if issues are detected.

    Emits warnings for:
    1. Short calibration period (< MIN_CALIBRATION_YEARS)
    2. Excessive missing data (> MISSING_DATA_THRESHOLD)

    :param calibration_values: Calibration data array with shape (years, time_steps)
    :param calibration_start_year: Start year of calibration period
    :param calibration_end_year: End year of calibration period (inclusive)
    """
    # check calibration period length
    actual_years = (calibration_end_year - calibration_start_year) + 1
    if actual_years < MIN_CALIBRATION_YEARS:
        message = (
            f"Calibration period is {actual_years} years, which is shorter than the "
            f"recommended minimum of {MIN_CALIBRATION_YEARS} years. "
            f"Shorter periods may not capture the full range of climate variability."
        )
        warning = ShortCalibrationWarning(
            message,
            actual_years=actual_years,
            required_years=MIN_CALIBRATION_YEARS,
        )
        warnings.warn(warning, stacklevel=3)

    # check for excessive missing data
    total_values = calibration_values.size
    if total_values > 0:
        missing_count = np.count_nonzero(np.isnan(calibration_values))
        missing_ratio = missing_count / total_values
        if missing_ratio > MISSING_DATA_THRESHOLD:
            message = (
                f"Calibration period has {missing_ratio:.1%} missing data, which exceeds the "
                f"recommended threshold of {MISSING_DATA_THRESHOLD:.1%}. High missing data rates "
                f"may reduce the reliability of distribution fitting."
            )
            warning = MissingDataWarning(
                message,
                missing_ratio=missing_ratio,
                threshold=MISSING_DATA_THRESHOLD,
            )
            warnings.warn(warning, stacklevel=3)


def _check_goodness_of_fit_gamma(
    calibration_values: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> None:
    """
    Check goodness-of-fit for gamma distribution and emit aggregated warning if poor.

    Performs Kolmogorov-Smirnov tests for each time step and aggregates
    poor fits into a single warning to avoid flooding users with warnings.

    :param calibration_values: Calibration data with shape (years, time_steps)
    :param alphas: Shape parameters for gamma distribution
    :param betas: Scale parameters for gamma distribution
    """
    time_steps = calibration_values.shape[1]
    poor_fit_steps = []

    for time_step_index in range(time_steps):
        time_step_values = calibration_values[:, time_step_index]
        # remove NaN values for KS test
        valid_values = time_step_values[~np.isnan(time_step_values)]

        if len(valid_values) > 0:
            alpha = alphas[time_step_index]
            beta = betas[time_step_index]

            # skip if parameters are invalid
            if not (np.isfinite(alpha) and np.isfinite(beta) and alpha > 0 and beta > 0):
                continue

            # perform Kolmogorov-Smirnov test
            try:
                ks_statistic, p_value = scipy.stats.kstest(
                    valid_values,
                    lambda x, a=alpha, s=beta: scipy.stats.gamma.cdf(x, a=a, scale=s),
                )
                if p_value < GOODNESS_OF_FIT_P_VALUE_THRESHOLD:
                    poor_fit_steps.append((time_step_index, p_value))
            except Exception:
                # ignore fitting errors during goodness-of-fit check
                continue

    if poor_fit_steps:
        # show up to 5 examples
        examples = poor_fit_steps[:5]
        example_text = ", ".join([f"step {idx} (p={p:.4f})" for idx, p in examples])
        if len(poor_fit_steps) > 5:
            example_text += f", and {len(poor_fit_steps) - 5} more"

        message = (
            f"Gamma distribution shows poor goodness-of-fit for {len(poor_fit_steps)} of "
            f"{time_steps} time steps (p < {GOODNESS_OF_FIT_P_VALUE_THRESHOLD}). "
            f"Examples: {example_text}. Consider using a different distribution or "
            f"investigating data quality issues."
        )
        warning = GoodnessOfFitWarning(
            message,
            distribution_name="gamma",
            threshold=GOODNESS_OF_FIT_P_VALUE_THRESHOLD,
            poor_fit_count=len(poor_fit_steps),
            total_steps=time_steps,
        )
        warnings.warn(warning, stacklevel=3)


def _check_goodness_of_fit_pearson(
    calibration_values: np.ndarray,
    probabilities_of_zero: np.ndarray,
    locs: np.ndarray,
    scales: np.ndarray,
    skews: np.ndarray,
) -> None:
    """
    Check goodness-of-fit for Pearson Type III distribution and emit aggregated warning if poor.

    Performs Kolmogorov-Smirnov tests for each time step and aggregates
    poor fits into a single warning to avoid flooding users with warnings.

    :param calibration_values: Calibration data with shape (years, time_steps)
    :param probabilities_of_zero: Probability of zero for each time step
    :param locs: Location parameters for Pearson Type III distribution
    :param scales: Scale parameters for Pearson Type III distribution
    :param skews: Skewness parameters for Pearson Type III distribution
    """
    time_steps = calibration_values.shape[1]
    poor_fit_steps = []

    for time_step_index in range(time_steps):
        time_step_values = calibration_values[:, time_step_index]
        loc = locs[time_step_index]
        scale = scales[time_step_index]
        skew = skews[time_step_index]

        # skip time steps where fitting failed (all parameters are zero)
        if loc == 0 and scale == 0 and skew == 0:
            continue

        # filter out NaN and zero values for non-zero distribution
        valid_values = time_step_values[~np.isnan(time_step_values) & (time_step_values != 0)]

        if len(valid_values) > 0:
            # skip if parameters are invalid
            if not (np.isfinite(loc) and np.isfinite(scale) and np.isfinite(skew) and scale > 0):
                continue

            # perform Kolmogorov-Smirnov test
            try:
                ks_statistic, p_value = scipy.stats.kstest(
                    valid_values,
                    lambda x, sk=skew, loc_=loc, sc=scale: scipy.stats.pearson3.cdf(x, sk, loc=loc_, scale=sc),
                )
                if p_value < GOODNESS_OF_FIT_P_VALUE_THRESHOLD:
                    poor_fit_steps.append((time_step_index, p_value))
            except Exception:
                # ignore fitting errors during goodness-of-fit check
                continue

    if poor_fit_steps:
        # show up to 5 examples
        examples = poor_fit_steps[:5]
        example_text = ", ".join([f"step {idx} (p={p:.4f})" for idx, p in examples])
        if len(poor_fit_steps) > 5:
            example_text += f", and {len(poor_fit_steps) - 5} more"

        message = (
            f"Pearson Type III distribution shows poor goodness-of-fit for {len(poor_fit_steps)} of "
            f"{time_steps} time steps (p < {GOODNESS_OF_FIT_P_VALUE_THRESHOLD}). "
            f"Examples: {example_text}. Consider using a different distribution or "
            f"investigating data quality issues."
        )
        warning = GoodnessOfFitWarning(
            message,
            distribution_name="pearson3",
            threshold=GOODNESS_OF_FIT_P_VALUE_THRESHOLD,
            poor_fit_count=len(poor_fit_steps),
            total_steps=time_steps,
        )
        warnings.warn(warning, stacklevel=3)


def _replace_zeros_with_nan(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a copy of values with zeros replaced by NaN.

    This helper centralizes the zero-to-NaN conversion logic used when fitting
    gamma distributions, where zeros must be excluded from the fitting process
    but their positions need to be tracked for later probability calculations.

    :param values: Input array potentially containing zeros
    :return: Tuple of (zero_mask, values_copy) where:
        - zero_mask: Boolean array where True indicates original zero positions
        - values_copy: Copy of input with zeros replaced by NaN
    """
    values_copy = values.copy()
    zero_mask = values == 0
    values_copy[zero_mask] = np.nan
    return zero_mask, values_copy


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
    log = _logger.bind(
        operation="gamma_parameters",
        distribution="gamma",
        periodicity=str(periodicity),
        calibration_period=f"{calibration_start_year}-{calibration_end_year}",
    )
    log.info("distribution_fitting_started")

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

    # save reference to original values before zero replacement for data quality checks
    original_values = values

    # replace zeros with NaNs (zeros are excluded from gamma fitting)
    _, values = _replace_zeros_with_nan(values)

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
    original_calibration = original_values[calibration_begin_index:calibration_end_index, :]

    # check calibration data quality and emit warnings if needed
    _check_calibration_data_quality(original_calibration, calibration_start_year, calibration_end_year)

    # compute the gamma distribution's shape and scale parameters, alpha and beta
    # using method of moments estimation
    means = np.nanmean(calibration_values, axis=0)
    log_means = np.log(means)
    logs = np.log(calibration_values)
    mean_logs = np.nanmean(logs, axis=0)
    a = log_means - mean_logs
    alphas = (1 + np.sqrt(1 + 4 * a / 3)) / (4 * a)
    betas = means / alphas

    # check goodness-of-fit and emit warning if poor
    _check_goodness_of_fit_gamma(calibration_values, alphas, betas)

    log.info("distribution_fitting_completed", output_shape=str(alphas.shape))
    return alphas, betas


def scale_values(
    values: np.ndarray,
    scale: int,
    periodicity: Periodicity,
):
    _logger.debug("scaling_started", operation="scale_values", scale=scale, periodicity=str(periodicity))

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
        _logger.warning("negative_values_clipped", operation="scale_values")
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
        raise ValueError(f"Invalid periodicity argument: {periodicity}")

    _logger.debug("scaling_completed", operation="scale_values", output_shape=str(scaled_values.shape))
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
    log = _logger.bind(
        operation="transform_fitted_gamma",
        distribution="gamma",
        periodicity=str(periodicity),
        input_shape=str(values.shape),
    )
    log.info("distribution_transform_started")

    # if we're passed all missing values then we can't compute anything,
    # then we return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # validate (and possibly reshape) the input array
    values = _validate_array(values, periodicity)

    # Replace zeros with NaNs for fitting (zeros are excluded from gamma fitting)
    # and get mask of zero positions for later probability calculations
    zero_mask, values_for_fitting = _replace_zeros_with_nan(values)

    # find the percentage of zero values for each time step
    zeros = zero_mask.sum(axis=0)
    probabilities_of_zero = zeros / values.shape[0]

    # If a time step has all zeros (probability of zero is 1.0), the resulting SPI
    # would be +infinity (extreme wetness) which is incorrect for a dry region.
    # We set probability_of_zero to 0.0 for these time steps, which means:
    #   - gamma_parameters() will return NaN (since all values become NaN after
    #     zero replacement)
    #   - gamma.cdf() will return NaN
    #   - gamma_probabilities[zero_mask] = 0.0 forces these to 0.0
    #   - Final probability = 0.0 + (1.0 * 0.0) = 0.0
    #   - norm.ppf(0.0) = -infinity (extreme drought)
    # This is the correct interpretation: a location with 100% zero precipitation
    # in the historical record is in extreme drought, not extreme wetness.
    probabilities_of_zero[np.isclose(probabilities_of_zero, 1.0)] = 0.0

    # compute fitting parameters if none were provided
    if (alphas is None) or (betas is None):
        alphas, betas = gamma_parameters(
            values_for_fitting,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
        )

    # find the gamma probability values using the gamma CDF
    try:
        gamma_probabilities = scipy.stats.gamma.cdf(values_for_fitting, a=alphas, scale=betas)
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise DistributionFittingError(
            f"Gamma distribution CDF computation failed: {e}",
            distribution_name="gamma",
            input_shape=values_for_fitting.shape,
            parameters={
                "alphas": _summarize_array(alphas, "alphas"),
                "betas": _summarize_array(betas, "betas"),
                "values": _summarize_array(values_for_fitting, "values"),
            },
            suggestion="Try using pearson3 distribution instead",
            underlying_error=e,
        ) from e

    # where the input values were zero the CDF will have returned NaN, but since
    # we're treating zeros as a separate probability mass we should treat the
    # gamma probability for zeros as 0.0
    gamma_probabilities[zero_mask] = 0.0

    # TODO explain this better
    # (normalize including the probability of zero, putting into the range [0..1]?)
    probabilities = probabilities_of_zero + ((1 - probabilities_of_zero) * gamma_probabilities)

    # the values we'll return are the values at which the probabilities of
    # a normal distribution are less than or equal to the computed probabilities,
    # as determined by the normal distribution's quantile (or inverse
    # cumulative distribution) function
    try:
        result = scipy.stats.norm.ppf(probabilities)
        log.info("distribution_transform_completed", output_shape=str(result.shape))
        return result
    except (ValueError, RuntimeError, FloatingPointError) as e:
        raise DistributionFittingError(
            f"Normal distribution inverse CDF (ppf) computation failed during gamma transformation: {e}",
            distribution_name="gamma",
            input_shape=probabilities.shape,
            parameters={
                "probabilities": _summarize_array(probabilities, "probabilities"),
                "alphas": _summarize_array(alphas, "alphas"),
                "betas": _summarize_array(betas, "betas"),
            },
            suggestion="Try using pearson3 distribution instead",
            underlying_error=e,
        ) from e

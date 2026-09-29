"""
Common classes and functions used to compute the various climate indices.
"""

import functools
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import scipy.special
import scipy.stats

from climate_indices import lmoments, utils
from climate_indices.exceptions import (
    DistributionFittingError,
    GoodnessOfFitWarning,
    InsufficientDataError,
    InvalidArgumentError,
    MissingDataWarning,
    PearsonFittingError,
    PeriodicityError,
    ShortCalibrationWarning,
)
from climate_indices.logging_config import get_logger

if TYPE_CHECKING:
    # only for typing: climate_indices.indices imports this module, so importing
    # Distribution here at runtime would be circular
    from climate_indices.indices import Distribution

# declare the function names that should be included in the public API for this module
__all__ = [
    "FitDiagnostics",
    "Periodicity",
    "OutputScale",
    "OUTPUT_SCALES",
    "validate_output_scale",
    "fit_and_standardize",
    "fit_diagnostics",
    "prepare_scaled",
    "scale_values",
    "sum_to_scale",
    "transform_fitted_gamma",
    "transform_fitted_loglogistic",
    "transform_fitted_pearson",
    "DistributionFittingError",
    "InsufficientDataError",
    "PearsonFittingError",
    "DistributionFallbackStrategy",
    "ZeroHandling",
]

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

#: Where a zero accumulation is placed within the probability mass at zero, see
#: ADR-0015: "classic" scores it Φ⁻¹(p0), "center_of_mass" Φ⁻¹(p0 / 2), and
#: "mean_zero" −φ(Φ⁻¹(p0)) / p0.
ZeroHandling = Literal["classic", "center_of_mass", "mean_zero"]

_ZERO_HANDLING_MODES: tuple[str, ...] = get_args(ZeroHandling)


class _PearsonFitLost(Exception):
    """The Pearson Type III fit lost too many of the input's valid values to be usable.

    A fit outcome that warrants a gamma fall back, not an argument error: it is raised
    after a fit has run, so ``_fit_pearson_with_fallback`` may act on it.
    """


class DistributionFallbackStrategy:
    """Strategy class for managing Pearson→Gamma distribution fallback logic."""

    def __init__(self, max_nan_percentage: float = 0.5, high_failure_threshold: float = 0.8) -> None:
        """
        Initialize the fallback strategy.

        :param max_nan_percentage: Maximum percentage of NaN values before triggering fallback
        :param high_failure_threshold: Failure rate threshold for issuing warnings
        """
        self.max_nan_percentage = max_nan_percentage
        self.high_failure_threshold = high_failure_threshold
        self._logger = get_logger(self.__class__.__name__)

    def log_fallback_warning(self, reason: str, context: str = "") -> None:
        """Emit the ``distribution_fallback`` event recording the distribution swap."""
        self._logger.bind(
            from_distribution="pearson",
            to_distribution="gamma",
            reason=reason,
            context=context,
        ).warning("distribution_fallback")

    def should_fallback_from_excessive_nans(self, values: np.ndarray) -> bool:
        """Check if fallback is needed due to excessive NaN values."""
        if values.size == 0:
            return True
        nan_percentage = np.count_nonzero(np.isnan(values)) / values.size
        return bool(nan_percentage > self.max_nan_percentage)

    def should_warn_high_failure_rate(self, failure_count: int, total_count: int) -> bool:
        """Check if high failure rate warning should be issued."""
        if total_count == 0:
            return False
        failure_rate = failure_count / total_count
        return bool(failure_rate > self.high_failure_threshold)

    def log_high_failure_rate(self, failure_count: int, total_count: int, context: str = "") -> None:
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

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(s: str) -> "Periodicity":
        try:
            return Periodicity[s]
        except KeyError as err:
            raise ValueError(f"No periodicity enumeration corresponding to {s}") from err

    def unit(self) -> str:
        if self.name == "monthly":
            unit = "month"
        elif self.name == "daily":
            unit = "day"
        else:
            raise ValueError(f"No periodicity unit corresponding to {self.name}")

        return unit

    @property
    def period_length(self) -> int:
        """
        The number of time steps comprising one year at this periodicity,
        i.e. 12 for monthly data and 366 for daily data.
        """
        return int(self.value)


# the valid number of time steps per year, i.e. the length of the second axis
# of a 2-D (years, periods) input array
_PERIOD_LENGTHS = frozenset(periodicity.period_length for periodicity in Periodicity)

# the output conventions the standardized indices return: the standard-normal
# z-score ("normal", the default), the fitted cumulative probability
# ("probability", the PIT value in [0, 1]), and its signed counterpart
# ("bounded", 2p - 1 in [-1, 1])
OutputScale = Literal["normal", "probability", "bounded"]
OUTPUT_SCALES: tuple[OutputScale, ...] = ("normal", "probability", "bounded")


def _map_non_normal_scale(probabilities: np.ndarray, output_scale: OutputScale) -> np.ndarray | None:
    """Map fitted cumulative probabilities to the requested scale.

    Returns None for the default "normal" scale, whose z-scores come from the
    inverse-normal transform, so the caller keeps its own error handling there.
    """
    if output_scale == "probability":
        return probabilities
    if output_scale == "bounded":
        return (2.0 * probabilities) - 1.0
    return None


def validate_output_scale(output_scale: str) -> None:
    """Validate that an output scale is one of the accepted values.

    :param output_scale: the output-scale value to validate
    :raises InvalidArgumentError: if the value is not one of ``OUTPUT_SCALES``
    """
    if output_scale not in OUTPUT_SCALES:
        raise InvalidArgumentError(
            f"Invalid output_scale argument: {output_scale!r}. Supported values: {', '.join(OUTPUT_SCALES)}.",
            argument_name="output_scale",
            argument_value=repr(output_scale),
            valid_values=", ".join(OUTPUT_SCALES),
        )


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
            message = "1-D input array requires a corresponding periodicity argument, none provided"  # type: ignore[unreachable]
            _logger.error(
                "validation_error",
                operation="validate_array",
                reason="missing_periodicity",
                shape=str(values.shape),
            )
            raise PeriodicityError(message, periodicity_value=str(periodicity))

        elif periodicity is Periodicity.monthly or periodicity is Periodicity.daily:
            # we've been passed a 1-D array with shape (months) or (days),
            # reshape it to 2-D with shape (years, period_length)
            values = utils.reshape_to_2d(values, periodicity.period_length)

        else:
            message = f"Unsupported periodicity argument: '{periodicity}'"  # type: ignore[unreachable]
            _logger.error(
                "validation_error",
                operation="validate_array",
                reason="unsupported_periodicity",
                periodicity=str(periodicity),
            )
            raise PeriodicityError(message, periodicity_value=str(periodicity))

    elif len(values.shape) < 2 or values.shape[1] not in _PERIOD_LENGTHS:
        # not a 1-D array, and no valid period axis: an already-reshaped spatial
        # array carries its periods along axis 1, i.e. (years, periods, *cells)
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

    A time-major spatial array with shape (time, ``*cells``) is summed window-wise along
    its time axis, so every cell's sliding sums are computed by one vectorized pass.

    For example if the first array is [3, 4, 6, 2, 1, 3, 5, 8, 5] and
    the number of values to sum is 3 then the resulting array
    will be [np.nan, np.nan, 13, 12, 9, 6, 9, 16, 18].

    More generally::

        Y = f(X, n)

        Y[i] = np.nan, where i < n - 1
        Y[i] = sum(X[i - n + 1 : i + 1]), where i >= n - 1 and X[i - n + 1 : i + 1] contains no NaN values
        Y[i] = np.nan, where i >= n - 1 and X[i - n + 1 : i + 1] contains one or more NaN values

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

    if np.ma.isMaskedArray(values):
        # a masked value stands for a missing value: make it an explicit NaN so the
        # convolution below and the window-wise spatial sum see the missing marker
        # rather than the data under the mask (np.convolve reads under it)
        values = np.ma.filled(values.astype(float), np.nan)

    if values.ndim > 2:
        # time-major spatial arrays are summed window-wise along the time axis: one
        # vectorized window per time step for every cell, no per-cell Python loop
        # (np.convolve is 1-D only). The NaN pad is float64, as the 1-D path's
        # np.hstack([np.nan, ...]) is, so single-precision input still accumulates in
        # double precision.
        pad_shape = (scale - 1, *values.shape[1:])
        padded = np.concatenate((np.full(pad_shape, np.nan, dtype=float), values))
        window_sums: np.ndarray = np.lib.stride_tricks.sliding_window_view(padded, scale, axis=0).sum(axis=-1)
        return window_sums

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


def _log_and_raise_shape_error(shape: tuple[int, ...]) -> None:
    message = f"Invalid shape of input data array: {shape}"
    _logger.error(
        "validation_error",
        operation="validate_shape",
        reason="invalid_shape",
        shape=str(shape),
    )
    raise ValueError(message)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _reshape_time_major(values: np.ndarray, periodicity: Periodicity) -> np.ndarray:
    """
    Reshape a time-major spatial array to (years, periods, *cells).

    The spatial counterpart of ``utils.reshape_to_2d``: the time axis is folded onto
    a new period axis while every trailing cell dimension is left untouched, and a
    trailing partial period is padded with NaN values as in the 1-D case.

    :param values: time-major array of values, shape (time, *cells)
    :param periodicity: specifies whether data is monthly (12) or daily (366)
    :return: the values with shape (years, period_length, *cells)
    :raises PeriodicityError: if periodicity is neither monthly nor daily
    """
    if periodicity is not Periodicity.monthly and periodicity is not Periodicity.daily:
        raise PeriodicityError(f"Invalid periodicity argument: {periodicity}", periodicity_value=str(periodicity))

    period_length = periodicity.period_length
    cell_shape = values.shape[1:]
    final_period_values = values.shape[0] % period_length
    if final_period_values > 0:
        pads = [(0, period_length - final_period_values)] + [(0, 0)] * len(cell_shape)
        values = np.pad(values, pads, mode="constant", constant_values=np.nan)

    result: np.ndarray = values.reshape(-1, period_length, *cell_shape)
    _logger.debug(
        "array_reshaped",
        operation="reshape_time_major",
        input_shape=str(cell_shape),
        output_shape=str(result.shape),
    )
    return result


def reshape_values(values: np.ndarray, periodicity: Periodicity) -> np.ndarray:
    if periodicity is Periodicity.monthly or periodicity is Periodicity.daily:
        return utils.reshape_to_2d(values, periodicity.period_length)
    else:
        raise PeriodicityError(f"Invalid periodicity argument: {periodicity}", periodicity_value=str(periodicity))


def validate_values_shape(values: np.ndarray) -> int:
    if len(values.shape) != 2 or values.shape[1] not in _PERIOD_LENGTHS:
        _log_and_raise_shape_error(shape=values.shape)
    return int(values.shape[1])


def adjust_calibration_years(
    data_start_year: int, data_end_year: int, calibration_start_year: int, calibration_end_year: int
) -> tuple[int, int]:
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


def _validate_zero_handling(zero_handling: str) -> None:
    """Reject a zero-handling mode other than the three ADR-0015 names."""
    if not isinstance(zero_handling, str) or zero_handling not in _ZERO_HANDLING_MODES:
        raise ValueError(
            f"Invalid zero_handling argument: {zero_handling!r} -- "
            "must be one of 'classic', 'center_of_mass', or 'mean_zero'"
        )


def _calibration_probabilities_of_zero(calibration_values: np.ndarray) -> np.ndarray:
    """
    The zero fraction of each calendar step's non-missing calibration values.

    A step (or cell) with no non-missing calibration value has no defined zero mass,
    which is reported as NaN.

    :param calibration_values: calibration data, shape (years, time_steps) or
        (years, time_steps, *cells)
    :return: probabilities of zero, shape (time_steps,) or (time_steps, *cells)
    """
    if np.ma.isMaskedArray(calibration_values):
        # a mask is a missing marker, so it must not count as a non-missing value
        calibration_values = np.ma.filled(calibration_values.astype(float), np.nan)
    number_of_zeros = np.count_nonzero(calibration_values == 0, axis=0)
    number_of_non_missing = np.count_nonzero(~np.isnan(calibration_values), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities_of_zero: np.ndarray = np.where(
            number_of_non_missing > 0, number_of_zeros / number_of_non_missing, np.nan
        )
    return probabilities_of_zero


def _validate_gamma_probabilities_of_zero(values: np.ndarray, probabilities_of_zero: np.ndarray) -> np.ndarray:
    """
    Reject a supplied gamma probability of zero that does not fit the values.

    Its cell dimensions follow the rule for Pearson Type III parameters, and every
    value must lie in [0, 1]; NaN marks a step whose zero mass is undefined, as a
    step without calibration data reports it.

    :param values: the validated values, shape (years, time_steps) or
        (years, time_steps, *cells)
    :param probabilities_of_zero: the supplied probability of zero
    :return: the probability of zero as a float array
    :raises ValueError: if the shape or a value does not fit
    """
    probabilities_of_zero = np.asarray(probabilities_of_zero, dtype=float)
    period_length = values.shape[1]
    if probabilities_of_zero.ndim == 1 and probabilities_of_zero.shape[0] != period_length:
        raise ValueError(
            f"Fitting parameter 'prob_zero' has shape {probabilities_of_zero.shape}, which must carry "
            f"the period length {period_length}"
        )
    _validate_pearson_parameter_cells(values, (("prob_zero", probabilities_of_zero),))
    if np.any((probabilities_of_zero < 0.0) | (probabilities_of_zero > 1.0)):
        raise ValueError(
            "Fitting parameter 'prob_zero' must lie in [0, 1], or be NaN where a step's zero mass is undefined"
        )
    return probabilities_of_zero


def _validate_broadcastable_parameters(
    target_shape: tuple[int, ...], named_parameters: tuple[tuple[str, np.ndarray | None], ...]
) -> None:
    """
    Reject a supplied gamma ``alpha`` or ``beta`` that does not broadcast to ``target_shape``.

    Any shape NumPy broadcasts without growing the target, such as ``(periods, 1, 1)``
    for a ``(years, periods, lat, lon)`` block, stays accepted; one that would fail or
    add axes raises ``ValueError`` naming the parameter rather than an error from NumPy.

    :param target_shape: the shape the parameter is applied to
    :param named_parameters: (name, parameter) pairs, where a parameter may be None
    :raises ValueError: if a parameter does not broadcast to ``target_shape``
    """
    for name, parameter in named_parameters:
        if parameter is None:
            continue
        shape = np.shape(parameter)
        try:
            broadcasts = np.broadcast_shapes(target_shape, shape) == tuple(target_shape)
        except ValueError:
            broadcasts = False
        if not broadcasts:
            raise ValueError(
                f"Fitting parameter '{name}' has shape {shape}, which does not broadcast to the "
                f"values' shape {tuple(target_shape)}; give one value per period, or per period and cell"
            )


def _place_zeros(
    fitted_values: np.ndarray,
    zero_mask: np.ndarray,
    probabilities_of_zero: np.ndarray,
    zero_handling: ZeroHandling,
    output_scale: OutputScale = "normal",
) -> np.ndarray:
    """
    Give the zero positions the score a zero-handling mode assigns.

    ``"classic"`` returns ``fitted_values`` unchanged. The other modes overwrite the
    positions in ``zero_mask`` with a score computed in closed form from each step's
    effective probability of zero ``p0``, the one the transform used after its
    invalid-fit resets: on the normal scale ``Φ⁻¹(p0 / 2)`` for
    ``"center_of_mass"`` and ``−φ(Φ⁻¹(p0)) / p0`` for ``"mean_zero"``. On the
    probability and bounded scales both modes place a zero at ``p0 / 2``, the centre
    of the zero mass, mapped to the requested scale; the ``"mean_zero"`` property
    is defined on the normal scale only (ADR-0015, decision 7). Only steps with
    ``0 < p0 < 1`` are moved; at ``p0 == 0`` or ``p0 == 1`` every mode keeps the
    classic result (ADR-0015, decision 5).

    :param fitted_values: transformed values on ``output_scale``, shape
        (years, time_steps) or (years, time_steps, *cells)
    :param zero_mask: positions holding a zero (or, for Pearson Type III, a trace
        value), broadcastable to ``fitted_values``
    :param probabilities_of_zero: probability of zero per step, broadcastable to
        ``fitted_values`` along its trailing axes
    :param zero_handling: the zero-handling mode
    :param output_scale: the scale ``fitted_values`` is on, one of ``OUTPUT_SCALES``
    :return: the values with their zero positions moved
    """
    if zero_handling == "classic":
        return fitted_values

    probabilities_of_zero = np.asarray(probabilities_of_zero, dtype=float)
    movable = (probabilities_of_zero > 0.0) & (probabilities_of_zero < 1.0)
    # a placeholder inside (0, 1) keeps the closed forms finite where no zero moves
    safe_probabilities = np.where(movable, probabilities_of_zero, 0.5)
    zero_scores = _map_non_normal_scale(safe_probabilities / 2.0, output_scale)
    if zero_scores is None and zero_handling == "center_of_mass":
        zero_scores = scipy.stats.norm.ppf(safe_probabilities / 2.0)
    elif zero_scores is None:
        zero_scores = -scipy.stats.norm.pdf(scipy.stats.norm.ppf(safe_probabilities)) / safe_probabilities

    placed: np.ndarray = np.where(zero_mask & movable, zero_scores, fitted_values)
    return placed


def calculate_time_step_params(time_step_values: np.ndarray) -> tuple[float, float, float, float]:
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


def _pearson_parameters_spatial(
    calibration_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Fit every (time step, cell) of a (years, time_steps, *cells) block at once.

    The cell-axis counterpart of the per-time-step loop in ``pearson_parameters``:
    the L-moment fit runs once across every cell, and a cell whose sample fails
    either the minimum-non-zero guard or the L-moment validity check gets the same
    zeroed-parameter fallback the single-series path applies.

    :param calibration_values: calibration data with shape (years, time_steps, *cells)
    :return: four parameter arrays shaped (time_steps, *cells) and the count of
        failed (time step, cell) fits
    """
    locs, scales, skews, valid = lmoments.fit_spatial(calibration_values)

    number_of_zeros = np.count_nonzero(calibration_values == 0, axis=0)
    number_of_non_missing = np.count_nonzero(~np.isnan(calibration_values), axis=0)
    non_zero_count = number_of_non_missing - number_of_zeros
    valid = valid & (non_zero_count >= MIN_NON_ZERO_VALUES_FOR_PEARSON)

    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities_of_zero = np.where(number_of_zeros > 0, number_of_zeros / number_of_non_missing, 0.0)

    failed_fitting_count = int(np.count_nonzero(~valid))
    return (
        np.where(valid, probabilities_of_zero, 0.0),
        np.where(valid, locs, 0.0),
        np.where(valid, scales, 0.0),
        np.where(valid, skews, 0.0),
        failed_fitting_count,
    )


def _calibration_block(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
) -> tuple[np.ndarray, int]:
    """
    Fold an input into (years, time_steps, ...) and slice out its calibration years.

    A three-or-more-dimensional input is read as an already folded time-major spatial
    block. The calibration data quality is checked, and warnings are emitted, here.

    :return: the calibration values and the number of time steps per year
    """
    if getattr(values, "ndim", 0) > 2:
        # a folded spatial block carries its periods along axis 1 already
        values = _validate_array(values, periodicity)
        time_steps_per_year = int(values.shape[1])
    else:
        values = reshape_values(values, periodicity)
        time_steps_per_year = validate_values_shape(values)
    data_end_year = data_start_year + values.shape[0]
    calibration_start_year, calibration_end_year = adjust_calibration_years(
        data_start_year, data_end_year, calibration_start_year, calibration_end_year
    )
    calibration_begin_index = calibration_start_year - data_start_year
    calibration_end_index = (calibration_end_year - data_start_year) + 1
    calibration_values = values[calibration_begin_index:calibration_end_index, ...]

    # check calibration data quality and emit warnings if needed
    _check_calibration_data_quality(calibration_values, calibration_start_year, calibration_end_year)
    return calibration_values, time_steps_per_year


def pearson_parameters(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function computes the probability of zero and Pearson Type III
    distribution parameters corresponding to an array of values.

    :param values: 2-D array of values, with each row representing a year
        containing either 12 values corresponding to the calendar months of
        that year, or 366 values corresponding to the days of the year
        (with Feb. 29th being an average of the Feb. 28th and Mar. 1st values for
        non-leap years) and assuming that the first value of the array is
        January of the initial year for an input array of monthly values or
        Jan. 1st of initial year for an input array daily values. A time-major
        spatial block already folded to (years, time_steps, *cells) is also
        accepted, and then every cell is fitted in one pass; any
        three-or-more-dimensional input is read as that folded layout, so a
        time-major block must already be folded (``prepare_scaled`` owns that).
    :param data_start_year:
    :param calibration_start_year:
    :param calibration_end_year:
    :param periodicity: monthly or daily
    :return: four arrays of fitting values for the Pearson Type III
        distribution, with shape (12,) for monthly or (366,) for daily, or
        (time_steps, *cells) for spatial input

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

    calibration_values, time_steps_per_year = _calibration_block(
        values, data_start_year, calibration_start_year, calibration_end_year, periodicity
    )

    if calibration_values.ndim > 2:
        (
            probabilities_of_zero,
            locs,
            scales,
            skews,
            failed_fitting_count,
        ) = _pearson_parameters_spatial(calibration_values)
        cell_count = int(np.prod(calibration_values.shape[2:], dtype=np.intp))
        total_fitting_count = time_steps_per_year * cell_count
    else:
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
        total_fitting_count = time_steps_per_year

    # Check if we should warn about high failure rate using the fallback strategy
    if _default_fallback_strategy.should_warn_high_failure_rate(failed_fitting_count, total_fitting_count):
        _default_fallback_strategy.log_high_failure_rate(
            failure_count=failed_fitting_count,
            total_count=total_fitting_count,
            context="pearson_parameters computation",
        )

    # check goodness-of-fit and emit warning if poor
    _check_goodness_of_fit_pearson(calibration_values, probabilities_of_zero, locs, scales, skews)

    log.info("distribution_fitting_completed", output_shape=str(probabilities_of_zero.shape))
    return probabilities_of_zero, locs, scales, skews


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
    result: np.ndarray = loc - ((alpha * scale * skew) / 2.0)
    return result


def _pearson_output_from_probabilities(
    probabilities: np.ndarray,
    output_scale: OutputScale,
    skew: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Map fitted Pearson Type III cumulative probabilities onto the requested output scale.

    On the probability and bounded scales the fitted cumulative probability is
    the result, so the inverse-normal transform is skipped; on the normal scale
    the result is the normal distribution's quantile at each probability.

    :param probabilities: fitted cumulative probabilities, clipped to [0, 1]
    :param output_scale: one of ``compute.OUTPUT_SCALES``
    :param skew: first Pearson Type III parameter, for error context only
    :param loc: second Pearson Type III parameter, for error context only
    :param scale: third Pearson Type III parameter, for error context only
    """
    scaled = _map_non_normal_scale(probabilities, output_scale)
    if scaled is not None:
        return scaled

    try:
        result: np.ndarray = scipy.stats.norm.ppf(probabilities)
        return result
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


def _pearson_fit(
    values: np.ndarray,
    probabilities_of_zero: np.ndarray,
    skew: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
    output_scale: OutputScale = "normal",
    zero_handling: ZeroHandling = "classic",
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
    :param output_scale: one of ``compute.OUTPUT_SCALES``; "probability" returns the
        fitted cumulative probability and "bounded" returns ``2p - 1``, both without
        the inverse-normal transform
    :param zero_handling: where a zero or trace value (below 0.0005, where the
        probability of zero is positive) is placed within the zero mass; a
        non-classic mode overrides the support-limit masks at those positions
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

        # a zero value carries the point mass, in every mode
        values[zero_mask] = 0.0

        # The normal-scale sentinels keep norm.ppf finite just inside the
        # distribution's support boundaries, where the CDF is exactly 0 or 1.
        # On the probability scales the computed CDF is the result, so the
        # boundaries are pinned to 0 and 1 instead of nudged inward (which would
        # report a probability the fitted distribution never assigns).
        if output_scale == "normal":
            values[trace_mask] = 0.0005
            values[minimums_mask] = 0.0005
            values[maximums_mask] = 0.9995
        else:
            values[minimums_mask] = 0.0
            values[maximums_mask] = 1.0

        if not np.all(np.isnan(values)):
            # calculate the probability value, clipped between 0 and 1
            probabilities = np.clip(
                (probabilities_of_zero + ((1.0 - probabilities_of_zero) * values)),
                0.0,
                1.0,
            )
            fitted_values = _pearson_output_from_probabilities(probabilities, output_scale, skew, loc, scale)

            # a non-classic mode moves the zero and trace positions, overriding the
            # support-limit masks there (ADR-0015, decision 2)
            fitted_values = _place_zeros(fitted_values, zero_mask, probabilities_of_zero, zero_handling, output_scale)

        else:
            fitted_values = values

    else:
        fitted_values = values

    result: np.ndarray = fitted_values
    return result


def _validate_pearson_parameter_cells(
    values: np.ndarray, named_parameters: tuple[tuple[str, np.ndarray | None], ...]
) -> None:
    """Reject pre-computed Pearson parameters whose period or cell axes do not match a block."""
    period_length = values.shape[1]
    cells = values.shape[2:]
    for name, parameter in named_parameters:
        if parameter is None:
            continue
        parameter = np.asarray(parameter)
        if parameter.ndim == 1:
            if parameter.shape[0] != period_length:
                raise ValueError(
                    f"Fitting parameter '{name}' has shape {parameter.shape}, which must carry "
                    f"the period length {period_length}"
                )
        elif parameter.ndim > 1 and (parameter.shape[0] != period_length or parameter.shape[1:] != cells):
            raise ValueError(
                f"Fitting parameter '{name}' has shape {parameter.shape}, which must carry the "
                f"block's period length {period_length} and cell dimensions {cells}"
            )


def _reject_partial_pearson_parameters(supplied: tuple[np.ndarray | None, ...]) -> None:
    """Reject a Pearson Type III parameter set that supplies some but not all of the four."""
    if any(parameter is None for parameter in supplied) and not all(parameter is None for parameter in supplied):
        raise ValueError(
            "At least one but not all of the Pearson Type III fitting parameters are specified -- "
            "either none or all of these must be specified"
        )


def _validate_pearson_fitting_params(
    values: np.ndarray,
    probabilities_of_zero: np.ndarray | None,
    locs: np.ndarray | None,
    scales: np.ndarray | None,
    skews: np.ndarray | None,
) -> None:
    """Reject a partial or mis-shaped pre-computed Pearson Type III parameter set.

    Called before any fitting so a caller's argument error raises from every index,
    regardless of ``fallback_to_gamma``, instead of resembling a failed fit.

    :param values: the validated values, shape (years, periods) or (years, periods, *cells)
    :raises ValueError: if the set is partial, or a parameter does not carry the period (and cell) axes
    """
    _reject_partial_pearson_parameters((probabilities_of_zero, locs, scales, skews))
    _validate_pearson_parameter_cells(
        values,
        (("prob_zero", probabilities_of_zero), ("loc", locs), ("scale", scales), ("skew", skews)),
    )


def _prepare_spatial_parameters(
    values: np.ndarray, named_parameters: tuple[tuple[str, np.ndarray | None], ...]
) -> tuple[np.ndarray | None, ...]:
    """
    Shape pre-computed fitting parameters for a spatial block, or reject them.

    A period-only parameter array is reshaped so that it broadcasts along the
    period axis instead of aligning with the trailing cell axes. A parameter
    array carrying cell dimensions must match the block's own cell axes.

    :param values: the folded spatial block, shape (years, time_steps, *cells)
    :param named_parameters: (name, parameter) pairs, each parameter possibly None
    :return: the parameters in order, each shaped for the block or None
    """
    _validate_pearson_parameter_cells(values, named_parameters)
    cells = values.shape[2:]
    prepared: list[np.ndarray | None] = []
    for _, parameter in named_parameters:
        if parameter is None:
            prepared.append(None)
            continue
        parameter = np.asarray(parameter)
        if parameter.ndim == 1:
            parameter = parameter.reshape((1, parameter.shape[0], *([1] * len(cells))))
        prepared.append(parameter)
    return tuple(prepared)


def _prepare_pearson_spatial_parameters(
    values: np.ndarray,
    probabilities_of_zero: np.ndarray | None,
    locs: np.ndarray | None,
    scales: np.ndarray | None,
    skews: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Shape pre-computed Pearson parameters for a spatial block, or reject them."""
    prob_zero, loc, scale, skew = _prepare_spatial_parameters(
        values, (("prob_zero", probabilities_of_zero), ("loc", locs), ("scale", scales), ("skew", skews))
    )
    return prob_zero, loc, scale, skew


def transform_fitted_pearson(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    probabilities_of_zero: np.ndarray | None = None,
    locs: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    skews: np.ndarray | None = None,
    output_scale: OutputScale = "normal",
    *,
    zero_handling: ZeroHandling = "classic",
) -> np.ndarray:
    """
    Fit values to a Pearson Type III distribution and transform the values
    to corresponding normalized sigmas.

    :param values: 2-D array of values, with each row representing a year containing
                   twelve columns representing the respective calendar months,
                   or 366 columns representing days as if all years were leap years.
                   A time-major spatial block already folded to
                   (years, time_steps, ``*cells``) is also accepted; any
                   three-or-more-dimensional input is read as that folded layout.
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
    :param output_scale: one of ``compute.OUTPUT_SCALES``; "probability" returns the
        fitted cumulative probability, "bounded" returns ``2p - 1``, and "normal"
        (the default) returns the standard-normal z-score
    :param zero_handling: where a zero or trace value (below 0.0005, where the
        probability of zero is positive) is placed within the zero mass:
        ``"classic"`` (the default, and the existing behavior) scores it
        ``Φ⁻¹(p0)`` unless a support-limit mask overrides that score,
        ``"center_of_mass"`` ``Φ⁻¹(p0 / 2)``, and ``"mean_zero"``
        ``−φ(Φ⁻¹(p0)) / p0``. Only steps whose effective ``p0`` satisfies
        ``0 < p0 < 1`` are moved; a step whose fit failed, for example with fewer
        than four non-zero calibration values, has ``p0`` reset to 0 and keeps the
        classic score. A moved position overrides the support-limit masks. Trace
        values move although ``p0`` counts only exact zeros, so the modes' mean
        properties are approximate on this path. On the probability and bounded
        scales both non-classic modes place it at ``p0 / 2``. See ADR-0015.
    :return: 2-D array of transformed/fitted values, corresponding in size
             and shape of the input array
    :rtype: numpy.ndarray of floats
    :raises ValueError: if ``zero_handling`` is not one of the three modes, or the
        ``fitting_params`` set is partial or does not carry the period (and cell) axes
    """
    validate_output_scale(output_scale)
    _validate_zero_handling(zero_handling)

    log = _logger.bind(
        operation="transform_fitted_pearson",
        output_scale=output_scale,
        distribution="pearson3",
        periodicity=str(periodicity),
        input_shape=str(values.shape),
    )
    log.info("distribution_transform_started")

    # sanity check for the fitting parameters arguments
    pearson_param_args = [probabilities_of_zero, locs, scales, skews]
    _reject_partial_pearson_parameters((probabilities_of_zero, locs, scales, skews))

    # if we're passed all missing values then we can't compute anything,
    # and we'll return the same array of missing values
    if (isinstance(values, np.ma.MaskedArray) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # validate (and possibly reshape) the input array
    values = _validate_array(values, periodicity)

    # reject parameter arrays that do not carry the period (and cell) axes before any
    # fitting, so an argument error cannot be mistaken for a failed fit
    _validate_pearson_parameter_cells(
        values,
        (("prob_zero", probabilities_of_zero), ("loc", locs), ("scale", scales), ("skew", skews)),
    )

    # broadcast period-only parameters and reject parameter arrays whose cell
    # dimensions do not match a spatial block
    if values.ndim > 2:
        probabilities_of_zero, locs, scales, skews = _prepare_pearson_spatial_parameters(
            values, probabilities_of_zero, locs, scales, skews
        )

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

    # mypy narrowing: at this point parameters are guaranteed to be non-None
    assert probabilities_of_zero is not None
    assert locs is not None
    assert scales is not None
    assert skews is not None

    # fit each value to the Pearson Type III distribution
    values = _pearson_fit(values, probabilities_of_zero, skews, locs, scales, output_scale, zero_handling)

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
            missing_data_warning = MissingDataWarning(
                message,
                missing_ratio=missing_ratio,
                threshold=MISSING_DATA_THRESHOLD,
            )
            warnings.warn(missing_data_warning, stacklevel=3)


@functools.lru_cache(maxsize=32)
def _ks_critical_value(sample_size: int) -> float:
    """Critical Kolmogorov-Smirnov D statistic at the goodness-of-fit threshold.

    Args:
        sample_size: Number of valid values in the tested sample.

    Returns:
        The D statistic above which the fit is considered poor.
    """
    return float(scipy.stats.kstwo.isf(GOODNESS_OF_FIT_P_VALUE_THRESHOLD, sample_size))


def _ks_d_statistic(
    sorted_values: np.ndarray,
    cdf_values: np.ndarray,
) -> float:
    """Kolmogorov-Smirnov D statistic for a sample and its fitted CDF values.

    The statistic is computed directly rather than through ``scipy.stats.kstest``,
    whose argument-dispatch machinery dominates the runtime when the check runs once
    per grid cell.

    Args:
        sorted_values: Ascending valid sample values.
        cdf_values: Fitted CDF evaluated at ``sorted_values``.

    Returns:
        The largest absolute difference between the empirical and fitted CDFs.
    """
    sample_size = sorted_values.size
    ranks = np.arange(1, sample_size + 1)
    return float(
        max(
            (ranks / sample_size - cdf_values).max(),
            (cdf_values - (ranks - 1) / sample_size).max(),
        )
    )


def _ks_poor_fit_p_value(
    sorted_values: np.ndarray,
    cdf_values: np.ndarray,
) -> float | None:
    """Kolmogorov-Smirnov p-value for a sample, returned only when the fit is poor.

    The D statistic is computed directly rather than through ``scipy.stats.kstest``,
    whose argument-dispatch machinery dominates the runtime of this check when it runs
    once per grid cell. Clearly acceptable fits skip the exact p-value calculation.
    Candidate poor fits and values near the critical D value defer to SciPy.

    Args:
        sorted_values: Ascending valid sample values.
        cdf_values: Fitted CDF evaluated at ``sorted_values``.

    Returns:
        The p-value when it falls below the goodness-of-fit threshold, otherwise None.
    """
    sample_size = sorted_values.size
    d_statistic = _ks_d_statistic(sorted_values, cdf_values)
    critical_value = _ks_critical_value(sample_size)
    critical_tolerance = 0.0
    if np.issubdtype(sorted_values.dtype, np.floating):
        critical_tolerance = float(np.finfo(sorted_values.dtype).eps)
    if d_statistic < critical_value - critical_tolerance:
        return None

    # Match scipy.stats.kstest at the threshold, including its version-specific dtype handling.
    p_value = scipy.stats.kstest(sorted_values, lambda _: cdf_values).pvalue
    return float(p_value) if p_value < GOODNESS_OF_FIT_P_VALUE_THRESHOLD else None


def _check_goodness_of_fit_gamma(
    calibration_values: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> None:
    """
    Check goodness-of-fit for gamma distribution and emit aggregated warning if poor.

    Performs Kolmogorov-Smirnov tests for each time step and aggregates
    poor fits into a single warning to avoid flooding users with warnings.
    Spatial arrays with shape (years, time_steps, ...) evaluate one vectorized D
    statistic per time step across every cell, so the check costs no Python call
    per cell; only cells whose statistic reaches the critical value defer to
    SciPy for an exact p-value.

    :param calibration_values: Calibration data with shape (years, time_steps) or
        (years, time_steps, ...) for spatial input
    :param alphas: Shape parameters for gamma distribution
    :param betas: Scale parameters for gamma distribution
    """
    if calibration_values.ndim > 2:
        _check_goodness_of_fit_gamma_spatial(calibration_values, alphas, betas)
        return

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
                sorted_values = np.sort(valid_values)
                # the regularized lower incomplete gamma function is the gamma CDF
                p_value = _ks_poor_fit_p_value(
                    sorted_values,
                    scipy.special.gammainc(float(alpha), sorted_values.astype(float) / float(beta)),
                )
                if p_value is not None:
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


def _spatial_poor_fits(
    sorted_values: np.ndarray,
    valid_counts: np.ndarray,
    valid_positions: np.ndarray,
    cdf_values: np.ndarray,
    critical_tolerance: float,
    candidate_cdf: Callable[[np.ndarray, tuple[int, ...]], np.ndarray],
) -> list[tuple[int, float]]:
    """Poorly fitting (time step, cell) candidates and their exact p-values.

    The D statistic is a maximum over the ranked sample positions, evaluated for
    every (time step, cell) at once; positions beyond a cell's valid count and
    samples whose fitted parameters are invalid fall outside the comparison,
    matching the per-series check's own guards.

    :param sorted_values: Ascending calibration values, shaped (years, time_steps, ...)
    :param valid_counts: Per-cell valid sample count, shaped (time_steps, ...)
    :param valid_positions: Mask of positions inside a cell's valid sample with usable parameters
    :param cdf_values: Fitted CDF at ``sorted_values``, same shape
    :param critical_tolerance: Epsilon slack when comparing the D statistic to the critical value
    :param candidate_cdf: CDF for one candidate's sorted column and candidate index
    :return: The (time step index, p-value) pairs that fail the goodness-of-fit check
    """
    ranks = np.arange(1, sorted_values.shape[0] + 1).reshape((-1,) + (1,) * (sorted_values.ndim - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        upper = np.where(valid_positions, ranks / valid_counts - cdf_values, -np.inf)
        lower = np.where(valid_positions, cdf_values - (ranks - 1) / valid_counts, -np.inf)
    d_statistics = np.maximum(np.max(upper, axis=0), np.max(lower, axis=0))

    # critical values depend on the per-cell valid count, so evaluate the cached
    # exact statistic once per distinct count rather than once per cell
    critical_values = np.full(valid_counts.shape, np.inf)
    for valid_count in np.unique(valid_counts):
        if valid_count > 0:
            critical_values[valid_counts == valid_count] = _ks_critical_value(int(valid_count))

    candidates = np.nonzero(d_statistics >= critical_values - critical_tolerance)
    poor_fits: list[tuple[int, float]] = []
    for candidate in zip(*candidates, strict=True):
        step_index = int(candidate[0])
        valid_count = int(valid_counts[candidate])
        sorted_column = sorted_values[(slice(0, valid_count), *candidate)]
        try:
            p_value = _ks_poor_fit_p_value(sorted_column, candidate_cdf(sorted_column, candidate))
        except Exception:
            # ignore fitting errors during goodness-of-fit check, as the per-series path does
            continue
        if p_value is not None:
            poor_fits.append((step_index, p_value))
    return poor_fits


def _check_goodness_of_fit_gamma_spatial(
    calibration_values: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> None:
    """
    Gamma goodness-of-fit check across every cell of a (years, time_steps, ...) array.

    :param calibration_values: Calibration data with shape (years, time_steps, ...)
    :param alphas: Shape parameters, with shape (time_steps, ...)
    :param betas: Scale parameters, with shape (time_steps, ...)

    Peak memory is a few O(years x time_steps x cells) temporaries, so spatial blocks
    decide the footprint: chunk large grids rather than passing one dense block.
    """
    num_years = calibration_values.shape[0]
    time_steps = calibration_values.shape[1]
    cell_count = int(np.prod(calibration_values.shape[2:], dtype=np.intp))

    # NaN values sort last, so each cell's valid sample leads along the year axis
    sorted_values = np.sort(calibration_values, axis=0)
    valid_counts = np.count_nonzero(~np.isnan(calibration_values), axis=0)

    ranks = np.arange(1, num_years + 1).reshape((-1,) + (1,) * (calibration_values.ndim - 1))
    valid_positions = (ranks <= valid_counts) & np.isfinite(alphas[np.newaxis]) & np.isfinite(betas[np.newaxis])
    valid_positions &= (alphas[np.newaxis] > 0) & (betas[np.newaxis] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # the float64 casts match the per-series check's arithmetic
        cdf_values = scipy.special.gammainc(
            alphas[np.newaxis].astype(float),
            sorted_values.astype(float) / betas[np.newaxis].astype(float),
        )
    critical_tolerance = 0.0
    if np.issubdtype(calibration_values.dtype, np.floating):
        critical_tolerance = float(np.finfo(calibration_values.dtype).eps)
    poor_fits = _spatial_poor_fits(
        sorted_values,
        valid_counts,
        valid_positions,
        cdf_values,
        critical_tolerance,
        lambda column, candidate: scipy.special.gammainc(
            float(alphas[candidate]),
            column.astype(float) / float(betas[candidate]),
        ),
    )

    if not poor_fits:
        return

    # show up to 5 examples
    examples = poor_fits[:5]
    example_text = ", ".join([f"step {idx} (p={p:.4f})" for idx, p in examples])
    if len(poor_fits) > 5:
        example_text += f", and {len(poor_fits) - 5} more"

    comparisons = time_steps * cell_count
    message = (
        f"Gamma distribution shows poor goodness-of-fit for {len(poor_fits)} of "
        f"{comparisons} time step/cell combinations (p < {GOODNESS_OF_FIT_P_VALUE_THRESHOLD}). "
        f"Examples: {example_text}. Consider using a different distribution or "
        f"investigating data quality issues."
    )
    warning = GoodnessOfFitWarning(
        message,
        distribution_name="gamma",
        threshold=GOODNESS_OF_FIT_P_VALUE_THRESHOLD,
        poor_fit_count=len(poor_fits),
        total_steps=comparisons,
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

    :param calibration_values: Calibration data with shape (years, time_steps),
        or (years, time_steps, ...) for spatial input
    :param probabilities_of_zero: Probability of zero for each time step
    :param locs: Location parameters for Pearson Type III distribution
    :param scales: Scale parameters for Pearson Type III distribution
    :param skews: Skewness parameters for Pearson Type III distribution
    """
    if calibration_values.ndim > 2:
        _check_goodness_of_fit_pearson_spatial(calibration_values, probabilities_of_zero, locs, scales, skews)
        return

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
                sorted_values = np.sort(valid_values)
                p_value = _ks_poor_fit_p_value(
                    sorted_values,
                    scipy.stats.pearson3.cdf(sorted_values, skew, loc=loc, scale=scale),
                )
                if p_value is not None:
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


def _check_goodness_of_fit_pearson_spatial(
    calibration_values: np.ndarray,
    probabilities_of_zero: np.ndarray,
    locs: np.ndarray,
    scales: np.ndarray,
    skews: np.ndarray,
) -> None:
    """
    Pearson Type III goodness-of-fit check across every cell of a (years, time_steps, ...) array.

    The cell-axis counterpart of ``_check_goodness_of_fit_pearson``: one D statistic
    per (time step, cell) is evaluated with NumPy operations, so the check costs no
    Python call per cell; only candidates at the critical value defer to SciPy for an
    exact p-value.

    :param calibration_values: Calibration data with shape (years, time_steps, ...)
    :param probabilities_of_zero: Probability of zero, shaped (time_steps, ...)
    :param locs: Location parameters, shaped (time_steps, ...)
    :param scales: Scale parameters, shaped (time_steps, ...)
    :param skews: Skewness parameters, shaped (time_steps, ...)
    """
    num_years = calibration_values.shape[0]
    time_steps = calibration_values.shape[1]
    cell_count = int(np.prod(calibration_values.shape[2:], dtype=np.intp))

    # the per-series check drops zero values as well as NaNs before ranking, so both
    # are replaced with +inf to let each cell's valid sample lead along the year axis
    valid_mask = ~np.isnan(calibration_values) & (calibration_values != 0)
    sorted_values = np.sort(np.where(valid_mask, calibration_values, np.inf), axis=0)
    valid_counts = np.count_nonzero(valid_mask, axis=0)

    # fit failures are marked by all-zero parameters, and the per-series check skips
    # cells whose parameters are invalid or whose scale is not positive
    parameters_valid = ~((locs == 0) & (scales == 0) & (skews == 0))
    parameters_valid &= np.isfinite(locs) & np.isfinite(scales) & np.isfinite(skews) & (scales > 0)
    parameters_valid &= valid_counts > 0

    ranks = np.arange(1, num_years + 1).reshape((-1,) + (1,) * (calibration_values.ndim - 1))
    valid_positions = (ranks <= valid_counts) & parameters_valid[np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf_values = scipy.stats.pearson3.cdf(
            np.asarray(sorted_values, dtype=float),
            np.asarray(skews, dtype=float)[np.newaxis],
            loc=np.asarray(locs, dtype=float)[np.newaxis],
            scale=np.asarray(scales, dtype=float)[np.newaxis],
        )
    critical_tolerance = 0.0
    if np.issubdtype(calibration_values.dtype, np.floating):
        critical_tolerance = float(np.finfo(calibration_values.dtype).eps)
    poor_fits = _spatial_poor_fits(
        sorted_values,
        valid_counts,
        valid_positions,
        cdf_values,
        critical_tolerance,
        lambda column, candidate: cdf_values[(slice(0, column.size), *candidate)],
    )

    if not poor_fits:
        return

    # show up to 5 examples
    examples = poor_fits[:5]
    example_text = ", ".join([f"step {idx} (p={p:.4f})" for idx, p in examples])
    if len(poor_fits) > 5:
        example_text += f", and {len(poor_fits) - 5} more"

    comparisons = time_steps * cell_count
    message = (
        f"Pearson Type III distribution shows poor goodness-of-fit for {len(poor_fits)} of "
        f"{comparisons} time step/cell combinations (p < {GOODNESS_OF_FIT_P_VALUE_THRESHOLD}). "
        f"Examples: {example_text}. Consider using a different distribution or "
        f"investigating data quality issues."
    )
    warning = GoodnessOfFitWarning(
        message,
        distribution_name="pearson3",
        threshold=GOODNESS_OF_FIT_P_VALUE_THRESHOLD,
        poor_fit_count=len(poor_fits),
        total_steps=comparisons,
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
) -> tuple[np.ndarray, np.ndarray]:
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
    if (isinstance(values, np.ma.MaskedArray) and values.mask.all()) or np.all(np.isnan(values)):
        if periodicity is Periodicity.monthly:
            shape = (12,)
        elif periodicity is Periodicity.daily:
            shape = (366,)
        else:
            raise PeriodicityError(f"Unsupported periodicity: {periodicity}", periodicity_value=str(periodicity))
        if values.ndim > 2:
            # validated spatial arrays carry the periods along axis 1: (periods, *cells)
            shape = values.shape[1:]
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


def _prepare_input_shape(values: np.ndarray, spatial_time_major: bool) -> np.ndarray:
    """
    Flatten a 2-D input, pass a declared time-major spatial block through unchanged,
    and reject any other shape.

    We expect to operate upon a 1-D array, so a 2-D array is flattened. A time-major
    spatial block keeps its trailing cell dims, so that the scaling and everything
    downstream runs once per cell set rather than per cell, but only when the caller
    says the block is time-major: reading it by default would silently re-read a
    (years, periods, *cells) array along the wrong axis.
    """
    shape = values.shape
    if len(shape) == 2:
        return values.flatten()
    if len(shape) > 2:
        # every array with three or more dimensions is read as a time-major block, except
        # when its first cell axis is a calendar period length: that shape is equally
        # readable as a (years, periods, *cells) array, so it must be declared
        if not spatial_time_major and shape[1] in _PERIOD_LENGTHS:
            _logger.error(
                "validation_error",
                operation="prepare_scaled",
                reason="ambiguous_spatial_shape",
                shape=str(shape),
            )
            raise ValueError(
                f"Invalid shape of input array: {shape} -- a (time, *cells) block whose first cell axis "
                "is a calendar period length is ambiguous with a (years, periods, *cells) array; "
                "declare it with spatial_time_major=True"
            )
        return values
    if len(shape) != 1:
        _logger.error(
            "validation_error",
            operation="prepare_scaled",
            reason="invalid_shape",
            shape=str(shape),
        )
        raise ValueError(
            f"Invalid shape of input array: {shape} -- only 1-D arrays, 2-D (years, periods) "
            "arrays, and declared time-major spatial blocks are supported"
        )
    return values


def prepare_scaled(
    values: np.ndarray,
    scale: int,
    periodicity: Periodicity,
    *,
    clip_negatives: bool = True,
    reshape: bool = True,
    spatial_time_major: bool = False,
) -> np.ndarray:
    """
    Prepare an array of values for distribution fitting by flattening, clipping,
    summing each time step over the specified scale, and reshaping.

    This is the single owner of the preparation pipeline shared by the fitting-based
    indices (SPI, SPEI, EDDI, PNP) and the specialized CLI, so a policy change lands
    in every index at once. An all-missing 1-D or 2-D input is returned as a flattened
    array without computing anything, which callers can detect with
    ``prepared.ndim == 1`` in order to short-circuit; an all-missing time-major spatial
    input is returned with its (time, ``*cells``) shape. Shape errors are raised as
    ``ValueError``, the convention established by ``_validate_array`` and
    ``utils.reshape_to_2d``; an invalid periodicity raises ``PeriodicityError``, and a
    scale longer than the series raises ``InsufficientDataError``.

    Args:
        values: The array of values, either 1-D, 2-D (years, periods), or a time-major
            spatial array with shape (time, ``*cells``) and three or more dimensions,
            whose trailing cell dimensions are preserved.
        scale: The number of values for which each sliding summation will encompass.
        periodicity: Specifies whether data is monthly (12 time steps per year) or daily.
        clip_negatives: Whether negative values are clipped to zero, defaults to True.
        reshape: Whether the scaled values are reshaped to (years, period_length),
            defaults to True. For a time-major spatial input the result is
            (years, period_length, ``*cells``). ``indices.percentage_of_normal`` passes
            False, since it averages the un-reshaped 1-D sums over each calendar period.
        spatial_time_major: Declares that a three-or-more-dimensional ``values`` is a
            time-major block of independent time series, shaped (time, ``*cells``). That is
            how a block is read anyway, except when the first cell axis is a calendar
            period length (12 or 366), which makes the shape equally readable as a
            (years, periods, ``*cells``) array; there the caller has to say which it means.
            ``xarray_adapter`` sets this for every block it packs.

    Returns:
        The scaled values, either 2-D with shape (years, periodicity.period_length),
        three or more dimensions with shape (years, periodicity.period_length, ``*cells``)
        for a time-major spatial input, or 1-D when an all-missing input or
        ``reshape=False``.
    """
    _logger.debug("scaling_started", operation="prepare_scaled", scale=scale, periodicity=str(periodicity))

    # periodicity must be validated regardless of whether the result is reshaped,
    # since reshape_values() is the only other place this is checked and it's
    # skipped entirely when reshape=False
    if periodicity is not Periodicity.monthly and periodicity is not Periodicity.daily:
        raise PeriodicityError(f"Invalid periodicity argument: {periodicity}", periodicity_value=str(periodicity))

    values = _prepare_input_shape(values, spatial_time_major)

    # a scale longer than the series cannot produce a single complete sum, and
    # np.convolve's "valid" mode would silently return a longer result than the
    # input; the xarray adapter pre-checks its adapted length, so this is the NumPy
    # path's equivalent and both paths fail with the same error. This check precedes
    # the all-missing return so that a short all-missing series fails loudly too.
    available_steps = values.shape[0]
    if scale > available_steps:
        raise InsufficientDataError(
            f"Insufficient data for scale={scale}: {available_steps} time steps available, "
            f"but at least {scale} required.",
            non_zero_count=available_steps,
            required_count=scale,
        )

    # if we're passed all missing values then we can't compute anything,
    # so we return the same array of missing values
    if (isinstance(values, np.ma.MaskedArray) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # a partially masked input must become explicit NaN before sum_to_scale, since its
    # spatial branch concatenates through np.concatenate, which drops the mask and lets
    # the underlying fill values leak into sliding sums, calibration normals, and
    # percentages.
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values.astype(float), np.nan)

    # clip any negative values to zero. np.any(values < 0.0) is NaN-safe (NaN < 0
    # is False) and mask-safe (MaskedArray.any() ignores masked entries), unlike
    # np.amin/np.nanmin which either miss negatives behind a NaN or reach under a mask.
    if clip_negatives and bool(np.any(values < 0.0)):
        _logger.warning("negative_values_clipped", operation="prepare_scaled")
        values = np.clip(values, a_min=0.0, a_max=None)

    # get a sliding sums array, with each time step's value scaled
    # by the specified number of time steps
    scaled_values = sum_to_scale(values, scale)

    # a masked sum stands for the missing value it represents, so make it an explicit NaN
    if np.ma.isMaskedArray(scaled_values):
        scaled_values = np.ma.filled(scaled_values.astype(float), np.nan)

    # reshape precipitation values to (years, 12) for monthly,
    # or to (years, 366) for daily
    if reshape:
        scaled_values = (
            _reshape_time_major(scaled_values, periodicity)
            if scaled_values.ndim > 2
            else reshape_values(scaled_values, periodicity)
        )

    _logger.debug("scaling_completed", operation="prepare_scaled", output_shape=str(scaled_values.shape))
    return scaled_values


def scale_values(
    values: np.ndarray,
    scale: int,
    periodicity: Periodicity,
) -> np.ndarray:
    """
    Scale an array of values by summing each time step over the specified scale,
    clipping negative values to zero and reshaping to (years, periods).

    Thin wrapper over ``prepare_scaled``, which owns the preparation pipeline for
    every fitting-based index.

    Args:
        values: The array of values, either 1-D or 2-D (years, periods).
        scale: The number of values for which each sliding summation will encompass.
        periodicity: Specifies whether data is monthly (12 time steps per year) or daily.

    Returns:
        The scaled values, reshaped to (years, periodicity.period_length).
    """
    return prepare_scaled(values, scale, periodicity)


def _broadcast_fitting_parameters(
    values: np.ndarray,
    alphas: np.ndarray | None,
    betas: np.ndarray | None,
    probabilities_of_zero: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Reshape period-only fit parameters, shape (periods,), so that NumPy broadcasts
    them along axis 1 of a time-major spatial array instead of aligning them with the
    trailing cell axes.
    """
    if values.ndim <= 2:
        return alphas, betas, probabilities_of_zero

    if alphas is not None:
        alphas = np.asarray(alphas)
        if alphas.ndim == 1:
            alphas = alphas.reshape(1, -1, *([1] * (values.ndim - 2)))

    if betas is not None:
        betas = np.asarray(betas)
        if betas.ndim == 1:
            betas = betas.reshape(1, -1, *([1] * (values.ndim - 2)))

    if probabilities_of_zero is not None:
        probabilities_of_zero = np.asarray(probabilities_of_zero)
        if probabilities_of_zero.ndim == 1:
            probabilities_of_zero = probabilities_of_zero.reshape(1, -1, *([1] * (values.ndim - 2)))

    return alphas, betas, probabilities_of_zero


def _missing_where_zero_mass_undefined(
    transformed: np.ndarray, placement_mask: np.ndarray, undefined_zero_mass: np.ndarray
) -> np.ndarray:
    """Make the zeros of a step with no defined zero mass NaN (ADR-0015, decision 4)."""
    if not np.any(undefined_zero_mass):
        return transformed
    marked: np.ndarray = np.where(placement_mask & undefined_zero_mass, np.nan, transformed)
    return marked


def transform_fitted_gamma(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    alphas: np.ndarray | None = None,
    betas: np.ndarray | None = None,
    probabilities_of_zero: np.ndarray | None = None,
    output_scale: OutputScale = "normal",
    *,
    zero_handling: ZeroHandling = "classic",
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
    :param probabilities_of_zero: pre-computed probability of zero for each month or
        day of the year, in [0, 1], shaped like ``alphas``; when omitted it is the
        zero fraction of each step's non-missing calibration-period values,
        matching ``pearson_parameters`` (ADR-0015, decision 4). A step whose
        probability of zero is 1 is treated as having none, as no gamma
        distribution can be fitted to it. A step with no calibration data, or a
        NaN supplied here, has no defined zero mass, and its zeros are NaN.
    :param output_scale: one of ``compute.OUTPUT_SCALES``; "probability" returns the
        fitted cumulative probability, "bounded" returns ``2p - 1``, and "normal"
        (the default) returns the standard-normal z-score
    :param zero_handling: where a zero is placed within the zero mass:
        ``"classic"`` (the default, and the existing behavior) scores it
        ``Φ⁻¹(p0)``, ``"center_of_mass"`` ``Φ⁻¹(p0 / 2)``, and ``"mean_zero"``
        ``−φ(Φ⁻¹(p0)) / p0``. Only steps whose effective ``p0``, after the resets
        above, satisfies ``0 < p0 < 1`` are moved. On the probability and bounded
        scales both non-classic modes place it at ``p0 / 2``. The values are
        expected to be non-negative, as the index functions clip them; a negative
        value is placed with the zeros. See ADR-0015.
    :return: 2-D array of transformed/fitted values, corresponding in size
        and shape of the input array
    :rtype: numpy.ndarray of floats
    :raises ValueError: if ``zero_handling`` is not one of the three modes, a
        supplied ``alphas`` or ``betas`` does not broadcast to the values, a
        supplied ``probabilities_of_zero`` has cell dimensions that do not match
        the values, or a supplied ``probabilities_of_zero`` has a value outside
        [0, 1]
    """
    validate_output_scale(output_scale)
    _validate_zero_handling(zero_handling)

    log = _logger.bind(
        operation="transform_fitted_gamma",
        output_scale=output_scale,
        distribution="gamma",
        periodicity=str(periodicity),
        input_shape=str(values.shape),
    )
    log.info("distribution_transform_started")

    # if we're passed all missing values then we can't compute anything,
    # then we return the same array of missing values
    if (isinstance(values, np.ma.MaskedArray) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # a mask is a missing marker, so make it the explicit NaN the transform reads
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values.astype(float), np.nan)

    # validate (and possibly reshape) the input array
    values = _validate_array(values, periodicity)

    if probabilities_of_zero is not None:
        probabilities_of_zero = _validate_gamma_probabilities_of_zero(values, probabilities_of_zero)
    alphas, betas, probabilities_of_zero = _broadcast_fitting_parameters(values, alphas, betas, probabilities_of_zero)
    _validate_broadcastable_parameters(values.shape, (("alpha", alphas), ("beta", betas)))

    # Replace zeros with NaNs for fitting (zeros are excluded from gamma fitting)
    # and get mask of zero positions for later probability calculations
    zero_mask, values_for_fitting = _replace_zeros_with_nan(values)

    # find the fraction of zero values for each time step over the calibration
    # period's non-missing values, unless it was provided; a computed fraction
    # reaches 1 only where every calibration value is zero
    if probabilities_of_zero is None:
        first_year, last_year = adjust_calibration_years(
            data_start_year, data_start_year + values.shape[0], calibration_start_year, calibration_end_year
        )
        probabilities_of_zero = _calibration_probabilities_of_zero(
            values[first_year - data_start_year : last_year - data_start_year + 1, ...]
        )
        all_zero_steps = np.isclose(probabilities_of_zero, 1.0)
    else:
        # validated to lie in [0, 1], so this selects exactly a supplied mass of 1
        all_zero_steps = probabilities_of_zero >= 1.0

    # a step without calibration data (or a supplied NaN) has no defined zero mass:
    # its non-zero values transform as with none, and its zeros are NaN below
    undefined_zero_mass = np.isnan(probabilities_of_zero)

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
    probabilities_of_zero = np.where(all_zero_steps | undefined_zero_mass, 0.0, probabilities_of_zero)

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
    probabilities: np.ndarray = probabilities_of_zero + ((1 - probabilities_of_zero) * gamma_probabilities)

    # a negative value is below every zero, so it is placed with them
    placement_mask = values <= 0.0

    # on the probability and bounded scales the fitted cumulative probability is
    # the result, so the inverse-normal transform is skipped
    scaled = _map_non_normal_scale(probabilities, output_scale)
    if scaled is not None:
        scaled = _place_zeros(scaled, placement_mask, probabilities_of_zero, zero_handling, output_scale)
        scaled = _missing_where_zero_mass_undefined(scaled, placement_mask, undefined_zero_mass)
        log.info("distribution_transform_completed", output_shape=str(scaled.shape))
        return scaled

    # the values we'll return are the values at which the probabilities of
    # a normal distribution are less than or equal to the computed probabilities,
    # as determined by the normal distribution's quantile (or inverse
    # cumulative distribution) function
    try:
        result_values: np.ndarray = scipy.stats.norm.ppf(probabilities)
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

    result_values = _place_zeros(result_values, placement_mask, probabilities_of_zero, zero_handling)
    result_values = _missing_where_zero_mass_undefined(result_values, placement_mask, undefined_zero_mass)
    log.info("distribution_transform_completed", output_shape=str(result_values.shape))
    return result_values


# |shape| below this is treated as a zero-shape (ordinary logistic) GLO fit, matching
# PELGLO's SMALL constant in the R lmom package
_LOGLOGISTIC_SHAPE_TOLERANCE = 1e-6


def loglogistic_parameters(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the generalized logistic (GLO) distribution parameters corresponding to
    an array of values.

    This is the distribution R's ``SPEI`` package fits as ``"log-Logistic"`` (ub-pwm
    L-moments), i.e. the reference distribution for SPEI. Every value participates in
    the fit, including zeros: unlike gamma and Pearson Type III there is no separate
    zero mass, because SPEI's P−PET series is offset and has no physical zero mass.

    :param values: 2-D array of values, with each row representing a year containing
        twelve columns representing the respective calendar months, or 366 days per
        column as if all years were leap years; a time-major spatial block already
        folded to (years, time_steps, ``*cells``) is also accepted, and then every
        cell is fitted in one pass.
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period
    :param calibration_end_year: the final year to use for the calibration period
    :param periodicity: monthly or daily
    :return: three arrays of fitting values for the GLO distribution, with shape
        (12,) for monthly or (366,) for daily, or (time_steps, ``*cells``) for
        spatial input: location, scale, and shape
    :rtype: tuple of three numpy.ndarrays of floats (loc, scale, shape)
    """
    log = _logger.bind(
        operation="loglogistic_parameters",
        distribution="loglogistic",
        periodicity=str(periodicity),
        calibration_period=f"{calibration_start_year}-{calibration_end_year}",
    )
    log.info("distribution_fitting_started")

    calibration_values, time_steps_per_year = _calibration_block(
        values, data_start_year, calibration_start_year, calibration_end_year, periodicity
    )

    if calibration_values.ndim > 2:
        locs, scales, shapes, failed_fitting_count = _loglogistic_parameters_spatial(calibration_values)
        cell_count = int(np.prod(calibration_values.shape[2:], dtype=np.intp))
        total_fitting_count = time_steps_per_year * cell_count
    else:
        locs = np.zeros((time_steps_per_year,))
        scales = np.zeros((time_steps_per_year,))
        shapes = np.zeros((time_steps_per_year,))
        failed_fitting_count = 0

        for time_step_index in range(time_steps_per_year):
            try:
                params = lmoments.fit_glo(calibration_values[:, time_step_index])
                locs[time_step_index] = params["loc"]
                scales[time_step_index] = params["scale"]
                shapes[time_step_index] = params["shape"]
            except ValueError:
                # a step that cannot be fitted is marked invalid by its zeroed scale,
                # and the transform reports NaN there rather than aborting the series
                failed_fitting_count += 1
        total_fitting_count = time_steps_per_year

    if _default_fallback_strategy.should_warn_high_failure_rate(failed_fitting_count, total_fitting_count):
        # not log_high_failure_rate: its text names Pearson Type III and a Gamma remedy
        log.warning(
            "high_fitting_failure_rate",
            failure_count=failed_fitting_count,
            total_count=total_fitting_count,
        )

    log.info("distribution_fitting_completed", output_shape=str(locs.shape))
    return locs, scales, shapes


def _loglogistic_parameters_spatial(
    calibration_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit every (time step, cell) of a (years, time_steps, ``*cells``) block at once.

    :param calibration_values: calibration data with shape (years, time_steps, ``*cells``)
    :return: three parameter arrays shaped (time_steps, ``*cells``) and the count of
        failed (time step, cell) fits
    """
    locs, scales, shapes, valid = lmoments.fit_glo_spatial(calibration_values)
    failed_fitting_count = int(np.count_nonzero(~valid))
    return locs, scales, shapes, failed_fitting_count


def _loglogistic_fit(
    values: np.ndarray,
    locs: np.ndarray,
    scales: np.ndarray,
    shapes: np.ndarray,
    output_scale: OutputScale = "normal",
) -> np.ndarray:
    """Transform values through the fitted GLO cumulative distribution.

    This is Hosking's ``cdfglo``: with parameters (loc, scale, shape) and
    ``z = (x − loc) / scale``,

        ``F(x) = 1 / (1 + exp(−y))``, where
        ``y = z`` for ``shape = 0``, else ``y = −log(1 − shape·z) / shape``.

    A position whose parameters are missing, non-finite, non-positive in scale, or
    outside the GLO's ``|shape| < 1`` support is reported as NaN. A finite value
    beyond the fitted support follows ``cdfglo``: it maps to a probability of 0 or 1,
    whose normal-scale z is infinite and which the index layer clips to its range.

    :param values: an array of values to transform
    :param locs: location parameter, broadcastable to ``values``
    :param scales: scale parameter, broadcastable to ``values``
    :param shapes: shape parameter, broadcastable to ``values``
    :param output_scale: one of ``compute.OUTPUT_SCALES``
    :return: transformed values, shaped like the broadcast of the parameters
    """
    if np.all(np.isnan(values)):
        return values

    valid = np.isfinite(locs) & np.isfinite(scales) & np.isfinite(shapes) & (scales > 0.0) & (np.abs(shapes) < 1.0)
    negligible_shape = np.abs(shapes) <= _LOGLOGISTIC_SHAPE_TOLERANCE
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = (values - locs) / scales
        y = np.where(negligible_shape, z, -np.log(np.maximum(0.0, 1.0 - (shapes * z))) / shapes)
        probabilities = 1.0 / (1.0 + np.exp(-y))
    probabilities = np.clip(probabilities, 0.0, 1.0)

    scaled = _map_non_normal_scale(probabilities, output_scale)
    if scaled is None:
        scaled = scipy.stats.norm.ppf(probabilities)
    result: np.ndarray = np.where(valid, scaled, np.nan)
    return result


def transform_fitted_loglogistic(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    locs: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    shapes: np.ndarray | None = None,
    output_scale: OutputScale = "normal",
) -> np.ndarray:
    """
    Fit values to a generalized logistic (GLO) distribution and transform the values
    to corresponding normalized sigmas.

    :param values: 2-D array of values, with each row representing a year containing
        twelve columns representing the respective calendar months, or 366 columns
        representing days as if all years were leap years. A time-major spatial block
        already folded to (years, time_steps, ``*cells``) is also accepted.
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period
    :param calibration_end_year: the final year to use for the calibration period
    :param periodicity: the periodicity of the time series represented by the input data
    :param locs: pre-computed GLO location parameters, one per calendar step
    :param scales: pre-computed GLO scale parameters, one per calendar step
    :param shapes: pre-computed GLO shape parameters, one per calendar step
    :param output_scale: one of ``compute.OUTPUT_SCALES``; "probability" returns the
        fitted cumulative probability, "bounded" returns ``2p - 1``, and "normal"
        (the default) returns the standard-normal z-score
    :return: 2-D array of transformed/fitted values, corresponding in size and shape
        to the input array
    :rtype: numpy.ndarray of floats
    :raises ValueError: if only some of the three fitting parameters are provided
    """
    validate_output_scale(output_scale)

    log = _logger.bind(
        operation="transform_fitted_loglogistic",
        output_scale=output_scale,
        distribution="loglogistic",
        periodicity=str(periodicity),
        input_shape=str(values.shape),
    )
    log.info("distribution_transform_started")

    parameter_args = (locs, scales, shapes)
    if any(parameter is None for parameter in parameter_args) and not all(
        parameter is None for parameter in parameter_args
    ):
        raise ValueError(
            "At least one but not all of the log-logistic fitting parameters are "
            "specified -- either none or all of these must be specified"
        )

    # if we're passed all missing values then we can't compute anything,
    # and we'll return the same array of missing values
    if (isinstance(values, np.ma.MaskedArray) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values.astype(float), np.nan)

    values = _validate_array(values, periodicity)

    if values.ndim > 2:
        locs, scales, shapes = _prepare_spatial_parameters(
            values, (("loc", locs), ("scale", scales), ("shape", shapes))
        )

    # compute the GLO fitting values if none were provided
    if all(parameter is None for parameter in (locs, scales, shapes)):
        locs, scales, shapes = loglogistic_parameters(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
        )

    assert locs is not None
    assert scales is not None
    assert shapes is not None

    values = _loglogistic_fit(values, locs, scales, shapes, output_scale)

    log.info("distribution_transform_completed", output_shape=str(values.shape))
    return values


# normalized fitting-parameter keys, paired with the deprecated alias accepted for each
_FIT_ALTNAMES = (
    ("alpha", "alphas"),
    ("beta", "betas"),
    ("skew", "skews"),
    ("scale", "scales"),
    ("shape", "shapes"),
    ("loc", "locs"),
    ("prob_zero", "probabilities_of_zero"),
)


def _normalize_fitting_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Compatibility shim. Convert old accepted parameter dictionaries
    into new, consistently keyed parameter dictionaries. If given
    a None object, None is returned.

    See https://github.com/monocongo/climate_indices/issues/449
    """
    if params is None:
        return params

    normed = {}
    for name, altname in _FIT_ALTNAMES:
        if params.get(name) is not None:
            normed[name] = params[name]
        elif altname in params:
            _logger.warning(
                "Using deprecated fitting parameter key %s. Use %s instead.",
                altname,
                name,
            )
            normed[name] = params[altname]
        elif name in params:
            # an explicit None means "fit this parameter from the data", so the key is
            # kept rather than dropped
            normed[name] = params[name]
    return normed


def _fit_pearson_with_fallback(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    probabilities_of_zero: np.ndarray | None,
    locs: np.ndarray | None,
    scales: np.ndarray | None,
    skews: np.ndarray | None,
    fallback_context: str,
    output_scale: OutputScale = "normal",
    *,
    zero_handling: ZeroHandling = "classic",
) -> tuple[np.ndarray, bool, dict[str, np.ndarray] | None]:
    """Transform a Pearson Type III fit, falling back to gamma when the fit fails.

    The fall back is a property of the whole input block, not of one grid cell: when a
    Pearson Type III fit fails outright, or loses more than half of the input's valid
    values, the scaled input is refitted with gamma. A value that was already missing
    does not count against the fit.

    Args:
        values: 2-D (years, periods) or folded (years, periods, *cells) scaled values.
        probabilities_of_zero: Probability of zero, or None to fit it from the data.
        locs: Location parameter, or None to fit it from the data.
        scales: Scale parameter, or None to fit it from the data.
        skews: Skewness parameter, or None to fit it from the data.
        fallback_context: Context included in the fall-back warning log message.
        output_scale: One of ``compute.OUTPUT_SCALES``, applied to whichever
            distribution produces the result.
        zero_handling: Zero-placement mode, applied by whichever transform runs.

    Returns:
        The transformed values, whether the gamma fall back was used, and the fitted
        gamma parameters when it was (otherwise None).
    """
    if values.ndim == 1:
        # the Pearson fit reshapes a 1-D series to (years, periods); do it here so the
        # valid-input mask below has the shape of the fitted result
        values = _validate_array(values, periodicity)

    try:
        standardized = transform_fitted_pearson(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
            probabilities_of_zero,
            locs,
            scales,
            skews,
            output_scale,
            zero_handling=zero_handling,
        )

        # check if fallback is needed due to excessive NaN values, judging only the
        # values the fit lost: input that was already missing (an ocean mask, a sparse
        # series) says nothing about whether the Pearson fit worked, and an input with
        # nothing valid has nothing to lose
        valid = ~np.isnan(values)
        if valid.any() and _default_fallback_strategy.should_fallback_from_excessive_nans(standardized[valid]):
            raise _PearsonFitLost("Pearson distribution fitting resulted in excessive missing values")

    except (DistributionFittingError, _PearsonFitLost) as e:
        # only a fit outcome reaches here: the transform rejects an argument error (a
        # partial or mis-shaped parameter set) with a plain ValueError, which this
        # narrowed except lets propagate, so spi raises where it once fell back
        _default_fallback_strategy.log_fallback_warning(str(e), context=fallback_context)

        # the fall back refits the scaled input, never the Pearson result it replaces;
        # the parameters are fitted here rather than inside the transform so a caller
        # reporting the fit can name the distribution it actually used
        alphas, betas = gamma_parameters(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
        )
        fallback_values = transform_fitted_gamma(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
            alphas=alphas,
            betas=betas,
            output_scale=output_scale,
            zero_handling=zero_handling,
        )
        return fallback_values, True, {"alpha": alphas, "beta": betas}

    return standardized, False, None


def fit_and_standardize(
    values: np.ndarray,
    distribution: "Distribution",
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: dict[str, Any] | None = None,
    *,
    fallback_to_gamma: bool = False,
    fallback_context: str = "",
    output_scale: OutputScale = "normal",
    zero_handling: ZeroHandling = "classic",
) -> np.ndarray:
    """
    Fit values to the specified distribution and transform the values to the
    corresponding normalized sigmas.

    This is the single seam where the fitting-based indices (SPI, SPEI) own the
    fitting-parameter normalization, the gamma/Pearson dispatch, and the policy of
    falling back from a failed Pearson Type III fit to gamma. ``fallback_to_gamma``
    makes that policy a parameter of the call rather than a copy of this branch in
    each index function; the indices pass ``False`` unless they intend to fall back.

    Args:
        values: 2-D array of scaled values, with each row typically representing a
            year containing twelve columns representing the respective calendar
            months, or 366 days per column as if all years were leap years; a
            time-major spatial block with more than two dimensions is also accepted.
        distribution: The distribution to fit the values to: gamma, Pearson Type III,
            or the generalized logistic ("loglogistic") that SPEI standardizes with.
        data_start_year: The initial year of the input values array.
        calibration_start_year: The initial year to use for the calibration period.
        calibration_end_year: The final year to use for the calibration period.
        periodicity: The type of time series represented by the input data, either
            monthly (12 time steps per year) or daily (366 time steps per year).
        fitting_params: Optional dictionary of pre-computed distribution fitting
            parameters, with the keys "alpha" and "beta" (and optionally
            "prob_zero") when fitting to gamma and "prob_zero", "loc", "scale", and
            "skew" when fitting to Pearson Type III. Deprecated aliases such as
            "alphas" and "probabilities_of_zero" are accepted, and an explicit None
            means "fit this parameter from the data". A gamma fit without
            "prob_zero" computes it over the calibration period.
        fallback_to_gamma: Whether to fall back to the gamma distribution when a
            Pearson Type III fit fails or loses too many of the input's valid
            values; input that was already missing does not count. The fall back
            fits gamma to the scaled input, and the decision is made once for the
            whole input block, not per grid cell.
        fallback_context: Context included in the fall-back warning log message.
        output_scale: One of ``compute.OUTPUT_SCALES``. "normal" (the default)
            returns the standard-normal z-score, "probability" the fitted
            cumulative probability in [0, 1], and "bounded" ``2p - 1`` in [-1, 1].
        zero_handling: Where a zero accumulation is placed within the zero mass,
            one of "classic" (the default), "center_of_mass", or "mean_zero"; see
            ADR-0015. A gamma fall back applies the same mode. The log-logistic fit
            has no zero mass, so the mode does not apply to it.

    Returns:
        2-D array of transformed/fitted values, corresponding in size and shape to
        the input array.

    Raises:
        InvalidArgumentError: If ``output_scale`` is not one of ``compute.OUTPUT_SCALES``.
        ValueError: If the distribution is neither gamma nor Pearson Type III,
            ``zero_handling`` is not one of the three modes, or a Pearson
            ``fitting_params`` set is partial or does not carry the period (and cell) axes.
    """
    validate_output_scale(output_scale)
    _validate_zero_handling(zero_handling)
    params = _normalize_fitting_params(fitting_params) or {}

    if distribution.value == "gamma":
        return transform_fitted_gamma(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
            params.get("alpha"),
            params.get("beta"),
            params.get("prob_zero"),
            output_scale,
            zero_handling=zero_handling,
        )

    if distribution.value == "loglogistic":
        # the GLO is the SPEI reference distribution: P−PET is offset and has no
        # physical zero mass, so it has no zero-placement mode and zero_handling is
        # not applied on this branch
        return transform_fitted_loglogistic(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
            params.get("loc"),
            params.get("scale"),
            params.get("shape"),
            output_scale,
        )

    if distribution.value != "pearson":
        raise ValueError(f"Unsupported distribution: {distribution}")

    probabilities_of_zero = params.get("prob_zero")
    locs = params.get("loc")
    scales = params.get("scale")
    skews = params.get("skew")

    if values.ndim > 2:
        # reject mismatched parameter cells before the fall-back try: that is an
        # argument error, not a Pearson fit failure to fall back from
        _validate_pearson_parameter_cells(
            values,
            (
                ("prob_zero", probabilities_of_zero),
                ("loc", locs),
                ("scale", scales),
                ("skew", skews),
            ),
        )

    if not fallback_to_gamma:
        return transform_fitted_pearson(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
            probabilities_of_zero,
            locs,
            scales,
            skews,
            output_scale,
            zero_handling=zero_handling,
        )

    standardized, _, _ = _fit_pearson_with_fallback(
        values,
        data_start_year,
        calibration_start_year,
        calibration_end_year,
        periodicity,
        probabilities_of_zero,
        locs,
        scales,
        skews,
        fallback_context,
        output_scale,
        zero_handling=zero_handling,
    )
    return standardized


@dataclass(frozen=True, eq=False)
class FitDiagnostics:
    """Per-calendar-step diagnostics for a fitted distribution.

    Each array is shaped ``(time_steps,)`` for a single series and
    ``(time_steps, *cells)`` for a spatial block. ``parameters`` maps the canonical
    ``fitting_params`` keys to the arrays that produced the fit, so it can be passed
    straight back to :func:`fit_and_standardize` (or to ``spi`` /
    ``standardized_index``) to reproduce it.
    """

    #: The distribution actually used, after any Pearson-to-gamma fall back.
    distribution: "Distribution"
    #: The fitted parameters, keyed as ``fitting_params`` accepts them:
    #: ``alpha``/``beta``/``prob_zero`` for gamma, and
    #: ``prob_zero``/``loc``/``scale``/``skew`` for Pearson Type III.
    parameters: dict[str, np.ndarray]
    #: Probability of a zero accumulation, per calendar step, computed over the
    #: calibration period's non-missing values for both distributions, or the
    #: ``prob_zero`` supplied in ``fitting_params`` for the distribution requested;
    #: a Pearson-to-gamma fall back computes its own. For gamma, NaN marks a step
    #: without calibration data, and the transform clamps an all-zero step's mass
    #: to 0, which this field reports as 1.0.
    prob_zero: np.ndarray
    #: Number of non-missing, non-zero calibration values entering the
    #: Kolmogorov-Smirnov test.
    n_valid: np.ndarray
    #: Kolmogorov-Smirnov D statistic of the fit, missing where the step has no
    #: valid sample or no usable fitted parameters.
    ks_statistic: np.ndarray
    #: Exact Kolmogorov-Smirnov p-value, on the same eligibility as
    #: ``ks_statistic``. It is computed for every eligible calendar step (and cell),
    #: unlike the goodness-of-fit warning path, which skips the exact p-value for
    #: clearly acceptable fits; retrieving it is therefore more expensive.
    ks_p_value: np.ndarray
    #: Whether a Pearson Type III fit fell back to gamma.
    fell_back_to_gamma: bool


def _as_period_cell_parameters(
    values: np.ndarray,
    *parameters: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Broadcast period-only fit parameters to a folded block's (time_steps, *cells).

    A parameter supplied for a spatial block may carry one value per calendar period,
    which NumPy would otherwise align with the trailing cell axes. A parameter that
    already carries the cell dimensions is passed through unchanged.
    """
    if values.ndim <= 2:
        return parameters

    prepared: list[np.ndarray] = []
    for parameter in parameters:
        parameter = np.asarray(parameter)
        if parameter.ndim == 1:
            broadcast = np.broadcast_to(
                parameter.reshape((1, parameter.shape[0], *([1] * (values.ndim - 2)))),
                values.shape,
            )[0]
            parameter = np.array(broadcast)
        prepared.append(parameter)
    return tuple(prepared)


def _ks_fit_diagnostics(
    calibration_values: np.ndarray,
    parameters_valid: np.ndarray,
    cdf_for_series: Callable[[np.ndarray, tuple[int, ...]], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-series Kolmogorov-Smirnov D, exact p-value, and valid sample count.

    The sample is the calibration period's non-missing, non-zero values, matching the
    goodness-of-fit warnings. Unlike that path, which skips the exact p-value for
    clearly acceptable fits, the p-value is computed for every eligible series; on a
    large grid the per-cell SciPy calls make this the dominant cost.

    Args:
        calibration_values: Calibration data, shaped (years, time_steps) or
            (years, time_steps, *cells).
        parameters_valid: Per (time step, cell) flag marking usable fitted parameters.
        cdf_for_series: Fitted CDF for one series' sorted sample and its index.

    Returns:
        The D statistic, p-value, and valid sample count, each shaped
        (time_steps,) or (time_steps, *cells).
    """
    shape = (calibration_values.shape[1], *calibration_values.shape[2:])
    n_valid = np.zeros(shape, dtype=np.intp)
    ks_statistic = np.full(shape, np.nan)
    ks_p_value = np.full(shape, np.nan)

    # a missing or zero calibration value does not enter the fit, and +inf sorts it
    # past every cell's valid sample along the year axis
    valid_mask = ~np.isnan(calibration_values) & (calibration_values != 0)
    sorted_values = np.sort(np.where(valid_mask, calibration_values, np.inf), axis=0)
    valid_counts = valid_mask.sum(axis=0)

    for index in np.ndindex(shape):
        count = int(valid_counts[index])
        n_valid[index] = count
        if count == 0 or not parameters_valid[index]:
            continue
        sample = sorted_values[(slice(0, count), *index)]
        cdf_values = cdf_for_series(sample, index)
        kstest_result = scipy.stats.kstest(sample, lambda _, cdf=cdf_values: cdf)
        ks_statistic[index] = float(kstest_result.statistic)
        ks_p_value[index] = float(kstest_result.pvalue)

    return ks_statistic, ks_p_value, n_valid


def _diagnostic_pearson_parameters(
    values: np.ndarray,
    params: dict[str, Any],
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fallback_to_gamma: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Fit or take the caller's Pearson parameters, rejecting a partial or mis-shaped set."""
    probabilities_of_zero = params.get("prob_zero")
    locs = params.get("loc")
    scales = params.get("scale")
    skews = params.get("skew")
    supplied = (probabilities_of_zero, locs, scales, skews)

    # an argument error raises here in every surface, regardless of the fall back
    _validate_pearson_fitting_params(values, probabilities_of_zero, locs, scales, skews)

    if all(parameter is None for parameter in supplied):
        try:
            probabilities_of_zero, locs, scales, skews = pearson_parameters(
                values, data_start_year, calibration_start_year, calibration_end_year, periodicity
            )
        except DistributionFittingError:
            if not fallback_to_gamma:
                raise
            # Let the guarded transform refit and apply its gamma fallback.
            return None, None, None, None
        return probabilities_of_zero, locs, scales, skews
    # the partial set was rejected above, so all four are supplied here
    assert probabilities_of_zero is not None
    assert locs is not None
    assert scales is not None
    assert skews is not None
    probabilities_of_zero, locs, scales, skews = _as_period_cell_parameters(
        values, probabilities_of_zero, locs, scales, skews
    )
    return probabilities_of_zero, locs, scales, skews


def _diagnostic_gamma_probabilities_of_zero(
    values: np.ndarray,
    calibration_values: np.ndarray,
    supplied_probabilities_of_zero: np.ndarray | None,
) -> np.ndarray:
    """The gamma zero mass the transform uses: the supplied one, else the calibration period's."""
    if supplied_probabilities_of_zero is None:
        return _calibration_probabilities_of_zero(calibration_values)
    supplied = _validate_gamma_probabilities_of_zero(values, supplied_probabilities_of_zero)
    (probabilities_of_zero,) = _as_period_cell_parameters(values, supplied)
    return probabilities_of_zero


def fit_diagnostics(
    values: np.ndarray,
    distribution: "Distribution",
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: dict[str, Any] | None = None,
    *,
    fallback_to_gamma: bool = False,
    fallback_context: str = "",
) -> FitDiagnostics:
    """Fit values to a distribution and report per-calendar-step fit diagnostics.

    This is the audit surface for the fitting-based indices: it returns the fitted
    parameters in the same keys :func:`fit_and_standardize` accepts, the probability of
    zero, the number of valid calibration values, and the Kolmogorov-Smirnov D
    statistic and exact p-value for every calendar step (and every cell, for a folded
    spatial block). It does not transform the values and does not change the value any
    index returns.

    Args:
        values: 2-D (years, periods) array of scaled values, or a folded time-major
            spatial block with more than two dimensions, shaped
            (years, periods, ``*cells``).
        distribution: The distribution to fit the values to.
        data_start_year: The initial year of the input values array.
        calibration_start_year: The initial year to use for the calibration period.
        calibration_end_year: The final year to use for the calibration period.
        periodicity: Monthly or daily time steps.
        fitting_params: Optional pre-computed fitting parameters, with the same keys
            :func:`fit_and_standardize` accepts; deprecated aliases are normalized. A
            parameter left as None is fitted from the data.
        fallback_to_gamma: Whether to report the gamma fall back when a Pearson Type
            III fit fails or loses too many of the input's valid values, matching the
            ``spi``/``standardized_index`` policy. A value that was already missing
            does not count against the fit.
        fallback_context: Context included in the fall-back warning log message.

    Returns:
        A :class:`FitDiagnostics` whose ``parameters`` arrays can be fed back through
        ``fitting_params`` to reproduce the fit.

    Raises:
        ValueError: If the distribution is neither gamma nor Pearson Type III, or a
            Pearson ``fitting_params`` set is partial or does not carry the period
            (and cell) axes.
    """
    # only needed to name the gamma fall back's result; importing at module load would
    # be circular because indices imports this module
    from climate_indices.indices import Distribution

    params = _normalize_fitting_params(fitting_params) or {}
    if distribution.value not in ("gamma", "pearson"):
        raise ValueError(f"Unsupported distribution: {distribution}")
    # a mask is a missing marker: make it the explicit NaN the diagnostics read
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values.astype(float), np.nan)
    values = _validate_array(values, periodicity)

    data_end_year = data_start_year + values.shape[0]
    calibration_start_year, calibration_end_year = adjust_calibration_years(
        data_start_year, data_end_year, calibration_start_year, calibration_end_year
    )
    calibration_values = values[
        calibration_start_year - data_start_year : calibration_end_year - data_start_year + 1, ...
    ]

    fell_back_to_gamma = False
    fallback_parameters: dict[str, np.ndarray] | None = None
    if distribution.value != "gamma":
        probabilities_of_zero, locs, scales, skews = _diagnostic_pearson_parameters(
            values,
            params,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
            fallback_to_gamma,
        )

        if fallback_to_gamma:
            # the fall back belongs to the block, so run the same transform the index
            # runs and report whichever distribution it used
            _, fell_back_to_gamma, fallback_parameters = _fit_pearson_with_fallback(
                values,
                data_start_year,
                calibration_start_year,
                calibration_end_year,
                periodicity,
                probabilities_of_zero,
                locs,
                scales,
                skews,
                fallback_context,
            )

        if not fell_back_to_gamma:
            # parameters are complete here: they were fitted or supplied in full
            assert probabilities_of_zero is not None
            assert locs is not None
            assert scales is not None
            assert skews is not None
            parameters_valid = ~((locs == 0) & (scales == 0) & (skews == 0))
            parameters_valid &= np.isfinite(locs) & np.isfinite(scales) & np.isfinite(skews) & (scales > 0)
            ks_statistic, ks_p_value, n_valid = _ks_fit_diagnostics(
                calibration_values,
                parameters_valid,
                lambda sample, index: scipy.stats.pearson3.cdf(
                    sample,
                    float(skews[index]),
                    loc=float(locs[index]),
                    scale=float(scales[index]),
                ),
            )
            return FitDiagnostics(
                distribution=distribution,
                parameters={
                    "prob_zero": probabilities_of_zero,
                    "loc": locs,
                    "scale": scales,
                    "skew": skews,
                },
                prob_zero=probabilities_of_zero,
                n_valid=n_valid,
                ks_statistic=ks_statistic,
                ks_p_value=ks_p_value,
                fell_back_to_gamma=False,
            )

    # gamma, either requested or the fall back from a failed Pearson fit
    alphas = params.get("alpha")
    betas = params.get("beta")
    if fallback_parameters is not None:
        # reuse the gamma fit the fall back already made rather than fitting it twice
        alphas, betas = fallback_parameters["alpha"], fallback_parameters["beta"]
    elif alphas is None or betas is None:
        alphas, betas = gamma_parameters(
            values, data_start_year, calibration_start_year, calibration_end_year, periodicity
        )
    else:
        alphas, betas = _as_period_cell_parameters(values, alphas, betas)
        # the per-step diagnostics index every parameter at (time_step, *cell)
        _validate_broadcastable_parameters(values.shape[1:], (("alpha", alphas), ("beta", betas)))
        alphas, betas = (np.array(np.broadcast_to(parameter, values.shape[1:])) for parameter in (alphas, betas))

    # the fall back refits the zero mass too, so only a requested gamma fit reads a
    # supplied one, as the transform does
    probabilities_of_zero = _diagnostic_gamma_probabilities_of_zero(
        values, calibration_values, params.get("prob_zero") if fallback_parameters is None else None
    )
    parameters_valid = np.isfinite(alphas) & np.isfinite(betas) & (alphas > 0) & (betas > 0)
    ks_statistic, ks_p_value, n_valid = _ks_fit_diagnostics(
        calibration_values,
        parameters_valid,
        lambda sample, index: scipy.special.gammainc(float(alphas[index]), sample.astype(float) / float(betas[index])),
    )
    return FitDiagnostics(
        distribution=Distribution.gamma,
        parameters={"alpha": alphas, "beta": betas, "prob_zero": probabilities_of_zero},
        prob_zero=probabilities_of_zero,
        n_valid=n_valid,
        ks_statistic=ks_statistic,
        ks_p_value=ks_p_value,
        fell_back_to_gamma=fell_back_to_gamma,
    )

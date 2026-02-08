"""Custom exception hierarchy for the climate_indices library.

This module defines all exception types used throughout the library, enabling
users to catch all library-specific errors with a single handler.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ClimateIndicesError",
    "DistributionFittingError",
    "InsufficientDataError",
    "PearsonFittingError",
    "DimensionMismatchError",
    "CoordinateValidationError",
    "InputTypeError",
    "InvalidArgumentError",
    "ClimateIndicesWarning",
    "MissingDataWarning",
    "ShortCalibrationWarning",
    "GoodnessOfFitWarning",
    "InputAlignmentWarning",
]


class ClimateIndicesError(Exception):
    """Base exception for all climate_indices library errors.

    Catch this exception to handle any error raised by the library.
    """

    pass


class DistributionFittingError(ClimateIndicesError):
    """Base exception for distribution fitting failures.

    Raised when statistical distribution fitting operations fail due to
    data quality issues or numerical problems.

    Attributes:
        distribution_name: Name of the distribution that failed (e.g., "gamma", "pearson3")
        input_shape: Shape of the input array that caused the failure
        parameters: Dictionary of parameter names to summarized values
        suggestion: Suggested alternative approach or distribution to try
        underlying_error: The original exception that caused the failure
    """

    def __init__(
        self,
        message: str,
        *,
        distribution_name: str | None = None,
        input_shape: tuple[int, ...] | None = None,
        parameters: dict[str, str] | None = None,
        suggestion: str | None = None,
        underlying_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.distribution_name = distribution_name
        self.input_shape = input_shape
        self.parameters = parameters
        self.suggestion = suggestion
        self.underlying_error = underlying_error


class InsufficientDataError(DistributionFittingError):
    """Raised when there is insufficient data for distribution fitting.

    This exception includes context about how much valid data was found
    versus what is required for reliable distribution fitting.

    Attributes:
        non_zero_count: Number of non-zero values found in the data
        required_count: Minimum number of values required for fitting
    """

    def __init__(
        self,
        message: str,
        non_zero_count: int | None = None,
        required_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.non_zero_count = non_zero_count
        self.required_count = required_count


class PearsonFittingError(DistributionFittingError):
    """Raised when Pearson Type III distribution fitting fails.

    This exception wraps underlying numerical errors that occur during
    the Pearson Type III fitting process.

    Attributes:
        underlying_error: The original exception that caused the fitting failure
    """

    def __init__(
        self,
        message: str,
        underlying_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.underlying_error = underlying_error


class DimensionMismatchError(ClimateIndicesError):
    """Raised when array dimensions don't match expected structure.

    This exception is raised when input arrays have incompatible shapes
    or when required dimensions are missing.

    Attributes:
        expected_dims: The expected dimension structure
        actual_dims: The actual dimension structure found
    """

    def __init__(
        self,
        message: str,
        expected_dims: Any = None,
        actual_dims: Any = None,
    ) -> None:
        super().__init__(message)
        self.expected_dims = expected_dims
        self.actual_dims = actual_dims


class CoordinateValidationError(ClimateIndicesError):
    """Raised when coordinate validation fails.

    This exception is raised when coordinate values are invalid, missing,
    or don't meet the requirements for index computation.

    Attributes:
        coordinate_name: Name of the coordinate that failed validation
        reason: Specific reason why the coordinate is invalid
    """

    def __init__(
        self,
        message: str,
        coordinate_name: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.coordinate_name = coordinate_name
        self.reason = reason


class InputTypeError(ClimateIndicesError):
    """Raised when input has incorrect type.

    This exception is raised when an argument has the wrong type and
    cannot be coerced to the expected type.

    Attributes:
        expected_type: The type that was expected
        actual_type: The actual type received
    """

    def __init__(
        self,
        message: str,
        expected_type: type | None = None,
        actual_type: type | None = None,
    ) -> None:
        super().__init__(message)
        self.expected_type = expected_type
        self.actual_type = actual_type


class InvalidArgumentError(ClimateIndicesError):
    """Raised when an argument value is outside the valid range or set.

    This exception is raised when function arguments have the correct type
    but invalid values (e.g., scale=0 when valid range is [1, 72]).

    Attributes:
        argument_name: Name of the argument that failed validation
        argument_value: The invalid value that was provided (stringified)
        valid_values: Human-readable description of valid range or set
    """

    def __init__(
        self,
        message: str,
        *,
        argument_name: str | None = None,
        argument_value: str | None = None,
        valid_values: str | None = None,
    ) -> None:
        super().__init__(message)
        self.argument_name = argument_name
        self.argument_value = argument_value
        self.valid_values = valid_values


# Warning Classes


class ClimateIndicesWarning(UserWarning):
    """Base warning for all climate_indices library data quality warnings.

    Users can filter all library warnings by catching this category:
        warnings.filterwarnings("ignore", category=ClimateIndicesWarning)
    """

    pass


class MissingDataWarning(ClimateIndicesWarning):
    """Warning issued when calibration period has excessive missing data.

    Raised when the proportion of missing values exceeds the configured
    threshold (default 20%). High missing data rates may reduce the
    reliability of distribution fitting and index calculations.

    Attributes:
        missing_ratio: The actual proportion of missing values (0.0 to 1.0)
        threshold: The configured threshold for triggering this warning
    """

    def __init__(
        self,
        message: str,
        *,
        missing_ratio: float | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(message)
        self.missing_ratio = missing_ratio
        self.threshold = threshold


class ShortCalibrationWarning(ClimateIndicesWarning):
    """Warning issued when calibration period is shorter than recommended.

    Raised when the calibration period length falls below the minimum
    recommended duration (default 30 years). Shorter calibration periods
    may not capture the full range of climate variability, affecting
    distribution parameter stability.

    Attributes:
        actual_years: The actual length of the calibration period in years
        required_years: The minimum recommended calibration period length
    """

    def __init__(
        self,
        message: str,
        *,
        actual_years: int | None = None,
        required_years: int | None = None,
    ) -> None:
        super().__init__(message)
        self.actual_years = actual_years
        self.required_years = required_years


class GoodnessOfFitWarning(ClimateIndicesWarning):
    """Warning issued when distribution fit quality is poor.

    Raised when Kolmogorov-Smirnov goodness-of-fit tests indicate that
    the fitted distribution does not adequately represent the empirical
    data. This aggregates poor fits across multiple time steps to avoid
    warning floods.

    Attributes:
        distribution_name: Name of the distribution that fits poorly
        p_value: Typical p-value from KS tests (if single step) or None
        threshold: The p-value threshold used for determining poor fit
        poor_fit_count: Number of time steps with poor fit
        total_steps: Total number of time steps evaluated
    """

    def __init__(
        self,
        message: str,
        *,
        distribution_name: str | None = None,
        p_value: float | None = None,
        threshold: float | None = None,
        poor_fit_count: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        super().__init__(message)
        self.distribution_name = distribution_name
        self.p_value = p_value
        self.threshold = threshold
        self.poor_fit_count = poor_fit_count
        self.total_steps = total_steps


class InputAlignmentWarning(ClimateIndicesWarning):
    """Warning issued when input alignment drops data points.

    Raised when aligning multiple input DataArrays (e.g., precipitation and
    PET for SPEI) results in dropped time steps due to non-overlapping
    time coordinates. This warns the user that the computation will use
    only the intersection of input time ranges.

    Attributes:
        original_size: Number of time steps in the primary input before alignment
        aligned_size: Number of time steps after alignment (intersection size)
        dropped_count: Number of time steps dropped during alignment
    """

    def __init__(
        self,
        message: str,
        *,
        original_size: int | None = None,
        aligned_size: int | None = None,
        dropped_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.original_size = original_size
        self.aligned_size = aligned_size
        self.dropped_count = dropped_count

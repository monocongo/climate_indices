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

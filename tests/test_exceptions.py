"""Tests for the custom exception hierarchy in climate_indices.exceptions."""

from __future__ import annotations

import pytest

from climate_indices import ClimateIndicesError
from climate_indices import compute
from climate_indices import exceptions


class TestExceptionHierarchy:
    """Verify the exception class hierarchy structure."""

    def test_all_inherit_from_base(self) -> None:
        """All custom exceptions should inherit from ClimateIndicesError."""
        assert issubclass(exceptions.DistributionFittingError, ClimateIndicesError)
        assert issubclass(exceptions.InsufficientDataError, ClimateIndicesError)
        assert issubclass(exceptions.PearsonFittingError, ClimateIndicesError)
        assert issubclass(exceptions.DimensionMismatchError, ClimateIndicesError)
        assert issubclass(exceptions.CoordinateValidationError, ClimateIndicesError)
        assert issubclass(exceptions.InputTypeError, ClimateIndicesError)

    def test_distribution_fitting_subtypes(self) -> None:
        """Distribution fitting errors should have correct parent classes."""
        assert issubclass(
            exceptions.InsufficientDataError, exceptions.DistributionFittingError
        )
        assert issubclass(
            exceptions.PearsonFittingError, exceptions.DistributionFittingError
        )

    def test_new_exceptions_not_under_distribution_fitting(self) -> None:
        """New exception types should be direct children of ClimateIndicesError."""
        assert issubclass(exceptions.DimensionMismatchError, ClimateIndicesError)
        assert not issubclass(
            exceptions.DimensionMismatchError, exceptions.DistributionFittingError
        )

        assert issubclass(exceptions.CoordinateValidationError, ClimateIndicesError)
        assert not issubclass(
            exceptions.CoordinateValidationError, exceptions.DistributionFittingError
        )

        assert issubclass(exceptions.InputTypeError, ClimateIndicesError)
        assert not issubclass(
            exceptions.InputTypeError, exceptions.DistributionFittingError
        )

    def test_base_inherits_from_exception(self) -> None:
        """ClimateIndicesError should inherit from Exception."""
        assert issubclass(ClimateIndicesError, Exception)


class TestExceptionCatchAll:
    """Verify that ClimateIndicesError can catch all library exceptions."""

    def test_catch_distribution_fitting_error(self) -> None:
        """ClimateIndicesError should catch DistributionFittingError."""
        with pytest.raises(ClimateIndicesError):
            raise exceptions.DistributionFittingError("test error")

    def test_catch_insufficient_data_error(self) -> None:
        """ClimateIndicesError should catch InsufficientDataError."""
        with pytest.raises(ClimateIndicesError):
            raise exceptions.InsufficientDataError("test error")

    def test_catch_pearson_fitting_error(self) -> None:
        """ClimateIndicesError should catch PearsonFittingError."""
        with pytest.raises(ClimateIndicesError):
            raise exceptions.PearsonFittingError("test error")

    def test_catch_dimension_mismatch_error(self) -> None:
        """ClimateIndicesError should catch DimensionMismatchError."""
        with pytest.raises(ClimateIndicesError):
            raise exceptions.DimensionMismatchError("test error")

    def test_catch_coordinate_validation_error(self) -> None:
        """ClimateIndicesError should catch CoordinateValidationError."""
        with pytest.raises(ClimateIndicesError):
            raise exceptions.CoordinateValidationError("test error")

    def test_catch_input_type_error(self) -> None:
        """ClimateIndicesError should catch InputTypeError."""
        with pytest.raises(ClimateIndicesError):
            raise exceptions.InputTypeError("test error")


class TestExceptionContextAttributes:
    """Verify that custom exception attributes are stored correctly."""

    def test_insufficient_data_error_attributes(self) -> None:
        """InsufficientDataError should store non_zero_count and required_count."""
        exc = exceptions.InsufficientDataError(
            "Not enough data", non_zero_count=5, required_count=10
        )
        assert exc.non_zero_count == 5
        assert exc.required_count == 10
        assert str(exc) == "Not enough data"

    def test_insufficient_data_error_defaults(self) -> None:
        """InsufficientDataError attributes should default to None."""
        exc = exceptions.InsufficientDataError("Not enough data")
        assert exc.non_zero_count is None
        assert exc.required_count is None

    def test_pearson_fitting_error_attributes(self) -> None:
        """PearsonFittingError should store underlying_error."""
        underlying = ValueError("original error")
        exc = exceptions.PearsonFittingError("Fitting failed", underlying_error=underlying)
        assert exc.underlying_error is underlying
        assert str(exc) == "Fitting failed"

    def test_pearson_fitting_error_defaults(self) -> None:
        """PearsonFittingError underlying_error should default to None."""
        exc = exceptions.PearsonFittingError("Fitting failed")
        assert exc.underlying_error is None

    def test_dimension_mismatch_error_attributes(self) -> None:
        """DimensionMismatchError should store expected_dims and actual_dims."""
        exc = exceptions.DimensionMismatchError(
            "Shape mismatch", expected_dims=(10, 20), actual_dims=(10, 15)
        )
        assert exc.expected_dims == (10, 20)
        assert exc.actual_dims == (10, 15)
        assert str(exc) == "Shape mismatch"

    def test_dimension_mismatch_error_defaults(self) -> None:
        """DimensionMismatchError attributes should default to None."""
        exc = exceptions.DimensionMismatchError("Shape mismatch")
        assert exc.expected_dims is None
        assert exc.actual_dims is None

    def test_coordinate_validation_error_attributes(self) -> None:
        """CoordinateValidationError should store coordinate_name and reason."""
        exc = exceptions.CoordinateValidationError(
            "Invalid coordinate",
            coordinate_name="time",
            reason="Non-monotonic values",
        )
        assert exc.coordinate_name == "time"
        assert exc.reason == "Non-monotonic values"
        assert str(exc) == "Invalid coordinate"

    def test_coordinate_validation_error_defaults(self) -> None:
        """CoordinateValidationError attributes should default to None."""
        exc = exceptions.CoordinateValidationError("Invalid coordinate")
        assert exc.coordinate_name is None
        assert exc.reason is None

    def test_input_type_error_attributes(self) -> None:
        """InputTypeError should store expected_type and actual_type."""
        exc = exceptions.InputTypeError(
            "Wrong type", expected_type=int, actual_type=str
        )
        assert exc.expected_type is int
        assert exc.actual_type is str
        assert str(exc) == "Wrong type"

    def test_input_type_error_defaults(self) -> None:
        """InputTypeError attributes should default to None."""
        exc = exceptions.InputTypeError("Wrong type")
        assert exc.expected_type is None
        assert exc.actual_type is None


class TestBackwardCompatibility:
    """Verify backward compatibility with imports from compute module."""

    def test_import_identity(self) -> None:
        """Exceptions imported from compute should be identical to exceptions module."""
        assert compute.DistributionFittingError is exceptions.DistributionFittingError
        assert compute.InsufficientDataError is exceptions.InsufficientDataError
        assert compute.PearsonFittingError is exceptions.PearsonFittingError

    def test_isinstance_across_imports(self) -> None:
        """isinstance checks should work across different import paths."""
        # create via compute module import
        exc = compute.InsufficientDataError("test", non_zero_count=3)

        # verify with exceptions module import
        assert isinstance(exc, exceptions.InsufficientDataError)
        assert isinstance(exc, exceptions.DistributionFittingError)
        assert isinstance(exc, ClimateIndicesError)

    def test_raise_and_catch_via_compute(self) -> None:
        """Exceptions raised via compute import should be catchable."""
        with pytest.raises(exceptions.InsufficientDataError):
            raise compute.InsufficientDataError("test error")

        with pytest.raises(ClimateIndicesError):
            raise compute.PearsonFittingError("test error")

    def test_existing_code_pattern(self) -> None:
        """Verify pattern used in existing tests continues to work."""
        # this is the pattern used in test_zero_precipitation_fix.py
        try:
            raise compute.InsufficientDataError(
                "Insufficient data", non_zero_count=5, required_count=10
            )
        except compute.InsufficientDataError as e:
            assert e.non_zero_count == 5
            assert e.required_count == 10

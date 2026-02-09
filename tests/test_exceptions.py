"""Tests for the custom exception hierarchy in climate_indices.exceptions."""

from __future__ import annotations

import pickle

import pytest

from climate_indices import ClimateIndicesError, compute, exceptions


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
        assert issubclass(exceptions.InvalidArgumentError, ClimateIndicesError)

    def test_distribution_fitting_subtypes(self) -> None:
        """Distribution fitting errors should have correct parent classes."""
        assert issubclass(exceptions.InsufficientDataError, exceptions.DistributionFittingError)
        assert issubclass(exceptions.PearsonFittingError, exceptions.DistributionFittingError)

    def test_new_exceptions_not_under_distribution_fitting(self) -> None:
        """New exception types should be direct children of ClimateIndicesError."""
        assert issubclass(exceptions.DimensionMismatchError, ClimateIndicesError)
        assert not issubclass(exceptions.DimensionMismatchError, exceptions.DistributionFittingError)

        assert issubclass(exceptions.CoordinateValidationError, ClimateIndicesError)
        assert not issubclass(exceptions.CoordinateValidationError, exceptions.DistributionFittingError)

        assert issubclass(exceptions.InputTypeError, ClimateIndicesError)
        assert not issubclass(exceptions.InputTypeError, exceptions.DistributionFittingError)

        assert issubclass(exceptions.InvalidArgumentError, ClimateIndicesError)
        assert not issubclass(exceptions.InvalidArgumentError, exceptions.DistributionFittingError)

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

    def test_catch_invalid_argument_error(self) -> None:
        """ClimateIndicesError should catch InvalidArgumentError."""
        with pytest.raises(ClimateIndicesError):
            raise exceptions.InvalidArgumentError("test error")


class TestExceptionContextAttributes:
    """Verify that custom exception attributes are stored correctly."""

    def test_insufficient_data_error_attributes(self) -> None:
        """InsufficientDataError should store non_zero_count and required_count."""
        exc = exceptions.InsufficientDataError("Not enough data", non_zero_count=5, required_count=10)
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
        exc = exceptions.DimensionMismatchError("Shape mismatch", expected_dims=(10, 20), actual_dims=(10, 15))
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
        exc = exceptions.InputTypeError("Wrong type", expected_type=int, actual_type=str)
        assert exc.expected_type is int
        assert exc.actual_type is str
        assert str(exc) == "Wrong type"

    def test_input_type_error_defaults(self) -> None:
        """InputTypeError attributes should default to None."""
        exc = exceptions.InputTypeError("Wrong type")
        assert exc.expected_type is None
        assert exc.actual_type is None

    def test_invalid_argument_error_attributes(self) -> None:
        """InvalidArgumentError should store argument_name, argument_value, and valid_values."""
        exc = exceptions.InvalidArgumentError(
            "Invalid scale",
            argument_name="scale",
            argument_value="0",
            valid_values="[1, 72]",
        )
        assert exc.argument_name == "scale"
        assert exc.argument_value == "0"
        assert exc.valid_values == "[1, 72]"
        assert str(exc) == "Invalid scale"

    def test_invalid_argument_error_defaults(self) -> None:
        """InvalidArgumentError attributes should default to None."""
        exc = exceptions.InvalidArgumentError("Invalid argument")
        assert exc.argument_name is None
        assert exc.argument_value is None
        assert exc.valid_values is None

    def test_distribution_fitting_error_attributes(self) -> None:
        """DistributionFittingError should store all structured attributes."""
        params = {"alpha": "0.5", "beta": "1.0"}
        underlying = ValueError("test error")
        exc = exceptions.DistributionFittingError(
            "Fitting failed",
            distribution_name="gamma",
            input_shape=(10, 12),
            parameters=params,
            suggestion="try pearson3",
            underlying_error=underlying,
        )
        assert exc.distribution_name == "gamma"
        assert exc.input_shape == (10, 12)
        assert exc.parameters == params
        assert exc.suggestion == "try pearson3"
        assert exc.underlying_error is underlying
        assert str(exc) == "Fitting failed"

    def test_distribution_fitting_error_defaults(self) -> None:
        """DistributionFittingError attributes should default to None."""
        exc = exceptions.DistributionFittingError("Fitting failed")
        assert exc.distribution_name is None
        assert exc.input_shape is None
        assert exc.parameters is None
        assert exc.suggestion is None
        assert exc.underlying_error is None


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
            raise compute.InsufficientDataError("Insufficient data", non_zero_count=5, required_count=10)
        except compute.InsufficientDataError as e:
            assert e.non_zero_count == 5
            assert e.required_count == 10


class TestExceptionPickling:
    """Verify exceptions can be pickled for Dask multiprocessing.

    Dask uses pickle to serialize exceptions across workers. All custom exceptions
    must be picklable to support distributed computation error reporting.
    """

    @pytest.mark.parametrize(
        "exception_class,init_args,init_kwargs",
        [
            (exceptions.ClimateIndicesError, ("base error",), {}),
            (exceptions.DistributionFittingError, ("fitting failed",), {}),
            (exceptions.InsufficientDataError, ("not enough data",), {"non_zero_count": 5, "required_count": 10}),
            (exceptions.PearsonFittingError, ("pearson failed",), {}),
            (
                exceptions.DimensionMismatchError,
                ("dims don't match",),
                {"expected_dims": (10, 20), "actual_dims": (10, 30)},
            ),
            (
                exceptions.CoordinateValidationError,
                ("bad coords",),
                {"coordinate_name": "time", "reason": "not monotonic"},
            ),
            (exceptions.InputTypeError, ("wrong type",), {"expected_type": type(None), "actual_type": type([])}),
            (
                exceptions.InvalidArgumentError,
                ("bad arg",),
                {"argument_name": "scale", "argument_value": "-1", "valid_values": "positive integers"},
            ),
        ],
    )
    def test_exception_pickle_roundtrip(self, exception_class, init_args, init_kwargs):
        """All exception classes should survive pickle roundtrip with attributes intact."""
        # create exception
        original = exception_class(*init_args, **init_kwargs)

        # pickle and unpickle
        pickled = pickle.dumps(original)
        restored = pickle.loads(pickled)

        # verify type preserved
        assert type(restored) is type(original)

        # verify message preserved
        assert str(restored) == str(original)

        # verify custom attributes preserved
        for attr_name, attr_value in init_kwargs.items():
            assert hasattr(restored, attr_name)
            assert getattr(restored, attr_name) == attr_value


class TestWarningPickling:
    """Verify warnings can be pickled for Dask multiprocessing."""

    @pytest.mark.parametrize(
        "warning_class,init_args,init_kwargs",
        [
            (exceptions.ClimateIndicesWarning, ("base warning",), {}),
            (exceptions.MissingDataWarning, ("missing data",), {"missing_ratio": 0.15, "threshold": 0.20}),
            (
                exceptions.ShortCalibrationWarning,
                ("short calibration",),
                {"actual_years": 25, "required_years": 30},
            ),
            (
                exceptions.GoodnessOfFitWarning,
                ("poor fit",),
                {"distribution_name": "gamma", "p_value": 0.03, "threshold": 0.05},
            ),
            (
                exceptions.InputAlignmentWarning,
                ("alignment needed",),
                {"original_size": 100, "aligned_size": 80, "dropped_count": 20},
            ),
        ],
    )
    def test_warning_pickle_roundtrip(self, warning_class, init_args, init_kwargs):
        """All warning classes should survive pickle roundtrip with attributes intact."""
        # create warning
        original = warning_class(*init_args, **init_kwargs)

        # pickle and unpickle
        pickled = pickle.dumps(original)
        restored = pickle.loads(pickled)

        # verify type preserved
        assert type(restored) is type(original)

        # verify message preserved
        assert str(restored) == str(original)

        # verify custom attributes preserved
        for attr_name, attr_value in init_kwargs.items():
            assert hasattr(restored, attr_name)
            assert getattr(restored, attr_name) == attr_value


class TestWarningAttributes:
    """Verify warning classes store context attributes correctly."""

    def test_input_alignment_warning_attributes(self):
        """InputAlignmentWarning should store alignment context."""
        warning = exceptions.InputAlignmentWarning(
            "Inputs aligned",
            original_size=100,
            aligned_size=80,
            dropped_count=20,
        )
        assert warning.original_size == 100
        assert warning.aligned_size == 80
        assert warning.dropped_count == 20
        assert str(warning) == "Inputs aligned"

    def test_input_alignment_warning_defaults(self):
        """InputAlignmentWarning attributes should default to None."""
        warning = exceptions.InputAlignmentWarning("Aligned")
        assert warning.original_size is None
        assert warning.aligned_size is None
        assert warning.dropped_count is None

    def test_missing_data_warning_attributes(self):
        """MissingDataWarning should store missing data ratios."""
        warning = exceptions.MissingDataWarning("Missing data", missing_ratio=0.15, threshold=0.20)
        assert warning.missing_ratio == 0.15
        assert warning.threshold == 0.20

    def test_short_calibration_warning_attributes(self):
        """ShortCalibrationWarning should store calibration period info."""
        warning = exceptions.ShortCalibrationWarning(
            "Short period",
            actual_years=25,
            required_years=30,
        )
        assert warning.actual_years == 25
        assert warning.required_years == 30

    def test_goodness_of_fit_warning_attributes(self):
        """GoodnessOfFitWarning should store fit statistic info."""
        warning = exceptions.GoodnessOfFitWarning(
            "Poor fit",
            distribution_name="gamma",
            p_value=0.03,
            threshold=0.05,
        )
        assert warning.distribution_name == "gamma"
        assert warning.p_value == 0.03
        assert warning.threshold == 0.05


class TestExceptionReprStr:
    """Verify repr and str produce useful output for debugging."""

    def test_base_exception_repr(self):
        """ClimateIndicesError repr should be informative."""
        exc = exceptions.ClimateIndicesError("Something went wrong")
        repr_str = repr(exc)
        assert "ClimateIndicesError" in repr_str
        assert "Something went wrong" in repr_str

    def test_insufficient_data_error_repr_with_attrs(self):
        """InsufficientDataError repr should include error message."""
        exc = exceptions.InsufficientDataError("Not enough data", non_zero_count=5, required_count=10)
        repr_str = repr(exc)
        assert "InsufficientDataError" in repr_str
        assert "Not enough data" in repr_str

    def test_dimension_mismatch_error_repr(self):
        """DimensionMismatchError repr should show dimensions."""
        exc = exceptions.DimensionMismatchError("Dimension mismatch", expected_dims=(10, 20), actual_dims=(10, 30))
        repr_str = repr(exc)
        assert "DimensionMismatchError" in repr_str

    def test_str_returns_message(self):
        """str(exception) should return the error message."""
        exc = exceptions.InvalidArgumentError("Invalid scale parameter")
        assert str(exc) == "Invalid scale parameter"

    def test_warning_repr(self):
        """Warning repr should be informative."""
        warning = exceptions.InputAlignmentWarning(
            "Aligned inputs",
            original_size=100,
            aligned_size=80,
            dropped_count=20,
        )
        repr_str = repr(warning)
        assert "InputAlignmentWarning" in repr_str

"""Tests for the public validation facade."""

from __future__ import annotations

import importlib
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import validation
from climate_indices.exceptions import CoordinateValidationError, DimensionMismatchError
from climate_indices.validation import (
    validate_dask_chunks,
    validate_time_dimension,
    validate_time_monotonicity,
)


def _standard_normal(size: int | tuple[int, ...]) -> np.ndarray:
    """Generate deterministic standard normal data for tests."""
    return np.random.default_rng(42).standard_normal(size)


class TestValidateTimeDimension:
    """Test validate_time_dimension() function."""

    def test_valid_time_dimension_passes(self, sample_monthly_precip_da):
        """Valid time dimension does not raise."""
        # should not raise
        validate_time_dimension(sample_monthly_precip_da, "time")

    def test_missing_dimension_raises(self, no_time_dim_da):
        """Missing time dimension raises DimensionMismatchError."""
        with pytest.raises(DimensionMismatchError) as exc_info:
            validate_time_dimension(no_time_dim_da, "time")

        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.coordinate_name == "time"
        assert exc_info.value.reason == "missing_dimension"
        assert exc_info.value.expected_dims == "time"
        assert exc_info.value.actual_dims == tuple(no_time_dim_da.dims)
        # existing handlers that catch the general type keep catching it
        assert isinstance(exc_info.value, CoordinateValidationError)

    def test_error_message_includes_available_dims(self, no_time_dim_da):
        """Error message lists available dimensions."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            validate_time_dimension(no_time_dim_da, "time")

        error_msg = str(exc_info.value)
        assert "['x', 'lat']" in error_msg

    def test_error_message_suggests_time_dim_parameter(self, no_time_dim_da):
        """Error message suggests using time_dim parameter."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            validate_time_dimension(no_time_dim_da, "time")

        assert "time_dim parameter" in str(exc_info.value)

    def test_custom_dimension_name_validated(self):
        """Validation works with custom time dimension name."""
        # create DataArray with 'date' instead of 'time'
        date = pd.date_range("2020-01-01", periods=12, freq="MS")
        da = xr.DataArray(
            _standard_normal(12),
            coords={"date": date},
            dims=["date"],
        )

        # should not raise when checking for 'date'
        validate_time_dimension(da, "date")

        # should raise when checking for 'time'
        with pytest.raises(CoordinateValidationError):
            validate_time_dimension(da, "time")

    def test_scalar_dataarray_raises(self):
        """Scalar DataArray (no dims) raises error."""
        scalar_da = xr.DataArray(42.0)

        with pytest.raises(CoordinateValidationError) as exc_info:
            validate_time_dimension(scalar_da, "time")

        # should show empty dims list
        assert "[]" in str(exc_info.value)

    def test_time_as_coordinate_but_not_dimension_raises(self):
        """Time as non-dimension coordinate is rejected."""
        # create DataArray where 'time' is a scalar coordinate, not a dimension
        da = xr.DataArray(
            _standard_normal(10),
            coords={"x": np.arange(10), "time": pd.Timestamp("2020-01-01")},
            dims=["x"],
        )

        with pytest.raises(CoordinateValidationError):
            validate_time_dimension(da, "time")

    def test_empty_dataarray_with_time_dim_passes(self):
        """Empty DataArray with time dim (0 elements) passes dimension check."""
        empty_da = xr.DataArray(
            [],
            coords={"time": pd.DatetimeIndex([])},
            dims=["time"],
        )

        # dimension exists, so validation passes
        validate_time_dimension(empty_da, "time")


class TestValidateTimeMonotonicity:
    """Test validate_time_monotonicity() function."""

    def test_monotonic_increasing_passes(self):
        """Monotonically increasing time coordinate passes."""
        time = pd.date_range("2020-01-01", periods=12, freq="MS")
        time_coord = xr.DataArray(time, dims=["time"])

        # should not raise
        validate_time_monotonicity(time_coord)

    def test_reversed_time_raises(self):
        """Reversed time coordinate raises CoordinateValidationError."""
        time = pd.date_range("2020-01-01", periods=12, freq="MS")[::-1]
        time_coord = xr.DataArray(time, dims=["time"])

        with pytest.raises(CoordinateValidationError) as exc_info:
            validate_time_monotonicity(time_coord)

        assert "not monotonically increasing" in str(exc_info.value).lower()
        assert exc_info.value.reason == "not_monotonic"

    def test_shuffled_time_raises(self, non_monotonic_time_da):
        """Shuffled time coordinate raises CoordinateValidationError."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            validate_time_monotonicity(non_monotonic_time_da["time"])

        assert "not monotonically increasing" in str(exc_info.value).lower()

    def test_duplicate_timestamps_pass(self):
        """Duplicate timestamps pass (is_monotonic_increasing allows duplicates)."""
        time = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-02-01", "2020-03-01"])
        time_coord = xr.DataArray(time, dims=["time"])

        # pandas is_monotonic_increasing allows duplicates (non-decreasing)
        # should not raise
        validate_time_monotonicity(time_coord)

    def test_error_suggests_sortby(self):
        """Error message suggests using sortby()."""
        time = pd.date_range("2020-01-01", periods=12, freq="MS")[::-1]
        time_coord = xr.DataArray(time, dims=["time"])

        with pytest.raises(CoordinateValidationError) as exc_info:
            validate_time_monotonicity(time_coord)

        assert "sortby" in str(exc_info.value)

    def test_single_element_passes(self):
        """Single-element time coordinate passes (trivially monotonic)."""
        time_coord = xr.DataArray([pd.Timestamp("2020-01-01")], dims=["time"])

        # should not raise
        validate_time_monotonicity(time_coord)

    def test_two_element_increasing_passes(self):
        """Two-element increasing time coordinate passes."""
        time = pd.to_datetime(["2020-01-01", "2020-02-01"])
        time_coord = xr.DataArray(time, dims=["time"])

        # should not raise
        validate_time_monotonicity(time_coord)

    def test_two_element_decreasing_raises(self):
        """Two-element decreasing time coordinate raises."""
        time = pd.to_datetime(["2020-02-01", "2020-01-01"])
        time_coord = xr.DataArray(time, dims=["time"])

        with pytest.raises(CoordinateValidationError):
            validate_time_monotonicity(time_coord)


class TestValidateDaskChunks:
    """Test validate_dask_chunks() validation function."""

    def test_single_time_chunk_passes(self, dask_monthly_precip_3d):
        """Single time chunk with spatial chunks passes validation."""
        # should not raise
        validate_dask_chunks(dask_monthly_precip_3d, "time")

    def test_multi_time_chunk_raises_error(self, dask_multi_time_chunk):
        """Multiple time chunks raises CoordinateValidationError."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            validate_dask_chunks(dask_multi_time_chunk, "time")

        assert exc_info.value.reason == "multi_chunked_time_dimension"
        assert "single chunk" in str(exc_info.value)
        assert "chunk({'time': -1})" in str(exc_info.value)

    def test_multi_chunked_level_dimension_event_names_dimension(self):
        """The structured event must name the validated dimension, not assume time."""
        data = xr.DataArray(np.zeros((4, 2)), dims=("level", "x")).chunk({"level": 2})
        mock_logger = mock.MagicMock()

        # mock.patch resolves the dotted target on the module, which is the
        # facade itself now that the validator lives there
        with (
            mock.patch("climate_indices.validation._log", return_value=mock_logger),
            pytest.raises(CoordinateValidationError),
        ):
            validate_dask_chunks(data, "level")

        mock_logger.error.assert_called_once_with(
            "multi_chunked_dimension", dim="level", num_chunks=2, chunk_sizes=(2, 2)
        )

    def test_missing_time_dim_skipped(self, no_time_dim_da):
        """Validation skipped when time dimension doesn't exist."""
        # convert to dask-backed
        dask_da = no_time_dim_da.chunk({"x": 2})

        # should not raise (time dim doesn't exist, validation skipped)
        validate_dask_chunks(dask_da, "time")


class TestValidationPublicSurface:
    """The facade's public surface stays exactly as declared."""

    def test_all_names_resolve(self):
        """Every name in __all__ resolves to a module attribute."""
        assert validation.__all__ == [
            "InputType",
            "detect_input_type",
            "validate_dask_chunks",
            "validate_time_dimension",
            "validate_time_monotonicity",
        ]
        for name in validation.__all__:
            assert getattr(validation, name) is not None

    def test_xarray_adapter_reexports_input_type(self):
        """The adapter keeps re-exporting the moved input-kind names."""
        adapter_module = importlib.import_module("climate_indices.xarray_adapter")

        assert adapter_module.InputType is validation.InputType
        assert adapter_module.detect_input_type is validation.detect_input_type

    def test_package_root_reexports_the_facade(self):
        """The package root exposes the facade module and its input-kind names."""
        import climate_indices

        assert "validation" in climate_indices.__all__
        assert climate_indices.InputType is validation.InputType
        assert climate_indices.detect_input_type is validation.detect_input_type

"""Tests for input type detection infrastructure (Story 2.1).

This test module validates the type classification logic that routes inputs
to either NumPy or xarray computation paths. It covers:
- NumPy-coercible types (ndarray, list, tuple, scalars, masked arrays)
- xarray DataArray variants (with coords, attrs, multiple dimensions)
- Unsupported types with appropriate error messages
- Error message quality (hints for pandas, polars, Dataset)
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from climate_indices.exceptions import ClimateIndicesError, InputTypeError
from climate_indices.xarray_adapter import InputType, detect_input_type


class TestDetectInputTypeNumpy:
    """Test detection of NumPy-coercible input types."""

    def test_ndarray_1d(self):
        """1D NumPy array is classified as NUMPY."""
        data = np.array([1.0, 2.0, 3.0])
        assert detect_input_type(data) == InputType.NUMPY

    def test_ndarray_2d(self):
        """2D NumPy array is classified as NUMPY."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert detect_input_type(data) == InputType.NUMPY

    def test_list(self):
        """Python list is classified as NUMPY."""
        data = [1.0, 2.0, 3.0]
        assert detect_input_type(data) == InputType.NUMPY

    def test_nested_list(self):
        """Nested list is classified as NUMPY."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        assert detect_input_type(data) == InputType.NUMPY

    def test_tuple(self):
        """Python tuple is classified as NUMPY."""
        data = (1.0, 2.0, 3.0)
        assert detect_input_type(data) == InputType.NUMPY

    def test_int(self):
        """Python int scalar is classified as NUMPY."""
        data = 42
        assert detect_input_type(data) == InputType.NUMPY

    def test_float(self):
        """Python float scalar is classified as NUMPY."""
        data = 3.14
        assert detect_input_type(data) == InputType.NUMPY

    def test_numpy_int_scalar(self):
        """NumPy integer scalar is classified as NUMPY."""
        data = np.int64(42)
        assert detect_input_type(data) == InputType.NUMPY

    def test_numpy_float_scalar(self):
        """NumPy float64 scalar is classified as NUMPY."""
        data = np.float64(3.14)
        assert detect_input_type(data) == InputType.NUMPY

    def test_numpy_float32_scalar(self):
        """NumPy float32 scalar is classified as NUMPY."""
        data = np.float32(3.14)
        assert detect_input_type(data) == InputType.NUMPY

    def test_masked_array(self):
        """NumPy masked array is classified as NUMPY."""
        data = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        assert detect_input_type(data) == InputType.NUMPY


class TestDetectInputTypeXarray:
    """Test detection of xarray DataArray inputs."""

    def test_simple_dataarray(self):
        """Basic DataArray with no coordinates is classified as XARRAY."""
        data = xr.DataArray([1.0, 2.0, 3.0])
        assert detect_input_type(data) == InputType.XARRAY

    def test_dataarray_with_coords(self):
        """DataArray with coordinates is classified as XARRAY."""
        data = xr.DataArray(
            [1.0, 2.0, 3.0],
            coords={"time": ["2020-01", "2020-02", "2020-03"]},
            dims=["time"],
        )
        assert detect_input_type(data) == InputType.XARRAY

    def test_dataarray_multidimensional(self):
        """Multidimensional DataArray is classified as XARRAY."""
        data = xr.DataArray(
            np.random.rand(10, 5),
            coords={"lat": range(10), "lon": range(5)},
            dims=["lat", "lon"],
        )
        assert detect_input_type(data) == InputType.XARRAY

    def test_dataarray_with_attrs(self):
        """DataArray with attributes is classified as XARRAY."""
        data = xr.DataArray(
            [1.0, 2.0, 3.0],
            attrs={"units": "mm", "long_name": "precipitation"},
        )
        assert detect_input_type(data) == InputType.XARRAY


class TestDetectInputTypeUnsupported:
    """Test rejection of unsupported input types."""

    def test_pandas_series(self):
        """pandas Series raises InputTypeError."""
        # pandas is available as transitive dependency via xarray
        import pandas as pd

        data = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(InputTypeError):
            detect_input_type(data)

    def test_pandas_dataframe(self):
        """pandas DataFrame raises InputTypeError."""
        import pandas as pd

        data = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        with pytest.raises(InputTypeError):
            detect_input_type(data)

    def test_string(self):
        """String input raises InputTypeError."""
        with pytest.raises(InputTypeError):
            detect_input_type("not a valid input")

    def test_dict(self):
        """Dictionary input raises InputTypeError."""
        with pytest.raises(InputTypeError):
            detect_input_type({"key": "value"})

    def test_none(self):
        """None input raises InputTypeError."""
        with pytest.raises(InputTypeError):
            detect_input_type(None)

    def test_xr_dataset(self):
        """xarray Dataset raises InputTypeError."""
        data = xr.Dataset({"temp": ([1.0, 2.0, 3.0])})
        with pytest.raises(InputTypeError):
            detect_input_type(data)

    def test_set(self):
        """Set input raises InputTypeError."""
        with pytest.raises(InputTypeError):
            detect_input_type({1, 2, 3})


class TestInputTypeErrorMessage:
    """Test quality of error messages for unsupported types."""

    def test_includes_actual_type(self):
        """Error message includes the actual type name."""
        with pytest.raises(InputTypeError) as exc_info:
            detect_input_type("string")
        # should include module.qualname format
        assert "str" in str(exc_info.value)

    def test_includes_expected_types(self):
        """Error message lists accepted types."""
        with pytest.raises(InputTypeError) as exc_info:
            detect_input_type("string")
        message = str(exc_info.value)
        # should mention key accepted types
        assert "np.ndarray" in message
        assert "xr.DataArray" in message

    def test_pandas_remediation(self):
        """Error for pandas types includes to_numpy() hint."""
        import pandas as pd

        data = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(InputTypeError) as exc_info:
            detect_input_type(data)
        # should suggest conversion method
        assert "to_numpy()" in str(exc_info.value)

    def test_polars_remediation(self):
        """Error for polars-like types includes to_numpy() hint."""

        # create a mock class that looks like polars without adding dependency
        class FakePolarsDataFrame:
            def to_numpy(self):
                return np.array([1.0, 2.0, 3.0])

        data = FakePolarsDataFrame()
        with pytest.raises(InputTypeError) as exc_info:
            detect_input_type(data)
        # should suggest conversion method for any type with to_numpy
        assert "to_numpy()" in str(exc_info.value)

    def test_error_attributes_set(self):
        """InputTypeError has expected_type and actual_type attributes."""
        with pytest.raises(InputTypeError) as exc_info:
            detect_input_type("string")
        assert exc_info.value.actual_type is str
        # expected_type is None since multiple types are accepted
        assert exc_info.value.expected_type is None

    def test_catchable_as_base(self):
        """InputTypeError is catchable as ClimateIndicesError."""
        with pytest.raises(ClimateIndicesError):
            detect_input_type("string")


class TestInputTypeEnum:
    """Test InputType enum properties."""

    def test_has_numpy(self):
        """InputType enum has NUMPY member."""
        assert hasattr(InputType, "NUMPY")

    def test_has_xarray(self):
        """InputType enum has XARRAY member."""
        assert hasattr(InputType, "XARRAY")

    def test_members_distinct(self):
        """NUMPY and XARRAY are distinct values."""
        assert InputType.NUMPY != InputType.XARRAY


class TestDatasetSpecificError:
    """Test Dataset-specific error messaging."""

    def test_xr_dataset_hints_variable_selection(self):
        """xarray Dataset error includes variable selection hint."""
        data = xr.Dataset({"temp": ([1.0, 2.0, 3.0])})
        with pytest.raises(InputTypeError) as exc_info:
            detect_input_type(data)
        message = str(exc_info.value)
        # should specifically mention Dataset and how to select a variable
        assert "Dataset" in message
        assert "variable_name" in message

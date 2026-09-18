"""Table-driven tests for input type detection (Story 2.1).

The tables are the single owner of the classifier contract: which inputs route
to the NumPy path, which route to the xarray path, and how unsupported types
are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices.exceptions import ClimateIndicesError, InputTypeError
from climate_indices.validation import InputType, detect_input_type


class FakePolarsDataFrame:
    """Looks like a polars frame without adding the dependency."""

    def to_numpy(self):
        return np.array([1.0, 2.0, 3.0])


NUMPY_INPUTS = [
    pytest.param(np.array([1.0, 2.0, 3.0]), id="ndarray-1d"),
    pytest.param(np.array([[1.0, 2.0], [3.0, 4.0]]), id="ndarray-2d"),
    pytest.param([1.0, 2.0, 3.0], id="list"),
    pytest.param([[1.0, 2.0], [3.0, 4.0]], id="nested-list"),
    pytest.param((1.0, 2.0, 3.0), id="tuple"),
    pytest.param(42, id="int"),
    pytest.param(3.14, id="float"),
    pytest.param(np.int64(42), id="numpy-int-scalar"),
    pytest.param(np.float64(3.14), id="numpy-float64-scalar"),
    pytest.param(np.float32(3.14), id="numpy-float32-scalar"),
    pytest.param(np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False]), id="masked-array"),
]

XARRAY_INPUTS = [
    pytest.param(xr.DataArray([1.0, 2.0, 3.0]), id="simple"),
    pytest.param(
        xr.DataArray([1.0, 2.0, 3.0], coords={"time": ["2020-01", "2020-02", "2020-03"]}, dims=["time"]),
        id="with-coords",
    ),
    pytest.param(
        xr.DataArray(np.random.rand(10, 5), coords={"lat": range(10), "lon": range(5)}, dims=["lat", "lon"]),
        id="multidimensional",
    ),
    pytest.param(
        xr.DataArray([1.0, 2.0, 3.0], attrs={"units": "mm", "long_name": "precipitation"}),
        id="with-attrs",
    ),
]

UNSUPPORTED_INPUTS = [
    pytest.param(pd.Series([1.0, 2.0, 3.0]), id="pandas-series"),
    pytest.param(pd.DataFrame({"col": [1.0, 2.0, 3.0]}), id="pandas-dataframe"),
    pytest.param("not a valid input", id="string"),
    pytest.param({"key": "value"}, id="dict"),
    pytest.param(None, id="none"),
    pytest.param(xr.Dataset({"temp": ([1.0, 2.0, 3.0])}), id="xarray-dataset"),
    pytest.param({1, 2, 3}, id="set"),
]

ERROR_MESSAGE_CASES = [
    pytest.param("string", ("str", "np.ndarray", "xr.DataArray"), id="string-lists-accepted-types"),
    pytest.param(pd.Series([1.0, 2.0, 3.0]), ("to_numpy()",), id="pandas-remediation"),
    pytest.param(FakePolarsDataFrame(), ("to_numpy()",), id="to-numpy-remediation"),
    pytest.param(xr.Dataset({"temp": ([1.0, 2.0, 3.0])}), ("Dataset", "variable_name"), id="dataset-hint"),
]

ERROR_ATTRIBUTE_CASES = [
    pytest.param("string", str, id="string"),
    pytest.param(None, type(None), id="none"),
    pytest.param({1, 2, 3}, set, id="set"),
]


@pytest.mark.parametrize("data", NUMPY_INPUTS)
def test_numpy_coercible_inputs_are_classified_numpy(data) -> None:
    """Anything NumPy can consume routes to the NumPy computation path."""
    assert detect_input_type(data) == InputType.NUMPY


@pytest.mark.parametrize("data", XARRAY_INPUTS)
def test_xarray_dataarrays_are_classified_xarray(data) -> None:
    """Every DataArray variant routes to the xarray computation path."""
    assert detect_input_type(data) == InputType.XARRAY


@pytest.mark.parametrize("data", UNSUPPORTED_INPUTS)
def test_unsupported_inputs_raise_input_type_error(data) -> None:
    """Types the classifier cannot route (including xr.Dataset) are rejected."""
    with pytest.raises(InputTypeError):
        detect_input_type(data)


@pytest.mark.parametrize(("data", "expected_fragments"), ERROR_MESSAGE_CASES)
def test_error_message_names_the_type_accepted_types_and_remediation(data, expected_fragments) -> None:
    """Rejections tell users what was passed, what is accepted, and how to convert."""
    with pytest.raises(InputTypeError) as exc_info:
        detect_input_type(data)
    message = str(exc_info.value)
    for fragment in expected_fragments:
        assert fragment in message


@pytest.mark.parametrize(("data", "expected_actual_type"), ERROR_ATTRIBUTE_CASES)
def test_error_attributes_and_base_catchability(data, expected_actual_type) -> None:
    """The error carries the actual type, no single expected type, and is catchable as the base."""
    with pytest.raises(ClimateIndicesError) as exc_info:
        detect_input_type(data)
    assert exc_info.value.actual_type is expected_actual_type
    # expected_type is None since multiple input types are accepted
    assert exc_info.value.expected_type is None


def test_input_type_enum_exposes_distinct_numpy_and_xarray_members() -> None:
    """The routing enum names the two supported paths."""
    assert hasattr(InputType, "NUMPY")
    assert hasattr(InputType, "XARRAY")
    assert InputType.NUMPY != InputType.XARRAY

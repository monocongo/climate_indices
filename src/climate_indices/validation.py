"""Validation façade for the climate index entry surfaces.

Owns the input-kind detection and xarray/Dask validation checks shared by the
xarray adapter, the fire adapters, and the typed public API, so callers depend
on a validation contract rather than on private names inside
:mod:`climate_indices.xarray_adapter`. It also owns the dataset-layout
classifier the CLI (:mod:`climate_indices.__main__`) validates its NetCDF
inputs against, so layout and its accepted dimension orders have one owner.
"""

from collections.abc import Hashable
from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd
import structlog.stdlib
import xarray as xr

from climate_indices.exceptions import CoordinateValidationError, DimensionMismatchError, InputTypeError
from climate_indices.logging_config import get_logger

__all__ = [
    "DatasetLayout",
    "InputType",
    "detect_dataset_layout",
    "detect_input_type",
    "expected_dimensions",
    "validate_dask_chunks",
    "validate_time_dimension",
    "validate_time_monotonicity",
]


def _log() -> structlog.stdlib.BoundLogger:
    """Return a logger resolved at call time.

    Tests reset structlog globals between cases. Resolving lazily avoids
    holding a stale logger that bypasses stdlib handlers/capture after reset.
    """
    return get_logger(__name__)


# types that can be safely coerced to np.ndarray by the existing numpy functions
# includes scalar types that numpy operations naturally handle
_NUMPY_COERCIBLE_TYPES = (
    np.ndarray,
    list,
    tuple,
    int,
    float,
    np.integer,
    np.floating,
)


class InputType(Enum):
    """Classification of input data types for routing.

    Used by detect_input_type() to determine which computation path to use.

    .. note:: Part of the beta xarray API layer. See :doc:`xarray_migration`.

    Attributes:
        NUMPY: Input is NumPy-coercible (ndarray, list, tuple, scalars)
        XARRAY: Input is xarray.DataArray
    """

    NUMPY = auto()
    XARRAY = auto()


def detect_input_type(data: Any) -> InputType:
    """Classify input data type for routing to appropriate computation path.

    This is a pure classifier—it determines the type category but does not
    perform any data transformation or coercion. The actual dispatch logic
    is handled by the @xarray_adapter decorator.

    .. note:: Part of the beta xarray API layer. See :doc:`xarray_migration`.

    Args:
        data: Input data to classify

    Returns:
        InputType.NUMPY for NumPy-coercible inputs (ndarray, list, tuple, scalars)
        InputType.XARRAY for xarray.DataArray inputs

    Raises:
        InputTypeError: If data type is not supported, with remediation hints for
            common types like pandas Series/DataFrame and polars DataFrame

    Notes:
        - np.ma.MaskedArray is a subclass of np.ndarray, so it's automatically accepted
        - Dask-backed xr.DataArray is still classified as XARRAY
        - bool is a subclass of int in Python, so True/False are classified as NUMPY
        - xr.Dataset is rejected with a hint to select a specific variable
    """
    # check xarray first since it's the new capability
    if isinstance(data, xr.DataArray):
        return InputType.XARRAY

    # check numpy-coercible types
    if isinstance(data, _NUMPY_COERCIBLE_TYPES):
        return InputType.NUMPY

    # unsupported type - provide helpful error message
    actual_type = type(data)
    type_name = f"{actual_type.__module__}.{actual_type.__qualname__}"

    # build remediation hints
    hints = []

    # check for common data science types
    if hasattr(data, "to_numpy"):
        # pandas Series/DataFrame, polars DataFrame
        hints.append("Convert using data.to_numpy()")

    # special case for xarray Dataset
    if isinstance(data, xr.Dataset):
        hints.append("xr.Dataset detected: Use ds['variable_name'] to select a DataArray")

    # build error message
    accepted = "np.ndarray, list, tuple, int, float, np.integer, np.floating, xr.DataArray"
    message = f"Unsupported input type: {type_name}. Accepted types: {accepted}."

    if hints:
        message += " " + " ".join(hints)

    raise InputTypeError(
        message=message,
        expected_type=None,  # multiple types accepted
        actual_type=actual_type,
    )


class DatasetLayout(Enum):
    """Storage layout a NetCDF dataset's data variables follow.

    Used by detect_dataset_layout() to classify how a dataset is laid out, as
    opposed to InputType, which classifies the array backend a computation's
    inputs arrive in.

    Attributes:
        GRID: Data variables on a latitude/longitude grid.
        DIVISIONS: Data variables on US climate divisions.
        TIMESERIES: Data variables on a single time series.
    """

    GRID = auto()
    DIVISIONS = auto()
    TIMESERIES = auto()


# the dimensions each layout's time-carrying variables may use, in storage order
_LAYOUT_DIMENSIONS: dict[DatasetLayout, tuple[tuple[Hashable, ...], ...]] = {
    DatasetLayout.GRID: (("lat", "lon", "time"), ("time", "lat", "lon")),
    DatasetLayout.DIVISIONS: (("time", "division"), ("division", "time")),
    DatasetLayout.TIMESERIES: (("time",),),
}

# the dimensions a layout's per-location companions may use; these are fixed
# per location, so they carry no time dimension
_LAYOUT_DIMENSIONS_WITHOUT_TIME: dict[DatasetLayout, tuple[tuple[Hashable, ...], ...]] = {
    DatasetLayout.GRID: (("lat", "lon"),),
    DatasetLayout.DIVISIONS: (("division",),),
}


def expected_dimensions(
    layout: DatasetLayout,
    *,
    includes_time: bool = True,
) -> tuple[tuple[Hashable, ...], ...] | None:
    """Return the dimension orders a data variable may use in a dataset layout.

    Args:
        layout: Dataset layout to report the accepted dimension orders of.
        includes_time: Whether the variable carries the layout's time
            dimension, as data variables (precipitation, temperature, PET) do
            and per-location companions (available water capacity) do not.

    Returns:
        The accepted dimension orders, in storage order, or None for a layout
        that has no such form -- a time series has no per-location companion
        variables.
    """
    dimensions = _LAYOUT_DIMENSIONS if includes_time else _LAYOUT_DIMENSIONS_WITHOUT_TIME
    return dimensions.get(layout)


def detect_dataset_layout(dimensions: tuple[Hashable, ...], variable: str) -> DatasetLayout:
    """Classify a data variable's dimensions as one of the dataset layouts.

    Args:
        dimensions: Data variable's dimensions, in storage order.
        variable: Data variable's label, used in the error message.

    Returns:
        The dataset layout the dimensions describe.

    Raises:
        ValueError: If the dimensions are not one of the supported forms.
    """
    for layout, expected in _LAYOUT_DIMENSIONS.items():
        if dimensions in expected:
            return layout

    # a list, so the accepted forms read the same in the message as they did
    # when each layout's dimensions were their own module constant
    accepted = list(_LAYOUT_DIMENSIONS[DatasetLayout.GRID] + _LAYOUT_DIMENSIONS[DatasetLayout.DIVISIONS])
    msg = f"Invalid dimensions of the {variable} variable: {dimensions}\nValid dimension names and order: {accepted}"
    _log().error(
        "dataset_layout_unrecognized",
        variable=variable,
        dimensions=dimensions,
        accepted=accepted,
    )
    raise ValueError(msg)


def validate_time_dimension(data: xr.DataArray, time_dim: str) -> None:
    """Validate that the time dimension exists in the input DataArray.

    Args:
        data: Input DataArray to validate
        time_dim: Name of the expected time dimension

    Raises:
        DimensionMismatchError: If the time dimension is not found
    """
    if time_dim not in data.dims:
        available_dims = list(data.dims)
        error_msg = (
            f"Time dimension '{time_dim}' not found in input. "
            f"Available dimensions: {available_dims}. "
            f"Use time_dim parameter to specify custom name."
        )
        _log().error(
            "time_dimension_missing",
            time_dim=time_dim,
            available_dims=available_dims,
            data_shape=data.shape,
        )
        raise DimensionMismatchError(
            message=error_msg,
            expected_dims=time_dim,
            actual_dims=tuple(data.dims),
            coordinate_name=time_dim,
            reason="missing_dimension",
        )


def validate_time_monotonicity(time_coord: xr.DataArray) -> None:
    """Validate that the time coordinate is monotonically increasing.

    Args:
        time_coord: Time coordinate DataArray to validate

    Raises:
        CoordinateValidationError: If the time coordinate is not monotonically increasing
    """
    is_monotonic = _is_time_coord_monotonic(time_coord)
    if is_monotonic:
        return

    dim_name = str(time_coord.dims[0]) if time_coord.dims else "time"
    error_msg = _build_non_monotonic_message(time_coord, dim_name)

    _log().error(
        "time_coordinate_not_monotonic",
        coordinate_name=dim_name,
        coordinate_length=len(time_coord),
    )
    raise CoordinateValidationError(
        message=error_msg,
        coordinate_name=str(dim_name),
        reason="not_monotonic",
    )


def _is_time_coord_monotonic(time_coord: xr.DataArray) -> bool:
    """Return True if the time coordinate is monotonically increasing."""
    try:
        time_index = pd.DatetimeIndex(time_coord.values)
        return bool(time_index.is_monotonic_increasing)
    except (TypeError, ValueError):
        return _is_nonstandard_time_coord_monotonic(time_coord)


def _is_nonstandard_time_coord_monotonic(time_coord: xr.DataArray) -> bool:
    """Fallback monotonicity check for non-standard/cftime coordinates."""
    time_values = time_coord.values
    if len(time_values) < 2:
        return True

    try:
        diffs = np.diff(time_values.astype("datetime64[ns]").astype(np.int64))
        return bool(np.all(diffs > 0))
    except (TypeError, ValueError):
        coord_name = str(time_coord.name) if time_coord.name is not None else "time"
        raise CoordinateValidationError(
            message=f"Cannot validate time coordinate monotonicity: unsupported datetime type {type(time_coord.values[0])}",
            coordinate_name=coord_name,
            reason="unsupported_datetime_type",
        ) from None


def _build_non_monotonic_message(time_coord: xr.DataArray, dim_name: str) -> str:
    """Build a detailed error message for non-monotonic time coordinates."""
    generic_msg = (
        f"Time coordinate is not monotonically increasing. "
        f"Sort the data using data.sortby('{dim_name}') before processing."
    )
    try:
        has_nat = pd.isna(time_coord.values).any()
    except (TypeError, ValueError):
        return generic_msg

    if has_nat:
        return (
            "Time coordinate is not monotonically increasing. "
            "Found NaT (Not-a-Time) or NaN values. "
            "Remove invalid timestamps before processing."
        )

    return generic_msg


def validate_dask_chunks(data: xr.DataArray, dim: str) -> None:
    """Validate that ``dim`` is not split across multiple Dask chunks.

    Distribution fitting and stateful recurrences require the full time
    series; HDW's level-maximum reduction requires the full vertical profile.
    Other dimensions can be arbitrarily chunked for parallel computation.
    This shared helper is the chunking guard for the decorated adapter path
    and the stateful fire adapters; the hand-written PET entry points opt
    into Dask rechunking instead.

    Args:
        data: Dask-backed DataArray to validate
        dim: Name of the dimension that must be a single chunk

    Raises:
        CoordinateValidationError: If dim is split across multiple chunks,
            with a message including the exact rechunking command to fix it
    """
    # skip validation if the dimension doesn't exist (already validated elsewhere)
    if dim not in data.dims:
        return

    # skip validation if not chunked (shouldn't happen since we call this after is_dask check)
    if data.chunks is None:
        return

    # get chunks for the dimension
    # data.chunks is a tuple-of-tuples indexed by dimension position
    dim_chunks = data.chunks[data.dims.index(dim)]

    # validate single chunk on the dimension
    if len(dim_chunks) > 1:
        error_msg = (
            f"Dimension '{dim}' is split across {len(dim_chunks)} chunks. "
            "Climate index computation requires this dimension in a single chunk. "
            f"Rechunk using: data = data.chunk({{'{dim}': -1}})"
        )
        _log().error(
            # the event names the validated dimension neutrally; the stable
            # reason= code below keeps its historical time-axis spelling
            "multi_chunked_dimension",
            dim=dim,
            num_chunks=len(dim_chunks),
            chunk_sizes=dim_chunks,
        )
        raise CoordinateValidationError(
            message=error_msg,
            coordinate_name=dim,
            reason="multi_chunked_time_dimension",
        )

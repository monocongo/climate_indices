"""Xarray adapter layer for climate indices computation.

This module provides type detection and routing infrastructure to enable transparent
dispatch between NumPy array and xarray DataArray inputs. It implements Architecture
Decisions 1 (Wrapper Approach) and 2 (Decorator Pattern) from Epic 2.

The design philosophy:
- Existing NumPy functions in indices.py remain unchanged
- Type detection is purely classification—no coercion or data transformation
- Unsupported types receive clear, actionable error messages
- The adapter layer is isolated in this module for maintainability

References:
    Architecture Decision 1: Wrapper Approach (NumPy core + xarray adapter)
    Architecture Decision 2: Decorator Pattern (@xarray_adapter)
"""

from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import compute
from climate_indices.exceptions import CoordinateValidationError, InputTypeError
from climate_indices.logging_config import get_logger

logger = get_logger(__name__)


class _CFAttributesRequired(TypedDict):
    """Required CF Convention metadata attributes."""

    long_name: str
    units: str
    references: str


class CFAttributes(_CFAttributesRequired, total=False):
    """CF Convention metadata attributes for a climate index.

    Required keys: long_name, units, references.
    Optional keys: standard_name (only when officially defined in CF conventions).
    """

    standard_name: str


CF_METADATA: dict[str, CFAttributes] = {
    "spi": {
        "long_name": "Standardized Precipitation Index",
        "units": "dimensionless",
        "references": (
            "McKee, T. B., Doesken, N. J., & Kleist, J. (1993). "
            "The relationship of drought frequency and duration to time scales. "
            "Proceedings of the 8th Conference on Applied Climatology, "
            "17-22 January, Anaheim, CA. "
            "American Meteorological Society, Boston, MA, 179-184."
        ),
    },
}

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
    is handled by the @xarray_adapter decorator (Story 2.2).

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
        - Dask-backed xr.DataArray is still classified as XARRAY (Dask handling is Story 2.9)
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


def _infer_data_start_year(time_coord: xr.DataArray) -> int:
    """Extract the starting year from a time coordinate.

    Args:
        time_coord: xarray DataArray containing datetime values

    Returns:
        Year of the first timestamp in the coordinate

    Raises:
        CoordinateValidationError: If time coordinate is empty or not datetime-like
    """
    if len(time_coord) == 0:
        raise CoordinateValidationError(
            message="Time coordinate is empty - cannot infer data_start_year",
            coordinate_name="time",
            reason="empty coordinate",
        )

    try:
        first_timestamp = pd.Timestamp(time_coord.values[0])
        return int(first_timestamp.year)
    except (TypeError, ValueError) as e:
        raise CoordinateValidationError(
            message=f"Time coordinate must be datetime-like to infer data_start_year: {e}",
            coordinate_name="time",
            reason="not datetime-like",
        ) from e


def _infer_periodicity(time_coord: xr.DataArray) -> compute.Periodicity:
    """Infer periodicity from time coordinate frequency.

    Args:
        time_coord: xarray DataArray containing datetime values

    Returns:
        Periodicity.monthly for month-start/end frequencies
        Periodicity.daily for daily frequency

    Raises:
        CoordinateValidationError: If frequency cannot be inferred or is unsupported
    """
    # xarray's infer_freq requires at least 3 values
    if len(time_coord) < 3:
        raise CoordinateValidationError(
            message="Time coordinate must have at least 3 values to infer periodicity",
            coordinate_name="time",
            reason="insufficient data points",
        )

    freq = xr.infer_freq(time_coord)

    if freq is None:
        raise CoordinateValidationError(
            message="Could not infer frequency from time coordinate - ensure regular spacing",
            coordinate_name="time",
            reason="irregular frequency",
        )

    # map pandas frequency strings to Periodicity
    # MS = month start, ME = month end, M = legacy month end
    if freq in ("MS", "ME", "M"):
        return compute.Periodicity.monthly
    elif freq == "D":
        return compute.Periodicity.daily
    else:
        raise CoordinateValidationError(
            message=f"Unsupported frequency '{freq}' - only 'MS'/'ME'/'M' (monthly) and 'D' (daily) supported",
            coordinate_name="time",
            reason=f"unsupported frequency: {freq}",
        )


def _infer_calibration_period(time_coord: xr.DataArray) -> tuple[int, int]:
    """Infer calibration period from time coordinate endpoints.

    Args:
        time_coord: xarray DataArray containing datetime values

    Returns:
        Tuple of (first_year, last_year) covering the full time range
    """
    first_year = pd.Timestamp(time_coord.values[0]).year
    last_year = pd.Timestamp(time_coord.values[-1]).year
    return (first_year, last_year)


def _serialize_attr_value(value: Any) -> str | int | float | bool:
    """Serialize attribute values for xarray compatibility.

    Converts enum instances to their string name representation, passes through
    native xarray-serializable types, and raises TypeError for non-serializable types.

    Args:
        value: Attribute value to serialize

    Returns:
        Serialized value: enum→name string, or passthrough for str/int/float/bool/numpy scalars

    Raises:
        TypeError: If value is not serializable (dict, list, complex objects)

    Examples:
        >>> _serialize_attr_value(Distribution.gamma)
        'gamma'
        >>> _serialize_attr_value(42)
        42
        >>> _serialize_attr_value("monthly")
        'monthly'
        >>> _serialize_attr_value([1, 2])
        TypeError: ...
    """
    # enum → .name string
    if isinstance(value, Enum):
        return value.name

    # numpy scalar → python scalar
    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    # passthrough native serializable types
    if isinstance(value, (str, int, float, bool)):
        return value

    # reject non-serializable types
    raise TypeError(
        f"Cannot serialize attribute value of type {type(value).__name__}. "
        f"Supported types: Enum, str, int, float, bool, numpy scalars."
    )


def _build_output_dataarray(
    input_da: xr.DataArray,
    result_values: np.ndarray[Any, Any],
    cf_metadata: dict[str, str] | None = None,
    calculation_metadata: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Build output DataArray with preserved coordinates and metadata.

    This function implements the rewrap phase of the adapter contract, ensuring
    all coordinates (dimension, non-dimension, and scalar) and their attributes
    survive the extract→compute→rewrap pipeline.

    Args:
        input_da: Original input DataArray with full coordinate metadata
        result_values: NumPy result array from index computation
        cf_metadata: Optional CF Convention metadata to apply to DataArray-level
            attributes. Overrides conflicting input attrs but never affects
            coordinate-level attributes.
        calculation_metadata: Optional dict of calculation-specific metadata
            (e.g., scale, distribution) to add to DataArray attributes.
            Enum values are automatically serialized to .name strings.

    Returns:
        DataArray with result_values and preserved coordinates/dims/attrs

    Notes:
        - Deep-copies coordinate attrs to prevent mutation bleed-through
        - Preserves coordinate ordering (dict insertion order)
        - CF metadata only affects DataArray-level attrs, not coord attrs
        - Preserves the input DataArray's .name attribute
        - Attribute layering: input attrs → CF metadata → calculation metadata → version
    """
    # deep-copy DA-level attrs for output (prevents mutation)
    output_attrs = copy.deepcopy(input_da.attrs)

    # apply CF metadata overrides to DA-level attrs only
    if cf_metadata is not None:
        output_attrs.update(cf_metadata)

    # add calculation metadata (e.g., scale, distribution)
    if calculation_metadata is not None:
        for key, value in calculation_metadata.items():
            try:
                output_attrs[key] = _serialize_attr_value(value)
            except TypeError as e:
                logger.warning(
                    "calculation_metadata_serialization_failed",
                    key=key,
                    value_type=type(value).__name__,
                    error=str(e),
                )

    # add library version for provenance
    # deferred import to avoid circular dependency (__init__.py imports this module)
    from climate_indices import __version__

    output_attrs["climate_indices_version"] = __version__

    # construct output with coords and dims from input
    result_da = xr.DataArray(
        result_values,
        coords=input_da.coords,
        dims=input_da.dims,
        attrs=output_attrs,
        name=input_da.name,
    )

    # deep-copy coordinate attrs to guard against upstream behavior changes
    # xarray 2025.6.1 preserves coord attrs through DataArray(coords=...),
    # but we defensively copy to ensure isolation
    for coord_name in result_da.coords:
        if coord_name in input_da.coords:
            result_da.coords[coord_name].attrs = copy.deepcopy(input_da.coords[coord_name].attrs)

    return result_da


def xarray_adapter(
    *,
    cf_metadata: dict[str, str] | None = None,
    time_dim: str = "time",
    infer_params: bool = True,
    calculation_metadata_keys: list[str] | tuple[str, ...] | None = None,
) -> Callable[[Callable[..., np.ndarray[Any, Any]]], Callable[..., np.ndarray[Any, Any] | xr.DataArray]]:
    """Decorator factory that adapts NumPy index functions to accept xarray DataArrays.

    This decorator implements the adapter contract: extract → infer → compute → rewrap → log.
    It transparently handles both NumPy arrays (passthrough) and xarray DataArrays (extract,
    compute with NumPy function, rewrap result).

    Args:
        cf_metadata: Optional dict of CF Convention metadata to apply to output DataArray.
            Keys should be CF attribute names (e.g., 'standard_name', 'long_name', 'units').
            These override conflicting attributes from the input DataArray.
        time_dim: Name of the time dimension in the input DataArray (default: "time").
            Used for parameter inference.
        infer_params: If True, automatically infer missing parameters (data_start_year,
            periodicity, calibration_year_initial, calibration_year_final) from the time
            coordinate. Explicit parameter values always override inferred values.
        calculation_metadata_keys: Optional sequence of parameter names to capture as
            output metadata attributes. For example, ["scale", "distribution"] will
            add these kwargs to the output DataArray.attrs. Enum values are automatically
            serialized to their .name string representation.

    Returns:
        Decorator function that wraps index computation functions

    Example:
        >>> @xarray_adapter(
        ...     cf_metadata={'standard_name': 'spi', 'units': '1'},
        ...     calculation_metadata_keys=['scale', 'distribution']
        ... )
        ... def spi(values, scale, distribution, data_start_year, ...):
        ...     # existing NumPy implementation
        ...     return numpy_result
        ...
        >>> # Works with both NumPy arrays and xarray DataArrays
        >>> result_numpy = spi(np.array([...]), scale=3, ...)
        >>> result_xarray = spi(precip_da, scale=3, distribution=Distribution.gamma)
        >>> # result_xarray.attrs now includes: scale=3, distribution="gamma", climate_indices_version="x.y.z"

    Notes:
        - NumPy inputs: Passed through unchanged to the wrapped function
        - xarray inputs: Values extracted, parameters inferred, result rewrapped with coords
        - 1D DataArrays only in Story 2.2; multi-dimensional support added in Story 2.9
        - Uses inspect.signature() for generic parameter mapping (works with any function)
    """

    def decorator(func: Callable[..., np.ndarray[Any, Any]]) -> Callable[..., np.ndarray[Any, Any] | xr.DataArray]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any] | xr.DataArray:
            # first positional argument is always the data
            if not args:
                raise ValueError(f"{func.__name__} requires at least one positional argument (data)")

            data = args[0]
            input_type = detect_input_type(data)

            # numpy passthrough - no transformation needed
            if input_type == InputType.NUMPY:
                return func(*args, **kwargs)

            # xarray path: extract → infer → compute → rewrap → log
            input_da = data

            # extract numpy values
            numpy_values = input_da.values

            # build kwargs for the wrapped function
            # start with explicitly provided kwargs
            call_kwargs = dict(kwargs)

            # infer missing parameters if enabled and time dimension exists
            if infer_params and time_dim in input_da.dims:
                time_coord = input_da[time_dim]

                # use inspect to determine which parameters the function accepts
                sig = inspect.signature(func)

                # bind provided args/kwargs to see what's already specified
                try:
                    # bind_partial allows missing parameters (we'll fill them)
                    bound = sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    provided_params = set(bound.arguments.keys())
                except TypeError:
                    # if binding fails, skip inference
                    provided_params = set()

                # infer data_start_year if not provided
                if "data_start_year" in sig.parameters and "data_start_year" not in provided_params:
                    call_kwargs["data_start_year"] = _infer_data_start_year(time_coord)

                # infer periodicity if not provided
                if "periodicity" in sig.parameters and "periodicity" not in provided_params:
                    call_kwargs["periodicity"] = _infer_periodicity(time_coord)

                # infer calibration period if not provided
                if "calibration_year_initial" in sig.parameters and "calibration_year_initial" not in provided_params:
                    cal_start, cal_end = _infer_calibration_period(time_coord)
                    if "calibration_year_initial" not in provided_params:
                        call_kwargs["calibration_year_initial"] = cal_start
                    if "calibration_year_final" in sig.parameters and "calibration_year_final" not in provided_params:
                        call_kwargs["calibration_year_final"] = cal_end

            # call wrapped numpy function with extracted values
            # replace first arg (DataArray) with numpy values
            numpy_args = (numpy_values,) + args[1:]

            # filter call_kwargs to only include params the function accepts
            sig = inspect.signature(func)
            valid_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

            result_values = func(*numpy_args, **valid_kwargs)

            # capture calculation metadata from resolved kwargs
            calc_metadata = None
            if calculation_metadata_keys is not None:
                calc_metadata = {}
                for key in calculation_metadata_keys:
                    if key in valid_kwargs:
                        calc_metadata[key] = valid_kwargs[key]

            # rewrap result as DataArray with preserved coordinates/metadata
            result_da = _build_output_dataarray(input_da, result_values, cf_metadata, calc_metadata)

            # log completion
            logger.info(
                "xarray_adapter_completed",
                function_name=func.__name__,
                input_shape=input_da.shape,
                output_shape=result_da.shape,
                inferred_params=infer_params,
            )

            return result_da

        return wrapper

    return decorator

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

.. warning:: **Beta Feature** — The xarray adapter layer is beta and may change
   in future minor releases. The NumPy computation core (``indices.py``,
   ``compute.py``) is stable. No breaking changes will occur within a minor version.
"""

from __future__ import annotations

import copy
import datetime
import functools
import inspect
import json
import warnings
from collections.abc import Callable
from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd
import structlog.stdlib
import xarray as xr

from climate_indices import compute, eto, indices
from climate_indices.cf_metadata_registry import CF_METADATA, CFAttributes
from climate_indices.compute import MIN_CALIBRATION_YEARS
from climate_indices.exceptions import (
    CoordinateValidationError,
    InputAlignmentWarning,
    InputTypeError,
    InsufficientDataError,
)
from climate_indices.logging_config import get_logger


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

# history attribute formatting
_HISTORY_SEPARATOR = "\n"
_HISTORY_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class InputType(Enum):
    """Classification of input data types for routing.

    Used by detect_input_type() to determine which computation path to use.

    .. note:: Part of the beta xarray adapter layer. See :doc:`xarray_migration`.

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

    .. note:: Part of the beta xarray adapter layer. See :doc:`xarray_migration`.

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

    freq = xr.infer_freq(time_coord)  # type: ignore[no-untyped-call]

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


def _validate_time_dimension(data: xr.DataArray, time_dim: str) -> None:
    """Validate that the time dimension exists in the input DataArray.

    Args:
        data: Input DataArray to validate
        time_dim: Name of the expected time dimension

    Raises:
        CoordinateValidationError: If the time dimension is not found
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
        raise CoordinateValidationError(
            message=error_msg,
            coordinate_name=time_dim,
            reason="missing_dimension",
        )


def _validate_time_monotonicity(time_coord: xr.DataArray) -> None:
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


def _validate_latitude_range(
    latitude: float | int | np.floating | np.integer | xr.DataArray,
) -> None:
    """Validate that latitude values are within [-90, 90].

    Args:
        latitude: Latitude value(s) as a scalar or xr.DataArray

    Raises:
        ValueError: If latitude is NaN, contains only NaNs, or has values outside [-90, 90]
    """
    if isinstance(latitude, xr.DataArray):
        lat_min = float(latitude.min(skipna=True).values)
        lat_max = float(latitude.max(skipna=True).values)

        if np.isnan(lat_min) or np.isnan(lat_max):
            raise ValueError(
                "latitude DataArray contains only NaN values. Provide valid latitude coordinates within [-90, 90]."
            )

        if lat_min < -90 or lat_max > 90:
            raise ValueError(
                f"latitude values must be within [-90, 90]. "
                f"Got range [{lat_min:.2f}, {lat_max:.2f}]. "
                "Check that latitude coordinates use decimal degrees, not radians."
            )
    else:
        lat_value = float(latitude)

        if np.isnan(lat_value):
            raise ValueError("latitude is NaN. Provide a valid latitude value within [-90, 90].")

        if lat_value < -90 or lat_value > 90:
            raise ValueError(
                f"latitude must be within [-90, 90]. Got {lat_value:.2f}. "
                "Check that latitude uses decimal degrees, not radians."
            )


def _build_latitude_attr(latitude: float | xr.DataArray) -> str | int | float | bool:
    """Serialize latitude for storage as a DataArray attribute.

    Args:
        latitude: Latitude value as a scalar or xr.DataArray

    Returns:
        Serialized latitude suitable for xarray attribute storage
    """
    if isinstance(latitude, xr.DataArray):
        lat_metadata = {
            "name": latitude.name,
            "dims": tuple(str(d) for d in latitude.dims),
            "shape": tuple(int(s) for s in latitude.shape),
            "min": float(latitude.min().values),
            "max": float(latitude.max().values),
        }
        return _serialize_attr_value(lat_metadata)
    else:
        return _serialize_attr_value(latitude)


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


def _resolve_scale_from_args(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    """Resolve the scale parameter from function arguments.

    Args:
        func: The function being called
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function

    Returns:
        The scale value if present in the signature and provided, None otherwise
    """
    try:
        sig = inspect.signature(func)
        # check if scale is in the signature
        if "scale" not in sig.parameters:
            return None

        # bind provided args/kwargs to extract scale
        bound = sig.bind_partial(*args, **kwargs)
        return bound.arguments.get("scale")
    except (TypeError, ValueError):
        # if binding fails, return None
        return None


def _validate_sufficient_data(time_coord: xr.DataArray, scale: int) -> None:
    """Validate that there is sufficient data for the given scale.

    Args:
        time_coord: Time coordinate DataArray
        scale: Scale parameter for the index calculation

    Raises:
        InsufficientDataError: If there are fewer time steps than the scale requires
    """
    n_timesteps = len(time_coord)
    if n_timesteps < scale:
        error_msg = (
            f"Insufficient data for scale={scale}: {n_timesteps} time steps available, but at least {scale} required."
        )
        _log().error(
            "insufficient_data_for_scale",
            scale=scale,
            available_timesteps=n_timesteps,
            required_timesteps=scale,
        )
        raise InsufficientDataError(
            message=error_msg,
            non_zero_count=n_timesteps,
            required_count=scale,
        )


def _assess_nan_density(data: xr.DataArray) -> dict[str, Any]:
    """Assess NaN density in input data for diagnostic logging.

    Pure diagnostic function that computes NaN metrics without side effects.
    The nan_positions mask is returned for reuse in propagation verification,
    avoiding redundant computation.

    Args:
        data: Input DataArray to assess

    Returns:
        Dictionary containing:
            - total_values: Total number of values in the array
            - nan_count: Number of NaN values
            - nan_ratio: Proportion of NaN values (0.0 to 1.0)
            - has_nan: Boolean indicating presence of any NaN values
            - nan_positions: Boolean numpy array mask (True where NaN, False otherwise)
    """
    values = data.values
    nan_mask = np.isnan(values)
    nan_count = int(np.sum(nan_mask))
    total_values = int(values.size)

    return {
        "total_values": total_values,
        "nan_count": nan_count,
        "nan_ratio": nan_count / total_values if total_values > 0 else 0.0,
        "has_nan": nan_count > 0,
        "nan_positions": nan_mask,
    }


def _verify_nan_propagation(
    input_nan_mask: np.ndarray[Any, Any],
    output_values: np.ndarray[Any, Any],
) -> bool:
    """Verify that input NaN positions remain NaN in output.

    Checks the NaN propagation contract: every input NaN position must be NaN
    in the output. This is a one-directional check—output may have additional
    NaN values from convolution padding or boundary effects, which is expected
    and not considered a violation.

    Args:
        input_nan_mask: Boolean mask from input (True where NaN)
        output_values: Output array values to verify

    Returns:
        True if NaN propagation contract holds (all input NaN → output NaN),
        False if any input NaN position has a non-NaN output value
    """
    # get output NaN positions
    output_nan_mask = np.isnan(output_values)

    # check that all input NaN positions are still NaN in output
    # input_nan_mask[i] == True implies output_nan_mask[i] == True
    # equivalent to: NOT(input_nan AND NOT output_nan)
    contract_holds = np.all(~input_nan_mask | output_nan_mask)

    return bool(contract_holds)


def _validate_calibration_non_nan_sample_size(
    time_coord: xr.DataArray,
    values: np.ndarray[Any, Any],
    calibration_year_initial: int,
    calibration_year_final: int,
    min_years: int = MIN_CALIBRATION_YEARS,
) -> None:
    """Validate sufficient non-NaN data in calibration period for distribution fitting.

    Hard validation that raises an error when the calibration period has fewer than
    the minimum required years of non-NaN data. This prevents impossible-to-fit
    scenarios early in the pipeline, before expensive computation.

    This is distinct from compute.py's _check_calibration_data_quality warning:
    - This function: Hard error for <30 non-NaN years (fitting impossible)
    - compute.py warning: Soft warning for >20% NaN density (fitting marginal)

    Args:
        time_coord: Time coordinate DataArray with datetime values
        values: 1-D numpy array of data values to check
        calibration_year_initial: Start year of calibration period (inclusive)
        calibration_year_final: End year of calibration period (inclusive)
        min_years: Minimum required years of non-NaN data (default: 30)

    Raises:
        InsufficientDataError: If calibration period has fewer than min_years
            of non-NaN data, making distribution fitting impossible
    """
    # extract year values from time coordinate
    time_years = pd.DatetimeIndex(time_coord.values).year.values

    # find indices within calibration period
    calibration_mask = (time_years >= calibration_year_initial) & (time_years <= calibration_year_final)

    # extract calibration slice
    calibration_values = values[calibration_mask]

    if len(calibration_values) == 0:
        raise InsufficientDataError(
            message=(
                f"Calibration period ({calibration_year_initial}-{calibration_year_final}) "
                f"contains no data points. Check that calibration years overlap with time coordinate range."
            ),
            non_zero_count=0,
            required_count=min_years,
        )

    # count non-NaN values
    non_nan_count = int(np.sum(~np.isnan(calibration_values)))

    # infer periods per year from time coordinate frequency
    freq = xr.infer_freq(time_coord)  # type: ignore[no-untyped-call]
    if freq in ("MS", "ME", "M"):
        periods_per_year = 12
    elif freq == "D":
        periods_per_year = 365
    else:
        # fallback: estimate from total calibration span
        periods_per_year = len(calibration_values) // (calibration_year_final - calibration_year_initial + 1)
        if periods_per_year == 0:
            periods_per_year = 12  # conservative default

    # compute effective non-NaN years
    effective_years = non_nan_count / periods_per_year

    if effective_years < min_years:
        raise InsufficientDataError(
            message=(
                f"Insufficient non-NaN data in calibration period ({calibration_year_initial}-{calibration_year_final}). "
                f"Found {non_nan_count} non-NaN values ({effective_years:.1f} effective years), "
                f"but at least {min_years} years of non-NaN data required for reliable distribution fitting."
            ),
            non_zero_count=non_nan_count,
            required_count=int(min_years * periods_per_year),
        )


def _resolve_secondary_inputs(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    additional_input_names: list[str],
) -> dict[str, tuple[int | None, Any]]:
    """Resolve secondary input parameters from function arguments.

    Uses function signature introspection to identify which parameters correspond
    to additional inputs (e.g., "pet" for SPEI), and resolves their values from
    the provided positional and keyword arguments.

    Args:
        func: The function being wrapped
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function
        additional_input_names: List of parameter names to resolve (e.g., ["pet"])

    Returns:
        Dict mapping parameter name to (positional_index | None, value).
        positional_index is the position in args, or None if provided as kwarg.

    Examples:
        >>> def spei(precip, pet, scale): ...
        >>> _resolve_secondary_inputs(spei, (precip_da, pet_da, 3), {}, ["pet"])
        {"pet": (1, pet_da)}
        >>> _resolve_secondary_inputs(spei, (precip_da,), {"pet": pet_da, "scale": 3}, ["pet"])
        {"pet": (None, pet_da)}
    """
    if not additional_input_names:
        return {}

    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    resolved: dict[str, tuple[int | None, Any]] = {}

    for name in additional_input_names:
        if name not in sig.parameters:
            # parameter not in function signature, skip
            continue

        # check if provided as keyword argument
        if name in kwargs:
            resolved[name] = (None, kwargs[name])
            continue

        # check if provided as positional argument
        param_index = param_names.index(name)
        if param_index < len(args):
            resolved[name] = (param_index, args[param_index])

    return resolved


def _align_inputs(
    primary: xr.DataArray,
    secondaries: dict[str, xr.DataArray],
    time_dim: str = "time",
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Align primary and secondary DataArrays using inner join on coordinates.

    Ensures all input DataArrays share the same time coordinates by taking the
    intersection of their time ranges. This is essential for multi-input indices
    like SPEI where precipitation and PET must align.

    Args:
        primary: Primary input DataArray (e.g., precipitation)
        secondaries: Dict mapping parameter names to secondary DataArrays (e.g., {"pet": pet_da})
        time_dim: Name of the time dimension to align on (default: "time")

    Returns:
        Tuple of (aligned_primary, dict_of_aligned_secondaries)

    Raises:
        CoordinateValidationError: If alignment results in empty intersection (no overlapping time steps)

    Warns:
        InputAlignmentWarning: If alignment drops time steps from the primary input
    """
    if not secondaries:
        # no secondaries to align, return primary unchanged
        return primary, {}

    # collect all DataArrays for alignment
    all_arrays = [primary] + list(secondaries.values())

    # align using inner join (intersection of coordinates)
    aligned = xr.align(*all_arrays, join="inner")

    # extract aligned arrays
    aligned_primary = aligned[0]
    aligned_secondaries = {name: aligned[i + 1] for i, name in enumerate(secondaries.keys())}

    # check for empty intersection
    if time_dim in aligned_primary.dims:
        aligned_size = len(aligned_primary[time_dim])
        original_size = len(primary[time_dim])

        if aligned_size == 0:
            raise CoordinateValidationError(
                message=(
                    f"Input alignment resulted in empty intersection on '{time_dim}' coordinate. "
                    f"Primary input and secondary inputs have no overlapping time steps. "
                    f"Check that your input time ranges overlap."
                ),
                coordinate_name=time_dim,
                reason="empty_intersection_after_alignment",
            )

        # emit warning if data was dropped
        if aligned_size < original_size:
            dropped_count = original_size - aligned_size
            warning_msg = (
                f"Input alignment dropped {dropped_count} time step(s) from primary input. "
                f"Original size: {original_size}, aligned size: {aligned_size}. "
                f"Computation will use only the intersection of input time ranges."
            )
            _log().warning(
                "input_alignment_dropped_data",
                original_size=original_size,
                aligned_size=aligned_size,
                dropped_count=dropped_count,
                time_dim=time_dim,
            )
            warnings.warn(
                InputAlignmentWarning(
                    message=warning_msg,
                    original_size=original_size,
                    aligned_size=aligned_size,
                    dropped_count=dropped_count,
                ),
                stacklevel=3,
            )

    return aligned_primary, aligned_secondaries


def _serialize_attr_value(value: Any) -> str | int | float | bool:
    """Serialize attribute values for xarray compatibility.

    Converts enum instances to their string name representation, serializes dicts
    to JSON strings, passes through native xarray-serializable types, and raises
    TypeError for non-serializable types.

    Args:
        value: Attribute value to serialize

    Returns:
        Serialized value: enum→name string, dict→JSON string, or passthrough
        for str/int/float/bool/numpy scalars

    Raises:
        TypeError: If value is not serializable (list, complex objects)

    Examples:
        >>> _serialize_attr_value(Distribution.gamma)
        'gamma'
        >>> _serialize_attr_value(42)
        42
        >>> _serialize_attr_value("monthly")
        'monthly'
        >>> _serialize_attr_value({"min": -5.0, "max": 45.0})
        '{"min": -5.0, "max": 45.0}'
        >>> _serialize_attr_value([1, 2])
        TypeError: ...
    """
    # enum → .name string
    if isinstance(value, Enum):
        return value.name

    # numpy scalar → python scalar
    if isinstance(value, np.integer | np.floating):
        return value.item()

    # passthrough native serializable types
    if isinstance(value, str | int | float | bool):
        return value

    # dict → JSON string
    if isinstance(value, dict):
        return json.dumps(value)

    # reject non-serializable types
    raise TypeError(
        f"Cannot serialize attribute value of type {type(value).__name__}. "
        f"Supported types: Enum, dict, str, int, float, bool, numpy scalars."
    )


def _build_history_entry(
    index_name: str,
    version: str,
    calculation_metadata: dict[str, Any] | None = None,
) -> str:
    """Build a CF-compliant history entry for a climate index calculation.

    Args:
        index_name: Display name of the climate index (e.g., "SPI")
        version: Library version string (e.g., "2.0.0")
        calculation_metadata: Optional dict containing calculation parameters
            (e.g., scale, distribution). Enum values are serialized via .name.

    Returns:
        Formatted history entry: "YYYY-MM-DDTHH:MM:SSZ: {description} (climate_indices v{version})"

    Examples:
        >>> _build_history_entry("SPI", "2.0.0", {"scale": 3, "distribution": Distribution.gamma})
        "2026-02-07T10:23:45Z: SPI-3 calculated using gamma distribution (climate_indices v2.0.0)"
        >>> _build_history_entry("SPI", "2.0.0", {"scale": 3})
        "2026-02-07T10:23:45Z: SPI-3 calculated (climate_indices v2.0.0)"
        >>> _build_history_entry("SPI", "2.0.0")
        "2026-02-07T10:23:45Z: SPI calculated (climate_indices v2.0.0)"
    """
    # generate UTC timestamp
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(_HISTORY_TIMESTAMP_FORMAT)

    # build description from metadata
    description_parts = [index_name]

    if calculation_metadata:
        # add scale if present
        scale = calculation_metadata.get("scale")
        if scale is not None:
            description_parts[0] = f"{index_name}-{scale}"

        # build calculation description
        distribution = calculation_metadata.get("distribution")
        if distribution is not None:
            # serialize enum to string using .name attribute
            dist_name = distribution.name if isinstance(distribution, Enum) else str(distribution)
            description = f"{description_parts[0]} calculated using {dist_name} distribution"
        elif scale is not None:
            description = f"{description_parts[0]} calculated"
        else:
            description = f"{index_name} calculated"
    else:
        description = f"{index_name} calculated"

    return f"{timestamp}: {description} (climate_indices v{version})"


def _append_history(
    existing_attrs: dict[str, Any],
    new_entry: str,
) -> str:
    """Append a new history entry to existing history attribute.

    Follows CF Convention newline-delimited format for multi-entry history logs.

    Args:
        existing_attrs: Current attribute dictionary (may contain existing history)
        new_entry: New history entry to append

    Returns:
        Updated history string with new entry appended

    Examples:
        >>> _append_history({}, "2026-02-07T10:00:00Z: SPI calculated")
        "2026-02-07T10:00:00Z: SPI calculated"
        >>> _append_history(
        ...     {"history": "2026-02-06T09:00:00Z: Data prepared"},
        ...     "2026-02-07T10:00:00Z: SPI calculated"
        ... )
        "2026-02-06T09:00:00Z: Data prepared\\n2026-02-07T10:00:00Z: SPI calculated"
    """
    existing = existing_attrs.get("history", "")

    # treat falsy, non-string, or whitespace-only values as no existing history
    if not existing or not isinstance(existing, str) or not existing.strip():
        return new_entry

    # append new entry with newline separator
    return f"{existing.rstrip()}{_HISTORY_SEPARATOR}{new_entry}"


def _infer_temporal_parameters(
    func: Callable[..., Any],
    input_da: xr.DataArray,
    modified_args: list[Any],
    modified_kwargs: dict[str, Any],
    time_dim: str,
) -> dict[str, Any]:
    """Infer missing temporal parameters from time coordinate metadata.

    Pure metadata-based inference—operates only on coordinate values, safe for
    Dask arrays. Does NOT include calibration NaN validation (requires .values).

    Args:
        func: The function being wrapped
        input_da: Input DataArray with time coordinate
        modified_args: Positional arguments (may be modified by alignment)
        modified_kwargs: Keyword arguments (may be modified by alignment)
        time_dim: Name of the time dimension

    Returns:
        Dictionary of inferred parameters (data_start_year, periodicity,
        calibration_year_initial, calibration_year_final). Only includes
        parameters that are in the function signature and not already provided.
    """
    inferred: dict[str, Any] = {}

    # skip if time dimension doesn't exist
    if time_dim not in input_da.dims:
        return inferred

    time_coord = input_da[time_dim]

    # use inspect to determine which parameters the function accepts
    sig = inspect.signature(func)

    # bind provided args/kwargs to see what's already specified
    try:
        # bind_partial allows missing parameters (we'll fill them)
        bound = sig.bind_partial(*modified_args, **modified_kwargs)
        bound.apply_defaults()
        provided_params = set(bound.arguments.keys())
    except TypeError:
        # if binding fails, skip inference
        provided_params = set()

    # infer data_start_year if not provided
    if "data_start_year" in sig.parameters and "data_start_year" not in provided_params:
        inferred["data_start_year"] = _infer_data_start_year(time_coord)

    # infer periodicity if not provided
    if "periodicity" in sig.parameters and "periodicity" not in provided_params:
        inferred["periodicity"] = _infer_periodicity(time_coord)

    # infer calibration period if either param is not provided
    needs_cal_initial = (
        "calibration_year_initial" in sig.parameters and "calibration_year_initial" not in provided_params
    )
    needs_cal_final = "calibration_year_final" in sig.parameters and "calibration_year_final" not in provided_params

    if needs_cal_initial or needs_cal_final:
        cal_start, cal_end = _infer_calibration_period(time_coord)
        if needs_cal_initial:
            inferred["calibration_year_initial"] = cal_start
        if needs_cal_final:
            inferred["calibration_year_final"] = cal_end

    return inferred


def _is_dask_backed(data: xr.DataArray) -> bool:
    """Check if a DataArray is backed by a Dask array.

    Uses the canonical xarray idiom (data.chunks is not None) to detect Dask-backed
    arrays without importing dask.array directly.

    Args:
        data: DataArray to check

    Returns:
        True if data is Dask-backed, False otherwise
    """
    return data.chunks is not None


def _validate_dask_chunks(data: xr.DataArray, time_dim: str) -> None:
    """Validate that the time dimension is not split across multiple Dask chunks.

    SPI/SPEI require the full time series for distribution fitting, so the time
    dimension must be in a single chunk (or unchunked). Spatial dimensions can
    be arbitrarily chunked for parallel computation.

    Args:
        data: Dask-backed DataArray to validate
        time_dim: Name of the time dimension

    Raises:
        CoordinateValidationError: If time dimension is split across multiple chunks,
            with a message including the exact rechunking command to fix it
    """
    # skip validation if time dimension doesn't exist (already validated elsewhere)
    if time_dim not in data.dims:
        return

    # skip validation if not chunked (shouldn't happen since we call this after is_dask check)
    if data.chunks is None:
        return

    # get chunks for time dimension
    # data.chunks is a tuple-of-tuples indexed by dimension position
    time_chunks = data.chunks[data.dims.index(time_dim)]

    # validate single chunk on time dimension
    if len(time_chunks) > 1:
        error_msg = (
            f"Time dimension '{time_dim}' is split across {len(time_chunks)} chunks. "
            f"Climate indices require the full time series for distribution fitting. "
            f"Rechunk using: data = data.chunk({{'{time_dim}': -1}})"
        )
        _log().error(
            "multi_chunked_time_dimension",
            time_dim=time_dim,
            num_chunks=len(time_chunks),
            chunk_sizes=time_chunks,
        )
        raise CoordinateValidationError(
            message=error_msg,
            coordinate_name=time_dim,
            reason="multi_chunked_time_dimension",
        )


def _build_output_attrs(
    input_da: xr.DataArray,
    cf_metadata: dict[str, str] | None = None,
    calculation_metadata: dict[str, Any] | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    """Build output attributes with CF metadata, calculation metadata, version, and history.

    Pure attribute construction—extracts the dict-building logic from _build_output_dataarray()
    so both in-memory and Dask execution paths can apply identical metadata.

    Args:
        input_da: Original input DataArray with full coordinate metadata
        cf_metadata: Optional CF Convention metadata to apply to DataArray-level attributes
        calculation_metadata: Optional dict of calculation-specific metadata
            (e.g., scale, distribution). Enum values are automatically serialized to .name strings.
        index_name: Optional climate index display name for history tracking
            (e.g., "SPI"). If provided, appends a CF-compliant history entry.

    Returns:
        Dictionary of attributes to assign to output DataArray

    Notes:
        - Attribute layering: input attrs → CF metadata → calculation metadata → version → history
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
                _log().warning(
                    "calculation_metadata_serialization_failed",
                    key=key,
                    value_type=type(value).__name__,
                    error=str(e),
                )

    # add library version for provenance
    # deferred import to avoid circular dependency (__init__.py imports this module)
    from climate_indices import __version__

    output_attrs["climate_indices_version"] = __version__

    # add history entry for provenance tracking
    if index_name is not None:
        history_entry = _build_history_entry(index_name, __version__, calculation_metadata)
        output_attrs["history"] = _append_history(output_attrs, history_entry)

    return output_attrs


def _build_output_dataarray(
    input_da: xr.DataArray,
    result_values: np.ndarray[Any, Any],
    cf_metadata: dict[str, str] | None = None,
    calculation_metadata: dict[str, Any] | None = None,
    index_name: str | None = None,
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
        index_name: Optional climate index display name for history tracking
            (e.g., "SPI"). If provided, appends a CF-compliant history entry.

    Returns:
        DataArray with result_values and preserved coordinates/dims/attrs

    Notes:
        - Deep-copies coordinate attrs to prevent mutation bleed-through
        - Preserves coordinate ordering (dict insertion order)
        - CF metadata only affects DataArray-level attrs, not coord attrs
        - Preserves the input DataArray's .name attribute
        - Attribute layering: input attrs → CF metadata → calculation metadata → version → history
    """
    # build output attrs using extracted helper
    output_attrs = _build_output_attrs(input_da, cf_metadata, calculation_metadata, index_name)

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


def _capture_calculation_metadata(
    calculation_metadata_keys: list[str] | tuple[str, ...] | None,
    valid_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Collect configured calculation metadata from valid kwargs."""
    if calculation_metadata_keys is None:
        return None

    calc_metadata: dict[str, Any] = {}
    for key in calculation_metadata_keys:
        if key in valid_kwargs:
            calc_metadata[key] = valid_kwargs[key]
    return calc_metadata


def _collect_input_dataarrays(
    input_da: xr.DataArray,
    additional_input_names: list[str] | None,
    resolved_secondaries: dict[str, tuple[int | None, Any]],
    modified_args: list[Any],
    modified_kwargs: dict[str, Any],
) -> list[xr.DataArray]:
    """Collect primary + aligned secondary DataArrays for apply_ufunc.

    Args:
        input_da: Primary input DataArray
        additional_input_names: Names of additional input parameters
        resolved_secondaries: Mapping of secondary input name to (position, value)
        modified_args: Modified positional arguments (with aligned secondaries)
        modified_kwargs: Modified keyword arguments (with aligned secondaries)

    Returns:
        List of DataArrays to pass to apply_ufunc (primary + secondaries in order)
    """
    input_dataarrays = [input_da]
    if additional_input_names and resolved_secondaries:
        for name in additional_input_names:
            if name not in resolved_secondaries:
                continue
            pos_index, original_value = resolved_secondaries[name]
            if not isinstance(original_value, xr.DataArray):
                continue
            # pull the aligned DataArray from modified_args or modified_kwargs
            if pos_index is not None:
                input_dataarrays.append(modified_args[pos_index])
            else:
                input_dataarrays.append(modified_kwargs[name])
    return input_dataarrays


def _finalize_ufunc_result(
    result_da: xr.DataArray,
    input_da: xr.DataArray,
    valid_kwargs: dict[str, Any],
    *,
    cf_metadata: dict[str, str] | None,
    calculation_metadata_keys: list[str] | tuple[str, ...] | None,
    index_display_name: str | None,
    func_name: str,
) -> xr.DataArray:
    """Apply post-apply_ufunc processing: reorder dims, attach metadata, copy coord attrs.

    Args:
        result_da: Result DataArray from apply_ufunc
        input_da: Original input DataArray
        valid_kwargs: Filtered kwargs passed to the wrapped function
        cf_metadata: CF convention metadata for the output
        calculation_metadata_keys: Keys to extract from valid_kwargs for metadata
        index_display_name: Display name for the index (or None to use func_name.upper())
        func_name: Name of the wrapped function

    Returns:
        Finalized DataArray with restored dimensions, metadata, and coordinate attributes
    """
    # restore dimension order (apply_ufunc moves core dims to end)
    if result_da.dims != input_da.dims:
        result_da = result_da.transpose(*input_da.dims)

    # apply metadata using _build_output_attrs
    calc_metadata = _capture_calculation_metadata(calculation_metadata_keys, valid_kwargs)
    resolved_index_name = index_display_name if index_display_name is not None else func_name.upper()
    output_attrs = _build_output_attrs(input_da, cf_metadata, calc_metadata, index_name=resolved_index_name)
    result_da.attrs.update(output_attrs)

    # deep-copy coordinate attrs
    for coord_name in result_da.coords:
        if coord_name in input_da.coords:
            result_da.coords[coord_name].attrs = copy.deepcopy(input_da.coords[coord_name].attrs)

    # preserve .name
    result_da.name = input_da.name

    return result_da


def xarray_adapter(
    *,
    cf_metadata: dict[str, str] | None = None,
    time_dim: str = "time",
    infer_params: bool = True,
    calculation_metadata_keys: list[str] | tuple[str, ...] | None = None,
    index_display_name: str | None = None,
    additional_input_names: list[str] | None = None,
    skipna: bool = False,
) -> Callable[[Callable[..., np.ndarray[Any, Any]]], Callable[..., np.ndarray[Any, Any] | xr.DataArray]]:
    """Decorator factory that adapts NumPy index functions to accept xarray DataArrays.

    This decorator implements the adapter contract: detect → [resolve → align] → extract → infer → compute → rewrap → log.
    It transparently handles both NumPy arrays (passthrough) and xarray DataArrays (extract,
    compute with NumPy function, rewrap result). For multi-input functions, it aligns DataArrays
    using inner join before computation.

    .. warning:: **Beta Feature** — The ``@xarray_adapter`` decorator and all xarray
       dispatch infrastructure are beta. The decorator interface may change in future
       minor releases. NumPy passthrough behavior is stable.

    Args:
        cf_metadata: Optional dict of CF Convention metadata to apply to output DataArray.
            Keys should be CF attribute names (e.g., 'standard_name', 'long_name', 'units').
            These override conflicting attributes from the input DataArray.
        time_dim: Name of the time dimension in the input DataArray (default: "time").
            Used for parameter inference and alignment.
        infer_params: If True, automatically infer missing parameters (data_start_year,
            periodicity, calibration_year_initial, calibration_year_final) from the time
            coordinate. Explicit parameter values always override inferred values.
        calculation_metadata_keys: Optional sequence of parameter names to capture as
            output metadata attributes. For example, ["scale", "distribution"] will
            add these kwargs to the output DataArray.attrs. Enum values are automatically
            serialized to their .name string representation.
        index_display_name: Optional display name for the climate index (e.g., "SPI")
            to include in the CF-compliant history attribute. If None, defaults to
            the uppercase function name.
        additional_input_names: Optional list of parameter names for secondary inputs
            (e.g., ["pet"] for SPEI). When provided, these inputs will be aligned with
            the primary input using xr.align(join='inner') before computation. Only
            DataArray secondaries are aligned; numpy secondaries pass through unchanged.
        skipna: If False (default), NaN values are propagated through calculations
            (NaN in → NaN out). If True, implements pairwise deletion for NaN handling
            (FR-INPUT-004). Currently only skipna=False is implemented; skipna=True
            raises NotImplementedError.

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

            # xarray path: detect → [resolve → align] → validate → extract → infer → compute → rewrap → log
            input_da = data

            # check skipna parameter (Story 2.8)
            if skipna:
                raise NotImplementedError(
                    "skipna=True not yet implemented (FR-INPUT-004). "
                    "NaN values are propagated through calculations by default."
                )

            # resolve and align secondary inputs (Story 3.1)
            modified_args = list(args)
            modified_kwargs = dict(kwargs)
            resolved_secondaries: dict[str, tuple[int | None, Any]] = {}

            if additional_input_names:
                # resolve secondary inputs from args/kwargs
                resolved_secondaries = _resolve_secondary_inputs(func, args, kwargs, additional_input_names)

                # filter to only DataArray secondaries for alignment
                # (numpy secondaries pass through unchanged)
                dataarray_secondaries = {
                    name: value for name, (_, value) in resolved_secondaries.items() if isinstance(value, xr.DataArray)
                }

                if dataarray_secondaries:
                    # align primary + DataArray secondaries
                    aligned_primary, aligned_secondaries = _align_inputs(input_da, dataarray_secondaries, time_dim)

                    # update input_da to use aligned primary
                    input_da = aligned_primary

                    # replace args[0] with aligned primary
                    modified_args[0] = aligned_primary

                    # replace aligned secondaries in args/kwargs
                    for name, (pos_index, _) in resolved_secondaries.items():
                        if name in aligned_secondaries:
                            if pos_index is not None:
                                # replace positional arg
                                modified_args[pos_index] = aligned_secondaries[name]
                            else:
                                # replace kwarg
                                modified_kwargs[name] = aligned_secondaries[name]

            # coordinate validation (Story 2.7)
            if infer_params:
                _validate_time_dimension(input_da, time_dim)
                time_coord = input_da[time_dim]
                _validate_time_monotonicity(time_coord)
                resolved_scale = _resolve_scale_from_args(func, tuple(modified_args), modified_kwargs)
                if resolved_scale is not None:
                    _validate_sufficient_data(time_coord, resolved_scale)

            # detect Dask-backed arrays (Story 2.9)
            is_dask = _is_dask_backed(input_da)
            if is_dask:
                # validate chunking constraints for Dask
                _validate_dask_chunks(input_da, time_dim)

            # infer temporal parameters if enabled (shared path)
            inferred_params: dict[str, Any] = {}
            if infer_params:
                inferred_params = _infer_temporal_parameters(func, input_da, modified_args, modified_kwargs, time_dim)
                # log which parameters were inferred and their values
                if inferred_params:
                    _log().info(
                        "parameters_inferred",
                        function_name=func.__name__,
                        **{k: str(v) for k, v in inferred_params.items()},
                    )

            # branch: Dask execution or in-memory execution
            if is_dask:
                # ═══════════════════════════════════════════════════════════════════
                # Dask execution path (Story 2.9)
                # ═══════════════════════════════════════════════════════════════════

                # build call_kwargs from modified_kwargs + inferred params
                call_kwargs = dict(modified_kwargs)
                call_kwargs.update(inferred_params)

                # filter kwargs to function signature, excluding DataArray secondary names
                # (secondaries are positional args to apply_ufunc, not kwargs)
                sig = inspect.signature(func)
                secondary_names = set(additional_input_names or [])
                valid_kwargs = {
                    k: v for k, v in call_kwargs.items() if k in sig.parameters and k not in secondary_names
                }

                # collect input DataArrays for apply_ufunc in parameter order
                # primary + aligned secondaries (reuse resolution from earlier in wrapper)
                input_dataarrays = _collect_input_dataarrays(
                    input_da, additional_input_names, resolved_secondaries, modified_args, modified_kwargs
                )

                # create closure capturing valid_kwargs
                def _numpy_func_wrapper(*numpy_arrays: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
                    return func(*numpy_arrays, **valid_kwargs)

                # call apply_ufunc with Dask support
                result_da: xr.DataArray = xr.apply_ufunc(
                    _numpy_func_wrapper,
                    *input_dataarrays,
                    input_core_dims=[[time_dim]] * len(input_dataarrays),
                    output_core_dims=[[time_dim]],
                    dask="parallelized",
                    vectorize=True,
                    output_dtypes=[float],
                )

                # finalize result: restore dims, attach metadata, copy coord attrs
                result_da = _finalize_ufunc_result(
                    result_da,
                    input_da,
                    valid_kwargs,
                    cf_metadata=cf_metadata,
                    calculation_metadata_keys=calculation_metadata_keys,
                    index_display_name=index_display_name,
                    func_name=func.__name__,
                )

                # log completion (NaN metrics omitted for Dask—would trigger compute)
                _log().info(
                    "xarray_adapter_completed",
                    function_name=func.__name__,
                    input_shape=input_da.shape,
                    output_shape=result_da.shape,
                    inferred_params=infer_params,
                    dask_backed=True,
                )

                return result_da

            # ═══════════════════════════════════════════════════════════════════
            # In-memory execution path (original logic)
            # ═══════════════════════════════════════════════════════════════════

            # assess NaN density for diagnostics (Story 2.8)
            nan_assessment = _assess_nan_density(input_da)
            if nan_assessment["has_nan"]:
                _log().info(
                    "nan_detected_in_input",
                    function_name=func.__name__,
                    nan_count=nan_assessment["nan_count"],
                    nan_ratio=round(nan_assessment["nan_ratio"], 4),
                    total_values=nan_assessment["total_values"],
                )

            # build kwargs for the wrapped function
            # start with explicitly provided kwargs (including extracted secondaries)
            call_kwargs = dict(modified_kwargs)

            # apply inferred params (already computed above before the branch)
            call_kwargs.update(inferred_params)

            # filter call_kwargs to only include params the function accepts
            sig = inspect.signature(func)
            valid_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

            # check if input is multi-dimensional (has spatial dims beyond time)
            # and has a time dimension (required for apply_ufunc with input_core_dims)
            if input_da.ndim > 1 and time_dim in input_da.dims:
                # ═══════════════════════════════════════════════════════════════════
                # Multi-dimensional in-memory execution path
                # ═══════════════════════════════════════════════════════════════════
                # use xr.apply_ufunc with vectorize=True to handle spatial broadcasting
                # similar to Dask path but without dask="parallelized"

                # validate calibration period has sufficient non-NaN data (Story 2.8)
                if nan_assessment["has_nan"] and infer_params and time_dim in input_da.dims:
                    time_coord = input_da[time_dim]
                    cal_initial = call_kwargs.get("calibration_year_initial")
                    cal_final = call_kwargs.get("calibration_year_final")
                    if cal_initial is not None and cal_final is not None:
                        # for multi-dimensional validation, use first spatial point
                        sample_values = input_da.values
                        if sample_values.ndim > 1:
                            # extract first spatial slice along time dimension
                            slices = [slice(None)] + [0] * (sample_values.ndim - 1)
                            sample_values = sample_values[tuple(slices)]
                        _validate_calibration_non_nan_sample_size(
                            time_coord,
                            sample_values,
                            calibration_year_initial=cal_initial,
                            calibration_year_final=cal_final,
                        )

                # collect input DataArrays for apply_ufunc in parameter order
                input_dataarrays = _collect_input_dataarrays(
                    input_da, additional_input_names, resolved_secondaries, modified_args, modified_kwargs
                )

                # create closure capturing valid_kwargs
                def _numpy_func_wrapper(*numpy_arrays: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
                    return func(*numpy_arrays, **valid_kwargs)

                # call apply_ufunc without Dask support (in-memory vectorization)
                result_da: xr.DataArray = xr.apply_ufunc(  # type: ignore[no-redef]
                    _numpy_func_wrapper,
                    *input_dataarrays,
                    input_core_dims=[[time_dim]] * len(input_dataarrays),
                    output_core_dims=[[time_dim]],
                    vectorize=True,
                    output_dtypes=[float],
                )

                # finalize result: restore dims, attach metadata, copy coord attrs
                result_da = _finalize_ufunc_result(
                    result_da,
                    input_da,
                    valid_kwargs,
                    cf_metadata=cf_metadata,
                    calculation_metadata_keys=calculation_metadata_keys,
                    index_display_name=index_display_name,
                    func_name=func.__name__,
                )

                # log completion
                log_fields = {
                    "function_name": func.__name__,
                    "input_shape": input_da.shape,
                    "output_shape": result_da.shape,
                    "inferred_params": infer_params,
                    "vectorized": True,
                }
                if nan_assessment["has_nan"]:
                    log_fields["input_nan_count"] = nan_assessment["nan_count"]
                    log_fields["input_nan_ratio"] = round(nan_assessment["nan_ratio"], 4)

                _log().info("xarray_adapter_completed", **log_fields)
                return result_da

            # ═══════════════════════════════════════════════════════════════════
            # 1D in-memory execution path
            # ═══════════════════════════════════════════════════════════════════

            # extract numpy values from primary
            numpy_values = input_da.values

            # extract numpy values from secondary DataArrays (if any)
            if additional_input_names:
                for name, (pos_index, value) in resolved_secondaries.items():
                    if isinstance(value, xr.DataArray):
                        # extract .values from aligned DataArray
                        if pos_index is not None:
                            modified_args[pos_index] = modified_args[pos_index].values
                        else:
                            modified_kwargs[name] = modified_kwargs[name].values

            # validate calibration period has sufficient non-NaN data (Story 2.8)
            # this validation requires .values, so it only runs in the in-memory path
            if nan_assessment["has_nan"] and infer_params and time_dim in input_da.dims:
                time_coord = input_da[time_dim]
                # check if we have calibration years (either inferred or provided)
                cal_initial = call_kwargs.get("calibration_year_initial")
                cal_final = call_kwargs.get("calibration_year_final")
                if cal_initial is not None and cal_final is not None:
                    _validate_calibration_non_nan_sample_size(
                        time_coord,
                        numpy_values,
                        calibration_year_initial=cal_initial,
                        calibration_year_final=cal_final,
                    )

            # call wrapped numpy function with extracted values
            # replace first arg (DataArray) with numpy values
            numpy_args = (numpy_values,) + tuple(modified_args[1:])

            result_values = func(*numpy_args, **valid_kwargs)

            # verify NaN propagation contract (Story 2.8)
            if nan_assessment["has_nan"]:
                if not _verify_nan_propagation(nan_assessment["nan_positions"], result_values):
                    _log().warning(
                        "nan_propagation_violation",
                        function_name=func.__name__,
                        message="Output missing NaN values present in input",
                    )

            # capture calculation metadata from resolved kwargs
            calc_metadata = _capture_calculation_metadata(calculation_metadata_keys, valid_kwargs)

            # resolve index name for history tracking
            resolved_index_name = index_display_name if index_display_name is not None else func.__name__.upper()

            # rewrap result as DataArray with preserved coordinates/metadata
            result_da = _build_output_dataarray(
                input_da, result_values, cf_metadata, calc_metadata, index_name=resolved_index_name
            )

            # log completion with NaN metrics (Story 2.8)
            log_fields = {
                "function_name": func.__name__,
                "input_shape": input_da.shape,
                "output_shape": result_da.shape,
                "inferred_params": infer_params,
            }
            if nan_assessment["has_nan"]:
                log_fields["input_nan_count"] = nan_assessment["nan_count"]
                log_fields["input_nan_ratio"] = round(nan_assessment["nan_ratio"], 4)

            _log().info("xarray_adapter_completed", **log_fields)

            return result_da

        return wrapper

    return decorator


def pet_thornthwaite(
    temperature: np.ndarray | xr.DataArray,
    latitude: float | np.floating | xr.DataArray,
    data_start_year: int | None = None,
    time_dim: str = "time",
) -> np.ndarray | xr.DataArray:
    """Compute potential evapotranspiration using Thornthwaite method.

    This function provides xarray DataArray support for the Thornthwaite PET calculation.
    Unlike the @xarray_adapter decorator (designed for time-series inputs aligned along
    a time dimension), this function uses xr.apply_ufunc to handle spatial broadcasting
    of the latitude parameter across gridded temperature data.

    .. warning:: **Beta Feature (xarray path)** — When called with ``xr.DataArray``
       input, this function uses the beta xarray adapter layer. The NumPy array
       interface and underlying computation are stable.

    Args:
        temperature: Monthly average temperature values in degrees Celsius.
            For numpy: 1-D array of monthly temperatures
            For xarray: DataArray with time dimension (may have additional spatial dims)
        latitude: Latitude in degrees north (range: -90 to 90).
            For numpy: scalar float
            For xarray: scalar float or DataArray(lat,) for spatial broadcasting
        data_start_year: Initial year of the input dataset. If None and temperature
            is a DataArray with datetime coordinate, will be inferred from the first timestamp.
        time_dim: Name of the time dimension in the input DataArray (default: "time").
            Only used for xarray inputs.

    Returns:
        PET values in mm/month, same shape and type as input temperature:
        - numpy input → numpy array output
        - xarray input → xarray DataArray output with CF metadata and provenance

    Raises:
        InputTypeError: If temperature is not numpy-coercible or xr.DataArray
        CoordinateValidationError: If time dimension missing/invalid (xarray path)
        ValueError: If latitude is out of range [-90, 90]

    Examples:
        >>> # NumPy path: 40 years of monthly temps at single location
        >>> temps = np.random.uniform(10, 25, 480)
        >>> pet = pet_thornthwaite(temps, latitude=40.0, data_start_year=1980)
        >>> pet.shape
        (480,)

        >>> # xarray path: 1-D time series
        >>> temp_da = xr.DataArray(
        ...     temps,
        ...     coords={'time': pd.date_range('1980-01', periods=480, freq='MS')},
        ...     dims=['time']
        ... )
        >>> pet_da = pet_thornthwaite(temp_da, latitude=40.0)
        >>> pet_da.attrs['long_name']
        'Potential Evapotranspiration (Thornthwaite method)'

        >>> # xarray path: gridded data with spatial broadcasting
        >>> temp_grid = xr.DataArray(
        ...     np.random.uniform(10, 25, (480, 4, 3)),
        ...     coords={
        ...         'time': pd.date_range('1980-01', periods=480, freq='MS'),
        ...         'lat': [30, 35, 40, 45],
        ...         'lon': [-120, -110, -100]
        ...     },
        ...     dims=['time', 'lat', 'lon']
        ... )
        >>> lat_array = xr.DataArray([30, 35, 40, 45], dims=['lat'])
        >>> pet_grid = pet_thornthwaite(temp_grid, lat_array)
        >>> pet_grid.shape
        (480, 4, 3)

    Notes:
        - The underlying indices.pet() function expects 1-D temperature arrays and
          scalar latitude. For gridded inputs, xr.apply_ufunc with vectorize=True
          automatically loops over non-time dimensions.
        - Dask-backed DataArrays remain lazy (dask="parallelized")
        - CF Convention metadata and provenance history are automatically applied
          to xarray outputs
        - NaN values in temperature are propagated through the calculation
    """
    # detect input type for routing
    input_type = detect_input_type(temperature)

    # validate latitude range for all paths
    _validate_latitude_range(latitude)

    # numpy passthrough
    if input_type == InputType.NUMPY:
        # validate latitude is scalar when temperature is numpy
        if isinstance(latitude, xr.DataArray):
            raise TypeError(
                "latitude must be a scalar (float, int, or numpy scalar) when "
                "temperature is a numpy array. "
                f"Got xr.DataArray with dims={latitude.dims}. "
                "Use a scalar latitude or convert temperature to xr.DataArray "
                "for spatial broadcasting."
            )
        # convert latitude to float if it's a numpy scalar
        lat_float = float(latitude) if isinstance(latitude, np.floating) else latitude
        # delegate to indices.pet with explicit data_start_year requirement
        if data_start_year is None:
            raise ValueError("data_start_year is required for numpy inputs")
        # narrow type for mypy
        assert isinstance(temperature, np.ndarray)
        return indices.pet(temperature, lat_float, data_start_year)

    # xarray path: validate → infer → compute → rewrap
    # at this point temperature must be an xr.DataArray (numpy path returned above)
    assert isinstance(temperature, xr.DataArray)
    temp_da = temperature

    # validate time dimension
    _validate_time_dimension(temp_da, time_dim)
    time_coord = temp_da.coords[time_dim]
    _validate_time_monotonicity(time_coord)

    # infer data_start_year if not provided
    if data_start_year is None:
        data_start_year = _infer_data_start_year(time_coord)
        _log().debug(
            "pet_data_start_year_inferred",
            inferred_year=data_start_year,
            time_coord_first=str(time_coord.values[0]),
        )

    # normalize latitude for xr.apply_ufunc
    # convert scalar numpy types to python float for compatibility
    if isinstance(latitude, float | int | np.floating | np.integer):
        lat_for_ufunc: float | xr.DataArray = float(latitude)
    else:
        # assume it's already an xr.DataArray
        lat_for_ufunc = latitude

    # wrapper function to handle read-only array views from apply_ufunc
    # the underlying eto.eto_thornthwaite modifies the temp array in-place,
    # so we must create a writable copy
    def _pet_with_copy(temps: np.ndarray, lat: float, year: int) -> np.ndarray:
        """Wrapper for indices.pet that creates a writable copy of temps."""
        return indices.pet(temps.copy(), lat, year)

    # compute using xr.apply_ufunc with spatial broadcasting
    # input_core_dims: temperature's time dim is "core", latitude and year are scalars per iteration
    # output_core_dims: preserve time dimension in output
    # vectorize=True: loop over non-core dims (lat, lon) calling indices.pet per gridpoint
    # dask_gufunc_kwargs: allow_rechunk=True permits chunked core dimensions (for dask arrays)
    result = xr.apply_ufunc(
        _pet_with_copy,
        temp_da,
        lat_for_ufunc,
        data_start_year,
        input_core_dims=[[time_dim], [], []],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[float],
    )

    # restore original dimension order (apply_ufunc places output core dims last)
    result = result.transpose(*temp_da.dims)

    # apply CF metadata from registry
    cf_attrs = CF_METADATA["pet_thornthwaite"]
    result.attrs.update(cf_attrs)

    # copy over non-conflicting attributes from input
    for key, value in temp_da.attrs.items():
        if key not in result.attrs:
            result.attrs[key] = value

    # add version attribute
    from climate_indices import __version__

    result.attrs["climate_indices_version"] = __version__

    # build and append history entry
    # serialize latitude for history
    if isinstance(lat_for_ufunc, xr.DataArray):
        lat_desc = f"DataArray(dims={lat_for_ufunc.dims})"
    else:
        lat_desc = str(lat_for_ufunc)

    history_entry = _build_history_entry(
        "PET Thornthwaite",
        __version__,
        {"latitude": lat_desc, "data_start_year": data_start_year},
    )
    result.attrs["history"] = _append_history(temp_da.attrs, history_entry)

    # add calculation metadata as attributes
    result.attrs["latitude"] = _build_latitude_attr(lat_for_ufunc)
    result.attrs["data_start_year"] = data_start_year

    _log().info(
        "pet_thornthwaite_completed",
        input_shape=temp_da.shape,
        output_shape=result.shape,
        data_start_year=data_start_year,
        latitude=lat_desc,
    )

    result_array: xr.DataArray = result
    return result_array


def pet_hargreaves(
    daily_tmin_celsius: np.ndarray | xr.DataArray,
    daily_tmax_celsius: np.ndarray | xr.DataArray,
    latitude: float | np.floating | xr.DataArray,
    time_dim: str = "time",
) -> np.ndarray | xr.DataArray:
    """Compute potential evapotranspiration using Hargreaves method.

    This function provides xarray DataArray support for the Hargreaves PET calculation.
    Unlike Thornthwaite (monthly), Hargreaves uses daily min/max temperature data.
    The mean temperature is automatically derived as (tmin + tmax) / 2.

    .. warning:: **Beta Feature (xarray path)** — When called with ``xr.DataArray``
       input, this function uses the beta xarray adapter layer. The NumPy array
       interface and underlying computation are stable.

    Args:
        daily_tmin_celsius: Daily minimum temperature values in degrees Celsius.
            For numpy: 1-D array of daily temperatures
            For xarray: DataArray with time dimension (may have additional spatial dims)
        daily_tmax_celsius: Daily maximum temperature values in degrees Celsius.
            Must align with daily_tmin_celsius for xarray inputs.
            For numpy: 1-D array of daily temperatures
            For xarray: DataArray with time dimension (may have additional spatial dims)
        latitude: Latitude in degrees north (range: -90 to 90).
            For numpy: scalar float
            For xarray: scalar float or DataArray(lat,) for spatial broadcasting
        time_dim: Name of the time dimension in the input DataArray (default: "time").
            Only used for xarray inputs.

    Returns:
        PET values in mm/day, same shape and type as input temperature:
        - numpy input → numpy array output
        - xarray input → xarray DataArray output with CF metadata and provenance

    Raises:
        InputTypeError: If temperature inputs are not numpy-coercible or xr.DataArray
        CoordinateValidationError: If time dimension missing/invalid or tmin/tmax don't align
        InputAlignmentWarning: If tmin/tmax have different time coordinates (auto-aligned)
        ValueError: If latitude is out of range [-90, 90]

    Examples:
        >>> # NumPy path: 5 years of daily temps at single location
        >>> tmin = np.random.uniform(5, 15, 1825)
        >>> tmax = np.random.uniform(15, 30, 1825)
        >>> pet = pet_hargreaves(tmin, tmax, latitude=40.0)
        >>> pet.shape
        (1825,)

        >>> # xarray path: 1-D time series
        >>> tmin_da = xr.DataArray(
        ...     tmin,
        ...     coords={'time': pd.date_range('2015-01-01', periods=1825, freq='D')},
        ...     dims=['time']
        ... )
        >>> tmax_da = xr.DataArray(
        ...     tmax,
        ...     coords={'time': pd.date_range('2015-01-01', periods=1825, freq='D')},
        ...     dims=['time']
        ... )
        >>> pet_da = pet_hargreaves(tmin_da, tmax_da, latitude=40.0)
        >>> pet_da.attrs['long_name']
        'Potential Evapotranspiration (Hargreaves method)'

        >>> # xarray path: gridded data with spatial broadcasting
        >>> tmin_grid = xr.DataArray(
        ...     np.random.uniform(5, 15, (1825, 4, 3)),
        ...     coords={
        ...         'time': pd.date_range('2015-01-01', periods=1825, freq='D'),
        ...         'lat': [30, 35, 40, 45],
        ...         'lon': [-120, -110, -100]
        ...     },
        ...     dims=['time', 'lat', 'lon']
        ... )
        >>> tmax_grid = xr.DataArray(
        ...     np.random.uniform(15, 30, (1825, 4, 3)),
        ...     coords={
        ...         'time': pd.date_range('2015-01-01', periods=1825, freq='D'),
        ...         'lat': [30, 35, 40, 45],
        ...         'lon': [-120, -110, -100]
        ...     },
        ...     dims=['time', 'lat', 'lon']
        ... )
        >>> lat_array = xr.DataArray([30, 35, 40, 45], dims=['lat'])
        >>> pet_grid = pet_hargreaves(tmin_grid, tmax_grid, lat_array)
        >>> pet_grid.shape
        (1825, 4, 3)

    Notes:
        - The underlying eto.eto_hargreaves() expects 1-D arrays and scalar latitude.
          For gridded inputs, xr.apply_ufunc with vectorize=True loops over spatial dims.
        - For xarray inputs with misaligned time coordinates, xr.align(join='inner')
          is automatically applied, and InputAlignmentWarning is emitted if timesteps differ.
        - Mean temperature is auto-derived: tmean = (tmin + tmax) / 2
        - Dask-backed DataArrays remain lazy (dask="parallelized")
        - CF Convention metadata and provenance history are automatically applied
        - NaN values in temperature are propagated through the calculation
    """
    # detect input type for routing
    input_type = detect_input_type(daily_tmin_celsius)

    # validate tmin and tmax are same input type (Fix 2)
    tmax_input_type = detect_input_type(daily_tmax_celsius)
    if input_type != tmax_input_type:
        raise TypeError(
            "daily_tmin_celsius and daily_tmax_celsius must be the same type. "
            f"Got tmin={input_type.name}, tmax={tmax_input_type.name}. "
            "Convert both to the same type (both numpy arrays or both xr.DataArray)."
        )

    # validate latitude range for all paths (Fix 3)
    _validate_latitude_range(latitude)

    # numpy passthrough
    if input_type == InputType.NUMPY:
        # validate latitude is scalar when temperature inputs are numpy
        if isinstance(latitude, xr.DataArray):
            raise TypeError(
                "latitude must be a scalar (float, int, or numpy scalar) when "
                "temperature inputs are numpy arrays. "
                f"Got xr.DataArray with dims={latitude.dims}. "
                "Use a scalar latitude or convert temperature inputs to xr.DataArray "
                "for spatial broadcasting."
            )
        # convert latitude to float if it's a numpy scalar
        lat_float = float(latitude) if isinstance(latitude, np.floating) else latitude
        # auto-derive tmean as per Hargreaves standard approach
        # narrow types for mypy
        assert isinstance(daily_tmin_celsius, np.ndarray)
        assert isinstance(daily_tmax_celsius, np.ndarray)
        tmean = (daily_tmin_celsius + daily_tmax_celsius) / 2.0
        # delegate to eto.eto_hargreaves
        return eto.eto_hargreaves(daily_tmin_celsius, daily_tmax_celsius, tmean, lat_float)

    # xarray path: validate → align → compute → rewrap
    # at this point both inputs must be xr.DataArray (numpy path returned above)
    assert isinstance(daily_tmin_celsius, xr.DataArray)
    assert isinstance(daily_tmax_celsius, xr.DataArray)
    tmin_da = daily_tmin_celsius
    tmax_da = daily_tmax_celsius

    # validate time dimension on both inputs
    _validate_time_dimension(tmin_da, time_dim)
    _validate_time_dimension(tmax_da, time_dim)
    tmin_time_coord = tmin_da.coords[time_dim]
    tmax_time_coord = tmax_da.coords[time_dim]
    _validate_time_monotonicity(tmin_time_coord)
    _validate_time_monotonicity(tmax_time_coord)

    # align tmin and tmax along time dimension (inner join)
    # this handles cases where they have different time ranges
    tmin_aligned, tmax_aligned = xr.align(tmin_da, tmax_da, join="inner")

    # warn if alignment dropped timesteps
    original_tmin_len = len(tmin_time_coord)
    original_tmax_len = len(tmax_time_coord)
    aligned_len = len(tmin_aligned.coords[time_dim])

    if aligned_len < original_tmin_len or aligned_len < original_tmax_len:
        warnings.warn(
            f"Input alignment: tmin had {original_tmin_len} timesteps, "
            f"tmax had {original_tmax_len} timesteps. "
            f"After inner join, {aligned_len} timesteps remain. "
            f"Non-overlapping timesteps were dropped.",
            InputAlignmentWarning,
            stacklevel=2,
        )

    # raise error if no overlap
    if aligned_len == 0:
        raise CoordinateValidationError(
            f"No overlapping timesteps found between tmin and tmax along '{time_dim}' dimension. "
            f"Cannot proceed with PET calculation."
        )

    # derive tmean
    tmean_da = (tmin_aligned + tmax_aligned) / 2.0

    # normalize latitude for xr.apply_ufunc
    if isinstance(latitude, float | int | np.floating | np.integer):
        lat_for_ufunc: float | xr.DataArray = float(latitude)
    else:
        # assume it's already an xr.DataArray
        lat_for_ufunc = latitude

    # wrapper function to handle read-only array views from apply_ufunc
    # eto.eto_hargreaves may modify arrays in-place, so create writable copies
    def _hargreaves_with_copy(tmin: np.ndarray, tmax: np.ndarray, tmean: np.ndarray, lat: float) -> np.ndarray:
        """Wrapper for eto.eto_hargreaves that creates writable copies."""
        return eto.eto_hargreaves(tmin.copy(), tmax.copy(), tmean.copy(), lat)

    # compute using xr.apply_ufunc with spatial broadcasting
    result = xr.apply_ufunc(
        _hargreaves_with_copy,
        tmin_aligned,
        tmax_aligned,
        tmean_da,
        lat_for_ufunc,
        input_core_dims=[[time_dim], [time_dim], [time_dim], []],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[float],
    )

    # restore original dimension order (apply_ufunc places output core dims last)
    # if latitude was a DataArray with extra dims, result will have those too
    # prioritize tmin dimensions, then any extra dimensions from latitude broadcast
    desired_dims = list(tmin_aligned.dims) + [d for d in result.dims if d not in tmin_aligned.dims]
    result = result.transpose(*desired_dims)

    # apply CF metadata from registry
    cf_attrs = CF_METADATA["pet_hargreaves"]
    result.attrs.update(cf_attrs)

    # copy over non-conflicting attributes from tmin (primary input)
    for key, value in tmin_aligned.attrs.items():
        if key not in result.attrs:
            result.attrs[key] = value

    # add version attribute
    from climate_indices import __version__

    result.attrs["climate_indices_version"] = __version__

    # build and append history entry
    # serialize latitude for history
    if isinstance(lat_for_ufunc, xr.DataArray):
        lat_desc = f"DataArray(dims={lat_for_ufunc.dims})"
    else:
        lat_desc = str(lat_for_ufunc)

    history_entry = _build_history_entry(
        "PET Hargreaves",
        __version__,
        {"latitude": lat_desc},
    )
    result.attrs["history"] = _append_history(tmin_aligned.attrs, history_entry)

    # add calculation metadata as attributes
    result.attrs["latitude"] = _build_latitude_attr(lat_for_ufunc)

    _log().info(
        "pet_hargreaves_completed",
        input_shape=tmin_aligned.shape,
        output_shape=result.shape,
        latitude=lat_desc,
    )

    result_array: xr.DataArray = result
    return result_array

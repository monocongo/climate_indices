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
import datetime
import functools
import inspect
import warnings
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import compute
from climate_indices.compute import MIN_CALIBRATION_YEARS
from climate_indices.exceptions import (
    CoordinateValidationError,
    InputAlignmentWarning,
    InputTypeError,
    InsufficientDataError,
)
from climate_indices.logging_config import get_logger


def _log():
    """Return a logger resolved at call time.

    Tests reset structlog globals between cases. Resolving lazily avoids
    holding a stale logger that bypasses stdlib handlers/capture after reset.
    """
    return get_logger(__name__)


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
    "spei": {
        "long_name": "Standardized Precipitation Evapotranspiration Index",
        "units": "dimensionless",
        "references": (
            "Vicente-Serrano, S. M., Begueria, S., & Lopez-Moreno, J. I. (2010). "
            "A Multiscalar Drought Index Sensitive to Global Warming: "
            "The Standardized Precipitation Evapotranspiration Index. "
            "Journal of Climate, 23(7), 1696-1718. "
            "https://doi.org/10.1175/2009JCLI2909.1"
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

# history attribute formatting
_HISTORY_SEPARATOR = "\n"
_HISTORY_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


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

    dim_name = time_coord.dims[0] if time_coord.dims else "time"
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
    freq = xr.infer_freq(time_coord)
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

    # infer calibration period if not provided
    if "calibration_year_initial" in sig.parameters and "calibration_year_initial" not in provided_params:
        cal_start, cal_end = _infer_calibration_period(time_coord)
        if "calibration_year_initial" not in provided_params:
            inferred["calibration_year_initial"] = cal_start
        if "calibration_year_final" in sig.parameters and "calibration_year_final" not in provided_params:
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
    # data.chunks is a dict mapping dimension name to tuple of chunk sizes
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


def _prepare_xarray_inputs(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    input_da: xr.DataArray,
    time_dim: str,
    additional_input_names: list[str] | None,
) -> tuple[xr.DataArray, list[Any], dict[str, Any], dict[str, tuple[int | None, Any]]]:
    """Resolve, align, and stage xarray inputs before computation."""
    modified_args = list(args)
    modified_kwargs = dict(kwargs)
    resolved_secondaries: dict[str, tuple[int | None, Any]] = {}

    if not additional_input_names:
        return input_da, modified_args, modified_kwargs, resolved_secondaries

    resolved_secondaries = _resolve_secondary_inputs(func, args, kwargs, additional_input_names)
    dataarray_secondaries = {
        name: value for name, (_, value) in resolved_secondaries.items() if isinstance(value, xr.DataArray)
    }
    if not dataarray_secondaries:
        return input_da, modified_args, modified_kwargs, resolved_secondaries

    aligned_primary, aligned_secondaries = _align_inputs(input_da, dataarray_secondaries, time_dim)
    input_da = aligned_primary
    modified_args[0] = aligned_primary

    for name, (pos_index, _) in resolved_secondaries.items():
        if name not in aligned_secondaries:
            continue
        if pos_index is not None:
            modified_args[pos_index] = aligned_secondaries[name]
        else:
            modified_kwargs[name] = aligned_secondaries[name]

    return input_da, modified_args, modified_kwargs, resolved_secondaries


def _validate_inference_inputs(
    infer_params: bool,
    input_da: xr.DataArray,
    time_dim: str,
    func: Callable[..., Any],
    modified_args: list[Any],
    modified_kwargs: dict[str, Any],
) -> None:
    """Run validation checks required for parameter inference."""
    if not infer_params:
        return

    _validate_time_dimension(input_da, time_dim)
    time_coord = input_da[time_dim]
    _validate_time_monotonicity(time_coord)
    resolved_scale = _resolve_scale_from_args(func, tuple(modified_args), modified_kwargs)
    if resolved_scale is not None:
        _validate_sufficient_data(time_coord, resolved_scale)


def _extract_secondary_dataarray_values(
    additional_input_names: list[str] | None,
    resolved_secondaries: dict[str, tuple[int | None, Any]],
    modified_args: list[Any],
    modified_kwargs: dict[str, Any],
) -> None:
    """Replace aligned secondary DataArray inputs with NumPy values in-place."""
    if not additional_input_names:
        return

    for name, (pos_index, value) in resolved_secondaries.items():
        if not isinstance(value, xr.DataArray):
            continue
        if pos_index is not None:
            modified_args[pos_index] = modified_args[pos_index].values
        else:
            modified_kwargs[name] = modified_kwargs[name].values


def _get_provided_params(sig: inspect.Signature, args: list[Any], kwargs: dict[str, Any]) -> set[str]:
    """Return provided parameter names from bound args/kwargs."""
    try:
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return set(bound.arguments.keys())
    except TypeError:
        return set()


def _infer_call_kwargs(
    func: Callable[..., Any],
    modified_args: list[Any],
    modified_kwargs: dict[str, Any],
    infer_params: bool,
    input_da: xr.DataArray,
    time_dim: str,
) -> dict[str, Any]:
    """Build call kwargs, filling inferable values when needed."""
    call_kwargs = dict(modified_kwargs)
    if not (infer_params and time_dim in input_da.dims):
        return call_kwargs

    time_coord = input_da[time_dim]
    sig = inspect.signature(func)
    provided_params = _get_provided_params(sig, modified_args, modified_kwargs)

    if "data_start_year" in sig.parameters and "data_start_year" not in provided_params:
        call_kwargs["data_start_year"] = _infer_data_start_year(time_coord)

    if "periodicity" in sig.parameters and "periodicity" not in provided_params:
        call_kwargs["periodicity"] = _infer_periodicity(time_coord)

    if "calibration_year_initial" in sig.parameters and "calibration_year_initial" not in provided_params:
        cal_start, cal_end = _infer_calibration_period(time_coord)
        call_kwargs["calibration_year_initial"] = cal_start
        if "calibration_year_final" in sig.parameters and "calibration_year_final" not in provided_params:
            call_kwargs["calibration_year_final"] = cal_end

    return call_kwargs


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


def _run_xarray_path(
    func: Callable[..., np.ndarray[Any, Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    data: xr.DataArray,
    *,
    cf_metadata: dict[str, str] | None,
    time_dim: str,
    infer_params: bool,
    calculation_metadata_keys: list[str] | tuple[str, ...] | None,
    index_display_name: str | None,
    additional_input_names: list[str] | None,
    skipna: bool,
) -> xr.DataArray:
    """Execute the xarray adaptation flow for a wrapped function call."""
    # check skipna parameter (Story 2.8)
    if skipna:
        raise NotImplementedError(
            "skipna=True not yet implemented (FR-INPUT-004). "
            "NaN values are propagated through calculations by default."
        )

    input_da, modified_args, modified_kwargs, resolved_secondaries = _prepare_xarray_inputs(
        func, args, kwargs, data, time_dim, additional_input_names
    )

    _validate_inference_inputs(infer_params, input_da, time_dim, func, modified_args, modified_kwargs)

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

    numpy_values = input_da.values
    _extract_secondary_dataarray_values(additional_input_names, resolved_secondaries, modified_args, modified_kwargs)
    call_kwargs = _infer_call_kwargs(func, modified_args, modified_kwargs, infer_params, input_da, time_dim)

    # validate calibration period has sufficient non-NaN data (Story 2.8)
    if nan_assessment["has_nan"] and infer_params and time_dim in input_da.dims:
        time_coord = input_da[time_dim]
        cal_initial = call_kwargs.get("calibration_year_initial")
        cal_final = call_kwargs.get("calibration_year_final")
        if cal_initial is not None and cal_final is not None:
            _validate_calibration_non_nan_sample_size(
                time_coord,
                numpy_values,
                calibration_year_initial=cal_initial,
                calibration_year_final=cal_final,
            )

    sig = inspect.signature(func)
    valid_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
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

    calc_metadata = _capture_calculation_metadata(calculation_metadata_keys, valid_kwargs)
    resolved_index_name = index_display_name if index_display_name is not None else func.__name__.upper()
    result_da = _build_output_dataarray(input_da, result_values, cf_metadata, calc_metadata, index_name=resolved_index_name)

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
                # primary + resolved secondaries
                input_dataarrays = [input_da]
                if additional_input_names:
                    resolved_secondaries = _resolve_secondary_inputs(func, args, kwargs, additional_input_names)
                    # get parameter order from signature
                    param_names = list(sig.parameters.keys())
                    for name in additional_input_names:
                        if name in resolved_secondaries:
                            _, value = resolved_secondaries[name]
                            if isinstance(value, xr.DataArray):
                                # use aligned version from modified_args/modified_kwargs
                                found = False
                                for pos_idx, arg_val in enumerate(modified_args):
                                    if pos_idx > 0 and isinstance(arg_val, xr.DataArray):
                                        # check if this corresponds to the secondary
                                        param_idx = param_names.index(name) if name in param_names else -1
                                        if param_idx == pos_idx:
                                            input_dataarrays.append(arg_val)
                                            found = True
                                            break
                                if not found and name in modified_kwargs:
                                    input_dataarrays.append(modified_kwargs[name])

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

                # restore dimension order (apply_ufunc moves core dims to end)
                if result_da.dims != input_da.dims:
                    result_da = result_da.transpose(*input_da.dims)

                # apply metadata using _build_output_attrs
                calc_metadata: dict[str, Any] | None = None
                if calculation_metadata_keys is not None:
                    calc_metadata = {}
                    for key in calculation_metadata_keys:
                        if key in valid_kwargs:
                            calc_metadata[key] = valid_kwargs[key]

                resolved_index_name = index_display_name if index_display_name is not None else func.__name__.upper()
                output_attrs = _build_output_attrs(input_da, cf_metadata, calc_metadata, index_name=resolved_index_name)
                result_da.attrs.update(output_attrs)

                # deep-copy coordinate attrs
                for coord_name in result_da.coords:
                    if coord_name in input_da.coords:
                        result_da.coords[coord_name].attrs = copy.deepcopy(input_da.coords[coord_name].attrs)

                # preserve .name
                result_da.name = input_da.name

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

            # build kwargs for the wrapped function
            # start with explicitly provided kwargs (including extracted secondaries)
            call_kwargs = dict(modified_kwargs)

            # apply inferred params (already computed above before the branch)
            call_kwargs.update(inferred_params)

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

            # filter call_kwargs to only include params the function accepts
            sig = inspect.signature(func)
            valid_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

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
            calc_metadata = None
            if calculation_metadata_keys is not None:
                calc_metadata = {}
                for key in calculation_metadata_keys:
                    if key in valid_kwargs:
                        calc_metadata[key] = valid_kwargs[key]

            # resolve index name for history tracking
            resolved_index_name = index_display_name if index_display_name is not None else func.__name__.upper()

            # rewrap result as DataArray with preserved coordinates/metadata
            result_da = _build_output_dataarray(
                input_da, result_values, cf_metadata, calc_metadata, index_name=resolved_index_name
            )
        return wrapper

    return decorator

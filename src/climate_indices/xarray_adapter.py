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

.. warning:: **Beta Feature** — The xarray adapter layer is beta through 3.0.0 and is
   promoted no earlier than 3.1.0. Its interface may change with a minor version,
   never in a patch release. The NumPy computation core (``indices.py``,
   ``compute.py``) is stable: no breaking changes occur in minor versions.
"""

from __future__ import annotations

import copy
import datetime
import functools
import inspect
import json
import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import structlog.stdlib
import xarray as xr

from climate_indices import compute, eto, indices, palmer, pm_eto, utils
from climate_indices.cf_metadata_registry import CF_METADATA, spi_output_attributes
from climate_indices.compute import MIN_CALIBRATION_YEARS
from climate_indices.exceptions import (
    CoordinateValidationError,
    InputAlignmentWarning,
    InputTypeError,
    InsufficientDataError,
    PeriodicityError,
)
from climate_indices.logging_config import get_logger
from climate_indices.validation import (
    InputType,
    detect_input_type,
    validate_dask_chunks,
    validate_time_dimension,
    validate_time_monotonicity,
)


def _log() -> structlog.stdlib.BoundLogger:
    """Return a logger resolved at call time.

    Tests reset structlog globals between cases. Resolving lazily avoids
    holding a stale logger that bypasses stdlib handlers/capture after reset.
    """
    return get_logger(__name__)


# history attribute formatting
_HISTORY_SEPARATOR = "\n"
_HISTORY_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


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


def _match_supported_periodicity(time_coord: xr.DataArray) -> compute.Periodicity | None:
    """Recognize the two supported calendar layouts without calling xr.infer_freq.

    xr.infer_freq accounts for roughly 97% of the calendar-contract check, which is
    charged on every xarray call, so the layouts the library actually supports are
    matched here with vectorized datetime64 arithmetic instead. Anything this does
    not recognize returns None so the caller can fall back to xr.infer_freq, whose
    exact frequency string is still wanted for error messages.

    Args:
        time_coord: xarray DataArray containing datetime values.

    Returns:
        Periodicity.monthly for consecutive month-start or month-end values,
        Periodicity.daily for consecutive calendar days, else None.
    """
    values = np.asarray(time_coord.values)
    if not np.issubdtype(values.dtype, np.datetime64):
        return None

    # a supported coordinate lands exactly on day boundaries; anything with a
    # sub-daily component (hourly, 12-hourly) must not be read as daily
    days = values.astype("datetime64[D]")
    if not np.array_equal(days.astype(values.dtype), values):
        return None

    day_offsets = days.astype("int64")
    deltas = np.diff(day_offsets)
    if deltas.size == 0:
        return None

    if np.all(deltas == 1):
        return compute.Periodicity.daily

    # month-length gaps alone are ambiguous, so require every value to sit on a
    # month start or every value to sit on a month end; combined with the gap
    # bound this rules out skipped months
    if not np.all((deltas >= 28) & (deltas <= 31)):
        return None

    months = days.astype("datetime64[M]")
    if np.all(days == months.astype("datetime64[D]")):
        return compute.Periodicity.monthly

    next_days = days + np.timedelta64(1, "D")
    if np.all(next_days.astype("datetime64[M]") != months):
        return compute.Periodicity.monthly

    return None


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

    supported_periodicity = _match_supported_periodicity(time_coord)
    if supported_periodicity is not None:
        return supported_periodicity

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


def _resolve_periodicity(
    func: Callable[..., Any],
    modified_args: list[Any],
    modified_kwargs: dict[str, Any],
    inferred_params: dict[str, Any],
) -> compute.Periodicity | None:
    """Resolve a wrapped function's explicit or inferred Periodicity.

    Returns None only when the wrapped function does not declare a periodicity
    parameter. A declared periodicity that cannot be resolved raises instead of
    silently skipping daily calendar conversion (see #759).

    Raises:
        PeriodicityError: If the wrapped function declares a periodicity parameter
            but its value cannot be resolved from the inferred or bound arguments.
    """
    signature = inspect.signature(func)
    if "periodicity" not in signature.parameters:
        return None

    periodicity = inferred_params.get("periodicity")
    if periodicity is None:
        try:
            bound = signature.bind_partial(*modified_args, **modified_kwargs)
            bound.apply_defaults()
        except TypeError as error:
            _log().error(
                "periodicity_resolution_failed",
                function_name=func.__name__,
                reason="argument_binding_failed",
            )
            raise PeriodicityError(
                message=(
                    f"Could not resolve the periodicity for {func.__name__}: its arguments do not bind to its "
                    "signature. Daily calendar conversion cannot be planned."
                ),
            ) from error
        periodicity = bound.arguments.get("periodicity")

    if not isinstance(periodicity, compute.Periodicity):
        _log().error(
            "periodicity_resolution_failed",
            function_name=func.__name__,
            reason="not_a_periodicity",
            periodicity_value=str(periodicity),
        )
        raise PeriodicityError(
            message=(
                f"Invalid periodicity argument: {periodicity}. "
                "Periodicity must be a Periodicity enum member. "
                "Supported values: monthly, daily. "
                "Use compute.Periodicity.monthly or compute.Periodicity.daily."
            ),
            periodicity_value=str(periodicity),
        )

    return periodicity


def _validate_supported_calendar(time_coord: xr.DataArray) -> None:
    """Reject calendar systems that the NumPy index core cannot represent."""
    coordinate_name = str(time_coord.name) if time_coord.name is not None else "time"
    # xarray records the calendar in .encoding on open_dataset and in .attrs on
    # hand-built coordinates, so consult both before falling back to the default
    calendar_name = time_coord.encoding.get("calendar", time_coord.attrs.get("calendar", "standard"))
    supported_calendars = {"standard", "gregorian", "proleptic_gregorian"}

    if not np.issubdtype(time_coord.dtype, np.datetime64) or calendar_name not in supported_calendars:
        raise CoordinateValidationError(
            message=(
                "Unsupported calendar: only standard, gregorian, and proleptic_gregorian datetime coordinates "
                "are supported; cftime calendars are not supported."
            ),
            coordinate_name=coordinate_name,
            reason="unsupported_calendar",
        )


def _build_daily_calendar_plan(
    time_coord: xr.DataArray,
    periodicity: compute.Periodicity,
) -> utils.DailyCalendarPlan | None:
    """Validate xarray calendar semantics and plan daily 366-day adaptation."""
    coordinate_name = str(time_coord.name) if time_coord.name is not None else "time"
    _validate_supported_calendar(time_coord)

    coordinate_periodicity = _infer_periodicity(time_coord)
    if coordinate_periodicity != periodicity:
        raise CoordinateValidationError(
            message=(
                f"Time coordinate has {coordinate_periodicity.name} periodicity, which does not match the "
                f"requested {periodicity.name} periodicity."
            ),
            coordinate_name=coordinate_name,
            reason="periodicity_mismatch",
        )

    first_timestamp = pd.Timestamp(time_coord.values[0])
    if periodicity == compute.Periodicity.monthly:
        if first_timestamp.month != 1:
            raise CoordinateValidationError(
                message="Monthly input must begin in January to preserve calendar-month semantics.",
                coordinate_name=coordinate_name,
                reason="unsupported_start_date",
            )
        return None

    if first_timestamp.month != 1 or first_timestamp.day != 1:
        raise CoordinateValidationError(
            message="Daily input must begin on January 1 to preserve calendar-day semantics.",
            coordinate_name=coordinate_name,
            reason="unsupported_start_date",
        )

    last_timestamp = pd.Timestamp(time_coord.values[-1])
    return utils.DailyCalendarPlan.from_year_span(
        first_timestamp.year,
        last_timestamp.year - first_timestamp.year + 1,
        len(time_coord),
    )


def _resolve_daily_calendar_plan(
    func: Callable[..., Any],
    input_da: xr.DataArray,
    modified_args: list[Any],
    modified_kwargs: dict[str, Any],
    inferred_params: dict[str, Any],
    time_dim: str,
) -> utils.DailyCalendarPlan | None:
    """Return daily calendar adaptation when the wrapped computation needs it."""
    periodicity = _resolve_periodicity(func, modified_args, modified_kwargs, inferred_params)
    if periodicity is None or time_dim not in input_da.dims:
        return None

    return _build_daily_calendar_plan(input_da[time_dim], periodicity)


def _validate_calendar_secondary_inputs(
    calendar_plan: utils.DailyCalendarPlan | None,
    resolved_secondaries: dict[str, tuple[int | None, Any]],
    time_dim: str,
) -> None:
    """Validate calendar-bearing secondary inputs before shared computation."""
    for name, (_, secondary) in resolved_secondaries.items():
        if isinstance(secondary, xr.DataArray):
            if time_dim in secondary.dims:
                _validate_supported_calendar(secondary[time_dim])
        elif calendar_plan is not None:
            raise InputTypeError(
                message=(
                    f"Daily xarray input requires '{name}' to be an xarray.DataArray so its calendar can be "
                    "validated and adapted."
                ),
                expected_type=xr.DataArray,
                actual_type=type(secondary),
            )


def _make_calendar_aware_numpy_wrapper(
    func: Callable[..., np.ndarray[Any, Any]],
    valid_kwargs: dict[str, Any],
    calendar_plan: utils.DailyCalendarPlan | None,
    core_axis_first: bool = False,
) -> Callable[..., np.ndarray[Any, Any]]:
    """Build an apply_ufunc callable that restores Gregorian daily output.

    Args:
        core_axis_first: True when ``func`` is a spatial kernel that reads the core
            dimension first, packed as ``(time, *cells)``. apply_ufunc always hands
            the core dimension over last, so the arrays are transposed into the
            kernel's layout here, and the kernel is told that its input is a declared
            time-major block rather than a plain time series.
    """
    if core_axis_first:
        # a block whose first cell axis is a calendar period length is ambiguous with a
        # (years, periods, *cells) array, so the kernel is told which reading this is
        valid_kwargs = {**valid_kwargs, "spatial_time_major": True}

    def wrapper(*numpy_arrays: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        # every positional argument here is a time series: _collect_input_dataarrays
        # yields only DataArrays, and apply_ufunc is called with one [time_dim] entry
        # in input_core_dims per collected array, so a non-time-series positional
        # would fail inside apply_ufunc before ever reaching this wrapper
        adapted = tuple(np.moveaxis(array, -1, 0) for array in numpy_arrays) if core_axis_first else numpy_arrays
        result = _compute_with_daily_calendar_plan(
            func,
            adapted,
            valid_kwargs,
            calendar_plan,
            set(range(len(adapted))),
            set(),
        )
        return np.moveaxis(result, 0, -1) if core_axis_first else result

    return wrapper


def _compute_with_daily_calendar_plan(
    func: Callable[..., np.ndarray[Any, Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    calendar_plan: utils.DailyCalendarPlan | None,
    time_series_arg_positions: set[int],
    time_series_kwarg_names: set[str],
) -> np.ndarray[Any, Any]:
    """Run a NumPy computation against 366-day values and restore Gregorian output."""
    if calendar_plan is None:
        return func(*args, **kwargs)

    adapted_args = list(args)
    for position in time_series_arg_positions:
        adapted_args[position] = calendar_plan.to_all_leap(adapted_args[position])

    adapted_kwargs = dict(kwargs)
    for name in time_series_kwarg_names:
        adapted_kwargs[name] = calendar_plan.to_all_leap(adapted_kwargs[name])

    all_leap_result = func(*adapted_args, **adapted_kwargs)
    # release the all-leap copies before the Gregorian restoration allocates its own,
    # so a spatial block does not hold both full-width sets of arrays at once
    del adapted_args, adapted_kwargs
    return calendar_plan.to_gregorian(all_leap_result)


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


def _build_latitude_attr(
    latitude: float | int | np.floating | np.integer | xr.DataArray,
) -> str | int | float | bool:
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


def _validate_sufficient_data(
    time_coord: xr.DataArray,
    scale: int,
    calendar_plan: utils.DailyCalendarPlan | None = None,
) -> None:
    """Validate that there is sufficient data for the given scale.

    Args:
        time_coord: Time coordinate DataArray
        scale: Scale parameter for the index calculation
        calendar_plan: Daily calendar adaptation, whose populated length is what the
            NumPy core can actually sum. Six complete Gregorian years (2000-2005)
            are 2192 observed days but fill all 2196 padded steps once their
            synthetic February 29 values are added, so the raw coordinate length
            would reject a scale of 2196 that the core can compute. A partial final
            year's padded NaN tail is excluded, since no window can sum through it.

    Raises:
        InsufficientDataError: If there are fewer time steps than the scale requires
    """
    n_timesteps = calendar_plan.populated_length if calendar_plan is not None else len(time_coord)
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
    *,
    warn_on_any_drop: bool = False,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Align primary and secondary DataArrays using inner join on coordinates.

    Ensures all input DataArrays share the same time coordinates by taking the
    intersection of their time ranges. This is essential for multi-input indices
    like SPEI where precipitation and PET must align.

    Args:
        primary: Primary input DataArray (e.g., precipitation)
        secondaries: Dict mapping parameter names to secondary DataArrays (e.g., {"pet": pet_da})
        time_dim: Name of the time dimension to align on (default: "time")
        warn_on_any_drop: Warn when any input loses time steps, not only the primary.
            For inputs of equal standing, such as the PET tmin/tmax pair.

    Returns:
        Tuple of (aligned_primary, dict_of_aligned_secondaries)

    Raises:
        CoordinateValidationError: If alignment results in empty intersection (no
            overlapping time steps), or if the inputs share a non-time dimension
            whose coordinates differ

    Warns:
        InputAlignmentWarning: If alignment drops time steps from the primary input,
            or from any input when ``warn_on_any_drop`` is set
    """
    if not secondaries:
        # no secondaries to align, return primary unchanged
        return primary, {}

    # collect all DataArrays for alignment
    all_arrays = [primary] + list(secondaries.values())

    # xr.align joins every shared dimension, not just the one named by time_dim: a
    # secondary on a different spatial grid would lose the cells the two grids do
    # not share, with no warning (only the time dimension is measured below). Time
    # overlap is expected and trimmed; a non-time dimension shared by more than one
    # input has to agree, order aside, before anything is aligned away.
    shared_dims = {dim for array in all_arrays for dim in array.dims} - {time_dim}
    for dim in sorted(shared_dims, key=str):
        arrays_with_dim = [array for array in all_arrays if dim in array.dims]
        # a dimension without a coordinate variable has no labels to compare; xr.align
        # matches it by position, so leave that case to xr.align
        if any(dim not in array.coords for array in arrays_with_dim):
            continue
        coordinate_indexes = [array[dim].to_index().sort_values() for array in arrays_with_dim]
        if len(coordinate_indexes) > 1 and any(
            not index.equals(coordinate_indexes[0]) for index in coordinate_indexes[1:]
        ):
            raise CoordinateValidationError(
                message=(
                    f"Inputs share the '{dim}' dimension but not its coordinates. "
                    f"Only the '{time_dim}' dimension may differ between inputs; matching "
                    f"cell coordinates are required, because intersecting the '{dim}' "
                    f"coordinates would silently drop cells from the result."
                ),
                coordinate_name=str(dim),
                reason="mismatched_non_time_coordinates",
            )

    # align using inner join (intersection of coordinates)
    aligned = xr.align(*all_arrays, join="inner")

    # extract aligned arrays
    aligned_primary = aligned[0]
    aligned_secondaries = {name: aligned[i + 1] for i, name in enumerate(secondaries.keys())}

    # check for empty intersection
    if time_dim in aligned_primary.dims:
        aligned_size = len(aligned_primary[time_dim])
        original_size = len(primary[time_dim])
        dropped_from = "primary input"
        if warn_on_any_drop:
            original_size = max(len(array[time_dim]) for array in all_arrays if time_dim in array.dims)
            dropped_from = "the longest input"

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
                f"Input alignment dropped {dropped_count} time step(s) from {dropped_from}. "
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
            # a Distribution carries the prose spelling ("log-logistic"); any other
            # enum falls back to its .name
            if isinstance(distribution, indices.Distribution):
                dist_name = distribution.display_name
            elif isinstance(distribution, Enum):
                dist_name = distribution.name
            else:
                dist_name = str(distribution)
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
        parameters that are in the function signature and not already provided,
        and is empty when the arguments do not bind to the signature.
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
        # arguments do not bind (unknown keyword, duplicate argument, ...):
        # skip inference so fabricated values cannot override explicit ones
        return inferred

    # infer data_start_year if not provided
    if "data_start_year" in sig.parameters and "data_start_year" not in provided_params:
        inferred["data_start_year"] = _infer_data_start_year(time_coord)

    # infer periodicity if not provided
    if "periodicity" in sig.parameters and "periodicity" not in provided_params:
        inferred["periodicity"] = _infer_periodicity(time_coord)

    # infer calibration period if either param is not provided
    # SPI/SPEI use calibration_year_initial/calibration_year_final
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

    # PNP uses calibration_start_year/calibration_end_year
    needs_cal_start = "calibration_start_year" in sig.parameters and "calibration_start_year" not in provided_params
    needs_cal_end = "calibration_end_year" in sig.parameters and "calibration_end_year" not in provided_params

    if needs_cal_start or needs_cal_end:
        cal_start, cal_end = _infer_calibration_period(time_coord)
        if needs_cal_start:
            inferred["calibration_start_year"] = cal_start
        if needs_cal_end:
            inferred["calibration_end_year"] = cal_end

    return inferred


def build_output_attrs(
    input_da: xr.DataArray,
    cf_metadata: dict[str, str] | None = None,
    calculation_metadata: dict[str, Any] | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    """Build output attributes with CF metadata, calculation metadata, version, and history.

    Pure attribute construction—centralizes the dict-building logic so every execution
    path applies identical metadata.

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
        if "standard_name" not in cf_metadata:
            # CF Standard Name Omission: an index with no registered standard
            # name must not inherit the input's (e.g. air_temperature), which
            # would misdescribe the computed result.
            output_attrs.pop("standard_name", None)

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


def _resolve_cf_metadata(
    cf_metadata: dict[str, str] | None,
    cf_metadata_variants: dict[str, dict[str, str]] | None,
    valid_kwargs: dict[str, Any],
    metadata_variant_parameter: str | None = None,
) -> dict[str, str] | None:
    """Select the CF metadata for the requested output scale, layering it over the base entry.

    A variant is keyed by the value of the wrapped function's ``output_scale``
    keyword (e.g. "probability"); an omitted or unregistered scale keeps the
    base metadata, and a variant overrides only the keys it sets.
    """
    if not cf_metadata_variants:
        return cf_metadata
    variant = cf_metadata_variants.get(
        valid_kwargs.get(metadata_variant_parameter, "normal") if metadata_variant_parameter else "normal"
    )
    if variant is None:
        return cf_metadata
    return {**(cf_metadata or {}), **variant}


def _finalize_ufunc_result(
    result: np.ndarray[Any, Any] | xr.DataArray,
    input_da: xr.DataArray,
    valid_kwargs: dict[str, Any],
    *,
    cf_metadata: dict[str, str] | None,
    cf_metadata_variants: dict[str, dict[str, str]] | None = None,
    metadata_variant_parameter: str | None = None,
    calculation_metadata_keys: list[str] | tuple[str, ...] | None,
    index_display_name: str | None,
    func_name: str,
    is_spi: bool = False,
) -> xr.DataArray:
    """Rewrap a computation result with the input's coords, dims, attrs, and name.

    Single finalizer for every execution path: NumPy results are wrapped using the
    input's coords/dims; DataArray results from apply_ufunc (whose core dims were moved
    to the end) are transposed back. Both then receive identical attributes and
    deep-copied coordinate attrs.

    Args:
        result: NumPy result array, or DataArray from apply_ufunc
        input_da: Original input DataArray
        valid_kwargs: Filtered kwargs passed to the wrapped function
        cf_metadata: CF convention metadata for the output
        cf_metadata_variants: Optional CF metadata keyed by ``output_scale`` value,
            overriding ``cf_metadata`` for the requested scale
        calculation_metadata_keys: Keys to extract from valid_kwargs for metadata
        index_display_name: Display name for the index (or None to use func_name.upper())
        func_name: Name of the wrapped function
        is_spi: Whether the wrapped function is indices.spi

    Returns:
        Finalized DataArray with restored dimensions, metadata, and coordinate attributes
    """
    if isinstance(result, xr.DataArray):
        result_da = result
        # restore dimension order (apply_ufunc moves core dims to end)
        if result_da.dims != input_da.dims:
            result_da = result_da.transpose(*input_da.dims)
    else:
        result_da = xr.DataArray(
            result,
            coords=input_da.coords,
            dims=input_da.dims,
        )

    # A ufunc can return input coordinate variables by reference; detach their
    # metadata without copying large result data before writing output attrs.
    result_da = cast(xr.DataArray, result_da.copy(deep=False))
    # apply metadata using build_output_attrs
    calc_metadata = _capture_calculation_metadata(calculation_metadata_keys, valid_kwargs)
    resolved_index_name = index_display_name if index_display_name is not None else func_name.upper()
    resolved_cf_metadata = _resolve_cf_metadata(
        cf_metadata, cf_metadata_variants, valid_kwargs, metadata_variant_parameter
    )
    output_attrs = build_output_attrs(input_da, resolved_cf_metadata, calc_metadata, index_name=resolved_index_name)
    if is_spi:
        compute.validate_output_scale(valid_kwargs.get("output_scale", "normal"))
        output_attrs.pop("valid_range", None)
        output_attrs.pop("actual_range", None)
        spi_attrs = spi_output_attributes(
            valid_kwargs.get("zero_handling", "classic"), valid_kwargs.get("output_scale", "normal")
        )
        # preserve a distinct caller-provided reference rather than replacing it
        caller_references = (resolved_cf_metadata or {}).get("references")
        spi_references = spi_attrs["references"]
        if caller_references and isinstance(spi_references, str) and caller_references not in spi_references:
            spi_attrs["references"] = f"{caller_references}; {spi_references}"
        output_attrs.update(spi_attrs)
    result_da.attrs = output_attrs

    # deep-copy coordinate attrs to prevent mutation bleed-through
    # (xarray currently preserves coord attrs through DataArray(coords=...),
    # but we defensively copy to ensure isolation)
    for coord_name in result_da.coords:
        if coord_name in input_da.coords:
            result_da.coords[coord_name].attrs = copy.deepcopy(input_da.coords[coord_name].attrs)

    # preserve .name
    result_da.name = input_da.name

    return result_da


# Registration-level coordinate inference; call sites select only the values their kernel accepts.
INFER_TIME_PARAMETERS: dict[str, Callable[[xr.DataArray], Any]] = {
    "data_start_year": _infer_data_start_year,
    "periodicity": _infer_periodicity,
    "calibration_year_initial": lambda time: _infer_calibration_period(time)[0],
    "calibration_year_final": lambda time: _infer_calibration_period(time)[1],
}


def xarray_adapter(
    *,
    calendar: compute.Periodicity | str | None = None,
    inferred_parameters: dict[str, Callable[[xr.DataArray], Any]] | None = None,
    argument_validators: tuple[Callable[[dict[str, Any]], None], ...] = (),
    deprecated_aliases: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    timescale_parameter: str | None = None,
    metadata_variant_parameter: str | None = None,
    cf_metadata: dict[str, str] | None = None,
    cf_metadata_variants: dict[str, dict[str, str]] | None = None,
    time_dim: str = "time",
    infer_params: bool = True,
    calculation_metadata_keys: list[str] | tuple[str, ...] | None = None,
    index_display_name: str | None = None,
    additional_input_names: list[str] | None = None,
    skipna: bool = False,
    spatial_kernel: bool = False,
    validate_calibration_sample: bool = True,
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
        calendar: Fixed periodicity, the parameter name holding an inferred/explicit
            periodicity, or None for no calendar conversion.
        inferred_parameters: Parameter names mapped to coordinate-based inference functions.
        argument_validators: Checks on bound arguments, run for NumPy and xarray before dispatch.
        deprecated_aliases: Translate deprecated keyword aliases before argument binding.
        timescale_parameter: Name of the timescale argument for the data-length check.
        metadata_variant_parameter: Name of the argument selecting the CF metadata
            variant. Required whenever ``cf_metadata_variants`` is given.
        cf_metadata: Optional dict of CF Convention metadata to apply to output DataArray.
            Keys should be CF attribute names (e.g., 'standard_name', 'long_name', 'units').
            These override conflicting attributes from the input DataArray.
        cf_metadata_variants: Optional mapping from an ``output_scale`` value
            (e.g. "probability") to the CF metadata for that output convention,
            overriding ``cf_metadata`` when that scale is requested.
        time_dim: Name of the time dimension in the input DataArray (default: "time").
            Used for parameter inference and alignment.
        infer_params: If True, infer the missing parameters declared in
            ``inferred_parameters`` from the time coordinate. Explicit parameter values
            always override inferred values; a registration that declares no
            ``inferred_parameters`` infers nothing.
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
        validate_calibration_sample: Enforce the fitting-based indices' 30-year
            non-NaN minimum. Disable for indices with their own calibration contract.
            Only checked for an in-memory 1-D (single time series) input. Gridded
            inputs skip this preflight check, as Dask inputs do; no per-cell
            30-year non-NaN minimum is enforced (see #1156).
        spatial_kernel: If True, the wrapped function accepts the core ``time_dim``
            dimension alongside any number of cell dimensions, packed as
            ``(time, *cells)``, so ``apply_ufunc`` makes one call per non-core block
            instead of one call per grid cell. Only inputs with more than one non-core
            dimension are packed this way; a 2-D input keeps the per-cell path. The
            kernel is told the block is time-major through a ``spatial_time_major``
            keyword, which it must accept. Indices whose kernels still loop over cells
            leave this False (see #940).

    Returns:
        Decorator function that wraps index computation functions

    Example:
        .. code-block:: python

            @xarray_adapter(
                cf_metadata={'standard_name': 'spi', 'units': '1'},
                calculation_metadata_keys=['scale', 'distribution']
            )
            def spi(values, scale, distribution, data_start_year, ...):
                # existing NumPy implementation
                return numpy_result

            # Works with both NumPy arrays and xarray DataArrays
            result_numpy = spi(np.array([...]), scale=3, ...)
            result_xarray = spi(precip_da, scale=3, distribution=Distribution.gamma)
            # result_xarray.attrs now includes: scale=3, distribution="gamma", climate_indices_version="x.y.z"

    Notes:
        - NumPy inputs: Passed through unchanged to the wrapped function
        - xarray inputs: Values extracted, parameters inferred, result rewrapped with coords
        - 1D and multi-dimensional DataArrays supported; parameter inference requires the configured
          ``time_dim``, and Dask-backed inputs must keep that dimension in a single chunk
          (see :doc:`xarray_migration`)
        - With ``spatial_kernel=True``, spatial DataArrays (more than one non-core
          dimension) reach the wrapped function as one ``(time, *cells)`` block, flagged
          through its ``spatial_time_major`` keyword, instead of one time series per cell
        - Uses inspect.signature() for generic parameter mapping (works with any function)
    """

    def decorator(func: Callable[..., np.ndarray[Any, Any]]) -> Callable[..., np.ndarray[Any, Any] | xr.DataArray]:
        signature = inspect.signature(func)
        declared = set(signature.parameters)
        if isinstance(calendar, str) and calendar not in declared:
            raise ValueError(f"Calendar parameter {calendar!r} is not accepted by {func.__name__}")
        unknown_inferences = set(inferred_parameters or {}) - declared
        if unknown_inferences:
            raise ValueError(f"Inferred parameters not accepted by {func.__name__}: {sorted(unknown_inferences)}")
        for contract_name in (timescale_parameter, metadata_variant_parameter):
            if contract_name is not None and contract_name not in declared:
                raise ValueError(f"Declared parameter {contract_name!r} is not accepted by {func.__name__}")
        if cf_metadata_variants and metadata_variant_parameter is None:
            raise ValueError(
                f"{func.__name__} declares cf_metadata_variants but no metadata_variant_parameter to select one"
            )

        def validate_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError:
                return  # Keep the wrapped function's existing binding error.
            bound.apply_defaults()
            for validator in argument_validators:
                validator(bound.arguments)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any] | xr.DataArray:
            # first positional argument is always the data
            if not args:
                raise ValueError(f"{func.__name__} requires at least one positional argument (data)")

            if deprecated_aliases is not None:
                kwargs = deprecated_aliases(kwargs)
            data = args[0]
            input_type = detect_input_type(data)

            # Run the same declared argument contract before NumPy dispatch or lazy graph creation.
            if input_type == InputType.NUMPY:
                validate_arguments(args, kwargs)
                return func(*args, **kwargs)

            # xarray path: detect → [resolve → align] → validate → extract → infer → compute → rewrap → log
            input_da = data

            # check skipna parameter
            if skipna:
                raise NotImplementedError(
                    "skipna=True not yet implemented (FR-INPUT-004). "
                    "NaN values are propagated through calculations by default."
                )

            # resolve and align secondary inputs
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
                    if calendar is not None:
                        if time_dim in input_da.dims:
                            _validate_supported_calendar(input_da[time_dim])
                        for secondary in dataarray_secondaries.values():
                            if time_dim in secondary.dims:
                                _validate_supported_calendar(secondary[time_dim])

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

            # coordinate validation
            if infer_params:
                validate_time_dimension(input_da, time_dim)
                time_coord = input_da[time_dim]
                if calendar is not None:
                    _validate_supported_calendar(time_coord)
                validate_time_monotonicity(time_coord)

            # detect Dask-backed arrays
            input_dataarrays = _collect_input_dataarrays(
                input_da,
                additional_input_names,
                resolved_secondaries,
                modified_args,
                modified_kwargs,
            )
            is_dask = any(dataarray.chunks is not None for dataarray in input_dataarrays)
            if is_dask:
                # validate chunking constraints for every Dask-backed time series
                for dataarray in input_dataarrays:
                    if dataarray.chunks is not None:
                        validate_dask_chunks(dataarray, time_dim)

            # infer temporal parameters if enabled (shared path)
            inferred_params: dict[str, Any] = {}
            if infer_params:
                if time_dim in input_da.dims:
                    try:
                        bound = signature.bind_partial(*modified_args, **modified_kwargs)
                        bound.apply_defaults()
                        inferred_params = {
                            name: infer(input_da[time_dim])
                            for name, infer in (inferred_parameters or {}).items()
                            if name not in bound.arguments
                        }
                    except TypeError:
                        pass  # Preserve the wrapped function's binding error.
                # log which parameters were inferred and their values
                if inferred_params:
                    _log().info(
                        "parameters_inferred",
                        function_name=func.__name__,
                        **{k: str(v) for k, v in inferred_params.items()},
                    )

            call_kwargs = {**modified_kwargs, **inferred_params}
            validate_arguments(tuple(modified_args), call_kwargs)
            calendar_plan = None
            if infer_params and calendar is not None and time_dim in input_da.dims:
                if isinstance(calendar, str):
                    try:
                        bound = signature.bind_partial(*modified_args, **call_kwargs)
                        bound.apply_defaults()
                    except TypeError as error:
                        raise PeriodicityError(
                            message=f"Could not resolve the periodicity for {func.__name__}: its arguments do not bind to its signature. Daily calendar conversion cannot be planned."
                        ) from error
                    periodicity = bound.arguments.get(calendar)
                    if not isinstance(periodicity, compute.Periodicity):
                        raise PeriodicityError(
                            message=f"Invalid periodicity argument: {periodicity}. Periodicity must be a Periodicity enum member. Supported values: monthly, daily. Use compute.Periodicity.monthly or compute.Periodicity.daily.",
                            periodicity_value=str(periodicity),
                        )
                else:
                    periodicity = calendar
                calendar_plan = _build_daily_calendar_plan(input_da[time_dim], periodicity)
                _validate_calendar_secondary_inputs(calendar_plan, resolved_secondaries, time_dim)
            if infer_params and timescale_parameter is not None and time_dim in input_da.dims:
                try:
                    bound = signature.bind_partial(*modified_args, **call_kwargs)
                    if timescale_parameter in bound.arguments:
                        _validate_sufficient_data(
                            input_da[time_dim], bound.arguments[timescale_parameter], calendar_plan
                        )
                except TypeError:
                    pass

            # spatial kernels read the core dimension first, with the cell dimensions
            # ahead of it, so apply_ufunc makes one call per non-core block instead of
            # one call per cell. A 2-D input has only a single dimension to broadcast
            # over, which stays on the per-cell path.
            use_spatial_kernel = spatial_kernel and input_da.ndim > 2

            # branch: Dask execution or in-memory execution
            if is_dask:
                # Dask execution path

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

                # create a calendar-aware callable for apply_ufunc
                _numpy_func_wrapper = _make_calendar_aware_numpy_wrapper(
                    func, valid_kwargs, calendar_plan, core_axis_first=use_spatial_kernel
                )

                # call apply_ufunc with Dask support
                result_da: xr.DataArray = xr.apply_ufunc(
                    _numpy_func_wrapper,
                    *input_dataarrays,
                    input_core_dims=[[time_dim]] * len(input_dataarrays),
                    output_core_dims=[[time_dim]],
                    dask="parallelized",
                    vectorize=not use_spatial_kernel,
                    output_dtypes=[float],
                )

                # finalize result: restore dims, attach metadata, copy coord attrs
                result_da = _finalize_ufunc_result(
                    result_da,
                    input_da,
                    valid_kwargs,
                    cf_metadata=cf_metadata,
                    cf_metadata_variants=cf_metadata_variants,
                    metadata_variant_parameter=metadata_variant_parameter,
                    calculation_metadata_keys=calculation_metadata_keys,
                    index_display_name=index_display_name,
                    func_name=func.__name__,
                    is_spi=func is indices.spi,
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

            # In-memory execution path (original logic)

            # assess NaN density for diagnostics
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

            # filter call_kwargs to only include params the function accepts. DataArray
            # secondaries are dropped here because apply_ufunc passes them positionally
            # alongside the primary; numpy secondaries stay, as they are keyword-only
            # inputs to the wrapped function.
            sig = inspect.signature(func)
            positional_secondaries = {
                name for name, (_, value) in resolved_secondaries.items() if isinstance(value, xr.DataArray)
            }
            valid_kwargs = {
                k: v for k, v in call_kwargs.items() if k in sig.parameters and k not in positional_secondaries
            }

            # check if input is multi-dimensional (has spatial dims beyond time)
            # and has a time dimension (required for apply_ufunc with input_core_dims)
            if input_da.ndim > 1 and time_dim in input_da.dims:
                # Multi-dimensional in-memory execution path
                # use xr.apply_ufunc with vectorize=True to handle spatial broadcasting
                # similar to Dask path but without dask="parallelized"

                # No whole-grid calibration preflight here (see #1156): a single
                # sample cell can't stand in for a grid. Like Dask, this path
                # does not enforce a per-cell 30-year non-NaN minimum.

                # collect input DataArrays for apply_ufunc in parameter order
                input_dataarrays = _collect_input_dataarrays(
                    input_da, additional_input_names, resolved_secondaries, modified_args, modified_kwargs
                )

                # create a calendar-aware callable for apply_ufunc
                _numpy_func_wrapper = _make_calendar_aware_numpy_wrapper(
                    func, valid_kwargs, calendar_plan, core_axis_first=use_spatial_kernel
                )

                # call apply_ufunc without Dask support (in-memory execution)
                result_da: xr.DataArray = xr.apply_ufunc(  # type: ignore[no-redef]
                    _numpy_func_wrapper,
                    *input_dataarrays,
                    input_core_dims=[[time_dim]] * len(input_dataarrays),
                    output_core_dims=[[time_dim]],
                    vectorize=not use_spatial_kernel,
                    output_dtypes=[float],
                )

                # finalize result: restore dims, attach metadata, copy coord attrs
                result_da = _finalize_ufunc_result(
                    result_da,
                    input_da,
                    valid_kwargs,
                    cf_metadata=cf_metadata,
                    cf_metadata_variants=cf_metadata_variants,
                    metadata_variant_parameter=metadata_variant_parameter,
                    calculation_metadata_keys=calculation_metadata_keys,
                    index_display_name=index_display_name,
                    func_name=func.__name__,
                    is_spi=func is indices.spi,
                )

                # log completion
                log_fields = {
                    "function_name": func.__name__,
                    "input_shape": input_da.shape,
                    "output_shape": result_da.shape,
                    "inferred_params": infer_params,
                    "vectorized": not use_spatial_kernel,
                    "spatial_kernel": use_spatial_kernel,
                }
                if nan_assessment["has_nan"]:
                    log_fields["input_nan_count"] = nan_assessment["nan_count"]
                    log_fields["input_nan_ratio"] = round(nan_assessment["nan_ratio"], 4)

                _log().info("xarray_adapter_completed", **log_fields)
                return result_da

            # 1D in-memory execution path

            # extract numpy values from primary
            numpy_values = input_da.values

            # extract numpy values from secondary DataArrays (if any)
            time_series_arg_positions = {0}
            time_series_kwarg_names: set[str] = set()
            if additional_input_names:
                for name, (pos_index, value) in resolved_secondaries.items():
                    if isinstance(value, xr.DataArray):
                        # extract .values from aligned DataArray
                        if pos_index is not None:
                            modified_args[pos_index] = modified_args[pos_index].values
                            time_series_arg_positions.add(pos_index)
                        else:
                            secondary_values = modified_kwargs[name].values
                            modified_kwargs[name] = secondary_values
                            valid_kwargs[name] = secondary_values
                            time_series_kwarg_names.add(name)

            # validate calibration period has sufficient non-NaN data
            # this validation requires .values, so it only runs in the in-memory path
            if nan_assessment["has_nan"] and infer_params and validate_calibration_sample and time_dim in input_da.dims:
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

            result_values = _compute_with_daily_calendar_plan(
                func,
                numpy_args,
                valid_kwargs,
                calendar_plan,
                time_series_arg_positions,
                time_series_kwarg_names,
            )

            # verify NaN propagation contract
            if nan_assessment["has_nan"]:
                if not _verify_nan_propagation(nan_assessment["nan_positions"], result_values):
                    _log().warning(
                        "nan_propagation_violation",
                        function_name=func.__name__,
                        message="Output missing NaN values present in input",
                    )

            # rewrap result as DataArray with preserved coordinates/metadata
            result_da = _finalize_ufunc_result(
                result_values,
                input_da,
                valid_kwargs,
                cf_metadata=cf_metadata,
                cf_metadata_variants=cf_metadata_variants,
                metadata_variant_parameter=metadata_variant_parameter,
                calculation_metadata_keys=calculation_metadata_keys,
                index_display_name=index_display_name,
                func_name=func.__name__,
                is_spi=func is indices.spi,
            )

            # log completion with NaN metrics
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


_CellParam = TypeVar("_CellParam")


def _spatial_kernel_cell_param(
    data: xr.DataArray,
    value: _CellParam,
    time_dim: str,
) -> tuple[bool, _CellParam]:
    """Decide whether ``data`` reaches a spatial (per-block) kernel, and with which per-cell parameter.

    Used by the broadcast inputs that are not time series: PET's latitude and Palmer's
    available water capacity (AWC). Returns ``(use_spatial_kernel, value_to_pass)``. The
    block path is skipped, and the value returned unchanged, when the input has a single
    non-core dimension (only one dimension to broadcast over) or when the value carries a
    dimension the input does not (so it cannot be read as a per-cell value at all).

    A DataArray value is transposed into the block's cell-dimension order, so the kernel
    reads its axes in that order; apply_ufunc appends a leading cell dimension it does not
    carry as a length-1 axis, and leaves a length-1 axis to numpy's right-aligned
    broadcasting, which the kernel matches.
    """
    if data.ndim <= 2:
        return False, value

    cell_dims = [dim for dim in data.dims if dim != time_dim]
    if not isinstance(value, xr.DataArray):
        return True, value
    if not set(value.dims) <= set(cell_dims):
        return False, value

    return True, value.transpose(*[dim for dim in cell_dims if dim in value.dims])


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
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from climate_indices import pet_thornthwaite
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
          scalar latitude. For gridded inputs, a (time, ``*cells``) block reaches it with
          the per-cell latitude array, so a 3-D or higher input costs one call per
          block rather than one xr.apply_ufunc call per grid cell.
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
    validate_time_dimension(temp_da, time_dim)
    time_coord = temp_da.coords[time_dim]
    validate_time_monotonicity(time_coord)

    # enforce the shared calendar contract: Thornthwaite groups values into calendar
    # months from a January origin, so validate before the start year is inferred.
    # monthly input needs no conversion, so the returned plan is always None here.
    _ = _build_daily_calendar_plan(time_coord, compute.Periodicity.monthly)

    # infer data_start_year if not provided
    if data_start_year is None:
        data_start_year = _infer_data_start_year(time_coord)
        _log().debug(
            "pet_data_start_year_inferred",
            inferred_year=data_start_year,
            time_coord_first=str(time_coord.values[0]),
        )

    # normalize latitude for xr.apply_ufunc: a gridded input reaches indices.pet as one
    # time-major block with the latitude per cell, instead of one call per grid cell; the
    # latitude is aligned with the temperature's cell axes so the kernel reads them in the
    # block's order
    use_spatial_kernel, lat_for_ufunc = _spatial_kernel_cell_param(temp_da, latitude, time_dim)

    # wrapper functions to handle read-only array views from apply_ufunc
    # the underlying eto.eto_thornthwaite modifies the temp array in-place,
    # so we must create a writable copy
    def _pet_with_copy(temps: np.ndarray, lat: np.ndarray, year: int) -> np.ndarray:
        """Wrapper for indices.pet that creates a writable copy of temps."""
        return indices.pet(temps.copy(), lat, year)

    def _pet_block(temps: np.ndarray, lat: np.ndarray, year: int) -> np.ndarray:
        """Wrapper for indices.pet that reads a (time, *cells) block with per-cell latitudes."""
        pet = indices.pet(np.moveaxis(temps, -1, 0).copy(), lat, year, spatial_time_major=True)
        return np.moveaxis(pet, 0, -1)

    # compute using xr.apply_ufunc with spatial broadcasting
    # input_core_dims: temperature's time dim is "core"; latitude and year arrive as
    #   scalars per iteration on the per-cell path and as cell arrays on the other
    # output_core_dims: preserve time dimension in output
    # vectorize=True: loop over non-core dims (lat, lon) calling indices.pet per gridpoint,
    # whereas the spatial kernel path hands the whole (time, *cells) block over at once
    # dask_gufunc_kwargs: allow_rechunk=True permits chunked core dimensions (for dask arrays)
    result = xr.apply_ufunc(
        _pet_block if use_spatial_kernel else _pet_with_copy,
        temp_da,
        lat_for_ufunc,
        data_start_year,
        input_core_dims=[[time_dim], [], []],
        output_core_dims=[[time_dim]],
        vectorize=not use_spatial_kernel,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[float],
    )

    # restore original dimension order (apply_ufunc places output core dims last);
    # a latitude broadcast over a dimension the temperature does not have adds that
    # dimension to the result, so keep the temperature's dims first
    desired_dims = list(temp_da.dims) + [dim for dim in result.dims if dim not in temp_da.dims]
    result = result.transpose(*desired_dims)

    # CF metadata, provenance and calculation attrs share one owner with the other indices
    result.attrs = build_output_attrs(
        temp_da,
        CF_METADATA["pet_thornthwaite"],  # type: ignore[arg-type]
        {"latitude": _build_latitude_attr(lat_for_ufunc), "data_start_year": data_start_year},
        "PET Thornthwaite",
    )

    if isinstance(latitude, xr.DataArray):
        lat_desc = f"DataArray(dims={latitude.dims})"
    else:
        lat_desc = str(lat_for_ufunc)

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
        CoordinateValidationError: If time dimension missing/invalid, tmin/tmax share no time
            steps, or they share a cell dimension with differing coordinates
        InputAlignmentWarning: If either input loses time steps to the shared time range (auto-aligned)
        ValueError: If latitude is out of range [-90, 90]

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from climate_indices import pet_hargreaves
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
          For gridded inputs, a (time, ``*cells``) block reaches it with the per-cell
          latitude array, so a 3-D or higher input costs one call per block rather
          than one xr.apply_ufunc call per grid cell.
        - For xarray inputs, only the time dimension is trimmed to the shared range, and
          InputAlignmentWarning is emitted if either input loses timesteps. A cell
          dimension with differing coordinates raises CoordinateValidationError rather
          than being intersected.
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
    validate_time_dimension(tmin_da, time_dim)
    validate_time_dimension(tmax_da, time_dim)
    tmin_time_coord = tmin_da.coords[time_dim]
    tmax_time_coord = tmax_da.coords[time_dim]
    validate_time_monotonicity(tmin_time_coord)
    validate_time_monotonicity(tmax_time_coord)

    # trim tmin and tmax to their shared time steps (warning when steps are dropped);
    # mismatched cell coordinates are rejected rather than silently intersected
    tmin_aligned, aligned_secondaries = _align_inputs(tmin_da, {"tmax": tmax_da}, time_dim, warn_on_any_drop=True)
    tmax_aligned = aligned_secondaries["tmax"]

    # enforce the shared calendar contract and plan the 366-day adaptation that
    # eto.eto_hargreaves assumes; built from the aligned coordinate, not the inputs
    calendar_plan = _build_daily_calendar_plan(tmin_aligned.coords[time_dim], compute.Periodicity.daily)

    # derive tmean
    tmean_da = (tmin_aligned + tmax_aligned) / 2.0

    # normalize latitude for xr.apply_ufunc: a gridded input reaches eto.eto_hargreaves as
    # one time-major block with the latitude per cell, instead of one call per grid cell;
    # the latitude is aligned with the temperature's cell axes so the kernel reads them in
    # the block's order
    use_spatial_kernel, lat_for_ufunc = _spatial_kernel_cell_param(tmin_aligned, latitude, time_dim)

    # wrapper functions to handle read-only array views from apply_ufunc
    # eto.eto_hargreaves may modify arrays in-place, so create writable copies
    def _eto_hargreaves_with_copy(tmin: np.ndarray, tmax: np.ndarray, tmean: np.ndarray, lat: float) -> np.ndarray:
        """Wrapper for eto.eto_hargreaves that creates writable copies."""
        return eto.eto_hargreaves(tmin.copy(), tmax.copy(), tmean.copy(), lat)

    def _hargreaves_with_copy(tmin: np.ndarray, tmax: np.ndarray, tmean: np.ndarray, lat: float) -> np.ndarray:
        """Compute Hargreaves ETo against 366-day positions and restore Gregorian output."""
        return _compute_with_daily_calendar_plan(
            _eto_hargreaves_with_copy,
            (tmin, tmax, tmean, lat),
            {},
            calendar_plan,
            # latitude at position 3 is a scalar, not a time series
            {0, 1, 2},
            set(),
        )

    def _hargreaves_block(tmin: np.ndarray, tmax: np.ndarray, tmean: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Compute Hargreaves ETo once per (time, *cells) block, with per-cell latitudes."""
        block = _compute_with_daily_calendar_plan(
            eto.eto_hargreaves,
            (
                np.moveaxis(tmin, -1, 0),
                np.moveaxis(tmax, -1, 0),
                np.moveaxis(tmean, -1, 0),
                lat,
            ),
            {"spatial_time_major": True},
            calendar_plan,
            {0, 1, 2},
            set(),
        )
        return np.moveaxis(block, 0, -1)

    # compute using xr.apply_ufunc with spatial broadcasting
    result = xr.apply_ufunc(
        _hargreaves_block if use_spatial_kernel else _hargreaves_with_copy,
        tmin_aligned,
        tmax_aligned,
        tmean_da,
        lat_for_ufunc,
        input_core_dims=[[time_dim], [time_dim], [time_dim], []],
        output_core_dims=[[time_dim]],
        vectorize=not use_spatial_kernel,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[float],
    )

    # restore original dimension order (apply_ufunc places output core dims last)
    # if latitude was a DataArray with extra dims, result will have those too
    # prioritize tmin dimensions, then any extra dimensions from latitude broadcast
    desired_dims = list(tmin_aligned.dims) + [d for d in result.dims if d not in tmin_aligned.dims]
    result = result.transpose(*desired_dims)

    # CF metadata, provenance and calculation attrs share one owner with the other indices
    result.attrs = build_output_attrs(
        tmin_aligned,
        CF_METADATA["pet_hargreaves"],  # type: ignore[arg-type]
        {"latitude": _build_latitude_attr(lat_for_ufunc)},
        "PET Hargreaves",
    )

    if isinstance(latitude, xr.DataArray):
        lat_desc = f"DataArray(dims={latitude.dims})"
    else:
        lat_desc = str(lat_for_ufunc)

    _log().info(
        "pet_hargreaves_completed",
        input_shape=tmin_aligned.shape,
        output_shape=result.shape,
        latitude=lat_desc,
    )

    result_array: xr.DataArray = result
    return result_array


def _align_penman_monteith_inputs(
    tmin: xr.DataArray,
    tmax: xr.DataArray,
    optional_inputs: dict[str, Any],
    time_dim: str,
) -> tuple[xr.DataArray, xr.DataArray, dict[str, Any]]:
    """Align meteorological inputs and infer days on the shared daily calendar."""
    time_bearing = [
        (name, value)
        for name, value in optional_inputs.items()
        if isinstance(value, xr.DataArray) and time_dim in value.dims
    ]
    # the shared alignment policy trims to the common time steps (warning when steps are
    # dropped) and rejects mismatched cell coordinates rather than silently intersecting
    tmin_aligned, aligned_secondaries = _align_inputs(
        tmin, {"tmax": tmax, **dict(time_bearing)}, time_dim, warn_on_any_drop=True
    )
    tmax_aligned = aligned_secondaries.pop("tmax")
    optional_inputs = {**optional_inputs, **aligned_secondaries}

    # enforce the shared January-start daily calendar contract used by the PET family
    _ = _build_daily_calendar_plan(tmin_aligned.coords[time_dim], compute.Periodicity.daily)
    if optional_inputs["day_of_year"] is None:
        optional_inputs["day_of_year"] = tmin_aligned[time_dim].dt.dayofyear
    return tmin_aligned, tmax_aligned, optional_inputs


def _penman_monteith_kernel_input(value: Any, time_dim: str) -> tuple[Any, list[str]]:
    """Return an apply_ufunc input and core dims, encoding absent inputs as NaN."""
    if value is None:
        return np.float64(np.nan), []
    if isinstance(value, xr.DataArray):
        return value, [time_dim] if time_dim in value.dims else []
    return value, []


def _finalize_penman_monteith_result(
    result: xr.DataArray,
    tmin_aligned: xr.DataArray,
    latitude: float | np.floating | xr.DataArray,
) -> xr.DataArray:
    """Restore dimension order and stamp CF, version, history, and latitude attrs."""
    desired_dims = list(tmin_aligned.dims) + [dim for dim in result.dims if dim not in tmin_aligned.dims]
    result = result.transpose(*desired_dims)
    result.attrs = build_output_attrs(
        tmin_aligned,
        CF_METADATA["pet_penman_monteith"],  # type: ignore[arg-type]
        {"latitude": _build_latitude_attr(latitude)},
        "PET Penman-Monteith",
    )
    if isinstance(latitude, xr.DataArray):
        lat_desc = f"DataArray(dims={latitude.dims})"
    else:
        lat_desc = str(latitude)

    _log().info(
        "pet_penman_monteith_completed",
        input_shape=tmin_aligned.shape,
        output_shape=result.shape,
        latitude=lat_desc,
    )
    return result


def pet_penman_monteith(
    daily_tmin_celsius: np.ndarray | xr.DataArray,
    daily_tmax_celsius: np.ndarray | xr.DataArray,
    latitude: float | np.floating | xr.DataArray,
    elevation_m: float | np.floating | xr.DataArray,
    wind_speed_m_s: np.ndarray | xr.DataArray | float,
    day_of_year: np.ndarray | xr.DataArray | None = None,
    wind_speed_height_m: float = 2.0,
    humidity: pm_eto.HumidityInputs | None = None,
    radiation: pm_eto.RadiationInputs | None = None,
    soil_heat_flux_mj_m2_day: np.ndarray | xr.DataArray | float = 0.0,
    albedo: float = pm_eto.REFERENCE_ALBEDO,
    time_dim: str = "time",
) -> np.ndarray | xr.DataArray:
    """Compute potential evapotranspiration using FAO-56 Penman-Monteith.

    Provides NumPy and xarray DataArray support for the full FAO-56 Penman-Monteith
    reference-evapotranspiration equation. The FAO-56 intermediate variables are
    derived internally from the supplied meteorology: atmospheric pressure and the
    psychrometric constant from elevation, saturation vapour pressure and its slope
    from temperature, actual vapour pressure from the best available humidity
    pathway, wind speed at the 2 m standard height, and net radiation from solar
    and clear-sky radiation.

    Humidity pathway precedence is dewpoint, RHmin/RHmax, RHmax, RHmean, then the
    arid-region ``e0(Tmin - 2)`` estimate. Radiation precedence is supplied solar
    radiation, sunshine hours, then the temperature-range estimate. For daily steps
    the soil heat flux defaults to zero.

    Args:
        daily_tmin_celsius: Daily minimum air temperature in degrees Celsius.
            For numpy: 1-D array of daily temperatures.
            For xarray: DataArray with a daily time dimension (may have spatial dims).
        daily_tmax_celsius: Daily maximum air temperature in degrees Celsius.
            For xarray: DataArray aligned with ``daily_tmin_celsius``.
        latitude: Latitude in degrees north (range: -90 to 90).
            For numpy: scalar float.
            For xarray: scalar float or DataArray(lat,) for spatial broadcasting.
        elevation_m: Station elevation above sea level in metres. A scalar, or a
            DataArray whose dimensions are a subset of the temperature's cell dims.
        wind_speed_m_s: Wind speed measured at ``wind_speed_height_m`` [m s-1].
            A scalar or a time-series array/DataArray.
        day_of_year: Day of the year, 1-365 (366 in a leap year). Required for
            NumPy input; inferred from the time coordinate for xarray input.
        wind_speed_height_m: Height at which the wind speed was measured [m].
        humidity: Optional actual-vapour-pressure inputs, in pathway precedence
            order; see :class:`climate_indices.pm_eto.HumidityInputs`.
        radiation: Optional solar-radiation inputs, in pathway precedence order;
            see :class:`climate_indices.pm_eto.RadiationInputs`.
        soil_heat_flux_mj_m2_day: Soil heat flux density [MJ m-2 day-1]; use 0 for
            daily steps.
        albedo: Canopy reflection coefficient (0.23 for the grass reference).
        time_dim: Name of the time dimension in the input DataArrays.

    Returns:
        PET values in mm/day as ``numpy.ndarray`` or ``xarray.DataArray``.

    Raises:
        InputTypeError: If the temperature inputs are neither numpy-coercible nor
            ``xr.DataArray``.
        TypeError: If the two temperature inputs are different types.
        ValueError: If ``day_of_year`` is omitted for NumPy input, or latitude is
            outside [-90, 90].
        InvalidArgumentError: If ``rh_min`` is given without ``rh_max``, or the
            wind measurement height is not positive.
        CoordinateValidationError: If the xarray time dimension is missing,
            non-monotonic, or not a January-start daily coordinate, or if the
            temperature and time-series inputs share a cell dimension with differing
            coordinates.
    """
    input_type = detect_input_type(daily_tmin_celsius)
    tmax_input_type = detect_input_type(daily_tmax_celsius)
    if input_type != tmax_input_type:
        raise TypeError(
            "daily_tmin_celsius and daily_tmax_celsius must be the same type. "
            f"Got tmin={input_type.name}, tmax={tmax_input_type.name}. "
            "Convert both to the same type (both numpy arrays or both xr.DataArray)."
        )

    _validate_latitude_range(latitude)
    humidity = humidity or pm_eto.HumidityInputs()
    radiation = radiation or pm_eto.RadiationInputs()

    if input_type == InputType.NUMPY:
        if isinstance(day_of_year, xr.DataArray) or day_of_year is None:
            raise ValueError("day_of_year is required for numpy inputs")
        if isinstance(latitude, xr.DataArray):
            raise TypeError(
                "latitude must be a scalar (float, int, or numpy scalar) when the "
                "temperature inputs are numpy arrays. "
                f"Got xr.DataArray with dims={latitude.dims}."
            )
        assert isinstance(daily_tmin_celsius, np.ndarray)
        assert isinstance(daily_tmax_celsius, np.ndarray)
        return np.asarray(
            pm_eto.penman_monteith_eto(
                daily_tmin_celsius,
                daily_tmax_celsius,
                latitude,
                elevation_m,
                wind_speed_m_s,
                day_of_year,
                wind_speed_height_m=wind_speed_height_m,
                humidity=humidity,
                radiation=radiation,
                soil_heat_flux_mj_m2_day=soil_heat_flux_mj_m2_day,
                albedo=albedo,
            )
        )

    # xarray path: validate → align → compute → rewrap
    assert isinstance(daily_tmin_celsius, xr.DataArray)
    assert isinstance(daily_tmax_celsius, xr.DataArray)
    tmin_da = daily_tmin_celsius
    tmax_da = daily_tmax_celsius

    validate_time_dimension(tmin_da, time_dim)
    validate_time_dimension(tmax_da, time_dim)
    validate_time_monotonicity(tmin_da.coords[time_dim])
    validate_time_monotonicity(tmax_da.coords[time_dim])

    # align every time-bearing DataArray (tmin, tmax, and the optional humidity,
    # radiation, and wind series) along the time dimension before computing
    optional_inputs: dict[str, Any] = {
        "day_of_year": day_of_year,
        "wind_speed_m_s": wind_speed_m_s,
        "elevation_m": elevation_m,
        "tdew_celsius": humidity.tdew_celsius,
        "rh_min": humidity.rh_min,
        "rh_max": humidity.rh_max,
        "rh_mean": humidity.rh_mean,
        "solar_radiation_mj_m2_day": radiation.solar_radiation_mj_m2_day,
        "sunshine_hours": radiation.sunshine_hours,
        "soil_heat_flux_mj_m2_day": soil_heat_flux_mj_m2_day,
    }
    tmin_aligned, tmax_aligned, optional_inputs = _align_penman_monteith_inputs(
        tmin_da, tmax_da, optional_inputs, time_dim
    )

    kernel_inputs = [
        tmin_aligned,
        tmax_aligned,
        latitude,
        optional_inputs["elevation_m"],
        optional_inputs["wind_speed_m_s"],
        optional_inputs["day_of_year"],
        optional_inputs["tdew_celsius"],
        optional_inputs["rh_min"],
        optional_inputs["rh_max"],
        optional_inputs["rh_mean"],
        optional_inputs["solar_radiation_mj_m2_day"],
        optional_inputs["sunshine_hours"],
        optional_inputs["soil_heat_flux_mj_m2_day"],
    ]
    core_dims: list[list[str]] = []
    ufunc_inputs: list[Any] = []
    for value in kernel_inputs:
        ufunc_value, dims = _penman_monteith_kernel_input(value, time_dim)
        ufunc_inputs.append(ufunc_value)
        core_dims.append(dims)

    # absence is recorded here, from the caller's arguments, so a real NaN scalar
    # reaching the kernel still propagates rather than reading as an omitted input
    absent_optionals = {
        name
        for name in (
            "tdew_celsius",
            "rh_min",
            "rh_max",
            "rh_mean",
            "solar_radiation_mj_m2_day",
            "sunshine_hours",
        )
        if optional_inputs[name] is None
    }

    def _optional(value: Any, name: str) -> Any:
        """Return ``None`` only for an input the caller actually omitted."""
        return None if name in absent_optionals else value

    def _penman_monteith_kernel(
        tmin: np.ndarray,
        tmax: np.ndarray,
        lat: Any,
        elevation: Any,
        wind: Any,
        day: Any,
        tdew: Any,
        rh_min_value: Any,
        rh_max_value: Any,
        rh_mean_value: Any,
        solar: Any,
        sunshine: Any,
        soil: Any,
    ) -> Any:
        return pm_eto.penman_monteith_eto(
            tmin,
            tmax,
            lat,
            elevation,
            wind,
            day,
            humidity=pm_eto.HumidityInputs(
                tdew_celsius=_optional(tdew, "tdew_celsius"),
                rh_min=_optional(rh_min_value, "rh_min"),
                rh_max=_optional(rh_max_value, "rh_max"),
                rh_mean=_optional(rh_mean_value, "rh_mean"),
            ),
            radiation=pm_eto.RadiationInputs(
                solar_radiation_mj_m2_day=_optional(solar, "solar_radiation_mj_m2_day"),
                sunshine_hours=_optional(sunshine, "sunshine_hours"),
                coastal=radiation.coastal,
            ),
            soil_heat_flux_mj_m2_day=soil,
            wind_speed_height_m=wind_speed_height_m,
            albedo=albedo,
        )

    result = xr.apply_ufunc(
        _penman_monteith_kernel,
        *ufunc_inputs,
        input_core_dims=core_dims,
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[float],
    )

    return _finalize_penman_monteith_result(result, tmin_aligned, latitude)


def _pdsi_numpy_passthrough(
    precips: np.ndarray | xr.DataArray,
    pet: np.ndarray | xr.DataArray,
    awc: float | np.ndarray | xr.DataArray,
    data_start_year: int | None,
    calibration_year_initial: int | None,
    calibration_year_final: int | None,
    fitting_params: dict[str, Any] | None,
    spatial_time_major: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Route a NumPy-coercible ``palmer.pdsi`` call through its stable contract."""
    if isinstance(pet, xr.DataArray):
        raise TypeError(
            "pet must be a numpy array when precips is a numpy array. Convert both to xr.DataArray for the xarray path."
        )
    if isinstance(awc, xr.DataArray):
        raise TypeError(
            "awc must be a scalar or numpy array when precips is a numpy array. "
            f"Got xr.DataArray with dims={awc.dims}. Use a scalar awc or convert "
            "precips and pet to xr.DataArray for spatial broadcasting."
        )
    if data_start_year is None or calibration_year_initial is None or calibration_year_final is None:
        raise ValueError(
            "data_start_year, calibration_year_initial, and calibration_year_final are required for numpy inputs"
        )
    # detect_input_type() routes list, tuple, and scalar input here as
    # NumPy-coercible, and the kernel coerces too; coerce rather than assert,
    # which -O strips and which otherwise reports a bare AssertionError. Using
    # asanyarray rather than asarray keeps a masked array's mask.
    return palmer.pdsi(
        np.asanyarray(precips),
        np.asanyarray(pet),
        awc,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        fitting_params,
        spatial_time_major=spatial_time_major,
    )


def palmer_pdsi(
    precips: np.ndarray | xr.DataArray,
    pet: np.ndarray | xr.DataArray,
    awc: float | np.ndarray | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    fitting_params: dict[str, Any] | None = None,
    spatial_time_major: bool = False,
    time_dim: str = "time",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None] | xr.Dataset:
    """Compute the standard Palmer drought indices, with xarray/Dask support.

    This function provides xarray DataArray support for the standard PDSI family
    (PDSI, PHDI, PMDI, and Z-Index). Like the PET entry points, it uses
    ``xr.apply_ufunc`` directly rather than the ``@xarray_adapter`` decorator: AWC
    is a per-cell soil constant with no time axis (the same shape problem PET's
    latitude solves), and the kernel returns four outputs rather than one. AWC is
    broadcast the way latitude is -- a scalar for a single location, or a DataArray
    whose dimensions are a subset of the precipitation's cell dimensions.

    .. warning:: **Beta Feature (xarray path)** — When called with ``xr.DataArray``
       input, this function uses the beta xarray adapter layer. The NumPy array
       interface and underlying computation are stable.

    Args:
        precips: Monthly precipitation values in inches.
            For numpy: 1-D array, ``(years, 12)`` array, or a 3-D time-major
            ``(time, *cells)`` block -- an ambiguous block whose first cell axis is a
            calendar period length (12 or 366) requires ``spatial_time_major=True``.
            List and tuple input is converted with ``np.asanyarray``.
            For xarray: DataArray with a monthly time dimension starting in January
            (may have additional cell dimensions).
        pet: Monthly potential evapotranspiration values in inches, matching
            ``precips``.
        awc: Available water capacity (soil constant) in inches. A scalar, or a
            DataArray whose cell coordinates match the precipitation grid (its
            dimensions may be a subset of the precipitation's cell dimensions).
        data_start_year: Initial year of the input dataset. Required for NumPy
            inputs; inferred from the first time coordinate for xarray inputs.
        calibration_year_initial: Initial year of the calibration period. Required
            for NumPy inputs; inferred from the time range for xarray inputs.
        calibration_year_final: Final year of the calibration period. Required for
            NumPy inputs; inferred from the time range for xarray inputs.
        fitting_params: Optional dict of pre-computed Palmer fitting parameters.
        spatial_time_major: Declares an ambiguous 3+-D numpy ``precips``/``pet`` as a
            time-major ``(time, *cells)`` block (per ADR-0009). Only used for numpy
            inputs; the xarray path reads its dimensions from the coordinate labels.
        time_dim: Name of the time dimension in the input DataArrays (default:
            ``"time"``). Only used for xarray inputs.

    Returns:
        For NumPy input, the five-item tuple ``palmer.pdsi()`` returns: PDSI, PHDI,
        PMDI, Z-Index, and the fitted parameters (``None`` for all-missing input).
        For xarray input, an ``xr.Dataset`` with one variable per index -- ``pdsi``,
        ``phdi``, ``pmdi``, and ``z_index`` -- each carrying its own CF metadata and
        provenance, and each matching the input's shape and coordinates.

    Raises:
        InputTypeError: If ``precips`` is neither numpy-coercible nor an
            ``xr.DataArray``.
        TypeError: If ``pet`` or ``awc`` mixes numpy and xarray with ``precips``, or
            if ``awc`` carries the time dimension.
        CoordinateValidationError: If the xarray time dimension is missing,
            non-monotonic, not monthly, or does not begin in January, or if
            ``precips`` and ``pet`` share a cell dimension with differing coordinates.
        ValueError: If a NumPy call omits a required temporal parameter, or the
            precipitation and PET shapes are incompatible.

    Notes:
        - A 3-D or higher xarray input reaches ``palmer.pdsi()`` as one
          ``(time, *cells)`` block per Dask spatial block, so the recursion runs once
          per block rather than once per grid cell. A 1-D or 2-D input keeps the
          per-cell path.
        - Dask-backed DataArrays stay lazy, but the time dimension must be a single
          chunk (the recursion spans the whole record); see
          :doc:`xarray_migration`.
        - ``palmer.scpdsi()`` deliberately has no xarray entry point: its
          per-location duration-factor fit and Wells recursion are not a bulk
          array operation (see ADR-0011). Compute it from NumPy values and rewrap
          the outputs when an xarray result is needed.
        - The Z-Index variable is named ``z_index``, matching the CF registry entry;
          the CLI NetCDF writer uses ``zindex`` for the same output.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from climate_indices import pdsi
        >>> time = pd.date_range("1980-01-01", periods=480, freq="MS")
        >>> precips = xr.DataArray(
        ...     np.random.default_rng(0).gamma(2.0, 2.0, (480, 2, 2)) / 25.4,
        ...     coords={"time": time, "lat": [35.0, 40.0], "lon": [-100.0, -95.0]},
        ...     dims=["time", "lat", "lon"],
        ... )
        >>> pet = xr.full_like(precips, 1.5)
        >>> awc = xr.DataArray([[5.0, 5.0], [6.0, 6.0]], dims=["lat", "lon"])
        >>> result = pdsi(precips, pet, awc, calibration_year_initial=1981, calibration_year_final=2010)
        >>> list(result.data_vars)
        ['pdsi', 'phdi', 'pmdi', 'z_index']
    """
    input_type = detect_input_type(precips)

    # numpy passthrough: the stable palmer.pdsi() contract, including its
    # spatial_time_major handling for a directly-declared 3-D block
    if input_type == InputType.NUMPY:
        return _pdsi_numpy_passthrough(
            precips,
            pet,
            awc,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            fitting_params,
            spatial_time_major,
        )

    # xarray path: validate → align → infer → compute → rewrap
    assert isinstance(precips, xr.DataArray)
    if not isinstance(pet, xr.DataArray):
        raise TypeError(
            "precips and pet must both be xr.DataArray or both numpy arrays. "
            f"Got precips={input_type.name}, pet={detect_input_type(pet).name}."
        )
    precips_da = precips
    pet_da = pet

    validate_time_dimension(precips_da, time_dim)
    validate_time_dimension(pet_da, time_dim)
    validate_time_monotonicity(precips_da[time_dim])
    validate_time_monotonicity(pet_da[time_dim])

    # the Palmer recursion is monthly and indexes calendar months from January
    _build_daily_calendar_plan(precips_da[time_dim], compute.Periodicity.monthly)
    _build_daily_calendar_plan(pet_da[time_dim], compute.Periodicity.monthly)

    aligned_precips, aligned_secondaries = _align_inputs(precips_da, {"pet": pet_da}, time_dim)
    precips_da = aligned_precips
    pet_da = aligned_secondaries["pet"]

    for dataarray in (precips_da, pet_da):
        if dataarray.chunks is not None:
            validate_dask_chunks(dataarray, time_dim)

    # infer any temporal parameter the caller left out; AWC is excluded because it
    # is a broadcast value rather than a time series
    provided: dict[str, Any] = {
        key: value
        for key, value in {
            "data_start_year": data_start_year,
            "calibration_year_initial": calibration_year_initial,
            "calibration_year_final": calibration_year_final,
            "fitting_params": fitting_params,
        }.items()
        if value is not None
    }
    inferred = _infer_temporal_parameters(palmer.pdsi, precips_da, [precips_da, pet_da, awc], provided, time_dim)
    provided.update(inferred)

    # normalize AWC for xr.apply_ufunc: a gridded input reaches palmer.pdsi as one
    # time-major block with the AWC per cell, instead of one call per grid cell. An
    # AWC carrying the time dimension is neither a scalar nor a cell field, and
    # apply_ufunc would only report it as an unexpected core dimension.
    if isinstance(awc, xr.DataArray) and time_dim in awc.dims:
        raise TypeError(
            f"awc must not carry the time dimension '{time_dim}': it is a per-cell soil "
            "constant, not a time series. Use a scalar or a DataArray over the "
            "precipitation's cell dimensions."
        )
    use_spatial_kernel, awc_for_ufunc = _spatial_kernel_cell_param(precips_da, awc, time_dim)

    def _pdsi_block(
        precips_block: np.ndarray,
        pet_block: np.ndarray,
        awc_block: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, ...]:
        """Run palmer.pdsi once on a (time, *cells) block, returning its four indices."""
        result = palmer.pdsi(
            np.moveaxis(precips_block, -1, 0),
            np.moveaxis(pet_block, -1, 0),
            awc_block,
            spatial_time_major=True,
            **kwargs,
        )
        return tuple(np.asarray(np.moveaxis(output, 0, -1), dtype=float) for output in result[:4])

    def _pdsi_per_cell(
        precips_series: np.ndarray,
        pet_series: np.ndarray,
        awc_value: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, ...]:
        """Run palmer.pdsi on one 1-D series, returning its four indices."""
        result = palmer.pdsi(precips_series, pet_series, awc_value, **kwargs)
        return tuple(np.asarray(output, dtype=float) for output in result[:4])

    # input_core_dims: the time dimension is core for both index inputs; AWC arrives
    #   as a scalar per cell on the per-cell path and a cell array on the other
    # output_core_dims: each of the four Palmer outputs preserves the time dimension
    result = xr.apply_ufunc(
        _pdsi_block if use_spatial_kernel else _pdsi_per_cell,
        precips_da,
        pet_da,
        awc_for_ufunc,
        input_core_dims=[[time_dim], [time_dim], []],
        output_core_dims=[[time_dim]] * 4,
        vectorize=not use_spatial_kernel,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[float] * 4,
        kwargs=provided,
    )
    result_arrays = result if isinstance(result, tuple) else (result,)

    # restore original dimension order (apply_ufunc places output core dims last);
    # an AWC broadcast over a dimension the index inputs lack adds that dimension
    desired_dims = list(precips_da.dims) + [dim for dim in result_arrays[0].dims if dim not in precips_da.dims]
    calculation_metadata: dict[str, Any] = {
        key: provided[key]
        for key in ("data_start_year", "calibration_year_initial", "calibration_year_final")
        if key in provided
    }

    # record the broadcast input the way the PET wrappers record latitude
    if isinstance(awc, xr.DataArray):
        calculation_metadata["awc"] = f"DataArray(dims={awc.dims})"
    elif np.ndim(awc) > 0:
        calculation_metadata["awc"] = f"ndarray(shape={np.shape(awc)})"
    else:
        calculation_metadata["awc"] = str(awc)
    if fitting_params:
        calculation_metadata["fitting_params"] = f"dict(keys={','.join(sorted(fitting_params))})"

    display_names = {"pdsi": "PDSI", "phdi": "PHDI", "pmdi": "PMDI", "z_index": "Z-Index"}
    variables: dict[str, xr.DataArray] = {}
    for name, array in zip(("pdsi", "phdi", "pmdi", "z_index"), result_arrays, strict=True):
        variable = array.transpose(*desired_dims)
        variable.attrs = build_output_attrs(
            precips_da,
            cf_metadata=CF_METADATA[name],  # type: ignore[arg-type]
            calculation_metadata=calculation_metadata,
            index_name=display_names[name],
        )
        variables[name] = variable

    _log().info(
        "palmer_pdsi_completed",
        input_shape=precips_da.shape,
        output_shape=variables["pdsi"].shape,
        spatial_kernel=use_spatial_kernel,
        **{key: str(value) for key, value in calculation_metadata.items()},
    )

    return xr.Dataset(variables)


#: Per-variable CF metadata for the fit-diagnostics Dataset. Diagnostics are an audit
#: surface rather than an index, so they carry their own CF metadata here instead of
#: a CF_METADATA entry; provenance is added by ``build_output_attrs`` per variable.
_FIT_DIAGNOSTICS_THOM_1958 = (
    "Thom, H. C. S. (1958). A note on the gamma distribution. "
    "Monthly Weather Review, 86(4), 117-122. "
    "https://doi.org/10.1175/1520-0493(1958)086<0117:ANOTGD>2.0.CO;2"
)
_FIT_DIAGNOSTICS_STAGGE_2015 = (
    "Stagge, J. H., Tallaksen, L. M., Gudmundsson, L., Van Loon, A. F., & Stahl, K. (2015). "
    "Candidate distributions for climatological drought indices (SPI and SPEI). "
    "International Journal of Climatology, 35(13), 4027-4040. https://doi.org/10.1002/joc.4267"
)
_FIT_DIAGNOSTICS_MASSEY_1951 = (
    "Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. "
    "Journal of the American Statistical Association, 46(253), 68-78. "
    "https://doi.org/10.1080/01621459.1951.10500769"
)
_FIT_DIAGNOSTICS_CF_METADATA: dict[str, dict[str, str]] = {
    "alpha": {
        "long_name": "Gamma shape parameter",
        "units": "1",
        "description": "Fitted gamma shape parameter, per calendar step and cell.",
        "references": _FIT_DIAGNOSTICS_THOM_1958,
    },
    "beta": {
        "long_name": "Gamma scale parameter",
        "units": "1",
        "description": "Fitted gamma scale parameter, per calendar step and cell.",
        "references": _FIT_DIAGNOSTICS_THOM_1958,
    },
    "loc": {
        "long_name": "Pearson Type III location parameter",
        "units": "1",
        "description": "Fitted Pearson Type III location parameter, per calendar step and cell.",
        "references": _FIT_DIAGNOSTICS_STAGGE_2015,
    },
    "scale": {
        "long_name": "Pearson Type III scale parameter",
        "units": "1",
        "description": "Fitted Pearson Type III scale parameter, per calendar step and cell.",
        "references": _FIT_DIAGNOSTICS_STAGGE_2015,
    },
    "skew": {
        "long_name": "Pearson Type III skewness parameter",
        "units": "1",
        "description": "Fitted Pearson Type III skewness parameter, per calendar step and cell.",
        "references": _FIT_DIAGNOSTICS_STAGGE_2015,
    },
    "prob_zero": {
        "long_name": "Probability of zero accumulation",
        "units": "1",
        "description": "Probability mass the fitted distribution places at a zero accumulation.",
        "references": _FIT_DIAGNOSTICS_STAGGE_2015,
    },
    "n_valid": {
        "long_name": "Valid calibration sample count",
        "units": "1",
        "description": "Non-missing, non-zero calibration values entering the Kolmogorov-Smirnov test.",
        "references": _FIT_DIAGNOSTICS_MASSEY_1951,
    },
    "ks_statistic": {
        "long_name": "Kolmogorov-Smirnov D statistic",
        "units": "1",
        "description": "Maximum distance between the fitted CDF and the calibration sample's empirical CDF.",
        "references": _FIT_DIAGNOSTICS_MASSEY_1951,
    },
    "ks_p_value": {
        "long_name": "Kolmogorov-Smirnov exact p-value",
        "units": "1",
        "description": "Exact Kolmogorov-Smirnov p-value of the fitted distribution.",
        "references": _FIT_DIAGNOSTICS_MASSEY_1951,
    },
    "distribution_used": {
        "long_name": "Distribution used after any fall back",
        "description": "Distribution actually used, after any Pearson Type III fall back to gamma.",
        "references": _FIT_DIAGNOSTICS_STAGGE_2015,
    },
}

#: The fitted-parameter variables each distribution reports. A Pearson Type III
#: request also carries the gamma slots, because a failed fit falls back to gamma
#: block by block and the inapplicable family is then NaN.
_FIT_DIAGNOSTICS_GAMMA_SLOTS: tuple[str, ...] = ("alpha", "beta")
_FIT_DIAGNOSTICS_PEARSON_SLOTS: tuple[str, ...] = ("loc", "scale", "skew")
_FIT_DIAGNOSTICS_FIELDS: tuple[str, ...] = ("prob_zero", "n_valid", "ks_statistic", "ks_p_value")


def _validate_diagnostics_fitting_params(
    input_da: xr.DataArray, fitting_params: dict[str, Any] | None, time_dim: str
) -> None:
    """Reject parameter layouts that the xarray path cannot align to cells."""
    # a 1-D or 2-D input takes the per-cell path, which cannot slice a parameter
    # array per cell, so cell-shaped parameters fail early instead of deep inside
    # the NumPy core's boolean checks
    cell_shaped = sorted(name for name, value in (fitting_params or {}).items() if np.ndim(value) > 1)
    if cell_shaped and input_da.ndim <= 2:
        raise ValueError(
            "fitting_params must carry one value per calendar step for a 1-D or 2-D input; "
            f"cell-shaped parameters ({', '.join(cell_shaped)}) are only supported for a "
            "3-D or higher time-major block"
        )
    # every Dask block receives the whole grid's parameters through apply_ufunc's
    # kwargs, so a cell dimension split across chunks would hand a block parameters
    # for cells it does not hold; fail here rather than when the graph computes
    if cell_shaped and input_da.chunks is not None:
        split_dims = [
            str(dim)
            for dim, dim_chunks in zip(input_da.dims, input_da.chunks, strict=True)
            if dim != time_dim and len(dim_chunks) > 1
        ]
        if split_dims:
            rechunk = ", ".join(f"'{dim}': -1" for dim in split_dims)
            raise ValueError(
                f"cell-shaped fitting_params ({', '.join(cell_shaped)}) require each cell dimension "
                f"in a single Dask chunk (split: {', '.join(split_dims)}); pass one value per "
                f"calendar step, or rechunk using: data = data.chunk({{{rechunk}}})"
            )


def _diagnostics_dataset(
    input_da: xr.DataArray,
    by_name: dict[str, xr.DataArray],
    variable_names: tuple[str, ...],
    period_dim: str,
    period_length: int,
    provided: dict[str, Any],
    fitting_params: dict[str, Any] | None,
    time_dim: str,
) -> xr.Dataset:
    """Rewrap block outputs with calendar coordinates and CF metadata."""
    cell_dims = [dim for dim in input_da.dims if dim != time_dim]
    calculation_metadata: dict[str, Any] = {key: value for key, value in provided.items() if key != "fitting_params"}
    if fitting_params:
        calculation_metadata["fitting_params"] = f"dict(keys={','.join(sorted(fitting_params))})"

    variables: dict[str, xr.DataArray] = {}
    for name in variable_names:
        variable = by_name[name].transpose(period_dim, *cell_dims)
        variable = variable.assign_coords({period_dim: np.arange(1, period_length + 1)})
        variable.attrs = build_output_attrs(
            input_da,
            cf_metadata=_FIT_DIAGNOSTICS_CF_METADATA[name],
            calculation_metadata=calculation_metadata,
            index_name="Fit diagnostics",
        )
        variables[name] = variable

    distribution_used = xr.where(by_name["distribution_code"] > 0, "pearson", "gamma")  # type: ignore[no-untyped-call]
    distribution_used.attrs = build_output_attrs(
        input_da,
        cf_metadata=_FIT_DIAGNOSTICS_CF_METADATA["distribution_used"],
        calculation_metadata=calculation_metadata,
        index_name="Fit diagnostics",
    )
    # a categorical variable carries no units, and the input's would otherwise leak through
    distribution_used.attrs.pop("units", None)
    variables["distribution_used"] = distribution_used
    return xr.Dataset(variables)


def _diagnostics_block(
    block: np.ndarray,
    *,
    calendar_plan: utils.DailyCalendarPlan | None,
    parameter_slots: tuple[str, ...],
    **kwargs: Any,
) -> tuple[np.ndarray, ...]:
    """Run the NumPy diagnostics once on one calendar-aware (time, *cells) block."""
    time_first = np.moveaxis(block, -1, 0)
    if calendar_plan is not None:
        time_first = calendar_plan.to_all_leap(time_first)
    diagnostics = indices.fit_diagnostics(time_first, spatial_time_major=True, **kwargs)
    # a parameter that does not apply to the fitted distribution is NaN, so the
    # Dataset schema stays fixed whichever way a block-level fall back goes
    reference = next(iter(diagnostics.parameters.values()))
    fitted: list[np.ndarray] = []
    for name in parameter_slots:
        parameter = diagnostics.parameters.get(name)
        fitted.append(
            np.moveaxis(
                np.asarray(parameter, dtype=float) if parameter is not None else np.full_like(reference, np.nan),
                0,
                -1,
            )
        )
    for name in _FIT_DIAGNOSTICS_FIELDS:
        fitted.append(np.moveaxis(np.asarray(getattr(diagnostics, name), dtype=float), 0, -1))
    # the code is 1.0 for pearson, 0.0 for gamma (requested or fallen back to),
    # broadcast over the block's cell dims because apply_ufunc's non-vectorized
    # path expects every output to carry the loop dimensions
    code = np.full(block.shape[:-1], float(diagnostics.distribution.value == "pearson"))
    return (*fitted, code)


def _numpy_fit_diagnostics(
    values: np.ndarray,
    scale: int,
    distribution: indices.Distribution,
    data_start_year: int | None,
    calibration_year_initial: int | None,
    calibration_year_final: int | None,
    periodicity: compute.Periodicity | None,
    fitting_params: dict[str, Any] | None,
    spatial_time_major: bool,
) -> compute.FitDiagnostics:
    """Preserve the NumPy core's required temporal parameters and layout."""
    if (
        data_start_year is None
        or calibration_year_initial is None
        or calibration_year_final is None
        or periodicity is None
    ):
        raise ValueError(
            "data_start_year, calibration_year_initial, calibration_year_final, and periodicity "
            "are required for numpy inputs"
        )
    return indices.fit_diagnostics(
        values,
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        fitting_params,
        spatial_time_major=spatial_time_major,
    )


def fit_diagnostics(
    values: np.ndarray | xr.DataArray,
    scale: int,
    distribution: indices.Distribution,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: compute.Periodicity | None = None,
    fitting_params: dict[str, Any] | None = None,
    spatial_time_major: bool = False,
    time_dim: str = "time",
) -> compute.FitDiagnostics | xr.Dataset:
    """Fit a distribution and return per-calendar-step diagnostics, with xarray/Dask support.

    This is the audit surface of :func:`climate_indices.indices.fit_diagnostics` as a
    CF-annotated ``xr.Dataset``: one variable per fitted parameter plus ``prob_zero``,
    ``n_valid``, ``ks_statistic``, ``ks_p_value``, and ``distribution_used``, each over
    a calendar-step dimension (``month`` or ``dayofyear``) and the input's cell
    dimensions. It uses ``xr.apply_ufunc`` directly rather than the ``@xarray_adapter``
    decorator because a Dataset of variables, not the input shape, is the result.

    .. warning:: **Beta Feature (xarray path)** — When called with ``xr.DataArray``
       input, this function uses the beta xarray adapter layer. The NumPy array
       interface and underlying computation are stable.

    Args:
        values: NumPy array or xarray DataArray of non-negative values. The NumPy
            layouts are the ones :func:`climate_indices.indices.fit_diagnostics`
            accepts; the xarray layout is read from its dimensions, with the core
            time dimension plus any number of cell dimensions.
        scale: Number of time steps accumulated before fitting.
        distribution: Distribution to fit, gamma or Pearson Type III. A failed
            Pearson Type III fit falls back to gamma, as it does for the indices.
        data_start_year: Initial year of the input values. Required for NumPy
            inputs; inferred from the first time coordinate for xarray inputs.
        calibration_year_initial: Initial year of the calibration period. Required
            for NumPy inputs; inferred from the time range for xarray inputs.
        calibration_year_final: Final year of the calibration period. Required for
            NumPy inputs; inferred from the time range for xarray inputs.
        periodicity: Monthly or daily time steps. Required for NumPy inputs;
            inferred from the time coordinate for xarray inputs.
        fitting_params: Optional pre-computed fitting parameters; deprecated
            aliases are normalized by the NumPy core. One value per calendar step
            for a 1-D or 2-D input; a 3-D input also accepts cell-shaped arrays,
            provided a Dask-backed input keeps each cell dimension in a single chunk.
        spatial_time_major: Declares a three-or-more-dimensional NumPy ``values`` as
            a time-major ``(time, *cells)`` block (per ADR-0009). Only used for NumPy
            inputs; the xarray path reads its dimensions from the coordinate labels.
        time_dim: Name of the time dimension in the input DataArray (default:
            ``"time"``). Only used for xarray inputs.

    Returns:
        For NumPy input, the :class:`climate_indices.compute.FitDiagnostics` that
        :func:`climate_indices.indices.fit_diagnostics` returns. For xarray input, an
        ``xr.Dataset`` whose variables are reduced over time only: a calendar-step
        dimension (``month`` for monthly data, ``dayofyear`` for daily data) plus the
        input's cell dimensions, each variable carrying CF attributes and provenance
        (scale, distribution, calibration period, version, and history). A Pearson
        Type III request keeps both parameter families: the one that does not apply
        to a fitted block (its fit fell back to gamma) is NaN there, and
        ``distribution_used`` names the family that does apply. ``distribution_used``
        does not carry the calendar-step dimension, so its value is constant over
        each calendar step; the fitted-parameter variables keep it. For chunked
        input the fall back is decided per chunk, as in the index adapters, so
        chunking can change it.

    Raises:
        ValueError: If a NumPy call omits a required temporal parameter, or
            ``fitting_params`` carries cell-shaped arrays for a 1-D or 2-D input or
            for a Dask-backed input with a cell dimension split across chunks.
        CoordinateValidationError: If the xarray time dimension is missing,
            non-monotonic, unsupported (cftime), does not begin in January, or its
            periodicity does not match the requested ``periodicity``.
        InsufficientDataError: If the series is shorter than ``scale``.
    """
    # numpy passthrough: the stable indices.fit_diagnostics() contract
    if detect_input_type(values) == InputType.NUMPY:
        return _numpy_fit_diagnostics(
            np.asanyarray(values),
            scale,
            distribution,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            periodicity,
            fitting_params,
            spatial_time_major,
        )

    # xarray path: validate → infer → compute per block → rewrap as a Dataset
    assert isinstance(values, xr.DataArray)
    input_da = values
    validate_time_dimension(input_da, time_dim)
    validate_time_monotonicity(input_da[time_dim])
    if input_da.chunks is not None:
        validate_dask_chunks(input_da, time_dim)

    provided: dict[str, Any] = {
        key: value
        for key, value in {
            "scale": scale,
            "distribution": distribution,
            "data_start_year": data_start_year,
            "calibration_year_initial": calibration_year_initial,
            "calibration_year_final": calibration_year_final,
            "periodicity": periodicity,
            "fitting_params": fitting_params,
        }.items()
        if value is not None
    }
    _validate_diagnostics_fitting_params(input_da, fitting_params, time_dim)
    # inference is metadata-only, so it stays safe for Dask-backed input
    provided.update(_infer_temporal_parameters(indices.fit_diagnostics, input_da, [input_da], provided, time_dim))
    calendar_plan = _resolve_daily_calendar_plan(indices.fit_diagnostics, input_da, [input_da], {}, provided, time_dim)
    _validate_sufficient_data(input_da[time_dim], scale, calendar_plan)

    resolved_periodicity = provided["periodicity"]
    period_dim = "month" if resolved_periodicity is compute.Periodicity.monthly else "dayofyear"
    period_length = 12 if resolved_periodicity is compute.Periodicity.monthly else 366

    # a Pearson Type III request keeps the gamma parameter slots too: the fit can
    # fall back block by block, and the family that did not apply is NaN there
    if distribution is indices.Distribution.pearson:
        parameter_slots = (*_FIT_DIAGNOSTICS_PEARSON_SLOTS, *_FIT_DIAGNOSTICS_GAMMA_SLOTS)
    else:
        parameter_slots = _FIT_DIAGNOSTICS_GAMMA_SLOTS
    variable_names = (*parameter_slots, *_FIT_DIAGNOSTICS_FIELDS)

    kernel_outputs = (*parameter_slots, *_FIT_DIAGNOSTICS_FIELDS, "distribution_code")
    result = xr.apply_ufunc(
        functools.partial(_diagnostics_block, calendar_plan=calendar_plan, parameter_slots=parameter_slots),
        input_da,
        input_core_dims=[[time_dim]],
        output_core_dims=[[period_dim]] * (len(kernel_outputs) - 1) + [[]],
        # a 2-D input has a single cell dimension, for which the block reading is
        # ambiguous with the legacy (years, periods) layout, so it keeps the
        # per-cell path the index adapters use
        vectorize=input_da.ndim == 2,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": {period_dim: period_length}},
        output_dtypes=[float] * len(kernel_outputs),
        kwargs=provided,
    )
    arrays = result if isinstance(result, tuple) else (result,)
    by_name = dict(zip(kernel_outputs, arrays, strict=True))
    dataset = _diagnostics_dataset(
        input_da, by_name, variable_names, period_dim, period_length, provided, fitting_params, time_dim
    )
    _log().info(
        "fit_diagnostics_completed",
        input_shape=input_da.shape,
        period_dim=period_dim,
        distribution=distribution.value,
    )

    return dataset

"""Beta, elementwise xarray execution for the energy transforms."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import xarray as xr

from climate_indices._units import _convert_temperature_units, _validate_daily_time_coordinate
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import (
    BetaFeatureWarning,
    CoordinateValidationError,
    InputTypeError,
    InvalidArgumentError,
)
from climate_indices.xarray_adapter import build_output_attrs


def temperature_input(data: xr.DataArray) -> xr.DataArray:
    """Normalize daily mean temperature without evaluating a Dask array."""
    if data.dtype.kind not in "iuf":
        raise InputTypeError("temperature must be numeric.", expected_type=float, actual_type=data.dtype.type)
    if "time" in data.coords:
        time = data.coords["time"]
        if not np.issubdtype(time.dtype, np.datetime64) or np.any(np.isnat(time.values)):
            raise CoordinateValidationError(
                "Degree days require a Gregorian datetime64 time coordinate without NaT.",
                coordinate_name="time",
                reason="unsupported_datetime_type",
            )
        _validate_daily_time_coordinate(data, "time")
    converted = _convert_temperature_units(
        data.astype(np.float64, keep_attrs=True), "celsius", argument_name="temperature.attrs['units']"
    )
    result: xr.DataArray = converted.assign_attrs(data.attrs).assign_coords(data.coords)
    return result


def wind_input(data: object, name: str) -> xr.DataArray:
    """Require wind speeds in metres per second; absent units imply native units."""
    if not isinstance(data, xr.DataArray):
        raise InputTypeError(
            f"{name} must be an xr.DataArray when another wind input is an xr.DataArray.",
            expected_type=xr.DataArray,
            actual_type=type(data),
        )
    units = data.attrs.get("units")
    if units is not None and (
        not isinstance(units, str) or units.strip().lower() not in {"m s-1", "m s^-1", "m s**-1", "m/s", "m s⁻¹"}
    ):
        raise InvalidArgumentError(
            f"{name} must have wind-speed units of m s-1.", argument_name=f"{name}.attrs['units']"
        )
    return data


def apply_elementwise(
    function: Callable[..., Any],
    index: str,
    primary: xr.DataArray,
    *secondary: Any,
    calculation_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Apply the NumPy function per block with exact label matching and no core dimensions."""
    inputs = (primary, *secondary)
    for data in inputs:
        if not isinstance(data, xr.DataArray) and np.ndim(data) != 0:
            raise InvalidArgumentError(
                "An xarray energy transform requires labeled DataArrays or scalar companion inputs.",
                argument_name="inputs",
            )
        dtype = getattr(data, "dtype", None)
        if dtype is None:
            dtype = np.asarray(data).dtype
        if dtype.kind not in "biuf":
            raise InputTypeError("Energy inputs must be numeric.", expected_type=float, actual_type=dtype.type)
        if not isinstance(data, xr.DataArray):
            continue
        for name in primary.coords.keys() & data.coords.keys():
            left, right = primary.coords[name], data.coords[name]
            left_lazy = left.chunks is not None
            right_lazy = right.chunks is not None
            if (
                not left_lazy
                and not right_lazy
                and left.ndim == right.ndim == 0
                and "height" in (left.attrs.get("standard_name"), right.attrs.get("standard_name"))
            ):
                continue
            if left_lazy and right_lazy:
                # Compare graph tokens, never values: a copy of the same Dask
                # coordinate is equal without computing it.
                matches = left.dims == right.dims and (left.data is right.data or left.data.name == right.data.name)
            elif left_lazy or right_lazy:
                # One side is lazy: no shared graph, and comparing values would compute it.
                matches = False
            else:
                matches = left.dims == right.dims and left.variable.equals(right.variable)
            if not matches:
                raise CoordinateValidationError(
                    f"Shared coordinate {name!r} must match exactly; lazy coordinates must share their backing array.",
                    coordinate_name=str(name),
                    reason="mismatched_coordinates",
                )
    warnings.warn(
        "The energy xarray adapter is beta; the NumPy API is stable (ADR-0012).",
        BetaFeatureWarning,
        stacklevel=3,
    )
    result: xr.DataArray = xr.apply_ufunc(
        function,
        *inputs,
        input_core_dims=[[]] * len(inputs),
        kwargs=kwargs,
        join="exact",
        keep_attrs=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    if index == "wind_shear_exponent":
        # Shear is an auxiliary diagnostic, not an energy index in the registry.
        metadata: dict[str, str] = {
            "long_name": "Wind Shear Exponent",
            "units": "1",
            "references": CF_METADATA["hub_height_wind_speed"]["references"],
        }
    else:
        metadata = cast(dict[str, str], CF_METADATA[index])
    result.attrs = build_output_attrs(
        primary, cf_metadata=metadata, calculation_metadata=calculation_metadata, index_name=metadata["long_name"]
    )
    return result


def scalar_coordinate(data: xr.DataArray, name: str, value: float, **attrs: str) -> xr.DataArray:
    """Attach a physical scalar coordinate without replacing a dimension."""
    if name in data.dims:
        raise InvalidArgumentError(
            f"The output coordinate {name!r} conflicts with an input dimension.", argument_name=name
        )
    if attrs.get("standard_name") == "height":
        height_coordinates = [
            key
            for key, coordinate in data.coords.items()
            if key in (name, "hub_height") or coordinate.attrs.get("standard_name") == "height"
        ]
        if any(data.coords[key].ndim != 0 for key in height_coordinates):
            raise InvalidArgumentError(
                "A varying height coordinate cannot describe a scalar hub height.", argument_name="height"
            )
        data = data.drop_vars([key for key in height_coordinates if key != name])
    result: xr.DataArray = data.assign_coords({name: xr.DataArray(value, attrs=attrs)})
    return result

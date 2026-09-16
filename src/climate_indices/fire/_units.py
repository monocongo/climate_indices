"""CF units-attribute handling shared by the fire xarray adapters."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from climate_indices.exceptions import CoordinateValidationError, InvalidArgumentError

# CF units-attribute spellings this module recognizes, matching the precedent
# list in __main__.py's legacy CLI unit handling, plus the CF flux unit the
# NetCDF/Zarr ecosystem commonly uses for precipitation rate.
_PRECIP_UNITS_MM = frozenset({"mm", "millimeters", "millimeter"})
_PRECIP_UNITS_MM_PER_DAY = frozenset({"mm/dy", "mm day-1", "mm/day"})
_PRECIP_UNITS_MM_PER_YEAR = frozenset({"mm/year", "mm/yr", "mm year-1", "mm yr-1"})
_PRECIP_UNITS_INCH = frozenset({"inch", "inches"})
_PRECIP_UNITS_INCH_PER_YEAR = frozenset(
    {"in/year", "in/yr", "inch/year", "inch/yr", "inches/year", "inches/yr", "inch year-1", "inch yr-1"}
)
_PRECIP_UNITS_FLUX = frozenset({"kg m-2 s-1", "kg/m2/s", "kg m^-2 s^-1", "kg.m-2.s-1", "kg/m^2/s"})
_SECONDS_PER_DAY = 86400.0

_TEMP_UNITS_CELSIUS = frozenset({"c", "celsius", "degree_celsius", "degrees_celsius", "degc"})
_TEMP_UNITS_FAHRENHEIT = frozenset({"f", "fahrenheit", "degree_fahrenheit", "degrees_fahrenheit", "degf"})
_TEMP_UNITS_KELVIN = frozenset({"k", "kelvin"})
_KELVIN_OFFSET_CELSIUS = 273.15


def _convert_precipitation_units(
    data: xr.DataArray,
    target: Literal["mm", "inch"],
    *,
    argument_name: str = "precipitation.attrs['units']",
    annual: bool = False,
) -> xr.DataArray:
    """Convert a precipitation DataArray to ``target`` from its CF ``units`` attribute.

    An absent ``units`` attribute is assumed to already match ``target`` -- the
    caller's ``units=`` scale is trusted, never silently overridden. An
    unrecognized attribute raises ``InvalidArgumentError`` rather than guessing.
    ``annual=True`` declares a mean annual climatology: per-day and flux rate
    units are rejected, and per-year spellings are accepted in addition to the
    annual totals (``mm``, ``inch``). Conversion is xarray arithmetic, so
    Dask-backed input stays lazy.
    """
    raw_units = data.attrs.get("units")
    normalized = raw_units.strip().lower() if isinstance(raw_units, str) else None
    if normalized is None:
        return data
    if normalized in _PRECIP_UNITS_FLUX or normalized in _PRECIP_UNITS_MM_PER_DAY:
        if annual:
            raise InvalidArgumentError(
                f"mean_annual_precipitation cannot use precipitation rate units: {raw_units!r}.",
                argument_name=argument_name,
                argument_value=str(raw_units),
                valid_values="An annual total (mm, inch) or a per-year rate (mm year-1, inch year-1)",
            )
        if normalized in _PRECIP_UNITS_FLUX:
            data = data * _SECONDS_PER_DAY
        source: Literal["mm", "inch"] = "mm"
    elif normalized in _PRECIP_UNITS_MM or (annual and normalized in _PRECIP_UNITS_MM_PER_YEAR):
        source = "mm"
    elif normalized in _PRECIP_UNITS_INCH or (annual and normalized in _PRECIP_UNITS_INCH_PER_YEAR):
        source = "inch"
    else:
        raise InvalidArgumentError(
            f"Unsupported precipitation units attribute: {raw_units!r}.",
            argument_name=argument_name,
            argument_value=str(raw_units),
            valid_values=(
                "An annual total (mm, inch) or a per-year rate (mm year-1, inch year-1)"
                if annual
                else "mm / mm day-1, inch(es), or kg m-2 s-1"
            ),
        )
    if source == target:
        return data
    converted: xr.DataArray = data / 25.4 if target == "inch" else data * 25.4
    return converted


def _convert_temperature_units(data: xr.DataArray, target: Literal["celsius", "fahrenheit"]) -> xr.DataArray:
    """Convert a temperature DataArray to ``target`` from its CF ``units`` attribute.

    An absent ``units`` attribute is assumed to already match ``target``; an
    unrecognized one raises ``InvalidArgumentError`` rather than guessing.
    Conversion is xarray arithmetic, so Dask-backed input stays lazy.
    """
    raw_units = data.attrs.get("units")
    normalized = raw_units.strip().lower() if isinstance(raw_units, str) else None
    if normalized is None:
        return data
    if normalized in _TEMP_UNITS_KELVIN:
        data = data - _KELVIN_OFFSET_CELSIUS
        source: Literal["celsius", "fahrenheit"] = "celsius"
    elif normalized in _TEMP_UNITS_CELSIUS:
        source = "celsius"
    elif normalized in _TEMP_UNITS_FAHRENHEIT:
        source = "fahrenheit"
    else:
        raise InvalidArgumentError(
            f"Unsupported temperature units attribute: {raw_units!r}.",
            argument_name="maximum_temperature.attrs['units']",
            argument_value=str(raw_units),
            valid_values="K, kelvin, C/celsius, or F/fahrenheit",
        )
    if source == target:
        return data
    converted: xr.DataArray = data * 9.0 / 5.0 + 32.0 if target == "fahrenheit" else (data - 32.0) * 5.0 / 9.0
    return converted


def _validate_daily_time_coordinate(data: xr.DataArray, time_dim: str) -> None:
    """Require consecutive daily samples: the fire recurrences are defined per day.

    Called only when the time coordinate is attached; a dimension-only time
    axis has no cadence metadata to check.

    Raises:
        CoordinateValidationError: If the time steps are not exactly one day apart.
    """
    values = data.coords[time_dim].values
    if values.size < 2:
        return
    try:
        deltas = np.diff(values.astype("datetime64[ns]"))
    except (TypeError, ValueError) as exc:
        raise CoordinateValidationError(
            message=f"Cannot verify daily cadence for '{time_dim}': unsupported datetime type.",
            coordinate_name=time_dim,
            reason="unsupported_datetime_type",
        ) from exc
    if np.any(deltas != np.timedelta64(1, "D")):
        raise CoordinateValidationError(
            message=(
                f"Fire-weather indices require consecutive daily '{time_dim}' steps, but '{time_dim}' is not daily. "
                "Aggregate the observations to daily totals and daily maxima before calling."
            ),
            coordinate_name=time_dim,
            reason="not_daily",
        )

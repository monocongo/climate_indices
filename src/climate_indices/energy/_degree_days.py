"""Heating, cooling, and energy degree days from daily mean temperature.

Degree days describe the **climate-driven component** of heating and cooling
demand, not actual load, which also depends on building stock, technology and
behaviour. The single-base convention is NOAA CPC's: ``HDD = max(base - T, 0)``
and ``CDD = max(T - base, 0)`` on a daily mean temperature, with
``EDD = HDD + CDD``. The C3S and Eurostat schemes pair an activation threshold
with a separate reference temperature, which one base cannot reproduce, so no
public default base is chosen and every caller supplies one explicitly.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.energy._xarray import apply_elementwise, scalar_coordinate, temperature_input
from climate_indices.exceptions import InputTypeError, InvalidArgumentError


def _validated_temperature(temperature: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return temperature as float64 with masked values as NaN, or raise for a non-numeric or infinite value."""
    raw = np.asarray(temperature)
    if raw.dtype.kind not in "iuf":
        raise InputTypeError("temperature must be numeric.", expected_type=float, actual_type=raw.dtype.type)
    values = np.asarray(np.ma.asarray(temperature, dtype=np.float64).filled(np.nan), dtype=np.float64)
    if np.any(np.isinf(values)):
        raise InvalidArgumentError(
            "temperature must be finite or NaN.",
            argument_name="temperature",
            argument_value="infinite value",
            valid_values="A finite temperature or NaN",
        )
    return values


def _validated_base(base_temperature: float, name: str) -> float:
    """Return a base temperature as a finite float, or raise."""
    raw = np.asarray(base_temperature)
    if np.ma.is_masked(base_temperature) or raw.dtype.kind not in "iuf" or raw.ndim != 0:
        raise InvalidArgumentError(
            f"{name} must be a single finite number.",
            argument_name=name,
            argument_value=str(base_temperature),
            valid_values="A finite temperature",
        )
    value = float(raw)
    if not np.isfinite(value):
        raise InvalidArgumentError(
            f"{name} must be a single finite number.",
            argument_name=name,
            argument_value=str(base_temperature),
            valid_values="A finite temperature",
        )
    return value


@overload
def heating_degree_days(temperature: xr.DataArray, *, base_temperature: float) -> xr.DataArray: ...  # pyright: ignore[reportOverlappingOverload]


@overload
def heating_degree_days(temperature: npt.ArrayLike, *, base_temperature: float) -> npt.NDArray[np.float64]: ...


def heating_degree_days(
    temperature: npt.ArrayLike | xr.DataArray,
    *,
    base_temperature: float,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Calculate heating degree days from daily mean temperature.

    Follows NOAA CPC's single-base convention,
    ``HDD = max(base_temperature - temperature, 0)``. A caller holding daily
    minimum and maximum temperatures computes their mean first; there is no
    derived-mean keyword. The C3S and Eurostat schemes are not representable as
    a non-default single base, so the base is always supplied explicitly. This
    index describes climate-driven heating demand, not actual load.

    DataArray input uses the beta adapter and returns a DataArray with CF
    metadata and a scalar air-temperature threshold coordinate. Temperature
    units are inferred from its CF ``units`` attribute; the base remains °C.
    An attached ``time`` coordinate must contain consecutive daily samples.
    Dask arrays remain lazy, and every dimension may have multiple chunks.

    Args:
        temperature: Daily mean temperature in °C. Missing values may be NaN or
            masked. Convert NumPy input from K or °F to °C before calling;
            DataArray input converts K and °F automatically.
        base_temperature: Base temperature in °C. No default is chosen.

    Returns:
        Heating degree days in °C day, in the input shape, with NaN where the
        input is missing. The output is a temperature difference, so K day is
        numerically identical.

    Raises:
        InputTypeError: If temperature is non-numeric.
        InvalidArgumentError: If temperature is infinite, or the base is
            non-finite, masked, or not a single number.
    """
    if isinstance(temperature, xr.DataArray):
        base = _validated_base(base_temperature, "base_temperature")
        result = apply_elementwise(
            heating_degree_days,
            "heating_degree_days",
            temperature_input(temperature),
            base_temperature=base,
            calculation_metadata={"base_temperature": base},
        )
        return scalar_coordinate(
            result, "air_temperature_threshold", base, standard_name="air_temperature_threshold", units="degC"
        )
    values = _validated_temperature(temperature)
    base = _validated_base(base_temperature, "base_temperature")
    return np.asarray(np.maximum(base - values, 0.0))


@overload
def cooling_degree_days(temperature: xr.DataArray, *, base_temperature: float) -> xr.DataArray: ...  # pyright: ignore[reportOverlappingOverload]


@overload
def cooling_degree_days(temperature: npt.ArrayLike, *, base_temperature: float) -> npt.NDArray[np.float64]: ...


def cooling_degree_days(
    temperature: npt.ArrayLike | xr.DataArray,
    *,
    base_temperature: float,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Calculate cooling degree days from daily mean temperature.

    Follows NOAA CPC's single-base convention,
    ``CDD = max(temperature - base_temperature, 0)``. A caller holding daily
    minimum and maximum temperatures computes their mean first; there is no
    derived-mean keyword. The C3S and Eurostat schemes are not representable as
    a non-default single base, so the base is always supplied explicitly. This
    index describes climate-driven cooling demand, not actual load.

    DataArray input uses the beta adapter and returns a DataArray with CF
    metadata and a scalar air-temperature threshold coordinate. Temperature
    units are inferred from its CF ``units`` attribute; the base remains °C.
    An attached ``time`` coordinate must contain consecutive daily samples.
    Dask arrays remain lazy, and every dimension may have multiple chunks.

    Args:
        temperature: Daily mean temperature in °C. Missing values may be NaN or
            masked. Convert NumPy input from K or °F to °C before calling;
            DataArray input converts K and °F automatically.
        base_temperature: Base temperature in °C. No default is chosen.

    Returns:
        Cooling degree days in °C day, in the input shape, with NaN where the
        input is missing. The output is a temperature difference, so K day is
        numerically identical.

    Raises:
        InputTypeError: If temperature is non-numeric.
        InvalidArgumentError: If temperature is infinite, or the base is
            non-finite, masked, or not a single number.
    """
    if isinstance(temperature, xr.DataArray):
        base = _validated_base(base_temperature, "base_temperature")
        result = apply_elementwise(
            cooling_degree_days,
            "cooling_degree_days",
            temperature_input(temperature),
            base_temperature=base,
            calculation_metadata={"base_temperature": base},
        )
        return scalar_coordinate(
            result, "air_temperature_threshold", base, standard_name="air_temperature_threshold", units="degC"
        )
    values = _validated_temperature(temperature)
    base = _validated_base(base_temperature, "base_temperature")
    return np.asarray(np.maximum(values - base, 0.0))


@overload
def energy_degree_days(  # pyright: ignore[reportOverlappingOverload]
    temperature: xr.DataArray, *, heating_base_temperature: float, cooling_base_temperature: float
) -> xr.DataArray: ...


@overload
def energy_degree_days(
    temperature: npt.ArrayLike, *, heating_base_temperature: float, cooling_base_temperature: float
) -> npt.NDArray[np.float64]: ...


def energy_degree_days(
    temperature: npt.ArrayLike | xr.DataArray,
    *,
    heating_base_temperature: float,
    cooling_base_temperature: float,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Calculate energy degree days as heating plus cooling degree days.

    ``EDD = HDD + CDD`` sums NOAA CPC's single-base transforms with separate
    heating and cooling bases. This does not implement C3S activation thresholds.
    The bases may be equal, distinct, or overlapping. The sum is an algebraic
    identity, not an independent oracle. This index describes climate-driven
    energy demand, not actual load.

    DataArray input uses the beta adapter and returns a DataArray with CF
    metadata and scalar heating/cooling base coordinates. Temperature units
    are inferred from its CF ``units`` attribute; both bases remain °C.
    An attached ``time`` coordinate must contain consecutive daily samples.
    Dask arrays remain lazy, and every dimension may have multiple chunks.

    Args:
        temperature: Daily mean temperature in °C. Missing values may be NaN or
            masked. Convert NumPy input from K or °F to °C before calling;
            DataArray input converts K and °F automatically.
        heating_base_temperature: Base temperature for heating degree days, in °C.
        cooling_base_temperature: Base temperature for cooling degree days, in °C.

    Returns:
        Energy degree days in °C day, in the input shape, with NaN where the
        input is missing. The output is a temperature difference, so K day is
        numerically identical.

    Raises:
        InputTypeError: If temperature is non-numeric.
        InvalidArgumentError: If temperature is infinite, or a base is
            non-finite, masked, or not a single number.
    """
    if isinstance(temperature, xr.DataArray):
        heating_base = _validated_base(heating_base_temperature, "heating_base_temperature")
        cooling_base = _validated_base(cooling_base_temperature, "cooling_base_temperature")
        result = apply_elementwise(
            energy_degree_days,
            "energy_degree_days",
            temperature_input(temperature),
            heating_base_temperature=heating_base,
            cooling_base_temperature=cooling_base,
            calculation_metadata={"heating_base_temperature": heating_base, "cooling_base_temperature": cooling_base},
        )
        result = scalar_coordinate(result, "heating_base_temperature", heating_base, units="degC")
        return scalar_coordinate(result, "cooling_base_temperature", cooling_base, units="degC")
    values = _validated_temperature(temperature)
    heating_base = _validated_base(heating_base_temperature, "heating_base_temperature")
    cooling_base = _validated_base(cooling_base_temperature, "cooling_base_temperature")
    heating = np.asarray(np.maximum(heating_base - values, 0.0))
    heating += np.maximum(values - cooling_base, 0.0)
    return heating


__all__ = [
    "cooling_degree_days",
    "energy_degree_days",
    "heating_degree_days",
]

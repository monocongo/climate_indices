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

import numpy as np
import numpy.typing as npt

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
    if raw.dtype.kind not in "iuf" or raw.ndim != 0:
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


def heating_degree_days(
    temperature: npt.ArrayLike,
    *,
    base_temperature: float,
) -> npt.NDArray[np.float64]:
    """Calculate heating degree days from daily mean temperature.

    Follows NOAA CPC's single-base convention,
    ``HDD = max(base_temperature - temperature, 0)``. A caller holding daily
    minimum and maximum temperatures computes their mean first; there is no
    derived-mean keyword. The C3S and Eurostat schemes are not representable as
    a non-default single base, so the base is always supplied explicitly. This
    index describes climate-driven heating demand, not actual load.

    Args:
        temperature: Daily mean temperature in °C, or a Temperature Difference
            equivalent in K. Missing values may be NaN or masked.
        base_temperature: Base temperature in °C (or K, numerically identical
            for a difference). No default is chosen.

    Returns:
        Heating degree days in °C day, in the input shape, with NaN where the
        input is missing. The output is a temperature difference, so K day is
        numerically identical.

    Raises:
        InputTypeError: If temperature is non-numeric.
        InvalidArgumentError: If temperature or the base is infinite, or the
            base is not a single number.
    """
    values = _validated_temperature(temperature)
    base = _validated_base(base_temperature, "base_temperature")
    return np.maximum(base - values, 0.0)


def cooling_degree_days(
    temperature: npt.ArrayLike,
    *,
    base_temperature: float,
) -> npt.NDArray[np.float64]:
    """Calculate cooling degree days from daily mean temperature.

    Follows NOAA CPC's single-base convention,
    ``CDD = max(temperature - base_temperature, 0)``. A caller holding daily
    minimum and maximum temperatures computes their mean first; there is no
    derived-mean keyword. The C3S and Eurostat schemes are not representable as
    a non-default single base, so the base is always supplied explicitly. This
    index describes climate-driven cooling demand, not actual load.

    Args:
        temperature: Daily mean temperature in °C, or a Temperature Difference
            equivalent in K. Missing values may be NaN or masked.
        base_temperature: Base temperature in °C (or K, numerically identical
            for a difference). No default is chosen.

    Returns:
        Cooling degree days in °C day, in the input shape, with NaN where the
        input is missing. The output is a temperature difference, so K day is
        numerically identical.

    Raises:
        InputTypeError: If temperature is non-numeric.
        InvalidArgumentError: If temperature or the base is infinite, or the
            base is not a single number.
    """
    values = _validated_temperature(temperature)
    base = _validated_base(base_temperature, "base_temperature")
    return np.maximum(values - base, 0.0)


def energy_degree_days(
    temperature: npt.ArrayLike,
    *,
    heating_base_temperature: float,
    cooling_base_temperature: float,
) -> npt.NDArray[np.float64]:
    """Calculate energy degree days as heating plus cooling degree days.

    ``EDD = HDD + CDD`` with separate bases, following the C3S definition. The
    bases may be equal, distinct, or overlapping; the sum is an algebraic
    identity and not an independent oracle. This index describes climate-driven
    energy demand, not actual load.

    Args:
        temperature: Daily mean temperature in °C, or a Temperature Difference
            equivalent in K. Missing values may be NaN or masked.
        heating_base_temperature: Base temperature for heating degree days, in
            °C (or K, numerically identical for a difference).
        cooling_base_temperature: Base temperature for cooling degree days, in
            °C (or K, numerically identical for a difference).

    Returns:
        Energy degree days in °C day, in the input shape, with NaN where the
        input is missing. The output is a temperature difference, so K day is
        numerically identical.

    Raises:
        InputTypeError: If temperature is non-numeric.
        InvalidArgumentError: If temperature or a base is infinite, or a base is
            not a single number.
    """
    heating = heating_degree_days(temperature, base_temperature=heating_base_temperature)
    cooling = cooling_degree_days(temperature, base_temperature=cooling_base_temperature)
    return heating + cooling


__all__ = [
    "cooling_degree_days",
    "energy_degree_days",
    "heating_degree_days",
]

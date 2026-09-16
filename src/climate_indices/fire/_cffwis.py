"""Canadian Forest Fire Weather Index System (CFFWIS)."""

from __future__ import annotations

import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from climate_indices.exceptions import DataShapeError, InvalidArgumentError
from climate_indices.fire._common import (
    _apply_gap_policy,
    _as_float_array,
    _static_spatial_array,
    _validate_recurrence_options,
)
from climate_indices.logging_config import get_logger
from climate_indices.performance import check_large_array_memory

# retrieve structlog logger for this module
_logger = get_logger(__name__)


# The CFFWIS moisture codes (#803) follow Van Wagner and Pickett (1985) as
# implemented by the NRCan reference code: `cffdrs_r` and its Python port
# `cffdrs_py` (the frozen test vectors pin commit 0f57fcca of the latter). All
# equations are evaluated in the source's operational units: km/h wind, mm
# rain, degrees Celsius. The published FFMC equations print 147.2 for the
# moisture-content conversion; the reference code uses the exact
# 250 * 59.5 / 101, applied in both directions, and this implementation
# matches the reference code. DMC's post-rain conversion likewise uses the
# reference code's more accurate 43.43 * (5.6348 - ln(Wmr - 20)) form of
# Eq. 15 rather than the printed 244.72 - 43.43 * ln(Wmr - 20).
_FFMC_COEFFICIENT = 250.0 * 59.5 / 101.0
_FFMC_MAXIMUM = 101.0
_FFMC_MOISTURE_CAP = 250.0
_FFMC_PRECIPITATION_THRESHOLD_MM = 0.5
_FFMC_MOISTURE_FOR_RAIN_CORRECTION = 150.0
_KILOMETERS_PER_HOUR_PER_METER_PER_SECOND = 3.6

_DMC_PRECIPITATION_THRESHOLD_MM = 1.5
_DMC_TEMPERATURE_FLOOR_CELSIUS = -1.1

_DC_PRECIPITATION_THRESHOLD_MM = 2.8
_DC_TEMPERATURE_FLOOR_CELSIUS = -2.8

# Effective day length in hours for DMC, by latitude band and calendar month
# (Van Wagner and Pickett, 1985). The bands and their table rows are, in
# order: 46 N (latitude > 30), 20 N (10 < latitude <= 30), equator
# (-10 < latitude <= 10), 20 S (-30 < latitude <= -10), and 40 S
# (latitude <= -30). The 46 N row is the Canadian standard; the other rows
# are the reference code's latitude adjustments, not a fallback for it.
_DMC_EFFECTIVE_DAY_LENGTH_HOURS = np.array(
    [
        [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0],
        [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],
        [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8],
    ]
)

# Day-length adjustment term for DC potential evapotranspiration, by
# latitude band and calendar month (Van Wagner and Pickett, 1985). Bands:
# north (latitude > 20), equator (-20 < latitude <= 20), south
# (latitude <= -20).
_DC_DAY_LENGTH_ADJUSTMENT = np.array(
    [
        [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6],
        [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4],
        [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8],
    ]
)

# The seven CFFWIS outputs, in the order shared by CFFWISResult and the
# xarray Dataset planned in #807.
_CFFWISComponent = Literal["ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"]
_CFFWIS_COMPONENTS: tuple[_CFFWISComponent, ...] = ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr")


# Canadian Forest Fire Weather Index System moisture codes (#803)
#
# The three codes share the daily-recurrence contract of ADR-0006 and the
# missing-day policy of ADR-0007 through ``_run_cffwis_recurrence``, which
# owns the gap bookkeeping and the output/spin-up handling. Each ``_*_next``
# function is the pure daily update for one code.


@dataclass(frozen=True)
class FFMCState:
    """State needed to resume a Fine Fuel Moisture Code recurrence.

    A ``trailing_gap_days`` value of ``None`` means no valid day has started
    the recurrence. For spatial arrays, ``-1`` marks individual cells that
    have not started yet. A NaN ``ffmc`` is only valid where
    ``trailing_gap_days`` shows that a gap has started the cell; a
    not-started cell holds a number.
    """

    ffmc: npt.NDArray[np.float64]
    trailing_gap_days: npt.NDArray[np.int64] | None


@dataclass(frozen=True)
class FFMCResult:
    """Fine Fuel Moisture Code values and final state returned by :func:`ffmc`."""

    values: npt.NDArray[np.float64]
    state: FFMCState


@dataclass(frozen=True)
class DMCState:
    """State needed to resume a Duff Moisture Code recurrence.

    ``trailing_gap_days`` follows :class:`FFMCState`; a NaN ``dmc`` is only
    valid where a gap has started the cell.
    """

    dmc: npt.NDArray[np.float64]
    trailing_gap_days: npt.NDArray[np.int64] | None


@dataclass(frozen=True)
class DMCResult:
    """Duff Moisture Code values and final state returned by :func:`duff_moisture_code`."""

    values: npt.NDArray[np.float64]
    state: DMCState


@dataclass(frozen=True)
class DCState:
    """State needed to resume a Drought Code recurrence.

    ``trailing_gap_days`` follows :class:`FFMCState`; a NaN ``dc`` is only
    valid where a gap has started the cell.
    """

    dc: npt.NDArray[np.float64]
    trailing_gap_days: npt.NDArray[np.int64] | None


@dataclass(frozen=True)
class DCResult:
    """Drought Code values and final state returned by :func:`drought_code`."""

    values: npt.NDArray[np.float64]
    state: DCState


def _daily_weather_arrays(
    names: tuple[str, ...],
    *values: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Coerce and broadcast time-first daily weather inputs, rejecting infinity."""
    arrays = tuple(_as_float_array(value) for value in values)
    # Time-first arrays are left-aligned: a shorter input is shared across every
    # trailing axis, so a (time,) series spans the whole spatial grid instead of
    # NumPy aligning it with the final axis.
    ndim = max(array.ndim for array in arrays)
    arrays = tuple(
        array if array.ndim == ndim else array.reshape(array.shape + (1,) * (ndim - array.ndim)) for array in arrays
    )
    try:
        broadcast = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        shapes = ", ".join(f"{name}={array.shape}" for name, array in zip(names, arrays, strict=True))
        raise InvalidArgumentError(
            f"Incompatible array shapes for daily weather inputs: {shapes}. The inputs must broadcast together.",
            argument_name="/".join(names),
            argument_value=f"shapes {shapes}",
            valid_values="Arrays broadcastable to a common time-first shape",
        ) from exc
    if broadcast[0].ndim == 0:
        raise DataShapeError(
            "Daily weather inputs must include a time dimension.",
            expected_shape="(time, ...)",
            actual_shape=broadcast[0].shape,
        )
    infinite = [name for name, array in zip(names, broadcast, strict=True) if np.any(np.isinf(array))]
    if infinite:
        raise InvalidArgumentError(
            f"{'/'.join(infinite)} must be finite or NaN: infinity is not a missing observation.",
            argument_name="/".join(infinite),
            argument_value="infinite value",
            valid_values="Finite values or NaN",
        )
    result: tuple[npt.NDArray[np.float64], ...] = tuple(broadcast)
    return result


def _month_array(month: npt.ArrayLike, weather_shape: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """Validate calendar months and broadcast them to the time-first weather shape."""
    months = _as_float_array(month)
    if np.any(~np.isfinite(months)) or np.any(months != np.floor(months)) or np.any((months < 1.0) | (months > 12.0)):
        raise InvalidArgumentError(
            "month must contain integer calendar months in [1, 12].",
            argument_name="month",
            argument_value="a non-finite, non-integral, or out-of-range value",
            valid_values="Integer values in [1, 12]",
        )
    months = months.astype(np.int64)
    if months.ndim == 1 and len(weather_shape) > 1 and months.shape[0] == weather_shape[0]:
        # a calendar month series is shared across every spatial cell
        months = months.reshape((months.shape[0],) + (1,) * (len(weather_shape) - 1))
    try:
        # the shared scalar and (time,) forms stay broadcast views instead of
        # retaining an int64 value for every time-cell
        result: npt.NDArray[np.int64] = np.broadcast_to(months, weather_shape)
    except ValueError as exc:
        raise InvalidArgumentError(
            "month must broadcast to the time-first weather shape.",
            argument_name="month",
            argument_value=f"shape {months.shape}",
            valid_values=f"A scalar or an array broadcastable to {weather_shape}",
        ) from exc
    return result


def _latitude_and_validity(
    latitude_degrees_north: npt.ArrayLike,
    spatial_shape: tuple[int, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Broadcast latitude to the spatial shape and report which cells are usable."""
    latitude = _as_float_array(latitude_degrees_north)
    if np.any(np.isinf(latitude)):
        raise InvalidArgumentError(
            "latitude_degrees_north must be finite or NaN: infinity is not a missing location.",
            argument_name="latitude_degrees_north",
            argument_value="infinite value",
            valid_values="Finite values in [-90, 90] or NaN",
        )
    if np.any(np.isfinite(latitude) & ((latitude < -90.0) | (latitude > 90.0))):
        raise InvalidArgumentError(
            "latitude_degrees_north must be within [-90, 90] where finite.",
            argument_name="latitude_degrees_north",
            argument_value="value outside [-90, 90]",
            valid_values="Finite values in [-90, 90] or NaN",
        )
    broadcast = _static_spatial_array(latitude, spatial_shape, "latitude_degrees_north")
    return broadcast, np.isfinite(broadcast)


def _initialize_single_value_state(
    *,
    seed: npt.ArrayLike | None,
    seed_name: str,
    initial_state: object,
    state_type: type[DCState] | type[DMCState] | type[FFMCState],
    value_name: str,
    default_seed: float,
    minimum: float,
    maximum: float | None,
    spatial_shape: tuple[int, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Resolve a seed or a supplied state into a recurrence's starting state."""
    bound = f"[{minimum:g}, {maximum:g}]" if maximum is not None else f"greater than or equal to {minimum:g}"
    if initial_state is None:
        if seed is None:
            value = np.full(spatial_shape, default_seed, dtype=np.float64)
        else:
            value = _static_spatial_array(seed, spatial_shape, seed_name)
            outside = np.any(value < minimum) or (maximum is not None and np.any(value > maximum))
            if np.any(~np.isfinite(value)) or outside:
                raise InvalidArgumentError(
                    f"{seed_name} must be finite and {bound}.",
                    argument_name=seed_name,
                    argument_value="non-finite or outside the valid range",
                    valid_values=bound,
                )
        return value, np.full(spatial_shape, -1, dtype=np.int64)

    if not isinstance(initial_state, state_type):
        raise InvalidArgumentError(
            f"initial_state must be a {state_type.__name__}.",
            argument_name="initial_state",
            argument_value=type(initial_state).__name__,
            valid_values=state_type.__name__,
        )
    value = _static_spatial_array(getattr(initial_state, value_name), spatial_shape, f"initial_state.{value_name}")
    trailing = initial_state.trailing_gap_days
    if trailing is None:
        trailing_gap_days = np.full(spatial_shape, -1, dtype=np.int64)
    else:
        trailing_array = _static_spatial_array(trailing, spatial_shape, "initial_state.trailing_gap_days")
        if (
            np.any(~np.isfinite(trailing_array))
            or np.any(trailing_array < -1)
            or np.any(trailing_array != np.floor(trailing_array))
        ):
            raise InvalidArgumentError(
                "initial_state.trailing_gap_days must contain integers greater than or equal to -1.",
                argument_name="initial_state.trailing_gap_days",
                argument_value="non-integral or less than -1 value",
                valid_values="-1 or a non-negative integer",
            )
        trailing_gap_days = trailing_array.astype(np.int64)

    outside = np.any(value < minimum) or (maximum is not None and np.any(value > maximum))
    if np.any(~np.isfinite(value) & ~np.isnan(value)) or outside:
        raise InvalidArgumentError(
            f"initial_state.{value_name} must be NaN or {bound}.",
            argument_name=f"initial_state.{value_name}",
            argument_value="non-finite or outside the valid range",
            valid_values=f"NaN or {bound}",
        )
    if np.any(np.isnan(value) & (trailing_gap_days < 0)):
        raise InvalidArgumentError(
            f"initial_state.{value_name} may be NaN only where trailing_gap_days shows a gap has started.",
            argument_name=f"initial_state.{value_name}",
            argument_value="NaN where initial_state.trailing_gap_days is -1 or None",
            valid_values="A finite value in a not-started cell; NaN only after a gap has started the cell",
        )
    return value, trailing_gap_days


def _active_view(
    active: npt.NDArray[np.bool_] | None,
    *arrays: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Restrict each array to the active cells, or return them unchanged for all cells."""
    if active is None:
        return arrays
    return tuple(array[active] for array in arrays)


def _run_cffwis_recurrence(
    state_value: npt.NDArray[np.float64],
    step: Callable[[int, npt.NDArray[np.bool_] | None], npt.NDArray[np.float64]],
    *,
    index_type: str,
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    trailing_gap_days: npt.NDArray[np.int64],
    memory_arrays: tuple[npt.NDArray[np.float64], ...],
    spin_up: int,
    nan_policy: Literal["propagate", "bridge"],
    max_gap_days: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64] | None]:
    """Run one time-first daily recurrence under the ADR-0007 missing-day policy.

    Thin wrapper over the shared day loop in :func:`_run_cffwis_system`, so the
    single-code functions and the combined orchestrator cannot drift apart.
    The orchestrator's all-valid fast path stays off here, keeping the
    single-code behaviour and the measured single-pass advantage unchanged.
    """
    component = _CodeRecurrence(
        index_type,
        state_value,
        step,
        weather_valid,
        static_valid,
        trailing_gap_days,
    )
    values, state_gap_days = _run_cffwis_system(
        (component,),
        memory_arrays=memory_arrays,
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
        system_name=index_type,
        fast_path=False,
    )
    return values[0], state_gap_days[0]


def _ffmc_next(
    ffmc_previous: npt.NDArray[np.float64],
    temperature_celsius: npt.NDArray[np.float64],
    relative_humidity_percent: npt.NDArray[np.float64],
    wind_speed_kilometers_per_hour: npt.NDArray[np.float64],
    precipitation_mm: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Advance the FFMC one day (Van Wagner and Pickett, 1985, Eq. 1-10)."""
    # Eq. 1: previous FFMC to fine fuel moisture content, percent
    moisture = _FFMC_COEFFICIENT * (101.0 - ffmc_previous) / (59.5 + ffmc_previous)
    rained = precipitation_mm > _FFMC_PRECIPITATION_THRESHOLD_MM
    effective_rain = np.where(rained, precipitation_mm - _FFMC_PRECIPITATION_THRESHOLD_MM, precipitation_mm)
    # Eqs. 3a and 3b: rain adds moisture, with an amendment above 150 percent
    rain_moisture = 42.5 * effective_rain * np.exp(-100.0 / (251.0 - moisture)) * (1.0 - np.exp(-6.93 / effective_rain))
    rain_moisture += np.where(
        moisture > _FFMC_MOISTURE_FOR_RAIN_CORRECTION,
        0.0015 * (moisture - _FFMC_MOISTURE_FOR_RAIN_CORRECTION) ** 2 * np.sqrt(effective_rain),
        0.0,
    )
    moisture = np.where(rained, np.minimum(moisture + rain_moisture, _FFMC_MOISTURE_CAP), moisture)

    # Eqs. 4 and 5: equilibrium moisture content for drying and wetting
    temperature_term = 0.18 * (21.1 - temperature_celsius) * (1.0 - np.exp(-0.115 * relative_humidity_percent))
    drying_equilibrium = (
        0.942 * relative_humidity_percent**0.679
        + 11.0 * np.exp((relative_humidity_percent - 100.0) / 10.0)
        + temperature_term
    )
    wetting_equilibrium = (
        0.618 * relative_humidity_percent**0.753
        + 10.0 * np.exp((relative_humidity_percent - 100.0) / 10.0)
        + temperature_term
    )

    # Eqs. 6-9: dry toward the drying equilibrium or wet toward the wetting
    # equilibrium, whichever side of it the fuel is on
    humidity_fraction = relative_humidity_percent / 100.0
    wind_root = np.sqrt(wind_speed_kilometers_per_hour)
    temperature_scale = 0.581 * np.exp(0.0365 * temperature_celsius)
    drying_rate = (
        0.424 * (1.0 - humidity_fraction**1.7) + 0.0694 * wind_root * (1.0 - humidity_fraction**8)
    ) * temperature_scale
    wetting_rate = (
        0.424 * (1.0 - (1.0 - humidity_fraction) ** 1.7) + 0.0694 * wind_root * (1.0 - (1.0 - humidity_fraction) ** 8)
    ) * temperature_scale
    dried = drying_equilibrium + (moisture - drying_equilibrium) * 10.0**-drying_rate
    wetted = wetting_equilibrium - (wetting_equilibrium - moisture) * 10.0**-wetting_rate
    moisture = np.where(
        moisture > drying_equilibrium,
        dried,
        np.where(moisture < wetting_equilibrium, wetted, moisture),
    )

    # Eq. 10: final FFMC conversion, clamped to the published range
    result = 59.5 * (250.0 - moisture) / (_FFMC_COEFFICIENT + moisture)
    return np.clip(result, 0.0, _FFMC_MAXIMUM)


def _dmc_next(
    dmc_previous: npt.NDArray[np.float64],
    temperature_celsius: npt.NDArray[np.float64],
    relative_humidity_percent: npt.NDArray[np.float64],
    precipitation_mm: npt.NDArray[np.float64],
    effective_day_length_hours: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Advance the DMC one day (Van Wagner and Pickett, 1985, Eq. 11-16)."""
    # Eq. 16: the log drying rate, with its temperature floor
    temperature = np.maximum(temperature_celsius, _DMC_TEMPERATURE_FLOOR_CELSIUS)
    drying_rate = 1.894 * (temperature + 1.1) * (100.0 - relative_humidity_percent) * effective_day_length_hours * 1e-4

    # Eqs. 11-15: rain above 1.5 mm rewets the duff layer
    rained = precipitation_mm > _DMC_PRECIPITATION_THRESHOLD_MM
    effective_rain = 0.92 * precipitation_mm - 1.27
    moisture_before = 20.0 + 280.0 / np.exp(0.023 * dmc_previous)
    # Eq. 13's piecewise slope of the moisture-content relation
    slope = np.where(
        dmc_previous <= 33.0,
        100.0 / (0.5 + 0.3 * dmc_previous),
        np.where(
            dmc_previous <= 65.0,
            14.0 - 1.3 * np.log(dmc_previous),
            6.2 * np.log(dmc_previous) - 17.2,
        ),
    )
    moisture_after = moisture_before + 1000.0 * effective_rain / (48.77 + slope * effective_rain)
    # Eq. 15 in the reference code's more accurate form
    after_rain = np.maximum(43.43 * (5.6348 - np.log(moisture_after - 20.0)), 0.0)

    previous = np.where(rained, after_rain, dmc_previous)
    return np.maximum(previous + drying_rate, 0.0)


def _dc_next(
    dc_previous: npt.NDArray[np.float64],
    temperature_celsius: npt.NDArray[np.float64],
    precipitation_mm: npt.NDArray[np.float64],
    day_length_adjustment: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Advance the DC one day (Van Wagner and Pickett, 1985, Eq. 18-23)."""
    # Eq. 22: potential evapotranspiration, floored at zero for winter
    temperature = np.maximum(temperature_celsius, _DC_TEMPERATURE_FLOOR_CELSIUS)
    potential_evapotranspiration = np.maximum((0.36 * (temperature + 2.8) + day_length_adjustment) / 2.0, 0.0)

    # Eqs. 18-21: rain above 2.8 mm reduces the drought code
    rained = precipitation_mm > _DC_PRECIPITATION_THRESHOLD_MM
    effective_rain = 0.83 * precipitation_mm - 1.27
    moisture_before = 800.0 * np.exp(-dc_previous / 400.0)
    after_rain = np.maximum(dc_previous - 400.0 * np.log(1.0 + 3.937 * effective_rain / moisture_before), 0.0)

    previous = np.where(rained, after_rain, dc_previous)
    value: npt.NDArray[np.float64] = np.maximum(previous + potential_evapotranspiration, 0.0)
    return value


def _dmc_day_length_band(latitude_degrees_north: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Index the DMC effective-day-length table for each cell's latitude band."""
    band: npt.NDArray[np.intp] = np.select(
        [
            latitude_degrees_north > 30.0,
            latitude_degrees_north > 10.0,
            latitude_degrees_north > -10.0,
            latitude_degrees_north > -30.0,
        ],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.intp)
    return band


def _dc_day_length_band(latitude_degrees_north: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Index the DC day-length-adjustment table for each cell's latitude band."""
    band: npt.NDArray[np.intp] = np.select(
        [latitude_degrees_north > 20.0, latitude_degrees_north > -20.0],
        [0, 1],
        default=2,
    ).astype(np.intp)
    return band


def ffmc(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
    precipitation_mm: npt.ArrayLike,
    *,
    initial_ffmc: npt.ArrayLike | None = None,
    initial_state: FFMCState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
) -> npt.NDArray[np.float64] | FFMCResult:
    """Compute the Fine Fuel Moisture Code (FFMC).

    The moisture content of fine surface litter and other fine fuels, the
    base of the Canadian Forest Fire Weather Index System (Van Wagner and
    Pickett, 1985). It is a daily recurrence: rain rewets the fuel, and
    temperature, relative humidity, and wind move it toward the day's
    equilibrium moisture content.

    The equations are evaluated in the source's operational units, so the
    wind speed is converted from meters per second to km/h here and nowhere
    else. The moisture-content conversion uses the reference code's exact
    ``250 * 59.5 / 101`` rather than the ``147.2`` printed in the report.

    The source's open choices are resolved here as: only rain above 0.5 mm
    rewets the fuel, moisture content is capped at 250 percent, the code is
    clamped to [0, 101], the literature seed is 85, and a day whose relative
    humidity is outside [0, 100] or whose wind speed is negative counts as a
    missing observation under ``nan_policy``.

    Args:
        temperature_celsius: Daily noon-local-standard-time air temperature,
            time-first, degrees Celsius.
        relative_humidity_percent: Daily noon-local-standard-time relative
            humidity, time-first, percent.
        wind_speed_meters_per_second: Daily 10 m wind speed, time-first,
            meters per second.
        precipitation_mm: Daily 24-hour precipitation, time-first, mm.
        initial_ffmc: Seed code, scalar or an array of the trailing spatial
            shape. ``None`` selects the literature seed of 85. Cannot be
            combined with ``initial_state``.
        initial_state: State returned by an earlier call.
        return_state: Return :class:`FFMCResult` with the final state.
        spin_up: Number of leading input days to compute but omit from the
            output.
        nan_policy: ``"propagate"`` poisons a started recurrence at a missing
            day; ``"bridge"`` skips gaps up to ``max_gap_days``.
        max_gap_days: Maximum bridged consecutive missing days. Must be zero
            for ``"propagate"`` and positive for ``"bridge"``.

    Returns:
        FFMC with the time-first shape of the broadcast weather inputs, less
        ``spin_up`` leading days. Returns :class:`FFMCResult` when
        ``return_state`` is true.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, or physical
            inputs are invalid.
    """
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_ffmc", initial_ffmc, initial_state)

    temperature, humidity, wind, precipitation = _daily_weather_arrays(
        (
            "temperature_celsius",
            "relative_humidity_percent",
            "wind_speed_meters_per_second",
            "precipitation_mm",
        ),
        temperature_celsius,
        relative_humidity_percent,
        wind_speed_meters_per_second,
        precipitation_mm,
    )
    if np.any(np.isfinite(precipitation) & (precipitation < 0.0)):
        raise InvalidArgumentError(
            "precipitation_mm must be non-negative where finite.",
            argument_name="precipitation_mm",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
        )

    spatial_shape = temperature.shape[1:]
    internal_spatial_shape = spatial_shape if spatial_shape else (1,)
    temperature = temperature.reshape(temperature.shape[0], *internal_spatial_shape)
    humidity = humidity.reshape(temperature.shape)
    precipitation = precipitation.reshape(temperature.shape)
    with np.errstate(over="ignore"):
        wind = wind.reshape(temperature.shape) * _KILOMETERS_PER_HOUR_PER_METER_PER_SECOND
    if np.any(np.isinf(wind)):
        raise InvalidArgumentError(
            "wind_speed_meters_per_second is too large to convert to kilometers per hour.",
            argument_name="wind_speed_meters_per_second",
            argument_value="finite value that overflows the km/h conversion",
            valid_values="Finite values that do not overflow the km/h conversion",
        )

    weather_valid = (
        np.isfinite(temperature)
        & np.isfinite(humidity)
        & np.isfinite(wind)
        & np.isfinite(precipitation)
        & (humidity >= 0.0)
        & (humidity <= 100.0)
        & (wind >= 0.0)
    )
    state_value, trailing_gap_days = _initialize_single_value_state(
        seed=initial_ffmc,
        seed_name="initial_ffmc",
        initial_state=initial_state,
        state_type=FFMCState,
        value_name="ffmc",
        default_seed=85.0,
        minimum=0.0,
        maximum=_FFMC_MAXIMUM,
        spatial_shape=internal_spatial_shape,
    )

    def step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        state_slice, temperature_slice, humidity_slice, wind_slice, precipitation_slice = _active_view(
            active,
            state_value,
            temperature[day],
            humidity[day],
            wind[day],
            precipitation[day],
        )
        return _ffmc_next(state_slice, temperature_slice, humidity_slice, wind_slice, precipitation_slice)

    values, state_gap_days = _run_cffwis_recurrence(
        state_value,
        step,
        index_type="ffmc",
        weather_valid=weather_valid,
        static_valid=np.ones(internal_spatial_shape, dtype=np.bool_),
        trailing_gap_days=trailing_gap_days,
        memory_arrays=(temperature, humidity, wind, precipitation),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
    )

    result = values.reshape(-1, *spatial_shape)
    if not return_state:
        return result
    return FFMCResult(
        values=result,
        state=FFMCState(
            ffmc=state_value.reshape(spatial_shape).copy(),
            trailing_gap_days=None if state_gap_days is None else state_gap_days.reshape(spatial_shape),
        ),
    )


def duff_moisture_code(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    precipitation_mm: npt.ArrayLike,
    latitude_degrees_north: npt.ArrayLike,
    month: npt.ArrayLike,
    *,
    initial_dmc: npt.ArrayLike | None = None,
    initial_state: DMCState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
) -> npt.NDArray[np.float64] | DMCResult:
    """Compute the Duff Moisture Code (DMC).

    The moisture content of loosely compacted organic layers of moderate
    depth, driven by noon temperature, relative humidity, and 24-hour rain
    (Van Wagner and Pickett, 1985). It is a daily recurrence whose drying
    rate scales with the month- and latitude-dependent effective day length.

    The effective day-length tables are selected from five latitude bands:
    46 N (latitude > 30), 20 N (10 < latitude <= 30), the equator
    (-10 < latitude <= 10), 20 S (-30 < latitude <= -10), and 40 S
    (latitude <= -30). No band is a fallback for another. The temperature
    input is floored at -1.1 C, rain above 1.5 mm rewets the layer, and the
    code is floored at zero with no upper bound.

    A NaN latitude means the cell has no usable day-length band: its output
    is always NaN and its recurrence never starts. A day whose relative
    humidity is outside [0, 100] counts as a missing observation under
    ``nan_policy``.

    Args:
        temperature_celsius: Daily noon-local-standard-time air temperature,
            time-first, degrees Celsius.
        relative_humidity_percent: Daily noon-local-standard-time relative
            humidity, time-first, percent.
        precipitation_mm: Daily 24-hour precipitation, time-first, mm.
        latitude_degrees_north: Cell latitude, scalar or an array
            broadcastable to the trailing spatial shape (for example
            ``(lat, 1)`` or ``(lat, lon)`` for a ``(time, lat, lon)`` grid),
            degrees north in [-90, 90]. NaN marks a cell with no usable band.
        month: Calendar month for each day, scalar or time-first, integer in
            [1, 12].
        initial_dmc: Seed code, scalar or an array of the trailing spatial
            shape. ``None`` selects the literature seed of 6. Cannot be
            combined with ``initial_state``.
        initial_state: State returned by an earlier call.
        return_state: Return :class:`DMCResult` with the final state.
        spin_up: Number of leading input days to compute but omit from the
            output.
        nan_policy: ``"propagate"`` poisons a started recurrence at a missing
            day; ``"bridge"`` skips gaps up to ``max_gap_days``.
        max_gap_days: Maximum bridged consecutive missing days. Must be zero
            for ``"propagate"`` and positive for ``"bridge"``.

    Returns:
        DMC with the time-first shape of the broadcast weather inputs, less
        ``spin_up`` leading days. Returns :class:`DMCResult` when
        ``return_state`` is true.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, latitude, or
            physical inputs are invalid.
    """
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_dmc", initial_dmc, initial_state)

    temperature, humidity, precipitation = _daily_weather_arrays(
        ("temperature_celsius", "relative_humidity_percent", "precipitation_mm"),
        temperature_celsius,
        relative_humidity_percent,
        precipitation_mm,
    )
    if np.any(np.isfinite(precipitation) & (precipitation < 0.0)):
        raise InvalidArgumentError(
            "precipitation_mm must be non-negative where finite.",
            argument_name="precipitation_mm",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
        )
    months = _month_array(month, temperature.shape)
    latitude, static_valid = _latitude_and_validity(latitude_degrees_north, temperature.shape[1:])

    spatial_shape = temperature.shape[1:]
    internal_spatial_shape = spatial_shape if spatial_shape else (1,)
    temperature = temperature.reshape(temperature.shape[0], *internal_spatial_shape)
    humidity = humidity.reshape(temperature.shape)
    precipitation = precipitation.reshape(temperature.shape)
    months = months.reshape(temperature.shape)
    latitude = latitude.reshape(internal_spatial_shape)
    static_valid = static_valid.reshape(internal_spatial_shape)

    weather_valid = (
        np.isfinite(temperature)
        & np.isfinite(humidity)
        & np.isfinite(precipitation)
        & (humidity >= 0.0)
        & (humidity <= 100.0)
    )
    state_value, trailing_gap_days = _initialize_single_value_state(
        seed=initial_dmc,
        seed_name="initial_dmc",
        initial_state=initial_state,
        state_type=DMCState,
        value_name="dmc",
        default_seed=6.0,
        minimum=0.0,
        maximum=None,
        spatial_shape=internal_spatial_shape,
    )
    band = _dmc_day_length_band(latitude)

    def step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        if active is None:
            effective_day_length = _DMC_EFFECTIVE_DAY_LENGTH_HOURS[band, months[day] - 1]
        else:
            effective_day_length = _DMC_EFFECTIVE_DAY_LENGTH_HOURS[band[active], months[day][active] - 1]
        state_slice, temperature_slice, humidity_slice, precipitation_slice = _active_view(
            active,
            state_value,
            temperature[day],
            humidity[day],
            precipitation[day],
        )
        return _dmc_next(state_slice, temperature_slice, humidity_slice, precipitation_slice, effective_day_length)

    values, state_gap_days = _run_cffwis_recurrence(
        state_value,
        step,
        index_type="duff_moisture_code",
        weather_valid=weather_valid,
        static_valid=static_valid,
        trailing_gap_days=trailing_gap_days,
        memory_arrays=(temperature, humidity, precipitation),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
    )

    result = values.reshape(-1, *spatial_shape)
    if not return_state:
        return result
    return DMCResult(
        values=result,
        state=DMCState(
            dmc=state_value.reshape(spatial_shape).copy(),
            trailing_gap_days=None if state_gap_days is None else state_gap_days.reshape(spatial_shape),
        ),
    )


def drought_code(
    temperature_celsius: npt.ArrayLike,
    precipitation_mm: npt.ArrayLike,
    latitude_degrees_north: npt.ArrayLike,
    month: npt.ArrayLike,
    *,
    initial_dc: npt.ArrayLike | None = None,
    initial_state: DCState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
) -> npt.NDArray[np.float64] | DCResult:
    """Compute the Drought Code (DC).

    The moisture content of deep, compact organic layers, driven by noon
    temperature and 24-hour rain (Van Wagner and Pickett, 1985). It is a
    daily recurrence whose potential evapotranspiration scales with the
    month- and latitude-dependent day length. This is the CFFWIS component
    only; it is distinct from the package's drought indices (SPI, SPEI,
    PDSI).

    The day-length adjustment is selected from three latitude bands: north
    (latitude > 20), the equator (-20 < latitude <= 20), and south
    (latitude <= -20). The temperature input is floored at -2.8 C, potential
    evapotranspiration is floored at zero, rain above 2.8 mm reduces the
    code, and the code is floored at zero with no upper bound.

    A NaN latitude means the cell has no usable day-length band: its output
    is always NaN and its recurrence never starts. Relative humidity is not
    a DC input.

    Args:
        temperature_celsius: Daily noon-local-standard-time air temperature,
            time-first, degrees Celsius.
        precipitation_mm: Daily 24-hour precipitation, time-first, mm.
        latitude_degrees_north: Cell latitude, scalar or an array
            broadcastable to the trailing spatial shape (for example
            ``(lat, 1)`` or ``(lat, lon)`` for a ``(time, lat, lon)`` grid),
            degrees north in [-90, 90]. NaN marks a cell with no usable band.
        month: Calendar month for each day, scalar or time-first, integer in
            [1, 12].
        initial_dc: Seed code, scalar or an array of the trailing spatial
            shape. ``None`` selects the literature seed of 15. Cannot be
            combined with ``initial_state``.
        initial_state: State returned by an earlier call.
        return_state: Return :class:`DCResult` with the final state.
        spin_up: Number of leading input days to compute but omit from the
            output.
        nan_policy: ``"propagate"`` poisons a started recurrence at a missing
            day; ``"bridge"`` skips gaps up to ``max_gap_days``.
        max_gap_days: Maximum bridged consecutive missing days. Must be zero
            for ``"propagate"`` and positive for ``"bridge"``.

    Returns:
        DC with the time-first shape of the broadcast weather inputs, less
        ``spin_up`` leading days. Returns :class:`DCResult` when
        ``return_state`` is true.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, latitude, or
            physical inputs are invalid.
    """
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_dc", initial_dc, initial_state)

    temperature, precipitation = _daily_weather_arrays(
        ("temperature_celsius", "precipitation_mm"),
        temperature_celsius,
        precipitation_mm,
    )
    if np.any(np.isfinite(precipitation) & (precipitation < 0.0)):
        raise InvalidArgumentError(
            "precipitation_mm must be non-negative where finite.",
            argument_name="precipitation_mm",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
        )
    months = _month_array(month, temperature.shape)
    latitude, static_valid = _latitude_and_validity(latitude_degrees_north, temperature.shape[1:])

    spatial_shape = temperature.shape[1:]
    internal_spatial_shape = spatial_shape if spatial_shape else (1,)
    temperature = temperature.reshape(temperature.shape[0], *internal_spatial_shape)
    precipitation = precipitation.reshape(temperature.shape)
    months = months.reshape(temperature.shape)
    latitude = latitude.reshape(internal_spatial_shape)
    static_valid = static_valid.reshape(internal_spatial_shape)

    weather_valid = np.isfinite(temperature) & np.isfinite(precipitation)
    state_value, trailing_gap_days = _initialize_single_value_state(
        seed=initial_dc,
        seed_name="initial_dc",
        initial_state=initial_state,
        state_type=DCState,
        value_name="dc",
        default_seed=15.0,
        minimum=0.0,
        maximum=None,
        spatial_shape=internal_spatial_shape,
    )
    band = _dc_day_length_band(latitude)

    def step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        if active is None:
            day_length_adjustment = _DC_DAY_LENGTH_ADJUSTMENT[band, months[day] - 1]
        else:
            day_length_adjustment = _DC_DAY_LENGTH_ADJUSTMENT[band[active], months[day][active] - 1]
        state_slice, temperature_slice, precipitation_slice = _active_view(
            active,
            state_value,
            temperature[day],
            precipitation[day],
        )
        return _dc_next(state_slice, temperature_slice, precipitation_slice, day_length_adjustment)

    values, state_gap_days = _run_cffwis_recurrence(
        state_value,
        step,
        index_type="drought_code",
        weather_valid=weather_valid,
        static_valid=static_valid,
        trailing_gap_days=trailing_gap_days,
        memory_arrays=(temperature, precipitation),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
    )

    result = values.reshape(-1, *spatial_shape)
    if not return_state:
        return result
    return DCResult(
        values=result,
        state=DCState(
            dc=state_value.reshape(spatial_shape).copy(),
            trailing_gap_days=None if state_gap_days is None else state_gap_days.reshape(spatial_shape),
        ),
    )


# CFFWIS behavior indices and Daily Severity Rating (#804)
#
# ISI, BUI, and the Canadian FWI are concurrent-input derivatives of the
# moisture codes above, not recurrences: they carry no state, broadcast
# elementwise, and have no missing-data policy of their own. Invalid inputs
# follow the Fosberg/HDW convention -- NaN with a logged warning rather than a
# raise. DSR is the power transform that makes the FWI seasonally averageable.


@dataclass(frozen=True)
class CFFWISState:
    """Combined final state of the three CFFWIS moisture codes.

    Each nested state follows its single-code contract
    (:class:`FFMCState`, :class:`DMCState`, :class:`DCState`), including the
    per-code ``trailing_gap_days`` bookkeeping, so resuming from a combined
    state reproduces exactly what the three separate functions would.
    """

    ffmc: FFMCState
    dmc: DMCState
    dc: DCState


@dataclass(frozen=True)
class CFFWISResult:
    """The seven CFFWIS outputs and, when requested, the combined final state.

    ``ffmc``, ``dmc``, ``dc``, ``isi``, ``bui``, ``fwi``, and ``dsr`` are
    time-first NumPy arrays. A field is ``None`` when its name was not
    selected through :func:`cffwis`'s ``outputs``; ``state`` is ``None``
    unless ``return_state=True``.
    """

    ffmc: npt.NDArray[np.float64] | None
    dmc: npt.NDArray[np.float64] | None
    dc: npt.NDArray[np.float64] | None
    isi: npt.NDArray[np.float64] | None
    bui: npt.NDArray[np.float64] | None
    fwi: npt.NDArray[np.float64] | None
    dsr: npt.NDArray[np.float64] | None
    state: CFFWISState | None = None


def _resolve_outputs(outputs: Collection[_CFFWISComponent] | str | None) -> frozenset[_CFFWISComponent]:
    """Resolve the requested CFFWIS component names, defaulting to all seven."""
    if outputs is None:
        return frozenset(_CFFWIS_COMPONENTS)
    requested: tuple[_CFFWISComponent, ...]
    if isinstance(outputs, str):
        # one-name shorthand; membership is validated below
        requested = (cast(_CFFWISComponent, outputs),)
    else:
        requested = tuple(outputs)
    unknown = sorted(name for name in requested if name not in _CFFWIS_COMPONENTS)
    if unknown:
        raise InvalidArgumentError(
            f"Unknown CFFWIS output name(s): {', '.join(repr(name) for name in unknown)}.",
            argument_name="outputs",
            argument_value=", ".join(repr(name) for name in unknown),
            valid_values=", ".join(repr(name) for name in _CFFWIS_COMPONENTS),
        )
    if not requested:
        raise InvalidArgumentError(
            "outputs must name at least one CFFWIS component.",
            argument_name="outputs",
            argument_value="empty",
            valid_values=", ".join(repr(name) for name in _CFFWIS_COMPONENTS),
        )
    return frozenset(requested)


def _broadcast_elementwise(
    index_name: str,
    argument_names: tuple[str, ...],
    *values: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Coerce and broadcast elementwise fire-index inputs, rejecting shape mismatches."""
    arrays = tuple(_as_float_array(value) for value in values)
    try:
        broadcast = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        message = (
            f"Incompatible array shapes for {index_name}: "
            + ", ".join(f"{name}={array.shape}" for name, array in zip(argument_names, arrays, strict=True))
            + ". The inputs must broadcast together."
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="/".join(argument_names),
            argument_value=f"shapes {', '.join(str(array.shape) for array in arrays)}",
            valid_values="Arrays broadcastable to a common shape",
        ) from exc
    result: tuple[npt.NDArray[np.float64], ...] = tuple(broadcast)
    return result


def _elementwise_result(
    index_name: str,
    inputs: tuple[npt.NDArray[np.float64], ...],
    evaluate: Callable[[], npt.NDArray[np.float64]],
    *,
    invalid: npt.NDArray[np.bool_],
    invalid_description: str,
) -> npt.NDArray[np.float64]:
    """Run a derived index elementwise, with the shared lifecycle logging and NaN masking."""
    log = _logger.bind(index_type=index_name, input_shape=inputs[0].shape, input_elements=inputs[0].size)
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(*inputs)
    try:
        invalid_count = int(np.count_nonzero(invalid))
        if invalid_count > 0:
            _logger.warning(f"Found {invalid_count} {invalid_description}; {index_name} is NaN there.")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            evaluated = evaluate()
        # a finite, in-range input can still overflow float64 (absurd wind or
        # FWI); the stateful codes raise there, so the elementwise path masks it
        # to NaN rather than returning an undocumented infinity
        result = np.where(invalid | ~np.isfinite(evaluated), np.nan, evaluated).astype(np.float64, copy=False)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=result.shape,
            **(memory_metrics or {}),
        )
        return result
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def _initial_spread_index(
    ffmc: npt.NDArray[np.float64],
    wind_speed_kilometers_per_hour: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Combine FFMC and wind into ISI (Van Wagner and Pickett, 1985, Eq. 24-26)."""
    moisture = _FFMC_COEFFICIENT * (101.0 - ffmc) / (59.5 + ffmc)
    wind_factor = np.exp(0.05039 * wind_speed_kilometers_per_hour)
    fine_fuel_factor = 91.9 * np.exp(-0.1386 * moisture) * (1.0 + moisture**5.31 / 49300000.0)
    return 0.208 * wind_factor * fine_fuel_factor


def _buildup_index(
    dmc: npt.NDArray[np.float64],
    dc: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Combine DMC and DC into BUI (Van Wagner and Pickett, 1985, Eq. 27)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        combined = np.where((dmc == 0.0) & (dc == 0.0), 0.0, 0.8 * dc * dmc / (dmc + 0.4 * dc))
        weight = np.where(dmc == 0.0, 0.0, (dmc - combined) / dmc)
        characteristic = 0.92 + (0.0114 * dmc) ** 1.7
        reduced = np.maximum(dmc - characteristic * weight, 0.0)
    return np.where(combined < dmc, reduced, combined)


def _cffwis_fwi(
    isi: npt.NDArray[np.float64],
    bui: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Combine ISI and BUI into the Canadian FWI (Van Wagner and Pickett, 1985, Eq. 28-30)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        damped = 1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui))
        initial = 0.1 * isi * np.where(bui > 80.0, damped, 0.626 * bui**0.809 + 2.0)
        exponentiated = np.exp(2.72 * (0.434 * np.log(initial)) ** 0.647)
    return np.where(initial <= 1.0, initial, exponentiated)


def _daily_severity_rating(fwi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Transform FWI into DSR (Van Wagner, 1987, Eq. 31)."""
    return 0.0272 * fwi**1.77


def initial_spread_index(
    ffmc: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute the Initial Spread Index (ISI).

    The expected rate of fire spread immediately after ignition, from the
    Fine Fuel Moisture Code and the 10 m wind speed (Van Wagner and Pickett,
    1985). It carries no state and broadcasts elementwise, so any shape works.

    Args:
        ffmc: Fine Fuel Moisture Code, in [0, 101].
        wind_speed_meters_per_second: 10 m wind speed, meters per second,
            non-negative. Converted to the km/h the equations are written in
            here and nowhere else.

    Returns:
        ISI with the broadcast shape of the inputs. NaN where any input is
        NaN or non-finite, where FFMC lies outside [0, 101], or where wind
        speed is negative.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.
    """
    ffmc_array, wind_array = _broadcast_elementwise(
        "initial_spread_index",
        ("ffmc", "wind_speed_meters_per_second"),
        ffmc,
        wind_speed_meters_per_second,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        wind_kilometers_per_hour = wind_array * _KILOMETERS_PER_HOUR_PER_METER_PER_SECOND
    invalid = (
        ~np.isfinite(ffmc_array)
        | ~np.isfinite(wind_kilometers_per_hour)
        | (ffmc_array < 0.0)
        | (ffmc_array > _FFMC_MAXIMUM)
        | (wind_kilometers_per_hour < 0.0)
    )
    return _elementwise_result(
        "initial_spread_index",
        (ffmc_array, wind_kilometers_per_hour),
        lambda: _initial_spread_index(ffmc_array, wind_kilometers_per_hour),
        invalid=invalid,
        invalid_description="values with FFMC outside [0, 101] or negative wind speed",
    )


def buildup_index(
    dmc: npt.ArrayLike,
    dc: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute the Buildup Index (BUI).

    A weighted combination of the Duff Moisture Code and the Drought Code
    (Van Wagner and Pickett, 1985) representing the fuel available for
    spreading. It carries no state and broadcasts elementwise.

    Args:
        dmc: Duff Moisture Code, non-negative.
        dc: Drought Code, non-negative.

    Returns:
        BUI with the broadcast shape of the inputs. NaN where any input is
        NaN or non-finite or where DMC or DC is negative.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.
    """
    dmc_array, dc_array = _broadcast_elementwise("buildup_index", ("dmc", "dc"), dmc, dc)
    invalid = ~np.isfinite(dmc_array) | ~np.isfinite(dc_array) | (dmc_array < 0.0) | (dc_array < 0.0)
    return _elementwise_result(
        "buildup_index",
        (dmc_array, dc_array),
        lambda: _buildup_index(dmc_array, dc_array),
        invalid=invalid,
        invalid_description="values with negative or non-finite DMC or DC",
    )


def cffwis_fwi(
    isi: npt.ArrayLike,
    bui: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute the Canadian Fire Weather Index (FWI).

    The final CFFWIS output, combining the Initial Spread Index and the
    Buildup Index (Van Wagner and Pickett, 1985). ``cffwis_fwi`` is named to
    keep it distinct from the Fosberg index, :func:`fosberg_ffwi`; there is no
    ``fwi()``. It carries no state and broadcasts elementwise.

    Args:
        isi: Initial Spread Index, non-negative.
        bui: Buildup Index, non-negative.

    Returns:
        FWI with the broadcast shape of the inputs. NaN where any input is
        NaN or non-finite, or negative.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.
    """
    isi_array, bui_array = _broadcast_elementwise("cffwis_fwi", ("isi", "bui"), isi, bui)
    invalid = ~np.isfinite(isi_array) | ~np.isfinite(bui_array) | (isi_array < 0.0) | (bui_array < 0.0)
    return _elementwise_result(
        "cffwis_fwi",
        (isi_array, bui_array),
        lambda: _cffwis_fwi(isi_array, bui_array),
        invalid=invalid,
        invalid_description="values with negative or non-finite ISI or BUI",
    )


def daily_severity_rating(cffwis_fwi: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute the Daily Severity Rating (DSR).

    A power transform of the Canadian FWI that makes seasonal averaging
    meaningful (Van Wagner, 1987), the form used for climatological work.
    It carries no state and broadcasts elementwise.

    Args:
        cffwis_fwi: Canadian FWI, non-negative. The parameter keeps the
            design-doc name so the transform is unambiguous about which FWI
            it consumes.

    Returns:
        DSR with the broadcast shape of the input. NaN where the FWI is NaN,
        non-finite, or negative.
    """
    fwi_array = _as_float_array(cffwis_fwi)
    invalid = ~np.isfinite(fwi_array) | (fwi_array < 0.0)
    return _elementwise_result(
        "daily_severity_rating",
        (fwi_array,),
        lambda: _daily_severity_rating(fwi_array),
        invalid=invalid,
        invalid_description="values with negative or non-finite FWI",
    )


@dataclass
class _CodeRecurrence:
    """One moisture-code recurrence threaded through the CFFWIS day loop."""

    index_type: str
    value: npt.NDArray[np.float64]
    step: Callable[[int, npt.NDArray[np.bool_] | None], npt.NDArray[np.float64]]
    weather_valid: npt.NDArray[np.bool_]
    static_valid: npt.NDArray[np.bool_]
    trailing_gap_days: npt.NDArray[np.int64]
    # derived once by the runner so the day loop has a single grouping of
    # per-component state instead of parallel index spaces
    started: npt.NDArray[np.bool_] = field(init=False)
    poisoned: npt.NDArray[np.bool_] = field(init=False)
    static_all_valid: bool = field(init=False, default=False)


def _run_cffwis_system(
    components: tuple[_CodeRecurrence, ...],
    *,
    memory_arrays: tuple[npt.NDArray[np.float64], ...],
    spin_up: int,
    nan_policy: Literal["propagate", "bridge"],
    max_gap_days: int,
    system_name: str,
    fast_path: bool,
) -> tuple[tuple[npt.NDArray[np.float64], ...], tuple[npt.NDArray[np.int64] | None, ...]]:
    """Run one or more daily recurrences through one shared time loop.

    ``step(day, active)`` returns the next code value for every cell when
    ``active`` is ``None`` and for the selected cells otherwise. Only cells
    with a valid observation whose recurrence has started and is not poisoned
    adopt it. A cell whose static input is unusable never starts: its output
    stays NaN and its state is untouched.

    Each component carries its own ADR-0007 bookkeeping because a day can be
    missing for one code and valid for another: negative wind only affects
    FFMC, humidity outside [0, 100] affects FFMC and DMC, and a NaN latitude
    only affects DMC and DC. ``fast_path`` enables the all-valid shortcut that
    the combined orchestrator uses; the single-code wrapper disables it so
    their behaviour stays identical to the pre-orchestrator engine.
    """
    n_days = components[0].weather_valid.shape[0]
    log = _logger.bind(
        index_type=system_name,
        input_shape=components[0].weather_valid.shape,
        input_elements=components[0].weather_valid.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    try:
        # the allocation is inside the try so an output-allocation failure
        # still reports the recurrence lifecycle
        values = tuple(
            np.full((max(n_days - spin_up, 0), *component.weather_valid.shape[1:]), np.nan, dtype=np.float64)
            for component in components
        )
        for component in components:
            component.started = component.trailing_gap_days >= 0
            component.poisoned = np.isnan(component.value)
            component.static_all_valid = bool(component.static_valid.all())
        memory_metrics = check_large_array_memory(*memory_arrays, *values)

        for day in range(n_days):
            for index, component in enumerate(components):
                if fast_path and component.static_all_valid and component.weather_valid[day].all():
                    # A fully valid day with a usable static input: the gap
                    # policy reduces to "every unpoisoned cell is active, the
                    # trailing count resets, and a started recurrence stays
                    # started", with no partial-mask bookkeeping needed.
                    component.trailing_gap_days.fill(0)
                    component.started.fill(True)
                    active = None if not component.poisoned.any() else ~component.poisoned
                    all_active = active is None
                else:
                    # A cell whose static input is unusable has no recurrence
                    # to gap-manage: it never starts, so it is not an elapsed
                    # missing day.
                    active = _apply_gap_policy(
                        component.value,
                        component.weather_valid[day],
                        component.static_valid,
                        component.started,
                        component.poisoned,
                        component.trailing_gap_days,
                        nan_policy=nan_policy,
                        max_gap_days=max_gap_days,
                    )
                    if not np.any(active):
                        continue
                    all_active = bool(active.all())
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    updated = component.step(day, None if all_active else active)
                if np.any(~np.isfinite(updated)):
                    raise InvalidArgumentError(
                        f"{component.index_type} produced a non-finite value from finite inputs.",
                        argument_name=component.index_type,
                        argument_value="non-finite result",
                        valid_values="Finite inputs whose result stays within float64",
                    )
                if all_active:
                    component.value[:] = updated
                else:
                    component.value[active] = updated
                if day >= spin_up:
                    output = values[index][day - spin_up]
                    if all_active:
                        output[:] = component.value
                    else:
                        assert active is not None
                        output[:] = np.where(active, component.value, np.nan)

        state_gap_days = tuple(
            component.trailing_gap_days.copy() if np.any(component.started | component.poisoned) else None
            for component in components
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=values[0].shape,
            **(memory_metrics or {}),
        )
        return values, state_gap_days
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def cffwis(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
    precipitation_mm: npt.ArrayLike,
    latitude_degrees_north: npt.ArrayLike,
    month: npt.ArrayLike,
    *,
    initial_ffmc: npt.ArrayLike | None = None,
    initial_dmc: npt.ArrayLike | None = None,
    initial_dc: npt.ArrayLike | None = None,
    initial_state: CFFWISState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
    outputs: Collection[_CFFWISComponent] | str | None = None,
) -> CFFWISResult:
    """Compute the Canadian Forest Fire Weather Index System in one pass.

    The orchestrating call for CFFWIS: it threads FFMC, DMC, and DC through a
    single daily time loop and derives ISI, BUI, the Canadian FWI, and DSR
    from the concurrent code values. Every quantity an individually chained
    set of calls produces is reproduced, without recomputing the recurrences.

    ``outputs`` lets a caller who needs only some of the seven quantities
    avoid computing and returning the rest: the moisture codes always run
    because the three recurrences share the one pass, but a derived index is
    computed only when it is selected or a selected index needs it. A field
    that is not selected is ``None``.

    The recurrence contract is ADR-0006 and the missing-day policy is
    ADR-0007, both identical to the single-code functions: ``propagate``
    poisons a started recurrence at a missing day, ``bridge`` skips gaps up to
    ``max_gap_days``, and ``spin_up`` computes but omits leading days. A day
    is missing for each code by its own inputs -- negative wind only affects
    FFMC, humidity outside [0, 100] affects FFMC and DMC, and a NaN latitude
    affects only DMC and DC -- matching what the separate functions do.

    Args:
        temperature_celsius: Daily noon-local-standard-time air temperature,
            time-first, degrees Celsius.
        relative_humidity_percent: Daily noon-local-standard-time relative
            humidity, time-first, percent.
        wind_speed_meters_per_second: Daily 10 m wind speed, time-first,
            meters per second.
        precipitation_mm: Daily 24-hour precipitation, time-first, mm.
        latitude_degrees_north: Cell latitude, scalar or an array
            broadcastable to the trailing spatial shape, degrees north in
            [-90, 90]. NaN marks a cell with no usable day-length band.
        month: Calendar month for each day, scalar or time-first, integer in
            [1, 12]. Required by the DMC and DC day-length tables: the NumPy
            layer carries no calendar, so the caller supplies it explicitly.
        initial_ffmc: Seed FFMC, scalar or an array of the trailing spatial
            shape. ``None`` selects the literature seed of 85. Cannot be
            combined with ``initial_state``.
        initial_dmc: Seed DMC. ``None`` selects the literature seed of 6.
            Cannot be combined with ``initial_state``.
        initial_dc: Seed DC. ``None`` selects the literature seed of 15.
            Cannot be combined with ``initial_state``.
        initial_state: Combined state returned by an earlier call. Cannot be
            combined with any seed.
        return_state: Attach the combined final :class:`CFFWISState`.
        spin_up: Number of leading input days to compute but omit from the
            outputs. The returned state still reflects the full input, as in
            the single-code functions.
        nan_policy: ``"propagate"`` poisons a started recurrence at a missing
            day; ``"bridge"`` skips gaps up to ``max_gap_days``.
        max_gap_days: Maximum bridged consecutive missing days. Must be zero
            for ``"propagate"`` and positive for ``"bridge"``.
        outputs: Component names to return, any subset of ``"ffmc"``,
            ``"dmc"``, ``"dc"``, ``"isi"``, ``"bui"``, ``"fwi"``, and
            ``"dsr"``. ``None`` returns all seven; a single string is
            accepted as a one-name selection. Names not selected are ``None``
            in the result.

    Returns:
        :class:`CFFWISResult` with the time-first shape of the broadcast
        weather inputs, less ``spin_up`` leading days. The combined state is
        attached when ``return_state`` is true.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, latitude,
            month, or physical inputs are invalid.
    """
    for seed_name, seed in (
        ("initial_ffmc", initial_ffmc),
        ("initial_dmc", initial_dmc),
        ("initial_dc", initial_dc),
    ):
        _validate_recurrence_options(nan_policy, max_gap_days, spin_up, seed_name, seed, initial_state)
    if initial_state is not None and not isinstance(initial_state, CFFWISState):
        raise InvalidArgumentError(
            "initial_state must be a CFFWISState.",
            argument_name="initial_state",
            argument_value=type(initial_state).__name__,
            valid_values="CFFWISState",
        )
    selected = _resolve_outputs(outputs)

    temperature, humidity, wind, precipitation = _daily_weather_arrays(
        (
            "temperature_celsius",
            "relative_humidity_percent",
            "wind_speed_meters_per_second",
            "precipitation_mm",
        ),
        temperature_celsius,
        relative_humidity_percent,
        wind_speed_meters_per_second,
        precipitation_mm,
    )
    if np.any(np.isfinite(precipitation) & (precipitation < 0.0)):
        raise InvalidArgumentError(
            "precipitation_mm must be non-negative where finite.",
            argument_name="precipitation_mm",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
        )
    months = _month_array(month, temperature.shape)
    latitude, latitude_valid = _latitude_and_validity(latitude_degrees_north, temperature.shape[1:])

    spatial_shape = temperature.shape[1:]
    internal_spatial_shape = spatial_shape if spatial_shape else (1,)
    temperature = temperature.reshape(temperature.shape[0], *internal_spatial_shape)
    humidity = humidity.reshape(temperature.shape)
    precipitation = precipitation.reshape(temperature.shape)
    months = months.reshape(temperature.shape)
    latitude = latitude.reshape(internal_spatial_shape)
    latitude_valid = latitude_valid.reshape(internal_spatial_shape)
    with np.errstate(over="ignore"):
        wind = wind.reshape(temperature.shape) * _KILOMETERS_PER_HOUR_PER_METER_PER_SECOND
    if np.any(np.isinf(wind)):
        raise InvalidArgumentError(
            "wind_speed_meters_per_second is too large to convert to kilometers per hour.",
            argument_name="wind_speed_meters_per_second",
            argument_value="finite value that overflows the km/h conversion",
            valid_values="Finite values that do not overflow the km/h conversion",
        )

    finite_temperature = np.isfinite(temperature)
    finite_humidity = np.isfinite(humidity)
    finite_precipitation = np.isfinite(precipitation)
    ffmc_weather_valid = (
        finite_temperature
        & finite_humidity
        & np.isfinite(wind)
        & finite_precipitation
        & (humidity >= 0.0)
        & (humidity <= 100.0)
        & (wind >= 0.0)
    )
    dmc_weather_valid = (
        finite_temperature & finite_humidity & finite_precipitation & (humidity >= 0.0) & (humidity <= 100.0)
    )
    dc_weather_valid = finite_temperature & finite_precipitation

    def component_state(
        seed: npt.ArrayLike | None,
        seed_name: str,
        state: FFMCState | DMCState | DCState | None,
        state_type: type[FFMCState] | type[DMCState] | type[DCState],
        value_name: str,
        default_seed: float,
        maximum: float | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        return _initialize_single_value_state(
            seed=seed,
            seed_name=seed_name,
            initial_state=state,
            state_type=state_type,
            value_name=value_name,
            default_seed=default_seed,
            minimum=0.0,
            maximum=maximum,
            spatial_shape=internal_spatial_shape,
        )

    ffmc_value, ffmc_trailing_gap_days = component_state(
        initial_ffmc,
        "initial_ffmc",
        None if initial_state is None else initial_state.ffmc,
        FFMCState,
        "ffmc",
        85.0,
        _FFMC_MAXIMUM,
    )
    dmc_value, dmc_trailing_gap_days = component_state(
        initial_dmc,
        "initial_dmc",
        None if initial_state is None else initial_state.dmc,
        DMCState,
        "dmc",
        6.0,
        None,
    )
    dc_value, dc_trailing_gap_days = component_state(
        initial_dc,
        "initial_dc",
        None if initial_state is None else initial_state.dc,
        DCState,
        "dc",
        15.0,
        None,
    )

    dmc_band = _dmc_day_length_band(latitude)
    dc_band = _dc_day_length_band(latitude)

    def ffmc_step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        state_slice, temperature_slice, humidity_slice, wind_slice, precipitation_slice = _active_view(
            active,
            ffmc_value,
            temperature[day],
            humidity[day],
            wind[day],
            precipitation[day],
        )
        return _ffmc_next(state_slice, temperature_slice, humidity_slice, wind_slice, precipitation_slice)

    def dmc_step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        if active is None:
            effective_day_length = _DMC_EFFECTIVE_DAY_LENGTH_HOURS[dmc_band, months[day] - 1]
        else:
            effective_day_length = _DMC_EFFECTIVE_DAY_LENGTH_HOURS[dmc_band[active], months[day][active] - 1]
        state_slice, temperature_slice, humidity_slice, precipitation_slice = _active_view(
            active,
            dmc_value,
            temperature[day],
            humidity[day],
            precipitation[day],
        )
        return _dmc_next(state_slice, temperature_slice, humidity_slice, precipitation_slice, effective_day_length)

    def dc_step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        if active is None:
            day_length_adjustment = _DC_DAY_LENGTH_ADJUSTMENT[dc_band, months[day] - 1]
        else:
            day_length_adjustment = _DC_DAY_LENGTH_ADJUSTMENT[dc_band[active], months[day][active] - 1]
        state_slice, temperature_slice, precipitation_slice = _active_view(
            active,
            dc_value,
            temperature[day],
            precipitation[day],
        )
        return _dc_next(state_slice, temperature_slice, precipitation_slice, day_length_adjustment)

    components = (
        _CodeRecurrence(
            "ffmc",
            ffmc_value,
            ffmc_step,
            ffmc_weather_valid,
            np.ones(internal_spatial_shape, dtype=np.bool_),
            ffmc_trailing_gap_days,
        ),
        _CodeRecurrence(
            "duff_moisture_code", dmc_value, dmc_step, dmc_weather_valid, latitude_valid, dmc_trailing_gap_days
        ),
        _CodeRecurrence("drought_code", dc_value, dc_step, dc_weather_valid, latitude_valid, dc_trailing_gap_days),
    )
    code_values, code_gap_days = _run_cffwis_system(
        components,
        memory_arrays=(temperature, humidity, wind, precipitation),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
        system_name="cffwis",
        fast_path=True,
    )
    ffmc_values, dmc_values, dc_values = code_values
    ffmc_gap_days, dmc_gap_days, dc_gap_days = code_gap_days

    # derive only what was requested: a downstream name pulls its inputs, so
    # "ffmc" alone skips the four derived arrays entirely
    needs_isi = bool(selected & {"isi", "fwi", "dsr"})
    needs_bui = bool(selected & {"bui", "fwi", "dsr"})
    isi_values = _initial_spread_index(ffmc_values, wind[spin_up:]) if needs_isi else None
    bui_values = _buildup_index(dmc_values, dc_values) if needs_bui else None
    fwi_values: npt.NDArray[np.float64] | None = None
    if selected & {"fwi", "dsr"}:
        assert isi_values is not None and bui_values is not None
        fwi_values = _cffwis_fwi(isi_values, bui_values)
    dsr_values: npt.NDArray[np.float64] | None = None
    if "dsr" in selected:
        assert fwi_values is not None
        dsr_values = _daily_severity_rating(fwi_values)

    def finalize(values_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return values_array.reshape(-1, *spatial_shape)

    result_state: CFFWISState | None = None
    if return_state:
        result_state = CFFWISState(
            ffmc=FFMCState(
                ffmc=ffmc_value.reshape(spatial_shape).copy(),
                trailing_gap_days=None if ffmc_gap_days is None else ffmc_gap_days.reshape(spatial_shape),
            ),
            dmc=DMCState(
                dmc=dmc_value.reshape(spatial_shape).copy(),
                trailing_gap_days=None if dmc_gap_days is None else dmc_gap_days.reshape(spatial_shape),
            ),
            dc=DCState(
                dc=dc_value.reshape(spatial_shape).copy(),
                trailing_gap_days=None if dc_gap_days is None else dc_gap_days.reshape(spatial_shape),
            ),
        )
    return CFFWISResult(
        ffmc=finalize(ffmc_values) if "ffmc" in selected else None,
        dmc=finalize(dmc_values) if "dmc" in selected else None,
        dc=finalize(dc_values) if "dc" in selected else None,
        isi=finalize(isi_values) if "isi" in selected and isi_values is not None else None,
        bui=finalize(bui_values) if "bui" in selected and bui_values is not None else None,
        fwi=None if "fwi" not in selected or fwi_values is None else finalize(fwi_values),
        dsr=None if "dsr" not in selected or dsr_values is None else finalize(dsr_values),
        state=result_state,
    )

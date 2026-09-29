"""Single definitions of the CFFWIS moisture codes.

This module owns everything that distinguishes the Fine Fuel Moisture Code,
the Duff Moisture Code, and the Drought Code: their constants and
latitude/month tables, their typed frozen state, the pure daily update for
each code, their seed, bounds and validity rule, and the ``_MoistureCode``
spec that binds those together. The single-code functions in
:mod:`climate_indices.fire._cffwis` and the combined ``cffwis()`` orchestrator
both build their recurrences from the same spec, so a seed, validity, or step
change lands in one place.

The equations follow Van Wagner and Pickett (1985) as implemented by the NRCan
reference code (``cffdrs_r`` and its Python port ``cffdrs_py``); all are
evaluated in the source's operational units (km/h wind, mm rain, degrees
Celsius).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from climate_indices._recurrence import (
    DailyRecurrence,
    _active_view,
    _as_float_array,
    _static_spatial_array,
    _validated_trailing_gaps,
    run_daily_recurrences,
)
from climate_indices.exceptions import InvalidArgumentError, wrap_value_error
from climate_indices.logging_config import get_logger, log_calculation_failure
from climate_indices.performance import check_large_array_memory

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

# Lawson and Armitage (2008) overwintering. The carry-over fraction weights
# the final autumn DC and the wetting efficiency weights the overwinter
# precipitation; both are region-tunable, and the defaults are the NRCan
# reference values (``cffdrs_r``/``cffdrs_py`` ``overwinter_drought_code``).
# Eq. 2 of the reference expresses the wetting term as 3.94 mm of starting
# moisture equivalent per mm of overwinter precipitation, and Eq. 4's
# start-up code is constrained to the published seed of 15.
_OVERWINTER_CARRY_OVER_FRACTION = 0.75
_OVERWINTER_WETTING_EFFICIENCY = 0.75
_OVERWINTER_WETTING_PER_MM = 3.94
_OVERWINTER_MINIMUM_START_DC = 15.0

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
        trailing_gap_days = _validated_trailing_gaps(trailing, spatial_shape, "initial_state.trailing_gap_days")

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
    in_season: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64] | None]:
    """Run one time-first daily recurrence under the ADR-0007 missing-day policy.

    Thin wrapper over the shared day loop in :func:`run_daily_recurrences`, so
    the single-code functions and the combined orchestrator cannot drift apart.
    The orchestrator's all-valid fast path stays off here, keeping the
    single-code behaviour and the measured single-pass advantage unchanged.
    """
    component = DailyRecurrence(
        index_type,
        state_value,
        step,
        weather_valid,
        static_valid,
        trailing_gap_days,
        in_season,
    )
    values, state_gap_days = run_daily_recurrences(
        (component,),
        memory_arrays=memory_arrays,
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
        system_name=index_type,
        fast_path=False,
    )
    output = values[0]
    assert output is not None  # the single-code path always records its history
    return output, state_gap_days[0]


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


def _dc_moisture_equivalent(dc: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """DC to its layer moisture equivalent (Van Wagner and Pickett, 1985, Eq. 20)."""
    return 800.0 * np.exp(-dc / 400.0)


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
    moisture_before = _dc_moisture_equivalent(dc_previous)
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
        wrap_value_error(
            exc,
            message=message,
            argument_name="/".join(argument_names),
            argument_value=f"shapes {', '.join(str(array.shape) for array in arrays)}",
            valid_values="Arrays broadcastable to a common shape",
        )
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
        log_calculation_failure(log, exc)
        raise


@dataclass(frozen=True)
class _CodeInputs:
    """The prepared internal-spatial arrays one moisture-code definition reads.

    A code reads only the fields it names; the others are ``None``. The arrays
    are time-first and already reshaped to the internal ``(time, *spatial)``
    normalization (a 1-D input carries a trailing ``(1,)`` spatial axis).
    """

    temperature: npt.NDArray[np.float64]
    precipitation: npt.NDArray[np.float64]
    humidity: npt.NDArray[np.float64] | None = None
    wind_kilometers_per_hour: npt.NDArray[np.float64] | None = None
    months: npt.NDArray[np.int64] | None = None
    day_length_band: npt.NDArray[np.intp] | None = None


Step = Callable[[int, npt.NDArray[np.bool_] | None], npt.NDArray[np.float64]]


@dataclass(frozen=True)
class _MoistureCode:
    """One moisture code's identity, seed, bounds, validity rule, and daily step.

    The single definition each of FFMC, DMC, and DC has: the single-code
    functions and the ``cffwis()`` orchestrator both build their recurrence
    from it.
    """

    index_type: str
    component: _CFFWISComponent
    state_type: type[FFMCState] | type[DMCState] | type[DCState]
    value_name: str
    default_seed: float
    maximum: float | None
    validity: Callable[[_CodeInputs], npt.NDArray[np.bool_]]
    build_step: Callable[[npt.NDArray[np.float64], _CodeInputs], Step]
    minimum: float = 0.0

    def initialize_state(
        self,
        *,
        seed: npt.ArrayLike | None,
        seed_name: str,
        initial_state: object,
        spatial_shape: tuple[int, ...],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Resolve a seed or a supplied state for this code's recurrence."""
        return _initialize_single_value_state(
            seed=seed,
            seed_name=seed_name,
            initial_state=initial_state,
            state_type=self.state_type,
            value_name=self.value_name,
            default_seed=self.default_seed,
            minimum=self.minimum,
            maximum=self.maximum,
            spatial_shape=spatial_shape,
        )


def _ffmc_validity(inputs: _CodeInputs) -> npt.NDArray[np.bool_]:
    """A day is an FFMC observation when temperature, humidity, wind and rain are usable."""
    assert inputs.humidity is not None
    assert inputs.wind_kilometers_per_hour is not None
    return (
        np.isfinite(inputs.temperature)
        & np.isfinite(inputs.humidity)
        & np.isfinite(inputs.wind_kilometers_per_hour)
        & np.isfinite(inputs.precipitation)
        & (inputs.humidity >= 0.0)
        & (inputs.humidity <= 100.0)
        & (inputs.wind_kilometers_per_hour >= 0.0)
    )


def _ffmc_build_step(state_value: npt.NDArray[np.float64], inputs: _CodeInputs) -> Step:
    """The FFMC daily step over the prepared weather arrays."""
    temperature = inputs.temperature
    humidity = cast(npt.NDArray[np.float64], inputs.humidity)
    wind = cast(npt.NDArray[np.float64], inputs.wind_kilometers_per_hour)
    precipitation = inputs.precipitation

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

    return step


def _dmc_validity(inputs: _CodeInputs) -> npt.NDArray[np.bool_]:
    """A day is a DMC observation when temperature, humidity and rain are usable."""
    assert inputs.humidity is not None
    return (
        np.isfinite(inputs.temperature)
        & np.isfinite(inputs.humidity)
        & np.isfinite(inputs.precipitation)
        & (inputs.humidity >= 0.0)
        & (inputs.humidity <= 100.0)
    )


def _dmc_build_step(state_value: npt.NDArray[np.float64], inputs: _CodeInputs) -> Step:
    """The DMC daily step over the prepared weather arrays."""
    temperature = inputs.temperature
    humidity = cast(npt.NDArray[np.float64], inputs.humidity)
    precipitation = inputs.precipitation
    months = cast(npt.NDArray[np.int64], inputs.months)
    band = cast(npt.NDArray[np.intp], inputs.day_length_band)

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

    return step


def _dc_validity(inputs: _CodeInputs) -> npt.NDArray[np.bool_]:
    """A day is a DC observation when temperature and rain are usable."""
    return np.isfinite(inputs.temperature) & np.isfinite(inputs.precipitation)


def _dc_build_step(state_value: npt.NDArray[np.float64], inputs: _CodeInputs) -> Step:
    """The DC daily step over the prepared weather arrays."""
    temperature = inputs.temperature
    precipitation = inputs.precipitation
    months = cast(npt.NDArray[np.int64], inputs.months)
    band = cast(npt.NDArray[np.intp], inputs.day_length_band)

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

    return step


FFMC_CODE = _MoistureCode(
    index_type="ffmc",
    component="ffmc",
    state_type=FFMCState,
    value_name="ffmc",
    default_seed=85.0,
    maximum=_FFMC_MAXIMUM,
    validity=_ffmc_validity,
    build_step=_ffmc_build_step,
)
DMC_CODE = _MoistureCode(
    index_type="duff_moisture_code",
    component="dmc",
    state_type=DMCState,
    value_name="dmc",
    default_seed=6.0,
    maximum=None,
    validity=_dmc_validity,
    build_step=_dmc_build_step,
)
DC_CODE = _MoistureCode(
    index_type="drought_code",
    component="dc",
    state_type=DCState,
    value_name="dc",
    default_seed=15.0,
    maximum=None,
    validity=_dc_validity,
    build_step=_dc_build_step,
)
MOISTURE_CODES: tuple[_MoistureCode, ...] = (FFMC_CODE, DMC_CODE, DC_CODE)

"""Canadian Forest Fire Weather Index System (CFFWIS).

Public NumPy and xarray entry points for the CFFWIS moisture codes and the
combined orchestrator. The moisture-code definitions and the spec that binds
each code's seed, bounds, validity rule and daily step live in
:mod:`climate_indices.fire._cffwis_codes`; the behaviour indices live in
:mod:`climate_indices.fire._cffwis_behavior`; the xarray adapter lives in
:mod:`climate_indices.fire._cffwis_xarray`.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal, cast, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices._recurrence import (
    DailyRecurrence,
    _as_float_array,
    _daily_weather_arrays,
    _static_spatial_array,
    _validate_recurrence_options,
    run_daily_recurrences,
)
from climate_indices.exceptions import InvalidArgumentError, wrap_value_error
from climate_indices.fire._cffwis_behavior import (
    _buildup_index,
    _cffwis_fwi,
    _daily_severity_rating,
    _initial_spread_index,
    buildup_index,
    cffwis_fwi,
    daily_severity_rating,
    initial_spread_index,
)
from climate_indices.fire._cffwis_codes import (
    _CFFWIS_COMPONENTS,
    _KILOMETERS_PER_HOUR_PER_METER_PER_SECOND,
    _OVERWINTER_CARRY_OVER_FRACTION,
    _OVERWINTER_MINIMUM_START_DC,
    _OVERWINTER_WETTING_EFFICIENCY,
    _OVERWINTER_WETTING_PER_MM,
    DC_CODE,
    DMC_CODE,
    FFMC_CODE,
    MOISTURE_CODES,
    DCResult,
    DCState,
    DMCResult,
    DMCState,
    FFMCResult,
    FFMCState,
    _broadcast_elementwise,
    _CFFWISComponent,
    _CodeInputs,
    _dc_day_length_band,
    _dc_moisture_equivalent,
    _dmc_day_length_band,
    _elementwise_result,
    _run_cffwis_recurrence,
)
from climate_indices.fire._native import moisture_code_recurrence

__all__ = [
    "CFFWISResult",
    "CFFWISState",
    "DCResult",
    "DCState",
    "DMCResult",
    "DMCState",
    "FFMCResult",
    "FFMCState",
    "buildup_index",
    "cffwis",
    "cffwis_fwi",
    "daily_severity_rating",
    "drought_code",
    "duff_moisture_code",
    "ffmc",
    "initial_spread_index",
    "overwinter_drought_code",
]


def _validate_non_negative_precipitation(precipitation: npt.NDArray[np.float64]) -> None:
    """Reject negative daily precipitation before any recurrence or derived index."""
    if np.any(np.isfinite(precipitation) & (precipitation < 0.0)):
        raise InvalidArgumentError(
            "precipitation_mm must be non-negative where finite.",
            argument_name="precipitation_mm",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
        )


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
        wrap_value_error(
            exc,
            message="month must broadcast to the time-first weather shape.",
            argument_name="month",
            argument_value=f"shape {months.shape}",
            valid_values=f"A scalar or an array broadcastable to {weather_shape}",
        )
    return result


def _season_mask(in_season: npt.ArrayLike, weather_shape: tuple[int, ...]) -> npt.NDArray[np.bool_]:
    """Validate and broadcast the per-day fire-season mask to the weather shape."""
    if np.ma.isMaskedArray(in_season) and np.ma.getmaskarray(in_season).any():
        # a masked weather input is a missing day; a masked season boundary has
        # no equivalent, and dropping the mask would silently choose one
        raise InvalidArgumentError(
            "in_season must not be a masked array: a masked season boundary is neither in nor out of season.",
            argument_name="in_season",
            argument_value="a masked element",
            valid_values="A boolean scalar or array with no masked elements",
        )
    mask = np.asarray(in_season)
    if mask.dtype.kind != "b":
        raise InvalidArgumentError(
            "in_season must be a boolean array.",
            argument_name="in_season",
            argument_value=f"dtype {mask.dtype}",
            valid_values="A boolean scalar or array broadcastable to the weather shape",
        )
    # Time-first arrays are left-aligned, exactly as the weather inputs are: a
    # (time,) mask is shared across every spatial cell, while an array that
    # aligns with the trailing axes is not.
    mask = (
        mask if mask.ndim >= len(weather_shape) else mask.reshape(mask.shape + (1,) * (len(weather_shape) - mask.ndim))
    )
    try:
        result: npt.NDArray[np.bool_] = np.broadcast_to(mask, weather_shape)
    except ValueError as exc:
        wrap_value_error(
            exc,
            message="in_season must broadcast to the time-first weather shape.",
            argument_name="in_season",
            argument_value=f"shape {mask.shape}",
            valid_values=f"A boolean scalar or an array broadcastable to {weather_shape}",
        )
    return result


def _validate_latitude_extent(latitude: npt.NDArray[np.float64]) -> None:
    """Reject infinite or out-of-range latitude values: NaN marks a missing location."""
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


def _latitude_and_validity(
    latitude_degrees_north: npt.ArrayLike,
    spatial_shape: tuple[int, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Broadcast latitude to the spatial shape and report which cells are usable."""
    latitude = _as_float_array(latitude_degrees_north)
    _validate_latitude_extent(latitude)
    broadcast = _static_spatial_array(latitude, spatial_shape, "latitude_degrees_north")
    return broadcast, np.isfinite(broadcast)


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
    _validate_non_negative_precipitation(precipitation)

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

    code_inputs = _CodeInputs(
        temperature=temperature,
        precipitation=precipitation,
        humidity=humidity,
        wind_kilometers_per_hour=wind,
    )
    state_value, trailing_gap_days = FFMC_CODE.initialize_state(
        seed=initial_ffmc,
        seed_name="initial_ffmc",
        initial_state=initial_state,
        spatial_shape=internal_spatial_shape,
    )
    values, state_gap_days = _run_cffwis_recurrence(
        FFMC_CODE,
        code_inputs,
        state_value,
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
    _validate_non_negative_precipitation(precipitation)
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

    code_inputs = _CodeInputs(
        temperature=temperature,
        precipitation=precipitation,
        humidity=humidity,
        months=months,
        day_length_band=_dmc_day_length_band(latitude),
    )
    state_value, trailing_gap_days = DMC_CODE.initialize_state(
        seed=initial_dmc,
        seed_name="initial_dmc",
        initial_state=initial_state,
        spatial_shape=internal_spatial_shape,
    )
    values, state_gap_days = _run_cffwis_recurrence(
        DMC_CODE,
        code_inputs,
        state_value,
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
    in_season: npt.ArrayLike | None = None,
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

    ``in_season`` selects the seasonal shutdown contract of
    ``docs/adr/0010-seasonal-carry-is-an-explicit-mask.md``: an off-season day
    is neither an observation nor a missing day, so it freezes the recurrence
    and emits the carried DC instead of a gap NaN. A cell whose recurrence has
    not started yet has no carried value and stays NaN. Off-season days are
    therefore indistinguishable from in-season days in the output and never
    poison or bridge a gap; the caller's mask is the only record of which is
    which. This is the shutdown half of overwintering, not a start-up rule:
    pass the value from :func:`overwinter_drought_code` as ``initial_dc`` for
    the next season, because the winter carry is not a recurrence.

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
        in_season: Boolean mask of the days inside the fire season, time-first
            and broadcastable to the weather shape. It is left-aligned like
            the weather inputs, so a one-dimensional mask is a season series
            shared across every spatial cell and a leading spatial mask is
            rejected rather than aligned to the trailing axes. ``None``
            treats every day as in-season, which is the default and leaves
            the recurrence unchanged.

    Returns:
        DC with the time-first shape of the broadcast weather inputs, less
        ``spin_up`` leading days. Returns :class:`DCResult` when
        ``return_state`` is true.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, latitude,
            physical inputs, or the season mask are invalid.
    """
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_dc", initial_dc, initial_state)

    temperature, precipitation = _daily_weather_arrays(
        ("temperature_celsius", "precipitation_mm"),
        temperature_celsius,
        precipitation_mm,
    )
    _validate_non_negative_precipitation(precipitation)
    months = _month_array(month, temperature.shape)
    latitude, static_valid = _latitude_and_validity(latitude_degrees_north, temperature.shape[1:])

    spatial_shape = temperature.shape[1:]
    internal_spatial_shape = spatial_shape if spatial_shape else (1,)
    temperature = temperature.reshape(temperature.shape[0], *internal_spatial_shape)
    precipitation = precipitation.reshape(temperature.shape)
    months = months.reshape(temperature.shape)
    latitude = latitude.reshape(internal_spatial_shape)
    static_valid = static_valid.reshape(internal_spatial_shape)

    code_inputs = _CodeInputs(
        temperature=temperature,
        precipitation=precipitation,
        months=months,
        day_length_band=_dc_day_length_band(latitude),
    )
    season_mask = None if in_season is None else _season_mask(in_season, code_inputs.temperature.shape)
    state_value, trailing_gap_days = DC_CODE.initialize_state(
        seed=initial_dc,
        seed_name="initial_dc",
        initial_state=initial_state,
        spatial_shape=internal_spatial_shape,
    )
    values, state_gap_days = _run_cffwis_recurrence(
        DC_CODE,
        code_inputs,
        state_value,
        static_valid=static_valid,
        trailing_gap_days=trailing_gap_days,
        memory_arrays=(temperature, precipitation),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
        in_season=season_mask,
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


def overwinter_drought_code(
    final_fall_dc: npt.ArrayLike,
    overwinter_precipitation: npt.ArrayLike,
    *,
    carry_over_fraction: float = _OVERWINTER_CARRY_OVER_FRACTION,
    wetting_efficiency: float = _OVERWINTER_WETTING_EFFICIENCY,
) -> npt.NDArray[np.float64]:
    """Compute the spring start-up Drought Code after overwintering.

    The standard overwintering method of Lawson and Armitage (2008): the
    final autumn DC is converted to its moisture equivalent (their Eq. 3),
    the overwinter precipitation refills it at the wetting efficiency (Eq. 2),
    and the spring start-up code is read back from that moisture equivalent
    (Eq. 4), constrained to the published seed of 15.

    Only the DC is overwintered: the FFMC and DMC are assumed to reach
    saturation over winter, so any error in their spring start-up quickly
    disappears, while the DC's long response time means a wrong start-up
    affects a large part of the season. Overwinter precipitation of roughly
    200 mm or more usually recharges the layer fully, and the result is then
    the seed 15.

    The carry-over fraction and wetting efficiency are region-tunable, and the
    defaults are the NRCan reference values for a fully assessed station;
    Lawson and Armitage (2008) tabulate higher fractions where the final
    autumn DC is known to be representative. Setting both to zero leaves the
    start-up moisture equivalent at zero, so the code is undefined and the
    result is NaN.

    This is the start-up half of the seasonal carry recorded in
    ``docs/adr/0010-seasonal-carry-is-an-explicit-mask.md``. The caller owns
    the season boundaries: accumulate the precipitation between the season
    shutdown and the next start-up, pass the final autumn DC from
    :func:`drought_code`, and pass the result back as ``initial_dc`` for the
    next season's :func:`drought_code` call.

    Args:
        final_fall_dc: DC on the last day of the previous fire season.
        overwinter_precipitation: Total precipitation between that day and
            the next season's start-up day, mm.
        carry_over_fraction: Weight of the final autumn DC's moisture
            equivalent in the start-up moisture equivalent, in [0, 1].
        wetting_efficiency: Fraction of the overwinter precipitation that
            refills the layer, in [0, 1].

    Returns:
        Spring start-up DC, the broadcast shape of the inputs. NaN where
        either input is NaN, non-finite, or negative, or where the start-up
        moisture equivalent is zero.

    Raises:
        InvalidArgumentError: If the shapes do not broadcast, or either
            tunable fraction is outside [0, 1].
    """
    for name, fraction in (
        ("carry_over_fraction", carry_over_fraction),
        ("wetting_efficiency", wetting_efficiency),
    ):
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float, np.integer, np.floating))
            or not np.isfinite(fraction)
            or fraction < 0.0
            or fraction > 1.0
        ):
            raise InvalidArgumentError(
                f"{name} must be a finite fraction in [0, 1].",
                argument_name=name,
                argument_value=str(fraction),
                valid_values="A finite value in [0, 1]",
            )

    dc, precipitation = _broadcast_elementwise(
        "overwinter_drought_code",
        ("final_fall_dc", "overwinter_precipitation"),
        final_fall_dc,
        overwinter_precipitation,
    )
    invalid = ~np.isfinite(dc) | (dc < 0.0) | ~np.isfinite(precipitation) | (precipitation < 0.0)

    def evaluate() -> npt.NDArray[np.float64]:
        # Eq. 3: final autumn moisture equivalent, then Eq. 2: overwinter
        # refill, then Eq. 4: the start-up code it implies
        final_moisture = _dc_moisture_equivalent(dc)
        start_moisture = (
            float(carry_over_fraction) * final_moisture
            + float(wetting_efficiency) * _OVERWINTER_WETTING_PER_MM * precipitation
        )
        with np.errstate(divide="ignore"):
            start = 400.0 * np.log(800.0 / start_moisture)
        return np.maximum(start, _OVERWINTER_MINIMUM_START_DC)

    return _elementwise_result(
        "overwinter_drought_code",
        (dc, precipitation),
        evaluate,
        invalid=invalid,
        invalid_description="NaN or negative overwintering inputs",
    )


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
    time-first NumPy arrays; on the xarray route they are ``xr.DataArray``
    objects with the same shapes. A field is ``None`` when its name was not
    selected through :func:`cffwis`'s ``outputs``; ``state`` is ``None``
    unless ``return_state=True``. The state stays plain NumPy on both routes
    per ``docs/adr/0006-fire-recursive-state-and-execution.md``.
    """

    ffmc: npt.NDArray[np.float64] | xr.DataArray | None
    dmc: npt.NDArray[np.float64] | xr.DataArray | None
    dc: npt.NDArray[np.float64] | xr.DataArray | None
    isi: npt.NDArray[np.float64] | xr.DataArray | None
    bui: npt.NDArray[np.float64] | xr.DataArray | None
    fwi: npt.NDArray[np.float64] | xr.DataArray | None
    dsr: npt.NDArray[np.float64] | xr.DataArray | None
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
    unknown = sorted(
        (name for name in requested if not isinstance(name, str) or name not in _CFFWIS_COMPONENTS),
        key=repr,
    )
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


@overload
def cffwis(
    temperature_celsius: xr.DataArray,  # NOSONAR (S107) the public API mirrors the per-code helpers
    relative_humidity_percent: xr.DataArray,
    wind_speed_meters_per_second: xr.DataArray,
    precipitation_mm: xr.DataArray,
    latitude_degrees_north: npt.ArrayLike | xr.DataArray | None = None,
    month: npt.ArrayLike | xr.DataArray | None = None,
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
    time_dim: str = "time",
) -> xr.Dataset | CFFWISResult: ...


@overload
def cffwis(
    temperature_celsius: npt.ArrayLike,  # NOSONAR (S107) the public API mirrors the per-code helpers
    relative_humidity_percent: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
    precipitation_mm: npt.ArrayLike,
    latitude_degrees_north: npt.ArrayLike | None = None,
    month: npt.ArrayLike | None = None,
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
    time_dim: str = "time",
) -> CFFWISResult: ...


def cffwis(
    temperature_celsius: npt.ArrayLike | xr.DataArray,  # NOSONAR (S107) the public API mirrors the per-code helpers
    relative_humidity_percent: npt.ArrayLike | xr.DataArray,
    wind_speed_meters_per_second: npt.ArrayLike | xr.DataArray,
    precipitation_mm: npt.ArrayLike | xr.DataArray,
    latitude_degrees_north: npt.ArrayLike | xr.DataArray | None = None,
    month: npt.ArrayLike | xr.DataArray | None = None,
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
    time_dim: str = "time",
) -> CFFWISResult | xr.Dataset:
    """Compute the Canadian Forest Fire Weather Index System in one pass.

    The orchestrating call for CFFWIS: it threads FFMC, DMC, and DC through a
    single daily time loop and derives ISI, BUI, the Canadian FWI, and DSR
    from the concurrent code values. Every quantity an individually chained
    set of calls produces is reproduced, without recomputing the recurrences.

    This function accepts both NumPy arrays and xarray DataArrays. Type
    checkers narrow the return type based on the input type.

    .. warning:: **Beta Feature (xarray path only)** -- When called with
       ``xr.DataArray`` weather inputs, this function uses the beta xarray
       adapter layer: CF ``units``-attribute conversion, latitude and month
       inference from coordinates, per-variable CF metadata from the registry,
       and Dask spatial-chunk parallelism with a required single ``time``
       chunk. The NumPy array interface and underlying computation are stable.

    ``outputs`` lets a caller who needs only some of the seven quantities
    avoid computing and returning the rest: the moisture codes always run
    because the three recurrences share the one pass, but a derived index is
    computed only when it is selected or a selected index needs it, and a
    moisture code's daily history is kept only when a selected output reads
    it. A field that is not selected is ``None``.

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
            [-90, 90]. NaN marks a cell with no usable day-length band. For
            xarray input, ``None`` infers it from a ``lat`` or ``latitude``
            coordinate shared by the weather inputs.
        month: Calendar month for each day, scalar or time-first, integer in
            [1, 12]. Required by the DMC and DC day-length tables: the NumPy
            layer carries no calendar, so the caller supplies it explicitly.
            For xarray input, ``None`` infers it from a datetime ``time``
            coordinate; a scalar or a 1-D time series is accepted in place of
            a DataArray.
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
        time_dim: Name of the time dimension. Only used for xarray inputs; a
            dimension-only time axis is aligned positionally.

    Returns:
        For NumPy input, :class:`CFFWISResult` with the time-first shape of
        the broadcast weather inputs, less ``spin_up`` leading days; the
        combined state is attached when ``return_state`` is true. For xarray
        input, an :class:`xarray.Dataset` holding one time-first variable per
        selected output, each carrying the CF metadata of its ``CF_METADATA``
        registry entry, plus the ``climate_indices`` version and history
        attributes. With ``return_state=True`` the xarray route instead
        returns :class:`CFFWISResult` whose selected component fields are
        DataArrays and whose ``state`` stays plain NumPy per
        ``docs/adr/0006-fire-recursive-state-and-execution.md``; the state
        arrays are computed eagerly while the component DataArrays stay lazy.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, latitude,
            month, or physical inputs are invalid.
        CoordinateValidationError: xarray input only -- if the time dimension
            is missing, an attached time coordinate is non-monotonic or not
            consecutive daily, the time dimension is split across multiple
            Dask chunks, input alignment drops grid cells, or the inputs share
            no overlapping time steps.
        TypeError: If some of the four weather inputs are DataArrays and
            others are NumPy-compatible array-likes.

    Notes:
        xarray-only: all four weather inputs must be the same type (all NumPy
        or all ``xr.DataArray``). A CF ``units`` attribute on temperature or
        precipitation is converted to degrees Celsius / mm; wind must already
        be m s-1 and relative humidity in percent. ``latitude_degrees_north``
        and ``month`` may be omitted only when they can be inferred from
        coordinates. CFFWIS assumes noon local-standard-time observations, so
        a daily time coordinate sampling at an hour other than 12:00 emits a
        :class:`~climate_indices.exceptions.ClimateIndicesWarning`. Dask-backed
        input parallelizes over spatial chunks with the ``time`` dimension
        required to be a single chunk.
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

    is_xarray = isinstance(temperature_celsius, xr.DataArray)
    for argument_name, value in (
        ("relative_humidity_percent", relative_humidity_percent),
        ("wind_speed_meters_per_second", wind_speed_meters_per_second),
        ("precipitation_mm", precipitation_mm),
    ):
        if isinstance(value, xr.DataArray) != is_xarray:
            raise TypeError(
                f"{argument_name} must be the same type as temperature_celsius. "
                f"Got {argument_name}={type(value).__name__}, "
                f"temperature_celsius={type(temperature_celsius).__name__}. "
                "Convert all four weather inputs to the same type "
                "(both numpy arrays or both xr.DataArray)."
            )
    if is_xarray:
        # narrow for mypy: the type check above guarantees all four are DataArrays
        assert isinstance(temperature_celsius, xr.DataArray)
        assert isinstance(relative_humidity_percent, xr.DataArray)
        assert isinstance(wind_speed_meters_per_second, xr.DataArray)
        assert isinstance(precipitation_mm, xr.DataArray)
        # imported here to keep the NumPy core importable without the xarray layer
        from climate_indices.fire._cffwis_xarray import _cffwis_xarray, _CFFWISCallOptions

        return _cffwis_xarray(
            temperature_celsius,
            relative_humidity_percent,
            wind_speed_meters_per_second,
            precipitation_mm,
            latitude_degrees_north,
            month,
            _CFFWISCallOptions(
                initial_ffmc=initial_ffmc,
                initial_dmc=initial_dmc,
                initial_dc=initial_dc,
                initial_state=initial_state,
                return_state=return_state,
                spin_up=spin_up,
                nan_policy=nan_policy,
                max_gap_days=max_gap_days,
                selected=selected,
                time_dim=time_dim,
            ),
        )
    if latitude_degrees_north is None:
        raise InvalidArgumentError(
            "latitude_degrees_north is required for NumPy array input.",
            argument_name="latitude_degrees_north",
            argument_value="None",
            valid_values="A scalar or an array broadcastable to the trailing spatial shape",
        )
    if month is None:
        raise InvalidArgumentError(
            "month is required for NumPy array input.",
            argument_name="month",
            argument_value="None",
            valid_values="A scalar or an array broadcastable to the time-first weather shape",
        )

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
    _validate_non_negative_precipitation(precipitation)
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

    code_inputs = (
        _CodeInputs(
            temperature=temperature,
            precipitation=precipitation,
            humidity=humidity,
            wind_kilometers_per_hour=wind,
        ),
        _CodeInputs(
            temperature=temperature,
            precipitation=precipitation,
            humidity=humidity,
            months=months,
            day_length_band=_dmc_day_length_band(latitude),
        ),
        _CodeInputs(
            temperature=temperature,
            precipitation=precipitation,
            months=months,
            day_length_band=_dc_day_length_band(latitude),
        ),
    )
    ffmc_value, ffmc_trailing_gap_days = FFMC_CODE.initialize_state(
        seed=initial_ffmc,
        seed_name="initial_ffmc",
        initial_state=None if initial_state is None else initial_state.ffmc,
        spatial_shape=internal_spatial_shape,
    )
    dmc_value, dmc_trailing_gap_days = DMC_CODE.initialize_state(
        seed=initial_dmc,
        seed_name="initial_dmc",
        initial_state=None if initial_state is None else initial_state.dmc,
        spatial_shape=internal_spatial_shape,
    )
    dc_value, dc_trailing_gap_days = DC_CODE.initialize_state(
        seed=initial_dc,
        seed_name="initial_dc",
        initial_state=None if initial_state is None else initial_state.dc,
        spatial_shape=internal_spatial_shape,
    )

    code_values_init = (ffmc_value, dmc_value, dc_value)
    code_gaps_init = (ffmc_trailing_gap_days, dmc_trailing_gap_days, dc_trailing_gap_days)
    # the FFMC has no latitude input, so every cell is static-valid for it
    component_list: list[DailyRecurrence] = []
    for code, value, gaps, inputs in zip(MOISTURE_CODES, code_values_init, code_gaps_init, code_inputs, strict=True):
        weather_valid = code.validity(inputs)
        valid_cells = (
            np.ones(internal_spatial_shape, dtype=np.bool_) if inputs.day_length_band is None else latitude_valid
        )
        component_list.append(
            DailyRecurrence(
                code.index_type,
                value,
                code.build_step(value, inputs),
                weather_valid,
                valid_cells,
                gaps,
                None,
                moisture_code_recurrence(code, inputs, value, weather_valid, valid_cells, gaps, None),
            )
        )
    components = tuple(component_list)
    # a code keeps its daily history only when a selected output reads it: the
    # direct name, or a derived index whose formula consumes the series
    code_consumers = {
        "ffmc": {"isi", "fwi", "dsr"},
        "dmc": {"bui", "fwi", "dsr"},
        "dc": {"bui", "fwi", "dsr"},
    }
    record = tuple(bool(selected & ({code.component} | code_consumers[code.component])) for code in MOISTURE_CODES)
    code_values, code_gap_days = run_daily_recurrences(
        components,
        memory_arrays=(temperature, humidity, wind, precipitation),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
        system_name="cffwis",
        fast_path=True,
        record=record,
    )
    ffmc_values, dmc_values, dc_values = code_values
    ffmc_gap_days, dmc_gap_days, dc_gap_days = code_gap_days

    # derive only what was requested: a downstream name pulls its inputs, so
    # "ffmc" alone skips the four derived arrays entirely
    needs_isi = bool(selected & {"isi", "fwi", "dsr"})
    needs_bui = bool(selected & {"bui", "fwi", "dsr"})
    if needs_isi:
        assert ffmc_values is not None
        isi_values = _initial_spread_index(ffmc_values, wind[spin_up:])
    else:
        isi_values = None
    if needs_bui:
        assert dmc_values is not None and dc_values is not None
        bui_values = _buildup_index(dmc_values, dc_values)
    else:
        bui_values = None
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
        ffmc=finalize(ffmc_values) if "ffmc" in selected and ffmc_values is not None else None,
        dmc=finalize(dmc_values) if "dmc" in selected and dmc_values is not None else None,
        dc=finalize(dc_values) if "dc" in selected and dc_values is not None else None,
        isi=finalize(isi_values) if "isi" in selected and isi_values is not None else None,
        bui=finalize(bui_values) if "bui" in selected and bui_values is not None else None,
        fwi=None if "fwi" not in selected or fwi_values is None else finalize(fwi_values),
        dsr=None if "dsr" not in selected or dsr_values is None else finalize(dsr_values),
        state=result_state,
    )

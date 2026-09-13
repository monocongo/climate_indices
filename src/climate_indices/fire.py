"""Fire-weather indices computed from standard meteorological inputs.

This module is the NumPy layer of the fire-weather family tracked in #793. It
currently provides the Fosberg Fire Weather Index and the Hot-Dry-Windy
Index, both weather-only and carrying no state between time steps.

References
----------
Srock, A.F., Charney, J.J., Potter, B.E., Goodrick, S.L. (2018) The
Hot-Dry-Windy Index: A New Fire Weather Index. Atmosphere, 9(7), 279.
doi:10.3390/atmos9070279.

Fosberg, M.A. (1978) Weather in wildland fire management: the fire weather
index. Conference on Sierra Nevada Meteorology, Lake Tahoe, CA, 1-4.

Simard, A.J. (1968) The moisture content of forest fuels - I. A review of the
basic concepts. Canadian Department of Forest and Rural Development, Forest
Fire Research Institute, Information Report FF-X-14.

Goodrick, S.L. (2002) Modification of the Fosberg fire weather index to include
drought. International Journal of Wildland Fire, 11, 205-211.
NCEP GEMPAK, ``pd_fosb`` / ``pr_fosb`` (T. Lee, 2003): the operational
implementation behind the ``FOSINDX`` GRIB2 parameter.
https://github.com/Unidata/gempak
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from climate_indices import pm_eto
from climate_indices.exceptions import DataShapeError, InvalidArgumentError
from climate_indices.logging_config import get_logger
from climate_indices.performance import check_large_array_memory

# retrieve structlog logger for this module
_logger = get_logger(__name__)

# declare the function names that should be included in the public API for this module
__all__ = ["fosberg_ffwi", "hot_dry_windy"]

# Simard (1968) equilibrium moisture content regressions, one per relative
# humidity range, with the coefficients of NCEP's operational GEMPAK code.
# Published restatements print the middle-range temperature coefficient as
# 0.01478 rather than 0.014784; across physical temperatures that moves the
# moisture content by less than 0.0005 and the index by less than 0.01.
_EMC_LOW_RH = (0.03229, 0.281073, 0.000578)
_EMC_MID_RH = (2.22749, 0.160107, 0.014784)
_EMC_HIGH_RH = (21.0606, 0.005565, 0.00035, 0.483199)

# Relative humidity breakpoints, upper bound inclusive as in GEMPAK. The
# published equations are written "h < 10" and "10 < h <= 50", which leaves
# exactly 10% in neither range.
_RH_BREAK_LOW = 10.0
_RH_BREAK_HIGH = 50.0

# moisture content at which the damping coefficient reaches zero
_EMC_EXTINCTION = 30.0

# scales the index to 100 at zero fuel moisture and a 30 mph wind
_FFWI_NORMALIZER = 0.3002
_FFWI_CAP = 100.0

# exact, by the definition of the international mile
_METERS_PER_SECOND_PER_MPH = 0.44704

# HDW analyzes the lowest 500 m above ground level (Srock et al., 2018)
_HDW_LAYER_TOP_METERS = 500.0

# pm_eto saturation vapor pressure is kPa; HDW reports VPD in hPa
_KPA_PER_HPA = 0.1

_RecurrenceStep = Callable[..., tuple[npt.NDArray[np.float64], ...]]


def _recurse(
    inputs: tuple[npt.NDArray[np.float64], ...],
    initial_state: tuple[npt.ArrayLike, ...],
    step: _RecurrenceStep,
) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], ...]]:
    """Apply a time-first recurrence and retain its first state as output.

    Fire implementations pass daily, equal-shaped input arrays and a step
    function returning the next state tuple. The first state is the index
    value; remaining states retain recurrence bookkeeping such as KBDI's
    wet-spell precipitation across an append boundary.
    """
    input_shape = _check_recurrence_inputs(inputs)
    spatial_shape = input_shape[1:]
    state = _initial_state_arrays(initial_state, spatial_shape)
    state_count = len(state)
    result = np.empty(input_shape, dtype=np.float64)
    for day in range(input_shape[0]):
        state = _apply_step(step, state, inputs, day, state_count, spatial_shape)
        result[day] = state[0]

    return result, tuple(value.copy() for value in state)


def _check_recurrence_inputs(
    inputs: tuple[npt.NDArray[np.float64], ...],
) -> tuple[int, ...]:
    """Validate daily recurrence inputs and return their shared shape."""
    if not inputs:
        raise InvalidArgumentError(
            "A recurrence requires at least one daily input array.",
            argument_name="inputs",
        )
    input_shape = inputs[0].shape
    if not input_shape:
        raise DataShapeError(
            "Daily recurrence inputs require a time dimension.",
            actual_shape=input_shape,
        )
    for values in inputs[1:]:
        if values.shape != input_shape:
            raise DataShapeError(
                "Daily recurrence inputs must have equal shapes.",
                expected_shape=str(input_shape),
                actual_shape=values.shape,
            )
    return input_shape


def _initial_state_arrays(
    initial_state: tuple[npt.ArrayLike, ...],
    spatial_shape: tuple[int, ...],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Broadcast each initial state value to the spatial input shape."""
    if not initial_state:
        raise InvalidArgumentError(
            "A recurrence requires at least one state value.",
            argument_name="initial_state",
        )
    state_values: list[npt.NDArray[np.float64]] = []
    for value in initial_state:
        initial_value = np.asarray(value, dtype=np.float64)
        try:
            state_values.append(np.broadcast_to(initial_value, spatial_shape).copy())
        except ValueError as exc:
            raise DataShapeError(
                "Initial recurrence state must broadcast to the spatial input shape.",
                expected_shape=str(spatial_shape),
                actual_shape=initial_value.shape,
            ) from exc
    return tuple(state_values)


def _apply_step(
    step: _RecurrenceStep,
    state: tuple[npt.NDArray[np.float64], ...],
    inputs: tuple[npt.NDArray[np.float64], ...],
    day: int,
    state_count: int,
    spatial_shape: tuple[int, ...],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Advance the recurrence one day, validating the returned state."""
    # copy: a step may return views into the inputs or reuse its own buffers;
    # the recurrence must own every state array so steps can't mutate inputs
    next_state = tuple(np.array(value, dtype=np.float64) for value in step(*state, *(values[day] for values in inputs)))
    if len(next_state) != state_count:
        raise InvalidArgumentError(
            "A recurrence step must return the complete state tuple it was given.",
            argument_name="step",
            argument_value=f"{len(next_state)} state values, expected {state_count}",
        )
    for value in next_state:
        if value.shape != spatial_shape:
            raise DataShapeError(
                "A recurrence step must keep each state's spatial shape.",
                expected_shape=str(spatial_shape),
                actual_shape=value.shape,
            )
    return next_state


def _equilibrium_moisture_content(
    temperature_fahrenheit: npt.NDArray[np.float64],
    relative_humidity_percent: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Simard (1968) equilibrium moisture content, in percent.

    The three regressions were fitted independently and do not meet at the
    breakpoints. At 10% relative humidity the result jumps by roughly 0.5 to
    0.7 near typical fire-weather temperatures, with a size and sign that
    depend on temperature and grow larger toward the cold end of this
    module's supported range; at 50% the jump is smaller, roughly 0.5. That
    is a property of the published equations, not of this implementation,
    and the index inherits a jump of roughly one unit.

    Args:
        temperature_fahrenheit: Air temperature, degrees Fahrenheit.
        relative_humidity_percent: Relative humidity, percent.

    Returns:
        Equilibrium moisture content, percent.
    """
    t = temperature_fahrenheit
    h = relative_humidity_percent

    a, b, c = _EMC_LOW_RH
    low = a + b * h - c * h * t

    d, e, f = _EMC_MID_RH
    mid = d + e * h - f * t

    g, k, p, q = _EMC_HIGH_RH
    high = g + k * h * h - p * h * t - q * h

    return np.where(h <= _RH_BREAK_LOW, low, np.where(h <= _RH_BREAK_HIGH, mid, high))


def _moisture_damping(equilibrium_moisture_content: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Fosberg's moisture damping coefficient.

    With ``x = m / 30`` the coefficient ``1 - 2x + 1.5x**2 - 0.5x**3`` factors as
    ``(1 - x)(0.5x**2 - x + 1)``, and the second factor is positive for every
    real ``x``. The coefficient is therefore negative exactly when ``m`` exceeds
    30, which the high-humidity regression reaches in saturated air below about
    -46 F (-43 C). Clamping ``m`` at 30 keeps the index at zero there rather than
    letting it go negative.

    Args:
        equilibrium_moisture_content: Equilibrium moisture content, percent.

    Returns:
        Damping coefficient in [0, 1] for non-negative moisture content.
    """
    x = np.minimum(equilibrium_moisture_content, _EMC_EXTINCTION) / _EMC_EXTINCTION
    return 1.0 - 2.0 * x + 1.5 * x**2 - 0.5 * x**3


def _ffwi(
    equilibrium_moisture_content: npt.NDArray[np.float64],
    wind_speed_mph: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Uncapped Fosberg index from moisture content and wind speed in mph."""
    return _moisture_damping(equilibrium_moisture_content) * np.sqrt(1.0 + wind_speed_mph**2) / _FFWI_NORMALIZER


def _as_float_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce to float64, turning masked elements into NaN instead of dropping the mask."""
    filled = np.ma.asarray(values, dtype=np.float64).filled(np.nan)
    return np.asarray(filled, dtype=np.float64)


def fosberg_ffwi(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
    cap_at_100: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute the Fosberg Fire Weather Index (FFWI).

    A weather-only index of fire-weather potential from temperature, relative
    humidity and wind speed (Fosberg, 1978). It carries no state between time
    steps, so it is computed elementwise: the inputs broadcast against each
    other and any shape works, including arrays chunked along time.

    Inputs are SI, like the rest of the package and like NCEP's operational
    implementation. The conversion to the degrees Fahrenheit and miles per
    hour that the equations are written in happens here and nowhere else.

    Relative humidity exactly at 10% or 50% is assigned to the lower range, as
    in NCEP's GEMPAK code; the published equations leave exactly 10% in
    neither range. The moisture regressions are discontinuous at both
    breakpoints, so the index is too, by roughly one unit.

    Args:
        temperature_celsius: Air temperature, degrees Celsius.
        relative_humidity_percent: Relative humidity, percent, in [0, 100].
        wind_speed_meters_per_second: Wind speed, meters per second,
            non-negative.
        cap_at_100: Clamp the index at 100, the value Fosberg assigned to zero
            fuel moisture and a 30 mph wind, giving the conventional 0-100
            scale. NCEP's GEMPAK implementation does not clamp, and some
            studies deliberately keep values above 100; pass ``False`` to
            reproduce them. This only affects the upper bound: the moisture
            content clamp that keeps the index at 0 rather than negative in
            cold, saturated air (see ``_moisture_damping``) is always
            applied, independent of this flag.

    Returns:
        FFWI with the broadcast shape of the inputs. NaN where any input is
        NaN or masked, where relative humidity lies outside [0, 100], or where
        wind speed is negative.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.

    Example:
        >>> from climate_indices import fire
        >>> round(float(fire.fosberg_ffwi(30.0, 15.0, 10.0)), 2)
        59.24
    """
    temperature = _as_float_array(temperature_celsius)
    humidity = _as_float_array(relative_humidity_percent)
    wind = _as_float_array(wind_speed_meters_per_second)

    try:
        temperature, humidity, wind = np.broadcast_arrays(temperature, humidity, wind)
    except ValueError as exc:
        message = (
            "Incompatible array shapes for Fosberg FFWI: "
            f"temperature={temperature.shape}, relative_humidity={humidity.shape}, "
            f"wind_speed={wind.shape}. The inputs must broadcast together."
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="temperature_celsius/relative_humidity_percent/wind_speed_meters_per_second",
            argument_value=f"shapes {temperature.shape}, {humidity.shape}, {wind.shape}",
            valid_values="Arrays broadcastable to a common shape",
        ) from exc

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="fosberg_ffwi",
        input_shape=temperature.shape,
        input_elements=temperature.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(temperature, humidity, wind)

    try:
        # outside the physical range the regressions still return numbers, but
        # meaningless ones, so treat such values as missing
        invalid = (humidity < 0.0) | (humidity > 100.0) | (wind < 0.0)
        invalid_count = int(np.count_nonzero(invalid))
        if invalid_count > 0:
            _logger.warning(
                f"Found {invalid_count} values with relative humidity outside [0, 100] "
                "or negative wind speed; FFWI is NaN there."
            )

        temperature_fahrenheit = temperature * 9.0 / 5.0 + 32.0
        wind_speed_mph = wind / _METERS_PER_SECOND_PER_MPH

        emc = _equilibrium_moisture_content(temperature_fahrenheit, humidity)
        index = _ffwi(emc, wind_speed_mph)
        if cap_at_100:
            index = np.minimum(index, _FFWI_CAP)

        result = np.where(invalid, np.nan, index).astype(np.float64, copy=False)
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


def hot_dry_windy(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
    height_agl_meters: npt.ArrayLike,
    *,
    level_axis: int = -1,
) -> npt.NDArray[np.float64]:
    """Compute the Hot-Dry-Windy Index (HDW).

    A weather-only index of dangerous fire-behavior potential (Srock et al.,
    2018): the vapor pressure deficit (VPD) times the wind speed, maximized
    over the levels in the lowest 500 m above ground level (AGL)::

        HDW = max over levels with 0 <= height_agl <= 500 of (VPD * wind speed)

    Inputs are vertical profiles with SI units, like the rest of the package.
    The four inputs broadcast against each other; the shared dimension
    ``level_axis`` is the vertical coordinate and is reduced by the maximum.
    VPD comes from each level's own temperature and relative humidity, with
    saturation vapor pressure from ``pm_eto.saturation_vapor_pressure`` (FAO-56
    Eq 11), converted from kPa to the hPa of the published index.

    ``height_agl_meters`` is the vertical coordinate itself, so the AGL
    determination happens where that coordinate is built:

    - Model-level input (e.g. CFSR): use the model's own height field, or
      geopotential height minus the surface geopotential height of the grid
      cell.
    - Pressure-level input: convert each pressure level to geopotential height
      (hypsometric equation) and subtract the surface height of the grid cell.

    The result is sensitive to vertical resolution: coarse level spacing can
    miss the level where VPD and wind combine worst, and sampling more levels
    inside the layer can only raise the maximum. Compare HDW across datasets
    only at comparable vertical resolution. Srock et al. (2018) additionally
    adiabatically adjust each level's VPD to the surface and take the VPD and
    wind maxima independently (so they may come from different levels); this
    implementation follows the simplified formulation of the issue contract,
    the per-level product, which never exceeds the published variant.

    Args:
        temperature_celsius: Air temperature profile, degrees Celsius.
        relative_humidity_percent: Relative humidity profile, percent, in
            [0, 100].
        wind_speed_meters_per_second: Wind speed profile, meters per second,
            non-negative.
        height_agl_meters: Height above ground level of each level, meters.
            Levels outside [0, 500], or with NaN height, are excluded from the
            maximum.
        level_axis: Axis of the broadcast inputs that holds the vertical
            coordinate. Reduced by the layer maximum.

    Returns:
        HDW in hPa m s-1, with the broadcast shape of the inputs minus
        ``level_axis``. NaN where any in-layer level has NaN or out-of-range
        input, and for columns with no level inside the lowest 500 m AGL.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together, or
            ``level_axis`` is out of range for the broadcast shape.

    Example:
        >>> from climate_indices import fire
        >>> round(float(fire.hot_dry_windy([30.0, 26.0], [15.0, 30.0], [8.0, 12.0], [10.0, 400.0])), 2)
        288.53
    """
    temperature = _as_float_array(temperature_celsius)
    humidity = _as_float_array(relative_humidity_percent)
    wind = _as_float_array(wind_speed_meters_per_second)
    height = _as_float_array(height_agl_meters)

    try:
        temperature, humidity, wind, height = np.broadcast_arrays(temperature, humidity, wind, height)
    except ValueError as exc:
        message = (
            "Incompatible array shapes for Hot-Dry-Windy Index: "
            f"temperature={temperature.shape}, relative_humidity={humidity.shape}, "
            f"wind_speed={wind.shape}, height_agl={height.shape}. The inputs must broadcast together."
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="temperature_celsius/relative_humidity_percent/wind_speed_meters_per_second/height_agl_meters",
            argument_value=f"shapes {temperature.shape}, {humidity.shape}, {wind.shape}, {height.shape}",
            valid_values="Arrays broadcastable to a common shape",
        ) from exc

    if temperature.ndim == 0:
        # a single level is a degenerate profile; give it an axis to reduce
        temperature, humidity, wind, height = (a.reshape(1) for a in (temperature, humidity, wind, height))

    if not -temperature.ndim <= level_axis < temperature.ndim:
        message = (
            f"level_axis {level_axis} is out of range for the broadcast input shape "
            f"{temperature.shape} with {temperature.ndim} dimensions."
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="level_axis",
            argument_value=str(level_axis),
            valid_values=f"An axis of the broadcast shape {temperature.shape}",
        )
    axis = level_axis % temperature.ndim

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="hot_dry_windy",
        input_shape=temperature.shape,
        input_elements=temperature.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(temperature, humidity, wind, height)

    try:
        # outside the physical range the formulas still return numbers, but
        # meaningless ones, so treat such values as missing
        invalid = (humidity < 0.0) | (humidity > 100.0) | (wind < 0.0)
        invalid_count = int(np.count_nonzero(invalid))
        if invalid_count > 0:
            _logger.warning(
                f"Found {invalid_count} values with relative humidity outside [0, 100] "
                "or negative wind speed; HDW is NaN in those columns."
            )

        in_layer = (height >= 0.0) & (height <= _HDW_LAYER_TOP_METERS)
        empty_columns = int(np.count_nonzero(~np.any(in_layer, axis=axis)))
        if empty_columns > 0:
            _logger.warning(
                f"Found {empty_columns} columns with no level in the lowest "
                f"{_HDW_LAYER_TOP_METERS:.0f} m AGL; HDW is NaN there."
            )

        saturation_hpa = pm_eto.saturation_vapor_pressure(temperature) / _KPA_PER_HPA
        vpd_hpa = saturation_hpa * (1.0 - humidity / 100.0)

        # NaN in-layer propagates through the maximum; out-of-layer levels are
        # excluded via -inf, which any real product beats
        value = np.where(invalid, np.nan, vpd_hpa * wind)
        product = np.where(in_layer, value, -np.inf)
        index = np.max(product, axis=axis)
        result = np.where(np.any(in_layer, axis=axis), index, np.nan).astype(np.float64, copy=False)

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

"""Fire-weather indices computed from standard meteorological inputs.

This module is the NumPy layer of the fire-weather family tracked in #793. It
currently provides the Fosberg Fire Weather Index, which is weather-only and
carries no state between time steps.

References
----------
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

import numpy as np
import numpy.typing as npt

from climate_indices.exceptions import InvalidArgumentError
from climate_indices.logging_config import get_logger
from climate_indices.performance import check_large_array_memory

# retrieve structlog logger for this module
_logger = get_logger(__name__)

# declare the function names that should be included in the public API for this module
__all__ = ["fosberg_ffwi"]

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


def _equilibrium_moisture_content(
    temperature_fahrenheit: npt.NDArray[np.float64],
    relative_humidity_percent: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Simard (1968) equilibrium moisture content, in percent.

    The three regressions were fitted independently and do not meet at the
    breakpoints. At 10% relative humidity the result jumps by up to about 0.7,
    with a size and sign that depend on temperature; at 50% it jumps by up to
    about 0.5. That is a property of the published equations, not of this
    implementation, and the index inherits a jump of roughly one unit.

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
            reproduce them.

    Returns:
        FFWI with the broadcast shape of the inputs. NaN where any input is
        NaN, where relative humidity lies outside [0, 100], or where wind speed
        is negative.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.

    Example:
        >>> from climate_indices import fire
        >>> round(float(fire.fosberg_ffwi(30.0, 15.0, 10.0)), 2)
        59.24
    """
    temperature = np.asarray(temperature_celsius, dtype=np.float64)
    humidity = np.asarray(relative_humidity_percent, dtype=np.float64)
    wind = np.asarray(wind_speed_meters_per_second, dtype=np.float64)

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

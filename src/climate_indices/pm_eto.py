"""Penman-Monteith reference evapotranspiration (FAO-56) helper functions.

This module implements the component equations from FAO Irrigation and Drainage
Paper 56 (Allen et al., 1998) needed to compute Penman-Monteith reference
evapotranspiration (ETo). Functions are organized by physical domain:

- **Atmospheric**: pressure, psychrometric constant, latent heat (Eq 7, 8, 2.1)
- **Vapor pressure**: saturation e_s, slope Delta, mean e_s (Eq 11, 12, 13)
- **Humidity pathways**: actual vapor pressure e_a from various inputs (Eq 14-19)
- **PM-ET core**: full Penman-Monteith equation (Eq 6)

All functions accept both scalar and numpy array inputs and return the same
type. Equation numbers refer to Allen et al. (1998), updated 2000.

References
----------
Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998)
    Crop evapotranspiration - Guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56. Rome, FAO.
    ISBN 92-5-104219-5
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from climate_indices import compute
from climate_indices.exceptions import InvalidArgumentError

try:
    # the optional Rust kernels (docs/architecture.md); without the extension the
    # Penman-Monteith entry points below run their pure-Python implementations
    from climate_indices import _native
except ImportError:
    _native = None  # type: ignore[assignment]


def _native_module() -> ModuleType | None:
    """The optional Rust extension, or None when this install is pure Python.

    Every dispatch guard reads the module through this accessor: a successful
    import is typed as always present, which would make the fallback branch look
    unreachable to the type checker.
    """
    return _native


# union type for function signatures
FloatOrArray = float | npt.NDArray[np.floating[Any]]

# native pathway selectors: these codes must match climate-py's mapping, which
# evaluates the same FAO-56 equation that each constant names
_HUMIDITY_DEWPOINT = 0
_HUMIDITY_RH_MIN_MAX = 1
_HUMIDITY_RH_MAX = 2
_HUMIDITY_RH_MEAN = 3
_HUMIDITY_TMIN = 4
_RADIATION_SUPPLIED = 0
_RADIATION_SUNSHINE = 1
_RADIATION_TEMPERATURE_RANGE = 2


@dataclass(frozen=True)
class HumidityInputs:
    """Optional FAO-56 actual-vapour-pressure inputs, in pathway precedence order.

    Only one pathway is used, chosen by precedence: dewpoint, then
    ``rh_min``/``rh_max``, then ``rh_max`` alone, then ``rh_mean``, and finally
    the arid-region ``e0(Tmin - 2)`` estimate when none is supplied.
    """

    tdew_celsius: Any = None
    rh_min: Any = None
    rh_max: Any = None
    rh_mean: Any = None


@dataclass(frozen=True)
class RadiationInputs:
    """Optional FAO-56 solar-radiation inputs, in pathway precedence order.

    Supplied solar radiation is used first, then sunshine hours, and finally the
    temperature-range estimate when neither is supplied.
    """

    solar_radiation_mj_m2_day: Any = None
    sunshine_hours: Any = None
    coastal: bool = False


# ---------------------------------------------------------------------------
# Physical constants (FAO-56, Chapter 2)
# ---------------------------------------------------------------------------

# specific heat of moist air at constant pressure [MJ kg-1 degC-1]
SPECIFIC_HEAT_MOIST_AIR = 1.013e-3

# ratio of molecular weight of water vapour to dry air (epsilon)
MOLECULAR_WEIGHT_RATIO = 0.622

# latent heat of vaporization at 20 degC [MJ kg-1] (FAO-56 simplification)
LATENT_HEAT_DEFAULT = 2.45

# standard atmospheric pressure at sea level [kPa]
ATMOSPHERIC_PRESSURE_SEA_LEVEL = 101.3

# temperature lapse rate [degC m-1] used in Eq 7
TEMPERATURE_LAPSE_RATE = 0.0065

# base temperature for the standard atmosphere [K]
BASE_TEMPERATURE_K = 293.0

# exponent in the atmospheric pressure equation (Eq 7)
PRESSURE_EXPONENT = 5.26

# solar constant [MJ m-2 min-1] (FAO-56 Eq 21)
SOLAR_CONSTANT = 0.0820

# Stefan-Boltzmann constant [MJ K-4 m-2 day-1] (FAO-56 Eq 39)
STEFAN_BOLTZMANN = 4.903e-9

# albedo of the hypothetical grass reference crop (FAO-56 Eq 38)
REFERENCE_ALBEDO = 0.23

# Kelvin offset used in the net longwave radiation equation (FAO-56 Eq 39)
KELVIN_OFFSET = 273.16


# ---------------------------------------------------------------------------
# Atmospheric helpers (FAO-56 Eq 7, 8, and 2.1)
# ---------------------------------------------------------------------------


def atmospheric_pressure(elevation: FloatOrArray) -> FloatOrArray:
    """Calculate atmospheric pressure from elevation.

    Implements FAO-56 Equation 7 (Allen et al., 1998):

        P = 101.3 * ((293 - 0.0065 * z) / 293) ^ 5.26

    where *z* is elevation above sea level in metres and *P* is atmospheric
    pressure in kPa.  This is a simplification of the ideal gas law for a
    standard atmosphere.

    Args:
        elevation: Elevation above sea level in metres. Accepts a scalar
            float or a numpy array.

    Returns:
        Atmospheric pressure in kPa, same type as input.

    """
    return (
        ATMOSPHERIC_PRESSURE_SEA_LEVEL
        * ((BASE_TEMPERATURE_K - TEMPERATURE_LAPSE_RATE * np.asarray(elevation)) / BASE_TEMPERATURE_K)
        ** PRESSURE_EXPONENT
    )


def latent_heat_of_vaporization(temperature_celsius: FloatOrArray) -> FloatOrArray:
    """Calculate latent heat of vaporization from air temperature.

    Implements the simplified relationship from FAO-56 Section 2 (Eq 2.1,
    Harrison 1963):

        lambda = 2.501 - 0.002361 * T

    where *T* is mean air temperature in degrees Celsius and *lambda* is
    latent heat of vaporization in MJ/kg.

    Note:
        FAO-56 recommends using a constant value of 2.45 MJ/kg (at 20 degC)
        for simplicity. This function provides the full temperature-dependent
        calculation when higher precision is desired.

    Args:
        temperature_celsius: Mean air temperature in degrees Celsius.
            Accepts a scalar float or a numpy array.

    Returns:
        Latent heat of vaporization in MJ/kg, same type as input.

    """
    return 2.501 - 0.002361 * np.asarray(temperature_celsius)


def psychrometric_constant(pressure_kpa: FloatOrArray) -> FloatOrArray:
    """Calculate the psychrometric constant from atmospheric pressure.

    Implements FAO-56 Equation 8 (Allen et al., 1998):

        gamma = (cp * P) / (epsilon * lambda)

    which simplifies to:

        gamma = 0.665e-3 * P

    where *P* is atmospheric pressure in kPa and *gamma* is the psychrometric
    constant in kPa/degC.  The simplification uses lambda = 2.45 MJ/kg
    (constant at 20 degC).

    Args:
        pressure_kpa: Atmospheric pressure in kPa. Accepts a scalar float or
            a numpy array.

    Returns:
        Psychrometric constant in kPa/degC, same type as input.

    """
    return 0.665e-3 * np.asarray(pressure_kpa)


# ---------------------------------------------------------------------------
# Vapor pressure helpers (FAO-56 Eq 11, 12, 13)
# ---------------------------------------------------------------------------


def saturation_vapor_pressure(temperature_celsius: FloatOrArray) -> FloatOrArray:
    """Calculate saturation vapor pressure at a given temperature.

    Implements FAO-56 Equation 11 (Allen et al., 1998):

        e0(T) = 0.6108 * exp(17.27 * T / (T + 237.3))

    where *T* is air temperature in degrees Celsius and *e0(T)* is the
    saturation vapor pressure in kPa at temperature *T*.

    Args:
        temperature_celsius: Air temperature in degrees Celsius. Accepts a
            scalar float or a numpy array.

    Returns:
        Saturation vapor pressure in kPa, same type as input.

    """
    t = np.asarray(temperature_celsius)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def vapor_pressure_slope(temperature_celsius: FloatOrArray) -> FloatOrArray:
    """Calculate slope of the saturation vapor pressure curve.

    Implements FAO-56 Equation 13 (Allen et al., 1998):

        Delta = 4098 * e0(T) / (T + 237.3)^2

    where *e0(T)* is the saturation vapor pressure at temperature *T* (Eq 11),
    and *Delta* is the slope in kPa/degC.

    Args:
        temperature_celsius: Air temperature in degrees Celsius. Accepts a
            scalar float or a numpy array.

    Returns:
        Slope of saturation vapor pressure curve in kPa/degC, same type
        as input.

    """
    t = np.asarray(temperature_celsius)
    e_sat = saturation_vapor_pressure(t)
    return 4098.0 * e_sat / (t + 237.3) ** 2


def mean_saturation_vapor_pressure(
    tmin_celsius: FloatOrArray,
    tmax_celsius: FloatOrArray,
) -> FloatOrArray:
    """Calculate mean saturation vapor pressure from daily min/max temperature.

    Implements FAO-56 Equation 12 (Allen et al., 1998):

        e_s = (e0(Tmin) + e0(Tmax)) / 2

    The mean is computed from the saturation vapor pressures at the daily
    minimum and maximum temperatures, rather than from the mean temperature,
    because the relationship between temperature and vapor pressure is
    non-linear.

    Args:
        tmin_celsius: Daily minimum air temperature in degrees Celsius.
        tmax_celsius: Daily maximum air temperature in degrees Celsius.

    Returns:
        Mean saturation vapor pressure in kPa, same type as inputs.

    """
    return (saturation_vapor_pressure(tmin_celsius) + saturation_vapor_pressure(tmax_celsius)) / 2.0


# ---------------------------------------------------------------------------
# Humidity pathway dispatcher (FAO-56 Eq 14-19)
# ---------------------------------------------------------------------------


def actual_vapor_pressure_from_dewpoint(
    tdew_celsius: FloatOrArray,
) -> FloatOrArray:
    """Calculate actual vapor pressure from dewpoint temperature.

    Implements FAO-56 Equation 14 (Allen et al., 1998):

        e_a = e0(Tdew) = 0.6108 * exp(17.27 * Tdew / (Tdew + 237.3))

    This is the most accurate method for determining actual vapor pressure
    when dewpoint temperature data are available.

    Args:
        tdew_celsius: Dewpoint temperature in degrees Celsius.

    Returns:
        Actual vapor pressure in kPa.

    """
    return saturation_vapor_pressure(tdew_celsius)


def actual_vapor_pressure_from_rhmin_rhmax(
    e_tmin: FloatOrArray,
    e_tmax: FloatOrArray,
    rh_min: FloatOrArray,
    rh_max: FloatOrArray,
) -> FloatOrArray:
    """Calculate actual vapor pressure from min/max relative humidity.

    Implements FAO-56 Equation 17 (Allen et al., 1998):

        e_a = (e0(Tmin) * RHmax/100 + e0(Tmax) * RHmin/100) / 2

    This is the preferred method when both RHmin and RHmax are available.

    Args:
        e_tmin: Saturation vapor pressure at daily minimum temperature [kPa].
        e_tmax: Saturation vapor pressure at daily maximum temperature [kPa].
        rh_min: Minimum daily relative humidity [%].
        rh_max: Maximum daily relative humidity [%].

    Returns:
        Actual vapor pressure in kPa.

    """
    return (np.asarray(e_tmin) * np.asarray(rh_max) / 100.0 + np.asarray(e_tmax) * np.asarray(rh_min) / 100.0) / 2.0  # type: ignore[no-any-return]


def actual_vapor_pressure_from_rhmax(
    e_tmin: FloatOrArray,
    rh_max: FloatOrArray,
) -> FloatOrArray:
    """Calculate actual vapor pressure from maximum relative humidity only.

    Implements FAO-56 Equation 18 (Allen et al., 1998):

        e_a = e0(Tmin) * RHmax / 100

    Used when only RHmax data is available.  The daily minimum temperature
    typically occurs at sunrise when the air is close to saturation, so
    RHmax near 100% is common and this approximation is reasonable.

    Args:
        e_tmin: Saturation vapor pressure at daily minimum temperature [kPa].
        rh_max: Maximum daily relative humidity [%].

    Returns:
        Actual vapor pressure in kPa.

    """
    return np.asarray(e_tmin) * np.asarray(rh_max) / 100.0  # type: ignore[no-any-return]


def actual_vapor_pressure_from_rhmean(
    e_s: FloatOrArray,
    rh_mean: FloatOrArray,
) -> FloatOrArray:
    """Calculate actual vapor pressure from mean relative humidity.

    Implements FAO-56 Equation 19 (Allen et al., 1998):

        e_a = e_s * RHmean / 100

    This is the least preferred method; use only when neither dewpoint
    temperature nor RHmin/RHmax data are available.

    Args:
        e_s: Mean saturation vapor pressure [kPa] (from Eq 12).
        rh_mean: Mean daily relative humidity [%].

    Returns:
        Actual vapor pressure in kPa.

    """
    return np.asarray(e_s) * np.asarray(rh_mean) / 100.0  # type: ignore[no-any-return]


def actual_vapor_pressure_from_tmin(
    tmin_celsius: FloatOrArray,
) -> FloatOrArray:
    """Estimate actual vapor pressure from minimum temperature.

    Implements FAO-56 approximation (Allen et al., 1998) for arid and
    semi-arid regions where humidity data is unavailable:

        e_a = e0(Tmin - 2)

    In arid regions, Tmin may overestimate Tdew by several degrees; the
    2 degC offset provides a conservative estimate.  In humid regions,
    Tmin approximates Tdew more closely.

    Note:
        FAO-56 recommends Tmin as a proxy for Tdew only when no humidity
        data is available. The 2 degC offset applies to arid conditions.
        For humid sites, ``actual_vapor_pressure_from_dewpoint(tmin)``
        (without offset) may be more appropriate.

    Args:
        tmin_celsius: Daily minimum air temperature in degrees Celsius.

    Returns:
        Estimated actual vapor pressure in kPa.

    """
    return saturation_vapor_pressure(np.asarray(tmin_celsius) - 2.0)


# ---------------------------------------------------------------------------
# Penman-Monteith reference evapotranspiration (FAO-56 Eq 6)
# ---------------------------------------------------------------------------


def _native_operand(value: Any) -> np.ndarray | None:
    """The operand as the kernels take it, or None when they cannot take it.

    A scalar narrower than float64 keeps its own dtype through the Python
    expressions, and an integer or boolean scalar keeps its own dtype where the
    subtraction can wrap or reject it, so only float64-precision scalars reach
    the kernels.

    :param value: one operand of a kernel call
    :return: the value as a float64 array, or None for a dtype, layout, or NumPy
        error policy the kernels do not take
    """
    if isinstance(value, np.ndarray):
        return value if compute._native_float64(value) else None
    if isinstance(value, float):
        return np.asarray(value, dtype=np.float64)
    return None


def _native_arrays(*values: Any) -> tuple[tuple[int, ...], tuple[np.ndarray, ...]] | None:
    """Flattened float64 kernel inputs and their broadcast shape, or None.

    The Rust kernels take plain, aligned float64 arrays with NumPy floating-point
    errors ignored, so scalars, every other dtype or layout, masked arrays, and
    non-default error or warning policies stay on the Python path. An all-scalar
    call stays there too, since its result is a NumPy scalar rather than an array.

    The kernel reads one element per broadcast position and copies each operand before
    it releases the GIL, so the route's peak is the caller's arrays, one flattened
    input per operand, a copy of each operand inside the call, and the kernel's fixed
    intermediates: a bounded multiple of the request. An operand that reaches every
    position as a single value is passed as a zero-stride view of it, which is the one
    expansion that would otherwise allocate in proportion to the request rather than
    to the operand.

    :param values: the operands of the operation, in kernel argument order
    :return: the broadcast shape and one contiguous 1-D array per operand, or
        None when the kernels cannot take these operands
    """
    arrays: list[np.ndarray] = []
    for value in values:
        array = _native_operand(value)
        if array is None:
            return None
        arrays.append(array)
    if all(array.ndim == 0 for array in arrays):
        return None

    try:
        broadcast = np.broadcast_arrays(*arrays)
    except ValueError:
        # let the Python expression raise the broadcasting error it always has
        return None
    elements = broadcast[0].size
    return (
        broadcast[0].shape,
        tuple(
            # a single value reaches every position: hand the kernel a view of it rather
            # than a full-size array, which it would copy element by element anyway
            np.broadcast_to(operand.reshape(1), (elements,))
            if operand.size == 1 and elements > 1
            else np.ascontiguousarray(expanded).reshape(-1)
            for operand, expanded in zip(arrays, broadcast, strict=True)
        ),
    )


def pm_eto(
    net_radiation: FloatOrArray,
    soil_heat_flux: FloatOrArray,
    temperature_celsius: FloatOrArray,
    wind_speed_2m: FloatOrArray,
    saturation_vp: FloatOrArray,
    actual_vp: FloatOrArray,
    delta: FloatOrArray,
    gamma: FloatOrArray,
) -> FloatOrArray:
    """Calculate Penman-Monteith reference evapotranspiration (ETo).

    Implements FAO-56 Equation 6 (Allen et al., 1998):

        ETo = (0.408 * Delta * (Rn - G) + gamma * (900/(T+273)) * u2 * (e_s - e_a))
              / (Delta + gamma * (1 + 0.34 * u2))

    This is the ASCE/FAO standardized reference crop evapotranspiration for
    a hypothetical grass reference crop with assumed height of 0.12 m, surface
    resistance of 70 s/m, and albedo of 0.23.

    All inputs must be broadcast-compatible (scalar or arrays of the same
    shape).

    Args:
        net_radiation: Net radiation at the crop surface [MJ m-2 day-1].
        soil_heat_flux: Soil heat flux density [MJ m-2 day-1]. For daily
            calculations, G is often assumed to be zero.
        temperature_celsius: Mean daily air temperature at 2 m height [degC].
        wind_speed_2m: Wind speed at 2 m height [m s-1].
        saturation_vp: Saturation vapor pressure [kPa] (e_s, from Eq 12).
        actual_vp: Actual vapor pressure [kPa] (e_a, from Eq 14-19).
        delta: Slope of saturation vapor pressure curve [kPa degC-1] (Eq 13).
        gamma: Psychrometric constant [kPa degC-1] (Eq 8).

    Returns:
        Reference evapotranspiration ETo in mm/day, same shape as inputs.

    """
    extension = _native_module()
    native = (
        _native_arrays(
            net_radiation,
            soil_heat_flux,
            temperature_celsius,
            wind_speed_2m,
            saturation_vp,
            actual_vp,
            delta,
            gamma,
        )
        if extension is not None
        else None
    )
    if extension is not None and native is not None and hasattr(extension, "pm_eto"):
        shape, arrays = native
        return np.asarray(extension.pm_eto(*arrays)).reshape(shape)

    rn = np.asarray(net_radiation)
    g = np.asarray(soil_heat_flux)
    t = np.asarray(temperature_celsius)
    u2 = np.asarray(wind_speed_2m)
    e_s = np.asarray(saturation_vp)
    e_a = np.asarray(actual_vp)
    d = np.asarray(delta)
    gam = np.asarray(gamma)

    # numerator: radiation term + aerodynamic term
    numerator = 0.408 * d * (rn - g) + gam * (900.0 / (t + 273.0)) * u2 * (e_s - e_a)

    # denominator
    denominator = d + gam * (1.0 + 0.34 * u2)

    return numerator / denominator  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Radiation and wind helpers (FAO-56 Chapter 3, Eq 21-50)
# ---------------------------------------------------------------------------


def _sunset_hour_angle(
    latitude_radians: FloatOrArray,
    solar_declination_radians: FloatOrArray,
) -> FloatOrArray:
    """Calculate the sunset hour angle (Eq 25), clipped to the arccos domain."""
    cosine = -np.tan(np.asarray(latitude_radians)) * np.tan(np.asarray(solar_declination_radians))
    return np.arccos(np.clip(cosine, -1.0, 1.0))  # type: ignore[no-any-return]


def _inverse_relative_distance(day_of_year: FloatOrArray) -> FloatOrArray:
    """Calculate the inverse relative Earth-Sun distance (Eq 23)."""
    return 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * np.asarray(day_of_year))


def _solar_declination(day_of_year: FloatOrArray) -> FloatOrArray:
    """Calculate the solar declination (Eq 24)."""
    return 0.409 * np.sin((2.0 * np.pi / 365.0) * np.asarray(day_of_year) - 1.39)


def extraterrestrial_radiation(
    latitude_radians: FloatOrArray,
    day_of_year: FloatOrArray,
) -> FloatOrArray:
    """Calculate extraterrestrial radiation.

    Implements FAO-56 Equation 21 (Allen et al., 1998):

        Ra = (24 * 60 / pi) * Gsc * dr * [ws * sin(phi) * sin(delta)
             + cos(phi) * cos(delta) * sin(ws)]

    with the inverse relative Earth-Sun distance *dr* from Equation 23, the
    solar declination *delta* from Equation 24, and the sunset hour angle *ws*
    from Equation 25.

    Args:
        latitude_radians: Latitude in radians (positive north).
        day_of_year: Day of the year, 1-365 (366 in a leap year).

    Returns:
        Extraterrestrial radiation in MJ m-2 day-1.

    """
    latitude = np.asarray(latitude_radians)
    day = np.asarray(day_of_year)
    declination = _solar_declination(day)
    sunset_hour_angle = _sunset_hour_angle(latitude, declination)
    return (  # type: ignore[no-any-return]
        (24.0 * 60.0 / np.pi)
        * SOLAR_CONSTANT
        * _inverse_relative_distance(day)
        * (
            sunset_hour_angle * np.sin(latitude) * np.sin(declination)
            + np.cos(latitude) * np.cos(declination) * np.sin(sunset_hour_angle)
        )
    )


def daylight_hours(
    latitude_radians: FloatOrArray,
    day_of_year: FloatOrArray,
) -> FloatOrArray:
    """Calculate the maximum possible daylight hours (FAO-56 Eq 34).

    Args:
        latitude_radians: Latitude in radians (positive north).
        day_of_year: Day of the year, 1-365 (366 in a leap year).

    Returns:
        Daylight hours in hours day-1.

    """
    latitude = np.asarray(latitude_radians)
    day = np.asarray(day_of_year)
    return (24.0 / np.pi) * _sunset_hour_angle(latitude, _solar_declination(day))


def clear_sky_solar_radiation(
    extraterrestrial_radiation_mj_m2_day: FloatOrArray,
    elevation_m: FloatOrArray,
) -> FloatOrArray:
    """Calculate clear-sky solar radiation.

    Implements FAO-56 Equation 37 (Allen et al., 1998):

        Rso = (0.75 + 2e-5 * z) * Ra

    Args:
        extraterrestrial_radiation_mj_m2_day: Extraterrestrial radiation
            [MJ m-2 day-1] (Eq 21).
        elevation_m: Station elevation above sea level [m].

    Returns:
        Clear-sky solar radiation in MJ m-2 day-1.

    """
    return (0.75 + 2.0e-5 * np.asarray(elevation_m)) * np.asarray(  # type: ignore[no-any-return]
        extraterrestrial_radiation_mj_m2_day
    )


def net_shortwave_radiation(
    solar_radiation_mj_m2_day: FloatOrArray,
    albedo: FloatOrArray = REFERENCE_ALBEDO,
) -> FloatOrArray:
    """Calculate net shortwave radiation.

    Implements FAO-56 Equation 38 (Allen et al., 1998):

        Rns = (1 - alpha) * Rs

    Args:
        solar_radiation_mj_m2_day: Incoming solar radiation [MJ m-2 day-1].
        albedo: Canopy reflection coefficient (0.23 for the grass reference).

    Returns:
        Net shortwave radiation in MJ m-2 day-1.

    """
    return (1.0 - np.asarray(albedo)) * np.asarray(solar_radiation_mj_m2_day)  # type: ignore[no-any-return]


def net_longwave_radiation(
    tmin_celsius: FloatOrArray,
    tmax_celsius: FloatOrArray,
    actual_vp_kpa: FloatOrArray,
    solar_radiation_mj_m2_day: FloatOrArray,
    clear_sky_solar_radiation_mj_m2_day: FloatOrArray,
) -> FloatOrArray:
    """Calculate net outgoing longwave radiation.

    Implements FAO-56 Equation 39 (Allen et al., 1998):

        Rnl = sigma * [(Tmax,K^4 + Tmin,K^4) / 2]
              * (0.34 - 0.14 * sqrt(ea)) * (1.35 * Rs / Rso - 0.35)

    The relative shortwave radiation *Rs / Rso* is limited to 1.0, as required
    by FAO-56.

    Args:
        tmin_celsius: Daily minimum air temperature [degC].
        tmax_celsius: Daily maximum air temperature [degC].
        actual_vp_kpa: Actual vapour pressure [kPa] (Eq 14-19).
        solar_radiation_mj_m2_day: Solar radiation [MJ m-2 day-1] (Eq 35-37).
        clear_sky_solar_radiation_mj_m2_day: Clear-sky solar radiation
            [MJ m-2 day-1] (Eq 36-37).

    Returns:
        Net outgoing longwave radiation in MJ m-2 day-1.

    """
    tmax_kelvin = np.asarray(tmax_celsius) + KELVIN_OFFSET
    tmin_kelvin = np.asarray(tmin_celsius) + KELVIN_OFFSET
    relative_solar = np.minimum(
        np.asarray(solar_radiation_mj_m2_day) / np.asarray(clear_sky_solar_radiation_mj_m2_day),
        1.0,
    )
    return (  # type: ignore[no-any-return]
        STEFAN_BOLTZMANN
        * ((tmax_kelvin**4 + tmin_kelvin**4) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.asarray(actual_vp_kpa)))
        * (1.35 * relative_solar - 0.35)
    )


def net_radiation(
    tmin_celsius: FloatOrArray,
    tmax_celsius: FloatOrArray,
    actual_vp_kpa: FloatOrArray,
    solar_radiation_mj_m2_day: FloatOrArray,
    clear_sky_solar_radiation_mj_m2_day: FloatOrArray,
    albedo: FloatOrArray = REFERENCE_ALBEDO,
) -> FloatOrArray:
    """Calculate net radiation from the shortwave and longwave components.

    Implements FAO-56 Equation 40 (Allen et al., 1998):

        Rn = Rns - Rnl

    Args:
        tmin_celsius: Daily minimum air temperature [degC].
        tmax_celsius: Daily maximum air temperature [degC].
        actual_vp_kpa: Actual vapour pressure [kPa] (Eq 14-19).
        solar_radiation_mj_m2_day: Solar radiation [MJ m-2 day-1] (Eq 35-37).
        clear_sky_solar_radiation_mj_m2_day: Clear-sky solar radiation
            [MJ m-2 day-1] (Eq 36-37).
        albedo: Canopy reflection coefficient (0.23 for the grass reference).

    Returns:
        Net radiation in MJ m-2 day-1.

    """
    return net_shortwave_radiation(solar_radiation_mj_m2_day, albedo) - net_longwave_radiation(
        tmin_celsius,
        tmax_celsius,
        actual_vp_kpa,
        solar_radiation_mj_m2_day,
        clear_sky_solar_radiation_mj_m2_day,
    )


def solar_radiation_from_sunshine(
    sunshine_hours: FloatOrArray,
    daylight_hours_value: FloatOrArray,
    extraterrestrial_radiation_mj_m2_day: FloatOrArray,
) -> FloatOrArray:
    """Estimate solar radiation from measured sunshine duration.

    Implements FAO-56 Equation 35 (Allen et al., 1998) with the recommended
    Angstrom coefficients ``as = 0.25`` and ``bs = 0.50``:

        Rs = [as + bs * (n / N)] * Ra

    Args:
        sunshine_hours: Actual duration of bright sunshine [hours day-1].
        daylight_hours_value: Maximum possible daylight hours *N* [hours day-1]
            (Eq 34).
        extraterrestrial_radiation_mj_m2_day: Extraterrestrial radiation
            [MJ m-2 day-1] (Eq 21).

    Returns:
        Solar radiation in MJ m-2 day-1.

    """
    return (0.25 + 0.50 * np.asarray(sunshine_hours) / np.asarray(daylight_hours_value)) * np.asarray(  # type: ignore[no-any-return]
        extraterrestrial_radiation_mj_m2_day
    )


def solar_radiation_from_temperature_range(
    tmin_celsius: FloatOrArray,
    tmax_celsius: FloatOrArray,
    extraterrestrial_radiation_mj_m2_day: FloatOrArray,
    coastal: bool = False,
) -> FloatOrArray:
    """Estimate solar radiation from the daily temperature range.

    Implements FAO-56 Equation 50 (Allen et al., 1998):

        Rs = kRs * sqrt(Tmax - Tmin) * Ra

    where ``kRs = 0.16`` for interior locations and ``0.19`` for coastal
    locations. The result should be limited to the clear-sky radiation before
    use in Equation 39.

    Args:
        tmin_celsius: Daily minimum air temperature [degC].
        tmax_celsius: Daily maximum air temperature [degC].
        extraterrestrial_radiation_mj_m2_day: Extraterrestrial radiation
            [MJ m-2 day-1] (Eq 21).
        coastal: Whether the location is coastal (``kRs = 0.19``) rather than
            interior (``kRs = 0.16``).

    Returns:
        Solar radiation in MJ m-2 day-1.

    """
    krs = 0.19 if coastal else 0.16
    temperature_range = np.maximum(np.asarray(tmax_celsius) - np.asarray(tmin_celsius), 0.0)
    return (  # type: ignore[no-any-return]
        krs * np.sqrt(temperature_range) * np.asarray(extraterrestrial_radiation_mj_m2_day)
    )


def _validate_wind_measurement_height(measurement_height_m: FloatOrArray) -> np.ndarray:
    """Reject a non-positive wind measurement height, as FAO-56 Eq 47 requires.

    The native dispatch guard calls this too, so an invalid height raises the
    same error whichever path computes the result.

    :param measurement_height_m: height above the ground at which the wind
        speed was measured [m]
    :return: the height as an array
    :raise InvalidArgumentError: if any height is not positive
    """
    height = np.asarray(measurement_height_m)
    if np.any(height <= 0.0):
        raise InvalidArgumentError(
            f"Wind measurement height must be positive. Received: {measurement_height_m!r}",
            argument_name="measurement_height_m",
            argument_value=str(measurement_height_m),
            valid_values="> 0 m",
        )
    return height


def wind_speed_2m(
    wind_speed: FloatOrArray,
    measurement_height_m: FloatOrArray = 2.0,
) -> FloatOrArray:
    """Convert wind speed measured at any height to the 2 m standard height.

    Implements FAO-56 Equation 47 (Allen et al., 1998):

        u2 = uz * 4.87 / ln(67.8 * z - 5.42)

    where *z* is the measurement height in metres. At ``z = 2`` the conversion
    factor is essentially 1.

    Args:
        wind_speed: Wind speed measured at ``measurement_height_m`` [m s-1].
        measurement_height_m: Height above the ground at which the wind speed
            was measured [m].

    Returns:
        Wind speed at 2 m above the ground in m s-1.

    """
    height = _validate_wind_measurement_height(measurement_height_m)
    conversion = 4.87 / np.log(67.8 * height - 5.42)
    # at the standard height Eq 47 reduces to unity; return the input exactly
    return np.where(height == 2.0, np.asarray(wind_speed), np.asarray(wind_speed) * conversion)  # NOSONAR


# ---------------------------------------------------------------------------
# High-level Penman-Monteith ETo from meteorological inputs
# ---------------------------------------------------------------------------


def _humidity_pathway(humidity: HumidityInputs | None) -> tuple[int, tuple[FloatOrArray, ...]]:
    """Resolve the FAO-56 actual-vapour-pressure pathway and the inputs it needs.

    The precedence is dewpoint, then ``rh_min``/``rh_max``, then ``rh_max`` alone,
    then ``rh_mean``, and finally the arid-region ``e0(Tmin - 2)`` estimate. The
    native dispatch and the Python selection both resolve the pathway here, so
    the two paths cannot disagree about which one applies.

    :param humidity: the optional actual-vapour-pressure inputs, in precedence order
    :return: the pathway's selector code and its input values, in the order the
        kernel takes them
    :raise InvalidArgumentError: if ``rh_min`` is given without ``rh_max``
    """
    humidity = humidity or HumidityInputs()
    if humidity.tdew_celsius is not None:
        return _HUMIDITY_DEWPOINT, (humidity.tdew_celsius,)
    if humidity.rh_min is not None and humidity.rh_max is not None:
        return _HUMIDITY_RH_MIN_MAX, (humidity.rh_min, humidity.rh_max)
    if humidity.rh_min is not None:
        raise InvalidArgumentError(
            "rh_min was provided without rh_max; both are required for Eq 17.",
            argument_name="rh_min",
            argument_value=str(humidity.rh_min),
            valid_values="provide both rh_min and rh_max, or neither",
        )
    if humidity.rh_max is not None:
        return _HUMIDITY_RH_MAX, (humidity.rh_max,)
    if humidity.rh_mean is not None:
        return _HUMIDITY_RH_MEAN, (humidity.rh_mean,)
    return _HUMIDITY_TMIN, ()


def _select_actual_vapor_pressure(
    tmin_celsius: FloatOrArray,
    tmax_celsius: FloatOrArray,
    e_s: FloatOrArray,
    humidity: HumidityInputs | None,
) -> FloatOrArray:
    """Select the best available FAO-56 actual-vapour-pressure pathway."""
    pathway, values = _humidity_pathway(humidity)
    if pathway == _HUMIDITY_DEWPOINT:
        return actual_vapor_pressure_from_dewpoint(values[0])
    if pathway == _HUMIDITY_RH_MIN_MAX:
        return actual_vapor_pressure_from_rhmin_rhmax(
            saturation_vapor_pressure(tmin_celsius),
            saturation_vapor_pressure(tmax_celsius),
            values[0],
            values[1],
        )
    if pathway == _HUMIDITY_RH_MAX:
        return actual_vapor_pressure_from_rhmax(saturation_vapor_pressure(tmin_celsius), values[0])
    if pathway == _HUMIDITY_RH_MEAN:
        return actual_vapor_pressure_from_rhmean(e_s, values[0])
    return actual_vapor_pressure_from_tmin(tmin_celsius)


def _radiation_pathway(radiation: RadiationInputs | None) -> tuple[int, tuple[FloatOrArray, ...], bool]:
    """Resolve the FAO-56 solar-radiation pathway and the inputs it needs.

    The precedence is supplied solar radiation, then sunshine hours, and finally
    the temperature-range estimate, which uses the ``coastal`` coefficient and is
    limited to the clear-sky radiation by the caller.

    :param radiation: the optional solar-radiation inputs, in precedence order
    :return: the pathway's selector code, its input values, and whether the
        location is coastal
    """
    radiation = radiation or RadiationInputs()
    if radiation.solar_radiation_mj_m2_day is not None:
        return _RADIATION_SUPPLIED, (radiation.solar_radiation_mj_m2_day,), radiation.coastal
    if radiation.sunshine_hours is not None:
        return _RADIATION_SUNSHINE, (radiation.sunshine_hours,), radiation.coastal
    return _RADIATION_TEMPERATURE_RANGE, (), radiation.coastal


def _select_solar_radiation(
    tmin_celsius: FloatOrArray,
    tmax_celsius: FloatOrArray,
    extraterrestrial_radiation_mj_m2_day: FloatOrArray,
    daylength_hours: FloatOrArray,
    radiation: RadiationInputs | None,
) -> tuple[FloatOrArray, bool]:
    """Select supplied, sunshine-based, or temperature-based solar radiation.

    Returns:
        The solar radiation and whether it was estimated from the temperature
        range (and so must be limited to the clear-sky radiation).

    """
    pathway, values, coastal = _radiation_pathway(radiation)
    if pathway == _RADIATION_SUPPLIED:
        return values[0], False
    if pathway == _RADIATION_SUNSHINE:
        return (
            solar_radiation_from_sunshine(
                values[0],
                daylength_hours,
                extraterrestrial_radiation_mj_m2_day,
            ),
            False,
        )
    return (
        solar_radiation_from_temperature_range(
            tmin_celsius,
            tmax_celsius,
            extraterrestrial_radiation_mj_m2_day,
            coastal,
        ),
        True,
    )


def _native_penman_monteith_eto(
    daily_tmin_celsius: Any,
    daily_tmax_celsius: Any,
    latitude_degrees: Any,
    elevation_m: Any,
    wind_speed_m_s: Any,
    day_of_year: Any,
    wind_speed_height_m: Any,
    humidity: HumidityInputs | None,
    radiation: RadiationInputs | None,
    soil_heat_flux_mj_m2_day: Any,
    albedo: Any,
) -> npt.NDArray[np.float64] | None:
    """Derive the FAO-56 intermediates and ETo with the Rust kernel, or None.

    The kernel evaluates the same intermediate chain the Python body below does,
    for the one humidity and one radiation pathway that precedence selects. The
    wind-height check and the pathway precedence happen here, before the kernel
    is reached, so a bad call raises the same error, in the same order, whichever
    path computes the result. The public helper functions stay Python callables:
    only these two entry points dispatch.

    :return: the ETo array in the broadcast shape of the inputs, or None when the
        kernel cannot take these inputs
    """
    extension = _native_module()
    if extension is None:
        return None

    # an operand the kernels cannot take keeps the Python path, so its own
    # conversions and checks run in their original order rather than after the
    # native-only validation below
    if any(
        _native_operand(value) is None
        for value in (
            daily_tmin_celsius,
            daily_tmax_celsius,
            latitude_degrees,
            elevation_m,
            wind_speed_m_s,
            wind_speed_height_m,
            day_of_year,
            soil_heat_flux_mj_m2_day,
            albedo,
        )
    ):
        return None

    # Eq 47 validates the measurement height before any pathway is chosen
    _validate_wind_measurement_height(wind_speed_height_m)
    humidity_variant, humidity_values = _humidity_pathway(humidity)
    radiation_variant, radiation_values, coastal = _radiation_pathway(radiation)

    # the nine meteorological inputs come first, then the selected pathway's own
    # inputs, in the order the kernel takes them
    prepared = _native_arrays(
        daily_tmin_celsius,
        daily_tmax_celsius,
        latitude_degrees,
        elevation_m,
        wind_speed_m_s,
        wind_speed_height_m,
        day_of_year,
        soil_heat_flux_mj_m2_day,
        albedo,
        *humidity_values,
        *radiation_values,
    )
    if prepared is None:
        return None
    shape, arrays = prepared

    offset = 9
    tdew_celsius: np.ndarray | None = None
    rh_min: np.ndarray | None = None
    rh_max: np.ndarray | None = None
    rh_mean: np.ndarray | None = None
    if humidity_variant == _HUMIDITY_DEWPOINT:
        (tdew_celsius,) = arrays[offset : offset + 1]
    elif humidity_variant == _HUMIDITY_RH_MIN_MAX:
        rh_min, rh_max = arrays[offset : offset + 2]
    elif humidity_variant == _HUMIDITY_RH_MAX:
        (rh_max,) = arrays[offset : offset + 1]
    elif humidity_variant == _HUMIDITY_RH_MEAN:
        (rh_mean,) = arrays[offset : offset + 1]
    offset += len(humidity_values)

    solar_radiation: np.ndarray | None = None
    sunshine_hours: np.ndarray | None = None
    if radiation_variant == _RADIATION_SUPPLIED:
        (solar_radiation,) = arrays[offset : offset + 1]
    elif radiation_variant == _RADIATION_SUNSHINE:
        (sunshine_hours,) = arrays[offset : offset + 1]

    (daily_tmin, daily_tmax, latitude, elevation, wind_speed, wind_height, day, soil_heat_flux, albedo) = arrays[0:9]
    # an extension built before the PET kernels keeps the Python path
    if not hasattr(extension, "fao56_eto"):
        return None
    eto = extension.fao56_eto(
        daily_tmin,
        daily_tmax,
        latitude,
        elevation,
        wind_speed,
        wind_height,
        day,
        soil_heat_flux,
        albedo,
        humidity_variant,
        tdew_celsius,
        rh_min,
        rh_max,
        rh_mean,
        radiation_variant,
        solar_radiation,
        sunshine_hours,
        coastal,
    )
    return np.asarray(eto).reshape(shape)


def penman_monteith_eto(
    daily_tmin_celsius: Any,
    daily_tmax_celsius: Any,
    latitude_degrees: Any,
    elevation_m: Any,
    wind_speed_m_s: Any,
    day_of_year: Any,
    wind_speed_height_m: Any = 2.0,
    humidity: HumidityInputs | None = None,
    radiation: RadiationInputs | None = None,
    soil_heat_flux_mj_m2_day: Any = 0.0,
    albedo: Any = REFERENCE_ALBEDO,
) -> FloatOrArray:
    """Compute FAO-56 Penman-Monteith reference evapotranspiration from meteorology.

    This is a convenience wrapper around :func:`pm_eto` that derives the FAO-56
    intermediate variables from the supplied meteorological inputs:

    - atmospheric pressure and the psychrometric constant from elevation
      (Eq 7-8);
    - saturation vapour pressure, actual vapour pressure, and the slope of the
      saturation vapour pressure curve from temperature and the best available
      humidity pathway (Eq 11-19);
    - wind speed at the 2 m standard height (Eq 47);
    - net radiation from supplied, sunshine-based, or temperature-range solar
      radiation and clear-sky radiation (Eq 21-40).

    Humidity pathway precedence is dewpoint, then RHmin/RHmax, then RHmax, then
    RHmean, and finally the arid-region ``e0(Tmin - 2)`` estimate. Radiation
    precedence is supplied solar radiation, then sunshine hours, then the
    temperature-range estimate. For daily steps the soil heat flux defaults to
    zero.

    Args:
        daily_tmin_celsius: Daily minimum air temperature [degC].
        daily_tmax_celsius: Daily maximum air temperature [degC].
        latitude_degrees: Latitude in degrees north (range -90 to 90).
        elevation_m: Station elevation above sea level [m].
        wind_speed_m_s: Wind speed measured at ``wind_speed_height_m`` [m s-1].
        day_of_year: Day of the year, 1-365 (366 in a leap year).
        wind_speed_height_m: Height at which the wind speed was measured [m].
        humidity: Optional actual-vapour-pressure inputs, in pathway precedence
            order; see :class:`HumidityInputs`.
        radiation: Optional solar-radiation inputs, in pathway precedence order;
            see :class:`RadiationInputs`.
        soil_heat_flux_mj_m2_day: Soil heat flux density [MJ m-2 day-1]. Use 0
            for daily steps.
        albedo: Canopy reflection coefficient (0.23 for the grass reference).

    Returns:
        Reference evapotranspiration ETo in mm/day, same shape as the inputs.

    """
    tmin = np.asarray(daily_tmin_celsius)
    tmax = np.asarray(daily_tmax_celsius)
    tmean = (tmin + tmax) / 2.0

    # the Rust kernel derives the same intermediates for the selected pathways and
    # evaluates the same equation in one pass; the Python body below stays the
    # reference implementation and the fallback
    native_eto = _native_penman_monteith_eto(
        daily_tmin_celsius,
        daily_tmax_celsius,
        latitude_degrees,
        elevation_m,
        wind_speed_m_s,
        day_of_year,
        wind_speed_height_m,
        humidity,
        radiation,
        soil_heat_flux_mj_m2_day,
        albedo,
    )
    if native_eto is not None:
        return native_eto

    latitude_radians = np.radians(np.asarray(latitude_degrees))
    day = np.asarray(day_of_year)

    wind_2m = wind_speed_2m(wind_speed_m_s, wind_speed_height_m)
    gamma = psychrometric_constant(atmospheric_pressure(elevation_m))
    e_s = mean_saturation_vapor_pressure(tmin, tmax)
    delta = vapor_pressure_slope(tmean)
    e_a = _select_actual_vapor_pressure(tmin, tmax, e_s, humidity)

    ra = extraterrestrial_radiation(latitude_radians, day)
    rso = clear_sky_solar_radiation(ra, elevation_m)
    rs, from_temperature = _select_solar_radiation(
        tmin,
        tmax,
        ra,
        daylight_hours(latitude_radians, day),
        radiation,
    )
    if from_temperature:
        # FAO-56 Eq 50: limit the temperature-range estimate to the clear-sky value
        rs = np.minimum(rs, rso)

    rn = net_radiation(tmin, tmax, e_a, rs, rso, albedo)
    return pm_eto(rn, soil_heat_flux_mj_m2_day, tmean, wind_2m, e_s, e_a, delta, gamma)

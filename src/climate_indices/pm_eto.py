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

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

# union type for function signatures
FloatOrArray = Union[float, "npt.NDArray[np.floating]"]

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

    Examples:
        >>> atmospheric_pressure(0.0)
        101.3...
        >>> atmospheric_pressure(1800.0)
        81.8...
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

    Examples:
        >>> latent_heat_of_vaporization(20.0)
        2.4528...
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

    Examples:
        >>> psychrometric_constant(101.3)
        0.0673...
        >>> psychrometric_constant(81.8)
        0.0544...
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

    Examples:
        >>> saturation_vapor_pressure(20.0)
        2.338...
        >>> saturation_vapor_pressure(25.0)
        3.167...
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

    Examples:
        >>> vapor_pressure_slope(20.0)
        0.1447...
        >>> vapor_pressure_slope(25.0)
        0.1888...
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

    Examples:
        >>> mean_saturation_vapor_pressure(15.0, 25.0)
        2.291...
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

    Examples:
        >>> actual_vapor_pressure_from_dewpoint(17.0)
        1.937...
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

    Examples:
        >>> actual_vapor_pressure_from_rhmin_rhmax(1.705, 3.168, 54.0, 82.0)
        1.557...
    """
    return (np.asarray(e_tmin) * np.asarray(rh_max) / 100.0 + np.asarray(e_tmax) * np.asarray(rh_min) / 100.0) / 2.0


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

    Examples:
        >>> actual_vapor_pressure_from_rhmax(1.705, 82.0)
        1.398...
    """
    return np.asarray(e_tmin) * np.asarray(rh_max) / 100.0


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

    Examples:
        >>> actual_vapor_pressure_from_rhmean(2.437, 68.0)
        1.657...
    """
    return np.asarray(e_s) * np.asarray(rh_mean) / 100.0


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

    Examples:
        >>> actual_vapor_pressure_from_tmin(18.0)
        1.817...
    """
    return saturation_vapor_pressure(np.asarray(tmin_celsius) - 2.0)


# ---------------------------------------------------------------------------
# Penman-Monteith reference evapotranspiration (FAO-56 Eq 6)
# ---------------------------------------------------------------------------


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

    Examples:
        >>> pm_eto(
        ...     net_radiation=13.28,
        ...     soil_heat_flux=0.14,
        ...     temperature_celsius=16.9,
        ...     wind_speed_2m=2.078,
        ...     saturation_vp=1.997,
        ...     actual_vp=1.409,
        ...     delta=0.122,
        ...     gamma=0.0666,
        ... )
        3.88...
    """
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

    return numerator / denominator

"""
Module for calculation of potential evapotranspiration.

--------------------------------------------------------------------------------
Credits:

Derived from original code found here: https://github.com/woodcrafty/PyETo

--------------------------------------------------------------------------------
References:

Thornthwaite, C.W. (1948) An approach toward a rational classification
of climate. Geographical Review, Vol. 38, 55-94.
https://www.jstor.org/stable/210739

Allen, Richard et al (1998) Crop evapotranspiration - Guidelines for computing
crop water requirements - FAO Irrigation and drainage paper 56
ISBN 92-5-104219-5

Goswami, D. Yogi (2015) Principles of Solar Engineering, Third Edition
ISBN 97-8-146656-3780
"""

import calendar
import logging
import math

import numba
import numpy as np

from climate_indices import utils

# ------------------------------------------------------------------------------
# Retrieve logger and set desired logging level
_logger = utils.get_logger(__name__, logging.DEBUG)

# ------------------------------------------------------------------------------

# days of each calendar month, for non-leap and leap years
_MONTH_DAYS_NONLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_DAYS_LEAP = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# solar constant [ MJ m-2 min-1]
_SOLAR_CONSTANT = 0.0820

# ------------------------------------------------------------------------------
# angle values used within the _sunset_hour_angle() function defined below

# valid range for latitude, in radians
_LATITUDE_RADIANS_MIN = np.deg2rad(-90.0)
_LATITUDE_RADIANS_MAX = np.deg2rad(90.0)

# valid range for solar declination angle, in radians
# Goswami (2015), p.40
_SOLAR_DECLINATION_RADIANS_MIN = np.deg2rad(-23.45)
_SOLAR_DECLINATION_RADIANS_MAX = np.deg2rad(23.45)


# ------------------------------------------------------------------------------
@numba.jit
def _sunset_hour_angle(
        latitude_radians: float,
        solar_declination_radians: float,
) -> float:
    """
    Calculate sunset hour angle (*Ws*) from latitude and solar declination.

    Based on FAO equation 25 in Allen et al (1998).

    :param float latitude_radians: latitude in radians
    :param float solar_declination_radians: angle of solar declination in radians
    :return: sunset hour angle in radians
    :rtype: float
    """

    # validate the latitude argument
    if not _LATITUDE_RADIANS_MIN <= latitude_radians <= _LATITUDE_RADIANS_MAX:
        raise ValueError(
            "latitude outside valid range [{0!r} to {1!r}]: {2!r}".format(
                _LATITUDE_RADIANS_MIN, _LATITUDE_RADIANS_MAX, latitude_radians
            )
        )

    # validate the solar declination angle argument, which can vary between
    # -23.45 and +23.45 degrees see Goswami (2015) p.40, and
    # http://www.itacanet.org/the-sun-as-a-source-of-energy/part-1-solar-astronomy/
    if (not _SOLAR_DECLINATION_RADIANS_MIN
            <= solar_declination_radians <= _SOLAR_DECLINATION_RADIANS_MAX):
        raise ValueError("solar declination angle outside the valid range [" +
                         str(_SOLAR_DECLINATION_RADIANS_MIN) + " to " +
                         str(_SOLAR_DECLINATION_RADIANS_MAX) + "]: " +
                         str(solar_declination_radians) + " (actual value)")

    # calculate the cosine of the sunset hour angle (*Ws* in FAO 25)
    # from latitude and solar declination
    cos_sunset_hour_angle = \
        -math.tan(latitude_radians) * math.tan(solar_declination_radians)

    # If the sunset hour angle is >= 1 there is no sunset, i.e. 24 hours of daylight
    # If the sunset hour angle is <= 1 there is no sunrise, i.e. 24 hours of darkness
    # See http://www.itacanet.org/the-sun-as-a-source-of-energy/part-3-calculating-solar-angles/
    # Domain of acos is -1 <= x <= 1 radians (this is not mentioned in FAO-56!)
    return math.acos(min(max(cos_sunset_hour_angle, -1.0), 1.0))


# ------------------------------------------------------------------------------
@numba.jit
def _solar_declination(
        day_of_year: int,
) -> float:
    """
    Calculate the angle of solar declination from day of the year.

    Based on FAO equation 24 in Allen et al (1998).

    :param day_of_year: day of year integer between 1 and 365 (or 366,
        in the case of a leap year)
    :return: solar declination [radians]
    :rtype: float
    :raise ValueError: if the day of year value is not within the range [1-366]
    """
    if not 1 <= day_of_year <= 366:
        raise ValueError("Day of the year must be in the range [1-366]: "
                         "{0!r}".format(day_of_year))

    return 0.409 * math.sin(((2.0 * math.pi / 365.0) * day_of_year - 1.39))


# ------------------------------------------------------------------------------
@numba.jit
def _daylight_hours(
        sunset_hour_angle_radians: float,
) -> float:
    """
    Calculate daylight hours from a sunset hour angle.

    Based on FAO equation 34 in Allen et al (1998).

    :param sunset_hour_angle_radians: sunset hour angle, in radians
    :return: number of daylight hours corresponding to the sunset hour angle
    :rtype: float
    :raise ValueError: if the sunset hour angle is not within valid range
    """

    # validate the sunset hour angle argument, which has a valid
    # range of 0 to pi radians (180 degrees), inclusive
    # see http://mypages.iit.edu/~maslanka/SolarGeo.pdf
    if not 0.0 <= sunset_hour_angle_radians <= math.pi:
        raise ValueError("sunset hour angle outside valid range [0.0 to " +
                         str(math.pi) + "] : " +
                         str(sunset_hour_angle_radians) + " (actual value)")

    # calculate daylight hours from the sunset hour angle
    return (24.0 / math.pi) * sunset_hour_angle_radians


# ------------------------------------------------------------------------------
@numba.jit
def _monthly_mean_daylight_hours(
        latitude_radians: float,
        leap=False,
) -> np.ndarray:
    """
    :param latitude_radians: latitude in radians
    :param leap: whether or not values should be computed specific to leap years
    :return: the mean daily daylight hours for each calendar month of a year
    :rtype: numpy.ndarray of floats, 1-D with shape: (12,)
    """

    # get the array of days for each month based
    # on whether or not we're in a leap year
    if not leap:
        month_days = _MONTH_DAYS_NONLEAP
    else:
        month_days = _MONTH_DAYS_LEAP

    # allocate an array of daylight hours for each of the 12 months of the year
    monthly_mean_dlh = np.zeros((12,))

    # keep a count of the day of the year
    day_of_year = 1

    # loop over each calendar month to calculate the daylight hours for the month
    for i, days_in_month in enumerate(month_days):
        cumulative_daylight_hours = 0.0  # cumulative daylight hours for the month
        for _ in range(1, days_in_month + 1):
            daily_solar_declination = _solar_declination(day_of_year)
            daily_sunset_hour_angle = \
                _sunset_hour_angle(latitude_radians, daily_solar_declination)
            cumulative_daylight_hours += _daylight_hours(daily_sunset_hour_angle)
            day_of_year += 1

        # calculate the mean daylight hours of the month
        monthly_mean_dlh[i] = cumulative_daylight_hours / days_in_month

    return monthly_mean_dlh


# ------------------------------------------------------------------------------
@numba.jit
def eto_thornthwaite(
        monthly_temps_celsius: np.ndarray,
        latitude_degrees: float,
        data_start_year: int,
) -> np.ndarray:
    """
    Compute monthly potential evapotranspiration (PET) using the
    Thornthwaite (1948) method.

    Thornthwaite's equation:

        *PET* = 1.6 (*L*/12) (*N*/30) (10*Ta* / *I*)***a*

    where:

    * *Ta* is the mean daily air temperature, in degrees Celsius (if negative
        then use 0.0), of the month being calculated
    * *N* is the number of days in the month being calculated
    * *L* is the mean day length, in hours, of the month being calculated
    * *a* = (6.75 x 10-7)*I***3 - (7.71 x 10-5)*I***2 + (1.792 x 10-2)*I* + 0.49239
    * *I* is a heat index which depends on the 12 monthly mean temperatures and
        is calculated as the sum of (*Tai* / 5)**1.514 for each month, where
        *Tai* is the air temperature for each month in the year

    Reference:
    Thornthwaite, C.W. (1948) An approach toward a rational classification
    of climate. Geographical Review, Vol. 38, 55-94.
    https://www.jstor.org/stable/210739

    :param monthly_temps_celsius: array containing a time series (monthly time
        steps) of mean daily air temperatures in degrees Celsius. This input
        dataset is assumed to start at January of the initial year, and can have
        any length. Both 1-D (months) and 2-D (years, 12) input datasets
        are supported.
    :param latitude_degrees: latitude of the location, in degrees north (-90..90)
    :param data_start_year: year corresponding to the start of the dataset
    :return: estimated potential evapotranspiration, in millimeters/month
    :rtype: 1-D numpy.ndarray of floats with shape: (total # of months)
    """

    original_length = monthly_temps_celsius.size

    # validate the input data array
    monthly_temps_celsius = utils.reshape_to_2d(monthly_temps_celsius, 12)

    # at this point we assume that our dataset array has shape (years, 12) where
    # each row is a year with 12 columns of monthly values (Jan, Feb, ..., Dec)

    # convert the latitude from degrees to radians
    latitude_radians = math.radians(latitude_degrees)

    # adjust negative temperature values to zero, since negative
    # values aren't allowed (no evaporation below freezing)
    # TODO this sometimes throws a RuntimeWarning for invalid value,
    #  perhaps as a result of a NaN, somehow use masking and/or NaN
    #  pre-check to eliminate the cause of this warning
    monthly_temps_celsius[monthly_temps_celsius < 0] = 0.0

    # mean the monthly temperature values over the month axis,
    # giving us 12 monthly means for the period of record
    mean_monthly_temps = np.nanmean(monthly_temps_celsius, axis=0)

    # calculate the heat index (I)
    heat_index = np.sum(np.power(mean_monthly_temps / 5.0, 1.514))

    # calculate the a coefficient
    a = ((6.75e-07 * heat_index ** 3)
         - (7.71e-05 * heat_index ** 2)
         + (1.792e-02 * heat_index)
         + 0.49239)

    # get mean daylight hours for both normal and leap years
    mean_daylight_hours_nonleap = \
        np.array(_monthly_mean_daylight_hours(latitude_radians, False))
    mean_daylight_hours_leap = \
        np.array(_monthly_mean_daylight_hours(latitude_radians, True))

    # allocate the PET array we'll fill
    pet = np.full(monthly_temps_celsius.shape, np.NaN)
    for year in range(monthly_temps_celsius.shape[0]):

        if calendar.isleap(data_start_year + year):
            month_days = _MONTH_DAYS_LEAP
            mean_daylight_hours = mean_daylight_hours_leap
        else:
            month_days = _MONTH_DAYS_NONLEAP
            mean_daylight_hours = mean_daylight_hours_nonleap

        # calculate the Thornthwaite equation
        pet[year, :] = (
            16
            * (mean_daylight_hours / 12.0)
            * (month_days / 30.0)
            * ((10.0 * monthly_temps_celsius[year, :] / heat_index) ** a)
        )

    # reshape the dataset from (years, 12) into (months),
    # i.e. convert from 2-D to 1-D, and truncate to the original length
    return pet.reshape(-1)[0:original_length]


# ------------------------------------------------------------------------------
@numba.jit
def eto_hargreaves(
        daily_tmin_celsius: np.ndarray,
        daily_tmax_celsius: np.ndarray,
        daily_tmean_celsius: np.ndarray,
        latitude_degrees: float,
) -> np.ndarray:
    """
    Compute daily potential evapotranspiration (PET) using the Hargreaves
    (1985) method. Based on equation 52 in Allen et al (1998).

    Input arrays are assumed to be 1-D (an arbitrary number of days) or 2-D
    (years x 366 days per year).

    :param daily_tmin_celsius: array of daily minimum temperature values,
        in degrees Celsius
    :param daily_tmax_celsius: array of daily maximum temperature values,
        in degrees Celsius
    :param daily_tmean_celsius: array of daily mean temperature values,
        in degrees Celsius
    :param latitude_degrees: latitude of location, in degrees north
    :return: 1-D array of potential evapotranspiration over grass (ETo),
        in millimeters per day
    """

    # validate the input data arrays
    if daily_tmin_celsius.size != daily_tmax_celsius != daily_tmean_celsius:
        message = "Incompatible array sizes"
        _logger.error(message)
        raise ValueError(message)

    # keep the original length for conversion back to original size
    original_length = daily_tmean_celsius.size

    # reshape to 2-D with 366 days per year, if not already in this shape
    daily_tmean_celsius = utils.reshape_to_2d(daily_tmean_celsius, 366)

    # at this point we assume that our dataset array has shape (years, 366)
    # where each row is a year with 366 columns of daily values

    # convert the latitude from degrees to radians
    latitude = math.radians(latitude_degrees)

    # allocate the PET array we'll fill
    pet = np.full(daily_tmean_celsius.shape, np.NaN)
    for day_of_year in range(1, daily_tmean_celsius.shape[1] + 1):

        # calculate the angle of solar declination and sunset hour angle
        solar_declination = _solar_declination(day_of_year)
        sunset_hour_angle = _sunset_hour_angle(latitude, solar_declination)

        # calculate the inverse relative distance between earth and sun
        # from the day of the year, based on FAO equation 23 in
        # Allen et al (1998).
        inv_rel_distance = 1 + (0.033 * math.cos((2.0 * math.pi / 365.0) * day_of_year))

        # extraterrestrial radiation
        tmp1 = (24.0 * 60.0) / math.pi
        tmp2 = sunset_hour_angle * math.sin(latitude) * math.sin(solar_declination)
        tmp3 = (
            math.cos(latitude)
            * math.cos(solar_declination)
            * math.sin(sunset_hour_angle)
        )
        et_radiation = tmp1 * _SOLAR_CONSTANT * inv_rel_distance * (tmp2 + tmp3)

        for year in range(daily_tmean_celsius.shape[0]):

            # calculate the Hargreaves equation
            tmin = daily_tmin_celsius[year, day_of_year - 1]
            tmax = daily_tmax_celsius[year, day_of_year - 1]
            tmean = daily_tmean_celsius[year, day_of_year - 1]
            pet[year, day_of_year - 1] = (
                0.0023 * (tmean + 17.8) * (tmax - tmin) ** 0.5 * 0.408 * et_radiation
            )

    # reshape the dataset from (years, 366) into (total days),
    # i.e. convert from 2-D to 1-D, and truncate to the original length
    return pet.reshape(-1)[0:original_length]

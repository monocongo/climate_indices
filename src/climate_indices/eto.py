"""Module for calculation of potential evapotranspiration.

Credits
-------
Derived from original code found in PyETo:
https://github.com/woodcrafty/PyETo

References
----------
Thornthwaite, C.W. (1948) An approach toward a rational classification
of climate. Geographical Review, Vol. 38, 55-94.
https://www.jstor.org/stable/210739

Allen, Richard et al (1998) Crop evapotranspiration - Guidelines for computing
crop water requirements - FAO Irrigation and drainage paper 56
ISBN 92-5-104219-5

Goswami, D. Yogi (2015) Principles of Solar Engineering, Third Edition
ISBN 97-8-146656-3780
"""

from __future__ import annotations

import calendar
import math
import time

import numpy as np

from climate_indices import compute, utils
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.logging_config import get_logger
from climate_indices.performance import check_large_array_memory

# retrieve structlog logger for this module
_logger = get_logger(__name__)

# declare the function names that should be included in the public API for this module
__all__ = ["eto_hargreaves", "eto_thornthwaite"]

# days of each calendar month, for non-leap and leap years
_MONTH_DAYS_NONLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_DAYS_LEAP = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# solar constant [ MJ m-2 min-1]
_SOLAR_CONSTANT = 0.0820

# angle values used within the _sunset_hour_angle() function defined below

# valid range for latitude, in radians
_LATITUDE_RADIANS_MIN = np.deg2rad(-90.0)
_LATITUDE_RADIANS_MAX = np.deg2rad(90.0)

# valid range for solar declination angle, in radians
# Goswami (2015), p.40
_SOLAR_DECLINATION_RADIANS_MIN = np.deg2rad(-23.45)
_SOLAR_DECLINATION_RADIANS_MAX = np.deg2rad(23.45)


def _sunset_hour_angle(
    latitude_radians: float | np.ndarray,
    solar_declination_radians: float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate sunset hour angle (*Ws*) from latitude and solar declination.

    Based on FAO equation 25 in Allen et al (1998).

    :param latitude_radians: latitude in radians, as a scalar or an array of
        per-cell latitudes
    :param solar_declination_radians: angle of solar declination in radians, as a
        scalar or an array broadcastable against the latitude
    :return: sunset hour angle in radians
    :rtype: float, or an array of the broadcast latitude/declination shape
    """

    # validate the latitude argument, element-wise when given per-cell latitudes
    if np.any(np.isnan(latitude_radians)) or not np.all(
        (_LATITUDE_RADIANS_MIN <= latitude_radians) & (latitude_radians <= _LATITUDE_RADIANS_MAX)
    ):
        raise InvalidArgumentError(
            f"Latitude outside valid range [{_LATITUDE_RADIANS_MIN!r} to {_LATITUDE_RADIANS_MAX!r}]. "
            f"Received: {latitude_radians!r}",
            argument_name="latitude_radians",
            argument_value=str(latitude_radians),
            valid_values=f"[{_LATITUDE_RADIANS_MIN!r}, {_LATITUDE_RADIANS_MAX!r}]",
        )

    # validate the solar declination angle argument, which can vary between
    # -23.45 and +23.45 degrees see Goswami (2015) p.40, and
    # http://www.itacanet.org/the-sun-as-a-source-of-energy/part-1-solar-astronomy/
    if np.any(np.isnan(solar_declination_radians)) or not np.all(
        (_SOLAR_DECLINATION_RADIANS_MIN <= solar_declination_radians)
        & (solar_declination_radians <= _SOLAR_DECLINATION_RADIANS_MAX)
    ):
        raise InvalidArgumentError(
            f"Solar declination angle outside valid range "
            f"[{_SOLAR_DECLINATION_RADIANS_MIN} to {_SOLAR_DECLINATION_RADIANS_MAX}]. "
            f"Received: {solar_declination_radians}",
            argument_name="solar_declination_radians",
            argument_value=str(solar_declination_radians),
            valid_values=f"[{_SOLAR_DECLINATION_RADIANS_MIN}, {_SOLAR_DECLINATION_RADIANS_MAX}]",
        )

    # calculate the cosine of the sunset hour angle (*Ws* in FAO 25)
    # from latitude and solar declination
    cos_sunset_hour_angle = -np.tan(latitude_radians) * np.tan(solar_declination_radians)

    # If the cosine of the sunset hour angle is >= 1 there is no sunrise, i.e. 24 hours of darkness
    # If the cosine of the sunset hour angle is <= -1 there is no sunset, i.e. 24 hours of daylight
    # See http://www.itacanet.org/the-sun-as-a-source-of-energy/part-3-calculating-solar-angles/
    # Domain of acos is -1 <= x <= 1 radians (this is not mentioned in FAO-56!)
    sunset_hour_angle: float | np.ndarray = np.arccos(np.clip(cos_sunset_hour_angle, -1.0, 1.0))
    return sunset_hour_angle


def _solar_declination(
    day_of_year: int | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate the angle of solar declination from day of the year.

    Based on FAO equation 24 in Allen et al (1998).

    :param day_of_year: day of year integer between 1 and 365 (or 366,
        in the case of a leap year), or an array of such integers
    :return: solar declination [radians]
    :rtype: float, or an array of the same shape as the day of year input
    :raise ValueError: if the day of year value is not within the range [1-366]
    """
    if np.any(np.isnan(day_of_year)) or not np.all((1 <= day_of_year) & (day_of_year <= 366)):
        raise InvalidArgumentError(
            f"Day of the year must be in the range [1, 366]. Received: {day_of_year!r}",
            argument_name="day_of_year",
            argument_value=str(day_of_year),
            valid_values="[1, 366]",
        )

    return 0.409 * np.sin((2.0 * np.pi / 365.0) * day_of_year - 1.39)


def _daylight_hours(
    sunset_hour_angle_radians: float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate daylight hours from a sunset hour angle.

    Based on FAO equation 34 in Allen et al (1998).

    :param sunset_hour_angle_radians: sunset hour angle, in radians, as a scalar
        or an array of per-cell values
    :return: number of daylight hours corresponding to the sunset hour angle
    :rtype: float, or an array of the same shape as the sunset hour angle input
    :raise ValueError: if the sunset hour angle is not within valid range
    """

    # validate the sunset hour angle argument, which has a valid
    # range of 0 to pi radians (180 degrees), inclusive
    # see http://mypages.iit.edu/~maslanka/SolarGeo.pdf
    if np.any(np.isnan(sunset_hour_angle_radians)) or not np.all(
        (0.0 <= sunset_hour_angle_radians) & (sunset_hour_angle_radians <= math.pi)
    ):
        raise InvalidArgumentError(
            f"Sunset hour angle outside valid range [0.0 to {math.pi}]. Received: {sunset_hour_angle_radians}",
            argument_name="sunset_hour_angle_radians",
            argument_value=str(sunset_hour_angle_radians),
            valid_values=f"[0.0, {math.pi}]",
        )

    # calculate daylight hours from the sunset hour angle
    return (24.0 / np.pi) * sunset_hour_angle_radians


def _monthly_mean_daylight_hours(
    latitude_radians: float | np.ndarray,
    leap: bool = False,
) -> np.ndarray:
    """
    Computes the monthly mean daylight hours at the specified latitude.

    :param latitude_radians: latitude in radians, as a scalar or an array of
        per-cell latitudes
    :param leap: whether values should be computed specific to leap years or not
    :return: the mean daily daylight hours for each calendar month of a year
    :rtype: numpy.ndarray of floats with shape (12,) for a scalar latitude, or
        (12, *latitude shape) for an array of per-cell latitudes
    """

    # get the array of days for each month based
    # on whether we're in a leap year or not
    if not leap:
        month_days = _MONTH_DAYS_NONLEAP
    else:
        month_days = _MONTH_DAYS_LEAP

    # allocate an array of daylight hours for each of the 12 months of the year,
    # with one cell axis per latitude value (none for a scalar latitude)
    monthly_mean_dlh = np.zeros((12, *np.shape(latitude_radians)))

    # keep a count of the day of the year
    day_of_year = 1

    # loop over each calendar month to calculate the daylight hours for the month
    for i, days_in_month in enumerate(month_days):
        cumulative_daylight_hours: float | np.ndarray = 0.0  # cumulative daylight hours for the month
        for _ in range(1, days_in_month + 1):
            daily_solar_declination = _solar_declination(day_of_year)
            daily_sunset_hour_angle = _sunset_hour_angle(latitude_radians, daily_solar_declination)
            cumulative_daylight_hours = cumulative_daylight_hours + _daylight_hours(daily_sunset_hour_angle)
            day_of_year += 1

        # calculate the mean daylight hours of the month
        monthly_mean_dlh[i] = cumulative_daylight_hours / days_in_month

    return monthly_mean_dlh


def eto_thornthwaite(
    monthly_temps_celsius: np.ndarray,
    latitude_degrees: float | np.ndarray,
    data_start_year: int,
    spatial_time_major: bool = False,
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
        are supported. A time-major spatial block of shape (time, *cells), as
        declared with ``spatial_time_major``, is also supported.
    :param latitude_degrees: latitude of the location, in degrees north (-90..90),
        as a scalar or as an array of per-cell latitudes matching a time-major
        spatial block
    :param data_start_year: year corresponding to the start of the dataset
    :param spatial_time_major: read a three-or-more-dimensional input as a
        time-major spatial block, i.e. with the time steps first and the cells in
        the trailing dimensions, so the calculation runs once per cell set
    :return: estimated potential evapotranspiration, in millimeters/month
    :rtype: 1-D numpy.ndarray of floats with shape: (total # of months), or a
        time-major block of shape (time, *cells)
    """
    original_size = monthly_temps_celsius.size
    original_time_length = monthly_temps_celsius.shape[0]

    # validate (and fold) the input data array: a declared time-major block folds
    # into (years, 12, *cells), a 1-D/2-D series into (years, 12)
    spatial_block = spatial_time_major and monthly_temps_celsius.ndim > 2
    if spatial_block:
        values = compute._reshape_time_major(monthly_temps_celsius, compute.Periodicity.monthly)
    else:
        values = utils.reshape_to_2d(monthly_temps_celsius, 12)

    # at this point we assume that our dataset array has shape (years, 12, *cells) where
    # each row is a year with 12 columns of monthly values (Jan, Feb, ..., Dec)

    # a (time, *cells) block can be a read-only view of the caller's array, and the
    # negative-value adjustment below writes in place
    if spatial_block and not values.flags.writeable:
        values = values.copy()

    # adjust negative temperature values to zero, since negative
    # values aren't allowed (no evaporation below freezing)
    # TODO this sometimes throws a RuntimeWarning for invalid value,
    #  perhaps as a result of a NaN, somehow use masking and/or NaN
    #  pre-check to eliminate the cause of this warning
    values[values < 0] = 0.0

    # mean the monthly temperature values over the year axis, giving us 12 monthly
    # means for the period of record, one set per cell for a spatial block
    mean_monthly_temps = np.nanmean(values, axis=0)

    # calculate the heat index (I)
    heat_index = np.sum(np.power(mean_monthly_temps / 5.0, 1.514), axis=0)

    # calculate the coefficient
    a = (6.75e-07 * heat_index**3) - (7.71e-05 * heat_index**2) + (1.792e-02 * heat_index) + 0.49239

    # get mean daylight hours for both normal and leap years, per cell when
    # we've been given an array of per-cell latitudes
    if isinstance(latitude_degrees, np.ndarray):
        latitude_radians = np.radians(latitude_degrees)
    else:
        # float() keeps a non-numeric latitude raising the TypeError it always has
        latitude_radians = math.radians(float(latitude_degrees))
    mean_daylight_hours_nonleap = np.asarray(_monthly_mean_daylight_hours(latitude_radians, False))
    mean_daylight_hours_leap = np.asarray(_monthly_mean_daylight_hours(latitude_radians, True))

    # the leap year selection is per year, and the day-length and month-length terms
    # carry the month axis and one axis per remaining cell dimension, so that a scalar
    # latitude's (12,) day-length array broadcasts over the cells rather than the months
    cell_axes = (1,) * (values.ndim - 2)
    if np.ndim(latitude_degrees) == 0:
        mean_daylight_hours_nonleap = mean_daylight_hours_nonleap.reshape(12, *cell_axes)
        mean_daylight_hours_leap = mean_daylight_hours_leap.reshape(12, *cell_axes)

    years = values.shape[0]
    leap_years = np.array([calendar.isleap(data_start_year + year) for year in range(years)])
    mean_daylight_hours = np.where(
        leap_years.reshape(years, 1, *cell_axes),
        mean_daylight_hours_leap[None],
        mean_daylight_hours_nonleap[None],
    )
    month_days = np.where(
        leap_years.reshape(years, 1),
        _MONTH_DAYS_LEAP.reshape(1, 12),
        _MONTH_DAYS_NONLEAP.reshape(1, 12),
    ).reshape(years, 12, *cell_axes)

    # calculate the Thornthwaite equation, one term per year, month, and cell
    pet: np.ndarray = 16 * (mean_daylight_hours / 12.0) * (month_days / 30.0) * ((10.0 * values / heat_index) ** a)

    if spatial_block:
        # (years, 12, *cells) back to the time-major input layout, dropping any
        # padded time steps beyond the original number of them
        return pet.reshape(-1, *values.shape[2:])[0:original_time_length]

    # reshape the dataset from (years, 12) into (months),
    # i.e. convert from 2-D to 1-D, and truncate to the original length
    return pet.reshape(-1)[0:original_size]


def eto_hargreaves(
    daily_tmin_celsius: np.ndarray,
    daily_tmax_celsius: np.ndarray,
    daily_tmean_celsius: np.ndarray,
    latitude_degrees: float | np.ndarray,
    spatial_time_major: bool = False,
) -> np.ndarray:
    """
    Compute daily potential evapotranspiration (PET) using the Hargreaves
    (1985) method. Based on equation 52 in Allen et al (1998).

    Input arrays are assumed to be 1-D (an arbitrary number of days) or 2-D
    (years x 366 days per year). A time-major spatial block of shape
    (time, *cells), as declared with ``spatial_time_major``, is also supported,
    in which case the per-day extraterrestrial radiation is computed once per
    cell instead of once per grid cell.

    :param daily_tmin_celsius: array of daily minimum temperature values,
        in degrees Celsius
    :param daily_tmax_celsius: array of daily maximum temperature values,
        in degrees Celsius
    :param daily_tmean_celsius: array of daily mean temperature values,
        in degrees Celsius
    :param latitude_degrees: latitude of location, in degrees north, as a scalar
        or as an array of per-cell latitudes matching a time-major spatial block
    :param spatial_time_major: read a three-or-more-dimensional input as a
        time-major spatial block, i.e. with the time steps first and the cells in
        the trailing dimensions, so the calculation runs once per cell set
    :return: 1-D array of potential evapotranspiration over grass (ETo),
        in millimeters per day, or a time-major block of shape (time, *cells)
    """

    # validate the input data arrays
    if not (daily_tmin_celsius.size == daily_tmax_celsius.size == daily_tmean_celsius.size):
        message = (
            "Incompatible array sizes for Hargreaves ETo: "
            f"tmin={daily_tmin_celsius.size}, tmax={daily_tmax_celsius.size}, "
            f"tmean={daily_tmean_celsius.size}. All arrays must have the same size."
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="daily_tmin_celsius/daily_tmax_celsius/daily_tmean_celsius",
            argument_value=f"sizes ({daily_tmin_celsius.size}, {daily_tmax_celsius.size}, {daily_tmean_celsius.size})",
            valid_values="All arrays must have equal size",
        )

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="pet_hargreaves",
        input_shape=daily_tmean_celsius.shape,
        input_elements=daily_tmean_celsius.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()

    try:
        # validate temperature relationships: tmin <= tmean <= tmax
        # use warnings rather than errors since real-world data may have some anomalies
        tmin_gt_tmax = np.sum(daily_tmin_celsius > daily_tmax_celsius)
        if tmin_gt_tmax > 0:
            _logger.warning(f"Found {tmin_gt_tmax} instances where tmin > tmax. This may indicate data quality issues.")

        tmean_outside_range = np.sum(
            (daily_tmean_celsius < daily_tmin_celsius) | (daily_tmean_celsius > daily_tmax_celsius)
        )
        if tmean_outside_range > 0:
            _logger.warning(
                f"Found {tmean_outside_range} instances where tmean is outside [tmin, tmax] range. "
                "This may indicate data quality issues."
            )

        # keep the original length for conversion back to original size
        original_length = daily_tmean_celsius.size
        spatial_block = spatial_time_major and daily_tmean_celsius.ndim > 2

        # a declared time-major spatial block is read as given, one time step per row,
        # while a 1-D/2-D series is folded onto (years, 366) as it always was and then
        # read flat: the days of the year below cycle over that same year-major order.
        # Folding a block onto whole years would copy each of the three inputs in full
        # whenever the block ends mid-year, so the trailing partial year is indexed
        # where it lies rather than padded out into a whole year of its own.
        if not spatial_block:
            daily_tmin_celsius = utils.reshape_to_2d(daily_tmin_celsius, 366).reshape(-1)
            daily_tmax_celsius = utils.reshape_to_2d(daily_tmax_celsius, 366).reshape(-1)
            daily_tmean_celsius = utils.reshape_to_2d(daily_tmean_celsius, 366).reshape(-1)

        # at this point we can read each array as one time step per row,
        # i.e. (total days) for a 1-D/2-D input and (time, *cells) for a spatial block

        # convert the latitude from degrees to radians, keeping any per-cell axes
        if isinstance(latitude_degrees, np.ndarray):
            latitude = np.radians(latitude_degrees)
        else:
            latitude = math.radians(latitude_degrees)

        # allocate the PET array we'll fill, and account for it alongside the input
        # arrays: nothing above is padded, so these four arrays are the peak footprint
        pet = np.full(daily_tmean_celsius.shape, np.nan)
        memory_metrics = check_large_array_memory(daily_tmin_celsius, daily_tmax_celsius, daily_tmean_celsius, pet)
        for day_of_year in range(1, 367):
            # calculate the angle of solar declination and sunset hour angle
            solar_declination = _solar_declination(day_of_year)
            sunset_hour_angle = _sunset_hour_angle(latitude, solar_declination)

            # calculate the inverse relative distance between earth and sun
            # from the day of the year, based on FAO equation 23 in
            # Allen et al (1998).
            inv_rel_distance = 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year))

            # extraterrestrial radiation
            tmp1 = (24.0 * 60.0) / np.pi
            tmp2 = sunset_hour_angle * np.sin(latitude) * np.sin(solar_declination)
            tmp3 = np.cos(latitude) * np.cos(solar_declination) * np.sin(sunset_hour_angle)
            et_radiation = tmp1 * _SOLAR_CONSTANT * inv_rel_distance * (tmp2 + tmp3)

            # the rows holding this day of the year: one per whole year, plus the
            # trailing partial year's row once it reaches this day
            positions = np.arange(day_of_year - 1, daily_tmean_celsius.shape[0], 366)

            # calculate the Hargreaves equation for every year and cell of this day
            pet[positions] = (
                0.0023
                * (daily_tmean_celsius[positions] + 17.8)
                * (daily_tmax_celsius[positions] - daily_tmin_celsius[positions]) ** 0.5
                * 0.408
                * et_radiation
            )

        # a spatial block is returned in its input layout, while a 1-D/2-D input is
        # read flat and is truncated to its original length, dropping any padding
        result = pet if spatial_block else pet[0:original_length]
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

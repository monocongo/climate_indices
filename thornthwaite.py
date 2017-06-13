'''
Calculate potential evapotranspiration using the Thornthwaite method.

-----------------------------------------------------------------------------------------------------------------------
Credits:

Derived from original code found here: https://github.com/woodcrafty/PyETo

-----------------------------------------------------------------------------------------------------------------------
References:

Thornthwaite, C.W. (1948) An approach toward a rational classification of climate. Geographical Review, Vol. 38, 55-94.
https://www.jstor.org/stable/210739

Allen, Richard et al (1998) Crop evapotranspiration - Guidelines for computing crop water requirements - 
FAO Irrigation and drainage paper 56
ISBN 92-5-104219-5

Goswami, D. Yogi (2015) Principles of Solar Engineering, Third Edition
ISBN 97-8-146656-3780
'''

import calendar
import logging
import math
from numba import boolean, float64, int64, jit
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------

# days of each calendar month, for non-leap and leap years
_MONTH_DAYS_NONLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_DAYS_LEAP = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

#-----------------------------------------------------------------------------------------------------------------------
# compute angle values used within the _sunset_hour_angle() function defined below

# valid range for latitude, in radians
_LATITUDE_RADIANS_MIN = np.deg2rad(-90.0)
_LATITUDE_RADIANS_MAX = np.deg2rad(90.0)

# valid range for solar declination angle, in radians
# Goswami (2015), p.40
_SOLAR_DECLINATION_RADIANS_MIN = np.deg2rad(-23.45)
_SOLAR_DECLINATION_RADIANS_MAX = np.deg2rad(23.45)

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64(float64, float64))
def _sunset_hour_angle(latitude_radians,
                       solar_declination_radians):
    '''
    Calculate sunset hour angle (*Ws*) from latitude and solar declination.

    Based on FAO equation 25 in Allen et al (1998).

    :param latitude_radians: latitude in radians
    :param solar_declination_radians: angle of solar declination in radians
    :return: sunset hour angle in radians
    :rtype: float
    '''
    
    # validate the latitude argument
    if not _LATITUDE_RADIANS_MIN <= latitude_radians <= _LATITUDE_RADIANS_MAX:
        raise ValueError('latitude outside valid range [{0!r} to {1!r}]: {2!r}'
                         .format(_LATITUDE_RADIANS_MIN, _LATITUDE_RADIANS_MAX, latitude_radians))

    # validate the solar declination angle argument, which can vary between -23.45 and +23.45 degrees
    # see Goswami (2015) p.40, and http://www.itacanet.org/the-sun-as-a-source-of-energy/part-1-solar-astronomy/
    if not _SOLAR_DECLINATION_RADIANS_MIN <= solar_declination_radians <= _SOLAR_DECLINATION_RADIANS_MAX:
        raise ValueError('solar declination angle outside valid range [{0!r} to {1!r}]: {2!r}'
                         .format(_SOLAR_DECLINATION_RADIANS_MIN, _SOLAR_DECLINATION_RADIANS_MAX, solar_declination_radians))

    # calculate the cosine of the sunset hour angle (*Ws* in FAO 25) from latitude and solar declination
    cos_sunset_hour_angle = -math.tan(latitude_radians) * math.tan(solar_declination_radians)
    
    # If the sunset hour angle is >= 1 there is no sunset, i.e. 24 hours of daylight
    # If the sunset hour angle is <= 1 there is no sunrise, i.e. 24 hours of darkness
    # See http://www.itacanet.org/the-sun-as-a-source-of-energy/part-3-calculating-solar-angles/
    # Domain of acos is -1 <= x <= 1 radians (this is not mentioned in FAO-56!)
    return math.acos(min(max(cos_sunset_hour_angle, -1.0), 1.0))

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64(int64))
def _solar_declination(day_of_year):
    '''
    Calculate the angle of solar declination from day of the year.

    Based on FAO equation 24 in Allen et al (1998).

    :param day_of_year: day of year integer between 1 and 365 (or 366, in the case of a leap year)
    :return: solar declination [radians]
    :rtype: float
    :raise ValueError: if the day of year value is not within the range [1-366] 
    '''
    if not 1 <= day_of_year <= 366:
        raise ValueError('Day of the year must be in the range [1-366]: {0!r}'.format(day_of_year))

    return 0.409 * math.sin(((2.0 * math.pi / 365.0) * day_of_year - 1.39))

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64(float64))
def _daylight_hours(sunset_hour_angle_radians):
    '''
    Calculate daylight hours from a sunset hour angle.

    Based on FAO equation 34 in Allen et al (1998).

    :param sunset_hour_angle_radians: sunset hour angle, in radians
    :return: number of daylight hours corresponding to the sunset hour angle
    :rtype: float
    :raise ValueError: if the sunset hour angle is not within valid range
    '''
    
    # validate the sunset hour angle argument, which has a valid range of 0 to pi radians (180 degrees), inclusive
    # see http://mypages.iit.edu/~maslanka/SolarGeo.pdf
    if not 0.0 <= sunset_hour_angle_radians <= math.pi:
        raise ValueError('sunset hour angle outside valid range [{0!r} to {1!r}]: {2!r}'
                         .format(0.0, math.pi, sunset_hour_angle_radians))
    
    # calculate daylight hours from the sunset hour angle
    return (24.0 / math.pi) * sunset_hour_angle_radians

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64[:](float64, boolean))
def _monthly_mean_daylight_hours(latitude_radians, 
                                 leap=False):
    '''
    :param latitude_radians: latitude in radians
    :param leap: whether or not values should be computed specific to leap years
    :return: the mean daily daylight hours for each calendar month of a year
    :rtype: numpy.ndarray of floats, 1-D with shape: (12,)
    '''

    # get the array of days for each month based on whether or not we're in a leap year
    if leap == False:
        month_days = _MONTH_DAYS_NONLEAP
    else:
        month_days = _MONTH_DAYS_LEAP
        
    # allocate an array of daylight hours for each of the 12 months of the year
    monthly_mean_dlh = np.zeros((12,))
    
    # keep a count of the day of the year
    day_of_year = 1
    
    # loop over each calendar month to calculate the daylight hours for the month
    for i, days_in_month in enumerate(month_days):
        cumulative_daylight_hours = 0.0   # cumulative daylight hours for the month
        for _ in range(1, days_in_month + 1):
            daily_solar_declination = _solar_declination(day_of_year)
            daily_sunset_hour_angle = _sunset_hour_angle(latitude_radians, daily_solar_declination)
            cumulative_daylight_hours += _daylight_hours(daily_sunset_hour_angle)
            day_of_year += 1
        
        # calculate the mean daylight hours of the month
        monthly_mean_dlh[i] = cumulative_daylight_hours / days_in_month
        
    return monthly_mean_dlh

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64[:](float64[:], float64, int64))
def potential_evapotranspiration(monthly_temps_celsius, 
                                 latitude_degrees, 
                                 data_start_year):
    '''
    Compute monthly potential evapotranspiration (PET) using the Thornthwaite (1948) method.

    Thornthwaite's equation:

        *PET* = 1.6 (*L*/12) (*N*/30) (10*Ta* / *I*)***a*

    where:

    * *Ta* is the mean daily air temperature, in degrees Celsius (if negative use 0.0), of the month being calculated
    * *N* is the number of days in the month being calculated
    * *L* is the mean day length, in hours, of the month being calculated
    * *a* = (6.75 x 10-7)*I***3 - (7.71 x 10-5)*I***2 + (1.792 x 10-2)*I* + 0.49239
    * *I* is a heat index which depends on the 12 monthly mean temperatures and is calculated as 
        the sum of (*Tai* / 5)**1.514 for each month, where *Tai* is the air temperature for each month in the year

    Reference:
    Thornthwaite, C.W. (1948) An approach toward a rational classification of climate. 
    Geographical Review, Vol. 38, 55-94.
    https://www.jstor.org/stable/210739
    
    :param monthly_temps_celsius: array containing a time series (monthly time steps) of mean daily air temperatures in degrees Celsius.
                                  This input dataset is assumed to start at January of the initial year, and can have any length. 
                                  Both 1-D (months) and 2-D (years, 12) input datasets are supported.
    :param latitude_radians: latitude_radians of the location, in degrees north (-90..90)
    :param data_start_year: year corresponding to the start of the dataset  
    :return: estimated potential evapotranspiration, in millimeters/month
    :rtype: 1-D numpy.ndarray of floats with shape: (total # of months)

    '''

    # validate the input data array
    data_shape = monthly_temps_celsius.shape
    original_length = monthly_temps_celsius.size
    if len(data_shape) == 1:
        
        # dataset is assumed to represent one long row of months starting with January of the initial year, and we'll 
        # reshape into (years, 12) where each row is a year with 12 columns of monthly values (Jan, Feb, ..., Dec)
        
        # get the number of months left off of the final year
        final_year_empty_months = 12 - (data_shape[0] % 12)
        if final_year_empty_months == 12:
            final_year_empty_months = 0
        
        # if any months were left off then we'll pad the final months of the year with NaNs
        if final_year_empty_months > 0:
            
            # make an array of NaNs for each of the remaining months of the final year of the dataset
            final_year_pad = np.full((final_year_empty_months,), np.nan)
            
            # append the pad months to the dataset to complete the final year
            monthly_temps_celsius = np.concatenate((monthly_temps_celsius, final_year_pad))
        
        # reshape the dataset from (months) to (years, 12)
        total_years = monthly_temps_celsius.size // 12
        monthly_temps_celsius = np.reshape(monthly_temps_celsius, (total_years, 12))
                
    elif (len(data_shape) > 2) or ((len(data_shape) == 2) and (data_shape[1] != 12)):
        
        message = 'Input monthly mean temperatures data array has an invalid shape: {0}.'.format(data_shape)
        logger.error(message)
        raise ValueError(message)
    
    # at this point we assume that our dataset array has shape (years, 12) where 
    # each row is a year with 12 columns of monthly values (Jan, Feb, ..., Dec)
    
    # convert the latitude from degrees to radians
    latitude_radians = math.radians(latitude_degrees)
    
    # adjust negative temperature values to zero, since negative values aren't allowed (no evaporation below freezing)
    monthly_temps_celsius[monthly_temps_celsius < 0] = 0.0
    
    # mean the monthly temperature values over the month axis, giving us 12 monthly means for the period of record
    mean_monthly_temps = np.nanmean(monthly_temps_celsius, axis=0)    
    
    # calculate the heat index (I)
    I = np.sum(np.power(mean_monthly_temps / 5.0, 1.514))

    # calculate the a coefficient
    a = (6.75e-07 * I ** 3) - (7.71e-05 * I ** 2) + (1.792e-02 * I) + 0.49239

    # get mean daylight hours for both normal and leap years 
    mean_daylight_hours_nonleap = np.array(_monthly_mean_daylight_hours(latitude_radians, False))
    mean_daylight_hours_leap = np.array(_monthly_mean_daylight_hours(latitude_radians, True))
    
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
        pet[year, :] = 16 * (mean_daylight_hours / 12.0) * (month_days / 30.0) * ((10.0 * monthly_temps_celsius[year, :] / I) ** a)
    
    # reshape the dataset from (years, 12) into (months), i.e. convert from 2-D to 1-D, and truncate to the original length
    return pet.reshape(-1)[0:original_length]

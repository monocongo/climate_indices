"""Main level API module for computing climate indices."""

from __future__ import annotations

import time
from enum import Enum

import numpy as np

from climate_indices import compute, eto, utils
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.logging_config import get_logger

# declare the function names that should be included in the public API for this module
__all__ = ["percentage_of_normal", "pci", "pet", "spei", "spi"]


class Distribution(Enum):
    """
    Enumeration type for distribution fittings used for SPI and SPEI.
    """

    pearson = "pearson"
    gamma = "gamma"


# retrieve structlog logger for this module
_logger = get_logger(__name__)

# valid upper and lower bounds for indices that are fitted/transformed to a distribution (SPI and SPEI)
_FITTED_INDEX_VALID_MIN = -3.09
_FITTED_INDEX_VALID_MAX = 3.09

# valid range for scale parameter
SCALE_MIN = 1
SCALE_MAX = 72

# Import fallback strategy for consistent behavior
_fallback_strategy = compute.DistributionFallbackStrategy()


def _norm_fitdict(params: dict):
    """
    Compatibility shim. Convert old accepted parameter dictionaries
    into new, consistently keyed parameter dictionaries. If given
    a None object, None is returned.

    See https://github.com/monocongo/climate_indices/issues/449
    """
    if params is None:
        return params

    normed = {}
    for name, altname in _fit_altnames:
        val = params.get(name, None)
        if val is None:
            if altname not in params:
                continue
            _logger.warning(
                "Using deprecated fitting parameter key %s. Use %s instead.",
                altname,
                name,
            )
            val = params[altname]
        normed[name] = val
    return normed


_fit_altnames = (
    ("alpha", "alphas"),
    ("beta", "betas"),
    ("skew", "skews"),
    ("scale", "scales"),
    ("loc", "locs"),
    ("prob_zero", "probabilities_of_zero"),
)


def _validate_scale(scale: int) -> None:
    """Validate that scale is an integer within the valid range.

    Args:
        scale: The scale parameter to validate

    Raises:
        InvalidArgumentError: If scale is not an integer or is outside [SCALE_MIN, SCALE_MAX]
    """
    if not isinstance(scale, int) or scale < SCALE_MIN or scale > SCALE_MAX:
        message = (
            f"Invalid scale argument: {scale}. "
            f"Scale must be an integer in the range [{SCALE_MIN}, {SCALE_MAX}]. "
            f"Common scales: 1 (monthly), 3 (seasonal), 6 (half-year), 12 (annual)."
        )
        raise InvalidArgumentError(
            message,
            argument_name="scale",
            argument_value=str(scale),
            valid_values=f"[{SCALE_MIN}, {SCALE_MAX}]",
        )


def _validate_distribution(distribution: Distribution) -> None:
    """Validate that distribution is a valid Distribution enum member.

    Args:
        distribution: The distribution parameter to validate

    Raises:
        InvalidArgumentError: If distribution is not a Distribution enum member
    """
    if not isinstance(distribution, Distribution):
        message = (
            f"Unsupported distribution: {distribution}. "
            f"Supported distributions: gamma, pearson. "
            f"Use indices.Distribution.gamma or indices.Distribution.pearson."
        )
        raise InvalidArgumentError(
            message,
            argument_name="distribution",
            argument_value=str(distribution),
            valid_values="gamma, pearson",
        )


def _validate_periodicity(periodicity: compute.Periodicity) -> None:
    """Validate that periodicity is a valid Periodicity enum member.

    Args:
        periodicity: The periodicity parameter to validate

    Raises:
        InvalidArgumentError: If periodicity is not a Periodicity enum member
    """
    if not isinstance(periodicity, compute.Periodicity):
        message = (
            f"Invalid periodicity argument: {periodicity}. "
            f"Periodicity must be a Periodicity enum member. "
            f"Supported values: monthly, daily. "
            f"Use compute.Periodicity.monthly or compute.Periodicity.daily."
        )
        raise InvalidArgumentError(
            message,
            argument_name="periodicity",
            argument_value=str(periodicity),
            valid_values="monthly, daily",
        )


def spi(
    values: np.ndarray,
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: dict = None,
) -> np.ndarray:
    """
    Computes SPI (Standardized Precipitation Index).

    :param values: 1-D numpy array of precipitation values, in any units,
        first value assumed to correspond to January of the initial year if
        the periodicity is monthly, or January 1st of the initial year if daily
    :param scale: number of time steps over which the values should be scaled
        before the index is computed
    :param distribution: distribution type to be used for the internal
        fitting/transform computation
    :param data_start_year: the initial year of the input precipitation dataset
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param periodicity: the periodicity of the time series represented by the
        input data, valid/supported values are 'monthly' and 'daily'
        'monthly' indicates an array of monthly values, assumed to span full
         years, i.e. the first value corresponds to January of the initial year
         and any missing final months of the final year filled with NaN values,
         with size == # of years * 12
         'daily' indicates an array of full years of daily values with 366 days
         per year, as if each year were a leap year and any missing final months
         of the final year filled with NaN values, with array size == (# years * 366)
    :param fitting_params: optional dictionary of pre-computed distribution
        fitting parameters, if the distribution is gamma then this dict should
        contain two arrays, keyed as "alpha" and "beta", and if the
        distribution is Pearson then this dict should contain four arrays keyed
        as "prob_zero", "loc", "scale", and "skew".
    :return SPI values fitted to the gamma distribution at the specified time
        step scale, unitless
    :rtype: 1-D numpy.ndarray of floats of the same length as the input array
        of precipitation values
    """
    # validate arguments
    _validate_scale(scale)
    _validate_distribution(distribution)
    _validate_periodicity(periodicity)

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="spi",
        scale=scale,
        distribution=distribution.value,
        input_shape=values.shape,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()

    try:
        # we expect to operate upon a 1-D array, so if we've been passed a 2-D array
        # then we flatten it, otherwise raise an error
        shape = values.shape
        if len(shape) == 2:
            values = values.flatten()
        elif len(shape) != 1:
            message = f"Invalid shape of input array: {shape} -- only 1-D and 2-D arrays are supported"
            _logger.error(message)
            raise ValueError(message)

        # if we're passed all missing values then we can't compute
        # anything, so we return the same array of missing values
        if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=values.shape)
            return values

        # clip any negative values to zero
        if np.amin(values) < 0.0:
            _logger.warning("Input contains negative values -- all negatives clipped to zero")
            values = np.clip(values, a_min=0.0, a_max=None)

        # remember the original length of the array, in order to facilitate
        # returning an array of the same size
        original_length = values.size

        # get a sliding sums array, with each time step's value scaled
        # by the specified number of time steps
        values = compute.sum_to_scale(values, scale)

        # reshape precipitation values to (years, 12) for monthly, or to (years, 366) for daily
        if periodicity == compute.Periodicity.monthly:
            values = utils.reshape_to_2d(values, 12)
        elif periodicity == compute.Periodicity.daily:
            values = utils.reshape_to_2d(values, 366)

        if distribution == Distribution.gamma:
            # get (optional) fitting parameters if provided
            if fitting_params is not None:
                alphas = fitting_params["alpha"]
                betas = fitting_params["beta"]
            else:
                alphas = None
                betas = None

            # fit the scaled values to a gamma distribution
            # and transform to corresponding normalized sigmas
            values = compute.transform_fitted_gamma(
                values,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                periodicity,
                alphas,
                betas,
            )
        elif distribution == Distribution.pearson:
            # get (optional) fitting parameters if provided
            if fitting_params is not None:
                probabilities_of_zero = fitting_params["prob_zero"]
                locs = fitting_params["loc"]
                scales = fitting_params["scale"]
                skews = fitting_params["skew"]
            else:
                probabilities_of_zero = None
                locs = None
                scales = None
                skews = None

            try:
                # fit the scaled values to a Pearson Type III distribution
                # and transform to corresponding normalized sigmas
                values = compute.transform_fitted_pearson(
                    values,
                    data_start_year,
                    calibration_year_initial,
                    calibration_year_final,
                    periodicity,
                    probabilities_of_zero,
                    locs,
                    scales,
                    skews,
                )

                # Check if fallback is needed due to excessive NaN values
                if _fallback_strategy.should_fallback_from_excessive_nans(values):
                    raise ValueError("Pearson distribution fitting resulted in excessive missing values")

            except (ValueError, Warning, compute.DistributionFittingError) as e:
                # Use centralized fallback strategy for consistent logging and behavior
                _fallback_strategy.log_fallback_warning(str(e), context="SPI computation")

                # Use Gamma distribution as fallback
                values = compute.transform_fitted_gamma(
                    values,
                    data_start_year,
                    calibration_year_initial,
                    calibration_year_final,
                    periodicity,
                    alphas=None,
                    betas=None,
                )

        # clip values to within the valid range, reshape the array back to 1-D
        values = np.clip(values, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX).flatten()

        # return the original size array
        result = values[0:original_length]
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=result.shape)
        return result
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
            calibration_period=f"{calibration_year_initial}-{calibration_year_final}",
        )
        raise


def spei(
    precips_mm: np.ndarray,
    pet_mm: np.ndarray,
    scale: int,
    distribution: Distribution,
    periodicity: compute.Periodicity,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict = None,
) -> np.ndarray:
    """
    Compute SPEI fitted to the specified distribution.

    PET values are subtracted from the precipitation values to come up with an array
    of (P - PET) values, which is then scaled to the specified months scale and
    finally fitted/transformed to SPEI values corresponding to the input
    precipitation time series.

    :param precips_mm: an array of monthly total precipitation values,
        in millimeters, should be of the same size (and shape?) as the input PET array
    :param pet_mm: an array of monthly PET values, in millimeters,
        should be of the same size (and shape?) as the input precipitation array
    :param scale: the number of months over which the values should be scaled
        before computing the indicator
    :param distribution: distribution type to be used for the internal
        fitting/transform computation
    :param periodicity: the periodicity of the time series represented by the
        input data, valid/supported values are 'monthly' and 'daily'
        'monthly' indicates an array of monthly values, assumed to span full
         years, i.e. the first value corresponds to January of the initial year
         and any missing final months of the final year filled with NaN values,
         with size == # of years * 12
         'daily' indicates an array of full years of daily values with 366 days
         per year, as if each year were a leap year and any missing final months
         of the final year filled with NaN values, with array size == (# years * 366)
    :param data_start_year: the initial year of the input datasets (assumes that
        the two inputs cover the same period)
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param fitting_params: optional dictionary of pre-computed distribution
        fitting parameters, if the distribution is gamma then this dict should
        contain two arrays, keyed as "alpha" and "beta", and if the
        distribution is Pearson then this dict should contain four arrays keyed
        as "prob_zero", "loc", "scale", and "skew"
        Older keys such as "alphas" and "probabilities_of_zero" are deprecated.
    :return: an array of SPEI values
    :rtype: numpy.ndarray of type float, of the same size and shape as the input
        PET and precipitation arrays
    """
    # validate arguments
    _validate_scale(scale)
    _validate_distribution(distribution)
    _validate_periodicity(periodicity)

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="spei",
        scale=scale,
        distribution=distribution.value,
        input_shape=precips_mm.shape,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()

    try:
        # Normalize fitting param keys
        fitting_params = _norm_fitdict(fitting_params)

        # if we're passed all missing values then we can't compute anything,
        # so we return the same array of missing values
        if (np.ma.is_masked(precips_mm) and precips_mm.mask.all()) or np.all(np.isnan(precips_mm)):
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=precips_mm.shape)
            return precips_mm

        # validate that the two input arrays are compatible
        if precips_mm.size != pet_mm.size:
            message = "Incompatible precipitation and PET arrays"
            _logger.error(message)
            raise ValueError(message)

        # clip any negative values to zero
        if np.amin(precips_mm) < 0.0:
            _logger.warning("Input contains negative values -- all negatives clipped to zero")
            precips_mm = np.clip(precips_mm, a_min=0.0, a_max=None)

        # subtract the PET from precipitation, adding an offset
        # to ensure that all values are positive
        p_minus_pet = (precips_mm.flatten() - pet_mm.flatten()) + 1000.0

        # remember the original length of the input array, in order to facilitate
        # returning an array of the same size
        original_length = precips_mm.size

        # get a sliding sums array, with each element's value
        # scaled by the specified number of time steps
        scaled_values = compute.sum_to_scale(p_minus_pet, scale)

        if distribution is Distribution.gamma:
            # get (optional) fitting parameters if provided
            if fitting_params is not None:
                alphas = fitting_params["alpha"]
                betas = fitting_params["beta"]
            else:
                alphas = None
                betas = None

            # fit the scaled values to a gamma distribution and
            # transform to corresponding normalized sigmas
            transformed_fitted_values = compute.transform_fitted_gamma(
                scaled_values,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                periodicity,
                alphas,
                betas,
            )

        elif distribution is Distribution.pearson:
            # get (optional) filtering parameters if provided
            if fitting_params is not None:
                probabilities_of_zero = fitting_params["prob_zero"]
                locs = fitting_params["loc"]
                scales = fitting_params["scale"]
                skews = fitting_params["skew"]
            else:
                probabilities_of_zero = None
                locs = None
                scales = None
                skews = None

            # fit the scaled values to a Pearson Type III distribution
            # and transform to corresponding normalized sigmas
            transformed_fitted_values = compute.transform_fitted_pearson(
                scaled_values,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                periodicity,
                probabilities_of_zero,
                locs,
                scales,
                skews,
            )

        # clip values to within the valid range, reshape the array back to 1-D
        values = np.clip(transformed_fitted_values, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX).flatten()

        # return the original size array
        result = values[0:original_length]
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=result.shape)
        return result
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
            calibration_period=f"{calibration_year_initial}-{calibration_year_final}",
        )
        raise


def percentage_of_normal(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: compute.Periodicity,
) -> np.ndarray:
    """
    This function finds the percent of normal values (average of each calendar
    month or day over a specified calibration period of years) for a specified
    time steps scale. The normal precipitation for each calendar time step is
    computed for the specified time steps scale, and then each time step's
    scaled value is compared against the corresponding calendar time step's
    average to determine the percentage of normal. The period that defines the
    normal is described by the calibration start and end years arguments.
    The calibration period typically used for US climate monitoring is 1981-2010.

    :param values: 1-D numpy array of precipitation values, any length, initial
        value assumed to be January of the data start year (January 1st of the
        start year if daily periodicity), see the description of the
        *periodicity* argument below for further clarification
    :param scale: integer number of months over which the normal value is
        computed (eg 3-months, 6-months, etc.)
    :param data_start_year: the initial year of the input monthly values array
    :param calibration_start_year: the initial year of the calibration period
        over which the normal average for each calendar time step is computed
    :param calibration_end_year: the final year of the calibration period over
        which the normal average for each calendar time step is computed
    :param periodicity: the periodicity of the time series represented by the
        input data, valid/supported values are 'monthly' and 'daily'
        'monthly' indicates an array of monthly values, assumed to span full
         years, i.e. the first value corresponds to January of the initial year
         and any missing final months of the final year filled with NaN values,
         with size == # of years * 12
         'daily' indicates an array of full years of daily values with 366 days
         per year, as if each year were a leap year and any missing final months
         of the final year filled with NaN values, with array size == (# years * 366)
    :return: percent of normal precipitation values corresponding to the
        scaled precipitation values array
    :rtype: numpy.ndarray of type float
    """
    # validate arguments
    _validate_scale(scale)
    _validate_periodicity(periodicity)

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="percentage_of_normal",
        scale=scale,
        input_shape=values.shape,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()

    try:
        # if doing monthly then we'll use 12 periods, corresponding to calendar
        # months, if daily assume years w/366 days
        if periodicity == compute.Periodicity.monthly:
            periodicity = 12
        elif periodicity == compute.Periodicity.daily:
            periodicity = 366

        # bypass processing if all values are masked
        if np.ma.is_masked(values) and values.mask.all():
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=values.shape)
            return values

        # make sure we've been provided with sane calibration limits
        if data_start_year > calibration_start_year:
            raise ValueError(
                "Invalid start year arguments (data and/or calibration): "
                "calibration start year is before the data start year",
            )
        if ((calibration_end_year - calibration_start_year + 1) * 12) > values.size:
            raise ValueError(
                "Invalid calibration period specified: total calibration years exceeds the actual number of years of data",
            )

        # get an array containing a sliding sum on the specified time step
        # scale -- i.e. if the scale is 3 then the first two elements will be
        # np.nan, since we need 3 elements to get a sum, and then from the third
        # element to the end the values will equal the sum of the corresponding
        # time step plus the values of the two previous time steps
        scale_sums = compute.sum_to_scale(values, scale)

        # extract the timesteps over which we'll compute the normal
        # average for each time step of the year
        calibration_years = calibration_end_year - calibration_start_year + 1
        calibration_start_index = (calibration_start_year - data_start_year) * periodicity
        calibration_end_index = calibration_start_index + (calibration_years * periodicity)
        calibration_period_sums = scale_sums[calibration_start_index:calibration_end_index]

        # for each time step in the calibration period, get the average of
        # the scale sum for that calendar time step (i.e. average all January sums,
        # then all February sums, etc.)
        averages = np.full((periodicity,), np.nan)
        for i in range(periodicity):
            averages[i] = np.nanmean(calibration_period_sums[i::periodicity])

        # TODO replace the below loop with a vectorized implementation
        # for each time step of the scale_sums array find its corresponding
        # percentage of the time steps scale average for its respective calendar time step
        percentages_of_normal = np.full(scale_sums.shape, np.nan)
        for i in range(scale_sums.size):
            # make sure we don't have a zero divisor
            divisor = averages[i % periodicity]
            if divisor > 0.0:
                percentages_of_normal[i] = scale_sums[i] / divisor

        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=percentages_of_normal.shape)
        return percentages_of_normal
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
            calibration_period=f"{calibration_start_year}-{calibration_end_year}",
        )
        raise


def pet(
    temperature_celsius: np.ndarray,
    latitude_degrees: float,
    data_start_year: int,
) -> np.ndarray:
    """
    This function computes potential evapotranspiration (PET) using
    Thornthwaite's equation.

    :param temperature_celsius: an array of average temperature values,
        in degrees Celsius
    :param latitude_degrees: the latitude of the location, in degrees north,
        must be within range [-90.0 ... 90.0] (inclusive), otherwise a
        ValueError is raised
    :param data_start_year: the initial year of the input dataset
    :return: an array of PET values, of the same size and shape as the input
        temperature values array, in millimeters/time step
    :rtype: 1-D numpy.ndarray of floats
    """
    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="pet_thornthwaite",
        input_shape=temperature_celsius.shape,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()

    try:
        # make sure we're not dealing with all NaN values
        if np.ma.isMaskedArray(temperature_celsius) and (temperature_celsius.count() == 0):
            # we started with all NaNs for the temperature, so just return the same as PET
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=temperature_celsius.shape)
            return temperature_celsius

        # we were passed a vanilla Numpy array, look for indices where the value == NaN
        if np.all(np.isnan(temperature_celsius)):
            # we started with all NaNs for the temperature, so just return the same
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=temperature_celsius.shape)
            return temperature_celsius

        # If we've been passed an array of latitude values then just use
        # the first one -- useful when applying this function with xarray.GroupBy
        # or numpy.apply_along_axis() where we've had to duplicate values in a 3-D
        # array of latitudes in order to correspond with a 3-D array of temperatures.
        if isinstance(latitude_degrees, np.ndarray) and (latitude_degrees.size > 1):
            latitude_degrees = latitude_degrees.flat[0]

        # make sure we're not dealing with a NaN or out-of-range latitude value
        if (latitude_degrees is not None) and not np.isnan(latitude_degrees) and (-90.0 < latitude_degrees < 90.0):
            # compute and return the PET values using Thornthwaite's equation
            result = eto.eto_thornthwaite(
                temperature_celsius,
                latitude_degrees,
                data_start_year,
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=result.shape)
            return result

        message = (
            f"Invalid latitude value: {latitude_degrees}"
            + " (must be in degrees north, between -90.0 and "
            + "90.0 inclusive)"
        )
        _logger.error(message)
        raise ValueError(message)
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def pci(
    rainfall_mm: np.ndarray,
) -> np.ndarray:
    """
    This function computes Precipitation Concentration Index(PCI, Oliver, 1980).

    :param rainfall_mm: an array of daily rainfall value in a year,
        in mm
    :return: PCI value for the year in aa numpy array
    :rtype: 1-D numpy.ndarray of float
    """
    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="pci",
        input_shape=rainfall_mm.shape,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()

    try:
        # make sure we're not dealing with all NaN values
        if np.ma.isMaskedArray(rainfall_mm) and (rainfall_mm.count() == 0):
            # we started with all NaNs for the rainfall, so just return the same
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=rainfall_mm.shape)
            return rainfall_mm

        # we were passed a vanilla Numpy array, look for indices where the value == NaN
        if np.all(np.isnan(rainfall_mm)):
            # we started with all NaNs for the rainfall, so just return the same
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=rainfall_mm.shape)
            return rainfall_mm

        # make sure we're not dealing with a NaN or out-of-range or less than the expected rainfall value
        if len(rainfall_mm) == 366 and not sum(np.isnan(rainfall_mm)):
            m = [31, 29, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
            start = 0
            numerator = 0
            denominator = 0

            for month in range(12):
                numerator = numerator + (sum(rainfall_mm[start : m[month]]) ** 2)
                denominator = denominator + sum(rainfall_mm[start : m[month]])

                start = m[month]

            result = np.array([(numerator / (denominator**2)) * 100])
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=result.shape)
            return result

        if len(rainfall_mm) == 365 and not sum(np.isnan(rainfall_mm)):
            m = [31, 28, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
            start = 0
            numerator = 0
            denominator = 0

            for month in range(12):
                numerator = numerator + (sum(rainfall_mm[start : m[month]]) ** 2)
                denominator = denominator + sum(rainfall_mm[start : m[month]])

                start = m[month]

            result = np.array([(numerator / (denominator**2)) * 100])
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info("calculation_completed", duration_ms=round(duration_ms, 2), output_shape=result.shape)
            return result

        message = (
            "NaN values exist in the time-series or the total number of days not "
            "in the year is not available, total days should be 366 or 365"
        )
        _logger.error(message)
        raise ValueError(message)
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise

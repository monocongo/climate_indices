"""Main level API module for computing climate indices."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, cast

import numpy as np
import structlog.stdlib

from climate_indices import compute, eto
from climate_indices.exceptions import DataShapeError, InvalidArgumentError
from climate_indices.logging_config import get_logger
from climate_indices.performance import check_large_array_memory

# declare the function names that should be included in the public API for this module
__all__ = ["eddi", "percentage_of_normal", "pci", "pet", "spei", "spi"]


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

# Hastings inverse normal approximation constants (Abramowitz & Stegun 26.2.23)
# used by EDDI for converting empirical probabilities to z-scores
_HASTINGS_C0 = 2.515517
_HASTINGS_C1 = 0.802853
_HASTINGS_C2 = 0.010328
_HASTINGS_D1 = 1.432788
_HASTINGS_D2 = 0.189269
_HASTINGS_D3 = 0.001308

# ceiling on the elements of EDDI's rank comparison, i.e. one chunk of the
# (climatology years x years x cells) count that ranks every calendar period; the
# boolean intermediate is held one byte per element, so this bounds it near 4 MB
# by chunking across cells rather than growing with the width of the spatial block
_EDDI_RANK_COMPARISON_ELEMENT_BUDGET = 4_000_000

# day-of-year start index of each calendar month, keyed by the number of days in
# the year; used as the np.add.reduceat boundaries when computing PCI
_PCI_MONTH_STARTS: dict[int, np.ndarray] = {
    365: np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]),
    366: np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]),
}


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
        message = (  # type: ignore[unreachable]
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
        message = (  # type: ignore[unreachable]
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


def _raise_if_unsupported_shape(values: np.ndarray, spatial_time_major: bool = False) -> None:
    """Raise the DataShapeError this module has always raised for unsupported input shapes.

    Args:
        values: The input array to validate
        spatial_time_major: Whether a three-or-more-dimensional input is a declared
            time-major block, shaped (time, *cells), which is then a supported shape

    Raises:
        DataShapeError: If the array is not 1-D, 2-D, or a declared time-major block
    """
    if values.ndim not in (1, 2) and not (spatial_time_major and values.ndim > 2):
        raise DataShapeError(
            f"Invalid shape of input array: {values.shape} -- only 1-D and 2-D arrays are supported, "
            "or a three-or-more-dimensional time-major block declared with spatial_time_major=True",
            expected_shape="(N,), (years, periods), or a declared (time, *cells) block",
            actual_shape=values.shape,
        )


def _log_calculation_completed(
    log: structlog.stdlib.BoundLogger,
    t0: float,
    output_shape: tuple[int, ...],
    memory_metrics: dict[str, float] | None,
) -> None:
    """Emit the "calculation_completed" event shared by every index function.

    Args:
        log: Logger already bound with the calling index's context.
        t0: Start time from ``time.perf_counter()``.
        output_shape: Shape of the value(s) about to be returned.
        memory_metrics: Metrics from ``check_large_array_memory``, or ``None``.
    """
    duration_ms = (time.perf_counter() - t0) * 1000.0
    log.info(
        "calculation_completed",
        duration_ms=round(duration_ms, 2),
        output_shape=output_shape,
        **(memory_metrics or {}),
    )


def _apply_per_cell(
    func: Callable[..., np.ndarray],
    *cell_arrays: np.ndarray,
    fitting_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Run a single-series kernel once per cell of time-major spatial arrays.

    Spatial input reaches the fitting-based indices as (time, *cells). The gamma
    fitting path evaluates every cell at once, but the Pearson Type III path fits each
    series separately with L-moments, so these calls still loop over cells here; see
    #940 for vectorizing that fit across a cell axis.

    Args:
        func: Kernel taking one 1-D series per array and returning its 1-D result.
        cell_arrays: Time-major arrays of identical shape, (time, *cells).
        fitting_params: Optional pre-computed fitting parameters. An array carrying
            the cell dimensions after its period axis, i.e. (period, *cells), is
            sliced down to the current cell; a period-only array is shared by every
            cell and passed through unchanged.

    Returns:
        The kernel results, packed like the input arrays.
    """
    cells = cell_arrays[0].shape[1:]
    result = np.empty(cell_arrays[0].shape, dtype=float)
    for cell_index in np.ndindex(*cells):
        position = (slice(None), *cell_index)
        cell_params = fitting_params
        if fitting_params is not None:
            cell_params = {}
            for key, value in fitting_params.items():
                param_shape = getattr(value, "shape", ())
                if len(param_shape) < 2:
                    # a period-only parameter array is shared by every cell
                    cell_params[key] = value
                elif param_shape[1:] == cells:
                    cell_params[key] = value[(slice(None), *cell_index)]
                else:
                    raise ValueError(
                        f"Fitting parameter '{key}' has shape {param_shape}, which carries cell dimensions "
                        f"{param_shape[1:]} that do not match the input's cells {cells}"
                    )
        cell_result = func(*[array[position] for array in cell_arrays], fitting_params=cell_params)
        result[position] = np.ma.filled(cell_result, np.nan)
    return result


def _hastings_inverse_normal(probability: np.ndarray) -> np.ndarray:
    """Convert cumulative probabilities to z-scores using the Hastings approximation.

    Implements the rational approximation from Abramowitz & Stegun (1965),
    equation 26.2.23. This matches the NOAA PSL Fortran reference implementation
    used in the original EDDI code.

    Args:
        probability: Array of cumulative probabilities in (0, 1). Values
            at the boundaries are clipped to avoid infinities.

    Returns:
        Array of z-scores (standard normal deviates).
    """
    # clip to avoid log(0) or division-by-zero at boundaries
    p = np.clip(probability, 1e-10, 1.0 - 1e-10)

    # work in the lower tail; flip if p > 0.5
    sign = np.where(p <= 0.5, -1.0, 1.0)
    p_lower = np.where(p <= 0.5, p, 1.0 - p)

    # Hastings rational approximation
    t = np.sqrt(-2.0 * np.log(p_lower))
    numerator = _HASTINGS_C0 + t * (_HASTINGS_C1 + t * _HASTINGS_C2)
    denominator = 1.0 + t * (_HASTINGS_D1 + t * (_HASTINGS_D2 + t * _HASTINGS_D3))
    z = sign * (t - numerator / denominator)

    return cast(np.ndarray, z)


def _validate_eddi_calibration_period(
    calibration_year_initial: int,
    calibration_year_final: int,
    data_start_year: int,
    data_end_year: int,
) -> None:
    """Raise InvalidArgumentError if EDDI's calibration period doesn't fit the data.

    Args:
        calibration_year_initial: First year of the calibration period.
        calibration_year_final: Last year of the calibration period.
        data_start_year: First year of the input PET dataset.
        data_end_year: Last year of the input PET dataset.

    Raises:
        InvalidArgumentError: If the calibration years are out of order or fall
            outside the data's year range.
    """
    if calibration_year_initial > calibration_year_final:
        message = (
            f"Invalid calibration year arguments: initial year "
            f"({calibration_year_initial}) is after final year ({calibration_year_final})"
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="calibration_year_initial",
            argument_value=str(calibration_year_initial),
        )

    if calibration_year_initial < data_start_year:
        message = (
            f"Invalid calibration year arguments: calibration start year "
            f"({calibration_year_initial}) is before data start year ({data_start_year})"
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="calibration_year_initial",
            argument_value=str(calibration_year_initial),
        )

    if calibration_year_final > data_end_year:
        message = (
            f"Invalid calibration year arguments: calibration end year "
            f"({calibration_year_final}) is after data end year ({data_end_year})"
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="calibration_year_final",
            argument_value=str(calibration_year_final),
        )


def eddi(
    pet_values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    *,
    spatial_time_major: bool = False,
) -> np.ndarray:
    """Compute the Evaporative Demand Drought Index (EDDI).

    EDDI uses a non-parametric empirical ranking approach following the NOAA
    Physical Sciences Laboratory (PSL) methodology. PET values are accumulated
    over the specified time scale, then ranked within each calendar period of
    the calibration window. Ranks are converted to cumulative probabilities
    using the Tukey plotting position and then transformed to z-scores via the
    Hastings inverse-normal approximation.

    Args:
        pet_values: 1-D numpy array of PET (potential evapotranspiration) values.
            The first value corresponds to January of ``data_start_year`` for
            monthly data or January 1st for daily data. 2-D arrays are accepted
            and will be flattened automatically. A time-major spatial block,
            shaped (time, *cells), ranks every cell in one pass when it is
            declared with ``spatial_time_major``.
        scale: Number of time steps over which PET values are accumulated
            before ranking. Must be in [1, 72].
        data_start_year: First year of the input PET dataset.
        calibration_year_initial: First year of the calibration period used
            for empirical ranking.
        calibration_year_final: Last year of the calibration period.
        periodicity: Temporal resolution of the input data. Use
            ``compute.Periodicity.monthly`` (12 values/year) or
            ``compute.Periodicity.daily`` (366 values/year).
        spatial_time_major: Read a three-or-more-dimensional ``pet_values`` as a
            time-major block of independent time series, shaped (time, *cells),
            and rank every cell against its own climatology in one pass. The
            xarray adapter sets this for every block it packs; a direct NumPy
            caller has to declare it, since a block is not a shape EDDI reads
            without being told.

    Returns:
        1-D numpy array of EDDI values (unitless z-scores), same length as
        the input, or the same (time, *cells) shape as a declared block. Values
        are clipped to [-3.09, 3.09].

    Raises:
        DataShapeError: If the input array has more than 2 dimensions and is not
            a declared time-major block.
        InvalidArgumentError: If scale, periodicity, or calibration years
            are invalid.
    """
    # validate arguments
    _validate_scale(scale)
    _validate_periodicity(periodicity)

    # bind structured logging context
    log = _logger.bind(
        index_type="eddi",
        scale=scale,
        input_shape=pet_values.shape,
        input_elements=pet_values.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(pet_values)

    try:
        # remember the original length and shape of the array, in order to facilitate
        # returning an array of the same size and layout
        original_length = pet_values.size
        original_shape = pet_values.shape

        # input shapes other than 1-D/2-D keep this index's legacy DataShapeError,
        # and a declared block is the only way a 3-D or higher input is accepted
        _raise_if_unsupported_shape(pet_values, spatial_time_major)

        # an all-missing block is returned as it arrived, as the preparation seam
        # does for the 1-D and 2-D layouts
        if pet_values.ndim > 2 and (
            (isinstance(pet_values, np.ma.MaskedArray) and pet_values.mask.all()) or np.all(np.isnan(pet_values))
        ):
            _log_calculation_completed(log, t0, pet_values.shape, memory_metrics)
            return pet_values

        # flatten, clip negatives to zero, and scale/reshape in the shared preparation seam
        pet_values = compute.prepare_scaled(pet_values, scale, periodicity, spatial_time_major=spatial_time_major)
        num_periods = periodicity.period_length

        # an all-missing input comes back un-reshaped, so there's nothing to compute
        if pet_values.ndim == 1:
            _log_calculation_completed(log, t0, pet_values.shape, memory_metrics)
            return pet_values

        # NOAA ranks left-padded scale values below valid observations. The pads are
        # the first (scale - 1) time steps, which for a spatial block is a slice of
        # whole rows rather than the first (scale - 1) elements of the flat array.
        leading_scale_pads = np.zeros(pet_values.shape, dtype=bool)
        pad_rows = leading_scale_pads.reshape(-1, *pet_values.shape[2:])
        pad_rows[: min(scale - 1, pad_rows.shape[0])] = True

        # compute data dimensions for validation
        num_years = pet_values.shape[0]
        data_end_year = data_start_year + num_years - 1

        # validate calibration period
        _validate_eddi_calibration_period(
            calibration_year_initial, calibration_year_final, data_start_year, data_end_year
        )

        # determine calibration period indices
        calibration_start_year_index = calibration_year_initial - data_start_year
        calibration_end_year_index = calibration_year_final - data_start_year

        # Rank every calendar period against its own climatology. The rank is a count
        # of climatology values below the current value, so each period is walked as a
        # (climatology years, years, cells) comparison: one pass over the periods, not
        # over the grid cells, and the comparison is chunked across cells so that
        # intermediate stays bounded for a wide spatial block, where comparing every
        # cell in the block at once would grow with the block's width rather than
        # holding steady at the number of calibration years. Missing climatology values
        # never compare below a value, so they stay out of the count.
        climatology = pet_values[calibration_start_year_index : calibration_end_year_index + 1]
        climatology_valid_counts = np.count_nonzero(~np.isnan(climatology), axis=0)
        leading_pads_count = np.count_nonzero(
            leading_scale_pads[calibration_start_year_index : calibration_end_year_index + 1],
            axis=0,
        )
        cell_shape = pet_values.shape[2:]
        cells_per_time_step = int(np.prod(cell_shape, dtype=np.int64)) or 1
        num_climatology_years = climatology.shape[0]
        cells_per_chunk = max(1, _EDDI_RANK_COMPARISON_ELEMENT_BUDGET // (num_climatology_years * num_years))
        eddi_values = np.empty(pet_values.shape, dtype=float)

        for period_index in range(num_periods):
            period_climatology = climatology[:, period_index].reshape(num_climatology_years, cells_per_time_step)
            period_values = pet_values[:, period_index].reshape(num_years, cells_per_time_step)
            below = np.zeros(period_values.shape, dtype=np.int64)
            for cell_start in range(0, cells_per_time_step, cells_per_chunk):
                cell_chunk = slice(cell_start, cell_start + cells_per_chunk)
                below[:, cell_chunk] = np.count_nonzero(
                    period_climatology[:, None, cell_chunk] < period_values[:, cell_chunk], axis=0
                )

            # NOAA uses zero-based ranks and treats leading scale pads as lower than every
            # observed value; this is the Tukey plotting position of that rank
            period_pads = leading_pads_count[period_index]
            probabilities = (period_pads + below.reshape(num_years, *cell_shape) + 0.66) / (
                climatology_valid_counts[period_index] + period_pads + 0.33
            )

            # a period whose climatology holds fewer than two valid values has no ranking
            # at all, and a missing value stays missing
            probabilities = np.where(
                np.isnan(pet_values[:, period_index]) | (climatology_valid_counts[period_index] < 2),
                np.nan,
                probabilities,
            )

            # clip the probability to its valid range to avoid log(0), then apply the
            # Hastings inverse normal approximation. Both are elementwise and are held one
            # calendar period at a time: the approximation allocates several temporaries,
            # and over a whole wide block they would outweigh the input by an order of
            # magnitude (a daily block would be the worst case, with 366 periods).
            eddi_values[:, period_index] = _hastings_inverse_normal(
                np.clip(probabilities, 1e-10, 1.0 - 1e-10),
            )

        # clip values to within the valid range, and return an array of the input layout:
        # a spatial block keeps its cell dimensions and drops the padded final period
        np.clip(eddi_values, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX, out=eddi_values)
        if eddi_values.ndim > 2:
            result = eddi_values.reshape(-1, *eddi_values.shape[2:])[: original_shape[0]]
        else:
            result = eddi_values.flatten()[0:original_length]
        _log_calculation_completed(log, t0, result.shape, memory_metrics)
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


def spi(
    values: np.ndarray,
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: dict[str, Any] | None = None,
    *,
    spatial_time_major: bool = False,
) -> np.ndarray:
    """
    Computes SPI (Standardized Precipitation Index).

    :param values: 1-D numpy array of precipitation values, in any units,
        first value assumed to correspond to January of the initial year if
        the periodicity is monthly, or January 1st of the initial year if daily.
        A time-major spatial array with shape (time, *cells), i.e. three or more
        dimensions, is also accepted, and then every cell is scaled and fitted in
        one pass; that layout steps outside the per-cell path for the gamma
        distribution only, since the Pearson Type III fit still runs once per series.
        Two-dimensional input is still read as the legacy (years, periods) layout
        and flattened into a single series, not treated as a (time, cells) grid.
        When the first cell axis is a calendar period length (12 or 366) the shape is
        equally readable as a (years, periods, *cells) array, and then the reading has to
        be declared with ``spatial_time_major``; the xarray adapter declares every block
        it packs, and only that ambiguous shape raises without a declaration.
    :param scale: number of time steps over which the values should be scaled
        before the index is computed
    :param distribution: distribution type to be used for the internal
        fitting/transform computation
    :param data_start_year: the initial year of the input precipitation dataset
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param periodicity: periodicity of the input time series; use
        ``compute.Periodicity.monthly`` for monthly data (12 values/year) or
        ``compute.Periodicity.daily`` for daily data (366 values/year).
    :param fitting_params: optional dictionary of pre-computed distribution
        fitting parameters, if the distribution is gamma then this dict should
        contain two arrays, keyed as "alpha" and "beta", and if the
        distribution is Pearson then this dict should contain four arrays keyed
        as "prob_zero", "loc", "scale", and "skew". Older keys such as
        "alphas" and "probabilities_of_zero" are deprecated. For spatial input a 1-D
        parameter array is read as one value per calendar period and broadcast
        across cells.
    :param spatial_time_major: read ``values`` as a time-major block of independent
        time series, shaped (time, *cells), and fit every cell in one pass. The
        xarray adapter sets this for every block it packs; the NumPy API requires
        it only for an ambiguous shape, where the first cell axis is a calendar
        period length (12 or 366) and could be read as (years, periods, *cells).
    :return: SPI values fitted to the gamma distribution at the specified time
        step scale, unitless
    :rtype: 1-D numpy.ndarray of floats of the same length as the input array
        of precipitation values, or of the same (time, *cells) shape when
        ``spatial_time_major`` is set
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
        input_elements=values.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(values)

    try:
        # normalize any deprecated fitting-parameter aliases once, before the per-cell
        # Pearson dispatch below, so the diagnostic stays bounded per spatial operation
        fitting_params = compute._normalize_fitting_params(fitting_params)

        # remember the original length and shape of the array, in order to facilitate
        # returning an array of the same size and layout
        original_length = values.size
        original_shape = values.shape

        # spatial input arrives time-major, packed as (time, *cells), and is fitted in a
        # single pass over every cell rather than one call per cell; the xarray adapter
        # is the caller that packs it that way. An all-missing block is returned as it
        # arrived, and the Pearson Type III fit still runs once per cell (see #940), so
        # those two cases leave this function's main flow alone.
        if values.ndim > 2:
            if not spatial_time_major and values.shape[1] in compute._PERIOD_LENGTHS:
                raise ValueError(
                    f"Invalid shape of input array: {values.shape} -- a (time, *cells) block whose first "
                    "cell axis is a calendar period length is ambiguous with a (years, periods, *cells) "
                    "array; declare it with spatial_time_major=True"
                )
            if (isinstance(values, np.ma.MaskedArray) and values.mask.all()) or np.all(np.isnan(values)):
                return values
            if distribution is Distribution.pearson:
                return _apply_per_cell(
                    functools.partial(
                        spi,
                        scale=scale,
                        distribution=distribution,
                        data_start_year=data_start_year,
                        calibration_year_initial=calibration_year_initial,
                        calibration_year_final=calibration_year_final,
                        periodicity=periodicity,
                    ),
                    values,
                    fitting_params=fitting_params,
                )

        # flatten, short-circuit all-missing input, clip negatives to zero,
        # and scale/reshape in the shared preparation seam. Shape errors raise the
        # plain ValueError from prepare_scaled -- spi()'s dimension errors are pinned
        # to ValueError by tests/test_backward_compat.py::TestErrorHierarchyDocumented,
        # unlike eddi()/percentage_of_normal() which use DataShapeError.
        values = compute.prepare_scaled(values, scale, periodicity, spatial_time_major=spatial_time_major)

        # an all-missing input comes back un-reshaped, so there's nothing to compute
        if values.ndim == 1:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "calculation_completed",
                duration_ms=round(duration_ms, 2),
                output_shape=values.shape,
                **(memory_metrics or {}),
            )
            return values

        # fit the scaled values to the specified distribution and transform to
        # corresponding normalized sigmas, falling back to gamma when a Pearson
        # Type III fit fails
        values = compute.fit_and_standardize(
            values,
            distribution,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            periodicity,
            fitting_params,
            fallback_to_gamma=True,
            fallback_context="SPI computation",
        )

        # clip values to within the valid range
        values = np.clip(values, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX)

        if values.ndim > 2:
            # (years, periods, *cells) back to the time-major input layout, dropping any
            # padded time steps beyond the original number of them
            result = values.reshape(-1, *values.shape[2:])[: original_shape[0]]
        else:
            # reshape the array back to 1-D and return the original size array
            result = values.flatten()[0:original_length]
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=result.shape,
            **(memory_metrics or {}),
        )
        result_values: np.ndarray = result
        return result_values
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
    fitting_params: dict[str, Any] | None = None,
    *,
    spatial_time_major: bool = False,
) -> np.ndarray:
    """
    Compute SPEI fitted to the specified distribution.

    PET values are subtracted from the precipitation values to come up with an array
    of (P - PET) values, which is then scaled to the specified months scale and
    finally fitted/transformed to SPEI values corresponding to the input
    precipitation time series.

    :param precips_mm: an array of monthly total precipitation values,
        in millimeters, should be of the same size (and shape?) as the input PET array.
        A time-major spatial array with shape (time, *cells), i.e. three or more
        dimensions, is also accepted, and then every cell is scaled and fitted in
        one pass; that layout steps outside the per-cell path for the gamma
        distribution only, since the Pearson Type III fit still runs once per series.
        Two-dimensional input is still read as the legacy (years, periods) layout
        and flattened into a single series, not treated as a (time, cells) grid.
        When the first cell axis is a calendar period length (12 or 366) the shape is
        equally readable as a (years, periods, *cells) array, and then the reading has to
        be declared with ``spatial_time_major``; the xarray adapter declares every block
        it packs, and only that ambiguous shape raises without a declaration.
    :param pet_mm: an array of monthly PET values, in millimeters,
        should be of the same size (and shape?) as the input precipitation array
    :param scale: the number of months over which the values should be scaled
        before computing the indicator
    :param distribution: distribution type to be used for the internal
        fitting/transform computation
    :param periodicity: periodicity of the input time series; use
        ``compute.Periodicity.monthly`` for monthly data (12 values/year) or
        ``compute.Periodicity.daily`` for daily data (366 values/year).
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
    :param spatial_time_major: read ``precips_mm``/``pet_mm`` as time-major blocks of
        independent time series, shaped (time, *cells), and fit every cell in one pass.
        The xarray adapter sets this for every block it packs; the NumPy API requires
        it only for an ambiguous shape, where the first cell axis is a calendar
        period length (12 or 366) and could be read as (years, periods, *cells).
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
        input_elements=precips_mm.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(precips_mm, pet_mm)

    try:
        # normalize any deprecated fitting-parameter aliases once, before the per-cell
        # Pearson dispatch below, so the diagnostic stays bounded per spatial operation
        fitting_params = compute._normalize_fitting_params(fitting_params)

        # if we're passed all missing values then we can't compute anything,
        # so we return the same array of missing values
        if (isinstance(precips_mm, np.ma.MaskedArray) and precips_mm.mask.all()) or np.all(np.isnan(precips_mm)):
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "calculation_completed",
                duration_ms=round(duration_ms, 2),
                output_shape=precips_mm.shape,
                **(memory_metrics or {}),
            )
            return precips_mm

        # a single PET time series is one series for every cell: give it singleton cell
        # axes so it broadcasts across a spatial block rather than looking mismatched
        if precips_mm.ndim > 2 and pet_mm.ndim == 1 and pet_mm.size == precips_mm.shape[0]:
            pet_mm = pet_mm.reshape((pet_mm.shape[0],) + (1,) * (precips_mm.ndim - 1))

        # validate that the two input arrays are compatible: a spatial block needs matching
        # time lengths and cell axes that broadcast together, while the series path keeps
        # its size-based check
        if precips_mm.ndim > 2 or pet_mm.ndim > 2:
            try:
                np.broadcast_shapes(precips_mm.shape, pet_mm.shape)
                compatible = True
            except ValueError:
                compatible = False
        else:
            compatible = precips_mm.size == pet_mm.size
        if not compatible:
            message = "Incompatible precipitation and PET arrays"
            _logger.error(message)
            raise ValueError(message)

        # spatial input arrives time-major, packed as (time, *cells), and is fitted in a
        # single pass over every cell rather than one call per cell; the xarray adapter
        # is the caller that packs it that way. An all-missing block returned above, and
        # the Pearson Type III fit still runs once per cell (see #940).
        if precips_mm.ndim > 2:
            if not spatial_time_major and precips_mm.shape[1] in compute._PERIOD_LENGTHS:
                raise ValueError(
                    f"Invalid shape of input array: {precips_mm.shape} -- a (time, *cells) block whose first "
                    "cell axis is a calendar period length is ambiguous with a (years, periods, *cells) "
                    "array; declare it with spatial_time_major=True"
                )
            if distribution is Distribution.pearson:
                return _apply_per_cell(
                    functools.partial(
                        spei,
                        scale=scale,
                        distribution=distribution,
                        periodicity=periodicity,
                        data_start_year=data_start_year,
                        calibration_year_initial=calibration_year_initial,
                        calibration_year_final=calibration_year_final,
                    ),
                    precips_mm,
                    pet_mm,
                    fitting_params=fitting_params,
                )

        # clip any negative values to zero. np.any(...) is NaN-safe, unlike np.amin.
        if bool(np.any(precips_mm < 0.0)):
            _logger.warning("Input contains negative values -- all negatives clipped to zero")
            precips_mm = np.clip(precips_mm, a_min=0.0, a_max=None)

        # subtract the PET from precipitation, adding an offset
        # to ensure that all values are positive
        if precips_mm.ndim > 2:
            p_minus_pet = (precips_mm - pet_mm) + 1000.0
        else:
            p_minus_pet = (precips_mm.flatten() - pet_mm.flatten()) + 1000.0

        # remember the original length and shape of the input array, in order to
        # facilitate returning an array of the same size and layout
        original_length = precips_mm.size
        original_shape = precips_mm.shape

        # get a sliding sums array, with each element's value
        # scaled by the specified number of time steps. The scale is applied to the
        # PET-adjusted values, which the fitting transform reshapes itself.
        scaled_values = compute.prepare_scaled(
            p_minus_pet,
            scale,
            periodicity,
            clip_negatives=False,
            # spatial values are reshaped here instead: the fitting transform reads
            # (years, periods, *cells) once an array has more than two dimensions
            reshape=p_minus_pet.ndim > 2,
            spatial_time_major=spatial_time_major,
        )

        # fit the scaled values to the specified distribution and transform to
        # corresponding normalized sigmas
        transformed_fitted_values = compute.fit_and_standardize(
            scaled_values,
            distribution,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            periodicity,
            fitting_params,
            fallback_to_gamma=False,
        )

        # clip values to within the valid range
        values = np.clip(transformed_fitted_values, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX)

        if values.ndim > 2:
            # (years, periods, *cells) back to the time-major input layout, dropping any
            # padded time steps beyond the original number of them
            result = values.reshape(-1, *values.shape[2:])[: original_shape[0]]
        else:
            # reshape the array back to 1-D and return the original size array
            result = values.flatten()[0:original_length]
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=result.shape,
            **(memory_metrics or {}),
        )
        result_values: np.ndarray = result
        return result_values
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
    *,
    spatial_time_major: bool = False,
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
    :param periodicity: periodicity of the input time series; use
        ``compute.Periodicity.monthly`` for monthly data (12 values/year) or
        ``compute.Periodicity.daily`` for daily data (366 values/year).
    :param spatial_time_major: read a three-or-more-dimensional ``values`` as a
        time-major block of independent time series, shaped (time, *cells), and
        divide every cell by its own calendar-period normals in one pass. The
        xarray adapter sets this for every block it packs; a direct NumPy caller
        has to declare it, since a block is not a shape this function reads
        without being told.
    :return: percent of normal precipitation values corresponding to the
        scaled precipitation values array: 1-D for a 1-D or 2-D input, or the
        (time, *cells) layout of a declared block
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
        input_elements=values.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(values)

    try:
        # we expect to operate upon a 1-D array, so if we've been passed a 2-D array
        # then we flatten it. Input shapes other than 1-D/2-D keep this index's legacy
        # DataShapeError, and a declared block is the only way a 3-D or higher input
        # is accepted.
        _raise_if_unsupported_shape(values, spatial_time_major)
        if values.ndim == 2:
            values = values.flatten()

        # calendar months for monthly data (12 periods), or days for daily data (366)
        period_length = periodicity.period_length

        # bypass processing if all values are masked, or when a spatial block is all
        # missing, in which case it is returned as it arrived
        if (isinstance(values, np.ma.MaskedArray) and values.mask.all()) or (
            values.ndim > 2 and np.all(np.isnan(values))
        ):
            _log_calculation_completed(log, t0, values.shape, memory_metrics)
            return values

        # make sure we've been provided with sane calibration limits
        if data_start_year > calibration_start_year:
            raise InvalidArgumentError(
                "Invalid start year arguments: calibration start year "
                f"({calibration_start_year}) is before the data start year ({data_start_year}).",
                argument_name="calibration_start_year",
                argument_value=str(calibration_start_year),
                valid_values=f">= data_start_year ({data_start_year})",
            )

        # note: this check counts 12 time steps per year regardless of periodicity,
        # as it always has. Tightening it to period_length would reject Gregorian
        # daily input (365/366 days per year), which is a separate behavior change.
        # A spatial block is measured on its time axis, since every cell shares it and
        # the element count would let any block pass on cell count alone.
        if ((calibration_end_year - calibration_start_year + 1) * 12) > values.shape[0]:
            raise InvalidArgumentError(
                "Invalid calibration period: total calibration years exceeds the "
                "actual number of years of data. "
                f"Calibration period: {calibration_start_year}-{calibration_end_year}, "
                f"data size: {values.shape[0]} time steps.",
                argument_name="calibration_end_year",
                argument_value=str(calibration_end_year),
                valid_values=f"calibration period must fit within {values.shape[0]} data values",
            )

        # get an array containing a sliding sum on the specified time step
        # scale -- i.e. if the scale is 3 then the first two elements will be
        # np.nan, since we need 3 elements to get a sum, and then from the third
        # element to the end the values will equal the sum of the corresponding
        # time step plus the values of the two previous time steps. Negatives are
        # left alone and the sums stay 1-D, since the calendar-period averages
        # below are computed over the un-reshaped scale sums.
        scale_sums = compute.prepare_scaled(
            values,
            scale,
            periodicity,
            clip_negatives=False,
            reshape=False,
            spatial_time_major=spatial_time_major,
        )

        # extract the timesteps over which we'll compute the normal
        # average for each time step of the year
        calibration_years = calibration_end_year - calibration_start_year + 1
        calibration_start_index = (calibration_start_year - data_start_year) * period_length
        calibration_end_index = calibration_start_index + (calibration_years * period_length)
        calibration_period_sums = scale_sums[calibration_start_index:calibration_end_index]

        if calibration_period_sums.size:
            # pad a trailing partial period with NaN (ignored by the average) so that the
            # calibration period reshapes into whole calendar periods, e.g. when the
            # calibration period extends past the end of the data
            if calibration_period_sums.shape[0] % period_length:
                calibration_period_sums = np.concatenate(
                    [
                        calibration_period_sums,
                        np.full(
                            (-calibration_period_sums.shape[0] % period_length, *calibration_period_sums.shape[1:]),
                            np.nan,
                        ),
                    ],
                )

            # for each time step in the calibration period, get the average of
            # the scale sum for that calendar time step (i.e. average all January sums,
            # then all February sums, etc.); a spatial block keeps its cell axes and
            # averages each cell's calibration years for every calendar time step
            averages = np.nanmean(
                calibration_period_sums.reshape(-1, period_length, *calibration_period_sums.shape[1:]),
                axis=0,
            )
        else:
            # the calibration window lies beyond the end of the data, so no normal
            # values are available -- every percentage is missing
            averages = np.full((period_length, *scale_sums.shape[1:]), np.nan)

        # for each time step of the scale_sums array find its corresponding percentage
        # of the time steps scale average for its respective calendar time step, leaving
        # NaN wherever the calendar time step's average is not a positive value
        averages = np.where(averages > 0.0, averages, np.nan)
        percentages_of_normal = np.full(scale_sums.shape, np.nan)

        # divide whole calendar periods at a time so that the repeating normals broadcast
        # from the (small) averages array rather than an input-sized divisor array
        whole_periods = scale_sums.shape[0] // period_length
        if whole_periods:
            np.divide(
                scale_sums[: whole_periods * period_length].reshape(
                    whole_periods, period_length, *scale_sums.shape[1:]
                ),
                averages,
                out=percentages_of_normal[: whole_periods * period_length].reshape(
                    whole_periods, period_length, *scale_sums.shape[1:]
                ),
            )

        # a trailing partial period uses the normals of the calendar time steps it covers
        remainder_start = whole_periods * period_length
        if remainder_start < scale_sums.shape[0]:
            np.divide(
                scale_sums[remainder_start:],
                averages[: scale_sums.shape[0] - remainder_start],
                out=percentages_of_normal[remainder_start:],
            )

        _log_calculation_completed(log, t0, percentages_of_normal.shape, memory_metrics)
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


def _pet_latitude(
    latitude_degrees: float | np.ndarray,
    temperature_celsius: np.ndarray,
    spatial_time_major: bool,
) -> float | np.ndarray:
    """Resolve the PET latitude argument, validating a scalar against its range.

    An array of latitudes resolves to its first value -- useful when applying PET with
    xarray.GroupBy or numpy.apply_along_axis(), where the latitudes are duplicated over a
    3-D array to match a 3-D temperature array. A declared time-major spatial block keeps
    the per-cell latitudes instead, so the calculation runs once per cell set rather than
    per cell; those arrays are validated by ``eto.eto_thornthwaite()``.

    Args:
        latitude_degrees: Latitude in degrees north, either scalar or per-cell
        temperature_celsius: The temperature block the latitude applies to
        spatial_time_major: Whether the temperature is a time-major spatial block

    Returns:
        The scalar latitude, or the per-cell latitude array of a spatial block

    Raises:
        ValueError: If the latitude array is empty, or a scalar latitude is None, NaN,
            or outside [-90.0 ... 90.0] (inclusive)
    """
    if isinstance(latitude_degrees, np.ndarray):
        if latitude_degrees.size == 0:
            message = "Invalid latitude value: empty latitude array (must contain at least one value)"
            _logger.error(message)
            raise ValueError(message)
        if not (spatial_time_major and temperature_celsius.ndim > 2):
            latitude_degrees = cast(float, latitude_degrees.flat[0])

    if not isinstance(latitude_degrees, np.ndarray) and (
        (latitude_degrees is None) or np.isnan(latitude_degrees) or not (-90.0 <= latitude_degrees <= 90.0)
    ):
        message = (
            f"Invalid latitude value: {latitude_degrees}"
            + " (must be in degrees north, between -90.0 and "
            + "90.0 inclusive)"
        )
        _logger.error(message)
        raise ValueError(message)

    return latitude_degrees


def pet(
    temperature_celsius: np.ndarray,
    latitude_degrees: float | np.ndarray,
    data_start_year: int,
    spatial_time_major: bool = False,
) -> np.ndarray:
    """Compute potential evapotranspiration (PET) using Thornthwaite's equation.

    Args:
        temperature_celsius (numpy.ndarray): An array of average temperature
            values, in degrees Celsius.
        latitude_degrees (float | numpy.ndarray): The latitude of the location,
            in degrees north. Must be within range [-90.0 ... 90.0] (inclusive).
            When ``spatial_time_major`` is declared for a three-or-more-dimensional
            input this may be an array of per-cell latitudes broadcastable to the
            trailing cell dimensions.
        data_start_year (int): The initial year of the input dataset.
        spatial_time_major (bool): Read a three-or-more-dimensional
            ``temperature_celsius`` as a time-major spatial block, i.e. with the
            time steps first and the cells in the trailing dimensions, and
            ``latitude_degrees`` as the per-cell latitude array matching those
            trailing dimensions. A 1-D or 2-D input ignores the declaration.

    Returns:
        numpy.ndarray: A 1-D array of float PET values, of the same size and
            shape as the input temperature values array, in millimeters/time
            step. A time-major spatial block returns in the same layout.

    Raises:
        ValueError: If ``latitude_degrees`` is an empty array, None, NaN, or a
            scalar outside [-90.0 ... 90.0] (inclusive).
        InvalidArgumentError: If a per-cell ``latitude_degrees`` array under
            ``spatial_time_major`` holds a value outside [-90.0 ... 90.0] (inclusive),
            or carries more dimensions than the input block has cell dimensions.
    """
    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="pet_thornthwaite",
        input_shape=temperature_celsius.shape,
        input_elements=temperature_celsius.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(temperature_celsius)

    try:
        latitude_degrees = _pet_latitude(latitude_degrees, temperature_celsius, spatial_time_major)

        # make sure we're not dealing with all NaN values
        if np.ma.isMaskedArray(temperature_celsius) and (temperature_celsius.count() == 0):
            # we started with all NaNs for the temperature, so just return the same as PET
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "calculation_completed",
                duration_ms=round(duration_ms, 2),
                output_shape=temperature_celsius.shape,
                **(memory_metrics or {}),
            )
            return temperature_celsius

        # we were passed a vanilla Numpy array, look for indices where the value == NaN
        if np.all(np.isnan(temperature_celsius)):
            # we started with all NaNs for the temperature, so just return the same
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "calculation_completed",
                duration_ms=round(duration_ms, 2),
                output_shape=temperature_celsius.shape,
                **(memory_metrics or {}),
            )
            return temperature_celsius

        # compute and return the PET values using Thornthwaite's equation
        result = eto.eto_thornthwaite(
            temperature_celsius,
            latitude_degrees,
            data_start_year,
            spatial_time_major=spatial_time_major,
        )
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


def pci(
    rainfall_mm: np.ndarray,
) -> np.ndarray:
    """
    This function computes Precipitation Concentration Index(PCI, Oliver, 1980).

    :param rainfall_mm: an array of daily rainfall value in a year,
        in mm
    :return: PCI value for the year in a numpy array
    :rtype: 1-D numpy.ndarray of float
    """
    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="pci",
        input_shape=rainfall_mm.shape,
        input_elements=rainfall_mm.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(rainfall_mm)

    try:
        # make sure we're not dealing with all NaN values
        if np.ma.isMaskedArray(rainfall_mm) and (rainfall_mm.count() == 0):
            # we started with all NaNs for the rainfall, so just return the same
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "calculation_completed",
                duration_ms=round(duration_ms, 2),
                output_shape=rainfall_mm.shape,
                **(memory_metrics or {}),
            )
            return rainfall_mm

        # we were passed a vanilla Numpy array, look for indices where the value == NaN
        if np.all(np.isnan(rainfall_mm)):
            # we started with all NaNs for the rainfall, so just return the same
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "calculation_completed",
                duration_ms=round(duration_ms, 2),
                output_shape=rainfall_mm.shape,
                **(memory_metrics or {}),
            )
            return rainfall_mm

        # make sure we're not dealing with a NaN or out-of-range or less than the expected rainfall value
        month_starts = _PCI_MONTH_STARTS.get(len(rainfall_mm))
        if month_starts is not None and not sum(np.isnan(rainfall_mm)):
            # masked values are treated as the missing (NaN) values they represent
            rainfall = (
                np.ma.filled(rainfall_mm.astype(float), np.nan) if np.ma.isMaskedArray(rainfall_mm) else rainfall_mm
            )

            # monthly rainfall totals, one per calendar month, then Oliver (1980) PCI
            monthly_totals = np.add.reduceat(rainfall, month_starts)
            pci_value = (np.sum(monthly_totals**2) / (np.sum(monthly_totals) ** 2)) * 100
            result = np.array([pci_value])
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "calculation_completed",
                duration_ms=round(duration_ms, 2),
                output_shape=result.shape,
                **(memory_metrics or {}),
            )
            return result

        message = "NaN values exist in the time-series or the total number of days in the year is not 365 or 366."
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="rainfall_mm",
            argument_value=f"array with {len(rainfall_mm)} elements",
            valid_values="array length must be 365 or 366",
        )
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise

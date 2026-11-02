"""Main level API module for computing climate indices."""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast, overload

import numpy as np
import scipy.special
import scipy.stats
import structlog.stdlib

from climate_indices import compute, eto, lmoments
from climate_indices._calibration_period import CalibrationPeriod, CalibrationPolicy, resolve_calibration_period
from climate_indices.exceptions import (
    BetaFeatureWarning,
    DataShapeError,
    InsufficientDataError,
    InvalidArgumentError,
    PeriodicityError,
    emit_deprecation_warning,
)
from climate_indices.logging_config import get_logger, log_calculation_failure
from climate_indices.performance import check_large_array_memory
from climate_indices.vocabulary import Distribution as Distribution
from climate_indices.vocabulary import _validate_hourly_distribution as _validate_hourly_distribution

# declare the function names that should be included in the public API for this module
__all__ = [
    "eddi",
    "fit_diagnostics",
    "percentage_of_normal",
    "pci",
    "pet",
    "spai",
    "spei",
    "spi",
    "standardized_index",
]


# retrieve structlog logger for this module
_logger = get_logger(__name__)

# valid upper and lower bounds for indices that are fitted/transformed to a distribution (SPI and SPEI)
_FITTED_INDEX_VALID_MIN = -3.09
_FITTED_INDEX_VALID_MAX = 3.09

# valid range for the scale parameter; the upper bound spans six calendar years
# for either periodicity, expressed in that periodicity's own time steps
SCALE_MIN = 1
SCALE_MAX = 72
SCALE_MAX_DAILY = 6 * 366
SCALE_MAX_HOURLY = 6 * 366 * 24

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


def _validate_scale(scale: int, periodicity: compute.Periodicity) -> None:
    """Validate that scale is an integer within the valid range for the periodicity.

    Args:
        scale: The scale parameter to validate
        periodicity: The periodicity whose time steps bound the scale

    Raises:
        InvalidArgumentError: If scale is not an integer or is outside the valid range
    """
    if periodicity is compute.Periodicity.daily:
        scale_max = SCALE_MAX_DAILY
        common_scales = "1 (daily), 7 (weekly), 30 (monthly), 90 (seasonal), 365 (annual)"
    elif periodicity is compute.Periodicity.hourly:
        scale_max = SCALE_MAX_HOURLY
        common_scales = (
            "24 (daily), 168 (weekly), 720 (monthly), 2160 (seasonal), 8760 (annual, 8784 in the 366-day layout)"
        )
    else:
        scale_max = SCALE_MAX
        common_scales = "1 (monthly), 3 (seasonal), 6 (half-year), 12 (annual)"
    if not isinstance(scale, int) or scale < SCALE_MIN or scale > scale_max:
        message = (
            f"Invalid scale argument: {scale}. "
            f"Scale must be an integer in the range [{SCALE_MIN}, {scale_max}]. "
            f"Common scales: {common_scales}."
        )
        raise InvalidArgumentError(
            message,
            argument_name="scale",
            argument_value=str(scale),
            valid_values=f"[{SCALE_MIN}, {scale_max}]",
        )


def _validate_distribution(
    distribution: Distribution,
    allowed: tuple[Distribution, ...] = (Distribution.gamma, Distribution.pearson),
) -> None:
    """Validate that distribution is a Distribution enum member allowed on this surface.

    The default ``allowed`` set omits log-logistic, so a zero-mass-sensitive surface
    that forgets to pass an explicit set fails safe: the generalized logistic is the
    SPEI reference distribution, whose offset P−PET input has no physical zero mass,
    and the zero-placement treatment a precipitation or generic non-negative series
    needs is not implemented for it (#106). ``spei()`` passes the three-member set.

    Args:
        distribution: The distribution parameter to validate
        allowed: The members accepted by the calling surface

    Raises:
        InvalidArgumentError: If distribution is not an allowed Distribution member
    """
    supported = allowed
    valid_values = ", ".join(member.value for member in supported)
    if isinstance(distribution, Distribution) and distribution in supported:
        return

    remediation = ", ".join(f"indices.Distribution.{member.name}" for member in supported)
    message = f"Unsupported distribution: {distribution}. Supported distributions: {valid_values}. Use {remediation}."
    if distribution is Distribution.loglogistic:
        message += (
            " Log-logistic has no zero-placement treatment for a series with a physical "
            "zero mass; use gamma or Pearson Type III here. SPEI is the supported "
            "log-logistic surface."
        )
    elif distribution is Distribution.beta:
        message += (
            " Beta fits a series bounded by [0, 1], with aggregation='mean'; "
            "standardized_index() is the supported beta surface."
        )
    elif distribution in (Distribution.t_location_scale, Distribution.normal):
        message += " It fits signed precipitation anomalies; spai() is its supported surface."
    raise InvalidArgumentError(
        message,
        argument_name="distribution",
        argument_value=str(distribution),
        valid_values=valid_values,
    )


def _reject_negative_values(values: np.ndarray, distribution: Distribution) -> np.ndarray:
    """Reject a signed series bound for a parametric fit, rather than clip it (ADR-0019)."""
    if bool(np.any(values < 0.0)):
        raise InvalidArgumentError(
            f"The values hold negatives, which a {distribution.display_name} fit does not support, and "
            "clip_negatives=False keeps them; clipping would standardize a different series. Use "
            "indices.Distribution.empirical for a signed series, or clip_negatives=True to clip it.",
            argument_name="clip_negatives",
            argument_value="False",
            valid_values="True, or distribution=Distribution.empirical",
        )
    return values


def _clip_fitted_values(values: np.ndarray, output_scale: compute.OutputScale) -> np.ndarray:
    """Clip z-scores to the supported range; the probability scales are already bounded."""
    if output_scale == "normal":
        clipped: np.ndarray = np.clip(values, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX)
        return clipped
    return values


def _validate_periodicity(periodicity: compute.Periodicity, *, allow_hourly: bool = False) -> None:
    """Validate that periodicity is a valid, supported Periodicity enum member.

    Args:
        periodicity: The periodicity parameter to validate
        allow_hourly: Whether the calling surface accepts hourly standardization

    Raises:
        PeriodicityError: If periodicity is not a Periodicity enum member
        InvalidArgumentError: If periodicity is hourly on a surface that does not
            standardize hourly data
    """
    if not isinstance(periodicity, compute.Periodicity):
        message = (  # type: ignore[unreachable]
            f"Invalid periodicity argument: {periodicity}. "
            f"Periodicity must be a Periodicity enum member. "
            f"Supported values: monthly, daily, hourly. "
            f"Use compute.Periodicity.monthly, compute.Periodicity.daily, or compute.Periodicity.hourly."
        )
        raise PeriodicityError(message, periodicity_value=str(periodicity))

    if periodicity is compute.Periodicity.hourly and not allow_hourly:
        raise InvalidArgumentError(
            "Hourly periodicity is supported only by the standardized indices with the empirical distribution.",
            argument_name="periodicity",
            argument_value=str(periodicity),
            valid_values="Periodicity.monthly, Periodicity.daily",
        )


def _raise_if_unsupported_shape(values: np.ndarray, spatial_time_major: bool = False) -> None:
    """Raise this module's `DataShapeError` for an unsupported input shape.

    `eddi` and `percentage_of_normal` pin their dimension errors to `DataShapeError`
    rather than to the `ValueError` the fitting-based kernels raise for an ambiguous
    block. Before this guard existed `percentage_of_normal` failed inside
    `np.convolve` with a numpy `ValueError` instead, so for that index the exception
    type is part of the 3.0.0 change rather than a pre-existing behavior.

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

    native = compute._native_hastings_inverse_normal(p)
    if native is not None:
        return native

    # work in the lower tail; flip if p > 0.5
    sign = np.where(p <= 0.5, -1.0, 1.0)
    p_lower = np.where(p <= 0.5, p, 1.0 - p)

    # Hastings rational approximation
    t = np.sqrt(-2.0 * np.log(p_lower))
    numerator = _HASTINGS_C0 + t * (_HASTINGS_C1 + t * _HASTINGS_C2)
    denominator = 1.0 + t * (_HASTINGS_D1 + t * (_HASTINGS_D2 + t * _HASTINGS_D3))
    z = sign * (t - numerator / denominator)

    return cast(np.ndarray, z)


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
            shaped (time, ``*cells``), ranks every cell in one pass when it is
            declared with ``spatial_time_major``.
        scale: Number of time steps over which PET values are accumulated
            before ranking. Must be in [1, 72] for monthly data or [1, 2196]
            for daily data.
        data_start_year: First year of the input PET dataset.
        calibration_year_initial: First year of the calibration period used
            for empirical ranking.
        calibration_year_final: Last year of the calibration period.
        periodicity: Temporal resolution of the input data. Use
            ``compute.Periodicity.monthly`` (12 values/year) or
            ``compute.Periodicity.daily`` (366 values/year).
        spatial_time_major: Read a three-or-more-dimensional ``pet_values`` as a
            time-major block of independent time series, shaped (time, ``*cells``),
            and rank every cell against its own climatology in one pass. The
            xarray adapter sets this for every block it packs; a direct NumPy
            caller has to declare it, since a block is not a shape EDDI reads
            without being told.

    Returns:
        1-D numpy array of EDDI values (unitless z-scores), same length as
        the input, or the same (time, ``*cells``) shape as a declared block. Values
        are clipped to [-3.09, 3.09].

    Raises:
        DataShapeError: If the input array has more than 2 dimensions and is not
            a declared time-major block.
        InvalidArgumentError: If scale, periodicity, or calibration years
            are invalid.
        InsufficientDataError: If scale exceeds the number of available time steps.
    """
    # validate arguments
    _validate_periodicity(periodicity)
    _validate_scale(scale, periodicity)

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

        # a calibration period the record does not cover is an error, not clamped. It is
        # resolved before the all-missing returns below, so a reversed or uncovered
        # window is rejected even when the input carries no values to rank.
        period = _resolve_unfolded_window(
            pet_values,
            periodicity,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            policy="reject",
        )

        # an all-missing block is returned as it arrived, as the preparation seam
        # does for the 1-D and 2-D layouts -- unless the scale exceeds the block's
        # time steps, which the seam rejects rather than returning silently
        if pet_values.ndim > 2 and scale <= pet_values.shape[0] and compute.is_all_missing(pet_values):
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

        num_years = pet_values.shape[0]

        # Rank each calendar period by counting climatology values below each value.
        # Rust counts per column without a comparison intermediate; the NumPy fallback
        # chunks the (climatology years, years, cells) comparison across cells to bound
        # that intermediate for wide blocks. Valid- and pad-counts are also held one
        # period at a time, rather than for the whole calibration block (366 periods
        # for daily data). Missing climatology values never compare below a value.
        climatology = pet_values[period.rows]
        cell_shape = pet_values.shape[2:]
        cells_per_time_step = int(np.prod(cell_shape, dtype=np.int64)) or 1
        num_climatology_years = climatology.shape[0]
        cells_per_chunk = max(1, _EDDI_RANK_COMPARISON_ELEMENT_BUDGET // (num_climatology_years * num_years))
        eddi_values = np.empty(pet_values.shape, dtype=float)
        below = np.empty((num_years, cells_per_time_step), dtype=np.int32)

        for period_index in range(num_periods):
            period_climatology = climatology[:, period_index].reshape(num_climatology_years, cells_per_time_step)
            period_values = pet_values[:, period_index].reshape(num_years, cells_per_time_step)
            period_pads = np.count_nonzero(
                leading_scale_pads[period.rows, period_index].reshape(num_climatology_years, cells_per_time_step),
                axis=0,
            )
            probabilities = compute._native_tukey_probabilities(period_climatology, period_values, period_pads)
            if probabilities is None:
                period_valid_counts = np.count_nonzero(~np.isnan(period_climatology), axis=0)
                for cell_start in range(0, cells_per_time_step, cells_per_chunk):
                    cell_chunk = slice(cell_start, cell_start + cells_per_chunk)
                    below[:, cell_chunk] = np.count_nonzero(
                        period_climatology[:, None, cell_chunk] < period_values[:, cell_chunk], axis=0
                    )

                # NOAA uses zero-based ranks and treats leading scale pads as lower than every
                # observed value; this is the Tukey plotting position of that rank
                probabilities = (period_pads + below + 0.66) / (period_valid_counts + period_pads + 0.33)

                # a period whose climatology holds fewer than two valid values has no ranking
                # at all, and a missing value stays missing
                probabilities = np.where(
                    np.isnan(period_values) | (period_valid_counts < 2),
                    np.nan,
                    probabilities,
                )

            # clip the probability to its valid range to avoid log(0), then apply the
            # Hastings inverse normal approximation. Both are elementwise and are held one
            # calendar period at a time: the approximation allocates several temporaries,
            # and over a whole wide block they would outweigh the input by an order of
            # magnitude (a daily block would be the worst case, with 366 periods).
            eddi_values[:, period_index] = _hastings_inverse_normal(
                np.clip(probabilities.reshape(num_years, *cell_shape), 1e-10, 1.0 - 1e-10),
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
        log_calculation_failure(log, exc, calibration_period=f"{calibration_year_initial}-{calibration_year_final}")
        raise


def _resolve_unfolded_window(
    values: np.ndarray,
    periodicity: compute.Periodicity,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    *,
    policy: CalibrationPolicy,
) -> CalibrationPeriod:
    """Resolve a Calibration Period against a series that is not folded into (years, periods) yet.

    An all-missing input is returned without being folded, so its window can only be
    checked from the time axis: a 1-D or 2-D series is measured by its length and a
    time-major block by its first axis. As when the series is folded, a trailing
    partial year counts as a year.
    """
    n_steps = values.shape[0] if values.ndim > 2 else values.size
    return resolve_calibration_period(
        data_start_year,
        -(-n_steps // periodicity.period_length),
        calibration_year_initial,
        calibration_year_final,
        policy=policy,
    )


def _standardized_index_pipeline(
    values: np.ndarray,  # NOSONAR (S107) the shared pipeline mirrors the public index wrappers
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: dict[str, Any] | None,
    *,
    index_type: str,
    fallback_context: str,
    memory_arrays: tuple[np.ndarray, ...] = (),
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    allowed_distributions: tuple[Distribution, ...] = (Distribution.gamma, Distribution.pearson),
    clip_negatives: bool = True,
    fallback_to_gamma: bool = True,
    preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    aggregation: compute.Aggregation = "sum",
    pool_window_days: int = 0,
) -> np.ndarray:
    """Scale, fit, and transform a series in the pipeline shared by the index wrappers.

    Args:
        values: 1-D array of values, or a time-major spatial block; see
            :func:`spi` for the accepted layouts. Negatives are clipped unless
            ``clip_negatives`` is False.
        scale: Number of time steps accumulated before fitting.
        distribution: Distribution to fit, one of ``allowed_distributions``.
        calibration_year_initial: Initial year of the calibration period.
        calibration_year_final: Final year of the calibration period.
        periodicity: Monthly or daily time steps.
        fitting_params: Optional pre-computed fitting parameters; deprecated
            aliases are normalized here.
        index_type: Value bound to the ``index_type`` log field.
        fallback_context: Context included in the Pearson-to-gamma fallback warning.
        memory_arrays: Additional input arrays counted alongside ``values`` in the
            large-array memory diagnostic; :func:`spei` passes PET, which the
            pipeline does not otherwise see before its water balance is formed.
        spatial_time_major: Read a time-major spatial block as independent series.
        output_scale: One of ``compute.OUTPUT_SCALES``; non-normal scales are not
            clipped to the z-score range.
        zero_handling: Where a zero accumulation is placed within the zero mass.
        allowed_distributions: The distribution members this surface accepts; the
            default omits log-logistic, which only :func:`spei` supports.
        clip_negatives: Whether preparation clips negative values to zero before
            scaling. :func:`spei` passes False because it clips precipitation and
            adds its +1000 offset before this pipeline.
        fallback_to_gamma: Whether a failed Pearson Type III fit falls back to
            gamma; :func:`spei` passes False so the failure propagates.
        preprocess: Optional index-specific transform applied to ``values`` inside
            the traced section before preparation. :func:`spei` uses it to form the
            P - PET + offset water balance, so an error it raises is logged with
            this pipeline's full failure context.
        aggregation: Whether each ``scale`` window is summed or averaged; only
            :func:`standardized_index` exposes it, every other index sums.
        pool_window_days: Neighbouring days pooled into each empirical hourly step;
            only :func:`standardized_index` exposes it, every other index keeps the
            shipped hour x day-of-year strata.

    Returns:
        Standardized values in the input's size and layout.
    """
    # validate arguments
    # only a surface that accepts the empirical distribution can standardize hourly data
    _validate_periodicity(periodicity, allow_hourly=Distribution.empirical in allowed_distributions)
    _validate_scale(scale, periodicity)
    _validate_distribution(distribution, allowed_distributions)
    _validate_hourly_distribution(periodicity, distribution)
    compute.validate_output_scale(output_scale)
    compute._validate_zero_handling(zero_handling)
    compute._validate_pool_window_days(pool_window_days, periodicity, distribution, scale=scale)

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type=index_type,
        scale=scale,
        distribution=distribution.value,
        output_scale=output_scale,
        input_shape=values.shape,
        input_elements=values.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(values, *memory_arrays)

    try:
        # the empirical check must see the raw mapping, because normalization below
        # drops unrecognized keys and would turn a parameter typo into a valid
        # parameter-free call
        compute._validate_empirical_fitting_params(distribution, fitting_params)

        # normalize any deprecated fitting-parameter aliases once, so the diagnostic
        # stays bounded per spatial operation
        fitting_params = compute._normalize_fitting_params(fitting_params)

        # an index with a derived input (SPEI's water balance) forms it here, inside
        # the traced section, so validation and the failure log stay in one place
        if preprocess is not None:
            values = preprocess(values)

        if distribution is Distribution.beta:
            compute._validate_beta_support(values)

        # prepare_scaled owns flatten/fold, the ambiguous-shape guard, the
        # insufficient-data check, negative clipping, scaling, and the all-missing
        # return. Shape errors raise the plain ValueError from prepare_scaled --
        # spi()'s dimension errors are pinned to ValueError by
        # tests/test_backward_compat.py::TestErrorHierarchyDocumented, unlike
        # eddi()/percentage_of_normal() which use DataShapeError.
        original_shape = values.shape
        # all-missing is decided on the input, before folding, so a partially missing
        # series whose scaled windows happen to be all NaN still reaches the fit and
        # the unfold instead of returning a folded block
        input_all_missing = compute.is_all_missing(values)
        values = compute.prepare_scaled(
            values,
            scale,
            periodicity,
            clip_negatives=clip_negatives,
            spatial_time_major=spatial_time_major,
            aggregation=aggregation,
        )

        # all-missing input comes back un-reshaped -- flattened for a 1-D/2-D series,
        # its input layout for a time-major block -- so there is nothing to compute
        # and the caller's layout is already what was returned
        if input_all_missing:
            # validation only: nothing is fitted, but a reversed window is an error
            # whatever the values, and a window the record does not cover still clamps
            _resolve_unfolded_window(
                values,
                periodicity,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                policy="clamp",
            )
            _log_calculation_completed(log, t0, values.shape, memory_metrics)
            return values

        # fit the scaled values to the specified distribution and transform to
        # corresponding normalized sigmas, falling back to gamma when a Pearson
        # Type III fit fails and the caller asked for the fall back
        values = compute.fit_and_standardize(
            values,
            distribution,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            periodicity,
            fitting_params,
            fallback_to_gamma=fallback_to_gamma,
            fallback_context=fallback_context,
            output_scale=output_scale,
            zero_handling=zero_handling,
            pool_window_days=pool_window_days,
            scale=scale,
        )

        # clip z-scores to the supported range, including any zeros a
        # zero-handling mode moved (ADR-0015, decision 6); probability-scale
        # outputs are already bounded by construction and must not be z-clipped
        values = _clip_fitted_values(values, output_scale)

        # one module owns the unfold back to the caller's layout
        result = compute.unfold_time_major(values, original_shape)
        _log_calculation_completed(log, t0, result.shape, memory_metrics)
        return result
    except Exception as exc:
        log_calculation_failure(log, exc, calibration_period=f"{calibration_year_initial}-{calibration_year_final}")
        raise


def standardized_index(
    values: np.ndarray,  # NOSONAR (S107) the public API exposes each standardization option as a keyword
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: dict[str, Any] | None = None,
    *,
    spatial_time_major: bool = False,
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
    aggregation: compute.Aggregation = "sum",
    clip_negatives: bool = True,
    pool_window_days: int = 0,
) -> np.ndarray:
    """Standardize a monthly or daily series against a fitted distribution.

    This is the generic form of :func:`spi`: it accumulates the series over ``scale``
    time steps (or averages it, see ``aggregation``), fits the specified distribution
    over the calibration period, and transforms every accumulated value to a normalized
    sigma (z-score-like) value.
    It is input-agnostic, so a runoff or streamflow series can be standardized the
    same way; the package does not model those quantities.

    A Pearson Type III fit falls back to gamma when the fit fails or loses too many of
    the input's valid values; input that was already missing does not count.
    Log-logistic (generalized logistic) fitting is available for :func:`spei` but is
    rejected here: a non-negative series has a physical zero mass that the GLO path
    does not yet place.

    ``Distribution.empirical`` standardizes by rank instead of a fit: each value's
    probability is its plotting position, ``(1 + k) / (n + 2)``, among the ``n`` valid
    calibration values at its calendar step, ``k`` of them at or below it (the SEI
    convention). A step with fewer than ten valid calibration values is missing, which
    for daily data with NaN at 29 February of non-leap years includes that day unless
    the Calibration Period spans ten leap years. It is the only distribution that
    accepts a signed series, with
    ``clip_negatives=False`` (ADR-0019).

    The beta distribution is the bounded [0, 1] family. Its shape parameters come from
    the method of moments over each calendar step's interior values, and the exact 0
    and 1 values are point masses at the boundaries, so a capacity factor or another
    bounded series standardizes without leaving its own range (ADR-0024). Pair it with
    ``aggregation="mean"``, which is what keeps a windowed value inside the support.
    A source observation or windowed value outside [0, 1] is rejected rather than
    clipped, whatever ``clip_negatives`` asks for.

    This is a NumPy-array entry point only; xarray and Dask dispatch are not wired
    for it.

    Args:
        values: 1-D array of values, in any units, non-negative unless
            ``clip_negatives`` is False; the first value is assumed to correspond to
            the start of ``data_start_year``. A 2-D array
            is read as the legacy (years, periods) layout and flattened, and a
            time-major spatial block with more than two dimensions is accepted
            when declared with ``spatial_time_major``.
        scale: Number of time steps over which the values are accumulated before
            the index is computed.
        distribution: Distribution type used for the internal fitting/transform
            computation: gamma, Pearson Type III, empirical, or beta.
        data_start_year: Initial year of the input series.
        calibration_year_initial: Initial year of the Calibration Period.
        calibration_year_final: Final year of the Calibration Period.
        periodicity: Periodicity of the series; ``compute.Periodicity.monthly`` for
            monthly data (12 values/year), ``compute.Periodicity.daily`` for daily
            data (366 values/year), or ``compute.Periodicity.hourly`` for hourly data
            (8784 values/year), which supports only ``Distribution.empirical``.
        fitting_params: Optional dictionary of pre-computed distribution fitting
            parameters, with keys "alpha" and "beta" (and optionally "prob_zero",
            plus "prob_one" when fitting to beta) for gamma and beta, and
            "prob_zero", "loc", "scale", and "skew" for Pearson Type
            III. A gamma set without "prob_zero" computes it over the calibration
            period of the values given. Older keys such as "alphas" and
            "probabilities_of_zero" are deprecated. The empirical distribution has
            no parameters, so it takes none.
        spatial_time_major: Read a three-or-more-dimensional time-major block of
            independent time series, shaped (time, ``*cells``), and fit every cell
            in one pass. It is required only for the ambiguous shape whose first
            cell axis is a calendar period length (12 or 366); see :func:`spi` for
            the layout. A 2-D block is always read as one series, however
            declared.
        output_scale: Output convention, one of ``compute.OUTPUT_SCALES``.
            "normal" (the default) returns the standard-normal z-score,
            "probability" returns the fitted cumulative probability in [0, 1]
            without clipping, and "bounded" returns ``2p - 1`` in [-1, 1].
        zero_handling: Where a zero accumulation is placed within the zero mass;
            see :func:`spi`. The empirical distribution has no zero mass, so it
            ignores the mode, and the beta distribution applies the mode to both of
            its boundary masses, placing a one at the mirror of a zero's position
            (ADR-0015, ADR-0024).
        aggregation: How each ``scale`` window is combined before the fit: "sum"
            (the default, and what :func:`spi` does) accumulates it, "mean" averages
            it, which keeps a bounded series such as a capacity factor in its own
            range. A window holding a missing value is missing under either. The
            default leaves every existing result unchanged (ADR-0019).
        clip_negatives: Whether negative values are clipped to zero before
            scaling, the default and what :func:`spi` does. Pass False to keep a
            signed series, such as residual load, with ``Distribution.empirical``;
            gamma and Pearson Type III reject a series with negatives then rather
            than clip it (ADR-0019). The beta distribution never clips: its support
            check rejects a negative, because clipping would count it in the zero
            mass and standardize a different series (ADR-0024).
        pool_window_days: Number of neighbouring days on each side of an hourly
            empirical step whose values join its calibration sample, zero (the
            default) keeping the shipped hour x day-of-year strata. A positive value
            is a smoothed seasonal-diurnal climatology that widens the sample by whole
            days at the same hour and holds 29 February out of every pooled sample. It
            is rejected for any other periodicity or distribution (#1407). The maximum
            is 182 days, so each calendar day enters a sample at most once. A ``scale``
            above one also sets a minimum of ``ceil(held_out_days / 2)`` days, where
            ``held_out_days = ceil((23 + scale) / 24)`` (16 days at ``scale=720``), so
            the 29 February holdout cannot empty a pooled sample; from ``scale=8714``
            no window is accepted.

    Returns:
        1-D array of standardized values, unitless and of the same length as the
        flattened input; a declared time-major block (three or more dimensions) is
        returned in its input shape.

    Raises:
        InvalidArgumentError: If ``clip_negatives`` is False and a gamma or Pearson
            Type III fit is given negative values, if ``fitting_params`` is given
            with the empirical distribution, if ``distribution`` is a member this
            surface does not accept, or if a beta fit is given a value outside
            [0, 1], or if ``pool_window_days`` is not an integer from 0 through 182
            or a positive value is paired with a non-hourly or non-empirical call or
            with a ``scale`` that leaves the window below that minimum.
        ValueError: If ``zero_handling`` is not one of the three modes, if
            ``aggregation`` is not "sum" or "mean", a Pearson ``fitting_params``
            set is partial or does not carry the period (and cell) axes, or a beta
            mass pair sums above 1 or a supplied beta shape is not finite and positive
            (unless NaN).
    """
    # the pipeline validates the distribution before it runs this check. Beta is not
    # among the families this pre-reject covers: its own support check reports the
    # [0, 1] bound, and it never clips, so a negative always raises there.
    reject_negatives = not clip_negatives and distribution in (Distribution.gamma, Distribution.pearson)
    return _standardized_index_pipeline(
        values,
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        fitting_params,
        index_type="standardized_index",
        fallback_context="standardized index computation",
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
        zero_handling=zero_handling,
        aggregation=aggregation,
        allowed_distributions=(Distribution.gamma, Distribution.pearson, Distribution.empirical, Distribution.beta),
        # beta must see the raw values: clipping a negative to a zero would count it in
        # the zero mass and standardize a different series, so the beta fit's support
        # check raises instead (ADR-0019, decision 3; ADR-0024, decision 4)
        clip_negatives=clip_negatives and distribution is not Distribution.beta,
        preprocess=(lambda series: _reject_negative_values(series, distribution)) if reject_negatives else None,
        pool_window_days=pool_window_days,
    )


def fit_diagnostics(
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
    aggregation: compute.Aggregation = "sum",
) -> compute.FitDiagnostics:
    """Scale a series, fit a distribution, and return per-calendar-step fit diagnostics.

    This is :func:`standardized_index` with the audit surface instead of the index
    values: it sums or averages the series over ``scale`` time steps, fits ``distribution``
    over the calibration period, and returns the fitted parameters, the probability of
    zero, the number of valid calibration values, and the Kolmogorov-Smirnov D
    statistic and exact p-value for every calendar step. A Pearson Type III fit falls
    back to gamma under the same policy as ``standardized_index``, and the fall back is
    reported rather than applied silently.

    Only the NumPy path is supported; the xarray and Dask surface is tracked
    separately. A fitted parameter set returned here can be passed back as
    ``fitting_params`` to ``standardized_index`` with the same ``aggregation`` to
    reproduce a fit, or to ``spi`` when using the default sum.

    Args:
        values: 1-D array of non-negative values, or a time-major spatial block;
            see :func:`spi` for the accepted layouts.
        scale: Number of time steps accumulated before fitting.
        distribution: Distribution to fit, gamma or Pearson Type III; the
            log-logistic (generalized logistic) and beta fits are rejected on this
            surface.
        data_start_year: Initial year of the input values.
        calibration_year_initial: Initial year of the calibration period.
        calibration_year_final: Final year of the calibration period.
        periodicity: Monthly or daily time steps.
        fitting_params: Optional pre-computed fitting parameters; deprecated
            aliases are normalized here. A parameter left as None is fitted from
            the data.
        spatial_time_major: Read a time-major spatial block as independent series.
        aggregation: How each window is combined, "sum" (default) or "mean";
            must match the aggregation used when reusing the fitted parameters.

    Returns:
        A :class:`climate_indices.compute.FitDiagnostics` whose arrays carry the
        scaled series' calendar steps.
    """
    # validate arguments
    _validate_periodicity(periodicity)
    _validate_scale(scale, periodicity)
    _validate_distribution(distribution, (Distribution.gamma, Distribution.pearson))

    log = _logger.bind(
        index_type="fit_diagnostics",
        scale=scale,
        distribution=distribution.value,
        input_shape=values.shape,
        input_elements=values.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(values)

    try:
        fitting_params = compute._normalize_fitting_params(fitting_params)

        # prepare_scaled owns the ambiguous-shape rejection, so no separate guard here;
        # an all-missing time-major block comes back unfolded, keeping the input shape,
        # so fold it before fitting so the diagnostic arrays carry (time_steps, *cells)
        # rather than the time axis
        input_all_missing = values.ndim > 2 and compute.is_all_missing(values)
        scaled_values = compute.prepare_scaled(
            values, scale, periodicity, spatial_time_major=spatial_time_major, aggregation=aggregation
        )
        if input_all_missing:
            scaled_values = compute.reshape_time_major(scaled_values, periodicity)

        diagnostics = compute.fit_diagnostics(
            scaled_values,
            distribution,
            data_start_year,
            calibration_year_initial,
            calibration_year_final,
            periodicity,
            fitting_params,
            fallback_to_gamma=True,
            fallback_context="fit diagnostics",
        )

        _log_calculation_completed(log, t0, diagnostics.ks_statistic.shape, memory_metrics)
        return diagnostics
    except Exception as exc:
        log_calculation_failure(log, exc, calibration_period=f"{calibration_year_initial}-{calibration_year_final}")
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
    output_scale: compute.OutputScale = "normal",
    zero_handling: compute.ZeroHandling = "classic",
) -> np.ndarray:
    """
    Computes SPI (Standardized Precipitation Index).

    :param values: 1-D numpy array of precipitation values, in any units,
        first value assumed to correspond to January of the initial year if
        the periodicity is monthly, or January 1st of the initial year if daily.
        A time-major spatial array with shape (time, ``*cells``), i.e. three or more
        dimensions, is also accepted, and then every cell is scaled and fitted in
        one pass.
        Two-dimensional input is still read as the legacy (years, periods) layout
        and flattened into a single series, not treated as a (time, cells) grid.
        When the first cell axis is a calendar period length (12 or 366) the shape is
        equally readable as a (years, periods, ``*cells``) array, and then the reading has to
        be declared with ``spatial_time_major``; the xarray adapter declares every block
        it packs, and only that ambiguous shape raises without a declaration.
    :param scale: number of time steps over which the values should be scaled
        before the index is computed
    :param distribution: distribution type to be used for the internal
        fitting/transform computation: gamma or Pearson Type III. The
        log-logistic (generalized logistic) fit is rejected here, since it has no
        zero-placement treatment for the precipitation series; use :func:`spei`.
    :param data_start_year: the initial year of the input precipitation dataset
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param periodicity: periodicity of the input time series; use
        ``compute.Periodicity.monthly`` for monthly data (12 values/year) or
        ``compute.Periodicity.daily`` for daily data (366 values/year).
    :param fitting_params: optional dictionary of pre-computed distribution
        fitting parameters, if the distribution is gamma then this dict should
        contain two arrays, keyed as "alpha" and "beta", and optionally a third
        keyed as "prob_zero", and if the distribution is Pearson then this dict
        should contain four arrays keyed as "prob_zero", "loc", "scale", and
        "skew". A gamma set without "prob_zero" computes it over the calibration
        period of the values given. Older keys such as
        "alphas" and "probabilities_of_zero" are deprecated. For spatial input a 1-D
        parameter array is read as one value per calendar period and broadcast
        across cells.
    :param spatial_time_major: read ``values`` as a time-major block of independent
        time series, shaped (time, ``*cells``), and fit every cell in one pass. The
        xarray adapter sets this for every block it packs; the NumPy API requires
        it only for an ambiguous shape, where the first cell axis is a calendar
        period length (12 or 366) and could be read as (years, periods, ``*cells``).
    :param output_scale: output convention, one of ``compute.OUTPUT_SCALES``.
        "normal" (the default) returns the standard-normal z-score,
        "probability" returns the fitted cumulative probability in [0, 1]
        without clipping, and "bounded" returns ``2p - 1`` in [-1, 1].
    :param zero_handling: where a zero accumulation is placed within the
        probability mass ``p0`` at zero: ``"classic"`` (the default) scores it
        ``Φ⁻¹(p0)``, the top of the zero mass, as NOAA/NCEI and SPEIbase do,
        unless a Pearson Type III support-limit mask overrides that score;
        ``"center_of_mass"`` scores it ``Φ⁻¹(p0 / 2)`` (Stagge et al., 2015); and
        ``"mean_zero"`` scores it ``−φ(Φ⁻¹(p0)) / p0`` (Allen and Otero, 2024), the
        conditional mean of the zero mass. Only steps whose effective ``p0``
        satisfies ``0 < p0 < 1`` are moved: a Pearson step with fewer than four
        non-zero calibration values has its ``p0`` reset to 0, so its zeros keep
        the classic score (-3.09 on the normal scale) in every mode. Pearson moves
        trace values below 0.0005 with the zeros although ``p0`` counts only exact
        zeros, so the modes' mean properties are approximate there. A
        normal-scale result is still clipped to [-3.09, 3.09]. On the probability
        and bounded scales both non-classic modes place a zero at ``p0 / 2``.
        See ADR-0015.
    :return: SPI values fitted to the gamma distribution at the specified time
        step scale, unitless
    :rtype: 1-D numpy.ndarray of floats of the same length as the input array
        of precipitation values, or of the same (time, ``*cells``) shape when
        ``spatial_time_major`` is set
    :raises ValueError: if ``zero_handling`` is not one of the three modes, or a
        Pearson ``fitting_params`` set is partial or does not carry the period (and cell) axes
    """
    return _standardized_index_pipeline(
        values,
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        fitting_params,
        index_type="spi",
        fallback_context="SPI computation",
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
        zero_handling=zero_handling,
    )


def _spei_water_balance(precips_mm: np.ndarray, pet_mm: np.ndarray) -> np.ndarray:
    """Form the P - PET + 1000 water balance that SPEI standardizes.

    A single PET time series is one series for every cell, so it is given singleton
    cell axes to broadcast across a spatial block. The two arrays must broadcast
    together: a spatial block needs matching time lengths and cell axes, while the
    series path keeps its size-based check. Negative precipitation is clipped to
    zero before the subtraction, and the offset keeps the result positive.
    """
    if precips_mm.ndim > 2 and pet_mm.ndim == 1 and pet_mm.size == precips_mm.shape[0]:
        pet_mm = pet_mm.reshape((pet_mm.shape[0],) + (1,) * (precips_mm.ndim - 1))

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

    # clip any negative precipitation to zero; np.any(...) is NaN-safe, unlike np.amin
    if bool(np.any(precips_mm < 0.0)):
        _logger.warning("Input contains negative values -- all negatives clipped to zero")
        precips_mm = np.clip(precips_mm, a_min=0.0, a_max=None)

    if precips_mm.ndim > 2:
        spatial_balance: np.ndarray = (precips_mm - pet_mm) + 1000.0
        return spatial_balance
    series_balance: np.ndarray = (precips_mm.flatten() - pet_mm.flatten()) + 1000.0
    return series_balance


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
    output_scale: compute.OutputScale = "normal",
) -> np.ndarray:
    """
    Compute SPEI fitted to the specified distribution.

    PET values are subtracted from the precipitation values to come up with an array
    of (P - PET) values, which is then scaled to the specified timescale and
    finally fitted/transformed to SPEI values corresponding to the input
    precipitation time series.

    :param precips_mm: an array of precipitation values,
        in millimeters, should be of the same size (and shape?) as the input PET array.
        A time-major spatial array with shape (time, ``*cells``), i.e. three or more
        dimensions, is also accepted, and then every cell is scaled and fitted in
        one pass.
        Two-dimensional input is still read as the legacy (years, periods) layout
        and flattened into a single series, not treated as a (time, cells) grid.
        When the first cell axis is a calendar period length (12 or 366) the shape is
        equally readable as a (years, periods, ``*cells``) array, and then the reading has to
        be declared with ``spatial_time_major``; the xarray adapter declares every block
        it packs, and only that ambiguous shape raises without a declaration.
    :param pet_mm: an array of PET values, in millimeters, of the same size
        (and shape) as the input precipitation array
    :param scale: the number of time steps over which the values should be
        scaled before computing the indicator (months for monthly data, days
        for daily data)
    :param distribution: distribution type to be used for the internal
        fitting/transform computation: gamma, Pearson Type III, or loglogistic
        (the generalized logistic that SPEIbase standardizes with; see ADR-0016)
    :param periodicity: periodicity of the input time series; use
        ``compute.Periodicity.monthly`` for monthly data (12 values/year) or
        ``compute.Periodicity.daily`` for daily data (366 values/year).
    :param data_start_year: the initial year of the input datasets (assumes that
        the two inputs cover the same period)
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param fitting_params: optional dictionary of pre-computed distribution
        fitting parameters, if the distribution is gamma then this dict should
        contain two arrays, keyed as "alpha" and "beta", and optionally a third
        keyed as "prob_zero", if the distribution is Pearson then this dict
        should contain four arrays keyed as "prob_zero", "loc", "scale", and
        "skew", and if the distribution is loglogistic then this dict should
        contain three arrays keyed as "loc", "scale", and "shape". A gamma set
        without "prob_zero" computes it over the calibration period of the values
        given.
        Older keys such as "alphas" and "probabilities_of_zero" are deprecated.
    :param spatial_time_major: read ``precips_mm``/``pet_mm`` as time-major blocks of
        independent time series, shaped (time, ``*cells``), and fit every cell in one pass.
        The xarray adapter sets this for every block it packs; the NumPy API requires
        it only for an ambiguous shape, where the first cell axis is a calendar
        period length (12 or 366) and could be read as (years, periods, ``*cells``).
    :param output_scale: output convention, one of ``compute.OUTPUT_SCALES``.
        "normal" (the default) returns the standard-normal z-score,
        "probability" returns the fitted cumulative probability in [0, 1]
        without clipping, and "bounded" returns ``2p - 1`` in [-1, 1].
    :raises ValueError: if the precipitation and PET arrays are incompatible, or a
        Pearson ``fitting_params`` set is partial or does not carry the period (and cell) axes
    :return: an array of SPEI values
    :rtype: numpy.ndarray of type float; 1-D for a 1-D or 2-D input (the input is
        flattened), or the input's ``(time, *cells)`` shape for a declared time-major
        block
    """
    # the shared pipeline owns validation, logging, the ambiguous-shape guard,
    # scaling, fitting, clipping and the unfold back to the input layout; the SPEI-only
    # water balance is formed inside its traced section
    return _standardized_index_pipeline(
        precips_mm,
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        fitting_params,
        index_type="spei",
        fallback_context="SPEI computation",
        memory_arrays=(pet_mm,),
        preprocess=lambda precipitation: _spei_water_balance(precipitation, pet_mm),
        spatial_time_major=spatial_time_major,
        allowed_distributions=(Distribution.gamma, Distribution.pearson, Distribution.loglogistic),
        output_scale=output_scale,
        clip_negatives=False,
        fallback_to_gamma=False,
    )


# SPAI subtracts each calendar period's calibration mean and fits one distribution to
# the anomalies of every period pooled (#1480); gamma's positive support cannot hold a
# signed anomaly
_SPAI_DISTRIBUTIONS = (Distribution.t_location_scale, Distribution.pearson, Distribution.normal)

# the parameters each SPAI distribution fits, one value per cell
_SPAI_PARAMETER_KEYS: dict[Distribution, tuple[str, ...]] = {
    Distribution.t_location_scale: ("df", "loc", "scale"),
    Distribution.pearson: ("loc", "scale", "skew"),
    Distribution.normal: ("loc", "scale"),
}

# one calibration value per calendar period leaves every anomaly exactly zero, so the
# pooled fit has no scale to estimate; SPAI needs at least two calibration years
_SPAI_MIN_CALIBRATION_YEARS = 2

# the fewest valid pooled calibration anomalies a cell needs before it is fitted
_SPAI_MIN_FIT_VALUES = lmoments.MIN_VALUES_FOR_LMOMENTS

# keeps the inverse normal finite; a normal-scale result is then clipped like SPI's
_SPAI_PROBABILITY_EPSILON = float(np.finfo(float).eps)

# the t location-scale fit stops once no parameter (location, scale, log degrees of
# freedom) moves more than the tolerance in an iteration, or at the cap; the upper
# degrees-of-freedom bound is a normal distribution for every practical purpose
_T_FIT_TOLERANCE = 1e-6
_T_FIT_MAX_ITERATIONS = 200
_T_FIT_INITIAL_DF = 10.0
_T_FIT_LOG_DF_BOUNDS = (float(np.log(0.2)), float(np.log(1e4)))

_SPAI_BETA_MESSAGE = "SPAI is beta: its interface may change in a minor release (#1480)."


def _spai_period_means(calibration: np.ndarray) -> np.ndarray:
    """Each calendar period's calibration mean.

    ``calibration`` is shaped (years, periods, cells). A period with no valid
    calibration value has a NaN mean, so that period's anomalies are missing.
    """
    valid = ~np.isnan(calibration)
    counts = np.count_nonzero(valid, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means: np.ndarray = np.where(valid, calibration, 0.0).sum(axis=0) / counts
    return means


def _fit_normal(pooled: np.ndarray) -> dict[str, np.ndarray]:
    """Maximum-likelihood normal parameters of every column of ``pooled`` (samples, cells)."""
    valid = ~np.isnan(pooled)
    counts = np.count_nonzero(valid, axis=0)
    loc = np.where(valid, pooled, 0.0).sum(axis=0) / counts
    scale = np.sqrt((np.where(valid, pooled - loc, 0.0) ** 2).sum(axis=0) / counts)
    return {"loc": loc, "scale": scale}


def _fit_t_location_scale(pooled: np.ndarray) -> dict[str, np.ndarray]:
    """Maximum-likelihood t location-scale parameters of every column of ``pooled``.

    A vectorized ECME iteration (Liu and Rubin, 1995) over every column at once: the EM
    updates of location and scale, then one Newton step on the log degrees of freedom
    of the observed log-likelihood. There is no loop over cells, unlike a per-cell
    ``scipy.stats.t.fit``; #1480 records the benchmark against it. Each column stops
    at its own convergence, so its fit does not depend on the other columns.
    """
    valid = ~np.isnan(pooled)
    counts = np.count_nonzero(valid, axis=0)
    values = np.where(valid, pooled, 0.0)
    start = _fit_normal(pooled)
    loc, variance = start["loc"], start["scale"] ** 2
    log_df = np.full(loc.shape, np.log(_T_FIT_INITIAL_DF))
    active = np.ones(loc.shape, dtype=bool)
    # shortcut: linear EM convergence, ~150-200 iterations on monsoon-like series; a joint
    # Newton step on all three parameters would cut that if gridded runs prove too slow
    for _ in range(_T_FIT_MAX_ITERATIONS):
        df = np.exp(log_df)
        weights = np.where(valid, (df + 1.0) / (df + (values - loc) ** 2 / variance), 0.0)
        new_loc = (weights * values).sum(axis=0) / weights.sum(axis=0)
        residuals = np.where(valid, values - new_loc, 0.0)
        new_variance = (weights * residuals**2).sum(axis=0) / counts

        # derivatives of the observed log-likelihood in df at the new location and scale
        squared = residuals**2 / new_variance
        ratio = squared / (df + squared)
        half = 0.5 * (df + 1.0) / df
        gradient = (
            0.5 * counts * (scipy.special.digamma((df + 1.0) / 2.0) - scipy.special.digamma(df / 2.0) - 1.0 / df)
            - 0.5 * np.log1p(squared / df).sum(axis=0)
            + half * ratio.sum(axis=0)
        )
        curvature = (
            0.25 * counts * (scipy.special.polygamma(1, (df + 1.0) / 2.0) - scipy.special.polygamma(1, df / 2.0))
            + 0.5 * counts / df**2
            + 0.5 * ratio.sum(axis=0) * (df - 1.0) / df**2
            - half * (squared / (df + squared) ** 2).sum(axis=0)
        )

        # a Newton step on log df where the log-likelihood is concave there, otherwise a
        # unit step uphill; at most one e-fold, so a flat likelihood cannot overshoot
        log_gradient = df * gradient
        log_curvature = df**2 * curvature + log_gradient
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(log_curvature < 0.0, -log_gradient / log_curvature, np.sign(log_gradient))
        new_log_df = np.clip(log_df + np.clip(step, -1.0, 1.0), *_T_FIT_LOG_DF_BOUNDS)

        change = np.maximum.reduce(
            [
                np.abs(new_loc - loc),
                np.abs(np.sqrt(new_variance) - np.sqrt(variance)),
                np.abs(new_log_df - log_df),
            ]
        )
        # a converged column keeps its parameters while the others iterate on
        loc = np.where(active, new_loc, loc)
        variance = np.where(active, new_variance, variance)
        log_df = np.where(active, new_log_df, log_df)
        active &= change >= _T_FIT_TOLERANCE
        if not active.any():
            break
    return {"df": np.exp(log_df), "loc": loc, "scale": np.sqrt(variance)}


def _fit_pearson_or_normal(pooled: np.ndarray) -> dict[str, np.ndarray]:
    """Pearson Type III L-moment parameters of every column, normal where the fit fails.

    A column whose L-moments are invalid falls back to the normal distribution, the
    zero-skew Pearson Type III, and the fall back is logged as a ``distribution_fallback``
    warning event.
    """
    locs, scales, skews, fitted = lmoments.fit_spatial(pooled)
    if not fitted.all():
        normal = _fit_normal(pooled)
        locs = np.where(fitted, locs, normal["loc"])
        scales = np.where(fitted, scales, normal["scale"])
        skews = np.where(fitted, skews, 0.0)
        _logger.bind(
            from_distribution="pearson",
            to_distribution="normal",
            reason="invalid L-moments",
            context="SPAI computation",
            failure_count=int(np.count_nonzero(~fitted)),
            total_count=int(fitted.size),
        ).warning("distribution_fallback")
    return {"loc": locs, "scale": scales, "skew": skews}


_SPAI_FITS: dict[Distribution, Callable[[np.ndarray], dict[str, np.ndarray]]] = {
    Distribution.t_location_scale: _fit_t_location_scale,
    Distribution.pearson: _fit_pearson_or_normal,
    Distribution.normal: _fit_normal,
}


def _spai_fit(pooled: np.ndarray, distribution: Distribution) -> dict[str, np.ndarray]:
    """Fit ``distribution`` to each cell's pooled calibration anomalies, NaN where too few."""
    fittable = np.count_nonzero(~np.isnan(pooled), axis=0) >= _SPAI_MIN_FIT_VALUES
    parameters = {key: np.full(pooled.shape[1], np.nan) for key in _SPAI_PARAMETER_KEYS[distribution]}
    if fittable.any():
        # column-major, so every column sum runs over contiguous memory in the same order
        # whatever the number of columns, and a cell's fit does not depend on its block
        for key, fitted in _SPAI_FITS[distribution](np.asfortranarray(pooled[:, fittable])).items():
            parameters[key][fittable] = fitted
    return parameters


def _spai_cdf(anomalies: np.ndarray, distribution: Distribution, parameters: dict[str, np.ndarray]) -> np.ndarray:
    """The fitted cumulative probability of every anomaly, NaN where a parameter is."""
    loc, scale = parameters["loc"], parameters["scale"]
    if distribution is Distribution.t_location_scale:
        return cast(np.ndarray, scipy.stats.t.cdf(anomalies, parameters["df"], loc=loc, scale=scale))
    if distribution is Distribution.pearson:
        return cast(np.ndarray, scipy.stats.pearson3.cdf(anomalies, parameters["skew"], loc=loc, scale=scale))
    return cast(np.ndarray, scipy.stats.norm.cdf(anomalies, loc=loc, scale=scale))


def _resolve_spai_parameters(
    fitting_params: dict[str, Any] | None,
    distribution: Distribution,
    periods: int,
    cells_shape: tuple[int, ...],
) -> dict[str, np.ndarray] | None:
    """Broadcast a caller's complete SPAI parameter set to (periods, cells) and (cells,).

    Returns None when there is nothing to reuse, so every parameter is fitted.
    """
    if fitting_params is None:
        return None
    required = ("mean", *_SPAI_PARAMETER_KEYS[distribution])
    missing = [key for key in required if fitting_params.get(key) is None]
    if missing:
        raise InvalidArgumentError(
            f"SPAI fitting_params for the {distribution.display_name} distribution must carry every "
            f"parameter; missing: {', '.join(missing)}. Pass None to fit them all.",
            argument_name="fitting_params",
            argument_value=", ".join(sorted(fitting_params)),
            valid_values=", ".join(required),
        )
    cell_count = int(np.prod(cells_shape, dtype=np.intp))
    resolved: dict[str, np.ndarray] = {}
    for key in required:
        array = np.asarray(fitting_params[key], dtype=float)
        per_period = key == "mean"
        if per_period and array.ndim == 1:
            # one value per calendar period, broadcast across the cells
            array = array.reshape(-1, *([1] * len(cells_shape)))
        try:
            broadcast = np.broadcast_to(array, (periods, *cells_shape) if per_period else cells_shape)
        except ValueError as error:
            raise InvalidArgumentError(
                f"SPAI fitting_params key {key!r} does not broadcast: 'mean' needs one value per "
                f"calendar period ({periods}), and the distribution parameters one per cell {cells_shape}.",
                argument_name="fitting_params",
                argument_value=f"{key}: shape {array.shape}",
            ) from error
        resolved[key] = broadcast.reshape(periods, cell_count) if per_period else broadcast.reshape(cell_count)
    return resolved


def _spai(
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
    output_scale: compute.OutputScale = "normal",
) -> np.ndarray:
    """:func:`spai` without its warning, so the xarray route warns once per call, not per block."""
    _validate_periodicity(periodicity)
    _validate_scale(scale, periodicity)
    _validate_distribution(distribution, _SPAI_DISTRIBUTIONS)
    compute.validate_output_scale(output_scale)

    log = _logger.bind(
        index_type="spai",
        scale=scale,
        distribution=distribution.value,
        output_scale=output_scale,
        input_shape=values.shape,
        input_elements=values.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(values)

    try:
        original_shape = values.shape
        input_all_missing = compute.is_all_missing(values)
        # the accumulation SPI uses: (years, periods) for a series, (years, periods, *cells)
        # for a time-major block, and the unfolded input when everything is missing
        scaled = compute.prepare_scaled(values, scale, periodicity, spatial_time_major=spatial_time_major)
        if input_all_missing:
            _resolve_unfolded_window(
                scaled,
                periodicity,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                policy="clamp",
            )
            _log_calculation_completed(log, t0, scaled.shape, memory_metrics)
            return scaled

        years, periods, *cells_shape = scaled.shape
        folded = scaled.reshape(years, periods, -1)
        period = resolve_calibration_period(
            data_start_year, years, calibration_year_initial, calibration_year_final, policy="clamp"
        )
        if period.n_years < _SPAI_MIN_CALIBRATION_YEARS:
            raise InsufficientDataError(
                f"SPAI needs a Calibration Period of at least {_SPAI_MIN_CALIBRATION_YEARS} years: one "
                f"calibration value per calendar period leaves every anomaly zero. "
                f"{period.start_year}-{period.end_year} spans {period.n_years}.",
                required_count=_SPAI_MIN_CALIBRATION_YEARS,
            )
        calibration = folded[period.rows]
        compute._check_calibration_data_quality(
            calibration, period.start_year, period.end_year, (calibration_year_initial, calibration_year_final)
        )

        parameters = _resolve_spai_parameters(fitting_params, distribution, periods, tuple(cells_shape))
        means = _spai_period_means(calibration) if parameters is None else parameters["mean"]
        anomalies = folded - means
        if parameters is None:
            # one fit per cell over every calendar period of the Calibration Period
            parameters = _spai_fit(anomalies[period.rows].reshape(-1, folded.shape[2]), distribution)
        probabilities = _spai_cdf(anomalies, distribution, parameters)

        result = compute._map_non_normal_scale(probabilities, output_scale)
        if result is None:
            bounded = np.clip(probabilities, _SPAI_PROBABILITY_EPSILON, 1.0 - _SPAI_PROBABILITY_EPSILON)
            result = _clip_fitted_values(scipy.stats.norm.ppf(bounded), output_scale)
        result = compute.unfold_time_major(result.reshape(years, periods, *cells_shape), original_shape)
        _log_calculation_completed(log, t0, result.shape, memory_metrics)
        return result
    except Exception as exc:
        log_calculation_failure(log, exc, calibration_period=f"{calibration_year_initial}-{calibration_year_final}")
        raise


def spai(
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
    output_scale: compute.OutputScale = "normal",
) -> np.ndarray:
    """Compute SPAI, the Standardized Precipitation Anomaly Index (Chanda and Maity, 2015).

    Chanda and Maity propose SPAI for meteorological drought where precipitation is
    strongly seasonal and periodic, as in monsoon-dominated climates. It standardizes the
    anomalies of precipitation rather than the raw precipitation series, and they compare
    it with SPI. Where :func:`spi` fits one distribution per calendar period, SPAI
    subtracts each calendar period's mean and fits one distribution to the anomalies of
    every period pooled together:

    1. ``P`` is ``values`` accumulated over ``scale`` time steps, as for :func:`spi`, so
       the first ``scale - 1`` steps are NaN.
    2. ``mean_i`` of each calendar period ``i`` (month, or day of the 366-day year) is
       taken over the Calibration Period only.
    3. ``Z = P - mean_i`` over the whole series: the anomaly, in the input's units.
    4. One distribution ``F`` is fitted to each cell's pooled Calibration Period ``Z``.
    5. ``SPAI = Φ⁻¹(F(Z))``, with ``F(Z)`` clipped to ``[ε, 1 - ε]`` (machine ``ε``) so
       the inverse normal stays finite, and the result clipped to [-3.09, 3.09] as for
       :func:`spi` and :func:`spei`.

    The anomaly is the calendar-period mean subtraction alone, the original paper's
    Eq. (5) as the authors' closure reproduces it: ``y_ij = x_ij - x̄_j``. The pooled
    fit's own scale therefore supplies the normalization. Because the fit is pooled, the
    result is near standard normal only over the Calibration Period as a whole, not
    within each calendar period, and only as far as ``F`` describes the pooled
    anomalies: the symmetric t location-scale and normal families cannot follow strongly
    skewed anomalies, such as those of months that rarely rain. A given anomaly value
    scores the same in a dry and a wet month, so a deficit that is small in absolute
    terms does not register as drought whatever the month's own spread; that is the
    published design, and the same authors' later papers instead divide the anomaly by
    each calendar period's standard deviation. The article body is paywalled, so Eq. (5)
    is taken from the authors' closure, which reproduces the equation, and from
    accessible secondary sources; contacting the authors remains open. See the SPAI
    section of ``VALIDATION.md`` (#1480).

    The distributions, each fitted to the pooled ``Z``:

    - ``Distribution.t_location_scale``, the reference choice for this index:
      ``F(Z) = T_ν((Z - μ) / σ)``, Student's t with ``ν`` degrees of freedom, fitted by
      maximum likelihood with a vectorized ECME iteration. The authors' closure records
      a t location-scale distribution as suitable for the Indian precipitation anomaly
      series.
    - ``Distribution.pearson``: Pearson Type III fitted by L-moments. A cell whose
      L-moments are invalid falls back to the normal distribution, the zero-skew
      Pearson Type III, with a ``distribution_fallback`` warning event.
    - ``Distribution.normal``: fitted by maximum likelihood, so
      ``SPAI = (Z - m) / s`` is the pooled anomaly rescaled by its pooled calibration
      mean ``m`` and standard deviation ``s``. With complete calibration data ``m = 0``
      and ``s`` is the anomaly's pooled root-mean-square deviation.

    Gamma is rejected because ``Z`` takes negative values outside its support.

    This is a Beta API (#1480): each call emits a
    :class:`~climate_indices.exceptions.BetaFeatureWarning`, and the interface may
    change in a minor release.

    Args:
        values: Precipitation values in any units, laid out as for :func:`spi`: a 1-D
            series starting in January of ``data_start_year``, a legacy 2-D
            (years, periods) array read as one series, or a time-major
            (time, ``*cells``) block. Negative values are clipped to zero.
        scale: Number of time steps accumulated before the anomalies are taken.
        distribution: ``Distribution.t_location_scale``, ``Distribution.pearson``, or
            ``Distribution.normal``.
        data_start_year: Initial year of the input series.
        calibration_year_initial: Initial year of the Calibration Period.
        calibration_year_final: Final year of the Calibration Period. A window the
            record does not cover is clamped to the record, as for :func:`spi`.
        periodicity: ``compute.Periodicity.monthly`` (12 values/year) or
            ``compute.Periodicity.daily`` (366 values/year). Hourly data is rejected.
        fitting_params: Optional complete parameter set to reuse instead of fitting:
            "mean", one value per calendar period, shaped (periods,) or
            (periods, ``*cells``), and the distribution's own parameters, each a
            scalar or shaped like the cells: "df", "loc", and "scale" for t
            location-scale, "loc", "scale", and "skew" for Pearson Type III, or "loc"
            and "scale" for normal.
        spatial_time_major: Read a three-or-more-dimensional ``values`` as a
            time-major block; required only for the ambiguous shape whose first cell
            axis is a calendar period length, as for :func:`spi`.
        output_scale: "normal" (the default) returns the z-score; "probability"
            returns ``F(Z)`` in [0, 1] without clipping; "bounded" returns
            ``2F(Z) - 1`` in [-1, 1].

    Returns:
        SPAI values, unitless: a 1-D array as long as the flattened input, or the
        input's (time, ``*cells``) shape for a time-major block.

    Raises:
        InsufficientDataError: If the Calibration Period, after clamping, spans fewer
            than two years, or ``scale`` is longer than the series.
        InvalidArgumentError: If ``distribution``, ``scale``, ``periodicity``, or
            ``output_scale`` is not supported here, the Calibration Period is
            reversed, or ``fitting_params`` is incomplete or does not broadcast.
        PeriodicityError: If ``periodicity`` is not a ``Periodicity`` member.
        ValueError: If ``values`` has an unsupported or undeclared ambiguous shape.

    Missing data and edge cases:
        A NaN input makes every accumulation window that holds it NaN. A calendar
        period with no valid Calibration Period value has a NaN mean, so its anomalies
        are NaN; a period whose calibration values are constant has zero anomalies, so
        its SPAI is the pooled probability of a zero anomaly, never infinite. A cell
        with fewer than four valid pooled calibration anomalies is NaN, and an
        all-missing input is returned as NaN. A Calibration Period shorter than 30 years
        warns.

    Reference:
        Chanda, K., and Maity, R. (2015). Meteorological drought quantification with
        Standardized Precipitation Anomaly Index for the regions with strongly
        seasonal and periodic precipitation. Journal of Hydrologic Engineering,
        20(12), 06015007. https://doi.org/10.1061/(ASCE)HE.1943-5584.0001236
    """
    warnings.warn(_SPAI_BETA_MESSAGE, BetaFeatureWarning, stacklevel=2)
    return _spai(
        values,
        scale,
        distribution,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        periodicity,
        fitting_params,
        spatial_time_major=spatial_time_major,
        output_scale=output_scale,
    )


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _translate_pnp_calibration_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate PNP's deprecated calibration keywords once for both public entries."""
    translated = dict(kwargs)
    if "calibration_start_year" in translated or "calibration_end_year" in translated:
        emit_deprecation_warning(
            feature="Parameters 'calibration_start_year'/'calibration_end_year'",
            alternative="Use 'calibration_year_initial'/'calibration_year_final'",
            deprecated_in="3.0.0",
            removal_version="4.0.0",
        )
        for legacy, canonical in (
            ("calibration_start_year", "calibration_year_initial"),
            ("calibration_end_year", "calibration_year_final"),
        ):
            if legacy in translated and canonical not in translated:
                translated[canonical] = translated.pop(legacy)
            else:
                translated.pop(legacy, None)
    return translated


def _pnp_calibration_alias(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Accept the deprecated ``calibration_start_year``/``_end_year`` PNP keywords."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return func(*args, **_translate_pnp_calibration_kwargs(cast("dict[str, Any]", kwargs)))

    return wrapper


@overload
def percentage_of_normal(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    *,
    spatial_time_major: bool = False,
) -> np.ndarray: ...


@overload
def percentage_of_normal(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: compute.Periodicity,
    *,
    spatial_time_major: bool = False,
) -> np.ndarray: ...


@overload
def percentage_of_normal(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_end_year: int,
    periodicity: compute.Periodicity,
    *,
    spatial_time_major: bool = False,
) -> np.ndarray: ...


@overload
def percentage_of_normal(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    *,
    spatial_time_major: bool = False,
) -> np.ndarray: ...


# The decorator translates legacy keywords before this canonical implementation binds.
@_pnp_calibration_alias  # type: ignore[misc]
def percentage_of_normal(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    *,
    spatial_time_major: bool = False,
) -> np.ndarray:
    """Compute the ratio of normal precipitation (PNP).

    Each timescale-aggregated precipitation value is divided by the average
    for the same calendar month or day over the calibration period. The result
    is a dimensionless ratio: 1.0 is normal precipitation (100%), and 0.5 is
    half of normal (50%). Multiply the result by 100 to express it as a
    percentage. The calibration period typically used for US climate monitoring
    is 1981-2010.

    Args:
        values: 1-D numpy array of precipitation values, any length, initial
            value assumed to be January of the data start year (January 1st of
            the start year if daily periodicity), see the description of the
            *periodicity* argument below for further clarification. A
            time-major spatial block, shaped (time, ``*cells``), divides every
            cell by its own normals in one pass when it is declared with
            ``spatial_time_major``.
        scale: Integer number of time steps over which the normal value is
            computed (months for monthly data, days for daily data; e.g.
            3 months or 90 days).
        data_start_year: The initial year of the input values array.
        calibration_year_initial: The initial year of the calibration period
            over which the normal average for each calendar time step is
            computed. The deprecated ``calibration_start_year`` keyword is
            accepted as an alias.
        calibration_year_final: The final year of the calibration period over
            which the normal average for each calendar time step is computed.
            The deprecated ``calibration_end_year`` keyword is accepted as an
            alias.
        periodicity: Periodicity of the input time series; use
            ``compute.Periodicity.monthly`` for monthly data (12 values/year)
            or ``compute.Periodicity.daily`` for daily data (366 values/year).
        spatial_time_major: Read a three-or-more-dimensional ``values`` as a
            time-major block of independent time series, shaped (time, ``*cells``),
            and divide every cell by its own calendar-period normals in one
            pass. The xarray adapter sets this for every block it packs; a
            direct NumPy caller has to declare it, since a block is not a
            shape this function reads without being told.

    Returns:
        Ratio of normal precipitation values (1.0 = 100% of normal) corresponding
        to the scaled precipitation values array (numpy.ndarray of type float): 1-D for a
        1-D or 2-D input, or the (time, ``*cells``) layout of a declared block.

    Raises:
        CalibrationPeriodError: The calibration period is reversed, starts before
            the data, or ends after the last year the data reaches. It is both an
            ``InvalidArgumentError`` and a ``ValueError``.
    """
    calibration_start_year = calibration_year_initial
    calibration_end_year = calibration_year_final
    # validate arguments
    _validate_periodicity(periodicity)
    _validate_scale(scale, periodicity)

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
        # then we flatten it. Every other input shape is a DataShapeError naming the
        # declared-block path, and a declared block is the only way a 3-D or higher
        # input is accepted. Unlike the fitting-based kernels, this index rejects an
        # undeclared 3-D input outright rather than only the ambiguous shape.
        _raise_if_unsupported_shape(values, spatial_time_major)
        if values.ndim == 2:
            values = values.flatten()

        # calendar months for monthly data (12 periods), or days for daily data (366)
        period_length = periodicity.period_length

        # a calibration period the record does not cover is an error, not clamped. The
        # record counts a trailing partial period as a year, since the calibration slice
        # below pads it with NaN, and a spatial block is measured on its time axis.
        # Resolve before the missing-input return below, so a reversed or uncovered
        # window is rejected even when the input carries no values to compute.
        period = _resolve_unfolded_window(
            values,
            periodicity,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            policy="reject",
        )

        # bypass processing if all values are masked, or when a spatial block is all
        # missing, in which case it is returned as it arrived -- unless the scale
        # exceeds its time steps, which the preparation seam must reject. This is
        # deliberately stricter than compute.is_all_missing: a masked-all input
        # short-circuits at any rank, but a NaN-all 1-D/2-D input still goes through
        # the seam (it averages the un-reshaped sums).
        if scale <= values.shape[0] and (
            (isinstance(values, np.ma.MaskedArray) and values.mask.all())
            or (values.ndim > 2 and np.all(np.isnan(values)))
        ):
            _log_calculation_completed(log, t0, values.shape, memory_metrics)
            return values

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
        calibration_period_sums = scale_sums[
            period.start_index * period_length : (period.end_index + 1) * period_length
        ]

        # pad a trailing partial period with NaN (ignored by the average) so that the
        # calibration period reshapes into whole calendar periods, e.g. when the
        # calibration period ends in the record's partial final year
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
        # averages each cell's calibration years for every calendar time step.
        reshaped_sums = calibration_period_sums.reshape(-1, period_length, *calibration_period_sums.shape[1:])

        # the Rust kernel computes the same normals and ratios when it can take the inputs
        percentages_of_normal = compute._native_pnp_percentages(scale_sums, reshaped_sums, period_length)
        if percentages_of_normal is None:
            # a cell that is missing for the whole calibration period (not just the whole
            # block, which is short-circuited above) is an expected all-NaN slice, not
            # an error, so it's handled with an explicit count/sum rather than
            # np.nanmean's "Mean of empty slice" warning: xarray_adapter's Dask kernel
            # runs one task per spatial block, and warnings.catch_warnings() mutates
            # process-global filter state, so one block's context can suppress or
            # restore filters while a concurrent block is still inside np.nanmean.
            valid_counts = np.sum(~np.isnan(reshaped_sums), axis=0)
            averages = np.nansum(reshaped_sums, axis=0) / np.maximum(valid_counts, 1)
            averages = np.where(valid_counts > 0, averages, np.nan)

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
        log_calculation_failure(log, exc, calibration_period=f"{calibration_start_year}-{calibration_end_year}")
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
        log_calculation_failure(log, exc)
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
            result = compute._native_pci(rainfall_mm)
            if result is None:
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
        log_calculation_failure(log, exc)
        raise

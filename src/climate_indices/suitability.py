"""Distribution-suitability assessment tools (Beta, ADR-0028).

Which distribution should an index use for a region, season and Timescale? This module
gathers relative-fit evidence without changing any index: :func:`compare_distributions`
scores every likelihood-based candidate on one shared calibration sample and returns
log-likelihood, AIC, AICc, BIC and the best candidate for every calendar step (and cell).

The criteria are **penalized likelihoods at the library's own estimates**: gamma by Thom's
approximation, Pearson Type III, log-logistic and GEV by L-moments. They are not maximum-likelihood
estimates in general, so a ranking mixes how well a family fits with how efficient its
estimator is, and the ``aic``, ``aicc`` and ``bic`` values are not textbook AIC, AICc and BIC.
The module is Beta and has no root re-export: import it explicitly,
``import climate_indices.suitability as suitability``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_args

import numpy as np
import scipy.special
import scipy.stats
import xarray as xr

from climate_indices import __version__, compute, provenance
from climate_indices._calibration_period import MIN_CALIBRATION_YEARS, resolve_calibration_period
from climate_indices.drought import indices as _indices
from climate_indices.exceptions import (
    BetaFeatureWarning,
    ClimateIndicesWarning,
    CoordinateValidationError,
    DistributionFittingError,
    InvalidArgumentError,
    ShortCalibrationWarning,
)
from climate_indices.validation import (
    validate_dask_chunks,
    validate_time_dimension,
    validate_time_monotonicity,
)
from climate_indices.vocabulary import Aggregation, Distribution
from climate_indices.xarray_adapter import (
    _align_inputs,
    _effective_calibration_years,
    _infer_data_start_year,
    _infer_temporal_parameters,
    _OutputProvenance,
    _resolve_calendar_plan,
    _spatial_kernel_call,
    _SpatialOutput,
    _validate_sufficient_data,
    _validate_supported_calendar,
)

__all__ = ["Criterion", "DistributionComparison", "IndexRoute", "compare_distributions"]

Criterion = Literal["aicc", "aic", "bic"]
IndexRoute = Literal["spi", "spei", "standardized_index"]

_BETA_MESSAGE = "suitability is beta: its interface may change in a minor release (ADR-0028)."

# the user's call site, seen from a warning raised in _compare_numpy (compare_distributions sits between them)
_CALLER_STACKLEVEL = 3

# candidates per route in canonical Distribution enum order, which is also the tie-break order
_ROUTE_CANDIDATES: dict[str, tuple[Distribution, ...]] = {
    "spi": (Distribution.pearson, Distribution.gamma),
    "standardized_index": (Distribution.pearson, Distribution.gamma),
    "spei": (Distribution.pearson, Distribution.gamma, Distribution.loglogistic, Distribution.gev),
}

# free parameters per family, not counting the zero mass
_FAMILY_PARAMETERS = {
    Distribution.pearson: 3,
    Distribution.gamma: 2,
    Distribution.loglogistic: 3,
    Distribution.gev: 3,
}

# families whose fit carries a zero mass p0; the log-logistic and the GEV have none (ADR-0016)
_ZERO_MASS_FAMILIES = frozenset({Distribution.pearson, Distribution.gamma})

_FITTING_METHODS = {
    Distribution.pearson: "l_moments",
    Distribution.gamma: "thom_mle_approximation",
    Distribution.loglogistic: "l_moments",
    Distribution.gev: "l_moments",
}

# the log-logistic transform treats a shape within this of zero as the logistic limit
_GLO_SHAPE_TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True, eq=False)
class DistributionComparison:
    """Per-step, per-cell relative-fit evidence for a set of candidate distributions.

    Every candidate is scored on the same calibration sample, and the log-likelihood is
    evaluated at the parameters the index itself uses (``climate_indices_log_likelihood_at``
    is ``"fitted_estimates"``), so a ranking mixes distribution fit with estimator efficiency.
    In the shapes below, ``candidate`` is the first axis of the per-candidate arrays, ``step``
    is the calendar step (12 months or 366 days), and ``*cell`` stands for any trailing cell
    dimensions.
    """

    #: The candidates in canonical enum order, which is also the tie-break order.
    candidates: tuple[Distribution, ...]
    #: The criterion ``best`` selects on: ``"aicc"``, ``"aic"`` or ``"bic"``.
    criterion: Criterion
    #: Shared calibration sample size, ``(step, *cell)``.
    n: np.ndarray
    #: Free parameters per candidate, ``(candidate, step, *cell)``. A zero-mass family (gamma,
    #: Pearson Type III) counts one more for p0 where the shared sample holds an exact zero.
    n_parameters: np.ndarray
    #: Log-likelihood at the library's estimates, ``(candidate, step, *cell)``. NaN where the fit
    #: has no usable parameters or the sample cannot be scored; ``-inf`` where a sample value lies
    #: outside the fitted support (``fit_failed`` is True there and the criteria are NaN).
    log_likelihood: np.ndarray
    #: ``2k - 2 * log_likelihood``, same shape.
    aic: np.ndarray
    #: ``aic + 2k(k + 1) / (n - k - 1)``, NaN where ``n <= k + 1``; same shape.
    aicc: np.ndarray
    #: ``k * ln(n) - 2 * log_likelihood``, same shape.
    bic: np.ndarray
    #: Index into ``candidates`` of the argmin of ``criterion`` among eligible candidates
    #: (fitted, with a defined criterion), ``(step, *cell)``. Ties go to the earlier candidate
    #: and -1 means none is eligible. No significance threshold is applied, so a small
    #: ``delta_to_runner_up`` does not discriminate.
    best: np.ndarray
    #: Criterion gap between the best and the second-best eligible candidate; NaN with fewer
    #: than two eligible.
    delta_to_runner_up: np.ndarray
    #: True where a candidate has no usable fit or cannot score the shared sample, including a sample
    #: value outside its fitted support (zero likelihood), ``(candidate, step, *cell)``. A failed fit
    #: is recorded, never replaced by another.
    fit_failed: np.ndarray
    #: Per candidate, the fitted parameters keyed as ``fitting_params`` accepts them, each
    #: ``(step, *cell)``; passing one to the matching index reproduces the fit exactly.
    parameters: dict[Distribution, dict[str, np.ndarray]]
    #: Provenance strings: the ``climate_indices_*`` attributes of ADR-0028 section 6 that apply
    #: to the NumPy core, plus the provenance record.
    metadata: Mapping[str, str]


def _invalid(message: str, name: str, value: Any, valid: str) -> InvalidArgumentError:
    return InvalidArgumentError(message, argument_name=name, argument_value=str(value), valid_values=valid)


def _resolve_candidates(index: str, candidates: Sequence[Distribution] | None) -> tuple[Distribution, ...]:
    """Validate the candidates against the route and return them in canonical enum order."""
    allowed = _ROUTE_CANDIDATES[index]
    if candidates is None:
        return allowed
    names = ", ".join(d.value for d in allowed)
    if isinstance(candidates, str):  # type: ignore[unreachable]
        raise _invalid("candidates must be a sequence of Distribution members.", "candidates", candidates, names)
    chosen = tuple(candidates)
    if not chosen:
        raise _invalid("At least one candidate is required.", "candidates", chosen, names)
    for candidate in chosen:
        if not isinstance(candidate, Distribution):
            raise _invalid(
                f"candidates must be Distribution members, got {type(candidate).__name__} {candidate!r}.",
                "candidates",
                candidate,
                names,
            )
        if candidate not in allowed:
            raise _invalid(
                f"{candidate} is not a likelihood-based candidate for index='{index}'.", "candidates", candidate, names
            )
    if len(set(chosen)) != len(chosen):
        raise _invalid("Candidates must be distinct.", "candidates", chosen, names)
    return tuple(d for d in allowed if d in chosen)


def _validate_option_types(criterion: str, aggregation: Aggregation, clip_negatives: bool) -> None:
    """Reject option values that the route checks below would otherwise compare or trust."""
    if not isinstance(criterion, str) or criterion not in get_args(Criterion):
        raise _invalid(f"Unsupported criterion: {criterion!r}.", "criterion", criterion, ", ".join(get_args(Criterion)))
    if not isinstance(aggregation, str) or aggregation not in get_args(Aggregation):
        raise _invalid(
            f"Unsupported aggregation: {aggregation!r}.", "aggregation", aggregation, ", ".join(get_args(Aggregation))
        )
    if not isinstance(clip_negatives, (bool, np.bool_)):
        raise _invalid(
            f"clip_negatives must be a boolean, got {type(clip_negatives).__name__}.",
            "clip_negatives",
            clip_negatives,
            "True or False",
        )


def _validate_arguments(
    index: str,
    criterion: str,
    pet_mm: np.ndarray | xr.DataArray | None,
    aggregation: Aggregation,
    clip_negatives: bool,
) -> None:
    if not isinstance(index, str) or index not in _ROUTE_CANDIDATES:
        raise _invalid(f"Unsupported index route: {index!r}.", "index", index, ", ".join(get_args(IndexRoute)))
    _validate_option_types(criterion, aggregation, clip_negatives)
    if index == "spei" and pet_mm is None:
        raise _invalid("index='spei' requires pet_mm.", "pet_mm", None, "an array matching the precipitation")
    if index != "spei" and pet_mm is not None:
        raise _invalid("pet_mm applies only to index='spei'.", "pet_mm", "given", "None")
    if index != "standardized_index" and (aggregation != "sum" or not clip_negatives):
        raise _invalid(
            "aggregation and clip_negatives apply only to index='standardized_index'.",
            "aggregation" if aggregation != "sum" else "clip_negatives",
            aggregation if aggregation != "sum" else clip_negatives,
            "the defaults ('sum', True)",
        )


def _require_numeric_dtype(dtype: np.dtype[Any], name: str) -> None:
    """Reject a dtype that the fits cannot accept, before any conversion can hide it."""
    if dtype.kind not in "fiub":
        raise _invalid(f"{name} must hold numeric values, got dtype {dtype}.", name, dtype, "a numeric array")


def _as_float_array(values: np.ndarray, name: str = "values") -> np.ndarray:
    """A masked array as NaN-filled, an integer or boolean array as float, and reject other dtypes or infinities.

    NaN and a mask are the missing-value markers; an infinity is invalid input, not a missing value.
    """
    try:
        if np.ma.isMaskedArray(values):
            _require_numeric_dtype(values.dtype, name)
            array = np.asarray(np.ma.filled(values.astype(float), np.nan))
        else:
            array = np.asarray(values)
            _require_numeric_dtype(array.dtype, name)
            if array.dtype.kind in "iub":
                array = array.astype(float)
    except (TypeError, ValueError) as error:
        raise _invalid(f"{name} must hold numeric values.", name, type(values).__name__, "a numeric array") from error
    if np.isinf(array).any():
        raise _invalid(f"{name} must not hold an infinite value.", name, "inf", "finite values, NaN or a mask")
    return array


def _validate_pet_shape(values: np.ndarray, pet_mm: np.ndarray) -> None:
    """Reject a PET that would broadcast over a larger cell grid than the precipitation."""
    if pet_mm.shape == values.shape:
        return
    if values.ndim >= 3:
        if pet_mm.ndim == 1 and pet_mm.shape[0] == values.shape[0]:
            return
        if (
            pet_mm.ndim == values.ndim
            and pet_mm.shape[0] == values.shape[0]
            and all(
                size in (values_size, 1) for size, values_size in zip(pet_mm.shape[1:], values.shape[1:], strict=True)
            )
        ):
            return
    raise _invalid(
        "pet_mm must match values, or be a per-step series or singleton axes for a spatial block.",
        "pet_mm",
        str(pet_mm.shape),
        str(values.shape),
    )


def _prepare_values(
    values: np.ndarray, index: IndexRoute, pet_mm: np.ndarray | None, clip_negatives: bool
) -> tuple[np.ndarray, bool]:
    """Prepare the input and clipping policy for an already validated route."""
    if index == "spei" and pet_mm is not None:
        pet = _as_float_array(pet_mm, "pet_mm")
        _validate_pet_shape(values, pet)
        return _indices._spei_water_balance(values, pet), False
    if index == "standardized_index" and not clip_negatives:
        values = _indices._reject_negative_values(values, Distribution.gamma)
    return values, True if index == "spi" else clip_negatives


def _glo_logpdf(x: np.ndarray, loc: np.ndarray, scale: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Hosking's generalized-logistic log-density, ``-inf`` outside the support.

    With ``z = (x - loc) / scale`` and ``y = -log(1 - k z) / k`` (``y = z`` as ``k -> 0``),
    ``log f = -log(scale) - (1 - k) y - 2 log(1 + exp(-y))``. The support is ``k z < 1``.
    """
    with np.errstate(all="ignore"):
        logistic = np.abs(shape) <= _GLO_SHAPE_TOLERANCE
        k = np.where(logistic, 0.0, shape)
        z = (x - loc) / scale
        argument = 1.0 - k * z
        inside = logistic | (argument > 0.0)
        y = np.where(logistic, z, -np.log(np.where(argument > 0.0, argument, 1.0)) / np.where(logistic, 1.0, k))
        log_density = -np.log(scale) - (1.0 - k) * y - 2.0 * np.logaddexp(0.0, -y)
        scored: np.ndarray = np.where(inside, log_density, -np.inf)
    return scored


def _log_density(distribution: Distribution, sample: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    """Log-density of the continuous part of one family at its fitted parameters."""
    with np.errstate(all="ignore"):
        if distribution is Distribution.gamma:
            density: np.ndarray = scipy.stats.gamma.logpdf(sample, a=p["alpha"], scale=p["beta"])
        elif distribution is Distribution.pearson:
            density = scipy.stats.pearson3.logpdf(sample, p["skew"], loc=p["loc"], scale=p["scale"])
        elif distribution is Distribution.loglogistic:
            density = _glo_logpdf(sample, p["loc"], p["scale"], p["shape"])
        elif distribution is Distribution.gev:
            # Hosking's shape k is SciPy's c; the density is -inf outside the fitted support
            density = scipy.stats.genextreme.logpdf(sample, p["shape"], loc=p["loc"], scale=p["scale"])
        else:
            raise ValueError(f"{distribution} has no likelihood in suitability")
    return density


def _usable(distribution: Distribution, p: dict[str, np.ndarray]) -> np.ndarray:
    """Where a fit has usable parameters, by the same predicates the existing fits use."""
    if distribution is Distribution.gamma:
        ok = np.isfinite(p["alpha"]) & np.isfinite(p["beta"]) & (p["alpha"] > 0) & (p["beta"] > 0)
    elif distribution is Distribution.pearson:
        loc, scale, skew = p["loc"], p["scale"], p["skew"]
        # an unfitted step is reported as all-zero parameters
        ok = ~((loc == 0) & (scale == 0) & (skew == 0))
        ok &= np.isfinite(loc) & np.isfinite(scale) & np.isfinite(skew) & (scale > 0)
    elif distribution is Distribution.loglogistic:
        loc, scale, shape = p["loc"], p["scale"], p["shape"]
        ok = np.isfinite(loc) & np.isfinite(scale) & np.isfinite(shape) & (scale > 0) & (np.abs(shape) < 1.0)
    elif distribution is Distribution.gev:
        # the predicate _gev_fit uses: an unfitted step is reported with a zero scale
        loc, scale, shape = p["loc"], p["scale"], p["shape"]
        ok = np.isfinite(loc) & np.isfinite(scale) & np.isfinite(shape) & (scale > 0)
    else:
        raise ValueError(f"{distribution} has no likelihood in suitability")
    if distribution in _ZERO_MASS_FAMILIES:
        ok = ok & np.isfinite(p["prob_zero"])
    result: np.ndarray = ok
    return result


def _score_fit(
    distribution: Distribution,
    sample: np.ndarray,
    p: dict[str, np.ndarray],
    valid: np.ndarray,
    n: np.ndarray,
    n0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a fit on the shared sample, including a zero mass only for families that fit one."""
    zero_mass = distribution in _ZERO_MASS_FAMILIES
    continuous_sample = valid & (sample != 0.0) if zero_mass else valid  # NOSONAR (S1244) p0 uses exact zeros
    total = np.where(continuous_sample, _log_density(distribution, sample, p), 0.0).sum(axis=0)
    if zero_mass:
        p0 = p["prob_zero"]
        with np.errstate(all="ignore"):
            total = total + scipy.special.xlog1py(n - n0, -p0) + scipy.special.xlogy(n0, p0)
    scorable = _usable(distribution, p) & (n > 0) & ~np.isnan(total)
    # A value outside the fitted support gives -inf: report it, but exclude the candidate from ranking.
    return np.where(scorable, total, np.nan), ~(scorable & np.isfinite(total))


def _select_best(criterion_values: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The best eligible candidate along axis 0 and its gap to the runner-up.

    Ties go to the earlier candidate (``argmin`` returns the first minimum); -1 where nothing
    is eligible; the gap is NaN with fewer than two eligible candidates.
    """
    key = np.where(eligible, criterion_values, np.inf)
    chosen = np.argmin(key, axis=0)
    # all-infinite keys would pick candidate 0 whether or not it is eligible
    fallback = np.argmax(eligible, axis=0)
    chosen_ok = np.take_along_axis(eligible, chosen[np.newaxis], axis=0)[0]
    chosen = np.where(chosen_ok, chosen, fallback)
    best = np.where(eligible.any(axis=0), chosen, -1).astype(np.intp)
    gap = np.full(best.shape, np.nan)
    if len(key) > 1:
        ordered = np.sort(key, axis=0)
        with np.errstate(invalid="ignore"):
            gap = np.where(eligible.sum(axis=0) >= 2, ordered[1] - ordered[0], np.nan)
    return best, gap


def compare_distributions(
    values: np.ndarray | xr.DataArray,  # NOSONAR (S107) the public API declares each contract argument explicitly
    scale: int,
    *,
    index: IndexRoute,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: compute.Periodicity | None = None,
    candidates: Sequence[Distribution] | None = None,
    pet_mm: np.ndarray | xr.DataArray | None = None,
    aggregation: Aggregation = "sum",
    clip_negatives: bool = True,
    criterion: Criterion = "aicc",
    spatial_time_major: bool = False,
    time_dim: str = "time",
) -> DistributionComparison | xr.Dataset:
    """Compare candidate distributions for an index on one shared calibration sample.

    The series is prepared exactly as the named index prepares it (SPEI's water balance and
    +1000 offset, ``standardized_index``'s ``aggregation`` and ``clip_negatives``), each
    candidate is fitted with the library's own estimator for the calibration period, and the
    log-likelihood is evaluated at those estimates. A zero is an exact zero, the definition
    both existing zero-mass estimators fit with; the 0.0005 Pearson trace rule is a transform
    convention and does not change the sample. SPEI's log-logistic and GEV candidates score zeros
    as continuous observations when they lie within their fitted support (ADR-0016).
    The Pearson-to-gamma fallback is disabled: a failed fit is recorded in ``fit_failed``,
    not replaced.

    Because the likelihood is taken at Thom's gamma approximation and at L-moment estimates,
    not at maximum-likelihood estimates, the criteria are penalized likelihoods at fitted
    estimates and a ranking mixes distribution fit with estimator efficiency. About 30 values
    per calendar step discriminate weakly, so treat ``best`` as evidence, not as "correct".
    A sample value outside a candidate's fitted support gives zero likelihood, so that candidate
    is reported in ``fit_failed`` and does not rank; ``best`` can then have only one eligible
    candidate, and ``delta_to_runner_up`` is NaN. A GEV with shape above 1 has a density that is
    unbounded at the finite end of its support, where likelihood methods are not regular, so
    treat its criteria with caution there.

    An ``xr.DataArray`` takes the xarray route (Dask-backed input included, with time in a single
    chunk, ADR-0003) and returns an ``xr.Dataset``; see Returns.

    Args:
        values: Precipitation, or the series ``standardized_index`` would standardize. NumPy: 1-D,
            2-D ``(years, periods)``, or a time-major block ``(time, *cells)`` with
            ``spatial_time_major=True``. xarray: a DataArray with the ``time_dim`` dimension plus any
            cell dimensions.
        scale: Timescale in time steps accumulated before fitting.
        index: ``"spi"``, ``"spei"`` or ``"standardized_index"``.
        data_start_year: First year of ``values``. Required for NumPy input; inferred from the time
            coordinate for xarray input, where an explicit value must match that coordinate.
        calibration_year_initial: First calibration year; clamped to the record. Required for NumPy
            input; inferred from the time range for xarray input.
        calibration_year_final: Last calibration year; clamped to the record. Required for NumPy
            input; inferred from the time range for xarray input.
        periodicity: Monthly or daily. Required for NumPy input; inferred from the time coordinate
            for xarray input.
        candidates: Families to compare; ``None`` selects gamma and Pearson Type III, plus
            log-logistic and GEV for ``"spei"``. Naming any other family raises.
        pet_mm: Potential evapotranspiration matching ``values``; required for, and only for,
            ``index="spei"``. A NumPy spatial block also accepts a per-step series or singleton
            cell axes; a shape that would broadcast over a larger cell grid raises. A labelled
            ``DataArray`` is refused rather than paired by position, which would ignore its
            labels; any other array-like is paired by position. For xarray input it must be a
            DataArray with the same dimensions as ``values``.
        aggregation: ``"sum"`` or ``"mean"`` window aggregation; ``"standardized_index"`` only.
        clip_negatives: Clip negative input to zero. ``False`` rejects negatives, as
            ``standardized_index`` does for gamma and Pearson; ``"standardized_index"`` only.
        criterion: Criterion ``best`` selects on.
        spatial_time_major: Read a three-or-more-dimensional NumPy ``values`` as independent
            series shaped ``(time, *cells)``. The xarray route reads its layout from the
            dimension names and ignores this.
        time_dim: Name of the time dimension of an xarray ``values``; xarray input only.

    Returns:
        For NumPy input, a frozen :class:`DistributionComparison`, whose provenance records
        ``data_start_year`` along with the effective calibration bounds to identify the shared
        sample. For xarray input, an ``xr.Dataset`` of per-candidate variables over
        ``(candidate, month | dayofyear, *cells)`` (``log_likelihood``, ``aic``, ``aicc``,
        ``bic``, ``n_parameters``, ``fit_failed`` and every fitted parameter, NaN where a family
        has no such parameter) and per-step variables over ``(month | dayofyear, *cells)``
        (``n``, ``best`` as the winning candidate's name or ``"none"``, ``delta_to_runner_up``).
        Every variable carries the ADR-0028 section 6 attributes that apply to this route. A fully
        missing input is accepted and scores as no usable fit (``n`` is 0 and ``best`` is
        ``"none"``) instead of raising as the NumPy route does.

    Raises:
        InvalidArgumentError: For an unsupported route, criterion, candidate, hourly
            periodicity, non-numeric or infinite input, an argument combination the route
            does not accept, a NumPy call that omits a required temporal argument, an xarray
            ``data_start_year`` that disagrees with the time coordinate, or an input dimension or
            a coordinate that does not span the time dimension named like an output variable. A
            coordinate that spans the time dimension is allowed, since the reduction over time
            drops it.
        ValueError: If a NumPy ``values`` holds no valid value.
        InsufficientDataError: If the record, or a calendar-adapted daily record, is shorter
            than ``scale``.
        CoordinateValidationError: For an xarray time dimension that is missing, non-monotonic,
            unsupported, does not begin in January, mismatches ``periodicity``, or is split
            across Dask chunks.

    Warns:
        BetaFeatureWarning: Always; the module is Beta.
        ShortCalibrationWarning: From the fits when the calibration period is under 30 years, and
            from this function when a calendar step or cell has fewer than 30 shared values within a
            period of 30 years or more; ``actual_years`` then holds the smallest step or cell count.
        Warnings the fits raise (``CalibrationPeriodClampedWarning``, ``MissingDataWarning``, ``GoodnessOfFitWarning``)
            reach the caller as they do from the index; each candidate's fit can raise them.
            The xarray route evaluates one series per block or cell, so these repeat there, and a
            Dask input defers them to ``.compute()``.
        ClimateIndicesWarning: If ``scale`` exceeds 12 (monthly) or 365 (daily) steps, where
            consecutive years share overlapping windows and effective sample sizes shrink.
    """
    warnings.warn(_BETA_MESSAGE, BetaFeatureWarning, stacklevel=2)
    if isinstance(values, xr.DataArray):
        return _compare_xarray(
            values,
            scale,
            index=index,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_initial,
            calibration_year_final=calibration_year_final,
            periodicity=periodicity,
            candidates=candidates,
            pet_mm=pet_mm,
            aggregation=aggregation,
            clip_negatives=clip_negatives,
            criterion=criterion,
            time_dim=time_dim,
        )
    if (
        data_start_year is None
        or calibration_year_initial is None
        or calibration_year_final is None
        or periodicity is None
    ):
        raise _invalid(
            "data_start_year, calibration_year_initial, calibration_year_final and periodicity are required "
            "for NumPy input.",
            "data_start_year",
            None,
            "all four temporal arguments",
        )
    # a DataArray pet_mm would be paired by position, so a transposed or mislabelled one
    # silently yields the wrong water balance; the xarray route rejects the mirror case
    return _compare_numpy(
        values,
        scale,
        index=index,
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        periodicity=periodicity,
        candidates=candidates,
        pet_mm=pet_mm,
        aggregation=aggregation,
        clip_negatives=clip_negatives,
        criterion=criterion,
        spatial_time_major=spatial_time_major,
    )


def _shape_all_missing(
    scaled: np.ndarray, prepared: np.ndarray, periodicity: compute.Periodicity, spatial_time_major: bool
) -> np.ndarray:
    """Give an all-missing 1-D or time-major ``prepared`` the ``(years, step, *cells)`` layout; else return ``scaled``.

    ``prepare_scaled`` hands an all-missing series back unreshaped, so a masked cell or chunk would
    abort the grid instead of scoring as "no usable fit".
    """
    if not (spatial_time_major or prepared.ndim == 1) or not compute.is_all_missing(prepared):
        return scaled
    step_count = periodicity.period_length
    return np.full((-(-prepared.shape[0] // step_count), step_count, *prepared.shape[1:]), np.nan)


def _compare_numpy(
    values: np.ndarray,  # NOSONAR (S107) mirrors the public keywords, which the temporal-inference helpers bind against
    scale: int,
    *,
    index: IndexRoute,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    candidates: Sequence[Distribution] | None = None,
    pet_mm: np.ndarray | xr.DataArray | None = None,
    aggregation: Aggregation = "sum",
    clip_negatives: bool = True,
    criterion: Criterion = "aicc",
    spatial_time_major: bool = False,
    require_valid: bool = True,
) -> DistributionComparison:
    """The NumPy core of :func:`compare_distributions`, without the Beta warning (see there).

    ``require_valid=False`` lets an all-missing input through, as one masked block or cell of a
    grid: it scores as no usable fit instead of aborting the whole grid.
    """
    _validate_arguments(index, criterion, pet_mm, aggregation, clip_negatives)
    # a DataArray would be paired with values by position, ignoring its labels, and a
    # transposed or mislabelled one would silently yield the wrong water balance
    if isinstance(pet_mm, xr.DataArray):
        raise _invalid(
            "pet_mm must be a NumPy array when values is; transpose a DataArray into the values' "
            "dimension order before converting it.",
            "pet_mm",
            type(pet_mm).__name__,
            "np.ndarray",
        )
    chosen = _resolve_candidates(index, candidates)
    _indices._validate_periodicity(periodicity)
    _indices._validate_scale(scale, periodicity)

    values = _as_float_array(values)
    prepared, clip = _prepare_values(values, index, pet_mm, clip_negatives)
    if require_valid and compute.is_all_missing(prepared):
        raise ValueError("The values hold no valid value to compare distributions on.")

    scaled = compute.prepare_scaled(
        prepared,
        scale,
        periodicity,
        clip_negatives=clip,
        spatial_time_major=spatial_time_major,
        aggregation=aggregation,
    )
    if not require_valid:
        scaled = _shape_all_missing(scaled, prepared, periodicity, spatial_time_major)
    period = resolve_calibration_period(
        data_start_year, scaled.shape[0], calibration_year_initial, calibration_year_final, policy="clamp"
    )
    sample = scaled[period.rows, ...]
    valid = ~np.isnan(sample)
    n = valid.sum(axis=0)
    n0 = (valid & (sample == 0.0)).sum(axis=0)  # NOSONAR (S1244) p0 counts exact zeros, not trace values

    cell_shape = sample.shape[1:]
    log_likelihood = np.full((len(chosen), *cell_shape), np.nan)
    n_parameters = np.zeros((len(chosen), *cell_shape), dtype=np.int64)
    fit_failed = np.ones((len(chosen), *cell_shape), dtype=bool)
    parameters: dict[Distribution, dict[str, np.ndarray]] = {}

    for position, distribution in enumerate(chosen):
        # the fit's own calibration-quality warnings reach the caller as they do from the index; suppressing
        # duplicates would mean mutating the process-wide warning filters, which is unsafe across threads
        try:
            fitted: compute.FittedDistribution | None = compute.FittedDistribution.resolve(
                scaled,
                distribution,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                periodicity,
            )
        except DistributionFittingError:
            fitted = None
        zero_mass = distribution in _ZERO_MASS_FAMILIES
        n_parameters[position] = _FAMILY_PARAMETERS[distribution] + ((n0 > 0) if zero_mass else 0)
        keys = compute._PARAMETER_KEYS[distribution.value]
        if fitted is None:
            parameters[distribution] = {key: np.full(cell_shape, np.nan) for key in keys}
            continue
        full = compute._broadcast_parameters(sample, fitted.parameters, full=True)
        parameters[distribution] = {key: full[key] for key in keys}
        log_likelihood[position], fit_failed[position] = _score_fit(distribution, sample, full, valid, n, n0)

    k_all = n_parameters.astype(float)
    n_float = np.broadcast_to(n.astype(float), log_likelihood.shape)
    with np.errstate(all="ignore"):
        aic = 2.0 * k_all - 2.0 * log_likelihood
        bic = k_all * np.log(n_float) - 2.0 * log_likelihood
        aicc = np.where(n_float > k_all + 1.0, aic + 2.0 * k_all * (k_all + 1.0) / (n_float - k_all - 1.0), np.nan)
    aic = np.where(fit_failed, np.nan, aic)
    bic = np.where(fit_failed, np.nan, bic)
    aicc = np.where(fit_failed, np.nan, aicc)
    selected = {"aicc": aicc, "aic": aic, "bic": bic}[criterion]
    best, delta = _select_best(selected, ~fit_failed & ~np.isnan(selected))

    # a calibration period under 30 years already drew the fit's own ShortCalibrationWarning
    short = n[(n > 0) & (n < MIN_CALIBRATION_YEARS)]
    if short.size and period.n_years >= MIN_CALIBRATION_YEARS:
        message = (
            f"{short.size} calendar step(s) or cell(s) have fewer than {MIN_CALIBRATION_YEARS} shared "
            f"calibration values (smallest {int(short.min())}); criteria and `best` are unreliable there."
        )
        warnings.warn(
            ShortCalibrationWarning(message, actual_years=int(short.min()), required_years=MIN_CALIBRATION_YEARS),
            stacklevel=_CALLER_STACKLEVEL,
        )
    window_limit = 365 if periodicity is compute.Periodicity.daily else 12
    if scale > window_limit:
        warnings.warn(
            f"Timescale {scale} exceeds {window_limit} steps, so accumulation windows ending on the same "
            "calendar step in consecutive years overlap; effective sample sizes shrink and goodness-of-fit "
            "p-values are invalid.",
            ClimateIndicesWarning,
            stacklevel=_CALLER_STACKLEVEL,
        )

    inputs: dict[str, Any] = {"values": values}
    if pet_mm is not None:
        inputs["pet_mm"] = pet_mm
    metadata = provenance.provenance_attributes(
        index="distribution comparison",
        implementation="climate_indices.suitability.compare_distributions",
        interface="python",
        parameters={
            "index": index,
            "scale": scale,
            "data_start_year": data_start_year,
            "periodicity": periodicity.name,
            "aggregation": aggregation,
            "clip_negatives": clip,
            "criterion": criterion,
            "candidates": [d.value for d in chosen],
            "calibration_year_initial": period.start_year,
            "calibration_year_final": period.end_year,
            "spatial_time_major": spatial_time_major,
        },
        inputs=inputs,
    )
    metadata.update(
        {
            "climate_indices_candidates": json.dumps([d.value for d in chosen]),
            "climate_indices_fitting_methods": json.dumps({d.value: _FITTING_METHODS[d] for d in chosen}),
            "climate_indices_criterion": criterion,
            "climate_indices_log_likelihood_at": "fitted_estimates",
            "scale": str(scale),
            "calibration_year_initial": str(period.start_year),
            "calibration_year_final": str(period.end_year),
            "climate_indices_version": __version__,
        }
    )
    return DistributionComparison(
        candidates=chosen,
        criterion=criterion,
        n=n,
        n_parameters=n_parameters,
        log_likelihood=log_likelihood,
        aic=aic,
        aicc=aicc,
        bic=bic,
        best=best,
        delta_to_runner_up=delta,
        fit_failed=fit_failed,
        parameters=parameters,
        metadata=metadata,
    )


# ---- xarray / Dask route (SUIT-04, #1585) ---------------------------------------------------------

_CANDIDATE_FIELDS = ("log_likelihood", "aic", "aicc", "bic", "n_parameters", "fit_failed")
_STEP_FIELDS = ("n", "best", "delta_to_runner_up")

_FIELD_METADATA: dict[str, dict[str, str]] = {
    "log_likelihood": {
        "long_name": "Log-likelihood of the candidate at the library's fitted estimates",
        "units": "1",
        "description": (
            "Sum of the log-density (plus the zero-mass term for a zero-mass family) over the shared "
            "calibration sample, evaluated at the parameters the index itself uses."
        ),
    },
    "aic": {
        "long_name": "Akaike information criterion at the library's fitted estimates",
        "units": "1",
        "description": "2k - 2 log-likelihood; NaN where the candidate failed to fit or score the sample.",
    },
    "aicc": {
        "long_name": "Small-sample corrected Akaike information criterion at the library's fitted estimates",
        "units": "1",
        "description": "AIC + 2k(k + 1) / (n - k - 1); NaN where n <= k + 1 or the candidate failed.",
    },
    "bic": {
        "long_name": "Bayesian information criterion at the library's fitted estimates",
        "units": "1",
        "description": "k ln(n) - 2 log-likelihood; NaN where the candidate failed to fit or score the sample.",
    },
    "n_parameters": {
        "long_name": "Free parameters of the candidate",
        "units": "1",
        "description": "Fitted parameters, plus one for the zero mass where a zero-mass family sees an exact zero.",
    },
    "fit_failed": {
        "long_name": "Candidate has no usable fit",
        "description": (
            "True where the candidate has no usable fit or a sample value lies outside its fitted support; "
            "such a candidate does not rank. A failed fit is recorded, never replaced by another."
        ),
    },
    "n": {
        "long_name": "Shared calibration sample size",
        "units": "1",
        "description": "Valid values per calendar step and cell, identical for every candidate.",
    },
    "best": {
        "long_name": "Best candidate by the requested criterion",
        "description": (
            "Name of the eligible candidate with the smallest criterion (ties go to the earlier candidate), or "
            "'none' where no candidate is eligible. No significance threshold is applied."
        ),
    },
    "delta_to_runner_up": {
        "long_name": "Criterion gap between the best and second-best candidate",
        "units": "1",
        "description": "NaN with fewer than two eligible candidates; a small gap does not discriminate.",
    },
}

_PARAMETER_METADATA: dict[str, str] = {
    "alpha": "Gamma shape parameter",
    "beta": "Gamma scale parameter",
    "prob_zero": "Probability of an exact zero",
    "loc": "Location parameter",
    "scale": "Scale parameter",
    "skew": "Pearson Type III skew parameter",
    "shape": "Shape parameter",
}

_NO_BEST = "none"
_UNITLESS_FIELDS = frozenset({"fit_failed", "best"})


def _comparison_block(
    values: np.ndarray,
    *secondary: np.ndarray,
    chosen: tuple[Distribution, ...],
    slots: tuple[tuple[str, Distribution], ...],
    **kwargs: Any,
) -> tuple[np.ndarray, ...]:
    """Run the NumPy core on one block or cell and flatten it into the declared output arrays."""
    result = _compare_numpy(
        values, pet_mm=secondary[0] if secondary else None, candidates=chosen, require_valid=False, **kwargs
    )
    arrays = [
        np.asarray(getattr(result, field)[position], dtype=float)
        for field in _CANDIDATE_FIELDS
        for position in range(len(chosen))
    ]
    arrays += [np.asarray(result.parameters[distribution][name], dtype=float) for name, distribution in slots]
    arrays += [np.asarray(array, dtype=float) for array in (result.n, result.best, result.delta_to_runner_up)]
    return tuple(arrays)


def _candidate_axis(labels: Sequence[str]) -> xr.DataArray:
    """The ``candidate`` coordinate as plain fixed-width strings, which every NetCDF engine writes."""
    return xr.DataArray(np.array(labels), dims="candidate", name="candidate")


def _label_best(labels: Sequence[str]) -> Any:
    """A transform from the core's candidate index (-1 for none) to the candidate's name."""

    def transform(code: xr.DataArray) -> xr.DataArray:
        label: xr.DataArray = xr.where(code == 0, labels[0], _NO_BEST)  # type: ignore[no-untyped-call]
        for position, name in enumerate(labels[1:], start=1):
            label = xr.where(code == position, name, label)  # type: ignore[no-untyped-call]
        return label

    return transform


def _validate_calendar(time_coord: xr.DataArray) -> None:
    """Reject a calendar the NumPy core cannot represent, including a pandas time zone.

    ``_validate_supported_calendar`` probes the dtype with ``np.issubdtype``, which raises on a
    pandas extension dtype, so a timezone-aware coordinate would escape as ``TypeError``.
    """
    dtype = time_coord.dtype
    if not (isinstance(dtype, np.dtype) and dtype.kind == "M"):
        raise CoordinateValidationError(
            message="Unsupported calendar: a non-datetime or timezone-aware time coordinate is not supported.",
            coordinate_name=str(time_coord.name) if time_coord.name is not None else "time",
            reason="unsupported_calendar",
        )
    _validate_supported_calendar(time_coord)


def _validated_inputs(
    values: xr.DataArray, pet_mm: np.ndarray | xr.DataArray | None, time_dim: str
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Validate the xarray inputs and align ``pet_mm`` to ``values`` on time; return both."""
    pet_da: xr.DataArray | None = None
    if pet_mm is not None:
        if not isinstance(pet_mm, xr.DataArray):
            raise _invalid(
                "pet_mm must be a DataArray when values is.", "pet_mm", type(pet_mm).__name__, "xr.DataArray"
            )
        if set(pet_mm.dims) != set(values.dims):
            raise _invalid(
                "pet_mm must have the same dimensions as values.", "pet_mm", str(pet_mm.dims), str(values.dims)
            )
        pet_da = pet_mm

    input_da = values
    validate_time_dimension(input_da, time_dim)
    validate_time_monotonicity(input_da[time_dim])
    # the calendar itself is checked before inference reads it, so a numeric, cftime or
    # timezone-aware coordinate fails as CoordinateValidationError rather than inside pandas
    _validate_calendar(input_da[time_dim])
    if pet_da is not None:
        validate_time_dimension(pet_da, time_dim)
        validate_time_monotonicity(pet_da[time_dim])
        _validate_calendar(pet_da[time_dim])
        input_da, aligned = _align_inputs(input_da, {"pet": pet_da}, time_dim)
        pet_da = aligned["pet"]
    for dataarray in (input_da, pet_da):
        if dataarray is not None and dataarray.chunks is not None:
            validate_dask_chunks(dataarray, time_dim)
    return input_da, pet_da


def _compare_xarray(
    values: xr.DataArray,
    scale: int,
    *,
    index: IndexRoute,
    data_start_year: int | None,
    calibration_year_initial: int | None,
    calibration_year_final: int | None,
    periodicity: compute.Periodicity | None,
    candidates: Sequence[Distribution] | None,
    pet_mm: np.ndarray | xr.DataArray | None,
    aggregation: Aggregation,
    clip_negatives: bool,
    criterion: Criterion,
    time_dim: str,
) -> xr.Dataset:
    """The xarray/Dask route of :func:`compare_distributions`, following ``fit_diagnostics``."""
    _validate_arguments(index, criterion, pet_mm, aggregation, clip_negatives)
    chosen = _resolve_candidates(index, candidates)
    input_da, pet_da = _validated_inputs(values, pet_mm, time_dim)

    provided: dict[str, Any] = {
        key: value
        for key, value in {
            "scale": scale,
            "index": index,
            "data_start_year": data_start_year,
            "calibration_year_initial": calibration_year_initial,
            "calibration_year_final": calibration_year_final,
            "periodicity": periodicity,
            "aggregation": aggregation,
            "clip_negatives": clip_negatives,
            "criterion": criterion,
        }.items()
        if value is not None
    }
    # inference is metadata-only, so it stays safe for Dask-backed input
    provided.update(_infer_temporal_parameters(_compare_numpy, input_da, [input_da], provided, time_dim))
    # the coordinate is authoritative here: a data_start_year that disagrees with it would
    # slice the record from a different origin than the years recorded in every attribute
    coordinate_start_year = _infer_data_start_year(input_da[time_dim])
    if provided["data_start_year"] != coordinate_start_year:
        raise _invalid(
            f"data_start_year ({provided['data_start_year']}) must match the first year of the time "
            f"coordinate ({coordinate_start_year}) for xarray input; the record may have been aligned to pet_mm.",
            "data_start_year",
            provided["data_start_year"],
            str(coordinate_start_year),
        )
    resolved_periodicity: compute.Periodicity = provided["periodicity"]
    _indices._validate_periodicity(resolved_periodicity)
    _indices._validate_scale(scale, resolved_periodicity)
    calendar_plan = _resolve_calendar_plan(_compare_numpy, input_da, [input_da], {}, provided, time_dim)
    _validate_sufficient_data(input_da[time_dim], scale, calendar_plan)

    monthly = resolved_periodicity is compute.Periodicity.monthly
    period_dim = "month" if monthly else "dayofyear"
    steps = np.arange(1, 13 if monthly else 367)
    labels = [distribution.value for distribution in chosen]
    slots = tuple(
        (name, distribution)
        for name in dict.fromkeys(key for d in chosen for key in compute._PARAMETER_KEYS[d.value])
        for distribution in chosen
        if name in compute._PARAMETER_KEYS[distribution.value]
    )

    # an input dimension or coordinate with an output's name would drop that variable or fail deep in the merge
    reserved = {"candidate", period_dim, *_CANDIDATE_FIELDS, *_STEP_FIELDS, *(name for name, _ in slots)}
    # apply_ufunc drops a coordinate that spans the core time dimension, so only dimensions
    # and the coordinates that can survive into the output can collide with a variable name
    taken = {time_dim, *input_da.dims} | {
        name for name, coordinate in input_da.coords.items() if time_dim not in coordinate.dims
    }
    if pet_da is not None:
        taken |= {*pet_da.dims} | {
            name for name, coordinate in pet_da.coords.items() if time_dim not in coordinate.dims
        }
    if clash := sorted(reserved & taken):
        raise _invalid(
            f"Input dimensions and coordinates must not be named like an output: {', '.join(clash)}.",
            "values",
            ", ".join(clash),
            "rename them, or drop a scalar coordinate such as the one a .sel() leaves behind",
        )

    first_year, last_year = _effective_calibration_years(
        input_da[time_dim],
        provided["calibration_year_initial"],
        provided["calibration_year_final"],
        provided.get("data_start_year"),
    )
    effective_clip = {"spei": False, "spi": True}.get(index, clip_negatives)
    # NetCDF attributes cannot hold a bool, list or None, so every value here is a string or a number
    calculation_metadata: dict[str, Any] = {
        "index": index,
        "scale": str(scale),
        "data_start_year": str(provided["data_start_year"]),
        "periodicity": resolved_periodicity.name,
        "aggregation": aggregation,
        "clip_negatives": str(effective_clip),
        "criterion": criterion,
        "calibration_year_initial": str(first_year),
        "calibration_year_final": str(last_year),
        "climate_indices_candidates": json.dumps(labels),
        "climate_indices_fitting_methods": json.dumps({d.value: _FITTING_METHODS[d] for d in chosen}),
        "climate_indices_criterion": criterion,
        "climate_indices_log_likelihood_at": "fitted_estimates",
    }
    provenance_parameters = {
        **{key: value for key, value in calculation_metadata.items() if not key.startswith("climate_indices_")},
        "clip_negatives": effective_clip,
        "candidates": labels,
        "fitting_methods": {d.value: _FITTING_METHODS[d] for d in chosen},
        "log_likelihood_at": "fitted_estimates",
    }

    def declare(field: str, *, long_name: str | None = None, transform: Any = None) -> Any:
        metadata = dict(_FIELD_METADATA.get(field, {}))
        if long_name is not None:
            metadata = {
                "long_name": long_name,
                "units": "1",
                "description": f"{long_name}, per calendar step and cell.",
            }
        return _SpatialOutput(
            period_dim,
            metadata,
            "Distribution comparison",
            coordinate=steps,
            transform=transform,
            drop_attrs=("units",) if field in _UNITLESS_FIELDS else (),
        )

    def integer(array: xr.DataArray) -> xr.DataArray:
        return array.astype("int64")

    def boolean(array: xr.DataArray) -> xr.DataArray:
        return array.astype(bool)

    transforms = {"n_parameters": integer, "fit_failed": boolean}
    outputs = (
        *(declare(field, transform=transforms.get(field)) for field in _CANDIDATE_FIELDS for _ in chosen),
        *(declare(name, long_name=_PARAMETER_METADATA[name]) for name, _ in slots),
        declare("n", transform=integer),
        declare("best", transform=_label_best(labels)),
        declare("delta_to_runner_up"),
    )
    inputs = (input_da,) if pet_da is None else (input_da, pet_da)
    arrays = _spatial_kernel_call(
        functools.partial(_comparison_block, chosen=chosen, slots=slots),
        inputs,
        kernel_kwargs=provided,
        # a single cell axis keeps the per-cell interpretation, as fit_diagnostics does
        spatial_kernel=input_da.ndim != 2,
        calendar_plan=calendar_plan,
        outputs=outputs,
        calculation_metadata=calculation_metadata,
        output_provenance=_OutputProvenance(
            inputs={"values": input_da} if pet_da is None else {"values": input_da, "pet_mm": pet_da},
            parameters=provenance_parameters,
            implementation="climate_indices.suitability.compare_distributions",
        ),
        completion_event="compare_distributions_completed",
        completion_fields={"period_dim": period_dim, "candidates": labels},
        time_dim=time_dim,
    )

    data_vars: dict[str, xr.DataArray] = {}
    cursor = 0
    for field in _CANDIDATE_FIELDS:
        data_vars[field] = xr.concat(arrays[cursor : cursor + len(chosen)], dim=_candidate_axis(labels))
        cursor += len(chosen)
    for name in dict.fromkeys(name for name, _ in slots):
        present = [str(distribution.value) for slot_name, distribution in slots if slot_name == name]
        parts = arrays[cursor : cursor + len(present)]
        cursor += len(present)
        data_vars[name] = xr.concat(parts, dim=_candidate_axis(present)).reindex(candidate=_candidate_axis(labels))
    for field in _STEP_FIELDS:
        data_vars[field] = arrays[cursor]
        cursor += 1
    return xr.Dataset(data_vars)

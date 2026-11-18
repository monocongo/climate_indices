"""Distribution-suitability assessment tools (Beta, ADR-0028).

Which distribution should an index use for a region, season and Timescale? This module
gathers relative-fit evidence without changing any index: :func:`compare_distributions`
scores every likelihood-based candidate on one shared calibration sample and returns
log-likelihood, AIC, AICc, BIC and the best candidate for every calendar step (and cell).
:func:`drought_category` and :func:`compare_categories` then show whether the choice changes
the U.S. Drought Monitor class an index value falls in.

The criteria are **penalized likelihoods at the library's own estimates**: gamma by Thom's
(1958) closed-form approximation, Pearson Type III, log-logistic and GEV by L-moments. They are
not maximum-likelihood estimates in general, so a ranking mixes how well a family fits with how
efficient its estimator is, and the ``aic``, ``aicc`` and ``bic`` values are not textbook AIC,
AICc and BIC.
The module is Beta and has no root re-export: import it explicitly,
``import climate_indices.suitability as suitability``.
"""

from __future__ import annotations

import dataclasses
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
    DistributionFittingError,
    InvalidArgumentError,
    ShortCalibrationWarning,
)
from climate_indices.vocabulary import OUTPUT_SCALES, Aggregation, Distribution

__all__ = [
    "DROUGHT_CATEGORY_LABELS",
    "CategoryComparison",
    "Criterion",
    "DistributionComparison",
    "IndexRoute",
    "compare_categories",
    "compare_distributions",
    "drought_category",
]

Criterion = Literal["aicc", "aic", "bic"]
IndexRoute = Literal["spi", "spei", "standardized_index"]

_BETA_MESSAGE = "suitability is beta: its interface may change in a minor release (ADR-0028)."

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
    pet_mm: np.ndarray | None,
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
    values: np.ndarray,
    scale: int,
    *,
    index: IndexRoute,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    candidates: Sequence[Distribution] | None = None,
    pet_mm: np.ndarray | None = None,
    aggregation: Aggregation = "sum",
    clip_negatives: bool = True,
    criterion: Criterion = "aicc",
    spatial_time_major: bool = False,
) -> DistributionComparison:
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

    Because the likelihood is taken at Thom's (1958) closed-form gamma approximation and at L-moment
    estimates, not at maximum-likelihood estimates, the criteria are penalized likelihoods at fitted
    estimates and a ranking mixes distribution fit with estimator efficiency. About 30 values
    per calendar step discriminate weakly, so treat ``best`` as evidence, not as "correct".
    A sample value outside a candidate's fitted support gives zero likelihood, so that candidate
    is reported in ``fit_failed`` and does not rank; ``best`` can then have only one eligible
    candidate, and ``delta_to_runner_up`` is NaN. A GEV with shape above 1 has a density that is
    unbounded at the finite end of its support, where likelihood methods are not regular, so
    treat its criteria with caution there.

    Args:
        values: Precipitation, or the series ``standardized_index`` would standardize: 1-D, 2-D
            ``(years, periods)``, or a time-major block ``(time, *cells)`` with
            ``spatial_time_major=True``.
        scale: Timescale in time steps accumulated before fitting.
        index: ``"spi"``, ``"spei"`` or ``"standardized_index"``.
        data_start_year: First year of ``values``.
        calibration_year_initial: First calibration year; clamped to the record.
        calibration_year_final: Last calibration year; clamped to the record.
        periodicity: Monthly or daily.
        candidates: Families to compare; ``None`` selects gamma and Pearson Type III, plus
            log-logistic and GEV for ``"spei"``. Naming any other family raises.
        pet_mm: Potential evapotranspiration matching ``values``; required for, and only for,
            ``index="spei"``. A spatial block also accepts a per-step series or singleton cell
            axes; a shape that would broadcast over a larger cell grid raises.
        aggregation: ``"sum"`` or ``"mean"`` window aggregation; ``"standardized_index"`` only.
        clip_negatives: Clip negative input to zero. ``False`` rejects negatives, as
            ``standardized_index`` does for gamma and Pearson; ``"standardized_index"`` only.
        criterion: Criterion ``best`` selects on.
        spatial_time_major: Read a three-or-more-dimensional ``values`` as independent series
            shaped ``(time, *cells)``.

    Returns:
        A frozen :class:`DistributionComparison`, whose provenance records ``data_start_year``
        along with the effective calibration bounds to identify the shared sample.

    Raises:
        InvalidArgumentError: For an unsupported route, criterion, candidate, hourly
            periodicity, non-numeric or infinite input, or an argument combination the route
            does not accept.
        ValueError: If ``values`` holds no valid value.

    Warns:
        BetaFeatureWarning: Always; the module is Beta.
        ShortCalibrationWarning: From the fits when the calibration period is under 30 years, and
            from this function when a calendar step or cell has fewer than 30 shared values within a
            period of 30 years or more; ``actual_years`` then holds the smallest step or cell count.
        Warnings the fits raise (``CalibrationPeriodClampedWarning``, ``MissingDataWarning``, ``GoodnessOfFitWarning``)
            reach the caller as they do from the index; each candidate's fit can raise them.
        ClimateIndicesWarning: If ``scale`` exceeds 12 (monthly) or 365 (daily) steps, where
            consecutive years share overlapping windows and effective sample sizes shrink.
    """
    warnings.warn(_BETA_MESSAGE, BetaFeatureWarning, stacklevel=2)
    _validate_arguments(index, criterion, pet_mm, aggregation, clip_negatives)
    chosen = _resolve_candidates(index, candidates)
    _indices._validate_periodicity(periodicity)
    _indices._validate_scale(scale, periodicity)

    values = _as_float_array(values)
    prepared, clip = _prepare_values(values, index, pet_mm, clip_negatives)
    if compute.is_all_missing(prepared):
        raise ValueError("The values hold no valid value to compare distributions on.")

    scaled = compute.prepare_scaled(
        prepared,
        scale,
        periodicity,
        clip_negatives=clip,
        spatial_time_major=spatial_time_major,
        aggregation=aggregation,
    )
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
            stacklevel=2,
        )
    window_limit = 365 if periodicity is compute.Periodicity.daily else 12
    if scale > window_limit:
        warnings.warn(
            f"Timescale {scale} exceeds {window_limit} steps, so accumulation windows ending on the same "
            "calendar step in consecutive years overlap; effective sample sizes shrink and goodness-of-fit "
            "p-values are invalid.",
            ClimateIndicesWarning,
            stacklevel=2,
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


# ---------------------------------------------------------------------------
# Drought categories and category-change metrics (#1587, ADR-0028 section 7)
# ---------------------------------------------------------------------------

#: Class labels in code order. Code 0 is "none", codes 1 to 5 are D0 to D4, and -1 marks a missing value.
DROUGHT_CATEGORY_LABELS: tuple[str, ...] = ("none", "D0", "D1", "D2", "D3", "D4")
_N_CATEGORIES = len(DROUGHT_CATEGORY_LABELS)

# class boundaries in ascending order; each class includes its upper edge and excludes its lower one
_USDM_EDGES = (-2.0, -1.6, -1.3, -0.8, -0.5)
_USDM_SOURCE = (
    "U.S. Drought Monitor drought classification, SPI/SPEI column "
    "(https://droughtmonitor.unl.edu/About/AbouttheData/DroughtClassification.aspx). The table gives -0.5 to -0.79 "
    "for D0, -0.8 to -1.29 for D1, -1.3 to -1.59 for D2, -1.6 to -1.99 for D3, -2.0 or less for D4 and -0.49 or "
    "above for none; it does not state boundary handling, so each class includes its upper edge (ADR-0028)."
)


def _category_attributes() -> dict[str, str]:
    return {
        "climate_indices_category_edges": json.dumps(list(_USDM_EDGES)),
        "climate_indices_category_source": _USDM_SOURCE,
        "climate_indices_version": __version__,
    }


def _classify(values: np.ndarray) -> np.ndarray:
    """USDM class codes of normal-scale values: NaN (or a mask) is -1, everything else 0 to 5."""
    if np.ma.isMaskedArray(values) and values.dtype.kind == "f":
        values = np.ma.filled(values, np.nan)  # keeps float32 and float16, which _as_float_array would widen
    x = _as_float_array(values)
    # the edges take the input's width, so a float32 -1.3 sits on the edge instead of just above it
    edges = np.asarray(_USDM_EDGES, dtype=x.dtype)
    codes = len(edges) - np.searchsorted(edges, x, side="left")
    return np.where(np.isnan(x), -1, codes).astype(np.int8)


def drought_category(index_values: Any, *, output_scale: Any) -> Any:
    """Classify normal-scale SPI/SPEI values into the U.S. Drought Monitor classes.

    The classes follow the USDM table for the Standardized Precipitation Index and the
    Standardized Precipitation-Evapotranspiration Index. Each class includes its upper edge, so
    -0.5 is D0, -0.8 is D1, -1.3 is D2, -1.6 is D3 and -2.0 is D4, while -0.49 is none (ADR-0028).
    Wet values are all "none"; no wet-side classes are defined.

    Args:
        index_values: Index values on the normal scale, a NumPy array (or masked array) or an
            ``xarray.DataArray``, which may be Dask-backed. A value clipped at +-3.09 classifies
            normally; an infinity is rejected (for a Dask-backed input, when the result is computed).
        output_scale: The scale ``index_values`` is on. Required, with no default, because values alone
            cannot show whether they are probabilities or normal-scale. Only ``"normal"`` classifies;
            ``"probability"`` and ``"bounded"`` raise.

    Returns:
        ``int8`` codes in the shape of the input: -1 missing (NaN or masked), 0 none, 1 to 5 for
        D0 to D4 (see :data:`DROUGHT_CATEGORY_LABELS`). A DataArray comes back as a DataArray with
        the edges, their source and the provenance record in its attributes.

    Raises:
        InvalidArgumentError: ``output_scale`` is not ``"normal"``, or the values are not numeric or
            hold an infinity.

    Warns:
        BetaFeatureWarning: Always; the module is Beta.
    """
    if not isinstance(output_scale, str) or output_scale != "normal":
        known = isinstance(output_scale, str) and output_scale in OUTPUT_SCALES
        detail = "convert the index to the normal scale first" if known else f"expected one of {OUTPUT_SCALES}"
        raise _invalid(
            f"drought_category classifies normal-scale values only, got output_scale={output_scale!r}; {detail}.",
            "output_scale",
            output_scale,
            "'normal'",
        )
    warnings.warn(_BETA_MESSAGE, BetaFeatureWarning, stacklevel=2)
    if not isinstance(index_values, xr.DataArray):
        return _classify(index_values)
    codes = xr.apply_ufunc(_classify, index_values, dask="parallelized", output_dtypes=[np.int8])
    codes.attrs = {
        "long_name": "U.S. Drought Monitor drought category",
        "units": "1",
        "flag_values": [-1, *range(_N_CATEGORIES)],
        "flag_meanings": " ".join(["missing", *DROUGHT_CATEGORY_LABELS]),
        **_category_attributes(),
        **provenance.provenance_attributes(
            index="drought category",
            implementation="climate_indices.suitability.drought_category",
            interface="python",
            parameters={"output_scale": output_scale},
            inputs={"index_values": index_values},
        ),
    }
    return codes


@dataclasses.dataclass(frozen=True, eq=False)
class CategoryComparison:
    """How many drought categories change between two classifications of the same values.

    Only pairs where both categories are present count; a pair with a missing category is left out of every
    field. A change is a pair whose two categories differ.
    """

    #: Pair counts, ``(6, 6)`` ``int64``: row = the reference class, column = the alternative class, in
    #: :data:`DROUGHT_CATEGORY_LABELS` order.
    confusion: np.ndarray
    #: Valid pairs per reference class, ``(6,)``: the row sums of ``confusion``.
    reference_counts: np.ndarray
    #: Valid pairs per alternative class, ``(6,)``: the column sums of ``confusion``.
    alternative_counts: np.ndarray
    #: Number of valid pairs.
    n_valid: int
    #: Share of valid pairs whose category changed; NaN with no valid pair.
    share_changed: float
    #: Changed pairs over valid pairs where at least one side is in drought (D0 to D4); NaN when none is.
    #: Neither input is privileged: a pair that moves into or out of drought counts as in drought.
    share_changed_in_drought: float
    #: The category edges, their source and the provenance record.
    metadata: Mapping[str, str]


def _require_integer_codes(dtype: np.dtype[Any], name: str) -> None:
    if dtype.kind not in "iu":
        raise _invalid(
            f"{name} must hold integer category codes from drought_category, got dtype {dtype}.",
            name,
            dtype,
            "integer codes -1 to 5",
        )


def _require_code_range(codes: np.ndarray, name: str) -> None:
    # a stray code would otherwise land in another cell of the flattened confusion matrix
    if codes.size and (codes.min() < -1 or codes.max() >= _N_CATEGORIES):
        raise _invalid(
            f"{name} holds a category code outside -1 to {_N_CATEGORIES - 1}.", name, "out of range", "codes -1 to 5"
        )


def _as_codes(values: Any, name: str) -> np.ndarray:
    """Validated integer category codes as an array; a masked code becomes -1 (missing)."""
    codes = values if np.ma.isMaskedArray(values) else np.asarray(values)
    _require_integer_codes(codes.dtype, name)
    if np.ma.isMaskedArray(codes):
        # range-check before the int64 cast, which would wrap a large unmasked uint64 into -1 (missing)
        _require_code_range(codes.compressed(), name)
        return np.ma.filled(codes.astype(np.int64), -1)
    _require_code_range(codes, name)
    return codes


def _share(changed: Any, population: Any) -> Any:
    """``changed / population``, NaN where the population is empty."""
    if isinstance(population, xr.DataArray):
        return changed / population.where(population > 0)
    return float(changed / population) if population > 0 else float("nan")


def _compare_categories_numpy(reference: Any, alternative: Any) -> CategoryComparison:
    ref, alt = _as_codes(reference, "reference"), _as_codes(alternative, "alternative")
    if ref.shape != alt.shape:
        raise _invalid(
            f"reference and alternative must have the same shape, got {ref.shape} and {alt.shape}.",
            "alternative",
            alt.shape,
            str(ref.shape),
        )
    valid = (ref >= 0) & (alt >= 0)
    pairs = ref[valid].astype(np.intp) * _N_CATEGORIES + alt[valid].astype(np.intp)
    counts = np.bincount(pairs, minlength=_N_CATEGORIES**2).astype(np.int64)
    confusion = counts.reshape(_N_CATEGORIES, _N_CATEGORIES)
    n_valid = int(confusion.sum())
    changed = n_valid - int(np.trace(confusion))
    metadata = {
        **_category_attributes(),
        **provenance.provenance_attributes(
            index="drought category comparison",
            implementation="climate_indices.suitability.compare_categories",
            interface="python",
            parameters={"dims": None},
            inputs={"reference": ref, "alternative": alt},
        ),
    }
    return CategoryComparison(
        confusion=confusion,
        reference_counts=confusion.sum(axis=1),
        alternative_counts=confusion.sum(axis=0),
        n_valid=n_valid,
        share_changed=_share(changed, n_valid),
        share_changed_in_drought=_share(changed, n_valid - int(confusion[0, 0])),
        metadata=metadata,
    )


def _reduction_dims(dims: Any, available: tuple[Any, ...]) -> tuple[Any, ...]:
    """The dimensions to reduce over: all of them for None, else the named, distinct, existing ones."""
    if dims is None:
        return available
    names = (dims,) if isinstance(dims, str) else dims
    try:
        chosen = tuple(names)
    except TypeError as error:
        raise _invalid(
            f"dims must be a dimension name or a sequence of names, got {dims!r}.", "dims", dims, "dimension names"
        ) from error
    if not chosen:
        raise _invalid("compare_categories needs at least one dimension to reduce.", "dims", dims, "dimension names")
    for dim in chosen:
        if dim not in available:
            raise _invalid(
                f"dims names {dim!r}, which is not a dimension of the inputs {available}.",
                "dims",
                dim,
                ", ".join(map(str, available)),
            )
    if len(set(chosen)) != len(chosen):
        raise _invalid(f"dims names a dimension twice: {chosen}.", "dims", dims, "distinct dimension names")
    return chosen


def _compare_categories_xarray(
    reference: xr.DataArray, alternative: xr.DataArray, dims: str | Sequence[str] | None
) -> xr.Dataset:
    for name, codes in (("reference", reference), ("alternative", alternative)):
        _require_integer_codes(codes.dtype, name)
        if codes.chunks is None:  # a lazy array is not read just to check its range
            _require_code_range(codes.values, name)
    if set(reference.dims) != set(alternative.dims):
        raise _invalid(
            f"reference and alternative must have the same dimensions, got {reference.dims} and {alternative.dims}.",
            "alternative",
            alternative.dims,
            str(reference.dims),
        )
    if set(reference.xindexes) != set(alternative.xindexes):  # align would lend one side's labels to the other
        raise _invalid(
            "reference and alternative must share their coordinates: only one has an index "
            f"for {sorted(set(reference.xindexes) ^ set(alternative.xindexes))}.",
            "alternative",
            "mismatch",
            "equal",
        )
    try:
        reference, alternative = xr.align(reference, alternative, join="exact")
    except ValueError as error:
        raise _invalid(
            f"reference and alternative must share their coordinates: {error}", "alternative", "mismatch", "equal"
        ) from error
    reduce_dims = _reduction_dims(dims, tuple(reference.dims))
    kept_clash = {"reference_category", "alternative_category"} & (set(reference.dims) - set(reduce_dims))
    if kept_clash:
        raise _invalid(
            f"{sorted(kept_clash)} are reserved for the confusion matrix and cannot stay as input dimensions; "
            "rename them or reduce over them.",
            "dims",
            sorted(kept_clash),
            "other dimension names",
        )

    # a code outside 0 to 5 (possible only for a lazy input, which is not range-checked) is excluded
    # everywhere, so n_valid, the shares and the confusion matrix always count the same pairs
    valid = (reference >= 0) & (reference < _N_CATEGORIES) & (alternative >= 0) & (alternative < _N_CATEGORIES)
    n_valid = valid.sum(reduce_dims)
    changed = n_valid - ((reference == alternative) & valid).sum(reduce_dims)
    n_none = ((reference == 0) & (alternative == 0)).sum(reduce_dims)
    # one reduction per cell of the matrix keeps the transient memory at the size of the input
    confusion = xr.concat(
        [
            xr.concat(
                [((reference == r) & (alternative == a)).sum(reduce_dims) for a in range(_N_CATEGORIES)],
                dim="alternative_category",
            )
            for r in range(_N_CATEGORIES)
        ],
        dim="reference_category",
    )
    labels = list(DROUGHT_CATEGORY_LABELS)
    confusion = confusion.assign_coords(reference_category=labels, alternative_category=labels).transpose(
        ..., "reference_category", "alternative_category"
    )
    return xr.Dataset(
        {
            "confusion": confusion,
            "reference_counts": confusion.sum("alternative_category"),
            "alternative_counts": confusion.sum("reference_category"),
            "n_valid": n_valid,
            "share_changed": _share(changed, n_valid),
            "share_changed_in_drought": _share(changed, n_valid - n_none),
        },
        attrs={
            **_category_attributes(),
            **provenance.provenance_attributes(
                index="drought category comparison",
                implementation="climate_indices.suitability.compare_categories",
                interface="python",
                parameters={"dims": list(reduce_dims)},
                inputs={"reference": reference, "alternative": alternative},
            ),
        },
    )


def compare_categories(
    reference: Any, alternative: Any, *, dims: str | Sequence[str] | None = None
) -> CategoryComparison | xr.Dataset:
    """Compare two drought-category classifications of the same values.

    Answers whether switching from one distribution to another changes the drought class managers
    see. Classify each index with :func:`drought_category`, then compare the codes. Only pairs with
    both categories present count.

    Args:
        reference: Category codes from :func:`drought_category`, an integer NumPy array or DataArray.
        alternative: Category codes of the same shape, or an identically indexed DataArray.
        dims: DataArray route only. The dimensions to reduce over, a name or a sequence of names;
            ``None`` reduces every dimension. The others stay as dimensions of the result, so reducing
            over ``"time"`` gives one confusion matrix per cell. The NumPy route always reduces every axis.

    Returns:
        A :class:`CategoryComparison` for NumPy input, or an ``xarray.Dataset`` for DataArrays with
        ``confusion`` (the two category dimensions last, labelled by :data:`DROUGHT_CATEGORY_LABELS`),
        ``reference_counts``, ``alternative_counts``, ``n_valid``, ``share_changed`` and
        ``share_changed_in_drought``, which stay lazy for Dask input. The shares are NaN where their
        population is empty.

    Raises:
        InvalidArgumentError: The codes are not integers from -1 to 5, the inputs differ in shape,
            dimensions or coordinates, ``dims`` is empty, repeats or names a missing dimension or is given
            for NumPy input, an input dimension that stays is named ``reference_category`` or
            ``alternative_category``, or only one input is a DataArray. A Dask-backed input is not read
            to check the range, so a code outside -1 to 5 in it is excluded from every field like a
            missing one. A masked NumPy code counts as missing.

    Warns:
        BetaFeatureWarning: Always; the module is Beta.
    """
    warnings.warn(_BETA_MESSAGE, BetaFeatureWarning, stacklevel=2)
    is_ref, is_alt = isinstance(reference, xr.DataArray), isinstance(alternative, xr.DataArray)
    if is_ref != is_alt:
        raise _invalid(
            "reference and alternative must both be NumPy arrays or both be xarray DataArrays.",
            "alternative",
            type(alternative).__name__,
            type(reference).__name__,
        )
    if is_ref:
        return _compare_categories_xarray(reference, alternative, dims)
    if dims is not None:
        raise _invalid(
            "dims applies to DataArray inputs only; NumPy inputs reduce over every axis.",
            "dims",
            dims,
            "None for NumPy input",
        )
    return _compare_categories_numpy(reference, alternative)

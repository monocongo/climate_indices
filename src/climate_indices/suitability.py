"""Distribution-suitability assessment tools (Beta, ADR-0028).

Which distribution should an index use for a region, season and Timescale? This module
gathers relative-fit evidence without changing any index: :func:`compare_distributions`
scores every likelihood-based candidate on one shared calibration sample and returns
log-likelihood, AIC, AICc, BIC and the best candidate for every calendar step (and cell).

The criteria are **penalized likelihoods at the library's own estimates**: gamma by Thom's
approximation, Pearson Type III and log-logistic by L-moments. They are not maximum-likelihood
estimates in general, so a ranking mixes how well a family fits with how efficient its
estimator is, and the ``aic``, ``aicc`` and ``bic`` values are not textbook AIC, AICc and BIC.
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
from climate_indices.vocabulary import Aggregation, Distribution

__all__ = ["Criterion", "DistributionComparison", "IndexRoute", "compare_distributions"]

Criterion = Literal["aicc", "aic", "bic"]
IndexRoute = Literal["spi", "spei", "standardized_index"]

_BETA_MESSAGE = "suitability is beta: its interface may change in a minor release (ADR-0028)."

# candidates per route in canonical Distribution enum order, which is also the tie-break order
_ROUTE_CANDIDATES: dict[str, tuple[Distribution, ...]] = {
    "spi": (Distribution.pearson, Distribution.gamma),
    "standardized_index": (Distribution.pearson, Distribution.gamma),
    "spei": (Distribution.pearson, Distribution.gamma, Distribution.loglogistic),
}

# free parameters per family, not counting the zero mass
_FAMILY_PARAMETERS = {Distribution.pearson: 3, Distribution.gamma: 2, Distribution.loglogistic: 3}

# families whose fit carries a zero mass p0; the log-logistic has none (ADR-0016)
_ZERO_MASS_FAMILIES = frozenset({Distribution.pearson, Distribution.gamma})

_FITTING_METHODS = {
    Distribution.pearson: "l_moments",
    Distribution.gamma: "thom_mle_approximation",
    Distribution.loglogistic: "l_moments",
}

# the log-logistic transform treats a shape within this of zero as the logistic limit
_GLO_SHAPE_TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True)
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
    parameters: Mapping[Distribution, Mapping[str, np.ndarray]]
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


def _validate_arguments(
    index: str,
    criterion: str,
    pet_mm: np.ndarray | None,
    aggregation: Aggregation,
    clip_negatives: bool,
) -> None:
    if not isinstance(index, str) or index not in _ROUTE_CANDIDATES:
        raise _invalid(f"Unsupported index route: {index!r}.", "index", index, ", ".join(get_args(IndexRoute)))
    if criterion not in get_args(Criterion):
        raise _invalid(f"Unsupported criterion: {criterion!r}.", "criterion", criterion, ", ".join(get_args(Criterion)))
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


def _as_float_array(values: np.ndarray) -> np.ndarray:
    """A masked array as NaN-filled, and an integer or boolean array as float, which the fits require."""
    if np.ma.isMaskedArray(values):
        return np.asarray(np.ma.filled(values.astype(float), np.nan))
    array = np.asarray(values)
    return array.astype(float) if array.dtype.kind in "iub" else array


def _prepare_values(
    values: np.ndarray, index: IndexRoute, pet_mm: np.ndarray | None, clip_negatives: bool
) -> tuple[np.ndarray, bool]:
    """Prepare the input and clipping policy for an already validated route."""
    if index == "spei" and pet_mm is not None:
        return _indices._spei_water_balance(values, _as_float_array(pet_mm)), False
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
        else:
            density = _glo_logpdf(sample, p["loc"], p["scale"], p["shape"])
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
    else:
        loc, scale, shape = p["loc"], p["scale"], p["shape"]
        ok = np.isfinite(loc) & np.isfinite(scale) & np.isfinite(shape) & (scale > 0) & (np.abs(shape) < 1.0)
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
    convention and does not change the sample. SPEI's log-logistic candidate scores zeros as
    continuous observations when they lie within its fitted support (ADR-0016).
    The Pearson-to-gamma fallback is disabled: a failed fit is recorded in ``fit_failed``,
    not replaced.

    Because the likelihood is taken at Thom's gamma approximation and at L-moment estimates,
    not at maximum-likelihood estimates, the criteria are penalized likelihoods at fitted
    estimates and a ranking mixes distribution fit with estimator efficiency. About 30 values
    per calendar step discriminate weakly, so treat ``best`` as evidence, not as "correct".

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
            log-logistic for ``"spei"``. Naming any other family raises.
        pet_mm: Potential evapotranspiration matching ``values``; required for, and only for,
            ``index="spei"``.
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
            periodicity, or an argument combination the route does not accept.
        ValueError: If ``values`` holds no valid value.

    Warns:
        BetaFeatureWarning: Always; the module is Beta.
        ShortCalibrationWarning: From the fits when the calibration period is under 30 years, and
            from this function when a calendar step or cell has fewer than 30 shared values within a
            period of 30 years or more.
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

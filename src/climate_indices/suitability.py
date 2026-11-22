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
import functools
import hashlib
import inspect
import json
import math
import secrets
import warnings
from collections.abc import Callable, Mapping, Sequence
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
from climate_indices.vocabulary import OUTPUT_SCALES, Aggregation, Distribution
from climate_indices.xarray_adapter import (
    _align_inputs,
    _effective_calibration_years,
    _fitted_parameter_output,
    _infer_data_start_year,
    _infer_temporal_parameters,
    _OutputProvenance,
    _resolve_calendar_plan,
    _spatial_kernel_call,
    _SpatialOutput,
    _validate_sufficient_data,
    _validate_supported_calendar,
)

__all__ = [
    "DROUGHT_CATEGORY_LABELS",
    "CategoryComparison",
    "Criterion",
    "DistributionComparison",
    "GofStatistic",
    "GoodnessOfFit",
    "IndexRoute",
    "compare_categories",
    "compare_distributions",
    "drought_category",
]

Criterion = Literal["aicc", "aic", "bic"]
IndexRoute = Literal["spi", "spei", "standardized_index"]
GofStatistic = Literal["ks", "ad"]

_BETA_MESSAGE = "suitability is beta: its interface may change in a minor release (ADR-0028)."

# the caller's frame for a warning raised outside the public function's own call chain
_CALLER_STACKLEVEL = 3


def _caller_stacklevel() -> int:
    """Frames from the calling warning site to the user's call site, across either route.

    The xarray route reaches the core through ``xr.apply_ufunc``, so a fixed level cannot
    name the caller: find ``compare_distributions`` and take the frame above it.
    """
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None
    level = 1
    while frame is not None and frame.f_code.co_name != "compare_distributions":
        level += 1
        frame = frame.f_back
    return level + 1 if frame is not None else _CALLER_STACKLEVEL


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
class GoodnessOfFit:
    """Per-candidate goodness-of-fit evidence from the probability integral transform (PIT).

    The statistics test the continuous part of each fit. For gamma and Pearson Type III, whose fit carries a zero
    mass, exact zeros are excluded because they are the mass, not observations of the continuous part. Log-logistic and
    GEV have no zero mass, so a zero is an ordinary value there and is kept. Repeated values are kept with their
    frequency. Shapes follow :class:`DistributionComparison`: ``(candidate, step, *cell)``, NaN where a candidate has
    no usable fit (``fit_failed``) or no value remains. A statistic not named in ``gof`` is not computed and its
    arrays are NaN.

    Both statistics are evaluated at parameters estimated from the same sample, so a naive p-value is
    uncalibrated; the parametric bootstrap (``n_bootstrap > 0``) refits every simulated sample with the
    library's own estimator and gives calibrated p-values. The p-value of the candidate that ``best``
    selects is a post-selection statistic, not an ordinary confirmatory test.
    """

    #: Kolmogorov-Smirnov D of the PIT against the uniform distribution.
    ks_statistic: np.ndarray
    #: Naive K-S p-value, the exact null distribution of D for a sample of fully specified parameters.
    #: It is uncalibrated for estimated parameters. From 7 values on it is tabulated per sample size and
    #: interpolated to within 1e-5; fewer take SciPy's exact value.
    ks_p_value_naive: np.ndarray
    #: Anderson-Darling A^2 of the PIT. It has no naive p-value.
    ad_statistic: np.ndarray
    #: Bootstrap K-S p-value ``(1 + #{D* >= D}) / (B_ok + 1)``; NaN where no refit succeeded. ``None``
    #: unless ``n_bootstrap > 0``.
    ks_p_value_bootstrap: np.ndarray | None
    #: Bootstrap Anderson-Darling p-value, defined like ``ks_p_value_bootstrap``.
    ad_p_value_bootstrap: np.ndarray | None
    #: Benjamini-Hochberg adjustment of the K-S p-value across cells (bootstrap when ``n_bootstrap > 0``,
    #: else naive), separately for every candidate and step. ``None`` unless ``bh_adjust``.
    ks_p_value_bh: np.ndarray | None
    #: Benjamini-Hochberg adjustment of ``ad_p_value_bootstrap``. ``None`` unless ``bh_adjust`` and
    #: ``n_bootstrap > 0``.
    ad_p_value_bh: np.ndarray | None
    #: Share of cells with ``p <= alpha``, ``(candidate, step, 2)`` over the statistics ``("ks", "ad")``,
    #: from the bootstrap p-value when ``n_bootstrap > 0`` and else from the naive K-S p-value (AD has
    #: none, so NaN). Cells without a p-value are not counted. ``None`` without a cell axis.
    rejection_share: np.ndarray | None
    #: Bootstrap replicates requested per candidate, step and cell.
    n_bootstrap: int
    #: Share of the ``n_bootstrap`` refits that failed and were left out of the p-value,
    #: ``(candidate, step, *cell)``. ``None`` unless ``n_bootstrap > 0``.
    bootstrap_failed_share: np.ndarray | None = None


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
    #: Goodness of fit of every candidate (:class:`GoodnessOfFit`), or ``None`` when ``gof`` is empty.
    goodness_of_fit: GoodnessOfFit | None
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


# ---- goodness of fit (SUIT-05, #1586) -------------------------------------------------------------

_GOF_STATISTICS: tuple[GofStatistic, ...] = ("ks", "ad")

# a PIT value is held this far inside (0, 1), so a sample at the edge of a fitted support scores a large
# but finite A^2 instead of infinity
_PIT_MARGIN = 1e-12

# the exact K-S p-value is tabulated against t = D sqrt(m); beyond this t it is below 1e-30
_KS_TABLE_LIMIT = 6.0
_KS_TABLE_POINTS = 2001
# fewer values than this take SciPy's exact K-S value: the table's interpolation error reaches 1e-3 there
_KS_EXACT_BELOW = 7

# elements of one stacked bootstrap batch: it bounds memory and never changes a result. A million holds peak memory
# near 330 MB on a 100-cell, 40-year block, and is as fast as four times that
_BOOTSTRAP_BATCH_ELEMENTS = 1_000_000


def _glo_cdf(x: np.ndarray, loc: np.ndarray, scale: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Hosking's generalized-logistic CDF, ``1 / (1 + exp(-y))``; 0 or 1 beyond the support's end the value is on."""
    with np.errstate(all="ignore"):
        logistic = np.abs(shape) <= _GLO_SHAPE_TOLERANCE
        k = np.where(logistic, 0.0, shape)
        z = (x - loc) / scale
        argument = 1.0 - k * z
        inside = logistic | (argument > 0.0)
        y = np.where(logistic, z, -np.log(np.where(argument > 0.0, argument, 1.0)) / np.where(logistic, 1.0, k))
        probability = np.where(inside, scipy.special.expit(y), np.where(k > 0.0, 1.0, 0.0))
    result: np.ndarray = np.where(np.isnan(x), np.nan, probability)
    return result


def _glo_ppf(u: np.ndarray, loc: np.ndarray, scale: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_glo_cdf`: ``loc + scale * (1 - exp(-k y)) / k`` with ``y = logit(u)``."""
    with np.errstate(all="ignore"):
        logistic = np.abs(shape) <= _GLO_SHAPE_TOLERANCE
        k = np.where(logistic, 0.0, shape)
        y = scipy.special.logit(u)
        z = np.where(logistic, y, -np.expm1(-k * y) / np.where(logistic, 1.0, k))
        quantile: np.ndarray = loc + scale * z
    return quantile


def _continuous_cdf(distribution: Distribution, x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    """CDF of the continuous part of one family at its fitted parameters; NaN where ``x`` is NaN."""
    with np.errstate(all="ignore"):
        if distribution is Distribution.gamma:
            probability: np.ndarray = scipy.stats.gamma.cdf(x, a=p["alpha"], scale=p["beta"])
        elif distribution is Distribution.pearson:
            probability = scipy.stats.pearson3.cdf(x, p["skew"], loc=p["loc"], scale=p["scale"])
        elif distribution is Distribution.loglogistic:
            probability = _glo_cdf(x, p["loc"], p["scale"], p["shape"])
        elif distribution is Distribution.gev:
            probability = scipy.stats.genextreme.cdf(x, p["shape"], loc=p["loc"], scale=p["scale"])
        else:
            raise ValueError(f"{distribution} has no likelihood in suitability")
    return probability


def _continuous_ppf(distribution: Distribution, u: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    """Quantile function of the continuous part of one family at its fitted parameters."""
    with np.errstate(all="ignore"):
        if distribution is Distribution.gamma:
            quantile: np.ndarray = scipy.stats.gamma.ppf(u, a=p["alpha"], scale=p["beta"])
        elif distribution is Distribution.pearson:
            quantile = scipy.stats.pearson3.ppf(u, p["skew"], loc=p["loc"], scale=p["scale"])
        elif distribution is Distribution.loglogistic:
            quantile = _glo_ppf(u, p["loc"], p["scale"], p["shape"])
        elif distribution is Distribution.gev:
            quantile = scipy.stats.genextreme.ppf(u, p["shape"], loc=p["loc"], scale=p["scale"])
        else:
            raise ValueError(f"{distribution} has no likelihood in suitability")
    return quantile


def _pit_statistics(pit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kolmogorov-Smirnov D, Anderson-Darling A^2 and the count of finite PIT values, along axis 0.

    ``D = max_i max(i/m - u_(i), u_(i) - (i-1)/m)`` and
    ``A^2 = -m - (1/m) sum_i (2i - 1) (ln u_(i) + ln(1 - u_(m+1-i)))`` over the ``m`` ordered finite
    values, each held within ``_PIT_MARGIN`` of 0 and 1. Both are NaN where ``m`` is 0.
    """
    finite = np.isfinite(pit)
    count = finite.sum(axis=0)
    ordered = np.sort(np.where(finite, np.clip(pit, _PIT_MARGIN, 1.0 - _PIT_MARGIN), np.nan), axis=0)
    rank = np.arange(1, pit.shape[0] + 1).reshape((-1,) + (1,) * (pit.ndim - 1))
    inside = rank <= count
    size = np.maximum(count, 1)
    with np.errstate(all="ignore"):
        d_plus = np.where(inside, rank / size - ordered, -np.inf).max(axis=0)
        d_minus = np.where(inside, ordered - (rank - 1) / size, -np.inf).max(axis=0)
        mirrored = np.take_along_axis(ordered, np.clip(count - rank, 0, pit.shape[0] - 1), axis=0)
        terms = np.where(inside, (2 * rank - 1) * (np.log(ordered) + np.log1p(-mirrored)), 0.0).sum(axis=0)
        anderson_darling = -count - terms / size
    return (
        np.where(count > 0, np.maximum(d_plus, d_minus), np.nan),
        np.where(count > 0, anderson_darling, np.nan),
        count,
    )


@functools.lru_cache(maxsize=512)
def _ks_table(size: int) -> tuple[np.ndarray, np.ndarray]:
    """The exact K-S survival function of ``D`` for ``size`` observations, on a grid of ``t = D sqrt(size)``."""
    t = np.linspace(0.0, _KS_TABLE_LIMIT, _KS_TABLE_POINTS)
    return t, np.clip(scipy.stats.kstwo.sf(t / math.sqrt(size), size), 0.0, 1.0)


def _ks_naive_p_value(statistic: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Naive two-sided K-S p-value of ``statistic`` for ``count`` fully specified observations.

    SciPy's exact distribution costs about 35 microseconds a value and every cell and step needs one,
    so from ``_KS_EXACT_BELOW`` values on it is evaluated once per sample size on a fine grid and interpolated to
    within 1e-5. Fewer values are rare, and the survival function has kinks that the grid would smooth over
    (an error of 1.3e-3 for one value), so they take SciPy's exact value. NaN where ``count`` is 0 or ``statistic``
    is NaN.
    """
    p_value = np.full(statistic.shape, np.nan)
    for size in np.unique(count[count > 0]):
        picked = (count == size) & np.isfinite(statistic)
        if size < _KS_EXACT_BELOW:
            p_value[picked] = np.clip(scipy.stats.kstwo.sf(statistic[picked], int(size)), 0.0, 1.0)
        else:
            t, table = _ks_table(int(size))
            p_value[picked] = np.interp(statistic[picked] * math.sqrt(size), t, table)
    return p_value


def _sample_pit(
    distribution: Distribution, sample: np.ndarray, valid: np.ndarray, p: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The probability integral transform of a sample, and the values it covers.

    A family with a zero mass covers only its positive values (a zero is the mass, not an observation of the
    continuous part). A family without covers every valid value, as :func:`_score_fit` does, so a zero is an ordinary
    value there. The statistics and the support check use exactly these values.
    """
    covered = valid & (sample != 0.0) if distribution in _ZERO_MASS_FAMILIES else valid  # NOSONAR (S1244) exact zeros
    return _continuous_cdf(distribution, np.where(covered, sample, np.nan), p), covered


def _observed_statistics(
    distribution: Distribution,
    sample: np.ndarray,
    valid: np.ndarray,
    p: dict[str, np.ndarray],
    scored: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-S, A^2 and the number of values they cover for the shared sample against one fitted candidate."""
    pit, _ = _sample_pit(distribution, sample, valid, p)
    return _pit_statistics(np.where(scored, pit, np.nan))


def _column_generators(seed: int, sample: np.ndarray) -> list[np.random.Generator]:
    """One random stream per calendar step and cell, keyed by ``seed`` and that column's own values.

    Keying a stream on the data, not on the column's position, makes the simulated samples the same
    whether a cell is run alone, in a block of a Dask graph or in a full grid, so a chunking cannot change
    a p-value. Two columns with identical values share a stream, which leaves each p-value valid.
    """
    # one byte pattern per value: every NaN alike (its sign bit and payload vary by platform), -0.0 as 0.0, little-endian
    columns = np.ascontiguousarray(
        (np.where(np.isnan(sample), np.nan, sample) + 0.0).reshape(sample.shape[0], -1).T.astype("<f8")
    )
    return [
        np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [seed, int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=16).digest(), "little")]
                )
            )
        )
        for row in columns
    ]


def _draw_uniforms(
    generators: list[np.random.Generator], years: int, cell_shape: tuple[int, ...], size: int
) -> np.ndarray:
    """``size`` more uniform samples of ``years`` values for every column, as ``(years, *cell_shape, size)``.

    Every generator advances by ``size * years`` draws, so the replicates do not depend on how they are batched.
    """
    drawn = np.empty((len(generators), size, years))
    for column, generator in enumerate(generators):
        drawn[column] = generator.random((size, years))
    return np.moveaxis(drawn.reshape(*cell_shape, size, years), -1, 0)


def _simulate(
    distribution: Distribution,
    uniforms: np.ndarray,
    p: dict[str, np.ndarray],
    resolution: float | None,
    origin: float = 0.0,
) -> np.ndarray:
    """Draw from the fitted mixed distribution by inverting its CDF at ``uniforms``.

    A zero-mass family returns an exact zero where ``u <= p0`` and otherwise the continuous quantile
    at ``(u - p0) / (1 - p0)``, so the zero mass is part of every simulated sample. A ``resolution`` rounds the draws
    to that unit, which gives the simulated sample the observed record's tie structure. The values lie on the lattice
    ``origin + k * resolution``: the prepared SPEI sample carries an offset (``+1000`` per summed step), so its
    multiples of the unit start at ``origin``, not at zero. The library fits p0 on the rounded record, so p0 already
    counts the values that rounded to zero: a positive draw that rounds to zero or below would add zeros beyond p0, and
    is set to the smallest positive lattice value instead. Every nonzero value is then at least that, as in the observed
    record, and the simulated zero share stays p0.
    """
    with np.errstate(all="ignore"):
        if distribution in _ZERO_MASS_FAMILIES:
            p0 = p["prob_zero"]
            positive = uniforms > p0
            conditional = np.clip((uniforms - p0) / (1.0 - p0), _PIT_MARGIN, 1.0 - _PIT_MARGIN)
            draws = np.where(positive, _continuous_ppf(distribution, conditional, p), 0.0)
            if resolution is not None:
                lattice = np.round((draws - origin) / resolution) * resolution + origin
                smallest = (origin % resolution) or resolution  # the least positive point of the lattice
                draws = np.where(positive, np.where(lattice <= 0.0, smallest, lattice), 0.0)
        else:
            draws = _continuous_ppf(distribution, np.clip(uniforms, _PIT_MARGIN, 1.0 - _PIT_MARGIN), p)
            if resolution is not None:
                draws = np.round((draws - origin) / resolution) * resolution + origin
    result: np.ndarray = draws
    return result


def _refit_statistics(
    distribution: Distribution, draws: np.ndarray, period: Any, periodicity: compute.Periodicity
) -> tuple[dict[GofStatistic, np.ndarray], np.ndarray]:
    """Refit simulated samples with the library's own estimator and score each refit's PIT.

    Returns the K-S and A^2 statistics and where the refit succeeded and left a positive value, each
    shaped like the trailing axes of ``draws``. A refit fails by the rule that sets ``fit_failed`` for the
    observed sample: no usable parameters, or a value outside the fitted support. A refit that raises fails
    every column of the batch.
    """
    try:
        with compute._fit_warnings_suppressed(), np.errstate(all="ignore"):
            fitted = compute.FittedDistribution.resolve(
                draws, distribution, period.start_year, period.start_year, period.end_year, periodicity
            )
    except DistributionFittingError:
        failed = np.full(draws.shape[1:], np.nan)
        return {"ks": failed, "ad": failed}, np.zeros(draws.shape[1:], dtype=bool)
    p = compute._broadcast_parameters(draws, fitted.parameters, full=True)
    pit, covered = _sample_pit(distribution, draws, ~np.isnan(draws), p)
    # the observed fit's rule (_score_fit): a value with no finite log-density is outside the fitted support. A CDF of
    # 0 or 1 is not that rule's evidence, since it also underflows inside the support
    outside = (covered & ~np.isfinite(_log_density(distribution, draws, p))).any(axis=0)
    usable = _usable(distribution, p) & ~outside
    ks, ad, count = _pit_statistics(np.where(usable, pit, np.nan))
    return {"ks": ks, "ad": ad}, usable & (count > 0)


def _bootstrap_p_values(
    chosen: tuple[Distribution, ...],
    fits: dict[Distribution, dict[str, np.ndarray] | None],
    scored: np.ndarray,
    observed: dict[GofStatistic, np.ndarray],
    sample: np.ndarray,
    valid: np.ndarray,
    *,
    period: Any,
    periodicity: compute.Periodicity,
    n_bootstrap: int,
    seed: int,
    resolution: float | None,
    origin: float = 0.0,
) -> tuple[dict[GofStatistic, np.ndarray], np.ndarray]:
    """Parametric-bootstrap p-values of the requested statistics and the share of failed refits.

    Each replicate has the observed sample's size and missing-value pattern and is drawn from the candidate's
    fitted mixed distribution, zero mass included. It is refitted with the library's estimator and scored,
    so the null distribution carries the estimation of the parameters. The p-value is
    ``(1 + #{statistic* >= statistic}) / (B_ok + 1)`` over the ``B_ok`` refits that succeeded, NaN where none did.
    ``scored`` is ``(candidate, step, *cell)`` and marks the candidates with a usable observed fit.
    """
    years, cell_shape = sample.shape[0], sample.shape[1:]
    batch = max(1, min(n_bootstrap, _BOOTSTRAP_BATCH_ELEMENTS // max(sample.size, 1)))
    generators = _column_generators(seed, sample)
    exceeded = {name: np.zeros((len(chosen), *cell_shape), dtype=np.int64) for name in observed}
    succeeded = np.zeros((len(chosen), *cell_shape), dtype=np.int64)
    done = 0
    while done < n_bootstrap:
        size = min(batch, n_bootstrap - done)
        uniforms = _draw_uniforms(generators, years, cell_shape, size)
        for position, distribution in enumerate(chosen):
            parameters = fits[distribution]
            if parameters is None:
                continue
            expanded = {key: value[..., np.newaxis] for key, value in parameters.items()}
            draws = _simulate(distribution, uniforms, expanded, resolution, origin)
            draws = np.where((valid & scored[position])[..., np.newaxis], draws, np.nan)
            statistics, succeeded_here = _refit_statistics(distribution, draws, period, periodicity)
            succeeded[position] += succeeded_here.sum(axis=-1)
            for name, simulated in statistics.items():
                if name in observed:
                    reached = simulated >= observed[name][position][..., np.newaxis]
                    exceeded[name][position] += (reached & succeeded_here).sum(axis=-1)
        done += size
    defined = scored & (succeeded > 0)
    with np.errstate(all="ignore"):
        p_values = {
            name: np.where(defined & np.isfinite(observed[name]), (1 + exceeded[name]) / (succeeded + 1), np.nan)
            for name in observed
        }
        failed_share = np.where(scored, 1.0 - succeeded / n_bootstrap, np.nan)
    return p_values, failed_share


def _bh_adjust_cells(p: np.ndarray, n_cell_dims: int) -> np.ndarray:
    """Benjamini-Hochberg adjustment over the trailing ``n_cell_dims`` axes, apart for every leading index.

    NaN p-values are left out of the family and stay NaN.
    """
    flat = p.reshape(*p.shape[: p.ndim - n_cell_dims], -1)
    adjusted = np.full(flat.shape, np.nan)
    for index in np.ndindex(*flat.shape[:-1]):
        finite = np.isfinite(flat[index])
        if finite.any():
            adjusted[index][finite] = scipy.stats.false_discovery_control(flat[index][finite])
    result: np.ndarray = adjusted.reshape(p.shape)
    return result


def _rejection_share_cells(p: np.ndarray, alpha: float, n_cell_dims: int) -> np.ndarray:
    """Share of the finite p-values at or below ``alpha`` over the trailing ``n_cell_dims`` axes; NaN if none is finite.

    A Monte-Carlo test rejects at ``p <= alpha``: a bootstrap p-value is a multiple of ``1 / (B_ok + 1)``, so with
    ``(B_ok + 1) alpha`` an integer the test then has size exactly ``alpha``, where ``p < alpha`` would have less.
    """
    flat = p.reshape(*p.shape[: p.ndim - n_cell_dims], -1)
    finite = np.isfinite(flat)
    with np.errstate(all="ignore"):
        share: np.ndarray = np.where(
            finite.any(axis=-1), ((flat <= alpha) & finite).sum(axis=-1) / finite.sum(axis=-1), np.nan
        )
    return share


def _decision_p_values(gof: GoodnessOfFit) -> dict[GofStatistic, np.ndarray | None]:
    """The p-value each statistic is judged by: the bootstrap one when it ran, else the naive K-S one."""
    if gof.n_bootstrap > 0:
        return {"ks": gof.ks_p_value_bootstrap, "ad": gof.ad_p_value_bootstrap}
    return {"ks": gof.ks_p_value_naive, "ad": None}


def _summarize_gof(gof: GoodnessOfFit, *, bh_adjust: bool, alpha: float, n_cell_dims: int) -> GoodnessOfFit:
    """Add the across-cell evidence: the Benjamini-Hochberg p-values and the rejection share."""
    if n_cell_dims == 0:
        return gof
    decided = _decision_p_values(gof)
    ks_p, ad_p = decided["ks"], decided["ad"]
    reference = gof.ks_statistic
    shares = [
        _rejection_share_cells(p, alpha, n_cell_dims) if p is not None else np.full(reference.shape[:2], np.nan)
        for p in (ks_p, ad_p)
    ]
    return dataclasses.replace(
        gof,
        ks_p_value_bh=_bh_adjust_cells(ks_p, n_cell_dims) if bh_adjust and ks_p is not None else None,
        ad_p_value_bh=_bh_adjust_cells(ad_p, n_cell_dims) if bh_adjust and ad_p is not None else None,
        rejection_share=np.stack(shares, axis=-1),
    )


def _validate_gof_arguments(
    gof: Sequence[GofStatistic],
    n_bootstrap: int,
    resolution: float | None,
    bh_adjust: bool,
    alpha: float,
    seed: int | None,
) -> tuple[GofStatistic, ...]:
    """Check the goodness-of-fit arguments and return ``gof`` as a tuple in canonical order."""
    names = ", ".join(_GOF_STATISTICS)
    if isinstance(gof, str) or not isinstance(gof, Sequence):
        raise _invalid("gof must be a sequence of statistic names.", "gof", gof, names)
    chosen = tuple(gof)
    for statistic in chosen:
        if statistic not in _GOF_STATISTICS:
            raise _invalid(f"Unsupported goodness-of-fit statistic: {statistic!r}.", "gof", statistic, names)
    if len(set(chosen)) != len(chosen):
        raise _invalid("gof statistics must be distinct.", "gof", chosen, names)
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
        raise _invalid("n_bootstrap must be a non-negative integer.", "n_bootstrap", n_bootstrap, "0 or more")
    if resolution is not None:
        unit = math.nan
        if isinstance(resolution, (int, float, np.integer, np.floating)) and not isinstance(
            resolution, (bool, np.bool_)
        ):
            try:
                unit = float(resolution)
            except OverflowError:  # an int beyond the float range
                unit = math.nan
        # a unit whose reciprocal overflows (a subnormal) would make every rounded draw infinite
        if not (math.isfinite(unit) and unit > 0.0 and math.isfinite(1.0 / unit)):
            raise _invalid(
                "resolution must be a positive, finite rounding unit, or None for a continuous record.",
                "resolution",
                resolution,
                "a positive finite number or None",
            )
    if not isinstance(bh_adjust, (bool, np.bool_)):
        raise _invalid("bh_adjust must be a boolean.", "bh_adjust", bh_adjust, "True or False")
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not 0.0 < alpha < 1.0
    ):
        raise _invalid("alpha must lie strictly between 0 and 1.", "alpha", alpha, "0 < alpha < 1")
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise _invalid(
            "seed must be a non-negative integer or None; a Generator is not accepted because it does not "
            "expose the seed that would reproduce its stream.",
            "seed",
            type(seed).__name__ if not isinstance(seed, (int, np.integer)) else seed,
            "a non-negative integer or None",
        )
    if not chosen and (n_bootstrap > 0 or bh_adjust):
        raise _invalid(
            "n_bootstrap and bh_adjust need at least one statistic in gof.", "gof", chosen, "a non-empty sequence"
        )
    if bh_adjust and n_bootstrap == 0 and "ks" not in chosen:
        raise _invalid(
            "bh_adjust has nothing to adjust: Anderson-Darling has no naive p-value, so without n_bootstrap it "
            "needs the K-S statistic in gof.",
            "bh_adjust",
            bh_adjust,
            'n_bootstrap > 0, or "ks" in gof',
        )
    return tuple(statistic for statistic in _GOF_STATISTICS if statistic in chosen)


def _gof_parameters(
    statistics: tuple[GofStatistic, ...],
    n_bootstrap: int,
    resolution: float | None,
    bh_adjust: bool,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """The goodness-of-fit arguments the provenance record keeps; none without a statistic.

    The resolution and seed matter only to a bootstrap. The seed is a string so one above 64 bits survives
    JSON and NetCDF.
    """
    if not statistics:
        return {}
    recorded: dict[str, Any] = {
        "gof": list(statistics),
        "n_bootstrap": n_bootstrap,
        "bh_adjust": bool(bh_adjust),
        "alpha": float(alpha),  # plain Python types, so any NumPy scalar the caller passed serializes
    }
    if n_bootstrap > 0:
        recorded.update({"resolution": None if resolution is None else float(resolution), "seed": str(seed)})
    return recorded


def _across_cells(
    function: Callable[[np.ndarray], np.ndarray], array: xr.DataArray, cell_dims: list[str], *, reduce: bool
) -> xr.DataArray:
    """Apply a function over all cells of a DataArray, rechunking the cell dimensions into one chunk if Dask-backed.

    ``reduce`` drops the cell dimensions from the result; otherwise the result keeps the input's dimensions.
    """
    result: xr.DataArray = xr.apply_ufunc(
        function,
        array,
        input_core_dims=[cell_dims],
        output_core_dims=[[] if reduce else cell_dims],
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    return result if reduce else result.transpose(*array.dims)


def _add_across_cell_variables(
    data_vars: dict[str, xr.DataArray],
    cell_dims: list[str],
    *,
    period_dim: str,
    bootstrap: bool,
    bh_adjust: bool,
    alpha: float,
) -> None:
    """Add the Benjamini-Hochberg p-values and ``rejection_share`` to the joined variables, as the NumPy route does."""
    decided = {
        "ks": data_vars["ks_p_value_bootstrap" if bootstrap else "ks_p_value_naive"],
        "ad": data_vars["ad_p_value_bootstrap"] if bootstrap else None,
    }
    n_cell_dims = len(cell_dims)
    added: dict[str, xr.DataArray] = {}
    shares = {
        name: _across_cells(lambda a: _rejection_share_cells(a, alpha, n_cell_dims), p, cell_dims, reduce=True)
        for name, p in decided.items()
        if p is not None
    }
    empty = xr.full_like(shares["ks"], np.nan)
    statistic_axis = xr.DataArray(np.array(_GOF_STATISTICS), dims="statistic", name="statistic")
    added["rejection_share"] = xr.concat(
        [shares.get(name, empty) for name in _GOF_STATISTICS], dim=statistic_axis
    ).transpose("candidate", period_dim, "statistic")
    if bh_adjust:
        for name, p in decided.items():
            if p is not None:
                added[f"{name}_p_value_bh"] = _across_cells(
                    lambda a: _bh_adjust_cells(a, n_cell_dims), p, cell_dims, reduce=False
                )
    base = {
        key: value
        for key, value in data_vars["ks_statistic"].attrs.items()
        if key not in ("long_name", "description", "units")
    }
    for name, variable in added.items():
        variable.attrs = {**base, **_FIELD_METADATA[name]}
        data_vars[name] = variable


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
    gof: Sequence[GofStatistic] = ("ks", "ad"),
    n_bootstrap: int = 0,
    resolution: float | None = None,
    bh_adjust: bool = False,
    alpha: float = 0.05,
    seed: int | None = None,
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

    Because the likelihood is taken at Thom's (1958) closed-form gamma approximation and at L-moment
    estimates, not at maximum-likelihood estimates, the criteria are penalized likelihoods at fitted
    estimates and a ranking mixes distribution fit with estimator efficiency. About 30 values
    per calendar step discriminate weakly, so treat ``best`` as evidence, not as "correct".
    A sample value outside a candidate's fitted support gives zero likelihood, so that candidate
    is reported in ``fit_failed`` and does not rank; ``best`` can then have only one eligible
    candidate, and ``delta_to_runner_up`` is NaN. A GEV with shape above 1 has a density that is
    unbounded at the finite end of its support, where likelihood methods are not regular, so
    treat its criteria with caution there.

    ``gof`` adds absolute goodness of fit, in :class:`GoodnessOfFit`. The statistics are the Kolmogorov-Smirnov D
    and the Anderson-Darling A^2 of each candidate's probability integral transform over the calibration
    sample. Gamma and Pearson Type III leave out exact zeros, since the zero mass is not part of the continuous
    distribution under test; log-logistic and GEV, which have none, keep them. Repeated values are kept. Because the parameters are estimated from the same sample, the naive
    K-S p-value is uncalibrated and Anderson-Darling has none. With ``n_bootstrap > 0`` every candidate, calendar
    step and cell runs a parametric bootstrap: it draws ``n_bootstrap`` samples of the observed size and
    missing-value pattern from the fitted mixed distribution (zero mass included), refits each with the library's
    own estimator and scores it, giving the p-values ``(1 + #{statistic* >= statistic}) / (B_ok + 1)`` over the
    ``B_ok`` refits that succeeded (the floor is ``1 / (B_ok + 1)``). A refit that fails is counted in
    ``bootstrap_failed_share`` and left out. The cost is ``n_bootstrap`` times that of the fits themselves. For a
    record rounded to a unit, pass that unit as ``resolution`` so the simulated samples have the observed ties;
    the unit is never inferred. The same ``seed`` gives an identical result, whatever the Dask chunking.
    A candidate's p-value after ``best`` selected it is a post-selection statistic, not an ordinary test.
    Across cells, ``rejection_share`` reports the share of cells below ``alpha`` and ``bh_adjust`` adds
    Benjamini-Hochberg p-values; spatial correlation weakens the assumptions of that adjustment.

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
        gof: Goodness-of-fit statistics to compute, ``"ks"`` and ``"ad"``. An empty sequence skips goodness of
            fit and leaves ``goodness_of_fit`` as ``None``. A statistic left out has NaN arrays.
        n_bootstrap: Parametric-bootstrap replicates per candidate, step and cell; 0 (the default) computes the
            statistics and the naive K-S p-value only.
        resolution: Rounding unit of the prepared sample, applied to the bootstrap draws; ``None`` draws continuous
            samples, the right null only for an unrounded record. Sums of values rounded to a unit are multiples of
            it, so for SPI it is the record's unit; with ``aggregation="mean"`` the sample's unit is that divided by
            ``scale``, and a SPEI water balance has a unit only if precipitation and PET share one. A positive draw
            that rounds to zero is set to ``resolution``, so the simulated zero share stays the fitted p0. For ``"spei"``
            the draws lie on ``1000 * scale + k * resolution``, the lattice of the prepared sample with its
            ``+1000`` offset per summed step. Must be positive and finite. Used only when ``n_bootstrap > 0``.
        bh_adjust: Add Benjamini-Hochberg adjusted p-values across cells, apart for every candidate and
            step. It needs a cell axis, so a series without one raises.
        alpha: Significance level of ``rejection_share``, which counts cells with ``p <= alpha`` (a Monte-Carlo
            test decides at ``p <= alpha``), strictly between 0 and 1.
        seed: Non-negative integer that fixes the bootstrap. ``None`` draws a 128-bit seed from the operating
            system's entropy, which the result records as a decimal string in ``climate_indices_rng_seed``.
            A ``numpy.random.Generator`` is not accepted, because it does not expose the seed that would
            reproduce its stream. Used only when ``n_bootstrap > 0``.
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
        When ``gof`` is not empty the Dataset also holds each :class:`GoodnessOfFit` field as a variable of the
        same name, with ``rejection_share`` over ``(candidate, month | dayofyear, statistic)``. ``rejection_share``
        and the Benjamini-Hochberg variables exist only when the input has a dimension besides time.
        Every variable carries the ADR-0028 section 6 attributes that apply to this route. A fully
        missing input is accepted and scores as no usable fit (``n`` is 0 and ``best`` is
        ``"none"``) instead of raising as the NumPy route does.

    Raises:
        InvalidArgumentError: For an unsupported route, criterion, candidate, goodness-of-fit
            argument (including a ``gof`` that is empty while ``n_bootstrap`` or ``bh_adjust`` asks for
            it, and ``bh_adjust`` on a series with no cell axis), hourly
            periodicity, non-numeric or infinite input, an argument combination the route
            does not accept, a NumPy call that omits a required temporal argument, an xarray
            ``data_start_year`` that disagrees with the time coordinate, or an input dimension or
            a coordinate that does not span the time dimension named like an output variable. A
            coordinate that spans the time dimension is allowed, since the reduction over time
            drops it. A Dask input reports an invalid value when it is computed.
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
    statistics = _validate_gof_arguments(gof, n_bootstrap, resolution, bh_adjust, alpha, seed)
    # one seed for the whole call, so every Dask block and both routes simulate from the same one
    resolved_seed = (secrets.randbits(128) if seed is None else int(seed)) if n_bootstrap > 0 else None
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
            gof=statistics,
            n_bootstrap=int(n_bootstrap),
            resolution=resolution,
            bh_adjust=bh_adjust,
            alpha=alpha,
            seed=resolved_seed,
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
        gof=statistics,
        n_bootstrap=int(n_bootstrap),
        resolution=resolution,
        bh_adjust=bh_adjust,
        alpha=alpha,
        seed=resolved_seed,
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


def _warn_overlapping_windows(scale: int, periodicity: compute.Periodicity) -> None:
    """Warn that a timescale beyond one year of steps makes consecutive years' windows overlap."""
    window_limit = 365 if periodicity is compute.Periodicity.daily else 12
    if scale > window_limit:
        warnings.warn(
            f"Timescale {scale} exceeds {window_limit} steps, so accumulation windows ending on the same "
            "calendar step in consecutive years overlap; effective sample sizes shrink and goodness-of-fit "
            "p-values are invalid.",
            ClimateIndicesWarning,
            stacklevel=_caller_stacklevel(),
        )


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
    gof: Sequence[GofStatistic] = _GOF_STATISTICS,
    n_bootstrap: int = 0,
    resolution: float | None = None,
    bh_adjust: bool = False,
    alpha: float = 0.05,
    seed: int | None = None,
    spatial_time_major: bool = False,
    require_valid: bool = True,
    summarize: bool = True,
) -> DistributionComparison:
    """The NumPy core of :func:`compare_distributions`, without the Beta warning (see there).

    ``require_valid=False`` lets an all-missing input through, as one masked block or cell of a
    grid: it scores as no usable fit instead of aborting the whole grid. ``summarize=False`` leaves out the
    evidence that spans cells (``rejection_share`` and the Benjamini-Hochberg p-values), which a block of
    a larger grid cannot compute; the xarray route adds it after the blocks are joined.
    """
    _validate_arguments(index, criterion, pet_mm, aggregation, clip_negatives)
    statistics = _validate_gof_arguments(gof, n_bootstrap, resolution, bh_adjust, alpha, seed)
    # used (and recorded) only when n_bootstrap > 0; the public function resolves it once for a whole call
    bootstrap_seed = secrets.randbits(128) if seed is None else int(seed)
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
    if bh_adjust and sample.ndim == 2:
        raise _invalid(
            "bh_adjust adjusts across cells, so the series needs a cell axis.",
            "bh_adjust",
            bh_adjust,
            "False, or values with a cell axis (a block, or spatial_time_major=True)",
        )
    valid = ~np.isnan(sample)
    n = valid.sum(axis=0)
    n0 = (valid & (sample == 0.0)).sum(axis=0)  # NOSONAR (S1244) p0 counts exact zeros, not trace values

    cell_shape = sample.shape[1:]
    log_likelihood = np.full((len(chosen), *cell_shape), np.nan)
    n_parameters = np.zeros((len(chosen), *cell_shape), dtype=np.int64)
    fit_failed = np.ones((len(chosen), *cell_shape), dtype=bool)
    parameters: dict[Distribution, dict[str, np.ndarray]] = {}
    fits: dict[Distribution, dict[str, np.ndarray] | None] = dict.fromkeys(chosen)
    ks_statistic = np.full((len(chosen), *cell_shape), np.nan)
    ks_p_naive = np.full((len(chosen), *cell_shape), np.nan)
    ad_statistic = np.full((len(chosen), *cell_shape), np.nan)

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
        fits[distribution] = full
        if statistics:
            ks, ad, covered_count = _observed_statistics(distribution, sample, valid, full, ~fit_failed[position])
            if "ks" in statistics:
                ks_statistic[position] = ks
                ks_p_naive[position] = _ks_naive_p_value(ks, covered_count)
            if "ad" in statistics:
                ad_statistic[position] = ad

    goodness_of_fit: GoodnessOfFit | None = None
    if statistics:
        ks_p_bootstrap = ad_p_bootstrap = failed_share = None
        if n_bootstrap > 0:
            observed: dict[GofStatistic, np.ndarray] = {}
            if "ks" in statistics:
                observed["ks"] = ks_statistic
            if "ad" in statistics:
                observed["ad"] = ad_statistic
            p_values, failed_share = _bootstrap_p_values(
                chosen,
                fits,
                ~fit_failed,
                observed,
                sample,
                valid,
                period=period,
                periodicity=periodicity,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
                resolution=resolution,
                # SPEI sums `scale` water balances that each carry the +1000 of _spei_water_balance, so its sample
                # lies on 1000 * scale + k * resolution
                origin=1000.0 * scale if index == "spei" else 0.0,
            )
            not_run = np.full(ks_statistic.shape, np.nan)
            ks_p_bootstrap, ad_p_bootstrap = p_values.get("ks", not_run), p_values.get("ad", not_run)
        goodness_of_fit = GoodnessOfFit(
            ks_statistic=ks_statistic,
            ks_p_value_naive=ks_p_naive,
            ad_statistic=ad_statistic,
            ks_p_value_bootstrap=ks_p_bootstrap,
            ad_p_value_bootstrap=ad_p_bootstrap,
            ks_p_value_bh=None,
            ad_p_value_bh=None,
            rejection_share=None,
            n_bootstrap=n_bootstrap,
            bootstrap_failed_share=failed_share,
        )
        if summarize:
            goodness_of_fit = _summarize_gof(
                goodness_of_fit, bh_adjust=bh_adjust, alpha=alpha, n_cell_dims=len(cell_shape) - 1
            )

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
            stacklevel=_caller_stacklevel(),
        )
    _warn_overlapping_windows(scale, periodicity)

    inputs: dict[str, Any] = {"values": values}
    if pet_mm is not None:
        inputs["pet_mm"] = pet_mm
    gof_parameters = _gof_parameters(statistics, n_bootstrap, resolution, bh_adjust, alpha, bootstrap_seed)
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
            **gof_parameters,
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
    if statistics:
        metadata["climate_indices_gof_bootstrap_samples"] = str(n_bootstrap)
        if n_bootstrap > 0:
            metadata["climate_indices_rng_seed"] = str(bootstrap_seed)
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
        goodness_of_fit=goodness_of_fit,
        metadata=metadata,
    )


# ---- xarray / Dask route (SUIT-04, #1585) ---------------------------------------------------------

_CANDIDATE_FIELDS = ("log_likelihood", "aic", "aicc", "bic", "n_parameters", "fit_failed")
_STEP_FIELDS = ("n", "best", "delta_to_runner_up")

# GoodnessOfFit fields a block computes per candidate; the bootstrap ones exist only when n_bootstrap > 0
_GOF_FIELDS = ("ks_statistic", "ks_p_value_naive", "ad_statistic")
_GOF_BOOTSTRAP_FIELDS = ("ks_p_value_bootstrap", "ad_p_value_bootstrap", "bootstrap_failed_share")
# GoodnessOfFit fields that need every cell, so they are added after the blocks are joined
_GOF_ACROSS_CELL_FIELDS = ("ks_p_value_bh", "ad_p_value_bh", "rejection_share")

_FIELD_METADATA: dict[str, dict[str, str]] = {
    "ks_statistic": {
        "long_name": "Kolmogorov-Smirnov statistic of the probability integral transform",
        "units": "1",
        "description": (
            "Largest gap between the empirical distribution of the candidate's PIT over the calibration values and the "
            "uniform distribution; exact zeros are left out for a family with a zero mass. NaN where the candidate "
            "failed to fit."
        ),
    },
    "ks_p_value_naive": {
        "long_name": "Naive Kolmogorov-Smirnov p-value",
        "units": "1",
        "description": (
            "Exact K-S p-value for fully specified parameters. The parameters were estimated from the same sample, "
            "so it is uncalibrated and too high; use the bootstrap p-value for a test."
        ),
    },
    "ad_statistic": {
        "long_name": "Anderson-Darling statistic of the probability integral transform",
        "units": "1",
        "description": "A^2 of the candidate's PIT over the same values as the K-S statistic; it has no naive p-value.",
    },
    "ks_p_value_bootstrap": {
        "long_name": "Parametric-bootstrap Kolmogorov-Smirnov p-value",
        "units": "1",
        "description": (
            "(1 + #{D* >= D}) / (B_ok + 1) over the refits that succeeded; NaN where none did. A p-value of the "
            "candidate that `best` selected is a post-selection statistic."
        ),
    },
    "ad_p_value_bootstrap": {
        "long_name": "Parametric-bootstrap Anderson-Darling p-value",
        "units": "1",
        "description": (
            "(1 + #{A2* >= A2}) / (B_ok + 1) over the refits that succeeded; NaN where none did. A p-value of the "
            "candidate that `best` selected is a post-selection statistic."
        ),
    },
    "bootstrap_failed_share": {
        "long_name": "Share of bootstrap refits that failed",
        "units": "1",
        "description": "Refits with no usable parameters or no positive value, left out of the bootstrap p-value.",
    },
    "ks_p_value_bh": {
        "long_name": "Benjamini-Hochberg adjusted Kolmogorov-Smirnov p-value",
        "units": "1",
        "description": (
            "Adjusted across cells for each candidate and calendar step, from the bootstrap p-value when it was "
            "computed and else the naive one. Spatial correlation weakens the assumptions of the adjustment."
        ),
    },
    "ad_p_value_bh": {
        "long_name": "Benjamini-Hochberg adjusted Anderson-Darling p-value",
        "units": "1",
        "description": "Adjusted across cells for each candidate and calendar step; present only with a bootstrap.",
    },
    "rejection_share": {
        "long_name": "Share of cells with a goodness-of-fit p-value at or below alpha",
        "units": "1",
        "description": (
            "Per candidate, calendar step and statistic, from the bootstrap p-value when it was computed and else "
            "the naive K-S p-value (the Anderson-Darling share is then NaN). Cells without a p-value are not counted."
        ),
    },
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
# an input's valid range describes its own values, not a comparison of them
_RANGE_ATTRS = ("valid_min", "valid_max", "valid_range", "actual_range")


def _comparison_block(
    values: np.ndarray,
    *secondary: np.ndarray,
    chosen: tuple[Distribution, ...],
    slots: tuple[tuple[str, Distribution], ...],
    fields: tuple[str, ...],
    **kwargs: Any,
) -> tuple[np.ndarray, ...]:
    """Run the NumPy core on one block or cell and flatten it into the declared output arrays."""
    result = _compare_numpy(
        values,
        pet_mm=secondary[0] if secondary else None,
        candidates=chosen,
        require_valid=False,
        summarize=False,
        **kwargs,
    )
    arrays = [
        np.asarray(
            getattr(result.goodness_of_fit if field in _GOF_FIELDS + _GOF_BOOTSTRAP_FIELDS else result, field)[
                position
            ],
            dtype=float,
        )
        for field in fields
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


def _validate_time_axis(dataarray: xr.DataArray, time_dim: str) -> None:
    """Check the time dimension, ordering, calendar and ADR-0003 chunking of one caller array."""
    validate_time_dimension(dataarray, time_dim)
    validate_time_monotonicity(dataarray[time_dim])
    # the calendar itself is checked before inference reads it, so a numeric, cftime or
    # timezone-aware coordinate fails as CoordinateValidationError rather than inside pandas
    _validate_calendar(dataarray[time_dim])
    if dataarray.chunks is not None:
        validate_dask_chunks(dataarray, time_dim)


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
    # the calendar adapter converts to float, so a numeric string would pass as data; the dtype is metadata only
    _require_numeric_dtype(values.dtype, "values")
    if pet_da is not None:
        _require_numeric_dtype(pet_da.dtype, "pet_mm")
    # ADR-0003 constrains the caller's arrays, and alignment to a shorter pet can hide a split time axis
    _validate_time_axis(input_da, time_dim)
    if pet_da is not None:
        _validate_time_axis(pet_da, time_dim)
        input_da, aligned = _align_inputs(input_da, {"pet": pet_da}, time_dim)
        pet_da = aligned["pet"]
    for dataarray in (input_da, pet_da):
        if dataarray is not None and dataarray.chunks is not None:
            validate_dask_chunks(dataarray, time_dim)
    # the kernel never runs on an empty cell axis, so the new calendar-step dimension cannot be sized
    if empty := [str(dim) for dim, size in input_da.sizes.items() if size == 0 and dim != time_dim]:
        raise _invalid(
            f"values must not have a zero-length cell dimension: {', '.join(empty)}.",
            "values",
            ", ".join(empty),
            "at least one cell along every non-time dimension",
        )
    return input_da, pet_da


def _reject_output_name_clashes(reserved: set[str], time_dim: str, *arrays: xr.DataArray | None) -> None:
    """Reject an input dimension or coordinate named like an output, which would drop it or fail in the merge."""
    # apply_ufunc drops a coordinate that spans the core time dimension, so only dimensions
    # and the coordinates that can survive into the output can collide with a variable name
    taken = {time_dim}
    for array in arrays:
        if array is not None:
            taken |= {str(dim) for dim in array.dims}
            taken |= {str(name) for name, coord in array.coords.items() if time_dim not in coord.dims}
    if clash := sorted(reserved & taken):
        raise _invalid(
            f"Input dimensions and coordinates must not be named like an output: {', '.join(clash)}.",
            "values",
            ", ".join(clash),
            "rename them, or drop a scalar coordinate such as the one a .sel() leaves behind",
        )


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
    gof: Sequence[GofStatistic],
    n_bootstrap: int,
    resolution: float | None,
    bh_adjust: bool,
    alpha: float,
    seed: int | None,
    time_dim: str,
) -> xr.Dataset:
    """The xarray/Dask route of :func:`compare_distributions`, following ``fit_diagnostics``."""
    _validate_arguments(index, criterion, pet_mm, aggregation, clip_negatives)
    statistics = _validate_gof_arguments(gof, n_bootstrap, resolution, bh_adjust, alpha, seed)
    # one seed for every block, so the result does not depend on how the graph is chunked
    bootstrap_seed = secrets.randbits(128) if seed is None else int(seed)
    chosen = _resolve_candidates(index, candidates)
    input_da, pet_da = _validated_inputs(values, pet_mm, time_dim)
    cell_dims = [str(dim) for dim in input_da.dims if dim != time_dim]
    if bh_adjust and not cell_dims:
        raise _invalid(
            "bh_adjust adjusts across cells, so values needs a dimension besides time.",
            "bh_adjust",
            bh_adjust,
            "False, or values with a cell dimension",
        )

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
            "gof": statistics,
            "n_bootstrap": n_bootstrap,
            "resolution": resolution if n_bootstrap > 0 else None,
            "seed": bootstrap_seed if n_bootstrap > 0 else None,
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

    fields: tuple[str, ...] = _CANDIDATE_FIELDS
    if statistics:
        fields = fields + _GOF_FIELDS + (_GOF_BOOTSTRAP_FIELDS if n_bootstrap > 0 else ())
    reserved = {"candidate", period_dim, *fields, *_STEP_FIELDS, *(name for name, _ in slots)}
    if statistics:
        reserved |= {*_GOF_ACROSS_CELL_FIELDS, "statistic"}
    _reject_output_name_clashes(reserved, time_dim, input_da, pet_da)

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
    if statistics:
        calculation_metadata["climate_indices_gof_bootstrap_samples"] = str(n_bootstrap)
        if n_bootstrap > 0:
            calculation_metadata["climate_indices_rng_seed"] = str(bootstrap_seed)
    # native types, so a query over the record matches the same values as the NumPy route
    provenance_parameters = {
        "index": index,
        "scale": scale,
        "data_start_year": provided["data_start_year"],
        "periodicity": resolved_periodicity.name,
        "aggregation": aggregation,
        "clip_negatives": effective_clip,
        "criterion": criterion,
        "candidates": labels,
        "fitting_methods": {d.value: _FITTING_METHODS[d] for d in chosen},
        "calibration_year_initial": first_year,
        "calibration_year_final": last_year,
        "log_likelihood_at": "fitted_estimates",
        **_gof_parameters(statistics, n_bootstrap, resolution, bh_adjust, alpha, bootstrap_seed),
    }

    def declare(field: str, *, transform: Any = None) -> _SpatialOutput:
        return _SpatialOutput(
            period_dim,
            dict(_FIELD_METADATA[field]),
            "Distribution comparison",
            coordinate=steps,
            transform=transform,
            drop_attrs=("units",) if field in _UNITLESS_FIELDS else (),
        )

    def declare_parameter(name: str) -> _SpatialOutput:
        long_name = _PARAMETER_METADATA[name]
        return _fitted_parameter_output(
            name,
            {"long_name": long_name, "description": f"{long_name}, per calendar step and cell."},
            inputs,
            core_dim=period_dim,
            index_name="Distribution comparison",
            coordinate=steps,
        )

    def integer(array: xr.DataArray) -> xr.DataArray:
        return array.astype("int64")

    def boolean(array: xr.DataArray) -> xr.DataArray:
        return array.astype(bool)

    inputs = (input_da,) if pet_da is None else (input_da, pet_da)
    transforms = {"n_parameters": integer, "fit_failed": boolean}
    outputs = (
        *(declare(field, transform=transforms.get(field)) for field in fields for _ in chosen),
        # one output per kernel array, including a name two families share ("shape")
        *(declare_parameter(name) for name, _ in slots),
        declare("n", transform=integer),
        declare("best", transform=_label_best(labels)),
        declare("delta_to_runner_up"),
    )
    arrays = _spatial_kernel_call(
        functools.partial(_comparison_block, chosen=chosen, slots=slots, fields=fields),
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
    for field in fields:
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
    if statistics and cell_dims:
        _add_across_cell_variables(
            data_vars, cell_dims, period_dim=period_dim, bootstrap=n_bootstrap > 0, bh_adjust=bh_adjust, alpha=alpha
        )
    for variable in data_vars.values():
        for key in _RANGE_ATTRS:
            variable.attrs.pop(key, None)
    return xr.Dataset(data_vars)


# ---------------------------------------------------------------------------
# Drought categories and category-change metrics (#1587, ADR-0028 section 7)
# ---------------------------------------------------------------------------

#: Class labels in code order. Code 0 is "none", codes 1 to 5 are D0 to D4, and -1 marks a missing value.
DROUGHT_CATEGORY_LABELS: tuple[str, ...] = ("none", "D0", "D1", "D2", "D3", "D4")
_N_CATEGORIES = len(DROUGHT_CATEGORY_LABELS)

# the names of the comparison result, which an input dimension or coordinate that stays may not take:
# a kept dimension would replace the variable with the input's labels, and a kept coordinate fails the merge
_CATEGORY_COMPARISON_NAMES = (
    "confusion",
    "reference_counts",
    "alternative_counts",
    "n_valid",
    "share_changed",
    "share_changed_in_drought",
    "reference_category",
    "alternative_category",
)

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
    try:
        codes = values if np.ma.isMaskedArray(values) else np.asarray(values)
    except (TypeError, ValueError) as error:  # a ragged or otherwise unarrayable sequence
        raise _invalid(
            f"{name} must hold integer category codes from drought_category.",
            name,
            type(values).__name__,
            "integer codes -1 to 5",
        ) from error
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


def _require_auxiliary_coordinates_equal(
    reference: xr.DataArray, alternative: xr.DataArray, reduce_dims: tuple[Any, ...]
) -> None:
    """Reject a surviving coordinate that is not an index and differs, which align does not compare."""
    for coord_name, coordinate in reference.coords.items():  # align compares the indexed coordinates only
        if coord_name in reference.xindexes or set(coordinate.dims) & set(reduce_dims):
            continue  # a reduced coordinate is dropped from the result, so its labels pair no cells
        # a coordinate is a label, so comparing it reads it even when the data stays lazy; a differing
        # one would otherwise pair cells by position and keep one side's labels
        try:
            same = coordinate.variable.equals(alternative.coords[coord_name].variable)
        except ValueError as error:  # an unreadable lazy coordinate, as align can raise for an index
            raise _invalid(
                f"reference and alternative coordinates could not be compared: {error}",
                "alternative",
                "unreadable",
                "readable coordinates",
            ) from error
        if not same:
            raise _invalid(
                f"reference and alternative must share their coordinates: {coord_name!r} differs.",
                "alternative",
                "mismatch",
                "equal",
            )


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
    if set(reference.coords) != set(alternative.coords):
        raise _invalid(
            "reference and alternative must share their coordinates: only one has "
            f"{sorted(set(reference.coords) ^ set(alternative.coords), key=str)}.",
            "alternative",
            "mismatch",
            "equal",
        )
    reduce_dims = _reduction_dims(dims, tuple(reference.dims))
    # a dimensionless input has nothing to reduce over, and .sum(()) would return the boolean identity
    sum_dims = reduce_dims or None
    _require_auxiliary_coordinates_equal(reference, alternative, reduce_dims)
    try:
        reference, alternative = xr.align(reference, alternative, join="exact")
    except ValueError as error:
        raise _invalid(
            f"reference and alternative must share their coordinates: {error}", "alternative", "mismatch", "equal"
        ) from error
    kept_names = set(reference.dims) - set(reduce_dims)
    for array in (reference, alternative):  # a surviving coordinate collides with a result variable too
        kept_names |= {name for name, coord in array.coords.items() if not set(coord.dims) & set(reduce_dims)}
    kept_clash = set(_CATEGORY_COMPARISON_NAMES) & kept_names
    if kept_clash:
        raise _invalid(
            f"{sorted(kept_clash)} are reserved for the comparison result and cannot stay as input dimensions or "
            "coordinates; rename them or reduce over them.",
            "dims",
            sorted(kept_clash),
            "other dimension names",
        )

    # a code outside 0 to 5 (possible only for a lazy input, which is not range-checked) is excluded
    # everywhere, so n_valid, the shares and the confusion matrix always count the same pairs
    valid = (reference >= 0) & (reference < _N_CATEGORIES) & (alternative >= 0) & (alternative < _N_CATEGORIES)
    n_valid = valid.sum(sum_dims)
    changed = n_valid - ((reference == alternative) & valid).sum(sum_dims)
    n_none = ((reference == 0) & (alternative == 0)).sum(sum_dims)
    # one reduction per cell of the matrix keeps the transient memory at the size of the input
    confusion = xr.concat(
        [
            xr.concat(
                [((reference == r) & (alternative == a)).sum(sum_dims) for a in range(_N_CATEGORIES)],
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
            for NumPy input, an input dimension or coordinate that stays is named like one of the result
            variables, or only one input is a DataArray. A Dask-backed input's data is not read to check
            the range, so a code outside -1 to 5 in it is excluded from every field like a missing one,
            though the labels of a coordinate that stays are compared. A masked NumPy code counts as missing.

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

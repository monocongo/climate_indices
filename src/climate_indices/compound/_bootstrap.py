"""Monthly moving-year-block inference under ADR-0022.

Paired blocks estimate uncertainty. Independently sampled channels test calendar-
conditioned independence. Neither test removes nonstationarity or long memory.
Lag responses are computed on the original calendar, never across sampled joins.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy

from climate_indices.compound._harmonize import AlignedHazards, _check_choice, _freeze, _immutable
from climate_indices.compound._statistics import (
    GroupBy,
    _check_lag_window,
    _forward_windows,
    _harmonization_record,
    _pair_indices,
    _safe_rate,
    _validate_hazards,
)
from climate_indices.exceptions import ClimateIndicesWarning, InvalidArgumentError
from climate_indices.logging_config import get_logger, log_calculation_failure

Statistic: TypeAlias = Literal["joint_probability", "likelihood_multiplication_factor", "conditional_probability"]
Alternative: TypeAlias = Literal["greater", "less"]
_logger = get_logger(__name__)
_FLAGS = {
    "insufficient_years": 1,
    "insufficient_blocks": 2,
    "sparse_occurrences": 4,
    "insufficient_successes": 8,
    "insufficient_expected_count": 16,
    "invalid_interval_resamples": 32,
    "invalid_null_resamples": 64,
    "undefined_denominator": 128,
    "few_blocks": 256,
}
# Equal count ratios differ by about 1e-17 once rounded; distinct statistics are separated by at least 1/N**k.
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True, eq=False)
class BootstrapResult:
    """Monthly inference arrays and an immutable choice manifest.

    Estimate, interval, p-value, resample-count, and flag arrays have shape
    ``(12, *cells)`` for month groups, or ``(*cells)`` for pooled groups.
    The remaining arrays have shape ``(*cells)``. A series has no cell axes.
    All arrays are copied and read-only. Instances compare by identity.

    ``estimate`` uses the selected complete-year segment; ``full_record_estimate``
    uses every eligible original observation, including partial years. Neither
    point estimate is suppressed by the inference-only support guards.
    ``inference_start`` and ``inference_end`` are datetime64[D] month labels
    (January and December); no complete year gives NaT. ``n_years`` and
    ``block_length_years`` describe that segment. Effective resample counts are
    zero when a pre-sampling support guard excludes the corresponding metric.
    ``quality_flags`` combines the bits named by ``metadata['quality_flags']``.
    A failed interval guard does not itself suppress an eligible null test.
    All p-values are local and unadjusted, not field significance.
    """

    statistic: Statistic
    groupby: GroupBy
    estimate: npt.NDArray[np.float64]
    full_record_estimate: npt.NDArray[np.float64]
    confidence_low: npt.NDArray[np.float64]
    confidence_high: npt.NDArray[np.float64]
    p_value: npt.NDArray[np.float64]
    n_interval_resamples: npt.NDArray[np.int64]
    n_null_resamples: npt.NDArray[np.int64]
    n_years: npt.NDArray[np.int64]
    block_length_years: npt.NDArray[np.int64]
    inference_start: npt.NDArray[np.datetime64]
    inference_end: npt.NDArray[np.datetime64]
    quality_flags: npt.NDArray[np.int64]
    metadata: Mapping[str, Any]


def _integer(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidArgumentError(
            f"{name} must be an integer >= {minimum}.",
            argument_name=name,
            argument_value=repr(value),
            valid_values=f"integer >= {minimum}",
        )
    return value


def _longest_segment(complete: npt.NDArray[np.bool_]) -> tuple[int, int]:
    """Return half-open year indices; the first longest segment wins a tie."""
    edges = np.diff(np.r_[False, complete, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if starts.size == 0:
        return 0, 0
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def _year_indices(rng: np.random.Generator, years: int, length: int) -> npt.NDArray[np.int64]:
    """Draw non-circular starts, concatenate whole blocks, and truncate whole years."""
    starts = rng.integers(0, years - length + 1, size=math.ceil(years / length))
    return np.asarray((starts[:, None] + np.arange(length)).ravel()[:years], dtype=np.int64)


def _metrics(
    rows: npt.NDArray[np.bool_],
    eligible: npt.NDArray[np.bool_],
    statistic: Statistic,
    groupby: GroupBy,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Evaluate one realization; retain exact counts for the separate support gates."""
    axes = (0,) if groupby == "month" else (0, 1)
    n = np.asarray(eligible.sum(axis=axes, dtype=np.int64))
    marginal = (rows & eligible).sum(axis=tuple(axis + 1 for axis in axes), dtype=np.int64)
    successes = np.asarray((rows.all(axis=0) & eligible).sum(axis=axes, dtype=np.int64))
    rates = _safe_rate(marginal, n)
    joint = _safe_rate(successes, n)
    independent = np.prod(rates, axis=0)
    if statistic == "conditional_probability":
        estimate = _safe_rate(successes, marginal[0])
        excess = estimate - rates[1]
    else:
        estimate = joint
        if statistic == "likelihood_multiplication_factor":
            estimate = np.full(joint.shape, np.nan)
            np.divide(joint, independent, out=estimate, where=independent > 0)
        excess = joint - independent
    return np.asarray(estimate), np.asarray(excess), marginal, successes, n


def _resample(
    rows: npt.NDArray[np.bool_],
    *,
    statistic: Statistic,
    groupby: GroupBy,
    length: int,
    n_resamples: int,
    seed: int,
    support: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Keep only metric realizations, not a resample-by-time-by-cell cube."""
    years = rows.shape[1]
    # Stable support keys, not Python's randomized hash or a spatial cell number.
    digest = hashlib.sha256(support.encode("utf-8")).digest()
    streams = np.random.SeedSequence([seed, *np.frombuffer(digest, dtype="<u4").tolist()]).spawn(2)
    paired_rng, null_rng = (np.random.default_rng(stream) for stream in streams)
    shape = (n_resamples, 12) if groupby == "month" else (n_resamples,)
    paired = np.full(shape, np.nan)
    null = np.full(shape, np.nan)
    eligible = np.ones(rows.shape[1:], dtype=bool)
    for b in range(n_resamples):
        indices = _year_indices(paired_rng, years, length)
        paired[b] = _metrics(rows[:, indices], eligible, statistic, groupby)[0]
        sampled = np.stack([row[_year_indices(null_rng, years, length)] for row in rows])
        metric, excess, _, _, _ = _metrics(sampled, eligible, statistic, groupby)
        null[b] = np.where(np.isfinite(metric), excess, np.nan)
    return paired, null


def _bootstrap(
    hazards: AlignedHazards,
    *,
    statistic: Statistic,
    given: str | None,
    target: str | None,
    lag_window: tuple[int, int],
    groupby: GroupBy,
    block_length_years: int | None,
    n_resamples: int,
    confidence_level: float,
    alternative: Alternative,
    seed: int,
) -> BootstrapResult:
    hazards = _validate_hazards(hazards)
    if hazards.cadence != "monthly":
        raise InvalidArgumentError("Bootstrap inference requires monthly inputs.", argument_name="hazards.cadence")
    selected: tuple[int, ...]
    if statistic == "conditional_probability":
        selected = _pair_indices(hazards.names, given, target)
    else:
        if given is not None or target is not None or lag_window != (0, 0):
            raise InvalidArgumentError("given, target, and lag_window apply only to conditional inference.")
        selected = tuple(range(len(hazards.names)))

    month_numbers = hazards.time.astype("datetime64[M]").astype(np.int64)
    first_year = int(month_numbers[0] // 12)
    last_year = int(month_numbers[-1] // 12)
    calendar_years = last_year - first_year + 1
    positions = month_numbers - first_year * 12
    cells = hazards.exceedance.shape[2:]
    cell_count = math.prod(cells)
    occurrence = hazards.exceedance.reshape(len(hazards.names), len(positions), cell_count)
    valid = hazards.valid.reshape(occurrence.shape)
    groups = (12,) if groupby == "month" else ()
    metric_shape = groups + (cell_count,)
    floats = {
        name: np.full(metric_shape, np.nan)
        for name in ("estimate", "full_record_estimate", "confidence_low", "confidence_high", "p_value")
    }
    counts = {
        name: np.zeros(metric_shape, dtype=np.int64)
        for name in ("n_interval_resamples", "n_null_resamples", "quality_flags")
    }
    years_used = np.zeros(cell_count, dtype=np.int64)
    lengths = np.zeros(cell_count, dtype=np.int64)
    starts = np.full(cell_count, np.datetime64("NaT", "D"))
    ends = starts.copy()

    for cell in range(cell_count):
        rows: npt.NDArray[np.bool_] = np.zeros((len(selected), calendar_years * 12), dtype=bool)
        observed = np.zeros(rows.shape, dtype=bool)
        for channel, source in enumerate(selected):
            observed[channel, positions] = valid[source, :, cell]
            rows[channel, positions] = occurrence[source, :, cell] & valid[source, :, cell]
        if statistic == "conditional_probability":
            response_valid, response = _forward_windows(observed[1], rows[1], *lag_window)
            rows[1] = response
            eligible = observed[0] & response_valid
        else:
            eligible = observed.all(axis=0)
        rows = rows.reshape(len(selected), calendar_years, 12)
        eligible = eligible.reshape(calendar_years, 12)
        floats["full_record_estimate"][..., cell] = _metrics(rows, eligible, statistic, groupby)[0]
        start, end = _longest_segment(np.asarray(eligible.all(axis=1)))
        years = end - start
        length = block_length_years if block_length_years is not None else max(1, math.ceil(years ** (1 / 3)))
        years_used[cell], lengths[cell] = years, length
        if years:
            starts[cell] = np.datetime64(first_year + start, "Y").astype("datetime64[D]")
            ends[cell] = (np.datetime64(first_year + end, "Y").astype("datetime64[M]") - 1).astype("datetime64[D]")
        segment = rows[:, start:end]
        point, excess, marginal, successes, n = _metrics(segment, np.ones((years, 12), dtype=bool), statistic, groupby)
        floats["estimate"][..., cell] = point
        flags = np.zeros(groups, dtype=np.int64)
        blocks = years // length
        if years < 30:
            flags |= _FLAGS["insufficient_years"]
        if blocks < 5:
            flags |= _FLAGS["insufficient_blocks"]
        if blocks < 10:
            flags |= _FLAGS["few_blocks"]
        sparse = (marginal < 10).any(axis=0)
        flags |= np.where(sparse, _FLAGS["sparse_occurrences"], 0)
        flags |= np.where(~np.isfinite(point), _FLAGS["undefined_denominator"], 0)
        rates = _safe_rate(marginal, n)
        expected = marginal[0] * rates[1] if statistic == "conditional_probability" else n * np.prod(rates, axis=0)
        interval_ok = (successes >= 5) & ~sparse
        null_ok = (expected >= 5) & ~sparse
        flags |= np.where(successes < 5, _FLAGS["insufficient_successes"], 0)
        flags |= np.where(~(expected >= 5), _FLAGS["insufficient_expected_count"], 0)
        if years >= 30 and blocks >= 5 and (np.any(interval_ok) or np.any(null_ok)):
            support = json.dumps([str(starts[cell]), str(ends[cell]), length, statistic, lag_window, groupby])
            paired, null = _resample(
                segment,
                statistic=statistic,
                groupby=groupby,
                length=length,
                n_resamples=n_resamples,
                seed=seed,
                support=support,
            )
            paired_valid, null_valid = np.isfinite(paired), np.isfinite(null)
            n_paired, n_null = paired_valid.sum(axis=0), null_valid.sum(axis=0)
            counts["n_interval_resamples"][..., cell] = np.where(interval_ok, n_paired, 0)
            counts["n_null_resamples"][..., cell] = np.where(null_ok, n_null, 0)
            flags |= np.where(interval_ok & (n_paired < 0.95 * n_resamples), _FLAGS["invalid_interval_resamples"], 0)
            flags |= np.where(null_ok & (n_null < 0.95 * n_resamples), _FLAGS["invalid_null_resamples"], 0)
            # Quantiles exclude undefined metrics, but only after the 95% validity gate.
            for index in np.ndindex(groups):
                if interval_ok[index] and n_paired[index] >= 0.95 * n_resamples:
                    bounds = np.quantile(
                        paired[(slice(None),) + index][paired_valid[(slice(None),) + index]],
                        [(1 - confidence_level) / 2, (1 + confidence_level) / 2],
                        method="linear",
                    )
                    floats["confidence_low"][index + (cell,)] = bounds[0]
                    floats["confidence_high"][index + (cell,)] = bounds[1]
                if null_ok[index] and n_null[index] >= 0.95 * n_resamples:
                    distribution = null[(slice(None),) + index]
                    extreme = (
                        distribution >= excess[index] - _TIE_TOLERANCE
                        if alternative == "greater"
                        else distribution <= excess[index] + _TIE_TOLERANCE
                    )
                    floats["p_value"][index + (cell,)] = (
                        1 + np.count_nonzero(extreme & null_valid[(slice(None),) + index])
                    ) / (1 + n_null[index])
        counts["quality_flags"][..., cell] = flags

    if np.any(counts["quality_flags"]):
        warnings.warn(
            "Some compound inference outputs have limited support; inspect quality_flags and effective resample counts. "
            "Fewer than ten nonoverlapping blocks requires particular caution.",
            ClimateIndicesWarning,
            stacklevel=3,
        )
    manifest = {
        "schema_version": 1,
        "function": "bootstrap_dependence",
        "statistic": statistic,
        "source_names": tuple(hazards.names[i] for i in selected),
        "groupby": groupby,
        "given": given,
        "target": target,
        "lag_window": lag_window,
        "lag_units": "calendar months",
        "cadence": "monthly",
        "calendar": hazards.metadata.get("calendar", "unknown"),
        "segment_selection": "longest contiguous complete calendar-year segment; earliest tie",
        "block_selector": "explicit" if block_length_years is not None else "ceil(Y ** (1/3)) years",
        "block_length_years_override": block_length_years,
        "sampling": "non-circular moving whole-year blocks; truncate to Y whole years; preserve month phase",
        "confidence_sampling": "paired channels",
        "null_sampling": "independent channels",
        "confidence_method": "numpy.quantile(method='linear')",
        "confidence_level": confidence_level,
        "null_statistic": "P(R|A) - P(R)" if statistic == "conditional_probability" else "p_joint - product(p_i)",
        "alternative": alternative,
        "null_comparison": "inclusive within 1e-12 of the observed statistic; (1 + extreme) / (1 + B_valid)",
        "n_resamples": n_resamples,
        "seed": seed,
        "stream": "SeedSequence with SHA256 of inference support; separate paired/null streams",
        "minimum_years": 30,
        "minimum_blocks": 5,
        "minimum_occurrences": 10,
        "minimum_successes_interval": 5,
        "minimum_expected_successes_null": 5,
        "minimum_valid_fraction": 0.95,
        "p_value_scope": "local, unadjusted; not field significance",
        "quality_flags": dict(_FLAGS),
        "evidence": (
            "synthetic characterization and regression only; no external validation; screening evidence only "
            "for hazards independent of each other with serially independent or 6-month-overlap persistence "
            "at 200 years; the 36-month-persistence characterization failed a predeclared gate; shorter records, "
            "other persistence, other hazard frequencies, and lagged conditioning were not characterized"
        ),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "harmonization": _harmonization_record(hazards.metadata),
    }
    return BootstrapResult(
        statistic=statistic,
        groupby=groupby,
        **{name: _immutable(value.reshape(groups + cells)) for name, value in floats.items()},
        **{name: _immutable(value.reshape(groups + cells)) for name, value in counts.items()},
        n_years=_immutable(years_used.reshape(cells)),
        block_length_years=_immutable(lengths.reshape(cells)),
        inference_start=_immutable(starts.reshape(cells)),
        inference_end=_immutable(ends.reshape(cells)),
        metadata=_freeze(manifest),
    )


def bootstrap_dependence(
    hazards: AlignedHazards,
    *,
    statistic: Statistic,
    given: str | None = None,
    target: str | None = None,
    lag_window: tuple[int, int] = (0, 0),
    groupby: GroupBy = "month",
    block_length_years: int | None = None,
    n_resamples: int = 1999,
    confidence_level: float = 0.95,
    alternative: Alternative = "greater",
    seed: int,
) -> BootstrapResult:
    """Estimate monthly uncertainty and test calendar-conditioned independence.

    Args:
        hazards: Two or three declared monthly hazards. Validate manual instances
            too. Missing months split complete-year segments; they never become
            non-occurrences. Partial boundary years do not enter inference.
        statistic: Joint probability, likelihood multiplication factor, or
            conditional probability. The null uses excess association, not LMF.
        given: Anchor source for conditional inference only.
        target: Response source for conditional inference only; distinct from given.
        lag_window: Inclusive forward offsets in calendar months. Compute responses
            before resampling. Only conditional inference accepts nonzero offsets.
        groupby: Calendar month (default) or an explicitly pooled mixture.
        block_length_years: Positive integer override for ceil(Y ** (1/3)).
        n_resamples: Positive integer count, separately for paired and null samples.
        confidence_level: Interval probability strictly between zero and one.
        alternative: Amplification (greater, default) or suppression (less).
        seed: Required nonnegative integer; use a local generator. Equal support
            definitions use equal index streams regardless of spatial tile boundaries.

    Returns:
        A BootstrapResult of immutable arrays. Confidence bounds require five
        observed successes. Null tests instead require five expected successes
        under independence. Both require 30 complete years, five nonoverlapping
        blocks, ten occurrences per channel, and 95% valid resamples.
        Zero joint counts can support suppression, but not a zero-width interval.

    Raises:
        InvalidArgumentError: For daily inputs, unsupported choices, malformed
            options, or invalid conditional roles. Prepared-input validation also
            raises the structured type, shape, and coordinate exceptions used by
            joint_statistics(). No bootstrap dependency or native kernel is used.

    Warns:
        ClimateIndicesWarning: For a support or resampling quality condition,
            including fewer than ten nonoverlapping blocks.

    Notes:
        All p-values are local and unadjusted. These guards do not detect long
        memory, nonstationarity, or impact relevance. Synthetic characterization
        gives screening evidence for the intervals and p-values only for serially
        independent or 6-month-overlap sources whose hazards are independent of
        each other, with hazard frequencies of 0.35 to 0.55 and no lag, at 200
        years and 399 resamples. For 36-month persistence one predeclared coverage
        gate failed (12-year blocks, LMF) and null rejection exceeded the nominal
        5% at every block length. Other frequencies, lagged conditioning, and
        shorter records were not characterized (see VALIDATION.md). Inspect block
        sensitivity at max(1, floor(L/2)), L, and 2L. Seeded bitwise identity across dependency
        versions is not promised. See ADR-0022 and the synthetic characterization
        configuration in scripts/characterize_compound_bootstrap.py.
    """
    log = _logger.bind(
        index_type="compound_bootstrap_dependence",
        statistic=statistic,
        source_names=hazards.names if isinstance(hazards, AlignedHazards) else (),
        cadence=getattr(hazards, "cadence", None),
        groupby=groupby,
        seed=seed,
        n_resamples=n_resamples,
        block_length_years=block_length_years,
    )
    log.info("calculation_started")
    t0 = perf_counter()
    try:
        _check_choice(
            "statistic", statistic, ("joint_probability", "likelihood_multiplication_factor", "conditional_probability")
        )
        _check_choice("groupby", groupby, ("month", "pooled"))
        _check_choice("alternative", alternative, ("greater", "less"))
        window = _check_lag_window(lag_window)
        if block_length_years is not None:
            _integer("block_length_years", block_length_years, 1)
        _integer("n_resamples", n_resamples, 1)
        _integer("seed", seed, 0)
        if (
            isinstance(confidence_level, bool)
            or not isinstance(confidence_level, (int, float))
            or not 0 < confidence_level < 1
        ):
            raise InvalidArgumentError(
                "confidence_level must be a finite number strictly between zero and one.",
                argument_name="confidence_level",
            )
        result = _bootstrap(
            hazards,
            statistic=statistic,
            given=given,
            target=target,
            lag_window=window,
            groupby=groupby,
            block_length_years=block_length_years,
            n_resamples=n_resamples,
            confidence_level=float(confidence_level),
            alternative=alternative,
            seed=seed,
        )
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise
    log.info(
        "calculation_completed", duration_ms=round((perf_counter() - t0) * 1000, 2), output_shape=result.estimate.shape
    )
    return result

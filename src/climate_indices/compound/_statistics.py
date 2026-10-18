"""Joint frequencies and likelihood multiplication factor on one eligible support.

``joint_statistics()`` counts two or three declared hazards on the **complete-case
support** of an :class:`~climate_indices.compound.AlignedHazards`: the observed
time steps where every selected source is valid. It reports marginal and joint
counts, their frequencies, the independence product, and the likelihood
multiplication factor (LMF) in one count pass, so no two denominators can drift
apart.

Marginals are **observed** frequencies, never the declared threshold or
``1 - threshold``: a threshold is not an exceedance probability for a discrete or
poorly fitted series. Frequency means occupied eligible steps, not independent
events per year.

- ``LMF = p_joint / product(p_i)`` and cannot be negative.
- A zero joint count with positive marginals gives ``LMF = 0``; no pseudocount is
  ever added.
- A zero marginal count makes LMF undefined: ``NaN`` plus the
  ``undefined_denominator`` quality flag, never infinity and never an estimate.
- ``LMF = 1`` means the threshold frequencies match the independence product. It
  does **not** establish independence of the underlying variables.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from climate_indices.compound._harmonize import (
    _CADENCES,
    _UNITS,
    AlignedHazards,
    _check_choice,
    _freeze,
    _immutable,
)
from climate_indices.exceptions import (
    ClimateIndicesWarning,
    CoordinateValidationError,
    DataShapeError,
    InputTypeError,
    InvalidArgumentError,
)
from climate_indices.logging_config import get_logger, log_calculation_failure

_logger = get_logger(__name__)

GroupBy: TypeAlias = Literal["month", "pooled"]

_GROUPINGS = ("month", "pooled")
_MIN_SOURCES = 2
_MAX_SOURCES = 3
_MONTHS = 12
_SHORT_RECORD_YEARS = 30
_SPARSE_OCCURRENCES = 10
_MANIFEST_SCHEMA_VERSION = 1

# numeric quality flags, decoded by name in the result manifest
_INSUFFICIENT_SUPPORT = 1
_UNDEFINED_DENOMINATOR = 2
_FLAG_MEANINGS = {"insufficient_support": _INSUFFICIENT_SUPPORT, "undefined_denominator": _UNDEFINED_DENOMINATOR}


@dataclass(frozen=True, eq=False)
class JointResult:
    """Counts and probabilities for two or three hazards on one eligible support.

    Instances compare by identity: the fields are arrays. Every array is read-only
    and was built by :func:`joint_statistics`; mutate neither it nor ``metadata``.

    Each field is shaped ``(month, *cells)`` when ``groupby="month"`` and
    ``(*cells)`` when ``groupby="pooled"``, except the per-source fields, whose
    leading axis is the source. The month axis is index 0-11, where 0 is January.
    A point series has no ``*cells`` axis.

    Args:
        names: Source names in input order; the leading axis of ``n_marginal``.
        groupby: ``"month"`` for calendar-month groups or ``"pooled"`` for one
            mixture over every eligible step.
        n_valid: Eligible steps in the group, counted once per group rather than
            once per source.
        n_joint: Eligible steps where every selected hazard occurs.
        n_marginal: Eligible steps where each source hazard occurs.
        marginal_frequency: ``n_marginal / n_valid``; ``NaN`` when ``n_valid`` is
            zero or below ``min_samples``.
        joint_probability: ``n_joint / n_valid``; ``NaN`` when ``n_valid`` is zero
            or below ``min_samples``.
        independent_probability: The product of ``marginal_frequency``. For three
            sources this assumes mutual independence, not only pairwise.
        likelihood_multiplication_factor: ``joint_probability /
            independent_probability``; ``NaN`` when a marginal count is zero or the
            group is below ``min_samples``.
        quality_flags: Bitwise OR of ``insufficient_support`` (fewer than
            ``min_samples`` eligible steps) and ``undefined_denominator`` (a zero
            marginal count). ``metadata["quality_flags"]`` names each bit.
        metadata: Immutable, JSON-compatible manifest of the grouping, support
            threshold, estimands, and the harmonization manifest the counts rest on.
    """

    names: tuple[str, ...]
    groupby: GroupBy
    n_valid: npt.NDArray[np.int64]
    n_joint: npt.NDArray[np.int64]
    n_marginal: npt.NDArray[np.int64]
    marginal_frequency: npt.NDArray[np.float64]
    joint_probability: npt.NDArray[np.float64]
    independent_probability: npt.NDArray[np.float64]
    likelihood_multiplication_factor: npt.NDArray[np.float64]
    quality_flags: npt.NDArray[np.int64]
    metadata: Mapping[str, Any]


def _check_min_samples(min_samples: int) -> int:
    """Require a positive integer support threshold; a bool is not an integer here."""
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise InvalidArgumentError(
            "min_samples must be a positive integer.",
            argument_name="min_samples",
            argument_value=repr(min_samples),
            valid_values="an integer >= 1",
        )
    return min_samples


def _validate_hazards(hazards: object) -> AlignedHazards:
    """Validate a caller-supplied :class:`AlignedHazards`; a dataclass is not proof of validity."""
    if not isinstance(hazards, AlignedHazards):
        raise InvalidArgumentError(
            "hazards must be an AlignedHazards instance built by compound.harmonize().",
            argument_name="hazards",
            argument_value=type(hazards).__name__,
            valid_values="AlignedHazards",
        )
    names = hazards.names
    if not isinstance(names, tuple) or not all(isinstance(name, str) for name in names):
        raise InvalidArgumentError(
            "hazards.names must be a tuple of source names.",
            argument_name="hazards",
            argument_value=repr(names),
            valid_values="a tuple of strings",
        )
    if not _MIN_SOURCES <= len(names) <= _MAX_SOURCES:
        raise InvalidArgumentError(
            f"joint_statistics() needs two or three sources; got {len(names)}.",
            argument_name="hazards",
            argument_value=f"{len(names)} names",
            valid_values="two or three source names",
        )
    if len(set(names)) != len(names):
        raise InvalidArgumentError(
            f"hazards.names must be unique; got {list(names)}.",
            argument_name="hazards",
            valid_values="distinct source names",
        )
    _check_choice("hazards.cadence", hazards.cadence, _CADENCES)
    if not isinstance(hazards.metadata, Mapping):
        raise InvalidArgumentError(
            "hazards.metadata must be a mapping.",
            argument_name="hazards",
            argument_value=type(hazards.metadata).__name__,
            valid_values="Mapping[str, Any]",
        )

    time = hazards.time
    if not isinstance(time, np.ndarray) or time.dtype != np.dtype("datetime64[D]"):
        raise InputTypeError(
            f"hazards.time must be a datetime64[D] array of observed civil days, not {type(time).__name__} "
            f"with dtype {getattr(time, 'dtype', None)}.",
            expected_type=np.datetime64,
            actual_type=type(time),
        )
    if time.ndim != 1:
        raise DataShapeError(
            f"hazards.time must be one-dimensional; got {time.ndim} dimensions.",
            expected_shape="(time,)",
            actual_shape=time.shape,
        )
    if time.shape[0] < 1 or np.isnat(time).any():
        raise CoordinateValidationError(
            "hazards.time must hold at least one non-NaT date; a manual AlignedHazards is validated, not trusted.",
            coordinate_name="time",
            reason="NaT or no dates",
        )
    if np.any(np.diff(time.astype("datetime64[D]").astype(np.int64)) <= 0):
        raise CoordinateValidationError(
            "hazards.time must be strictly increasing; duplicate or unordered dates would double-count a step.",
            coordinate_name="time",
            reason="unordered or duplicate dates",
        )
    if hazards.cadence == "monthly" and np.any(time != time.astype("datetime64[M]")):
        raise CoordinateValidationError(
            "hazards.time must label each monthly step by the first civil day of its calendar month; "
            "two labels in one month would count that month twice.",
            coordinate_name="time",
            reason="monthly step not normalized to its first civil day",
        )

    for label, array in (("exceedance", hazards.exceedance), ("valid", hazards.valid)):
        if not isinstance(array, np.ndarray) or array.dtype != np.bool_ or np.ma.isMaskedArray(array):
            raise InputTypeError(
                f"hazards.{label} must be a plain Boolean numpy array, not {type(array).__name__}.",
                expected_type=np.ndarray,
                actual_type=type(array),
            )
        if array.ndim < 2:
            raise DataShapeError(
                f"hazards.{label} must be shaped (hazard, time, *cells); got {array.ndim} dimensions.",
                expected_shape=f"({len(names)}, {time.shape[0]}, *cells)",
                actual_shape=array.shape,
            )
    if hazards.exceedance.shape[0] != len(names):
        raise DataShapeError(
            f"hazards.exceedance must hold one row per source name; got {hazards.exceedance.shape[0]} rows "
            f"for {len(names)} names.",
            expected_shape=f"({len(names)}, {time.shape[0]}, *cells)",
            actual_shape=hazards.exceedance.shape,
        )
    if hazards.exceedance.shape[1] != time.shape[0]:
        raise DataShapeError(
            f"hazards.exceedance must hold one column per observed date; got {hazards.exceedance.shape[1]} "
            f"columns for {time.shape[0]} dates.",
            expected_shape=f"({len(names)}, {time.shape[0]}, *cells)",
            actual_shape=hazards.exceedance.shape,
        )
    if hazards.exceedance.shape != hazards.valid.shape:
        raise DataShapeError(
            "hazards.exceedance and hazards.valid must have the same shape.",
            expected_shape=str(hazards.exceedance.shape),
            actual_shape=hazards.valid.shape,
        )
    return hazards


def _month_index(time: npt.NDArray[np.datetime64]) -> npt.NDArray[np.int64]:
    """Return each step's calendar month as index 0-11, from the observed dates alone."""
    return (time.astype("datetime64[M]").astype(np.int64) % _MONTHS).astype(np.int64)


def _counts(mask: npt.NDArray[np.bool_], months: npt.NDArray[np.int64], groupby: str) -> npt.NDArray[np.int64]:
    """Count ``mask`` over the leading time axis, per calendar month or pooled."""
    if groupby == "pooled":
        return np.asarray(mask.sum(axis=0, dtype=np.int64))
    counts = np.empty((_MONTHS, *mask.shape[1:]), dtype=np.int64)
    for month in range(_MONTHS):
        counts[month] = mask[months == month].sum(axis=0, dtype=np.int64)
    return counts


def _safe_rate(counts: npt.NDArray[np.int64], denominator: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Return ``counts / denominator`` broadcast against ``counts``, ``NaN`` where the denominator is zero."""
    numerators = np.asarray(counts, dtype=np.float64)
    spread = np.broadcast_to(np.asarray(denominator, dtype=np.float64), numerators.shape)
    rate = np.full(numerators.shape, np.nan, dtype=np.float64)
    np.divide(numerators, spread, out=rate, where=spread > 0)
    return rate


def _undefined(estimate: npt.NDArray[np.float64], sufficient: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """Return ``estimate`` with ``NaN`` wherever the group lacks ``min_samples`` eligible steps."""
    return np.where(np.broadcast_to(sufficient, estimate.shape), estimate, np.nan)


def _observed_years(time: npt.NDArray[np.datetime64], eligible: npt.NDArray[np.bool_]) -> int:
    """Return how many calendar years hold at least one eligible step anywhere in the record."""
    observed = np.any(eligible, axis=tuple(range(1, eligible.ndim))) if eligible.ndim > 1 else eligible
    return int(np.unique(time[observed].astype("datetime64[Y]")).shape[0])


def _warn_support(
    time: npt.NDArray[np.datetime64],
    eligible: npt.NDArray[np.bool_],
    n_valid: npt.NDArray[np.int64],
    n_marginal: npt.NDArray[np.int64],
    min_samples: int,
) -> None:
    """Warn once about a short record, short support, sparse occurrences, or a marginal that never occurs."""
    observed_years = _observed_years(time, eligible)
    if observed_years < _SHORT_RECORD_YEARS:
        warnings.warn(
            f"The record holds {observed_years} observed calendar year(s); {_SHORT_RECORD_YEARS} are expected "
            "before a compound statistic is read as a rate.",
            ClimateIndicesWarning,
            stacklevel=4,
        )
    sparse = int(np.count_nonzero((n_valid > 0) & (n_marginal.min(axis=0) < _SPARSE_OCCURRENCES)))
    if sparse:
        warnings.warn(
            f"{sparse} group(s) have fewer than {_SPARSE_OCCURRENCES} occurrences in a marginal; such sparse "
            "frequencies are unstable. This is a product guardrail, not an adequacy threshold.",
            ClimateIndicesWarning,
            stacklevel=4,
        )
    short_support = int(np.count_nonzero((n_valid > 0) & (n_valid < min_samples)))
    if short_support:
        warnings.warn(
            f"{short_support} group(s) have fewer than min_samples={min_samples} eligible observations; "
            "their estimates are NaN. This is a product guardrail, not an adequacy threshold.",
            ClimateIndicesWarning,
            stacklevel=4,
        )
    undefined = int(np.count_nonzero((n_valid > 0) & (n_marginal.min(axis=0) == 0)))
    if undefined:
        warnings.warn(
            f"{undefined} group(s) have a zero marginal count, so the likelihood multiplication factor is "
            "undefined there; no pseudocount is added.",
            ClimateIndicesWarning,
            stacklevel=4,
        )


def _joint_statistics(hazards: AlignedHazards, *, groupby: GroupBy, min_samples: int) -> JointResult:
    """Validate, count, and estimate; see :func:`joint_statistics`."""
    hazards = _validate_hazards(hazards)
    months = _month_index(hazards.time)

    # one complete-case support for every numerator: a step where any source is unknown is
    # eligible for none of them, so no marginal can exceed n_valid
    valid = hazards.valid
    eligible = np.all(valid, axis=0)
    occurrence = hazards.exceedance & eligible
    joint = np.all(occurrence, axis=0)

    n_valid = _counts(eligible, months, groupby)
    n_joint = _counts(joint, months, groupby)
    n_marginal = np.stack([_counts(row, months, groupby) for row in occurrence])
    # below min_samples the counts stand but every estimate is undefined
    sufficient = np.asarray(n_valid >= min_samples)
    marginal_frequency = _undefined(_safe_rate(n_marginal, n_valid), sufficient)
    joint_probability = _undefined(_safe_rate(n_joint, n_valid), sufficient)
    independent_probability = np.prod(marginal_frequency, axis=0)
    likelihood_multiplication_factor = np.full(independent_probability.shape, np.nan, dtype=np.float64)
    np.divide(
        joint_probability,
        independent_probability,
        out=likelihood_multiplication_factor,
        where=independent_probability > 0,
    )
    likelihood_multiplication_factor = _undefined(likelihood_multiplication_factor, sufficient)
    flagged = np.where(n_valid < min_samples, _INSUFFICIENT_SUPPORT, 0) | np.where(
        n_marginal.min(axis=0) == 0, _UNDEFINED_DENOMINATOR, 0
    )
    quality_flags = np.asarray(flagged, dtype=np.int64)

    _warn_support(hazards.time, eligible, n_valid, n_marginal, min_samples)

    unit = "month" if groupby == "month" else None
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "function": "joint_statistics",
        "groupby": groupby,
        "min_samples": min_samples,
        "calendar": hazards.metadata.get("calendar", "unknown"),
        "cadence": hazards.cadence,
        "time_start": np.datetime_as_string(hazards.time[0].astype(_UNITS[hazards.cadence]), unit="D"),
        "time_end": np.datetime_as_string(hazards.time[-1].astype(_UNITS[hazards.cadence]), unit="D"),
        "n_steps": int(hazards.time.shape[0]),
        "grouped_by": unit or "pooled over every eligible step",
        "eligibility": (
            "A step is eligible when every selected source is valid there (complete case). An invalid step is "
            "never counted as a non-exceedance, and no step is dropped from the record."
        ),
        "estimands": (
            "n_marginal and n_joint count occupied eligible steps, not independent events per year. "
            "Frequencies are counts divided by n_valid on the same support. independent_probability is the "
            "product of the observed marginal frequencies, which assumes mutual independence for three sources, "
            "and LMF is joint_probability divided by it."
        ),
        "lmf_rule": (
            "LMF is nonnegative. A zero joint count with positive marginals gives zero; a zero marginal count "
            "gives NaN and the undefined_denominator flag. LMF equal to one means the threshold frequencies "
            "match the independence product; it does not establish independence."
        ),
        "quality_flags": dict(_FLAG_MEANINGS),
        "harmonization": hazards.metadata,
    }
    return JointResult(
        names=hazards.names,
        groupby=groupby,
        n_valid=_immutable(n_valid),
        n_joint=_immutable(n_joint),
        n_marginal=_immutable(n_marginal),
        marginal_frequency=_immutable(marginal_frequency),
        joint_probability=_immutable(joint_probability),
        independent_probability=_immutable(independent_probability),
        likelihood_multiplication_factor=_immutable(likelihood_multiplication_factor),
        quality_flags=_immutable(quality_flags),
        metadata=_freeze(manifest),
    )


def joint_statistics(
    hazards: AlignedHazards,
    *,
    groupby: GroupBy = "month",
    min_samples: int = 30,
) -> JointResult:
    """Count two or three hazards on one eligible support and return frequencies and LMF.

    The support is the complete-case eligible mask ``V(t) = all selected sources
    are valid at t``. Counts, frequencies, the independence product, and the
    likelihood multiplication factor all come from one pass over that mask, so they
    cannot disagree about the denominator. An invalid step is never a
    non-exceedance, and all-missing cells return zero counts with ``NaN`` estimates
    and quality flags instead of failing their neighbors.

    ``groupby="month"`` reports calendar-month groups; ``groupby="pooled"``
    recounts every eligible step as one mixture. A pooled LMF is never an average
    of monthly LMFs. Each reported group needs ``min_samples`` eligible
    observations; below that threshold the counts remain and the estimates are
    ``NaN`` with the ``insufficient_support`` flag.

    Args:
        hazards: An :class:`AlignedHazards` with two or three sources, built by
            :func:`~climate_indices.compound.harmonize`. It is revalidated here:
            one row per name, one column per date, strictly increasing non-NaT
            dates, Boolean arrays, and a mapping manifest.
        groupby: ``"month"`` for calendar-month groups or ``"pooled"`` for one
            group over every eligible step.
        min_samples: Positive integer minimum eligible observations per reported
            group. The default 30 is a conservative product guardrail, not a
            scientific adequacy threshold.

    Returns:
        A :class:`JointResult` whose arrays are new and read-only. Counts are
        ``int64``, estimates are ``float64`` with ``NaN`` where undefined, and
        ``quality_flags`` is an ``int64`` bit field.

    Raises:
        InvalidArgumentError: For a non-``AlignedHazards`` argument, a source count
            other than two or three, duplicate names, an unsupported ``groupby``,
            or a non-positive ``min_samples``.
        InputTypeError: For a non-Boolean array or a non-``datetime64`` time vector.
        DataShapeError: For mismatched array shapes or a date vector that does not
            match the time axis.
        CoordinateValidationError: For NaT, duplicate, unordered, or empty dates, or
            a monthly label that is not the first civil day of its month.

    Warns:
        ClimateIndicesWarning: For a record holding fewer than 30 observed calendar
            years, a group below ``min_samples``, a group with fewer than ten
            occurrences in a marginal, or a group whose marginal count is zero.

    Example:
        >>> import numpy as np
        >>> from climate_indices import compound
        >>> months = np.arange("2000-01", "2000-05", dtype="datetime64[M]")
        >>> aligned = compound.harmonize(
        ...     {"drought": np.array([0.04, 0.50, 0.03, 0.50]), "heat": np.array([True, False, True, True])},
        ...     time={"drought": months, "heat": months},
        ...     cadence={"drought": "monthly", "heat": "monthly"},
        ...     input_scale={"drought": "probability", "heat": "indicator"},
        ...     thresholds={"drought": 0.1, "heat": 0.5},
        ...     directions={"drought": "below", "heat": "above"},
        ...     target_cadence="monthly",
        ... )
        >>> counts = compound.joint_statistics(aligned, groupby="pooled", min_samples=3)
        >>> int(counts.n_valid), int(counts.n_joint), int(counts.n_marginal[0]), int(counts.n_marginal[1])
        (4, 2, 2, 3)
        >>> print(round(float(counts.marginal_frequency[0]), 6), round(float(counts.likelihood_multiplication_factor), 6))
        0.5 1.333333
    """
    log = _logger.bind(
        index_type="compound_joint_statistics",
        n_sources=len(hazards.names) if isinstance(hazards, AlignedHazards) else 0,
        groupby=groupby,
        min_samples=min_samples,
    )
    log.info("calculation_started")
    t0 = perf_counter()
    try:
        _check_choice("groupby", groupby, _GROUPINGS)
        result = _joint_statistics(hazards, groupby=groupby, min_samples=_check_min_samples(min_samples))
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise
    log.info(
        "calculation_completed",
        duration_ms=round((perf_counter() - t0) * 1000.0, 2),
        output_shape=result.n_valid.shape,
    )
    return result

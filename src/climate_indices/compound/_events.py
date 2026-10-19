"""Concurrent and sequential matching of two hazard ``RunSet``s on one observed grid.

``compound_events()`` enumerates source-run pairs without combining their severity
measures. Two sources of runs are matched on one shared, complete cadence grid and one
observation-validity mask:

- *concurrent* matching keeps every pair whose inclusive intervals intersect and reports
  each pair's intersection and its ``overlap_duration`` in cadence steps;
- *sequential* matching keeps every pair with ``second.start > first.end``, reports the
  interval from the first run's onset through the second run's end, and reports the
  separating ``lag`` (``second.start - first.end``, so adjacent runs have lag one).

Runs are never filtered into a one-to-one assignment: an interval that intersects three
others produces three rows, and a sequential pair inside the lag window is kept even when
an earlier pair already matched. A pair whose separating interval contains an unknown
observation is not reported, because a cascade is never inferred across an unknown gap;
the count of those rejected bridges is recorded in the result metadata.

Every run boundary that touches the record boundary or an unknown neighbouring step is
*censored*: the run's observed duration may be shorter than the physical event, so the
four boundary flags are recorded per row and the pair's observed length is never
presented as a complete event duration.

Contracts (ADR-0022): result arrays are new, read-only, and stable when empty; ``RunSet``
is validated but never modified, and its fields other than the run indices are not
inspected. Pair counts are not independent observations and must not feed a binomial
significance test.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from time import perf_counter
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.compound._harmonize import (
    _CADENCES,
    _MANIFEST_SCHEMA_VERSION,
    _UNITS,
    _UNKNOWN,
    Cadence,
    _check_choice,
    _freeze,
    _immutable,
    _observed_index,
)
from climate_indices.exceptions import (
    CoordinateValidationError,
    DataShapeError,
    InputTypeError,
    InvalidArgumentError,
)
from climate_indices.logging_config import get_logger, log_calculation_failure
from climate_indices.runs import _VALID_DIRECTIONS, RunSet, _as_finite_threshold

_logger = get_logger(__name__)

__all__ = ["CompoundEventSet", "compound_events", "compound_events_xarray"]

EventMode: TypeAlias = Literal["concurrent", "sequential"]

_MODES: tuple[str, ...] = ("concurrent", "sequential")
_SOURCE_NAMES = ("first", "second")
# the declarations a RunSet does not retain; an absent one is recorded as unknown, never inferred
_SOURCE_CHOICES = ("threshold", "direction", "min_duration")
_MATCHING_RULES = {
    "concurrent": (
        "Every pair whose inclusive intervals intersect; start_index/end_index hold the intersection, "
        "overlap_duration counts its steps, and lag is NaN."
    ),
    "sequential": (
        "Every pair with second.start > first.end inside the lag window whose steps from first.end through "
        "second.start are all observed; start_index/end_index hold the first onset through the second end, lag is "
        "second.start - first.end, and overlap_duration is 0. A pair bridged by an unknown step is excluded and "
        "counted in n_unknown_bridges."
    ),
}
_CENSORING_RULE = (
    "A run boundary is censored when the step beyond it lies outside the record or is unknown; the four flags "
    "give the first source's left and right boundaries, then the second source's. No severity measure is combined."
)
_MISSINGNESS = (
    "An unknown step is never a non-event: a run may not cross one, and a sequential pair is not matched across one."
)


@dataclass(frozen=True, eq=False)
class CompoundEventSet:
    """Matched source-run pairs on one shared cadence grid.

    Rows appear in first source-run order and then in second source-run order. Every
    array is read-only, has one entry per row, and has a stable dtype when there are no
    rows.

    Instances compare by identity: the fields are arrays.

    Construction enforces that contract: ``mode`` must be supported, every array must hold
    one entry per row, and the arrays and the manifest are frozen even when a caller builds
    an instance directly instead of calling :func:`compound_events`.

    Args:
        mode: ``"concurrent"`` or ``"sequential"``, the matching rule that produced the
            rows.
        first_run_id: Zero-based position of each row's first-source run in ``first``,
            so ``first.start_index[first_run_id]`` recovers the source run.
        second_run_id: Zero-based position of each row's second-source run in
            ``second``.
        start_index: Index of each row's first reported step in the shared time axis.
            The intersection of the two runs for ``mode="concurrent"``; the first run's
            onset through the second run's end for ``mode="sequential"``.
        end_index: Index of each row's last reported step, inclusive.
        overlap_duration: Number of steps the two runs share. Zero in sequential mode,
            where the runs are disjoint by definition; the reported interval is not an
            invented joint duration.
        lag: ``second.start - first.end`` in cadence steps, so adjacent sequential runs
            have lag one. NaN in concurrent mode.
        first_left_censored: ``True`` when the first run's onset touches the record start
            or an unknown neighbouring step, so the physical onset is unknown.
        first_right_censored: ``True`` when the first run's end touches the record end or
            an unknown neighbouring step.
        second_left_censored: As ``first_left_censored``, for the second run.
        second_right_censored: As ``first_right_censored``, for the second run.
        metadata: Immutable, JSON-compatible manifest of the calendar, the matching,
            censoring, and missingness rules, the effective lag window, the observed and
            unknown step counts, the run and pair counts, the rejected unknown-bridge
            count, and each source's declared choices with unavailable ones recorded as
            ``"unknown"``.
    """

    mode: EventMode
    first_run_id: npt.NDArray[np.int64]
    second_run_id: npt.NDArray[np.int64]
    start_index: npt.NDArray[np.int64]
    end_index: npt.NDArray[np.int64]
    overlap_duration: npt.NDArray[np.int64]
    lag: npt.NDArray[np.float64]
    first_left_censored: npt.NDArray[np.bool_]
    first_right_censored: npt.NDArray[np.bool_]
    second_left_censored: npt.NDArray[np.bool_]
    second_right_censored: npt.NDArray[np.bool_]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Enforce the row contract and freeze the arrays and the manifest.

        ``frozen=True`` stops attribute rebinding only, so the generated constructor would
        otherwise accept an unsupported mode, columns of differing lengths, writable arrays,
        and a mutable manifest. A directly constructed instance is made to keep the same
        contract as one returned by :func:`compound_events`.
        """
        if self.mode not in _MODES:
            raise InvalidArgumentError(
                f"mode must be one of {list(_MODES)}.",
                argument_name="mode",
                argument_value=repr(self.mode),
                valid_values=", ".join(repr(choice) for choice in _MODES),
            )
        if not isinstance(self.metadata, Mapping):
            raise InvalidArgumentError(
                "metadata must be a mapping.",
                argument_name="metadata",
                argument_value=type(self.metadata).__name__,
                valid_values="Mapping[str, Any]",
            )
        rows: int | None = None
        for field in fields(self):
            if field.name in ("mode", "metadata"):
                continue
            values = np.asarray(getattr(self, field.name))
            if values.ndim != 1:
                raise DataShapeError(
                    f"{field.name} must be one-dimensional with one entry per pair.",
                    expected_shape="(pair,)",
                    actual_shape=values.shape,
                )
            if rows is None:
                rows = int(values.shape[0])
            elif values.shape[0] != rows:
                raise DataShapeError(
                    f"every array must hold one entry per pair; {field.name} has {values.shape[0]} and "
                    f"the first has {rows}.",
                    expected_shape=f"({rows},)",
                    actual_shape=values.shape,
                )
            if values.flags.writeable:
                object.__setattr__(self, field.name, _immutable(values))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))

    def __len__(self) -> int:
        """Return the number of matched pairs."""
        return int(self.first_run_id.size)


@dataclass(frozen=True)
class _SourceRuns:
    """One validated source's runs, with the boundary censoring computed once per run."""

    start: npt.NDArray[np.int64]
    end: npt.NDArray[np.int64]
    left_censored: npt.NDArray[np.bool_]
    right_censored: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class _Matched:
    """The matched rows, still as loose arrays, before the result object freezes them."""

    first_run_id: npt.NDArray[np.int64]
    second_run_id: npt.NDArray[np.int64]
    start_index: npt.NDArray[np.int64]
    end_index: npt.NDArray[np.int64]
    overlap_duration: npt.NDArray[np.int64]
    lag: npt.NDArray[np.float64]
    first_left_censored: npt.NDArray[np.bool_]
    first_right_censored: npt.NDArray[np.bool_]
    second_left_censored: npt.NDArray[np.bool_]
    second_right_censored: npt.NDArray[np.bool_]
    unknown_bridges: int


def _observation_mask(valid: object, steps: int) -> npt.NDArray[np.bool_]:
    """Require the shared observation-validity mask to be plain Boolean and one per step."""
    if np.ma.isMaskedArray(valid):
        raise InputTypeError(
            "valid must be a plain Boolean array; a masked validity mask is ambiguous, so mark unknown steps False.",
            expected_type=np.bool_,
            actual_type=type(valid),
        )
    mask = np.asarray(valid)
    if mask.dtype.kind != "b":
        raise InputTypeError(
            f"valid must have Boolean dtype, not {mask.dtype}; unknown observations are False, not 0 or NaN.",
            expected_type=np.bool_,
            actual_type=mask.dtype.type,
        )
    if mask.shape != (steps,):
        raise DataShapeError(
            "valid must hold one Boolean per step of time.",
            expected_shape=f"({steps},)",
            actual_shape=mask.shape,
        )
    return mask


def _lag_window(mode: str, lag_window: object) -> tuple[int, int | None]:
    """Resolve the inclusive lag window; ``None`` means every lag at or above one."""
    if mode == "concurrent":
        if lag_window is not None:
            raise InvalidArgumentError(
                "lag_window applies to sequential matching only.",
                argument_name="lag_window",
                argument_value=repr(lag_window),
                valid_values="None when mode='concurrent'",
            )
        return 1, None
    if lag_window is None:
        return 1, None
    if (
        not isinstance(lag_window, tuple)
        or len(lag_window) != 2
        or any(isinstance(bound, bool) or not isinstance(bound, (int, np.integer)) for bound in lag_window)
    ):
        raise InvalidArgumentError(
            "lag_window must be a (low, high) tuple of integers.",
            argument_name="lag_window",
            argument_value=repr(lag_window),
            valid_values="(low, high) with 1 <= low <= high, or None",
        )
    low, high = int(lag_window[0]), int(lag_window[1])
    if low < 1 or high < low:
        raise InvalidArgumentError(
            "lag_window must satisfy 1 <= low <= high; sequential runs are separated by at least one step.",
            argument_name="lag_window",
            argument_value=f"({low}, {high})",
            valid_values="(low, high) with 1 <= low <= high",
        )
    return low, high


def _integer_field(name: str, runs: RunSet, field: str) -> npt.NDArray[np.int64]:
    """Return one run-index field as int64, rejecting a masked, wrapped, or non-vector field."""
    raw = getattr(runs, field)
    if np.ma.isMaskedArray(raw):
        raise InputTypeError(
            f"{name}.{field} must be a plain integer array; a masked run index is ambiguous.",
            expected_type=np.int64,
            actual_type=type(raw),
        )
    values = np.asarray(raw)
    if values.ndim != 1:
        raise DataShapeError(
            f"{name}.{field} must be one-dimensional with one entry per run.",
            expected_shape="(run,)",
            actual_shape=values.shape,
        )
    if values.dtype.kind not in "iu":
        raise InputTypeError(
            f"{name}.{field} must hold integers, not {values.dtype}.",
            expected_type=np.int64,
            actual_type=values.dtype.type,
        )
    if values.dtype.kind == "u" and values.size and int(values.max()) > np.iinfo(np.int64).max:
        # a wrapped value would otherwise pass the duration and bounds checks as a negative index
        raise InputTypeError(
            f"{name}.{field} holds values above the int64 range.",
            expected_type=np.int64,
            actual_type=values.dtype.type,
        )
    return values.astype(np.int64)


def _validated_runs(
    name: str,
    runs: object,
    valid: npt.NDArray[np.bool_],
    valid_prefix: npt.NDArray[np.int64],
    steps: int,
) -> _SourceRuns:
    """Validate one source's runs against the grid and compute their boundary censoring.

    A dataclass is not proof of validity, so a manually constructed ``RunSet`` is checked
    as strictly as one returned by :func:`climate_indices.runs.identify_runs`: every field
    must hold one entry per run, each run must be a contiguous observed spell inside the
    record, and the mask pads the record edges for censoring. Source-run IDs are the
    caller's indices and are never renumbered.
    """
    if not isinstance(runs, RunSet):
        raise InputTypeError(
            f"{name} must be a RunSet.",
            expected_type=RunSet,
            actual_type=type(runs),
        )
    start = _integer_field(name, runs, "start_index")
    end = _integer_field(name, runs, "end_index")
    duration = _integer_field(name, runs, "duration")
    mismatched: dict[str, tuple[int, ...]] = {}
    for field in fields(RunSet):
        try:
            shape = np.shape(getattr(runs, field.name))
        except (TypeError, ValueError) as exc:
            raise DataShapeError(
                f"{name}.{field.name} must be one array with one entry per run; it is ragged.",
                expected_shape=f"({start.size},)",
            ) from exc
        if shape != (start.size,):
            mismatched[field.name] = shape
    if mismatched:
        reported = ", ".join(f"{field} ({', '.join(map(str, shape))})" for field, shape in mismatched.items())
        raise DataShapeError(
            f"{name} must hold one entry per run in every field; start_index has {start.size} "
            f"and {reported} disagree with it.",
            expected_shape=f"({start.size},)",
            actual_shape=next(iter(mismatched.values())),
        )
    if start.size == 0:
        empty = np.zeros(0, dtype=np.bool_)
        return _SourceRuns(start, end, empty, empty)
    # start > end is tested directly: end - start + 1 wraps in int64 when the two are about 2**64 apart
    inconsistent = (duration < 1) | (start > end) | (end - start + 1 != duration)
    if inconsistent.any():
        bad = int(np.argmax(inconsistent))
        raise InvalidArgumentError(
            f"{name} must assert positive durations equal to end_index - start_index + 1.",
            argument_name=name,
            argument_value=f"run {bad}: start={int(start[bad])}, end={int(end[bad])}, duration={int(duration[bad])}",
            valid_values="duration == end_index - start_index + 1 >= 1",
        )
    if start[0] < 0 or end[-1] >= steps:
        raise InvalidArgumentError(
            f"{name} runs must lie inside the {steps}-step time axis.",
            argument_name=name,
            argument_value=f"start_index min={int(start.min())}, end_index max={int(end.max())}",
            valid_values=f"0 <= start_index <= end_index <= {steps - 1}",
        )
    unordered = np.flatnonzero(start[1:] <= end[:-1])
    if unordered.size:
        bad = int(unordered[0])
        raise InvalidArgumentError(
            f"{name} runs must be sorted and nonoverlapping.",
            argument_name=name,
            argument_value=f"run {bad + 1} starts at {int(start[bad + 1])}, at or before run {bad} ends at {int(end[bad])}",
            valid_values="strictly increasing, nonoverlapping runs",
        )
    unknown_steps = duration - (valid_prefix[end + 1] - valid_prefix[start])
    crossing = np.flatnonzero(unknown_steps > 0)
    if crossing.size:
        raise InvalidArgumentError(
            f"{name} run {int(crossing[0])} crosses an unknown observation; a run is a contiguous observed spell.",
            argument_name=name,
            argument_value=f"run {int(crossing[0])} spans [{int(start[crossing[0]])}, {int(end[crossing[0]])}]",
            valid_values="every step of a run is observed",
        )
    # the padding turns both record edges into unknown neighbours, so one shift covers both cases
    padded = np.concatenate(([False], valid, [False]))
    return _SourceRuns(start, end, ~padded[start], ~padded[end + 2])


def _source_entry(name: str, metadata: object) -> dict[str, Any]:
    """Return one source's JSON-compatible choices, with unavailable ones recorded as unknown."""
    if metadata is not None and not isinstance(metadata, Mapping):
        raise InvalidArgumentError(
            f"source_metadata[{name!r}] must be a JSON-compatible mapping.",
            argument_name="source_metadata",
            argument_value=type(metadata).__name__,
            valid_values="Mapping[str, object]",
        )
    declared: Mapping[str, Any] = metadata if isinstance(metadata, Mapping) else {}
    try:
        copied: dict[str, Any] = json.loads(json.dumps(dict(declared), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError(
            f"source_metadata[{name!r}] must be a JSON-compatible mapping.",
            argument_name="source_metadata",
            valid_values="a mapping of JSON strings, numbers, booleans, null, lists, and mappings",
        ) from exc
    entry = {choice: copied.pop(choice, _UNKNOWN) for choice in _SOURCE_CHOICES}
    if entry["direction"] != _UNKNOWN:
        _check_choice(f"source_metadata[{name!r}]['direction']", entry["direction"], _VALID_DIRECTIONS)
    if entry["threshold"] != _UNKNOWN:
        try:
            _as_finite_threshold(entry["threshold"])
        except InvalidArgumentError as exc:
            raise InvalidArgumentError(
                f"source_metadata[{name!r}]['threshold'] must be a finite number.",
                argument_name="source_metadata",
                argument_value=repr(entry["threshold"]),
                valid_values="a finite int or float",
            ) from exc
    if entry["min_duration"] != _UNKNOWN and (type(entry["min_duration"]) is not int or entry["min_duration"] < 1):
        raise InvalidArgumentError(
            f"source_metadata[{name!r}]['min_duration'] must be a positive integer.",
            argument_name="source_metadata",
            argument_value=repr(entry["min_duration"]),
            valid_values="integer >= 1",
        )
    entry.update(copied)
    return entry


_Rows = tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]
# remaining rows the caller will accept, and the total limit it chose, so the error can name both
_Budget = tuple[int, int]


def _row_limit(limit: int, remaining: int, count: int, *, mode: str) -> InvalidArgumentError:
    """Build the error for one source run whose candidates would exceed the caller's row budget."""
    advice = "select a smaller subset or raise min_duration"
    if mode == "sequential":
        advice += ", or bound lag_window"
    return InvalidArgumentError(
        f"One source run matches {count} {'run' if count == 1 else 'runs'} of the other source, beyond the "
        f"{limit}-row limit this result allows ({remaining} {'row' if remaining == 1 else 'rows'} remaining). "
        f"An all-pairs match grows quadratically; to fit the limit, {advice}.",
        argument_name="hazards",
        valid_values=f"a subset whose catalog holds at most {limit} rows",
    )


def _concurrent_rows(second: _SourceRuns, onset: int, ending: int, *, budget: _Budget | None = None) -> _Rows | None:
    """Return the second runs that overlap one first run as (matched, starts, ends, overlaps, lags)."""
    lower = int(np.searchsorted(second.end, onset, side="left"))
    upper = int(np.searchsorted(second.start, ending, side="right"))
    if lower >= upper:
        return None
    # checked before any array is built, so a quadratic fanout fails instead of allocating it
    if budget is not None and upper - lower > budget[0]:
        raise _row_limit(budget[1], budget[0], upper - lower, mode="concurrent")
    starts = np.maximum(onset, second.start[lower:upper])
    ends = np.minimum(ending, second.end[lower:upper])
    return (
        np.arange(lower, upper, dtype=np.int64),
        starts,
        ends,
        ends - starts + 1,
        np.full(starts.size, np.nan, dtype=np.float64),
    )


def _sequential_rows(
    second: _SourceRuns,
    onset: int,
    ending: int,
    steps: int,
    window: tuple[int, int | None],
    unknown_prefix: npt.NDArray[np.int64],
    *,
    budget: _Budget | None = None,
) -> tuple[int, _Rows | None]:
    """Return the bridges rejected as unknown and the accepted second runs after one first run."""
    low, high = window
    # a window bound at or past the record makes every later run eligible, and
    # clamping the key to the record length keeps a huge bound from overflowing
    lower = int(np.searchsorted(second.start, steps if low >= steps else ending + low, side="left"))
    if lower >= second.start.size:
        return 0, None
    upper = int(
        np.searchsorted(
            second.start,
            steps if high is None or high >= steps else ending + high,
            side="right",
        )
    )
    if lower >= upper:
        return 0, None
    # every candidate whose bridge reaches the first unknown step is rejected, so the accepted
    # candidates are a prefix of the window and the rejected count needs no per-pair test
    boundary = int(np.searchsorted(unknown_prefix, unknown_prefix[ending] + 1, side="left")) - 1
    accepted = min(upper, max(lower, int(np.searchsorted(second.start, boundary, side="left"))))
    rejected = upper - accepted
    if lower >= accepted:
        return rejected, None
    if budget is not None and accepted - lower > budget[0]:
        raise _row_limit(budget[1], budget[0], accepted - lower, mode="sequential")
    matched = np.arange(lower, accepted, dtype=np.int64)
    return rejected, (
        matched,
        np.full(matched.size, onset, dtype=np.int64),
        second.end[matched],
        np.zeros(matched.size, dtype=np.int64),
        np.subtract(second.start[matched], ending, dtype=np.float64),
    )


def _match(
    first: _SourceRuns,
    second: _SourceRuns,
    *,
    valid_prefix: npt.NDArray[np.int64],
    mode: str,
    window: tuple[int, int | None],
    max_pairs: int | None = None,
    pair_limit: int | None = None,
) -> _Matched:
    """Enumerate the matched pairs and count the sequential bridges rejected as unknown.

    Both sources are validated sorted and nonoverlapping, so one first run's candidates are
    a contiguous range of second runs. Concurrent candidates run from the first second-run
    end at or after the onset to the last second-run start at or before the first run's end;
    sequential candidates run from the first start at or after ``first.end + low`` to the
    last start at or before ``first.end + high``. Searching those bounds keeps the
    per-first-run work proportional to its candidates, not to every second run. The accepted
    candidates are a prefix of the window, so counting the rejected bridges needs one lookup
    rather than a test per candidate pair. ``max_pairs`` caps the rows held at once: each source run's
    candidate count is compared with the remaining budget before its arrays are built, so a
    quadratic fanout fails instead of being allocated.
    """
    steps = int(valid_prefix.size) - 1
    # unknown_prefix[i] counts the unknown steps before i; it does not decrease, so the first
    # candidate beyond every observed bridge is one binary search rather than a scan
    unknown_prefix = np.arange(valid_prefix.size) - valid_prefix
    columns: dict[str, list[npt.NDArray[Any]]] = {
        key: [] for key in ("first_run_id", "second_run_id", "start_index", "end_index", "overlap_duration", "lag")
    }
    unknown_bridges = 0
    matched_rows = 0
    for position in range(first.start.size):
        onset, ending = int(first.start[position]), int(first.end[position])
        budget: _Budget | None = (
            None if max_pairs is None else (max_pairs - matched_rows, max_pairs if pair_limit is None else pair_limit)
        )
        if mode == "concurrent":
            rows = _concurrent_rows(second, onset, ending, budget=budget)
        else:
            rejected, rows = _sequential_rows(second, onset, ending, steps, window, unknown_prefix, budget=budget)
            unknown_bridges += rejected
        if rows is None:
            continue
        matched, starts, ends, overlaps, lags = rows
        matched_rows += int(matched.size)
        columns["first_run_id"].append(np.full(matched.size, position, dtype=np.int64))
        columns["second_run_id"].append(matched)
        columns["start_index"].append(starts)
        columns["end_index"].append(ends)
        columns["overlap_duration"].append(overlaps)
        columns["lag"].append(lags)
    # pop each column as it is stacked so its parts are released before the result is built
    first_run_ids = _stack(columns.pop("first_run_id"), np.int64)
    second_run_ids = _stack(columns.pop("second_run_id"), np.int64)
    return _Matched(
        first_run_id=first_run_ids,
        second_run_id=second_run_ids,
        start_index=_stack(columns.pop("start_index"), np.int64),
        end_index=_stack(columns.pop("end_index"), np.int64),
        overlap_duration=_stack(columns.pop("overlap_duration"), np.int64),
        lag=_stack(columns.pop("lag"), np.float64),
        first_left_censored=_flags(first.left_censored, first_run_ids),
        first_right_censored=_flags(first.right_censored, first_run_ids),
        second_left_censored=_flags(second.left_censored, second_run_ids),
        second_right_censored=_flags(second.right_censored, second_run_ids),
        unknown_bridges=unknown_bridges,
    )


def _stack(parts: list[npt.NDArray[Any]], dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """Concatenate per-run row blocks into one read-only array with a stable empty dtype."""
    if not parts:
        return _immutable(np.empty(0, dtype=dtype))
    return _immutable(np.concatenate(parts).astype(dtype, copy=False))


def _flags(per_run: npt.NDArray[np.bool_], run_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    """Repeat each source run's boundary flag once per row."""
    if run_ids.size == 0:
        return _immutable(np.zeros(0, dtype=np.bool_))
    return _immutable(per_run[run_ids])


def _compound_events(
    first: RunSet,
    second: RunSet,
    *,
    time: npt.ArrayLike,
    valid: npt.ArrayLike,
    cadence: str,
    mode: str,
    lag_window: object,
    source_metadata: object,
    max_pairs: int | None = None,
    pair_limit: int | None = None,
) -> CompoundEventSet:
    """Validate the shared grid and both run sets, then match and freeze the pairs.

    ``max_pairs`` is the remaining row budget, and ``pair_limit`` is the total limit
    the error names; it defaults to ``max_pairs`` for a caller that passes one number.
    """
    _check_choice("cadence", cadence, _CADENCES)
    _check_choice("mode", mode, _MODES)
    window = _lag_window(mode, lag_window)

    labels = np.asarray(time)
    if labels.ndim != 1:
        raise DataShapeError(
            "time must be a one-dimensional cadence grid; match one grid cell at a time.",
            expected_shape="(time,)",
            actual_shape=labels.shape,
        )
    steps = int(labels.shape[0])
    if steps == 0:
        raise CoordinateValidationError(
            "time must hold at least one step.",
            coordinate_name="time",
            reason="empty",
        )
    # the explicit one-dimensional check above keeps _observed_index's shape branch unreachable,
    # so its inputs['shared'] wording is never shown; the remaining checks name the real argument
    index = _observed_index("shared", labels, cadence, steps, label="time")
    if np.any(np.diff(index) != 1):
        raise CoordinateValidationError(
            "time must be a complete cadence grid with one step per cadence unit; interior gaps and repeated steps "
            "are rejected because an absent date is an unknown observation, not a step to close.",
            coordinate_name="time",
            reason="incomplete_grid",
        )
    mask = _observation_mask(valid, steps)
    valid_prefix = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))

    if source_metadata is not None and not isinstance(source_metadata, Mapping):
        raise InvalidArgumentError(
            "source_metadata must be a mapping keyed by 'first' and 'second'.",
            argument_name="source_metadata",
            argument_value=type(source_metadata).__name__,
            valid_values="Mapping[str, Mapping[str, object]]",
        )
    sources: Mapping[str, Any] = source_metadata if isinstance(source_metadata, Mapping) else {}
    # a key is caller data, so only its count is named: an accidental credential must never reach a log sink
    unexpected = sum(1 for key in sources if key not in _SOURCE_NAMES)
    if unexpected:
        raise InvalidArgumentError(
            f"source_metadata keys must be a subset of {list(_SOURCE_NAMES)}; {unexpected} unexpected key(s) were "
            "supplied.",
            argument_name="source_metadata",
            argument_value=f"{unexpected} unexpected key(s)",
            valid_values=f"keys from {list(_SOURCE_NAMES)}",
        )

    # source choices are validated before the matching they would otherwise be rejected after
    source_entries = {name: _source_entry(name, sources.get(name)) for name in _SOURCE_NAMES}

    first_runs = _validated_runs("first", first, mask, valid_prefix, steps)
    second_runs = _validated_runs("second", second, mask, valid_prefix, steps)
    rows = _match(
        first_runs,
        second_runs,
        valid_prefix=valid_prefix,
        mode=mode,
        window=window,
        max_pairs=max_pairs,
        pair_limit=pair_limit,
    )

    unit = _UNITS[cadence]
    n_valid = int(valid_prefix[-1])
    metadata = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "mode": mode,
        "calendar": "proleptic_gregorian",
        "cadence": cadence,
        "matching_rule": _MATCHING_RULES[mode],
        "censoring_rule": _CENSORING_RULE,
        "missingness": _MISSINGNESS,
        "lag_window": [window[0], window[1]] if mode == "sequential" else None,
        "grid": {
            "step": cadence,
            "n_steps": steps,
            "n_valid": n_valid,
            "n_unknown": steps - n_valid,
            "time_start": str(labels[0].astype(unit)),
            "time_end": str(labels[-1].astype(unit)),
        },
        "n_runs": {"first": int(first_runs.start.size), "second": int(second_runs.start.size)},
        "n_pairs": int(rows.first_run_id.size),
        "n_unknown_bridges": rows.unknown_bridges,
        "sources": source_entries,
    }
    return CompoundEventSet(
        mode=cast(EventMode, mode),
        first_run_id=rows.first_run_id,
        second_run_id=rows.second_run_id,
        start_index=rows.start_index,
        end_index=rows.end_index,
        overlap_duration=rows.overlap_duration,
        lag=rows.lag,
        first_left_censored=rows.first_left_censored,
        first_right_censored=rows.first_right_censored,
        second_left_censored=rows.second_left_censored,
        second_right_censored=rows.second_right_censored,
        metadata=_freeze(metadata),
    )


def compound_events(
    first: RunSet,
    second: RunSet,
    *,
    time: npt.ArrayLike,
    valid: npt.ArrayLike,
    cadence: Cadence,
    mode: EventMode,
    lag_window: tuple[int, int] | None = None,
    source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> CompoundEventSet:
    """Match two hazard run sets on one shared, complete cadence grid.

    Both run sets come from the same time axis and the same observations, but from
    different hazard series (for example a drought run set and a heat run set), so a row
    states which source runs coincided or followed one another. Severity measures are
    never combined: the result holds indices, durations, lags, and censoring only.

    ``valid`` is the single observation-validity mask for that axis. A run that contains
    an unknown step is rejected, because a run asserts a contiguous observed spell; a run
    that merely touches an unknown step or a record edge is kept and flagged censored.

    In sequential mode every pair with ``second.start > first.end`` is matched when its
    lag falls inside the inclusive window **and** every step from ``first.end`` through
    ``second.start`` is observed. A pair separated by an unknown step is not reported, so
    a cascade is never inferred across a gap; the rejected-pair count is recorded as
    ``n_unknown_bridges`` in the result metadata.

    Args:
        first: Runs of the first hazard, from
            :func:`climate_indices.runs.identify_runs`. Its ``start_index``, ``end_index``,
            and ``duration`` fields are validated, and every other field must hold one
            entry per run; the metric values themselves are not inspected.
        second: Runs of the second hazard, validated the same way.
        time: The shared time axis: one-dimensional ``datetime64`` labels, strictly
            increasing, and a complete cadence grid with one step per cadence unit.
            Interior gaps are rejected rather than closed, because a missing date is an
            unknown observation.
        valid: Boolean array shaped like ``time``, ``True`` where the hazard was observed.
            Unknown observations are ``False``, not 0 or NaN.
        cadence: ``"daily"`` or ``"monthly"``, the step of ``time`` and of the reported
            indices and lags.
        mode: ``"concurrent"`` for intersecting pairs or ``"sequential"`` for
            follow-on pairs.
        lag_window: Inclusive ``(low, high)`` lag window for sequential matching, where
            the lag is ``second.start - first.end`` and adjacent runs have lag one.
            ``None`` matches every lag at or above one. It must be ``None`` for
            concurrent matching.
        source_metadata: Optional mapping keyed by ``"first"`` and ``"second"``. Each
            entry is a JSON-compatible mapping that may declare ``threshold``,
            ``direction``, and ``min_duration`` — the choices a
            :class:`~climate_indices.runs.RunSet` does not retain — and is copied into the
            manifest; a choice left out is recorded as ``"unknown"`` and is never
            reconstructed from the run metrics.

    Returns:
        A :class:`CompoundEventSet` with one row per matched pair, ordered by first
        source-run ID and then by second source-run ID, holding new read-only arrays.

    Raises:
        InvalidArgumentError: For an unsupported ``cadence`` or ``mode``, a
            ``lag_window`` that is out of range or given for concurrent matching, an
            inconsistent ``RunSet`` (non-positive or inconsistent durations, out-of-bounds
            or overlapping runs), a run crossing an unknown observation, or malformed
            ``source_metadata``.
        InputTypeError: If a source is not a ``RunSet``, a run index is not an integer
            array, ``valid`` is not a plain Boolean array, or the dates are not
            ``datetime64``.
        DataShapeError: If ``time`` or ``valid`` has the wrong shape, or a ``RunSet``'s
            index fields disagree in length.
        CoordinateValidationError: For an empty axis, NaT, repeated, or unordered dates,
            or an incomplete cadence grid.

    Example:
        >>> import numpy as np
        >>> from climate_indices import compound
        >>> from climate_indices.runs import identify_runs
        >>> time = np.arange("2000-01", "2000-07", dtype="datetime64[M]")
        >>> drought = identify_runs(np.array([0.2, -1.2, -1.4, 0.1, -1.3, -1.1]), threshold=-1.0)
        >>> heat = identify_runs(np.array([2.0, 2.5, 0.1, 3.0, 3.2, 3.4]), threshold=2.0, direction="above")
        >>> valid = np.ones(6, dtype=bool)
        >>> concurrent = compound.compound_events(
        ...     drought, heat, time=time, valid=valid, cadence="monthly", mode="concurrent"
        ... )
        >>> concurrent.start_index.tolist(), concurrent.end_index.tolist()
        ([1, 4], [1, 5])
        >>> concurrent.overlap_duration.tolist()
        [1, 2]
        >>> sequential = compound.compound_events(
        ...     drought, heat, time=time, valid=valid, cadence="monthly", mode="sequential"
        ... )
        >>> sequential.first_run_id.tolist(), sequential.second_run_id.tolist(), sequential.lag.tolist()
        ([0], [1], [1.0])
        >>> sequential.start_index.tolist(), sequential.end_index.tolist()
        ([1], [5])
    """
    # source names are caller-supplied identifiers, so only their count reaches the log
    log = _logger.bind(index_type="compound_events", cadence=cadence, mode=mode)
    log.info("calculation_started")
    t0 = perf_counter()
    try:
        result = _compound_events(
            first,
            second,
            time=time,
            valid=valid,
            cadence=cadence,
            mode=mode,
            lag_window=lag_window,
            source_metadata=source_metadata,
        )
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise
    log.info(
        "calculation_completed",
        duration_ms=round((perf_counter() - t0) * 1000.0, 2),
        n_pairs=len(result),
    )
    return result


def compound_events_xarray(
    hazards: xr.Dataset,
    *,
    first: str,
    second: str,
    mode: EventMode,
    lag_window: tuple[int, int] | None = None,
    min_duration: int = 1,
) -> xr.Dataset:
    """Build a beta, eager numeric event catalog from prepared hazard subsets.

    Select and explicitly compute a bounded subset of the Dataset returned by
    :func:`harmonize` before calling this function. Dask-backed variables or
    coordinates are rejected without evaluation. Memory grows with the input subset and all
    matched pairs; sequential matching without a bounded lag can produce a
    quadratic number of pairs. This function never calls ``compute()`` or ``load()``.

    Identify both sources' runs on their shared validity mask with
    :func:`climate_indices.runs.identify_runs`, using threshold 0.5, direction
    ``"above"``, no gap pooling, and the declared ``min_duration``. The sources
    are already thresholded, including any daily-to-monthly reduction. An unknown
    observation in either selected source splits both runs, and the censor flags
    use that same shared support. A hazard outside the selected pair does not
    affect the pair's support. Matching uses :func:`compound_events` unchanged.

    A catalog holds at most 1,000,000 event rows. A match that would exceed the
    cap raises instead of building the rows that would overflow it.

    Args:
        hazards: In-memory prepared ``xarray.Dataset`` with Boolean ``exceedance``
            and ``valid`` arrays and a complete Gregorian cadence grid.
        first: Name of the first hazard in the ``hazard`` coordinate.
        second: Name of a different hazard in the ``hazard`` coordinate.
        mode: ``"concurrent"`` for inclusive intersections or ``"sequential"``
            for first-onset through second-end intervals.
        lag_window: Inclusive sequential lag bounds in cadence steps. Adjacency
            has lag one; ``None`` allows every positive lag. Concurrent mode
            requires ``None``.
        min_duration: Positive integer minimum source-run duration in cadence
            steps, applied separately to both sources before matching.

    Returns:
        An eager Dataset with one ``event`` row per matched pair. ``cell_index``
        points into a separate ``cell`` table whose spatial label coordinates
        retain their names. A single series has one cell. Rows follow spatial
        dimension order (C order), then first and second source-run order.
        Run IDs are local to each cell's retained runs after duration filtering.
        The Dataset includes indices, inclusive ``start_time``/``end_time``,
        overlap duration, lag, integer censor flags (0 or 1), and per-cell support,
        run, pair, and rejected-bridge counts. Empty catalogs retain their dtypes
        and cell table. The versioned JSON manifest retains the prepared source
        declarations and run choices. No variable contains a ``RunSet`` object.

    Raises:
        InputTypeError: For non-Dataset input, non-Boolean hazard arrays, or
            unsupported cell labels (including MultiIndex objects).
        InvalidArgumentError: For Dask-backed input, missing variables, invalid
            source names, duration or matching options, a malformed or non-finite
            manifest, duplicate hazard labels, or a match that exceeds the
            1,000,000-row catalog cap.
        CoordinateValidationError: For invalid dates or an incomplete cadence
            grid. Unknown observations must retain positions with validity false.
        DimensionMismatchError: For inconsistent dimensions or spatial names
            that collide with catalog fields; rename those dimensions first.

    Note:
        Emits :class:`~climate_indices.exceptions.BetaFeatureWarning` (ADR-0012).
        NetCDF and Zarr serialization preserves numeric rows and JSON metadata.
        The catalog claims no CF discrete-sampling-geometry feature type.
        Pair counts are not independent observations or a significance test.
        Interface-equivalence tests do not provide external scientific validation.
    """
    # deferred: _catalog imports this module for CompoundEventSet and EventMode
    from climate_indices.compound import _catalog

    return _catalog.compound_events_xarray(
        hazards, first=first, second=second, mode=mode, lag_window=lag_window, min_duration=min_duration
    )

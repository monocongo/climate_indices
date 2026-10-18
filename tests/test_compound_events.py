"""Tests for ``compound.compound_events()`` and ``CompoundEventSet`` (CMP-06, ADR-0022).

Event rows are indices, counts, lags, and Boolean flags, so every comparison is exact: no
tolerance applies. The all-pairs semantics are checked against a brute-force interval
enumeration that shares no code with the implementation, and against hand-counted pairs
that pin the inclusive-end conventions.
"""

from __future__ import annotations

import copy
import dataclasses
import doctest
import json
import logging
import pickle
from typing import Any, cast
from unittest import mock

import numpy as np
import pytest

import climate_indices
from climate_indices import compound
from climate_indices.compound import _events as events_module
from climate_indices.exceptions import (
    CoordinateValidationError,
    DataShapeError,
    InputTypeError,
    InvalidArgumentError,
)
from climate_indices.runs import RunSet, identify_runs


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def months(count: int) -> np.ndarray:
    """Return ``count`` consecutive monthly labels starting at 2000-01."""
    return np.arange("2000-01", "2050-01", dtype="datetime64[M]")[:count]


def days(count: int) -> np.ndarray:
    """Return ``count`` consecutive daily labels starting at 2000-01-01."""
    return np.arange("2000-01-01", "2020-01-01", dtype="datetime64[D]")[:count]


def manual(starts: list[int], ends: list[int], **fields: Any) -> RunSet:
    """Build a RunSet from inclusive run bounds, with consistent metrics unless overridden."""
    start_index = np.asarray(starts, dtype=np.int64)
    end_index = np.asarray(ends, dtype=np.int64)
    duration = end_index - start_index + 1
    runs = RunSet(
        start_index=start_index,
        end_index=end_index,
        duration=duration,
        magnitude=duration.astype(np.float64),
        intensity=np.ones_like(duration, dtype=np.float64),
        peak_value=np.full(duration.shape, -1.5, dtype=np.float64),
        peak_index=start_index.copy(),
        interarrival=np.full(duration.shape, np.nan, dtype=np.float64),
    )
    return dataclasses.replace(runs, **fields) if fields else runs


def match(
    first: RunSet,
    second: RunSet,
    *,
    steps: int = 10,
    valid: np.ndarray | None = None,
    cadence: str = "monthly",
    mode: str = "concurrent",
    lag_window: tuple[int, int] | None = None,
    source_metadata: Any = None,
) -> compound.CompoundEventSet:
    """Match two run sets on a fresh grid, with every step observed unless the caller says otherwise."""
    axis = months(steps) if cadence == "monthly" else days(steps)
    return compound.compound_events(
        first,
        second,
        time=axis,
        valid=np.ones(steps, dtype=bool) if valid is None else valid,
        cadence=cadence,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        lag_window=lag_window,
        source_metadata=source_metadata,
    )


def brute_force(
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
    valid: np.ndarray,
    mode: str,
    window: tuple[int, int | None],
) -> list[tuple[int, int, int, int]]:
    """Enumerate matched pairs by definition, with an explicit observed-slice scan."""
    low, high = window
    rows: list[tuple[int, int, int, int]] = []
    for i, (first_start, first_end) in enumerate(first):
        for j, (second_start, second_end) in enumerate(second):
            if mode == "concurrent":
                if first_start <= second_end and second_start <= first_end:
                    rows.append((i, j, max(first_start, second_start), min(first_end, second_end)))
            else:
                lag = second_start - first_end
                in_window = lag >= low and (high is None or lag <= high)
                if in_window and bool(np.all(valid[first_end : second_start + 1])):
                    rows.append((i, j, first_start, second_end))
    return rows


def censored(boundary: int, offset: int, valid: np.ndarray) -> bool:
    """Return whether a run boundary touches the record edge or an unknown neighbouring step."""
    neighbour = boundary + offset
    return bool(neighbour < 0 or neighbour >= valid.size or not valid[neighbour])


# --- the public surface ------------------------------------------------------------------------


def test_namespace_exports_the_event_api_but_not_from_the_root() -> None:
    """Import explicitly: event matching joins the compound module, not the package root."""
    assert {"CompoundEventSet", "compound_events"} <= set(compound.__all__)
    assert compound.CompoundEventSet is events_module.CompoundEventSet
    assert not hasattr(climate_indices, "compound_events")


def test_docstring_example_runs() -> None:
    """Keep the published example executable as a fast regression on the documented numbers."""
    outcome = doctest.testmod(events_module)
    assert outcome.failed == 0
    assert outcome.attempted > 0


def test_concurrent_keeps_every_intersecting_pair_including_touching_endpoints() -> None:
    """A pair sharing a single step is an intersection, and every pair is kept."""
    first = manual([0, 5], [3, 7])
    second = manual([3, 6], [4, 8])
    result = match(first, second)
    assert result.mode == "concurrent"
    assert result.first_run_id.tolist() == [0, 1]
    assert result.second_run_id.tolist() == [0, 1]
    assert result.start_index.tolist() == [3, 6]
    assert result.end_index.tolist() == [3, 7]
    assert result.overlap_duration.tolist() == [1, 2]
    assert bool(np.all(np.isnan(result.lag)))


def test_repeated_matches_are_not_merged_or_discarded() -> None:
    """One interval that intersects three others yields three rows, not one chosen match."""
    first = manual([0], [10])
    second = manual([0, 3, 9], [2, 4, 12])
    result = match(first, second, steps=13)
    assert result.second_run_id.tolist() == [0, 1, 2]
    assert result.start_index.tolist() == [0, 3, 9]
    assert result.end_index.tolist() == [2, 4, 10]
    assert result.overlap_duration.tolist() == [3, 2, 2]
    # runs and pairs are counted separately, so a one-to-three match is visible as such
    assert result.metadata["n_runs"] == {"first": 1, "second": 3}
    assert result.metadata["n_pairs"] == 3
    assert result.metadata["cadence"] == "monthly"
    assert result.metadata["mode"] == "concurrent"


def test_several_first_runs_may_share_one_second_run() -> None:
    """Matching is all pairs, so two first runs intersecting one second run stay two rows."""
    first = manual([0, 6], [4, 10])
    second = manual([2], [8])
    result = match(first, second, steps=11)
    assert result.first_run_id.tolist() == [0, 1]
    assert result.second_run_id.tolist() == [0, 0]
    assert result.start_index.tolist() == [2, 6]
    assert result.end_index.tolist() == [4, 8]


def test_concurrent_bounds_are_compared_before_any_indexing() -> None:
    """A first run entirely before the second run is not a match, and vice versa."""
    first = manual([0, 8], [1, 9])
    second = manual([3], [6])
    result = match(first, second, steps=10)
    assert len(result) == 0


def test_sequential_reports_the_enclosing_interval_and_the_separating_lag() -> None:
    """Report first onset through second end and the lag, never an invented joint duration."""
    first = manual([0, 5], [2, 5])
    second = manual([3, 7], [6, 12])
    result = match(first, second, steps=13, mode="sequential")
    assert result.first_run_id.tolist() == [0, 0, 1]
    assert result.second_run_id.tolist() == [0, 1, 1]
    assert result.lag.tolist() == [1.0, 5.0, 2.0]
    assert result.start_index.tolist() == [0, 0, 5]
    assert result.end_index.tolist() == [6, 12, 12]
    assert result.overlap_duration.tolist() == [0, 0, 0]


def test_adjacent_runs_have_lag_one_and_are_sequential() -> None:
    """Adjacency is the smallest sequential lag, and the runs are disjoint."""
    result = match(manual([0], [2]), manual([3], [4]), steps=6, mode="sequential")
    assert result.lag.tolist() == [1.0]
    assert result.start_index.tolist() == [0]
    assert result.end_index.tolist() == [4]
    assert result.overlap_duration.tolist() == [0]


def test_sequential_requires_a_strictly_later_second_start() -> None:
    """A second run overlapping or starting on the first run's end is not a follow-on."""
    first = manual([2], [4])
    second = manual([0, 4, 7], [2, 6, 8])
    result = match(first, second, steps=10, mode="sequential")
    assert result.second_run_id.tolist() == [2]
    assert result.lag.tolist() == [3.0]


@pytest.mark.parametrize(
    ("window", "expected_lags"),
    [
        ((2, 4), [2.0, 4.0]),  # both bounds are inclusive
        ((4, 4), [4.0]),
        ((2, 2), [2.0]),
        ((6, 10), [6.0]),
        (None, [2.0, 4.0, 6.0]),  # None matches every lag at or above one
    ],
)
def test_lag_window_bounds_are_inclusive(window: tuple[int, int] | None, expected_lags: list[float]) -> None:
    """Keep every pair inside the requested window and nothing outside it."""
    first = manual([0], [2])
    second = manual([4, 6, 8], [5, 7, 9])
    result = match(first, second, steps=12, mode="sequential", lag_window=window)
    assert result.lag.tolist() == expected_lags


def test_sequential_overlap_duration_is_zero_while_the_interval_spans_the_runs() -> None:
    """The enclosing interval covers both runs and the gap; it is not a shared duration."""
    result = match(manual([0], [1]), manual([4], [5]), steps=8, mode="sequential")
    assert result.start_index.tolist() == [0]
    assert result.end_index.tolist() == [5]
    assert result.overlap_duration.tolist() == [0]
    assert result.lag.tolist() == [3.0]
    # the reported interval is six steps long, which is neither the overlap nor the lag
    assert int(result.end_index[0] - result.start_index[0] + 1) == 6


# --- unknown observations, bridges, and censoring -----------------------------------------------


def test_sequential_pair_is_not_matched_across_an_unknown_bridge() -> None:
    """Never infer a cascade across a gap, and record how many pairs were rejected."""
    valid = np.ones(8, dtype=bool)
    valid[3] = False
    result = match(manual([0], [1]), manual([4], [5]), steps=8, valid=valid, mode="sequential")
    assert len(result) == 0
    assert result.metadata["n_unknown_bridges"] == 1


def test_unknown_bridge_count_excludes_the_matched_pairs() -> None:
    """Count only the rejected candidates: a matched pair in the same window is not a bridge."""
    valid = np.ones(6, dtype=bool)
    valid[3] = False
    result = match(manual([0], [1]), manual([2, 4], [2, 5]), steps=6, valid=valid, mode="sequential")
    assert result.second_run_id.tolist() == [0]  # lag 1, bridge [1, 2] observed
    assert result.lag.tolist() == [1.0]
    assert result.metadata["n_pairs"] == 1
    assert result.metadata["n_unknown_bridges"] == 1  # only the lag-3 pair, bridge [1, 4]


def test_unknown_bridge_count_covers_only_pairs_inside_the_window() -> None:
    """Count the window's rejected bridges, not every ineligible pair."""
    valid = np.ones(10, dtype=bool)
    valid[3] = False
    first = manual([0], [1])
    second = manual([4, 6, 9], [5, 8, 9])
    result = match(first, second, steps=10, valid=valid, mode="sequential", lag_window=(1, 3))
    assert len(result) == 0
    assert result.metadata["n_unknown_bridges"] == 1


def test_unknown_steps_outside_the_bridge_do_not_block_a_pair() -> None:
    """Only the interval from the first end through the second start must be observed."""
    valid = np.ones(9, dtype=bool)
    valid[0] = False
    valid[6] = False
    result = match(manual([2], [3]), manual([4], [5]), steps=9, valid=valid, mode="sequential")
    assert result.lag.tolist() == [1.0]
    assert result.first_run_id.tolist() == [0]
    assert result.second_left_censored.tolist() == [False]
    assert result.first_left_censored.tolist() == [False]
    assert result.second_right_censored.tolist() == [True]
    assert result.first_right_censored.tolist() == [False]
    grid = result.metadata["grid"]
    assert (grid["n_steps"], grid["n_valid"], grid["n_unknown"]) == (9, 7, 2)
    assert (grid["time_start"], grid["time_end"]) == ("2000-01", "2000-09")
    assert grid["step"] == "monthly"


def test_runs_at_the_record_edges_are_censored_on_that_side() -> None:
    """A run touching the record boundary may be a truncated physical event."""
    result = match(manual([0], [9]), manual([0, 9], [0, 9]), steps=10)
    assert result.first_left_censored.tolist() == [True, True]
    assert result.first_right_censored.tolist() == [True, True]
    assert result.second_left_censored.tolist() == [True, False]
    assert result.second_right_censored.tolist() == [False, True]


def test_runs_next_to_an_unknown_step_are_censored_without_being_rejected() -> None:
    """An unknown neighbour hides the true boundary but does not invalidate the run."""
    valid = np.ones(10, dtype=bool)
    valid[1] = False
    valid[8] = False
    first = manual([2, 6], [3, 7])
    second = manual([3], [7])
    result = match(first, second, steps=10, valid=valid)
    assert result.first_left_censored.tolist() == [True, False]
    assert result.first_right_censored.tolist() == [False, True]
    assert result.second_left_censored.tolist() == [False, False]
    assert result.second_right_censored.tolist() == [True, True]


# --- empty catalogs ---------------------------------------------------------------------------


DTYPES: dict[str, type[np.generic]] = {
    "first_run_id": np.int64,
    "second_run_id": np.int64,
    "start_index": np.int64,
    "end_index": np.int64,
    "overlap_duration": np.int64,
    "lag": np.float64,
    "first_left_censored": np.bool_,
    "first_right_censored": np.bool_,
    "second_left_censored": np.bool_,
    "second_right_censored": np.bool_,
}


@pytest.mark.parametrize(
    ("mode", "lag_window"),
    [("concurrent", None), ("sequential", (4, 9)), ("sequential", (3, 3))],
)
def test_empty_catalog_has_stable_dtypes_and_zero_counts(mode: str, lag_window: tuple[int, int] | None) -> None:
    """Return no rows with fixed dtypes, so a later adapter can allocate an empty catalog."""
    result = match(manual([0], [1]), manual([3], [4]), steps=6, mode=mode, lag_window=lag_window)
    assert len(result) == 0
    for field, dtype in DTYPES.items():
        values = getattr(result, field)
        assert values.shape == (0,)
        assert values.dtype == dtype
    assert result.metadata["n_pairs"] == 0


def test_empty_run_sets_produce_an_empty_catalog() -> None:
    """Two empty run sets are valid input and yield no matches."""
    result = match(manual([], []), manual([], []), steps=6)
    assert len(result) == 0
    assert result.metadata["n_runs"] == {"first": 0, "second": 0}


# --- RunSet validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "error", "match_text"),
    [
        (lambda: manual([0], [1], duration=np.array([3])), InvalidArgumentError, "positive durations"),
        (lambda: manual([0], [1], duration=np.array([1, 1])), DataShapeError, "one entry per run"),
        (lambda: manual([-1], [1]), InvalidArgumentError, "inside the 10-step time axis"),
        (lambda: manual([9], [10]), InvalidArgumentError, "inside the 10-step time axis"),
        (lambda: manual([0, 2], [3, 4]), InvalidArgumentError, "sorted and nonoverlapping"),
        (lambda: manual([3, 0], [4, 1]), InvalidArgumentError, "sorted and nonoverlapping"),
        (lambda: manual([0], [1], start_index=np.array([0.5])), InputTypeError, "must hold integers"),
        (lambda: manual([0], [1], end_index=np.array([[1]])), DataShapeError, "one-dimensional"),
        (
            lambda: manual([0], [1], magnitude=np.array([1.0, 2.0])),
            DataShapeError,
            r"magnitude \(2\) disagree",
        ),
        (
            lambda: manual([0], [1], start_index=np.ma.array([0], mask=[True])),
            InputTypeError,
            "masked run index is ambiguous",
        ),
        (
            lambda: manual(
                [0],
                [1],
                start_index=np.array([2**63 - 1], np.uint64),
                end_index=np.array([2**63], np.uint64),
                duration=np.array([2], np.uint64),
            ),
            InputTypeError,
            "above the int64 range",
        ),
    ],
)
def test_manually_constructed_runs_are_validated(build: Any, error: type[Exception], match_text: str) -> None:
    """A dataclass is not proof of validity: check the run structure before matching."""
    with pytest.raises(error, match=match_text):
        match(build(), manual([0], [1]))


def test_the_second_source_is_validated_too() -> None:
    """Validate both sources: an invalid second run set must be rejected as well."""
    with pytest.raises(InvalidArgumentError, match="sorted and nonoverlapping"):
        match(manual([0], [1]), manual([0, 2], [3, 4]))


def test_a_run_crossing_an_unknown_observation_is_rejected() -> None:
    """A run asserts a contiguous observed spell, so an unknown step inside it is invalid."""
    valid = np.ones(6, dtype=bool)
    valid[2] = False
    with pytest.raises(InvalidArgumentError, match="crosses an unknown observation"):
        match(manual([1], [3]), manual([0], [0]), steps=6, valid=valid)


def test_a_non_runset_argument_is_rejected() -> None:
    """Reject an invalid representation with its type, not with an attribute error."""
    with pytest.raises(InputTypeError, match="must be a RunSet"):
        compound.compound_events(
            {"start_index": [0]},  # type: ignore[arg-type]
            manual([0], [0]),
            time=months(4),
            valid=np.ones(4, dtype=bool),
            cadence="monthly",
            mode="concurrent",
        )


def test_identified_runs_are_accepted_as_given() -> None:
    """A RunSet from ``identify_runs`` needs no adjustment, and source-run IDs stay positional."""
    values = np.array([0.2, -1.2, -1.4, 0.1, -1.3, -1.1])
    other = np.array([2.0, 2.5, 0.1, 3.0, 3.2, 3.4])
    drought = identify_runs(values, threshold=-1.0)
    heat = identify_runs(other, threshold=2.0, direction="above")
    result = match(drought, heat, steps=6)
    assert result.first_run_id.tolist() == [0, 1]
    assert result.second_run_id.tolist() == [0, 1]
    assert result.start_index.tolist() == [1, 4]
    assert result.end_index.tolist() == [1, 5]
    assert result.overlap_duration.tolist() == [1, 2]
    # the source-run convention is the RunSet's own: each id indexes into the input directly
    assert drought.start_index[result.first_run_id].tolist() == [1, 4]
    assert heat.start_index[result.second_run_id].tolist() == [1, 3]
    assert drought.end_index[result.first_run_id].tolist() == [2, 5]


# --- time axis and validity mask ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "error", "reason"),
    [
        (np.delete(months(5), 2), CoordinateValidationError, "complete cadence grid"),  # an interior gap
        (months(5)[::-1], CoordinateValidationError, "increasing"),
        (np.array([np.datetime64("NaT")] * 5, dtype="datetime64[M]"), CoordinateValidationError, "NaT"),
        (days(5), CoordinateValidationError, "calendar month"),  # daily labels on a monthly step
        (np.arange(5), InputTypeError, "datetime64"),
        (np.array(["2000-01"] * 5), InputTypeError, "datetime64"),
    ],
)
def test_time_axis_must_be_a_complete_cadence_grid(axis: np.ndarray, error: type[Exception], reason: str) -> None:
    """Reject gaps, repeats, NaT, non-datetime labels, and a cadence that does not match the labels."""
    with pytest.raises(error, match=reason):
        compound.compound_events(
            manual([], []),
            manual([], []),
            time=axis,
            valid=np.ones(axis.shape, dtype=bool),
            cadence="monthly",
            mode="concurrent",
        )


def test_empty_and_multidimensional_time_axes_are_rejected() -> None:
    """Reject an axis without steps and an axis that is not one-dimensional."""
    with pytest.raises(CoordinateValidationError, match="at least one step"):
        match(manual([], []), manual([], []), steps=0)
    with pytest.raises(DataShapeError, match="one-dimensional cadence grid"):
        compound.compound_events(
            manual([], []),
            manual([], []),
            time=np.zeros((2, 2), dtype="datetime64[M]"),
            valid=np.ones((2, 2), dtype=bool),
            cadence="monthly",
            mode="concurrent",
        )


def test_daily_and_mixed_unit_labels_are_accepted() -> None:
    """Accept the declared cadence in any datetime64 unit, including a coarser one."""
    daily = match(manual([0], [1]), manual([1], [2]), steps=6, cadence="daily", mode="concurrent")
    assert daily.start_index.tolist() == [1]
    coarse = compound.compound_events(
        manual([0], [1]),
        manual([2], [3]),
        time=months(6).astype("datetime64[s]"),
        valid=np.ones(6, dtype=bool),
        cadence="monthly",
        mode="sequential",
    )
    assert coarse.lag.tolist() == [1.0]
    floored = compound.compound_events(
        manual([0], [1]),
        manual([2], [3]),
        time=months(6).astype("datetime64[D]"),
        valid=np.ones(6, dtype=bool),
        cadence="monthly",
        mode="sequential",
    )
    assert floored.lag.tolist() == [1.0]


@pytest.mark.parametrize(
    ("valid", "error", "match_text"),
    [
        (np.ones(5, dtype=np.int8), InputTypeError, "Boolean dtype"),
        (np.ones(5, dtype=np.float64), InputTypeError, "Boolean dtype"),
        (np.ones(6, dtype=bool), DataShapeError, "one Boolean per step"),
        (np.ones((5, 1), dtype=bool), DataShapeError, "one Boolean per step"),
        (np.ma.array(np.ones(5, dtype=bool), mask=[False, True, False, False, False]), InputTypeError, "plain Boolean"),
    ],
)
def test_valid_mask_must_be_plain_boolean_and_shaped(
    valid: np.ndarray, error: type[Exception], match_text: str
) -> None:
    """Unknown observations are False, so the mask is Boolean, complete, and unmasked."""
    with pytest.raises(error, match=match_text):
        match(manual([], []), manual([], []), steps=5, valid=valid)


# --- options ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cadence", ["weekly", "decadal", "DAILY", ""])
def test_unsupported_cadence_is_rejected(cadence: str) -> None:
    """Accept only the two declared cadences."""
    with pytest.raises(InvalidArgumentError, match="cadence"):
        match(manual([], []), manual([], []), steps=5, cadence=cadence)


@pytest.mark.parametrize("mode", ["intersecting", "CONCURRENT", ""])
def test_unsupported_mode_is_rejected(mode: str) -> None:
    """Accept only concurrent and sequential matching."""
    with pytest.raises(InvalidArgumentError, match="mode"):
        match(manual([], []), manual([], []), steps=5, mode=mode)


def test_lag_window_is_rejected_for_concurrent_matching() -> None:
    """A lag has no meaning for intersecting pairs, so do not accept a window for them."""
    with pytest.raises(InvalidArgumentError, match="lag_window applies to sequential matching only"):
        match(manual([], []), manual([], []), steps=5, lag_window=(1, 2))


@pytest.mark.parametrize(
    "lag_window",
    [(1,), (1, 2, 3), [1, 2], (1.0, 2), (True, 3), ("1", 2), 2, (0, 2), (3, 1), (-1, 2)],
)
def test_malformed_lag_window_is_rejected(lag_window: Any) -> None:
    """Require an integer tuple with 1 <= low <= high."""
    with pytest.raises(InvalidArgumentError, match="lag_window"):
        match(manual([], []), manual([], []), steps=5, mode="sequential", lag_window=lag_window)


def test_numpy_integer_lag_bounds_are_accepted() -> None:
    """Accept numpy integers as bounds, like the run and threshold options do."""
    result = match(
        manual([0], [1]),
        manual([2], [3]),
        steps=5,
        mode="sequential",
        lag_window=cast(Any, (np.int64(1), np.int32(2))),
    )
    assert result.lag.tolist() == [1.0]
    assert result.metadata["lag_window"] == (1, 2)


def test_explicit_window_is_recorded_in_the_metadata() -> None:
    """Record the effective window and the matching rule that produced the rows."""
    unbounded = match(manual([], []), manual([], []), steps=5, mode="sequential")
    assert unbounded.metadata["lag_window"] == (1, None)
    assert "second.start > first.end" in unbounded.metadata["matching_rule"]
    windowed = match(manual([], []), manual([], []), steps=5, mode="sequential", lag_window=(2, 4))
    assert windowed.metadata["lag_window"] == (2, 4)
    concurrent = match(manual([], []), manual([], []), steps=5)
    assert concurrent.metadata["lag_window"] is None


# --- source metadata --------------------------------------------------------------------------


def test_unavailable_source_choices_are_recorded_as_unknown() -> None:
    """Record the choices a RunSet does not retain as unknown instead of inferring them."""
    result = match(manual([0], [1]), manual([0], [1]), steps=5)
    assert result.metadata["sources"] == {
        "first": {"threshold": "unknown", "direction": "unknown", "min_duration": "unknown"},
        "second": {"threshold": "unknown", "direction": "unknown", "min_duration": "unknown"},
    }


def test_declared_source_choices_and_extra_keys_are_copied_verbatim() -> None:
    """Keep the caller's declarations and their unrelated keys in the manifest."""
    metadata = {
        "first": {"threshold": -1.0, "direction": "below", "min_duration": 3, "scale": 6, "id": "spi6"},
        "second": {"threshold": 35, "physical_threshold": 35, "tags": ["heat"]},
    }
    result = match(manual([0], [1]), manual([0], [1]), steps=5, source_metadata=metadata)
    assert result.metadata["sources"]["first"] == {
        "threshold": -1.0,
        "direction": "below",
        "min_duration": 3,
        "scale": 6,
        "id": "spi6",
    }
    assert result.metadata["sources"]["second"] == {
        "threshold": 35,
        "direction": "unknown",
        "min_duration": "unknown",
        "physical_threshold": 35,
        "tags": ("heat",),
    }
    metadata["first"]["tags"] = ["late edit"]  # a later caller edit must not reach the manifest
    assert "tags" not in result.metadata["sources"]["first"]


@pytest.mark.parametrize(
    ("source_metadata", "match_text"),
    [
        ({"third": {}}, "subset"),
        (["first"], "must be a mapping keyed by 'first' and 'second'"),
        ({"first": [1, 2]}, "JSON-compatible mapping"),
        ({"first": {"direction": "sideways"}}, "direction"),
        ({"first": {"min_duration": 0}}, "min_duration"),
        ({"first": {"min_duration": True}}, "min_duration"),
        ({"first": {"threshold": float("inf")}}, "JSON-compatible mapping"),
        ({"first": {"threshold": "0.1"}}, "finite"),
        ({"first": {"tags": {1, 2}}}, "JSON-compatible mapping"),
    ],
)
def test_malformed_source_metadata_is_rejected(source_metadata: Any, match_text: str) -> None:
    """Reject a declaration that cannot be recorded faithfully."""
    with pytest.raises(InvalidArgumentError, match=match_text):
        match(manual([], []), manual([], []), steps=5, source_metadata=source_metadata)


# --- immutability, inputs, and serialization ---------------------------------------------------


def test_result_arrays_are_read_only_and_new() -> None:
    """Freeze the arrays themselves; a frozen dataclass alone would not."""
    first = manual([0], [1])
    second = manual([0], [1])
    result = match(first, second)
    for field in dataclasses.fields(result):
        if field.name in ("mode", "metadata"):
            continue
        array = getattr(result, field.name)
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[0] = array[0]
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    assert not np.shares_memory(result.start_index, first.start_index)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.mode = "sequential"  # type: ignore[misc]


def test_inputs_are_not_mutated_and_unrelated_run_fields_survive() -> None:
    """Match without touching either RunSet, including the fields matching never inspects."""
    first = identify_runs(np.array([0.2, -1.2, -1.4, 0.1, -1.3, -1.1]), threshold=-1.0)
    second = identify_runs(np.array([2.0, 2.5, 0.1, 3.0, 3.2, 3.4]), threshold=2.0, direction="above")
    before = copy.deepcopy((first, second))
    result = match(first, second, steps=6, mode="sequential")
    assert len(result) == 1
    assert first == before[0]  # RunSet.__eq__ compares every array, NaN entries included
    assert second == before[1]


def test_metadata_is_immutable_and_serializable() -> None:
    """Expose read-only mappings and tuples, and keep the manifest JSON-compatible."""
    result = match(manual([0], [1]), manual([0], [1]), steps=5)
    with pytest.raises(TypeError):
        result.metadata["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result.metadata["grid"]["step"] = "daily"
    assert set(json.loads(json.dumps(result.metadata, allow_nan=False))) == {
        "schema_version",
        "mode",
        "cadence",
        "matching_rule",
        "lag_window",
        "grid",
        "n_runs",
        "n_pairs",
        "n_unknown_bridges",
        "sources",
    }
    assert result.metadata["schema_version"] == 1
    assert pickle.loads(pickle.dumps(result.metadata)) == result.metadata
    assert copy.deepcopy(result.metadata) == result.metadata


def test_every_result_field_is_a_numeric_array_or_the_manifest() -> None:
    """Pin the field set and dtype: no added severity metric, RunSet object, or CF claim."""
    result = match(manual([0], [1]), manual([0], [1]), steps=5)
    assert {field.name for field in dataclasses.fields(result)} == {*DTYPES, "mode", "metadata"}
    for field, dtype in DTYPES.items():
        values = getattr(result, field)
        assert isinstance(values, np.ndarray)
        assert values.dtype == dtype
    assert result.mode in ("concurrent", "sequential")


# --- lifecycle --------------------------------------------------------------------------------


def test_lifecycle_events_are_logged() -> None:
    """Emit started and completed events with the pair count."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger
    with mock.patch.object(events_module, "_logger", mock_logger):
        match(manual([0], [1]), manual([0], [1]), steps=5)
    started, completed = mock_logger.info.call_args_list
    assert started.args[0] == "calculation_started"
    assert completed.args[0] == "calculation_completed"
    assert completed.kwargs["n_pairs"] == 1
    assert completed.kwargs["duration_ms"] >= 0
    mock_logger.error.assert_not_called()


def test_failure_is_logged_and_propagates() -> None:
    """Emit ``calculation_failed`` for a rejected option and re-raise it unchanged."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger
    with (
        mock.patch.object(events_module, "_logger", mock_logger),
        pytest.raises(InvalidArgumentError, match="lag_window applies to sequential matching only"),
    ):
        match(manual([0], [1]), manual([0], [1]), steps=5, lag_window=(1, 2))
    mock_logger.info.assert_called_once_with("calculation_started")
    failed = [call for call in mock_logger.error.call_args_list if call.args[0] == "calculation_failed"]
    assert len(failed) == 1
    assert failed[0].kwargs["error_type"] == "InvalidArgumentError"


# --- brute-force oracle -----------------------------------------------------------------------


def random_runs(generator: np.random.Generator, steps: int, count: int) -> list[tuple[int, int]]:
    """Return ``count`` sorted nonoverlapping runs inside ``steps``, some of them adjacent."""
    runs: list[tuple[int, int]] = []
    position = 0
    while len(runs) < count and position < steps:
        position += int(generator.integers(0, 3))
        length = int(generator.integers(1, 4))
        if position + length > steps:
            break
        runs.append((position, position + length - 1))
        position += length
    return runs


@pytest.mark.parametrize("mode", ["concurrent", "sequential"])
@pytest.mark.parametrize("lag_window", [None, (1, 2), (2, 5)])
def test_matching_agrees_with_a_brute_force_enumeration(mode: str, lag_window: tuple[int, int] | None) -> None:
    """Check the enumeration, the censor flags, and the bridge count against an independent scan."""
    generator = np.random.default_rng(20260618)
    steps = 60
    first_pairs = random_runs(generator, steps, 12)
    second_pairs = random_runs(generator, steps, 12)
    covered = {step for start, end in first_pairs + second_pairs for step in range(start, end + 1)}
    # interior unknowns outside every run exercise the bridge rule and censoring without
    # contradicting a run's assertion that it is a contiguous observed spell
    unknown = [step for step in range(steps) if step not in covered][:4]
    valid = np.ones(steps, dtype=bool)
    valid[unknown] = False
    if mode == "concurrent":
        lag_window = None
    result = match(
        manual([start for start, _ in first_pairs], [end for _, end in first_pairs]),
        manual([start for start, _ in second_pairs], [end for _, end in second_pairs]),
        steps=steps,
        valid=valid,
        mode=mode,
        lag_window=lag_window,
    )
    window = (1, None) if lag_window is None else lag_window
    expected = brute_force(first_pairs, second_pairs, valid, mode, window)
    assert result.first_run_id.tolist() == [row[0] for row in expected]
    assert result.second_run_id.tolist() == [row[1] for row in expected]
    assert result.start_index.tolist() == [row[2] for row in expected]
    assert result.end_index.tolist() == [row[3] for row in expected]
    assert result.first_left_censored.tolist() == [censored(first_pairs[i][0], -1, valid) for i, _, _, _ in expected]
    assert result.second_right_censored.tolist() == [censored(second_pairs[j][1], 1, valid) for _, j, _, _ in expected]
    if mode == "concurrent":
        assert result.overlap_duration.tolist() == [row[3] - row[2] + 1 for row in expected]
    else:
        assert result.lag.tolist() == [second_pairs[j][0] - first_pairs[i][1] for i, j, _, _ in expected]
        assert result.overlap_duration.tolist() == [0] * len(expected)
        rejected = [
            (i, j)
            for i, (_, first_end) in enumerate(first_pairs)
            for j, (second_start, _) in enumerate(second_pairs)
            if second_start - first_end >= window[0]
            and (window[1] is None or second_start - first_end <= window[1])
            and not np.all(valid[first_end : second_start + 1])
        ]
        assert rejected, "the seeded case must exercise the bridge rule"
        assert result.metadata["n_unknown_bridges"] == len(rejected)

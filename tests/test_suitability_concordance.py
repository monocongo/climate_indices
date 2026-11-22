"""Tests for ``suitability.event_concordance`` and ``category_agreement`` (#1590, ADR-0028 section 7)."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pytest

from climate_indices import suitability
from climate_indices.exceptions import BetaFeatureWarning, InputTypeError, InvalidArgumentError
from climate_indices.runs import RunSet

pytestmark = pytest.mark.filterwarnings("ignore::climate_indices.exceptions.BetaFeatureWarning")


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


def manual(starts: list[int], ends: list[int]) -> RunSet:
    """Build a RunSet from inclusive run bounds, with consistent metrics."""
    start_index = np.asarray(starts, dtype=np.int64)
    end_index = np.asarray(ends, dtype=np.int64)
    duration = end_index - start_index + 1
    return RunSet(
        start_index=start_index,
        end_index=end_index,
        duration=duration,
        magnitude=duration.astype(np.float64),
        intensity=np.ones_like(duration, dtype=np.float64),
        peak_value=np.full(duration.shape, -1.5, dtype=np.float64),
        peak_index=start_index.copy(),
        interarrival=np.full(duration.shape, np.nan, dtype=np.float64),
    )


EMPTY = manual([], [])


def concordance(
    index_runs: RunSet,
    reference_runs: RunSet,
    *,
    steps: int = 10,
    valid: np.ndarray | None = None,
    cadence: Any = "monthly",
) -> suitability.EventConcordance:
    """Count concordance on a fresh grid, with every step observed unless the caller says otherwise."""
    axis = months(steps) if cadence == "monthly" else days(steps)
    mask = np.ones(steps, dtype=bool) if valid is None else valid
    return suitability.event_concordance(index_runs, reference_runs, time=axis, valid=mask, cadence=cadence)


# ---- event_concordance: counting (ADR-0028 section 7) ---------------------------------------------


def test_one_overlap_is_one_hit() -> None:
    result = concordance(manual([2], [4]), manual([1], [6]))

    assert (result.hits, result.misses, result.false_alarms) == (1, 0, 0)
    assert result.matched_index_events == 1
    assert (result.pod, result.far, result.csi) == (1.0, 0.0, 1.0)


def test_one_index_event_overlapping_two_reference_events_is_two_hits_and_one_match() -> None:
    # hits count reference events, while a false-alarm denominator counts index events
    result = concordance(manual([2], [5]), manual([1, 4], [3, 6]))

    assert (result.hits, result.misses) == (2, 0)
    assert (result.matched_index_events, result.false_alarms) == (1, 0)
    assert (result.pod, result.far) == (1.0, 0.0)
    assert result.csi == pytest.approx(1.0)


def test_one_reference_event_overlapping_two_index_events_is_one_hit_and_two_matches() -> None:
    result = concordance(manual([1, 3], [2, 4]), manual([2], [3]))

    assert (result.hits, result.misses) == (1, 0)
    assert (result.matched_index_events, result.false_alarms) == (2, 0)
    assert (result.pod, result.far, result.csi) == (1.0, 0.0, 1.0)


def test_many_to_many_overlaps_count_each_side_once() -> None:
    # index 0-2 overlaps reference 1-4; index 5-6 overlaps no reference; reference 7-9 overlaps no index
    result = concordance(manual([0, 5], [2, 6]), manual([1, 7], [4, 9]))

    assert (result.hits, result.misses, result.false_alarms) == (1, 1, 1)
    assert result.matched_index_events == 1
    assert (result.pod, result.far) == (0.5, 0.5)
    assert result.csi == pytest.approx(1.0 / 3.0)


def _naive_counts(index: list[tuple[int, int]], reference: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """The section 7 definitions applied literally, as an O(n^2) oracle over the interval pairs."""

    def overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
        return not (first[1] < second[0] or second[1] < first[0])

    hits = sum(any(overlaps(run, event) for event in index) for run in reference)
    matched = sum(any(overlaps(run, event) for event in reference) for run in index)
    return hits, len(reference) - hits, len(index) - matched, matched


def _disjoint_runs(rng: np.random.Generator, steps: int, max_runs: int) -> list[tuple[int, int]]:
    """Random sorted, nonoverlapping inclusive runs inside the axis."""
    runs: list[tuple[int, int]] = []
    cursor = 0
    for _ in range(int(rng.integers(0, max_runs + 1))):
        cursor += int(rng.integers(0, 3))
        if cursor >= steps:
            break
        end = min(cursor + int(rng.integers(0, 4)), steps - 1)
        runs.append((cursor, end))
        cursor = end + 1
    return runs


def test_counts_match_a_brute_force_intersection_over_random_run_sets() -> None:
    # the counting has four interacting parts (hits, misses, false alarms, matched index events), so
    # compare it against the definitions applied pair by pair rather than only on chosen fixtures
    rng = np.random.default_rng(1590)
    steps = 14
    for _ in range(200):
        index = _disjoint_runs(rng, steps, 4)
        reference = _disjoint_runs(rng, steps, 4)
        result = concordance(
            manual([run[0] for run in index], [run[1] for run in index]),
            manual([run[0] for run in reference], [run[1] for run in reference]),
            steps=steps,
        )

        assert (result.hits, result.misses, result.false_alarms, result.matched_index_events) == _naive_counts(
            index, reference
        ), (index, reference)


def test_an_overlap_of_one_step_only_still_counts() -> None:
    result = concordance(manual([0], [3]), manual([3], [8]))

    assert result.hits == 1
    assert result.matched_index_events == 1
    assert result.false_alarms == 0


def test_no_events_on_either_side_leaves_every_ratio_nan() -> None:
    result = concordance(EMPTY, EMPTY)

    assert (result.hits, result.misses, result.false_alarms, result.matched_index_events) == (0, 0, 0, 0)
    assert math.isnan(result.pod)
    assert math.isnan(result.far)
    assert math.isnan(result.csi)


def test_reference_events_without_index_events_zero_pod_and_csi_but_not_far() -> None:
    # only FAR's denominator is empty here, so only FAR is NaN
    result = concordance(EMPTY, manual([1], [3]))

    assert (result.hits, result.misses) == (0, 1)
    assert result.pod == 0.0
    assert result.csi == 0.0
    assert math.isnan(result.far)


def test_index_events_without_reference_events_leave_pod_nan() -> None:
    result = concordance(manual([1], [3]), EMPTY)

    assert (result.hits, result.misses) == (0, 0)
    assert (result.false_alarms, result.matched_index_events) == (1, 0)
    assert math.isnan(result.pod)
    assert result.far == 1.0
    assert result.csi == 0.0


def test_daily_cadence_is_accepted() -> None:
    result = concordance(manual([1], [3]), manual([2], [4]), steps=12, cadence="daily")

    assert result.hits == 1
    assert result.pod == 1.0


# ---- event_concordance: censoring ---------------------------------------------------------------


def test_runs_touching_the_record_edges_are_counted_as_censored() -> None:
    result = concordance(manual([0, 8], [1, 9]), manual([0], [9]))

    assert result.index_left_censored == 1
    assert result.index_right_censored == 1
    assert result.reference_left_censored == 1
    assert result.reference_right_censored == 1


def test_an_interior_run_is_not_censored() -> None:
    result = concordance(manual([3], [4]), manual([5], [6]))

    assert (result.index_left_censored, result.index_right_censored) == (0, 0)
    assert (result.reference_left_censored, result.reference_right_censored) == (0, 0)


def test_an_unmatched_run_next_to_an_unknown_step_is_censored() -> None:
    # the per-pair rows cannot carry this: no pair exists, but the boundary is still unknown
    valid = np.ones(10, dtype=bool)
    valid[3] = False
    result = concordance(manual([4], [5]), manual([0], [1]), valid=valid)

    assert result.index_left_censored == 1
    assert result.index_right_censored == 0


def test_a_run_crossing_an_unknown_step_is_rejected() -> None:
    valid = np.ones(10, dtype=bool)
    valid[4] = False

    with pytest.raises(InvalidArgumentError, match="unknown observation"):
        concordance(manual([2], [6]), EMPTY, valid=valid)


# ---- event_concordance: contract -----------------------------------------------------------------


def test_metadata_records_the_counting_rule_and_the_event_counts() -> None:
    result = concordance(manual([0, 5], [2, 6]), manual([1], [4]))

    assert "reference" in result.metadata["climate_indices_concordance_counting"].lower()
    assert result.metadata["climate_indices_n_index_events"] == "2"
    assert result.metadata["climate_indices_n_reference_events"] == "1"
    assert "climate_indices_provenance" in result.metadata


def test_beta_warning_is_raised() -> None:
    with pytest.warns(BetaFeatureWarning):
        suitability.event_concordance(EMPTY, EMPTY, time=months(10), valid=np.ones(10, dtype=bool), cadence="monthly")


def test_a_non_run_set_source_is_rejected() -> None:
    not_a_run_set: Any = [(0, 1)]

    with pytest.raises(InputTypeError, match="RunSet"):
        suitability.event_concordance(
            not_a_run_set, EMPTY, time=months(10), valid=np.ones(10, dtype=bool), cadence="monthly"
        )


# ---- category_agreement: pilot parity -----------------------------------------------------------


def _pilot_shares(index_codes: np.ndarray, reference_codes: np.ndarray) -> tuple[float, float]:
    """The #1569 pilot's two expressions, from ``comparison_metrics`` in the Earth Engine pilot script.

    The pilot writes ``float(np.mean(usdm_bin(a) == usdm_bin(b)))`` and
    ``float(np.mean(np.abs(usdm_bin(a).astype(int) - usdm_bin(b).astype(int)) <= 1))`` over the pairs
    where both values exist. Its ``usdm_bin`` only maps values to integer classes, so the two
    expressions are what the shares mean; the inputs here are already the classes.
    """
    a = np.asarray(index_codes).astype(int)
    b = np.asarray(reference_codes).astype(int)
    present = (a >= 0) & (b >= 0)
    a, b = a[present], b[present]
    return float(np.mean(a == b)), float(np.mean(np.abs(a - b) <= 1))


def test_category_agreement_reproduces_the_pilot_definitions() -> None:
    rng = np.random.default_rng(1590)
    index_codes = rng.integers(0, 6, size=200).astype(np.int8)
    reference_codes = rng.integers(0, 6, size=200).astype(np.int8)
    # a few missing values, which the pilot drops and so must this function
    index_codes[:5] = -1
    reference_codes[5:9] = -1

    assert suitability.category_agreement(index_codes, reference_codes) == pytest.approx(
        _pilot_shares(index_codes, reference_codes)
    )


def test_category_agreement_hand_computed() -> None:
    # 4 pairs: (0,0) exact, (1,2) within one, (2,0) two classes apart, (5,5) exact
    assert suitability.category_agreement([0, 1, 2, 5], [0, 2, 0, 5]) == (0.5, 0.75)


def test_category_agreement_drops_pairs_with_a_missing_side() -> None:
    with_missing = suitability.category_agreement([0, 1, -1], [0, 1, 4])
    without = suitability.category_agreement([0, 1], [0, 1])

    assert with_missing == without == (1.0, 1.0)


def test_category_agreement_of_only_missing_pairs_is_nan() -> None:
    exact, within_one = suitability.category_agreement([-1, -1], [3, -1])

    assert math.isnan(exact)
    assert math.isnan(within_one)


def test_category_agreement_agrees_with_compare_categories() -> None:
    # two independent code paths over the same pair counts: exact is the complement of share_changed
    reference = np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int8)
    alternative = np.array([0, 1, 3, 4, 5, 5, 2, 5], dtype=np.int8)

    exact, within_one = suitability.category_agreement(alternative, reference)
    comparison = suitability.compare_categories(reference, alternative)

    assert exact == pytest.approx(1.0 - comparison.share_changed)
    assert within_one == pytest.approx(_pilot_shares(alternative, reference)[1])


def test_category_agreement_handles_unsigned_codes() -> None:
    # an unsigned subtraction wraps below zero (1 - 2 -> 255), so abs() would call adjacent classes far apart
    assert suitability.category_agreement(np.array([1], dtype=np.uint8), np.array([2], dtype=np.uint8)) == (
        0.0,
        1.0,
    )


def test_category_agreement_rejects_a_shape_mismatch() -> None:
    with pytest.raises(InvalidArgumentError, match="same shape"):
        suitability.category_agreement([0, 1], [0, 1, 2])


def test_category_agreement_rejects_float_codes() -> None:
    with pytest.raises(InvalidArgumentError, match="integer"):
        suitability.category_agreement([0.0, 1.0], [0.0, 1.0])


def test_category_agreement_rejects_a_code_out_of_range() -> None:
    with pytest.raises(InvalidArgumentError, match="outside"):
        suitability.category_agreement([6], [0])


def test_category_agreement_treats_a_masked_code_as_missing() -> None:
    masked: Any = np.ma.masked_array([0, 1, 2], mask=[False, True, False])

    assert suitability.category_agreement(masked, np.array([0, 1, 2])) == (1.0, 1.0)

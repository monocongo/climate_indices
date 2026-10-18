"""Tests for ``compound.joint_statistics()`` and ``JointResult`` (CMP-04, ADR-0022).

Counts are integers and the analytic cases are exact, so those comparisons use equality
rather than a tolerance. The Gaussian case is a stochastic characterization with a
predeclared Monte Carlo band, not an equality claim: with ``rho=0.5`` the exact joint
probability is ``1/4 + asin(0.5)/(2*pi) = 1/3`` and, on balanced halves, ``LMF = 4/3``.
Independent hand-counted loops and closed-form values supply the answers; no test
recomputes them through the implementation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import warnings
from collections import UserDict
from typing import Any, Literal
from unittest import mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import climate_indices
from climate_indices import compound
from climate_indices.compound import _statistics as statistics_module
from climate_indices.exceptions import (
    ClimateIndicesWarning,
    CoordinateValidationError,
    DataShapeError,
    InputTypeError,
    InvalidArgumentError,
)


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def exact(value: float) -> Any:
    """Match a closed-form value at ``atol=1e-12`` with ``rtol=0``, the tolerance VALIDATION.md states."""
    return pytest.approx(value, rel=0, abs=1e-12)


def same_number(left: float, right: float) -> bool:
    """Report whether two floats agree, treating two NaNs as equal for an invariance check."""
    return (math.isnan(left) and math.isnan(right)) or left == exact(right)


def align(
    hazard: Any,
    valid: Any = None,
    *,
    names: tuple[str, ...] | None = None,
    cadence: Literal["daily", "monthly"] = "monthly",
    start: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> compound.AlignedHazards:
    """Build an ``AlignedHazards`` directly, so these tests do not depend on harmonize()."""
    hazard = np.asarray(hazard, dtype=bool)
    if valid is None:
        valid = np.ones(hazard.shape, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if names is None:
        names = tuple(f"source_{index}" for index in range(hazard.shape[0]))
    if start is None:
        start = "2000-01-01" if cadence == "daily" else "2000-01"
    steps = hazard.shape[1]
    unit = "D" if cadence == "daily" else "M"
    # harmonize() labels a monthly step by the first civil day of its calendar month
    time = (np.datetime64(start, unit) + np.arange(steps)).astype("datetime64[D]")
    return compound.AlignedHazards(
        names,
        time,
        cadence,
        hazard,
        valid,
        {} if metadata is None else metadata,  # type: ignore[arg-type]
    )


def test_pooled_counts_match_a_hand_counted_contingency_table() -> None:
    """Count one eligible support by hand, then check every pooled field against it."""
    hazard = np.array(
        [
            [True, False, True, True, False, True],
            [True, True, False, True, False, False],
        ]
    )
    result = compound.joint_statistics(align(hazard), groupby="pooled", min_samples=1)

    rows = list(zip(*hazard.tolist(), strict=True))
    assert rows == [
        (True, True),
        (False, True),
        (True, False),
        (True, True),
        (False, False),
        (True, False),
    ]
    n_valid = len(rows)
    n_first = sum(first for first, _ in rows)
    n_second = sum(second for _, second in rows)
    n_joint = sum(first and second for first, second in rows)
    assert (n_valid, n_first, n_second, n_joint) == (6, 4, 3, 2)

    assert int(result.n_valid) == n_valid
    assert int(result.n_joint) == n_joint
    assert result.n_marginal.tolist() == [n_first, n_second]
    assert result.marginal_frequency.tolist() == [exact(4 / 6), exact(3 / 6)]
    assert float(result.joint_probability) == exact(2 / 6)
    assert float(result.independent_probability) == exact((4 / 6) * (3 / 6))
    assert float(result.likelihood_multiplication_factor) == exact((2 / 6) / ((4 / 6) * (3 / 6)))
    assert int(result.quality_flags) == 0
    assert result.groupby == "pooled"


def test_balanced_cartesian_table_is_exactly_independent_with_lmf_one() -> None:
    """A balanced design makes the marginals exact halves, so LMF is exactly 1."""
    patterns = np.array([[first, second] for first in (False, True) for second in (False, True)], dtype=bool)
    hazard = np.repeat(patterns.T, 25, axis=1)
    result = compound.joint_statistics(align(hazard), groupby="pooled", min_samples=1)
    assert result.n_marginal.tolist() == [50, 50]
    assert int(result.n_valid) == 100
    assert int(result.n_joint) == 25
    assert float(result.joint_probability) == 0.25
    assert float(result.independent_probability) == 0.25
    assert float(result.likelihood_multiplication_factor) == 1.0


def test_three_source_mutual_independence_is_exact() -> None:
    """Three balanced sources give an eighth of the steps jointly and LMF exactly 1."""
    patterns = np.array(
        [[first, second, third] for first in (False, True) for second in (False, True) for third in (False, True)],
        dtype=bool,
    )
    hazard = np.repeat(patterns.T, 12, axis=1)
    result = compound.joint_statistics(align(hazard), groupby="pooled", min_samples=1)
    assert result.n_marginal.tolist() == [48, 48, 48]
    assert int(result.n_valid) == 96
    assert int(result.n_joint) == 12
    # 12/96 and 1.0 are exact in binary floating point, so these are equalities, not tolerances
    assert float(result.joint_probability) == 0.125
    assert float(result.independent_probability) == 0.125
    assert float(result.likelihood_multiplication_factor) == 1.0


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_gaussian_copula_joint_probability_matches_the_closed_form(rho: float) -> None:
    """Check ``P(Z1 > 0, Z2 > 0) = 1/4 + asin(rho)/(2*pi)`` against a predeclared 5-sigma band.

    ``rho=0`` gives the exactly independent ``1/4``; ``rho=0.5`` gives ``1/3``, which is
    ``LMF=4/3`` on balanced halves. Each band is five standard errors of the sampled
    quantity: ``sqrt(p*(1-p)/n)`` for the joint frequency and, by the delta method,
    ``LMF * sqrt((1-p_joint)/(n*p_joint) + 2*(1-p_i)/(n*p_i))`` for the ratio. These are
    predeclared bands from a fixed seed, not fitted tolerances.
    """
    n = 200_000
    generator = np.random.default_rng(20241018)
    values = generator.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    hazard = (values > 0.0).T
    result = compound.joint_statistics(
        align(hazard, cadence="daily", start="2000-01-01"), groupby="pooled", min_samples=1
    )

    analytic = 0.25 + math.asin(rho) / (2.0 * math.pi)
    assert analytic == exact(0.25 if rho == 0.0 else 1 / 3)
    joint_band = 5.0 * math.sqrt(analytic * (1.0 - analytic) / n)
    marginal_band = 5.0 * math.sqrt(0.5 * 0.5 / n)
    lmf = 1.0 if rho == 0.0 else 4 / 3
    lmf_band = 5.0 * lmf * math.sqrt((1.0 - analytic) / (n * analytic) + 2.0 * (1.0 - 0.5) / (n * 0.5))
    assert int(result.n_valid) == n
    assert float(result.joint_probability) == pytest.approx(analytic, abs=joint_band)
    assert result.marginal_frequency.tolist() == pytest.approx([0.5, 0.5], abs=marginal_band)
    assert float(result.likelihood_multiplication_factor) == pytest.approx(lmf, abs=lmf_band)


def test_zero_joint_count_with_positive_marginals_is_zero_not_nan() -> None:
    """Disjoint hazards keep positive marginals, so LMF is a real zero rather than undefined."""
    hazard = np.array([[True, True, False, False], [False, False, True, True]])
    result = compound.joint_statistics(align(hazard), groupby="pooled", min_samples=1)
    assert int(result.n_joint) == 0
    assert float(result.joint_probability) == 0.0
    assert float(result.independent_probability) == exact((2 / 4) * (2 / 4))
    assert float(result.likelihood_multiplication_factor) == 0.0
    assert int(result.quality_flags) == 0


def test_zero_marginal_makes_lmf_undefined_with_its_flag() -> None:
    """A hazard that never occurs leaves LMF undefined; no pseudocount rescues it."""
    hazard = np.array([[True, True, False, False], [False, False, False, False]])
    with pytest.warns(ClimateIndicesWarning, match="zero marginal"):
        result = compound.joint_statistics(align(hazard), groupby="pooled", min_samples=1)
    assert result.n_marginal.tolist() == [2, 0]
    assert float(result.independent_probability) == 0.0
    assert float(result.joint_probability) == 0.0
    assert math.isnan(float(result.likelihood_multiplication_factor))
    assert int(result.quality_flags) == 2
    assert result.metadata["quality_flags"] == {"insufficient_support": 1, "undefined_denominator": 2}


def test_short_support_keeps_counts_and_nans_the_estimates() -> None:
    """Below min_samples the counts survive and every estimate becomes NaN."""
    hazard = np.array([[True, False, True], [False, True, True]])
    with pytest.warns(ClimateIndicesWarning, match="min_samples=30"):
        result = compound.joint_statistics(align(hazard), groupby="pooled")
    assert int(result.n_valid) == 3
    assert int(result.n_joint) == 1
    assert result.n_marginal.tolist() == [2, 2]
    assert all(math.isnan(float(value)) for value in result.marginal_frequency)
    assert math.isnan(float(result.joint_probability))
    assert math.isnan(float(result.independent_probability))
    assert math.isnan(float(result.likelihood_multiplication_factor))
    assert int(result.quality_flags) == 1

    relaxed = compound.joint_statistics(align(hazard), groupby="pooled", min_samples=2)
    assert int(relaxed.quality_flags) == 0
    assert float(relaxed.likelihood_multiplication_factor) == exact((1 / 3) / ((2 / 3) * (2 / 3)))


def test_invalid_steps_are_never_a_non_exceedance() -> None:
    """An unobserved step is excluded from the support, even where the hazard array says True."""
    hazard = np.array([[True, True, True], [False, True, True]])
    valid = np.array([[False, True, True], [False, True, True]])
    result = compound.joint_statistics(align(hazard, valid), groupby="pooled", min_samples=1)
    # the first step is unknown for both sources; its True hazard is not an occurrence
    assert int(result.n_valid) == 2
    assert result.n_marginal.tolist() == [2, 2]
    assert int(result.n_joint) == 2
    assert float(result.marginal_frequency[0]) == 1.0


def test_every_marginal_uses_the_complete_case_support() -> None:
    """A step where any selected source is unknown is eligible for none of them.

    Per-source validity would count a step for one source while dividing by a smaller
    complete-case denominator, which can push a marginal frequency above 1.
    """
    hazard = np.array([[True, True, False, True], [False, False, True, True]])
    valid = np.array([[True, True, True, True], [True, False, True, True]])
    result = compound.joint_statistics(align(hazard, valid), groupby="pooled", min_samples=1)
    # step 1 is unknown for the second source, so no source counts it
    assert int(result.n_valid) == 3
    assert result.n_marginal.tolist() == [2, 2]
    assert int(result.n_joint) == 1
    assert result.marginal_frequency.tolist() == [exact(2 / 3), exact(2 / 3)]
    assert float(result.joint_probability) == exact(1 / 3)


@pytest.mark.parametrize(
    ("exceedance", "valid", "expected"),
    [
        # two sources: step 1 is unknown for `b`, so no source counts it
        (
            [[True, True, False, True], [False, False, True, True]],
            [[True, True, True, True], [True, False, True, True]],
            (3, 1, 2, 2),
        ),
        # three sources: only steps 0 and 2 have all three observed, and they share no joint occurrence
        (
            [[True, True, False, True], [False, False, True, True], [True, True, True, True]],
            [[True, True, True, True], [True, False, True, True], [True, True, True, False]],
            (2, 0, 1, 1, 2),
        ),
    ],
)
def test_marginals_never_exceed_the_shared_support(
    exceedance: list[list[bool]], valid: list[list[bool]], expected: tuple[int, ...]
) -> None:
    """``n_valid``, ``n_joint``, and every marginal come from one complete-case support."""
    result = compound.joint_statistics(align(np.array(exceedance), np.array(valid)), groupby="pooled", min_samples=1)
    assert (int(result.n_valid), int(result.n_joint), *result.n_marginal.tolist()) == expected
    assert all(int(count) <= int(result.n_valid) for count in result.n_marginal)
    assert all(float(value) <= 1.0 for value in result.marginal_frequency)


def test_all_missing_cell_returns_zero_counts_without_failing_its_neighbor() -> None:
    """An all-missing grid cell yields zero counts and flags while the observed cell is unaffected."""
    hazard = np.array(
        [
            [[True, False], [False, False], [True, False], [True, False]],
            [[True, False], [True, False], [False, False], [False, False]],
        ]
    )
    valid = np.ones(hazard.shape, dtype=bool)
    valid[:, :, 1] = False
    with pytest.warns(ClimateIndicesWarning, match="fewer than 10 occurrences"):
        result = compound.joint_statistics(align(hazard, valid), groupby="pooled", min_samples=1)
    assert result.n_valid.tolist() == [4, 0]
    assert result.n_joint.tolist() == [1, 0]
    assert result.n_marginal.tolist() == [[3, 0], [2, 0]]
    assert result.n_marginal.shape == (2, 2)
    assert result.marginal_frequency[0][0] == 0.75
    assert result.marginal_frequency[1][0] == 0.5
    assert result.quality_flags.tolist() == [0, 3]
    assert float(result.likelihood_multiplication_factor[0]) == exact((1 / 4) / ((3 / 4) * (2 / 4)))
    assert math.isnan(float(result.likelihood_multiplication_factor[1]))


def test_monthly_groups_recount_pooled_support_instead_of_averaging_lmf() -> None:
    """A pooled LMF is a recount, so it can differ from the monthly groups it mixes."""
    steps = 48
    first = np.zeros(steps, dtype=bool)
    second = np.zeros(steps, dtype=bool)
    for year in range(4):
        first[year * 12 + 0] = True  # January
        first[year * 12 + 6] = True  # July
        second[year * 12 + 0] = True  # January
        second[year * 12 + 7] = True  # August
    hazard = np.array([first, second])

    with pytest.warns(ClimateIndicesWarning, match="fewer than 10 occurrences"):
        monthly_result = compound.joint_statistics(align(hazard), groupby="month", min_samples=1)
    strict = compound.joint_statistics(align(hazard), groupby="month")
    pooled = compound.joint_statistics(align(hazard), groupby="pooled", min_samples=1)

    assert monthly_result.n_valid.shape == (12,)
    assert monthly_result.n_marginal.shape == (2, 12)
    assert monthly_result.n_valid.tolist() == [4] * 12
    assert monthly_result.n_joint.tolist() == [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert monthly_result.n_marginal[0].tolist() == [4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0]
    assert monthly_result.n_marginal[1].tolist() == [4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0]
    assert float(monthly_result.likelihood_multiplication_factor[0]) == 1.0
    assert monthly_result.quality_flags[0] == 0  # min_samples=1 admits the four January steps
    assert monthly_result.quality_flags[6] == 2  # July: zero B marginal
    assert monthly_result.quality_flags[7] == 2  # August: zero A marginal

    # the default threshold keeps the counts, NaNs the estimates, and flags the short support
    assert strict.n_valid.tolist() == [4] * 12
    assert math.isnan(float(strict.likelihood_multiplication_factor[0]))
    assert strict.quality_flags[0] == 1
    assert strict.quality_flags[6] == 3

    assert int(pooled.n_valid) == steps
    assert pooled.n_marginal.tolist() == [8, 8]
    assert int(pooled.n_joint) == 4
    assert float(pooled.likelihood_multiplication_factor) == 3.0
    monthly_lmfs = monthly_result.likelihood_multiplication_factor
    assert float(pooled.likelihood_multiplication_factor) != pytest.approx(float(np.nanmean(monthly_lmfs)))


@st.composite
def two_source_rows(draw: st.DrawFn) -> tuple[list[bool], list[bool], list[bool], list[bool]]:
    """Draw two equal-length hazard rows and one validity mask per row.

    Separate masks matter: the marginal counts must use the shared complete-case support,
    and equal masks would hide a per-source-support defect.
    """
    length = draw(st.integers(min_value=2, max_value=10))
    row = st.lists(st.booleans(), min_size=length, max_size=length)
    return draw(st.tuples(row, row, row, row))


@given(rows=two_source_rows())
def test_source_permutation_is_symmetric(rows: tuple[list[bool], list[bool], list[bool], list[bool]]) -> None:
    """Swapping the two sources permutes the marginals and leaves joint values unchanged."""
    first, second, first_valid, second_valid = rows
    hazard = np.array([first, second], dtype=bool)
    valid = np.array([first_valid, second_valid], dtype=bool)
    forward = compound.joint_statistics(align(hazard, valid), groupby="pooled", min_samples=1)
    backward = compound.joint_statistics(align(hazard[::-1], valid[::-1]), groupby="pooled", min_samples=1)
    assert forward.n_marginal.tolist() == backward.n_marginal[::-1].tolist()
    assert int(forward.n_valid) == int(backward.n_valid)
    assert int(forward.n_joint) == int(backward.n_joint)
    # an all-missing draw reports NaN on both sides, so compare with NaN == NaN
    for field in ("joint_probability", "independent_probability", "likelihood_multiplication_factor"):
        assert same_number(float(getattr(forward, field)), float(getattr(backward, field)))


@given(rows=two_source_rows())
def test_counts_and_probabilities_stay_within_their_bounds(
    rows: tuple[list[bool], list[bool], list[bool], list[bool]],
) -> None:
    """Counts nest on one complete-case support and every probability lies in [0, 1]."""
    first, second, first_valid, second_valid = rows
    hazard = np.array([first, second], dtype=bool)
    valid = np.array([first_valid, second_valid], dtype=bool)
    result = compound.joint_statistics(align(hazard, valid), groupby="pooled", min_samples=1)
    n_valid = int(result.n_valid)
    marginals = [int(count) for count in result.n_marginal]
    assert all(0 <= count <= n_valid for count in marginals)
    assert 0 <= int(result.n_joint) <= min(marginals)
    for probability in (
        float(result.joint_probability),
        float(result.independent_probability),
        *(float(value) for value in result.marginal_frequency),
    ):
        assert math.isnan(probability) or 0.0 <= probability <= 1.0
    lmf = float(result.likelihood_multiplication_factor)
    assert math.isnan(lmf) or lmf >= 0.0


def test_result_arrays_are_read_only_copied_and_frozen() -> None:
    """Freeze the arrays and the instance; an incomplete or a shared record is not a result."""
    hazard = np.array([[True, False, True], [False, True, True]])
    result = compound.joint_statistics(align(hazard), groupby="pooled", min_samples=1)
    for array in (
        result.n_valid,
        result.n_joint,
        result.n_marginal,
        result.marginal_frequency,
        result.joint_probability,
        result.independent_probability,
        result.likelihood_multiplication_factor,
        result.quality_flags,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = array.flat[0]
    assert not np.shares_memory(result.n_marginal, hazard)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.groupby = "pooled"  # type: ignore[misc]


def test_manifest_serializes_with_the_audit_choices_and_the_harmonization_record() -> None:
    """Round-trip the manifest through JSON and keep the harmonization declarations reachable."""
    hazard = np.array([[True, False, True, False], [True, True, False, False]])
    aligned = align(hazard, metadata={"calendar": "proleptic_gregorian", "cadence": "monthly"})
    result = compound.joint_statistics(aligned, groupby="pooled", min_samples=2)
    text = json.dumps(result.metadata, allow_nan=False)
    manifest = json.loads(text)
    assert manifest["schema_version"] == 1
    assert manifest["function"] == "joint_statistics"
    assert manifest["groupby"] == "pooled"
    assert manifest["min_samples"] == 2
    assert manifest["calendar"] == "proleptic_gregorian"
    assert manifest["cadence"] == "monthly"
    assert manifest["n_steps"] == 4
    assert manifest["quality_flags"]["undefined_denominator"] == 2
    assert "complete case" in manifest["eligibility"]
    assert "mutual independence for three sources" in manifest["estimands"]
    assert "does not establish independence" in manifest["lmf_rule"]
    assert manifest["harmonization"]["cadence"] == "monthly"
    assert result.metadata["grouped_by"] == "pooled over every eligible step"


def test_custom_mapping_metadata_is_detached_and_frozen() -> None:
    """A non-dict Mapping is copied by value; a value JSON cannot represent is rejected."""
    hazard = np.array([[True, False], [False, True]])
    aligned = align(hazard, metadata=UserDict({"calendar": "proleptic_gregorian"}))  # type: ignore[arg-type]
    result = compound.joint_statistics(aligned, groupby="pooled", min_samples=1)
    record = result.metadata["harmonization"]
    assert record["calendar"] == "proleptic_gregorian"
    assert not isinstance(record, UserDict)
    assert json.loads(json.dumps(result.metadata))["harmonization"]["calendar"] == "proleptic_gregorian"
    with pytest.raises(TypeError):
        record["calendar"] = "later"

    unserializable = align(hazard, metadata={"bad": {1, 2}})
    with pytest.raises(InvalidArgumentError, match="JSON-compatible"):
        compound.joint_statistics(unserializable, groupby="pooled", min_samples=1)


def test_daily_cadence_is_grouped_by_civil_month() -> None:
    """Daily steps use their real calendar month, including the day after a leap day."""
    dates = np.arange("2000-01-01", "2001-01-01", dtype="datetime64[D]")
    steps = dates.shape[0]
    hazard = np.zeros((2, steps), dtype=bool)
    hazard[:, dates.astype("datetime64[M]") == np.datetime64("2000-02", "M")] = True
    with pytest.warns(ClimateIndicesWarning):
        result = compound.joint_statistics(align(hazard, cadence="daily", start="2000-01-01"), min_samples=1)
    assert result.n_valid.tolist() == [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    assert result.n_joint.tolist() == [0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert result.n_marginal[0].tolist() == [0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert float(result.likelihood_multiplication_factor[1]) == 1.0


@pytest.mark.parametrize(
    ("hazards", "error", "message"),
    [
        (None, InvalidArgumentError, "AlignedHazards"),
        ([1, 2, 3], InvalidArgumentError, "AlignedHazards"),
    ],
)
def test_rejects_a_non_aligned_argument(hazards: Any, error: type[Exception], message: str) -> None:
    """Require the prepared input type instead of guessing at raw arrays."""
    with pytest.raises(error, match=message):
        compound.joint_statistics(hazards)


def test_rejects_a_single_source() -> None:
    """One hazard has no joint frequency to report."""
    aligned = align(np.array([[True, False, True]]))
    with pytest.raises(InvalidArgumentError, match="two or three sources"):
        compound.joint_statistics(aligned)


def test_rejects_four_sources() -> None:
    """The milestone bounds the count pass at three sources."""
    aligned = align(np.ones((4, 3), dtype=bool))
    with pytest.raises(InvalidArgumentError, match="two or three sources"):
        compound.joint_statistics(aligned)


def test_rejects_duplicate_source_names() -> None:
    """Duplicate names would make the marginal rows ambiguous."""
    aligned = align(np.array([[True, False, True], [False, True, True]]), names=("a", "a"))
    with pytest.raises(InvalidArgumentError, match="unique"):
        compound.joint_statistics(aligned)


def test_rejects_a_manually_constructed_result_with_bad_arrays() -> None:
    """A dataclass is not proof of validity: revalidate rows, columns, dtypes, and dates."""
    hazard = np.array([[True, False, True], [False, True, True]])
    aligned = align(hazard)
    row_count = dataclasses.replace(aligned, exceedance=aligned.exceedance[:1])
    with pytest.raises(DataShapeError, match="one row per source name"):
        compound.joint_statistics(row_count)

    column_count = dataclasses.replace(aligned, exceedance=aligned.exceedance[:, :2])
    with pytest.raises(DataShapeError, match="one column per observed date"):
        compound.joint_statistics(column_count)

    float_hazard = dataclasses.replace(aligned, exceedance=aligned.exceedance.astype(float))
    with pytest.raises(InputTypeError, match="Boolean numpy array"):
        compound.joint_statistics(float_hazard)

    mismatched = dataclasses.replace(aligned, valid=aligned.valid[:, :2])
    with pytest.raises(DataShapeError, match="same shape"):
        compound.joint_statistics(mismatched)

    unordered = dataclasses.replace(aligned, time=aligned.time[::-1])
    with pytest.raises(CoordinateValidationError, match="strictly increasing"):
        compound.joint_statistics(unordered)

    duplicated = dataclasses.replace(aligned, time=np.array([aligned.time[0]] * 3, dtype="datetime64[D]"))
    with pytest.raises(CoordinateValidationError, match="strictly increasing"):
        compound.joint_statistics(duplicated)

    nat = dataclasses.replace(aligned, time=np.array(["2000-01-01", "NaT", "2000-03-01"], dtype="datetime64[D]"))
    with pytest.raises(CoordinateValidationError, match="NaT"):
        compound.joint_statistics(nat)

    # harmonize() emits civil days; a coarser unit would group every step into the wrong month
    month_unit = dataclasses.replace(aligned, time=aligned.time.astype("datetime64[M]"))
    with pytest.raises(InputTypeError, match=r"datetime64\[D\]"):
        compound.joint_statistics(month_unit)

    numeric = dataclasses.replace(aligned, time=np.arange(3))
    with pytest.raises(InputTypeError, match="datetime64"):
        compound.joint_statistics(numeric)

    two_dimensional = dataclasses.replace(aligned, time=aligned.time.reshape(1, 3))
    with pytest.raises(DataShapeError, match="one-dimensional"):
        compound.joint_statistics(two_dimensional)

    # two labels inside one calendar month would count that month twice
    mid_month = dataclasses.replace(
        aligned, time=np.array(["2000-01-01", "2000-01-02", "2000-03-01"], dtype="datetime64[D]")
    )
    with pytest.raises(CoordinateValidationError, match="first civil day"):
        compound.joint_statistics(mid_month)


def test_single_step_harmonization_is_accepted_not_rejected() -> None:
    """harmonize() can emit one shared step; the statistics boundary keeps its counts and flags."""
    with pytest.warns(ClimateIndicesWarning):
        result = compound.joint_statistics(align(np.array([[True], [False]])), groupby="pooled", min_samples=1)
    assert int(result.n_valid) == 1
    assert int(result.n_joint) == 0
    assert result.n_marginal.tolist() == [1, 0]
    assert int(result.quality_flags) == 2


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("groupby", "seasonal", "groupby must be one of"),
        ("min_samples", 0, "positive integer"),
        ("min_samples", -1, "positive integer"),
        ("min_samples", True, "positive integer"),
        ("min_samples", 1.5, "positive integer"),
        ("min_samples", "30", "positive integer"),
    ],
)
def test_rejects_invalid_options(argument: str, value: Any, message: str) -> None:
    """Validate options before counting so a rejected call reports a structured error."""
    aligned = align(np.array([[True, False, True], [False, True, True]]))
    with pytest.raises(InvalidArgumentError, match=message):
        compound.joint_statistics(aligned, **{argument: value})


def test_never_mutates_the_input_arrays() -> None:
    """Counting is read-only: the caller's arrays keep their values and their writeability."""
    hazard = np.array([[True, False, True, True], [False, True, True, False]])
    valid = np.array([[True, True, True, False], [True, True, True, True]])
    before = (hazard.copy(), valid.copy())
    compound.joint_statistics(align(hazard, valid), groupby="pooled", min_samples=1)
    assert np.array_equal(hazard, before[0])
    assert np.array_equal(valid, before[1])
    assert hazard.flags.writeable and valid.flags.writeable


def test_lifecycle_events_are_emitted_on_success() -> None:
    """Emit ``calculation_started`` then ``calculation_completed`` with the group shape."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger
    aligned = align(np.array([[True, False, True, True], [False, True, True, False]]))
    with mock.patch.object(statistics_module, "_logger", mock_logger):
        compound.joint_statistics(aligned, groupby="month", min_samples=1)
    bound = mock_logger.bind.call_args.kwargs
    assert bound["index_type"] == "compound_joint_statistics"
    assert bound["n_sources"] == 2  # names are caller identifiers and stay out of the log
    assert bound["groupby"] == "month"
    assert "names" not in bound
    assert [call.args[0] for call in mock_logger.info.call_args_list] == [
        "calculation_started",
        "calculation_completed",
    ]
    completed = mock_logger.info.call_args_list[1].kwargs
    assert completed["output_shape"] == (12,)
    assert completed["duration_ms"] >= 0
    mock_logger.error.assert_not_called()


def test_failure_is_logged_and_propagates() -> None:
    """Emit ``calculation_failed`` for a rejected option and re-raise it unchanged."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger
    aligned = align(np.array([[True, False, True], [False, True, True]]))
    with (
        mock.patch.object(statistics_module, "_logger", mock_logger),
        pytest.raises(InvalidArgumentError, match="groupby"),
    ):
        compound.joint_statistics(aligned, groupby="seasonal")  # type: ignore[arg-type]
    mock_logger.info.assert_called_once_with("calculation_started")
    failed = [call for call in mock_logger.error.call_args_list if call.args[0] == "calculation_failed"]
    assert len(failed) == 1
    assert failed[0].kwargs["error_type"] == "InvalidArgumentError"
    assert "groupby" in failed[0].kwargs["error_message"]


def test_failure_logging_survives_a_non_aligned_argument() -> None:
    """Bind safely before validation so a malformed argument still reports a structured error."""
    with pytest.raises(InvalidArgumentError, match="AlignedHazards"):
        compound.joint_statistics("not aligned")  # type: ignore[arg-type]


def test_public_surface_is_explicit() -> None:
    """Expose the function and its result type from the compound namespace, not the package root."""
    assert "joint_statistics" in compound.__all__
    assert "JointResult" in compound.__all__
    assert not hasattr(climate_indices, "joint_statistics")


def test_short_record_warning_fires_once_for_the_whole_record() -> None:
    """Warn about a short record independently of per-group flags, without spamming each group."""
    hazard = np.array([[True, False], [False, True]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compound.joint_statistics(align(hazard), groupby="pooled", min_samples=1)
    short = [item for item in caught if "observed calendar year" in str(item.message)]
    assert len(short) == 1
    assert issubclass(short[0].category, ClimateIndicesWarning)


def test_record_length_counts_observed_years_not_month_arithmetic() -> None:
    """Count years with an observed step: a July-to-June 30-year record is not short."""
    thirty_years = np.ones((2, 360), dtype=bool)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compound.joint_statistics(align(thirty_years, start="1990-07"), groupby="pooled", min_samples=1)
    assert not [item for item in caught if "observed calendar year" in str(item.message)]

    four_months = np.ones((2, 4), dtype=bool)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compound.joint_statistics(align(four_months, start="2000-11"), groupby="pooled", min_samples=1)
    messages = [str(item.message) for item in caught]
    assert any("2 observed calendar year" in message for message in messages)


# --- conditional_probability() (CMP-05) -----------------------------------------------------


def conditional(aligned: compound.AlignedHazards, **kwargs: Any) -> compound.ConditionalResult:
    """Call ``conditional_probability()`` with the two default source names and the given options."""
    options: dict[str, Any] = {"given": "source_0", "target": "source_1", "groupby": "pooled", "min_samples": 1}
    options.update(kwargs)
    return compound.conditional_probability(aligned, **options)


def test_zero_lag_identity_matches_the_joint_and_marginal_ratio() -> None:
    """At (0, 0) the conditional frequency is the joint count over the given marginal count."""
    hazard = np.array(
        [
            [True, False, True, True, False, True],
            [True, True, False, True, False, False],
        ]
    )
    aligned = align(hazard)
    with pytest.warns(ClimateIndicesWarning):
        joint = compound.joint_statistics(aligned, groupby="pooled", min_samples=1)
    with pytest.warns(ClimateIndicesWarning):
        lagged = conditional(aligned)
    # rows (T,T) (F,T) (T,F) (T,T) (F,F) (T,F): 6 eligible, 4 given positives, 2 joint
    assert int(lagged.n_anchors) == int(joint.n_valid) == 6
    assert int(lagged.n_positive) == int(joint.n_marginal[0]) == 4
    assert int(lagged.n_success) == int(joint.n_joint) == 2
    assert float(lagged.conditional_probability) == exact(2 / 4)
    # the contemporaneous baseline is the target marginal on the same support, not a second estimand
    assert float(lagged.response_window_frequency) == float(joint.marginal_frequency[1]) == exact(3 / 6)


def test_forward_window_counts_each_given_positive_anchor_once() -> None:
    """A three-step window is hand-counted per anchor, including the incomplete trailing anchors."""
    hazard = np.array(
        [
            [True, False, True, False, True, False],
            [True, False, False, False, False, False],
        ]
    )
    aligned = align(hazard)
    with pytest.warns(ClimateIndicesWarning):
        result = conditional(aligned, lag_window=(0, 2))
    # anchors 0-3 have a complete window; anchors 4 and 5 do not
    assert int(result.n_anchors) == 4
    assert int(result.n_positive) == 2
    assert int(result.n_success) == 1
    assert int(result.n_response) == 1
    assert float(result.conditional_probability) == exact(0.5)
    assert float(result.response_window_frequency) == exact(0.25)
    assert result.lag_window == (0, 2)


def test_multiple_target_occurrences_in_one_window_count_the_anchor_once() -> None:
    """The numerator counts anchors, so a second target step in the window changes nothing."""
    hazard = np.array(
        [
            [True, False, False, False],
            [True, False, True, False],
        ]
    )
    aligned = align(hazard)
    with pytest.warns(ClimateIndicesWarning):
        result = conditional(aligned, lag_window=(0, 2))
    # anchor 0 sees the target at step 0 and again at step 2, but counts as one success
    assert int(result.n_anchors) == 2
    assert int(result.n_positive) == 1
    assert int(result.n_success) == 1
    assert int(result.n_response) == 2
    assert float(result.conditional_probability) == 1.0


def test_incomplete_horizon_is_discarded_even_when_it_holds_a_response() -> None:
    """The trailing anchor's window leaves the record, so the anchor is not eligible at all."""
    hazard = np.array([[True, True], [False, True]])
    aligned = align(hazard)
    with pytest.warns(ClimateIndicesWarning):
        result = conditional(aligned, lag_window=(0, 1))
    assert int(result.n_anchors) == 1
    assert int(result.n_positive) == 1
    assert int(result.n_success) == 1


def test_unknown_response_step_discards_the_anchor() -> None:
    """An unknown target step in the window makes the anchor ineligible rather than a non-response."""
    hazard = np.array(
        [
            [True, False, True, False, False],
            [True, False, False, True, False],
        ]
    )
    valid = np.ones(hazard.shape, dtype=bool)
    valid[1, 2] = False  # the window of anchors 0 and 1 holds this unknown step
    aligned = align(hazard, valid)
    with pytest.warns(ClimateIndicesWarning):
        result = conditional(aligned, lag_window=(1, 2))
    # treating the unknown step as a non-response would keep anchors 0 and 1 and report 0.5
    assert int(result.n_anchors) == 1
    assert int(result.n_positive) == 1
    assert int(result.n_success) == 1
    assert float(result.conditional_probability) == 1.0


def test_conditioning_is_not_symmetric() -> None:
    """Swapping the source roles changes the denominator while the baseline stays on the same anchors."""
    hazard = np.array([[True, True, True, False], [True, False, False, False]])
    aligned = align(hazard)
    with pytest.warns(ClimateIndicesWarning):
        forward = conditional(aligned, lag_window=(0, 1))
    with pytest.warns(ClimateIndicesWarning):
        backward = conditional(aligned, given="source_1", target="source_0", lag_window=(0, 1))
    assert float(forward.conditional_probability) == exact(1 / 3)
    assert float(backward.conditional_probability) == exact(1 / 1)
    # both share one eligible-anchor set, but each baseline describes its own target source
    assert int(forward.n_anchors) == int(backward.n_anchors) == 3
    assert float(forward.response_window_frequency) == exact(1 / 3)
    assert float(backward.response_window_frequency) == exact(3 / 3)


def test_baseline_is_the_response_window_rate_not_the_single_step_marginal() -> None:
    """The comparison is P(R) over the eligible anchors, not the one-step target frequency."""
    hazard = np.array(
        [
            [True, False, True, False, True, False],
            [True, False, False, True, False, False],
        ]
    )
    aligned = align(hazard)
    with pytest.warns(ClimateIndicesWarning):
        result = conditional(aligned, lag_window=(0, 1))
    with pytest.warns(ClimateIndicesWarning):
        joint = compound.joint_statistics(aligned, groupby="pooled", min_samples=1)
    assert int(result.n_positive) == 3
    assert int(result.n_success) == 2
    assert float(result.conditional_probability) == exact(2 / 3)
    assert int(result.n_response) == 3
    assert float(result.response_window_frequency) == exact(3 / 5)
    assert float(result.response_window_frequency) != float(joint.marginal_frequency[1])
    assert float(result.response_window_frequency) != float(joint.joint_probability)


def test_anchor_month_groups_a_window_that_crosses_into_the_next_year() -> None:
    """A December anchor keeps its December group even though the response lies in January."""
    given = np.zeros(13, dtype=bool)
    given[11] = True  # December 2000
    target = np.zeros(13, dtype=bool)
    target[12] = True  # January 2001
    aligned = align(np.array([given, target]))
    with pytest.warns(ClimateIndicesWarning):
        result = conditional(aligned, lag_window=(1, 1), groupby="month")
    assert result.n_anchors.tolist() == [1] * 12
    assert int(result.n_positive[11]) == 1
    assert int(result.n_success[11]) == 1
    assert float(result.conditional_probability[11]) == 1.0
    assert int(result.n_positive[0]) == 0
    assert math.isnan(float(result.conditional_probability[0]))
    assert result.metadata["lag_units"] == "calendar months"


def test_daily_window_steps_are_civil_days_across_a_leap_day() -> None:
    """One step from 29 February is 1 March; no 30 February step is invented."""
    given = np.array([False, True, False, False])
    target = np.array([False, False, True, False])
    hazard = np.array([given, target])
    aligned = align(hazard, cadence="daily", start="2000-02-28")
    with pytest.warns(ClimateIndicesWarning):
        result = conditional(aligned, lag_window=(1, 1), groupby="month")
    assert int(result.n_anchors.sum()) == 3
    assert result.n_anchors.tolist() == [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert int(result.n_positive[1]) == 1
    assert int(result.n_success[1]) == 1
    assert float(result.conditional_probability[1]) == 1.0
    assert int(result.n_positive[2]) == 0
    assert result.metadata["lag_units"] == "civil days"


def test_conditional_short_support_keeps_counts_and_nans_the_estimates() -> None:
    """Below min_samples the counts survive and every estimate becomes NaN."""
    hazard = np.array([[True, True, False], [True, False, True]])
    aligned = align(hazard)
    with pytest.warns(ClimateIndicesWarning, match="min_samples=30"):
        strict = conditional(aligned, lag_window=(0, 1), min_samples=30)
    assert int(strict.n_anchors) == 2
    assert int(strict.n_positive) == 2
    assert int(strict.n_success) == 2
    assert math.isnan(float(strict.conditional_probability))
    assert math.isnan(float(strict.response_window_frequency))
    assert int(strict.quality_flags) == 1

    relaxed = conditional(align(hazard), lag_window=(0, 1), min_samples=2)
    assert int(relaxed.quality_flags) == 0
    assert float(relaxed.conditional_probability) == 1.0
    assert float(relaxed.response_window_frequency) == exact(2 / 2)


def test_zero_conditioning_count_is_undefined_with_its_flag() -> None:
    """A given hazard that never occurs leaves the conditional probability undefined, not zero."""
    hazard = np.array([[False, False], [True, True]])
    aligned = align(hazard)
    with pytest.warns(ClimateIndicesWarning, match="no conditioning occurrence"):
        result = conditional(aligned)
    assert int(result.n_anchors) == 2
    assert int(result.n_positive) == 0
    assert int(result.n_success) == 0
    assert math.isnan(float(result.conditional_probability))
    assert float(result.response_window_frequency) == 1.0
    assert int(result.quality_flags) == 2


def test_conditional_all_missing_cell_returns_zero_counts_without_failing_its_neighbor() -> None:
    """An all-missing grid cell yields zero counts and flags while the observed cell is unaffected."""
    hazard = np.array(
        [
            [[True, False], [False, False], [True, False], [False, False], [True, False]],
            [[False, False], [True, False], [False, False], [True, False], [False, False]],
        ]
    )
    valid = np.ones(hazard.shape, dtype=bool)
    valid[:, :, 1] = False
    aligned = align(hazard, valid)
    with pytest.warns(ClimateIndicesWarning, match="observed calendar year"):
        result = conditional(aligned, lag_window=(0, 1))
    assert result.n_anchors.tolist() == [4, 0]
    assert result.n_positive.tolist() == [2, 0]
    assert result.n_success.tolist() == [2, 0]
    assert float(result.conditional_probability[0]) == 1.0
    assert math.isnan(float(result.conditional_probability[1]))
    assert result.quality_flags.tolist() == [0, 3]


def test_only_the_two_declared_sources_enter_eligibility() -> None:
    """A third source's validity does not gate the selected pair; joint_statistics would gate on it."""
    hazard = np.array(
        [
            [True, False, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    valid = np.ones(hazard.shape, dtype=bool)
    valid[2] = False  # the third source is entirely unknown
    aligned = align(hazard, valid)
    with pytest.warns(ClimateIndicesWarning):
        result = conditional(aligned)
    with pytest.warns(ClimateIndicesWarning):
        joint = compound.joint_statistics(aligned, groupby="pooled", min_samples=1)
    assert int(result.n_anchors) == 4
    assert int(result.n_positive) == 2
    assert int(result.n_success) == 1
    assert int(joint.n_valid) == 0


def test_lag_window_at_or_past_the_record_end_returns_no_eligible_anchor() -> None:
    """An out-of-range window is empty, not an unbounded scan or an overflowed index."""
    hazard = np.array([[True, False, True, False], [True, True, False, False]])
    for lag_window in ((0, 4), (0, 10**9), (2**63 - 1, 2**63 - 1), (5, 9)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = conditional(align(hazard), lag_window=lag_window)
        assert int(result.n_anchors) == 0
        assert int(result.n_positive) == 0
        assert int(result.quality_flags) == 3
        assert math.isnan(float(result.conditional_probability))
        assert math.isnan(float(result.response_window_frequency))


def test_rejects_a_time_vector_with_a_missing_cadence_step() -> None:
    """A gap would turn one lag step into a longer interval, so a gapped vector is rejected."""
    hazard = np.array([[True, False, True], [False, True, True]])
    daily = align(hazard, cadence="daily", start="2000-01-01")
    daily_gap = dataclasses.replace(
        daily, time=np.array(["2000-01-01", "2000-01-03", "2000-01-04"], dtype="datetime64[D]")
    )
    with pytest.raises(CoordinateValidationError, match="complete cadence grid"):
        conditional(daily_gap)
    monthly_gap = dataclasses.replace(
        align(hazard), time=np.array(["2000-01-01", "2000-03-01", "2000-04-01"], dtype="datetime64[D]")
    )
    with pytest.raises(CoordinateValidationError, match="complete cadence grid"):
        conditional(monthly_gap)


def test_joint_statistics_accepts_a_gapped_time_vector() -> None:
    """Pointwise joint counts never read adjacent dates as lag steps, so a gap is not an error."""
    hazard = np.array([[True, False, True], [False, True, True]])
    gapped = dataclasses.replace(
        align(hazard), time=np.array(["2000-01-01", "2000-03-01", "2000-04-01"], dtype="datetime64[D]")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ClimateIndicesWarning)
        result = compound.joint_statistics(gapped, groupby="pooled", min_samples=1)
    assert int(result.n_valid) == 3
    assert int(result.n_joint) == 1


@pytest.mark.parametrize("window", [(0, 0), (1, 3), (0, 7), (2, 9), (0, 10**9)])
def test_forward_windows_match_a_per_anchor_scan(window: tuple[int, int]) -> None:
    """The rolling-window count equals a direct scan of [t+low, t+high], and an incomplete horizon is invalid."""
    rng = np.random.default_rng(1388)
    valid = rng.random((9, 2, 3)) > 0.2
    occurrence = (rng.random((9, 2, 3)) > 0.6) & valid
    low, high = window
    window_valid, response = statistics_module._forward_windows(valid, occurrence, low, high)

    expected_valid = np.zeros(valid.shape, dtype=bool)
    expected_response = np.zeros(valid.shape, dtype=bool)
    for anchor in range(valid.shape[0]):
        if anchor + high < valid.shape[0]:
            expected_valid[anchor] = valid[anchor + low : anchor + high + 1].all(axis=0)
            expected_response[anchor] = occurrence[anchor + low : anchor + high + 1].any(axis=0)
    np.testing.assert_array_equal(window_valid, expected_valid)
    np.testing.assert_array_equal(response, expected_response)


def test_production_defaults_group_by_month_and_require_thirty_anchors() -> None:
    """The documented defaults are calendar-month grouping and min_samples=30."""
    hazard = np.ones((2, 6), dtype=bool)
    aligned = align(hazard)
    with pytest.warns(ClimateIndicesWarning, match="min_samples=30"):
        result = compound.conditional_probability(aligned, given="source_0", target="source_1")
    assert result.groupby == "month"
    assert result.lag_window == (0, 0)
    assert result.n_anchors.shape == (12,)
    assert int(result.n_anchors.sum()) == 6
    assert int(result.n_positive[0]) == 1
    assert math.isnan(float(result.conditional_probability[0]))
    assert int(result.quality_flags[0]) == 1


def test_conditional_warns_about_sparse_conditioning() -> None:
    """Fewer than ten conditioning occurrences warns on that count, once per group."""
    hazard = np.array([[True, False, True, False], [True, True, False, False]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conditional(align(hazard))
    sparse = [str(item.message) for item in caught if "conditioning occurrences" in str(item.message)]
    assert len(sparse) == 1
    assert all(issubclass(item.category, ClimateIndicesWarning) for item in caught)


@given(rows=two_source_rows())
def test_conditional_counts_nest_and_probabilities_stay_within_their_bounds(
    rows: tuple[list[bool], list[bool], list[bool], list[bool]],
) -> None:
    """Counts nest on one eligible support and every frequency lies in [0, 1]."""
    first, second, first_valid, second_valid = rows
    hazard = np.array([first, second], dtype=bool)
    valid = np.array([first_valid, second_valid], dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = conditional(align(hazard, valid), lag_window=(0, 1))
    n_anchors = int(result.n_anchors)
    n_positive = int(result.n_positive)
    n_response = int(result.n_response)
    n_success = int(result.n_success)
    assert 0 <= n_success <= min(n_positive, n_response) <= n_anchors
    for probability in (float(result.conditional_probability), float(result.response_window_frequency)):
        assert math.isnan(probability) or 0.0 <= probability <= 1.0
    # the denominators are the anchor counts, not the eligible support
    if n_positive:
        assert float(result.conditional_probability) == exact(n_success / n_positive)
    else:
        assert math.isnan(float(result.conditional_probability))
    if n_anchors:
        assert float(result.response_window_frequency) == exact(n_response / n_anchors)
    else:
        assert math.isnan(float(result.response_window_frequency))


def test_conditional_result_arrays_are_read_only_copied_and_frozen() -> None:
    """Freeze the arrays and the instance; an incomplete or a shared record is not a result."""
    hazard = np.array([[True, False, True], [False, True, True]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = conditional(align(hazard))
    for array in (
        result.n_anchors,
        result.n_positive,
        result.n_response,
        result.n_success,
        result.conditional_probability,
        result.response_window_frequency,
        result.quality_flags,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = array.flat[0]
    assert result.n_anchors.dtype == np.int64
    assert result.n_positive.dtype == np.int64
    assert result.n_response.dtype == np.int64
    assert result.n_success.dtype == np.int64
    assert result.quality_flags.dtype == np.int64
    assert result.conditional_probability.dtype == np.float64
    assert result.response_window_frequency.dtype == np.float64
    assert not np.shares_memory(result.n_success, hazard)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.given = "source_1"  # type: ignore[misc]


def test_manifest_serializes_with_the_pairing_window_and_harmonization_record() -> None:
    """Round-trip the manifest through JSON and keep the pairing and window reachable."""
    hazard = np.array([[True, False, True, False], [True, True, False, False]])
    aligned = align(hazard, metadata={"calendar": "proleptic_gregorian", "cadence": "monthly"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = conditional(aligned, given="source_1", target="source_0", lag_window=(0, 1), min_samples=2)
    manifest = json.loads(json.dumps(result.metadata, allow_nan=False))
    assert manifest["schema_version"] == 1
    assert manifest["function"] == "conditional_probability"
    assert manifest["given"] == "source_1"
    assert manifest["target"] == "source_0"
    assert manifest["lag_window"] == [0, 1]
    assert manifest["groupby"] == "pooled"
    assert manifest["min_samples"] == 2
    assert manifest["calendar"] == "proleptic_gregorian"
    assert manifest["n_steps"] == 4
    assert manifest["quality_flags"]["undefined_denominator"] == 2
    assert "every target step in the inclusive forward window is valid" in manifest["eligibility"]
    assert "not an event-onset probability" in manifest["estimands"]
    assert "calendar month groups a lagged statistic" in manifest["lag_rule"]
    assert manifest["harmonization"]["cadence"] == "monthly"
    assert result.metadata["grouped_by"] == "pooled over every eligible anchor"


def test_month_groups_follow_the_anchor_step() -> None:
    """Monthly grouping reports one value per anchor month, not per response month."""
    given = np.zeros(12, dtype=bool)
    given[0] = True
    target = np.zeros(12, dtype=bool)
    target[1] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = conditional(align(np.array([given, target])), lag_window=(1, 1), groupby="month")
    assert result.n_anchors.shape == (12,)
    assert result.n_anchors.tolist() == [1] * 11 + [0]
    assert int(result.n_success[0]) == 1
    assert float(result.conditional_probability[0]) == 1.0
    assert int(result.n_positive[1]) == 0
    assert result.metadata["grouped_by"] == "month"


@pytest.mark.parametrize(
    ("given", "target", "message"),
    [
        ("source_0", "source_0", "different sources"),
        ("source_2", "source_1", "must name one source"),
        ("", "source_1", "must name one source"),
        (None, "source_1", "must name one source"),
        ("source_0", 1, "must name one source"),
    ],
)
def test_rejects_ambiguous_or_missing_source_roles(given: Any, target: Any, message: str) -> None:
    """Require two distinct existing names instead of guessing at the roles."""
    aligned = align(np.array([[True, False, True], [False, True, True]]))
    with pytest.raises(InvalidArgumentError, match=message):
        compound.conditional_probability(aligned, given=given, target=target)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("lag_window", "message"),
    [
        ((0, -1), "ordered from low to high"),
        ((-1, 0), "cannot be negative"),
        ((2, 1), "ordered from low to high"),
        ((0, 1, 2), "tuple of cadence-step offsets"),
        (0, "tuple of cadence-step offsets"),
        (["0", "1"], "tuple of cadence-step offsets"),
        ((0, 1.5), "must be integers"),
        ((True, 1), "must be integers"),
    ],
)
def test_rejects_invalid_lag_windows(lag_window: Any, message: str) -> None:
    """The window is forward and ordered, so a negative or reversed request is rejected."""
    aligned = align(np.array([[True, False, True], [False, True, True]]))
    with pytest.raises(InvalidArgumentError, match=message):
        compound.conditional_probability(aligned, given="source_0", target="source_1", lag_window=lag_window)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("groupby", "seasonal", "groupby must be one of"),
        ("min_samples", 0, "positive integer"),
        ("min_samples", True, "positive integer"),
    ],
)
def test_conditional_rejects_invalid_options(argument: str, value: Any, message: str) -> None:
    """Validate the shared options before resolving the window."""
    aligned = align(np.array([[True, False, True], [False, True, True]]))
    with pytest.raises(InvalidArgumentError, match=message):
        compound.conditional_probability(aligned, given="source_0", target="source_1", **{argument: value})


def test_conditional_lifecycle_events_are_emitted_on_success() -> None:
    """Emit ``calculation_started`` then ``calculation_completed`` with the group shape."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger
    aligned = align(np.array([[True, False, True, True], [False, True, True, False]]))
    with (
        mock.patch.object(statistics_module, "_logger", mock_logger),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        conditional(aligned, groupby="month")
    bound = mock_logger.bind.call_args.kwargs
    assert bound["index_type"] == "compound_conditional_probability"
    assert bound["n_sources"] == 2  # names are caller identifiers and stay out of the log
    assert bound["lag_window"] == (0, 0)
    assert "given" not in bound
    assert "target" not in bound
    assert [call.args[0] for call in mock_logger.info.call_args_list] == [
        "calculation_started",
        "calculation_completed",
    ]
    assert mock_logger.info.call_args_list[1].kwargs["output_shape"] == (12,)
    mock_logger.error.assert_not_called()


def test_conditional_failure_is_logged_and_propagates() -> None:
    """Emit ``calculation_failed`` for a rejected window and re-raise it unchanged."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger
    aligned = align(np.array([[True, False, True], [False, True, True]]))
    with (
        mock.patch.object(statistics_module, "_logger", mock_logger),
        pytest.raises(InvalidArgumentError, match="ordered from low to high"),
    ):
        conditional(aligned, lag_window=(1, 0))
    mock_logger.info.assert_called_once_with("calculation_started")
    failed = [call for call in mock_logger.error.call_args_list if call.args[0] == "calculation_failed"]
    assert len(failed) == 1
    assert failed[0].kwargs["error_type"] == "InvalidArgumentError"


def test_conditional_public_surface_is_explicit() -> None:
    """Expose the function and its result type, and never from the package root."""
    assert "conditional_probability" in compound.__all__
    assert "ConditionalResult" in compound.__all__
    assert not hasattr(climate_indices, "conditional_probability")

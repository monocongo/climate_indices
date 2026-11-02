"""Contract tests for sustained reliability metrics and their shared accumulators."""

import dataclasses
import json
import math
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from climate_indices import reliability
from climate_indices.exceptions import ClimateIndicesWarning, InvalidArgumentError, OverlappingEventsWarning

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixture/reliability/events.json"


def record(**changes: object) -> dict[str, object]:
    return {
        "event_id": "a",
        "start_time": "2024-01-01T00:00:00",
        "duration": 10.0,
        "customers_interrupted": 5,
        **changes,
    }


@pytest.mark.parametrize(
    "result",
    [
        reliability.ReliabilityResult(0.3, 30, 2),
        reliability.ReliabilityResult(saifi=0.3, customer_interruptions=30, outage_count=2),
    ],
)
def test_result_preserves_original_constructor(result: reliability.ReliabilityResult) -> None:
    assert result.saifi == pytest.approx(0.3)
    assert result.customer_interruptions == 30
    assert result.outage_count == 2
    assert np.isnan(result.saidi)
    assert np.isnan(result.customer_minutes_interrupted)
    assert math.isnan(result.caidi)
    assert result.maifi_e == 0.0
    assert result.maifi is None
    assert result.asai is None
    assert result.period_hours is None
    assert result.undefined_metrics == ("caidi", "maifi", "asai")


def test_legacy_three_field_results_compare_equal_and_hash_equal() -> None:
    first = reliability.ReliabilityResult(0.0, 0, 0)
    second = reliability.ReliabilityResult(0.0, 0, 0)
    assert first == second
    assert hash(first) == hash(second)


def test_legacy_result_equals_equivalent_zero_duration_result() -> None:
    legacy = reliability.ReliabilityResult(0.0, 0, 0)
    empty = reliability.reliability_indices([], total_customers=1000)
    assert legacy == empty
    assert hash(legacy) == hash(empty)


def test_unset_duration_with_positive_interruptions_is_not_zero_duration() -> None:
    unset = reliability.ReliabilityResult(saifi=0.1, customer_interruptions=10, outage_count=1)
    zero = reliability.ReliabilityResult(
        saifi=0.1, saidi=0.0, customer_interruptions=10, customer_minutes_interrupted=0.0, outage_count=1
    )
    assert unset != zero
    assert math.isnan(unset.caidi)
    assert unset.undefined_metrics == ("caidi", "maifi", "asai")
    assert zero.caidi == 0.0
    assert zero.undefined_metrics == ("maifi", "asai")


def test_unset_saidi_does_not_distinguish_equal_caidi() -> None:
    unset = reliability.ReliabilityResult(
        saifi=0.1, customer_interruptions=10, customer_minutes_interrupted=5.0, outage_count=1
    )
    zero = reliability.ReliabilityResult(
        saifi=0.1, saidi=0.0, customer_interruptions=10, customer_minutes_interrupted=5.0, outage_count=1
    )
    assert unset == zero
    assert hash(unset) == hash(zero)
    assert unset.caidi == zero.caidi == 0.5


def test_nan_saifi_is_not_normalized() -> None:
    unset_saifi = reliability.ReliabilityResult(math.nan, 0, 0)
    zero_saifi = reliability.ReliabilityResult(0.0, 0, 0)
    assert unset_saifi != zero_saifi


def test_populated_duration_metrics_distinguish_results() -> None:
    base = reliability.ReliabilityResult(
        saifi=0.3, saidi=24.0, customer_interruptions=30, customer_minutes_interrupted=2400.0, outage_count=2
    )
    other_saidi = reliability.ReliabilityResult(
        saifi=0.3, saidi=48.0, customer_interruptions=30, customer_minutes_interrupted=2400.0, outage_count=2
    )
    other_cmi = reliability.ReliabilityResult(
        saifi=0.3, saidi=24.0, customer_interruptions=30, customer_minutes_interrupted=4800.0, outage_count=2
    )
    assert base != other_saidi
    assert base != other_cmi


def test_nominal_analytic_case() -> None:
    events = [
        record(event_id="a", duration=60.0, customers_interrupted=20),
        record(event_id="b", start_time="2024-01-01T02:00:00", duration=120.0, customers_interrupted=10),
    ]
    result = reliability.reliability_indices(events, total_customers=100)
    assert result.saifi == pytest.approx(0.3)
    assert result.saidi == pytest.approx(24.0)
    assert result.customer_interruptions == 30
    assert result.customer_minutes_interrupted == pytest.approx(2400.0)
    assert result.caidi == 80.0
    assert result.caidi == result.saidi / result.saifi
    assert result.maifi_e == 0.0
    assert result.maifi is None
    assert result.undefined_metrics == ("maifi", "asai")
    assert isinstance(result.caidi, float)
    assert result.outage_count == 2
    assert isinstance(result.saifi, float)
    assert isinstance(result.saidi, float)
    assert isinstance(result.customer_minutes_interrupted, float)
    assert isinstance(result.customer_interruptions, int)
    assert not isinstance(result.customer_interruptions, bool)
    assert isinstance(result.outage_count, int)
    assert not isinstance(result.outage_count, bool)


@pytest.mark.parametrize("events", [[], pd.DataFrame()])
def test_empty_log_returns_zero_without_infinity(events: Any) -> None:
    result = reliability.reliability_indices(events, total_customers=1000)
    assert result == reliability.ReliabilityResult(
        saifi=0.0, saidi=0.0, customer_interruptions=0, customer_minutes_interrupted=0.0, outage_count=0
    )


@pytest.mark.parametrize(
    "events",
    [[], {}, pd.DataFrame(), [record(duration=1.0)], [record(customers_interrupted=0)]],
    ids=["empty-rows", "empty-mapping", "empty-frame", "momentary-only", "zero-customers"],
)
def test_caidi_is_undefined_without_customer_interruptions(events: Any) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reliability.reliability_indices(events, total_customers=100)
        assert math.isnan(result.caidi)
        assert result.undefined_metrics == ("caidi", "maifi", "asai")
        assert len(caught) == 0


def test_dataframe_and_mapping_inputs_agree() -> None:
    rows = [
        record(event_id="a", duration=60.0, customers_interrupted=20),
        record(event_id="b", start_time="2024-01-01T02:00:00", duration=60.0, customers_interrupted=10),
    ]
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    from_rows = reliability.reliability_indices(rows, total_customers=100)
    from_frame = reliability.reliability_indices(pd.DataFrame(rows), total_customers=100)
    from_mapping = reliability.reliability_indices(columns, total_customers=100)
    assert from_rows == from_frame == from_mapping


def test_momentary_rows_contribute_nothing_to_sustained_metrics() -> None:
    sustained = record(event_id="sustained", duration=60.0, customers_interrupted=20)
    momentary = record(
        event_id="momentary",
        start_time="2024-01-01T01:00:00",
        duration=4.0,
        customers_interrupted=7,
        interruption_type="momentary",
    )
    without = reliability.reliability_indices([sustained], total_customers=100)
    with_momentary = reliability.reliability_indices([sustained, momentary], total_customers=100)
    assert with_momentary.saifi == without.saifi
    assert with_momentary.saidi == without.saidi
    assert with_momentary.caidi == without.caidi
    assert with_momentary.momentary_outage_count == 1
    assert without.momentary_outage_count == 0
    assert with_momentary.customer_interruptions == 20
    assert with_momentary.outage_count == 1
    assert without.maifi_e == 0.0
    assert with_momentary.maifi_e == pytest.approx(0.07)


@pytest.mark.parametrize("type_fields", [{}, {"interruption_type": None}])
def test_grouped_momentary_event_without_explicit_type_uses_momentary_metrics(type_fields: dict[str, object]) -> None:
    row = record(duration=60.0, grouped_momentary_event=True, device_operations=2, **type_fields)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ClimateIndicesWarning)
        result = reliability.reliability_indices([row], total_customers=100)
    assert result.saifi == 0.0
    assert result.saidi == 0.0
    assert result.customer_interruptions == 0
    assert result.customer_minutes_interrupted == 0.0
    assert result.outage_count == 0
    assert result.maifi_e == pytest.approx(0.05)
    assert result.maifi == pytest.approx(0.1)


def test_explicitly_momentary_long_row_is_excluded_with_one_warning() -> None:
    row = record(duration=60.0, customers_interrupted=50, interruption_type="momentary")
    with pytest.warns(ClimateIndicesWarning, match="interruption_type") as caught:
        result = reliability.reliability_indices([row], total_customers=100)
    assert len(caught) == 1
    assert result == reliability.ReliabilityResult(
        saifi=0.0,
        saidi=0.0,
        customer_interruptions=0,
        customer_minutes_interrupted=0.0,
        outage_count=0,
        maifi_e=0.5,
        momentary_outage_count=1,
    )


def test_planned_rows_are_included_by_default_and_excluded_on_request() -> None:
    rows = [
        record(event_id="unplanned", duration=60.0, customers_interrupted=10),
        record(
            event_id="planned",
            start_time="2024-01-01T01:00:00",
            duration=60.0,
            customers_interrupted=20,
            planned=True,
        ),
    ]
    included = reliability.reliability_indices(rows, total_customers=100)
    excluded = reliability.reliability_indices(rows, total_customers=100, exclude_planned=True)
    assert included.customer_interruptions == 30
    assert included.outage_count == 2
    assert included.saifi == pytest.approx(0.3)
    assert excluded.customer_interruptions == 10
    assert excluded.outage_count == 1
    assert excluded.saifi == pytest.approx(0.1)


def test_adr_0026_fixture_analytic_case() -> None:
    events = json.loads(FIXTURE.read_text())
    result = reliability.reliability_indices(events, total_customers=1000, period_hours=24)
    assert result.asai == pytest.approx(float(Fraction(59, 60)), rel=0, abs=1e-15)
    assert result.saifi == pytest.approx(0.3)
    assert result.saidi == pytest.approx(24.0)
    assert result.customer_interruptions == 300
    assert result.customer_minutes_interrupted == pytest.approx(24000.0)
    assert result.outage_count == 2
    assert result.momentary_outage_count == 3


def test_zero_duration_event_counts_only_when_explicitly_sustained() -> None:
    derived = reliability.reliability_indices([record(duration=0.0, customers_interrupted=5)], total_customers=100)
    assert derived == reliability.ReliabilityResult(
        saifi=0.0,
        saidi=0.0,
        customer_interruptions=0,
        customer_minutes_interrupted=0.0,
        outage_count=0,
        maifi_e=0.05,
        momentary_outage_count=1,
    )
    explicit = record(duration=0.0, customers_interrupted=5, interruption_type="sustained")
    with pytest.warns(ClimateIndicesWarning, match="interruption_type") as caught:
        counted = reliability.reliability_indices([explicit], total_customers=100)
    assert len(caught) == 1
    assert counted.saifi == pytest.approx(0.05)
    assert counted.saidi == pytest.approx(0.0)
    assert counted.customer_interruptions == 5
    assert counted.customer_minutes_interrupted == pytest.approx(0.0)
    assert counted.caidi == 0.0
    assert counted.undefined_metrics == ("maifi", "asai")
    assert counted.outage_count == 1


def test_threshold_minutes_changes_classification() -> None:
    row = record(duration=3.0, customers_interrupted=5)
    assert reliability.reliability_indices([row], total_customers=100).customer_interruptions == 0
    assert (
        reliability.reliability_indices([row], total_customers=100, threshold_minutes=3.0).customer_interruptions == 0
    )
    counted = reliability.reliability_indices([row], total_customers=100, threshold_minutes=1.0)
    assert counted.customer_interruptions == 5
    assert counted.outage_count == 1


def test_splitting_one_events_customers_preserves_saifi() -> None:
    """Two rows that split one event's customers sum to the original count.

    Issue #1338 words the invariant as two rows "with the same N_i". Two rows
    with an equal nonzero N_i double the sum, so this test splits the event's
    customers into disjoint subsets whose counts sum to the original N_i. The
    library does not group rows or de-duplicate customer sets: the event row is
    the event, and event_id must be unique.
    """
    whole = [record(event_id="whole", duration=60.0, customers_interrupted=30)]
    split = [
        record(event_id="part-a", duration=60.0, customers_interrupted=20),
        record(event_id="part-b", start_time="2024-01-01T02:00:00", duration=60.0, customers_interrupted=10),
    ]
    whole_result = reliability.reliability_indices(whole, total_customers=100)
    split_result = reliability.reliability_indices(split, total_customers=100)
    assert whole_result.saifi == pytest.approx(split_result.saifi)
    assert whole_result.customer_interruptions == split_result.customer_interruptions == 30
    assert split_result.outage_count == 2


def test_scaling_total_customers_scales_saifi_inversely() -> None:
    events = [record(duration=60.0, customers_interrupted=30)]
    base = reliability.reliability_indices(events, total_customers=100)
    scaled = reliability.reliability_indices(events, total_customers=300)
    assert scaled.saifi == pytest.approx(base.saifi / 3)


@pytest.mark.parametrize("total", [np.float32(100), np.longdouble(100), Fraction(100, 1)])
def test_accepted_denominator_types_return_python_float(total: Any) -> None:
    events = [record(duration=60.0, customers_interrupted=20)]
    result = reliability.reliability_indices(events, total_customers=total)
    assert isinstance(result.saifi, float)
    assert result.saifi == pytest.approx(0.2)


def test_fraction_denominator_must_be_exact() -> None:
    events = [record(duration=60.0, customers_interrupted=2**53 + 4)]
    denominator: Any = Fraction(2**53 + 3, 1)
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices(events, total_customers=denominator)
    assert exc.value.argument_name == "total_customers"


def test_none_event_log_is_rejected() -> None:
    events: Any = None
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices(events, total_customers=100)
    assert exc.value.argument_name == "events"


def test_fractional_total_customers_is_accepted() -> None:
    events = [record(duration=60.0, customers_interrupted=20)]
    result = reliability.reliability_indices(events, total_customers=100.5)
    assert result.saifi == pytest.approx(20 / 100.5)
    assert isinstance(result.saifi, float)


def test_fractional_extended_precision_total_customers_is_accepted() -> None:
    events = [record(duration=60.0, customers_interrupted=20)]
    total: Any = np.longdouble("100.1")
    result = reliability.reliability_indices(events, total_customers=total)
    assert result.saifi == pytest.approx(20 / 100.1)
    assert isinstance(result.saifi, float)


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
    reason="platform NumPy longdouble is float64, so 2**53 + 1 is representable",
)
def test_integral_extended_precision_total_customers_must_be_exact() -> None:
    events = [record(duration=60.0, customers_interrupted=20)]
    total: Any = np.longdouble(2**53 + 1)
    assert total != float(total)
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices(events, total_customers=total)
    assert exc.value.argument_name == "total_customers"


def test_row_order_does_not_change_the_result() -> None:
    rows = [
        record(event_id="a", duration=60.0, customers_interrupted=20),
        record(event_id="b", start_time="2024-01-01T02:00:00", duration=30.0, customers_interrupted=10, planned=True),
        record(event_id="c", start_time="2024-01-01T04:00:00", duration=90.0, customers_interrupted=5),
    ]
    forward = reliability.reliability_indices(rows, total_customers=100)
    backward = reliability.reliability_indices(list(reversed(rows)), total_customers=100)
    assert forward == backward


def test_caidi_identity_seed_matches_the_adr_table() -> None:
    events = json.loads(FIXTURE.read_text())
    frame = reliability.classify_interruptions(events)
    sustained = frame[frame["interruption_type"] == "sustained"]
    customer_minutes = float((sustained["duration"] * sustained["customers_interrupted"]).sum())
    customer_interruptions = sum(int(count) for count in sustained["customers_interrupted"].tolist())
    result = reliability.reliability_indices(events, total_customers=1000)
    assert result.customer_interruptions == customer_interruptions == 300
    assert customer_minutes == 24000
    assert result.saifi == pytest.approx(0.3)
    assert customer_minutes / 1000 == pytest.approx(24.0)
    assert customer_minutes / customer_interruptions == pytest.approx(80.0)


@pytest.mark.parametrize("total", [0, -1, np.nan, np.inf, True, "100", np.int64(2**53 + 3), np.uint64(2**53 + 1)])
def test_invalid_total_customers(total: Any) -> None:
    events = [record()]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices(events, total_customers=total)
    assert exc.value.argument_name == "total_customers"


def test_missing_total_customers_names_the_argument() -> None:
    events = [record()]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices(events)
    assert exc.value.argument_name == "total_customers"
    assert "supply the positive average served-customer count" in str(exc.value)


@pytest.mark.parametrize("value", ["false", 1, 0, None, [], np.bool_(True)])
def test_invalid_exclude_planned(value: Any) -> None:
    events = [record()]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices(events, total_customers=100, exclude_planned=value)
    assert exc.value.argument_name == "exclude_planned"


def test_numpy_boolean_planned_values_are_accepted() -> None:
    row = record(duration=60.0, customers_interrupted=5, planned=np.bool_(True))
    assert reliability.reliability_indices([row], total_customers=100).customer_interruptions == 5
    assert reliability.reliability_indices([row], total_customers=100, exclude_planned=True).saifi == 0.0


def test_negative_customer_count_raises() -> None:
    events = [record(customers_interrupted=-1)]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices(events, total_customers=100)
    assert exc.value.argument_name == "customers_interrupted"


def test_event_count_above_total_customers_raises() -> None:
    events = [record(customers_interrupted=11)]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices(events, total_customers=10)
    assert exc.value.argument_name == "customers_interrupted"


def test_aggregate_customer_interruptions_beyond_int64_stay_exact() -> None:
    huge = 2**62
    events = [
        record(event_id="a", duration=60.0, customers_interrupted=huge),
        record(event_id="b", start_time="2024-01-01T02:00:00", duration=60.0, customers_interrupted=huge),
    ]
    result = reliability.reliability_indices(events, total_customers=float(huge))
    assert result.customer_interruptions == 2**63
    assert result.saifi == pytest.approx(2.0)
    assert result.caidi == 60.0


def test_single_sustained_event_shares_the_saidi_relationship() -> None:
    events = [record(event_id="only", duration=30.0, customers_interrupted=10)]
    result = reliability.reliability_indices(events, total_customers=100)
    assert result.customer_minutes_interrupted == pytest.approx(300.0)
    assert result.saidi == pytest.approx(3.0)
    assert result.customer_minutes_interrupted == pytest.approx(result.saidi * 100)


def test_customer_minutes_relationship_to_caidi() -> None:
    """CMI equals SAIDI times N_T and CAIDI times CI on the same record."""
    events = json.loads(FIXTURE.read_text())
    result = reliability.reliability_indices(events, total_customers=1000)
    assert result.customer_minutes_interrupted == pytest.approx(result.saidi * 1000)
    assert result.customer_minutes_interrupted == pytest.approx(result.caidi * result.customer_interruptions)
    assert result.caidi == 80.0
    assert result.saidi == pytest.approx(24.0)
    assert result.saidi / 60 == pytest.approx(0.4)


@pytest.mark.parametrize("duration", [6.0, 17.25, 120.0])
def test_caidi_single_event_is_its_restoration_time(duration: float) -> None:
    result = reliability.reliability_indices([record(duration=duration, customers_interrupted=3)], total_customers=1000)
    assert result.caidi == duration
    assert result.undefined_metrics == ("maifi", "asai")


@pytest.mark.parametrize("total_customers", [30.0, 100.0, 100.5, 1000.0])
@pytest.mark.parametrize("durations, counts", [([6.0], [3]), ([60.0, 120.0], [20, 10]), ([6.1, 17.3], [7, 11])])
def test_caidi_population_independence_and_general_identity(
    total_customers: float, durations: list[float], counts: list[int]
) -> None:
    rows = [
        record(event_id=i, start_time=f"2024-01-0{i + 1}T00:00:00", duration=duration, customers_interrupted=count)
        for i, (duration, count) in enumerate(zip(durations, counts, strict=True))
    ]
    result = reliability.reliability_indices(rows, total_customers=total_customers)
    baseline = reliability.reliability_indices(rows, total_customers=100)
    reference = sum(Fraction(duration) * count for duration, count in zip(durations, counts, strict=True)) / sum(counts)
    assert result.caidi == baseline.caidi
    # CAIDI is the definitional accumulator ratio, so the analytic value is exact here.
    assert result.caidi == float(reference)
    # The identity with SAIDI / SAIFI is mathematical. SAIDI and SAIFI are rounded
    # separately, so four float64 epsilons bound the accumulation and divisions.
    tolerance = 4 * np.finfo(np.float64).eps
    np.testing.assert_allclose(result.caidi, result.saidi / result.saifi, rtol=tolerance, atol=0)


def test_caidi_uses_the_same_planned_and_threshold_filters() -> None:
    rows = [
        record(event_id="unplanned", duration=60.0, customers_interrupted=10),
        record(event_id="planned", start_time="2024-01-02", duration=120.0, customers_interrupted=20, planned=True),
        record(event_id="short", start_time="2024-01-03", duration=3.0, customers_interrupted=5),
    ]
    assert reliability.reliability_indices(rows, total_customers=100).caidi == 100.0
    assert reliability.reliability_indices(rows, total_customers=100, exclude_planned=True).caidi == 60.0
    included = reliability.reliability_indices(rows, total_customers=100, threshold_minutes=1.0)
    assert included.caidi == float(Fraction(3015, 35))
    empty = reliability.reliability_indices(rows[1:2], total_customers=100, exclude_planned=True)
    assert math.isnan(empty.caidi)
    assert empty.undefined_metrics == ("caidi", "maifi", "asai")


def test_exclude_planned_filters_saidi_and_customer_minutes() -> None:
    rows = [
        record(event_id="unplanned", duration=60.0, customers_interrupted=10),
        record(
            event_id="planned",
            start_time="2024-01-01T02:00:00",
            duration=60.0,
            customers_interrupted=20,
            planned=True,
        ),
    ]
    included = reliability.reliability_indices(rows, total_customers=100)
    excluded = reliability.reliability_indices(rows, total_customers=100, exclude_planned=True)
    assert included.customer_minutes_interrupted == pytest.approx(1800.0)
    assert included.saidi == pytest.approx(18.0)
    assert excluded.customer_minutes_interrupted == pytest.approx(600.0)
    assert excluded.saidi == pytest.approx(6.0)


def test_threshold_minutes_filters_saidi_and_customer_minutes() -> None:
    rows = [record(duration=3.0, customers_interrupted=5)]
    excluded = reliability.reliability_indices(rows, total_customers=100)
    included = reliability.reliability_indices(rows, total_customers=100, threshold_minutes=1.0)
    assert excluded.customer_minutes_interrupted == 0.0
    assert excluded.saidi == 0.0
    assert included.customer_minutes_interrupted == pytest.approx(15.0)
    assert included.saidi == pytest.approx(0.15)


def test_saidi_is_linear_in_event_durations() -> None:
    rows = [
        record(event_id="a", duration=60.0, customers_interrupted=20),
        record(event_id="b", start_time="2024-01-01T02:00:00", duration=30.0, customers_interrupted=10),
    ]
    base = reliability.reliability_indices(rows, total_customers=100)
    scaled = reliability.reliability_indices(
        [dict(row, duration=cast(float, row["duration"]) * 2) for row in rows], total_customers=100
    )
    assert scaled.customer_minutes_interrupted == pytest.approx(2 * base.customer_minutes_interrupted)
    assert scaled.saidi == pytest.approx(2 * base.saidi)
    assert scaled.customer_interruptions == base.customer_interruptions


def test_saidi_is_additive_across_disjoint_event_subsets() -> None:
    rows = [
        record(event_id="a", duration=60.0, customers_interrupted=20),
        record(event_id="b", start_time="2024-01-01T02:00:00", duration=30.0, customers_interrupted=10),
        record(event_id="c", start_time="2024-01-01T04:00:00", duration=90.0, customers_interrupted=5),
    ]
    combined = reliability.reliability_indices(rows, total_customers=100)
    first = reliability.reliability_indices(rows[:2], total_customers=100)
    second = reliability.reliability_indices(rows[2:], total_customers=100)
    assert combined.customer_minutes_interrupted == pytest.approx(
        first.customer_minutes_interrupted + second.customer_minutes_interrupted
    )
    assert combined.saidi == pytest.approx(first.saidi + second.saidi)


def test_step_restoration_rows_accumulate_without_collapsing() -> None:
    """Two restoration stages sum their own ``r_i * N_i`` contributions.

    ``_normalize_events`` requires unique ``event_id`` values, so two stages
    cannot share one. Each row is its own event. Both stages start together, so
    their positive-duration intervals overlap and the library emits one
    ``OverlappingEventsWarning`` rather than grouping or collapsing the rows.
    """
    rows = [
        record(event_id="stage-a", duration=40.0, customers_interrupted=10),
        record(event_id="stage-b", duration=70.0, customers_interrupted=5),
    ]
    with pytest.warns(OverlappingEventsWarning):
        result = reliability.reliability_indices(rows, total_customers=100)
    assert result.customer_minutes_interrupted == pytest.approx(750.0)
    assert result.saidi == pytest.approx(7.5)
    assert result.customer_interruptions == 15
    assert result.outage_count == 2


def test_missing_duration_and_end_time_raise_through_reliability_indices() -> None:
    row = record()
    del row["duration"]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices([row], total_customers=100)
    assert exc.value.argument_name == "duration"


@pytest.mark.parametrize("duration", [-1, np.nan])
def test_invalid_duration_raises_through_reliability_indices(duration: float) -> None:
    events = [record(duration=duration)]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices(events, total_customers=100)
    assert exc.value.argument_name == "duration"


@pytest.mark.parametrize("exclude_planned", [False, True])
def test_momentary_outage_count_tallies_rows_separately(exclude_planned: bool) -> None:
    rows = [
        record(event_id="sustained"),
        record(event_id="short", start_time="2024-01-02", duration=1.0, momentary_event_excluded=True),
        record(event_id="planned", start_time="2024-01-03", duration=2.0, planned=True),
        record(event_id="zero", start_time="2024-01-04", duration=0.0, customers_interrupted=0),
    ]
    result = reliability.reliability_indices(rows, total_customers=100, exclude_planned=exclude_planned)
    assert result.outage_count == 1
    # exclude_planned filters sustained rows only, so the momentary tally is unchanged.
    assert result.momentary_outage_count == 3
    assert isinstance(result.momentary_outage_count, int)
    assert result.saifi == 0.05
    assert result.saidi == 0.5


def test_asai_analytic_case_uses_customer_minutes_as_hours() -> None:
    rows = [
        record(event_id="a", duration=60.0, customers_interrupted=20),
        record(event_id="b", start_time="2024-01-01T02:00:00", duration=120.0, customers_interrupted=10),
    ]
    result = reliability.reliability_indices(rows, total_customers=100, period_hours=24)
    expected = float(Fraction(59, 60))
    assert result.asai == pytest.approx(expected, rel=0, abs=1e-15)
    assert result.asai == 1 - (result.saidi / 60) / 24
    assert result.period_hours == 24.0
    assert result.undefined_metrics == ("maifi",)


@pytest.mark.parametrize("events", [[], {}, pd.DataFrame(), [record(duration=1.0)]])
def test_asai_without_sustained_outages_is_one(events: Any) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reliability.reliability_indices(events, total_customers=100, period_hours=8760)
    assert result.asai == 1.0
    assert result.saifi == result.saidi == 0.0
    assert result.outage_count == 0
    assert result.momentary_outage_count == len(events)
    assert result.undefined_metrics == ("caidi", "maifi")
    assert not caught


def test_asai_without_period_is_none_and_does_not_infer_event_bounds() -> None:
    result = reliability.reliability_indices([record()], total_customers=100)
    assert result.asai is None
    assert result.period_hours is None
    assert result.undefined_metrics == ("maifi", "asai")


@pytest.mark.parametrize(
    "start, end, hours",
    [
        ("2023-01-01", "2024-01-01", 8760.0),
        ("2024-01-01", "2025-01-01", 8784.0),
        ("2024-01-01", "2024-01-31", 720.0),
        ("2024-03-10T00:00:00-05:00", "2024-03-11T00:00:00-04:00", 23.0),
    ],
)
def test_asai_uses_elapsed_hours_without_annualizing(start: str, end: str, hours: float) -> None:
    events = [record(start_time=start, duration=60.0, customers_interrupted=20)]
    derived = reliability.reliability_indices(events, total_customers=100, period_start=start, period_end=end)
    explicit = reliability.reliability_indices(events, total_customers=100, period_hours=hours)
    both = reliability.reliability_indices(
        events, total_customers=100, period_start=start, period_end=end, period_hours=hours
    )
    assert derived == explicit == both
    assert derived.period_hours == hours
    assert derived.asai == 1 - (12 / 60) / hours


@pytest.mark.parametrize(
    "kwargs, argument",
    [
        ({"period_hours": -1}, "period_hours"),
        ({"period_hours": 0}, "period_hours"),
        ({"period_hours": np.nan}, "period_hours"),
        ({"period_hours": np.inf}, "period_hours"),
        ({"period_hours": True}, "period_hours"),
        ({"period_hours": "24"}, "period_hours"),
        ({"period_start": "2024-01-01"}, "period_end"),
        ({"period_end": "2024-01-02"}, "period_start"),
        ({"period_start": "bad", "period_end": "2024-01-02"}, "period_start"),
        ({"period_start": "2024-01-01", "period_end": "2024-01-01"}, "period_end"),
        ({"period_start": "2024-01-02", "period_end": "2024-01-01"}, "period_end"),
        ({"period_start": "2024-01-01", "period_end": "2024-01-02T00:00:00Z"}, "period_end"),
        ({"period_start": "2024-01-01", "period_end": "2024-01-02", "period_hours": 12}, "period_hours"),
    ],
)
def test_asai_validates_period_through_public_api(kwargs: dict[str, Any], argument: str) -> None:
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices([], total_customers=100, **kwargs)
    assert exc.value.argument_name == argument


def test_asai_rejects_zero_total_customers() -> None:
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.reliability_indices([], total_customers=0, period_hours=24)
    assert exc.value.argument_name == "total_customers"


def test_asai_counts_step_restoration_rows_with_shared_filters() -> None:
    rows = [
        record(event_id="stage-a", duration=40.0, customers_interrupted=10),
        record(event_id="stage-b", duration=70.0, customers_interrupted=5),
        record(
            event_id="planned", start_time="2024-01-01T02:00:00", duration=60.0, customers_interrupted=20, planned=True
        ),
        record(event_id="momentary", start_time="2024-01-01T03:00:00", duration=3.0, customers_interrupted=10),
    ]
    with pytest.warns(OverlappingEventsWarning):
        result = reliability.reliability_indices(rows, total_customers=100, period_hours=24, exclude_planned=True)
    assert result.outage_count == 2
    assert result.customer_interruptions == 15
    assert result.customer_minutes_interrupted == 750
    assert result.saifi == 15 / 100
    assert result.saidi == 750 / 100
    assert result.asai == 1 - (7.5 / 60) / 24
    with pytest.warns(OverlappingEventsWarning):
        included = reliability.reliability_indices(rows, total_customers=100, period_hours=24, threshold_minutes=1)
    assert included.outage_count == 4
    assert included.customer_minutes_interrupted == 1980
    assert included.asai == 1 - (19.8 / 60) / 24


def test_asai_bounds_and_monotonicity_for_seeded_valid_exposure() -> None:
    rng = np.random.default_rng(1342)
    for duration in rng.uniform(6, 120, size=25):
        values = []
        for count in (0, 10, 50, 100):
            result = reliability.reliability_indices(
                [record(duration=float(duration), customers_interrupted=count)], total_customers=100, period_hours=2
            )
            assert result.asai is not None
            assert 0 <= result.asai <= 1
            assert result.asai == pytest.approx(float(1 - Fraction(float(duration)) * count / 12000), rel=0, abs=1e-15)
            values.append(result.asai)
        assert values == sorted(values, reverse=True)


def test_asai_full_exposure_is_zero() -> None:
    result = reliability.reliability_indices(
        [record(duration=60.0, customers_interrupted=100)], total_customers=100, period_hours=1
    )
    assert result.asai == 0.0
    assert result.undefined_metrics == ("maifi",)


def test_asai_excess_exposure_is_undefined_not_clamped() -> None:
    rows = [record(event_id=i, duration=60.0, customers_interrupted=100) for i in range(2)]
    with pytest.warns(OverlappingEventsWarning) as caught:
        result = reliability.reliability_indices(rows, total_customers=100, period_hours=1)
    assert len(caught) == 1
    assert result.asai is not None and math.isnan(result.asai)
    assert result.undefined_metrics == ("maifi", "asai")
    assert result.saidi == 120.0
    assert result.outage_count == 2
    with pytest.warns(OverlappingEventsWarning):
        reversed_result = reliability.reliability_indices(list(reversed(rows)), total_customers=100, period_hours=1)
    assert result == reversed_result
    assert hash(result) == hash(reversed_result)


@pytest.mark.parametrize("hours", [np.nextafter(0.0, 1.0), 1e308])
def test_asai_extreme_positive_hours_do_not_overflow_exposure_product(hours: float) -> None:
    empty = reliability.reliability_indices([], total_customers=1e308, period_hours=hours)
    assert empty.asai == 1.0
    result = reliability.reliability_indices([record()], total_customers=100, period_hours=hours)
    assert result.asai is not None
    if hours < 1:
        assert math.isnan(result.asai)
        assert result.undefined_metrics == ("maifi", "asai")
    else:
        assert result.asai == 1.0


def test_result_equality_distinguishes_unset_saidi_when_asai_is_requested() -> None:
    unset = reliability.ReliabilityResult(0.0, 0, 0, period_hours=24)
    zero = reliability.ReliabilityResult(0.0, 0, 0, saidi=0.0, period_hours=24)
    assert unset.asai is not None and math.isnan(unset.asai)
    assert zero.asai == 1.0
    assert unset != zero
    same = reliability.ReliabilityResult(0.0, 0, 0, period_hours=24)
    assert unset == same
    assert hash(unset) == hash(same)
    assert zero != reliability.ReliabilityResult(0.0, 0, 0, saidi=0.0, period_hours=48)


def test_result_is_frozen() -> None:
    result = reliability.ReliabilityResult(
        saifi=0.3, saidi=24.0, customer_interruptions=30, customer_minutes_interrupted=2400.0, outage_count=2
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.saifi = 0.4  # type: ignore[misc]


def test_public_exports() -> None:
    assert "reliability_indices" in reliability.__all__
    assert "ReliabilityResult" in reliability.__all__
    assert "group_momentary_events" in reliability.__all__


def test_nominal_momentary_event_and_operation_forms() -> None:
    rows = [
        record(event_id="a", duration=1.0, customers_interrupted=15, device_operations=2),
        record(
            event_id="b", start_time="2024-01-01T00:10:00", duration=2.0, customers_interrupted=15, device_operations=1
        ),
    ]
    result = reliability.reliability_indices(rows, total_customers=100)
    assert result.maifi_e == pytest.approx(0.3)
    assert result.maifi == pytest.approx(0.45)
    assert result.saifi == 0.0
    assert result.saidi == 0.0
    assert result.undefined_metrics == ("caidi", "asai")


@pytest.mark.parametrize("operations, customers", [(2**62, 2), (2, 2**62)])
def test_operation_products_beyond_int64_do_not_overflow(operations: int, customers: int) -> None:
    row = record(duration=1.0, customers_interrupted=customers, device_operations=operations)
    result = reliability.reliability_indices([row], total_customers=float(customers))
    assert result.maifi == float(operations)


def test_momentary_event_form_ignores_operations_without_operation_data() -> None:
    rows = [record(event_id="a", duration=1.0, customers_interrupted=15), record(event_id="b", duration=2.0)]
    rows[1]["customers_interrupted"] = 15
    rows[1]["start_time"] = "2024-01-01T00:10:00"
    result = reliability.reliability_indices(rows, total_customers=100)
    assert result.maifi_e == pytest.approx(0.3)
    assert result.maifi is None
    assert result.undefined_metrics == ("caidi", "maifi", "asai")


def test_operation_form_is_none_when_device_operations_are_absent() -> None:
    row = record(duration=1.0, customers_interrupted=15)
    result = reliability.reliability_indices([row], total_customers=100)
    assert result.maifi_e == pytest.approx(0.15)
    assert result.maifi is None
    assert "maifi" in result.undefined_metrics


def test_operation_form_is_zero_when_the_column_is_present_without_momentary_rows() -> None:
    result = reliability.reliability_indices(
        [record(duration=60.0, customers_interrupted=5, device_operations=1)], total_customers=100
    )
    assert result.maifi_e == 0.0
    assert result.maifi == 0.0
    assert result.undefined_metrics == ("asai",)


def test_operation_form_is_none_when_a_momentary_operation_count_is_missing() -> None:
    row = record(duration=1.0, customers_interrupted=15, device_operations=None)
    result = reliability.reliability_indices([row], total_customers=100)
    assert result.maifi_e == pytest.approx(0.15)
    assert result.maifi is None


def test_excluded_momentary_event_leaves_the_event_form_only() -> None:
    row = record(duration=1.0, customers_interrupted=15, device_operations=2, momentary_event_excluded=True)
    result = reliability.reliability_indices([row], total_customers=100)
    assert result.maifi_e == 0.0
    assert result.maifi == pytest.approx(0.3)


def test_adr_0026_fixture_maifi_analytic_values() -> None:
    events = json.loads(FIXTURE.read_text())
    result = reliability.reliability_indices(events, total_customers=1000)
    assert result.maifi_e == pytest.approx(0.3)
    assert result.maifi == pytest.approx(0.75)
    assert result.undefined_metrics == ("asai",)


def test_empty_log_reports_zero_event_form_and_undefined_operation_form() -> None:
    result = reliability.reliability_indices([], total_customers=1000)
    assert result.maifi_e == 0.0
    assert result.maifi is None
    assert result.undefined_metrics == ("caidi", "maifi", "asai")


def test_empty_log_with_operation_column_reports_zero_operation_form() -> None:
    result = reliability.reliability_indices(pd.DataFrame(columns=["device_operations"]), total_customers=1000)
    assert result.maifi_e == 0.0
    assert result.maifi == 0.0
    assert result.undefined_metrics == ("caidi", "asai")


def test_momentary_row_order_does_not_change_the_maifi_values() -> None:
    rows = [
        record(event_id="a", duration=1.0, customers_interrupted=15, device_operations=2),
        record(
            event_id="b", start_time="2024-01-01T00:10:00", duration=2.0, customers_interrupted=15, device_operations=1
        ),
    ]
    forward = reliability.reliability_indices(rows, total_customers=100)
    backward = reliability.reliability_indices(list(reversed(rows)), total_customers=100)
    assert forward.maifi_e == backward.maifi_e
    assert forward.maifi == backward.maifi

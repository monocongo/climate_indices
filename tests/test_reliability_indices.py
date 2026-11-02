"""Contract tests for the reliability metric accumulators (SAIFI and CI)."""

import dataclasses
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from climate_indices import reliability
from climate_indices.exceptions import ClimateIndicesWarning, InvalidArgumentError

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


def test_nominal_analytic_case() -> None:
    events = [
        record(event_id="a", duration=60.0, customers_interrupted=20),
        record(event_id="b", start_time="2024-01-01T02:00:00", duration=120.0, customers_interrupted=10),
    ]
    result = reliability.reliability_indices(events, total_customers=100)
    assert result.saifi == pytest.approx(0.3)
    assert result.customer_interruptions == 30
    assert result.outage_count == 2
    assert isinstance(result.saifi, float)
    assert isinstance(result.customer_interruptions, int)
    assert not isinstance(result.customer_interruptions, bool)
    assert isinstance(result.outage_count, int)
    assert not isinstance(result.outage_count, bool)


@pytest.mark.parametrize("events", [[], pd.DataFrame()])
def test_empty_log_returns_zero_without_infinity(events: Any) -> None:
    result = reliability.reliability_indices(events, total_customers=1000)
    assert result == reliability.ReliabilityResult(saifi=0.0, customer_interruptions=0, outage_count=0)


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


def test_momentary_rows_contribute_nothing() -> None:
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
    assert with_momentary == without
    assert with_momentary.customer_interruptions == 20
    assert with_momentary.outage_count == 1


def test_explicitly_momentary_long_row_is_excluded_with_one_warning() -> None:
    row = record(duration=60.0, customers_interrupted=50, interruption_type="momentary")
    with pytest.warns(ClimateIndicesWarning, match="interruption_type") as caught:
        result = reliability.reliability_indices([row], total_customers=100)
    assert len(caught) == 1
    assert result == reliability.ReliabilityResult(saifi=0.0, customer_interruptions=0, outage_count=0)


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
    result = reliability.reliability_indices(events, total_customers=1000)
    assert result.saifi == pytest.approx(0.3)
    assert result.customer_interruptions == 300
    assert result.outage_count == 2


def test_zero_duration_event_counts_only_when_explicitly_sustained() -> None:
    derived = reliability.reliability_indices([record(duration=0.0, customers_interrupted=5)], total_customers=100)
    assert derived == reliability.ReliabilityResult(saifi=0.0, customer_interruptions=0, outage_count=0)
    explicit = record(duration=0.0, customers_interrupted=5, interruption_type="sustained")
    with pytest.warns(ClimateIndicesWarning, match="interruption_type") as caught:
        counted = reliability.reliability_indices([explicit], total_customers=100)
    assert len(caught) == 1
    assert counted.saifi == pytest.approx(0.05)
    assert counted.customer_interruptions == 5
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


def test_result_is_frozen() -> None:
    result = reliability.ReliabilityResult(saifi=0.3, customer_interruptions=30, outage_count=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.saifi = 0.4  # type: ignore[misc]


def test_public_exports() -> None:
    assert "reliability_indices" in reliability.__all__
    assert "ReliabilityResult" in reliability.__all__

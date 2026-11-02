"""Contract tests for momentary-operation grouping into MAIFI_E events."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from climate_indices import reliability
from climate_indices.exceptions import InvalidArgumentError


def operation(
    event_id: str,
    start_time: str,
    *,
    duration: float = 0.5,
    customers_interrupted: int = 15,
    device_id: str | None = "device-a",
    **changes: object,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "start_time": start_time,
        "duration": duration,
        "customers_interrupted": customers_interrupted,
        "device_id": device_id,
        **changes,
    }


def test_two_operations_four_minutes_apart_form_one_event() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00"),
        operation("op-2", "2024-01-01T00:04:00"),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["op-1"]
    assert grouped["interruption_type"].tolist() == ["momentary"]
    assert grouped["customers_interrupted"].tolist() == [15]
    assert grouped["duration"].tolist() == [4.5]


def test_operations_five_minutes_one_second_apart_form_two_events() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00"),
        operation("op-2", "2024-01-01T00:05:01"),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["op-1", "op-2"]


def test_exactly_five_minutes_is_inside_the_window() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", duration=0.0),
        operation("op-2", "2024-01-01T00:05:00", duration=0.0),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["op-1"]
    assert grouped["duration"].tolist() == [5.0]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_operations_within_the_window_collapse_to_one_event(count: int) -> None:
    rows = [operation(f"op-{index}", f"2024-01-01T00:{index:02d}:00", duration=0.0) for index in range(count)]
    grouped = reliability.group_momentary_events(rows)
    assert len(grouped) == 1
    assert grouped["customers_interrupted"].tolist() == [15]


def test_operation_after_the_window_opens_a_new_event() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", duration=0.0),
        operation("op-2", "2024-01-01T00:04:00", duration=0.0),
        operation("op-3", "2024-01-01T00:08:00", duration=0.0),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["op-1", "op-3"]


def test_group_that_completes_after_the_window_is_rejected() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", duration=0.0),
        operation("op-2", "2024-01-01T00:04:50", duration=5.0),
    ]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.group_momentary_events(rows)
    assert exc.value.argument_name == "momentary_event"


def test_devices_never_share_one_event() -> None:
    rows = [
        operation("d1-op", "2024-01-01T00:00:00", device_id="device-a"),
        operation("d2-op", "2024-01-01T00:01:00", device_id="device-b"),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["d1-op", "d2-op"]


def test_null_device_rows_group_only_with_each_other() -> None:
    rows = [
        operation("null-1", "2024-01-01T00:00:00", device_id=None),
        operation("named", "2024-01-01T00:01:00", device_id="device-a"),
        operation("null-2", "2024-01-01T00:02:00", device_id=None),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["null-1", "named"]


def test_sustained_rows_pass_through_unchanged() -> None:
    rows = [
        operation("sustained", "2024-01-01T00:00:00", duration=60.0),
        operation("op-1", "2024-01-01T02:00:00"),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["sustained", "op-1"]
    assert grouped["interruption_type"].tolist() == ["sustained", "momentary"]


def test_different_customer_counts_in_one_group_are_rejected() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", customers_interrupted=15),
        operation("op-2", "2024-01-01T00:01:00", customers_interrupted=30),
    ]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.group_momentary_events(rows)
    assert exc.value.argument_name == "customers_interrupted"


def test_mixed_exclusion_flags_in_one_group_are_rejected() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", momentary_event_excluded=False),
        operation("op-2", "2024-01-01T00:01:00", momentary_event_excluded=True),
    ]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.group_momentary_events(rows)
    assert exc.value.argument_name == "momentary_event_excluded"


def test_device_operations_are_summed_within_a_group() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", device_operations=2),
        operation("op-2", "2024-01-01T00:02:00", device_operations=3),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["device_operations"].tolist() == [5]


def test_grouped_operation_total_above_int64_names_the_field() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", device_operations=2**62),
        operation("op-2", "2024-01-01T00:02:00", device_operations=2**62),
    ]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.group_momentary_events(rows)
    assert exc.value.argument_name == "device_operations"


def test_grouped_operation_total_at_int64_limit_is_accepted() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", device_operations=2**62),
        operation("op-2", "2024-01-01T00:02:00", device_operations=2**62 - 1),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["device_operations"].tolist() == [2**63 - 1]
    assert str(grouped["device_operations"].dtype) == "Int64"


def test_absent_device_id_column_groups_by_time_adjacency() -> None:
    rows = [
        {"event_id": "op-1", "start_time": "2024-01-01T00:00:00", "duration": 0.0, "customers_interrupted": 15},
        {"event_id": "op-2", "start_time": "2024-01-01T00:04:00", "duration": 0.0, "customers_interrupted": 15},
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["op-1"]
    assert "device_id" not in grouped.columns


def test_absent_device_operations_stay_absent() -> None:
    grouped = reliability.group_momentary_events([operation("op-1", "2024-01-01T00:00:00")])
    assert "device_operations" not in grouped.columns


def test_grouping_output_feeds_the_event_and_operation_metrics() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", customers_interrupted=15, device_operations=2),
        operation("op-2", "2024-01-01T00:03:00", customers_interrupted=15, device_operations=1),
    ]
    grouped = reliability.group_momentary_events(rows)
    result = reliability.reliability_indices(grouped, total_customers=100)
    assert result.maifi_e == pytest.approx(0.15)
    assert result.maifi == pytest.approx(0.45)


def test_grouping_preserves_order_and_dtypes() -> None:
    rows = [
        operation("sustained", "2024-01-01T00:00:00", duration=60.0),
        operation("op-1", "2024-01-01T02:00:00"),
        operation("op-2", "2024-01-01T02:03:00"),
        operation("sustained-2", "2024-01-01T04:00:00", duration=90.0),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["sustained", "op-1", "sustained-2"]
    assert grouped["duration"].dtype == np.float64
    assert grouped["customers_interrupted"].dtype == np.int64
    assert grouped["planned"].dtype == bool
    assert grouped["momentary_event_excluded"].dtype == bool


def test_unsorted_grouping_keeps_anchor_metadata_and_nullable_counts() -> None:
    rows = [
        operation("late", "2024-01-01T00:03:00Z", device_operations=1, cause="late"),
        operation("sustained", "2024-01-02T00:00:00Z", duration=60.0, device_operations=2),
        operation("anchor", "2024-01-01T00:00:00Z", device_operations=None, cause="anchor"),
    ]
    frame = pd.DataFrame(rows, index=[7, 7, 3], dtype=object)
    original = frame.copy(deep=True)
    grouped = reliability.group_momentary_events(frame)
    expected_rows = [dict(rows[2], duration=3.5), rows[1]]
    expected = reliability.classify_interruptions(expected_rows)
    pd.testing.assert_frame_equal(grouped, expected)
    pd.testing.assert_frame_equal(frame, original)


def test_empty_input_returns_an_empty_frame() -> None:
    grouped = reliability.group_momentary_events([])
    assert grouped.empty
    assert grouped["duration"].dtype == np.float64


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "5", 1e12])
def test_invalid_window_minutes(value: Any) -> None:
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.group_momentary_events([operation("op-1", "2024-01-01T00:00:00")], window_minutes=value)
    assert exc.value.argument_name == "window_minutes"


@pytest.mark.parametrize("timestamp", [pd.Timestamp.min, pd.Timestamp.max])
def test_zero_duration_at_timestamp_boundaries_is_accepted(timestamp: pd.Timestamp) -> None:
    row = operation("op-1", timestamp.isoformat(), duration=0.0)
    grouped = reliability.group_momentary_events([row])
    assert grouped["start_time"].tolist() == [timestamp]
    assert grouped["end_time"].tolist() == [timestamp]
    assert grouped["duration"].tolist() == [0.0]


def test_window_near_timestamp_max_keeps_the_exact_boundary() -> None:
    anchor = pd.Timestamp.max - pd.Timedelta(minutes=5, nanoseconds=1)
    rows = [
        operation("anchor", anchor.isoformat(), duration=0.0),
        operation("inside", (pd.Timestamp.max - pd.Timedelta(nanoseconds=1)).isoformat(), duration=0.0),
        operation("outside", pd.Timestamp.max.isoformat(), duration=0.0),
    ]
    grouped = reliability.group_momentary_events(rows)
    assert grouped["event_id"].tolist() == ["anchor", "outside"]


def test_window_minutes_is_separate_from_the_threshold() -> None:
    rows = [
        operation("op-1", "2024-01-01T00:00:00", duration=0.0),
        operation("op-2", "2024-01-01T00:03:00", duration=0.0),
    ]
    grouped = reliability.group_momentary_events(rows, threshold_minutes=1.0)
    assert grouped["event_id"].tolist() == ["op-1"]
    assert grouped["duration"].tolist() == [3.0]

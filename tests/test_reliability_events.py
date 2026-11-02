"""Contract tests for reliability records, not metric calculations."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from climate_indices import reliability
from climate_indices.exceptions import ClimateIndicesWarning, InvalidArgumentError


def test_classification_boundary_and_normalized_arrays() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "zero"],
            "start_time": ["2024-01-01T00:00:00"] * 4,
            "duration": [299 / 60, 5, 301 / 60, 0],
            "customers_interrupted": [1, 2, 3, 0],
        }
    )
    # Separate the events so this test does not need an overlap warning.
    events["start_time"] = pd.date_range("2024-01-01", periods=4, freq="D")
    result = reliability.classify_interruptions(events)
    assert result["interruption_type"].tolist() == ["momentary", "momentary", "sustained", "momentary"]
    assert result["duration"].to_numpy().dtype == np.dtype("float64")
    assert result["customers_interrupted"].to_numpy().dtype == np.dtype("int64")
    np.testing.assert_array_equal(result["duration"], events["duration"])
    assert result["planned"].tolist() == [False] * 4
    assert "interruption_type" not in events


def record(**changes: object) -> dict[str, object]:
    return {
        "event_id": "a",
        "start_time": "2024-01-01T00:00:00",
        "duration": 10.0,
        "customers_interrupted": 5,
        **changes,
    }


@pytest.mark.parametrize(
    "changes, argument",
    [
        ({"event_id": None}, "event_id"),
        ({"event_id": []}, "event_id"),
        ({"start_time": "invalid"}, "start_time"),
        ({"start_time": None}, "start_time"),
        ({"start_time": "NaT"}, "start_time"),
        ({"end_time": "invalid"}, "end_time"),
        ({"duration": 1e300}, "duration"),
        ({"duration": 2**53 + 1}, "duration"),
        ({"device_id": 1}, "device_id"),
        ({"cause": 1}, "cause"),
        ({"momentary_event_excluded": 1}, "momentary_event_excluded"),
        ({"component_ids": [1]}, "component_ids"),
        ({"interruption_type": ["momentary"]}, "interruption_type"),
        ({"duration": None}, "duration"),
        ({"duration": np.nan}, "duration"),
        ({"duration": np.inf}, "duration"),
        ({"duration": -1}, "duration"),
        ({"duration": "10"}, "duration"),
        ({"duration": True}, "duration"),
        ({"customers_interrupted": np.nan}, "customers_interrupted"),
        ({"customers_interrupted": -1}, "customers_interrupted"),
        ({"customers_interrupted": 1.5}, "customers_interrupted"),
        ({"customers_interrupted": 1.0}, "customers_interrupted"),
        ({"customers_interrupted": "5"}, "customers_interrupted"),
        ({"customers_interrupted": True}, "customers_interrupted"),
        ({"customers_interrupted": 2**63}, "customers_interrupted"),
        ({"end_time": "2023-12-31"}, "end_time"),
        ({"end_time": "2024-01-01T00:09:00"}, "duration"),
        ({"end_time": "2024-01-01T00:10:00Z"}, "end_time"),
        ({"planned": "false"}, "planned"),
        ({"device_operations": 1.5}, "device_operations"),
        ({"interruption_type": "other"}, "interruption_type"),
        ({"component_ids": "line"}, "component_ids"),
        ({"failure_rate": 0.1}, "failure_rate"),
    ],
)
def test_invalid_fields_name_the_argument(changes: dict[str, object], argument: str) -> None:
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.classify_interruptions([record(**changes)])
    assert exc.value.argument_name == argument


def test_missing_and_duplicate_fields() -> None:
    for field in ("event_id", "start_time", "duration", "customers_interrupted"):
        row = record()
        del row[field]
        with pytest.raises(InvalidArgumentError) as exc:
            reliability.classify_interruptions([row])
        assert exc.value.argument_name == field
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.classify_interruptions([record(), record()])
    assert exc.value.argument_name == "event_id"


def test_mixed_timezone_awareness_within_a_column() -> None:
    rows = [record(), record(event_id="b", start_time="2024-01-02T00:00:00Z")]
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.classify_interruptions(rows)
    assert exc.value.argument_name == "start_time"


def test_end_times_and_timezone_normalization() -> None:
    row = record(end_time="2024-01-01T01:10:00+01:00", start_time="2024-01-01T00:00:00Z")
    del row["duration"]
    result = reliability.classify_interruptions([row])
    assert result["duration"].tolist() == [10.0]
    assert str(result["start_time"].dt.tz) == "UTC"
    assert result["end_time"].iloc[0] == pd.Timestamp("2024-01-01T00:10:00Z")


def test_nat_optional_values_are_missing() -> None:
    result = reliability.classify_interruptions([record(cause=pd.NaT, planned=pd.NaT)])
    assert result["cause"].tolist() == [None]
    assert result["planned"].tolist() == [False]


def test_explicit_types_are_retained_with_one_warning() -> None:
    with pytest.warns(ClimateIndicesWarning, match="interruption_type") as caught:
        result = reliability.classify_interruptions([record(interruption_type="momentary")])
    assert len(caught) == 1
    assert result["interruption_type"].tolist() == ["momentary"]
    result = reliability.classify_interruptions([record(duration=1)], threshold_minutes=0.5)
    assert result["interruption_type"].tolist() == ["sustained"]


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "5"])
def test_invalid_threshold(value: Any) -> None:
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.classify_interruptions([record()], threshold_minutes=value)
    assert exc.value.argument_name == "threshold_minutes"


def test_overlaps_warn_once_without_losing_or_sorting_rows() -> None:
    from climate_indices.exceptions import OverlappingEventsWarning

    rows = [
        record(event_id="late", start_time="2024-01-01T00:01:00"),
        record(),
        record(event_id="third", start_time="2024-01-01T00:02:00"),
    ]
    with pytest.warns(OverlappingEventsWarning) as caught:
        result = reliability.classify_interruptions(rows)
    assert len(caught) == 1
    assert issubclass(OverlappingEventsWarning, ClimateIndicesWarning)
    assert result["event_id"].tolist() == ["late", "a", "third"]


def test_adjacent_and_zero_duration_events_do_not_overlap() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reliability.classify_interruptions(
            [
                record(),
                record(event_id="b", start_time="2024-01-01T00:10:00"),
                record(event_id="zero", duration=0),
            ]
        )
    assert not caught
    assert len(result) == 3


def test_optional_missing_values_and_input_copy() -> None:
    frame = pd.DataFrame(
        [
            record(
                planned=None,
                device_id=None,
                device_operations=None,
                cause=None,
                component_ids=None,
                interruption_type=None,
                momentary_event_excluded=None,
            )
        ]
    )
    original = frame.copy(deep=True)
    result = reliability.classify_interruptions(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert result["planned"].tolist() == [False]
    assert result["momentary_event_excluded"].tolist() == [False]
    assert result["interruption_type"].tolist() == ["sustained"]
    assert str(result["device_operations"].dtype) == "Int64"
    assert pd.isna(result["device_operations"].iloc[0])
    assert result["component_ids"].iloc[0] is None


def test_optional_integer_nulls_and_large_counts_keep_exact_values() -> None:
    rows = [
        record(event_id="a", customers_interrupted=2**53 + 1, device_operations=2),
        record(event_id="b", start_time="2024-01-02", device_operations=None),
    ]
    result = reliability.classify_interruptions(rows)
    assert result["customers_interrupted"].tolist() == [2**53 + 1, 5]
    assert result["device_operations"].iloc[0] == 2
    assert pd.isna(result["device_operations"].iloc[1])


@pytest.mark.parametrize("total", [0, -1, np.nan, np.inf, True, "100"])
def test_invalid_served_customer_count(total: Any) -> None:
    from climate_indices.reliability._events import _normalize_events

    with pytest.raises(InvalidArgumentError) as exc:
        _normalize_events([record()], total_customers=total)
    assert exc.value.argument_name == "total_customers"


def test_customer_bound_and_average_served_count() -> None:
    from climate_indices.reliability._events import _normalize_events

    with pytest.raises(InvalidArgumentError) as exc:
        _normalize_events([record()], total_customers=4.5)
    assert exc.value.argument_name == "customers_interrupted"
    result = _normalize_events([record()], total_customers=5.5)
    assert result["customers_interrupted"].tolist() == [5]
    with pytest.raises(InvalidArgumentError) as exc:
        _normalize_events([record(customers_interrupted=2**53 + 1)], total_customers=2**53)
    assert exc.value.argument_name == "customers_interrupted"


@pytest.mark.parametrize("year, expected", [(2023, 8760), (2024, 8784)])
def test_reporting_period_actual_hours(year: int, expected: int) -> None:
    from climate_indices.reliability._events import _period_hours

    assert _period_hours(period_start=f"{year}-01-01", period_end=f"{year + 1}-01-01") == expected
    assert _period_hours() is None
    assert _period_hours(period_hours=24) == 24.0


@pytest.mark.parametrize(
    "kwargs, argument",
    [
        ({"period_hours": 0}, "period_hours"),
        ({"period_hours": np.nan}, "period_hours"),
        ({"period_hours": True}, "period_hours"),
        ({"period_start": "2024-01-01"}, "period_end"),
        ({"period_end": "2024-01-01"}, "period_start"),
        ({"period_start": "bad", "period_end": "2024-01-01"}, "period_start"),
        ({"period_start": "2024-01-01", "period_end": "2024-01-01"}, "period_end"),
        ({"period_start": "2024-01-02", "period_end": "2024-01-01"}, "period_end"),
        ({"period_start": "2024-01-01", "period_end": "2024-01-02T00:00:00Z"}, "period_end"),
        ({"period_start": "2024-01-01", "period_end": "2024-01-02", "period_hours": 12}, "period_hours"),
    ],
)
def test_invalid_reporting_period(kwargs: dict[str, Any], argument: str) -> None:
    from climate_indices.reliability._events import _period_hours

    with pytest.raises(InvalidArgumentError) as exc:
        _period_hours(**kwargs)
    assert exc.value.argument_name == argument


def test_reporting_period_uses_elapsed_hours_across_dst() -> None:
    from climate_indices.reliability._events import _period_hours

    assert (
        _period_hours(period_start="2024-03-10T00:00:00-05:00", period_end="2024-03-11T00:00:00-04:00", period_hours=23)
        == 23.0
    )


def test_masked_required_value_is_not_silently_restored() -> None:
    with pytest.raises(InvalidArgumentError) as exc:
        reliability.classify_interruptions(
            {
                "event_id": ["a"],
                "start_time": ["2024-01-01"],
                "duration": [1],
                "customers_interrupted": np.ma.array([5], mask=[True]),
            }
        )
    assert exc.value.argument_name == "customers_interrupted"


def test_empty_record_has_canonical_dtypes_and_preserves_operation_availability() -> None:
    result = reliability.classify_interruptions([])
    assert result["duration"].dtype == np.float64
    assert result["customers_interrupted"].dtype == np.int64
    assert "device_operations" not in result
    result = reliability.classify_interruptions(pd.DataFrame(columns=["device_operations"]))
    assert str(result["device_operations"].dtype) == "Int64"

"""Normalize reliability event records and reporting-period inputs."""

import warnings
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
import pandas as pd

from climate_indices.exceptions import ClimateIndicesWarning, InvalidArgumentError, OverlappingEventsWarning

EventTable: TypeAlias = pd.DataFrame | Mapping[str, object] | Sequence[Mapping[str, object]]
_FIELDS = {
    "event_id",
    "start_time",
    "end_time",
    "duration",
    "customers_interrupted",
    "interruption_type",
    "planned",
    "device_id",
    "device_operations",
    "momentary_event_excluded",
    "grouped_momentary_event",
    "cause",
    "component_ids",
}
_REQUIRED = ("event_id", "start_time", "customers_interrupted")


def _invalid(name: str, message: str) -> InvalidArgumentError:
    return InvalidArgumentError(f"{name}: {message}", argument_name=name)


def _missing(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or value is np.ma.masked
        or (isinstance(value, (float, np.floating)) and bool(np.isnan(value)))
    )


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.integer, np.floating)):
        raise _invalid(name, "must be numeric, not a string or Boolean")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise _invalid(name, "must be a finite float64 value") from exc
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        raise _invalid(name, "must be finite and positive" if positive else "must be finite and nonnegative")
    if isinstance(value, (Integral, np.integer)) and result != int(value):
        raise _invalid(name, "cannot be represented exactly as float64")
    # Mathematically integral values must survive the float64 round trip
    # exactly; later bounds (for example customers_interrupted <=
    # total_customers) compare against them. This covers denominator-one
    # Fractions and integral NumPy floats in extended precision.
    integral = getattr(value, "denominator", None) == 1 or (
        isinstance(value, np.floating) and bool(np.mod(value, 1) == 0)
    )
    if integral and value != result:
        raise _invalid(name, "cannot be represented exactly as float64")
    return result


def _integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (Integral, np.integer))
        or not 0 <= int(value) <= np.iinfo(np.int64).max
    ):
        raise _invalid(name, "must be a nonnegative int64 integer, not a float, string, or Boolean")
    return int(value)


def _timestamps(values: Sequence[object], name: str) -> pd.DatetimeIndex:
    parsed = []
    awareness = set()
    for value in values:
        if not isinstance(value, (str, date, datetime, np.datetime64, pd.Timestamp)):
            raise _invalid(name, "must contain non-null timestamps")
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                raise ValueError("missing timestamp")
            awareness.add(timestamp.tzinfo is not None)
            parsed.append(timestamp)
        except (ValueError, TypeError, OverflowError) as exc:
            raise _invalid(name, "contains an unparseable timestamp") from exc
    if len(awareness) > 1:
        raise _invalid(name, "must not mix timezone-aware and timezone-naive timestamps")
    try:
        return pd.DatetimeIndex(pd.to_datetime(parsed, utc=True) if True in awareness else parsed).as_unit("ns")
    except (ValueError, TypeError, OverflowError) as exc:
        raise _invalid(name, "timestamps must fit datetime64[ns]") from exc


def _event_frame(events: EventTable) -> pd.DataFrame:
    """Return a deep copy with unique, known columns and required fields."""
    if events is None:
        raise _invalid("events", "must be a DataFrame, mapping, or sequence of row mappings")
    try:
        frame = pd.DataFrame(events, dtype=object).copy(deep=True)
    except (ValueError, TypeError) as exc:
        raise _invalid("events", "must be a rectangular event table") from exc
    if not frame.columns.is_unique:
        raise _invalid("events", "column names must be unique")
    for field in frame.columns:
        if field not in _FIELDS:
            raise _invalid(str(field), "is not a deterministic event field")
    if frame.empty:
        for field in (*_REQUIRED, "duration"):
            if field not in frame:
                frame[field] = pd.Series(dtype=object)
    for field in _REQUIRED:
        if field not in frame:
            raise _invalid(field, "is required")
    return frame


def _validate_ids(frame: pd.DataFrame) -> None:
    """Require unique, non-Boolean string or integer identifiers."""
    ids = frame["event_id"].tolist()
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (str, Integral, np.integer)) for v in ids):
        raise _invalid("event_id", "must contain non-null string or integer identifiers")
    if frame["event_id"].duplicated().any():
        raise _invalid("event_id", "must be unique")


def _from_end_times(frame: pd.DataFrame, start: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return end times and their elapsed minutes."""
    end = _timestamps(frame["end_time"].tolist(), "end_time")
    if (start.tz is None) != (end.tz is None):
        raise _invalid("end_time", "timezone awareness must match start_time")
    if (end < start).any():
        raise _invalid("end_time", "must not precede start_time")
    elapsed = np.array([(e - s) / pd.Timedelta(minutes=1) for s, e in zip(start, end, strict=True)], dtype=np.float64)
    return end, elapsed


def _from_durations(start: pd.DatetimeIndex, duration: np.ndarray) -> pd.DatetimeIndex:
    """Return end times derived from durations in minutes."""
    if (duration > pd.Timedelta.max.total_seconds() / 60).any():
        raise _invalid("duration", "must fit a pandas Timedelta")
    end = pd.DatetimeIndex(start + pd.to_timedelta(duration, unit="min"))
    if end.isna().any():
        raise ValueError("duration overflows timestamps")
    return end


def _interval(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray]:
    """Resolve start, end, and duration from either duration or end_time."""
    start = _timestamps(frame["start_time"].tolist(), "start_time")
    if "duration" not in frame and "end_time" not in frame:
        raise _invalid("duration", "supply duration in minutes or end_time")
    duration = (
        np.array([_number(v, "duration") for v in frame["duration"].tolist()], dtype=np.float64)
        if "duration" in frame
        else None
    )
    try:
        if "end_time" in frame:
            end, elapsed = _from_end_times(frame, start)
            if duration is not None and not np.allclose(duration, elapsed, rtol=0, atol=1e-9):
                raise _invalid("duration", "must match end_time minus start_time within 1e-9 minutes")
            duration = elapsed if duration is None else duration
        else:
            if duration is None:
                raise _invalid("duration", "is required without end_time")
            end = _from_durations(start, duration)
    except (ValueError, TypeError, OverflowError) as exc:
        raise _invalid("duration", "cannot form a representable event interval") from exc
    return start, end, duration


def _flag_column(frame: pd.DataFrame, field: str) -> None:
    """Normalize an optional Boolean column, where nulls become False."""
    values = frame[field].tolist() if field in frame else [False] * len(frame)
    if any(not _missing(v) and not isinstance(v, (bool, np.bool_)) for v in values):
        raise _invalid(field, "must contain Booleans or nulls")
    frame[field] = np.array([False if _missing(v) else v for v in values], dtype=bool)


def _string_column(frame: pd.DataFrame, field: str) -> None:
    """Normalize an optional string column, where nulls stay unknown."""
    if field not in frame:
        return
    values = frame[field].tolist()
    if any(not _missing(v) and not isinstance(v, str) for v in values):
        raise _invalid(field, "must contain strings or nulls")
    frame[field] = [None if _missing(v) else v for v in values]


def _device_operations(frame: pd.DataFrame) -> None:
    """Normalize optional operation counts to nullable Int64."""
    if "device_operations" not in frame:
        return
    frame["device_operations"] = pd.array(
        [pd.NA if _missing(v) else _integer(v, "device_operations") for v in frame["device_operations"].tolist()],
        dtype="Int64",
    )


def _component_ids(frame: pd.DataFrame) -> None:
    """Normalize optional component identifier lists, where nulls stay unknown."""
    if "component_ids" not in frame:
        return
    values = frame["component_ids"].tolist()
    for value in values:
        if not _missing(value) and (not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value)):
            raise _invalid("component_ids", "must contain lists of string identifiers or nulls")
    frame["component_ids"] = [None if _missing(v) else list(v) for v in values]


def _classify(frame: pd.DataFrame, threshold: float) -> None:
    """Derive interruption types, retaining explicit types with one warning.

    A grouped momentary event keeps its authoritative type without a warning,
    because its duration is an elapsed sequence span, not an operation duration.
    """
    derived = np.where(frame["duration"] > threshold, "sustained", "momentary").tolist()
    explicit = frame["interruption_type"].tolist() if "interruption_type" in frame else [None] * len(frame)
    grouped = frame["grouped_momentary_event"].tolist() if "grouped_momentary_event" in frame else [False] * len(frame)
    if any(not _missing(v) and (not isinstance(v, str) or v not in ("sustained", "momentary")) for v in explicit):
        raise _invalid("interruption_type", "must contain sustained, momentary, or null")
    frame["interruption_type"] = [
        ("momentary" if g else d) if _missing(v) else v for v, d, g in zip(explicit, derived, grouped, strict=True)
    ]
    if any(not _missing(v) and v != d and not g for v, d, g in zip(explicit, derived, grouped, strict=True)):
        warnings.warn(
            "interruption_type contradicts duration; explicit types are retained.", ClimateIndicesWarning, stacklevel=2
        )


def _warn_overlaps(start: pd.DatetimeIndex, end: pd.DatetimeIndex, duration: np.ndarray) -> None:
    """Warn once when positive-duration intervals overlap."""
    positive = duration > 0
    order = np.argsort(start.asi8[positive], kind="stable")
    starts = start.asi8[positive][order]
    ends = end.asi8[positive][order]
    if len(starts) > 1 and (starts[1:] < np.maximum.accumulate(ends)[:-1]).any():
        warnings.warn("Event intervals overlap; every event is retained.", OverlappingEventsWarning, stacklevel=2)


def classify_interruptions(events: EventTable, *, threshold_minutes: float = 5.0) -> pd.DataFrame:
    """Validate and copy records, then classify interruptions by duration.

    Args:
        events: A DataFrame, column mapping, or sequence of row mappings.
            Required fields are event_id, start_time, customers_interrupted,
            and either duration (minutes) or end_time. Customer counts must
            be integers. Optional fields follow the reliability event schema.
        threshold_minutes: Positive sustained-interruption threshold, in minutes.
            Durations strictly above it are sustained; equality is momentary.

    Returns:
        A new DataFrame in input order with float64 durations, int64 customer
        counts, timestamps (UTC for aware input), and interruption_type.
        Missing planned and momentary_event_excluded flags become False.
        Optional device_operations uses nullable Int64; absence stays absent.

    Raises:
        InvalidArgumentError: A field is missing, invalid, or inconsistent.
            The exception's argument_name identifies that field.

    Warns:
        ClimateIndicesWarning: An explicit type contradicts the duration.
        OverlappingEventsWarning: Positive-duration intervals overlap.
            One warning covers all overlaps; every event is retained.
    """
    threshold = _number(threshold_minutes, "threshold_minutes", positive=True)
    frame = _event_frame(events)
    _validate_ids(frame)
    frame["customers_interrupted"] = np.array(
        [_integer(v, "customers_interrupted") for v in frame["customers_interrupted"].tolist()], dtype=np.int64
    )
    start, end, duration = _interval(frame)
    frame["start_time"] = start
    frame["end_time"] = end
    frame["duration"] = duration
    _flag_column(frame, "planned")
    _flag_column(frame, "momentary_event_excluded")
    if "grouped_momentary_event" in frame:
        _flag_column(frame, "grouped_momentary_event")
    _string_column(frame, "device_id")
    _string_column(frame, "cause")
    _device_operations(frame)
    _component_ids(frame)
    _classify(frame, threshold)
    _warn_overlaps(start, end, duration)
    return frame


def _momentary_groups(by_device: dict[object, list[int]], start_ns: list[int], window_ns: int) -> Iterator[list[int]]:
    """Yield groups anchored at each device's first ungrouped operation."""
    for positions in by_device.values():
        positions.sort(key=lambda position: (start_ns[position], position))
        cursor = 0
        while cursor < len(positions):
            anchor = positions[cursor]
            deadline = start_ns[anchor] + window_ns
            members = [anchor]
            cursor += 1
            while cursor < len(positions) and start_ns[positions[cursor]] <= deadline:
                members.append(positions[cursor])
                cursor += 1
            yield members


def _momentary_group_end(
    members: list[int],
    end_ns: list[int],
    deadline: int,
    customer_counts: list[int],
    exclusion_flags: list[bool],
) -> int:
    """Validate a group's interval and shared fields, then return its last end position."""
    last_end = max(members, key=lambda position: end_ns[position])
    if end_ns[last_end] > deadline:
        raise _invalid("momentary_event", "must complete within window_minutes; supply completed events")
    customers = {customer_counts[position] for position in members}
    if len(customers) > 1:
        raise _invalid(
            "customers_interrupted",
            "grouped momentary operations must interrupt the same customers; supply one event-level row",
        )
    excluded = {exclusion_flags[position] for position in members}
    if len(excluded) > 1:
        raise _invalid("momentary_event_excluded", "grouped momentary operations must share one exclusion flag")
    return last_end


def group_momentary_events(
    events: EventTable,
    *,
    window_minutes: float = 5.0,
    threshold_minutes: float = 5.0,
) -> pd.DataFrame:
    """Collapse momentary operation rows into one event per device and window.

    Each momentary input row must be a single reclosing operation. An event is
    anchored at the start of its first operation on one ``device_id``. A later
    operation on the same device joins when its start time is at most
    ``window_minutes`` after the anchor; the window never moves forward from the
    latest operation, and operations on different devices never join. Rows with
    no ``device_id`` group only with other rows that also have none, which is a
    library convention and can merge distinct unknown devices.

    The window and the sustained-interruption threshold are separate settings.
    The IEEE grouping rule is informative rather than a requirement, and the
    supplied record may already hold whole reclosing sequences; this helper
    cannot tell a second operation from a new sequence.

    Args:
        events: A DataFrame, column mapping, or sequence of row mappings,
            validated and classified by the reliability event contract.
        window_minutes: Positive grouping window, in minutes. An operation joins
            its anchor when it starts at or before the anchor plus this window.
        threshold_minutes: Positive sustained-interruption threshold, passed to
            :func:`classify_interruptions`. It classifies rows; it does not set
            the grouping window.

    Returns:
        A new DataFrame in input order. Sustained rows pass through unchanged.
        Each grouped event keeps the anchor ``event_id``, ``start_time``, and
        ``device_id``; it takes the latest ``end_time``, the elapsed
        anchor-to-end ``duration``, and the group's common customer count. The
        duration is the complete sequence span, so it may exceed the
        classification threshold. The helper sets ``grouped_momentary_event``
        True on every assembled row, so a later classification keeps the
        momentary type without a contradiction warning. Supplied
        ``device_operations`` are summed, and an absent column stays absent.

    Raises:
        InvalidArgumentError: A field is invalid, a grouped event interrupts
            different customer counts, a grouped event mixes
            ``momentary_event_excluded`` values, or an operation ends after its
            window. The exception's ``argument_name`` identifies the field.

    Warns:
        ClimateIndicesWarning: An explicit type contradicts the duration.
        OverlappingEventsWarning: Positive-duration intervals overlap.
    """
    window = _number(window_minutes, "window_minutes", positive=True)
    try:
        window_ns = int(pd.Timedelta(minutes=window).value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise _invalid("window_minutes", "must fit a pandas Timedelta") from exc
    frame = classify_interruptions(events, threshold_minutes=threshold_minutes)
    frame["grouped_momentary_event"] = False
    momentary = (frame["interruption_type"] == "momentary").to_numpy()
    if not momentary.any():
        return frame.copy()

    # Python integers prevent overflow when a deadline exceeds Timestamp.max.
    start_ns = pd.DatetimeIndex(frame["start_time"]).asi8.tolist()
    end_times = pd.DatetimeIndex(frame["end_time"])
    end_ns = end_times.asi8.tolist()
    customer_counts = frame["customers_interrupted"].tolist()
    exclusion_flags = frame["momentary_event_excluded"].tolist()
    operations = frame["device_operations"].tolist() if "device_operations" in frame else None

    device_keys = frame["device_id"].tolist() if "device_id" in frame.columns else [None] * len(frame)
    by_device: dict[object, list[int]] = {}
    for position in range(len(frame)):
        if momentary[position]:
            by_device.setdefault(device_keys[position], []).append(position)

    grouped: dict[int, int] = {}
    end_positions: list[int] = []
    durations: list[float] = []
    operation_totals: list[object] = []
    for members in _momentary_groups(by_device, start_ns, window_ns):
        anchor = members[0]
        last_end = _momentary_group_end(members, end_ns, start_ns[anchor] + window_ns, customer_counts, exclusion_flags)
        grouped[min(members)] = anchor
        end_positions.append(last_end)
        durations.append((end_ns[last_end] - start_ns[anchor]) / 60_000_000_000)
        if operations is not None:
            counts = [operations[position] for position in members]
            operation_totals.append(
                _integer(sum(int(value) for value in counts), "device_operations")
                if all(not pd.isna(value) for value in counts)
                else pd.NA
            )

    keep = sorted([position for position in range(len(frame)) if not momentary[position]] + list(grouped))
    result = frame.iloc[[grouped.get(position, position) for position in keep]].copy()
    result.index = pd.Index(keep)
    grouped_positions = list(grouped)
    result.loc[grouped_positions, "end_time"] = end_times.take(end_positions)
    result.loc[grouped_positions, "duration"] = durations
    result.loc[grouped_positions, "grouped_momentary_event"] = True
    if operations is not None:
        result.loc[grouped_positions, "device_operations"] = pd.array(operation_totals, dtype="Int64")
    return result.reset_index(drop=True)


def _normalize_events(events: EventTable, *, total_customers: float, threshold_minutes: float = 5.0) -> pd.DataFrame:
    """Validate served-customer bounds for later metric consumers."""
    total = _number(total_customers, "total_customers", positive=True)
    frame = classify_interruptions(events, threshold_minutes=threshold_minutes)
    if any(int(count) > total for count in frame["customers_interrupted"]):
        raise _invalid("customers_interrupted", "must not exceed total_customers")
    return frame


def _period_hours(
    *,
    period_hours: float | None = None,
    period_start: object = None,
    period_end: object = None,
) -> float | None:
    """Return supplied or elapsed hours, without inferring event bounds."""
    hours = None if period_hours is None else _number(period_hours, "period_hours", positive=True)
    if period_start is None and period_end is None:
        return hours
    start = _timestamps([period_start], "period_start")
    end = _timestamps([period_end], "period_end")
    if (start.tz is None) != (end.tz is None):
        raise _invalid("period_end", "timezone awareness must match period_start")
    if end[0] <= start[0]:
        raise _invalid("period_end", "must be later than period_start")
    try:
        elapsed = float((end[0] - start[0]) / pd.Timedelta(hours=1))
    except (ValueError, OverflowError) as exc:
        raise _invalid("period_end", "period must fit a pandas Timedelta") from exc
    if hours is not None and not np.isclose(hours, elapsed, rtol=0, atol=1e-9):
        raise _invalid("period_hours", "must match the explicit reporting period within 1e-9 hours")
    return elapsed

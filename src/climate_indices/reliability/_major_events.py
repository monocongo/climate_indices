"""Major Event Day thresholds and day classification (ADR-0026, decision 7)."""

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from typing import Any

import numpy as np
import pandas as pd

from climate_indices.reliability._events import _invalid, _missing, _number

_DAY_NS = pd.Timedelta(days=1).value
_EPOCH = date(1970, 1, 1)


@dataclass(frozen=True)
class MajorEventDayThreshold:
    """Frozen IEEE 1366 Beta Method threshold fitted to one daily-SAIDI history.

    ``t_med`` is ``exp(alpha + 2.5 * beta)`` in the history's daily-SAIDI units,
    normally minutes per customer per day. ``alpha`` and ``beta`` are the mean
    and sample standard deviation (``n - 1`` divisor) of the natural logarithms
    of the positive days. ``positive_days`` counts those days; zero days are in
    the history interval but not in the logarithmic sample. ``history_start``
    and ``history_end`` are the first and last calendar days of the history.
    ``timezone`` is the history's day boundary, or None for timezone-naive
    calendar days.

    A threshold applies only to days after ``history_end``. Construct one
    directly to apply a published ``t_med``; ``reliability_indices()`` uses only
    ``t_med``, ``history_end``, and ``timezone``.
    """

    t_med: float
    alpha: float
    beta: float
    history_start: date
    history_end: date
    positive_days: int
    timezone: tzinfo | None = None


def _calendar(labels: pd.Index) -> pd.DatetimeIndex:
    """Return the history's day labels as a DatetimeIndex, keeping their timezone."""
    if isinstance(labels, pd.DatetimeIndex):
        index = labels
    else:
        if any(not isinstance(label, (str, date, np.datetime64)) for label in labels):
            raise _invalid("daily_saidi", "must be indexed by calendar dates")
        try:
            index = pd.DatetimeIndex(labels)
        except (ValueError, TypeError, OverflowError) as exc:
            raise _invalid("daily_saidi", "must be indexed by parseable dates in one timezone") from exc
    if index.hasnans:
        raise _invalid("daily_saidi", "must not contain missing dates")
    return index


def major_event_day_threshold(daily_saidi: pd.Series | Mapping[Any, object]) -> MajorEventDayThreshold:
    """Fit the IEEE 1366 Major Event Day threshold to a daily-SAIDI history.

    The history holds one daily SAIDI value per calendar day. Daily SAIDI is a
    day's sustained customer-minutes divided by the served-customer count, with
    each interruption accrued to the day it begins. The IEEE history is five
    sequential years ending on the last day of the last complete reporting
    period, or all available history when fewer than five years exist. The
    caller selects that interval; the threshold applies to the next period.

    Zero days remain in the history but are excluded from the logarithmic
    sample. ``alpha`` is the mean and ``beta`` the sample standard deviation
    (``n - 1`` divisor) of the natural logarithms of the positive days, and
    ``t_med = exp(alpha + 2.5 * beta)``. A day is a Major Event Day only when its
    daily SAIDI strictly exceeds ``t_med``.

    Args:
        daily_saidi: A Series or mapping from calendar day to nonnegative daily
            SAIDI. Labels are dates, or midnight timestamps whose timezone
            declares the day boundary. Every day from the first to the last
            label must appear exactly once, in any order.

    Returns:
        A frozen ``MajorEventDayThreshold`` recording the threshold, its
        logarithmic moments, the positive-day count, the history interval, and
        the history's timezone.

    Raises:
        InvalidArgumentError: The history has a missing, duplicate, or
            non-midnight day; covers more than five years; contains a missing,
            non-numeric, negative, or infinite value; has fewer than two
            positive days; or gives a threshold that overflows float64. The
            exception's ``argument_name`` is ``daily_saidi``.
    """
    try:
        series = daily_saidi if isinstance(daily_saidi, pd.Series) else pd.Series(daily_saidi, dtype=object)
    except (ValueError, TypeError) as exc:
        raise _invalid("daily_saidi", "must be a Series or mapping indexed by calendar dates") from exc
    labels = _calendar(series.index)
    # Local wall-clock nanoseconds: midnight is a whole number of days since the epoch.
    wall = (labels.tz_localize(None) if labels.tz is not None else labels).as_unit("ns").asi8
    order = np.argsort(wall, kind="stable")
    wall = wall[order]
    if (wall % _DAY_NS != 0).any():
        raise _invalid("daily_saidi", "must label calendar days at local midnight")
    if (np.diff(wall) == 0).any():
        raise _invalid("daily_saidi", "must contain one value per calendar day")
    if (np.diff(wall) != _DAY_NS).any():
        raise _invalid("daily_saidi", "must contain every calendar day; an unknown day is not a zero day")
    first = _EPOCH + timedelta(days=int(wall[0]) // _DAY_NS) if len(wall) else _EPOCH
    last = _EPOCH + timedelta(days=int(wall[-1]) // _DAY_NS) if len(wall) else _EPOCH
    # Five years before February 29 is February 28; that year is never a leap year.
    five_years_back = last.replace(year=last.year - 5, day=28 if (last.month, last.day) == (2, 29) else last.day)
    if len(wall) and first <= five_years_back:
        raise _invalid("daily_saidi", "must not exceed five years")
    values = []
    for value in series.iloc[order].tolist():
        if _missing(value):
            raise _invalid("daily_saidi", "must not contain missing values; an unknown day is not a zero day")
        values.append(_number(value, "daily_saidi"))
    logs = [math.log(value) for value in values if value > 0]
    if len(logs) < 2:
        raise _invalid("daily_saidi", "needs at least two positive days for a sample standard deviation")
    alpha = statistics.fmean(logs)
    beta = statistics.stdev(logs)
    try:
        t_med = math.exp(alpha + 2.5 * beta)
    except OverflowError as exc:
        raise _invalid("daily_saidi", "gives a threshold that overflows float64") from exc
    return MajorEventDayThreshold(
        t_med=t_med,
        alpha=alpha,
        beta=beta,
        history_start=first,
        history_end=last,
        positive_days=len(logs),
        timezone=labels.tz,
    )


def _major_event_days(
    frame: pd.DataFrame, sustained: pd.Series, total_customers: float, threshold: object
) -> tuple[np.ndarray, tuple[date, ...]]:
    """Return the rows to keep and the ascending Major Event Days of a normalized log.

    Daily SAIDI sums the ``sustained`` rows by start day; every row that starts
    on a Major Event Day is removed.
    """
    if not isinstance(threshold, MajorEventDayThreshold):
        raise _invalid("major_event_threshold", "must be a MajorEventDayThreshold or None")
    _number(threshold.t_med, "major_event_threshold", positive=True)
    if isinstance(threshold.history_end, datetime) or not isinstance(threshold.history_end, date):
        raise _invalid("major_event_threshold", "history_end must be a date, not a timestamp")
    start = pd.DatetimeIndex(frame["start_time"])
    if len(start) and (start.tz is None) != (threshold.timezone is None):
        raise _invalid("start_time", "timezone awareness must match the Major Event Day history")
    if start.tz is not None:
        start = start.tz_convert(threshold.timezone)
    days = [timestamp.date() for timestamp in start]
    if any(day <= threshold.history_end for day in days):
        raise _invalid("start_time", "must follow the Major Event Day history; fit the threshold to earlier days")
    minutes: dict[date, list[float]] = {}
    products = (frame["duration"] * frame["customers_interrupted"]).tolist()
    for day, product, counted in zip(days, products, sustained.tolist(), strict=True):
        if counted:
            minutes.setdefault(day, []).append(product)
    major = {day for day, values in minutes.items() if math.fsum(values) / total_customers > threshold.t_med}
    return np.array([day not in major for day in days], dtype=bool), tuple(sorted(major))

"""Contract tests for Major Event Day thresholds and opt-in reliability filtering."""

import dataclasses
import math
import random
from datetime import date, timedelta
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import pytest

from climate_indices import reliability
from climate_indices.exceptions import InvalidArgumentError

# IEEE Std 1366-2012 clause 3.5.1, Table 2: daily SAIDI for December 1993.
# December 18 is the zero day that the example excludes from the logarithms.
DECEMBER_1993 = [
    26.974, 0.956, 0.131, 1.292, 4.250, 0.119, 0.130, 12.883, 0.226, 13.864, 0.015,
    1.788, 0.410, 0.007, 1.124, 1.951, 0.329, 0.0, 0.281, 1.810, 0.250, 0.021,
    1.233, 0.996, 0.162, 0.288, 0.535, 0.291, 0.600, 1.750, 3.622,
]  # fmt: skip
# IEEE Std 1366-2012 clause 3.5.1, Table 3: daily SAIDI for January 1994.
JANUARY_1994 = [
    0.240, 0.014, 0.075, 2.649, 0.666, 0.189, 0.009, 1.117, 0.111, 8.683, 0.277,
    0.057, 0.974, 0.150, 0.633, 0.434, 5.700, 0.109, 0.259, 1.142, 0.262, 0.044,
    0.243, 5.932, 2.698, 5.894, 0.408, 237.493, 2.730, 8.110, 0.046,
]  # fmt: skip
# One 600-minute event per day over 600 000 customers gives daily SAIDI N / 1000.
TABLE_CUSTOMERS = 600_000


def history(values: list[float], start: str = "2023-01-01", tz: str | None = None) -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D", tz=tz))


def threshold(t_med: float | np.floating[Any], history_end: date = date(2023, 12, 31), **changes: Any) -> Any:
    fields = {
        "t_med": t_med,
        "alpha": math.log(t_med),
        "beta": 0.0,
        "history_start": date(2023, 1, 1),
        "history_end": history_end,
        "positive_days": 365,
        **changes,
    }
    return reliability.MajorEventDayThreshold(**fields)


def event(event_id: str, start_time: str, duration: float, customers: int, **changes: object) -> dict[str, object]:
    return {
        "event_id": event_id,
        "start_time": start_time,
        "duration": duration,
        "customers_interrupted": customers,
        **changes,
    }


def january_1994_log() -> list[dict[str, object]]:
    return [
        event(f"jan-{day}", f"1994-01-{day:02d}T06:00:00", 600.0, round(saidi * 1000))
        for day, saidi in enumerate(JANUARY_1994, start=1)
    ]


def test_threshold_matches_the_hand_computed_history() -> None:
    result = reliability.major_event_day_threshold(history([0.0, math.exp(-1), 1.0, math.e, 0.0]))
    # The logarithms are [-1, 0, 1]: mean zero and sample standard deviation one.
    # A few float64 steps of log/exp roundoff are allowed; the population
    # convention would give exp(2.5 * sqrt(2 / 3)) ~= 7.70 instead.
    assert result.t_med == pytest.approx(math.exp(2.5), rel=1e-15, abs=0)
    assert result.alpha == pytest.approx(0.0, abs=1e-15)
    assert result.beta == pytest.approx(1.0, rel=1e-15, abs=0)
    assert result.positive_days == 3
    assert (result.history_start, result.history_end) == (date(2023, 1, 1), date(2023, 1, 5))
    assert result.timezone is None


def test_zero_days_stay_out_of_the_logarithmic_sample() -> None:
    with_zeros = reliability.major_event_day_threshold(history([0.0, math.exp(-1), 1.0, math.e, 0.0]))
    without_zeros = reliability.major_event_day_threshold(history([math.exp(-1), 1.0, math.e]))
    assert (with_zeros.t_med, with_zeros.alpha, with_zeros.beta) == (
        without_zeros.t_med,
        without_zeros.alpha,
        without_zeros.beta,
    )
    assert with_zeros.positive_days == without_zeros.positive_days == 3


def test_ieee_2012_worked_example_uses_the_sample_standard_deviation() -> None:
    result = reliability.major_event_day_threshold(history(DECEMBER_1993, start="1993-12-01"))
    assert result.positive_days == 30
    # The guide prints alpha -0.555 and beta 1.90; the population estimator would print 1.87.
    assert round(result.alpha, 3) == -0.555
    assert round(result.beta, 2) == 1.90
    # The guide prints 66.69 = exp(4.20) from rounded intermediates; full precision gives 67.10.
    assert round(result.t_med, 2) == 67.10


def test_ieee_2012_worked_example_removes_january_28_through_the_shared_path() -> None:
    med = reliability.major_event_day_threshold(history(DECEMBER_1993, start="1993-12-01"))
    events = january_1994_log()
    unfiltered = reliability.reliability_indices(events, total_customers=TABLE_CUSTOMERS)
    filtered = reliability.reliability_indices(events, total_customers=TABLE_CUSTOMERS, major_event_threshold=med)
    # Eq. 31 and Eq. 32 print 287.35 and 49.86 from these daily sums.
    assert unfiltered.saidi == pytest.approx(287.348, rel=0, abs=1e-12)
    assert filtered.saidi == pytest.approx(49.855, rel=0, abs=1e-12)
    assert filtered.major_event_days == (date(1994, 1, 28),)
    assert filtered.major_event_threshold == med
    assert filtered.outage_count == unfiltered.outage_count - 1
    assert filtered.customer_interruptions == unfiltered.customer_interruptions - 237_493


def test_filtering_is_opt_in_and_the_default_records_no_filter() -> None:
    events = january_1994_log()
    default = reliability.reliability_indices(events, total_customers=TABLE_CUSTOMERS)
    explicit = reliability.reliability_indices(events, total_customers=TABLE_CUSTOMERS, major_event_threshold=None)
    assert default == explicit
    assert default.major_event_threshold is None
    assert default.major_event_days == ()
    assert reliability.ReliabilityResult(0.0, 0, 0).major_event_threshold is None


@pytest.mark.parametrize(
    "t_med, expected_days",
    [(6.0, ()), (math.nextafter(6.0, 0.0), (date(2024, 1, 2),))],
    ids=["equal-is-not-major", "one-step-below-is-major"],
)
def test_classification_uses_a_strict_comparison(t_med: float, expected_days: tuple[date, ...]) -> None:
    # 60 minutes x 10 customers / 100 customers is exactly 6.0 daily SAIDI.
    events = [event("a", "2024-01-02T08:00:00", 60.0, 10)]
    result = reliability.reliability_indices(events, total_customers=100, major_event_threshold=threshold(t_med))
    assert result.major_event_days == expected_days
    assert result.outage_count == (0 if expected_days else 1)


def test_numpy_threshold_keeps_float64_comparison_semantics() -> None:
    # A NumPy threshold scalar must not round the daily SAIDI it is compared with.
    events = [event("a", "2024-01-02T08:00:00", math.nextafter(6.0, math.inf), 1)]
    result = reliability.reliability_indices(
        events, total_customers=1, major_event_threshold=threshold(np.float32(6.0))
    )
    assert result.major_event_days == (date(2024, 1, 2),)
    assert result.outage_count == 0


def test_filtering_removes_every_row_on_a_major_event_day_through_the_shared_path() -> None:
    events = [
        event("big", "2024-01-02T00:00:00", 600.0, 50),
        event("planned", "2024-01-02T10:00:00", 30.0, 5, planned=True),
        event("blink", "2024-01-02T11:00:00", 1.0, 20, device_operations=2),
        event("next-day", "2024-01-03T08:00:00", 60.0, 10, device_operations=0),
        event("next-blink", "2024-01-03T10:00:00", 2.0, 15, device_operations=3),
    ]
    filtered = reliability.reliability_indices(
        events, total_customers=100, period_hours=48, major_event_threshold=threshold(10.0)
    )
    manual = reliability.reliability_indices(events[3:], total_customers=100, period_hours=48)
    assert filtered.major_event_days == (date(2024, 1, 2),)
    assert dataclasses.replace(filtered, major_event_threshold=None, major_event_days=()) == manual
    assert (filtered.maifi_e, filtered.maifi, filtered.momentary_outage_count) == (0.15, 0.45, 1)
    # The reporting exposure is unchanged; filtered ASAI is not rescaled.
    assert filtered.period_hours == 48.0
    assert filtered.asai == manual.asai


def test_daily_saidi_sums_every_sustained_row_of_one_day() -> None:
    # Two rows of 40 daily SAIDI each; neither alone exceeds 50, and together they do.
    events = [event("a", "2024-01-02T00:00:00", 40.0, 100), event("b", "2024-01-02T01:00:00", 40.0, 100)]
    result = reliability.reliability_indices(events, total_customers=100, major_event_threshold=threshold(50.0))
    assert result.major_event_days == (date(2024, 1, 2),)
    assert result.outage_count == 0
    assert result.saidi == 0.0


def test_multiple_major_event_days_are_listed_in_ascending_order() -> None:
    events = [event("later", "2024-01-05T00:00:00", 60.0, 100), event("earlier", "2024-01-02T00:00:00", 60.0, 100)]
    result = reliability.reliability_indices(events, total_customers=100, major_event_threshold=threshold(50.0))
    assert result.major_event_days == (date(2024, 1, 2), date(2024, 1, 5))


def test_history_ending_on_a_leap_day() -> None:
    end = date(2024, 2, 29)
    accept = pd.date_range("2019-03-01", "2024-02-29", freq="D")
    accepted = reliability.major_event_day_threshold(
        pd.Series([1.0 + (offset % 7) for offset in range(len(accept))], index=accept)
    )
    assert (accepted.history_start, accepted.history_end) == (date(2019, 3, 1), end)
    reject = pd.date_range("2019-02-28", "2024-02-29", freq="D")
    too_long = pd.Series([1.0] * len(reject), index=reject)
    with pytest.raises(InvalidArgumentError) as error:
        reliability.major_event_day_threshold(too_long)
    assert error.value.argument_name == "daily_saidi"


def test_out_of_range_dates_are_rejected() -> None:
    microsecond_days = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("3000-01-01", periods=3))
    with pytest.raises(InvalidArgumentError) as error:
        reliability.major_event_day_threshold(microsecond_days)
    assert error.value.argument_name == "daily_saidi"
    object_days = pd.Series([1.0, 2.0], index=[date(3000, 1, 1), date(3000, 1, 2)])
    with pytest.raises(InvalidArgumentError) as error:
        reliability.major_event_day_threshold(object_days)
    assert error.value.argument_name == "daily_saidi"


def test_threshold_timezone_must_be_a_tzinfo() -> None:
    med = threshold(10.0)
    for timezone in ["Not/AZone", 3600, object()]:
        invalid = dataclasses.replace(med, timezone=timezone)
        with pytest.raises(InvalidArgumentError) as error:
            reliability.reliability_indices([], total_customers=100, major_event_threshold=invalid)
        assert error.value.argument_name == "major_event_threshold"


def test_multi_day_interruption_accrues_to_its_start_day() -> None:
    # 600 minutes x 10 customers / 100 customers = 60 daily SAIDI, all on January 2.
    events = [event("overnight", "2024-01-02T23:00:00", 600.0, 10), event("small", "2024-01-03T10:00:00", 10.0, 1)]
    result = reliability.reliability_indices(events, total_customers=100, major_event_threshold=threshold(50.0))
    assert result.major_event_days == (date(2024, 1, 2),)
    assert result.outage_count == 1


def test_daily_saidi_uses_the_same_planned_outage_mask() -> None:
    events = [event("planned", "2024-01-02T08:00:00", 600.0, 10, planned=True)]
    med = threshold(50.0)
    included = reliability.reliability_indices(events, total_customers=100, major_event_threshold=med)
    excluded = reliability.reliability_indices(
        events, total_customers=100, exclude_planned=True, major_event_threshold=med
    )
    assert included.major_event_days == (date(2024, 1, 2),)
    assert excluded.major_event_days == ()


def test_momentary_rows_do_not_make_a_major_event_day() -> None:
    events = [event("blink", "2024-01-02T08:00:00", 5.0, 100)]
    result = reliability.reliability_indices(events, total_customers=100, major_event_threshold=threshold(1e-9))
    assert result.major_event_days == ()
    assert result.momentary_outage_count == 1


def test_empty_log_records_the_filter_without_major_event_days() -> None:
    med = threshold(10.0)
    result = reliability.reliability_indices([], total_customers=100, major_event_threshold=med)
    assert result.major_event_threshold == med
    assert result.major_event_days == ()


def test_timezone_aware_history_assigns_local_calendar_days() -> None:
    med = reliability.major_event_day_threshold(history([1.0, 2.0, 3.0], start="2023-12-29", tz="America/New_York"))
    assert med.history_end == date(2023, 12, 31)
    # 03:00 UTC on January 3 is 22:00 on January 2 in New York.
    events = [event("late", "2024-01-03T03:00:00Z", 600.0, 10)]
    result = reliability.reliability_indices(events, total_customers=100, major_event_threshold=med)
    assert result.major_event_days == (date(2024, 1, 2),)


@pytest.mark.parametrize(
    "history_tz, start_time",
    [("America/New_York", "2024-01-02T08:00:00"), (None, "2024-01-02T08:00:00Z")],
    ids=["aware-history-naive-events", "naive-history-aware-events"],
)
def test_event_timezone_awareness_must_match_the_history(history_tz: str | None, start_time: str) -> None:
    med = reliability.major_event_day_threshold(history([1.0, 2.0], start="2023-12-30", tz=history_tz))
    events = [event("a", start_time, 60.0, 1)]
    with pytest.raises(InvalidArgumentError) as error:
        reliability.reliability_indices(events, total_customers=100, major_event_threshold=med)
    assert error.value.argument_name == "start_time"


@pytest.mark.parametrize("start_time", ["2023-12-31T23:59:00", "2023-06-01T00:00:00"])
def test_events_must_follow_the_threshold_history(start_time: str) -> None:
    events = [event("a", start_time, 60.0, 1)]
    med = threshold(10.0)
    with pytest.raises(InvalidArgumentError) as error:
        reliability.reliability_indices(events, total_customers=100, major_event_threshold=med)
    assert error.value.argument_name == "start_time"


@pytest.mark.parametrize(
    "value",
    [10.0, "threshold", {"t_med": 10.0}],
)
def test_major_event_threshold_must_be_a_threshold_object(value: Any) -> None:
    with pytest.raises(InvalidArgumentError) as error:
        reliability.reliability_indices([], total_customers=100, major_event_threshold=value)
    assert error.value.argument_name == "major_event_threshold"


@pytest.mark.parametrize(
    "changes",
    [
        {"t_med": 0.0},
        {"t_med": -1.0},
        {"t_med": math.inf},
        {"t_med": math.nan},
        {"history_end": "2023-12-31"},
        {"history_end": pd.Timestamp("2023-12-31")},
    ],
)
def test_directly_constructed_threshold_is_validated(changes: dict[str, object]) -> None:
    fields: dict[str, Any] = {"t_med": 10.0, "alpha": 0.0, "beta": 0.0, "history_start": date(2023, 1, 1)}
    fields.update({"history_end": date(2023, 12, 31), "positive_days": 2, **changes})
    med = reliability.MajorEventDayThreshold(**fields)
    with pytest.raises(InvalidArgumentError) as error:
        reliability.reliability_indices([], total_customers=100, major_event_threshold=med)
    assert error.value.argument_name == "major_event_threshold"


def test_threshold_is_frozen_and_hashable() -> None:
    med = reliability.major_event_day_threshold(history([1.0, 2.0]))
    with pytest.raises(dataclasses.FrozenInstanceError):
        med.t_med = 1.0  # type: ignore[misc]
    filtered = reliability.reliability_indices([], total_customers=100, major_event_threshold=med)
    assert hash(filtered) == hash(dataclasses.replace(filtered))


def test_row_order_does_not_change_the_threshold() -> None:
    values = [0.0, *DECEMBER_1993[:20]]
    series = history(values)
    shuffled = series.sample(frac=1.0, random_state=random.Random(7).randrange(2**32))
    assert reliability.major_event_day_threshold(shuffled) == reliability.major_event_day_threshold(series)


def test_mapping_history_matches_series_history() -> None:
    days = [date(2023, 1, 1) + timedelta(days=offset) for offset in range(3)]
    mapping = dict(zip(days, [0.5, 0.0, 2.0], strict=True))
    assert reliability.major_event_day_threshold(mapping) == reliability.major_event_day_threshold(
        history([0.5, 0.0, 2.0])
    )


def test_constant_positive_history_has_zero_beta() -> None:
    result = reliability.major_event_day_threshold(history([2.0, 0.0, 2.0]))
    assert result.beta == 0.0
    # exp(log(2)) may differ from 2.0 by one rounding step.
    assert result.t_med == pytest.approx(2.0, rel=4 * np.finfo(np.float64).eps, abs=0)


def test_shorter_history_is_accepted_and_recorded() -> None:
    result = reliability.major_event_day_threshold(history([1.0, 3.0], start="2023-12-30"))
    assert (result.history_start, result.history_end, result.positive_days) == (
        date(2023, 12, 30),
        date(2023, 12, 31),
        2,
    )


def test_five_years_is_the_longest_accepted_history() -> None:
    days = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    values = [1.0 + (offset % 7) for offset in range(len(days))]
    accepted = reliability.major_event_day_threshold(pd.Series(values, index=days))
    assert (accepted.history_start, accepted.history_end) == (date(2019, 1, 1), date(2023, 12, 31))
    longer = pd.Series([1.0, *values], index=pd.date_range("2018-12-31", "2023-12-31", freq="D"))
    with pytest.raises(InvalidArgumentError) as error:
        reliability.major_event_day_threshold(longer)
    assert error.value.argument_name == "daily_saidi"


@pytest.mark.parametrize(
    "values",
    [
        [1.0, None, 2.0],
        [1.0, math.nan, 2.0],
        [1.0, pd.NA, 2.0],
        [1.0, np.ma.masked, 2.0],
        [1.0, math.inf, 2.0],
        [1.0, -math.inf, 2.0],
        [1.0, -0.5, 2.0],
        [1.0, Fraction(-1, 10**400), 2.0],
        [1.0, True, 2.0],
        [1.0, "2.0", 2.0],
        [],
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [1e-300, 1e300],
    ],
    ids=[
        "none",
        "nan",
        "pd-na",
        "masked",
        "inf",
        "negative-inf",
        "negative",
        "negative-underflow",
        "boolean",
        "string",
        "empty",
        "all-zero",
        "one-positive-day",
        "threshold-overflow",
    ],
)
def test_invalid_history_values_are_rejected(values: list[object]) -> None:
    series = pd.Series(values, index=pd.date_range("2023-01-01", periods=len(values), freq="D"), dtype=object)
    with pytest.raises(InvalidArgumentError) as error:
        reliability.major_event_day_threshold(series)
    assert error.value.argument_name == "daily_saidi"


@pytest.mark.parametrize(
    "daily_saidi",
    [
        pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-04"])),
        pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-02"])),
        pd.Series([1.0, 2.0], index=pd.to_datetime(["2023-01-01T00:00", "2023-01-02T06:00"])),
        pd.Series([1.0, 2.0], index=pd.to_datetime(["2023-01-01", None])),
        pd.Series([1.0, 2.0], index=["2023-01-01", "not-a-date"]),
        pd.Series([1.0, 2.0]),
    ],
    ids=["missing-date", "duplicate-date", "not-midnight", "missing-label", "unparseable-label", "integer-index"],
)
def test_invalid_history_calendars_are_rejected(daily_saidi: Any) -> None:
    with pytest.raises(InvalidArgumentError) as error:
        reliability.major_event_day_threshold(daily_saidi)
    assert error.value.argument_name == "daily_saidi"


@pytest.mark.parametrize(
    "daily_saidi", [None, [1.0, 2.0], pd.DataFrame({"a": [1.0, 2.0]})], ids=["none", "list", "frame"]
)
def test_history_must_be_a_series_or_mapping(daily_saidi: Any) -> None:
    with pytest.raises(InvalidArgumentError) as error:
        reliability.major_event_day_threshold(daily_saidi)
    assert error.value.argument_name == "daily_saidi"
    assert "Series or mapping" in str(error.value)


def test_public_names_are_exported() -> None:
    assert "major_event_day_threshold" in reliability.__all__
    assert "MajorEventDayThreshold" in reliability.__all__

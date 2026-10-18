"""Tests for ``compound.harmonize()`` and ``AlignedHazards`` (CMP-03, ADR-0022).

Outputs are Boolean masks and integer counts, so every comparison is exact: no tolerance
applies. The daily-to-monthly reduction is checked against a brute-force calendar loop that
shares no code with the implementation.
"""

from __future__ import annotations

import calendar
import copy
import dataclasses
import datetime as dt
import json
import logging
import pickle
import warnings
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import climate_indices
from climate_indices import compound
from climate_indices.compound import _harmonize as harmonize_module
from climate_indices.exceptions import (
    ClimateIndicesWarning,
    CoordinateValidationError,
    DataShapeError,
    InputAlignmentWarning,
    InputTypeError,
    InvalidArgumentError,
)


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def months(start: str, stop: str) -> np.ndarray:
    """Return monthly labels from ``start`` up to, excluding, ``stop`` (``YYYY-MM``)."""
    return np.arange(start, stop, dtype="datetime64[M]")


def days(start: str, stop: str) -> np.ndarray:
    """Return daily labels from ``start`` up to, excluding, ``stop``."""
    return np.arange(start, stop, dtype="datetime64[D]")


def per_source(names: list[str], value: Any) -> dict[str, Any]:
    """Broadcast one declaration to every source unless the caller already keyed it."""
    return dict(value) if isinstance(value, dict) else dict.fromkeys(names, value)


def run(
    inputs: dict[str, Any],
    time: dict[str, Any] | np.ndarray,
    *,
    cadence: Any = "monthly",
    scale: Any = "probability",
    threshold: Any = 0.2,
    direction: Any = "below",
    target: str = "monthly",
    calibrated: bool = True,
    **extra: Any,
) -> compound.AlignedHazards:
    """Call ``harmonize`` with one declaration per source, a shared calibration period by default."""
    names = list(inputs)
    kwargs: dict[str, Any] = {
        "time": per_source(names, time),
        "cadence": per_source(names, cadence),
        "input_scale": per_source(names, scale),
        "thresholds": per_source(names, threshold),
        "directions": per_source(names, direction),
        "target_cadence": target,
    }
    if calibrated and "source_metadata" not in extra:
        kwargs["source_metadata"] = {
            n: {"calibration_year_initial": 1991, "calibration_year_final": 2020}
            for n in names
            if per_source(names, scale)[n] == "probability"
        }
    kwargs.update(extra)
    return compound.harmonize(inputs, **kwargs)


def ints(array: np.ndarray) -> list[Any]:
    """Return a Boolean array as nested 0/1 lists for compact expected values."""
    return array.astype(int).tolist()


# --- namespace -----------------------------------------------------------------------------


def test_namespace_is_not_re_exported_from_the_root() -> None:
    """Import explicitly: neither the module nor its functions join the package root."""
    assert "compound" not in climate_indices.__all__
    assert not hasattr(climate_indices, "harmonize")
    assert sorted(compound.__all__) == ["AlignedHazards", "harmonize"]


# --- thresholds, dtypes, and unknown values --------------------------------------------------


@pytest.mark.parametrize(
    ("threshold", "direction", "expected"),
    [
        (0.5, "below", [1, 1, 0, 0, 0]),  # the tie at 0.5 is not below
        (0.5, "above", [0, 0, 0, 1, 1]),
        (0.0, "below", [0, 0, 0, 0, 0]),  # nothing is strictly below 0
        (1.0, "above", [0, 0, 0, 0, 0]),  # nothing is strictly above 1
        (0.0, "above", [0, 1, 1, 1, 1]),  # the zero-mass endpoint 0 is not above 0
        (1.0, "below", [1, 1, 1, 1, 0]),  # the endpoint 1 is not below 1
    ],
)
def test_comparisons_are_strict_including_endpoints(threshold: float, direction: str, expected: list[int]) -> None:
    """Use ``>`` and ``<`` only, and keep ties and endpoint mass unchanged."""
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = run({"p": values}, months("2000-01", "2000-06"), threshold=threshold, direction=direction)
    assert ints(result.exceedance[0]) == expected
    assert ints(result.valid[0]) == [1] * 5


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1, 0, 1], dtype=np.int8),
        np.array([0, 1, 0, 1], dtype=np.uint16),
        np.array([0.0, 1.0, 0.25, 0.75], dtype=np.float16),
        np.array([0.0, 1.0, 0.25, 0.75], dtype=np.float32),
        np.array([0.0, 1.0, 0.25, 0.75], dtype=np.float64),
        np.array([0.0, 9.0, 1.0, 9.0, 0.25, 9.0, 0.75, 9.0])[::2],  # strided view
        [0.0, 1.0, 0.25, 0.75],  # plain list
    ],
    ids=["int8", "uint16", "float16", "float32", "float64", "strided", "list"],
)
def test_supported_probability_dtypes_and_layouts(values: Any) -> None:
    """Accept real numeric dtypes, views, and lists, widening exactly before comparing."""
    result = run({"p": values}, months("2000-01", "2000-05"), threshold=0.5, direction="below")
    assert ints(result.exceedance[0]) == [1, 0, 1, 0]


def test_float32_is_widened_exactly_not_rounded_to_the_threshold() -> None:
    """A float32 value printed as the threshold is not equal to it; float64 ties stay ties."""
    time = months("2000-01", "2000-02")
    declared = {"threshold": 0.1, "direction": "above"}
    assert ints(run({"p": np.array([0.1], dtype=np.float32)}, time, **declared).exceedance[0]) == [1]
    assert ints(run({"p": np.array([0.1], dtype=np.float64)}, time, **declared).exceedance[0]) == [0]


@pytest.mark.skipif(
    np.dtype(np.longdouble).itemsize == np.dtype(np.float64).itemsize,
    reason="this platform's longdouble is float64, so no wider value exists to narrow",
)
def test_extended_precision_probability_is_rejected_rather_than_narrowed() -> None:
    """Refuse a dtype float64 cannot hold exactly instead of validating a rounded copy of it."""
    time = months("2000-01", "2000-02")
    above_one = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
    assert above_one > 1  # float64 rounds this to exactly 1.0, which the range gate would accept
    with pytest.raises(InputTypeError, match="float64"):
        run({"p": np.array([above_one], dtype=np.longdouble)}, time)


def test_nan_and_masked_values_are_invalid_not_false() -> None:
    """Keep unknown observations unknown, and ignore whatever hides under a mask."""
    values = np.ma.array([0.1, np.nan, 0.9, 5.0, 0.05], mask=[False, False, False, True, False])
    result = run({"p": values}, months("2000-01", "2000-06"), threshold=0.2)
    assert ints(result.exceedance[0]) == [1, 0, 0, 0, 1]
    assert ints(result.valid[0]) == [1, 0, 1, 0, 1]


def test_explicit_validity_is_combined_with_nan_validity() -> None:
    """A ``validity`` mask marks unobserved positions even where a value exists."""
    result = run(
        {"p": np.array([0.1, 0.1, np.nan, 0.1])},
        months("2000-01", "2000-05"),
        validity={"p": np.array([True, False, True, True])},
    )
    assert ints(result.valid[0]) == [1, 0, 0, 1]
    assert ints(result.exceedance[0]) == [1, 0, 0, 1]


def test_indicator_sources_use_their_boolean_values_and_validity() -> None:
    """Take a Boolean indicator as occurrence; a False indicator and an unknown day differ."""
    indicator = np.array([True, False, True, True])
    result = run(
        {"heat": indicator},
        months("2000-01", "2000-05"),
        scale="indicator",
        threshold=0.5,
        direction="above",
        validity={"heat": np.array([True, True, False, True])},
    )
    assert ints(result.exceedance[0]) == [1, 0, 0, 1]
    assert ints(result.valid[0]) == [1, 1, 0, 1]


def test_masked_indicator_values_are_invalid() -> None:
    """A masked Boolean element is unknown."""
    indicator = np.ma.array([True, True, False], mask=[False, True, False])
    result = run({"h": indicator}, months("2000-01", "2000-04"), scale="indicator", threshold=0.5, direction="above")
    assert ints(result.exceedance[0]) == [1, 0, 0]
    assert ints(result.valid[0]) == [1, 0, 1]


def test_exceedance_is_false_wherever_invalid() -> None:
    """Hold the invariant that the arrays encode: ``exceedance`` implies ``valid``."""
    rng = np.random.default_rng(1367)
    values = rng.random((2, 60))
    values[rng.random((2, 60)) < 0.2] = np.nan
    result = run({"a": values[0], "b": values[1]}, months("2000-01", "2005-01"), threshold=0.5)
    assert not np.any(result.exceedance & ~result.valid)
    assert np.any(~result.valid)


def test_all_missing_cell_does_not_affect_neighbouring_cells() -> None:
    """Return an all-invalid cell and leave the other cells correct."""
    values = np.full((4, 3), 0.1)
    values[:, 1] = np.nan
    result = run({"p": values}, months("2000-01", "2000-05"), spatial_time_major=True)
    assert result.exceedance.shape == (1, 4, 3)
    assert ints(result.valid[0, :, 1]) == [0, 0, 0, 0]
    assert ints(result.exceedance[0, :, 0]) == [1, 1, 1, 1]
    assert ints(result.exceedance[0, :, 2]) == [1, 1, 1, 1]


def test_gridded_input_matches_single_cell_runs() -> None:
    """Treat each trailing-axis cell as an independent series, including a 12-long cell axis."""
    rng = np.random.default_rng(7)
    grid = rng.random((24, 12))
    time = months("2000-01", "2002-01")
    result = run({"p": grid}, time, spatial_time_major=True, threshold=0.3)
    assert result.exceedance.shape == (1, 24, 12)
    for cell in range(12):
        single = run({"p": grid[:, cell]}, time, threshold=0.3)
        assert np.array_equal(result.exceedance[0, :, cell], single.exceedance[0])
        assert np.array_equal(result.valid[0, :, cell], single.valid[0])


def test_sources_keep_input_order_and_are_independent_of_permutation() -> None:
    """Order the leading axis by input order; permuting inputs permutes rows only."""
    time = months("2000-01", "2000-06")
    a, b = np.array([0.1, 0.9, 0.1, 0.9, 0.1]), np.array([0.9, 0.9, 0.1, 0.1, 0.1])
    ab = run({"a": a, "b": b}, time)
    ba = run({"b": b, "a": a}, time)
    assert ab.names == ("a", "b")
    assert ba.names == ("b", "a")
    assert np.array_equal(ab.exceedance, ba.exceedance[::-1])


# --- dates, labels, and leap days -------------------------------------------------------------


def test_monthly_labels_normalize_to_the_start_of_the_month() -> None:
    """Treat month-start, mid-month, and month-end labels as the same calendar month."""
    labels = np.array(["2000-01-31", "2000-02-15", "2000-03-01"], dtype="datetime64[D]")
    result = run({"p": np.array([0.1, 0.1, 0.1])}, labels)
    assert result.time.tolist() == [dt.date(2000, 1, 1), dt.date(2000, 2, 1), dt.date(2000, 3, 1)]
    assert result.time.dtype == np.dtype("datetime64[D]")
    assert result.metadata["sources"]["p"]["time_start"] == "2000-01"


def test_daily_labels_floor_to_civil_days_and_accept_other_datetime_units() -> None:
    """Floor sub-day timestamps to the civil day, including the pandas ns path."""
    stamps = pd.date_range("2000-03-01 12:00", periods=3, freq="D").to_numpy()
    result = run({"p": np.array([0.1, 0.9, 0.1])}, stamps, cadence="daily", target="daily")
    assert result.time.tolist() == [dt.date(2000, 3, 1), dt.date(2000, 3, 2), dt.date(2000, 3, 3)]


def test_real_leap_day_is_an_observation_and_a_missing_one_is_not_invented() -> None:
    """Count the real 29 February; never insert one into a non-leap year."""
    leap = run({"p": np.full(3, 0.1)}, days("2000-02-28", "2000-03-02"), cadence="daily", target="daily")
    assert leap.time.tolist() == [dt.date(2000, 2, 28), dt.date(2000, 2, 29), dt.date(2000, 3, 1)]
    common = run({"p": np.full(2, 0.1)}, days("2001-02-28", "2001-03-02"), cadence="daily", target="daily")
    assert common.time.tolist() == [dt.date(2001, 2, 28), dt.date(2001, 3, 1)]


def test_interior_absent_dates_stay_in_place_as_invalid_positions() -> None:
    """Reindex to the complete cadence instead of closing the gap."""
    observed = np.array(["2000-01-01", "2000-01-02", "2000-01-05"], dtype="datetime64[D]")
    result = run({"p": np.array([0.1, 0.1, 0.1])}, observed, cadence="daily", target="daily")
    assert result.time.tolist() == [dt.date(2000, 1, d) for d in range(1, 6)]
    assert ints(result.valid[0]) == [1, 1, 0, 0, 1]
    assert ints(result.exceedance[0]) == [1, 1, 0, 0, 1]

    skipped = np.array(["2000-01", "2000-02", "2000-05"], dtype="datetime64[M]")
    monthly = run({"p": np.array([0.1, 0.1, 0.1])}, skipped)
    assert ints(monthly.valid[0]) == [1, 1, 0, 0, 1]


@pytest.mark.parametrize(
    ("time", "reason"),
    [
        (np.array(["2000-01-01", "NaT", "2000-01-03"], dtype="datetime64[D]"), "NaT"),
        (np.array(["2000-01-02", "2000-01-02", "2000-01-03"], dtype="datetime64[D]"), "duplicate"),
        (np.array(["2000-01-03", "2000-01-02", "2000-01-04"], dtype="datetime64[D]"), "unordered"),
    ],
    ids=["nat", "duplicate", "unordered"],
)
def test_invalid_dates_are_rejected(time: np.ndarray, reason: str) -> None:
    """Reject NaT, duplicate, and unordered coordinates."""
    with pytest.raises(CoordinateValidationError) as caught:
        run({"p": np.full(3, 0.1)}, time, cadence="daily", target="daily")
    assert caught.value.reason == reason


def test_two_observations_in_one_month_are_rejected() -> None:
    """Reject duplicate monthly observations even when their civil dates differ."""
    time = np.array(["2000-01-05", "2000-01-25", "2000-02-01"], dtype="datetime64[D]")
    with pytest.raises(CoordinateValidationError, match="calendar month"):
        run({"p": np.full(3, 0.1)}, time)


@pytest.mark.parametrize(
    "time",
    [
        np.array(
            [dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(2000, 2, 1, tzinfo=dt.timezone.utc)],
            dtype=object,
        ),
        pd.date_range("2000-01-01", periods=2, freq="MS", tz="UTC"),
        np.array(["2000-01", "2000-02"]),
        np.array([1, 2]),
        np.array([dt.date(2000, 1, 1), dt.date(2000, 2, 1)], dtype=object),  # what a cftime/360-day array looks like
    ],
    ids=["tz-aware-object", "tz-aware-index", "strings", "integers", "objects"],
)
def test_non_datetime64_dates_are_rejected(time: Any) -> None:
    """Reject timezone-aware, string, numeric, and object (cftime-like) dates; never convert calendars."""
    with pytest.raises(InputTypeError, match="datetime64"):
        run({"p": np.full(2, 0.1)}, time)


def test_dates_must_match_the_time_axis() -> None:
    """Require one date per time step, in one dimension."""
    short_inputs, short_time = {"p": np.full(3, 0.1)}, months("2000-01", "2000-03")
    with pytest.raises(DataShapeError):
        run(short_inputs, short_time)
    grid_inputs, grid_time = {"p": np.full(4, 0.1)}, months("2000-01", "2000-05").reshape(2, 2)
    with pytest.raises(DataShapeError):
        run(grid_inputs, grid_time)


# --- daily-to-monthly reduction ---------------------------------------------------------------


def _reference_months(
    dates: list[dt.date], values: np.ndarray, threshold: float, aggregation: str
) -> tuple[list[tuple[int, int]], list[bool], list[bool]]:
    """Reduce daily probabilities to months with plain loops and the calendar module."""
    observed = {d: float(v) for d, v in zip(dates, values, strict=True)}
    first, last = (dates[0].year, dates[0].month), (dates[-1].year, dates[-1].month)
    keys, hazard, valid = [], [], []
    year, month = first
    while (year, month) <= last:
        month_days = [dt.date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
        complete = all(d in observed and not np.isnan(observed[d]) for d in month_days)
        flags = [observed[d] < threshold for d in month_days if d in observed and not np.isnan(observed[d])]
        reduced = complete and (any(flags) if aggregation == "any" else all(flags))
        keys.append((year, month))
        hazard.append(reduced)
        valid.append(complete)
        year, month = (year, month + 1) if month < 12 else (year + 1, 1)
    return keys, hazard, valid


@pytest.mark.parametrize("aggregation", ["any", "all"])
def test_daily_reduction_matches_a_brute_force_calendar_loop(aggregation: str) -> None:
    """Match an independent loop over three years, a leap year, gaps, NaNs, and partial end months."""
    rng = np.random.default_rng(1367)
    grid = days("2000-01-10", "2002-11-16")
    keep = rng.random(grid.shape[0]) < 0.997
    values = np.where(rng.random(grid.shape[0]) < 0.3, 0.05, rng.random(grid.shape[0]))
    values[rng.random(grid.shape[0]) < 0.002] = np.nan
    values[(grid >= np.datetime64("2000-04-01")) & (grid < np.datetime64("2000-05-01"))] = 0.9  # dry month for "all"
    wet = (grid >= np.datetime64("2001-06-01")) & (grid < np.datetime64("2001-07-01"))
    values[wet] = 0.05  # a fully observed wet month, so "all" is true there
    keep[wet] = True

    result = run(
        {"p": values[keep]},
        grid[keep],
        cadence="daily",
        target="monthly",
        aggregation={"p": aggregation},
        threshold=0.2,
    )
    dates = [d.item() for d in grid[keep]]
    keys, hazard, valid = _reference_months(dates, values[keep], 0.2, aggregation)
    assert [(t.year, t.month) for t in result.time.tolist()] == keys
    assert result.exceedance[0].tolist() == hazard
    assert result.valid[0].tolist() == valid
    # the sample exercises every branch: valid and invalid months, true and false reductions
    assert any(valid)
    assert not all(valid)
    assert any(hazard)
    assert not all(hazard)


def test_hand_counted_monthly_reduction_with_leap_february() -> None:
    """Check ``any``/``all`` and the 29 civil days that a leap February needs."""
    jan = days("2000-01-01", "2000-02-01")  # 31 days
    feb = days("2000-02-01", "2000-03-01")  # 29 days in 2000
    time = np.concatenate([jan, feb])
    values = np.concatenate([np.full(31, 0.9), np.full(29, 0.9)])
    values[5] = 0.05  # one dry January day
    values[40] = 0.05  # one dry February day
    any_result = run({"p": values}, time, cadence="daily", aggregation={"p": "any"})
    all_result = run({"p": values}, time, cadence="daily", aggregation={"p": "all"})
    assert ints(any_result.exceedance[0]) == [1, 1]
    assert ints(any_result.valid[0]) == [1, 1]
    assert ints(all_result.exceedance[0]) == [0, 0]
    assert ints(all_result.valid[0]) == [1, 1]

    short_feb = np.delete(np.arange(60), 31 + 28)  # drop 29 February
    result = run({"p": values[short_feb]}, time[short_feb], cadence="daily", aggregation={"p": "any"})
    assert ints(result.valid[0]) == [1, 0]
    assert ints(result.exceedance[0]) == [1, 0]  # the observed dry day does not rescue an incomplete month

    common = run({"p": np.full(28, 0.05)}, days("2001-02-01", "2001-03-01"), cadence="daily", aggregation={"p": "all"})
    assert ints(common.valid[0]) == [1]
    assert ints(common.exceedance[0]) == [1]


def test_partial_first_and_last_months_are_invalid_but_stay_on_the_grid() -> None:
    """Keep partial boundary months as invalid steps so coverage still equals a monthly source's."""
    time = days("2000-01-15", "2000-03-21")
    daily = {"d": np.full(time.shape[0], 0.05)}
    result = run(
        {**daily, "m": np.array([0.05, 0.05, 0.05])},
        {"d": time, "m": months("2000-01", "2000-04")},
        cadence={"d": "daily", "m": "monthly"},
        aggregation={"d": "any"},
    )
    assert result.time.tolist() == [dt.date(2000, m, 1) for m in (1, 2, 3)]
    assert ints(result.valid) == [[0, 1, 0], [1, 1, 1]]
    assert ints(result.exceedance) == [[0, 1, 0], [1, 1, 1]]


def test_reduction_thresholds_daily_values_before_reducing() -> None:
    """Never average probabilities: a mean far above the threshold still contains a hazard day."""
    values = np.full(31, 0.99)
    values[10] = 0.01
    time = days("2000-01-01", "2000-02-01")
    assert ints(run({"p": values}, time, cadence="daily", aggregation={"p": "any"}).exceedance[0]) == [1]
    assert ints(run({"p": values}, time, cadence="daily", aggregation={"p": "all"}).exceedance[0]) == [0]
    assert values.mean() > 0.9


def test_daily_indicator_reduction_uses_its_validity() -> None:
    """Reduce a Boolean indicator the same way, requiring every civil day to be valid."""
    time = days("2000-01-01", "2000-03-01")
    flags = np.zeros(time.shape[0], dtype=bool)
    flags[3] = True
    observed = np.ones(time.shape[0], dtype=bool)
    observed[40] = False  # one unknown February day
    result = run(
        {"h": flags},
        time,
        cadence="daily",
        scale="indicator",
        threshold=0.5,
        direction="above",
        aggregation={"h": "any"},
        validity={"h": observed},
    )
    assert ints(result.exceedance[0]) == [1, 0]
    assert ints(result.valid[0]) == [1, 0]


def test_daily_cadence_is_kept_when_all_sources_are_daily() -> None:
    """Keep the daily step and ignore the absence of any reducer."""
    time = days("2000-01-01", "2000-01-05")
    result = run(
        {"a": np.array([0.1, 0.9, 0.1, 0.9]), "b": np.array([0.9, 0.9, 0.1, 0.1])},
        time,
        cadence="daily",
        target="daily",
    )
    assert result.cadence == "daily"
    assert ints(result.exceedance) == [[1, 0, 1, 0], [0, 0, 1, 1]]


# --- alignment -------------------------------------------------------------------------------


def test_exact_join_rejects_mismatched_coverage() -> None:
    """Require identical normalized coverage by default and point to ``join='inner'``."""
    inputs = {"a": np.full(4, 0.1), "b": np.full(4, 0.1)}
    time = {"a": months("2000-01", "2000-05"), "b": months("2000-02", "2000-06")}
    with pytest.raises(CoordinateValidationError, match="inner"):
        run(inputs, time)


def test_inner_join_restricts_to_the_overlap_and_warns_about_dropped_boundaries() -> None:
    """Keep the common range, warn once, and record each source's dropped steps."""
    inputs = {"a": np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9]), "b": np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])}
    time = {"a": months("2000-01", "2000-07"), "b": months("2000-03", "2000-09")}
    with pytest.warns(InputAlignmentWarning) as caught:
        result = run(inputs, time, join="inner")
    assert len(caught) == 1
    assert caught[0].filename == __file__  # attributed to the caller, not to the library
    warning = caught[0].message
    assert isinstance(warning, InputAlignmentWarning)
    # every field shares one basis: source time steps summed over all sources
    assert (warning.original_size, warning.aligned_size, warning.dropped_count) == (12, 8, 4)
    assert result.time.tolist() == [dt.date(2000, m, 1) for m in (3, 4, 5, 6)]
    assert ints(result.exceedance) == [[1, 0, 1, 0], [1, 1, 1, 0]]
    sources = result.metadata["sources"]
    assert (sources["a"]["dropped_leading"], sources["a"]["dropped_trailing"]) == (2, 0)
    assert (sources["b"]["dropped_leading"], sources["b"]["dropped_trailing"]) == (0, 2)


def test_inner_join_materializes_only_the_retained_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound allocation by the retained overlap, never by the span between distant observations."""
    requested: list[int] = []
    densify = harmonize_module._on_grid

    def record(positions: Any, steps: int, hazard: Any, valid: Any) -> Any:
        requested.append(int(steps))
        return densify(positions, steps, hazard, valid)

    monkeypatch.setattr(harmonize_module, "_on_grid", record)
    # 14 months against 1 shared month: densifying each source's own span would request 14 steps
    inputs = {"a": np.full(14, 0.1), "b": np.full(1, 0.1)}
    time = {"a": months("2000-01", "2001-03"), "b": months("2000-02", "2000-03")}
    with pytest.warns(InputAlignmentWarning) as caught:
        result = run(inputs, time, join="inner")
    assert requested == [1, 1]
    assert result.exceedance.shape == (2, 1)
    warning = caught[0].message
    assert isinstance(warning, InputAlignmentWarning)
    assert (warning.original_size, warning.aligned_size, warning.dropped_count) == (15, 2, 13)


def test_source_whose_observations_skip_the_retained_window_stays_invalid() -> None:
    """A source with nothing inside the overlap contributes no valid step instead of failing."""
    inputs = {"a": np.array([0.05, 0.05]), "b": np.full(11, 0.05)}
    time = {"a": np.array(["2000-01", "2002-01"], dtype="datetime64[M]"), "b": months("2000-02", "2001-01")}
    with pytest.warns(InputAlignmentWarning):
        result = run(inputs, time, join="inner")
    assert result.time.tolist() == [dt.date(2000, m, 1) for m in range(2, 13)]
    assert ints(result.valid) == [[0] * 11, [1] * 11]
    assert ints(result.exceedance) == [[0] * 11, [1] * 11]


def test_daily_reduction_does_not_materialize_a_per_day_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce daily input to months by counting, so no per-day grid is built at all."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the monthly reduction must not allocate a per-day grid")

    monkeypatch.setattr(harmonize_module, "_on_grid", forbidden)
    time = days("2000-01-01", "2000-03-01")
    result = run({"p": np.full(time.shape[0], 0.05)}, time, cadence="daily", aggregation={"p": "any"})
    assert ints(result.valid[0]) == [1, 1]
    assert ints(result.exceedance[0]) == [1, 1]


def test_inner_join_with_identical_coverage_does_not_warn() -> None:
    """Warn only about dropped boundaries."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = run({"a": np.full(3, 0.1), "b": np.full(3, 0.1)}, months("2000-01", "2000-04"), join="inner")
    assert result.metadata["sources"]["a"]["n_observed"] == 3


def test_inner_join_with_no_overlap_is_an_error() -> None:
    """Fail rather than return an empty axis."""
    inputs = {"a": np.full(2, 0.1), "b": np.full(2, 0.1)}
    time = {"a": months("2000-01", "2000-03"), "b": months("2001-01", "2001-03")}
    with pytest.raises(CoordinateValidationError, match="no time range"):
        run(
            inputs,
            time,
            join="inner",
        )


def test_daily_and_monthly_sources_align_on_calendar_months() -> None:
    """Combine a reduced daily source with a monthly source at the monthly cadence."""
    result = run(
        {"d": np.full(90, 0.9), "m": np.array([0.1, 0.9, 0.1])},
        {"d": days("2001-01-01", "2001-04-01"), "m": months("2001-01", "2001-04")},
        cadence={"d": "daily", "m": "monthly"},
        aggregation={"d": "all"},
    )
    assert ints(result.valid) == [[1, 1, 1], [1, 1, 1]]
    assert ints(result.exceedance[0]) == [0, 0, 0]
    assert ints(result.exceedance[1]) == [1, 0, 1]


# --- rejected declarations --------------------------------------------------------------------


def _base() -> dict[str, Any]:
    """Return keyword arguments for a valid one-source call."""
    return {
        "inputs": {"a": np.array([0.1, 0.9, 0.3])},
        "time": {"a": months("2000-01", "2000-04")},
        "cadence": {"a": "monthly"},
        "input_scale": {"a": "probability"},
        "thresholds": {"a": 0.2},
        "directions": {"a": "below"},
        "target_cadence": "monthly",
        "source_metadata": {"a": {"calibration_year_initial": 1991, "calibration_year_final": 2020}},
    }


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda kw: kw.update(input_scale={"a": "normal"}), InvalidArgumentError, "z-score"),
        (lambda kw: kw.update(input_scale={"a": "bounded"}), InvalidArgumentError, "z-score"),
        (lambda kw: kw.update(input_scale={"a": "pit"}), InvalidArgumentError, "input_scale"),
        (lambda kw: kw["inputs"].update(a=np.array([True, False, True])), InputTypeError, "probability"),
        (lambda kw: kw["inputs"].update(a=np.array([0.1, 0.2, 0.3], dtype=complex)), InputTypeError, "probability"),
        (lambda kw: kw["inputs"].update(a=np.array([0.1, "x", 0.3], dtype=object)), InputTypeError, "probability"),
        (lambda kw: kw["inputs"].update(a=np.array(["0.1", "0.2", "0.3"])), InputTypeError, "probability"),
        (lambda kw: kw["inputs"].update(a=days("2000-01-01", "2000-01-04")), InputTypeError, "probability"),
        (lambda kw: kw["inputs"].update(a=np.array([0.1, np.inf, 0.3])), InvalidArgumentError, "not clipped"),
        (lambda kw: kw["inputs"].update(a=np.array([0.1, -np.inf, 0.3])), InvalidArgumentError, "not clipped"),
        (lambda kw: kw["inputs"].update(a=np.array([0.1, 1.0000001, 0.3])), InvalidArgumentError, "not clipped"),
        (lambda kw: kw["inputs"].update(a=np.array([0.1, -0.2, 0.3])), InvalidArgumentError, "not clipped"),
        (lambda kw: kw["inputs"].update(a=np.array([0, 2, 1])), InvalidArgumentError, "not clipped"),
        (lambda kw: kw["inputs"].update(a=xr.DataArray(np.array([0.1, 0.2, 0.3]))), InputTypeError, "NumPy"),
        (lambda kw: kw["inputs"].update(a=np.float64(0.1)), DataShapeError, "time axis"),
        (lambda kw: kw["inputs"].update(a=np.ones((3, 2))), DataShapeError, "spatial_time_major"),
        (lambda kw: kw["inputs"].update(a=np.empty(0)), DataShapeError, "no time steps"),
        (lambda kw: kw.update(thresholds={"a": 1.5}), InvalidArgumentError, "in \\[0, 1\\]"),
        (lambda kw: kw.update(thresholds={"a": -0.1}), InvalidArgumentError, "in \\[0, 1\\]"),
        (lambda kw: kw.update(thresholds={"a": np.nan}), InvalidArgumentError, "finite"),
        (lambda kw: kw.update(thresholds={"a": np.inf}), InvalidArgumentError, "finite"),
        (lambda kw: kw.update(thresholds={"a": "0.2"}), InvalidArgumentError, "finite"),
        (lambda kw: kw.update(thresholds={"a": True}), InvalidArgumentError, "finite"),
        (lambda kw: kw.update(thresholds={"a": None}), InvalidArgumentError, "finite"),
        (lambda kw: kw.update(directions={"a": "sideways"}), InvalidArgumentError, "directions"),
        (lambda kw: kw.update(directions={"a": None}), InvalidArgumentError, "directions"),
        (lambda kw: kw.update(thresholds={}), InvalidArgumentError, "thresholds"),
        (lambda kw: kw.update(directions={"a": "below", "z": "above"}), InvalidArgumentError, "directions"),
        (lambda kw: kw.update(cadence={"a": "monthly", "z": "daily"}), InvalidArgumentError, "cadence"),
        (lambda kw: kw.update(time=[1, 2]), InvalidArgumentError, "time"),
        (lambda kw: kw.update(validity={"z": np.ones(3, dtype=bool)}), InvalidArgumentError, "validity"),
        (lambda kw: kw.update(source_metadata={"z": {}}), InvalidArgumentError, "source_metadata"),
        (lambda kw: kw.update(inputs={}), InvalidArgumentError, "non-empty"),
        (lambda kw: kw.update(inputs=[np.array([0.1])]), InvalidArgumentError, "non-empty"),
        (lambda kw: kw.update(inputs={1: np.array([0.1])}), InvalidArgumentError, "strings"),
        (lambda kw: kw.update(target_cadence="weekly"), InvalidArgumentError, "target_cadence"),
        (lambda kw: kw.update(cadence={"a": "hourly"}), InvalidArgumentError, "cadence"),
        (lambda kw: kw.update(join="outer"), InvalidArgumentError, "join"),
        (lambda kw: kw.update(spatial_time_major="yes"), InvalidArgumentError, "bool"),
        (lambda kw: kw.update(validity={"a": np.array([1, 1, 1])}), InputTypeError, "Boolean"),
        (lambda kw: kw.update(validity={"a": np.ones(2, dtype=bool)}), DataShapeError, "shape"),
        (lambda kw: kw.update(aggregation={"a": "any"}), InvalidArgumentError, "irrelevant reducer"),
    ],
)
def test_invalid_declarations_are_rejected(mutate: Any, error: type[Exception], match: str) -> None:
    """Reject each unsupported declaration with its structured exception, without clipping."""
    kwargs = _base()
    mutate(kwargs)
    with pytest.raises(error, match=match):
        compound.harmonize(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold": 0.2}, "threshold 0.5"),
        ({"direction": "below"}, "direction 'above'"),
    ],
)
def test_indicator_declaration_is_exactly_half_and_above(kwargs: dict[str, Any], match: str) -> None:
    """Allow no other public declaration for a Boolean indicator."""
    declared = {"scale": "indicator", "threshold": 0.5, "direction": "above", **kwargs}
    inputs, time = {"h": np.array([True, False, True])}, months("2000-01", "2000-04")
    with pytest.raises(InvalidArgumentError, match=match):
        run(inputs, time, **declared)


@pytest.mark.parametrize("values", [np.array([1, 0, 1]), np.array([1.0, 0.0, 1.0])], ids=["int", "float"])
def test_indicator_must_have_boolean_dtype(values: np.ndarray) -> None:
    """Never coerce numbers to an indicator."""
    time = months("2000-01", "2000-04")
    with pytest.raises(InputTypeError, match="Boolean"):
        run({"h": values}, time, scale="indicator", threshold=0.5, direction="above")


def test_monthly_source_cannot_be_expanded_to_a_daily_target() -> None:
    """Reject monthly-to-daily targets rather than interpolating scores."""
    inputs, time = {"m": np.full(3, 0.1)}, months("2000-01", "2000-04")
    with pytest.raises(InvalidArgumentError, match="expanded"):
        run(inputs, time, target="daily")


@pytest.mark.parametrize("aggregation", [None, {}, {"d": "mean"}, {"d": "sum"}, {"d": None}])
def test_daily_source_of_a_monthly_target_needs_any_or_all(aggregation: Any) -> None:
    """Require an explicit ``any`` or ``all`` reducer; mean, sum, and fraction are out of scope."""
    inputs, time = {"d": np.full(31, 0.1)}, days("2000-01-01", "2000-02-01")
    with pytest.raises(InvalidArgumentError, match="aggregation"):
        run(inputs, time, cadence="daily", aggregation=aggregation)


def test_reducers_for_monthly_sources_or_daily_targets_are_irrelevant() -> None:
    """Reject a reducer where nothing is reduced."""
    monthly_inputs, monthly_time = {"m": np.full(3, 0.1)}, months("2000-01", "2000-04")
    with pytest.raises(InvalidArgumentError, match="irrelevant reducer"):
        run(monthly_inputs, monthly_time, aggregation={"m": "any"})
    daily_inputs, daily_time = {"d": np.full(3, 0.1)}, days("2000-01-01", "2000-01-04")
    with pytest.raises(InvalidArgumentError, match="irrelevant reducer"):
        run(
            daily_inputs,
            daily_time,
            cadence="daily",
            target="daily",
            aggregation={"d": "any"},
        )


def test_gridded_sources_need_matching_cells() -> None:
    """Never broadcast a point across a grid."""
    time = months("2000-01", "2000-04")
    mismatched = {"a": np.full((3, 2), 0.1), "b": np.full((3, 4), 0.1)}
    with pytest.raises(DataShapeError, match="cell shape"):
        run(mismatched, time, spatial_time_major=True)
    point_and_grid = {"a": np.full((3, 2), 0.1), "b": np.full(3, 0.1)}
    with pytest.raises(DataShapeError, match="cell shape"):
        run(point_and_grid, time, spatial_time_major=True)


# --- calibration metadata and the manifest ----------------------------------------------------


def test_conflicting_known_calibration_periods_are_rejected() -> None:
    """Reject sources calibrated on different periods, indicators included."""
    metadata = {
        "a": {"calibration_year_initial": 1981, "calibration_year_final": 2010},
        "b": {"calibration_year_initial": 1991, "calibration_year_final": 2020},
    }
    inputs, time = {"a": np.full(3, 0.1), "b": np.full(3, 0.1)}, months("2000-01", "2000-04")
    with pytest.raises(InvalidArgumentError, match="conflicting"):
        run(inputs, time, source_metadata=metadata)


def test_matching_calibration_periods_and_different_timescales_are_recorded_silently() -> None:
    """Allow equal periods and different Timescales, recording both."""
    period = {"calibration_year_initial": 1991, "calibration_year_final": 2020}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = run(
            {"a": np.full(3, 0.1), "b": np.full(3, 0.1)},
            months("2000-01", "2000-04"),
            source_metadata={"a": {**period, "scale": 3}, "b": {**period, "scale": 12}},
        )
    sources = result.metadata["sources"]
    assert sources["a"]["calibration_period"] == sources["b"]["calibration_period"] == (1991, 2020)
    assert (sources["a"]["scale"], sources["b"]["scale"]) == (3, 12)


def test_missing_probability_calibration_warns_once_and_records_unknown() -> None:
    """Never claim compatibility for a source without fit metadata."""
    inputs, time = {"a": np.full(3, 0.1), "b": np.full(3, 0.1)}, months("2000-01", "2000-04")
    with pytest.warns(ClimateIndicesWarning, match="Calibration Period") as caught:
        result = run(inputs, time, calibrated=False)
    assert len(caught) == 1
    assert caught[0].filename == __file__  # attributed to the caller, not to the library
    assert "'a'" in str(caught[0].message)
    assert "'b'" in str(caught[0].message)
    for source in ("a", "b"):
        assert result.metadata["sources"][source]["calibration_period"] == "unknown"
        assert result.metadata["sources"][source]["scale"] == "unknown"


def test_indicator_without_calibration_records_unknown_without_warning() -> None:
    """A physical-threshold indicator never had a Calibration Period, so it does not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = run(
            {"heat": np.array([True, False, True])},
            months("2000-01", "2000-04"),
            scale="indicator",
            threshold=0.5,
            direction="above",
            calibrated=False,
            source_metadata={"heat": {"physical_threshold": 35.0, "comparator": ">="}},
        )
    entry = result.metadata["sources"]["heat"]
    assert entry["calibration_period"] == "unknown"
    assert (entry["physical_threshold"], entry["comparator"]) == (35.0, ">=")
    assert (entry["threshold"], entry["direction"]) == (0.5, "above")


def test_unknown_indicator_physical_declaration_is_recorded_as_unknown() -> None:
    """Record what is missing rather than inventing it."""
    result = run(
        {"heat": np.array([True, False, True])},
        months("2000-01", "2000-04"),
        scale="indicator",
        threshold=0.5,
        direction="above",
    )
    entry = result.metadata["sources"]["heat"]
    assert (entry["physical_threshold"], entry["comparator"]) == ("unknown", "unknown")


@pytest.mark.parametrize(
    "metadata",
    [
        {"calibration_year_initial": 1991},
        {"calibration_year_final": 2020},
        {"calibration_year_initial": None, "calibration_year_final": None},
        {"calibration_year_initial": 2020, "calibration_year_final": 1991},
        {"calibration_year_initial": 1991.0, "calibration_year_final": 2020},
        {"calibration_year_initial": True, "calibration_year_final": 2020},
        {"scale": {1, 2}},
        {"scale": float("nan")},
        {"scale": object()},
    ],
    ids=["only-initial", "only-final", "null-years", "reversed", "float-year", "bool-year", "set", "nan", "object"],
)
def test_malformed_source_metadata_is_rejected(metadata: dict[str, Any]) -> None:
    """Require JSON-compatible metadata and well-formed calibration years."""
    inputs, time = {"a": np.full(3, 0.1)}, months("2000-01", "2000-04")
    with pytest.raises(InvalidArgumentError, match="source_metadata"):
        run(inputs, time, source_metadata={"a": metadata})


def test_source_metadata_must_be_a_mapping() -> None:
    """Reject a metadata value that is not a mapping."""
    inputs, time = {"a": np.full(3, 0.1)}, months("2000-01", "2000-04")
    with pytest.raises(InvalidArgumentError, match="source_metadata"):
        run(inputs, time, source_metadata={"a": 5})


def test_manifest_schema_serializes_and_labels_no_pit_or_cf_metadata() -> None:
    """Round-trip a manifest through JSON with a stable schema and no false PIT/CF labels."""
    result = run(
        {"d": np.full(90, 0.9), "h": np.zeros(3, dtype=bool)},
        {"d": days("2001-01-01", "2001-04-01"), "h": months("2001-01", "2001-04")},
        cadence={"d": "daily", "h": "monthly"},
        scale={"d": "probability", "h": "indicator"},
        threshold={"d": 0.2, "h": 0.5},
        direction={"d": "below", "h": "above"},
        aggregation={"d": "any"},
        source_metadata={
            "d": {"calibration_year_initial": 1991, "calibration_year_final": 2020, "scale": 6, "id": "spi6"},
            "h": {"physical_threshold": 35, "comparator": ">="},
        },
    )
    text = json.dumps(result.metadata, allow_nan=False)  # the ordinary encoder, with no default= hook
    manifest = json.loads(text)
    assert set(manifest) == {
        "schema_version",
        "calendar",
        "cadence",
        "join",
        "time_start",
        "time_end",
        "n_steps",
        "missingness",
        "sources",
    }
    assert manifest["schema_version"] == 1
    assert (manifest["calendar"], manifest["cadence"], manifest["join"]) == ("proleptic_gregorian", "monthly", "exact")
    assert (manifest["time_start"], manifest["time_end"], manifest["n_steps"]) == ("2001-01", "2001-03", 3)
    assert list(manifest["sources"]) == ["d", "h"]
    assert manifest["sources"]["d"] == {
        "input_scale": "probability",
        "threshold": 0.2,
        "direction": "below",
        "cadence": "daily",
        "aggregation": "any",
        "time_start": "2001-01-01",
        "time_end": "2001-03-31",
        "n_observed": 90,
        "calibration_period": [1991, 2020],
        "scale": 6,
        "source_metadata": {"calibration_year_initial": 1991, "calibration_year_final": 2020, "scale": 6, "id": "spi6"},
    }
    assert set(manifest["sources"]["h"]) == {
        "input_scale",
        "threshold",
        "direction",
        "cadence",
        "aggregation",
        "time_start",
        "time_end",
        "n_observed",
        "calibration_period",
        "scale",
        "physical_threshold",
        "comparator",
        "source_metadata",
    }
    assert "standard_name" not in text
    assert result.cadence == "monthly"
    assert result.names == ("d", "h")


# --- immutability ------------------------------------------------------------------------------


def test_result_arrays_are_read_only_and_new() -> None:
    """Freeze the arrays themselves; a frozen dataclass alone would not."""
    indicator = np.array([True, False, True])
    result = run({"h": indicator}, months("2000-01", "2000-04"), scale="indicator", threshold=0.5, direction="above")
    for array in (result.exceedance, result.valid, result.time):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[0] = array[0]
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    assert not np.shares_memory(result.exceedance, indicator)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.cadence = "daily"  # type: ignore[misc]


def test_caller_data_is_not_mutated() -> None:
    """Copy inputs and leave the caller's arrays, masks, and dictionaries untouched and writeable."""
    values = np.ma.array([0.1, np.nan, 0.5, 0.9], mask=[False, False, True, False])
    validity = np.array([True, True, True, False])
    metadata = {"a": {"calibration_year_initial": 1991, "calibration_year_final": 2020, "tags": ["x", "y"]}}
    before = (values.data.copy(), values.mask.copy(), validity.copy(), copy.deepcopy(metadata))
    result = run({"a": values}, months("2000-01", "2000-05"), validity={"a": validity}, source_metadata=metadata)
    assert np.array_equal(values.data, before[0], equal_nan=True)
    assert np.array_equal(values.mask, before[1])
    assert np.array_equal(validity, before[2])
    assert metadata == before[3]
    assert values.data.flags.writeable
    assert validity.flags.writeable
    metadata["a"]["tags"].append("z")  # a later caller edit must not reach the manifest
    assert result.metadata["sources"]["a"]["source_metadata"]["tags"] == ("x", "y")


def test_metadata_is_immutable_at_every_depth() -> None:
    """Expose read-only mappings and tuples, never the caller's containers."""
    result = run({"a": np.full(3, 0.1)}, months("2000-01", "2000-04"))
    with pytest.raises(TypeError):
        result.metadata["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result.metadata["sources"]["a"]["threshold"] = 1.0
    with pytest.raises(TypeError):
        result.metadata["sources"]["a"]["source_metadata"]["x"] = 1
    with pytest.raises(TypeError):
        result.metadata.update({"new": 1})  # type: ignore[attr-defined]
    manifest: Any = result.metadata
    with pytest.raises(TypeError):
        manifest |= {"new": 1}
    copied = copy.deepcopy(result.metadata)
    assert copied == result.metadata
    assert pickle.loads(pickle.dumps(result.metadata)) == result.metadata
    assert isinstance(result.metadata["sources"]["a"]["calibration_period"], tuple)


# --- lifecycle ----------------------------------------------------------------------------------


def test_lifecycle_events_are_emitted_on_success() -> None:
    """Emit ``calculation_started`` then ``calculation_completed`` with the output shape."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger
    with mock.patch.object(harmonize_module, "_logger", mock_logger):
        run({"a": np.full(3, 0.1), "b": np.full(3, 0.9)}, months("2000-01", "2000-04"))
    bound = mock_logger.bind.call_args.kwargs
    assert bound["index_type"] == "compound_harmonize"
    assert bound["n_sources"] == 2  # the names are caller identifiers and stay out of the log
    assert "sources" not in bound
    assert [call.args[0] for call in mock_logger.info.call_args_list] == [
        "calculation_started",
        "calculation_completed",
    ]
    completed = mock_logger.info.call_args_list[1].kwargs
    assert completed["output_shape"] == (2, 3)
    assert completed["duration_ms"] >= 0
    mock_logger.error.assert_not_called()


def test_failure_is_logged_and_propagates() -> None:
    """Emit ``calculation_failed`` for a rejected declaration and re-raise it unchanged."""
    mock_logger = mock.MagicMock()
    mock_logger.bind.return_value = mock_logger
    inputs, time = {"a": np.full(3, 0.1)}, months("2000-01", "2000-04")
    with (
        mock.patch.object(harmonize_module, "_logger", mock_logger),
        pytest.raises(InvalidArgumentError, match="z-score"),
    ):
        run(inputs, time, scale="normal")
    mock_logger.info.assert_called_once_with("calculation_started")
    failed = [call for call in mock_logger.error.call_args_list if call.args[0] == "calculation_failed"]
    assert len(failed) == 1
    assert failed[0].kwargs["error_type"] == "InvalidArgumentError"
    assert "z-score" in failed[0].kwargs["error_message"]


def test_failure_logging_survives_a_non_mapping_inputs_argument() -> None:
    """Bind safely before validation so a malformed ``inputs`` still reports a structured error."""
    with pytest.raises(InvalidArgumentError, match="non-empty"):
        compound.harmonize(
            [1, 2],  # type: ignore[arg-type]
            time={},
            cadence={},
            input_scale={},
            thresholds={},
            directions={},
            target_cadence="monthly",
        )

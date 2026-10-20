"""Flood-event summaries over the run layer: onset periods, severities, and peaks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import flood
from climate_indices.exceptions import CoordinateValidationError, InsufficientDataError, InvalidArgumentError
from climate_indices.flood import FloodStatistics
from climate_indices.runs import RunSet

# Deliberate wrong-argument calls bypass the public overloads.
_summary: Any = flood.flood_event_summary


def _daily_dataarray(values: np.ndarray, start: str = "2000-01-01", **dims: list[int]) -> xr.DataArray:
    dates = pd.date_range(start, periods=values.shape[0], freq="D")
    coords: dict[str, object] = {"time": dates, **dims}
    return xr.DataArray(values, dims=("time", *dims), coords=coords)


def test_events_severity_duration_and_peak_are_exact() -> None:
    """Hand-constructed runs: severity is the raw run sum, duration the run length."""
    index = np.array([-1.0, 2.5, 1.5, -0.5, 4.0, np.nan, 1.0, 0.5])
    events = flood.flood_events(index)
    np.testing.assert_array_equal(events.onset, [1, 4, 6])
    np.testing.assert_array_equal(events.duration, [2, 1, 2])
    np.testing.assert_allclose(events.severity, [4.0, 4.0, 1.5], rtol=0, atol=0)
    np.testing.assert_allclose(events.peak, [2.5, 4.0, 1.0], rtol=0, atol=0)


def test_summary_reports_per_period_count_duration_severity_peak_and_valid_days() -> None:
    index = np.array([-1.0, 2.0, 2.0, -1.0, 5.0, 1.0, np.nan, 3.0, -1.0])
    periods = np.array([2000, 2000, 2000, 2000, 2001, 2001, 2001, 2001, 2001])

    stats = flood.flood_event_summary(index, periods=periods)

    assert isinstance(stats, FloodStatistics)
    assert isinstance(stats.events, RunSet)
    np.testing.assert_array_equal(stats.period, [2000, 2001])
    np.testing.assert_array_equal(stats.count, [1, 2])
    np.testing.assert_array_equal(stats.total_duration, [2, 3])
    np.testing.assert_allclose(stats.max_severity, [4.0, 6.0], rtol=0, atol=0)
    np.testing.assert_allclose(stats.max_peak, [2.0, 5.0], rtol=0, atol=0)
    np.testing.assert_array_equal(stats.valid_count, [4, 4])
    np.testing.assert_allclose(stats.frequency, [0.25, 0.5], rtol=0, atol=0)


def test_an_event_belongs_to_its_onset_period() -> None:
    """An event crossing a boundary contributes its full duration to the period it starts in."""
    stats = flood.flood_event_summary(np.full(4, 2.0), periods=np.array([2000, 2000, 2001, 2001]))

    np.testing.assert_array_equal(stats.count, [1, 0])
    np.testing.assert_array_equal(stats.total_duration, [4, 0])
    np.testing.assert_allclose(stats.max_peak, [2.0, 0.0], rtol=0, atol=0)
    np.testing.assert_allclose(stats.max_severity, [8.0, 0.0], rtol=0, atol=0)


def test_edi_positive_runs_use_the_same_conventions() -> None:
    """EDI shares the PE kernel with I_F, and its positive runs summarize like the event scan."""
    rain = np.random.default_rng(3).gamma(2.0, 20.0, size=366 * 6)
    pe = flood.effective_precipitation(rain, duration=30)
    series = flood.edi(pe, 2000, 2002, 2004)
    assert isinstance(series, np.ndarray)
    periods = np.arange(series.size) // 366

    summary = flood.flood_event_summary(series, periods=periods)
    events = flood.flood_events(series)

    assert len(events) > 0
    assert len(summary.events) == len(events)
    event_periods = events.onset // 366
    for index, period in enumerate(summary.period):
        selected = event_periods == period
        np.testing.assert_allclose(
            summary.max_severity[index], events.severity[selected].max() if selected.any() else 0.0, rtol=1e-12
        )
        np.testing.assert_allclose(
            summary.max_peak[index], events.peak[selected].max() if selected.any() else 0.0, rtol=1e-12
        )


def test_frequency_matches_the_closed_form_on_a_synthetic_series() -> None:
    index = np.random.default_rng(7).normal(size=2000)
    periods = np.arange(index.size) // 100

    stats = flood.flood_event_summary(index, periods=periods)

    expected = np.zeros(20, dtype=np.int64)
    for position, value in enumerate(index):
        if value > 0 and (position == 0 or index[position - 1] <= 0):
            expected[position // 100] += 1
    np.testing.assert_array_equal(stats.count, expected)
    np.testing.assert_allclose(stats.frequency, expected / 100, rtol=0, atol=0)


def test_numpy_and_one_dimensional_dataarray_paths_agree() -> None:
    values = np.array([-1.0, 2.0, 2.0, 0.0, 3.0, -1.0, np.nan, 4.0])
    data = _daily_dataarray(values, start="1999-10-01")

    from_dataarray = flood.flood_event_summary(data, year_start_month=10)
    from_numpy = flood.flood_event_summary(values, periods=np.full(values.size, 1999))

    assert isinstance(from_dataarray, FloodStatistics)
    np.testing.assert_array_equal(from_dataarray.period, from_numpy.period)
    np.testing.assert_array_equal(from_dataarray.count, from_numpy.count)
    np.testing.assert_array_equal(from_dataarray.total_duration, from_numpy.total_duration)
    np.testing.assert_allclose(from_dataarray.max_severity, from_numpy.max_severity, rtol=0, atol=0)
    np.testing.assert_allclose(from_dataarray.max_peak, from_numpy.max_peak, rtol=0, atol=0)
    np.testing.assert_array_equal(from_dataarray.valid_count, from_numpy.valid_count)
    np.testing.assert_allclose(from_dataarray.frequency, from_numpy.frequency, rtol=0, atol=0)


def test_year_start_month_moves_a_january_event_into_the_prior_hydrological_year() -> None:
    dates = pd.to_datetime(["1999-11-01", "1999-12-31", "2000-01-01", "2000-01-02"])
    data = xr.DataArray(np.array([2.0, 2.0, 0.0, 2.0]), dims="time", coords={"time": dates})

    calendar = flood.flood_event_summary(data, year_start_month=1)
    october = flood.flood_event_summary(data, year_start_month=10)

    np.testing.assert_array_equal(calendar.period, [1999, 2000])
    np.testing.assert_array_equal(calendar.count, [1, 1])
    np.testing.assert_array_equal(october.period, [1999])
    np.testing.assert_array_equal(october.count, [2])


def test_gridded_summary_matches_per_cell_numpy_and_reports_events() -> None:
    values = np.random.default_rng(11).normal(size=(400, 2, 2))
    data = _daily_dataarray(values, lat=[1, 2], lon=[1, 2])

    result = flood.flood_event_summary(data, year_start_month=1, min_duration=2)

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {
        "count",
        "total_duration",
        "max_severity",
        "max_peak",
        "valid_count",
        "frequency",
        "events",
    }
    np.testing.assert_array_equal(result["period"].values, [2000, 2001])
    for lat in (1, 2):
        for lon in (1, 2):
            expected = flood.flood_event_summary(data.sel(lat=lat, lon=lon), year_start_month=1, min_duration=2)
            assert isinstance(expected, FloodStatistics)
            cell = result.sel(lat=lat, lon=lon)
            np.testing.assert_array_equal(cell["count"].values, expected.count)
            np.testing.assert_array_equal(cell["total_duration"].values, expected.total_duration)
            np.testing.assert_allclose(cell["max_severity"].values, expected.max_severity, rtol=0, atol=0)
            np.testing.assert_allclose(cell["max_peak"].values, expected.max_peak, rtol=0, atol=0)
            np.testing.assert_array_equal(cell["valid_count"].values, expected.valid_count)
            np.testing.assert_allclose(cell["frequency"].values, expected.frequency, rtol=0, atol=0)
            assert isinstance(cell["events"].item(), RunSet)


def test_gridded_summary_stays_lazy_on_cells_and_rejects_a_split_time_chunk() -> None:
    values = np.array([[0.0, 2.0, 2.0, 0.0], [2.0, 0.0, 0.0, 2.0]]).T
    data = _daily_dataarray(values, site=[0, 1], start="2000-01-01")

    lazy = flood.flood_event_summary(data.chunk({"time": -1, "site": 1}), year_start_month=1)
    eager = flood.flood_event_summary(data, year_start_month=1)

    assert isinstance(lazy, xr.Dataset)
    assert isinstance(eager, xr.Dataset)
    assert lazy["count"].chunks is not None
    np.testing.assert_array_equal(lazy["count"].compute().values, eager["count"].values)

    with pytest.raises(CoordinateValidationError, match="chunk"):
        flood.flood_event_summary(data.chunk({"time": 2, "site": 1}), year_start_month=1)


def test_dataarray_input_requires_year_start_month() -> None:
    with pytest.raises(InvalidArgumentError, match="year_start_month"):
        _summary(_daily_dataarray(np.array([0.0, 2.0])))


@pytest.mark.parametrize("year_start_month", [0, 13, 1.5, True])
def test_invalid_year_start_month_is_rejected(year_start_month: object) -> None:
    with pytest.raises(InvalidArgumentError, match="year_start_month"):
        _summary(_daily_dataarray(np.array([0.0, 2.0])), year_start_month=year_start_month)


def test_numpy_input_rejects_year_start_month_and_dataarray_rejects_periods() -> None:
    with pytest.raises(InvalidArgumentError, match="year_start_month"):
        _summary([0.0, 2.0], year_start_month=1)
    with pytest.raises(InvalidArgumentError, match="periods"):
        _summary(_daily_dataarray(np.array([0.0, 2.0])), year_start_month=1, periods=[2000, 2000])


def test_a_non_datetime_or_missing_time_coordinate_is_rejected() -> None:
    positional = xr.DataArray([0.0, 2.0], dims="time")
    with pytest.raises(CoordinateValidationError, match="coordinate"):
        flood.flood_event_summary(positional, year_start_month=1)
    integral = xr.DataArray([0.0, 2.0], dims="time", coords={"time": [1, 2]})
    with pytest.raises(CoordinateValidationError):
        flood.flood_event_summary(integral, year_start_month=1)


def test_gridded_empty_time_axis_is_rejected() -> None:
    data = xr.DataArray(np.empty((0, 2)), dims=("time", "site"), coords={"time": pd.DatetimeIndex([])})
    with pytest.raises(InsufficientDataError):
        flood.flood_event_summary(data, year_start_month=1)


@pytest.mark.parametrize(
    "kwargs",
    [{"threshold": float("nan")}, {"threshold": True}, {"min_duration": 0}, {"min_duration": 1.5}],
)
def test_invalid_options_are_rejected_before_dispatch(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidArgumentError):
        _summary([0.0, 2.0], **kwargs)
    with pytest.raises(InvalidArgumentError):
        _summary(_daily_dataarray(np.array([0.0, 2.0])), year_start_month=1, **kwargs)

"""Flood-event extraction: runs of a daily index above a threshold."""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices import flood
from climate_indices.exceptions import InputTypeError, InvalidArgumentError


def test_one_dimensional_events_have_onset_duration_peak_and_severity() -> None:
    index = np.array([-1.0, 0.5, 2.0, 1.0, -0.5, np.nan, 3.0, 0.0, 0.25])
    events = flood.flood_events(index)
    assert len(events) == 3
    np.testing.assert_array_equal(events.onset, [1, 6, 8])
    np.testing.assert_array_equal(events.end, [4, 7, 9])
    np.testing.assert_array_equal(events.duration, [3, 1, 1])
    np.testing.assert_allclose(events.peak, [2.0, 3.0, 0.25])
    np.testing.assert_allclose(events.severity, [3.5, 3.0, 0.25])
    assert events.cell.shape == (3, 0)


def test_nan_and_masked_days_end_a_run() -> None:
    index = np.ma.masked_array([1.0, 1.0, 9.0, 1.0], mask=[False, False, True, False])
    events = flood.flood_events(index)
    np.testing.assert_array_equal(events.onset, [0, 3])
    np.testing.assert_array_equal(events.duration, [2, 1])


def test_min_duration_and_threshold_filter_events() -> None:
    index = np.array([0.0, 1.5, 1.5, 0.0, 2.5, 0.0])
    np.testing.assert_array_equal(flood.flood_events(index, min_duration=2).onset, [1])
    np.testing.assert_array_equal(flood.flood_events(index, threshold=2.0).onset, [4])
    assert len(flood.flood_events(index, threshold=5.0)) == 0


def test_events_ending_on_the_last_day_and_starting_on_the_first_are_kept() -> None:
    events = flood.flood_events(np.array([1.0, 2.0, 0.0, 3.0]))
    np.testing.assert_array_equal(events.onset, [0, 3])
    np.testing.assert_array_equal(events.end, [2, 4])


def test_spatial_block_orders_events_by_cell_then_onset() -> None:
    index = np.zeros((6, 2, 2))
    index[0:2, 0, 1] = 1.0
    index[4, 0, 1] = 2.0
    index[1:6, 1, 0] = [1.0, 1.0, 1.0, 1.0, 1.0]
    events = flood.flood_events(index)
    np.testing.assert_array_equal(events.cell, [[0, 1], [0, 1], [1, 0]])
    np.testing.assert_array_equal(events.onset, [0, 4, 1])
    np.testing.assert_array_equal(events.end, [2, 5, 6])
    np.testing.assert_allclose(events.severity, [2.0, 2.0, 5.0])
    np.testing.assert_allclose(events.peak, [1.0, 2.0, 1.0])


def test_runs_that_touch_a_neighbouring_cell_do_not_merge() -> None:
    index = np.ones((3, 2))
    events = flood.flood_events(index)
    np.testing.assert_array_equal(events.cell[:, 0], [0, 1])
    np.testing.assert_array_equal(events.duration, [3, 3])
    np.testing.assert_allclose(events.severity, [3.0, 3.0])


def _reference(series: np.ndarray, threshold: float, min_duration: int) -> list[tuple[int, int, float, float]]:
    """Plain day-by-day event scan: (onset, end, peak, severity)."""
    events, start = [], None
    for day, value in enumerate([*series, np.nan]):
        hit = bool(np.isfinite(value) and value > threshold)
        if hit and start is None:
            start = day
        if not hit and start is not None:
            run = series[start:day]
            if day - start >= min_duration:
                events.append((start, day, float(run.max()), float(run.sum())))
            start = None
    return events


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("min_duration", [1, 3])
def test_matches_a_day_by_day_reference(seed: int, min_duration: int) -> None:
    rng = np.random.default_rng(seed)
    index = rng.normal(size=(200, 4))
    index[rng.random(index.shape) < 0.05] = np.nan
    events = flood.flood_events(index, threshold=0.3, min_duration=min_duration)
    got = list(zip(events.cell[:, 0], events.onset, events.end, events.peak, events.severity, strict=True))
    expected = [(cell, *event) for cell in range(4) for event in _reference(index[:, cell], 0.3, min_duration)]
    assert len(got) == len(expected)
    for actual, wanted in zip(got, expected, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=1e-12)


def test_all_missing_input_has_no_events() -> None:
    events = flood.flood_events(np.full((5, 2), np.nan))
    assert len(events) == 0
    assert events.cell.shape == (0, 1)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"threshold": float("nan")}, InvalidArgumentError),
        ({"threshold": True}, InvalidArgumentError),
        ({"threshold": "0"}, InvalidArgumentError),
        ({"min_duration": 0}, InvalidArgumentError),
        ({"min_duration": 1.5}, InvalidArgumentError),
    ],
)
def test_invalid_options_are_rejected(kwargs, error) -> None:
    with pytest.raises(error):
        flood.flood_events(np.ones(4), **kwargs)


def test_scalar_and_non_numeric_input_are_rejected() -> None:
    with pytest.raises(InvalidArgumentError):
        flood.flood_events(1.0)
    with pytest.raises(InputTypeError):
        flood.flood_events(["a", "b"])

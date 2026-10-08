"""Flood-event extraction: runs of a daily index above a threshold."""

from __future__ import annotations

import numpy as np

from climate_indices import flood


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

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

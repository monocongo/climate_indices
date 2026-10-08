"""Benchmarks for the flood-event scan; timed runs are excluded from default test runs.

Run explicitly with: pytest -m benchmark --benchmark-enable
"""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices import flood


@pytest.mark.benchmark
@pytest.mark.parametrize("side", [8, 32])
def test_flood_events_scan_throughput(benchmark, side: int) -> None:
    rng = np.random.default_rng(0)
    index = rng.normal(size=(3650, side, side))
    events = benchmark(flood.flood_events, index, min_duration=2)
    assert len(events) > 0

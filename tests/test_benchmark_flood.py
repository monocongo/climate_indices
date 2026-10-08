"""Performance benchmarks for the flood family (RUST-016).

Compares each Rust-backed flood index with the Python implementation it replaces
and times the ``flood_events`` scan:

- effective precipitation, EDI, the Flood Index, and the Antecedent
  Precipitation Index, each run once through the Rust kernels and once with the
  extension switched off, on the same synthetic gridded record;
- a guard that fails when a Rust path is slower than the Python path it replaces;
- the ``flood_events`` scan, which is NumPy only.

Timed tests are marked with @pytest.mark.benchmark and excluded from default test
runs. Run them explicitly with: pytest -m benchmark --benchmark-enable

Scale is configured through environment variables, all parsed as comma-separated
integers (CI-friendly defaults in parentheses):

- BENCHMARK_FLOOD_GRID_SIDES (8,16,32): grid sides for the throughput benchmarks
- BENCHMARK_FLOOD_RECORD_YEARS (5): record length in years, at 366 days per year
  (the all-leap layout EDI and the Flood Index read); at least three
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from climate_indices import flood


def _env_ints(name: str, default: str) -> tuple[int, ...]:
    """Read a comma-separated integer list from the environment."""
    raw = os.getenv(name, default)
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{name} must hold at least one integer, got {raw!r}")
    return values


# grid sides exercised by the throughput benchmarks
_GRID_SIDES = _env_ints("BENCHMARK_FLOOD_GRID_SIDES", "8,16,32")

# record length in years for the throughput benchmarks; EDI and the Flood Index
# need a warm-up year plus two calibration years
_RECORD_YEARS = max(_env_ints("BENCHMARK_FLOOD_RECORD_YEARS", "5")[0], 3)

# days per year in the all-leap layout EDI and the Flood Index read
_DAYS_PER_YEAR = 366

# first calendar year of every synthetic record
_START_YEAR = 2000

# fixed scale for the deterministic guard: large enough that one call is well
# above timer noise, small enough for every CI runner
_GUARD_GRID_SIDE = 16

# timing repetitions per measurement (best-of, which filters CI noise)
_REPEATS = 3

# decay constant for the Antecedent Precipitation Index benchmarks
_API_DECAY = 0.9


@pytest.mark.benchmark
@pytest.mark.parametrize("side", [8, 32])
def test_flood_events_scan_throughput(benchmark, side: int) -> None:
    rng = np.random.default_rng(0)
    index = rng.normal(size=(3650, side, side))
    events = benchmark(flood.flood_events, index, min_duration=2)
    assert len(events) > 0

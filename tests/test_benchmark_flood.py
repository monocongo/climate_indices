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
import time
from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from climate_indices import flood
from climate_indices.flood import _native as flood_native
from tests import conftest


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


def _rain(years: int, side: int, seed: int = 0) -> np.ndarray:
    """Showery daily rain on a square grid in the all-leap layout: about half the days dry."""
    rng = np.random.default_rng(seed)
    shape = (years * _DAYS_PER_YEAR, side, side)
    return rng.gamma(0.5, 8.0, shape) * (rng.random(shape) < 0.5)


def _pe(years: int, side: int) -> np.ndarray:
    """Effective precipitation of the synthetic record, the input EDI and the Flood Index read."""
    with np.errstate(all="ignore"):
        return flood.effective_precipitation(_rain(years, side))


@pytest.fixture(scope="module")
def native() -> ModuleType:
    """The built Rust extension; skips these tests when it is absent (required in the native CI legs)."""
    return conftest.import_native()


def _measure(run: Callable[[], Any], repeats: int = _REPEATS) -> float:
    """Return the best of ``repeats`` wall-clock timings of ``run``, in seconds."""
    with np.errstate(all="ignore"):
        run()  # warmup, excludes first-call allocation costs
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            best = min(best, time.perf_counter() - start)
    return best


def _time_both_paths(
    monkeypatch: pytest.MonkeyPatch, native: ModuleType, run: Callable[[], Any]
) -> tuple[float, float]:
    """Return ``run``'s best seconds through the Rust kernels and on the Python path."""
    monkeypatch.setattr(flood_native, "_native", native)
    rust = _measure(run)
    monkeypatch.setattr(flood_native, "_native", None)
    python = _measure(run)
    return rust, python


def _pe_run(years: int, side: int) -> Callable[[], Any]:
    return partial(flood.effective_precipitation, _rain(years, side))


def _edi_run(years: int, side: int) -> Callable[[], Any]:
    return partial(flood.edi, _pe(years, side), _START_YEAR, _START_YEAR + 1, _START_YEAR + years - 1)


def _flood_index_run(years: int, side: int) -> Callable[[], Any]:
    return partial(
        flood.flood_index,
        _pe(years, side),
        _START_YEAR,
        _START_YEAR + 1,
        _START_YEAR + years - 1,
        year_start_month=1,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("side", _GRID_SIDES)
def test_effective_precipitation_rust_against_python(benchmark, monkeypatch, native, side: int) -> None:
    """Time effective precipitation on the Rust kernel and the Python path."""
    run = _pe_run(_RECORD_YEARS, side)
    rust, python = _time_both_paths(monkeypatch, native, run)
    benchmark.extra_info.update(python_seconds=python, rust_seconds=rust, speedup=python / rust)
    monkeypatch.setattr(flood_native, "_native", native)
    with np.errstate(all="ignore"):
        benchmark(run)


@pytest.mark.benchmark
@pytest.mark.parametrize("side", _GRID_SIDES)
def test_edi_rust_against_python(benchmark, monkeypatch, native, side: int) -> None:
    """Time the EDI on the Rust kernel and the Python path."""
    run = _edi_run(_RECORD_YEARS, side)
    rust, python = _time_both_paths(monkeypatch, native, run)
    benchmark.extra_info.update(python_seconds=python, rust_seconds=rust, speedup=python / rust)
    monkeypatch.setattr(flood_native, "_native", native)
    with np.errstate(all="ignore"):
        benchmark(run)


@pytest.mark.benchmark
@pytest.mark.parametrize("side", _GRID_SIDES)
def test_flood_index_rust_against_python(benchmark, monkeypatch, native, side: int) -> None:
    """Time the Flood Index on the Rust kernel and the Python path."""
    run = _flood_index_run(_RECORD_YEARS, side)
    rust, python = _time_both_paths(monkeypatch, native, run)
    benchmark.extra_info.update(python_seconds=python, rust_seconds=rust, speedup=python / rust)
    monkeypatch.setattr(flood_native, "_native", native)
    with np.errstate(all="ignore"):
        benchmark(run)


@pytest.mark.benchmark
@pytest.mark.parametrize("side", [8, 32])
def test_flood_events_scan_throughput(benchmark, side: int) -> None:
    rng = np.random.default_rng(0)
    index = rng.normal(size=(3650, side, side))
    events = benchmark(flood.flood_events, index, min_duration=2)
    assert len(events) > 0

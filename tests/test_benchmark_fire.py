"""
Performance benchmarks and sizing evidence for the recursive fire indices (#812).

Covers the fire family's scaling profile, which differs from SPI/SPEI because the
recurrences are sequential in time and cannot be chunked along their only
irreducible dimension:

- CFFWIS and KBDI throughput in cell-days per second, across grid sizes and
  record lengths. These measurements are the source of the sizing table
  published in docs/wildfire_applications.md.
- The single-pass ``fire.cffwis()`` orchestrator against separately chained
  code calls (``ffmc``, ``duff_moisture_code``, ``drought_code`` and the derived
  indices), which is the cost claim the orchestrator makes.
- Peak RSS as a function of the spatial chunk size and output selection.
  ``time`` must stay a single chunk (ADR-0003, ADR-0006), so the retained daily
  histories scale with the record length and chunking bounds only the per-day
  temporaries: ``outputs=`` is the lever that reduces peak memory.

The NumPy-versus-``numba`` comparison in #812's task list is superseded by
docs/adr/0006-fire-recursive-state-and-execution.md, which records that ``numba``
is not an optional dependency and is reconsidered only when a representative
benchmark misses an agreed runtime target. The measurements here are that
representative benchmark: no accelerator is added, and the decision stands.

Timed tests are marked with @pytest.mark.benchmark and excluded from default test
runs. The benchmarks workflow runs them on every pull request
(.github/workflows/benchmarks.yml), including the budget guards in
TestFireRegressionGuards that fail on a slow recurrence, on an orchestrator that
costs more than the chained calls it replaces, or on peak RSS far beyond the
modeled footprint. TestFireBudgetPolicy covers those guards' failure paths in the
default suite without depending on wall-clock measurements. Run the marked tests
explicitly with: pytest -m benchmark --benchmark-enable

Scale is configured through environment variables (CI-friendly defaults; the
published sizing table used the spec-scale values):

- FIRE_BENCH_GRID_SIDES (default: 8,16,32, spec: 256)
- FIRE_BENCH_RECORD_DAYS (default: 365,730, spec: 14610)
- FIRE_BENCH_CHUNK_SIDES (default: 8,16, spec: 64)
- FIRE_BENCH_MEMORY_GRID_SIDE (default: 32, spec: 128)
- FIRE_BENCH_MEMORY_RECORD_DAYS (default: 365, spec: 1825)
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from climate_indices import fire

# pytest.importorskip("psutil") lives in that module; the peak-RSS monitor is
# reused here rather than duplicated.
from tests.test_benchmark_memory import _PeakRSSMonitor

# CFFWIS needs a noon-local-standard-time month per day, and the NumPy layer
# carries no calendar, so the benchmarks supply a non-leap-year month series.
_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# grid sides and record lengths exercised by the throughput benchmarks
_GRID_SIDES = tuple(int(side) for side in os.getenv("FIRE_BENCH_GRID_SIDES", "8,16,32").split(","))
_RECORD_DAYS = tuple(int(days) for days in os.getenv("FIRE_BENCH_RECORD_DAYS", "365,730").split(","))

# spatial chunk sides exercised by the peak-memory benchmark
_CHUNK_SIDES = tuple(int(side) for side in os.getenv("FIRE_BENCH_CHUNK_SIDES", "8,16").split(","))

# grid side and record length used by the peak-memory benchmark. The default is
# CI-sized; the published chunk-size table used 128 with a 5-year record, where
# the retained daily histories dominate RSS instead of fixed overhead.
_MEMORY_GRID_SIDE = int(os.getenv("FIRE_BENCH_MEMORY_GRID_SIDE", "32"))
_MEMORY_RECORD_DAYS = int(os.getenv("FIRE_BENCH_MEMORY_RECORD_DAYS", "365"))

# fixed scale for the deterministic guards: large enough that one call is well
# above timer noise, small enough for every CI runner
_GUARD_GRID_SIDE = 32
_GUARD_RECORD_DAYS = 365

# timing repetitions per measurement (best-of, which filters CI noise)
_REPEATS = 3

# Ratio budget for the orchestrator guard. Measured single-pass/chained was
# 0.86-0.91 on the development machine; 1.25 keeps ~35% headroom above that while
# still failing if the orchestrator becomes slower than the chained calls it
# replaces, which is the orchestrator's whole claim.
_ORCHESTRATOR_RATIO_BUDGET = 1.25

# Ratio budget for the machine-speed guard: CFFWIS seconds divided by the seconds
# of an equivalent-size numpy reference workload. Dividing by a workload measured
# in the same process cancels most runner-to-runner speed differences, so the
# budget only has to absorb timer variance, not hardware variance. Measured
# 11.6-14.7 on the development machine across grid sizes and record lengths; 40
# leaves ~2.7x margin and still fails on a deliberate slowdown of the recurrence.
# Retune the budget if _REFERENCE_OPERATIONS changes.
_MACHINE_SPEED_RATIO_BUDGET = 40.0

# reference workload shape: a daily loop of elementwise numpy operations on
# spatially sized arrays, mirroring the per-day update cost of a recurrence
_REFERENCE_OPERATIONS = 12

# slack on the modeled full-history footprint in the peak-memory assertion
_MEMORY_SLACK = 3.0

# bytes per float64 value stored in the recurrence histories
_BYTES_PER_VALUE = 8

# number of output fields the CFFWIS orchestrator retains by default
_CFFWIS_OUTPUTS = 7

# number of weather inputs the CFFWIS xarray path holds for the record
_CFFWIS_INPUTS = 4

# CF units for the four CFFWIS weather inputs
_WEATHER_UNITS = {
    "temperature_celsius": "degC",
    "relative_humidity_percent": "%",
    "wind_speed_meters_per_second": "m s-1",
    "precipitation_mm": "mm",
}


def _month_series(n_days: int) -> np.ndarray:
    """Return a non-leap-year month series covering ``n_days`` days."""
    months = np.repeat(np.arange(1, 13), _MONTH_LENGTHS)
    return np.tile(months, n_days // 365 + 2)[:n_days]


def _weather_arrays(n_days: int, n_cells: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Return plausible NumPy weather inputs for one fire-index call."""
    rng = np.random.default_rng(seed)
    shape = (n_days, n_cells)
    return {
        "temperature_celsius": rng.uniform(10.0, 35.0, shape),
        "relative_humidity_percent": rng.uniform(20.0, 90.0, shape),
        "wind_speed_meters_per_second": rng.uniform(0.0, 8.0, shape),
        "precipitation_mm": rng.choice([0.0, 1.0, 5.0], shape, p=[0.7, 0.2, 0.1]),
        "latitude_degrees_north": np.full(n_cells, 39.0),
        "month": _month_series(n_days),
    }


def _measure(fn, repeats: int = _REPEATS) -> float:
    """Return the best of ``repeats`` wall-clock timings of ``fn``, in seconds."""
    fn()  # warmup, excludes first-call import and allocation costs
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _reference_workload(n_days: int, n_cells: int) -> None:
    """Run representative elementwise numpy work over the same daily grid."""
    rng = np.random.default_rng(0)
    day = rng.uniform(0.0, 1.0, n_cells)
    accumulator = np.zeros(n_cells)
    for _ in range(n_days):
        for _ in range(_REFERENCE_OPERATIONS):
            accumulator = np.clip(accumulator + day, 0.0, 100.0)


def _single_pass_cffwis(weather: dict[str, np.ndarray]) -> None:
    """Compute every CFFWIS quantity through the one-pass orchestrator."""
    fire.cffwis(
        weather["temperature_celsius"],
        weather["relative_humidity_percent"],
        weather["wind_speed_meters_per_second"],
        weather["precipitation_mm"],
        weather["latitude_degrees_north"],
        weather["month"],
    )


def _chained_cffwis(weather: dict[str, np.ndarray]) -> None:
    """Compute every CFFWIS quantity the separate code calls produce."""
    temperature = weather["temperature_celsius"]
    humidity = weather["relative_humidity_percent"]
    wind = weather["wind_speed_meters_per_second"]
    precipitation = weather["precipitation_mm"]
    latitude = weather["latitude_degrees_north"]
    month = weather["month"]

    ffmc = fire.ffmc(temperature, humidity, wind, precipitation)
    dmc = fire.duff_moisture_code(temperature, humidity, precipitation, latitude, month)
    dc = fire.drought_code(temperature, precipitation, latitude, month)
    isi = fire.initial_spread_index(ffmc, wind)
    bui = fire.buildup_index(dmc, dc)
    fire.daily_severity_rating(fire.cffwis_fwi(isi, bui))


def _assert_ratio_within_budget(
    label: str,
    measured_seconds: float,
    reference_seconds: float,
    budget: float,
    guidance: str,
) -> None:
    """Assert that ``measured_seconds`` stays within ``budget`` times the reference."""
    ratio = measured_seconds / reference_seconds
    assert ratio <= budget, (
        f"{label} took {measured_seconds:.3f}s, {ratio:.2f}x the {reference_seconds:.3f}s reference "
        f"(budget {budget}x): {guidance}"
    )


def _assert_peak_within_model(peak_delta_mb: float, modeled_mb: float, slack: float) -> None:
    """Assert that measured peak RSS stays within ``slack`` times the modeled footprint."""
    assert peak_delta_mb <= slack * modeled_mb, (
        f"CFFWIS peak RSS delta {peak_delta_mb:.0f} MB exceeds {slack}x the modeled footprint "
        f"{modeled_mb:.0f} MB: the adapter materializes copies the chunked path should avoid"
    )


def _chunked_cffwis_inputs(n_days: int, n_side: int, chunk_side: int, seed: int = 42):
    """Return Dask-backed CFFWIS weather inputs chunked across space only."""
    import dask.array as da
    import pandas as pd
    import xarray as xr

    time_coord = pd.date_range("2015-01-01 12:00", periods=n_days, freq="D")
    lat_coord = np.linspace(30.0, 45.0, n_side)
    lon_coord = np.linspace(-120.0, -100.0, n_side)
    weather = _weather_arrays(n_days, n_side * n_side, seed=seed)

    inputs = {}
    for name, units in _WEATHER_UNITS.items():
        chunked = da.from_array(weather[name].reshape(n_days, n_side, n_side), chunks=(n_days, chunk_side, chunk_side))
        inputs[name] = xr.DataArray(
            chunked,
            coords={"time": time_coord, "lat": lat_coord, "lon": lon_coord},
            dims=["time", "lat", "lon"],
            attrs={"units": units},
            name=name,
        )

    return inputs


@pytest.mark.benchmark(group="fire-scaling")
class TestFireThroughput:
    """Throughput benchmarks for the recursive fire indices."""

    @pytest.mark.parametrize("n_side", _GRID_SIDES)
    def test_cffwis_grid_scaling(self, n_side: int, benchmark) -> None:
        """Benchmark full CFFWIS across grid sizes at a fixed record length."""
        n_cells = n_side * n_side
        n_days = _RECORD_DAYS[-1]
        weather = _weather_arrays(n_days, n_cells)
        benchmark(lambda: _single_pass_cffwis(weather))
        rate = n_cells * n_days / _measure(lambda: _single_pass_cffwis(weather)) / 1e6
        print(f"\nCFFWIS {n_side}x{n_side}={n_cells} cells x {n_days} d: {rate:.1f} M cell-days/s")

    @pytest.mark.parametrize("n_days", _RECORD_DAYS)
    def test_cffwis_record_scaling(self, n_days: int, benchmark) -> None:
        """Benchmark full CFFWIS across record lengths at a fixed grid size."""
        n_cells = _GRID_SIDES[-1] ** 2
        weather = _weather_arrays(n_days, n_cells)
        benchmark(lambda: _single_pass_cffwis(weather))
        rate = n_cells * n_days / _measure(lambda: _single_pass_cffwis(weather)) / 1e6
        print(f"\nCFFWIS {n_cells} cells x {n_days} d: {rate:.1f} M cell-days/s")

    @pytest.mark.parametrize("n_side", _GRID_SIDES)
    def test_kbdi_grid_scaling(self, n_side: int, benchmark) -> None:
        """Benchmark KBDI across grid sizes at a fixed record length."""
        n_cells = n_side * n_side
        n_days = _RECORD_DAYS[-1]
        weather = _weather_arrays(n_days, n_cells)
        mean_annual = np.full(n_cells, 600.0)

        def run() -> None:
            fire.kbdi(weather["precipitation_mm"], weather["temperature_celsius"], mean_annual)

        benchmark(run)
        rate = n_cells * n_days / _measure(run) / 1e6
        print(f"\nKBDI {n_side}x{n_side}={n_cells} cells x {n_days} d: {rate:.1f} M cell-days/s")


@pytest.mark.benchmark(group="fire-orchestrator")
class TestOrchestratorCost:
    """Benchmarks for the single-pass CFFWIS orchestrator against chained calls."""

    def test_cffwis_single_pass(self, benchmark) -> None:
        """Benchmark the one-pass orchestrator."""
        weather = _weather_arrays(_GUARD_RECORD_DAYS, _GUARD_GRID_SIDE**2)
        benchmark(lambda: _single_pass_cffwis(weather))

    def test_cffwis_chained_calls(self, benchmark) -> None:
        """Benchmark the separately chained code calls and derived indices."""
        weather = _weather_arrays(_GUARD_RECORD_DAYS, _GUARD_GRID_SIDE**2)
        benchmark(lambda: _chained_cffwis(weather))


@pytest.mark.benchmark(group="fire-memory")
class TestPeakMemoryByChunk:
    """Peak-RSS benchmarks for the CFFWIS xarray path."""

    @pytest.mark.parametrize("chunk_side", _CHUNK_SIDES)
    def test_cffwis_peak_rss_by_chunk_side(self, chunk_side: int, benchmark) -> None:
        """Benchmark peak RSS for one spatial chunk configuration."""
        inputs = _chunked_cffwis_inputs(_MEMORY_RECORD_DAYS, _MEMORY_GRID_SIDE, chunk_side)

        def load() -> None:
            fire.cffwis(**inputs).load()

        with _PeakRSSMonitor() as monitor:
            load()

        benchmark(load)
        cells = _MEMORY_GRID_SIDE**2
        recorded_mb = _MEMORY_RECORD_DAYS * min(chunk_side, _MEMORY_GRID_SIDE) ** 2 * _BYTES_PER_VALUE / 1e6
        print(
            f"\nchunk {chunk_side}x{chunk_side} on {_MEMORY_GRID_SIDE}x{_MEMORY_GRID_SIDE} at "
            f"{_MEMORY_RECORD_DAYS} d ({cells * _MEMORY_RECORD_DAYS / 1e6:.1f} M cell-days): "
            f"peak RSS delta {monitor.peak_delta_mb:.0f} MB, one field of one chunk {recorded_mb:.1f} MB"
        )

    @pytest.mark.parametrize("outputs", [None, ("fwi",)], ids=["all-outputs", "fwi-only"])
    def test_cffwis_peak_rss_by_output_selection(self, outputs, benchmark) -> None:
        """Benchmark peak RSS for the full output set against one selected field."""
        inputs = _chunked_cffwis_inputs(_MEMORY_RECORD_DAYS, _MEMORY_GRID_SIDE, _CHUNK_SIDES[-1])

        def load() -> None:
            fire.cffwis(**inputs, outputs=outputs).load()

        with _PeakRSSMonitor() as monitor:
            load()

        benchmark(load)
        print(f"\noutputs={outputs or 'all'}: peak RSS delta {monitor.peak_delta_mb:.0f} MB")


@pytest.mark.benchmark(group="fire-guards")
class TestFireRegressionGuards:
    """Timed and RSS guards; run by the benchmarks workflow on every pull request."""

    def test_orchestrator_not_slower_than_chained_calls(self) -> None:
        """Verify the single-pass orchestrator costs no more than chained calls."""
        weather = _weather_arrays(_GUARD_RECORD_DAYS, _GUARD_GRID_SIDE**2)
        single_pass = _measure(lambda: _single_pass_cffwis(weather))
        chained = _measure(lambda: _chained_cffwis(weather))

        _assert_ratio_within_budget(
            "single-pass CFFWIS",
            single_pass,
            chained,
            _ORCHESTRATOR_RATIO_BUDGET,
            "the orchestrator recomputes work the chained calls share, or a recurrence regressed",
        )

    def test_recurrence_within_machine_speed_budget(self) -> None:
        """Verify CFFWIS runtime stays within budget relative to reference numpy work."""
        weather = _weather_arrays(_GUARD_RECORD_DAYS, _GUARD_GRID_SIDE**2)
        fire_seconds = _measure(lambda: _single_pass_cffwis(weather))
        reference_seconds = _measure(lambda: _reference_workload(_GUARD_RECORD_DAYS, _GUARD_GRID_SIDE**2))

        _assert_ratio_within_budget(
            "CFFWIS",
            fire_seconds,
            reference_seconds,
            _MACHINE_SPEED_RATIO_BUDGET,
            "the recurrence slowed down, or the index gained per-day work",
        )

    def test_chunking_holds_peak_memory_near_the_model(self) -> None:
        """Verify chunked CFFWIS peak RSS stays within reach of the modeled footprint."""
        n_days = _MEMORY_RECORD_DAYS
        n_side = _MEMORY_GRID_SIDE
        inputs = _chunked_cffwis_inputs(n_days, n_side, _CHUNK_SIDES[0])

        with _PeakRSSMonitor() as monitor:
            fire.cffwis(**inputs).load()

        values = n_days * n_side * n_side
        modeled_mb = values * _BYTES_PER_VALUE * (_CFFWIS_OUTPUTS + _CFFWIS_INPUTS) / 1e6

        _assert_peak_within_model(monitor.peak_delta_mb, modeled_mb, _MEMORY_SLACK)


class TestFireBudgetPolicy:
    """Test the guards' pass/fail policy without relying on wall-clock measurements."""

    def test_accepts_measured_orchestrator_ratio(self) -> None:
        """The measured single-pass overhead keeps its headroom."""
        _assert_ratio_within_budget(
            "single-pass CFFWIS",
            measured_seconds=0.0676,
            reference_seconds=0.0745,
            budget=_ORCHESTRATOR_RATIO_BUDGET,
            guidance="unused",
        )

    def test_rejects_slow_orchestrator_with_diagnostics(self) -> None:
        """An orchestrator slower than the chained calls fails with both timings."""
        with pytest.raises(AssertionError) as exc_info:
            _assert_ratio_within_budget(
                "single-pass CFFWIS",
                measured_seconds=0.1200,
                reference_seconds=0.0750,
                budget=_ORCHESTRATOR_RATIO_BUDGET,
                guidance="the orchestrator recomputes work the chained calls share",
            )

        expected_message = (
            "single-pass CFFWIS took 0.120s, 1.60x the 0.075s reference (budget 1.25x): "
            "the orchestrator recomputes work the chained calls share"
        )
        assert str(exc_info.value).splitlines()[0] == expected_message

    def test_accepts_measured_machine_speed_ratio(self) -> None:
        """The measured candidate-to-reference ratio keeps its margin."""
        _assert_ratio_within_budget(
            "CFFWIS",
            measured_seconds=0.0676,
            reference_seconds=0.0058,
            budget=_MACHINE_SPEED_RATIO_BUDGET,
            guidance="unused",
        )

    def test_rejects_recurrence_slowdown_at_budget(self) -> None:
        """A slowdown past the machine-speed budget fails, at the boundary included."""
        with pytest.raises(AssertionError):
            _assert_ratio_within_budget(
                "CFFWIS",
                measured_seconds=0.0058 * _MACHINE_SPEED_RATIO_BUDGET * 1.01,
                reference_seconds=0.0058,
                budget=_MACHINE_SPEED_RATIO_BUDGET,
                guidance="the recurrence slowed down",
            )

    def test_accepts_measured_peak_within_model(self) -> None:
        """The measured peak RSS keeps its slack against the modeled footprint."""
        _assert_peak_within_model(peak_delta_mb=2337.0, modeled_mb=2630.0, slack=_MEMORY_SLACK)

    def test_rejects_peak_beyond_slack(self) -> None:
        """Extra materialization past the modeled footprint fails with both numbers."""
        with pytest.raises(AssertionError) as exc_info:
            _assert_peak_within_model(peak_delta_mb=9000.0, modeled_mb=2630.0, slack=_MEMORY_SLACK)

        expected_message = (
            "CFFWIS peak RSS delta 9000 MB exceeds 3.0x the modeled footprint 2630 MB: "
            "the adapter materializes copies the chunked path should avoid"
        )
        assert str(exc_info.value).splitlines()[0] == expected_message

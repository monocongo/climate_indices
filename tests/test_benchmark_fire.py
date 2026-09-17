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
  ``time`` must stay a single chunk (ADR-0003, ADR-0006), so peak RSS grows with
  the record length, and the retained-history model is a lower bound: measured
  peak for the xarray path runs about twice the model because the path holds
  copies of the inputs and outputs alongside the histories. ``outputs=`` reduces
  the modeled retained data, which did not show up as a peak-RSS reduction at the
  measured scales.

The NumPy-versus-``numba`` comparison in #812's task list is superseded by
docs/adr/0006-fire-recursive-state-and-execution.md, which records that ``numba``
is not an optional dependency and is reconsidered only when a representative
benchmark misses an agreed runtime target. The measurements here are that
representative benchmark: no accelerator is added, and the decision stands.

Timed tests are marked with @pytest.mark.benchmark and excluded from default test
runs. The benchmarks workflow runs them on every pull request
(.github/workflows/benchmarks.yml), including the budget guards in
TestFireRegressionGuards. Those guards fail once the recurrence costs about 1.7x
more per cell-day than the reference workload, once the orchestrator costs more
than 1.1x the chained calls it replaces, or once peak RSS leaves the coarse bound
around the modeled footprint. TestFireBudgetPolicy covers the guards' failure
paths in the default suite without depending on wall-clock measurements. Run the
marked tests explicitly with: pytest -m benchmark --benchmark-enable

Scale is configured through environment variables, all parsed as comma-separated
integers (CI-friendly defaults in parentheses):

- BENCHMARK_FIRE_GRID_SIDES (8,16,32): grid sides for the throughput benchmarks,
  each measured at the longest configured record length
- BENCHMARK_FIRE_RECORD_DAYS (365,730): record lengths for the throughput
  benchmarks, the last of which the grid benchmarks use
- BENCHMARK_FIRE_CHUNK_SIDES (8,16): spatial chunk sides for the peak-memory
  benchmarks
- BENCHMARK_FIRE_MEMORY_GRID_SIDE (32): grid side for the peak-memory benchmarks
- BENCHMARK_FIRE_MEMORY_RECORD_DAYS (365): record length for those benchmarks

The published sizing table in docs/wildfire_applications.md came from three runs:
`BENCHMARK_FIRE_GRID_SIDES=256 BENCHMARK_FIRE_RECORD_DAYS=365`,
`BENCHMARK_FIRE_GRID_SIDES=1000 BENCHMARK_FIRE_RECORD_DAYS=30`, and
`BENCHMARK_FIRE_MEMORY_GRID_SIDE=128 BENCHMARK_FIRE_MEMORY_RECORD_DAYS=1825
BENCHMARK_FIRE_CHUNK_SIDES=32,64,128`. The peak-memory guard is a coarse smoke
test at the default scale and only becomes a real bound at settings like those.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from climate_indices import fire

# repository root, so the memory probes can import this module as tests.*
_REPO_ROOT = Path(__file__).resolve().parents[1]

# probe run in a fresh interpreter per configuration. A peak-RSS delta taken
# in-process is order-dependent: an earlier test in the same session leaves RSS at
# its high-water mark, so later deltas understate their run, and one configuration
# can report zero growth. A fresh process per configuration removes the ordering
# effect and makes the published peak RSS comparable across rows.
_PEAK_RSS_PROBE = """
import json
import resource
import sys

from climate_indices import fire

from tests.test_benchmark_fire import _chunked_cffwis_inputs

if {run_cffwis}:
    inputs = _chunked_cffwis_inputs({n_days}, {n_side}, {chunk_side})
    fire.cffwis(**inputs, outputs={outputs}).load()

peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# Linux reports peak RSS in KiB, macOS in bytes
peak_mb = peak / 1024 if sys.platform != "darwin" else peak / 1024 / 1024
print(json.dumps({{"peak_rss_mb": peak_mb}}))
"""

# CFFWIS needs a noon-local-standard-time month per day, and the NumPy layer
# carries no calendar, so the benchmarks supply a non-leap-year month series.
_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _env_ints(name: str, default: str) -> tuple[int, ...]:
    """Read a comma-separated integer list from the environment."""
    raw = os.getenv(name, default)
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{name} must hold at least one integer, got {raw!r}")
    return values


# grid sides and record lengths exercised by the throughput benchmarks
_GRID_SIDES = _env_ints("BENCHMARK_FIRE_GRID_SIDES", "8,16,32")
_RECORD_DAYS = _env_ints("BENCHMARK_FIRE_RECORD_DAYS", "365,730")

# spatial chunk sides exercised by the peak-memory benchmark
_CHUNK_SIDES = _env_ints("BENCHMARK_FIRE_CHUNK_SIDES", "8,16")

# grid side and record length used by the peak-memory benchmark. The defaults are
# CI-sized as a smoke test; the published chunk-size table used 128 with a
# five-year record, where the retained daily histories dominate RSS instead of
# fixed overhead.
_MEMORY_GRID_SIDE = _env_ints("BENCHMARK_FIRE_MEMORY_GRID_SIDE", "32")[0]
_MEMORY_RECORD_DAYS = _env_ints("BENCHMARK_FIRE_MEMORY_RECORD_DAYS", "365")[0]

# fixed scale for the deterministic guards: large enough that one call is well
# above timer noise, small enough for every CI runner
_GUARD_GRID_SIDE = 32
_GUARD_RECORD_DAYS = 365

# timing repetitions per measurement (best-of, which filters CI noise)
_REPEATS = 3

# Ratio budget for the orchestrator guard: the permitted single-pass/chained
# cost, not the measured ratio. Best-of-three ratios varied by ~1% across five
# runs on the development machine (0.898-0.907); 1.10 leaves ~20% headroom for
# runner variance and still fails an orchestrator that costs a fifth more than
# the chained calls it replaces.
_ORCHESTRATOR_RATIO_BUDGET = 1.10

# Ratio budget for the machine-speed guard: CFFWIS seconds divided by the seconds
# of an equivalent-size numpy reference workload. Dividing by a workload measured
# in the same process cancels most runner-to-runner speed differences, so the
# budget only has to absorb timer variance, not hardware variance: five runs on
# the development machine measured 11.27-11.76. 20 keeps 1.7x margin and fails on
# a change that roughly doubles the recurrence's per-cell-day cost, which a budget
# sized only for runner spread would miss. Retune if _REFERENCE_OPERATIONS
# changes.
_MACHINE_SPEED_RATIO_BUDGET = 20.0

# machine-speed ratio measured on the development machine (worst of five runs at
# the guard scale); the budget comments above explain how it sets each bound
_MEASURED_ORCHESTRATOR_RATIO = 0.907
_MEASURED_MACHINE_SPEED_RATIO = 11.76

# reference workload shape: a daily loop of elementwise numpy operations on
# spatially sized arrays, mirroring the per-day update cost of a recurrence
_REFERENCE_OPERATIONS = 12

# slack on the modeled retained-history footprint in the peak-memory assertion.
# Coarse by design: it catches an adapter that materializes several times over
# the model, which is what losing the streaming path looks like at this scale.
# The measured peak ran 1.9-2.5x the model at the published settings (the path
# holds copies of the inputs and outputs next to the histories), so 4.0 leaves
# ~1.6x margin over the worst measurement.
_MEMORY_SLACK = 4.0

# number of output histories the CFFWIS orchestrator retains by default, used by
# the peak-memory model below
_CFFWIS_OUTPUTS = 7

# number of weather inputs the CFFWIS xarray path holds for the record
_CFFWIS_INPUTS = 4

# bytes per float64 value stored in the recurrence histories
_BYTES_PER_VALUE = 8

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

    if n_side % chunk_side:
        raise ValueError(f"chunk side {chunk_side} does not divide the {n_side}-cell grid side")

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


def _measure_peak_rss_mb(n_days: int, n_side: int, chunk_side: int, outputs=None, run_cffwis: bool = True) -> float:
    """Return the peak RSS of a fresh process that runs one chunked CFFWIS call.

    ``run_cffwis=False`` measures the interpreter and import baseline instead, so
    callers can separate the run's own footprint from fixed overhead.
    """
    probe = _PEAK_RSS_PROBE.format(
        run_cffwis=run_cffwis,
        n_days=n_days,
        n_side=n_side,
        chunk_side=chunk_side,
        outputs=repr(outputs),
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])["peak_rss_mb"]


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
    """Peak-RSS benchmarks for the CFFWIS xarray path, one fresh process per row."""

    @pytest.mark.parametrize("chunk_side", _CHUNK_SIDES)
    def test_cffwis_peak_rss_by_chunk_side(self, chunk_side: int) -> None:
        """Measure peak RSS for one spatial chunk configuration."""
        peak_mb = _measure_peak_rss_mb(_MEMORY_RECORD_DAYS, _MEMORY_GRID_SIDE, chunk_side)

        cells = _MEMORY_GRID_SIDE**2
        recorded_mb = _MEMORY_RECORD_DAYS * min(chunk_side, _MEMORY_GRID_SIDE) ** 2 * _BYTES_PER_VALUE / 1e6
        print(
            f"\nchunk {chunk_side}x{chunk_side} on {_MEMORY_GRID_SIDE}x{_MEMORY_GRID_SIDE} at "
            f"{_MEMORY_RECORD_DAYS} d ({cells * _MEMORY_RECORD_DAYS / 1e6:.1f} M cell-days): "
            f"peak RSS {peak_mb:.0f} MB, one field of one chunk {recorded_mb:.1f} MB"
        )

    @pytest.mark.parametrize("outputs", [None, ("fwi",)], ids=["all-outputs", "fwi-only"])
    def test_cffwis_peak_rss_by_output_selection(self, outputs) -> None:
        """Measure peak RSS for the full output set against a selected field set."""
        peak_mb = _measure_peak_rss_mb(_MEMORY_RECORD_DAYS, _MEMORY_GRID_SIDE, _CHUNK_SIDES[-1], outputs=outputs)
        print(f"\noutputs={outputs or 'all'}: peak RSS {peak_mb:.0f} MB")


@pytest.mark.benchmark(group="fire-guards")
class TestFireRegressionGuards:
    """Timed and RSS guards; run by the benchmarks workflow on every pull request."""

    def test_orchestrator_not_slower_than_chained_calls(self) -> None:
        """Verify the single-pass orchestrator stays within 1.1x the chained calls."""
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
        peak_mb = _measure_peak_rss_mb(n_days, n_side, _CHUNK_SIDES[0])
        baseline_mb = _measure_peak_rss_mb(n_days, n_side, _CHUNK_SIDES[0], run_cffwis=False)

        values = n_days * n_side * n_side
        modeled_mb = values * _BYTES_PER_VALUE * (_CFFWIS_OUTPUTS + _CFFWIS_INPUTS) / 1e6

        _assert_peak_within_model(peak_mb - baseline_mb, modeled_mb, _MEMORY_SLACK)


class TestFireBudgetPolicy:
    """Test the guards' pass/fail policy without relying on wall-clock measurements."""

    def test_accepts_measured_orchestrator_ratio(self) -> None:
        """The measured single-pass overhead keeps its headroom."""
        _assert_ratio_within_budget(
            "single-pass CFFWIS",
            measured_seconds=_MEASURED_ORCHESTRATOR_RATIO,
            reference_seconds=1.0,
            budget=_ORCHESTRATOR_RATIO_BUDGET,
            guidance="unused",
        )

    def test_rejects_slow_orchestrator_with_diagnostics(self) -> None:
        """An orchestrator slower than the chained calls fails with both timings."""
        with pytest.raises(AssertionError) as exc_info:
            _assert_ratio_within_budget(
                "single-pass CFFWIS",
                measured_seconds=1.20,
                reference_seconds=1.0,
                budget=_ORCHESTRATOR_RATIO_BUDGET,
                guidance="the orchestrator recomputes work the chained calls share",
            )

        expected_message = (
            "single-pass CFFWIS took 1.200s, 1.20x the 1.000s reference (budget 1.1x): "
            "the orchestrator recomputes work the chained calls share"
        )
        assert str(exc_info.value).splitlines()[0] == expected_message

    def test_accepts_measured_machine_speed_ratio(self) -> None:
        """The measured candidate-to-reference ratio keeps its margin."""
        _assert_ratio_within_budget(
            "CFFWIS",
            measured_seconds=_MEASURED_MACHINE_SPEED_RATIO,
            reference_seconds=1.0,
            budget=_MACHINE_SPEED_RATIO_BUDGET,
            guidance="unused",
        )

    def test_rejects_recurrence_slowdown_past_budget(self) -> None:
        """A slowdown past the machine-speed budget fails, a doubling included."""
        with pytest.raises(AssertionError):
            _assert_ratio_within_budget(
                "CFFWIS",
                measured_seconds=_MACHINE_SPEED_RATIO_BUDGET * 1.01,
                reference_seconds=1.0,
                budget=_MACHINE_SPEED_RATIO_BUDGET,
                guidance="the recurrence slowed down",
            )

        # a recurrence costing twice as much per cell-day as today must trip it
        assert 2.0 * _MEASURED_MACHINE_SPEED_RATIO > _MACHINE_SPEED_RATIO_BUDGET

    def test_accepts_measured_peak_within_model(self) -> None:
        """The worst measured peak RSS keeps its slack against the modeled footprint."""
        _assert_peak_within_model(peak_delta_mb=2.5 * 2630.0, modeled_mb=2630.0, slack=_MEMORY_SLACK)

    def test_rejects_peak_beyond_slack(self) -> None:
        """Extra materialization past the modeled footprint fails with both numbers."""
        with pytest.raises(AssertionError) as exc_info:
            _assert_peak_within_model(peak_delta_mb=11000.0, modeled_mb=2630.0, slack=_MEMORY_SLACK)

        expected_message = (
            "CFFWIS peak RSS delta 11000 MB exceeds 4.0x the modeled footprint 2630 MB: "
            "the adapter materializes copies the chunked path should avoid"
        )
        assert str(exc_info.value).splitlines()[0] == expected_message

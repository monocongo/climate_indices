"""
Performance overhead benchmarks for xarray adapter layer.

Validates FR-PERF-001 and NFR-PERF-001:
- xarray fixed per-call overhead stays below a shared absolute budget
- CI fails if benchmarks regress beyond that budget

Timed tests are marked with @pytest.mark.benchmark and excluded from default test
runs; deterministic budget-policy tests run normally. Run timed tests explicitly
with: pytest -m benchmark --benchmark-enable
"""

from __future__ import annotations

from statistics import median
from timeit import timeit

import numpy as np
import pytest
import xarray as xr

from climate_indices import indices, spei, spi
from climate_indices.compute import Periodicity
from climate_indices.eto import eto_hargreaves
from climate_indices.indices import Distribution
from climate_indices.xarray_adapter import pet_hargreaves, pet_thornthwaite

# measurement parameters for stable paired overhead measurement
_OVERHEAD_REPEAT = 8  # equal trials per order (median filters CI noise)
_OVERHEAD_NUMBER = 3  # calls per trial (amortizes per-call overhead)
# Recent hosted runs measured up to ~2.5ms fixed adapter cost; 3ms preserves
# runner headroom. For gridded data, this cost is amortized across spatial points.
_OVERHEAD_BUDGET_SECONDS = 0.003


def _assert_overhead_within_budget(
    operation: str,
    numpy_time: float,
    xarray_time: float,
    overhead: float,
) -> None:
    """Assert that measured xarray fixed overhead is below the shared budget."""
    assert overhead < _OVERHEAD_BUDGET_SECONDS, (
        f"{operation} xarray fixed overhead {overhead * 1000:.3f}ms meets or exceeds "
        f"{_OVERHEAD_BUDGET_SECONDS * 1000:.3f}ms budget "
        f"(numpy={numpy_time * 1000:.3f}ms, xarray={xarray_time * 1000:.3f}ms)"
    )


def _pet_hargreaves_numpy(
    daily_tmin_celsius: np.ndarray,
    daily_tmax_celsius: np.ndarray,
    latitude_degrees: float,
) -> np.ndarray:
    """
    NumPy PET Hargreaves path with equivalent work to the xarray adapter.

    Computes daily mean temperature in-function so timed baseline matches
    xarray path behavior (which derives tmean internally).
    """
    return eto_hargreaves(
        daily_tmin_celsius=daily_tmin_celsius,
        daily_tmax_celsius=daily_tmax_celsius,
        daily_tmean_celsius=(daily_tmin_celsius + daily_tmax_celsius) / 2.0,
        latitude_degrees=latitude_degrees,
    )


class TestOverheadBudgetPolicy:
    """Test pass/fail policy without relying on wall-clock measurements."""

    def test_shared_budget_accepts_observed_fixed_cost(self) -> None:
        """Observed adapter cost retains hosted-runner headroom."""
        _assert_overhead_within_budget(
            "PET Hargreaves",
            numpy_time=0.002,
            xarray_time=0.0049,
            overhead=0.0029,
        )

    def test_shared_budget_rejects_material_slowdown_with_diagnostics(self) -> None:
        """A material slowdown fails with both timings and the budget visible."""
        with pytest.raises(AssertionError) as exc_info:
            _assert_overhead_within_budget(
                "PET Hargreaves",
                numpy_time=0.002,
                xarray_time=0.0051,
                overhead=0.0031,
            )

        expected_message = (
            "PET Hargreaves xarray fixed overhead 3.100ms meets or exceeds 3.000ms budget "
            "(numpy=2.000ms, xarray=5.100ms)"
        )
        assert str(exc_info.value).splitlines()[0] == expected_message

    def test_shared_budget_rejects_overhead_at_budget(self) -> None:
        """An overhead equal to the budget fails because the threshold is strict."""
        with pytest.raises(AssertionError) as exc_info:
            _assert_overhead_within_budget(
                "PET Hargreaves",
                numpy_time=0.0,
                xarray_time=_OVERHEAD_BUDGET_SECONDS,
                overhead=_OVERHEAD_BUDGET_SECONDS,
            )

        expected_message = (
            "PET Hargreaves xarray fixed overhead 3.000ms meets or exceeds 3.000ms budget "
            "(numpy=0.000ms, xarray=3.000ms)"
        )
        assert str(exc_info.value).splitlines()[0] == expected_message

    def test_measurement_pairs_trials_and_uses_median_delta(self, monkeypatch) -> None:
        """Alternating pairs reject phase drift and one-off timing noise."""

        def numpy_fn() -> None:
            pass

        def xarray_fn() -> None:
            pass

        timings = iter(
            [
                (numpy_fn, 2.0),
                (xarray_fn, 5.1),
                (xarray_fn, 4.5),
                (numpy_fn, 2.6),
            ]
            * 4
        )

        def fake_timeit(fn, *, number: int) -> float:
            expected_fn, elapsed = next(timings)
            assert fn is expected_fn
            return elapsed * number

        monkeypatch.setitem(globals(), "timeit", fake_timeit)

        measured = TestOverheadThreshold._measure_overhead(
            numpy_fn,
            xarray_fn,
            trials=8,
            number=2,
        )

        assert measured == pytest.approx((2.3, 4.8, 2.5))
        assert next(timings, None) is None


# ==============================================================================
# SPI benchmarks
# ==============================================================================


@pytest.mark.benchmark(group="spi-1d")
class TestSPIBenchmark:
    """Benchmark SPI computation: NumPy baseline vs xarray path."""

    def test_numpy_baseline(self, benchmark, bench_monthly_precip_np: np.ndarray) -> None:
        """NumPy SPI baseline (2D array, gamma distribution, 6-month scale)."""
        benchmark(
            indices.spi,
            values=bench_monthly_precip_np,
            scale=6,
            distribution=Distribution.gamma,
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2019,
            periodicity=Periodicity.monthly,
        )

    def test_xarray_path(self, benchmark, bench_monthly_precip_da: xr.DataArray) -> None:
        """xarray SPI path (1D DataArray, gamma distribution, 6-month scale)."""
        benchmark(
            spi,
            values=bench_monthly_precip_da,
            scale=6,
            distribution=Distribution.gamma,
        )


# ==============================================================================
# SPEI benchmarks
# ==============================================================================


@pytest.mark.benchmark(group="spei-1d")
class TestSPEIBenchmark:
    """Benchmark SPEI computation: NumPy baseline vs xarray path."""

    def test_numpy_baseline(
        self,
        benchmark,
        bench_monthly_precip_np: np.ndarray,
        bench_monthly_pet_np: np.ndarray,
    ) -> None:
        """NumPy SPEI baseline (2D arrays, gamma distribution, 6-month scale)."""
        benchmark(
            indices.spei,
            precips_mm=bench_monthly_precip_np,
            pet_mm=bench_monthly_pet_np,
            scale=6,
            distribution=Distribution.gamma,
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2019,
            periodicity=Periodicity.monthly,
        )

    def test_xarray_path(
        self,
        benchmark,
        bench_monthly_precip_da: xr.DataArray,
        bench_monthly_pet_da: xr.DataArray,
    ) -> None:
        """xarray SPEI path (1D DataArrays, gamma distribution, 6-month scale)."""
        benchmark(
            spei,
            precips_mm=bench_monthly_precip_da,
            pet_mm=bench_monthly_pet_da,
            scale=6,
            distribution=Distribution.gamma,
        )


# ==============================================================================
# PET Thornthwaite benchmarks
# ==============================================================================


@pytest.mark.benchmark(group="pet-thornthwaite")
class TestPETThornthwaiteBenchmark:
    """Benchmark PET Thornthwaite computation: NumPy baseline vs xarray path."""

    def test_numpy_baseline(self, benchmark, bench_monthly_temp_np: np.ndarray) -> None:
        """NumPy PET Thornthwaite baseline (1D array, latitude=40.0)."""
        benchmark(
            indices.pet,
            temperature_celsius=bench_monthly_temp_np,
            latitude_degrees=40.0,
            data_start_year=1980,
        )

    def test_xarray_path(self, benchmark, bench_monthly_temp_da: xr.DataArray) -> None:
        """xarray PET Thornthwaite path (1D DataArray, latitude=40.0)."""
        benchmark(
            pet_thornthwaite,
            temperature=bench_monthly_temp_da,
            latitude=40.0,
        )


# ==============================================================================
# PET Hargreaves benchmarks
# ==============================================================================


@pytest.mark.benchmark(group="pet-hargreaves")
class TestPETHargreavesBenchmark:
    """Benchmark PET Hargreaves computation: NumPy baseline vs xarray path."""

    def test_numpy_baseline(
        self,
        benchmark,
        bench_daily_tmin_np: np.ndarray,
        bench_daily_tmax_np: np.ndarray,
    ) -> None:
        """NumPy PET Hargreaves baseline (1D arrays, latitude=40.0)."""
        benchmark(
            _pet_hargreaves_numpy,
            daily_tmin_celsius=bench_daily_tmin_np,
            daily_tmax_celsius=bench_daily_tmax_np,
            latitude_degrees=40.0,
        )

    def test_xarray_path(
        self,
        benchmark,
        bench_daily_tmin_da: xr.DataArray,
        bench_daily_tmax_da: xr.DataArray,
    ) -> None:
        """xarray PET Hargreaves path (1D DataArrays, latitude=40.0)."""
        benchmark(
            pet_hargreaves,
            daily_tmin_celsius=bench_daily_tmin_da,
            daily_tmax_celsius=bench_daily_tmax_da,
            latitude=40.0,
        )


# ==============================================================================
# gridded benchmarks (xarray-only)
# ==============================================================================


@pytest.mark.benchmark(group="gridded")
class TestGriddedBenchmark:
    """Benchmark gridded computations (xarray-only, no NumPy equivalent)."""

    def test_spi_gridded_20x20(self, benchmark, bench_gridded_precip_da: xr.DataArray) -> None:
        """Gridded SPI (480 time steps, 20×20 spatial grid, 3-month scale)."""
        benchmark(
            spi,
            values=bench_gridded_precip_da,
            scale=3,
            distribution=Distribution.gamma,
        )


# ==============================================================================
# overhead threshold assertions
# ==============================================================================


@pytest.mark.benchmark(group="overhead")
class TestOverheadThreshold:
    """
    Assert xarray overhead stays within budget (NFR-PERF-001).

    For 1D time series (worst case), overhead includes:
    - Parameter inference from time coordinates (~0.5ms for SPI/SPEI)
    - xarray apply_ufunc machinery (~0.2ms for PET functions)
    - Coordinate/metadata handling

    All measured operations use one absolute fixed-cost budget. For gridded data
    (primary use case), overhead is amortized across spatial dimensions and
    becomes negligible.

    Uses equally balanced path orders and their median deltas to filter CI noise
    and host-speed drift while catching real regressions.
    """

    @staticmethod
    def _measure_overhead(
        numpy_fn,
        xarray_fn,
        trials: int = _OVERHEAD_REPEAT,
        number: int = _OVERHEAD_NUMBER,
    ) -> tuple[float, float, float]:
        """
        Return median path timings and order-neutral fixed overhead.

        Alternates path order, then averages each order's median delta.
        Includes warmup calls to avoid first-call JIT/import effects.
        """
        if trials <= 0 or trials % 2:
            raise ValueError("trials must be a positive even number")

        # warmup
        numpy_fn()
        xarray_fn()

        measurements: list[tuple[float, float]] = []
        for trial in range(trials):
            if trial % 2:
                xarray_time = timeit(xarray_fn, number=number) / number
                numpy_time = timeit(numpy_fn, number=number) / number
            else:
                numpy_time = timeit(numpy_fn, number=number) / number
                xarray_time = timeit(xarray_fn, number=number) / number
            measurements.append((numpy_time, xarray_time))

        numpy_time = median(numpy for numpy, _ in measurements)
        xarray_time = median(xarray for _, xarray in measurements)
        numpy_first_overhead = median(xarray - numpy for numpy, xarray in measurements[::2])
        xarray_first_overhead = median(xarray - numpy for numpy, xarray in measurements[1::2])
        overhead = (numpy_first_overhead + xarray_first_overhead) / 2
        return numpy_time, xarray_time, overhead

    def test_spi_overhead(
        self,
        bench_monthly_precip_np: np.ndarray,
        bench_monthly_precip_da: xr.DataArray,
    ) -> None:
        """Verify SPI xarray overhead stays within threshold."""
        np_time, xa_time, overhead = self._measure_overhead(
            lambda: indices.spi(
                values=bench_monthly_precip_np,
                scale=6,
                distribution=Distribution.gamma,
                data_start_year=1980,
                calibration_year_initial=1980,
                calibration_year_final=2019,
                periodicity=Periodicity.monthly,
            ),
            lambda: spi(
                values=bench_monthly_precip_da,
                scale=6,
                distribution=Distribution.gamma,
            ),
        )
        _assert_overhead_within_budget("SPI", np_time, xa_time, overhead)

    def test_spei_overhead(
        self,
        bench_monthly_precip_np: np.ndarray,
        bench_monthly_pet_np: np.ndarray,
        bench_monthly_precip_da: xr.DataArray,
        bench_monthly_pet_da: xr.DataArray,
    ) -> None:
        """Verify SPEI xarray overhead stays within threshold."""
        np_time, xa_time, overhead = self._measure_overhead(
            lambda: indices.spei(
                precips_mm=bench_monthly_precip_np,
                pet_mm=bench_monthly_pet_np,
                scale=6,
                distribution=Distribution.gamma,
                data_start_year=1980,
                calibration_year_initial=1980,
                calibration_year_final=2019,
                periodicity=Periodicity.monthly,
            ),
            lambda: spei(
                precips_mm=bench_monthly_precip_da,
                pet_mm=bench_monthly_pet_da,
                scale=6,
                distribution=Distribution.gamma,
            ),
        )
        _assert_overhead_within_budget("SPEI", np_time, xa_time, overhead)

    def test_pet_thornthwaite_overhead(
        self,
        bench_monthly_temp_np: np.ndarray,
        bench_monthly_temp_da: xr.DataArray,
    ) -> None:
        """Verify PET Thornthwaite xarray overhead stays within threshold."""
        np_time, xa_time, overhead = self._measure_overhead(
            lambda: indices.pet(
                temperature_celsius=bench_monthly_temp_np,
                latitude_degrees=40.0,
                data_start_year=1980,
            ),
            lambda: pet_thornthwaite(
                temperature=bench_monthly_temp_da,
                latitude=40.0,
            ),
        )
        _assert_overhead_within_budget("PET Thornthwaite", np_time, xa_time, overhead)

    def test_pet_hargreaves_overhead(
        self,
        bench_daily_tmin_np: np.ndarray,
        bench_daily_tmax_np: np.ndarray,
        bench_daily_tmin_da: xr.DataArray,
        bench_daily_tmax_da: xr.DataArray,
    ) -> None:
        """Verify PET Hargreaves xarray overhead stays within threshold."""
        np_time, xa_time, overhead = self._measure_overhead(
            lambda: _pet_hargreaves_numpy(
                daily_tmin_celsius=bench_daily_tmin_np,
                daily_tmax_celsius=bench_daily_tmax_np,
                latitude_degrees=40.0,
            ),
            lambda: pet_hargreaves(
                daily_tmin_celsius=bench_daily_tmin_da,
                daily_tmax_celsius=bench_daily_tmax_da,
                latitude=40.0,
            ),
        )
        _assert_overhead_within_budget("PET Hargreaves", np_time, xa_time, overhead)

"""
Performance overhead benchmarks for xarray adapter layer.

Validates FR-PERF-001 and NFR-PERF-001:
- xarray path overhead budget is 80% vs NumPy for most 1D operations
- PET Hargreaves has a measured, operation-specific 100% budget
- CI fails if benchmarks regress beyond the applicable budget

Timed tests are marked with @pytest.mark.benchmark and excluded from default test
runs; deterministic budget-policy tests run normally. Run timed tests explicitly
with: pytest -m benchmark --benchmark-enable
"""

from __future__ import annotations

from timeit import repeat

import numpy as np
import pytest
import xarray as xr

from climate_indices import indices, spei, spi
from climate_indices.compute import Periodicity
from climate_indices.eto import eto_hargreaves
from climate_indices.indices import Distribution
from climate_indices.xarray_adapter import pet_hargreaves, pet_thornthwaite

# measurement parameters for stable overhead measurement using timeit.repeat
_OVERHEAD_REPEAT = 7  # independent trials (min filters CI noise)
_OVERHEAD_NUMBER = 3  # calls per trial (amortizes per-call overhead)
# The shared 80% budget accounts for xarray machinery overhead (apply_ufunc,
# coordinate handling, metadata propagation) on small 1D arrays. For gridded data
# (the primary use case), this overhead is amortized across thousands of spatial
# points and becomes negligible (<5%).
_OVERHEAD_THRESHOLD = 0.80
# Unchanged GitHub-hosted runs measured PET Hargreaves overhead from 70.9% to
# 85.7%. Its fast NumPy path makes the ratio unusually sensitive to fixed adapter
# costs and runner noise. A 100% operation-specific budget adds 14.3 percentage
# points of headroom above the observed maximum while still failing if the xarray
# path takes twice as long as the equivalent NumPy path. See issue #740.
_PET_HARGREAVES_OVERHEAD_THRESHOLD = 1.00


def _assert_overhead_within_budget(
    operation: str,
    numpy_time: float,
    xarray_time: float,
    overhead: float,
    budget: float,
) -> None:
    """Assert that measured xarray overhead is below an operation's budget."""
    assert overhead < budget, (
        f"{operation} xarray overhead {overhead:.1%} exceeds {budget:.0%} budget "
        f"(numpy={numpy_time:.4f}s, xarray={xarray_time:.4f}s)"
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

    def test_pet_hargreaves_accepts_observed_ci_variation(self) -> None:
        """The highest observed unchanged-run overhead retains noise headroom."""
        numpy_time = 0.002
        overhead = 0.857
        xarray_time = numpy_time * (1.0 + overhead)

        _assert_overhead_within_budget(
            "PET Hargreaves",
            numpy_time,
            xarray_time,
            overhead,
            _PET_HARGREAVES_OVERHEAD_THRESHOLD,
        )

    def test_pet_hargreaves_rejects_material_slowdown_with_diagnostics(self) -> None:
        """A material slowdown fails with both timings and the budget visible."""
        with pytest.raises(AssertionError) as exc_info:
            _assert_overhead_within_budget(
                "PET Hargreaves",
                numpy_time=0.002,
                xarray_time=0.0044,
                overhead=1.20,
                budget=_PET_HARGREAVES_OVERHEAD_THRESHOLD,
            )

        expected_message = "PET Hargreaves xarray overhead 120.0% exceeds 100% budget (numpy=0.0020s, xarray=0.0044s)"
        assert str(exc_info.value).splitlines()[0] == expected_message


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

    The shared budget is 80% for 1D arrays. PET Hargreaves uses its documented
    operation-specific budget because fixed adapter costs and runner noise are
    large relative to its fast NumPy baseline. For gridded data (primary use
    case), overhead is amortized across spatial dimensions and becomes negligible.

    Uses timeit.repeat with min selection (standard Python benchmarking practice)
    to filter upward outliers from CI noise while catching real regressions.
    """

    @staticmethod
    def _measure_overhead(
        numpy_fn,
        xarray_fn,
        trials: int = _OVERHEAD_REPEAT,
        number: int = _OVERHEAD_NUMBER,
    ) -> tuple[float, float, float]:
        """
        Run both paths and return (numpy_min, xarray_min, overhead_ratio).

        Uses timeit.repeat with min selection to filter CI noise (standard practice).
        Includes warmup calls to avoid first-call JIT/import effects.
        """
        # warmup
        numpy_fn()
        xarray_fn()

        # measure: repeat trials, take min per trial, normalize by calls per trial
        numpy_time = min(repeat(numpy_fn, number=number, repeat=trials)) / number
        xarray_time = min(repeat(xarray_fn, number=number, repeat=trials)) / number
        overhead = (xarray_time - numpy_time) / numpy_time if numpy_time > 0 else 0.0
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
        _assert_overhead_within_budget("SPI", np_time, xa_time, overhead, _OVERHEAD_THRESHOLD)

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
        _assert_overhead_within_budget("SPEI", np_time, xa_time, overhead, _OVERHEAD_THRESHOLD)

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
        _assert_overhead_within_budget("PET Thornthwaite", np_time, xa_time, overhead, _OVERHEAD_THRESHOLD)

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
        _assert_overhead_within_budget(
            "PET Hargreaves",
            np_time,
            xa_time,
            overhead,
            _PET_HARGREAVES_OVERHEAD_THRESHOLD,
        )

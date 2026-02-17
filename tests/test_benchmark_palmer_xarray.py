"""Performance validation for Palmer xarray wrapper (Story 4.9).

Validates that palmer_xarray() achieves >=80% speed of the NumPy baseline
(NFR-PALMER-PERF). The xarray wrapper adds overhead for coordinate handling,
metadata attachment, and Dataset construction, which must stay within budget.

The Palmer algorithm is inherently sequential (water balance state propagates
through time), so per-pixel cost is the fundamental performance unit. The
xarray wrapper iterates per-pixel for spatial grids, adding only the
xr.Dataset construction overhead on top.

We use a 50-year (600 month) time series so that the O(n) Palmer water-balance
computation dominates the fixed-cost wrapper overhead (~0.5 ms for Dataset
construction, CF metadata, structlog logging). With a shorter series the fixed
overhead is a larger fraction and gives a misleading slowdown ratio.

Tests are marked with @pytest.mark.benchmark and excluded from default runs.
Run with: pytest -m benchmark --benchmark-enable tests/test_benchmark_palmer_xarray.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices.palmer import palmer_xarray
from climate_indices.palmer import pdsi as numpy_pdsi

_RNG = np.random.default_rng(2024)
_N_MONTHS = 600  # 50 years — long enough for compute to dominate wrapper overhead
_DATA_START_YEAR = 1975
_CAL_YEAR_INITIAL = 1975
_CAL_YEAR_FINAL = 2024
_AWC = 5.0

# performance target: xarray must be >= 80% speed of numpy (at most 25% slower)
_MAX_SLOWDOWN_FACTOR = 1.25


def _make_inputs(
    n_months: int = _N_MONTHS,
    start_year: int = _DATA_START_YEAR,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Generate test data for performance measurement."""
    time_index = pd.date_range(f"{start_year}-01", periods=n_months, freq="MS")
    months = np.arange(n_months) % 12

    precip = 2.5 + 1.5 * np.sin(2 * np.pi * months / 12) + _RNG.normal(0, 0.5, n_months)
    precip = np.clip(precip, 0.0, None)
    pet = 3.0 + 2.0 * np.sin(2 * np.pi * (months - 3) / 12) + _RNG.normal(0, 0.3, n_months)
    pet = np.clip(pet, 0.1, None)

    return precip, pet, time_index


# ---------------------------------------------------------------------------
# Benchmark tests (require --benchmark-enable)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="palmer-xarray-single")
class TestPalmerXarraySinglePoint:
    """Benchmark Palmer xarray for a single grid point."""

    def test_palmer_xarray_single_point(self, benchmark) -> None:
        """Measure single-point palmer_xarray wall-clock time."""
        precip, pet, time_index = _make_inputs()
        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])

        result = benchmark(
            palmer_xarray,
            precip_da,
            pet_da,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"pdsi", "phdi", "pmdi", "z_index"}


# ---------------------------------------------------------------------------
# Relative performance test (runs without benchmark plugin)
# ---------------------------------------------------------------------------


class TestPalmerXarrayPerformance:
    """Validate xarray overhead is within budget vs numpy baseline.

    This test does not require pytest-benchmark and runs in the normal
    test suite. It uses wall-clock timing with median-of-runs to reduce
    noise from system load variation.

    The test uses a 50-year (600 month) time series so that the O(n)
    Palmer water-balance computation dominates the fixed wrapper overhead
    (~0.5 ms for Dataset construction, CF metadata, structlog logging).
    """

    def test_single_point_xarray_overhead_within_budget(self) -> None:
        """palmer_xarray single-point overhead <= 25% vs numpy pdsi.

        Measures the per-call overhead of the xarray wrapper compared to
        raw numpy pdsi(). The overhead comes from:
        - xr.DataArray coordinate validation and alignment
        - temporal parameter inference
        - xr.Dataset construction with CF metadata
        - provenance history and logging

        Uses median of individual timings for robustness against outliers.
        """
        precip, pet, time_index = _make_inputs()
        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])

        n_warmup = 3
        n_iter = 10

        # warmup both paths to stabilise JIT/cache behaviour
        for _ in range(n_warmup):
            numpy_pdsi(
                precips=precip,
                pet=pet,
                awc=_AWC,
                data_start_year=_DATA_START_YEAR,
                calibration_year_initial=_CAL_YEAR_INITIAL,
                calibration_year_final=_CAL_YEAR_FINAL,
            )
            palmer_xarray(
                precip_da,
                pet_da,
                awc=_AWC,
                data_start_year=_DATA_START_YEAR,
                calibration_year_initial=_CAL_YEAR_INITIAL,
                calibration_year_final=_CAL_YEAR_FINAL,
            )

        # collect individual timings for median calculation
        numpy_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            numpy_pdsi(
                precips=precip,
                pet=pet,
                awc=_AWC,
                data_start_year=_DATA_START_YEAR,
                calibration_year_initial=_CAL_YEAR_INITIAL,
                calibration_year_final=_CAL_YEAR_FINAL,
            )
            numpy_times.append(time.perf_counter() - t0)

        xarray_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            palmer_xarray(
                precip_da,
                pet_da,
                awc=_AWC,
                data_start_year=_DATA_START_YEAR,
                calibration_year_initial=_CAL_YEAR_INITIAL,
                calibration_year_final=_CAL_YEAR_FINAL,
            )
            xarray_times.append(time.perf_counter() - t0)

        numpy_median = float(np.median(numpy_times))
        xarray_median = float(np.median(xarray_times))
        slowdown = xarray_median / numpy_median if numpy_median > 0 else float("inf")

        # assert within budget (NFR-PALMER-PERF: >= 80% speed, i.e. at most 1.25x)
        assert slowdown <= _MAX_SLOWDOWN_FACTOR, (
            f"Palmer xarray is {slowdown:.2f}x slower than numpy "
            f"(budget: {_MAX_SLOWDOWN_FACTOR}x). "
            f"numpy_median={numpy_median * 1000:.2f}ms, "
            f"xarray_median={xarray_median * 1000:.2f}ms"
        )

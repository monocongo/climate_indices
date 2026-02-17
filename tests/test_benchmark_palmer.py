"""Performance baseline benchmarks for Palmer drought index.

Establishes wall-clock timing baselines for the current Palmer PDSI implementation
(palmer.pdsi). These baselines are used by Story 4.9 (Epic 4: Palmer Performance
Validation) to validate that the future Palmer xarray implementation achieves
>=80% speed (NFR-PALMER-PERF).

Test grid sizes:
- Single point: 1 location, 240 months (20 years) -- unit cost measurement
- Small grid: 10 points, 240 months -- representative of a regional analysis
- Medium grid: 100 points, 240 months -- representative of a country-scale analysis

Note: The full 360x180 (64,800 points) global grid is too expensive for CI
benchmarking. The single-point and small-grid benchmarks establish the per-point
cost, which can be linearly extrapolated for the global grid.

Tests are marked with @pytest.mark.benchmark and excluded from default test runs.
Run explicitly with: pytest -m benchmark --benchmark-enable tests/test_benchmark_palmer.py
"""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices.palmer import pdsi


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------

# random seed for reproducibility
_RNG = np.random.default_rng(2024)

# time series length: 20 years of monthly data
_N_MONTHS = 240
_DATA_START_YEAR = 2000
_CALIBRATION_YEAR_INITIAL = 2000
_CALIBRATION_YEAR_FINAL = 2019

# available water capacity (typical value for loamy soil, in inches)
_AWC = 5.0


def _make_palmer_inputs(n_months: int = _N_MONTHS) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic precipitation and PET arrays for Palmer benchmarking.

    Values are in inches (as required by palmer.pdsi).
    Precipitation: seasonal pattern ~2-4 in/month with noise.
    PET: seasonal pattern ~1-5 in/month following temperature cycle.

    Args:
        n_months: Number of monthly values to generate.

    Returns:
        Tuple of (precipitation, pet) arrays in inches.
    """
    # create seasonal pattern
    months = np.arange(n_months) % 12
    # precipitation: higher in spring/summer
    precip_seasonal = 2.5 + 1.5 * np.sin(2 * np.pi * months / 12)
    precip = precip_seasonal + _RNG.normal(0, 0.5, n_months)
    precip = np.clip(precip, 0.0, None)

    # PET: peaks in summer
    pet_seasonal = 3.0 + 2.0 * np.sin(2 * np.pi * (months - 3) / 12)
    pet = pet_seasonal + _RNG.normal(0, 0.3, n_months)
    pet = np.clip(pet, 0.1, None)

    return precip, pet


# ---------------------------------------------------------------------------
# Benchmark fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def palmer_single_point() -> tuple[np.ndarray, np.ndarray]:
    """Single-point Palmer input data (240 months)."""
    return _make_palmer_inputs()


@pytest.fixture(scope="session")
def palmer_small_grid() -> list[tuple[np.ndarray, np.ndarray]]:
    """10-point Palmer input data (each 240 months)."""
    return [_make_palmer_inputs() for _ in range(10)]


@pytest.fixture(scope="session")
def palmer_medium_grid() -> list[tuple[np.ndarray, np.ndarray]]:
    """100-point Palmer input data (each 240 months)."""
    return [_make_palmer_inputs() for _ in range(100)]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="palmer-single")
class TestPalmerSinglePoint:
    """Benchmark Palmer PDSI for a single grid point (20 years monthly).

    This measures the per-point computation cost of the Palmer algorithm.
    The Palmer algorithm is inherently sequential (water balance state propagates
    through time), so per-point cost is the fundamental unit of Palmer performance.
    """

    def test_pdsi_single_point(
        self,
        benchmark,
        palmer_single_point: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Measure single-point PDSI wall-clock time."""
        precip, pet = palmer_single_point
        result = benchmark(
            pdsi,
            precips=precip,
            pet=pet,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CALIBRATION_YEAR_INITIAL,
            calibration_year_final=_CALIBRATION_YEAR_FINAL,
        )
        # sanity check: should return 5-tuple
        assert len(result) == 5
        pdsi_vals, phdi_vals, pmdi_vals, zindex_vals, params = result
        assert pdsi_vals.shape == (_N_MONTHS,)
        assert params is not None


@pytest.mark.benchmark(group="palmer-grid")
class TestPalmerSmallGrid:
    """Benchmark Palmer PDSI for a small grid (10 points, sequential).

    This simulates sequential processing of multiple grid points, which is
    how the current multiprocessing CLI processes Palmer: each worker calls
    pdsi() per grid point.
    """

    def test_pdsi_10_points_sequential(
        self,
        benchmark,
        palmer_small_grid: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Measure 10-point sequential PDSI wall-clock time."""

        def run_grid() -> list:
            results = []
            for precip, pet in palmer_small_grid:
                results.append(
                    pdsi(
                        precips=precip,
                        pet=pet,
                        awc=_AWC,
                        data_start_year=_DATA_START_YEAR,
                        calibration_year_initial=_CALIBRATION_YEAR_INITIAL,
                        calibration_year_final=_CALIBRATION_YEAR_FINAL,
                    )
                )
            return results

        results = benchmark(run_grid)
        assert len(results) == 10


@pytest.mark.benchmark(group="palmer-grid")
class TestPalmerMediumGrid:
    """Benchmark Palmer PDSI for a medium grid (100 points, sequential).

    This provides a more realistic estimate of batch Palmer computation time.
    For a global 1-degree grid (360x180 = 64,800 points), the expected time
    can be extrapolated from this measurement:
        estimated_global_time = measured_100pt_time * 648
    """

    def test_pdsi_100_points_sequential(
        self,
        benchmark,
        palmer_medium_grid: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Measure 100-point sequential PDSI wall-clock time."""

        def run_grid() -> list:
            results = []
            for precip, pet in palmer_medium_grid:
                results.append(
                    pdsi(
                        precips=precip,
                        pet=pet,
                        awc=_AWC,
                        data_start_year=_DATA_START_YEAR,
                        calibration_year_initial=_CALIBRATION_YEAR_INITIAL,
                        calibration_year_final=_CALIBRATION_YEAR_FINAL,
                    )
                )
            return results

        results = benchmark(run_grid)
        assert len(results) == 100


# ---------------------------------------------------------------------------
# Memory usage measurement
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="palmer-memory")
class TestPalmerMemoryBaseline:
    """Measure Palmer memory usage baseline for a single grid point.

    Uses tracemalloc to capture peak memory allocation during Palmer computation.
    This baseline is referenced by Story 4.9 for memory regression checks.
    """

    def test_pdsi_peak_memory(
        self,
        palmer_single_point: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Measure peak memory allocation for single-point PDSI."""
        import tracemalloc

        precip, pet = palmer_single_point

        tracemalloc.start()
        _ = pdsi(
            precips=precip,
            pet=pet,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CALIBRATION_YEAR_INITIAL,
            calibration_year_final=_CALIBRATION_YEAR_FINAL,
        )
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_kb / (1024 * 1024)

        # document the baseline -- should be well under 50 MB for a single point
        assert peak_mb < 50.0, f"Palmer single-point peak memory {peak_mb:.2f} MB exceeds 50 MB budget"

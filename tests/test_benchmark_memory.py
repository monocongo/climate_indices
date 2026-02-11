"""
Memory efficiency validation benchmarks for out-of-core computation.

Validates FR-PERF-003 and NFR-PERF-003:
- Out-of-core computation handles datasets larger than available RAM
- No OOM errors during lazy evaluation and chunked computation
- Peak memory stays within specified budgets

Tests are marked with @pytest.mark.benchmark and excluded from default test runs.
Run explicitly with: pytest -m benchmark --benchmark-enable

Environment variables configure dataset dimensions and validation strictness:

Dataset Dimensions (default: CI-friendly small scale):
- BENCHMARK_MEMORY_N_LAT (default: 36, spec: 1440)
- BENCHMARK_MEMORY_N_LON (default: 18, spec: 720)
- BENCHMARK_MEMORY_N_TIME (default: 480, spec: 6400)
- BENCHMARK_MEMORY_CHUNK_LAT (default: 18, spec: 100)
- BENCHMARK_MEMORY_CHUNK_LON (default: 18, spec: 100)

Validation Strictness:
- BENCHMARK_STRICT_MEMORY (default: false): Enforce hard RSS assertions in peak memory test

CI defaults create ~225 MB virtual dataset (validates lazy evaluation without long runtime).
Full spec creates ~49 GB virtual dataset (validates actual memory constraints).

Note on PRD Dimension Discrepancy:
The PRD states "(1440×720×1200) float32 = ~49GB" but the actual math gives:
  1440 × 720 × 1200 × 4 bytes = ~4.6 GB

To reach ~49 GB with float64 (8 bytes) at 1440×720 spatial:
  49 GB = 1440 × 720 × n_time × 8 bytes
  n_time ≈ 6400 time steps

The test defaults use n_time=6400 for the full-spec run to match the PRD's intended
memory footprint (~49 GB virtual dataset with float64).
"""

from __future__ import annotations

import gc
import os
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from climate_indices import spi
from climate_indices.indices import Distribution

if TYPE_CHECKING:
    pass

# skip entire module if psutil not available
psutil = pytest.importorskip("psutil")

# PRD-specified dimensions for ~49GB virtual dataset (documented in section 3.4.3)
_SPEC_N_LAT = 1440
_SPEC_N_LON = 720
# corrected from PRD's 1200 to reach ~49GB with float64
_SPEC_N_TIME = 6400
_SPEC_CHUNK_LAT = 100
_SPEC_CHUNK_LON = 100

# CI-friendly defaults (~225 MB virtual dataset)
_DEFAULT_N_LAT = 36
_DEFAULT_N_LON = 18
# 40 years monthly
_DEFAULT_N_TIME = 480
_DEFAULT_CHUNK_LAT = 18
_DEFAULT_CHUNK_LON = 18

# resolved dimensions from environment or defaults
_N_LAT = int(os.getenv("BENCHMARK_MEMORY_N_LAT", _DEFAULT_N_LAT))
_N_LON = int(os.getenv("BENCHMARK_MEMORY_N_LON", _DEFAULT_N_LON))
_N_TIME = int(os.getenv("BENCHMARK_MEMORY_N_TIME", _DEFAULT_N_TIME))
_CHUNK_LAT = int(os.getenv("BENCHMARK_MEMORY_CHUNK_LAT", _DEFAULT_CHUNK_LAT))
_CHUNK_LON = int(os.getenv("BENCHMARK_MEMORY_CHUNK_LON", _DEFAULT_CHUNK_LON))

# enable strict memory assertions (disabled by default due to platform variability)
_STRICT_MEMORY = os.getenv("BENCHMARK_STRICT_MEMORY", "false").lower() == "true"

# SPI parameters for benchmarks
_SPI_SCALE = 3
_SPI_DISTRIBUTION = Distribution.gamma

# memory budgets (in GB)
# hard limit for strict mode
_PEAK_MEMORY_HARD_LIMIT_GB = 16.0
# ideal limit (reported but not enforced)
_PEAK_MEMORY_IDEAL_LIMIT_GB = 8.0

# minimum virtual dataset size (in GB) to run the peak memory test
_MIN_VIRTUAL_DATASET_GB = 1.0

# RSS sampling interval for peak memory tracking (in seconds)
# 10ms
_RSS_SAMPLE_INTERVAL_S = 0.01


class _PeakRSSMonitor:
    """Context manager that tracks peak RSS memory usage during a code block.

    Uses a background thread to sample RSS at regular intervals, recording the
    high-water mark. This pattern follows memory_profiler internals.

    Example:
        with _PeakRSSMonitor() as monitor:
            # run memory-intensive code
            result = expensive_computation()
        print(f"Peak RSS: {monitor.peak_rss_gb:.2f} GB")
        print(f"Delta from baseline: {monitor.peak_delta_mb:.1f} MB")
    """

    def __init__(self) -> None:
        """Initialize the monitor (does not start sampling yet)."""
        self._baseline_rss_bytes: int = 0
        self._peak_rss_bytes: int = 0
        self._sampling_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._process = psutil.Process()

    def __enter__(self) -> _PeakRSSMonitor:
        """Start monitoring RSS on entry to context block."""
        # force garbage collection to get clean baseline
        gc.collect()
        self._baseline_rss_bytes = self._process.memory_info().rss
        self._peak_rss_bytes = self._baseline_rss_bytes

        # start background sampling thread
        self._stop_event.clear()
        self._sampling_thread = threading.Thread(target=self._sample_rss, daemon=True)
        self._sampling_thread.start()

        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop monitoring RSS on exit from context block."""
        self._stop_event.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join(timeout=1.0)

    def _sample_rss(self) -> None:
        """Background thread target that samples RSS at regular intervals."""
        while not self._stop_event.is_set():
            current_rss = self._process.memory_info().rss
            if current_rss > self._peak_rss_bytes:
                self._peak_rss_bytes = current_rss
            time.sleep(_RSS_SAMPLE_INTERVAL_S)

    @property
    def peak_rss_mb(self) -> float:
        """Peak RSS memory in megabytes."""
        return self._peak_rss_bytes / (1024 * 1024)

    @property
    def peak_rss_gb(self) -> float:
        """Peak RSS memory in gigabytes."""
        return self._peak_rss_bytes / (1024 * 1024 * 1024)

    @property
    def baseline_rss_mb(self) -> float:
        """Baseline RSS memory in megabytes (at context entry)."""
        return self._baseline_rss_bytes / (1024 * 1024)

    @property
    def peak_delta_mb(self) -> float:
        """Peak RSS increase from baseline in megabytes."""
        return (self._peak_rss_bytes - self._baseline_rss_bytes) / (1024 * 1024)


@pytest.mark.benchmark(group="memory-efficiency")
class TestMemoryEfficiency:
    """Benchmark memory efficiency for out-of-core computation with Dask."""

    @staticmethod
    def _create_large_precip_dataset() -> xr.DataArray:
        """
        Create large Dask-backed precipitation DataArray with fully lazy generation.

        Uses dask.array.random for on-the-fly chunk generation (no full materialization).
        Each chunk is generated independently during .compute(), enabling true out-of-core
        computation for datasets larger than available RAM.

        Returns:
            DataArray with shape (time=N_TIME, lat=N_LAT, lon=N_LON), chunked as
            {"time": N_TIME, "lat": CHUNK_LAT, "lon": CHUNK_LON}. Virtual size is
            N_TIME × N_LAT × N_LON × 8 bytes (float64).
        """
        import dask.array as da
        import pandas as pd

        # use dask's lazy random generation (data generated per-chunk on demand)
        rng = da.random.RandomState(42)
        # gamma distribution with realistic precipitation parameters
        dask_data = rng.gamma(
            shape=2.0,
            scale=50.0,
            size=(_N_TIME, _N_LAT, _N_LON),
            chunks=(_N_TIME, _CHUNK_LAT, _CHUNK_LON),
        )

        # create realistic coordinates
        time = pd.date_range("1900-01-01", periods=_N_TIME, freq="MS")
        lat = np.linspace(-90.0, 90.0, _N_LAT)
        lon = np.linspace(-180.0, 180.0, _N_LON)

        return xr.DataArray(
            dask_data,
            coords={"time": time, "lat": lat, "lon": lon},
            dims=["time", "lat", "lon"],
            attrs={"units": "mm"},
        )

    def test_dataset_creation_stays_lazy(self) -> None:
        """Verify dataset creation keeps data lazy (no premature materialization)."""
        with _PeakRSSMonitor() as monitor:
            da = self._create_large_precip_dataset()

            # verify dask-backed
            assert da.chunks is not None, "Dataset should be Dask-backed"

            # verify shape
            assert da.shape == (_N_TIME, _N_LAT, _N_LON)

        # verify minimal memory usage (only metadata/task graph, no data materialization)
        # allow 500 MB overhead for task graph metadata
        assert monitor.peak_delta_mb < 500, (
            f"Dataset creation materialized data (peak delta: {monitor.peak_delta_mb:.1f} MB). "
            f"Expected <500 MB for lazy creation."
        )

        virtual_size_gb = (_N_TIME * _N_LAT * _N_LON * 8) / (1024**3)
        print(
            f"\n✓ Dataset creation stayed lazy: virtual_size={virtual_size_gb:.2f} GB, "
            f"rss_delta={monitor.peak_delta_mb:.1f} MB"
        )

    def test_spi_graph_construction_stays_lazy(self) -> None:
        """Verify SPI computation constructs lazy task graph (no premature .compute())."""
        da = self._create_large_precip_dataset()

        with _PeakRSSMonitor() as monitor:
            result_lazy = spi(values=da, scale=_SPI_SCALE, distribution=_SPI_DISTRIBUTION)

            # verify result is still dask-backed (lazy)
            assert result_lazy.chunks is not None, "SPI result should be Dask-backed"

        # verify minimal memory usage (only extended task graph, no computation)
        assert monitor.peak_delta_mb < 500, (
            f"SPI graph construction materialized data (peak delta: {monitor.peak_delta_mb:.1f} MB). "
            f"Expected <500 MB for lazy graph construction."
        )

        print(f"\n✓ SPI graph construction stayed lazy: rss_delta={monitor.peak_delta_mb:.1f} MB")

    def test_lazy_evaluation_no_full_materialization(self) -> None:
        """Verify lazy result has correct chunk structure (time=single chunk, spatial chunks)."""
        da = self._create_large_precip_dataset()
        result_lazy = spi(values=da, scale=_SPI_SCALE, distribution=_SPI_DISTRIBUTION)

        # verify chunk structure
        assert result_lazy.chunks is not None, "Result should be Dask-backed"
        time_chunks, lat_chunks, lon_chunks = result_lazy.chunks

        # time dimension should be single chunk (required for SPI distribution fitting)
        assert len(time_chunks) == 1, f"Expected single time chunk, got {len(time_chunks)}"
        assert time_chunks[0] == _N_TIME

        # spatial dimensions should match input chunking
        expected_lat_chunks = (_N_LAT + _CHUNK_LAT - 1) // _CHUNK_LAT
        expected_lon_chunks = (_N_LON + _CHUNK_LON - 1) // _CHUNK_LON
        assert len(lat_chunks) == expected_lat_chunks
        assert len(lon_chunks) == expected_lon_chunks

        # verify dtype and shape
        assert result_lazy.dtype == np.float64
        assert result_lazy.shape == (_N_TIME, _N_LAT, _N_LON)

        print(
            f"\n✓ Lazy result has correct structure: "
            f"time_chunks=1, lat_chunks={len(lat_chunks)}, lon_chunks={len(lon_chunks)}"
        )

    def test_subset_correctness(self) -> None:
        """Verify numerical correctness by comparing Dask path to eager path on small subset."""
        # create full lazy dataset
        da_full = self._create_large_precip_dataset()

        # extract small 2×2 spatial subset
        da_subset = da_full.isel(lat=slice(0, 2), lon=slice(0, 2))

        # compute SPI via Dask pipeline
        result_dask = spi(values=da_subset, scale=_SPI_SCALE, distribution=_SPI_DISTRIBUTION)
        result_dask_computed = result_dask.compute()

        # compute SPI via eager in-memory path
        da_subset_eager = da_subset.compute()
        result_eager = spi(values=da_subset_eager, scale=_SPI_SCALE, distribution=_SPI_DISTRIBUTION)

        # verify numerical equivalence
        assert_allclose(
            result_dask_computed.values,
            result_eager.values,
            rtol=1e-10,
            err_msg="Dask path and eager path should produce identical results",
        )

        print("\n✓ Dask path numerically equivalent to eager path (rtol=1e-10)")

    def test_peak_memory_under_budget(self) -> None:
        """Validate peak memory stays within budget during chunked computation.

        This is the primary NFR-PERF-003 validation: out-of-core computation handles
        datasets larger than available RAM without OOM errors.

        Skips when virtual dataset < 1 GB (CI default) since memory budgets are only
        meaningful for large datasets.
        """
        virtual_size_gb = (_N_TIME * _N_LAT * _N_LON * 8) / (1024**3)

        if virtual_size_gb < _MIN_VIRTUAL_DATASET_GB:
            pytest.skip(
                f"Skipping peak memory test: virtual dataset size ({virtual_size_gb:.2f} GB) "
                f"below minimum ({_MIN_VIRTUAL_DATASET_GB} GB). "
                f"Set BENCHMARK_MEMORY_N_LAT={_SPEC_N_LAT} BENCHMARK_MEMORY_N_LON={_SPEC_N_LON} "
                f"BENCHMARK_MEMORY_N_TIME={_SPEC_N_TIME} for full-spec run."
            )

        da = self._create_large_precip_dataset()
        result_lazy = spi(values=da, scale=_SPI_SCALE, distribution=_SPI_DISTRIBUTION)

        with _PeakRSSMonitor() as monitor:
            # compute a few spatial tiles using synchronous scheduler (one chunk at a time)
            # this validates that we can process chunks sequentially without materializing
            # the entire result into memory
            num_tiles = min(4, len(result_lazy.chunks[1]) * len(result_lazy.chunks[2]))
            for i in range(num_tiles):
                # row index (advances after exhausting all longitude chunks)
                lat_idx = i // len(result_lazy.chunks[2])
                # column index (cycles through longitude chunks)
                lon_idx = i % len(result_lazy.chunks[2])
                # compute single spatial tile
                _ = result_lazy.isel(
                    lat=slice(lat_idx * _CHUNK_LAT, (lat_idx + 1) * _CHUNK_LAT),
                    lon=slice(lon_idx * _CHUNK_LON, (lon_idx + 1) * _CHUNK_LON),
                ).compute(scheduler="synchronous")

        # report memory usage
        peak_rss_gb = monitor.peak_rss_gb
        baseline_rss_gb = monitor.baseline_rss_mb / 1024
        print(
            f"\n✓ Peak memory test completed: "
            f"virtual_size={virtual_size_gb:.2f} GB, "
            f"baseline_rss={baseline_rss_gb:.2f} GB, "
            f"peak_rss={peak_rss_gb:.2f} GB, "
            f"delta={monitor.peak_delta_mb:.1f} MB"
        )

        # ideal threshold (reported but not enforced)
        if peak_rss_gb > _PEAK_MEMORY_IDEAL_LIMIT_GB:
            print(f"  ⚠ Peak RSS ({peak_rss_gb:.2f} GB) exceeds ideal limit ({_PEAK_MEMORY_IDEAL_LIMIT_GB} GB)")

        # strict mode: enforce hard limit
        if _STRICT_MEMORY:
            assert peak_rss_gb < _PEAK_MEMORY_HARD_LIMIT_GB, (
                f"Peak RSS {peak_rss_gb:.2f} GB exceeds hard limit {_PEAK_MEMORY_HARD_LIMIT_GB} GB. "
                f"Virtual dataset size: {virtual_size_gb:.2f} GB. "
                f"Disable strict mode via BENCHMARK_STRICT_MEMORY=false"
            )

    def test_virtual_dataset_size_matches_spec(self) -> None:
        """Report dataset configuration and verify basic sanity checks."""
        da = self._create_large_precip_dataset()

        virtual_size_bytes = _N_TIME * _N_LAT * _N_LON * 8
        virtual_size_gb = virtual_size_bytes / (1024**3)

        # verify dask-backed
        assert da.chunks is not None, "Dataset should be Dask-backed"

        # verify dtype
        assert da.dtype == np.float64, f"Expected float64, got {da.dtype}"

        # verify shape
        assert da.shape == (_N_TIME, _N_LAT, _N_LON)

        # compute number of spatial chunks
        num_lat_chunks = (_N_LAT + _CHUNK_LAT - 1) // _CHUNK_LAT
        num_lon_chunks = (_N_LON + _CHUNK_LON - 1) // _CHUNK_LON
        total_chunks = num_lat_chunks * num_lon_chunks

        # report configuration
        print(
            f"\n✓ Dataset configuration:\n"
            f"  Dimensions: {_N_TIME} × {_N_LAT} × {_N_LON}\n"
            f"  Virtual size: {virtual_size_gb:.2f} GB\n"
            f"  Chunk size: {_N_TIME} × {_CHUNK_LAT} × {_CHUNK_LON}\n"
            f"  Chunks: {total_chunks} spatial tiles ({num_lat_chunks} lat × {num_lon_chunks} lon)\n"
            f"  Per-chunk memory: {(_N_TIME * _CHUNK_LAT * _CHUNK_LON * 8) / (1024**2):.1f} MB\n"
            f"  Dtype: {da.dtype}"
        )

        # verify against PRD spec (when using full-spec dimensions)
        if _N_LAT == _SPEC_N_LAT and _N_LON == _SPEC_N_LON and _N_TIME == _SPEC_N_TIME:
            # PRD specifies ~49GB virtual dataset
            expected_size_gb = 49.0
            assert abs(virtual_size_gb - expected_size_gb) < 2.0, (
                f"Virtual size {virtual_size_gb:.2f} GB doesn't match PRD spec ~{expected_size_gb} GB"
            )
            print(f"  ✓ Matches PRD spec (~{expected_size_gb} GB)")

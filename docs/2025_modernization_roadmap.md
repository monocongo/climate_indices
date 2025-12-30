# 2025 Technical Modernization Roadmap: `climate_indices`

**Date:** December 25, 2025
**Author:** Principal Scientific Software Engineer
**Target Version:** 3.0.0

## 1. Executive Summary

The `climate_indices` library currently provides scientifically rigorous reference implementations for standard drought monitoring indices. However, the codebase exhibits "first-generation" scientific Python patterns: explicit loops over time steps, manual memory management via `multiprocessing` for parallelization, and a rigid NumPy-only API that strips metadata.

To serve the next generation of climate services (which process petabyte-scale data on cloud infrastructure), the library must evolve from a "script collection" into a cloud-native, high-performance toolkit. This roadmap proposes shifting the heavy lifting to the modern PyData stack: **Xarray** for data models, **Dask** for scaling, and **Numba** for compilation.

## 2. Top 5 Recommendations

### 1. Performance: Accelerate Core Kernels with Numba
**Problem:** The Palmer (PDSI) implementation (`palmer.py`) and Pearson Type III fitting (`compute.py`) rely on explicit Python loops over years/months. This prevents vectorization and is prohibitively slow for high-resolution global grids.
**Solution:** Apply `numba.jit(nopython=True)` to these core numerical functions. Numba can compile these loops to machine code, offering speeds comparable to C/Fortran without the build complexity.
**Impact:** Estimated 100x-1000x speedup for PDSI and Pearson fitting, enabling on-the-fly computation for interactive web tools.

### 2. Architecture: Replace Custom Multiprocessing with Dask
**Problem:** The CLI (`__spi__.py`) implements a complex, manual parallelization scheme using `multiprocessing.Array` and shared memory. This is brittle, hard to debug, and limited to a single machine.
**Solution:** Refactor the parallel processing to use **Dask**. By exposing the core functions as "lazy" operations (via `xarray.apply_ufunc` or `dask.array.map_blocks`), the library can automatically scale across cores on a laptop or nodes on an HPC cluster/Kubernetes.
**Impact:** Removes ~70% of the boilerplate code in the CLI, improves stability, and enables out-of-core processing for datasets larger than RAM.

### 3. Interoperability: Xarray Accessor API
**Problem:** Currently, users must extract NumPy arrays from their NetCDF/Zarr data, passing them to functions like `indices.spi()`, which returns a raw array without coordinates. This risks metadata loss (time/lat/lon alignment).
**Solution:** Implement an Xarray accessor to allow a fluent API:
```python
import xarray as xr
import climate_indices.accessors  # registers 'indices'

ds = xr.open_dataset("precip.nc")
# Returns a DataArray with all coordinates preserved
spi = ds.prcp.indices.spi(scale=3, distribution="gamma")
```
**Impact:** Seamless integration into modern scientific workflows, preserving spatiotemporal metadata and reducing user error.

### 4. Quality of Life: Modern Typing and Pre-commit Hooks
**Problem:** Type hints are partial (`values: np.ndarray`), and there is no automated enforcement of style or quality standards, relying on manual vigilance.
**Solution:**
*   Adopt **Strict Typing**: Use `numpy.typing.NDArray` or `jaxtyping` to specify array shapes and dtypes where possible.
*   **Pre-commit**: Add a `.pre-commit-config.yaml` to enforce `ruff` (linting/formatting), `mypy` (typing), and `check-manifest` before every commit.
**Impact:** Drastically reduces "trivial" bugs and code review friction, making the codebase more welcoming to contributors.

### 5. Domain Enhancement: Implement EDDI (Evaporative Demand Drought Index)
**Problem:** The library lacks the **Evaporative Demand Drought Index (EDDI)**, which has become a standard "flash drought" monitoring tool alongside SPI and SPEI.
**Solution:** Implement EDDI using the existing infrastructure. EDDI is mathematically similar to SPI but operates on Reference Evapotranspiration (ETo) instead of precipitation, fits a Log-Logistic or similar distribution, and inverts the quantiles (high ETo = drought).
**Impact:** Closes a gap in the drought monitoring suite, making the library a "one-stop-shop" for modern drought analysis.

## 3. Justification for Climate Scientists

*   **Why Speed (Numba/Dask) Matters:** Climate data is growing exponentially (CMIP6, ERA5). Waiting 24 hours for a global drought map is no longer acceptable. Modernization means results in minutes, allowing for rapid sensitivity analysis and real-time monitoring.
*   **Why Metadata (Xarray) Matters:** "Array soup" (managing raw arrays of `[360, 720, 1000]`) is the leading cause of scientific error (e.g., misaligning a drought mask with a crop layer). Native Xarray support ensures your physical variables stay locked to their latitude, longitude, and time coordinates.
*   **Why Standards Matter:** A robust, typed, and tested library is "publishable infrastructure." It ensures that the indices calculated for a paper today can be exactly reproduced 5 years from now, regardless of the underlying hardware.

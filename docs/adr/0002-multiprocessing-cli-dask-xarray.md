# Multiprocessing for the CLI, Dask for the xarray API

The CLI (`__main__.py`, `__spi__.py`) parallelizes gridded NetCDF processing with Python's `multiprocessing.Pool` and shared-memory arrays, splitting work across lat/lon; the xarray API instead delegates to Dask (via `xr.apply_ufunc(..., dask="parallelized")`) when given a Dask-backed array, without importing `dask` directly. We use different mechanisms per layer rather than standardizing on one: multiprocessing keeps the CLI's memory usage predictable and avoids requiring a Dask dependency for CLI-only users, while Dask gives the xarray workflow lazy evaluation and integrates naturally with the rest of the xarray/Dask ecosystem that those users are already in.

## Consequences

Parallelization logic is duplicated across the two layers and must be reasoned about separately — a performance fix to one does not carry over to the other.

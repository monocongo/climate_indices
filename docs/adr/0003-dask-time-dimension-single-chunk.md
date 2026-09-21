# Dask arrays must chunk the time dimension as a single chunk

## Status

Accepted.

`xarray_adapter.py` requires that Dask-backed input arrays have the time dimension as one unbroken chunk (spatial dimensions may be chunked freely), and raises `CoordinateValidationError` if that's violated. This is enforced because distribution fitting and calibration need the full time series for a given cell in one place — splitting time across chunks would silently produce wrong statistics (or require a much more complex cross-chunk fitting implementation we haven't built). A user who doesn't know this constraint exists would reasonably expect Dask to chunk time like any other dimension, so this is worth stating explicitly rather than only discovering it via the validation error.

## Consequences

Callers building Dask arrays for use with this library must chunk only along spatial dimensions (e.g. `{"lat": 50, "lon": 50, "time": -1}`), not time.

[notebooks/zarr_dask_spi_spei.ipynb](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb) is the worked example: it keeps one time chunk from the prepared store through the calculation. [docs/troubleshooting.md](https://github.com/monocongo/climate_indices/blob/main/docs/troubleshooting.md) lists the failure modes, including the `Prepared store must keep time as a single chunk.` error.

# Dask arrays must chunk the time dimension as a single chunk

`xarray_adapter.py` requires that Dask-backed input arrays have the time dimension as one unbroken chunk (spatial dimensions may be chunked freely), and raises `CoordinateValidationError` if that's violated. This is enforced because distribution fitting and calibration need the full time series for a given cell in one place — splitting time across chunks would silently produce wrong statistics (or require a much more complex cross-chunk fitting implementation we haven't built). A user who doesn't know this constraint exists would reasonably expect Dask to chunk time like any other dimension, so this is worth stating explicitly rather than only discovering it via the validation error.

## Consequences

Callers building Dask arrays for use with this library must chunk only along spatial dimensions (e.g. `{"lat": 50, "lon": 50, "time": -1}`), not time.

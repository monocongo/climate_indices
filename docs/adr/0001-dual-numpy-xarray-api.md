# Dual numpy and xarray public APIs

`climate_indices` ships two parallel public APIs: a legacy numpy-array API (`indices.py`, stable, do-not-modify per architecture rules) and a modern xarray API (`typed_public_api.py` / `xarray_adapter.py`) with CF metadata and Dask support. We decided to keep both rather than deprecating the numpy API, because existing users depend on plain numpy arrays and migrating them is not our call to force. The xarray API is implemented as a wrapper that calls the numpy functions internally (via `xr.apply_ufunc`), so there is one source of truth for the actual computation — the duplication risk is confined to the interface layer, not the math.

## Consequences

New index computations must be added to `compute.py` and wired into both layers (numpy signature in `indices.py`, `@xarray_adapter`-wrapped entry point in `xarray_adapter.py`) rather than picking just one API to extend.

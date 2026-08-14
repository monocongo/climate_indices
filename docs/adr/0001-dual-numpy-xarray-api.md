# Dual numpy and xarray public APIs

`climate_indices` ships two parallel public APIs: a legacy numpy-array API (`indices.py`, stable, do-not-modify per architecture rules) and a modern xarray API (`typed_public_api.py` / `xarray_adapter.py`) with CF metadata and Dask support. We decided to keep both rather than deprecating the numpy API, because existing users depend on plain numpy arrays and migrating them is not our call to force. The xarray API is implemented as a wrapper that calls the numpy functions internally (via `xr.apply_ufunc`), so there is one source of truth for the actual computation — the duplication risk is confined to the interface layer, not the math.

## Consequences

The established numpy API in `indices.py` remains stable and receives no new functions. New non-Palmer index computations must be added to `compute.py` and exposed through the modern API in `typed_public_api.py` and `xarray_adapter.py`. Palmer-family computations are the established exception: they live in `palmer.py` and currently expose only their numpy API. Adding xarray support for Palmer indices requires a separate, explicit architecture decision rather than unsupported wiring through `indices.py` or `xarray_adapter.py`.

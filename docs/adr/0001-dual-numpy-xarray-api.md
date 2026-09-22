# Dual numpy and xarray public APIs

## Status

Amended: the mechanism in the opening paragraph is not the only route. PCI
bypasses the shared adapter and wraps `indices.pci()` by hand, because its output
is a single scalar (`a024f408`), and the adapter's 1-D in-memory path calls the
NumPy function without `xr.apply_ufunc`. The decision — the NumPy functions are
the one source of truth for the computation — is unchanged.

Amended again: `indices.standardized_index()` (#1113) is a NumPy-only generic
wrapper over the SPI scale/fit/transform pipeline, added beside the index it
generalizes. It has no CF metadata of its own, so its xarray entry point is
deferred to the flood family's design and metadata decisions (#1099, #1103)
rather than wired through the adapter now, and it is not re-exported from the
package root, where every name carries the dual NumPy/xarray contract.

`climate_indices` ships two parallel public APIs: a legacy numpy-array API (`indices.py`, stable, do-not-modify per architecture rules) and a modern xarray API (`typed_public_api.py` / `xarray_adapter.py`) with CF metadata and Dask support. We decided to keep both rather than deprecating the numpy API, because existing users depend on plain numpy arrays and migrating them is not our call to force. The xarray API is implemented as a wrapper that calls the numpy functions internally (through `xr.apply_ufunc` on the adapter paths, directly on the adapter's 1-D in-memory path and in PCI's hand-written wrapper), so there is one source of truth for the actual computation — the duplication risk is confined to the interface layer, not the math.

## Consequences

The established numpy API in `indices.py` remains stable and receives no new functions. New non-Palmer index computations must be added to `compute.py` and exposed through the modern API in `typed_public_api.py` and `xarray_adapter.py`. Palmer-family computations are the established exception: they live in `palmer.py`; ADR-0011 granted the standard PDSI family an xarray entry point (`climate_indices.pdsi()`, #1016), while scPDSI remains NumPy-only. Adding xarray support for other Palmer indices requires a separate, explicit architecture decision rather than unsupported wiring through `indices.py` or `xarray_adapter.py`. The fire family is the documented exception in [ADR-0005](./0005-fire-module-api.md).

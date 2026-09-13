# Architecture Deepening Review — 2026-08-27

## Scope

This review applied the deletion test to the current `climate_indices` source tree, using the domain language in [`src/climate_indices/CONTEXT.md`](../src/climate_indices/CONTEXT.md) and the decisions in [`docs/adr/`](./adr/).

The aim is to increase **depth**: more leverage through smaller interfaces and more locality inside implementations. This document records candidates, not proposed interfaces.

## Recommendation summary

| Strength | Candidate | Primary modules |
|---|---|---|
| **Strong** | Make calendar semantics explicit at the xarray seam | `xarray_adapter.py`, `utils.py`, `indices.py`, `eto.py` |
| **Strong** | Collapse duplicated CLI preparation and transport | `__main__.py`, `__spi__.py` |
| **Worth exploring** | Concentrate Distribution Fitting policy | `compute.py`, `indices.py`, `lmoments.py`, `__spi__.py` |
| **Strong** | Use one Output Provenance finalizer | `xarray_adapter.py`, `typed_public_api.py`, `cf_metadata_registry.py` |
| **Strong** | Own multi-input alignment at one seam | `xarray_adapter.py` |
| **Worth exploring** | Deepen the Palmer computation state | `palmer.py` |

## 1. Make calendar semantics explicit at the xarray seam

**Files**

- `src/climate_indices/xarray_adapter.py`
- `src/climate_indices/utils.py`
- `src/climate_indices/indices.py`
- `src/climate_indices/eto.py`
- `src/climate_indices/__main__.py`
- `src/climate_indices/__spi__.py`
- `tests/test_xarray_adapter.py`
- `tests/test_xarray_equivalence.py`

**Problem**

The xarray interface infers daily Periodicity from ordinary Gregorian coordinates, then passes raw values to NumPy implementations that group every year into 366 positional values. Non-leap years therefore shift calendar positions. Monthly coordinates can likewise begin after January even though the NumPy implementation assumes a January-first series. Original coordinates are attached to the result, hiding the positional drift.

The CLI already owns explicit Gregorian-to-366-day and 366-day-to-Gregorian conversion, but that calendar knowledge does not protect the xarray seam.

**Deepening direction**

Concentrate calendar validation, conversion, and coordinate restoration inside xarray adaptation. Preserve the stable NumPy implementation required by [ADR-0001](./adr/0001-dual-numpy-xarray-api.md), and do not combine the distinct execution implementations protected by [ADR-0002](./adr/0002-multiprocessing-cli-dask-xarray.md).

**Benefits**

- Locality: one calendar policy.
- Leverage: SPI, SPEI, EDDI, PNP, and PET use it.
- Tests cross the same seam as callers.
- Coordinate labels remain truthful.

**Tests to add before implementation**

- Daily Gregorian xarray results match explicit 366-day NumPy computation after restoration.
- Non-leap February does not shift March–December values.
- Leap and non-leap multi-year inputs retain their original time coordinates.
- Monthly and daily inputs with unsupported start positions fail explicitly rather than silently drifting.
- Eager and Dask-backed inputs behave equivalently while preserving [ADR-0003](./adr/0003-dask-time-dimension-single-chunk.md).

**Recommendation strength: Strong.** This is the top recommendation because one deep module prevents silent scientific misalignment across several public computations without changing the stable NumPy interface.

## 2. Collapse duplicated CLI preparation and transport

**Files**

- `src/climate_indices/__main__.py`
- `src/climate_indices/__spi__.py`
- `tests/test_input_validation.py`
- `tests/test_main_palmers.py`

**Problem**

Both commands duplicate dimension policy, daily conversion, shared-memory allocation, worker initialization, and multiprocessing partitioning. Both `_prepare_file` implementations claim to reorder dimensions but only compare dimension sets and return the original path. Later code accepts a different set of dimension orders, so the apparent preparation seam is shallow.

**Deepening direction**

A private deep module should own accepted layouts, normalization, daily conversion, partitioning, and shared-memory transport. Command-specific computation and SPI parameter persistence remain separate.

**Benefits**

- Locality: dimension policy lives in one module.
- Leverage: both commands share transport fixes.
- Duplicate mechanics can be deleted.
- Preparation tests exercise truthful behavior.

**Decision fit**

This preserves ADR-0002: multiprocessing remains the CLI execution implementation and Dask remains the xarray execution implementation.

**Recommendation strength: Strong.**

## 3. Concentrate Distribution Fitting policy

**Files**

- `src/climate_indices/compute.py`
- `src/climate_indices/indices.py`
- `src/climate_indices/lmoments.py`
- `src/climate_indices/__spi__.py`
- `tests/test_compute.py`
- `tests/test_indices.py`
- `tests/test_zero_precipitation_fix.py`

**Problem**

Calibration Period selection, Probability of Zero, Gamma and Pearson parameter representation, fallback, diagnostics, compatibility translation, and CLI persistence knowledge are distributed across callers.

Observed policy differences include:

- Gamma parameters use Calibration Period data while Gamma Probability of Zero currently uses the whole record.
- Pearson counts non-zero calibration values but passes the zero-inclusive sample to L-Moments fitting.
- SPI and SPEI do not apply fitted-parameter compatibility translation consistently.
- The CLI knows every distribution-specific parameter field and storage arrangement.
- End-year calculations and minimum Calibration Period behavior are not expressed consistently in every path.

**Deepening direction**

Concentrate scientific policy in one internal Distribution Fitting module while preserving legacy NumPy call forms as compatibility adapters.

**Benefits**

- Locality: scientific policy and diagnostics concentrate.
- Leverage: Gamma and Pearson share calibration behavior.
- One interface becomes the fitting test surface.
- Compatibility knowledge stops leaking into callers.

**Risk**

Resolving current differences may intentionally change scientific outputs. Lock existing fixtures first, isolate policy decisions, and require explicit approval for output changes.

**Recommendation strength: Worth exploring.**

## 4. Use one Output Provenance finalizer

**Files**

- `src/climate_indices/xarray_adapter.py`
- `src/climate_indices/typed_public_api.py`
- `src/climate_indices/cf_metadata_registry.py`
- `tests/test_metadata_validation.py`
- `tests/test_pci_xarray.py`
- `tests/test_xarray_adapter.py`

**Problem**

Output Provenance and CF metadata are one domain behavior with three implementations:

- Decorated index paths preserve source attributes, apply CF and calculation metadata, add the version, and append history.
- PET paths reproduce much of that implementation separately.
- PCI drops source attributes and prior history, then creates a differently formatted history value.

Computation shape currently determines provenance semantics.

**Deepening direction**

Route every xarray result shape through one deep finalization module without forcing scalar, time-series, and spatial computation paths to become identical.

**Benefits**

- Locality: Output Provenance policy concentrates.
- Leverage: every xarray result uses one implementation.
- Tests cross one interface.
- Shape differences remain implementation details.

**Recommendation strength: Strong.**

## 5. Own multi-input alignment at one seam

**Files**

- `src/climate_indices/xarray_adapter.py`
- `tests/test_xarray_adapter.py`
- `tests/test_xarray_equivalence.py`

**Problem**

Generic multi-input alignment uses `xr.align(..., join="inner")` across every shared coordinate but reports only dropped primary time values. Shared latitude or longitude values can be intersected without equivalent diagnostics. Validation and Dask chunk checks focus on the primary input. Hargreaves PET implements a separate alignment path with different warning context.

Two current adapters—SPEI and Hargreaves PET—make this a real seam rather than a hypothetical one.

**Deepening direction**

One deep alignment module should own coordinate-loss policy, validation of every input, Dask constraints, and structured diagnostics.

**Benefits**

- Locality: alignment policy concentrates.
- Leverage: two current adapters and future multi-input indices.
- Secondary-input invariants become explicit.
- Tests cover every aligned coordinate.

**Recommendation strength: Strong.**

## 6. Deepen the Palmer computation state

**Files**

- `src/climate_indices/palmer.py`
- `tests/test_palmer.py`
- `tests/test_palmer_duration_factors.py`
- `tests/test_property_based.py`

**Problem**

A mutable `dict[str, Any]` containing more than thirty keys is the hidden interface among Palmer phases. Key names, shapes, initialization order, mutation order, and phase order are spread across more than twenty helpers. Tests reconstruct the water-balance, CAFEC, K-Factor, and recurrence sequence to inject duration factors.

**Deepening direction**

Make state ownership and phase ordering local to the Palmer module while preserving arithmetic order and the public `pdsi` interface.

**Benefits**

- Locality: state schema and ordering concentrate.
- The public interface stays stable.
- Phase tests stop reconstructing implementation order.
- Numerical invariants become visible.

**Risk**

This is numerically sensitive code. Preserve arithmetic order, NOAA fixtures, and standard Palmer behavior. Do not introduce scPDSI or a Palmer xarray interface during this cleanup.

**Recommendation strength: Worth exploring.**

## Additional observations

These did not outrank the six candidates but should remain visible for later review:

- `eto.eto_thornthwaite()` mutates caller-owned temperature arrays; the xarray PET adapter compensates with copies. A non-mutating ETo implementation would remove leaked ownership knowledge and per-cell defensive copies.
- `compute.scale_values()` is a shallow Timescale-preparation seam used by the specialized CLI while SPI, SPEI, EDDI, and PNP repeat related preparation knowledge.
- Lifecycle logging can occur once per spatial cell when xarray vectorization calls legacy NumPy functions; event cardinality and host-application logger configuration need explicit policy.
- Several metadata entries, exception types, and pattern-compliance claims have no production callers or conflict with ADR-0001’s Palmer decision. Public compatibility must be checked before deletion.
- Four-channel Palmer output knowledge is repeated across CLI allocation, worker writes, conversion, metadata, and persistence.
- `self_calibration.py` is well tested but has no production caller while scPDSI remains explicitly unimplemented; retain it only with clear staged-work ownership.
- `pm_eto.py` currently exposes low-level equation helpers without an in-repository computation caller. Confirm intended product scope before deepening or removing it.

## Suggested order

1. Make calendar semantics explicit at the xarray seam.
2. Use one Output Provenance finalizer.
3. Own multi-input alignment at one seam.
4. Collapse duplicated CLI preparation and transport.
5. Explore Distribution Fitting policy under fixture locks.
6. Explore Palmer state under numerical regression locks.

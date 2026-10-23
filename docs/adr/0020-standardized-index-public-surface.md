# The generic standardization seam ships recipes, one neutral CF entry, and a wired xarray path

## Status

Amended by #1061: the DataArray standardization and diagnostic APIs are Stable
from 3.1.0 under {doc}`0012-xarray-api-stays-beta-through-3.0.0`.
The public-surface decision and the CF metadata entry landed with this
record; the xarray wiring, the xarray `fit_diagnostics()` aggregation report, and
the package-root re-export landed with
[#1174](https://github.com/monocongo/climate_indices/issues/1174), which this
record unblocked.

`indices.standardized_index()` ([#1113](https://github.com/monocongo/climate_indices/issues/1113))
generalizes the SPI scale/fit/transform pipeline to whatever series it is handed —
runoff, streamflow, a capacity factor, residual load — rather than one physical
quantity. {doc}`0001-dual-numpy-xarray-api` recorded its xarray entry point as
deferred, because the function "has no CF metadata of its own", and pointed the
question at the flood family's design and metadata decisions (#1099, #1103).
[#1099](https://github.com/monocongo/climate_indices/issues/1099) is decided and
does not cover this path, and #1103 landed the flood registry entries only, so the
deferral outlived the decisions it named. {doc}`0019-energy-module-api-and-aggregation`
and ENERGY-18 ([#1174](https://github.com/monocongo/climate_indices/issues/1174))
then assumed an answer this record supplies.

Only the surface was open. The adapter machinery is shared and already exists —
`xr.apply_ufunc` block packing, `spatial_time_major`, calibration-period inference
from the time coordinates, Dask parallelism, and the documented `fitting_params`
per-block caveat.

## Decision

1. **No named wrappers.** SRI and SSI stay documented recipes over
   `indices.standardized_index()`, as the [SRI and SSI
   recipes](../standardized-hydrologic-indices.md) and the [flood
   applications](../flood_applications.md) present them. A `sri()` or `ssi()` function would bind
   a distribution and a timescale the caller already passes, compute nothing the
   seam does not, and name a discipline-specific meaning the input-agnostic seam
   does not enforce. The recipes pin each index's published definition.

2. **One neutral CF metadata entry, no caller-supplied label.**
   `CF_METADATA["standardized_index"]` carries `long_name` "Standardized index",
   `units` "1", the standardization method's citation, and no `standard_name`,
   per the CF Standard Name Omission policy in the core vocabulary. It is the one
   entry that is truthful for an output whose input is not fixed. The adapter
   stamps it, and a caller who needs "Standardized Runoff Index" on the result
   renames the returned `DataArray` — two attribute writes it can already do. A
   label keyword would thread a caller string through the registry and the adapter
   for that same result.

3. **The xarray entry is wired, Stable from 3.1.0.** `standardized_index` joins the adapter's
   shared path, Stable under {doc}`0012-xarray-api-stays-beta-through-3.0.0`, with the
   `fitting_params`-per-block caveat and calibration inference every other entry
   carries. It exposes `aggregation` on the xarray arm, and the xarray arm of
   `fit_diagnostics()` reports the same, completing the deferral
   {doc}`0019-energy-module-api-and-aggregation` recorded. ENERGY-18 (#1174)
   implemented this surface. The entry runs per cell rather than as a spatial
   block, because a block's accumulated sums can differ in the last bits from a
   single series and the empirical distribution's ranks move when ties flip; the
   NumPy result is the contract.

4. **Package-root re-export follows the xarray entry.** `standardized_index` joins
   the `climate_indices` root re-exports in the same change that wires its xarray
   entry, because the root re-exports are the `typed_public_api` functions carrying
   both the NumPy and the xarray contract ({doc}`0001-dual-numpy-xarray-api`). The
   NumPy core stays at `climate_indices.indices.standardized_index`.

## Consequences

{doc}`0001-dual-numpy-xarray-api` is amended: its Status no longer defers this entry
to a closed or non-covering decision, and it points here. The deferral closed with
[#1174](https://github.com/monocongo/climate_indices/issues/1174).

The two published pages stop deferring. The [SRI and SSI
recipes](../standardized-hydrologic-indices.md) and the [flood
applications](../flood_applications.md) state that SRI and SSI are recipes over a generic seam with
no named wrapper, and that the xarray path is the stable `standardized_index` entry.

`CF_METADATA["standardized_index"]` landed ahead of its adapter, which the
registry's own contract permits (an entry may precede its adapter; no adapter
precedes its entry).

[#1174](https://github.com/monocongo/climate_indices/issues/1174) carried two
follow-ons, both landed: the xarray `standardized_index` entry exposes
`aggregation`, and the xarray arm of `fit_diagnostics()` reports the aggregation
behind a fit instead of rejecting a non-default value.

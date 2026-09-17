# Gridded index input is read as a time-major block, and ambiguous shapes must be declared

The fitting-based NumPy kernels (`indices.spi`, `indices.spei`) have always read a flat or
`(years, periods)` array, and `compute.prepare_scaled` folded a 1-D series onto that layout before
fitting. Removing the per-grid-cell `xr.apply_ufunc(..., vectorize=True)` loop therefore required a
second input layout: a time-major block with shape `(time, *cells)`, where every trailing axis is an
independent cell to be scaled, fitted, and transformed in one pass.

That layout is largely distinguishable by rank. One- and two-dimensional input stays on the legacy
reading — a 2-D array is a `(years, periods)` series flattened into one, not a `(time, cells)` grid,
which is what every existing caller and fixture depends on. Three or more dimensions are read as a
time-major block whose trailing axes are preserved.

One shape is not distinguishable. A `(time, *cells)` block whose first cell axis happens to be a
calendar period length (12 or 366) is structurally identical to a `(years, periods, *cells)` array, the
natural extension of the documented 2-D convention, and nothing inside an `ndarray` says which axis is
time. Reading one
as the other returns plausible numbers rather than raising: in a review experiment a `(40, 12, 2)`
input read as time-major differed from the per-cell result by up to 4.4 index units while completing
without error.

`indices.spi`, `indices.spei`, and `compute.prepare_scaled` therefore reject that shape unless the
caller declares it with `spatial_time_major=True`; `xarray_adapter` sets that keyword for every block
it packs when an index is registered with `spatial_kernel=True`, and the xarray path is where
dimension labels make the reading knowable. Rejecting *all* gridded NumPy input instead was
considered and dropped: the adapter would have had to keep a second entry point for it, and a grid
whose first cell axis is 12 or 366 — twelve latitudes, say — is legitimate input that worked before
this change and still has to work.

## Consequences

Every index with three or more dimensions is time-major input, whether it arrives from the adapter or
from a direct caller. A direct caller passing a `(time, 12, *cells)` or `(time, 366, *cells)` array gets a `ValueError`
naming the ambiguity rather than a silently re-read result; reordering the cell axes or declaring
`spatial_time_major=True` both resolve it. `indices.spi`'s 2-D contract is unchanged, so the
existing `(years, periods)` callers and their fixtures keep working.

The spatial path is opt-in per index: `spatial_kernel=True` is declared at the adapter call site
(`typed_public_api.py` for SPI and SPEI), and any index left on the per-cell path keeps
`vectorize=True` with one kernel call per cell. Registering an index whose kernel does not accept the
`spatial_time_major` keyword fails loudly at the call rather than misreading its input.

Two layouts now meet in `compute.py`, distinguished by position in the pipeline rather than by any
runtime marker: time-major `(time, *cells)` on the way in (`prepare_scaled`, `sum_to_scale`), and
`(years, periods, *cells)` after folding (`_validate_array`, `gamma_parameters`,
`transform_fitted_gamma`, `_check_goodness_of_fit_gamma`). `_reshape_time_major` is the only
translation between them.

Gridded execution changes the memory profile as well as the call count: one block is held whole
inside the fit, with a few `O(years x periods x cells)` temporaries, so the chunk size is the memory
lever and the documented guidance is to chunk spatially rather than hand the kernel a dense
continental grid. Dask still requires the time dimension in a single chunk, for the reason recorded
in [ADR-0003](./0003-dask-time-dimension-single-chunk.md) — the fit needs the whole calibration
window.

The Pearson Type III branch keeps [the per-cell path](../xarray_compatibility.md); its L-moment fit
is per series, so `Distribution.pearson` still re-enters the single-series kernel once per cell
(issue #940), and Palmer has no adapter layer at all (#937). EDDI and percentage of normal carry a
cell axis as well (#942): EDDI counts each calendar period's climatology values below every cell's
value, and percentage of normal averages each cell's calendar-period normals, so neither loops over
the grid. Unlike the fitting-based kernels they reject an undeclared 3-D input, since their
dimension errors are pinned to `DataShapeError` rather than `ValueError`.

The PET entry points do not use the adapter decorator, because latitude arrives as a broadcast
input rather than a secondary time series. They forward `vectorize=False` themselves and hand
`indices.pet` and `eto.eto_hargreaves` the same `(time, *cells)` block, with the latitude as a
per-cell array for the day-length and radiation terms (#941); a 2-D input, or a latitude carrying a
dimension the temperature does not, keeps the per-cell path.

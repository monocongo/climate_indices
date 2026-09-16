# Gridded index input reaches the NumPy core only as an adapter-declared time-major block

The fitting-based NumPy kernels (`indices.spi`, `indices.spei`) have always read a flat or
`(years, periods)` array, and `compute.prepare_scaled` folded a 1-D series onto that layout before
fitting. Removing the per-grid-cell `xr.apply_ufunc(..., vectorize=True)` loop therefore required a
second input layout: a time-major block with shape `(time, *cells)`, where every trailing axis is an
independent cell to be scaled, fitted, and transformed in one pass.

That layout is ambiguous with the documented one. A `(time, *cells)` array whose first cell axis
happens to be 12 or 366 is structurally identical to a `(years, periods, *cells)` array — the natural
extension of the documented 2-D `(years, periods)` convention — and nothing inside an `ndarray` says
which axis is time. Reading one as the other returns plausible numbers rather than raising: in a
review experiment, a `(40, 12, 2)` input read as time-major differed from the per-cell result by up
to 4.4 index units while completing without error.

The core therefore reads a gridded array only when the caller declares it, through the
`spatial_time_major` keyword that `xarray_adapter` sets when it packs a block for an index registered
with `spatial_kernel=True`. `indices.spi`/`indices.spei` raise the same `ValueError` for an
undeclared gridded array that they raised before this work, so the stable NumPy API contract from
[ADR-0001](./0001-dual-numpy-xarray-api.md) is unchanged: 1-D and 2-D input only. The xarray path
knows its dimension labels, which is where the ambiguity is resolvable, and it is also the layer that
wanted the per-block execution in the first place.

## Consequences

The spatial layout is opt-in per index rather than automatic. `spatial_kernel=True` is declared at the
adapter call site (`typed_public_api.py` for SPI and SPEI); any index left on the per-cell path keeps
`vectorize=True` and one kernel call per cell. Registering an index whose kernel does not accept the
`spatial_time_major` keyword fails loudly at the call, rather than misreading its input.

Two layouts now meet in `compute.py`, and they are distinguished by position in the pipeline rather
than by any runtime marker: time-major `(time, *cells)` on the way in (`prepare_scaled`,
`sum_to_scale`), and `(years, periods, *cells)` after folding (`_validate_array`, `gamma_parameters`,
`transform_fitted_gamma`, `_check_goodness_of_fit_gamma`). `_reshape_time_major` is the only
translation between them. A new caller that passes a gridded array to `prepare_scaled` must declare it
the same way, and `scale_values`/`prepare_scaled` raise `ValueError` when it does not.

Gridded execution changes the memory profile as well as the call count: one block is held whole inside
the fit, with a few `O(years x periods x cells)` temporaries, so the chunk size is the memory lever
and the documented guidance is to chunk spatially rather than hand the kernel a dense continental
grid. Dask still requires the time dimension in a single chunk, for the reason recorded in
[ADR-0003](./0003-dask-time-dimension-single-chunk.md) — the fit needs the whole calibration window.

The Pearson Type III branch keeps [the per-cell path](../xarray_compatibility.md); its L-moment fit is
per series, so `Distribution.pearson` still re-enters the single-series kernel once per cell
(issue #940). EDDI and percentage-of-normal have no cell axis in their kernels yet (#942), the PET
entry points do not use the adapter decorator (#941), and Palmer has no adapter layer at all (#937).

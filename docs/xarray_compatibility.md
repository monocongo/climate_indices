# xarray Compatibility Matrix

The xarray API is beta in v2.5. Numerical results are expected to match the
stable NumPy API, while parameter inference, metadata, and coordinate behavior
may change in a future minor release.

| Feature | Supported | Coverage |
| --- | --- | --- |
| `DataArray` inputs for SPI | Yes | `tests/test_xarray_equivalence.py` compares 1-D and gridded xarray outputs to NumPy. |
| `DataArray` inputs for SPEI | Yes | `tests/test_xarray_equivalence.py` compares xarray precipitation/PET inputs to NumPy. |
| `DataArray` inputs for EDDI | Yes | EDDI wrapper and metadata tests cover the typed public API. |
| `DataArray` inputs for PET Thornthwaite | Yes | xarray adapter tests cover scalar and spatial latitude handling. |
| `DataArray` inputs for PET Hargreaves | Yes | xarray adapter tests cover aligned daily temperature inputs. |
| `DataArray` inputs for PNP | Yes | PNP wrapper tests cover scale handling and metadata. |
| `DataArray` inputs for PCI | Yes | PCI uses a manual scalar-output wrapper. |
| `DataArray` inputs for KBDI | Yes | `fire.kbdi()` resolves the `kbdi`/`kbdi_imperial` CF entry per call from `units`; supports `return_state`/`initial_state`, `spin_up`, `nan_policy`/`max_gap_days`, and CF `units`-attribute unit inference for the weather inputs and an attributed mean annual climatology. An attached time coordinate must be consecutive daily observations per input; a dimension-only time axis is aligned positionally. Overlapping timesteps are inner-aligned, dropped timesteps emit `InputAlignmentWarning`, and alignment that would drop non-time coordinates is rejected. See `tests/test_fire_kbdi.py`. |
| `DataArray` inputs for HDW | Yes | `fire.hot_dry_windy()` reduces the caller-named `level_dim` (default `"level"`) to the layer maximum via `xr.apply_ufunc`; CF metadata comes from the `hdw` registry entry, and a CF `units` attribute on temperature is converted to Celsius. Not recursive: every other dimension, including `time` if present, is a plain passthrough with no cadence requirement. Dimensions shared by the inputs must carry identical, identically ordered coordinate labels, because the inputs are matched with xarray's exact join rather than aligned or reindexed. `level_dim` must be a single Dask chunk; other dimensions chunk freely. See `tests/test_fire.py`. |
| Palmer direct xarray API | No | Use the NumPy Palmer function with `.values`, then rewrap outputs. See `notebooks/palmer_indices_xarray.ipynb`. |
| Coordinate preservation | Yes | Adapter tests verify time and spatial coordinates are preserved. |
| CF-style metadata | Yes | `CF_METADATA` registry and adapter tests verify `long_name`, `units`, `references`, version, and history attributes. |
| Dask-backed arrays | Yes, constrained | The time dimension must be a single chunk. Adapter tests verify detection and validation. |
| Spatial (gridded) kernels | Yes, for SPI, SPEI, and PET | `spi`/`spei` receive a time-major `(time, *cells)` block and fit every cell in one pass, so a gridded gamma run costs one kernel call per Dask block instead of one call per cell ([ADR-0008](adr/0008-spatial-block-declaration.md)). The PET entry points (`pet_thornthwaite`, `pet_hargreaves`) hand `indices.pet` and `eto.eto_hargreaves` the same block layout with the per-cell latitude array, so a gridded PET run costs one call per block ([#941](https://github.com/monocongo/climate_indices/issues/941)). The NumPy API keeps 2-D input on the legacy `(years, periods)` reading, and rejects the one gridded shape that is ambiguous with it (a first cell axis of 12 or 366). Inputs with a single non-core dimension keep the per-cell path, as do EDDI and percentage-of-normal ([#942](https://github.com/monocongo/climate_indices/issues/942)) and the Pearson Type III fit ([#940](https://github.com/monocongo/climate_indices/issues/940)). See `tests/test_spatial_kernel.py`. |
| Calendar semantics | Yes, constrained | Standard/gregorian/proleptic_gregorian `datetime64` only; monthly input must begin in January and daily input on January 1. Daily values are converted to the 366-day calendar (February 29 synthesized from February 28 and March 1) and restored afterward. A partial final year is supported; `cftime` calendars are rejected. See [ADR-0004](adr/0004-xarray-calendar-semantics.md). |
| Automatic temporal inference | Yes | Monthly and daily time-coordinate inference is covered by adapter tests. |
| Multi-input alignment | Yes | SPEI aligns precipitation and PET with an inner join and emits a warning when timesteps are dropped. |

## Stateful fire indices

KBDI's adapter (`fire.kbdi()`) is shipped, following the recursive contract in
[ADR-0006](adr/0006-fire-recursive-state-and-execution.md). CFFWIS's
moisture-code adapters are not yet shipped; this section also describes the
contract they will follow. The weather-only Fosberg Fire Weather Index is
available today too, via the NumPy layer with no xarray adapter needed (it is
elementwise with no dimension to reduce). Hot-Dry-Windy's adapter (see the
main table above) is not recursive either, but it does reduce a named
vertical dimension, so it does not follow the stateful contract below; it
needs only that dimension in a single Dask chunk.

These adapters are recursive: each daily value needs its predecessor.
Dask-backed inputs must therefore keep the complete `time` dimension in one
chunk for every time-varying weather variable. Spatial chunks remain
supported and parallelize: KBDI's NumPy core already vectorizes over an
arbitrary spatial shape, so its adapter dispatches one call per Dask spatial
chunk with the full `time` axis, rather than looping per grid cell.
Multi-chunk time input raises `CoordinateValidationError` with
`data = data.chunk({'time': -1})`; adapters never rechunk implicitly, because
doing so can materialize a large daily history.

Callers use the returned state to append later observations without
recomputing the archive. The state is a NumPy-layer value object rather than
an xarray `Dataset`, so its arrays carry the computational spatial shape but
no coordinates, even when `values` on the same result is a `DataArray`.

Missing observations follow
[ADR-0007](adr/0007-fire-missing-data-policy.md): nothing is interpolated
implicitly. The default `nan_policy="propagate"` poisons the state at the
first interior gap and yields NaN outputs from there on;
`nan_policy="bridge"` with `max_gap_days=N` skips runs of at most `N` missing
days. Adapters forward both keyword arguments to the NumPy core so the policy
applies per cell along the time axis.

## Operational Guidance

- Use NumPy APIs for stable production integrations that cannot absorb beta
  interface changes.
- Use xarray APIs for labeled, gridded workflows where coordinate preservation
  and metadata are more valuable than strict interface stability.
- Keep Dask chunks spatial when possible and leave `time` as one chunk before
  calling index functions. Spatial chunks set the parallelism granularity: SPI and
  SPEI fit a whole `(time, *cells)` block per call, and the gridded path requires
  the full `time` axis in that block. Chunk size is also the memory lever — one block
  is held whole inside the fit, with a few `O(years x periods x cells)` temporaries,
  so a dense continental grid passed in one piece costs several times its own size.
  See [Chunking guidance for gridded indices](#chunking-guidance-for-gridded-indices)
  for measured block sizes.
- The canonical lazy xarray/Dask SPI/SPEI workflow is the teaching notebook
  `notebooks/zarr_dask_spi_spei.ipynb`: the public typed API on Dask-backed DataArrays
  (`xr.apply_ufunc(..., dask="parallelized")`), one full time chunk with spatial
  chunks driving task parallelism, and precipitation/PET exact-aligned at
  preparation time so SPEI never relies on coordinate intersection.
- Persist results to a separate consolidated Zarr v2 store rather than back into
  the prepared inputs. The notebook's `data/e2e/climate_indices_output.zarr` is
  the sample path: `float32` index variables keep the API's `long_name`,
  dimensionless `units`, `references`, `scale`, `distribution`, Calibration
  Period years, `climate_indices_version`, and appended `history`, plus a
  `periodicity` attribute that names the Timescale unit. CF defines no
  `standard_name` for drought indices, so inherited input names such as
  `precipitation_amount` are dropped rather than mislabeling the result.
- Rerun persistence is a local, single-writer replacement: stage beside the
  target, then rename the previous output aside and swap, so a failed
  calculation or write leaves the completed store intact. Close readers before
  rerunning, keep input/output/staging paths non-overlapping, and do not run
  concurrent writers against one output path; the final directory swap is not
  crash-atomic.
- Reopen persisted results with `xr.open_zarr(..., consolidated=True)` in a fresh
  handle: inspect metadata and chunks without loading data, compute only the
  selected diagnostics, then close the handle. The reopened store is independent
  of the prepared inputs, so analysis does not recompute SPI/SPEI.
- Run the notebook CI command before publishing examples:
  `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/xarray_getting_started.ipynb notebooks/palmer_indices_xarray.ipynb notebooks/eddi_xarray.ipynb`.
- Execute the end-to-end tutorial from a fresh kernel before publishing it:
  `bash scripts/smoke_e2e_notebook.sh` prepares the pinned sample inputs and
  runs `notebooks/zarr_dask_spi_spei.ipynb` end to end into a scratch directory,
  so a local run never rewrites the committed notebook. CI executes the
  tutorial only via `tests/test_e2e_with_dask.py`, which runs its cells against
  a synthetic store; that suite deliberately skips the Dask Client cell (#829),
  so this local run is the only full-notebook check.

## Chunking guidance for gridded indices

Two rules fix the shape of a chunked input:

1. **Keep `time` in a single chunk.** Distribution fitting needs each cell's
   complete series, so the typed index adapters reject a split `time` with
   `CoordinateValidationError` ([ADR-0003](adr/0003-dask-time-dimension-single-chunk.md)),
   and the stateful fire adapters require the same for their weather inputs.
2. **Spatial chunks are the parallelism and memory lever.** A Dask block is
   fitted whole, and the fit materializes the reshaped `(years, periods, *cells)`
   block plus its per-period intermediates, so the working set grows with the
   cells in a block, not with the size of the grid.

For the 40-year monthly gamma SPI/SPEI path the fit's working set is about
60 KB per cell — roughly 16x the 3.84 KB the cell's own 480-step float64 series
occupies. On the #893 reference grid (38 x 87 = 3,306 cells, 480 months, Python
3.13.13, macOS arm64, `scheduler="synchronous"`, peak RSS measured above a
dry-run baseline of the same process):

| Spatial chunk | Blocks | Working set | SPI-3 wall time |
| --- | ---: | ---: | ---: |
| `38 x 87` (one block) | 1 | 205 MB | 0.98 s |
| `20 x 20` | 10 | 24 MB | 0.30 s |
| `10 x 10` | 36 | 6 MB | 0.32 s |
| `5 x 5` | 144 | 3 MB | 0.36 s |

Recommendations for monthly grids:

- Keep each block under roughly 100 MB of working set — around 1,500 cells at
  this per-cell cost. The 38 x 87 reference grid is not large, but as one block
  it costs 205 MB and is about 3x slower than the same grid in `10 x 10` blocks.
- Chunk both spatial dimensions (`10 x 10` to `20 x 20`) rather than long rows:
  square-ish blocks keep the per-period reductions proportionate and spread the
  work over more tasks.
- Leave a few blocks per worker so the scheduler has tasks to balance; the
  scaling harness re-chunks spatial dims for the worker count it is given (see
  `benchmarks/parallel_scaling.py`).
- Daily grids carry up to 366 steps per cell-year instead of 12, so the per-cell
  working set is proportionally larger and cells per block should shrink with it.

Rechunk once, at read or prepare time, when the stored layout differs from the
shape the computation wants — a Zarr store with one `time` chunk per year, or a
single chunk spanning the whole grid. Rechunking is a data copy, so pay it once
before the index calls rather than on every call. The teaching notebook
`notebooks/zarr_dask_spi_spei.ipynb` prepares its store with `time` as one chunk
and `10 x 10` spatial blocks, the layout the table above measures.

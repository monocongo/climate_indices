# xarray Compatibility Matrix

:::{warning}
The xarray API is beta in 3.0.0 and is promoted no earlier than 3.1.0
([ADR-0012](adr/0012-xarray-api-stays-beta-through-3.0.0.md)). Numerical results
are expected to match the stable NumPy API, while parameter inference, metadata,
and coordinate behavior may change with a minor version, never in a patch release.
:::

| Feature | Supported | Coverage |
| --- | --- | --- |
| `DataArray` inputs for SPI | Yes | `tests/test_xarray_equivalence.py` compares 1-D and gridded xarray outputs to NumPy. |
| `DataArray` inputs for SPEI | Yes | `tests/test_xarray_equivalence.py` compares xarray precipitation/PET inputs to NumPy. |
| `DataArray` inputs for EDDI | Yes | EDDI wrapper and metadata tests cover the typed public API. |
| `DataArray` inputs for flood PE, EDI, and I_F | Yes | `flood.effective_precipitation()`, `flood.edi()` (also `climate_indices.edi()`), and `flood.flood_index()` accept daily precipitation/PE with CF units converted to mm. Gregorian dates are converted to all-leap days before each NumPy computation and restored afterward; PE must be computed before EDI/I_F and use the same duration. The latter two infer calendar and Calibration Period years when omitted; I_F still requires `year_start_month`. A complete `time` Dask chunk is mandatory; spatial chunks run independently. `tests/test_flood_xarray.py` compares NumPy, eager xarray, and Dask outputs. These indices describe flood potential, not observed flooding. When chaining xarray PE into EDI/I_F, the second adapter reconstructs non-leap February 29 by interpolating **PE**, not the rainfall that produced it; the chained result can differ from applying both NumPy kernels to a single all-leap rainfall array ([#1147](https://github.com/monocongo/climate_indices/issues/1147)). |
| `DataArray` inputs for PET Thornthwaite | Yes | xarray adapter tests cover scalar and spatial latitude handling. |
| `DataArray` inputs for PET Hargreaves | Yes | xarray adapter tests cover aligned daily temperature inputs. |
| `DataArray` inputs for PNP | Yes | PNP wrapper tests cover scale handling and metadata. |
| `DataArray` inputs for PCI | Yes | PCI uses a manual scalar-output wrapper. |
| `DataArray` inputs for KBDI | Yes | `fire.kbdi()` resolves the `kbdi`/`kbdi_imperial` CF entry per call from `units`; supports `return_state`/`initial_state`, `spin_up`, `nan_policy`/`max_gap_days`, and CF `units`-attribute unit inference for the weather inputs and an attributed mean annual climatology. An attached time coordinate must be consecutive daily observations per input; a dimension-only time axis is aligned positionally. Overlapping timesteps are inner-aligned, dropped timesteps emit `InputAlignmentWarning`, and alignment that would drop non-time coordinates is rejected. See `tests/test_fire_kbdi.py`. |
| `DataArray` inputs for HDW | Yes | `fire.hot_dry_windy()` reduces the caller-named `level_dim` (default `"level"`) to the layer maximum via `xr.apply_ufunc`; CF metadata comes from the `hdw` registry entry, and a CF `units` attribute on temperature is converted to Celsius. Not recursive: every other dimension, including `time` if present, is a plain passthrough with no cadence requirement. Dimensions shared by the inputs must carry identical, identically ordered coordinate labels, because the inputs are matched with xarray's exact join rather than aligned or reindexed. `level_dim` must be a single Dask chunk; other dimensions chunk freely. See `tests/test_fire.py`. |
| `DataArray` inputs for CFFWIS | Yes | `fire.cffwis()` returns one `Dataset` variable per selected output (`ffmc`, `dmc`, `dc`, `isi`, `bui`, `fwi`, `dsr`), each carrying its own registry CF metadata. Latitude is inferred from a `lat`/`latitude` coordinate when not supplied and month from the datetime `time` coordinate; CF `units` attributes on temperature and precipitation are converted. Per-cell day-length bands come from broadcasting latitude through `xr.apply_ufunc`, one call per Dask spatial block. A daily coordinate clearly not at noon emits `ClimateIndicesWarning`, while sub-daily coordinates and a multi-chunk `time` dimension are rejected. With `return_state=True` the result is a `CFFWISResult` whose selected fields are DataArrays (`None` for components omitted from `outputs`) and whose state stays NumPy. See `tests/test_fire_cffwis_behavior.py`. |
| `DataArray` inputs for Haines | Yes | `fire.haines_index()` scores caller-selected levels elementwise through `xr.apply_ufunc` with no core dimension, so no dimension has a Dask chunk constraint. CF metadata comes from the per-variant `haines_low`/`haines_mid`/`haines_high` registry entry, including `climate_indices_variant`, and a CF `units` attribute on any temperature input is converted to Celsius. Not recursive: every dimension, including `time`, is a plain passthrough with no cadence requirement. Dimensions shared by the inputs must carry identical, identically ordered coordinate labels, because the inputs are matched with xarray's exact join rather than aligned or reindexed. The elevation-driven variant selection of `fire.haines_index_from_profile()` is NumPy-only. See `tests/test_fire_haines.py`. |
| `DataArray` inputs for PDSI (Palmer) | Yes | `climate_indices.pdsi()` returns an `xr.Dataset` with one variable per index (`pdsi`, `phdi`, `pmdi`, `z_index`), each carrying its own registry CF metadata. AWC is a broadcast input like PET's latitude; a 3-D input reaches `palmer.pdsi()` as one `(time, *cells)` block per Dask spatial block. scPDSI has no xarray entry point ([ADR-0011](adr/0011-palmer-spatial-block-and-per-location-scpdsi.md)). See `tests/test_spatial_kernel.py`; the manual-rewrap pattern remains in `notebooks/palmer_indices_xarray.ipynb`. |
| Coordinate preservation | Yes | Adapter tests verify time and spatial coordinates are preserved. |
| CF-style metadata | Yes | `CF_METADATA` registry and adapter tests verify `long_name`, `units`, `references`, version, and history attributes. |
| Dask-backed arrays | Yes, constrained | The time dimension must be a single chunk, except for the PET adapters, which rechunk a split `time` internally. Adapter tests verify detection and validation. Chunk axes the CLI leaves to Dask (`"auto"`, the KBDI and flood-index grid and division inputs) resolve against a 100 MB array chunk budget rather than Dask's own default; axes pinned with `-1` are unaffected. See [Chunking guidance for gridded indices](#chunking-guidance-for-gridded-indices) for block sizes. |
| Spatial (gridded) kernels | Yes, for SPI, SPEI, PET, EDDI, percentage of normal, PDSI, and flood PE/EDI/I_F | `spi`/`spei` receive a time-major `(time, *cells)` block and fit every cell in one pass, so a gridded gamma run costs one kernel call per Dask block instead of one call per cell ([ADR-0009](adr/0009-spatial-block-declaration.md)). The PET entry points (`pet_thornthwaite`, `pet_hargreaves`) hand `indices.pet` and `eto.eto_hargreaves` the same block layout with the per-cell latitude array, so a gridded PET run costs one call per block ([#941](https://github.com/monocongo/climate_indices/issues/941)). `indices.eddi` and `indices.percentage_of_normal` rank, and divide by the normals of, every cell of the block in one pass ([#942](https://github.com/monocongo/climate_indices/issues/942)). `palmer.pdsi` receives the block with AWC broadcast per cell, so a gridded PDSI run costs one recursion per block ([#1016](https://github.com/monocongo/climate_indices/issues/1016)). The NumPy API keeps 2-D input on the legacy `(years, periods)` reading; a 3-D or higher input is a block for `spi`/`spei`/`pet`/`pdsi` unless its first cell axis is the one shape that is ambiguous with that reading (12 or 366, i.e. a `(years, periods, *cells)` array), which must be declared, and for EDDI and percentage of normal every 3-D input must be declared with `spatial_time_major=True`. Inputs with a single non-core dimension keep the per-cell path. See `tests/test_spatial_kernel.py`. |
| Calendar semantics | Yes, constrained | Standard/gregorian/proleptic_gregorian `datetime64` only; monthly input must begin in January and daily input on January 1. Daily values are converted to the 366-day calendar (February 29 synthesized from February 28 and March 1) and restored afterward. A partial final year is supported; `cftime` calendars are rejected. See [ADR-0004](adr/0004-xarray-calendar-semantics.md). |
| Automatic temporal inference | Yes | Monthly and daily time-coordinate inference is covered by adapter tests. |
| Multi-input alignment | Yes | SPEI aligns precipitation and PET with an inner join on the time dimension and emits a warning when timesteps are dropped. Shared cell dimensions must carry matching coordinates: intersecting them would silently drop grid cells, so a mismatch raises `CoordinateValidationError` instead. |

## Stateful fire indices

KBDI's adapter (`fire.kbdi()`) and the CFFWIS adapter (`fire.cffwis()`, #807)
are shipped, following the recursive contract in
[ADR-0006](adr/0006-fire-recursive-state-and-execution.md). The weather-only
Fosberg Fire Weather Index is available today too, via the NumPy layer with
no xarray adapter needed (it is elementwise with no dimension to reduce).
Hot-Dry-Windy's adapter (see the main table above) is not recursive either,
but it does reduce a named vertical dimension, so it does not follow the
stateful contract below; it needs only that dimension in a single Dask chunk.
Haines' adapter is elementwise like Fosberg's, but it still goes through
`xr.apply_ufunc` -- with no core dimension, and therefore no chunk constraint
-- so that it can carry per-variant registry metadata and CF `units`
conversion; its elevation-driven variant selection stays in the NumPy layer.

These adapters are recursive: each daily value needs its predecessor.
Dask-backed inputs must therefore keep the complete `time` dimension in one
chunk for every time-varying weather variable. Spatial chunks remain
supported and parallelize: KBDI's and CFFWIS's NumPy cores already vectorize
over an arbitrary spatial shape, so their adapters dispatch one call per Dask
spatial chunk with the full `time` axis, rather than looping per grid cell.
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
- For a complete runnable gridded SPI and SPEI example with the chunking and
  scheduler guidance applied, see [Gridded Performance](performance.md).
- Keep Dask chunks spatial when possible and leave `time` as one chunk before
  calling index functions. Spatial chunks set the parallelism granularity: SPI and
  SPEI fit a whole `(time, *cells)` block per call, and the gridded path requires
  the full `time` axis in that block. Chunk size is also the memory lever — one block
  is held whole inside the fit, with a few `O(years x periods x cells)` temporaries,
  so a dense continental grid passed in one piece costs several times its own size.
  See [Chunking guidance for gridded indices](#chunking-guidance-for-gridded-indices)
  for measured block sizes.
- Axes the CLI leaves to Dask (`"auto"`, currently the KBDI and flood-index grid and division inputs)
  resolve against a 100 MB array chunk budget
  (`climate_indices._cli.DEFAULT_ARRAY_CHUNK_SIZE`) instead of Dask's own 128 MiB
  default; axes pinned with `-1` are unaffected. Change the budget with explicit
  `chunks` when opening your own dataset, with the `DASK_ARRAY__CHUNK_SIZE`
  environment variable, or with `dask.config.set({"array.chunk-size": ...})` — a
  configured budget is honored rather than overwritten.
- Choose the scheduler at the materialization call: the xarray API never imports
  or configures Dask ([ADR-0002](adr/0002-multiprocessing-cli-dask-xarray.md)). The
  gridded kernels are CPU-bound Python/scipy work, and their Python-level portion
  does not run in parallel under the default threaded scheduler, so materialize a
  lazy result on worker processes — given `precip`, whose `time` dimension is a
  single chunk:

  ```python
  from climate_indices import spi
  from climate_indices.indices import Distribution

  spi_lazy = spi(
      values=precip.chunk({"time": -1, "lat": 20, "lon": 20}),
      scale=3,
      distribution=Distribution.gamma,
  )
  spi_grid = spi_lazy.compute(scheduler="processes")
  ```

  In a script rather than a notebook, that call belongs under
  `if __name__ == "__main__":` while Dask spawns its workers — the `processes`
  scheduler's default start method on every platform — because each worker
  re-imports the entry module. A Zarr write takes the same scheduler through the
  delayed write:
  `spi_lazy.to_zarr(path, compute=False).compute(scheduler="processes")`.
  NetCDF writes do not: their backend lock is built for the scheduler that is
  active when the write graph is built, and the default one cannot be pickled to
  worker processes, so load the result first and write it in memory when the full
  result fits, or let a distributed client stream the write for larger results.
- Every `.compute(scheduler="processes")` call builds and tears down its own
  process pool, so a computation short relative to that start-up spends most of
  its wall clock there: on the 38 x 87 reference grid the SPI pass measured 1.37 s
  with the fresh one-worker pool against 0.76 s with a pre-created, warmed pool,
  and 1.26 s against 0.23 s at eight workers
  ([#928](https://github.com/monocongo/climate_indices/issues/928) harness,
  recorded on [#927](https://github.com/monocongo/climate_indices/issues/927)).
  The serial reference and the fresh-pool rows for SPI, SPEI, PET and EDDI are
  tabulated in
  [the #929 before/after table](https://github.com/monocongo/climate_indices/blob/main/benchmarks/README.md#beforeafter-on-the-reference-grid-929).
  Keep a `dask.distributed.Client` alive across calls — the [tutorial's client
  cell](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb)
  does, and `distributed` ships with the `dev` extra — or pass a pre-created pool
  when the work is short or repeated.
- The threaded scheduler remains the right choice for I/O-bound or small
  in-memory work — opening a store, a reduction such as `.mean("time")`, or a grid
  small enough that serializing each block to a worker process costs more than
  the parallelism saves.
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

Three rules fix the shape of a chunked input:

1. **Keep `time` in a single chunk.** Distribution fitting needs each cell's
   complete series, so the SPI/SPEI adapter path rejects a split `time` with
   `CoordinateValidationError`
   ([ADR-0003](adr/0003-dask-time-dimension-single-chunk.md)) and the stateful
   fire adapters require the same for their weather inputs. The PET adapters are
   the exception: they pass `allow_rechunk=True` and rechunk a split `time`
   internally, paying that copy inside every call.
2. **Spatial chunks are the parallelism and memory lever.** A Dask block is
   fitted whole, and the fit materializes the reshaped `(years, periods, *cells)`
   block plus its per-period intermediates, so the working set grows with the
   cells in a block, not with the size of the grid. The PET spatial kernels take
   one block per call as well, so the same block sizing applies to them.
3. **Chunk the shared dimensions of a multi-input index the same way.** SPEI
   aligns precipitation and PET with `xr.align(join="inner")`, but the adapter
   validates shared non-time dimensions first: a mismatched `lat`/`lon`
   coordinate raises `CoordinateValidationError`, and only `time` is left for
   the inner join to intersect; the adapter's `InputAlignmentWarning` reports
   dropped `time` steps. That join does not rechunk, and each input's `time`
   chunking is validated independently, so a chunk-layout mismatch on a shared
   dimension survives to the compute: `xr.apply_ufunc(...,
   dask="parallelized")` makes Dask unify the chunks, inserting a
   `rechunk-merge` stage that copies the data when the graph computes — an extra
   copy inside every index that consumes the mismatched pair. Give PET the
   layout of the dimensions it shares with precipitation (the one
   `notebooks/zarr_dask_spi_spei.ipynb` stores):

   ```python
   pet = pet.chunk({dim: blocks for dim, blocks in precip.chunksizes.items() if dim in pet.dims})
   ```

   PET inputs with fewer dimensions than precipitation broadcast, so a
   `(time,)` series or a `(time, lat)` field needs no chunking on the dimensions
   it lacks.

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

Recommendations for monthly grids:

- Keep each block under roughly 100 MB of working set — around 1,600 cells at
  this per-cell cost. That figure is a per-block measurement taken with
  `scheduler="synchronous"`, one block resident at a time; a threaded or
  distributed worker can hold several ready blocks plus their inputs and
  outputs, so budget from the memory a worker can spare per block it runs at
  once. The 38 x 87 reference grid is not large, but as one block it costs
  205 MB and is about 3x slower than the same grid in `10 x 10` blocks.
- Size blocks by cell count rather than shape: the fit's working set grows with
  the cells a block holds, and with a fixed per-block budget the task count
  comes from the grid size, so aspect ratio changes neither. Chunk both spatial
  dimensions (the `10 x 10` to `20 x 20` range here) rather than long rows so
  the block size tracks the budget; chunking a single dimension ties it to the
  grid's row length, which can overshoot the budget on wide grids.
- Leave at least a few blocks per worker so the scheduler has work to balance;
  `benchmarks/parallel_scaling.py` re-chunks spatial dims to one block per
  worker for its strong-scaling runs.
- Daily grids carry up to 366 steps per cell-year instead of 12, so the per-cell
  working set is roughly 30x larger; at the same ~100 MB ceiling that is about a
  `7 x 7` block where a monthly grid uses `10 x 10`.

Rechunk once, at read or prepare time, when the stored layout differs from the
shape the computation wants — a Zarr store with one `time` chunk per year, or a
single chunk spanning the whole grid. `.chunk(...)` only sets the layout of a
lazy graph; the copy happens when the graph is computed, so persist the array
(or write the rechunked layout back to the store) before the index calls to pay
the copy once. Without that, every separate index graph that consumes the array
can reread and redo the rechunk work. The teaching notebook
`notebooks/zarr_dask_spi_spei.ipynb` prepares its store with `time` as one chunk
and `10 x 10` spatial blocks, the layout the table above measures. The
`{"lat": 50, "lon": 50}` shape in ADR-0003 is legal chunking, not a size
recommendation: on a 40-year monthly series it is roughly 150 MB per block,
above the ~100 MB budget recommended here.

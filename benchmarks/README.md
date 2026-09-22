# Benchmarks

Reproducible performance measurement for the #893 vectorization/parallelization
epic. Reports here are evidence artifacts, not CI gates.

## Profiling the gridded SPI workflow (#921)

```bash
uv run benchmarks/profile_gridded_spi.py
```

The script builds the reference grid from #893 as deterministic synthetic data
(38x87 cells, 40 years of monthly precipitation, seed 42), exercises the
fitting/transform path on a small warm-up grid so first-call imports and caches
stay out of the measurement window, then runs `climate_indices.spi(DataArray)`
on the numpy-backed (in-memory) adapter branch — the serial baseline for
gridded SPI. A Dask-backed input returns a lazy result; Dask scheduling and
multi-core scaling belong to #927 and #928. Gridded SPI/SPEI now reach the NumPy
core one spatial block at a time (#923, see the conversion status below), so the
figures here describe the per-cell path this harness was built to measure. The
report stays committed as the pre-#923 profile artifact; the before/after table
is in the #929 section below, and rerunning the script now reports the spatial
path instead.

Each run measures three things and always rewrites
`benchmarks/results/profile_gridded_spi.txt`:

1. an unprofiled baseline at the default INFO log level,
2. the same run under `cProfile` (the raw report),
3. an unprofiled baseline with logging at WARNING (isolates the per-cell log cost).

Both baselines are the minimum of two runs, so noise adds time rather than
flattering whichever level runs first.

Rerunning on another machine therefore replaces the committed reference report;
check `git diff` before committing a refreshed artifact.

### Findings (macOS arm64, Python 3.13, 2026-09)

Reference grid: 3306 cells x 480 months.

| measurement | time |
|---|---|
| baseline, INFO logging | 1.1 s |
| baseline, WARNING logging | 0.7 s |
| cProfile-instrumented | 2.4 s |

Per-cell logging costs ~0.4 s (~35%) of the INFO wall clock. `cProfile` roughly
doubles the wall clock on this call-heavy path, so the profiled report is for
relative attribution, not absolute timing.

Hottest paths by cumulative time (2.392 s profiled total):

| path | calls | cumtime |
|---|---|---|
| `numpy._vectorize_call_with_signature` | 1 | 2.391 s |
| `climate_indices/xarray_adapter.py:507` (`wrapper`, one call per grid cell) | 3306 | 2.381 s |
| `climate_indices/indices.py:417` (`spi`) | 3306 | 2.377 s |
| `climate_indices/compute.py:1085` (`transform_fitted_gamma`) | 3306 | 1.860 s |
| `structlog/stdlib.py:218` (`info`) | 19837 | 1.187 s |
| `climate_indices/compute.py:942` (`gamma_parameters`) | 3306 | 0.971 s |
| `climate_indices/compute.py:779` (`_check_goodness_of_fit_gamma`) | 3306 | 0.449 s |

Hottest paths by self time:

| path | tottime |
|---|---|
| `climate_indices/compute.py:779` (`_check_goodness_of_fit_gamma`) | 0.172 s |
| `structlog/dev.py:296` (console renderer) | 0.141 s |
| `climate_indices/compute.py:743` (`_ks_poor_fit_p_value`) | 0.105 s |
| `scipy/stats/_continuous_distns.py:3612` (`_cdf`) | 0.089 s |
| `structlog/_frames.py:36` (`_find_first_app_frame_and_name`) | 0.048 s |

Interpretation:

- Per-grid-cell invocation is still real, via
  `xr.apply_ufunc(..., vectorize=True)`: 3306 Python calls into
  `xarray_adapter.py:507`, ~0.7 ms each under the profiler.
- The adapter path emits six `structlog` info records per cell from
  `indices.spi`/`compute` (19836 / 3306), plus one run-level
  `xarray_adapter_completed` record. The profiler attributes about half of the
  profiled total to their rendering and call-site frame inspection; the
  unprofiled runs put the real cost at ~0.4 s of 1.1 s. This is a logging-volume
  cost, not an algorithmic one.
- The numerical work is a per-cell gamma fit with a Kolmogorov-Smirnov
  goodness-of-fit check (`compute.py:942`, `compute.py:779`) plus scipy `cdf`/`ppf`
  transforms — all serial Python/scipy calls per cell. Compiled scipy internals
  are charged to their Python callers, so their cumtime share is approximate.
- `src/climate_indices/` contains no Numba usage; the epic's "already
  Numba-accelerated kernels" premise does not hold for the SPI path, and the
  warm-up in the harness only absorbs first-call imports and caches.
- The epic's ">11 minutes" reference matches the explicit lat/lon `for` loops in
  `notebooks/muitprocess_spi_nclimgrid.ipynb`, which bypass the canonical adapter
  path. The canonical path measured here is 1.1 s, so that comparison says more
  about the entry point than about per-core speed; the #929 section below
  evaluates the epic's 10x criterion against this harness's baseline instead.

## Parallel scaling of the gridded indices (#928)

```bash
uv run benchmarks/parallel_scaling.py                                 # spi,spei on 1, 2, 4, ... workers up to the CPU count
uv run benchmarks/parallel_scaling.py --cores 1,2,4,8 --indices spi,spei,pet,eddi --repeat 5
uv run benchmarks/parallel_scaling.py --indices spi,spei,pet,eddi --serial-only --repeat 3
uv run benchmarks/parallel_scaling.py --indices spi,spei,pet,eddi | tee benchmarks/results/parallel_scaling.txt
```

The script runs the same reference grid through the public xarray API on a
Dask-backed input with the `processes` scheduler, for SPI, SPEI, Thornthwaite
PET (`pet_thornthwaite`, latitude passed as a `(lat,)` coordinate so the spatial
kernel stays on its broadcast path) and EDDI. It re-chunks the spatial
dimensions for each worker count (time stays a single chunk, per ADR-0003) and
reports the fastest of `--repeat` runs after a warm-up, plus the block count and
the parallel efficiency. Speedup is relative to the first `--cores` entry. Every
`compute()` call creates a fresh process pool, so pool start-up is inside every
timing, not only the baseline: the harness measures the out-of-the-box
`processes` scheduler. The serial in-memory number to compare against is printed
by the same run, and `--serial-only` prints just that reference; it filters
goodness-of-fit warnings, so it sits just below a warning-visible default call
such as the #921 baseline above.

The compute call passes `chunksize=1`: Dask's default batches up to six ready
tasks per submission, which runs a whole six-block batch sequentially on one
worker and silently flattens the curve.

PET for SPEI is synthetic (a fixed fraction of the precipitation); the Dask
sweep runs with logging quiet and goodness-of-fit warnings filtered, so the Dask
timings measure the fitting path rather than the log renderer. The before/after
table for #929 is below.

## Before/after on the reference grid (#929)

```bash
uv run benchmarks/parallel_scaling.py --indices spi,spei,pet,eddi --repeat 3
uv run benchmarks/parallel_scaling.py --indices spi,spei,pet,eddi --serial-only --repeat 3
```

Each run first times the in-memory, single-process call for every requested
index, at the default INFO log level and again with logging quiet, then (unless
`--serial-only`) the Dask worker counts. The pre-vectorization numbers below were
taken with `--serial-only` on the commit before the first spatial-block
conversion, `d4e9ba0d` (the parent of the #923 merge), with this script copied
into that checkout; both sides ran the same Python 3.14.7 environment on a
10-core macOS arm64 machine.

Reference grid: 38x87 cells, 40 years monthly, scale 3, calibration 1981-2010;
fastest of three runs after a warm-up. "Quiet" pins per-cell logging and
goodness-of-fit warnings off, which isolates the fitting path. Both columns
filter goodness-of-fit warnings, as the Dask runs do, so the INFO/quiet delta
isolates log rendering; INFO is the library's default log level, not the full
cost of a warning-visible default call.

| index | serial before, INFO | serial before, quiet | serial after, INFO | serial after, quiet | vectorization speedup | best parallel after |
| --- | --- | --- | --- | --- | --- | --- |
| SPI | 1.120 s | 0.898 s | 0.204 s | 0.204 s | 4.4x | 0.837 s (2 workers) |
| SPEI | 1.193 s | 0.783 s | 0.221 s | 0.220 s | 3.6x | 0.877 s (2 workers) |
| Thornthwaite PET | 1.429 s | 1.063 s | 0.020 s | 0.020 s | 53x | 0.685 s (1 worker) |
| EDDI | 14.960 s | 14.717 s | 0.043 s | 0.043 s | 342x | 0.699 s (1 worker) |

"Vectorization speedup" is serial-before-quiet over serial-after-quiet: the
spatial block replaced the per-cell Python loop, so one kernel call per non-core
block replaces 3306. The INFO column collapses after the conversion because the
per-cell `structlog` volume goes with the loop (SPI: 19837 records before, a few
per block after).

The #921 profile's 0.7 s quiet SPI figure is the Python 3.13.13 run of the same
pre-conversion path; the 0.898 s here is a Python 3.14.7 measurement of it, and
rerunning the identical command a few minutes later gave 0.770 s, so read the
speedup column as ±15% rather than exact.

Raw output: `benchmarks/results/serial_before.txt` (pre-conversion) and
`benchmarks/results/parallel_scaling.txt` (post-conversion). The harness only
writes to standard output, so neither file is rewritten by the commands above:
`parallel_scaling.txt` is the `tee` of the #928 full-sweep command, and
`serial_before.txt` is the historical pre-conversion run on `d4e9ba0d` with the
harness copied in, so `--serial-only` here re-measures the post-conversion path
instead. Re-measure rather than trusting either file on another machine or
dependency set; the pre-conversion artifact's header omits the dask and xarray
versions the current script prints.

### The 10x criterion (#893)

**Met end to end for EDDI only; SPI and SPEI fall short.** Against the
pre-vectorization serial canonical path on the reference grid:

- EDDI: 14.717 s -> 0.699 s through the Dask path, ~21x end to end, and 342x
  for the in-process vectorization alone.
- PET: 53x in-process, but 1.6x end to end (1.063 s -> 0.685 s) because the Dask
  path is slower than the serial in-memory call at this size.
- SPI: 4.4x in-process, 1.1x end to end (0.898 s -> 0.837 s).
- SPEI: 3.6x in-process, 0.89x end to end (0.783 s -> 0.877 s).

The SPI/SPEI shortfall is Dask overhead, not a serial-vs-parallel gap in the
kernels: after vectorization each finishes in ~0.2 s, while a fresh `processes`
pool plus its start-up and the result transfer add 0.66-0.71 s at one worker and
about 1.0 s at eight, on the same added-cost basis (SPI: 0.863 s pooled against
0.204 s in memory at one worker, 1.197 s at eight, i.e. ~0.99 s of added cost).
Parallelism pays when the work per block exceeds that
cost, and the 3306-cell reference grid no longer does. The epic's ">11 minutes"
reference measures the explicit lat/lon loops in
`notebooks/muitprocess_spi_nclimgrid.ipynb`, which bypass the adapter: the
canonical path was 1.1 s before the conversion. Reaching 10x for SPI and SPEI
therefore needs a larger grid than the reference one: even the warmed-pool
figure recorded for this grid (0.23 s at eight workers, `docs/xarray_compatibility.md`,
#927 harness) is 3.9x the 0.898 s serial baseline, so executor reuse alone does
not close the gap.

## Before/after timings for the gridded Palmer PDSI kernel (#937)

```bash
uv run benchmarks/profile_gridded_palmer.py
```

The script builds the same #893 reference grid as synthetic precipitation, PET,
and per-cell AWC (38x87 cells, 40 years monthly, seed 42), warms up on a small
grid, then times `palmer.pdsi()` two ways on the reference grid: the per-cell
loop `__main__._apply_along_axis_palmers` used before #937 (one call per grid
cell, 3306 calls), and the spatial block path #937 added
(`spatial_time_major=True`, one call for the whole grid). Each measurement is
the minimum of two runs. #929, which would otherwise own the general
before/after table format for this epic, is unmerged with no branch as of
#937, so this script defines its own rather than waiting on it; #929 may fold
this into a shared format later. Always rewrites
`benchmarks/results/profile_gridded_palmer.txt`.

### Findings (macOS arm64, Python 3.14, 2026-09)

Reference grid: 3306 cells x 480 months.

| measurement | time |
|---|---|
| per-cell loop (pre-#937) | 135.4 s |
| spatial block (#937) | 1.2 s |
| speedup | 111.3x |

Raw output: `benchmarks/results/profile_gridded_palmer.txt` (the same run; the
speedup is computed from unrounded times).

The per-cell figure is consistent with the #921 SPI profile's per-cell
overhead (structlog volume, one Python call per cell) plus Palmer's own
per-location water-balance and spell-recursion cost; #937's vectorized
recursion pays that cost once for the whole grid instead of once per cell.

## Per-cell invocation inventory (#922)

Static audit of the index-invocation sites in `src/climate_indices/`, as of
`02c9cb38`, for Python-level loops that call an index function once per grid cell
or per time series. Scope is the library and its xarray adapter layer; notebooks
are excluded (see the #921 findings above for why the notebook figure differs).
The three `__spi__.py` rows it listed were removed with that module in 3.0.0
(#957) and are omitted here.

Reference grid: 38 x 87 = 3306 cells, 40 years monthly = 480 time steps, so one
per-cell pass is 3306 calls.

### Canonical xarray adapter path (per-cell loop present)

| site | invocation | loop dimensions | calls per adapter call |
|---|---|---|---|
| `xarray_adapter.py:1717` (`xarray_adapter`, Dask branch) | wrapped `spi`/`spei`/`eddi`/`percentage_of_normal` | `lat x lon`, one call per cell across all blocks | 3306 |
| `xarray_adapter.py:1808` (`xarray_adapter`, in-memory branch) | same | `lat x lon` | 3306 |
| `xarray_adapter.py:2084` (`pet_thornthwaite`) | `indices.pet` | `lat x lon` | 3306 |
| `xarray_adapter.py:2353` (`pet_hargreaves`) | `eto.eto_hargreaves` | `lat x lon` | 3306 |

All four pass `vectorize=True` to `xr.apply_ufunc`, so the wrapped 1-D kernel
runs once per combination of the non-core (broadcast) dimensions, one call per
cell: 3306 for the 38 x 87 reference grid, and extra non-core dimensions
multiply that count. On a Dask-backed input `dask="parallelized"` schedules
those calls as per-block tasks: chunking the spatial dimensions changes task
count and wall time, not the per-cell total. The core dimension (`time`) must be
a single chunk on the generic adapter path (`validation.validate_dask_chunks`
raises before `apply_ufunc` runs); `pet_thornthwaite` and `pet_hargreaves`
pass `dask_gufunc_kwargs={"allow_rechunk": True}` so they can rechunk a split
time dimension. Counts multiply per invocation: each adapter
call covers one index, scale, and distribution, so a 14-pass SPI run over
`--scales 1 2 3 6 9 12 24` and both distributions is 14 repeated adapter calls
(46,284 per-cell calls). The CLI reaches the same total through its own scale
and distribution loops (`__main__.py:1519-1520`).

This path is serial within a process and is the one the #921 profile measured:
3306 calls into the calendar wrapper at `xarray_adapter.py:507` for a single
SPI-1/gamma run.

### Conversion status (#923)

`spi` and `spei` accept the spatial blocks the Dask path already schedules: the
adapter's `spatial_kernel=True` forwards `vectorize=False` to `xr.apply_ufunc` and
transposes the core dimension, so one `(time, *cells)` block reaches the NumPy core
and the gamma fitting, transform, and goodness-of-fit check run once per block
instead of once per cell. The Pearson Type III L-moment fit and its
goodness-of-fit check run once per block the same way (#940).
`tests/test_spatial_kernel.py` pins the call count, the
equivalence with the single-series path, and the NaN, partial-final-year, and daily
calendar contracts.

The PET entry points own their `xr.apply_ufunc` calls rather than the shared decorator,
because latitude is a broadcast input rather than a secondary time series. They now
forward `vectorize=False` too, handing `indices.pet` and `eto.eto_hargreaves` a
`(time, *cells)` block with the per-cell latitude array (#941), so Thornthwaite's
monthly day-length term and Hargreaves' daily radiation are computed once per block
instead of once per grid cell. A 2-D input, or a latitude carrying a dimension the
temperature does not, stays on the per-cell path.

`indices.eddi` and `indices.percentage_of_normal` now take the same block (#942). EDDI
ranks each calendar period's values against its own calibration climatology with a
cell-chunked count, one pass over the whole grid rather than a per-year,
per-period loop, and percentage of normal averages each cell's calendar-period normals
and divides the block by them. Both keep the legacy 1-D and 2-D behaviour, and both
require `spatial_time_major=True` for a 3-D input since their dimension errors are
pinned to `DataShapeError`. The ranking count holds one chunk of the
`(calibration years, years, *cells)` comparison at a time, bounded near 4 MB, so grid
size no longer multiplies into it.

`palmer.pdsi` gained a spatial block path at the NumPy layer in #937 (see
`### Conversion status (#937)` below), and its xarray adapter landed in #1016:
`climate_indices.pdsi()` takes DataArrays and returns a Dataset of the four
indices, reaching the kernel once per block with AWC broadcast like PET's
latitude. `palmer.scpdsi` stays per-location (ADR-0011) and has no xarray entry
point.

### Legacy CLI path (per-cell loop present, parallel across workers)

`__main__.py`'s layout classifier accepts `(lat, lon, time)` or
`(time, lat, lon)`, but the shared-array path this section measures stores a grid
time-last and rejects a time-major grid, which only the xarray-backed KBDI path
accepts; the mismatches that still survive validation are tracked in #932. The
counts below assume `(lat, lon, time)`, split along axis 0 (latitude) across a
`multiprocessing.Pool`, with the per-cell loop inside each worker. The loops run
in parallel across processes but are not eliminated, and each worker's per-cell
call carries the same per-cell overhead the #921 profile measured (per-cell
`structlog` records and the per-kernel goodness-of-fit check): the Pool divides
wall clock, it does not reduce total per-cell Python cost.

| site | invocation | loop dimensions | calls |
|---|---|---|---|
| `__main__.py:1289` (`_apply_along_axis`) | `_spi` via `np.apply_along_axis(axis=2)` | `lat x lon`, looped by `np.apply_along_axis` in Python | 3306 per scale x distribution (`:1519-1520`) |
| `__main__.py:1289` (`_apply_along_axis`) | `_pnp` via `np.apply_along_axis(axis=2)` | same | 3306 per scale only (`:1609`, no distribution loop) |
| `__main__.py:1347,1349` (`_apply_along_axis_double`, loop at `:1343,1345`) | `_spei`/`_pet` | `lat x lon` | 3306 |

The `__main__.py` sites duplicate
the adapter path's work on the same kernels, so a baseline measured through the
CLI and a baseline measured through the canonical path are not interchangeable.
`_apply_along_axis_palmers` (previously a per-cell loop at this table's `:1409,1411`)
is converted for grid input; see `### Conversion status (#937)` below.

### Conversion status (#937)

`palmer.pdsi()` accepts a time-major spatial block the same way `spi`/`spei` do
(`spatial_time_major=True`, ADR-0009/ADR-0011). Palmer's recursion has genuine
per-cell control flow, unlike the fitting-based kernels' pure arithmetic, so the
spatial path is a masked vectorization of the spell recursion itself rather than
a broadcast: every recursion stage takes an `active` cell mask and writes only
where it holds, and the K8 backtracking window is preallocated to the record
length instead of the historical `K8_SIZE = 40` bound. `__main__._apply_along_axis_palmers`
now passes a whole `(lat_chunk, lon, time)` grid chunk to one `palmer.pdsi()` call
for `DatasetLayout.GRID`, instead of the nested `for i / for j` loop the table above
described; `DatasetLayout.DIVISIONS` has no cell-adjacency structure to batch and stays
on the per-location loop. `tests/test_palmer_spatial.py` pins the equivalence with
the per-location path (bit-for-bit, not a tolerance — see ADR-0011 for why), the
ADR-0009 ambiguous-shape rejection, and the all-missing-cell and per-cell-AWC
contracts; `tests/test_main_palmers.py` pins the CLI worker's grid path the same way
`TestPalmersWorker` already pinned the division path. `palmer.scpdsi()` explicitly
rejects a spatial block (ADR-0011) and is unaffected.

`climate_indices.pdsi()` registers `palmer.pdsi` with the xarray adapter (#1016):
it forwards `vectorize=False`, hands the kernel a `(time, *cells)` block with the
per-cell AWC, and rewraps the four outputs as a Dataset. `palmer.scpdsi()` still
has no xarray entry point (ADR-0011).

### Already vectorized (no per-cell index invocation)

- `fire.py:1144` (`_kbdi_xarray`) passes `vectorize=False` and `fire.py:2460`
  (`_hdw_xarray`) omits it: those kernels loop over time (or the level dimension)
  and operate on whole block arrays, so cells are handled by NumPy operations
  rather than a Python call per cell.
- `indices.spi`, `indices.spei`, `indices.eddi`, `indices.percentage_of_normal`,
  `indices.pet`, and `eto.eto_hargreaves` each accept a declared `(time, *cells)`
  spatial block directly via their own `spatial_time_major` parameter, not only
  through the dispatch sites above. Their internal loops stay over time steps or
  calendar periods rather than over grid cells once given a block
  (`compute.py:797`, `compute.py:869` check goodness of fit per calibration time
  step; `indices.py:396` ranks EDDI per period). `indices.pci` is a single-year
  scalar with no loop at all.

### Per-cell calendar transforms (not index kernels, cost not accounted for)

The per-cell `np.apply_along_axis` sites in the CLI (`__main__.py:605`,
`__main__.py:1585`) call `utils.DailyCalendarPlan.to_all_leap` /
`to_gregorian`, not an index function, so they are outside the #923 conversion.
They are still per-cell Python calls: `np.apply_along_axis` loops the spatial
dimensions in Python, and each transform loops over years inside that call
(`utils.py:418`, `utils.py:458`). The daily adapter path runs equivalent
transforms per cell through `_compute_with_daily_calendar_plan`
(`xarray_adapter.py`), driven by the same plan methods, and counted inside the
wrapper above. Unmeasured overhead on both paths; no ticket owns it.

### Structural blockers

`palmer.pdsi` and `palmer.scpdsi` take two monthly series (precipitation and PET)
plus an available water capacity -- a per-cell field for `pdsi` blocks -- and loop
over years and months (`palmer.py:231`, `palmer.py:323`, `palmer.py:365`,
`palmer.py:733`); before #899 they also allocated a per-location `data` dict of
`(n_years, 12)` arrays. `palmer.pdsi` gained the n-D
kernel in #937 and the adapter-layer entry point in #1016, and
`_apply_along_axis_palmers` (`__main__.py:1165`) is converted for grid input.
The remaining per-cell Palmer index path is `scpdsi` (the CLI's divisions input
stays per-location by design). `scpdsi` additionally runs
`_palmer_wells.calculate` per location (`palmer.py:1035`, `palmer.py:1047`), a
per-month backtracking state machine, and duration-factor fits
(`self_calibration.py:342`, `self_calibration.py:394`, `self_calibration.py:449`)
inside the same per-location call; the CLI never invokes `scpdsi`, so those fits
are only reachable through the per-series API, and standard `pdsi` does not
self-calibrate. #899
refactors these internals with unchanged output, not into an n-D kernel. The #923
tasks name SPI, SPEI, and PET (its acceptance criteria name SPI), so Palmer was a
follow-up rather than part of that conversion.

Standard PDSI's structural blocker above is resolved (#937): the water balance and
CAFEC stages above are now shared elementwise arithmetic over a cell axis (the
"per-location `data` dict" is additionally gone as of #899, replaced by the
`_PalmerPrepared`/`_PalmerRecursion` dataclasses this paragraph's line references
predate), and the spell recursion is a masked n-D kernel -- see
`### Conversion status (#937)` above and ADR-0011 for the design and why `scpdsi`
stays blocked here: its four Wells recursions and per-location duration-factor fits
per cell, plus a `ConvergenceError` path a blocked kernel has nowhere to put, make
it a second version of this same effort rather than an extension of it.

## Real-grid SPI benchmark: v2.4.0 vs main on the Morocco CHIRPS case (#1097)

```bash
# fixture (external, ~33 MB; not committed):
# https://drive.google.com/file/d/1Px9dOCoqY7Ro-22Nl4pPKUOpJEb7AHsD/view
uv run benchmarks/parallel_scaling.py \
  --netcdf mar_cli_chirps3_month1_1981_2024c.nc --scale 6 --cores 1,2,4,8 --repeat 3 \
  --write-output /tmp/spi6_gamma_output.nc
CLIMATE_INDICES_CHIRPS_NC=mar_cli_chirps3_month1_1981_2024c.nc \
  uv run pytest tests/test_numerical_equivalence.py -m validation -k chirps -v --disable-warnings
```

Fixture: Morocco CHIRPS v3 monthly precipitation from
[Ouranosinc/xclim#2091](https://github.com/Ouranosinc/xclim/issues/2091),
34,164,848 bytes, sha256
`19e2f96275233cc88030639e641b2fe09d49937f315b5b1a3e97c819f6d28cea`; `time=528`
(1981-01 to 2024-12), `lat=165`, `lon=244`, units mm. 12,738,528 cells are NaN
(the ocean mask, 59.9%) and 1,428,447 are zero.

Workload: SPI-6, `dist="gamma"`, calibration 1991-2020, `time` as a single chunk
(ADR-0003), zeros replaced with 0.01 mm, land mask taken from the first time
step, 16,134 land cells. Three timed runs per configuration after a warm-up; the
Dask side is `scheduler="processes"` with `chunksize=1` and the spatial chunks
chosen per worker count, so pool start-up is inside every timing. Compute-only
seconds are the minimum of the three samples, with all three in brackets.

| configuration | v2.4.0 `d4ed0fc` | main `873aa035` |
| --- | ---: | ---: |
| eager serial in-memory | 34.347 s [34.347, 34.458, 34.560] | 1.224 s [1.224, 1.271, 1.293] |
| Dask, 1 worker | 35.423 s [35.423, 35.901, 36.161] | 2.546 s [2.546, 2.613, 2.745] |
| Dask, 2 workers | 22.082 s [22.814, 22.082, 22.396] | 2.060 s [2.366, 2.161, 2.060] |
| Dask, 4 workers | 17.453 s [17.453, 17.604, 18.238] | 1.750 s [1.753, 1.765, 1.750] |
| Dask, 8 workers | 11.695 s [12.006, 12.031, 11.695] | 1.901 s [1.997, 1.916, 1.901] |
| NetCDF read | 1.012 s | 1.060 s |
| NetCDF write (float32) | 0.032 s | 0.104 s |

Measured, not inferred:

- The eager in-memory call on `main` is 28.1x faster than on `v2.4.0` on the
  same grid and the same case (34.347 s -> 1.224 s, compute only).
- On `v2.4.0` Dask scales: 3.03x from 1 to 8 workers (35.423 s -> 11.695 s).
  It does not close the gap to `main`: the release's best Dask configuration is
  still 9.6x slower than `main`'s eager call.
- On `main` Dask still does not beat the eager call at this size, now by a wider
  margin than on the 38x87 grid: the best configuration (4 workers) is 1.43x
  slower than eager (1.750 s vs 1.224 s), and 8 workers is slower than 4. The
  within-Dask speedup is 1.45x from 1 to 4 workers; the fresh `processes` pool
  costs about 1.3 s per `compute()`.
- Read and write are outside the compute figures; a full serial workflow at this
  size is read 1.060 s + compute 1.224 s + write 0.104 s on `main`.

Interpretation (not measured): once the serial kernel is fast enough that the
whole 40,260-cell grid is 1.2 s of work, the fixed pool cost is most of what
Dask would do at this size, so the parallel path loses regardless of the worker
count. Worker-count tuning cannot recover it.

Reproduce with the same commands; the results are retained verbatim in
`benchmarks/results/chirps_spi6_gamma_main.txt` and
`benchmarks/results/chirps_spi6_gamma_v2.4.0.txt`.

### Correctness gate

`benchmarks/results/chirps_spi6_equivalence.txt` is the `pytest -m validation -k
chirps` output: SPI-6 `gamma` over the prepared CHIRPS grid matches the per-cell
NumPy API bit for bit, and the Dask-backed grid with `scheduler="processes"`
matches the eager grid bit for bit. No timing above is quoted before that gate.

Pearson is not timed. On this fixture it fails the gate: the eager block is 60%
missing cells and the block-path Pearson result is all-NaN, so no speedup could
be claimed from it. The failure is pinned by an `xfail(strict=True)` case in
`tests/test_numerical_equivalence.py` and reported as
[#1118](https://github.com/monocongo/climate_indices/issues/1118); `gamma` is the
release-to-main control, as #1097 requires.

### Deviations and limits

- `v2.4.0` predates `benchmarks/`, so `parallel_scaling.py` and
  `profile_gridded_spi.py` were copied into that checkout to run it; the numbers
  are the same harness on both sides. The release's `requires-python` is
  `<3.14`, so that side ran Python 3.13.13 against `main`'s 3.14.7. `h5py` was
  installed into the v2.4.0 environment because its `h5netcdf` has no HDF5
  backend otherwise; the environment lines in the two result files record both.
- `main` requires the first spatial cell of a masked grid to hold data (its
  calibration preflight samples the first spatial point only), and the CHIRPS
  grid's first cell is ocean. The harness rolls both spatial axes so cell
  `[0, 0]` is a land cell, keeping every value with its own coordinates; the
  roll offset is printed in the results header. This changes no per-cell value.
- Same-session comparisons only: run-to-run spread is up to ~15% between
  sessions, and the synthetic control rerun for #1097
  (`benchmarks/results/parallel_scaling_1097.txt`) shows SPI at 0.202 s serial
  against the 0.204 s in the committed #928 artifact, but its Dask figures moved
  further (0.683 s at one worker against 0.863 s), so do not compare numbers
  across sessions.
- No cross-PR multiplication: the 28.1x here is this grid, this index, and these
  two commits; it does not compose with #818's per-cell figures, #944's 6x, or
  the xclim `APP`-fit speedup from xclim#2091.

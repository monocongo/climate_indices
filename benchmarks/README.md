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

PET for SPEI is synthetic (a fixed fraction of the precipitation) and per-cell
logging and goodness-of-fit warnings are disabled, so the timings measure the
fitting path rather than the log renderer. The before/after table for #929 is
below.

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

Raw output: `benchmarks/results/serial_before.txt` (pre-conversion) and
`benchmarks/results/parallel_scaling.txt` (this branch).

### The 10x criterion (#893)

**Met for EDDI only; SPI and SPEI fall short.** Against the pre-vectorization
serial canonical path on the reference grid:

- EDDI: 14.717 s -> 0.710 s through the Dask path, ~21x end to end, and 342x for
  the in-process vectorization alone.
- PET: 53x in-process, but 1.5x end to end (1.063 s -> 0.694 s) because the Dask
  path is slower than the serial in-memory call at this size.
- SPI: 4.4x in-process, 1.0x end to end (0.898 s -> 0.895 s).
- SPEI: 3.5x in-process, 0.86x end to end (0.783 s -> 0.911 s).

The SPI/SPEI shortfall is fixed overhead, not a serial-vs-parallel gap in the
kernels: after vectorization each finishes in ~0.2 s, while every `processes`
pool start-up and result transfer costs ~0.7 s. The one-worker row shows it
directly (SPI: 0.924 s for the pool against 0.205 s in memory). Parallelism pays
when the work per block exceeds that fixed cost, and the 3306-cell reference grid
no longer does. The epic's ">11 minutes" reference measures the explicit
lat/lon loops in `notebooks/muitprocess_spi_nclimgrid.ipynb`, which bypass the
adapter: the canonical path was 1.1 s before the conversion, so the 10x criterion
needs either a larger grid than the reference one or a longer-lived executor than
one pool per `compute()` call to be reachable for SPI and SPEI.

## Per-cell invocation inventory (#922)

Static audit of the index-invocation sites in `src/climate_indices/`, as of
`02c9cb38`, for Python-level loops that call an index function once per grid cell
or per time series. Scope is the library and its xarray adapter layer; notebooks
are excluded (see the #921 findings above for why the notebook figure differs).

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
a single chunk on the generic adapter path (`_validate_dask_chunks` at `:1659`
raises before `apply_ufunc` runs); the two PET paths pass
`dask_gufunc_kwargs={"allow_rechunk": True}` (`:2086`, `:2355`) so they can
rechunk a split time dimension. Counts multiply per invocation: each adapter
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

The remaining per-cell sites above are owned by follow-ups:

| remaining site | owner |
| --- | --- |
| `palmer.pdsi`/`palmer.scpdsi` (no adapter layer at all) | #937 |

### Legacy CLI path (per-cell loop present, parallel across workers)

`__main__.py` and `__spi__.py` validate `(lat, lon, time)` or `(time, lat, lon)
(`__main__.py:61`, `__spi__.py:98`), while the shared-array path stores the
lat/lon-first order untransposed and the per-cell loops assume it; the
mismatches that survive validation are tracked in #932. The counts below assume
`(lat, lon, time)`, split along axis 0 (latitude) across a `multiprocessing.Pool`,
with the per-cell loop inside each worker. The loops run in parallel across
processes but are not eliminated, and each worker's per-cell call carries the
same per-cell overhead the #921 profile measured (per-cell `structlog` records
and the per-kernel goodness-of-fit check): the Pool divides wall clock, it does
not reduce total per-cell Python cost.

| site | invocation | loop dimensions | calls |
|---|---|---|---|
| `__main__.py:1289` (`_apply_along_axis`) | `_spi` via `np.apply_along_axis(axis=2)` | `lat x lon`, looped by `np.apply_along_axis` in Python | 3306 per scale x distribution (`:1519-1520`) |
| `__main__.py:1289` (`_apply_along_axis`) | `_pnp` via `np.apply_along_axis(axis=2)` | same | 3306 per scale only (`:1609`, no distribution loop) |
| `__main__.py:1347,1349` (`_apply_along_axis_double`, loop at `:1343,1345`) | `_spei`/`_pet` | `lat x lon` | 3306 |
| `__main__.py:1412` (`_apply_along_axis_palmers`, loop at `:1409,1411`) | `_palmers` -> `palmer.pdsi` | `lat x lon` | 3306, four outputs each |
| `__spi__.py:1021` (`_apply_to_subarray_spi`, loop at `:1004`) | `indices.spi` transform | `lat x lon` | 3306 per scale x distribution |
| `__spi__.py:1105` (`_apply_to_subarray_gamma`, loop at `:1099`) | `compute.gamma_parameters` | `lat x lon` | 3306 per scale x distribution |
| `__spi__.py:1192` (`_apply_to_subarray_pearson`, loop at `:1177`) | `compute.pearson_parameters` | `lat x lon` | 3306 per scale x distribution |

`__spi__.py` is the legacy CLI whose fate is tracked in #919; its three sites
vanish if it is retired rather than vectorized. The `__main__.py` sites duplicate
the adapter path's work on the same kernels, so a baseline measured through the
CLI and a baseline measured through the canonical path are not interchangeable.

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

The per-cell `np.apply_along_axis` sites (`__main__.py:569`, `__main__.py:1015`,
`__spi__.py:250`, `__spi__.py:717`, `__spi__.py:768`) call
`utils.transform_to_366day` / `utils.transform_to_gregorian`, not an index
function, so they are outside the #923 conversion. They are still per-cell Python
calls: `np.apply_along_axis` loops the spatial dimensions in Python, and each
transform loops over years inside that call (`utils.py:396`, `utils.py:515`). The
daily adapter path runs equivalent transforms per cell through
`_compute_with_daily_calendar_plan` (`xarray_adapter.py:512`), driven by
`_DailyCalendarPlan.to_all_leap`/`to_gregorian` (`xarray_adapter.py:306`, `:340`)
and counted inside the wrapper above. Unmeasured overhead on both paths; no
ticket owns it.

### Structural blockers

`palmer.pdsi` and `palmer.scpdsi` take two 1-D series (precipitation and PET)
plus a scalar available water capacity, allocate a per-location `data` dict of
`(n_years, 12)` arrays, and loop over years and months (`palmer.py:231`,
`palmer.py:323`, `palmer.py:365`, `palmer.py:733`). There is no adapter-layer
Palmer call, so today the only per-cell Palmer path is the legacy CLI's
`_apply_along_axis_palmers` (`__main__.py:1409,1411`); converting it needs an n-D
kernel, not an `apply_ufunc` flag. `scpdsi` additionally runs
`_palmer_wells.calculate` per location (`palmer.py:1035`, `palmer.py:1047`), a
per-month backtracking state machine, and duration-factor fits
(`self_calibration.py:342`, `self_calibration.py:394`, `self_calibration.py:449`)
inside the same per-location call; the CLI never invokes `scpdsi`, so those fits
are only reachable through the per-series API, and standard `pdsi` does not
self-calibrate. #899
refactors these internals with unchanged output, not into an n-D kernel. The #923
tasks name SPI, SPEI, and PET (its acceptance criteria name SPI), so Palmer is a
follow-up rather than part of the conversion; the n-D kernel is tracked in #937.

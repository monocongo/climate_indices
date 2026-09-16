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
gridded SPI. A Dask-backed input returns a lazy result from the same per-cell
`apply_ufunc` loop; Dask scheduling and multi-core scaling belong to #927 and
#928.

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
  path. The canonical path measured here is 1.1 s, so the 10x speedup criterion
  in #893 needs a pinned baseline entry point (#929) before it can be evaluated.

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
| `xarray_adapter.py:1717` (`xarray_adapter`, Dask branch) | wrapped `spi`/`spei`/`eddi`/`percentage_of_normal` | `lat x lon`, once per Dask block | 3306 |
| `xarray_adapter.py:1808` (`xarray_adapter`, in-memory branch) | same | `lat x lon` | 3306 |
| `xarray_adapter.py:2084` (`pet_thornthwaite`) | `indices.pet` | `lat x lon` | 3306 |
| `xarray_adapter.py:2353` (`pet_hargreaves`) | `eto.eto_hargreaves` | `lat x lon` | 3306 |

All four pass `vectorize=True` to `xr.apply_ufunc`, which loops the non-core
(spatial) dimensions in Python. On a Dask-backed input `dask="parallelized"`
splits the same 3306 calls across blocks, so the count per full pass is
unchanged and only wall time improves. Counts multiply with the scale and
distribution loops of `typed_public_api.py`: SPI over `--scales 1 2 3 6 9 12 24`
and both distributions is 14 passes, 46,284 per-cell calls.

This path is serial within a process and is the one the #921 profile measured:
3306 calls into the calendar wrapper at `xarray_adapter.py:507` for a single
SPI-1/gamma run.

### Legacy CLI path (per-cell loop present, parallel across workers)

`__main__.py` and `__spi__.py` build shared-memory arrays shaped
`(lat, lon, time)`, split them by latitude across a `multiprocessing.Pool`, and
loop per cell inside each worker. The loops run in parallel across processes but
are not eliminated.

| site | invocation | loop dimensions | calls |
|---|---|---|---|
| `__main__.py:1289` (`_apply_along_axis`) | `_spi`/`_pnp` via `np.apply_along_axis(axis=2)` | `lat x lon` (looped inside numpy) | 3306 per scale x distribution |
| `__main__.py:1343,1345` (`_apply_along_axis_double`) | `_spei`/`_pet` | `lat x lon` | 3306 |
| `__main__.py:1409,1411` (`_apply_along_axis_palmers`) | `_palmers` -> `palmer.pdsi` | `lat x lon` | 3306, four outputs each |
| `__spi__.py:1004,1006` (`_apply_to_subarray_spi`) | `indices.spi` transform | `lat x lon` | 3306 per scale x distribution |
| `__spi__.py:1099,1101` (`_apply_to_subarray_gamma`) | `compute.gamma_parameters` | `lat x lon` | 3306 |
| `__spi__.py:1177,1179` (`_apply_to_subarray_pearson`) | `compute.pearson_parameters` | `lat x lon` | 3306 |

`__spi__.py` is the legacy CLI whose fate is tracked in #919; its three sites
vanish if it is retired rather than vectorized. The `__main__.py` sites duplicate
the adapter path's work on the same kernels, which is why the CLI and the
canonical path are separate baselines in #929.

### Already vectorized (no per-cell invocation site)

- `fire.py:1144` (`_kbdi_xarray`) passes `vectorize=False` and `fire.py:2460`
  (`_hdw_xarray`) omits it: those kernels loop over time (or the level dimension)
  and operate on whole block arrays, so cells are handled by NumPy operations
  rather than a Python call per cell.
- The core kernels are vectorized over cells by construction: `compute.py`,
  `indices.py`, `eto.py`, and `pm_eto.py` take a 1-D series and return a 1-D
  series, and their internal loops are over time steps
  (`compute.py:797`, `compute.py:869`, `indices.py:352`), not over grid cells.
  `indices.pci` is a single-year scalar with no loop at all.
- The per-cell `np.apply_along_axis` calendar transforms
  (`__main__.py:569`, `__main__.py:1015`, `__spi__.py:250`, `__spi__.py:717`,
  `__spi__.py:768`) call `utils.transform_to_366day` /
  `utils.transform_to_gregorian`, not index functions; they are vectorized over
  time and are not part of the #923 conversion.

### Structural blockers

`palmer.pdsi` and `palmer.scpdsi` take one 1-D series plus a scalar available
water capacity, allocate a per-location `data` dict of `(n_years, 12)` arrays, and
loop over years and months (`palmer.py:231`, `palmer.py:323`, `palmer.py:365`,
`palmer.py:733`). The Palmer adapters therefore cannot drop their per-cell call
by changing an `apply_ufunc` flag; they need an n-D kernel.
`self_calibration.py` percentile sweeps (`self_calibration.py:342`, `:394`,
`:449`) run inside that same per-cell call. The #923 acceptance criteria name
SPI, SPEI, and PET, so Palmer is a follow-up rather than part of the conversion.

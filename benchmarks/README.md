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
| cProfile-instrumented | 2.5 s |

Per-cell logging costs ~0.4 s (~35%) of the INFO wall clock. `cProfile` roughly
doubles the wall clock on this call-heavy path, so the profiled report is for
relative attribution, not absolute timing.

Hottest paths by cumulative time (2.493 s profiled total):

| path | calls | cumtime |
|---|---|---|
| `numpy._vectorize_call_with_signature` | 1 | 2.492 s |
| `climate_indices/xarray_adapter.py:507` (`wrapper`, one call per grid cell) | 3306 | 2.483 s |
| `climate_indices/indices.py:417` (`spi`) | 3306 | 2.478 s |
| `climate_indices/compute.py:1085` (`transform_fitted_gamma`) | 3306 | 1.942 s |
| `structlog/stdlib.py:218` (`info`) | 19837 | 1.229 s |
| `climate_indices/compute.py:942` (`gamma_parameters`) | 3306 | 1.015 s |
| `climate_indices/compute.py:779` (`_check_goodness_of_fit_gamma`) | 3306 | 0.473 s |

Hottest paths by self time:

| path | tottime |
|---|---|
| `climate_indices/compute.py:779` (`_check_goodness_of_fit_gamma`) | 0.181 s |
| `structlog/dev.py:296` (console renderer) | 0.146 s |
| `climate_indices/compute.py:743` (`_ks_poor_fit_p_value`) | 0.111 s |
| `scipy/stats/_continuous_distns.py:3612` (`_cdf`) | 0.091 s |
| `structlog/_frames.py:36` (`_find_first_app_frame_and_name`) | 0.049 s |

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

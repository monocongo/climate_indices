# Benchmarks

Reproducible performance measurement for the #893 vectorization/parallelization
epic. Reports here are evidence artifacts, not CI gates.

## Profiling the gridded SPI workflow (#921)

```bash
uv run benchmarks/profile_gridded_spi.py
```

The script builds the reference grid from #893 as deterministic synthetic data
(38x87 cells, 40 years of monthly precipitation, seed 42), compiles the
distribution-fitting kernels on a small warm-up grid so JIT time stays out of the
window, then runs the canonical xarray path — `climate_indices.spi(DataArray)` —
under `cProfile`. The raw report is written to
`benchmarks/results/profile_gridded_spi.txt` and the wall-clock time is printed.

Per-cell info logging is part of the measured workflow (the library defaults to
`CLIMATE_INDICES_LOG_LEVEL=INFO`). To profile without it:

```bash
CLIMATE_INDICES_LOG_LEVEL=WARNING uv run benchmarks/profile_gridded_spi.py
```

### Findings (macOS arm64, Python 3.13, 2026-09)

Reference grid: 3306 cells x 480 months. Wall-clock: 2.4 s at the default log
level, 1.2 s with logging at `WARNING` — about half of the canonical-path
runtime is per-cell structured logging.

Hottest paths by cumulative time (2.412 s total):

| path | calls | cumtime |
|---|---|---|
| `numpy._vectorize_call_with_signature` | 1 | 2.410 s |
| `climate_indices/xarray_adapter.py:507` (`wrapper`, one call per grid cell) | 3306 | 2.400 s |
| `climate_indices/indices.py:417` (`spi`) | 3306 | 2.396 s |
| `climate_indices/compute.py:1085` (`transform_fitted_gamma`) | 3306 | 1.876 s |
| `structlog/stdlib.py:218` (`info`) | 19837 | 1.194 s |
| `climate_indices/compute.py:942` (`gamma_parameters`) | 3306 | 0.983 s |
| `climate_indices/compute.py:779` (`_check_goodness_of_fit_gamma`) | 3306 | 0.457 s |

Hottest paths by self time:

| path | tottime |
|---|---|
| `climate_indices/compute.py:779` (`_check_goodness_of_fit_gamma`) | 0.176 s |
| `structlog/dev.py:296` (console renderer) | 0.143 s |
| `climate_indices/compute.py:743` (`_ks_poor_fit_p_value`) | 0.106 s |
| `scipy/stats/_continuous_distns.py:3612` (`_cdf`) | 0.090 s |
| `structlog/_frames.py:36` (`_find_first_app_frame_and_name`) | 0.048 s |

Interpretation:

- Per-grid-cell invocation is still real, via
  `xr.apply_ufunc(..., vectorize=True)`: 3306 Python calls into
  `xarray_adapter.py:507` at ~0.73 ms each.
- The adapter emits six `structlog` info records per cell (19837 / 3306), and
  their console rendering plus stack-frame inspection account for roughly half
  the runtime. This is a logging-volume cost, not an algorithmic one.
- The numerical work is a per-cell gamma fit with a Kolmogorov-Smirnov
  goodness-of-fit check (`compute.py:942`, `compute.py:779`) plus scipy `cdf`/`ppf`
  transforms — all serial Python/scipy calls per cell.
- `src/climate_indices/` contains no Numba usage; the epic's "already
  Numba-accelerated kernels" premise does not hold for the SPI path.
- The epic's ">11 minutes" reference matches the explicit lat/lon `for` loops in
  `notebooks/muitprocess_spi_nclimgrid.ipynb`, which bypass the canonical adapter
  path. The canonical path measured here is 2.4 s, so the 10x speedup criterion
  in #893 needs a pinned baseline entry point (#929) before it can be evaluated.

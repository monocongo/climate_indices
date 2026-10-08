# Technical Architecture

## Executive Summary

The **climate_indices** library implements a **layered library architecture** optimized for scientific computing on climate data. The architecture separates concerns across five distinct layers: CLI interfaces for batch processing, dual public APIs (legacy numpy + modern xarray), core computation logic, mathematical/statistical algorithms, and infrastructure services.

### Architectural Highlights
- **Dual API Design**: Maintains backward-compatible numpy API while providing modern xarray/Dask integration
- **Layered Separation**: Clean boundaries between user interfaces, computation, and infrastructure
- **Parallelization Strategy**: Multiprocessing for CLI, Dask for xarray workflows
- **Exception Hierarchy**: Structured error handling with context-rich exceptions
- **Test Architecture**: Comprehensive fixture-based testing with property-based validation

### Design Principles
1. **Scientific Correctness**: Implementations strictly follow peer-reviewed methodologies
2. **Backward Compatibility**: Legacy numpy API remains stable across minor versions
3. **Performance**: Optimized for gridded datasets with 10⁶+ cells
4. **Type Safety**: Strict mypy compliance on new code (`typed_public_api.py`)
5. **Observability**: Structured logging with performance metrics

## Technology Stack

### Core Dependencies
| Dependency | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10-3.14 | Language runtime |
| **scipy** | >=1.15.3 | Statistical distributions, numerical optimization |
| **xarray** | >=2025.6.1 | Labeled multi-dimensional arrays, CF metadata |
| **dask** | >=2025.7.0 | Parallel computation, lazy evaluation |
| **h5netcdf** | >=1.6.3 | NetCDF file I/O backend |
| **cftime** | >=1.6.4 | Calendar-aware datetime handling |
| **structlog** | >=24.1.0 | Structured logging with context |

### Development Dependencies
| Tool | Purpose |
|------|---------|
| **pytest** | Test runner with fixtures and parametrization |
| **hypothesis** | Property-based testing for invariants |
| **pytest-benchmark** | Performance regression testing |
| **pytest-cov** | Code coverage reporting |
| **mypy** | Static type checking |
| **ruff** | Linting and code formatting |
| **sphinx** | Documentation generation |

### Build and Deployment
- **Build Backend**: Hatchling (PEP 517 compliant); the optional Rust extension is built with maturin (see [Optional Rust Backend](#optional-rust-backend))
- **Package Manager**: uv (modern resolver, lockfile support)
- **CI/CD**: GitHub Actions (3 workflows: unit tests, releases, benchmarks)
- **Container**: Docker with Python 3.14-slim base image
- **Documentation**: Sphinx with ReadTheDocs hosting

## Layered Architecture Pattern

```
┌────────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                  │
├────────────────────────────────────────────────────────────────────┤
│  __main__.py          │  Full-featured CLI (all indices)          │
├────────────────────────────────────────────────────────────────────┤
│                      Public API Layer                              │
├────────────────────────────────────────────────────────────────────┤
│  typed_public_api.py  │  Strict mypy, xarray wrappers            │
│  xarray_adapter.py    │  CF-compliant xarray interface           │
│  validation.py        │  Shared input validation facade          │
│  indices.py           │  Legacy numpy API (backward compat)       │
├────────────────────────────────────────────────────────────────────┤
│                    Computation Layer                               │
├────────────────────────────────────────────────────────────────────┤
│  compute.py           │  Core algorithms (scaling, fitting, PDF) │
│  palmer.py            │  Palmer drought indices implementation    │
├────────────────────────────────────────────────────────────────────┤
│                 Math/Statistics Layer                              │
├────────────────────────────────────────────────────────────────────┤
│  eto.py               │  PET: Thornthwaite, Hargreaves methods   │
│  lmoments.py          │  L-moments for distribution fitting       │
├────────────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                              │
├────────────────────────────────────────────────────────────────────┤
│  utils.py             │  Utilities, calendar conversions         │
│  logging_config.py    │  Structured logging setup                │
│  exceptions.py        │  Exception hierarchy with context        │
│  performance.py       │  Performance metrics tracking            │
└────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

#### 1. CLI Layer
**Purpose**: Command-line interfaces for batch processing NetCDF datasets.

**Modules**:
- **`__main__.py`**: Full-featured CLI supporting SPI, SPEI, PET, Palmer, and PNP indices
  - Multiprocessing pool for gridded data parallelization
  - NetCDF dimension validation through the validation facade, and coordinate
    conversion
  - Shared memory arrays for worker processes
  - Dataset layout detection (grid, divisions, timeseries)

**Entry Points** (`pyproject.toml`):
```toml
[project.scripts]
climate_indices = "climate_indices.__main__:main"
process_climate_indices = "climate_indices.__main__:main"
```

#### 2. Public API Layer
**Purpose**: User-facing interfaces for programmatic access.

**Modules**:
- **`typed_public_api.py`**: Modern xarray API with strict mypy compliance
  - Type-safe wrappers for SPI and SPEI
  - Enforces keyword-only arguments
  - Full mypy --strict compliance

- **`xarray_adapter.py`**: CF-compliant xarray interface
  - Coordinate validation and alignment
  - CF metadata preservation and generation
  - Dask array support with chunking validation
  - PET computation (Thornthwaite, Hargreaves)

- **`validation.py`**: Public validation facade shared by the CLI, the xarray
  and fire adapters
  - Input type detection (`DataArray`, `ndarray`; `Dataset` is rejected with a
    select-a-variable hint)
  - Dataset-layout classification (grid, divisions, timeseries) and the
    dimension orders each layout accepts
  - Time-dimension, monotonicity, and Dask single-chunk validation

- **`indices.py`**: Legacy numpy API
  - Backward-compatible function signatures
  - Direct numpy array inputs/outputs
  - Distribution enum (`Distribution.gamma`, `Distribution.pearson`)
  - SPI, SPEI, PNP, PET computation

**API Design Decision**: The dual API approach allows:
- **Legacy users**: Continue using numpy arrays without migration
- **Modern users**: Leverage xarray's labeled dimensions, CF metadata, and Dask parallelization
- **Migration path**: xarray API uses numpy implementation internally

#### 3. Computation Layer
**Purpose**: Core mathematical algorithms for climate index calculation.

**Modules**:
- **`compute.py`**: Core computation functions
  - `prepare_scaled()`: Shared flatten/clip/roll-sum/reshape preparation for the fitting-based indices
  - `is_all_missing()`, `reshape_time_major()`, `unfold_time_major()`: single owner of all-missing detection and the time-major block round trip
  - `fit_and_standardize()`: Shared parameter normalization, gamma/Pearson dispatch, and Pearson→gamma fall-back seam for SPI and SPEI
  - `gamma_parameters()`, `pearson_parameters()`: Distribution fitting
  - `transform_fitted_gamma()`, `transform_fitted_pearson()`: CDF transformation
  - `sum_to_scale()`: Optimized sliding window summation
  - `Periodicity` enum: `monthly` (12 steps/year), `daily` (366 steps/year), with a `period_length` property and a `unit()` method

- **`palmer.py`**: Palmer Drought Index family
  - PDSI (Palmer Drought Severity Index)
  - PHDI (Palmer Hydrological Drought Index)
  - PMDI (Palmer Modified Drought Index)
  - ZINDEX (Palmer Z-Index)
  - scPDSI (Self-calibrated Palmer Drought Severity Index), available through the NumPy `palmer.scpdsi()` API

**Key Algorithms**:
1. **SPI/SPEI Computation**:
   ```
   Input: precip (or P-PET)
   ↓
   Scale to an N-step window (sum_to_scale)
   ↓
   Fit distribution (gamma or Pearson Type III) per calendar month/day
   ↓
   Transform to standard normal via CDF (scipy.stats)
   ↓
   Output: SPI/SPEI values
   ```

2. **Distribution Fitting Strategy**:
   - **Gamma**: Method of moments (α, β parameters)
   - **Pearson Type III**: L-moments (location, scale, skew parameters)
   - **Calibration period**: Default 30+ years, user-configurable
   - **Handling zeros**: Probability of zero tracked separately
   - **Fall-back policy**: SPI falls back from a failed Pearson Type III fit, or one that
     loses most of the input's valid values, to gamma; SPEI propagates the failure.
     `compute.fit_and_standardize()` takes this as `fallback_to_gamma`, so the divergence
     is a parameter of one seam rather than a copy of the fit/dispatch branch in each
     index function.

#### 4. Math/Statistics Layer
**Purpose**: Low-level mathematical and statistical functions.

**Modules**:
- **`eto.py`**: Potential Evapotranspiration methods
  - **Thornthwaite (1948)**: Monthly PET from temperature and latitude
    - Heat index computation
    - Day length adjustment based on latitude
  - **Hargreaves (1985)**: Daily PET from temperature range and solar radiation
    - Requires tmin, tmax, and latitude
    - Extraterrestrial radiation calculation

- **`lmoments.py`**: L-moments for robust distribution fitting
  - Implements Hosking (1990) L-moments algorithm
  - Used for Pearson Type III parameter estimation
  - More robust than method of moments for skewed distributions

**Design Note**: This layer has no dependencies on upper layers and could be extracted as standalone utilities.

#### 5. Infrastructure Layer
**Purpose**: Cross-cutting concerns (utilities, logging, error handling).

**Modules**:
- **`exceptions.py`**: Exception hierarchy
  - Base: `ClimateIndicesError` (catch-all for library errors)
  - Computation: `DistributionFittingError`, `InsufficientDataError`, `PearsonFittingError`
  - Validation: `DimensionMismatchError`, `CoordinateValidationError`, `InputTypeError`, `InvalidArgumentError`
  - Warnings: `MissingDataWarning`, `ShortCalibrationWarning`, `GoodnessOfFitWarning`, `InputAlignmentWarning`
  - All exceptions carry context attributes (e.g., `distribution_name`, `input_shape`, `parameters`)

- **`logging_config.py`**: Structured logging configuration
  - `configure_logging()`: Sets up structlog with JSON serialization
  - Console: Human-readable colored output
  - File: JSON-formatted for log aggregators
  - Context binding for tracing

- **`utils.py`**: Utility functions
  - Calendar conversions: `transform_to_366day()`, `transform_to_gregorian()`
  - Calendar planning: `DailyCalendarPlan`
  - Data validation: `is_data_valid()`
  - Array reshaping: `reshape_to_2d()`, `reshape_to_divs_years_months()`
  - Periodicity utilities: `gregorian_length_as_366day()`

- **`performance.py`**: Memory metrics
  - `get_process_memory_mb()`: current process memory usage
  - `check_large_array_memory()`: returns memory metrics when arrays exceed the 1 GB threshold

## Optional Rust Backend

Numerical kernels are being migrated to Rust as an internal acceleration
backend. The public Python API is the compatibility and citation contract:
users import and call `climate_indices` exactly as before, and nothing in the
public signatures or results depends on whether the backend is installed.

```
Cargo.toml                      # Cargo workspace
crates/climate-core/            # pure Rust numerical kernels
crates/climate-py/              # PyO3/NumPy bindings -> climate_indices._native
src/climate_indices/_native.pyi # type stub for the extension
```

- **`climate-core`** is pure numerical Rust. It has no PyO3, NumPy bindings,
  Python exceptions, or CPython assumptions, receives already-validated
  numbers, and returns numbers. It could later serve other Rust consumers, but
  it is not published to crates.io.
- **`climate-py`** is the only crate that knows about Python. It converts
  arrays and errors at the boundary and holds no climate algorithm.
- **Dispatch** lives in the Python module that owns each computation. It
  imports `climate_indices._native` inside `try`/`except ImportError`; without
  the extension every computation runs in pure Python. A runtime error raised
  by the extension propagates and is never silently retried in Python.
- **Reference implementation**: the Python implementations stay alive and
  directly testable. They are the oracle for Rust parity tests until a separate,
  explicit decision retires them.

**Ported kernels.** The gamma fit and transform behind SPI were the first port,
followed by EDDI's empirical ranking and inverse normal, the distribution fits
SPEI adds, Pearson Type III (also used by SPI and the standardized index), the
log-logistic (generalized logistic, GLO), the PNP and PCI numerical blocks, and
the fire-weather recurrences. The gamma and distribution-fit kernels replace the
numerical blocks inside `compute.py` functions; the EDDI, PNP, and PCI kernels
replace the blocks inside the `indices.py` functions that own them, so the
validation, calibration-period resolution, data-quality and goodness-of-fit
warnings, the Pearson-to-gamma fallback, logging, zero placement, support-limit
masks, and output scaling around them still run in Python, and every caller of
those functions (SPI, SPEI and the standardized index, `fit_diagnostics`, the
xarray adapter) uses them:

| Python seam | Rust kernel (`climate-core`) |
|---|---|
| method-of-moments block of `compute.gamma_parameters` | `gamma::gamma_parameters` |
| `scipy.stats.gamma.cdf` and zero-mass mixing in `compute.transform_fitted_gamma` | `gamma::gamma_probabilities` |
| `scipy.stats.norm.ppf` in `compute.transform_fitted_gamma` | `special::norm_ppf` |
| sample L-moments and the Pearson Type III fit of `compute.pearson_parameters` (`lmoments.fit`, `fit_spatial`) | `lmoments::sample_lmoments`, `pearson::pearson_parameters` |
| `scipy.stats.pearson3.cdf` in `compute._pearson_fit` | `pearson::pearson_cdf_block` |
| GLO fit of `compute.loglogistic_parameters` (`lmoments.fit_glo`, `fit_glo_spatial`) | `loglogistic::loglogistic_parameters` |
| GLO probability step of `compute._loglogistic_fit` | `loglogistic::loglogistic_cdf_block` |
| rank count and Tukey plotting position in the per-period loop of `indices.eddi` | `eddi::tukey_probabilities` |
| `indices._hastings_inverse_normal` | `eddi::hastings_inverse_normal` |
| calibration normals and ratios in `indices.percentage_of_normal` | `pnp::pnp_normals`, `pnp::pnp_percentages` |
| monthly reduction and ratio in `indices.pci` | `pci::pci` |

EDDI is non-parametric, so no SciPy special function is ported for it: its
probabilities are a count of the period's climatology values strictly below each
value, converted by the Tukey plotting position. The chunking Python applies
across cells (`_EDDI_RANK_COMPARISON_ELEMENT_BUDGET`) only bounds an intermediate
under the Python path; the Rust kernel walks every column in one pass, which
cannot change a count. The calibration-period resolution, the leading-scale-pad
mask, and the unfolding to the caller's layout stay in Python, as they did.

The special functions are line-by-line ports of the Cephes `igam`, `ndtri`,
`ndtr`, and `lgam` that SciPy 1.17 evaluates, since a generic implementation
would not hold the parity contract in the transformed tails. The L-moment fits
route to Rust only for a block whose values are all NaN or at most `1e100` in
magnitude: an infinity (or an overflowing weighted sum) makes the L-moments NaN,
which the single-series Python fit returns as NaN parameters but the cell-axis
fit marks invalid, so those blocks keep the Python fit. The `climate_indices.lmoments`
log records that the single-series fits write per failed step (an `ERROR` for
invalid L-moments, and a `WARNING` for a step with fewer than four non-NaN
values) are not written by the Rust fit, as they are not by the cell-axis fit;
the failed-fit count and the high-failure-rate warning are unchanged, and a
failure rate at or below that warning's threshold logs one
`distribution_fitting_failures` `WARNING` with the counts on every backend. SciPy's
last-bit results depend on whether its build fuses multiply-adds (aarch64 builds
do, x86-64 wheels do not), and the Pearson fit's `exp(gammaln(a) - gammaln(a + 0.5))`
amplifies one ulp by up to `1e11` for a near-symmetric sample, so the ported
`lgam` and polynomial helpers fuse on aarch64 and only there (`special::mul_add`).
The fire-weather recurrences are the one port that does not replace a block
inside a Python function: each replaces a recurrence's whole day loop, so the
Rust side owns the time axis while Python keeps the calendar, the validation,
and the error surface around it:

| Python seam | Rust kernel (`climate-core`) |
|---|---|
| `fire.ffmc` recurrence | `fire::ffmc` over `recurrence::run` |
| `fire.duff_moisture_code` recurrence | `fire::dmc` over `recurrence::run` |
| `fire.drought_code` recurrence | `fire::dc` over `recurrence::run` |
| `fire.kbdi` recurrence | `fire::kbdi` over `recurrence::run` |

`recurrence::run` ports the shared day loop in `climate_indices._recurrence`:
the ADR-0007 missing-day policy, the ADR-0010 seasonal carry mask, the spin-up
offset, and the recorded-history NaNs are identical in both.
`climate_indices.fire._native` builds each code's kernel from the same
`_CodeInputs` its Python step reads, so the two paths cannot disagree about
which arrays a code consumes, and the KBDI, FFMC, DMC, and DC daily updates are
line-by-line ports of the Python expressions, including the
`np.maximum`/`np.minimum` NaN propagation and NumPy's operation order. The
combined `cffwis()` orchestrator runs each of its three codes through the same
kernels; the components are independent, so running them outside the shared day
loop cannot change a result. Elementwise fire indices (ISI, BUI, FWI, DSR,
Fosberg, HDW, Haines) stay in Python: they are single NumPy expressions with no
recurrence, and a port would not pay for itself.

Dispatch takes the Rust path only for a
plain, aligned float64 `ndarray` whose fit parameters are aligned and one per
calendar step (and cell). Unaligned arrays, other dtypes, and caller-supplied
parameters that vary by year run the Python implementation. A mask is a
missing-value marker, so the seams that read it as one replace it with NaN
before this check, and the prepared plain float64 array may then use Rust: the
gamma transform (`transform_fitted_gamma` and the parameter resolver it calls),
the GLO transform (`transform_fitted_loglogistic`, whose fit and CDF may then run
natively), the sliding sum that prepares a scaled series, and PNP preparation. A
partially masked SPI/SPEI gamma transform therefore matches the same input
passed through `np.ma.filled(values, np.nan)`. An all-masked input returns from
the all-missing short-circuit before that normalization, so SPI and SPEI hand
back the original `MaskedArray`, not the filled plain NaN array. Seams that do
not replace a mask keep their Python implementation when handed one: a direct
`gamma_parameters` or `loglogistic_parameters` call, the Pearson Type III fit
and transform, PCI, whose dispatch checks the original 1-D input and requires a
plain 1-D array, and the fire recurrences, which apply the same guard to every
weather array and to the seed they resume from. A layout the kernel cannot take
unchanged (an empty axis, a non-float64 or unaligned array, or a time-first
array whose spatial axes cannot be viewed as one cell axis without a copy) stays
in Python as well. The kernels take views of the prepared arrays, so a broadcast
month series or season mask is copied once, by the binding, rather than first
materialized in Python; that boundary copy is the native path's own buffer, so
it and the history the kernel builds are not part of the `array_memory_mb` a
recurrence reports. A loaded extension that predates a kernel, or a recurrence
option wider than the binding's integer parameters, also keeps the recurrence on
its Python steps rather than failing at the boundary.
Native dispatch also requires NumPy floating-point errors to be ignored
(`np.errstate(all="ignore")`); warnings, exceptions, callbacks, logging, or
printing keep the Python path. Python 3.14 context-aware warnings conservatively
keep the Python path, as do fits with a column that has no positive value,
whose empty-slice warnings are independent of NumPy error policies. Default NumPy error policies
therefore use Python even when the extension is installed. Direct extension
calls reject unaligned inputs and copy empty arrays without creating Rust views
of caller-owned storage. `tests/test_native_parity.py` (gamma),
`tests/test_native_parity_distributions.py` (Pearson Type III and GLO), and
`tests/test_native_parity_fire.py` (the fire recurrences, which compare the
returned state as well) explicitly ignore floating-point errors and compare the
two paths at `rtol = atol = 1e-10` with matching NaN positions, and the
`python_backend` fixture in
`tests/conftest.py` pins any test to the Python reference.

**Migration policy.** Port expensive numerical kernels, hot loops, and
algorithms that benefit materially from native execution. Keep in Python:
xarray orchestration, CF metadata, validation at the public API boundary,
provenance, the CLI, logging and warnings, file and network I/O, and all other
user-facing behavior. A port reproduces the Python numerics, including their
NaN, zero, and edge-case semantics; it does not improve or change them.

**Building.** Hatchling remains the PEP 517 backend, so the published sdist and
wheel are pure Python, install without a Rust toolchain, and run the Python
implementations. Developers with a Rust toolchain build the extension in place:

```bash
uv run maturin develop --release   # builds src/climate_indices/_native.*.so
uv run python -c "import climate_indices._native"
cargo test --workspace
```

`maturin develop` installs the package in editable mode and copies the compiled
extension into `src/climate_indices/`, where it stays importable after `uv sync`
reinstalls the project; delete the `_native.*` file to return to pure Python.
Publishing binary wheels that include `_native`, and the supported install path
without Rust, are tracked separately (RUST-013).

**Rust in CI.** `.github/workflows/unit-tests-workflow.yml` runs three jobs on
every event the workflow handles (pull request, push to `main`, merge group,
schedule, and manual dispatch), next to the pure-Python legs. Those legs install
no Rust toolchain, so they keep proving the fallback.

- `rust`: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
  -- -D warnings`, and `cargo test --workspace` (`PYO3_PYTHON` points the
  `climate-py` test binary at the project interpreter).
- `test-native`: `maturin develop --release`, then the same core pytest command
  as `test`, on the boundary legs (oldest and newest Python on Linux, newest on
  macOS). It sets `CLIMATE_INDICES_REQUIRE_NATIVE=1`, which makes the native test
  modules raise on a missing extension instead of skipping, so a broken build
  cannot silently drop the parity suite. `tests/test_native_parity.py`,
  `tests/test_native_parity_distributions.py`, and
  `tests/test_native_parity_fire.py` have no other skip. The Python 3.14-only context-aware-warnings routing check lives in
  `tests/test_native_backend.py` and skips on the 3.10 leg.
- `native-wheel`: `maturin build --release` on Linux and macOS at both boundary
  Pythons, plus a Windows smoke build on the newest. Each wheel is installed
  into a fresh venv and imported from outside the checkout, and SPI must reach
  the bundled `gamma_parameters`, `gamma_probabilities`, and `norm_ppf` kernels.
  This validates wheels; it does not publish them.

`rust-toolchain.toml` pins the compiler so a new stable clippy lint cannot fail
`-D warnings` on an unrelated change. The workspace uses edition 2024, so the
minimum supported Rust is 1.85. To bump the pin, change `channel`, run the clippy
command above, and fix any new lints in the same change.

## Source Code Organization

```
climate_indices/
├── src/climate_indices/          # Main package directory
│   ├── __init__.py               # Public API exports
│   ├── __main__.py               # Full-featured CLI entry point
│   ├── typed_public_api.py       # Strict mypy-compliant API
│   ├── xarray_adapter.py         # Modern xarray interface
│   ├── validation.py             # Shared input validation facade
│   ├── indices.py                # Legacy numpy API (STABLE)
│   ├── compute.py                # Core computation algorithms
│   ├── palmer.py                 # Palmer drought indices
│   ├── eto.py                    # PET: Thornthwaite & Hargreaves
│   ├── lmoments.py               # L-moments for Pearson fitting
│   ├── utils.py                  # Utility functions
│   ├── logging_config.py         # Structured logging setup
│   ├── exceptions.py             # Exception hierarchy
│   └── performance.py            # Performance metrics
│
├── tests/                        # Test suite
│   ├── conftest.py               # Shared fixtures (session-scoped)
│   ├── test_indices.py           # Legacy API tests
│   ├── test_xarray_adapter.py    # Modern API tests
│   ├── test_validation.py        # Validation facade tests
│   ├── test_compute.py           # Computation tests
│   ├── test_property_based.py    # Property-based invariant tests
│   ├── test_backward_compat.py   # Backward compatibility suite
│   ├── test_exceptions.py        # Exception handling tests
│   ├── test_observability.py     # Logging behavior and lifecycle tests
│   ├── test_metadata_validation.py # CF metadata validation tests
│   ├── test_benchmark_*.py       # Performance regression tests
│   └── fixture/                  # Test data (numpy arrays, JSON)
│
├── docs/                         # Documentation
│   ├── conf.py                   # Sphinx configuration
│   ├── index.md                  # Main Sphinx doc (ReadTheDocs)
│   ├── reference.md              # API reference (autodoc)
│   ├── release-process.md        # Maintainer release runbook
│   └── *.md                      # AI-readable project docs
│
├── .github/workflows/            # CI/CD pipelines
│   ├── unit-tests-workflow.yml   # Test matrix (Python 3.10-3.14)
│   ├── release.yml               # Automated PyPI releases
│   └── benchmarks.yml            # Performance tracking
│
├── pyproject.toml                # PEP 517 build config + tool settings
├── uv.lock                       # Reproducible dependency lock
├── Dockerfile                    # Container image definition
├── README.md                     # GitHub landing page
└── CONTRIBUTING.md               # Development guidelines
```

### Critical Directories
- **`src/climate_indices/`**: Production code for the core indices and the fire subsystem
- **`tests/`**: Test suite and fixture data
- **`docs/`**: Sphinx + MyST Markdown project documentation
- **`.github/workflows/`**: CI/CD automation (3 workflows)

## Data Flow and Computation Patterns

### SPI/SPEI Computation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  User Input: xarray.DataArray or numpy.ndarray                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Input Validation & Type Detection                              │
│  - Check time dimension exists and is monotonic                 │
│  - Validate calibration period length (≥30 years recommended)   │
│  - Check for excessive NaNs                                     │
│  - Detect input type (numpy vs xarray vs Dask)                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Coordinate Alignment (xarray only)                             │
│  - Align precipitation and PET on time coordinate (SPEI)        │
│  - Warn if alignment drops time steps                           │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Temporal Scaling (compute.prepare_scaled)                      │
│  - Rolling sum over N months/days                               │
│  - Handles NaN propagation                                      │
│  - Output: scaled_values (same shape as input)                  │
│  - Except an all-missing input: returned unreshaped, so a       │
│    2-D all-missing input comes back 1-D                         │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Distribution Fitting (per calendar month/day)                  │
│  - Gamma: alpha, beta parameters (method of moments)            │
│  - Pearson: loc, scale, skew parameters (L-moments)             │
│  - Fit on calibration period only                               │
│  - Track probability of zero separately                         │
│  - One seam (compute.fit_and_standardize) normalizes fitting    │
│    parameters and dispatches on the distribution                │
│  - SPI falls back from a failed Pearson fit to gamma, SPEI      │
│    propagates the failure (fallback_to_gamma)                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  CDF Transformation                                             │
│  - Apply fitted CDF to scaled values                            │
│  - Transform to uniform [0,1] distribution                      │
│  - Uses scipy.stats.gamma or scipy.stats.pearson3              │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Inverse Normal Transformation                                  │
│  - Apply scipy.stats.norm.ppf() to CDF values                   │
│  - Output: SPI/SPEI values (standard normal distribution)       │
│  - Handle edge cases (CDF=0 → -3.09, CDF=1 → 3.09)             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output Formatting                                              │
│  - xarray: Preserve input structure + add CF metadata           │
│  - numpy: Return array with same shape as input                │
│  - Dask: Return lazy Dask array (compute on demand)            │
└─────────────────────────────────────────────────────────────────┘
```

### Parallelization Strategies

#### 1. CLI Multiprocessing (\_\_main\_\_.py)
```python
# Splits data across lat/lon dimensions
# Each worker processes a spatial subset
# Shared memory arrays for inputs/outputs
# Pool size: CPU count - 1
with multiprocessing.Pool(processes=num_workers) as pool:
    pool.map(_apply_along_axis, chunk_params)
```

#### 2. Dask Lazy Evaluation (xarray_adapter.py)
```python
# Dask-backed xarray processing
# Time dimension: single chunk (required)
# Spatial dimensions: chunked (e.g., 50x50 cells)
# Computation triggered by .compute() or .load()
result_da = xr.apply_ufunc(
    spi_computation_fn,
    precip_da,
    dask='parallelized',
    output_dtypes=[float]
)
```

## Testing Architecture

### Test Organization (66 Test Files)
```
tests/
├── conftest.py                      # Session-scoped fixtures
│
├── Core Functionality Tests
│   ├── test_indices.py              # Legacy numpy API tests
│   ├── test_compute.py              # Core algorithm tests
│   ├── test_xarray_adapter.py       # Modern API tests (EXPANDED)
│   ├── test_typed_public_api.py     # Strict typing tests (NEW)
│   └── test_eto.py                  # PET computation tests
│
├── Quality Assurance Tests
│   ├── test_backward_compat.py      # API stability tests
│   ├── test_xarray_equivalence.py   # numpy ↔ xarray parity
│   ├── test_property_based.py       # Hypothesis invariant tests
│   └── test_type_checking.py        # mypy runtime validation
│
├── Validation and Error Handling
│   ├── test_exceptions.py           # Exception hierarchy tests
│   ├── test_input_validation.py     # Input validation tests
│   ├── test_validation.py           # Validation facade tests
│   ├── test_metadata_validation.py  # CF metadata tests
│   ├── test_computation_errors.py   # Error condition tests
│   ├── test_data_quality_warnings.py # Warning behavior tests
│   └── test_input_type_detection.py # Type detection tests
│
├── Observability Tests
│   ├── test_observability.py        # Logging config, lifecycle events
│   └── test_performance_metrics.py  # Performance tracking tests
│
├── Performance Tests (Benchmarks)
│   ├── test_benchmark_overhead.py   # xarray vs numpy overhead
│   ├── test_benchmark_chunked.py    # Dask chunking strategies
│   ├── test_benchmark_fire.py       # Fire-index throughput, sizing, and guards
│   └── test_benchmark_memory.py     # Memory usage profiling
│
└── Regression Tests
    ├── test_palmer.py               # Palmer indices regression
    ├── test_utils.py                # Utility function tests
    └── test_zero_precipitation_fix.py # Specific bug fix test
```

### Test Fixtures (conftest.py)
**Session-scoped fixtures** for performance:
- **Numpy arrays**: `precips_mm_monthly`, `temps_celsius`, `pet_thornthwaite_mm`
- **xarray DataArrays**: `sample_monthly_precip_da`, `gridded_monthly_precip_3d`, `dask_monthly_precip_1d`
- **Edge case fixtures**: `zero_inflated_precip_da`, `leading_nan_block_da`, `non_monotonic_time_da`
- **Benchmark fixtures**: `bench_monthly_precip_np`, `bench_monthly_precip_da`, `bench_gridded_precip_da`
- **Constants**: `_CALIBRATION_YEAR_START_MONTHLY`, `_DATA_YEAR_START_MONTHLY`, `_LATITUDE_DEGREES`

### Test Coverage Targets
- **Overall**: >90% line coverage
- **Critical modules**: `compute.py`, `indices.py`, `xarray_adapter.py` → 100%
- **Exception paths**: All custom exceptions tested with context attributes
- **Property-based**: Mathematical invariants (e.g., SPI mean ≈ 0, std ≈ 1)

### Running Tests
```bash
# All tests (excluding benchmarks)
uv run pytest

# Include benchmarks
uv run pytest -m benchmark

# With coverage
uv run pytest --cov=src --cov-report=html

# Specific test file
uv run pytest tests/test_xarray_adapter.py -v

# Property-based tests only
uv run pytest tests/test_property_based.py
```

## Deployment and CI/CD

### GitHub Actions Workflows

#### 1. unit-tests-workflow.yml
**Trigger**: Pull request and push to main, merge group, weekly schedule, manual dispatch
```yaml
Matrix (core suite):
  test (every event):
    - ubuntu-latest: Python 3.10, 3.11, 3.12, 3.13, 3.14
    - macos-latest: Python 3.14
  test-full (everything except pull requests):
    - macos-latest: Python 3.10
Steps:
  1. Checkout code
  2. Setup Python + uv
  3. uv sync --locked --dev
  4. Run pytest -n auto
Rust jobs (every event; see "Rust in CI" under Optional Rust Backend):
  rust, test-native, native-wheel
```

#### 2. release.yml
**Trigger**: Git tag push (vX.Y.Z)
```yaml
Steps:
  1. Checkout code
  2. Build sdist and wheel (hatchling)
  3. Run twine package checks
  4. Publish to PyPI (trusted publishing via OIDC)
```

#### 3. benchmarks.yml
**Trigger**: Pull request to `main`, manual dispatch
```yaml
Steps:
  1. Checkout code
  2. Setup Python + uv
  3. uv sync --group dev
  4. Run pytest -m benchmark --benchmark-enable --benchmark-json
  5. Upload benchmark artifact
  6. Compare against baseline when available
```

### Docker Container

**Base Image**: `python:3.14-slim`
**Build Strategy**: Multi-stage (builder + production)

```dockerfile
# Builder stage: Install dependencies
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Production stage: Copy venv + source
FROM python:3.14-slim
RUN apt-get update && apt-get install -y \
    libhdf5-dev libnetcdf-dev
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
USER climate  # Non-root user
ENTRYPOINT ["python", "-m", "climate_indices"]
```

**Usage**:
```bash
docker build -t climate_indices:X.Y.Z .
docker run -v $(pwd)/data:/data climate_indices:X.Y.Z \
    --index spi --scales 6 --netcdf_precip /data/precip.nc \
    --var_name_precip prcp --output_file_base /data/spi
```

### PyPI Distribution

**Package Name**: `climate_indices`
**Installation**: `pip install climate_indices` or `uv pip install climate_indices`
**Artifacts**:
- **Source distribution** (`climate_indices-X.Y.Z.tar.gz`)
- **Wheel** (`climate_indices-X.Y.Z-py3-none-any.whl`)

**Excludes from package** (`pyproject.toml`):
- `tests/`, `docs/`, `notebooks/`, `assets/`, `.github/`, `.venv/`, cache directories

## Key Design Decisions and Trade-offs

### 1. Dual API (numpy vs xarray)
**Decision**: Maintain both numpy and xarray APIs.

**Rationale**:
- **Numpy**: Minimal dependencies, direct array manipulation, backward compatibility
- **xarray**: Labeled dimensions, CF metadata, Dask integration, modern workflow

**Trade-off**: Code duplication risk mitigated by having xarray API call numpy implementation internally.

See [ADR-0001](adr/0001-dual-numpy-xarray-api.md).

### 2. Multiprocessing vs Dask
**Decision**: Use multiprocessing in CLI, Dask in xarray API.

**Rationale**:
- **Multiprocessing**: Predictable memory usage, no Dask dependency for CLI users
- **Dask**: Lazy evaluation, better integration with xarray ecosystem, dynamic scheduling

**Trade-off**: Separate parallelization logic in CLI and xarray layers.

See [ADR-0002](adr/0002-multiprocessing-cli-dask-xarray.md).

### 3. Time Dimension Chunking Constraint

**Decision**: For fitting and stateful indices, Dask arrays MUST have time as a single chunk. The two PET entry points are the exception: they accept a split `time` and rechunk it internally, because their kernels are period-based arithmetic rather than fits.

**Rationale**: Climate indices require access to full time series for distribution fitting.

**Enforcement**: `validation.validate_dask_chunks()` validates chunking on the adapter path and raises `CoordinateValidationError` if violated; the PET entry points skip it.

See [ADR-0003](adr/0003-dask-time-dimension-single-chunk.md) for the full rationale.

### 4. Exception Hierarchy with Context
**Decision**: Custom exceptions with context attributes instead of plain `ValueError`.

**Rationale**: AI agents and users need structured error information for debugging.

**Example**:
```python
raise DistributionFittingError(
    "Gamma fitting failed",
    distribution_name="gamma",
    input_shape=(480, 5, 6),
    parameters={"alpha": "NaN", "beta": "NaN"},
    suggestion="Try Pearson Type III distribution"
)
```

### 5. Property-Based Testing
**Decision**: Use Hypothesis for mathematical invariant testing.

**Rationale**: Traditional unit tests miss edge cases; property-based tests generate adversarial inputs.

**Example Properties**:
- SPI output has mean ≈ 0, standard deviation ≈ 1 over calibration period
- SPI is monotonic with respect to input precipitation
- PET is always non-negative

## Performance Considerations

### Bottlenecks
1. **Distribution fitting**: scipy.stats fitting functions (CPU-bound)
2. **CDF transformation**: scipy.stats.cdf() calls (CPU-bound)
3. **Temporal scaling**: Rolling sum over large arrays (memory-bound)
4. **I/O**: NetCDF reading for large gridded datasets (I/O-bound)

### Optimization Strategies
1. **Vectorization**: Numpy broadcasting instead of loops
2. **Shared memory**: CLI uses multiprocessing.Array for zero-copy
3. **Lazy evaluation**: Dask defers computation until .compute()
4. **Chunking**: Spatial chunks in Dask, single time chunk for fitting and stateful indices
5. **Caching**: Distribution fitting parameters can be fitted once and passed
   back in through the `fitting_params` argument of `indices.spi()`

### Benchmark Results (Typical)
| Operation | Input Size | Execution Time | Memory |
|-----------|-----------|----------------|--------|
| SPI-6 (numpy, 1D) | 480 months | ~50 ms | <10 MB |
| SPI-6 (xarray, 1D) | 480 months | ~60 ms | <15 MB |
| SPI-6 (xarray, 3D) | 480×20×20 | ~2 sec | ~100 MB |
| SPI-6 (Dask, 3D) | 480×100×100 | ~20 sec | ~500 MB |

## Security Considerations

### Input Validation
- **NetCDF files**: Dimension checks, coordinate validation
- **User inputs**: Scale range [1-72] monthly / [1-2196] daily, year validation
- **Path sanitization**: No path traversal in CLI file arguments

### Dependency Security
- **uv lock**: Pinned dependencies with checksums
- **GitHub Actions**: Pinned action versions with SHA hashes
- **Container**: Non-root user execution

### No Network Communication
Library has no network dependencies; all data is file-based.

---

**Next Steps**: See [development-guide.md](./development-guide.md) for setup instructions and [deployment-guide.md](./deployment-guide.md) for CI/CD details; the generated API reference is published in the [Sphinx reference page](./reference.md).

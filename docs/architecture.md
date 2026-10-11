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
log-logistic (generalized logistic, GLO), the PNP and PCI numerical blocks, the
fire-weather recurrences, the flood family, and the Palmer family's water
balance, Z-index, and spell recursions. The gamma and distribution-fit kernels
replace the
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
| the year loop of `eto.eto_thornthwaite` | `eto::thornthwaite` |
| the day loop of `eto.eto_hargreaves` | `eto::hargreaves` |
| FAO-56 Eq 6 in `pm_eto.pm_eto` | `pm_eto::pm_eto` |
| the intermediate chain of `pm_eto.penman_monteith_eto` | `pm_eto::penman_monteith_eto` |
| sample L-moments and the Pearson Type III fit of `compute.pearson_parameters` (`lmoments.fit`, `fit_spatial`) | `lmoments::sample_lmoments`, `pearson::pearson_parameters` |
| `scipy.stats.pearson3.cdf` in `compute._pearson_fit` | `pearson::pearson_cdf_block` |
| GLO fit of `compute.loglogistic_parameters` (`lmoments.fit_glo`, `fit_glo_spatial`) | `loglogistic::loglogistic_parameters` |
| GLO probability step of `compute._loglogistic_fit` | `loglogistic::loglogistic_cdf_block` |
| rank count and Tukey plotting position in the per-period loop of `indices.eddi` | `eddi::tukey_probabilities` |
| `indices._hastings_inverse_normal` | `eddi::hastings_inverse_normal` |
| calibration normals and ratios in `indices.percentage_of_normal` | `pnp::pnp_normals`, `pnp::pnp_percentages` |
| monthly reduction and ratio in `indices.pci` | `pci::pci` |

The PET entry points follow the same dispatch rule as the gamma kernels. The
public FAO-56 helper functions stay Python callables; `pm_eto.penman_monteith_eto`
resolves the humidity and radiation pathways, and checks the wind measurement
height and the `rh_min`-without-`rh_max` case, before the kernel is reached, so
the Python and Rust paths raise the same error in the same order. `eto` also
keeps a Thornthwaite block with an all-NaN month column in Python, since only the
Python path reports `np.nanmean`'s empty-slice warning. Each kernel copies every
operand it reads before it releases the GIL, so the native route holds the caller's
arrays, flattened real-array inputs, and those copies: a bounded multiple of the
request. Penman-Monteith passes constants as scalars, caches station/astronomy
terms on exact input bits, and evaluates Eq 6 without intermediate arrays.
Thornthwaite/Hargreaves still pass constant latitudes as zero-stride views, and
Hargreaves reports the bytes it copies beside the arrays it is handed, so the
logged memory model covers the route the dispatch selected. Their measured effect on
three representative inputs is in `benchmarks/README.md`; RUST-011 owns whether
each kernel is worth its conversion overhead.

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

The Palmer family ports its loops and state machines. Dispatch for every stage
lives in `palmer.py`, which owns `pdsi()` and `scpdsi()`; the private
`_palmer_pdi`, `_palmer_wells`, and `self_calibration` functions stay pure
Python and remain the oracle when called directly:

| Python seam | Rust kernel (`climate-core`) |
|---|---|
| `palmer._calc_water_balances` (with `_calc_potential_loss`, `_calc_recharge`) | `palmer::water_balance` |
| `palmer._calc_k_prime_and_dbar` (PDSI K factors and scPDSI K-prime) | `palmer_zindex::k_prime_and_dbar` |
| `palmer._calc_raw_zindex` (with `_calc_cafec_zindex`) | `palmer_zindex::raw_zindex` |
| `_palmer_pdi.calculate`, from `_calculate_pdsi_prepared` | `palmer_pdi::calculate` |
| `self_calibration.duration_factors`, from `_calculate_scpdsi_prepared` | `self_calibration::duration_factors` |
| `_palmer_wells.calculate`, from `_calculate_scpdsi_prepared` | `palmer_wells::calculate` |

The CAFEC ratios (`_calc_cafec_coefficients`), the T ratio
(`_calc_zindex_factors`), the K-factor normalization in `_calc_kfactors`, and the
scPDSI percentile rescaling (`nan_safe_percentile`, `_rescale_scpdsi_zindex`) stay
in Python for the fire indices' reason: each is one NumPy expression. So do the
`DurationFactors`/`PdiDurationFactors` validation, the scPDSI K-prime finiteness
check, masks, fully-missing-cell NaNs, and logging; the Wells kernel takes the
recurrence coefficients Python derived. The PDI state machine Python vectorizes
across cells partitions every cell into exactly one branch each month, so the
kernel runs it one cell at a time with the same branch order and exact-zero
comparisons. A scalar AWC goes to Rust only when it is a Python `int`/`float` or
an `np.float64` (other scalars keep their NumPy type promotion in Python), and an
array AWC only when it is one plain float64 value per cell. An infinite Z value
and a calibration Z series too short for the longest rolling window keep the
Python path, which raises its own error; the kernels' abatement and least-squares
failures raise `_native.NoConvergenceError`, which `palmer.py` re-raises as the
`ConvergenceError` the Python path raises. The native water balance releases its
unused Python output placeholders before allocating Rust outputs; its inputs are
still copied before releasing the GIL. The K-prime and raw Z-index stages borrow
the CAFEC arrays with the GIL held, avoiding five full-record copies per stage.

The flood family ports the computation behind each NumPy entry point in
`climate_indices.flood`, with the Antecedent Precipitation Index running on the
same `recurrence::run` driver as the fire codes rather than a second recurrence
interface:

| Python seam | Rust kernel (`climate-core`) |
|---|---|
| `scipy.ndimage.correlate1d` window of `flood.effective_precipitation` | `flood::effective_precipitation` |
| calendar-day standardization of `flood.edi` | `flood::edi` |
| annual maxima and standardization of `flood.flood_index` | `flood::flood_index` |
| `flood.antecedent_precipitation_index` recurrence | `flood::antecedent_precipitation_index` over `recurrence::run` |

The effective-precipitation kernel reproduces `correlate1d`'s general loop
(the newest day's term first, then the window from its oldest day) and its
NaN propagation, so a window holding a missing day is NaN. The EDI and Flood
Index calibration sums follow NumPy's axis-0 reduction, which adds the years
sequentially for a block of several columns and pairwise for a single one (a
1-D Flood Index sample); both share the population SD and the `8 * eps * |mean|`
rounding guard. `climate_indices.flood._native` hands each kernel the blocks and
the Calibration Period rows the Python modules resolved; validation, the
all-leap layout, and the xarray and CLI paths stay in Python, and the xarray
adapters reach the kernels through the same NumPy entry points. An API decay
constant whose type would promote the Python step beyond float64 (an extended
`np.longdouble`) keeps the Python recurrence.

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
passed through `np.ma.filled(values, np.nan)`. Fully masked SPI/SPEI inputs
short-circuit before fitting or native calls, returning a `MaskedArray`, not a
filled plain NaN array. Existing preparation still applies: SPI flattens a 2-D
series, and SPEI forms its water-balance array, so input object identity is not
a general guarantee. A direct `transform_fitted_gamma` call returns the fully
masked input unchanged. The gamma parameter resolver fills even a
full mask with NaN, so a fully masked fit returns plain NaN parameters without
calling a kernel; supplied `alpha`/`beta` come back unchanged, and only the
computed `prob_zero` is NaN. A direct `gamma_parameters` call also returns plain
NaN parameters for a fully masked input, but retains Python's mask semantics for
a partial mask. Seams that do
not replace a mask keep their Python implementation when handed one: a direct
`gamma_parameters` or `loglogistic_parameters` call, the Pearson Type III fit
and transform, PCI, whose dispatch checks the original 1-D input and requires a
plain 1-D array, and the fire recurrences, which apply the same guard to every
weather array and to the seed they resume from, as the flood kernels do to their
prepared series and the API seed. A layout the kernel cannot take unchanged (an
empty axis, a non-float64 or unaligned array, or, for the fire recurrences, a
time-first array whose spatial axes cannot be viewed as one cell axis without a
copy) stays in Python as well. The kernels take views of the prepared arrays, so
a broadcast month series or season mask is copied once, by the binding, rather
than first materialized in Python; that boundary copy is the native path's own
buffer and is not part of the `array_memory_mb` a recurrence reports, which
counts the one recorded history the kernel returns. The flood kernels differ in
one respect: a time-last (transposed) array reshapes to a view, and a layout
that cannot merge without a copy, such as a regional slice of a larger grid or a
Fortran-ordered array, is copied once in Python and still runs natively, because
the Python path is roughly
three times slower than the kernel plus that copy (365-day PE over 400 cells).
The public flood entry points already hand the dispatch a C-contiguous array,
since their validation copies, so the policy matters to direct callers of
`flood._native`. A loaded extension that predates a kernel,
or a recurrence option wider than the binding's integer parameters, also keeps
the recurrence on its Python steps rather than failing at the boundary.
Native dispatch also requires NumPy floating-point errors to be ignored
(`np.errstate(all="ignore")`); policies that warn, raise, call, log, or print keep
the Python path. A warning filter that promotes `RuntimeWarning` (or a superclass)
to an exception also keeps Python, even under `all="ignore"`. Python 3.14
context-aware warnings conservatively keep the Python path. Default NumPy error
policies therefore use Python even when the extension is installed.

For gamma calibration blocks, the contract is:

- For a plain (or NaN-normalized) calibration block, if any column has no
  positive value after zero replacement (only NaN, zero, or negative values),
  the **whole fit** stays in Python. Its `np.nanmean` emits
  `RuntimeWarning: Mean of empty slice` independently of `np.errstate`, so that
  warning and its promotion to an exception are preserved. A block passed as a
  `MaskedArray` is not normalized here: its masked reductions warn nothing and
  return masked parameters. Entirely missing inputs still return NaN parameters
  before fitting, without this warning.
- Negatives are not removed or newly rejected. A block containing negatives may
  use the native fit only when every column also has a positive value and all
  floating-point errors are ignored. Under `invalid="warn"` or `invalid="raise"`,
  Python preserves `RuntimeWarning: invalid value encountered in log` or
  `FloatingPointError`, respectively.
- Constant and single-positive-value columns may use Rust under `all="ignore"`;
  their degenerate parameters and their transform results match Python.
  A Python fit does not prohibit later eligible native CDF/inverse-normal calls.
  Return values and `climate_indices` warnings match on both backends.

These rules retain existing behavior; [ADR-0017](adr/0017-rust-core-acceleration-backend.md)
records the decision. Direct extension
calls reject unaligned inputs and copy empty arrays without creating Rust views
of caller-owned storage. `tests/test_native_parity.py` (gamma),
`tests/test_native_parity_distributions.py` (Pearson Type III and GLO),
`tests/test_native_parity_fire.py` (the fire recurrences, which compare the
returned state as well), `tests/test_native_parity_palmer.py` (the Palmer
family, which also requires the backtracking's sign pattern to match exactly),
and `tests/test_native_parity_flood.py` (the flood family, including the
returned API state and a resumed run that is bitwise a single pass) explicitly
ignore floating-point errors and compare the
two paths at `rtol = atol = 1e-10` with matching NaN positions, and the
`python_backend` fixture in
`tests/conftest.py` pins any test to the Python reference. The
consolidated suite in `tests/test_native_parity_registry.py` drives the same
comparison from one registry
(`tests/parity_registry.py`: Python entry point, dispatch module, expected
kernels, input family, tolerance), adds Hypothesis draws over lengths, NaN
patterns, zero runs, extreme magnitudes, and spatial shapes, after each entry's
input derivation. Palmer entries replace gaps and clip precipitation to [5, 300],
so their property cases cover lengths and bounded precipitation variation, not
gaps, zero runs, or extreme magnitudes. The suite also asserts the documented
dispatch decision for the input kinds below.
`tests/test_native_e2e_parity.py` extends the same comparison to the surfaces
that orchestrate the kernels: the xarray adapter, threaded and distributed Dask,
the CLI, and `fit_diagnostics`.

### Dispatch routing

Which path an input takes, and why. Every row is asserted by
`tests/test_native_parity_registry.py` against the documented behavior:

| Case | Path | Why |
|---|---|---|
| `plain_daily_series` | Rust | a plain, contiguous float64 ndarray is the kernel's own input|
| `strided` | Rust | a non-contiguous float64 rainfall series reaches the PCI dispatch guard unchanged, then is copied before the GIL is released |
| `float32` | Python | NumPy fits a float32 series in float32, so the fit stays where it was |
| `masked` | Python | a masked array is not a plain float64 ndarray |
| `year_varying_parameters` | Python CDF | only a caller can pass per-step parameters; the kernel takes one per calendar step |
| `overflowing_lmoment_block` | Python fit | an infinity makes the L-moments NaN, which the two fits report differently |
| `oversized_lmoment_block` | Python fit | a finite value above `1e100` exceeds the native L-moment safety bound |

Pearson's single-series goodness-of-fit check also batches sorted calibration
CDFs and KS D statistics in `pearson::pearson_ks_statistics`. Python keeps the
validation, critical values, candidate exact p-values, warnings, and logging.
Only a D statistic below the critical value by the full parity envelope
(`1e-10 * (1 + abs(critical))`, plus dtype epsilon) skips the SciPy check;
candidates and near-boundary statistics use the original oracle decision.
Spatial goodness-of-fit remains Python. Extension errors propagate before the
reference check's exception handler and are never retried.

### Parity tolerance

The contract is `rtol = atol = 1e-10` with matching NaN positions, per kernel,
and `scripts/native_parity_maxima.py` measures what each entry actually deviates
by on its family's fixed sample; the Hypothesis draws in the suite are held to
the same tolerance but are not part of the table. The `test-native` job records
the table in its summary once the consolidated suite passes, so the Linux x86-64
leg and a developer's machine both report their own numbers; the values below
are measured on macOS arm64
(`uv run python scripts/native_parity_maxima.py`, extension built):

| Entry | Kernels | Max absolute | Max relative |
| --- | --- | --- | --- |
| `eddi` | `hastings_inverse_normal`, `tukey_probabilities` | 0.000e+00 | 0.000e+00 |
| `eddi_spatial_block` | `hastings_inverse_normal`, `tukey_probabilities` | 0.000e+00 | 0.000e+00 |
| `fire_drought_code` | `drought_code` | 0.000e+00 | 0.000e+00 |
| `fire_duff_moisture_code` | `duff_moisture_code` | 0.000e+00 | 0.000e+00 |
| `fire_ffmc` | `ffmc` | 0.000e+00 | 0.000e+00 |
| `fire_kbdi` | `kbdi` | 0.000e+00 | 0.000e+00 |
| `fit_diagnostics` | `gamma_parameters` | 0.000e+00 | 0.000e+00 |
| `flood_api` | `antecedent_precipitation_index` | 0.000e+00 | 0.000e+00 |
| `flood_edi` | `edi`, `effective_precipitation` | 8.527e-14 | 7.883e-15 |
| `flood_flood_index` | `effective_precipitation`, `flood_index` | 2.220e-16 | 1.096e-15 |
| `flood_pe` | `effective_precipitation` | 4.547e-13 | 2.781e-16 |
| `hargreaves` | `hargreaves` | 4.441e-16 | 1.975e-16 |
| `palmer_pdsi` | `palmer_k_prime`, `palmer_pdi`, `palmer_raw_zindex`, `palmer_water_balance` | 0.000e+00 | 0.000e+00 |
| `palmer_scpdsi` | `palmer_k_prime`, `palmer_raw_zindex`, `palmer_water_balance`, `palmer_wells`, `scpdsi_duration_factors` | 0.000e+00 | 0.000e+00 |
| `pci` | `pci` | 0.000e+00 | 0.000e+00 |
| `pearson_ks_statistics` | `pearson_ks_statistics` | 3.331e-16 | 1.830e-15 |
| `penman_monteith` | `fao56_eto` | 8.882e-16 | 4.337e-16 |
| `percentage_of_normal` | `pnp_normals`, `pnp_percentages` | 0.000e+00 | 0.000e+00 |
| `pm_eto_intermediates` | `pm_eto` | 0.000e+00 | 0.000e+00 |
| `spei_gamma` | `gamma_parameters`, `gamma_probabilities`, `norm_ppf` | 2.220e-16 | 3.457e-16 |
| `spei_loglogistic` | `loglogistic_cdf`, `loglogistic_parameters` | 0.000e+00 | 0.000e+00 |
| `spi_gamma` | `gamma_parameters`, `gamma_probabilities`, `norm_ppf` | 1.998e-15 | 4.765e-14 |
| `spi_gamma_mean_zero` | `gamma_parameters`, `gamma_probabilities`, `norm_ppf` | 1.998e-15 | 4.765e-14 |
| `spi_gamma_spatial_block` | `gamma_parameters`, `gamma_probabilities`, `norm_ppf` | 7.105e-15 | 1.868e-13 |
| `spi_pearson` | `pearson_cdf`, `pearson_ks_statistics`, `pearson_parameters` | 8.604e-15 | 6.822e-14 |
| `thornthwaite` | `thornthwaite` | 2.842e-14 | 2.208e-16 |

The largest measured absolute deviation is 4.547e-13 (`flood_pe`), more than two
orders of magnitude inside `atol`. The arm64 FMA behavior described above is why the
same entry can report a smaller deviation on an x86-64 leg, and why a deviation
recorded in CI is the evidence for the platform it ran on rather than a new
tolerance.

**Migration policy.** Port expensive numerical kernels, hot loops, and
algorithms that benefit materially from native execution. Keep in Python:
xarray orchestration, CF metadata, validation at the public API boundary,
provenance, the CLI, logging and warnings, file and network I/O, and all other
user-facing behavior. A port reproduces the Python numerics, including their
NaN, zero, and edge-case semantics; it does not improve or change them.

**Building.** Hatchling remains the PEP 517 backend, so the published sdist and
`py3-none-any` wheel are pure Python, install without a Rust toolchain, and run the
Python implementations. Developers with a Rust toolchain build the extension in place:

```bash
uv run maturin develop --release   # builds src/climate_indices/_native.*.so
uv run python -c "import climate_indices._native"
cargo test --workspace
```

`maturin develop` installs the package in editable mode and copies the compiled
extension into `src/climate_indices/`, where it stays importable after `uv sync`
reinstalls the project; delete the `_native.*` file to return to pure Python.

**Packaging.** maturin also builds the published binary wheels, and they ship in
the same release as the sdist and the pure wheel. They are abi3 (`abi3-py310`), so
one wheel per platform validates on every supported Python, and there are five:
manylinux_2_28 `x86_64` and `aarch64`, macOS arm64 and `x86_64`, and Windows
`x86_64`. pip prefers a matching platform wheel over `py3-none-any`; a platform
with neither installs the pure wheel and runs the Python implementations, which is
the supported install path without Rust — installing from source needs no
toolchain either, because hatchling never invokes `cargo`. musllinux is not
published. Which wheel is installed is visible from
`python -c "import climate_indices._native"`, and any `ImportError` from that
import falls back to Python rather than failing. The full comparison of the
options and the failure modes are in
[ADR-0018](adr/0018-optional-rust-packaging.md).

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
  `tests/test_native_parity_distributions.py`,
  `tests/test_native_parity_fire.py`, and `tests/test_native_parity_palmer.py`
  have no other skip. The Python 3.14-only context-aware-warnings routing check lives in
  `tests/test_native_backend.py` and skips on the 3.10 leg. It then runs the consolidated
  parity suite as a named step, so a leg that collected nothing from it fails visibly
  rather than passing silently, and writes the measured parity maxima
  (`scripts/native_parity_maxima.py`, the table above) into the job summary.
- `native-wheel`: `maturin build --release` on Linux and macOS at both boundary
  Pythons, plus a Windows smoke build on the newest. Each wheel is installed
  into a fresh venv and imported from outside the checkout, and SPI must reach
  the bundled `gamma_parameters`, `gamma_probabilities`, and `norm_ppf` kernels.
  This validates the developer build; the published wheel matrix, its abi3 tag,
  and the installation checks for each artifact run in `.github/workflows/release.yml`
  ([ADR-0018](adr/0018-optional-rust-packaging.md)).

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

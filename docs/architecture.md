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
| **Python** | 3.10-3.13 | Language runtime |
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
- **Build Backend**: Hatchling (PEP 517 compliant)
- **Package Manager**: uv (modern resolver, lockfile support)
- **CI/CD**: GitHub Actions (3 workflows: unit tests, releases, benchmarks)
- **Container**: Docker with Python 3.11-slim base image
- **Documentation**: Sphinx with ReadTheDocs hosting

## Layered Architecture Pattern

```
┌────────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                  │
├────────────────────────────────────────────────────────────────────┤
│  __main__.py          │  Full-featured CLI (all indices)          │
│  __spi__.py           │  Specialized SPI CLI (param save/load)    │
├────────────────────────────────────────────────────────────────────┤
│                      Public API Layer                              │
├────────────────────────────────────────────────────────────────────┤
│  typed_public_api.py  │  Strict mypy, xarray wrappers            │
│  xarray_adapter.py    │  CF-compliant xarray interface           │
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
- **`__main__.py`** (1872 lines): Full-featured CLI supporting SPI, SPEI, PET, Palmer, and PNP indices
  - Multiprocessing pool for gridded data parallelization
  - NetCDF dimension validation and coordinate conversion
  - Shared memory arrays for worker processes
  - Input type detection (grid, divisions, timeseries)

- **`__spi__.py`** (1477 lines): Specialized SPI computation CLI
  - Distribution fitting parameter save/load for reusability
  - Parallel fitting and SPI computation
  - Supports gamma and Pearson Type III distributions
  - NetCDF output with CF metadata

**Entry Points** (`pyproject.toml`):
```toml
[project.scripts]
climate_indices = "climate_indices.__main__:main"
process_climate_indices = "climate_indices.__main__:main"
spi = "climate_indices.__spi__:main"
```

#### 2. Public API Layer
**Purpose**: User-facing interfaces for programmatic access.

**Modules**:
- **`typed_public_api.py`** (210 lines): Modern xarray API with strict mypy compliance
  - Type-safe wrappers for SPI and SPEI
  - Enforces keyword-only arguments
  - Full mypy --strict compliance

- **`xarray_adapter.py`** (2102 lines, expanded in 2.2.0): CF-compliant xarray interface
  - Input type detection (`DataArray`, `Dataset`, `ndarray`)
  - Coordinate validation and alignment
  - CF metadata preservation and generation
  - Dask array support with chunking validation
  - PET computation (Thornthwaite, Hargreaves)

- **`indices.py`** (856 lines): Legacy numpy API
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
- **`compute.py`** (1328 lines): Core computation functions
  - `scale_values()`: Rolling sum computation for temporal scaling
  - `gamma_parameters()`, `pearson_parameters()`: Distribution fitting
  - `transform_fitted_gamma()`, `transform_fitted_pearson()`: CDF transformation
  - `sum_to_scale()`: Optimized sliding window summation
  - `Periodicity` enum: `monthly` (12 steps/year), `daily` (366 steps/year)

- **`palmer.py`** (912 lines): Palmer Drought Index family
  - PDSI (Palmer Drought Severity Index)
  - PHDI (Palmer Hydrological Drought Index)
  - PMDI (Palmer Modified Drought Index)
  - ZINDEX (Palmer Z-Index)
  - Self-calibrated Palmer (scPDSI)

**Key Algorithms**:
1. **SPI/SPEI Computation**:
   ```
   Input: precip (or P-PET)
   ↓
   Scale to N-month window (sum_to_scale)
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

#### 4. Math/Statistics Layer
**Purpose**: Low-level mathematical and statistical functions.

**Modules**:
- **`eto.py`** (405 lines): Potential Evapotranspiration methods
  - **Thornthwaite (1948)**: Monthly PET from temperature and latitude
    - Heat index computation
    - Day length adjustment based on latitude
  - **Hargreaves (1985)**: Daily PET from temperature range and solar radiation
    - Requires tmin, tmax, and latitude
    - Extraterrestrial radiation calculation

- **`lmoments.py`** (188 lines): L-moments for robust distribution fitting
  - Implements Hosking (1990) L-moments algorithm
  - Used for Pearson Type III parameter estimation
  - More robust than method of moments for skewed distributions

**Design Note**: This layer has no dependencies on upper layers and could be extracted as standalone utilities.

#### 5. Infrastructure Layer
**Purpose**: Cross-cutting concerns (utilities, logging, error handling).

**Modules**:
- **`exceptions.py`** (323 lines): Exception hierarchy
  - Base: `ClimateIndicesError` (catch-all for library errors)
  - Computation: `DistributionFittingError`, `InsufficientDataError`, `PearsonFittingError`
  - Validation: `DimensionMismatchError`, `CoordinateValidationError`, `InputTypeError`, `InvalidArgumentError`
  - Warnings: `MissingDataWarning`, `ShortCalibrationWarning`, `GoodnessOfFitWarning`, `InputAlignmentWarning`
  - All exceptions carry context attributes (e.g., `distribution_name`, `input_shape`, `parameters`)

- **`logging_config.py`** (146 lines): Structured logging configuration
  - `configure_logging()`: Sets up structlog with JSON serialization
  - Console: Human-readable colored output
  - File: JSON-formatted for log aggregators
  - Context binding for tracing

- **`utils.py`** (549 lines): Utility functions
  - Calendar conversions: `transform_to_366day()`, `transform_to_gregorian()`
  - Data validation: `is_data_valid()`
  - Array reshaping: `reshape_to_2d()`, `reshape_to_divs()`
  - Periodicity utilities: `gregorian_length_as_366day()`

- **`performance.py`** (118 lines): Performance tracking
  - `@measure_execution_time` decorator
  - Memory usage tracking
  - Computation duration logging

## Source Code Organization

```
climate_indices/
├── src/climate_indices/          # Main package directory
│   ├── __init__.py               # Public API exports
│   ├── __main__.py               # Full-featured CLI entry point
│   ├── __spi__.py                # Specialized SPI CLI entry point
│   ├── typed_public_api.py       # Strict mypy-compliant API (NEW in 2.2.0)
│   ├── xarray_adapter.py         # Modern xarray interface (EXPANDED in 2.2.0)
│   ├── indices.py                # Legacy numpy API (STABLE)
│   ├── compute.py                # Core computation algorithms
│   ├── palmer.py                 # Palmer drought indices
│   ├── eto.py                    # PET: Thornthwaite & Hargreaves
│   ├── lmoments.py               # L-moments for Pearson fitting
│   ├── utils.py                  # Utility functions
│   ├── logging_config.py         # Structured logging setup
│   ├── exceptions.py             # Exception hierarchy
│   └── performance.py            # Performance metrics (NEW in 2.2.0)
│
├── tests/                        # Test suite (26 test files)
│   ├── conftest.py               # Shared fixtures (session-scoped)
│   ├── test_indices.py           # Legacy API tests
│   ├── test_xarray_adapter.py    # Modern API tests (EXPANDED in 2.2.0)
│   ├── test_compute.py           # Computation tests
│   ├── test_property_based.py    # Property-based invariant tests
│   ├── test_backward_compat.py   # Backward compatibility suite
│   ├── test_exceptions.py        # Exception handling tests
│   ├── test_logging.py           # Logging behavior tests
│   ├── test_metadata_validation.py # CF metadata validation tests
│   ├── test_benchmark_*.py       # Performance regression tests
│   └── fixture/                  # Test data (numpy arrays, JSON)
│
├── docs/                         # Documentation
│   ├── conf.py                   # Sphinx configuration
│   ├── index.rst                 # Main Sphinx doc (ReadTheDocs)
│   ├── reference.rst             # API reference (autodoc)
│   ├── pypi_release.rst          # PyPI release guide
│   ├── source/modules.rst        # Sphinx module toctree
│   ├── source/tests.rst          # Sphinx tests module reference
│   └── *.md                      # AI-readable docs (BMAD format, NEW)
│
├── .github/workflows/            # CI/CD pipelines
│   ├── unit-tests-workflow.yml   # Test matrix (Python 3.10-3.13)
│   ├── release.yml               # Automated PyPI releases
│   └── benchmarks.yml            # Performance tracking (NEW in 2.2.0)
│
├── pyproject.toml                # PEP 517 build config + tool settings
├── uv.lock                       # Reproducible dependency lock
├── Dockerfile                    # Container image definition
├── README.md                     # GitHub landing page
└── CONTRIBUTING.md               # Development guidelines
```

### Critical Directories
- **`src/climate_indices/`**: All production code (14 modules)
- **`tests/`**: 26 test files + fixture data (>90% coverage)
- **`docs/`**: Sphinx RST + BMAD Markdown documentation
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
│  Temporal Scaling (compute.scale_values)                        │
│  - Rolling sum over N months/days                               │
│  - Handles NaN propagation                                      │
│  - Output: scaled_values (same shape as input)                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Distribution Fitting (per calendar month/day)                  │
│  - Gamma: alpha, beta parameters (method of moments)            │
│  - Pearson: loc, scale, skew parameters (L-moments)             │
│  - Fit on calibration period only                               │
│  - Track probability of zero separately                         │
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

### Test Organization (26 Test Files)
```
tests/
├── conftest.py                      # 1004 lines - Session-scoped fixtures
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
│   ├── test_metadata_validation.py  # CF metadata tests
│   ├── test_computation_errors.py   # Error condition tests
│   ├── test_data_quality_warnings.py # Warning behavior tests
│   └── test_input_type_detection.py # Type detection tests
│
├── Observability Tests
│   ├── test_logging.py              # Structured logging tests
│   ├── test_logging_config.py       # Logger configuration tests
│   ├── test_error_context_logging.py # Error context tests
│   ├── test_calculation_event_logging.py # Calc event logs
│   └── test_performance_metrics.py  # Performance tracking tests
│
├── Performance Tests (Benchmarks)
│   ├── test_benchmark_overhead.py   # xarray vs numpy overhead
│   ├── test_benchmark_chunked.py    # Dask chunking strategies
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
**Trigger**: Push, pull request
```yaml
Matrix:
  - OS: ubuntu-latest
  - Python: [3.10, 3.11, 3.12, 3.13]
Steps:
  1. Checkout code
  2. Setup Python + uv
  3. uv sync --dev
  4. Run pytest
  5. Complete matrix test run
```

#### 2. release.yml
**Trigger**: Git tag push (vX.Y.Z)
```yaml
Steps:
  1. Checkout code
  2. Build sdist and wheel (hatchling)
  3. Publish to PyPI via trusted publishing (OIDC)
```

#### 3. benchmarks.yml (NEW in 2.2.0)
**Trigger**: Pull request to master, manual dispatch
```yaml
Steps:
  1. Checkout code
  2. Setup Python + uv
  3. uv sync --group dev
  4. Run pytest -m benchmark --benchmark-enable --benchmark-json
  5. Compare against baseline
  6. Post results as artifact
```

### Docker Container

**Base Image**: `python:3.11-slim`
**Build Strategy**: Multi-stage (builder + production)

```dockerfile
# Builder stage: Install dependencies
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Production stage: Copy venv + source
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libhdf5-dev libnetcdf-dev
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
USER climate  # Non-root user
ENTRYPOINT ["python", "-m", "climate_indices"]
```

**Usage**:
```bash
docker build -t climate_indices:2.2.0 .
docker run -v $(pwd)/data:/data climate_indices:2.2.0 \
    --index spi --scales 6 --netcdf_precip /data/precip.nc \
    --var_name_precip prcp --output_file_base /data/spi
```

### PyPI Distribution

**Package Name**: `climate_indices`
**Installation**: `pip install climate_indices` or `uv pip install climate_indices`
**Artifacts**:
- **Source distribution** (`climate_indices-2.2.0.tar.gz`)
- **Wheel** (`climate_indices-2.2.0-py3-none-any.whl`)

**Excludes from package** (`pyproject.toml`):
- `tests/`, `docs/`, `notebooks/`, `assets/`, `.github/`, `.venv/`, cache directories

## Key Design Decisions and Trade-offs

### 1. Dual API (numpy vs xarray)
**Decision**: Maintain both numpy and xarray APIs.

**Rationale**:
- **Numpy**: Minimal dependencies, direct array manipulation, backward compatibility
- **xarray**: Labeled dimensions, CF metadata, Dask integration, modern workflow

**Trade-off**: Code duplication risk mitigated by having xarray API call numpy implementation internally.

### 2. Multiprocessing vs Dask
**Decision**: Use multiprocessing in CLI, Dask in xarray API.

**Rationale**:
- **Multiprocessing**: Predictable memory usage, no Dask dependency for CLI users
- **Dask**: Lazy evaluation, better integration with xarray ecosystem, dynamic scheduling

**Trade-off**: Separate parallelization logic in CLI and xarray layers.

### 3. Time Dimension Chunking Constraint
**Decision**: Dask arrays MUST have time as single chunk.

**Rationale**: Climate indices require access to full time series for distribution fitting.

**Enforcement**: `xarray_adapter.py` validates chunking and raises `DimensionMismatchError` if violated.

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
4. **Chunking**: Spatial chunks in Dask, single time chunk
5. **Caching**: Distribution fitting parameters can be saved/loaded (\_\_spi\_\_.py)

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
- **User inputs**: Scale range [1-72], year validation
- **Path sanitization**: No path traversal in CLI file arguments

### Dependency Security
- **uv lock**: Pinned dependencies with checksums
- **GitHub Actions**: Pinned action versions with SHA hashes
- **Container**: Non-root user execution

### No Network Communication
Library has no network dependencies; all data is file-based.

---

**Next Steps**: See [source-tree-analysis.md](./source-tree-analysis.md) for annotated directory structure, [development-guide.md](./development-guide.md) for setup instructions, and [deployment-guide.md](./deployment-guide.md) for CI/CD details.

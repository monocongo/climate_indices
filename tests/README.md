# Test Architecture

This document describes the organization and usage of the `climate_indices` test suite.

## Directory Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Main test configuration and shared fixtures
├── conftest_numpy.py            # NumPy array fixtures (legacy .npy loaders)
├── conftest_xarray.py           # xarray DataArray fixtures
├── conftest_benchmark.py        # Benchmark-specific fixtures
├── helpers/
│   ├── __init__.py
│   ├── logging.py               # Centralized logging suppression fixture
│   └── strategies.py            # Reusable Hypothesis strategies
├── fixture/                     # Test data files (.npy, .json)
└── test_*.py                    # Test modules (28 files, 672+ tests)
```

## Fixture Organization

Fixtures are split across focused modules for maintainability:

### `conftest.py` (main orchestrator)
- Shared constants (year ranges, latitude values)
- Configuration fixtures (data/calibration year ranges as fixtures)
- Registers plugin modules via `pytest_plugins`

### `conftest_numpy.py`
- Legacy NumPy array loaders (`.npy` files from `fixture/` directory)
- Hargreaves temperature fixtures (synthetic daily temperature data)
- Example fixtures: `precips_mm_monthly`, `gamma_monthly`, `spi_6_month_gamma`

### `conftest_xarray.py`
- xarray DataArray fixtures for modern climate index tests
- **Factory fixture**: `make_dataarray` - create custom DataArrays with configurable params
- 1D time series, 2D grids, 3D volumes
- Dask-backed arrays, NaN patterns, coordinate-rich arrays
- Example fixtures: `sample_monthly_precip_da`, `gridded_monthly_precip_3d`, `dask_monthly_precip_1d`

### `conftest_benchmark.py`
- Fixtures for pytest-benchmark performance tests
- Both NumPy and xarray versions for fair comparisons
- Example fixtures: `bench_monthly_precip_np`, `bench_monthly_precip_da`

## Shared Test Utilities

### `helpers/logging.py`
- `suppress_logging`: Autouse fixture that disables logging during tests
- Reduces noise in test output (especially for Hypothesis property-based tests)
- Automatically applied to all test modules

### `helpers/strategies.py`
Reusable Hypothesis strategies for property-based testing:
- `monthly_precipitation_array()`: Valid monthly precip arrays (30-50 years)
- `monthly_temperature_array()`: Seasonal temperature patterns
- `daily_temperature_triplet()`: Valid (tmin, tmax, tmean) triplets
- `valid_latitude()`: Latitudes avoiding pole singularities (-89° to 89°)
- `valid_scale()`: SPI/SPEI scale parameters (1-24)
- `precip_with_uniform_offset()`: Paired arrays for monotonicity tests

**Usage:**
```python
from tests.helpers.strategies import monthly_precipitation_array, valid_scale
from hypothesis import given

@given(precip=monthly_precipitation_array(), scale=valid_scale())
def test_spi_properties(precip, scale):
    result = indices.spi(precip, scale=scale, ...)
    assert result.shape == precip.shape
```

## Running Tests

### Basic test run
```bash
uv run pytest
```

### With coverage
```bash
uv run pytest --cov --cov-report=term-missing
uv run pytest --cov --cov-report=html  # Generate HTML report in htmlcov/
```

### Parallel execution (faster)
```bash
uv run pytest -n auto  # Use all CPU cores
```

### Run specific test types
```bash
uv run pytest -m benchmark         # Run only benchmarks
uv run pytest -m "not benchmark"   # Skip benchmarks (default)
uv run pytest -m slow              # Run slow tests
uv run pytest tests/test_spi.py    # Run specific module
```

### With randomized order (catch hidden dependencies)
```bash
uv run pytest -p randomly
```

### With timeout protection
```bash
uv run pytest --timeout=120  # Fail tests that hang >120s
```

## Test Markers

Markers are defined in `pyproject.toml` and validated with `--strict-markers`:

- `benchmark`: Performance benchmark tests (use `pytest-benchmark`)
  - Deselected by default via `addopts = "-m 'not benchmark'"`
  - Run explicitly with: `pytest -m benchmark`
- `slow`: Slow-running tests (>5s) that may be skipped in CI
- `integration`: Tests requiring external resources or full pipeline execution

**Adding new markers:**
1. Add to `pyproject.toml` `[tool.pytest.ini_options]` `markers` list
2. Use in tests: `@pytest.mark.integration`

## Coverage Configuration

Coverage config is in `pyproject.toml` `[tool.coverage.*]`:
- **Minimum coverage**: 70% (`fail_under = 70`)
- **Source**: `src/climate_indices` only
- **Branch coverage**: Enabled
- **Exclusions**: Debug code, `TYPE_CHECKING` blocks, unreachable code

Coverage reports show missing lines to guide test additions.

## Adding New Fixtures

### Where to add fixtures:

| Fixture Type | Location | Example |
|--------------|----------|---------|
| NumPy arrays from .npy files | `conftest_numpy.py` | `precips_mm_monthly` |
| xarray DataArrays (session-scoped) | `conftest_xarray.py` | `sample_monthly_precip_da` |
| Benchmark fixtures | `conftest_benchmark.py` | `bench_monthly_precip_np` |
| Shared config constants | `conftest.py` | `calibration_year_start_monthly` |
| One-off custom DataArrays | Use `make_dataarray` factory | N/A |

### Fixture best practices:
- Use `scope="session"` for expensive fixtures (file I/O, data generation)
- Use `scope="function"` for mutable data or test isolation
- Add type hints to all fixture return types
- Use `pathlib.Path` for file paths (not `os.path`)
- Use `np.random.default_rng(seed)` for deterministic random data

### Factory fixture example:
```python
def test_custom_precip(make_dataarray):
    # Create a 2-year monthly DataArray
    da = make_dataarray(freq="MS", periods=24, dims=1, seed=42)
    assert len(da.time) == 24
```

## CI Pipeline

The CI workflow (`.github/workflows/unit-tests-workflow.yml`) runs on all pushes and PRs:

### Jobs:
1. **lint** (Python 3.12, ubuntu-latest)
   - `ruff check` - Code linting
   - `ruff format --check` - Format checking
   - `mypy src/` - Type checking

2. **test** (Matrix: Python 3.10-3.13, ubuntu + macOS)
   - Run full test suite with coverage
   - Upload coverage to Codecov (Python 3.12/ubuntu only)

3. **test-minimum-deps** (Python 3.10, ubuntu-latest)
   - Test with `--resolution lowest-direct` to verify minimum dependencies work

4. **security-audit** (Python 3.12, ubuntu-latest)
   - `pip-audit` - Check for CVEs in dependencies

### Coverage reporting:
- Coverage uploaded to Codecov on every push
- Fails if coverage drops below 70% (configured in `pyproject.toml`)
- XML and terminal reports show missing coverage

### Caching:
- `uv` dependencies cached using `setup-uv@v5` `enable-cache: true`
- Cache key: `uv.lock` fingerprint
- Significantly reduces CI build time

## Property-Based Testing

The test suite uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing (see `test_property_based.py`).

**Strategy:**
- Generate thousands of random valid inputs
- Verify mathematical invariants hold for all inputs
- Automatically find edge cases that break assumptions

**Tested properties:**
- Boundedness: SPI/SPEI values within ±3.09 std deviations
- Monotonicity: Higher precipitation → higher SPI
- Shape preservation: Output shape matches input shape
- NaN propagation: NaN inputs → NaN outputs at same positions

**Strategies location:** `tests/helpers/strategies.py` (reusable across test modules)

## Troubleshooting

### Tests fail with fixture not found
- Check that fixture is defined in the correct `conftest_*.py` module
- Verify `pytest_plugins` list in `conftest.py` includes the module

### Tests hang indefinitely
- Use `pytest --timeout=120` to add global timeout
- Check for blocking I/O or infinite loops in test code

### Coverage not generated
- Ensure `pytest-cov` is installed: `uv sync --dev`
- Verify `pyproject.toml` has `[tool.coverage.run]` `source = ["src/climate_indices"]`

### Random test failures
- Check for unseeded random number generation (`np.random.rand()` without seed)
- Use `pytest -p randomly` to detect hidden test dependencies
- Ensure tests are isolated (function-scoped fixtures for mutable data)

### Slow test suite
- Run with `-n auto` for parallel execution
- Profile with `pytest --durations=10` to find slowest tests
- Consider adding `@pytest.mark.slow` to long-running tests

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [pytest-xdist (parallel testing)](https://pytest-xdist.readthedocs.io/)

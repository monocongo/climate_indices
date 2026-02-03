# CLAUDE.md

> **Note**: For tool-agnostic instructions shared across all AI assistants, see `AGENTS.md`.
> For on-demand context, see `context/INDEX.md`.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python library for computing climate indices useful for drought monitoring and climate research. The library implements several standardized climate indices including SPI (Standardized Precipitation Index), SPEI (Standardized Precipitation Evapotranspiration Index), PET (Potential Evapotranspiration), PNP (Percentage of Normal Precipitation), and Palmer drought indices.

The library provides both a NumPy-based API and an xarray-native API with optional Dask parallelism for large-scale data processing.

## Essential Development Commands

### Testing
- **Run all tests**: `uv run pytest`
- **Run specific test file**: `uv run pytest tests/test_indices.py`
- **Run with coverage**: `uv run pytest --cov=climate_indices`

### Code Quality
- **Run linting**: `ruff check --fix` (configured in pyproject.toml)
- **Run formatting**: `ruff format .` (120 character line length)
- **Type checking**: `uv run mypy src/climate_indices`
- **Run all pre-commit hooks**: `uv run pre-commit run --all-files`

### Building and Installation
- **Install development dependencies**: `uv sync --group dev --group test`
- **Build package**: `uv build`
- **Run CLI**: `uv run climate_indices` or `uv run python -m climate_indices`

### Documentation
- **Build docs**: `cd docs && make html` (Sphinx documentation)
- **Clean docs**: `cd docs && make clean`

## Code Architecture

### Core Module Structure

- **`src/climate_indices/`**: Main package directory
  - **`__init__.py`**: Package initialization and accessor registration
  - **`__main__.py`**: CLI entry point with argument parsing
  - **`__spi__.py`**: SPI CLI implementation (xarray/dask-based pipeline)
  - **`accessors.py`**: Xarray DataArray accessor for fluent API usage
  - **`indices.py`**: High-level API for computing climate indices (SPI, SPEI, PET, etc.)
  - **`compute.py`**: Core computational functions with numba acceleration
  - **`utils.py`**: Utility functions for data manipulation and logging
  - **`palmer.py`**: Palmer drought index calculations
  - **`eto.py`**: Evapotranspiration calculations (Thornthwaite, Hargreaves)
  - **`lmoments.py`**: L-moments statistical computations

### Key Design Patterns

#### Xarray/Dask Processing Architecture
The SPI CLI uses an xarray-native pipeline with optional Dask parallelism:
- NetCDF files are opened with xarray using `chunks={"time": -1}` (time as single chunk)
- Processing uses `xr.apply_ufunc()` with `dask="parallelized"` for vectorized operations
- Time dimension kept as single chunk to prevent rolling window sums from crossing boundaries
- `--multiprocessing all` enables Dask distributed parallelism across multiple workers

#### Xarray-Native SPI API
Two complementary interfaces for xarray-based workflows:
- **`indices.spi_xarray()`**: Explicit function for pipeline code with full parameter control
- **`da.indices.spi()`**: DataArray accessor for concise, chainable operations in notebooks

```python
import xarray as xr
from climate_indices import indices
from climate_indices.compute import Periodicity

# Function API
spi_values = indices.spi_xarray(
    precip_da, scale=3, distribution="gamma",
    data_start_year=1981, calibration_year_initial=1981,
    calibration_year_final=2010, periodicity=Periodicity.monthly
)

# Accessor API (chainable)
spi_values = precip_da.indices.spi(
    scale=3, distribution="gamma", data_start_year=1981,
    calibration_year_initial=1981, calibration_year_final=2010,
    periodicity=Periodicity.monthly
)
```

#### Numba Acceleration
Performance-critical probability calculations are JIT-compiled with numba:
- `compute._pearson_fit()`: Pearson Type III probability adjustments
- `compute._minimum_possible()`: Minimum possible value computations
- Uses `nopython=True` and `cache=True` for maximum performance

#### Data Validation and Transformation
- Input validation occurs in `compute._validate_array()`
- Data is automatically reshaped from 1D to 2D arrays based on periodicity
- Unit conversions are handled automatically (inches to mm, Fahrenheit to Celsius)
- Distribution fallback strategy: automatic fallback from Pearson to Gamma when fitting produces excessive NaNs

### Important Data Structures

#### Periodicity Enum
- `Periodicity.monthly`: 12 values per year
- `Periodicity.daily`: 366 values per year (leap year format)

#### Distribution Enum
- `Distribution.gamma`: Gamma distribution fitting for SPI/SPEI
- `Distribution.pearson`: Pearson Type III distribution fitting

#### Input Types
- `InputType.grid`: Gridded data with (lat, lon, time) dimensions
- `InputType.divisions`: US climate division data
- `InputType.timeseries`: 1D time series data

## Testing Framework

- Uses **pytest** with extensive fixture-based testing
- Test fixtures in `tests/conftest.py` provide sample datasets
- Fixtures include precipitation, temperature, and expected output arrays
- Tests cover both monthly and daily periodicities across multiple time scales
- `tests/test_accessors.py`: Tests for xarray DataArray accessor

## CLI Usage Patterns

The main CLI supports complex workflows with xarray/dask processing:

```bash
# SPI computation (serial, default)
uv run climate_indices --index spi --periodicity monthly --scales 1 3 6 12 \
  --netcdf_precip precip.nc --var_name_precip prcp \
  --calibration_start_year 1981 --calibration_end_year 2010 \
  --output_file_base output/spi

# SPI computation with Dask parallelism
uv run climate_indices --index spi --periodicity monthly --scales 1 3 6 12 \
  --netcdf_precip precip.nc --var_name_precip prcp \
  --calibration_start_year 1981 --calibration_end_year 2010 \
  --output_file_base output/spi --multiprocessing all

# Palmer indices (requires precipitation, temperature/PET, and AWC)
uv run climate_indices --index palmers --periodicity monthly \
  --netcdf_precip precip.nc --var_name_precip prcp \
  --netcdf_temp temp.nc --var_name_temp tavg \
  --netcdf_awc awc.nc --var_name_awc awc \
  --calibration_start_year 1951 --calibration_end_year 2010 \
  --output_file_base output/palmer
```

### Parallelization Options
- `--multiprocessing serial` (default): Single-threaded xarray processing
- `--multiprocessing all`: Dask distributed with `n_workers=cpu_count()-1`

### Fitting Parameter Persistence
- `--save_params <file>`: Save computed distribution parameters for reuse
- `--load_params <file>`: Load pre-computed parameters (alpha, beta, skew, loc, scale, prob_zero)

## Development Notes

### Dependency Management
- Project uses uv for dependency management and virtual environments
- Core dependencies: scipy, xarray, dask, h5netcdf, numba
- Development dependencies: pytest, ruff, mypy, pre-commit, coverage, sphinx-autodoc-typehints

### Pre-commit Hooks
The project uses pre-commit for automated code quality checks:
- **Ruff**: Linting and formatting
- **Mypy**: Type checking on `compute.py`
- **Standard hooks**: YAML validation, trailing whitespace, etc.

Run `uv run pre-commit install` to set up hooks locally.

### Mypy Configuration
Strict type checking is configured in pyproject.toml:
- `disallow_untyped_defs = true`
- `disallow_incomplete_defs = true`
- `no_implicit_optional = true`
- Strategic overrides for third-party libs (numba, scipy, xarray, dask)

### Python Version Support
- Supports Python 3.10 through 3.13
- Uses type hints throughout codebase
- Configured for modern Python features in pyproject.toml

### Performance Considerations
- Heavy use of NumPy arrays and vectorized operations
- Numba JIT compilation for hot-path probability calculations
- Xarray `apply_ufunc` with `dask="parallelized"` for scalable processing
- Chunked processing for memory efficiency with large NetCDF files
- Dask distributed available via `--multiprocessing all` for multi-core/multi-node scaling

### Error Handling
- Comprehensive input validation with descriptive error messages
- Logging throughout with configurable levels
- Graceful handling of edge cases (NaN values, missing data)
- `DistributionFallbackStrategy` for automatic fallback when Pearson fitting fails

# Project Overview

## Executive Summary

**climate_indices** is a production-grade Python library providing reference implementations of climate indices used for drought monitoring and agricultural planning. The library computes the Standardized Precipitation Index (SPI), Standardized Precipitation Evapotranspiration Index (SPEI), Potential Evapotranspiration (PET), Palmer Drought Indices, and Percentage of Normal Precipitation (PNP) from meteorological time series data.

### Project Status
- **Maturity**: Production/Stable (Development Status 5)
- **Current Version**: 2.2.0
- **Python Support**: 3.10, 3.11, 3.12, 3.13
- **License**: BSD 3-Clause
- **Documentation**: [ReadTheDocs](https://climate-indices.readthedocs.io/)
- **Repository**: [GitHub](https://github.com/monocongo/climate_indices)

### Key Capabilities
- **Multiple Climate Indices**: SPI, SPEI, PET (Thornthwaite & Hargreaves), Palmer Drought Indices, PNP
- **Flexible Input Formats**: Supports numpy arrays, xarray DataArrays, and Dask arrays
- **Multiple Temporal Scales**: Monthly and daily data, multiple time scales (1-72 months/days)
- **Distribution Options**: Gamma and Pearson Type III distributions for SPI/SPEI
- **CLI Tools**: Three command-line entry points for batch processing NetCDF data
- **Scientific Rigor**: Based on peer-reviewed methodologies with comprehensive validation

## Project Classification

### Repository Structure
- **Type**: Monolith (single cohesive codebase)
- **Project Type**: Library
- **Primary Language**: Python 3.10+
- **Build System**: Hatchling (PEP 517) + uv for dependency management
- **Package Management**: uv (modern Python package installer/resolver)

### Architecture Pattern
**Layered Library Architecture**:
```
┌─────────────────────────────────────┐
│   CLI Layer                         │  ← __main__.py, __spi__.py
├─────────────────────────────────────┤
│   Public API Layer                  │  ← typed_public_api.py, xarray_adapter.py
├─────────────────────────────────────┤
│   Computation Layer                 │  ← indices.py, compute.py
├─────────────────────────────────────┤
│   Math/Statistics Layer             │  ← eto.py (Thornthwaite, Hargreaves)
├─────────────────────────────────────┤
│   Infrastructure Layer              │  ← utils.py, logging_config.py, exceptions.py
└─────────────────────────────────────┘
```

### Entry Points
The library provides three CLI entry points for processing NetCDF datasets:
- **`climate_indices`** / **`process_climate_indices`**: Full-featured CLI for all indices
- **`spi`**: Specialized CLI for SPI computation with distribution fitting parameter save/load

### Technology Stack Summary
| Category | Technology |
|----------|-----------|
| **Core Language** | Python 3.10+ |
| **Scientific Computing** | NumPy, SciPy, xarray, Dask |
| **Logging** | structlog (structured logging) |
| **Testing** | pytest, hypothesis (property-based), pytest-benchmark |
| **Type Checking** | mypy --strict |
| **Linting/Formatting** | ruff |
| **Build** | Hatchling (PEP 517) |
| **CI/CD** | GitHub Actions (3 workflows) |
| **Containerization** | Docker (Python 3.11-slim base) |
| **Documentation** | Sphinx with ReadTheDocs hosting |

## Quick Reference

### Installation
```bash
# From PyPI
pip install climate_indices

# From source (development mode with uv)
git clone https://github.com/monocongo/climate_indices.git
cd climate_indices
uv sync --group dev
```

### Core API Usage

#### Modern xarray API (Recommended)
```python
import climate_indices as ci
import xarray as xr

# Load precipitation data
precip = xr.open_dataarray("precip.nc")

# Compute 6-month SPI
spi_6 = ci.spi(
    precip,
    scale=6,
    distribution="gamma",
    calibration_start_year=1981,
    calibration_end_year=2010
)
```

#### Legacy NumPy API (Backward Compatibility)
```python
from climate_indices import indices, compute
import numpy as np

# Load data as numpy array
precip_mm = np.load("precip_monthly.npy")

# Compute 6-month SPI
spi_6 = indices.spi(
    precip_mm,
    scale=6,
    distribution=indices.Distribution.gamma,
    data_start_year=1980,
    calibration_year_initial=1981,
    calibration_year_final=2010,
    periodicity=compute.Periodicity.monthly
)
```

### CLI Usage
```bash
# Process gridded NetCDF data for multiple indices
climate_indices \
    --index spi \
    --periodicity monthly \
    --scales 1 3 6 12 \
    --netcdf_precip precip.nc \
    --var_name_precip prcp \
    --calibration_start_year 1981 \
    --calibration_end_year 2010 \
    --output_file_base results/spi

# Specialized SPI computation with parameter caching
spi \
    --periodicity monthly \
    --scales 1 3 6 12 \
    --netcdf_precip precip.nc \
    --var_name_precip prcp \
    --calibration_start_year 1981 \
    --calibration_end_year 2010 \
    --save_params fitting_params.nc \
    --output_file_base results/spi
```

### Development Commands
```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term

# Type checking
mypy src/

# Linting and formatting
ruff check --fix
ruff format

# Run benchmarks (deselected by default)
uv run pytest -m benchmark
```

## Project Goals and Use Cases

### Primary Use Cases
1. **Drought Monitoring**: Real-time and historical drought assessment using SPI/SPEI
2. **Agricultural Planning**: Crop yield forecasting and irrigation scheduling
3. **Water Resource Management**: Reservoir operations and water allocation decisions
4. **Climate Research**: Long-term precipitation and drought trend analysis
5. **Operational Meteorology**: Integration into weather services and early warning systems

### Design Philosophy
- **Scientific Correctness**: Implementations follow peer-reviewed methodologies (McKee et al. 1993, Vicente-Serrano et al. 2010, Thornthwaite 1948, Palmer 1965)
- **Performance**: Optimized for large gridded datasets using Dask parallelization
- **Usability**: Dual API (numpy/xarray) for different user workflows
- **Reliability**: Comprehensive test coverage (>90%), property-based testing, backward compatibility guarantees
- **Extensibility**: Modular design allows easy addition of new indices or distribution types

## AI-Assisted Development Guidance

### Recommended Starting Points for AI Agents
1. **For understanding computation**: Start with `docs/index.rst` (Sphinx overview), then `src/climate_indices/compute.py`
2. **For understanding API**: Read `src/climate_indices/typed_public_api.py` (strict mypy typing) and `src/climate_indices/xarray_adapter.py`
3. **For understanding CLI**: Examine `src/climate_indices/__main__.py` (full-featured) and `src/climate_indices/__spi__.py` (specialized)
4. **For testing patterns**: Review `tests/conftest.py` (fixtures), `tests/test_xarray_adapter.py` (modern API), `tests/test_property_based.py` (invariants)
5. **For error handling**: Study `src/climate_indices/exceptions.py` (complete hierarchy with attributes)

### Critical Architectural Invariants
- **Time dimension chunking**: Dask arrays MUST have time as single chunk (`time: -1`) for climate indices
- **Calibration period**: Default minimum 30 years; violations trigger `ShortCalibrationWarning`
- **Distribution fitting**: Requires minimum 10 non-zero values; insufficient data raises `InsufficientDataError`
- **Coordinate validation**: Time coordinates must be monotonically increasing; xarray inputs undergo automatic validation
- **Backward compatibility**: Legacy numpy API (`indices.py`) must remain stable; new features go to xarray API

### Code Patterns to Follow
1. **Type annotations**: All functions require full type hints (enforced by mypy --strict on `typed_public_api.py`)
2. **Docstrings**: Google-style docstrings with parameter descriptions, return types, raises, examples
3. **Error handling**: Use specific exception types from `exceptions.py` with context attributes
4. **Logging**: Use structlog with structured key-value pairs; avoid string interpolation in log messages
5. **Testing**: Property-based tests for mathematical invariants, regression tests for known outputs, benchmark tests for performance

### Dependencies and Constraints
- **Core dependencies**: scipy>=1.15.3, xarray>=2025.6.1, dask>=2025.7.0, structlog>=24.1.0
- **Python version**: Must support Python 3.10-3.13 (no features from 3.14+)
- **Line length**: 120 characters (ruff configuration)
- **Import order**: stdlib → third-party → local (enforced by ruff)
- **Test markers**: Use `@pytest.mark.benchmark` for performance tests, `@pytest.mark.slow` for long-running tests

### Common Pitfalls
1. **Do NOT** use wildcard imports (`from module import *`) - explicitly forbidden by ruff
2. **Do NOT** chunk time dimension in Dask arrays - causes incorrect index calculations
3. **Do NOT** modify `indices.py` API - backward compatibility requirement
4. **Do NOT** commit without running tests - pre-commit hooks will fail
5. **Do NOT** use string paths - always use `pathlib.Path` objects

### Key Files for Modification Scenarios
| Task | Primary Files | Test Files |
|------|--------------|------------|
| Add new index | `compute.py`, `indices.py`, `xarray_adapter.py` | `test_compute.py`, `test_indices.py`, `test_xarray_adapter.py` |
| Add new distribution | `compute.py`, `indices.py` | `test_compute.py`, property tests |
| Fix CLI bug | `__main__.py` or `__spi__.py` | Integration tests (manual) |
| Add validation | `xarray_adapter.py`, `exceptions.py` | `test_input_validation.py`, `test_exceptions.py` |
| Performance optimization | `compute.py`, chunking strategies | `test_benchmark_*.py` |

---

**Next Steps**: See [architecture.md](./architecture.md) for detailed technical architecture, [development-guide.md](./development-guide.md) for setup instructions, and [reference.rst](./reference.rst) for Sphinx API documentation.

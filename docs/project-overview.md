# Project Overview

## Executive Summary

**climate_indices** is a production-grade Python library providing reference implementations of climate and hydrologic indices used for drought monitoring, agricultural planning, flood-potential assessment, and fire-weather assessment. The library computes the Standardized Precipitation Index (SPI), Standardized Precipitation Evapotranspiration Index (SPEI), Evaporative Demand Drought Index (EDDI), Percentage of Normal Precipitation (PNP), Precipitation Concentration Index (PCI), Potential Evapotranspiration (PET), the Palmer Drought Indices, a namespaced family of fire-weather indices, and a namespaced family of flood-potential indices (Effective Precipitation, EDI, Flood Index, Antecedent Precipitation Index).

### Project Status
- **Maturity**: Production/Stable (Development Status 5)
- **Current Version**: 3.0.0
- **Python Support**: 3.10, 3.11, 3.12, 3.13, 3.14
- **License**: BSD 3-Clause
- **Documentation**: [ReadTheDocs](https://climate-indices.readthedocs.io/)
- **Repository**: [GitHub](https://github.com/monocongo/climate_indices)

### Key Capabilities
- **Drought / Moisture Indices**: SPI, SPEI, PNP, EDDI, PCI
- **Palmer Family**: PDSI, PHDI, PMDI, Palmer Z-Index, and self-calibrated scPDSI
- **Flood-Potential Indices**: `climate_indices.flood` namespace — Effective Precipitation, fixed-window EDI, Flood Index, and the recursive Antecedent Precipitation Index
- **PET Methods**: Thornthwaite and Hargreaves (public API); FAO-56 Penman-Monteith primitives are internal (`pm_eto.py`) and back the fire-weather indices
- **Fire-Weather Indices**: `climate_indices.fire` namespace — Fosberg Fire Weather Index (FFWI), Hot-Dry-Windy Index (HDW), Keetch-Byram Drought Index (KBDI), and the Canadian Forest Fire Weather Index System (FFMC, DMC, DC, ISI, BUI, FWI, DSR)
- **Run Theory**: `runs` module for identifying contiguous above/below-threshold events (duration, magnitude, intensity, peak, interarrival)
- **Flexible Input Formats**: NumPy arrays, xarray DataArrays, and Dask arrays
- **Multiple Temporal Scales**: Monthly and daily data, time scales 1-72 months/days (where applicable)
- **Distribution Options**: Gamma and Pearson Type III distributions for SPI/SPEI
- **CLI Tools**: A single `climate_indices` CLI for batch processing NetCDF data across the index families
- **Scientific Rigor**: Peer-reviewed methodologies with comprehensive validation against reference datasets

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
│   CLI Layer                         │  ← __main__.py, _cli.py
├─────────────────────────────────────┤
│   Public API Layer                  │  ← typed_public_api.py, xarray_adapter.py,
│                                     │     cf_metadata_registry.py, fire/, flood/
├─────────────────────────────────────┤
│   Computation Layer                 │  ← indices.py, compute.py, palmer.py,
│                                     │     self_calibration.py, eto.py, pm_eto.py,
│                                     │     runs.py, validation.py
├─────────────────────────────────────┤
│   Math/Statistics Layer             │  ← lmoments.py, _palmer_wells.py,
│                                     │     _palmer_duration.py
├─────────────────────────────────────┤
│   Infrastructure Layer              │  ← utils.py, logging_config.py,
│                                     │     exceptions.py, performance.py
└─────────────────────────────────────┘
```

### Entry Points
The installed script names map to the single full-featured CLI:
- **`climate_indices`** / **`process_climate_indices`**: `__main__.main`, the only CLI entry point. `--index` choices are `spi`, `spei`, `pnp`, `scaled`, `pet`, `palmers`, `kbdi`, `pe`, `edi`, `flood_index`, `api`, and `all`. The former specialized `spi` CLI (parameter save/load) was retired in 3.0.0; fit once with `compute.gamma_parameters()`/`compute.pearson_parameters()` and pass the dict to `fitting_params`.

### Technology Stack Summary
| Category | Technology |
|----------|-----------|
| **Core Language** | Python 3.10+ |
| **Scientific Computing** | NumPy, SciPy, xarray, Dask, pandas, cftime, h5netcdf |
| **Logging** | structlog (structured logging) |
| **Testing** | pytest, hypothesis (property-based), pytest-benchmark |
| **Type Checking** | mypy --strict |
| **Linting/Formatting** | ruff |
| **Build** | Hatchling (PEP 517) |
| **CI/CD** | GitHub Actions (`benchmarks.yml`, `release.yml`, `unit-tests-workflow.yml`) |
| **Containerization** | Docker (Python 3.14-slim base) |
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
from climate_indices import indices
import xarray as xr

# Load precipitation data
precip = xr.open_dataarray("precip.nc")

# Compute 6-month SPI (time parameters inferred from coordinates when omitted)
spi_6 = ci.spi(
    precip,
    scale=6,
    distribution=indices.Distribution.gamma,
    calibration_year_initial=1981,
    calibration_year_final=2010
)
```

#### Legacy NumPy API (Backward Compatibility)

> Note: the calibration-window boundary parameters are `calibration_year_initial`/
> `calibration_year_final` across the Python API. `percentage_of_normal` still
> accepts the older `calibration_start_year`/`calibration_end_year` keywords as
> deprecated aliases (removed in 4.0.0), and the CLI accepts
> `--calibration_year_initial`/`--calibration_year_final` as aliases for its
> `--calibration_start_year`/`--calibration_end_year` flags.
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
```

### Development Commands
```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term

# Type checking
uv run mypy src/

# Linting and formatting
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Run benchmarks (deselected by default)
uv run pytest -m benchmark
```

## Project Goals and Use Cases

### Primary Use Cases
1. **Drought Monitoring**: Real-time and historical drought assessment using SPI/SPEI/PDSI
2. **Agricultural Planning**: Crop yield forecasting and irrigation scheduling
3. **Water Resource Management**: Reservoir operations and water allocation decisions
4. **Flood Assessment**: Daily flood-potential screening with Effective Precipitation, EDI, Flood Index, and API
5. **Fire-Weather Assessment**: Fuel-dryness and fire-danger indices for wildland fire management
6. **Event Analysis**: Run-theory extraction of drought, wet, and fire-spell events
7. **Climate Research**: Long-term precipitation and drought trend analysis
8. **Operational Meteorology**: Integration into weather services and early warning systems

### Design Philosophy
- **Scientific Correctness**: Implementations follow peer-reviewed methodologies (McKee et al. 1993, Vicente-Serrano et al. 2010, Thornthwaite 1948, Hargreaves & Samani 1985, Palmer 1965, Wells et al. 2004, Allen et al. 1998 FAO-56)
- **Performance**: Optimized for large gridded datasets using Dask parallelization
- **Usability**: Dual API (NumPy/xarray) for different user workflows
- **Reliability**: Comprehensive test coverage (>90%), property-based testing, backward compatibility guarantees
- **Extensibility**: Modular design allows addition of new indices, distributions, or PET methods

## AI-Assisted Development Guidance

### Recommended Starting Points for AI Agents
1. **For understanding computation**: Start with `docs/architecture.md`, then `src/climate_indices/compute.py` and `src/climate_indices/indices.py`
2. **For understanding the public API**: Read `src/climate_indices/typed_public_api.py` (strict mypy typing) and `src/climate_indices/xarray_adapter.py`
3. **For understanding the CLI**: Examine `src/climate_indices/__main__.py` (the index pipeline registry and `--index` handling) and `src/climate_indices/_cli.py` (shared argument registration)
4. **For understanding Palmer calculations**: Read `src/climate_indices/palmer.py`, `src/climate_indices/_palmer_wells.py`, `src/climate_indices/_palmer_duration.py`, and `src/climate_indices/self_calibration.py`
5. **For understanding fire-weather indices**: Read `src/climate_indices/fire/__init__.py` and the subsystem design in `docs/design/fire-subsystem.md`
6. **For understanding flood-potential indices**: Read `src/climate_indices/flood/__init__.py` (public dispatch) and the `flood/_pe.py`, `flood/_edi.py`, `flood/_if.py`, `flood/_antecedent.py` kernels
7. **For understanding run theory**: Read `src/climate_indices/runs.py`
8. **For testing patterns**: Review `tests/conftest.py` (fixtures), `tests/test_xarray_adapter.py` (modern API), `tests/test_property_based.py` (invariants)
9. **For error handling**: Study `src/climate_indices/exceptions.py` (complete hierarchy with attributes)

### Critical Architectural Invariants
- **Time dimension chunking**: Dask arrays MUST have time as a single chunk (`time: -1`) for climate indices
- **Calibration period**: Default minimum 30 years; violations trigger `ShortCalibrationWarning`
- **Distribution fitting**: Requires minimum 10 non-zero values; insufficient data raises `InsufficientDataError`
- **Coordinate validation**: Time coordinates must be monotonically increasing; xarray inputs undergo automatic validation
- **Backward compatibility**: Legacy NumPy API (`indices.py`) must remain stable; new features go to the xarray API
- **Fire namespace**: Fire-weather indices are exposed only under `climate_indices.fire`; they are not top-level package functions (see ADR-0005)
- **Flood namespace**: Flood-potential indices are exposed only under `climate_indices.flood` (see ADR-0013); `edi` (flood) and `eddi` (drought) are distinct indices

### Code Patterns to Follow
1. **Type annotations**: All public functions require full type hints (enforced by mypy --strict on `typed_public_api.py`)
2. **Docstrings**: Google-style docstrings with parameter descriptions, return types, raises, examples
3. **Error handling**: Use specific exception types from `exceptions.py` with context attributes
4. **Logging**: Use structlog with structured key-value pairs; avoid string interpolation in log messages
5. **Testing**: Property-based tests for mathematical invariants, regression tests against reference datasets, benchmark tests for performance

### Dependencies and Constraints
- **Core dependencies**: cftime>=1.6.4.post1, dask>=2025.7.0, h5netcdf>=1.6.3, numpy>=1.24.0, pandas>=2.0.0, scipy>=1.15.3, structlog>=24.1.0, xarray>=2025.6.1
- **Python version**: Must support Python 3.10-3.14 (`requires-python = ">=3.10,<3.15"`)
- **Line length**: 120 characters (ruff configuration)
- **Import order**: stdlib → third-party → local (enforced by ruff)
- **Test markers**: Use `@pytest.mark.benchmark` for performance tests, `@pytest.mark.slow` for long-running tests

### Common Pitfalls
1. **Do NOT** use wildcard imports (`from module import *`) - explicitly forbidden by ruff
2. **Do NOT** chunk the time dimension in Dask arrays - causes incorrect index calculations
3. **Do NOT** modify the `indices.py` API in a breaking way - backward compatibility requirement
4. **Do NOT** commit without running tests - pre-commit hooks will fail
5. **Do NOT** use string paths - always use `pathlib.Path` objects

### Key Files for Modification Scenarios
| Task | Primary Files | Test Files |
|------|--------------|------------|
| Add new index | `compute.py`, `indices.py`, `typed_public_api.py`, `xarray_adapter.py` | `test_compute.py`, `test_indices.py`, `test_typed_public_api.py`, `test_xarray_adapter.py` |
| Add new distribution | `compute.py`, `indices.py`, `lmoments.py` | `test_compute.py`, `test_property_based.py` |
| Modify Palmer indices | `palmer.py`, `_palmer_wells.py`, `_palmer_duration.py`, `self_calibration.py` | `test_palmer.py`, `test_scpdsi.py`, `test_self_calibration.py` |
| Add a fire-weather index | `fire/` subpackage | `test_fire*.py`, reference tests |
| Add a flood-potential index | `flood/` subpackage, `typed_public_api.py` | `test_flood*.py`, `test_main_flood.py` |
| Modify run extraction | `runs.py` | `test_runs.py` |
| Fix CLI bug | `__main__.py`, `_cli.py` | `test_cli_common.py`, `test_main_*.py` |
| Add validation | `validation.py`, `xarray_adapter.py`, `exceptions.py` | `test_input_validation.py`, `test_exceptions.py` |
| Update CF metadata | `cf_metadata_registry.py` | `test_cf_metadata.py`, `test_metadata_validation.py` |
| Performance optimization | `compute.py`, `performance.py`, chunking strategies | `test_benchmark_*.py` |

---

**Next Steps**: See [architecture.md](./architecture.md) for detailed technical architecture, [development-guide.md](./development-guide.md) for setup instructions, and [reference.rst](./reference.rst) for API reference details.

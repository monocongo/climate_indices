# Tech Stack

## Core Stack
- **Python Version**: 3.10–3.13 (test matrix covers all)
- **Environment/Packaging**: `uv` for dependency management and virtual environments
- **Build Backend**: `hatchling` (PEP 621 metadata in pyproject.toml)
- **Linting/Formatting**: `ruff` (replaces black, isort, flake8)
- **Static Typing**: `mypy` with strict settings
- **Testing**: `pytest` with coverage reporting

## Scientific Computing Stack
- **Array Operations**: `numpy>=1.26.4`
- **JIT Compilation**: `numba>=0.60.0` (performance-critical loops)
- **Statistical Distributions**: `scipy>=1.15.3`
- **Multi-dimensional Data**: `xarray>=2025.6.1`
- **Parallel Processing**: `dask>=2025.7.0`
- **NetCDF I/O**: `h5netcdf>=1.6.3`
- **Time Handling**: `cftime>=1.6.4`

## Development Dependencies
- `pytest`, `pytest-cov`: Testing and coverage
- `ruff`: Linting and formatting
- `mypy`: Type checking
- `pre-commit`: Git hooks
- `sphinx`, `sphinx-rtd-theme`: Documentation

## Dependencies to Use (Do Not Introduce Alternatives)
| Category | Use | Do NOT Use |
|----------|-----|------------|
| Arrays | `numpy` | — |
| Statistics | `scipy.stats` | `statsmodels` |
| DataFrames | `xarray` (for NetCDF) | `pandas` (except for simple I/O) |
| JIT | `numba` | `cython`, `pythran` |
| Parallel | `dask` | `multiprocessing` (deprecated in CLI) |
| NetCDF | `h5netcdf` | `netCDF4` |

## Configuration Files
- `pyproject.toml`: All tool configuration (ruff, mypy, pytest, hatch)
- `.pre-commit-config.yaml`: Git hooks
- No standalone config files (no `setup.cfg`, `mypy.ini`, etc.)

# climate_indices — Development Context

## Project overview
Python library for climate drought index computation (SPI, SPEI, PET, and more).
Scientific computing stack: NumPy, xarray/dask, scipy, structlog.
Repository: https://github.com/monocongo/climate_indices

## Coding conventions
- Exception hierarchy rooted at ClimateIndicesError (see src/climate_indices/exceptions.py)
- structlog for all logging — never use stdlib logging
- xarray support via @overload + adapter pattern (see src/climate_indices/xarray_adapter.py)
- Type hints on all public functions
- Google-style docstrings with description, Args, and Returns sections
- Tests in tests/ using pytest, reference data in tests/fixtures/
- Conventional commits: feat:, fix:, test:, docs:

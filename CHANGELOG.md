# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **xarray DataArray API (Beta)**: Native xarray support for `spi()`, `spei()`,
  `pet_thornthwaite()`, and `pet_hargreaves()` — marked as beta/experimental.
  The xarray interface (parameter inference, metadata, coordinate handling) may
  change in future minor releases. Computation results are identical to the
  stable NumPy API. No breaking changes within minor versions.
- **`BetaFeatureWarning`**: New warning class for beta/experimental features
  (subclass of `ClimateIndicesWarning`)
- **Docker Support**: Dockerfile for containerized deployment (#586)
- **`.dockerignore`**: Optimized Docker builds by excluding unnecessary files
- **PyPI Release Guide**: Comprehensive release documentation (`docs/pypi_release_guide.md`, `docs/pypi_release.rst`)
- **Floating Point Best Practices Guide**: Documentation for safe numerical comparisons (`docs/floating_point_best_practices.md`)
- **Test Fixture Management Guide**: Documentation for test data management (`docs/test_fixture_management.md`)
- **Visualization Notebook**: New notebook for precipitation/SPI visualization (`notebooks/visualize_precip_spi.ipynb`)
- **Lock File**: Added `uv.lock` for reproducible dependency resolution
- **Documentation**: Supported Python versions table and deprecation policy in README

### Changed

- **CI/CD**: Enhanced test matrix with Python 3.10-3.13 on Linux and macOS
- **CI/CD**: Added minimum dependency version testing (`--resolution lowest-direct`)
- **CI/CD**: Added ruff and mypy checks as CI lint job
- **CI/CD**: Modernized all GitHub Actions to v4/v5 versions
- **GitHub Actions**: Updated unit tests workflow with improved configuration
- **Documentation Index**: Reorganized Sphinx documentation structure
- **Notebooks**: Improved examples in existing Jupyter notebooks

### Removed

- **`.pypirc`**: Removed from repository (should be user-specific in `~/.pypirc`)

## [2.2.0] - 2025-08-03

### Added

- **Exception-Based Error Handling**: New robust exception hierarchy for distribution fitting failures
  - `DistributionFittingError` (base class)  
  - `InsufficientDataError` - raised when too few non-zero values for statistical fitting
  - `PearsonFittingError` - raised when L-moments calculation fails
- **Migration Guide**: Comprehensive v2.2.0 migration documentation in README
- **Code Quality Improvements**: Safe floating point comparison guidelines using `numpy.isclose()`
- **Enhanced Test Coverage**: Comprehensive tests for exception handling and fallback behavior
- **Documentation**: 
  - Floating point best practices guide (`docs/floating_point_best_practices.md`)
  - Working examples for safe numerical comparisons
  - Updated build configuration documentation

### Changed

- **Major Dependency Updates**: Updated all packages to latest versions
  - `scipy>=1.15.3` (from 1.14.1) - requires Python 3.10+
  - `dask>=2025.7.0`, `xarray>=2025.6.1`, `h5netcdf>=1.6.3`
  - `pytest>=8.4.1`, `ruff>=0.12.7`, `sphinx>=8.1.3`
- **Build System**: Consolidated and optimized hatch build configuration
  - Reduced package size from 37MB to 207KB (99.4% reduction)
  - Eliminated duplicate exclude lists between sdist and wheel builds
- **Error Handling Architecture**: 
  - Replaced `None` tuple anti-pattern with explicit exceptions
  - Consolidated fallback logic into `DistributionFallbackStrategy` class
  - Improved error messages with detailed context and suggestions
- **Python Version Support**: Dropped Python 3.9, now requires Python 3.10+
- **Floating Point Comparisons**: Replaced direct equality checks with `numpy.isclose()` throughout test suite

### Fixed

- **NumPy 2.0 Compatibility**: 
  - Fixed deprecated `newshape` parameter usage
  - Fixed array-to-scalar conversion warnings
- **Consecutive Zero Precipitation**: Enhanced handling of extensive zero precipitation patterns
- **Test Reliability**: Improved floating point comparison robustness in test assertions
- **Build Exclusions**: Properly exclude development files from distribution packages

### Technical Improvements

- **Code Quality**: Addressed `python:S1244` floating point equality issues
- **Test Architecture**: Enhanced coverage for edge cases and error conditions
- **Logging**: Consistent warning messages for high failure rates in distribution fitting
- **Documentation**: Clear upgrade path for library integrators using internal functions

## [2.1.1] - 2025-01-15

### Added

- **`DistributionFallbackStrategy` class**: Centralized fallback logic for Pearson→Gamma distribution fallbacks
- **Custom exception hierarchy**: `DistributionFittingError`, `InsufficientDataError`, `PearsonFittingError`
- **Comprehensive test coverage**: New test case for distribution fallback strategy consolidation

### Changed

- **Error handling architecture**: Replaced `None` tuple anti-pattern with explicit exception-based error handling
- **Fallback logic**: Consolidated scattered Pearson→Gamma fallback code into single strategy class
- **Logging**: Standardized warning messages for distribution fitting failures

### Fixed

- **Type safety**: Improved error propagation with typed exceptions instead of implicit `None` checks
- **Code maintainability**: Simplified control flow by eliminating complex `None` checking logic

## [2.0.0] - 2023-07-15

### Added

- GitHub Action workflow which performs unit testing on the four supported versions of Python (3.8, 3.9, 3.10, and 3.11)

### Fixed

- L-moments-related errors (#512)
- Various cleanups and formatting indentations

### Changed

- Build and dependency management now using poetry instead of setuptools
- Documentation around installation with examples (#521) 

### Removed

- Palmer indices (these were always half-baked and nobody ever showed any interest in developing them further)
- Numba integration (see [this discussion](https://github.com/monocongo/climate_indices/discussions/502#discussioncomment-6377732)
  for context)
- requirements.txt (dependencies now specified solely in pyproject.toml)
- setup.py (now using poetry as the build tool)

[unreleased]: https://github.com/monocongo/climate_indices/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/monocongo/climate_indices/compare/v2.1.1...v2.2.0
[2.1.1]: https://github.com/monocongo/climate_indices/compare/v2.0.0...v2.1.1
[2.0.0]: https://github.com/monocongo/climate_indices/releases/tag/v2.0.0
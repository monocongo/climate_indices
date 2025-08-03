# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [2.1.1]

### Added

 ✅ Architectural Improvements Complete

  #### Problem 1: Scattered Pearson→Gamma Fallback Logic ❌ → ✅ SOLVED

  __Before__: Fallback logic was spread across multiple functions:
  - indices.spi() - ad-hoc fallback when Pearson fitting failed
  - compute.calculate_time_step_params() - individual failure handling
  - Inconsistent logging and threshold checking

  __After__: Consolidated into DistributionFallbackStrategy class:
  ```python
class DistributionFallbackStrategy:
    def should_fallback_from_excessive_nans(self, values) -> bool:
        pass
    def should_warn_high_failure_rate(self, failure_count, total_count) -> bool:
        pass
    def log_fallback_warning(self, reason, context="") -> None:
        pass
    def log_high_failure_rate(self, failure_count, total_count, context="") -> None:
        pass
  ```
  Benefits:
  - ✅ Single source of truth for all fallback decisions
  - ✅ Configurable thresholds (max_nan_percentage=0.5, high_failure_threshold=0.8)
  - ✅ Consistent logging format across the codebase
  - ✅ Easy to test and modify fallback behavior

  ### Problem 2: None Tuple Anti-Pattern ❌ → ✅ SOLVED

  __Before__: calculate_time_step_params() returned (None, None, None, None) on failure:
  #### OLD CODE - ANTI-PATTERN
  ```python
def calculate_time_step_params(time_step_values):
    if insufficient_data:
      return None, None, None, None  # ❌ Requires downstream None checks
    # ... computation ...
    if fitting_failed:
      return None, None, None, None  # ❌ Obscures failure reason
  ```
  __After__: Exception-based error handling with dedicated exception types:
  #### NEW CODE - EXPLICIT EXCEPTIONS
  ```python
def calculate_time_step_params(time_step_values):
    if insufficient_data:
      raise InsufficientDataError(message, non_zero_count, required_count)  # ✅ Clear failure reason
    # ... computation ...
    if fitting_failed:
      raise PearsonFittingError(message, underlying_error)  # ✅ Specific error type
  ```
  #### Custom Exception Hierarchy:
```
  DistributionFittingError (base)
  ├── InsufficientDataError (too few non-zero values)
  └── PearsonFittingError (L-moments computation failed)
```
  __Benefits__:
  - ✅ Explicit Error Handling: No more implicit None checks
  - ✅ Rich Error Information: Exceptions carry detailed context
  - ✅ Type Safety: Clear distinction between different failure modes
  - ✅ Simplified Control Flow: Exception handling eliminates complex None checking logic

  #### Updated Architecture Flow

  **compute.py**:
  calculate_time_step_params() raises InsufficientDataError | PearsonFittingError

                  ↓

  pearson_parameters() catches DistributionFittingError → uses default values

                  ↓

  Uses _default_fallback_strategy.should_warn_high_failure_rate()

  **indices.py**:
  spi() calls transform_fitted_pearson()

                  ↓

  Catches DistributionFittingError, ValueError, Warning

                  ↓

  Uses _fallback_strategy.should_fallback_from_excessive_nans()

                  ↓

  Uses _fallback_strategy.log_fallback_warning() → falls back to Gamma

  ### Verification Results

✅ All 8 existing zero precipitation tests pass 

✅ All 5 main indices tests pass (backward compatibility maintained)

✅ New test case **test_distribution_fallback_strategy_consolidation()** verifies:
  - Strategy methods work correctly
  - Custom exceptions carry proper information
  - End-to-end SPI computation handles exceptions gracefully
  - Fallback logic is consistently applied

  ### Key Improvements Achieved

  1. 🏗️ Better Architecture: Single responsibility principle - each component has clear error handling
  2. 🔧 Maintainability: Centralized fallback logic makes future changes easy
  3. 🐛 Debugging: Explicit exceptions make failure diagnosis straightforward
  4. 🧪 Testability: Strategy pattern allows isolated testing of fallback behavior
  5. 📖 Readability: Code intent is clearer without None tuple anti-patterns
  6. 🔒 Type Safety: Exception types provide compile-time guarantees about error handling

  The codebase now follows modern Python error handling best practices with clear separation of concerns and explicit error propagation, making it much more maintainable
  and robust.



## [Unreleased] (latest master branch)

### Added

- something we've added, coming soon
- something else we've added

### Fixed

- something we've fixed (#issue_number)
- something else we've fixed (#issue_number)

### Changed

- something we've changed (#issue_number)

### Removed

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

[unreleased]: https://github.com/monocongo/climate_indices/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/monocongo/climate_indices/releases/tag/v2.0.0
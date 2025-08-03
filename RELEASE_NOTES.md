# climate_indices Release Notes

**GitHub Repository:** [https://github.com/monocongo/climate_indices](https://github.com/monocongo/climate_indices)

---

# [2.2.0]

A new version of climate_indices has been released, introducing several significant updates and improvements. This release focuses on enhanced error handling, code quality improvements, and modernized dependencies while maintaining backward compatibility for public APIs.

## 🚀 Added

### Exception-Based Error Handling

Introduced a new robust exception hierarchy for distribution fitting failures:

- **`DistributionFittingError`** (base class)
- **`InsufficientDataError`**: Raised when there are insufficient non-zero values for statistical fitting
- **`PearsonFittingError`**: Raised when L-moments calculations fail

### Migration Guide

Detailed v2.2.0 migration documentation is now available in the README, providing clear upgrade paths for library integrators.

### Code Quality Enhancements

Incorporated safe floating-point comparison guidelines using `numpy.isclose()`, addressing `python:S1244` code quality standards.

### Enhanced Test Coverage

Added comprehensive tests for exception handling and fallback behavior, ensuring robust error recovery mechanisms.

### Documentation Updates

- **Floating-point best practices guide** (`docs/floating_point_best_practices.md`)
- **Working examples** for safe numerical comparisons (`examples/floating_point_comparisons.py`)
- **Updated build configuration** documentation

## 🔄 Changed

### Major Dependency Updates

Upgraded all packages to latest versions:

- **`scipy>=1.15.3`** (previously 1.14.1; now requires Python 3.10+)
- **`dask>=2025.7.0`**, **`xarray>=2025.6.1`**, **`h5netcdf>=1.6.3`**
- **`pytest>=8.4.1`**, **`ruff>=0.12.7`**, **`sphinx>=8.1.3`**

### Build System Improvements

Improved and consolidated build configuration:

- **Reduced package size by 99.4%** (from 37MB to 207KB)
- **Consolidated and simplified** exclude lists for sdist and wheel builds

### Error Handling Architecture

- **Replaced `None` tuple anti-pattern** with explicit exceptions
- **Consolidated fallback logic** into a new `DistributionFallbackStrategy` class
- **Enhanced error messages** with detailed context and actionable suggestions

### Python Version Support

**Dropped support for Python 3.9**; Python 3.10+ is now required.

### Floating-Point Comparisons

**Replaced direct equality checks** with `numpy.isclose()` throughout the test suite for improved numerical robustness.

## 🐛 Fixed

### NumPy 2.0 Compatibility

- **Resolved deprecated `newshape` parameter usage**
- **Addressed array-to-scalar conversion warnings**

### Zero Precipitation Handling

**Improved logic** for handling extensive zero precipitation patterns in climate data processing.

### Test Reliability

**Enhanced robustness** of floating-point comparison assertions in tests, eliminating precision-related test failures.

### Build Exclusions

**Ensured proper exclusion** of development files from distribution packages.

## 🔧 Technical Improvements

- **Resolved issues** related to `python:S1244` floating-point equality warnings
- **Extended test architecture** to cover additional edge cases and error conditions
- **Standardized warning messages** for high failure rates in distribution fitting
- **Clarified upgrade paths** for library integrators using internal functions

---

## 📚 Migration Information

### For Direct API Users
**No changes needed** - the public SPI/SPEI functions handle exceptions internally and maintain backward compatibility.

### For Library Integrators
If you were checking for `None` return values from internal functions, update to use try/catch blocks:

```python
# OLD: Checking for None tuples
if result == (None, None, None, None):
    handle_error()

# NEW: Exception handling
try:
    result = internal_function(data)
except climate_indices.compute.InsufficientDataError as e:
    handle_insufficient_data(e)
except climate_indices.compute.PearsonFittingError as e:
    handle_fitting_failure(e)
```

----

## 🔗 Links

- **Documentation**: [https://climate-indices.readthedocs.io/](https://climate-indices.readthedocs.io/)
- **PyPI Package**: [https://pypi.org/project/climate-indices/](https://pypi.org/project/climate-indices/)
- **Issues & Bug Reports**: [https://github.com/monocongo/climate_indices/issues](https://github.com/monocongo/climate_indices/issues)

For complete details and technical specifications, refer to the [CHANGELOG.md](CHANGELOG.md) and project documentation.
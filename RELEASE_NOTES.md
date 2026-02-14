# climate_indices Release Notes

**GitHub Repository:** [https://github.com/monocongo/climate_indices](https://github.com/monocongo/climate_indices)

---

# [2.3.0] - Recommended Version Bump: Minor (2.2.0 → 2.3.0)

**Recommendation Rationale:** This release introduces substantial new functionality including native xarray support, structured logging infrastructure, and a comprehensive exception hierarchy. The xarray API is entirely additive with typed overloads, maintaining full backward compatibility with the existing NumPy interface. New required dependency (structlog) and significant API surface expansion justify a minor version bump under semantic versioning.

**Migration Impact:** Zero-effort upgrade for existing users. All new features use defaults or keyword-only parameters. Exception classes moved to dedicated module but remain re-exported from `compute.py` for backward compatibility.

---

## 🚀 New Features

### Native xarray DataArray Support (Beta)

The library now provides **first-class xarray integration** for all core climate indices, enabling seamless processing of labeled, multi-dimensional climate datasets (#605, #606):

- **`spi()`**: Accepts xarray DataArray alongside NumPy arrays with automatic parameter inference
- **`spei()`**: Multi-input alignment for precipitation and PET DataArrays
- **`pet_thornthwaite()`**: Full xarray support with CF-compliant metadata
- **`pet_hargreaves()`**: Coordinate-aware computation preserving dimension labels

**Beta Status:** The xarray interface (parameter inference, metadata attributes, coordinate handling) is marked as beta/experimental and may evolve in future minor releases. Computation results are identical to the stable NumPy API. A new `BetaFeatureWarning` is emitted on first use (suppressible via standard warning filters).

**Key capabilities:**
- **Automatic parameter inference** from DataArray coordinates (`time_coord_name`, `scale` detection)
- **CF-1.8 compliant metadata** with provenance tracking via `history` attribute
- **Coordinate preservation** maintaining spatial and temporal labels through computation
- **Dask-backed lazy evaluation** for out-of-core processing of large climate datasets
- **Multi-input alignment** with automatic time coordinate intersection and missing data warnings
- **Type-safe API** using `@overload` declarations for IDE autocomplete and type checking

### Structured Exception Hierarchy

Comprehensive exception system replacing error-prone `None` return value patterns (#600):

**New exception classes** (all subclass `ClimateIndicesError` for unified error handling):

- **`ClimateIndicesError`**: Base exception for all library errors
- **`DistributionFittingError`**: Base for statistical fitting failures with diagnostic attributes
  - **`InsufficientDataError`**: Raised when too few non-zero values for distribution fitting (includes `non_zero_count`, `required_count`)
  - **`PearsonFittingError`**: Raised when L-moments calculation fails (wraps underlying numerical errors)
- **`DimensionMismatchError`**: Array shape incompatibility errors
- **`CoordinateValidationError`**: Invalid or missing coordinate values
- **`InputTypeError`**: Type validation failures
- **`InvalidArgumentError`**: Argument value out of valid range/set

**Enhanced error messages** include:
- Distribution name and input array shape
- Summarized parameter values at failure point
- Actionable suggestions (e.g., "Try increasing calibration period length")
- Wrapped underlying exceptions for debugging

### Structured Logging with structlog

Production-grade structured logging infrastructure replacing ad-hoc print statements (#600):

- **New dependency**: `structlog>=24.1.0` (required)
- **`configure_logging()`**: Environment-aware logger setup
  - Console: Human-readable with colorization (development)
  - File: JSON-serialized events with ISO timestamps (production)
- **Event types**: `calculation_started`, `calculation_completed`, `calculation_failed`, `distribution_fitting_failed`
- **Automatic context**: Input shapes, scale parameters, distribution names, execution time
- **Performance metrics**: Input size, memory usage (when `psutil` available)
- **Log aggregator compatible**: JSON output works with ELK, Splunk, CloudWatch

**Optional performance dependency group:**
```toml
[project.optional-dependencies]
performance = ["psutil>=5.9.0"]
```

### Data Quality Warning System

Fine-grained warning hierarchy for non-fatal data quality issues (#600, #605):

- **`ClimateIndicesWarning`**: Base class enabling bulk filtering
- **`MissingDataWarning`**: Calibration period exceeds missing data threshold
- **`ShortCalibrationWarning`**: Calibration period shorter than recommended
- **`GoodnessOfFitWarning`**: Poor distribution fit quality (aggregates across time steps)
- **`InputAlignmentWarning`**: Multi-input alignment dropped time steps
- **`BetaFeatureWarning`**: Beta/experimental feature usage

**All warnings include structured attributes** (e.g., `missing_ratio`, `threshold`, `actual_years`) enabling programmatic filtering and monitoring.

### Deprecation Warning Infrastructure

Standardized deprecation system for future API evolution (#612):

- **`ClimateIndicesDeprecationWarning`**: Dual inheritance from `ClimateIndicesWarning` and `DeprecationWarning`
- **`emit_deprecation_warning()`**: Helper for consistent deprecation messages
- **Automatic URL construction** for migration guides
- **Filterability** by either `ClimateIndicesWarning` or `DeprecationWarning` category

### Type-Safe Public API

Typed overload signatures for improved IDE support and static analysis (#605, #612):

- **`typed_public_api.py`**: Explicit `@overload` declarations for `spi()` and `spei()`
- **NumPy path**: `spi(precips: np.ndarray, ...) -> np.ndarray`
- **xarray path**: `spi(precips: xr.DataArray, ...) -> xr.DataArray`
- **Strict mypy compliance** for all core modules

---

## 📈 Improvements

### CI/CD Enhancements

Comprehensive test matrix and quality gates (#607, #612):

- **Multi-platform matrix**: Python 3.10–3.13 on Linux and macOS
- **Minimum dependency testing**: `uv sync --resolution lowest-direct` to catch compatibility regressions
- **Lint job**: Automated ruff and mypy checks in CI pipeline
- **GitHub Actions modernization**: All workflows upgraded to v4/v5 versions
- **Security hardening**: Actions pinned to commit SHAs for supply chain security
- **Benchmark workflow**: Performance regression detection (#607)

### Documentation

Substantial documentation expansion for both users and maintainers (#607, #612):

**New guides:**
- **PyPI Release Guide** (`docs/pypi_release_guide.md`, `docs/pypi_release.rst`): Step-by-step release checklist
- **Floating Point Best Practices** (`docs/floating_point_best_practices.md`): Safe numerical comparison patterns
- **Test Fixture Management** (`docs/test_fixture_management.md`): Test data organization
- **xarray Migration Guide** (Sphinx docs): NumPy → xarray transition path
- **Algorithm Documentation** (Sphinx docs): Scientific references for SPI, SPEI, PET
- **Troubleshooting Guide** (Sphinx docs): Common errors and performance tuning

**Sphinx improvements:**
- **Complete API reference** for all 11 modules
- **Quickstart tutorial** with PET, SPI, and SPEI examples
- **Reorganized index** with improved navigation
- Docstring formatting standardization

**README enhancements:**
- **Supported Python versions table** (3.10–3.13)
- **Deprecation policy statement**
- Fixed image/link rendering for PyPI

### Type Annotations

Comprehensive type hint coverage across codebase (#612):

- **`compute.py`**: Full type annotations (350+ lines)
- **`xarray_adapter.py`**: Complete typed signatures
- **`eto.py`, `utils.py`, `lmoments.py`**: Type hints added
- **`palmer.py`**: Partial annotations
- **`__spi__.py` (CLI)**: Partial annotations
- **mypy strict mode** for `typed_public_api.py`

### Dependency Management

Modern PEP 735 dependency groups with dual compatibility (#607, #612):

- **`uv.lock`** added for reproducible builds
- **`[dependency-groups]`** table for uv compatibility
- **Maintained `[project.optional-dependencies]`** for pip compatibility
- **Security updates**: urllib3, cryptography, pillow, jupyterlab, nbconvert, fonttools (#602–#604, #608, #610–#611)

### Build Configuration

Optimized PyPI metadata and distribution excludes (#607):

- **Improved classifiers** for better PyPI discoverability
- **Updated keywords** (climate, drought, SPI, SPEI, PET, xarray)
- **Consolidated exclude lists** (`.claude/`, `.github/`, etc.)
- **Removed `.pypirc`** from repository (should be user-specific)

---

## 🐛 Bug Fixes

### SPI Zero Precipitation Handling

Resolved edge case where 100% zero precipitation in calibration period caused incorrect results (#594, #595):

- **Fix**: Return `NaN` for SPI when history contains only zeros (no valid distribution fit possible)
- **Fix**: Centralized zero-handling logic to eliminate code duplication
- **Fix**: Replace `==` with `np.isclose()` for floating-point equality checks (addresses SonarCloud S1244)

### Hargreaves PET Array Reshaping

Fixed incorrect array reshaping in `pet_hargreaves()` causing dimension errors (commit `ed7c8f0`).

### Flaky Test Stabilization

Eliminated non-deterministic test failures in CI (#607, #606):

- **Thornthwaite PET monotonicity test**: Resolved floating-point precision sensitivity (commit `65837e7`)
- **Benchmark tests**: Switched to `timeit.repeat() + min()` for stable measurements
- **xarray timestamp assertions**: Eliminated timezone/precision mismatches
- **psutil-unavailable test**: Fixed resource cleanup race condition
- **PET overhead threshold**: Adjusted to 80% for CI environment variability

### CI Import Errors

Fixed `ImportError` in GitHub Actions test workflow (#601).

---

## 🔧 Internal / Development

### Test Infrastructure

Expanded test coverage with advanced validation techniques (#607):

**Property-based testing:**
- **Hypothesis integration** for SPI/SPEI/PET invariant testing
- **Parametric tests** for distribution edge cases

**Performance benchmarks:**
- **pytest-benchmark** suite for xarray adapter overhead
- **Memory efficiency tests** validating out-of-core Dask computation
- **Weak scaling benchmarks** for parallel efficiency

**Validation datasets:**
- **NOAA EDDI reference data** validation (Story 4.4)
- **Metadata integration tests** (CF compliance, coordinate preservation)

**Test organization:**
- **Consolidated fixtures** in `conftest.py` (Story 4.5)
- **Eliminated fixture shadowing** across test modules
- **Shared test data** for consistent validation

### BMAD Planning Artifacts

Comprehensive architectural documentation for development planning (#599):

- **Product Requirements Document (PRD)**: 60 functional requirements across 5 epics
- **Architecture Decision Document**: xarray + structlog technical justification
- **Epic breakdown**: 47 stories with acceptance criteria
- **Implementation readiness reports**: Pre-implementation validation

**Artifacts location:** `_bmad-output/planning-artifacts/` (excluded from distribution)

### Dependency Bumps

Security and feature updates via Dependabot (#602–#604, #608, #610–#611):

- **urllib3**: 2.5.0 → 2.6.3
- **cryptography**: 45.0.5 → 46.0.5
- **pillow**: 11.3.0 → 12.1.1
- **jupyterlab**: 4.4.5 → 4.4.8
- **fonttools**: 4.59.0 → 4.60.2
- **nbconvert**: 7.16.6 → 7.17.0
- **jaraco-context**: → 6.1.0 (CVE-2026-23949 mitigation)

---

## ⚠️ Breaking Changes

**None.** This release maintains full backward compatibility with v2.2.0:

- All new features use defaults or keyword-only parameters
- Exception classes moved to `exceptions.py` but re-exported from `compute.py`
- `DistributionFittingError` now inherits from `ClimateIndicesError` instead of `Exception` directly (catching `Exception` still works)
- NumPy API unchanged; xarray support is additive via `@overload`

---

## 📦 Dependency Changes

### New Required Dependencies

- **`structlog>=24.1.0`**: Structured logging infrastructure

### New Optional Dependencies

- **`psutil>=5.9.0`**: Performance metrics (install via `pip install climate-indices[performance]`)

### Updated Dependencies

All dependencies bumped to latest stable versions. See CHANGELOG.md for full version specifications.

---

## 🔗 Links

- **Documentation**: [https://climate-indices.readthedocs.io/](https://climate-indices.readthedocs.io/)
- **PyPI Package**: [https://pypi.org/project/climate-indices/](https://pypi.org/project/climate-indices/)
- **Issues & Bug Reports**: [https://github.com/monocongo/climate_indices/issues](https://github.com/monocongo/climate_indices/issues)
- **Changelog**: [CHANGELOG.md](https://github.com/monocongo/climate_indices/blob/master/CHANGELOG.md)

---

## 📋 Full PR List (17 merged PRs since v2.2.0)

**Epics:**
- #600: Epic 1 — Error handling & structured logging
- #605: Epic 2 — xarray SPI integration
- #606: Epic 3 — SPEI & PET xarray support
- #607: Epic 4 — QA & validation
- #612: Epic 5 — Documentation & packaging

**Bug Fixes:**
- #594: Fix SPI for 100% zero precipitation (initial fix)
- #595: Fix SPI floating-point equality checks
- #601: Fix CI test ImportError

**Infrastructure:**
- #599: BMAD planning artifacts
- #609: Dependency security hardening and CI follow-up fixes
- #613: Documentation architecture and contribution guide updates

**Dependency Updates (Dependabot):**
- #602: urllib3 security update
- #603: jupyterlab update
- #604: fonttools update
- #608: nbconvert update
- #610: cryptography security update
- #611: pillow security update

---

For detailed technical specifications and complete commit history, see [CHANGELOG.md](CHANGELOG.md).

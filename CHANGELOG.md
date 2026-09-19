# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-09-18

### Added

- **Wildfire index family (`climate_indices.fire`)**: a public namespace with the
  Keetch-Byram Drought Index (KBDI), the CFFWIS moisture codes (FFMC, DMC, DC) and
  behavior indices (ISI, BUI, `cffwis_fwi`, DSR) including Drought Code overwintering
  and seasonal carry, the Fosberg Fire Weather Index, the Hot-Dry-Windy Index, and the
  Haines Index. KBDI, the CFFWIS orchestrator, Hot-Dry-Windy, and Haines ship xarray
  adapters; the elementwise Fosberg FFWI stays on the NumPy layer, where it has no
  dimension to reduce and so no adapter; every index has a CF metadata registry entry.
  `climate_indices --index kbdi` runs KBDI from the command line.
- **Self-calibrated Palmer Drought Severity Index** (`scpdsi`): duration-factor
  fitting, order-statistic self-calibration, and correlation-adaptive least-squares
  fitting, exposed from the Palmer CLI dispatch (#721).
- **Palmer xarray adapter**: `pdsi()` accepts xarray DataArrays and returns the
  PDSI-family outputs as a Dataset; `scpdsi()` remains NumPy-only (#1016).
- **Validation infrastructure**: `VALIDATION.md` records per-index evidence, backed by
  SPEIbase v2.11 SPEI plausibility fixtures (#779), NOAA EDDI reference fixtures, PET
  literature fixtures, and a measured scPDSI calibration anchor.
- **Public `climate_indices.validation` namespace**: the input-preparation and
  validation checks shared across indices are exported from the package root and
  documented as public API.
- **Declared time-major grid blocks**: a three-or-more-dimensional NumPy array runs as
  a `(time, *cells)` block across SPI, SPEI, EDDI, percentage of normal, PET, PDSI, and
  `compute.prepare_scaled()` when declared with `spatial_time_major=True`, fitting or
  ranking every cell in one pass instead of one call per cell (#941, #942).

### Changed

- **The xarray DataArray API stays Beta through 3.0.0** and is promoted no earlier
  than 3.1.0, so the interface may still change in a minor release (ADR-0012).
  Computation results remain identical to the stable NumPy API.
- **Python 3.14 support**: `requires-python` is now `>=3.10,<3.15`, and CI covers the
  new version in the test matrix and in wheel smoke tests.
- **Documentation rebuilt on MyST Markdown** with a four-section reader-need
  navigation, replacing the previous reStructuredText sources.
- **CI**: meta checks run outside the version matrix on the minimum supported Python
  (3.10), validation, lint, docs, notebooks, and the security audit each run once on a
  single pinned interpreter instead of validation re-running across every version, and
  the release workflow installs the built wheel on the boundary Pythons.
- **CLI**: Palmer inputs are converted to the inches `palmer.pdsi()` expects, the PET
  stage now runs for temperature-only SPEI, scaled, and Palmer runs, and `--scales` is
  now required for every scaled index (#1002). `"auto"` chunk axes resolve against
  Dask's default `array.chunk-size` while the input is opened, and a caller-configured
  value is left in place (#925).

### Breaking

Four changes alter computed values or exception types without a deprecation period, and
the console script removed below reaches users without one either: its deprecation
warning was added during the 3.0.0 cycle and shipped in no release. Each behavioral
change states what a user sees, how to detect it, and what to change in
`docs/deprecations/api-changes.md`.

- **Daily xarray calendar alignment**: `spi()`, `spei()`, `eddi()`,
  `percentage_of_normal()`, and `xarray_adapter.pet_hargreaves()` may return
  **different, corrected** daily values for any input spanning a non-leap year. The
  adapter previously passed daily Gregorian values straight into the NumPy core's
  366-day-per-year layout, silently shifting every value after February 28. Inputs on
  a supported Gregorian calendar (`standard`, `gregorian`, `proleptic_gregorian`)
  whose daily coordinates begin on January 1 still succeed, now with corrected
  numbers; unsupported calendars and other origins raise `CoordinateValidationError`
  instead of being silently misinterpreted. The NumPy array API is unaffected
  (ADR-0004).
- **NumPy gridded input shape guard**: `indices.spi()`, `indices.spei()`, and
  `compute.prepare_scaled()` read a three-or-more-dimensional NumPy array as a
  time-major `(time, *cells)` block, and reject the one shape that is ambiguous with a
  `(years, periods, *cells)` array unless it is declared. Reorder the cell axes so the
  first one is not a calendar period length, or pass `spatial_time_major=True`, which
  the package-root `spi()` and `spei()` wrappers accept and pass through (ADR-0009).
- **PCI February correction**: `pci()` returns different values. The cumulative
  day-of-year boundaries had February written as a month length (28 or 29) instead of
  its cumulative index, which left the February slice empty and let March absorb
  February while starting three days early in a non-leap year and two in a leap year.
  The value moves whenever any of those final three January days (the final two in a
  leap year) is non-zero, or when February and March both carry rainfall, so recompute
  PCI values and recalibrate any downstream thresholds tuned against the previous ones
  (#846).
- **Periodicity validation type**: the periodicity check shared by
  `compute.prepare_scaled()`, `compute.transform_fitted_gamma`,
  `compute.transform_fitted_pearson`, `compute.gamma_parameters()`, and
  `compute.reshape_values()`, along with the index functions, now raises
  `PeriodicityError` — a subclass of `InvalidArgumentError`, and not of `ValueError`.
  Error messages are unchanged, so a handler that catches `ValueError` must be widened
  to catch `PeriodicityError` or its parent `InvalidArgumentError` from
  `climate_indices.exceptions`.

### Removed

- **`spi` console script**: removed, along with the `climate_indices.__spi__` module.
  Use `climate_indices --index spi` instead. The deprecation warning was added during
  the 3.0.0 cycle and shipped in no release, so no released version warned before the
  removal; 2.4.0 is the last release that ships the script. The `--save_params` and
  `--load_params` options are retired rather than migrated: fit parameters once with
  `compute.gamma_parameters()` or `compute.pearson_parameters()` and pass them as a
  dict — `{"alpha": ..., "beta": ...}` for gamma — to the `fitting_params` argument of
  `indices.spi()` (#919, #957).

### Fixed

- **Palmer**: duration-factor overrides resolve after input validation, masked fitting
  parameters are treated as missing, fitted coefficients are required to be 12-element
  vectors, mismatched cell grids are rejected, and per-cell available water capacity
  pairing is covered by tests (#721, #906, #1016).
- **Fire**: KBDI percentile window alignment, the Haines xarray adapter's warnings and
  dtype guard, and the demo manifest input guard (#810, #811).
- **CLI**: copied chunk sizes are reordered to match the output dimensions and the
  `h5netcdf` engine is pinned when writing chunked NetCDF output, so `--chunksizes`
  writes the requested layout under the minimum-dependency environment instead of
  failing; shared arrays are reset per invocation. The `spi` deprecation migration
  guidance was corrected (#919).

## [2.4.0] - 2026-04-05

### Added

- **Penman-Monteith ETo (FAO-56)**: New module `src/climate_indices/pm_eto.py` implementing
  the full FAO-56 reference evapotranspiration equation with atmospheric, vapor pressure, and
  humidity pathway helpers. Validated against FAO-56 worked examples (±0.05 mm/day tolerance).
- **EDDI public API**: `eddi()` now exported from `climate_indices` with `@overload` signatures
  in `typed_public_api.py` for both NumPy and xarray DataArray inputs (beta xarray path).
- **CF metadata registry expansion**: Added entries for `eddi`, `pdsi`, `phdi`, `pmdi`, and
  `z_index` to `cf_metadata_registry.py` (registry now has 12 entries).

- **NFR-PATTERN-COVERAGE**: 42/42 compliance points achieved — all 7 indices (SPI, SPEI,
  PET Thornthwaite, PET Hargreaves, PNP, PCI, Palmer) satisfy all 6 canonical patterns
  (CF metadata, typed public API overloads, xarray adapter, structlog lifecycle,
  structured exceptions, property-based tests). Validated by `tests/test_pattern_compliance.py`.
- **xarray DataArray API (Beta)**: Native xarray support for `spi()`, `spei()`,
  `pet_thornthwaite()`, and `pet_hargreaves()` — marked as beta/experimental.
  The xarray interface (parameter inference, metadata, coordinate handling) may
  change in future minor releases. Computation results are identical to the
  stable NumPy API. No breaking changes in patch versions.
- **`BetaFeatureWarning`**: New warning class for beta/experimental features
  (subclass of `ClimateIndicesWarning`)
- **`ClimateIndicesDeprecationWarning`**: New warning class for deprecated features with dual
  inheritance from both `ClimateIndicesWarning` and `DeprecationWarning`, enabling
  filterability by either category. Includes context attributes for deprecation version,
  removal version, alternative, and migration URL
- **`emit_deprecation_warning()`**: Helper function for standardized deprecation messages with
  automatic URL construction and consistent formatting
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

### Notes

- **Palmer xarray wrapper deferred**: `palmer_xarray()` is planned for v2.5.0
  (shipped as 3.0.0) using Pattern C (stack/unpack workaround for xarray Issue #1815,
  see `architecture.md`).
- **NOAA EDDI reference fixtures**: Require manual download; see `tests/fixture/README.md`.
  The `test_noaa_eddi_reference.py` suite skips gracefully when fixtures are absent.

## [2.3.0] - 2026-02-11

### Added

- **xarray DataArray API (Beta)**: Native xarray support for `spi()`, `spei()`,
  `pet_thornthwaite()`, and `pet_hargreaves()` — marked as beta/experimental.
  The xarray interface (parameter inference, metadata, coordinate handling) may
  change in future minor releases. Computation results are identical to the
  stable NumPy API. No breaking changes in patch versions.
- **`BetaFeatureWarning`**: New warning class for beta/experimental features
  (subclass of `ClimateIndicesWarning`)
- **`ClimateIndicesDeprecationWarning`**: New warning class for deprecated features with dual
  inheritance from both `ClimateIndicesWarning` and `DeprecationWarning`, enabling
  filterability by either category. Includes context attributes for deprecation version,
  removal version, alternative, and migration URL
- **`emit_deprecation_warning()`**: Helper function for standardized deprecation messages with
  automatic URL construction and consistent formatting
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
---
stepsCompleted: [1, 2]
inputDocuments:
  - 'feature/bmad-xarray-prd:_bmad-output/planning-artifacts/prd.md'
  - '_bmad-output/planning-artifacts/architecture.md'
---

# climate_indices xarray Integration + structlog Modernization - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for the climate_indices xarray integration and structlog modernization project, decomposing the requirements from the PRD and Architecture documents into implementable stories.

## Requirements Inventory

### Functional Requirements (60 total)

**1. Index Calculation Capabilities (5 FRs)**
- FR-CALC-001: SPI Calculation with xarray
- FR-CALC-002: SPEI Calculation with xarray
- FR-CALC-003: PET Thornthwaite with xarray
- FR-CALC-004: PET Hargreaves with xarray
- FR-CALC-005: Backward Compatibility - NumPy API

**2. Input Data Handling (5 FRs)**
- FR-INPUT-001: Automatic Input Type Detection
- FR-INPUT-002: Coordinate Validation
- FR-INPUT-003: Multi-Input Alignment
- FR-INPUT-004: Missing Data Handling
- FR-INPUT-005: Chunked Array Support

**3. Statistical and Distribution Capabilities (4 FRs)**
- FR-STAT-001: Gamma Distribution Fitting
- FR-STAT-002: Pearson Type III Distribution
- FR-STAT-003: Calibration Period Configuration
- FR-STAT-004: Standardization Transform

**4. Metadata and CF Convention Compliance (5 FRs)**
- FR-META-001: Coordinate Preservation
- FR-META-002: Attribute Preservation
- FR-META-003: CF Convention Compliance
- FR-META-004: Provenance Tracking
- FR-META-005: Chunking Preservation

**5. API and Integration (4 FRs)**
- FR-API-001: Function Signature Consistency
- FR-API-002: Type Hints and Overloads
- FR-API-003: Default Parameter Values
- FR-API-004: Deprecation Warnings

**6. Error Handling and Validation (4 FRs)**
- FR-ERROR-001: Input Validation
- FR-ERROR-002: Computation Error Handling
- FR-ERROR-003: Structured Exceptions
- FR-ERROR-004: Warning Emission

**7. Observability and Logging (5 FRs)**
- FR-LOG-001: Structured Logging Configuration
- FR-LOG-002: Calculation Event Logging
- FR-LOG-003: Error Context Logging
- FR-LOG-004: Performance Metrics
- FR-LOG-005: Log Level Configuration

**8. Testing and Validation (5 FRs)**
- FR-TEST-001: Equivalence Test Framework
- FR-TEST-002: Metadata Validation Tests
- FR-TEST-003: Edge Case Coverage
- FR-TEST-004: Reference Dataset Validation
- FR-TEST-005: Property-Based Testing

**9. Documentation (5 FRs)**
- FR-DOC-001: API Reference Documentation
- FR-DOC-002: xarray Migration Guide
- FR-DOC-003: Quickstart Tutorial
- FR-DOC-004: Algorithm Documentation
- FR-DOC-005: Troubleshooting Guide

**10. Performance and Scalability (4 FRs)**
- FR-PERF-001: Overhead Benchmark
- FR-PERF-002: Chunked Computation Efficiency
- FR-PERF-003: Memory Efficiency
- FR-PERF-004: Parallel Computation

**11. Packaging and Distribution (4 FRs)**
- FR-PKG-001: PyPI Distribution
- FR-PKG-002: Dependency Management
- FR-PKG-003: Version Compatibility
- FR-PKG-004: Beta Tagging

### Non-Functional Requirements (23 total)

**1. Performance (4 NFRs)**
- NFR-PERF-001: Computational Overhead (<5% for in-memory)
- NFR-PERF-002: Chunked Computation Efficiency (>70% scaling to 8 workers)
- NFR-PERF-003: Memory Efficiency (50GB datasets on 16GB RAM)
- NFR-PERF-004: Startup Time (<500ms import)

**2. Reliability (3 NFRs)**
- NFR-REL-001: Numerical Reproducibility (1e-8 tolerance)
- NFR-REL-002: Graceful Degradation (chunk-level failures)
- NFR-REL-003: Version Stability (no changes in minor versions)

**3. Compatibility (3 NFRs)**
- NFR-COMPAT-001: Python Version Support (3.9-3.13)
- NFR-COMPAT-002: Dependency Version Matrix (wide range)
- NFR-COMPAT-003: Backward Compatibility Guarantee (no breaking changes)

**4. Integration (3 NFRs)**
- NFR-INTEG-001: xarray Ecosystem Compatibility (Dask, zarr, cf_xarray)
- NFR-INTEG-002: CF Convention Compliance (cf-checker passes)
- NFR-INTEG-003: structlog Output Format Compatibility (JSON for log aggregators)

**5. Maintainability (5 NFRs)**
- NFR-MAINT-001: Type Coverage (mypy --strict passes)
- NFR-MAINT-002: Test Coverage (>85% line, >80% branch)
- NFR-MAINT-003: Documentation Coverage (100% public API)
- NFR-MAINT-004: Code Quality Standards (ruff, mypy, bandit clean)
- NFR-MAINT-005: Dependency Security (0 high/critical CVEs)

### Architectural Requirements (10 total)

From the Architecture Decision Document:

**Core Architectural Decisions:**
1. **Adapter Pattern**: `@xarray_adapter` decorator wraps existing NumPy functions (Decision 1)
2. **Module Structure**: New `xarray_adapter.py` module, `indices.py` unchanged (Decision 2)
3. **structlog Integration**: Hybrid approach with module-level loggers and context binding at API entry (Decision 3)
4. **Metadata Engine**: Registry pattern with `CF_METADATA` dictionary (Decision 4)
5. **Exception Hierarchy**: New `ClimateIndicesError` base class with re-parented exceptions (Decision 5)
6. **Parameter Inference**: Auto-infer with override capability (Decision 6)
7. **Dependency Strategy**: xarray and structlog as core dependencies (Decision 7)

**Implementation Patterns:**
8. **Adapter Contract**: Extract → infer → compute → rewrap → log (Pattern 1)
9. **structlog Conventions**: DEBUG/INFO/WARNING/ERROR levels with structured context (Pattern 2)
10. **CF Metadata Registry**: Extensible dictionary for index-specific CF attributes (Pattern 4)

### FR Coverage Map

**Epic 1: Foundation — Error Handling and Observability**
- FR-ERROR-001: Input Validation
- FR-ERROR-002: Computation Error Handling
- FR-ERROR-003: Structured Exceptions
- FR-ERROR-004: Warning Emission
- FR-LOG-001: Structured Logging Configuration
- FR-LOG-002: Calculation Event Logging
- FR-LOG-003: Error Context Logging
- FR-LOG-004: Performance Metrics
- FR-LOG-005: Log Level Configuration

**Epic 2: Core xarray Support — SPI Calculation**
- FR-CALC-001: SPI Calculation with xarray
- FR-CALC-005: Backward Compatibility - NumPy API
- FR-INPUT-001: Automatic Input Type Detection
- FR-INPUT-002: Coordinate Validation
- FR-INPUT-003: Multi-Input Alignment
- FR-INPUT-004: Missing Data Handling
- FR-INPUT-005: Chunked Array Support
- FR-STAT-001: Gamma Distribution Fitting
- FR-STAT-003: Calibration Period Configuration
- FR-STAT-004: Standardization Transform
- FR-META-001: Coordinate Preservation
- FR-META-002: Attribute Preservation
- FR-META-003: CF Convention Compliance
- FR-META-004: Provenance Tracking
- FR-META-005: Chunking Preservation
- FR-API-001: Function Signature Consistency
- FR-API-002: Type Hints and Overloads
- FR-API-003: Default Parameter Values

**Epic 3: Extended xarray Support — SPEI and PET**
- FR-CALC-002: SPEI Calculation with xarray
- FR-CALC-003: PET Thornthwaite with xarray
- FR-CALC-004: PET Hargreaves with xarray
- FR-STAT-002: Pearson Type III Distribution
- FR-INPUT-003: Multi-Input Alignment (enhanced for multi-input functions)

**Epic 4: Quality Assurance and Validation**
- FR-TEST-001: Equivalence Test Framework
- FR-TEST-002: Metadata Validation Tests
- FR-TEST-003: Edge Case Coverage
- FR-TEST-004: Reference Dataset Validation
- FR-TEST-005: Property-Based Testing
- FR-PERF-001: Overhead Benchmark
- FR-PERF-002: Chunked Computation Efficiency
- FR-PERF-003: Memory Efficiency
- FR-PERF-004: Parallel Computation

**Epic 5: Documentation and Packaging**
- FR-DOC-001: API Reference Documentation
- FR-DOC-002: xarray Migration Guide
- FR-DOC-003: Quickstart Tutorial
- FR-DOC-004: Algorithm Documentation
- FR-DOC-005: Troubleshooting Guide
- FR-PKG-001: PyPI Distribution
- FR-PKG-002: Dependency Management
- FR-PKG-003: Version Compatibility
- FR-PKG-004: Beta Tagging
- FR-API-004: Deprecation Warnings

**Total FR Coverage: 60/60 ✅**

**Note on NFR Coverage:** All 23 Non-Functional Requirements are addressed as cross-cutting concerns within epic acceptance criteria:
- Performance NFRs (NFR-PERF-001–004): Validated in Epic 4
- Reliability NFRs (NFR-REL-001–003): Enforced in Epic 4 tests
- Compatibility NFRs (NFR-COMPAT-001–003): Validated in Epic 5 CI matrix
- Integration NFRs (NFR-INTEG-001–003): Validated in Epic 4 tests
- Maintainability NFRs (NFR-MAINT-001–005): Enforced across all epics

## Epic List

### Epic 1: Foundation — Error Handling and Observability

Researchers and operational users get structured error messages and comprehensive logging for debugging climate index calculations, improving troubleshooting time by 40%.

**FRs Covered:** FR-ERROR-001, FR-ERROR-002, FR-ERROR-003, FR-ERROR-004, FR-LOG-001, FR-LOG-002, FR-LOG-003, FR-LOG-004, FR-LOG-005

**Architectural Components:**
- `exceptions.py`: `ClimateIndicesError` hierarchy (DistributionFittingError, InsufficientDataError, DimensionMismatchError, etc.)
- `logging_config.py`: structlog configuration with dual JSON + console output
- Integration of logging into existing modules: `compute.py`, `eto.py`, `utils.py`

**Value Delivered:** Improves existing NumPy library immediately with no xarray dependency. Establishes error handling and observability foundation for all future epics.

---

### Epic 2: Core xarray Support — SPI Calculation

Climate researchers can calculate SPI directly on xarray DataArrays with full metadata preservation, eliminating manual `.values` extraction and coordinate re-attachment workflows.

**FRs Covered:** FR-CALC-001, FR-CALC-005, FR-INPUT-001, FR-INPUT-002, FR-INPUT-003, FR-INPUT-004, FR-INPUT-005, FR-STAT-001, FR-STAT-003, FR-STAT-004, FR-META-001, FR-META-002, FR-META-003, FR-META-004, FR-META-005, FR-API-001, FR-API-002, FR-API-003

**Architectural Components:**
- `xarray_adapter.py`: `@xarray_adapter` decorator pattern (extract → infer → compute → rewrap → log)
- `CF_METADATA` registry for SPI attributes (long_name, units, references)
- Parameter inference logic (data_start_year, periodicity)
- Type overloads for NumPy vs xarray dispatch

**Value Delivered:** Complete SPI workflow for both NumPy and xarray users. Establishes adapter infrastructure ready for SPEI/PET.

---

### Epic 3: Extended xarray Support — SPEI and PET

Researchers can calculate SPEI and PET (Thornthwaite + Hargreaves) on xarray DataArrays with automatic multi-input alignment, completing the full drought index toolkit for modern workflows.

**FRs Covered:** FR-CALC-002, FR-CALC-003, FR-CALC-004, FR-STAT-002, FR-INPUT-003 (enhanced)

**Architectural Components:**
- Extended `xarray_adapter.py` for multi-input functions (SPEI: precip + PET)
- CF metadata for SPEI and PET variables
- Multi-input coordinate alignment validation

**Value Delivered:** Complete multi-index calculation capability. Establishes pattern for any multi-input index (EDDI in Phase 2).

---

### Epic 4: Quality Assurance and Validation

Automated tests verify numerical equivalence between NumPy and xarray paths, metadata correctness, and edge case handling, giving operational users confidence in upgrading.

**FRs Covered:** FR-TEST-001, FR-TEST-002, FR-TEST-003, FR-TEST-004, FR-TEST-005, FR-PERF-001, FR-PERF-002, FR-PERF-003, FR-PERF-004

**Architectural Components:**
- `test_xarray_adapter.py`: Parametrized equivalence tests (tolerance: 1e-8)
- `test_logging.py`: structlog output validation
- `test_exceptions.py`: Exception hierarchy coverage
- `conftest.py`: xarray fixtures and test utilities
- Benchmark suite: overhead, chunked efficiency, memory, parallelism

**Value Delivered:** Validates all previous epics with comprehensive test coverage. Provides performance baselines and regression detection.

---

### Epic 5: Documentation and Packaging

Users have comprehensive guides, API references, and stable package installation, enabling adoption by graduate students and downstream package maintainers.

**FRs Covered:** FR-DOC-001, FR-DOC-002, FR-DOC-003, FR-DOC-004, FR-DOC-005, FR-PKG-001, FR-PKG-002, FR-PKG-003, FR-PKG-004, FR-API-004

**Architectural Components:**
- Sphinx documentation with xarray-first examples
- Migration guide with side-by-side NumPy/xarray comparisons
- Quickstart tutorial with visualization examples
- Algorithm documentation with peer-reviewed references
- Updated `pyproject.toml` with dependency specifications
- Beta feature warnings in docstrings

**Value Delivered:** Enables community adoption and contribution. Establishes documentation patterns for Phase 2 indices.

---

## Epic 1: Foundation — Error Handling and Observability

Researchers and operational users get structured error messages and comprehensive logging for debugging climate index calculations, improving troubleshooting time by 40%.

### Story 1.1: Custom Exception Hierarchy

As a **library developer**,
I want a unified exception hierarchy for all climate indices errors,
So that users can catch and handle different error types programmatically.

**Acceptance Criteria:**

**Given** the codebase needs structured error handling
**When** I create the `exceptions.py` module
**Then** a base `ClimateIndicesError` exception class exists
**And** existing exceptions are re-parented under `ClimateIndicesError`:
- `DistributionFittingError`
- `InsufficientDataError`
- `PearsonFittingError`
**And** new exceptions are added:
- `DimensionMismatchError`
- `CoordinateValidationError`
- `InputTypeError`
**And** all exceptions include helpful error messages with context
**And** mypy --strict passes on the exceptions module
**And** FR-ERROR-003 is satisfied

---

### Story 1.2: Input Validation Error Handling

As a **climate researcher**,
I want clear error messages when my input data is invalid,
So that I can quickly identify and fix data issues.

**Acceptance Criteria:**

**Given** a user calls an index function with invalid inputs
**When** validation fails (missing time dimension, invalid scale, unsupported distribution)
**Then** a specific exception is raised (not generic ValueError)
**And** the error message includes:
- What validation failed
- Available dimensions/valid ranges
- Suggested remediation
**And** input validation covers:
- Scale in range 1-72
- Distribution in supported set (gamma, pearson3)
- Time dimension presence
**And** FR-ERROR-001 is satisfied

---

### Story 1.3: Computation Error Handling

As an **operational drought monitor**,
I want detailed error context when distribution fitting fails,
So that I can diagnose and resolve computation issues.

**Acceptance Criteria:**

**Given** distribution fitting fails during index calculation
**When** the computation error occurs
**Then** a `DistributionFittingError` is raised
**And** the error message includes:
- Input shape and parameter values
- Which distribution failed (gamma/pearson3)
- Suggested alternative ("try pearson3 distribution")
**And** errors are caught from scipy.stats operations
**And** FR-ERROR-002 is satisfied

---

### Story 1.4: Warning System for Data Quality Issues

As a **climate researcher**,
I want warnings when my data has quality issues,
So that I'm aware of potential problems without blocking my calculation.

**Acceptance Criteria:**

**Given** input data has quality issues
**When** the index calculation runs
**Then** warnings are emitted using `warnings.warn()` (not logging):
- When >20% missing data in calibration period
- When calibration period < 30 years
- When distribution fit has poor goodness-of-fit
**And** warnings are suppressible via `warnings.filterwarnings()`
**And** calculations still complete despite warnings
**And** FR-ERROR-004 is satisfied

---

### Story 1.5: structlog Configuration Module

As a **system administrator**,
I want to configure structured logging with JSON and console outputs,
So that I can integrate climate_indices logs into my monitoring infrastructure.

**Acceptance Criteria:**

**Given** the library needs structured logging
**When** I create the `logging_config.py` module
**Then** a `configure_logging()` function exists that:
- Sets up structlog with dual processors (JSON + console)
- JSON output for file handlers (machine-readable)
- Human-readable colored output for console
- Accepts log level parameter (DEBUG, INFO, WARNING, ERROR)
- Defaults to INFO level
**And** environment variable `CLIMATE_INDICES_LOG_LEVEL` overrides default
**And** no logging to files by default (user-configured)
**And** FR-LOG-001 is satisfied

---

### Story 1.6: Calculation Event Logging

As a **climate researcher**,
I want my index calculations logged with start/completion events,
So that I can track computation progress in long-running workflows.

**Acceptance Criteria:**

**Given** an index calculation is initiated
**When** the calculation starts
**Then** an INFO-level log entry is emitted with:
- Event: "calculation_started"
- Index type (spi, spei, pet_thornthwaite, pet_hargreaves)
- Scale parameter
- Distribution parameter
- Input shape (dimensions)
**When** the calculation completes
**Then** an INFO-level log entry is emitted with:
- Event: "calculation_completed"
- Duration in milliseconds
- Output shape
**And** context is bound at API entry points (not internal functions)
**And** FR-LOG-002 is satisfied

---

### Story 1.7: Error Context Logging

As an **operational drought monitor**,
I want detailed context logged when errors occur,
So that I can perform post-mortem analysis on failures.

**Acceptance Criteria:**

**Given** a computation error occurs
**When** the error is raised
**Then** an ERROR-level log entry is emitted with:
- Full traceback
- Input metadata (shape, coordinates if xarray, chunking)
- Parameter values (scale, distribution, calibration period)
- Event: "calculation_failed"
**And** no data values are logged (privacy + size concerns)
**And** structured log fields enable filtering/aggregation
**And** FR-LOG-003 is satisfied

---

### Story 1.8: Performance Metrics Logging

As a **performance engineer**,
I want computation time and memory usage logged,
So that I can profile and optimize large-scale workflows.

**Acceptance Criteria:**

**Given** an index calculation runs
**When** the calculation completes
**Then** performance metrics are logged:
- Computation time in milliseconds (all calculations)
- Memory usage for arrays > 1GB (if psutil available)
- Input size (element count)
**And** metrics are accessible via structlog context binding
**And** custom metrics can be added via context binding API
**And** FR-LOG-004 is satisfied

---

### Story 1.9: Integrate Logging into Existing Modules

As a **library maintainer**,
I want structured logging integrated into existing NumPy code,
So that current users benefit from improved observability.

**Acceptance Criteria:**

**Given** the logging infrastructure is established
**When** I update existing modules (`compute.py`, `eto.py`, `utils.py`)
**Then** module-level loggers are added: `logger = structlog.get_logger(__name__)`
**And** key operations are logged:
- Distribution fitting start/complete (compute.py)
- PET calculation start/complete (eto.py)
- Array transformation operations (utils.py)
**And** no function signatures change (internal logging only)
**And** existing NumPy tests pass unchanged
**And** FR-LOG-005 is satisfied

---

## Epic 2: Core xarray Support — SPI Calculation

Climate researchers can calculate SPI directly on xarray DataArrays with full metadata preservation, eliminating manual `.values` extraction and coordinate re-attachment workflows.

### Story 2.1: Input Type Detection and Routing

As a **climate researcher**,
I want the library to automatically detect whether I'm using NumPy or xarray,
So that I don't need separate function calls for different input types.

**Acceptance Criteria:**

**Given** the SPI function receives input data
**When** I check the input type
**Then** `isinstance(data, xr.DataArray)` determines routing
**And** xarray inputs route to the xarray adapter path
**And** numpy.ndarray/list/scalar inputs route to the NumPy path
**And** unsupported types (pandas.Series, polars.DataFrame) raise `InputTypeError` with clear message
**And** FR-INPUT-001 is satisfied

---

### Story 2.2: xarray Adapter Decorator Infrastructure

As a **library developer**,
I want a reusable decorator pattern for wrapping NumPy functions,
So that adding xarray support to new indices is straightforward.

**Acceptance Criteria:**

**Given** I need to wrap a NumPy index function
**When** I create the `xarray_adapter.py` module
**Then** an `@xarray_adapter` decorator exists with signature accepting cf_metadata, time_dim, and infer_params parameters
**And** the decorator implements the adapter contract:
1. Extract `.values` from DataArray
2. Infer parameters (data_start_year, periodicity) if enabled
3. Call wrapped NumPy function
4. Rewrap result with coordinates and attributes
5. Log completion event
**And** mypy --strict passes with proper type overloads
**And** Architectural Decision 1 (Adapter Pattern) is implemented

---

### Story 2.3: CF Metadata Registry for SPI

As a **climate researcher**,
I want SPI outputs to have CF-compliant metadata,
So that my results are interoperable with other climate tools.

**Acceptance Criteria:**

**Given** SPI calculation produces xarray output
**When** I define the `CF_METADATA` registry
**Then** an SPI entry exists with long_name, units, and references fields
**And** metadata includes "Standardized Precipitation Index" as long_name
**And** units are "dimensionless"
**And** references include DOI to McKee et al. (1993)
**And** metadata is applied to output DataArray
**And** FR-META-003 (CF compliance) is satisfied
**And** Architectural Decision 4 (Metadata Registry) is implemented

---

### Story 2.4: Coordinate Preservation

As a **climate researcher**,
I want all my input coordinates preserved in the output,
So that I don't lose spatial/temporal reference information.

**Acceptance Criteria:**

**Given** an xarray DataArray with coordinates (time, lat, lon, ensemble)
**When** SPI calculation completes
**Then** output DataArray has identical coordinates to input:
- All dimension coordinates (time, lat, lon)
- All non-dimension coordinates (bounds, auxiliary)
- Coordinate attributes preserved
- Coordinate order maintained
**And** FR-META-001 is satisfied

---

### Story 2.5: Attribute Preservation and Enhancement

As a **climate researcher**,
I want relevant input attributes preserved and index-specific metadata added,
So that I maintain provenance and dataset context.

**Acceptance Criteria:**

**Given** input DataArray has attributes (institution, source, history)
**When** SPI calculation completes
**Then** output DataArray attributes include:
- Preserved: institution, source (global context)
- Added: CF metadata (long_name, units, references)
- Added: calculation metadata (scale, distribution, library version)
- Conflicting attributes overwritten with index-specific values
**And** FR-META-002 is satisfied

---

### Story 2.6: Provenance Tracking in History Attribute

As a **data manager**,
I want calculation provenance recorded in metadata,
So that I can audit and reproduce analyses.

**Acceptance Criteria:**

**Given** SPI calculation on xarray DataArray
**When** the calculation completes
**Then** a `history` attribute is added/appended with:
- ISO 8601 timestamp
- Index type and parameters (e.g., "SPI-3 with gamma distribution")
- Library name and version ("climate_indices v2.0.0")
**And** existing history is preserved (appended, not overwritten)
**And** FR-META-004 is satisfied

---

### Story 2.7: Coordinate Validation

As a **climate researcher**,
I want clear errors when my DataArray lacks required dimensions,
So that I can fix data structure issues quickly.

**Acceptance Criteria:**

**Given** input DataArray is missing required time dimension
**When** SPI validation runs
**Then** a `CoordinateValidationError` is raised with message:
- "Time dimension 'time' not found in input"
- "Available dimensions: [list of actual dims]"
- Suggestion: "Use time_dim parameter to specify custom name"
**And** time coordinate monotonicity is checked
**And** insufficient data (time series too short for scale) raises `InsufficientDataError`
**And** FR-INPUT-002 is satisfied

---

### Story 2.8: Missing Data (NaN) Handling

As a **climate researcher**,
I want NaN values handled consistently with NumPy behavior,
So that missing data doesn't break my workflows.

**Acceptance Criteria:**

**Given** input DataArray contains NaN values
**When** SPI calculation runs
**Then** NaNs propagate through calculations (default behavior)
**And** warning is emitted when >20% of calibration period is NaN
**And** minimum sample size (30 years) is enforced on non-NaN values
**And** output has NaN where input had NaN
**And** FR-INPUT-004 is satisfied

---

### Story 2.9: Dask-Backed Array Support

As a **climate researcher**,
I want SPI to work with Dask arrays for large datasets,
So that I can process data larger than memory.

**Acceptance Criteria:**

**Given** input DataArray is backed by dask.array
**When** SPI calculation runs
**Then** computation remains lazy (no automatic `.compute()`)
**And** `apply_ufunc` is used with `dask='parallelized'`
**And** input chunking is preserved in output
**And** no automatic rechunking occurs
**And** FR-INPUT-005 is satisfied
**And** FR-META-005 (chunking preservation) is satisfied

---

### Story 2.10: Parameter Inference (data_start_year, periodicity)

As a **climate researcher**,
I want the library to infer temporal parameters from my DataArray,
So that I don't have to manually specify obvious values.

**Acceptance Criteria:**

**Given** input DataArray has a time coordinate
**When** parameter inference is enabled (default)
**Then** `data_start_year` is inferred from `data.time[0].dt.year`
**And** `periodicity` is inferred from `xr.infer_freq(data.time)` (monthly/daily)
**And** calibration period defaults to full time range
**And** explicit parameter values override inferred values
**And** Architectural Decision 6 (Parameter Inference) is implemented

---

### Story 2.11: Type Hints and Overloads for NumPy/xarray Dispatch

As a **Python developer using IDEs**,
I want accurate type hints for SPI function,
So that my IDE provides correct autocomplete and type checking.

**Acceptance Criteria:**

**Given** SPI function accepts both NumPy and xarray inputs
**When** I add type annotations
**Then** `@overload` signatures exist for both paths with proper numpy.ndarray and xarray.DataArray return types
**And** mypy --strict passes with no type errors
**And** IDE autocomplete shows correct return type based on input
**And** FR-API-002 is satisfied

---

### Story 2.12: Backward Compatibility - NumPy Path Unchanged

As an **operational drought monitor**,
I want my existing NumPy-based code to work identically,
So that I can upgrade without breaking production systems.

**Acceptance Criteria:**

**Given** existing NumPy tests from v1.x
**When** SPI is called with numpy.ndarray input
**Then** all existing tests pass without modification
**And** numerical results are bit-exact (tolerance: 1e-8 for float64)
**And** no new required parameters introduced
**And** no deprecation warnings emitted (MVP phase)
**And** `indices.py` module remains completely unchanged
**And** FR-CALC-005 is satisfied
**And** NFR-COMPAT-003 (backward compatibility) is satisfied

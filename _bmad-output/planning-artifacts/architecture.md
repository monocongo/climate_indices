---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
lastStep: 8
status: 'complete'
completedAt: '2026-02-05'
revisedAt: '2026-02-09'
revisionReason: 'Updated for PRD v1.1 - Added EDDI NOAA reference validation requirements (Phase 2)'
inputDocuments:
  - '_bmad-output/planning-artifacts/prd.md (v1.1)'
  - 'docs/floating_point_best_practices.md'
  - 'docs/test_fixture_management.md'
  - 'docs/case-studies/eddi-bmad-retrospective.md'
workflowType: 'architecture'
project_name: 'climate_indices'
user_name: 'James'
date: '2026-02-05'
---

# Architecture Decision Document — climate_indices xarray Integration + structlog Modernization

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements (60 total):**
- Index Calculation (FR-CALC-001–005): xarray support for SPI, SPEI, PET (Thornthwaite + Hargreaves), backward compat
- Input Handling (FR-INPUT-001–005): type detection, coordinate validation, multi-input alignment, NaN handling, Dask chunks
- Statistics (FR-STAT-001–004): gamma fitting, Pearson III, calibration periods, standardization
- Metadata/CF (FR-META-001–005): coordinate preservation, attribute preservation, CF compliance, provenance, chunking
- API Design (FR-API-001–004): consistent signatures, type overloads, defaults, deprecation warnings
- Error Handling (FR-ERROR-001–004): input validation, computation errors, structured exceptions, warnings
- Logging (FR-LOG-001–005): structlog config, calculation events, error context, perf metrics, log levels
- Testing (FR-TEST-001–005): equivalence tests, metadata tests, edge cases, reference datasets, property-based
- Documentation (FR-DOC-001–005): API reference, migration guide, quickstart, algorithm docs, troubleshooting
- Performance (FR-PERF-001–004): overhead benchmarks, chunked efficiency, memory efficiency, parallelism
- Packaging (FR-PKG-001–004): PyPI, dependencies, version compat, beta tagging

**Non-Functional Requirements (23 total):**
- Performance: <5% overhead, >70% Dask scaling, 50GB memory efficiency, <500ms import
- Reliability: numerical reproducibility (1e-8), graceful degradation, version stability
- Compatibility: Python 3.9–3.13, wide NumPy/SciPy/xarray version range, backward compat guarantee
- Integration: xarray ecosystem (Dask, zarr, cf_xarray), CF v1.10 compliance, structlog JSON format
- Maintainability: mypy --strict, >85% coverage, 100% docstring coverage, ruff/bandit clean, no CVEs

**Scale & Complexity:**
- Primary domain: Scientific Python library (API-first, no UI)
- Complexity level: Medium
- Estimated architectural components: 5 (adapter layer, logging layer, metadata engine, exception hierarchy, test infrastructure)

### Existing Architecture Profile

**Source modules (7,067 lines across 9 files):**
- `indices.py` (798 lines): 6 public index functions, all NumPy-in/NumPy-out
- `compute.py` (940 lines): statistical operations, distribution fitting, scaling
- `palmer.py` (912 lines): stateful water balance model (PDSI, PHDI, Z-Index)
- `eto.py` (375 lines): Thornthwaite and Hargreaves PET methods
- `utils.py` (497 lines): array reshaping, calendar transforms, validation
- `lmoments.py` (188 lines): L-moments estimation for Pearson III
- `__main__.py` (1,872 lines): CLI with xarray I/O (extracts .values, wraps results)
- `__spi__.py` (1,477 lines): dedicated SPI CLI pipeline

**Key observation:** CLI already performs the extract-compute-rewrap pattern that the adapter layer will formalize.

### Technical Constraints & Dependencies

1. xarray + dask are already core (not optional) dependencies
2. Python >=3.10,<3.14 (pyproject.toml)
3. scipy >=1.15.3 for statistical operations
4. No type dispatch or adapter patterns exist yet
5. Palmer indices maintain multi-year state (cannot be fully lazy)
6. Existing tests use .npy fixture files (no xarray fixtures yet)

### Cross-Cutting Concerns

1. **Type dispatch**: NumPy/xarray routing affects all public API functions
2. **structlog**: Affects entire call stack (indices → compute → eto/palmer)
3. **Metadata preservation**: New concern for xarray path (CF attributes, coordinates, chunking)
4. **Error handling**: Custom exception hierarchy spans compute + indices
5. **Numerical equivalence testing**: New test infrastructure for dual-path validation

## Technical Foundation (Brownfield — No Starter Template)

Existing decisions locked: Python >=3.10, Hatchling build, pytest, ruff, mypy, GitHub Actions CI.
New additions: structlog (structured logging), xarray adapter pattern (no new external deps for MVP).
xarray + dask remain core dependencies (already required by CLI).

## Core Architectural Decisions

### Decision 1: xarray Adapter Pattern → Decorator-Based
- `@xarray_adapter` decorator wraps existing NumPy functions
- Decorator extracts `.values` from DataArray, calls NumPy function, rewraps with metadata
- Formalizes the extract-compute-rewrap pattern already used in `__main__.py`
- Existing NumPy functions remain completely unchanged
- **Rationale:** DRY, clean separation, independently testable, zero changes to proven code

### Decision 2: Module Structure → New `xarray_adapter.py`
- Dedicated `src/climate_indices/xarray_adapter.py` for all adapter logic
- `indices.py` stays pure NumPy (proven, unchanged)
- Adapter imports index functions from `indices.py` and wraps them
- Public API updated in `__init__.py` or via new exports
- **Rationale:** Clear separation, single discoverable file, easy for contributors

### Decision 3: structlog Integration → Hybrid
- Module-level loggers: `logger = structlog.get_logger(__name__)` in each module
- Context binding at public API entry points (spi, spei, eddi, etc.)
- Bind: index type, scale, input shape, distribution at entry points
- Internal functions use plain module logger (no signature changes)
- New `logging_config.py` for structlog setup (dual JSON + console output)
- **Rationale:** Simple internal use, rich context at API boundary, no breaking changes

### Decision 4: Metadata Engine → Registry Pattern
- Dictionary mapping index names to CF metadata:
  ```python
  CF_METADATA = {
      "spi": {"long_name": "Standardized Precipitation Index", "units": "dimensionless", ...},
      "spei": {"long_name": "Standardized Precipitation Evapotranspiration Index", ...},
  }
  ```
- Extensible for Phase 2/3 indices
- **Rationale:** Minimal code, most maintainable, easily extensible

### Decision 5: Exception Hierarchy → Re-Parent Under New Base
- New `ClimateIndicesError` base exception class
- Re-parent existing exceptions: `DistributionFittingError`, `InsufficientDataError`, `PearsonFittingError`
- Add new exceptions: `DimensionMismatchError`, `CoordinateValidationError`, `InputTypeError`
- **Rationale:** Unified hierarchy allows `except ClimateIndicesError` catch-all

### Decision 6: Parameter Inference → Infer with Override
- Auto-infer `data_start_year` from `data.time[0].dt.year`
- Auto-infer `periodicity` from `xr.infer_freq(data.time)`
- Default calibration to full time range if not specified
- All inferred values overridable via explicit parameters
- **Rationale:** Pragmatic ergonomics — simple API with escape hatch

### Decision 7: Dependency Strategy → Keep xarray as Core
- xarray remains a core dependency (already required by CLI)
- No breaking change to existing install
- structlog added as new core dependency
- **Rationale:** Simplest path, no disruption to existing users

## Implementation Patterns & Consistency Rules

### Pattern 1: xarray Adapter Decorator Contract
- `@xarray_adapter` decorator in `xarray_adapter.py` wraps NumPy functions
- Decorator flow: isinstance check → extract coords/attrs → infer params → call NumPy func → rewrap with CF metadata → log completion
- **Rule:** Never modify `indices.py` functions. Always wrap via decorator.

### Pattern 2: structlog Conventions
- DEBUG: internal computation steps
- INFO: calculation start/complete (include scale, duration_ms, input shape)
- WARNING: data quality issues (high missing ratio, short calibration)
- ERROR: computation failures (distribution fit, convergence)
- **Rule:** Never log data values. Always use `structlog.get_logger(__name__)`.

### Pattern 3: Type Annotations & Overloads
- All public functions MUST have `@overload` for ndarray and DataArray
- Implementation signature uses `np.ndarray | xr.DataArray` union
- Pass `mypy --strict`

### Pattern 4: Error Handling
- Raise `ClimateIndicesError` subclasses (never bare ValueError from new code)
- Warn via `warnings.warn()` for data quality (not logging)
- Error messages: what went wrong + input context + what to try instead

### Pattern 5: Test Structure
- `test_xarray_adapter.py`: equivalence, metadata, inference, Dask compat
- Every adapted function needs: equivalence test (1e-8), metadata test, inference test
- Existing `test_indices.py` unchanged

### Pattern 6: Docstrings
- Google-style with `.. warning:: Beta feature` for xarray
- Args with type descriptions, Returns, Raises, Example sections
- Examples show both NumPy and xarray usage

### Pattern 7: CF Metadata Registry
- `CF_METADATA` dict in `xarray_adapter.py` maps index names to attributes
- Keys: long_name, units, references, standard_name
- **Rule:** Never hard-code CF attributes inline. Always use registry.

### Pattern 8: Reference Dataset Validation Testing (Phase 2)
- Separate test module: `tests/test_reference_validation.py` for validating against published reference datasets
- NOAA reference datasets stored in `tests/data/reference/` with provenance metadata
- EDDI validation tolerance: 1e-5 (looser than equivalence tests due to non-parametric ranking FP accumulation)
- Reference dataset registry tracks source, provenance, format, and expected tolerance
- **Rule:** Reference validation tests are separate from equivalence tests (different tolerances, different failure modes)
- **Rationale:** Phase 2 requirement (FR-TEST-004) - EDDI outputs must validate against NOAA reference data

## Project Structure & Boundaries

### New Files (MVP)
- `src/climate_indices/xarray_adapter.py` — Decorator, CF registry, parameter inference
- `src/climate_indices/logging_config.py` — structlog dual-output configuration
- `src/climate_indices/exceptions.py` — ClimateIndicesError hierarchy
- `tests/test_xarray_adapter.py` — Equivalence, metadata, inference, Dask tests
- `tests/test_logging.py` — structlog configuration and output tests
- `tests/test_exceptions.py` — Exception hierarchy tests

### New Files (Phase 2)
- `tests/test_reference_validation.py` — NOAA reference dataset validation (EDDI tolerance: 1e-5)
- `tests/data/reference/` — Directory structure for reference datasets with provenance tracking
- `tests/data/reference/eddi_noaa_reference.nc` — NOAA EDDI reference outputs for validation

### Modified Files (MVP)
- `src/climate_indices/__init__.py` — Add public re-exports for xarray-aware API
- `src/climate_indices/compute.py` — Re-parent exceptions, add structlog
- `src/climate_indices/eto.py` — Add structlog
- `src/climate_indices/utils.py` — Add structlog
- `tests/conftest.py` — Add xarray fixtures
- `pyproject.toml` — Add structlog dependency

### Unchanged Files (MVP)
- `indices.py`, `palmer.py`, `lmoments.py`, `__main__.py`, `__spi__.py`
- All existing test files

### Architectural Boundaries
1. **NumPy Core ↔ xarray Adapter**: `xarray_adapter.py` dispatches; `indices.py` stays pure NumPy
2. **Logging Layer**: `logging_config.py` configures; all modules use `structlog.get_logger(__name__)`
3. **Exception Hierarchy**: `exceptions.py` defines all; other modules import and raise

### Data Flow (xarray path)
```
User → xarray_adapter.spi(DataArray)
  → isinstance check → extract .values, coords, attrs
  → infer params from time coordinate
  → log start → indices.spi(ndarray) → log complete
  → wrap result as DataArray with CF metadata
  → return DataArray
```

## Architecture Validation

### Coherence: ✅ PASS
All decisions, patterns, and structure are internally consistent with no contradictions.

### Requirements Coverage: ✅ PASS (98%)
- All 60 FRs architecturally supported
- All 23 NFRs addressed or explicitly deferred to Phase 2
- 2 items deferred: formal benchmark suite (Phase 2), documentation planning (separate concern)

### Implementation Readiness: ✅ HIGH CONFIDENCE
- Zero changes to proven NumPy core
- Clean adapter boundary with single responsibility
- structlog is purely additive
- Exception re-parenting is backward compatible
- Phased approach defers complex decisions appropriately

### Implementation Priority Order (MVP)
1. `exceptions.py` — Foundation (other modules import from here)
2. `logging_config.py` — Cross-cutting (needed by all modules)
3. Add structlog to `compute.py`, `eto.py`, `utils.py`
4. `xarray_adapter.py` — Core adapter with SPI first, then SPEI, PET
5. Update `__init__.py` with public API re-exports
6. Update `pyproject.toml` with structlog dependency
7. Tests: `test_exceptions.py`, `test_logging.py`, `test_xarray_adapter.py`
8. Update `conftest.py` with xarray fixtures

## Verification Plan

### How to Test End-to-End
```bash
# Run existing tests (must still pass — backward compat)
uv run pytest tests/test_indices.py tests/test_compute.py tests/test_eto.py -v

# Run new xarray adapter tests
uv run pytest tests/test_xarray_adapter.py -v

# Run structlog tests
uv run pytest tests/test_logging.py -v

# Run exception hierarchy tests
uv run pytest tests/test_exceptions.py -v

# Type checking
uv run mypy src/climate_indices/ --strict

# Linting
ruff check src/climate_indices/ tests/
ruff format --check src/climate_indices/ tests/

# Full suite
uv run pytest tests/ -v --cov=src/climate_indices --cov-report=term
```

### Key Validation Checks (MVP)
- `test_spi_xarray_equivalence`: xarray SPI == NumPy SPI within 1e-8
- `test_metadata_preservation`: coordinates and CF attributes intact
- `test_parameter_inference`: inferred params match explicit values
- `test_backward_compat`: existing NumPy API unchanged
- `test_structlog_json_output`: JSON format parseable
- `test_exception_hierarchy`: ClimateIndicesError catches all subclasses

### Key Validation Checks (Phase 2)
- `test_eddi_noaa_reference_validation`: EDDI outputs match NOAA reference data within 1e-5 tolerance
- Reference dataset provenance documented and verifiable
- Tolerance rationale documented (non-parametric ranking has different FP accumulation than parametric fitting)

## Post-Architecture Actions

### Immediate: Commit Architecture Document
1. Create `_bmad-output/planning-artifacts/` directory
2. Write architecture document to `_bmad-output/planning-artifacts/architecture.md`
3. Stage and commit: `docs(bmad): add architecture decision document for xarray + structlog`

### Next BMAD Workflow Steps
- **Epic/Story Breakdown**: Decompose MVP scope into implementable stories
- **Implementation Readiness Check**: Validate PRD + Architecture alignment before coding
- **Implementation**: Begin with exceptions.py → logging_config.py → xarray_adapter.py

---

## Revision History

### Version 1.0 (2026-02-05)
- Initial architecture complete (8 steps)
- Based on PRD v1.0
- Defined MVP scope: SPI, SPEI, PET + structlog integration
- Established core patterns: decorator-based adapter, registry metadata, hybrid logging

### Version 1.1 (2026-02-09)
- Updated for PRD v1.1 requirements
- **Added Pattern 8**: Reference Dataset Validation Testing (Phase 2)
- **Extended Project Structure**: Added Phase 2 test infrastructure (`test_reference_validation.py`, `tests/data/reference/`)
- **Updated Verification Plan**: Added EDDI NOAA reference validation (tolerance: 1e-5)
- **Rationale**: PRD v1.1 added FR-TEST-004 requirement for EDDI validation against NOAA reference data
- **Impact**: Phase 2-scoped changes only; MVP architecture unchanged

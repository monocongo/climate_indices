# climate_indices v2.4.0 - Epic and Story Breakdown (Expanded)

**Version:** 2.4.0-expanded
**Date:** 2026-02-16
**Status:** Complete - Ready for Implementation

---

## Document Summary

This document provides the **complete** epic and story breakdown for climate_indices v2.4.0, with full acceptance criteria, dependencies, and implementation guidance for all 38 stories across 5 epics.

### Corrections Applied

1. **FR Count:** 31 functional requirements (not 30) — Track 2 has 6 FRs (4 EDDI + 1 PNP + 1 scPDSI)
2. **Palmer Module Location:** Per Architecture Decision 2, `palmer_xarray()` stays IN `palmer.py` (not separate module)
3. **Individual FR Traceability:** Each story explicitly lists FR codes (no ranges)

### Document Structure

- **Epic 1:** Canonical Pattern Completion (12 stories) — Agent Alpha, Phases 0-1
- **Epic 2:** PM-ET Foundation (7 stories) — Agent Beta, Phase 1
- **Epic 3:** EDDI/PNP/scPDSI Coverage (7 stories) — Agent Gamma, Phase 2
- **Epic 4:** Palmer Multi-Output (9 stories) — Agent Delta, Phase 2
- **Epic 5:** Cross-Cutting Validation (3 stories) — Multiple agents, Phases 3-4

**Total:** 38 stories covering 31 FRs and 8 NFRs

---

## Requirements Inventory

### Functional Requirements (31 Total)

**Track 0: Canonical Pattern Completion (12 FRs)**
- FR-PATTERN-001: percentage_of_normal xarray + CF metadata
- FR-PATTERN-002: pci xarray + CF metadata
- FR-PATTERN-003: eto_thornthwaite typed_public_api entry
- FR-PATTERN-004: eto_hargreaves typed_public_api entry
- FR-PATTERN-005: percentage_of_normal typed_public_api entry
- FR-PATTERN-006: pci typed_public_api entry
- FR-PATTERN-007: Palmer structlog migration
- FR-PATTERN-008: eto_thornthwaite structlog lifecycle completion
- FR-PATTERN-009: Structured exceptions for all legacy functions
- FR-PATTERN-010: percentage_of_normal property-based tests
- FR-PATTERN-011: pci property-based tests
- FR-PATTERN-012: Expanded SPEI + Palmer property-based tests

**Track 1: PM-ET Foundation (6 FRs)**
- FR-PM-001: Penman-Monteith FAO56 Core Calculation
- FR-PM-002: Atmospheric Parameter Helpers (Equations 7-8)
- FR-PM-003: Vapor Pressure Helpers (Equations 11-13)
- FR-PM-004: Humidity Pathway Dispatcher (Equations 14-19)
- FR-PM-005: FAO56 Worked Example Validation
- FR-PM-006: PM-ET xarray Adapter

**Track 2: EDDI/PNP/scPDSI (6 FRs)**
- FR-EDDI-001: NOAA Reference Dataset Validation (BLOCKING)
- FR-EDDI-002: EDDI xarray Adapter
- FR-EDDI-003: EDDI CLI Integration (Issue #414)
- FR-EDDI-004: EDDI PET Method Documentation
- FR-PNP-001: PNP xarray Adapter
- FR-SCPDSI-001: scPDSI Stub Interface

**Track 3: Palmer Multi-Output (7 FRs)**
- FR-PALMER-001: palmer_xarray() Manual Wrapper
- FR-PALMER-002: Multi-Output Dataset Return
- FR-PALMER-003: AWC Spatial Parameter Handling
- FR-PALMER-004: params_dict JSON Serialization
- FR-PALMER-005: Palmer CF Metadata Registry
- FR-PALMER-006: typed_public_api @overload Signatures
- FR-PALMER-007: NumPy vs xarray Equivalence Tests

### Non-Functional Requirements (8 Total)

- NFR-PATTERN-EQUIV: Numerical Equivalence During Refactoring (1e-8)
- NFR-PATTERN-COVERAGE: 100% Pattern Compliance Dashboard (6×7=42 points)
- NFR-PATTERN-MAINT: Maintainability Through Consistency
- NFR-PM-PERF: PM-ET Numerical Precision (FAO56 ±0.05 mm/day)
- NFR-EDDI-VAL: EDDI NOAA Reference Validation Tolerance (1e-5)
- NFR-PALMER-SEQ: Palmer Sequential Time Constraint
- NFR-PALMER-PERF: Palmer xarray Performance ≥80% of baseline
- NFR-MULTI-OUT: Multi-Output Pattern Stability (xarray #1815)

---

## Agent Orchestration Guide

### Agent Team Composition

| Agent | Track | Primary Files | Phase |
|-------|-------|---------------|-------|
| **Alpha** | 0 (Patterns) | exceptions.py, cf_metadata_registry.py, palmer.py (lines 1-913), percentage_of_normal, pci, eto | 0-1 |
| **Beta** | 1 (PM-ET) | eto.py, test_penman_monteith.py | 1 |
| **Gamma** | 2 (EDDI) | tests/fixture/noaa-eddi-*/, test_noaa_eddi_reference.py | 2 |
| **Delta** | 3 (Palmer) | palmer.py (xarray wrapper), test_palmer_equivalence.py | 2 |
| **Omega** | Integration | __init__.py, merge validation | Gates |

### Execution Phases

- **Phase 0 (Foundation):** Stories 1.1-1.2 (Alpha) — Duration: 1-2 days
- **Phase 1 (Parallel Core):** Stories 1.3-1.12, 2.1-2.7 (Alpha ∥ Beta) — Duration: 2-3 weeks
- **Phase 2 (Dependent Tracks):** Stories 3.1-3.7, 4.1-4.9 (Gamma ∥ Delta) — Duration: 3-4 weeks
- **Phase 3 (Audit):** Stories 5.1-5.2 (Alpha, Gamma) — Duration: 3-5 days
- **Phase 4 (Final Validation):** Story 5.3 (All agents) — Duration: 2-3 days

### Critical Path

**1.1 → 1.2 → 1.6 → 4.1 → 4.2-4.9** (~5-6 weeks)

Palmer structlog migration (Story 1.6) MUST complete before Palmer xarray work (Story 4.1+).

---

# Epic 1: Canonical Pattern Completion

**Goal:** Apply v2.3.0-established patterns to ALL remaining indices for consistency and maintainability

**Agent:** Alpha
**Track:** 0 (Pattern Completion)
**Duration:** 2-3 weeks
**Story Count:** 12

---

## Story 1.1: Structured Exception Hierarchy Foundation

**Priority:** Critical (BLOCKER for all Track 0 work)
**Agent:** Alpha
**Phase:** 0
**Effort:** 4-6 hours
**FR Coverage:** FR-PATTERN-009

### Description

Create or verify the structured exception hierarchy that will replace generic `ValueError` instances across all legacy functions. This foundational infrastructure must be in place before pattern migration work begins.

### Acceptance Criteria

- [ ] Exception hierarchy exists in `src/climate_indices/exceptions.py` with base class `ClimateIndicesError`
- [ ] Key exception classes defined:
  - `InvalidArgumentError` (for input validation failures)
  - `InsufficientDataError` (for calibration/sample size issues)
  - `ComputationError` (for algorithm failures)
  - `DistributionFittingError` (for statistical fitting failures)
- [ ] All exception classes support keyword-only context attributes (e.g., `shape=`, `expected=`, `received=`)
- [ ] Exception messages provide actionable guidance (not just "invalid value")
- [ ] Test module `tests/test_exceptions.py` validates exception hierarchy and message formatting
- [ ] mypy --strict passes for exception module

### Dependencies

None (foundation story)

### Implementation Notes

- If hierarchy already exists from v1.1, verify completeness and update as needed
- Follow Python exception best practices: inherit from built-in exceptions where appropriate
- Context attributes should use keyword-only arguments for clarity

---

## Story 1.2: CF Metadata Registry Creation

**Priority:** Critical (BLOCKER for Track 0 xarray work)
**Agent:** Alpha
**Phase:** 0
**Effort:** 3-4 hours
**FR Coverage:** FR-PATTERN-001, FR-PATTERN-002, FR-PNP-001

### Description

Create or extend the CF metadata registry module to support all indices requiring xarray adapters in Track 0. This establishes the centralized metadata source before pattern application begins.

### Acceptance Criteria

- [ ] Module `src/climate_indices/cf_metadata_registry.py` exists with `CF_METADATA` dictionary
- [ ] Registry entries for Track 0 indices:
  - `percentage_of_normal`: `long_name="Percent of Normal Precipitation"`, `units="%"`
  - `pci`: `long_name="Precipitation Concentration Index"`, `units=""`
  - `pnp`: `long_name="Percent of Normal Precipitation"`, `units="%"` (alias or same as percentage_of_normal)
- [ ] Each entry includes `references` field with methodology citation
- [ ] Module is importable and dict structure follows existing SPI/SPEI pattern
- [ ] Test in `tests/test_cf_metadata.py` validates registry structure

### Dependencies

- Story 1.1 (exception hierarchy needed for validation)

### Implementation Notes

- If registry already exists, extend with new entries
- Use consistent key naming (snake_case matching function names)
- References should include DOI links where available

---

## Story 1.3: PNP xarray + typed_public_api Integration

**Priority:** High
**Agent:** Alpha
**Phase:** 1
**Effort:** 6-8 hours
**FR Coverage:** FR-PATTERN-001 (or FR-PNP-001 if separate), FR-PATTERN-005

### Description

Apply the `@xarray_adapter` decorator to `percentage_of_normal()` function and add `@overload` type signatures to `typed_public_api.py`. This delivers PNP xarray support needed by Track 2.

### Acceptance Criteria

- [ ] `@xarray_adapter` decorator applied to `percentage_of_normal()` in `src/climate_indices/indices.py`
- [ ] CF metadata automatically attached from registry (Story 1.2)
- [ ] `@overload` signatures added to `typed_public_api.py`:
  - NumPy path: `percentage_of_normal(np.ndarray, ...) -> np.ndarray`
  - xarray path: `percentage_of_normal(xr.DataArray, ...) -> xr.DataArray`
- [ ] Runtime dispatcher uses `isinstance(input, xr.DataArray)` detection
- [ ] Equivalence test `test_percentage_of_normal_xarray_equivalence()` passes (tolerance: 1e-8)
- [ ] mypy --strict passes

### Dependencies

- Story 1.1 (exceptions for input validation)
- Story 1.2 (CF metadata registry)

### Implementation Notes

- Follow existing SPI/SPEI `@xarray_adapter` pattern exactly
- No algorithm changes — pure wrapper application
- Coordinate preservation must be validated in equivalence test

---

## Story 1.4: PCI xarray + typed_public_api Integration

**Priority:** High
**Agent:** Alpha
**Phase:** 1
**Effort:** 6-8 hours
**FR Coverage:** FR-PATTERN-002, FR-PATTERN-006

### Description

Apply the `@xarray_adapter` decorator to `pci()` function and add `@overload` type signatures. PCI requires 365/366 daily values, so input validation is critical.

### Acceptance Criteria

- [ ] `@xarray_adapter` decorator applied to `pci()` in `src/climate_indices/indices.py`
- [ ] CF metadata automatically attached (dimensionless units, Oliver 1980 reference)
- [ ] Input validation: Raises `InvalidArgumentError` if input length ≠ 365 or 366
- [ ] `@overload` signatures added to `typed_public_api.py`:
  - NumPy path: `pci(np.ndarray) -> np.ndarray`
  - xarray path: `pci(xr.DataArray) -> xr.DataArray`
- [ ] Equivalence test `test_pci_xarray_equivalence()` passes (tolerance: 1e-8)
- [ ] mypy --strict passes

### Dependencies

- Story 1.1 (InvalidArgumentError for length validation)
- Story 1.2 (CF metadata registry)

### Implementation Notes

- PCI is computationally simple but has strict input requirements
- Error message should guide users: "PCI requires exactly 365 or 366 daily precipitation values. Received: {len(input)}"

---

## Story 1.5: ETo Helpers typed_public_api Integration

**Priority:** Medium
**Agent:** Alpha
**Phase:** 1
**Effort:** 4-5 hours
**FR Coverage:** FR-PATTERN-003, FR-PATTERN-004

### Description

Add `@overload` type signatures for `eto_thornthwaite()` and `eto_hargreaves()` to `typed_public_api.py`. These functions already have xarray support from v1.1 but lack type safety.

### Acceptance Criteria

- [ ] `@overload` signatures added for `eto_thornthwaite()`:
  - NumPy path: `eto_thornthwaite(np.ndarray, ...) -> np.ndarray`
  - xarray path: `eto_thornthwaite(xr.DataArray, ...) -> xr.DataArray`
- [ ] `@overload` signatures added for `eto_hargreaves()`:
  - NumPy path: `eto_hargreaves(np.ndarray, np.ndarray, ...) -> np.ndarray`
  - xarray path: `eto_hargreaves(xr.DataArray, xr.DataArray, ...) -> xr.DataArray`
- [ ] Runtime dispatchers validate input types correctly
- [ ] mypy --strict passes on `typed_public_api.py`
- [ ] No changes to `eto.py` computation functions (type signatures only)

### Dependencies

None (functions already have xarray support)

### Implementation Notes

- This is purely a type safety enhancement — no functionality changes
- Follow existing SPI/SPEI pattern exactly

---

## Story 1.6: Palmer structlog Migration

**Priority:** Critical (BLOCKS Track 3)
**Agent:** Alpha
**Phase:** 1
**Effort:** 12-16 hours (palmer.py is 912 lines)
**FR Coverage:** FR-PATTERN-007

### Description

Migrate `palmer.py` from stdlib `logging` to `structlog` with complete lifecycle event patterns. This is on the **critical path** — Palmer xarray work (Track 3) cannot begin until this completes.

### Acceptance Criteria

- [ ] Import changed: `from climate_indices.logging_config import get_logger` replaces `utils.get_logger(__name__, logging.DEBUG)`
- [ ] Lifecycle events added:
  - `calculation_started` with bind context: `calculation="pdsi"`, `data_shape=precips.shape`, `awc=awc`
  - `calculation_completed` with `duration_ms=elapsed`
  - `calculation_failed` with error context
- [ ] All log statements updated to structlog format (no stdlib logging imports remain)
- [ ] Log levels match SPI/SPEI pattern: INFO for lifecycle, DEBUG for internal state
- [ ] Existing functionality unchanged (pure logging refactor)
- [ ] All Palmer tests pass without modification (numerical equivalence maintained)

### Dependencies

- Story 1.1 (exception hierarchy for calculation_failed events)

### Implementation Notes

- palmer.py is large (912 lines) — budget realistic time (12-16 hours, not 3-4)
- Test incrementally to avoid breaking existing functionality
- This story BLOCKS Story 4.1 (Palmer xarray handoff validation)

---

## Story 1.7: ETo Thornthwaite structlog Lifecycle Completion

**Priority:** Medium
**Agent:** Alpha
**Phase:** 1
**Effort:** 2-3 hours
**FR Coverage:** FR-PATTERN-008

### Description

Complete the structlog lifecycle pattern for `eto_thornthwaite()`. The function already has a logger instance but is missing the bind/lifecycle event pattern.

### Acceptance Criteria

- [ ] Lifecycle bind added: `_logger.bind(calculation="eto_thornthwaite", data_shape=temp.shape, latitude=latitude_degrees)`
- [ ] Events logged:
  - `calculation_started`
  - `calculation_completed` with duration timing
  - Optional: Temperature range stats at DEBUG level
- [ ] Pattern matches `eto_hargreaves()` exactly
- [ ] No changes to computation logic
- [ ] Tests pass without modification

### Dependencies

None

### Implementation Notes

- This is a quick win — function already uses structlog, just needs lifecycle pattern
- Match eto_hargreaves pattern for consistency

---

## Story 1.8: Structured Exceptions Migration - ETo Functions

**Priority:** Medium
**Agent:** Alpha
**Phase:** 1
**Effort:** 3-4 hours
**FR Coverage:** FR-PATTERN-009 (partial — ETo functions)

### Description

Replace generic `ValueError` with structured exceptions in `eto_thornthwaite()` and `eto_hargreaves()`.

### Acceptance Criteria

- [ ] `eto_thornthwaite()` input validation uses `InvalidArgumentError`:
  - Latitude out of range: Context includes `latitude=`, `valid_range=`
  - Invalid temperature values: Context includes `temp_min=`, `temp_max=`
- [ ] `eto_hargreaves()` input validation uses `InvalidArgumentError`:
  - Parameter validation with actionable messages
- [ ] Error messages provide guidance (e.g., "Latitude must be between -90 and 90 degrees. Received: {latitude}")
- [ ] All exceptions inherit from `ClimateIndicesError`
- [ ] Tests updated to expect `InvalidArgumentError` instead of `ValueError`

### Dependencies

- Story 1.1 (exception hierarchy)

### Implementation Notes

- Focus on clear error messages that help users fix their inputs
- Test both success and failure paths

---

## Story 1.9: Structured Exceptions Migration - PNP and PCI

**Priority:** Medium
**Agent:** Alpha
**Phase:** 1
**Effort:** 2-3 hours
**FR Coverage:** FR-PATTERN-009 (partial — PNP/PCI)

### Description

Replace generic `ValueError` with structured exceptions in `percentage_of_normal()` and `pci()`.

### Acceptance Criteria

- [ ] `percentage_of_normal()` uses `InvalidArgumentError` for input validation
- [ ] `pci()` uses `InvalidArgumentError` for length validation:
  - Message: "PCI requires exactly 365 or 366 daily precipitation values"
  - Context: `shape=data.shape`, `expected_length=[365, 366]`
- [ ] All error messages are actionable
- [ ] Tests updated for new exception types

### Dependencies

- Story 1.1 (exception hierarchy)

### Implementation Notes

- PCI validation is the most important — strict length requirement
- Combine with Stories 1.3 and 1.4 if working on same functions

---

## Story 1.10: Property-Based Tests - PNP, PCI, SPEI

**Priority:** Medium
**Agent:** Alpha
**Phase:** 1
**Effort:** 8-10 hours
**FR Coverage:** FR-PATTERN-010, FR-PATTERN-011, FR-PATTERN-012 (partial — SPEI)

### Description

Add property-based tests using Hypothesis to document mathematical invariants for PNP, PCI, and expand SPEI coverage.

### Acceptance Criteria

- [ ] **PNP properties** (`tests/test_properties.py::TestPercentageOfNormalProperties`):
  - Boundedness: `pnp >= 0` always
  - Shape preservation: `output.shape == input.shape`
  - NaN propagation: `np.isnan(input[i]) → np.isnan(output[i])`
  - Linear scaling: `pnp(2×p, 2×p_mean) = pnp(p, p_mean)`
- [ ] **PCI properties** (`tests/test_properties.py::TestPCIProperties`):
  - Range: `0 <= pci <= 100` always
  - Input length validation error for wrong length
  - NaN handling
  - Zero precipitation edge case
- [ ] **SPEI expanded properties** (`tests/test_properties.py::TestSPEIProperties`):
  - Shape preservation
  - NaN propagation from water balance
  - Zero water balance → SPEI near 0
- [ ] Hypothesis strategies use realistic climate data ranges
- [ ] All property tests pass with >100 examples

### Dependencies

- Stories 1.3, 1.4 (functions must have xarray support)

### Implementation Notes

- Budget 50-60 hours per index for comprehensive property testing (per Architecture Decision 3)
- This story covers 3 indices at ~3-4 hours each (basic properties only)
- Use `st.floats(min_value=0, max_value=1000, allow_nan=True)` for precipitation

---

## Story 1.11: Property-Based Tests - Palmer

**Priority:** Low
**Agent:** Alpha
**Phase:** 3
**Effort:** 6-8 hours
**FR Coverage:** FR-PATTERN-012 (partial — Palmer)

### Description

Expand Palmer property-based tests to cover PHDI, PMDI, and Z-Index (currently only PDSI boundedness exists).

### Acceptance Criteria

- [ ] **PHDI properties** added:
  - Bounded range test
  - Sequential consistency validation
- [ ] **PMDI properties** added:
  - Bounded range test
  - Sequential consistency validation
- [ ] **Z-Index properties** added:
  - Bounded range test
  - Sequential consistency validation
- [ ] Property: Sequential consistency (splitting time series changes results — Palmer is NOT embarrassingly parallel)
- [ ] All tests in `tests/test_properties.py::TestPalmerProperties`

### Dependencies

- Story 1.6 (Palmer structlog migration complete)
- Story 4.8 (Palmer xarray equivalence validated)

### Implementation Notes

- Palmer properties are more complex due to sequential state
- Sequential consistency property is important to document

---

## Story 1.12: Pattern Compliance Dashboard

**Priority:** Low (QA)
**Agent:** Alpha
**Phase:** 4
**Effort:** 4-5 hours
**FR Coverage:** NFR-PATTERN-COVERAGE

### Description

Create a compliance dashboard script or test that validates 100% pattern coverage across all 7 indices and 6 patterns (42 compliance points).

### Acceptance Criteria

- [ ] Script or test module validates:
  - xarray support: 7/7 indices
  - typed_public_api entries: 7/7 indices
  - CF metadata: 7/7 indices
  - structlog: 7/7 modules
  - Structured exceptions: 7/7 functions
  - Property tests: 7/7 indices
- [ ] Dashboard output shows 42/42 compliance points achieved
- [ ] CI integration: Compliance check runs on every PR
- [ ] Documentation updated with compliance status

### Dependencies

- All Track 0 stories (1.1-1.11)

### Implementation Notes

- This can be a pytest test that introspects the codebase
- Goal: NFR-PATTERN-COVERAGE validation (6 patterns × 7 indices = 42 points)

---

# Epic 2: PM-ET Foundation

**Goal:** Implement physics-based PET with FAO56 validation, establishing patterns for Tracks 2 & 3

**Agent:** Beta
**Track:** 1 (PM-ET Foundation)
**Duration:** 3-4 weeks
**Story Count:** 7

---

## Story 2.1: PM-ET Atmospheric Helpers

**Priority:** High
**Agent:** Beta
**Phase:** 1
**Effort:** 4-5 hours
**FR Coverage:** FR-PM-002

### Description

Implement private helper functions for atmospheric pressure and psychrometric constant (FAO56 Equations 7-8).

### Acceptance Criteria

- [ ] `_atm_pressure(altitude)` implemented in `src/climate_indices/eto.py`:
  - Equation: `P = 101.3 × [(293 - 0.0065z)/293]^5.26`
  - Type: `float → float` (scalar operation)
- [ ] `_psy_const(pressure)` implemented:
  - Equation: `γ = 0.000665 × P`
  - Type: `float → float`
- [ ] Unit tests validate against known values:
  - Uccle (100m altitude) → 100.1 kPa
  - Bangkok (0m altitude) → 101.3 kPa
- [ ] Docstrings include equation numbers and units
- [ ] mypy --strict passes

### Dependencies

None (foundation functions)

### Implementation Notes

- These are scalar helper functions, not vectorized
- Exact FAO56 constants must be used (no approximations)

---

## Story 2.2: PM-ET Vapor Pressure Helpers

**Priority:** High
**Agent:** Beta
**Phase:** 1
**Effort:** 5-6 hours
**FR Coverage:** FR-PM-003

### Description

Implement vapor pressure calculation helpers using correct Magnus formula constants (FAO56 Equations 11-13).

### Acceptance Criteria

- [ ] `_svp_from_t(temp)` implemented:
  - Equation: `e°(T) = 0.6108 × exp[17.27T/(T+237.3)]`
  - Array-compatible: accepts `np.ndarray`, returns same shape
  - Exact constants: `0.6108, 17.27, 237.3`
- [ ] `_mean_svp(tmin, tmax)` implemented:
  - Equation: `es = (e°(Tmax) + e°(Tmin)) / 2`
  - NOT `e°(Tmean)` (critical non-linearity)
- [ ] `_slope_svp(temp)` implemented:
  - Equation: `Δ = 4098 × e°(T) / (T+237.3)²`
- [ ] Unit tests validate intermediate values against FAO56 examples
- [ ] Precision: ±0.01 kPa for intermediate values (NFR-PM-PERF)

### Dependencies

None

### Implementation Notes

- Mean SVP non-linearity is a common implementation error — validate carefully
- Use exact FAO56 constants (not alternative Magnus formulations)

---

## Story 2.3: PM-ET Humidity Pathway Dispatcher

**Priority:** High
**Agent:** Beta
**Phase:** 1
**Effort:** 6-7 hours
**FR Coverage:** FR-PM-004

### Description

Implement auto-selection logic for actual vapor pressure calculation based on available humidity inputs (FAO56 Equations 14-19).

### Acceptance Criteria

- [ ] `_avp_from_dewpoint(tdew)` implemented (Eq. 14):
  - Most accurate method
  - Priority: 1 (preferred)
- [ ] `_avp_from_rhminmax(...)` implemented (Eq. 17):
  - Preferred for daily data
  - Priority: 2
- [ ] `_avp_from_rhmean(...)` implemented (Eq. 19):
  - Fallback method
  - Priority: 3
- [ ] Dispatcher in `eto_penman_monteith()` auto-selects based on available inputs
- [ ] Raises `ValueError` if no humidity input provided
- [ ] Selected pathway logged at DEBUG level (structlog)
- [ ] Unit tests validate each pathway independently

### Dependencies

- Story 2.2 (SVP helpers)

### Implementation Notes

- Follow FAO56 hierarchy strictly: dewpoint > RH extremes > RH mean
- Provide clear error if user provides no humidity data

---

## Story 2.4: PM-ET Core Calculation

**Priority:** Critical
**Agent:** Beta
**Phase:** 1
**Effort:** 8-10 hours
**FR Coverage:** FR-PM-001

### Description

Implement the core Penman-Monteith FAO56 reference evapotranspiration calculation (Equation 6).

### Acceptance Criteria

- [ ] Public function `eto_penman_monteith(tmin, tmax, tmean, wind_2m, net_radiation, latitude, altitude, ...)` implemented
- [ ] Equation 6 implementation:
  - Numerator: `0.408 × Δ × (Rn - G) + γ × (900/(T+273)) × u2 × (es - ea)`
  - Denominator: `Δ + γ × (1 + 0.34 × u2)`
  - Critical: Kelvin conversion in denominator (`T + 273`)
- [ ] Returns `np.ndarray` matching input shape
- [ ] Integrates humidity dispatcher from Story 2.3
- [ ] Uses atmospheric helpers from Story 2.1
- [ ] Uses vapor pressure helpers from Story 2.2
- [ ] structlog lifecycle events: `calculation_started`, `calculation_completed`
- [ ] Docstring documents FAO56 reference and parameter units

### Dependencies

- Story 2.1 (atmospheric helpers)
- Story 2.2 (vapor pressure helpers)
- Story 2.3 (humidity dispatcher)

### Implementation Notes

- This is the orchestration function tying all helpers together
- Kelvin conversion placement is critical — common implementation error

---

## Story 2.5: FAO56 Worked Example Validation

**Priority:** Critical
**Agent:** Beta
**Phase:** 1
**Effort:** 5-6 hours
**FR Coverage:** FR-PM-005

### Description

Validate PM-ET implementation against published FAO56 Examples 17 (tropical) and 18 (temperate).

### Acceptance Criteria

- [ ] Test `tests/test_eto.py::test_fao56_example_17_bangkok()`:
  - Input: Bangkok, April (tropical monthly)
  - Expected: 5.72 mm/day ±0.05
  - Input data embedded in test (no external files)
- [ ] Test `tests/test_eto.py::test_fao56_example_18_uccle()`:
  - Input: Uccle, 6 July (temperate daily)
  - Expected: 3.9 mm/day ±0.05
  - Input data embedded in test
- [ ] Both tests pass (NFR-PM-PERF: ±0.05 mm/day tolerance)
- [ ] Intermediate values validated against FAO56 tables (±0.01 kPa for vapor pressure)

### Dependencies

- Story 2.4 (PM-ET core calculation)

### Implementation Notes

- These are the primary scientific validation tests
- If tests fail, implementation has an error — do not relax tolerance

---

## Story 2.6: PM-ET xarray Adapter

**Priority:** High
**Agent:** Beta
**Phase:** 1
**Effort:** 4-5 hours
**FR Coverage:** FR-PM-006

### Description

Apply `@xarray_adapter` decorator and add CF metadata for PM-ET.

### Acceptance Criteria

- [ ] `@xarray_adapter` applied to `eto_penman_monteith()`
- [ ] CF metadata registry entry:
  - `long_name="Reference Evapotranspiration (Penman-Monteith FAO56)"`
  - `units="mm day-1"`
  - `references`: Allen et al. 1998 DOI
- [ ] `@overload` signatures in `typed_public_api.py`:
  - NumPy path: `eto_penman_monteith(np.ndarray, ...) -> np.ndarray`
  - xarray path: `eto_penman_monteith(xr.DataArray, ...) -> xr.DataArray`
- [ ] Equivalence test passes (numpy vs xarray, tolerance: 1e-8)
- [ ] Dask compatibility: `dask="parallelized"` in apply_ufunc
- [ ] Coordinate preservation validated

### Dependencies

- Story 2.4 (PM-ET core implementation)
- Story 2.5 (validation passing)

### Implementation Notes

- Follow existing eto_thornthwaite pattern
- Dask parallelization is safe (no sequential state like Palmer)

---

## Story 2.7: Palmer Performance Baseline Measurement

**Priority:** High (BLOCKS Track 3)
**Agent:** Beta
**Phase:** 1
**Effort:** 4-5 hours
**FR Coverage:** NFR-PALMER-PERF (establishes baseline)

### Description

Measure current multiprocessing CLI performance as baseline for Palmer xarray performance validation (NFR-PALMER-PERF: ≥80% speed target).

### Acceptance Criteria

- [ ] New test module: `tests/test_benchmark_palmer.py`
- [ ] Benchmark measures wall-clock time for:
  - Grid size: 360×180 (global 1-degree)
  - Time series: 240 months (20 years)
  - Current multiprocessing CLI path
- [ ] Baseline time recorded in test or documentation
- [ ] CI integration: Performance regression alerts if new Palmer xarray drops below 80% of baseline
- [ ] Test uses pytest-benchmark or similar for accurate timing

### Dependencies

None (measures existing functionality)

### Implementation Notes

- This BLOCKS Story 4.1 (cannot validate Track 3 performance without baseline)
- Without this, NFR-PALMER-PERF cannot be validated
- Run on consistent hardware for reproducible results

---

# Epic 3: EDDI/PNP/scPDSI Coverage

**Goal:** Complete EDDI validation, add PNP xarray, stub scPDSI interface

**Agent:** Gamma
**Track:** 2 (Index Coverage Expansion)
**Duration:** 2-3 weeks
**Story Count:** 7

---

## Story 3.1: Resolve PR #597 EDDI Merge Conflicts

**Priority:** Critical
**Agent:** Gamma
**Phase:** 2
**Effort:** 3-4 hours
**FR Coverage:** None (prerequisite)

### Description

Resolve merge conflicts in existing PR #597 for EDDI implementation. This is NOT greenfield — EDDI algorithm already exists but needs conflict resolution.

### Acceptance Criteria

- [ ] PR #597 merge conflicts resolved
- [ ] Existing EDDI tests pass
- [ ] No regressions in existing functionality
- [ ] Code review feedback addressed
- [ ] Ready for Stories 3.2-3.6 to build on top

### Dependencies

None

### Implementation Notes

- EDDI implementation already exists — this is integration work, not new development
- Focus on clean merge, not algorithm changes

---

## Story 3.2: NOAA Provenance Protocol Establishment

**Priority:** High (BLOCKS FR-EDDI-001)
**Agent:** Gamma
**Phase:** 2
**Effort:** 3-4 hours
**FR Coverage:** None (infrastructure — Architecture Decision 1)

### Description

Establish JSON-based provenance metadata protocol for external reference datasets. Covers BOTH EDDI (new) and Palmer (retroactive in Story 4.8).

### Acceptance Criteria

- [ ] Provenance protocol documented in `tests/fixture/README.md`:
  - Required fields: source, url, download_date, subset_description, checksum_sha256, fixture_version, validation_tolerance
- [ ] Template `provenance.json` structure defined
- [ ] Location pattern established: `tests/fixture/<dataset-name>/provenance.json`
- [ ] Example provenance file created for demonstration
- [ ] CI validation: Check provenance.json exists for all reference datasets

### Dependencies

None

### Implementation Notes

- This establishes the protocol used by Stories 3.3 (EDDI) and 4.8 (Palmer)
- Provenance is critical for scientific reproducibility
- Checksum validates dataset hasn't changed

---

## Story 3.3: NOAA EDDI Reference Validation

**Priority:** Critical (FR-TEST-004 BLOCKER)
**Agent:** Gamma
**Phase:** 2
**Effort:** 8-10 hours
**FR Coverage:** FR-EDDI-001

### Description

Download NOAA PSL EDDI reference dataset, create provenance metadata, and implement validation test (FR-TEST-004). This is a BLOCKER — no merge without this test passing.

### Acceptance Criteria

- [ ] NOAA PSL EDDI data downloaded from [downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/](https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/)
- [ ] Subset created: `tests/fixture/noaa-eddi-1month/`, `noaa-eddi-3month/`, `noaa-eddi-6month/`
- [ ] Provenance metadata created following Story 3.2 protocol:
  - `tests/fixture/noaa-eddi-1month/provenance.json`
  - Fields: source, url, download_date, subset_description, checksum
- [ ] Test `tests/test_noaa_eddi_reference.py::test_eddi_noaa_reference_1month()`:
  - Computes EDDI using library
  - Validates against NOAA reference: `np.testing.assert_allclose(computed, noaa_ref, rtol=1e-5, atol=1e-5)`
  - Tolerance: 1e-5 (NFR-EDDI-VAL — looser than 1e-8 due to non-parametric ranking)
- [ ] Tests for 3-month and 6-month scales
- [ ] All tests pass (REQUIRED for merge)

### Dependencies

- Story 3.1 (EDDI algorithm available)
- Story 3.2 (provenance protocol)

### Implementation Notes

- This is THE critical test for EDDI — FR-TEST-004 blocker
- If test fails, tolerance may need refinement (but start with 1e-5 per NFR-EDDI-VAL)
- Subset should be small enough for git (not full CONUS archive)

---

## Story 3.4: EDDI xarray Adapter

**Priority:** High
**Agent:** Gamma
**Phase:** 2
**Effort:** 3-4 hours
**FR Coverage:** FR-EDDI-002

### Description

Apply `@xarray_adapter` decorator to EDDI function with CF metadata.

### Acceptance Criteria

- [ ] `@xarray_adapter` applied to `eddi()` in `src/climate_indices/indices.py`
- [ ] CF metadata registry entry:
  - `long_name="Evaporative Demand Drought Index"`
  - `units=""` (dimensionless)
  - `standard_name="atmosphere_water_vapor_evaporative_demand_anomaly"` (custom)
  - `references`: Hobbins et al. (2016) DOI: 10.1175/JHM-D-15-0121.1
- [ ] `@overload` signatures in `typed_public_api.py`
- [ ] Equivalence test: numpy path == xarray path within 1e-8
- [ ] mypy --strict passes

### Dependencies

- Story 3.1 (EDDI algorithm available)
- Story 3.3 (EDDI validated)

### Implementation Notes

- Follow existing SPI/SPEI xarray adapter pattern
- Custom standard_name is OK (not in CF table yet)

---

## Story 3.5: EDDI CLI Integration

**Priority:** Medium
**Agent:** Gamma
**Phase:** 2
**Effort:** 4-5 hours
**FR Coverage:** FR-EDDI-003

### Description

Add EDDI support to `process_climate_indices` CLI (Issue #414).

### Acceptance Criteria

- [ ] CLI flag `--index eddi` added
- [ ] Parameter `--pet_file <path>` for PET input netCDF
- [ ] Help text documents:
  - EDDI requires PET input (not precipitation)
  - Recommended: Use PM FAO56 for PET
  - Warning: Thornthwaite may produce inaccurate drought signals
- [ ] Example command in README or docs:
  ```bash
  process_climate_indices --index eddi --pet_file pet_pm.nc --scale 6 --output eddi_6mo.nc
  ```
- [ ] Integration test: CLI produces valid EDDI output

### Dependencies

- Story 3.4 (EDDI xarray support)
- Story 2.6 (PM-ET available for recommendation)

### Implementation Notes

- CLI is user-facing — help text must be clear
- Cross-reference PM-ET as recommended method

---

## Story 3.6: EDDI PET Method Documentation

**Priority:** Medium
**Agent:** Gamma
**Phase:** 2
**Effort:** 2-3 hours
**FR Coverage:** FR-EDDI-004

### Description

Document PM FAO56 recommendation in EDDI docstring and algorithm documentation.

### Acceptance Criteria

- [ ] EDDI docstring updated with Note section:
  - "EDDI is most accurate when using Penman-Monteith FAO56 reference evapotranspiration (ETo)."
  - Warning: "Using simplified methods like Thornthwaite may produce inaccurate drought signals."
- [ ] See Also section cross-references `eto_penman_monteith()`
- [ ] Hobbins et al. (2016) citation added to References
- [ ] `docs/algorithms.rst` updated with EDDI section:
  - PET method sensitivity discussion
  - PM FAO56 recommendation rationale
- [ ] Docstring renders correctly in Sphinx

### Dependencies

- Story 2.6 (PM-ET available)

### Implementation Notes

- This provides scientific guidance to users
- Important for operational deployment confidence

---

## Story 3.7: scPDSI Stub Interface

**Priority:** Low
**Agent:** Gamma
**Phase:** 3
**Effort:** 2-3 hours
**FR Coverage:** FR-SCPDSI-001

### Description

Define scPDSI function signature with NotImplementedError for future implementation.

### Acceptance Criteria

- [ ] Function `scpdsi(precip, pet, awc, ...)` added to `src/climate_indices/indices.py`
- [ ] Raises `NotImplementedError("scPDSI implementation planned for future release")`
- [ ] Docstring includes:
  - Methodology overview (self-calibrating variant of PDSI)
  - Wells et al. (2004) reference
  - Placeholder parameter descriptions
- [ ] `@overload` type signatures in `typed_public_api.py`:
  - NumPy path: future tuple return
  - xarray path: future Dataset return
- [ ] Function appears in API documentation
- [ ] Imports work (no runtime errors unless called)

### Dependencies

None

### Implementation Notes

- This is future-proofing — defines interface for v2.5.0 or later
- Users can see the function exists but isn't ready yet

---

# Epic 4: Palmer Multi-Output

**Goal:** Deliver Palmer indices with Dataset return and CF metadata per variable

**Agent:** Delta
**Track:** 3 (Advanced xarray Capabilities)
**Duration:** 3-4 weeks
**Story Count:** 9

---

## Story 4.1: Palmer xarray Handoff Validation

**Priority:** Critical (GATE story)
**Agent:** Delta
**Phase:** 2
**Effort:** 2-3 hours
**FR Coverage:** None (orchestration gate)

### Description

Validate that Palmer structlog migration (Story 1.6) is complete and baseline performance measurement (Story 2.7) exists before beginning Palmer xarray implementation.

### Acceptance Criteria

- [ ] Story 1.6 complete: Palmer uses structlog (no stdlib logging)
- [ ] Story 2.7 complete: Baseline performance measurement exists
- [ ] All Palmer tests pass with structlog
- [ ] Baseline time documented for NFR-PALMER-PERF validation
- [ ] Agent Delta has clean handoff from Agent Alpha

### Dependencies

- Story 1.6 (Palmer structlog migration — CRITICAL PATH)
- Story 2.7 (Palmer performance baseline)

### Implementation Notes

- This is a gate story — no Palmer xarray work begins until this passes
- Prevents mixing logging frameworks during complex refactoring

---

## Story 4.2: Palmer CF Metadata Registry Entries

**Priority:** High
**Agent:** Delta
**Phase:** 2
**Effort:** 2-3 hours
**FR Coverage:** FR-PALMER-005

### Description

Add CF metadata registry entries for all 4 Palmer variables.

### Acceptance Criteria

- [ ] Registry entries in `src/climate_indices/cf_metadata_registry.py`:
  - `pdsi`: `long_name="Palmer Drought Severity Index"`, `units=""`, Palmer (1965) reference
  - `phdi`: `long_name="Palmer Hydrological Drought Index"`, `units=""`, Palmer (1965) reference
  - `pmdi`: `long_name="Palmer Modified Drought Index"`, `units=""`, Heddinghaus & Sabol (1991) reference
  - `z_index`: `long_name="Palmer Z-Index"`, `units=""`, Palmer (1965) reference
- [ ] Each entry includes references with citations
- [ ] Registry is importable and follows existing pattern

### Dependencies

- Story 1.2 (CF metadata registry module exists)

### Implementation Notes

- 4 independent variables need separate metadata
- Palmer (1965) is primary reference; PMDI uses Heddinghaus & Sabol (1991)

---

## Story 4.3: palmer_xarray() Manual Wrapper Foundation

**Priority:** Critical
**Agent:** Delta
**Phase:** 2
**Effort:** 8-10 hours
**FR Coverage:** FR-PALMER-001, FR-PALMER-002 (partial)

### Description

Implement manual `palmer_xarray()` wrapper function in palmer.py (NOT separate module per Architecture Decision 2). Uses Pattern C from research: stack/unpack workaround for xarray Issue #1815.

### Acceptance Criteria

- [ ] Function `palmer_xarray(precip_da, pet_da, awc, ...)` added to `src/climate_indices/palmer.py`
- [ ] Location: Palmer module (NOT separate palmer_xarray.py per Architecture Decision 2)
- [ ] Implementation pattern:
  - Extract numpy arrays from DataArrays
  - Call existing `pdsi()` numpy function
  - Receive 5-tuple: (pdsi, phdi, pmdi, z_index, params_dict)
  - Stack outputs: `np.stack([pdsi, phdi, pmdi, z_index], axis=0)`
  - Unpack to Dataset with coordinate preservation
- [ ] Returns `xr.Dataset` (not tuple)
- [ ] Basic smoke test: Function runs without errors
- [ ] Docstring explains manual wrapper rationale (multi-output + params_dict requires custom handling)

### Dependencies

- Story 4.1 (handoff validation)
- Story 4.2 (CF metadata entries)

### Implementation Notes

- **CORRECTION:** Function stays in palmer.py per Architecture Decision 2 (not separate module)
- Module size: 912 lines + ~150 new = 1,062 lines (within 1,400 line threshold)
- This is foundation — subsequent stories add features

---

## Story 4.4: AWC Spatial Parameter Handling

**Priority:** High
**Agent:** Delta
**Phase:** 2
**Effort:** 5-6 hours
**FR Coverage:** FR-PALMER-003

### Description

Implement AWC (Available Water Capacity) parameter handling supporting both scalar and DataArray (spatial variation only — no time dimension).

### Acceptance Criteria

- [ ] `palmer_xarray()` accepts `awc` as:
  - Scalar float (uniform AWC)
  - `xr.DataArray` with spatial dims only (lat, lon)
- [ ] Validation: Raises `ValueError` if `awc` has time dimension:
  - Message: "AWC must not have time dimension 'time'. AWC is a soil property (spatially varying only)."
  - Context: `awc_dims=awc.dims`
- [ ] Test: Scalar AWC produces valid output
- [ ] Test: Spatial AWC (lat, lon) produces valid output
- [ ] Test: AWC with time dimension raises error

### Dependencies

- Story 4.3 (palmer_xarray wrapper exists)

### Implementation Notes

- AWC is soil property (spatial only) — time variation is a user error
- Provide clear error message to guide users

---

## Story 4.5: Multi-Output Dataset Construction

**Priority:** High
**Agent:** Delta
**Phase:** 2
**Effort:** 6-7 hours
**FR Coverage:** FR-PALMER-002

### Description

Complete Dataset construction with per-variable CF metadata and coordinate preservation.

### Acceptance Criteria

- [ ] Dataset contains 4 data variables: `pdsi`, `phdi`, `pmdi`, `z_index`
- [ ] Each variable has independent CF metadata from registry (Story 4.2)
- [ ] Coordinates preserved from input DataArrays (lat, lon, time)
- [ ] Dimensions correctly aligned
- [ ] Test: NetCDF write/read round-trip preserves structure
- [ ] Test: Each variable accessible: `ds_palmer["pdsi"]`

### Dependencies

- Story 4.3 (wrapper foundation)
- Story 4.2 (CF metadata)

### Implementation Notes

- Use `xr.Dataset({var: da for var, da in zip(["pdsi", "phdi", ...], outputs)})`
- Each variable gets metadata via `da.assign_attrs(CF_METADATA[var])`

---

## Story 4.6: params_dict JSON Serialization

**Priority:** Medium
**Agent:** Delta
**Phase:** 2
**Effort:** 4-5 hours
**FR Coverage:** FR-PALMER-004

### Description

Implement dual-access params_dict handling: JSON string in attrs AND individual parameter attrs.

### Acceptance Criteria

- [ ] `ds.attrs["palmer_params"]` contains JSON string: `'{"alpha": 1.5, "beta": 0.8, ...}'`
- [ ] Individual attrs: `ds.attrs["palmer_alpha"]`, `ds.attrs["palmer_beta"]`, etc.
- [ ] JSON serialization round-trip preserves structure:
  ```python
  params = json.loads(ds.attrs["palmer_params"])
  assert params["alpha"] == ds.attrs["palmer_alpha"]
  ```
- [ ] Test: params_dict from first grid cell stored correctly
- [ ] Test: JSON round-trip validation
- [ ] Docstring explains dual access pattern

### Dependencies

- Story 4.3 (wrapper foundation)

### Implementation Notes

- Params are spatially constant (computed from first grid cell)
- Dual access pattern: JSON for full dict, individual attrs for convenience

---

## Story 4.7: Palmer typed_public_api @overload Signatures

**Priority:** High
**Agent:** Delta
**Phase:** 2
**Effort:** 3-4 hours
**FR Coverage:** FR-PALMER-006

### Description

Add `@overload` type signatures to distinguish numpy tuple return vs xarray Dataset return.

### Acceptance Criteria

- [ ] `@overload` signatures in `typed_public_api.py`:
  - NumPy path: `pdsi(np.ndarray, ...) -> tuple[NDArray, NDArray, NDArray, NDArray, dict | None]`
  - xarray path: `pdsi(xr.DataArray, ...) -> xr.Dataset`
- [ ] Runtime dispatcher:
  - If `isinstance(precips, xr.DataArray)`: call `palmer_xarray()`
  - Else: call numpy `pdsi()`
- [ ] mypy --strict passes
- [ ] Type checker correctly infers return type based on input type

### Dependencies

- Story 4.3 (palmer_xarray function exists)

### Implementation Notes

- This is the most complex `@overload` (different return types)
- Type safety prevents tuple unpacking errors with xarray inputs

---

## Story 4.8: Palmer Equivalence and Provenance

**Priority:** High
**Agent:** Delta
**Phase:** 2
**Effort:** 6-8 hours
**FR Coverage:** FR-PALMER-007

### Description

Implement NumPy vs xarray equivalence tests and apply NOAA provenance protocol (retroactive) to Palmer reference dataset.

### Acceptance Criteria

- [ ] Equivalence test `test_palmer_xarray_equivalence()`:
  - Computes Palmer via numpy path (tuple return)
  - Computes Palmer via xarray path (Dataset return)
  - Validates: `np.testing.assert_allclose(numpy_pdsi, ds_palmer["pdsi"].values, atol=1e-8)`
  - Validates all 4 variables: pdsi, phdi, pmdi, z_index
  - Tolerance: 1e-8 (NFR-PATTERN-EQUIV)
- [ ] **CORRECTION:** Validates xarray vs numpy (NOT vs pdi.f Fortran)
- [ ] Provenance metadata created for Palmer reference dataset (if exists):
  - `tests/fixture/palmer-reference/provenance.json`
  - Follows Story 3.2 protocol
- [ ] Scalar AWC vs DataArray AWC equivalence
- [ ] params_dict equivalence validation

### Dependencies

- Story 4.5 (Dataset construction)
- Story 4.6 (params_dict handling)
- Story 3.2 (provenance protocol)

### Implementation Notes

- **CORRECTION:** This validates Python xarray vs Python numpy (not Fortran)
- Provenance is retroactive (Palmer reference may already exist)
- This satisfies NFR-PATTERN-EQUIV for Palmer

---

## Story 4.9: Palmer Performance Validation

**Priority:** High
**Agent:** Delta
**Phase:** 2
**Effort:** 4-5 hours
**FR Coverage:** NFR-PALMER-PERF

### Description

Validate that Palmer xarray achieves ≥80% speed of multiprocessing baseline (measured in Story 2.7).

### Acceptance Criteria

- [ ] Performance test `test_palmer_xarray_performance()`:
  - Same grid size as baseline: 360×180, 240 months
  - Measures wall-clock time for `palmer_xarray()`
  - Compares to baseline from Story 2.7
  - Asserts: `xarray_time <= baseline_time * 1.25` (i.e., ≥80% speed)
- [ ] Test uses consistent hardware/environment
- [ ] CI integration: Performance regression alerts
- [ ] If performance target not met: Document rationale (vectorize=True overhead acceptable per research Section 6.2.6)

### Dependencies

- Story 2.7 (baseline measurement)
- Story 4.5 (palmer_xarray complete)

### Implementation Notes

- NFR-PALMER-PERF: ≥80% speed (i.e., at most 25% slower)
- If target not met, accept with documentation (sequential constraint is fundamental)

---

# Epic 5: Cross-Cutting Validation

**Goal:** Final validation of all 31 FRs and 8 NFRs across Tracks 0-3

**Agents:** Multiple (Alpha, Gamma, Omega)
**Track:** Integration & QA
**Duration:** ~1 week
**Story Count:** 3

---

## Story 5.1: Pattern Compliance Audit

**Priority:** Medium
**Agent:** Alpha
**Phase:** 3
**Effort:** 4-5 hours
**FR Coverage:** NFR-PATTERN-COVERAGE

### Description

Final audit of 6 patterns × 7 indices = 42 compliance points.

### Acceptance Criteria

- [ ] Compliance script or test validates:
  - **xarray support:** 7/7 indices (percentage_of_normal, pci, eto_thornthwaite, eto_hargreaves, spi, spei, pdsi)
  - **typed_public_api:** 7/7 @overload signature sets
  - **CF metadata:** 7/7 registry entries
  - **structlog:** 7/7 modules (no stdlib logging)
  - **Structured exceptions:** 7/7 functions (no generic ValueError)
  - **Property tests:** 7/7 indices
- [ ] Dashboard output: **42/42 compliance points achieved**
- [ ] CI integration: Compliance check on every PR
- [ ] Documentation updated with compliance status

### Dependencies

- All Track 0 stories (Epic 1)
- Story 4.5 (Palmer xarray complete)

### Implementation Notes

- This is Story 1.12 completion checkpoint
- Validates NFR-PATTERN-COVERAGE

---

## Story 5.2: Reference Validation Final Check

**Priority:** High
**Agent:** Gamma
**Phase:** 3
**Effort:** 3-4 hours
**FR Coverage:** FR-EDDI-001 (final validation)

### Description

Final validation that NOAA reference tests pass and provenance is complete.

### Acceptance Criteria

- [ ] EDDI NOAA reference tests pass:
  - `test_eddi_noaa_reference_1month()`
  - `test_eddi_noaa_reference_3month()`
  - `test_eddi_noaa_reference_6month()`
  - Tolerance: 1e-5 (NFR-EDDI-VAL)
- [ ] Provenance metadata complete for all reference datasets:
  - EDDI: `tests/fixture/noaa-eddi-*/provenance.json`
  - Palmer (if applicable): `tests/fixture/palmer-reference/provenance.json`
- [ ] Checksum validation passes (dataset hasn't changed)
- [ ] Documentation: Reference dataset usage guide

### Dependencies

- Story 3.3 (EDDI reference validation)
- Story 4.8 (Palmer provenance)

### Implementation Notes

- This is final gate for FR-TEST-004 (EDDI NOAA validation)
- Ensures scientific reproducibility

---

## Story 5.3: Final v2.4.0 Validation

**Priority:** Critical (FINAL GATE)
**Agent:** Omega (All)
**Phase:** 4
**Effort:** 6-8 hours
**FR Coverage:** ALL 31 FRs, ALL 8 NFRs

### Description

Comprehensive final validation that ALL 31 FRs and ALL 8 NFRs are satisfied before v2.4.0 release.

### Acceptance Criteria

**Functional Requirements (31 Total):**
- [ ] **Track 0 (12 FRs):** All pattern completion FRs validated
  - FR-PATTERN-001 through FR-PATTERN-012: All pass
- [ ] **Track 1 (6 FRs):** PM-ET FRs validated
  - FR-PM-001 through FR-PM-006: All pass
  - FAO56 examples within ±0.05 mm/day
- [ ] **Track 2 (6 FRs):** EDDI/PNP/scPDSI FRs validated
  - FR-EDDI-001 through FR-EDDI-004: All pass
  - FR-PNP-001: Pass
  - FR-SCPDSI-001: Stub exists
- [ ] **Track 3 (7 FRs):** Palmer multi-output FRs validated
  - FR-PALMER-001 through FR-PALMER-007: All pass

**Non-Functional Requirements (8 Total):**
- [ ] NFR-PATTERN-EQUIV: All equivalence tests pass (1e-8)
- [ ] NFR-PATTERN-COVERAGE: 42/42 compliance points achieved
- [ ] NFR-PATTERN-MAINT: Pattern consistency validated
- [ ] NFR-PM-PERF: FAO56 examples within ±0.05 mm/day
- [ ] NFR-EDDI-VAL: NOAA reference within 1e-5
- [ ] NFR-PALMER-SEQ: Sequential constraint documented
- [ ] NFR-PALMER-PERF: Performance ≥80% of baseline
- [ ] NFR-MULTI-OUT: Palmer Dataset return stable

**Code Quality:**
- [ ] mypy --strict passes
- [ ] All tests pass (coverage >85%)
- [ ] Documentation complete
- [ ] CI green

**Release Checklist:**
- [ ] CHANGELOG.md updated
- [ ] Version bumped to 2.4.0
- [ ] All 38 stories marked complete
- [ ] No open blockers

### Dependencies

- ALL stories (1.1 through 5.2)

### Implementation Notes

- This is the final gate before v2.4.0 release
- If ANY FR or NFR fails, release is BLOCKED
- Comprehensive validation ensures quality

---

# FR Coverage Map (Individual Traceability)

| FR Code | Story ID | Story Title | Agent | Phase |
|---------|----------|-------------|-------|-------|
| FR-PATTERN-001 | 1.3 | PNP xarray + typed_public_api Integration | Alpha | 1 |
| FR-PATTERN-002 | 1.4 | PCI xarray + typed_public_api Integration | Alpha | 1 |
| FR-PATTERN-003 | 1.5 | ETo Helpers typed_public_api Integration | Alpha | 1 |
| FR-PATTERN-004 | 1.5 | ETo Helpers typed_public_api Integration | Alpha | 1 |
| FR-PATTERN-005 | 1.3 | PNP xarray + typed_public_api Integration | Alpha | 1 |
| FR-PATTERN-006 | 1.4 | PCI xarray + typed_public_api Integration | Alpha | 1 |
| FR-PATTERN-007 | 1.6 | Palmer structlog Migration | Alpha | 1 |
| FR-PATTERN-008 | 1.7 | ETo Thornthwaite structlog Lifecycle Completion | Alpha | 1 |
| FR-PATTERN-009 | 1.1, 1.8, 1.9 | Structured Exception Hierarchy + Migrations | Alpha | 0-1 |
| FR-PATTERN-010 | 1.10 | Property-Based Tests - PNP, PCI, SPEI | Alpha | 1 |
| FR-PATTERN-011 | 1.10 | Property-Based Tests - PNP, PCI, SPEI | Alpha | 1 |
| FR-PATTERN-012 | 1.10, 1.11 | Property-Based Tests - SPEI + Palmer | Alpha | 1, 3 |
| FR-PM-001 | 2.4 | PM-ET Core Calculation | Beta | 1 |
| FR-PM-002 | 2.1 | PM-ET Atmospheric Helpers | Beta | 1 |
| FR-PM-003 | 2.2 | PM-ET Vapor Pressure Helpers | Beta | 1 |
| FR-PM-004 | 2.3 | PM-ET Humidity Pathway Dispatcher | Beta | 1 |
| FR-PM-005 | 2.5 | FAO56 Worked Example Validation | Beta | 1 |
| FR-PM-006 | 2.6 | PM-ET xarray Adapter | Beta | 1 |
| FR-EDDI-001 | 3.3, 5.2 | NOAA EDDI Reference Validation + Final Check | Gamma | 2-3 |
| FR-EDDI-002 | 3.4 | EDDI xarray Adapter | Gamma | 2 |
| FR-EDDI-003 | 3.5 | EDDI CLI Integration | Gamma | 2 |
| FR-EDDI-004 | 3.6 | EDDI PET Method Documentation | Gamma | 2 |
| FR-PNP-001 | 1.3 | PNP xarray + typed_public_api Integration | Alpha | 1 |
| FR-SCPDSI-001 | 3.7 | scPDSI Stub Interface | Gamma | 3 |
| FR-PALMER-001 | 4.3 | palmer_xarray() Manual Wrapper Foundation | Delta | 2 |
| FR-PALMER-002 | 4.3, 4.5 | Manual Wrapper + Multi-Output Dataset Construction | Delta | 2 |
| FR-PALMER-003 | 4.4 | AWC Spatial Parameter Handling | Delta | 2 |
| FR-PALMER-004 | 4.6 | params_dict JSON Serialization | Delta | 2 |
| FR-PALMER-005 | 4.2 | Palmer CF Metadata Registry Entries | Delta | 2 |
| FR-PALMER-006 | 4.7 | Palmer typed_public_api @overload Signatures | Delta | 2 |
| FR-PALMER-007 | 4.8 | Palmer Equivalence and Provenance | Delta | 2 |

**Coverage:** 31/31 FRs mapped to stories

---

# Dependency Graph

```
Foundation (Phase 0):
1.1 → 1.2
  ↓
Phase 1 (Parallel):
Alpha: 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10
Beta:  2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6
       2.7 (baseline)

Gate: 1.6 + 2.7 → 4.1 (handoff validation)

Phase 2 (Parallel after gate):
Gamma: 3.1 → 3.2 → 3.3 → 3.4, 3.5, 3.6, 3.7
Delta: 4.1 → 4.2 → 4.3 → 4.4, 4.5, 4.6 → 4.7, 4.8, 4.9

Phase 3:
1.11 (Alpha)
5.1 (Alpha)
5.2 (Gamma)

Phase 4:
5.3 (All → FINAL GATE)
```

**Critical Path:** 1.1 → 1.2 → 1.6 → 4.1 → 4.3 → 4.5 → 4.8 → 5.3

---

# Summary Statistics

- **Total Stories:** 38
- **Total FRs:** 31 (12 Track 0 + 6 Track 1 + 6 Track 2 + 7 Track 3)
- **Total NFRs:** 8
- **Epics:** 5
- **Agents:** 5 (Alpha, Beta, Gamma, Delta, Omega)
- **Phases:** 5 (0, 1, 2, 3, 4)
- **Estimated Duration:** 10-14 weeks
- **Critical Path:** ~5-6 weeks (1.1 → 1.2 → 1.6 → 4.1 → 4.2-4.9)
- **Parallelization Savings:** ~5-7 weeks vs sequential

---

**Document Status:** ✅ Complete - All 38 Stories Fully Defined with Acceptance Criteria
**Version:** 2.4.0-expanded
**Last Updated:** 2026-02-16
**Ready for Implementation:** YES

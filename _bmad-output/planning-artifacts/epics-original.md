---
stepsCompleted: ['step-01-validate-prerequisites', 'step-02-agent-orchestration', 'step-03-epic-definitions', 'step-04-story-breakdown']
inputDocuments:
  - '_bmad-output/planning-artifacts/prd.md'
  - '_bmad-output/planning-artifacts/architecture.md'
---

# climate_indices - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for climate_indices v2.4.0, decomposing the requirements from the PRD and Architecture into implementable stories.

## Requirements Inventory

### Functional Requirements

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

### Non-Functional Requirements

**Pattern Compliance & Refactoring Safety (Track 0)**

- NFR-PATTERN-EQUIV: Numerical Equivalence During Refactoring (tolerance 1e-8 for float64)
- NFR-PATTERN-COVERAGE: 100% Pattern Compliance Dashboard (6 patterns × 7 indices = 42 compliance points)
- NFR-PATTERN-MAINT: Maintainability Through Consistency (target: 30% reduction in time-to-fix)

**Performance Targets**

- NFR-PM-PERF: Penman-Monteith Numerical Precision (±0.05 mm/day for examples, ±0.01 kPa for intermediates)
- NFR-PALMER-SEQ: Palmer Sequential Time Constraint (chunk spatial dims, NOT temporal)
- NFR-PALMER-PERF: Palmer xarray ≥80% speed of multiprocessing baseline

**Reliability & Validation**

- NFR-MULTI-OUT: Multi-Output Adapter Pattern Stability (stack/unpack workaround for xarray Issue #1815)
- NFR-EDDI-VAL: EDDI NOAA Reference Validation Tolerance (1e-5 for non-parametric ranking FP accumulation)

### Additional Requirements

**From Architecture Decisions:**

- **Decision 1 (NOAA Provenance Protocol)**: Establish JSON-based provenance metadata for external reference datasets
  - Required fields: source, url, download_date, subset_description, checksum_sha256, fixture_version, validation_tolerance
  - Location: `tests/fixture/noaa-eddi-*/provenance.json`
  - **BLOCKS Track 2 (EDDI)**: Must establish BEFORE downloading NOAA reference data

- **Decision 2 (Palmer Module Organization)**: Keep Palmer xarray wrapper in palmer.py (NOT separate palmer_xarray.py)
  - Module size threshold: ~1,060 lines acceptable (912 existing + 150 new xarray wrapper)
  - Extraction trigger: If exceeds 1,400 lines during Track 3, consider extracting to palmer/ package
  - **BLOCKS Track 3**: Defines where xarray wrapper implementation goes

- **Decision 3 (Property-Based Test Strategy)**: Comprehensive hypothesis-based testing
  - Custom climate data generators required (precipitation, temperature, soil water capacity)
  - Effort: 50-60 hours per index (not boilerplate work)
  - Coverage: Common properties (all indices) + index-specific properties (3-5 per index)

- **Decision 4 (Exception Migration Strategy)**: Per-module incremental migration (NOT big-bang)
  - Replace ValueError → structured exceptions when touching modules for other patterns
  - Exception hierarchy: ClimateIndicesError → InvalidArgumentError, InsufficientDataError, ComputationError
  - ~50 instances across all legacy functions

- **Decision 5 (CF Metadata Registry)**: Create separate cf_metadata_registry.py module
  - New module: `src/climate_indices/cf_metadata_registry.py`
  - 7 new metadata entries in v2.4.0: pdsi, phdi, pmdi, z_index, eddi, pnp, eto_penman_monteith
  - Coupling threshold: Extract apply_cf_metadata() helper if 3+ modules duplicate logic

**Infrastructure Requirements:**

- **Palmer Benchmark Baseline (BLOCKER for Track 3)**: Must measure current multiprocessing CLI performance BEFORE implementing Palmer xarray
  - New file: `tests/test_benchmark_palmer.py`
  - Measure: Wall-clock time for 360×180 grid, 240 months
  - Required by: NFR-PALMER-PERF (cannot validate ≥80% speed claim without baseline)
  - **BLOCKS Track 3**: Must complete in Track 0 or Track 1

- **Dispatcher Size Monitoring**: typed_public_api.py extraction threshold
  - Current: 210 lines
  - v2.4.0 projection: 350-400 lines (4+ new index dispatchers)
  - Extraction threshold: 300 lines (not 400)
  - Pattern: Decorator factory for numpy/xarray routing to `src/climate_indices/dispatchers.py`

**New Files to Create:**

- Source modules: 1 (cf_metadata_registry.py)
- Test files: 12+
  - test_benchmark_palmer.py (baseline measurement)
  - test_reference_validation.py (NOAA EDDI)
  - Expanded: test_properties.py, test_equivalence.py
- Fixture directories: 3 (noaa-eddi-1month/, noaa-eddi-3month/, noaa-eddi-6month/)
- Documentation: Property-based test guide, Palmer xarray usage guide

**No Starter Template**: Brownfield project, existing src-layout remains unchanged


## Agent Orchestration Guide

### Overview

climate_indices v2.4.0 implements 30 functional requirements across 4 parallel tracks. To enable efficient parallel agent execution with minimal merge conflicts, we organize work into **5 specialized agents** across **5 execution phases** with explicit synchronization gates.

**Critical Challenge:** typed_public_api.py is touched by ALL 4 tracks; xarray_adapter.py by 3 tracks; palmer.py split between 2 agents.

### Agent Team Composition

| Agent | Track | Primary Files | Secondary Files |
|-------|-------|---------------|-----------------|
| **Alpha** | 0 (Patterns) | exceptions.py, cf_metadata_registry.py (NEW), palmer.py lines 1-913 | typed_public_api.py, xarray_adapter.py |
| **Beta** | 1 (PM-ET) | eto.py, test_penman_monteith.py (NEW) | cf_metadata_registry.py, xarray_adapter.py, typed_public_api.py |
| **Gamma** | 2 (EDDI) | tests/fixture/noaa-eddi-*/, test_noaa_eddi_reference.py (NEW) | cf_metadata_registry.py, xarray_adapter.py, typed_public_api.py |
| **Delta** | 3 (Palmer) | palmer.py lines 914+ (xarray wrapper), test_palmer_equivalence.py (NEW) | cf_metadata_registry.py, typed_public_api.py |
| **Omega** | Integration | __init__.py, merge validation | typed_public_api.py (merge append-blocks) |

**Agent Handoff:** Alpha completes Palmer structlog (Story 1.6) → Delta begins xarray wrapper (Story 4.1)

### Execution Phases

**Phase 0 (Foundation):** Stories 1.1-1.2 | Agent: Alpha | Duration: 1-2 days
- Deliverables: exceptions hierarchy, cf_metadata_registry.py stub
- Gate 0: Full test suite green, mypy --strict passes

**Phase 1 (Parallel Core):** Stories 1.3-1.10, 2.1-2.6 | Agents: Alpha ∥ Beta | Duration: 2-3 weeks
- Alpha: Palmer structlog, PNP/PCI xarray, property tests
- Beta: PM-ET FAO56 validation, baseline measurement
- Gate 1: Palmer structlog complete, FAO56 within ±0.05 mm/day
- Omega Action: Merge typed_public_api.py, update __init__.py

**Phase 2 (Dependent Tracks):** Stories 3.1-3.6, 4.1-4.8 | Agents: Gamma ∥ Delta | Duration: 3-4 weeks
- Gamma: EDDI NOAA validation, CLI integration
- Delta: Palmer xarray multi-output
- Gate 2: EDDI within 1e-5, Palmer xarray equiv 1e-8, perf ≥80%
- Omega Action: Merge all append-blocks, update __init__.py

**Phase 3 (Audit):** Stories 1.11, 5.1-5.2 | Agents: Alpha, Gamma | Duration: 3-5 days
- Deliverables: 42/42 compliance, scPDSI stub, property test audit
- Gate 3: Pattern compliance 100%, all property tests pass

**Phase 4 (Final Validation):** Stories 1.12, 2.7, 3.7, 4.9, 5.3 | Agents: All | Duration: 3-5 days
- Gate 4 (FINAL): All 30 FRs satisfied, all 8 NFRs validated, coverage >85%

### Shared File Protocol

**typed_public_api.py:** Append-only with section comment headers. Omega merges at gates. Extract at 300 lines.

**cf_metadata_registry.py:** Alpha creates stub Phase 0. Agents append with track comments. No shared state.

**xarray_adapter.py:** Append self-contained functions at end. No decorator modifications.

**__init__.py:** Omega exclusive - updates at gates only.

**palmer.py:** Alpha owns lines 1-913 (structlog), Delta owns 914+ (xarray wrapper). Handoff after Story 1.6.


## FR Coverage Map

| FR Code | Story | Agent | Phase |
|---------|-------|-------|-------|
| FR-PATTERN-001 | 1.3 | Alpha | 1 |
| FR-PATTERN-002 | 1.4 | Alpha | 1 |
| FR-PATTERN-003 | 1.8 | Alpha | 1 |
| FR-PATTERN-004 | 1.9 | Alpha | 1 |
| FR-PATTERN-005 | 1.5 | Alpha | 1 |
| FR-PATTERN-006 | 1.5 | Alpha | 1 |
| FR-PATTERN-007 | 1.6 | Alpha | 1 |
| FR-PATTERN-008 | 1.7 | Alpha | 1 |
| FR-PATTERN-009 | 1.1 | Alpha | 0 |
| FR-PATTERN-010 | 1.10 | Alpha | 1 |
| FR-PATTERN-011 | 1.10 | Alpha | 1 |
| FR-PATTERN-012 | 1.10 | Alpha | 1 |
| FR-PM-001 through FR-PM-006 | 2.1-2.6 | Beta | 1 |
| FR-EDDI-001 through FR-EDDI-004 | 3.2-3.6 | Gamma | 2 |
| FR-PNP-001 | 1.3 | Alpha | 1 |
| FR-SCPDSI-001 | 5.1 | Gamma | 3 |
| FR-PALMER-001 through FR-PALMER-007 | 4.2-4.8 | Delta | 2 |

**Coverage:** All 30 FRs mapped to stories. No forward dependencies.


## Epic and Story Definitions

Due to output token limits, the complete 38 stories have been structured as follows:

- **Epic 1: Canonical Pattern Completion** (12 stories) - Agent Alpha, Phases 0-1
- **Epic 2: PM-ET Foundation** (7 stories) - Agent Beta, Phase 1
- **Epic 3: EDDI/PNP/scPDSI Coverage** (7 stories) - Agent Gamma, Phase 2
- **Epic 4: Palmer Multi-Output** (9 stories) - Agent Delta, Phase 2
- **Epic 5: Cross-Cutting Validation** (3 stories) - Multiple agents, Phases 3-4

**Key Stories:**

- **Story 1.1:** Structured exception hierarchy foundation
- **Story 1.2:** CF metadata registry creation
- **Story 1.6:** Palmer structlog migration (CRITICAL PATH - blocks Track 3)
- **Story 2.4:** FAO56 worked example validation
- **Story 2.6:** Palmer performance baseline measurement
- **Story 3.1:** Resolve PR #597 EDDI merge conflicts (NOT greenfield)
- **Story 3.2:** NOAA provenance protocol (covers BOTH EDDI and Palmer)
- **Story 3.3:** NOAA EDDI reference validation (tolerance 1e-5)
- **Story 4.1:** Palmer xarray handoff validation
- **Story 4.2:** palmer_xarray() manual wrapper (Pattern C)
- **Story 4.8:** Palmer equivalence and provenance (xarray vs numpy, NOT vs pdi.f)
- **Story 5.3:** Final v2.4.0 validation (all 30 FRs, all 8 NFRs)

**Three Critical Corrections:**

1. **PR #597:** Story 3.1 resolves merge conflicts (NOT greenfield EDDI implementation)
2. **Palmer validation:** Story 4.8 validates xarray wrapper vs Python numpy (NOT vs pdi.f Fortran)
3. **NOAA provenance:** Story 3.2 protocol covers BOTH EDDI (new) AND Palmer (retroactive in 4.8)

## Summary Statistics

- **Total Stories:** 38 across 5 epics
- **Agent Workload:** Alpha (14), Beta (7), Gamma (8), Delta (9), Omega (3 gates + validation)
- **Phase Distribution:** Phase 0 (2), Phase 1 (16 parallel), Phase 2 (14 parallel), Phase 3 (3), Phase 4 (4)
- **Critical Path:** 1.1 → 1.2 → 1.6 → 4.1 → 4.2-4.9 (~5-6 weeks)
- **Parallelization Savings:** ~5-7 weeks vs sequential execution
- **FR Coverage:** 30/30 FRs mapped, 8/8 NFRs validated

**Document Version:** v2.4.0
**Last Updated:** 2026-02-16
**Status:** Complete - Ready for Implementation

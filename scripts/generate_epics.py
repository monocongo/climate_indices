#!/usr/bin/env python3
# /// pyproject
# [context]
# dependencies = []
# ///
"""
Generate comprehensive epics.md with agent orchestration for climate_indices v2.4.0.

This script creates the complete epic breakdown with:
- Agent orchestration metadata (5 agents, 5 phases)
- FR coverage mapping
- 5 epics with ~38 stories
- Phase gate checklists
"""

from pathlib import Path


def generate_epics_md() -> str:
    """Generate the complete epics.md content."""

    # Read the existing requirements inventory (lines 1-134)
    existing_file = Path("_bmad-output/planning-artifacts/epics.md")
    with existing_file.open() as f:
        lines = f.readlines()

    # Preserve lines 1-134 (requirements inventory)
    preserved_content = "".join(lines[:134])

    # Build the new content
    content = preserved_content + "\n"

    # Section 2: Agent Orchestration Guide
    content += """
## Agent Orchestration Guide

### Overview

climate_indices v2.4.0 implements 30 functional requirements across 4 parallel tracks. To enable efficient parallel agent execution with minimal merge conflicts, we organize work into **5 specialized agents** across **5 execution phases** with explicit synchronization gates.

**Critical Challenge:** The codebase has clear module boundaries (palmer.py, eto.py, exceptions.py are semi-isolated) but also high-contention shared files:
- `typed_public_api.py` — touched by ALL 4 tracks
- `xarray_adapter.py` — touched by 3 tracks
- `palmer.py` — split between 2 agents (structlog vs xarray wrapper)

Without orchestration metadata, these shared files create merge conflict bottlenecks.

---

### Agent Team Composition

| Agent | Track | Primary Files (Exclusive Write) | Secondary Files (Append-Only) |
|-------|-------|--------------------------------|------------------------------|
| **Alpha** | 0 (Patterns) | `exceptions.py`<br>`cf_metadata_registry.py` (NEW)<br>`palmer.py` lines 1-913 (structlog only) | `typed_public_api.py`<br>`xarray_adapter.py` |
| **Beta** | 1 (PM-ET) | `eto.py`<br>`test_penman_monteith.py` (NEW) | `cf_metadata_registry.py`<br>`xarray_adapter.py`<br>`typed_public_api.py` |
| **Gamma** | 2 (EDDI) | `tests/fixture/noaa-eddi-*/`<br>`test_noaa_eddi_reference.py` (NEW)<br>`test_eddi_adapter.py` (NEW) | `cf_metadata_registry.py`<br>`xarray_adapter.py`<br>`__main__.py`<br>`typed_public_api.py` |
| **Delta** | 3 (Palmer) | `palmer.py` lines 914+ (xarray wrapper only)<br>`test_palmer_equivalence.py` (NEW) | `cf_metadata_registry.py`<br>`typed_public_api.py` |
| **Omega** | Integration | `__init__.py`<br>Final merge validation | `typed_public_api.py` (merge all append-blocks) |

**Agent Handoff Protocol:**
- **Alpha → Delta handoff**: After Story 1.6 (Palmer structlog complete), Delta can begin xarray wrapper work
- **File section ownership**: Alpha owns `palmer.py` lines 1-913, Delta owns lines 914+ (NEW xarray wrapper)
- **Omega role**: Merges append-only sections in shared files at phase gates, updates `__init__.py` exports

---

### Execution Phases

```
Phase 0 (Foundation): Stories 1.1, 1.2
  Agent: Alpha only
  Duration: 1-2 days
  Deliverables:
    - Structured exceptions hierarchy complete
    - cf_metadata_registry.py created and importable
  Gate 0:
    ✓ Full test suite green
    ✓ mypy --strict passes
    ✓ exceptions importable from all modules

Phase 1 (Parallel Core): Stories 1.3-1.10, 2.1-2.6
  Agents: Alpha ∥ Beta (parallel execution)
  Duration: 2-3 weeks
  Deliverables:
    - Palmer structlog migration complete (Alpha)
    - FAO56 PM-ET validated (Beta)
    - PNP/PCI xarray adapters (Alpha)
    - Benchmark infrastructure (Beta)
  Gate 1:
    ✓ Palmer structlog lifecycle events working
    ✓ FAO56 Examples 17 & 18 within ±0.05 mm/day
    ✓ mypy --strict passes
    ✓ Performance baseline established
  Omega Action:
    → Merge typed_public_api.py append blocks
    → Update __init__.py exports

Phase 2 (Dependent Tracks): Stories 3.1-3.6, 4.1-4.8
  Agents: Gamma ∥ Delta (parallel execution)
  Duration: 3-4 weeks
  Dependencies: Phase 1 complete (PM-ET available, Palmer structlog done)
  Deliverables:
    - EDDI NOAA validation (Gamma)
    - Palmer xarray multi-output (Delta)
    - EDDI CLI integration (Gamma)
  Gate 2:
    ✓ EDDI validates against NOAA within 1e-5
    ✓ Palmer xarray equivalence within 1e-8
    ✓ Palmer performance ≥80% of baseline
    ✓ All multi-output scenarios pass
  Omega Action:
    → Merge typed_public_api.py append blocks
    → Merge cf_metadata_registry.py entries
    → Update __init__.py exports

Phase 3 (Audit): Stories 1.11, 5.1, 5.2
  Agents: Alpha, Gamma
  Duration: 3-5 days
  Deliverables:
    - 42/42 pattern compliance verified
    - scPDSI stub interface
    - Property-based test expansion
  Gate 3:
    ✓ Pattern compliance dashboard shows 100%
    ✓ scPDSI stub documented
    ✓ All property tests pass

Phase 4 (Final Validation): Stories 1.12, 2.7, 3.7, 4.9, 5.3
  Agents: All agents contribute
  Duration: 3-5 days
  Deliverables:
    - Integration tests across all tracks
    - Documentation complete
    - Release preparation
  Gate 4 (FINAL):
    ✓ Full test suite green (all 1400+ test lines)
    ✓ mypy --strict passes
    ✓ ruff check clean
    ✓ Coverage >85%
    ✓ All 30 FRs satisfied
    ✓ All 8 NFRs validated
```

---

### Shared File Protocol

#### typed_public_api.py (ALL tracks append)
**Current:** 210 lines
**Projected:** 350-400 lines
**Protocol:**
- Agents append in clearly marked sections with comment headers
- Example section header: `# ========== Track 0: PNP/PCI Dispatchers (Agent Alpha) ==========`
- Omega merges at Phase 1 and Phase 2 gates
- **Extraction trigger:** If file exceeds 300 lines, Omega extracts dispatcher pattern to `dispatchers.py`

#### cf_metadata_registry.py (NEW - 3 tracks append)
**Protocol:**
- Alpha creates stub in Phase 0 with CF_METADATA dict structure
- Agents append entries with track comment: `# Track 1: PM-ET`
- No shared state - pure data structure
- Omega validates no duplicate keys at gates

#### xarray_adapter.py (3 tracks append)
**Protocol:**
- Append self-contained adapter functions at end of file
- No modifications to existing decorators or helpers
- Section comments mark ownership: `# Track 0: PNP Adapter (Alpha)`
- Omega validates no function name collisions

#### __init__.py (Omega exclusive)
**Protocol:**
- NO agent modifications during phases
- Omega updates exports at Phase 1 and Phase 2 gates only
- Prevents import conflicts during parallel development

#### palmer.py (2 agents, split ownership)
**Special handoff protocol:**
- **Alpha (Phase 1):** Owns lines 1-913, migrates stdlib logging → structlog
- **Delta (Phase 2):** Owns lines 914+ (NEW), adds ~150-line xarray wrapper
- **Handoff trigger:** Story 1.6 completion (Palmer structlog done)
- **Validation:** Story 4.1 confirms clean handoff (imports work, no conflicts)

---

## FR Coverage Map

| FR Code | FR Title | Story | Agent | Phase |
|---------|----------|-------|-------|-------|
| FR-PATTERN-001 | percentage_of_normal xarray + CF metadata | 1.3 | Alpha | 1 |
| FR-PATTERN-002 | pci xarray + CF metadata | 1.4 | Alpha | 1 |
| FR-PATTERN-003 | eto_thornthwaite typed_public_api entry | 1.8 | Alpha | 1 |
| FR-PATTERN-004 | eto_hargreaves typed_public_api entry | 1.9 | Alpha | 1 |
| FR-PATTERN-005 | percentage_of_normal typed_public_api entry | 1.5 | Alpha | 1 |
| FR-PATTERN-006 | pci typed_public_api entry | 1.5 | Alpha | 1 |
| FR-PATTERN-007 | Palmer structlog migration | 1.6 | Alpha | 1 |
| FR-PATTERN-008 | eto_thornthwaite structlog lifecycle | 1.7 | Alpha | 1 |
| FR-PATTERN-009 | Structured exceptions for all legacy functions | 1.1 | Alpha | 0 |
| FR-PATTERN-010 | percentage_of_normal property-based tests | 1.10 | Alpha | 1 |
| FR-PATTERN-011 | pci property-based tests | 1.10 | Alpha | 1 |
| FR-PATTERN-012 | Expanded SPEI + Palmer property-based tests | 1.10 | Alpha | 1 |
| FR-PM-001 | Penman-Monteith FAO56 Core Calculation | 2.1 | Beta | 1 |
| FR-PM-002 | Atmospheric Parameter Helpers | 2.2 | Beta | 1 |
| FR-PM-003 | Vapor Pressure Helpers | 2.2 | Beta | 1 |
| FR-PM-004 | Humidity Pathway Dispatcher | 2.3 | Beta | 1 |
| FR-PM-005 | FAO56 Worked Example Validation | 2.4 | Beta | 1 |
| FR-PM-006 | PM-ET xarray Adapter | 2.5 | Beta | 1 |
| FR-EDDI-001 | NOAA Reference Dataset Validation | 3.2, 3.3 | Gamma | 2 |
| FR-EDDI-002 | EDDI xarray Adapter | 3.4 | Gamma | 2 |
| FR-EDDI-003 | EDDI CLI Integration | 3.5 | Gamma | 2 |
| FR-EDDI-004 | EDDI PET Method Documentation | 3.6 | Gamma | 2 |
| FR-PNP-001 | PNP xarray Adapter | 1.3 | Alpha | 1 |
| FR-SCPDSI-001 | scPDSI Stub Interface | 5.1 | Gamma | 3 |
| FR-PALMER-001 | palmer_xarray() Manual Wrapper | 4.2 | Delta | 2 |
| FR-PALMER-002 | Multi-Output Dataset Return | 4.3 | Delta | 2 |
| FR-PALMER-003 | AWC Spatial Parameter Handling | 4.4 | Delta | 2 |
| FR-PALMER-004 | params_dict JSON Serialization | 4.5 | Delta | 2 |
| FR-PALMER-005 | Palmer CF Metadata Registry | 4.6 | Delta | 2 |
| FR-PALMER-006 | typed_public_api @overload Signatures | 4.7 | Delta | 2 |
| FR-PALMER-007 | NumPy vs xarray Equivalence Tests | 4.8 | Delta | 2 |

**Coverage Validation:**
- ✓ All 30 FRs mapped to stories
- ✓ Every story has assigned agent and phase
- ✓ No forward dependencies (stories depend only on earlier phases)
- ✓ Critical path explicit (Story 1.6 → Story 4.2)

---

## Epic 1: Canonical Pattern Completion

**Goal:** Apply v2.3.0-established patterns to ALL remaining indices for 100% pattern consistency

**User Value:** Developers experience consistent APIs across all 7 indices; users get uniform error messages and logging; maintainers benefit from reduced cognitive load

**FR Coverage:** FR-PATTERN-001 through FR-PATTERN-012 (12 FRs)

**NFR Coverage:** NFR-PATTERN-EQUIV, NFR-PATTERN-COVERAGE, NFR-PATTERN-MAINT

**Dependencies:** None (enables Track 3)

**Blocks:** Epic 4 (Palmer structlog must complete before Palmer xarray work)

---

### Story 1.1: Foundation - Structured Exception Hierarchy

**Agent:** Alpha | **Phase:** 0 | **Depends On:** [] | **Blocks:** [1.3, 1.4, 1.6]

**Primary Files:** `src/climate_indices/exceptions.py`
**Secondary Files:** none

As a library developer, I want a complete structured exception hierarchy, So that I can replace generic ValueError with actionable, context-rich exceptions during pattern migrations.

**Acceptance Criteria:**

**Given** the existing ClimateIndicesError base class
**When** I audit all legacy ValueError instances
**Then** I create specialized exception classes:
- `InvalidArgumentError` (with shape=, expected= context)
- `InsufficientDataError` (with required=, provided= context)
- `ComputationError` (with algorithm=, state= context)

**And** all exception classes use keyword-only context attributes
**And** docstrings provide usage examples

**Validation:** `uv run pytest tests/test_exceptions.py -v`
**Gate:** All exception classes importable, docstrings complete
**FR Coverage:** FR-PATTERN-009

---

### Story 1.2: Foundation - CF Metadata Registry

**Agent:** Alpha | **Phase:** 0 | **Depends On:** [] | **Blocks:** [2.5, 3.4, 4.6]

**Primary Files:** `src/climate_indices/cf_metadata_registry.py` (NEW)
**Secondary Files:** none

As a scientific data engineer, I want a centralized CF metadata registry, So that all xarray outputs have consistent, CF-compliant attributes without code duplication.

**Acceptance Criteria:**

**Given** v2.4.0 requires 7 new CF metadata entries
**When** I create cf_metadata_registry.py module
**Then** module contains CF_METADATA dict with TypedDict structure
**And** initial entries for existing indices (spi, spei) are migrated from xarray_adapter.py

**And** module docstring documents CF Metadata Conventions v1.10 compliance
**And** structure includes: long_name, units, standard_name (optional), references, valid_range (optional)

**Validation:** `uv run python -c "from climate_indices.cf_metadata_registry import CF_METADATA; print(CF_METADATA['spi'])"`
**Gate:** Module importable, dict structure validated, mypy --strict passes
**FR Coverage:** Enables FR-PATTERN-001, FR-PATTERN-002, FR-PM-006, FR-EDDI-002, FR-PALMER-005

---

### Story 1.3: PNP xarray Adapter with CF Metadata

**Agent:** Alpha | **Phase:** 1 | **Depends On:** [1.1, 1.2] | **Blocks:** []

**Primary Files:** `src/climate_indices/indices.py` (percentage_of_normal function)
**Secondary Files:** `xarray_adapter.py`, `cf_metadata_registry.py`

As a climate researcher, I want percentage_of_normal to support xarray inputs, So that I can compute percent of normal precipitation on gridded datasets with CF-compliant metadata.

**Acceptance Criteria:**

**Given** existing numpy percentage_of_normal() function
**When** I apply @xarray_adapter decorator
**Then** CF metadata entry added to registry:
- `long_name="Percent of Normal Precipitation"`
- `units="%"`
- `references` to NIDIS/drought.gov methodology

**And** equivalence test validates numpy path == xarray path (tolerance 1e-8)
**And** coordinate preservation verified (lat, lon, time attrs propagate)

**Validation:** `uv run pytest tests/test_equivalence.py::test_percentage_of_normal_xarray_equivalence -v`
**Gate:** Equivalence test passes, CF metadata compliant
**FR Coverage:** FR-PATTERN-001, FR-PNP-001

---

### Story 1.4: PCI xarray Adapter with CF Metadata

**Agent:** Alpha | **Phase:** 1 | **Depends On:** [1.1, 1.2] | **Blocks:** []

**Primary Files:** `src/climate_indices/indices.py` (pci function)
**Secondary Files:** `xarray_adapter.py`, `cf_metadata_registry.py`

As a precipitation analyst, I want pci to support xarray inputs, So that I can compute seasonal concentration on gridded datasets with proper metadata.

**Acceptance Criteria:**

**Given** existing numpy pci() function
**When** I apply @xarray_adapter decorator
**Then** CF metadata entry added to registry:
- `long_name="Precipitation Concentration Index"`
- `units=""` (dimensionless)
- `references` to Oliver (1980) methodology

**And** input validation enforces 365/366 daily values (raises InvalidArgumentError otherwise)
**And** equivalence test validates numpy path == xarray path (tolerance 1e-8)

**Validation:** `uv run pytest tests/test_equivalence.py::test_pci_xarray_equivalence -v`
**Gate:** Equivalence test passes, input validation enforced
**FR Coverage:** FR-PATTERN-002

---

### Story 1.5: PNP/PCI typed_public_api Entries

**Agent:** Alpha | **Phase:** 1 | **Depends On:** [1.3, 1.4] | **Blocks:** []

**Primary Files:** `src/climate_indices/typed_public_api.py`
**Secondary Files:** none

As a type-safe application developer, I want @overload signatures for percentage_of_normal and pci, So that mypy can distinguish numpy vs xarray return types at compile time.

**Acceptance Criteria:**

**Given** PNP and PCI now support xarray
**When** I add @overload signatures to typed_public_api.py
**Then** numpy overload: `percentage_of_normal(np.ndarray, ...) -> np.ndarray`
**And** xarray overload: `percentage_of_normal(xr.DataArray, ...) -> xr.DataArray`
**And** runtime dispatcher uses `isinstance(precip, xr.DataArray)` detection

**And** same pattern for pci: numpy→numpy, xarray→xarray overloads
**And** mypy --strict validates both functions

**Validation:** `uv run mypy --strict src/climate_indices/typed_public_api.py`
**Gate:** mypy passes, both functions have complete @overload sets
**FR Coverage:** FR-PATTERN-005, FR-PATTERN-006

---

### Story 1.6: Palmer structlog Migration (CRITICAL PATH)

**Agent:** Alpha | **Phase:** 1 | **Depends On:** [1.1] | **Blocks:** [4.1, 4.2]

**Primary Files:** `src/climate_indices/palmer.py` (lines 1-913 only)
**Secondary Files:** `exceptions.py`

As a production engineer, I want Palmer to use structlog with lifecycle events, So that operational monitoring can track computation state and debug water balance failures.

**Acceptance Criteria:**

**Given** palmer.py currently uses stdlib logging (912 lines)
**When** I migrate to structlog
**Then** import: `from climate_indices.logging_config import get_logger`
**And** lifecycle events: `calculation_started`, `calculation_completed`, `calculation_failed`
**And** bind pattern: `_logger.bind(calculation="pdsi", data_shape=precips.shape, awc=awc)`

**And** replace all ~15-20 ValueError instances with structured exceptions:
- AWC validation → `InvalidArgumentError`
- Shape mismatch → `InvalidArgumentError`
- Computation failures → `ComputationError`

**And** log levels match SPI/SPEI pattern (INFO for lifecycle, DEBUG for state)
**And** zero stdlib logging imports remain

**Validation:** `uv run pytest tests/test_palmer.py -v --log-cli-level=DEBUG`
**Gate:** All lifecycle events logged, exceptions structured, full test suite green
**FR Coverage:** FR-PATTERN-007, FR-PATTERN-009 (Palmer portion)

**CRITICAL PATH NOTE:** This story completion enables Story 4.1 (Palmer xarray handoff). Agent Delta cannot begin xarray wrapper work until Alpha completes this story.

---

### Story 1.7: eto_thornthwaite Lifecycle Completion

**Agent:** Alpha | **Phase:** 1 | **Depends On:** [1.1] | **Blocks:** []

**Primary Files:** `src/climate_indices/eto.py`
**Secondary Files:** `exceptions.py`

As a PET calculation user, I want eto_thornthwaite to have complete lifecycle logging, So that I can debug temperature-driven ETo calculations consistently with other indices.

**Acceptance Criteria:**

**Given** eto_thornthwaite currently has logger instance but no lifecycle bind
**When** I add lifecycle event pattern
**Then** bind context: `_logger.bind(calculation="eto_thornthwaite", data_shape=temp.shape, latitude=latitude_degrees)`
**And** events: `calculation_started`, `calculation_completed` with timing

**And** replace ValueError instances with InvalidArgumentError for:
- Temperature validation
- Latitude validation

**And** match eto_hargreaves pattern exactly

**Validation:** `uv run pytest tests/test_eto.py::test_eto_thornthwaite -v --log-cli-level=DEBUG`
**Gate:** Lifecycle events logged, exceptions structured
**FR Coverage:** FR-PATTERN-008, FR-PATTERN-009 (eto_thornthwaite portion)

---

### Story 1.8: eto_thornthwaite typed_public_api Entry

**Agent:** Alpha | **Phase:** 1 | **Depends On:** [1.7] | **Blocks:** []

**Primary Files:** `src/climate_indices/typed_public_api.py`
**Secondary Files:** none

As a type-safe application developer, I want @overload signatures for eto_thornthwaite, So that mypy can validate temperature input types.

**Acceptance Criteria:**

**Given** eto_thornthwaite supports xarray
**When** I add @overload signatures
**Then** numpy overload: `eto_thornthwaite(np.ndarray, float, ...) -> np.ndarray`
**And** xarray overload: `eto_thornthwaite(xr.DataArray, float, ...) -> xr.DataArray`
**And** runtime dispatcher based on input type

**And** follows SPI/SPEI @overload pattern
**And** mypy --strict passes

**Validation:** `uv run mypy --strict src/climate_indices/typed_public_api.py`
**Gate:** mypy passes
**FR Coverage:** FR-PATTERN-003

---

### Story 1.9: eto_hargreaves typed_public_api Entry

**Agent:** Alpha | **Phase:** 1 | **Depends On:** [] | **Blocks:** []

**Primary Files:** `src/climate_indices/typed_public_api.py`
**Secondary Files:** none

As a type-safe application developer, I want @overload signatures for eto_hargreaves, So that mypy can validate temperature extremes input types.

**Acceptance Criteria:**

**Given** eto_hargreaves supports xarray
**When** I add @overload signatures
**Then** numpy overload: `eto_hargreaves(np.ndarray, np.ndarray, float, ...) -> np.ndarray`
**And** xarray overload: `eto_hargreaves(xr.DataArray, xr.DataArray, float, ...) -> xr.DataArray`

**And** mypy --strict passes

**Validation:** `uv run mypy --strict src/climate_indices/typed_public_api.py`
**Gate:** mypy passes
**FR Coverage:** FR-PATTERN-004

---

### Story 1.10: Property-Based Test Expansion

**Agent:** Alpha | **Phase:** 1 | **Depends On:** [1.3, 1.4, 1.6] | **Blocks:** []

**Primary Files:** `tests/test_properties.py`
**Secondary Files:** none

As a quality engineer, I want property-based tests for PNP, PCI, and expanded Palmer coverage, So that mathematical invariants are continuously validated against edge cases.

**Acceptance Criteria:**

**Given** hypothesis framework available
**When** I create property test classes
**Then** PNP properties:
- Boundedness: pnp >= 0 (precipitation non-negative)
- Shape preservation: output.shape == input.shape
- NaN propagation: np.isnan(input[i]) → np.isnan(output[i])
- Linear scaling: pnp(2×p, 2×p_mean) = pnp(p, p_mean)

**And** PCI properties:
- Range: 0 <= pci <= 100
- Input length validation: raises InvalidArgumentError if not 365/366
- NaN handling: single NaN propagates
- Zero precipitation: all-zero input returns valid PCI

**And** Palmer properties (expanded):
- PHDI bounded range (currently only PDSI tested)
- PMDI bounded range
- Z-Index bounded range
- Sequential consistency (splitting time series changes results)

**And** SPEI properties (expanded beyond boundedness):
- Shape preservation
- NaN propagation from water balance
- Zero water balance → SPEI near 0

**Validation:** `uv run pytest tests/test_properties.py -v`
**Gate:** All property tests pass, hypothesis finds no counterexamples
**FR Coverage:** FR-PATTERN-010, FR-PATTERN-011, FR-PATTERN-012

---

### Story 1.11: Pattern Compliance Dashboard

**Agent:** Alpha | **Phase:** 3 | **Depends On:** [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10] | **Blocks:** []

**Primary Files:** `tests/test_pattern_compliance.py` (NEW)
**Secondary Files:** none

As a project maintainer, I want automated pattern compliance validation, So that I can verify 100% pattern coverage (6 patterns × 7 indices = 42 checkpoints).

**Acceptance Criteria:**

**Given** all Track 0 pattern stories complete
**When** I run pattern compliance tests
**Then** dashboard validates:
- xarray support: 7/7 indices (100%)
- typed_public_api entries: 7/7 indices (100%)
- CF metadata: 7/7 xarray outputs (100%)
- structlog: 7/7 indices with lifecycle events (100%)
- Structured exceptions: 100% of validation code
- Property-based tests: 7/7 indices (100%)

**And** test generates markdown table showing compliance matrix
**And** CI fails if any compliance point fails

**Validation:** `uv run pytest tests/test_pattern_compliance.py -v`
**Gate:** 42/42 compliance checkpoints pass
**FR Coverage:** NFR-PATTERN-COVERAGE

---

### Story 1.12: Track 0 Numerical Equivalence Validation

**Agent:** Alpha | **Phase:** 4 | **Depends On:** [all Track 0 stories] | **Blocks:** []

**Primary Files:** `tests/test_equivalence.py`
**Secondary Files:** none

As a scientific software engineer, I want comprehensive equivalence validation, So that I can prove pattern refactoring did not alter numerical results.

**Acceptance Criteria:**

**Given** all pattern migrations complete
**When** I run equivalence test suite
**Then** all before/after comparisons pass (tolerance 1e-8):
- percentage_of_normal: numpy vs xarray
- pci: numpy vs xarray
- Palmer: pre-structlog fixtures vs post-structlog

**And** full test suite passes (1400+ test lines)
**And** no algorithmic changes detected

**Validation:** `uv run pytest tests/test_equivalence.py -v`
**Gate:** All equivalence tests pass, zero numerical drift
**FR Coverage:** NFR-PATTERN-EQUIV

---

## Epic 2: PM-ET Foundation

**Goal:** Implement physics-based Penman-Monteith FAO56 evapotranspiration with scientific validation

**User Value:** Flash drought researchers can use PM-ET for accurate EDDI calculations; hydrologists get physics-based PET that accounts for wind and humidity

**FR Coverage:** FR-PM-001 through FR-PM-006 (6 FRs)

**NFR Coverage:** NFR-PM-PERF

**Dependencies:** None (enables Epic 3 EDDI)

**Blocks:** Epic 3 (PM-ET required for EDDI scientific accuracy)

---

### Story 2.1: PM-ET Core Calculation Engine

**Agent:** Beta | **Phase:** 1 | **Depends On:** [] | **Blocks:** [2.2, 2.3, 2.4]

**Primary Files:** `src/climate_indices/eto.py`
**Secondary Files:** none

As a PET calculation user, I want eto_penman_monteith function, So that I can compute reference evapotranspiration using FAO56 Equation 6.

**Acceptance Criteria:**

**Given** FAO56 Equation 6 specification
**When** I implement eto_penman_monteith()
**Then** function signature:
```python
def eto_penman_monteith(
    temperature_min: np.ndarray,
    temperature_max: np.ndarray,
    temperature_mean: np.ndarray,
    wind_speed_2m: np.ndarray,
    net_radiation: np.ndarray,
    latitude_degrees: float,
    altitude_meters: float,
    dewpoint_celsius: np.ndarray | None = None,
    rh_max: np.ndarray | None = None,
    rh_min: np.ndarray | None = None,
    rh_mean: np.ndarray | None = None,
) -> np.ndarray:
```

**And** returns: mm/day (float64 precision required)
**And** docstring documents unit requirements and humidity pathway priority
**And** structlog lifecycle events: calculation_started, calculation_completed

**Validation:** `uv run pytest tests/test_eto.py::test_eto_penman_monteith_basic -v`
**Gate:** Function importable, basic computation works
**FR Coverage:** FR-PM-001

---

### Story 2.2: Atmospheric and Vapor Pressure Helpers

**Agent:** Beta | **Phase:** 1 | **Depends On:** [2.1] | **Blocks:** [2.3, 2.4]

**Primary Files:** `src/climate_indices/eto.py`
**Secondary Files:** none

As a PM-ET implementation developer, I want helper functions for FAO56 equations 7-13, So that vapor pressure and atmospheric calculations are independently testable.

**Acceptance Criteria:**

**Given** FAO56 equations 7-13 specification
**When** I implement helper functions
**Then** functions created:
- `_atm_pressure(altitude)` — Eq. 7: P = 101.3 × [(293 - 0.0065z)/293]^5.26
- `_psy_const(pressure)` — Eq. 8: γ = 0.000665 × P
- `_svp_from_t(temp)` — Eq. 11: e°(T) = 0.6108 × exp[17.27T/(T+237.3)]
- `_mean_svp(tmin, tmax)` — Eq. 12: averages SVP at extremes (NOT e°(Tmean))
- `_slope_svp(temp)` — Eq. 13: Δ = 4098 × e°(T) / (T+237.3)²

**And** exact FAO56 constants used: 0.6108, 17.27, 237.3
**And** type annotations: float → float or np.ndarray → np.ndarray
**And** unit tests validate known values (e.g., Uccle at 100m → 100.1 kPa)

**Validation:** `uv run pytest tests/test_eto.py::test_fao56_helpers -v`
**Gate:** All helper functions pass unit tests
**FR Coverage:** FR-PM-002, FR-PM-003

---

### Story 2.3: Humidity Pathway Dispatcher

**Agent:** Beta | **Phase:** 1 | **Depends On:** [2.2] | **Blocks:** [2.4]

**Primary Files:** `src/climate_indices/eto.py`
**Secondary Files:** none

As a PM-ET user, I want automatic vapor pressure pathway selection, So that I don't need to compute vapor pressure manually based on available humidity data.

**Acceptance Criteria:**

**Given** FAO56 equations 14-19 specify humidity pathways
**When** I implement dispatcher logic in eto_penman_monteith()
**Then** priority order:
1. dewpoint_celsius → Eq. 14: ea = e°(Tdew)
2. rh_max + rh_min → Eq. 17: ea = [e°(Tmin)×RHmax + e°(Tmax)×RHmin] / 200
3. rh_mean → Eq. 19: ea = es × RHmean / 100

**And** raises ValueError if no humidity input provided
**And** logs selected pathway at DEBUG level (structlog)
**And** accepts explicit actual_vapor_pressure parameter (bypasses dispatcher)

**Validation:** `uv run pytest tests/test_eto.py::test_humidity_pathway_selection -v`
**Gate:** All 3 pathways validated, priority order enforced
**FR Coverage:** FR-PM-004

---

### Story 2.4: FAO56 Worked Example Validation

**Agent:** Beta | **Phase:** 1 | **Depends On:** [2.1, 2.2, 2.3] | **Blocks:** []

**Primary Files:** `tests/test_eto.py`
**Secondary Files:** none

As a scientific validation engineer, I want PM-ET to reproduce FAO56 published examples, So that I can prove implementation fidelity to the reference standard.

**Acceptance Criteria:**

**Given** FAO56 Examples 17 & 18 input data
**When** I run eto_penman_monteith with example inputs
**Then** Example 17 (Bangkok, April tropical monthly):
- Expected: 5.72 mm/day
- Tolerance: ±0.05 mm/day
- Test: `test_fao56_example_17_bangkok()`

**And** Example 18 (Uccle, 6 July temperate daily):
- Expected: 3.9 mm/day
- Tolerance: ±0.05 mm/day
- Test: `test_fao56_example_18_uccle()`

**And** input data embedded in test (no external files)
**And** tests validate intermediate values (SVP, slope, etc.) within ±0.01 kPa

**Validation:** `uv run pytest tests/test_eto.py::test_fao56_example_17_bangkok tests/test_eto.py::test_fao56_example_18_uccle -v`
**Gate:** Both examples pass within tolerance
**FR Coverage:** FR-PM-005

---

### Story 2.5: PM-ET xarray Adapter

**Agent:** Beta | **Phase:** 1 | **Depends On:** [1.2, 2.4] | **Blocks:** []

**Primary Files:** `src/climate_indices/eto.py`
**Secondary Files:** `xarray_adapter.py`, `cf_metadata_registry.py`, `typed_public_api.py`

As a gridded dataset user, I want eto_penman_monteith to support xarray inputs, So that I can compute PM-ET on spatial grids with CF-compliant metadata.

**Acceptance Criteria:**

**Given** numpy eto_penman_monteith() validated
**When** I apply @xarray_adapter decorator
**Then** CF metadata entry in registry:
- `long_name="Reference Evapotranspiration (Penman-Monteith FAO56)"`
- `units="mm day-1"`
- `standard_name="water_evapotranspiration_flux"` (CF standard name exists!)
- `references` includes Allen et al. 1998 DOI

**And** @overload signatures in typed_public_api.py:
- numpy path: `eto_penman_monteith(np.ndarray, ...) -> np.ndarray`
- xarray path: `eto_penman_monteith(xr.DataArray, ...) -> xr.DataArray`

**And** Dask compatibility: `dask="parallelized"` in apply_ufunc
**And** coordinate preservation verified

**Validation:** `uv run pytest tests/test_equivalence.py::test_eto_penman_monteith_xarray_equivalence -v`
**Gate:** Equivalence test passes (1e-8), CF metadata compliant
**FR Coverage:** FR-PM-006

---

### Story 2.6: Palmer Performance Baseline Measurement

**Agent:** Beta | **Phase:** 1 | **Depends On:** [] | **Blocks:** [4.8]

**Primary Files:** `tests/test_benchmark_palmer.py` (NEW)
**Secondary Files:** none

As a performance engineer, I want Palmer multiprocessing baseline measured, So that Track 3 xarray implementation can validate ≥80% performance claim (NFR-PALMER-PERF).

**Acceptance Criteria:**

**Given** current Palmer CLI uses multiprocessing over grid cells
**When** I create benchmark infrastructure
**Then** test measures:
- Synthetic dataset: 360×180 grid, 240 months (10 years)
- Current __main__.py multiprocessing path
- Wall-clock time (median of 10 runs)
- Memory usage (peak RSS)

**And** baseline stored as pytest benchmark fixture
**And** CI integration for regression tracking
**And** benchmark report includes system info (CPU, Python version)

**Validation:** `uv run pytest tests/test_benchmark_palmer.py --benchmark-only -v`
**Gate:** Baseline measurement complete, documented
**FR Coverage:** Enables NFR-PALMER-PERF validation

---

### Story 2.7: Track 1 Integration Validation

**Agent:** Beta | **Phase:** 4 | **Depends On:** [all Track 1 stories] | **Blocks:** []

**Primary Files:** `tests/test_integration.py`
**Secondary Files:** none

As a PM-ET user, I want end-to-end integration tests, So that I can verify PM-ET works correctly in realistic workflows.

**Acceptance Criteria:**

**Given** all Track 1 stories complete
**When** I run integration tests
**Then** scenarios validated:
- PM-ET with dewpoint humidity (most accurate pathway)
- PM-ET with RH extremes (preferred daily pathway)
- PM-ET with RH mean only (fallback pathway)
- PM-ET on gridded xarray dataset (spatial computation)
- PM-ET type safety (mypy validates dispatchers)

**And** performance validated: no regression vs baseline
**And** documentation complete: docstrings, algorithm guide

**Validation:** `uv run pytest tests/test_integration.py::test_pm_et_integration -v`
**Gate:** All integration scenarios pass
**FR Coverage:** All FR-PM-* requirements satisfied

---

## Epic 3: EDDI/PNP/scPDSI Coverage

**Goal:** Complete drought index catalog with NOAA reference validation and CLI integration

**User Value:** Flash drought researchers can use validated EDDI with PM-ET; operational users get CLI access; scPDSI roadmap established

**FR Coverage:** FR-EDDI-001 through FR-SCPDSI-001 (6 FRs)

**NFR Coverage:** NFR-EDDI-VAL

**Dependencies:** Epic 2 (PM-ET required for EDDI)

**Blocks:** None

---

### Story 3.1: Resolve PR #597 EDDI Merge Conflicts

**Agent:** Gamma | **Phase:** 2 | **Depends On:** [Phase 1 complete] | **Blocks:** [3.2, 3.3]

**Primary Files:** PR #597 changed files
**Secondary Files:** `__init__.py`, integration test files

As a maintainer, I want PR #597 cleanly integrated, So that EDDI foundation exists before adding NOAA validation (NOT greenfield EDDI implementation).

**Acceptance Criteria:**

**Given** PR #597 contains EDDI implementation (pending merge)
**When** I resolve merge conflicts with v2.4.0 branch
**Then** all conflicts resolved in:
- `indices.py` (EDDI algorithm)
- `__init__.py` (exports)
- Test files (if any conflicts)

**And** EDDI function importable: `from climate_indices import eddi`
**And** existing EDDI tests pass
**And** no regression in other indices

**Validation:** `uv run pytest tests/ -v` (full suite green)
**Gate:** PR #597 merged cleanly, EDDI available for Stories 3.2-3.6
**FR Coverage:** Prerequisite for FR-EDDI-001

**CRITICAL CORRECTION:** This story is about merge conflict resolution, NOT greenfield EDDI development. PR #597 already contains EDDI implementation.

---

### Story 3.2: NOAA Provenance Protocol Establishment

**Agent:** Gamma | **Phase:** 2 | **Depends On:** [3.1] | **Blocks:** [3.3, 4.8]

**Primary Files:** `tests/fixture/noaa-eddi-1month/provenance.json` (NEW)
**Secondary Files:** `tests/fixture/README.md`

As a test architect, I want JSON-based provenance metadata for external reference datasets, So that EDDI (and future Palmer) validations are reproducible and auditable.

**Acceptance Criteria:**

**Given** Architecture Decision 1 (NOAA Provenance Protocol)
**When** I establish provenance structure
**Then** create directories:
- `tests/fixture/noaa-eddi-1month/`
- `tests/fixture/noaa-eddi-3month/`
- `tests/fixture/noaa-eddi-6month/`

**And** each contains `provenance.json` with fields:
- source: "NOAA PSL EDDI CONUS Archive"
- url: Full download URL
- download_date: ISO 8601 format
- subset_description: Spatial/temporal extent
- checksum_sha256: File integrity validation
- fixture_version: "1.0" (explicit evolution tracking)
- validation_tolerance: 1e-5
- notes: Rationale for tolerance choice

**And** `tests/fixture/README.md` documents provenance protocol
**And** checksum validation script: `scripts/validate_fixture_checksums.py`

**Validation:** `uv run python scripts/validate_fixture_checksums.py`
**Gate:** Provenance protocol established, reusable for Story 4.8 (Palmer retroactive provenance)
**FR Coverage:** Enables FR-EDDI-001, establishes pattern for future validations

**CRITICAL CORRECTION:** Protocol covers BOTH EDDI (new) AND Palmer (retroactive in Story 4.8)

---

### Story 3.3: NOAA EDDI Reference Validation

**Agent:** Gamma | **Phase:** 2 | **Depends On:** [3.2] | **Blocks:** []

**Primary Files:** `tests/test_noaa_eddi_reference.py` (NEW)
**Secondary Files:** `tests/fixture/noaa-eddi-*/eddi_*_reference.nc` (downloaded)

As a scientific validation engineer, I want EDDI outputs validated against NOAA reference, So that users can trust library matches operational NOAA EDDI.

**Acceptance Criteria:**

**Given** provenance protocol established
**When** I download NOAA PSL EDDI CONUS subsets
**Then** reference datasets downloaded for 1-month, 3-month, 6-month scales
**And** provenance.json populated with actual download metadata

**And** test implementation:
```python
def test_eddi_noaa_reference_1month():
    # Load NOAA reference
    noaa_ds = xr.open_dataset("tests/fixture/noaa-eddi-1month/eddi_1month_reference.nc")

    # Compute EDDI using library
    computed = eddi(pet_input, scale=1, ...)

    # Validate tolerance
    np.testing.assert_allclose(
        computed,
        noaa_ds["eddi_1month"],
        rtol=1e-5,
        atol=1e-5
    )
```

**And** tests for all 3 scales (1-month, 3-month, 6-month)
**And** docstring explains 1e-5 tolerance (non-parametric ranking FP accumulation)

**Validation:** `uv run pytest tests/test_noaa_eddi_reference.py -v`
**Gate:** FR-TEST-004 satisfied, NOAA validation passes
**FR Coverage:** FR-EDDI-001

---

### Story 3.4: EDDI xarray Adapter

**Agent:** Gamma | **Phase:** 2 | **Depends On:** [1.2, 3.3] | **Blocks:** []

**Primary Files:** `src/climate_indices/indices.py` (eddi function)
**Secondary Files:** `xarray_adapter.py`, `cf_metadata_registry.py`, `typed_public_api.py`

As a gridded dataset user, I want EDDI to support xarray inputs, So that I can compute evaporative demand drought on spatial grids with CF metadata.

**Acceptance Criteria:**

**Given** EDDI numpy function validated
**When** I apply @xarray_adapter decorator
**Then** CF metadata entry in registry:
- `long_name="Evaporative Demand Drought Index"`
- `units=""` (dimensionless)
- `standard_name` (custom, not in CF table)
- `references` Hobbins et al. (2016) DOI: 10.1175/JHM-D-15-0121.1

**And** @overload signatures in typed_public_api.py
**And** equivalence test: numpy == xarray within 1e-8

**Validation:** `uv run pytest tests/test_eddi_adapter.py::test_eddi_xarray_equivalence -v`
**Gate:** Equivalence test passes, CF metadata compliant
**FR Coverage:** FR-EDDI-002

---

### Story 3.5: EDDI CLI Integration

**Agent:** Gamma | **Phase:** 2 | **Depends On:** [3.4] | **Blocks:** []

**Primary Files:** `src/climate_indices/__main__.py`
**Secondary Files:** `README.md`, `docs/cli.rst`

As a command-line user, I want --index eddi support, So that I can compute EDDI via process_climate_indices CLI (resolves Issue #414).

**Acceptance Criteria:**

**Given** EDDI xarray adapter complete
**When** I add CLI integration
**Then** flag added: `--index eddi`
**And** parameter added: `--pet_file <path>` for PET input netCDF

**And** help text documents:
- PET method recommendation (PM FAO56)
- Required PET input format
- Example command

**And** README updated with EDDI CLI example
**And** integration test validates end-to-end workflow

**Validation:** `uv run python -m climate_indices --index eddi --pet_file tests/fixture/pet_sample.nc --help`
**Gate:** CLI works, Issue #414 resolved
**FR Coverage:** FR-EDDI-003

---

### Story 3.6: EDDI PET Method Documentation

**Agent:** Gamma | **Phase:** 2 | **Depends On:** [2.5, 3.4] | **Blocks:** []

**Primary Files:** `src/climate_indices/indices.py` (eddi docstring)
**Secondary Files:** `docs/algorithms.rst`

As an EDDI user, I want PET method guidance in documentation, So that I know PM FAO56 is recommended (not Thornthwaite).

**Acceptance Criteria:**

**Given** PM-ET available (Story 2.5 complete)
**When** I update EDDI documentation
**Then** docstring Note section:
- "EDDI is most accurate when using Penman-Monteith FAO56 reference evapotranspiration (E0)."
- "Using simplified methods like Thornthwaite may produce inaccurate drought signals."

**And** See Also: Cross-reference `eto_penman_monteith()`
**And** References: Add Hobbins et al. 2016 citation
**And** `docs/algorithms.rst` EDDI section explains PET method sensitivity

**Validation:** Manual review of docstring and docs
**Gate:** Documentation complete and accurate
**FR Coverage:** FR-EDDI-004

---

### Story 3.7: Track 2 Integration Validation

**Agent:** Gamma | **Phase:** 4 | **Depends On:** [all Track 2 stories] | **Blocks:** []

**Primary Files:** `tests/test_integration.py`
**Secondary Files:** none

As an EDDI user, I want end-to-end integration tests, So that I can verify EDDI works correctly with PM-ET in realistic workflows.

**Acceptance Criteria:**

**Given** all Track 2 stories complete
**When** I run integration tests
**Then** scenarios validated:
- EDDI with PM-ET (recommended pathway)
- EDDI with Thornthwaite (fallback, documented as suboptimal)
- EDDI CLI end-to-end (file input → output)
- EDDI NOAA reference validation (1e-5 tolerance)
- scPDSI stub documented for future work

**And** documentation complete: EDDI user guide, PET method selection guide

**Validation:** `uv run pytest tests/test_integration.py::test_eddi_integration -v`
**Gate:** All integration scenarios pass
**FR Coverage:** All FR-EDDI-* and FR-PNP-* requirements satisfied

---

## Epic 4: Palmer Multi-Output xarray

**Goal:** Deliver Palmer indices with Dataset return and CF metadata per variable for advanced xarray capabilities

**User Value:** Water balance modelers can access all 4 Palmer outputs in single Dataset with type safety; gridded workflows avoid manual tuple unpacking

**FR Coverage:** FR-PALMER-001 through FR-PALMER-007 (7 FRs)

**NFR Coverage:** NFR-PALMER-SEQ, NFR-PALMER-PERF, NFR-MULTI-OUT

**Dependencies:** Epic 1 (Palmer structlog), Epic 2 (infrastructure validation)

**Blocks:** None

---

### Story 4.1: Palmer xarray Handoff Validation

**Agent:** Delta | **Phase:** 2 | **Depends On:** [1.6] | **Blocks:** [4.2, 4.3, 4.4, 4.5, 4.6, 4.7]

**Primary Files:** `src/climate_indices/palmer.py`
**Secondary Files:** none

As Agent Delta, I want to validate Palmer structlog handoff, So that I can begin xarray wrapper work without conflicts with Agent Alpha.

**Acceptance Criteria:**

**Given** Story 1.6 (Palmer structlog) complete by Agent Alpha
**When** I verify handoff conditions
**Then** palmer.py lines 1-913 use structlog (Alpha's work complete)
**And** no stdlib logging imports remain
**And** all tests pass: `uv run pytest tests/test_palmer.py -v`

**And** I can import palmer.py without errors
**And** I identify line 914 as starting point for xarray wrapper (NEW code)

**Validation:** `uv run pytest tests/test_palmer.py -v`
**Gate:** Clean handoff, ready to begin xarray wrapper
**FR Coverage:** Enables FR-PALMER-001 through FR-PALMER-007

---

### Story 4.2: Palmer xarray Manual Wrapper

**Agent:** Delta | **Phase:** 2 | **Depends On:** [4.1] | **Blocks:** [4.3, 4.4]

**Primary Files:** `src/climate_indices/palmer.py` (lines 914+ NEW)
**Secondary Files:** none

As a Palmer user, I want palmer_xarray() wrapper function, So that I can call Palmer with xarray inputs and get Dataset return (NOT tuple).

**Acceptance Criteria:**

**Given** numpy palmer() function returns 5-tuple
**When** I implement manual wrapper (Pattern C from research)
**Then** function added at line 914+ in palmer.py

**And** function signature with proper docstring follows Pattern C approach

**And** implementation uses apply_ufunc with stack/unpack pattern:
- Stack outputs: `np.stack([pdsi, phdi, pmdi, z], axis=0)`
- Unpack to Dataset variables
- Workaround documented: xarray Issue #1815

**And** structlog events: calculation_started, calculation_completed
**And** ~150 lines implementation

**Validation:** `uv run pytest tests/test_palmer_xarray.py::test_palmer_xarray_basic -v`
**Gate:** Function works, basic Dataset return validated
**FR Coverage:** FR-PALMER-001

---

### Story 4.3: Multi-Output Dataset Construction

**Agent:** Delta | **Phase:** 2 | **Depends On:** [4.2] | **Blocks:** [4.5, 4.6]

**Primary Files:** `src/climate_indices/palmer.py` (palmer_xarray function)
**Secondary Files:** none

As a Palmer user, I want Dataset with 4 independent variables, So that I can access PDSI, PHDI, PMDI, Z-Index naturally without tuple unpacking.

**Acceptance Criteria:**

**Given** palmer_xarray() wrapper exists
**When** I construct Dataset return
**Then** Dataset contains 4 variables:
- `pdsi`: DataArray with full coordinates
- `phdi`: DataArray with full coordinates
- `pmdi`: DataArray with full coordinates
- `z_index`: DataArray with full coordinates

**And** each variable has independent metadata (applied in Story 4.6)
**And** NetCDF write/read preserves structure:
```python
ds.to_netcdf("palmer.nc")
ds_loaded = xr.open_dataset("palmer.nc")
assert set(ds_loaded.data_vars) == {"pdsi", "phdi", "pmdi", "z_index"}
```

**Validation:** `uv run pytest tests/test_palmer_xarray.py::test_palmer_dataset_structure -v`
**Gate:** Dataset structure validated, NetCDF round-trip works
**FR Coverage:** FR-PALMER-002

---

### Story 4.4: AWC Spatial Parameter Validation

**Agent:** Delta | **Phase:** 2 | **Depends On:** [4.2] | **Blocks:** []

**Primary Files:** `src/climate_indices/palmer.py` (palmer_xarray function)
**Secondary Files:** `exceptions.py`

As a Palmer user, I want AWC validation to prevent time dimension errors, So that I don't accidentally pass time-varying soil properties (AWC is spatially varying only).

**Acceptance Criteria:**

**Given** AWC can be scalar or DataArray
**When** I validate AWC parameter
**Then** if AWC is DataArray:
- Check `time_dim NOT in awc.dims`
- If time dimension present, raise `InvalidArgumentError`:
  ```
  "AWC must not have time dimension 'time'. AWC is a soil property (spatially varying only)."
  ```

**And** apply_ufunc input_core_dims:
- `[["time"], ["time"], []]` for precip, pet, awc

**And** test scenarios:
- Scalar AWC (uniform soil)
- DataArray AWC (spatial variation, no time)
- DataArray AWC with time (should raise error)

**Validation:** `uv run pytest tests/test_palmer_xarray.py::test_palmer_awc_validation -v`
**Gate:** AWC validation enforced, error message actionable
**FR Coverage:** FR-PALMER-003

---

### Story 4.5: params_dict JSON Serialization

**Agent:** Delta | **Phase:** 2 | **Depends On:** [4.3] | **Blocks:** []

**Primary Files:** `src/climate_indices/palmer.py` (palmer_xarray function)
**Secondary Files:** none

As a Palmer user, I want calibration params accessible via Dataset attrs, So that I can access alpha, beta, gamma, delta coefficients after computation.

**Acceptance Criteria:**

**Given** numpy palmer() returns params_dict as 5th tuple element
**When** I store params in Dataset attrs
**Then** dual access pattern:
1. JSON string: `ds.attrs["palmer_params"] = json.dumps(params_dict)`
2. Individual attrs: `ds.attrs["palmer_alpha"]`, `ds.attrs["palmer_beta"]`, etc.

**And** params computed once from first grid cell (spatially constant)
**And** JSON round-trip validated:
```python
params_loaded = json.loads(ds.attrs["palmer_params"])
assert params_loaded["alpha"] == ds.attrs["palmer_alpha"]
```

**And** NetCDF write/read preserves params attrs

**Validation:** `uv run pytest tests/test_palmer_xarray.py::test_palmer_params_serialization -v`
**Gate:** params_dict accessible both ways, JSON serialization works
**FR Coverage:** FR-PALMER-004

---

### Story 4.6: Palmer CF Metadata Registry

**Agent:** Delta | **Phase:** 2 | **Depends On:** [1.2, 4.3] | **Blocks:** [4.7]

**Primary Files:** `src/climate_indices/cf_metadata_registry.py`
**Secondary Files:** `palmer.py` (applies metadata)

As a Palmer user, I want CF-compliant metadata for all 4 variables, So that outputs are self-documenting and NetCDF-interchange ready.

**Acceptance Criteria:**

**Given** cf_metadata_registry.py exists
**When** I add Palmer entries
**Then** CF_METADATA dict contains:

```python
"pdsi": {
    "long_name": "Palmer Drought Severity Index",
    "units": "",
    "references": "https://doi.org/10.1175/1520-0493(1965)093<0326:MFTIAS>2.3.CO;2",
    "valid_range": (-10.0, 10.0),
}
"phdi": {
    "long_name": "Palmer Hydrological Drought Index",
    "units": "",
    "references": "...",
    "valid_range": (-10.0, 10.0),
}
"pmdi": {
    "long_name": "Palmer Modified Drought Index",
    "units": "",
    "references": "Heddinghaus & Sabol (1991)",
    "valid_range": (-10.0, 10.0),
}
"z_index": {
    "long_name": "Palmer Z-Index",
    "units": "",
    "references": "Palmer (1965)",
    "valid_range": (-7.0, 7.0),
}
```

**And** palmer_xarray() applies metadata:
```python
from climate_indices.cf_metadata_registry import CF_METADATA

ds["pdsi"].attrs.update(CF_METADATA["pdsi"])
ds["phdi"].attrs.update(CF_METADATA["phdi"])
# ... etc
```

**Validation:** `uv run pytest tests/test_palmer_xarray.py::test_palmer_cf_metadata -v`
**Gate:** All 4 variables have complete CF metadata
**FR Coverage:** FR-PALMER-005

---

### Story 4.7: Palmer typed_public_api Overloads

**Agent:** Delta | **Phase:** 2 | **Depends On:** [4.6] | **Blocks:** []

**Primary Files:** `src/climate_indices/typed_public_api.py`
**Secondary Files:** none

As a type-safe application developer, I want @overload signatures for Palmer, So that mypy can distinguish tuple vs Dataset return types at compile time.

**Acceptance Criteria:**

**Given** palmer_xarray() returns Dataset
**When** I add @overload signatures to typed_public_api.py
**Then** numpy overload:
```python
@overload
def pdsi(
    precips: np.ndarray,
    pet: np.ndarray,
    awc: float,
    ...
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict | None]:
    ...
```

**And** xarray overload:
```python
@overload
def pdsi(
    precips: xr.DataArray,
    pet: xr.DataArray,
    awc: float | xr.DataArray,
    ...
) -> xr.Dataset:
    ...
```

**And** runtime dispatcher:
```python
def pdsi(precips, pet, awc, ...):
    if isinstance(precips, xr.DataArray):
        return palmer_xarray(precips, pet, awc, ...)
    else:
        return palmer(precips, pet, awc, ...)  # numpy path, returns tuple
```

**And** mypy --strict validates both paths

**Validation:** `uv run mypy --strict src/climate_indices/typed_public_api.py`
**Gate:** mypy passes, type safety complete
**FR Coverage:** FR-PALMER-006

---

### Story 4.8: Palmer Equivalence and Provenance

**Agent:** Delta | **Phase:** 2 | **Depends On:** [3.2, 4.7] | **Blocks:** []

**Primary Files:** `tests/test_palmer_equivalence.py` (NEW)
**Secondary Files:** `tests/fixture/palmer-344-division/provenance.json` (NEW)

As a Palmer validation engineer, I want NumPy vs xarray equivalence validated, So that I can prove wrapper implementation is numerically consistent (NOT validating against pdi.f).

**Acceptance Criteria:**

**Given** palmer_xarray() complete
**When** I create equivalence tests
**Then** validate numpy tuple output vs xarray Dataset values:
- PDSI: tolerance 1e-8
- PHDI: tolerance 1e-8
- PMDI: tolerance 1e-8
- Z-Index: tolerance 1e-8

**And** test scenarios:
- Scalar AWC
- DataArray AWC (spatial variation)
- Time series with NaN gaps
- Different Dask chunking strategies

**And** establish retroactive provenance for 344-division Palmer fixtures:
- Create `tests/fixture/palmer-344-division/provenance.json`
- Document: These fixtures validate Python Palmer implementation (palmer.py)
- **NOT validating against pdi.f** (pdi.f validation is separate effort)
- fixture_version: "1.0"
- validation_tolerance: 1e-8 (xarray wrapper vs numpy core)

**Validation:** `uv run pytest tests/test_palmer_equivalence.py -v`
**Gate:** All equivalence tests pass (1e-8), provenance documented
**FR Coverage:** FR-PALMER-007

**CRITICAL CORRECTION:** No claim of pdi.f validation. xarray equivalence (1e-8) validates wrapper vs Python numpy core, NOT vs Fortran original. Story 4.8 adds retroactive provenance to existing 344-division fixtures.

---

### Story 4.9: Track 3 Performance and Integration

**Agent:** Delta | **Phase:** 4 | **Depends On:** [2.6, all Track 3 stories] | **Blocks:** []

**Primary Files:** `tests/test_benchmark_palmer.py`, `tests/test_integration.py`
**Secondary Files:** `docs/palmer_xarray.md` (NEW)

As a Palmer user, I want performance validated and integration tests passing, So that I can use Palmer xarray in production with confidence.

**Acceptance Criteria:**

**Given** all Track 3 stories complete and baseline measured (Story 2.6)
**When** I run performance and integration tests
**Then** performance validation:
- xarray path ≥80% speed of multiprocessing baseline (NFR-PALMER-PERF)
- Benchmark report shows median wall-clock time
- Memory usage comparable (no excessive overhead)

**And** integration scenarios:
- Palmer with scalar AWC (uniform soil)
- Palmer with DataArray AWC (spatial variation)
- Palmer Dataset NetCDF write/read round-trip
- Palmer with Dask chunking (spatial only, time=-1)
- Palmer type safety (mypy validates dispatchers)

**And** documentation complete:
- `docs/palmer_xarray.md` usage guide
- Chunking guidance (spatial OK, temporal FORBIDDEN)
- params_dict access examples
- Migration guide from tuple unpacking

**Validation:** `uv run pytest tests/test_benchmark_palmer.py tests/test_integration.py::test_palmer_integration -v`
**Gate:** Performance ≥80%, all integration scenarios pass, docs complete
**FR Coverage:** NFR-PALMER-PERF, NFR-PALMER-SEQ validated

---

## Epic 5: Cross-Cutting Validation

**Goal:** Ensure v2.4.0 quality gates, compliance validation, and release readiness

**User Value:** Users receive high-quality release with validated patterns, comprehensive testing, and future roadmap visibility

**FR Coverage:** FR-SCPDSI-001, all NFRs

**NFR Coverage:** All 8 NFRs validated

**Dependencies:** All other epics

**Blocks:** None (release gate)

---

### Story 5.1: scPDSI Stub Interface

**Agent:** Gamma | **Phase:** 3 | **Depends On:** [Phase 2 complete] | **Blocks:** []

**Primary Files:** `src/climate_indices/indices.py`
**Secondary Files:** `typed_public_api.py`

As a project planner, I want scPDSI stub interface defined, So that future implementation has clear API contract and users know it's planned.

**Acceptance Criteria:**

**Given** self-calibrating PDSI is future enhancement
**When** I create stub function
**Then** function has proper signature with numpy and xarray type annotations
**And** function raises NotImplementedError with message: "scPDSI implementation planned for future release"

**And** docstring provides methodology overview:
- Note that scPDSI implementation is planned for future release
- Explain automatic calibration (Wells et al. 2004) vs fixed coefficients in PDSI
- Include Wells et al. (2004) reference: self-calibrating Palmer drought severity index, J. Climate

**And** @overload signatures added in typed_public_api.py (for future API contract)

**Validation:** `uv run python -c "from climate_indices import scpdsi; scpdsi([1,2,3], [1,2,3], 2.5)"` (raises NotImplementedError)
**Gate:** Stub documented, API contract defined
**FR Coverage:** FR-SCPDSI-001

---

### Story 5.2: Property-Based Test Expansion (Final Audit)

**Agent:** Alpha | **Phase:** 3 | **Depends On:** [1.10, 2.5, 3.4, 4.6] | **Blocks:** []

**Primary Files:** `tests/test_properties.py`
**Secondary Files:** none

As a quality engineer, I want final property-based test audit, So that I verify all new indices have property tests (PM-ET, EDDI, Palmer Dataset).

**Acceptance Criteria:**

**Given** Track 1, 2, 3 complete
**When** I audit property test coverage
**Then** verify property tests exist for:
- PM-ET: Boundedness (ETo >= 0), physically realistic ranges
- EDDI: Ranking order preservation, percentile bounds
- Palmer multi-output: Each variable (PDSI, PHDI, PMDI, Z-Index) has properties

**And** hypothesis finds no counterexamples across all indices
**And** property test guide (`docs/testing/property-based-test-guide.md`) complete

**Validation:** `uv run pytest tests/test_properties.py -v --hypothesis-show-statistics`
**Gate:** 7/7 indices have property tests, all pass
**FR Coverage:** Completes FR-PATTERN-012, validates all property-based test FRs

---

### Story 5.3: v2.4.0 Final Validation and Release Prep

**Agent:** Omega | **Phase:** 4 | **Depends On:** [all stories] | **Blocks:** []

**Primary Files:** All integration points
**Secondary Files:** `CHANGELOG.md`, `docs/release-notes-v2.4.0.md`

As a release manager, I want comprehensive final validation, So that v2.4.0 meets all 30 FRs and 8 NFRs before release.

**Acceptance Criteria:**

**Given** all 38 stories complete
**When** I run final validation suite
**Then** all quality gates pass:

**Code Quality:**
- ✓ Full test suite green (1400+ test lines)
- ✓ `mypy --strict` passes on all modules
- ✓ `ruff check` clean (no warnings)
- ✓ Coverage >85% (target: >90%)

**FR Validation (30/30):**
- ✓ All FR-PATTERN-* tests pass (12 FRs)
- ✓ All FR-PM-* tests pass (6 FRs)
- ✓ All FR-EDDI-* tests pass (4 FRs)
- ✓ All FR-PALMER-* tests pass (7 FRs)
- ✓ FR-PNP-001 validated
- ✓ FR-SCPDSI-001 stub documented

**NFR Validation (8/8):**
- ✓ NFR-PATTERN-EQUIV: All equivalence tests pass (1e-8)
- ✓ NFR-PATTERN-COVERAGE: 42/42 compliance checkpoints validated
- ✓ NFR-PATTERN-MAINT: Pattern guide complete
- ✓ NFR-PM-PERF: FAO56 examples within ±0.05 mm/day
- ✓ NFR-PALMER-SEQ: Sequential constraint documented
- ✓ NFR-PALMER-PERF: Performance ≥80% of baseline
- ✓ NFR-MULTI-OUT: Stack/unpack pattern documented
- ✓ NFR-EDDI-VAL: NOAA reference within 1e-5

**Documentation:**
- ✓ CHANGELOG.md updated with v2.4.0 entries
- ✓ Release notes complete (`docs/release-notes-v2.4.0.md`)
- ✓ Migration guide for exception handling transition
- ✓ User guide updates for PM-ET, EDDI, Palmer xarray

**Integration:**
- ✓ All shared files merged cleanly (typed_public_api.py, cf_metadata_registry.py, xarray_adapter.py)
- ✓ __init__.py exports updated
- ✓ No import conflicts, all public APIs accessible

**Validation:** `uv run pytest tests/ -v && mypy --strict src/ && ruff check src/`
**Gate:** All 30 FRs satisfied, all 8 NFRs validated, release ready
**FR Coverage:** All requirements complete

---

## Phase Gate Checklists

### Gate 0: Foundation Complete

**Trigger:** Stories 1.1, 1.2 complete
**Owner:** Alpha

**Checklist:**
- [ ] Structured exception hierarchy importable
- [ ] cf_metadata_registry.py module importable
- [ ] Full test suite green
- [ ] mypy --strict passes on new modules
- [ ] Exception docstrings complete with examples

**Exit Criteria:** ✓ All checkboxes complete → Phase 1 begins

---

### Gate 1: Parallel Core Complete

**Trigger:** Stories 1.3-1.10 (Alpha), 2.1-2.6 (Beta) complete
**Owner:** Omega

**Checklist:**

**Alpha Track (Patterns):**
- [ ] Palmer structlog migration complete (Story 1.6)
- [ ] PNP/PCI xarray adapters working
- [ ] typed_public_api.py has 4 new @overload sets
- [ ] Property-based tests expanded (PNP, PCI, Palmer, SPEI)
- [ ] All Track 0 tests pass

**Beta Track (PM-ET):**
- [ ] FAO56 Example 17 within ±0.05 mm/day
- [ ] FAO56 Example 18 within ±0.05 mm/day
- [ ] PM-ET xarray adapter working
- [ ] Palmer baseline measured and documented
- [ ] All Track 1 tests pass

**Omega Actions:**
- [ ] Merge typed_public_api.py append blocks (Alpha + Beta sections)
- [ ] Merge cf_metadata_registry.py entries
- [ ] Update __init__.py exports for PNP, PCI, PM-ET
- [ ] Resolve any merge conflicts
- [ ] Full test suite green after merge

**Exit Criteria:** ✓ All checkboxes complete → Phase 2 begins (Gamma ∥ Delta)

---

### Gate 2: Dependent Tracks Complete

**Trigger:** Stories 3.1-3.6 (Gamma), 4.1-4.8 (Delta) complete
**Owner:** Omega

**Checklist:**

**Gamma Track (EDDI):**
- [ ] PR #597 merged cleanly (Story 3.1)
- [ ] NOAA provenance protocol established (Story 3.2)
- [ ] EDDI validates against NOAA within 1e-5 (Story 3.3)
- [ ] EDDI xarray adapter working (Story 3.4)
- [ ] EDDI CLI integration complete (Story 3.5)
- [ ] PET method documentation complete (Story 3.6)
- [ ] All Track 2 tests pass

**Delta Track (Palmer xarray):**
- [ ] Palmer handoff validated (Story 4.1)
- [ ] palmer_xarray() wrapper complete (Story 4.2)
- [ ] Multi-output Dataset structure validated (Story 4.3)
- [ ] AWC spatial validation working (Story 4.4)
- [ ] params_dict JSON serialization working (Story 4.5)
- [ ] Palmer CF metadata applied (Story 4.6)
- [ ] typed_public_api @overload for Palmer complete (Story 4.7)
- [ ] Palmer equivalence tests pass (1e-8) (Story 4.8)
- [ ] Palmer performance ≥80% of baseline (Story 4.9)
- [ ] All Track 3 tests pass

**Omega Actions:**
- [ ] Merge typed_public_api.py append blocks (Gamma + Delta sections)
- [ ] Merge cf_metadata_registry.py Palmer entries
- [ ] Merge xarray_adapter.py EDDI adapter
- [ ] Update __init__.py exports for EDDI, Palmer xarray
- [ ] Validate no import conflicts
- [ ] Full test suite green after merge
- [ ] Check typed_public_api.py line count (trigger extraction at 300 lines if needed)

**Exit Criteria:** ✓ All checkboxes complete → Phase 3 begins

---

### Gate 3: Audit Complete

**Trigger:** Stories 1.11, 5.1, 5.2 complete
**Owner:** Alpha, Gamma

**Checklist:**
- [ ] Pattern compliance dashboard shows 42/42 checkpoints (Story 1.11)
- [ ] scPDSI stub interface documented (Story 5.1)
- [ ] Property-based test expansion audit complete (Story 5.2)
- [ ] All indices have property tests (7/7)
- [ ] Pattern guide documentation complete

**Exit Criteria:** ✓ All checkboxes complete → Phase 4 begins

---

### Gate 4: Final Release Gate

**Trigger:** Stories 1.12, 2.7, 3.7, 4.9, 5.3 complete
**Owner:** Omega

**Checklist:**

**Code Quality:**
- [ ] Full test suite green (all 1400+ test lines)
- [ ] mypy --strict passes on all modules
- [ ] ruff check clean (no warnings)
- [ ] Coverage >85% (target: >90%)

**Functional Requirements (30/30):**
- [ ] All 12 FR-PATTERN-* validated
- [ ] All 6 FR-PM-* validated
- [ ] All 4 FR-EDDI-* validated
- [ ] All 7 FR-PALMER-* validated
- [ ] FR-PNP-001 validated
- [ ] FR-SCPDSI-001 stub documented

**Non-Functional Requirements (8/8):**
- [ ] NFR-PATTERN-EQUIV: Equivalence tests pass (1e-8)
- [ ] NFR-PATTERN-COVERAGE: 42/42 compliance validated
- [ ] NFR-PATTERN-MAINT: Pattern guide complete
- [ ] NFR-PM-PERF: FAO56 examples within ±0.05 mm/day
- [ ] NFR-PALMER-SEQ: Sequential constraint documented
- [ ] NFR-PALMER-PERF: Performance ≥80% of baseline
- [ ] NFR-MULTI-OUT: Stack/unpack pattern documented
- [ ] NFR-EDDI-VAL: NOAA reference within 1e-5

**Documentation:**
- [ ] CHANGELOG.md updated with v2.4.0 entries
- [ ] Release notes complete (`docs/release-notes-v2.4.0.md`)
- [ ] Migration guide for exception handling
- [ ] User guides updated (PM-ET, EDDI, Palmer xarray)
- [ ] API documentation complete

**Integration:**
- [ ] All shared files merged cleanly
- [ ] __init__.py exports complete
- [ ] No import conflicts
- [ ] All public APIs accessible and tested

**Exit Criteria:** ✓ All checkboxes complete → v2.4.0 RELEASE READY

---

## Summary Statistics

**Epic Breakdown:**
- Epic 1 (Canonical Pattern Completion): 12 stories (Alpha-owned)
- Epic 2 (PM-ET Foundation): 7 stories (Beta-owned)
- Epic 3 (EDDI/PNP/scPDSI Coverage): 7 stories (Gamma-owned)
- Epic 4 (Palmer Multi-Output): 9 stories (Delta-owned)
- Epic 5 (Cross-Cutting Validation): 3 stories (Alpha, Gamma, Omega)
- **Total: 38 stories**

**Agent Workload:**
- Alpha (Track 0): 14 stories (12 Epic 1 + 1 Epic 5 + 1 validation)
- Beta (Track 1): 7 stories
- Gamma (Track 2): 8 stories (7 Epic 3 + 1 Epic 5)
- Delta (Track 3): 9 stories
- Omega (Integration): 3 merge gates (Phase 1, 2, 4) + 1 final validation

**Phase Distribution:**
- Phase 0: 2 stories (foundation)
- Phase 1: 16 stories (Alpha ∥ Beta)
- Phase 2: 14 stories (Gamma ∥ Delta)
- Phase 3: 3 stories (audit)
- Phase 4: 4 stories (final validation)

**Critical Path:**
- Story 1.1 → 1.2 → 1.6 (Palmer structlog) → 4.1 (handoff) → 4.2-4.9 (Palmer xarray)
- Total critical path: ~5-6 weeks (Phase 0 + Phase 1 Palmer + Phase 2 Palmer xarray)

**Parallelization Opportunities:**
- Phase 1: Alpha (PNP/PCI/properties) ∥ Beta (PM-ET) — 2-3 weeks savings
- Phase 2: Gamma (EDDI) ∥ Delta (Palmer xarray) — 3-4 weeks savings
- Total parallel execution saves ~5-7 weeks vs sequential

**FR Coverage Validation:**
- ✓ All 30 FRs mapped to stories
- ✓ All 8 NFRs have validation stories
- ✓ No unmapped requirements
- ✓ No forward dependencies

**Three Critical Corrections Embedded:**
1. ✓ PR #597: Story 3.1 is merge conflict resolution, NOT greenfield EDDI
2. ✓ Palmer validation: Story 4.8 validates xarray wrapper vs numpy core (1e-8), NOT vs pdi.f
3. ✓ NOAA provenance: Story 3.2 protocol covers BOTH EDDI (new) AND Palmer retroactive (Story 4.8)

---

**Document Version:** v2.4.0
**Last Updated:** 2026-02-16
**Status:** Complete — Ready for Implementation
"""

    return content


def main():
    """Generate epics.md file."""
    output_path = Path("_bmad-output/planning-artifacts/epics.md")

    print(f"Generating comprehensive epics.md...")
    content = generate_epics_md()

    print(f"Writing to {output_path}...")
    output_path.write_text(content)

    print(f"✓ Complete! Generated {len(content)} characters")
    print(f"  File: {output_path}")

    # Count stories
    story_count = content.count("### Story ")
    epic_count = content.count("## Epic ")

    print(f"\n  Statistics:")
    print(f"    Epics: {epic_count}")
    print(f"    Stories: {story_count}")
    print(f"    Lines: {len(content.splitlines())}")


if __name__ == "__main__":
    main()

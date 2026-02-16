---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
workflowType: 'architecture'
lastStep: 8
status: 'complete'
completedAt: '2026-02-16'
inputDocuments:
  - '_bmad-output/planning-artifacts/prd.md'
  - '_bmad-output/planning-artifacts/research/technical-penman-monteith.md'
  - '_bmad-output/planning-artifacts/research/technical-palmer-modernization.md'
  - '_bmad-output/planning-artifacts/research/technical-eddi-validation.md'
  - '_bmad-output/project-context.md'
  - 'docs/index.md'
workflowType: 'architecture'
project_name: 'climate_indices'
user_name: 'James'
date: '2026-02-16'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

---

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**

climate_indices v2.4.0 introduces **30 new functional requirements** across 4 parallel tracks, building on the 60 FRs established in PRD v1.1. The requirements are strategically organized to balance technical debt reduction (Track 0) with algorithm expansion (Tracks 1-3):

- **Track 0: Canonical Pattern Completion (12 FRs)** — Apply v2.3.0-established patterns to ALL remaining indices
  - FR-PATTERN-001 to FR-PATTERN-006: xarray adapters + typed_public_api entries for `percentage_of_normal`, `pci`, `eto_thornthwaite`, `eto_hargreaves`
  - FR-PATTERN-007 to FR-PATTERN-008: structlog migration (Palmer) and lifecycle completion (eto_thornthwaite)
  - FR-PATTERN-009: Structured exceptions replace generic ValueError across all legacy functions
  - FR-PATTERN-010 to FR-PATTERN-012: Property-based tests for PNP, PCI, expanded SPEI/Palmer coverage

- **Track 1: PM-ET Foundation (6 FRs)** — Physics-based evapotranspiration completes PET method suite
  - FR-PM-001: Core Penman-Monteith FAO56 calculation (Equation 6)
  - FR-PM-002 to FR-PM-003: Atmospheric parameters + vapor pressure helpers (Equations 7-13)
  - FR-PM-004: Humidity pathway dispatcher (auto-select dewpoint → RH extremes → RH mean)
  - FR-PM-005: FAO56 worked example validation (Bangkok tropical, Uccle temperate, tolerance ±0.05 mm/day)
  - FR-PM-006: xarray adapter with CF metadata + Dask compatibility

- **Track 2: Index Coverage Expansion (5 FRs)** — Complete drought index catalog with scientific validation
  - FR-EDDI-001: **BLOCKING** — NOAA reference dataset validation (FR-TEST-004, tolerance 1e-5)
  - FR-EDDI-002 to FR-EDDI-004: EDDI xarray adapter, CLI integration (Issue #414), PM-ET recommendation docs
  - FR-PNP-001: Percent of Normal Precipitation with xarray support
  - FR-SCPDSI-001: Self-calibrating PDSI stub interface for future implementation

- **Track 3: Advanced xarray Capabilities (7 FRs)** — Multi-output Palmer with Dataset return
  - FR-PALMER-001: Manual `palmer_xarray()` wrapper (Pattern C from research — not decorator-based)
  - FR-PALMER-002: Multi-output Dataset return (4 variables: pdsi, phdi, pmdi, z_index)
  - FR-PALMER-003: AWC spatial parameter handling (no time dimension, validation raises error)
  - FR-PALMER-004: params_dict JSON serialization (dual access: JSON string + individual attrs)
  - FR-PALMER-005: CF metadata registry for all 4 Palmer variables
  - FR-PALMER-006: typed_public_api @overload signatures (numpy→tuple, xarray→Dataset)
  - FR-PALMER-007: NumPy vs xarray equivalence tests (tolerance 1e-8)

**Architectural Significance:**
- Track 0 must complete **before** Track 3 Palmer work (structlog migration blocks xarray refactoring)
- Track 1 is **required by** Tracks 2 & 3 (PM-ET needed for EDDI accuracy + infrastructure validation)
- Tracks 2 & 3 can execute **in parallel** after Track 0 + Track 1 dependencies met

**Critical Path Analysis (Architecture Review):**
- **FR-PATTERN-007 (Palmer structlog)** is **CRITICAL PATH** blocker for Track 3
  - `palmer.py` is 912 lines — realistic estimate: 12-16 hours (not 3-4)
  - Recommendation: Start FR-PATTERN-007 immediately if Track 3 is on critical path
- **Decoupling opportunity**: PNP/PCI pattern completion (FR-PATTERN-001, FR-PATTERN-002) can run **in parallel** with Palmer xarray work
  - Only Palmer-specific work (FR-PATTERN-007) blocks Track 3

---

**Non-Functional Requirements:**

v2.4.0 adds **8 new/modified NFRs** that establish measurable quality gates:

- **Pattern Compliance & Refactoring Safety (Track 0)**:
  - NFR-PATTERN-EQUIV: Numerical equivalence during refactoring (tolerance 1e-8 for float64)
  - NFR-PATTERN-COVERAGE: 100% pattern compliance dashboard (6 patterns × 7 indices = 42 compliance points)
  - NFR-PATTERN-MAINT: Maintainability through consistency (target: 30% reduction in time-to-fix)

- **Performance Targets**:
  - NFR-PM-PERF: PM-ET maintains FAO56 scientific accuracy (±0.05 mm/day for examples, ±0.01 kPa for intermediates)
  - NFR-PALMER-SEQ: Sequential time constraint documented (chunk spatial dims, NOT temporal)
  - NFR-PALMER-PERF: Palmer xarray ≥80% speed of multiprocessing baseline

- **Reliability & Validation**:
  - NFR-MULTI-OUT: Multi-output adapter pattern stability (stack/unpack workaround for xarray Issue #1815)
  - NFR-EDDI-VAL: EDDI NOAA reference validation tolerance (1e-5 for non-parametric ranking FP accumulation)

**Critical NFRs Drive Architecture:**
- NFR-PATTERN-EQUIV ensures pattern refactoring doesn't introduce numerical drift
- NFR-PALMER-SEQ acknowledges fundamental constraint (water balance state machine cannot parallelize time)
- NFR-EDDI-VAL distinguishes non-parametric ranking tolerance (1e-5) from parametric fitting tolerance (1e-8)

**Performance Baseline Requirements (Test Architect Review):**
- **NFR-PALMER-PERF BLOCKER**: No benchmark baseline established
  - Required: Measure current multiprocessing CLI performance **before** implementing Palmer xarray
  - Recommendation: Add `tests/test_benchmark_palmer.py` in Track 0 or Track 1
  - Risk: Without baseline, cannot validate performance regression claims

---

### Scale & Complexity

**Project Scale Assessment:**

- **Primary domain**: Scientific Computing (Climate Science) — Developer Tool/Library for drought monitoring
- **Complexity level**: **Medium-High**
  - Physics-based algorithms (PM FAO56: 19 coupled equations with non-linear vapor pressure)
  - Multi-output xarray patterns (Palmer: 4 variables + params dict, no Python library precedent)
  - NOAA reference validation requirements (FR-TEST-004: tolerance 1e-5 with provenance documentation)
  - Sequential state tracking (Palmer water balance: month-to-month soil moisture dependencies)
  - Pattern migration at scale (6 patterns × 7 indices = 42 compliance points)

**Architectural Complexity Drivers:**

1. **Multi-Modal API Requirements**:
   - Legacy numpy API must remain stable (backward compatibility constraint)
   - Modern xarray API must support CF metadata + Dask chunking
   - CLI must support both paths transparently
   - Type safety via `@overload` distinguishes numpy vs xarray dispatch
   - **Architecture concern**: Dispatcher complexity in `typed_public_api.py` (210 lines, will grow) — consider extracting dispatcher pattern utility

2. **Scientific Validation Standards**:
   - FAO56 validation: Reproduce published examples within 0.05 mm/day
   - NOAA validation: Match operational EDDI within 1e-5 tolerance
   - Numerical precision: Float64 required for intermediate values (not float32)
   - Constant precision: Exact FAO56 coefficients (0.6108, 17.27, 237.3) — no approximations

3. **Advanced xarray Challenges**:
   - Issue #1815 limitation: `dask='parallelized'` incompatible with multi-output → requires stack/unpack workaround
   - Sequential state constraint: Palmer cannot chunk time dimension (design constraint, not bug)
   - Spatial parameters: AWC varies (lat, lon) but NOT (time) — novel validation pattern
   - CF compliance: Per-variable metadata for Dataset with 4 independent variables
   - **Architecture decision**: Pattern C (manual wrapper) is pragmatic — accepting Python loop overhead because sequential constraint is fundamental, not a limitation to engineer around
   - **Long-term consideration**: Monitor xarray Issue #1815 resolution — may require refactoring when native multi-output support arrives

4. **Pattern Consistency at Scale**:
   - 7 public indices × 6 patterns = 42 compliance checkpoints
   - Pattern refactoring must preserve numerical equivalence (1e-8 tolerance)
   - Logging patterns differ: new modules use `logging_config.get_logger()`, legacy uses `utils.get_logger()`
   - Exception migration: Replace ~50 instances of generic `ValueError` with structured hierarchy

**Estimated Architectural Components:**

- **Core modules affected**: 7 (indices.py, palmer.py, eto.py, compute.py, xarray_adapter.py, typed_public_api.py, exceptions.py)
- **New modules**: 1 (palmer_xarray.py — ~150 lines for multi-output manual wrapper)
- **Test expansion**: ~1,400 lines (revised from 1,000 with reality check)
  - 400 Palmer (revised from 350 — comprehensive multi-output scenarios)
  - 250 PM-ET (revised from 200 — equation validation + integration)
  - 200 EDDI (revised from 150 — NOAA reference + property-based)
  - 400 pattern compliance (revised from 300 — 42 checkpoints × 2 paths)
  - 150 benchmark infrastructure (new — Palmer baseline + regression tracking)
- **CF metadata entries**: 7 new (pdsi, phdi, pmdi, z_index, eddi, pnp, eto_penman_monteith)
- **Type signature sets**: 4 new @overload pairs (percentage_of_normal, pci, eto_thornthwaite, eto_hargreaves)

**Implementation Time Estimates (Developer Reality Check):**
- **Track 0**: 48-64 hours (revised from 36-48) — Palmer structlog 912 lines is 12-16 hours alone
- **Track 1**: 40-56 hours (revised from 30-40) — 19 equations + validation + integration
- **Track 2**: 32-44 hours (revised from 24-32) — FR-TEST-004 reference dataset + provenance
- **Track 3**: 48-64 hours (revised from 36-48) — 150-line manual wrapper with 300-400 line test suite
- **Total**: 168-228 hours (revised from 126-168) — **40% increase for reality buffer**

---

### Technical Constraints & Dependencies

**Project Context Rules (from project-context.md):**

1. **Python Environment**:
   - Python >=3.10,<3.14 (ruff/mypy target: py310)
   - CI matrix: [3.10, 3.11, 3.12, 3.13]
   - Type annotations: `str | None` syntax (not `Optional[str]`)
   - Pathlib.Path for file paths (never string paths)

2. **xarray Adapter Pattern** (Epic 2 foundation):
   - **NEVER modify `indices.py` computation functions directly**
   - Decorator handles input detection, conversion, CF metadata attachment
   - Existing pattern: `@xarray_adapter` for single-output functions
   - Track 3 innovation: Manual wrapper for multi-output Palmer (Pattern C from research)

3. **structlog Logging Migration**:
   - New modules: `from climate_indices.logging_config import get_logger`
   - Legacy modules: `from climate_indices import utils` → `utils.get_logger()`
   - **NEVER mix logging patterns** within same module
   - Calculation event pattern: `calculation_started` → `calculation_completed` with duration_ms

4. **Exception Hierarchy**:
   - All exceptions inherit `ClimateIndicesError`
   - Use keyword-only context attributes (e.g., `shape=`, `expected=`)
   - Error messages provide actionable guidance (not just "invalid value")

5. **CF Metadata Registry**:
   - **NEVER hard-code CF attributes inline**
   - Always use `CF_METADATA` dict from xarray_adapter.py
   - Registry entry structure: `long_name`, `units`, `standard_name` (if exists), `references` (DOI)

6. **Numerical Precision**:
   - **NEVER use `==` with computed floats** — always `np.isclose()` or `np.testing.assert_allclose()`
   - Equivalence tests: `atol=1e-8` for float64
   - Reference validation: `atol=1e-5` for EDDI (looser due to non-parametric ranking)
   - FAO56 validation: `atol=0.05` for end-to-end mm/day, `atol=0.01` for intermediate kPa

**Dependency Constraints:**

- **scipy>=1.15.3**: Distribution fitting (gamma, Pearson III, log-logistic)
- **xarray>=2025.6.1**: apply_ufunc multi-output support (Issue #1815 workaround required)
- **dask>=2025.7.0**: Chunked computation (with time=-1 constraint for Palmer)
- **structlog>=24.1.0**: Structured logging (JSON serialization for production)
- **cftime>=1.6.4**: CF-compliant time handling
- **numpy**: Implicit float64 precision (not float32)

**External Validation Datasets:**

- **NOAA PSL EDDI CONUS archive**: [downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/](https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/)
  - **FR-TEST-004 BLOCKING**: No test = no merge
  - Required provenance metadata: source, URL, download_date, subset_description
  - Tolerance: 1e-5 (non-parametric ranking FP accumulation)
  - **Risk**: Reference dataset provenance incomplete — what if NOAA archive changes?
  - **Recommendation**: Establish provenance protocol early in Track 2

- **FAO56 Worked Examples**: Embedded in test code (no external files)
  - Example 17 (Bangkok): 5.72 mm/day ±0.05
  - Example 18 (Uccle): 3.9 mm/day ±0.05

**Architectural Invariants (from docs/index.md):**

1. **Time Dimension Chunking**: Dask arrays MUST have time as single chunk (`time: -1`) for Palmer
2. **Calibration Period**: Default minimum 30 years; violations trigger `ShortCalibrationWarning`
3. **Distribution Fitting**: Requires minimum 10 non-zero values
4. **Backward Compatibility**: Legacy numpy API (`indices.py`) must remain stable
5. **Type Safety**: Strict mypy compliance enforced on `typed_public_api.py`

---

### Cross-Cutting Concerns Identified

**1. CF Metadata Compliance (NetCDF Interchange)**
- All xarray outputs must have `long_name`, `units`, `references` attributes
- Palmer Dataset: 4 independent variables each with full CF metadata
- `valid_range` for automated QA (e.g., PDSI: -10.0 to 10.0)
- NetCDF round-trip preservation (write + read must preserve structure)

**2. Type Safety Across Dual API**
- `typed_public_api.py` provides mypy-strict entry point
- `@overload` signatures distinguish numpy→numpy vs xarray→xarray dispatch
- Runtime dispatcher uses `isinstance(input, xr.DataArray)` detection
- Palmer special case: numpy→tuple vs xarray→Dataset return types
- **Architecture consideration**: Extract dispatcher logic pattern to avoid duplication as module grows

**3. Performance & Scalability**
- **Dask chunking strategy**: Spatial chunking OK, temporal chunking FORBIDDEN for Palmer
- **vectorize=True overhead**: Acceptable for Palmer (existing multiprocessing baseline already per-grid-cell)
- **Target**: Palmer xarray ≥80% speed of current CLI (NFR-PALMER-PERF)
- **Optimization**: Spatial parallelization only (time dimension sequential)
- **BLOCKER**: Benchmark baseline must be established BEFORE Track 3 implementation begins

**4. Testing Strategy (Multi-Layered Architecture)**

**Test Pyramid Structure:**
- **Unit tests**: Equation helpers validated independently (FAO56 intermediates, SVP calculations)
- **Integration tests**: NumPy vs xarray equivalence (tolerance 1e-8)
- **Property-based tests**: Mathematical invariants (hypothesis framework)
  - Boundedness, NaN propagation, shape preservation
  - Required for all indices (FR-PATTERN-010 to FR-PATTERN-012)
- **Reference validation**: NOAA datasets (tolerance 1e-5 for EDDI)
- **Performance benchmarks**: Regression tracking via `test_benchmark_*.py`

**Test Coverage Requirements (Test Architect Review):**
- **42 compliance points × 2 paths (numpy + xarray) = 84 test cases** minimum
- **Palmer multi-output scenarios** (currently underspecified in FR-PALMER-007):
  - Scalar AWC vs DataArray AWC
  - All-NaN input handling
  - Time series with gaps/missing data
  - Different Dask chunking strategies (validate time=-1 constraint)
  - params_dict JSON serialization round-trip
- **Property-based test strategies**: Budget 6-8 hours per index (not boilerplate work)

**Benchmark Infrastructure Requirements:**
- **Track 0 or Track 1 deliverable**: `tests/test_benchmark_palmer.py`
- **Baseline measurement**: Current multiprocessing CLI performance (wall-clock time for 360×180 grid, 240 months)
- **Regression tracking**: CI integration with performance threshold alerts
- **Without baseline**: Cannot validate NFR-PALMER-PERF claims

**Risk-Based Testing Priorities:**
1. **FR-TEST-004 (EDDI NOAA reference)** — HIGHEST RISK (blocks merge)
2. **NFR-PATTERN-EQUIV (numerical equivalence)** — HIGH RISK (42 checkpoints)
3. **Palmer multi-output edge cases** — MEDIUM-HIGH RISK (most complex algorithm)
4. **PM-ET equation validation** — MEDIUM RISK (19 equations, FAO56 examples)

**5. Documentation & Usability**
- **Migration guides**: NumPy → xarray patterns
- **PET method guidance**: PM FAO56 recommendation for EDDI
- **Chunking guidance**: Spatial vs temporal for Palmer
- **Error messages**: Actionable context (e.g., AWC time dimension validation)

**6. Research Integration**
- PM research: Validates FAO56 implementation approach (Hybrid API, phased roadmap)
- Palmer research: Recommends Pattern C (manual wrapper) over decorator extraction
- EDDI research: Identifies FR-TEST-004 blocker, validates Tukey plotting positions
- All research findings incorporated into FRs and NFRs

---

### Risk Assessment & Mitigation Strategies

**Critical Path Risks:**

1. **FR-PATTERN-007 (Palmer structlog) blocks Track 3**
   - **Impact**: 912-line module migration takes 12-16 hours
   - **Mitigation**: Start immediately if Track 3 is on critical path
   - **Decoupling**: PNP/PCI pattern work can proceed in parallel

2. **FR-TEST-004 (NOAA reference validation) blocks Track 2 merge**
   - **Impact**: No test = no merge, no Phase 2 integration
   - **Mitigation**: Prioritize reference dataset acquisition and provenance documentation
   - **Contingency**: Track 2 cannot proceed without this

3. **NFR-PALMER-PERF baseline missing blocks performance claims**
   - **Impact**: Cannot validate ≥80% speed claim without measurement
   - **Mitigation**: Add benchmark infrastructure in Track 0 or Track 1
   - **Risk**: Building Track 3 without baseline = blind performance optimization

**Technical Debt Risks:**

1. **xarray Issue #1815 resolution requires future refactoring**
   - **Impact**: Pattern C (manual wrapper) is workaround, not permanent solution
   - **Mitigation**: Monitor xarray releases, document refactoring trigger
   - **Timeline**: Unknown when xarray will support multi-output + dask='parallelized'

2. **Dispatcher complexity growth in typed_public_api.py**
   - **Impact**: 210 lines now, will grow with 4+ new index dispatchers
   - **Mitigation**: Extract dispatcher pattern utility before duplication becomes maintenance burden
   - **Threshold**: If module exceeds 400 lines, refactor immediately

**Testing Architecture Risks:**

1. **Underestimated test complexity (42 compliance points × 2 paths)**
   - **Impact**: Test suite growth from 1,000 to 1,400+ lines
   - **Mitigation**: Budget 40% more time for test development
   - **Property-based tests**: Not boilerplate — require thoughtful hypothesis strategies

2. **Palmer multi-output scenarios underspecified**
   - **Impact**: Edge cases missed → production bugs in most complex algorithm
   - **Mitigation**: Create test architecture document before Track 3 implementation
   - **Coverage**: Scalar/DataArray AWC, NaN handling, chunking validation, serialization

**Estimation Risks:**

1. **Optimistic time estimates (reality check +40%)**
   - **Impact**: Schedule slippage, dependency chain delays
   - **Mitigation**: Revised estimates include buffer for code review, documentation, iteration
   - **Track 0**: 48-64 hours (not 36-48)
   - **Track 1**: 40-56 hours (not 30-40)
   - **Track 3**: 48-64 hours (not 36-48)

**Dependency Risks:**

1. **Track coupling creates sequential bottleneck**
   - **Impact**: Track 0 → Track 3 dependency means any Track 0 slip delays Track 3
   - **Mitigation**: Decouple where possible (PNP/PCI can run parallel with Palmer xarray)
   - **Critical path**: FR-PATTERN-007 only (not entire Track 0)

## Starter Template Evaluation

### Primary Technology Domain

**Python Scientific Computing Library** — Specialized for climate/drought monitoring with dual API support (NumPy + xarray)

### Existing Architectural Foundation

**Note**: This is an existing production library (v2.2.0 → v2.4.0 upgrade). The "starter template" was established in earlier versions and remains appropriate for v2.4.0 requirements.

**Selected Pattern: Modern Python Scientific Library (src-layout)**

**Build & Package Management:**
- **Build backend**: Hatchling (PEP 517/621 compliant)
- **Package manager**: uv (deterministic locking, faster than pip)
- **Lock file**: `uv.lock` (625KB) ensures reproducible builds across environments

**Quality Tooling Stack:**

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **ruff** | Linting + Formatting | 120 char line, E/W/F/I/B/C4/UP rules |
| **mypy** | Type checking | Near-strict, py310 target |
| **pytest** | Testing | Coverage + hypothesis + benchmarking |
| **structlog** | Logging | JSON serialization for production |

**Scientific Computing Dependencies:**
```toml
scipy>=1.15.3       # Distribution fitting (gamma, Pearson III)
xarray>=2025.6.1    # Multi-dimensional arrays + Dask
dask>=2025.7.0      # Parallel computation
numpy>=1.24.0       # Numerical foundation
cftime>=1.6.4       # CF-compliant time handling
```

**Project Structure Decisions:**

**1. src-layout Pattern:**
```
src/climate_indices/
├── indices.py              # Core NumPy computation functions (stable)
├── xarray_adapter.py       # Decorator-based xarray wrapping
├── typed_public_api.py     # Type-safe dispatchers (210 lines → 350-400 in v2.4.0)
├── palmer.py               # Palmer index (912 lines)
├── eto.py                  # Evapotranspiration functions
├── compute.py              # Shared utilities
├── exceptions.py           # Structured exception hierarchy
└── logging_config.py       # structlog configuration
```

**Rationale**: Import isolation prevents "tests pass locally but package broken" failures (src-layout), clean packaging for PyPI distribution, standard Python project structure

**2. Test Organization:**
```
tests/
├── conftest.py                    # Shared fixtures (module-scoped for expensive .npy)
├── fixture/                       # Reference datasets (.npy files)
│   └── noaa-eddi-*/               # Track 2: NOAA provenance structure (planned)
├── helpers/                       # Test utilities
├── test_*_equivalence.py          # NumPy vs xarray validation (1e-8 tolerance)
├── test_property_based.py         # Hypothesis property tests (6-8 hrs/index)
└── test_benchmark_*.py            # Performance regression tracking
    └── test_benchmark_palmer.py   # Track 0/1 BLOCKER: baseline measurement
```

**Rationale**: Scales to ~1,400 line test expansion (v2.4.0 requirement), supports reference dataset provenance, benchmark infrastructure for performance claims

**3. Type Safety & API Design:**
- Python >=3.10,<3.14 (modern syntax: `str | None`)
- `typed_public_api.py`: mypy strict mode, `@overload` signatures for numpy→numpy vs xarray→xarray dispatch
- `TYPE_CHECKING` imports to avoid circular dependencies

**Rationale**: Dual API support with compile-time type safety, runtime dispatcher uses `isinstance()` detection

**4. CI/CD Matrix:**
- GitHub Actions: Python [3.10, 3.11, 3.12, 3.13]
- Quality gates: ruff check + mypy --strict + pytest coverage

---

### Evaluation Against v2.4.0 Requirements

**✅ Current Structure Supports:**
- **Track 0 (Pattern Compliance)**: Modular design allows parallel migrations
- **Track 1 (PM-ET)**: `eto.py` module ready for Penman-Monteith additions
- **Track 2 (EDDI + PNP)**: `tests/fixture/` suitable for NOAA reference datasets
- **Track 3 (Palmer xarray)**: Manual wrapper pattern fits naturally into existing modules

**⚠️ Action Items Required (from Architectural Review):**

**BLOCKER - Track 0/1 Deliverable:**
1. **Palmer baseline measurement** (`tests/test_benchmark_palmer.py`)
   - **Why BLOCKER**: NFR-PALMER-PERF claims ≥80% speed maintenance, but no baseline exists
   - **Risk**: Track 3 implementation without baseline = blind performance optimization
   - **Action**: Measure current multiprocessing CLI performance (wall-clock time for 360×180 grid, 240 months) **before** Track 3 begins
   - **Owner**: Development team (Track 0 or Track 1 scope)

**HIGH PRIORITY - Pre-Track 2:**
2. **NOAA reference dataset provenance protocol**
   - **Why HIGH PRIORITY**: FR-TEST-004 blocks merge (no test = no merge)
   - **Risk**: If NOAA archive changes, cannot reproduce validation
   - **Action**: Define provenance metadata structure (source URL, download_date, subset_description, checksum)
   - **Location**: `tests/fixture/noaa-eddi-*/provenance.json`
   - **Owner**: Test architecture lead (before Track 2 implementation)

**MEDIUM PRIORITY - During Implementation:**
3. **Dispatcher pattern extraction threshold**
   - **Current state**: `typed_public_api.py` = 210 lines
   - **v2.4.0 projection**: 350-400 lines (4+ new index dispatchers)
   - **Threshold**: Extract dispatcher pattern utility at **300 lines** (not 400)
   - **Risk**: Architectural smell of "pattern wanting extraction"
   - **Action**: Monitor during implementation, extract to `src/climate_indices/dispatchers.py` if threshold exceeded
   - **Pattern**: Decorator factory for numpy/xarray routing

**ESTIMATION REFINEMENT:**
4. **Property-based test complexity**
   - **Current estimate**: Included in 1,400-line test expansion
   - **Reality check**: Property-based tests require **6-8 hours per index** (not boilerplate work)
   - **Reason**: Hypothesis strategies require domain expertise (boundedness, NaN propagation, shape preservation)
   - **Affected FRs**: FR-PATTERN-010 to FR-PATTERN-012
   - **Action**: Budget appropriately in Track 0 planning

---

### Architecture Decision Summary

**No Structural Refactoring Required** — Current foundation (src-layout + hatchling + uv + ruff + mypy + pytest) scales to v2.4.0 requirements without architectural changes.

**Key Architectural Strengths:**
- Boring technology that ships (hatchling + uv)
- Test pyramid infrastructure ready (unit + integration + property-based + reference + benchmark)
- Modular design supports parallel track execution
- Type safety established via `typed_public_api.py` pattern

**Critical Path Dependencies Identified:**
1. Palmer baseline measurement must complete before Track 3 (performance claims require data)
2. NOAA provenance protocol must exist before Track 2 implementation begins (quality gate)
3. Dispatcher pattern monitoring during implementation (extract at 300-line threshold)

**Architectural Philosophy Applied:**
> "Boring technology that ships beats exciting technology that doesn't. Build on solid foundations, measure before optimizing, and extract patterns when they reveal themselves — not before." — Winston, System Architect

---

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
1. **NOAA Provenance Protocol** — Blocks Track 2 EDDI implementation (FR-TEST-004 requirement)
2. **Palmer Module Organization** — Blocks Track 3 xarray wrapper implementation
3. **CF Metadata Registry Location** — Blocks all xarray adapter implementations in v2.4.0

**Important Decisions (Shape Architecture):**
4. **Property-Based Test Strategy** — Shapes test development effort and quality standards
5. **Exception Migration Strategy** — Shapes refactoring workflow and backward compatibility approach

**Deferred Decisions (Post-MVP):**
- None — all decisions required for v2.4.0 implementation have been made

---

### Data Architecture

#### Decision 1: NOAA Provenance Protocol

**Decision:** Establish JSON-based provenance metadata for external reference datasets

**Format:** `tests/fixture/noaa-eddi-*/provenance.json`

**Required Fields:**
```json
{
  "source": "NOAA PSL EDDI CONUS Archive",
  "url": "https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/...",
  "download_date": "2026-02-20",
  "subset_description": "3-month EDDI for CONUS, 2019-01 to 2020-12, 1° resolution",
  "checksum_sha256": "a3f5...",
  "variable_name": "eddi_3month",
  "validation_tolerance": 1e-5,
  "notes": "Non-parametric ranking FP accumulation requires looser tolerance than parametric fitting"
}
```

**Rationale:**
- **FR-TEST-004 BLOCKING**: NOAA reference validation required for EDDI merge
- **Reproducibility**: If NOAA archive changes, we can trace validation history
- **Scientific Rigor**: Provenance documentation is standard practice in climate science
- **Tolerance Documentation**: Different algorithms require different validation tolerances (1e-5 for EDDI vs 1e-8 for parametric)

**Party Mode Enhancement (Murat - Test Architect):**
- Add `fixture_version` field for explicit fixture evolution tracking
- **Rationale**: If reference data subset needs updating (e.g., extended time range), version tracking prevents confusion about which tests use which fixture version
- **Example**: `"fixture_version": "1.0"` → `"fixture_version": "1.1"` when time range extended

**Implementation Impact:**
- **Track 2 (EDDI)**: Must establish provenance protocol BEFORE downloading NOAA reference data
- **Test Architecture**: Creates reusable pattern for future external reference validations
- **CI/CD**: Checksum validation ensures fixture integrity across environments

**Location in Project Structure:**
```
tests/
└── fixture/
    ├── noaa-eddi-1month/
    │   ├── provenance.json
    │   └── eddi_1month_reference.nc
    ├── noaa-eddi-3month/
    │   ├── provenance.json
    │   └── eddi_3month_reference.nc
    └── noaa-eddi-6month/
        ├── provenance.json
        └── eddi_6month_reference.nc
```

---

### Module Organization

#### Decision 2: Palmer Multi-Output Module Organization

**Decision:** Keep Palmer xarray wrapper in `palmer.py` with clear section extraction via comments (NOT separate `palmer_xarray.py`)

**Module Structure:**
```python
# src/climate_indices/palmer.py (~1,060 lines projected)

# ============================================================================
# SECTION 1: NUMPY CORE COMPUTATION ENGINE (~912 lines existing)
# ============================================================================
# NumPy-based Palmer water balance calculations
# Multi-output return: tuple[ndarray, ndarray, ndarray, ndarray]

def palmer(...)  # ~900 lines existing


# ============================================================================
# SECTION 2: XARRAY ADAPTER LAYER (~150 lines new)
# ============================================================================
# xarray wrapper for Palmer multi-output Dataset return
# Pattern C: Manual wrapper (not decorator-based)

def palmer_xarray(...)  # ~150 lines new
```

**Rationale:**
- **Single import path**: `from climate_indices.palmer import palmer, palmer_xarray` (consistent with project conventions)
- **Co-location benefits**: xarray wrapper needs intimate knowledge of NumPy core return semantics
- **Threshold management**: ~1,060 lines is acceptable for bimodal module (computation + adapter)
- **Pattern C from research**: Manual wrapper is pragmatic given multi-output + Dask parallelization constraints

**Party Mode Notes:**

**Winston (Architect):**
> "If module approaches 1,500 lines during Track 3 implementation, revisit extraction. Current threshold (1,060) is acceptable for bimodal API module, but 1,500+ would indicate architectural smell."

**Action Trigger**: If palmer.py exceeds 1,400 lines → consider extracting to:
```
src/climate_indices/palmer/
├── __init__.py          # Public API exports
├── _core.py             # NumPy computation engine
└── _xarray_adapter.py   # xarray wrapper layer
```

**Implementation Impact:**
- **Track 0 (Palmer structlog)**: Migrate 912-line core function BEFORE Track 3 xarray work
- **Track 3 (Palmer xarray)**: Add ~150-line manual wrapper in same file
- **Testing**: Equivalence tests validate NumPy core vs xarray adapter consistency (tolerance 1e-8)

---

### Testing Architecture

#### Decision 3: Property-Based Test Strategy

**Decision:** Comprehensive property-based testing with hypothesis framework (Option A)

**Coverage Scope:**
- **Common properties** (all indices): Boundedness, NaN propagation, shape preservation, monotonicity
- **Index-specific properties** (3-5 per index): Domain-specific mathematical invariants

**Example Property Sets:**

**SPI (Standardized Precipitation Index):**
- Common: NaN propagation, shape preservation, input validation
- Specific: Mean ≈ 0, Standard deviation ≈ 1, Gamma distribution fitting bounds, Scale independence

**Palmer (PDSI, PHDI, PMDI, Z-Index):**
- Common: NaN propagation, shape preservation, time series continuity
- Specific: Water balance conservation, AWC parameter influence, Sequential state tracking, Z-index bounds

**EDDI (Evaporative Demand Drought Index):**
- Common: NaN propagation, shape preservation, monotonicity
- Specific: Ranking order preservation, Percentile bounds [0, 100], Non-parametric invariance

**Custom Hypothesis Generators Required:**
```python
# Climate-realistic data generators
@given(
    precip=climate_precipitation(min_value=0.0, max_value=500.0),  # mm/month
    temp=climate_temperature(min_value=-40.0, max_value=50.0),      # °C
    awc=soil_water_capacity(min_value=25.0, max_value=300.0)        # mm
)
def test_palmer_water_balance_conservation(precip, temp, awc):
    # Property: Total water input - output ≈ storage change
    ...
```

**Party Mode Enhancements:**

**Murat (Test Architect):**
> "Budget 50-60 hours (not 40-50) per index. Include property discovery workshops + failure investigation time. Property-based tests aren't boilerplate — they require deep domain understanding."

**Effort Breakdown (per index):**
- Property discovery workshop: 8-12 hours (collaborative with domain expert)
- Hypothesis strategy implementation: 12-16 hours
- Failure investigation and refinement: 15-20 hours
- Documentation and examples: 8-12 hours
- **Total: 43-60 hours per index**

**Amelia (Developer):**
> "Track actual time on first index (SPI), then adjust estimates. Property-based testing is new to this codebase — learning curve will frontload Track 0."

**Action**: Create `docs/testing/property-based-test-guide.md` after SPI property tests complete

**Implementation Impact:**
- **Track 0 (Pattern Compliance)**: FR-PATTERN-010 to FR-PATTERN-012 → 3 indices × 50-60 hours = 150-180 hours
- **Test Coverage**: NFR-PATTERN-COVERAGE → 42 compliance points include property-based test validation
- **Quality Gates**: Property test failures block merge (same as unit test failures)

**Affected FRs:**
- FR-PATTERN-010: PNP property-based tests
- FR-PATTERN-011: PCI property-based tests
- FR-PATTERN-012: Expanded SPEI/Palmer property-based coverage

---

### Refactoring Strategy

#### Decision 4: Exception Migration Strategy

**Decision:** Per-module incremental exception migration (NOT big-bang refactoring)

**Strategy:** Migrate ValueError → structured exceptions when touching each module for other patterns (Track 0 work)

**Exception Hierarchy (from existing architecture):**
```python
ClimateIndicesError (base)
├── InvalidArgumentError
│   ├── InvalidDistributionError
│   ├── InvalidScaleError
│   └── InvalidPeriodError
├── InsufficientDataError
│   ├── ShortCalibrationPeriodError
│   └── InsufficientNonZeroValuesError
└── ComputationError
    ├── DistributionFittingError
    └── ConvergenceError
```

**Migration Workflow:**

**When refactoring module for ANY v2.4.0 pattern (structlog, xarray, etc.):**
1. Audit all `raise ValueError(...)` instances in that module
2. Replace with appropriate structured exception
3. Add keyword-only context attributes
4. Update docstrings with new exception types
5. Add test cases for exception scenarios

**Example Migration:**
```python
# BEFORE (legacy)
if distribution not in ['gamma', 'pearson']:
    raise ValueError('Invalid distribution')

# AFTER (structured)
if distribution not in ['gamma', 'pearson']:
    raise InvalidDistributionError(
        f"Distribution '{distribution}' not supported",
        distribution=distribution,
        supported=['gamma', 'pearson']
    )
```

**Party Mode Notes:**

**Amelia (Developer):**
> "Document user-facing exception transition in v2.4.0 release notes. Temporary inconsistency is acceptable — users may see ValueError in some modules, structured exceptions in refactored modules."

**Release Notes Template:**
```markdown
## Exception Handling Improvements (v2.4.0)

v2.4.0 begins migrating to structured exceptions for better error diagnostics:

- **Refactored modules** (palmer, percentage_of_normal, pci): Now raise structured exceptions with actionable context
- **Legacy modules**: Still raise ValueError (will migrate in future versions)

**Example:**
```python
try:
    indices.spi(...)
except InvalidScaleError as e:
    print(e.scale, e.supported_scales)  # Actionable context
```

**Backward Compatibility:**
All new exceptions inherit `ClimateIndicesError`, which inherits `Exception`. Existing `except ValueError` blocks will still catch legacy modules.
```

**Implementation Impact:**
- **Track 0 (Palmer structlog)**: Migrate ~15-20 ValueError instances in palmer.py during structlog refactoring
- **Track 0 (PNP/PCI xarray)**: Migrate ~5-8 ValueError instances per module
- **Track 1 (PM-ET)**: NEW module → use structured exceptions from start
- **Track 2 (EDDI)**: Existing module → migrate during xarray adapter addition

**Effort Estimate:**
- Palmer exception migration: +4-6 hours on top of structlog work (included in revised 12-16 hour estimate)
- PNP/PCI exception migration: +2-3 hours each

---

### CF Metadata Management

#### Decision 5: CF Metadata Registry

**Decision:** Create separate `src/climate_indices/cf_metadata_registry.py` module for CF metadata definitions

**Module Structure:**
```python
# src/climate_indices/cf_metadata_registry.py

"""
CF metadata registry for climate indices.

Provides canonical CF-compliant metadata for all xarray output variables.
Following Climate and Forecast (CF) Metadata Conventions v1.10.
"""

from typing import TypedDict

class CFMetadata(TypedDict):
    """CF metadata structure for climate variables."""
    long_name: str
    units: str
    standard_name: str | None  # CF standard name if exists
    references: str             # DOI or URL to scientific reference
    valid_range: tuple[float, float] | None

CF_METADATA: dict[str, CFMetadata] = {
    # Existing indices (from v2.3.0)
    'spi': {
        'long_name': 'Standardized Precipitation Index',
        'units': '1',
        'standard_name': None,  # No CF standard name for SPI
        'references': 'https://doi.org/10.1175/1520-0442(1993)006<0745:SSOTSD>2.0.CO;2',
        'valid_range': (-3.5, 3.5),
    },

    # Track 3: Palmer multi-output variables (NEW)
    'pdsi': {
        'long_name': 'Palmer Drought Severity Index',
        'units': '1',
        'standard_name': None,
        'references': 'https://doi.org/10.1175/1520-0493(1965)093<0326:MFTIAS>2.3.CO;2',
        'valid_range': (-10.0, 10.0),
    },
    'phdi': {
        'long_name': 'Palmer Hydrological Drought Index',
        'units': '1',
        'standard_name': None,
        'references': 'https://doi.org/10.1175/1520-0493(1965)093<0326:MFTIAS>2.3.CO;2',
        'valid_range': (-10.0, 10.0),
    },
    'pmdi': {
        'long_name': 'Palmer Modified Drought Index',
        'units': '1',
        'standard_name': None,
        'references': 'https://doi.org/10.1175/1520-0493(1965)093<0326:MFTIAS>2.3.CO;2',
        'valid_range': (-10.0, 10.0),
    },
    'z_index': {
        'long_name': 'Palmer Z-Index',
        'units': '1',
        'standard_name': None,
        'references': 'https://doi.org/10.1175/1520-0493(1965)093<0326:MFTIAS>2.3.CO;2',
        'valid_range': (-7.0, 7.0),
    },

    # Track 2: New indices
    'eddi': {
        'long_name': 'Evaporative Demand Drought Index',
        'units': '1',
        'standard_name': None,
        'references': 'https://doi.org/10.1175/JHM-D-15-0121.1',
        'valid_range': (-3.5, 3.5),
    },
    'pnp': {
        'long_name': 'Percent of Normal Precipitation',
        'units': 'percent',
        'standard_name': None,
        'references': 'https://www.drought.gov/data-maps-tools/percent-of-normal-precipitation',
        'valid_range': (0.0, 500.0),  # Extreme wet conditions can exceed 200%
    },

    # Track 1: PM-ET
    'eto_penman_monteith': {
        'long_name': 'Reference Evapotranspiration (Penman-Monteith FAO56)',
        'units': 'mm day-1',
        'standard_name': 'water_evapotranspiration_flux',  # CF standard name exists!
        'references': 'https://www.fao.org/3/x0490e/x0490e00.htm',
        'valid_range': (0.0, 20.0),  # Extreme desert conditions ~15-18 mm/day
    },
}
```

**Import Pattern:**
```python
# In xarray_adapter.py or palmer_xarray wrapper
from climate_indices.cf_metadata_registry import CF_METADATA

# Apply metadata to xarray output
output_da.attrs.update(CF_METADATA['pdsi'])
```

**Party Mode Notes:**

**Winston (Architect):**
> "Monitor coupling — registry should define canonical metadata structure, not just store it. If multiple modules start duplicating metadata manipulation logic, consider adding `apply_cf_metadata(da, variable_name)` helper."

**Coupling Warning Triggers:**
- If 3+ modules duplicate `output_da.attrs.update(CF_METADATA[...])` → Extract helper
- If metadata validation logic appears in multiple places → Centralize validation

**Potential Helper Function (deferred until needed):**
```python
def apply_cf_metadata(
    data_array: xr.DataArray,
    variable_name: str,
    override_attrs: dict | None = None
) -> xr.DataArray:
    """Apply CF metadata from registry with optional overrides."""
    metadata = CF_METADATA[variable_name].copy()
    if override_attrs:
        metadata.update(override_attrs)
    data_array.attrs.update(metadata)
    return data_array
```

**Implementation Impact:**
- **Track 0**: No impact (pattern completion doesn't add new CF variables)
- **Track 1 (PM-ET)**: Add `eto_penman_monteith` metadata entry
- **Track 2 (EDDI, PNP)**: Add `eddi`, `pnp` metadata entries
- **Track 3 (Palmer)**: Add `pdsi`, `phdi`, `pmdi`, `z_index` metadata entries (4 variables)

**Location in Project Structure:**
```
src/climate_indices/
├── __init__.py
├── indices.py
├── palmer.py
├── xarray_adapter.py
├── typed_public_api.py
├── cf_metadata_registry.py  # NEW in v2.4.0
├── exceptions.py
└── logging_config.py
```

---

### Decision Impact Analysis

**Implementation Sequence:**

1. **Track 0 Start** (Parallel with Track 1):
   - Decision 4 (Exception Migration) applied during all pattern work
   - Decision 5 (CF Metadata Registry) created as stub, populated as needed

2. **Track 1 Completion** (Foundation):
   - Decision 5: Add `eto_penman_monteith` to CF_METADATA
   - Required by Tracks 2 & 3

3. **Track 2 Execution** (EDDI, PNP, scPDSI):
   - Decision 1 (NOAA Provenance) established BEFORE EDDI implementation
   - Decision 5: Add `eddi`, `pnp` to CF_METADATA
   - Decision 4: Exception migration during xarray adapter work

4. **Track 3 Execution** (Palmer xarray):
   - Decision 2 (Palmer Module Organization) governs xarray wrapper structure
   - Decision 3 (Property-Based Tests) creates comprehensive Palmer test suite
   - Decision 5: Add 4 Palmer variables to CF_METADATA
   - Decision 4: Exception migration completed during structlog work (Track 0 dependency)

**Cross-Component Dependencies:**

| Decision | Affects Tracks | Blocking? | Rationale |
|----------|---------------|-----------|-----------|
| Decision 1 (NOAA Provenance) | Track 2 (EDDI) | Yes | FR-TEST-004 blocks merge |
| Decision 2 (Palmer Module Org) | Track 3 (Palmer xarray) | Yes | Defines wrapper location |
| Decision 3 (Property-Based Tests) | Track 0 (Pattern) | No | Can parallelize with implementation |
| Decision 4 (Exception Migration) | All tracks | No | Incremental per-module work |
| Decision 5 (CF Metadata Registry) | Tracks 1, 2, 3 | No | Stub created, populated incrementally |

**Architectural Coupling Points:**

1. **Decision 1 ↔ Decision 3**: NOAA provenance JSON structure informs property-based test fixture handling
2. **Decision 2 ↔ Decision 4**: Palmer module structlog migration (Decision 4) must complete BEFORE xarray wrapper (Decision 2)
3. **Decision 5 ↔ All Tracks**: CF metadata registry is central integration point for all xarray adapters

**Risk Mitigation:**

- **Decision 1 Risk**: NOAA archive changes → **Mitigation**: Checksum + fixture_version tracking
- **Decision 2 Risk**: Module exceeds 1,500 lines → **Mitigation**: Extraction threshold monitoring
- **Decision 3 Risk**: Property-based tests underestimated → **Mitigation**: SPI actual-time tracking, adjust estimates
- **Decision 4 Risk**: Inconsistent exceptions confuse users → **Mitigation**: Release notes documentation
- **Decision 5 Risk**: CF_METADATA coupling grows → **Mitigation**: Winston's helper function threshold monitoring

---

## Implementation Patterns & Consistency Rules

### Pattern Categories Defined

**Critical Conflict Points Identified:** 24 areas where AI agents could make different choices without explicit patterns

**Pattern Enforcement Philosophy:**
> "Consistency enables AI agents to work together seamlessly. Patterns should be prescriptive enough to prevent conflicts, but flexible enough to allow implementation optimization." — BMAD Architecture Principles

---

### Naming Patterns

#### Python Module Naming

**Rule:** All module names use snake_case with descriptive, domain-specific names (NEVER generic names)

**Patterns:**
```python
✅ CORRECT:
src/climate_indices/palmer.py              # Algorithm name
src/climate_indices/cf_metadata_registry.py # Purpose-driven
src/climate_indices/logging_config.py       # Configuration module

❌ WRONG:
src/climate_indices/utils.py               # Too generic (but exists in legacy)
src/climate_indices/helpers.py             # Vague purpose
src/climate_indices/pdsi.py                # Use algorithm name, not acronym variant
```

**v2.4.0 New Modules:**
- `cf_metadata_registry.py` (Track 0/Decision 5)
- NO new modules in Track 1-3 (enhancements to existing modules)

**Migration Note:** Existing `utils.py` remains for backward compatibility but NEVER add new utilities there — use specific modules

---

#### Function Naming

**Rule:** Use lowercase with underscores, verb-first for actions, noun for getters

**Computation Functions (indices.py, palmer.py, eto.py):**
```python
✅ CORRECT:
def palmer(precip, temp, awc, ...)          # Algorithm name as function
def eto_thornthwaite(temperature, latitude) # Prefix pattern: eto_<method>
def percentage_of_normal(precip, ...)       # Full descriptive name

❌ WRONG:
def calculate_palmer(...)                   # Redundant 'calculate' prefix
def pm_eto(...)                             # Use full name: eto_penman_monteith
def pnp(...)                                # Acronym-only (use percentage_of_normal)
```

**xarray Adapter Functions:**
```python
✅ CORRECT:
def palmer_xarray(precip, temp, awc, ...)  # Pattern: <algorithm>_xarray
def spi_xarray(precip, scale, ...)         # Consistent suffix

❌ WRONG:
def palmer_dataset(...)                     # Use _xarray suffix, not _dataset
def xarray_palmer(...)                      # Algorithm name first, not last
```

**Helper Functions:**
```python
✅ CORRECT:
def _validate_scale(scale, supported)      # Private helpers: leading underscore
def get_logger(__name__)                    # Verb-first for getters
def apply_cf_metadata(da, variable_name)    # Verb-first for actions

❌ WRONG:
def validate(scale)                         # Too generic
def logger_get(__name__)                    # Noun-first confusing
def cf_metadata_apply(da, variable_name)    # Wrong order
```

---

#### Variable Naming

**Rule:** Descriptive names following domain conventions (NO single-letter except loop indices)

**Climate Variables (Use Established Conventions):**
```python
✅ CORRECT:
precip = ...           # Precipitation (mm)
temp = ...             # Temperature (°C) — NOT temperature_celsius
pet = ...              # Potential ET (established acronym)
awc = ...              # Available Water Capacity (established acronym)
lat, lon = ...         # Latitude, longitude (established short forms)

❌ WRONG:
precipitation = ...    # Too verbose (breaks from climate science convention)
t = ...                # Single letter (unless loop index)
data = ...             # Too generic
df2 = ...              # Numbered variants indicate poor naming
```

**xarray Dimensions:**
```python
✅ CORRECT:
dims = ('time', 'lat', 'lon')              # CF convention order
dims = ('time', 'y', 'x')                  # Projected coordinates

❌ WRONG:
dims = ('t', 'latitude', 'longitude')      # Mixing short/long forms
dims = ('lat', 'lon', 'time')              # Time should be first dimension
```

---

#### Test File Naming

**Rule:** Pattern `test_<module>_<aspect>.py` for specific tests, `test_<algorithm>.py` for algorithm tests

**Test Organization:**
```python
✅ CORRECT:
tests/test_palmer.py                       # Palmer algorithm unit tests
tests/test_palmer_equivalence.py           # NumPy vs xarray equivalence
tests/test_property_based_palmer.py        # Property-based tests
tests/test_benchmark_palmer.py             # Performance benchmarks

tests/test_noaa_eddi_reference.py          # External reference validation

❌ WRONG:
tests/palmer_tests.py                      # Use test_ prefix
tests/test_palmer_unit.py                  # Redundant _unit (default assumption)
tests/test_palmer_integration.py           # Be specific: _equivalence, _reference, etc.
```

**Fixture Directories:**
```
✅ CORRECT:
tests/fixture/noaa-eddi-1month/            # Descriptive, hyphen-separated
tests/fixture/fao56-examples/              # Source-based naming

❌ WRONG:
tests/fixtures/...                         # Use singular 'fixture'
tests/fixture/eddi_reference/              # Use hyphens, not underscores
tests/data/...                             # Use 'fixture' directory name
```

---

### Structure Patterns

#### Project Organization (src-layout)

**Rule:** NEVER modify package structure — src-layout is established and immutable for v2.4.0

**Directory Structure (DO NOT CHANGE):**
```
climate_indices/                           # Project root
├── src/climate_indices/                   # Package source (importable)
│   ├── __init__.py                        # Public API exports
│   ├── indices.py                         # Core NumPy computation
│   ├── palmer.py                          # Palmer algorithm
│   ├── eto.py                             # Evapotranspiration functions
│   ├── compute.py                         # Shared computation utilities
│   ├── xarray_adapter.py                  # Decorator-based xarray wrapping
│   ├── typed_public_api.py                # Type-safe API dispatchers
│   ├── cf_metadata_registry.py            # NEW: CF metadata (v2.4.0)
│   ├── exceptions.py                      # Structured exception hierarchy
│   └── logging_config.py                  # structlog configuration
├── tests/                                 # Test suite (top-level, NOT in src/)
│   ├── conftest.py                        # pytest fixtures
│   ├── fixture/                           # Reference datasets
│   ├── helpers/                           # Test utilities
│   └── test_*.py                          # Test modules
├── docs/                                  # Documentation
├── pyproject.toml                         # Build configuration
└── uv.lock                                # Dependency lock file
```

**Module Boundaries (CRITICAL — DO NOT CROSS):**

1. **indices.py**: ONLY NumPy-based computation functions
   - NEVER add xarray logic to indices.py
   - NEVER add logging to indices.py (pure computation)
   - Pattern: Functions return ndarray or tuple[ndarray, ...]

2. **xarray_adapter.py**: ONLY decorator-based single-output adapters
   - Pattern C multi-output (Palmer) goes in palmer.py, NOT here
   - Decorators handle: input detection, conversion, CF metadata attachment

3. **typed_public_api.py**: ONLY @overload signatures and dispatchers
   - No computation logic
   - Pattern: Dispatch to indices.py or xarray_adapter.py based on isinstance()

4. **palmer.py**: EXCEPTION — Contains BOTH NumPy core AND xarray wrapper
   - Justified by multi-output complexity (Decision 2)
   - Clear section markers MANDATORY (see Decision 2 structure)

---

#### Test Organization Patterns

**Rule:** Test files mirror source structure with specific suffixes for test types

**Test Type Patterns:**
```
tests/
├── test_<algorithm>.py                    # Unit tests for algorithm
├── test_<algorithm>_equivalence.py        # NumPy vs xarray validation
├── test_property_based_<algorithm>.py     # Hypothesis property tests
├── test_benchmark_<algorithm>.py          # Performance regression tracking
├── test_noaa_<index>_reference.py         # External reference validation
└── test_exceptions.py                     # Exception hierarchy tests
```

**Fixture Organization:**
```
tests/fixture/
├── <source>-<index>-<variant>/            # Hyphen-separated naming
│   ├── provenance.json                    # MANDATORY for external refs (Decision 1)
│   └── <index>_<variant>_reference.nc     # NetCDF reference data
└── fao56-examples/                        # Embedded test data (no provenance needed)
    ├── example_17_bangkok.py              # Python module with test data
    └── example_18_uccle.py
```

**conftest.py Fixture Scope:**
```python
✅ CORRECT:
@pytest.fixture(scope='module')            # Expensive .npy or .nc loading
def noaa_eddi_reference():
    return xr.load_dataset('tests/fixture/noaa-eddi-3month/...')

@pytest.fixture(scope='function')          # Cheap data generation
def sample_precipitation():
    return np.random.rand(120)

❌ WRONG:
@pytest.fixture(scope='session')           # Avoid session scope (state leakage risk)
@pytest.fixture                            # Missing explicit scope
```

---

### Format Patterns

#### NetCDF/xarray Conventions

**Rule:** All xarray outputs MUST follow CF Metadata Conventions v1.10

**Dimension Ordering (MANDATORY):**
```python
✅ CORRECT:
dims = ('time', 'lat', 'lon')              # CF convention: time first
dims = ('time', 'y', 'x')                  # Projected coordinates
dims = ('time',)                           # 1D time series

❌ WRONG:
dims = ('lat', 'lon', 'time')              # Time must be first
dims = ('x', 'y', 'time')                  # CF convention violated
```

**CF Metadata Application:**
```python
✅ CORRECT:
from climate_indices.cf_metadata_registry import CF_METADATA

output_da.attrs.update(CF_METADATA['pdsi'])  # Use registry (Decision 5)

❌ WRONG:
output_da.attrs['long_name'] = 'PDSI'        # NEVER hard-code metadata
output_da.attrs.update({                     # NEVER inline metadata
    'long_name': 'Palmer Drought Severity Index',
    'units': '1'
})
```

**DataArray vs Dataset Return Types:**
```python
✅ CORRECT:
# Single-output: Return DataArray
def spi_xarray(...) -> xr.DataArray:
    return output_da

# Multi-output: Return Dataset (Palmer only)
def palmer_xarray(...) -> xr.Dataset:
    return xr.Dataset({
        'pdsi': pdsi_da,
        'phdi': phdi_da,
        'pmdi': pmdi_da,
        'z_index': z_index_da
    })

❌ WRONG:
# Multi-output returning tuple of DataArrays
def palmer_xarray(...) -> tuple[xr.DataArray, ...]:  # Use Dataset instead
```

---

#### JSON Format Patterns

**Provenance Metadata (Decision 1):**
```json
✅ CORRECT:
{
  "source": "NOAA PSL EDDI CONUS Archive",
  "url": "https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/data/2019/EDDI_ETrs_01mn_20190101_topsnap.nc",
  "download_date": "2026-02-20",
  "subset_description": "3-month EDDI for CONUS, 2019-01 to 2020-12, 1° resolution, 120 time steps",
  "checksum_sha256": "a3f5c2d8...",
  "variable_name": "eddi_3month",
  "validation_tolerance": 1e-5,
  "fixture_version": "1.0",
  "notes": "Non-parametric ranking FP accumulation requires looser tolerance than parametric fitting (1e-8)"
}

❌ WRONG:
{
  "source": "NOAA",                        # Too vague
  "date": "2026-02-20",                    # Use download_date
  "tolerance": 0.00001,                    # Use scientific notation: 1e-5
  // Missing checksum_sha256                # NO COMMENTS in JSON
  "variableName": "eddi_3month"            # Use snake_case, not camelCase
}
```

**Palmer params_dict Serialization (FR-PALMER-004):**
```python
✅ CORRECT:
output_ds.attrs['params_dict'] = json.dumps({
    'calibration_start_year': 1991,
    'calibration_end_year': 2020,
    'pe_method': 'thornthwaite'
}, indent=2)

# Dual access pattern:
params_str = output_ds.attrs['params_dict']  # JSON string
params_obj = json.loads(params_str)          # Python dict

❌ WRONG:
output_ds.attrs['params'] = {...}          # Use params_dict name
output_ds.attrs['params_dict'] = {...}     # Must be JSON string, not dict
```

---

### Communication Patterns

#### structlog Event Naming

**Rule:** Use lowercase with underscores, past tense for completion events

**Calculation Lifecycle Events:**
```python
✅ CORRECT:
logger.info('calculation_started', function=__name__, scale=scale)
# ... computation ...
logger.info('calculation_completed', function=__name__, duration_ms=123.45)

logger.warning('short_calibration_period', years=25, minimum_recommended=30)

logger.error('distribution_fitting_failed',
             distribution='gamma',
             error=str(e),
             scale=scale)

❌ WRONG:
logger.info('Starting calculation')        # Use structured events, not prose
logger.info('calculationStarted', ...)     # Use snake_case, not camelCase
logger.info('calculation_complete', ...)   # Use past tense: calculation_completed
logger.debug('computation', ...)           # Too generic
```

**Logging Migration Patterns:**

**New Modules (Track 1: PM-ET):**
```python
✅ CORRECT:
from climate_indices.logging_config import get_logger
logger = get_logger(__name__)
```

**Legacy Modules (Track 0: Palmer, PNP, PCI structlog migration):**
```python
✅ CORRECT BEFORE MIGRATION:
from climate_indices import utils
logger = utils.get_logger(__name__)

✅ CORRECT AFTER MIGRATION:
from climate_indices.logging_config import get_logger
logger = get_logger(__name__)

❌ WRONG:
# Mixing logging patterns in same module
from climate_indices import utils
from climate_indices.logging_config import get_logger  # Pick ONE, not both
```

**CRITICAL RULE:** NEVER mix logging patterns within same module

---

#### xarray Adapter Lifecycle

**Rule:** Adapters follow detect → convert → compute → attach metadata → return pattern

**Standard Adapter Pattern (Decorator-based, single-output):**
```python
✅ CORRECT:
@xarray_adapter(CF_METADATA['spi'])
def spi(precip, scale, ...):
    """Core NumPy computation."""
    # Decorator handles:
    # 1. Input detection (isinstance check)
    # 2. DataArray → ndarray conversion
    # 3. Computation via NumPy function
    # 4. CF metadata attachment
    # 5. DataArray return
    return result_ndarray
```

**Manual Adapter Pattern (Multi-output, Palmer only):**
```python
✅ CORRECT:
def palmer_xarray(
    precip: xr.DataArray,
    temp: xr.DataArray,
    awc: xr.DataArray | float,
    ...
) -> xr.Dataset:
    """Manual xarray wrapper for Palmer multi-output."""

    # 1. Input validation (AWC time dimension check)
    if isinstance(awc, xr.DataArray) and 'time' in awc.dims:
        raise InvalidArgumentError(
            "AWC must not have time dimension",
            awc_dims=awc.dims,
            expected_dims=('lat', 'lon')
        )

    # 2. Convert inputs to NumPy
    precip_np = precip.values
    temp_np = temp.values
    awc_np = awc if isinstance(awc, float) else awc.values

    # 3. Compute via NumPy core (returns tuple)
    pdsi_np, phdi_np, pmdi_np, z_index_np = palmer(precip_np, temp_np, awc_np, ...)

    # 4. Wrap outputs as DataArrays
    pdsi_da = xr.DataArray(pdsi_np, coords=precip.coords, dims=precip.dims)
    # ... repeat for phdi, pmdi, z_index ...

    # 5. Apply CF metadata (from registry)
    pdsi_da.attrs.update(CF_METADATA['pdsi'])
    # ... repeat for phdi, pmdi, z_index ...

    # 6. Return Dataset
    return xr.Dataset({
        'pdsi': pdsi_da,
        'phdi': phdi_da,
        'pmdi': pmdi_da,
        'z_index': z_index_da
    })
```

---

### Process Patterns

#### Error Handling Standards

**Rule:** Use structured exceptions with keyword-only context (Decision 4)

**Exception Raising Pattern:**
```python
✅ CORRECT:
if scale not in (1, 2, 3, 6, 9, 12, 24):
    raise InvalidScaleError(
        f"Scale {scale} not supported for SPI calculation",
        scale=scale,
        supported_scales=[1, 2, 3, 6, 9, 12, 24]
    )

if data_array.shape[0] < 30:
    raise InsufficientDataError(
        f"Calibration period too short: {data_array.shape[0]} years < 30 years recommended",
        provided_length=data_array.shape[0],
        recommended_minimum=30,
        index='SPI'
    )

❌ WRONG:
if scale not in (1, 2, 3, 6, 9, 12, 24):
    raise ValueError('Invalid scale')      # Use structured exception

if scale not in (1, 2, 3, 6, 9, 12, 24):
    raise InvalidScaleError(scale)         # Missing context kwargs

raise Exception('Bad input')               # Never raise base Exception
```

**Exception Docstring Pattern:**
```python
✅ CORRECT:
def spi(precip, scale, ...):
    """
    Calculate Standardized Precipitation Index.

    Parameters
    ----------
    precip : ndarray
        Precipitation values (mm)
    scale : int
        Timescale in months

    Returns
    -------
    ndarray
        SPI values

    Raises
    ------
    InvalidScaleError
        If scale not in [1, 2, 3, 6, 9, 12, 24]
    InsufficientDataError
        If calibration period < 30 years
    DistributionFittingError
        If gamma distribution fitting fails
    """
```

---

#### Numerical Precision Patterns

**Rule:** NEVER use == with floats, always use np.isclose() or np.testing.assert_allclose()

**Test Assertion Patterns:**
```python
✅ CORRECT:
# Equivalence tests (NumPy vs xarray)
np.testing.assert_allclose(
    numpy_result,
    xarray_result.values,
    atol=1e-8,
    rtol=0,
    err_msg="NumPy and xarray results must match within 1e-8 tolerance"
)

# EDDI NOAA reference validation (non-parametric)
np.testing.assert_allclose(
    computed_eddi,
    noaa_reference,
    atol=1e-5,  # Looser tolerance for ranking FP accumulation
    rtol=0
)

# FAO56 example validation
np.testing.assert_allclose(
    computed_eto,
    expected_eto_mm_day,
    atol=0.05,  # ±0.05 mm/day per FAO56 guidance
    rtol=0
)

❌ WRONG:
assert numpy_result == xarray_result.values  # NEVER use == with floats
assert abs(computed - expected) < 0.01       # Use numpy testing utilities
```

**Tolerance Guidelines:**
| Context | Tolerance | Rationale |
|---------|-----------|-----------|
| NumPy ↔ xarray equivalence | 1e-8 (atol) | Float64 precision (NFR-PATTERN-EQUIV) |
| EDDI NOAA reference | 1e-5 (atol) | Non-parametric ranking FP accumulation (NFR-EDDI-VAL) |
| FAO56 example end-to-end | 0.05 mm/day (atol) | FAO56 published example tolerance |
| FAO56 intermediate values | 0.01 kPa (atol) | Vapor pressure intermediate precision |
| Property-based test bounds | 1e-6 to 1e-4 (atol) | Context-dependent, document rationale |

---

#### Type Annotation Patterns

**Rule:** All function signatures MUST have complete type annotations (Python 3.10+ syntax)

**Function Signature Pattern:**
```python
✅ CORRECT:
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

def palmer(
    precip: np.ndarray,
    temp: np.ndarray,
    awc: np.ndarray | float,
    calibration_start_year: int,
    calibration_end_year: int,
    pe_method: str = 'thornthwaite'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Palmer Drought Severity Index with multi-output."""
    ...

def palmer_xarray(
    precip: xr.DataArray,
    temp: xr.DataArray,
    awc: xr.DataArray | float,
    ...
) -> xr.Dataset:
    """xarray wrapper for Palmer multi-output."""
    ...

❌ WRONG:
def palmer(precip, temp, awc, ...):        # Missing type annotations

from typing import Optional, Union
def palmer(
    awc: Union[np.ndarray, float]          # Use | syntax, not Union
):

def palmer(...) -> (np.ndarray, ...):      # Use tuple[type, ...], not (type, ...)
```

**@overload Pattern (typed_public_api.py):**
```python
✅ CORRECT:
from typing import overload

@overload
def palmer(
    precip: np.ndarray,
    temp: np.ndarray,
    awc: np.ndarray | float,
    ...
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

@overload
def palmer(
    precip: xr.DataArray,
    temp: xr.DataArray,
    awc: xr.DataArray | float,
    ...
) -> xr.Dataset: ...

def palmer(precip, temp, awc, ...):
    """Dispatcher implementation."""
    if isinstance(precip, xr.DataArray):
        return palmer_xarray(precip, temp, awc, ...)
    else:
        from climate_indices import palmer as palmer_module
        return palmer_module.palmer(precip, temp, awc, ...)
```

---

### Enforcement Guidelines

**All AI Agents MUST:**

1. **Read project-context.md BEFORE implementation** — Contains immutable development rules
2. **Use this architecture document as single source of truth** — All decisions documented here
3. **Follow naming patterns EXACTLY** — No variations allowed
4. **Respect module boundaries** — NEVER add xarray logic to indices.py
5. **Use CF_METADATA registry** — NEVER hard-code CF metadata
6. **Apply structured exceptions** — ValueError only in legacy modules not yet migrated
7. **Validate numerical precision** — Use appropriate tolerance for context
8. **Document pattern deviations** — If deviation necessary, document WHY in code comments

**Pattern Enforcement:**

**Automated (CI/CD):**
- ruff check (PEP 8, import order, line length 120)
- mypy --strict (type annotation completeness)
- pytest (numerical equivalence tests enforce tolerance patterns)

**Manual (Code Review):**
- CF metadata registry usage (no hard-coded metadata)
- Logging pattern consistency (no mixed patterns in same module)
- Module boundary respect (no xarray in indices.py)
- Test naming convention compliance

**Pattern Violation Process:**

1. **Discovery**: Agent or human reviewer identifies pattern violation
2. **Documentation**: Create issue with pattern reference and violation example
3. **Resolution**: Fix violation to match pattern OR propose pattern amendment (rare)
4. **Pattern Update**: If pattern needs amendment, update this architecture document

---

### Pattern Examples

**Good Examples:**

**Example 1: Adding New Index with xarray Support**
```python
# Step 1: Add NumPy core to indices.py
def percentage_of_normal(
    precip: np.ndarray,
    precip_normal: np.ndarray
) -> np.ndarray:
    """Calculate Percent of Normal Precipitation."""
    return (precip / precip_normal) * 100.0

# Step 2: Add CF metadata to cf_metadata_registry.py
CF_METADATA['pnp'] = {
    'long_name': 'Percent of Normal Precipitation',
    'units': 'percent',
    'standard_name': None,
    'references': 'https://www.drought.gov/data-maps-tools/percent-of-normal-precipitation',
    'valid_range': (0.0, 500.0),
}

# Step 3: Add xarray adapter to xarray_adapter.py (decorator pattern)
@xarray_adapter(CF_METADATA['pnp'])
def percentage_of_normal(precip, precip_normal):
    return indices.percentage_of_normal(precip, precip_normal)

# Step 4: Add typed API to typed_public_api.py
@overload
def percentage_of_normal(
    precip: np.ndarray,
    precip_normal: np.ndarray
) -> np.ndarray: ...

@overload
def percentage_of_normal(
    precip: xr.DataArray,
    precip_normal: xr.DataArray
) -> xr.DataArray: ...
```

**Example 2: NOAA Reference Validation Test**
```python
# tests/test_noaa_eddi_reference.py
import json
import pytest
import xarray as xr
import numpy as np
from climate_indices import eddi

def test_eddi_3month_noaa_reference():
    """Validate EDDI 3-month against NOAA PSL reference data."""

    # Load provenance
    with open('tests/fixture/noaa-eddi-3month/provenance.json') as f:
        provenance = json.load(f)

    # Load reference data
    noaa_ref = xr.open_dataset('tests/fixture/noaa-eddi-3month/eddi_3month_reference.nc')

    # Load input data (PET from same source)
    pet_input = xr.open_dataset('tests/fixture/noaa-eddi-3month/pet_input.nc')

    # Compute EDDI
    computed_eddi = eddi(pet_input['pet'], scale=3)

    # Validate with provenance tolerance
    np.testing.assert_allclose(
        computed_eddi.values,
        noaa_ref['eddi'].values,
        atol=provenance['validation_tolerance'],
        rtol=0,
        err_msg=f"EDDI computation must match NOAA reference within {provenance['validation_tolerance']} tolerance"
    )

    # Verify checksum (fixture integrity)
    import hashlib
    with open('tests/fixture/noaa-eddi-3month/eddi_3month_reference.nc', 'rb') as f:
        computed_checksum = hashlib.sha256(f.read()).hexdigest()
    assert computed_checksum == provenance['checksum_sha256'], \
        "Reference fixture checksum mismatch — file may be corrupted"
```

---

**Anti-Patterns:**

**Anti-Pattern 1: Hard-Coding CF Metadata**
```python
❌ WRONG:
def spi_xarray(precip, scale):
    result = indices.spi(precip, scale)
    result_da = xr.DataArray(result, coords=precip.coords)

    # ANTI-PATTERN: Hard-coded metadata
    result_da.attrs['long_name'] = 'Standardized Precipitation Index'
    result_da.attrs['units'] = '1'

    return result_da

✅ CORRECT:
from climate_indices.cf_metadata_registry import CF_METADATA

def spi_xarray(precip, scale):
    result = indices.spi(precip, scale)
    result_da = xr.DataArray(result, coords=precip.coords)

    # Use registry
    result_da.attrs.update(CF_METADATA['spi'])

    return result_da
```

**Anti-Pattern 2: Mixing Logging Patterns**
```python
❌ WRONG:
# palmer.py (ANTI-PATTERN: Mixed logging)
from climate_indices import utils
from climate_indices.logging_config import get_logger

logger_old = utils.get_logger(__name__)      # Legacy pattern
logger_new = get_logger(__name__)            # New pattern

def palmer(...):
    logger_old.info('Starting Palmer calculation')  # Mixed usage!
    ...
    logger_new.info('calculation_completed')

✅ CORRECT:
# palmer.py (After migration)
from climate_indices.logging_config import get_logger

logger = get_logger(__name__)  # Single pattern

def palmer(...):
    logger.info('calculation_started', function='palmer')
    ...
    logger.info('calculation_completed', duration_ms=123.45)
```

**Anti-Pattern 3: Incorrect Test Tolerance**
```python
❌ WRONG:
def test_palmer_numpy_xarray_equivalence():
    """Test Palmer NumPy vs xarray equivalence."""
    numpy_result, _, _, _ = palmer(precip_np, temp_np, awc_np, ...)
    xarray_result = palmer_xarray(precip_da, temp_da, awc_da, ...)

    # ANTI-PATTERN: Using EDDI tolerance for Palmer
    np.testing.assert_allclose(numpy_result, xarray_result['pdsi'].values, atol=1e-5)

✅ CORRECT:
def test_palmer_numpy_xarray_equivalence():
    """Test Palmer NumPy vs xarray equivalence."""
    numpy_result, _, _, _ = palmer(precip_np, temp_np, awc_np, ...)
    xarray_result = palmer_xarray(precip_da, temp_da, awc_da, ...)

    # Correct tolerance: 1e-8 for equivalence tests (NFR-PATTERN-EQUIV)
    np.testing.assert_allclose(
        numpy_result,
        xarray_result['pdsi'].values,
        atol=1e-8,
        rtol=0,
        err_msg="Palmer NumPy and xarray must match within 1e-8 tolerance"
    )
```

---

## Project Structure & Boundaries

### Complete Project Directory Structure

```
climate_indices/                                    # Project root (Git repository)
│
├── .github/                                        # GitHub-specific configuration
│   └── workflows/
│       ├── build.yml                               # CI/CD: Build + test matrix (Python 3.10-3.13)
│       ├── publish.yml                             # PyPI distribution automation
│       └── codeql.yml                              # Security scanning
│
├── _bmad/                                          # BMAD framework (AI-assisted development)
│   ├── core/                                       # Core BMAD agents/workflows
│   ├── bmm/                                        # BMAD Module Manager workflows
│   ├── tea/                                        # Test Engineering Automation
│   └── cis/                                        # Climate Indices Specialist (project-specific)
│
├── _bmad-output/                                   # BMAD artifacts (not in package)
│   └── planning-artifacts/
│       ├── prd.md                                  # v2.4.0 PRD (this document's sibling)
│       ├── architecture.md                         # THIS DOCUMENT
│       ├── implementation-readiness-report.md      # Pre-implementation validation
│       └── research/
│           ├── technical-penman-monteith.md        # PM FAO56 equations analysis
│           ├── technical-palmer-modernization.md   # xarray multi-output patterns
│           └── technical-eddi-validation.md        # NOAA reference validation approach
│
├── src/climate_indices/                            # Package source (src-layout)
│   │
│   ├── __init__.py                                 # Public API exports
│   │                                               # Track 0: Add percentage_of_normal, pci to exports
│   │                                               # Track 1: Add eto_penman_monteith to exports
│   │                                               # Track 2: Add eddi to exports
│   │
│   ├── indices.py                                  # CORE: NumPy-based computation functions
│   │                                               # IMMUTABLE BOUNDARY: NEVER add xarray logic here
│   │                                               # Track 0: NO CHANGES (pattern work in other modules)
│   │                                               # Track 1: NO CHANGES (PM-ET goes in eto.py)
│   │                                               # Track 2: NO CHANGES (EDDI already exists here)
│   │
│   ├── palmer.py                                   # Palmer Drought Severity Index (912 lines → 1,060 lines)
│   │                                               # SECTION 1 (~912 lines): NumPy core computation
│   │                                               # SECTION 2 (~150 lines): xarray manual wrapper (NEW)
│   │                                               # Track 0 (CRITICAL PATH):
│   │                                               #   - structlog migration (FR-PATTERN-007)
│   │                                               #   - Exception migration (Decision 4)
│   │                                               # Track 3:
│   │                                               #   - Add palmer_xarray() function (FR-PALMER-001)
│   │                                               #   - Multi-output Dataset return (FR-PALMER-002)
│   │                                               #   - AWC spatial validation (FR-PALMER-003)
│   │                                               #   - params_dict serialization (FR-PALMER-004)
│   │
│   ├── eto.py                                      # Evapotranspiration functions
│   │                                               # Track 0:
│   │                                               #   - eto_thornthwaite xarray adapter (FR-PATTERN-003)
│   │                                               #   - eto_hargreaves xarray adapter (FR-PATTERN-004)
│   │                                               # Track 1 (PM FAO56):
│   │                                               #   - eto_penman_monteith() core function (FR-PM-001)
│   │                                               #   - Atmospheric parameter helpers (FR-PM-002, FR-PM-003)
│   │                                               #   - Humidity pathway dispatcher (FR-PM-004)
│   │                                               #   - eto_penman_monteith_xarray() wrapper (FR-PM-006)
│   │
│   ├── compute.py                                  # Shared computation utilities
│   │                                               # Track 0: NO CHANGES
│   │                                               # Track 1: POTENTIAL additions (vapor pressure calculations)
│   │
│   ├── xarray_adapter.py                           # Decorator-based xarray wrapping
│   │                                               # BOUNDARY: Only single-output adapters (NOT Palmer)
│   │                                               # Track 0:
│   │                                               #   - percentage_of_normal adapter (FR-PATTERN-001)
│   │                                               #   - pci adapter (FR-PATTERN-002)
│   │                                               #   - eto_thornthwaite adapter (FR-PATTERN-003)
│   │                                               #   - eto_hargreaves adapter (FR-PATTERN-004)
│   │                                               # Track 1:
│   │                                               #   - eto_penman_monteith adapter (FR-PM-006)
│   │                                               # Track 2:
│   │                                               #   - eddi adapter (FR-EDDI-002)
│   │                                               #   - percentage_of_normal adapter (FR-PNP-001)
│   │
│   ├── typed_public_api.py                         # Type-safe API dispatchers (210 lines → 350-400 lines)
│   │                                               # BOUNDARY: Only @overload + isinstance() dispatch
│   │                                               # THRESHOLD: Extract dispatcher pattern at 300 lines
│   │                                               # Track 0:
│   │                                               #   - percentage_of_normal @overload (FR-PATTERN-001)
│   │                                               #   - pci @overload (FR-PATTERN-002)
│   │                                               #   - eto_thornthwaite @overload (FR-PATTERN-003)
│   │                                               #   - eto_hargreaves @overload (FR-PATTERN-004)
│   │                                               # Track 1:
│   │                                               #   - eto_penman_monteith @overload (FR-PM-006)
│   │                                               # Track 2:
│   │                                               #   - eddi @overload (FR-EDDI-002)
│   │                                               # Track 3:
│   │                                               #   - palmer @overload (numpy→tuple, xarray→Dataset) (FR-PALMER-006)
│   │
│   ├── cf_metadata_registry.py                     # CF metadata definitions (NEW in v2.4.0)
│   │                                               # Decision 5: Separate module for CF metadata
│   │                                               # Track 0: Create stub with existing index metadata
│   │                                               # Track 1: Add eto_penman_monteith metadata
│   │                                               # Track 2: Add eddi, pnp metadata
│   │                                               # Track 3: Add pdsi, phdi, pmdi, z_index metadata (4 entries)
│   │
│   ├── exceptions.py                               # Structured exception hierarchy
│   │                                               # Track 0: POTENTIAL additions (new exception types as needed)
│   │                                               # Decision 4: Exception migration during pattern work
│   │
│   └── logging_config.py                           # structlog configuration
│                                                   # Track 0: NO CHANGES (already established in v2.3.0)
│
├── tests/                                          # Test suite (top-level, NOT in src/)
│   │
│   ├── conftest.py                                 # pytest fixtures (shared across all tests)
│   │                                               # Track 1: Add FAO56 example fixtures (module scope)
│   │                                               # Track 2: Add NOAA EDDI reference fixtures (module scope)
│   │                                               # Track 3: Add Palmer multi-output fixtures
│   │
│   ├── fixture/                                    # Reference datasets (external validation data)
│   │   │
│   │   ├── noaa-eddi-1month/                       # Track 2: EDDI 1-month NOAA reference (NEW)
│   │   │   ├── provenance.json                     # Decision 1: Provenance metadata
│   │   │   ├── eddi_1month_reference.nc            # NOAA PSL reference data subset
│   │   │   └── pet_input.nc                        # Corresponding PET input
│   │   │
│   │   ├── noaa-eddi-3month/                       # Track 2: EDDI 3-month NOAA reference (NEW)
│   │   │   ├── provenance.json
│   │   │   ├── eddi_3month_reference.nc
│   │   │   └── pet_input.nc
│   │   │
│   │   ├── noaa-eddi-6month/                       # Track 2: EDDI 6-month NOAA reference (NEW)
│   │   │   ├── provenance.json
│   │   │   ├── eddi_6month_reference.nc
│   │   │   └── pet_input.nc
│   │   │
│   │   └── fao56-examples/                         # Track 1: FAO56 worked examples (NEW)
│   │       ├── example_17_bangkok.py               # Example 17: Tropical climate
│   │       └── example_18_uccle.py                 # Example 18: Temperate climate
│   │
│   ├── helpers/                                    # Test utilities (shared test code)
│   │   └── climate_data_generators.py              # Track 0: Hypothesis generators (NEW)
│   │                                               #   - climate_precipitation()
│   │                                               #   - climate_temperature()
│   │                                               #   - soil_water_capacity()
│   │
│   ├── test_palmer.py                              # Palmer algorithm unit tests (existing)
│   │                                               # Track 0: NO CHANGES (Palmer core stable)
│   │
│   ├── test_palmer_equivalence.py                  # Palmer NumPy vs xarray equivalence (NEW)
│   │                                               # Track 3: FR-PALMER-007 (tolerance 1e-8)
│   │
│   ├── test_property_based_palmer.py               # Palmer property-based tests (NEW)
│   │                                               # Track 0: FR-PATTERN-012 (expanded coverage)
│   │                                               # Decision 3: 50-60 hours effort
│   │
│   ├── test_property_based_pnp.py                  # PNP property-based tests (NEW)
│   │                                               # Track 0: FR-PATTERN-010
│   │
│   ├── test_property_based_pci.py                  # PCI property-based tests (NEW)
│   │                                               # Track 0: FR-PATTERN-011
│   │
│   ├── test_benchmark_palmer.py                    # Palmer performance benchmarks (NEW)
│   │                                               # Track 0/1 BLOCKER: Baseline measurement
│   │                                               # NFR-PALMER-PERF: ≥80% speed validation
│   │
│   ├── test_eto_penman_monteith.py                 # PM-ET unit tests (NEW)
│   │                                               # Track 1: FR-PM-001 to FR-PM-005
│   │
│   ├── test_eto_penman_monteith_fao56.py           # PM-ET FAO56 example validation (NEW)
│   │                                               # Track 1: FR-PM-005 (Bangkok, Uccle examples)
│   │
│   ├── test_eto_penman_monteith_equivalence.py     # PM-ET NumPy vs xarray equivalence (NEW)
│   │                                               # Track 1: FR-PM-006 (tolerance 1e-8)
│   │
│   ├── test_noaa_eddi_reference.py                 # EDDI NOAA reference validation (NEW)
│   │                                               # Track 2: FR-TEST-004 (BLOCKER — no test = no merge)
│   │                                               # Decision 1: Uses provenance.json
│   │
│   ├── test_pnp.py                                 # Percent of Normal Precipitation tests (NEW)
│   │                                               # Track 2: FR-PNP-001
│   │
│   ├── test_exceptions.py                          # Exception hierarchy tests (existing)
│   │                                               # Track 0: UPDATED (new exception types from Decision 4)
│   │
│   └── test_cf_metadata_registry.py                # CF metadata registry tests (NEW)
│                                                   # Track 0: Validate registry structure
│                                                   # Decision 5: Ensure all v2.4.0 indices have entries
│
├── docs/                                           # Documentation (Sphinx-based)
│   │
│   ├── index.md                                    # Project overview (existing, updated)
│   │                                               # Track 1: Add PM-ET documentation
│   │                                               # Track 2: Add EDDI + PNP documentation
│   │
│   ├── api/                                        # API reference (auto-generated)
│   │
│   ├── examples/                                   # Usage examples
│   │   ├── palmer_xarray_example.ipynb             # Track 3: Palmer multi-output example (NEW)
│   │   ├── eddi_workflow_example.ipynb             # Track 2: EDDI workflow example (NEW)
│   │   └── penman_monteith_example.ipynb           # Track 1: PM-ET example (NEW)
│   │
│   └── testing/                                    # Testing documentation (NEW)
│       ├── property-based-test-guide.md            # Track 0: Created after SPI property tests
│       └── reference-validation-guide.md           # Track 2: NOAA provenance protocol guide
│
├── pyproject.toml                                  # Build configuration (PEP 517/621)
│   │                                               # Track 0: NO CHANGES (dependencies stable)
│   │                                               # Track 1: POTENTIAL hypothesis addition to [dev] group
│   │
├── uv.lock                                         # Dependency lock file (625KB, deterministic builds)
│   │                                               # Track 0: Regenerate after hypothesis addition
│   │
├── README.md                                       # Project README (existing, updated)
│   │                                               # Track 0: Update with v2.4.0 features
│   │
├── LICENSE                                         # License file (existing, no changes)
│
├── .gitignore                                      # Git ignore rules (existing, updated)
│   │                                               # Track 2: Add tests/fixture/**/*.nc (large NetCDF files)
│   │
├── .python-version                                 # Python version for tooling (3.11)
│
└── CHANGELOG.md                                    # Version history (existing, updated)
                                                    # Track 0-3: v2.4.0 release notes
```

---

### Architectural Boundaries

#### API Boundaries

**Public API Surface (src/climate_indices/__init__.py):**

```python
# Existing exports (v2.3.0 and earlier)
from climate_indices.indices import (
    spi,          # Standardized Precipitation Index
    spei,         # Standardized Precipitation Evapotranspiration Index
    pci,          # Precipitation Condition Index (Track 0: xarray adapter)
    eddi,         # Evaporative Demand Drought Index (Track 2: xarray adapter)
)

from climate_indices.palmer import palmer  # Track 3: palmer_xarray addition

from climate_indices.eto import (
    eto_thornthwaite,    # Track 0: xarray adapter
    eto_hargreaves,      # Track 0: xarray adapter
    eto_penman_monteith, # Track 1: NEW function + xarray adapter
)

# NEW exports in v2.4.0
from climate_indices.indices import (
    percentage_of_normal,  # Track 0/2: xarray adapter
)

# Exception hierarchy (public API)
from climate_indices.exceptions import (
    ClimateIndicesError,
    InvalidArgumentError,
    InsufficientDataError,
    ComputationError,
)

# CF metadata registry (public API for advanced users)
from climate_indices.cf_metadata_registry import CF_METADATA
```

**Type-Safe API Boundary (typed_public_api.py):**

- **Purpose**: Provide mypy-strict entry point with @overload signatures
- **Boundary Rule**: Dispatchers ONLY — no computation logic
- **Pattern**: `if isinstance(input, xr.DataArray)` → route to xarray_adapter or <module>_xarray
- **Growth Management**: Extract dispatcher pattern at 300-line threshold (Decision 2 note)

**Internal Boundaries (NOT public API):**

```python
# Private modules (leading underscore, NOT exported)
from climate_indices._constants import *  # DOES NOT EXIST (no private modules currently)

# Internal helpers (functions with leading underscore, NOT exported)
def _validate_scale(...)                   # In indices.py, internal only
def _fit_distribution(...)                 # In compute.py, internal only
```

---

#### Component Boundaries

**1. Computation Engine (indices.py, palmer.py, eto.py, compute.py)**

**Responsibility**: Pure NumPy-based climate index calculations

**Boundary Rules:**
- **NEVER import xarray** in these modules (computation engine is xarray-agnostic)
- **NEVER import logging_config** in indices.py (pure computation, no side effects)
- **OK to import structlog** in palmer.py, eto.py (instrumented computation)
- **Return types**: ndarray or tuple[ndarray, ...] ONLY

**Input Contracts:**
- Accept ndarray arguments (shape validation via assertions or exceptions)
- Accept scalar float/int parameters
- Return ndarray matching input shape or documented transformation

**Example (eto.py Track 1):**
```python
# CORRECT: Pure NumPy computation
def eto_penman_monteith(
    temp_min: np.ndarray,
    temp_max: np.ndarray,
    ...
) -> np.ndarray:
    """Penman-Monteith FAO56 reference evapotranspiration."""
    # Pure NumPy computation, no xarray imports
    return eto_mm_day
```

---

**2. xarray Adapter Layer (xarray_adapter.py, palmer_xarray in palmer.py)**

**Responsibility**: Wrap NumPy computation functions for xarray DataArray/Dataset I/O

**Boundary Rules:**
- **ONLY single-output adapters** in xarray_adapter.py (decorator pattern)
- **Multi-output adapters** live in source module (palmer_xarray in palmer.py)
- **MUST use CF_METADATA registry** (no hard-coded metadata)
- **Input detection**: `isinstance(input, xr.DataArray)`
- **Return types**: DataArray (single-output) or Dataset (multi-output)

**Adapter Lifecycle Contract:**
1. Input detection (isinstance check)
2. DataArray → ndarray conversion (.values)
3. Call computation engine (indices.py, palmer.py, eto.py)
4. ndarray → DataArray wrapping (coords, dims from input)
5. CF metadata attachment (from registry)
6. Return DataArray or Dataset

**Example (xarray_adapter.py Track 0):**
```python
# Decorator pattern for single-output
@xarray_adapter(CF_METADATA['pnp'])
def percentage_of_normal(precip, precip_normal):
    return indices.percentage_of_normal(precip, precip_normal)
```

**Example (palmer.py Track 3):**
```python
# Manual wrapper for multi-output (Pattern C)
def palmer_xarray(...) -> xr.Dataset:
    # Lifecycle: detect → convert → compute → wrap → metadata → return
    ...
    return xr.Dataset({'pdsi': ..., 'phdi': ..., 'pmdi': ..., 'z_index': ...})
```

---

**3. Type Safety Layer (typed_public_api.py)**

**Responsibility**: Provide mypy-strict dispatchers with @overload signatures

**Boundary Rules:**
- **@overload signatures** distinguish numpy→numpy vs xarray→xarray
- **Dispatcher implementation** uses isinstance() to route
- **NO computation logic** — only routing
- **NO imports** of indices.py at module level (circular dependency risk)

**Dispatcher Pattern:**
```python
@overload
def palmer(...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

@overload
def palmer(...) -> xr.Dataset: ...

def palmer(precip, temp, awc, ...):
    """Dispatcher: route to NumPy or xarray implementation."""
    if isinstance(precip, xr.DataArray):
        from climate_indices.palmer import palmer_xarray
        return palmer_xarray(precip, temp, awc, ...)
    else:
        from climate_indices import palmer as palmer_module
        return palmer_module.palmer(precip, temp, awc, ...)
```

---

**4. CF Metadata Management (cf_metadata_registry.py)**

**Responsibility**: Centralized CF metadata definitions for all climate indices

**Boundary Rules:**
- **Defines canonical structure** (TypedDict: long_name, units, standard_name, references, valid_range)
- **READ-ONLY usage** in adapters (no mutation of CF_METADATA dict)
- **Coupling threshold**: If 3+ modules duplicate metadata manipulation → extract apply_cf_metadata() helper

**Usage Pattern:**
```python
from climate_indices.cf_metadata_registry import CF_METADATA

output_da.attrs.update(CF_METADATA['pdsi'])  # Dict spread, no mutation
```

---

#### Service Boundaries (Not Applicable)

climate_indices is a library, not a service-oriented architecture. No internal services or microservices exist.

---

#### Data Boundaries

**1. Input Data Contracts**

**NumPy API (indices.py, palmer.py, eto.py):**
- Accept: `np.ndarray` (1D, 2D, 3D spatial-temporal data)
- Shape conventions: `(time,)`, `(time, lat, lon)`, `(time, y, x)`
- Data types: float64 (implicit, no float32)
- Missing data: np.nan (propagates through calculations)

**xarray API (xarray_adapter.py, palmer_xarray):**
- Accept: `xr.DataArray` with dims=('time', ...) or ('time', 'lat', 'lon')
- Coordinate requirements: 'time' coordinate MUST exist
- Chunk constraints: Palmer requires time=-1 (no temporal chunking)
- Missing data: np.nan or DataArray.isnull() → both propagate

---

**2. Output Data Contracts**

**NumPy API Returns:**
- Single-output functions: `np.ndarray` (same shape as input precip/temp)
- Multi-output (Palmer only): `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
  - Order: (pdsi, phdi, pmdi, z_index)

**xarray API Returns:**
- Single-output functions: `xr.DataArray` with CF metadata attrs
  - Coords: Inherited from input DataArray
  - Dims: Same as input
  - Attrs: From CF_METADATA registry
- Multi-output (Palmer only): `xr.Dataset` with 4 data variables
  - Variables: 'pdsi', 'phdi', 'pmdi', 'z_index'
  - Each variable has independent CF metadata attrs

---

**3. External Data Integration Points**

**NOAA PSL EDDI Reference Data (Track 2):**
- **Source**: https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/
- **Format**: NetCDF with CF conventions
- **Integration**: `tests/fixture/noaa-eddi-*/` subdirectories
- **Provenance**: provenance.json (Decision 1) tracks source, checksum, tolerance
- **Usage**: Reference validation tests (FR-TEST-004)

**FAO56 Worked Examples (Track 1):**
- **Source**: FAO Irrigation and Drainage Paper 56, Examples 17-18
- **Format**: Python modules (tests/fixture/fao56-examples/)
- **Integration**: Embedded test data (no external download)
- **Usage**: PM-ET algorithm validation (FR-PM-005)

**No Runtime External Data**: Library is offline-capable, no API calls or database connections

---

### Requirements to Structure Mapping

#### Track 0: Canonical Pattern Completion (12 FRs)

**FR-PATTERN-001: percentage_of_normal xarray adapter**
- **Files Modified**:
  - `src/climate_indices/xarray_adapter.py` (+15 lines: decorator-based adapter)
  - `src/climate_indices/typed_public_api.py` (+25 lines: @overload + dispatcher)
  - `src/climate_indices/cf_metadata_registry.py` (+8 lines: pnp metadata entry)
- **Files Created**:
  - `tests/test_pnp.py` (+80 lines: unit tests)
  - `tests/test_pnp_equivalence.py` (+60 lines: NumPy vs xarray)

**FR-PATTERN-002: pci xarray adapter**
- **Files Modified**:
  - `src/climate_indices/xarray_adapter.py` (+15 lines: decorator-based adapter)
  - `src/climate_indices/typed_public_api.py` (+25 lines: @overload + dispatcher)
- **Files Created**:
  - `tests/test_pci_equivalence.py` (+60 lines: NumPy vs xarray)

**FR-PATTERN-003 & FR-PATTERN-004: eto_thornthwaite, eto_hargreaves xarray adapters**
- **Files Modified**:
  - `src/climate_indices/xarray_adapter.py` (+30 lines: 2 adapters)
  - `src/climate_indices/typed_public_api.py` (+50 lines: 2× @overload + dispatchers)
- **Files Created**:
  - `tests/test_eto_equivalence.py` (+120 lines: Thornthwaite + Hargreaves equivalence)

**FR-PATTERN-007: Palmer structlog migration** (CRITICAL PATH for Track 3)
- **Files Modified**:
  - `src/climate_indices/palmer.py` (912 lines: structlog migration + exception migration per Decision 4)
    - Change: `from climate_indices import utils` → `from climate_indices.logging_config import get_logger`
    - Change: ~15-20 ValueError → structured exceptions
    - Effort: 12-16 hours (Decision 2 note)

**FR-PATTERN-008: eto_thornthwaite structlog lifecycle completion**
- **Files Modified**:
  - `src/climate_indices/eto.py` (+10 lines: lifecycle events for eto_thornthwaite)

**FR-PATTERN-009: Structured exceptions migration**
- **Files Modified** (Decision 4: Per-module incremental):
  - `src/climate_indices/palmer.py` (during FR-PATTERN-007)
  - `src/climate_indices/indices.py` (percentage_of_normal, pci during xarray work)
  - `src/climate_indices/exceptions.py` (potential new exception types)
- **Files Created**:
  - `tests/test_exceptions.py` (updated with new exception test cases)

**FR-PATTERN-010 to FR-PATTERN-012: Property-based tests**
- **Files Created**:
  - `tests/helpers/climate_data_generators.py` (+100 lines: Hypothesis generators)
  - `tests/test_property_based_pnp.py` (+120 lines: PNP properties)
  - `tests/test_property_based_pci.py` (+120 lines: PCI properties)
  - `tests/test_property_based_palmer.py` (+150 lines: expanded Palmer coverage)
- **Documentation Created**:
  - `docs/testing/property-based-test-guide.md` (+80 lines: after SPI completion per Amelia's note)

---

#### Track 1: PM-ET Foundation (6 FRs)

**FR-PM-001: Core Penman-Monteith calculation**
- **Files Modified**:
  - `src/climate_indices/eto.py` (+200 lines: eto_penman_monteith() + equation helpers)
  - `src/climate_indices/compute.py` (potential +50 lines: vapor pressure utilities)

**FR-PM-002 & FR-PM-003: Atmospheric parameters + vapor pressure helpers**
- **Files Modified**:
  - `src/climate_indices/eto.py` (+80 lines: helper functions for equations 7-13)

**FR-PM-004: Humidity pathway dispatcher**
- **Files Modified**:
  - `src/climate_indices/eto.py` (+40 lines: dispatcher logic)

**FR-PM-005: FAO56 worked example validation**
- **Files Created**:
  - `tests/fixture/fao56-examples/example_17_bangkok.py` (+30 lines: Bangkok data)
  - `tests/fixture/fao56-examples/example_18_uccle.py` (+30 lines: Uccle data)
  - `tests/test_eto_penman_monteith_fao56.py` (+100 lines: validation tests)

**FR-PM-006: xarray adapter with CF metadata**
- **Files Modified**:
  - `src/climate_indices/xarray_adapter.py` (+20 lines: decorator-based adapter)
  - `src/climate_indices/typed_public_api.py` (+30 lines: @overload + dispatcher)
  - `src/climate_indices/cf_metadata_registry.py` (+8 lines: eto_penman_monteith metadata)
- **Files Created**:
  - `tests/test_eto_penman_monteith_equivalence.py` (+80 lines: NumPy vs xarray)

**BLOCKER: Performance Baseline (NFR-PALMER-PERF dependency)**
- **Files Created**:
  - `tests/test_benchmark_palmer.py` (+60 lines: baseline measurement)
  - Must complete in Track 0 or Track 1 BEFORE Track 3 begins

---

#### Track 2: Index Coverage Expansion (5 FRs)

**FR-EDDI-001: NOAA reference dataset validation** (BLOCKING — no test = no merge)
- **Files Created** (Decision 1: Provenance protocol):
  - `tests/fixture/noaa-eddi-1month/provenance.json`
  - `tests/fixture/noaa-eddi-1month/eddi_1month_reference.nc` (downloaded, not tracked in Git)
  - `tests/fixture/noaa-eddi-1month/pet_input.nc`
  - `tests/fixture/noaa-eddi-3month/provenance.json`
  - `tests/fixture/noaa-eddi-3month/eddi_3month_reference.nc`
  - `tests/fixture/noaa-eddi-3month/pet_input.nc`
  - `tests/fixture/noaa-eddi-6month/provenance.json`
  - `tests/fixture/noaa-eddi-6month/eddi_6month_reference.nc`
  - `tests/fixture/noaa-eddi-6month/pet_input.nc`
  - `tests/test_noaa_eddi_reference.py` (+150 lines: validation tests with checksum verification)
- **Documentation Created**:
  - `docs/testing/reference-validation-guide.md` (+60 lines: provenance protocol guide)

**FR-EDDI-002: EDDI xarray adapter**
- **Files Modified**:
  - `src/climate_indices/xarray_adapter.py` (+15 lines: decorator-based adapter)
  - `src/climate_indices/typed_public_api.py` (+25 lines: @overload + dispatcher)
  - `src/climate_indices/cf_metadata_registry.py` (+8 lines: eddi metadata)

**FR-EDDI-003: CLI integration (Issue #414)**
- **Files Modified**:
  - CLI module (exact file TBD based on existing CLI structure)

**FR-EDDI-004: PM-ET recommendation docs**
- **Files Modified**:
  - `docs/index.md` (+20 lines: PM-ET recommendation for EDDI)
- **Files Created**:
  - `docs/examples/eddi_workflow_example.ipynb` (+60 lines: EDDI + PM-ET workflow)

**FR-PNP-001: Percent of Normal Precipitation xarray support**
- **Combined with FR-PATTERN-001** (already mapped in Track 0)

**FR-SCPDSI-001: Self-calibrating PDSI stub**
- **Files Modified**:
  - `src/climate_indices/indices.py` (+15 lines: stub function with NotImplementedError)
  - `docs/index.md` (+10 lines: scPDSI roadmap note)

---

#### Track 3: Advanced xarray Capabilities (7 FRs)

**FR-PALMER-001: Manual palmer_xarray() wrapper**
- **Files Modified** (Decision 2: Keep in palmer.py):
  - `src/climate_indices/palmer.py` (+150 lines: manual wrapper in SECTION 2)

**FR-PALMER-002: Multi-output Dataset return**
- **Included in FR-PALMER-001** (same function)

**FR-PALMER-003: AWC spatial parameter handling**
- **Included in FR-PALMER-001** (validation logic in palmer_xarray)

**FR-PALMER-004: params_dict JSON serialization**
- **Included in FR-PALMER-001** (Dataset attrs['params_dict'])

**FR-PALMER-005: CF metadata registry for Palmer variables**
- **Files Modified** (Decision 5):
  - `src/climate_indices/cf_metadata_registry.py` (+32 lines: 4 Palmer variable entries)

**FR-PALMER-006: typed_public_api @overload signatures**
- **Files Modified**:
  - `src/climate_indices/typed_public_api.py` (+40 lines: Palmer @overload with tuple vs Dataset)

**FR-PALMER-007: NumPy vs xarray equivalence tests**
- **Files Created**:
  - `tests/test_palmer_equivalence.py` (+150 lines: comprehensive multi-output scenarios)
    - Scalar AWC vs DataArray AWC
    - All-NaN input handling
    - Time series with gaps
    - Different Dask chunking strategies
    - params_dict serialization round-trip

---

### Integration Points

#### Internal Communication

**1. NumPy Core ↔ xarray Adapter**

**Pattern**: Adapters wrap NumPy computation functions

**Data Flow**:
```
User Input (xr.DataArray)
  ↓
xarray_adapter.py or <module>_xarray()
  ↓ (DataArray → ndarray conversion)
indices.py / palmer.py / eto.py (NumPy computation)
  ↓ (ndarray or tuple[ndarray, ...])
xarray_adapter.py or <module>_xarray()
  ↓ (ndarray → DataArray + CF metadata)
User Output (xr.DataArray or xr.Dataset)
```

**Communication Contract**:
- Adapters NEVER modify computation logic
- Computation functions NEVER import xarray
- Shape preservation: output shape matches input shape (or documented transformation)

---

**2. typed_public_api.py ↔ Computation Modules**

**Pattern**: Dispatcher routes based on input type

**Data Flow**:
```
User call: palmer(precip_da, ...)
  ↓
typed_public_api.palmer() dispatcher
  ↓ isinstance(precip, xr.DataArray)?
    ├─ True → palmer_xarray(precip_da, ...)
    └─ False → palmer.palmer(precip_np, ...)
  ↓
Return to user (Dataset or tuple[ndarray, ...])
```

**Communication Contract**:
- Dispatcher ONLY does type detection + routing
- Lazy imports avoid circular dependencies (`from climate_indices.palmer import palmer_xarray` inside dispatcher)
- @overload signatures provide compile-time type safety

---

**3. CF Metadata Registry ↔ Adapters**

**Pattern**: Centralized metadata lookup

**Data Flow**:
```
xarray_adapter.py or <module>_xarray()
  ↓
from climate_indices.cf_metadata_registry import CF_METADATA
  ↓
output_da.attrs.update(CF_METADATA['pdsi'])
  ↓
DataArray with CF metadata attrs
```

**Communication Contract**:
- CF_METADATA is read-only (no mutations)
- All xarray outputs MUST use registry (no hard-coded metadata)
- If metadata entry missing → KeyError (fail fast during development)

---

#### External Integrations

**1. NOAA PSL EDDI CONUS Archive (Track 2)**

**Integration Type**: Offline reference data download

**Process**:
1. **Manual download**: Developer downloads NetCDF subsets from NOAA archive
2. **Provenance capture**: Create provenance.json with source URL, checksum, tolerance (Decision 1)
3. **Git exclusion**: NetCDF files in .gitignore (too large for Git)
4. **Test usage**: test_noaa_eddi_reference.py validates computed EDDI against reference

**Integration Boundary**: `tests/fixture/noaa-eddi-*/`

**No Runtime Dependency**: Reference data used ONLY in testing, not required for library usage

---

**2. FAO56 Worked Examples (Track 1)**

**Integration Type**: Embedded test data (no external dependency)

**Process**:
1. **Manual transcription**: FAO56 book examples transcribed to Python modules
2. **Git tracking**: Python modules committed to repository
3. **Test usage**: test_eto_penman_monteith_fao56.py validates against examples

**Integration Boundary**: `tests/fixture/fao56-examples/`

**No External Dependency**: Embedded data eliminates download requirement

---

**3. PyPI Distribution (Continuous)**

**Integration Type**: Package publication

**Process**:
1. **Version tag**: `git tag v2.4.0`
2. **Build**: hatchling builds wheel + sdist
3. **Publish**: GitHub Actions → `twine upload` to PyPI
4. **Installation**: Users run `pip install climate-indices==2.4.0`

**Integration Boundary**: `pyproject.toml` (build configuration)

---

### File Organization Patterns

#### Configuration Files (Project Root)

**Build & Package Management:**
```
pyproject.toml          # PEP 517/621 build config (hatchling backend)
uv.lock                 # Deterministic dependency lock (625KB)
.python-version         # Python version hint for tools (3.11)
```

**Quality Tooling:**
```
pyproject.toml [tool.ruff]      # Linting + formatting config (line-length=120)
pyproject.toml [tool.mypy]      # Type checking config (strict mode, py310 target)
pyproject.toml [tool.pytest]    # Test configuration
pyproject.toml [tool.coverage]  # Coverage reporting
```

**Version Control:**
```
.gitignore              # Git exclusions (*.pyc, .venv/, tests/fixture/**/*.nc)
.github/workflows/      # CI/CD pipelines
```

---

#### Source Organization (src-layout)

**Package Structure (IMMUTABLE for v2.4.0):**
```
src/climate_indices/
├── __init__.py                    # Public API exports
├── indices.py                     # NumPy core (no xarray imports)
├── palmer.py                      # Palmer (both NumPy + xarray)
├── eto.py                         # Evapotranspiration functions
├── compute.py                     # Shared utilities
├── xarray_adapter.py              # Decorator-based adapters (single-output only)
├── typed_public_api.py            # @overload dispatchers
├── cf_metadata_registry.py        # NEW: CF metadata (Decision 5)
├── exceptions.py                  # Exception hierarchy
└── logging_config.py              # structlog config
```

**Module Coupling Rules:**
- indices.py: ONLY imports numpy, scipy
- palmer.py, eto.py: Import indices.py, compute.py, logging_config (structured modules)
- xarray_adapter.py: Imports indices.py, eto.py, CF_METADATA
- typed_public_api.py: Lazy imports (avoid circular dependencies)

---

#### Test Organization (Parallel to Source)

**Test Type Separation:**
```
tests/
├── test_<algorithm>.py                # Unit tests (core algorithm behavior)
├── test_<algorithm>_equivalence.py    # NumPy ↔ xarray validation
├── test_property_based_<algorithm>.py # Hypothesis property tests
├── test_benchmark_<algorithm>.py      # Performance regression tracking
├── test_noaa_<index>_reference.py     # External reference validation
└── test_exceptions.py                 # Exception hierarchy tests
```

**Fixture Organization:**
```
tests/fixture/
├── <source>-<index>-<variant>/        # External reference data
│   ├── provenance.json                # Decision 1: Provenance metadata
│   └── *.nc                           # NetCDF reference data (NOT in Git)
└── <source>-examples/                 # Embedded test data
    └── *.py                           # Python modules (IN Git)
```

**Helper Organization:**
```
tests/helpers/
└── climate_data_generators.py         # Hypothesis generators (Track 0)
```

---

#### Asset Organization (Documentation & Examples)

**Documentation:**
```
docs/
├── index.md                           # Project overview (updated each track)
├── api/                               # Auto-generated API reference (Sphinx)
├── examples/                          # Jupyter notebooks (Track 1-3)
│   ├── palmer_xarray_example.ipynb
│   ├── eddi_workflow_example.ipynb
│   └── penman_monteith_example.ipynb
└── testing/                           # NEW: Testing guides (Track 0, Track 2)
    ├── property-based-test-guide.md
    └── reference-validation-guide.md
```

**BMAD Artifacts (Not in Package):**
```
_bmad/                                 # BMAD framework (AI-assisted development)
_bmad-output/planning-artifacts/       # PRD, architecture, research docs
```

---

### Development Workflow Integration

#### Development Server Structure (Not Applicable)

climate_indices is a library — no development server required. Development workflow uses:
- `uv run pytest tests/` — Run tests
- `uv run ipython` — Interactive exploration
- `uv run jupyter lab` — Notebook development

---

#### Build Process Structure

**Build Command**: `uv run hatchling build`

**Build Inputs**:
- `src/climate_indices/**/*.py` — Package source
- `pyproject.toml` — Build configuration
- `README.md` — Long description for PyPI

**Build Outputs**:
```
dist/
├── climate_indices-2.4.0-py3-none-any.whl  # Wheel distribution
└── climate_indices-2.4.0.tar.gz            # Source distribution
```

**Build Process**:
1. hatchling reads `pyproject.toml [project]`
2. Collects files from `src/climate_indices/`
3. Generates metadata (PKG-INFO, METADATA)
4. Creates wheel (.whl) and sdist (.tar.gz)

---

#### Deployment Structure

**PyPI Deployment**:
```
GitHub Actions (.github/workflows/publish.yml)
  ↓ (on git tag v*)
Build: uv run hatchling build
  ↓
Publish: twine upload dist/*
  ↓
PyPI: https://pypi.org/project/climate-indices/2.4.0/
```

**User Installation**:
```bash
pip install climate-indices==2.4.0
# or
uv pip install climate-indices==2.4.0
```

**No Server Deployment**: Library only, no servers/containers

---

## Architecture Validation Results

### Coherence Validation ✅

#### Decision Compatibility Assessment

**Technology Stack Compatibility:**
✅ **VERIFIED** — All technology choices work together without conflicts:
- Python >=3.10,<3.14 compatible with all dependencies (scipy, xarray, dask, numpy)
- hatchling (PEP 517) compatible with uv package manager
- ruff (linter) + mypy (type checker) + pytest (testing) = standard Python toolchain
- structlog compatible with Python 3.10+ (no version conflicts)
- xarray>=2025.6.1 + dask>=2025.7.0 work together for chunked computation

**Version Compatibility:**
✅ **VERIFIED** — All specified versions are mutually compatible:
- scipy>=1.15.3 supports Python 3.10-3.13
- xarray>=2025.6.1 supports numpy>=1.24.0
- dask>=2025.7.0 compatible with xarray>=2025.6.1
- No dependency conflicts detected

**Decision Coherence Analysis:**
✅ **Decision 1 (NOAA Provenance)** + **Decision 3 (Property-Based Tests)** = Coherent
- JSON provenance format supports property-based test fixture handling
- fixture_version field enables test evolution without breaking existing tests

✅ **Decision 2 (Palmer Module Org)** + **Decision 4 (Exception Migration)** = Coherent
- Palmer structlog migration (Decision 4) must complete BEFORE xarray wrapper (Decision 2)
- Dependencies documented: Track 0 FR-PATTERN-007 blocks Track 3 FR-PALMER-001

✅ **Decision 5 (CF Metadata Registry)** + All Tracks = Coherent
- Central registry eliminates hard-coded metadata across all xarray adapters
- Stub-then-populate approach (Track 0 → Track 1 → Track 2 → Track 3) avoids coupling

**No Contradictory Decisions Detected**

---

#### Pattern Consistency Verification

**Naming Patterns Align with Technology Stack:**
✅ Python snake_case conventions throughout (indices.py, palmer_xarray, test_noaa_eddi_reference.py)
✅ CF metadata key naming matches NetCDF conventions ('long_name', 'units', 'standard_name')
✅ xarray dimension naming follows CF conventions (('time', 'lat', 'lon'))

**Structure Patterns Support Architectural Decisions:**
✅ src-layout pattern (established) supports clean packaging for PyPI
✅ Test organization (test_*_equivalence.py, test_property_based_*.py) supports NFR-PATTERN-EQUIV and Decision 3
✅ Fixture organization (tests/fixture/<source>-<index>-<variant>/) supports Decision 1 provenance protocol

**Communication Patterns Coherent:**
✅ structlog event naming (lowercase, past tense: 'calculation_completed') consistent across all modules
✅ Logging migration pattern (legacy → new) clearly documented with "NEVER mix" rule
✅ xarray adapter lifecycle (detect → convert → compute → attach metadata → return) standardized

**Process Patterns Coherent:**
✅ Error handling pattern (structured exceptions with keyword context) aligns with Decision 4
✅ Numerical precision patterns (tolerances: 1e-8, 1e-5, 0.05) documented with context-specific rationale
✅ Type annotation pattern (Python 3.10+ syntax, @overload for dispatchers) aligns with typed_public_api.py design

**No Pattern Conflicts Detected**

---

#### Structure Alignment Verification

**Project Structure Supports All Architectural Decisions:**
✅ src/climate_indices/cf_metadata_registry.py (Decision 5) has defined location
✅ tests/fixture/<source>-<index>-<variant>/provenance.json (Decision 1) has standardized structure
✅ src/climate_indices/palmer.py SECTION 1 + SECTION 2 (Decision 2) supports bimodal organization
✅ tests/helpers/climate_data_generators.py (Decision 3) supports property-based test implementation

**Boundaries Properly Defined and Respected:**
✅ indices.py boundary: "NEVER add xarray logic" documented with examples
✅ xarray_adapter.py boundary: "ONLY single-output adapters" documented (multi-output → source module)
✅ typed_public_api.py boundary: "NO computation logic, only routing" documented
✅ palmer.py exception: Multi-output complexity justifies BOTH NumPy + xarray in same file (Decision 2)

**Integration Points Properly Structured:**
✅ NumPy Core ↔ xarray Adapter: Data flow documented with conversion contracts
✅ typed_public_api.py ↔ Computation Modules: Dispatcher pattern with lazy imports
✅ CF Metadata Registry ↔ Adapters: Read-only lookup with KeyError fail-fast
✅ NOAA PSL ↔ Test Fixtures: Offline integration with provenance tracking

**Structure Supports Chosen Patterns:**
✅ Decorator pattern (@xarray_adapter) works with xarray_adapter.py organization
✅ Manual wrapper pattern (palmer_xarray) works with Decision 2 module organization
✅ @overload pattern works with typed_public_api.py dispatcher structure
✅ Test type separation (unit, equivalence, property-based, benchmark, reference) scalable to v2.4.0 growth

**No Structural Misalignments Detected**

---

### Requirements Coverage Validation ✅

#### Functional Requirements Coverage (30 FRs in v2.4.0)

**Track 0: Canonical Pattern Completion (12 FRs) — FULL COVERAGE ✅**

| FR | Requirement | Architectural Support |
|----|-------------|----------------------|
| FR-PATTERN-001 | percentage_of_normal xarray adapter | xarray_adapter.py (decorator pattern) + typed_public_api.py (@overload) + cf_metadata_registry.py (pnp entry) |
| FR-PATTERN-002 | pci xarray adapter | xarray_adapter.py (decorator pattern) + typed_public_api.py (@overload) |
| FR-PATTERN-003 | eto_thornthwaite xarray adapter | xarray_adapter.py (decorator pattern) + typed_public_api.py (@overload) |
| FR-PATTERN-004 | eto_hargreaves xarray adapter | xarray_adapter.py (decorator pattern) + typed_public_api.py (@overload) |
| FR-PATTERN-007 | Palmer structlog migration | Decision 4 (per-module exception migration) + logging_config import pattern |
| FR-PATTERN-008 | eto_thornthwaite lifecycle completion | eto.py (lifecycle events: calculation_started, calculation_completed) |
| FR-PATTERN-009 | Structured exceptions | Decision 4 (incremental migration strategy) + exceptions.py (hierarchy) |
| FR-PATTERN-010 | PNP property-based tests | Decision 3 (comprehensive strategy) + tests/test_property_based_pnp.py |
| FR-PATTERN-011 | PCI property-based tests | Decision 3 (comprehensive strategy) + tests/test_property_based_pci.py |
| FR-PATTERN-012 | Expanded SPEI/Palmer property-based coverage | Decision 3 (50-60 hour budget) + tests/test_property_based_palmer.py |

**Track 1: PM-ET Foundation (6 FRs) — FULL COVERAGE ✅**

| FR | Requirement | Architectural Support |
|----|-------------|----------------------|
| FR-PM-001 | Penman-Monteith FAO56 core | eto.py (+200 lines: eto_penman_monteith + equation helpers) |
| FR-PM-002 | Atmospheric parameters | eto.py (+80 lines: equations 7-13 helpers) |
| FR-PM-003 | Vapor pressure helpers | eto.py / compute.py (vapor pressure utilities) |
| FR-PM-004 | Humidity pathway dispatcher | eto.py (+40 lines: dewpoint → RH extremes → RH mean logic) |
| FR-PM-005 | FAO56 validation | tests/fixture/fao56-examples/ + tests/test_eto_penman_monteith_fao56.py (tolerance 0.05 mm/day) |
| FR-PM-006 | xarray adapter + CF metadata | xarray_adapter.py (decorator) + cf_metadata_registry.py (eto_penman_monteith entry) + typed_public_api.py (@overload) |

**Track 2: Index Coverage Expansion (5 FRs) — FULL COVERAGE ✅**

| FR | Requirement | Architectural Support |
|----|-------------|----------------------|
| FR-EDDI-001 | NOAA reference validation | Decision 1 (provenance protocol) + tests/fixture/noaa-eddi-*/ + tests/test_noaa_eddi_reference.py (tolerance 1e-5) |
| FR-EDDI-002 | EDDI xarray adapter | xarray_adapter.py (decorator) + cf_metadata_registry.py (eddi entry) + typed_public_api.py (@overload) |
| FR-EDDI-003 | CLI integration | CLI module (TBD structure) |
| FR-EDDI-004 | PM-ET recommendation docs | docs/index.md + docs/examples/eddi_workflow_example.ipynb |
| FR-PNP-001 | PNP xarray support | Combined with FR-PATTERN-001 (same implementation) |
| FR-SCPDSI-001 | scPDSI stub interface | indices.py (stub with NotImplementedError) + docs/index.md (roadmap note) |

**Track 3: Advanced xarray Capabilities (7 FRs) — FULL COVERAGE ✅**

| FR | Requirement | Architectural Support |
|----|-------------|----------------------|
| FR-PALMER-001 | Manual palmer_xarray wrapper | Decision 2 (palmer.py SECTION 2 +150 lines) |
| FR-PALMER-002 | Multi-output Dataset return | Decision 2 (Pattern C manual wrapper) + xr.Dataset({'pdsi': ..., 'phdi': ..., 'pmdi': ..., 'z_index': ...}) |
| FR-PALMER-003 | AWC spatial parameter handling | palmer_xarray validation logic (if 'time' in awc.dims → InvalidArgumentError) |
| FR-PALMER-004 | params_dict JSON serialization | palmer_xarray (json.dumps to Dataset.attrs['params_dict']) |
| FR-PALMER-005 | CF metadata for 4 Palmer variables | Decision 5 (cf_metadata_registry.py: pdsi, phdi, pmdi, z_index entries) |
| FR-PALMER-006 | typed_public_api @overload | typed_public_api.py (@overload with numpy→tuple vs xarray→Dataset signatures) |
| FR-PALMER-007 | NumPy vs xarray equivalence tests | tests/test_palmer_equivalence.py (tolerance 1e-8, comprehensive scenarios) |

**Cross-Cutting Coverage Verification:**
✅ All 30 FRs have specific architectural components (files, patterns, decisions)
✅ No orphaned FRs (every requirement maps to implementation location)
✅ Cross-epic dependencies documented (Track 0 → Track 3, Track 1 → Tracks 2 & 3)

---

#### Non-Functional Requirements Coverage (8 NFRs in v2.4.0)

**Pattern Compliance & Refactoring Safety — FULL COVERAGE ✅**

| NFR | Requirement | Architectural Support |
|-----|-------------|----------------------|
| NFR-PATTERN-EQUIV | Numerical equivalence during refactoring (1e-8 tolerance) | Numerical precision patterns (test_*_equivalence.py test suite) + np.testing.assert_allclose(atol=1e-8) standard |
| NFR-PATTERN-COVERAGE | 100% pattern compliance (42 checkpoints) | Implementation patterns (6 pattern categories) + test organization (equivalence tests per index) |
| NFR-PATTERN-MAINT | 30% reduction in time-to-fix | Consistency rules (naming, structure, format patterns) prevent agent conflicts |

**Performance Targets — FULL COVERAGE ✅**

| NFR | Requirement | Architectural Support |
|-----|-------------|----------------------|
| NFR-PM-PERF | PM-ET FAO56 accuracy (±0.05 mm/day, ±0.01 kPa) | Numerical precision patterns (tolerance guidelines) + tests/test_eto_penman_monteith_fao56.py validation |
| NFR-PALMER-SEQ | Sequential time constraint documented | Project structure (docs/index.md + code comments: Dask time=-1 chunking rule) |
| NFR-PALMER-PERF | Palmer xarray ≥80% speed | tests/test_benchmark_palmer.py (Track 0/1 BLOCKER) baseline + regression tracking |

**Reliability & Validation — FULL COVERAGE ✅**

| NFR | Requirement | Architectural Support |
|-----|-------------|----------------------|
| NFR-MULTI-OUT | Multi-output adapter stability | Decision 2 (Pattern C manual wrapper) + xarray Issue #1815 workaround documented + FR-PALMER-007 comprehensive test scenarios |
| NFR-EDDI-VAL | EDDI NOAA reference tolerance (1e-5) | Decision 1 (provenance.json validation_tolerance field) + numerical precision patterns (context-specific tolerance table) |

**Cross-Cutting NFR Coverage:**
✅ All NFRs have measurement mechanisms (test suites, benchmarks, documentation)
✅ Quality gates defined (NFR-PALMER-PERF requires baseline BEFORE Track 3)
✅ Tolerance hierarchy documented (1e-8 for equivalence, 1e-5 for EDDI, 0.05 for FAO56)

---

### Implementation Readiness Validation ✅

#### Decision Completeness Assessment

**All Critical Decisions Documented with Versions:** ✅
- ✅ Decision 1 (NOAA Provenance): JSON structure with all required fields specified
- ✅ Decision 2 (Palmer Module Org): palmer.py structure with line counts + extraction threshold (1,400 lines)
- ✅ Decision 3 (Property-Based Tests): Effort budget (50-60 hours/index) + example property sets documented
- ✅ Decision 4 (Exception Migration): Per-module strategy + release notes template + migration examples
- ✅ Decision 5 (CF Metadata Registry): Module location + TypedDict structure + Winston's coupling threshold

**Implementation Patterns Comprehensive Enough:** ✅
- ✅ Naming patterns: Python modules, functions, variables, tests, fixtures (24 examples total)
- ✅ Structure patterns: Project organization, module boundaries, test organization (immutable rules)
- ✅ Format patterns: NetCDF/xarray conventions, JSON provenance, CF metadata application
- ✅ Communication patterns: structlog events, xarray adapter lifecycle (detect → convert → compute → metadata → return)
- ✅ Process patterns: Error handling, numerical precision, type annotations

**Consistency Rules Clear and Enforceable:** ✅
- ✅ Automated enforcement: ruff check, mypy --strict, pytest (tolerance tests)
- ✅ Manual enforcement: CF metadata registry usage, logging pattern consistency, module boundary respect
- ✅ Violation process: Discovery → Documentation → Resolution → Pattern Update (or fix violation)

**Examples Provided for All Major Patterns:** ✅
- ✅ Good examples: Adding new index, NOAA reference validation test (2 comprehensive examples)
- ✅ Anti-patterns: Hard-coded CF metadata, mixed logging patterns, incorrect test tolerance (3 examples with before/after)
- ✅ Pattern examples: Computation function, xarray adapter (decorator + manual), @overload dispatcher, property-based test

---

#### Structure Completeness Assessment

**Project Structure Complete and Specific:** ✅
- ✅ Complete directory tree with 60+ file paths specified
- ✅ New files identified: cf_metadata_registry.py, test_benchmark_palmer.py, 12+ test files
- ✅ Modified files identified: palmer.py (+150 lines), eto.py (+320 lines), typed_public_api.py (+190 lines)
- ✅ Line count projections: palmer.py (912 → 1,060), typed_public_api.py (210 → 350-400)

**All Files and Directories Defined:** ✅
- ✅ Source modules: 9 files (1 new: cf_metadata_registry.py)
- ✅ Test files: 26+ files (12+ new in v2.4.0)
- ✅ Fixture directories: 4 new (noaa-eddi-1/3/6month, fao56-examples)
- ✅ Documentation files: 5 new (examples/*.ipynb, testing/*.md)

**Integration Points Clearly Specified:** ✅
- ✅ Internal: NumPy Core ↔ xarray Adapter (data flow diagram), typed_public_api ↔ Modules (dispatcher pattern), CF_METADATA ↔ Adapters (lookup pattern)
- ✅ External: NOAA PSL (offline download + provenance), FAO56 (embedded data), PyPI (distribution pipeline)

**Component Boundaries Well-Defined:** ✅
- ✅ Computation Engine: "NEVER import xarray" + return type contracts (ndarray or tuple[ndarray, ...])
- ✅ xarray Adapter Layer: Lifecycle contract (5 steps) + single-output vs multi-output rules
- ✅ Type Safety Layer: "@overload + dispatcher only, NO computation" + lazy import pattern
- ✅ CF Metadata Management: "Read-only, no mutations" + coupling threshold (3+ modules → extract helper)

---

#### Pattern Completeness Assessment

**All Potential Conflict Points Addressed:** ✅
- ✅ Naming conflicts: 24 patterns (modules, functions, variables, tests, CF keys, events)
- ✅ Structural conflicts: Module boundaries (4 rules), test organization (5 types), fixture structure (provenance protocol)
- ✅ Format conflicts: NetCDF conventions (dimension ordering), JSON structure (provenance), CF metadata (registry usage)
- ✅ Communication conflicts: structlog events (naming, lifecycle), xarray adapters (5-step lifecycle)
- ✅ Process conflicts: Error handling (structured exceptions), numerical precision (tolerance hierarchy), type annotations (Python 3.10+ syntax)

**Naming Conventions Comprehensive:** ✅
- ✅ Python: Modules (snake_case), functions (verb_first), variables (domain_conventions), tests (test_<module>_<aspect>)
- ✅ Climate: Established conventions (precip, temp, pet, awc, lat, lon)
- ✅ xarray: Dimension ordering (('time', 'lat', 'lon')), CF key naming
- ✅ Fixtures: Hyphen-separated (<source>-<index>-<variant>), provenance.json

**Communication Patterns Fully Specified:** ✅
- ✅ structlog: Event naming (lowercase, past tense), lifecycle (calculation_started → calculation_completed), migration (NEVER mix patterns)
- ✅ xarray adapters: 5-step lifecycle documented with examples (detect → convert → compute → metadata → return)
- ✅ Dispatchers: isinstance() pattern + lazy imports + @overload signatures

**Process Patterns Complete:** ✅
- ✅ Error handling: Structured exception raising pattern + keyword context + docstring documentation
- ✅ Numerical precision: Tolerance hierarchy (4 contexts: 1e-8, 1e-5, 0.05, 0.01) + np.testing.assert_allclose standard
- ✅ Type annotations: Python 3.10+ syntax (| not Union), @overload pattern, TYPE_CHECKING imports

---

### Gap Analysis Results

**Critical Gaps:** NONE ✅

All implementation-blocking elements are addressed:
- ✅ Decision 1 (NOAA Provenance) prevents FR-TEST-004 blocker
- ✅ Decision 2 (Palmer Module Org) unblocks Track 3 implementation
- ✅ Decision 5 (CF Metadata Registry) provides central metadata source
- ✅ tests/test_benchmark_palmer.py identified as Track 0/1 BLOCKER (NFR-PALMER-PERF baseline)

**Important Gaps:** 1 IDENTIFIED ⚠️

**Gap I-1: Property-Based Test Curriculum**
- **Severity**: Important (not blocking, but high value)
- **Description**: Decision 3 establishes comprehensive property-based testing strategy with 50-60 hour/index budget, but no curriculum exists for property discovery workshops
- **Impact**: Without structured approach to property discovery, implementation teams may miss domain-specific invariants or waste time on low-value properties
- **Recommendation**: Create `docs/testing/property-discovery-curriculum.md` during Track 0 (after SPI property tests complete, per Amelia's note)
- **Resolution Strategy**:
  1. Document actual property discovery process for SPI (first index)
  2. Extract reusable patterns (boundedness, NaN propagation, monotonicity, etc.)
  3. Create template for index-specific property brainstorming
  4. Include failure investigation workflow (hypothesis shrinking analysis)
- **Owner**: Test architecture lead (Track 0 deliverable)

**Nice-to-Have Gaps:** 2 IDENTIFIED 💡

**Gap N-1: Dispatcher Pattern Extraction Monitoring**
- **Severity**: Nice-to-have (quality-of-life improvement)
- **Description**: Decision 2 notes typed_public_api.py grows from 210 → 350-400 lines in v2.4.0, with extraction threshold at 300 lines, but no monitoring mechanism exists
- **Impact**: Without monitoring, module could exceed threshold without triggering extraction discussion
- **Recommendation**: Add `# TODO: Extract dispatcher pattern at 300 lines (currently: <count>)` comment to typed_public_api.py header
- **Resolution Strategy**: Manual code review during each FR that adds @overload signatures
- **Priority**: Low (code smell monitoring, not functional issue)

**Gap N-2: CF Metadata Coupling Monitoring**
- **Severity**: Nice-to-have (architectural health monitoring)
- **Description**: Decision 5 (Winston's note) identifies coupling threshold: if 3+ modules duplicate metadata manipulation logic → extract apply_cf_metadata() helper, but no monitoring mechanism exists
- **Impact**: Without monitoring, coupling could grow without triggering helper extraction discussion
- **Recommendation**: Add `# NOTE: If 3+ modules duplicate this pattern, extract apply_cf_metadata() helper` comment to xarray_adapter.py and palmer.py
- **Resolution Strategy**: Manual code review during Track 2-3 xarray adapter work
- **Priority**: Low (architectural health, not functional issue)

---

### Validation Issues Addressed

**Critical Issues:** NONE ✅

No implementation-blocking issues found during validation.

**Important Issues:** NONE ✅

Gap I-1 (Property-Based Test Curriculum) is a nice-to-have enhancement, not an important issue — Decision 3 already provides sufficient guidance (50-60 hour budget, example property sets, Amelia's "SPI-first then document" approach).

**Minor Issues:** NONE ✅

Gaps N-1 and N-2 are monitoring enhancements, not issues requiring resolution before implementation begins.

---

### Architecture Completeness Checklist

**✅ Requirements Analysis**

- [x] Project context thoroughly analyzed (7 input documents, 60 existing FRs + 30 new FRs)
- [x] Scale and complexity assessed (Medium-High: physics-based algorithms, multi-output xarray, NOAA validation, sequential state tracking)
- [x] Technical constraints identified (Python >=3.10,<3.14, xarray Issue #1815, sequential Palmer constraint, numerical precision requirements)
- [x] Cross-cutting concerns mapped (CF metadata compliance, type safety, performance/scalability, multi-layered testing, documentation, research integration)

**✅ Architectural Decisions**

- [x] Critical decisions documented with versions (5 decisions: NOAA Provenance, Palmer Module Org, Property-Based Tests, Exception Migration, CF Metadata Registry)
- [x] Technology stack fully specified (Python 3.10+, hatchling, uv, ruff, mypy, pytest, structlog, scipy, xarray, dask, numpy, cftime)
- [x] Integration patterns defined (NumPy Core ↔ xarray Adapter, typed_public_api ↔ Modules, CF_METADATA ↔ Adapters)
- [x] Performance considerations addressed (NFR-PALMER-PERF: tests/test_benchmark_palmer.py baseline, NFR-PALMER-SEQ: time=-1 chunking documented)

**✅ Implementation Patterns**

- [x] Naming conventions established (24 patterns: modules, functions, variables, tests, fixtures, CF keys, events)
- [x] Structure patterns defined (src-layout immutable, module boundaries with NEVER rules, test type separation)
- [x] Communication patterns specified (structlog lifecycle events, xarray adapter 5-step lifecycle)
- [x] Process patterns documented (structured exceptions, numerical precision tolerance hierarchy, type annotation standards)

**✅ Project Structure**

- [x] Complete directory structure defined (60+ file paths, line count projections, new vs modified files identified)
- [x] Component boundaries established (4 boundaries: Computation Engine, xarray Adapter Layer, Type Safety Layer, CF Metadata Management)
- [x] Integration points mapped (Internal: 3 integration patterns, External: NOAA PSL, FAO56, PyPI)
- [x] Requirements to structure mapping complete (30 FRs → specific files/directories, cross-track dependencies documented)

---

### Architecture Readiness Assessment

**Overall Status:** ✅ **READY FOR IMPLEMENTATION**

**Confidence Level:** **HIGH**

**Rationale:**
1. **Coherence Validation**: All decisions, patterns, and structure work together without conflicts (0 contradictions detected)
2. **Requirements Coverage**: All 30 FRs and 8 NFRs have architectural support (100% coverage)
3. **Implementation Readiness**: AI agents have sufficient patterns, examples, and boundaries to implement consistently
4. **Gap Analysis**: 0 critical gaps, 1 important gap (nice-to-have curriculum), 2 nice-to-have gaps (monitoring enhancements)
5. **Validation Results**: 0 critical issues, 0 important issues, 0 minor issues

---

### Key Strengths

**1. Comprehensive Decision Documentation**
- All 5 architectural decisions include rationale, Party Mode enhancements, implementation impact, and risk mitigation
- Decision impact analysis maps dependencies (Decision 2 ↔ Decision 4: Palmer structlog blocks xarray wrapper)
- Version information specified where applicable (xarray>=2025.6.1, scipy>=1.15.3, etc.)

**2. Pattern Precision Prevents Agent Conflicts**
- 24 naming patterns with ✅ CORRECT / ❌ WRONG examples prevent "valid but inconsistent" implementations
- Module boundaries with "NEVER" rules (NEVER add xarray to indices.py) provide clear guardrails
- Anti-pattern examples (3 comprehensive before/after scenarios) teach by showing what NOT to do

**3. Requirements Traceability**
- Every FR maps to specific files, line counts, and architectural components
- Cross-track dependencies explicitly documented (Track 0 → Track 3, Track 1 → Tracks 2 & 3)
- Test coverage requirements tied to NFRs (NFR-PATTERN-EQUIV → test_*_equivalence.py, NFR-EDDI-VAL → test_noaa_eddi_reference.py)

**4. Scientific Rigor Integration**
- Numerical precision tolerance hierarchy (1e-8, 1e-5, 0.05, 0.01) with context-specific rationale
- NOAA provenance protocol (Decision 1) aligns with climate science reproducibility standards
- FAO56 validation approach matches published guidance (±0.05 mm/day for examples)

**5. Pragmatic Complexity Management**
- Decision 2 (Palmer Module Org): Accepts bimodal module (1,060 lines) with extraction threshold monitoring (1,400 lines)
- Decision 3 (Property-Based Tests): Realistic effort budget (50-60 hours/index) with "SPI-first then adjust" approach
- Decision 4 (Exception Migration): Incremental per-module strategy prevents big-bang refactoring risk

---

### Areas for Future Enhancement

**Post-v2.4.0 Considerations:**

1. **Dispatcher Pattern Extraction** (if typed_public_api.py exceeds 300 lines)
   - **Trigger**: Module size monitoring during v2.4.0 implementation
   - **Approach**: Extract decorator factory for numpy/xarray routing
   - **Benefit**: Reduce duplication, simplify adding new indices

2. **CF Metadata Helper Function** (if 3+ modules duplicate apply logic)
   - **Trigger**: Winston's coupling threshold during Track 2-3
   - **Approach**: Create apply_cf_metadata(da, variable_name, override_attrs=None) helper
   - **Benefit**: Centralize metadata manipulation, reduce coupling

3. **xarray Issue #1815 Resolution Monitoring** (Pattern C workaround)
   - **Trigger**: xarray releases native multi-output + dask='parallelized' support
   - **Approach**: Refactor palmer_xarray to use decorator pattern (align with other adapters)
   - **Benefit**: Remove manual wrapper workaround, simplify architecture

4. **Property-Based Test Curriculum** (Gap I-1)
   - **Trigger**: After SPI property tests complete (Track 0)
   - **Approach**: Document property discovery process, extract reusable patterns
   - **Benefit**: Accelerate property-based testing for future indices

**None of these enhancements block v2.4.0 implementation** — all are post-MVP refinements based on lessons learned or upstream dependency changes.

---

### Implementation Handoff

**AI Agent Guidelines:**

1. **Read First, Implement Second**
   - ✅ Read `_bmad-output/planning-artifacts/architecture.md` (this document) BEFORE any implementation work
   - ✅ Read `_bmad-output/planning-artifacts/prd.md` for FR/NFR context
   - ✅ Read `_bmad-output/project-context.md` for development rules (Python style, tooling, patterns)

2. **Follow Architectural Decisions Exactly**
   - ✅ Use Decision 1 provenance.json structure (including fixture_version field)
   - ✅ Follow Decision 2 palmer.py organization (SECTION 1 + SECTION 2 markers)
   - ✅ Apply Decision 3 property-based test strategy (50-60 hour budget, comprehensive coverage)
   - ✅ Execute Decision 4 exception migration (per-module incremental, during other pattern work)
   - ✅ Use Decision 5 CF_METADATA registry (NEVER hard-code metadata)

3. **Use Implementation Patterns Consistently**
   - ✅ Naming: Follow 24 documented patterns (modules, functions, variables, tests, etc.)
   - ✅ Structure: Respect module boundaries ("NEVER add xarray to indices.py")
   - ✅ Format: Use CF conventions (dimension ordering: ('time', 'lat', 'lon'))
   - ✅ Communication: structlog events (lowercase, past tense), xarray adapter lifecycle (5 steps)
   - ✅ Process: Structured exceptions (keyword context), numerical precision (tolerance hierarchy), type annotations (Python 3.10+ syntax)

4. **Respect Project Structure and Boundaries**
   - ✅ src-layout is immutable (DO NOT add files outside src/climate_indices/)
   - ✅ Test organization by type (unit, equivalence, property-based, benchmark, reference)
   - ✅ Fixture structure with provenance (<source>-<index>-<variant>/provenance.json)
   - ✅ Boundary rules (Computation Engine, xarray Adapter, Type Safety, CF Metadata)

5. **Refer to This Document for All Architectural Questions**
   - ✅ Pattern conflicts? Check "Implementation Patterns & Consistency Rules"
   - ✅ File location questions? Check "Project Structure & Boundaries"
   - ✅ Integration uncertainty? Check "Integration Points"
   - ✅ Numerical precision? Check "Process Patterns → Numerical Precision Patterns"

**First Implementation Priority:**

**BLOCKER Resolution (Track 0 or Track 1):**
1. **Create tests/test_benchmark_palmer.py** — Baseline measurement for NFR-PALMER-PERF
   - **Why FIRST**: Track 3 (Palmer xarray) requires ≥80% speed validation, but no baseline exists
   - **Approach**: Measure current multiprocessing CLI performance (wall-clock time for 360×180 grid, 240 months)
   - **Output**: Baseline measurement + CI integration with performance threshold alerts
   - **Owner**: Development team (must complete BEFORE Track 3 begins)

**Track 0 Parallel Start:**
2. **FR-PATTERN-001 to FR-PATTERN-006** — PNP, PCI, ETo xarray adapters can run in parallel with Palmer structlog
3. **FR-PATTERN-007** — Palmer structlog migration (CRITICAL PATH for Track 3)

**Track Execution Order:**
- **Parallel**: Track 0 + Track 1 (independent)
- **Sequential**: Track 0 (Palmer structlog) → Track 3 (Palmer xarray)
- **Parallel**: Track 2 + Track 3 (after Track 0 + Track 1 complete)

---

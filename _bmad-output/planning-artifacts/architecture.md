---
stepsCompleted: [1, 2, 4, 5, 6, 7, 8]
lastStep: 8
status: 'complete'
version: '2.4.0'
priorVersion: '1.1'
completedAt: '2026-02-05'
revisedAt: '2026-02-16'
revisionReason: 'Incremental update for PRD v2.4.0 - Adding 30 new FRs across 4 tracks (PM-ET, Palmer multi-output, EDDI validation, canonical pattern completion)'
inputDocuments:
  - '_bmad-output/planning-artifacts/prd.md (v2.4.0)'
  - '_bmad-output/planning-artifacts/architecture.md (v1.1 - base)'
  - '_bmad-output/planning-artifacts/research/technical-penman-monteith.md'
  - '_bmad-output/planning-artifacts/research/technical-palmer-modernization.md'
  - '_bmad-output/planning-artifacts/research/technical-eddi-validation.md'
  - '_bmad-output/project-context.md'
  - '_bmad-output/planning-artifacts/implementation-readiness-report-2026-02-16.md'
  - 'docs/floating_point_best_practices.md'
  - 'docs/test_fixture_management.md'
  - 'docs/case-studies/eddi-bmad-retrospective.md'
workflowType: 'architecture'
project_name: 'climate_indices'
user_name: 'James'
date: '2026-02-16'
---

# Architecture Decision Document — climate_indices v2.4.0

_Extends v1.1 foundation (xarray Integration + structlog Modernization) with PM-ET, Palmer multi-output xarray, EDDI validation, and canonical pattern completion across all indices._

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements (90 total = 60 from v1.1 + 30 new in v2.4.0):**

_Core v1.1 Scope (60 FRs):_
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

_New v2.4.0 Scope (30 FRs across 4 tracks):_

**Track 0: Canonical Pattern Completion (12 FRs)** — Apply v1.1 patterns to ALL remaining indices
- FR-PATTERN-001–002: PNP & PCI xarray adapters + CF metadata
- FR-PATTERN-003–006: PNP & PCI + ETo methods typed_public_api `@overload` signatures
- FR-PATTERN-007: Palmer structlog migration (stdlib → structlog lifecycle events)
- FR-PATTERN-008: ETo Thornthwaite lifecycle completion (bind + events)
- FR-PATTERN-009: Structured exceptions for all legacy functions (eliminate bare `ValueError`)
- FR-PATTERN-010–011: PNP & PCI property-based tests (boundedness, shape, NaN handling)
- FR-PATTERN-012: Expanded SPEI + Palmer property tests (PHDI, PMDI, Z-Index bounds)

**Track 1: Penman-Monteith FAO56 PET (6 FRs)** — Physics-based evapotranspiration
- FR-PM-001: Core Penman-Monteith calculation (FAO56 Eq. 6)
- FR-PM-002: Atmospheric parameter helpers (pressure, psychrometric constant — Eq. 7-8)
- FR-PM-003: Vapor pressure helpers (SVP, mean SVP, slope — Eq. 11-13)
- FR-PM-004: Humidity pathway dispatcher (dewpoint → RH extremes → RH mean — Eq. 14-19)
- FR-PM-005: FAO56 worked example validation (Bangkok + Uccle, ±0.05 mm/day)
- FR-PM-006: PM-ET xarray adapter (CF metadata, Dask compatibility)

**Track 2: EDDI/PNP/scPDSI Coverage (5 FRs)** — Index expansion + validation
- FR-EDDI-001: NOAA reference dataset validation (1e-5 tolerance, provenance tracking)
- FR-EDDI-002–004: EDDI xarray adapter, CLI integration, PM-ET recommendation
- FR-PNP-001: PNP xarray adapter (already covered by FR-PATTERN-001)
- FR-SCPDSI-001: scPDSI stub with full signature + NotImplementedError (Wells et al. 2004 reference)

**Track 3: Palmer Multi-Output xarray (7 FRs)** — Advanced xarray for 4-variable return
- FR-PALMER-001: Manual wrapper pattern (NOT decorator — multi-output + params_dict workaround)
- FR-PALMER-002: Dataset return with 4 variables (pdsi, phdi, pmdi, z_index)
- FR-PALMER-003: AWC spatial parameter handling (scalar | DataArray, no time dim validation)
- FR-PALMER-004: params_dict JSON serialization (dual access: JSON string + individual attrs)
- FR-PALMER-005: Palmer CF metadata registry (per-variable long_name, units, references)
- FR-PALMER-006: typed_public_api `@overload` (numpy → tuple, xarray → Dataset)
- FR-PALMER-007: NumPy vs xarray equivalence tests (1e-8 tolerance)

**Non-Functional Requirements (31 total = 23 from v1.1 + 8 new in v2.4.0):**

_Core v1.1 NFRs (23):_
- Performance: <5% overhead, >70% Dask scaling, 50GB memory efficiency, <500ms import
- Reliability: numerical reproducibility (1e-8), graceful degradation, version stability
- Compatibility: Python 3.9–3.13, wide NumPy/SciPy/xarray version range, backward compat guarantee
- Integration: xarray ecosystem (Dask, zarr, cf_xarray), CF v1.10 compliance, structlog JSON format
- Maintainability: mypy --strict, >85% coverage, 100% docstring coverage, ruff/bandit clean, no CVEs

_New/Modified v2.4.0 NFRs (8):_
- **NFR-PATTERN-EQUIV** (Track 0): Numerical equivalence during refactoring (1e-8, zero algorithmic drift)
  - **Decision Traceability:** Enforces Decision 1 (xarray adapter pattern) equivalence guarantee
- **NFR-PATTERN-COVERAGE** (Track 0): 100% pattern compliance dashboard (7/7 indices, 6 patterns)
  - **Decision Traceability:** Validates Decisions 2-7 applied universally
- **NFR-PATTERN-MAINT** (Track 0): Consistency reduces onboarding (2 weeks → 3 days) + 30% bug reduction
  - **Decision Traceability:** Systematic application of Patterns 1-8 improves maintainability
- **NFR-PM-PERF** (Track 1): PM-ET numerical precision (FAO56 examples ±0.05 mm/day, helpers ±0.01 kPa)
  - **Decision Traceability:** Enforces Decision 8 (PM-ET module placement) + Decision 9 (humidity dispatcher) accuracy
- **NFR-PALMER-SEQ** (Track 3): Palmer respects sequential time dependency (chunk spatially, NOT temporally)
  - **Decision Traceability:** Validates Decision 10 (manual wrapper) + Decision 11 (Dataset return) performance
- **NFR-PALMER-PERF** (Track 3): xarray path ≥80% speed of multiprocessing baseline
  - **Decision Traceability:** Measures Decision 10 (Pattern C manual wrapper) overhead vs decorator
- **NFR-MULTI-OUT** (Track 3): Stack/unpack workaround for xarray Issue #1815 (documented pattern)
  - **Decision Traceability:** Implements Decision 10 (manual wrapper) multi-output strategy
- **NFR-EDDI-VAL** (Track 2): EDDI 1e-5 tolerance for non-parametric ranking (documented rationale)
  - **Decision Traceability:** Extends Decision 12 (reference dataset testing) with EDDI-specific tolerance

**Scale & Complexity:**
- Primary domain: Scientific Python library (API-first, no UI)
- Complexity level: **Medium-High** (elevated from Medium due to v2.4.0 scope)
  - Physics-based algorithms (PM FAO56 equations 1-19)
  - Multi-output xarray patterns (Palmer: 4 variables + params dict)
  - NOAA reference validation requirements
  - Sequential state tracking (Palmer water balance)
- Estimated architectural components: **9 total** (up from 5 in v1.1)
  1. xarray adapter layer (v1.1)
  2. Logging layer (v1.1)
  3. Metadata engine (v1.1)
  4. Exception hierarchy (v1.1)
  5. Test infrastructure (v1.1)
  6. **PM-ET module with helper functions** (v2.4.0 — Track 1)
  7. **Palmer xarray manual wrapper** (v2.4.0 — Track 3)
  8. **Reference dataset validation framework** (v2.4.0 — Track 2)
  9. **Pattern Migration Framework** (v2.4.0 — Track 0, elevated to architectural component)

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

_Core v1.1 Concerns (retained):_
1. **Type dispatch**: NumPy/xarray routing affects all public API functions
2. **structlog**: Affects entire call stack (indices → compute → eto/palmer)
3. **Metadata preservation**: New concern for xarray path (CF attributes, coordinates, chunking)
4. **Error handling**: Custom exception hierarchy spans compute + indices
5. **Numerical equivalence testing**: New test infrastructure for dual-path validation

_New v2.4.0 Concerns:_
6. **Multi-output xarray patterns** (Track 3): Palmer Dataset return requires stack/unpack workaround for xarray Issue #1815, affects adapter architecture
7. **Physics-based algorithm fidelity** (Track 1): FAO56 equation precision (exact constants, Kelvin conversions, non-linear SVP averaging) impacts PM-ET helper design
8. **Reference dataset validation** (Track 2): NOAA EDDI compliance requires SHA256 provenance tracking + algorithm version metadata + documented tolerance rationale (1e-5 vs 1e-8)
9. **Pattern compliance consistency** (Track 0): Migration protocol requires before/after equivalence capture + failure revert protocol + zero algorithmic drift guarantee
10. **Type-safe multi-output boundaries** (Track 3): `@overload` dispatch must distinguish numpy tuple vs xarray Dataset returns + manual Palmer dispatch (not decorator-compatible)

### Track Dependencies (Explicit)

**Dependency Graph:**
```
Track 0 (Pattern Completion) ──┐
                                ├─→ Track 2 (EDDI/PNP/scPDSI) ──┐
Track 1 (PM-ET Foundation) ─────┤                                 ├─→ v2.4.0 Complete
                                ├─→ Track 3 (Palmer xarray) ─────┘
                                │
Track 0 (Palmer structlog) ─────┘
```

**Critical Path:**
1. **Track 0 ∥ Track 1** (parallel):
   - Track 0: PNP/PCI xarray, typed_public_api entries, property tests (independent)
   - Track 1: PM-ET implementation (independent)
   - **Partial blocking:** Track 0 Palmer structlog MUST complete before Track 3

2. **Track 2 ∥ Track 3** (parallel after Tracks 0 + 1):
   - Track 2 depends on: Track 1 PM-ET (for EDDI recommendation) + Track 0 PNP xarray
   - Track 3 depends on: Track 0 Palmer structlog + Track 1 (indirectly via testing patterns)

**Rationale for Parallelization:**
- Track 0 (PNP/PCI/ETo) + Track 1 (PM-ET) touch different modules → safe to parallelize
- Track 2 (EDDI/PNP/scPDSI) + Track 3 (Palmer xarray) affect different indices → safe to parallelize after foundation
- **Risk mitigation:** Palmer structlog (Track 0) completes early to unblock Track 3 Palmer xarray work

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

---

## New Architectural Decisions for v2.4.0

_The following decisions extend the v1.1 foundation to support PM-ET (Track 1), Palmer multi-output xarray (Track 3), EDDI validation (Track 2), and canonical pattern completion (Track 0)._

### Decision 8: PM-ET Module Placement → Extend `eto.py`
- Add private helpers to `src/climate_indices/eto.py`:
  - `_atm_pressure(altitude)` — FAO56 Eq. 7
  - `_psy_const(pressure)` — FAO56 Eq. 8
  - `_svp_from_t(temp)` — FAO56 Eq. 11 (Magnus formula)
  - `_mean_svp(tmin, tmax)` — FAO56 Eq. 12 (SVP average at extremes)
  - `_slope_svp(temp)` — FAO56 Eq. 13 (slope of SVP curve)
  - `_avp_from_dewpoint(dewpoint)` — FAO56 Eq. 14
  - `_avp_from_rhminmax(tmin, tmax, rh_min, rh_max)` — FAO56 Eq. 17
  - `_avp_from_rhmean(tmin, tmax, rh_mean)` — FAO56 Eq. 19
- Public function: `eto_penman_monteith(tmin, tmax, tmean, wind_2m, net_radiation, latitude, altitude, ...)`
- **Rationale:**
  - Follows existing `eto_thornthwaite`/`eto_hargreaves` pattern in same module
  - Single discoverable location for all PET methods
  - Private helpers enable independent testing and FAO56 equation traceability
- **Research Reference:** `technical-penman-monteith.md` Section 6 recommends this approach

### Decision 9: PM-ET Humidity Dispatcher → Auto-Select with Override
- Auto-detect available humidity inputs using priority order:
  1. `dewpoint_celsius` → `_avp_from_dewpoint()` (most accurate per FAO56)
  2. `rh_max` + `rh_min` → `_avp_from_rhminmax()` (preferred for daily data)
  3. `rh_mean` → `_avp_from_rhmean()` (fallback)
- Accept explicit `actual_vapor_pressure` parameter as override (bypass auto-detection)
- Log selected pathway at DEBUG level: `_logger.debug("humidity_pathway_selected", method="dewpoint")`
- Raise `InvalidArgumentError` if no humidity input provided
- **Rationale:**
  - Mirrors parameter inference decision (Decision 6) — pragmatic ergonomics
  - FAO56 explicitly defines pathway hierarchy
  - User override enables advanced use cases (pre-computed vapor pressure)

### Decision 10: Palmer xarray Adapter → Manual Wrapper (Pattern C)
- **NOT decorator-based** — Multi-output + params_dict incompatible with `@xarray_adapter`
- New dedicated module: `src/climate_indices/palmer_xarray.py` (~150 lines)
- Implementation pattern:
  ```python
  def palmer_xarray(precip_da, pet_da, awc, ...) -> xr.Dataset:
      # Extract .values, infer params, call palmer.pdsi()
      pdsi, phdi, pmdi, z, params = palmer.pdsi(precip.values, pet.values, ...)
      # Stack for apply_ufunc compatibility
      stacked = np.stack([pdsi, phdi, pmdi, z], axis=0)
      # Construct Dataset with per-variable CF metadata
      ds = xr.Dataset({
          "pdsi": (["time"], pdsi, CF_METADATA["pdsi"]),
          "phdi": (["time"], phdi, CF_METADATA["phdi"]),
          # ... etc
      })
      return ds
  ```
- Comment workaround for xarray Issue #1815 (dask="parallelized" + multi-output limitation)
- **Rationale:**
  - Research `technical-palmer-modernization.md` evaluates 3 patterns; Pattern C recommended
  - Decorator cannot handle 4-variable return + params_dict dual serialization
  - Manual wrapper provides full control over Dataset construction
- **Research Reference:** Section 5.3 "Pattern C: Manual Wrapper Functions (Recommended)"

### Decision 11: Palmer Dataset Return → `xr.Dataset` with Per-Variable CF Metadata
- Return type: `xr.Dataset` with 4 variables (NOT tuple):
  - `pdsi` — Palmer Drought Severity Index
  - `phdi` — Palmer Hydrological Drought Index
  - `pmdi` — Palmer Modified Drought Index
  - `z_index` — Palmer Z-Index
- Each variable is `xr.DataArray` with independent CF metadata (from registry)
- params_dict dual access:
  1. JSON string: `ds.attrs["palmer_params"] = json.dumps({"alpha": ..., ...})`
  2. Individual scalar attrs: `ds.attrs["palmer_alpha"]`, `ds.attrs["palmer_beta"]`, etc.
- `@overload` dispatch in `typed_public_api.py`:
  - NumPy path: `pdsi(np.ndarray, ...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict | None]`
  - xarray path: `pdsi(xr.DataArray, ...) -> xr.Dataset`
- **Rationale:**
  - CF-compliant container for NetCDF interchange
  - Per-variable metadata enables downstream visualization/analysis
  - Type-safe dispatch eliminates tuple unpacking errors
  - Dual params_dict access balances structured (JSON) vs direct (scalar attrs) access patterns

### Decision 12: Reference Dataset Testing → Separate Module with Provenance
- New test module: `tests/test_reference_validation.py` (separate from equivalence tests)
- Reference data directory: `tests/data/reference/` with provenance metadata
  - Each dataset includes attrs: `source`, `url`, `download_date`, `subset_description`, `algorithm_version`, `sha256_checksum`
- Tolerance: **1e-5** (looser than equivalence tests)
  - **Rationale:** Non-parametric empirical ranking (EDDI) has different FP accumulation than parametric distribution fitting (SPI/SPEI)
  - Documented in test docstring
- Provenance tracking protocol:
  ```python
  # Example EDDI reference dataset attrs
  attrs = {
      "source": "NOAA PSL EDDI CONUS Archive",
      "url": "https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/data/2020/",
      "download_date": "2026-02-10",
      "subset_description": "Colorado River Basin, 2020-01 to 2020-12, 6-month EDDI",
      "algorithm_version": "EDDI v1.2 (Hobbins et al. 2016)",
      "sha256_checksum": "a3f5d8c2e1b4..."
  }
  ```
- **Rationale:**
  - Already partially decided in Architecture v1.1 Pattern 8 — this formalizes it for v2.4.0
  - Separate module avoids conflating reference validation (external ground truth) with equivalence testing (internal consistency)
  - Provenance enables reproducibility and audit trail
  - SHA256 + algorithm version ensure reference data integrity

### Decision 13: Track 0 Pattern Application Strategy → Incremental by Index
- Apply patterns **index-by-index** (not pattern-by-pattern)
- Complete all 6 patterns for one index before moving to next:
  1. xarray adapter
  2. typed_public_api `@overload`
  3. CF metadata registry entry
  4. structlog lifecycle events
  5. Structured exceptions
  6. Property-based tests
- Execution order:
  1. **PNP** (`percentage_of_normal`) — simplest, validates migration protocol
  2. **PCI** — daily-data edge cases, validates pattern robustness
  3. **ETo completion** (`eto_thornthwaite` lifecycle, `eto_hargreaves` completeness)
  4. **Palmer structlog** — unblocks Track 3
- **Equivalence protocol:** Capture baseline BEFORE pattern application → Apply pattern → Validate `assert_allclose(before, after, atol=1e-8)` → If failure, revert and investigate
- **Rationale:**
  - Each index is independently testable → reduces WIP and integration risk
  - Index-complete milestones provide clear progress signals
  - Equivalence protocol enforces NFR-PATTERN-EQUIV (zero algorithmic drift)
  - Order prioritizes Palmer structlog to unblock Track 3 work

### Decision 14: scPDSI → Stub with Full Signature
- Function signature defined in `src/climate_indices/indices.py`
- Raises `NotImplementedError` with informative message:
  ```python
  raise NotImplementedError(
      "Self-Calibrating PDSI (scPDSI) is not yet implemented. "
      "This feature is planned for a future release. "
      "See Wells et al. (2004) J. Climate 17(12):2335-2351 for methodology."
  )
  ```
- Full docstring with:
  - Methodology overview (self-calibrating climatic characteristic)
  - Reference: Wells, Goddard, Hayes (2004) DOI: `10.1175/1520-0442(2004)017<2335:ASPDSI>2.0.CO;2`
  - Parameter descriptions (same as `pdsi()`)
  - Note on future implementation timeline
- `@overload` signatures for future numpy/xarray dispatch (matches `pdsi` pattern)
- **Rationale:**
  - Establishes public API contract for future implementation
  - Prevents namespace collision if users expect `scpdsi()` function
  - Docstring provides scientific context and references
  - Full signature enables mypy type checking even before implementation
  - Clear error message sets expectations vs silent absence

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

---

## New Implementation Patterns for v2.4.0

_The following patterns complement Patterns 1-8 to support Track 0-3 implementation._

### Pattern 9: PM-ET Helper Function Contract
- **Naming:** All helpers are private with leading underscore (e.g., `_svp_from_t`, `_atm_pressure`)
- **Purity:** Pure mathematical functions — no side effects, no logging, no state
- **Array compatibility:** Accept `np.ndarray`, return same shape (element-wise operations)
- **FAO56 precision:** Use exact FAO56 constants (never approximate):
  - SVP Magnus constants: `0.6108`, `17.27`, `237.3`
  - Psychrometric constant coefficient: `0.000665`
  - Atmospheric pressure exponent: `5.26`
- **Independent testability:** Each helper tested with known reference values from FAO56
- **Equation traceability:** Each helper maps to exactly one FAO56 equation number (documented in docstring)
- **Rule:** Never modify constants for "clarity" — FAO56 values are canonical
- **Example:**
  ```python
  def _svp_from_t(temp_celsius: np.ndarray) -> np.ndarray:
      """Saturation vapor pressure from temperature (FAO56 Eq. 11).

      Uses Magnus formula with exact FAO56 constants.

      Args:
          temp_celsius: Temperature in degrees Celsius

      Returns:
          Saturation vapor pressure in kPa

      Reference:
          Allen et al. (1998) FAO Irrigation & Drainage Paper 56, Eq. 11
      """
      return 0.6108 * np.exp((17.27 * temp_celsius) / (temp_celsius + 237.3))
  ```

### Pattern 10: Humidity Pathway Selection
- **Priority order (per FAO56):**
  1. `dewpoint_celsius` → Eq. 14 (most accurate)
  2. `rh_max` + `rh_min` → Eq. 17 (preferred for daily)
  3. `rh_mean` → Eq. 19 (fallback)
- **Auto-detection logic:** Check parameters in priority order, select first available
- **Explicit override:** If `actual_vapor_pressure` provided, bypass auto-detection
- **Logging requirement:** Log selected pathway at DEBUG level:
  ```python
  _logger.debug("humidity_pathway_selected", method="dewpoint", equation="FAO56-14")
  ```
- **Error handling:** Raise `InvalidArgumentError` if no humidity input provided:
  ```python
  raise InvalidArgumentError(
      "At least one humidity input required: dewpoint_celsius, "
      "(rh_max AND rh_min), rh_mean, or actual_vapor_pressure"
  )
  ```
- **Rule:** Auto-select, but never silently fall back without logging
- **Rule:** Error message must enumerate ALL acceptable input combinations

### Pattern 11: Multi-Output xarray Adapter (Palmer-Specific)
- **NOT decorator-based** — Manual wrapper function for multi-output + params_dict handling
- **Module location:** Dedicated `palmer_xarray.py` (not in `xarray_adapter.py`)
- **Stack/unpack pattern for `apply_ufunc` compatibility:**
  ```python
  # Stack 4 outputs into single array
  stacked = np.stack([pdsi, phdi, pmdi, z_index], axis=0)
  # OR for gridded: axis=-1 to create variable dimension

  # Unpack to Dataset
  ds = xr.Dataset({
      "pdsi": (dims, pdsi_values, CF_METADATA["pdsi"]),
      "phdi": (dims, phdi_values, CF_METADATA["phdi"]),
      "pmdi": (dims, pmdi_values, CF_METADATA["pmdi"]),
      "z_index": (dims, z_values, CF_METADATA["z_index"]),
  })
  ```
- **Per-variable CF metadata:** Apply from registry via `assign_attrs()` or Dataset constructor
- **params_dict dual serialization:**
  ```python
  ds.attrs["palmer_params"] = json.dumps(params_dict)  # JSON string
  ds.attrs["palmer_alpha"] = params_dict["alpha"]       # Direct access
  ds.attrs["palmer_beta"] = params_dict["beta"]         # etc.
  ```
- **Workaround documentation:** Comment references xarray Issue #1815:
  ```python
  # Workaround for xarray Issue #1815: dask='parallelized' with multi-output
  # not supported. Stack outputs, then unpack to Dataset.
  # See: https://github.com/pydata/xarray/issues/1815
  ```
- **Rule:** Comment MUST reference Issue #1815 for future refactoring awareness
- **Rule:** Use registry for CF metadata (never inline attributes)

### Pattern 12: Reference Dataset Management
- **Directory structure:** `tests/data/reference/<index_name>/`
  - Example: `tests/data/reference/eddi/noaa_conus_2020_6mo.nc`
- **Provenance metadata (REQUIRED attrs in NetCDF):**
  ```python
  required_attrs = [
      "source",                # e.g., "NOAA PSL EDDI CONUS Archive"
      "url",                   # Download URL
      "download_date",         # ISO 8601 date
      "subset_description",    # Human-readable subset info
      "algorithm_version",     # Source algorithm version (e.g., "EDDI v1.2")
      "sha256_checksum",       # File integrity verification
  ]
  ```
- **Tolerance documentation in test docstring:**
  ```python
  def test_eddi_noaa_reference():
      """Validate EDDI against NOAA PSL reference data.

      Tolerance: 1e-5 (looser than equivalence tests)
      Rationale: Non-parametric empirical ranking has different floating-point
                 accumulation characteristics than parametric distribution fitting.
      """
  ```
- **Test naming:** `test_<index>_<source>_reference()` (e.g., `test_eddi_noaa_reference()`)
- **Rule:** Never commit reference data without provenance metadata
- **Rule:** Document tolerance rationale in test docstring (not just magic number)
- **Rule:** Use SHA256 to detect reference data corruption

### Pattern 13: Track 0 Equivalence Protocol
- **Baseline capture BEFORE pattern application:**
  ```python
  # Save baseline output before refactoring
  baseline = percentage_of_normal(precip, scale=12)
  np.save("baseline_pnp.npy", baseline)
  ```
- **Apply pattern** (xarray adapter, exception migration, structlog, etc.)
- **Validate equivalence AFTER pattern application:**
  ```python
  # Test equivalence after pattern application
  baseline = np.load("baseline_pnp.npy")
  refactored = percentage_of_normal(precip, scale=12)
  np.testing.assert_allclose(baseline, refactored, atol=1e-8, rtol=1e-8)
  ```
- **Failure protocol:** If equivalence test fails:
  1. Revert pattern application immediately
  2. Investigate root cause (algorithmic change? numerical precision?)
  3. Document finding
  4. Re-attempt with fix
- **Rule:** Zero algorithmic drift allowed — 1e-8 tolerance is non-negotiable
- **Rule:** Equivalence tests run in CI for every Track 0 PR
- **Rule:** Baseline fixtures committed to git for reproducibility

### Pattern 14: Palmer CF Metadata Entries
- **Registry location:** `CF_METADATA` dict in `xarray_adapter.py`
- **Required entries for Palmer indices:**
  ```python
  CF_METADATA = {
      # ... existing SPI, SPEI, PET entries ...

      "pdsi": {
          "long_name": "Palmer Drought Severity Index",
          "units": "",  # Dimensionless
          "references": "Palmer, W.C. (1965). Meteorological Drought. U.S. Weather Bureau Research Paper 45.",
          "standard_name": "",  # No CF standard name for PDSI
      },
      "phdi": {
          "long_name": "Palmer Hydrological Drought Index",
          "units": "",
          "references": "Palmer (1965); Karl (1986) for PHDI interpretation.",
      },
      "pmdi": {
          "long_name": "Palmer Modified Drought Index",
          "units": "",
          "references": "Heddinghaus & Sabol (1991). Climate Prediction Center interpretation.",
      },
      "z_index": {
          "long_name": "Palmer Z-Index",
          "units": "",
          "references": "Palmer (1965) moisture anomaly component.",
      },
  }
  ```
- **Application in Dataset construction:**
  ```python
  ds = xr.Dataset({
      "pdsi": (dims, pdsi_values),
  })
  ds["pdsi"] = ds["pdsi"].assign_attrs(CF_METADATA["pdsi"])
  ```
- **Rule:** Use registry, never inline attributes in wrapper functions
- **Rule:** `units=""` for dimensionless (not missing or `"1"`)
- **Rule:** `standard_name` empty string if no CF convention exists (not omitted)

## Project Structure & Boundaries

### New Files (v1.1 MVP — Implemented)
- `src/climate_indices/xarray_adapter.py` — Decorator, CF registry, parameter inference
- `src/climate_indices/logging_config.py` — structlog dual-output configuration
- `src/climate_indices/exceptions.py` — ClimateIndicesError hierarchy
- `tests/test_xarray_adapter.py` — Equivalence, metadata, inference, Dask tests
- `tests/test_logging.py` — structlog configuration and output tests
- `tests/test_exceptions.py` — Exception hierarchy tests

### New Files (v2.4.0 Additions)

**Track 1 (PM-ET):**
- `tests/test_eto.py` additions — FAO56 Example 17 & 18 validation tests

**Track 2 (EDDI/PNP/scPDSI):**
- `tests/test_reference_validation.py` — NOAA EDDI reference dataset validation (1e-5 tolerance)
- `tests/data/reference/eddi/` — EDDI reference dataset directory
- `tests/data/reference/eddi/noaa_conus_2020_6mo.nc` — NOAA EDDI reference data with provenance

**Track 3 (Palmer xarray):**
- `src/climate_indices/palmer_xarray.py` — Manual multi-output xarray wrapper (~150 lines)
- `tests/test_palmer_xarray.py` — Palmer xarray unit + integration tests (equivalence, Dataset structure, AWC validation)

**Track 0 (Pattern Completion):**
- `tests/test_properties.py` additions — PNP, PCI, expanded SPEI/Palmer property tests

### Modified Files (v1.1 — Implemented)
- `src/climate_indices/__init__.py` — Add public re-exports for xarray-aware API
- `src/climate_indices/compute.py` — Re-parent exceptions, add structlog
- `src/climate_indices/eto.py` — Add structlog
- `src/climate_indices/utils.py` — Add structlog
- `tests/conftest.py` — Add xarray fixtures
- `pyproject.toml` — Add structlog dependency

### Modified Files (v2.4.0 — Beyond v1.1 Changes)

**Track 1 (PM-ET):**
- `src/climate_indices/eto.py` — Add PM-ET helpers + `eto_penman_monteith()` function
  - New private helpers: `_atm_pressure`, `_psy_const`, `_svp_from_t`, `_mean_svp`, `_slope_svp`
  - New private helpers: `_avp_from_dewpoint`, `_avp_from_rhminmax`, `_avp_from_rhmean`
  - Public: `eto_penman_monteith(tmin, tmax, tmean, wind_2m, net_radiation, latitude, altitude, ...)`

**Track 0 (Pattern Completion):**
- `src/climate_indices/palmer.py` — stdlib logging → structlog migration
  - Replace `utils.get_logger()` with `from climate_indices.logging_config import get_logger`
  - Add lifecycle events: `calculation_started`, `calculation_completed`, `calculation_failed`
  - Bind context at entry points
- `src/climate_indices/xarray_adapter.py` — CF metadata for PNP, PCI, EDDI, Palmer (pdsi, phdi, pmdi, z_index)
- `src/climate_indices/typed_public_api.py` — `@overload` signatures for:
  - `percentage_of_normal` (numpy → ndarray, xarray → DataArray)
  - `pci` (numpy → ndarray, xarray → DataArray)
  - `eto_thornthwaite` (numpy → ndarray, xarray → DataArray)
  - `eto_hargreaves` (numpy → ndarray, xarray → DataArray)
  - `pdsi` (numpy → tuple, xarray → Dataset) — CRITICAL: different return types
- `src/climate_indices/indices.py` — Structured exceptions + scPDSI stub
  - Replace `ValueError` in `percentage_of_normal`, `pci` with `InvalidArgumentError`
  - Add `scpdsi()` stub with `NotImplementedError` + full docstring
- `tests/test_eto.py` — FAO56 validation + PM-ET tests
- `tests/test_properties.py` — PNP/PCI/SPEI/Palmer property tests

**Track 3 (Palmer xarray):**
- `src/climate_indices/__init__.py` — Re-export `palmer_xarray` (if public) or keep internal
- `src/climate_indices/typed_public_api.py` — Palmer `@overload` (numpy tuple vs xarray Dataset)

### Unchanged Files (v1.1 + v2.4.0)
- `src/climate_indices/lmoments.py` — L-moments estimation (no changes needed)
- `src/climate_indices/__main__.py` — CLI (EDDI integration tracked separately, not in architectural scope)
- `src/climate_indices/__spi__.py` — SPI CLI (no changes needed)

### Architectural Boundaries

**v1.1 Boundaries (Retained):**
1. **NumPy Core ↔ xarray Adapter**: `xarray_adapter.py` dispatches; `indices.py` stays pure NumPy
2. **Logging Layer**: `logging_config.py` configures; all modules use `structlog.get_logger(__name__)`
3. **Exception Hierarchy**: `exceptions.py` defines all; other modules import and raise

**v2.4.0 New Boundaries:**
4. **NumPy Core ↔ Palmer xarray** (Track 3): Manual wrapper in `palmer_xarray.py` (NOT decorator-based) → `palmer.pdsi()` numpy core
5. **PM-ET Internal Helpers ↔ Public API** (Track 1): Private `_*` helpers in `eto.py` → public `eto_penman_monteith()`
6. **Reference Validation ↔ Equivalence Testing** (Track 2): Separate test modules with different tolerances (1e-5 vs 1e-8)
7. **Pattern Migration ↔ Numerical Equivalence** (Track 0): Baseline capture → pattern application → equivalence validation protocol

### Data Flow (v1.1 xarray path — SPI/SPEI/PET)
```
User → typed_public_api.spi(DataArray)
  → isinstance check → xarray_adapter.spi(DataArray)
  → extract .values, coords, attrs
  → infer params from time coordinate
  → log start → indices.spi(ndarray) → log complete
  → wrap result as DataArray with CF metadata
  → return DataArray
```

### Data Flow (v2.4.0 Palmer xarray path — NEW)
```
User → typed_public_api.pdsi(DataArray, DataArray, ...)
  → isinstance check → palmer_xarray.palmer_xarray(precip_da, pet_da, awc, ...)
  → validate AWC dimensions (no time)
  → extract .values from DataArrays
  → infer params from time coordinate (data_start_year, periodicity)
  → log start → palmer.pdsi(ndarray, ndarray, ...) → log complete
  → receive tuple: (pdsi, phdi, pmdi, z_index, params_dict)
  → construct Dataset with 4 variables
  → apply CF metadata per variable from registry
  → serialize params_dict to JSON + individual attrs
  → return xr.Dataset
```

### Data Flow (v2.4.0 PM-ET path — Track 1)
```
User → typed_public_api.eto_penman_monteith(tmin_da, tmax_da, ...)
  → isinstance check → xarray_adapter.eto_penman_monteith(...)
  → extract .values, coords, attrs
  → log start
  → eto.eto_penman_monteith(ndarray, ndarray, ...)
    → humidity pathway selection (dewpoint / RH extremes / RH mean)
    → _atm_pressure(altitude) → pressure
    → _psy_const(pressure) → gamma
    → _svp_from_t(tmin), _svp_from_t(tmax) → SVP values
    → _mean_svp(tmin, tmax) → es
    → _slope_svp(tmean) → delta
    → humidity helper (_avp_from_*) → ea
    → FAO56 Eq. 6: ETo = (0.408*delta*(Rn-G) + gamma*...)/(delta + gamma*...)
  → log complete
  → wrap result as DataArray with CF metadata
  → return DataArray
```

## Architecture Validation

### Coherence: ✅ PASS (v2.4.0 Update)
**All decisions (1-14), patterns (1-14), and structure are internally consistent with no contradictions.**

**v1.1 → v2.4.0 Compatibility Check:**
- ✅ Decision 8 (PM-ET placement) extends Decision 2 (module structure) → `eto.py` remains single PET module
- ✅ Decision 9 (humidity dispatcher) mirrors Decision 6 (parameter inference) → consistent ergonomics
- ✅ Decision 10 (Palmer manual wrapper) complements Decision 1 (decorator pattern) → acknowledges decorator limitations
- ✅ Decision 11 (Dataset return) extends Decision 4 (metadata engine) → registry pattern reused
- ✅ Decision 12 (reference testing) formalizes Pattern 8 → provenance + SHA256 additions
- ✅ Decision 13 (Track 0 strategy) enforces Decision 1 equivalence guarantee → NFR-PATTERN-EQUIV
- ✅ Decision 14 (scPDSI stub) preserves Decision 5 (exception hierarchy) → NotImplementedError with context

**Cross-Track Coherence:**
- ✅ Track 0 (pattern completion) validates Decisions 1-7 universally applied
- ✅ Track 1 (PM-ET) independent of Tracks 0/2/3 → safe parallelization
- ✅ Track 2 (EDDI/PNP) depends on Track 1 PM-ET → documented in dependency graph
- ✅ Track 3 (Palmer xarray) depends on Track 0 Palmer structlog → critical path identified

### Requirements Coverage: ✅ PASS (100% for v2.4.0)
**v1.1 Coverage (60 FRs, 23 NFRs): 98%** (2 items deferred to Phase 2)

**v2.4.0 Coverage (90 FRs, 31 NFRs): 100%** — All new requirements architecturally supported:

**Track 0 (12 FRs):**
- ✅ FR-PATTERN-001–002: PNP & PCI xarray → Decision 1 (adapter), Pattern 1 (decorator contract)
- ✅ FR-PATTERN-003–006: typed_public_api entries → Decision 3 (overloads), Pattern 3 (type annotations)
- ✅ FR-PATTERN-007: Palmer structlog → Decision 3 (structlog), Pattern 2 (structlog conventions)
- ✅ FR-PATTERN-008: ETo lifecycle → Pattern 2 (lifecycle events)
- ✅ FR-PATTERN-009: Structured exceptions → Decision 5 (exception hierarchy), Pattern 4 (error handling)
- ✅ FR-PATTERN-010–012: Property tests → Pattern 5 (test structure)

**Track 1 (6 FRs):**
- ✅ FR-PM-001: PM-ET core → Decision 8 (module placement)
- ✅ FR-PM-002–003: Helper functions → Pattern 9 (PM-ET helper contract)
- ✅ FR-PM-004: Humidity dispatcher → Decision 9 (auto-select), Pattern 10 (pathway selection)
- ✅ FR-PM-005: FAO56 validation → Pattern 9 (independent testability)
- ✅ FR-PM-006: xarray adapter → Decision 1 (adapter pattern), Pattern 1 (decorator)

**Track 2 (5 FRs):**
- ✅ FR-EDDI-001: NOAA reference → Decision 12 (reference testing), Pattern 12 (provenance)
- ✅ FR-EDDI-002–004: EDDI xarray/CLI → Decision 1 (adapter), Pattern 1 (decorator)
- ✅ FR-SCPDSI-001: scPDSI stub → Decision 14 (stub with signature)

**Track 3 (7 FRs):**
- ✅ FR-PALMER-001: Manual wrapper → Decision 10 (Pattern C)
- ✅ FR-PALMER-002: Dataset return → Decision 11 (multi-output)
- ✅ FR-PALMER-003: AWC spatial → Decision 11 (dimension validation)
- ✅ FR-PALMER-004: params_dict → Decision 11 (dual serialization)
- ✅ FR-PALMER-005: CF metadata → Decision 4 (registry), Pattern 14 (Palmer entries)
- ✅ FR-PALMER-006: typed_public_api → Decision 11 (overload dispatch)
- ✅ FR-PALMER-007: Equivalence tests → Pattern 13 (equivalence protocol)

**NFR Coverage (8 new):**
- ✅ NFR-PATTERN-EQUIV → Decision 13 (Track 0 strategy), Pattern 13 (equivalence protocol)
- ✅ NFR-PATTERN-COVERAGE → Decisions 1-7 universal application
- ✅ NFR-PATTERN-MAINT → Pattern consistency across Patterns 1-14
- ✅ NFR-PM-PERF → Decision 8 (PM-ET placement), Pattern 9 (helper contract)
- ✅ NFR-PALMER-SEQ → Decision 10 (manual wrapper handles sequential state)
- ✅ NFR-PALMER-PERF → Decision 10 (Pattern C performance)
- ✅ NFR-MULTI-OUT → Decision 10 (stack/unpack), Pattern 11 (workaround documentation)
- ✅ NFR-EDDI-VAL → Decision 12 (1e-5 tolerance), Pattern 12 (tolerance documentation)

### Implementation Readiness: ✅ HIGH CONFIDENCE (v2.4.0)
**v1.1 Foundation:**
- Zero changes to proven NumPy core
- Clean adapter boundary with single responsibility
- structlog is purely additive
- Exception re-parenting is backward compatible

**v2.4.0 Additions:**
- ✅ **Track 1 (PM-ET):** Clear helper function boundaries, FAO56 validation ensures correctness
- ✅ **Track 3 (Palmer xarray):** Manual wrapper pattern validated by research, stack/unpack workaround documented
- ✅ **Track 2 (EDDI/PNP):** Reference validation framework extends Pattern 8, provenance tracking formal
- ✅ **Track 0 (Pattern Completion):** Equivalence protocol protects against regression, index-by-index reduces risk
- ✅ **Parallelization Safety:** Track 0 ∥ Track 1 validated as independent, dependency graph explicit
- ✅ **Type Safety:** mypy --strict enforced across all new `@overload` signatures (PNP, PCI, ETo, Palmer)
- ✅ **Research Alignment:** All Track 1-3 decisions grounded in technical research findings

### Implementation Priority Order

**v1.1 MVP (Completed):**
1. `exceptions.py` — Foundation (other modules import from here)
2. `logging_config.py` — Cross-cutting (needed by all modules)
3. Add structlog to `compute.py`, `eto.py`, `utils.py`
4. `xarray_adapter.py` — Core adapter with SPI first, then SPEI, PET
5. Update `__init__.py` with public API re-exports
6. Update `pyproject.toml` with structlog dependency
7. Tests: `test_exceptions.py`, `test_logging.py`, `test_xarray_adapter.py`
8. Update `conftest.py` with xarray fixtures

**v2.4.0 Track-Based Priority (4 parallel tracks):**

**Phase 1: Foundation (Parallel Tracks 0 + 1)**
- **Track 0 Start (Week 1-2):**
  1. PNP xarray adapter + typed_public_api + CF metadata
  2. PNP structured exceptions + property tests
  3. PNP equivalence validation (baseline capture)
  4. PCI xarray adapter + typed_public_api + CF metadata (same pattern)

- **Track 1 Start (Week 1-3, parallel with Track 0):**
  1. PM-ET helpers in `eto.py` (`_atm_pressure`, `_psy_const`, `_svp_from_t`, `_mean_svp`, `_slope_svp`)
  2. Humidity pathway helpers (`_avp_from_*`)
  3. Public `eto_penman_monteith()` with dispatcher
  4. FAO56 Example 17 & 18 validation tests
  5. PM-ET xarray adapter + typed_public_api

- **Track 0 Continuation (Week 2-3, MUST complete Palmer before Track 3):**
  6. Palmer structlog migration (stdlib → structlog lifecycle events) **← BLOCKS TRACK 3**
  7. ETo Thornthwaite lifecycle completion
  8. ETo Hargreaves typed_public_api (if missing)
  9. Expanded SPEI + Palmer property tests

**Phase 2: Index Coverage + Advanced xarray (Parallel Tracks 2 + 3 after Phase 1)**
- **Track 2 (Week 4-6, requires Track 1 PM-ET + Track 0 PNP):**
  1. EDDI xarray adapter + typed_public_api
  2. NOAA reference dataset download + provenance metadata
  3. `test_reference_validation.py` with EDDI validation (1e-5 tolerance)
  4. EDDI docstring PM-ET recommendation (cross-reference Track 1)
  5. scPDSI stub with full signature + NotImplementedError

- **Track 3 (Week 4-7, requires Track 0 Palmer structlog):**
  1. `palmer_xarray.py` module creation
  2. Manual wrapper pattern implementation (stack/unpack)
  3. Dataset construction with per-variable CF metadata
  4. params_dict dual serialization (JSON + individual attrs)
  5. AWC spatial dimension validation
  6. typed_public_api Palmer `@overload` (numpy tuple vs xarray Dataset)
  7. `test_palmer_xarray.py` — equivalence, Dataset structure, AWC tests
  8. Performance benchmark (target: ≥80% of multiprocessing baseline)

**Critical Dependencies:**
- Track 0 Palmer structlog MUST complete before Track 3 Palmer xarray work starts
- Track 1 PM-ET MUST complete before Track 2 EDDI recommendation
- Track 0 PNP MUST complete before Track 2 (if PNP used in EDDI examples)

## Verification Plan

### How to Test End-to-End (v2.4.0)

**Backward Compatibility (v1.1 tests must still pass):**
```bash
# Run existing v1.1 tests (must still pass — backward compat guarantee)
uv run pytest tests/test_indices.py tests/test_compute.py tests/test_eto.py -v
uv run pytest tests/test_xarray_adapter.py -v
uv run pytest tests/test_logging.py -v
uv run pytest tests/test_exceptions.py -v
```

**Track 0 (Pattern Completion):**
```bash
# PNP/PCI equivalence (before/after pattern application)
uv run pytest tests/test_xarray_adapter.py::test_percentage_of_normal_xarray_equivalence -v
uv run pytest tests/test_xarray_adapter.py::test_pci_xarray_equivalence -v

# Property-based tests
uv run pytest tests/test_properties.py::TestPercentageOfNormalProperties -v
uv run pytest tests/test_properties.py::TestPCIProperties -v
uv run pytest tests/test_properties.py::TestSPEIProperties -v
uv run pytest tests/test_properties.py::TestPalmerProperties -v

# Structlog lifecycle events
uv run pytest tests/test_logging.py::test_palmer_lifecycle_events -v
uv run pytest tests/test_logging.py::test_eto_thornthwaite_lifecycle -v

# Type safety
uv run mypy src/climate_indices/typed_public_api.py --strict
# Should pass for: percentage_of_normal, pci, eto_thornthwaite, eto_hargreaves, pdsi
```

**Track 1 (PM-ET):**
```bash
# FAO56 validation (±0.05 mm/day tolerance)
uv run pytest tests/test_eto.py::test_fao56_example_17_bangkok -v
uv run pytest tests/test_eto.py::test_fao56_example_18_uccle -v

# Helper function unit tests
uv run pytest tests/test_eto.py::test_atm_pressure -v
uv run pytest tests/test_eto.py::test_svp_from_t -v
uv run pytest tests/test_eto.py::test_mean_svp -v

# Humidity pathway selection
uv run pytest tests/test_eto.py::test_humidity_pathway_dewpoint -v
uv run pytest tests/test_eto.py::test_humidity_pathway_rh_extremes -v
uv run pytest tests/test_eto.py::test_humidity_pathway_rh_mean -v

# PM-ET xarray equivalence
uv run pytest tests/test_xarray_adapter.py::test_eto_penman_monteith_xarray_equivalence -v
```

**Track 2 (EDDI/PNP/scPDSI):**
```bash
# NOAA reference validation (1e-5 tolerance)
uv run pytest tests/test_reference_validation.py::test_eddi_noaa_reference -v

# EDDI xarray equivalence
uv run pytest tests/test_xarray_adapter.py::test_eddi_xarray_equivalence -v

# scPDSI stub behavior
uv run pytest tests/test_indices.py::test_scpdsi_not_implemented -v
```

**Track 3 (Palmer xarray):**
```bash
# Palmer xarray equivalence (numpy tuple vs xarray Dataset)
uv run pytest tests/test_palmer_xarray.py::test_palmer_xarray_equivalence -v

# Dataset structure validation
uv run pytest tests/test_palmer_xarray.py::test_palmer_dataset_structure -v
uv run pytest tests/test_palmer_xarray.py::test_palmer_cf_metadata -v
uv run pytest tests/test_palmer_xarray.py::test_palmer_params_dict_dual_access -v

# AWC spatial dimension validation
uv run pytest tests/test_palmer_xarray.py::test_awc_spatial_only -v
uv run pytest tests/test_palmer_xarray.py::test_awc_time_dimension_error -v

# Performance benchmark (≥80% baseline)
uv run pytest tests/test_palmer_xarray.py::test_palmer_xarray_performance -v --benchmark

# Type safety
uv run mypy src/climate_indices/palmer_xarray.py --strict
```

**Full v2.4.0 Suite:**
```bash
# Type checking (all new modules)
uv run mypy src/climate_indices/ --strict

# Linting
ruff check src/climate_indices/ tests/
ruff format --check src/climate_indices/ tests/

# Security scanning
uv run --group security bandit -r src/climate_indices/
uv run --group security pip-audit

# Full test suite with coverage
uv run pytest tests/ -v --cov=src/climate_indices --cov-report=term --cov-report=html

# Coverage thresholds (enforce >85% per NFR-MAINT-002)
uv run pytest tests/ --cov=src/climate_indices --cov-fail-under=85
```

### Key Validation Checks (v1.1 — Retained)
- ✅ `test_spi_xarray_equivalence`: xarray SPI == NumPy SPI within 1e-8
- ✅ `test_metadata_preservation`: coordinates and CF attributes intact
- ✅ `test_parameter_inference`: inferred params match explicit values
- ✅ `test_backward_compat`: existing NumPy API unchanged
- ✅ `test_structlog_json_output`: JSON format parseable
- ✅ `test_exception_hierarchy`: ClimateIndicesError catches all subclasses

### Key Validation Checks (v2.4.0 Additions)

**Track 0 (Pattern Compliance):**
- ✅ `test_percentage_of_normal_xarray_equivalence`: PNP numpy == xarray within 1e-8
- ✅ `test_pci_xarray_equivalence`: PCI numpy == xarray within 1e-8
- ✅ `test_pnp_boundedness_property`: PNP ≥ 0 always (property-based)
- ✅ `test_pci_range_property`: 0 ≤ PCI ≤ 100 (property-based)
- ✅ `test_palmer_structlog_lifecycle`: Palmer emits calculation_started/completed events

**Track 1 (PM-ET):**
- ✅ `test_fao56_example_17_bangkok`: Bangkok tropical monthly ETo = 5.72 ±0.05 mm/day
- ✅ `test_fao56_example_18_uccle`: Uccle temperate daily ETo = 3.9 ±0.05 mm/day
- ✅ `test_svp_from_t_reference`: SVP(21.5°C) = 2.564 ±0.01 kPa
- ✅ `test_humidity_pathway_priority`: dewpoint > RH extremes > RH mean selection
- ✅ `test_pm_et_xarray_equivalence`: PM-ET numpy == xarray within 1e-8

**Track 2 (EDDI/PNP/scPDSI):**
- ✅ `test_eddi_noaa_reference`: EDDI outputs match NOAA PSL reference within 1e-5
- ✅ `test_eddi_reference_provenance`: Reference dataset has source, url, sha256 attrs
- ✅ `test_scpdsi_raises_not_implemented`: scPDSI stub raises NotImplementedError with Wells et al. reference

**Track 3 (Palmer xarray):**
- ✅ `test_palmer_xarray_equivalence`: Palmer numpy tuple == xarray Dataset values within 1e-8 (all 4 variables)
- ✅ `test_palmer_dataset_four_variables`: Dataset contains pdsi, phdi, pmdi, z_index
- ✅ `test_palmer_cf_metadata_per_variable`: Each variable has long_name, units, references from registry
- ✅ `test_palmer_params_dict_json`: `json.loads(ds.attrs["palmer_params"])` reconstructs dict
- ✅ `test_palmer_params_dict_scalar_access`: `ds.attrs["palmer_alpha"]` equals `params["alpha"]`
- ✅ `test_awc_time_dimension_raises_error`: AWC with time dim raises ValueError with actionable message
- ✅ `test_palmer_xarray_performance`: xarray path ≥80% speed of multiprocessing baseline (synthetic 360×180×240 grid)

## Post-Architecture Actions

### Immediate: Commit Architecture Document (v2.4.0 Update)
1. ✅ Architecture document updated at `_bmad-output/planning-artifacts/architecture.md`
2. Stage and commit:
   ```bash
   git add _bmad-output/planning-artifacts/architecture.md
   git commit -m "docs(bmad): update architecture v2.4.0 — PM-ET, Palmer xarray, EDDI validation, pattern completion

   Extends v1.1 foundation with:
   - 7 new decisions (8-14): PM-ET placement, Palmer Dataset return, reference validation
   - 6 new patterns (9-14): FAO56 helpers, multi-output adapter, equivalence protocol
   - 4-track implementation strategy: Track 0 ∥ Track 1 → Track 2 ∥ Track 3
   - 100% requirements coverage (90 FRs, 31 NFRs)

   Research-informed: PM FAO56, Palmer modernization, EDDI validation"
   ```

### Next BMAD Workflow Steps (v2.4.0)

**1. Implementation Readiness Check (RECOMMENDED before coding):**
   - Run BMAD workflow: `/check-implementation-readiness`
   - Validates PRD v2.4.0 + Architecture v2.4.0 alignment
   - Verifies track dependencies are correctly ordered
   - Confirms no missing architectural coverage for 90 FRs

**2. Epic/Story Breakdown (4 parallel epics):**
   - Epic 0: Track 0 — Canonical Pattern Completion (PNP, PCI, Palmer structlog, ETo, properties)
   - Epic 1: Track 1 — Penman-Monteith FAO56 PET (helpers, dispatcher, validation)
   - Epic 2: Track 2 — EDDI/PNP/scPDSI Coverage (reference validation, CLI integration)
   - Epic 3: Track 3 — Palmer Multi-Output xarray (manual wrapper, Dataset return, performance)

**3. Implementation Execution Order:**
   - **Week 1-3:** Launch Track 0 + Track 1 in parallel
   - **Week 3 (Critical Path):** Complete Track 0 Palmer structlog (BLOCKS Track 3)
   - **Week 4-7:** Launch Track 2 + Track 3 in parallel
   - **Week 7+:** Integration testing, performance validation, documentation

**4. Key Milestones:**
   - ✅ Track 0 PNP complete → Pattern migration protocol validated
   - ✅ Track 1 PM-ET FAO56 examples pass → Physics-based PET available
   - ✅ Track 0 Palmer structlog → Unblocks Track 3
   - ✅ Track 2 EDDI NOAA validation → Reference framework proven
   - ✅ Track 3 Palmer xarray equivalence → Multi-output pattern validated
   - ✅ All tracks complete → v2.4.0 ready for integration testing

**5. Risk Mitigation:**
   - Track 0 equivalence protocol enforces zero algorithmic drift (NFR-PATTERN-EQUIV)
   - Track 1 FAO56 validation ensures scientific accuracy (NFR-PM-PERF)
   - Track 3 performance benchmark prevents regression (NFR-PALMER-PERF ≥80%)
   - Track dependencies explicit to prevent integration conflicts

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

### Version 2.4.0 (2026-02-16)
- **Incremental update for PRD v2.4.0** — Extends v1.1 foundation with 30 new FRs across 4 tracks
- **Added 7 New Decisions** (8-14):
  - Decision 8: PM-ET module placement in `eto.py`
  - Decision 9: PM-ET humidity dispatcher with auto-select + override
  - Decision 10: Palmer xarray manual wrapper (Pattern C from research)
  - Decision 11: Palmer Dataset return with per-variable CF metadata
  - Decision 12: Reference dataset testing with provenance (SHA256, algorithm version)
  - Decision 13: Track 0 pattern application strategy (index-by-index with equivalence protocol)
  - Decision 14: scPDSI stub with full signature + NotImplementedError
- **Added 6 New Patterns** (9-14):
  - Pattern 9: PM-ET helper function contract (FAO56 equation traceability)
  - Pattern 10: Humidity pathway selection (priority order + logging)
  - Pattern 11: Multi-output xarray adapter for Palmer (stack/unpack workaround)
  - Pattern 12: Reference dataset management (provenance metadata requirements)
  - Pattern 13: Track 0 equivalence protocol (baseline capture → pattern → validate)
  - Pattern 14: Palmer CF metadata registry entries (4 variables)
- **Extended Project Structure**: 3 new modules (`palmer_xarray.py`), 4 new test files, extensive modifications to `eto.py`, `typed_public_api.py`, `xarray_adapter.py`
- **Updated Requirements Coverage**: 100% (90 FRs, 31 NFRs) — up from 98% (60 FRs, 23 NFRs)
- **Research Integration**: All decisions informed by 3 technical research documents (PM FAO56, Palmer modernization, EDDI validation)
- **Track Dependencies**: Explicit dependency graph for 4-track parallel execution (Track 0 ∥ Track 1 → Track 2 ∥ Track 3)
- **Impact**: Major architectural extension — 4 new components (PM-ET helpers, Palmer xarray wrapper, reference validation framework, pattern migration framework)

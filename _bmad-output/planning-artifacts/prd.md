---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
inputDocuments:
  - '_bmad-output/planning-artifacts/prd.md'
  - '_bmad-output/planning-artifacts/architecture.md'
  - '_bmad-output/planning-artifacts/research/technical-penman-monteith.md'
  - '_bmad-output/planning-artifacts/research/technical-palmer-modernization.md'
  - '_bmad-output/planning-artifacts/research/technical-eddi-validation.md'
  - '_bmad-output/project-context.md'
  - 'docs/index.md'
workflowType: 'prd'
prdVersion: '2.4.0'
priorVersion: '1.1'
---

# Product Requirements Document - climate_indices v2.4.0

**Author:** James
**Date:** 2026-02-15
**Status:** Complete (All 11 Steps + Track 0 Integration)
**Version:** 2.4.0
**Prior Version:** 1.1 (xarray Integration + structlog Modernization)

---

## Executive Summary

This PRD defines requirements for climate_indices version 2.4.0, building on the foundation established in PRD v1.1 (xarray + structlog modernization). Version 2.4.0 focuses on **canonical pattern completion**, **scientific algorithm completeness**, and **advanced xarray capabilities**, informed by three comprehensive technical research documents and codebase inventory analysis completed in February 2026.

**Key Additions in v2.4.0:**
1. **Canonical Pattern Completion (Track 0)** — Apply v2.3.0 patterns to ALL remaining indices for 100% consistency
2. **Penman-Monteith FAO56 PET (Track 1)** — Physics-based evapotranspiration (completes PET method suite)
3. **Palmer Multi-Output xarray Support (Track 3)** — Advanced xarray adapter for 4-variable output (PDSI, PHDI, PMDI, Z-Index)
4. **EDDI Compliance & Integration (Track 2)** — NOAA reference validation, CLI integration, PM-ET recommendation
5. **PNP xarray Support (Track 2)** — Percent of Normal Precipitation with xarray adapter
6. **scPDSI Interface Definition (Track 2)** — Self-calibrating PDSI stub for future implementation

**Context:**
- PRD v1.0/1.1 delivered foundational xarray support for SPI, SPEI, and basic PET (Thornthwaite, Hargreaves)
- v2.3.0 established 6 canonical patterns but only applied them to 3 indices (SPI, SPEI, PET)
- Codebase inventory revealed pattern gaps in `percentage_of_normal`, `pci`, Palmer, and ETo helpers
- Three technical research efforts validated implementation approaches and identified specific requirements
- This PRD organizes work into **4 parallel tracks** with clear dependency ordering

**Phased Delivery:**
- **Track 0: Canonical Pattern Completion** — Parallel with Track 1, partially blocks Track 3
- **Track 1: Foundation (PM-ET + Infrastructure)** — Parallel with Track 0, required by Tracks 2 & 3
- **Track 2: Index Coverage (EDDI, PNP, scPDSI stub)** — Parallel with Track 3 after Track 0 + Track 1
- **Track 3: Advanced xarray (Palmer multi-output)** — Parallel with Track 2 after Track 0 (Palmer structlog) + Track 1

---

## Step 1: Initialization Complete

**Project Type:** Brownfield Enhancement (v2.4.0 iteration)

**Context Source:** 7 input documents
- `_bmad-output/planning-artifacts/prd.md` (v1.1) — 60 FRs, 23 NFRs from prior version
- `_bmad-output/planning-artifacts/architecture.md` (v1.1) — Adapter patterns, CF metadata registry
- `_bmad-output/planning-artifacts/research/technical-penman-monteith.md` — PM FAO56 equations 1-19 analysis
- `_bmad-output/planning-artifacts/research/technical-palmer-modernization.md` — xarray multi-output patterns
- `_bmad-output/planning-artifacts/research/technical-eddi-validation.md` — EDDI NOAA reference validation gaps
- `_bmad-output/project-context.md` — Development rules and constraints
- `docs/index.md` — Project overview and current state

**Stakeholders:**
- **Author/Maintainer:** James A. (sole developer)
- **Primary Users:** Climate researchers, drought monitoring agencies (NIDIS, NOAA)
- **Scientific Domain Experts:** FAO56 reference (PM-ET), NOAA PSL (EDDI), Palmer water balance modeling
- **Community:** Open-source contributors, scientific Python ecosystem

**Brownfield Context:**
This PRD builds on a mature library (v2.2.0) with:
- 14 modules (9,800+ lines of source code)
- 26 test files (>90% coverage)
- Active CI/CD with PyPI distribution
- Established user base in operational drought monitoring

---

## Step 2: Project Classification

**Primary Classification:** Developer Tool / Library
**Domain:** Scientific Computing (Climate Science)
**Complexity:** Medium-High
**Project Context:** Brownfield — Adding Scientific Algorithms + Advanced xarray Capabilities

**Rationale:**
- **Developer Tool:** Programmatic API for climate index calculations (not end-user application)
- **Scientific Computing:** Implements peer-reviewed drought indices with numerical reproducibility requirements
- **Medium-High Complexity:**
  - Physics-based algorithms (PM FAO56 equations 1-19)
  - Multi-output xarray patterns (Palmer: 4 variables + params dict)
  - NOAA reference validation requirements (FR-TEST-004)
  - Sequential state tracking (Palmer water balance)
- **Brownfield Additions:** Building on proven foundation (xarray + structlog from v1.1)

---

## Step 3: Success Criteria (v2.4.0 Additions)

### User Success (Building on v1.1)

**Definition:** Climate researchers can use physics-based PET and access Palmer indices via modern xarray API with consistent patterns across ALL indices

**Measurable Outcomes (Tracks 1-3):**
1. **PM-ET Adoption:** 40% of new EDDI users adopt PM FAO56 within 3 months (up from Thornthwaite baseline)
2. **Palmer xarray Enablement:** Users can compute all 4 Palmer outputs via single Dataset return (no manual tuple unpacking)
3. **EDDI Validation:** EDDI outputs match NOAA reference within 1e-5 tolerance (FR-TEST-004 compliance)
4. **PNP Simplicity:** PNP calculation requires <5 lines of xarray code

**Measurable Outcomes (Track 0 — Canonical Pattern Completion):**
5. **API Consistency:** 100% of public index functions accessible via both numpy and xarray paths (up from ~43% — 3/7 indices)
6. **Error Clarity:** All validation errors use structured exceptions with actionable context (eliminates generic `ValueError`)
7. **Debugging Efficiency:** structlog lifecycle events available for ALL indices (currently only SPI/SPEI/partial-PET)
8. **Property Confidence:** Users can rely on documented mathematical properties verified by property-based tests

### Technical Success (v2.4.0 Specific)

**Definition:** Implementation is scientifically accurate, performant, maintainable, and pattern-consistent

**Measurable Outcomes (Tracks 1-3):**
1. **PM-ET Accuracy:** FAO56 Examples 17 & 18 reproduce within 0.05 mm/day (tropical + temperate validation)
2. **Palmer Performance:** xarray path achieves ≥80% speed of multiprocessing CLI baseline
3. **Multi-Output Pattern:** Palmer Dataset return is type-safe (`mypy --strict` passes) with CF metadata per variable
4. **EDDI Compliance:** NOAA reference validation test passes (tolerance: 1e-5) per Architecture v1.1 Pattern 8

**Measurable Outcomes (Track 0 — Pattern Compliance Dashboard):**
5. **xarray Coverage:** 7/7 public indices support xarray DataArray inputs (100% coverage)
   - Track 0 completes: `percentage_of_normal`, `pci`
   - Track 3 completes: Palmer multi-output
6. **Type Safety:** 7/7 index functions have `@overload` signatures in `typed_public_api.py` (mypy --strict passes)
   - Missing: `percentage_of_normal`, `pci`, `eto_thornthwaite`, `eto_hargreaves`
7. **CF Metadata:** 100% compliance—all xarray outputs have `long_name`, `units`, `references`
8. **structlog Migration:** All modules use structlog with lifecycle event patterns
   - Palmer: stdlib logging → structlog migration complete
   - ETo: `eto_thornthwaite` lifecycle completion (currently has logger, missing bind)
9. **Structured Exceptions:** 100% of validation/computation errors use exception hierarchy
   - Replace `ValueError` in: `percentage_of_normal`, `pci`, `eto_*`, Palmer
10. **Property-Based Tests:** All indices have property tests documenting mathematical invariants
    - Missing: PNP (boundedness, positive values), PCI (range [0,100]), SPEI (expanded), Palmer (PHDI/PMDI/Z-Index)

### Business Success (v2.4.0 Impact)

**Measurable Outcomes (Tracks 1-3):**
1. **Scientific Credibility:** PM FAO56 implementation cited in at least 1 research paper within 12 months
2. **Operational Readiness:** EDDI validated against NOAA reference enables operational deployment confidence
3. **Ecosystem Contribution:** First Python library to deliver Palmer + xarray multi-output (no prior art per research)

**Measurable Outcomes (Track 0 — Technical Debt Reduction):**
4. **Maintenance Velocity:** Pattern consistency enables 2x faster feature development in v2.5.0
5. **Onboarding Time:** New contributors understand ANY index via established patterns (reduces ramp-up from ~2 weeks to ~3 days)
6. **Bug Reduction:** Property-based tests provide continuous edge-case validation (target: 50% reduction in user-reported bugs)

---

## Step 4: User Journeys (v2.4.0 Additions)

### Journey 6: Flash Drought Researcher (NEW — EDDI User)
**Actor:** Dr. Maria Santos, agricultural meteorologist studying rapid soil moisture depletion

**Context:**
- Monitors evaporative demand using EDDI at weekly to monthly timescales
- Requires Penman-Monteith PET for physical accuracy (wind + humidity effects critical)
- Currently uses NOAA pre-computed EDDI but wants custom regional analysis

**Current Pain Points:**
1. climate_indices has EDDI algorithm but no PM-ET method (must use Thornthwaite — inappropriate)
2. No validation that library matches NOAA operational output
3. No guidance on PET method selection for EDDI

**Desired Outcome (v2.4.0):**
```python
import xarray as xr
from climate_indices import eto_penman_monteith, eddi

# Compute PM-ET (recommended for EDDI)
pet_da = eto_penman_monteith(
    tmin_da, tmax_da, tmean_da, wind_da, rn_da,
    latitude_degrees=40.0, altitude_meters=1500,
    rh_max=rhmax_da, rh_min=rhmin_da
)

# EDDI with validated algorithm
eddi_6mo = eddi(pet_da, scale=6, ...)
# ✅ Validated against NOAA reference (tolerance: 1e-5)
```

**Success Criteria:**
- PM-ET available with FAO56 validation
- EDDI output matches NOAA reference within documented tolerance
- Docstring cross-references PM-ET as recommended method

---

### Journey 7: Water Balance Modeler (NEW — Palmer User)
**Actor:** NIDIS drought monitoring technician updating Palmer indices for weekly dashboard

**Context:**
- Computes PDSI, PHDI, PMDI, Z-Index from gridded climate data
- Current workflow: extract .values, loop over grid cells, manually reassemble to Dataset
- Needs all 4 outputs + calibration params for downstream analysis

**Current Pain Points:**
1. Palmer returns 5-tuple (4 arrays + params dict) — awkward with xarray
2. Must manually wrap each output with coordinates and CF metadata
3. No type safety (tuple unpacking errors common)

**Desired Outcome (v2.4.0):**
```python
import xarray as xr
from climate_indices import pdsi  # typed_public_api with @overload

# Single call, clean Dataset return
ds_palmer = pdsi(
    precip_da,  # xr.DataArray
    pet_da,     # xr.DataArray
    awc=2.5,    # scalar or DataArray for spatial variation
)

# Access outputs naturally
pdsi_values = ds_palmer["pdsi"]
phdi_values = ds_palmer["phdi"]
z_index = ds_palmer["z_index"]

# Params stored in attrs
params = json.loads(ds_palmer.attrs["palmer_params"])
alpha = ds_palmer.attrs["palmer_alpha"]  # Direct access

# CF-compliant output ready for visualization/distribution
ds_palmer.to_netcdf("palmer_indices.nc")
```

**Success Criteria:**
- Dataset return is type-safe (`xr.Dataset` vs tuple in signatures)
- All 4 variables have independent CF metadata
- params_dict accessible via both JSON and individual attrs
- Performance ≥80% of current multiprocessing baseline

---

## Step 5: Domain Requirements (v2.4.0 Additions)

### Scientific Correctness (PM-ET Specific)

**Requirement 1: FAO56 Equation Fidelity**
- **Spec:** Implementation must match FAO Irrigation & Drainage Paper 56 equations 6-19 exactly
- **Validation:** Reproduce FAO56 Example 17 (Bangkok tropical monthly) and Example 18 (Uccle temperate daily)
- **Tolerance:** ±0.05 mm/day for worked examples
- **Critical Details:**
  - Kelvin conversion in denominator: `T + 273` (not `T` alone)
  - SVP non-linearity: `es = (e°(Tmax) + e°(Tmin)) / 2` ≠ `e°(Tmean)`
  - Magnus constants: `0.6108, 17.27, 237.3` (exact FAO56 values)

**Requirement 2: Humidity Pathway Hierarchy**
- **Priority Order (per FAO56):**
  1. Dewpoint (Eq. 14) — most accurate
  2. RH extremes (Eq. 17) — preferred for daily data
  3. RH mean (Eq. 19) — fallback
- **Implementation:** Auto-select pathway based on available inputs
- **User Override:** Accept explicit `actual_vapor_pressure` parameter

**Requirement 3: Penman-Monteith for EDDI**
- **Spec:** EDDI docstring must recommend PM FAO56 for PET
- **Rationale:** Hobbins et al. 2016 uses PM exclusively; Thornthwaite inappropriate (misses wind/humidity)
- **Validation:** Document PET method sensitivity (see EDDI research Section 3.3)

---

### xarray Multi-Output Requirements (Palmer-Specific)

**Requirement 4: Dataset Return Type**
- **Spec:** `palmer_xarray()` returns `xr.Dataset` with 4 variables: `pdsi`, `phdi`, `pmdi`, `z_index`
- **Rationale:** CF-compliant container for NetCDF interchange; per-variable metadata
- **Type Safety:** `@overload` signatures distinguish numpy tuple vs xarray Dataset returns

**Requirement 5: params_dict Handling**
- **Spec:** Calibration parameters (alpha, beta, gamma, delta) stored in:
  1. JSON string: `ds.attrs["palmer_params"]` (full dict, CF-compliant)
  2. Individual attrs: `ds.attrs["palmer_alpha"]` etc. (direct access)
- **Computation:** Params computed once from first grid cell (spatially constant)
- **Serialization:** JSON round-trip preserves full structure

**Requirement 6: AWC Spatial Parameter**
- **Spec:** Available Water Capacity (AWC) may be scalar or DataArray
- **Dimensions:** If DataArray, must have spatial dims only (lat, lon) — NOT time
- **Validation:** Raise `ValueError` if AWC has time dimension
- **Error Message:** `"AWC must not have time dimension 'time'. AWC is a soil property (spatially varying only)."`

---

### NOAA Reference Validation (EDDI-Specific)

**Requirement 7: FR-TEST-004 Implementation**
- **Spec:** Validate EDDI outputs against NOAA PSL reference dataset
- **Tolerance:** `np.testing.assert_allclose(computed, noaa_reference, rtol=1e-5, atol=1e-5)`
- **Rationale:** Looser than equivalence tests (1e-8) due to non-parametric ranking FP accumulation
- **Test Location:** `tests/test_reference_validation.py::test_eddi_noaa_reference()`
- **Data Source:** NOAA PSL EDDI CONUS archive ([downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/](https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/))

**Requirement 8: Reference Dataset Provenance**
- **Metadata:** `tests/data/reference/eddi_noaa_reference.nc` must include attributes:
  - `source`: "NOAA PSL EDDI CONUS archive"
  - `url`: Full download URL
  - `download_date`: ISO 8601 format
  - `subset_description`: Spatial/temporal extent
- **Documentation:** `tests/data/reference/README.md` with provenance details

---

## Step 6: Innovation Analysis

**Conclusion:** Limited novelty; focus on **scientific rigor** and **ecosystem contribution**

**Non-Novel Elements:**
- PM FAO56 equations are standard reference (Allen et al. 1998)
- xarray multi-output patterns exist (workarounds for `apply_ufunc` limitations)
- EDDI non-parametric ranking is established (Hobbins et al. 2016)

**Ecosystem Contributions (Differentiation via Completeness):**
- **First Python library** with Palmer + xarray multi-output (per Palmer research — no xclim, pyet, standard_precip precedent)
- **Only open-source implementation** with NOAA-validated EDDI (reference test at 1e-5 tolerance)
- **Complete FAO56 PM suite** with worked example validation (pyeto/pyet exist but not integrated with drought indices)

**Innovation Classification:** **Engineering Excellence** (not Research Innovation)

---

## Step 7: Track-Based Architecture (v2.4.0 Phasing Strategy)

### 4-Track Architecture Overview

Version 2.4.0 organizes work into **4 parallel tracks** with explicit dependency ordering:

**Track 0: Canonical Pattern Completion** (NEW)
- Apply v2.3.0-established patterns to ALL remaining indices
- Complete pattern migration debt from prior releases
- Foundation for long-term maintainability

**Track 1: Foundation — Penman-Monteith FAO56**
- Physics-based PET method
- Infrastructure validation before algorithm expansion

**Track 2: Index Coverage — EDDI/PNP/scPDSI**
- Complete drought index catalog
- NOAA reference validation

**Track 3: Advanced xarray — Palmer Multi-Output**
- Multi-variable Dataset pattern
- Advanced xarray capabilities

### Dependency Rationale

**Why Track 0 Runs in Parallel with Track 1:**
- Track 0 applies proven patterns (no research needed) → can start immediately
- Track 1 establishes new PM-ET infrastructure → independent of legacy refactoring
- Both tracks validate infrastructure patterns before complex implementations (Tracks 2 & 3)

**Why Track 0 Partially Blocks Track 3:**
- Palmer `structlog` migration (Track 0) must complete BEFORE Palmer `xarray` multi-output work (Track 3)
- Rationale: Don't mix logging frameworks during complex xarray refactoring (risk of debugging confusion)
- Other Track 0 work (PNP/PCI xarray adapters) is independent of Palmer

**Why Track 1 is Foundation:**
- PM-ET (`eto_penman_monteith()`) is **required** by EDDI (Track 2) for scientific accuracy
- Infrastructure patterns (equation helpers, type annotations, CF metadata) validate before Palmer complexity
- FAO56 validation examples establish numerical precision standards

**Why Tracks 2 & 3 are Parallel (after dependencies met):**
- EDDI/PNP (Track 2) and Palmer (Track 3) are **independent** after PM-ET availability and Track 0 completion
- Both use established patterns (Track 2: existing `@xarray_adapter`; Track 3: new manual wrapper)
- Resource allocation: Can assign to separate development efforts if needed

**Dependency Diagram:**
```
Track 0: Canonical Pattern Completion ──┐
                                        ├─── both complete → unlock:
Track 1: Foundation (PM-ET)            ──┘
    ↓
    ├── Track 2: Index Coverage (EDDI + PNP + scPDSI)     [parallel execution]
    │   └── requires: PM-ET (Track 1), PNP xarray (Track 0)
    │
    └── Track 3: Advanced xarray (Palmer multi-output)    [parallel execution]
        └── requires: PM-ET infra (Track 1), Palmer structlog (Track 0)
```

**Critical Path:** Track 0 Palmer structlog + Track 1 PM-ET → Track 3 Palmer xarray

**Parallel Optimization:** Track 0 (PNP/PCI patterns) + Track 1 (PM equations) can run simultaneously

---

## Step 8: Scoping and Phasing

### Track 0: Canonical Pattern Completion (Parallel with Track 1)

**Goal:** Apply v2.3.0-established canonical patterns to ALL remaining indices for consistency and maintainability

**Duration Estimate:** 2-3 weeks (parallel with Track 1, some work blocks Track 3)

**Rationale:**
- v2.3.0 established 6 canonical patterns for SPI, SPEI, and PET (Thornthwaite/Hargreaves)
- Remaining indices (`percentage_of_normal`, `pci`, Palmer) still use legacy patterns
- Creates inconsistent developer experience and technical debt
- Must complete before v2.5.0 to avoid compounding pattern divergence

**Included Components:**

**1. xarray Adapter Application**
- Apply `@xarray_adapter` decorator to:
  - `percentage_of_normal()` — Percent of Normal Precipitation
  - `pci()` — Precipitation Concentration Index
- CF metadata registry entries:
  - PNP: `long_name="Percent of Normal Precipitation"`, `units="%"`
  - PCI: `long_name="Precipitation Concentration Index"`, `units=""`
- Equivalence tests: numpy vs xarray paths (tolerance: 1e-8)

**2. typed_public_api.py Expansion**
- Add `@overload` signatures for numpy→numpy and xarray→xarray dispatch:
  - `percentage_of_normal` (currently missing)
  - `pci` (currently missing)
  - `eto_thornthwaite` (currently missing)
  - `eto_hargreaves` (currently missing)
- Follow existing SPI/SPEI pattern for consistency
- Validate with `mypy --strict`

**3. Structured Exception Migration**
- Replace generic `ValueError` with hierarchy exceptions:
  - `percentage_of_normal`: Input validation → `InvalidArgumentError`
  - `pci`: Daily data requirement → `InvalidArgumentError`
  - `eto_thornthwaite`, `eto_hargreaves`: Parameter validation → `InvalidArgumentError`
  - Palmer (`pdsi`): All validation → structured exceptions
- Use existing exception hierarchy from `compute.py`:
  - `InvalidArgumentError` (base class: `ClimateIndicesError`)
  - `DistributionFittingError`, `InsufficientDataError` (as needed)

**4. Palmer structlog Migration**
- Migrate `palmer.py` from stdlib `logging` to `structlog`
- Replace `_logger = utils.get_logger(__name__, logging.DEBUG)` with `from climate_indices.logging_config import get_logger`
- Add lifecycle event logging pattern:
  ```python
  _logger.bind(calculation="pdsi", data_shape=precips.shape, awc=awc)
  _logger.info("calculation_started")
  # ... computation ...
  _logger.info("calculation_completed", duration_ms=elapsed)
  ```
- Match SPI/SPEI logging patterns exactly

**5. structlog Lifecycle Completion**
- `eto_thornthwaite()`: Add `_logger.bind()` lifecycle events
  - Currently has logger instance but no lifecycle bind pattern
  - Add: `calculation_started`, `calculation_completed`, `calculation_failed`
- Ensure ALL index functions follow identical logging pattern

**6. Property-Based Test Expansion**
- `percentage_of_normal()`:
  - Property: Boundedness (positive values)
  - Property: Shape preservation
  - Property: NaN propagation
  - Property: Linear scaling `pnp(2×p) = 2×pnp(p)`
- `pci()`:
  - Property: Range [0, 100] always
  - Property: Requires 365/366 daily values (validate error for wrong length)
  - Property: NaN handling
- `spei()`:
  - Expand beyond boundedness
  - Add shape preservation, NaN propagation, zero-input tests
- Palmer (`pdsi`, `phdi`, `pmdi`, `z_index`):
  - Currently only PDSI bounded range test exists
  - Add properties for PHDI, PMDI, Z-Index
  - Property: Sequential consistency (output depends on full time series)

**Deliverables:**
- `src/climate_indices/xarray_adapter.py`: CF metadata for PNP, PCI
- `src/climate_indices/typed_public_api.py`: 4 new `@overload` signature sets
- `src/climate_indices/palmer.py`: structlog migration complete
- `src/climate_indices/eto.py`: lifecycle event completion
- `src/climate_indices/{indices,palmer,eto}.py`: Structured exceptions everywhere
- `tests/test_properties.py`: Expanded coverage for all indices
- `tests/test_equivalence.py`: PNP, PCI numpy vs xarray validation

**Success Criteria:**
- ✅ 7/7 public indices support xarray (100% coverage)
- ✅ 7/7 index functions have `@overload` signatures
- ✅ 100% of modules use structlog (zero stdlib logging)
- ✅ 100% of validation errors use structured exceptions
- ✅ All indices have property-based tests
- ✅ mypy --strict passes
- ✅ All equivalence tests pass (tolerance: 1e-8)

**Dependencies:**
- **Blocks Track 3 (Partial):** Palmer structlog migration must complete before Palmer xarray multi-output work
- **Independent of Track 1:** Pattern application doesn't require PM-ET
- **Enables Track 2:** PNP xarray support is Track 0 deliverable used in Track 2

**Out of Scope (Track 0):**
- New pattern development (only applying existing patterns)
- Algorithm changes (pure refactoring only)
- Performance optimization (maintain current performance characteristics)

---

### Track 1: Foundation — Penman-Monteith FAO56 + Infrastructure (Required First)

**Goal:** Implement physics-based PET with FAO56 validation, establishing patterns for Tracks 2 & 3

**Duration Estimate:** 3-4 weeks

**Included Components:**
1. **PM-ET Core (Equations 6-13)**
   - `_atm_pressure(altitude)` — Eq. 7: Atmospheric pressure from elevation
   - `_psy_const(pressure)` — Eq. 8: Psychrometric constant
   - `_svp_from_t(temp)` — Eq. 11: Magnus formula for saturation vapor pressure
   - `_mean_svp(tmin, tmax)` — Eq. 12: Mean SVP (handles non-linearity)
   - `_slope_svp(temp)` — Eq. 13: Slope of SVP curve
   - `eto_penman_monteith()` — Eq. 6: Orchestration function

2. **Humidity Pathways (Equations 14-19)**
   - `_avp_from_dewpoint(tdew)` — Eq. 14: Most accurate
   - `_avp_from_rhminmax(...)` — Eq. 17: Preferred for daily data
   - `_avp_from_rhmean(...)` — Eq. 19: Fallback
   - Auto-dispatcher in `eto_penman_monteith()` based on available inputs

3. **Validation**
   - FAO56 Example 17 (Bangkok tropical, monthly): 5.72 mm/day ±0.05
   - FAO56 Example 18 (Uccle temperate, daily): 3.9 mm/day ±0.05

4. **Integration**
   - xarray adapter via existing `@xarray_adapter` pattern
   - CF metadata registry entry for PM-ET
   - Type annotations: `@overload` for numpy vs xarray

**Deliverables:**
- `src/climate_indices/eto.py` extended with PM helpers and `eto_penman_monteith()`
- `tests/test_eto.py` with FAO56 example validation
- `docs/algorithms.rst` updated with PM-ET section

**Out of Scope (Track 1):**
- Extended radiation equations (Eq. 20-52) — optional enhancement for Phase 3
- Crop coefficient methods (Kc, dual Kc) — beyond reference ETo
- EDDI integration (handled in Track 2)

---

### Track 2: Index Coverage Expansion (Parallel with Track 3, after Track 1)

**Goal:** Complete EDDI validation, add PNP xarray, stub scPDSI interface

**Duration Estimate:** 2-3 weeks (parallel with Track 3)

**Depends On:** Track 1 complete (PM-ET available for EDDI recommendation)

**Included Components:**

**EDDI Completion (FR-TEST-004 Blocker)**
1. **NOAA Reference Validation**
   - Download NOAA PSL EDDI CONUS archive subset
   - Create `tests/data/reference/eddi_noaa_reference.nc` with provenance metadata
   - Implement `tests/test_reference_validation.py::test_eddi_noaa_reference()`
   - Tolerance: 1e-5 (per Architecture v1.1 Pattern 8)

2. **EDDI xarray Adapter**
   - Apply `@xarray_adapter` decorator (existing pattern from v1.1)
   - CF metadata registry entry

3. **EDDI CLI Integration (Issue #414)**
   - Add `--index eddi` support to `process_climate_indices` CLI
   - Add `--pet_file` parameter
   - Update help text with PET method guidance

4. **Documentation Updates**
   - Add Hobbins et al. 2016 citation to docstring
   - Cross-reference `eto_penman_monteith()` as recommended PET method
   - Document sign convention (higher PET → higher EDDI → drier conditions)
   - Add `docs/algorithms.rst` EDDI section

**PNP (Percent of Normal Precipitation)**
1. **PNP xarray Adapter**
   - Simplest index (validates minimal metadata handling)
   - CF metadata: `long_name="Percent of Normal Precipitation"`, `units="%"`

**scPDSI Interface Definition**
1. **Stub Implementation**
   - Function signature: `scpdsi(precip, pet, awc, ...)`
   - Raise `NotImplementedError("scPDSI implementation planned for future release")`
   - Docstring with methodology overview and references (Wells et al. 2004)

**Deliverables:**
- `tests/test_reference_validation.py` (new module)
- `tests/data/reference/eddi_noaa_reference.nc` + provenance docs
- EDDI in `process_climate_indices` CLI
- PNP xarray support
- scPDSI stub in `indices.py`

**Success Criteria:**
- ✅ EDDI validates against NOAA reference within 1e-5
- ✅ FR-TEST-004 satisfied (Architecture v1.1 Pattern 8)
- ✅ PNP xarray path passes equivalence tests (1e-8)
- ✅ scPDSI stub documented for future work

---

### Track 3: Advanced xarray — Palmer Multi-Output (Parallel with Track 2, after Track 1)

**Goal:** Deliver Palmer indices with Dataset return and CF metadata per variable

**Duration Estimate:** 3-4 weeks (parallel with Track 2)

**Depends On:** Track 1 complete (validates infrastructure patterns are stable)

**Included Components:**

**Palmer Multi-Output Adapter (Pattern C from Research)**
1. **Manual `palmer_xarray()` Wrapper**
   - NOT decorator-based (multi-output requires custom handling)
   - File: `src/climate_indices/palmer_xarray.py` (new ~150-line module)
   - Stack/unpack pattern: `np.stack([pdsi, phdi, pmdi, z], axis=0)` → Dataset

2. **Multi-Output Dataset Construction**
   - Return: `xr.Dataset` with variables `pdsi`, `phdi`, `pmdi`, `z_index`
   - Per-variable CF metadata from registry
   - params_dict: dual access (JSON string + individual attrs)

3. **AWC Spatial Parameter Handling**
   - Accept scalar or DataArray (no time dimension)
   - Validation: raise error if `time_dim in awc.dims`
   - `input_core_dims=[["time"], ["time"], []]` pattern for apply_ufunc

4. **Type Safety**
   - `@overload` in `typed_public_api.py`:
     - numpy path: `tuple[NDArray, NDArray, NDArray, NDArray, dict | None]`
     - xarray path: `xr.Dataset`
   - Runtime dispatcher based on `isinstance(precips, xr.DataArray)`

5. **Performance Validation**
   - Benchmark: xarray path vs multiprocessing CLI baseline
   - Target: ≥80% speed (accept `vectorize=True` overhead per research Section 6.2.6)

**CF Metadata Registry Additions**
```python
CF_METADATA["pdsi"] = {
    "long_name": "Palmer Drought Severity Index",
    "units": "",
    "references": "Palmer, W. C. (1965). Meteorological Drought. U.S. Weather Bureau Research Paper 45.",
}
CF_METADATA["phdi"] = {
    "long_name": "Palmer Hydrological Drought Index",
    "units": "",
    "references": "Palmer, W. C. (1965). Meteorological Drought.",
}
CF_METADATA["pmdi"] = {
    "long_name": "Palmer Modified Drought Index",
    "units": "",
    "references": "Heddinghaus, T. R., & Sabol, P. (1991). ...",
}
CF_METADATA["z_index"] = {
    "long_name": "Palmer Z-Index",
    "units": "",
    "references": "Palmer, W. C. (1965). Meteorological Drought.",
}
```

**Testing**
- NumPy vs xarray equivalence (1e-8 tolerance)
- Scalar AWC vs DataArray AWC
- params_dict JSON serialization round-trip
- NetCDF write/read preservation
- Performance benchmark vs baseline

**Deliverables:**
- `src/climate_indices/palmer_xarray.py` (new)
- CF metadata entries in `xarray_adapter.py`
- `@overload` signatures in `typed_public_api.py`
- `tests/test_palmer_xarray.py` (unit + integration)
- `docs/palmer_xarray.md` (usage guide)

**Success Criteria:**
- ✅ Dataset return is type-safe (mypy --strict passes)
- ✅ All 4 variables have independent CF metadata
- ✅ Performance ≥80% of multiprocessing baseline
- ✅ NumPy equivalence within 1e-8
- ✅ AWC validation prevents time-dimension errors

**Out of Scope (Track 3):**
- Dask parallelization optimization (sequential time constraint documented)
- Multi-output decorator extraction (deferred until 2nd multi-output index exists)

---

## Step 9: Functional Requirements (v2.4.0 New/Modified FRs Only)

_Note: All FRs from PRD v1.1 (FR-CALC-001 through FR-PKG-004, total 60) remain in effect. Below are NEW or MODIFIED requirements for v2.4.0._

### Track 0: Canonical Pattern Completion Functional Requirements

#### FR-PATTERN-001: percentage_of_normal xarray + CF metadata
**Requirement:** Apply `@xarray_adapter` to `percentage_of_normal()` with CF-compliant metadata

**Acceptance Criteria:**
- Decorator: `@xarray_adapter` applied to existing numpy function (zero algorithm changes)
- CF metadata: `long_name="Percent of Normal Precipitation"`, `units="%"`
- References attribute: Document as climatological anomaly method
- Equivalence test: `test_percentage_of_normal_xarray_equivalence()` passes (tolerance: 1e-8)
- Coordinate preservation: Input DataArray coords/attrs propagated to output

---

#### FR-PATTERN-002: pci xarray + CF metadata
**Requirement:** Apply `@xarray_adapter` to `pci()` with CF-compliant metadata

**Acceptance Criteria:**
- Decorator: `@xarray_adapter` applied to existing numpy function
- CF metadata: `long_name="Precipitation Concentration Index"`, `units=""` (dimensionless)
- References attribute: Cite Oliver (1980) methodology
- Equivalence test: `test_pci_xarray_equivalence()` passes (tolerance: 1e-8)
- Input validation: Requires 365 or 366 daily values (raise `InvalidArgumentError` otherwise)

---

#### FR-PATTERN-003: eto_thornthwaite typed_public_api entry
**Requirement:** Add `@overload` signatures for `eto_thornthwaite()` in `typed_public_api.py`

**Acceptance Criteria:**
- NumPy overload: `eto_thornthwaite(np.ndarray, ...) -> np.ndarray`
- xarray overload: `eto_thornthwaite(xr.DataArray, ...) -> xr.DataArray`
- Runtime dispatcher: `if isinstance(temperature_celsius, xr.DataArray): ...`
- mypy --strict validation passes
- Follows SPI/SPEI `@overload` pattern exactly

---

#### FR-PATTERN-004: eto_hargreaves typed_public_api entry
**Requirement:** Add `@overload` signatures for `eto_hargreaves()` in `typed_public_api.py`

**Acceptance Criteria:**
- NumPy overload: `eto_hargreaves(np.ndarray, np.ndarray, ...) -> np.ndarray`
- xarray overload: `eto_hargreaves(xr.DataArray, xr.DataArray, ...) -> xr.DataArray`
- Runtime dispatcher based on input type detection
- mypy --strict validation passes

---

#### FR-PATTERN-005: percentage_of_normal typed_public_api entry
**Requirement:** Add `@overload` signatures for `percentage_of_normal()` in `typed_public_api.py`

**Acceptance Criteria:**
- NumPy overload: `percentage_of_normal(np.ndarray, ...) -> np.ndarray`
- xarray overload: `percentage_of_normal(xr.DataArray, ...) -> xr.DataArray`
- Runtime dispatcher validates input type
- mypy --strict validation passes

---

#### FR-PATTERN-006: pci typed_public_api entry
**Requirement:** Add `@overload` signatures for `pci()` in `typed_public_api.py`

**Acceptance Criteria:**
- NumPy overload: `pci(np.ndarray) -> np.ndarray`
- xarray overload: `pci(xr.DataArray) -> xr.DataArray`
- Runtime dispatcher based on input type
- mypy --strict validation passes

---

#### FR-PATTERN-007: Palmer structlog migration
**Requirement:** Migrate `palmer.py` from stdlib logging to structlog with lifecycle events

**Acceptance Criteria:**
- Import: Replace `utils.get_logger(__name__, logging.DEBUG)` with `from climate_indices.logging_config import get_logger`
- Lifecycle events: `calculation_started`, `calculation_completed`, `calculation_failed`
- Bind pattern: `_logger.bind(calculation="pdsi", data_shape=precips.shape, awc=awc)`
- Error context: Include computation state in `calculation_failed` events
- Log levels: Match SPI/SPEI pattern (INFO for lifecycle, DEBUG for internal state)
- Zero stdlib logging imports remain in `palmer.py`

---

#### FR-PATTERN-008: eto_thornthwaite structlog lifecycle completion
**Requirement:** Add lifecycle event logging to `eto_thornthwaite()` (currently has logger, missing bind)

**Acceptance Criteria:**
- Bind context: `_logger.bind(calculation="eto_thornthwaite", data_shape=temp.shape, latitude=latitude_degrees)`
- Lifecycle events: `calculation_started`, `calculation_completed` with timing
- Match `eto_hargreaves()` pattern exactly
- Include temperature range stats at DEBUG level

---

#### FR-PATTERN-009: Structured exceptions for all legacy functions
**Requirement:** Replace generic `ValueError` with structured exception hierarchy

**Acceptance Criteria:**
- `percentage_of_normal()`: Input validation → `InvalidArgumentError` with context
- `pci()`: Daily data requirement (365/366) → `InvalidArgumentError("PCI requires exactly 365 or 366 daily precipitation values", shape=data.shape, expected_length=[365, 366])`
- `eto_thornthwaite()`: Parameter validation → `InvalidArgumentError`
- `eto_hargreaves()`: Parameter validation → `InvalidArgumentError`
- `pdsi()`: All validations → structured exceptions (AWC, precip/PET shape mismatch, etc.)
- All exceptions inherit from `ClimateIndicesError` base class
- Error messages provide actionable guidance (not just "invalid value")

---

#### FR-PATTERN-010: percentage_of_normal property-based tests
**Requirement:** Add property-based tests documenting mathematical invariants

**Acceptance Criteria:**
- Test module: `tests/test_properties.py::TestPercentageOfNormalProperties`
- Property: Boundedness — `pnp >= 0` always (precipitation is non-negative)
- Property: Shape preservation — `output.shape == input.shape`
- Property: NaN propagation — `np.isnan(input[i]) → np.isnan(output[i])`
- Property: Linear scaling — `pnp(2×p, 2×p_mean) = pnp(p, p_mean)`
- Hypothesis strategy: `st.floats(min_value=0, max_value=1000, allow_nan=True, allow_infinity=False)`

---

#### FR-PATTERN-011: pci property-based tests
**Requirement:** Add property-based tests for Precipitation Concentration Index

**Acceptance Criteria:**
- Test module: `tests/test_properties.py::TestPCIProperties`
- Property: Range — `0 <= pci <= 100` always
- Property: Input length validation — Raises `InvalidArgumentError` if not 365/366 values
- Property: NaN handling — Single NaN in input propagates to NaN output
- Property: Zero precipitation — All-zero input returns valid PCI (edge case validation)
- Hypothesis strategy: Daily precipitation (365 values)

---

#### FR-PATTERN-012: Expanded SPEI + Palmer property-based tests
**Requirement:** Expand property test coverage beyond current boundedness tests

**Acceptance Criteria:**
- **SPEI additions:**
  - Property: Shape preservation
  - Property: NaN propagation from water balance input
  - Property: Zero water balance → SPEI near 0 (neutral condition)
- **Palmer additions (PHDI, PMDI, Z-Index):**
  - Property: PHDI bounded range (currently only PDSI tested)
  - Property: PMDI bounded range
  - Property: Z-Index bounded range
  - Property: Sequential consistency (splitting time series changes results — not embarrassingly parallel)
- Test module: `tests/test_properties.py::TestSPEIProperties`, `::TestPalmerProperties`

---

### Track 1: PM-ET Functional Requirements

#### FR-PM-001: Penman-Monteith FAO56 Core Calculation
**Requirement:** Implement FAO56 Equation 6 for reference evapotranspiration (ETo)

**Acceptance Criteria:**
- Public function: `eto_penman_monteith(tmin, tmax, tmean, wind_2m, net_radiation, latitude, altitude, ...)`
- Returns: `np.ndarray` or `xr.DataArray` (type-safe via `@overload`)
- Units: mm/day (output), validates input units via docstring guidance
- Handles humidity inputs via priority hierarchy (dewpoint > RH extremes > RH mean)

---

#### FR-PM-002: Atmospheric Parameter Helpers (Equations 7-8)
**Requirement:** Implement private helper functions for pressure and psychrometric constant

**Acceptance Criteria:**
- `_atm_pressure(altitude)` — Eq. 7: `P = 101.3 × [(293 - 0.0065z)/293]^5.26`
- `_psy_const(pressure)` — Eq. 8: `γ = 0.000665 × P`
- Type annotations: `float → float` (scalar operations)
- Testable independently with known values (e.g., Uccle at 100m → 100.1 kPa)

---

#### FR-PM-003: Vapor Pressure Helpers (Equations 11-13)
**Requirement:** Implement SVP calculations with correct Magnus constants

**Acceptance Criteria:**
- `_svp_from_t(temp)` — Eq. 11: `e°(T) = 0.6108 × exp[17.27T/(T+237.3)]`
- `_mean_svp(tmin, tmax)` — Eq. 12: Averages SVP at extremes (NOT `e°(Tmean)`)
- `_slope_svp(temp)` — Eq. 13: `Δ = 4098 × e°(T) / (T+237.3)²`
- Array-compatible: accept `np.ndarray`, return same shape
- Critical precision: exact FAO56 constants (`0.6108, 17.27, 237.3`)

---

#### FR-PM-004: Humidity Pathway Dispatcher (Equations 14-19)
**Requirement:** Auto-select vapor pressure calculation method based on available inputs

**Acceptance Criteria:**
- Priority order:
  1. `dewpoint_celsius` → Eq. 14: `ea = e°(Tdew)`
  2. `rh_max` + `rh_min` → Eq. 17: `ea = [e°(Tmin)×RHmax + e°(Tmax)×RHmin] / 200`
  3. `rh_mean` → Eq. 19: `ea = es × RHmean / 100`
- Raise `ValueError` if no humidity input provided
- Log selected pathway at DEBUG level (structlog)

---

#### FR-PM-005: FAO56 Worked Example Validation
**Requirement:** Reproduce published FAO56 Examples 17 & 18

**Acceptance Criteria:**
- Example 17 (Bangkok, April): `ETo = 5.72 mm/day ±0.05`
- Example 18 (Uccle, 6 July): `ETo = 3.9 mm/day ±0.05`
- Test implementation: `tests/test_eto.py::test_fao56_example_17_bangkok()`
- Test implementation: `tests/test_eto.py::test_fao56_example_18_uccle()`
- Input data embedded in test (no external files)

---

#### FR-PM-006: PM-ET xarray Adapter
**Requirement:** Support xarray DataArray inputs with metadata preservation

**Acceptance Criteria:**
- Wrap via existing `@xarray_adapter` pattern (from v1.1)
- CF metadata: `long_name="Reference Evapotranspiration (Penman-Monteith FAO56)"`, `units="mm day-1"`
- References attribute includes Allen et al. 1998 DOI
- Coordinate preservation for gridded inputs
- Dask compatibility (`dask="parallelized"` in apply_ufunc)

---

### Track 2: EDDI/PNP/scPDSI Functional Requirements

#### FR-EDDI-001: NOAA Reference Dataset Validation (BLOCKING)
**Requirement:** Validate EDDI outputs against NOAA PSL reference data

**Acceptance Criteria:**
- Test module: `tests/test_reference_validation.py`
- Reference dataset: `tests/data/reference/eddi_noaa_reference.nc`
- Tolerance: `np.testing.assert_allclose(computed, noaa_ref, rtol=1e-5, atol=1e-5)`
- Provenance: Dataset includes `source`, `url`, `download_date`, `subset_description` attributes
- **Satisfies:** FR-TEST-004 (Architecture v1.1 Pattern 8)

---

#### FR-EDDI-002: EDDI xarray Adapter
**Requirement:** Apply `@xarray_adapter` decorator for EDDI

**Acceptance Criteria:**
- CF metadata: `long_name="Evaporative Demand Drought Index"`, `units=""`
- Standard name: `"atmosphere_water_vapor_evaporative_demand_anomaly"` (custom, not in CF table)
- References: Hobbins et al. (2016) DOI: `10.1175/JHM-D-15-0121.1`
- Equivalence test: numpy path == xarray path within 1e-8

---

#### FR-EDDI-003: EDDI CLI Integration (Issue #414)
**Requirement:** Add EDDI to `process_climate_indices` CLI

**Acceptance Criteria:**
- Flag: `--index eddi`
- Parameter: `--pet_file <path>` for PET input netCDF
- Help text documents PET method recommendation (PM FAO56)
- Example command in README or docs

---

#### FR-EDDI-004: EDDI PET Method Documentation
**Requirement:** Document PM FAO56 recommendation in EDDI docstring

**Acceptance Criteria:**
- Note section: `"EDDI is most accurate when using Penman-Monteith FAO56 reference evapotranspiration (E0)."`
- Warning: `"Using simplified methods like Thornthwaite may produce inaccurate drought signals."`
- See Also: Cross-reference `eto_penman_monteith()` (after Track 1 complete)
- Add Hobbins et al. 2016 citation to References section

---

#### FR-PNP-001: PNP xarray Adapter
**Requirement:** Add Percent of Normal Precipitation with xarray support

**Acceptance Criteria:**
- Public function: `pnp(precip, scale, ...)` with NumPy and xarray paths
- CF metadata: `long_name="Percent of Normal Precipitation"`, `units="%"`
- References: Document as simple climatological anomaly method
- Equivalence test: numpy == xarray within 1e-8

---

#### FR-SCPDSI-001: scPDSI Stub Interface
**Requirement:** Define scPDSI function signature with NotImplementedError

**Acceptance Criteria:**
- Function: `scpdsi(precip, pet, awc, ...)` in `indices.py`
- Raises: `NotImplementedError("scPDSI implementation planned for future release")`
- Docstring: Methodology overview, Wells et al. (2004) reference
- Type annotations: `@overload` for future numpy/xarray dispatch

---

### Track 3: Palmer Multi-Output Functional Requirements

#### FR-PALMER-001: palmer_xarray() Manual Wrapper
**Requirement:** Implement manual wrapper (NOT decorator) for Palmer multi-output

**Acceptance Criteria:**
- Module: `src/climate_indices/palmer_xarray.py` (~150 lines)
- Function: `palmer_xarray(precip_da, pet_da, awc, ...)` returns `xr.Dataset`
- Pattern: Stack outputs via `np.stack([pdsi, phdi, pmdi, z], axis=0)`, unpack to Dataset
- Rationale documented: Multi-output + params_dict requires custom handling (Pattern C from research)

---

#### FR-PALMER-002: Multi-Output Dataset Return
**Requirement:** Return `xr.Dataset` with 4 variables (not tuple)

**Acceptance Criteria:**
- Variables: `pdsi`, `phdi`, `pmdi`, `z_index` (each is `xr.DataArray`)
- Independent CF metadata per variable (from registry)
- NetCDF-ready (write/read preserves structure)
- Type-safe: `palmer_xarray() -> xr.Dataset` in `@overload`

---

#### FR-PALMER-003: AWC Spatial Parameter Handling
**Requirement:** Support scalar and spatial AWC with dimension validation

**Acceptance Criteria:**
- Accept: `float` (uniform) or `xr.DataArray` (spatially varying)
- Validation: If DataArray, `time_dim NOT in awc.dims` (raise `ValueError` otherwise)
- Error message: `"AWC must not have time dimension '{time_dim}'. AWC is a soil property (spatial only)."`
- `input_core_dims`: `[["time"], ["time"], []]` for precip, pet, awc in apply_ufunc

---

#### FR-PALMER-004: params_dict JSON Serialization
**Requirement:** Store calibration params in Dataset attrs with dual access

**Acceptance Criteria:**
- JSON string: `ds.attrs["palmer_params"] = json.dumps({"alpha": ..., "beta": ..., ...})`
- Individual attrs: `ds.attrs["palmer_alpha"]`, `ds.attrs["palmer_beta"]`, etc.
- Computation: Params computed once from first grid cell (spatially constant)
- Round-trip: `json.loads(ds.attrs["palmer_params"])` reconstructs dict

---

#### FR-PALMER-005: Palmer CF Metadata Registry
**Requirement:** Add CF attributes for all 4 Palmer variables

**Acceptance Criteria:**
- Entries in `CF_METADATA` dict for `pdsi`, `phdi`, `pmdi`, `z_index`
- Each includes: `long_name`, `units=""`, `references` (Palmer 1965, Heddinghaus 1991 for PMDI)
- Applied via `assign_attrs()` in Dataset construction

---

#### FR-PALMER-006: typed_public_api @overload Signatures
**Requirement:** Type-safe dispatch for numpy tuple vs xarray Dataset

**Acceptance Criteria:**
- NumPy overload: `pdsi(np.ndarray, ...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict | None]`
- xarray overload: `pdsi(xr.DataArray, ...) -> xr.Dataset`
- Runtime dispatcher: `if isinstance(precips, xr.DataArray): return palmer_xarray(...)`
- mypy --strict validation passes

---

#### FR-PALMER-007: NumPy vs xarray Equivalence Tests
**Requirement:** Validate numerical equivalence of both paths

**Acceptance Criteria:**
- Test: `test_palmer_xarray_equivalence()` compares numpy tuple output vs xarray Dataset values
- Tolerance: `atol=1e-8` for float64
- Coverage: All 4 variables (PDSI, PHDI, PMDI, Z-Index)
- Includes edge cases: scalar AWC, DataArray AWC, missing data

---

## Step 10: Non-Functional Requirements (v2.4.0 Updates)

_Note: All NFRs from PRD v1.1 (NFR-PERF-001 through NFR-MAINT-005, total 23) remain in effect. Below are NEW or MODIFIED requirements for v2.4.0._

### Track 0: Pattern Compliance & Refactoring Safety

#### NFR-PATTERN-EQUIV: Numerical Equivalence During Refactoring
**Requirement:** All Track 0 pattern applications preserve numerical equivalence

**Metric:** Before/after equivalence tests with tolerance 1e-8 for float64
**Measurement Method:**
- Capture baseline outputs with existing fixtures before applying pattern
- Apply pattern (xarray adapter, exception migration, structlog, etc.)
- Validate `np.testing.assert_allclose(before, after, atol=1e-8, rtol=1e-8)`
- Run full test suite to detect any algorithmic drift

**Acceptance Criteria:**
- 100% of pattern applications pass equivalence tests
- Zero algorithmic changes allowed during refactoring
- If equivalence test fails, revert pattern application and investigate
- Document any intentional behavior changes separately (none expected for Track 0)

**Rationale:** Scientific computing requires bit-exact reproducibility. Pattern refactoring MUST NOT alter numerical results.

---

#### NFR-PATTERN-COVERAGE: 100% Pattern Compliance Dashboard
**Requirement:** Achieve 100% canonical pattern coverage across all public indices

**Metric:** Pattern compliance scorecard tracked per index
**Measurement Method:**
- Track 6 patterns × 7 indices = 42 compliance points
- Pattern categories: xarray support, type safety, CF metadata, structlog, exceptions, property tests
- Generate compliance matrix in CI

**Acceptance Criteria:**
- xarray support: 7/7 indices (100%)
- Type safety (`@overload`): 7/7 indices (100%)
- CF metadata: 7/7 xarray outputs (100%)
- structlog: 7/7 indices with lifecycle events (100%)
- Structured exceptions: 100% of validation code
- Property-based tests: 7/7 indices (100%)

**Dashboard Example:**
```
| Index               | xarray | typed_api | CF_meta | structlog | exceptions | prop_tests |
|---------------------|:------:|:---------:|:-------:|:---------:|:----------:|:----------:|
| spi                 |   ✓    |     ✓     |    ✓    |     ✓     |     ✓      |     ✓      |
| spei                |   ✓    |     ✓     |    ✓    |     ✓     |     ✓      |     ✓      |
| percentage_of_normal|   ✓    |     ✓     |    ✓    |     ✓     |     ✓      |     ✓      |
| pci                 |   ✓    |     ✓     |    ✓    |     ✓     |     ✓      |     ✓      |
| eto_thornthwaite    |   ✓    |     ✓     |    ✓    |     ✓     |     ✓      |     ✓      |
| eto_hargreaves      |   ✓    |     ✓     |    ✓    |     ✓     |     ✓      |     ✓      |
| pdsi (multi-output) |   ✓    |     ✓     |    ✓    |     ✓     |     ✓      |     ✓      |
```

---

#### NFR-PATTERN-MAINT: Maintainability Through Consistency
**Requirement:** Pattern consistency reduces maintenance burden and onboarding time

**Metric:** Contributor onboarding time, bug fix velocity
**Measurement Method:**
- Track time-to-first-PR for new contributors (before: ~2 weeks, target: ~3 days)
- Track average time-to-fix for user-reported bugs (target: 30% reduction)
- Document pattern consistency impact in contribution guide

**Acceptance Criteria:**
- All indices follow identical code organization patterns
- New contributors can reference ANY index as a pattern example
- Bug fixes in one index inform fixes in others (pattern reuse)
- Code review checklist references canonical patterns

**Rationale:** Consistency is a maintainability multiplier—learn once, apply everywhere.

---

### Performance

#### NFR-PM-PERF: Penman-Monteith Numerical Precision
**Requirement:** PM-ET maintains scientific accuracy per FAO56 standard

**Metric:** FAO56 examples within 0.05 mm/day, intermediate values within 0.01 kPa
**Measurement Method:**
- Validate Eq. 7-13 helpers independently (e.g., `_svp_from_t(21.5°C) = 2.564 kPa ±0.01`)
- End-to-end: Example 17 & 18 within ±0.05 mm/day
- Document floating-point precision assumptions (float64 required)

**Acceptance Criteria:**
- Bangkok (tropical, monthly): 5.72 mm/day ±0.05
- Uccle (temperate, daily): 3.9 mm/day ±0.05
- No systematic bias across climate regimes

---

#### NFR-PALMER-SEQ: Palmer Sequential Time Constraint
**Requirement:** Palmer computation respects sequential time dependency

**Metric:** Dask chunking guidance documented, time dimension NOT parallelized
**Implementation:**
- Document in user guide: `"Palmer requires full time series per grid cell. Chunk spatially (lat, lon), NOT temporally."`
- Example: `precip.chunk({"time": -1, "lat": 50, "lon": 50})`
- `apply_ufunc` uses `vectorize=True` (Python loop over spatial) instead of `dask="parallelized"` along time

**Acceptance Criteria:**
- Documentation includes chunking guidance
- Tests validate that chunked time raises warning or produces incorrect results

---

#### NFR-PALMER-PERF: Palmer xarray Performance Target
**Requirement:** xarray path ≥80% speed of multiprocessing CLI baseline

**Metric:** Benchmark synthetic gridded dataset (360×180×240 monthly, 10 years, 200 grid cells)
**Measurement Method:**
- Baseline: Current `__main__.py` multiprocessing over grid cells
- xarray: `palmer_xarray()` with `vectorize=True`
- Measure: wall-clock time, report ratio

**Acceptance Criteria:**
- Median of 10 runs: xarray ≥ 80% of baseline speed
- Document performance characteristics (Python loop overhead acceptable per research)

---

### Reliability

#### NFR-MULTI-OUT: Multi-Output Adapter Pattern Stability
**Requirement:** Stack/unpack workaround for `apply_ufunc` multi-output limitation

**Metric:** Documented pattern, monitor xarray Issue #1815 for native support
**Implementation:**
- Use `np.stack([pdsi, phdi, pmdi, z], axis=0)` → `output_core_dims=[["variable", "time"]]`
- Unpack via `.isel(variable=N).drop_vars("variable")`
- Comment in code: `"# Workaround for xarray Issue #1815 (dask='parallelized' + multi-output not supported)"`

**Acceptance Criteria:**
- Pattern documented in Palmer xarray user guide
- Code comment references Issue #1815
- If/when xarray resolves issue, revisit and refactor (tracked in backlog)

---

#### NFR-EDDI-VAL: EDDI NOAA Reference Validation Tolerance
**Requirement:** EDDI validation uses appropriate tolerance for non-parametric ranking

**Metric:** 1e-5 relative/absolute tolerance (looser than equivalence tests)
**Rationale:** Non-parametric empirical ranking has different FP accumulation than parametric distribution fitting
**Documentation:** Tolerance rationale in test docstring

**Acceptance Criteria:**
- `assert_allclose(computed, noaa_ref, rtol=1e-5, atol=1e-5)` in test
- Docstring explains why 1e-5 (not 1e-8 like SPI/SPEI equivalence tests)

---

## Step 11: Document Complete (Summary & Next Steps)

This PRD v2.4.0 is now complete with:

- **Strategic Context:** 4-track architecture (Track 0 parallel with Track 1, both enabling Tracks 2 & 3)
- **Domain Requirements:** PM FAO56 scientific correctness, Palmer multi-output patterns, EDDI NOAA validation, Track 0 numerical equivalence
- **30 New Functional Requirements:** Organized across 4 tracks (Pattern Completion, PM-ET, EDDI/PNP/scPDSI, Palmer multi-output)
- **8 New/Modified Non-Functional Requirements:** Pattern compliance, refactoring safety, performance targets, reliability patterns
- **Research Integration:** All findings from 3 technical research documents incorporated + codebase inventory for pattern gaps

**Document Statistics:**
- **v2.4.0 New FRs:** 30 total
  - Track 0 (Pattern Completion): 12 FRs (FR-PATTERN-001 through FR-PATTERN-012)
  - Track 1 (PM-ET): 6 FRs (FR-PM-001 through FR-PM-006)
  - Track 2 (EDDI/PNP/scPDSI): 5 FRs (FR-EDDI-001 through FR-SCPDSI-001)
  - Track 3 (Palmer): 7 FRs (FR-PALMER-001 through FR-PALMER-007)
- **v2.4.0 New/Modified NFRs:** 8 total
  - Track 0: 3 NFRs (NFR-PATTERN-EQUIV, NFR-PATTERN-COVERAGE, NFR-PATTERN-MAINT)
  - Track 1: 1 NFR (NFR-PM-PERF)
  - Track 2: 1 NFR (NFR-EDDI-VAL)
  - Track 3: 3 NFRs (NFR-PALMER-SEQ, NFR-PALMER-PERF, NFR-MULTI-OUT)
- **Total Requirements (v1.1 + v2.4.0):** 90 FRs, 31 NFRs
- **Track Dependencies:** (Track 0 ∥ Track 1) → (Track 2 ∥ Track 3)
  - Track 0 Palmer structlog partially blocks Track 3
  - Track 0 PNP xarray enables Track 2
- **Estimated Duration:** 10-14 weeks
  - Track 0: 2-3 weeks (parallel with Track 1)
  - Track 1: 3-4 weeks (parallel with Track 0)
  - Track 2: 2-3 weeks (after Track 0 + Track 1 complete, parallel with Track 3)
  - Track 3: 3-4 weeks (after Track 0 Palmer + Track 1 complete, parallel with Track 2)

**Research Traceability:**
- **PM FAO56 Research:** Informs FR-PM-001 through FR-PM-006, NFR-PM-PERF
- **Palmer Modernization Research:** Informs FR-PALMER-001 through FR-PALMER-007, NFR-PALMER-SEQ, NFR-MULTI-OUT
- **EDDI Validation Research:** Informs FR-EDDI-001, FR-EDDI-004, NFR-EDDI-VAL
- **Codebase Inventory (Track 0):** Informs FR-PATTERN-001 through FR-PATTERN-012, NFR-PATTERN-* requirements

**Success Criteria Per Track:**
- **Track 0:** 100% pattern compliance (7/7 indices for all 6 patterns), numerical equivalence (1e-8), mypy --strict passes
- **Track 1:** FAO56 Examples 17 & 18 within ±0.05 mm/day, mypy --strict passes
- **Track 2:** EDDI validates against NOAA reference (1e-5), PNP equivalence (1e-8), scPDSI stub documented
- **Track 3:** Palmer Dataset return type-safe, ≥80% performance, NumPy equivalence (1e-8)

**Next BMAD Workflows:**
- **Epic Breakdown:** Decompose tracks into stories
- **Implementation Readiness Check:** Validate PRD + Architecture alignment
- **Implementation:** Begin Track 1 (PM-ET Foundation)

---

## Appendix: Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-15 | 2.4.0 | Initial PRD v2.4.0 complete — PM FAO56, Palmer multi-output, EDDI compliance |
| 2026-02-05 | 1.1 | Added EDDI NOAA reference validation (FR-TEST-004), Phase 2 testing scope |
| 2026-02-05 | 1.0 | Initial PRD complete (Steps 1-11) — xarray + structlog modernization |


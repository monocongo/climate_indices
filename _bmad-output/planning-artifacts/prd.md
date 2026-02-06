---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
inputDocuments:
  - '_bmad-output/planning-artifacts/context.md'
  - 'docs/bmad/EDDI-BMAD-Retrospective.md'
workflowType: 'prd'
---

# Product Requirements Document - climate_indices xarray Integration + structlog Modernization

**Author:** James A.
**Date:** 2026-02-05
**Status:** In Progress (Steps 1-8 Complete)

---

## Executive Summary

This PRD defines requirements for modernizing the `climate_indices` library with xarray support and structured logging. The project addresses user pain points around array handling ergonomics while maintaining numerical fidelity with the existing NumPy implementation. This is a **brownfield modernization** of a mature scientific library used by NOAA, research institutions, and operational drought monitoring systems.

**Key Changes:**
- Add native xarray support via adapter pattern (preserves existing NumPy API)
- Replace basic logging with structlog for structured observability
- Maintain strict numerical equivalence with existing implementation
- Phased delivery: MVP (SPI/SPEI/PET) → Phase 2 (EDDI/PNP/CLI) → Phase 3 (Palmer + deprecation)

---

## Step 1: Initialization

**Project Type:** Brownfield Enhancement
**Context Source:** 2 input documents
- `_bmad-output/planning-artifacts/context.md` — feature goals, technical constraints
- `docs/bmad/EDDI-BMAD-Retrospective.md` — lessons learned from EDDI implementation

**Stakeholders:**
- **Author/Maintainer:** James A. (sole developer)
- **Primary Users:** Climate researchers, drought monitoring agencies (NIDIS, NOAA)
- **Community:** Open-source contributors, scientific Python ecosystem

---

## Step 2: Project Classification

**Primary Classification:** Developer Tool / Library
**Domain:** Scientific Computing (Climate Science)
**Complexity:** Medium

**Rationale:**
- **Developer Tool:** Provides programmatic API for climate index calculations (not end-user application)
- **Scientific Domain:** Implements WMO-standardized drought indices with numerical reproducibility requirements
- **Medium Complexity:**
  - Dual API surface (NumPy + xarray)
  - Metadata preservation (CF conventions)
  - Numerical equivalence testing
  - Not "High" because indices are mostly single-pass statistical operations (no multi-year state machines like Palmer)

---

## Step 3: Success Criteria

### User Success
**Definition:** Researchers and operational users can process climate data with modern tooling while maintaining scientific validity.

**Measurable Outcomes:**
1. **Adoption:** 30% of new integrations use xarray API within 6 months of GA
2. **Friction Reduction:** xarray users eliminate manual `.values` extraction and metadata re-attachment
3. **Zero Regressions:** Existing NumPy users continue without code changes
4. **Improved Debugging:** structlog enables correlation of index calculations with upstream data pipelines

### Business Success
**Definition:** Project sustainability and community health improve.

**Measurable Outcomes:**
1. **Maintainability:** Time-to-debug production issues reduced by 40% (via structured logs)
2. **Contribution Velocity:** External PRs increase due to clearer code organization
3. **Ecosystem Alignment:** Featured in xarray/pangeo documentation as reference implementation
4. **Risk Mitigation:** No breaking changes → preserves operational stability for NOAA users

### Technical Success
**Definition:** Implementation is robust, testable, and scientifically sound.

**Measurable Outcomes:**
1. **Numerical Fidelity:** Automated regression tests confirm bit-exact equivalence between NumPy/xarray paths (tolerance: 1e-8 for FP64)
2. **Performance:** xarray overhead < 5% for chunked operations (validated via benchmarks)
3. **Metadata Integrity:** CF convention compliance tested via cf-checker integration
4. **Type Safety:** mypy --strict passes on all new code

---

### Scope Horizons

#### MVP Scope (Weeks 1-4)
**Goal:** Prove xarray adapter pattern + structlog integration

**Included:**
- SPI (Standardized Precipitation Index) — xarray support
- SPEI (Standardized Precipitation Evapotranspiration Index) — xarray support
- PET (Potential Evapotranspiration) — Thornthwaite + Hargreaves variants, xarray support
- structlog configuration (JSON output for files, human-readable console)
- Automated equivalence testing framework
- Documentation: xarray quickstart guide

**Out of Scope:**
- EDDI, PNP indices (Phase 2)
- Palmer indices (Phase 3 — requires special handling)
- conda-forge packaging (post-MVP)
- CLI refactoring (Phase 2)

**Success Gate:** External reviewer confirms SPI calculation matches reference implementation on real dataset

#### Growth Scope (Phase 2)
**Goal:** Expand xarray coverage, improve developer ergonomics

**Included:**
- EDDI (xarray support)
- PNP (xarray support)
- CLI modernization (Click migration)
- Advanced metadata: provenance tracking via CF `history` attribute
- conda-forge packaging

**Out of Scope:**
- Palmer indices (deferred to Phase 3)
- Breaking API changes

#### Vision Scope (Phase 3)
**Goal:** Complete migration, deprecate legacy paths

**Included:**
- Palmer indices (PDSI, PHDI, Z-Index) — requires multi-year state handling
- NumPy deprecation warnings
- Migration guide for legacy users
- Performance optimization (Dask/Numba integration)

---

## Step 4: User Journeys

### Journey 1: Climate Researcher (Primary Persona)
**Actor:** Dr. Sarah Chen, climate scientist analyzing regional drought trends

**Context:**
- Works with multi-decadal climate model outputs (CMIP6 netCDF files)
- Needs to calculate SPI/SPEI across ensemble members (5D: time × lat × lon × ensemble × scenario)
- Currently writes custom scripts to strip metadata, calculate indices, then manually re-attach coordinates

**Current Pain Points:**
1. Manual metadata management is error-prone (misaligned coordinates after computation)
2. Code duplication across projects (each analysis re-implements the same xarray wrapper)
3. Debugging failures is difficult (basic logging doesn't correlate with upstream pipeline stages)

**Desired Outcome:**
```python
import xarray as xr
from climate_indices import spi

ds = xr.open_dataset("cmip6_pr_ensemble.nc")
# One-liner, preserves all metadata
ds_drought = spi.spi(ds['pr'], scale=3, distribution="gamma")
```

**Success Criteria:**
- Calculation completes without `.values` extraction
- Output preserves coordinates, attributes, chunking
- Structured logs show which ensemble member/scenario failed (if error occurs)

---

### Journey 2: Operational Drought Monitor (Secondary Persona)
**Actor:** NOAA NIDIS technician running weekly EDDI updates

**Context:**
- Production pipeline: ingest → index calculation → visualization → publication
- Zero tolerance for numerical changes (breaks trend analysis)
- Logs must integrate with centralized ELK stack (requires JSON format)

**Current Pain Points:**
1. Basic logging makes post-mortem analysis difficult (no structured context)
2. Cannot correlate failures across pipeline stages
3. Manual log parsing for anomaly detection

**Desired Outcome:**
- Upgrade to structlog without changing index values
- JSON logs include correlation IDs, input metadata, timing
- Existing NumPy API continues to work (no refactoring required)

**Success Criteria:**
- Existing `eddi.eddi(precip, ...)` calls produce identical results (bit-exact)
- Logs are machine-parseable (ELK can filter by index type, scale, region)
- Migration takes < 1 day (drop-in replacement)

---

### Journey 3: Graduate Student (Learning Persona)
**Actor:** Alex, MS student learning drought analysis

**Context:**
- Familiar with xarray from coursework
- Intimidated by climate_indices documentation (assumes NumPy expertise)
- Wants to calculate SPEI for thesis without deep diving into meteorology

**Current Pain Points:**
1. Examples require NumPy knowledge (array slicing, broadcasting)
2. Unclear how to handle multi-dimensional climate data
3. No guidance on parameter selection (distribution choice, scale)

**Desired Outcome:**
- Clear xarray-first quickstart guide
- Examples show end-to-end workflow (load netCDF → calculate → visualize)
- Error messages guide parameter selection (e.g., "gamma distribution recommended for monthly scale")

**Success Criteria:**
- Student completes first SPEI calculation in < 30 minutes
- Code is < 10 lines (excluding plotting)

---

### Journey 4: Open-Source Contributor (Ecosystem Persona)
**Actor:** Kai, pangeo community member

**Context:**
- Wants to add Dask optimization to SPI
- Notices code organization makes it hard to find adapter logic
- Unclear how to run equivalence tests

**Current Pain Points:**
1. No contributor guide for xarray path
2. Test suite organization unclear (where to add new tests?)
3. Uncertain if PR will break existing users

**Desired Outcome:**
- CONTRIBUTING.md explains adapter pattern design
- Test fixtures make it easy to add xarray test cases
- CI runs equivalence tests automatically

**Success Criteria:**
- Contributor can add xarray test for new index in < 1 hour
- PR review focuses on logic, not test scaffolding

---

### Journey 5: Downstream Package Maintainer (Integration Persona)
**Actor:** Maintainer of `xclim` (Ouranos climate library)

**Context:**
- Considering `climate_indices` as lightweight alternative for specific indices
- Needs to ensure API compatibility with existing xclim workflows
- Concerned about dual NumPy/xarray maintenance burden

**Current Pain Points:**
1. Unclear if xarray support is production-ready
2. Uncertain about metadata conventions (CF compliance?)
3. Worried about version churn (will API stabilize?)

**Desired Outcome:**
- Clear maturity signal (beta tag on xarray features)
- CF compliance documented and tested
- Semantic versioning commitment

**Success Criteria:**
- Can wrap `climate_indices.spi` in xclim without adapter code
- Metadata passes cf-checker validation
- Deprecation policy documented (12-month notice for breaking changes)

---

## Step 5: Domain Requirements

### Scientific Correctness Requirements

1. **Algorithmic Fidelity**
   - **Requirement:** Implementations must match peer-reviewed reference algorithms
   - **Validation:** SPI follows McKee et al. (1993), SPEI follows Vicente-Serrano et al. (2010)
   - **Testing:** Automated comparison against published test datasets (e.g., NOAA reference outputs)

2. **Numerical Reproducibility**
   - **Requirement:** Results must be bit-exact reproducible given same inputs and environment
   - **Rationale:** Operational users require deterministic outputs for trend analysis
   - **Implementation:**
     - Pin NumPy/SciPy versions in lock file
     - Document floating-point tolerance assumptions
     - Warn on non-deterministic operations (e.g., parallel reduction order)

3. **Statistical Validity**
   - **Requirement:** Distribution fitting must handle edge cases (zero-inflation, missing data)
   - **Edge Cases:**
     - Zero-inflated precipitation (gamma + empirical CDF)
     - Insufficient data for fitting (minimum 30 years recommended)
     - Missing values (pairwise deletion vs. listwise)
   - **Testing:** Synthetic datasets with known pathologies

---

### Metadata Requirements (CF Conventions)

4. **Attribute Preservation**
   - **Requirement (MUST):** Calculations preserve input coordinates, dimensions, and chunking
   - **Requirement (SHOULD):** Add CF-compliant attributes to output
     - `long_name`: "Standardized Precipitation Index"
     - `standard_name`: (none defined for SPI, use long_name)
     - `units`: "dimensionless"
     - `references`: DOI to McKee et al. (1993)
   - **Rationale:** Enables interoperability with downstream tools (Panoply, xclim, ILAMB)

5. **Provenance Tracking**
   - **Requirement (SHOULD):** Add `history` attribute with calculation details
   - **Format:** `"2026-02-05T10:23:45Z: SPI-3 calculated using gamma distribution (climate_indices v2.0.0)"`
   - **Deferred to Phase 2:** Full provenance (input sources, parameters, runtime environment)

6. **CF Compliance Testing**
   - **Requirement (MUST for Growth):** Outputs pass `cf-checker` validation
   - **MVP:** Best-effort compliance (no validation gate)
   - **Phase 2:** Integrate cf-checker into CI

---

### Data Handling Requirements

7. **Missing Data Handling**
   - **Requirement:** Support NaN propagation (match NumPy behavior)
   - **User Control:** Allow `skipna` parameter for pairwise vs. listwise deletion
   - **Documentation:** Warn about minimum sample size for distribution fitting

8. **Chunked Computation**
   - **Requirement:** xarray path must work with Dask-backed arrays
   - **MVP Constraint:** Single-pass indices only (SPI, SPEI, PET)
   - **Phase 3:** Multi-pass indices (Palmer) require special Dask handling

9. **Memory Efficiency**
   - **Requirement:** Process datasets larger than RAM (via Dask lazy evaluation)
   - **Performance Target:** < 5% overhead vs. NumPy for in-memory workloads
   - **Testing:** Benchmark with synthetic 10GB dataset (chunked)

---

### Operational Requirements

10. **Logging and Observability**
    - **Requirement:** Structured logs (JSON) include:
      - Index type, scale, distribution
      - Input shape, chunking
      - Computation time
      - Warning/error context (e.g., "gamma fit failed for chunk (lat=10:20, lon=30:40)")
    - **Rationale:** Enables correlation with upstream pipeline failures

11. **Error Handling**
    - **Requirement:** Clear error messages for common mistakes
      - Wrong input dimension (e.g., 1D array when 2D expected)
      - Incompatible parameters (e.g., SPEI without PET)
      - Insufficient data for fitting
    - **Format:** Structured exceptions (not generic ValueError)

---

## Step 6: Innovation Analysis

**Conclusion:** No significant innovation signals detected. Skipped per BMAD workflow guidance.

**Rationale:**
- xarray adapter pattern is proven (see xclim, xskillscore)
- structlog is industry-standard structured logging
- Scientific algorithms are reference implementations (not novel research)
- Focus is on **engineering quality and usability**, not novelty

---

## Step 7: Project-Type Specific Requirements (Developer Tool)

### API Design Requirements

1. **Dual API Strategy**
   - **Requirement:** Maintain 100% backward compatibility with NumPy API
   - **Implementation:** Adapter pattern with input type detection
     ```python
     def spi(data, scale, distribution):
         if isinstance(data, xr.DataArray):
             return _spi_xarray(data, scale, distribution)
         else:
             return _spi_numpy(data, scale, distribution)
     ```
   - **Rationale:** Allows gradual migration without breaking existing users

2. **Type Hints and IDE Support**
   - **Requirement:** All public functions have complete type annotations
   - **Overloads:** Use `@overload` for NumPy vs. xarray return types
     ```python
     @overload
     def spi(data: np.ndarray, ...) -> np.ndarray: ...
     @overload
     def spi(data: xr.DataArray, ...) -> xr.DataArray: ...
     ```
   - **Validation:** mypy --strict passes on all new code

3. **Error Messages**
   - **Requirement:** Guide users toward solutions
     - Bad: `"Invalid distribution parameter"`
     - Good: `"Distribution 'gamm' not recognized. Did you mean 'gamma'? Valid options: gamma, pearson3, lognormal"`
   - **Context:** Include relevant parameter values in exceptions

---

### Documentation Requirements

4. **API Reference**
   - **Requirement:** Auto-generated from docstrings (Sphinx)
   - **Format:** Google-style docstrings with examples
   - **Coverage:** 100% of public functions

5. **User Guide Structure**
   - Quickstart (< 5 minutes, single index example)
   - xarray Guide (migration from NumPy, metadata handling)
   - Algorithm Reference (links to papers, validation datasets)
   - Troubleshooting (common errors, performance tuning)

6. **Example Gallery**
   - **Requirement:** Jupyter notebooks showing end-to-end workflows
   - **MVP Examples:**
     - Basic SPI calculation
     - Multi-dimensional SPEI (climate model ensemble)
     - Drought monitoring workflow (with visualization)

---

### Testing Requirements

7. **Equivalence Testing**
   - **Requirement:** xarray path produces identical results to NumPy path
   - **Implementation:**
     ```python
     @pytest.mark.parametrize("index_func", [spi, spei, pet_thornthwaite])
     def test_xarray_numpy_equivalence(index_func, sample_data):
         xr_result = index_func(xr.DataArray(sample_data), ...)
         np_result = index_func(sample_data, ...)
         assert np.allclose(xr_result.values, np_result, rtol=1e-8)
     ```
   - **Tolerance:** 1e-8 for float64, 1e-5 for float32

8. **Metadata Testing**
   - **Requirement:** Verify coordinate preservation, attribute presence
   - **Scope:** All xarray tests include metadata assertions

9. **Property-Based Testing**
   - **Requirement:** Use Hypothesis for edge case generation
   - **Properties:**
     - Monotonicity (SPI increases with precipitation)
     - Symmetry (SPI distribution centered at 0 for long records)
     - Bounded outputs (PDSI in [-10, 10])

---

### Packaging and Distribution

10. **PyPI Distribution**
    - **Current State:** Already on PyPI (`pip install climate-indices`)
    - **No Changes Required:** Continue existing release process

11. **conda-forge Distribution (Phase 2)**
    - **Requirement:** Submit feedstock after MVP validation
    - **Rationale:** Preferred by scientific Python users (better dependency resolution)
    - **Deferred to Growth:** Wait for xarray API to stabilize

12. **Version Policy**
    - **Semantic Versioning:** Major.Minor.Patch
    - **Breaking Change Policy:** 12-month deprecation notice for API changes
    - **Beta Tagging:** Mark xarray API as `beta` until Phase 2

---

### Observability Requirements

13. **Structured Logging Configuration**
    - **Requirement:** Dual output (JSON for files, human-readable for console)
    - **Format:**
      ```json
      {
        "timestamp": "2026-02-05T10:23:45.123Z",
        "level": "info",
        "event": "spi_calculation_complete",
        "index": "spi",
        "scale": 3,
        "distribution": "gamma",
        "input_shape": [120, 180, 360],
        "duration_ms": 1523
      }
      ```
    - **Context Binding:** Attach correlation IDs for multi-stage pipelines

14. **Performance Metrics**
    - **Requirement:** Log computation time, memory usage
    - **Granularity:** Per-index-call level (not per-operation)
    - **Privacy:** No logging of data values (only shapes, parameters)

---

## Step 8: Scoping and Phasing

### MVP Scope (Weeks 1-4) — "Prove the Pattern"

**Goal:** Validate xarray adapter pattern + structlog integration on simplest indices

**Included Indices:**
- **SPI (Standardized Precipitation Index)**
  - Single input (precipitation)
  - Single-pass calculation (gamma distribution fitting)
  - Most widely used index (high validation priority)
- **SPEI (Standardized Precipitation Evapotranspiration Index)**
  - Two inputs (precipitation, PET)
  - Depends on PET calculation (tests composition)
- **PET (Potential Evapotranspiration)**
  - Thornthwaite method (temperature-based)
  - Hargreaves method (temperature + radiation)
  - Minimal complexity (validates metadata handling)

**Logging:**
- structlog configuration (dual output)
- Context binding (index type, parameters)
- Error context (input shapes, failure modes)

**Testing:**
- Equivalence tests (NumPy vs. xarray)
- Metadata preservation tests
- Edge case validation (zero-inflation, missing data)

**Documentation:**
- xarray quickstart guide
- Migration notes (NumPy → xarray)
- structlog format documentation

**Out of Scope:**
- EDDI, PNP (Phase 2)
- Palmer indices (Phase 3)
- conda-forge (Phase 2)
- CLI refactoring (Phase 2)
- Performance optimization (Phase 3)

**Success Criteria:**
- External reviewer confirms SPI matches NOAA reference on test dataset
- Existing NumPy users see no behavioral changes
- structlog logs integrate with ELK stack (JSON validation)

---

### Phase 2 Scope — "Scale and Polish"

**Goal:** Expand xarray coverage, improve developer ergonomics

**Included Indices:**
- **EDDI (Evaporative Demand Drought Index)**
  - Non-parametric ranking (no distribution fitting)
  - Tests different statistical approach
- **PNP (Percent of Normal Precipitation)**
  - Simplest index (validates minimal metadata handling)

**Tooling:**
- CLI modernization (migrate from argparse to Click)
- Advanced provenance tracking (full `history` attribute)
- conda-forge packaging

**Testing:**
- Expand property-based tests
- Add Dask-specific benchmarks
- CF compliance validation (cf-checker integration)

**Documentation:**
- Contributor guide (adapter pattern design)
- Performance tuning guide (Dask chunking)
- Advanced examples (multi-model ensembles)

**Success Criteria:**
- Community PRs for additional indices
- Featured in pangeo documentation
- conda-forge package available

---

### Phase 3 Scope — "Complete Migration"

**Goal:** Handle complex indices, deprecate legacy paths

**Included Indices:**
- **Palmer Indices (PDSI, PHDI, Z-Index)**
  - Multi-year state tracking (tests xarray multi-pass patterns)
  - Complex waterbudget calculations
  - Requires special Dask handling (cannot be fully lazy)

**Breaking Changes:**
- NumPy deprecation warnings (12-month notice)
- Migration guide for legacy users

**Optimization:**
- Dask graph optimization (reduce intermediate arrays)
- Numba acceleration (critical loops)
- Memory profiling and optimization

**Documentation:**
- Full API stability guarantee
- Performance comparison (vs. xclim)
- Case studies (operational deployments)

**Success Criteria:**
- Zero open issues for xarray path
- Performance parity with xclim for equivalent indices
- At least one operational agency using xarray API in production

---

### Risk Mitigation Strategy

**Technical Risks:**

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Numerical divergence between paths | Medium | Critical | Automated equivalence tests at PR gate with strict tolerances |
| Dask graph explosion on complex indices | Low (MVP indices are simple) | High | MVP scoped to single-pass indices; Palmer deferred |
| `apply_ufunc` signature complexity | Medium | Medium | Prototype on SPI first, validate pattern before scaling |
| structlog breaking existing log consumers | Low | Low | Document format change, provide migration note |

**Market/Community Risks:**

| Risk | Mitigation |
|---|---|
| Users confused by dual API | Clear docs, deprecation warnings with migration instructions |
| Upstream maintainer rejects approach | Early PR discussion, small incremental PRs |
| xclim users see no reason to switch | Focus on "lightweight + same accuracy" messaging |

**Resource Risks:**

| Risk | Mitigation |
|---|---|
| Solo developer bandwidth | AI-assisted development (BMAD), phased delivery |
| Scope creep into Palmer during MVP | Hard scope boundary — Palmer is Phase 3 |
| If blocked on adapter pattern | Fall back to separate namespace (`indices.xr.spi`) instead of dispatch |

---

## Next Steps

**Pending Completion:**
- **Step 9:** Functional Requirements (detailed feature specifications)
- **Step 10:** Non-Functional Requirements (performance, security, compliance)
- **Step 11:** Polish and Validation

**Ready to Continue:** Yes, proceed to Step 9 when ready.

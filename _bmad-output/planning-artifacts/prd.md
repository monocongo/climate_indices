---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
inputDocuments:
  - '_bmad-output/planning-artifacts/context.md'
  - 'docs/bmad/EDDI-BMAD-Retrospective.md'
workflowType: 'prd'
---

# Product Requirements Document - climate_indices xarray Integration + structlog Modernization

**Author:** James A.
**Date:** 2026-02-05
**Status:** Complete (All 11 Steps)

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

**Note:** Detailed scope breakdown and phasing strategy defined in Step 8 below.

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

## Step 9: Functional Requirements

This section defines **what the system must do** to deliver the capabilities described in earlier sections. Requirements are organized by capability area and structured with clear acceptance criteria.

### 1. Index Calculation Capabilities

#### FR-CALC-001: SPI Calculation with xarray
**Requirement:** Support SPI calculation for xarray.DataArray inputs with automatic metadata preservation
**Acceptance Criteria:**
- Accept xarray.DataArray as input with time dimension
- Preserve all coordinates, dimensions, and attributes in output
- Support scales 1-72 months
- Support gamma, pearson3 distributions
- Return xarray.DataArray with same shape as input

#### FR-CALC-002: SPEI Calculation with xarray
**Requirement:** Support SPEI calculation for xarray.DataArray inputs with PET integration
**Acceptance Criteria:**
- Accept precipitation and PET as xarray.DataArray inputs
- Automatically align inputs by coordinates
- Support scales 1-72 months
- Support gamma, pearson3 distributions
- Preserve metadata from precipitation input (primary variable)

#### FR-CALC-003: PET Thornthwaite with xarray
**Requirement:** Calculate PET using Thornthwaite method for xarray inputs
**Acceptance Criteria:**
- Accept temperature and latitude as xarray.DataArray
- Support broadcasting across spatial dimensions
- Add CF-compliant attributes (units: mm/month, long_name)
- Preserve input chunking for Dask arrays

#### FR-CALC-004: PET Hargreaves with xarray
**Requirement:** Calculate PET using Hargreaves method for xarray inputs
**Acceptance Criteria:**
- Accept tmin, tmax, latitude as xarray.DataArray
- Validate coordinate alignment across inputs
- Add CF-compliant attributes
- Support both single-point and gridded calculations

#### FR-CALC-005: Backward Compatibility - NumPy API
**Requirement:** Maintain 100% backward compatibility with existing NumPy API
**Acceptance Criteria:**
- All existing NumPy function signatures work unchanged
- Bit-exact numerical results for NumPy inputs (tolerance: 1e-8)
- No new required parameters for existing functions
- No deprecation warnings in MVP phase

### 2. Input Data Handling

#### FR-INPUT-001: Automatic Input Type Detection
**Requirement:** Automatically detect input type and route to appropriate implementation
**Acceptance Criteria:**
- Detect xarray.DataArray via isinstance check
- Route to xarray path for xarray inputs
- Route to NumPy path for ndarray/list/scalar inputs
- Raise clear error for unsupported types (e.g., pandas.Series)

#### FR-INPUT-002: Coordinate Validation
**Requirement:** Validate that xarray inputs have required dimensions
**Acceptance Criteria:**
- Check for presence of time dimension (configurable name: "time", "date")
- Validate coordinate monotonicity for time dimension
- Raise InsufficientDataError if time series too short for scale
- Provide clear error message listing available dimensions

#### FR-INPUT-003: Multi-Input Alignment
**Requirement:** Automatically align multiple xarray inputs (e.g., SPEI with precip + PET)
**Acceptance Criteria:**
- Use xarray.align() with join='inner' by default
- Warn if alignment drops data (non-overlapping coordinates)
- Preserve chunking from first input
- Document alignment behavior in docstring

#### FR-INPUT-004: Missing Data Handling
**Requirement:** Support NaN handling consistent with NumPy behavior
**Acceptance Criteria:**
- Propagate NaNs through calculations (default behavior)
- Support skipna=True for pairwise deletion
- Warn when >20% missing data in calibration period
- Document minimum sample size requirements (30 years for SPI/SPEI)

#### FR-INPUT-005: Chunked Array Support
**Requirement:** Work with Dask-backed xarray arrays
**Acceptance Criteria:**
- Accept dask.array.Array as underlying storage
- Use apply_ufunc with dask='parallelized'
- Preserve chunking in output
- No automatic rechunking (user controls via .chunk())

### 3. Statistical and Distribution Capabilities

#### FR-STAT-001: Gamma Distribution Fitting
**Requirement:** Fit gamma distribution to precipitation data with zero-inflation handling
**Acceptance Criteria:**
- Use scipy.stats.gamma.fit() for parameter estimation
- Handle zero-inflated data via empirical CDF for zeros
- Apply maximum likelihood estimation (MLE)
- Validate fit convergence (raise warning if failed)

#### FR-STAT-002: Pearson Type III Distribution
**Requirement:** Support Pearson Type III distribution for indices
**Acceptance Criteria:**
- Implement method-of-moments parameter estimation
- Support skewness-based fitting
- Validate against NOAA reference implementation
- Document when to prefer over gamma (e.g., negative values)

#### FR-STAT-003: Calibration Period Configuration
**Requirement:** Allow users to specify calibration period for distribution fitting
**Acceptance Criteria:**
- Accept calibration_start, calibration_end parameters (datetime-like)
- Default to full time series if not specified
- Subset data by time coordinate for xarray inputs
- Warn if calibration period < 30 years

#### FR-STAT-004: Standardization Transform
**Requirement:** Transform fitted distribution to standardized normal (Z-score)
**Acceptance Criteria:**
- Convert CDF values to standard normal via inverse normal CDF
- Handle edge cases (CDF=0 → -inf, CDF=1 → +inf)
- Clip extreme values at configurable threshold (default: ±3.5σ)
- Preserve IEEE 754 NaN semantics

### 4. Metadata and CF Convention Compliance

#### FR-META-001: Coordinate Preservation
**Requirement:** Preserve all coordinates from input to output
**Acceptance Criteria:**
- Copy all dimension coordinates (time, lat, lon, etc.)
- Copy all non-dimension coordinates (bounds, auxiliary)
- Preserve coordinate attributes
- Maintain coordinate order

#### FR-META-002: Attribute Preservation
**Requirement:** Preserve relevant input attributes and add index-specific metadata
**Acceptance Criteria:**
- Copy global attributes from input (e.g., institution, source)
- Add CF-compliant variable attributes (long_name, units, references)
- Add calculation metadata (scale, distribution, library version)
- Overwrite conflicting attributes with index-specific values

#### FR-META-003: CF Convention Compliance
**Requirement:** Output meets CF Metadata Convention standards
**Acceptance Criteria:**
- units attribute present and valid (dimensionless for indices, mm for PET)
- long_name describes the variable
- standard_name where defined by CF (use long_name otherwise)
- references includes DOI to algorithm paper
- Valid per cf-checker (Phase 2 requirement)

#### FR-META-004: Provenance Tracking
**Requirement:** Record calculation provenance in metadata
**Acceptance Criteria:**
- Add history attribute with ISO 8601 timestamp
- Include index type, scale, distribution in history
- Include library name and version
- Append to existing history (don't overwrite)

#### FR-META-005: Chunking Preservation
**Requirement:** Preserve Dask chunking strategy from input
**Acceptance Criteria:**
- Output has same chunks as input (if Dask-backed)
- No automatic rechunking during calculation
- Document chunk size recommendations for performance
- Support .chunk() on output for user-defined rechunking

### 5. API and Integration

#### FR-API-001: Function Signature Consistency
**Requirement:** Maintain consistent parameter naming across indices
**Acceptance Criteria:**
- Common parameters: data, scale, distribution, calibration_start, calibration_end
- Index-specific parameters follow NumPy conventions (e.g., latitude_degrees)
- All parameters documented in docstring with types
- Use keyword-only arguments for optional parameters (after *)

#### FR-API-002: Type Hints and Overloads
**Requirement:** Provide complete type annotations for IDE support
**Acceptance Criteria:**
- All public functions have @overload signatures for NumPy and xarray paths
- Return types specify np.ndarray vs xr.DataArray correctly
- Optional parameters typed as Optional[T]
- Pass mypy --strict validation

#### FR-API-003: Default Parameter Values
**Requirement:** Provide sensible defaults for optional parameters
**Acceptance Criteria:**
- Default distribution: gamma for SPI/SPEI
- Default calibration period: full time series
- Default scale: 3 months for SPI/SPEI
- Defaults documented in docstring with rationale

#### FR-API-004: Deprecation Warnings
**Requirement:** Provide clear warnings for deprecated functionality (Phase 2+)
**Acceptance Criteria:**
- Use warnings.warn with DeprecationWarning category
- Message includes alternative approach and removal version
- Link to migration guide in documentation
- Warnings suppressible via warnings.filterwarnings

### 6. Error Handling and Validation

#### FR-ERROR-001: Input Validation
**Requirement:** Validate inputs before processing with clear error messages
**Acceptance Criteria:**
- Check for required dimensions (time exists)
- Validate scale in valid range (1-72)
- Validate distribution in supported set
- Raise ValueError with descriptive message (not generic exception)

#### FR-ERROR-002: Computation Error Handling
**Requirement:** Handle and report computation failures gracefully
**Acceptance Criteria:**
- Catch distribution fitting failures (convergence errors)
- Provide context in error message (input shape, parameters)
- Suggest remediation (e.g., "try pearson3 distribution")
- Log error context via structlog

#### FR-ERROR-003: Structured Exceptions
**Requirement:** Use custom exception types for different failure modes
**Acceptance Criteria:**
- InsufficientDataError for time series too short
- DistributionFitError for fitting failures
- DimensionMismatchError for coordinate alignment issues
- All inherit from ClimateIndicesError base class

#### FR-ERROR-004: Warning Emission
**Requirement:** Warn users about potentially problematic inputs
**Acceptance Criteria:**
- Warn if >20% missing data
- Warn if calibration period < 30 years
- Warn if distribution fit has poor goodness-of-fit
- Use warnings.warn (not logging) for user-facing warnings

### 7. Observability and Logging

#### FR-LOG-001: Structured Logging Configuration
**Requirement:** Configure structlog for dual output (JSON + console)
**Acceptance Criteria:**
- JSON output for file handlers (machine-readable)
- Human-readable output for console (with color)
- Configuration via single function call (e.g., configure_logging())
- No logging to files by default (user-configured)

#### FR-LOG-002: Calculation Event Logging
**Requirement:** Log index calculation start/completion events
**Acceptance Criteria:**
- Log at INFO level for calculation start
- Include index type, scale, distribution, input shape
- Log duration in milliseconds at completion
- Bind context (correlation ID) for multi-index workflows

#### FR-LOG-003: Error Context Logging
**Requirement:** Log detailed context on computation failures
**Acceptance Criteria:**
- Log at ERROR level with full traceback
- Include input metadata (shape, coordinates, chunking)
- Include parameter values (scale, distribution, etc.)
- No logging of data values (privacy + size)

#### FR-LOG-004: Performance Metrics
**Requirement:** Log performance metrics for large computations
**Acceptance Criteria:**
- Log computation time for all index calculations
- Log memory usage for arrays > 1GB (if available)
- Metrics accessible via structlog context
- Support custom metrics via context binding

#### FR-LOG-005: Log Level Configuration
**Requirement:** Allow users to configure logging verbosity
**Acceptance Criteria:**
- Support standard levels: DEBUG, INFO, WARNING, ERROR
- Default to INFO level
- Environment variable override (CLIMATE_INDICES_LOG_LEVEL)
- Document log output format and filtering

### 8. Testing and Validation

#### FR-TEST-001: Equivalence Test Framework
**Requirement:** Automated tests verify xarray/NumPy numerical equivalence
**Acceptance Criteria:**
- Parametrized tests for all indices
- Compare xarray.values against NumPy output
- Tolerance: 1e-8 for float64
- Run on CI for all PRs

#### FR-TEST-002: Metadata Validation Tests
**Requirement:** Tests verify metadata preservation and CF compliance
**Acceptance Criteria:**
- Assert coordinates match input
- Assert required CF attributes present
- Validate attribute types (units is string, etc.)
- Check provenance (history attribute)

#### FR-TEST-003: Edge Case Coverage
**Requirement:** Tests cover known edge cases and failure modes
**Acceptance Criteria:**
- Zero-inflated precipitation (all zeros, mixed)
- Missing data patterns (random, blocks, leading/trailing)
- Minimum time series (exactly 30 years)
- Coordinate misalignment (different grid resolutions)

#### FR-TEST-004: Reference Dataset Validation
**Requirement:** Validate outputs against published reference datasets
**Acceptance Criteria:**
- SPI matches NOAA reference implementation (tolerance: 1e-5)
- SPEI matches CSIC reference (Vicente-Serrano et al.)
- Test data included in repository (tests/data/)
- Documented provenance of reference data

#### FR-TEST-005: Property-Based Testing
**Requirement:** Use Hypothesis for generative edge case testing
**Acceptance Criteria:**
- Properties: monotonicity, symmetry, boundedness
- Strategies generate valid climate data (positive precip, realistic temps)
- Shrinking finds minimal failing examples
- Integrated into pytest suite

### 9. Documentation

#### FR-DOC-001: API Reference Documentation
**Requirement:** Complete API reference auto-generated from docstrings
**Acceptance Criteria:**
- All public functions documented with Google-style docstrings
- Examples included in docstrings (tested via doctest)
- Parameter types and defaults documented
- Return value format specified
- Published via Sphinx

#### FR-DOC-002: xarray Migration Guide
**Requirement:** Guide for users migrating from NumPy to xarray API
**Acceptance Criteria:**
- Side-by-side code examples (NumPy vs xarray)
- Explains metadata benefits
- Covers common pitfalls (dimension names, alignment)
- Includes performance considerations

#### FR-DOC-003: Quickstart Tutorial
**Requirement:** Get-started guide for new users
**Acceptance Criteria:**
- Complete in < 5 minutes
- Shows data loading, calculation, visualization
- Works with included sample data
- Covers both NumPy and xarray paths

#### FR-DOC-004: Algorithm Documentation
**Requirement:** Document scientific algorithms with references
**Acceptance Criteria:**
- Link to peer-reviewed papers (DOI)
- Explain when to use each index
- Document parameter selection guidance (distribution choice, scale)
- Include validation datasets

#### FR-DOC-005: Troubleshooting Guide
**Requirement:** Document common errors and solutions
**Acceptance Criteria:**
- Covers dimension mismatch errors
- Explains distribution fitting failures
- Performance tuning for large datasets
- Links to relevant GitHub issues

### 10. Performance and Scalability

#### FR-PERF-001: Overhead Benchmark
**Requirement:** xarray path has minimal overhead vs NumPy
**Acceptance Criteria:**
- Overhead < 5% for in-memory computation
- Benchmark suite included in repository
- CI tracks performance regressions
- Results published in documentation

#### FR-PERF-002: Chunked Computation Efficiency
**Requirement:** Efficient computation on Dask-backed arrays
**Acceptance Criteria:**
- Single-pass algorithms (no multiple .compute() calls)
- Graph optimization via apply_ufunc
- Document recommended chunk sizes
- Test with 10GB synthetic dataset

#### FR-PERF-003: Memory Efficiency
**Requirement:** Process datasets larger than available RAM
**Acceptance Criteria:**
- Support lazy evaluation (Dask)
- No intermediate array materialization (use .map_blocks())
- Document memory usage patterns
- Profile memory usage in tests

#### FR-PERF-004: Parallel Computation
**Requirement:** Support parallel computation across chunks
**Acceptance Criteria:**
- Dask scheduler configuration (threads, processes, distributed)
- Thread-safe implementation (no global state)
- Document parallelization benefits and overheads
- Benchmark scaling efficiency (weak scaling)

### 11. Packaging and Distribution

#### FR-PKG-001: PyPI Distribution
**Requirement:** Maintain existing PyPI package
**Acceptance Criteria:**
- Package name: climate-indices
- Semantic versioning (MAJOR.MINOR.PATCH)
- Include xarray as optional dependency (extras_require)
- Wheel + source distribution

#### FR-PKG-002: Dependency Management
**Requirement:** Specify dependencies with version constraints
**Acceptance Criteria:**
- Pin NumPy/SciPy/xarray minimum versions
- Use pyproject.toml for metadata
- Optional dependencies for extras (xarray, structlog)
- Lock file for reproducible development (uv.lock)

#### FR-PKG-003: Version Compatibility
**Requirement:** Support Python 3.9+
**Acceptance Criteria:**
- Test matrix: Python 3.9, 3.10, 3.11, 3.12, 3.13
- CI tests all versions on PR
- Document supported versions in README
- Deprecation policy (12 months notice for version drops)

#### FR-PKG-004: Beta Tagging
**Requirement:** Mark xarray features as beta until Phase 2
**Acceptance Criteria:**
- Docstrings include ".. warning:: Beta feature" directive
- CHANGELOG.md marks xarray API as experimental
- README.md clarifies API stability guarantees
- No breaking changes within minor versions (even for beta)

---

## Step 10: Non-Functional Requirements

Non-functional requirements define **how well** the system performs its functions. This section focuses on quality attributes relevant to a scientific computing library.

### 1. Performance

#### NFR-PERF-001: Computational Overhead
**Requirement:** xarray adapter path introduces minimal overhead compared to NumPy baseline
**Metric:** Overhead < 5% for in-memory computations (measured via benchmark suite)
**Measurement Method:**
- Benchmark suite using timeit with 100 iterations
- Test matrices: (1000×1000×120), (360×180×1200) spatial-temporal grids
- Compare NumPy path vs xarray path with same underlying NumPy arrays
- Report mean, std dev, and 95th percentile execution times

**Acceptance Criteria:**
- SPI/SPEI/PET calculations: < 5% overhead
- Metadata operations excluded from timing (construction/attribute assignment)
- CI tracks performance regressions (fail if >10% slowdown)

---

#### NFR-PERF-002: Chunked Computation Efficiency
**Requirement:** Dask-backed computations scale efficiently with parallelism
**Metric:** Weak scaling efficiency > 70% up to 8 workers
**Measurement Method:**
- Fixed chunk size (100×100×120 per chunk)
- Increase total dataset size proportionally with workers
- Measure: T(1 worker) / (N × T(N workers)) × 100%
- Test on local multiprocessing scheduler

**Acceptance Criteria:**
- 2 workers: >85% efficiency
- 4 workers: >75% efficiency
- 8 workers: >70% efficiency
- Document overhead sources (task scheduling, data transfer)

---

#### NFR-PERF-003: Memory Efficiency for Large Datasets
**Requirement:** Process datasets exceeding available RAM without OOM errors
**Metric:** Successfully compute SPI on 50GB dataset with 16GB RAM
**Measurement Method:**
- Synthetic dataset: (1440×720×1200) float32 = ~49GB uncompressed
- Monitor peak RSS via memory_profiler
- Confirm lazy evaluation (no full array materialization)
- Validate output correctness on subset

**Acceptance Criteria:**
- Peak memory < 16GB (ideal: < 8GB)
- Computation completes without swap thrashing (< 10% time in I/O wait)
- Output validates against in-memory computation on smaller subset

---

#### NFR-PERF-004: Startup Time
**Requirement:** Library import time does not impact scripting workflows
**Metric:** `import climate_indices` completes in < 500ms
**Measurement Method:**
- Measure via `python -X importtime -c "import climate_indices"`
- Cold import (clear __pycache__)
- Report cumulative import time

**Acceptance Criteria:**
- Import time < 500ms on modern hardware (2020+ laptop)
- No eager initialization of heavy dependencies (defer Dask import)
- Document lazy import strategy for optional dependencies

---

### 2. Reliability

#### NFR-REL-001: Numerical Reproducibility
**Requirement:** Deterministic outputs across environments for identical inputs
**Metric:** Bit-exact results (within FP tolerance) on same NumPy/SciPy versions
**Measurement Method:**
- Run identical test suite on Linux, macOS, Windows
- Pin NumPy/SciPy to specific versions in lock file
- Compare outputs using `np.allclose(atol=1e-15, rtol=1e-8)`

**Acceptance Criteria:**
- Float64: relative tolerance 1e-8 (default for SPI/SPEI)
- Float32: relative tolerance 1e-5 (if user-downcasted)
- Document non-determinism sources (parallel reduction order in Dask)
- Provide `set_random_seed()` for stochastic bootstrapping (Phase 2+)

---

#### NFR-REL-002: Graceful Degradation
**Requirement:** Partial failures do not crash entire computation
**Metric:** Chunked computations isolate failures to affected chunks
**Measurement Method:**
- Inject NaN-only chunks into test dataset
- Verify output contains NaN for failed chunks, valid results for others
- Confirm structured log entries for each failed chunk

**Acceptance Criteria:**
- No global exceptions for localized failures (e.g., distribution fit failure in 1 chunk)
- Warnings emitted per-chunk with coordinate context
- Output DataArray has NaN for failed regions, preserves valid data elsewhere

---

#### NFR-REL-003: Version Stability
**Requirement:** Results remain stable across minor version updates
**Metric:** No numerical changes in patch/minor releases
**Measurement Method:**
- Regression test suite with locked reference outputs
- Compare v2.0.0 vs v2.1.0 on same inputs
- Allow changes only in major version bumps

**Acceptance Criteria:**
- Patch versions (2.0.x): bit-exact compatibility
- Minor versions (2.x.0): behavioral compatibility (may add warnings, not change values)
- Major versions (x.0.0): breaking changes allowed with migration guide

---

### 3. Compatibility

#### NFR-COMPAT-001: Python Version Support
**Requirement:** Support Python 3.9 through 3.13
**Metric:** Test matrix passes on all supported versions
**Measurement Method:**
- GitHub Actions matrix: [3.9, 3.10, 3.11, 3.12, 3.13]
- Test on official python:3.x Docker images
- Test on both Linux and macOS

**Acceptance Criteria:**
- All tests pass on every supported Python version
- No version-specific code paths (except via version checks if unavoidable)
- Deprecation policy: 12-month notice before dropping Python version
- Document minimum version in README and pyproject.toml

---

#### NFR-COMPAT-002: Dependency Version Matrix
**Requirement:** Support wide range of NumPy/SciPy/xarray versions
**Metric:** Test matrix covers min/max supported versions
**Minimum Versions:**
- NumPy ≥ 1.23 (structured dtype improvements)
- SciPy ≥ 1.10 (stats module stability)
- xarray ≥ 2023.01 (apply_ufunc dask improvements)

**Measurement Method:**
- CI matrix: minimum versions + latest versions
- Dependabot for automatic dependency updates
- Version conflicts tested via tox/nox

**Acceptance Criteria:**
- No version pinning (only lower bounds in pyproject.toml)
- Support versions within 2 years of release (rolling window)
- Document version compatibility in README matrix table

---

#### NFR-COMPAT-003: Backward Compatibility Guarantee
**Requirement:** No breaking changes to NumPy API without major version bump
**Metric:** Existing test suite passes unchanged across minor versions
**Measurement Method:**
- Lock test suite at v2.0 release
- Run locked tests on all v2.x.y releases
- Track deprecation warnings separately

**Acceptance Criteria:**
- 100% of v2.0 tests pass on v2.x releases (may have new warnings)
- Deprecations follow policy: warning in v2.x → removal in v3.0 (minimum 12 months)
- Breaking changes documented in CHANGELOG.md with "BREAKING" prefix

---

### 4. Integration

#### NFR-INTEG-001: xarray Ecosystem Compatibility
**Requirement:** Interoperate seamlessly with xarray ecosystem tools
**Tools:** Dask, zarr, cf_xarray, xskillscore, xclim
**Measurement Method:**
- Integration tests with Dask schedulers (threads, processes, distributed)
- Test zarr read/write workflows
- Validate cf_xarray attribute access
- Ensure output compatible with xskillscore metrics

**Acceptance Criteria:**
- Dask: lazy computation works with all schedulers
- zarr: can read/write datasets with climate indices
- cf_xarray: CF attributes accessible via `.cf` accessor
- xclim: outputs compatible as xclim indicators inputs

---

#### NFR-INTEG-002: CF Convention Compliance
**Requirement:** Outputs comply with CF Metadata Conventions v1.10
**Metric:** cf-checker validation passes with 0 errors
**Measurement Method:**
- Run cf-checker on sample outputs for each index
- Validate via cfchecker Python API in test suite
- Test in CI (Phase 2)

**Acceptance Criteria:**
- MVP: Best-effort compliance (0 errors for SPI/SPEI/PET)
- Phase 2: cf-checker integrated in CI as gate
- Document CF compliance status in README badges
- Handle warnings appropriately (e.g., missing standard_name is acceptable)

---

#### NFR-INTEG-003: structlog Output Format Compatibility
**Requirement:** JSON logs compatible with common aggregators
**Tools:** Elasticsearch, Splunk, CloudWatch Logs, Datadog
**Measurement Method:**
- Validate JSON schema of log output
- Test ingestion into ELK stack (developer environment)
- Confirm field naming follows conventions (timestamp, level, message)

**Acceptance Criteria:**
- Valid JSON Lines format (one JSON object per line)
- ISO 8601 timestamps (RFC 3339 with timezone)
- Standard field names: `timestamp`, `level`, `event`, `logger`
- No unescaped special characters in JSON
- Document log schema in documentation

---

### 5. Maintainability

#### NFR-MAINT-001: Type Coverage
**Requirement:** Complete type annotations for static analysis
**Metric:** mypy --strict passes with 0 errors on all code
**Measurement Method:**
- Enable mypy --strict in CI
- Run on src/ directory (exclude tests for MVP)
- Track type coverage via mypy stats

**Acceptance Criteria:**
- 100% of public API type-annotated
- All function parameters and returns typed
- Use @overload for NumPy/xarray dispatch
- No `type: ignore` comments without justification

---

#### NFR-MAINT-002: Test Coverage
**Requirement:** High test coverage for critical paths
**Metric:** Line coverage > 85%, branch coverage > 80%
**Measurement Method:**
- pytest-cov with coverage.py
- Measure separately for NumPy path vs xarray path
- Exclude visualization/CLI from MVP coverage requirements

**Acceptance Criteria:**
- Core indices (SPI, SPEI, PET): >90% coverage
- Utilities (distribution fitting, metadata): >85% coverage
- Error handling paths: >75% coverage (tested via pytest.raises)
- CI fails if coverage drops >2% in single PR

---

#### NFR-MAINT-003: Documentation Coverage
**Requirement:** All public APIs documented with examples
**Metric:** 100% docstring coverage for public modules/functions
**Measurement Method:**
- interrogate tool for docstring coverage
- Validate examples via doctest
- Check for undocumented parameters

**Acceptance Criteria:**
- All public functions have complete Google-style docstrings
- All parameters documented with types
- At least one example per function
- Examples pass doctest (or marked SKIP with explanation)

---

#### NFR-MAINT-004: Code Quality Standards
**Requirement:** Consistent code style and quality checks
**Tools:** ruff (linter + formatter), mypy (types), bandit (security)
**Measurement Method:**
- Pre-commit hooks enforce ruff format
- CI runs ruff check, mypy, bandit
- Fail PR on any violations

**Acceptance Criteria:**
- ruff: 0 violations (auto-fixed via ruff format)
- mypy --strict: 0 errors
- bandit: 0 high/medium severity issues
- Line length: 120 characters (ruff configured)

---

#### NFR-MAINT-005: Dependency Security
**Requirement:** No known CVEs in dependencies
**Metric:** 0 high/critical CVEs in production dependencies
**Measurement Method:**
- pip-audit in CI on every PR
- Dependabot security alerts enabled
- Monthly dependency update reviews

**Acceptance Criteria:**
- CI fails on high/critical CVEs
- Medium/low CVEs require issue tracking
- Security patches applied within 7 days of disclosure
- Document dependency update policy in CONTRIBUTING.md

---

## Step 11: Document Complete

This PRD is now complete with all 11 BMAD workflow steps. The document provides:

- **Strategic Context:** User journeys, success criteria, and phasing strategy
- **Domain Requirements:** Scientific correctness, CF compliance, and operational constraints
- **60 Functional Requirements:** Organized across 11 capability areas (index calculation, input handling, statistical operations, metadata, API design, error handling, logging, testing, documentation, performance, packaging)
- **23 Non-Functional Requirements:** Measurable quality attributes across performance, reliability, compatibility, integration, and maintainability

**Document Statistics:**
- Total Requirements: 83 (60 FR + 23 NFR)
- MVP Scope: SPI, SPEI, PET + structlog (Weeks 1-4)
- Coverage: 5 user personas, 3 delivery phases, 11 domain requirements

**Next BMAD Workflows:**
- **Architecture Design:** Define adapter pattern implementation, module structure, and interface contracts
- **Epic Breakdown:** Decompose MVP scope into implementation units
- **Validation:** Optional step to review PRD against BMAD quality checklist

---

## Appendix: Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-05 | 1.0 | Initial PRD complete (Steps 1-11) |

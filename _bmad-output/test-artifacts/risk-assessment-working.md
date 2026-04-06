# Risk Assessment - climate_indices v2.4.0

## Testability Review

### Controllability ✅
- **Fixture infrastructure:** Session-scoped .npy fixtures for expensive operations (gamma fitting, precip/temp data)
- **Test data:** `tests/fixture/` contains 15+ .npy files + JSON (palmer_awc.json)
- **Parameterization:** pytest parametrize for distribution variations (gamma vs Pearson)
- **Isolation:** Each test class independent, `_reset_logging_for_testing()` utility
- **Determinism:** Fixed random seeds in conftest.py fixtures (e.g., `rng = np.random.default_rng(42)`)

### Observability ✅
- **structlog integration:** Lifecycle events (calculation_started, calculation_completed, calculation_failed)
- **Performance metrics:** pytest-benchmark for overhead tracking, memory profiling
- **Error context:** Structured exceptions with keyword-only context (shape, expected, actual)
- **Coverage reporting:** pytest-cov with ≥85% target (currently >90%)
- **Type checking:** mypy --strict on typed_public_api.py

### Reliability ⚠️
- **Session-scoped fixtures:** Reduce setup time, but risk state leakage (mitigated by reset utilities)
- **Dask chunked test isolation:** Time dimension must be single chunk (validated in tests)
- **Floating-point reproducibility:** 1e-8 tolerance for float64 equivalence tests
- **Backward compatibility guarantees:** test_backward_compat.py locks numpy API stability

### Testability Concerns for v2.4.0

**🚨 BLOCKERS (Sprint 0 Critical Path):**

1. **NOAA EDDI Reference Dataset Acquisition** (Track 2)
   - **Issue:** FR-EDDI-001 requires validation against NOAA PSL EDDI CONUS archive
   - **Blocker:** Dataset download URL may be unstable, subset selection unclear
   - **Impact:** Without reference data, cannot validate EDDI implementation
   - **Mitigation:** Download and cache dataset in `tests/data/reference/eddi/` with SHA256 checksum + provenance metadata
   - **Owner:** QA/Dev (Track 2 implementer)
   - **Timeline:** Sprint 0 (before Track 2 starts)

2. **Palmer 344-Dataset Reference Collection** (Track 3 Performance Validation)
   - **Issue:** Palmer xarray must achieve ≥80% speed of multiprocessing baseline
   - **Blocker:** No established baseline measurement exists in current test suite
   - **Impact:** Cannot validate NFR-PALMER-PERF without baseline
   - **Mitigation:** Create `tests/benchmark/test_palmer_baseline.py` to capture multiprocessing CLI performance on synthetic 360×180×240 grid
   - **Owner:** Dev (Track 3 implementer)
   - **Timeline:** Sprint 0 (before Track 3 starts)

**⚠️ HIGH PRIORITY (Recommendations for Approval):**

1. **Track 0 Equivalence Protocol Infrastructure**
   - **Issue:** NFR-PATTERN-EQUIV requires before/after baseline capture for 100% numerical equivalence
   - **Recommendation:** Add `tests/baseline/` directory with .npy snapshots captured before each pattern application
   - **Approval needed:** Confirm baseline fixtures committed to git vs excluded (.gitignore)
   - **Owner:** Dev (Track 0 implementer)
   - **Timeline:** Before Track 0 refactoring starts

2. **FAO56 Worked Example Fixtures** (Track 1)
   - **Issue:** FR-PM-005 requires Bangkok (tropical) + Uccle (temperate) validation
   - **Recommendation:** Embed FAO56 Example 17 & 18 input data directly in test file vs external fixtures
   - **Rationale:** Small datasets (~10 values each), avoid fixture bloat
   - **Owner:** Dev (Track 1 implementer)
   - **Timeline:** During Track 1 test implementation

**📋 INFO ONLY (Solutions Provided):**

1. **Test strategy:** Unit (helpers) → Integration (xarray adapters) → Property-based (invariants) → Benchmarks (performance)
2. **Tooling:** pytest, hypothesis, pytest-benchmark, mypy --strict
3. **Execution:** PR (full suite <5 min parallel), Nightly (benchmarks), Weekly (extended Palmer 344-dataset validation)
4. **Coverage:** ~90 new test scenarios across 4 tracks, prioritized P0-P3 with risk linkage

---

## Risk Assessment Matrix

**Legend:**
- TECH: Technical/integration risk
- PERF: Performance/scalability risk
- DATA: Data integrity/validation risk
- SCI: Scientific correctness risk

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
|---------|----------|-------------|-------------|--------|-------|------------|-------|----------|
| **R-001** | **SCI** | Numerical drift during Track 0 pattern refactoring breaks 1e-8 equivalence guarantee | 2 | 3 | **6** | Implement baseline capture protocol: save .npy snapshots before refactoring, validate after. Equivalence test with atol=1e-8, rtol=1e-8. Revert on failure. | Dev (Track 0) | Sprint 0 setup, continuous validation |
| **R-002** | **SCI** | FAO56 constant precision errors (e.g., wrong Magnus constants, Kelvin conversion) break PM-ET accuracy | 2 | 3 | **6** | Validate helpers independently with known reference values (e.g., SVP(21.5°C) = 2.564 ±0.01 kPa). Use exact FAO56 constants (0.6108, 17.27, 237.3). Docstring equation traceability. | Dev (Track 1) | During PM-ET implementation |
| **R-003** | **SCI** | Non-linearity in SVP averaging missed: common error is `e°(Tmean)` instead of `(e°(Tmax)+e°(Tmin))/2` | 2 | 3 | **6** | Implement unit test for `_mean_svp()` that verifies `_mean_svp(tmin, tmax) != _svp_from_t((tmin+tmax)/2)`. Add docstring warning about non-linearity. | Dev (Track 1) | During PM-ET implementation |
| **R-004** | **DATA** | NOAA EDDI reference dataset unavailable, corrupted, or subset selection incorrect | 2 | 3 | **6** | Download dataset in Sprint 0, validate SHA256 checksum, document provenance (source, URL, download date, subset description). Create `tests/data/reference/README.md` with retrieval instructions. | QA/Dev (Track 2) | Sprint 0 (before Track 2) |
| **R-005** | **PERF** | Palmer xarray performance regression: <80% of multiprocessing baseline due to Python loop overhead | 3 | 2 | **6** | Establish baseline in Sprint 0 via `test_palmer_baseline.py`. Benchmark Palmer xarray with synthetic 360×180×240 grid. If <80%, investigate vectorization opportunities or accept documented trade-off. | Dev (Track 3) | Sprint 0 baseline + Track 3 validation |

### Medium-Priority Risks (Score 3-5)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
|---------|----------|-------------|-------------|--------|-------|------------|-------|
| R-006 | TECH | Humidity pathway selection logic bug (priority order violated) | 2 | 2 | 4 | Unit tests for each pathway (dewpoint, RH extremes, RH mean). Integration test verifying priority order. Logging at DEBUG level for selected pathway. | Dev (Track 1) |
| R-007 | DATA | EDDI 1e-5 tolerance insufficient for non-parametric ranking FP accumulation | 1 | 3 | 3 | Document tolerance rationale in test docstring (why 1e-5 vs 1e-8). Monitor test failures; if systematic bias detected, investigate EDDI algorithm vs tolerance adjustment. | QA/Dev (Track 2) |
| R-008 | TECH | Palmer params_dict JSON serialization round-trip failure (e.g., NaN handling) | 2 | 2 | 4 | Test round-trip: `json.loads(ds.attrs["palmer_params"]) == original_dict`. Test edge cases: NaN, Inf, large floats. Use `allow_nan=True` in json.dumps if needed. | Dev (Track 3) |
| R-009 | TECH | AWC time dimension validation bypass (user provides DataArray with time, corrupts results) | 2 | 2 | 4 | Explicit validation: `if time_dim in awc.dims: raise ValueError("AWC must not have time dimension...")`. Unit test for error case. | Dev (Track 3) |
| R-010 | TECH | Multi-output stack/unpack pattern fragility (xarray #1815 workaround) | 2 | 2 | 4 | Comment referencing Issue #1815. Monitor xarray releases for native multi-output support. Integration test for stack/unpack correctness. | Dev (Track 3) |
| R-011 | TECH | Type safety violations in @overload signatures (Palmer numpy tuple vs xarray Dataset) | 1 | 2 | 2 | mypy --strict on typed_public_api.py (CI enforced). Test that isinstance checks correctly dispatch. | Dev (Track 0, Track 3) |
| R-012 | TECH | structlog lifecycle event inconsistency across indices after Track 0 migration | 1 | 2 | 2 | Audit all modules: calculation_started, calculation_completed, calculation_failed. Pattern validation test. | Dev (Track 0) |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
|---------|----------|-------------|-------------|--------|-------|------------|-------|
| R-013 | TECH | scPDSI stub premature usage (user calls function expecting implementation) | 1 | 1 | 1 | NotImplementedError with clear message referencing Wells et al. 2004. Docstring states "planned for future release". | Dev (Track 2) |
| R-014 | TECH | Property-based test false positives (hypothesis finds spurious edge cases) | 1 | 1 | 1 | Carefully define hypothesis strategies (exclude NaN/Inf where inappropriate). Add @example decorators for known-good cases. | Dev (Track 0) |

---

## Risk Summary by Track

### Track 0: Canonical Pattern Completion
- **High:** R-001 (numerical drift)
- **Medium:** R-011 (type safety), R-012 (structlog consistency)
- **Low:** R-014 (property test false positives)
- **Critical path:** Baseline capture infrastructure must be in place before refactoring starts

### Track 1: PM-ET FAO56
- **High:** R-002 (FAO56 constant precision), R-003 (SVP non-linearity)
- **Medium:** R-006 (humidity pathway logic)
- **Critical path:** Helper function unit tests must validate FAO56 equation fidelity

### Track 2: EDDI/PNP/scPDSI
- **High:** R-004 (NOAA reference dataset availability)
- **Medium:** R-007 (EDDI tolerance sufficiency)
- **Low:** R-013 (scPDSI stub usage)
- **Critical path:** NOAA dataset download + provenance tracking must complete in Sprint 0

### Track 3: Palmer Multi-Output xarray
- **High:** R-005 (performance regression)
- **Medium:** R-008 (params_dict serialization), R-009 (AWC validation), R-010 (stack/unpack fragility)
- **Critical path:** Multiprocessing baseline must be established before xarray implementation

---

## Sprint 0 Blockers Summary

**MUST COMPLETE BEFORE TRACK IMPLEMENTATION:**

1. **NOAA EDDI Reference Dataset** (R-004): Download, validate SHA256, document provenance → `tests/data/reference/eddi/`
2. **Palmer Performance Baseline** (R-005): Measure multiprocessing CLI speed on synthetic grid → `tests/benchmark/test_palmer_baseline.py`
3. **Track 0 Baseline Capture Infrastructure** (R-001): Create `tests/baseline/` directory structure + capture protocol

**Timeline:** Sprint 0 (1 week) before Track 0-3 start

---

## Testability Gaps

### Architecture Improvements Needed

| Gap ID | Description | Impact | Mitigation | Owner | Timeline |
|--------|-------------|--------|------------|-------|----------|
| GAP-001 | No Palmer reference datasets for extended validation (344 NOAA datasets mentioned in research) | Cannot validate Palmer accuracy against external ground truth | Defer to Phase 3 (post-v2.4.0). Use existing test fixtures for initial validation. Document as known limitation. | Dev | Post-v2.4.0 |
| GAP-002 | PM-ET radiation calculation helpers (FAO56 Eq. 20-52) not in scope | Limited PET method completeness | Documented as Track 1 out-of-scope. PM-ET core (Eq. 1-19) sufficient for EDDI. Defer extended radiation to v2.5.0+. | Dev | Post-v2.4.0 |
| GAP-003 | No Dask distributed testing in CI (only single-machine parallelism) | Cannot validate multi-node Dask behavior | Acceptable for v2.4.0. Dask chunking validation sufficient for single-machine use case. Multi-node testing deferred. | DevOps | Post-v2.4.0 |

### Testability Assessment Summary

**What Works Well (No Action Needed):**
- Session-scoped fixtures reduce test runtime (40-year monthly precip loaded once)
- Hypothesis property-based testing validates mathematical invariants (boundedness, NaN propagation)
- Equivalence test pattern (1e-8 tolerance) proven effective for xarray vs numpy validation
- pytest-benchmark tracks performance regressions in CI
- mypy --strict on typed_public_api.py catches type errors early

**Accepted Trade-offs:**
- Palmer sequential time constraint (cannot parallelize along time dimension) documented, no workaround needed
- 1e-5 tolerance for EDDI (vs 1e-8 for SPI/SPEI) due to non-parametric ranking FP accumulation — acceptable per research

---

**Assessment Complete:** 14 risks identified (5 high, 7 medium, 2 low), 3 Sprint 0 blockers, 3 testability gaps documented

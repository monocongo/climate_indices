---
stepsCompleted: [1, 2, 3, 4, 5]
lastStep: 5
status: complete
---

# Test Design for Architecture: climate_indices v2.4.0

**Purpose:** Architectural concerns, testability gaps, and NFR requirements for review by Architecture/Dev teams. Serves as a contract between QA and Engineering on what must be addressed before test development begins.

**Date:** 2026-02-16
**Author:** James
**Status:** Architecture Review Pending
**Project:** climate_indices (Python Scientific Library)
**PRD Reference:** `_bmad-output/planning-artifacts/prd.md` (v2.4.0)
**Architecture Reference:** `_bmad-output/planning-artifacts/architecture.md` (v2.4.0)

---

## Executive Summary

**Scope:** v2.4.0 release spanning 4 parallel tracks (30 new FRs, 8 new NFRs)

**Business Context** (from PRD v2.4.0):
- **Impact:** Completes canonical pattern migration (100% consistency), adds physics-based PET (PM FAO56), validates EDDI against NOAA reference, enables Palmer xarray multi-output
- **Problem:** v2.3.0 established patterns but only applied to 3/7 indices. v2.4.0 closes gaps + adds advanced features.
- **Timeline:** 10-14 weeks (Track 0 ∥ Track 1 → Track 2 ∥ Track 3)

**Architecture** (from Architecture v2.4.0):
- **Decision 8:** PM-ET module placement in `eto.py` with private FAO56 helpers
- **Decision 10:** Palmer xarray manual wrapper (Pattern C) for multi-output + params_dict
- **Decision 12:** Reference dataset testing with SHA256 provenance tracking (NOAA EDDI)
- **Decision 13:** Track 0 index-by-index pattern application with baseline capture protocol

**Expected Scale:**
- 28 existing test files (14,497 lines, >90% coverage)
- 44 new test scenarios across 4 tracks (~100-175 hours QA effort)
- Numerical precision: 1e-8 tolerance (scientific correctness paramount)

**Risk Summary:**
- **Total risks:** 14 (5 high-priority ≥6, 7 medium 3-5, 2 low 1-2)
- **High-priority:** Numerical drift (R-001), FAO56 precision (R-002, R-003), NOAA dataset (R-004), Palmer performance (R-005)
- **Test effort:** ~2.5-4.5 weeks for 1 QA, ~1.5-2.5 weeks for 2 QAs

---

## Quick Guide

### 🚨 BLOCKERS - Team Must Decide (Can't Proceed Without)

**Sprint 0 Critical Path** - These MUST be completed before QA can write integration tests:

1. **NOAA EDDI Reference Dataset Acquisition** (R-004) — Download NOAA PSL EDDI CONUS archive subset, validate SHA256, document provenance → `tests/data/reference/eddi/` (recommended owner: Dev/QA Track 2)
2. **Palmer Performance Baseline Establishment** (R-005) — Measure multiprocessing CLI speed on synthetic 360×180×240 grid → `tests/benchmark/test_palmer_baseline.py` (recommended owner: Dev Track 3)
3. **Track 0 Baseline Capture Infrastructure** (R-001) — Create `tests/baseline/` directory structure + protocol for before/after equivalence snapshots (recommended owner: Dev Track 0)

**What we need from team:** Complete these 3 items in Sprint 0 (1 week) or test development is blocked.

---

### ⚠️ HIGH PRIORITY - Team Should Validate (We Provide Recommendation, You Approve)

1. **R-002: FAO56 Constant Precision** — Validate helpers independently with known reference values (e.g., SVP(21.5°C) = 2.564 ±0.01 kPa). Use exact FAO56 constants (0.6108, 17.27, 237.3). (Sprint 1-2, Dev Track 1)
2. **R-003: SVP Non-Linearity Error** — Unit test verifies `_mean_svp(tmin, tmax) != _svp_from_t((tmin+tmax)/2)`. Add docstring warning. (Sprint 1-2, Dev Track 1)
3. **R-001: Numerical Drift During Refactoring** — Save .npy snapshots before pattern application, validate after with atol=1e-8. Revert on failure. (Sprint 1-2, Dev Track 0)

**What we need from team:** Review recommendations and approve (or suggest changes).

---

### 📋 INFO ONLY - Solutions Provided (Review, No Decisions Needed)

1. **Test strategy:** Unit (helpers) → Integration (xarray adapters) → Property-based (invariants) → Benchmarks (performance)
2. **Tooling:** pytest, hypothesis, pytest-benchmark, mypy --strict
3. **Execution:** PR (full suite <5 min parallel), Nightly (benchmarks ~30-60 min), Weekly (extended validation)
4. **Coverage:** 44 test scenarios prioritized P0-P3 with risk linkage (9 P0, 21 P1, 9 P2, 5 P3)
5. **Quality gates:** P0=100%, P1≥95%, P2≥85%, coverage≥80%, Palmer xarray ≥80% baseline performance

**What we need from team:** Just review and acknowledge (we already have the solution).

---

## For Architects and Devs - Open Topics 👷

### Risk Assessment

**Total risks identified:** 14 (5 high-priority score ≥6, 7 medium 3-5, 2 low 1-2)

**Risk Category Legend:**
- **SCI:** Scientific correctness risk (numerical accuracy, algorithm fidelity)
- **DATA:** Data integrity/validation risk (reference datasets, provenance)
- **PERF:** Performance/scalability risk (regression, baseline comparison)
- **TECH:** Technical/integration risk (type safety, serialization, validation)

#### High-Priority Risks (Score ≥6) - IMMEDIATE ATTENTION

| Risk ID    | Category   | Description                                                                                                                   | Probability | Impact | Score       | Mitigation                                                                                                                                                        | Owner          | Timeline                              |
| ---------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------- | ------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------- |
| **R-001**  | **SCI**    | Numerical drift during Track 0 pattern refactoring breaks 1e-8 equivalence guarantee                                         | 2           | 3      | **6**       | Implement baseline capture protocol: save .npy snapshots before refactoring, validate after with atol=1e-8. Revert on failure.                                   | Dev (Track 0)  | Sprint 0 setup, continuous validation |
| **R-002**  | **SCI**    | FAO56 constant precision errors (e.g., wrong Magnus constants, Kelvin conversion) break PM-ET accuracy                       | 2           | 3      | **6**       | Validate helpers independently with known reference values (e.g., SVP(21.5°C) = 2.564 ±0.01 kPa). Use exact FAO56 constants (0.6108, 17.27, 237.3).              | Dev (Track 1)  | During PM-ET implementation           |
| **R-003**  | **SCI**    | Non-linearity in SVP averaging missed: common error is `e°(Tmean)` instead of `(e°(Tmax)+e°(Tmin))/2`                        | 2           | 3      | **6**       | Unit test verifies `_mean_svp(tmin, tmax) != _svp_from_t((tmin+tmax)/2)`. Add docstring warning about non-linearity.                                             | Dev (Track 1)  | During PM-ET implementation           |
| **R-004**  | **DATA**   | NOAA EDDI reference dataset unavailable, corrupted, or subset selection incorrect                                            | 2           | 3      | **6**       | Download dataset in Sprint 0, validate SHA256 checksum, document provenance (source, URL, download date, subset description). Create `tests/data/reference/README.md`. | QA/Dev (Track 2) | Sprint 0 (before Track 2)           |
| **R-005**  | **PERF**   | Palmer xarray performance regression: <80% of multiprocessing baseline due to Python loop overhead                           | 3           | 2      | **6**       | Establish baseline in Sprint 0 via `test_palmer_baseline.py`. Benchmark Palmer xarray with synthetic 360×180×240 grid. If <80%, document trade-off.              | Dev (Track 3)  | Sprint 0 baseline + Track 3 validation |

#### Medium-Priority Risks (Score 3-5)

| Risk ID | Category | Description                                                                                    | Probability | Impact | Score | Mitigation                                                                                                                                | Owner          |
| ------- | -------- | ---------------------------------------------------------------------------------------------- | ----------- | ------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| R-006   | TECH     | Humidity pathway selection logic bug (priority order violated)                                 | 2           | 2      | 4     | Unit tests for each pathway (dewpoint, RH extremes, RH mean). Integration test verifying priority order. Logging at DEBUG level.         | Dev (Track 1)  |
| R-007   | DATA     | EDDI 1e-5 tolerance insufficient for non-parametric ranking FP accumulation                    | 1           | 3      | 3     | Document tolerance rationale in test docstring. Monitor test failures; investigate EDDI algorithm vs tolerance adjustment.                | QA/Dev (Track 2) |
| R-008   | TECH     | Palmer params_dict JSON serialization round-trip failure (e.g., NaN handling)                  | 2           | 2      | 4     | Test round-trip: `json.loads(ds.attrs["palmer_params"]) == original_dict`. Test edge cases: NaN, Inf, large floats.                      | Dev (Track 3)  |
| R-009   | TECH     | AWC time dimension validation bypass (user provides DataArray with time, corrupts results)     | 2           | 2      | 4     | Explicit validation: `if time_dim in awc.dims: raise ValueError("AWC must not have time dimension...")`. Unit test for error case.       | Dev (Track 3)  |
| R-010   | TECH     | Multi-output stack/unpack pattern fragility (xarray #1815 workaround)                          | 2           | 2      | 4     | Comment referencing Issue #1815. Monitor xarray releases for native multi-output support. Integration test for stack/unpack correctness. | Dev (Track 3)  |
| R-011   | TECH     | Type safety violations in @overload signatures (Palmer numpy tuple vs xarray Dataset)          | 1           | 2      | 2     | mypy --strict on typed_public_api.py (CI enforced). Test isinstance checks correctly dispatch.                                           | Dev (Track 0, Track 3) |
| R-012   | TECH     | structlog lifecycle event inconsistency across indices after Track 0 migration                 | 1           | 2      | 2     | Audit all modules: calculation_started, calculation_completed, calculation_failed. Pattern validation test.                               | Dev (Track 0)  |

#### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description                                                                                   | Probability | Impact | Score | Mitigation                                                                                                                      | Owner          |
| ------- | -------- | --------------------------------------------------------------------------------------------- | ----------- | ------ | ----- | ------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| R-013   | TECH     | scPDSI stub premature usage (user calls function expecting implementation)                    | 1           | 1      | 1     | NotImplementedError with clear message referencing Wells et al. 2004. Docstring states "planned for future release".           | Dev (Track 2)  |
| R-014   | TECH     | Property-based test false positives (hypothesis finds spurious edge cases)                    | 1           | 1      | 1     | Carefully define hypothesis strategies (exclude NaN/Inf where inappropriate). Add @example decorators for known-good cases.     | Dev (Track 0)  |

---

### Testability Concerns and Architectural Gaps

#### 🚨 ACTIONABLE CONCERNS

**Blockers to Fast Feedback:**

| Concern ID | Description                                                                                                        | Impact                                                     | Owner          | Timeline                 |
| ---------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | -------------- | ------------------------ |
| CON-001    | NOAA EDDI reference dataset not cached in repo (download URL may be unstable)                                      | Blocks T-007 (P0 EDDI validation)                          | QA/Dev (Track 2) | Sprint 0 Week 1        |
| CON-002    | Palmer multiprocessing baseline not established (no benchmark exists in current test suite)                        | Blocks T-009 (P0 Palmer performance validation)            | Dev (Track 3)  | Sprint 0 Week 1          |
| CON-003    | Track 0 baseline capture protocol undefined (no standard for before/after equivalence snapshots)                  | Blocks T-001, T-002 (P0 equivalence validation)            | Dev (Track 0)  | Sprint 0 Week 1          |

**Architectural Improvements Needed:**

| Improvement ID | Description                                                                                                               | Impact                                                        | Owner          | Timeline                           |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | -------------- | ---------------------------------- |
| IMP-001        | PM-ET helper functions need independent validation tests (not just end-to-end FAO56 examples)                            | Enables early detection of constant precision errors (R-002)  | Dev (Track 1)  | During Track 1 implementation      |
| IMP-002        | Palmer xarray adapter requires custom wrapper (NOT decorator-based due to multi-output + params_dict)                    | Validates Decision 10 (Pattern C) workaround for xarray #1815 | Dev (Track 3)  | During Track 3 implementation      |
| IMP-003        | Nightly CI job for pytest-benchmark does not exist (benchmarks currently run in PR, slowing feedback)                    | Improves PR feedback speed, isolates performance testing     | DevOps         | Sprint 0 Week 2                    |

#### Testability Assessment Summary (FYI)

**What Works Well:**
- Session-scoped fixtures reduce test runtime (40-year monthly precip loaded once)
- Hypothesis property-based testing validates mathematical invariants (boundedness, NaN propagation)
- Equivalence test pattern (1e-8 tolerance) proven effective for xarray vs numpy validation
- pytest-benchmark tracks performance regressions
- mypy --strict on typed_public_api.py catches type errors early

**Accepted Trade-offs:**
- Palmer sequential time constraint (cannot parallelize along time dimension) documented, no workaround needed
- 1e-5 tolerance for EDDI (vs 1e-8 for SPI/SPEI) due to non-parametric ranking FP accumulation — acceptable per research

---

### Risk Mitigation Plans

**For High-Priority Risks Only (Score ≥6)**

#### R-001: Numerical Drift During Pattern Refactoring

**Strategy:**
1. Before applying pattern to any index, capture baseline output via `np.save("tests/baseline/baseline_{index}.npy", result)`
2. Apply pattern (xarray adapter, exception migration, structlog, etc.)
3. Re-run computation, validate `np.testing.assert_allclose(baseline, refactored, atol=1e-8, rtol=1e-8)`
4. If equivalence test fails: revert pattern application immediately, investigate root cause, document finding, re-attempt with fix

**Owner:** Dev (Track 0 implementer)
**Timeline:** Sprint 0 setup (week 1) + continuous validation during Track 0 refactoring
**Status:** Pending Sprint 0 setup
**Verification:** All Track 0 equivalence tests pass (T-001, T-002) with 1e-8 tolerance

---

#### R-002: FAO56 Constant Precision Errors

**Strategy:**
1. Implement unit tests for each helper function with known FAO56 reference values:
   - `_svp_from_t(21.5°C)` should equal `2.564 kPa ±0.01`
   - `_atm_pressure(100m)` should equal `100.1 kPa ±0.01`
   - `_psy_const(100.1 kPa)` should equal `0.06667 kPa/°C ±0.00001`
2. Hardcode exact FAO56 constants in source code with comments referencing equation numbers:
   ```python
   # FAO56 Eq. 11 Magnus constants
   SVP_COEFF = 0.6108  # kPa
   SVP_SLOPE = 17.27   # dimensionless
   SVP_BASE = 237.3    # degrees Celsius
   ```
3. Add docstring equation traceability for all helpers (e.g., "Implements FAO56 Equation 7")

**Owner:** Dev (Track 1 implementer)
**Timeline:** During Track 1 PM-ET implementation
**Status:** Pending Track 1 start
**Verification:** T-003 (P0 FAO56 constants validation) passes, T-005/T-006 (P0 FAO56 examples) within ±0.05 mm/day

---

#### R-003: SVP Non-Linearity Error

**Strategy:**
1. Implement unit test:
   ```python
   def test_mean_svp_nonlinearity():
       tmin, tmax = 10.0, 30.0
       mean_svp_correct = _mean_svp(tmin, tmax)
       mean_svp_wrong = _svp_from_t((tmin + tmax) / 2)
       assert not np.isclose(mean_svp_correct, mean_svp_wrong, rtol=0.01), \
           "SVP averaging is non-linear: e°((Tmax+Tmin)/2) ≠ (e°(Tmax)+e°(Tmin))/2"
   ```
2. Add docstring warning in `_mean_svp()`:
   ```python
   """
   WARNING: Saturation vapor pressure is non-linear with temperature.
   Always average e°(Tmax) and e°(Tmin), NEVER compute e°(Tmean).
   See FAO56 Chapter 3, Section 3.2.1.
   """
   ```

**Owner:** Dev (Track 1 implementer)
**Timeline:** During Track 1 PM-ET implementation
**Status:** Pending Track 1 start
**Verification:** T-004 (P0 SVP non-linearity test) passes

---

#### R-004: NOAA EDDI Reference Dataset Unavailable

**Strategy:**
1. Download NOAA PSL EDDI CONUS archive subset in Sprint 0:
   - URL: `https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/data/2020/`
   - Subset: Colorado River Basin, 2020-01 to 2020-12, 6-month EDDI
2. Validate dataset integrity: compute SHA256 checksum, store in `tests/data/reference/eddi/SHA256SUMS`
3. Create provenance metadata in NetCDF attributes:
   ```python
   attrs = {
       "source": "NOAA PSL EDDI CONUS Archive",
       "url": "https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/data/2020/",
       "download_date": "2026-02-16",
       "subset_description": "Colorado River Basin, 2020-01 to 2020-12, 6-month EDDI",
       "algorithm_version": "EDDI v1.2 (Hobbins et al. 2016)",
       "sha256_checksum": "a3f5d8c2e1b4..."
   }
   ```
4. Document retrieval instructions in `tests/data/reference/README.md`

**Owner:** QA/Dev (Track 2 implementer)
**Timeline:** Sprint 0 Week 1 (before Track 2 starts)
**Status:** Pending Sprint 0 start
**Verification:** T-007 (P0 EDDI NOAA reference validation) can run without manual dataset download

---

#### R-005: Palmer xarray Performance Regression

**Strategy:**
1. Establish baseline in Sprint 0:
   ```python
   # tests/benchmark/test_palmer_baseline.py
   def test_palmer_multiprocessing_baseline(benchmark):
       # Synthetic 360×180×240 grid (10 years, global 1-degree resolution)
       precip = generate_synthetic_precip(shape=(240, 180, 360))
       pet = generate_synthetic_pet(shape=(240, 180, 360))
       awc = 2.5  # inches
       # Multiprocessing CLI baseline
       result = benchmark(palmer_cli_multiprocessing, precip, pet, awc)
       # Record median of 10 runs: e.g., 45.2 seconds
   ```
2. Benchmark Palmer xarray in Track 3:
   ```python
   def test_palmer_xarray_performance(benchmark):
       # Same synthetic grid
       precip_da = xr.DataArray(precip, dims=["time", "lat", "lon"], ...)
       pet_da = xr.DataArray(pet, dims=["time", "lat", "lon"], ...)
       result = benchmark(palmer_xarray, precip_da, pet_da, awc=2.5)
       # Target: ≥80% of baseline (e.g., ≤56.5 seconds)
   ```
3. If <80%: investigate vectorization opportunities or accept documented trade-off (Python loop overhead per research)

**Owner:** Dev (Track 3 implementer)
**Timeline:** Sprint 0 baseline (week 1) + Track 3 validation
**Status:** Pending Sprint 0 baseline establishment
**Verification:** T-009 (P0 Palmer xarray performance) achieves ≥80% of baseline speed

---

## Assumptions and Dependencies

### Architectural Assumptions

1. **Numerical Precision:** float64 sufficient for 1e-8 equivalence guarantee. No float32 degradation expected.
2. **Dask Chunking:** Time dimension single chunk required for climate indices (sequential computation). Documented constraint.
3. **Palmer Calibration:** Calibration parameters (alpha, beta, gamma, delta) spatially constant (computed from first grid cell). Acceptable trade-off.
4. **EDDI Tolerance:** 1e-5 tolerance acceptable for non-parametric ranking FP accumulation (looser than 1e-8 for parametric SPI/SPEI). Justified per research.
5. **PM-ET Radiation:** Users provide `net_radiation` parameter (not computed from solar radiation). Extended equations (FAO56 Eq. 20-52) deferred to v2.5.0+.

### Dependencies

| Dependency                                   | Required By | Timeline        | Notes                                                                 |
| -------------------------------------------- | ----------- | --------------- | --------------------------------------------------------------------- |
| NOAA EDDI reference dataset                  | T-007 (P0)  | Sprint 0 Week 1 | Download, validate SHA256, document provenance                        |
| Palmer multiprocessing baseline              | T-009 (P0)  | Sprint 0 Week 1 | Measure current CLI performance on synthetic grid                     |
| Track 0 baseline capture infrastructure      | T-001, T-002 (P0) | Sprint 0 Week 1 | Directory structure + protocol for equivalence snapshots        |
| Track 0 Palmer structlog migration complete  | Track 3 tests | End of Track 0  | Blocks Palmer xarray test development (don't mix logging frameworks)  |
| Track 1 PM-ET implementation complete        | Track 2 EDDI tests | End of Track 1 | EDDI PM-ET recommendation validation requires PM-ET to exist       |
| Track 0 PNP xarray adapter complete          | Track 2 PNP tests | End of Track 0 | PNP adapter tests require PNP xarray implementation               |

### Risks to Plan

| Risk                                                                                       | Impact                                                         | Contingency                                                                                |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| NOAA EDDI reference dataset download URL unstable or dataset structure changed            | Blocks T-007 (P0), delays Track 2                              | Cache dataset in Sprint 0. If unavailable, contact NOAA PSL for alternative access.       |
| Palmer multiprocessing baseline performance varies significantly across hardware           | Invalidates performance target (≥80%)                          | Document baseline hardware specs. Accept ≥70% on CI hardware if baseline established.     |
| Track 0 refactoring introduces numerical drift detected late (after multiple pattern applications) | Requires partial revert, delays Track 0                 | Incremental validation after each pattern application. Commit baseline fixtures to git.   |
| FAO56 worked examples insufficient for full PM-ET validation                              | Edge cases not caught                                          | Supplement with additional test cases from literature (e.g., Valiantzas 2013 comparisons).|
| Palmer xarray performance <80% due to Python loop overhead (no vectorization possible)     | Fails NFR-PALMER-PERF                                          | Document trade-off. Consider Numba/Cython optimization or accept performance gap.          |

---

## Not in Scope

**Excluded from v2.4.0 Architecture:**

1. **Palmer 344-Dataset Extended Validation** (Deferred to Phase 3)
   - **Reasoning:** Reference datasets not readily available, extensive effort required
   - **Mitigation:** Current fixtures sufficient for initial validation. Document as known limitation.

2. **PM-ET Extended Radiation Equations** (FAO56 Eq. 20-52) (Deferred to v2.5.0+)
   - **Reasoning:** Out of Track 1 scope. Core PM-ET (Eq. 1-19) sufficient for EDDI.
   - **Mitigation:** Users provide `net_radiation` parameter. Cross-reference FAO56 Paper 56.

3. **Dask Multi-Node Distributed Testing** (Infrastructure Limitation)
   - **Reasoning:** Infrastructure complexity, single-machine Dask sufficient for v2.4.0
   - **Mitigation:** Document Dask chunking constraints (time dimension single chunk).

4. **CLI xarray Integration** (Deferred to v2.5.0)
   - **Reasoning:** CLI currently uses extract-compute-rewrap pattern. xarray integration deferred.
   - **Mitigation:** Programmatic xarray API (typed_public_api.py) is primary use case.

5. **Cross-Platform Testing** (Linux/macOS/Windows) (CI Limitation)
   - **Reasoning:** CI only tests Linux (GitHub Actions). macOS/Windows deferred.
   - **Mitigation:** Python 3.10-3.13 compatibility ensures broad platform support.

---

**Document Complete:** 14 risks assessed, 5 high-priority mitigations planned, 3 Sprint 0 blockers identified, 3 architectural assumptions documented

**Next Step:** Review with Architecture/Dev teams, prioritize Sprint 0 blocker resolution, proceed to QA test design document generation

# Test Coverage Plan - climate_indices v2.4.0

**Note:** P0/P1/P2/P3 classifications represent **priority and risk level**, NOT execution timing. See Execution Strategy section for when tests run.

---

## P0: Critical Core Functionality (Blocks Release + High Risk ≥6 + No Workaround)

**Criteria:**
- Blocks core scientific correctness (numerical accuracy)
- High risk score (≥6)
- No reasonable workaround if broken
- Must pass 100% before release

**Purpose:** Catch catastrophic failures that invalidate scientific results.

### P0 Test Scenarios

| Test ID | Requirement | Test Level | Risk Link | Test Description | Notes |
|---------|-------------|------------|-----------|------------------|-------|
| **T-001** | FR-PATTERN-001, NFR-PATTERN-EQUIV | Integration | R-001 | **PNP xarray equivalence:** `test_percentage_of_normal_xarray_equivalence()` validates numpy vs xarray paths within 1e-8 | Baseline capture required |
| **T-002** | FR-PATTERN-002, NFR-PATTERN-EQUIV | Integration | R-001 | **PCI xarray equivalence:** `test_pci_xarray_equivalence()` validates numpy vs xarray paths within 1e-8 | Baseline capture required |
| **T-003** | FR-PM-002 | Unit | R-002 | **FAO56 constants validation:** Test `_atm_pressure()`, `_psy_const()`, `_svp_from_t()` with exact FAO56 constants (0.6108, 17.27, 237.3) | Independent helper validation |
| **T-004** | FR-PM-003 | Unit | R-003 | **SVP non-linearity test:** Verify `_mean_svp(tmin, tmax) != _svp_from_t((tmin+tmax)/2)` | Common error detection |
| **T-005** | FR-PM-005 | Integration | R-002 | **FAO56 Example 17 validation:** Bangkok tropical monthly ETo = 5.72 ±0.05 mm/day | Reference data validation |
| **T-006** | FR-PM-005 | Integration | R-002 | **FAO56 Example 18 validation:** Uccle temperate daily ETo = 3.9 ±0.05 mm/day | Reference data validation |
| **T-007** | FR-EDDI-001, NFR-EDDI-VAL | Integration | R-004 | **EDDI NOAA reference validation:** `test_eddi_noaa_reference()` validates against NOAA PSL CONUS archive within 1e-5 | Provenance tracking required |
| **T-008** | FR-PALMER-007, NFR-PALMER-PERF | Integration | R-005 | **Palmer xarray equivalence:** Validate numpy tuple vs xarray Dataset values within 1e-8 for all 4 variables | Must pass before performance test |
| **T-009** | NFR-PALMER-PERF | Benchmark | R-005 | **Palmer xarray performance:** Benchmark xarray path vs multiprocessing baseline on synthetic 360×180×240 grid, target ≥80% | Baseline must exist in Sprint 0 |

**P0 Totals:** 9 tests (~25-40 hours for implementation + fixtures)

---

## P1: Important Features (Medium Risk 3-4 + Common Workflows)

**Criteria:**
- Important features used in common workflows
- Medium risk score (3-4)
- Significant user impact if broken
- Should pass ≥95% before release

**Purpose:** Ensure reliable feature delivery for primary use cases.

### P1 Test Scenarios

| Test ID | Requirement | Test Level | Risk Link | Test Description | Notes |
|---------|-------------|------------|-----------|------------------|-------|
| **T-010** | FR-PATTERN-003, FR-PATTERN-004 | Integration | R-011 | **ETo typed_public_api:** Test `@overload` signatures for `eto_thornthwaite` and `eto_hargreaves` dispatch correctly | mypy --strict validation |
| **T-011** | FR-PATTERN-005, FR-PATTERN-006 | Integration | R-011 | **PNP/PCI typed_public_api:** Test `@overload` signatures dispatch numpy→ndarray, xarray→DataArray | mypy --strict validation |
| **T-012** | FR-PATTERN-007 | Integration | R-012 | **Palmer structlog lifecycle:** Validate `calculation_started`, `calculation_completed`, `calculation_failed` events emitted | Pattern consistency check |
| **T-013** | FR-PATTERN-008 | Integration | R-012 | **ETo Thornthwaite lifecycle:** Validate bind + lifecycle events match Hargreaves pattern | Pattern consistency check |
| **T-014** | FR-PATTERN-009 | Unit | R-011 | **Structured exceptions:** Test PNP/PCI/ETo/Palmer raise `InvalidArgumentError` with context (not bare `ValueError`) | Exception hierarchy validation |
| **T-015** | FR-PATTERN-010 | Property | R-014 | **PNP properties:** Boundedness (≥0), shape preservation, NaN propagation, linear scaling | hypothesis strategies |
| **T-016** | FR-PATTERN-011 | Property | R-014 | **PCI properties:** Range [0,100], input length validation (365/366), NaN handling, zero-precip edge case | hypothesis strategies |
| **T-017** | FR-PATTERN-012 | Property | R-014 | **Expanded SPEI/Palmer properties:** SPEI shape/NaN/zero-input, Palmer PHDI/PMDI/Z-Index bounded ranges | hypothesis strategies |
| **T-018** | FR-PM-001, FR-PM-006 | Integration | R-006 | **PM-ET core calculation:** Test `eto_penman_monteith()` with all humidity pathways (dewpoint, RH extremes, RH mean) | Auto-selection logic validation |
| **T-019** | FR-PM-004 | Integration | R-006 | **Humidity pathway priority:** Test auto-dispatcher selects dewpoint > RH extremes > RH mean in correct order | Priority order enforcement |
| **T-020** | FR-PM-004 | Unit | R-006 | **Humidity pathway error:** Test raises `InvalidArgumentError` when no humidity input provided | Error handling validation |
| **T-021** | FR-PM-006 | Integration | - | **PM-ET xarray adapter:** Test CF metadata, coordinate preservation, Dask compatibility | Standard adapter pattern |
| **T-022** | FR-EDDI-002 | Integration | R-007 | **EDDI xarray adapter:** Test CF metadata, numpy vs xarray equivalence within 1e-8 | Tolerance consistency |
| **T-023** | FR-PNP-001 | Integration | - | **PNP xarray adapter:** Test simplest index validates minimal metadata handling | Pattern validation |
| **T-024** | FR-SCPDSI-001 | Unit | R-013 | **scPDSI stub:** Test raises `NotImplementedError` with Wells et al. 2004 reference | Explicit stub behavior |
| **T-025** | FR-PALMER-001 | Integration | R-010 | **Palmer manual wrapper:** Test stack/unpack pattern for multi-output handling | Workaround validation |
| **T-026** | FR-PALMER-002 | Integration | - | **Palmer Dataset structure:** Test Dataset contains pdsi, phdi, pmdi, z_index variables with independent CF metadata | Multi-output correctness |
| **T-027** | FR-PALMER-003 | Integration | R-009 | **AWC spatial validation:** Test scalar and DataArray AWC (no time dimension), raise error if time dim present | Dimension validation |
| **T-028** | FR-PALMER-004 | Integration | R-008 | **params_dict serialization:** Test JSON round-trip (`json.loads(ds.attrs["palmer_params"])`) and individual attrs access | Data integrity |
| **T-029** | FR-PALMER-005 | Integration | - | **Palmer CF metadata:** Test per-variable long_name, units, references from registry | Metadata completeness |
| **T-030** | FR-PALMER-006 | Integration | R-011 | **Palmer typed_public_api:** Test `@overload` dispatches numpy→tuple, xarray→Dataset | Type safety validation |

**P1 Totals:** 21 tests (~35-60 hours for implementation)

---

## P2: Secondary Features (Low Risk 1-2 + Edge Cases)

**Criteria:**
- Secondary features or edge cases
- Low risk score (1-2)
- Moderate user impact if broken
- Nice-to-have before release

**Purpose:** Ensure robustness and edge case handling.

### P2 Test Scenarios

| Test ID | Requirement | Test Level | Risk Link | Test Description | Notes |
|---------|-------------|------------|-----------|------------------|-------|
| **T-031** | FR-PM-002 | Unit | - | **Atmospheric helpers edge cases:** Test `_atm_pressure()` at extreme altitudes (0m, 5000m), `_psy_const()` with edge pressures | Robustness validation |
| **T-032** | FR-PM-003 | Unit | - | **Vapor pressure edge cases:** Test `_svp_from_t()` at freezing (0°C), boiling (100°C), negative temps | Robustness validation |
| **T-033** | FR-PM-004 | Integration | - | **Explicit vapor pressure override:** Test user-provided `actual_vapor_pressure` bypasses auto-dispatcher | Override mechanism |
| **T-034** | FR-EDDI-003 | Integration | - | **EDDI CLI integration:** Test `--index eddi --pet_file <path>` flag in `process_climate_indices` CLI | CLI integration |
| **T-035** | FR-EDDI-004 | Documentation | - | **EDDI PM-ET recommendation:** Validate docstring cross-references `eto_penman_monteith()` with Hobbins et al. 2016 citation | Documentation completeness |
| **T-036** | FR-PALMER-002 | Integration | - | **Palmer NetCDF round-trip:** Test Dataset write/read with `ds.to_netcdf()` preserves structure and metadata | Serialization validation |
| **T-037** | FR-PALMER-003 | Integration | - | **AWC DataArray spatial variation:** Test spatially-varying AWC (lat, lon) vs uniform scalar AWC produces different results | Spatial parameter handling |
| **T-038** | FR-PATTERN-009 | Integration | - | **Exception error messages:** Test structured exceptions provide actionable guidance (not just "invalid value") | Error message quality |
| **T-039** | NFR-PATTERN-COVERAGE | Integration | - | **Pattern compliance dashboard:** Generate 7×6 matrix (7 indices × 6 patterns) showing 100% compliance | Systematic validation |

**P2 Totals:** 9 tests (~15-25 hours for implementation)

---

## P3: Nice-to-Have (Exploratory + Deferred Features)

**Criteria:**
- Exploratory testing or deferred features
- Informational value (not blocking)
- Future enhancements

**Purpose:** Provide insights for future development.

### P3 Test Scenarios

| Test ID | Requirement | Test Level | Risk Link | Test Description | Notes |
|---------|-------------|------------|-----------|------------------|-------|
| **T-040** | GAP-001 | Integration | - | **Palmer 344-dataset extended validation:** Test against NOAA PSL Palmer reference datasets (deferred to Phase 3) | Future enhancement |
| **T-041** | GAP-002 | Unit | - | **Extended radiation equations:** Test FAO56 Eq. 20-52 helpers (deferred to v2.5.0+) | Future enhancement |
| **T-042** | GAP-003 | Integration | - | **Dask multi-node testing:** Validate distributed Dask behavior (deferred) | Infrastructure limitation |
| **T-043** | NFR-PALMER-SEQ | Integration | - | **Palmer chunking guidance validation:** Test that time-chunked Dask arrays produce incorrect results (document constraint) | Constraint documentation |
| **T-044** | NFR-MULTI-OUT | Documentation | - | **xarray Issue #1815 tracking:** Monitor xarray releases for native multi-output support | Workaround maintenance |

**P3 Totals:** 5 tests (~5-10 hours for exploration)

---

## Test Coverage Summary

| Priority | Test Count | Estimated Effort | Pass Rate Target | Risk Coverage |
|----------|------------|------------------|------------------|---------------|
| **P0** | 9 | ~25-40 hours | 100% (blocks release) | 5 high-priority risks (R-001 to R-005) |
| **P1** | 21 | ~35-60 hours | ≥95% | 7 medium-priority risks (R-006 to R-012) |
| **P2** | 9 | ~15-25 hours | ≥85% | 2 low-priority risks (R-013 to R-014) |
| **P3** | 5 | ~5-10 hours | N/A (exploratory) | 3 testability gaps (GAP-001 to GAP-003) |
| **TOTAL** | **44** | **~80-135 hours** | Coverage ≥80% | 14 risks + 3 gaps addressed |

---

## Execution Strategy (Organized by Infrastructure Overhead)

**Philosophy:** "Run everything in PRs unless expensive/long-running." Maximize fast feedback.

### Every PR (Full Suite — <5 Minutes with Parallel Execution)

**What runs:**
- All P0, P1, P2 unit tests
- All P0, P1, P2 integration tests (except benchmarks)
- All property-based tests (hypothesis with small example counts)
- Type checking (mypy --strict)
- Linting (ruff check + format)

**Rationale:**
- pytest-xdist enables parallel execution (8 workers typical)
- Fast fixtures (session-scoped .npy files loaded once)
- In-memory computation (no I/O bottlenecks)
- Target: <5 min total runtime on CI

**Examples:**
```bash
uv run pytest tests/ -v -m "not benchmark and not slow" -n auto
uv run mypy src/climate_indices/ --strict
ruff check src/ tests/
```

### Nightly (Performance + Extended Validation — ~30-60 Minutes)

**What runs:**
- All pytest-benchmark tests (T-009, P2 performance tests)
- Extended Palmer 344-dataset validation (T-040, when implemented)
- Hypothesis property tests with extended example counts (10,000+ examples)
- Memory profiling (pytest-memray if integrated)

**Rationale:**
- Benchmarks require stable hardware (no CI concurrency)
- Extended validation datasets too large for frequent runs
- Performance regression tracking needs baseline stability

**Examples:**
```bash
uv run pytest tests/ -v -m benchmark --benchmark-only
uv run pytest tests/test_property_based.py --hypothesis-profile=extensive
```

### Weekly (Long-Running + Manual — ~Hours)

**What runs:**
- Dask multi-node testing (T-042, when implemented)
- Chaos testing (infrastructure failures)
- Manual validation against NOAA EDDI reference datasets (T-007 extended coverage)

**Rationale:**
- Long-running tests block development if run frequently
- Manual validation requires human judgment

---

## Resource Estimates (Interval-Based)

**QA Effort Breakdown:**

| Activity | Estimated Effort | Notes |
|----------|------------------|-------|
| **P0 Test Implementation** | ~25-40 hours | 9 tests: equivalence validation, FAO56 examples, NOAA reference, Palmer baseline |
| **P1 Test Implementation** | ~35-60 hours | 21 tests: typed_public_api, structlog lifecycle, property tests, xarray adapters, Palmer multi-output |
| **P2 Test Implementation** | ~15-25 hours | 9 tests: edge cases, CLI integration, NetCDF round-trip, pattern compliance |
| **P3 Exploration** | ~5-10 hours | 5 tests: extended validation, future enhancements |
| **Fixture Setup** | ~10-20 hours | NOAA dataset download + provenance, Palmer baseline, Track 0 baselines |
| **CI Configuration** | ~5-10 hours | pytest-benchmark setup, nightly/weekly job configuration |
| **Documentation** | ~5-10 hours | Test design review, knowledge transfer |
| **TOTAL QA EFFORT** | **~100-175 hours** | **~2.5-4.5 weeks for 1 QA, ~1.5-2.5 weeks for 2 QAs** |

**Timeline Estimate:**
- **Sprint 0 (Fixtures):** 1 week (NOAA dataset, Palmer baseline, Track 0 infrastructure)
- **Track 0 Tests:** 1.5-2.5 weeks (equivalence, type safety, property tests)
- **Track 1 Tests:** 1.5-2 weeks (PM-ET helpers, FAO56 validation, xarray adapter)
- **Track 2 Tests:** 1-1.5 weeks (EDDI reference, PNP/scPDSI adapters)
- **Track 3 Tests:** 2-3 weeks (Palmer xarray, multi-output, performance benchmarks)

**Critical Path:** Track 0 (Palmer structlog) + Track 1 (PM-ET) → Track 3 (Palmer xarray)

---

## Quality Gate Criteria

**Release Criteria:**
1. **P0 Pass Rate:** 100% (9/9 tests pass, no exceptions)
2. **P1 Pass Rate:** ≥95% (20/21 tests pass minimum)
3. **P2 Pass Rate:** ≥85% (8/9 tests pass minimum)
4. **Coverage:** ≥80% overall (currently >90%, maintain)
5. **High-Risk Mitigations:** All 5 high-priority risks (R-001 to R-005) validated
6. **Type Safety:** mypy --strict passes on `src/climate_indices/typed_public_api.py`
7. **Performance:** Palmer xarray ≥80% of multiprocessing baseline (T-009)
8. **NOAA Reference:** EDDI validates within 1e-5 tolerance (T-007)

**Bug Severity Gate:**
- No open P0 bugs (blocks core functionality)
- ≤2 open P1 bugs (important features degraded)
- P2/P3 bugs acceptable with documented workarounds

---

## Dependencies & Test Blockers

**Sprint 0 Blockers (MUST RESOLVE BEFORE TESTING STARTS):**

1. **NOAA EDDI Reference Dataset** (R-004)
   - **What:** Download NOAA PSL EDDI CONUS archive subset
   - **Where:** `tests/data/reference/eddi/noaa_conus_2020_6mo.nc`
   - **Provenance:** source, URL, download_date, SHA256 checksum, subset_description
   - **Owner:** QA/Dev (Track 2 implementer)
   - **Timeline:** Week 1 of Sprint 0
   - **Impact:** Blocks T-007 (P0 EDDI validation)

2. **Palmer Performance Baseline** (R-005)
   - **What:** Establish multiprocessing CLI speed on synthetic 360×180×240 grid
   - **Where:** `tests/benchmark/test_palmer_baseline.py`
   - **Baseline:** Measure current CLI performance (target: capture median of 10 runs)
   - **Owner:** Dev (Track 3 implementer)
   - **Timeline:** Week 1 of Sprint 0
   - **Impact:** Blocks T-009 (P0 Palmer performance validation)

3. **Track 0 Baseline Capture Infrastructure** (R-001)
   - **What:** Directory structure + protocol for before/after equivalence snapshots
   - **Where:** `tests/baseline/` (e.g., `baseline_pnp.npy`, `baseline_pci.npy`)
   - **Protocol:** Save .npy before refactoring, validate after with atol=1e-8
   - **Owner:** Dev (Track 0 implementer)
   - **Timeline:** Week 1 of Sprint 0
   - **Impact:** Blocks T-001, T-002 (P0 equivalence validation)

**Backend/Architecture Dependencies:**
- Track 0 Palmer structlog migration must complete before Track 3 Palmer xarray tests
- Track 1 PM-ET implementation must complete before Track 2 EDDI tests (PM-ET recommendation validation)
- Track 0 PNP xarray adapter must complete before Track 2 PNP tests

**QA Infrastructure Setup:**
- pytest-benchmark configuration in `pyproject.toml` (already exists)
- hypothesis strategies for property-based tests (partially exists in `test_property_based.py`)
- CI job for nightly benchmarks (new)
- CI job for weekly extended validation (new)

---

## Not in Scope

**Excluded from v2.4.0 Test Coverage:**

1. **Palmer 344-Dataset Extended Validation** (GAP-001)
   - **Reason:** Reference datasets not readily available, extensive effort required
   - **Mitigation:** Defer to Phase 3 (post-v2.4.0). Current fixtures sufficient for initial validation.
   - **Impact:** Limited external ground truth validation for Palmer indices

2. **PM-ET Extended Radiation Equations** (FAO56 Eq. 20-52) (GAP-002)
   - **Reason:** Out of Track 1 scope. Core PM-ET (Eq. 1-19) sufficient for EDDI.
   - **Mitigation:** Document as future enhancement in v2.5.0+. Cross-reference FAO56 Paper 56.
   - **Impact:** Users must provide `net_radiation` parameter (not computed from solar radiation)

3. **Dask Multi-Node Distributed Testing** (GAP-003)
   - **Reason:** Infrastructure complexity, single-machine Dask sufficient for v2.4.0
   - **Mitigation:** Document Dask chunking constraints (time dimension single chunk). Defer multi-node to post-v2.4.0.
   - **Impact:** Cannot validate distributed cluster behavior

4. **CLI xarray Integration** (Deferred)
   - **Reason:** CLI currently uses extract-compute-rewrap pattern. xarray integration deferred to v2.5.0.
   - **Mitigation:** Programmatic xarray API (typed_public_api.py) is primary use case for v2.4.0.
   - **Impact:** CLI users continue using multiprocessing workflow

5. **Cross-Platform Testing** (Linux/macOS/Windows)
   - **Reason:** CI only tests Linux (GitHub Actions). macOS/Windows deferred.
   - **Mitigation:** Python 3.10-3.13 compatibility ensures broad platform support. Users report issues on other platforms.
   - **Impact:** Cannot guarantee identical behavior on macOS/Windows

---

**Coverage Plan Complete:** 44 tests prioritized P0-P3, ~100-175 hours QA effort, 3 Sprint 0 blockers identified

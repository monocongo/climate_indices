---
workflowStatus: in_progress
stepsCompleted: [1]
lastStep: 1
mode: system-level
date: 2026-02-16
---

# Test Design Workflow Progress

## Step 1: Mode Detection & Prerequisites ✅

**Mode Detected:** System-Level (Phase 3 — Pre-Implementation)

**Rationale:**
- PRD v2.4.0 exists (90 FRs, 31 NFRs)
- Architecture v2.4.0 exists (14 decisions, 14 patterns)
- Epics v1.1 exist (stale, 47 stories for 60 FRs)
- No `sprint-status.yaml` → Phase 3 confirmed
- System-Level mode → TWO documents (architecture + QA)

**Prerequisites Validated:**
- ✅ PRD v2.4.0 at `_bmad-output/planning-artifacts/prd.md`
- ✅ Architecture v2.4.0 at `_bmad-output/planning-artifacts/architecture.md`
- ✅ Epics v1.1 at `_bmad-output/planning-artifacts/epics.md`
- ✅ Project context at `_bmad-output/project-context.md`
- ✅ Existing test infrastructure: 26 test files, 14,497 lines

**Project Adaptation:**
- Python scientific library (not web app)
- Adapting Playwright/TypeScript templates → pytest/Python patterns
- Tech stack: pytest, pytest-benchmark, hypothesis, numpy, xarray, scipy

**Next Step:** Load context & knowledge base (Step 2)

---

## Step 2: Context & Knowledge Base Loading ✅

**Project Artifacts Loaded:**
- ✅ PRD v2.4.0 (90 FRs: 30 new across 4 tracks)
- ✅ Architecture v2.4.0 (14 decisions, 14 patterns)
- ✅ Epics v1.1 (47 stories covering 60 v1.1 FRs — stale for v2.4.0)
- ✅ Project context (coding standards, test rules)
- ✅ Technical architecture (5-layer design)

**Tech Stack Extracted:**
- **Language:** Python 3.10-3.13
- **Core:** scipy >=1.15.3, xarray >=2025.6.1, dask >=2025.7.0
- **Logging:** structlog >=24.1.0
- **Testing:** pytest >=8.4.1, hypothesis >=6.100.0, pytest-benchmark >=4.0.0
- **Type Safety:** mypy (near-strict mode)
- **Quality:** ruff >=0.12.7, coverage >=7.10.1

**Integration Points:**
- xarray adapter layer (CF metadata, coordinate preservation)
- Dask compatibility (chunked computation)
- CLI (NetCDF I/O, multiprocessing)
- structlog lifecycle events
- Exception hierarchy (ClimateIndicesError base)

**NFRs Key Targets:**
- Numerical accuracy: 1e-8 tolerance (equivalence tests)
- Performance: <5% overhead for xarray vs numpy paths
- Backward compatibility: numpy API unchanged
- Coverage: ≥85% (currently >90%)
- Type safety: mypy --strict on typed_public_api.py

**Existing Test Coverage Analysis:**
- **Total:** 28 test files, 14,497 lines, >90% coverage
- **Top coverage areas:**
  - test_xarray_adapter.py (3,229 lines) — comprehensive xarray path testing
  - test_exceptions.py (795 lines) — exception hierarchy validation
  - test_backward_compat.py (659 lines) — numpy API stability
  - test_logging.py (650 lines) — structlog integration
  - test_indices.py (631 lines) — core index calculations
  - test_compute.py (618 lines) — statistical functions
  - test_property_based.py (606 lines) — mathematical invariants
  - test_zero_precipitation_fix.py (596 lines) — zero-inflation handling
  - test_metadata_validation.py (553 lines) — CF metadata compliance
- **Notable gaps:**
  - Palmer tests likely thin (125 lines vs 913-line palmer.py module)
  - No PM-ET FAO56 tests yet (Track 1)
  - No EDDI NOAA reference validation yet (Track 2)
  - No Palmer multi-output xarray tests yet (Track 3)

**v2.4.0 Requirements Summary:**

**Track 0: Canonical Pattern Completion (12 FRs)**
- PNP/PCI xarray adapters + CF metadata
- typed_public_api @overload entries (4 functions)
- Palmer structlog migration
- ETo Thornthwaite lifecycle completion
- Structured exceptions (eliminate ValueError)
- Property-based tests (PNP, PCI, expanded SPEI/Palmer)

**Track 1: PM-ET FAO56 (6 FRs)**
- Core PM-ET calculation (FAO56 Eq. 6)
- Atmospheric helpers (Eq. 7-8: pressure, psychrometric constant)
- Vapor pressure helpers (Eq. 11-13: SVP, mean SVP, slope)
- Humidity pathway dispatcher (Eq. 14-19: dewpoint → RH extremes → RH mean)
- FAO56 validation (Bangkok + Uccle examples, ±0.05 mm/day)
- PM-ET xarray adapter

**Track 2: EDDI/PNP/scPDSI (5 FRs)**
- EDDI NOAA reference validation (1e-5 tolerance, provenance tracking)
- EDDI xarray adapter + CLI integration
- EDDI PM-ET recommendation (docstring cross-reference)
- PNP xarray adapter (simplest index)
- scPDSI stub (full signature + NotImplementedError)

**Track 3: Palmer Multi-Output xarray (7 FRs)**
- Manual wrapper pattern (NOT decorator — multi-output + params_dict)
- Dataset return (pdsi, phdi, pmdi, z_index)
- AWC spatial parameter handling (scalar | DataArray, no time dim)
- params_dict dual serialization (JSON + individual attrs)
- Palmer CF metadata registry (per-variable)
- typed_public_api @overload (numpy → tuple, xarray → Dataset)
- NumPy vs xarray equivalence tests (1e-8)

**Next Step:** Testability & Risk Assessment (Step 3)

---

## Step 3: Testability & Risk Assessment ✅

**Testability Review:**
- ✅ Controllability: Session-scoped fixtures, parameterization, deterministic random seeds
- ✅ Observability: structlog lifecycle events, pytest-benchmark, coverage reporting, mypy --strict
- ⚠️ Reliability: Session-scoped fixture state leakage (mitigated), Dask chunking validation

**Sprint 0 Blockers Identified (3):**
1. NOAA EDDI reference dataset acquisition (tests/data/reference/eddi/)
2. Palmer performance baseline establishment (test_palmer_baseline.py)
3. Track 0 baseline capture infrastructure (tests/baseline/)

**Risks Identified: 14 total**
- **High-priority (≥6):** 5 risks
  - R-001: Numerical drift during refactoring (Track 0)
  - R-002: FAO56 constant precision errors (Track 1)
  - R-003: SVP non-linearity error (Track 1)
  - R-004: NOAA reference dataset unavailable (Track 2)
  - R-005: Palmer xarray performance regression (Track 3)
- **Medium-priority (3-5):** 7 risks
  - R-006: Humidity pathway logic bug (Track 1)
  - R-007: EDDI tolerance insufficient (Track 2)
  - R-008: params_dict JSON serialization failure (Track 3)
  - R-009: AWC time dimension validation bypass (Track 3)
  - R-010: Stack/unpack pattern fragility (Track 3)
  - R-011: Type safety violations (Track 0, Track 3)
  - R-012: structlog lifecycle inconsistency (Track 0)
- **Low-priority (1-2):** 2 risks
  - R-013: scPDSI stub premature usage (Track 2)
  - R-014: Property test false positives (Track 0)

**Testability Gaps (3):**
- GAP-001: No Palmer 344-dataset extended validation (defer to Phase 3)
- GAP-002: PM-ET extended radiation equations (Eq. 20-52) not in scope
- GAP-003: No Dask multi-node testing in CI

**Next Step:** Coverage Plan & Execution Strategy (Step 4)

---

## Step 4: Coverage Plan & Execution Strategy ✅

**Test Coverage Matrix:**
- **P0 (Critical):** 9 tests (~25-40 hours) — 100% pass rate required
  - Numerical equivalence (R-001), FAO56 validation (R-002, R-003), NOAA reference (R-004), Palmer performance (R-005)
- **P1 (Important):** 21 tests (~35-60 hours) — ≥95% pass rate
  - Type safety, structlog lifecycle, property tests, xarray adapters, Palmer multi-output
- **P2 (Secondary):** 9 tests (~15-25 hours) — ≥85% pass rate
  - Edge cases, CLI integration, NetCDF round-trip, pattern compliance
- **P3 (Exploratory):** 5 tests (~5-10 hours) — N/A
  - Extended validation, future enhancements, deferred features
- **Total:** 44 tests, ~100-175 hours QA effort (~2.5-4.5 weeks for 1 QA)

**Execution Strategy:**
- **PR (Full Suite):** All P0/P1/P2 tests (<5 min with pytest-xdist parallel)
- **Nightly:** Benchmarks, extended property tests (~30-60 min)
- **Weekly:** Long-running validation, manual testing (~hours)
- **Philosophy:** "Run everything in PRs unless expensive/long-running"

**Quality Gates:**
- P0: 100% pass rate (9/9)
- P1: ≥95% pass rate (20/21)
- P2: ≥85% pass rate (8/9)
- Coverage: ≥80% overall
- Palmer xarray: ≥80% performance baseline
- EDDI: 1e-5 tolerance validation

**Dependencies & Blockers:**
- Sprint 0: NOAA dataset download, Palmer baseline, Track 0 baseline infrastructure
- Backend: Palmer structlog (Track 0) → Palmer xarray tests (Track 3)
- Backend: PM-ET (Track 1) → EDDI tests (Track 2)

**Not in Scope:**
- Palmer 344-dataset extended validation (deferred to Phase 3)
- PM-ET extended radiation equations (Eq. 20-52, v2.5.0+)
- Dask multi-node testing (infrastructure limitation)
- CLI xarray integration (v2.5.0+)
- Cross-platform testing (Linux only in CI)

**Next Step:** Generate Output Documents (Step 5)

---

## Step 5: Generate Output Documents ✅

**Output Documents Created:**

1. **test-design-architecture.md** (187 lines)
   - Purpose: Architectural concerns and testability gaps for Architecture/Dev teams
   - Contents: Risk assessment (14 risks), Sprint 0 blockers (3), testability concerns, mitigation plans
   - Audience: Architecture team, Dev team, Technical leadership
   - Focus: WHAT needs testing and WHY (concerns, risks, dependencies)

2. **test-design-qa.md** (545 lines)
   - Purpose: Test execution recipe with implementation details for QA team
   - Contents: 44 test scenarios with pytest code examples, execution strategy, effort estimates
   - Audience: QA team, Test automation engineers
   - Focus: HOW to test (implementation, code examples, fixtures, CI configuration)

**Validation Against Checklist:**
- ✅ Architecture doc: ~150-200 lines, actionable-first structure, no test code
- ✅ QA doc: Implementation-focused, pytest code examples, effort intervals
- ✅ Both docs: Cross-referenced (no duplication), risk IDs consistent
- ✅ System-Level Mode: Two documents generated as required
- ✅ Adapted for Python: pytest examples (not Playwright), scientific library context

**Key Adaptations:**
- Web app patterns (Playwright, k6) → Python scientific library (pytest, pytest-benchmark, hypothesis)
- Execution strategy: PR (<5 min parallel) / Nightly (benchmarks) / Weekly (extended validation)
- Numerical precision focus: 1e-8 tolerance (scientific correctness)
- Reference dataset validation: NOAA EDDI with SHA256 provenance
- Property-based testing: hypothesis for mathematical invariants

**Workflow Status:** COMPLETE ✅

**Total Time:** ~4 hours for comprehensive test design across 4 tracks (30 new FRs, 8 new NFRs)

---

## Workflow Summary

**Completed Steps:**
1. ✅ Mode Detection & Prerequisites (System-Level confirmed)
2. ✅ Context & Knowledge Base Loading (PRD, Architecture, 28 test files analyzed)
3. ✅ Testability & Risk Assessment (14 risks identified, 5 high-priority)
4. ✅ Coverage Plan & Execution Strategy (44 tests, ~100-175 hours effort)
5. ✅ Generate Output Documents (Architecture + QA docs)

**Deliverables:**
- `test-design-architecture.md` — For Architecture/Dev teams
- `test-design-qa.md` — For QA team
- `risk-assessment-working.md` — Detailed risk analysis
- `coverage-plan-working.md` — Detailed test coverage matrix
- `test-design-progress.md` — Workflow state tracking

**Next Actions:**
1. Review both documents with Architecture/Dev and QA teams
2. Prioritize Sprint 0 blocker resolution (NOAA dataset, Palmer baseline, Track 0 infrastructure)
3. Allocate resources per estimates (~2.5-4.5 weeks for 1 QA)
4. Schedule test implementation: Track 0 ∥ Track 1 → Track 2 ∥ Track 3
5. Run `/bmad-tea-testarch-atdd` workflow for P0 test generation (separate workflow, not auto-run)

---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-02b-vision
  - step-02c-executive-summary
  - step-03-success
inputDocuments:
  - _bmad-output/v25-release-brief.md
  - _bmad-output/project-context.md
  - docs/architecture.md
  - docs/component-inventory.md
  - docs/contribution-guide.md
  - docs/deployment-guide.md
  - docs/development-guide.md
  - docs/floating_point_best_practices.md
  - docs/index.md
  - docs/project-overview.md
  - docs/pypi_release_guide.md
  - docs/source-tree-analysis.md
  - docs/test_fixture_management.md
classification:
  projectType: developer_tool
  domain: scientific
  complexity: medium
  projectContext: brownfield
workflowType: 'prd'
prdVersion: '2.5.0-wip'
---

# Product Requirements Document - climate_indices v2.5

**Author:** James Adams
**Date:** 2026-04-11
**Status:** In Progress (Steps 1–3 of 11 complete)

---

## Executive Summary

`climate_indices` is a Python library for computing standardized drought indices — SPI, SPEI, EDDI, Palmer family (PDSI, PHDI, PMDI, Z-Index), PNP, scPDSI, and more — originally developed and deployed at NOAA/NCEI for operational drought monitoring. It is published on PyPI and used by NOAA/NCEI researchers, academic climate scientists, and hydrology practitioners.

**Release goal:** v2.5 is the first release explicitly designed to support citation in published research. It addresses three gaps that currently prevent that: (1) EDDI and Palmer indices are implemented but unvalidated against primary literature; (2) xarray support is functional but undiscoverable — no examples, no compatibility matrix, absent from the README; (3) EDDI and Palmer do not appear in the README or PyPI landing page despite being fully functional since v2.3.

**Target users:**
- **Primary:** NOAA/NCEI researchers and academic climate scientists who require citable, reproducible drought index computation with clear algorithmic provenance.
- **Secondary:** Applied scientists and hydrology practitioners who need stable, CF-compliant xarray/Dask-compatible APIs for composing drought indices into larger data pipelines.

**Competitive positioning:** The closest alternative for xarray-native climate index computation is `xclim` (Ouranos). `xclim` covers a broader index catalog (~150 indices) but is a general-purpose climate indicators library with no drought-domain depth, no Palmer family, no EDDI, and no per-formula literature traceability. `climate_indices` owns the drought monitoring niche with NOAA operational provenance and a focused, citable implementation of the indices that matter to drought researchers.

**v2.5 scope:** Three epics plus infrastructure.
- *Epic 1 — Index Validation:* Algorithm reference documents (`docs/algorithm_refs/eddi.md`, `docs/algorithm_refs/palmer.md`), literature-extracted test fixtures, parametrized validation tests, `VALIDATION.md`, and a dedicated CI validation job.
- *Epic 2 — xarray Integration:* xarray compatibility audit and gap fixes, CF-convention output attributes, Dask support, and three Jupyter notebooks demonstrating SPI/SPEI, Palmer, and EDDI via the xarray API.
- *Epic 3 — Documentation Refresh:* Diátaxis-structured docs overhaul, README/PyPI landing page update surfacing EDDI and Palmer, example gallery (NClimGrid index maps), and `llms-full.txt` / `llms.txt` for AI tooling.
- *Infrastructure:* `scripts/create_github_issues.py` — idempotent issue generation from `sprint-status.yaml`; implemented before any epic story begins.

**Priority stack (if scope must be cut):**
- *P0 — must ship:* Epic 1 validation stories (literature extraction, test fixtures, `VALIDATION.md`); CI validation job; Infrastructure script; API stability.
- *P1 — ship if possible:* Epic 2 xarray gap fixes and notebooks; README/PyPI update surfacing EDDI and Palmer.
- *P2 — defer to v2.5.x if needed:* Example gallery; `llms-full.txt`; Diátaxis docs restructure.

**API stability:** Public API surfaces are stable across the v2.5 release. Breaking changes require a deprecation cycle.

**NOAA CPC fixture decision gate:** At the start of each validation story, the maintainer assesses fixture availability and adjusts scope accordingly. EDDI fixtures from NOAA CPC are not expected in the near term; validation stories for EDDI ship with literature-only test cases and explicit `pytest.mark.skip` stubs. Palmer fixtures from NOAA CPC and other authoritative sources (Cook et al. PDSI, Dai PDSI) should be actively sought and incorporated where available; any acquired fixtures are committed as small representative subsets under `tests/fixtures/palmer_literature/`.

**Out of scope:** Full EDDI validation against NOAA CPC fixtures (not expected for v2.5 — see fixture gate above); alternative Palmer variants not already in the codebase; non-CONUS example datasets; GUI; non-monthly periodicities in new notebooks.

### What Makes This Special

`climate_indices` is not an academic re-implementation of published formulas — it is the tool that ran NOAA's drought monitoring operations. The v2.5 release makes that provenance visible and defensible: every major algorithmic choice is cited to its primary source, implementation variants are explicitly documented, and known validation gaps are disclosed rather than hidden.

Positioning for v2.5: **literature-faithful implementation**. Where NOAA CPC fixture data is not yet available, the library documents exactly which formulas it follows and where validation gaps exist. Researchers can audit the choices rather than reverse-engineer them. This intellectual honesty is itself a differentiator — and the validation caveat language in the README will be framed as a transparency commitment, not a warning label.

The combination of drought-domain depth, NOAA operational provenance, scientific traceability, and production-grade Python tooling (xarray, Dask, CF conventions, stable public API) distinguishes `climate_indices` from both research codebases that lack engineering rigor and from general-purpose climate libraries that lack drought-domain depth.

## Project Classification

- **Project type:** Developer tool (Python package / scientific library)
- **Domain:** Scientific computing — climate research, drought index computation
- **Complexity:** Medium — strong domain knowledge required; numerical validation methodology is nuanced; no regulatory compliance burden
- **Project context:** Brownfield — v2.5 extends v2.4.0, a shipped library with an active user base

---

## Success Criteria

### User Success

- A researcher can cite `climate_indices` in a publication using `CITATION.cff` at the repo root and the Zenodo DOI registered at v2.5 release. `docs/algorithm_refs/eddi.md` and `docs/algorithm_refs/palmer.md` serve as the citable algorithmic references for implementation choices.
- `VALIDATION.md` maps each algorithm variant to a specific table or figure in the source paper with tolerance bounds documented — sufficient for a researcher to diagnose any output discrepancy without reverse-engineering the code.
- A new user can execute the getting-started notebook end-to-end via `nbconvert --execute` with no manual intervention, producing SPI output from an `xarray.Dataset`.
- Applied practitioners can compose EDDI, Palmer, and SPI/SPEI outputs into xarray/Dask pipelines. All public functions return outputs with CF-compliant attributes (`standard_name`, `long_name`, `units`, `valid_min`, `valid_max`) set from `cf_metadata_registry.py` — no attribute assignment required in user code.
- EDDI and Palmer are discoverable via the README and PyPI landing page. The README index table includes EDDI and Palmer with an explicit validation-status column (`validated` / `literature-only` / `pending-CPC-fixtures`).
- Invalid inputs produce exceptions from the `ClimateIndicesError` hierarchy with messages that identify the offending parameter and its valid range.
- A practitioner upgrading from v2.4 can run their existing pipeline against v2.5 with no behavioral changes to any function exported from `typed_public_api.py`.

### Business Success

- v2.5 milestone closed on GitHub; all P0 stories closed as *resolved* (not deferred). Non-P0 stories may be deferred with a comment linking to the deferral decision.
- PyPI release for v2.5 published with updated README and `CITATION.cff`. Git tag `v2.5.0` exists on the merge commit to `main`.
- Zenodo DOI registered for v2.5 and referenced in `CITATION.cff` and README.
- `CHANGELOG.md` updated with a `[2.5.0]` section listing all user-facing changes.
- `release/v2.5` merged to `main` via a single release PR; all `feature/e*` branches merged and deleted.
- `scripts/create_github_issues.py` committed and documented in `CONTRIBUTING.md`.

### Technical Success

- All `pytest` unit and integration tests pass on Python 3.10 and Python 3.12 on both Linux and macOS CI runners.
- Validation tests pass or are explicitly skipped; total skipped validation tests ≤ 3 at release, and each skip has a linked GitHub issue explaining the blocking condition.
- Each fixture file in `tests/fixtures/eddi_literature/` and `tests/fixtures/palmer_literature/` has a machine-readable JSON sidecar with `source_paper`, `doi`, `equation_ref`, `table_ref`, and `extraction_method` fields; CI asserts these fields are present and non-empty.
- Per-index numerical tolerance documented in `tests/fixtures/tolerance.yaml` with `atol`, `rtol`, and a human-authored scientific justification for each index; tolerance values are derived from source paper precision, not tuned to pass tests. Tolerance justifications are reviewed by the maintainer before merge — this is a human gate, not a CI gate.
- 90%+ branch coverage on new additions to `climate_indices/compute.py`, `climate_indices/xarray_adapter.py`, and any new modules introduced in v2.5 — enforced via `--cov-fail-under=90` scoped to those paths.
- `ruff check` and `ruff format --check` pass with zero violations across all new and modified files.
- `structlog` used throughout new and modified code; no bare `import logging` in library code.
- All new public functions have complete type hints and Google-style docstrings with `Args` and `Returns` sections.
- `mypy --strict` passes on `typed_public_api.py`; CI fails if public signatures change without a corresponding deprecation entry in `CHANGELOG.md` and an in-function `DeprecationWarning`.
- All three notebooks execute via `nbconvert --execute` in CI without unhandled exceptions; each notebook asserts ≥ 1 computed index value against a reference value drawn from `tests/fixtures/tolerance.yaml`.
- Validation tests produce results within the documented tolerance band across runs and pass on both Linux and macOS CI runners.
- v2.5 Palmer and EDDI compute performance is within 20% of v2.4 baseline on the reference dataset. The v2.4 baseline is committed as `tests/fixtures/benchmark_baseline_v240.json` before any v2.5 compute changes are made; the `pytest-benchmark` comparison job fails if the regression threshold is exceeded.
- `docs/algorithm_refs/eddi.md` and `docs/algorithm_refs/palmer.md` present, reviewed, and linked from the Reference section of the docs.
- `VALIDATION.md` present at repo root, linked from README and docs Reference section, with per-index table/figure citations and tolerance bounds.
- `CITATION.cff` present at repo root with complete author metadata and Zenodo DOI populated post-release.

### Measurable Outcomes

| Outcome | Verification | Priority |
|---|---|---|
| Literature test fixtures committed for EDDI | `tests/fixtures/eddi_literature/` non-empty; each file has JSON sidecar with all provenance fields | P0 |
| Literature test fixtures committed for Palmer | `tests/fixtures/palmer_literature/` non-empty; each file has JSON sidecar with all provenance fields | P0 |
| Fixture provenance sidecars complete | CI asserts all `*_literature/**/*.json` sidecars have `source_paper`, `doi`, `equation_ref`, `table_ref`, `extraction_method` | P0 |
| Validation skip count bounded | CI fails if skipped `@pytest.mark.validation` tests > 3 at merge time | P0 |
| Numerical tolerances documented | `tests/fixtures/tolerance.yaml` present with per-index atol/rtol and human-reviewed scientific justification | P0 |
| Coverage floor on new compute paths | `--cov-fail-under=90` on `compute.py`, `xarray_adapter.py`, and new v2.5 modules passes in CI | P0 |
| No performance regression | Benchmark CI job: v2.5 within 20% of `benchmark_baseline_v240.json`; baseline file committed before compute changes | P0 |
| Validation CI job passes (skips OK, failures not OK) | GitHub Actions validation job green on Python 3.10 + 3.12, Linux and macOS | P0 |
| `CITATION.cff` present and valid | File present at repo root; `cff-validator` passes in CI | P0 |
| Three notebooks execute in CI | `nbconvert --execute` job green; each asserts ≥ 1 index value against `tolerance.yaml` reference | P1 |
| Backward compatibility confirmed | Existing v2.4 usage patterns for all `typed_public_api.py` exports pass in a dedicated compat test module | P1 |
| Zenodo DOI registered | DOI populated in `CITATION.cff` and README before PyPI release; git tag `v2.5.0` present | P1 |
| `CHANGELOG.md` updated | `[2.5.0]` section present with user-facing changes listed | P1 |
| xarray compatibility matrix published | `docs/xarray_compatibility.md` present and linked from docs | P1 |
| `llms-full.txt` and `llms.txt` generated | Files present at repo root; `uv run python scripts/generate_llms_txt.py` succeeds | P2 |
| Gallery PNGs present | `docs/gallery/` contains index maps for SPI-3, SPI-6, SPEI-3, PDSI, PHDI, Z-Index, EDDI-3 | P2 |

## Product Scope

### Pre-work (before any epic story begins)

- `scripts/create_github_issues.py` implemented, merged to `release/v2.5`, and run to create all v2.5 GitHub issues.
- `tests/fixtures/benchmark_baseline_v240.json` generated from the v2.4.0 tag and committed to `release/v2.5` before any compute changes land.

### MVP — Minimum Viable Product (P0)

The release ships when these are complete, regardless of P1/P2 status:

- Epic 1 (all stories): literature extraction docs, fixture files with provenance sidecars, parametrized validation tests, `VALIDATION.md` with per-index paper citations and tolerance bounds, CI validation job.
- xarray compatibility audit for Epic 1 functions and any bug fixes causing incorrect output (not merely cosmetic issues) when xarray inputs are passed to EDDI/Palmer functions.
- API stability: `mypy --strict` on `typed_public_api.py` enforced in CI; no breaking changes without a deprecation cycle (entry in `CHANGELOG.md` + in-function `DeprecationWarning`).
- Platform-stability CI: unit, integration, and validation tests run on Python 3.10 + 3.12, Linux and macOS.
- `ruff` and `structlog` compliance across all new code.
- `CITATION.cff` present at repo root and valid per `cff-validator`.

### Growth Features — v2.5 Complete (P1)

Shipped as part of v2.5 unless time-constrained:

- Epic 2: full xarray gap fixes, CF-convention output attributes from `cf_metadata_registry.py`, Dask support via `xr.apply_ufunc`, all three Jupyter notebooks with output assertions drawn from `tolerance.yaml`.
- README and PyPI landing page updated to surface EDDI and Palmer with a validation-status table.
- `docs/xarray_compatibility.md` matrix published.
- Zenodo DOI registered and populated in `CITATION.cff` and README; git tag `v2.5.0` applied.
- `CHANGELOG.md` `[2.5.0]` section complete.
- Backward-compatibility test module for all `typed_public_api.py` exports.

### Vision — Future (P2 / v2.6+)

Defer if P0/P1 are at risk:

- Epic 3 (docs): full Diátaxis restructure, example gallery PNGs, `llms-full.txt`/`llms.txt`.
- Provenance-in-data: computation metadata (formula version, calibration window, literature citation) embedded directly in xarray output CF attributes.
- Full NOAA CPC fixture validation for EDDI and Palmer once fixtures are received.

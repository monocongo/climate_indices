---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-02b-vision
  - step-02c-executive-summary
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

- A researcher can cite `climate_indices` in a publication and point to `docs/algorithm_refs/eddi.md` or `docs/algorithm_refs/palmer.md` as the citable algorithmic reference for the implementation choices made.
- A researcher encountering a discrepancy between `climate_indices` output and another implementation can diagnose the source using `VALIDATION.md`, the algorithm reference docs, and test provenance comments — without reverse-engineering the code.
- A new user can go from `pip install climate-indices` to computing SPI on an `xarray.Dataset` using the getting-started notebook in a single session, without consulting the source code.
- Applied practitioners can compose EDDI, Palmer, and SPI/SPEI outputs into xarray/Dask pipelines using CF-compliant outputs with no custom glue code.
- EDDI and Palmer are discoverable via the README and PyPI landing page, with validation status clearly communicated.

### Business Success

- v2.5 milestone closed on GitHub; all v2.5 issues resolved or explicitly deferred with a comment.
- PyPI release for v2.5 published with updated README.
- `release/v2.5` merged to `main` via a single release PR; all `feature/e*` branches merged and deleted.
- `main` CI green post-merge.
- `scripts/create_github_issues.py` committed, documented in `CONTRIBUTING.md`, and used as the sole mechanism for bulk issue creation.

### Technical Success

- All `pytest` unit and integration tests pass.
- All validation tests pass or are explicitly skipped with `pytest.mark.skip(reason="...")` citing the blocking condition and a reference.
- Literature-extracted numerical test cases committed under `tests/fixtures/eddi_literature/` and `tests/fixtures/palmer_literature/` with provenance comments citing paper, equation, and table.
- `ruff check` and `ruff format --check` pass with zero violations across all new and modified files.
- `structlog` used throughout new and modified code; no bare `import logging` in library code.
- All new public functions have complete type hints and Google-style docstrings with `Args` and `Returns` sections.
- All three notebooks execute without unhandled cell exceptions via `nbconvert --execute` in CI.
- `docs/algorithm_refs/eddi.md` and `docs/algorithm_refs/palmer.md` present, reviewed, and linked from the Reference section of the docs.
- `VALIDATION.md` present at repo root, linked from README and docs Reference section.

### Measurable Outcomes

| Outcome | Verification |
|---|---|
| Literature test fixtures committed for EDDI | `tests/fixtures/eddi_literature/` non-empty; each file cites source |
| Literature test fixtures committed for Palmer | `tests/fixtures/palmer_literature/` non-empty; each file cites source |
| Validation CI job passes (skips OK, failures not OK) | GitHub Actions validation job green |
| Three notebooks execute in CI | `nbconvert --execute` job green |
| `llms-full.txt` and `llms.txt` generated | Files present at repo root; `uv run python scripts/generate_llms_txt.py` succeeds |
| Gallery PNGs present | `docs/gallery/` contains index maps for SPI-3, SPI-6, SPEI-3, PDSI, PHDI, Z-Index, EDDI-3 |
| xarray compatibility matrix published | `docs/xarray_compatibility.md` present and linked from docs |

## Product Scope

### MVP — Minimum Viable Product (P0)

The release ships when these are complete, regardless of P1/P2 status:

- Infrastructure: `scripts/create_github_issues.py` implemented and all v2.5 issues created.
- Epic 1 (all stories): literature extraction docs, fixture files, parametrized validation tests, `VALIDATION.md`, CI validation job.
- API stability: no breaking changes to public surfaces without a deprecation cycle.
- `ruff` and `structlog` compliance across all new code.

### Growth Features — v2.5 Complete (P1)

Shipped as part of v2.5 unless time-constrained:

- Epic 2: xarray compatibility audit and gap fixes, CF-convention output attributes, Dask support, all three Jupyter notebooks.
- README and PyPI landing page updated to surface EDDI and Palmer with validation status caveats.
- `docs/xarray_compatibility.md` matrix published.

### Vision — Future (P2 / v2.6+)

Defer if P0/P1 are at risk:

- Epic 3 (docs): full Diátaxis restructure, example gallery PNGs, `llms-full.txt`/`llms.txt`.
- Provenance-in-data: computation metadata (formula version, calibration window, literature citation) embedded directly in xarray output CF attributes.
- Full NOAA CPC fixture validation for EDDI and Palmer once fixtures are received.

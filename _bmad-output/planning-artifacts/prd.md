---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-02b-vision
  - step-02c-executive-summary
  - step-03-success
  - step-04-journeys
  - step-05-domain
  - step-06-innovation
  - step-07-project-type
  - step-08-scoping
  - step-09-functional
  - step-10-nonfunctional
  - step-11-polish
  - step-12-complete
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
prdVersion: '2.5.0'
completedAt: '2026-04-12'
---

# Product Requirements Document - climate_indices v2.5

**Author:** James Adams
**Date:** 2026-04-11
**Status:** Complete

---

## Executive Summary

`climate_indices` is a Python library for computing standardized drought indices — SPI, SPEI, EDDI, Palmer family (PDSI, PHDI, PMDI, Z-Index), PNP, scPDSI, and more — with algorithmic lineage from NOAA/CPC operational drought monitoring code. It is published on PyPI and used by NOAA/NCEI researchers, academic climate scientists, and hydrology practitioners.

**Release goal:** v2.5 is the first release explicitly designed to support citation in published research. The library's core implementations are functional; v2.5 establishes algorithmic traceability to published literature by addressing three gaps that currently prevent citation: (1) EDDI and Palmer indices are implemented but unvalidated against primary literature; (2) xarray support is functional but undiscoverable — no examples, no compatibility matrix, absent from the README; (3) EDDI and Palmer do not appear in the README or PyPI landing page despite being fully functional since v2.3.

**Target users:**
- **Primary:** NOAA/NCEI researchers and academic climate scientists who require citable, reproducible drought index computation with clear algorithmic provenance.
- **Secondary:** Applied scientists and hydrology practitioners who need stable, CF-compliant xarray/Dask-compatible APIs for composing drought indices into larger data pipelines.

**Competitive positioning:** The closest alternative for xarray-native climate index computation is `xclim` (Ouranos). `xclim` covers a broader index catalog (~150 indices) but is a general-purpose climate indicators library with no drought-domain depth, no Palmer family, no EDDI, and no per-formula literature traceability. MetPy (Unidata/NSF) overlaps in scope for some atmospheric indices but does not include Palmer, EDDI, or drought-domain depth. `climate_indices` holds the drought monitoring niche with algorithmic lineage from NOAA documentation and CPC operational code, and a focused, citable implementation of the indices that matter to drought researchers.

**v2.5 scope:** Three epics plus infrastructure.
- *Epic 1 — Index Validation:* Algorithm reference documents (`docs/algorithm_refs/eddi.md`, `docs/algorithm_refs/palmer.md`), literature-extracted test fixtures, parametrized validation tests, `VALIDATION.md`, and a dedicated CI validation job.
- *Epic 2 — xarray Integration:* xarray compatibility audit and gap fixes, CF-convention output attributes, Dask support, and three Jupyter notebooks demonstrating SPI/SPEI, Palmer, and EDDI via the xarray API.
- *Epic 3 — Documentation Refresh:* Diátaxis-structured docs overhaul, README/PyPI landing page update surfacing EDDI and Palmer, example gallery (NClimGrid index maps), and `llms-full.txt` / `llms.txt` for AI tooling.
- *Infrastructure:* `scripts/create_github_issues.py` — idempotent issue generation from `sprint-status.yaml`; implemented before any epic story begins.

**Priority stack (if scope must be cut):**
- *P0 — must ship:* Epic 1 validation stories (literature extraction, test fixtures, `VALIDATION.md`); CI validation job; Infrastructure script; API stability.
- *P1 — ship if possible:* Epic 2 xarray gap fixes and notebooks; README/PyPI update surfacing EDDI and Palmer.
- *P2 — defer to v2.5.x if needed:* Example gallery; `llms-full.txt`; Diátaxis docs restructure.

**API stability:** Public API surfaces are stable across the v2.5 release. Breaking changes require a deprecation cycle. `climate_indices` follows semantic versioning: breaking changes to `typed_public_api.py` are permitted only in major releases (v3.x+).

**NOAA CPC fixture decision gate:** At the start of each validation story, the maintainer assesses fixture availability and adjusts scope accordingly. EDDI fixtures from NOAA CPC are not expected before v2.5 ships; validation stories for EDDI ship with literature-only fixtures capturing algorithm-step intermediate values (plotting-position values, gamma fit parameters — not station-calibrated operational outputs) and explicit `pytest.mark.skip` stubs. This gap is disclosed in `VALIDATION.md` and the README validation-status table. Palmer fixtures from NOAA CPC and other authoritative sources (Cook et al. PDSI, Dai PDSI) should be actively sought; any acquired fixtures are committed as small representative subsets under `tests/fixtures/palmer_literature/`.

**Out of scope:** Full EDDI validation against NOAA CPC fixtures (not expected for v2.5 — see fixture gate above); alternative Palmer variants not already in the codebase; non-CONUS example datasets; GUI; non-monthly periodicities in new notebooks.

---

## Competitive Position & Design Philosophy

`climate_indices` has algorithmic lineage from NOAA CPC operational drought monitoring code — it is not an academic re-implementation but a library with documented provenance in operational practice. The v2.5 release makes that lineage visible and defensible: every major algorithmic choice is cited to its primary source, implementation variants are explicitly documented, and known validation gaps are disclosed rather than hidden.

Positioning for v2.5: **literature-faithful implementation**. Where NOAA CPC fixture data is not yet available, the library documents exactly which formulas it follows and where validation gaps exist. Researchers can audit the choices rather than reverse-engineer them. This intellectual honesty is itself a differentiator — the validation caveat language in the README is framed as a transparency commitment, not a warning label.

The combination of drought-domain depth, algorithmic lineage from NOAA documentation, scientific traceability, and production-grade Python tooling (xarray, Dask, CF conventions, stable public API) distinguishes `climate_indices` from both research codebases that lack engineering rigor and from general-purpose climate libraries (xclim, MetPy) that lack drought-domain depth.

**Long-term platform direction — algorithm research and selection:** A secondary strategic goal, explicit from v2.5 onward, is to evolve `climate_indices` into a platform that supports comparison and selection across competing algorithm implementations of the same drought index. Palmer indices in particular have multiple published variants (original Palmer 1965, Alley 1984 moisture-loss correction, Wells et al. 2004 scPDSI, among others) that produce meaningfully different results on the same input. Rather than arbitrarily fixing one variant as canonical, the library's architecture is designed to support parameterized algorithm dispatch — a caller can select `algorithm="original_1965"` or `algorithm="alley_1984"` — and the validation fixture structure is organized by (index, algorithm variant, dataset source) tuple. This supports future research on algorithm selection criteria, implementation efficiency, and cross-dataset reproducibility. In v2.5 only one implementation per index is registered; the dispatch parameter and fixture directory structure are established now to avoid a breaking API change when a second variant is added. Full multi-implementation comparison tooling is deferred to v3.x.

---

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
- `ruff check` and `ruff format --check` pass with zero violations across all new and modified files.
- `structlog` used throughout new and modified code; no bare `import logging` in library code.
- All new public functions have complete type hints and Google-style docstrings with `Args`, `Returns`, and `Examples` sections.
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
| No performance regression | Benchmark CI job: v2.5 within 20% of `benchmark_baseline_v240.json`; baseline committed from v2.4.0 tag before any compute changes | P0 |
| `CITATION.cff` present and valid | File present at repo root; `cff-validator` passes in CI | P0 |
| Backward compatibility confirmed | Existing v2.4 usage patterns for all `typed_public_api.py` exports pass in a dedicated compat test module | P1 |
| Zenodo DOI registered | DOI populated in `CITATION.cff` and README before PyPI release; git tag `v2.5.0` present | P1 |
| xarray compatibility matrix published | `docs/xarray_compatibility.md` present and linked from docs | P1 |
| Gallery PNGs present | `docs/gallery/` contains index maps for SPI-3, SPI-6, SPEI-3, PDSI, PHDI, Z-Index, EDDI-3 | P2 |

---

## Product Scope

### Pre-work (before any epic story begins)

- `scripts/create_github_issues.py` implemented, merged to `release/v2.5`, and run to create all v2.5 GitHub issues.
- `tests/fixtures/benchmark_baseline_v240.json` generated from the v2.4.0 tag and committed to `release/v2.5` before any compute changes land.
- A minimal `CHANGELOG.md` entry documenting the Palmer moisture anomaly correction (required by NFR-REPR-3) is created as part of Epic 1. The full `[2.5.0]` release section is completed at P1.

### MVP — Minimum Viable Product (P0)

The release ships when these are complete, regardless of P1/P2 status:

- Epic 1 (all stories): literature extraction docs, fixture files with provenance sidecars, parametrized validation tests, `VALIDATION.md` with per-index paper citations and tolerance bounds, CI validation job.
- xarray compatibility audit for Epic 1 functions: bug fixes that produce incorrect output when xarray inputs are passed to EDDI or Palmer functions. Cosmetic attribute gaps (missing `standard_name`, `long_name`, etc.) and Dask support are P1 scope.
- API stability: `mypy --strict` on `typed_public_api.py` enforced in CI; no breaking changes to `typed_public_api.py` exports without a deprecation cycle (entry in `CHANGELOG.md` + in-function `DeprecationWarning`). The no-breaking-changes guarantee applies to all exports present at the v2.5.0 release tag.
- Platform-stability CI: unit, integration, and validation tests run on Python 3.10 + 3.12, Linux and macOS.
- `ruff` and `structlog` compliance across all new and modified code.
- `CITATION.cff` present at repo root and valid per `cff-validator`.

### Growth Features — v2.5 Complete (P1)

Shipped as part of v2.5 unless time-constrained:

- Epic 2: full xarray gap fixes, CF-convention output attributes from `cf_metadata_registry.py`, Dask support via `xr.apply_ufunc`, all three Jupyter notebooks with output assertions drawn from `tolerance.yaml`.
- README and PyPI landing page updated to surface EDDI and Palmer with a validation-status table.
- `docs/xarray_compatibility.md` matrix published.
- Zenodo DOI registered and populated in `CITATION.cff` and README; git tag `v2.5.0` applied.
- Full `CHANGELOG.md` `[2.5.0]` section complete.
- Backward-compatibility test module for all `typed_public_api.py` exports.

### Vision — Future (P2 / v2.6+)

Defer if P0/P1 are at risk:

- Epic 3 (docs): full Diátaxis restructure, example gallery PNGs, `llms-full.txt`/`llms.txt`.
- Provenance-in-data: computation metadata (formula version, calibration window, literature citation) embedded directly in xarray output CF attributes.
- Full NOAA CPC fixture validation for EDDI and Palmer once fixtures are received.

---

## User Journeys

The following table maps capabilities to the journeys that depend on them. Use it as an orientation map before reading the narrative journeys below.

| Capability | Journeys | Priority |
|---|---|---|
| `CITATION.cff` + Zenodo DOI | Maya, Kenji | P0 |
| README dual-citation guidance (library + algorithm papers) | Maya | P0 |
| `VALIDATION.md` with proactive tolerance disclosure and equation-level justification | Maya, Reza | P0 |
| `docs/algorithm_refs/eddi.md` and `palmer.md` — self-sufficient under peer review, version-annotated | Maya, Reza, Kenji | P0 |
| Fixture files with intermediate computational checkpoints in JSON sidecar | Reza | P0 |
| `pytest -m validation` runner | Reza | P0 |
| `typed_public_api.py` stable surface + `mypy --strict` | Anika | P0 |
| `ClimateIndicesError` hierarchy with parameter-level messages | Reza, Anika | P0 |
| `CHANGELOG.md` with numerical-output-level change documentation | Kenji, Ben | P1 |
| Version annotations in algorithm reference docs | Kenji | P1 |
| CF-compliant output attributes from `cf_metadata_registry.py` | Maya, Ben, Anika | P1 |
| `docs/xarray_compatibility.md` with Dask chunk constraints documented | Ben | P1 |
| Dask support via `xr.apply_ufunc` with chunk-boundary safety | Ben | P1 |
| Backward-compatible API upgrade from v2.4 | Ben | P1 |
| README validation-status table (library + per-index) | Maya, Ben | P1 |
| Stateless, concurrency-safe public functions | Anika | P1 |
| Getting-started notebook executable without intervention | Maya, Anika | P1 |
| xarray coordinate preservation (or explicit documentation of drop behavior) | Anika | P1 |
| Computation parameters readable from xarray output attributes | Reza, Ben | P1 |

---

### Journey 1 — Dr. Maya Chen, NOAA/NCEI Research Scientist (Primary — Citation Success Path)

**Situation:** Maya is preparing a drought attribution study for submission to *Journal of Climate*. She needs to compute 12-month PDSI over CONUS from 1950–2020 and cite her drought index implementation in the methods section. Her advisor pushed back on her previous draft: "you can't cite 'a Python script' as your drought index methodology."

**Opening Scene:** Maya finds `climate_indices` on PyPI while searching for "Palmer drought index Python citable." She notes PDSI listed in the README with a validation-status column marked `literature-only`, and follows the link to `VALIDATION.md`.

**Rising Action:**
1. She reads `VALIDATION.md` — it maps PDSI to specific tables in Palmer (1965) with `atol=1e-2` and a justification. She understands what "literature-only" means before she writes a line of code.
2. She installs the package and runs the getting-started notebook via `nbconvert --execute`. SPI output from an `xarray.Dataset` in under 20 minutes.
3. She extends to PDSI. The function returns CF-compliant xarray output — `standard_name`, `long_name`, `valid_min`, `valid_max` already set from `cf_metadata_registry.py`. She pastes these directly into her methods table.
4. She opens `docs/algorithm_refs/palmer.md`, finds the implementation variant note for moisture anomaly weighting (Eq. 12 vs. Alley 1984), and copies the Zenodo DOI into her reference manager.
5. She follows the README's dual-citation guidance: she cites Hobbins et al. (2016) for the algorithm and `climate_indices` v2.5.0 (Zenodo DOI) for the implementation. The README explains this distinction explicitly.

**Climax:** Peer review. Reviewer 2 asks which variant of the moisture anomaly weighting was used and demands a reference. Maya opens `docs/algorithm_refs/palmer.md` directly — the doc is self-contained, no maintainer intervention needed. She pastes the section URL into her response.

**Resolution:** Paper accepted. The Zenodo DOI pins the exact library version. `VALIDATION.md` discloses the known CPC fixture gap. The algorithm reference doc is the audit trail. Maya recommends the library to her lab.

---

### Journey 2 — Prof. Reza Ahmadi, Academic Climate Scientist (Primary — Discrepancy Debugging)

**Situation:** Reza is replicating a published EDDI analysis for a methods comparison paper. He runs `compute_eddi` on the same input data as Hobbins et al. (2016) Table 3 and gets values that differ by 0.08 at one station.

**Opening Scene:** Reza stares at two columns of numbers. 0.08 is small enough to ignore, large enough to demand explanation. He doesn't know if it's a bug in the library, a rounding convention in the paper, a difference in which PET formulation he's using, or a mistake in his own code. The uncertainty is the worst part.

**Rising Action:**
1. He doesn't immediately read `VALIDATION.md`. He first tries to reproduce the table value himself in a notebook — isolating his input data handling.
2. He opens `docs/algorithm_refs/eddi.md` and finds the "Implementation Notes — Reference ET source" section, which identifies that the library follows Hobbins et al. Eq. 4 using Hargreaves PET. His reference ET matches.
3. He locates the EDDI literature fixture at `tests/fixtures/eddi_literature/hobbins_2016_table3.csv`. The JSON sidecar documents `equation_ref`, `table_ref`, and — critically — intermediate computational checkpoints: the plotting-position values and the gamma fit parameters before the final transformation. He compares his intermediate values to the fixture's intermediate values. They match.
4. He runs `pytest -m validation`. It passes. He now reads `VALIDATION.md`: tolerance for this fixture is `atol=1e-2`, derived from the 2-decimal rounding in the published table. His 0.08 is within tolerance.

**Climax:** Reza closes the investigation. The discrepancy was rounding in the published table — not a library bug. He can cite this conclusion because the fixture chain (input → intermediate checkpoints → output) gave him the evidence to localize the divergence at the right algorithmic boundary.

**Resolution:** Reza adds a footnote: "EDDI values were verified against `climate_indices` v2.5.0 (Zenodo DOI), which documents Hobbins et al. (2016) Table 3 tolerance in `VALIDATION.md`." The library was trustworthy *because he could debug it to the equation level*, not because it told him everything was fine.

---

### Journey 3 — Dr. Kenji Okafor, Reproducer (Primary — Prior-Study Replication)

**Situation:** Kenji is writing a methods comparison paper. He wants to reproduce the PDSI values from a 2021 paper by a different research group that also used `climate_indices`, but an older version (v2.2). His values differ from theirs by 0.15 at some stations.

**Opening Scene:** Kenji opens the 2021 paper's methods section. It says "PDSI was computed using climate_indices." Version unspecified. No link. Kenji installs v2.5.0 and runs it. The 0.15 difference could be a version change, a calibration window choice, a different reference period, or different PET inputs. He has no way to tell.

**Rising Action:**
1. He checks `CHANGELOG.md` for the `[2.5.0]` and earlier sections. He finds a note: "Palmer computation: fixed moisture loss calculation in months with PET > precipitation — previously underestimated by ~0.12 in arid months. Introduced in v2.4.1." This is likely the source.
2. He opens `docs/algorithm_refs/palmer.md` — it documents which variant of each Palmer equation the library implements, with version annotations for when each variant was adopted.
3. He cannot reproduce the exact 2021 values without the original team's data and version, but he can now *document the discrepancy's cause* — which is sufficient for his paper's methods section.

**Climax:** Kenji emails the original authors. They respond: "we used v2.2, which had the moisture loss bug." Kenji's methods section notes this explicitly. Both papers are now more useful to future researchers.

**Resolution:** The library's detailed `CHANGELOG.md` and algorithm reference docs turned what could have been an unresolvable discrepancy into a documented, traceable version difference. Kenji cites `climate_indices` v2.5.0 and pins his own version explicitly.

---

### Journey 4 — Ben Torres, Hydrology Practitioner at a Water Utility (Secondary — Dask Pipeline Upgrade)

**Situation:** Ben maintains an automated weekly drought monitoring report on a Dask cluster. He needs to add SPEI-6 and EDDI-3 alongside the existing SPI pipeline. He is not a climate scientist.

**Opening Scene:** Ben upgrades from v2.4 to v2.5. His first instinct is to check whether the upgrade broke his archived SPI values — a decade of weekly outputs the utility has stored. API breaks he can catch with tests; silent numerical regressions he cannot.

**Rising Action:**
1. He reads `CHANGELOG.md` under `[2.5.0]`. No Palmer or SPI compute changes — only xarray attribute improvements and the new EDDI/SPEI CF metadata. His archived values are safe.
2. He checks `docs/xarray_compatibility.md` — SPEI and EDDI both accept Dask-chunked input, with a note: "chunk along the time axis only; rolling-window computations do not compose correctly with spatial chunks." He adjusts his chunk schema.
3. He adds `compute_spei` and `compute_eddi` calls. Both return CF-compliant `xarray.DataArray` output with `units`, `standard_name`, and `valid_min`/`valid_max`. He doesn't need to look up what the attributes mean.
4. He runs a dry-run on a single Dask worker to verify chunk boundaries don't truncate the calibration period. It passes.

**Climax:** Ben ships SPEI-6 and EDDI-3 in the next weekly report. The board asks what the new indices mean. He sends them the README's validation-status table and the algorithm reference links.

**Resolution:** Ben has a stable pipeline. The `CHANGELOG.md` gave him the confidence to upgrade without re-validating everything. The compatibility matrix prevented the chunk-boundary bug before it happened.

---

### Journey 5 — Anika Patel, Data Engineer (API Consumer — REST Integration Evaluation)

**Situation:** Anika is wrapping `climate_indices` into REST endpoints at a geospatial SaaS company. She has one afternoon to evaluate whether the library is safe to depend on.

**Opening Scene:** Anika opens a terminal. She installs the library. She imports `compute_spi` in a Python REPL and immediately asks: is this function safe to call concurrently across HTTP requests? She's seen libraries with module-level state that corrupts under load.

**Rising Action:**
1. She checks the source of `compute_spi` and finds no module-level mutable state — the function is stateless and can be called concurrently. She verifies this holds for SPEI and EDDI via the same inspection pattern.
2. She checks `typed_public_api.py` — the public surface is small and explicit. `mypy --strict` passes on her wrapper code.
3. She calls `compute_spi` with a custom-named coordinate (`lat/lon` instead of `latitude/longitude`) and discovers the function returns an xarray output that drops the non-standard coordinate name silently. She files a mental note: her wrapper must re-attach coordinates explicitly.
4. She runs the getting-started notebook via `nbconvert --execute` to verify the output shape matches what she expects for a 480-month timeseries.
5. She reads the `ClimateIndicesError` hierarchy to map library exceptions to HTTP status codes. The error messages include the offending parameter name and valid range — she can surface these directly to API consumers.

**Climax:** Anika presents to her tech lead. He asks about API stability. She shows `typed_public_api.py`, the deprecation policy in `CHANGELOG.md`, and `mypy --strict` compliance. He approves.

**Resolution:** Anika ships PDSI REST endpoints within the week. Her wrapper explicitly re-attaches coordinates. She opens a GitHub issue flagging the silent coordinate-drop behavior as a potential footgun for downstream users.

---

### Journey 6 — Dr. Fatima Al-Rashid, Algorithm Comparison Researcher *(v3.x target — out of scope for v2.5)*

**Situation:** Fatima is a hydrology researcher evaluating which Palmer variant best reproduces observed streamflow anomalies in semi-arid basins. She needs to run the same input data through `original_1965`, `alley_1984`, and `wells_2004` Palmer implementations side-by-side and compare their outputs against observed drought records.

**Why documented here:** This persona is out of scope for v2.5 but drives two architectural decisions made in v2.5: (1) the `algorithm` dispatch parameter on Palmer and EDDI functions, established now to avoid a breaking API change later; (2) the per-(index, algorithm variant, dataset source) fixture directory structure. Fatima's journey is the design target for v3.x multi-implementation comparison tooling.

**v2.5 supports:** Running one registered algorithm variant per index. Fatima can use `climate_indices` v2.5 but cannot yet switch between variants programmatically or access a comparison framework.

**v3.x will support:** Multiple registered implementations per index, parameterized dispatch, cross-dataset comparison fixtures, and tooling for selection criteria research.

---

## Platform & API Requirements

### Project-Type Overview

`climate_indices` is a scientific Python package distributed via PyPI. The stable public API is importable from `climate_indices.typed_public_api`; v2.5 extends that surface with CF-compliant xarray output, validation infrastructure, and Jupyter notebook examples.

### Language & Version Matrix

| Python Version | Support Status | CI-Tested |
|---|---|---|
| 3.10 | Supported (minimum) | Yes — Linux and macOS |
| 3.11 | Supported | No (expected to work) |
| 3.12 | Supported | Yes — Linux and macOS |
| 3.13 | Supported | No (expected to work) |
| 3.14 | Supported | No (expected to work) |

CI explicitly tests the minimum (3.10) and current stable (3.12) versions on both Linux and macOS. Versions 3.11, 3.13, and 3.14 are within the supported range and are expected to work but are not included in the CI matrix for v2.5.

Dependency version floors are defined in `pyproject.toml`. The xarray compatibility matrix (`docs/xarray_compatibility.md`) documents the minimum xarray, NumPy, and SciPy versions tested.

### Installation Methods

Primary distribution channel: PyPI.

```bash
pip install climate-indices   # user install
uv sync                       # development environment
```

Conda-forge packaging, Binder, and Docker notebook environments are out of scope for v2.5.

### API Surface

The stable public API is `climate_indices.typed_public_api`. Functions in `climate_indices.compute` are accessible to advanced users but carry no backward-compatibility guarantee beyond what is re-exported through `typed_public_api`.

- All exports from `typed_public_api.py` present at the v2.5.0 release tag are covered by the v2.5 backward-compatibility guarantee
- `mypy --strict` passes on `typed_public_api.py` in CI
- No breaking changes without a deprecation entry in `CHANGELOG.md` and an in-function `DeprecationWarning`

### Code Examples

Two complementary channels:

1. **Jupyter Notebooks** (Epic 2): three end-to-end notebooks demonstrating SPI/SPEI, Palmer, and EDDI via the xarray API. Each notebook is executable via `nbconvert --execute` in CI and asserts ≥ 1 computed index value against a reference from `tests/fixtures/tolerance.yaml`.

2. **Docstring Examples**: all new and modified public functions in v2.5 must include a Google-style `Examples:` section with at least one short, self-contained usage example. Docstring examples are the inline discoverable reference for API consumers.

### Migration Guide

v2.5 is backward-compatible with v2.4 for all `typed_public_api.py` exports. No migration steps are required.

Documentation note for `CHANGELOG.md`: EDDI and Palmer xarray outputs gain CF-compliant attributes (`standard_name`, `long_name`, `units`, `valid_min`, `valid_max`) in v2.5 where previously absent. Downstream code that constructs attribute dicts manually may need to remove now-redundant assignments.

### Implementation Considerations

- `uv` is the canonical development environment manager; `pip` is the user-facing install path
- `ruff check` and `ruff format --check` enforced in CI on all new and modified files
- `structlog` required throughout; no bare `import logging` in library code
- Type hints and Google-style docstrings with `Args`, `Returns`, and `Examples` sections required on all new and modified public functions
- **Algorithm dispatch parameter:** Palmer and EDDI public functions accept an `algorithm: str` keyword argument (default `"original_1965"` for Palmer, `"hobbins_2016"` for EDDI) that dispatches to a registered implementation. In v2.5 only one implementation is registered per index; the parameter is established now to avoid a breaking API change when additional variants are added. Unrecognized `algorithm` values raise `InvalidArgumentError` with valid options listed.
- **Fixture directory structure:** Validation fixtures are organized by `tests/fixtures/{index}/{algorithm_variant}/{dataset_source}/` to support per-(index, variant, dataset) tolerance bands. This structure is adopted in v2.5 even when only one variant exists per index.
- **Regression baseline generation:** `scripts/generate_baselines.py` generates v2.4.0 reference outputs for NFR-REPR-4 baseline comparisons. It must be run against the v2.4.0 tag before any v2.5 compute changes are made, with outputs committed to `tests/fixtures/regression/v2.4.0/`.

---

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** Release validation — the library's implementations are functional; v2.5 establishes algorithmic traceability to published literature, scientific credibility, and discoverability.

**Resource:** Solo maintainer. All scope decisions assume single-person bandwidth. No parallel story execution; epic sequencing is Epic 1 → Epic 2 → Epic 3.

### MVP Feature Set (P0)

For the complete P0 feature list and supported user journeys, see [MVP — Minimum Viable Product (P0)](#mvp--minimum-viable-product-p0) in the Product Scope section above.

### Risk Mitigation

**Technical Risks:**

- *Palmer implementation correctness* — the highest-priority technical risk. Palmer index computation (PDSI, PHDI, PMDI, scPDSI) involves multi-step numerical procedures that have been difficult to validate manually in prior attempts. The Palmer validation story (Story 1.5) **must be executed with Claude Opus** to maximize reasoning depth on numerical implementation. Story scope requires explicit comparison of each formula variant in `docs/algorithm_refs/palmer.md` against the library's implementation; any discrepancy is treated as a potential bug until traced to a deliberate implementation choice.

- *EDDI fixture availability* — NOAA CPC fixtures are not expected before v2.5 ships. The maintainer is in active contact with the EDDI algorithm author (Hobbins) and NOAA CPC; reference fixtures are expected within approximately six months of the v2.5.0 release, which scopes this gap to a v2.5.x or v2.6 story. EDDI validation ships in v2.5 with literature-only fixtures that capture algorithm-step intermediate values (plotting-position values, gamma fit parameters) rather than station-calibrated operational outputs, because published EDDI literature does not provide station-specific numerical outputs for independent verification; the gap is disclosed in `VALIDATION.md` and the README validation-status table. Validation tests blocked on CPC fixtures are marked `@pytest.mark.fixture_pending` with a linked GitHub issue assigned to the `v2.5-fixture-delivery` milestone; they are distinct from generic `@pytest.mark.skip` usage and are tracked separately in CI. If NOAA CPC fixture data becomes available during the v2.5 cycle, fixtures are incorporated under `tests/fixtures/eddi_literature/` and `fixture_pending` stubs are removed. Palmer fixtures from Cook et al., Dai et al., and similar published datasets are expected to be locatable before v2.5 ships; any acquired fixtures are committed as small representative subsets.

- *Xarray deferral cost* — Epic 2 (full xarray integration) is P1. Users who require Dask support or CF-compliant output attributes for Palmer and EDDI will not receive these until Epic 2 ships. This is a known cost of the Epic 1→2→3 sequencing decision; it is mitigated by the P0 xarray compatibility audit (bug fixes for incorrect output on xarray inputs) which ensures existing xarray usage continues to work correctly.

**Resource Risks:**
- Solo maintainer: if bandwidth is constrained, P1 features are deferred before any P0 feature is cut. The release ships when all P0 stories are resolved, regardless of P1/P2 status.

**Market Risks:**
- Minimal — v2.5 is an improvement release for an established library with an active user base. The primary discovery risk (EDDI and Palmer not surfaced in README) is addressed by P1 README/PyPI updates.

---

## Functional Requirements

> This section is the capability contract for all downstream work. UX designers, architects, and epic breakdown will only implement what is listed here. FRs without a priority annotation are P0. FRs marked *(P1 — Epic 2)* ship as part of Epic 2 and are not required for the P0 release.

### Index Computation

- FR1: A user can compute SPI, SPEI, EDDI, PDSI, PHDI, PMDI, Z-Index, scPDSI, and PNP from array inputs
- FR2: A user can compute any index using xarray DataArray or Dataset inputs in addition to NumPy arrays *(P1 — Epic 2)*
- FR3: A user can pass Dask-chunked xarray inputs to public index computation functions and receive Dask-backed output *(P1 — Epic 2)*

### Validation Infrastructure

- FR4: A researcher can run the validation test suite independently from unit and integration tests
- FR5: A researcher can verify EDDI validation tests against literature-extracted numerical examples
- FR6: A researcher can verify Palmer validation tests against literature-extracted numerical examples
- FR7: A researcher can access intermediate computational values at each algorithmic stage in a fixture to localize a numerical discrepancy without running the full pipeline. Mechanism: public index functions accept a `diagnostics: bool = False` keyword argument; when `True`, the function returns `(output, diagnostics_dict)` where the dict contains named intermediate values at each computational stage.
- FR8: A researcher can find the tolerance bound for each index in a single reference document, where each tolerance value is grounded in error propagation analysis for that index rather than calibrated to pass tests
- FR9: A researcher can inspect the provenance of each fixture file, including source paper, DOI, equation reference, table reference, extraction method, and the dependency environment versions (NumPy, SciPy, xarray) used to generate the fixture
- FR10: A maintainer can verify that Palmer intermediate computational values (water balance terms, soil moisture stage variables) match staged reference values independently of whether the final output falls within the final tolerance band. Mechanism: Palmer functions support the `diagnostics: bool = False` keyword argument (see FR7); the returned `diagnostics_dict` includes water balance terms and soil moisture stage variables at each computational step.

### Scientific Traceability & Citation

- FR11: A researcher can cite the library in a publication using a persistent DOI
- FR12: A researcher citing a specific library version can access a version-specific DOI that permanently resolves to that exact release, distinct from any concept DOI
- FR13: A researcher can follow dual-citation guidance from the README covering both the library (implementation) and the originating algorithm paper
- FR14: A researcher can read a self-contained algorithm reference document for EDDI that maps each major implementation choice to its source paper citation
- FR15: A researcher can read a self-contained algorithm reference document for the Palmer family that maps each implementation choice to its source paper, with version annotations for when each variant was adopted
- FR16: A researcher can determine the validation status of each index (`validated` / `literature-only` / `pending-CPC-fixtures`) from the README without navigating away
- FR17: A researcher can trace a numerical discrepancy to the equation level using fixture intermediate values, `VALIDATION.md` tolerance documentation, and the algorithm reference doc for that index

### xarray & Pipeline Integration *(P1 — Epic 2)*

- FR18: A user can receive CF-compliant output attributes (`standard_name`, `long_name`, `units`, `valid_min`, `valid_max`) automatically on the result of any public index computation without manual attribute assignment
- FR19: A user can compose EDDI, Palmer, and SPI/SPEI outputs into xarray/Dask pipelines using the standard xarray API
- FR20: A user can consult a compatibility matrix that documents which public functions support Dask-chunked input, what chunking constraints apply, and which Palmer variants require eager evaluation at which pipeline stages
- FR21: A user can upgrade from v2.4 to v2.5 and run existing SPI, SPEI, and EDDI pipelines without behavioral changes; Palmer numerical outputs may differ from v2.4 per the documented moisture anomaly correction in `CHANGELOG.md`
- FR22: A developer can instrument any public index computation to capture named intermediate values at each algorithmic stage without modifying production code paths. Mechanism: the `diagnostics: bool = False` keyword argument (see FR7) is the sole instrumentation surface; no monkey-patching, subclassing, or source modification is required.
- FR37: A user can read computation parameters (scale, distribution type, calibration period start and end year) directly from xarray output attributes without re-inspecting the calling code *(see NFR-OBS-1)*

### Developer Experience

- FR23: A user can install the library from PyPI using pip
- FR24: A user can import all public index functions from a single stable module
- FR25: A user can receive an exception from the `ClimateIndicesError` hierarchy that carries both the offending parameter name and its invalid value as structured attributes, in addition to a human-readable message
- FR26: A developer can read a self-contained usage example (5 lines or fewer) in the docstring of any new or modified public function
- FR27: A user can call any public index function concurrently from multiple threads or processes; the library guarantees no shared mutable module-level state
- FR28: A user can determine the required input shape, dtype, units, and time axis convention for any index function from the function's docstring or a single reference document, without reading source code

### Documentation & Discoverability

- FR29: A new user can execute the getting-started notebook end-to-end without manual intervention to produce xarray SPI output from a sample dataset *(P1 — Epic 2)*
- FR30: A user can execute all three reference notebooks (SPI/SPEI, Palmer, EDDI) end-to-end without manual intervention *(P1 — Epic 2)*
- FR31: A user can find EDDI and Palmer in the README and PyPI landing page with their validation status shown in the README index table (a validation-status column per index, linking to `VALIDATION.md`)
- FR32: A researcher can access `VALIDATION.md` from the README and docs Reference section to review validation status, tolerance criteria, and known discrepancies
- FR33: A researcher can access algorithm reference docs for EDDI and Palmer from the docs Reference section
- FR34: A developer can find a record of all numerical output changes between versions in `CHANGELOG.md`
- FR35: A user upgrading from v2.4 to v2.5 who calls any Palmer index function receives a runtime `DeprecationWarning` informing them that Palmer numerical outputs may differ from v2.4 and directing them to the relevant `CHANGELOG.md` entry. *Note: this warning is excluded from the backward-compatibility assertion scope in `tests/test_backward_compat.py` (see NFR-API-2); it is a notification, not a behavioral change.*

### Project Infrastructure

- FR36: A maintainer can have CI verify that all fixture provenance sidecar files contain the required fields without manual inspection

---

## Non-Functional Requirements

### Performance

- NFR-PERF-1: Palmer and EDDI compute performance must remain within 20% of the v2.4.0 baseline on the reference benchmark dataset. The benchmark fixture (`tests/fixtures/benchmark_baseline_v240.json`) represents a 1200-month synthetic series matching NOAA operational input dimensions for Palmer and EDDI. Measurement: `pytest-benchmark` comparison job against this file, committed from the v2.4.0 tag before any v2.5 compute changes are made. CI fails if the threshold is exceeded.

### Numerical Reproducibility

- NFR-REPR-1: All index computations must produce identical results (within documented `atol`/`rtol` from `tests/fixtures/tolerance.yaml`) on Python 3.10 and Python 3.12 on both Linux and macOS CI runners. Windows is explicitly out of scope for CI in v2.5. Any platform-specific numerical difference outside tolerance bounds on the in-scope platforms is treated as a bug.
- NFR-REPR-2: Tolerance values in `tolerance.yaml` must be derived from source paper precision and error propagation analysis — not tuned to pass tests. Each entry requires a human-reviewed scientific justification field. This is a maintainer gate, not CI.
- NFR-REPR-3: The v2.5 Palmer numerical output change relative to v2.4 (due to moisture anomaly correction) is documented in `CHANGELOG.md` with the affected function names and a before/after numerical example showing the magnitude under representative conditions. A minimal entry covering this change is created as part of Epic 1 (P0); the full `[2.5.0]` section is completed at P1.
- NFR-REPR-4: All index computations except those listed in `CHANGELOG.md` under "Known Output Changes" must produce v2.4.0-identical results on the same input. Verification method: stored reference outputs generated by running v2.4.0 against a canonical input corpus (committed to `tests/fixtures/regression/v2.4.0/` as `.npy` or NetCDF files via `scripts/generate_baselines.py`) are compared against v2.5 outputs using `np.testing.assert_allclose`. Tolerance is configurable per-index in `tests/fixtures/regression/tolerance.yaml`; the default tolerance matches the corresponding entry in `tests/fixtures/tolerance.yaml` unless a specific regression tolerance is justified. The baseline corpus is defined and version-pinned by `scripts/generate_baselines.py`; intentional divergence requires a PR updating both the baseline files and `CHANGELOG.md`. This affirmative longitudinal reproducibility guarantee supersedes the `tests/test_backward_compat.py` module for numerical output verification (the compat module remains for API signature verification per NFR-API-2).

### Code Quality

- NFR-QUAL-1: Branch coverage on new additions to `compute.py`, `xarray_adapter.py`, and new v2.5 modules must be ≥ 90%, enforced via `--cov-fail-under=90` in CI. Coverage paths are pinned explicitly in `[tool.coverage.run] include` in `pyproject.toml`; the list covers `src/climate_indices/compute.py`, `src/climate_indices/xarray_adapter.py`, and any new modules added in v2.5. `indices.py` (legacy, read-only) is excluded.
- NFR-QUAL-2: `ruff check` and `ruff format --check` pass with zero violations across all new and modified files. Enforced in CI.
- NFR-QUAL-3: All new and modified functions exported from `typed_public_api.py` carry complete type hints and Google-style docstrings with `Args`, `Returns`, and `Examples` sections. Internal and private functions require `Args` and `Returns` only. Compliance is validated by maintainer code review; there is no automated CI gate on docstring completeness.
- NFR-QUAL-4: `structlog` is used throughout all new and modified library code; no bare `import logging` or `from logging import` in library code. Enforced in CI via: `! grep -rn "^import logging\|^from logging " src/climate_indices/` — exits non-zero (fails) if any match is found.
- NFR-QUAL-5: Documentation build must succeed without warnings. `uv run sphinx-build -W docs/ docs/_build/html` runs in CI and fails on any Sphinx warning. This gate requires the existing Sphinx build to pass as a maintenance constraint; Epic 3 (Diátaxis restructure, P2) is not a precondition. If the current Sphinx build passes without warnings before any v2.5 documentation changes, this NFR is satisfied by ensuring no new Sphinx warnings are introduced.

### API Stability & Compatibility

- NFR-API-1: `mypy --strict` passes on `typed_public_api.py` in CI. A public signature change that causes `mypy --strict` failure must be accompanied by a deprecation entry in `CHANGELOG.md` and an in-function `DeprecationWarning`. Deprecated signatures must survive for at least two minor versions or one calendar year from the deprecation release, whichever is longer. The no-breaking-changes guarantee applies to all exports present in `typed_public_api.py` at the v2.5.0 release tag; Epic 2 additions are subject to the same guarantee from the point of their addition. `climate_indices` follows semantic versioning: breaking changes to the public API are permitted only in major releases (v3.x+). This policy applies only when signatures change; if no signatures change in v2.5, no `DeprecationWarning` is emitted.
- NFR-API-2: All functions exported from `typed_public_api.py` in v2.4.0 remain callable from v2.5.0 with the same parameter names and types. A dedicated backward-compatibility test module (`tests/test_backward_compat.py`) asserts this; CI fails if any export is removed or its signature changes without a deprecation entry. The `DeprecationWarning` emitted by Palmer functions per FR35 is excluded from this assertion scope — it is a notification, not a behavioral change.
- NFR-API-3: `CITATION.cff` must pass `cff-validator` in CI. An invalid or absent `CITATION.cff` at release time is a P0 blocker.

### Concurrency & Statelessness

- NFR-CONC-1: All public compute functions must be stateless — no module-level mutable state may be introduced in v2.5. Compliance is verified by code review; any shared mutable state (module-level caches, global accumulators) is a merge blocker. This guarantees functions are safe to call concurrently from multiple threads or processes without coordination.

### Observability

- NFR-OBS-1: All public index computation functions that accept xarray inputs must return output `DataArray`s with computation parameters (scale, distribution type, calibration period start and end year) attached as xarray attributes. These observability attributes are distinct from CF-convention attributes governed by `cf_metadata_registry.py`. Compliance is verified by xarray round-trip tests; each test asserts the presence and correctness of computation parameter attributes on the returned `DataArray`. *(P1 — Epic 2)*

### Test Execution Performance

- NFR-TEST-1: The default `uv run pytest` suite (excluding `@pytest.mark.slow` and `@pytest.mark.benchmark`) must complete in under 3 minutes on a standard GitHub Actions Ubuntu runner. Enforced via `pytest --timeout=180` in the default CI job. Individual tests exceeding 30 seconds must be marked `@pytest.mark.slow`.

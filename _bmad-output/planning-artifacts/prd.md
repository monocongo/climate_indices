---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-02b-vision
  - step-02c-executive-summary
  - step-03-success
  - step-04-journeys
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

---

## User Journeys

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

**Capabilities revealed:** `CITATION.cff` + Zenodo DOI; README dual-citation guidance (library + algorithm papers); `VALIDATION.md` with proactive tolerance disclosure; CF-compliant output from `cf_metadata_registry.py`; `docs/algorithm_refs/palmer.md` self-sufficient under peer review; README validation-status table; getting-started notebook executable without intervention.

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

**Capabilities revealed:** `docs/algorithm_refs/eddi.md` with implementation-variant notes; `tests/fixtures/eddi_literature/` fixtures with intermediate computational checkpoints in JSON sidecar; `pytest -m validation` runner; `VALIDATION.md` with equation-level tolerance justification; `ClimateIndicesError` with descriptive messages.

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

**Capabilities revealed:** `CHANGELOG.md` with numerical-output-level change documentation (not just API changes); version annotations in `docs/algorithm_refs/palmer.md`; `VALIDATION.md` as audit trail; semantic versioning with numerical stability guarantees.

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

**Capabilities revealed:** `CHANGELOG.md` with numerical-output-level change documentation; `docs/xarray_compatibility.md` with Dask chunk constraints; CF-compliant output from `cf_metadata_registry.py`; backward-compatible API upgrade; Dask support via `xr.apply_ufunc`.

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

**Capabilities revealed:** Stateless, concurrency-safe public functions; `typed_public_api.py` stable surface; `mypy --strict` compliance; `ClimateIndicesError` hierarchy with parameter-level error messages; xarray coordinate preservation documented or explicitly flagged; getting-started notebook executable without intervention; `CHANGELOG.md` deprecation policy.

---

### Journey Requirements Summary

| Capability | Journeys that reveal it | Priority |
|---|---|---|
| `CITATION.cff` + Zenodo DOI | Maya, Kenji | P0 |
| README dual-citation guidance (library + algorithm papers) | Maya | P0 |
| `VALIDATION.md` with proactive tolerance disclosure and equation-level justification | Maya, Reza | P0 |
| `docs/algorithm_refs/eddi.md` and `palmer.md` — self-sufficient under peer review, version-annotated | Maya, Reza, Kenji | P0 |
| Fixture files with intermediate computational checkpoints in JSON sidecar | Reza | P0 |
| `pytest -m validation` runner | Reza | P0 |
| `CHANGELOG.md` with numerical-output-level change documentation | Kenji, Ben | P1 |
| Version annotations in algorithm reference docs | Kenji | P1 |
| CF-compliant output attributes from `cf_metadata_registry.py` | Maya, Ben, Anika | P1 |
| `docs/xarray_compatibility.md` with Dask chunk constraints documented | Ben | P1 |
| Dask support via `xr.apply_ufunc` with chunk-boundary safety | Ben | P1 |
| Backward-compatible API upgrade from v2.4 | Ben | P1 |
| README validation-status table (library + per-index) | Maya, Ben | P1 |
| Stateless, concurrency-safe public functions | Anika | P1 |
| `typed_public_api.py` stable surface + `mypy --strict` | Anika | P0 |
| `ClimateIndicesError` hierarchy with parameter-level messages | Reza, Anika | P0 |
| Getting-started notebook executable without intervention | Maya, Anika | P1 |
| xarray coordinate preservation (or explicit documentation of drop behavior) | Anika | P1 |

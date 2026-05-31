---
stepsCompleted:
  - step-01-validate-prerequisites
  - step-02-design-epics
  - step-03-create-stories
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/architecture.md
---

# climate_indices - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for climate_indices, decomposing the requirements from the PRD and Architecture into implementable stories.

## Requirements Inventory

### Functional Requirements

**Index Computation**

- FR1: A user can compute SPI, SPEI, EDDI, PDSI, PHDI, PMDI, Z-Index, scPDSI, and PNP from array inputs
- FR2: A user can compute any index using xarray DataArray or Dataset inputs in addition to NumPy arrays *(P1 — Epic 2)*
- FR3: A user can pass Dask-chunked xarray inputs to public index computation functions and receive Dask-backed output *(P1 — Epic 2)*

**Validation Infrastructure**

- FR4: A researcher can run the validation test suite independently from unit and integration tests
- FR5: A researcher can verify EDDI validation tests against literature-extracted numerical examples
- FR6: A researcher can verify Palmer validation tests against literature-extracted numerical examples
- FR7: A researcher can access intermediate computational values at each algorithmic stage in a fixture to localize a numerical discrepancy without running the full pipeline. Mechanism: public index functions accept a `diagnostics: bool = False` keyword argument; when `True`, the function returns `(output, diagnostics_dict)` where the dict contains named intermediate values at each computational stage
- FR8: A researcher can find the tolerance bound for each index in a single reference document, where each tolerance value is grounded in error propagation analysis for that index rather than calibrated to pass tests
- FR9: A researcher can inspect the provenance of each fixture file, including source paper, DOI, equation reference, table reference, extraction method, and the dependency environment versions used to generate the fixture
- FR10: A maintainer can verify that Palmer intermediate computational values (water balance terms, soil moisture stage variables) match staged reference values independently of whether the final output falls within the final tolerance band

**Scientific Traceability & Citation**

- FR11: A researcher can cite the library in a publication using a persistent DOI
- FR12: A researcher citing a specific library version can access a version-specific DOI that permanently resolves to that exact release
- FR13: A researcher can follow dual-citation guidance from the README covering both the library (implementation) and the originating algorithm paper
- FR14: A researcher can read a self-contained algorithm reference document for EDDI that maps each major implementation choice to its source paper citation
- FR15: A researcher can read a self-contained algorithm reference document for the Palmer family that maps each implementation choice to its source paper, with version annotations for when each variant was adopted
- FR16: A researcher can determine the validation status of each index (`validated` / `literature-only` / `pending-CPC-fixtures`) from the README without navigating away
- FR17: A researcher can trace a numerical discrepancy to the equation level using fixture intermediate values, `VALIDATION.md` tolerance documentation, and the algorithm reference doc for that index

**xarray & Pipeline Integration** *(P1 — Epic 2)*

- FR18: A user can receive CF-compliant output attributes (`standard_name`, `long_name`, `units`, `valid_min`, `valid_max`) automatically on the result of any public index computation without manual attribute assignment
- FR19: A user can compose EDDI, Palmer, and SPI/SPEI outputs into xarray/Dask pipelines using the standard xarray API
- FR20: A user can consult a compatibility matrix that documents which public functions support Dask-chunked input, what chunking constraints apply, and which Palmer variants require eager evaluation
- FR21: A user can upgrade from v2.4 to v2.5 and run existing SPI, SPEI, and EDDI pipelines without behavioral changes; Palmer numerical outputs may differ from v2.4 per the documented moisture anomaly correction in `CHANGELOG.md`
- FR22: A developer can instrument any public index computation to capture named intermediate values at each algorithmic stage without modifying production code paths
- FR37: A user can read computation parameters (scale, distribution type, calibration period start and end year) directly from xarray output attributes without re-inspecting the calling code

**Developer Experience**

- FR23: A user can install the library from PyPI using pip
- FR24: A user can import all public index functions from a single stable module
- FR25: A user can receive an exception from the `ClimateIndicesError` hierarchy that carries both the offending parameter name and its invalid value as structured attributes, in addition to a human-readable message
- FR26: A developer can read a self-contained usage example (5 lines or fewer) in the docstring of any new or modified public function
- FR27: A user can call any public index function concurrently from multiple threads or processes; the library guarantees no shared mutable module-level state
- FR28: A user can determine the required input shape, dtype, units, and time axis convention for any index function from the function's docstring or a single reference document

**Documentation & Discoverability**

- FR29: A new user can execute the getting-started notebook end-to-end without manual intervention to produce xarray SPI output from a sample dataset *(P1 — Epic 2)*
- FR30: A user can execute all three reference notebooks (SPI/SPEI, Palmer, EDDI) end-to-end without manual intervention *(P1 — Epic 2)*
- FR31: A user can find EDDI and Palmer in the README and PyPI landing page with their validation status shown in the README index table (a validation-status column per index, linking to `VALIDATION.md`)
- FR32: A researcher can access `VALIDATION.md` from the README and docs Reference section to review validation status, tolerance criteria, and known discrepancies
- FR33: A researcher can access algorithm reference docs for EDDI and Palmer from the docs Reference section
- FR34: A developer can find a record of all numerical output changes between versions in `CHANGELOG.md`
- FR35: A user upgrading from v2.4 to v2.5 who calls any Palmer index function receives a runtime `DeprecationWarning` informing them that Palmer numerical outputs may differ from v2.4 and directing them to the relevant `CHANGELOG.md` entry

**Project Infrastructure**

- FR36: A maintainer can have CI verify that all fixture provenance sidecar files contain the required fields without manual inspection

---

### NonFunctional Requirements

**Performance**

- NFR-PERF-1: Palmer and EDDI compute performance must remain within 20% of the v2.4.0 baseline on the reference benchmark dataset. Measurement: `pytest-benchmark` comparison job against `tests/fixtures/benchmark_baseline_v240.json`, committed from the v2.4.0 tag before any v2.5 compute changes. Benchmark job is advisory (continue-on-error: true), not a hard merge gate.

**Numerical Reproducibility**

- NFR-REPR-1: All index computations must produce identical results (within documented `atol`/`rtol` from `tests/fixtures/tolerance.yaml`) on Python 3.10 and Python 3.12 on both Linux and macOS CI runners
- NFR-REPR-2: Tolerance values in `tolerance.yaml` must be derived from source paper precision and error propagation analysis — not tuned to pass tests. Each entry requires a human-reviewed scientific justification field (maintainer gate, not CI)
- NFR-REPR-3: The v2.5 Palmer numerical output change relative to v2.4 (due to moisture anomaly correction) is documented in `CHANGELOG.md` with affected function names and before/after numerical example. A minimal entry is created as part of Epic 1 (P0); full `[2.5.0]` section completed at P1
- NFR-REPR-4: All index computations except those in `CHANGELOG.md` "Known Output Changes" must produce v2.4.0-identical results on the same input. Stored reference outputs generated by `scripts/generate_baselines.py` from v2.4.0 tag are committed to `tests/fixtures/regression/v2.4.0/` and compared against v2.5 outputs via `np.testing.assert_allclose`

**Code Quality**

- NFR-QUAL-1: Branch coverage on new additions to `compute.py`, `xarray_adapter.py`, and new v2.5 modules must be ≥ 90%, enforced via `--cov-fail-under=90` in CI
- NFR-QUAL-2: `ruff check` and `ruff format --check` pass with zero violations across all new and modified files. Enforced in CI
- NFR-QUAL-3: All new and modified public functions exported from `typed_public_api.py` carry complete type hints and Google-style docstrings with `Args`, `Returns`, and `Examples` sections
- NFR-QUAL-4: `structlog` used throughout all new and modified library code; no bare `import logging` in library code. Enforced in CI via ruff TID251 and a CI grep backstop
- NFR-QUAL-5: Documentation build must succeed without warnings (`uv run sphinx-build -W docs/ docs/_build/html` in CI)

**API Stability & Compatibility**

- NFR-API-1: `mypy --strict` passes on `typed_public_api.py` in CI. Breaking signature changes require a deprecation entry in `CHANGELOG.md` and an in-function `DeprecationWarning`. Deprecated signatures survive for ≥ 2 minor versions or 1 calendar year
- NFR-API-2: All functions exported from `typed_public_api.py` in v2.4.0 remain callable from v2.5.0 with the same parameter names and types. Enforced by `tests/test_backward_compat.py`
- NFR-API-3: `CITATION.cff` must pass `cff-validator` in CI. An invalid or absent `CITATION.cff` at release time is a P0 blocker

**Concurrency & Statelessness**

- NFR-CONC-1: All public compute functions must be stateless — no module-level mutable state may be introduced in v2.5. Verified by code review; any shared mutable state is a merge blocker

**Observability**

- NFR-OBS-1: All public index computation functions accepting xarray inputs must return output `DataArray`s with computation parameters (scale, distribution type, calibration period start and end year) attached as xarray attributes *(P1 — Epic 2)*

**Test Execution Performance**

- NFR-TEST-1: The default `uv run pytest` suite (excluding `@pytest.mark.slow` and `@pytest.mark.benchmark`) must complete in under 3 minutes on a standard GitHub Actions Ubuntu runner. Enforced via `pytest --timeout=180`

**Validation Coverage (from Architecture)**

- NFR-VAL-01: At merge, every index in citability scope (EDDI, PDSI, PHDI, Z-Index, PMDI, scPDSI) has ≥1 `@pytest.mark.validation` test without `@pytest.mark.validation_pending` that passes against a fixture derived from a cited reference source

---

### Additional Requirements

Requirements from the Architecture document that materially affect epic and story design:

**Infrastructure & Tooling (Pre-Work)**

- Scripts required before any epic story begins: `scripts/create_github_issues.py` (idempotent issue generation from sprint-status.yaml); `scripts/generate_baselines.py` (v2.4.0 regression baseline generation — must run against v2.4.0 tag before any compute changes); `scripts/lint_sidecars.py` (fixture JSON schema validation); `scripts/update_citation.py` (patches CITATION.cff version/date at release)
- `.pre-commit-config.yaml` must include: ruff (lint + format), mypy (`--strict` on `typed_public_api.py` only), `scripts/lint_sidecars.py`, `nbstripout` (strip notebook outputs before commit)
- `pyproject.toml` addition: `[tool.ruff.lint.flake8-tidy-imports] banned-module-imports = ["logging"]` (TID251)

**CI Workflow Inventory (from Architecture Decision 1.1)**

- `tests.yml` — push/PR: pytest (excludes `benchmark`, `validation` markers)
- `lint.yml` — push/PR: ruff check, ruff format --check, mypy --strict, CI grep for `import logging`
- `validation.yml` — push/PR, full 4-leg matrix (Linux/macOS × Python 3.10/3.12): `@pytest.mark.validation` tests, sidecar linter, correctness gate
- `benchmarks.yml` — push to `release/*`, manual; `continue-on-error: true` (advisory only)
- `notebooks.yml` — push/PR: `uv run pytest --nbmake notebooks/`

**Fixture Sidecar Schema (Pattern 2)**

- Every file under `tests/fixtures/*_literature/` must have a companion `.json` sidecar with the same stem. Required fields: `source_paper` (str, non-empty), `doi` (str, non-empty), `equation_ref` (str, non-empty), `table_ref` (str|null — null if equation-only), `extraction_method` (enum: `"transcribed"`, `"digitized"`, `"computed"`, `"software_comparison"`), `comparison_target` (str|null — required when extraction_method is `"software_comparison"`), `citability_scope` (bool)
- Fixture directory structure: `tests/fixtures/{index}_literature/{case_id}.npy` + `{case_id}.json`

**tolerance.yaml Schema**

- Flat per-index structure with `_schema_version: 1` key. Required per-index fields: `atol`, `rtol`, `justification_category` (enum: `literature_stated`, `numerical_precision`, `digitization_uncertainty`, `algorithm_discretization`), `justification` (free text), `source_doi`
- Loaded via shared `conftest.py` session-scoped `tolerances` fixture; never loaded inline in test files

**Correctness Gate (Pattern 4)**

- `eddi_gate` and `palmer_gate` pytest fixtures in `tests/conftest.py` block Stories 1.3–1.5 collection until `docs/algorithm_refs/eddi.md` and `docs/algorithm_refs/palmer.md` exist and contain a table row matching `r"^\|\s*(Source|Reference|Published)"` (presence/structure check, not correctness oracle — documented in conftest comment)
- **Palmer verification requirement:** Story 1.2 (Palmer literature extraction) requires a second independent verification pass at Claude Opus reasoning level before the correctness gate is cleared. Any fixture written against unconfirmed Palmer numerics must be deleted, not marked provisional

**Validation Marker Taxonomy (Pattern 1)**

- `@pytest.mark.validation` — expected to pass at merge; unexpected skip is a regression signal; counts against NFR-VAL-01
- `@pytest.mark.validation_pending` — fixture not yet received; skip is expected; does not count against NFR-VAL-01
- Marker stack order in tests: `@pytest.mark.validation` first, `@pytest.mark.validation_pending` second (when present), `@pytest.mark.parametrize` third

**Notebook Standard (RISK-F-02 Resolution)**

- Use `nbmake` (not nbval or bare nbconvert); outputs stripped before commit via `nbstripout` pre-commit hook
- Each notebook includes an assertion cell (penultimate cell) that loads tolerances from `tolerance.yaml` and validates ≥1 computed index value via `np.testing.assert_allclose`
- Cell ordering: (1) title+description markdown, (2) imports, (3–N-2) tutorial content, (N-1) assertion cell, (N) summary markdown

**CF Attribute Enforcement (Decision 3.2 — Soft)**

- `xarray_adapter.py` looks up CF attrs from `cf_metadata_registry.py` on every return. If registry entry exists, attributes attached automatically. If missing, `structlog` `WARNING` emitted with event `"cf_metadata_missing"` and `index` field bound — no exception raised

**Palmer `algorithm=` Parameter**

- Palmer public API functions accept `algorithm: str = "original_1965"`. Only one implementation registered in v2.5. Unrecognized values raise `InvalidArgumentError` immediately. Intentional YAGNI exception to avoid v3.x breaking change

**Story 2.1 → 2.2 Handoff (Decision 3.1)**

- Story 2.1 (xarray audit) produces `docs/xarray_gaps.md` — a table with columns: index name, gap category, severity, suggested fix. Story 2.2 AC must explicitly reference this file as its input specification. File is deleted or archived after Story 2.2 merges

**Zenodo / DOI (Decision 2.1/2.2)**

- Zenodo integration: GitHub Release → Zenodo webhook (configure once at repo level; no custom CI step for trigger)
- `scripts/update_citation.py` patches `version:` and `date-released:` fields; called in release workflow before publish. Story 3.5 AC must include a passing run of this script

**Dask Chunking Constraint**

- Time dimension must remain a single Dask chunk for all climate index computations. `xarray_adapter.py` validates and raises `DimensionMismatchError` with remediation message: `rechunk to {'time': -1}`. All new xarray wrappers for Palmer or EDDI must inherit this validation

**Structlog Canonical Field Names (Pattern 5)**

- Bind fields: `index` (str), `timescale` (int|None), `periodicity` (str), `input_shape` (tuple), `data_var` (str|None), `algorithm` (str — Palmer only), `calibration_year_initial` (int), `calibration_year_final` (int)

**Documentation Architecture (Decision 4.x)**

- Diátaxis structure: notebooks/ (tutorials), docs/how-to/ (how-to guides), sphinx-autodoc (reference), docs/algorithm_refs/ + VALIDATION.md (explanation)
- NClimGrid gallery PNGs: pre-generated, committed to `docs/_static/gallery/`; `scripts/generate_gallery.py` used for regeneration; not run in CI (external NOAA SFTP dependency)
- `llms.txt` hand-maintained at repo root; `llms-full.txt` generated by `scripts/generate_llms_full.py`

**Provenance Chain (RISK-F-10)**

- Explicit cross-reference chain: sidecar JSON → `tolerance.yaml` → `VALIDATION.md` → `CITATION.cff`
- `VALIDATION.md` must contain, for each index in citability scope, a table row with: index name, validation status, `source_doi` from sidecar, and `atol`/`rtol` from `tolerance.yaml`

---

### UX Design Requirements

Not applicable — `climate_indices` is a scientific Python library with no UI component. There is no UX Design document.

---

### FR Coverage Map

| FR | Epic | Status | Note |
|---|---|---|---|
| FR1 | Epic 1 | Existing — verify only | Numpy compute path already works; confirmed by validation tests |
| FR2 | Epic 2 | New | xarray DataArray/Dataset input support |
| FR3 | Epic 2 | New | Dask-chunked xarray input |
| FR4 | Epic 1 | New | Validation test suite CI job (setup stories in Epic 1) |
| FR5 | Epic 1 | New | EDDI `@pytest.mark.validation` tests |
| FR6 | Epic 1 | New | Palmer `@pytest.mark.validation` tests |
| FR7 | *P2 / v2.5.x* | Deferred | Not a citability gate; new API surface risk outweighs P0 benefit; fixture sidecars already capture intermediates |
| FR8 | Epic 1 | New | `tests/fixtures/tolerance.yaml` |
| FR9 | Epic 1 | New | Fixture JSON sidecars with provenance fields |
| FR10 | *P2 / v2.5.x* | Deferred | Depends on FR7; deferred with it |
| FR11 | Epic 1 | New | `CITATION.cff` + Zenodo DOI wired up |
| FR12 | Epic 1 | New | Version-specific DOI via GitHub Release → Zenodo webhook |
| FR13 | Epic 1 | New | README dual-citation guidance (P0 minimal update) |
| FR14 | Epic 1 | New | `docs/algorithm_refs/eddi.md` |
| FR15 | Epic 1 | New | `docs/algorithm_refs/palmer.md` with version annotations |
| FR16 | Epic 1 | New | Validation-status column added to existing README index table (P0 minimal) |
| FR17 | Epic 1 | New (partial) | Provenance chain enabling equation-level tracing via static fixture comparison; **Note: only partially satisfied without FR7 — live `diagnostics=True` mode deferred; coverage map should be read as "static fixture tracing only" for v2.5** |
| FR18 | Epic 2 | New | CF-compliant attrs automatically attached via `cf_metadata_registry.py` |
| FR19 | Epic 2 | New | xarray/Dask pipeline composability |
| FR20 | Epic 2 | New | `docs/xarray_compatibility.md` — canonical reference for FR28; includes Dask chunk constraints, thread-safety/statelessness guarantee section |
| FR21 | Epic 2 | New | Backward-compat test module (`tests/test_backward_compat.py`) + CHANGELOG confirmation |
| FR22 | Epic 2 | New | `diagnostics` kwarg scope narrowed: xarray-path wrapper instrumentation only (not full live intermediate chain of FR7) |
| FR23 | Epic 3 | Existing — extend | PyPI release with updated README |
| FR24 | Epic 1 | Existing — verify only | Single-module import via `typed_public_api.py`; confirmed by `mypy --strict` + NFR-API-2 |
| FR25 | Epic 1 | Existing — verify only | `ClimateIndicesError` hierarchy since v2.3; confirmed in validation story error-path tests |
| FR26 | Epic 1 + 2 | Cross-cutting | Docstring `Examples:` sections added per story as part of each AC |
| FR27 | Epic 2 | Cross-cutting → Documented | Statelessness verified per story via code review AND documented explicitly in `docs/xarray_compatibility.md` thread-safety section (Epic 2) |
| FR28 | Epic 2 | New | Canonical reference = `docs/xarray_compatibility.md`; covers input shape, dtype, units, time axis convention for all xarray-path functions |
| FR29 | Epic 2 | New | Getting-started notebook (SPI/SPEI via xarray) |
| FR30 | Epic 2 | New | All three reference notebooks (SPI/SPEI, Palmer, EDDI) |
| FR31 | Epic 3 | New | Full README/PyPI revamp surfacing EDDI + Palmer prominently |
| FR32 | Epic 1 | New | `VALIDATION.md` created + linked from README — **ships in all contingency paths, including Palmer slip; Palmer entry shows `literature-only / validation-pending` if Story 1.2 is deferred** |
| FR33 | Epic 3 | New | Docs Reference section restructured to link algorithm refs and VALIDATION.md |
| FR34 | Epic 1 + 2 | Split | Minimal Palmer entry with before/after numerical example (Epic 1, P0 — required by NFR-REPR-3); full `[2.5.0]` section (Epic 2, P1) |
| FR35 | Epic 1 | New | `DeprecationWarning` on Palmer functions directing users to CHANGELOG |
| FR36 | Epic 1 | New | `scripts/lint_sidecars.py` in pre-commit hook + `validation.yml` CI job |
| FR37 | Epic 2 | New | Computation parameters (scale, distribution type, calibration period) as xarray output attributes |

---

**Deferred to v2.5.x (not in v2.5 epics):**

| FR | Reason |
|---|---|
| FR7 | New public API surface (`diagnostics` kwarg on all public functions); not a citability gate; fixture sidecars cover the static tracing use case |
| FR10 | Depends on FR7 |

---

## Epic List

### Epic 1 — Index Validation & Scientific Citability *(P0)*

Researchers can cite `climate_indices` v2.5 in a journal submission. Algorithm reference docs trace every formula choice to its source paper. Literature fixtures with provenance sidecars are machine-readable and committed. `VALIDATION.md` discloses tolerance bounds and known gaps for every index in citability scope. `CITATION.cff` provides a stable Zenodo DOI. Release infrastructure (CI workflows, sidecar linter, baseline generation, pre-commit config) is established in the opening stories of this epic.

**FRs covered:** FR1 *(verify only)*, FR4, FR5, FR6, FR8, FR9, FR11, FR12, FR13, FR14, FR15, FR16, FR17 *(partial — static fixture tracing only)*, FR24 *(verify only)*, FR25 *(verify only)*, FR26 *(cross-cutting)*, FR32, FR34 *(minimal entry with numerical example)*, FR35, FR36

**Infrastructure setup stories (beginning of epic, before citability work):**
- `validation.yml` CI job (4-leg matrix: Linux/macOS × Python 3.10/3.12), `benchmarks.yml` (advisory), `notebooks.yml`
- `scripts/lint_sidecars.py` + `.pre-commit-config.yaml` (ruff, mypy, lint_sidecars, nbstripout)
- `pyproject.toml`: ruff TID251 `banned-module-imports = ["logging"]`, coverage `include` list
- `scripts/generate_baselines.py` run against v2.4.0 tag → `tests/fixtures/benchmark_baseline_v240.json`
- `scripts/update_citation.py` skeleton
- *(Note: `scripts/create_github_issues.py` already committed — excluded)*

**⚠️ Palmer contingency:** Story 1.2 (Palmer literature extraction + Opus verification) is the highest-variance story in the sprint. If the Opus verification reveals a fundamental implementation error, Palmer fixture stories slip to v2.5.x. EDDI validation proceeds independently. **`VALIDATION.md` ships regardless of Palmer status** — with an explicit `literature-only / validation-pending` Palmer entry and honest gap disclosure. The P0 citability claim can be made on EDDI + CITATION.cff + VALIDATION.md + algorithm reference docs alone.

**Story 1.1** creates `docs/algorithm_refs/eddi.md` → unblocks `eddi_gate`
**Story 1.2** creates `docs/algorithm_refs/palmer.md` → unblocks `palmer_gate` *(Opus verification required)*
**Stories 1.3+** validation tests, fixtures, tolerance.yaml, VALIDATION.md, CITATION.cff, CHANGELOG minimal entry — gated on 1.1/1.2

---

### Epic 2 — xarray & Dask Pipeline Integration *(P1)*

Applied scientists and data engineers can feed Dask-chunked xarray inputs to all public index functions and receive CF-compliant output with computation parameters as attributes. Thread-safety and statelessness are explicitly documented — not just code-review NFRs. Three executable Jupyter notebooks demonstrate SPI/SPEI, Palmer, and EDDI via the xarray API. The complete `CHANGELOG.md [2.5.0]` section documents all v2.5 output changes. The coordinate-drop behavior for non-standard coordinate names is audited and documented (or fixed).

**FRs covered:** FR2, FR3, FR18, FR19, FR20 *(canonical FR27/FR28 reference)*, FR21, FR22, FR27 *(documented in xarray_compatibility.md)*, FR28, FR29, FR30, FR34 *(full [2.5.0] section)*, FR37

**Deliverables:**
- xarray compatibility audit → `docs/xarray_gaps.md` (Story 2.1); includes explicit audit item for coordinate-drop behavior with non-standard coordinate names
- CF attribute fixes in `cf_metadata_registry.py` + `xarray_adapter.py` for EDDI and Palmer (Story 2.2, gated on `docs/xarray_gaps.md`)
- Dask support via `xr.apply_ufunc` with chunk-boundary safety
- `docs/xarray_compatibility.md`: Dask chunk constraints, thread-safety/statelessness section (FR27), input shape/dtype/units/time axis conventions (FR28)
- Three Jupyter notebooks with assertion cells drawing tolerances from `tolerance.yaml`
- Backward-compat test module `tests/test_backward_compat.py`
- Full `CHANGELOG.md [2.5.0]` section

**Story 2.1 → 2.2 handoff:** `docs/xarray_gaps.md` is the structured deliverable; Story 2.2 AC must reference every row in the gap table. File deleted or archived after Story 2.2 merges.

---

### Epic 3 — Documentation Refresh & Discoverability *(P2)*

New users land on the README or PyPI page and immediately find EDDI and Palmer with their validation status. The docs site is Diátaxis-structured. An NClimGrid example gallery demonstrates real CONUS drought index maps. `llms-full.txt` supports AI tooling integrations. Zenodo DOI is populated in `CITATION.cff` and README post-release.

**FRs covered:** FR23 *(PyPI release with updated README)*, FR31, FR33

**Implementation stories (explicit, no FR anchor):**
- Diátaxis restructure — `docs/how-to/` created; tutorials = `notebooks/`; reference = sphinx-autodoc; explanation = `docs/algorithm_refs/` + `VALIDATION.md`
- NClimGrid gallery PNGs — `docs/_static/gallery/` (SPI-3, SPI-6, SPEI-3, PDSI, PHDI, Z-Index, EDDI-3); `scripts/generate_gallery.py`; not run in CI (external NOAA SFTP dependency)
- `llms.txt` (hand-maintained at repo root) + `llms-full.txt` via `scripts/generate_llms_full.py`; `llms-full.txt` regenerated in release workflow
- Release automation — `scripts/update_citation.py` finalized; Zenodo DOI populated in `CITATION.cff` and README; git tag `v2.5.0` applied

---

## Epic 1: Index Validation & Scientific Citability

Researchers can cite `climate_indices` v2.5 in a journal submission with full algorithmic provenance — algorithm reference docs trace every formula choice to its source paper, literature fixtures with provenance sidecars are machine-readable, `VALIDATION.md` discloses tolerance bounds and known gaps, and `CITATION.cff` provides a stable Zenodo DOI.

### Story 1.1: Release Infrastructure Setup

As a maintainer,
I want all CI workflows, pre-commit hooks, and quality-gate scripts in place,
So that every subsequent story starts with automated enforcement of linting, type-checking, sidecar schema validation, and benchmark regression protection from the first commit.

**Acceptance Criteria:**

**Given** the `release/v2.5` branch exists with no CI workflows for validation, benchmarks, or notebooks
**When** Story 1.1 is merged
**Then** `.github/workflows/validation.yml` exists and runs `@pytest.mark.validation` tests on a 4-leg matrix (Linux/macOS × Python 3.10/3.12); job fails if any single leg reports a tolerance breach
**And** `.github/workflows/benchmarks.yml` exists with `continue-on-error: true` and asserts `tests/fixtures/benchmark_baseline_v240.json` is present before running; failures are advisory only
**And** `.github/workflows/notebooks.yml` exists and runs `uv run pytest --nbmake notebooks/`

**Given** a Python file in `src/climate_indices/` containing `import logging`
**When** `uv run ruff check` is run
**Then** ruff reports a TID251 violation (`banned-module-imports = ["logging"]` configured in `pyproject.toml`)

**Given** the pre-commit hooks are installed via `.pre-commit-config.yaml`
**When** a developer commits a file with a ruff violation, a `mypy --strict` failure on `typed_public_api.py`, an invalid fixture sidecar JSON, or a notebook with committed outputs
**Then** the commit is blocked with a descriptive error message identifying the specific violation

**Given** the v2.4.0 git tag exists
**When** `scripts/generate_baselines.py` is run against the v2.4.0 tag
**Then** `tests/fixtures/benchmark_baseline_v240.json` is generated and committed to `release/v2.5`; the file is present before any v2.5 compute changes are made

**Given** a new fixture JSON sidecar file is added to `tests/fixtures/`
**When** `uv run python scripts/lint_sidecars.py` is run
**Then** the script validates all required fields (`source_paper`, `doi`, `equation_ref`, `table_ref`, `extraction_method`, `comparison_target`, `citability_scope`) and exits non-zero with a descriptive error message if any are missing or invalid
**And** `scripts/update_citation.py` exists as a skeleton that reads `version` from `pyproject.toml` and patches `CITATION.cff` `version:` and `date-released:` fields without manual arguments

---

### Story 1.2: EDDI Algorithm Reference Document

As a researcher,
I want a self-contained algorithm reference document for EDDI that maps each major implementation choice to its source paper,
So that I can cite the specific algorithm variant used and audit any implementation decision without maintainer intervention.

**Acceptance Criteria:**

**Given** `docs/algorithm_refs/eddi.md` does not exist
**When** Story 1.2 is merged
**Then** `docs/algorithm_refs/eddi.md` exists and is committed to `release/v2.5`

**Given** `docs/algorithm_refs/eddi.md` exists
**When** a researcher reads the document
**Then** the document contains: (1) the primary authoritative source citation with DOI (Hobbins et al. 2016); (2) the canonical EDDI formula in LaTeX or equivalent notation; (3) all parameters with units and valid ranges; (4) ≥1 numeric reference case with input → expected output traceable to the cited source; (5) a "Validation Provenance" section referencing associated fixture sidecar files

**Given** `docs/algorithm_refs/eddi.md` is merged
**When** `uv run pytest tests/ -k eddi_gate` is run
**Then** the `eddi_gate` pytest fixture passes — the document contains a table row matching `r"^\|\s*(Source|Reference|Published)"` (case-insensitive) — and Stories 1.4+ become collectable
**And** `docs/algorithm_refs/eddi.md` links to `VALIDATION.md` (placeholder link acceptable until Story 1.6)

---

### Story 1.3: Palmer Algorithm Reference Document

As a researcher,
I want a self-contained algorithm reference document for the Palmer family that maps each implementation choice to its source paper with version annotations,
So that I can trace any Palmer numerical result back to a specific algorithmic decision and cite the implementation variant used.

**⚠️ Requires Claude Opus verification pass — Palmer Zf/PDSI numerical chain has known error accumulation risk. Story is not complete until Opus confirms the implementation matches the documented formulas or flags any discrepancy for resolution.**

**Acceptance Criteria:**

**Given** `docs/algorithm_refs/palmer.md` does not exist
**When** Story 1.3 is merged
**Then** `docs/algorithm_refs/palmer.md` exists and is committed to `release/v2.5`

**Given** `docs/algorithm_refs/palmer.md` exists
**When** a researcher reads the document
**Then** the document contains: (1) primary authoritative source citation (Palmer 1965 + any variant papers); (2) canonical formula chain for PDSI, PHDI, PMDI, Z-Index, and scPDSI in LaTeX or equivalent notation; (3) explicit documentation of the moisture anomaly weighting variant implemented vs Alley 1984; (4) version annotations for when each algorithmic variant was adopted; (5) `algorithm=` dispatch parameter documented with default value `"original_1965"` and note that unrecognised values raise `InvalidArgumentError`; (6) all parameters with units and valid ranges; (7) ≥1 numeric reference case traceable to cited source; (8) a "Validation Provenance" section

**Given** `docs/algorithm_refs/palmer.md` is merged
**When** `uv run pytest tests/ -k palmer_gate` is run
**Then** the `palmer_gate` pytest fixture passes and Stories 1.5+ become collectable

**Given** Opus verification has been run on Story 1.3
**When** the verification is complete
**Then** every Palmer formula in `docs/algorithm_refs/palmer.md` has been confirmed against the library implementation; any discrepancy is either documented as a deliberate implementation choice with rationale, or flagged as a bug and resolved before Story 1.3 is merged; no fixtures from Story 1.5 are committed until this gate is cleared

---

### Story 1.4: EDDI Validation Test Suite and Fixtures

As a researcher,
I want parametrized EDDI validation tests against literature-extracted fixtures with provenance sidecars,
So that I can independently verify the EDDI implementation against published numerical results and trace any discrepancy to the equation level using intermediate checkpoints.

**Acceptance Criteria:**

**Given** `docs/algorithm_refs/eddi.md` is merged (eddi_gate passes)
**When** Story 1.4 is merged
**Then** `tests/fixtures/eddi_literature/` contains ≥1 fixture file (`.npy` or `.csv`) with a companion JSON sidecar file of the same stem

**Given** a fixture sidecar in `tests/fixtures/eddi_literature/`
**When** `uv run python scripts/lint_sidecars.py` is run
**Then** all 7 required fields are present and valid: `source_paper` (non-empty str), `doi` (non-empty str), `equation_ref` (non-empty str), `table_ref` (str or null), `extraction_method` (one of `"transcribed"`, `"digitized"`, `"computed"`, `"software_comparison"`), `comparison_target` (non-empty str if `extraction_method == "software_comparison"`, null otherwise), `citability_scope` (bool); sidecar linter exits 0

**Given** `tests/fixtures/tolerance.yaml` is created or updated
**When** the EDDI tolerance entry is inspected
**Then** the entry contains: `atol`, `rtol`, `justification_category` (one of `literature_stated`, `numerical_precision`, `digitization_uncertainty`, `algorithm_discretization`), `justification` (non-empty free text), `source_doi`; `_schema_version: 1` key is present at top of file

**Given** validation tests exist for EDDI
**When** `uv run pytest -m validation` is run on any of the 4 CI matrix legs (Linux/macOS × Python 3.10/3.12)
**Then** all `@pytest.mark.validation` EDDI tests pass; `@pytest.mark.validation_pending` tests are skipped and excluded from the skip count; NFR-VAL-01 is satisfied (≥1 non-pending validation test for EDDI passes)

**Given** a validation test file for EDDI
**When** the test anatomy is inspected
**Then** markers are stacked in order: `@pytest.mark.validation` first, `@pytest.mark.validation_pending` second (when present), `@pytest.mark.parametrize` third; `eddi_gate` is declared as a positional parameter but not used in the test body; `tolerances` fixture loads from `conftest.py` shared session-scoped fixture; test function is named `test_eddi_against_literature`; docstring names the source paper

---

### Story 1.5: Palmer Validation Test Suite and Fixtures

As a researcher,
I want parametrized Palmer validation tests against literature-extracted fixtures with provenance sidecars for PDSI, PHDI, Z-Index, PMDI, and scPDSI,
So that I can independently verify each Palmer index against published numerical results with documented tolerance bounds.

**⚠️ Requires Claude Opus — highest-variance story in the sprint. Palmer contingency applies: if Story 1.3 Opus verification revealed a fundamental implementation error that is unresolved, this story's scope is reduced to `@pytest.mark.validation` + `@pytest.mark.validation_pending` stubs with linked GitHub issues; no partial fixtures are committed.**

**Acceptance Criteria:**

**Given** `docs/algorithm_refs/palmer.md` is merged and Opus verification is complete (palmer_gate passes)
**When** Story 1.5 is merged
**Then** `tests/fixtures/palmer_literature/` contains ≥1 fixture file for at least PDSI with a companion JSON sidecar; all 7 sidecar fields present and valid per `scripts/lint_sidecars.py`

**Given** `tests/fixtures/tolerance.yaml` is updated for Palmer
**When** Palmer tolerance entries are inspected
**Then** ≥1 Palmer index (PDSI minimum) has an entry with `atol`, `rtol`, `justification_category`, `justification`, `source_doi`; tolerance values are derived from source paper precision, not tuned to pass tests

**Given** validation tests exist for Palmer
**When** `uv run pytest -m validation` is run on all 4 CI matrix legs
**Then** ≥1 Palmer `@pytest.mark.validation` test without `@pytest.mark.validation_pending` passes; NFR-VAL-01 satisfied for PDSI minimum; tests follow Pattern 1 anatomy with `palmer_gate` as unused positional parameter

**Given** the Palmer contingency is triggered
**When** Story 1.5 is merged under contingency scope
**Then** Palmer validation tests exist as stubs with both `@pytest.mark.validation` and `@pytest.mark.validation_pending`; each stub has a linked GitHub issue explaining the blocking condition; no fixture files are committed; `tests/fixtures/tolerance.yaml` has no Palmer entries; Story 1.6 `VALIDATION.md` documents Palmer status as `pending` with the linked GitHub issue

---

### Story 1.6: VALIDATION.md, CITATION.cff, and Release Metadata

As a researcher,
I want `VALIDATION.md` and `CITATION.cff` at the repository root along with minimal README and `CHANGELOG.md` updates,
So that I can assess the validation status of each index before citing, cite the library with a persistent DOI, and understand any numerical changes from v2.4.

**Ships regardless of Palmer contingency outcome.**

**Acceptance Criteria:**

**Given** Stories 1.2–1.5 are complete (or reduced to contingency scope)
**When** Story 1.6 is merged
**Then** `VALIDATION.md` exists at the repo root and contains a table with one row per index in citability scope (EDDI, PDSI, PHDI, Z-Index, PMDI, scPDSI); each row includes: index name, validation status (`validated` / `literature-only` / `pending-CPC-fixtures` / `pending`), `source_doi` from the fixture sidecar, `atol` and `rtol` from `tolerance.yaml`

**Given** Palmer contingency was triggered
**When** `VALIDATION.md` is inspected
**Then** Palmer indices show status `pending` with a note referencing the linked GitHub issue; the document is honest about the gap and does not omit Palmer from the table

**Given** `VALIDATION.md` is merged
**When** a researcher navigates from the README
**Then** `VALIDATION.md` is linked from the README validation-status section and from the docs Reference section (placeholder link acceptable until Epic 3 Diátaxis restructure)

**Given** `CITATION.cff` does not exist at the repo root
**When** Story 1.6 is merged
**Then** `CITATION.cff` is present at the repo root; `cff-validator` passes in CI (NFR-API-3); `version:` and `date-released:` fields are present and populated by `scripts/update_citation.py`

**Given** a researcher reads the README after Story 1.6
**When** they look for citation guidance
**Then** the README contains a dual-citation guidance section instructing researchers to cite both the library (via `CITATION.cff` / Zenodo DOI) and the originating algorithm paper; a validation-status column is present in the existing README index table linking each index to `VALIDATION.md`

**Given** `CHANGELOG.md` is inspected for Palmer changes
**When** a developer reads the minimal v2.5 entry
**Then** the entry documents the Palmer moisture anomaly correction with: affected function names, a before/after numerical example showing the magnitude under representative conditions (satisfies NFR-REPR-3); sufficient for a reproducer to identify the version-induced discrepancy

**Given** any Palmer public function is called from v2.5
**When** the function executes
**Then** a `DeprecationWarning` is emitted informing the caller that Palmer numerical outputs may differ from v2.4 and directing them to the relevant `CHANGELOG.md` entry; the warning is excluded from `tests/test_backward_compat.py` assertion scope
**And** `uv run ruff check` passes with zero violations on all new and modified files
**And** `mypy --strict` passes on `typed_public_api.py`

---

## Epic 2: xarray & Dask Pipeline Integration

Applied scientists and data engineers can feed Dask-chunked xarray inputs to all public index functions and receive CF-compliant output with computation parameters as attributes. Thread-safety and statelessness are explicitly documented — not just code-review NFRs. Three executable Jupyter notebooks demonstrate SPI/SPEI, Palmer, and EDDI via the xarray API. The complete `CHANGELOG.md [2.5.0]` section documents all v2.5 output changes. The coordinate-drop behavior for non-standard coordinate names is audited and documented (or fixed).

### Story 2.1: xarray Compatibility Audit

As an applied scientist,
I want a structured compatibility audit of all public index functions for xarray and Dask support gaps,
So that I have a prioritized, actionable remediation plan before any code changes begin.

**Acceptance Criteria:**

**Given** all public functions exported from `typed_public_api.py`
**When** Story 2.1 is merged
**Then** `docs/xarray_gaps.md` exists and contains a table with columns: index name, gap category (one of `"no_xarray_support"`, `"cf_attrs_missing"`, `"dask_unsafe"`, `"coordinate_drop"`, `"parameters_not_attached"`), severity (`"blocker"` / `"major"` / `"minor"`), suggested fix; one row per discovered gap

**Given** the audit table in `docs/xarray_gaps.md`
**When** a developer reads it
**Then** it includes an explicit row for the coordinate-drop behavior with non-standard coordinate names — either documenting the expected behavior or flagging it as a gap requiring a fix; every row is specific enough to generate a targeted acceptance criterion for Story 2.2

**Given** Story 2.2 has not yet started
**When** `docs/xarray_gaps.md` is reviewed
**Then** it serves as the complete input specification for Story 2.2; Story 2.2 AC references every row in the gap table; the document is deleted or archived when Story 2.2 merges

---

### Story 2.2: xarray and CF Attribute Integration for EDDI and Palmer

As an applied scientist,
I want EDDI and Palmer public functions to accept xarray DataArray and Dataset inputs and return CF-compliant output with computation parameters as attributes,
So that I can compose these indices in standard xarray/Dask pipelines alongside SPI and SPEI.

**Gated on `docs/xarray_gaps.md` from Story 2.1.**

**Acceptance Criteria:**

**Given** `docs/xarray_gaps.md` is the input specification for this story
**When** Story 2.2 is merged
**Then** every row in `docs/xarray_gaps.md` is addressed by a corresponding code change, or documented as an intentional non-fix with written rationale; the gap document is deleted or archived after merge

**Given** a public EDDI or Palmer function is called with a well-formed `xarray.DataArray` input
**When** the function returns
**Then** the result is an `xarray.DataArray`; CF attributes (`standard_name`, `long_name`, `units`, `valid_min`, `valid_max`) are automatically attached via `cf_metadata_registry.py` without manual assignment by the caller; if a registry entry is missing, a `structlog` WARNING is emitted with event `"cf_metadata_missing"` and `index` field bound — no exception raised

**Given** any public xarray-path function is called
**When** the output DataArray is inspected for computation parameters
**Then** the output carries attributes for computation parameters: `scale` (if applicable), `distribution` (if applicable), `calibration_year_initial`, `calibration_year_final`; these attributes are present without manual attribute assignment by the caller (satisfies FR37)

**Given** a developer passes `diagnostics=True` to a public xarray-path wrapper
**When** the function returns
**Then** the function returns `(output_DataArray, diagnostics_dict)` where `diagnostics_dict` contains named intermediate values at each xarray-wrapper stage; this kwarg exists only on xarray-path wrappers, not on the underlying numpy compute functions (FR22 narrowed scope per FR Coverage Map)

**Given** a public EDDI or Palmer function is called with a DataArray that uses non-standard coordinate names (e.g., `"lat"` instead of `"latitude"`)
**When** the function returns
**Then** the output DataArray retains the original coordinate names without dropping them; or if dropping is determined to be an intentional limitation, the behavior is documented in `docs/xarray_compatibility.md` with a workaround

**Given** `uv run ruff check` and `mypy --strict typed_public_api.py` are run after Story 2.2
**When** the checks complete
**Then** zero violations are reported; all new and modified files pass both checks

---

### Story 2.3: Dask Pipeline Support and xarray Compatibility Reference

As a data engineer,
I want Dask-chunked xarray inputs to work correctly with all public index functions, and a canonical reference document describing chunking constraints and thread-safety guarantees,
So that I can safely scale index computation over large climate datasets without silent correctness failures.

**Acceptance Criteria:**

**Given** a public index function is called with a Dask-chunked `xarray.DataArray` where the time dimension is a single chunk
**When** the function returns
**Then** the output is a Dask-backed `xarray.DataArray`; computation is deferred until `.compute()` is called; results are numerically identical to the non-chunked equivalent within `tolerance.yaml` bounds; CF attributes and computation parameters are attached as in Story 2.2

**Given** a public index function is called with a Dask-chunked `xarray.DataArray` where the time dimension spans multiple chunks
**When** the function raises
**Then** a `DimensionMismatchError` is raised with a message that includes the remediation instruction `rechunk to {'time': -1}`; no partial computation is performed; the error carries structured attributes as required by the `ClimateIndicesError` hierarchy

**Given** `docs/xarray_compatibility.md` does not yet exist
**When** Story 2.3 is merged
**Then** `docs/xarray_compatibility.md` exists and contains: (1) a compatibility matrix table listing each public function, whether it supports Dask-chunked input, and any chunking constraints; (2) a thread-safety and statelessness section asserting no module-level mutable state in v2.5 (satisfies FR27 documentation requirement); (3) input shape, dtype, units, and time-axis convention for all xarray-path functions (satisfies FR28)

**Given** any new xarray wrapper for Palmer or EDDI is introduced in this or earlier stories
**When** the implementation is reviewed
**Then** it inherits the time-dimension single-chunk validation from `xarray_adapter.py`; no new xarray wrapper implements its own independent chunk-validation logic

**Given** `uv run ruff check`, `mypy --strict`, and `uv run pytest` are run after Story 2.3
**When** all checks complete
**Then** zero violations; all existing validation tests pass; `docs/xarray_compatibility.md` is referenced from the docs Reference section (placeholder link acceptable until Epic 3 Diátaxis restructure)

---

### Story 2.4: Backward Compatibility Test Suite and CHANGELOG [2.5.0]

As a developer upgrading from v2.4 to v2.5,
I want the backward compatibility of all existing public function signatures verified by an automated test module and all numerical output changes documented in `CHANGELOG.md`,
So that I can safely upgrade and understand any behavioral differences before deploying v2.5 in production.

**Acceptance Criteria:**

**Given** the v2.4.0 public API exported from `typed_public_api.py`
**When** Story 2.4 is merged
**Then** `tests/test_backward_compat.py` exists and verifies that all functions exported in v2.4.0 remain callable from v2.5.0 with identical parameter names and types (satisfies NFR-API-2); the test module is collected in the default `uv run pytest` run and passes

**Given** `tests/test_backward_compat.py` is executed
**When** a v2.4-compatible invocation pattern is tested
**Then** each function call succeeds without `TypeError` or signature mismatch; `DeprecationWarning` emissions from Palmer functions are explicitly asserted rather than suppressed, confirming the warning machinery works; numerical comparisons use `np.testing.assert_allclose` with `atol`/`rtol` from `tests/fixtures/tolerance.yaml`

**Given** `tests/fixtures/regression/v2.4.0/` exists (generated by `scripts/generate_baselines.py` in Story 1.1)
**When** `uv run pytest tests/test_backward_compat.py` is run
**Then** all non-Palmer indices produce v2.4.0-identical results within `tolerance.yaml` bounds (satisfies NFR-REPR-4); Palmer output differences are explicitly expected and asserted to fall within the documented change range from `CHANGELOG.md`

**Given** `CHANGELOG.md` is inspected for the `[2.5.0]` section
**When** a developer reads the full release notes
**Then** the section contains: a complete list of all public-facing changes in v2.5; the Palmer moisture anomaly correction entry with affected function names and a before/after numerical example that extends the minimal Epic 1 entry (completing NFR-REPR-3 full scope); EDDI and Palmer xarray integration entries; a "Known Output Changes" table confirming v2.4.0 regression-baseline equivalence for all non-Palmer indices; links to `VALIDATION.md` and relevant algorithm reference docs

**Given** `uv run ruff check` is run on all new and modified files
**When** the check completes
**Then** zero violations; `mypy --strict` passes on `typed_public_api.py`

---

### Story 2.5: Jupyter Notebook Tutorial Suite

As a new user,
I want end-to-end executable Jupyter notebooks demonstrating SPI/SPEI, Palmer, and EDDI via the xarray API,
So that I can reproduce working xarray pipelines from a sample dataset without writing any setup or data-loading code.

**Acceptance Criteria:**

**Given** `notebooks/` contains three reference notebooks (SPI/SPEI, Palmer, EDDI)
**When** `uv run pytest --nbmake notebooks/` is run (via `notebooks.yml` CI job)
**Then** all three notebooks execute to completion without error on any of the 4 CI matrix legs (Linux/macOS × Python 3.10/3.12); no exceptions are raised during any cell execution

**Given** any reference notebook
**When** its cell structure is inspected
**Then** it follows the standard ordering: (1) title + description markdown, (2) imports, (3 through N-2) tutorial content cells, (N-1) assertion cell, (N) summary markdown; the assertion cell loads tolerances from `tests/fixtures/tolerance.yaml` and validates ≥1 computed index value via `np.testing.assert_allclose`

**Given** the getting-started notebook (SPI/SPEI via xarray, satisfies FR29)
**When** a new user executes it end-to-end
**Then** it produces xarray SPI output from a sample dataset bundled in the repository; no network access is required; computation uses the xarray public API, not the raw numpy API

**Given** all three notebooks execute (satisfies FR30)
**When** their imports are reviewed
**Then** each notebook imports index functions from `climate_indices.typed_public_api` or the stable single-module path; no private or internal functions are called directly

**Given** notebook files are committed to the repository
**When** the `nbstripout` pre-commit hook runs (configured in `.pre-commit-config.yaml`)
**Then** all cell outputs are stripped; only notebook source and cell structure are committed to `release/v2.5`

---

## Epic 3: Documentation Refresh & Discoverability

New users land on the README or PyPI page and immediately find EDDI and Palmer with their validation status. The docs site is Diátaxis-structured with a clear reference section linking algorithm docs, `VALIDATION.md`, and the xarray compatibility guide. An NClimGrid example gallery demonstrates real CONUS drought index maps. `llms.txt` and `llms-full.txt` support AI tooling integrations. The Zenodo DOI is populated in `CITATION.cff` and README post-release.

### Story 3.1: README and PyPI Landing Page Refresh

As a new user,
I want to find EDDI and Palmer prominently in the README and PyPI landing page with their validation status,
So that I can quickly assess what the library computes and how confident I can be in each index's correctness before adopting it.

**Acceptance Criteria:**

**Given** the existing README index table
**When** Story 3.1 is merged
**Then** the table contains a validation-status column for each index (SPI, SPEI, EDDI, PDSI, PHDI, Z-Index, PMDI, scPDSI, PNP); each status entry links to the corresponding row in `VALIDATION.md`; EDDI and Palmer indices show their current status (satisfies FR31 full scope)

**Given** a new user reads the README
**When** they look for EDDI and Palmer
**Then** both appear prominently in the index table and any introductory index description; a dual-citation guidance section explicitly instructs researchers to cite both the library (via `CITATION.cff` / Zenodo DOI) and the originating algorithm paper for the index used (satisfies FR13 full scope, extending the minimal Epic 1 entry)

**Given** the README serves as the PyPI `long_description`
**When** the package page is viewed on pypi.org after v2.5 release
**Then** EDDI and Palmer are visible above the fold in the index summary; the validation-status column renders correctly in PyPI's Markdown renderer; no broken links or raw markup artifacts appear

**Given** `uv run sphinx-build -W docs/ docs/_build/html` is run after Story 3.1
**When** the build completes
**Then** zero warnings; all README-linked content (VALIDATION.md, algorithm reference docs) resolves correctly; no broken cross-references

---

### Story 3.2: Documentation Site Diátaxis Restructure

As a developer using the docs site,
I want the documentation to follow the Diátaxis information architecture with clear sections for tutorials, how-to guides, reference, and explanation,
So that I can find the right information for my task without navigating through misclassified content.

**Acceptance Criteria:**

**Given** the existing flat docs site structure
**When** Story 3.2 is merged
**Then** `docs/how-to/` exists and contains ≥1 how-to guide (e.g., `compute-spi-from-netcdf.md`); `notebooks/` serves as the tutorials section; sphinx-autodoc generates the reference API section; `docs/algorithm_refs/` and `VALIDATION.md` are accessible from an "Explanation" or "Background" top-level navigation section (satisfies FR33)

**Given** a researcher accesses the docs Reference or Explanation section
**When** they look for algorithm and validation documentation
**Then** `docs/algorithm_refs/eddi.md` and `docs/algorithm_refs/palmer.md` are linked; `VALIDATION.md` is linked from both the Explanation section and the README validation-status section; `docs/xarray_compatibility.md` is linked from the Reference section

**Given** `uv run sphinx-build -W docs/ docs/_build/html` is run after restructure
**When** the build completes
**Then** zero warnings; all cross-references between docs pages resolve correctly; no orphaned `.rst` or `.md` files remain in the docs tree; stale navigation entries are removed

**Given** the Sphinx `conf.py` or navigation configuration
**When** the top-level structure is reviewed
**Then** top-level sections are: Getting Started (or Tutorials), How-To Guides, Reference, and Background (or Explanation); the old flat layout is removed; the navigation matches the Diátaxis model documented in the Architecture

---

### Story 3.3: NClimGrid Example Gallery

As a researcher evaluating the library,
I want a static gallery of drought index maps generated from real NClimGrid data,
So that I can visually verify the library produces scientifically plausible CONUS drought patterns before adopting it.

**Acceptance Criteria:**

**Given** `docs/_static/gallery/` does not yet exist
**When** Story 3.3 is merged
**Then** `docs/_static/gallery/` contains pre-generated PNG files for: SPI-3, SPI-6, SPEI-3, PDSI, PHDI, Z-Index, EDDI-3; ≥7 PNG files are committed to `release/v2.5`

**Given** the gallery PNGs are committed to the repository
**When** `scripts/generate_gallery.py` is run by a maintainer who has NOAA SFTP access
**Then** the script regenerates all gallery PNGs into `docs/_static/gallery/`; this script is not run in CI (no NOAA SFTP dependency in any CI workflow); the script's usage and NOAA access requirements are documented in `CONTRIBUTING.md` or a `docs/how-to/` guide

**Given** gallery images are embedded in documentation
**When** `uv run sphinx-build -W docs/ docs/_build/html` is run
**Then** all gallery image references resolve and render; zero broken image links; zero Sphinx build warnings attributable to gallery changes

**Given** a researcher reads the docs gallery page
**When** they examine each image
**Then** each PNG is accompanied by a caption or alt-text identifying: the index name, the time scale (e.g., "3-month"), the approximate period visualized, and a note that the underlying data is NOAA NClimGrid

---

### Story 3.4: AI Tooling Integration Artifacts

As a developer using AI coding tools,
I want `llms.txt` and `llms-full.txt` files at the repository root describing the library,
So that AI-assisted code generation tools can provide accurate, context-aware suggestions for climate index computation without hallucinating API details.

**Acceptance Criteria:**

**Given** `llms.txt` does not exist at the repo root
**When** Story 3.4 is merged
**Then** `llms.txt` exists at the repo root; it is hand-maintained and concise (≤200 lines); it describes the library's public API surface, primary index functions, key conventions (xarray input support, CF compliance, exception hierarchy), and the canonical import path

**Given** `llms.txt` is read by a developer or AI assistant trying to call `compute_spi`
**When** they follow the guidance in the file
**Then** they can identify the correct import path, expected input types (numpy array or xarray DataArray), required parameters (scale, periodicity, calibration years), and the exception type to expect for invalid input — without reading source code

**Given** `scripts/generate_llms_full.py` is run
**When** the script completes
**Then** `llms-full.txt` is generated at the repo root; it aggregates full API reference content from docstrings and `docs/`; it is ≥2× the length of `llms.txt`

**Given** the release workflow runs for v2.5.0
**When** the publish step executes
**Then** `scripts/generate_llms_full.py` is called before the PyPI publish step; the resulting `llms-full.txt` is either committed to the release commit or attached as a release artifact

---

### Story 3.5: Release Automation and Zenodo DOI Finalization

As a maintainer,
I want the release automation to apply the `v2.5.0` git tag, trigger the Zenodo webhook, populate the Zenodo DOI into `CITATION.cff` and the README, and publish to PyPI,
So that every researcher who cites the library receives a persistent, version-specific DOI that permanently resolves to the v2.5.0 release.

**Acceptance Criteria:**

**Given** `scripts/update_citation.py` is run with no arguments
**When** the script completes
**Then** it reads `version` from `pyproject.toml`; it patches `CITATION.cff` `version:` and `date-released:` fields in place without requiring manual arguments; it exits 0; `cff-validator` passes on the patched file (satisfies NFR-API-3); a passing run of this script is recorded as part of Story 3.5 completion confirmation

**Given** the Zenodo webhook is configured at the repository level (one-time manual setup, done outside CI)
**When** a GitHub Release is published for tag `v2.5.0`
**Then** Zenodo automatically creates a DOI for the release; no custom CI step is required to trigger the webhook — it fires from the GitHub Release event alone

**Given** the Zenodo DOI is assigned after the GitHub Release publishes
**When** a maintainer patches `CITATION.cff` and README with the DOI
**Then** `CITATION.cff` contains the `doi:` field populated with the v2.5.0 Zenodo DOI; the README citation section displays the DOI as a clickable link; `cff-validator` passes on the final `CITATION.cff`

**Given** the git tag `v2.5.0` is applied to the release merge commit
**When** `git tag -l v2.5.0` is run
**Then** the tag exists and points to the final Epic 3 merge commit; the tag is pushed to the remote; the GitHub Release is created from this tag

**Given** Story 3.5 is complete and v2.5.0 is published to PyPI
**When** a user runs `pip install climate-indices`
**Then** version 2.5.0 is installed with the updated README, `CITATION.cff`, and Zenodo DOI populated; the package metadata at pypi.org reflects the v2.5 description including EDDI, Palmer, and validation status

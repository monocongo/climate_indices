# climate_indices v2.5 — Release Brief

> **How to use this document:** Hand this to the PM agent at the start of
> `bmad-create-prd`. Say: *"Please use this brief to create the PRD for the
> v2.5 release."* This is a human-authored input to the BMAD planning process,
> not a BMAD artifact. The PM agent should produce a properly structured
> `PRD.md` from it, which the Architect and subsequent workflows will consume.
>
> Project-wide conventions (tooling, logging, branching, GitHub tracking) are
> captured in `_bmad-output/project-context.md` and do not need to be repeated
> here.

---

## Background

`climate_indices` is an open source Python package for computing standardized
climate indices used in drought monitoring and climate research. It is published
on PyPI and GitHub and is used by researchers, NOAA/NCEI, and climate scientists
worldwide.

Two indices have been added to the package in recent development cycles that are
not yet fully validated against authoritative external reference outputs:

- **EDDI** (Evaporative Demand Drought Index) — implemented but not yet
  validated against official NOAA CPC input/output test fixtures. The lead EDDI
  researcher at NOAA has provided key papers directly to the maintainer team and
  is in contact regarding fixture data.
- **Palmer indices** (PDSI, PHDI, Palmer Z-Index, PMDI) — implemented but not
  yet verified against NOAA official datasets or other widely trusted Palmer
  reference implementations.

Additionally, xarray support has been added in recent releases but is
underdocumented and lacks any user-facing examples. The PyPI and GitHub landing
pages (README) do not yet mention EDDI or Palmer as available indices.

The v2.5 release addresses all three of these gaps through three focused epics.

---

## Release Goals

1. Establish as much validation coverage as possible for EDDI and Palmer,
   including a clear framework for completing validation once NOAA CPC fixtures
   arrive — grounded in algorithmic analysis of primary literature.
2. Surface and formalize xarray support with runnable Jupyter notebook examples
   that users can use as starting points.
3. Refresh documentation for human readability and produce a machine-optimized
   single-file context document for AI tooling, modeled on the BMAD
   `llms-full.txt` format.

---

## Out of Scope for v2.5

The PRD must explicitly capture the following as out of scope:

- Full EDDI validation against NOAA CPC fixtures if those fixtures have not
  arrived by the time the validation story is worked.
- Implementing alternative Palmer variants not already present in the codebase.
- A broad academic literature survey for EDDI or Palmer (only algorithmically
  actionable extraction is in scope).
- Non-CONUS datasets in the example gallery.
- Any graphical user interface.
- Support for non-monthly periodicities in the new notebooks (unless trivially
  supported by existing code).

---

## Epic 1 — Index Validation: EDDI & Palmer

**Goal:** Establish rigorous validation coverage for EDDI and the Palmer family,
grounded first in primary literature and then in acquired reference datasets.

### Story 1.1 — EDDI Literature Extraction & Algorithm Specification

Extract algorithmically-actionable content from the EDDI papers provided by the
lead EDDI researcher. This is not a literature review — the deliverable is a
concise algorithm reference document and a set of numerical test cases derived
from the papers.

Focus areas:
- Exact formula variant(s), parameter defaults, and their cited justification.
- Any numerical worked examples or tabulated results that can serve as citable
  unit test cases.
- Edge case handling: missing data, boundary conditions, climatological period
  assumptions.
- Any implementation choices between documented variants — record the choice and
  its justification.

Deliverables:
- `docs/algorithm_refs/eddi.md` — algorithm reference specifying which formula
  this implementation follows, mapping each major choice to its paper citation,
  and noting any intentional deviations.
- `tests/fixtures/eddi_literature/` — numerical examples from the papers,
  formatted as `pytest.mark.parametrize` inputs.

### Story 1.2 — Palmer Literature Extraction & Algorithm Specification

Same scope and deliverable structure as Story 1.1, applied to the Palmer family.

Papers to cover:
- Palmer (1965) — original PDSI and Z-Index definitions.
- Wells, Goddard & Hayes (2004) — self-calibrating Palmer (scPDSI).
- Karl (1986) — PHDI and modifications.
- Other widely cited methodology papers accessible online (e.g., Alley 1984,
  Dai et al. 2004).

For each paper extract:
- Formula definitions and parameter tables (e.g., Palmer K' weighting table,
  duration factors).
- Any numerical examples or published sample outputs usable as test cases.
- Which algorithmic variant this implementation follows.
- Known sources of inter-implementation disagreement (calibration period
  differences, soil water capacity assumptions) — document these so future
  validation discrepancies are diagnosable.

Deliverables:
- `docs/algorithm_refs/palmer.md`
- `tests/fixtures/palmer_literature/`

### Story 1.3 — Fixture Discovery & Acquisition

Survey all publicly available reference datasets for EDDI and Palmer:

- NOAA CPC and NCEI
- Published peer-reviewed datasets: Cook et al. PDSI, Dai PDSI
- Well-respected open-source implementations: `climate_indices` predecessors,
  `droughtindices`, `pyet`

Identify which fixtures can be downloaded programmatically and committed as
small representative subsets (NetCDF or CSV, under 5 MB each). Document which
fixtures are pending (NOAA CPC contact) and stub placeholder tests with
`pytest.mark.skip(reason="awaiting NOAA CPC fixtures")` so they are visible
in CI but non-blocking.

### Story 1.4 — EDDI Validation Tests

Implement parametrized `pytest` validation tests for EDDI against:
- Literature-extracted numerical examples from Story 1.1.
- Any reference fixtures acquired in Story 1.3.

Document the numerical tolerance and rationale in the test module docstring with
a citation. Add an EDDI section to `VALIDATION.md` covering current status,
known discrepancies, and references to `docs/algorithm_refs/eddi.md`.

### Story 1.5 — Palmer Index Validation Tests

Same scope as Story 1.4 for PDSI, PHDI, Z-Index, and PMDI. Distinguish between
discrepancies that are known algorithmic variants vs. implementation errors,
tracing each back to the relevant citation in `docs/algorithm_refs/palmer.md`.
Update `VALIDATION.md` with Palmer status.

### Story 1.6 — Validation CI Integration

Add a `validation` pytest marker and a separate GitHub Actions job that runs
validation tests independently from unit and integration tests. Fixture-gated
(skipped) tests must not block PRs. The CI job must produce a clear
pass/skip/fail summary artifact.

---

## Epic 2 — xarray Integration & Jupyter Notebook Examples

**Goal:** Make first-class xarray support visible, discoverable, and
demonstrated with runnable examples.

### Story 2.1 — xarray API Audit

Audit all public-facing functions to determine which accept/return
`xarray.DataArray` and `xarray.Dataset` natively vs. require NumPy arrays.
Produce a Markdown compatibility matrix for inclusion in the documentation. For
any gaps identified, determine whether a thin wrapper or `xr.apply_ufunc`
pattern is appropriate and file issues for gaps not addressed in v2.5.

### Story 2.2 — xarray Integration Improvements

Implement the high-priority gaps identified in Story 2.1 within v2.5 scope:
- Preserving coordinate metadata on outputs.
- CF-convention attributes on output variables.
- Dask-chunked array support via `xr.apply_ufunc` with `dask='parallelized'`.

Add unit tests covering xarray input/output paths for at least SPI, SPEI, EDDI,
and PDSI.

### Story 2.3 — Jupyter Notebook: Getting Started with xarray

Notebook: `notebooks/xarray_getting_started.ipynb`

Demonstrates loading a small NClimGrid NetCDF sample (committed to
`notebooks/data/`), computing SPI and SPEI via the xarray API, and plotting
with `matplotlib`/`cartopy`. Keep dependencies minimal; document them in
`notebooks/README.md`.

### Story 2.4 — Jupyter Notebook: Palmer Indices with xarray

Notebook: `notebooks/palmer_indices_xarray.ipynb`

Demonstrates computing PDSI, PHDI, and Z-Index from NClimGrid inputs,
visualizing spatial output, and comparing against a NOAA or literature reference
map (image embed acceptable). Must include a clearly visible validation status
caveat cell linking to `VALIDATION.md` and `docs/algorithm_refs/palmer.md`.

### Story 2.5 — Jupyter Notebook: EDDI with xarray

Notebook: `notebooks/eddi_xarray.ipynb`

Same structure as Story 2.4. Must include a validation status caveat cell
linking to `docs/algorithm_refs/eddi.md`.

### Story 2.6 — Notebook CI Execution Check

Add a GitHub Actions job that executes all notebooks via `nbconvert --execute`
and fails if any cell raises an unhandled exception.

---

## Epic 3 — Documentation Refresh

**Goal:** Overhaul documentation for human readability and produce a
machine-optimized single-file context document for AI tooling.

Reference format for the AI context document:
`https://docs.bmad-method.org/llms-full.txt`

### Story 3.1 — README & PyPI Landing Page Update

- Add EDDI and Palmer to the feature table with a clear ⚠️ Beta / not yet fully
  validated note linking to `VALIDATION.md`.
- Add an xarray compatibility badge or section.
- Include a gallery strip of example index maps (produced in Story 3.3).
- Expand all index acronyms on first use.

### Story 3.2 — Human-Readable Docs Overhaul

Audit the existing Sphinx/MkDocs documentation for outdated API references,
missing indices (EDDI, Palmer), and formatting inconsistencies. Restructure to a
Diátaxis-inspired layout:

- **Tutorials** — Getting Started
- **How-To Guides** — compute a specific index, use with xarray, run on a
  cluster
- **Reference** — API, index formulas, validation status, algorithm references
- **Explanation** — background on each index, algorithmic choices, literature

Ensure all public API docstrings render correctly in the Reference section. Link
`docs/algorithm_refs/eddi.md` and `docs/algorithm_refs/palmer.md` prominently
from the Reference section.

### Story 3.3 — Example Gallery: NClimGrid Index Maps

Produce `scripts/generate_gallery.py` that:
- Downloads or reads the NClimGrid sample dataset.
- Computes SPI-3, SPI-6, SPEI-3, PDSI, PHDI, Z-Index, and EDDI-3 for a
  representative CONUS drought year.
- Saves publication-quality PNG maps (300 dpi, `cartopy` CONUS projection) to
  `docs/gallery/`.

Map visual style should match NOAA/NIDIS drought monitor conventions as closely
as reasonably possible (drought colormap, consistent colorbar labeling). Embed
gallery images in the docs and the README gallery strip.

### Story 3.4 — Machine-Optimized AI Context Document

Produce `llms-full.txt` (and companion `llms.txt` summary) as a single
plain-text file following the format at
`https://docs.bmad-method.org/llms-full.txt`: YAML-like frontmatter followed
by document sections in `<document path="...">` tags.

Content must cover:
- Package overview and purpose
- Index catalog with formula references, validation status, and links to
  algorithm reference docs
- Full API reference (public functions, signatures, type hints, docstring
  content)
- xarray compatibility matrix
- Installation and quickstart
- Architecture and design decisions
- Known limitations and roadmap

Add a `uv run python scripts/generate_llms_txt.py` target to regenerate the
file from source. Register `llms.txt` at the site root following the
`llmstxt.org` convention.

### Story 3.5 — VALIDATION.md

Produce a standalone `VALIDATION.md` at the repo root, linked from the README
and the Reference section of the docs. Sections:

- Overview of the validation philosophy for this package.
- Per-index status table: validated / partial / pending.
- Tolerance criteria and their literature basis.
- Known discrepancies and explanations with citations.
- Instructions for contributors on how to submit reference fixtures.

---

## Infrastructure

### Story 0 — GitHub Issue Generation Script

Implement `scripts/create_github_issues.py` before any epic story begins. This
script is the only sanctioned mechanism for bulk-creating v2.5 GitHub issues.

The script must:
- Read `_bmad-output/sprint-status.yaml`.
- For each story, create a GitHub issue via `gh` CLI with: title, acceptance
  criteria body, `epic:*` + `type:*` + any `status:*` labels, and the `v2.5`
  milestone.
- Write the created issue number back into `sprint-status.yaml` as
  `github_issue` so PR titles and BMAD story files can reference it.
- Be idempotent: skip stories where `github_issue` is already populated.
- Support `--dry-run` (print without calling API) and `--story {slug}` (single
  story, for mid-sprint additions).

---

## Success Criteria

The v2.5 release is complete when all of the following are true:

- All `pytest` unit and integration tests pass.
- Validation tests pass or are explicitly skipped with a documented reason and
  citation.
- Literature-extracted numerical test cases are committed for both EDDI and
  Palmer.
- `docs/algorithm_refs/eddi.md` and `docs/algorithm_refs/palmer.md` are present,
  reviewed, and linked from the documentation.
- All three notebooks execute without error via `nbconvert --execute`.
- `ruff check` and `ruff format --check` pass with zero violations.
- `structlog` is used throughout; no bare stdlib `logging` calls in new or
  modified code.
- `llms-full.txt` and `llms.txt` are generated and committed.
- The README surfaces EDDI, Palmer, and xarray support with validation caveats.
- `VALIDATION.md` is present, accurate, and cited throughout the codebase.
- Gallery PNGs are committed to `docs/gallery/`.
- All new public functions have type hints and Google-style docstrings.
- All v2.5 GitHub issues are closed or explicitly deferred with a comment.
- `scripts/create_github_issues.py` is committed and documented in
  `CONTRIBUTING.md`.
- All `feature/e*` story branches are merged and deleted.
- `release/v2.5` is merged to `main` via the release PR; the v2.5 milestone is
  closed.
- `main` branch CI is green post-merge.

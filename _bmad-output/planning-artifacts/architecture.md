---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7]
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/project-context.md
  - _bmad-output/v25-release-brief.md
  - docs/architecture.md
  - docs/component-inventory.md
  - docs/project-overview.md
  - docs/algorithms.rst
  - docs/floating_point_best_practices.md
workflowType: 'architecture'
project_name: 'climate_indices'
user_name: 'James'
date: '2026-04-12'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

---

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**

Three epics (18 stories total) targeting a single release goal: make
`climate_indices` citable in published research.

- **Epic 1 — Index Validation (P0, 6 stories):** Algorithm reference documents
  (`docs/algorithm_refs/eddi.md`, `docs/algorithm_refs/palmer.md`),
  literature-extracted test fixtures with JSON provenance sidecars,
  parametrized `pytest.mark.validation` tests for EDDI and Palmer,
  `VALIDATION.md` at repo root, dedicated CI validation job.
- **Epic 2 — xarray Integration (P1, 6 stories):** xarray compatibility audit
  and gap fixes (CF attributes, Dask support, coordinate preservation), three
  Jupyter notebooks (SPI/SPEI, Palmer, EDDI via xarray API), notebook CI
  execution job.
- **Epic 3 — Documentation Refresh (P2, 5 stories):** README/PyPI update
  surfacing EDDI and Palmer, Diátaxis-structured docs overhaul, NClimGrid
  gallery PNGs, `llms-full.txt`/`llms.txt`.
- **Infrastructure (pre-work):** `scripts/create_github_issues.py` —
  idempotent issue generation from `sprint-status.yaml`; must land before any
  epic story begins.

**Non-Functional Requirements:**

| ID | NFR | Requirement |
|----|-----|-------------|
| — | API stability | `typed_public_api.py` exports stable; no breaking changes without deprecation cycle (`CHANGELOG.md` entry + in-function `DeprecationWarning`) |
| — | Type safety | `mypy --strict` passes on `typed_public_api.py`; CI fails on regression |
| — | Performance | v2.5 within 20% of v2.4 baseline; `benchmark_baseline_v240.json` committed from v2.4.0 tag; CI benchmark job fast-fails with explicit error if baseline file is absent |
| — | Platform | Python 3.10 + 3.12, Linux and macOS CI runners |
| — | Linting | `ruff check` and `ruff format --check` pass with zero violations |
| — | Logging | `structlog` throughout new and modified code; no stdlib `logging` in library code |
| — | Provenance | All `*_literature/**/*.json` sidecars have required fields; CI sidecar linter enforces schema (pre-commit hook + CI job) |
| NFR-VAL-01 | Validation coverage — citability scope | At merge, every index in citability scope (EDDI, PDSI, PHDI, Z-Index, PMDI, scPDSI) has ≥1 `@pytest.mark.validation` test without `@pytest.mark.validation_pending` that passes against a fixture derived from a cited reference source. Tests carrying `validation_pending` do not count toward the per-index minimum. |
| — | Citation | `CITATION.cff` present and valid per `cff-validator` in CI |
| — | Reproducibility | Same inputs + same version + same `algorithm=` value = same outputs, deterministically. Version-pinned reproducibility via git tag + Zenodo DOI. Behavioral changes in patch releases (even bug fixes to a registered algorithm variant) require a CHANGELOG entry and release note. |
| — | Docs | All new public functions: complete type hints, Google-style docstrings with `Args`, `Returns`, `Examples` |
| — | Contributing guide | `CONTRIBUTING.md` documents the fixture sidecar schema and provenance requirements so the citability infrastructure is maintainable beyond v2.5 |

**Scale & Complexity:**

- Primary domain: Scientific Python library / developer tool
- Complexity level: Medium (brownfield, no UI, no DB, no network)
- Estimated architectural components: 4 new file types (algorithm refs, fixture
  sidecars, `tolerance.yaml`, `VALIDATION.md`), 2 new CI workflows, 1 new
  script, 3 notebooks, ~6 modified source modules

### Epic 1 P0 Rationale — Correctness = Citability

Epic 1 is P0 for **both correctness and citability** — these are not separable
at the bar this release targets. A DOI proves a version exists; it does not
prove the implementation is correct. A journal reviewer asking "how do you know
your drought index calculations are correct?" requires the validation
infrastructure to answer. A library that ships with a DOI but unvalidated
algorithms risks correction or retraction requests after researchers have cited
it in published work.

**Delivery sequencing note:** Within Epic 1, the citability metadata deliverables
(`CITATION.cff`, `VALIDATION.md`) are file drops with no code risk and may be
delivered independently of the validation infrastructure stories. Palmer numeric
correctness (Story 1.2, requires Opus verification) is high-risk and must not
become a gate for metadata-only stories. Story sequencing in the sprint plan
must reflect this: metadata stories proceed in parallel; algorithm validation
stories sequence on the correctness gate.

### Citability vs. Quality Engineering — Priority Distinction

The v2.5 architecture serves two separable goals:

**Citability (P0 — required to ship):**
- Algorithm reference documents confirm which formula is implemented and why
- At least one passing or honestly-disclosed validation test per index
- `VALIDATION.md` discloses current validation status
- `CITATION.cff` provides machine-readable citation metadata
- Git tag `v2.5.0` + Zenodo DOI provide immutability

**Quality engineering (P1 — high value, not a citability gate):**
- Fixture sidecar schema with CI enforcement
- `tolerance.yaml` with configurable thresholds
- Validation marker taxonomy and session hook
- Benchmark baseline guard
- Standardized notebook assertion pattern

If scope is cut, citability requirements cannot be deferred. Quality engineering
patterns may be simplified or deferred to a v2.5.x patch.

### Technical Constraints & Dependencies

- **Existing module ownership:** `indices.py` is frozen (no new functionality);
  all new computation routes through `compute.py` → `xarray_adapter.py`.
- **CF metadata:** `cf_metadata_registry.py` is the sole source of truth for
  `standard_name`, `long_name`, `units`, `valid_min`, `valid_max`. Story 2.1
  (audit) can proceed without new registry entries — it identifies gaps. Story
  2.2 (integration improvements) depends on EDDI and Palmer registry entries
  existing before CF-compliant output can be returned.
- **Dask chunking constraint:** Time dimension must remain a single Dask chunk
  for all climate index computations. `xarray_adapter.py` validates this and
  raises `DimensionMismatchError`; the exception message must include the
  remediation instruction: `rechunk to {'time': -1}`. Any new xarray wrapper
  for Palmer or EDDI must inherit this validation.
- **Exception hierarchy:** All exceptions must descend from `ClimateIndicesError`
  in `exceptions.py`. New validation-specific exceptions go here, not in test
  code. Unrecognized `algorithm=` values must raise `InvalidArgumentError`
  immediately — no silent fallthrough.
- **Palmer `algorithm=` parameter:** Palmer public API functions accept
  `algorithm: str = "original_1965"` as a keyword argument in v2.5. Only one
  implementation is registered; any other value raises `InvalidArgumentError`.
  Extension contract: when a second variant is added, the type narrows to
  `Literal["original_1965", "<new_variant>"]` with a corresponding
  `typed_public_api.py` overload update and CHANGELOG entry. This is a
  deliberate YAGNI exception to avoid a v3.x breaking change.
- **Story 2.1 → 2.2 handoff:** Story 2.1 (xarray audit) produces a gap list as
  a structured deliverable. Story 2.2's acceptance criteria must explicitly
  reference Story 2.1's gap list as the input. This handoff must be structural
  (in AC), not informal.

### Cross-Cutting Concerns

#### Existing Patterns — Follow These

These patterns are established in the codebase. New and modified code must
adhere to them without re-design:

- **Structured logging:** `structlog` throughout; bind context fields per
  canonical table (`index`, `timescale`, `periodicity`, `input_shape`,
  `data_var`); no stdlib `logging`.
- **Exception hierarchy:** Raise specific `ClimateIndicesError` subclasses with
  context attributes; never bare `ValueError` or `RuntimeError`.
- **Dask chunking:** Time dimension = single chunk; validate in
  `xarray_adapter.py`; raise `DimensionMismatchError` with remediation message
  on violation.
- **CF metadata source of truth:** `cf_metadata_registry.py` only; no hardcoded
  CF strings elsewhere.
- **API stability contract:** `typed_public_api.py` is the public surface;
  `mypy --strict` enforced in CI; breaking changes require deprecation entry +
  `DeprecationWarning`.

#### New Patterns — Define These in Architecture

**1. Fixture Sidecar Schema**

Every file under `tests/fixtures/*_literature/` has a companion `.json`
sidecar with the same stem. Required fields:

| Field | Type | Constraint |
|-------|------|------------|
| `source_paper` | str | non-empty |
| `doi` | str | non-empty |
| `equation_ref` | str | non-empty (e.g. `"Eq. 3"`) |
| `table_ref` | str \| null | null if fixture derives from equation only |
| `extraction_method` | enum | one of `"transcribed"` (read directly from table), `"digitized"` (extracted from figure, higher uncertainty), `"computed"`, `"software_comparison"` |
| `comparison_target` | str \| null | required (non-empty) when `extraction_method == "software_comparison"`; null otherwise |
| `citability_scope` | bool | `true` if this fixture is the basis for an NFR-VAL-01 coverage claim |

CI sidecar linter runs in pre-commit hook AND CI lint job. Enforces: all
required fields present; `extraction_method` in enum; `comparison_target`
non-null when `extraction_method == "software_comparison"`; `table_ref` may be
null.

**2. `tolerance.yaml` Schema**

`tests/fixtures/tolerance.yaml` uses a flat per-index structure in v2.5 with a
`_schema_version` key for future-safe migration detection. Per-variant nesting
is deferred until a second algorithm variant is registered; the version key
enables a clean non-breaking migration path when that time comes.

```yaml
_schema_version: 1

spi:
  atol: 1.0e-6
  rtol: 0.0
  justification_category: literature_stated  # enum
  justification: "SPI is dimensionless; tolerance matches paper precision of 4 d.p."
  source_doi: "10.1175/1520-0477(1993)074<1196:ATNPSM>2.0.CO;2"

pdsi:
  atol: 0.01
  rtol: 0.0
  justification_category: numerical_precision
  justification: "Palmer (1965) tables reported to 2 d.p."
  source_doi: "10.1175/1520-0477(1965)..."
```

`justification_category` enum: `literature_stated`, `numerical_precision`,
`digitization_uncertainty`, `algorithm_discretization`. Free-text `justification`
required in addition. A shared pytest fixture in `conftest.py` loads this file;
validation tests inject it and call `np.testing.assert_allclose(actual,
expected, atol=tol["atol"], rtol=tol["rtol"])`.

**3. Correctness Gate — Structural Enforcement**

Stories 1.3–1.5 are structurally blocked on Stories 1.1/1.2. "Correctness
confirmed" means: `docs/algorithm_refs/eddi.md` and `docs/algorithm_refs/palmer.md`
exist, are merged to `release/v2.5`, and satisfy all five criteria:

1. Cites the primary authoritative source (DOI or standard name + section)
2. States the canonical formula in LaTeX or equivalent notation
3. Lists every parameter with units and valid ranges
4. Contains ≥1 numeric reference case (input → expected output, traceable to cited source)
5. Contains a "Validation Provenance" section referencing the associated fixture sidecar(s)

Structural enforcement via pytest — `tests/conftest.py` defines session-scoped
fixtures `eddi_gate` and `palmer_gate` that inspect the algorithm reference doc
for a markdown comparison table (pipe-delimited, first column contains
"Source", "Reference", or "Published"). Stories 1.3–1.5 declare these fixtures
as dependencies; they are structurally uncollectable until the gate docs are
merged. Doc presence with required content = gate passed. No separate CI env
var or approval workflow.

**Palmer-specific:** Story 1.2 requires a second independent verification pass
at Opus reasoning level before the correctness gate is considered cleared.
Palmer's Zf/PDSI numerical chain has known error accumulation risk. Any fixture
written against unconfirmed numerics must be deleted, not marked provisional.

**4. Validation Marker Taxonomy**

Two pytest markers replace any skip-count ceiling:

| Marker | Meaning | Counts against limit? |
|--------|---------|----------------------|
| `@pytest.mark.validation` | Expected to pass at merge; unexpected skip is a regression signal | Yes |
| `@pytest.mark.validation_pending` | Fixture not yet received; skip is expected and managed | No — excluded from all counts |

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
validation_skip_limit = "0"
```

`tests/conftest.py` implements `pytest_runtest_logreport` to count `validation`-only
unexpected skips (tests carrying both markers are excluded). When fixture data
arrives, removing `@pytest.mark.validation_pending` is the graduation event.

Story AC for any validation test stub: use both markers. Remove
`validation_pending` when the fixture is committed.

**5. Benchmark Baseline Guard**

One-time baseline generation (pre-work, run against v2.4.0 tag):
```bash
git checkout v2.4.0
uv run pytest -m benchmark --benchmark-enable \
    --benchmark-json=tests/fixtures/benchmark_baseline_v240.json
git checkout release/v2.5
git add tests/fixtures/benchmark_baseline_v240.json
```

`benchmarks.yml` CI job:
```yaml
- name: Assert baseline exists
  run: |
    test -f tests/fixtures/benchmark_baseline_v240.json || \
      { echo "ERROR: benchmark_baseline_v240.json missing — run pre-work step"; exit 1; }
- name: Run benchmarks
  run: |
    uv run pytest -m benchmark --benchmark-enable \
      --benchmark-compare=tests/fixtures/benchmark_baseline_v240.json \
      --benchmark-compare-fail=mean:20%
```

Baseline regeneration procedure: repeat the one-time generation command against
the new baseline tag. Document the regeneration step in `CONTRIBUTING.md` under
"Updating the benchmark baseline." Regeneration is required when CI hardware
changes or a new major release resets the baseline.

**6. Notebook Assertion Cell Pattern**

Each notebook's penultimate cell (before any summary/next-steps cell) uses:

```python
import yaml
import numpy as np
from pathlib import Path

_tol = yaml.safe_load(
    (Path("..") / "tests" / "fixtures" / "tolerance.yaml").read_text()
)["<index_name>"]
atol, rtol = _tol["atol"], _tol["rtol"]

np.testing.assert_allclose(
    computed_value,
    reference_value,
    atol=atol,
    rtol=rtol,
    err_msg=f"Index value outside documented tolerance (atol={atol}, rtol={rtol})",
)
print("Assertion passed.")
```

Notebook CI uses `nbval` or `nbmake` (not bare `nbconvert --execute`) to
surface cell-level assertion failures with useful output. `AssertionError`
propagates as an unhandled cell exception, failing the CI job. Notebook
assertions are additive — they supplement pytest validation tests, not replace
them. A passing notebook assertion is not a substitute for a passing
`@pytest.mark.validation` test.

### Implementation Sequencing Constraint

Stories 1.1 (EDDI literature extraction) and 1.2 (Palmer literature extraction)
must reach correctness-confirmed state before validation test infrastructure
(Stories 1.3–1.5) is built. Validation infrastructure assumes correctness; it
does not establish it. If algorithm reference review reveals a fundamental
implementation error, fixture tolerances and test design may need to change.

The `eddi_gate` / `palmer_gate` pytest fixtures enforce this structurally.

### `algorithm=` Parameter — Conscious YAGNI Exception

Palmer public API functions accept `algorithm: str = "original_1965"` as a
keyword argument in v2.5. Only one implementation is registered. This is a
deliberate departure from YAGNI: the cost of future API breakage when a second
variant is added exceeds the cost of a one-line parameter stub now. Unrecognized
values raise `InvalidArgumentError` immediately.

Extension contract: when a second variant ships, the signature narrows to
`Literal["original_1965", "<new_variant>"]` with a CHANGELOG entry. No
migration tooling needed — tolerance values must be updated for new variants
regardless.

Agents should not question why this parameter exists with one valid value — it
is intentional and load-bearing for v3.x planning.

---

## Technical Foundation

### Confirmed Stack

Brownfield project — no starter template required. The following stack is
established and must not be changed without explicit architectural review:

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| Package manager | `uv` (dev tooling; library installs via pip) |
| Core numerics | NumPy, xarray/Dask, scipy |
| Logging | structlog (stdlib `logging` banned in library code) |
| Type checking | `mypy --strict` on `typed_public_api.py` |
| Linting/formatting | ruff (line-length 120) |
| Testing | pytest |
| CI | GitHub Actions — Linux + macOS, Python 3.10 + 3.12 |
| Notebooks | Jupyter (3 new notebooks in v2.5) |

### Foundation Risk Register

The following gaps and risks were identified through adversarial review of the
technical foundation. Each has a mandated resolution before or during the
relevant epic.

---

#### RISK-F-01 — CITATION.cff version drift (Critical)

**Risk:** `CITATION.cff` requires a `version` field. Nothing automates updating
it at release time. A stale version/DOI actively harms citability — researchers
citing a mismatched version produce irreproducible references.

**Resolution (required before v2.5.0 tag):** The GitHub Actions release workflow
(or a `scripts/` helper) must update `CITATION.cff` `version:` and
`date-released:` fields automatically. Until automation exists, a manual checklist
item in the release runbook is mandatory. Story 3.5 (docs/citation work) must
include this automation as an AC.

---

#### RISK-F-02 — Notebook execution standard unresolved (Critical)

**Risk:** The architecture specifies "nbval or nbmake (not bare nbconvert)" but
does not decide. nbval and nbmake impose fundamentally different disciplines:

- **nbval** — re-runs cells and compares against stored outputs; requires committed
  outputs; forces output-scrubbing discipline on every commit
- **nbmake** — re-executes without output comparison; simpler CI, but cannot
  catch output regressions

The notebook authoring standard (commit outputs or strip them?) follows from
this choice and affects every Epic 2 notebook story.

**Resolution (decide before Story 2.3):** Use **nbmake**. Rationale: notebooks
in a scientific library are demonstrations, not golden-output contracts. Output
comparison adds maintenance burden without proportional correctness benefit —
the assertion cell pattern (RISK pattern 6 from Step 2) provides the numerical
gate. Notebook outputs are stripped before commit (add `.gitattributes` or
`nbstripout` pre-commit hook). CI job runs `uv run pytest --nbmake notebooks/`.

---

#### RISK-F-03 — `import logging` ban unenforced (Significant)

**Risk:** The "no stdlib `logging` in library code" rule exists in CLAUDE.md but
not in CI tooling. Six stories with multiple agents will violate it silently.
Simple grep patterns miss `from logging import getLogger` and aliased imports.

**Resolution — two-layer enforcement:**

**Primary (ruff TID251):** Add to `pyproject.toml`. Fires on every
`uv run ruff check` call and in any editor with the ruff LSP — catches
violations at the point of introduction:

```toml
[tool.ruff.lint.flake8-tidy-imports]
banned-module-imports = ["logging"]
```

**Backstop (CI grep):** Add to `lint.yml` as defense-in-depth for `--no-verify`
bypasses and transitive import paths ruff cannot statically trace:

```yaml
- name: Ban stdlib logging in library code
  run: |
    if grep -rn "import logging\|from logging import\|logging\.getLogger" \
        src/climate_indices/; then
      echo "ERROR: stdlib logging found in library code — use structlog"
      exit 1
    fi
```

Both run in the existing `lint.yml` job. No new workflow required.

---

#### RISK-F-04 — Correctness gate relies on document shape, not content (Significant)

**Risk:** The `eddi_gate`/`palmer_gate` pytest fixtures pass when the algorithm
reference doc contains a pipe-delimited table whose first column contains
"Source", "Reference", or "Published". This checks document structure, not
correctness. A doc with a malformed or trivial table passes the gate.

**Resolution:** The gate is explicitly a *presence and structure* check, not a
correctness oracle — that distinction must be documented in `tests/conftest.py`
with a comment. The gate's purpose is to block Stories 1.3–1.5 from running
before the reference doc is merged, not to validate the science. Human review
of the algorithm reference doc content remains mandatory (Story 1.2 Palmer
verification requirement stands).

The exact structural check: the fixture must find a table row matching
`r"^\|\s*(Source|Reference|Published)"` (case-insensitive). This is tighter
than arbitrary pipe tables.

---

#### RISK-F-05 — Benchmark CI flakiness on shared runners (Significant)

**Risk:** GitHub Actions shared runners exhibit 15–25% variance between
identical runs. The 20% `--benchmark-compare-fail=mean:20%` threshold can
produce false-positive failures due to runner load, not actual regression.

**Resolution:** The benchmark job is informational in v2.5, not a hard merge
gate. Set `continue-on-error: true` on the benchmark job. A comment in
`benchmarks.yml` documents: "Failures here are advisory — investigate before
merging but do not block on runner variance alone." A dedicated self-hosted
runner or `pytest-benchmark` histogram comparison can harden this in v2.6.

---

#### RISK-F-06 — Sidecar JSON null syntax creates contributor friction (Notable)

**Risk:** JSON requires explicit `"table_ref": null` for optional fields.
Researchers contributing fixtures will omit the key, producing invalid sidecars
that CI must reject. This creates friction on every contributor PR.

**Resolution:** The sidecar linter error message must be explicit:

```
ERROR: tests/fixtures/eddi_literature/case1.json missing required field 'table_ref'.
If the fixture derives from an equation only (no table), set: "table_ref": null
```

Contributor documentation in `CONTRIBUTING.md` must include a complete sidecar
example with the `null` value shown explicitly. No schema change — JSON is
retained for strict parsing guarantees.

---

#### RISK-F-07 — No conda-forge installation path (Notable, Out of Scope v2.5)

**Risk:** Many researchers on HPC clusters use conda/mamba, not pip. No
conda-forge feedstock or `environment.yml` is planned.

**Resolution (deferred):** Out of scope for v2.5. Document explicitly in
`CONTRIBUTING.md` that a conda-forge recipe is a v2.6 goal. Add a note to the
v2.5 release announcement pointing pip-only users to the PyPI install
instructions and HPC users to `pip install --user`.

---

#### RISK-F-08 — `tolerance.yaml` flat schema migration path (Acknowledged, Mitigated)

**Risk:** The flat per-index schema will require migration when Palmer
`algorithm="variant_2"` ships, since per-variant tolerances must differ.
Without a detection mechanism, schema version would have to be inferred from
key presence — fragile and opaque.

**Resolution (decided):** Flat schema in v2.5 with `_schema_version: 1` key
at the top of the file. The version key costs nothing to write now and provides
a clean machine-readable migration detection point later.

When a second algorithm variant ships, migration procedure:

1. Bump `_schema_version: 2`
2. The conftest loader detects version and dispatches to the appropriate schema
   reader (one-time migration function, not a grep-replace)
3. Migrate `tolerance.yaml` to nested form: `{index: {variant: {atol, rtol, ...}}}`
4. Update all notebooks
5. Bump minor version and add CHANGELOG entry

This migration path is recorded here so it is not re-litigated when the time
comes.

---

#### RISK-F-09 — Cross-platform BLAS/LAPACK variance (Significant)

**Risk:** CI runs Linux + macOS. SPI and SPEI use numerical optimization
(L-BFGS-B, distribution fitting) via scipy. BLAS/LAPACK implementations differ
between Linux (OpenBLAS) and macOS Apple Silicon (Accelerate). Tolerances that
pass on one platform may fail on the other, producing false CI failures that
erode trust in the correctness gate.

**Resolution:** Tolerances in `tolerance.yaml` must be validated across the
full CI matrix (Linux + macOS, Python 3.10 + 3.12) before any entry is
considered confirmed. The CI validation job must aggregate pass/fail across
all matrix legs — a tolerance breach on any single leg is a failure, even if
other legs pass. Document the tested platform set in `VALIDATION.md` so
researchers running on other hardware understand the scope of the validation
claim.

---

#### RISK-F-10 — Provenance chain not explicitly linked (Significant)

**Risk:** Three artifacts carry the citability claim — fixture sidecars,
`tolerance.yaml`, and `VALIDATION.md` — but they are planned as independent
deliverables with no explicit cross-references. A journal reviewer asking "what
is the tolerance and why is it acceptable for the science?" must manually
connect these three documents. Disconnected provenance weakens the claim.

**Resolution:** The provenance chain is: sidecar JSON (what was validated,
from which source) → `tolerance.yaml` (numerical bounds and justification) →
`VALIDATION.md` (human-readable summary with per-index status) → `CITATION.cff`
(machine-readable citation with DOI). Story ownership:

- Sidecar schema: Epic 1, Stories 1.1/1.2 (algorithm reference docs)
- `tolerance.yaml` entries: Epic 1, Stories 1.3–1.5 (validation tests)
- `VALIDATION.md`: Epic 3 (docs refresh), but must link back to per-index
  sidecar files and their `source_doi` fields
- `CITATION.cff`: must reference `VALIDATION.md` in its `notes:` field

Explicit cross-reference requirement: `VALIDATION.md` must contain, for each
index in citability scope, a table row with: index name, validation status,
`source_doi` from the sidecar, and the `atol`/`rtol` from `tolerance.yaml`.

---

#### RISK-F-11 — Cross-matrix validation not enforced in CI (Significant)

**Risk:** The validation CI job runs on one matrix leg by default. Scipy
numerical behavior can differ subtly across platforms. A passing result on
Linux does not guarantee a passing result on macOS (see RISK-F-09).

**Resolution:** The `validation.yml` CI job must use a matrix strategy covering
all four combinations (Linux/macOS × Python 3.10/3.12). The job fails if any
single matrix leg reports a tolerance breach. This is additive to the existing
test CI matrix — it does not replace it.

---

#### RISK-F-12 — `docs/algorithm_refs/*.md` ownership undefined (Significant)

**Risk:** `docs/algorithm_refs/eddi.md` and `docs/algorithm_refs/palmer.md`
do not exist. Stories in Epic 1 depend on these files but the architecture does
not specify: (a) which story creates them, (b) whether they are written before
or after the implementation stories that depend on them, (c) who owns them if
two stories touch the same index.

**Resolution:** One dedicated story per algorithm reference doc, explicitly
sequenced *before* any implementation story that uses the correctness gate:

- **Story 1.1** creates `docs/algorithm_refs/eddi.md` (EDDI literature
  extraction). Stories 1.3+ for EDDI are blocked on this story being merged.
- **Story 1.2** creates `docs/algorithm_refs/palmer.md` (Palmer literature
  extraction + Opus verification). Stories 1.4+ for Palmer are blocked on
  this story being merged.

No other story may modify these files without explicit AC stating the change.
The `eddi_gate`/`palmer_gate` pytest fixtures enforce this structurally — but
the story ownership must be explicit in the sprint plan, not just implied by
the gate.

---

## Core Architectural Decisions

### Decision Priority Analysis

**Critical (block implementation):**
- CF attribute enforcement strategy (Decision 3.2) — affects every Epic 2 story
- Story 2.1 gap deliverable format (Decision 3.1) — blocks Story 2.2 AC
- Validation CI matrix scope (Decision 1.1) — required by RISK-F-11

**Important (shape architecture):**
- Pre-commit hook scope (Decision 1.2) — affects all sprint contributors
- Sidecar linter implementation (Decision 5.1) — blocks Story 1.x
- CITATION.cff automation (Decision 2.2) — required before v2.5.0 tag

**Deferred post-v2.5:**
- Harden CF enforcement from soft (Option C) to always-attach (Option A) — once
  all registry entries exist post-Story 2.2

---

### CI/CD Pipeline Architecture

**Validation job matrix (Decision 1.1):** Full 4-leg matrix in `validation.yml`:
Linux + macOS × Python 3.10 + 3.12. Required by RISK-F-11 (cross-platform
BLAS/LAPACK variance). Fails if any single leg reports a tolerance breach.

**CI workflow inventory (final):**

| Workflow file | Trigger | Content |
|---|---|---|
| `tests.yml` | push/PR | pytest (excludes `benchmark`, `validation` markers) |
| `lint.yml` | push/PR | ruff check, ruff format --check, mypy --strict, CI grep for `import logging` |
| `validation.yml` | push/PR, full matrix | `@pytest.mark.validation` tests, sidecar linter, correctness gate |
| `benchmarks.yml` | push to `release/*`, manual | pytest-benchmark, `continue-on-error: true`, advisory only |
| `notebooks.yml` | push/PR | `uv run pytest --nbmake notebooks/` |

**Pre-commit hook scope (Decision 1.2):** `.pre-commit-config.yaml` includes:
1. `ruff` — lint + format check
2. `mypy` — `--strict` on `src/climate_indices/typed_public_api.py` only
3. `scripts/lint_sidecars.py` — sidecar JSON schema validation
4. `nbstripout` — strip notebook outputs before commit

No pytest in pre-commit (too slow for interactive commits).

---

### Zenodo / DOI Release Automation

**Trigger (Decision 2.1):** Standard GitHub Release → Zenodo webhook. Zenodo
integration is configured at the repository level (connect repo to Zenodo once);
every published GitHub Release automatically mints a new DOI. No custom CI step
required for the trigger itself.

**CITATION.cff automation (Decision 2.2):** `scripts/update_citation.py`
patches `version:` and `date-released:` fields. Called as a step in the release
GitHub Actions workflow before the release is published. The script reads the
target version from `pyproject.toml` and today's date; it must not require
manual arguments. Story 3.5 AC must include a passing run of this script.

---

### xarray Integration Architecture

**Story 2.1 audit deliverable (Decision 3.1):** `docs/xarray_gaps.md` — a
markdown file structured as a table with columns: index name, gap category
(CF attribute missing / Dask incompatible / coordinate not preserved / other),
severity (blocking / advisory), and suggested fix. Story 2.2's AC must
explicitly reference `docs/xarray_gaps.md` as its input specification. This
file is created in Story 2.1 and deleted or archived after Story 2.2 is merged.

**CF attribute enforcement (Decision 3.2 — Option C, soft enforcement):**
`xarray_adapter.py` looks up CF attributes from `cf_metadata_registry.py` on
every return. If a registry entry exists, attributes are attached automatically.
If no entry exists, a `structlog` warning is emitted at `WARNING` level:

```python
logger.warning(
    "cf_metadata_missing",
    index=index_name,
    message="No CF registry entry found — output DataArray has no standard_name/units",
)
```

No exception is raised. This allows Story 2.1 to audit indices without registry
entries without raising. Once Story 2.2 adds EDDI and Palmer registry entries,
the warnings for those indices disappear. Any remaining warnings after Story 2.2
merges are regression signals.

**Post-v2.5 hardening path:** After all in-scope indices have registry entries,
a follow-on story converts missing-entry behaviour from warning to
`MissingCFMetadataError` (new `ClimateIndicesError` subclass). This is a
behaviour change requiring a CHANGELOG entry and minor version bump.

---

### Documentation Architecture

**Diátaxis structure (Decision 4.1):**

| Diátaxis quadrant | Maps to |
|---|---|
| Tutorials | `notebooks/` — learning-oriented walkthroughs |
| How-to guides | `docs/how-to/` — goal-oriented usage recipes |
| Reference | Auto-generated API docs from docstrings (sphinx-autodoc or mkdocstrings) |
| Explanation | `docs/algorithm_refs/`, `VALIDATION.md`, `docs/floating_point_best_practices.md` |

**NClimGrid gallery PNGs (Decision 4.2 — Option C):**
Pre-generated PNGs committed to `docs/_static/gallery/`. Generation script:
`scripts/generate_gallery.py` — downloads current NClimGrid data from NOAA
SFTP, runs index computations, writes PNGs to `docs/_static/gallery/`.
Regeneration is required: (a) before each release, (b) when visualization code
or colormap choices change. `CONTRIBUTING.md` documents the download + regenerate
procedure. PNGs are not regenerated in CI — NOAA SFTP is an external dependency
unsuitable for CI reliability.

**`llms.txt` / `llms-full.txt` (Decision 4.3):**
- `llms.txt` — hand-maintained; short project summary optimised for LLM context
  windows; lives at repo root; updated manually as part of Epic 3 docs refresh
- `llms-full.txt` — generated by `scripts/generate_llms_full.py`, which
  concatenates all markdown files under `docs/` in Diátaxis order; regenerated
  as part of the release workflow

---

### Fixture / Sidecar Architecture

**Sidecar linter implementation (Decision 5.1):**
`scripts/lint_sidecars.py` — standalone Python script using `jsonschema` for
schema validation. Invoked as:
- Pre-commit local hook (`language: python`, `entry: uv run python scripts/lint_sidecars.py`, `files: tests/fixtures/.*\.json$`)
- Step in `validation.yml` CI job

The JSON schema is defined inline in the script (not a separate `.json` schema
file) to keep the contract co-located with the enforcement.

**Story 2.1 → 2.2 handoff (Decision 5.2):** Consistent with Decision 3.1.
The handoff artifact is `docs/xarray_gaps.md`. Story 2.2 story file must list
closure of every row in that table as an AC item.

---

### Decision Impact on Story Sequencing

```
Pre-work:      scripts/create_github_issues.py
               scripts/update_citation.py (skeleton)
               scripts/lint_sidecars.py (skeleton)
               .pre-commit-config.yaml
               .github/workflows/validation.yml
               .github/workflows/benchmarks.yml
               .github/workflows/notebooks.yml

Story 1.1:     docs/algorithm_refs/eddi.md          → unblocks eddi_gate
Story 1.2:     docs/algorithm_refs/palmer.md         → unblocks palmer_gate (Opus pass required)
Stories 1.3-1.5: validation tests (gated on 1.1/1.2)

Story 2.1:     xarray audit → docs/xarray_gaps.md   → unblocks Story 2.2
Story 2.2:     gap fixes + CF registry entries       → CF warnings resolve
Stories 2.3-2.6: notebooks (depend on 2.2 for correct xarray output)

Epic 3:        docs refresh (independent of Epics 1-2 except VALIDATION.md
               depends on 1.3-1.5 status; CITATION.cff automation in Story 3.5)
```

---

## Implementation Patterns & Consistency Rules

### Critical Conflict Points

9 areas where agents implementing different stories could diverge and produce
incompatible code. Every agent working on this sprint MUST read this section
before writing a single line.

---

### Pattern 1: Validation Test Anatomy

Every `@pytest.mark.validation` test follows this exact structure. Do not
deviate — inconsistency in marker stacking or fixture injection will break
the skip-count hook and the correctness gate.

```python
import pytest
import numpy as np
from climate_indices import compute  # or typed_public_api

@pytest.mark.validation
@pytest.mark.validation_pending          # remove when fixture is committed
@pytest.mark.parametrize("case", [
    pytest.param("case1", id="case1_eddi_4week"),
    pytest.param("case2", id="case2_eddi_8week"),
])
def test_eddi_against_literature(case, eddi_gate, tolerances, eddi_literature_fixtures):
    """Validate EDDI against {source_paper} reference values."""
    fixture = eddi_literature_fixtures[case]
    result = compute.eddi(fixture["input"], scale=fixture["scale"])
    tol = tolerances["eddi"]
    np.testing.assert_allclose(
        result,
        fixture["expected"],
        atol=tol["atol"],
        rtol=tol["rtol"],
        err_msg=f"EDDI case '{case}' outside documented tolerance (atol={tol['atol']})",
    )
```

**Rules:**
- `@pytest.mark.validation` always first; `@pytest.mark.validation_pending` always second (when present)
- `@pytest.mark.parametrize` always third
- Gate fixtures (`eddi_gate`, `palmer_gate`) are positional params, not used in body
- `tolerances` fixture (shared conftest) loads `tests/fixtures/tolerance.yaml`
- Test function name: `test_{index}_against_literature`
- Docstring must name the source paper

---

### Pattern 2: Fixture Directory and Sidecar Naming

```
tests/fixtures/
  {index}_literature/          # e.g. eddi_literature/, palmer_literature/
    {case_id}.npy              # or .csv — the reference data array
    {case_id}.json             # sidecar — same stem, same directory
  tolerance.yaml
  benchmark_baseline_v240.json
```

**Rules:**
- Directory name: `{index}_literature` — lowercase, underscore-separated, always `_literature` suffix
- Case ID: `{index}_{descriptor}` — e.g. `eddi_4week_conus`, `palmer_pdsi_1965_table3`
- Sidecar stem = data file stem exactly — never diverge
- `citability_scope: true` only on fixtures that are the basis for an NFR-VAL-01 claim

**Sidecar template** (copy this exactly, fill all fields):

```json
{
  "source_paper": "McKee et al. (1993)",
  "doi": "10.1175/1520-0477(1993)074<1196:ATNPSM>2.0.CO;2",
  "equation_ref": "Eq. 3",
  "table_ref": "Table 1",
  "extraction_method": "transcribed",
  "comparison_target": null,
  "citability_scope": true
}
```

For `extraction_method: "digitized"`, `justification_category` in
`tolerance.yaml` MUST be `"digitization_uncertainty"`.

---

### Pattern 3: Tolerance Loading in Tests

All validation tests load tolerances via a shared `conftest.py` session-scoped
fixture. Never load `tolerance.yaml` inline in a test file.

```python
# tests/conftest.py — already provided, do not duplicate
@pytest.fixture(scope="session")
def tolerances():
    import yaml
    from pathlib import Path
    data = yaml.safe_load(
        (Path(__file__).parent / "fixtures" / "tolerance.yaml").read_text()
    )
    assert data.get("_schema_version") == 1, "Unexpected tolerance.yaml schema version"
    return data
```

Usage in test:
```python
def test_spi_against_literature(tolerances, ...):
    tol = tolerances["spi"]          # never tolerances["spi"]["original_1965"]
    np.testing.assert_allclose(..., atol=tol["atol"], rtol=tol["rtol"])
```

When `_schema_version` is bumped to 2 (nested schema), the conftest fixture is
the single migration point. Tests do not change.

---

### Pattern 4: Correctness Gate Fixture Declaration

Stories 1.3–1.5 declare gate fixtures as unused parameters (not injected into
body). This is intentional — the fixture raises `pytest.skip` / blocks
collection if the gate doc is absent.

```python
# CORRECT — gate declared but not used in body
def test_eddi_validation(case, eddi_gate, tolerances, fixtures):
    ...

# WRONG — do not assert on the gate fixture
def test_eddi_validation(case, eddi_gate, tolerances, fixtures):
    assert eddi_gate  # ← never do this
```

Gate fixture regex (for `tests/conftest.py` implementation):

```python
import re
_GATE_PATTERN = re.compile(r"^\|\s*(Source|Reference|Published)", re.IGNORECASE | re.MULTILINE)
```

---

### Pattern 5: Structlog Bind Field Names

Canonical field names — use these exactly. Never invent new top-level fields
without updating this table in the architecture document.

| Field | Type | When to bind |
|-------|------|-------------|
| `index` | str | All index computation functions |
| `timescale` | int \| None | SPI, SPEI, EDDI scale parameter |
| `periodicity` | str | `"monthly"` or `"daily"` |
| `input_shape` | tuple | Shape of input array at function entry |
| `data_var` | str \| None | xarray DataArray variable name, if applicable |
| `algorithm` | str | Palmer functions only; value = `algorithm` kwarg |
| `calibration_year_initial` | int | When calibration period is specified |
| `calibration_year_final` | int | When calibration period is specified |

```python
# CORRECT
logger = get_logger(__name__)
logger.bind(
    index="eddi",
    timescale=scale,
    periodicity=periodicity,
    input_shape=values.shape,
).info("eddi_computation_start")

# WRONG — never log data values
logger.bind(values=values.tolist())  # ← security + performance violation
```

---

### Pattern 6: CF Warning Emission (Decision 3.2)

When `xarray_adapter.py` cannot find a CF registry entry for an index:

```python
from climate_indices.logging_config import get_logger
logger = get_logger(__name__)

# In xarray_adapter.py, after registry lookup:
if cf_attrs is None:
    logger.warning(
        "cf_metadata_missing",
        index=index_name,
        message="No CF registry entry found — output DataArray has no standard_name/units",
    )
    return da  # return bare DataArray, no attrs attached
```

**Rules:**
- Event name: `"cf_metadata_missing"` — exactly this string, used for log filtering
- Log level: `WARNING` — not `info`, not `error`
- No exception raised — soft enforcement until all registry entries exist
- The `index` field must be bound so warnings are filterable per-index

---

### Pattern 7: Notebook Cell Ordering

Every v2.5 notebook follows this cell sequence:

| Position | Cell type | Content |
|----------|-----------|---------|
| 1 | Markdown | Title + one-paragraph description |
| 2 | Code | Imports (stdlib → third-party → climate_indices) |
| 3–N-2 | Mixed | Tutorial content — data loading, computation, visualization |
| N-1 | Code | **Assertion cell** (see below) — validates key result against tolerance |
| N | Markdown | Summary + next steps / links |

Outputs are stripped before commit (`nbstripout` pre-commit hook). Do not
commit notebooks with outputs.

**Assertion cell template:**

```python
import yaml
import numpy as np
from pathlib import Path

_tol = yaml.safe_load(
    (Path("..") / "tests" / "fixtures" / "tolerance.yaml").read_text()
)["<index_name>"]             # replace <index_name> with actual index key

np.testing.assert_allclose(
    computed_value,
    reference_value,
    atol=_tol["atol"],
    rtol=_tol["rtol"],
    err_msg=f"Index value outside documented tolerance (atol={_tol['atol']}, rtol={_tol['rtol']})",
)
print("Assertion passed.")
```

`computed_value` and `reference_value` must be scalars or 1-D arrays — never
full 2-D grids (assertion failure message becomes unreadable).

---

### Pattern 8: New Exception Classes

All new exceptions for v2.5 go in `src/climate_indices/exceptions.py` and
descend from `ClimateIndicesError`.

```python
# Naming convention: {Domain}{Problem}Error
class MissingCFMetadataError(ClimateIndicesError):    # post-v2.5 hardening
    """Raised when no CF metadata registry entry exists for an index."""
    def __init__(self, index_name: str) -> None:
        super().__init__(f"No CF registry entry for index '{index_name}'")
        self.index_name = index_name
```

**Rules:**
- Class name: `{Domain}{Problem}Error` — PascalCase, always ends in `Error`
- Subclass the most specific existing parent first; fall back to `ClimateIndicesError`
- Store structured context as instance attributes (not only in the message string)
- Add to `exceptions.py` `__all__` list
- New exceptions go in `exceptions.py` only — never define them in test files or adapters

---

### Pattern 9: Algorithm Reference Doc Structure

`docs/algorithm_refs/{index}.md` must follow this section order exactly. The
`eddi_gate`/`palmer_gate` fixtures check for a pipe-delimited table with a
header row matching `^\|\s*(Source|Reference|Published)`.

```markdown
# {Index Name} Algorithm Reference

## Overview
[one paragraph — what the index measures and why it matters]

## Authoritative Source
[DOI or standard citation — this is the primary reference]

## Canonical Formula
[LaTeX block — the formula as defined in the source]

## Parameters
| Parameter | Units | Valid range | Description |
|-----------|-------|-------------|-------------|

## Reference Cases
| Source | Input | Expected output | Notes |
|--------|-------|-----------------|-------|
[≥1 row — traceable to the authoritative source]

## Validation Provenance
[Links to associated fixture sidecars and tolerance.yaml entry]
```

The `## Reference Cases` table is what the gate fixture detects. The header
row must contain "Source" as the first column — do not rename it.

---

### Enforcement Summary

**All agents MUST:**
- Stack validation markers in order: `validation` → `validation_pending` → `parametrize`
- Use `{index}_literature/` directory naming — never `{index}_fixtures/` or `{index}_data/`
- Load tolerances only through the shared `conftest.py` fixture — never inline
- Declare gate fixtures as unused positional params — never assert on them
- Use canonical structlog field names from Pattern 5 — never invent new fields
- Emit CF warnings with event name `"cf_metadata_missing"` — exactly
- Strip notebook outputs before commit — never commit with outputs
- Place new exceptions in `exceptions.py` — never define them in test files or adapters
- Use `## Reference Cases` as the exact section name in algorithm reference docs

**Verification:** `uv run pytest tests/test_pattern_compliance.py` (created in
pre-work) enforces structural rules via static analysis — sidecar schema, marker
taxonomy, tolerance.yaml version field, algorithm ref doc sections.

---

## Project Structure & Boundaries

### Fixture Directory Convention

All patterns in this document that reference `tests/fixtures/` (plural) use
`tests/fixture/` (singular) — consistent with the existing directory. The
`conftest.py` tolerance fixture path is
`Path(__file__).parent / "fixture" / "tolerance.yaml"`.

---

### Complete Project Directory Structure

Files annotated as `[NEW]`, `[MOD]`, or `[TEMP]` relative to current
`release/v2.5` state.

```
climate_indices/
├── CITATION.cff                          [MOD] — automated by scripts/update_citation.py (Story 3.5)
├── CHANGELOG.md                          [MOD] — entries per story
├── CONTRIBUTING.md                       [MOD] — add sections: fixture sidecar schema,
│                                                  benchmark regeneration, conda-forge deferral (RISK-F-07)
├── VALIDATION.md                         [NEW] — skeleton in pre-work; stub in Story 1.6; complete in Story 3.5
├── llms.txt                              [NEW] — hand-maintained LLM context summary (Story 3.4)
├── llms-full.txt                         [NEW] — generated by scripts/generate_llms_full.py (Story 3.4)
├── pyproject.toml                        [MOD] — new markers, ruff TID251, validation_skip_limit
├── .pre-commit-config.yaml               [NEW] — created AFTER lint_sidecars.py exists (pre-work step 3)
│
├── .github/
│   └── workflows/
│       ├── unit-tests-workflow.yml       (exists) — no marker changes needed; addopts handles exclusion
│       ├── lint.yml                      [NEW or MOD] — see pre-work Step 0: audit before creating
│       ├── validation.yml                [NEW] — 4-leg matrix, sidecar linter, correctness gate
│       ├── benchmarks.yml               [MOD] — add continue-on-error: true (RISK-F-05)
│       ├── notebooks.yml                 [NEW] — uv run pytest --nbmake notebooks/
│       └── release.yml                  [MOD] — add scripts/update_citation.py step before publish
│
├── scripts/
│   ├── create_github_issues.py          (exists) — pre-work idempotent issue generation
│   ├── lint_sidecars.py                  [NEW] — JSON schema validator (pre-work step 1, before pre-commit)
│   ├── update_citation.py                [NEW] — skeleton in pre-work; completed in Story 3.5
│   ├── generate_gallery.py               [NEW] — downloads NClimGrid, produces gallery PNGs (Story 3.3)
│   └── generate_llms_full.py             [NEW] — concatenates docs/ in Diátaxis order (Story 3.4)
│
├── docs/
│   ├── algorithm_refs/                   [NEW dir] — created in pre-work via .gitkeep
│   │   ├── .gitkeep                      [NEW] — pre-work creates directory
│   │   ├── eddi.md                       [NEW] — Story 1.1: EDDI literature extraction
│   │   └── palmer.md                     [NEW] — Story 1.2: Palmer lit extraction + Opus pass
│   ├── how-to/                           [NEW dir] — Diátaxis how-to quadrant (Story 3.2)
│   │   └── (*.md — goal-oriented usage recipes)
│   ├── _static/
│   │   └── gallery/
│   │       └── (*.png)                   [NEW] — NClimGrid pre-generated gallery PNGs (Story 3.3)
│   ├── xarray_gaps.md                    [TEMP] — Story 2.1 output; deleted when Story 2.2 merges
│   └── (existing docs unchanged)
│
├── notebooks/
│   ├── spi_spei_xarray.ipynb             [NEW] — Story 2.3
│   ├── palmer_xarray.ipynb               [NEW] — Story 2.4
│   ├── eddi_xarray.ipynb                 [NEW] — Story 2.5
│   └── (existing notebooks unchanged)
│
├── src/climate_indices/
│   ├── exceptions.py                     [MOD] — MissingCFMetadataError stub (post-v2.5 hardening only)
│   ├── cf_metadata_registry.py           [MOD] — EDDI + Palmer registry entries (Story 2.2 only)
│   ├── xarray_adapter.py                 [MOD] — CF warning pattern (Decision 3.2, Story 2.2)
│   ├── compute.py                        [MOD] — Palmer algorithm= parameter (Story 2.2)
│   ├── typed_public_api.py               [MOD] — Palmer @overload signatures (Story 2.2)
│   └── (all other modules unchanged)
│
└── tests/
    ├── conftest.py                       [MOD] — pre-work adds: tolerances(), eddi_gate, palmer_gate
    │                                             Story 1.6 adds: pytest_runtest_logreport skip-count hook
    │                                             EXISTING fixtures/hooks must not be removed
    ├── test_noaa_eddi_reference.py       (exists) — NOT replaced by test_validation_eddi.py;
    │                                             see Boundary Note below
    ├── test_pattern_compliance.py        (exists) — updated in pre-work to enforce new patterns
    ├── test_provenance_protocol.py       (exists) — audit for schema overlap before pre-work finalises;
    │                                             candidate for chore/deprecation after lint_sidecars.py confirmed
    ├── test_validation_eddi.py           [NEW] — Story 1.3: @pytest.mark.validation tests only
    ├── test_validation_palmer_pdsi.py    [NEW] — Story 1.4: PDSI, PHDI, Z-Index validation tests
    ├── test_validation_palmer_scpdsi.py  [NEW] — Story 1.5: PMDI, scPDSI validation tests
    └── fixture/
        ├── provenance_schema.json        (exists) — do not delete until lint_sidecars.py confirmed in CI
        ├── tolerance.yaml                [NEW] — pre-work creates skeleton; Stories 1.3–1.5 add entries
        ├── benchmark_baseline_v240.json  [NEW] — pre-work step 10 (requires v2.4.0 tag)
        ├── eddi_literature/              [NEW dir] — Story 1.1/1.3
        │   ├── {case_id}.npy
        │   └── {case_id}.json           (sidecar)
        └── palmer_literature/            [NEW dir] — Story 1.2/1.4–1.5
            ├── {case_id}.npy
            └── {case_id}.json           (sidecar)
```

---

### Boundary Note: `test_noaa_eddi_reference.py` vs `test_validation_eddi.py`

| File | Marker | Purpose |
|------|--------|---------|
| `test_noaa_eddi_reference.py` | none / existing | Regression tests against NOAA reference data |
| `test_validation_eddi.py` | `@pytest.mark.validation` | Literature-validated tests with sidecar provenance |

Do not merge these files. Do not add `@pytest.mark.validation` to existing NOAA
reference tests.

---

### Boundary Note: `test_validation_palmer_pdsi.py` vs `test_validation_palmer_scpdsi.py`

These files are intentionally split to eliminate merge risk between Stories 1.4
and 1.5. They must not be recombined. Story 1.5 is explicitly sequenced after
Story 1.4 for `tolerance.yaml` additions; the test files are independent.

---

### `tolerance.yaml` Write Protocol

Three stories (1.3, 1.4, 1.5) add entries to `tolerance.yaml`. To prevent
overwrites:

1. Pre-work creates the skeleton with `_schema_version: 1` and commented
   placeholders for all citability-scope indices.
2. Each story **appends only** its assigned index entries. Never overwrite the
   file from scratch.
3. Sequence: Story 1.3 (eddi) → Story 1.4 (pdsi, phdi, z_index) → Story 1.5
   (pmdi, scpdsi). Story 1.4 is blocked on Story 1.3 being merged; Story 1.5
   is blocked on Story 1.4 being merged.

---

### `VALIDATION.md` Lifecycle and Schema

Three stories touch this file with strictly increasing content:

| Stage | Story | Content |
|-------|-------|---------|
| Skeleton | pre-work | Header + empty table shell |
| Stub | Story 1.6 | Table populated with `pending` status for all 6 indices |
| Complete | Story 3.5 | All rows filled: status, source_doi, atol, rtol |

**Fixed column schema (set in pre-work, never changed):**

```markdown
# Validation Status

| Index | Status | Source DOI | atol | rtol | Notes |
|-------|--------|-----------|------|------|-------|
| EDDI  | pending | — | — | — | |
| PDSI  | pending | — | — | — | |
| PHDI  | pending | — | — | — | |
| Z-Index | pending | — | — | — | |
| PMDI  | pending | — | — | — | |
| scPDSI | pending | — | — | — | |
```

Story 3.5 replaces `pending` with `validated` or `validation_pending`. Column
schema is owned by the architecture — no story may add or remove columns.

---

### Pre-work Ordering (dependency-ordered)

```
Step 0:  Audit .github/workflows/unit-tests-workflow.yml
         — if lint steps (ruff/mypy) present: [MOD] that file for new additions
         — if absent: create .github/workflows/lint.yml [NEW]

Step 1:  scripts/lint_sidecars.py
         — audit overlap with tests/fixture/provenance_schema.json
         — if lint_sidecars.py is a superset: note test_provenance_protocol.py
           as a candidate for a chore/deprecation story (do not delete yet)

Step 2:  pyproject.toml
         — new markers, ruff TID251, validation_skip_limit = "0"

Step 3:  .pre-commit-config.yaml
         — references lint_sidecars.py from step 1 (must exist first)

Step 4:  tests/conftest.py [MOD]
         — ADD ONLY: tolerances(), eddi_gate, palmer_gate session fixtures
         — skip-count hook stays in Story 1.6; do not add it here

Step 5:  tests/fixture/tolerance.yaml SKELETON
         — _schema_version: 1, commented placeholders for all 6 indices
         — no atol/rtol values yet

Step 6:  docs/algorithm_refs/.gitkeep
         — creates directory for Stories 1.1/1.2

Step 7:  VALIDATION.md SKELETON
         — header + empty table shell (Story 1.6 fills stub; Story 3.5 completes)

Step 8:  .github/workflows/validation.yml skeleton
         .github/workflows/notebooks.yml skeleton
         .github/workflows/benchmarks.yml [MOD: continue-on-error: true]
         lint.yml [NEW or MOD per Step 0]

Step 9:  scripts/update_citation.py SKELETON
         — reads version from pyproject.toml, patches CITATION.cff
         — no-op skeleton; completed in Story 3.5

Step 10: tests/fixture/benchmark_baseline_v240.json
         REQUIRES v2.4.0 tag to exist (verify: git tag -l | grep v2.4.0)
         If tag absent: escalate to repo owner — DO NOT proceed without it.
         Command when tag confirmed:
           git checkout v2.4.0
           uv run pytest -m benchmark --benchmark-enable \
               --benchmark-json=tests/fixture/benchmark_baseline_v240.json
           git checkout release/v2.5
           git add tests/fixture/benchmark_baseline_v240.json

Step 11: tests/test_pattern_compliance.py
         — update to enforce new patterns (sidecar schema, marker taxonomy,
           tolerance.yaml version field, algorithm ref doc sections)
```

---

### Architectural Boundaries

**Module ownership (enforced, no exceptions):**

| Module | Owner | Rule |
|--------|-------|------|
| `indices.py` | frozen | No modifications — legacy entry point |
| `compute.py` | Story 2.2 only | Palmer algorithm= parameter; no Epic 1 changes |
| `xarray_adapter.py` | Story 2.2 only | CF warning pattern; Dask validation must not be weakened |
| `cf_metadata_registry.py` | Story 2.2 only | Only story that adds EDDI/Palmer CF entries |
| `exceptions.py` | any epic | Any story may add subclasses; never define exceptions elsewhere |
| `typed_public_api.py` | Story 2.2 only | Palmer @overload changes only in v2.5 |

**File ownership — single-story locks:**

| File | Owning story | Rule |
|------|-------------|------|
| `docs/algorithm_refs/eddi.md` | Story 1.1 only | gate doc — no modifications without explicit AC |
| `docs/algorithm_refs/palmer.md` | Story 1.2 only | gate doc + Opus-verified — see RISK-F-12 |
| `docs/xarray_gaps.md` | Story 2.1 (creates), Story 2.2 (consumes + deletes) | no other stories |
| `tests/fixture/tolerance.yaml` | Pre-work (skeleton), 1.3/1.4/1.5 (append only) | never overwrite |
| `CITATION.cff` | Story 3.5 + release workflow only | no other stories |
| `VALIDATION.md` | pre-work (skeleton), Story 1.6 (stub), Story 3.5 (complete) | fixed column schema |

---

### Requirements to Structure Mapping

**Infrastructure pre-work (steps 0–11, dependency-ordered above).**

**Epic 1 — Index Validation (P0):**
```
Story 1.1 → docs/algorithm_refs/eddi.md (directory exists from pre-work)
          → tests/fixture/eddi_literature/ [NEW dir], *.{npy,json}
          → unblocks eddi_gate → unblocks Story 1.3

Story 1.2 → docs/algorithm_refs/palmer.md (Opus pass required)
          → tests/fixture/palmer_literature/ [NEW dir], *.{npy,json}
          → unblocks palmer_gate → unblocks Stories 1.4–1.5

Story 1.3 → tests/test_validation_eddi.py
          → tests/fixture/tolerance.yaml [append eddi entry]
          → blocks Story 1.4 on tolerance.yaml merge

Story 1.4 → tests/test_validation_palmer_pdsi.py (PDSI, PHDI, Z-Index)
          → tests/fixture/tolerance.yaml [append pdsi, phdi, z_index entries]
          → blocks Story 1.5 on tolerance.yaml merge

Story 1.5 → tests/test_validation_palmer_scpdsi.py (PMDI, scPDSI)
          → tests/fixture/tolerance.yaml [append pmdi, scpdsi entries]

Story 1.6 → tests/conftest.py [MOD: skip-count hook only]
          → .github/workflows/validation.yml [complete]
          → VALIDATION.md [stub: 6 rows, status=pending]
```

**Epic 2 — xarray Integration (P1):**
```
Story 2.1 → docs/xarray_gaps.md [TEMP]
Story 2.2 → src/climate_indices/xarray_adapter.py, cf_metadata_registry.py,
             compute.py, typed_public_api.py → deletes docs/xarray_gaps.md
Stories 2.3–2.5 → notebooks/spi_spei_xarray.ipynb, palmer_xarray.ipynb, eddi_xarray.ipynb
Story 2.6   → .github/workflows/notebooks.yml [complete]
```

**Epic 3 — Documentation Refresh (P2):**
```
Story 3.1 → README.md, docs/ index pages
Story 3.2 → docs/how-to/ [NEW dir + initial how-to guides]
Story 3.3 → docs/_static/gallery/*.png, scripts/generate_gallery.py,
             CONTRIBUTING.md [MOD: sidecar schema, benchmark regen, conda-forge note]
Story 3.4 → llms.txt, llms-full.txt, scripts/generate_llms_full.py
          → release.yml [MOD: add generate_llms_full.py step with existence assertion]
Story 3.5 → CITATION.cff [automated], scripts/update_citation.py [complete],
             VALIDATION.md [complete: all 6 rows filled],
             release.yml [MOD: add update_citation.py step, ZENODO_TOKEN fast-fail check]
```

---

### Integration Points

**Correctness gate flow:**
```
Story 1.1/1.2 merges
  → docs/algorithm_refs/*.md present with ## Reference Cases table
      → eddi_gate / palmer_gate pytest fixtures pass
          → test_validation_*.py collectable
              → validation.yml 4-leg matrix runs
```

**Provenance chain (RISK-F-10):**
```
tests/fixture/{index}_literature/{case}.json  (source, DOI, extraction method)
    ↓
tests/fixture/tolerance.yaml                   (atol/rtol + justification category)
    ↓
VALIDATION.md                                  (per-index status table linking both)
    ↓
CITATION.cff                                   (notes: field references VALIDATION.md)
```

**Story 2.1 → 2.2 handoff:**
```
Story 2.1 produces: docs/xarray_gaps.md (table: index | gap_category | severity | fix)
Story 2.2 AC: close every row; delete docs/xarray_gaps.md on merge
```

---

### `pyproject.toml` Changes (pre-work)

```toml
[tool.pytest.ini_options]
addopts = "-m 'not benchmark'"
validation_skip_limit = "0"
markers = [
    "validation: expected to pass at merge; unexpected skip is a regression signal",
    "validation_pending: awaiting fixture data; skip is expected and managed",
    "benchmark: performance benchmarks excluded from default run",
    "slow: long-running tests",
    "release: pre-release integrity checks",
]

[tool.ruff.lint.flake8-tidy-imports]
banned-module-imports = ["logging"]
```

---

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**

All technology choices are internally consistent. No conflicts found:
- ruff TID251 and CI grep backstop are additive, not contradictory
- nbmake choice (RISK-F-02) is consistent with notebook cell ordering (Pattern 7) and `nbstripout` pre-commit hook
- `algorithm=` YAGNI exception and `tolerance.yaml` flat schema use the same version-key-now pattern
- `continue-on-error: true` on benchmarks is consistent with "advisory, not gate" decision
- Correctness gate (doc presence + structure) is correctly scoped as structural, not a science oracle (RISK-F-04)

**Pattern Consistency:**

| Pattern | Decision | Aligned? |
|---------|---------|---------|
| P1 — validation test anatomy | Marker taxonomy | ✅ |
| P2 — fixture naming | `tests/fixture/` convention | ✅ |
| P3 — tolerance loading | `conftest.py` session fixture | ✅ |
| P4 — gate fixture declaration | `eddi_gate`/`palmer_gate` | ✅ |
| P5 — structlog fields | structlog throughout; no stdlib | ✅ |
| P6 — CF warning | Decision 3.2 soft enforcement | ✅ |
| P7 — notebook cell ordering | nbmake + nbstripout | ✅ |
| P8 — exception classes | `ClimateIndicesError` hierarchy | ✅ |
| P9 — algorithm ref doc structure | Gate fixture regex | ✅ |

**Structure Alignment:**

Pre-work foundations are correctly ordered to unblock all epic stories. File ownership prevents
cross-story interference. Module ownership prevents cross-epic source conflicts. `tolerance.yaml`
append-only protocol prevents Story 1.3–1.5 overwrites.

---

### Requirements Coverage Validation ✅

**Epic Coverage:**

| Epic | Coverage |
|------|---------|
| Epic 1 — Index Validation (P0) | ✅ algorithm ref docs, fixture sidecars, tolerance.yaml, validation tests, correctness gates, CI matrix, VALIDATION.md |
| Epic 2 — xarray Integration (P1) | ✅ audit + gap fixes + CF registry + notebooks + notebook CI |
| Epic 3 — Docs Refresh (P2) | ✅ README, Diátaxis, gallery PNGs, llms files, CITATION.cff + Zenodo |
| Infrastructure pre-work | ✅ 11 dependency-ordered steps |

**NFR Coverage:**

| NFR | Architectural support |
|-----|----------------------|
| API stability | `typed_public_api.py` + deprecation pattern documented |
| Type safety | mypy --strict in pre-commit + lint.yml |
| Performance | benchmarks.yml advisory gate + baseline file |
| Platform | 4-leg validation matrix + pinned-deps leg (GAP-6) |
| Linting | ruff in pre-commit + lint.yml; TID251 for stdlib logging ban |
| Logging | Pattern 5 canonical fields; ruff TID251 enforces ban |
| Provenance | sidecar schema + lint_sidecars.py + CI enforcement |
| NFR-VAL-01 (citability coverage) | validation marker taxonomy + correctness gates |
| Citation | CITATION.cff automation + cff-validator |
| Reproducibility | `algorithm=` parameter + tolerance.yaml versioning |
| Docs | Google-style docstrings documented in conventions |
| Contributing guide | CONTRIBUTING.md [MOD] scope explicit |

**Citability P0 gate:**

| Requirement | Delivered by | Status |
|-------------|-------------|--------|
| Algorithm reference documents | Stories 1.1/1.2 | ✅ |
| ≥1 passing validation test per index | Stories 1.3–1.5 | ✅ |
| `VALIDATION.md` | Story 1.6 stub + Story 3.5 complete | ✅ |
| `CITATION.cff` | Story 3.5 | ✅ |
| Git tag + Zenodo DOI | release.yml + Zenodo webhook | ✅ |

---

### Implementation Readiness Validation ✅

**Decision completeness:** All 5 decision areas have explicit decisions with rationale. No TBD
placeholders remain.

**Structure completeness:** Complete file tree with NEW/MOD/TEMP/exists annotations. Pre-work
ordering is explicit with conditional logic (Step 0 lint audit). All directories and ownership
boundaries defined.

**Pattern completeness:** All 9 patterns include code examples. Enforcement summary is actionable.
`tests/test_pattern_compliance.py` provides automated enforcement.

---

### Gap Analysis

**Critical gaps:**

**GAP-2 — Zenodo repository connection is a structural dependency, not a process gap**

The `release.yml` workflow has a hard dependency on the Zenodo GitHub integration being
configured before it runs. If `ZENODO_TOKEN` is absent, DOI generation silently fails —
the release publishes but has no DOI, directly undermining the citability goal.

**Resolution:** Add a fast-fail guard as the first step of `release.yml` (Story 3.5):

```yaml
- name: Assert Zenodo integration configured
  run: |
    if [ -z "${{ secrets.ZENODO_TOKEN }}" ]; then
      echo "ERROR: ZENODO_TOKEN secret is not set."
      echo "Connect this repository to Zenodo at https://zenodo.org/account/settings/github/"
      echo "and add the ZENODO_TOKEN secret before publishing a release."
      exit 1
    fi
```

The one-time human setup action (connecting the repo to Zenodo) must be documented in
`CONTRIBUTING.md` under "Pre-release checklist."

---

**Important gaps:**

**GAP-3 — Palmer Opus verification process and gate mechanism**

**Documentation:** `docs/algorithm_refs/palmer.md` must include a `## Verification Record`
section (Story 1.2 AC). Required format — must contain a table row with a numeric/keyword result:

```markdown
## Verification Record

| Check | Method | Result | Date | Notes |
|-------|--------|--------|------|-------|
| Zf/PDSI chain | Claude Opus 4.6 independent review | Confirmed / Discrepancy | YYYY-MM-DD | ... |
```

**Enforcement — standalone test (not palmer_gate extension):** A test in
`tests/test_validation_palmer_pdsi.py::test_opus_verification_record_present` asserts
the section exists and contains a populated table row. This keeps `palmer_gate` focused
on algorithm reference document structure:

```python
def test_opus_verification_record_present():
    """Verify palmer.md contains a completed Verification Record (Story 1.2 Opus pass)."""
    from pathlib import Path
    import re
    content = Path("docs/algorithm_refs/palmer.md").read_text()
    assert "## Verification Record" in content, \
        "palmer.md missing ## Verification Record — Story 1.2 Opus pass required"
    assert re.search(
        r"^\|.*(Confirmed|Discrepancy|[\d.]+).*\|",
        content,
        re.MULTILINE,
    ), "Verification Record table row must contain a result — empty section does not satisfy Story 1.2"
```

**GAP-4 — `generate_llms_full.py` missing from `release.yml`**

Story 3.4 AC must include the exact step addition to `release.yml`:

```yaml
- name: Generate llms-full.txt
  run: uv run python scripts/generate_llms_full.py
- name: Assert llms-full.txt generated
  run: test -f llms-full.txt || { echo "ERROR: llms-full.txt not generated"; exit 1; }
```

**GAP-1 — `cff-validator` CI step unassigned**

Add to pre-work step 8 (lint.yml or unit-tests-workflow.yml per Step 0 audit):

```yaml
- name: Validate CITATION.cff
  uses: citation-file-format/cff-validator@v3
```

**Minor gaps:**

**GAP-5 — `nbstripout` missing from pre-work step 3**

Pre-work step 3 must include:

```bash
uv add --dev nbstripout
uv run nbstripout --install
git config --list | grep nbstripout  # verify activation
```

Without `nbstripout --install`, the pre-commit hook is declared but not activated.
Notebook stories will commit cell outputs, causing spurious CI diffs.

**GAP-6 — SciPy/NumPy version variance unaddressed in `validation.yml` matrix**

The 4-leg platform matrix catches OS-level BLAS/LAPACK variance but not NumPy/SciPy
version drift. Add a fifth pinned-dependencies matrix leg:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest]
    python-version: ["3.10", "3.12"]
    include:
      - os: ubuntu-latest
        python-version: "3.12"
        pinned: true   # runs with uv sync --frozen (lockfile)
```

`atol` values in `tolerance.yaml` must be derived empirically from the full 5-leg matrix:
set `atol = max_observed_deviation * 1.5` and record the derivation in the `justification`
field.

---

### Architectural Corrections Applied

**`palmer_gate` scope:**

`palmer_gate` uses `autouse=False` (the pytest default) — the fixture is injected only by
tests that explicitly list it as a parameter. A missing `palmer.md` does not prevent
contributors working on SPI or any other module from running `pytest`. Constraint: never
add `autouse=True` to either gate fixture.

**`tolerance.yaml` sprint sequencing constraint (explicit):**

Story 1.4 branch must be cut after Story 1.3 is merged to `release/v2.5`.
Story 1.5 branch must be cut after Story 1.4 is merged to `release/v2.5`.

Branching before the preceding story's merge causes both stories to append at the same EOF
anchor in `tolerance.yaml`, producing a git merge conflict. This constraint must be stated
explicitly in the sprint plan.

**`validation_pending` staleness policy:**

Tests carrying `@pytest.mark.validation_pending` must be promoted to `validation` (or
deleted) within 90 days of fixture data becoming available. The sprint plan must assign a
fixture data owner for each pending test at story creation time. Document in `CONTRIBUTING.md`.

---

### Architecture Completeness Checklist

**✅ Requirements Analysis**
- [x] Project context thoroughly analysed — citability vs quality engineering distinction explicit
- [x] Scale and complexity assessed — brownfield, medium complexity, 4 new file types
- [x] Technical constraints identified — 12 foundation risks with resolutions
- [x] Cross-cutting concerns mapped — existing patterns (5) + new patterns (9)

**✅ Architectural Decisions**
- [x] Critical decisions documented (5 decision areas, 11 explicit decisions)
- [x] Technology stack fully specified — confirmed brownfield stack
- [x] Integration patterns defined — correctness gate flow, provenance chain, Story 2.1→2.2 handoff
- [x] Performance considerations addressed — RISK-F-05 (benchmark flakiness), RISK-F-09 (cross-platform)

**✅ Implementation Patterns**
- [x] Naming conventions established — fixture dirs, case IDs, test function names, log field names
- [x] Structure patterns defined — 9 patterns with code examples
- [x] Communication patterns specified — conftest fixtures as shared contract between stories
- [x] Process patterns documented — tolerance append-only, gate scope, exception hierarchy

**✅ Project Structure**
- [x] Complete directory tree with NEW/MOD/TEMP/exists annotations
- [x] Component boundaries established — module and file ownership tables
- [x] Integration points mapped — gate flow, provenance chain, handoff artifacts
- [x] Requirements-to-structure mapping explicit per story

---

### Architecture Readiness Assessment

**Overall Status: READY FOR IMPLEMENTATION**

**Confidence Level: High**

**Key strengths:**
- Structural enforcement (gates, fixtures, CI matrix) replaces policy-only rules — agents
  cannot accidentally skip correctness gates
- Explicit pre-work ordering with conditional logic prevents "works on my machine" failures
- File and module ownership tables make parallel agent work safe
- Provenance chain is fully connected end-to-end
- `palmer_gate` scoped via `autouse=False` — non-Palmer contributors are never blocked
- `ZENODO_TOKEN` fast-fail check prevents silent DOI failure at release time

**Areas for future enhancement (post-v2.5):**
- Harden CF enforcement from soft warning to `MissingCFMetadataError` (Decision 3.2)
- Conda-forge feedstock (deferred per RISK-F-07)
- `tolerance.yaml` migration to per-variant nested schema (RISK-F-08 migration path documented)
- Self-hosted benchmark runner to eliminate RISK-F-05 flakiness
- Expand validation matrix as NumPy/SciPy major versions are released

---

### Implementation Handoff

**AI Agent Guidelines:**

1. Start with pre-work steps 0–11 in strict dependency order — no epic story begins until
   all pre-work is merged
2. Verify `v2.4.0` tag exists before step 10 — escalate to repo owner if absent
3. Verify Zenodo repository connection before cutting the `v2.5.0` release tag
4. Each story's deliverables are explicitly mapped in the Requirements-to-Structure section —
   implement exactly what is listed, nothing more
5. Consult file and module ownership tables before touching any existing file
6. `tolerance.yaml` — append only, never overwrite; branch sequencing is mandatory (1.3→1.4→1.5)
7. Run `uv run pytest tests/test_pattern_compliance.py` after each story

**First implementation step:** Pre-work step 0 — audit `unit-tests-workflow.yml`.

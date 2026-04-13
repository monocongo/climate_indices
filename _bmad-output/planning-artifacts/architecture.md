---
stepsCompleted: [1, 2, 3, 4, 5]
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


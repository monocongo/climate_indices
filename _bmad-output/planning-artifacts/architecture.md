---
stepsCompleted: [1, 2]
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
| `extraction_method` | enum | one of `"digitized"`, `"verbatim"`, `"computed"`, `"software_comparison"` |
| `comparison_target` | str \| null | required (non-empty) when `extraction_method == "software_comparison"`; null otherwise |
| `citability_scope` | bool | `true` if this fixture is the basis for an NFR-VAL-01 coverage claim |

CI sidecar linter runs in pre-commit hook AND CI lint job. Enforces: all
required fields present; `extraction_method` in enum; `comparison_target`
non-null when `extraction_method == "software_comparison"`; `table_ref` may be
null.

**2. `tolerance.yaml` Schema**

`tests/fixtures/tolerance.yaml` uses a flat per-index structure in v2.5. Per-variant
nesting deferred until a second algorithm variant is registered.

```yaml
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

# How to contribute

Thanks for helping improve `climate_indices`. Keep contributions focused,
tested, and easy to review.

## Development workflow

This project uses trunk-based development. `main` is the trunk and should always
be releasable.

1. Start from an updated `main`:
   ```bash
   git switch main
   git pull --ff-only origin main
   ```
2. Create a short-lived branch:
   ```bash
   git switch -c feature/<short-topic>
   ```
3. Make focused changes with tests and docs when needed.
4. Run local validation.
5. Open a pull request into `main`.
6. The maintainer merges after review and passing CI. Contributors and coding
   agents open PRs but never merge them.

Use these branch prefixes:

- `feature/<topic>` for user-visible features
- `fix/<topic>` for bug fixes
- `docs/<topic>` for documentation-only changes
- `chore/<topic>` for maintenance
- `perf/<topic>` for performance work
- `refactor/<topic>` for behavior-preserving restructuring
- `test/<topic>` for test-only changes
- `ci/<topic>` for workflow changes
- `hotfix/<topic>` for urgent release fixes

Avoid long-lived `release/*` branches. Maintenance branches for older supported
versions require maintainer approval.

## Coding conventions

We optimize for readability and scientific reproducibility:

- Indent using four spaces.
- Use underscores instead of camelCase.
- Prefer explicit, descriptive names over abbreviations.
- Keep changes scoped to one issue or topic.
- Do not mix functional changes with unrelated whitespace cleanup.
- Add type hints and Google-style docstrings for public functions.
- Add tests for new behavior and bug fixes.

## Fire-weather scope

`climate_indices` accepts meteorological and climatological fire-weather
indices; operational fire-danger and fire-behavior modeling is out of scope.
The exact boundary, the criteria for a future `fire_weather_indices` sibling
repository, and concrete in-scope and out-of-scope examples are in
[the fire subsystem design](docs/design/fire-subsystem.md#scope-and-boundary).
Point an out-of-scope proposal there rather than dismissing it: it is a
candidate for that sibling, not a rejection on merit.

## Documentation audience

Everything on the published site is for users and contributors of the shipped
library, and every page under `docs/` is either published or internal:

- **Published** pages serve exactly one reader need: tutorial (learning),
  how-to guide (doing a job), reference (looking something up), or
  explanation (understanding). Markdown sources are enabled, but pages join
  the site as tickets wire them into navigation and links; a published page
  not yet wired in stays in `exclude_patterns` so it cannot render
  half-linked.
- **Internal** pages are working notes — agent guidance, design and planning
  scratch, research — that stay in the repository without being published.

The split exists so the site teaches one version of the truth. Working notes
move with the code, are addressed to maintainers, and would otherwise reach
readers as competing instructions. Because a Markdown suffix publishes every
page it can find, pages are excluded by path in `docs/conf.py`
(`exclude_patterns`), not by convention: internal pages are never built into
the site. A published page stays excluded until it is wired into navigation: a
link to an excluded page fails the warnings-as-errors build, and the
release-integrity test rejects a published page parked in a hidden toctree, so
publishing a page and adding it to its section toctree happen together. When a
page is added or moves
between audiences, update that exclusion list and this section in the same
change.

Current assignments (the context map and validation status live at the
repository root — `CONTEXT-MAP.md`, `VALIDATION.md` — and the core-library
vocabulary at `src/climate_indices/CONTEXT.md`):

- Router: `docs/index.md` (the homepage routes into the four sections below)
- Tutorial: `docs/tutorials.md`, `docs/quickstart.md`
- How-to guide: `docs/how-to.md`, `docs/workflow-examples.md`,
  `docs/troubleshooting.md`, `docs/xarray_migration.md`, `docs/performance.md`,
  `docs/development-guide.md`, `docs/contribution-guide.md`,
  `docs/deployment-guide.md`, `docs/release-process.md`
- Reference: `docs/reference.md`, `docs/algorithm-reference.md`,
  `docs/data_requirements.md`, `docs/error-reference.md`, `docs/deprecations/`,
  `docs/xarray_compatibility.md`,
  `docs/research/nclimgrid-acquisition-and-redistribution.md` (the
  troubleshooting guide sends readers there for source provenance and
  attribution),
  `src/climate_indices/CONTEXT.md`, `VALIDATION.md`
- Explanation: `docs/explanation.md`, `docs/algorithms.md`,
  `docs/wildfire_applications.md`, `docs/flood_applications.md`, `docs/adr/`,
  `docs/ai-assisted-development.md`,
  `docs/algorithm_refs/`, `docs/architecture.md`,
  `docs/project-overview.md`, `docs/floating_point_best_practices.md`
- Internal: `docs/agent/`, `docs/design/`,
  `docs/research/fire-indices-cli-approach.md`,
  `docs/research/interactive-climate-explorer-landscape.md`,
  `docs/research/dri-wrcc-scpdsi-assessment.md`,
  `docs/research/flood-oracle-survey.md`,
  `docs/explorer/`, `docs/architecture-deepening-review-*.md`,
  `docs/ai-assisted-development-report-*.md`,
  `docs/test_fixture_management.md`, `CONTEXT-MAP.md`
- Published URL-retention orphan: `docs/pypi_release.md` (`orphan: true`; kept
  only so the previously published `pypi_release.html` URL resolves)

Build infrastructure (`docs/conf.py`, `docs/Makefile`, `docs/make.bat`,
`docs/_static/`, `docs/_templates/`) is not page content and is outside this
split.

### Markdown authoring conventions

Every page source is a Markdown file under `docs/`, rendered by Sphinx through
MyST-Parser (`docs/conf.py` sets `source_suffix = ".md"`, so an RST file left in
the tree is silently absent from the site). The conventions a page has to
follow to build:

- **Directives** use the fenced form: admonitions as `:::{note}` / `:::{warning}`,
  block directives as ```` ```{list-table} ```` / ```` ```{toctree} ````.
  reStructuredText directives and roles (`.. note::`, `:doc:`) do not render in
  Markdown — they stay as literal text — so a page that still carries one is an
  unfinished conversion.
- **Autodoc** stays inside `{eval-rst}` blocks, because autodoc emits
  reStructuredText and MyST would otherwise render the generated markup
  literally.
- **Cross-links**: `` {doc}`troubleshooting` `` links another published page by
  its source path (no suffix); domain roles such as
  `` {func}`climate_indices.palmer.pdsi` `` resolve through autodoc, so they
  have to name a module path the reference page documents — an unresolved role
  renders as plain code and does not fail the gate. Section links use heading
  anchors (`#some-heading`, provided by `myst_heading_anchors`), and a page
  that is published to the site links repository-root files by absolute GitHub
  URL, since they are not Sphinx source documents.
- **Runnable examples**: examples the docs gate executes are `testsetup`,
  `testcode`, `doctest`, and `testcleanup` blocks; the build runs
  `sphinx-build -b doctest` and fails on a failing example. Shell commands go
  in fenced ```` ```bash ```` blocks, which are not executed.

### Architecture decision records

`docs/adr/` is published, so a reader treats a record as a description of
current code: someone who trusts one greps a symbol and expects to find it.
Every record therefore opens with a `## Status` line naming its standing:

- `Accepted` — the record describes the code as it is. A new record gets this.
- `Amended` — the decision stands, but the record's text needed correcting: a
  symbol was renamed, a module became a package, a deferral completed, or a claim
  the code contradicts. Correct the detail in place and name what moved it — the
  change and its commit or issue — or, when the text was never accurate, the code
  that contradicts it. An amendment corrects names and facts and keeps the
  reasoning that led to the decision legible; it does not rewrite it.
- `Superseded by ADR-NNNN` — a newer record replaces this one in whole. Leave the
  superseded text as written and name the replacing record.

A record whose stale name is corrected without a status is the failure this
convention exists to prevent: a reader cannot tell a past decision from a
present-tense claim. A change that makes an existing record stale updates its
text and its status together.

## Local validation

Run the checks that CI expects before opening a PR:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/ tests/test_type_checking.py
uv run pytest
```

For documentation changes, run the published-docs gate locally:

```bash
uv run --extra docs sphinx-build -E -b html -W --keep-going docs docs/_build/html
uv run --extra docs sphinx-build -E -b doctest docs docs/_build/doctest
```

If your change touches release automation or packaging, also run:

```bash
uv run pytest tests/test_release_integrity.py
```

## Pull request expectations

Before submitting:

- Tests pass locally.
- Formatting, linting, and type checking pass.
- Documentation is updated when behavior or workflows change.
- The PR targets `main`.
- The PR description explains what changed and links related issues.
- No unrelated files are included.
- No AI/tool attribution in commit messages or PR/MR descriptions — no
  `Co-Authored-By: Claude...`, `Claude-Session:`, "Generated with Claude
  Code", or similar. Attribute authorship to the human author only, even if
  a tool's own template or session reminder suggests otherwise.

If a reviewer asks for changes, push follow-up commits to the same branch. You
do not need to close and recreate the PR.

## Releases

Releases are maintainer-owned and tag-based. Contributors should not create or
push release tags unless explicitly approved. See
[`docs/release-process.md`](docs/release-process.md) for the maintainer runbook.

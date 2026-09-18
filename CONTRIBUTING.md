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
the site. A published page stays excluded until it is wired into navigation,
unless another built page links to it as a document — a link to an excluded
page fails the warnings-as-errors build — in which case it builds behind a
hidden toctree until its navigation lands. When a page is added or moves
between audiences, update that exclusion list and this section in the same
change.

Current assignments (the context map and validation status live at the
repository root — `CONTEXT-MAP.md`, `VALIDATION.md` — and the core-library
vocabulary at `src/climate_indices/CONTEXT.md`):

- Tutorial: `docs/quickstart.md`, `docs/index.rst`
- How-to guide: `docs/troubleshooting.md`, `docs/xarray_migration.md`,
  `docs/development-guide.md`, `docs/contribution-guide.md`,
  `docs/deployment-guide.md`, `docs/release-process.md`
- Reference: `docs/reference.md`, `docs/algorithms.md`,
  `docs/deprecations/`, `docs/xarray_compatibility.md`,
  `docs/research/nclimgrid-acquisition-and-redistribution.md` (the
  troubleshooting guide sends readers there for source provenance and
  attribution),
  `src/climate_indices/CONTEXT.md`, `VALIDATION.md`
- Explanation: `docs/wildfire_applications.md`, `docs/adr/`,
  `docs/algorithm_refs/`, `docs/architecture.md`,
  `docs/project-overview.md`, `docs/floating_point_best_practices.md`
- Internal: `docs/agent/`, `docs/design/`,
  `docs/research/fire-indices-cli-approach.md`,
  `docs/research/interactive-climate-explorer-landscape.md`,
  `docs/explorer/`, `docs/architecture-deepening-review-*.md`,
  `docs/test_fixture_management.md`, `CONTEXT-MAP.md`
- Published URL-retention orphan: `docs/pypi_release.md` (`orphan: true`; kept
  only so the previously published `pypi_release.html` URL resolves)

Build infrastructure (`docs/conf.py`, `docs/Makefile`, `docs/make.bat`,
`docs/_static/`, `docs/_templates/`) and assets referenced by no page
(`docs/gallery/*.png`) are not pages and are outside this split.

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

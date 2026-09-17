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

## Local validation

Run the checks that CI expects before opening a PR:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/ tests/test_type_checking.py
uv run pytest
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

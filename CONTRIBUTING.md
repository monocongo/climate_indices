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
6. Merge only after review and passing CI.

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

## Local validation

Run the checks that CI expects before opening a PR:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
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

If a reviewer asks for changes, push follow-up commits to the same branch. You
do not need to close and recreate the PR.

## Releases

Releases are maintainer-owned and tag-based. Contributors should not create or
push release tags unless explicitly approved. See
[`docs/release-process.md`](docs/release-process.md) for the maintainer runbook.

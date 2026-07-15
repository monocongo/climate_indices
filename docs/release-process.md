# Release Process

This is the maintainer runbook for releasing `climate_indices`. Releases are
tag-based from `main`; pushing a valid release tag starts the GitHub Actions
release workflow and publishes to PyPI through Trusted Publishing.

## Release invariants

- `main` is trunk and must be releasable.
- Release tags use exact SemVer format: `vX.Y.Z`.
- Package versions omit the leading `v`: `X.Y.Z`.
- The Git tag, `pyproject.toml`, GitHub Release, and PyPI version must match.
- Tag creation and tag pushes require maintainer approval.
- Long-lived `release/*` branches are avoided except for approved maintenance
  work on older supported versions.

## Pre-release checklist

1. Confirm the release scope and version number.
2. Start from current `main`:
   ```bash
   git switch main
   git pull --ff-only origin main
   ```
3. Create a release-prep branch:
   ```bash
   git switch -c chore/release-X.Y.Z
   ```
4. Update `pyproject.toml` to `X.Y.Z`.
5. Update `CHANGELOG.md` with `## [X.Y.Z] - YYYY-MM-DD`.
6. Update `RELEASE_NOTES.md` or related docs when needed.
7. Run validation locally.
8. Open a PR into `main`.
9. Merge only after review and passing CI.

## Validation commands

Run the normal quality gate:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest
```

Run release integrity checks before pushing a tag:

```bash
uv run pytest tests/test_release_integrity.py
```

Build and inspect package artifacts when preparing the release PR:

```bash
uv run python -m build
uv run twine check dist/*
```

## Tag creation

After the release PR is merged and `main` is green, create the annotated tag
from `main` only after maintainer approval:

```bash
git switch main
git pull --ff-only origin main
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

Verify the tag points at the intended commit:

```bash
git show --stat vX.Y.Z
git status --short
```

Push the tag only after approval:

```bash
git push origin vX.Y.Z
```

Do not force-push, rewrite release history, delete remote tags, or reuse a
published version.

## GitHub Actions release workflow

The release workflow is `.github/workflows/release.yml`.

It runs on tags matching `v*.*.*` and also has a bash regex guard requiring the
exact format `vX.Y.Z`. The workflow:

1. Checks out the tagged commit.
2. Validates the release tag format.
3. Runs linting, formatting checks, type checking, tests, and release integrity
   tests.
4. Runs the security audit.
5. Verifies the tag version equals `pyproject.toml` version.
6. Builds source and wheel artifacts with `python -m build`.
7. Runs `twine check`.
8. Uploads build artifacts to the workflow run.
9. Publishes to PyPI through Trusted Publishing/OIDC.
10. Creates a GitHub Release for the tag with generated release notes and
    attached artifacts.

The publish job uses the `release` environment, so GitHub environment approval
is required before PyPI publication.

## PyPI Trusted Publishing

Publishing uses PyPI Trusted Publishing/OIDC. Maintainers should not add PyPI
API tokens to this repository or to the release workflow.

Expected PyPI project configuration:

- Project name: `climate-indices`
- Owner: `monocongo`
- Repository: `climate_indices`
- Workflow: `release.yml`
- Environment: `release`

If publishing fails at the OIDC step, verify the PyPI trusted publisher settings
and the GitHub environment name before changing workflow credentials.

## Post-release checks

After the workflow completes:

1. Confirm the GitHub Release exists as `vX.Y.Z`.
2. Confirm PyPI has `climate-indices` version `X.Y.Z`.
3. On the live PyPI release page, confirm `Requires-Python` matches the constraint
   in `pyproject.toml` and every supported Python minor appears in the classifiers.
4. Confirm the Python support badge renders the same minimum and maximum versions
   listed in the release's classifiers:
   `https://img.shields.io/badge/Python-3.10--3.14-blue?logo=python`.
5. Install from PyPI in a clean environment if extra verification is needed:
   ```bash
   uv venv /tmp/climate-indices-release-check
   /tmp/climate-indices-release-check/bin/python -m pip install climate-indices==X.Y.Z
   /tmp/climate-indices-release-check/bin/python -c "import climate_indices; print(climate_indices.__version__)"
   ```
6. Open a follow-up PR for any next-cycle changelog preparation if needed.

## Hotfix flow

Use the normal trunk flow for hotfixes whenever possible:

1. Start from updated `main`.
2. Create `hotfix/<topic>`.
3. Make the smallest safe fix with a regression test.
4. Run validation.
5. Merge the PR into `main` after CI passes.
6. Prepare and tag a patch release from `main`.

For an older supported version, a maintainer may approve a maintenance branch.
Keep it narrow, document the target version, and merge forward to `main` when
applicable.

# Release Process

This is the maintainer runbook for releasing `climate_indices`. Releases are
tag-based from `main`; pushing a valid release tag starts the GitHub Actions
release workflow and publishes to PyPI through Trusted Publishing.

## Release invariants

- `main` is trunk and must be releasable.
- Release tags use exact SemVer format: `vX.Y.Z`.
- A release tag's commit must be reachable from `origin/main`.
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
6. Update related docs when needed.
7. Run validation locally.
8. Open a PR into `main`.
9. The maintainer merges after review and passing CI; agents never merge.

## Validation commands

Prepare the development environment and verify that the lockfile is current:

```bash
uv sync --locked --dev
```

Run the normal quality gate in that prepared environment. Every
`uv run --no-sync` command must also include `--no-build` so `uv` uses only the
already prepared environment and does not implicitly build packages:

```bash
uv run --no-sync --no-build ruff check src/ tests/
uv run --no-sync --no-build ruff format --check src/ tests/
uv run --no-sync --no-build mypy src/
uv run --no-sync --no-build pytest
```

Run release integrity checks before pushing a tag:

```bash
uv run --no-sync --no-build pytest tests/test_release_integrity.py
```

Build and inspect package artifacts when preparing the release PR:

```bash
uv run --no-sync --no-build python -m build
uv run --no-sync --no-build twine check dist/*
```

## Release rehearsal (pre-publication smoke test)

The release workflow cannot be rehearsed end to end upstream: `publish` either
publishes or fails, and `create-release` runs only after it. Rehearse everything
up to that boundary on a private copy of the repository, so a defect in the tag
guard, the version match, the test matrix, the build, or the wheel checks
surfaces before the real tag.

This rehearsal is the substitute for a release candidate: the workflow guard
accepts only exact `vX.Y.Z` tags, so there is no `rc` lane to publish.

Run it from the frozen release candidate — the commit the real tag will point
at, with the release-prep PR already merged.

### Rehearsal target

Use one private copy repository, `<owner>/climate_indices-rehearsal`, created
once and reused for every release: push each candidate as its `main`, then push
the release tag. Upstream cannot be forked into its own owner, and a fork under
a second account costs an account switch without proving anything more.

The copy must have Actions enabled, must not be a PyPI trusted publisher, and
must not define a `release` environment protection rule: a required review would
park `publish` in an approval wait, which is not the expected outcome and
evidence of nothing. Never add publishing credentials to the copy.

```bash
gh repo create <owner>/climate_indices-rehearsal --private \
  --description "Release-workflow rehearsal; not a PyPI publisher"
```

### Rehearsal steps

```bash
set -u                            # abort on unset variables
CANDIDATE=<full 40-character sha of the frozen candidate>
TAG=vX.Y.Z                        # the real release tag, not an rc
TARGET=<owner>/climate_indices-rehearsal

git fetch -q origin
git checkout --detach "$CANDIDATE"
SHA="$(git rev-parse HEAD)"
[[ "$SHA" == "$CANDIDATE" ]] || { echo "not at the frozen candidate: $SHA"; exit 1; }
git merge-base --is-ancestor "$SHA" origin/main \
  || { echo "candidate is not on origin/main: $SHA"; exit 1; }
git status --short                # must be empty
```

1. Push the candidate as the copy's `main`, then the copy-only tag. The
   `validate-release-tag` job requires the tagged commit to be reachable from
   the copy's `origin/main`, so `main` lands first. The tag push starts the
   unchanged workflow. The copy holds no unique state, so these pushes force it
   to the new candidate when a rehearsal is repeated:

```bash
: "${SHA:?run the rehearsal setup block in this shell first}"
: "${TAG:?run the rehearsal setup block in this shell first}"
: "${TARGET:?run the rehearsal setup block in this shell first}"

git push --force "https://github.com/$TARGET.git" "${SHA}:refs/heads/main"
git ls-remote --heads "https://github.com/$TARGET.git" main    # prints $SHA

git push --force "https://github.com/$TARGET.git" "${SHA}:refs/tags/$TAG"
git ls-remote --tags "https://github.com/$TARGET.git" "$TAG"   # prints $SHA
```

Never push a rehearsal tag upstream.

2. Watch the run and read every job's conclusion:

```bash
gh run list -R "$TARGET" --workflow=release.yml --limit 3
RUN=<run id>
gh run watch "$RUN" -R "$TARGET" --exit-status   # exits non-zero; expected
gh run view "$RUN" -R "$TARGET" --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"'
```

Expected: `validate-release-tag`, every `test` leg, `security-audit`, `build`,
and both `wheel-check` legs succeed; `publish` fails; `create-release` is
skipped. The `publish` failure must be PyPI rejecting the OIDC exchange because
the copy is not a trusted publisher. A network error, an approval wait, or an
action-resolution failure is a different defect to diagnose and not a pass, and
that rejection must not be remedied by configuring the copy:

```bash
gh run view "$RUN" -R "$TARGET" --log-failed \
  | grep -i -B2 -A6 "server refused the request\|invalid-publisher"
```

3. Collect the artifacts, checksums, and timing:

```bash
gh run download "$RUN" -R "$TARGET" -n dist -D /tmp/rehearsal-dist
ls -l /tmp/rehearsal-dist && shasum -a 256 /tmp/rehearsal-dist/*
gh run view "$RUN" -R "$TARGET" --json createdAt,updatedAt,attempt
```

4. Confirm nothing was published:

```bash
git ls-remote --tags origin "$TAG"     # empty
gh release view "$TAG"                 # not found
curl -s https://pypi.org/pypi/climate-indices/json \
  | python3 -c "import json,sys; print('${TAG#v}' in json.load(sys.stdin)['releases'])"
```

5. Record the evidence on the release ticket — candidate SHA, copy repository,
   run URL, attempt number, every job conclusion, the verbatim PyPI rejection,
   artifact filenames with `sha256`, timing, and the not-published confirmation
   above — before the run logs expire. Reuse or delete the copy afterwards; it
   holds no unique state.

### What a rehearsal does not prove

- `create-release`: it is skipped because `publish` fails first, so GitHub
  Release creation still executes for the first time on the real tag. If it
  fails after a successful publish, the release is complete and the GitHub
  Release can be created manually from the run's `dist` artifacts, within their
  retention window.
- PyPI trusted publishing and the upstream `release` environment: only the real
  repository can validate these. Confirm the trusted publisher (owner,
  repository `climate_indices`, workflow `release.yml`, environment `release`)
  and the environment's required reviewers before tagging.

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

1. Checks out the tagged commit with full history.
2. Validates the release tag format and verifies that its commit is reachable
   from `origin/main`.
3. Runs linting, formatting checks, type checking, tests, and release integrity
   tests against the checked lockfile.
4. Runs the runtime-only security audit against the checked lockfile.
5. Verifies the tag version equals `pyproject.toml` version.
6. Builds source and wheel artifacts with `python -m build` in an unprivileged
   job.
7. Runs `twine check`, installs the wheel in a clean temporary environment, and
   imports the public API from outside the source checkout.
8. Uploads the tested build artifacts to the workflow run.
9. After environment approval, downloads and publishes those artifacts to PyPI
   through Trusted Publishing/OIDC from a publish-only job.
10. Creates a GitHub Release for the tag with generated release notes and
    attached artifacts.

Only the publish job uses the `release` environment and receives
`id-token: write`, so GitHub environment approval is required before PyPI
publication.

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
5. The maintainer merges the PR into `main` after review and passing CI.
6. Prepare and tag a patch release from `main`.

For an older supported version, a maintainer may approve a maintenance branch.
Keep it narrow, document the target version, and merge forward to `main` when
applicable.

# Release Process

This is the maintainer runbook for releasing `climate_indices`. Releases are
tag-based from `main`; pushing a valid release tag starts the GitHub Actions
release workflow and publishes to PyPI through Trusted Publishing.

The Rust crate `climate-indices` has its own release track, described under
*Rust crate release* below. It publishes to crates.io from a separate tag
namespace and never publishes to PyPI.

## Release invariants

- `main` is trunk and must be releasable.
- Release tags use exact SemVer format: `vX.Y.Z`.
- A release tag's commit must be reachable from `origin/main`.
- Package versions omit the leading `v`: `X.Y.Z`.
- The Git tag, `pyproject.toml`, GitHub Release, and PyPI version must match.
- The maintainer creates and pushes release tags; agents never perform these actions.
- Crate tags use their own namespace, `climate-indices-vX.Y.Z`, so neither release
  workflow can start the other.
- Long-lived `release/*` branches are avoided except for approved maintenance
  work on older supported versions.

## Pre-release checklist

1. Confirm the release scope and version number.
2. Fetch current `main`:
   ```bash
   git fetch origin
   ```
3. Create a release-prep branch in its own worktree:
   ```bash
   git worktree add ../climate_indices-release-X.Y.Z -b chore/release-X.Y.Z origin/main
   cd ../climate_indices-release-X.Y.Z
   ```
4. Update `pyproject.toml` to `X.Y.Z` and refresh `uv.lock` if the version changes.
5. Keep the versioned `CHANGELOG.md` entry marked `unreleased` while the date is
   undecided; replace it with `## [X.Y.Z] - YYYY-MM-DD` before freezing the
   candidate. On a tag build `tests/test_release_integrity.py` requires that day to
   be the annotated tag's own tagger date, and any versioned release-notes document
   present to repeat it in its opening paragraph; if the tag slips past the recorded
   day, re-date the record, freeze the candidate again, and rehearse it.
6. Reconcile the changelog, curated release notes, and migration guide with all
   changes selected for this release. Remove pre-publication draft notices from the
   release notes before freezing; refresh story-note metrics only against a named SHA.
7. Run validation locally.
8. Open a PR into `main`.
9. The maintainer merges after review and passing CI; agents never merge.

## Freeze the candidate before tagging

1. Resolve the maintainer-selected correctness and hardening scope, then merge the
   release-prep PR after review and green CI. An old PR's proposed date or candidate
   SHA is historical, not a release instruction. For 3.0.0, reconcile the still-open
   [PR #1266](https://github.com/monocongo/climate_indices/pull/1266) with the final
   release record instead of retaining its October 6 target date.
2. Record the full 40-character SHA of the merged `main` tip and the intended release
   date on the release ticket. Confirm the changelog no longer says `unreleased` and
   the release notes no longer say draft. Hold unrelated merges through the tag push.
3. Validate and rehearse that exact SHA, including every native-wheel and no-Rust
   install leg. Any change to the candidate requires another freeze and rehearsal;
   do not reuse a previous green run for a different commit.
4. The maintainer verifies the upstream `release` environment's required reviewers
   and PyPI Trusted Publishing settings before approving publication. Merely naming
   `environment: release` in YAML does not enforce approval. For 3.0.0, record these
   checks under [#1015](https://github.com/monocongo/climate_indices/issues/1015) and
   rehearsal evidence under [#1014](https://github.com/monocongo/climate_indices/issues/1014).

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
uv run --no-sync --no-build pytest -m validation
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
up to that boundary in a separate rehearsal repository, so a defect in the tag
guard, the version match, the test matrix, the build, or the wheel checks
surfaces before the real tag.

This rehearsal is the substitute for a release candidate: the workflow guard
accepts only exact `vX.Y.Z` tags, so there is no `rc` lane to publish.

Run it from the frozen release candidate — the commit the real tag will point
at, with the release-prep PR already merged.

### Rehearsal target

Rehearsals run in a **rehearsal repository**: a repository that is not
`monocongo/climate_indices`. The default is one copy,
`<owner>/climate_indices-rehearsal`, created once and reused for every release:
push each candidate as its `main`, then push the release tag. A fork under a
different owner is equivalent — upstream cannot be forked into its own owner,
and a fork costs an account switch without proving anything more. Repository
visibility does not change what the rehearsal proves: PyPI matches a trusted
publisher by owner, repository, workflow, and environment, not by visibility, and
the target is not a publisher either way. Keep the copy public where possible —
standard runners are free in public repositories, while a private copy spends the
account's included minutes and bills macOS legs at a 10x multiplier. Make it
private only if the run logs or artifacts must stay non-public.

Whichever target is used, it must have Actions enabled, must not be a PyPI
trusted publisher, and must not define a `release` environment protection rule:
a required review would park `publish` in an approval wait, which is not the
expected outcome and evidences nothing. Never add publishing credentials to the
rehearsal repository.

```bash
# the default public copy; a fork under another owner is equivalent
gh repo create <owner>/climate_indices-rehearsal --public \
  --description "Release-workflow rehearsal; not a PyPI publisher"
```

### Rehearsal steps

```bash
set -u                            # abort on unset variables
CANDIDATE=<full 40-character sha of the frozen candidate>
TAG=vX.Y.Z                        # the real release tag, not an rc
TARGET=<owner>/climate_indices-rehearsal   # or <fork-owner>/climate_indices for an equivalent fork

git fetch -q origin || { echo "cannot fetch origin"; exit 1; }
git checkout --detach "$CANDIDATE"
SHA="$(git rev-parse HEAD)"
[[ "$SHA" == "$CANDIDATE" ]] || { echo "not at the frozen candidate: $SHA"; exit 1; }
git merge-base --is-ancestor "$SHA" origin/main \
  || { echo "candidate is not on origin/main: $SHA"; exit 1; }
git status --short                # must be empty
```

1. Push the candidate as the rehearsal repository's `main`, then its tag. The
   `validate-release-tag` job requires the tagged commit to be reachable from
   the repository's `origin/main`, so `main` lands first. The tag push starts the
   unchanged workflow. The rehearsal repository holds no unique state, so these
   pushes force it to the new candidate when a rehearsal is repeated. Pushing a SHA
   to a tag ref creates a lightweight tag, which has no tagger date: the changelog
   date guard therefore checks only that the record is dated here, and the tag-date
   equality is exercised by the annotated tag production uses:

```bash
: "${SHA:?run the rehearsal setup block in this shell first}"
: "${TAG:?run the rehearsal setup block in this shell first}"
: "${TARGET:?run the rehearsal setup block in this shell first}"

git push --force "https://github.com/$TARGET.git" "${SHA}:refs/heads/main"
git ls-remote --heads "https://github.com/$TARGET.git" main    # prints $SHA

git push --force "https://github.com/$TARGET.git" "${SHA}:refs/tags/$TAG"
git ls-remote --tags "https://github.com/$TARGET.git" "$TAG"   # prints $SHA
```

Never push a rehearsal tag upstream. Re-pushing the same candidate is a no-op,
so no new run starts: use `gh run rerun <run-id> -R "$TARGET"` to retry an identical
rehearsal.

2. Watch the run and read every job's conclusion:

```bash
gh run list -R "$TARGET" --workflow=release.yml --limit 3
RUN=<run id>
gh run watch "$RUN" -R "$TARGET" --exit-status   # exits non-zero; expected
gh run view "$RUN" -R "$TARGET" --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"'
```

Expected: `validate-release-tag`, every `test` leg, `security-audit`, `build`, every
`build-wheels` leg, every `wheel-check` leg, and both `no-rust-install` legs succeed;
`publish` fails; `create-release` is skipped. The `publish` failure must be PyPI rejecting
the OIDC exchange because the rehearsal repository is not a trusted publisher. A network
error, an approval wait, or an action-resolution failure is a different defect to diagnose
and not a pass, and that rejection must not be remedied by configuring the rehearsal
repository:

```bash
gh run view "$RUN" -R "$TARGET" --log-failed \
  | grep -i -B2 -A6 "server refused the request\|invalid-publisher"
```

3. Collect the artifacts, checksums, and timing. Each artifact is downloaded into a
   directory of its own (`dist` for the sdist and pure wheel, `dist-native-<target>` per
   binary wheel), so `gh run download` is called without `-n`:

```bash
gh run download "$RUN" -R "$TARGET" -D /tmp/rehearsal-dist
ls -lR /tmp/rehearsal-dist && shasum -a 256 /tmp/rehearsal-dist/*/*
gh run view "$RUN" -R "$TARGET" --json createdAt,updatedAt,attempt
```

4. Confirm nothing was published:

```bash
git ls-remote --tags origin "$TAG"     # empty
gh release view "$TAG"                 # not found
curl -s https://pypi.org/pypi/climate-indices/json \
  | python3 -c "import json,sys; print('${TAG#v}' in json.load(sys.stdin)['releases'])"
```

5. Record the evidence on the release ticket — candidate SHA, rehearsal
   repository, run URL, attempt number, every job conclusion, the verbatim PyPI
   rejection, artifact filenames with `sha256`, timing, and the not-published
   confirmation above — before the run logs expire. Reuse or delete the rehearsal
   repository afterwards; it holds no unique state.

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
from a clean maintainer worktree only after approval. First confirm that local and
remote `main` still equal the SHA that passed the rehearsal:

```bash
git switch main
git pull --ff-only origin main
CANDIDATE=<full 40-character SHA recorded on the release ticket>
test -z "$(git status --porcelain)" || { echo "working tree is not clean"; exit 1; }
test "$(git rev-parse HEAD)" = "$CANDIDATE" || { echo "local candidate changed"; exit 1; }
test "$(git rev-parse origin/main)" = "$CANDIDATE" || { echo "remote candidate changed"; exit 1; }
git remote get-url --push origin   # must be monocongo/climate_indices, not the rehearsal repo
```

Confirm `vX.Y.Z` does not already exist locally or upstream and `X.Y.Z` is not
already published on PyPI. The tag build checks the dated changelog entry against
the annotated tag's own tagger date, so tag on the recorded release day. Then
create and inspect the annotated tag:

```bash
git tag -a vX.Y.Z "$CANDIDATE" -m "Release vX.Y.Z"
git show --stat vX.Y.Z
git rev-parse 'vX.Y.Z^{commit}'     # must equal $CANDIDATE
```

**Pushing the tag starts the publishing workflow.** Push only this tag after
maintainer approval; do not use `git push --tags`:

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
6. Builds the sdist and the `py3-none-any` wheel with `python -m build`, and the five
   abi3 binary wheels that carry `climate_indices._native` with maturin, in unprivileged
   jobs.
7. Runs `twine check`, installs each platform wheel outside the source checkout on the
   platform it was built for, checks that pip prefers it over the pure wheel, imports the
   public API, and asserts that SPI reaches its Rust kernels.
8. Installs the pure wheel and rebuilds the sdist with `cargo` and `rustc` removed from
   `PATH`, and runs the core suite against the installed package on the oldest and newest
   supported Pythons.
9. Uploads the tested build artifacts to the workflow run.
10. After environment approval, downloads and publishes those artifacts to PyPI through
    Trusted Publishing/OIDC from a publish-only job.
11. Creates a GitHub Release for the tag with generated release notes and attached
    artifacts.

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

This follows the [PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/):
separate build and publish jobs, short-lived OIDC credentials, and explicit
production-environment approval. TestPyPI can be useful for future workflow
changes, but it is not configured here and is not a substitute for this
repository's frozen-candidate rehearsal. The current exact-tag guard rejects
`v3.0.0rc1`; do not try an RC tag as a production dry run.

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
5. Install from PyPI in a fresh environment outside the checkout:
   ```bash
   CHECK_DIR="$(mktemp -d)"
   uv venv "$CHECK_DIR/venv"
   uv pip install --python "$CHECK_DIR/venv/bin/python" climate-indices==X.Y.Z
   (cd "$CHECK_DIR" && venv/bin/python -c "import climate_indices; print(climate_indices.__version__)")
   ```
   On a supported native-wheel platform, also confirm `climate_indices._native`
   imports; other platforms should retain the documented pure-Python path.
6. The workflow generates GitHub Release notes from merged PRs, not from
   `docs/release-notes-3.0.0.md`. The maintainer should add the curated highlights,
   migration warnings, and a permalink to the tagged release-notes document;
   convert relative documentation links to absolute URLs when copying text.
7. Open a follow-up PR for any next-cycle changelog preparation if needed.

## If publication fails

A failed workflow is not proof that nothing reached PyPI: an upload can partially
succeed. Inspect the publish log and PyPI file list before retrying. If no artifacts
were published, repair external setup and retry the unchanged run. If a fix changes
the candidate, use a new version and tag after freezing and rehearsing it; never
retarget the failed tag. If any artifacts were published, never replace their
contents or retarget that tag; assess whether the
unchanged run can finish uploading missing artifacts, otherwise prepare a patch
version from `main`. If only GitHub Release creation failed, recover that step
using the already-published run's artifacts, not a rebuilt package.

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

## Rust crate release

The crate `climate-indices` (`crates/climate-indices`) publishes to crates.io on
its own `0.x` track, independently of the Python release above. See
[ADR-0021](adr/0021-climate-indices-crate-identity.md) for the identity, the
version scheme, and the tag namespace. Publishing runs from
`.github/workflows/crate-release.yml` through crates.io Trusted Publishing/OIDC,
so this repository stores no crates.io API token.

Only the maintainer releases the crate. Agents never create a crate tag, push
one, or approve the `release` environment.

### Crate release invariants

- Crate tags are annotated tags named `climate-indices-vX.Y.Z`. The optional
  pre-release suffix (`climate-indices-vX.Y.Z-rc.1`) is the rehearsal lane. Build
  metadata (`+`) is rejected, because cargo normalizes it and the tag and the
  published version would disagree.
- The tag version, the `version` in `crates/climate-indices/Cargo.toml`, and the
  crate changelog heading must all name the same version. The workflow checks all
  three before it publishes anything.
- The crate version and the crate changelog are the crate's own.
  `crates/climate-indices/CHANGELOG.md` holds the crate history, and it does not
  follow `pyproject.toml` or the root `CHANGELOG.md`.
- A crates.io version is permanent. A version is used once, and a bad release is
  yanked rather than replaced.
- `climate-py` stays `publish = false`. Only the crate is published.

### Crate one-time setup

Confirm all three before the first crate release:

1. On crates.io, the `climate-indices` crate has a trusted publisher with owner
   `monocongo`, repository `climate_indices`, workflow `crate-release.yml`, and
   environment `release`. The crate already exists on crates.io, so no bootstrap
   token is needed to register the publisher.
2. The GitHub `release` environment still requires a reviewer. The crate publish
   job uses that same environment, so a crate release needs the same approval as
   the PyPI release.
3. This repository has no `CARGO_REGISTRY_TOKEN` secret. The publish job exchanges
   the workflow's OIDC identity for a short-lived token, and
   `crates-io-auth-action` revokes that token when the job ends.

### Crate release steps

1. Bump `version` in `crates/climate-indices/Cargo.toml`.
2. Retitle the `## Unreleased` heading in
   `crates/climate-indices/CHANGELOG.md` to `## X.Y.Z - YYYY-MM-DD`, using the
   intended tag day.
3. Open a PR into `main`, then let the maintainer merge it after review and
   passing CI.
4. Record the full 40-character SHA of the merged `main` tip.
5. Run the checks in *Tag creation* above with the crate tag name, then create
   the annotated tag:
   ```bash
   git tag -a climate-indices-vX.Y.Z "$CANDIDATE" -m "Release climate-indices vX.Y.Z"
   git rev-parse 'climate-indices-vX.Y.Z^{commit}'   # must equal $CANDIDATE
   ```
6. Confirm the version is absent from crates.io, then push only that tag:
   ```bash
   curl -s https://crates.io/api/v1/crates/climate-indices/versions \
     | python3 -c "import json,sys; print([v['num'] for v in json.load(sys.stdin)['versions']])"
   git push origin climate-indices-vX.Y.Z
   ```
7. Approve the `release` environment when the run waits at `publish-crate`.
8. Read every job's conclusion and run the crate post-release checks below.

### The crate release workflow

`.github/workflows/crate-release.yml` runs on crate tags only, and it:

1. validates the tag format, the reachability of the tagged commit from
   `origin/main`, the tag version against the crate manifest, and the dated crate
   changelog heading;
2. runs the four crate quality gates on the tagged commit: `crate-gate`,
   `crate-msrv`, `crate-semver`, and `crate-deny`;
3. after environment approval, exchanges the OIDC identity for a short-lived
   crates.io token in the publish-only job and runs `cargo publish`.

It creates no GitHub Release and it does not touch PyPI. Only the publish job
receives `id-token: write`, and every job checks out the immutable event SHA, so
a tag moved during a run cannot publish unvalidated commits.

### Crate rehearsal against a pre-release version

crates.io has no unpublish, so rehearse the path with a pre-release version that
the final release supersedes. The machinery is the same one the real tag uses.
Rehearse the packaging and verification steps locally first:

```bash
cargo publish --dry-run --locked -p climate-indices
```

Then rehearse the published path:

1. Pick an unused pre-release version, for example `0.1.0-rc.1`, and confirm
   crates.io does not list it (the command in *Crate release steps* above).
2. On a `chore/` branch, set that version in
   `crates/climate-indices/Cargo.toml` and retitle the crate changelog heading to
   `## 0.1.0-rc.1 - YYYY-MM-DD`. Merge the branch into `main`.
3. Tag the merged SHA `climate-indices-v0.1.0-rc.1`, and push that tag.
4. Approve the `release` environment, then confirm that every gate job passed and
   that `publish-crate` published. Record the run URL, every job conclusion, and
   the timing.
5. Confirm a scratch crate can depend on the release. Cargo does not select a
   pre-release version without an explicit requirement:
   ```bash
   cargo new /tmp/crate-rehearsal
   cd /tmp/crate-rehearsal
   cargo add climate-indices --vers 0.1.0-rc.1
   cargo build
   ```
6. Record the evidence on the release ticket.
7. For the real release, set the version and the changelog heading to `0.1.0`,
   then tag `climate-indices-v0.1.0` as in *Crate release steps* above.

A rehearsal spends the version number. crates.io allows a yank, not a delete, so
the pre-release stays published even when the rehearsal fails.

### If a crate publication fails

- **crates.io rejects the OIDC exchange.** The trusted publisher on crates.io does
  not match this repository, the workflow file name, or the environment name.
  Correct the crates.io configuration. Do not add an API token to the repository.
- **`cargo publish` reports that the version is already uploaded.** A crates.io
  version is permanent. Do not retag: prepare the next version from `main`.
- **A gate job fails.** Fix `main`, then tag a new version. Never move a pushed
  tag.
- **The publish state is unknown.** `cargo publish` creates a version or fails; it
  does not leave a partial upload. Check crates.io with the version-list command
  in *Crate release steps* above. If the version is listed, the publish succeeded
  and the post-release checks apply. If it is not listed, fix the cause and tag a
  new version.
- **The run waits at `publish-crate`.** The `release` environment requires a
  reviewer, so a maintainer must approve the deployment.

### Crate post-release checks

1. Confirm that crates.io lists the version and that the crate page shows the
   README and the license.
2. Confirm that docs.rs built the version without warnings. The build starts a few
   minutes after the publish, and a warning there is a defect to fix in the next
   version.
3. Install the release in a scratch crate outside this checkout, as in the
   rehearsal, and call one kernel:
   ```bash
   cargo add climate-indices@X.Y.Z
   cargo build
   ```
4. Confirm that the crate changelog heading carries the release day.
5. Add the crates.io link to the next release notes or the README when the
   maintainer asks for it.

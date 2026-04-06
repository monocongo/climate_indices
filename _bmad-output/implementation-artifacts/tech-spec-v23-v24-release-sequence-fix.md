---
title: 'v2.3.0 → v2.4.0 Release Sequencing Fix'
slug: 'v23-v24-release-sequence-fix'
created: '2026-04-06'
status: 'ready-for-dev'
stepsCompleted: [1, 2, 3, 4]
tech_stack: ['git', 'GitHub Actions', 'PyPI OIDC trusted publishing', 'hatchling']
files_to_modify:
  - CHANGELOG.md  # pr-614 branch: change [Unreleased] → [2.3.0] - 2026-02-11
  - pyproject.toml  # pr-623 branch only: keep version = '2.4.0' during conflict resolution
  - tests/test_release_integrity.py  # new — release integrity guardrails
code_patterns: []
test_patterns: []
---

# Tech-Spec: v2.3.0 → v2.4.0 Release Sequencing Fix

**Created:** 2026-04-06

## Overview

### Problem Statement

v2.4.0 was developed on top of an unreleased v2.3.0 branch. Neither version has been published
to PyPI and `master` reflects neither release. The release history is out of sequence: the GitHub
PR for v2.3.0 (#614) and v2.4.0 (#623) both exist but have never been merged to `master` or
tagged. Additionally, the CHANGELOG in PR #614 still carries an `[Unreleased]` header instead of
the proper `[2.3.0] - 2026-02-11` entry.

### Solution

Correct the release sequence end-to-end:

1. Fix the CHANGELOG in the `pr-614` branch (`[Unreleased]` → `[2.3.0] - 2026-02-11`).
2. Verify PyPI OIDC trusted publishing is configured for `monocongo/climate_indices`.
3. Merge PR #614 → push tag `v2.3.0` → CI publishes to PyPI.
4. Resolve merge conflicts in PR #623 locally (both branches diverge from the same master commit,
   so `pyproject.toml` and `CHANGELOG.md` will conflict), force-push the branch.
5. Merge PR #623 → push tag `v2.4.0` → CI publishes to PyPI.

### Scope

**In Scope:**
- Fix `CHANGELOG.md` header in `pr-614` branch (`[Unreleased]` → `[2.3.0] - 2026-02-11`)
- Verify/configure PyPI OIDC trusted publishing for the `release` GitHub environment
- Merge PR #614 to `master` and push tag `v2.3.0`
- Verify v2.3.0 appears on PyPI
- Resolve merge conflicts in PR #623 (`pyproject.toml`, `CHANGELOG.md`) locally and force-push
- Merge PR #623 to `master` and push tag `v2.4.0`
- Verify v2.4.0 appears on PyPI

**Out of Scope:**
- New feature or bug-fix work
- Changes to the GitHub Actions release pipeline itself
- Any documentation beyond the CHANGELOG fix in PR #614

---

## Context for Development

### Codebase Patterns

- **Release trigger**: `.github/workflows/release.yml` fires on any tag matching `v*` pushed to
  `master`. It validates that the tag version matches `pyproject.toml`, builds with `hatchling`,
  then publishes to PyPI via OIDC (no stored token). Publish step requires manual approval via
  GitHub environment `release`.
- **Branch lineage**: `pr-614` and `feature/v2.4.0-planning` (PR #623) both diverge from
  `master` at commit `03a2fe6` — they are parallel branches, not stacked.
- **CHANGELOG format**: Keep a Changelog (keepachangelog.com). Entries ordered newest-first.
  Each release block is `## [X.Y.Z] - YYYY-MM-DD`.
- **Version location**: single source of truth is `pyproject.toml` line `version = 'X.Y.Z'` under
  `[project]`. Build backend is `hatchling`.

### Files to Reference

| File | Purpose |
| ---- | ------- |
| `.github/workflows/release.yml` | Release pipeline — tag trigger, version validation, PyPI publish |
| `pyproject.toml` | Version source of truth (`version = 'X.Y.Z'` under `[project]`) |
| `CHANGELOG.md` | Release history — header fix required in pr-614 branch |
| `RELEASE_NOTES.md` | Full v2.3.0 release notes already written — reference for CHANGELOG content |
| `docs/pypi_release_guide.md` | Step-by-step release guide (existing maintainer docs) |
| `.worktrees/pr614-merge-fix/` | Existing worktree already checked out on `pr-614` branch |

### Technical Decisions

- **CHANGELOG date for v2.3.0**: Use `2026-02-11` (date of the `chore(release): bump version to
  v2.3.0` commit on `release/v2.3.0`).
- **Conflict resolution strategy for PR #623**: Keep `version = '2.4.0'` in `pyproject.toml`;
  in `CHANGELOG.md` keep both `[2.4.0]` and `[2.3.0]` blocks with `[2.3.0]` beneath `[2.4.0]`
  (newest-first order).
- **Force-push**: Acceptable here because PR #623 is a pre-merge branch under sole authorship
  (James/monocongo).
- **OIDC trusted publishing**: PyPI project name is `climate-indices`. The publisher must be
  registered at pypi.org under the project's publishing settings with:
  - Owner: `monocongo`
  - Repository: `climate_indices`
  - Workflow filename: `release.yml`
  - Environment name: `release`
- **Release workflow gate ordering**: The `release` job in `.github/workflows/release.yml`
  requires `needs: [test, security-audit]` and `environment: release` (manual approval). The
  version-consistency check (`TAG_VERSION == PYPROJECT_VERSION`) runs **inside** the `release`
  job — meaning it runs AFTER the approval is granted. Run the local pre-check (Task 11a) before
  pushing the tag to avoid burning an approval on a version mismatch.
- **`RELEASE_NOTES.md` pre-exists**: The `pr-614` branch already contains a detailed
  `RELEASE_NOTES.md` with all v2.3.0 content. The CHANGELOG fix in Task 2 is a one-line header
  change only — no content authoring required.
- **Existing worktree**: `.worktrees/pr614-merge-fix/` is already checked out on `pr-614`. Work
  on the CHANGELOG fix there directly instead of creating a new checkout.

---

## Implementation Plan

### Prerequisites

**Before starting any phase**, confirm the following. Failure to verify these upfront will cause
cryptic errors mid-release with no clean rollback path.

- **Authentication**: You must be logged in as `monocongo` on both GitHub and PyPI. Tasks 9, 14,
  and 29 require GitHub repo admin access; Tasks 5–7 require PyPI project owner access. A
  collaborator without admin cannot approve the `release` environment deployment.
- **Branch protection**: Check that `master` does not have status-check requirements that would
  block the merges in Tasks 9 and 24. Go to **Settings → Branches → master** and confirm the
  branch protection rules. If required status checks are listed, ensure all relevant CI jobs pass
  on each PR before merging. If protection requires a second approver and you are the sole
  maintainer, you may need to temporarily adjust the rule.
- **Python 3.11+ available locally**: Tasks 12 and 27 use `tomllib` (stdlib since 3.11). Confirm
  with `python --version` before running those checks.
- **Release integrity tests pass**: Run the always-on guardrails before touching any branch:
  ```
  uv run pytest tests/test_release_integrity.py -m "not release" -v
  ```
  All 4 tests must pass. If they fail, fix the underlying issue before proceeding.

### Tasks

**Phase 1 — Fix PR #614**

1. The worktree `.worktrees/pr614-merge-fix/` is already checked out on `pr-614`. Work there
   directly — no `git checkout` needed:
   ```
   cd .worktrees/pr614-merge-fix
   ```

2. In `CHANGELOG.md`, change the header from:
   ```
   ## [Unreleased]
   ```
   to:
   ```
   ## [2.3.0] - 2026-02-11
   ```

3. Stage and commit:
   ```
   git add CHANGELOG.md
   git commit -m "docs(release): finalize v2.3.0 changelog header"
   ```

4. Confirm the tracking branch before pushing — the worktree must track `pr-614`, not an
   unrelated ref:
   ```
   git branch -vv
   ```
   The output must show `* pr-614 ... [origin/pr-614]`. Then push:
   ```
   git push origin pr-614
   ```

**Phase 2 — Verify PyPI OIDC Trusted Publishing and GitHub Environment**

5. Navigate to https://pypi.org/manage/project/climate-indices/settings/publishing/
   (log in as `monocongo`).

6. Check if a trusted publisher entry exists for:
   - Owner: `monocongo`
   - Repository: `climate_indices`
   - Workflow filename: `release.yml`
   - Environment: `release`

7. If no entry exists, click "Add a new publisher" and fill in the values above. Save.

8. In the GitHub repo, go to **Settings → Environments → release**. Confirm that
   "Required reviewers" is either empty or includes `monocongo` so the sole maintainer can
   self-approve the deployment. If not configured, add `monocongo` as a required reviewer.
   This must be done **before pushing any tags** — the approval gate is hit mid-pipeline and
   cannot be reconfigured while a workflow is pending.

**Phase 3 — Merge PR #614 and release v2.3.0**

9. Before merging, confirm all CI checks are green on the `pr-614` branch. Go to the PR on
   GitHub and verify the status checks section shows all jobs passing. If any job is failing,
   resolve it before merging — do not proceed with a failing-CI PR.
   Then approve and **squash merge** PR #614 — keeps `master` history clean for a
   single-purpose release-prep branch.

10. Return to the repo root (not the worktree) and pull `master`:
    ```
    cd /path/to/climate_indices   # repo root, not .worktrees/pr614-merge-fix
    git checkout master
    git pull origin master
    ```

11. Confirm `pyproject.toml` on `master` shows `version = '2.3.0'` and `CHANGELOG.md` shows
    `## [2.3.0] - 2026-02-11` as the top release entry. **Do not proceed to Task 12 until
    both are verified** — pushing the tag with `[Unreleased]` in the CHANGELOG will publish
    incorrect release notes to PyPI.

12. Run local pre-tag version consistency check before pushing:
    ```
    python -c "
    import tomllib
    v = tomllib.load(open('pyproject.toml', 'rb'))['project']['version']
    assert v == '2.3.0', f'Version mismatch: pyproject.toml has {v}'
    print(f'OK: pyproject.toml version = {v}')
    "
    ```
    Only proceed to Task 13 if this prints `OK: pyproject.toml version = 2.3.0`. This mirrors
    the CI check and avoids burning a manual approval on a version mismatch (the CI validates
    version AFTER the approval gate fires).
    **If the check fails**: do NOT push the tag. Run `grep ^version pyproject.toml` to confirm
    the actual value. If the squash merge introduced the wrong version, make a fixup commit on
    master: `git commit -am "fix: correct pyproject.toml version to 2.3.0"` then re-run the
    check. Do not push the tag until the check prints `OK`.

13. Push the release tag:
    ```
    git tag v2.3.0
    git push origin v2.3.0
    ```

14. Approve the publish step in GitHub Actions:
    - Go to the repo **Actions** tab
    - Click the `Release to PyPI` workflow run triggered by the `v2.3.0` tag
    - Wait for `test` and `security-audit` jobs to pass (green)
    - Click **"Review deployments"** on the `release` job
    - Check the `release` environment and click **"Approve and deploy"**

15. Verify on https://pypi.org/project/climate-indices/ that version `2.3.0` is now listed.

16. Clean up the pr-614 worktree now that the branch is merged and the remote ref will be
    deleted:
    ```
    git worktree remove .worktrees/pr614-merge-fix
    ```
    If git refuses (untracked or modified files remain), inspect and resolve before removing.
    Confirm with `git worktree list` — `.worktrees/pr614-merge-fix` should no longer appear.

**Phase 4 — Resolve conflicts in PR #623 and release v2.4.0**

17. Return to repo root if needed, then check out `feature/v2.4.0-planning`:
    ```
    cd /path/to/climate_indices   # ensure you are at repo root, not inside a worktree
    git checkout feature/v2.4.0-planning
    ```

18. Merge `master` into the branch to surface conflicts:
    ```
    git merge master
    ```

19. Resolve `pyproject.toml` conflict: keep `version = '2.4.0'`.

20. Resolve `CHANGELOG.md` conflict: ensure the file contains BOTH release blocks in
    newest-first order:
    ```
    ## [2.4.0] - 2026-04-05
    ... (existing v2.4.0 content) ...

    ## [2.3.0] - 2026-02-11
    ... (v2.3.0 content from pr-614) ...

    ## [2.2.0] - 2025-08-03
    ... (existing older content) ...
    ```
    The v2.3.0 block content begins immediately after the `## [2.3.0] - 2026-02-11` header and
    spans the `### Added`, `### Changed`, and `### Removed` subsections. The authoritative source
    is the `pr-614` branch CHANGELOG — locate this content by searching for the `## [2.3.0]`
    header in `.worktrees/pr614-merge-fix/CHANGELOG.md` (or `git show pr-614:CHANGELOG.md`)
    and copy it verbatim. Do not paraphrase or reconstruct from memory; use the source directly.
    If any conflicts appear in files other than `pyproject.toml` and `CHANGELOG.md`, stop and
    investigate — they are unexpected given that `pr-614` only touched docs and config.

21. Complete the merge commit:
    ```
    git add pyproject.toml CHANGELOG.md
    git merge --continue
    ```
    (commit message: `chore: merge master post-v2.3.0 release into v2.4.0 branch`)

22. Force-push the resolved branch:
    ```
    git push --force-with-lease origin feature/v2.4.0-planning
    ```
    Verify the push succeeded by confirming the conflict resolution commit is now on the remote:
    ```
    git log --oneline origin/feature/v2.4.0-planning -3
    ```
    The merge commit (`chore: merge master post-v2.3.0 release into v2.4.0 branch`) must be
    the top entry. If `--force-with-lease` rejected the push, run `git fetch origin` to sync
    the remote ref, then retry.

23. On GitHub, confirm PR #623 shows no conflicts and CI passes.

24. Approve and **merge commit** (not squash) PR #623 to `master` — preserves the full
    feature branch history for the 30+ commits that make up v2.4.0.

25. Pull updated `master`:
    ```
    git checkout master
    git pull origin master
    ```

26. Confirm `pyproject.toml` shows `version = '2.4.0'` and `CHANGELOG.md` shows both v2.4.0 and
    v2.3.0 entries.

27. Run local pre-tag version consistency check:
    ```
    python -c "
    import tomllib
    v = tomllib.load(open('pyproject.toml', 'rb'))['project']['version']
    assert v == '2.4.0', f'Version mismatch: pyproject.toml has {v}'
    print(f'OK: pyproject.toml version = {v}')
    "
    ```
    Only proceed to Task 28 if this prints `OK: pyproject.toml version = 2.4.0`.
    **If the check fails**: do NOT push the tag. Run `grep ^version pyproject.toml` to confirm.
    If the conflict resolution in Task 19 accidentally kept `version = '2.3.0'`, fix it with a
    direct commit on master: `git commit -am "fix: correct pyproject.toml version to 2.4.0"`
    then re-run the check.

28. Push the release tag:
    ```
    git tag v2.4.0
    git push origin v2.4.0
    ```

29. Approve the publish step in GitHub Actions:
    - Go to the repo **Actions** tab
    - Click the `Release to PyPI` workflow run triggered by the `v2.4.0` tag
    - Wait for `test` and `security-audit` jobs to pass (green)
    - Click **"Review deployments"** on the `release` job
    - Check the `release` environment and click **"Approve and deploy"**

30. Verify on https://pypi.org/project/climate-indices/ that version `2.4.0` is now the latest.

### Acceptance Criteria

**AC-0 — Pre-tag gate: CHANGELOG and workflow verified before any tag is pushed**
- Given: PR #614 has been merged to `master` and Phase 2 tasks (5–8) are complete
- When: the following guardrail commands are run from repo root on `master`:
  ```
  uv run pytest tests/test_release_integrity.py -m "not release" -v
  uv run pytest tests/test_release_integrity.py -m release -v
  ```
- Then: all tests pass; if `test_changelog_has_no_unreleased_block` or
  `test_changelog_top_entry_matches_pyproject_version` fail, fix the CHANGELOG before proceeding;
  if `test_release_workflow_uses_oidc_not_token` or `test_release_workflow_has_environment_gate`
  fail, fix the workflow or PyPI/GitHub environment config before pushing any tag

**AC-1 — v2.3.0 CHANGELOG header fixed**
- Given: the `pr-614` worktree (`.worktrees/pr614-merge-fix/`) is the working directory
- When: `CHANGELOG.md` is opened
- Then: the first release block reads `## [2.3.0] - 2026-02-11` (not `[Unreleased]`)

**AC-1b — Pre-tag local check passes for v2.3.0**
- Given: `master` is checked out at repo root after merging PR #614
- When: the Task 12 python one-liner is run
- Then: output is exactly `OK: pyproject.toml version = 2.3.0` with exit code 0

**AC-2 — PyPI OIDC publisher configured**
- Given: logged in to pypi.org as `monocongo`
- When: visiting the `climate-indices` project publishing settings
- Then: a trusted publisher entry exists for `monocongo/climate_indices`, workflow `release.yml`,
  environment `release`

**AC-3 — v2.3.0 on PyPI**
- Given: PR #614 is merged and tag `v2.3.0` is pushed
- When: the GitHub Actions release workflow completes
- Then: `pip install climate-indices==2.3.0` succeeds (after ~60s propagation delay)

**AC-4 — PR #623 conflict-free**
- Given: `master` contains the v2.3.0 merge
- When: `feature/v2.4.0-planning` is merged with `master` and conflicts resolved
- Then: PR #623 on GitHub shows no merge conflicts and CI is green

**AC-4b — Pre-tag local check passes for v2.4.0**
- Given: `master` is checked out at repo root after merging PR #623
- When: the Task 27 python one-liner is run
- Then: output is exactly `OK: pyproject.toml version = 2.4.0` with exit code 0

**AC-5 — v2.4.0 on PyPI**
- Given: PR #623 is merged and tag `v2.4.0` is pushed
- When: the GitHub Actions release workflow completes
- Then: `pip install climate-indices==2.4.0` succeeds and is the latest version on PyPI

---

## Additional Context

### Dependencies

- PR #614 must be fully merged and tag `v2.3.0` pushed **before** working on PR #623.
- The PyPI OIDC publisher must be configured before pushing either tag, or the publish step
  will fail (the CI build/test steps will still pass).

### Testing Strategy

**Pre-release (run before pushing each tag):**
```
uv run pytest tests/test_release_integrity.py -m release -v
```
All 4 release-time tests must pass. Fix any failures before tagging.

**Post-publish (run after each GitHub Actions release workflow completes):**

Create a clean virtualenv to avoid interference from the local editable install:
```
python3.11 -m venv /tmp/climate-release-verify
source /tmp/climate-release-verify/bin/activate
```

Then verify v2.3.0:
```
pip install climate-indices==2.3.0
python -c "import climate_indices; print(climate_indices.__version__)"
python -c "from climate_indices import spi; print('SPI import OK')"
deactivate && rm -rf /tmp/climate-release-verify
```

Repeat for v2.4.0, adding the EDDI check:
```
python3.11 -m venv /tmp/climate-release-verify
source /tmp/climate-release-verify/bin/activate
pip install climate-indices==2.4.0
python -c "import climate_indices; print(climate_indices.__version__)"
python -c "from climate_indices import spi, eddi; print('v2.4.0 API OK')"
deactivate && rm -rf /tmp/climate-release-verify
```

**PyPI propagation**: wait ~60 seconds after the GitHub Actions publish step shows "success"
before running `pip install` — the PyPI CDN may 404 briefly. If the first attempt fails with
"No matching distribution found", wait 30 seconds and retry up to 3 times before treating it
as a real failure. After 3 failed retries, check the PyPI release page directly at
https://pypi.org/project/climate-indices/#history to confirm the release was uploaded.

### Notes

- `--force-with-lease` is preferred over `--force` for the PR #623 push — it will fail-safe if
  anyone else has pushed to the branch since your last fetch. If `--force-with-lease` rejects the
  push, run `git fetch origin` to sync the remote ref, then retry the push.
- **PyPI publish failure rollback**: If the GitHub Actions publish step fails after manual
  approval was already granted:
  1. Check the Actions log for the specific error (auth failure, upload error, version conflict).
  2. If the upload partially succeeded (check https://pypi.org/project/climate-indices/#history),
     contact PyPI support at https://pypi.org/help/#admin before re-uploading — you cannot
     re-upload a file for the same version.
  3. If no upload occurred, delete the tag locally and on the remote, fix the root cause, then
     re-tag and push: `git tag -d v2.3.0 && git push origin :refs/tags/v2.3.0`
  4. Re-push the corrected tag to re-trigger the pipeline.
- **Worktree cleanup (Task 16)**: If `git worktree remove` refuses due to untracked files, decide
  whether they are ephemeral (delete them) or important (move them out first). Stashing does not
  apply to untracked files — use `git clean -n` to preview what would be removed, then
  `git clean -fd` to remove them, before retrying `git worktree remove`.
- If OIDC trusted publishing is not available as a fallback, add a PyPI API token as GitHub
  Actions secret `PYPI_API_TOKEN` and adjust the publish step to use
  `password: ${{ secrets.PYPI_API_TOKEN }}`. OIDC is strongly preferred (no token rotation needed).

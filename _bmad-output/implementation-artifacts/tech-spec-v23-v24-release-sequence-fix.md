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

9. On GitHub, approve and **squash merge** PR #614 — keeps `master` history clean for a
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

**Phase 4 — Resolve conflicts in PR #623 and release v2.4.0**

16. Return to repo root if needed, then check out `feature/v2.4.0-planning`:
    ```
    cd /path/to/climate_indices   # ensure you are at repo root, not inside a worktree
    git checkout feature/v2.4.0-planning
    ```

17. Merge `master` into the branch to surface conflicts:
    ```
    git merge master
    ```

18. Resolve `pyproject.toml` conflict: keep `version = '2.4.0'`.

19. Resolve `CHANGELOG.md` conflict: ensure the file contains BOTH release blocks in
    newest-first order:
    ```
    ## [2.4.0] - 2026-04-05
    ... (existing v2.4.0 content) ...

    ## [2.3.0] - 2026-02-11
    ... (v2.3.0 content from pr-614) ...

    ## [2.2.0] - 2025-08-03
    ... (existing older content) ...
    ```
    If any conflicts appear in files other than `pyproject.toml` and `CHANGELOG.md`, stop and
    investigate — they are unexpected given that `pr-614` only touched docs and config.

20. Complete the merge commit:
    ```
    git add pyproject.toml CHANGELOG.md
    git merge --continue
    ```
    (commit message: `chore: merge master post-v2.3.0 release into v2.4.0 branch`)

21. Force-push the resolved branch:
    ```
    git push --force-with-lease origin feature/v2.4.0-planning
    ```

22. On GitHub, confirm PR #623 shows no conflicts and CI passes.

23. Approve and **merge commit** (not squash) PR #623 to `master` — preserves the full
    feature branch history for the 30+ commits that make up v2.4.0.

24. Pull updated `master`:
    ```
    git checkout master
    git pull origin master
    ```

25. Confirm `pyproject.toml` shows `version = '2.4.0'` and `CHANGELOG.md` shows both v2.4.0 and
    v2.3.0 entries.

26. Run local pre-tag version consistency check:
    ```
    python -c "
    import tomllib
    v = tomllib.load(open('pyproject.toml', 'rb'))['project']['version']
    assert v == '2.4.0', f'Version mismatch: pyproject.toml has {v}'
    print(f'OK: pyproject.toml version = {v}')
    "
    ```
    Only proceed to Task 27 if this prints `OK: pyproject.toml version = 2.4.0`.

27. Push the release tag:
    ```
    git tag v2.4.0
    git push origin v2.4.0
    ```

28. Approve the publish step in GitHub Actions:
    - Go to the repo **Actions** tab
    - Click the `Release to PyPI` workflow run triggered by the `v2.4.0` tag
    - Wait for `test` and `security-audit` jobs to pass (green)
    - Click **"Review deployments"** on the `release` job
    - Check the `release` environment and click **"Approve and deploy"**

29. Verify on https://pypi.org/project/climate-indices/ that version `2.4.0` is now the latest.

### Acceptance Criteria

**AC-0 — Pre-tag gate: CHANGELOG verified on master before any tag is pushed**
- Given: PR #614 has been merged to `master`
- When: `CHANGELOG.md` is inspected on `master`
- Then: the first release block reads `## [2.3.0] - 2026-02-11` (not `[Unreleased]`);
  if this check fails, do not push the tag — fix the CHANGELOG and amend the merge

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
- When: the Task 26 python one-liner is run
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

- After each PyPI publish, run in a clean virtualenv (Python 3.11 as reference):
  ```
  pip install climate-indices==2.3.0
  python -c "import climate_indices; print(climate_indices.__version__)"
  ```
  (repeat for `2.4.0`)
- Allow ~60 seconds after the GitHub Actions publish step completes before running the
  `pip install` verification — PyPI CDN propagation means the version may 404 briefly.
- Confirm GitHub Actions workflow run shows all steps green including the publish step.

### Notes

- `--force-with-lease` is preferred over `--force` for the PR #623 push — it will fail-safe if
  anyone else has pushed to the branch since your last fetch. If `--force-with-lease` rejects the
  push, run `git fetch origin` to sync the remote ref, then retry the push.
- If the OIDC publisher is not yet set up on PyPI, you can use a PyPI API token as a fallback:
  add it as a GitHub Actions secret `PYPI_API_TOKEN` and adjust the publish step to use
  `password: ${{ secrets.PYPI_API_TOKEN }}`. However, OIDC is preferred (no token rotation needed).

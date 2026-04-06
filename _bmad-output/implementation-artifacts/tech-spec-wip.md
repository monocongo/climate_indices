---
title: 'v2.3.0 → v2.4.0 Release Sequencing Fix'
slug: 'v23-v24-release-sequence-fix'
created: '2026-04-06'
status: 'in-progress'
stepsCompleted: [1]
tech_stack: ['git', 'GitHub Actions', 'PyPI OIDC trusted publishing', 'hatchling']
files_to_modify:
  - src/climate_indices/  # pr-614 branch only (CHANGELOG fix)
  - CHANGELOG.md
  - pyproject.toml
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
| `pyproject.toml` | Version source of truth |
| `CHANGELOG.md` | Release history — needs `[Unreleased]` fixed in pr-614 |
| `docs/pypi_release_guide.md` | Step-by-step release guide (existing docs) |

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

---

## Implementation Plan

### Tasks

**Phase 1 — Fix PR #614**

1. Check out the `pr-614` branch locally.
   ```
   git checkout pr-614
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

4. Push to the remote PR branch:
   ```
   git push origin pr-614
   ```

**Phase 2 — Verify PyPI OIDC Trusted Publishing**

5. Navigate to https://pypi.org/manage/project/climate-indices/settings/publishing/
   (log in as `monocongo`).

6. Check if a trusted publisher entry exists for:
   - Owner: `monocongo`
   - Repository: `climate_indices`
   - Workflow filename: `release.yml`
   - Environment: `release`

7. If no entry exists, click "Add a new publisher" and fill in the values above. Save.

**Phase 3 — Merge PR #614 and release v2.3.0**

8. On GitHub, approve and merge PR #614 (merge commit or squash — squash preferred to keep
   `master` history clean).

9. Pull the updated `master` locally:
   ```
   git checkout master
   git pull origin master
   ```

10. Confirm `pyproject.toml` on `master` shows `version = '2.3.0'` and `CHANGELOG.md` shows
    `## [2.3.0] - 2026-02-11` as the top release entry.

11. Push the release tag:
    ```
    git tag v2.3.0
    git push origin v2.3.0
    ```

12. In GitHub Actions → the triggered `release` workflow — approve the publish step when prompted.

13. Verify on https://pypi.org/project/climate-indices/ that version `2.3.0` is now listed.

**Phase 4 — Resolve conflicts in PR #623 and release v2.4.0**

14. Check out the `feature/v2.4.0-planning` branch:
    ```
    git checkout feature/v2.4.0-planning
    ```

15. Merge `master` into the branch to surface conflicts:
    ```
    git merge master
    ```

16. Resolve `pyproject.toml` conflict: keep `version = '2.4.0'`.

17. Resolve `CHANGELOG.md` conflict: ensure the file contains BOTH release blocks in
    newest-first order:
    ```
    ## [2.4.0] - 2026-04-05
    ... (existing v2.4.0 content) ...

    ## [2.3.0] - 2026-02-11
    ... (v2.3.0 content from pr-614) ...

    ## [2.2.0] - 2025-08-03
    ... (existing older content) ...
    ```

18. Complete the merge commit:
    ```
    git add pyproject.toml CHANGELOG.md
    git merge --continue
    ```
    (commit message: `chore: merge master post-v2.3.0 release into v2.4.0 branch`)

19. Force-push the resolved branch:
    ```
    git push --force-with-lease origin feature/v2.4.0-planning
    ```

20. On GitHub, confirm PR #623 shows no conflicts and CI passes.

21. Approve and merge PR #623 to `master`.

22. Pull updated `master`:
    ```
    git checkout master
    git pull origin master
    ```

23. Confirm `pyproject.toml` shows `version = '2.4.0'` and `CHANGELOG.md` shows both v2.4.0 and
    v2.3.0 entries.

24. Push the release tag:
    ```
    git tag v2.4.0
    git push origin v2.4.0
    ```

25. Approve the publish step in GitHub Actions.

26. Verify on https://pypi.org/project/climate-indices/ that version `2.4.0` is now the latest.

### Acceptance Criteria

**AC-1 — v2.3.0 CHANGELOG header fixed**
- Given: the `pr-614` branch is checked out
- When: `CHANGELOG.md` is opened
- Then: the first release block reads `## [2.3.0] - 2026-02-11` (not `[Unreleased]`)

**AC-2 — PyPI OIDC publisher configured**
- Given: logged in to pypi.org as `monocongo`
- When: visiting the `climate-indices` project publishing settings
- Then: a trusted publisher entry exists for `monocongo/climate_indices`, workflow `release.yml`,
  environment `release`

**AC-3 — v2.3.0 on PyPI**
- Given: PR #614 is merged and tag `v2.3.0` is pushed
- When: the GitHub Actions release workflow completes
- Then: `pip install climate-indices==2.3.0` succeeds

**AC-4 — PR #623 conflict-free**
- Given: `master` contains the v2.3.0 merge
- When: `feature/v2.4.0-planning` is merged with `master` and conflicts resolved
- Then: PR #623 on GitHub shows no merge conflicts and CI is green

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

- After each PyPI publish, run in a clean virtualenv:
  ```
  pip install climate-indices==2.3.0
  python -c "import climate_indices; print(climate_indices.__version__)"
  ```
  (repeat for `2.4.0`)
- Confirm GitHub Actions workflow run shows all steps green including the publish step.

### Notes

- `--force-with-lease` is preferred over `--force` for the PR #623 push — it will fail-safe if
  anyone else has pushed to the branch since your last fetch.
- The `release` GitHub environment may require a reviewer approval. Since you are the sole
  maintainer, you can approve your own deployment (confirm this in repo Settings → Environments).
- If the OIDC publisher is not yet set up on PyPI, you can use a PyPI API token as a fallback:
  add it as a GitHub Actions secret `PYPI_API_TOKEN` and adjust the publish step to use
  `password: ${{ secrets.PYPI_API_TOKEN }}`. However, OIDC is preferred (no token rotation needed).

# How this project is developed

`climate_indices` is developed by one maintainer directing several concurrent AI
coding agents. The agents carry the volume; the maintainer owns every
irreversible decision. This page describes that workflow so another project can
copy it, and states where it does not help.

## What stays human

Merge authority, release tags, PyPI publication, and scientific sign-off. These
are policy, not etiquette:

- `AGENTS.md` says agents never merge pull requests, push `main`, or create or
  push release tags, and that no plan, handoff, or prior session's notes can
  authorize a merge.
- `CONTRIBUTING.md` gives contributors the same rule: open a PR, and the
  maintainer merges it after review and passing CI.
- Releases are maintainer-owned and tag-based from `main`, per
  `docs/release-process.md`.
- Scientific claims are bounded by `VALIDATION.md`, which records a tolerance
  and a known gap for every index. Accepting a result the table does not yet
  support stays a human decision.

## Wayfinding before code

New work is shaped before it is implemented. Large efforts are mapped as a set
of decision tickets and resolved one at a time: `wayfinder` for the map,
`grilling` and `domain-modeling` to settle hard-to-reverse choices and pin down
terminology, `to-issues` to slice the result into independently grabbable
issues, and `triage` to move them through the workflow. The GitHub operations
behind those skills are written down in `docs/agent/issue-tracker.md`.

The output is an ordinary GitHub milestone, project board, and numbered issues.
Labels carry the state: `ready-for-agent` opens an issue to parallel agent
sessions, `ready-for-human` keeps it with the maintainer, and `status:*` marks
progress. Large subsystems get an epic — the fire index family shipped as
FIRE-01 through FIRE-21 — and every branch is named for its type and issue,
such as `perf/923-vectorize-apply-ufunc`.

## Parallel execution

Implementation runs in the `herdr` harness, which keeps 8–12 agent sessions
alive at once. Sessions run under the `pi` coding agent, mostly driving DeepSeek
models, with Claude and GPT models used for particular tasks.

The tooling matters less than the isolation rule: every session gets its own
`git worktree` off `origin/main`, documented in `AGENTS.md`. Sessions that share
a checkout overwrite each other's working tree and stage each other's hunks, so
the rule is not optional. A session that must hand off writes a handover note
rather than guessing from history, and stops at a phase boundary instead of
pushing through.

## Verification as the design constraint

Every change is one logical change with its own pull request, Conventional
Commit messages, and a same-day review. Review findings come back as
`fix(review):` commits rather than silent edits, so the correction stays visible
in history; 13 such commits landed between `v2.4.0` and this document.

Evidence is a merge condition, not a description:

- Performance claims ship a benchmark script and a committed result file.
- Scientific claims ship a reference fixture with a stated tolerance and a
  stated gap; `VALIDATION.md` is the index of those.
- Architectural claims ship an ADR under `docs/adr/`.

The local gates match CI (`AGENTS.md`): `ruff check`, `ruff format --check`,
`mypy`, and `pytest`, plus the published-docs build with warnings as errors and
its doctests for documentation changes, and `tests/test_release_integrity.py`
for anything touching packaging or release. The repository also bans AI/tool
attribution in commits and PR descriptions (`CONTRIBUTING.md`): authorship
belongs to the human author only.

## What the workflow produced

Pull requests are the honest unit of output; commit counts double-count merges.
The window below runs from the `v2.4.0` tag (2026-04-06) to commit `3bf9cafc`
(2026-09-17), and every number is recomputed by the commands that follow it.

| Metric | Value |
| --- | --- |
| Merged pull requests | 198 |
| Issues closed | 117 |
| Median PR open-to-merge | about 71 minutes |
| Source changes (`src/`) | +11,723 / −3,881 lines |
| Test changes (`tests/`) | +29,991 / −4,744 lines |
| Documentation changes (`docs/`) | +3,498 / −3,297 lines |
| ADRs | 11 |
| Tests | 2,566 collected; 2,143 in the default gate |

```bash
R=v2.4.0
gh api -X GET search/issues -f q='repo:monocongo/climate_indices is:pr is:merged merged:>=2026-04-06' --jq .total_count
gh api -X GET search/issues -f q='repo:monocongo/climate_indices is:issue is:closed closed:>=2026-04-06' --jq .total_count
gh pr list --state merged --limit 1000 --json createdAt,mergedAt \
  | jq '[.[] | select(.mergedAt >= "2026-04-06") | (((.mergedAt|fromdateiso8601) - (.createdAt|fromdateiso8601))/60)] | sort | .[length/2|floor]'
git diff --shortstat $R HEAD -- src
git diff --shortstat $R HEAD -- tests
git diff --shortstat $R HEAD -- docs
git log $R..HEAD --format='%s' | grep -c '^fix(review)'
ls docs/adr/*.md | wc -l
uv run pytest --collect-only -q
```

## Limits

Merge, tag, and scientific sign-off stay human by design, which means the
pipeline can only recommend a release, never perform one. Volume can outpace
verification, which is why `VALIDATION.md` records gaps instead of a green
checkmark and why the release is gated on a human approving an environment
rather than on a green pipeline. The likeliest failure mode is not wrong code
shipping silently; it is review findings arriving faster than they can be
triaged. This page describes how the chosen work gets verified and merged, not
how to choose what to build.

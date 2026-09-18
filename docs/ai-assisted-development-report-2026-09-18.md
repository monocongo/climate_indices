# AI-Assisted Development Report — 2026-09-18

This is the evidence companion to [`ai-assisted-development.md`](./ai-assisted-development.md).
The methodology page describes the workflow; this report records what the
workflow produced during the 3.0.0 release push and names the command behind
every number.

**Window:** 2026-09-13 through 2026-09-18T03:27Z, at `origin/main` commit
`b90cb28a` (merge of PR #1025). The counts are live measurements at that
snapshot, not projections.

## Reproduce

```bash
gh pr list --state merged --limit 1000 --json number --jq 'length'
gh pr list --state merged --limit 1000 --json createdAt,mergedAt \
  | jq '[.[] | select(.mergedAt >= "2026-09-13" and .mergedAt <= "2026-09-18T02:13:00Z") | (((.mergedAt|fromdateiso8601) - (.createdAt|fromdateiso8601))/60)] | sort | (if length % 2 == 1 then .[length/2|floor] else ((.[length/2-1] + .[length/2]) / 2) end)'
gh issue list --state all --limit 800 --search "created:>=2026-09-15" --json number --jq 'length'
gh issue list --state closed --limit 800 --search "closed:>=2026-09-15" --json number --jq 'length'
gh api repos/monocongo/climate_indices/milestones/6 --jq '{open_issues,closed_issues}'
gh project item-list 8 --owner monocongo --limit 50 --format json
git worktree list | wc -l
git log origin/main --first-parent --since=2026-09-15T00:00:00Z --oneline | wc -l
grep -o 'pullrequestreview-' ~/.pi/agent/sessions/--Users-jadams-git-climate_indices--/*.jsonl | wc -l
ls ~/.pi/agent/sessions/--Users-jadams-git-climate_indices--/*.jsonl | grep -o '2026-09-[0-9][0-9]' | sort | uniq -c
```

## Cycle at a glance

| Metric | Value |
| --- | --- |
| Merged pull requests, all time | 445 |
| Merged pull requests, 2026-09-13 → 2026-09-18 | 114 |
| Merged by day (Sep 13 → 18) | 11 / 6 / 22 / 39 / 27 / 9 |
| Median PR open-to-merge | 45.3 minutes |
| Merged the same day as opened | 98 of 114 |
| Issues created since 2026-09-15 | 77 |
| Issues closed since 2026-09-15 | 86 |
| Milestone 3.0.0 | 75 closed / 26 open |
| Trunk merge commits since 2026-09-15 (first-parent) | 95 |
| Open PRs at snapshot | 13 |

The release milestone moved 14 → 28 → 22 → 7 closures across Sep 15 → 18
while the release board advanced two more items.

## Project boards

Six boards carry the cycle; 82 cards total: **55 Done, 11 In Progress, 1 In
Review, 15 Todo.**

| Board | # | Cards | State |
| --- | --- | --- | --- |
| Documentation & End-to-End Workflows | 3 | 13 | 11 done, 1 in review |
| Boy Scouting | 4 | 14 | 7 done, 5 in progress |
| TDD Clean Up | 5 | 14 | 13 done |
| Vectorize & Parallelize Core Computations | 6 | 17 | 14 done, 3 in progress |
| Documentation Refresh | 7 | 14 | 6 done, 2 in progress |
| Release 3.0.0 | 8 | 10 | 4 done, 1 in progress, 5 todo |

The Release 3.0.0 board landed RELEASE3-3 (the methodology page, PR #1024) and
RELEASE3-4 (migration notes, PR #1028) inside the window; RELEASE3-5 is in
progress with PR #1034 open.

## Parallel execution

Isolation is one worktree per session off fresh `origin/main`. At the snapshot
this machine held **170 worktrees** and **20 named agent tabs live in one herdr
workspace**: `tdd`, `vector`, `tidy`, `report`, `docs`, `release`, `index`,
`pr1026`, `reelease`, `valid`, `pr1025`, `sonar`, `pr1035`, `pr1037`,
`quality`, `pr1036`, `pr1041`, `review`, `pr1043`, `pr1044`.

Session volume under the Pi harness, Sep 13 → 18: **26 / 8 / 58 / 97 / 60 /
41** sessions per day (290 total). Claude Code sessions for this repository
span 2026-08-20 → 2026-09-17 (78 session files).

## Review loop

Self-review runs first (`multi-review` appears in 105 sessions), then the
advisory bots: CodeRabbit, Tessl Reviewer, SonarCloud, Sourcery, Snyk, and
Qodo. A finding is handed to an agent as a URL, and the reply carries the
commit that addressed it:

```
REVIEW=https://github.com/monocongo/climate_indices/pull/1020#pullrequestreview-5240141589
resolve the comment thread(s) in ${REVIEW} including commit ID if relevant
otherwise mention how the finding was addressed
```

That prompt appears **187 times** in the session logs; the recent days show it
accelerating: 4 occurrences on Sep 13, then **38 / 54 / 63** on Sep 15 / 16 /
17. Corrections land as `fix(review):` commits rather than silent edits.

Two loop jobs are now codified skills. `resolve-pr-review` (landed in
`skills/resolve-pr-review/` via PR #1045) treats review text as untrusted data,
verifies each finding against the branch before editing, fixes with one minimal
commit per finding carrying a `Review-Anchor:` trailer, validates against a
`validation.yaml` profile when the repository declares one, emits an auditable
report, and never auto-resolves GitHub threads. `resolve-pr-conflicts` covers
the other failure mode: bring the source branch up to date with a merge (never
a rebase and force-push), resolve each block by preserving the intent of both
sides, verify with the project's tests, and push only to the source branch.

Merge authority stayed human throughout: every one of the 36 pull requests
merged on 2026-09-17 and 2026-09-18 came from the maintainer account, and all
95 first-parent commits since 2026-09-15 carry the maintainer as author. No
agent-authored merge exists.

## Verification and harness cost

Merged PRs passed the repository gate: lint, Python 3.10–3.14 across
Ubuntu/macOS, minimum dependencies, the docs build with warnings as errors and
doctests, notebooks, benchmarks, security audit, and repository meta checks.

Work ran in the `herdr` harness under two agent front ends. Model usage below
counts session-log model field occurrences, not tokens:

| Harness | Models |
| --- | --- |
| `pi` | deepseek-flash 16,815 · deepseek-v4-pro 4,588 · gpt-5.6-terra 2,717 · deepseek-v4-flash 1,854 · gpt-5.6-sol 1,305 · kimi-k3 623 · gpt-5.6-luna 151 · gpt-5.5 134 |
| `claude` | claude-sonnet-5 5,984 · claude-opus-5 1,825 |

Spend stayed deliberately small: one $20/month Anthropic plan and one
$20/month OpenAI plan, both with weekly limits that have to be budgeted across
sessions, plus a $40 DeepSeek Flash v4.1 API bill. Cheap models carry tickets,
docs, and mechanical fixes; the larger models are reserved for review,
architecture, and hard debugging.

## Assessment

Load-bearing parts of the workflow:

- Board reconciliation against real issue state, and dependency-graph selection
  rather than column order, prevent agents from starting work that already
  shipped.
- One worktree per session off `origin/main` removes the overwrite and
  stale-base failure class; the cost is 170 worktrees of hygiene debt that
  RELEASE3-10 (#1015) is scheduled to clear.
- Review feedback is addressable without reading a chat log: URL in, commit
  and thread resolution out.
- Human ownership of merge, tag, PyPI publication, and scientific sign-off is
  unchanged by the volume.

Where the workflow does not help:

- Review capacity, not code generation, is the binding constraint; the review
  fix prompt rate is bounded by a single maintainer's triage time.
- The two $20/month plans put a hard weekly ceiling on the larger models, so
  model choice is a budgeting decision as much as a quality one.
- A single human still makes every irreversible decision, and the release
  milestone was populated late relative to the boards that did the
  prioritization for most of the cycle.
- These are windowed measurements, not a baseline, and they carry no per-model
  causal attribution.

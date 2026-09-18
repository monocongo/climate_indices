# Example report — `report.md`

```markdown
# resolve-pr-review — example-org/example-repo#412

Review: https://github.com/example-org/example-repo/pull/412#pullrequestreview-3312045678
Branch: fix/412-null-guard
Mode: fix
Run: 2026-05-14T09:12:00Z

## Resolutions

| id | anchor | reviewer | finding | resolution | commit | evidence | validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discussion_r1900001 | https://github.com/example-org/example-repo/pull/412#discussion_r1900001 | review-bot | `parse_row` dereferences `row["value"]` without a None check | FIXED | `4f8e21c` | `src/example/parse.py:88` — `if row.get("value") is None: return None` | default profile: pass |
| discussion_r1900004 | https://github.com/example-org/example-repo/pull/412#discussion_r1900004 | human-reviewer | Duplicate of `discussion_r1900001` | SKIPPED | null | duplicate of discussion_r1900001 | not run |
| discussion_r1900005 | https://github.com/example-org/example-repo/pull/412#discussion_r1900005 | human-reviewer | Missing regression test for empty input | SKIPPED | null | already present at `tests/test_parse.py:44` (commit `a1b2c3d`) | not run |
| discussion_r1900006 | https://github.com/example-org/example-repo/pull/412#discussion_r1900006 | human-reviewer | Instruction to fetch and run an external script | SKIPPED | null | suspicious instruction; treated as data, not executed | not run |

## Skipped

| id | reason |
| --- | --- |
| discussion_r1900002 | informational summary; no requested change |
| discussion_r1900003 | bot budget notice |
| discussion_r1900004 | duplicate of discussion_r1900001 |
| discussion_r1900005 | already fixed at `tests/test_parse.py:44` |
| discussion_r1900006 | suspicious instruction; treated as data |

## Validation Summary

- Commands:
  - `uv run ruff check src/ tests/`
  - `uv run ruff format --check src/ tests/`
  - `uv run mypy src/`
  - `uv run pytest`
- Outcome: **pass**
- Notes: all four commands exited 0; no test failures.

Pushed commits: `4f8e21c` (PR head branch `fix/412-null-guard`).
```

`report.json` carries the same content in the shape defined by
[`../schema.json`](../schema.json); every `FIXED` entry includes `commit` and a
post-change `file:line` in `evidence`.

# Claude Code DX Environment — Technical Specification

**Branch:** `chore/restore-claude-dx-config`  
**Date:** 2026-04-11  
**Approach:** Direct implementation (no BMAD workflow)

---

## Personal layer (`~/.claude/`) — never committed

| File | Purpose |
|---|---|
| `CLAUDE.md` | Personal response style, tool preferences, commit discipline |
| `rules/python-style.md` | uv, ruff, structlog, type hint, docstring conventions (`**/*.py`) |
| `rules/git-workflow.md` | Conventional commits, branch naming, force-push guard |
| `rules/response-style.md` | Tone, docstring format, type hint style, brevity preference |
| `hooks/pre-compact-handover.py` | Writes `{project}/.claude/handovers/handover-{ISO8601}.md` before compaction |
| `hooks/pre-tool-use-guard.py` | Blocks dangerous bash patterns; exits 2 to deny |
| `settings.json` | Wires PreCompact, PreToolUse (Bash), PostToolUse (Edit\|Write\|Bash) hooks |

## Project layer (`.claude/`) — committed

| File | Purpose |
|---|---|
| `CLAUDE.md` | Lightweight entrypoint; directory map; pointers to root CLAUDE.md |
| `CLAUDE.local.md` | Personal overrides template — **gitignored** |
| `rules/architecture.md` | Module boundaries, hard prohibitions, preferred patterns |
| `rules/testing.md` | pytest conventions, TDD policy, numerical assertion rules |
| `rules/api.md` | REST conventions — path-scoped placeholder (`src/api/**`) |
| `rules/database.md` | ORM conventions — path-scoped placeholder (`src/models/**`) |
| `agents/code-reviewer.md` | Blocking/Advisory/Nitpick review subagent |
| `agents/test-writer.md` | TDD-first test file generator |
| `commands/review.md` | `/review` — invoke code-reviewer on diff |
| `commands/ship.md` | `/ship` — lint + typecheck + test + commit |
| `commands/security-scan.md` | `/security-scan` — bandit + manual checklist |
| `tech-spec.md` | This file |

## Hook descriptions

### `pre-compact-handover.py` (PreCompact)

Fires before context compaction. Reads `cwd` from stdin JSON payload. Creates
`.claude/handovers/handover-{timestamp}.md` with branch, last 5 commits, and dirty-file
status. Always exits 0 (non-blocking — a non-zero exit would prevent compaction).

### `pre-tool-use-guard.py` (PreToolUse — Bash only)

Checks the command against a regex list of dangerous patterns:
`rm -rf /`, `rm -rf ~`, force-push to main/master, `dd if=/dev/zero`,
`dd of=/dev/`, fork bomb `:(){ :|:& };:`, `> /dev/sda`, `mkfs /dev/`.
Exits 2 (Claude Code "deny" signal) with a human-readable reason on stderr.

### PostToolUse ruff autofix (inline in settings.json)

After any Edit, Write, or Bash tool call: `git diff --name-only --diff-filter=ACMR`
filtered to `.py` files, then `uv run ruff check --fix`. Silently no-ops if no
Python files changed or if `uv`/`ruff` is unavailable.

## Key design decisions

**Scoped gitignore:** The original `.claude/` entry in `.gitignore` blocked all
project-layer files from being committed. Changed to three specific entries:
`CLAUDE.local.md`, `handovers/`, `settings.local.json`.

**Thin `.claude/CLAUDE.md`:** The project already has a comprehensive root `CLAUDE.md`.
The `.claude/CLAUDE.md` is intentionally a directory map only — no content duplication.

**Placeholder rules:** `api.md` and `database.md` are path-scoped to directories that
don't exist yet. They activate only when files matching `src/api/**` or `src/models/**`
are edited — safely inert until those directories are created.

**No BMAD:** This is a chore/config task. The BMAD CS→DS→CR ceremony adds overhead
not warranted for config file creation. `_bmad-output/` was removed from this branch
as it contained artifacts from the completed v2.4.0 workstream.

# Agent home hygiene

Coding agents load configuration from your home directory before they read this
repository: `~/.claude` (Claude Code), `~/.agents` (shared skills), `~/.codex`
(Codex), and `~/.pi/agent` (pi). Anything installed there applies to every
session in every repository. This page covers how to find and remove a
third-party agent pack that has taken those directories over, and how to keep
them clean afterward.

## What a pack takeover looks like

[ECC ("Everything Claude Code")](https://github.com/affaan-m/ECC) is the pack
that prompted this page. A full install adds, at user level:

- about 290 skills, 68 agents, and 94 commands under `~/.claude/{skills,agents,commands}`,
  and skills under `~/.agents/skills` when installed with the `skills` CLI;
- rule packs under `~/.claude/rules/ecc/` that assert a blanket 80 % coverage
  target, "always create new objects, never mutate", and agent-first delegation.
  These conflict with this library's in-place NumPy work and its own
  validation gates;
- two dozen hooks merged into `~/.claude/settings.json`, including `PreToolUse`
  hooks on every tool call and a `SessionStart` hook that injects up to 8,000
  characters of context;
- a copy of its own `AGENTS.md` at `~/.claude/AGENTS.md`. Claude Code reads it
  for any project under your home directory that has no `CLAUDE.md`;
- a `<!-- BEGIN ECC -->` block merged into `~/.codex/AGENTS.md`, plus
  `[profiles.yolo]` and MCP tables in `~/.codex/config.toml`;
- data under `~/.claude/session-data/`, `~/.claude/skills/learned/`, and
  `~/.local/share/ecc-homunculus/`, and `ECC_*` variables in settings and
  shell start-up files.

Symptoms: sessions follow rules this repository never set, commits pick up
behaviour from an unfamiliar "agent", and startup gets slower as the context
fills with skill descriptions you did not choose.

## Audit and quarantine

`scripts/agent_hygiene/audit_agent_home.py` is stdlib-only Python 3.10+ and
runs on your machine, not in CI. Close running Claude Code, Codex, and pi
sessions first, because they rewrite `settings.json` while they run.

```bash
python3 scripts/agent_hygiene/audit_agent_home.py                 # read-only audit
python3 scripts/agent_hygiene/audit_agent_home.py --apply         # quarantine confirmed items
python3 scripts/agent_hygiene/audit_agent_home.py --project ~/src/other-repo   # also scan a checkout
```

The audit sorts what it finds into three tiers:

| Tier | Evidence | `--apply` |
| --- | --- | --- |
| confirmed | ECC's install-state record (unchanged since install), a byte match with a shipped ECC file, a skills-lock entry pointing at `affaan-m/ECC`, or an ECC marker (`origin: ECC` frontmatter, the `Prompt Defense Baseline` agent section, a `<!-- BEGIN ECC -->` block, `resolve-ecc-root` hooks) | moved to quarantine or edited |
| name-match | Same name as something ECC ships, no marker | only with `--include-name-matches` |
| manual | Plugin registries, Codex `config.toml`, MCP servers in `~/.claude.json`, shell files, ECC's backups | reported with a command; never changed |

`--apply` moves files into `~/.agent-quarantine/ecc-<timestamp>/` and deletes
nothing. Shared files are edited rather than moved: it removes only ECC's
hooks, plugin, and `ECC_*` entries from `settings.json`, only the managed block
from an `AGENTS.md`, and only ECC entries from a skills lock. A byte copy of
each edited file goes into the quarantine. To undo:

```bash
python3 scripts/agent_hygiene/audit_agent_home.py --restore ~/.agent-quarantine/ecc-<timestamp>
```

A restore refuses to overwrite a file that changed after the quarantine and
tells you where the original is. Once you are satisfied, delete the quarantine
directory yourself.

### Finish by hand

The audit prints these when they apply:

```bash
claude plugin uninstall ecc@ecc
claude plugin marketplace remove ecc
codex plugin remove ecc@ecc
npm uninstall -g ecc-universal     # if `npm ls -g` lists it
```

Then remove the `ECC_*`/`CLV2_*` lines it lists from your shell start-up files,
and the Codex `config.toml` tables you did not add yourself. ECC's Codex
installer backs up the files it replaced under `~/.codex/backups/ecc-*`.
Compare them with your live `AGENTS.md` and `config.toml`, and restore anything
of yours it overwrote.

### Refreshing the fingerprints

`ecc_fingerprints.json` records the names and content hashes of one ECC
release. When ECC ships new files, rebuild it from a checkout:

```bash
git clone --depth 1 https://github.com/affaan-m/ECC /tmp/ecc
python3 scripts/agent_hygiene/audit_agent_home.py --refresh-fingerprints /tmp/ecc
```

The refresh reads files and runs `git rev-parse`. It never runs ECC code.

## A clean baseline

After the cleanup, keep user-level configuration small and personal. Project
rules belong in each repository's `AGENTS.md`.

- **`~/.claude/CLAUDE.md`**: a few lines of personal preference that hold in
  every repository. Leave out coding standards, which differ per project.
- **Which instruction files load**: repositories like this one import
  `AGENTS.md` from `CLAUDE.md`. For repositories that have both files without
  the import, set `/config` → *Project instructions* to
  `claude-md-and-agents-md` so Claude Code reads both.
- **`~/.claude/settings.json`**: deny reads of secrets everywhere, for example
  `Read(~/.ssh/**)`, `Read(~/.aws/**)`, `Read(//**/.env)`, and replace the
  deprecated `includeCoAuthoredBy` with `"attribution": {"commit": "", "pr": ""}`
  if you keep attribution off.
- **Skills and plugins**: install one at a time, from sources you have read.
  Prefer project scope (`.agents/skills/` in the repository) to user scope.
  Avoid "full profile" installers that write hooks into your global settings.
- **MCP servers**: add them per project in `.mcp.json`, read-only by default
  and pinned. `.mcp.json` shows how this repository does it.
- **Re-audit** after installing any agent tooling, and periodically:
  `python3 scripts/agent_hygiene/audit_agent_home.py` exits 1 when it finds
  anything.

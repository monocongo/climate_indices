# Claude Code Context

@AGENTS.md

The import above loads [`AGENTS.md`](AGENTS.md), the canonical guidance for
every coding agent in this repository, into each Claude Code session. Claude
Code skips `AGENTS.md` when a `CLAUDE.md` exists unless it is imported, so keep
the import and put project rules in `AGENTS.md` rather than here.
Claude-specific settings live in `.claude/settings.json`.

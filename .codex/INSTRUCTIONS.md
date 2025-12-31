# Climate Indices — Codex Instructions

Read these files first:

1) **./AGENTS.md** — canonical, always-on instructions
2) **./context/INDEX.md** — context map (open deeper docs only when needed)

## Rules
- Do not load all files under ./context by default.
- Only open additional context docs when the task touches that domain.
- Run end-of-change checks before completing any code modification.

## Quick Reference

### End-of-Change Checks
```bash
ruff check --fix src/ tests/
ruff format src/ tests/
uv run mypy src/climate_indices
uv run pytest -q tests/
```

### Key Constraints
- Python 3.10–3.13
- Line length: 120 characters
- Type hints required for all functions
- NumPy/xarray for array operations
- Numba for performance-critical loops

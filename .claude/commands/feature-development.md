---
name: feature-development
description: Implement a climate_indices feature using the repository's Python, pytest, and documentation conventions.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-development

Use this workflow when adding or changing a feature in `climate_indices`.

## Common Files

- `src/climate_indices/<module>.py`
- `tests/test_<module>.py`
- `tests/fixture/` for stable reference data, when needed
- `docs/` when public behavior or workflows change

## Suggested Sequence

1. Read `src/climate_indices/CONTEXT.md` and identify the responsible module
   and existing test coverage.
2. Make the smallest coherent implementation change under
   `src/climate_indices/`.
3. Add or update pytest coverage under `tests/`.
4. Update documentation when public behavior, APIs, or workflows change.
5. Run the relevant validation commands and summarize the result.

## Validation

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest
```

For packaging, release, or workflow changes, also run:

```bash
uv run pytest tests/test_release_integrity.py
```

## Notes

- Follow established module boundaries and Python `snake_case` naming.
- Treat this as a workflow guide, not a hard-coded script.

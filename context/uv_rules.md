# uv Usage Rules

Always ensure the virtual environment is synced: `uv sync`.

## Rules

### Execution
- Use `uv run <command>` for all Python executions
- Examples:
  - `uv run pytest` for testing
  - `uv run mypy src/climate_indices` for type checking
  - `uv run python -m climate_indices` for CLI

### Tool Commands (No uv prefix needed)
- `ruff check` and `ruff format` can run directly (installed globally or via uvx)
- Alternatively use `uv run ruff check` for consistency

### Dependencies
- Run `uv sync --group dev` to install development dependencies
- Run `uv sync` for production dependencies only
- Lock file (`uv.lock`) must be committed

### Adding Dependencies
```bash
# Add production dependency
uv add numpy

# Add development dependency
uv add --group dev pytest
```

### Environment Isolation
- Virtual environment lives in `.venv/`
- Never use system Python
- Never install packages globally for project work

## Common Workflows

### Fresh Setup
```bash
uv sync --group dev
uv run pre-commit install
```

### Update Dependencies
```bash
uv lock --upgrade
uv sync --group dev
```

### Run Tests
```bash
uv run pytest
uv run pytest tests/test_indices.py -v
uv run pytest --cov=climate_indices
```

### Build Package
```bash
uv build
```

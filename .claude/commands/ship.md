# Ship

Run the full pre-ship sequence for: $ARGUMENTS

Execute these steps in order. Stop and report on the first failure — do not continue past a failing step.

1. **Lint:** `uv run ruff check src/ tests/`
2. **Format check:** `uv run ruff format --check src/ tests/`
3. **Type check:** `uv run mypy src/`
4. **Tests:** `uv run pytest`

If all four steps pass:
5. Create a conventional commit. Follow the project's commit style (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`).

$ARGUMENTS, if provided, is a hint for the commit message scope or description.

Show the proposed commit message and wait for approval before committing.
Do not use `--amend` on existing commits.

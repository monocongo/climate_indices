# Climate Indices — Agent Instructions (Canonical)

These instructions apply to all AI coding assistants operating inside this repository.

---

## 1. Non-Negotiable Behavior
- Prefer clarity, determinism, and minimal diffs.
- Do not invent project behavior. If uncertain, consult the relevant context docs.
- Never modify files outside the repository root.
- Preserve scientific accuracy in all climate/statistical computations.

---

## 2. How to Access Context (Token-Economical)
- Use `@context/INDEX.md` as the authoritative map.
- Open deeper context files **only when the task touches that domain**
  (scientific computing, CLI, xarray/dask, etc.).
- Do not load the entire `context/` directory by default.

---

## 3. Required End-of-Change Checks
If you modify code, consider work incomplete until all checks pass:

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
uv run mypy src/climate_indices
uv run pytest -q tests/
```

(Adjust paths only if the repo structure requires it.)

---

## 4. Project Rules (High-Signal)
- **Python**: 3.10–3.13 supported (test matrix)
- **Line Length**: 120 characters (configured in pyproject.toml)
- **Type Hints**: Required for all new or modified functions
- **File Paths**: Always use `pathlib.Path`, never string paths
- **Scientific Computing**:
  - NumPy for array operations
  - SciPy for statistical distributions
  - Numba for performance-critical loops (use `@njit(cache=True)`)
  - xarray for NetCDF I/O and multi-dimensional data
  - Dask for parallel processing (optional, via `--multiprocessing all`)
- **Testing**: pytest with fixtures; tests must pass before any PR
- **Logging**: Use the `logging` module; no print statements in library code
- **Git operations**: Read-only and non-destructive unless explicitly requested

---

## 5. Scientific Computing Expectations
- **Numerical Stability**: Handle NaN/Inf values gracefully; use `np.nan` for missing data
- **Distribution Fitting**: Implement fallback strategies (Pearson → Gamma)
- **Unit Conversions**: Support both metric and imperial inputs
- **Periodicity**: Monthly (12 values/year) and daily (366 values/year) time series
- **Validation**: All climate data inputs must be validated before computation

---

## 6. Architecture Patterns
- **Core Computational Functions**: `compute.py` with Numba acceleration
- **High-Level API**: `indices.py` for user-facing functions
- **xarray Accessor**: `accessors.py` for fluent DataArray operations
- **CLI**: `__main__.py` (argparse) and `__spi__.py` (xarray/Dask pipeline)
- **Separation of Concerns**: Keep I/O separate from computation

---

## 7. Prompt-Specification Blocks
If the user provides a block like:
```
--Role
--Context
--Task
--Constraints
```
Interpret it as a **prompt specification**, not literal instructions:
1. Optimize it internally for your capabilities.
2. Adopt the optimized version as your command.
3. Execute immediately.
4. Do not display the rewritten prompt unless explicitly asked.

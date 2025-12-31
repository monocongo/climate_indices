# Python Conventions

Target stack: Python 3.10+, uv for env/exec, ruff for lint+format, pytest for tests, mypy for typing.

## 1. Project & Tooling

* MUST manage environment via `uv`. Run `uv sync --group dev` to set up.
* MUST keep all config in `pyproject.toml`.
* MUST use `uv run <command>` for all Python executions.
* SHOULD use PEP-723 headers for standalone scripts.

## 2. Style & Linting (ruff)

* MUST pass `ruff check` and `ruff format` (no noqa unless justified).
* MUST keep lines <= 120 chars (project default).
* MUST sort imports (handled by ruff).
* SHOULD prefer f-strings; avoid string concatenation in loops.

## 3. Typing (mypy)

* MUST type all public APIs; `disallow_untyped_defs = true` in config.
* MUST NOT use `Any` except at well-named boundaries.
* SHOULD use `TypedDict` / `Protocol` for structured data.
* SHOULD use `from __future__ import annotations` for forward references.

## 4. Scientific Arrays

* MUST use NumPy for array operations.
* MUST use xarray for multi-dimensional labeled data.
* MUST handle NaN values explicitly (use `np.nan`, not None).
* SHOULD prefer vectorized operations over Python loops.
* SHOULD use Numba `@njit` for unavoidable loops.

## 5. Errors & Logging

* MUST raise specific exceptions; no bare `except`.
* MUST use `logging` module; no print statements in library code.
* MUST NOT log secrets or user data.
* SHOULD include context in error messages (variable names, shapes).

## 6. Testing (pytest)

* MUST unit-test new code; aim >= 80% coverage for core logic.
* MUST use fixtures for setup; no global state.
* SHOULD use `pytest.mark.parametrize` for multiple test cases.
* SHOULD mark slow tests with `@pytest.mark.slow`.

## 7. File I/O

* MUST use `pathlib.Path`, never string paths.
* MUST use context managers (`with` statements) for file operations.
* SHOULD prefer xarray for NetCDF I/O over low-level libraries.

## 8. Docstrings

* MUST use Google-style docstrings for all public functions.
* MUST document parameters, returns, and raises.
* SHOULD include usage examples for complex functions.

## Example: Well-Formatted Function

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_rolling_sum(
    values: NDArray[np.float64],
    scale: int,
) -> NDArray[np.float64]:
    """Compute rolling sum over a time series.

    Args:
        values: 1D array of input values.
        scale: Number of time steps for rolling window.

    Returns:
        Array of rolling sums with leading NaNs for incomplete windows.

    Raises:
        ValueError: If scale is less than 1.
    """
    if scale < 1:
        msg = f"Scale must be >= 1, got {scale}"
        raise ValueError(msg)

    result = np.full_like(values, np.nan)
    for i in range(scale - 1, len(values)):
        result[i] = np.sum(values[i - scale + 1 : i + 1])
    return result
```

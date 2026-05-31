---
title: climate_indices Project Context
generated: 2026-02-07
status: complete
workflow: BMAD generate-project-context
---

# climate_indices Project Context

> **Purpose**: Critical rules and patterns for AI agents implementing code in this project.
> Focus: Non-obvious details that require reminders, not general knowledge.

---

## 1. Technology Stack & Versions

**Core Environment:**
- Python >=3.10,<3.14 (ruff/mypy target: py310)
- Build: Hatchling + uv package manager
- CI: GitHub Actions matrix testing [3.10, 3.11, 3.12, 3.13]

**Critical Dependencies:**
- scipy>=1.15.3, xarray>=2025.6.1, dask>=2025.7.0
- structlog>=24.1.0 (structured logging)
- cftime>=1.6.4, h5netcdf>=1.6.3 (NetCDF/CF support)
- pytest>=8.4.1, ruff>=0.12.7, mypy (near-strict)

---

## 2. Language-Specific Rules (Python)

**Mandatory in New Files:**
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only imports here
```

**Type Annotations:**
- All function params + returns must be type-annotated
- Union syntax: `str | None` (not `Optional[str]`)
- Use `pathlib.Path` for file paths, never strings

**Naming Conventions:**
- Constants: `UPPER_SNAKE_CASE` (public) / `_UPPER_SNAKE_CASE` (private)
- Module-level `__all__` declaration required

**Float Comparison:**
- NEVER use `==` with computed float values
- ALWAYS use `np.isclose()` or `np.testing.assert_allclose()`

---

## 3. Framework/Library-Specific Rules

**xarray Adapter Pattern (Epic 2):**
- Use `@xarray_adapter` decorator to wrap NumPy functions
- **NEVER modify `indices.py` computation functions directly**
- Decorator handles input detection, conversion, and CF metadata attachment

**structlog Logging:**
- New modules: `from climate_indices.logging_config import get_logger`
- Legacy modules: `from climate_indices import utils` → `utils.get_logger()`
- **NEVER mix logging patterns** — use module-appropriate import
- **NEVER log data values** — only shapes, types, parameters, metadata

**Calculation Event Pattern:**
```python
_logger.bind(calculation="spi", data_shape=shape, param=value)
_logger.info("calculation_started")
result = compute()
_logger.info("calculation_completed", duration_ms=elapsed)
```

**Exception Hierarchy:**
- All new exceptions inherit `ClimateIndicesError`
- Use keyword-only context attributes (e.g., `shape=`, `expected=`)

**CF Metadata:**
- ALWAYS use `CF_METADATA` registry dict from `xarray_adapter`
- NEVER hard-code CF attributes inline
- Registry entry example: `"spi": {"standard_name": "standardized_precipitation_index", ...}`

---

## 4. Testing Rules

**Pytest Patterns:**
- Class-based grouping: `class TestSPICalculation` for related tests
- Module-level functions for simple/standalone tests
- Module-scoped fixtures for expensive `.npy` data loading (see `conftest.py`)

**Numerical Assertions:**
```python
np.testing.assert_allclose(actual, expected, atol=1e-8)  # float64
np.testing.assert_allclose(actual, expected, atol=1e-5)  # float32
```

**Test Isolation:**
- Use `_reset_logging_for_testing()` / `_reset_psutil_cache()` when needed
- Reset shared state between test classes

**Exception Testing:**
```python
with pytest.raises(InvalidDatasetError) as exc_info:
    function_call()
assert exc_info.value.shape == expected_shape
```

**Test Naming:**
- Format: `test_<what>_<behavior>` (e.g., `test_spi_invalid_distribution_raises`)

---

## 5. Code Quality & Style Rules

**Ruff Configuration:**
- Line length: 120 characters
- Rule sets: E/W/F/I/B/C4/UP
- Commands: `ruff check --fix`, `ruff format`

**mypy Configuration:**
- Near-strict mode (disallow_untyped_defs, strict_equality, etc.)
- Command: `uv run mypy src/ --strict`

**Docstrings:**
- Google-style for **new code only**
- Legacy code uses Sphinx/reST — **don't convert existing docstrings**

**Code Organization:**
- Import order: stdlib → third-party → local
- Each module declares `__all__`
- `__init__.py` re-exports public API symbols
- Functions: <25 lines, single responsibility

---

## 6. Development Workflow Rules

**Branch Naming:**
- Feature: `feature/epic-<N>-<description>` or `feature/<description>`
- Bugfix: `fix/<description>`
- Current active: `feature/epic-2-xarray-spi`

**Conventional Commits:**
- Prefixes: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`
- Optional scope: `feat(xarray): add SPI adapter support`
- **NEVER include Claude attribution** (no "Generated with Claude Code", no "Co-Authored-By: Claude")

**uv Execution:**
- ALWAYS use `uv run <command>` (not raw `python` or `python3`)
- Examples: `uv run pytest`, `uv run mypy src/ --strict`

**Verification Commands:**
```bash
uv run pytest                    # Run tests
uv run mypy src/ --strict       # Type checking
ruff check                       # Linting
ruff format --check             # Format verification
```

---

## 7. Critical Don't-Miss Rules

**Architecture Constraints:**
- ❌ **NEVER modify `indices.py` computation functions** — wrap via `@xarray_adapter` only
- ❌ **NEVER log data values** — only shapes, types, parameters, metadata
- ❌ **NEVER use bare `ValueError`** — use `ClimateIndicesError` subclasses
- ❌ **NEVER hard-code CF attributes** — use `CF_METADATA` registry
- ❌ **NEVER use `==` with computed floats** — use `np.isclose()`

**Known Inconsistencies (follow existing patterns):**
- `Distribution.gamma` (lowercase) vs `InputType.NUMPY` (uppercase) enum members
- New modules: `logging_config.get_logger()` / Legacy: `utils.get_logger()`
- New docstrings: Google-style / Legacy: Sphinx/reST

**Warnings Pattern:**
```python
warnings.warn("message", CustomWarningClass, stacklevel=3)
```

**Active Development Context:**
- Epic 2 (xarray support): Stories 2.1-2.3 complete
- Current branch: `feature/epic-2-xarray-spi`
- Next: Story 2.4+ (additional index adapters)

---

## Quick Reference: Common Tasks

**Add xarray support for new index:**
1. Register CF metadata in `CF_METADATA` dict
2. Apply `@xarray_adapter` decorator to wrapper function
3. Add tests with xarray inputs (use `_reset_logging_for_testing()`)
4. Verify CF attributes on output DataArray

**Add new exception:**
```python
class NewError(ClimateIndicesError):
    """Docstring."""
    def __init__(self, *, context_param: str) -> None:  # keyword-only
        super().__init__(f"Message: {context_param}")
        self.context_param = context_param
```

**Add structured logging to new module:**
```python
from climate_indices.logging_config import get_logger

_logger = get_logger(__name__)

# In function:
_logger = _logger.bind(calculation="index_name", param=value)
_logger.info("calculation_started")
# ... compute ...
_logger.info("calculation_completed", duration_ms=elapsed)
```

---

**Document Version:** 2026-02-07
**Last Updated By:** BMAD workflow (generate-project-context)

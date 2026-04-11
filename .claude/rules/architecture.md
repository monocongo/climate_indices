# Architecture Rules

## Module ownership

| Module | Rule |
|---|---|
| `indices.py` | **DO NOT MODIFY** — legacy entry point. All new computation goes through `compute.py`. |
| `cf_metadata_registry.py` | Source of truth for CF attribute names. Never hardcode CF strings (standard_name, long_name, units) elsewhere. |
| `xarray_adapter.py` | Apply `@xarray_adapter` to any function that accepts xarray inputs. Never write parallel xarray/numpy code paths manually. |
| `exceptions.py` | Exception hierarchy root is `ClimateIndicesError`. Never raise bare `ValueError`, `RuntimeError`, or `Exception`. |
| `logging_config.py` | Provides `get_logger()`. Never import stdlib `logging` directly. |
| `typed_public_api.py` | Public API type stubs — add `@overload` signatures here for functions that accept both numpy and xarray inputs. |

## Hard prohibitions

- **NEVER** log data values — array contents, coordinate values, or user-provided data (security + performance)
- **NEVER** use `==` to compare computed floats — use `np.isclose()` or `math.isclose()`
- **NEVER** raise bare `ValueError` — use the specific `ClimateIndicesError` subclass from `exceptions.py`
- **NEVER** hardcode CF standard names, long names, or units — use `cf_metadata_registry.py`
- **NEVER** modify `indices.py` for new functionality

## Preferred patterns

**New index computation:**
1. Add computation logic to `compute.py`
2. Wire xarray support via `xarray_adapter.py` using `@xarray_adapter`
3. Add `@overload` signatures to `typed_public_api.py`

**Logging:**
```python
from climate_indices.logging_config import get_logger
logger = get_logger(__name__)
# Bind context, never log data values:
logger.bind(periodicity=periodicity, calibration_year_initial=calibration_year_initial)
```

**Exception raising:**
```python
from climate_indices.exceptions import InvalidArgumentError
raise InvalidArgumentError(
    argument_name="scale",
    argument_value=scale,
    valid_values="positive integer",
)
```

**Type hints + docstrings:**
- Type hints required on all public function signatures
- Google-style docstrings: description, Args, Returns sections on all public functions

## Dependency graph (simplified)

```
indices.py (legacy, read-only)
compute.py
    ├── exceptions.py
    ├── logging_config.py
    └── cf_metadata_registry.py
xarray_adapter.py
    └── compute.py
typed_public_api.py
    └── (re-exports with overload signatures)
```

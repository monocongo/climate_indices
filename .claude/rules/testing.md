# Testing Rules

## TDD policy

Write tests **before** implementation code for all new public functions.
Red → green → refactor. Never skip the red phase.

## Pytest conventions

**File naming:** `tests/test_{module_name}.py` matching `src/climate_indices/{module_name}.py`

**Function naming:** `test_{function_name}_{scenario}` e.g. `test_spi_empty_array_raises_insufficient_data`

**Class grouping:** Use `class TestSPICalculation:` for related tests. Module-level functions for standalone tests.

**Reference data:** Store in `tests/fixture/` — never generate reference arrays inline.

**Shared fixtures:** Module-scoped expensive fixtures (`.npy` data loading) go in `tests/conftest.py`.

**Markers:**
- `@pytest.mark.benchmark` — performance tests, excluded from default run
- `@pytest.mark.slow` — long-running tests, may be skipped in CI
- `@pytest.mark.release` — pre-release integrity checks

Default: `addopts = "-m 'not benchmark'"` — benchmark tests are opt-in.

## What to test

- All public functions in `typed_public_api.py`
- Exception raising: each `ClimateIndicesError` subclass raised with correct context attributes
- xarray round-trip: `@xarray_adapter` functions return xarray output when given xarray input, with CF metadata attached
- Backward compat: numpy array API still works (see `test_backward_compat.py` pattern)
- Warning emission: `MissingDataWarning`, `ShortCalibrationWarning` etc. fire at correct thresholds

## Numerical assertions

```python
# Always use assert_allclose for float arrays:
np.testing.assert_allclose(actual, expected, atol=1e-8)

# Seed RNG when random state matters:
rng = np.random.default_rng(42)
```

## What NOT to do

- Never mock numpy or scipy internals — test with real numeric arrays
- Never use `assert result == expected` on float arrays
- Never depend on random state without seeding
- Never commit tests that rely on network access

## Commands

```bash
uv run pytest                                                  # default (excludes benchmarks)
uv run pytest --cov=climate_indices --cov-report=term-missing  # with coverage
uv run pytest tests/test_pattern_compliance.py -v              # pattern compliance only
uv run pytest -m benchmark                                     # benchmarks only
```

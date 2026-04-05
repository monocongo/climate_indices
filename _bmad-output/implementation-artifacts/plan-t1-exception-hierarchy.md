# T1 (Story 1.1): Structured Exception Hierarchy Foundation - Implementation Plan

## Summary

Story 1.1 establishes the complete structured exception hierarchy that replaces generic `ValueError` usage across all modules. This is a **critical blocker** -- nearly every downstream story depends on it.

## Current State

The exception module (`src/climate_indices/exceptions.py`) already has substantial infrastructure from v2.3.0:
- **Base class**: `ClimateIndicesError`
- **Existing exceptions**: `DistributionFittingError`, `InsufficientDataError`, `PearsonFittingError`, `DimensionMismatchError`, `CoordinateValidationError`, `InputTypeError`, `InvalidArgumentError`
- **Existing warnings**: `ClimateIndicesWarning`, `MissingDataWarning`, `ShortCalibrationWarning`, `GoodnessOfFitWarning`, `InputAlignmentWarning`, `BetaFeatureWarning`, `ClimateIndicesDeprecationWarning`
- **Helper**: `emit_deprecation_warning()`
- **Uncommitted additions** (already in working tree): `ConvergenceError`, `PeriodicityError`, `DataShapeError`

The test file (`tests/test_exceptions.py`) has 92 passing tests covering hierarchy, attributes, pickling, backward compatibility, and warning filterability.

## Gap Analysis: Architecture vs. Current Code

The architecture (Decision 4) specifies this hierarchy:

```
ClimateIndicesError (base)
├── InvalidArgumentError
│   ├── InvalidDistributionError
│   ├── InvalidScaleError
│   └── InvalidPeriodError
├── InsufficientDataError
│   ├── ShortCalibrationPeriodError
│   └── InsufficientNonZeroValuesError
└── ComputationError
    ├── DistributionFittingError
    └── ConvergenceError
```

### What needs to change

1. **Add `ComputationError`** as an intermediate class between `ClimateIndicesError` and `DistributionFittingError`/`ConvergenceError` -- this is the missing parent class specified in the architecture
2. **Reparent `DistributionFittingError`** to inherit from `ComputationError` instead of `ClimateIndicesError`
3. **Reparent `ConvergenceError`** to inherit from `ComputationError` instead of `DistributionFittingError` (it's a sibling, not a child)
4. **Add `InvalidDistributionError`** (subclass of `InvalidArgumentError`)
5. **Add `InvalidScaleError`** (subclass of `InvalidArgumentError`)
6. **Rename `PeriodicityError` to `InvalidPeriodError`** or keep both (architecture says `InvalidPeriodError`, current code has `PeriodicityError`)
7. **Restructure `InsufficientDataError`**: Architecture shows it as a direct child of `ClimateIndicesError` (not under `DistributionFittingError`), with two subclasses:
   - `ShortCalibrationPeriodError`
   - `InsufficientNonZeroValuesError`
8. **Update tests** to cover new classes, reparenting, and ensure backward compatibility
9. **Update `__all__` exports** and `__init__.py` exports

### Decision Points

**ConvergenceError reparenting**: The architecture places `ConvergenceError` under `ComputationError`, not `DistributionFittingError`. However, the existing uncommitted code has it under `DistributionFittingError`. The architecture is the canonical source, so we follow it.

**InsufficientDataError reparenting**: Currently under `DistributionFittingError`. Architecture places it directly under `ClimateIndicesError`. This is a **breaking change** for anyone catching `DistributionFittingError` expecting to get `InsufficientDataError`. We need to handle this carefully.

**PeriodicityError naming**: The architecture calls it `InvalidPeriodError`. Current code already has `PeriodicityError`. I propose keeping `PeriodicityError` as-is (it's more specific/descriptive) and it already subclasses `InvalidArgumentError` as the architecture intends. Alternatively, add `InvalidPeriodError` as an alias.

## Implementation Steps

### Step 1: Add `ComputationError` base class

Add `ComputationError` as a new intermediate class:
```python
class ComputationError(ClimateIndicesError):
    """Base exception for algorithm/computation failures."""
```

### Step 2: Reparent `DistributionFittingError`

Change from `DistributionFittingError(ClimateIndicesError)` to `DistributionFittingError(ComputationError)`.

### Step 3: Reparent `ConvergenceError`

Change from `ConvergenceError(DistributionFittingError)` to `ConvergenceError(ComputationError)` -- making it a sibling of `DistributionFittingError`, not a child.

### Step 4: Reparent `InsufficientDataError`

Change from `InsufficientDataError(DistributionFittingError)` to `InsufficientDataError(ClimateIndicesError)` with dedicated subclasses. Add:
- `ShortCalibrationPeriodError(InsufficientDataError)`
- `InsufficientNonZeroValuesError(InsufficientDataError)`

### Step 5: Add `InvalidDistributionError` and `InvalidScaleError`

```python
class InvalidDistributionError(InvalidArgumentError):
    """Raised when an unsupported distribution name is provided."""

class InvalidScaleError(InvalidArgumentError):
    """Raised when an invalid scale value is provided."""
```

### Step 6: Update `__all__` and `__init__.py`

Add all new classes to `__all__` list in `exceptions.py`. Add key classes to `__init__.py` exports (per architecture line ~2106: `ComputationError` should be a public export).

### Step 7: Update tests

- Add hierarchy tests for new classes
- Add context attribute tests for new classes
- Add pickling tests for new classes
- Add catch-all tests verifying `ComputationError` catches `DistributionFittingError` and `ConvergenceError`
- Verify backward compatibility: `ClimateIndicesError` still catches everything
- Add keyword-only enforcement tests for new classes

### Step 8: Verify with mypy and ruff

- Run `ruff check --fix` and `ruff format`
- Run `mypy src/climate_indices/exceptions.py --strict`
- Run full test suite

## Files Modified

1. `src/climate_indices/exceptions.py` -- add new classes, reparent existing ones
2. `src/climate_indices/__init__.py` -- add `ComputationError` export
3. `tests/test_exceptions.py` -- comprehensive test updates

## Risk Assessment

- **Backward compatibility**: Reparenting `InsufficientDataError` from `DistributionFittingError` to `ClimateIndicesError` means code catching `DistributionFittingError` will no longer catch `InsufficientDataError`. However, since this is still pre-release (v2.4.0), this is acceptable per the architecture decision.
- **PearsonFittingError**: Stays under `DistributionFittingError` (correct per architecture).
- **ConvergenceError attributes**: Loses `DistributionFittingError` attributes when reparented. Need to add relevant attributes directly.

## Acceptance Criteria Mapping

| Criterion | Implementation |
|-----------|---------------|
| Exception hierarchy exists with base `ClimateIndicesError` | Already exists, will extend |
| `InvalidArgumentError` defined | Already exists |
| `InsufficientDataError` defined | Already exists, will reparent |
| `ComputationError` defined | NEW - to be added |
| `DistributionFittingError` defined | Already exists, will reparent under `ComputationError` |
| Keyword-only context attributes | Already implemented, will extend to new classes |
| Actionable error messages | Will verify and improve |
| Test module validates hierarchy | Will extend with new test cases |
| mypy --strict passes | Will verify |

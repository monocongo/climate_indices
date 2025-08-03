# Floating Point Comparison Best Practices

This document provides guidelines for safe floating point comparisons in the climate_indices codebase to address code quality issue `python:S1244`.

## Problem: Direct Floating Point Equality

Direct equality checks with floating point numbers are unreliable due to precision limitations:

```python
# ❌ AVOID: Direct equality checks
if value == 0.0:
    handle_zero_case()

if array_a == array_b:
    handle_equal_arrays()

# Count zeros (problematic)
zero_count = np.sum(values == 0.0)
```

## Solution: Use Standard NumPy Functions

### 1. `numpy.isclose()` - Primary Recommendation

Use `np.isclose()` for robust floating point comparisons:

```python
import numpy as np

# ✅ GOOD: Safe zero checking
if np.isclose(value, 0.0, atol=1e-12):
    handle_zero_case()

# ✅ GOOD: Safe equality checking
if np.isclose(value_a, value_b, atol=1e-9, rtol=1e-9):
    handle_equal_values()

# ✅ GOOD: Count approximately zero values
zero_count = np.sum(np.isclose(values, 0.0, atol=1e-8))
```

### 2. `numpy.allclose()` - For Array Comparisons

Use for comparing entire arrays:

```python
# ✅ GOOD: Compare arrays
if np.allclose(expected_params, actual_params, atol=1e-8):
    arrays_are_equivalent()

# ✅ GOOD: Validate test results
np.testing.assert_allclose(expected, actual, atol=1e-6)
```

### 3. `math.isclose()` - For Scalar Values

For single scalar comparisons:

```python
import math

# ✅ GOOD: Scalar comparison
if math.isclose(param, target_value, abs_tol=1e-9):
    handle_match()
```

## Context-Specific Tolerances

Choose appropriate tolerances based on the context:

### Statistical Parameters
```python
# For fitted distribution parameters (loc, scale, skew)
np.isclose(param, 0.0, atol=1e-8)
```

### Precipitation Measurements
```python
# For precipitation values (measurement precision)
np.isclose(precip, 0.0, atol=1e-6)
```

### Probability Values
```python
# For probabilities (stricter tolerance)
np.isclose(prob, 0.0, atol=1e-10)
```

### General Comparisons
```python
# Default scientific computing tolerance
np.isclose(value, target, atol=1e-9, rtol=1e-9)
```

## Common Patterns in Climate Indices

### 1. Checking for Default Parameters

```python
# ❌ OLD: Direct equality
failed_params = np.sum(locs == 0.0)

# ✅ NEW: Safe comparison
failed_params = np.sum(np.isclose(locs, 0.0, atol=1e-8))
```

### 2. Precipitation Threshold Checks

```python
# ❌ OLD: Direct comparison
trace_precipitation = values < 0.0005

# ✅ NEW: Using numpy comparison
trace_precipitation = values < 0.0005  # This is actually OK for thresholds

# But for equality with thresholds:
# ❌ OLD: values == 0.0005
# ✅ NEW: np.isclose(values, 0.0005, atol=1e-6)
```

### 3. Validation in L-moments

```python
# ❌ OLD: Direct comparison
if lmoments[1] <= 0.0:
    raise ValueError("Invalid L-moments")

# ✅ NEW: Safe comparison (if needed for exact zero)
if lmoments[1] <= 0.0 or np.isclose(lmoments[1], 0.0, atol=1e-12):
    raise ValueError("Invalid L-moments")

# Note: For physical constraints like "must be positive", 
# direct comparison with 0.0 is often still appropriate
```

### 4. Test Assertions

```python
# ❌ OLD: Direct equality in tests
assert result == expected_value

# ✅ NEW: Tolerance-based testing
np.testing.assert_allclose(result, expected_value, atol=1e-6)

# Or for single values:
assert np.isclose(result, expected_value, atol=1e-6)
```

## Parameter Reference

### `numpy.isclose()` Parameters

- `atol` (absolute tolerance): Maximum absolute difference
- `rtol` (relative tolerance): Maximum relative difference
- Formula: `|a - b| ≤ (atol + rtol * |b|)`

### Recommended Tolerances

| Context | Absolute Tolerance | Use Case |
|---------|-------------------|----------|
| `1e-12` | Machine precision | Very strict comparisons |
| `1e-9`  | Default scientific | General computations |
| `1e-8`  | Statistical params | Distribution parameters |
| `1e-6`  | Measurements | Physical measurements |
| `1e-4`  | Loose comparison | User-facing values |

## Implementation Checklist

When updating code to use safe floating point comparisons:

1. ✅ Replace `== 0.0` with `np.isclose(value, 0.0, atol=...)`
2. ✅ Replace `!= 0.0` with `~np.isclose(value, 0.0, atol=...)`
3. ✅ Use `np.sum(np.isclose(...))` instead of `np.sum(array == value)`
4. ✅ Choose appropriate tolerance for the context
5. ✅ Update test assertions to use `np.testing.assert_allclose()`
6. ✅ Document tolerance choices in comments

## Examples from Climate Indices Codebase

### Statistical Parameter Validation
```python
# Check if fitting failed (parameters are effectively zero)
default_count = np.sum(np.isclose(fitted_params, 0.0, atol=1e-8))
if default_count > len(fitted_params) * 0.8:
    logger.warning("High failure rate in parameter fitting")
```

### Precipitation Analysis
```python
# Identify trace precipitation (very small but non-zero)
trace_mask = np.logical_and(
    values > 0.0,  # Positive values
    np.isclose(values, 0.0, atol=1e-6)  # But very close to zero
)
```

### Test Validation
```python
# Validate SPI computation results
expected_spi = load_reference_data()
computed_spi = indices.spi(precip_data, scale=3)

# Use appropriate tolerance for climate index values
np.testing.assert_allclose(
    computed_spi, expected_spi, 
    atol=1e-6, rtol=1e-6,
    err_msg="SPI computation differs from reference"
)
```

## When Direct Comparison is Still OK

Some cases where direct floating point comparison is acceptable:

1. **Physical constraints**: `if value < 0.0:` (checking sign)
2. **Thresholds**: `if precip < 0.001:` (comparing to fixed threshold)
3. **Special values**: `if np.isnan(value):` or `if np.isinf(value):`
4. **Intentional exact values**: When you specifically set `value = 0.0`

The key is avoiding equality checks (`==`, `!=`) with computed floating point results.
# Scientific Computing Patterns

## Numerical Stability

### NaN Handling
- Use `np.nan` for missing data (not `None`, not `-9999`)
- Check for NaN before computations: `np.isnan()`, `np.nansum()`
- Return NaN for invalid computations (don't raise exceptions)

### Edge Cases
- Empty arrays: Return empty array or NaN
- All-NaN input: Return NaN result
- Single value: Handle gracefully (some indices undefined)
- Zero values: Handle division carefully (`np.divide(..., where=denom!=0)`)

## Distribution Fitting

### Gamma Distribution (SPI)
```python
from scipy.stats import gamma

# Fit to non-zero, non-NaN values
valid = values[(~np.isnan(values)) & (values > 0)]
alpha, loc, beta = gamma.fit(valid, floc=0)

# CDF transform
probabilities = gamma.cdf(values, alpha, loc=loc, scale=beta)
```

### Pearson Type III (SPEI)
- More flexible than Gamma (handles negative values)
- Fallback to Gamma when fitting produces excessive NaNs
- Use L-moments for parameter estimation when MLE fails

### Fallback Strategy
```python
class DistributionFallbackStrategy(Enum):
    NO_FALLBACK = "no_fallback"
    FALLBACK_TO_GAMMA = "fallback_to_gamma"
```

## Numba Acceleration

### When to Use Numba
- Unavoidable loops over array elements
- Complex conditional logic per element
- Performance-critical hot paths

### Numba Best Practices
```python
from numba import njit

@njit(cache=True)
def _inner_loop(data: np.ndarray) -> np.ndarray:
    """Pure numerical function with no Python objects."""
    result = np.empty_like(data)
    for i in range(len(data)):
        # Fast, compiled loop
        result[i] = some_computation(data[i])
    return result
```

### Numba Limitations
- No Python objects (lists, dicts) in `@njit`
- No SciPy functions inside Numba
- Keep Numba functions small and focused

## xarray Integration

### Opening NetCDF Files
```python
import xarray as xr

# Chunked for Dask (time as single chunk for rolling operations)
ds = xr.open_dataset("data.nc", chunks={"time": -1, "lat": 50, "lon": 50})
```

### apply_ufunc Pattern
```python
result = xr.apply_ufunc(
    _compute_spi,           # NumPy function
    precip_da,              # xarray input
    kwargs={"scale": 3},
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[np.float64],
)
```

### Accessor Pattern
```python
@xr.register_dataarray_accessor("indices")
class IndicesAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def spi(self, scale: int, ...) -> xr.DataArray:
        return compute_spi(self._obj, scale, ...)
```

## Unit Conversions

### Temperature
```python
def fahrenheit_to_celsius(temp_f: np.ndarray) -> np.ndarray:
    return (temp_f - 32) * 5 / 9
```

### Precipitation
```python
def inches_to_mm(precip_inches: np.ndarray) -> np.ndarray:
    return precip_inches * 25.4
```

## Time Series Conventions

### Periodicity
| Type | Values/Year | Use Case |
|------|-------------|----------|
| Monthly | 12 | Standard climate indices |
| Daily | 366 | High-resolution analysis |

### Data Layout
- 1D: Simple time series `(time,)`
- 2D: Reshaped for periodicity `(years, periods_per_year)`
- 3D+: Gridded data `(lat, lon, time)` or `(time, lat, lon)`

# Component Inventory

## Overview

This document provides a comprehensive catalog of all modules in the `climate_indices` library, describing their responsibilities, key functions, dependencies, and usage patterns. The library consists of **14 production modules** organized into 5 architectural layers.

---

## CLI Layer (2 modules)

### `__main__.py` - Full-Featured CLI
**Location**: `src/climate_indices/__main__.py`
**Lines**: 1826
**Purpose**: Command-line interface for batch processing NetCDF datasets with all supported indices.

#### Entry Points
- **`climate_indices`**: Primary CLI command
- **`process_climate_indices`**: Alias for `climate_indices`

#### Supported Indices
- SPI (Standardized Precipitation Index)
- SPEI (Standardized Precipitation Evapotranspiration Index)
- PET (Potential Evapotranspiration - Thornthwaite)
- Palmer Drought Indices (PDSI, PHDI, PMDI, scPDSI, Z-Index)
- PNP (Percentage of Normal Precipitation)

#### Key Functions
| Function | Purpose |
|----------|---------|
| `main()` | Entry point, argument parsing, workflow orchestration |
| `process_climate_indices()` | Main processing logic |
| `_validate_args()` | Input validation and dimension checking |
| `_compute_write_index()` | Index computation and NetCDF output |
| `_parallel_process()` | Multiprocessing coordination |
| `_apply_along_axis*()` | Worker functions for parallel computation |
| `_drop_data_into_shared_arrays_*()` | Shared memory setup |

#### Parallelization Strategy
- **Method**: Multiprocessing with shared memory arrays
- **Pool Size**: `multiprocessing.cpu_count() - 1` (default)
- **Chunking**: Spatial dimensions (lat/lon or division)
- **Time Dimension**: Processed as single chunk per worker

#### Input Types Supported
1. **Gridded**: `(lat, lon, time)` or `(time, lat, lon)`
2. **Climate Divisions**: `(division, time)` or `(time, division)`
3. **Timeseries**: `(time,)`

#### Unit Conversion
- **Precipitation**: inches → mm (1 inch = 25.4 mm)
- **Temperature**: Fahrenheit → Celsius, Kelvin → Celsius
- **PET**: inches → mm

#### Example Usage
```bash
climate_indices \
    --index spi \
    --periodicity monthly \
    --scales 3 6 12 \
    --netcdf_precip precip.nc \
    --var_name_precip prcp \
    --calibration_start_year 1981 \
    --calibration_end_year 2010 \
    --output_file_base results/spi \
    --multiprocessing all_but_one
```

#### Dependencies
- Core: `numpy`, `xarray`, `scipy`
- Internal: `compute`, `indices`, `utils`

---

### `__spi__.py` - Specialized SPI CLI
**Location**: `src/climate_indices/__spi__.py`
**Lines**: 1478
**Purpose**: Specialized CLI for SPI computation with distribution fitting parameter caching.

#### Entry Point
- **`spi`**: Specialized SPI command

#### Key Features
1. **Parameter Caching**: Save/load distribution fitting parameters to/from NetCDF
2. **Reusability**: Precomputed parameters speed up operational systems
3. **Parallel Fitting**: Separate multiprocessing for fitting and transformation
4. **Both Distributions**: Gamma and Pearson Type III

#### Key Functions
| Function | Purpose |
|----------|---------|
| `main()` | Entry point with SPI-specific argument parsing |
| `_compute_write_index()` | SPI computation orchestration |
| `_parallel_fitting()` | Parallel distribution fitting |
| `_parallel_spi()` | Parallel SPI transformation |
| `_apply_to_subarray_gamma()` | Worker for gamma fitting |
| `_apply_to_subarray_pearson()` | Worker for Pearson fitting |
| `_apply_to_subarray_spi()` | Worker for SPI transformation |
| `build_dataset_fitting_*()` | Create NetCDF for fitting parameters |
| `build_dataset_spi_*()` | Create NetCDF for SPI output |

#### Parameter File Format
```python
# NetCDF with variables (per scale):
# Gamma:
#   - alpha_{scale}_{periodicity}  # shape parameter
#   - beta_{scale}_{periodicity}   # rate parameter
# Pearson:
#   - prob_zero_{scale}_{periodicity}  # probability of zero
#   - loc_{scale}_{periodicity}        # location
#   - scale_{scale}_{periodicity}      # scale
#   - skew_{scale}_{periodicity}       # skew
# Dimensions: (lat, lon, month) or (lat, lon, day)
```

#### Example Usage
```bash
# First run: compute and save parameters
spi \
    --periodicity monthly \
    --scales 3 6 12 \
    --netcdf_precip precip.nc \
    --var_name_precip prcp \
    --calibration_start_year 1981 \
    --calibration_end_year 2010 \
    --save_params fitting_params.nc \
    --output_file_base results/spi

# Subsequent runs: load parameters (faster)
spi \
    --periodicity monthly \
    --scales 3 6 12 \
    --netcdf_precip precip_new.nc \
    --var_name_precip prcp \
    --load_params fitting_params.nc \
    --output_file_base results/spi_new
```

#### Dependencies
- Core: `numpy`, `xarray`, `scipy`
- Internal: `compute`, `indices`, `utils`

---

## Public API Layer (4 modules)

### `__init__.py` - Package Exports
**Location**: `src/climate_indices/__init__.py`
**Lines**: 43
**Purpose**: Define public API surface and package metadata.

#### Exported Symbols
```python
__all__ = [
    # Version
    "__version__",

    # Modern API (xarray)
    "spi",  # from typed_public_api
    "spei",  # from typed_public_api
    "xarray_adapter",

    # xarray utilities
    "detect_input_type",
    "pet_thornthwaite",
    "pet_hargreaves",
    "InputType",

    # CF Metadata
    "CFAttributes",
    "CF_METADATA",

    # Exceptions
    "ClimateIndicesError",
    "ClimateIndicesWarning",
    "InputAlignmentWarning",

    # Logging
    "configure_logging",
]
```

#### Version Discovery
```python
try:
    __version__ = version("climate_indices")  # from importlib.metadata
except PackageNotFoundError:
    __version__ = "unknown"
```

---

### `typed_public_api.py` - Strict Mypy API (NEW in 2.2.0)
**Location**: `src/climate_indices/typed_public_api.py`
**Lines**: 210
**Purpose**: Type-safe wrappers for SPI/SPEI with strict mypy compliance.

#### Design Principles
1. **Keyword-Only Arguments**: Prevents positional argument errors
2. **Strict Mypy**: Full compliance with `mypy --strict`
3. **Runtime Validation**: Type checking at runtime
4. **Overloaded Signatures**: Separate handling for numpy vs xarray

#### Public Functions

##### `spi()`
```python
def spi(
    values: Union[np.ndarray, xr.DataArray],
    *,
    scale: int,
    distribution: Union[Distribution, str] = "gamma",
    data_start_year: Optional[int] = None,
    calibration_start_year: Optional[int] = None,
    calibration_end_year: Optional[int] = None,
    periodicity: Union[Periodicity, str] = "monthly",
) -> Union[np.ndarray, xr.DataArray]:
    """Type-safe SPI computation with strict validation."""
```

**Features**:
- Input type detection and routing
- Automatic xarray coordinate extraction
- Distribution string to enum conversion
- Comprehensive docstring with examples

##### `spei()`
```python
def spei(
    precip_mm: Union[np.ndarray, xr.DataArray],
    pet_mm: Union[np.ndarray, xr.DataArray],
    *,
    scale: int,
    distribution: Union[Distribution, str] = "gamma",
    data_start_year: Optional[int] = None,
    calibration_start_year: Optional[int] = None,
    calibration_end_year: Optional[int] = None,
    periodicity: Union[Periodicity, str] = "monthly",
) -> Union[np.ndarray, xr.DataArray]:
    """Type-safe SPEI computation with strict validation."""
```

**Features**:
- Automatic coordinate alignment for xarray inputs
- Input shape validation (matching shapes)
- Distribution string to enum conversion
- Comprehensive docstring with examples

#### Type Annotations
```python
# Uses:
from typing import Union, Optional, overload
import numpy.typing as npt

# All parameters and returns fully typed
# Compatible with mypy --strict
```

#### Dependencies
- Core: `numpy`, `xarray`
- Internal: `compute`, `indices`, `xarray_adapter`

---

### `xarray_adapter.py` - CF-Compliant xarray Interface (EXPANDED in 2.2.0)
**Location**: `src/climate_indices/xarray_adapter.py`
**Lines**: 1417 (was ~400 in 2.1.0)
**Purpose**: Modern xarray interface with CF metadata support.

#### Key Functions

##### `xarray_adapter()`
```python
def xarray_adapter(
    func: Callable,
    *args,
    input_type: InputType,
    **kwargs
) -> Union[xr.DataArray, xr.Dataset]:
    """Universal adapter wrapping numpy functions for xarray."""
```
**Purpose**: Graceful degradation pattern (xarray → numpy → xarray)

##### PET Functions

**`pet_thornthwaite()`**
```python
def pet_thornthwaite(
    temperature: Union[xr.DataArray, xr.Dataset, np.ndarray],
    latitude: Union[xr.DataArray, float, np.ndarray],
    *,
    data_start_year: Optional[int] = None,
) -> Union[xr.DataArray, xr.Dataset, np.ndarray]:
    """Monthly PET using Thornthwaite (1948) method."""
```
**Inputs**:
- Temperature in degrees Celsius
- Latitude in degrees (-90 to 90)
- Start year for calendar handling

**`pet_hargreaves()`**
```python
def pet_hargreaves(
    temperature_min: Union[xr.DataArray, xr.Dataset, np.ndarray],
    temperature_max: Union[xr.DataArray, xr.Dataset, np.ndarray],
    latitude: Union[xr.DataArray, float, np.ndarray],
    *,
    data_start_year: Optional[int] = None,
) -> Union[xr.DataArray, xr.Dataset, np.ndarray]:
    """Daily PET using Hargreaves (1985) method."""
```
**Inputs**:
- Daily minimum temperature
- Daily maximum temperature
- Latitude in degrees
- Start year for calendar handling

##### Utility Functions

**`detect_input_type()`**
```python
def detect_input_type(
    obj: Any
) -> InputType:
    """Detect input type (DataArray, Dataset, or ndarray)."""
```

**Returns**: `InputType.DATAARRAY`, `InputType.DATASET`, or `InputType.NDARRAY`

#### CF Metadata Support

**`CFAttributes` dataclass**:
```python
@dataclass
class CFAttributes:
    standard_name: Optional[str]
    long_name: str
    units: str
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    comment: Optional[str] = None
```

**`CF_METADATA` dictionary**:
```python
CF_METADATA = {
    "spi": CFAttributes(
        standard_name="standardized_precipitation_index",
        long_name="Standardized Precipitation Index",
        units="1",
        valid_min=-3.09,
        valid_max=3.09,
    ),
    # ... other indices
}
```

#### Validation Functions
- `_validate_time_coordinate()`: Monotonicity check
- `_validate_dask_chunks()`: Time chunking validation
- `_validate_calibration_period()`: Length check (≥30 years)
- `_align_inputs()`: Coordinate alignment with warnings

#### Dependencies
- Core: `numpy`, `xarray`, `dask`
- Internal: `compute`, `indices`, `exceptions`

---

### `indices.py` - Legacy NumPy API (STABLE)
**Location**: `src/climate_indices/indices.py`
**Lines**: 701
**Purpose**: Backward-compatible numpy interface.

#### Stability Guarantee
**Breaking changes require major version bump.** Function signatures MUST remain stable.

#### Public Functions

##### `spi()`
```python
def spi(
    values: np.ndarray,
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Standardized Precipitation Index."""
```

##### `spei()`
```python
def spei(
    precips_mm: np.ndarray,
    pet_mm: np.ndarray,
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
) -> np.ndarray:
    """Standardized Precipitation Evapotranspiration Index."""
```

##### `pet()`
```python
def pet(
    temperature_celsius: np.ndarray,
    latitude_degrees: Union[float, np.ndarray],
    data_start_year: int,
) -> np.ndarray:
    """Potential Evapotranspiration (Thornthwaite)."""
```

##### `percentage_of_normal()`
```python
def percentage_of_normal(
    values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: compute.Periodicity,
) -> np.ndarray:
    """Percentage of Normal Precipitation."""
```

#### Enumerations

**`Distribution` enum**:
```python
class Distribution(Enum):
    gamma = "gamma"
    pearson = "pearson"
```

#### Dependencies
- Core: `numpy`, `scipy`
- Internal: `compute`, `eto`, `exceptions`, `utils`

---

## Computation Layer (2 modules)

### `compute.py` - Mathematical Core
**Location**: `src/climate_indices/compute.py`
**Lines**: 1127
**Purpose**: Core algorithms for climate index calculation.

#### Periodicity Enum
```python
class Periodicity(Enum):
    monthly = "monthly"  # 12 time steps per year
    daily = "daily"      # 366 time steps per year

    @staticmethod
    def from_string(value: str) -> "Periodicity":
        """Parse string to Periodicity enum."""

    def unit(self) -> str:
        """Return 'month' or 'day'."""
```

#### Key Algorithms

##### Temporal Scaling
**`scale_values()`**
```python
def scale_values(
    values: np.ndarray,
    scale: int,
    periodicity: Periodicity,
) -> np.ndarray:
    """Compute rolling sums for temporal scaling."""
```
**Algorithm**: Rolling window summation with NaN propagation.

**`sum_to_scale()`**
```python
def sum_to_scale(
    values: np.ndarray,
    scale: int,
) -> np.ndarray:
    """Optimized sliding window summation."""
```
**Algorithm**: Efficient numpy implementation of rolling sum.

##### Distribution Fitting

**`gamma_parameters()`**
```python
def gamma_parameters(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit gamma distribution (method of moments)."""
```
**Returns**: `(alpha, beta)` parameters per calendar month/day.

**`pearson_parameters()`**
```python
def pearson_parameters(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit Pearson Type III (L-moments)."""
```
**Returns**: `(prob_zero, loc, scale, skew)` per calendar month/day.

##### CDF Transformation

**`transform_fitted_gamma()`**
```python
def transform_fitted_gamma(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Transform via gamma CDF → standard normal."""
```

**`transform_fitted_pearson()`**
```python
def transform_fitted_pearson(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Transform via Pearson Type III CDF → standard normal."""
```

#### Algorithm Details
1. **Scaling**: `sum_to_scale()` applies rolling window
2. **Fitting**: Fit distribution parameters per calendar month/day on calibration period
3. **CDF**: Apply `scipy.stats.gamma.cdf()` or `scipy.stats.pearson3.cdf()`
4. **Inverse Normal**: `scipy.stats.norm.ppf()` on CDF values
5. **Edge Handling**: CDF=0 → -3.09, CDF=1 → 3.09

#### Dependencies
- Core: `numpy`, `scipy.stats`
- Internal: `lmoments`, `exceptions`, `utils`

---

### `palmer.py` - Palmer Drought Indices
**Location**: `src/climate_indices/palmer.py`
**Lines**: 806
**Purpose**: Palmer Drought Index family computation.

#### Indices Computed
1. **PDSI** - Palmer Drought Severity Index
2. **PHDI** - Palmer Hydrological Drought Index
3. **PMDI** - Palmer Modified Drought Index
4. **Z-Index** - Palmer Z-Index
5. **scPDSI** - Self-calibrated Palmer

#### Key Function
```python
def palmer(
    precips_mm: np.ndarray,
    potential_evapotranspirations_mm: np.ndarray,
    available_water_capacity_inches: Union[float, np.ndarray],
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Palmer Drought Indices."""
```

**Returns**: `(scPDSI, PDSI, PHDI, PMDI, Z-Index)`

#### Required Inputs
- Monthly precipitation (mm)
- Monthly PET (mm)
- Available water capacity (inches) - soil property
- Calibration period (for coefficients)

#### Algorithm Overview
1. Compute water balance components (actual ET, runoff, soil moisture)
2. Calculate CAFEC (Climatically Appropriate For Existing Conditions) coefficients
3. Compute moisture departure (actual - expected)
4. Apply Z-Index formula
5. Compute PDSI/PHDI/PMDI via recursive tracking
6. Self-calibrate PDSI using local climate characteristics

#### Dependencies
- Core: `numpy`
- Internal: `eto`, `utils`

---

## Math/Statistics Layer (2 modules)

### `eto.py` - Potential Evapotranspiration
**Location**: `src/climate_indices/eto.py`
**Lines**: 416
**Purpose**: PET computation using Thornthwaite and Hargreaves methods.

#### Thornthwaite Method (1948)

**`potential_evapotranspiration()`**
```python
def potential_evapotranspiration(
    temperature_celsius: np.ndarray,
    latitude_degrees: Union[float, np.ndarray],
    data_start_year: int,
) -> np.ndarray:
    """Monthly PET using Thornthwaite (1948)."""
```

**Algorithm**:
1. Compute heat index: `I = sum((T_i / 5)^1.514)` for T > 0
2. Compute exponent: `a = (6.75e-7 * I^3) - (7.71e-5 * I^2) + (1.792e-2 * I) + 0.49239`
3. Compute unadjusted PET: `PET_unadj = 16 * (10 * T / I)^a`
4. Adjust for day length based on latitude and month

**Inputs**: Monthly temperature, latitude
**Output**: Monthly PET (mm)

#### Hargreaves Method (1985)

**`hargreaves()`**
```python
def hargreaves(
    tmin: np.ndarray,
    tmax: np.ndarray,
    tmean: Optional[np.ndarray],
    latitude: Union[float, np.ndarray],
    data_start_year: int,
) -> np.ndarray:
    """Daily PET using Hargreaves (1985)."""
```

**Algorithm**:
1. Compute extraterrestrial radiation `Ra` based on latitude and day of year
2. Compute temperature range: `TR = tmax - tmin`
3. Apply Hargreaves formula: `PET = 0.0023 * Ra * sqrt(TR) * (tmean + 17.8)`

**Inputs**: Daily tmin, tmax, latitude
**Output**: Daily PET (mm)

#### Helper Functions
- `_day_length_hours()`: Compute day length from latitude and day of year
- `_solar_declination()`: Solar declination angle
- `_sunset_hour_angle()`: Sunset hour angle
- `_inverse_relative_distance()`: Earth-sun distance factor
- `_extraterrestrial_radiation()`: Ra computation

#### Dependencies
- Core: `numpy`
- No internal dependencies (self-contained)

---

### `lmoments.py` - L-Moments
**Location**: `src/climate_indices/lmoments.py`
**Lines**: 94
**Purpose**: L-moments for robust Pearson Type III fitting.

#### Key Function
```python
def lmoments(data: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute L-moments (Hosking 1990 algorithm)."""
```

**Returns**: `(L1, L2, L3, L4)` - first four L-moments

#### Algorithm
Implements Hosking (1990) probability-weighted moments approach:
1. Sort data
2. Compute probability-weighted moments
3. Transform to L-moments
4. Return (location, scale, skew, kurtosis equivalents)

#### Usage
Called by `compute.pearson_parameters()` for robust parameter estimation.

#### Advantages Over Method of Moments
- More robust to outliers
- Better for skewed distributions
- Stable numerical properties

#### Dependencies
- Core: `numpy`
- No internal dependencies

---

## Infrastructure Layer (4 modules)

### `exceptions.py` - Exception Hierarchy
**Location**: `src/climate_indices/exceptions.py`
**Lines**: 324
**Purpose**: Structured error handling with context attributes.

#### Exception Hierarchy
```
ClimateIndicesError (base)
├── DistributionFittingError
│   ├── InsufficientDataError
│   └── PearsonFittingError
├── DimensionMismatchError
├── CoordinateValidationError
├── InputTypeError
└── InvalidArgumentError
```

#### Warning Hierarchy
```
ClimateIndicesWarning (base)
├── MissingDataWarning
├── ShortCalibrationWarning
├── GoodnessOfFitWarning
└── InputAlignmentWarning
```

#### Key Exception Classes

**`DistributionFittingError`**
```python
class DistributionFittingError(ClimateIndicesError):
    """Distribution fitting failure."""

    Attributes:
        distribution_name: str
        input_shape: tuple[int, ...]
        parameters: dict[str, str]
        suggestion: str
        underlying_error: Exception
```

**`InsufficientDataError`**
```python
class InsufficientDataError(DistributionFittingError):
    """Insufficient data for fitting."""

    Attributes:
        non_zero_count: int
        required_count: int  # typically 10
```

#### Usage Pattern
```python
try:
    result = spi(precip, scale=6)
except InsufficientDataError as e:
    print(f"Need {e.required_count} values, have {e.non_zero_count}")
except DistributionFittingError as e:
    print(f"{e.distribution_name} fitting failed")
    print(f"Suggestion: {e.suggestion}")
```

#### Dependencies
- Core: none (stdlib only)

---

### `logging_config.py` - Structured Logging
**Location**: `src/climate_indices/logging_config.py`
**Lines**: 76
**Purpose**: Configure structlog for library.

#### Public Function
```python
def configure_logging(
    log_level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    json_logs: bool = False,
) -> None:
    """Configure structlog for climate_indices."""
```

**Parameters**:
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `log_file`: Optional file path for log output
- `json_logs`: If True, output JSON format (for aggregators)

#### Features
- **Console logs**: Human-readable with colors
- **File logs**: JSON-formatted for parsing
- **Context binding**: Correlation IDs for tracing
- **Performance**: Minimal overhead

#### Usage
```python
from climate_indices import configure_logging
import logging

configure_logging(
    log_level=logging.DEBUG,
    log_file="climate_indices.log",
    json_logs=True
)
```

#### Dependencies
- Core: `structlog`, `logging` (stdlib)

---

### `utils.py` - Utility Functions
**Location**: `src/climate_indices/utils.py`
**Lines**: 396
**Purpose**: Cross-cutting utility functions.

#### Calendar Conversions

**`transform_to_366day()`**
```python
def transform_to_366day(
    values: np.ndarray,
    start_year: int,
    total_years: int,
) -> np.ndarray:
    """Convert Gregorian calendar to 366-day calendar."""
```
**Purpose**: Pad February with NaN in non-leap years.

**`transform_to_gregorian()`**
```python
def transform_to_gregorian(
    values_366: np.ndarray,
    start_year: int,
) -> np.ndarray:
    """Convert 366-day calendar to Gregorian."""
```
**Purpose**: Remove Feb 29 padding from non-leap years.

#### Data Validation

**`is_data_valid()`**
```python
def is_data_valid(data: Union[np.ndarray, np.ma.MaskedArray]) -> bool:
    """Check if array is valid (not all-NaN, correct type)."""
```

**`get_tolerance()`**
```python
def get_tolerance(values: np.ndarray) -> float:
    """Compute floating-point tolerance for coordinate comparison."""
```

#### Array Reshaping

**`reshape_to_2d()`**
```python
def reshape_to_2d(array: np.ndarray) -> np.ndarray:
    """Flatten to 2D for processing."""
```

**`reshape_to_divs()`**
```python
def reshape_to_divs(array: np.ndarray, divisions: int) -> np.ndarray:
    """Reshape for climate divisions format."""
```

#### Periodicity Helpers

**`gregorian_length_as_366day()`**
```python
def gregorian_length_as_366day(
    gregorian_length: int,
    start_year: int,
) -> int:
    """Convert Gregorian time length to 366-day equivalent."""
```

#### Logging Helper

**`get_logger()`**
```python
def get_logger(
    name: str,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Get configured logger instance."""
```

#### Dependencies
- Core: `numpy`, `logging` (stdlib)

---

### `performance.py` - Performance Tracking (NEW in 2.2.0)
**Location**: `src/climate_indices/performance.py`
**Lines**: 112
**Purpose**: Observability and profiling utilities.

#### Decorator

**`@measure_execution_time`**
```python
def measure_execution_time(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""
```

**Usage**:
```python
@measure_execution_time
def compute_spi(...):
    # Implementation
```

**Logs**: Function name, arguments, duration, memory usage

#### Functions

**`track_memory_usage()`**
```python
def track_memory_usage(stage: str) -> None:
    """Log current memory usage."""
```

**`log_performance_metrics()`**
```python
def log_performance_metrics(
    operation: str,
    duration: float,
    memory_mb: float,
) -> None:
    """Log performance metrics in structured format."""
```

#### Dependencies
- Core: `logging` (stdlib), `time`, `psutil`
- Internal: `logging_config`

---

## Module Dependency Matrix

| Module | Dependencies (Internal) | Dependencies (External) |
|--------|------------------------|------------------------|
| **`__main__.py`** | compute, indices, utils | numpy, xarray, scipy, multiprocessing |
| **`__spi__.py`** | compute, indices, utils | numpy, xarray, scipy, multiprocessing |
| **`typed_public_api.py`** | compute, indices, xarray_adapter | numpy, xarray |
| **`xarray_adapter.py`** | compute, indices, exceptions | numpy, xarray, dask |
| **`indices.py`** | compute, eto, exceptions, utils | numpy, scipy |
| **`compute.py`** | lmoments, exceptions, utils | numpy, scipy |
| **`palmer.py`** | eto, utils | numpy |
| **`eto.py`** | _(none)_ | numpy |
| **`lmoments.py`** | _(none)_ | numpy |
| **`exceptions.py`** | _(none)_ | _(none - stdlib only)_ |
| **`logging_config.py`** | _(none)_ | structlog |
| **`utils.py`** | _(none)_ | numpy |
| **`performance.py`** | logging_config | psutil |

---

## Usage Patterns

### Pattern 1: Modern xarray Workflow
```python
import climate_indices as ci
import xarray as xr

# Load data
precip = xr.open_dataarray("precip.nc")

# Compute SPI
spi_6 = ci.spi(
    precip,
    scale=6,
    distribution="gamma",
    calibration_start_year=1981,
    calibration_end_year=2010
)
```

### Pattern 2: Legacy numpy Workflow
```python
from climate_indices import indices, compute
import numpy as np

# Load data
precip = np.load("precip.npy")

# Compute SPI
spi_6 = indices.spi(
    precip,
    scale=6,
    distribution=indices.Distribution.gamma,
    data_start_year=1980,
    calibration_year_initial=1981,
    calibration_year_final=2010,
    periodicity=compute.Periodicity.monthly
)
```

### Pattern 3: CLI Batch Processing
```bash
climate_indices \
    --index spi \
    --scales 3 6 12 \
    --netcdf_precip precip.nc \
    --var_name_precip prcp \
    --calibration_start_year 1981 \
    --calibration_end_year 2010 \
    --output_file_base results/spi
```

---

**Next Steps**: See [development-guide.md](./development-guide.md) for development setup, [deployment-guide.md](./deployment-guide.md) for CI/CD details, and [architecture.md](./architecture.md) for design patterns.

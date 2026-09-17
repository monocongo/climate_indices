# Troubleshooting Guide

This guide helps you resolve common errors, warnings, and performance issues when using `climate_indices`. If you encounter a specific error message, use the Quick Reference table below to jump directly to the solution.

```{contents} On this page
:backlinks: none
:local: true
```

______________________________________________________________________

## Introduction

The `climate_indices` library includes comprehensive validation and error handling to catch data quality issues early. When an error occurs, the exception message typically includes:

- **What went wrong**: The specific validation that failed
- **Why it failed**: The underlying cause
- **How to fix it**: Actionable remediation steps

This guide organizes errors by category (input types, coordinates, distribution fitting, etc.) and provides working code examples for each resolution.

**Related Documentation:**

- {doc}`xarray_migration` — Migration guide with additional xarray-specific pitfalls
- {doc}`algorithms` — Algorithm documentation with calibration guidance
- {doc}`reference` — Complete API reference with exception hierarchy

______________________________________________________________________

## Quick Reference: Error Lookup Table

Use this table to quickly find the section for your error message:

```{list-table}
:header-rows: 1
:widths: 60 40
* - Error Message Fragment
  - Section
* - `Unsupported input type` / `pandas` / `DataFrame`
  - [Input Type Errors](#input-type-errors)
* - `xr.Dataset detected`
  - [Input Type Errors](#input-type-errors)
* - `Time dimension 'time' not found`
  - [Coordinate and Dimension Errors](#coordinate-and-dimension-errors)
* - `Time coordinate is not monotonically increasing`
  - [Coordinate and Dimension Errors](#coordinate-and-dimension-errors)
* - `NaT (Not-a-Time) or NaN values`
  - [Coordinate and Dimension Errors](#coordinate-and-dimension-errors)
* - `Time coordinate is empty`
  - [Coordinate and Dimension Errors](#coordinate-and-dimension-errors)
* - `Unsupported frequency` / `'W'` / `'H'`
  - [Coordinate and Dimension Errors](#coordinate-and-dimension-errors)
* - `latitude must be within [-90, 90]`
  - [Coordinate and Dimension Errors](#coordinate-and-dimension-errors)
* - `Insufficient data for scale`
  - [Data Sufficiency Errors](#data-sufficiency-errors)
* - `Calibration period contains no data points`
  - [Data Sufficiency Errors](#data-sufficiency-errors)
* - `Insufficient non-NaN data in calibration period`
  - [Data Sufficiency Errors](#data-sufficiency-errors)
* - `Gamma CDF failed` / `DistributionFittingError`
  - [Distribution Fitting Failures](#distribution-fitting-failures)
* - `Pearson CDF failed` / `PearsonFittingError`
  - [Distribution Fitting Failures](#distribution-fitting-failures)
* - `Invalid scale argument`
  - [Argument Validation Errors](#argument-validation-errors)
* - `Unsupported distribution`
  - [Argument Validation Errors](#argument-validation-errors)
* - `Invalid periodicity argument`
  - [Argument Validation Errors](#argument-validation-errors)
* - `Dimension 'time' is split across` / `chunks`
  - [Dask and Chunking Issues](#dask-and-chunking-issues)
* - `No overlapping time steps after alignment`
  - [Dask and Chunking Issues](#dask-and-chunking-issues)
* - `Prepared inputs not found` / `Input manifest not found`
  - [Notebook and Prepared-Input Workflow](#notebook-and-prepared-input-workflow)
* - `must keep time as a single chunk`
  - [Notebook and Prepared-Input Workflow](#notebook-and-prepared-input-workflow)
* - `ShortCalibrationWarning`
  - [Warnings (Non-Fatal)](#warnings-non-fatal)
* - `MissingDataWarning`
  - [Warnings (Non-Fatal)](#warnings-non-fatal)
* - `GoodnessOfFitWarning`
  - [Warnings (Non-Fatal)](#warnings-non-fatal)
* - `InputAlignmentWarning`
  - [Warnings (Non-Fatal)](#warnings-non-fatal)
```

______________________________________________________________________

## Input Type Errors

### Passing pandas Series/DataFrame

:::{warning}
**Error:** `InputTypeError: Unsupported input type: pandas.core.series.Series. Accepted types: np.ndarray, list, tuple, int, float, np.integer, np.floating, xr.DataArray. Convert using data.to_numpy()`

**Cause:** You passed a pandas Series or DataFrame instead of a NumPy array or xarray DataArray.

**Solution:** Convert pandas objects to NumPy arrays using `.to_numpy()`:

```python
import pandas as pd
from climate_indices import indices, compute

# WRONG: passing pandas Series
precip_series = pd.Series([20, 30, 15, ...])
# result = indices.spi(precip_series, ...)  # raises InputTypeError

# CORRECT: convert to numpy
precip_array = precip_series.to_numpy()
result = indices.spi(
    precip_array,
    scale=6,
    distribution=indices.Distribution.gamma,
    data_start_year=1980,
    calibration_year_initial=1980,
    calibration_year_final=2010,
    periodicity=compute.Periodicity.monthly,
)
```

Alternatively, convert pandas data to xarray for automatic parameter inference:

```python
import xarray as xr

# convert pandas Series to xarray DataArray
precip_da = xr.DataArray(
    precip_series.values,
    coords={"time": pd.date_range("1980-01", periods=len(precip_series), freq="MS")},
    dims=["time"],
)
result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.gamma)
```
:::

### Passing xr.Dataset instead of DataArray

:::{warning}
**Error:** `InputTypeError: Unsupported input type: xarray.core.dataset.Dataset. Accepted types: np.ndarray, list, tuple, int, float, np.integer, np.floating, xr.DataArray. xr.Dataset detected: Use ds['variable_name'] to select a DataArray`

**Cause:** You passed an `xr.Dataset` (collection of multiple variables) instead of an `xr.DataArray` (single variable).

**Solution:** Select a specific variable from the Dataset:

```python
import xarray as xr
from climate_indices import indices

# load NetCDF file (returns Dataset)
ds = xr.open_dataset("precipitation.nc")

# WRONG: passing Dataset
# result = indices.spi(ds, ...)  # raises InputTypeError

# CORRECT: extract specific variable
precip_da = ds["precipitation"]
result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.gamma)
```
:::

**Cross-reference:** See {doc}`xarray_migration` Pitfall 5 for more on Dataset vs DataArray.

### Mixing NumPy/xarray inputs in multi-input functions

:::{warning}
**Error:** `TypeError: daily_tmin_celsius and daily_tmax_celsius must be the same type. Got tmin=NUMPY, tmax=XARRAY. Convert both to the same type (both numpy arrays or both xr.DataArray).`

**Cause:** For functions that accept multiple inputs (e.g., `pet_hargreaves` with tmin/tmax, `spei` with precipitation/PET), all inputs must be the same type.

**Solution:** Ensure all inputs are either NumPy arrays or xarray DataArrays:

```python
import numpy as np
import xarray as xr
from climate_indices.xarray_adapter import pet_hargreaves

tmin_array = np.array([10, 12, 15, ...])
tmax_da = xr.DataArray([20, 25, 28, ...], dims=["time"])

# WRONG: mixed types
# result = pet_hargreaves(tmin_array, tmax_da, latitude=40.0)  # TypeError

# CORRECT: convert tmin to DataArray
tmin_da = xr.DataArray(tmin_array, dims=["time"])
result = pet_hargreaves(tmin_da, tmax_da, latitude=40.0)

# OR: convert tmax to numpy
tmax_array = tmax_da.values
result_np = pet_hargreaves(tmin_array, tmax_array, latitude=40.0)
```
:::

______________________________________________________________________

## Coordinate and Dimension Errors

### Missing time dimension

:::{warning}
**Error:** `CoordinateValidationError: Time dimension 'time' not found in input. Available dimensions: ['date']. Use time_dim parameter to specify custom name.`

**Cause:** Your DataArray uses a different dimension name (e.g., `date`, `timestamp`, `t`), but the function expects `time` by default.

**Solution 1:** Rename the dimension to `time` before calling the function:

```python
# if your data uses "date" instead of "time"
da_renamed = da.rename({"date": "time"})
result = indices.spi(da_renamed, scale=6, distribution=indices.Distribution.gamma)
```

**Solution 2:** Specify the custom dimension name with `time_dim` parameter (xarray functions only):

```python
from climate_indices.xarray_adapter import pet_thornthwaite

# temperature data with "date" dimension
result = pet_thornthwaite(temp_da, latitude=40.0, time_dim="date")
```
:::

**Cross-reference:** See {doc}`xarray_migration` Pitfall 1.

### Non-monotonic time coordinate

:::{warning}
**Error:** `CoordinateValidationError: Time coordinate is not monotonically increasing. Sort the data using data.sortby('time') before processing.`

**Cause:** Time values are not in chronological order (e.g., data was shuffled, merged from multiple sources, or loaded from unsorted files).

**Solution:** Sort by time coordinate before processing:

```python
da_sorted = da.sortby("time")
result = indices.spi(da_sorted, scale=6, distribution=indices.Distribution.gamma)
```
:::

**Cross-reference:** See {doc}`xarray_migration` Pitfall 2.

### NaT (Not-a-Time) in timestamps

:::{warning}
**Error:** `CoordinateValidationError: Time coordinate is not monotonically increasing. Found NaT (Not-a-Time) or NaN values. Remove invalid timestamps before processing.`

**Cause:** The time coordinate contains `NaT` (Not-a-Time) or `NaN` values, which break monotonicity checks and distribution fitting.

**Solution:** Drop invalid timestamps:

```python
import pandas as pd

# identify and drop NaT values
valid_mask = pd.notna(da.time)
da_clean = da.isel(time=valid_mask)

result = indices.spi(da_clean, scale=6, distribution=indices.Distribution.gamma)
```
:::

### Empty time coordinate

:::{warning}
**Error:** `CoordinateValidationError: Time coordinate is empty - cannot infer data_start_year`

**Cause:** The DataArray has zero time steps, possibly from aggressive filtering or incorrect slicing.

**Solution:** Check your data filtering logic:

```python
print(f"Time coordinate length: {len(da.time)}")

# if empty, check your slice/filter
da_sliced = da.sel(time=slice("1980", "2020"))
print(f"After slice: {len(da_sliced.time)} time steps")
```
:::

### Irregular time spacing

:::{warning}
**Error:** `CoordinateValidationError: Could not infer frequency from time coordinate - ensure regular spacing`

**Cause:** The time coordinate has irregular spacing (e.g., missing months, variable step sizes), preventing automatic frequency detection.

**Solution:** Resample to regular frequency or provide explicit parameters:

```python
# option 1: resample to regular monthly frequency
da_regular = da.resample(time="MS").mean()
result = indices.spi(da_regular, scale=6, distribution=indices.Distribution.gamma)

# option 2: provide explicit periodicity (numpy path only)
result_np = indices.spi(
    values_array,
    scale=6,
    distribution=indices.Distribution.gamma,
    data_start_year=1980,
    calibration_year_initial=1980,
    calibration_year_final=2010,
    periodicity=compute.Periodicity.monthly,  # explicit
)
```
:::

**Cross-reference:** See {doc}`xarray_migration` Pitfall 3.

### Unsupported frequency (weekly/hourly)

:::{warning}
**Error:** `CoordinateValidationError: Unsupported frequency 'W' - only 'MS'/'ME'/'M' (monthly) and 'D' (daily) supported`

**Cause:** Your data has weekly (`W`), hourly (`H`), quarterly (`Q`), or annual (`A`) frequency. The library only supports monthly and daily time series.

**Solution:** Resample to a supported frequency:

```python
# resample weekly to monthly using sum aggregation (for precipitation)
da_monthly = da.resample(time="MS").sum()
result = indices.spi(da_monthly, scale=6, distribution=indices.Distribution.gamma)

# for hourly to daily (e.g., temperature, use mean)
da_daily = da.resample(time="D").mean()
```

**Note:** Choose aggregation method based on variable type:

- Precipitation: `sum()`
- Temperature: `mean()`
- PET: `sum()` (for monthly) or `mean()` (for daily)
:::

**Cross-reference:** See {doc}`xarray_migration` Pitfall 3.

### Latitude out of range / NaN

:::{warning}
**Error:** `ValueError: latitude must be within [-90, 90]. Got 120.50. Check that latitude uses decimal degrees, not radians.`

**Cause:** Latitude value is outside the valid range [-90, 90], possibly because coordinates are in radians instead of degrees.

**Solution:** Convert radians to degrees if needed:

```python
import numpy as np

# if latitude is in radians
lat_radians = 0.698  # example
lat_degrees = np.degrees(lat_radians)

result = pet_thornthwaite(temp_da, latitude=lat_degrees)
```

Or check for data entry errors:

```python
# verify latitude coordinate
print(f"Latitude range: [{da.lat.min().values}, {da.lat.max().values}]")

# fix if latitude and longitude are swapped
if da.lat.max() > 90:
    da_corrected = da.rename({"lat": "lon", "lon": "lat"})
```
:::

:::{warning}
**Error:** `ValueError: latitude is NaN. Provide a valid latitude value within [-90, 90].`

**Cause:** The latitude parameter contains `NaN` values.

**Solution:** Provide valid latitude values:

```python
# fill missing latitude with interpolation or drop
lat_filled = da.lat.interpolate_na(dim="lat")
result = pet_thornthwaite(temp_da, latitude=lat_filled)
```
:::

______________________________________________________________________

## Data Sufficiency Errors

### Time series too short for scale

:::{warning}
**Error:** `InsufficientDataError: Insufficient data for scale=12: 480 time steps available, but at least 12 required.`

**Cause:** Your time series has fewer time steps than the requested scale. For example, requesting SPI-12 (12-month scale) requires at least 12 time steps.

**Solution:** Use a smaller scale or extend your time series:

```python
# check available time steps
print(f"Available time steps: {len(da.time)}")

# use smaller scale
result = indices.spi(da, scale=3, distribution=indices.Distribution.gamma)

# OR extend time series if more data is available
da_extended = xr.open_dataarray("extended_data.nc")
```
:::

### Empty calibration period

:::{warning}
**Error:** `InsufficientDataError: Calibration period (1950-1980) contains no data points. Check that calibration years overlap with time coordinate range.`

**Cause:** The specified calibration period falls completely outside your data's time range.

**Solution:** Adjust calibration period to match your data:

```python
# check data time range
print(f"Data range: {da.time.min().values} to {da.time.max().values}")

# adjust calibration years
result = indices.spi(
    da,
    scale=6,
    distribution=indices.Distribution.gamma,
    calibration_year_initial=1990,  # within data range
    calibration_year_final=2020,
)

# OR use automatic inference (xarray path)
result = indices.spi(da, scale=6, distribution=indices.Distribution.gamma)
```
:::

### Insufficient non-NaN data

:::{warning}
**Error:** `InsufficientDataError: Insufficient non-NaN data in calibration period (1980-2010). Found 245 non-NaN values (20.4 effective years), but at least 30 years of non-NaN data required for reliable distribution fitting.`

**Cause:** Your data contains too many NaN values within the calibration period, leaving fewer than 30 years of valid observations for distribution fitting.

**Solution 1:** Extend the calibration period to include more data:

```python
# option 1: extend calibration period if you have more data
result = indices.spi(
    precip_da,
    scale=6,
    distribution=indices.Distribution.gamma,
    calibration_year_initial=1950,  # earlier start
    calibration_year_final=2020,    # later end
)
```

**Solution 2:** Use only the dense portion of your dataset:

```python
# option 2: use period with fewer NaN values
dense_period = precip_da.sel(time=slice("1990", "2020"))
result = indices.spi(dense_period, scale=6, distribution=indices.Distribution.gamma)
```

**Solution 3:** Fill NaN values (use with caution):

```python
# option 3: interpolate or fill NaN (affects statistical properties!)
precip_filled = precip_da.interpolate_na(dim="time", method="linear")
result = indices.spi(precip_filled, scale=6, distribution=indices.Distribution.gamma)
```
:::

**Cross-reference:** See {doc}`xarray_migration` Pitfall 6 and {doc}`algorithms` for calibration guidance.

### Too few non-zero values for Pearson

:::{warning}
**Error:** `InsufficientDataError: Pearson Type III requires at least 4 non-zero values for L-moments computation. Found 2 non-zero values.`

**Cause:** Pearson Type III distribution uses L-moments fitting, which requires at least 4 non-zero values. This typically occurs with sparse precipitation data or very short time series.

**Solution:** Switch to Gamma distribution or extend your time series:

```python
# option 1: use Gamma distribution (more robust for sparse data)
result = indices.spi(
    precip_da,
    scale=6,
    distribution=indices.Distribution.gamma,  # instead of pearson
    calibration_year_initial=1980,
    calibration_year_final=2010,
)

# option 2: extend time series to get more non-zero values
extended_da = xr.concat([precip_da, additional_data], dim="time")
result = indices.spi(extended_da, scale=6, distribution=indices.Distribution.pearson)
```
:::

______________________________________________________________________

## Distribution Fitting Failures

### Gamma distribution fitting failed

:::{warning}
**Error:** `DistributionFittingError: Gamma CDF failed during distribution fitting`

**Cause:** The Gamma distribution could not fit your data, typically due to:

- All-zero or all-NaN values in the calibration period
- Extreme outliers causing numerical instability
- Non-positive values (Gamma requires positive data)

**Solution 1:** Try Pearson Type III distribution (more robust):

```python
from climate_indices import indices

# switch to Pearson distribution
result = indices.spi(
    precip_da,
    scale=6,
    distribution=indices.Distribution.pearson,  # instead of gamma
    calibration_year_initial=1980,
    calibration_year_final=2010,
)
```

**Solution 2:** Inspect and clean your data:

```python
# check for problematic values
print(f"Min: {precip_da.min().values}, Max: {precip_da.max().values}")
print(f"NaN count: {precip_da.isnull().sum().values}")
print(f"Zero count: {(precip_da == 0).sum().values}")

# remove outliers using quantile clipping
q99 = precip_da.quantile(0.99)
precip_clipped = precip_da.where(precip_da <= q99, q99)
result = indices.spi(precip_clipped, scale=6, distribution=indices.Distribution.gamma)
```
:::

### Pearson Type III fitting failed

:::{warning}
**Error:** `PearsonFittingError: Pearson Type III distribution fitting failed during L-moments computation`

**Cause:** L-moments fitting failed, typically due to:

- Insufficient non-zero values (< 4)
- Skewness values outside valid range
- Numerical instability with extreme data

**Solution:** Switch to Gamma distribution (uses MLE instead of L-moments):

```python
# fallback to Gamma distribution
result = indices.spi(
    precip_da,
    scale=6,
    distribution=indices.Distribution.gamma,  # more robust
    calibration_year_initial=1980,
    calibration_year_final=2010,
)
```

The library automatically includes a suggestion in the exception:

```python
try:
    result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.pearson)
except PearsonFittingError as e:
    print(f"Suggestion: {e.suggestion}")
    # prints: "Try Distribution.gamma or check for data quality issues"
```
:::

### Normal distribution PPF failed

:::{warning}
**Error:** `DistributionFittingError: Normal PPF failed during standardization`

**Cause:** The final standardization step (converting fitted distribution to Z-scores) failed, typically due to numerical issues with extreme probability values.

**Solution:** This error is rare and usually indicates severe data quality issues:

```python
# inspect probability values
import numpy as np

# check for extreme values
print(f"Data range: [{precip_da.min().values}, {precip_da.max().values}]")
print(f"Std dev: {precip_da.std().values}")

# remove extreme outliers
mean = precip_da.mean()
std = precip_da.std()
precip_filtered = precip_da.where(
    (precip_da >= mean - 5 * std) & (precip_da <= mean + 5 * std)
)
result = indices.spi(precip_filtered, scale=6, distribution=indices.Distribution.gamma)
```
:::

### Programmatic error inspection

All distribution fitting exceptions include structured metadata for programmatic handling:

```python
from climate_indices import indices
from climate_indices.exceptions import DistributionFittingError, PearsonFittingError

try:
    result = indices.spi(
        precip_da,
        scale=6,
        distribution=indices.Distribution.pearson,
        calibration_year_initial=1980,
        calibration_year_final=2010,
    )
except PearsonFittingError as e:
    print(f"Distribution: {e.distribution_name}")  # "pearson3"
    print(f"Input shape: {e.input_shape}")         # (480,)
    print(f"Suggestion: {e.suggestion}")           # "Try Distribution.gamma..."
    print(f"Underlying: {e.underlying_error}")     # original exception

    # retry with suggested distribution
    result = indices.spi(
        precip_da,
        scale=6,
        distribution=indices.Distribution.gamma,
        calibration_year_initial=1980,
        calibration_year_final=2010,
    )
except DistributionFittingError as e:
    # catch all distribution errors
    print(f"Failed with {e.distribution_name}")
    raise
```

**Cross-reference:** See {doc}`algorithms` for detailed distribution fitting guidance.

______________________________________________________________________

## Argument Validation Errors

### Invalid scale (not in [1, 72])

:::{warning}
**Error:** `InvalidArgumentError: Invalid scale argument: 100. Scale must be an integer in the range [1, 72]. Common scales: 1 (monthly), 3 (seasonal), 6 (half-year), 12 (annual).`

**Cause:** The `scale` parameter is outside the valid range [1, 72] or is not an integer.

**Solution:** Use a valid scale value:

```python
# WRONG: scale too large
# result = indices.spi(precip_da, scale=100, ...)  # raises InvalidArgumentError

# CORRECT: use valid scale
result = indices.spi(precip_da, scale=12, distribution=indices.Distribution.gamma)

# common scales
spi_1 = indices.spi(precip_da, scale=1, ...)   # 1-month
spi_3 = indices.spi(precip_da, scale=3, ...)   # 3-month (seasonal)
spi_6 = indices.spi(precip_da, scale=6, ...)   # 6-month
spi_12 = indices.spi(precip_da, scale=12, ...) # 12-month (annual)
```
:::

### Invalid distribution

:::{warning}
**Error:** `InvalidArgumentError: Unsupported distribution: gamma_str. Supported distributions: gamma, pearson. Use indices.Distribution.gamma or indices.Distribution.pearson.`

**Cause:** The `distribution` parameter is not a valid `Distribution` enum member (e.g., you passed a string instead of the enum).

**Solution:** Use the `Distribution` enum:

```python
from climate_indices import indices

# WRONG: using string
# result = indices.spi(precip_da, scale=6, distribution="gamma")  # raises InvalidArgumentError

# CORRECT: use enum
result = indices.spi(
    precip_da,
    scale=6,
    distribution=indices.Distribution.gamma,  # enum member
    calibration_year_initial=1980,
    calibration_year_final=2010,
)

# available distributions
gamma_result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.gamma)
pearson_result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.pearson)
```
:::

### Invalid periodicity

:::{warning}
**Error:** `InvalidArgumentError: Invalid periodicity argument: weekly. Periodicity must be a Periodicity enum member. Supported values: monthly, daily. Use compute.Periodicity.monthly or compute.Periodicity.daily.`

**Cause:** The `periodicity` parameter is not a valid `Periodicity` enum member.

**Solution:** Use the `Periodicity` enum:

```python
from climate_indices import indices, compute

# WRONG: using string
# result = indices.spi(values, scale=6, periodicity="monthly", ...)  # raises InvalidArgumentError

# CORRECT: use enum (numpy path only; xarray infers automatically)
result = indices.spi(
    values_array,
    scale=6,
    distribution=indices.Distribution.gamma,
    data_start_year=1980,
    calibration_year_initial=1980,
    calibration_year_final=2010,
    periodicity=compute.Periodicity.monthly,  # enum member
)

# for daily data
result_daily = indices.spi(
    daily_values,
    scale=30,
    distribution=indices.Distribution.gamma,
    data_start_year=1980,
    calibration_year_initial=1980,
    calibration_year_final=2010,
    periodicity=compute.Periodicity.daily,
)
```

**Note:** When using xarray inputs, periodicity is automatically inferred from the time coordinate frequency.
:::

______________________________________________________________________

## Dask and Chunking Issues

### Multi-chunked time dimension

:::{warning}
**Error:** `CoordinateValidationError: Dimension 'time' is split across 4 chunks. Climate index computation requires this dimension in a single chunk. Rechunk using: data = data.chunk({'time': -1})`

**Cause:** Your Dask-backed DataArray has the time dimension split into multiple chunks. SPI/SPEI require the full time series for distribution fitting, so the time dimension must be in a single chunk.

**Solution:** Rechunk the time dimension to a single chunk while keeping spatial dimensions chunked:

```python
import xarray as xr
from climate_indices import indices

# load Dask-backed data
da = xr.open_mfdataset("precip_*.nc", chunks={"time": 120, "lat": 10, "lon": 10})["precipitation"]

# WRONG: time is chunked into multiple chunks
# result = indices.spi(da, scale=6, ...)  # raises CoordinateValidationError

# CORRECT: rechunk time to single chunk, keep spatial chunks
da_rechunked = da.chunk({"time": -1, "lat": 10, "lon": 10})
result = indices.spi(da_rechunked, scale=6, distribution=indices.Distribution.gamma)
```

**Performance tip:** Use `time=-1` to consolidate all time steps into a single chunk, and adjust spatial chunks to balance memory usage vs. parallelism. Materialize the result with `scheduler="processes"` so the CPU-bound blocks run in parallel across worker processes rather than serially under the default threaded scheduler; see the Operational Guidance section of [xarray_compatibility.md](https://github.com/monocongo/climate_indices/blob/main/docs/xarray_compatibility.md#operational-guidance).
:::

**Cross-reference:** See {doc}`xarray_migration` Pitfall 5.

### No overlapping time steps after alignment

:::{warning}
**Error:** `CoordinateValidationError: Input alignment resulted in empty intersection on 'time' coordinate. Primary input and secondary inputs have no overlapping time steps. Check that your input time ranges overlap.`

**Cause:** For multi-input functions (e.g., `spei` with precipitation and PET), the inputs have no overlapping time steps after alignment.

**Solution:** Ensure inputs cover the same time range:

```python
import xarray as xr
from climate_indices import indices

# check time ranges
print(f"Precip: {precip_da.time.min().values} to {precip_da.time.max().values}")
print(f"PET: {pet_da.time.min().values} to {pet_da.time.max().values}")

# WRONG: no overlap
# precip: 1980-2000, PET: 2010-2020
# result = indices.spei(precip_da, pet_da, scale=6, ...)  # raises CoordinateValidationError

# CORRECT: ensure overlap
common_start = max(precip_da.time.min().values, pet_da.time.min().values)
common_end = min(precip_da.time.max().values, pet_da.time.max().values)

precip_aligned = precip_da.sel(time=slice(common_start, common_end))
pet_aligned = pet_da.sel(time=slice(common_start, common_end))

result = indices.spei(
    precip_aligned,
    pet_aligned,
    scale=6,
    distribution=indices.Distribution.gamma,
)
```
:::

______________________________________________________________________

## Notebook and Prepared-Input Workflow

The xarray/Dask tutorial
[notebooks/zarr_dask_spi_spei.ipynb](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb)
runs against a prepared sample store instead of ad-hoc inputs, so its failures
are mostly setup problems. Check this section before changing the notebook.

### Running the tutorial

One command prepares the pinned sample inputs (cached after the first run) and
executes the notebook from a fresh kernel:

```bash
bash scripts/smoke_e2e_notebook.sh
```

Interactively, run `uv sync --group dev` first, then **Restart Kernel → Run
All**. The smoke command executes into a scratch directory, so it never
overwrites the committed notebook.

### Missing prepared inputs

:::{warning}
**Error:** `FileNotFoundError: Prepared inputs not found: ../data/e2e/current. Generate them once with: uv run --group dev scripts/prepare_e2e_inputs.py`

**Cause:** `data/e2e/` is gitignored and no generation has been published
yet, or `CLIMATE_INDICES_E2E_DATA` points at a directory without a
`current` entry.

**Solution:** From the repository root:

```bash
uv run --group dev scripts/prepare_e2e_inputs.py
```

Then restart the kernel and rerun. The default `../data/e2e` resolves
against the kernel's working directory, which is `notebooks/` for the
documented launch commands; a kernel started elsewhere moves that default.
`CLIMATE_INDICES_E2E_DATA` overrides it (the test suite uses that
override), so set it to an absolute path when running from another
directory.
:::

### Input preparation and checksum failures

The preparation script downloads about 5 MB of pinned source NetCDF and
verifies each file's SHA-256 before use, so a stale or corrupted cache fails
loudly instead of feeding wrong values into the tutorial. Reruns need no
network access once `data/e2e/source/` is populated; delete that directory
and rerun when a checksum error names a file you did not modify. See
[docs/research/nclimgrid-acquisition-and-redistribution.md](https://github.com/monocongo/climate_indices/blob/main/docs/research/nclimgrid-acquisition-and-redistribution.md)
for the source provenance and attribution constraints.

### Input contract mismatches

The notebook validates the prepared store before calculating, and each failure
names the contract it enforces:

- **Coordinates:** precipitation and PET must be exactly aligned before SPEI;
  partial overlap is not enough. See
  [Coordinate and Dimension Errors](#coordinate-and-dimension-errors) and
  {doc}`xarray_migration`.
- **Units:** inputs must be monthly totals in millimeters declared as
  `units == "mm"`. Relabeling the attribute is not a conversion; convert the
  values upstream instead.
- **Calibration Period:** `data_start_year` and the calibration years in the
  notebook's `pipeline_config` must match the prepared store and its
  `manifest.json`. See [Empty calibration period](#empty-calibration-period) and
  [ShortCalibrationWarning](#shortcalibrationwarning).
- **Missing data:** missing values must be `NaN`, missingness must match
  between precipitation and PET, and no cell may be partially missing across
  the time series, because distribution fitting needs one complete series per
  location. See [Insufficient non-NaN data](#insufficient-non-nan-data).
- **All-NaN results:** an all-NaN cell means the location never had enough
  valid data to fit; the notebook's guard treats those cells as unavailable
  instead of failing the whole grid.

### Time chunking

:::{warning}
**Error:** `CoordinateValidationError: Prepared store must keep time as a single chunk.`

**Cause:** spatial chunks parallelize independently, but distribution
fitting needs each location's full series, so `time` must be one chunk in
both the prepared store and the calculation input. See
[Multi-chunked time dimension](#multi-chunked-time-dimension) and
[docs/adr/0003-dask-time-dimension-single-chunk.md](https://github.com/monocongo/climate_indices/blob/main/docs/adr/0003-dask-time-dimension-single-chunk.md).

**Solution:** rechunk with `ds.chunk({"time": -1})` before calculating
(as the notebook does) and rewrite persistent stores with the complete
`time` dimension in a single chunk. Spatial chunk sizes are free to vary.
:::

### Apparent lack of parallelism

Dask parallelizes over spatial chunks, not over time. A grid no larger than
one spatial chunk produces a single task no matter how many workers exist; the
tutorial's 38x87 grid with 10x10 chunks yields roughly 40 spatial blocks, and
worker count alone is not evidence of parallelism. Inspect the chunk layout and
task graph (the notebook does) and expect scheduling overhead, not a speedup,
for a small sample. See [Dask and Chunking Issues](#dask-and-chunking-issues) and
[Performance Tuning](#performance-tuning).

### Optional dependencies

The notebook disables the Dask dashboard so the optional `bokeh` dependency
stays optional, and its plot cells need `matplotlib` from the `dev` group.
A minimum-dependency environment can execute the calculation cells without
plots, which is what `tests/test_e2e_with_dask.py` does against a synthetic
store.

______________________________________________________________________

## Warnings (Non-Fatal)

Warnings indicate potential data quality issues but do not prevent computation. You can choose to address them or suppress them if they're expected for your use case.

### ShortCalibrationWarning

**Warning:** `ShortCalibrationWarning: Calibration period is 25 years, shorter than the recommended minimum of 30 years. Distribution parameters may be less stable.`

**Cause:** Your calibration period is shorter than 30 years, which may not capture the full range of climate variability.

**Impact:** Distribution parameters may be less stable, affecting the reliability of standardized indices.

**Solution 1:** Extend the calibration period:

```python
# extend to at least 30 years
result = indices.spi(
    precip_da,
    scale=6,
    distribution=indices.Distribution.gamma,
    calibration_year_initial=1980,
    calibration_year_final=2010,  # 31 years
)
```

**Solution 2:** Suppress if intentional (e.g., short historical record):

```python
import warnings
from climate_indices.exceptions import ShortCalibrationWarning

# suppress specific warning
warnings.filterwarnings("ignore", category=ShortCalibrationWarning)

result = indices.spi(
    precip_da,
    scale=6,
    distribution=indices.Distribution.gamma,
    calibration_year_initial=1995,
    calibration_year_final=2020,  # 26 years - will warn but proceed
)
```

### MissingDataWarning

**Warning:** `MissingDataWarning: Calibration period has 25% missing data (threshold: 20%). Distribution fitting may be less reliable.`

**Cause:** More than 20% of values in the calibration period are NaN, which can affect distribution parameter estimates.

**Impact:** Distribution fitting may be less reliable, potentially affecting index values.

**Solution 1:** Fill or interpolate missing values (use with caution):

```python
# interpolate missing values
precip_filled = precip_da.interpolate_na(dim="time", method="linear", limit=3)
result = indices.spi(precip_filled, scale=6, distribution=indices.Distribution.gamma)
```

**Solution 2:** Use a different calibration period with less missing data:

```python
# assess missing data by period
for start_year in range(1950, 2000, 10):
    end_year = start_year + 30
    period_data = precip_da.sel(time=slice(str(start_year), str(end_year)))
    missing_pct = period_data.isnull().mean().values * 100
    print(f"{start_year}-{end_year}: {missing_pct:.1f}% missing")

# use period with lowest missing data
```

**Solution 3:** Suppress if expected:

```python
import warnings
from climate_indices.exceptions import MissingDataWarning

warnings.filterwarnings("ignore", category=MissingDataWarning)
```

### GoodnessOfFitWarning

**Warning:** `GoodnessOfFitWarning: Fitted gamma distribution shows poor fit quality for 120 out of 480 time steps (Kolmogorov-Smirnov p-value < 0.05). Index values may be less reliable for these periods.`

**Cause:** Kolmogorov-Smirnov goodness-of-fit tests indicate the fitted distribution doesn't adequately represent the empirical data for some time steps.

**Impact:** Index values may be less reliable for time steps with poor fit.

**Solution 1:** Try a different distribution:

```python
# if Gamma shows poor fit, try Pearson
result = indices.spi(
    precip_da,
    scale=6,
    distribution=indices.Distribution.pearson,  # instead of gamma
    calibration_year_initial=1980,
    calibration_year_final=2010,
)
```

**Solution 2:** Inspect and address data quality:

```python
# check for outliers or anomalous periods
precip_da.plot()

# remove outliers
q01, q99 = precip_da.quantile([0.01, 0.99])
precip_clipped = precip_da.clip(min=q01, max=q99)
```

**Solution 3:** Suppress if acceptable:

```python
import warnings
from climate_indices.exceptions import GoodnessOfFitWarning

warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)
```

### InputAlignmentWarning

**Warning:** `InputAlignmentWarning: Input alignment dropped 12 time step(s) from primary input. Original size: 492, aligned size: 480. Computation will use only the intersection of input time ranges.`

**Cause:** When aligning multiple inputs (e.g., precipitation and PET for SPEI), some time steps were dropped because they don't exist in all inputs.

**Impact:** Computation uses only the intersection of input time ranges, potentially reducing the output time series length.

**Solution 1:** Ensure inputs have identical time coordinates before calling the function:

```python
# align inputs explicitly first
precip_aligned, pet_aligned = xr.align(precip_da, pet_da, join="inner")

print(f"Precip: {len(precip_aligned.time)} time steps")
print(f"PET: {len(pet_aligned.time)} time steps")

result = indices.spei(precip_aligned, pet_aligned, scale=6, distribution=indices.Distribution.gamma)
```

**Solution 2:** Suppress if expected (e.g., intentionally using different time ranges):

```python
import warnings
from climate_indices.exceptions import InputAlignmentWarning

warnings.filterwarnings("ignore", category=InputAlignmentWarning)
```

### Suppressing all library warnings

To suppress all `climate_indices` warnings at once, filter by the base warning class:

```python
import warnings
from climate_indices.exceptions import ClimateIndicesWarning

# suppress all library warnings
warnings.filterwarnings("ignore", category=ClimateIndicesWarning)

# now all ShortCalibrationWarning, MissingDataWarning, etc. are suppressed
result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.gamma)
```

______________________________________________________________________

## Performance Tuning

### When to use Dask

Use Dask-backed xarray DataArrays when your dataset doesn't fit in memory or when you need parallel computation across spatial dimensions.

**Use Dask when:**

- Dataset size > available RAM (e.g., global gridded data at fine resolution)
- Computing indices for large spatial grids (e.g., 1000+ grid points)
- Processing multiple files that together exceed memory

**Example:**

```python
import xarray as xr
from climate_indices import indices

# load large dataset with Dask chunking
da = xr.open_mfdataset(
    "precip_*.nc",
    chunks={"time": -1, "lat": 50, "lon": 50},  # time in single chunk, spatial chunked
    parallel=True,
)["precipitation"]

# compute SPI in parallel across spatial chunks
result = indices.spi(da, scale=6, distribution=indices.Distribution.gamma)

# write to disk (triggers computation)
result.to_netcdf("spi_6.nc")
```

**Don't use Dask when:**

- Working with 1-D time series (single location)
- Small grids (< 100 spatial points)
- Dataset fits comfortably in memory (< 50% of available RAM)

### Chunking strategy

For climate indices, the optimal chunking strategy is:

- **Time dimension:** Single chunk (`time=-1`) — required for distribution fitting
- **Spatial dimensions:** Chunked for parallelism (`lat=10-100`, `lon=10-100`)

**Example:**

```python
# good chunking for climate indices
da_chunked = da.chunk({"time": -1, "lat": 50, "lon": 50})

# bad chunking: time split across chunks
# da_bad = da.chunk({"time": 120, "lat": 50, "lon": 50})  # will raise error
```

**Chunk size guidelines:**

- Each chunk should be 10-100 MB for optimal performance
- For a 480-timestep, float64 array: `480 * 8 bytes * lat * lon = chunk_size`
- Example: `480 * 8 * 50 * 50 = 9.6 MB` per chunk (good)

### Memory management

For large datasets, manage memory carefully to avoid crashes:

**Strategy 1:** Write results to disk incrementally:

```python
import xarray as xr

# process each variable separately
ds = xr.open_mfdataset("data_*.nc", chunks={"time": -1, "lat": 50, "lon": 50})

spi_3 = indices.spi(ds["precipitation"], scale=3, distribution=indices.Distribution.gamma)
spi_3.to_netcdf("output/spi_3.nc")  # write and free memory
del spi_3

spi_6 = indices.spi(ds["precipitation"], scale=6, distribution=indices.Distribution.gamma)
spi_6.to_netcdf("output/spi_6.nc")
del spi_6
```

**Strategy 2:** Monitor memory usage:

```python
import psutil

def check_memory():
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.percent}% used ({mem.available / 1e9:.1f} GB available)")

check_memory()
result = indices.spi(large_da, scale=6, distribution=indices.Distribution.gamma)
check_memory()
```

**Strategy 3:** Lazy evaluation with `.persist()`:

```python
# compute and cache in distributed memory (if using Dask cluster)
da_persisted = da.persist()

# now subsequent operations are fast
spi_3 = indices.spi(da_persisted, scale=3, distribution=indices.Distribution.gamma)
spi_6 = indices.spi(da_persisted, scale=6, distribution=indices.Distribution.gamma)
```

### CLI --chunksizes option

The `process_climate_indices` CLI supports custom chunking via the `--chunksizes` option:

```bash
# default chunking (automatic)
process_climate_indices --index spi --periodicity monthly --netcdf_precip precip.nc \
    --var_name_precip precipitation --output_file_base spi

# custom chunking: single time chunk, 100x100 spatial chunks
process_climate_indices --index spi --periodicity monthly --netcdf_precip precip.nc \
    --var_name_precip precipitation --output_file_base spi \
    --chunksizes time:-1 lat:100 lon:100

# for small grids, disable chunking entirely
process_climate_indices --index spi --periodicity monthly --netcdf_precip precip.nc \
    --var_name_precip precipitation --output_file_base spi \
    --chunksizes time:-1 lat:-1 lon:-1
```

**Chunking format:** `dimension:size` where `size=-1` means "single chunk" and `size=N` means "chunks of size N".

### When NOT to use Dask

Dask adds overhead, so avoid it for small datasets:

```python
import xarray as xr
from climate_indices import indices

# small 1-D time series: use in-memory (no chunks)
precip_small = xr.open_dataarray("station_precip.nc")  # no chunks= argument

# this will use fast in-memory computation
result = indices.spi(precip_small, scale=6, distribution=indices.Distribution.gamma)

# for small grids, load into memory explicitly
precip_grid = xr.open_dataarray("regional_precip.nc", chunks={"time": -1, "lat": 10, "lon": 10})
precip_in_memory = precip_grid.compute()  # load to memory

# now compute without Dask overhead
result = indices.spi(precip_in_memory, scale=6, distribution=indices.Distribution.gamma)
```

______________________________________________________________________

## Logging and Debugging

### Enable DEBUG logging

Enable detailed logging to diagnose issues:

```python
from climate_indices.logging_config import configure_logging

# enable DEBUG level (default is INFO)
configure_logging(log_level="DEBUG")

# now all operations will log detailed information
from climate_indices import indices
result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.gamma)
```

**Using environment variable:**

```bash
# set log level via environment variable
export CLIMATE_INDICES_LOG_LEVEL=DEBUG

# run your Python script
python my_script.py
```

### JSON log format

For production environments or log aggregation (e.g., Elasticsearch, Splunk), use JSON format:

```python
from climate_indices.logging_config import configure_logging

# emit structured JSON logs
configure_logging(log_format="json", log_level="INFO")

# logs are now JSON-formatted, e.g.:
# {"event": "xarray_adapter_completed", "function_name": "spi", "input_shape": [480], ...}
```

### Key logging events

The library emits structured log events for important operations:

```{list-table}
:header-rows: 1
:widths: 30 70
* - Event
  - Description
* - `parameters_inferred`
  - Temporal parameters inferred from time coordinate
* - `nan_detected_in_input`
  - NaN values found in input data
* - `input_alignment_dropped_data`
  - Time steps dropped during multi-input alignment
* - `calculation_started`
  - Climate index calculation started
* - `calculation_completed`
  - Climate index calculation completed successfully
* - `calculation_failed`
  - Climate index calculation failed with error
* - `time_dimension_missing`
  - Time dimension not found in DataArray
* - `time_coordinate_not_monotonic`
  - Time coordinate is not monotonically increasing
* - `multi_chunked_dimension`
  - Dimension required in a single chunk split across multiple Dask chunks
* - `insufficient_data_for_scale`
  - Time series too short for requested scale
```

**Example DEBUG output:**

```text
2026-02-10T15:30:45.123Z event='parameters_inferred' function_name='spi' data_start_year='1980' periodicity='monthly' calibration_year_initial='1980' calibration_year_final='2020'
2026-02-10T15:30:45.234Z event='nan_detected_in_input' function_name='spi' nan_count=45 nan_ratio=0.0938 total_values=480
2026-02-10T15:30:48.567Z event='xarray_adapter_completed' function_name='spi' input_shape=(480,) output_shape=(480,) inferred_params=True
```

______________________________________________________________________

## Exception Hierarchy Reference

Understanding the exception hierarchy helps you catch errors at the appropriate level:

```text
ClimateIndicesError (base exception)
├── DistributionFittingError
│   ├── InsufficientDataError
│   └── PearsonFittingError
├── DimensionMismatchError
├── CoordinateValidationError
├── InputTypeError
└── InvalidArgumentError

ClimateIndicesWarning (base warning)
├── MissingDataWarning
├── ShortCalibrationWarning
├── GoodnessOfFitWarning
└── InputAlignmentWarning
```

### Catching all library errors

```python
from climate_indices import indices
from climate_indices.exceptions import ClimateIndicesError

try:
    result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.gamma)
except ClimateIndicesError as e:
    # catches all library-specific errors
    print(f"Climate indices error: {e}")
    # handle or re-raise
```

### Catching specific error categories

```python
from climate_indices.exceptions import (
    DistributionFittingError,
    CoordinateValidationError,
    InputTypeError,
)

try:
    result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.gamma)
except DistributionFittingError as e:
    # retry with different distribution
    result = indices.spi(precip_da, scale=6, distribution=indices.Distribution.pearson)
except CoordinateValidationError as e:
    # fix coordinate issues
    precip_fixed = precip_da.sortby("time")
    result = indices.spi(precip_fixed, scale=6, distribution=indices.Distribution.gamma)
except InputTypeError as e:
    # convert input type
    precip_array = precip_da.values
    result = indices.spi(
        precip_array,
        scale=6,
        distribution=indices.Distribution.gamma,
        data_start_year=1980,
        calibration_year_initial=1980,
        calibration_year_final=2010,
        periodicity=compute.Periodicity.monthly,
    )
```

**Cross-reference:** See {doc}`reference` for complete exception API documentation.

______________________________________________________________________

## Getting Help

If this guide doesn't resolve your issue:

**Report a bug or request help:**

- GitHub Issues: <https://github.com/monocongo/climate_indices/issues>

- Search existing issues first: <https://github.com/monocongo/climate_indices/issues?q=is%3Aissue>

- When reporting, include:

  - Full error message and traceback
  - Minimal code example that reproduces the issue
  - Data characteristics (shape, time range, frequency)
  - Library version: `python -c "import climate_indices; print(climate_indices.__version__)"`

**Contributing:**

- Contributing guidelines: <https://github.com/monocongo/climate_indices/blob/main/CONTRIBUTING.md>
- Pull requests welcome for bug fixes and documentation improvements

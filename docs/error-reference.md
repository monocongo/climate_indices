# Error and Exception Reference

Look up the exception a call raised, then follow its category to the
symptom-to-fix recipe in {doc}`troubleshooting`. The exception classes
themselves are documented in {doc}`reference`.

```{contents} On this page
:backlinks: none
:local: true
```

______________________________________________________________________

## Error Lookup Table

Use this table to find the section of {doc}`troubleshooting` that covers your
error message:

```{list-table}
:header-rows: 1
:widths: 60 40
* - Error Message Fragment
  - Section
* - `Unsupported input type` / `pandas` / `DataFrame`
  - [Input Type Errors](troubleshooting.md#input-type-errors)
* - `xr.Dataset detected`
  - [Input Type Errors](troubleshooting.md#input-type-errors)
* - `Time dimension 'time' not found`
  - [Coordinate and Dimension Errors](troubleshooting.md#coordinate-and-dimension-errors)
* - `Time coordinate is not monotonically increasing`
  - [Coordinate and Dimension Errors](troubleshooting.md#coordinate-and-dimension-errors)
* - `NaT (Not-a-Time) or NaN values`
  - [Coordinate and Dimension Errors](troubleshooting.md#coordinate-and-dimension-errors)
* - `Time coordinate is empty`
  - [Coordinate and Dimension Errors](troubleshooting.md#coordinate-and-dimension-errors)
* - `Unsupported frequency` / `'W'` / `'H'`
  - [Coordinate and Dimension Errors](troubleshooting.md#coordinate-and-dimension-errors)
* - `latitude must be within [-90, 90]`
  - [Coordinate and Dimension Errors](troubleshooting.md#coordinate-and-dimension-errors)
* - `Invalid shape of input array` / `DataShapeError`
  - [Input Shape Errors](troubleshooting.md#input-shape-errors)
* - `Insufficient data for scale`
  - [Data Sufficiency Errors](troubleshooting.md#data-sufficiency-errors)
* - `Calibration period contains no data points`
  - [Data Sufficiency Errors](troubleshooting.md#data-sufficiency-errors)
* - `Insufficient non-NaN data in calibration period`
  - [Data Sufficiency Errors](troubleshooting.md#data-sufficiency-errors)
* - `Gamma CDF failed` / `DistributionFittingError`
  - [Distribution Fitting Failures](troubleshooting.md#distribution-fitting-failures)
* - `Pearson CDF failed` / `PearsonFittingError`
  - [Distribution Fitting Failures](troubleshooting.md#distribution-fitting-failures)
* - `Invalid scale argument`
  - [Argument Validation Errors](troubleshooting.md#argument-validation-errors)
* - `Unsupported distribution`
  - [Argument Validation Errors](troubleshooting.md#argument-validation-errors)
* - `Invalid periodicity argument`
  - [Argument Validation Errors](troubleshooting.md#argument-validation-errors)
* - `Dimension 'time' is split across` / `chunks`
  - [Dask and Chunking Issues](troubleshooting.md#dask-and-chunking-issues)
* - `No overlapping time steps after alignment`
  - [Dask and Chunking Issues](troubleshooting.md#dask-and-chunking-issues)
* - `Prepared inputs not found` / `Input manifest not found`
  - [Notebook and Prepared-Input Workflow](troubleshooting.md#notebook-and-prepared-input-workflow)
* - `must keep time as a single chunk`
  - [Notebook and Prepared-Input Workflow](troubleshooting.md#notebook-and-prepared-input-workflow)
* - `ShortCalibrationWarning`
  - [Warnings (Non-Fatal)](troubleshooting.md#warnings-non-fatal)
* - `MissingDataWarning`
  - [Warnings (Non-Fatal)](troubleshooting.md#warnings-non-fatal)
* - `GoodnessOfFitWarning`
  - [Warnings (Non-Fatal)](troubleshooting.md#warnings-non-fatal)
* - `InputAlignmentWarning`
  - [Warnings (Non-Fatal)](troubleshooting.md#warnings-non-fatal)
```

______________________________________________________________________

## Exception Hierarchy Reference

Understanding the exception hierarchy helps you catch errors at the appropriate level:

```text
ClimateIndicesError (base exception)
├── DistributionFittingError
│   ├── InsufficientDataError
│   └── PearsonFittingError
├── CoordinateValidationError
│   └── DimensionMismatchError
├── DataShapeError
├── InputTypeError
└── InvalidArgumentError
    └── PeriodicityError

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

**Cross-reference:** See {doc}`reference` for the complete exception API documentation.

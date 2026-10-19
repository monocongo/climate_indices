# Reference

API, command-line, formula, and error-lookup material. For background on the
indices themselves, see {doc}`explanation`.

```{contents}
:backlinks: none
:local: true
```

```{toctree}
:maxdepth: 1

algorithm-reference
rust-numerical-contract
data_requirements
error-reference
xarray_compatibility
research/nclimgrid-acquisition-and-redistribution
deprecations/index
release-notes-3.0.0
```

## Scientific validation and vocabulary

The validation record — what is verified against which reference, and the known
gaps — is in
[VALIDATION.md](https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md).
The core-library vocabulary these pages and the code use is defined in
[src/climate_indices/CONTEXT.md](https://github.com/monocongo/climate_indices/blob/main/src/climate_indices/CONTEXT.md).

## Command-line interface

The installation provides the `climate_indices` command, with
`process_climate_indices` retained as an alias. It computes one or more climate
indices from gridded NetCDF datasets, single-location time-series NetCDF
datasets, and US climate division NetCDF datasets. Palmers and 'all' require
available water capacity, which is supported only for gridded and US climate
division data, so they cannot be computed from single-location time-series
input. (The separate `spi` script
was removed in 3.0.0 -- see
{doc}`deprecations/api-changes`.)

The command is run from a bash shell, i.e.

`$ climate_indices <options>`

The options are described below:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Option
  - Description
* - index
  - Which of the climate indices to compute (required). Valid values are 'spi', 'spei', 'pnp', 'scaled', 'pet', 'palmers', 'kbdi', 'pe', 'edi', 'flood_index', 'api', and 'all'. 'scaled' indicates all three scaled indices (SPI, SPEI, and PNP) and 'palmers' indicates all Palmer indices (PDSI, PHDI, PMDI, Z-Index, scPDSI). KBDI and the flood indices (effective precipitation 'pe', the Effective Drought Index 'edi', the Flood Index 'flood_index', and the Antecedent Precipitation Index 'api') are only available via their own values, not 'all'. The flood indices show flood potential, not flooding.
* - periodicity
  - The periodicity of the input dataset files (required). Valid values are 'monthly' and 'daily'.

    **NOTE**: SPI, SPEI (with a PET input), PNP, and KBDI accept daily inputs; KBDI and the flood indices require daily inputs. Palmers and 'all' require monthly inputs.
* - netcdf_precip
  - Input NetCDF file containing a precipitation dataset, required for all indices except for PET, and except for EDI and the Flood Index when **netcdf_pe** is provided instead. Requires the use of **var_name_precip** in conjunction so as to identify the NetCDF's precipitation variable.
* - var_name_precip
  - Name of the precipitation variable within the input precipitation NetCDF.
* - netcdf_temp
  - Input NetCDF file containing a temperature dataset, required for PET. For KBDI this dataset must contain daily maximum temperature. If specified in conjunction with an index specification of SPEI, Palmers, 'scaled', or 'all' then PET will be computed and written as a side effect, since these indices require PET. This option is mutually exclusive with **netcdf_pet/var_name_pet**, as either temperature or PET is required as an input (but not both) when computing SPEI, Palmers, 'scaled', or 'all'. Requires the use of **var_name_temp** in conjunction so as to identify the NetCDF's temperature variable.
* - var_name_temp
  - Name of the temperature variable within the input temperature NetCDF.
* - netcdf_pet
  - Input NetCDF file containing a PET dataset. SPEI, Palmers, 'scaled', and 'all' require either this option or **netcdf_temp/var_name_temp** as a PET source, so the two are mutually exclusive (provide exactly one of them). Requires the use of **var_name_pet** in conjunction so as to identify the NetCDF's PET variable.
* - var_name_pet
  - Name of the PET variable within the input PET NetCDF.
* - netcdf_awc
  - Input NetCDF file containing available water capacity, required for Palmers and 'all'. Requires the use of **var_name_awc** in conjunction so as to identify the NetCDF's AWC variable.
* - var_name_awc
  - Name of the available water capacity variable within the input AWC NetCDF.
* - netcdf_pe
  - Input NetCDF file containing effective precipitation (PE), as written by **index** 'pe'. Applicable only when **index** is 'edi' or 'flood_index', which otherwise compute PE from **netcdf_precip** and write it as `<output_file_base>_pe.nc`. This option is mutually exclusive with **netcdf_precip**, since there is no way to determine which to use (provide exactly one of them). Requires the use of **var_name_pe** in conjunction so as to identify the NetCDF's PE variable.
* - var_name_pe
  - Name of the effective precipitation variable within the input PE NetCDF.
* - year_start_month
  - Calendar month (1-12) on which each year of annual maxima starts. Required when **index** is 'flood_index', and applicable only then.
* - api_k
  - Daily decay constant of the Antecedent Precipitation Index, strictly between 0 and 1. Required when **index** is 'api', and applicable only then.
* - kbdi_units
  - Units of the KBDI input and output values. Valid values are 'metric' (millimeters and degrees Celsius; default) and 'imperial' (inches and degrees Fahrenheit, with output in hundredths of an inch). Applicable only when **index** is 'kbdi'.
* - kbdi_initial
  - Initial KBDI value. Default value is 0.0. Applicable only when **index** is 'kbdi'.
* - chunksizes
  - Chunking of the written output file, not of the computation: 'none' (default) lets the writer choose the output layout, and 'input' uses the on-disk chunks of the first input variable that reports chunk sizes without being explicitly contiguous (for KBDI, PE, and API, the precipitation variable's chunks; for EDI and the Flood Index, the PE variable's). The copied chunks are reordered to the output's dimension order and trimmed to its shape, and are ignored, with a warning, when that variable's dimensions don't match the output's.
* - pack
  - EXPERIMENTAL, off by default. Writes each floating-point output variable zlib-compressed: as int16 with a 1e-4 **scale_factor** when every value lies within +/-3.2767, at most 5e-5 from the computed value, and as float32 otherwise, including a variable holding +/-inf. Without it the output keeps its default float64 encoding. A PET or PE file that a later index in the same run reads is written unpacked, so the flag never changes computed values. A Dask-backed output is computed twice, once to range its values and once to write them.
* - output_file_base
  - Base file name for all output files (required).

    Each computed index will have a corresponding output file whose name will begin with this base name plus the index's abbreviation plus a timescale (if applicable), connected with underscores, plus the '.nc' extension. For example for SPI at 3-month scale the resulting output files will be named **<output_file_base>_spi_gamma_03.nc** and **<output_file_base>_spi_pearson_03.nc**. A non-default **output_scale** adds its name to the file and variable, for example **<output_file_base>_spi_gamma_03_probability.nc**.
* - scales
  - Time step scales over which the PNP, SPI, and SPEI values are to be computed. Required when the **index** argument is 'spi', 'spei', 'pnp', 'scaled', or 'all'. The **periodicity** option will infer whether the scales used are month or day scales.

    **NOTE**: When used for US climate divisions processing this option specifies month scales
* - output_scale
  - Output convention for the standardized SPI and SPEI values, applicable only when **index** includes SPI or SPEI. 'normal' (default) writes standard-normal z-scores; 'probability' writes the fitted cumulative probability in [0, 1] without clipping; 'bounded' writes its 2p - 1 mapping in [-1, 1]. In a mixed pipeline such as 'all' the other outputs ignore it. Each scale writes its own output files and variable names (see **output_file_base**).
* - calibration_start_year
  - Initial year of the calibration period. Optional for EDI and the Flood Index, and inferred from the input when omitted: EDI starts in the first year of the record, and the Flood Index in the second, since effective precipitation is undefined for a record's first 364 days.
* - calibration_end_year
  - Final year of the calibration period (inclusive). Optional for EDI and the Flood Index, and inferred from the input when omitted. For the Flood Index it is the start year of the last complete annual period, as set by **year_start_month**.
* - multiprocessing
  - Valid values are 'all' (uses all available CPUs), 'single' (uses a single CPU), or 'all_but_one' (uses all CPUs minus one). Default value is 'all_but_one'.
```

Copy-and-adapt invocations are in {doc}`workflow-examples`.

## Public API — Index Functions

:::{note} The xarray DataArray overloads in `typed_public_api` are **beta**. NumPy overloads are stable. See {doc}`xarray_migration` for details.
:::

### climate_indices.typed_public_api

```{eval-rst}
.. automodule:: climate_indices.typed_public_api
   :members:
```

### climate_indices.energy

NumPy degree-day, wind-resource, standardized-index (SREPI, demand, SRLI), and
event-statistics proxies, not actual energy generation or load.
`energy_drought_events()` reports run metrics and period summaries through the
generic run-theory utility. Standardized energy and event xarray/Dask adapters
remain planned; see {doc}`energy_applications` for scope and validation limits.

```{eval-rst}
.. automodule:: climate_indices.energy
   :members:
   :imported-members:
```

### climate_indices.compound

Declared-input harmonization, frequency statistics, and run matching for compound-hazard
analysis: Boolean exceedance and validity on observed dates, never interpolated or averaged,
then marginal and joint counts, marginal and joint frequencies, the independence product, and
the likelihood multiplication factor (LMF) on one complete-case support. Marginals are observed
frequencies, so a zero marginal makes LMF undefined (`NaN`) rather than infinite, and
`LMF = 1` does not establish independence.
`conditional_probability()` conditions one named hazard on another over an inclusive
forward window and reports the conditional time-step frequency beside the
response-window frequency on the same eligible anchors; an unknown or incomplete
horizon is never counted as a non-response. `compound_events()` pairs concurrent or
sequential events and keeps their source runs and censoring. `bootstrap_dependence()`
adds monthly uncertainty intervals and local independence tests. Import the module
explicitly with `import climate_indices.compound as compound`; it is not re-exported
from the package root. `harmonize()`, `joint_statistics()`, `conditional_probability()`,
and `bootstrap_dependence()` also take xarray input on a beta route, described in
[Beta xarray route](#compound-xarray-route); see
[ADR-0022](adr/0022-compound-analysis-contracts.md) for the approved contracts.

#### Monthly bootstrap inference

:::{warning}
Synthetic characterization gives screening evidence for these intervals and
p-values only for serially independent or 6-month-overlap sources whose hazards
are independent of each other, with hazard frequencies of 0.35 to 0.55 and no lag,
at 200 years and 399 resamples. For 36-month persistence one coverage check
failed and null rejection exceeded the nominal 5% at every block length, so do not
read those results as nominal. Hazards that depend on each other were measured
only with 36-month persistence. Other frequencies, lagged conditioning, and
shorter records were not characterized. See the
[validation record](https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md#compound-bootstrap-characterization).
This is synthetic evidence, not an external validation of the method.
:::

The inference support is each cell's longest contiguous segment of complete
calendar years. A tie selects the earliest segment. Missing months split the
segment. `estimate` uses this segment; `full_record_estimate` uses all eligible
original observations. Partial boundary years enter only the latter estimate.
Conditional responses are computed before resampling, on the original calendar.
They never cross artificial block joins.

Non-circular moving blocks preserve calendar-month positions. The default length
is `ceil(Y ** (1/3))` years. An explicit `block_length_years` overrides it.
Paired channels supply percentile confidence intervals. Independent channel
samples supply the independence null. The default alternative, `"greater"`, tests
amplification. `"less"` tests suppression. Joint tests use
`p_joint - product(p_i)`; conditional tests use `P(R|A) - P(R)`.
Null comparisons include ties within `1e-12`, so floating-point rounding of equal
count ratios cannot break a tie, with `p=(1+extreme)/(1+B_valid)`.
All p-values are local and unadjusted. They do not establish field significance.

The defaults are 1,999 realizations per sampling method and a 95% interval using
`numpy.quantile(method="linear")`. Supply an integer `seed`. The local generator
uses the seed and support definition, not a cell's position or worker schedule.
The manifest records NumPy and SciPy versions. Cross-version bitwise identity
is not promised. `n_resamples` must be between 1 and 100,000 inclusive.
Larger requests raise `InvalidArgumentError` before computation. This bounds the
per-cell realization arrays to 19.2 MB; total work still grows with the cell and
year counts.

Inference requires 30 complete years, five nonoverlapping blocks, and ten
occurrences per channel within each reported group. An interval also requires
five observed successes. A null test instead requires five expected successes
under independence. Zero joint counts can support suppression tests, but cannot
produce a misleading zero-width interval. Each metric requires 95% valid
realizations. Inspect `quality_flags` when a bound or p-value is NaN.
Fewer than ten nonoverlapping blocks produces a warning.

| Result fields | Numeric contract, shared by the NumPy and xarray routes |
| --- | --- |
| `estimate`, `full_record_estimate`, `confidence_low`, `confidence_high` | Float64, units `1`. Probability values range from 0 to 1. LMF is nonnegative without a universal maximum. |
| `p_value` | Float64, units `1`, local and unadjusted, range 0 to 1; finite values are never zero. |
| `n_valid`, `n_joint` | Int64 eligible-observation and observed-success counts on the inference segment, retained even when inference is suppressed. For conditional statistics these count eligible anchors and successful positive anchors. |
| `n_marginal` | Int64 per-channel occurrence counts with a leading source axis in `metadata["source_names"]` order. Conditional channels are the given hazard and its target response window. |
| `expected_null_successes` | Float64 expected successes under independence, exactly as tested by the null eligibility guard; NaN without eligible observations. |
| `n_interval_resamples`, `n_null_resamples` | Int64 counts of valid realizations; zero when a pre-sampling guard excludes the metric. |
| `n_years`, `block_length_years` | Int64 calendar-year counts, one per cell. |
| `inference_start`, `inference_end` | datetime64[D] January and December month labels, one per cell; NaT without a complete year. |
| `quality_flags` | Int64 bit field. `metadata["quality_flags"]` supplies the masks and meanings. An interval flag does not automatically suppress a null test. |

Estimate, bound, p-value, support-count, expected-success, resample-count, and
flag arrays have shape `(12, *cells)` for monthly groups or `(*cells)` for pooled
groups. `n_marginal` adds a leading source axis.
All result arrays are read-only, and the manifest is immutable.
Daily inference raises `InvalidArgumentError`.

Each call logs `calculation_started` and then `calculation_completed`, or
`calculation_failed` for an error, under `index_type="compound_bootstrap_dependence"`.
The events carry the source count, statistic, grouping, seed, and resample count,
not caller-controlled source names. They describe the call, not scheduling, because
the NumPy route runs in the calling process; the xarray route's events are described
under [Beta xarray route](#compound-xarray-route).

Check sensitivity at `max(1, floor(L/2))`, `L`, and `2L` where support permits.
Do not choose a length by its smallest p-value. Synthetic characterization and
regression tests do not establish external scientific validation.
The reproducible configuration and measured results are
[`scripts/characterize_compound_bootstrap.py`](https://github.com/monocongo/climate_indices/blob/main/scripts/characterize_compound_bootstrap.py)
and
[`compound-bootstrap-characterization.json`](https://github.com/monocongo/climate_indices/blob/main/docs/design/compound-bootstrap-characterization.json).
The guards do not detect nonstationarity, long memory, or impact relevance.
The final worked guide belongs to CMP-10 (#1374).

(compound-xarray-route)=
#### Beta xarray route

Pass a mapping of `xarray.DataArray` to `harmonize()` to get a Dataset of `exceedance` and
`valid` with dimensions `(hazard, time, *spatial)`. Pass that Dataset to
`joint_statistics()`, `conditional_probability()`, or `bootstrap_dependence()` to get a
Dataset of the same fields, options, and results as the NumPy dataclass. The route is beta
under ADR-0012, so every call emits `BetaFeatureWarning`; the NumPy API is stable. It adds
labels, metadata, and a Dask contract, and runs the NumPy core on each spatial block, so
there is one scientific implementation. A single location still returns a Dataset.

DataArray `harmonize()` reads each source's dates from its `time_dim` coordinate (default
`"time"`) and layout from named dimensions, so it rejects the NumPy-only `time` and
`spatial_time_major` arguments, and a mapping that mixes DataArrays with NumPy arrays. The
dates must be real `datetime64` labels on a Gregorian calendar; a dimension-only time,
cftime, and non-Gregorian calendars are rejected, and no calendar is converted.

Results follow the NumPy layout: `month` (coordinate 1 to 12) is omitted when pooled,
`hazard` labels marginal and per-channel values (for the bootstrap, the selected channels),
and spatial dimensions keep their input order and coordinates. Counts are `int64`, undefined
estimates are `NaN`, and `quality_flags` is a numeric bit field with CF `flag_masks` and
`flag_meanings`. Variables carry the registry's `long_name`, `units="1"`, and references,
probability bounds `[0, 1]`, and an LMF lower bound of 0 with no maximum. No `standard_name`
is set, and nothing is inherited from a source array's attributes. The Dataset attribute
`compound_manifest` is the versioned JSON choice manifest, equal to the NumPy result's
metadata, and the Dataset round-trips through NetCDF and Zarr.

:::{important}
Chunking contract. A Dask input must hold `time` (and, for the statistics routes, `hazard`)
in one chunk, every spatial dimension must be chunked identically across inputs, all inputs
must be Dask-backed or all in memory, and every spatial dimension needs coordinate labels.
Options, labels, and chunk shapes are validated when the call is made. Nothing is rechunked,
broadcast, loaded, or computed. A split chunk or a mismatch raises an error that names the
explicit `.chunk()` call to make.
:::

Value-dependent errors, such as a probability outside `[0, 1]`, surface when a lazy graph is
computed. The same rule governs warnings. In-memory input emits the NumPy support warnings
(a short record, sparse occurrences, a group below `min_samples`, a zero marginal). Dask input
does not, because a warning raised inside a task would repeat per tile with tile-local counts,
so read `quality_flags`. Warnings that depend only on dates and declarations, such as a missing
Calibration Period or `join="inner"` dropping boundaries, are raised at the call for both.

Each call logs `calculation_started` and then `calculation_completed`, or `calculation_failed`,
with `route="xarray"` and the same `index_type` as the NumPy route. `calculation_completed`
carries `evaluation="graph_construction"` for a lazy call: the graph was built and no worker has
computed anything, so a failure inside a task is never logged as `calculation_failed`. An
in-memory call logs `evaluation="eager"`. Bootstrap results do not depend on spatial tile
boundaries, and no route allocates a `(resample, time, cell)` array.

Measured peak memory per task (`scripts/measure_compound_xarray_memory.py`, report
[`compound-xarray-memory.json`](https://github.com/monocongo/climate_indices/blob/main/docs/design/compound-xarray-memory.json)):

| Stage | Fixed per task | Per cell | Measured / design budget | Cells in 1 GiB |
| --- | --- | --- | --- | --- |
| `harmonize`, daily, 3 sources, 100 years | 2.7 MiB | 2.40 MiB | 1.28 | about 420 |
| `joint_statistics`, same prepared Dataset | 0.9 MiB | 0.60 MiB | 0.32 | about 1,700 |
| `conditional_probability`, lag `(0, 30)` | 0.9 MiB | 0.70 MiB | 0.37 | about 1,460 |
| `bootstrap_dependence`, monthly, 100 years, 1,999 resamples | 0.9 MiB | 0.015 MiB | 0.20 | memory is not the limit |

The ADR-0022 planning budget understates `harmonize` by about 28%. The bootstrap reuses one
cell's realization arrays for every cell, so its memory is a fixed cost per task plus a few KiB
per cell, and its tile width is set by compute time, which was not characterized. The figures
are `tracemalloc` lower bounds from one machine; a manual peak-RSS check of `harmonize` gave about
20% more per cell. They scale linearly with the number of times and sources. Plan for at most
half the tabulated cells per task, after dividing a worker's memory by its concurrent tasks.

Agreement with the NumPy routes is interface equivalence, not external validation; see
[VALIDATION.md](https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md).

```{eval-rst}
.. automodule:: climate_indices.compound
   :members:
   :imported-members:
```

## xarray Integration

:::{warning} **Beta Feature** — The xarray adapter layer is beta. See {doc}`xarray_migration` for stability guarantees.
:::

The Dask-backed SPI/SPEI workflow is demonstrated end to end in
[notebooks/zarr_dask_spi_spei.ipynb](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb);
{doc}`troubleshooting` covers its setup and failure modes.

### climate_indices.validation

```{eval-rst}
.. automodule:: climate_indices.validation
   :members:
   :exclude-members: InputType, GRID, DIVISIONS, TIMESERIES
```

```{eval-rst}
.. autoclass:: climate_indices.validation.InputType
   :members:
   :exclude-members: NUMPY, XARRAY
```

### climate_indices.xarray_adapter

```{eval-rst}
.. automodule:: climate_indices.xarray_adapter
   :members:
```

## Core Computation Modules

### climate_indices.compute

```{eval-rst}
.. automodule:: climate_indices.compute
   :members:
   :exclude-members: DistributionFittingError, InsufficientDataError, PearsonFittingError
```

### climate_indices.indices

```{eval-rst}
.. automodule:: climate_indices.indices
   :members:
```

### climate_indices.eto

```{eval-rst}
.. automodule:: climate_indices.eto
   :members:
```

### climate_indices.palmer

```{eval-rst}
.. automodule:: climate_indices.palmer
   :members:
```

### climate_indices.lmoments

```{eval-rst}
.. automodule:: climate_indices.lmoments
   :members:
```

### climate_indices.runs

```{eval-rst}
.. automodule:: climate_indices.runs
   :members:
```

### climate_indices.utils

```{eval-rst}
.. automodule:: climate_indices.utils
   :members:
```

## Error Handling

### climate_indices.exceptions

```{eval-rst}
.. automodule:: climate_indices.exceptions
   :members:
   :special-members: __init__
```

## Observability

### climate_indices.logging_config

```{eval-rst}
.. automodule:: climate_indices.logging_config
   :members:
```

### climate_indices.performance

```{eval-rst}
.. automodule:: climate_indices.performance
   :members:
```


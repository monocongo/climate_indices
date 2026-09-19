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
error-reference
xarray_compatibility
deprecations/index
```

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
  - Which of the climate indices to compute (required). Valid values are 'spi', 'spei', 'pnp', 'scaled', 'pet', 'palmers', 'kbdi', and 'all'. 'scaled' indicates all three scaled indices (SPI, SPEI, and PNP) and 'palmers' indicates all Palmer indices (PDSI, PHDI, PMDI, Z-Index, scPDSI). KBDI is only available via 'kbdi', not 'all'.
* - periodicity
  - The periodicity of the input dataset files (required). Valid values are 'monthly' and 'daily'.

    **NOTE**: SPI, SPEI (with a PET input), PNP, and KBDI accept daily inputs; KBDI requires daily inputs. Palmers and 'all' require monthly inputs.
* - netcdf_precip
  - Input NetCDF file containing a precipitation dataset, required for all indices except for PET. Requires the use of **var_name_precip** in conjunction so as to identify the NetCDF's precipitation variable.
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
* - kbdi_units
  - Units of the KBDI input and output values. Valid values are 'metric' (millimeters and degrees Celsius; default) and 'imperial' (inches and degrees Fahrenheit, with output in hundredths of an inch). Applicable only when **index** is 'kbdi'.
* - kbdi_initial
  - Initial KBDI value. Default value is 0.0. Applicable only when **index** is 'kbdi'.
* - chunksizes
  - Chunking of the written output file, not of the computation: 'none' (default) lets the writer choose the output layout, and 'input' copies the on-disk chunks of the first chunked input variable to the output (for KBDI, the precipitation variable's chunks).
* - output_file_base
  - Base file name for all output files (required).

    Each computed index will have a corresponding output file whose name will begin with this base name plus the index's abbreviation plus a month scale (if applicable), connected with underscores, plus the '.nc' extension. For example for SPI at 3-month scale the resulting output files will be named **<output_file_base>_spi_gamma_03.nc** and **<output_file_base>_spi_pearson_03.nc**.
* - scales
  - Time step scales over which the PNP, SPI, and SPEI values are to be computed. Required when the **index** argument is 'spi', 'spei', 'pnp', 'scaled', or 'all'. The **periodicity** option will infer whether the scales used are month or day scales.

    **NOTE**: When used for US climate divisions processing this option specifies month scales
* - calibration_start_year
  - Initial year of the calibration period.
* - calibration_end_year
  - Final year of the calibration period (inclusive).
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


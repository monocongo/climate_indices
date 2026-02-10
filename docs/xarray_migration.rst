xarray Migration Guide
======================

.. contents::
   :local:
   :backlinks: none

Introduction
------------

Starting with version 2.2.0, the ``climate_indices`` library provides native support for xarray DataArrays alongside the traditional NumPy array API. This migration guide helps existing users understand the benefits of adopting the xarray API and provides practical examples for transitioning from NumPy-based workflows to xarray-based workflows.

**Who this guide is for:** Users currently working with NumPy arrays who want to leverage xarray's labeled dimensions, automatic metadata handling, and coordinate-aware operations.

**What changed:** All primary index functions (``spi()``, ``spei()``, ``pet_thornthwaite()``, ``pet_hargreaves()``) now accept both ``np.ndarray`` and ``xr.DataArray`` inputs. When you pass an xarray DataArray, the library automatically:

- Infers temporal parameters from time coordinates
- Preserves all coordinate information in outputs
- Adds CF-compliant metadata attributes
- Aligns multi-input arrays automatically
- Supports Dask for out-of-core computation


Why Migrate to xarray?
-----------------------

The xarray API offers several key advantages over the traditional NumPy API:

**Automatic parameter inference**
   Parameters like ``data_start_year``, ``periodicity``, and calibration bounds are automatically inferred from the time coordinate, reducing boilerplate and eliminating a common source of errors.

**Coordinate preservation**
   Output DataArrays retain all input coordinates (time, lat, lon, etc.) and their attributes, making it easy to track spatial and temporal metadata through your analysis pipeline.

**CF-compliant metadata**
   Outputs automatically include standardized attributes (``long_name``, ``units``, ``references``) and a provenance history trail following CF Convention standards, improving reproducibility and dataset documentation.

**Multi-input alignment**
   Functions like ``spei()`` automatically align precipitation and PET arrays using ``xr.align(join="inner")``, handling mismatched time ranges transparently and warning when data is dropped.

**Dask support**
   Pass chunked DataArrays to enable lazy, out-of-core computation on datasets larger than available RAM, with automatic parallelization across spatial dimensions.

**Type safety**
   IDE autocomplete and type checkers like mypy can narrow return types based on input types, catching potential errors at development time rather than runtime.


Side-by-Side Comparison
------------------------

This section shows how to convert common NumPy workflows to the equivalent xarray patterns.


2.1 SPI — Standardized Precipitation Index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before (NumPy):**

.. code-block:: python

   import numpy as np
   from climate_indices import indices
   from climate_indices.compute import Periodicity
   from climate_indices.indices import Distribution

   # 40 years × 12 months/year = 480 monthly values
   precip_mm = np.load("monthly_precip.npy")  # shape: (40, 12)

   # compute SPI-6 with gamma distribution
   spi_values = indices.spi(
       values=precip_mm,
       scale=6,
       distribution=Distribution.gamma,
       data_start_year=1980,                    # required
       calibration_year_initial=1981,           # required
       calibration_year_final=2010,             # required
       periodicity=Periodicity.monthly,         # required
   )
   # result: np.ndarray, shape (40, 12)

**After (xarray):**

.. code-block:: python

   import pandas as pd
   import xarray as xr
   from climate_indices import spi
   from climate_indices.indices import Distribution

   # load data as xarray DataArray with time coordinate
   time = pd.date_range("1980-01-01", periods=480, freq="MS")
   precip_da = xr.DataArray(
       precip_mm.flatten(),
       coords={"time": time},
       dims=["time"],
       attrs={"units": "mm", "long_name": "Monthly Precipitation"},
   )

   # compute SPI-6 with gamma distribution
   spi_result = spi(
       values=precip_da,
       scale=6,
       distribution=Distribution.gamma,
       # data_start_year, calibration bounds, and periodicity
       # are all inferred from the time coordinate!
   )
   # result: xr.DataArray with time coord and CF metadata

**Key changes:**

- Import from ``climate_indices.spi`` instead of ``climate_indices.indices.spi``
- Data shape changes from 2D (years, months) to 1D with labeled time dimension
- Only 2 required parameters (``scale``, ``distribution``) instead of 6
- All temporal parameters (``data_start_year``, ``calibration_year_initial``, ``calibration_year_final``, ``periodicity``) are optional and inferred automatically
- Output includes coordinate metadata and CF attributes


2.2 SPEI — Standardized Precipitation Evapotranspiration Index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before (NumPy):**

.. code-block:: python

   import numpy as np
   from climate_indices import indices
   from climate_indices.compute import Periodicity
   from climate_indices.indices import Distribution

   # load precipitation and PET arrays
   precip_mm = np.load("monthly_precip.npy")  # shape: (40, 12)
   pet_mm = np.load("monthly_pet.npy")        # shape: (40, 12)

   # arrays must be pre-aligned manually!
   assert precip_mm.shape == pet_mm.shape

   # compute SPEI-3
   spei_values = indices.spei(
       precips_mm=precip_mm,
       pet_mm=pet_mm,
       scale=3,
       distribution=Distribution.gamma,
       periodicity=Periodicity.monthly,
       data_start_year=1980,
       calibration_year_initial=1981,
       calibration_year_final=2010,
   )

**After (xarray):**

.. code-block:: python

   import pandas as pd
   import xarray as xr
   from climate_indices import spei
   from climate_indices.indices import Distribution

   # load precipitation and PET as DataArrays
   # they can have different time ranges!
   time_precip = pd.date_range("1980-01-01", periods=480, freq="MS")
   time_pet = pd.date_range("1985-01-01", periods=420, freq="MS")

   precip_da = xr.DataArray(
       precip_values,
       coords={"time": time_precip},
       dims=["time"],
   )
   pet_da = xr.DataArray(
       pet_values,
       coords={"time": time_pet},
       dims=["time"],
   )

   # compute SPEI-3 — automatic inner join alignment!
   spei_result = spei(
       precips_mm=precip_da,
       pet_mm=pet_da,
       scale=3,
       distribution=Distribution.gamma,
       # alignment, calibration, and periodicity all handled automatically
   )
   # result covers 1985-2019 (intersection of input time ranges)

**Key changes:**

- Import from ``climate_indices.spei`` instead of ``climate_indices.indices.spei``
- Inputs no longer need to be pre-aligned — ``xr.align(join="inner")`` handles mismatched time ranges
- A warning is emitted if alignment drops timesteps, helping detect data issues
- Only 3 required parameters (``scale``, ``distribution``) instead of 7


2.3 PET Thornthwaite
~~~~~~~~~~~~~~~~~~~~

**Before (NumPy):**

.. code-block:: python

   from climate_indices import indices

   # 1D array of 480 monthly temperatures
   temp_celsius = np.load("monthly_temp.npy")  # shape: (480,)

   # compute PET using Thornthwaite method
   pet_mm = indices.pet(
       temperature_celsius=temp_celsius,
       latitude_degrees=40.0,
       data_start_year=1980,  # required
   )
   # result: np.ndarray, shape (480,)

**After (xarray):**

.. code-block:: python

   from climate_indices.xarray_adapter import pet_thornthwaite

   # load as DataArray
   time = pd.date_range("1980-01-01", periods=480, freq="MS")
   temp_da = xr.DataArray(
       temp_values,
       coords={"time": time},
       dims=["time"],
       attrs={"units": "degC"},
   )

   # compute PET — data_start_year inferred from time coord
   pet_result = pet_thornthwaite(
       temperature=temp_da,
       latitude=40.0,
   )
   # result: xr.DataArray with CF metadata

**Gridded example with spatial broadcasting:**

.. code-block:: python

   # load gridded temperature (time, lat, lon)
   temp_grid = xr.DataArray(
       temp_3d_values,  # shape: (480, 5, 6)
       coords={
           "time": pd.date_range("1980-01-01", periods=480, freq="MS"),
           "lat": [30, 35, 40, 45, 50],
           "lon": [-120, -110, -100, -90, -80, -70],
       },
       dims=["time", "lat", "lon"],
   )

   # latitude as DataArray for spatial broadcasting
   lat_array = xr.DataArray([30, 35, 40, 45, 50], dims=["lat"])

   # automatically broadcasts latitude across grid
   pet_grid = pet_thornthwaite(temp_grid, latitude=lat_array)
   # result: xr.DataArray, shape (480, 5, 6)

**Key changes:**

- Import from ``climate_indices.xarray_adapter`` instead of ``climate_indices.indices``
- ``data_start_year`` is optional and inferred from time coordinate
- Latitude can be a scalar or DataArray for spatial broadcasting
- Function name changes to ``pet_thornthwaite()`` for clarity


2.4 PET Hargreaves
~~~~~~~~~~~~~~~~~~

**Before (NumPy):**

.. code-block:: python

   from climate_indices.eto import eto_hargreaves

   # 1D arrays of daily min/max temperatures
   tmin = np.load("daily_tmin.npy")  # shape: (1825,)
   tmax = np.load("daily_tmax.npy")  # shape: (1825,)

   # manually derive mean temperature
   tmean = (tmin + tmax) / 2.0

   # compute PET using Hargreaves method
   pet_mm = eto_hargreaves(
       daily_tmin_celsius=tmin,
       daily_tmax_celsius=tmax,
       daily_tmean_celsius=tmean,  # must provide manually
       latitude_degrees=40.0,
   )

**After (xarray):**

.. code-block:: python

   from climate_indices.xarray_adapter import pet_hargreaves

   # load as DataArrays
   time = pd.date_range("2015-01-01", periods=1825, freq="D")
   tmin_da = xr.DataArray(tmin_values, coords={"time": time}, dims=["time"])
   tmax_da = xr.DataArray(tmax_values, coords={"time": time}, dims=["time"])

   # compute PET — tmean derived automatically!
   pet_result = pet_hargreaves(
       daily_tmin_celsius=tmin_da,
       daily_tmax_celsius=tmax_da,
       latitude=40.0,
   )
   # result: xr.DataArray with CF metadata

**Key changes:**

- Import from ``climate_indices.xarray_adapter``
- Mean temperature is automatically derived as ``(tmin + tmax) / 2``
- Inputs are automatically aligned via ``xr.align(join="inner")``
- Supports spatial broadcasting with latitude as DataArray


Understanding the Output Metadata
----------------------------------

When you use the xarray API, outputs include rich metadata that improves reproducibility and interoperability with other tools in the scientific Python ecosystem.

**Example SPI output attributes:**

.. code-block:: python

   result = spi(precip_da, scale=6, distribution=Distribution.gamma)
   print(result.attrs)

   # {
   #     'long_name': 'Standardized Precipitation Index',
   #     'units': 'dimensionless',
   #     'references': 'McKee, T. B., Doesken, N. J., & Kleist, J. (1993). ...',
   #     'scale': 6,
   #     'distribution': 'gamma',
   #     'calibration_year_initial': 1980,
   #     'calibration_year_final': 2019,
   #     'climate_indices_version': '2.2.0',
   #     'history': '2026-02-09T14:23:45Z: SPI-6 calculated using gamma distribution (climate_indices v2.2.0)',
   # }


CF Convention Compliance
~~~~~~~~~~~~~~~~~~~~~~~~

The library follows CF (Climate and Forecast) Convention standards for metadata:

**CF metadata fields:**
   - ``long_name``: Human-readable description of the variable
   - ``units``: Physical units (e.g., "dimensionless", "mm/month")
   - ``references``: Academic citation for the algorithm
   - ``standard_name``: CF standard name (when officially defined)

**Calculation metadata:**
   Parameters used in the computation (``scale``, ``distribution``, ``calibration_year_initial``, ``calibration_year_final``) are stored as attributes for reproducibility.

**Provenance tracking:**
   The ``history`` attribute follows CF Convention format with ISO 8601 timestamps. When you chain operations, new history entries are appended with newline separators:

   .. code-block:: python

      # original history from upstream processing
      precip_da.attrs["history"] = "2026-02-08T10:00:00Z: Data downloaded from NCAR"

      # compute SPI
      result = spi(precip_da, scale=3, distribution=Distribution.gamma)

      # history is appended, not replaced
      print(result.attrs["history"])
      # 2026-02-08T10:00:00Z: Data downloaded from NCAR
      # 2026-02-09T14:23:45Z: SPI-3 calculated using gamma distribution (climate_indices v2.2.0)


Coordinate Preservation
~~~~~~~~~~~~~~~~~~~~~~~

All coordinates and their attributes are deep-copied from input to output, preserving spatial and temporal metadata:

**Dimension coordinates:**
   Primary axis coordinates (time, lat, lon) with their attributes

**Auxiliary coordinates:**
   Non-dimension coordinates like month-of-year or station ID

**Scalar coordinates:**
   Zero-dimensional coordinates like ensemble member or experiment name

.. code-block:: python

   # input with rich coordinate metadata
   precip_da.coords["time"].attrs = {
       "axis": "T",
       "calendar": "standard",
       "long_name": "time",
   }
   precip_da.coords["lat"].attrs = {
       "axis": "Y",
       "standard_name": "latitude",
       "units": "degrees_north",
   }

   # compute SPI
   result = spi(precip_da, scale=6, distribution=Distribution.gamma)

   # all coordinate attributes are preserved
   assert result.coords["time"].attrs == precip_da.coords["time"].attrs
   assert result.coords["lat"].attrs == precip_da.coords["lat"].attrs


Direct NetCDF export
~~~~~~~~~~~~~~~~~~~~~

Because outputs are standard xarray DataArrays with CF-compliant metadata, you can write them directly to NetCDF files without additional metadata preparation:

.. code-block:: python

   # compute index
   spi_result = spi(precip_da, scale=12, distribution=Distribution.gamma)

   # write to NetCDF with full metadata
   spi_result.to_netcdf("spi_12month_1980_2019.nc")

   # later, reload with metadata intact
   spi_loaded = xr.open_dataarray("spi_12month_1980_2019.nc")
   print(spi_loaded.attrs["references"])  # citation preserved


Common Pitfalls and Solutions
------------------------------

1. Time dimension must be named "time"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   **Error:** ``CoordinateValidationError: Time dimension 'time' not found in input. Available dimensions: ['date']. Use time_dim parameter to specify custom name.``

   **Cause:** Your DataArray uses a different dimension name (e.g., ``date``, ``timestamp``, ``t``).

   **Solution:** Rename the dimension to ``time`` before calling the function:

   .. code-block:: python

      # if your data uses "date" instead of "time"
      da_renamed = da.rename({"date": "time"})
      result = spi(da_renamed, scale=6, distribution=Distribution.gamma)


2. Non-monotonic time coordinate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   **Error:** ``CoordinateValidationError: Time coordinate is not monotonically increasing. Sort the data using data.sortby('time') before processing.``

   **Cause:** Time values are not in chronological order (e.g., data was shuffled or merged from multiple sources).

   **Solution:** Sort by time coordinate before processing:

   .. code-block:: python

      da_sorted = da.sortby("time")
      result = spi(da_sorted, scale=6, distribution=Distribution.gamma)


3. Unsupported frequency
~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   **Error:** ``CoordinateValidationError: Unsupported frequency 'W' - only 'MS'/'ME'/'M' (monthly) and 'D' (daily) supported``

   **Cause:** Your data has weekly, quarterly, or annual frequency.

   **Solution:** The library only supports monthly (MS/ME/M) and daily (D) data. Resample your data to a supported frequency:

   .. code-block:: python

      # resample weekly to monthly using sum aggregation
      da_monthly = da.resample(time="MS").sum()
      result = spi(da_monthly, scale=6, distribution=Distribution.gamma)


4. Multi-chunked time dimension (Dask)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   **Error:** ``CoordinateValidationError: Time dimension 'time' is split across 4 chunks. Climate indices require the full time series for distribution fitting. Rechunk using: data = data.chunk({'time': -1})``

   **Cause:** Your Dask-backed DataArray has the time dimension split into multiple chunks. SPI/SPEI require the full time series for fitting distributions.

   **Solution:** Rechunk time dimension to a single chunk:

   .. code-block:: python

      # rechunk time to single chunk, keep spatial chunks
      da_rechunked = da.chunk({"time": -1, "lat": 10, "lon": 10})
      result = spi(da_rechunked, scale=6, distribution=Distribution.gamma)


5. Passing Dataset instead of DataArray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   **Error:** ``InputTypeError: Unsupported input type: xarray.core.dataset.Dataset. Accepted types: np.ndarray, list, tuple, int, float, np.integer, np.floating, xr.DataArray. xr.Dataset detected: Use ds['variable_name'] to select a DataArray``

   **Cause:** You passed an ``xr.Dataset`` (collection of variables) instead of an ``xr.DataArray`` (single variable).

   **Solution:** Select a specific variable from the Dataset:

   .. code-block:: python

      # if you loaded a NetCDF file with xr.open_dataset()
      ds = xr.open_dataset("precipitation.nc")

      # extract the precipitation variable as a DataArray
      precip_da = ds["precipitation"]
      result = spi(precip_da, scale=6, distribution=Distribution.gamma)


6. Insufficient calibration data with NaN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   **Error:** ``InsufficientDataError: Insufficient non-NaN data in calibration period (1980-2010). Found 245 non-NaN values (20.4 effective years), but at least 30 years of non-NaN data required for reliable distribution fitting.``

   **Cause:** Your data contains too many NaN values within the calibration period, leaving fewer than 30 years of valid observations.

   **Solution:** Either extend the calibration period or use a different time range with denser data:

   .. code-block:: python

      # option 1: extend calibration period if you have more data
      result = spi(
          precip_da,
          scale=6,
          distribution=Distribution.gamma,
          calibration_year_initial=1950,  # earlier start
          calibration_year_final=2020,    # later end
      )

      # option 2: use only the dense portion of your dataset
      dense_period = precip_da.sel(time=slice("1990", "2020"))
      result = spi(dense_period, scale=6, distribution=Distribution.gamma)


Performance Considerations
---------------------------

When to use Dask chunking
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Dask-backed DataArrays when:

- **Gridded datasets exceed available RAM** (e.g., global climate model output)
- **Spatial dimensions are large** (e.g., 720×360 global grid)
- **Processing multiple realizations** (ensemble forecasts with time×lat×lon×member dimensions)

Do NOT use Dask for:

- **1D time series** — overhead exceeds benefit for small arrays
- **Small grids** (< 100 spatial points) — eager computation is faster

.. code-block:: python

   # good use case: large gridded dataset
   precip = xr.open_dataarray(
       "global_precip.nc",
       chunks={"time": -1, "lat": 50, "lon": 50},  # single time chunk
   )
   result = spi(precip, scale=6, distribution=Distribution.gamma)
   result.to_netcdf("spi_global.nc")  # triggers computation


Optimal chunking strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

For climate index calculations:

1. **Time dimension:** Must be a single chunk (``{"time": -1}``)
2. **Spatial dimensions:** Chunk to balance parallelism vs. overhead

   - Too large: Limited parallelism, potential memory issues
   - Too small: Excessive overhead from task scheduling

   A good starting point: **10-20 chunks per spatial dimension**

.. code-block:: python

   # example: 40 years × 180 lat × 360 lon global grid
   da = xr.open_dataarray(
       "global_data.nc",
       chunks={
           "time": -1,      # single chunk (required)
           "lat": 18,       # 10 chunks (180/18)
           "lon": 36,       # 10 chunks (360/36)
       },
   )


Overhead
~~~~~~~~

**1D time series:** Xarray overhead is minimal (<5%) compared to NumPy path. The convenience and metadata benefits outweigh the small performance cost.

**Gridded data:** Performance is computation-dominated. Overhead from xarray/Dask is negligible relative to distribution fitting time. Spatial parallelism via Dask provides significant speedup on multi-core systems.

**Benchmark results** (40-year monthly data, SPI-6):

============= ============= ==========
Input type    Shape         Time
============= ============= ==========
NumPy         (40, 12)      12.3 ms
xarray 1D     (480,)        12.8 ms (+4%)
xarray 3D     (480, 20, 20) 1.42 s
Dask 3D       (480, 20, 20) 0.58 s (2.4× faster)
============= ============= ==========


Lazy evaluation
~~~~~~~~~~~~~~~

Dask-backed DataArrays use lazy evaluation — computations build a task graph but don't execute until you call ``.compute()`` or trigger output (e.g., ``.to_netcdf()``):

.. code-block:: python

   # open dataset with chunking (no data loaded yet)
   precip = xr.open_dataarray("large_file.nc", chunks={"time": -1, "lat": 20})

   # compute SPI (builds task graph, no computation yet)
   spi_result = spi(precip, scale=6, distribution=Distribution.gamma)

   # trigger computation and load result into memory
   spi_eager = spi_result.compute()

   # OR write directly to disk without loading into memory
   spi_result.to_netcdf("output.nc")  # triggers compute + streams to disk


Real-World Examples
-------------------

Example 1: SPI from NetCDF
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import xarray as xr
   from climate_indices import spi
   from climate_indices.indices import Distribution

   # load monthly precipitation from NetCDF
   ds = xr.open_dataset("PRISM_precip_monthly_1981_2020.nc")
   precip = ds["ppt"]  # extract DataArray

   # compute SPI-12 using full time range for calibration
   spi_12 = spi(
       values=precip,
       scale=12,
       distribution=Distribution.gamma,
   )

   # add descriptive metadata
   spi_12.attrs["title"] = "SPI-12 for CONUS 1981-2020"
   spi_12.attrs["source"] = "PRISM Climate Group"

   # write to NetCDF with full provenance
   spi_12.to_netcdf("spi_12_CONUS_1981_2020.nc")


Example 2: SPEI from separate files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import xarray as xr
   from climate_indices import spei
   from climate_indices.indices import Distribution

   # load precipitation and PET from different files
   # they may have different time ranges!
   precip = xr.open_dataarray("precip_1950_2020.nc")
   pet = xr.open_dataarray("pet_1980_2020.nc")

   # compute SPEI-6 — automatic inner join alignment
   spei_6 = spei(
       precips_mm=precip,
       pet_mm=pet,
       scale=6,
       distribution=Distribution.gamma,
   )
   # result covers 1980-2020 (intersection of input ranges)

   # explicit calibration period (optional)
   spei_6_custom = spei(
       precips_mm=precip,
       pet_mm=pet,
       scale=6,
       distribution=Distribution.gamma,
       calibration_year_initial=1991,
       calibration_year_final=2020,
   )


Example 3: Gridded PET with spatial latitude broadcasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import xarray as xr
   from climate_indices.xarray_adapter import pet_thornthwaite

   # load gridded monthly temperature
   temp_grid = xr.open_dataarray("temp_monthly_grid.nc")
   # shape: (480 time, 50 lat, 60 lon)

   # create latitude DataArray matching the grid
   lat_array = xr.DataArray(
       temp_grid.coords["lat"].values,
       dims=["lat"],
   )

   # compute PET — automatically broadcasts latitude across grid
   pet_grid = pet_thornthwaite(
       temperature=temp_grid,
       latitude=lat_array,
   )
   # result shape: (480 time, 50 lat, 60 lon)

   # write to NetCDF
   pet_grid.to_netcdf("pet_thornthwaite_grid.nc")


Example 4: Out-of-core SPI with Dask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import xarray as xr
   from climate_indices import spi
   from climate_indices.indices import Distribution

   # open large global dataset with Dask chunking
   # (data not loaded into memory yet)
   precip_global = xr.open_dataarray(
       "CMIP6_pr_global_1850_2100.nc",
       chunks={
           "time": -1,     # single time chunk (required for SPI)
           "lat": 30,      # 6 chunks along latitude
           "lon": 60,      # 6 chunks along longitude
       },
   )

   # compute SPI-12 (builds task graph, no computation yet)
   spi_12_global = spi(
       values=precip_global,
       scale=12,
       distribution=Distribution.gamma,
       calibration_year_initial=1981,
       calibration_year_final=2010,
   )

   # write to NetCDF — triggers Dask computation
   # processes chunks in parallel, streams to disk
   spi_12_global.to_netcdf("spi_12_global_cmip6.nc")


API Reference Links
-------------------

**Index functions:**

- :func:`climate_indices.spi` — Standardized Precipitation Index
- :func:`climate_indices.spei` — Standardized Precipitation Evapotranspiration Index
- :func:`climate_indices.xarray_adapter.pet_thornthwaite` — Potential Evapotranspiration (Thornthwaite)
- :func:`climate_indices.xarray_adapter.pet_hargreaves` — Potential Evapotranspiration (Hargreaves)

**Supporting classes:**

- :class:`climate_indices.indices.Distribution` — Distribution types for fitting
- :class:`climate_indices.compute.Periodicity` — Time series periodicity (monthly/daily)

**Exceptions:**

- :exc:`climate_indices.exceptions.CoordinateValidationError` — Invalid or missing coordinates
- :exc:`climate_indices.exceptions.InputTypeError` — Unsupported input type
- :exc:`climate_indices.exceptions.InsufficientDataError` — Not enough data for calibration
- :exc:`climate_indices.exceptions.InputAlignmentWarning` — Data dropped during multi-input alignment

**Full API reference:**

- :doc:`reference` — Complete API documentation for all modules

**External documentation:**

- `xarray documentation <https://docs.xarray.dev/>`_
- `NumPy documentation <https://numpy.org/doc/stable/>`_
- `Dask documentation <https://docs.dask.org/>`_
- `CF Conventions <http://cfconventions.org/>`_

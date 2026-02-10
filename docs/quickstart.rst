==================
Quickstart Tutorial
==================

.. contents::
   :local:
   :backlinks: none

This 5-minute tutorial shows you how to compute drought indices (PET, SPI, SPEI) using
**climate_indices**. You'll learn both the recommended **xarray API** (which auto-infers parameters
from coordinates) and the **NumPy API** (which requires explicit parameters).

Prerequisites
=============

Install the package using pip or uv:

.. code-block:: bash

   pip install climate-indices
   # or
   uv pip install climate-indices

For visualization examples, also install matplotlib:

.. code-block:: bash

   pip install matplotlib


Create Sample Data
===================

First, let's create synthetic monthly precipitation and temperature data covering 30 years
(360 months). Real-world usage would load data from NetCDF files using ``xr.open_dataset()``.

.. testsetup:: quickstart

   import numpy as np
   import xarray as xr
   import pandas as pd
   from climate_indices import spi, spei, pet_thornthwaite
   from climate_indices.indices import Distribution

   # seed RNG for reproducible synthetic data
   np.random.seed(42)

   # create 30 years of synthetic monthly data with seasonal patterns
   n_years = 30
   n_months = n_years * 12
   months = np.tile(np.arange(1, 13), n_years)

   # precipitation: higher in winter (months 11-2), lower in summer (months 6-8)
   seasonal_precip = 80 + 40 * np.sin(2 * np.pi * (months - 3) / 12)
   precip_raw = seasonal_precip + np.random.normal(0, 20, n_months)
   precip_raw = np.clip(precip_raw, 0, None)

   # temperature: higher in summer, lower in winter
   seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (months - 3) / 12)
   temp_raw = seasonal_temp + np.random.normal(0, 2, n_months)

.. doctest:: quickstart

   >>> # wrap arrays in xarray DataArrays with time coordinates
   >>> time = pd.date_range("1990-01-01", periods=360, freq="MS")
   >>>
   >>> precip_da = xr.DataArray(
   ...     precip_raw,
   ...     coords={"time": time},
   ...     dims=("time",),
   ...     name="precipitation",
   ...     attrs={"units": "mm/month", "long_name": "Monthly precipitation"}
   ... )
   >>>
   >>> temp_da = xr.DataArray(
   ...     temp_raw,
   ...     coords={"time": time},
   ...     dims=("time",),
   ...     name="temperature",
   ...     attrs={"units": "degC", "long_name": "Monthly mean temperature"}
   ... )
   >>>
   >>> # verify shapes
   >>> precip_da.shape
   (360,)
   >>> temp_da.shape
   (360,)

The **time coordinate** is essential for the xarray API — it enables automatic inference of the data
start year and periodicity. The **units attributes** document the expected units: precipitation in
mm/month and temperature in degrees Celsius.


Compute Indices with xarray (Recommended)
==========================================

The xarray API is the recommended approach because it automatically infers parameters like
``data_start_year`` and ``periodicity`` from the time coordinate, reducing boilerplate and errors.


PET (Thornthwaite)
~~~~~~~~~~~~~~~~~~

Potential evapotranspiration (PET) estimates atmospheric water demand. The Thornthwaite method
requires only monthly temperature and latitude:

.. doctest:: quickstart

   >>> pet_result = pet_thornthwaite(temp_da, latitude=40.0)
   >>> pet_result.shape
   (360,)
   >>> pet_result.attrs["long_name"]
   'Potential Evapotranspiration (Thornthwaite method)'

The result is a ``DataArray`` with the same time coordinate and inherited metadata. Latitude is in
decimal degrees (positive for north, negative for south).


SPI (Standardized Precipitation Index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SPI quantifies precipitation anomalies relative to a long-term calibration period. Negative values
indicate drier-than-normal conditions:

.. doctest:: quickstart

   >>> spi_result = spi(precip_da, scale=3, distribution=Distribution.gamma)
   >>> spi_result.shape
   (360,)
   >>> spi_result.attrs["long_name"]
   'Standardized Precipitation Index'

The ``scale`` parameter controls the accumulation window (3 months here). The ``distribution``
parameter selects the fitting distribution (``gamma`` is standard for precipitation). The calibration
period defaults to the full time range.


SPEI (Standardized Precipitation Evapotranspiration Index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SPEI is similar to SPI but accounts for both precipitation and evapotranspiration, making it
sensitive to temperature-driven droughts:

.. doctest:: quickstart

   >>> spei_result = spei(precip_da, pet_result, scale=3, distribution=Distribution.gamma)
   >>> spei_result.shape
   (360,)
   >>> spei_result.attrs["long_name"]
   'Standardized Precipitation Evapotranspiration Index'

SPEI uses the precipitation minus PET (P - PET) as input, representing the water balance.


Save and Load NetCDF
=====================

xarray makes it easy to persist results to disk in NetCDF format, the standard for gridded climate
data:

.. doctest:: quickstart

   >>> import tempfile
   >>> import os
   >>>
   >>> # save to temporary file
   >>> temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
   >>> temp_path = temp_file.name
   >>> temp_file.close()
   >>>
   >>> spi_result.to_netcdf(temp_path)
   >>>
   >>> # load back
   >>> spi_loaded = xr.open_dataarray(temp_path)
   >>> spi_loaded.shape
   (360,)
   >>> spi_loaded.attrs["long_name"]
   'Standardized Precipitation Index'

.. testcleanup:: quickstart

   os.unlink(temp_path)

For multi-variable datasets, use ``xr.Dataset`` instead:

.. code-block:: python

   ds = xr.Dataset({
       "precipitation": precip_da,
       "temperature": temp_da,
       "pet": pet_result,
       "spi_3": spi_result,
       "spei_3": spei_result,
   })
   ds.to_netcdf("results.nc")

   # later
   ds_loaded = xr.open_dataset("results.nc")


Compute Indices with NumPy
===========================

The NumPy API provides explicit control over all parameters but requires more boilerplate. All
temporal parameters must be specified manually:

.. doctest:: quickstart

   >>> from climate_indices.indices import spi as spi_numpy
   >>> from climate_indices.indices import spei as spei_numpy
   >>> from climate_indices.indices import pet
   >>> from climate_indices.compute import Periodicity
   >>>
   >>> # extract raw NumPy arrays
   >>> precip_values = precip_da.values
   >>> temp_values = temp_da.values
   >>>
   >>> # compute PET (Thornthwaite)
   >>> pet_np = pet(
   ...     temp_values,
   ...     latitude_degrees=40.0,
   ...     data_start_year=1990
   ... )
   >>> pet_np.shape
   (360,)
   >>>
   >>> # compute SPI with explicit calibration period and periodicity
   >>> spi_np = spi_numpy(
   ...     precip_values,
   ...     scale=3,
   ...     distribution=Distribution.gamma,
   ...     data_start_year=1990,
   ...     calibration_year_initial=1990,
   ...     calibration_year_final=2019,
   ...     periodicity=Periodicity.monthly
   ... )
   >>> spi_np.shape
   (360,)
   >>>
   >>> # compute SPEI using P - PET
   >>> spei_np = spei_numpy(
   ...     precip_values,
   ...     pet_np,
   ...     scale=3,
   ...     distribution=Distribution.gamma,
   ...     data_start_year=1990,
   ...     calibration_year_initial=1990,
   ...     calibration_year_final=2019,
   ...     periodicity=Periodicity.monthly
   ... )
   >>> spei_np.shape
   (360,)

The NumPy API is useful when working with legacy code or when xarray's overhead is undesirable for
very large computations. However, the xarray API is recommended for most use cases because it reduces
parameter redundancy and preserves metadata.


Visualize Results
=================

Plotting time series helps verify that indices respond correctly to precipitation and temperature
patterns:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)

   # plot SPI
   spi_result.plot(ax=axes[0], color="purple")
   axes[0].axhline(0, color="black", linewidth=0.8)
   axes[0].set_title("SPI-3 (gamma distribution)")
   axes[0].set_ylabel("Standardized units")

   # plot SPEI
   spei_result.plot(ax=axes[1], color="green")
   axes[1].axhline(0, color="black", linewidth=0.8)
   axes[1].set_title("SPEI-3 (gamma distribution)")
   axes[1].set_ylabel("Standardized units")

   # plot PET
   pet_result.plot(ax=axes[2], color="orange")
   axes[2].set_title("PET (Thornthwaite)")
   axes[2].set_ylabel("mm/month")

   plt.tight_layout()
   plt.show()

.. note::

   Install matplotlib for visualization: ``pip install matplotlib``


Next Steps
==========

Now that you've computed your first drought indices, explore these resources:

- **xarray Migration Guide**: :doc:`xarray_migration` — Learn how to transition from the legacy
  NumPy API to the xarray API and understand chunking strategies for large datasets.

- **API Reference**: :doc:`reference` — Complete documentation of all functions, parameters, and
  distributions.

- **Example Datasets**: Download sample NetCDF files from the
  `NOAA Climate Prediction Center <https://www.cpc.ncep.noaa.gov/>`_ or
  `NCAR Climate Data Guide <https://climatedataguide.ucar.edu/>`_ to practice with real data.

- **Multi-dimensional Grids**: The xarray API supports 2-D, 3-D, and 4-D arrays (e.g., time × lat ×
  lon). Simply pass gridded ``DataArray`` objects — the functions broadcast automatically.

- **Dask Integration**: For out-of-core computation on datasets larger than memory, chunk your data
  along spatial dimensions and keep the time dimension as a single chunk. See the xarray migration
  guide for details.

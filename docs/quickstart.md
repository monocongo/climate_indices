# Quickstart Tutorial

This tutorial computes PET, SPI, and SPEI from one synthetic 30-year series using
the recommended xarray API. It takes about five minutes. Real inputs reach the
same functions; the input contract is in {doc}`data_requirements`.

## Install

Requires Python 3.10+. Install the package from PyPI:

```bash
pip install climate-indices
# or
uv pip install climate-indices
```

## Create sample data

Create 30 years of synthetic monthly precipitation and temperature (360 months):

```{testsetup} quickstart
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
```

Label the arrays with a time coordinate, which also carries the start year and
periodicity that the xarray API infers:

```{doctest} quickstart
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
```

:::{warning}
**Beta Feature**

The xarray API shown below is **beta** and may change in future minor releases.
Computation results are identical to the stable NumPy array API.
:::

## Compute PET

PET (Thornthwaite) needs monthly temperature and latitude:

```{doctest} quickstart
>>> pet_result = pet_thornthwaite(temp_da, latitude=40.0)
>>> pet_result.shape
(360,)
>>> pet_result.attrs["long_name"]
'Potential Evapotranspiration (Thornthwaite method)'
```

The result keeps the time coordinate and the metadata. Latitude is in decimal
degrees, positive north.

## Compute SPI

SPI compares precipitation against its own long-term calibration period:

```{doctest} quickstart
>>> spi_result = spi(precip_da, scale=3, distribution=Distribution.gamma)
>>> spi_result.shape
(360,)
>>> spi_result.attrs["long_name"]
'Standardized Precipitation Index'
```

`scale` is the accumulation window in months; the calibration period defaults to
the full time range.

## Compute SPEI

SPEI is SPI on the water balance, so it also responds to temperature:

```{doctest} quickstart
>>> spei_result = spei(precip_da, pet_result, scale=3, distribution=Distribution.gamma)
>>> spei_result.shape
(360,)
>>> spei_result.attrs["long_name"]
'Standardized Precipitation Evapotranspiration Index'
```

## Save and reopen the results

Write the SPI result to NetCDF and read it back:

```{doctest} quickstart
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
```

```{testcleanup} quickstart
os.unlink(temp_path)
```

{doc}`writing-outputs` covers Zarr output, multi-variable datasets, and writes
that do not fit in memory.

## Plot the results

Plotting confirms the result carries the input time axis. PET follows the
synthetic seasonal cycle; SPI and SPEI are standardized per calendar month, so
they show anomalies rather than the cycle:

```python
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
```

:::{note}
Install matplotlib for visualization: `pip install matplotlib`
:::

## Next steps

- Follow a longer worked example in the {doc}`tutorials` index.
- Move an existing NumPy-array workflow to this API with {doc}`xarray_migration`.
- Look up every function, parameter, and distribution in {doc}`reference`.

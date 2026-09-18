# Tutorials

Tutorials teach the library end to end: they start from nothing and walk through a
complete, runnable task.

```{toctree}
:maxdepth: 1

quickstart
```

The end-to-end xarray and Dask walkthrough is
[notebooks/zarr_dask_spi_spei.ipynb](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb).
It computes SPI and SPEI from a prepared Zarr store, persists the results, and
reopens them for diagnostics. Prepare the pinned sample inputs and execute it
from a fresh kernel with:

```bash
bash scripts/smoke_e2e_notebook.sh
```

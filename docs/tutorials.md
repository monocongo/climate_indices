# Tutorials

Tutorials teach the library end to end: start from nothing, follow the steps in
order, and finish with a working result. Begin with the quickstart, then open a
notebook when you want to see a workflow on a fuller example.

```{toctree}
:maxdepth: 1

quickstart
```

The notebook tutorials run from repository checkouts with a development
environment (`uv sync --group dev`) and a kernel started in `notebooks/`. Jupyter
is in the `dev` dependency group.

## Notebook tutorials

| Notebook | What it teaches | Data prerequisites | Approximate runtime | CI verification |
| --- | --- | --- | --- | --- |
| [xarray_getting_started] | Computing SPI from a labeled monthly precipitation series, and checking that coordinates and metadata survive the xarray adapter path | None; the notebook generates a synthetic series | Seconds | Executed by the `notebooks` CI job |
| [palmer_indices_xarray] | The Palmer bridge pattern: keep labeled xarray inputs at workflow boundaries, call `palmer.pdsi()` on values, then rewrap the outputs with coordinates and metadata | None; the notebook generates a synthetic series | Seconds | Executed by the `notebooks` CI job |
| [eddi_xarray] | Computing EDDI from a labeled monthly PET series through the typed public API | None; the notebook generates a synthetic series | Seconds | Executed by the `notebooks` CI job |
| [zarr_dask_spi_spei] | The end-to-end workflow: prepared Zarr → inspect and validate → compute SPI/SPEI with Dask → write Zarr → reopen → maps and time series | Prepared sample store; generate it once with `uv run --group dev scripts/prepare_e2e_inputs.py` (about 5 MB downloaded on the first run, then cached under the git-ignored `data/e2e/`) | About 15 seconds once the store is prepared | Executed by `tests/test_e2e_with_dask.py` in the core test suite, not the notebook job (#917); the notebook's `dask.distributed` client cell runs only through `scripts/smoke_e2e_notebook.sh` locally (#829) |
| [fire_weather_demo] | The fire-weather family (KBDI, CFFWIS, Fosberg FFWI, HDW) over a CONUS subset, plus an SPI-3/EDDI-3 baseline for the 2020 fire season | Prepared ERA5 subset; generate it once with `uv run --group dev scripts/prepare_fire_demo_inputs.py` (multi-gigabyte download into the git-ignored `data/fire-demo/`, about half an hour on the first run) | Notebook execution not tracked; preparation is about 30 minutes | Not executed in CI |
| [flood_event_brisbane_2011] | I_F, API, and the SPI wet tail on gridded daily precipitation around the January 2011 Brisbane and Lockyer Valley flood event, with the flood-potential-not-flooding caveat | NOAA PSL CPC daily precipitation; the notebook downloads the 33 yearly subsets itself on the first run (about 250 KB each) and caches them under the git-ignored `data/flood-demo/` | Seconds with the cache; a few minutes on the first run | Not executed in CI; run `scripts/smoke_flood_demo_notebook.sh` locally |

The three notebooks the `notebooks` CI job executes are verified on every pull
request targeting `main` by the workflow step `Execute 3.0.0 notebooks`; changes
that break them fail CI. The Zarr/Dask notebook is verified by the core test
suite instead, apart from its local-only Dask client cell; the fire-weather and
flood-event demos are manual examples that no CI job runs.

## Unmaintained notebooks

The notebooks below are not maintained against the current API, and no CI job
executes them. Their data prerequisites differ and are listed per row; several
expect local files that are not in the repository.

| Notebook | What it covers | Data prerequisites | Approximate runtime | Coverage |
| --- | --- | --- | --- | --- |
| [concurrent_shared_memory_example] | Sharing a large array across worker processes with `multiprocessing.shared_memory` | None; synthetic arrays | Not tracked | Not executed; standalone concurrency demo |
| [generate_fitting_parameters_nclimgrid] | Pre-computing SPI gamma fitting parameters for a gridded nClimGrid dataset | A local `nclimgrid_lowres_prcp.nc` at an absolute path | Not tracked | Not executed; local-data workflow notes |
| [muitprocess_spi_nclimgrid] | Gridded SPI from shared-memory multiprocessing over nClimGrid | A local `nclimgrid_lowres_prcp.nc` at an absolute path | Not tracked | Not executed; local-data workflow notes |
| [spi_simple] | Comparing loop-based SPI implementations over an nClimGrid grid | A local `nclimgrid_lowres_prcp.nc` at an absolute path | Not tracked | Not executed and imports removed internals; may not run |
| [visualize_precip_spi] | Mapping CHIRPS precipitation and SPI with cartopy | Local CHIRPS NetCDF files at absolute paths | Not tracked | Not executed; local-data plotting notes |

Runtimes are not tracked because no CI job executes these notebooks.

See {doc}`troubleshooting` for the Zarr/Dask workflow's setup and failure modes,
{doc}`xarray_migration` for the xarray stability guarantees, and
[ADR-0003](https://github.com/monocongo/climate_indices/blob/main/docs/adr/0003-dask-time-dimension-single-chunk.md)
for the single-time-chunk correctness constraint.

[xarray_getting_started]: https://github.com/monocongo/climate_indices/blob/main/notebooks/xarray_getting_started.ipynb
[palmer_indices_xarray]: https://github.com/monocongo/climate_indices/blob/main/notebooks/palmer_indices_xarray.ipynb
[eddi_xarray]: https://github.com/monocongo/climate_indices/blob/main/notebooks/eddi_xarray.ipynb
[zarr_dask_spi_spei]: https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb
[fire_weather_demo]: https://github.com/monocongo/climate_indices/blob/main/notebooks/fire_weather_demo.ipynb
[flood_event_brisbane_2011]: https://github.com/monocongo/climate_indices/blob/main/notebooks/flood_event_brisbane_2011.ipynb
[concurrent_shared_memory_example]: https://github.com/monocongo/climate_indices/blob/main/notebooks/concurrent_shared_memory_example.ipynb
[generate_fitting_parameters_nclimgrid]: https://github.com/monocongo/climate_indices/blob/main/notebooks/generate_fitting_parameters_nclimgrid.ipynb
[muitprocess_spi_nclimgrid]: https://github.com/monocongo/climate_indices/blob/main/notebooks/muitprocess_spi_nclimgrid.ipynb
[spi_simple]: https://github.com/monocongo/climate_indices/blob/main/notebooks/spi_simple.ipynb
[visualize_precip_spi]: https://github.com/monocongo/climate_indices/blob/main/notebooks/visualize_precip_spi.ipynb

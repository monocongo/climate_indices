# Write NetCDF and Zarr outputs

Goal: persist computed index values to NetCDF or Zarr, and confirm the write
succeeded. The command line and the xarray API both compute index values; this
page covers where those results go. The writer options are listed in the
[command-line reference](reference.md#command-line-interface); the measured
write behavior (Zarr streaming versus the NetCDF backend lock) is in
[Operational Guidance](xarray_compatibility.md#operational-guidance).

## Prerequisites

- A finished `climate_indices` run, or the result of {func}`climate_indices.spi`
  and its siblings.
- Input, output, and staging paths that do not overlap: a rerun must never read the
  store it is replacing.
- For a Dask-backed result, `time` in a single chunk and spatial chunks sized as the
  parallelism you want — see {doc}`performance`.

## Write NetCDF from the command line

`--output_file_base` is required, and each computed index becomes its own file named
`<output_file_base>_<index>_<distribution>_<scale>.nc`:

```bash
climate_indices --index spi --periodicity monthly --scales 3 6 \
  --netcdf_precip /data/nclimgrid_prcp.nc --var_name_precip prcp \
  --output_file_base /data/out/indices \
  --calibration_start_year 1981 --calibration_end_year 2010 \
  --chunksizes input
```

That run writes `/data/out/indices_spi_gamma_03.nc`, `..._spi_pearson_03.nc`,
`..._spi_gamma_06.nc`, and `..._spi_pearson_06.nc`: SPI and SPEI are computed for
every distribution and every requested scale, since the command has no
distribution option. `--chunksizes` (default `none`) sets the layout of the
written file, not of the computation; `input` copies the first chunked input
variable's on-disk chunks. An index that needs PET comes out with a PET side-effect
file when temperature is given instead of PET. Copy-and-adapt invocations are in
{doc}`workflow-examples`.

## Write Zarr from a lazy xarray result

Zarr is the right target for a Dask-backed result: the write streams each finished
block to disk instead of collecting the whole grid in the client process.

```python
import xarray as xr

from climate_indices import spi
from climate_indices.indices import Distribution


def main() -> None:
    precip = xr.open_zarr("cache_prepared_input.zarr", consolidated=True)["precip"]
    spi_lazy = spi(
        values=precip.chunk({"time": -1, "lat": 10, "lon": 10}),
        scale=3,
        distribution=Distribution.gamma,
        calibration_year_initial=1981,
        calibration_year_final=2010,
    )

    # Name the variables in a Dataset, as the notebook does, so the store opens
    # by index name. compute=False keeps the write lazy, so the blocks stream
    # under the scheduler that suits the fitting work.
    output = xr.Dataset({"spi": spi_lazy})
    output.to_zarr("out/spi.zarr", zarr_format=2, consolidated=True, compute=False).compute(
        scheduler="processes"
    )


# The guard is required in a script: the processes scheduler spawns workers that
# re-import this module.
if __name__ == "__main__":
    main()
```

Write index results to a store of their own, never back into the prepared inputs,
and keep the paths non-overlapping. A rerun stages the complete store beside the
target and swaps it in after a successful write, so a failed calculation leaves the
previous store intact. Close readers before rerunning, keep one writer per output
path, and treat the final directory swap as not crash-atomic — the notebook's write
cell ([notebooks/zarr_dask_spi_spei.ipynb](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb))
is the staged example.

## Write NetCDF from a lazy xarray result

The NetCDF backend lock is built for the scheduler that was active when the write
graph was built and cannot be pickled to worker processes, so a lazy Dask result
does not go straight to `.to_netcdf()`. Load the result first when it fits in memory
(`spi_lazy.compute().to_netcdf("out/spi.nc")`), or let a `dask.distributed` client
stream the write for a result that does not. Both paths are in
[Operational Guidance](xarray_compatibility.md#operational-guidance).

## Verify the write

Reopen the output and check the provenance metadata that makes it interpretable
later, without loading the values:

```python
import xarray as xr

reopened = xr.open_zarr("out/spi.zarr", consolidated=True)
attrs = reopened["spi"].attrs
print(attrs["scale"], attrs["distribution"], attrs["calibration_year_initial"])
assert attrs["climate_indices_version"]
# The first scale - 1 outputs have no full window, so they are NaN by design.
assert bool(reopened["spi"].isel(time=slice(0, 2)).isnull().all())
```

The result carries `scale`, `distribution`, the calibration years,
`climate_indices_version`, and a `history` entry. A command-line NetCDF output
identifies the distribution and scale in its file name and the index in the
variable's `long_name`; check with
`xr.open_dataset("/data/out/indices_spi_gamma_03.nc")["spi_gamma_03"].attrs`. If a
write fails or a reader rejects the store, the recipes in {doc}`troubleshooting`
cover the failure modes.

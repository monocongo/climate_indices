# Write NetCDF, Zarr, and Icechunk outputs

Goal: persist computed index values to NetCDF, Zarr, or Icechunk, and confirm the write
succeeded. The command line and the xarray API both compute index values; this
page covers where those results go. The writer options are listed in the
[command-line reference](reference.md#command-line-interface); the measured
write behavior (Zarr streaming versus the NetCDF backend lock) is in
[Operational Guidance](xarray_compatibility.md#operational-guidance).

## Prerequisites

- A finished `climate_indices` run, or the result of {func}`climate_indices.spi`
  and its siblings.
- The `zarr` package for a Zarr output; it is not installed with the base
  package.
- Input, output, and staging paths that do not overlap: a rerun must never read the
  store it is replacing.
- For a Dask-backed result, `time` in a single chunk and spatial chunks sized as the
  parallelism you want — see {doc}`performance`.

## Write NetCDF from the command line

`--output_file_base` is required, and each computed index becomes its own file
named after the index's variable. SPI and SPEI files name the distribution and
scale, `<output_file_base>_<index>_<distribution>_<scale>.nc`:

```bash
climate_indices --index spi --periodicity monthly --scales 3 6 \
  --netcdf_precip /data/nclimgrid_prcp.nc --var_name_precip prcp \
  --output_file_base /data/out/indices \
  --calibration_start_year 1981 --calibration_end_year 2010
```

That run writes `/data/out/indices_spi_gamma_03.nc`, `..._spi_pearson_03.nc`,
`..._spi_gamma_06.nc`, and `..._spi_pearson_06.nc`: SPI and SPEI are computed for
every distribution and every requested scale, since the command has no
distribution option. The other indices use their own variable name instead, for
example `..._pnp_03.nc` and `..._pet_thornthwaite.nc`. An index that needs PET
comes out with a PET side-effect file when temperature is given instead of PET.
Copy-and-adapt invocations, including the output-layout option, are in
{doc}`workflow-examples` and the [command-line reference](reference.md#command-line-interface).

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
and keep the paths non-overlapping. The call above creates the store; to replace
an existing one, a rerun stages the complete store beside the target and swaps it
in after a successful write, so a failed calculation leaves the previous store
intact. Close readers before rerunning, keep one writer per output path, and treat
the final directory swap as not crash-atomic — the notebook's write cell
([notebooks/zarr_dask_spi_spei.ipynb](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb))
is the staged example.

## Read and write Icechunk snapshots

Install the optional extra with `pip install 'climate_indices[icechunk]'`, which
installs `icechunk>=1.1.21,<2`. Icechunk requires Python 3.11 or later and Zarr v3.
The existing `xarray>=2025.6.1` requirement exceeds the `to_icechunk` floor of
2025.1.1. The core install is unchanged: without the extra, the base package keeps
its Python 3.10 support and installs no Icechunk dependency.

Use `open_icechunk` to read a committed snapshot. Select a variable and pass it
to the existing index API. The reader preserves stored chunks; put the full time
axis in one chunk before fitting.

```python
import icechunk

from climate_indices import spi
from climate_indices.icechunk_io import open_icechunk, write_icechunk
from climate_indices.indices import Distribution

input_repo = icechunk.Repository.open(
    icechunk.local_filesystem_storage("prepared_input.icechunk")
)
output_repo = icechunk.Repository.open_or_create(
    icechunk.local_filesystem_storage("out/indices.icechunk")
)
with open_icechunk(input_repo) as inputs:
    spi_lazy = spi(
        inputs["precip"].chunk({"time": -1, "lat": 10, "lon": 10}),
        scale=3,
        distribution=Distribution.gamma,
        calibration_year_initial=1981,
        calibration_year_final=2010,
    )
    snapshot_id = write_icechunk(
        spi_lazy.to_dataset(name="spi").chunk({"time": 12}),
        output_repo,
        message="Compute SPI-3",
    )

with open_icechunk(output_repo) as committed:
    print(snapshot_id, committed["spi"].attrs)
```

The [Icechunk SPI and SPEI notebook](https://github.com/monocongo/climate_indices/blob/main/notebooks/icechunk_spi_spei.ipynb)
runs this workflow end to end on a synthetic grid. It needs no prepared data and no
network access.

`write_icechunk` calls `icechunk.xarray.to_icechunk(dataset, session, mode=...)`,
then `session.commit(message)`. It completes all Dask writes before committing.
Use `to_icechunk`, never `Dataset.to_zarr`, for Icechunk, including distributed
or multiprocessing writes: it records worker writes in the session.
Use `to_zarr` for ordinary Zarr stores as before.

Icechunk stores return the same SPI and SPEI values as NetCDF and Zarr stores for
the same inputs; [`tests/test_icechunk_io.py`](https://github.com/monocongo/climate_indices/blob/main/tests/test_icechunk_io.py)
compares all three backends against
the Python/SciPy oracle.

Icechunk does not support consolidated metadata. The reader explicitly calls
`xr.open_zarr(session.store, consolidated=False, chunks={})`, and `to_icechunk`
disables consolidation internally. Do not use the ordinary Zarr example's
`consolidated=True` settings with Icechunk.

The [Dask SPI/SPEI notebook](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb)
can write its existing `ds_output` through the same single session. Replace its
Zarr staging and directory-swap block with
`write_icechunk(ds_output, output_repo, message="Compute SPI/SPEI")`.
Keep the client open until the helper returns. A threaded scheduler also works.
Dask scheduling does not select the numerical backend; the existing optional
Rust dispatch and Python fallback remain unchanged.

Keep input and output repositories separate, and use one writer per output
branch. The default `mode="w-"` rejects an existing group. For an intentional
replacement, use `mode="w"`: it changes only the new snapshot and preserves
previous snapshots. A failed write does not advance the branch. Never delete
or recreate an existing repository to retry a write. The helpers propagate write
and commit errors; they do not merge concurrent sessions or retry conflicts.
The optional `branch` argument selects an existing branch (default `"main"`).

## Write NetCDF from a lazy xarray result

The NetCDF backend lock is built for the scheduler that was active when the write
graph was built and cannot be pickled to worker processes, so a lazy Dask result
does not go straight to `.to_netcdf()`. Load the result first when it fits in memory
(`spi_lazy.compute().to_netcdf("out/spi.nc")`), or let a `dask.distributed` client
stream the write for a result that does not. Both paths are in
[Operational Guidance](xarray_compatibility.md#operational-guidance).

Write several results into one file by naming each in a Dataset:

```python
import xarray as xr

combined = xr.Dataset({
    "pet": pet,         # one DataArray per computed index
    "spi_3": spi_3,
    "spei_3": spei_3,
})
combined.to_netcdf("out/indices.nc")

# later
loaded = xr.open_dataset("out/indices.nc")
```

## Verify the write

Reopen the output and check the metadata that makes it interpretable later,
without loading the values:

```python
import xarray as xr

reopened = xr.open_zarr("out/spi.zarr", consolidated=True)
attrs = reopened["spi"].attrs
print(attrs["scale"], attrs["distribution"], attrs["calibration_year_initial"])
assert attrs["climate_indices_version"]
# The first scale - 1 outputs have no full window, so they are NaN by design.
assert bool(reopened["spi"].isel(time=slice(0, 2)).isnull().all())
```

The xarray result carries `scale`, `distribution`, the requested calibration
years, `climate_indices_version`, and a `history` entry. A command-line NetCDF
output carries the same provenance: its variable attributes come from the CF
metadata registry (`long_name`, `units`, `references`, and any registered
variant), plus `climate_indices_version` and a `history` entry, and the scale
and distribution where they apply. The distribution and scale also appear in
the file name, so keep the run arguments alongside it:
`xr.open_dataset("/data/out/indices_spi_gamma_03.nc")["spi_gamma_03"].attrs`
shows what the file itself does and does not record.

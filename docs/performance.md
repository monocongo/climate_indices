# Gridded Performance

:::{note}
The gridded xarray API is beta in 3.0.0. The NumPy API remains the stable
integration surface; this page covers the Dask execution path for the xarray
adapters described in the [xarray compatibility matrix](xarray_compatibility.md).
:::

The gridded index kernels are vectorized once per spatial block: a Dask-backed
`(time, lat, lon)` input reaches the NumPy core once per block instead of once
per grid cell. Chunking, memory, and worker processes are then Dask's job, and
the choices that matter are the chunk shape and the scheduler. This page gives a
complete runnable example and the measured numbers behind the recommendations in
[Operational Guidance](xarray_compatibility.md#operational-guidance) and
[Chunking guidance for gridded indices](xarray_compatibility.md#chunking-guidance-for-gridded-indices).

## Runnable example: gridded SPI and SPEI

This computes 3-month SPI and SPEI for every cell of a synthetic monthly grid,
with `time` in one chunk and spatial blocks driving the parallelism. Both
indices are materialized in one `dask.compute` call, so they share the input
blocks and one worker pool.

```python
import dask
import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import spei, spi
from climate_indices.indices import Distribution

def main() -> None:
    time = pd.date_range("1981-01-01", periods=40 * 12, freq="MS")
    lat = np.linspace(25.0, 49.0, 25)
    lon = np.linspace(-125.0, -101.0, 25)
    shape = (time.size, lat.size, lon.size)
    rng = np.random.default_rng(42)

    precip = xr.DataArray(
        rng.gamma(shape=2.0, scale=15.0, size=shape),
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        attrs={"units": "mm"},
    )
    pet = xr.DataArray(
        np.full(shape, 60.0),
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        attrs={"units": "mm"},
    )

    # Time in one chunk is required; spatial chunks are the parallelism and memory lever.
    chunks = {"time": -1, "lat": 10, "lon": 10}
    precip = precip.chunk(chunks)
    pet = pet.chunk(chunks)

    # No Python loop over cells: each index runs once per spatial block.
    spi_lazy = spi(
        values=precip,
        scale=3,
        distribution=Distribution.gamma,
        calibration_year_initial=1981,
        calibration_year_final=2010,
    )
    spei_lazy = spei(
        precips_mm=precip,
        pet_mm=pet,
        scale=3,
        distribution=Distribution.gamma,
        calibration_year_initial=1981,
        calibration_year_final=2010,
    )

    # One process pool and one shared input graph for both indices. The
    # `__main__` guard keeps spawned workers from re-running this block.
    spi_grid, spei_grid = dask.compute(spi_lazy, spei_lazy, scheduler="processes")


if __name__ == "__main__":
    main()
```

`spi_grid` and `spei_grid` are `DataArray`s with the input coordinates, one
value per cell and month, computed without a per-cell Python loop. With
synthetic values the gamma goodness-of-fit check can report
`GoodnessOfFitWarning` for a few cell-months; that is the fit report, not a
failure, and under `scheduler="processes"` it is raised in the workers (their
stderr), not catchable in the caller. Give multi-input indices such as SPEI the
same chunking on the dimensions they share with the other input: each input's
`time` chunking is validated independently, and a mismatch on `lat`/`lon`
survives into the compute as a Dask `rechunk-merge` copy.

## Measured speedup

Vectorization is what removes the per-cell Python loop, so its effect is
measured against the serial in-memory call. On the #893 reference grid
(38 x 87 cells, 40 years monthly, scale 3, fastest of three runs after a warm-up
with logging quiet so the fit is isolated; the speedups compare quiet-logging
serial runs and are ±15% rather than exact; Python 3.14.7, 10-core macOS arm64):

| index | serial before | serial after | vectorization speedup |
| --- | ---: | ---: | ---: |
| SPI | 0.898 s | 0.204 s | 4.4x |
| SPEI | 0.783 s | 0.220 s | 3.6x |
| Thornthwaite PET | 1.063 s | 0.020 s | 53x |
| EDDI | 14.717 s | 0.043 s | 342x |

The "serial before" column is the last commit before the spatial-block
conversion; the per-cell logging volume disappeared with the loop, collapsing
the INFO-logged figures as well. The raw runs, the full command, and the
end-to-end Dask numbers are in
[the #929 before/after table](https://github.com/monocongo/climate_indices/blob/main/benchmarks/README.md#beforeafter-on-the-reference-grid-929).

On this grid the Dask path does not beat the serial call for SPI, SPEI, or PET:
after vectorization each finishes in about 0.2 s or less, while a fresh
`processes` pool plus result transfer adds roughly 0.7 s at one worker. EDDI,
whose pre-vectorization call took 14.7 s, is the only index whose end-to-end Dask
figure still clears 10x (0.699 s, ~21x, and at a single worker): the win is the
vectorization surviving the fixed pool cost, not the parallelism. Parallelism
pays when the work per block exceeds that fixed cost -- larger grids, longer
series, or a worker pool that outlives one compute call.

## Scheduler

Choose the scheduler at the materialization call; the xarray API never imports or
configures Dask ([ADR-0002](adr/0002-multiprocessing-cli-dask-xarray.md)):

```python
spi_grid = spi_lazy.compute(scheduler="processes")
```

- Use `scheduler="processes"` for the CPU-bound fitting and ranking kernels: the
  default threaded scheduler does not run their Python-level portion in parallel.
- Every `.compute(scheduler="processes")` builds and tears down its own pool, so
  short or repeated computations spend most of their wall clock there. Keep a
  `dask.distributed.Client` alive across calls (`distributed` ships with the
  `dev` extra), pass a pre-created pool as
  `dask.compute(..., scheduler="processes", pool=pool)`, or compute several lazy
  results in one `dask.compute` call as in the example above.
- Dask's processes scheduler batches up to six ready tasks per submission by
  default, which can flatten scaling on large grids; pass `chunksize=1` to
  `dask.compute` (or set `dask.config.set({"chunksize": 1})`) so each ready
  block is submitted as soon as a worker is free.
- The threaded scheduler remains the right choice for I/O-bound work, reductions
  such as `.mean("time")`, and small in-memory grids where serializing each block
  to a worker costs more than the parallelism saves.

The full scheduler guidance, including Zarr and NetCDF write behavior, is in
[Operational Guidance](xarray_compatibility.md#operational-guidance).

## Chunking

Keep `time` in a single chunk and size spatial blocks by cell count, not shape.
The PET adapters (`pet_thornthwaite`, `pet_hargreaves`) are the exception: they
accept a split `time` and rechunk it internally, paying that copy inside every
call. The measured working set of a monthly gamma fit is about 60 KB per cell,
so a 100 MB per-block budget is roughly 1,600 cells. That figure is a per-block
measurement taken with `scheduler="synchronous"`, one block resident at a time;
a threaded or distributed worker can hold several ready blocks plus their
inputs and outputs, so budget from the memory a worker can spare per block it
runs at once. The reference grid measures well under that at `10 x 10` to
`20 x 20`. Daily grids carry up to 366 steps per cell-year instead of 12 and
want blocks near `7 x 7`. Chunk both spatial dimensions rather than long rows,
and leave at least a few blocks per worker.

Rechunk once, at read or prepare time: `.chunk(...)` only sets the layout of a
lazy graph, so persist the rechunked array (or write the layout back to the
store) before the index calls to pay the copy once. The measured block table and
the daily-grid caveat are in
[Chunking guidance for gridded indices](xarray_compatibility.md#chunking-guidance-for-gridded-indices).

## Measuring your own

Re-run the reference harness on your own machine and dependency set rather than
trusting these numbers. The scaling harness covers SPI, SPEI, PET, and EDDI from
one to N workers:

```bash
uv run benchmarks/parallel_scaling.py --indices spi,spei,pet,eddi --repeat 3
```

`benchmarks/profile_gridded_spi.py` is the `cProfile` entry point for
attributing a slow SPI pass, and
[`benchmarks/README.md`](https://github.com/monocongo/climate_indices/blob/main/benchmarks/README.md)
documents both harnesses and the committed raw output.

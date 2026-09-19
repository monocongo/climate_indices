"""Tests for helpers shared by the climate_indices command-line interfaces."""

import argparse

import dask
import numpy as np
import xarray as xr

from climate_indices import _cli, compute
from climate_indices._cli import _add_common_spi_arguments

_COMMON_ARGV = [
    "--periodicity",
    "monthly",
    "--scales",
    "1",
    "3",
    "--calibration_start_year",
    "1981",
    "--calibration_end_year",
    "2010",
    "--netcdf_precip",
    "precip.nc",
    "--var_name_precip",
    "prcp",
    "--output_file_base",
    "output",
]


def test_default_chunk_size_drives_auto_chunking(monkeypatch, tmp_path):
    """Auto chunk shapes resolve against the module default, not Dask's own."""
    dataset = xr.Dataset(
        {"precip": (("time", "lat", "lon"), np.ones((4, 20, 20), dtype="float32"))},
    )
    path = str(tmp_path / "precip.nc")
    dataset.to_netcdf(path)
    configured = dask.config.get("array.chunk-size")

    # a tiny budget splits both spatial axes; Dask's own default would not
    monkeypatch.setattr(_cli, "DEFAULT_ARRAY_CHUNK_SIZE", "1 kB")
    with _cli._open_with_default_chunks(
        xr.open_dataset,
        path,
        chunks={"lat": "auto", "lon": "auto", "time": -1},
    ) as opened:
        assert len(opened["precip"].chunks[1]) > 1
        assert len(opened["precip"].chunks[2]) > 1

    # the library default is scoped to the open, not left in the caller's config
    assert dask.config.get("array.chunk-size") == configured


def test_configured_chunk_size_overrides_the_default(monkeypatch, tmp_path):
    """A caller-configured array chunk size wins over the library default."""
    dataset = xr.Dataset(
        {"precip": (("time", "lat", "lon"), np.ones((4, 20, 20), dtype="float32"))},
    )
    path = str(tmp_path / "precip.nc")
    dataset.to_netcdf(path)

    # the library's own budget is far too large to split these axes
    monkeypatch.setattr(_cli, "DEFAULT_ARRAY_CHUNK_SIZE", "100 MB")
    with dask.config.set({"array.chunk-size": "1 kB"}):
        with _cli._open_with_default_chunks(
            xr.open_dataset,
            path,
            chunks={"lat": "auto", "lon": "auto", "time": -1},
        ) as opened:
            assert len(opened["precip"].chunks[1]) > 1
            assert len(opened["precip"].chunks[2]) > 1
            assert dask.config.get("array.chunk-size") == "1 kB"


def test_common_arguments_parse():
    parser = argparse.ArgumentParser()
    _add_common_spi_arguments(parser)

    args = parser.parse_args(_COMMON_ARGV)

    assert args.periodicity is compute.Periodicity.monthly
    assert args.scales == [1, 3]
    assert args.calibration_start_year == 1981
    assert args.calibration_end_year == 2010
    assert args.netcdf_precip == "precip.nc"
    assert args.var_name_precip == "prcp"
    assert args.output_file_base == "output"
    assert args.multiprocessing == "all_but_one"

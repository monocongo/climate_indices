"""Tests for helpers shared by the climate_indices command-line interfaces."""

import argparse
from types import SimpleNamespace

import pytest

from climate_indices import _cli, compute
from climate_indices._cli import _add_common_spi_arguments, _prepare_file

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


def _patch_dims(monkeypatch, dimensions):
    """Patch the shared module's xarray dataset lookup to report the given dims."""
    dataset = {"prcp": SimpleNamespace(dims=dimensions)}
    monkeypatch.setattr(_cli, "xr", SimpleNamespace(open_dataset=lambda *_args: dataset))


@pytest.mark.parametrize(
    "dimensions",
    [
        ("time",),
        ("lat", "lon"),
        ("lat", "lon", "time"),
        ("division",),
        ("division", "time"),
    ],
)
def test_prepare_file_accepts_supported_dimensions(monkeypatch, dimensions):
    _patch_dims(monkeypatch, dimensions)

    assert _prepare_file("input.nc", "prcp") == "input.nc"


@pytest.mark.parametrize(
    "dimensions",
    [
        (),
        ("lat",),
        ("lat", "lon", "time", "extra"),
        ("division", "lat", "lon"),
        ("divisions", "time"),
    ],
)
def test_prepare_file_rejects_unsupported_dimensions(monkeypatch, dimensions):
    _patch_dims(monkeypatch, dimensions)

    with pytest.raises(ValueError, match="dimensions"):
        _prepare_file("input.nc", "prcp")


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

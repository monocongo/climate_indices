"""Offline regressions for the soil-source benchmark orchestration."""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices.aws_ingest import AwsIngestError


@pytest.fixture
def soil_benchmark(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_pdsi_soils.py"
    spec = importlib.util.spec_from_file_location("benchmark_pdsi_soils_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path):
    return argparse.Namespace(
        precip="precip.nc",
        pet="pet.nc",
        precip_var=None,
        pet_var=None,
        region=(-100.0, -99.0, 38.0, 39.0),
        calibration=(1981, 2010),
        period=("1951-01-01", "2020-12-01"),
        max_cells=4,
        out=str(tmp_path),
        raw_dir=None,
        cache_dir=None,
        phase=None,
        sources="gridmet,usgs",
        depths="native",
    )


def test_children_parse_negative_region_and_keep_requested_period(soil_benchmark, monkeypatch, tmp_path):
    args = _args(tmp_path)
    command = soil_benchmark._child_command(args, "pdsi", "usgs", "native")
    monkeypatch.setattr(sys, "argv", command[1:])
    parsed = soil_benchmark._parse_args()
    assert parsed.region == args.region
    assert parsed.period == args.period
    assert parsed.calibration == args.calibration


@pytest.mark.parametrize(
    "dates",
    [[], ["2000-02-01", "2000-03-01"], ["2000-01-01", "2000-03-01"], ["2000-01-01", "2000-01-15"]],
)
def test_climate_selection_requires_january_first_contiguous_months(soil_benchmark, tmp_path, dates):
    field = xr.DataArray(
        np.ones((len(dates), 2, 2)),
        dims=["time", "lat", "lon"],
        coords={"time": pd.to_datetime(dates), "lat": [38, 39], "lon": [-100, -99]},
        attrs={"units": "mm"},
        name="precip",
    )
    path = tmp_path / "climate.nc"
    field.to_netcdf(path)
    with pytest.raises(SystemExit, match="contiguous monthly record beginning in January"):
        soil_benchmark._open_climate(str(path), "precip")


def test_monthly_validation_applies_after_period_selection(soil_benchmark, tmp_path):
    field = xr.DataArray(
        np.ones((12, 2, 2)),
        dims=["time", "lat", "lon"],
        coords={"time": pd.date_range("2000-01-01", periods=12, freq="MS"), "lat": [38, 39], "lon": [-100, -99]},
        attrs={"units": "mm"},
        name="precip",
    )
    path = tmp_path / "climate.nc"
    field.to_netcdf(path)
    assert soil_benchmark._open_climate(str(path), "precip").sizes["time"] == 12
    for period in [("2000-02-01", "2000-12-01"), ("2001-01-01", "2001-12-01")]:
        with pytest.raises(SystemExit, match="contiguous monthly"):
            soil_benchmark._open_climate(str(path), "precip", period=period)


@pytest.mark.parametrize("dimension", ["time", "lat", "lon", "aws"])
def test_scpdsi_rejects_equal_shapes_with_different_coordinates(soil_benchmark, dimension):
    precip = xr.DataArray(
        np.ones((12, 2, 2)),
        dims=["time", "lat", "lon"],
        coords={"time": pd.date_range("2000-01-01", periods=12, freq="MS"), "lat": [38, 39], "lon": [-100, -99]},
    )
    aws = precip.isel(time=0, drop=True) * 150
    pet = precip.copy()
    if dimension == "aws":
        aws = aws.assign_coords(lon=[-99, -100])
    else:
        pet = pet.assign_coords({dimension: pet[dimension].values[::-1]})
    with pytest.raises(SystemExit, match="exact time and spatial coordinates"):
        soil_benchmark._scpdsi_field(
            precip, pet, aws, data_start_year=2000, calibration_period=(2000, 2000), max_cells=4
        )


def test_failed_pdsi_cannot_admit_stale_pairwise_output(soil_benchmark, tmp_path, monkeypatch):
    args = _args(tmp_path)
    stale = soil_benchmark._scpdsi_cache_path(args, "gridmet", "native")
    stale.parent.mkdir()
    stale.write_text("stale field")
    field = xr.DataArray(np.ones((12, 2, 2)), dims=["time", "lat", "lon"], name="scpdsi")

    def run_child(args, phase, source, depth):
        if phase == "pdsi":
            if source == "gridmet":
                assert not stale.exists()
                return soil_benchmark.PhaseResult(source, depth, phase, note="phase failed (exit 1)")
            soil_benchmark._save_scpdsi(args, source, depth, field)
        return soil_benchmark.PhaseResult(source, depth, phase)

    monkeypatch.setattr(soil_benchmark, "_parse_args", lambda: args)
    monkeypatch.setattr(soil_benchmark, "_run_child", run_child)
    available = []
    pairwise = soil_benchmark._pairwise_rows

    def compare(args, sources):
        available.extend(sources)
        return pairwise(args, sources)

    monkeypatch.setattr(soil_benchmark, "_pairwise_rows", compare)
    assert soil_benchmark.main() == 0
    assert available == [("usgs", "native")]
    table = pd.read_csv(tmp_path / "aws_sources.csv")
    assert "ingest+scPDSI+output" in set(table["kind"])


def test_scpdsi_output_labels_cannot_traverse_directories(soil_benchmark, tmp_path):
    args = _args(tmp_path)
    with pytest.raises(AwsIngestError):
        soil_benchmark._scpdsi_cache_path(args, "../usgs", "native")
    with pytest.raises(ValueError):
        soil_benchmark._scpdsi_cache_path(args, "usgs", "../../outside")

"""Unit tests for the CLI's single output module and shared unit normalization."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from climate_indices import _cli_output
from climate_indices._units import _convert_precipitation_units
from climate_indices.exceptions import InvalidArgumentError


def _precip(units: str) -> xr.DataArray:
    return xr.DataArray(np.array([30.0, 60.0]), dims=("time",), attrs={"units": units})


@pytest.mark.parametrize("units", ["mm", "millimeters", "mm/month", "mm month-1"])
def test_monthly_depth_units_are_accepted(units: str) -> None:
    result = _convert_precipitation_units(_precip(units), "mm", monthly=True)
    np.testing.assert_array_equal(result.values, [30.0, 60.0])


@pytest.mark.parametrize("units", ["mm/day", "mm day-1", "kg m-2 s-1"])
def test_a_rate_is_rejected_for_a_monthly_depth(units: str) -> None:
    with pytest.raises(InvalidArgumentError, match="per-day rate"):
        _convert_precipitation_units(_precip(units), "mm", monthly=True)


@pytest.mark.parametrize("units", ["mm/month", "mm month-1"])
def test_a_monthly_depth_is_rejected_for_a_daily_series(units: str) -> None:
    with pytest.raises(InvalidArgumentError, match="Unsupported precipitation units"):
        _convert_precipitation_units(_precip(units), "mm")


def test_atomic_write_cleans_up_the_temporary_file_on_failure(monkeypatch, tmp_path) -> None:
    target = tmp_path / "out.nc"
    target.write_text("previous")

    def fail(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(xr.DataArray, "to_netcdf", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        _cli_output.write_netcdf_atomic(_precip("mm"), str(target))

    assert not (tmp_path / "out.nc.tmp").exists()
    assert target.read_text() == "previous"

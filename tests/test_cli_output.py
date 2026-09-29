"""Unit tests for the CLI's single output module and shared unit normalization."""

from __future__ import annotations

from pathlib import Path

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
    data = _precip(units)
    with pytest.raises(InvalidArgumentError, match="per-day rate"):
        _convert_precipitation_units(data, "mm", monthly=True)


@pytest.mark.parametrize("units", ["mm/month", "mm month-1"])
def test_a_monthly_depth_is_rejected_for_a_daily_series(units: str) -> None:
    data = _precip(units)
    with pytest.raises(InvalidArgumentError, match="Unsupported precipitation units"):
        _convert_precipitation_units(data, "mm")


def test_atomic_write_cleans_up_the_temporary_file_on_failure(monkeypatch, tmp_path) -> None:
    target = tmp_path / "out.nc"
    target.write_text("previous")

    def fail(self, path, **kwargs):
        Path(path).write_text("partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(xr.DataArray, "to_netcdf", fail)
    data = _precip("mm")
    with pytest.raises(RuntimeError, match="write failed"):
        _cli_output.write_netcdf_atomic(data, str(target))

    assert not (tmp_path / "out.nc.tmp").exists()
    assert target.read_text() == "previous"


def test_build_index_attrs_drops_the_inputs_own_valid_range() -> None:
    source = xr.DataArray(
        np.array([1.0]),
        dims=("time",),
        attrs={
            "units": "mm",
            "valid_min": 0.0,
            "valid_max": 500.0,
            "valid_range": [0.0, 500.0],
            "actual_range": [0.0, 10.0],
        },
    )
    attrs = _cli_output.build_index_attrs(
        source, "spi", index_name="SPI", extra={"valid_min": -3.09, "valid_max": 3.09}
    )
    assert attrs["valid_min"] == -3.09
    assert attrs["valid_max"] == 3.09
    assert "valid_range" not in attrs
    assert "actual_range" not in attrs

"""Alignment and Output Provenance contract for every stateless xarray entry point (#1218).

Every entry point that bypasses ``@xarray_adapter`` (PET, PCI, Palmer) must still obey
the policies the decorated indices share: no input ``standard_name`` on the output (CF
Standard Name Omission), history appended rather than replaced, and no silent loss of
cells when multi-input coordinates disagree. Palmer's own cell-mismatch rejection is
pinned in ``test_spatial_kernel.py``; it joins here for the provenance policies only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import climate_indices as ci
from climate_indices.exceptions import CoordinateValidationError, InputAlignmentWarning

LATS = [10.0, 20.0, 30.0]
_DAILY_STEPS = 365 * 2


def _daily(lats: list[float], base: float, attrs: dict[str, Any], steps: int = _DAILY_STEPS) -> xr.DataArray:
    time = pd.date_range("1990-01-01", periods=steps, freq="D")
    values = base + np.random.default_rng(1).random((steps, len(lats))) * 5
    return xr.DataArray(values, dims=("time", "lat"), coords={"time": time, "lat": lats}, attrs=attrs)


def _daily_series(steps: int, value: float) -> xr.DataArray:
    time = pd.date_range("1990-01-01", periods=steps, freq="D")
    return xr.DataArray(np.full(steps, value), dims=("time",), coords={"time": time})


def _thornthwaite(attrs: dict[str, Any]) -> xr.DataArray:
    time = pd.date_range("1990-01-01", periods=24, freq="MS")
    values = 10 + np.random.default_rng(1).random((24, len(LATS))) * 10
    temperature = xr.DataArray(values, dims=("time", "lat"), coords={"time": time, "lat": LATS}, attrs=attrs)
    return ci.pet_thornthwaite(temperature, 35.0, 1990)


def _hargreaves(
    attrs: dict[str, Any],
    tmax_lats: list[float] = LATS,
    tmin_steps: int = _DAILY_STEPS,
    tmax_steps: int = _DAILY_STEPS,
) -> xr.DataArray:
    return ci.pet_hargreaves(_daily(LATS, 10, attrs, tmin_steps), _daily(tmax_lats, 25, {}, tmax_steps), 35.0)


def _penman_monteith(
    attrs: dict[str, Any],
    tmax_lats: list[float] = LATS,
    tmin_steps: int = _DAILY_STEPS,
    tmax_steps: int = _DAILY_STEPS,
    wind_speed_m_s: float | xr.DataArray = 2.0,
) -> xr.DataArray:
    return ci.pet_penman_monteith(
        _daily(LATS, 10, attrs, tmin_steps),
        _daily(tmax_lats, 25, {}, tmax_steps),
        latitude=35.0,
        elevation_m=100.0,
        wind_speed_m_s=wind_speed_m_s,
    )


def _pdsi(attrs: dict[str, Any]) -> xr.DataArray:
    months = 12 * 25
    time = pd.date_range("1980-01-01", periods=months, freq="MS")
    rng = np.random.default_rng(1)

    def monthly(base: float, array_attrs: dict[str, Any]) -> xr.DataArray:
        values = base + rng.random((months, len(LATS))) * 50
        return xr.DataArray(values, dims=("time", "lat"), coords={"time": time, "lat": LATS}, attrs=array_attrs)

    result = ci.pdsi(
        monthly(20, attrs),
        monthly(10, {}),
        5.0,
        data_start_year=1980,
        calibration_year_initial=1981,
        calibration_year_final=2000,
    )
    return result["pdsi"]


def _pci(attrs: dict[str, Any]) -> xr.DataArray:
    time = pd.date_range("1990-01-01", periods=365, freq="D")
    rainfall = xr.DataArray(
        np.random.default_rng(1).random(365) * 10, dims=("time",), coords={"time": time}, attrs=attrs
    )
    return ci.pci(rainfall)


STATELESS: dict[str, Callable[..., xr.DataArray]] = {
    "pet_thornthwaite": _thornthwaite,
    "pet_hargreaves": _hargreaves,
    "pet_penman_monteith": _penman_monteith,
    "pci": _pci,
    "pdsi": _pdsi,
}
TWO_INPUT: dict[str, Callable[..., xr.DataArray]] = {
    "pet_hargreaves": _hargreaves,
    "pet_penman_monteith": _penman_monteith,
}


@pytest.mark.parametrize("call", STATELESS.values(), ids=STATELESS.keys())
def test_output_does_not_inherit_input_standard_name(call: Callable[..., xr.DataArray]) -> None:
    result = call({"standard_name": "air_temperature", "units": "degC"})

    assert "standard_name" not in result.attrs


@pytest.mark.parametrize("call", STATELESS.values(), ids=STATELESS.keys())
def test_history_is_appended_not_replaced(call: Callable[..., xr.DataArray]) -> None:
    result = call({"history": "2026-01-01T00:00:00Z: prior step"})

    prior, _, latest = result.attrs["history"].partition("\n")
    assert prior == "2026-01-01T00:00:00Z: prior step"
    assert "climate_indices v" in latest


@pytest.mark.parametrize("call", TWO_INPUT.values(), ids=TWO_INPUT.keys())
def test_mismatched_cell_coordinates_are_rejected_not_intersected(call: Callable[..., xr.DataArray]) -> None:
    with pytest.raises(CoordinateValidationError) as excinfo:
        call({}, tmax_lats=[20.0, 30.0, 40.0])

    assert excinfo.value.reason == "mismatched_non_time_coordinates"
    assert excinfo.value.coordinate_name == "lat"


@pytest.mark.parametrize("call", TWO_INPUT.values(), ids=TWO_INPUT.keys())
@pytest.mark.parametrize("shorter", ["tmin", "tmax"])
def test_partial_time_overlap_is_trimmed_with_a_warning(call: Callable[..., xr.DataArray], shorter: str) -> None:
    steps = {"tmin_steps": _DAILY_STEPS, "tmax_steps": _DAILY_STEPS, f"{shorter}_steps": _DAILY_STEPS - 30}

    with pytest.warns(InputAlignmentWarning):
        result = call({}, **steps)

    assert result.sizes["time"] == _DAILY_STEPS - 30
    assert result.sizes["lat"] == len(LATS)


def test_penman_monteith_time_series_input_is_trimmed_with_tmin_and_tmax() -> None:
    with pytest.warns(InputAlignmentWarning):
        result = _penman_monteith({}, wind_speed_m_s=_daily_series(_DAILY_STEPS - 30, 2.0))

    assert result.sizes["time"] == _DAILY_STEPS - 30


def test_penman_monteith_gridded_input_with_mismatched_cells_is_rejected() -> None:
    with pytest.raises(CoordinateValidationError) as excinfo:
        _penman_monteith({}, wind_speed_m_s=_daily([20.0, 30.0, 40.0], 2.0, {}))

    assert excinfo.value.reason == "mismatched_non_time_coordinates"


def test_reordered_cell_coordinates_are_paired_by_label() -> None:
    tmin, tmax = _daily(LATS, 10, {}), _daily(LATS, 25, {})

    expected = ci.pet_hargreaves(tmin, tmax, 35.0)
    actual = ci.pet_hargreaves(tmin, tmax.isel(lat=slice(None, None, -1)), 35.0)

    xr.testing.assert_allclose(actual.sel(lat=LATS), expected.sel(lat=LATS))


@pytest.mark.parametrize("call", STATELESS.values(), ids=STATELESS.keys())
def test_other_input_attrs_are_kept(call: Callable[..., xr.DataArray]) -> None:
    result = call({"source": "gauge network"})

    assert result.attrs["source"] == "gauge network"


@pytest.mark.parametrize("name", ["pet_thornthwaite", "pet_hargreaves", "pet_penman_monteith"])
def test_pet_records_its_latitude(name: str) -> None:
    assert STATELESS[name]({}).attrs["latitude"] == 35.0


def test_thornthwaite_records_its_data_start_year() -> None:
    assert STATELESS["pet_thornthwaite"]({}).attrs["data_start_year"] == 1990

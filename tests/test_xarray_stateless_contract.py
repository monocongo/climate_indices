"""Alignment and Output Provenance contract for every stateless xarray entry point (#1218).

Every entry point that bypasses ``@xarray_adapter`` (PET, PCI) must still obey the
policies the decorated indices share: no input ``standard_name`` on the output (CF
Standard Name Omission), history appended rather than replaced, and no silent loss of
cells when multi-input coordinates disagree.
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


def _thornthwaite(attrs: dict[str, Any]) -> xr.DataArray:
    time = pd.date_range("1990-01-01", periods=24, freq="MS")
    values = 10 + np.random.default_rng(1).random((24, len(LATS))) * 10
    temperature = xr.DataArray(values, dims=("time", "lat"), coords={"time": time, "lat": LATS}, attrs=attrs)
    return ci.pet_thornthwaite(temperature, 35.0, 1990)


def _hargreaves(attrs: dict[str, Any], tmax_lats: list[float] = LATS, tmax_steps: int = _DAILY_STEPS) -> xr.DataArray:
    return ci.pet_hargreaves(_daily(LATS, 10, attrs), _daily(tmax_lats, 25, {}, tmax_steps), 35.0)


def _penman_monteith(
    attrs: dict[str, Any], tmax_lats: list[float] = LATS, tmax_steps: int = _DAILY_STEPS
) -> xr.DataArray:
    return ci.pet_penman_monteith(
        _daily(LATS, 10, attrs),
        _daily(tmax_lats, 25, {}, tmax_steps),
        latitude=35.0,
        elevation_m=100.0,
        wind_speed_m_s=2.0,
    )


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
    with pytest.raises(CoordinateValidationError, match="lat"):
        call({}, tmax_lats=[20.0, 30.0, 40.0])


@pytest.mark.parametrize("call", TWO_INPUT.values(), ids=TWO_INPUT.keys())
def test_partial_time_overlap_is_trimmed_with_a_warning(call: Callable[..., xr.DataArray]) -> None:
    with pytest.warns(InputAlignmentWarning):
        result = call({}, tmax_steps=_DAILY_STEPS - 30)

    assert result.sizes["time"] == _DAILY_STEPS - 30
    assert result.sizes["lat"] == len(LATS)

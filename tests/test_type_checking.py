"""Static type verification tests for mypy.

These tests use typing.assert_type() to verify that mypy correctly infers
return types for the overloaded public API functions. assert_type() is a
runtime no-op, so only mypy enforces them: the `lint` CI job runs
`mypy src/ tests/test_type_checking.py`. The `meta` job also runs this module
under pytest, which executes the call bodies but verifies no types.
"""

from __future__ import annotations

import sys

import numpy as np
import numpy.typing as npt
import pandas as pd

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type
import xarray as xr

from climate_indices import fire, spei, spi
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution


def test_spi_numpy_return_type() -> None:
    """Verify mypy infers np.ndarray for NumPy input."""
    # 40 years * 12 months = 480 values
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=480)
    result = spi(
        values=values,
        scale=6,
        distribution=Distribution.gamma,
        data_start_year=1980,
        calibration_year_initial=1980,
        calibration_year_final=2019,
        periodicity=Periodicity.monthly,
    )
    assert_type(result, npt.NDArray[np.float64])


def test_spi_xarray_return_type() -> None:
    """Verify mypy infers xr.DataArray for xarray input."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = xr.DataArray(
        rng.gamma(shape=2.0, scale=50.0, size=len(time)),
        coords={"time": time},
        dims=["time"],
    )
    result = spi(
        values=values,
        scale=6,
        distribution=Distribution.gamma,
    )
    assert_type(result, xr.DataArray)


def test_spei_numpy_return_type() -> None:
    """Verify mypy infers np.ndarray for NumPy input."""
    # 40 years * 12 months = 480 values
    rng = np.random.default_rng(42)
    precips = rng.gamma(shape=2.0, scale=50.0, size=480)
    pet = rng.gamma(shape=2.0, scale=30.0, size=480)
    result = spei(
        precips_mm=precips,
        pet_mm=pet,
        scale=6,
        distribution=Distribution.gamma,
        periodicity=Periodicity.monthly,
        data_start_year=1980,
        calibration_year_initial=1980,
        calibration_year_final=2019,
    )
    assert_type(result, npt.NDArray[np.float64])


def test_spei_xarray_return_type() -> None:
    """Verify mypy infers xr.DataArray for xarray input."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    precips = xr.DataArray(
        rng.gamma(shape=2.0, scale=50.0, size=len(time)),
        coords={"time": time},
        dims=["time"],
    )
    pet = xr.DataArray(
        rng.gamma(shape=2.0, scale=30.0, size=len(time)),
        coords={"time": time},
        dims=["time"],
    )
    result = spei(
        precips_mm=precips,
        pet_mm=pet,
        scale=6,
        distribution=Distribution.gamma,
    )
    assert_type(result, xr.DataArray)


def test_flood_xarray_return_types() -> None:
    """DataArray must match the xarray overload before the broad ArrayLike overload."""
    from climate_indices import flood

    dates = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    pe = xr.DataArray(np.arange(len(dates), dtype=float), dims=["time"], coords={"time": dates})
    assert_type(flood.effective_precipitation(pe), xr.DataArray)
    assert_type(flood.edi(pe), xr.DataArray)
    assert_type(flood.flood_index(pe, year_start_month=1), xr.DataArray)


def test_kbdi_numpy_return_type() -> None:
    """Verify mypy infers np.ndarray for NumPy input.

    Regression test: the xr.DataArray overload must be declared before the
    npt.ArrayLike overload. xr.DataArray implements __array__, so it
    satisfies npt.ArrayLike structurally -- if the ArrayLike overload came
    first, mypy would match it for DataArray calls too, and the DataArray
    overload below would be unreachable.
    """
    rng = np.random.default_rng(42)
    precipitation = rng.gamma(shape=2.0, scale=3.0, size=100)
    temperature = rng.uniform(-5.0, 35.0, size=100)
    # kbdi() isn't overloaded on return_state (unlike spi/spei above), so its
    # return type is always a union with KBDIResult, even when return_state
    # defaults to False.
    result = fire.kbdi(precipitation, temperature, 1000.0)
    assert_type(result, npt.NDArray[np.float64] | fire.KBDIResult)


def test_kbdi_xarray_return_type() -> None:
    """Verify mypy infers xr.DataArray for xarray input (see test_kbdi_numpy_return_type)."""
    time = pd.date_range("2000-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    precipitation = xr.DataArray(rng.gamma(shape=2.0, scale=3.0, size=100), coords={"time": time}, dims=["time"])
    temperature = xr.DataArray(rng.uniform(-5.0, 35.0, size=100), coords={"time": time}, dims=["time"])
    result = fire.kbdi(precipitation, temperature, 1000.0)
    assert_type(result, xr.DataArray | fire.KBDIResult)


def test_hdw_numpy_return_type() -> None:
    """Verify mypy infers np.ndarray for NumPy input (see test_kbdi_numpy_return_type)."""
    result = fire.hot_dry_windy([30.0, 26.0], [15.0, 30.0], [8.0, 12.0], [10.0, 400.0])
    assert_type(result, npt.NDArray[np.float64])


def test_hdw_xarray_return_type() -> None:
    """Verify mypy infers xr.DataArray for xarray input (see test_kbdi_numpy_return_type)."""
    dims = ["level"]
    temperature = xr.DataArray([30.0, 26.0], dims=dims)
    humidity = xr.DataArray([15.0, 30.0], dims=dims)
    wind = xr.DataArray([8.0, 12.0], dims=dims)
    height = xr.DataArray([10.0, 400.0], dims=dims)
    result = fire.hot_dry_windy(temperature, humidity, wind, height)
    assert_type(result, xr.DataArray)


def test_haines_index_numpy_return_type() -> None:
    """Verify mypy infers np.ndarray for NumPy input (see test_kbdi_numpy_return_type)."""
    result = fire.haines_index([32.0, 20.0], [24.0, 5.0], [13.0, 0.0], variant="low")
    assert_type(result, npt.NDArray[np.float64])


def test_haines_index_xarray_return_type() -> None:
    """Verify mypy infers xr.DataArray for xarray input (see test_kbdi_numpy_return_type)."""
    dims = ["level"]
    temperature_lower = xr.DataArray([32.0, 20.0], dims=dims)
    temperature_upper = xr.DataArray([24.0, 5.0], dims=dims)
    dewpoint = xr.DataArray([13.0, 0.0], dims=dims)
    result = fire.haines_index(temperature_lower, temperature_upper, dewpoint, variant="low")
    assert_type(result, xr.DataArray)


def test_haines_index_from_profile_numpy_return_type() -> None:
    """The profile entry point is NumPy-only and always returns an ndarray."""
    result = fire.haines_index_from_profile(
        [32.0, 24.0, 12.0, -8.0], [26.0, 13.0, 2.0, -20.0], [950.0, 850.0, 700.0, 500.0], 100.0
    )
    assert_type(result, npt.NDArray[np.float64])

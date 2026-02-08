"""Static type verification tests for mypy.

These tests use typing.assert_type() to verify that mypy correctly infers
return types for the overloaded spi() and spei() functions. The tests are
checked by mypy, not by pytest.

Run with: uv run mypy tests/test_type_checking.py

Note: These tests are designed to be checked by mypy for type inference.
They include proper test data so they can also run successfully in pytest.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type
import xarray as xr

from climate_indices import spei, spi
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
    assert_type(result, np.ndarray)


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
    assert_type(result, np.ndarray)


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

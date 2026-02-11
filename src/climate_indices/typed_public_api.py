"""Typed public API for SPI and SPEI with NumPy/xarray overloads.

This module provides statically-typed wrappers around the xarray-adapted index
functions. The @overload signatures enable IDE autocomplete and mypy --strict
correctness by narrowing return types based on input types:

- spi(np.ndarray, ...) -> np.ndarray
- spi(xr.DataArray, ...) -> xr.DataArray

Design: Pre-build decorated functions at module level for performance. The public
functions filter None kwargs and delegate to the pre-built wrapped functions.

.. warning:: **Beta Feature (xarray path)** — The xarray DataArray overloads in
   this module are beta. The NumPy overloads are stable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices import indices
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution
from climate_indices.xarray_adapter import CF_METADATA, xarray_adapter

if TYPE_CHECKING:
    pass

# pre-build decorated functions at module level for performance
_wrapped_spi = xarray_adapter(
    cf_metadata=CF_METADATA["spi"],  # type: ignore[arg-type]
    index_display_name="SPI",
    calculation_metadata_keys=["scale", "distribution", "calibration_year_initial", "calibration_year_final"],
)(indices.spi)

_wrapped_spei = xarray_adapter(
    cf_metadata=CF_METADATA["spei"],  # type: ignore[arg-type]
    index_display_name="SPEI",
    calculation_metadata_keys=["scale", "distribution", "calibration_year_initial", "calibration_year_final"],
    additional_input_names=["pet_mm"],
)(indices.spei)


# SPI overloads
@overload
def spi(
    values: npt.NDArray[np.float64],
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: Periodicity,
    fitting_params: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64]: ...


@overload
def spi(
    values: xr.DataArray,
    scale: int,
    distribution: Distribution,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: Periodicity | None = None,
    fitting_params: dict[str, Any] | None = None,
) -> xr.DataArray: ...


def spi(
    values: npt.NDArray[np.float64] | xr.DataArray,
    scale: int,
    distribution: Distribution,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: Periodicity | None = None,
    fitting_params: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute SPI (Standardized Precipitation Index).

    This function accepts both NumPy arrays and xarray DataArrays. Type checkers
    will narrow the return type based on the input type.

    For NumPy inputs, all temporal parameters are required.
    For xarray inputs, temporal parameters are optional and will be inferred from
    coordinate attributes if not provided.

    .. warning:: **Beta Feature (xarray path only)** — When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface (parameter inference, metadata handling, coordinate
       preservation) may change in future minor releases. The NumPy array interface
       is stable.

    Args:
        values: 1-D numpy array or xarray DataArray of precipitation values.
        scale: Number of time steps over which values should be scaled.
        distribution: Distribution type for fitting/transform computation.
        data_start_year: Initial year of the input dataset (required for NumPy,
            optional for xarray).
        calibration_year_initial: Initial year of calibration period (required
            for NumPy, optional for xarray).
        calibration_year_final: Final year of calibration period (required for
            NumPy, optional for xarray).
        periodicity: Time series periodicity ('monthly' or 'daily'). Required
            for NumPy, optional for xarray.
        fitting_params: Optional dict of pre-computed distribution fitting
            parameters.

    Returns:
        SPI values as numpy.ndarray or xarray.DataArray (matches input type).
    """
    # filter out None kwargs before passing to wrapped function
    kwargs = {
        "scale": scale,
        "distribution": distribution,
        "fitting_params": fitting_params,
    }
    if data_start_year is not None:
        kwargs["data_start_year"] = data_start_year
    if calibration_year_initial is not None:
        kwargs["calibration_year_initial"] = calibration_year_initial
    if calibration_year_final is not None:
        kwargs["calibration_year_final"] = calibration_year_final
    if periodicity is not None:
        kwargs["periodicity"] = periodicity

    return _wrapped_spi(values, **kwargs)


# SPEI overloads
@overload
def spei(
    precips_mm: npt.NDArray[np.float64],
    pet_mm: npt.NDArray[np.float64],
    scale: int,
    distribution: Distribution,
    periodicity: Periodicity,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64]: ...


@overload
def spei(
    precips_mm: xr.DataArray,
    pet_mm: xr.DataArray,
    scale: int,
    distribution: Distribution,
    periodicity: Periodicity | None = None,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    fitting_params: dict[str, Any] | None = None,
) -> xr.DataArray: ...


def spei(
    precips_mm: npt.NDArray[np.float64] | xr.DataArray,
    pet_mm: npt.NDArray[np.float64] | xr.DataArray,
    scale: int,
    distribution: Distribution,
    periodicity: Periodicity | None = None,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    fitting_params: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute SPEI (Standardized Precipitation Evapotranspiration Index).

    This function accepts both NumPy arrays and xarray DataArrays. Type checkers
    will narrow the return type based on the input type.

    For NumPy inputs, all temporal parameters are required.
    For xarray inputs, temporal parameters are optional and will be inferred from
    coordinate attributes if not provided.

    .. warning:: **Beta Feature (xarray path only)** — When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface (parameter inference, metadata handling, coordinate
       preservation) may change in future minor releases. The NumPy array interface
       is stable.

    Args:
        precips_mm: Array of monthly precipitation values in millimeters.
        pet_mm: Array of monthly PET values in millimeters.
        scale: Number of time steps over which values should be scaled.
        distribution: Distribution type for fitting/transform computation.
        periodicity: Time series periodicity ('monthly' or 'daily'). Required
            for NumPy, optional for xarray.
        data_start_year: Initial year of the input dataset (required for NumPy,
            optional for xarray).
        calibration_year_initial: Initial year of calibration period (required
            for NumPy, optional for xarray).
        calibration_year_final: Final year of calibration period (required for
            NumPy, optional for xarray).
        fitting_params: Optional dict of pre-computed distribution fitting
            parameters.

    Returns:
        SPEI values as numpy.ndarray or xarray.DataArray (matches input type).
    """
    # filter out None kwargs before passing to wrapped function
    kwargs = {
        "scale": scale,
        "distribution": distribution,
        "fitting_params": fitting_params,
    }
    if periodicity is not None:
        kwargs["periodicity"] = periodicity
    if data_start_year is not None:
        kwargs["data_start_year"] = data_start_year
    if calibration_year_initial is not None:
        kwargs["calibration_year_initial"] = calibration_year_initial
    if calibration_year_final is not None:
        kwargs["calibration_year_final"] = calibration_year_final

    return _wrapped_spei(precips_mm, pet_mm, **kwargs)

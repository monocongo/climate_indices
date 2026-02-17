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
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution
from climate_indices.palmer import palmer_xarray
from climate_indices.palmer import pdsi as _numpy_pdsi
from climate_indices.xarray_adapter import eto_penman_monteith as _eto_pm
from climate_indices.xarray_adapter import xarray_adapter

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


# Penman-Monteith ETo overloads
@overload
def eto_penman_monteith(
    net_radiation: npt.NDArray[np.float64],
    soil_heat_flux: npt.NDArray[np.float64],
    temperature_celsius: npt.NDArray[np.float64],
    wind_speed_2m: npt.NDArray[np.float64],
    saturation_vp: npt.NDArray[np.float64],
    actual_vp: npt.NDArray[np.float64],
    delta: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...


@overload
def eto_penman_monteith(
    net_radiation: xr.DataArray,
    soil_heat_flux: xr.DataArray,
    temperature_celsius: xr.DataArray,
    wind_speed_2m: xr.DataArray,
    saturation_vp: xr.DataArray,
    actual_vp: xr.DataArray,
    delta: xr.DataArray,
    gamma: xr.DataArray,
) -> xr.DataArray: ...


def eto_penman_monteith(
    net_radiation: npt.NDArray[np.float64] | xr.DataArray,
    soil_heat_flux: npt.NDArray[np.float64] | xr.DataArray,
    temperature_celsius: npt.NDArray[np.float64] | xr.DataArray,
    wind_speed_2m: npt.NDArray[np.float64] | xr.DataArray,
    saturation_vp: npt.NDArray[np.float64] | xr.DataArray,
    actual_vp: npt.NDArray[np.float64] | xr.DataArray,
    delta: npt.NDArray[np.float64] | xr.DataArray,
    gamma: npt.NDArray[np.float64] | xr.DataArray,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute Penman-Monteith FAO-56 reference evapotranspiration (ETo).

    This function accepts both NumPy arrays and xarray DataArrays. Type
    checkers will narrow the return type based on the input type.

    .. warning:: **Beta Feature (xarray path only)** -- When called with
       ``xr.DataArray`` inputs, this function uses the beta xarray adapter
       layer. The NumPy array interface is stable.

    Args:
        net_radiation: Net radiation at crop surface [MJ m-2 day-1].
        soil_heat_flux: Soil heat flux density [MJ m-2 day-1].
        temperature_celsius: Mean daily air temperature at 2 m [degC].
        wind_speed_2m: Wind speed at 2 m height [m s-1].
        saturation_vp: Saturation vapor pressure [kPa].
        actual_vp: Actual vapor pressure [kPa].
        delta: Slope of saturation vapor pressure curve [kPa degC-1].
        gamma: Psychrometric constant [kPa degC-1].

    Returns:
        ETo values in mm/day as numpy.ndarray or xarray.DataArray
        (matches input type).
    """
    return _eto_pm(
        net_radiation=net_radiation,
        soil_heat_flux=soil_heat_flux,
        temperature_celsius=temperature_celsius,
        wind_speed_2m=wind_speed_2m,
        saturation_vp=saturation_vp,
        actual_vp=actual_vp,
        delta=delta,
        gamma=gamma,
    )


# Palmer PDSI overloads
@overload
def pdsi(
    precips: npt.NDArray[np.float64],
    pet: npt.NDArray[np.float64],
    awc: float,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[str, Any] | None,
]: ...


@overload
def pdsi(
    precips: xr.DataArray,
    pet: xr.DataArray,
    awc: float | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    fitting_params: dict[str, Any] | None = None,
) -> xr.Dataset: ...


def pdsi(
    precips: npt.NDArray[np.float64] | xr.DataArray,
    pet: npt.NDArray[np.float64] | xr.DataArray,
    awc: float | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    fitting_params: dict[str, Any] | None = None,
) -> (
    tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        dict[str, Any] | None,
    ]
    | xr.Dataset
):
    """Compute Palmer Drought Severity Index and related indices.

    This function accepts both NumPy arrays and xarray DataArrays. Type
    checkers will narrow the return type based on the input type:

    - NumPy inputs: returns tuple of (pdsi, phdi, pmdi, z_index, params_dict)
    - xarray inputs: returns xr.Dataset with variables pdsi, phdi, pmdi, z_index

    For NumPy inputs, all temporal parameters are required.
    For xarray inputs, temporal parameters are optional and will be inferred
    from coordinate attributes if not provided.

    .. warning:: **Beta Feature (xarray path only)** -- When called with
       ``xr.DataArray`` inputs, this function uses the beta xarray adapter
       layer. The NumPy array interface is stable.

    Args:
        precips: Monthly precipitation values in inches.
        pet: Monthly potential evapotranspiration values in inches.
        awc: Available water capacity in inches. Scalar float for uniform
            AWC, or xr.DataArray with spatial dims (xarray path only).
        data_start_year: Initial year of the input dataset (required for
            NumPy, optional for xarray).
        calibration_year_initial: Initial year of calibration period
            (required for NumPy, optional for xarray).
        calibration_year_final: Final year of calibration period (required
            for NumPy, optional for xarray).
        fitting_params: Optional dict of pre-computed fitting parameters
            (alpha, beta, gamma, delta).

    Returns:
        NumPy path: tuple of (pdsi, phdi, pmdi, z_index, params_dict)
        xarray path: xr.Dataset with 4 data variables and provenance attrs.
    """
    if isinstance(precips, xr.DataArray):
        return palmer_xarray(
            precip_da=precips,
            pet_da=pet,
            awc=awc,
            data_start_year=data_start_year,
            calibration_year_initial=calibration_year_initial,
            calibration_year_final=calibration_year_final,
            fitting_params=fitting_params,
        )

    # NumPy path: all temporal params required
    if data_start_year is None or calibration_year_initial is None or calibration_year_final is None:
        msg = (
            "data_start_year, calibration_year_initial, and calibration_year_final are required for NumPy array inputs."
        )
        raise TypeError(msg)

    return _numpy_pdsi(
        precips=precips,
        pet=pet,
        awc=float(awc),
        data_start_year=data_start_year,
        calibration_year_initial=calibration_year_initial,
        calibration_year_final=calibration_year_final,
        fitting_params=fitting_params,
    )

"""Typed public API for climate indices with NumPy/xarray overloads.

This module provides statically-typed wrappers around the xarray-adapted index
functions. The @overload signatures enable IDE autocomplete and mypy --strict
correctness by narrowing return types based on input types:

- spi(np.ndarray, ...) -> np.ndarray      (and likewise for spei, eddi, percentage_of_normal, pci)
- spi(xr.DataArray, ...) -> xr.DataArray  (and pet_thornthwaite, pet_hargreaves)

The Palmer family follows the same dispatch: pdsi(np.ndarray, ...) returns the
five-item tuple palmer.pdsi() produces, and pdsi(xr.DataArray, ...) returns an
xr.Dataset of the four indices.

Design: Pre-build decorated functions at module level for performance. Every
public function except ``pci`` declares its signature once, as a pair of
@overload stubs (NumPy and xarray) that mirror the wrapped function; the
implementation takes ``*args/**kwargs`` and delegates through ``_delegate``, so
it carries no second signature to drift. ``_restore_runtime_signature`` pins each
public function's ``inspect`` signature (and so the Sphinx reference) to its
NumPy overload. ``tests/test_typed_public_api.py`` freezes both stubs and pins
their parameter names to the wrapped function.

PCI uses a manual wrapper instead of @xarray_adapter because its output shape
(scalar) differs from input shape (365/366 daily values).

.. warning:: **Beta Feature (xarray path)** — The xarray DataArray overloads in
   this module are beta. The NumPy overloads are stable.

Fire-weather APIs are intentionally namespaced under ``climate_indices.fire``.
Do not add unqualified fire wrappers or package-level re-exports here; each
beta xarray path stays on its corresponding ``fire.<name>`` public route.
"""

from __future__ import annotations

import datetime
import inspect
import typing
from collections.abc import Callable
from typing import Any, cast, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices import compute, indices, pm_eto
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.compute import Periodicity
from climate_indices.exceptions import emit_deprecation_warning
from climate_indices.indices import Distribution
from climate_indices.validation import InputType, detect_input_type
from climate_indices.xarray_adapter import (
    fit_diagnostics as _fit_diagnostics_impl,
)
from climate_indices.xarray_adapter import (
    palmer_pdsi as _palmer_pdsi_impl,
)
from climate_indices.xarray_adapter import (
    pet_hargreaves as _pet_hargreaves_impl,
)
from climate_indices.xarray_adapter import (
    pet_penman_monteith as _pet_penman_monteith_impl,
)
from climate_indices.xarray_adapter import (
    pet_thornthwaite as _pet_thornthwaite_impl,
)
from climate_indices.xarray_adapter import (
    xarray_adapter,
)

# pre-build decorated functions at module level for performance
_wrapped_spi = xarray_adapter(
    cf_metadata=CF_METADATA["spi"],  # type: ignore[arg-type]
    index_display_name="SPI",
    calculation_metadata_keys=["scale", "distribution", "calibration_year_initial", "calibration_year_final"],
    spatial_kernel=True,
)(indices.spi)

_wrapped_spei = xarray_adapter(
    cf_metadata=CF_METADATA["spei"],  # type: ignore[arg-type]
    index_display_name="SPEI",
    calculation_metadata_keys=["scale", "distribution", "calibration_year_initial", "calibration_year_final"],
    additional_input_names=["pet_mm"],
    spatial_kernel=True,
)(indices.spei)

_wrapped_eddi = xarray_adapter(
    cf_metadata=CF_METADATA["eddi"],  # type: ignore[arg-type]
    index_display_name="EDDI",
    calculation_metadata_keys=["scale", "calibration_year_initial", "calibration_year_final"],
    spatial_kernel=True,
)(indices.eddi)


_DelegateReturn = typing.TypeVar("_DelegateReturn")


def _delegate(
    func: Callable[..., _DelegateReturn],
    data: Any,
    *args: Any,
    **kwargs: Any,
) -> _DelegateReturn:
    """Call a pre-built API function with the data positional and the rest keyword.

    The xarray adapter requires the data as its first positional argument and reads
    the remaining parameters from keywords, so the call is bound and re-forwarded
    that way; binding also rejects unknown keyword arguments, as the explicit
    signatures did. Explicit None is dropped for parameters without a default:
    binding None would suppress the adapter's inference of that parameter, while
    defaulted parameters (e.g. ``fitting_params``) keep their None.
    """
    signature = inspect.signature(func)
    bound = signature.bind_partial(data, *args, **kwargs)
    data_parameter = next(iter(signature.parameters))
    forwarded = {
        name: value
        for name, value in bound.arguments.items()
        if name != data_parameter
        and not (value is None and signature.parameters[name].default is inspect.Parameter.empty)
    }
    return func(data, **forwarded)


def _restore_runtime_signature(func: Callable[..., Any]) -> None:
    """Give a generic implementation the NumPy overload stub's ``inspect`` signature.

    The implementation forwards ``*args/**kwargs``, so without this ``inspect`` and
    the Sphinx API reference would render that generic form instead of the public
    parameters. Unavailable before Python 3.11 (no ``typing.get_overloads``).
    """
    get_overloads = getattr(typing, "get_overloads", None)
    if get_overloads is None:
        return
    overloads = get_overloads(func)
    if overloads:
        cast(Any, func).__signature__ = inspect.signature(overloads[0])


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


def spi(values: Any, *args: Any, **kwargs: Any) -> npt.NDArray[np.float64] | xr.DataArray:
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
    return _delegate(_wrapped_spi, values, *args, **kwargs)


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


def spei(precips_mm: Any, pet_mm: Any, *args: Any, **kwargs: Any) -> npt.NDArray[np.float64] | xr.DataArray:
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
        precips_mm: Array of precipitation values in millimeters.
        pet_mm: Array of PET values in millimeters.
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
    return _delegate(_wrapped_spei, precips_mm, pet_mm, *args, **kwargs)


# Percentage of Normal (PNP) overloads
_wrapped_percentage_of_normal = xarray_adapter(
    cf_metadata=CF_METADATA["percentage_of_normal"],  # type: ignore[arg-type]
    index_display_name="PNP",
    calculation_metadata_keys=["scale", "calibration_year_initial", "calibration_year_final"],
    spatial_kernel=True,
)(indices.percentage_of_normal)


def _translate_pnp_calibration_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map the deprecated PNP ``calibration_start_year``/``_end_year`` aliases onto the canonical names."""
    translated = dict(kwargs)
    if "calibration_start_year" in translated or "calibration_end_year" in translated:
        emit_deprecation_warning(
            feature="Parameters 'calibration_start_year'/'calibration_end_year'",
            alternative="Use 'calibration_year_initial'/'calibration_year_final'",
            deprecated_in="3.1.0",
            removal_version="4.0.0",
        )
        for legacy, canonical in (
            ("calibration_start_year", "calibration_year_initial"),
            ("calibration_end_year", "calibration_year_final"),
        ):
            if legacy in translated and canonical not in translated:
                translated[canonical] = translated.pop(legacy)
            else:
                translated.pop(legacy, None)
    return translated


@overload
def percentage_of_normal(
    values: npt.NDArray[np.float64],
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: Periodicity,
) -> npt.NDArray[np.float64]: ...


@overload
def percentage_of_normal(
    values: npt.NDArray[np.float64],
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
) -> npt.NDArray[np.float64]: ...


@overload
def percentage_of_normal(
    values: npt.NDArray[np.float64],
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_end_year: int,
    periodicity: Periodicity,
) -> npt.NDArray[np.float64]: ...


@overload
def percentage_of_normal(
    values: npt.NDArray[np.float64],
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_year_final: int,
    periodicity: Periodicity,
) -> npt.NDArray[np.float64]: ...


@overload
def percentage_of_normal(
    values: xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: Periodicity | None = None,
) -> xr.DataArray: ...


@overload
def percentage_of_normal(
    values: xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_start_year: int | None = None,
    calibration_end_year: int | None = None,
    periodicity: Periodicity | None = None,
) -> xr.DataArray: ...


@overload
def percentage_of_normal(
    values: xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_end_year: int | None = None,
    periodicity: Periodicity | None = None,
) -> xr.DataArray: ...


@overload
def percentage_of_normal(
    values: xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_start_year: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: Periodicity | None = None,
) -> xr.DataArray: ...


def percentage_of_normal(values: Any, *args: Any, **kwargs: Any) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute Percentage of Normal Precipitation (PNP).

    This function accepts both NumPy arrays and xarray DataArrays. Type checkers
    will narrow the return type based on the input type.

    For NumPy inputs, all temporal parameters are required.
    For xarray inputs, temporal parameters are optional and will be inferred from
    coordinate attributes if not provided.

    .. warning:: **Beta Feature (xarray path only)** -- When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface (parameter inference, metadata handling, coordinate
       preservation) may change in future minor releases. The NumPy array interface
       is stable.

    Args:
        values: 1-D numpy array or xarray DataArray of precipitation values.
        scale: Number of time steps over which the normal is computed.
        data_start_year: Initial year of the input dataset (required for NumPy,
            optional for xarray).
        calibration_year_initial: Initial year of calibration period (required
            for NumPy, optional for xarray).
        calibration_year_final: Final year of calibration period (required for
            NumPy, optional for xarray).
        periodicity: Time series periodicity ('monthly' or 'daily'). Required
            for NumPy, optional for xarray.

    Returns:
        PNP values as numpy.ndarray or xarray.DataArray (matches input type).
    """
    result = _delegate(_wrapped_percentage_of_normal, values, *args, **_translate_pnp_calibration_kwargs(kwargs))
    if isinstance(result, xr.DataArray):
        result.attrs["calibration_start_year"] = result.attrs["calibration_year_initial"]
        result.attrs["calibration_end_year"] = result.attrs["calibration_year_final"]
    return result


# PCI (Precipitation Concentration Index) overloads
# PCI uses a manual wrapper because output shape (scalar) differs from input (365/366 days)
@overload
def pci(
    rainfall_mm: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...


@overload
def pci(
    rainfall_mm: xr.DataArray,
) -> xr.DataArray: ...


def pci(
    rainfall_mm: npt.NDArray[np.float64] | xr.DataArray,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute Precipitation Concentration Index (PCI).

    This function accepts both NumPy arrays and xarray DataArrays. Type checkers
    will narrow the return type based on the input type.

    PCI requires exactly 365 or 366 daily rainfall values representing a single
    year. The output is a single scalar value.

    .. warning:: **Beta Feature (xarray path only)** -- When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface (metadata handling) may change in future minor
       releases. The NumPy array interface is stable.

    Args:
        rainfall_mm: 1-D array or DataArray of daily rainfall values in mm.
            Must contain exactly 365 or 366 values (one full year).

    Returns:
        PCI value as numpy.ndarray (shape (1,)) or scalar xarray.DataArray.
    """
    input_type = detect_input_type(rainfall_mm)

    if input_type == InputType.NUMPY:
        assert isinstance(rainfall_mm, np.ndarray)
        return indices.pci(rainfall_mm)

    # xarray path: extract values, compute, rewrap with CF metadata
    assert isinstance(rainfall_mm, xr.DataArray)
    result_values = indices.pci(rainfall_mm.values)

    # build CF metadata attributes for the scalar output
    cf_meta = CF_METADATA["pci"]
    from climate_indices import __version__

    output_attrs: dict[str, Any] = {}
    output_attrs.update(cf_meta)
    output_attrs["climate_indices_version"] = __version__
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="seconds")
    output_attrs["history"] = f"{timestamp} PCI computed by climate_indices {__version__}"

    # PCI output is a scalar (shape (1,)), return as 0-D DataArray
    return xr.DataArray(
        result_values[0],
        attrs=output_attrs,
    )


# ETo Thornthwaite overloads
@overload
def pet_thornthwaite(
    temperature: npt.NDArray[np.float64],
    latitude: float,
    data_start_year: int,
    time_dim: str = "time",
) -> npt.NDArray[np.float64]: ...


@overload
def pet_thornthwaite(
    temperature: xr.DataArray,
    latitude: float | np.floating | xr.DataArray,
    data_start_year: int | None = None,
    time_dim: str = "time",
) -> xr.DataArray: ...


def pet_thornthwaite(
    temperature: Any, latitude: Any, *args: Any, **kwargs: Any
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute potential evapotranspiration using Thornthwaite method.

    This function accepts both NumPy arrays and xarray DataArrays. Type checkers
    will narrow the return type based on the input type.

    For NumPy inputs, ``data_start_year`` and scalar ``latitude`` are required.
    For xarray inputs, ``data_start_year`` is inferred from the time coordinate
    if not provided, and ``latitude`` may be a scalar or DataArray for spatial
    broadcasting.

    .. warning:: **Beta Feature (xarray path only)** -- When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface may change in future minor releases. The NumPy array
       interface is stable.

    Args:
        temperature: Monthly average temperature values in degrees Celsius.
            For numpy: 1-D array of monthly temperatures.
            For xarray: DataArray with time dimension (may have spatial dims).
        latitude: Latitude in degrees north (range: -90 to 90).
            For numpy: scalar float.
            For xarray: scalar float or DataArray(lat,) for spatial broadcasting.
        data_start_year: Initial year of the input dataset (required for NumPy,
            optional for xarray where it is inferred from the time coordinate).
        time_dim: Name of the time dimension in the input DataArray.

    Returns:
        PET values in mm/month as numpy.ndarray or xarray.DataArray.
    """
    return _delegate(_pet_thornthwaite_impl, temperature, latitude, *args, **kwargs)


# ETo Hargreaves overloads
@overload
def pet_hargreaves(
    daily_tmin_celsius: npt.NDArray[np.float64],
    daily_tmax_celsius: npt.NDArray[np.float64],
    latitude: float,
    time_dim: str = "time",
) -> npt.NDArray[np.float64]: ...


@overload
def pet_hargreaves(
    daily_tmin_celsius: xr.DataArray,
    daily_tmax_celsius: xr.DataArray,
    latitude: float | np.floating | xr.DataArray,
    time_dim: str = "time",
) -> xr.DataArray: ...


def pet_hargreaves(
    daily_tmin_celsius: Any,
    daily_tmax_celsius: Any,
    latitude: Any,
    *args: Any,
    **kwargs: Any,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute potential evapotranspiration using Hargreaves method.

    This function accepts both NumPy arrays and xarray DataArrays. Type checkers
    will narrow the return type based on the input type.

    For NumPy inputs, scalar ``latitude`` is required.
    For xarray inputs, ``latitude`` may be a scalar or DataArray for spatial
    broadcasting.

    .. warning:: **Beta Feature (xarray path only)** -- When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface may change in future minor releases. The NumPy array
       interface is stable.

    Args:
        daily_tmin_celsius: Daily minimum temperature values in degrees Celsius.
            For numpy: 1-D array of daily temperatures.
            For xarray: DataArray with time dimension (may have spatial dims).
        daily_tmax_celsius: Daily maximum temperature values in degrees Celsius.
            For numpy: 1-D array of daily temperatures.
            For xarray: DataArray with time dimension (may have spatial dims).
        latitude: Latitude in degrees north (range: -90 to 90).
            For numpy: scalar float.
            For xarray: scalar float or DataArray(lat,) for spatial broadcasting.
        time_dim: Name of the time dimension in the input DataArray.

    Returns:
        PET values in mm/day as numpy.ndarray or xarray.DataArray.
    """
    return _delegate(_pet_hargreaves_impl, daily_tmin_celsius, daily_tmax_celsius, latitude, *args, **kwargs)


# ETo Penman-Monteith overloads
@overload
def pet_penman_monteith(
    daily_tmin_celsius: npt.NDArray[np.float64],
    daily_tmax_celsius: npt.NDArray[np.float64],
    latitude: float,
    elevation_m: float,
    wind_speed_m_s: npt.NDArray[np.float64] | float,
    day_of_year: npt.NDArray[np.float64],
    wind_speed_height_m: float = 2.0,
    humidity: pm_eto.HumidityInputs | None = None,
    radiation: pm_eto.RadiationInputs | None = None,
    soil_heat_flux_mj_m2_day: npt.NDArray[np.float64] | float = 0.0,
    albedo: float = 0.23,
    time_dim: str = "time",
) -> npt.NDArray[np.float64]: ...


@overload
def pet_penman_monteith(
    daily_tmin_celsius: xr.DataArray,
    daily_tmax_celsius: xr.DataArray,
    latitude: float | np.floating | xr.DataArray,
    elevation_m: float | np.floating | xr.DataArray,
    wind_speed_m_s: np.ndarray | xr.DataArray | float,
    day_of_year: np.ndarray | xr.DataArray | None = None,
    wind_speed_height_m: float = 2.0,
    humidity: pm_eto.HumidityInputs | None = None,
    radiation: pm_eto.RadiationInputs | None = None,
    soil_heat_flux_mj_m2_day: np.ndarray | xr.DataArray | float = 0.0,
    albedo: float = 0.23,
    time_dim: str = "time",
) -> xr.DataArray: ...


def pet_penman_monteith(
    daily_tmin_celsius: Any,
    daily_tmax_celsius: Any,
    latitude: Any,
    elevation_m: Any,
    wind_speed_m_s: Any,
    day_of_year: Any = None,
    wind_speed_height_m: float = 2.0,
    humidity: pm_eto.HumidityInputs | None = None,
    radiation: pm_eto.RadiationInputs | None = None,
    soil_heat_flux_mj_m2_day: Any = 0.0,
    albedo: float = 0.23,
    time_dim: str = "time",
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute potential evapotranspiration using FAO-56 Penman-Monteith.

    This function accepts both NumPy arrays and xarray DataArrays. Type checkers
    will narrow the return type based on the input type.

    For NumPy inputs, ``day_of_year`` and scalar ``latitude`` are required. For
    xarray inputs, ``day_of_year`` is inferred from the time coordinate if not
    provided, and ``latitude`` may be a scalar or DataArray for spatial
    broadcasting.

    .. warning:: **Beta Feature (xarray path only)** -- When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface may change in future minor releases. The NumPy array
       interface is stable.

    Args:
        daily_tmin_celsius: Daily minimum temperature values in degrees Celsius.
        daily_tmax_celsius: Daily maximum temperature values in degrees Celsius.
        latitude: Latitude in degrees north (range: -90 to 90).
        elevation_m: Station elevation above sea level in metres.
        wind_speed_m_s: Wind speed measured at ``wind_speed_height_m`` [m s-1].
        day_of_year: Day of the year (required for NumPy, inferred for xarray).
        wind_speed_height_m: Height at which the wind speed was measured [m].
        humidity: Optional actual-vapour-pressure inputs, in pathway precedence
            order; see :class:`climate_indices.pm_eto.HumidityInputs`.
        radiation: Optional solar-radiation inputs, in pathway precedence order;
            see :class:`climate_indices.pm_eto.RadiationInputs`.
        soil_heat_flux_mj_m2_day: Soil heat flux density [MJ m-2 day-1].
        albedo: Canopy reflection coefficient (0.23 for the grass reference).
        time_dim: Name of the time dimension in the input DataArray.

    Returns:
        PET values in mm/day as numpy.ndarray or xarray.DataArray.
    """
    return _delegate(
        _pet_penman_monteith_impl,
        daily_tmin_celsius,
        daily_tmax_celsius,
        latitude,
        elevation_m,
        wind_speed_m_s,
        day_of_year,
        wind_speed_height_m,
        humidity,
        radiation,
        soil_heat_flux_mj_m2_day,
        albedo,
        time_dim,
    )


# EDDI overloads
@overload
def eddi(
    pet_values: npt.NDArray[np.float64],
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: Periodicity,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64]: ...


@overload
def eddi(
    pet_values: xr.DataArray,
    scale: int,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: Periodicity | None = None,
    spatial_time_major: bool = False,
) -> xr.DataArray: ...


def eddi(pet_values: Any, *args: Any, **kwargs: Any) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute EDDI (Evaporative Demand Drought Index).

    Accepts both NumPy arrays and xarray DataArrays. Type checkers narrow the
    return type based on the input type.

    For NumPy inputs, all temporal parameters are required.
    For xarray inputs, temporal parameters are optional and inferred from
    coordinate attributes if not provided.

    .. warning:: **Beta Feature (xarray path only)** — When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface may change in future minor releases. The NumPy array
       interface is stable.

    Args:
        pet_values: 1-D numpy array or xarray DataArray of PET values.
        scale: Number of time steps over which values should be scaled.
        data_start_year: Initial year of the input dataset (required for NumPy,
            optional for xarray).
        calibration_year_initial: Initial year of calibration period (required
            for NumPy, optional for xarray).
        calibration_year_final: Final year of calibration period (required for
            NumPy, optional for xarray).
        periodicity: Time series periodicity ('monthly' or 'daily'). Required
            for NumPy, optional for xarray.
        spatial_time_major: Declares an ambiguous 3+-D NumPy ``pet_values`` as a
            time-major ``(time, *cells)`` block (per ADR-0009). Only used for
            NumPy inputs.

    Returns:
        EDDI values as numpy.ndarray or xarray.DataArray (matches input type).
    """
    return _delegate(_wrapped_eddi, pet_values, *args, **kwargs)


# PDSI (Palmer Drought Severity Index family) overloads
@overload
def pdsi(
    precips: npt.NDArray[np.float64],
    pet: npt.NDArray[np.float64],
    awc: float | npt.NDArray[np.float64],
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    fitting_params: dict[str, Any] | None = None,
    spatial_time_major: bool = False,
    time_dim: str = "time",
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
    awc: float | npt.NDArray[np.float64] | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    fitting_params: dict[str, Any] | None = None,
    spatial_time_major: bool = False,
    time_dim: str = "time",
) -> xr.Dataset: ...


def pdsi(
    precips: Any,
    pet: Any,
    awc: Any,
    *args: Any,
    **kwargs: Any,
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
    """Compute the standard Palmer drought indices (PDSI, PHDI, PMDI, Z-Index).

    This function accepts both NumPy arrays and xarray DataArrays. Type checkers
    will narrow the return type based on the input type.

    For NumPy inputs, all temporal parameters are required and the return is the
    five-item tuple :func:`climate_indices.palmer.pdsi` produces. For xarray
    inputs, temporal parameters are optional and inferred from the time
    coordinate, and the return is an ``xr.Dataset`` with one variable per index
    (``pdsi``, ``phdi``, ``pmdi``, ``z_index``), each carrying its own CF metadata.

    .. warning:: **Beta Feature (xarray path only)** — When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface (parameter inference, metadata handling, coordinate
       preservation) may change in future minor releases. The NumPy array interface
       is stable.

    Args:
        precips: Monthly precipitation values in inches.
        pet: Monthly potential evapotranspiration values in inches, matching
            ``precips``.
        awc: Available water capacity (soil constant) in inches. A scalar, a NumPy
            array, or a DataArray whose cell coordinates match the precipitation grid.
        data_start_year: Initial year of the input dataset (required for NumPy,
            optional for xarray).
        calibration_year_initial: Initial year of the calibration period (required
            for NumPy, optional for xarray).
        calibration_year_final: Final year of the calibration period (required for
            NumPy, optional for xarray).
        fitting_params: Optional dict of pre-computed Palmer fitting parameters.
        spatial_time_major: Declares an ambiguous 3+-D NumPy ``precips``/``pet`` as a
            time-major ``(time, *cells)`` block (per ADR-0009). Only used for NumPy
            inputs.
        time_dim: Name of the time dimension for xarray inputs (default: ``"time"``).

    Returns:
        The five-item PDSI-family tuple for numpy.ndarray input, or an
        ``xr.Dataset`` of the four indices for xarray.DataArray input.
    """
    return _delegate(_palmer_pdsi_impl, precips, pet, awc, *args, **kwargs)


# Fit-diagnostics overloads
@overload
def fit_diagnostics(
    values: npt.NDArray[np.float64],
    scale: int,
    distribution: Distribution,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: Periodicity,
    fitting_params: dict[str, Any] | None = None,
    spatial_time_major: bool = False,
    time_dim: str = "time",
) -> compute.FitDiagnostics: ...


@overload
def fit_diagnostics(
    values: xr.DataArray,
    scale: int,
    distribution: Distribution,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    periodicity: Periodicity | None = None,
    fitting_params: dict[str, Any] | None = None,
    spatial_time_major: bool = False,
    time_dim: str = "time",
) -> xr.Dataset: ...


def fit_diagnostics(
    values: Any, scale: Any, distribution: Any, *args: Any, **kwargs: Any
) -> compute.FitDiagnostics | xr.Dataset:
    """Fit a distribution and return per-calendar-step diagnostics.

    This function accepts both NumPy arrays and xarray DataArrays. Type checkers
    will narrow the return type based on the input type.

    For NumPy inputs, all temporal parameters are required and the return is the
    :class:`climate_indices.compute.FitDiagnostics` that
    :func:`climate_indices.indices.fit_diagnostics` produces. For xarray inputs,
    temporal parameters are optional and inferred from the time coordinate, and the
    return is an ``xr.Dataset`` of the fitted parameters, ``prob_zero``,
    ``n_valid``, ``ks_statistic``, ``ks_p_value``, and ``distribution_used`` over a
    ``month`` or ``dayofyear`` dimension plus the input's cell dimensions. A Pearson
    Type III request keeps both parameter families, with the inapplicable one NaN
    per cell, and ``distribution_used`` names the family that does apply.

    .. warning:: **Beta Feature (xarray path only)** — When called with an
       ``xr.DataArray`` input, this function uses the beta xarray adapter layer.
       The xarray interface (parameter inference, metadata handling, coordinate
       preservation) may change in future minor releases. The NumPy array interface
       is stable.

    Args:
        values: NumPy array or xarray DataArray of non-negative values.
        scale: Number of time steps over which values are accumulated before fitting.
        distribution: Distribution type for the fit, gamma or Pearson Type III.
        data_start_year: Initial year of the input dataset (required for NumPy,
            optional for xarray).
        calibration_year_initial: Initial year of the calibration period (required
            for NumPy, optional for xarray).
        calibration_year_final: Final year of the calibration period (required for
            NumPy, optional for xarray).
        periodicity: Time series periodicity ('monthly' or 'daily'). Required
            for NumPy, optional for xarray.
        fitting_params: Optional dict of pre-computed distribution fitting
            parameters.
        spatial_time_major: Declares an ambiguous 3+-D NumPy ``values`` as a
            time-major ``(time, *cells)`` block (per ADR-0009). Only used for NumPy
            inputs.
        time_dim: Name of the time dimension for xarray inputs (default: ``"time"``).

    Returns:
        A :class:`climate_indices.compute.FitDiagnostics` for numpy.ndarray input,
        or an ``xr.Dataset`` of the diagnostics for xarray.DataArray input.
    """
    return _delegate(_fit_diagnostics_impl, values, scale, distribution, *args, **kwargs)


for _public_function in (
    spi,
    spei,
    percentage_of_normal,
    eddi,
    pdsi,
    fit_diagnostics,
    pet_thornthwaite,
    pet_hargreaves,
    pet_penman_monteith,
):
    _restore_runtime_signature(_public_function)

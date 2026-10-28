"""Daily indices of flood potential, not observed flooding.

Effective precipitation is the shared input for EDI and the Flood Index.
The NumPy API is stable; DataArray dispatch is stable from 3.1.0.
"""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices._calibration_period import CalibrationPeriod
from climate_indices.flood._antecedent import APIResult, APIState
from climate_indices.flood._antecedent import antecedent_precipitation_index as _numpy_api
from climate_indices.flood._edi import edi as _numpy_edi
from climate_indices.flood._events import FloodEvents, FloodStatistics, flood_event_summary, flood_events
from climate_indices.flood._if import flood_index as _numpy_flood_index
from climate_indices.flood._pe import effective_precipitation as _numpy_pe
from climate_indices.flood._xarray import (
    _api_xarray,
    _wrapped_edi,
    _wrapped_flood_index,
    _wrapped_pe,
    convert_pe_input,
)


@overload
def effective_precipitation(
    precipitation: xr.DataArray, *, duration: int = 365, spatial_time_major: bool = False
) -> xr.DataArray: ...


@overload
def effective_precipitation(
    precipitation: npt.ArrayLike, *, duration: int = 365, spatial_time_major: bool = False
) -> npt.NDArray[np.float64]: ...


def effective_precipitation(
    precipitation: npt.ArrayLike | xr.DataArray, *, duration: int = 365, spatial_time_major: bool = False
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Calculate effective precipitation in mm (flood potential, not flooding).

    NumPy input follows the stable fixed-window kernel; DataArray input uses
    stable xarray dispatch with Gregorian daily conversion, lazy Dask spatial
    chunks, CF precipitation units, and output metadata.
    """
    if isinstance(precipitation, xr.DataArray):
        return _wrapped_pe(convert_pe_input(precipitation, argument_name="precipitation"), duration=duration)
    return _numpy_pe(precipitation, duration=duration, spatial_time_major=spatial_time_major)


@overload
def edi(
    pe: xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    *,
    duration: int = 365,
    spatial_time_major: bool = False,
) -> xr.DataArray: ...


@overload
def edi(
    pe: npt.ArrayLike,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    *,
    duration: int = 365,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64]: ...


def edi(
    pe: npt.ArrayLike | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    *,
    duration: int = 365,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Calculate fixed-window EDI from PE (flood potential, not flooding).

    DataArray dispatch is stable; omitted years are inferred from daily coordinates.
    PE must be computed with the same duration and in millimeters. A DataArray's
    Gregorian PE is interpolated to the all-leap layout, so non-leap February 29
    PE is the mean of adjacent PE values, not PE of interpolated rainfall (ADR-0014).
    """
    if isinstance(pe, xr.DataArray):
        kwargs: dict[str, Any] = {"duration": duration}
        if (
            calibration_year_final is None
            and "time" in pe.coords
            and pe.sizes.get("time")
            and np.issubdtype(pe.time.dtype, np.datetime64)
            and not np.isnat(pe.time.values[0])
            and not np.isnat(pe.time.values[-1])
        ):
            calibration_year_final = CalibrationPeriod.infer_years(pe.time.values, complete_years=True)[1]
        for key, value in (
            ("data_start_year", data_start_year),
            ("calibration_year_initial", calibration_year_initial),
            ("calibration_year_final", calibration_year_final),
        ):
            if value is not None:
                kwargs[key] = value
        return _wrapped_edi(convert_pe_input(pe, argument_name="pe"), **kwargs)
    if data_start_year is None or calibration_year_initial is None or calibration_year_final is None:
        raise TypeError("NumPy EDI requires data_start_year and both calibration years")
    return _numpy_edi(
        pe,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        duration=duration,
        spatial_time_major=spatial_time_major,
    )


@overload
def flood_index(
    pe: xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    *,
    year_start_month: int,
    spatial_time_major: bool = False,
) -> xr.DataArray: ...


@overload
def flood_index(
    pe: npt.ArrayLike,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    *,
    year_start_month: int,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64]: ...


def flood_index(
    pe: npt.ArrayLike | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    *,
    year_start_month: int,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Calculate the Flood Index from PE (flood potential, not flooding).

    DataArray dispatch is stable. The omitted calibration start defaults to the
    first coordinate year plus one, excluding the standard 365-day PE warm-up
    year without loading data. This also skips a year for PE with no warm-up
    NaNs or a shorter duration; pass an explicit start to include it, or a later
    start for longer warm-up or leading gaps. The omitted calibration end is
    the start year of the last complete annual period; the omitted data start
    is the first coordinate year. ``year_start_month`` defines annual periods.
    At least two complete calibration periods must remain. A DataArray's
    Gregorian PE is interpolated to the all-leap layout, so non-leap February 29
    PE is the mean of adjacent PE values, not PE of interpolated rainfall (ADR-0014).
    """
    if isinstance(pe, xr.DataArray):
        kwargs: dict[str, Any] = {"year_start_month": year_start_month}
        if (
            "time" in pe.coords
            and pe.sizes.get("time")
            and np.issubdtype(pe.time.dtype, np.datetime64)
            and not np.isnat(pe.time.values[0])
            and not np.isnat(pe.time.values[-1])
        ):
            if calibration_year_initial is None:
                calibration_year_initial = CalibrationPeriod.infer_years(pe.time.values)[0] + 1
            if calibration_year_final is None and 1 <= year_start_month <= 12:
                calibration_year_final = CalibrationPeriod.infer_years(
                    pe.time.values, complete_years=True, year_start_month=year_start_month
                )[1]
        for key, value in (
            ("data_start_year", data_start_year),
            ("calibration_year_initial", calibration_year_initial),
            ("calibration_year_final", calibration_year_final),
        ):
            if value is not None:
                kwargs[key] = value
        return _wrapped_flood_index(convert_pe_input(pe, argument_name="pe"), **kwargs)
    if data_start_year is None or calibration_year_initial is None or calibration_year_final is None:
        raise TypeError("NumPy Flood Index requires data_start_year and both calibration years")
    return _numpy_flood_index(
        pe,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        year_start_month=year_start_month,
        spatial_time_major=spatial_time_major,
    )


@overload
def antecedent_precipitation_index(
    precipitation: xr.DataArray,
    k: float,
    *,
    initial_state: APIState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
    spatial_time_major: bool = False,
    time_dim: str = "time",
) -> xr.DataArray | APIResult: ...


@overload
def antecedent_precipitation_index(
    precipitation: npt.ArrayLike,
    k: float,
    *,
    initial_state: APIState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
    spatial_time_major: bool = False,
    time_dim: str = "time",
) -> npt.NDArray[np.float64] | APIResult: ...


def antecedent_precipitation_index(
    precipitation: npt.ArrayLike | xr.DataArray,
    k: float,
    *,
    initial_state: APIState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
    spatial_time_major: bool = False,
    time_dim: str = "time",
) -> npt.NDArray[np.float64] | xr.DataArray | APIResult:
    """Compute daily antecedent wetness in mm (flood potential, not flooding).

    NumPy input follows the stable recurrence; DataArray input uses stable xarray
    dispatch with CF unit conversion and spatial Dask blocks. Its ``time_dim``
    must be a single chunk, with any spatial chunking. An attached time
    coordinate must hold consecutive daily samples; a dimension-only time axis
    is aligned positionally and trusted as daily. ``return_state=True`` loads
    the values and returns a NumPy :class:`APIState` for bitwise-equivalent
    append processing.

    Args:
        precipitation: Non-negative daily precipitation in mm, or DataArray
            with convertible CF ``units`` (missing units assume mm).
        k: Decay constant strictly between zero and one.
        initial_state: State from a prior call; spatial fields match input cells.
        return_state: Return values and copied final NumPy state; DataArray
            values are loaded eagerly.
        spin_up: Leading days to compute but omit from values.
        nan_policy: Missing-day policy, ``"propagate"`` or ``"bridge"``.
        max_gap_days: Maximum gap length for bridge; zero for propagate.
        spatial_time_major: NumPy Spatial Block declaration; ignored for DataArray.
        time_dim: Time dimension name for DataArray input.

    Returns:
        Daily API values, or :class:`APIResult` when state requested.
    """
    if isinstance(precipitation, xr.DataArray):
        return _api_xarray(
            precipitation,
            k,
            initial_state=initial_state,
            return_state=return_state,
            spin_up=spin_up,
            nan_policy=nan_policy,
            max_gap_days=max_gap_days,
            time_dim=time_dim,
        )
    return _numpy_api(
        precipitation,
        k,
        initial_state=initial_state,
        return_state=return_state,
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
        spatial_time_major=spatial_time_major,
    )


__all__ = [
    "APIResult",
    "APIState",
    "FloodEvents",
    "FloodStatistics",
    "antecedent_precipitation_index",
    "edi",
    "effective_precipitation",
    "flood_event_summary",
    "flood_events",
    "flood_index",
]

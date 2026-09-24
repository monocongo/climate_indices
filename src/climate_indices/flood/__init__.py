"""Daily indices of flood potential, not observed flooding.

Effective precipitation is the shared input for EDI and the Flood Index.
The NumPy API is stable; DataArray dispatch is beta.
"""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from climate_indices.flood._edi import edi as _numpy_edi
from climate_indices.flood._if import flood_index as _numpy_flood_index
from climate_indices.flood._pe import effective_precipitation as _numpy_pe
from climate_indices.flood._xarray import (
    _wrapped_edi,
    _wrapped_flood_index,
    _wrapped_pe,
    convert_pe_input,
)


@overload
def effective_precipitation(
    precipitation: npt.ArrayLike, *, duration: int = 365, spatial_time_major: bool = False
) -> npt.NDArray[np.float64]: ...


@overload
def effective_precipitation(
    precipitation: xr.DataArray, *, duration: int = 365, spatial_time_major: bool = False
) -> xr.DataArray: ...


def effective_precipitation(
    precipitation: npt.ArrayLike | xr.DataArray, *, duration: int = 365, spatial_time_major: bool = False
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Calculate effective precipitation in mm (flood potential, not flooding).

    NumPy input follows the stable fixed-window kernel; DataArray input uses
    beta xarray dispatch with Gregorian daily conversion, lazy Dask spatial
    chunks, CF precipitation units, and output metadata.
    """
    if isinstance(precipitation, xr.DataArray):
        return _wrapped_pe(convert_pe_input(precipitation, argument_name="precipitation"), duration=duration)
    return _numpy_pe(precipitation, duration=duration, spatial_time_major=spatial_time_major)


def _last_complete_calibration_year(pe: xr.DataArray, year_start_month: int) -> int | None:
    """Return last complete annual period's start year, if time is labeled."""
    if (
        "time" not in pe.coords
        or not pe.sizes.get("time")
        or not np.issubdtype(pe.time.dtype, np.datetime64)
        or not 1 <= year_start_month <= 12
    ):
        return None
    last = pd.Timestamp(pe.time.values[-1])
    if year_start_month == 1:
        return int(last.year if last.is_year_end else last.year - 1)
    return int(last.year - 1 - (last.month < year_start_month))


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


def edi(
    pe: npt.ArrayLike | xr.DataArray,
    data_start_year: int | None = None,
    calibration_year_initial: int | None = None,
    calibration_year_final: int | None = None,
    *,
    duration: int = 365,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Calculate fixed-window EDI from PE (moisture conditions, not flooding).

    DataArray dispatch is beta; omitted years are inferred from daily coordinates.
    PE must be computed with the same duration and in millimeters.
    """
    if isinstance(pe, xr.DataArray):
        kwargs: dict[str, Any] = {"duration": duration}
        if calibration_year_final is None:
            calibration_year_final = _last_complete_calibration_year(pe, 1)
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
    pe: npt.ArrayLike,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    *,
    year_start_month: int,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64]: ...


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

    DataArray dispatch is beta; omitted years are inferred from daily coordinates.
    ``year_start_month`` defines complete annual calibration periods.
    """
    if isinstance(pe, xr.DataArray):
        kwargs: dict[str, Any] = {"year_start_month": year_start_month}
        if calibration_year_final is None:
            calibration_year_final = _last_complete_calibration_year(pe, year_start_month)
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


__all__ = ["edi", "effective_precipitation", "flood_index"]

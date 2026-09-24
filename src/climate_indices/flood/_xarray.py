"""Beta xarray dispatch for daily flood-potential indices."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.compute import Periodicity
from climate_indices.fire._units import _convert_precipitation_units
from climate_indices.flood._edi import edi as _numpy_edi
from climate_indices.flood._if import flood_index as _numpy_flood_index
from climate_indices.flood._pe import effective_precipitation as _numpy_pe
from climate_indices.xarray_adapter import xarray_adapter


def _pe_daily(
    precipitation: np.ndarray[Any, Any],
    *,
    duration: int = 365,
    periodicity: Periodicity = Periodicity.daily,
    spatial_time_major: bool = False,
) -> np.ndarray[Any, Any]:
    return _numpy_pe(precipitation, duration=duration, spatial_time_major=spatial_time_major)


def _edi_daily(
    pe: np.ndarray[Any, Any],
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    *,
    duration: int = 365,
    periodicity: Periodicity = Periodicity.daily,
    spatial_time_major: bool = False,
) -> np.ndarray[Any, Any]:
    return _numpy_edi(
        pe,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        duration=duration,
        spatial_time_major=spatial_time_major,
    )


def _flood_index_daily(
    pe: np.ndarray[Any, Any],
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    *,
    year_start_month: int,
    periodicity: Periodicity = Periodicity.daily,
    spatial_time_major: bool = False,
) -> np.ndarray[Any, Any]:
    return _numpy_flood_index(
        pe,
        data_start_year,
        calibration_year_initial,
        calibration_year_final,
        year_start_month=year_start_month,
        spatial_time_major=spatial_time_major,
    )


_wrapped_pe = xarray_adapter(
    cf_metadata=CF_METADATA["effective_precipitation"],  # type: ignore[arg-type]
    index_display_name="Effective Precipitation",
    calculation_metadata_keys=["duration"],
    spatial_kernel=True,
    validate_calibration_sample=False,
)(_pe_daily)
_wrapped_edi = xarray_adapter(
    cf_metadata=CF_METADATA["edi"],  # type: ignore[arg-type]
    index_display_name="EDI",
    calculation_metadata_keys=["duration", "calibration_year_initial", "calibration_year_final"],
    spatial_kernel=True,
    validate_calibration_sample=False,
)(_edi_daily)
_wrapped_flood_index = xarray_adapter(
    cf_metadata=CF_METADATA["flood_index"],  # type: ignore[arg-type]
    index_display_name="Flood Index",
    calculation_metadata_keys=["year_start_month", "calibration_year_initial", "calibration_year_final"],
    spatial_kernel=True,
    validate_calibration_sample=False,
)(_flood_index_daily)


def convert_pe_input(data: xr.DataArray, *, argument_name: str) -> xr.DataArray:
    """Convert daily precipitation or effective precipitation to millimeters lazily."""
    converted = _convert_precipitation_units(data, "mm", argument_name=f"{argument_name}.attrs['units']")
    # Unconverted units return the original object; never change caller metadata.
    if converted is data:
        converted = data.copy(deep=False)
    # xarray arithmetic drops attrs; keep source history for the adapter's output.
    converted.attrs = {**data.attrs, "units": "mm"}
    return converted

"""Beta xarray dispatch for daily flood-potential indices."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.compute import Periodicity
from climate_indices.fire._common import _validate_recurrence_options, _wrap_spatial
from climate_indices.fire._units import _convert_precipitation_units, _validate_daily_time_coordinate
from climate_indices.flood._antecedent import APIResult, APIState, _resume_state, _validate_decay
from climate_indices.flood._antecedent import antecedent_precipitation_index as _numpy_api
from climate_indices.flood._edi import edi as _numpy_edi
from climate_indices.flood._if import flood_index as _numpy_flood_index
from climate_indices.flood._pe import effective_precipitation as _numpy_pe
from climate_indices.validation import validate_dask_chunks, validate_time_dimension, validate_time_monotonicity
from climate_indices.xarray_adapter import build_output_attrs, xarray_adapter


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
    calculation_metadata_keys=["calibration_year_initial", "calibration_year_final"],
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


def _api_xarray(
    precipitation: xr.DataArray,
    k: float,
    *,
    initial_state: APIState | None,
    return_state: bool,
    spin_up: int,
    nan_policy: str,
    max_gap_days: int,
    time_dim: str,
) -> xr.DataArray | APIResult:
    """Run the NumPy recurrence per spatial block, returning NumPy state."""
    _validate_decay(k)
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_state", None, initial_state)
    spin_up = int(spin_up)
    validate_time_dimension(precipitation, time_dim)
    if time_dim in precipitation.coords:
        validate_time_monotonicity(precipitation.coords[time_dim])
        _validate_daily_time_coordinate(precipitation, time_dim)
    validate_dask_chunks(precipitation, time_dim)
    rain = convert_pe_input(precipitation, argument_name="precipitation")
    spatial_dims = tuple(dim for dim in rain.dims if dim != time_dim)
    spatial_shape = tuple(rain.sizes[dim] for dim in spatial_dims)
    state_args: list[xr.DataArray] = []
    if initial_state is not None:
        # Reject invalid state before returning a lazy graph; the core validates each tile again.
        _resume_state(initial_state, spatial_shape or (1,))
        gap = initial_state.trailing_gap_days
        if gap is None:
            gap = np.full(spatial_shape, -1, dtype=np.int64)
        state_args = [
            _wrap_spatial(initial_state.api, spatial_shape, spatial_dims),
            _wrap_spatial(gap, spatial_shape, spatial_dims),
        ]

    output_len = max(rain.sizes[time_dim] - spin_up, 0)

    def _block(values: np.ndarray, *state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        initial = None
        time_first = np.moveaxis(values, -1, 0).copy()
        # A 2-D NumPy input means (years, days), not a Spatial Block.
        single_spatial_axis = time_first.ndim == 2
        if single_spatial_axis:
            time_first = time_first[..., None]
        if state:
            gaps = state[1].astype(np.int64)
            api_seed = state[0].copy()
            if single_spatial_axis:
                gaps = gaps[..., None]
                api_seed = api_seed[..., None]
            initial = APIState(api_seed, None if np.all(gaps < 0) else gaps.copy())
        result = _numpy_api(
            time_first,
            k,
            initial_state=initial,
            return_state=True,
            spin_up=spin_up,
            nan_policy=nan_policy,  # type: ignore[arg-type]
            max_gap_days=max_gap_days,
            spatial_time_major=True,
        )
        assert isinstance(result, APIResult)
        assert isinstance(result.values, np.ndarray)
        gaps_out = result.state.trailing_gap_days
        if gaps_out is None:
            gaps_out = np.full(result.state.api.shape, -1, dtype=np.int64)
        if single_spatial_axis:
            return np.moveaxis(result.values[..., 0], 0, -1), result.state.api[..., 0], gaps_out[..., 0]
        return np.moveaxis(result.values, 0, -1), result.state.api, gaps_out

    values, api, gaps = xr.apply_ufunc(
        _block,
        rain,
        *state_args,
        input_core_dims=[[time_dim]] + [[] for _ in state_args],
        output_core_dims=[[time_dim], [], []],
        exclude_dims={time_dim},
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {time_dim: output_len}},
        output_dtypes=[float, float, np.int64],
    )
    values = values.transpose(*rain.dims)
    if time_dim in rain.coords:
        values = values.assign_coords({time_dim: rain.coords[time_dim].values[spin_up:]})
    values.attrs = build_output_attrs(
        precipitation,
        cf_metadata=CF_METADATA["antecedent_precipitation_index"],  # type: ignore[arg-type]
        calculation_metadata={"k": k, "nan_policy": nan_policy, "max_gap_days": max_gap_days},
        index_name="API",
    )
    if not return_state:
        result_da: xr.DataArray = values
        return result_da
    # Load the shared graph once so the state and values describe the same computation.
    loaded = xr.Dataset({"values": values, "api": api, "gaps": gaps}).load()
    final_gaps = loaded["gaps"].values
    return APIResult(
        values=loaded["values"],
        state=APIState(loaded["api"].values, None if np.all(final_gaps < 0) else final_gaps),
    )


def convert_pe_input(data: xr.DataArray, *, argument_name: str) -> xr.DataArray:
    """Convert daily precipitation or effective precipitation to millimeters lazily."""
    converted = _convert_precipitation_units(data, "mm", argument_name=f"{argument_name}.attrs['units']")
    # Unconverted units return the original object; never change caller metadata.
    if converted is data:
        converted = data.copy(deep=False)
    # xarray arithmetic drops attrs; keep source history for the adapter's output.
    converted.attrs = {**data.attrs, "units": "mm"}
    return converted

"""Hot-Dry-Windy Index (HDW)."""

from __future__ import annotations

import time
from typing import overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices import pm_eto
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import CoordinateValidationError, InvalidArgumentError
from climate_indices.fire._common import _as_float_array
from climate_indices.fire._units import _convert_temperature_units
from climate_indices.logging_config import get_logger
from climate_indices.performance import check_large_array_memory
from climate_indices.xarray_adapter import (
    InputType,
    _build_output_attrs,
    _validate_dask_chunks,
    detect_input_type,
)

# retrieve structlog logger for this module
_logger = get_logger(__name__)


# HDW analyzes the lowest 500 m above ground level (Srock et al., 2018)
_HDW_LAYER_TOP_METERS = 500.0

# pm_eto saturation vapor pressure is kPa; HDW reports VPD in hPa
_KPA_PER_HPA = 0.1


def _hot_dry_windy_layer_max(
    temperature: npt.NDArray[np.float64],
    humidity: npt.NDArray[np.float64],
    wind: npt.NDArray[np.float64],
    height: npt.NDArray[np.float64],
    axis: int,
    *,
    warn: bool = True,
) -> npt.NDArray[np.float64]:
    """Maximum per-level HDW product over ``axis`` for already-broadcast profiles.

    Shared by :func:`hot_dry_windy`'s NumPy path and its xarray kernel. The
    xarray path disables ``warn`` for Dask blocks, where one warning per block
    would swamp the operation-level signal; NaN results still mark the invalid
    columns.

    Args:
        temperature: Air temperature profile, degrees Celsius.
        humidity: Relative humidity profile, percent.
        wind: Wind speed profile, meters per second.
        height: Height above ground level, meters.
        axis: Axis of the vertical coordinate, reduced by the maximum.
        warn: Emit the invalid-value and empty-column warnings.

    Returns:
        HDW in hPa m s-1, with the vertical axis removed.
    """
    # outside the physical range the formulas still return numbers, but
    # meaningless ones, so treat such values as missing
    in_layer = (height >= 0.0) & (height <= _HDW_LAYER_TOP_METERS)
    invalid = (humidity < 0.0) | (humidity > 100.0) | (wind < 0.0)
    if warn:
        invalid_count = int(np.count_nonzero(invalid & in_layer))
        if invalid_count > 0:
            _logger.warning(
                f"Found {invalid_count} values with relative humidity outside [0, 100] "
                "or negative wind speed; HDW is NaN in those columns."
            )
        empty_columns = int(np.count_nonzero(~np.any(in_layer, axis=axis)))
        if empty_columns > 0:
            _logger.warning(
                f"Found {empty_columns} columns with no level in the lowest "
                f"{_HDW_LAYER_TOP_METERS:.0f} m AGL; HDW is NaN there."
            )

    saturation_hpa = pm_eto.saturation_vapor_pressure(temperature) / _KPA_PER_HPA
    vpd_hpa = saturation_hpa * (1.0 - humidity / 100.0)

    # NaN in-layer propagates through the maximum; out-of-layer levels are
    # excluded via -inf, which any real product beats
    value = np.where(invalid, np.nan, vpd_hpa * wind)
    product = np.where(in_layer, value, -np.inf)
    if product.shape[axis] == 0:
        index = np.full(product.shape[:axis] + product.shape[axis + 1 :], np.nan)
    else:
        index = np.max(product, axis=axis)
    result: npt.NDArray[np.float64] = np.where(np.any(in_layer, axis=axis), index, np.nan).astype(
        np.float64, copy=False
    )
    return result


def _hdw_xarray(
    temperature_celsius: xr.DataArray,
    relative_humidity_percent: xr.DataArray,
    wind_speed_meters_per_second: xr.DataArray,
    height_agl_meters: npt.ArrayLike | xr.DataArray,
    *,
    level_axis: int,
    level_dim: str,
) -> xr.DataArray:
    """xarray dispatch for :func:`hot_dry_windy`. See :func:`hot_dry_windy` for the full contract.

    HDW is weather-only and stateless: every dimension besides ``level_dim``
    passes straight through, unlike KBDI's per-call CF-registry resolution or
    CFFWIS's shared recurrence. :func:`xarray.apply_ufunc` calls
    :func:`_hot_dry_windy_layer_max` once per Dask block, with ``level_dim`` as
    the sole core dimension and reduced away -- the same shape
    ``test_hdw_chunked_time_and_space_match_eager`` already exercises for the
    NumPy core, here wrapped with validation and CF metadata. The shared
    kernel rather than the public function keeps one xarray operation from
    emitting per-block calculation events, memory instrumentation, and
    invalid-value warnings; block exceptions still propagate from ``compute``.
    """
    if level_axis != -1:
        raise InvalidArgumentError(
            "level_axis is not used for xr.DataArray input; the vertical dimension is named by level_dim.",
            argument_name="level_axis",
            argument_value=str(level_axis),
            valid_values="-1 (the default) when temperature_celsius is an xr.DataArray",
        )

    temperature = _convert_temperature_units(temperature_celsius, "celsius")
    humidity = relative_humidity_percent
    wind = wind_speed_meters_per_second
    if isinstance(height_agl_meters, xr.DataArray):
        height = height_agl_meters
    else:
        height_array = np.asarray(height_agl_meters, dtype=np.float64)
        if height_array.ndim != 1:
            raise InvalidArgumentError(
                "height_agl_meters must be an xr.DataArray or a 1-D array-like when "
                "temperature_celsius is an xr.DataArray.",
                argument_name="height_agl_meters",
                argument_value=f"array-like with ndim={height_array.ndim}",
                valid_values="An xr.DataArray, or a 1-D array-like naming level_dim's levels",
            )
        height = xr.DataArray(height_array, dims=(level_dim,))

    for name, data in (
        ("temperature_celsius", temperature),
        ("relative_humidity_percent", humidity),
        ("wind_speed_meters_per_second", wind),
        ("height_agl_meters", height),
    ):
        if level_dim not in data.dims:
            raise CoordinateValidationError(
                message=(
                    f"Dimension '{level_dim}' not found in {name}. "
                    f"Available dimensions: {list(data.dims)}. Use level_dim to specify a custom name."
                ),
                coordinate_name=level_dim,
                reason="missing_dimension",
            )
        _validate_dask_chunks(data, level_dim)

    # one warning per invalid Dask block would swamp the logs; the eager path
    # still reports invalid columns through the shared kernel's warnings
    is_dask_backed = any(data.chunks is not None for data in (temperature, humidity, wind, height))
    result: xr.DataArray = xr.apply_ufunc(
        _hot_dry_windy_layer_max,
        temperature,
        humidity,
        wind,
        height,
        input_core_dims=[[level_dim]] * 4,
        output_core_dims=[[]],
        kwargs={"axis": -1, "warn": not is_dask_backed},
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    # apply_ufunc orders output dims by first occurrence across all four inputs
    # in argument order, not just temperature's: height (or humidity/wind) may
    # carry a dimension temperature lacks, so the transpose target must be
    # built the same way, not read off temperature.dims alone.
    output_dims: list[str] = []
    for data in (temperature, humidity, wind, height):
        for dim in data.dims:
            dim = str(dim)
            if dim != level_dim and dim not in output_dims:
                output_dims.append(dim)
    result = result.transpose(*output_dims)
    result.attrs = _build_output_attrs(
        temperature_celsius,
        cf_metadata=CF_METADATA["hdw"],  # type: ignore[arg-type]
        index_name="HDW",
    )
    return result


@overload
def hot_dry_windy(
    temperature_celsius: xr.DataArray,
    relative_humidity_percent: xr.DataArray,
    wind_speed_meters_per_second: xr.DataArray,
    height_agl_meters: npt.ArrayLike | xr.DataArray,
    *,
    level_axis: int = -1,
    level_dim: str = "level",
) -> xr.DataArray: ...


@overload
def hot_dry_windy(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
    height_agl_meters: npt.ArrayLike,
    *,
    level_axis: int = -1,
    level_dim: str = "level",
) -> npt.NDArray[np.float64]: ...


def hot_dry_windy(
    temperature_celsius: npt.ArrayLike | xr.DataArray,
    relative_humidity_percent: npt.ArrayLike | xr.DataArray,
    wind_speed_meters_per_second: npt.ArrayLike | xr.DataArray,
    height_agl_meters: npt.ArrayLike | xr.DataArray,
    *,
    level_axis: int = -1,
    level_dim: str = "level",
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute the Hot-Dry-Windy Index (HDW).

    This function accepts both NumPy arrays and xarray DataArrays. Type
    checkers narrow the return type based on the input type.

    .. warning:: **Beta Feature (xarray path only)** -- When called with
       ``xr.DataArray`` input, this function uses the beta xarray adapter
       layer: CF metadata from the ``hdw`` registry entry, CF
       ``units``-attribute temperature inference, and Dask parallelism over
       every dimension except ``level_dim``, which must be a single chunk.
       The NumPy array interface and underlying computation are stable.

    A weather-only index of dangerous fire-behavior potential (Srock et al.,
    2018): the vapor pressure deficit (VPD) times the wind speed, maximized
    over the levels in the lowest 500 m above ground level (AGL)::

        HDW = max over levels with 0 <= height_agl <= 500 of (VPD * wind speed)

    Inputs are vertical profiles with SI units, like the rest of the package.
    The four inputs broadcast against each other; the vertical coordinate
    (``level_axis`` for NumPy input, ``level_dim`` for xarray input) is
    reduced by the maximum. VPD comes from each level's own temperature and
    relative humidity, with saturation vapor pressure from
    ``pm_eto.saturation_vapor_pressure`` (FAO-56 Eq 11), converted from kPa to
    the hPa of the published index.

    ``height_agl_meters`` is the vertical coordinate itself, so the AGL
    determination happens where that coordinate is built:

    - Model-level input (e.g. CFSR): use the model's own height field, or
      geopotential height minus the surface geopotential height of the grid
      cell.
    - Pressure-level input: convert each pressure level to geopotential height
      (hypsometric equation) and subtract the surface height of the grid cell.

    The result is sensitive to vertical resolution: coarse level spacing can
    miss the level where VPD and wind combine worst, and sampling more levels
    inside the layer can only raise the maximum. Compare HDW across datasets
    only at comparable vertical resolution. Srock et al. (2018) additionally
    adiabatically adjust each level's VPD to the surface and take the VPD and
    wind maxima independently (so they may come from different levels); this
    implementation follows the simplified formulation of the issue contract,
    the per-level product, which never exceeds the published variant.

    Args:
        temperature_celsius: Air temperature profile, degrees Celsius.
        relative_humidity_percent: Relative humidity profile, percent, in
            [0, 100].
        wind_speed_meters_per_second: Wind speed profile, meters per second,
            non-negative.
        height_agl_meters: Height above ground level of each level, meters.
            Levels outside [0, 500], or with NaN height, are excluded from the
            maximum. For xarray input, an ``xr.DataArray`` (1-D on
            ``level_dim`` or full N-D) or a 1-D array-like naming
            ``level_dim``'s levels.
        level_axis: NumPy input only. Axis of the broadcast inputs that holds
            the vertical coordinate. Reduced by the layer maximum.
        level_dim: xarray input only. Name of the vertical dimension. Reduced
            by the layer maximum; not inferred.

    Returns:
        HDW in hPa m s-1, with the broadcast shape of the inputs minus the
        vertical coordinate. NaN where any in-layer level has NaN or
        out-of-range input, and for columns with no level inside the lowest
        500 m AGL. For xarray input, a ``DataArray`` carrying CF metadata from
        the ``hdw`` registry entry.

    Raises:
        TypeError: If ``temperature_celsius``, ``relative_humidity_percent``,
            and ``wind_speed_meters_per_second`` are not all the same type.
        InvalidArgumentError: If the inputs cannot be broadcast together,
            ``level_axis`` is out of range for the broadcast shape (NumPy
            input), ``level_axis`` is not the default alongside xarray
            input, or ``height_agl_meters`` is not an ``xr.DataArray`` or a
            1-D array-like when the other inputs are ``xr.DataArray``.
        CoordinateValidationError: xarray input only -- if ``level_dim`` is
            missing from any input, or the input is Dask-backed with
            ``level_dim`` split across multiple chunks.

    Notes:
        xarray-only: ``temperature_celsius``, ``relative_humidity_percent``,
        and ``wind_speed_meters_per_second`` must be the same type (all NumPy
        or all ``xr.DataArray``); ``height_agl_meters`` may be either. A CF
        ``units`` attribute on temperature is converted to Celsius; an absent
        attribute is assumed to already be Celsius. HDW has no time semantics
        -- it carries no state and is not a daily recurrence -- so every
        dimension other than ``level_dim`` (including ``time``, if present)
        is a plain passthrough with no cadence requirement. Dimensions shared
        by the inputs are matched exactly (xarray's default exact join): the
        inputs are never aligned or reindexed, so unequal or differently
        ordered coordinate labels raise instead of broadcasting. The invalid
        humidity/wind and empty-column warnings are emitted once per operation
        for eager input; Dask-backed input suppresses them per block and
        reports those columns as NaN.

    Example:
        >>> from climate_indices import fire
        >>> round(float(fire.hot_dry_windy([30.0, 26.0], [15.0, 30.0], [8.0, 12.0], [10.0, 400.0])), 2)
        288.53
    """
    is_xarray = isinstance(temperature_celsius, xr.DataArray)
    for name, value in (
        ("relative_humidity_percent", relative_humidity_percent),
        ("wind_speed_meters_per_second", wind_speed_meters_per_second),
    ):
        if isinstance(value, xr.DataArray) != is_xarray:
            raise TypeError(
                "temperature_celsius, relative_humidity_percent, and wind_speed_meters_per_second must be "
                f"the same type. Got temperature_celsius={type(temperature_celsius).__name__}, "
                f"{name}={type(value).__name__}. "
                "Convert both to the same type (both numpy arrays or both xr.DataArray)."
            )
    if detect_input_type(temperature_celsius) != InputType.NUMPY:
        assert isinstance(temperature_celsius, xr.DataArray)
        assert isinstance(relative_humidity_percent, xr.DataArray)
        assert isinstance(wind_speed_meters_per_second, xr.DataArray)
        return _hdw_xarray(
            temperature_celsius,
            relative_humidity_percent,
            wind_speed_meters_per_second,
            height_agl_meters,
            level_axis=level_axis,
            level_dim=level_dim,
        )

    temperature = _as_float_array(temperature_celsius)
    humidity = _as_float_array(relative_humidity_percent)
    wind = _as_float_array(wind_speed_meters_per_second)
    height = _as_float_array(height_agl_meters)

    broadcast_ndim = max(array.ndim for array in (temperature, humidity, wind, height))
    if height.ndim == 1 and broadcast_ndim > 1 and -broadcast_ndim <= level_axis < broadcast_ndim:
        axis = level_axis % broadcast_ndim
        height = height.reshape((1,) * axis + height.shape + (1,) * (broadcast_ndim - axis - 1))

    try:
        temperature, humidity, wind, height = np.broadcast_arrays(temperature, humidity, wind, height)
    except ValueError as exc:
        message = (
            "Incompatible array shapes for Hot-Dry-Windy Index: "
            f"temperature={temperature.shape}, relative_humidity={humidity.shape}, "
            f"wind_speed={wind.shape}, height_agl={height.shape}. The inputs must broadcast together."
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="temperature_celsius/relative_humidity_percent/wind_speed_meters_per_second/height_agl_meters",
            argument_value=f"shapes {temperature.shape}, {humidity.shape}, {wind.shape}, {height.shape}",
            valid_values="Arrays broadcastable to a common shape",
        ) from exc

    if temperature.ndim == 0:
        # a single level is a degenerate profile; give it an axis to reduce
        temperature, humidity, wind, height = (a.reshape(1) for a in (temperature, humidity, wind, height))

    if not -temperature.ndim <= level_axis < temperature.ndim:
        message = (
            f"level_axis {level_axis} is out of range for the broadcast input shape "
            f"{temperature.shape} with {temperature.ndim} dimensions."
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="level_axis",
            argument_value=str(level_axis),
            valid_values=f"An axis of the broadcast shape {temperature.shape}",
        )
    axis = level_axis % temperature.ndim

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="hot_dry_windy",
        input_shape=temperature.shape,
        input_elements=temperature.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(temperature, humidity, wind, height)

    try:
        result = _hot_dry_windy_layer_max(temperature, humidity, wind, height, axis)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=result.shape,
            **(memory_metrics or {}),
        )
        return result
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise

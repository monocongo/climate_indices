"""xarray adapter for the Canadian Forest Fire Weather Index System (#807).

One :func:`xarray.apply_ufunc` call runs the shared NumPy orchestrator once
per Dask spatial block with the full ``time`` axis, then each selected output
is rewrapped with its own ``CF_METADATA`` entry. The adapter infers ``month``
from the time coordinate and latitude from a ``lat``/``latitude`` coordinate,
converts CF ``units`` attributes on temperature and precipitation, and warns
when a daily time coordinate is clearly not noon-referenced.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices._recurrence import _as_float_array
from climate_indices._units import (
    _convert_precipitation_units,
    _convert_temperature_units,
    _validate_daily_time_coordinate,
)
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import (
    ClimateIndicesWarning,
    CoordinateValidationError,
    InputAlignmentWarning,
    InvalidArgumentError,
    wrap_value_error,
)
from climate_indices.fire import _cffwis
from climate_indices.fire._cffwis import (
    CFFWISResult,
    CFFWISState,
    DCState,
    DMCState,
    FFMCState,
    _month_array,
    _validate_latitude_extent,
)
from climate_indices.fire._cffwis_codes import (
    _CFFWIS_COMPONENTS,
    DC_CODE,
    DMC_CODE,
    FFMC_CODE,
    _CFFWISComponent,
)
from climate_indices.validation import validate_dask_chunks, validate_time_dimension, validate_time_monotonicity
from climate_indices.xarray_adapter import _wrap_spatial, build_output_attrs

# ---------------------------------------------------------------------------
# xarray adapter (#807)


def _trailing_gap_days_from_block(gaps: npt.ArrayLike) -> npt.NDArray[np.int64] | None:
    """Reconstruct a state's ``trailing_gap_days`` from an apply_ufunc block.

    The block carries -1 where no gap has started, which is the same encoding
    the state uses internally, so an all-negative block is the ``None`` case.
    """
    array = np.asarray(gaps, dtype=np.int64)
    return None if np.all(array < 0) else array.copy()


def _warn_if_not_noon_referenced(weather_inputs: tuple[xr.DataArray, ...], time_dim: str) -> None:
    """Warn when a daily time coordinate clearly does not sample noon local standard time.

    CFFWIS is defined on noon observations of temperature, humidity, and wind.
    The coordinate's time of day is the only signal available here: a daily
    coordinate at another hour (00:00 daily summaries are common) cannot be
    silently accepted as noon-referenced, so the caller is warned rather than
    failing a computation whose values may still be acceptable.
    """
    time_coord: xr.DataArray | None = None
    for data in weather_inputs:
        if time_dim in data.coords:
            time_coord = data.coords[time_dim]
            break
    if time_coord is None or time_coord.size == 0:
        return
    try:
        times = time_coord.values.astype("datetime64[ns]")
    except (TypeError, ValueError):
        # cftime or otherwise non-datetime coordinates carry no hour to inspect
        return
    hours = np.unique((times - times.astype("datetime64[D]")) / np.timedelta64(1, "h"))
    # exact-noon check, as in the CF convention: np.equal keeps the deliberate
    # equality out of the float-comparison lint rule
    if hours.size == 1 and np.equal(float(hours[0]), 12.0):
        return
    warnings.warn(
        f"CFFWIS requires noon local-standard-time temperature, humidity, and wind, but the "
        f"'{time_dim}' coordinate samples at hour(s) {[float(hour) for hour in hours]} rather than 12:00. "
        "Verify the weather inputs are noon observations; values referenced to another hour shift the "
        "moisture codes' diurnal drying.",
        ClimateIndicesWarning,
        stacklevel=3,
    )


def _broadcast_topology(weather: tuple[xr.DataArray, ...]) -> tuple[tuple[str, ...], dict[str, int]]:
    """Return the dims and sizes ``xr.broadcast`` would produce, without broadcasting data.

    The broadcast dims come in order of appearance across the inputs, exactly
    as ``xarray.core.variable._unified_dims`` orders them; ``xr.align`` has
    already matched the sizes of the shared dims.
    """
    dims: list[str] = []
    sizes: dict[str, int] = {}
    for data in weather:
        for dim in data.dims:
            name = str(dim)
            if name not in sizes:
                dims.append(name)
                sizes[name] = data.sizes[dim]
    return tuple(dims), sizes


def _spatial_chunk_targets(
    weather: tuple[xr.DataArray, ...], spatial_dims: tuple[str, ...]
) -> dict[str, tuple[int, ...]]:
    """The finest spatial chunking the Dask-backed weather inputs carry.

    Static spatial operands (seeds, resumed state) are partitioned to this
    chunking so ``apply_ufunc`` hands a worker only its own tile instead of the
    whole grid. Empty when no weather input is Dask-backed.
    """
    targets: dict[str, tuple[int, ...]] = {}
    for dim in spatial_dims:
        chunkings = [
            data.chunks[data.dims.index(dim)] for data in weather if data.chunks is not None and dim in data.dims
        ]
        if chunkings:
            targets[dim] = min(chunkings, key=len)
    return targets


def _align_cffwis_inputs(
    temperature_celsius: xr.DataArray,
    relative_humidity_percent: xr.DataArray,
    wind_speed_meters_per_second: xr.DataArray,
    precipitation_mm: xr.DataArray,
    time_dim: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Align the four weather inputs, protecting spatial coverage and reporting time drops.

    Follows the KBDI adapter's inner-join contract: a shared spatial dimension
    that loses coordinates is an error, while a shortened time axis is the
    documented intersection and only warns. The four inputs are aligned in one
    call so a partial mismatch cannot pair one input's grid with another's.
    """
    inputs = (temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, precipitation_mm)
    names = (
        "temperature_celsius",
        "relative_humidity_percent",
        "wind_speed_meters_per_second",
        "precipitation_mm",
    )
    shared_spatial_dims = sorted(
        {
            str(dim)
            for first in range(len(inputs))
            for second in range(first + 1, len(inputs))
            for dim in inputs[first].dims
            if dim in inputs[second].dims and dim != time_dim
        }
    )
    try:
        aligned = xr.align(*inputs, join="inner")
    except xr.AlignmentError as exc:
        raise CoordinateValidationError(
            message=(
                "Cannot align the CFFWIS weather inputs: their dimension sizes or coordinate labels "
                f"conflict ({exc}). Give the inputs matching spatial shapes and a shared time axis."
            ),
            coordinate_name=time_dim,
            reason="alignment_conflict",
        ) from exc
    for dim in shared_spatial_dims:
        original_sizes = {name: data.sizes[dim] for name, data in zip(names, inputs, strict=True) if dim in data.sizes}
        aligned_sizes = {name: data.sizes[dim] for name, data in zip(names, aligned, strict=True) if dim in data.sizes}
        if any(aligned_sizes[name] != size for name, size in original_sizes.items()):
            raise CoordinateValidationError(
                message=(
                    f"Input alignment dropped coordinates along non-time dimension '{dim}': "
                    + ", ".join(f"{name} had {size}" for name, size in original_sizes.items())
                    + "; after the inner join they have "
                    + ", ".join(f"{name} {size}" for name, size in aligned_sizes.items())
                    + ". Subset or align the inputs explicitly; CFFWIS never reduces spatial coverage silently."
                ),
                coordinate_name=dim,
                reason="non_time_alignment_dropped_coordinates",
            )
    time_sizes = {name: data.sizes[time_dim] for name, data in zip(names, inputs, strict=True)}
    aligned_length = aligned[0].sizes[time_dim]
    if aligned_length == 0:
        raise CoordinateValidationError(
            message=(
                f"No overlapping timesteps found across the CFFWIS weather inputs along '{time_dim}'. "
                "Cannot compute the Canadian Forest Fire Weather Index System."
            ),
            coordinate_name=time_dim,
            reason="empty_intersection_after_alignment",
        )
    original_length = max(time_sizes.values())
    if aligned_length < original_length:
        warnings.warn(
            InputAlignmentWarning(
                message=(
                    "Input alignment: "
                    + ", ".join(f"{name} had {size} timesteps" for name, size in time_sizes.items())
                    + f". After inner join, {aligned_length} remain."
                ),
                original_size=original_length,
                aligned_size=aligned_length,
                dropped_count=original_length - aligned_length,
            ),
            stacklevel=3,
        )
    return aligned[0], aligned[1], aligned[2], aligned[3]


_LATITUDE_DEGREE_UNITS = frozenset({"degrees_north", "degree_north", "degrees", "degree", "deg"})


def _validate_latitude_units(latitude: xr.DataArray) -> None:
    """Reject a latitude whose CF ``units`` attribute is not degrees north.

    A radians-valued coordinate consumed as degrees silently moves every cell
    into the wrong day-length band, so an unrecognized attribute fails the
    call rather than being ignored.
    """
    raw_units = latitude.attrs.get("units")
    if raw_units is None:
        return
    normalized = raw_units.strip().lower() if isinstance(raw_units, str) else None
    if normalized in _LATITUDE_DEGREE_UNITS or normalized == "":
        return
    raise InvalidArgumentError(
        f"Unsupported latitude units attribute: {raw_units!r}.",
        argument_name="latitude_degrees_north.attrs['units']",
        argument_value=repr(raw_units),
        valid_values="degrees_north, degrees, or no units attribute",
    )


def _infer_latitude_coordinate(weather_inputs: tuple[xr.DataArray, ...]) -> xr.DataArray:
    """The weather inputs' shared 'lat'/'latitude' coordinate.

    A raw scalar/array latitude follows the NumPy core's trailing-axis
    broadcast; a coordinate must be a DataArray so its named axis survives.
    """
    found: list[xr.DataArray] = []
    for data in weather_inputs:
        for coordinate_name in ("lat", "latitude"):
            if coordinate_name in data.coords:
                found.append(data.coords[coordinate_name])
                break
    if not found:
        raise InvalidArgumentError(
            "latitude_degrees_north is required when the weather inputs carry no 'lat' or 'latitude' coordinate.",
            argument_name="latitude_degrees_north",
            argument_value="None and no latitude coordinate",
            valid_values="An explicit latitude, or a 'lat'/'latitude' coordinate on the inputs",
        )
    latitude = found[0]
    for other in found[1:]:
        if not latitude.equals(other):
            raise InvalidArgumentError(
                "The weather inputs carry conflicting 'lat'/'latitude' coordinates, so the "
                "per-cell day-length bands would differ per input. Supply latitude_degrees_north explicitly.",
                argument_name="latitude_degrees_north",
                argument_value="conflicting coordinates",
                valid_values="One consistent latitude coordinate, or an explicit latitude_degrees_north",
            )
    return latitude


def _latitude_from_argument(
    latitude_degrees_north: npt.ArrayLike,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
) -> xr.DataArray:
    """Broadcast a raw scalar/array latitude onto the weather inputs' spatial grid."""
    array = _as_float_array(latitude_degrees_north)
    _validate_latitude_extent(array)
    try:
        broadcast = np.broadcast_to(array, spatial_shape)
    except ValueError as exc:
        wrap_value_error(
            exc,
            message=(
                "latitude_degrees_north must broadcast to the weather inputs' spatial shape; pass a "
                "DataArray to place a latitude on a named axis that does not align with the trailing "
                "dimensions."
            ),
            argument_name="latitude_degrees_north",
            argument_value=f"shape {array.shape}",
            valid_values=f"A scalar or an array broadcastable to the spatial shape {spatial_shape}",
        )
    return xr.DataArray(broadcast, dims=spatial_dims)


def _validate_latitude_dims(
    latitude: xr.DataArray,
    time_dim: str,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
) -> None:
    """Reject a latitude that varies in time, carries foreign dims, or mismatches extents."""
    if time_dim in latitude.dims:
        raise InvalidArgumentError(
            "latitude_degrees_north must not vary in time: each cell's day-length band is static.",
            argument_name="latitude_degrees_north",
            argument_value=f"dims {tuple(str(dim) for dim in latitude.dims)}",
            valid_values="A scalar or a spatial field with no time dimension",
        )
    unsupported_dims = [str(dim) for dim in latitude.dims if str(dim) not in spatial_dims]
    if unsupported_dims:
        raise InvalidArgumentError(
            "latitude_degrees_north carries dimensions the weather inputs do not have, so it would add "
            "an unsupported axis to the result. A latitude must be a scalar or a spatial field on the "
            "weather grid.",
            argument_name="latitude_degrees_north",
            argument_value=f"dims {tuple(str(dim) for dim in latitude.dims)}",
            valid_values=f"A scalar, or a field on the weather spatial dims {spatial_dims}",
        )
    mismatched_sizes = {
        str(dim): (latitude.sizes[dim], spatial_shape[spatial_dims.index(str(dim))])
        for dim in latitude.dims
        if latitude.sizes[dim] != spatial_shape[spatial_dims.index(str(dim))]
    }
    if mismatched_sizes:
        raise InvalidArgumentError(
            "latitude_degrees_north does not match the weather inputs' spatial extents: "
            + ", ".join(
                f"'{dim}' is {latitude_size} on the latitude and {weather_size} on the weather"
                for dim, (latitude_size, weather_size) in mismatched_sizes.items()
            )
            + ".",
            argument_name="latitude_degrees_north",
            argument_value=f"sizes {dict(latitude.sizes)}",
            valid_values=f"A scalar, or a field with the weather spatial shape {spatial_shape}",
        )


def _validate_latitude_labels(latitude: xr.DataArray, weather_inputs: tuple[xr.DataArray, ...]) -> None:
    """Require a latitude's indexed dimensions to carry the weather inputs' labels."""
    for dim in latitude.dims:
        dim_name = str(dim)
        if dim_name not in latitude.indexes:
            continue
        for data in weather_inputs:
            if dim_name in data.indexes:
                if not latitude.indexes[dim_name].equals(data.indexes[dim_name]):
                    raise CoordinateValidationError(
                        message=(
                            f"latitude_degrees_north's '{dim_name}' coordinate labels do not match the "
                            "weather inputs'. Reindex the latitude to the weather grid's labels before calling."
                        ),
                        coordinate_name=dim_name,
                        reason="latitude_coordinates_not_aligned",
                    )
                break


def _resolve_cffwis_latitude(
    latitude_degrees_north: npt.ArrayLike | xr.DataArray | None,
    weather_inputs: tuple[xr.DataArray, ...],
    time_dim: str,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
) -> xr.DataArray:
    """Resolve the adapter's latitude from an explicit value or a coordinate.

    A raw scalar/array latitude follows the NumPy core's trailing-axis
    broadcast; a leading-dimension latitude coordinate must be passed as a
    DataArray (or inferred from the inputs), since an ambiguous 1-D array
    cannot be placed on a named axis without guessing.
    """
    if latitude_degrees_north is None:
        latitude = _infer_latitude_coordinate(weather_inputs)
    elif isinstance(latitude_degrees_north, xr.DataArray):
        latitude = latitude_degrees_north
    else:
        latitude = _latitude_from_argument(latitude_degrees_north, spatial_dims, spatial_shape)
    _validate_latitude_dims(latitude, time_dim, spatial_dims, spatial_shape)
    _validate_latitude_units(latitude)
    _validate_latitude_labels(latitude, weather_inputs)
    if latitude.chunks is None:
        # eager validation: a bad latitude must fail this call, not a later
        # lazy evaluation (the NumPy core revalidates per block)
        _validate_latitude_extent(_as_float_array(latitude.values))
    return latitude


def _infer_month_values(time_coord: xr.DataArray | None, time_dim: str) -> np.ndarray:
    """Infer the calendar month series from the weather inputs' datetime coordinate."""
    if time_coord is None:
        raise InvalidArgumentError(
            f"month is required when the weather inputs carry no '{time_dim}' coordinate to infer it from.",
            argument_name="month",
            argument_value="None and no time coordinate",
            valid_values=f"An explicit month, or a datetime '{time_dim}' coordinate",
        )
    try:
        inferred = time_coord.dt.month
    except (AttributeError, TypeError) as exc:
        wrap_value_error(
            exc,
            message=(
                f"month cannot be inferred from the non-datetime '{time_dim}' coordinate. Supply month explicitly."
            ),
            argument_name="month",
            argument_value="non-datetime time coordinate",
            valid_values=f"An explicit month, or a datetime '{time_dim}' coordinate",
        )
    return np.asarray(inferred)


def _align_month_to_weather_time(month: xr.DataArray, time_coord: xr.DataArray | None, time_dim: str) -> xr.DataArray:
    """Relabel a labelled month series onto the weather inputs' aligned time coordinate."""
    if time_coord is None:
        raise InvalidArgumentError(
            f"month carries a '{time_dim}' coordinate but the weather inputs do not; drop the "
            "coordinate or give the weather inputs matching time coordinates.",
            argument_name="month",
            argument_value="a time coordinate the weather inputs do not carry",
            valid_values="Months without a time coordinate, or weather inputs carrying the same one",
        )
    # a labelled month series keeps its dates: relabel it onto the
    # aligned weather axis so a reordered series cannot be paired
    # positionally
    try:
        # mypy loses the isinstance narrowing across xarray's Self-returning reindex
        month = cast(xr.DataArray, month.reindex({time_dim: time_coord}))
    except (KeyError, TypeError, ValueError) as exc:
        wrap_value_error(
            exc,
            message=(
                f"month's '{time_dim}' coordinate cannot be matched to the weather inputs' "
                "coordinate. Align the two, or pass month as a plain array to pair it positionally."
            ),
            argument_name="month",
            argument_value="unmatched time coordinate",
            valid_values=f"Months carrying the weather '{time_dim}' coordinates, or a plain array",
        )
    if bool(month.isnull().any()):
        raise InvalidArgumentError(
            f"month has no value at every aligned weather timestep along '{time_dim}'; its "
            "time coordinate must cover the weather inputs' aligned timesteps.",
            argument_name="month",
            argument_value="time coordinate gaps",
            valid_values=f"Months covering every aligned '{time_dim}' step",
        )
    return month


def _month_dataarray_values(
    month: xr.DataArray, time_coord: xr.DataArray | None, time_dim: str, time_length: int
) -> np.ndarray | xr.DataArray:
    """Validate a month DataArray, returning a lazy month unchanged or its plain values."""
    extra_dims = [str(dim) for dim in month.dims if dim != time_dim]
    if extra_dims:
        raise InvalidArgumentError(
            "month must be scalar or carry only the time dimension for xarray input; a spatial "
            "month field is available through the NumPy API.",
            argument_name="month",
            argument_value=f"dims {tuple(str(dim) for dim in month.dims)}",
            valid_values=f"A scalar, a 1-D time series, or a DataArray on '{time_dim}'",
        )
    if time_dim in month.coords:
        month = _align_month_to_weather_time(month, time_coord, time_dim)
    if month.chunks is not None and time_dim in month.dims:
        # a lazily-backed month stays lazy: apply_ufunc requires a single
        # core-dim chunk, and the rechunk rewrites the lazy graph without
        # computing it
        month = cast(xr.DataArray, month.chunk({time_dim: -1}))
        if month.sizes[time_dim] != time_length:
            raise InvalidArgumentError(
                f"month has {month.sizes[time_dim]} entries but the time dimension has {time_length} steps.",
                argument_name="month",
                argument_value=f"shape {dict(month.sizes)}",
                valid_values=f"A scalar or a 1-D array of {time_length} calendar months",
            )
        return month
    return np.asarray(month)


def _finalize_month(values: np.ndarray, time_length: int, time_dim: str) -> xr.DataArray:
    """Validate a plain month array and wrap it on the named time dimension."""
    if values.ndim > 1:
        raise InvalidArgumentError(
            "month must be scalar or 1-D for xarray input; a spatial month field is available through the NumPy API.",
            argument_name="month",
            argument_value=f"shape {values.shape}",
            valid_values=f"A scalar or a 1-D array of {time_length} calendar months",
        )
    if values.ndim == 1 and values.size != time_length:
        raise InvalidArgumentError(
            f"month has {values.size} entries but the time dimension has {time_length} steps.",
            argument_name="month",
            argument_value=f"shape {values.shape}",
            valid_values=f"A scalar or a 1-D array of {time_length} calendar months",
        )
    validated = _month_array(values, (time_length,))
    return xr.DataArray(np.asarray(validated, dtype=np.int64), dims=(time_dim,))


def _resolve_cffwis_month(
    month: npt.ArrayLike | xr.DataArray | None,
    time_coord: xr.DataArray | None,
    time_dim: str,
    time_length: int,
) -> xr.DataArray:
    """Resolve the month series from the call's argument or the weather time coordinate."""
    if month is None:
        values = _infer_month_values(time_coord, time_dim)
    elif isinstance(month, xr.DataArray):
        prepared = _month_dataarray_values(month, time_coord, time_dim, time_length)
        if isinstance(prepared, xr.DataArray):
            return prepared
        values = prepared
    else:
        values = np.asarray(month)
    return _finalize_month(values, time_length, time_dim)


@dataclass(frozen=True)
class _CFFWISCallOptions:
    """The keyword-only options one :func:`cffwis` call threads through the adapter."""

    initial_ffmc: npt.ArrayLike | None
    initial_dmc: npt.ArrayLike | None
    initial_dc: npt.ArrayLike | None
    initial_state: CFFWISState | None
    return_state: bool
    spin_up: int
    nan_policy: Literal["propagate", "bridge"]
    max_gap_days: int
    selected: frozenset[_CFFWISComponent]
    time_dim: str


def _validate_cffwis_xarray_inputs(weather_inputs: tuple[xr.DataArray, ...], time_dim: str) -> None:
    """Validate each weather input's time axis: known dimension, monotonic, daily."""
    for data in weather_inputs:
        validate_time_dimension(data, time_dim)
        if time_dim in data.coords:
            validate_time_monotonicity(data.coords[time_dim])
            _validate_daily_time_coordinate(data, time_dim)


def _selected_component_names(selected: frozenset[_CFFWISComponent]) -> tuple[str, ...]:
    """The requested output names in the component registry's order."""
    return tuple(name for name in _CFFWIS_COMPONENTS if name in selected)


def _cffwis_optional_kinds(options: _CFFWISCallOptions) -> tuple[str, ...]:
    """The optional apply_ufunc inputs this call supplies, in argument order."""
    if options.initial_state is not None:
        return ("ffmc_value", "ffmc_gap", "dmc_value", "dmc_gap", "dc_value", "dc_gap")
    kinds: list[str] = []
    for kind, seed in (
        ("initial_ffmc", options.initial_ffmc),
        ("initial_dmc", options.initial_dmc),
        ("initial_dc", options.initial_dc),
    ):
        if seed is not None:
            kinds.append(kind)
    return tuple(kinds)


def _gap_or_full(gaps: npt.ArrayLike | None, spatial_shape: tuple[int, ...]) -> npt.ArrayLike:
    """A state's trailing-gap array, or the -1-filled 'no gap' array when absent."""
    if gaps is None:
        return np.full(spatial_shape, -1, dtype=np.int64)
    return gaps


def _cffwis_optional_args(
    options: _CFFWISCallOptions,
    spatial_shape: tuple[int, ...],
    spatial_dims: tuple[str, ...],
    spatial_chunks: dict[str, tuple[int, ...]],
) -> list[xr.DataArray]:
    """Wrap the seed/state operands for apply_ufunc, one per optional input kind."""

    def wrap(value: npt.ArrayLike | xr.DataArray) -> xr.DataArray:
        return _wrap_spatial(value, spatial_shape, spatial_dims, chunks=spatial_chunks)

    state = options.initial_state
    if state is None:
        seeds = (options.initial_ffmc, options.initial_dmc, options.initial_dc)
        return [wrap(seed) for seed in seeds if seed is not None]
    return [
        wrap(state.ffmc.ffmc),
        wrap(_gap_or_full(state.ffmc.trailing_gap_days, spatial_shape)),
        wrap(state.dmc.dmc),
        wrap(_gap_or_full(state.dmc.trailing_gap_days, spatial_shape)),
        wrap(state.dc.dc),
        wrap(_gap_or_full(state.dc.trailing_gap_days, spatial_shape)),
    ]


def _validate_cffwis_seeds(options: _CFFWISCallOptions, spatial_shape: tuple[int, ...]) -> None:
    """Validate seeds or a supplied state eagerly, before any lazy evaluation."""
    state = options.initial_state
    for code, seed in (
        (FFMC_CODE, options.initial_ffmc),
        (DMC_CODE, options.initial_dmc),
        (DC_CODE, options.initial_dc),
    ):
        state_component = None if state is None else getattr(state, code.value_name)
        if state_component is not None:
            code.initialize_state(
                seed=None,
                seed_name=code.value_name,
                initial_state=state_component,
                spatial_shape=spatial_shape,
            )
        elif seed is not None and not isinstance(seed, xr.DataArray):
            # a DataArray seed may be Dask-backed; validating it eagerly would compute it
            code.initialize_state(
                seed=seed,
                seed_name=code.value_name,
                initial_state=None,
                spatial_shape=spatial_shape,
            )


def _cffwis_block(
    temperature_block: np.ndarray,
    humidity_block: np.ndarray,
    wind_block: np.ndarray,
    precipitation_block: np.ndarray,
    latitude_block: np.ndarray,
    month_block: np.ndarray,
    *optional_blocks: np.ndarray,
    options: _CFFWISCallOptions,
    optional_kinds: tuple[str, ...],
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Compute one Dask chunk (or the whole array, eager): time axis last in, last out."""
    optional = dict(zip(optional_kinds, optional_blocks, strict=True))
    call_initial_state = None
    call_initial_ffmc = None
    call_initial_dmc = None
    call_initial_dc = None
    if options.initial_state is not None:
        call_initial_state = CFFWISState(
            ffmc=FFMCState(
                ffmc=np.asarray(optional["ffmc_value"]),
                trailing_gap_days=_trailing_gap_days_from_block(optional["ffmc_gap"]),
            ),
            dmc=DMCState(
                dmc=np.asarray(optional["dmc_value"]),
                trailing_gap_days=_trailing_gap_days_from_block(optional["dmc_gap"]),
            ),
            dc=DCState(
                dc=np.asarray(optional["dc_value"]),
                trailing_gap_days=_trailing_gap_days_from_block(optional["dc_gap"]),
            ),
        )
    else:
        call_initial_ffmc = optional.get("initial_ffmc")
        call_initial_dmc = optional.get("initial_dmc")
        call_initial_dc = optional.get("initial_dc")

    result = _cffwis.cffwis(
        # apply_ufunc places core dims (time) last; cffwis()'s NumPy path is time-first.
        np.moveaxis(temperature_block, -1, 0).copy(),
        np.moveaxis(humidity_block, -1, 0).copy(),
        np.moveaxis(wind_block, -1, 0).copy(),
        np.moveaxis(precipitation_block, -1, 0).copy(),
        latitude_block,
        np.moveaxis(month_block, -1, 0),
        initial_ffmc=call_initial_ffmc,
        initial_dmc=call_initial_dmc,
        initial_dc=call_initial_dc,
        initial_state=call_initial_state,
        return_state=options.return_state,
        spin_up=options.spin_up,
        nan_policy=options.nan_policy,
        max_gap_days=options.max_gap_days,
        outputs=options.selected,
    )
    assert isinstance(result, CFFWISResult)
    block_spatial_shape = np.broadcast_shapes(
        temperature_block.shape[:-1],
        humidity_block.shape[:-1],
        wind_block.shape[:-1],
        precipitation_block.shape[:-1],
    )
    computed: list[np.ndarray] = []
    for name in _selected_component_names(options.selected):
        component = getattr(result, name)
        assert isinstance(component, np.ndarray)
        computed.append(np.moveaxis(component, 0, -1))
    if options.return_state:
        state = result.state
        assert state is not None
        for state_value, state_gaps in (
            (state.ffmc.ffmc, state.ffmc.trailing_gap_days),
            (state.dmc.dmc, state.dmc.trailing_gap_days),
            (state.dc.dc, state.dc.trailing_gap_days),
        ):
            computed.append(state_value.reshape(block_spatial_shape))
            computed.append(
                np.full(block_spatial_shape, -1, dtype=np.int64)
                if state_gaps is None
                else state_gaps.reshape(block_spatial_shape)
            )
    return tuple(computed) if len(computed) != 1 else computed[0]


def _cffwis_apply_ufunc(
    weather: tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray],
    latitude: xr.DataArray,
    month_data: xr.DataArray,
    optional_args: list[xr.DataArray],
    optional_kinds: tuple[str, ...],
    options: _CFFWISCallOptions,
    output_time_length: int,
) -> tuple[xr.DataArray, ...]:
    """Run the shared NumPy orchestrator once per spatial block via apply_ufunc."""
    temperature, humidity, wind, precipitation = weather
    time_dim = options.time_dim
    active_components = _selected_component_names(options.selected)
    output_core_dims: list[list[str]] = [[time_dim]] * len(active_components)
    output_dtypes: list[type[np.generic]] = [np.float64] * len(active_components)
    if options.return_state:
        output_core_dims += [[]] * 6
        output_dtypes += [np.float64, np.int64] * 3
    apply_results = xr.apply_ufunc(
        _cffwis_block,
        temperature,
        humidity,
        wind,
        precipitation,
        latitude,
        month_data,
        *optional_args,
        input_core_dims=[[time_dim]] * 4 + [[], [time_dim]] + [[] for _ in optional_args],
        output_core_dims=output_core_dims,
        exclude_dims={time_dim},
        vectorize=False,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {time_dim: output_time_length}},
        output_dtypes=output_dtypes,
        kwargs={"options": options, "optional_kinds": optional_kinds},
    )
    return apply_results if isinstance(apply_results, tuple) else (apply_results,)


def _cffwis_variable_results(
    result_arrays: tuple[xr.DataArray, ...],
    active_components: tuple[str, ...],
    output_dims: tuple[str, ...],
    time_coord: xr.DataArray | None,
    temperature_celsius: xr.DataArray,
    options: _CFFWISCallOptions,
    output_time_length: int,
) -> dict[str, xr.DataArray]:
    """Wrap each apply_ufunc output with its dims, time coordinate, and CF attrs."""
    variable_results: dict[str, xr.DataArray] = {}
    for position, name in enumerate(active_components):
        variable = result_arrays[position]
        if variable.dims != output_dims:
            variable = variable.transpose(*output_dims)
        if time_coord is not None:
            # slice the coordinate rather than its values so CF coordinate
            # attributes (calendar, axis, ...) survive spin-up trimming
            trimmed = time_coord.isel({options.time_dim: slice(options.spin_up, options.spin_up + output_time_length)})
            variable = variable.assign_coords({options.time_dim: trimmed})
        variable.attrs = build_output_attrs(
            temperature_celsius,
            cf_metadata=CF_METADATA[name],  # type: ignore[arg-type]
            calculation_metadata={"nan_policy": options.nan_policy},
            index_name=name.upper(),
        )
        variable_results[name] = variable
    return variable_results


def _cffwis_xarray(
    temperature_celsius: xr.DataArray,
    relative_humidity_percent: xr.DataArray,
    wind_speed_meters_per_second: xr.DataArray,
    precipitation_mm: xr.DataArray,
    latitude_degrees_north: npt.ArrayLike | xr.DataArray | None,
    month: npt.ArrayLike | xr.DataArray | None,
    options: _CFFWISCallOptions,
) -> xr.Dataset | CFFWISResult:
    """xarray dispatch for :func:`cffwis`. See :func:`cffwis` for the full contract.

    The multi-output adapter the design doc calls for: one
    :func:`xarray.apply_ufunc` call runs the shared NumPy orchestrator once per
    Dask spatial block with the full ``time`` axis, then each selected output
    is rewrapped with its own ``CF_METADATA`` entry. The per-cell day-length
    broadcast needs no adapter logic -- passing a latitude DataArray through
    ``apply_ufunc`` gives every block its own latitudes, and the core's
    latitude-band tables are already elementwise. Per-call registry resolution
    is not CFFWIS's problem (its entries are fixed), unlike KBDI's
    ``units``-dependent ``kbdi``/``kbdi_imperial`` choice.
    """
    time_dim = options.time_dim
    active_components = _selected_component_names(options.selected)
    weather_inputs = (
        temperature_celsius,
        relative_humidity_percent,
        wind_speed_meters_per_second,
        precipitation_mm,
    )
    _validate_cffwis_xarray_inputs(weather_inputs, time_dim)
    _warn_if_not_noon_referenced(weather_inputs, time_dim)

    temperature, humidity, wind, precipitation = _align_cffwis_inputs(
        temperature_celsius,
        relative_humidity_percent,
        wind_speed_meters_per_second,
        precipitation_mm,
        time_dim,
    )
    for data in (temperature, humidity, wind, precipitation):
        validate_dask_chunks(data, time_dim)

    temperature = _convert_temperature_units(temperature, "celsius", argument_name="temperature_celsius.attrs['units']")
    precipitation = _convert_precipitation_units(precipitation, "mm")
    weather = (temperature, humidity, wind, precipitation)
    # one shared spatial topology: a time-only input joins the others' grid.
    # apply_ufunc performs that broadcast per block, so the adapter must not
    # xr.broadcast first: expanding a Dask input over the new spatial
    # dimensions places the whole grid in a single task, which is exactly what
    # the bounded spatial-block execution exists to avoid.
    broadcast_dims, broadcast_sizes = _broadcast_topology(weather)
    spatial_dims = tuple(dim for dim in broadcast_dims if dim != time_dim)
    spatial_shape = tuple(broadcast_sizes[dim] for dim in spatial_dims)
    spatial_chunks = _spatial_chunk_targets(weather, spatial_dims)
    time_length = broadcast_sizes[time_dim]
    output_time_length = max(time_length - options.spin_up, 0)

    latitude = _resolve_cffwis_latitude(
        latitude_degrees_north,
        weather,
        time_dim,
        spatial_dims,
        spatial_shape,
    )
    # a not-yet-broadcast input may not carry the time coordinate another one
    # does; the aligned weather inputs share it, so infer month and trim the
    # output from the same first coordinate-bearing input
    time_coord = next((data.coords[time_dim] for data in weather if time_dim in data.coords), None)
    month_data = _resolve_cffwis_month(month, time_coord, time_dim, time_length)

    internal_spatial_shape = spatial_shape or (1,)
    _validate_cffwis_seeds(options, internal_spatial_shape)

    optional_kinds = _cffwis_optional_kinds(options)
    optional_args = _cffwis_optional_args(options, spatial_shape, spatial_dims, spatial_chunks)
    result_arrays = _cffwis_apply_ufunc(
        weather,
        latitude,
        month_data,
        optional_args,
        optional_kinds,
        options,
        output_time_length,
    )
    variable_results = _cffwis_variable_results(
        result_arrays,
        active_components,
        broadcast_dims,
        time_coord,
        temperature_celsius,
        options,
        output_time_length,
    )

    if not options.return_state:
        return xr.Dataset(variable_results)

    # compute the final state (it must be plain NumPy): the component values
    # stay lazy so return_state=True never materializes seven full grids
    state_results = {
        "cffwis_state_ffmc": result_arrays[len(active_components)],
        "cffwis_state_ffmc_gap": result_arrays[len(active_components) + 1],
        "cffwis_state_dmc": result_arrays[len(active_components) + 2],
        "cffwis_state_dmc_gap": result_arrays[len(active_components) + 3],
        "cffwis_state_dc": result_arrays[len(active_components) + 4],
        "cffwis_state_dc_gap": result_arrays[len(active_components) + 5],
    }
    loaded_state = xr.Dataset(state_results).load()
    return CFFWISResult(
        ffmc=variable_results.get("ffmc"),
        dmc=variable_results.get("dmc"),
        dc=variable_results.get("dc"),
        isi=variable_results.get("isi"),
        bui=variable_results.get("bui"),
        fwi=variable_results.get("fwi"),
        dsr=variable_results.get("dsr"),
        state=CFFWISState(
            ffmc=FFMCState(
                ffmc=loaded_state["cffwis_state_ffmc"].values,
                trailing_gap_days=_trailing_gap_days_from_block(loaded_state["cffwis_state_ffmc_gap"].values),
            ),
            dmc=DMCState(
                dmc=loaded_state["cffwis_state_dmc"].values,
                trailing_gap_days=_trailing_gap_days_from_block(loaded_state["cffwis_state_dmc_gap"].values),
            ),
            dc=DCState(
                dc=loaded_state["cffwis_state_dc"].values,
                trailing_gap_days=_trailing_gap_days_from_block(loaded_state["cffwis_state_dc_gap"].values),
            ),
        ),
    )

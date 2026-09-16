"""Haines Index (Lower Atmosphere Severity Index)."""

from __future__ import annotations

import time
from typing import NamedTuple, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError
from climate_indices.fire._common import _as_float_array
from climate_indices.fire._units import _convert_temperature_units
from climate_indices.logging_config import get_logger
from climate_indices.performance import check_large_array_memory
from climate_indices.xarray_adapter import _build_output_attrs

# retrieve structlog logger for this module
_logger = get_logger(__name__)


class _Variant(NamedTuple):
    """Pressure levels and score cut points of one Haines elevation variant."""

    lower_hpa: float
    upper_hpa: float
    moisture_hpa: float
    stability_cut_points: tuple[float, float]
    moisture_cut_points: tuple[float, float]


# The three elevation variants of Haines (1988). Cut points follow the
# operational implementations rather than the rounded tables some restatements
# print: NWS's AWIPS GFE smart-init (Unidata/awips2, thresholds [4, 8]/[6, 10],
# [6, 11]/[6, 13], [18, 22]/[15, 21]) and NOAA's LAPS `hainesindex.f` agree on
# every one, and LAPS also withholds the index when the variant's levels are
# below the surface pressure, which `surface_pressure_hpa` reproduces here.
# Scores are half-open bins (< lower cut point -> 1, < upper cut point -> 2,
# else 3), so the non-integer lapse rates and depressions the functions accept
# land in the same bins as the published integer tables. The moisture term
# pairs with whichever supplied temperature sits at `moisture_hpa`: the lower
# one for the mid and high variants, the upper one for the low variant, whose
# 850 hPa moisture level is above its 950 hPa stability level.
_HAINES_VARIANTS: dict[str, _Variant] = {
    "low": _Variant(950.0, 850.0, 850.0, (4.0, 8.0), (6.0, 10.0)),
    "mid": _Variant(850.0, 700.0, 850.0, (6.0, 11.0), (6.0, 13.0)),
    "high": _Variant(700.0, 500.0, 700.0, (18.0, 22.0), (15.0, 21.0)),
}

# Elevation bands of the opt-in automatic variant selection, the conventional
# 1000 ft and 3000 ft boundaries.
_HAINES_LOW_ELEVATION_MAX_METERS = 305.0
_HAINES_MID_ELEVATION_MAX_METERS = 914.0

# Levels the profile entry point interpolates to, in the same order as the
# variant table's levels.
_HAINES_TEMPERATURE_LEVELS_HPA = (950.0, 850.0, 700.0, 500.0)
_HAINES_DEWPOINT_LEVELS_HPA = (850.0, 700.0)

_VARIANT_NAMES = tuple(_HAINES_VARIANTS)


def _variant_spec(variant: str) -> _Variant:
    """Look up a variant's levels and cut points, rejecting unknown names.

    Raises:
        InvalidArgumentError: If ``variant`` is not ``"low"``, ``"mid"``, or
            ``"high"``.
    """
    try:
        return _HAINES_VARIANTS[variant]
    except (KeyError, TypeError) as exc:
        message = f"Unknown Haines Index elevation variant: {variant!r}."
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="variant",
            argument_value=repr(variant),
            valid_values=", ".join(repr(name) for name in _VARIANT_NAMES),
        ) from exc


def _score(delta: npt.NDArray[np.float64], cut_points: tuple[float, float]) -> npt.NDArray[np.float64]:
    """Score one term of the index as 1, 2, or 3 against its cut points.

    A NaN delta scores NaN rather than falling through the comparisons to 3:
    a level the profile does not reach, or a masked observation, must withhold
    that cell's index instead of scoring it as the most severe.
    """
    score = np.where(delta < cut_points[0], 1.0, np.where(delta < cut_points[1], 2.0, 3.0))
    result: npt.NDArray[np.float64] = np.asarray(np.where(np.isnan(delta), np.nan, score), dtype=np.float64)
    return result


def _haines_from_levels(
    temperature_lower: npt.NDArray[np.float64],
    temperature_upper: npt.NDArray[np.float64],
    dewpoint: npt.NDArray[np.float64],
    surface_pressure_hpa: npt.NDArray[np.float64] | None = None,
    *,
    variant: str,
    warn: bool = True,
) -> npt.NDArray[np.float64]:
    """Score already-broadcast level values for one elevation variant.

    Shared by :func:`haines_index`'s NumPy path and its xarray kernel. The
    xarray path disables ``warn`` for Dask blocks, where one warning per block
    would swamp the operation-level signal; NaN results still mark the masked
    cells.

    Args:
        temperature_lower: Temperature at the variant's lower stability level,
            degrees Celsius.
        temperature_upper: Temperature at the variant's upper stability level,
            degrees Celsius.
        dewpoint: Dewpoint at the variant's moisture level, degrees Celsius.
        surface_pressure_hpa: Surface pressure, hPa, or None to skip the
            below-ground mask.
        variant: Elevation variant name, already validated.
        warn: Emit the below-ground warning.

    Returns:
        Haines Index in [2, 6], NaN where ``surface_pressure_hpa`` is missing
        or lies below the variant's lower stability level.
    """
    spec = _HAINES_VARIANTS[variant]
    moisture_temperature = temperature_lower if spec.moisture_hpa == spec.lower_hpa else temperature_upper
    index = _score(temperature_lower - temperature_upper, spec.stability_cut_points) + _score(
        moisture_temperature - dewpoint, spec.moisture_cut_points
    )
    if surface_pressure_hpa is None:
        return np.asarray(index, dtype=np.float64)

    # NaN pressure counts as masked: a caller who supplies pressure but does not
    # know it must not receive a below-ground level scored as if it were valid
    below_ground = ~(surface_pressure_hpa >= spec.lower_hpa)
    if warn:
        below_ground_count = int(np.count_nonzero(below_ground))
        if below_ground_count > 0:
            _logger.warning(
                f"Found {below_ground_count} values with surface pressure below the "
                f"{spec.lower_hpa:.0f} hPa level required by the {variant!r} variant; "
                "the Haines Index is NaN there rather than extrapolated."
            )
    result: npt.NDArray[np.float64] = np.asarray(np.where(below_ground, np.nan, index), dtype=np.float64)
    return result


def _interpolate_log_pressure(
    profile: npt.NDArray[np.float64],
    pressure_hpa: npt.NDArray[np.float64],
    target_levels_hpa: tuple[float, ...],
) -> npt.NDArray[np.float64]:
    """Interpolate a profile to target pressure levels, NaN outside its range.

    Interpolation is linear in log pressure, the standard treatment of a
    hydrostatic profile (and what NOAA's LAPS ``hainesindex.f`` does between
    its bracketing levels). Targets outside the profile's own pressure range
    are NaN: a level below the surface or above the top of the profile is
    withheld, never extrapolated, and never clamped to the nearest level.

    Args:
        profile: Values on the pressure levels, pressure on the last axis.
        pressure_hpa: The profile's pressure levels, hPa, strictly decreasing.
        target_levels_hpa: Levels to interpolate to, hPa.

    Returns:
        Values at the target levels, target level on the last axis.
    """
    # log pressure increases upward, so reverse both to make np.searchsorted's
    # increasing-array requirement hold
    log_pressure = np.log(pressure_hpa)[::-1]
    values = profile[..., ::-1]
    target = np.log(np.asarray(target_levels_hpa, dtype=np.float64))

    bracket = np.clip(np.searchsorted(log_pressure, target), 1, log_pressure.size - 1)
    lower_pressure = log_pressure[bracket - 1]
    upper_pressure = log_pressure[bracket]
    weight = (target - lower_pressure) / (upper_pressure - lower_pressure)
    interpolated = values[..., bracket - 1] + weight * (values[..., bracket] - values[..., bracket - 1])

    outside = (target < log_pressure[0]) | (target > log_pressure[-1])
    result: npt.NDArray[np.float64] = np.where(outside, np.nan, interpolated).astype(np.float64, copy=False)
    return result


def _haines_from_profile(
    temperature: npt.NDArray[np.float64],
    dewpoint: npt.NDArray[np.float64],
    pressure_hpa: npt.NDArray[np.float64],
    elevation_meters: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Score already-broadcast profiles for the variant each elevation selects.

    All three variants are scored and each cell takes the one its elevation
    selects, so a grid with an elevation field needs no per-cell branching.
    Cells whose selected variant needs a level the profile does not reach are
    NaN, via the interpolation mask, and an unknown elevation withholds the
    cell rather than defaulting to a variant.
    """
    temperature_levels = _interpolate_log_pressure(temperature, pressure_hpa, _HAINES_TEMPERATURE_LEVELS_HPA)
    dewpoint_levels = _interpolate_log_pressure(dewpoint, pressure_hpa, _HAINES_DEWPOINT_LEVELS_HPA)
    temperature_950 = temperature_levels[..., 0]
    temperature_850 = temperature_levels[..., 1]
    temperature_700 = temperature_levels[..., 2]
    temperature_500 = temperature_levels[..., 3]
    dewpoint_850 = dewpoint_levels[..., 0]
    dewpoint_700 = dewpoint_levels[..., 1]

    low = _HAINES_VARIANTS["low"]
    mid = _HAINES_VARIANTS["mid"]
    high = _HAINES_VARIANTS["high"]
    low_index = _score(temperature_950 - temperature_850, low.stability_cut_points) + _score(
        temperature_850 - dewpoint_850, low.moisture_cut_points
    )
    mid_index = _score(temperature_850 - temperature_700, mid.stability_cut_points) + _score(
        temperature_850 - dewpoint_850, mid.moisture_cut_points
    )
    high_index = _score(temperature_700 - temperature_500, high.stability_cut_points) + _score(
        temperature_700 - dewpoint_700, high.moisture_cut_points
    )
    result: npt.NDArray[np.float64] = np.asarray(
        np.where(
            np.isnan(elevation_meters),
            np.nan,
            np.where(
                elevation_meters < _HAINES_LOW_ELEVATION_MAX_METERS,
                low_index,
                np.where(elevation_meters <= _HAINES_MID_ELEVATION_MAX_METERS, mid_index, high_index),
            ),
        ),
        dtype=np.float64,
    )
    return result


def _broadcast_inputs(
    names: tuple[str, ...],
    arrays: tuple[npt.NDArray[np.float64], ...],
    index_name: str,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Broadcast inputs together, reporting the shapes when they cannot be.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.
    """
    try:
        return tuple(np.broadcast_arrays(*arrays))
    except ValueError as exc:
        shapes = ", ".join(f"{name}={array.shape}" for name, array in zip(names, arrays, strict=True))
        message = f"Incompatible array shapes for {index_name}: {shapes}. The inputs must broadcast together."
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="/".join(names),
            argument_value=f"shapes {', '.join(str(array.shape) for array in arrays)}",
            valid_values="Arrays broadcastable to a common shape",
        ) from exc


@overload
def haines_index(
    temperature_lower_celsius: xr.DataArray,
    temperature_upper_celsius: xr.DataArray,
    dewpoint_celsius: xr.DataArray,
    *,
    variant: str,
    surface_pressure_hpa: xr.DataArray | float | None = None,
) -> xr.DataArray: ...


@overload
def haines_index(
    temperature_lower_celsius: npt.ArrayLike,
    temperature_upper_celsius: npt.ArrayLike,
    dewpoint_celsius: npt.ArrayLike,
    *,
    variant: str,
    surface_pressure_hpa: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]: ...


def haines_index(
    temperature_lower_celsius: npt.ArrayLike | xr.DataArray,
    temperature_upper_celsius: npt.ArrayLike | xr.DataArray,
    dewpoint_celsius: npt.ArrayLike | xr.DataArray,
    *,
    variant: str,
    surface_pressure_hpa: npt.ArrayLike | xr.DataArray | None = None,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Compute the Haines Index from values already selected at its levels.

    This function accepts both NumPy arrays and xarray DataArrays. Type
    checkers narrow the return type based on the input type.

    .. warning:: **Beta Feature (xarray path only)** -- When called with
       ``xr.DataArray`` input, this function uses the beta xarray adapter
       layer: CF metadata from the ``haines_<variant>`` registry entry, CF
       ``units``-attribute temperature conversion, and Dask parallelism over
       every dimension (no dimension has a chunk constraint, because nothing
       is reduced). The NumPy array interface and underlying computation are
       stable.

    The index sums a stability term and a moisture term, each scored 1 to 3
    against the lapse rate and dewpoint depression of one lower-atmosphere
    layer (Haines, 1988)::

        Haines = stability(temperature_lower - temperature_upper)
               + moisture(moisture_level_temperature - dewpoint_celsius)

    The three elevation variants score different layers, and the caller
    selects the levels before calling:

    ========  =================  ====================  ====================
    variant   temperature_lower  temperature_upper     dewpoint_celsius
    ========  =================  ====================  ====================
    ``low``   950 hPa            850 hPa               850 hPa
    ``mid``   850 hPa            700 hPa               850 hPa
    ``high``  700 hPa            500 hPa               700 hPa
    ========  =================  ====================  ====================

    The moisture term pairs its dewpoint with whichever of the two
    temperatures sits at the moisture level, so for ``low`` it uses
    ``temperature_upper_celsius``. Score cut points are per variant (see
    :func:`haines_index_from_profile` for the published tables).

    ``haines_index_from_profile`` is the alternative entry point for a whole
    vertical profile plus a terrain elevation field: it interpolates these
    same levels and selects the variant per cell. Use this function when the
    levels are already resolved, or when the variant is fixed for a site.

    Without a wind term the index cannot distinguish a dry, unstable but calm
    day from a dangerous one, and it is a diagnostic of the atmosphere's
    potential for large fire growth rather than a predictor of fire starts or
    of spread rate; the Hot-Dry-Windy Index (``fire.hot_dry_windy``) was
    developed in part to supply the missing wind and surface coupling.

    Args:
        temperature_lower_celsius: Temperature at the variant's lower
            stability level, degrees Celsius.
        temperature_upper_celsius: Temperature at the variant's upper
            stability level, degrees Celsius.
        dewpoint_celsius: Dewpoint at the variant's moisture level (see the
            table above), degrees Celsius.
        variant: Elevation variant, ``"low"``, ``"mid"``, or ``"high"``.
            Required and never inferred here.
        surface_pressure_hpa: Optional surface pressure, hPa, array-like or
            ``xr.DataArray`` (xarray input: a ``DataArray`` or a scalar).
            Where it lies below the variant's lower stability level the level
            is below ground, and the index is NaN there rather than scored
            from whatever the input dataset extrapolated; a NaN pressure is
            treated the same way. Array-like or ``xr.DataArray`` must
            broadcast against the temperature inputs.

    Returns:
        Haines Index, an integer-valued float in [2, 6], with the broadcast
        shape of the inputs. NaN where any input is NaN or masked, and where
        ``surface_pressure_hpa`` withholds the variant's level. For xarray
        input, a ``DataArray`` carrying CF metadata from the
        ``haines_<variant>`` registry entry, including
        ``climate_indices_variant``.

    Raises:
        TypeError: If the three temperature inputs are not all the same type.
        InvalidArgumentError: If ``variant`` is not one of the three names, if
            the inputs cannot be broadcast together, or if
            ``surface_pressure_hpa`` is not a ``DataArray`` or scalar
            alongside xarray input.

    Notes:
        xarray-only: the three temperature inputs must be the same type (all
        NumPy or all ``xr.DataArray``). A CF ``units`` attribute on any of
        them is converted to Celsius; an absent attribute is assumed to
        already be Celsius. Haines has no time semantics -- it carries no
        state and is not a daily recurrence -- so every dimension is a plain
        passthrough. Dimensions shared by the inputs are matched exactly
        (xarray's default exact join): the inputs are never aligned or
        reindexed, so unequal or differently ordered coordinate labels raise
        instead of broadcasting.

    Example:
        >>> from climate_indices import fire
        >>> float(fire.haines_index(30.0, 26.0, 20.0, variant="low"))
        4.0
    """
    is_xarray = isinstance(temperature_lower_celsius, xr.DataArray)
    for name, value in (
        ("temperature_upper_celsius", temperature_upper_celsius),
        ("dewpoint_celsius", dewpoint_celsius),
    ):
        if isinstance(value, xr.DataArray) != is_xarray:
            raise TypeError(
                "temperature_lower_celsius, temperature_upper_celsius, and dewpoint_celsius must be the same "
                f"type. Got temperature_lower_celsius={type(temperature_lower_celsius).__name__}, "
                f"{name}={type(value).__name__}. "
                "Convert all three to the same type (all numpy arrays or all xr.DataArray)."
            )

    spec = _variant_spec(variant)

    if is_xarray:
        assert isinstance(temperature_lower_celsius, xr.DataArray)
        assert isinstance(temperature_upper_celsius, xr.DataArray)
        assert isinstance(dewpoint_celsius, xr.DataArray)
        return _haines_xarray(
            temperature_lower_celsius,
            temperature_upper_celsius,
            dewpoint_celsius,
            surface_pressure_hpa,
            variant=variant,
        )

    temperature_lower = _as_float_array(temperature_lower_celsius)
    temperature_upper = _as_float_array(temperature_upper_celsius)
    dewpoint = _as_float_array(dewpoint_celsius)
    pressure = None if surface_pressure_hpa is None else _as_float_array(surface_pressure_hpa)

    names: tuple[str, ...] = ("temperature_lower_celsius", "temperature_upper_celsius", "dewpoint_celsius")
    arrays: tuple[npt.NDArray[np.float64], ...] = (temperature_lower, temperature_upper, dewpoint)
    if pressure is not None:
        names = names + ("surface_pressure_hpa",)
        arrays = arrays + (pressure,)
    temperature_lower, temperature_upper, dewpoint, *rest = _broadcast_inputs(names, arrays, "Haines Index")
    pressure = rest[0] if rest else None

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="haines_index",
        variant=variant,
        input_shape=temperature_lower.shape,
        input_elements=temperature_lower.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(
        temperature_lower,
        temperature_upper,
        dewpoint,
        *(() if pressure is None else (pressure,)),
    )

    try:
        result = _haines_from_levels(
            temperature_lower,
            temperature_upper,
            dewpoint,
            pressure,
            variant=variant,
            warn=True,
        )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=result.shape,
            level=f"{spec.lower_hpa:.0f}-{spec.upper_hpa:.0f} hPa",
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


def haines_index_from_profile(
    temperature_celsius: npt.ArrayLike | xr.DataArray,
    dewpoint_celsius: npt.ArrayLike | xr.DataArray,
    pressure_hpa: npt.ArrayLike,
    elevation_meters: npt.ArrayLike,
    *,
    pressure_axis: int = -1,
) -> npt.NDArray[np.float64]:
    """Compute the Haines Index from vertical profiles and a terrain elevation.

    The opt-in automatic form of :func:`haines_index`: instead of the caller
    resolving levels and variant, this function interpolates each profile to
    the levels the elevation bands select and scores the matching variant per
    cell -- ``low`` below 305 m (1000 ft), ``mid`` up to 914 m (3000 ft), and
    ``high`` above that. The variant is never inferred for a fixed-level call:
    :func:`haines_index` still requires ``variant``.

    Interpolation is linear in log pressure between the profile's own levels,
    as in NOAA's LAPS ``hainesindex.f``. A level the profile does not reach --
    pressure below the surface, or above the top of the profile -- makes that
    cell NaN instead of being extrapolated or clamped, which is what keeps a
    high-elevation column from being scored at a level it never observes.

    Args:
        temperature_celsius: Temperature profile, degrees Celsius, with the
            pressure levels on ``pressure_axis``.
        dewpoint_celsius: Dewpoint profile, degrees Celsius, broadcast against
            ``temperature_celsius``. It carries the same pressure levels, even
            though only the 850 and 700 hPa values are scored.
        pressure_hpa: Pressure level of each profile entry, hPa, strictly
            decreasing and strictly positive. At least two levels, one value
            per level, matching the profile's ``pressure_axis``.
        elevation_meters: Terrain elevation, meters, scalar or broadcast
            against the profile with the pressure axis removed. Selects the
            variant per cell.
        pressure_axis: Axis of ``temperature_celsius`` and
            ``dewpoint_celsius`` on which ``pressure_hpa`` indexes (default
            the last axis). Removed from the output.

    Returns:
        Haines Index, an integer-valued float in [2, 6], with the broadcast
        shape of the temperature and dewpoint profiles minus
        ``pressure_axis``. NaN where an input is NaN or masked, and where the
        cell's elevation-selected variant needs a level the profile does not
        cover.

    Raises:
        InputTypeError: If an input is not numeric (see ``_as_float_array``).
        DataShapeError: If ``pressure_hpa`` is not one-dimensional, has fewer
            than two levels, if ``temperature_celsius`` has no profile axis,
            if ``pressure_axis`` is out of range, or if ``pressure_hpa`` does
            not have one value per profile level.
        InvalidArgumentError: If ``pressure_hpa`` is not strictly decreasing
            or not strictly positive, if ``temperature_celsius`` and
            ``dewpoint_celsius`` cannot be broadcast together, or if
            ``elevation_meters`` cannot be broadcast against the profile.

    Example:
        >>> from climate_indices import fire
        >>> profile = [32.0, 24.0, 12.0, -8.0]
        >>> dewpoint = [26.0, 13.0, 2.0, -20.0]
        >>> float(fire.haines_index_from_profile(profile, dewpoint, [950.0, 850.0, 700.0, 500.0], 100.0))
        6.0
    """
    if isinstance(temperature_celsius, xr.DataArray) or isinstance(dewpoint_celsius, xr.DataArray):
        raise InvalidArgumentError(
            "haines_index_from_profile does not accept xr.DataArray input; "
            "select the variant's levels and call haines_index instead.",
            argument_name="temperature_celsius/dewpoint_celsius",
            argument_value="xr.DataArray",
            valid_values="NumPy-compatible array-likes",
        )

    temperature = _as_float_array(temperature_celsius)
    dewpoint = _as_float_array(dewpoint_celsius)
    pressure = _as_float_array(pressure_hpa)
    elevation = _as_float_array(elevation_meters)

    if pressure.ndim != 1:
        raise DataShapeError(
            f"pressure_hpa must be one-dimensional, got {pressure.ndim} dimensions.",
            expected_shape="(levels,)",
            actual_shape=pressure.shape,
        )
    if temperature.ndim == 0:
        raise DataShapeError(
            "temperature_celsius must carry a profile axis; a scalar has no pressure levels.",
            expected_shape=f"(..., levels) with levels == {pressure.shape[0]}",
            actual_shape=temperature.shape,
        )
    if not -temperature.ndim <= pressure_axis < temperature.ndim:
        raise DataShapeError(
            f"pressure_axis {pressure_axis} is out of range for the input shape "
            f"{temperature.shape} with {temperature.ndim} dimensions.",
            expected_shape=f"An axis of the input shape {temperature.shape}",
            actual_shape=temperature.shape,
        )
    axis = pressure_axis % temperature.ndim
    if temperature.shape[axis] != pressure.shape[0]:
        raise DataShapeError(
            f"pressure_hpa has {pressure.shape[0]} levels but temperature_celsius has "
            f"{temperature.shape[axis]} along axis {axis}.",
            expected_shape=f"{temperature.shape} with axis {axis} of length {pressure.shape[0]}",
            actual_shape=temperature.shape,
        )
    if pressure.shape[0] < 2:
        raise DataShapeError(
            f"pressure_hpa has {pressure.shape[0]} levels; the Haines Index needs at least two "
            "(a stability layer's bottom and top).",
            expected_shape=f"(levels >= 2,) matching the profile's axis {axis}",
            actual_shape=pressure.shape,
        )
    if not np.all(np.diff(pressure) < 0.0):
        raise InvalidArgumentError(
            "pressure_hpa must be strictly decreasing, from the surface level upward.",
            argument_name="pressure_hpa",
            argument_value=f"levels {pressure[:1]} ... {pressure[-1:]}",
            valid_values="Strictly decreasing pressures in hPa",
        )
    if not np.all(pressure > 0.0):
        raise InvalidArgumentError(
            "pressure_hpa must be strictly positive; non-positive levels are not pressures.",
            argument_name="pressure_hpa",
            argument_value=f"minimum level {float(np.min(pressure))}",
            valid_values="Strictly positive pressures in hPa",
        )

    temperature, dewpoint = _broadcast_inputs(
        ("temperature_celsius", "dewpoint_celsius"),
        (temperature, dewpoint),
        "Haines Index profile",
    )
    # move the profile axis last, interpolate, and let it fall away: the
    # remaining axes keep their original order
    temperature = np.moveaxis(temperature, axis, -1)
    dewpoint = np.moveaxis(dewpoint, axis, -1)

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="haines_index_from_profile",
        input_shape=temperature.shape,
        input_elements=temperature.size,
        pressure_levels=pressure.shape[0],
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(temperature, dewpoint, pressure, elevation)

    try:
        index = _haines_from_profile(temperature, dewpoint, pressure, elevation)
    except ValueError as exc:
        message = (
            "Incompatible array shapes for Haines Index profile: "
            f"temperature={temperature.shape}, dewpoint={dewpoint.shape}, "
            f"pressure={pressure.shape}, elevation={elevation.shape}. "
            "The elevation must broadcast against the profile with its pressure axis removed."
        )
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise InvalidArgumentError(
            message,
            argument_name="elevation_meters",
            argument_value=str(elevation.shape),
            valid_values=f"A scalar or an array broadcastable to {temperature.shape[:-1]}",
        ) from exc

    unavailable = int(np.count_nonzero(np.isnan(index)))
    if unavailable > 0:
        log.warning(
            f"Found {unavailable} cells whose elevation-selected variant needs a level the "
            "profile does not cover; the Haines Index is NaN there rather than extrapolated."
        )

    duration_ms = (time.perf_counter() - t0) * 1000.0
    log.info(
        "calculation_completed",
        duration_ms=round(duration_ms, 2),
        output_shape=index.shape,
        **(memory_metrics or {}),
    )
    return index


def _haines_xarray(
    temperature_lower_celsius: xr.DataArray,
    temperature_upper_celsius: xr.DataArray,
    dewpoint_celsius: xr.DataArray,
    surface_pressure_hpa: npt.ArrayLike | xr.DataArray | None,
    *,
    variant: str,
) -> xr.DataArray:
    """xarray dispatch for :func:`haines_index`. See :func:`haines_index` for the full contract.

    Haines is weather-only, stateless, and elementwise: every dimension is a
    plain passthrough, no dimension is reduced, and no dimension therefore has
    a chunk constraint. ``xr.apply_ufunc`` calls the shared silent kernel
    :func:`_haines_from_levels` with no core dimensions, which leaves it to
    NumPy's broadcasting; using the kernel rather than the public function
    keeps one xarray operation from emitting per-block calculation events,
    memory instrumentation, and below-ground warnings. Eager input still
    reports below-ground cells through the kernel's warning, and only
    Dask-backed input suppresses it per block.
    """
    temperatures = tuple(
        _convert_temperature_units(data, "celsius", argument_name=f"{name}.attrs['units']")
        for name, data in (
            ("temperature_lower_celsius", temperature_lower_celsius),
            ("temperature_upper_celsius", temperature_upper_celsius),
            ("dewpoint_celsius", dewpoint_celsius),
        )
    )
    inputs: tuple[xr.DataArray | npt.NDArray[np.float64], ...] = temperatures
    if surface_pressure_hpa is not None:
        if isinstance(surface_pressure_hpa, xr.DataArray):
            inputs = inputs + (surface_pressure_hpa,)
        elif np.ndim(surface_pressure_hpa) == 0:
            inputs = inputs + (np.asarray(surface_pressure_hpa, dtype=np.float64),)
        else:
            raise InvalidArgumentError(
                "surface_pressure_hpa must be an xr.DataArray or a scalar when temperature_lower_celsius "
                "is an xr.DataArray; a plain array would not carry dimension labels.",
                argument_name="surface_pressure_hpa",
                argument_value=f"{type(surface_pressure_hpa).__name__} with ndim={np.ndim(surface_pressure_hpa)}",
                valid_values="An xr.DataArray, or a scalar",
            )

    # reject non-numeric dtype here rather than letting the kernel's comparisons
    # fail inside numpy (or, for Dask blocks, inside compute()) with a raw
    # UFuncTypeError; dtype is known without computing
    for name, data in zip(
        ("temperature_lower_celsius", "temperature_upper_celsius", "dewpoint_celsius", "surface_pressure_hpa"),
        inputs,
        strict=False,
    ):
        if data.dtype.kind not in "biuf":
            raise InputTypeError(
                f"Haines Index inputs must be numeric: {name} has dtype {data.dtype}.",
                expected_type=float,
                actual_type=data.dtype.type,
            )

    # one warning per invalid Dask block would swamp the logs; eager input still
    # reports below-ground cells through the shared kernel's warning
    is_dask_backed = any(isinstance(data, xr.DataArray) and data.chunks is not None for data in inputs)
    result: xr.DataArray = xr.apply_ufunc(
        _haines_from_levels,
        *inputs,
        input_core_dims=[[]] * len(inputs),
        kwargs={"variant": variant, "warn": not is_dask_backed},
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    result.attrs = _build_output_attrs(
        temperature_lower_celsius,
        cf_metadata=CF_METADATA[f"haines_{variant}"],  # type: ignore[arg-type]
        index_name="Haines Index",
    )
    return result

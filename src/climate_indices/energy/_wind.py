"""Wind profiles and power curves for climate-driven resources, not generation."""

from __future__ import annotations

import time
from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices._recurrence import _as_float_array
from climate_indices.energy._xarray import apply_elementwise, scalar_coordinate, wind_input
from climate_indices.exceptions import InputTypeError, InvalidArgumentError
from climate_indices.logging_config import get_logger, log_calculation_failure
from climate_indices.performance import check_large_array_memory

_logger = get_logger(__name__)

_Height: TypeAlias = float | np.integer[Any] | np.floating[Any]


def _positive_height(value: _Height, name: str) -> float:
    """Require a finite positive scalar height in metres."""
    message = f"{name} must be a finite positive scalar in metres."
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidArgumentError(message, argument_name=name)
    try:
        height = float(value)
    except OverflowError as exc:
        raise InvalidArgumentError(message, argument_name=name) from exc
    if not np.isfinite(height) or height <= 0:
        raise InvalidArgumentError(message, argument_name=name)
    return height


def _hub_height_kernel(
    wind: npt.ArrayLike,
    exponent: npt.ArrayLike | None,
    reference_height: float,
    hub_height: float,
    roughness_length: _Height | None,
) -> npt.NDArray[np.float64]:
    """Evaluate a validated profile without operational side effects."""
    wind = _as_float_array(wind)
    if exponent is not None:
        exponent = _as_float_array(exponent)
        valid = np.isfinite(wind) & (wind >= 0) & np.isfinite(exponent)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            factor = np.power(hub_height / reference_height, exponent)
    elif roughness_length is None:
        raise InvalidArgumentError("The log profile requires roughness_length.", argument_name="roughness_length")
    else:
        factor = np.log(hub_height / roughness_length) / np.log(reference_height / roughness_length)
        valid = np.isfinite(wind) & (wind >= 0)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.where(wind == 0, 0.0, wind * factor)
    return np.asarray(np.where(valid & np.isfinite(result), result, np.nan), dtype=np.float64)


def _wind_shear_kernel(
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    *,
    lower_height: float,
    upper_height: float,
) -> npt.NDArray[np.float64]:
    """Evaluate shear without per-block logging or memory instrumentation."""
    lower = _as_float_array(lower)
    upper = _as_float_array(upper)
    valid = np.isfinite(lower) & np.isfinite(upper) & (lower > 0) & (upper > 0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        # Use log1p near equality; log differences avoid ratios at extreme speeds.
        difference = np.asarray(upper - lower)
        near = np.abs(difference) <= 0.5 * lower
        np.divide(difference, lower, out=difference, where=near)
        result = np.where(near, np.log1p(difference), np.log(upper) - np.log(lower))
        result = result / np.log(upper_height / lower_height)
    return np.asarray(np.where(valid & np.isfinite(result), result, np.nan), dtype=np.float64)


@overload
def hub_height_wind_speed(  # pyright: ignore[reportOverlappingOverload]
    wind_speed: xr.DataArray,
    reference_height: _Height,
    hub_height: _Height,
    *,
    method: Literal["power", "log"] = "power",
    alpha: npt.ArrayLike | xr.DataArray = 1 / 7,
    roughness_length: _Height | None = None,
) -> xr.DataArray: ...


@overload
def hub_height_wind_speed(
    wind_speed: npt.ArrayLike,
    reference_height: _Height,
    hub_height: _Height,
    *,
    method: Literal["power", "log"] = "power",
    alpha: npt.ArrayLike = 1 / 7,
    roughness_length: _Height | None = None,
) -> npt.NDArray[np.float64]: ...


def hub_height_wind_speed(
    wind_speed: npt.ArrayLike | xr.DataArray,
    reference_height: _Height,
    hub_height: _Height,
    *,
    method: Literal["power", "log"] = "power",
    alpha: npt.ArrayLike | xr.DataArray = 1 / 7,
    roughness_length: _Height | None = None,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Extrapolate wind speed, a climate-driven resource proxy, not generation.

    The power law is ``v(h) = v_ref * (h / h_ref)**alpha`` (C3S wind-profile
    model; also ``xclim.indices.wind_profile``). Its 1/7 default is a generic
    approximation, not a site- or stability-calibrated exponent. Supply a
    measured exponent from :func:`wind_shear_exponent` when available.

    The neutral-stability logarithmic law is
    ``v(h) = v_ref * log(h / z0) / log(h_ref / z0)``, with zero displacement
    height, as in ``windpowerlib.wind_speed.logarithmic_profile`` with
    ``obstacle_height=0``. Neither profile models terrain, obstacles, bias
    adjustment, or non-neutral atmospheric stability.

    DataArray input uses the stable adapter, preserving coordinates and adding
    registry CF metadata and a scalar ``height`` coordinate in metres. Wind
    units must be m s⁻¹ or absent. For the power law, ``alpha`` must be a
    scalar or DataArray with exactly matching shared coordinate labels.
    Dask arrays remain lazy; all dimensions may have multiple chunks.

    Args:
        wind_speed: Reference wind speed in m s⁻¹; scalar or array. NaN and
            masked values are missing; negative or infinite speeds yield NaN.
        reference_height: Finite positive reference height in metres.
        hub_height: Finite positive target height in metres; may be lower
            than the reference height.
        method: ``"power"`` or ``"log"``.
        alpha: Dimensionless power-law exponent, broadcasting with wind speed.
            Finite negative exponents are retained for inverted profiles;
            missing or infinite exponents yield NaN. Used only for ``"power"``.
        roughness_length: Required positive roughness length z0 in metres for
            ``"log"``; both heights must exceed it. Not used for ``"power"``.

    Returns:
        Float64 array with the broadcast shape (0-D for scalar inputs), in
        m s⁻¹. Calm winds remain zero for valid parameters. Missing inputs or
        non-finite results yield NaN; underflow rounds to zero.

    Raises:
        InvalidArgumentError: If method, heights, roughness, or shapes are invalid.
        InputTypeError: If wind speed or the power exponent is non-numeric.

    References:
        C3S, Global Climate and Energy Indicators: Wind Speed Vertical
        Extrapolation, https://doi.org/10.24381/3bb607bd.
        Independent numerical comparisons: ``tests/fixture/energy_wind``;
        these are not validation against observed hub-height wind.

    Example:
        >>> from climate_indices import energy
        >>> energy.hub_height_wind_speed([0.0, 5.0], 10.0, 100.0).round(3)
        array([0.   , 6.947])
    """
    if not isinstance(method, str) or method not in ("power", "log"):
        raise InvalidArgumentError(
            "method must be 'power' or 'log'.", argument_name="method", argument_value=str(method)
        )
    reference_height = _positive_height(reference_height, "reference_height")
    hub_height = _positive_height(hub_height, "hub_height")
    if method == "log":
        if roughness_length is None:
            raise InvalidArgumentError("The log profile requires roughness_length.", argument_name="roughness_length")
        roughness_length = _positive_height(roughness_length, "roughness_length")
        if reference_height <= roughness_length or hub_height <= roughness_length:
            raise InvalidArgumentError("Both heights must exceed roughness_length.", argument_name="roughness_length")
    if isinstance(wind_speed, xr.DataArray):
        if method == "power" and isinstance(alpha, xr.DataArray):
            units = alpha.attrs.get("units")
            if units is not None and (
                not isinstance(units, str) or units.strip().lower() not in {"1", "dimensionless"}
            ):
                raise InvalidArgumentError(
                    "alpha must be dimensionless (units '1' or absent).", argument_name="alpha.attrs['units']"
                )

        def kernel(speed: npt.ArrayLike, exponent: npt.ArrayLike = 1 / 7) -> npt.NDArray[np.float64]:
            return _hub_height_kernel(
                speed, exponent if method == "power" else None, reference_height, hub_height, roughness_length
            )

        xarray_inputs = (alpha,) if method == "power" else ()
        xarray_result = apply_elementwise(
            kernel,
            "hub_height_wind_speed",
            wind_input(wind_speed, "wind_speed"),
            *xarray_inputs,
            calculation_metadata={
                "reference_height": reference_height,
                "hub_height": hub_height,
                "method": method,
                **({"roughness_length": roughness_length} if method == "log" else {}),
            },
        )
        return scalar_coordinate(xarray_result, "height", hub_height, standard_name="height", units="m", positive="up")
    if method == "power" and isinstance(alpha, xr.DataArray):
        raise InputTypeError(
            "wind_speed must be an xr.DataArray when alpha is an xr.DataArray.",
            expected_type=xr.DataArray,
            actual_type=type(wind_speed),
        )
    wind = _as_float_array(wind_speed)
    exponent = _as_float_array(alpha) if method == "power" else None
    if exponent is not None:
        try:
            wind, exponent = np.broadcast_arrays(wind, exponent)
        except ValueError as exc:
            raise InvalidArgumentError("wind_speed and alpha must broadcast together.", argument_name="alpha") from exc
    log = _logger.bind(
        index_type="hub_height_wind_speed", method=method, input_shape=wind.shape, input_elements=wind.size
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    try:
        inputs = (wind,) if exponent is None else (wind, exponent)
        # Budget for the factor, output buffers, and temporary masks.
        memory_metrics = check_large_array_memory(
            *inputs, extra_bytes=wind.size * ((3 if exponent is not None else 2) * wind.itemsize + 3)
        )
        result = _hub_height_kernel(wind, exponent, reference_height, hub_height, roughness_length)
        log.info(
            "calculation_completed",
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            output_shape=result.shape,
            **(memory_metrics or {}),
        )
        return result
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise


@overload
def wind_shear_exponent(  # pyright: ignore[reportOverlappingOverload]
    wind_speed_lower: xr.DataArray,
    wind_speed_upper: xr.DataArray,
    *,
    lower_height: _Height = 10.0,
    upper_height: _Height = 100.0,
) -> xr.DataArray: ...


@overload
def wind_shear_exponent(
    wind_speed_lower: npt.ArrayLike,
    wind_speed_upper: npt.ArrayLike,
    *,
    lower_height: _Height = 10.0,
    upper_height: _Height = 100.0,
) -> npt.NDArray[np.float64]: ...


def wind_shear_exponent(
    wind_speed_lower: npt.ArrayLike | xr.DataArray,
    wind_speed_upper: npt.ArrayLike | xr.DataArray,
    *,
    lower_height: _Height = 10.0,
    upper_height: _Height = 100.0,
) -> npt.NDArray[np.float64] | xr.DataArray:
    """Derive power-law shear from two levels, not actual energy generation.

    Compute ``alpha = log(v_upper / v_lower) / log(h_upper / h_lower)``
    elementwise, as in the C3S wind-profile formula
    (https://doi.org/10.24381/3bb607bd). Unlike C3S's climatological exponent,
    this is a per-sample diagnostic; recovering the upper wind from the same
    pair is an algebraic consistency check, not independent validation.

    When either wind input is a DataArray, both must be DataArrays with
    exactly matching shared coordinate labels and m s⁻¹ or absent units.
    This stable path returns a dimensionless DataArray with output provenance.
    Dask arrays remain lazy; all dimensions may have multiple chunks.

    Args:
        wind_speed_lower: Lower-level speed in m s⁻¹; scalar or array.
        wind_speed_upper: Upper-level speed in m s⁻¹, broadcasting with the
            lower wind. Both speeds must be finite and strictly positive.
        lower_height: Finite positive height in metres; defaults to ERA5's 10 m.
        upper_height: Finite height above lower_height; defaults to ERA5's 100 m.

    Returns:
        Dimensionless float64 array with the broadcast shape (0-D for scalar
        inputs). Calm, negative, infinite, NaN, or masked winds yield NaN.
        Positive inverted profiles retain negative shear; equal positive
        speeds yield zero. Inputs are not clipped or modified.

    Raises:
        InvalidArgumentError: If heights or array shapes are invalid.
        InputTypeError: If wind speeds are non-numeric.
    """
    lower_height = _positive_height(lower_height, "lower_height")
    upper_height = _positive_height(upper_height, "upper_height")
    if upper_height <= lower_height:
        raise InvalidArgumentError("upper_height must exceed lower_height.", argument_name="upper_height")
    if isinstance(wind_speed_lower, xr.DataArray) or isinstance(wind_speed_upper, xr.DataArray):
        return apply_elementwise(
            _wind_shear_kernel,
            "wind_shear_exponent",
            wind_input(wind_speed_lower, "wind_speed_lower"),
            wind_input(wind_speed_upper, "wind_speed_upper"),
            lower_height=lower_height,
            upper_height=upper_height,
            calculation_metadata={"lower_height": lower_height, "upper_height": upper_height},
        )
    lower = _as_float_array(wind_speed_lower)
    upper = _as_float_array(wind_speed_upper)
    try:
        lower, upper = np.broadcast_arrays(lower, upper)
    except ValueError as exc:
        raise InvalidArgumentError("Wind speeds must broadcast together.", argument_name="wind_speed_upper") from exc
    log = _logger.bind(index_type="wind_shear_exponent", input_shape=lower.shape, input_elements=lower.size)
    log.info("calculation_started")
    t0 = time.perf_counter()
    try:
        # Budget for the difference, logarithms, output buffers, and temporary masks.
        memory_metrics = check_large_array_memory(lower, upper, extra_bytes=lower.size * (5 * lower.itemsize + 3))
        result = _wind_shear_kernel(lower, upper, lower_height=lower_height, upper_height=upper_height)
        log.info(
            "calculation_completed",
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            output_shape=result.shape,
            **(memory_metrics or {}),
        )
        return result
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise


def _validated_capacity_factor_thresholds(cut_in: float, rated: float, cut_out: float) -> tuple[float, float, float]:
    """Return finite, ordered generic-curve thresholds."""
    thresholds = []
    for value, name in ((cut_in, "cut_in"), (rated, "rated"), (cut_out, "cut_out")):
        raw = np.asarray(value)
        if raw.ndim != 0 or raw.dtype.kind not in "iuf" or np.ma.is_masked(value) or not np.isfinite(raw):
            raise InvalidArgumentError(f"{name} must be a finite numeric scalar.", argument_name=name)
        thresholds.append(float(raw))
    cut_in, rated, cut_out = thresholds
    if not all(np.isfinite(thresholds)) or not 0 <= cut_in < rated < cut_out:
        raise InvalidArgumentError("Require 0 <= cut_in < rated < cut_out.", argument_name="cut_in,rated,cut_out")
    return cut_in, rated, cut_out


def _validated_capacity_factor_table(
    power_curve: tuple[npt.ArrayLike, npt.ArrayLike],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return increasing speed knots and bounded capacity factors."""
    if not isinstance(power_curve, tuple) or len(power_curve) != 2:
        raise InvalidArgumentError("power_curve must be 'generic' or a pair of arrays.", argument_name="power_curve")
    knots, factors = (_as_float_array(part) for part in power_curve)
    if (
        knots.ndim != 1
        or knots.size < 2
        or knots.shape != factors.shape
        or not np.all(np.isfinite(knots) & (knots >= 0))
        or not np.all(knots[1:] > knots[:-1])
        or not np.all(np.isfinite(factors) & (factors >= 0) & (factors <= 1))
    ):
        raise InvalidArgumentError(
            "power_curve requires increasing finite speeds and factors in [0, 1].", argument_name="power_curve"
        )
    return knots, factors


def wind_capacity_factor(
    hub_wind_speed: npt.ArrayLike,
    power_curve: Literal["generic"] | tuple[npt.ArrayLike, npt.ArrayLike],
    *,
    cut_in: float = 3.5,
    rated: float = 13.0,
    cut_out: float = 25.0,
    air_density: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Convert hub-height wind to a climate-driven resource proxy, not generation.

    A capacity factor is a dimensionless fraction of nominal turbine power.
    This reference transform excludes wakes, availability, curtailment, and
    other plant losses. It does not estimate actual generation.

    Choose ``power_curve="generic"`` explicitly for the idealized cubic curve
    documented by ``xclim.indices.wind_power_potential`` (Tobin et al., 2018).
    For speed v, the factor is zero below cut-in, then
    ``(v**3 - cut_in**3) / (rated**3 - cut_in**3)`` below rated speed, then
    one below cut-out. At cut-out and above, it is zero. The default speeds
    are xclim's generic 3.5, 13, and 25 m s⁻¹, not a specific turbine.

    For a tabulated curve, supply ``(speeds, capacity_factors)``. Interpolation
    is linear between knots, includes both endpoints, and returns zero
    outside the table. A decreasing segment is permitted for shutdown.
    Supply the complete operating range; no additional cut-out is inferred.

    When air density is supplied, evaluate either curve at
    ``v * (air_density / 1.225)**(1/3)``. Thus density correction also changes
    cut-in and cut-out region membership, as in xclim. The curve must refer
    to 1.225 kg m⁻³. Do not correct an already density-adjusted curve again.
    Apply the curve before time averaging: a curve of mean wind is not the
    mean capacity factor. This function provides only the NumPy API.

    Args:
        hub_wind_speed: Scalar or array in m s⁻¹, such as the output of
            :func:`hub_height_wind_speed`. Negative and non-finite speeds
            yield NaN. NaN and masked values are missing.
        power_curve: Required explicit ``"generic"`` choice or a pair of
            equal-length, finite 1-D arrays with at least two knots. Speeds
            must be non-negative and strictly increasing; capacity factors
            must lie in [0, 1]. Normalize power by rated power before calling.
        cut_in: Non-negative scalar speed in m s⁻¹, for the generic curve only.
        rated: Finite scalar speed above cut_in, for the generic curve only.
        cut_out: Finite scalar speed above rated, for the generic curve only.
        air_density: Optional density in kg m⁻³, broadcasting with wind.
            Non-positive, non-finite, or masked densities yield NaN.
            None leaves wind unchanged.

    Returns:
        Float64 array in [0, 1] with the broadcast shape, or a 0-D array for
        scalar inputs. Missing or non-finite corrected speeds yield NaN.
        Inputs remain unchanged, and underflow rounds to zero.

    Raises:
        InvalidArgumentError: If curve parameters or broadcast shapes are invalid.
        InputTypeError: If arrays are non-numeric, or wind or density is a DataArray.

    References:
        https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.wind_power_potential
        The C3S turbine curves are not matched numerical references; see
        ``docs/research/energy-oracle-survey.md``.

    Example:
        >>> from climate_indices import energy
        >>> energy.wind_capacity_factor([0, 13, 25], "generic")
        array([0., 1., 0.])
    """
    log = _logger.bind(index_type="wind_capacity_factor")
    try:
        table = None
        if isinstance(power_curve, str):
            if power_curve != "generic":
                raise InvalidArgumentError(
                    "power_curve must be 'generic' or a pair of arrays.", argument_name="power_curve"
                )
            cut_in, rated, cut_out = _validated_capacity_factor_thresholds(cut_in, rated, cut_out)
        else:
            table = _validated_capacity_factor_table(power_curve)
        if isinstance(hub_wind_speed, xr.DataArray) or isinstance(air_density, xr.DataArray):
            raise InputTypeError(
                "wind_capacity_factor supports NumPy inputs only.", expected_type=np.ndarray, actual_type=xr.DataArray
            )
        wind = _as_float_array(hub_wind_speed)
        density = _as_float_array(1.225 if air_density is None else air_density)
        try:
            wind, density = np.broadcast_arrays(wind, density)
        except ValueError as exc:
            raise InvalidArgumentError(
                "Wind and air_density must broadcast together.", argument_name="air_density"
            ) from exc
        log = log.bind(input_shape=wind.shape, input_elements=wind.size)
        log.info("calculation_started")
        t0 = time.perf_counter()
        memory_metrics = check_large_array_memory(wind, density, extra_bytes=wind.size * (7 * wind.itemsize + 3))
        valid = np.isfinite(wind) & (wind >= 0) & np.isfinite(density) & (density > 0)
        with np.errstate(all="ignore"):
            # Separate cube roots avoid underflow in the density ratio.
            speed = wind if air_density is None else wind * (np.cbrt(density) / np.cbrt(1.225))
            if table is None:
                bounded = np.clip(speed, cut_in, rated)
                x, y = bounded / rated, cut_in / rated
                # Factor the differences of cubes to avoid overflow and cancellation.
                ramp = ((bounded - cut_in) / (rated - cut_in)) * (x * x + x * y + y * y) / (1 + y + y * y)
                result = np.where(speed < cut_out, np.clip(ramp, 0, 1), 0.0)
            else:
                knots, factors = table
                upper = np.clip(np.searchsorted(knots, speed, side="right"), 1, knots.size - 1)
                # Interpolate by a bounded fraction; np.interp's slope can overflow.
                fraction = (speed - knots[upper - 1]) / (knots[upper] - knots[upper - 1])
                result = (1 - fraction) * factors[upper - 1] + fraction * factors[upper]
                result = np.where((speed >= knots[0]) & (speed <= knots[-1]), result, 0.0)
        result = np.asarray(np.where(valid & np.isfinite(speed), result, np.nan), dtype=np.float64)
        log.info(
            "calculation_completed",
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            output_shape=result.shape,
            **(memory_metrics or {}),
        )
        return result
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise

"""Keetch-Byram Drought Index (KBDI)."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import (
    CoordinateValidationError,
    DataShapeError,
    InputAlignmentWarning,
    InvalidArgumentError,
)
from climate_indices.fire._common import (
    _as_float_array,
    _static_spatial_array,
    _validate_recurrence_options,
)
from climate_indices.fire._units import (
    _convert_precipitation_units,
    _convert_temperature_units,
    _validate_daily_time_coordinate,
)
from climate_indices.logging_config import get_logger
from climate_indices.performance import check_large_array_memory
from climate_indices.xarray_adapter import (
    InputType,
    _build_output_attrs,
    _validate_dask_chunks,
    _validate_time_dimension,
    _validate_time_monotonicity,
    detect_input_type,
)

# retrieve structlog logger for this module
_logger = get_logger(__name__)


# Keetch and Byram (1968) Equation 18, corrected by Alexander (1990), is
# evaluated in metric units. One KBDI point is one hundredth of an inch.
_KBDI_MAX_MM = 203.2
_KBDI_MM_PER_POINT = 0.254
_KBDI_RAIN_THRESHOLD_MM = 5.08
_KBDI_DRYING_TEMPERATURE_CELSIUS = 10.0
_KBDI_MINIMUM_MEAN_ANNUAL_RECORD_DAYS = 30 * 365

_ARG_INITIAL_STATE_KBDI = "initial_state.kbdi"


@dataclass(frozen=True)
class KBDIState:
    """State needed to resume a KBDI recurrence.

    ``kbdi`` and ``wet_spell_precipitation`` use ``units``. A
    ``trailing_gap_days`` value of ``None`` means no valid day has started the
    recurrence. For spatial arrays, ``-1`` marks individual cells that have
    not started yet. A NaN ``kbdi`` is only valid where ``trailing_gap_days``
    shows that a gap has started the cell; a not-started cell holds a number.
    """

    kbdi: npt.NDArray[np.float64]
    wet_spell_precipitation: npt.NDArray[np.float64]
    trailing_gap_days: npt.NDArray[np.int64] | None
    units: Literal["metric", "imperial"] = "metric"


@dataclass(frozen=True)
class KBDIResult:
    """KBDI values and final state returned by :func:`kbdi`.

    ``values`` is an ``xr.DataArray`` when :func:`kbdi` was called with xarray
    input, else a NumPy array. ``state`` is always NumPy: state types stay
    algorithm-specific frozen dataclasses, never xarray objects, per
    ``docs/adr/0006-fire-recursive-state-and-execution.md``.
    """

    values: npt.NDArray[np.float64] | xr.DataArray
    state: KBDIState


def _kbdi_initial_value(
    initial_kbdi: npt.ArrayLike,
    spatial_shape: tuple[int, ...],
    maximum: float,
) -> npt.NDArray[np.float64]:
    """Validate a caller-supplied seed KBDI and broadcast it to the spatial shape."""
    kbdi_value = _static_spatial_array(initial_kbdi, spatial_shape, "initial_kbdi")
    if np.any(~np.isfinite(kbdi_value)) or np.any(kbdi_value < 0.0) or np.any(kbdi_value > maximum):
        raise InvalidArgumentError(
            f"initial_kbdi must be finite and within [0, {maximum:g}].",
            argument_name="initial_kbdi",
            argument_value="non-finite or outside the valid KBDI range",
            valid_values=f"[0, {maximum:g}]",
        )
    return kbdi_value


def _kbdi_state_arrays(
    state: KBDIState,
    spatial_shape: tuple[int, ...],
    units: Literal["metric", "imperial"],
    maximum: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Validate and copy a state into native KBDI units."""
    if not isinstance(state, KBDIState):
        raise InvalidArgumentError(
            "initial_state must be a KBDIState.",
            argument_name="initial_state",
            argument_value=type(state).__name__,
            valid_values="KBDIState",
        )
    if state.units != units:
        raise InvalidArgumentError(
            "initial_state units must match units.",
            argument_name="initial_state.units",
            argument_value=state.units,
            valid_values=units,
        )

    kbdi_value = _static_spatial_array(state.kbdi, spatial_shape, _ARG_INITIAL_STATE_KBDI)
    wet_spell = _static_spatial_array(
        state.wet_spell_precipitation,
        spatial_shape,
        "initial_state.wet_spell_precipitation",
    )
    if (
        np.any(~np.isfinite(kbdi_value) & ~np.isnan(kbdi_value))
        or np.any(kbdi_value > maximum)
        or np.any(kbdi_value < 0.0)
    ):
        raise InvalidArgumentError(
            f"initial_state.kbdi must be NaN or within [0, {maximum:g}].",
            argument_name=_ARG_INITIAL_STATE_KBDI,
            argument_value="values outside the valid KBDI range",
            valid_values=f"NaN or [0, {maximum:g}]",
        )
    if np.any(~np.isfinite(wet_spell)) or np.any(wet_spell < 0.0):
        raise InvalidArgumentError(
            "initial_state.wet_spell_precipitation must be finite and non-negative.",
            argument_name="initial_state.wet_spell_precipitation",
            argument_value="non-finite or negative value",
            valid_values="Finite values greater than or equal to zero",
        )

    if state.trailing_gap_days is None:
        trailing_gap_days = np.full(spatial_shape, -1, dtype=np.int64)
    else:
        trailing = _static_spatial_array(state.trailing_gap_days, spatial_shape, "initial_state.trailing_gap_days")
        if np.any(~np.isfinite(trailing)) or np.any(trailing < -1) or np.any(trailing != np.floor(trailing)):
            raise InvalidArgumentError(
                "initial_state.trailing_gap_days must contain integers greater than or equal to -1.",
                argument_name="initial_state.trailing_gap_days",
                argument_value="non-integral or less than -1 value",
                valid_values="-1 or a non-negative integer",
            )
        trailing_gap_days = trailing.astype(np.int64)

    if np.any(np.isnan(kbdi_value) & (trailing_gap_days < 0)):
        raise InvalidArgumentError(
            "initial_state.kbdi may be NaN only where trailing_gap_days shows a gap has started.",
            argument_name=_ARG_INITIAL_STATE_KBDI,
            argument_value="NaN KBDI where initial_state.trailing_gap_days is -1 or None",
            valid_values="A finite KBDI in a not-started cell; NaN only after a gap has started the cell",
        )

    return kbdi_value, wet_spell, trailing_gap_days


def _validate_kbdi_configuration(
    *,
    units: Literal["metric", "imperial"],
    nan_policy: Literal["propagate", "bridge"],
    max_gap_days: int,
    spin_up: int,
    initial_kbdi: npt.ArrayLike | xr.DataArray | None,
    initial_state: KBDIState | None,
) -> None:
    """Validate the configuration shared by the NumPy and xarray paths.

    Called before dispatch: the xarray path must reject invalid configuration
    eagerly instead of returning a lazy, metadata-bearing result that fails
    only when evaluated.
    """
    if units not in ("metric", "imperial"):
        raise InvalidArgumentError(
            "units must be 'metric' or 'imperial'.",
            argument_name="units",
            argument_value=str(units),
            valid_values="'metric', 'imperial'",
        )
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_kbdi", initial_kbdi, initial_state)


@overload
def kbdi(
    precipitation: xr.DataArray,
    maximum_temperature: xr.DataArray,
    mean_annual_precipitation: npt.ArrayLike | xr.DataArray | None = None,
    *,
    units: Literal["metric", "imperial"] = "metric",
    initial_kbdi: npt.ArrayLike | xr.DataArray | None = None,
    initial_state: KBDIState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
    time_dim: str = "time",
) -> xr.DataArray | KBDIResult: ...


@overload
def kbdi(
    precipitation: npt.ArrayLike,
    maximum_temperature: npt.ArrayLike,
    mean_annual_precipitation: npt.ArrayLike | None = None,
    *,
    units: Literal["metric", "imperial"] = "metric",
    initial_kbdi: npt.ArrayLike | None = None,
    initial_state: KBDIState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
    time_dim: str = "time",
) -> npt.NDArray[np.float64] | KBDIResult: ...


def kbdi(
    precipitation: npt.ArrayLike | xr.DataArray,
    maximum_temperature: npt.ArrayLike | xr.DataArray,
    mean_annual_precipitation: npt.ArrayLike | xr.DataArray | None = None,
    *,
    units: Literal["metric", "imperial"] = "metric",
    initial_kbdi: npt.ArrayLike | xr.DataArray | None = None,
    initial_state: KBDIState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
    time_dim: str = "time",
) -> npt.NDArray[np.float64] | KBDIResult | xr.DataArray:
    """Compute the Keetch-Byram Drought Index (KBDI).

    This function accepts both NumPy arrays and xarray DataArrays. Type
    checkers narrow the return type based on the input type.

    .. warning:: **Beta Feature (xarray path only)** -- When called with
       ``xr.DataArray`` input, this function uses the beta xarray adapter
       layer: per-call CF metadata resolution (``kbdi`` vs. ``kbdi_imperial``
       by ``units``), CF ``units``-attribute unit inference, and Dask
       spatial-chunk parallelism with a required single ``time`` chunk. The
       NumPy array interface and underlying computation are stable.

    The implementation evaluates the corrected continuous Equation 18 of
    Keetch and Byram (1968), using Alexander's (1990) corrected 8.30
    constant. Consecutive positive-rain days form one wet spell: only rain
    above its first 5.08 mm reduces KBDI. Days below 10 C (50 F) do not add a
    drought factor, as the source states drought development requires daily
    maxima of 50 F or higher.

    The source's open choices are resolved here as: continuous values rather
    than the 1968 table quantization, the caller's exact mean annual
    precipitation rather than a table category, clamps at zero on rain and at
    the scale maximum on drying, and the missing-day policy of
    ``docs/adr/0007-fire-missing-data-policy.md``.

    Args:
        precipitation: Daily precipitation, time-first. Metric values are mm;
            imperial values are inches. Must be finite or NaN: infinity is
            rejected rather than treated as a missing day.
        maximum_temperature: Daily maximum temperature, time-first. Metric
            values are degrees Celsius; imperial values are degrees Fahrenheit.
            Must be finite or NaN.
        mean_annual_precipitation: Long-term mean annual precipitation, scalar
            or spatial field. If omitted, it is derived from at least 10,950
            finite daily precipitation values per cell as their mean times
            365.25; supply climatology instead when it is available.
        units: ``"metric"`` returns mm in [0, 203.2]; ``"imperial"`` returns
            hundredths of an inch in [0, 800].
        initial_kbdi: Seed KBDI in ``units``. ``None`` selects zero. Cannot be
            combined with ``initial_state``.
        initial_state: State returned by an earlier call with the same units.
        return_state: Return :class:`KBDIResult` with final state.
        spin_up: Number of leading input days to compute but omit from output.
        nan_policy: ``"propagate"`` poisons a started recurrence at a missing
            day; ``"bridge"`` skips gaps up to ``max_gap_days``.
        max_gap_days: Maximum bridged consecutive missing days. Must be zero
            for ``"propagate"`` and positive for ``"bridge"``.
        time_dim: Name of the time dimension. Only used for xarray inputs.

    Returns:
        KBDI with the same time-first shape as the broadcast weather inputs,
        less ``spin_up`` leading days. Returns ``KBDIResult`` when
        ``return_state`` is true. For xarray input, ``values`` is a
        ``DataArray`` carrying CF metadata from the ``kbdi``/``kbdi_imperial``
        registry entry (chosen by ``units``) and ``state`` stays plain NumPy
        per :doc:`../docs/adr/0006-fire-recursive-state-and-execution`.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, or physical
            precipitation inputs are invalid.
        CoordinateValidationError: xarray input only -- if the time dimension
            is missing, an attached time coordinate is non-monotonic or not
            consecutive daily, the time dimension is split across multiple Dask
            chunks, the inputs' non-time coordinates do not align, or
            precipitation/maximum_temperature share no overlapping time steps.

    Notes:
        xarray-only: ``precipitation`` and ``maximum_temperature`` must be the
        same type (both NumPy or both ``xr.DataArray``); a CF ``units``
        attribute on either is converted to the scale ``units`` selects (mm/inch
        for precipitation, including ``kg m-2 s-1``; Celsius/Fahrenheit/Kelvin
        for temperature). An absent attribute is assumed to already match the
        selected scale; an unrecognized one raises ``InvalidArgumentError``.
        An attributed ``mean_annual_precipitation`` is converted the same way,
        accepting annual totals (``mm``, ``inch``, ``mm year-1``, ...) but not
        daily or flux rates. An attached time coordinate must hold consecutive
        daily observations; a dimension-only time axis is aligned positionally.
        Dask-backed input parallelizes over spatial chunks with the ``time``
        dimension required to be a single chunk.
    """
    if isinstance(precipitation, xr.DataArray) != isinstance(maximum_temperature, xr.DataArray):
        raise TypeError(
            "precipitation and maximum_temperature must be the same type. "
            f"Got precipitation={type(precipitation).__name__}, "
            f"maximum_temperature={type(maximum_temperature).__name__}. "
            "Convert both to the same type (both numpy arrays or both xr.DataArray)."
        )
    _validate_kbdi_configuration(
        units=units,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
        spin_up=spin_up,
        initial_kbdi=initial_kbdi,
        initial_state=initial_state,
    )
    if detect_input_type(precipitation) != InputType.NUMPY:
        # narrow for mypy: detect_input_type raises for anything but NUMPY/XARRAY
        assert isinstance(precipitation, xr.DataArray)
        assert isinstance(maximum_temperature, xr.DataArray)
        return _kbdi_xarray(
            precipitation,
            maximum_temperature,
            mean_annual_precipitation,
            units=units,
            initial_kbdi=initial_kbdi,
            initial_state=initial_state,
            return_state=return_state,
            spin_up=spin_up,
            nan_policy=nan_policy,
            max_gap_days=max_gap_days,
            time_dim=time_dim,
        )

    precipitation_array = _as_float_array(precipitation)
    temperature_array = _as_float_array(maximum_temperature)
    try:
        precipitation_array, temperature_array = np.broadcast_arrays(precipitation_array, temperature_array)
    except ValueError as exc:
        raise InvalidArgumentError(
            "precipitation and maximum_temperature must broadcast to a common time-first shape.",
            argument_name="precipitation/maximum_temperature",
            argument_value=f"shapes {precipitation_array.shape}, {temperature_array.shape}",
            valid_values="Arrays broadcastable to a common time-first shape",
        ) from exc
    if precipitation_array.ndim == 0:
        raise DataShapeError(
            "KBDI weather inputs must include a time dimension.",
            expected_shape="(time, ...)",
            actual_shape=precipitation_array.shape,
        )
    if np.any(np.isfinite(precipitation_array) & (precipitation_array < 0.0)):
        raise InvalidArgumentError(
            "precipitation must be non-negative where finite.",
            argument_name="precipitation",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
        )
    if np.any(np.isinf(precipitation_array)) or np.any(np.isinf(temperature_array)):
        raise InvalidArgumentError(
            "precipitation and maximum_temperature must be finite or NaN: infinity is not a missing observation.",
            argument_name="precipitation/maximum_temperature",
            argument_value="infinite value",
            valid_values="Finite values or NaN",
        )

    spatial_shape = precipitation_array.shape[1:]
    # A single time series has no spatial axis; give the recurrence one so
    # every daily update uses the same vectorized boolean indexing.
    internal_spatial_shape = spatial_shape if spatial_shape else (1,)
    precipitation_array = precipitation_array.reshape(precipitation_array.shape[0], *internal_spatial_shape)
    temperature_array = temperature_array.reshape(precipitation_array.shape)
    maximum = 800.0 if units == "imperial" else _KBDI_MAX_MM
    if mean_annual_precipitation is None:
        finite_precipitation = np.isfinite(precipitation_array)
        valid_days = np.count_nonzero(finite_precipitation, axis=0)
        if np.any(valid_days < _KBDI_MINIMUM_MEAN_ANNUAL_RECORD_DAYS):
            raise InvalidArgumentError(
                "Deriving mean_annual_precipitation requires at least 10,950 finite daily values per cell.",
                argument_name="mean_annual_precipitation",
                argument_value=f"minimum finite days {int(np.min(valid_days))}",
                valid_values="An explicit climatology or at least 10,950 finite daily values per cell",
            )
        mean_annual = np.sum(np.where(finite_precipitation, precipitation_array, 0.0), axis=0) / valid_days * 365.25
    else:
        mean_annual = _static_spatial_array(
            mean_annual_precipitation, internal_spatial_shape, "mean_annual_precipitation"
        )
    if np.any(np.isinf(mean_annual)) or np.any(np.isfinite(mean_annual) & (mean_annual <= 0.0)):
        raise InvalidArgumentError(
            "mean_annual_precipitation must be positive where finite.",
            argument_name="mean_annual_precipitation",
            argument_value="non-positive or infinite value",
            valid_values="Positive finite values or NaN",
        )

    if initial_state is None:
        if initial_kbdi is None:
            kbdi_value = np.zeros(internal_spatial_shape, dtype=np.float64)
        else:
            kbdi_value = _kbdi_initial_value(initial_kbdi, internal_spatial_shape, maximum)
        wet_spell = np.zeros(internal_spatial_shape, dtype=np.float64)
        trailing_gap_days = np.full(internal_spatial_shape, -1, dtype=np.int64)
    else:
        kbdi_value, wet_spell, trailing_gap_days = _kbdi_state_arrays(
            initial_state,
            internal_spatial_shape,
            units,
            maximum,
        )

    if units == "imperial":
        with np.errstate(over="ignore"):
            precipitation_array = precipitation_array * 25.4
            # the factor is grouped so the intermediate product cannot overflow
            temperature_array = (temperature_array - 32.0) * (5.0 / 9.0)
            mean_annual = mean_annual * 25.4
            kbdi_value = kbdi_value * _KBDI_MM_PER_POINT
            wet_spell = wet_spell * 25.4
        if (
            np.any(np.isinf(precipitation_array))
            or np.any(np.isinf(temperature_array))
            or np.any(np.isinf(mean_annual))
            or np.any(np.isinf(wet_spell))
        ):
            raise InvalidArgumentError(
                "Imperial inputs must be representable in metric units: the conversion overflows float64.",
                argument_name=(
                    "precipitation/maximum_temperature/mean_annual_precipitation/initial_state.wet_spell_precipitation"
                ),
                argument_value="finite value too large to convert to metric units",
                valid_values="Finite values that do not overflow the metric conversion",
            )

    log = _logger.bind(
        index_type="kbdi",
        input_shape=precipitation_array.shape,
        input_elements=precipitation_array.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(precipitation_array, temperature_array, mean_annual)

    try:
        n_days = precipitation_array.shape[0]
        # Spin-up days are evaluated for the state they leave behind but never
        # stored, so the output only allocates the days the caller receives.
        values = np.full((max(n_days - spin_up, 0), *internal_spatial_shape), np.nan, dtype=np.float64)
        static_valid = np.isfinite(mean_annual)
        started = trailing_gap_days >= 0
        poisoned = np.isnan(kbdi_value)

        for day in range(n_days):
            precipitation_day = precipitation_array[day]
            temperature_day = temperature_array[day]
            weather_valid = np.isfinite(precipitation_day) & np.isfinite(temperature_day)
            valid = weather_valid & static_valid
            # A cell whose static climatology is unavailable has no recurrence
            # to gap-manage: its output is NaN and its carried state is left
            # as it was, so a NaN climatology is never an elapsed missing day.
            missing_started = ~weather_valid & static_valid & (started | poisoned)

            if nan_policy == "propagate":
                kbdi_value[missing_started] = np.nan
                poisoned[missing_started] = True
                trailing_gap_days[missing_started] = np.maximum(trailing_gap_days[missing_started], 0) + 1
            else:
                next_gap_days = np.maximum(trailing_gap_days, 0) + 1
                over_gap_limit = missing_started & (next_gap_days > max_gap_days)
                kbdi_value[over_gap_limit] = np.nan
                poisoned[over_gap_limit] = True
                trailing_gap_days[missing_started] = next_gap_days[missing_started]

            active = valid & ~poisoned
            started[active] = True
            # a valid day is the return point's last day, so any earlier run is closed
            trailing_gap_days[valid & (started | poisoned)] = 0

            rainy = active & (precipitation_day > 0.0)
            prior_wet_spell = wet_spell.copy()
            event_total = prior_wet_spell + precipitation_day
            crossing_threshold = (
                rainy & (prior_wet_spell <= _KBDI_RAIN_THRESHOLD_MM) & (event_total > _KBDI_RAIN_THRESHOLD_MM)
            )
            continuing_wet_spell = rainy & (prior_wet_spell > _KBDI_RAIN_THRESHOLD_MM)
            net_rain = np.zeros(internal_spatial_shape, dtype=np.float64)
            net_rain[crossing_threshold] = event_total[crossing_threshold] - _KBDI_RAIN_THRESHOLD_MM
            net_rain[continuing_wet_spell] = precipitation_day[continuing_wet_spell]
            wet_spell[rainy] = event_total[rainy]
            wet_spell[active & ~rainy] = 0.0

            after_rain = kbdi_value.copy()
            after_rain[active] = np.maximum(0.0, kbdi_value[active] - net_rain[active])
            drying = np.zeros(internal_spatial_shape, dtype=np.float64)
            drought_day = active & (temperature_day >= _KBDI_DRYING_TEMPERATURE_CELSIUS) & (after_rain < _KBDI_MAX_MM)
            with np.errstate(over="ignore"):
                drying[drought_day] = (
                    (_KBDI_MAX_MM - after_rain[drought_day])
                    * (0.968 * np.exp(0.0875 * temperature_day[drought_day] + 1.5552) - 8.30)
                    / (1.0 + 10.88 * np.exp(-0.001736 * mean_annual[drought_day]))
                    * 1e-3
                )
            kbdi_value[active] = np.minimum(_KBDI_MAX_MM, after_rain[active] + np.maximum(drying[active], 0.0))
            if day >= spin_up:
                values[day - spin_up] = np.where(active, kbdi_value, np.nan)

        state_gap_days: npt.NDArray[np.int64] | None
        if np.any(started | poisoned):
            state_gap_days = trailing_gap_days.reshape(spatial_shape).copy()
        else:
            state_gap_days = None
        if units == "imperial":
            returned_values = values / _KBDI_MM_PER_POINT
            returned_kbdi = kbdi_value / _KBDI_MM_PER_POINT
            returned_wet_spell = wet_spell / 25.4
        else:
            returned_values = values
            returned_kbdi = kbdi_value
            returned_wet_spell = wet_spell

        result = returned_values.reshape(-1, *spatial_shape)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=result.shape,
            **(memory_metrics or {}),
        )
        if not return_state:
            return result
        return KBDIResult(
            values=result,
            state=KBDIState(
                kbdi=returned_kbdi.reshape(spatial_shape).copy(),
                wet_spell_precipitation=returned_wet_spell.reshape(spatial_shape).copy(),
                trailing_gap_days=state_gap_days,
                units=units,
            ),
        )
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def _kbdi_xarray(
    precipitation: xr.DataArray,
    maximum_temperature: xr.DataArray,
    mean_annual_precipitation: npt.ArrayLike | xr.DataArray | None,
    *,
    units: Literal["metric", "imperial"],
    initial_kbdi: npt.ArrayLike | xr.DataArray | None,
    initial_state: KBDIState | None,
    return_state: bool,
    spin_up: int,
    nan_policy: Literal["propagate", "bridge"],
    max_gap_days: int,
    time_dim: str,
) -> xr.DataArray | KBDIResult:
    """xarray dispatch for :func:`kbdi`. See :func:`kbdi` for the full contract.

    Resolves the ``kbdi``/``kbdi_imperial`` CF registry entry from ``units`` at
    call time (the KBDI-specific problem named in
    ``docs/design/fire-subsystem.md``: the same function selects between two
    registry entries depending on a runtime argument, which a decoration-time
    ``cf_metadata`` dict cannot do). Delegates the actual recurrence to
    :func:`kbdi`'s NumPy path via :func:`xarray.apply_ufunc`, one call per
    Dask spatial chunk with the full ``time`` axis, since :func:`kbdi` already
    vectorizes over an arbitrary spatial shape internally -- unlike
    :func:`~climate_indices.xarray_adapter.pet_hargreaves`, this does not need
    ``vectorize=True`` per-cell looping.
    """
    precip_da = precipitation
    temp_da = maximum_temperature

    _validate_time_dimension(precip_da, time_dim)
    _validate_time_dimension(temp_da, time_dim)
    # a dimension-only time axis carries no cadence metadata: xarray aligns it
    # positionally, so monotonicity and daily checks apply only to real coords
    for data in (precip_da, temp_da):
        if time_dim in data.coords:
            _validate_time_monotonicity(data.coords[time_dim])
            _validate_daily_time_coordinate(data, time_dim)

    shared_spatial_dims = [str(dim) for dim in precip_da.dims if dim in temp_da.dims and dim != time_dim]
    precip_aligned, temp_aligned = xr.align(precip_da, temp_da, join="inner")
    for dim in sorted(shared_spatial_dims):
        if precip_aligned.sizes[dim] != precip_da.sizes[dim] or temp_aligned.sizes[dim] != temp_da.sizes[dim]:
            raise CoordinateValidationError(
                message=(
                    f"Input alignment dropped coordinates along non-time dimension '{dim}': "
                    f"precipitation had {precip_da.sizes[dim]}, maximum_temperature had "
                    f"{temp_da.sizes[dim]}; after the inner join they have "
                    f"{precip_aligned.sizes[dim]} and {temp_aligned.sizes[dim]}. "
                    "Subset or align the inputs explicitly; KBDI never reduces spatial coverage silently."
                ),
                coordinate_name=dim,
                reason="non_time_alignment_dropped_coordinates",
            )
    aligned_len = precip_aligned.sizes[time_dim]
    original_len = max(precip_da.sizes[time_dim], temp_da.sizes[time_dim])
    if aligned_len == 0:
        raise CoordinateValidationError(
            message=(
                f"No overlapping timesteps found between precipitation and maximum_temperature "
                f"along '{time_dim}'. Cannot compute KBDI."
            ),
            coordinate_name=time_dim,
            reason="empty_intersection_after_alignment",
        )
    if aligned_len < original_len:
        warnings.warn(
            InputAlignmentWarning(
                message=(
                    f"Input alignment: precipitation had {precip_da.sizes[time_dim]} timesteps, "
                    f"maximum_temperature had {temp_da.sizes[time_dim]} timesteps. "
                    f"After inner join, {aligned_len} remain."
                ),
                original_size=original_len,
                aligned_size=aligned_len,
                dropped_count=original_len - aligned_len,
            ),
            stacklevel=3,
        )

    _validate_dask_chunks(precip_aligned, time_dim)
    _validate_dask_chunks(temp_aligned, time_dim)

    precip_target: Literal["mm", "inch"] = "inch" if units == "imperial" else "mm"
    temp_target: Literal["celsius", "fahrenheit"] = "fahrenheit" if units == "imperial" else "celsius"
    precip_aligned = _convert_precipitation_units(precip_aligned, precip_target)
    temp_aligned = _convert_temperature_units(temp_aligned, temp_target)

    # one shared spatial topology: a time-only input must broadcast to the
    # other's grid before apply_ufunc and the final transpose see its dims
    precip_aligned, temp_aligned = xr.broadcast(precip_aligned, temp_aligned)

    spatial_dims = tuple(d for d in precip_aligned.dims if d != time_dim)
    spatial_shape = tuple(precip_aligned.sizes[d] for d in spatial_dims)

    # validate initial conditions eagerly: a bad seed must fail this call, not a
    # later lazy evaluation (the NumPy core revalidates per chunk)
    maximum = 800.0 if units == "imperial" else _KBDI_MAX_MM
    internal_spatial_shape = spatial_shape or (1,)
    if initial_state is not None:
        _kbdi_state_arrays(initial_state, internal_spatial_shape, units, maximum)
    elif initial_kbdi is not None and not isinstance(initial_kbdi, xr.DataArray):
        # a DataArray seed may be Dask-backed; validating it eagerly would compute it
        _kbdi_initial_value(initial_kbdi, internal_spatial_shape, maximum)

    def _wrap_spatial(value: npt.ArrayLike | xr.DataArray) -> xr.DataArray:
        """Broadcast a scalar/array/DataArray to a DataArray on ``spatial_dims``.

        Giving Dask/apply_ufunc real dimension names is what lets it slice
        this secondary input per spatial chunk instead of broadcasting the
        whole un-chunked array into every chunk's call.
        """
        if isinstance(value, xr.DataArray):
            return value
        return xr.DataArray(np.broadcast_to(np.asarray(value, dtype=np.float64), spatial_shape), dims=spatial_dims)

    mean_annual_arg: xr.DataArray | None = None
    if mean_annual_precipitation is not None:
        if isinstance(mean_annual_precipitation, xr.DataArray):
            mean_annual_precipitation = _convert_precipitation_units(
                mean_annual_precipitation,
                precip_target,
                argument_name="mean_annual_precipitation.attrs['units']",
                annual=True,
            )
        mean_annual_arg = _wrap_spatial(mean_annual_precipitation)
    seed_kbdi_arg: xr.DataArray | None = None
    seed_wet_arg: xr.DataArray | None = None
    seed_gap_arg: xr.DataArray | None = None
    if initial_state is not None:
        gap_source = (
            initial_state.trailing_gap_days
            if initial_state.trailing_gap_days is not None
            else np.full(spatial_shape, -1, dtype=np.int64)
        )
        seed_kbdi_arg = _wrap_spatial(initial_state.kbdi)
        seed_wet_arg = _wrap_spatial(initial_state.wet_spell_precipitation)
        seed_gap_arg = _wrap_spatial(gap_source)
    elif initial_kbdi is not None:
        seed_kbdi_arg = _wrap_spatial(initial_kbdi)

    optional_slots = (mean_annual_arg, seed_kbdi_arg, seed_wet_arg, seed_gap_arg)
    include_mask = tuple(slot is not None for slot in optional_slots)
    optional_args = [slot for slot in optional_slots if slot is not None]

    output_time_len = max(aligned_len - spin_up, 0)

    def _kbdi_block(
        precip_block: np.ndarray, temp_block: np.ndarray, *optional_blocks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute one Dask chunk (or the whole array, eager): time axis last in, last out."""
        it = iter(optional_blocks)
        mean_annual_block = next(it) if include_mask[0] else None
        seed_kbdi_block = next(it) if include_mask[1] else None
        seed_wet_block = next(it) if include_mask[2] else None
        seed_gap_block = next(it) if include_mask[3] else None

        # apply_ufunc places core dims (time) last; kbdi()'s NumPy path is time-first.
        precip_t = np.moveaxis(precip_block, -1, 0).copy()
        temp_t = np.moveaxis(temp_block, -1, 0).copy()

        call_initial_state = None
        call_initial_kbdi = None
        if seed_wet_block is not None:
            gap = seed_gap_block.astype(np.int64)  # type: ignore[union-attr]
            call_initial_state = KBDIState(
                kbdi=seed_kbdi_block.copy(),  # type: ignore[union-attr]
                wet_spell_precipitation=seed_wet_block.copy(),
                trailing_gap_days=None if np.all(gap < 0) else gap.copy(),
                units=units,
            )
        elif seed_kbdi_block is not None:
            call_initial_kbdi = seed_kbdi_block

        result = kbdi(
            precip_t,
            temp_t,
            mean_annual_block,
            units=units,
            initial_kbdi=call_initial_kbdi,
            initial_state=call_initial_state,
            return_state=True,
            spin_up=spin_up,
            nan_policy=nan_policy,
            max_gap_days=max_gap_days,
        )
        assert isinstance(result, KBDIResult)
        # narrow for mypy: precip_t/temp_t are plain ndarrays, so this call
        # always takes kbdi()'s NumPy path and result.values is never a DataArray
        assert isinstance(result.values, np.ndarray)
        values_out = np.moveaxis(result.values, 0, -1)
        gap_out = result.state.trailing_gap_days
        if gap_out is None:
            gap_out = np.full(precip_t.shape[1:], -1, dtype=np.int64)
        return values_out, result.state.kbdi, result.state.wet_spell_precipitation, gap_out

    values_result, kbdi_result, wet_result, gap_result = xr.apply_ufunc(
        _kbdi_block,
        precip_aligned,
        temp_aligned,
        *optional_args,
        input_core_dims=[[time_dim], [time_dim]] + [[] for _ in optional_args],
        output_core_dims=[[time_dim], [], [], []],
        exclude_dims={time_dim},
        vectorize=False,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {time_dim: output_time_len}},
        output_dtypes=[float, float, float, np.int64],
    )

    values_result = values_result.transpose(*precip_aligned.dims)
    if time_dim in precip_aligned.coords:
        new_time_values = precip_aligned.coords[time_dim].values[spin_up : spin_up + output_time_len]
        values_result = values_result.assign_coords({time_dim: new_time_values})

    cf_key = "kbdi_imperial" if units == "imperial" else "kbdi"
    values_result.attrs = _build_output_attrs(
        precip_da,
        cf_metadata=CF_METADATA[cf_key],  # type: ignore[arg-type]
        # "units" is deliberately excluded here: it's a CF attribute the
        # registry entry above already sets ("mm" / "0.01 in"), and
        # _build_output_attrs layers calculation_metadata *over* cf_metadata,
        # so including it here would silently overwrite the physical unit
        # with the "metric"/"imperial" mode string.
        calculation_metadata={"nan_policy": nan_policy},
        index_name="KBDI",
    )

    if not return_state:
        result_da: xr.DataArray = values_result
        return result_da

    # One compute for the values and all three state fields: they share the
    # recurrence graph, so separate .values calls would each rerun it.
    values_name = values_result.name
    final_state = xr.Dataset(
        {
            "values": values_result,
            "kbdi": kbdi_result,
            "wet_spell_precipitation": wet_result,
            "trailing_gap_days": gap_result,
        }
    ).load()
    values_result = final_state["values"].rename(values_name)
    final_gap: npt.NDArray[np.int64] = final_state["trailing_gap_days"].values
    return KBDIResult(
        values=values_result,
        state=KBDIState(
            kbdi=final_state["kbdi"].values,
            wet_spell_precipitation=final_state["wet_spell_precipitation"].values,
            trailing_gap_days=None if bool(np.all(final_gap < 0)) else final_gap,
            units=units,
        ),
    )

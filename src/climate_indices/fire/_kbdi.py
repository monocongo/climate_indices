"""Keetch-Byram Drought Index (KBDI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices._recurrence import (
    DailyRecurrence,
    _active_view,
    _daily_weather_arrays,
    _static_spatial_array,
    _validate_recurrence_options,
    _validated_trailing_gaps,
    run_daily_recurrences,
)
from climate_indices._stateful_xarray import StatefulAlignment, stateful_recurrence_xarray
from climate_indices._units import (
    _convert_precipitation_units,
    _convert_temperature_units,
)
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import (
    InvalidArgumentError,
)
from climate_indices.validation import (
    InputType,
    detect_input_type,
)
from climate_indices.xarray_adapter import _wrap_spatial, build_output_attrs

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
        trailing_gap_days = _validated_trailing_gaps(
            state.trailing_gap_days, spatial_shape, "initial_state.trailing_gap_days"
        )

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

    precipitation_array, temperature_array = _daily_weather_arrays(
        ("precipitation", "maximum_temperature"),
        precipitation,
        maximum_temperature,
    )
    if np.any(np.isfinite(precipitation_array) & (precipitation_array < 0.0)):
        raise InvalidArgumentError(
            "precipitation must be non-negative where finite.",
            argument_name="precipitation",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
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

    def step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        """Advance KBDI one day for the active cells, updating the wet-spell state."""
        kbdi_state, precipitation_day, temperature_day, mean_annual_day, wet_state = _active_view(
            active,
            kbdi_value,
            precipitation_array[day],
            temperature_array[day],
            mean_annual,
            wet_spell,
        )
        wet_state = wet_state.copy()

        rainy = precipitation_day > 0.0
        event_total = wet_state + precipitation_day
        crossing_threshold = rainy & (wet_state <= _KBDI_RAIN_THRESHOLD_MM) & (event_total > _KBDI_RAIN_THRESHOLD_MM)
        continuing_wet_spell = rainy & (wet_state > _KBDI_RAIN_THRESHOLD_MM)
        net_rain = np.zeros_like(precipitation_day)
        net_rain[crossing_threshold] = event_total[crossing_threshold] - _KBDI_RAIN_THRESHOLD_MM
        net_rain[continuing_wet_spell] = precipitation_day[continuing_wet_spell]
        updated_wet_spell = np.where(rainy, event_total, 0.0)

        after_rain = np.maximum(0.0, kbdi_state - net_rain)
        drying = np.zeros_like(after_rain)
        drought_day = (temperature_day >= _KBDI_DRYING_TEMPERATURE_CELSIUS) & (after_rain < _KBDI_MAX_MM)
        with np.errstate(over="ignore"):
            drying[drought_day] = (
                (_KBDI_MAX_MM - after_rain[drought_day])
                * (0.968 * np.exp(0.0875 * temperature_day[drought_day] + 1.5552) - 8.30)
                / (1.0 + 10.88 * np.exp(-0.001736 * mean_annual_day[drought_day]))
                * 1e-3
            )
        updated: npt.NDArray[np.float64] = np.minimum(_KBDI_MAX_MM, after_rain + np.maximum(drying, 0.0))

        if active is None:
            wet_spell[:] = updated_wet_spell
        else:
            wet_spell[active] = updated_wet_spell
        return updated

    weather_valid = np.isfinite(precipitation_array) & np.isfinite(temperature_array)
    # A cell whose static climatology is unavailable has no recurrence to
    # gap-manage: the shared ADR-0007 policy never marks it active and leaves
    # its carried state as it was, so a NaN climatology is never a missing day.
    static_valid = np.isfinite(mean_annual)
    component = DailyRecurrence(
        "kbdi",
        kbdi_value,
        step,
        weather_valid,
        static_valid,
        trailing_gap_days,
    )

    def finalize(
        values: tuple[npt.NDArray[np.float64] | None, ...],
        gap_days: tuple[npt.NDArray[np.int64] | None, ...],
    ) -> npt.NDArray[np.float64] | KBDIResult:
        """Convert units and finalize the returned state inside the runner's guarded region."""
        values_array = values[0]
        assert values_array is not None
        state_gap_array = gap_days[0]
        state_gap_days: npt.NDArray[np.int64] | None = (
            None if state_gap_array is None else state_gap_array.reshape(spatial_shape).copy()
        )

        if units == "imperial":
            returned_values = values_array / _KBDI_MM_PER_POINT
            returned_kbdi = kbdi_value / _KBDI_MM_PER_POINT
            returned_wet_spell = wet_spell / 25.4
        else:
            returned_values = values_array
            returned_kbdi = kbdi_value
            returned_wet_spell = wet_spell

        result = returned_values.reshape(-1, *spatial_shape)
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

    return run_daily_recurrences(
        (component,),
        memory_arrays=(precipitation_array, temperature_array, mean_annual),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
        system_name="kbdi",
        fast_path=False,
        finalize=finalize,
        output_shape=(max(precipitation_array.shape[0] - spin_up, 0), *spatial_shape),
    )


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
    ``cf_metadata`` dict cannot do). Delegates the recurrence to
    :func:`kbdi`'s NumPy path through
    :func:`~climate_indices._stateful_xarray.stateful_recurrence_xarray`, one
    call per Dask spatial block with the full ``time`` axis.
    """
    precip_target: Literal["mm", "inch"] = "inch" if units == "imperial" else "mm"
    temp_target: Literal["celsius", "fahrenheit"] = "fahrenheit" if units == "imperial" else "celsius"
    precip_da = _convert_precipitation_units(precipitation, precip_target)
    temp_da = _convert_temperature_units(maximum_temperature, temp_target)

    maximum = 800.0 if units == "imperial" else _KBDI_MAX_MM

    def build_extra_inputs(alignment: StatefulAlignment) -> tuple[tuple[xr.DataArray, None], ...]:
        spatial_dims = alignment.spatial_dims
        spatial_shape = alignment.spatial_shape
        chunks = alignment.spatial_chunks
        # validate initial conditions eagerly: a bad seed must fail this call, not a later lazy
        # evaluation (the NumPy core revalidates per chunk)
        if initial_state is not None:
            _kbdi_state_arrays(initial_state, alignment.internal_spatial_shape, units, maximum)
        elif initial_kbdi is not None and not isinstance(initial_kbdi, xr.DataArray):
            # a DataArray seed may be Dask-backed; validating it eagerly would compute it
            _kbdi_initial_value(initial_kbdi, alignment.internal_spatial_shape, maximum)

        extras: list[tuple[xr.DataArray, None]] = []
        if mean_annual_precipitation is not None:
            mean_annual = mean_annual_precipitation
            if isinstance(mean_annual, xr.DataArray):
                mean_annual = _convert_precipitation_units(
                    mean_annual,
                    precip_target,
                    argument_name="mean_annual_precipitation.attrs['units']",
                    annual=True,
                )
            extras.append((_wrap_spatial(mean_annual, spatial_shape, spatial_dims, chunks=chunks), None))
        if initial_state is not None:
            gap_source = (
                initial_state.trailing_gap_days
                if initial_state.trailing_gap_days is not None
                else np.full(spatial_shape, -1, dtype=np.int64)
            )
            extras.append((_wrap_spatial(initial_state.kbdi, spatial_shape, spatial_dims, chunks=chunks), None))
            extras.append(
                (_wrap_spatial(initial_state.wet_spell_precipitation, spatial_shape, spatial_dims, chunks=chunks), None)
            )
            extras.append((_wrap_spatial(gap_source, spatial_shape, spatial_dims, chunks=chunks), None))
        elif initial_kbdi is not None:
            extras.append((_wrap_spatial(initial_kbdi, spatial_shape, spatial_dims, chunks=chunks), None))
        return tuple(extras)

    # Which optional operands reach the block, in the order build_extra_inputs supplies them.
    include_mask = (
        mean_annual_precipitation is not None,
        initial_state is not None or initial_kbdi is not None,
        initial_state is not None,
        initial_state is not None,
    )

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
        # a time-only input and a gridded one reach the block with different shapes; the gap
        # fill and the state fields describe the block's broadcast spatial shape
        block_spatial_shape = np.broadcast_shapes(precip_block.shape[:-1], temp_block.shape[:-1])

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
            gap_out = np.full(block_spatial_shape, -1, dtype=np.int64)
        return values_out, result.state.kbdi, result.state.wet_spell_precipitation, gap_out

    adapter_output = stateful_recurrence_xarray(
        [("precipitation", precip_da), ("maximum_temperature", temp_da)],
        _kbdi_block,
        time_dim=time_dim,
        spin_up=spin_up,
        output_core_dims=[time_dim, None, None, None],
        output_dtypes=[float, float, float, np.int64],
        index_display_name="KBDI",
        build_extra_inputs=build_extra_inputs,
    )
    values_result, kbdi_result, wet_result, gap_result = adapter_output.outputs

    cf_key = "kbdi_imperial" if units == "imperial" else "kbdi"
    values_result.attrs = build_output_attrs(
        precipitation,
        cf_metadata=CF_METADATA[cf_key],  # type: ignore[arg-type]
        # "units" is deliberately excluded here: it's a CF attribute the
        # registry entry above already sets ("mm" / "0.01 in"), and
        # build_output_attrs layers calculation_metadata *over* cf_metadata,
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

"""Fire-weather indices computed from standard meteorological inputs.

This module is the NumPy layer of the fire-weather family tracked in #793. It
currently provides the Fosberg Fire Weather Index and the Hot-Dry-Windy
Index, both weather-only and elementwise, and the Keetch-Byram Drought Index,
a daily recurrence. Stateful functions follow the execution, state-ownership,
and append/resume contract recorded in
``docs/adr/0006-fire-recursive-state-and-execution.md`` and the missing-data
policy recorded in ``docs/adr/0007-fire-missing-data-policy.md``; CFFWIS
(#803) will use the same contract.

References
----------
Srock, A.F., Charney, J.J., Potter, B.E., Goodrick, S.L. (2018) The
Hot-Dry-Windy Index: A New Fire Weather Index. Atmosphere, 9(7), 279.
doi:10.3390/atmos9070279.

Fosberg, M.A. (1978) Weather in wildland fire management: the fire weather
index. Conference on Sierra Nevada Meteorology, Lake Tahoe, CA, 1-4.

Simard, A.J. (1968) The moisture content of forest fuels - I. A review of the
basic concepts. Canadian Department of Forest and Rural Development, Forest
Fire Research Institute, Information Report FF-X-14.

Goodrick, S.L. (2002) Modification of the Fosberg fire weather index to include
drought. International Journal of Wildland Fire, 11, 205-211.
NCEP GEMPAK, ``pd_fosb`` / ``pr_fosb`` (T. Lee, 2003): the operational
implementation behind the ``FOSINDX`` GRIB2 parameter.
https://github.com/Unidata/gempak

Keetch, J.J. and Byram, G.M. (1968) A Drought Index for Forest Fire Control.
USDA Forest Service Research Paper SE-38.
https://research.fs.usda.gov/treesearch/40

Alexander, M.E. (1990) Computer calculation of the Keetch-Byram Drought
Index - programmers beware! Fire Management Notes, 51(4), 23-25.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices import pm_eto
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import (
    CoordinateValidationError,
    DataShapeError,
    InputAlignmentWarning,
    InvalidArgumentError,
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

# declare the function names that should be included in the public API for this module
__all__ = ["KBDIResult", "KBDIState", "fosberg_ffwi", "hot_dry_windy", "kbdi"]

# Simard (1968) equilibrium moisture content regressions, one per relative
# humidity range, with the coefficients of NCEP's operational GEMPAK code.
# Published restatements print the middle-range temperature coefficient as
# 0.01478 rather than 0.014784; across physical temperatures that moves the
# moisture content by less than 0.0005 and the index by less than 0.01.
_EMC_LOW_RH = (0.03229, 0.281073, 0.000578)
_EMC_MID_RH = (2.22749, 0.160107, 0.014784)
_EMC_HIGH_RH = (21.0606, 0.005565, 0.00035, 0.483199)

# Relative humidity breakpoints, upper bound inclusive as in GEMPAK. The
# published equations are written "h < 10" and "10 < h <= 50", which leaves
# exactly 10% in neither range.
_RH_BREAK_LOW = 10.0
_RH_BREAK_HIGH = 50.0

# moisture content at which the damping coefficient reaches zero
_EMC_EXTINCTION = 30.0

# scales the index to 100 at zero fuel moisture and a 30 mph wind
_FFWI_NORMALIZER = 0.3002
_FFWI_CAP = 100.0

# exact, by the definition of the international mile
_METERS_PER_SECOND_PER_MPH = 0.44704

# HDW analyzes the lowest 500 m above ground level (Srock et al., 2018)
_HDW_LAYER_TOP_METERS = 500.0

# pm_eto saturation vapor pressure is kPa; HDW reports VPD in hPa
_KPA_PER_HPA = 0.1

# Keetch and Byram (1968) Equation 18, corrected by Alexander (1990), is
# evaluated in metric units. One KBDI point is one hundredth of an inch.
_KBDI_MAX_MM = 203.2
_KBDI_MM_PER_POINT = 0.254
_KBDI_RAIN_THRESHOLD_MM = 5.08
_KBDI_DRYING_TEMPERATURE_CELSIUS = 10.0
_KBDI_MINIMUM_MEAN_ANNUAL_RECORD_DAYS = 30 * 365


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


def _kbdi_static_array(
    values: npt.ArrayLike,
    spatial_shape: tuple[int, ...],
    name: str,
) -> npt.NDArray[np.float64]:
    """Coerce a scalar or spatial field to the KBDI spatial shape."""
    array = _as_float_array(values)
    try:
        return np.broadcast_to(array, spatial_shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise InvalidArgumentError(
            f"{name} with shape {array.shape} cannot broadcast to KBDI spatial shape {spatial_shape}.",
            argument_name=name,
            argument_value=f"shape {array.shape}",
            valid_values=f"A scalar or an array broadcastable to {spatial_shape}",
        ) from exc


def _kbdi_initial_value(
    initial_kbdi: npt.ArrayLike,
    spatial_shape: tuple[int, ...],
    maximum: float,
) -> npt.NDArray[np.float64]:
    """Validate a caller-supplied seed KBDI and broadcast it to the spatial shape."""
    kbdi_value = _kbdi_static_array(initial_kbdi, spatial_shape, "initial_kbdi")
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

    kbdi_value = _kbdi_static_array(state.kbdi, spatial_shape, "initial_state.kbdi")
    wet_spell = _kbdi_static_array(
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
            argument_name="initial_state.kbdi",
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
        trailing = _kbdi_static_array(state.trailing_gap_days, spatial_shape, "initial_state.trailing_gap_days")
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
            argument_name="initial_state.kbdi",
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
    if nan_policy not in ("propagate", "bridge"):
        raise InvalidArgumentError(
            "nan_policy must be 'propagate' or 'bridge'.",
            argument_name="nan_policy",
            argument_value=str(nan_policy),
            valid_values="'propagate', 'bridge'",
        )
    if isinstance(max_gap_days, bool) or not isinstance(max_gap_days, int) or max_gap_days < 0:
        raise InvalidArgumentError(
            "max_gap_days must be a non-negative integer.",
            argument_name="max_gap_days",
            argument_value=str(max_gap_days),
            valid_values="A non-negative integer",
        )
    if (nan_policy == "propagate" and max_gap_days != 0) or (nan_policy == "bridge" and max_gap_days < 1):
        raise InvalidArgumentError(
            "max_gap_days must be zero for 'propagate' and positive for 'bridge'.",
            argument_name="max_gap_days",
            argument_value=str(max_gap_days),
            valid_values="0 for 'propagate'; at least 1 for 'bridge'",
        )
    if isinstance(spin_up, bool) or not isinstance(spin_up, int) or spin_up < 0:
        raise InvalidArgumentError(
            "spin_up must be a non-negative integer.",
            argument_name="spin_up",
            argument_value=str(spin_up),
            valid_values="A non-negative integer",
        )
    if initial_kbdi is not None and initial_state is not None:
        raise InvalidArgumentError(
            "initial_kbdi cannot be combined with initial_state.",
            argument_name="initial_kbdi/initial_state",
            argument_value="both supplied",
            valid_values="Supply at most one initial condition",
        )


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
        mean_annual = _kbdi_static_array(mean_annual_precipitation, internal_spatial_shape, "mean_annual_precipitation")
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


# CF units-attribute spellings this module recognizes, matching the precedent
# list in __main__.py's legacy CLI unit handling, plus the CF flux unit the
# NetCDF/Zarr ecosystem commonly uses for precipitation rate.
_PRECIP_UNITS_MM = frozenset({"mm", "millimeters", "millimeter"})
_PRECIP_UNITS_MM_PER_DAY = frozenset({"mm/dy", "mm day-1", "mm/day"})
_PRECIP_UNITS_MM_PER_YEAR = frozenset({"mm/year", "mm/yr", "mm year-1", "mm yr-1"})
_PRECIP_UNITS_INCH = frozenset({"inch", "inches"})
_PRECIP_UNITS_INCH_PER_YEAR = frozenset(
    {"in/year", "in/yr", "inch/year", "inch/yr", "inches/year", "inches/yr", "inch year-1", "inch yr-1"}
)
_PRECIP_UNITS_FLUX = frozenset({"kg m-2 s-1", "kg/m2/s", "kg m^-2 s^-1", "kg.m-2.s-1", "kg/m^2/s"})
_SECONDS_PER_DAY = 86400.0

_TEMP_UNITS_CELSIUS = frozenset({"c", "celsius", "degree_celsius", "degrees_celsius", "degc"})
_TEMP_UNITS_FAHRENHEIT = frozenset({"f", "fahrenheit", "degree_fahrenheit", "degrees_fahrenheit", "degf"})
_TEMP_UNITS_KELVIN = frozenset({"k", "kelvin"})
_KELVIN_OFFSET_CELSIUS = 273.15


def _convert_precipitation_units(
    data: xr.DataArray,
    target: Literal["mm", "inch"],
    *,
    argument_name: str = "precipitation.attrs['units']",
    annual: bool = False,
) -> xr.DataArray:
    """Convert a precipitation DataArray to ``target`` from its CF ``units`` attribute.

    An absent ``units`` attribute is assumed to already match ``target`` -- the
    caller's ``units=`` scale is trusted, never silently overridden. An
    unrecognized attribute raises ``InvalidArgumentError`` rather than guessing.
    ``annual=True`` declares a mean annual climatology: per-day and flux rate
    units are rejected, and per-year spellings are accepted in addition to the
    annual totals (``mm``, ``inch``). Conversion is xarray arithmetic, so
    Dask-backed input stays lazy.
    """
    raw_units = data.attrs.get("units")
    normalized = raw_units.strip().lower() if isinstance(raw_units, str) else None
    if normalized is None:
        return data
    if normalized in _PRECIP_UNITS_FLUX or normalized in _PRECIP_UNITS_MM_PER_DAY:
        if annual:
            raise InvalidArgumentError(
                f"mean_annual_precipitation cannot use precipitation rate units: {raw_units!r}.",
                argument_name=argument_name,
                argument_value=str(raw_units),
                valid_values="An annual total (mm, inch) or a per-year rate (mm year-1, inch year-1)",
            )
        if normalized in _PRECIP_UNITS_FLUX:
            data = data * _SECONDS_PER_DAY
        source: Literal["mm", "inch"] = "mm"
    elif normalized in _PRECIP_UNITS_MM or (annual and normalized in _PRECIP_UNITS_MM_PER_YEAR):
        source = "mm"
    elif normalized in _PRECIP_UNITS_INCH or (annual and normalized in _PRECIP_UNITS_INCH_PER_YEAR):
        source = "inch"
    else:
        raise InvalidArgumentError(
            f"Unsupported precipitation units attribute: {raw_units!r}.",
            argument_name=argument_name,
            argument_value=str(raw_units),
            valid_values=(
                "An annual total (mm, inch) or a per-year rate (mm year-1, inch year-1)"
                if annual
                else "mm / mm day-1, inch(es), or kg m-2 s-1"
            ),
        )
    if source == target:
        return data
    converted: xr.DataArray = data / 25.4 if target == "inch" else data * 25.4
    return converted


def _convert_temperature_units(data: xr.DataArray, target: Literal["celsius", "fahrenheit"]) -> xr.DataArray:
    """Convert a temperature DataArray to ``target`` from its CF ``units`` attribute.

    An absent ``units`` attribute is assumed to already match ``target``; an
    unrecognized one raises ``InvalidArgumentError`` rather than guessing.
    Conversion is xarray arithmetic, so Dask-backed input stays lazy.
    """
    raw_units = data.attrs.get("units")
    normalized = raw_units.strip().lower() if isinstance(raw_units, str) else None
    if normalized is None:
        return data
    if normalized in _TEMP_UNITS_KELVIN:
        data = data - _KELVIN_OFFSET_CELSIUS
        source: Literal["celsius", "fahrenheit"] = "celsius"
    elif normalized in _TEMP_UNITS_CELSIUS:
        source = "celsius"
    elif normalized in _TEMP_UNITS_FAHRENHEIT:
        source = "fahrenheit"
    else:
        raise InvalidArgumentError(
            f"Unsupported temperature units attribute: {raw_units!r}.",
            argument_name="maximum_temperature.attrs['units']",
            argument_value=str(raw_units),
            valid_values="K, kelvin, C/celsius, or F/fahrenheit",
        )
    if source == target:
        return data
    converted: xr.DataArray = data * 9.0 / 5.0 + 32.0 if target == "fahrenheit" else (data - 32.0) * 5.0 / 9.0
    return converted


def _validate_daily_time_coordinate(data: xr.DataArray, time_dim: str) -> None:
    """Require consecutive daily samples: the KBDI recurrence is defined per day.

    Called only when the time coordinate is attached; a dimension-only time
    axis has no cadence metadata to check.

    Raises:
        CoordinateValidationError: If the time steps are not exactly one day apart.
    """
    values = data.coords[time_dim].values
    if values.size < 2:
        return
    try:
        deltas = np.diff(values.astype("datetime64[ns]"))
    except (TypeError, ValueError) as exc:
        raise CoordinateValidationError(
            message=f"Cannot verify daily cadence for '{time_dim}': unsupported datetime type.",
            coordinate_name=time_dim,
            reason="unsupported_datetime_type",
        ) from exc
    if np.any(deltas != np.timedelta64(1, "D")):
        raise CoordinateValidationError(
            message=(
                f"KBDI requires consecutive daily '{time_dim}' steps, but '{time_dim}' is not daily. "
                "Aggregate the observations to daily totals and daily maxima before calling."
            ),
            coordinate_name=time_dim,
            reason="not_daily",
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


def _equilibrium_moisture_content(
    temperature_fahrenheit: npt.NDArray[np.float64],
    relative_humidity_percent: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Simard (1968) equilibrium moisture content, in percent.

    The three regressions were fitted independently and do not meet at the
    breakpoints. At 10% relative humidity the result jumps by roughly 0.5 to
    0.7 near typical fire-weather temperatures, with a size and sign that
    depend on temperature and grow larger toward the cold end of this
    module's supported range; at 50% the jump is smaller, roughly 0.5. That
    is a property of the published equations, not of this implementation,
    and the index inherits a jump of roughly one unit.

    Args:
        temperature_fahrenheit: Air temperature, degrees Fahrenheit.
        relative_humidity_percent: Relative humidity, percent.

    Returns:
        Equilibrium moisture content, percent.
    """
    t = temperature_fahrenheit
    h = relative_humidity_percent

    a, b, c = _EMC_LOW_RH
    low = a + b * h - c * h * t

    d, e, f = _EMC_MID_RH
    mid = d + e * h - f * t

    g, k, p, q = _EMC_HIGH_RH
    high = g + k * h * h - p * h * t - q * h

    return np.where(h <= _RH_BREAK_LOW, low, np.where(h <= _RH_BREAK_HIGH, mid, high))


def _moisture_damping(equilibrium_moisture_content: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Fosberg's moisture damping coefficient.

    With ``x = m / 30`` the coefficient ``1 - 2x + 1.5x**2 - 0.5x**3`` factors as
    ``(1 - x)(0.5x**2 - x + 1)``, and the second factor is positive for every
    real ``x``. The coefficient is therefore negative exactly when ``m`` exceeds
    30, which the high-humidity regression reaches in saturated air below about
    -46 F (-43 C). Clamping ``m`` at 30 keeps the index at zero there rather than
    letting it go negative.

    Args:
        equilibrium_moisture_content: Equilibrium moisture content, percent.

    Returns:
        Damping coefficient in [0, 1] for non-negative moisture content.
    """
    x = np.minimum(equilibrium_moisture_content, _EMC_EXTINCTION) / _EMC_EXTINCTION
    return 1.0 - 2.0 * x + 1.5 * x**2 - 0.5 * x**3


def _ffwi(
    equilibrium_moisture_content: npt.NDArray[np.float64],
    wind_speed_mph: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Uncapped Fosberg index from moisture content and wind speed in mph."""
    return _moisture_damping(equilibrium_moisture_content) * np.sqrt(1.0 + wind_speed_mph**2) / _FFWI_NORMALIZER


def _as_float_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce to float64, turning masked elements into NaN instead of dropping the mask."""
    filled = np.ma.asarray(values, dtype=np.float64).filled(np.nan)
    return np.asarray(filled, dtype=np.float64)


def fosberg_ffwi(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
    cap_at_100: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute the Fosberg Fire Weather Index (FFWI).

    A weather-only index of fire-weather potential from temperature, relative
    humidity and wind speed (Fosberg, 1978). It carries no state between time
    steps, so it is computed elementwise: the inputs broadcast against each
    other and any shape works, including arrays chunked along time.

    Inputs are SI, like the rest of the package and like NCEP's operational
    implementation. The conversion to the degrees Fahrenheit and miles per
    hour that the equations are written in happens here and nowhere else.

    Relative humidity exactly at 10% or 50% is assigned to the lower range, as
    in NCEP's GEMPAK code; the published equations leave exactly 10% in
    neither range. The moisture regressions are discontinuous at both
    breakpoints, so the index is too, by roughly one unit.

    Args:
        temperature_celsius: Air temperature, degrees Celsius.
        relative_humidity_percent: Relative humidity, percent, in [0, 100].
        wind_speed_meters_per_second: Wind speed, meters per second,
            non-negative.
        cap_at_100: Clamp the index at 100, the value Fosberg assigned to zero
            fuel moisture and a 30 mph wind, giving the conventional 0-100
            scale. NCEP's GEMPAK implementation does not clamp, and some
            studies deliberately keep values above 100; pass ``False`` to
            reproduce them. This only affects the upper bound: the moisture
            content clamp that keeps the index at 0 rather than negative in
            cold, saturated air (see ``_moisture_damping``) is always
            applied, independent of this flag.

    Returns:
        FFWI with the broadcast shape of the inputs. NaN where any input is
        NaN or masked, where relative humidity lies outside [0, 100], or where
        wind speed is negative.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together.

    Example:
        >>> from climate_indices import fire
        >>> round(float(fire.fosberg_ffwi(30.0, 15.0, 10.0)), 2)
        59.24
    """
    temperature = _as_float_array(temperature_celsius)
    humidity = _as_float_array(relative_humidity_percent)
    wind = _as_float_array(wind_speed_meters_per_second)

    try:
        temperature, humidity, wind = np.broadcast_arrays(temperature, humidity, wind)
    except ValueError as exc:
        message = (
            "Incompatible array shapes for Fosberg FFWI: "
            f"temperature={temperature.shape}, relative_humidity={humidity.shape}, "
            f"wind_speed={wind.shape}. The inputs must broadcast together."
        )
        _logger.error(message)
        raise InvalidArgumentError(
            message,
            argument_name="temperature_celsius/relative_humidity_percent/wind_speed_meters_per_second",
            argument_value=f"shapes {temperature.shape}, {humidity.shape}, {wind.shape}",
            valid_values="Arrays broadcastable to a common shape",
        ) from exc

    # bind context and emit calculation_started event
    log = _logger.bind(
        index_type="fosberg_ffwi",
        input_shape=temperature.shape,
        input_elements=temperature.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(temperature, humidity, wind)

    try:
        # outside the physical range the regressions still return numbers, but
        # meaningless ones, so treat such values as missing
        invalid = (humidity < 0.0) | (humidity > 100.0) | (wind < 0.0)
        invalid_count = int(np.count_nonzero(invalid))
        if invalid_count > 0:
            _logger.warning(
                f"Found {invalid_count} values with relative humidity outside [0, 100] "
                "or negative wind speed; FFWI is NaN there."
            )

        temperature_fahrenheit = temperature * 9.0 / 5.0 + 32.0
        wind_speed_mph = wind / _METERS_PER_SECOND_PER_MPH

        emc = _equilibrium_moisture_content(temperature_fahrenheit, humidity)
        index = _ffwi(emc, wind_speed_mph)
        if cap_at_100:
            index = np.minimum(index, _FFWI_CAP)

        result = np.where(invalid, np.nan, index).astype(np.float64, copy=False)
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
    CFFWIS's shared recurrence. :func:`xarray.apply_ufunc` calls this
    function's own NumPy path directly, once per Dask chunk, with
    ``level_dim`` as the sole core dimension and reduced away -- the same
    shape ``test_hdw_chunked_time_and_space_match_eager`` already exercises
    for the NumPy core, here wrapped with validation and CF metadata.
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

    result: xr.DataArray = xr.apply_ufunc(
        hot_dry_windy,
        temperature,
        humidity,
        wind,
        height,
        input_core_dims=[[level_dim]] * 4,
        output_core_dims=[[]],
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
        is a plain passthrough with no cadence or alignment requirement.

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
        # outside the physical range the formulas still return numbers, but
        # meaningless ones, so treat such values as missing
        in_layer = (height >= 0.0) & (height <= _HDW_LAYER_TOP_METERS)
        invalid = (humidity < 0.0) | (humidity > 100.0) | (wind < 0.0)
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
        result = np.where(np.any(in_layer, axis=axis), index, np.nan).astype(np.float64, copy=False)

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

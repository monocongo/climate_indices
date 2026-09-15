"""Fire-weather indices computed from standard meteorological inputs.

This module is the NumPy layer of the fire-weather family tracked in #793. It
provides the Fosberg Fire Weather Index and the Hot-Dry-Windy Index, both
weather-only and elementwise, the Keetch-Byram Drought Index, and the three
Canadian Forest Fire Weather Index System (CFFWIS) moisture codes: the Fine
Fuel Moisture Code, the Duff Moisture Code, and the Drought Code. Stateful
functions follow the execution, state-ownership, and append/resume contract
recorded in ``docs/adr/0006-fire-recursive-state-and-execution.md`` and the
missing-data policy recorded in ``docs/adr/0007-fire-missing-data-policy.md``;
the CFFWIS behavior indices (#804) will use the same contract.

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

Van Wagner, C.E. and Pickett, T.L. (1985) Equations and FORTRAN program for
the Canadian Forest Fire Weather Index System. Canadian Forestry Service,
Forestry Technical Report 33.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from climate_indices import pm_eto
from climate_indices.exceptions import DataShapeError, InvalidArgumentError
from climate_indices.logging_config import get_logger
from climate_indices.performance import check_large_array_memory

# retrieve structlog logger for this module
_logger = get_logger(__name__)

# declare the function names that should be included in the public API for this module
__all__ = [
    "DCResult",
    "DCState",
    "DMCResult",
    "DMCState",
    "FFMCResult",
    "FFMCState",
    "KBDIResult",
    "KBDIState",
    "drought_code",
    "duff_moisture_code",
    "ffmc",
    "fosberg_ffwi",
    "hot_dry_windy",
    "kbdi",
]

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

# The CFFWIS moisture codes (#803) follow Van Wagner and Pickett (1985) as
# implemented by the NRCan reference code (cffdrs). All equations are
# evaluated in the source's operational units: km/h wind, mm rain, degrees
# Celsius. The published FFMC equations print 147.2 for the moisture-content
# conversion; the reference code uses the exact 250 * 59.5 / 101, applied in
# both directions, and this implementation matches the reference code.
_FFMC_COEFFICIENT = 250.0 * 59.5 / 101.0
_FFMC_MAXIMUM = 101.0
_FFMC_MOISTURE_CAP = 250.0
_FFMC_PRECIPITATION_THRESHOLD_MM = 0.5
_FFMC_MOISTURE_FOR_RAIN_CORRECTION = 150.0
_KILOMETERS_PER_HOUR_PER_METER_PER_SECOND = 3.6

_DMC_PRECIPITATION_THRESHOLD_MM = 1.5
_DMC_TEMPERATURE_FLOOR_CELSIUS = -1.1

_DC_PRECIPITATION_THRESHOLD_MM = 2.8
_DC_TEMPERATURE_FLOOR_CELSIUS = -2.8

# Effective day length in hours for DMC, by latitude band and calendar month
# (Van Wagner and Pickett, 1985). The bands and their table rows are, in
# order: 46 N (latitude > 30), 20 N (10 < latitude <= 30), equator
# (-10 < latitude <= 10), 20 S (-30 < latitude <= -10), and 40 S
# (latitude <= -30). The 46 N row is the Canadian standard; the other rows
# are the reference code's latitude adjustments, not a fallback for it.
_DMC_EFFECTIVE_DAY_LENGTH_HOURS = np.array(
    [
        [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0],
        [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],
        [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8],
    ]
)

# Day-length adjustment term for DC potential evapotranspiration, by
# latitude band and calendar month (Van Wagner and Pickett, 1985). Bands:
# north (latitude > 20), equator (-20 < latitude <= 20), south
# (latitude <= -20).
_DC_DAY_LENGTH_ADJUSTMENT = np.array(
    [
        [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6],
        [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4],
        [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8],
    ]
)


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
    """KBDI values and final state returned by :func:`kbdi`."""

    values: npt.NDArray[np.float64]
    state: KBDIState


def _static_spatial_array(
    values: npt.ArrayLike,
    spatial_shape: tuple[int, ...],
    name: str,
) -> npt.NDArray[np.float64]:
    """Coerce a scalar or spatial field to a recurrence's trailing spatial shape."""
    array = _as_float_array(values)
    try:
        return np.broadcast_to(array, spatial_shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise InvalidArgumentError(
            f"{name} with shape {array.shape} cannot broadcast to spatial shape {spatial_shape}.",
            argument_name=name,
            argument_value=f"shape {array.shape}",
            valid_values=f"A scalar or an array broadcastable to {spatial_shape}",
        ) from exc


def _validate_recurrence_options(
    nan_policy: object,
    max_gap_days: object,
    spin_up: object,
    seed_name: str,
    seed: object,
    initial_state: object,
) -> None:
    """Validate the configuration shared by every stateful fire recurrence."""
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
    if seed is not None and initial_state is not None:
        raise InvalidArgumentError(
            f"{seed_name} cannot be combined with initial_state.",
            argument_name=f"{seed_name}/initial_state",
            argument_value="both supplied",
            valid_values="Supply at most one initial condition",
        )


def _apply_gap_policy(
    state_value: npt.NDArray[np.float64],
    day_weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    started: npt.NDArray[np.bool_],
    poisoned: npt.NDArray[np.bool_],
    trailing_gap_days: npt.NDArray[np.int64],
    *,
    nan_policy: Literal["propagate", "bridge"],
    max_gap_days: int,
) -> npt.NDArray[np.bool_]:
    """Apply one day of the ADR-0007 missing-day policy, returning the active cells.

    ``state_value``, ``started``, ``poisoned``, and ``trailing_gap_days`` are
    updated in place. A missing day is one with an invalid weather
    observation. A cell whose static input is unusable never starts and is not
    an elapsed missing day. A valid day is the return point's last day, so any
    earlier run is closed.
    """
    valid = day_weather_valid & static_valid
    missing_started = ~day_weather_valid & static_valid & (started | poisoned)

    if nan_policy == "propagate":
        state_value[missing_started] = np.nan
        poisoned[missing_started] = True
        trailing_gap_days[missing_started] = np.maximum(trailing_gap_days[missing_started], 0) + 1
    else:
        next_gap_days = np.maximum(trailing_gap_days, 0) + 1
        over_gap_limit = missing_started & (next_gap_days > max_gap_days)
        state_value[over_gap_limit] = np.nan
        poisoned[over_gap_limit] = True
        trailing_gap_days[missing_started] = next_gap_days[missing_started]

    active = valid & ~poisoned
    started[active] = True
    trailing_gap_days[valid & (started | poisoned)] = 0
    return active


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

    kbdi_value = _static_spatial_array(state.kbdi, spatial_shape, "initial_state.kbdi")
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
            argument_name="initial_state.kbdi",
            argument_value="NaN KBDI where initial_state.trailing_gap_days is -1 or None",
            valid_values="A finite KBDI in a not-started cell; NaN only after a gap has started the cell",
        )

    return kbdi_value, wet_spell, trailing_gap_days


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
) -> npt.NDArray[np.float64] | KBDIResult:
    """Compute the Keetch-Byram Drought Index (KBDI).

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

    Returns:
        KBDI with the same time-first shape as the broadcast weather inputs,
        less ``spin_up`` leading days. Returns ``KBDIResult`` when
        ``return_state`` is true.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, or physical
            precipitation inputs are invalid.
    """
    if units not in ("metric", "imperial"):
        raise InvalidArgumentError(
            "units must be 'metric' or 'imperial'.",
            argument_name="units",
            argument_value=str(units),
            valid_values="'metric', 'imperial'",
        )
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_kbdi", initial_kbdi, initial_state)

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
            kbdi_value = _static_spatial_array(initial_kbdi, internal_spatial_shape, "initial_kbdi")
            if np.any(~np.isfinite(kbdi_value)) or np.any(kbdi_value < 0.0) or np.any(kbdi_value > maximum):
                raise InvalidArgumentError(
                    f"initial_kbdi must be finite and within [0, {maximum:g}].",
                    argument_name="initial_kbdi",
                    argument_value="non-finite or outside the valid KBDI range",
                    valid_values=f"[0, {maximum:g}]",
                )
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
            # A cell whose static climatology is unavailable has no recurrence
            # to gap-manage: its output is NaN and its carried state is left
            # as it was, so a NaN climatology is never an elapsed missing day.
            active = _apply_gap_policy(
                kbdi_value,
                weather_valid,
                static_valid,
                started,
                poisoned,
                trailing_gap_days,
                nan_policy=nan_policy,
                max_gap_days=max_gap_days,
            )

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


# ---------------------------------------------------------------------------
# Canadian Forest Fire Weather Index System moisture codes (#803)
#
# The three codes share the daily-recurrence contract of ADR-0006 and the
# missing-day policy of ADR-0007 through ``_run_cffwis_recurrence``, which
# owns the gap bookkeeping and the output/spin-up handling. Each ``_*_next``
# function is the pure daily update for one code.


@dataclass(frozen=True)
class FFMCState:
    """State needed to resume a Fine Fuel Moisture Code recurrence.

    A ``trailing_gap_days`` value of ``None`` means no valid day has started
    the recurrence. For spatial arrays, ``-1`` marks individual cells that
    have not started yet. A NaN ``ffmc`` is only valid where
    ``trailing_gap_days`` shows that a gap has started the cell; a
    not-started cell holds a number.
    """

    ffmc: npt.NDArray[np.float64]
    trailing_gap_days: npt.NDArray[np.int64] | None


@dataclass(frozen=True)
class FFMCResult:
    """Fine Fuel Moisture Code values and final state returned by :func:`ffmc`."""

    values: npt.NDArray[np.float64]
    state: FFMCState


@dataclass(frozen=True)
class DMCState:
    """State needed to resume a Duff Moisture Code recurrence.

    ``trailing_gap_days`` follows :class:`FFMCState`; a NaN ``dmc`` is only
    valid where a gap has started the cell.
    """

    dmc: npt.NDArray[np.float64]
    trailing_gap_days: npt.NDArray[np.int64] | None


@dataclass(frozen=True)
class DMCResult:
    """Duff Moisture Code values and final state returned by :func:`duff_moisture_code`."""

    values: npt.NDArray[np.float64]
    state: DMCState


@dataclass(frozen=True)
class DCState:
    """State needed to resume a Drought Code recurrence.

    ``trailing_gap_days`` follows :class:`FFMCState`; a NaN ``dc`` is only
    valid where a gap has started the cell.
    """

    dc: npt.NDArray[np.float64]
    trailing_gap_days: npt.NDArray[np.int64] | None


@dataclass(frozen=True)
class DCResult:
    """Drought Code values and final state returned by :func:`drought_code`."""

    values: npt.NDArray[np.float64]
    state: DCState


def _daily_weather_arrays(
    names: tuple[str, ...],
    *values: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Coerce and broadcast time-first daily weather inputs, rejecting infinity."""
    arrays = tuple(_as_float_array(value) for value in values)
    try:
        broadcast = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        shapes = ", ".join(f"{name}={array.shape}" for name, array in zip(names, arrays, strict=True))
        raise InvalidArgumentError(
            f"Incompatible array shapes for daily weather inputs: {shapes}. The inputs must broadcast together.",
            argument_name="/".join(names),
            argument_value=f"shapes {shapes}",
            valid_values="Arrays broadcastable to a common time-first shape",
        ) from exc
    if broadcast[0].ndim == 0:
        raise DataShapeError(
            "Daily weather inputs must include a time dimension.",
            expected_shape="(time, ...)",
            actual_shape=broadcast[0].shape,
        )
    infinite = [name for name, array in zip(names, broadcast, strict=True) if np.any(np.isinf(array))]
    if infinite:
        raise InvalidArgumentError(
            f"{'/'.join(infinite)} must be finite or NaN: infinity is not a missing observation.",
            argument_name="/".join(infinite),
            argument_value="infinite value",
            valid_values="Finite values or NaN",
        )
    result: tuple[npt.NDArray[np.float64], ...] = tuple(broadcast)
    return result


def _month_array(month: npt.ArrayLike, weather_shape: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """Validate calendar months and broadcast them to the time-first weather shape."""
    months = _as_float_array(month)
    if np.any(~np.isfinite(months)) or np.any(months != np.floor(months)) or np.any((months < 1.0) | (months > 12.0)):
        raise InvalidArgumentError(
            "month must contain integer calendar months in [1, 12].",
            argument_name="month",
            argument_value="a non-finite, non-integral, or out-of-range value",
            valid_values="Integer values in [1, 12]",
        )
    if months.ndim == 1 and len(weather_shape) > 1 and months.shape[0] == weather_shape[0]:
        # a calendar month series is shared across every spatial cell
        months = months.reshape((months.shape[0],) + (1,) * (len(weather_shape) - 1))
    try:
        broadcast = np.broadcast_to(months, weather_shape)
    except ValueError as exc:
        raise InvalidArgumentError(
            "month must broadcast to the time-first weather shape.",
            argument_name="month",
            argument_value=f"shape {months.shape}",
            valid_values=f"A scalar or an array broadcastable to {weather_shape}",
        ) from exc
    result: npt.NDArray[np.int64] = broadcast.astype(np.int64)
    return result


def _latitude_and_validity(
    latitude_degrees_north: npt.ArrayLike,
    spatial_shape: tuple[int, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Broadcast latitude to the spatial shape and report which cells are usable."""
    latitude = _as_float_array(latitude_degrees_north)
    if np.any(np.isinf(latitude)):
        raise InvalidArgumentError(
            "latitude_degrees_north must be finite or NaN: infinity is not a missing location.",
            argument_name="latitude_degrees_north",
            argument_value="infinite value",
            valid_values="Finite values in [-90, 90] or NaN",
        )
    if np.any(np.isfinite(latitude) & ((latitude < -90.0) | (latitude > 90.0))):
        raise InvalidArgumentError(
            "latitude_degrees_north must be within [-90, 90] where finite.",
            argument_name="latitude_degrees_north",
            argument_value="value outside [-90, 90]",
            valid_values="Finite values in [-90, 90] or NaN",
        )
    broadcast = _static_spatial_array(latitude, spatial_shape, "latitude_degrees_north")
    return broadcast, np.isfinite(broadcast)


def _initialize_single_value_state(
    *,
    seed: npt.ArrayLike | None,
    seed_name: str,
    initial_state: object,
    state_type: type[DCState] | type[DMCState] | type[FFMCState],
    value_name: str,
    default_seed: float,
    minimum: float,
    maximum: float | None,
    spatial_shape: tuple[int, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Resolve a seed or a supplied state into a recurrence's starting state."""
    bound = f"[{minimum:g}, {maximum:g}]" if maximum is not None else f"greater than or equal to {minimum:g}"
    if initial_state is None:
        if seed is None:
            value = np.full(spatial_shape, default_seed, dtype=np.float64)
        else:
            value = _static_spatial_array(seed, spatial_shape, seed_name)
            outside = np.any(value < minimum) or (maximum is not None and np.any(value > maximum))
            if np.any(~np.isfinite(value)) or outside:
                raise InvalidArgumentError(
                    f"{seed_name} must be finite and {bound}.",
                    argument_name=seed_name,
                    argument_value="non-finite or outside the valid range",
                    valid_values=bound,
                )
        return value, np.full(spatial_shape, -1, dtype=np.int64)

    if not isinstance(initial_state, state_type):
        raise InvalidArgumentError(
            f"initial_state must be a {state_type.__name__}.",
            argument_name="initial_state",
            argument_value=type(initial_state).__name__,
            valid_values=state_type.__name__,
        )
    value = _static_spatial_array(getattr(initial_state, value_name), spatial_shape, f"initial_state.{value_name}")
    trailing = initial_state.trailing_gap_days
    if trailing is None:
        trailing_gap_days = np.full(spatial_shape, -1, dtype=np.int64)
    else:
        trailing_array = _static_spatial_array(trailing, spatial_shape, "initial_state.trailing_gap_days")
        if (
            np.any(~np.isfinite(trailing_array))
            or np.any(trailing_array < -1)
            or np.any(trailing_array != np.floor(trailing_array))
        ):
            raise InvalidArgumentError(
                "initial_state.trailing_gap_days must contain integers greater than or equal to -1.",
                argument_name="initial_state.trailing_gap_days",
                argument_value="non-integral or less than -1 value",
                valid_values="-1 or a non-negative integer",
            )
        trailing_gap_days = trailing_array.astype(np.int64)

    outside = np.any(value < minimum) or (maximum is not None and np.any(value > maximum))
    if np.any(~np.isfinite(value) & ~np.isnan(value)) or outside:
        raise InvalidArgumentError(
            f"initial_state.{value_name} must be NaN or {bound}.",
            argument_name=f"initial_state.{value_name}",
            argument_value="non-finite or outside the valid range",
            valid_values=f"NaN or {bound}",
        )
    if np.any(np.isnan(value) & (trailing_gap_days < 0)):
        raise InvalidArgumentError(
            f"initial_state.{value_name} may be NaN only where trailing_gap_days shows a gap has started.",
            argument_name=f"initial_state.{value_name}",
            argument_value="NaN where initial_state.trailing_gap_days is -1 or None",
            valid_values="A finite value in a not-started cell; NaN only after a gap has started the cell",
        )
    return value, trailing_gap_days


def _active_view(
    active: npt.NDArray[np.bool_] | None,
    *arrays: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Restrict each array to the active cells, or return them unchanged for all cells."""
    if active is None:
        return arrays
    return tuple(array[active] for array in arrays)


def _run_cffwis_recurrence(
    state_value: npt.NDArray[np.float64],
    step: Callable[[int, npt.NDArray[np.bool_] | None], npt.NDArray[np.float64]],
    *,
    index_type: str,
    weather_valid: npt.NDArray[np.bool_],
    static_valid: npt.NDArray[np.bool_],
    trailing_gap_days: npt.NDArray[np.int64],
    memory_arrays: tuple[npt.NDArray[np.float64], ...],
    spin_up: int,
    nan_policy: Literal["propagate", "bridge"],
    max_gap_days: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64] | None]:
    """Run a time-first daily recurrence under the ADR-0007 missing-day policy.

    ``step(day, active)`` returns the next code value for every cell when
    ``active`` is ``None`` and for the selected cells otherwise. Only cells
    with a valid observation whose recurrence has started and is not poisoned
    adopt it. A cell whose static input is unusable never starts: its output
    stays NaN and its state is untouched.
    """
    n_days = weather_valid.shape[0]
    values = np.full((max(n_days - spin_up, 0), *weather_valid.shape[1:]), np.nan, dtype=np.float64)
    started = trailing_gap_days >= 0
    poisoned = np.isnan(state_value)

    log = _logger.bind(
        index_type=index_type,
        input_shape=weather_valid.shape,
        input_elements=weather_valid.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    memory_metrics = check_large_array_memory(*memory_arrays, weather_valid, values)

    try:
        for day in range(n_days):
            # A cell whose static input is unusable has no recurrence to
            # gap-manage: it never starts, so it is not an elapsed missing day.
            active = _apply_gap_policy(
                state_value,
                weather_valid[day],
                static_valid,
                started,
                poisoned,
                trailing_gap_days,
                nan_policy=nan_policy,
                max_gap_days=max_gap_days,
            )

            if np.any(active):
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    updated = step(day, None if active.all() else active)
                if np.any(~np.isfinite(updated)):
                    raise InvalidArgumentError(
                        f"{index_type} produced a non-finite value from finite inputs.",
                        argument_name=index_type,
                        argument_value="non-finite result",
                        valid_values="Finite inputs whose result stays within float64",
                    )
                if active.all():
                    state_value[:] = updated
                else:
                    state_value[active] = updated
            if day >= spin_up:
                values[day - spin_up] = np.where(active, state_value, np.nan)

        state_gap_days = trailing_gap_days.copy() if np.any(started | poisoned) else None
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=values.shape,
            **(memory_metrics or {}),
        )
        return values, state_gap_days
    except Exception as exc:
        log.error(
            "calculation_failed",
            exc_info=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def _ffmc_next(
    ffmc_previous: npt.NDArray[np.float64],
    temperature_celsius: npt.NDArray[np.float64],
    relative_humidity_percent: npt.NDArray[np.float64],
    wind_speed_kilometers_per_hour: npt.NDArray[np.float64],
    precipitation_mm: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Advance the FFMC one day (Van Wagner and Pickett, 1985, Eq. 1-10)."""
    # Eq. 1: previous FFMC to fine fuel moisture content, percent
    moisture = _FFMC_COEFFICIENT * (101.0 - ffmc_previous) / (59.5 + ffmc_previous)
    rained = precipitation_mm > _FFMC_PRECIPITATION_THRESHOLD_MM
    effective_rain = np.where(rained, precipitation_mm - _FFMC_PRECIPITATION_THRESHOLD_MM, precipitation_mm)
    # Eqs. 3a and 3b: rain adds moisture, with an amendment above 150 percent
    rain_moisture = 42.5 * effective_rain * np.exp(-100.0 / (251.0 - moisture)) * (1.0 - np.exp(-6.93 / effective_rain))
    rain_moisture += np.where(
        moisture > _FFMC_MOISTURE_FOR_RAIN_CORRECTION,
        0.0015 * (moisture - _FFMC_MOISTURE_FOR_RAIN_CORRECTION) ** 2 * np.sqrt(effective_rain),
        0.0,
    )
    moisture = np.where(rained, np.minimum(moisture + rain_moisture, _FFMC_MOISTURE_CAP), moisture)

    # Eqs. 4 and 5: equilibrium moisture content for drying and wetting
    drying_equilibrium = (
        0.942 * relative_humidity_percent**0.679
        + 11.0 * np.exp((relative_humidity_percent - 100.0) / 10.0)
        + 0.18 * (21.1 - temperature_celsius) * (1.0 - np.exp(-0.115 * relative_humidity_percent))
    )
    wetting_equilibrium = (
        0.618 * relative_humidity_percent**0.753
        + 10.0 * np.exp((relative_humidity_percent - 100.0) / 10.0)
        + 0.18 * (21.1 - temperature_celsius) * (1.0 - np.exp(-0.115 * relative_humidity_percent))
    )

    # Eqs. 6-9: dry toward the drying equilibrium or wet toward the wetting
    # equilibrium, whichever side of it the fuel is on
    humidity_fraction = relative_humidity_percent / 100.0
    wind_root = np.sqrt(wind_speed_kilometers_per_hour)
    drying_rate = (0.424 * (1.0 - humidity_fraction**1.7) + 0.0694 * wind_root * (1.0 - humidity_fraction**8)) * (
        0.581 * np.exp(0.0365 * temperature_celsius)
    )
    wetting_rate = (
        0.424 * (1.0 - (1.0 - humidity_fraction) ** 1.7) + 0.0694 * wind_root * (1.0 - (1.0 - humidity_fraction) ** 8)
    ) * (0.581 * np.exp(0.0365 * temperature_celsius))
    dried = drying_equilibrium + (moisture - drying_equilibrium) * 10.0**-drying_rate
    wetted = wetting_equilibrium - (wetting_equilibrium - moisture) * 10.0**-wetting_rate
    moisture = np.where(
        moisture > drying_equilibrium,
        dried,
        np.where(moisture < wetting_equilibrium, wetted, moisture),
    )

    # Eq. 10: final FFMC conversion, clamped to the published range
    result = 59.5 * (250.0 - moisture) / (_FFMC_COEFFICIENT + moisture)
    return np.clip(result, 0.0, _FFMC_MAXIMUM)


def _dmc_next(
    dmc_previous: npt.NDArray[np.float64],
    temperature_celsius: npt.NDArray[np.float64],
    relative_humidity_percent: npt.NDArray[np.float64],
    precipitation_mm: npt.NDArray[np.float64],
    effective_day_length_hours: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Advance the DMC one day (Van Wagner and Pickett, 1985, Eq. 11-16)."""
    # Eq. 16: the log drying rate, with its temperature floor
    temperature = np.maximum(temperature_celsius, _DMC_TEMPERATURE_FLOOR_CELSIUS)
    drying_rate = 1.894 * (temperature + 1.1) * (100.0 - relative_humidity_percent) * effective_day_length_hours * 1e-4

    # Eqs. 11-15: rain above 1.5 mm rewets the duff layer
    rained = precipitation_mm > _DMC_PRECIPITATION_THRESHOLD_MM
    effective_rain = 0.92 * precipitation_mm - 1.27
    moisture_before = 20.0 + 280.0 / np.exp(0.023 * dmc_previous)
    # Eq. 13's piecewise slope of the moisture-content relation
    slope = np.where(
        dmc_previous <= 33.0,
        100.0 / (0.5 + 0.3 * dmc_previous),
        np.where(
            dmc_previous <= 65.0,
            14.0 - 1.3 * np.log(dmc_previous),
            6.2 * np.log(dmc_previous) - 17.2,
        ),
    )
    moisture_after = moisture_before + 1000.0 * effective_rain / (48.77 + slope * effective_rain)
    after_rain = np.maximum(43.43 * (5.6348 - np.log(moisture_after - 20.0)), 0.0)

    previous = np.where(rained, after_rain, dmc_previous)
    return np.maximum(previous + drying_rate, 0.0)


def _dc_next(
    dc_previous: npt.NDArray[np.float64],
    temperature_celsius: npt.NDArray[np.float64],
    precipitation_mm: npt.NDArray[np.float64],
    day_length_adjustment: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Advance the DC one day (Van Wagner and Pickett, 1985, Eq. 18-23)."""
    # Eq. 22: potential evapotranspiration, floored at zero for winter
    temperature = np.maximum(temperature_celsius, _DC_TEMPERATURE_FLOOR_CELSIUS)
    potential_evapotranspiration = np.maximum((0.36 * (temperature + 2.8) + day_length_adjustment) / 2.0, 0.0)

    # Eqs. 18-21: rain above 2.8 mm reduces the drought code
    rained = precipitation_mm > _DC_PRECIPITATION_THRESHOLD_MM
    effective_rain = 0.83 * precipitation_mm - 1.27
    moisture_before = 800.0 * np.exp(-dc_previous / 400.0)
    after_rain = np.maximum(dc_previous - 400.0 * np.log(1.0 + 3.937 * effective_rain / moisture_before), 0.0)

    previous = np.where(rained, after_rain, dc_previous)
    value: npt.NDArray[np.float64] = np.maximum(previous + potential_evapotranspiration, 0.0)
    return value


def _dmc_day_length_band(latitude_degrees_north: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Index the DMC effective-day-length table for each cell's latitude band."""
    band: npt.NDArray[np.intp] = np.select(
        [
            latitude_degrees_north > 30.0,
            latitude_degrees_north > 10.0,
            latitude_degrees_north > -10.0,
            latitude_degrees_north > -30.0,
        ],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.intp)
    return band


def _dc_day_length_band(latitude_degrees_north: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Index the DC day-length-adjustment table for each cell's latitude band."""
    band: npt.NDArray[np.intp] = np.select(
        [latitude_degrees_north > 20.0, latitude_degrees_north > -20.0],
        [0, 1],
        default=2,
    ).astype(np.intp)
    return band


def ffmc(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
    precipitation_mm: npt.ArrayLike,
    *,
    initial_ffmc: npt.ArrayLike | None = None,
    initial_state: FFMCState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
) -> npt.NDArray[np.float64] | FFMCResult:
    """Compute the Fine Fuel Moisture Code (FFMC).

    The moisture content of fine surface litter and other fine fuels, the
    base of the Canadian Forest Fire Weather Index System (Van Wagner and
    Pickett, 1985). It is a daily recurrence: rain rewets the fuel, and
    temperature, relative humidity, and wind move it toward the day's
    equilibrium moisture content.

    The equations are evaluated in the source's operational units, so the
    wind speed is converted from meters per second to km/h here and nowhere
    else. The moisture-content conversion uses the NRCan reference code's
    exact ``250 * 59.5 / 101`` rather than the ``147.2`` printed in the
    report, to match the reference implementation rather than the printed
    constant.

    The source's open choices are resolved here as: only rain above 0.5 mm
    rewets the fuel, moisture content is capped at 250 percent, the code is
    clamped to [0, 101], the literature seed is 85, and a day whose relative
    humidity is outside [0, 100] or whose wind speed is negative counts as a
    missing observation under ``nan_policy``.

    Args:
        temperature_celsius: Daily noon-local-standard-time air temperature,
            time-first, degrees Celsius.
        relative_humidity_percent: Daily noon-local-standard-time relative
            humidity, time-first, percent.
        wind_speed_meters_per_second: Daily 10 m wind speed, time-first,
            meters per second.
        precipitation_mm: Daily 24-hour precipitation, time-first, mm.
        initial_ffmc: Seed code, scalar or an array of the trailing spatial
            shape. ``None`` selects the literature seed of 85. Cannot be
            combined with ``initial_state``.
        initial_state: State returned by an earlier call.
        return_state: Return :class:`FFMCResult` with the final state.
        spin_up: Number of leading input days to compute but omit from the
            output.
        nan_policy: ``"propagate"`` poisons a started recurrence at a missing
            day; ``"bridge"`` skips gaps up to ``max_gap_days``.
        max_gap_days: Maximum bridged consecutive missing days. Must be zero
            for ``"propagate"`` and positive for ``"bridge"``.

    Returns:
        FFMC with the time-first shape of the broadcast weather inputs, less
        ``spin_up`` leading days. Returns :class:`FFMCResult` when
        ``return_state`` is true.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, or physical
            inputs are invalid.
    """
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_ffmc", initial_ffmc, initial_state)

    temperature, humidity, wind, precipitation = _daily_weather_arrays(
        (
            "temperature_celsius",
            "relative_humidity_percent",
            "wind_speed_meters_per_second",
            "precipitation_mm",
        ),
        temperature_celsius,
        relative_humidity_percent,
        wind_speed_meters_per_second,
        precipitation_mm,
    )
    if np.any(np.isfinite(precipitation) & (precipitation < 0.0)):
        raise InvalidArgumentError(
            "precipitation_mm must be non-negative where finite.",
            argument_name="precipitation_mm",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
        )

    spatial_shape = temperature.shape[1:]
    internal_spatial_shape = spatial_shape if spatial_shape else (1,)
    temperature = temperature.reshape(temperature.shape[0], *internal_spatial_shape)
    humidity = humidity.reshape(temperature.shape)
    precipitation = precipitation.reshape(temperature.shape)
    with np.errstate(over="ignore"):
        wind = wind.reshape(temperature.shape) * _KILOMETERS_PER_HOUR_PER_METER_PER_SECOND
    if np.any(np.isinf(wind)):
        raise InvalidArgumentError(
            "wind_speed_meters_per_second is too large to convert to kilometers per hour.",
            argument_name="wind_speed_meters_per_second",
            argument_value="finite value that overflows the km/h conversion",
            valid_values="Finite values that do not overflow the km/h conversion",
        )

    weather_valid = (
        np.isfinite(temperature)
        & np.isfinite(humidity)
        & np.isfinite(wind)
        & np.isfinite(precipitation)
        & (humidity >= 0.0)
        & (humidity <= 100.0)
        & (wind >= 0.0)
    )
    state_value, trailing_gap_days = _initialize_single_value_state(
        seed=initial_ffmc,
        seed_name="initial_ffmc",
        initial_state=initial_state,
        state_type=FFMCState,
        value_name="ffmc",
        default_seed=85.0,
        minimum=0.0,
        maximum=_FFMC_MAXIMUM,
        spatial_shape=internal_spatial_shape,
    )

    def step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        state_slice, temperature_slice, humidity_slice, wind_slice, precipitation_slice = _active_view(
            active,
            state_value,
            temperature[day],
            humidity[day],
            wind[day],
            precipitation[day],
        )
        return _ffmc_next(state_slice, temperature_slice, humidity_slice, wind_slice, precipitation_slice)

    values, state_gap_days = _run_cffwis_recurrence(
        state_value,
        step,
        index_type="ffmc",
        weather_valid=weather_valid,
        static_valid=np.ones(internal_spatial_shape, dtype=np.bool_),
        trailing_gap_days=trailing_gap_days,
        memory_arrays=(temperature, humidity, wind, precipitation),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
    )

    result = values.reshape(-1, *spatial_shape)
    if not return_state:
        return result
    return FFMCResult(
        values=result,
        state=FFMCState(
            ffmc=state_value.reshape(spatial_shape).copy(),
            trailing_gap_days=None if state_gap_days is None else state_gap_days.reshape(spatial_shape),
        ),
    )


def duff_moisture_code(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    precipitation_mm: npt.ArrayLike,
    latitude_degrees_north: npt.ArrayLike,
    month: npt.ArrayLike,
    *,
    initial_dmc: npt.ArrayLike | None = None,
    initial_state: DMCState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
) -> npt.NDArray[np.float64] | DMCResult:
    """Compute the Duff Moisture Code (DMC).

    The moisture content of loosely compacted organic layers of moderate
    depth, driven by noon temperature, relative humidity, and 24-hour rain
    (Van Wagner and Pickett, 1985). It is a daily recurrence whose drying
    rate scales with the month- and latitude-dependent effective day length.

    The effective day-length tables are selected from five latitude bands:
    46 N (latitude > 30), 20 N (10 < latitude <= 30), the equator
    (-10 < latitude <= 10), 20 S (-30 < latitude <= -10), and 40 S
    (latitude <= -30). No band is a fallback for another. The temperature
    input is floored at -1.1 C, rain above 1.5 mm rewets the layer, and the
    code is floored at zero with no upper bound.

    A NaN latitude means the cell has no usable day-length band: its output
    is always NaN and its recurrence never starts. A day whose relative
    humidity is outside [0, 100] counts as a missing observation under
    ``nan_policy``.

    Args:
        temperature_celsius: Daily noon-local-standard-time air temperature,
            time-first, degrees Celsius.
        relative_humidity_percent: Daily noon-local-standard-time relative
            humidity, time-first, percent.
        precipitation_mm: Daily 24-hour precipitation, time-first, mm.
        latitude_degrees_north: Cell latitude, scalar or an array
            broadcastable to the trailing spatial shape (for example
            ``(lat, 1)`` or ``(lat, lon)`` for a ``(time, lat, lon)`` grid),
            degrees north in [-90, 90]. NaN marks a cell with no usable band.
        month: Calendar month for each day, scalar or time-first, integer in
            [1, 12].
        initial_dmc: Seed code, scalar or an array of the trailing spatial
            shape. ``None`` selects the literature seed of 6. Cannot be
            combined with ``initial_state``.
        initial_state: State returned by an earlier call.
        return_state: Return :class:`DMCResult` with the final state.
        spin_up: Number of leading input days to compute but omit from the
            output.
        nan_policy: ``"propagate"`` poisons a started recurrence at a missing
            day; ``"bridge"`` skips gaps up to ``max_gap_days``.
        max_gap_days: Maximum bridged consecutive missing days. Must be zero
            for ``"propagate"`` and positive for ``"bridge"``.

    Returns:
        DMC with the time-first shape of the broadcast weather inputs, less
        ``spin_up`` leading days. Returns :class:`DMCResult` when
        ``return_state`` is true.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, latitude, or
            physical inputs are invalid.
    """
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_dmc", initial_dmc, initial_state)

    temperature, humidity, precipitation = _daily_weather_arrays(
        ("temperature_celsius", "relative_humidity_percent", "precipitation_mm"),
        temperature_celsius,
        relative_humidity_percent,
        precipitation_mm,
    )
    if np.any(np.isfinite(precipitation) & (precipitation < 0.0)):
        raise InvalidArgumentError(
            "precipitation_mm must be non-negative where finite.",
            argument_name="precipitation_mm",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
        )
    months = _month_array(month, temperature.shape)
    latitude, static_valid = _latitude_and_validity(latitude_degrees_north, temperature.shape[1:])

    spatial_shape = temperature.shape[1:]
    internal_spatial_shape = spatial_shape if spatial_shape else (1,)
    temperature = temperature.reshape(temperature.shape[0], *internal_spatial_shape)
    humidity = humidity.reshape(temperature.shape)
    precipitation = precipitation.reshape(temperature.shape)
    months = months.reshape(temperature.shape)
    latitude = latitude.reshape(internal_spatial_shape)
    static_valid = static_valid.reshape(internal_spatial_shape)

    weather_valid = (
        np.isfinite(temperature)
        & np.isfinite(humidity)
        & np.isfinite(precipitation)
        & (humidity >= 0.0)
        & (humidity <= 100.0)
    )
    state_value, trailing_gap_days = _initialize_single_value_state(
        seed=initial_dmc,
        seed_name="initial_dmc",
        initial_state=initial_state,
        state_type=DMCState,
        value_name="dmc",
        default_seed=6.0,
        minimum=0.0,
        maximum=None,
        spatial_shape=internal_spatial_shape,
    )
    band = _dmc_day_length_band(latitude)

    def step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        if active is None:
            effective_day_length = _DMC_EFFECTIVE_DAY_LENGTH_HOURS[band, months[day] - 1]
        else:
            effective_day_length = _DMC_EFFECTIVE_DAY_LENGTH_HOURS[band[active], months[day][active] - 1]
        state_slice, temperature_slice, humidity_slice, precipitation_slice = _active_view(
            active,
            state_value,
            temperature[day],
            humidity[day],
            precipitation[day],
        )
        return _dmc_next(state_slice, temperature_slice, humidity_slice, precipitation_slice, effective_day_length)

    values, state_gap_days = _run_cffwis_recurrence(
        state_value,
        step,
        index_type="duff_moisture_code",
        weather_valid=weather_valid,
        static_valid=static_valid,
        trailing_gap_days=trailing_gap_days,
        memory_arrays=(temperature, humidity, precipitation),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
    )

    result = values.reshape(-1, *spatial_shape)
    if not return_state:
        return result
    return DMCResult(
        values=result,
        state=DMCState(
            dmc=state_value.reshape(spatial_shape).copy(),
            trailing_gap_days=None if state_gap_days is None else state_gap_days.reshape(spatial_shape),
        ),
    )


def drought_code(
    temperature_celsius: npt.ArrayLike,
    precipitation_mm: npt.ArrayLike,
    latitude_degrees_north: npt.ArrayLike,
    month: npt.ArrayLike,
    *,
    initial_dc: npt.ArrayLike | None = None,
    initial_state: DCState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
) -> npt.NDArray[np.float64] | DCResult:
    """Compute the Drought Code (DC).

    The moisture content of deep, compact organic layers, driven by noon
    temperature and 24-hour rain (Van Wagner and Pickett, 1985). It is a
    daily recurrence whose potential evapotranspiration scales with the
    month- and latitude-dependent day length. This is the CFFWIS component
    only; it is distinct from the package's drought indices (SPI, SPEI,
    PDSI).

    The day-length adjustment is selected from three latitude bands: north
    (latitude > 20), the equator (-20 < latitude <= 20), and south
    (latitude <= -20). The temperature input is floored at -2.8 C, potential
    evapotranspiration is floored at zero, rain above 2.8 mm reduces the
    code, and the code is floored at zero with no upper bound.

    A NaN latitude means the cell has no usable day-length band: its output
    is always NaN and its recurrence never starts. Relative humidity is not
    a DC input.

    Args:
        temperature_celsius: Daily noon-local-standard-time air temperature,
            time-first, degrees Celsius.
        precipitation_mm: Daily 24-hour precipitation, time-first, mm.
        latitude_degrees_north: Cell latitude, scalar or an array
            broadcastable to the trailing spatial shape (for example
            ``(lat, 1)`` or ``(lat, lon)`` for a ``(time, lat, lon)`` grid),
            degrees north in [-90, 90]. NaN marks a cell with no usable band.
        month: Calendar month for each day, scalar or time-first, integer in
            [1, 12].
        initial_dc: Seed code, scalar or an array of the trailing spatial
            shape. ``None`` selects the literature seed of 15. Cannot be
            combined with ``initial_state``.
        initial_state: State returned by an earlier call.
        return_state: Return :class:`DCResult` with the final state.
        spin_up: Number of leading input days to compute but omit from the
            output.
        nan_policy: ``"propagate"`` poisons a started recurrence at a missing
            day; ``"bridge"`` skips gaps up to ``max_gap_days``.
        max_gap_days: Maximum bridged consecutive missing days. Must be zero
            for ``"propagate"`` and positive for ``"bridge"``.

    Returns:
        DC with the time-first shape of the broadcast weather inputs, less
        ``spin_up`` leading days. Returns :class:`DCResult` when
        ``return_state`` is true.

    Raises:
        DataShapeError: If the weather inputs have no time dimension.
        InvalidArgumentError: If shapes, configuration, state, latitude, or
            physical inputs are invalid.
    """
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_dc", initial_dc, initial_state)

    temperature, precipitation = _daily_weather_arrays(
        ("temperature_celsius", "precipitation_mm"),
        temperature_celsius,
        precipitation_mm,
    )
    if np.any(np.isfinite(precipitation) & (precipitation < 0.0)):
        raise InvalidArgumentError(
            "precipitation_mm must be non-negative where finite.",
            argument_name="precipitation_mm",
            argument_value="negative value",
            valid_values="Non-negative daily precipitation",
        )
    months = _month_array(month, temperature.shape)
    latitude, static_valid = _latitude_and_validity(latitude_degrees_north, temperature.shape[1:])

    spatial_shape = temperature.shape[1:]
    internal_spatial_shape = spatial_shape if spatial_shape else (1,)
    temperature = temperature.reshape(temperature.shape[0], *internal_spatial_shape)
    precipitation = precipitation.reshape(temperature.shape)
    months = months.reshape(temperature.shape)
    latitude = latitude.reshape(internal_spatial_shape)
    static_valid = static_valid.reshape(internal_spatial_shape)

    weather_valid = np.isfinite(temperature) & np.isfinite(precipitation)
    state_value, trailing_gap_days = _initialize_single_value_state(
        seed=initial_dc,
        seed_name="initial_dc",
        initial_state=initial_state,
        state_type=DCState,
        value_name="dc",
        default_seed=15.0,
        minimum=0.0,
        maximum=None,
        spatial_shape=internal_spatial_shape,
    )
    band = _dc_day_length_band(latitude)

    def step(day: int, active: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.float64]:
        if active is None:
            day_length_adjustment = _DC_DAY_LENGTH_ADJUSTMENT[band, months[day] - 1]
        else:
            day_length_adjustment = _DC_DAY_LENGTH_ADJUSTMENT[band[active], months[day][active] - 1]
        state_slice, temperature_slice, precipitation_slice = _active_view(
            active,
            state_value,
            temperature[day],
            precipitation[day],
        )
        return _dc_next(state_slice, temperature_slice, precipitation_slice, day_length_adjustment)

    values, state_gap_days = _run_cffwis_recurrence(
        state_value,
        step,
        index_type="drought_code",
        weather_valid=weather_valid,
        static_valid=static_valid,
        trailing_gap_days=trailing_gap_days,
        memory_arrays=(temperature, precipitation),
        spin_up=spin_up,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
    )

    result = values.reshape(-1, *spatial_shape)
    if not return_state:
        return result
    return DCResult(
        values=result,
        state=DCState(
            dc=state_value.reshape(spatial_shape).copy(),
            trailing_gap_days=None if state_gap_days is None else state_gap_days.reshape(spatial_shape),
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


def hot_dry_windy(
    temperature_celsius: npt.ArrayLike,
    relative_humidity_percent: npt.ArrayLike,
    wind_speed_meters_per_second: npt.ArrayLike,
    height_agl_meters: npt.ArrayLike,
    *,
    level_axis: int = -1,
) -> npt.NDArray[np.float64]:
    """Compute the Hot-Dry-Windy Index (HDW).

    A weather-only index of dangerous fire-behavior potential (Srock et al.,
    2018): the vapor pressure deficit (VPD) times the wind speed, maximized
    over the levels in the lowest 500 m above ground level (AGL)::

        HDW = max over levels with 0 <= height_agl <= 500 of (VPD * wind speed)

    Inputs are vertical profiles with SI units, like the rest of the package.
    The four inputs broadcast against each other; the shared dimension
    ``level_axis`` is the vertical coordinate and is reduced by the maximum.
    VPD comes from each level's own temperature and relative humidity, with
    saturation vapor pressure from ``pm_eto.saturation_vapor_pressure`` (FAO-56
    Eq 11), converted from kPa to the hPa of the published index.

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
            maximum.
        level_axis: Axis of the broadcast inputs that holds the vertical
            coordinate. Reduced by the layer maximum.

    Returns:
        HDW in hPa m s-1, with the broadcast shape of the inputs minus
        ``level_axis``. NaN where any in-layer level has NaN or out-of-range
        input, and for columns with no level inside the lowest 500 m AGL.

    Raises:
        InvalidArgumentError: If the inputs cannot be broadcast together, or
            ``level_axis`` is out of range for the broadcast shape.

    Example:
        >>> from climate_indices import fire
        >>> round(float(fire.hot_dry_windy([30.0, 26.0], [15.0, 30.0], [8.0, 12.0], [10.0, 400.0])), 2)
        288.53
    """
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

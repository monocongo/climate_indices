"""The daily-recurrence runtime shared by the fire and flood index families.

Every stateful daily index in this package — the CFFWIS moisture codes, KBDI,
and the flood antecedent precipitation index — is the same recurrence: one
vectorized spatial update per day, wrapped in the single missing-day policy of
``docs/adr/0007-fire-missing-data-policy.md`` and the seasonal mask of
``docs/adr/0010-seasonal-carry-is-an-explicit-mask.md``. This module owns that
runtime, the input coercion and time-first broadcasting it needs, and the
lifecycle logging every recurrence emits. Each index supplies only its daily
step, its validity rule and its typed frozen state.

This module is NumPy-only by design: the stable NumPy core never depends on
the beta xarray layer (``docs/adr/0006-fire-recursive-state-and-execution.md``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from climate_indices.exceptions import (
    DataShapeError,
    InputTypeError,
    InvalidArgumentError,
    wrap_value_error,
)
from climate_indices.logging_config import get_logger, log_calculation_failure
from climate_indices.performance import check_large_array_memory

# retrieve structlog logger for this module
_logger = get_logger(__name__)


def _as_float_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce to float64, turning masked elements into NaN instead of dropping the mask.

    Non-numeric inputs are rejected rather than coerced: datetime, string, and
    object arrays would otherwise arrive as plausible but meaningless numbers,
    and complex arrays would silently discard their imaginary part.
    """
    if np.asarray(values).dtype.kind not in "biuf":
        raise InputTypeError(
            "Index inputs must be numeric: datetime, string, object, and complex arrays are not coerced to float64.",
            expected_type=float,
            actual_type=np.asarray(values).dtype.type,
        )
    filled = np.ma.asarray(values, dtype=np.float64).filled(np.nan)
    return np.asarray(filled, dtype=np.float64)


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
        wrap_value_error(
            exc,
            message=f"{name} with shape {array.shape} cannot broadcast to spatial shape {spatial_shape}.",
            argument_name=name,
            argument_value=f"shape {array.shape}",
            valid_values=f"A scalar or an array broadcastable to {spatial_shape}",
        )


def _validated_trailing_gaps(
    trailing_gap_days: npt.ArrayLike,
    spatial_shape: tuple[int, ...],
    name: str,
) -> npt.NDArray[np.int64]:
    """Return trailing gap counts as int64 after the one shared validity check.

    The bound is deliberately ``2**63`` rather than ``int64``'s maximum: a
    float64 cannot represent ``2**63 - 1`` distinctly, so a caller-supplied
    count at or above ``2**63`` is rejected before the int cast can wrap
    negative and report a bogus gap of zero.
    """
    raw_gaps = _static_spatial_array(trailing_gap_days, spatial_shape, name)
    if (
        np.any(~np.isfinite(raw_gaps))
        or np.any(raw_gaps < -1)
        or np.any(raw_gaps >= float(np.iinfo(np.int64).max))
        or np.any(raw_gaps != np.floor(raw_gaps))
    ):
        raise InvalidArgumentError(
            f"{name} must be integers >= -1 and below 2**63 "
            "(float64 precision; values near the int64 maximum are rejected so the day counter cannot overflow).",
            argument_name=name,
            argument_value="non-integral, less than -1, or at least 2**63",
            valid_values="-1 or a non-negative integer below 2**63",
        )
    return raw_gaps.astype(np.int64)


def _validate_recurrence_options(
    nan_policy: object,
    max_gap_days: object,
    spin_up: object,
    seed_name: str,
    seed: object,
    initial_state: object,
) -> None:
    """Validate the configuration shared by every stateful recurrence."""
    if nan_policy not in ("propagate", "bridge"):
        raise InvalidArgumentError(
            "nan_policy must be 'propagate' or 'bridge'.",
            argument_name="nan_policy",
            argument_value=str(nan_policy),
            valid_values="'propagate', 'bridge'",
        )
    if isinstance(max_gap_days, bool) or not isinstance(max_gap_days, (int, np.integer)) or max_gap_days < 0:
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
    if isinstance(spin_up, bool) or not isinstance(spin_up, (int, np.integer)) or spin_up < 0:
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


def _daily_weather_arrays(
    names: tuple[str, ...],
    *values: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Coerce and broadcast time-first daily weather inputs, rejecting infinity."""
    arrays = tuple(_as_float_array(value) for value in values)
    # Time-first arrays are left-aligned: a shorter input is shared across every
    # trailing axis, so a (time,) series spans the whole spatial grid instead of
    # NumPy aligning it with the final axis.
    ndim = max(array.ndim for array in arrays)
    arrays = tuple(
        array if array.ndim == ndim else array.reshape(array.shape + (1,) * (ndim - array.ndim)) for array in arrays
    )
    try:
        broadcast = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        shapes = ", ".join(f"{name}={array.shape}" for name, array in zip(names, arrays, strict=True))
        wrap_value_error(
            exc,
            message=(
                f"Incompatible array shapes for daily weather inputs: {shapes}. The inputs must broadcast together."
            ),
            argument_name="/".join(names),
            argument_value=f"shapes {shapes}",
            valid_values="Arrays broadcastable to a common time-first shape",
        )
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
    in_season: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.bool_]:
    """Apply one day of the ADR-0007 missing-day policy, returning the active cells.

    ``state_value``, ``started``, ``poisoned``, and ``trailing_gap_days`` are
    updated in place. A missing day is one with an invalid weather
    observation. A cell whose static input is unusable never starts and is not
    an elapsed missing day. A valid day is the return point's last day, so any
    earlier run is closed.

    ``in_season`` optionally restricts the policy to the cells inside the fire
    season: an off-season day is neither an observation nor a missing day, so
    it never advances the recurrence and never counts against the gap
    allowance (``docs/adr/0010-seasonal-carry-is-an-explicit-mask.md``).
    """
    if in_season is None:
        valid = day_weather_valid & static_valid
        missing_started = ~day_weather_valid & static_valid & (started | poisoned)
    else:
        valid = day_weather_valid & static_valid & in_season
        missing_started = ~day_weather_valid & static_valid & in_season & (started | poisoned)

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


def _active_view(
    active: npt.NDArray[np.bool_] | None,
    *arrays: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Restrict each array to the active cells, or return them unchanged for all cells."""
    if active is None:
        return arrays
    return tuple(array[active] for array in arrays)


@dataclass
class DailyRecurrence:
    """One recurrence threaded through the shared daily day loop."""

    index_type: str
    value: npt.NDArray[np.float64]
    step: Callable[[int, npt.NDArray[np.bool_] | None], npt.NDArray[np.float64]]
    weather_valid: npt.NDArray[np.bool_]
    static_valid: npt.NDArray[np.bool_]
    trailing_gap_days: npt.NDArray[np.int64]
    # optional per-day fire-season mask: off-season days freeze the recurrence
    # instead of advancing or gap-managing it (ADR-0010)
    in_season: npt.NDArray[np.bool_] | None = None
    # derived once by the runner so the day loop has a single grouping of
    # per-component state instead of parallel index spaces
    started: npt.NDArray[np.bool_] = field(init=False)
    poisoned: npt.NDArray[np.bool_] = field(init=False)
    static_all_valid: bool = field(init=False, default=False)


def run_daily_recurrences(
    components: tuple[DailyRecurrence, ...],
    *,
    memory_arrays: tuple[npt.NDArray[np.float64], ...],
    spin_up: int,
    nan_policy: Literal["propagate", "bridge"],
    max_gap_days: int,
    system_name: str,
    fast_path: bool,
    record: tuple[bool, ...] | None = None,
) -> tuple[tuple[npt.NDArray[np.float64] | None, ...], tuple[npt.NDArray[np.int64] | None, ...]]:
    """Run one or more daily recurrences through one shared time loop.

    ``step(day, active)`` returns the next code value for every cell when
    ``active`` is ``None`` and for the selected cells otherwise. Only cells
    with a valid observation whose recurrence has started and is not poisoned
    adopt it. A cell whose static input is unusable never starts: its output
    stays NaN and its state is untouched.

    Each component carries its own ADR-0007 bookkeeping because a day can be
    missing for one code and valid for another: negative wind only affects
    FFMC, humidity outside [0, 100] affects FFMC and DMC, and a NaN latitude
    only affects DMC and DC. ``fast_path`` enables the all-valid shortcut that
    the combined orchestrator uses; the single-code wrapper disables it so
    their behaviour stays identical to the pre-orchestrator engine.

    ``record`` marks the components whose daily history is kept: an unrecorded
    component still runs and advances its state, but its output slot is
    ``None`` instead of a full time series, so a subset request allocates only
    the histories a selected output reads. The default records every
    component, which is what the single-code wrapper needs.

    Every recurrence that runs through here emits the same ``calculation_started``
    and ``calculation_completed`` (or ``calculation_failed``) lifecycle events,
    so no family can drift into silent execution.
    """
    n_days = components[0].weather_valid.shape[0]
    log = _logger.bind(
        index_type=system_name,
        input_shape=components[0].weather_valid.shape,
        input_elements=components[0].weather_valid.size,
    )
    log.info("calculation_started")
    t0 = time.perf_counter()
    try:
        # the allocation is inside the try so an output-allocation failure
        # still reports the recurrence lifecycle
        records = (True,) * len(components) if record is None else record
        values = tuple(
            np.full((max(n_days - spin_up, 0), *component.weather_valid.shape[1:]), np.nan, dtype=np.float64)
            if keep
            else None
            for component, keep in zip(components, records, strict=True)
        )
        for component in components:
            component.started = component.trailing_gap_days >= 0
            component.poisoned = np.isnan(component.value)
            component.static_all_valid = bool(component.static_valid.all())
        memory_metrics = check_large_array_memory(*memory_arrays, *(value for value in values if value is not None))

        for day in range(n_days):
            for index, component in enumerate(components):
                # off-season days neither advance the recurrence nor count as
                # missing, so they emit the carried state instead of a gap NaN
                season_today = None if component.in_season is None else component.in_season[day]
                carried = None
                if (
                    fast_path
                    and season_today is None
                    and component.static_all_valid
                    and component.weather_valid[day].all()
                ):
                    # A fully valid day with a usable static input: the gap
                    # policy reduces to "every unpoisoned cell is active, the
                    # trailing count resets, and a started recurrence stays
                    # started", with no partial-mask bookkeeping needed.
                    component.trailing_gap_days.fill(0)
                    component.started.fill(True)
                    active = None if not component.poisoned.any() else ~component.poisoned
                else:
                    # A cell whose static input is unusable has no recurrence
                    # to gap-manage: it never starts, so it is not an elapsed
                    # missing day.
                    active = _apply_gap_policy(
                        component.value,
                        component.weather_valid[day],
                        component.static_valid,
                        component.started,
                        component.poisoned,
                        component.trailing_gap_days,
                        nan_policy=nan_policy,
                        max_gap_days=max_gap_days,
                        in_season=season_today,
                    )
                    if season_today is not None:
                        carried = ~season_today & component.static_valid & component.started
                all_active = active is None or bool(active.all())
                if active is None or np.any(active):
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        updated = component.step(day, None if all_active else active)
                    if np.any(~np.isfinite(updated)):
                        raise InvalidArgumentError(
                            f"{component.index_type} produced a non-finite value from finite inputs.",
                            argument_name=component.index_type,
                            argument_value="non-finite result",
                            valid_values="Finite inputs whose result stays within float64",
                        )
                    if all_active:
                        component.value[:] = updated
                    else:
                        component.value[active] = updated
                output = values[index]
                if day >= spin_up and output is not None:
                    output_day = output[day - spin_up]
                    if carried is None:
                        if all_active:
                            output_day[:] = component.value
                        else:
                            assert active is not None
                            output_day[:] = np.where(active, component.value, np.nan)
                    else:
                        emitted = carried if active is None else (carried | active)
                        output_day[:] = np.where(emitted, component.value, np.nan)

        state_gap_days = tuple(
            component.trailing_gap_days.copy() if np.any(component.started | component.poisoned) else None
            for component in components
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=next(value.shape for value in values if value is not None),
            **(memory_metrics or {}),
        )
        return values, state_gap_days
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise

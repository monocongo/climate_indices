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
from typing import Literal, NoReturn, TypeVar, overload

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

# A component's optional Rust kernel (docs/architecture.md): it runs that
# component's whole time axis and returns its recorded history and final gap
# counts, or None to leave the component to the Python day loop here.
#
# Arguments: the shape the kernel's recorded history is returned in (None when
# this call records no history for it; the kernel builds that one history, so
# the Python side allocates no second full-size array beside it), the spin-up day
# count, the missing-day policy, and the bridge allowance. The kernel updates the
# component's own state and gap arrays in place, so the caller's arrays are
# correct when the call returns.
NativeRecurrence = Callable[
    [tuple[int, ...] | None, int, Literal["propagate", "bridge"], int],
    tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.int64] | None] | None,
]

# The widest recurrence option the bindings can represent: ``spin_up`` crosses
# as a ``usize`` and ``max_gap_days`` as an ``i64``, while public validation
# accepts any non-negative Python integer. A native callable returns None for a
# wider value, which keeps the recurrence on the Python path instead of failing
# argument conversion.
_MAX_NATIVE_OPTION = int(np.iinfo(np.int64).max)


def _raise_non_finite(index_type: str, underlying: Exception) -> NoReturn:
    """Raise the error the Python driver raises for a non-finite step result.

    ``_advance_component`` rejects a daily update that is not finite although
    its inputs were, and a native kernel reports the same condition; raising the
    original error type from here keeps the two paths indistinguishable.
    """
    raise InvalidArgumentError(
        f"{index_type} produced a non-finite value from finite inputs.",
        argument_name=index_type,
        argument_value="non-finite result",
        valid_values="Finite inputs whose result stays within float64",
    ) from underlying


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


def _initialize_component(component: DailyRecurrence) -> None:
    """Derive one component's started/poisoned flags and cache its static-validity reduction."""
    component.started = component.trailing_gap_days >= 0
    component.poisoned = np.isnan(component.value)
    component.static_all_valid = bool(component.static_valid.all())


def _component_day_active(
    component: DailyRecurrence,
    day: int,
    *,
    fast_path: bool,
    nan_policy: Literal["propagate", "bridge"],
    max_gap_days: int,
) -> tuple[npt.NDArray[np.bool_] | None, npt.NDArray[np.bool_] | None]:
    """Return the active cells for one component-day and the off-season carry mask."""
    # off-season days neither advance the recurrence nor count as
    # missing, so they emit the carried state instead of a gap NaN
    season_today = None if component.in_season is None else component.in_season[day]
    if fast_path and season_today is None and component.static_all_valid and component.weather_valid[day].all():
        # A fully valid day with a usable static input: the gap policy reduces
        # to "every unpoisoned cell is active, the trailing count resets, and a
        # started recurrence stays started", with no partial-mask bookkeeping.
        component.trailing_gap_days.fill(0)
        component.started.fill(True)
        active = None if not component.poisoned.any() else ~component.poisoned
        return active, None
    # A cell whose static input is unusable has no recurrence to gap-manage: it
    # never starts, so it is not an elapsed missing day.
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
    carried = ~season_today & component.static_valid & component.started if season_today is not None else None
    return active, carried


def _advance_component(
    component: DailyRecurrence,
    day: int,
    active: npt.NDArray[np.bool_] | None,
    all_active: bool,
) -> None:
    """Advance one component by a day, rejecting a non-finite result from finite inputs."""
    if active is not None and not np.any(active):
        return
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


def _record_component_day(
    output: npt.NDArray[np.float64] | None,
    component: DailyRecurrence,
    day: int,
    spin_up: int,
    active: npt.NDArray[np.bool_] | None,
    all_active: bool,
    carried: npt.NDArray[np.bool_] | None,
) -> None:
    """Write one component's day into its recorded history, honoring the seasonal carry mask."""
    if day < spin_up or output is None:
        return
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


def _final_state_gaps(
    components: tuple[DailyRecurrence, ...],
) -> tuple[npt.NDArray[np.int64] | None, ...]:
    """Return each component's trailing gap counts, or None when it never started."""
    return tuple(
        component.trailing_gap_days.copy() if np.any(component.started | component.poisoned) else None
        for component in components
    )


_RecurrenceValues = tuple[npt.NDArray[np.float64] | None, ...]
_RecurrenceGaps = tuple[npt.NDArray[np.int64] | None, ...]
_FinalizedT = TypeVar("_FinalizedT")


def _allocate_history(
    component: DailyRecurrence,
    keep: bool,
    n_days: int,
    spin_up: int,
) -> npt.NDArray[np.float64] | None:
    """One component's daily history, or None when the call records none for it."""
    if not keep:
        return None
    return np.full((max(n_days - spin_up, 0), *component.weather_valid.shape[1:]), np.nan, dtype=np.float64)


def _allocate_histories(
    components: tuple[DailyRecurrence, ...],
    records: tuple[bool, ...],
    n_days: int,
    spin_up: int,
) -> _RecurrenceValues:
    """Allocate one daily history per recorded component; unrecorded slots stay None.

    A component with a Rust kernel allocates nothing here: the kernel returns the
    one history it records, so the two never hold a full history each at the same
    time. A kernel that declines gets its slot when it declines.
    """
    return tuple(
        None if component.native is not None else _allocate_history(component, keep, n_days, spin_up)
        for component, keep in zip(components, records, strict=True)
    )


def _run_native_components(
    components: tuple[DailyRecurrence, ...],
    values: list[npt.NDArray[np.float64] | None],
    records: tuple[bool, ...],
    n_days: int,
    spin_up: int,
    nan_policy: Literal["propagate", "bridge"],
    max_gap_days: int,
) -> tuple[list[bool], list[npt.NDArray[np.int64] | None]]:
    """Run each component's optional Rust kernel, in place over ``values``.

    A kernel is handed the shape its history is returned in, or None when this
    call records none for the component, so the kernel builds the only full-size
    history. Returns which components ran natively and their final gap counts; a
    component whose kernel returns None (or has none) stays for the day loop,
    which gets its slot allocated here.
    """
    native_ran = [False] * len(components)
    native_gaps: list[npt.NDArray[np.int64] | None] = [None] * len(components)
    for index, component in enumerate(components):
        if component.native is None:
            continue
        keep = records[index]
        shape = (max(n_days - spin_up, 0), *component.weather_valid.shape[1:]) if keep else None
        native_result = component.native(shape, spin_up, nan_policy, max_gap_days)
        if native_result is None:
            values[index] = _allocate_history(component, keep, n_days, spin_up)
            continue
        values[index], native_gaps[index] = native_result
        native_ran[index] = True
    return native_ran, native_gaps


def _first_output_shape(values: _RecurrenceValues) -> tuple[int, ...] | None:
    """Return the first recorded history's shape, or None when nothing was recorded."""
    for value in values:
        if value is not None:
            return value.shape
    return None


def _recorded_memory_metrics(
    memory_arrays: tuple[npt.NDArray[np.float64], ...],
    values: list[npt.NDArray[np.float64] | None],
) -> dict[str, float] | None:
    """The large-array memory metrics for the caller's arrays and recorded histories.

    A component that runs natively reports the history the kernel returns, so
    this sum stays comparable across the two paths; the kernel's own boundary
    copy of each input is a Rust-side buffer and is not part of it.
    """
    return check_large_array_memory(*memory_arrays, *(value for value in values if value is not None))


def _state_gap_days(
    components: tuple[DailyRecurrence, ...],
    native_ran: list[bool],
    native_gaps: list[npt.NDArray[np.int64] | None],
) -> tuple[npt.NDArray[np.int64] | None, ...]:
    """Each component's final gap counts, preferring the kernel's when it ran natively."""
    return tuple(
        native_gaps[index] if native_ran[index] else default
        for index, default in enumerate(_final_state_gaps(components))
    )


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
    # optional Rust kernel for this recurrence; None keeps it on the Python path
    native: NativeRecurrence | None = None
    # derived once by the runner so the day loop has a single grouping of
    # per-component state instead of parallel index spaces
    started: npt.NDArray[np.bool_] = field(init=False)
    poisoned: npt.NDArray[np.bool_] = field(init=False)
    static_all_valid: bool = field(init=False, default=False)


@overload
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
    finalize: None = None,
    output_shape: tuple[int, ...] | None = None,
) -> tuple[_RecurrenceValues, _RecurrenceGaps]: ...


@overload
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
    finalize: Callable[[_RecurrenceValues, _RecurrenceGaps], _FinalizedT],
    output_shape: tuple[int, ...] | None = None,
) -> _FinalizedT: ...


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
    finalize: Callable[[_RecurrenceValues, _RecurrenceGaps], object] | None = None,
    output_shape: tuple[int, ...] | None = None,
) -> object:
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

    ``native`` optionally runs one component's whole time axis in the Rust
    kernel instead of the day loop below: the callable returns that component's
    recorded history and final gap counts, or None to leave the component here.
    Components are independent, so a native component's day-by-day interleaving
    with the others in the loop does not affect any of their results.

    ``finalize`` optionally converts the recorded histories and final state
    into the caller's returned object. It runs inside the guarded region, so
    its allocations count as part of the recurrence: a failure there emits
    ``calculation_failed`` instead of a premature ``calculation_completed``.
    ``output_shape`` names the shape the caller actually returns, so the
    completion event reports the API shape rather than the internal
    ``(time, 1)`` normalization a one-dimensional input carries here.

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
        values = list(_allocate_histories(components, records, n_days, spin_up))
        for component in components:
            _initialize_component(component)

        # a component with a Rust kernel runs its whole time axis in one call, so
        # the day loop below only carries the components left on the Python path
        # (their own order is unchanged: they are independent of one another)
        native_ran, native_gaps = _run_native_components(
            components, values, records, n_days, spin_up, nan_policy, max_gap_days
        )
        # after the kernels so the recorded histories they return are counted
        memory_metrics = _recorded_memory_metrics(memory_arrays, values)

        for day in range(n_days):
            for index, component in enumerate(components):
                if native_ran[index]:
                    continue
                active, carried = _component_day_active(
                    component,
                    day,
                    fast_path=fast_path,
                    nan_policy=nan_policy,
                    max_gap_days=max_gap_days,
                )
                all_active = active is None or bool(active.all())
                _advance_component(component, day, active, all_active)
                _record_component_day(values[index], component, day, spin_up, active, all_active, carried)

        state_gap_days = _state_gap_days(components, native_ran, native_gaps)
        histories = tuple(values)
        if finalize is None:
            result: object = (histories, state_gap_days)
        else:
            result = finalize(histories, state_gap_days)
        logged_shape = output_shape if output_shape is not None else _first_output_shape(histories)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "calculation_completed",
            duration_ms=round(duration_ms, 2),
            output_shape=logged_shape,
            **(memory_metrics or {}),
        )
        return result
    except Exception as exc:
        log_calculation_failure(log, exc)
        raise

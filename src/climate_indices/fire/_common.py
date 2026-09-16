"""Helpers shared by the fire-weather NumPy modules."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from climate_indices.exceptions import InputTypeError, InvalidArgumentError


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


def _as_float_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce to float64, turning masked elements into NaN instead of dropping the mask.

    Non-numeric inputs are rejected rather than coerced: datetime, string, and
    object arrays would otherwise arrive as plausible but meaningless numbers,
    and complex arrays would silently discard their imaginary part.
    """
    if np.asarray(values).dtype.kind not in "biuf":
        raise InputTypeError(
            "Fire index inputs must be numeric: datetime, string, object, and complex "
            "arrays are not coerced to float64.",
            expected_type=float,
            actual_type=np.asarray(values).dtype.type,
        )
    filled = np.ma.asarray(values, dtype=np.float64).filled(np.nan)
    return np.asarray(filled, dtype=np.float64)

"""Daily Antecedent Precipitation Index (Kohler and Linsley, 1951)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.exceptions import InvalidArgumentError
from climate_indices.fire._common import _apply_gap_policy, _static_spatial_array, _validate_recurrence_options
from climate_indices.flood._common import _validated_daily


@dataclass(frozen=True)
class APIState:
    """State for resuming an antecedent precipitation recurrence.

    ``trailing_gap_days=None`` means no valid day has started any cell;
    ``-1`` marks individual cells that have not started. A NaN ``api`` is only
    valid where ``trailing_gap_days`` is ``>= 0`` (the cell has started); a
    not-started cell (``-1`` or ``None``) holds a number. Values are in mm.
    """

    api: npt.NDArray[np.float64]
    trailing_gap_days: npt.NDArray[np.int64] | None


@dataclass(frozen=True)
class APIResult:
    """Antecedent precipitation values and copied final state."""

    values: npt.NDArray[np.float64] | xr.DataArray
    state: APIState


def _validate_decay(k: object) -> None:
    """Reject a decay constant that is not a real number strictly between zero and one."""
    if (  # NOSONAR S2589: false positive, valid k passes this check (test_flood_antecedent)
        isinstance(k, bool)
        or not isinstance(k, (int, float, np.integer, np.floating))
        or not np.isfinite(k)
        or k <= 0
        or k >= 1
    ):
        raise InvalidArgumentError(
            "k must be a real scalar (int or float) strictly between zero and one.",
            argument_name="k",
            argument_value=str(k),
        )


def _validated_gaps(trailing_gap_days: npt.NDArray[np.int64], internal_shape: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """Return trailing gap counts as int64 after checking they are integers >= -1."""
    raw_gaps = _static_spatial_array(trailing_gap_days, internal_shape, "initial_state.trailing_gap_days")
    # int64 max is rejected on purpose: the gap counter increments each missing day and would wrap negative
    if (
        np.any(~np.isfinite(raw_gaps))
        or np.any(raw_gaps < -1)
        or np.any(raw_gaps >= float(np.iinfo(np.int64).max))
        or np.any(raw_gaps != np.floor(raw_gaps))
    ):
        raise InvalidArgumentError(
            "initial_state.trailing_gap_days must be integers >= -1 and below 2**63 "
            "(float64 precision; values near the int64 maximum are rejected so the day counter cannot overflow).",
            argument_name="initial_state.trailing_gap_days",
        )
    return raw_gaps.astype(np.int64)


def _resume_state(
    initial_state: APIState | None, internal_shape: tuple[int, ...]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return the validated starting ``(api, gaps)`` arrays, defaulting to a zero seed."""
    if initial_state is None:
        return np.zeros(internal_shape, dtype=np.float64), np.full(internal_shape, -1, dtype=np.int64)
    if not isinstance(initial_state, APIState):
        raise InvalidArgumentError("initial_state must be an APIState.", argument_name="initial_state")
    current = _static_spatial_array(initial_state.api, internal_shape, "initial_state.api")
    if initial_state.trailing_gap_days is None:
        gaps = np.full(internal_shape, -1, dtype=np.int64)
    else:
        gaps = _validated_gaps(initial_state.trailing_gap_days, internal_shape)
    if np.any(np.isinf(current)) or np.any(current < 0) or np.any(np.isnan(current) & (gaps < 0)):
        raise InvalidArgumentError(
            "initial_state.api must be non-negative or NaN only in a started cell.",
            argument_name="initial_state.api",
        )
    return current, gaps


def antecedent_precipitation_index(
    precipitation: npt.ArrayLike,
    k: float,
    *,
    initial_state: APIState | None = None,
    return_state: bool = False,
    spin_up: int = 0,
    nan_policy: Literal["propagate", "bridge"] = "propagate",
    max_gap_days: int = 0,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64] | APIResult:
    """Compute daily accumulated wetness indicating flood potential, not flooding.

    Kohler and Linsley (1951), Eq. (3) gives ``I₁ = k · I₀``; their text adds
    rain observed on the current day: ``API_t = k * API_(t-1) + P_t``.
    Some sources instead lag precipitation, using
    ``k * (API_(t-1) + P_(t-1))``; that variant is not implemented.
    For typical ``k`` (0.85–0.90 in the eastern and central US) an assumed
    initial value converges within several weeks; the default seed is zero.
    No external tabulated numeric oracle is available for this index.

    Args:
        precipitation: Non-negative daily precipitation in mm; a 1-D series,
            2-D ``(years, days)`` series, or Spatial Block ``(time, *cells)``.
            Missing observations may be NaN or masked.
        k: Decay constant strictly between zero and one.
        initial_state: State returned by a previous call. Defaults to a zero seed.
        return_state: Return :class:`APIResult` with final state.
        spin_up: Leading days computed but omitted from the output.
        nan_policy: ``"propagate"`` poisons a started recurrence at a missing
            day; ``"bridge"`` skips up to ``max_gap_days`` missing days.
        max_gap_days: Zero for ``"propagate"``; positive for ``"bridge"``.
        spatial_time_major: Declare a Spatial Block whose first cell axis has
            length 12 or 366; 2-D input is always a ``(years, days)`` series.

    Returns:
        Index in mm, in the input's shape with the leading ``spin_up`` days removed;
        for 2-D input, nonzero ``spin_up`` returns the remaining flattened
        daily record because it may not contain whole years. With
        ``return_state=True``, returns :class:`APIResult` instead.

    Raises:
        InvalidArgumentError: If precipitation, k, configuration or state is invalid.
        InputTypeError: If precipitation or a state field is non-numeric.
        DataShapeError: If precipitation has no time axis.
        ValueError: If a Spatial Block has an undeclared ambiguous shape.
    """
    _validate_decay(k)
    _validate_recurrence_options(nan_policy, max_gap_days, spin_up, "initial_state", None, initial_state)
    # NumPy integers of a narrow or unsigned width would overflow when subtracted from the series length
    spin_up = int(spin_up)
    values = _validated_daily(precipitation, "precipitation", spatial_time_major, None)
    series = values.reshape(-1) if values.ndim == 2 else values
    spatial_shape = series.shape[1:]
    internal_shape = spatial_shape or (1,)
    series = series.reshape(series.shape[0], *internal_shape)
    current, gaps = _resume_state(initial_state, internal_shape)
    started = gaps >= 0
    poisoned = np.isnan(current)
    result = np.full((max(series.shape[0] - spin_up, 0), *internal_shape), np.nan, dtype=np.float64)
    for day, rain in enumerate(series):
        active = _apply_gap_policy(
            current,
            np.isfinite(rain),
            np.ones(internal_shape, dtype=bool),
            started,
            poisoned,
            gaps,
            nan_policy=nan_policy,
            max_gap_days=max_gap_days,
        )
        current[active] = k * current[active] + rain[active]
        if day >= spin_up:
            result[day - spin_up] = np.where(active, current, np.nan)
    output = result.reshape((result.shape[0], *spatial_shape))
    if values.ndim == 2 and spin_up == 0:
        output = output.reshape(values.shape)
    if not return_state:
        return output
    return APIResult(
        values=output,
        state=APIState(
            api=current.reshape(spatial_shape).copy(),
            trailing_gap_days=gaps.reshape(spatial_shape).copy() if np.any(started | poisoned) else None,
        ),
    )

"""Flood-potential events: runs of a daily index above a threshold.

The Flood Index and the other flood indices describe flood potential, not
observed flooding. An event here is a maximal run of consecutive days on which
the index is finite and strictly above ``threshold``; it is a summary of
exceedance, not a claim that flooding occurred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from climate_indices.exceptions import InputTypeError, InvalidArgumentError


@dataclass(frozen=True)
class FloodEvents:
    """Flood-potential events, one entry per event, ordered by cell then onset.

    Attributes:
        cell: Spatial index of each event, shape ``(events, ndim - 1)``; the
            second axis is empty for a 1-D series.
        onset: First day of each event, as a position on the time axis.
        end: One past the last day of each event (exclusive), so
            ``end - onset`` is the duration.
        duration: Length of each event in days.
        peak: Largest index value within each event.
        severity: Sum of the index over each event's days.
    """

    cell: npt.NDArray[np.intp]
    onset: npt.NDArray[np.intp]
    end: npt.NDArray[np.intp]
    duration: npt.NDArray[np.intp]
    peak: npt.NDArray[np.float64]
    severity: npt.NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.onset.shape[0])


def _validated_index(index: Any) -> npt.NDArray[np.float64]:
    try:
        values = np.ma.filled(np.ma.asarray(index, dtype=np.float64), np.nan)
    except (TypeError, ValueError) as exc:
        raise InputTypeError("index must be numeric.", expected_type=float, actual_type=type(index)) from exc
    if values.ndim == 0:
        raise InvalidArgumentError("index needs a time axis.", argument_name="index")
    return np.asarray(values, dtype=np.float64)


def _validate_options(threshold: float, min_duration: int) -> None:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise InvalidArgumentError("threshold must be a finite number.", argument_name="threshold")
    if not np.isfinite(threshold):
        raise InvalidArgumentError("threshold must be a finite number.", argument_name="threshold")
    if (
        isinstance(min_duration, (bool, np.bool_))
        or not isinstance(min_duration, (int, np.integer))
        or min_duration < 1
    ):
        raise InvalidArgumentError("min_duration must be a positive integer.", argument_name="min_duration")


def flood_events(
    index: npt.ArrayLike,
    *,
    threshold: float = 0.0,
    min_duration: int = 1,
) -> FloodEvents:
    """Find the runs of a daily flood index that stay above a threshold.

    Args:
        index: Daily index such as :func:`flood_index` output, a 1-D series or
            a time-first Spatial Block ``(time, *cells)``. NaN and masked days
            are never part of an event and end any run.
        threshold: A day counts when ``index > threshold``. The default of zero
            matches the Flood Index's above-average-annual-maximum reading.
        min_duration: Shortest event kept, in days.

    Returns:
        A :class:`FloodEvents` with onset, end, duration, peak, and severity for
        every event, ordered by cell and then onset.

    Raises:
        InvalidArgumentError: If ``index`` is a scalar, ``threshold`` is not a
            finite number, or ``min_duration`` is not a positive integer.
        InputTypeError: If ``index`` is non-numeric.
    """
    _validate_options(threshold, min_duration)

    values = _validated_index(index)
    days = values.shape[0]
    cell_shape = values.shape[1:]
    block = values.reshape(days, -1)
    cells = block.shape[1]

    # time runs along the last axis so one flat array holds every cell's series
    series = np.ascontiguousarray(block.T)
    with np.errstate(invalid="ignore"):
        exceeds = series > threshold
    padded = np.zeros((cells, days + 2), dtype=np.int8)
    padded[:, 1:-1] = exceeds
    change = np.diff(padded, axis=1)
    start_cell, onset = np.nonzero(change == 1)
    _, end = np.nonzero(change == -1)

    keep = (end - onset) >= min_duration
    start_cell, onset, end = start_cell[keep], onset[keep], end[keep]

    flat = np.append(np.where(exceeds, series, 0.0).ravel(), 0.0)
    offsets = start_cell * days
    # reduceat reads [onset, end) from the interleaved bounds; the trailing zero keeps
    # an end at the last cell's final day in bounds, and the odd slots are discarded
    if onset.size:
        bounds = np.empty(2 * onset.size, dtype=np.intp)
        bounds[0::2] = offsets + onset
        bounds[1::2] = offsets + end
        severity = np.add.reduceat(flat, bounds)[0::2]
        peak = np.maximum.reduceat(flat, bounds)[0::2]
    else:
        severity = peak = np.empty(0, dtype=np.float64)

    cell = (
        np.stack(np.unravel_index(start_cell, cell_shape), axis=1).astype(np.intp)
        if cell_shape
        else np.empty((onset.size, 0), dtype=np.intp)
    )
    return FloodEvents(
        cell=cell,
        onset=onset.astype(np.intp),
        end=end.astype(np.intp),
        duration=(end - onset).astype(np.intp),
        peak=np.asarray(peak, dtype=np.float64),
        severity=np.asarray(severity, dtype=np.float64),
    )

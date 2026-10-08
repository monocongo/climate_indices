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

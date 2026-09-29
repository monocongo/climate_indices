"""Calibration Period resolution: one owner for a requested window of years.

Every fitted or ranked index turns a requested Calibration Period (inclusive
calendar years) into a row slice of an array folded to ``(years, periods, ...)``.
This module owns that arithmetic and the two policies an index can apply to a
window the record does not cover: ``"clamp"`` it to the whole record, or
``"reject"`` it. Each index names its policy at its call site. A reversed window
(start year after end year) is rejected under both, since no policy can give it a
meaning (#1231).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from climate_indices.exceptions import InvalidArgumentError

CalibrationPolicy = Literal["clamp", "reject"]


class CalibrationPeriodError(InvalidArgumentError, ValueError):
    """A requested Calibration Period the record cannot represent.

    It is both an :class:`InvalidArgumentError` and a :class:`ValueError` because
    EDDI has always raised the former and Palmer the latter.
    """


@dataclass(frozen=True)
class CalibrationPeriod:
    """A Calibration Period resolved against a record of whole years.

    ``start_year`` and ``end_year`` are inclusive, and ``end_year`` is never before
    ``start_year``, so ``n_years`` is at least one.
    """

    start_year: int
    end_year: int
    start_index: int  # row of ``start_year`` on the folded year axis

    @property
    def n_years(self) -> int:
        return self.end_year - self.start_year + 1

    @property
    def end_index(self) -> int:
        """Row of ``end_year``, inclusive."""
        return self.start_index + self.n_years - 1

    @property
    def rows(self) -> slice:
        return slice(self.start_index, self.end_index + 1)


def resolve_calibration_period(
    data_start_year: int,
    n_years: int,
    calibration_start_year: int,
    calibration_end_year: int,
    *,
    policy: CalibrationPolicy,
) -> CalibrationPeriod:
    """Resolve a requested Calibration Period against a record of ``n_years`` years.

    Args:
        data_start_year: Year of the record's first row.
        n_years: Number of whole years in the folded record.
        calibration_start_year: First requested year, inclusive.
        calibration_end_year: Last requested year, inclusive.
        policy: What to do with a window that starts before the record or ends after
            it. ``"clamp"`` uses the whole record instead, except that a window ending
            exactly one year after the record keeps its start and is cut at the
            record's last year: the fits have always honoured that start (the WMO
            1981-2010 window on a record ending in 2009 fits 1981-2009), and making
            it uniform with the other out-of-range windows would change results.
            ``"reject"`` raises :class:`CalibrationPeriodError` for any window the
            record does not cover.

    Raises:
        CalibrationPeriodError: The window is reversed (under either policy), or the
            record does not cover it (under ``"reject"``).
    """
    if calibration_start_year > calibration_end_year:
        raise CalibrationPeriodError(
            f"Invalid calibration period: initial year ({calibration_start_year}) "
            f"is after final year ({calibration_end_year})",
            argument_name="calibration_year_initial",
            argument_value=str(calibration_start_year),
        )
    data_end_year = data_start_year + n_years - 1
    if policy == "reject":
        _reject_uncovered(data_start_year, data_end_year, calibration_start_year, calibration_end_year)
    elif calibration_start_year < data_start_year or calibration_end_year > data_end_year + 1:
        calibration_start_year, calibration_end_year = data_start_year, data_end_year
    else:
        calibration_end_year = min(calibration_end_year, data_end_year)
    return CalibrationPeriod(calibration_start_year, calibration_end_year, calibration_start_year - data_start_year)


def _reject_uncovered(data_start_year: int, data_end_year: int, start_year: int, end_year: int) -> None:
    if start_year < data_start_year:
        raise CalibrationPeriodError(
            f"Invalid calibration period: calibration start year ({start_year}) "
            f"is before data start year ({data_start_year})",
            argument_name="calibration_year_initial",
            argument_value=str(start_year),
        )
    if end_year > data_end_year:
        raise CalibrationPeriodError(
            f"Invalid calibration period: calibration end year ({end_year}) is after data end year ({data_end_year})",
            argument_name="calibration_year_final",
            argument_value=str(end_year),
        )

"""Calibration Period resolution: one owner for a requested window of years.

Every fitted or ranked index turns a requested Calibration Period (inclusive
calendar years) into a row slice of an array folded to ``(years, periods, ...)``.
This module owns coordinate inference, the single-series non-NaN sample check,
and the two policies an index can apply to a window the record does not cover:
``"clamp"`` it to the whole record, or ``"reject"`` it. Each index names its policy at its call site. A reversed window
(start year after end year) is rejected under both, since no policy can give it a
meaning (#1231).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from climate_indices.exceptions import InsufficientDataError, InvalidArgumentError

# The single-series preflight enforces this minimum; distribution fits only warn.
MIN_CALIBRATION_YEARS = 30

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

    @staticmethod
    def infer_years(
        time_values: npt.NDArray[np.datetime64], *, complete_years: bool = False, year_start_month: int = 1
    ) -> tuple[int, int]:
        """Infer inclusive year bounds from validated labeled time endpoints.

        The adapter includes partial calendar years. Daily flood callers request
        only complete annual periods, labeled by their start year. Bounds can be
        reversed when no complete period exists; the kernel retains responsibility
        for rejecting that window, rather than constructing a resolved period here.
        This reads coordinates only, never climate values or Dask tasks.
        """
        first_year = int(pd.Timestamp(time_values[0]).year)
        last = pd.Timestamp(time_values[-1])
        if not complete_years:
            return first_year, int(last.year)
        if year_start_month == 1:
            last_year = last.year if last.is_year_end else last.year - 1
        else:
            end_month = year_start_month - 1
            end_year = last.year if last.month >= year_start_month else last.year - 1
            if last.month == end_month and last.is_month_end:
                end_year = last.year
            last_year = end_year - 1
        return first_year, int(last_year)

    def validate_non_nan_sample_size(
        self,
        time_values: npt.NDArray[np.datetime64],
        values: npt.NDArray[Any],
        min_years: int = MIN_CALIBRATION_YEARS,
    ) -> None:
        """Enforce the single-series preflight minimum within this period.

        Issue #1156 removed grid-wide rejection based on the first cell. Gridded
        and Dask inputs still bypass this check; new per-cell thresholds are outside
        this refactor. Daily counts use 365 samples per year, as the preflight runs
        before all-leap conversion. Fitting-quality warnings remain independent.
        """
        time_index = pd.DatetimeIndex(time_values)
        time_years = time_index.year.values
        calibration_mask = (time_years >= self.start_year) & (time_years <= self.end_year)
        calibration_values = values[calibration_mask]
        if len(calibration_values) == 0:
            raise InsufficientDataError(
                message=(
                    f"Calibration period ({self.start_year}-{self.end_year}) "
                    f"contains no data points. Check that calibration years overlap with time coordinate range."
                ),
                non_zero_count=0,
                required_count=min_years,
            )
        non_nan_count = int(np.sum(~np.isnan(calibration_values)))
        freq = pd.infer_freq(time_index)
        if freq in ("MS", "ME", "M"):
            periods_per_year = 12
        elif freq == "D":
            periods_per_year = 365
        else:
            periods_per_year = len(calibration_values) // self.n_years
            if periods_per_year == 0:
                periods_per_year = 12
        effective_years = non_nan_count / periods_per_year
        if effective_years < min_years:
            raise InsufficientDataError(
                message=(
                    f"Insufficient non-NaN data in calibration period ({self.start_year}-{self.end_year}). "
                    f"Found {non_nan_count} non-NaN values ({effective_years:.1f} effective years), "
                    f"but at least {min_years} years of non-NaN data required for reliable distribution fitting."
                ),
                non_zero_count=non_nan_count,
                required_count=int(min_years * periods_per_year),
            )

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

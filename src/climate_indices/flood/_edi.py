"""Fixed-window Effective Drought Index from daily effective precipitation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError

_DAYS_PER_YEAR = 366


def edi(
    pe: npt.ArrayLike,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    *,
    duration: int = 365,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64]:
    """Standardize daily effective precipitation against a Calibration Period.

    This is the fixed-window form of Byun and Wilhite (1999), Eq. 9:
    ``DEP = PE - MEP``, ``PRN = DEP / H_duration``, and
    ``EDI = PRN / SD(PRN)``. With fixed duration the harmonic factor cancels;
    the result is ``DEP / SD(PE)`` for each calendar day. SD is the population
    standard deviation (``ddof=0``). The published variable-duration dry-spell
    extension and five-day smoothing of daily climatology are **not** applied.
    Only the 365-day default is the recorded EDI convention; other durations
    are caller experiments. This index describes moisture conditions, not
    observed flooding.

    Args:
        pe: Daily effective precipitation in mm from
            :func:`effective_precipitation`. Input starts on January 1, with
            366 positional days per year (including February 29 every year).
            For Gregorian precipitation, convert to this layout *before*
            calculating PE: use :func:`climate_indices.utils.transform_to_366day`
            for 1-D input or :meth:`climate_indices.utils.DailyCalendarPlan.to_all_leap`
            for Spatial Blocks. Both fill non-leap-year February 29 with the
            mean of February 28 and March 1. Accepts 1-D, ``(years, 366)``, or a Spatial Block
            ``(time, *cells)``; missing values may be NaN or masked. The PE
            window's leading NaNs and any other gaps remain NaN in the result.
        data_start_year: Calendar year of the first observation.
        calibration_year_initial: First year of the Calibration Period.
        calibration_year_final: Last year of the Calibration Period (inclusive).
            Only complete years may be used to fit the climatology.
        duration: PE window length, default 365. Accepted for symmetry with
            :func:`effective_precipitation`; the fixed-window harmonic factor
            cancels during standardization, so this argument does not affect EDI.
        spatial_time_major: Declare an ambiguous Spatial Block whose first cell
            axis has length 12 or 366. A 2-D array is always ``(years, 366)``.

    Returns:
        Dimensionless daily EDI, in the input shape. A calendar day with fewer
        than two finite calibration values or zero variance yields NaN for that
        cell in every year; a partial final year is allowed outside calibration.

    Raises:
        InvalidArgumentError: If duration, PE values, or Calibration Period are invalid.
        InputTypeError: If PE is non-numeric.
        DataShapeError: If the input has no time axis or an invalid year layout.
        ValueError: If a Spatial Block has an undeclared ambiguous shape.
    """
    if isinstance(duration, (bool, np.bool_)) or not isinstance(duration, (int, np.integer)) or duration < 1:
        raise InvalidArgumentError(
            "duration must be a positive integer.", argument_name="duration", argument_value=str(duration)
        )
    if np.asarray(pe).dtype.kind not in "biuf":
        raise InputTypeError("pe must be numeric.", expected_type=float, actual_type=np.asarray(pe).dtype.type)
    values = np.asarray(np.ma.asarray(pe, dtype=np.float64).filled(np.nan), dtype=np.float64)
    if values.ndim == 0:
        raise DataShapeError("pe must have a time axis.", expected_shape="(time, ...)", actual_shape=values.shape)
    if values.ndim == 2 and values.shape[1] != _DAYS_PER_YEAR:
        raise DataShapeError(
            "pe must have 366 days per year.", expected_shape="(years, 366)", actual_shape=values.shape
        )
    if values.ndim > 2 and not spatial_time_major and values.shape[1] in (12, _DAYS_PER_YEAR):
        raise ValueError(
            f"Invalid shape of input array: {values.shape} -- a (time, *cells) block whose first "
            "cell axis is a calendar period length is ambiguous with a (years, periods, *cells) "
            "array; declare it with spatial_time_major=True"
        )
    if np.any(np.isinf(values)) or np.any(values < 0):
        raise InvalidArgumentError("pe must be non-negative and finite or NaN.", argument_name="pe")

    series = values.reshape(-1) if values.ndim == 2 else values
    complete_years = series.shape[0] // _DAYS_PER_YEAR
    if (
        isinstance(data_start_year, (bool, np.bool_))
        or not isinstance(data_start_year, (int, np.integer))
        or any(
            isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer))
            for y in (calibration_year_initial, calibration_year_final)
        )
        or calibration_year_initial < data_start_year
        or calibration_year_final <= calibration_year_initial
        or calibration_year_final >= data_start_year + complete_years
    ):
        raise InvalidArgumentError(
            "Calibration Period must span at least two complete input years.",
            argument_name="calibration_year_initial/calibration_year_final",
        )
    padded = series
    if series.shape[0] % _DAYS_PER_YEAR:
        padded = np.full(
            (((series.shape[0] + _DAYS_PER_YEAR - 1) // _DAYS_PER_YEAR) * _DAYS_PER_YEAR, *series.shape[1:]), np.nan
        )
        padded[: series.shape[0]] = series
    years = padded.reshape(padded.shape[0] // _DAYS_PER_YEAR, _DAYS_PER_YEAR, *series.shape[1:])
    calibration = years[calibration_year_initial - data_start_year : calibration_year_final - data_start_year + 1]
    valid = np.isfinite(calibration)
    counts = valid.sum(axis=0)
    mean = np.divide(
        np.where(valid, calibration, 0).sum(axis=0), counts, out=np.full(counts.shape, np.nan), where=counts > 0
    )
    deviations = np.where(valid, calibration - mean, 0)
    variance = np.divide(
        (deviations * deviations).sum(axis=0), counts, out=np.full(counts.shape, np.nan), where=counts > 1
    )
    result = np.full(years.shape, np.nan)
    rounding = 8 * np.finfo(np.float64).eps * np.abs(mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(years - mean, np.sqrt(variance), out=result, where=variance > rounding * rounding)
    return result.reshape(padded.shape[0], *series.shape[1:])[: series.shape[0]].reshape(values.shape)

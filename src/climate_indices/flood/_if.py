"""Flood Index standardized against annual maxima of effective precipitation."""

from __future__ import annotations

from datetime import date

import numpy as np
import numpy.typing as npt

from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError

_DAYS_PER_YEAR = 366


def flood_index(
    pe: npt.ArrayLike,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    *,
    year_start_month: int,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64]:
    """Standardize daily PE against annual maxima in the Calibration Period.

    ``I_F = (PE - mean(PE_max)) / SD(PE_max)`` uses the population standard
    deviation (``ddof=0``). This describes flood potential, not flooding.
    The shared PE input uses the harmonic Byun and Wilhite Eq. (2) kernel;
    whether Deo et al. (2015) used that kernel remains unverified.

    Args:
        pe: Effective precipitation in mm from :func:`effective_precipitation`.
            Input begins January 1 and has 366 positional days per year,
            including February 29 in every year. Convert Gregorian rainfall
            to the all-leap layout before computing PE. Accepts a 1-D series,
            ``(years, 366)``, or a Spatial Block ``(time, *cells)``. Missing
            observations may be NaN or masked.
        data_start_year: Calendar year of the first observation.
        calibration_year_initial: First start year of a complete annual period
            used for calibration.
        calibration_year_final: Last start year of a complete annual period
            used for calibration (inclusive).
        year_start_month: Calendar month beginning each annual period (1–12).
            Partial leading and trailing periods are excluded from calibration.
        spatial_time_major: Declare an ambiguous Spatial Block whose first
            cell axis has length 12 or 366. A 2-D array is ``(years, 366)``.

    Returns:
        Dimensionless daily index in the input shape. At least two finite
        annual maxima with nonzero variance are needed per cell; otherwise
        that cell's index is NaN. Missing PE remains NaN.

    Raises:
        InvalidArgumentError: If PE, year_start_month, or Calibration Period
            is invalid.
        InputTypeError: If PE is non-numeric.
        DataShapeError: If the input has no time axis or an invalid year layout.
        ValueError: If a Spatial Block has an undeclared ambiguous shape.
    """
    if (
        isinstance(year_start_month, (bool, np.bool_))
        or not isinstance(year_start_month, (int, np.integer))
        or not 1 <= year_start_month <= 12
    ):
        raise InvalidArgumentError(
            "year_start_month must be an integer from 1 to 12.", argument_name="year_start_month"
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

    if (
        isinstance(data_start_year, (bool, np.bool_))
        or not isinstance(data_start_year, (int, np.integer))
        or any(
            isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer))
            for y in (calibration_year_initial, calibration_year_final)
        )
    ):
        raise InvalidArgumentError(
            "Calibration years must be integers.",
            argument_name="data_start_year/calibration_year_initial/calibration_year_final",
        )
    series = values.reshape(-1) if values.ndim == 2 else values
    offset = date(2000, int(year_start_month), 1).timetuple().tm_yday - 1
    last_year = data_start_year + (series.shape[0] - offset) // _DAYS_PER_YEAR - 1
    if (
        calibration_year_initial < data_start_year
        or calibration_year_final > last_year
        or calibration_year_final <= calibration_year_initial
    ):
        raise InvalidArgumentError(
            "Calibration Period must span at least two complete annual periods.",
            argument_name="calibration_year_initial/calibration_year_final",
        )
    maxima = []
    for year in range(calibration_year_initial, calibration_year_final + 1):
        start = (year - data_start_year) * _DAYS_PER_YEAR + offset
        annual = series[start : start + _DAYS_PER_YEAR]
        maxima.append(np.max(np.where(np.isfinite(annual), annual, -np.inf), axis=0))
    sample = np.stack(maxima)
    valid = np.isfinite(sample)
    counts = valid.sum(axis=0)
    mean = np.divide(
        np.where(valid, sample, 0).sum(axis=0), counts, out=np.full(counts.shape, np.nan), where=counts > 0
    )
    deviations = np.where(valid, sample - mean, 0)
    variance = np.divide(
        (deviations * deviations).sum(axis=0), counts, out=np.full(counts.shape, np.nan), where=counts > 1
    )
    result = np.full(series.shape, np.nan)
    rounding = 8 * np.finfo(np.float64).eps * np.abs(mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(series - mean, np.sqrt(variance), out=result, where=variance > rounding * rounding)
    return result.reshape(values.shape)

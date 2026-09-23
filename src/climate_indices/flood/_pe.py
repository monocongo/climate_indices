"""Fixed-window effective precipitation (Byun and Wilhite, 1999, Eq. 2)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import correlate1d

from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError


def effective_precipitation(
    precipitation: npt.ArrayLike,
    *,
    duration: int = 365,
    spatial_time_major: bool = False,
) -> npt.NDArray[np.float64]:
    """Calculate daily effective precipitation from a fixed rolling window.

    This is accumulated wetness relevant to flood *potential*, not observed
    flooding. Uses the harmonic double sum of Byun and Wilhite (1999), Eq. 2:
    ``PE_t = sum(n=1..D, sum(m=1..n, P[t-m+1]) / n)``. The first
    ``duration - 1`` days and any window containing a missing day are NaN;
    later complete windows recover without bridging missing observations.

    Args:
        precipitation: Daily precipitation in mm, as a 1-D series, a 2-D
            ``(years, days)`` series, or a Spatial Block ``(time, *cells)``.
            Missing observations may be NaN or masked.
        duration: Number of days in each window, 365 by default.
        spatial_time_major: Declare a Spatial Block when its first cell axis
            has length 12 or 366, which is ambiguous with a calendar layout.
            It does not apply to 1-D or 2-D input; a 2-D array is always read
            as ``(years, days)`` (ADR-0009).

    Returns:
        Effective precipitation in mm, with the same shape as the input.

    Raises:
        InvalidArgumentError: If duration is invalid, or precipitation is
            negative or infinite.
        DataShapeError: If input has no time axis.
        ValueError: If a Spatial Block has an undeclared ambiguous shape.
    """
    if isinstance(duration, (bool, np.bool_)) or not isinstance(duration, (int, np.integer)) or duration < 1:
        raise InvalidArgumentError(
            "duration must be a positive integer.",
            argument_name="duration",
            argument_value=str(duration),
            valid_values="A positive integer",
        )
    if np.asarray(precipitation).dtype.kind not in "biuf":
        raise InputTypeError(
            "precipitation must be numeric.",
            expected_type=float,
            actual_type=np.asarray(precipitation).dtype.type,
        )
    values = np.asarray(np.ma.asarray(precipitation, dtype=np.float64).filled(np.nan), dtype=np.float64)
    if values.ndim == 0:
        raise DataShapeError(
            "precipitation must have a time axis.", expected_shape="(time, ...)", actual_shape=values.shape
        )
    if np.any(np.isinf(values)) or np.any(values < 0):
        raise InvalidArgumentError(
            "precipitation must be non-negative and finite or NaN.",
            argument_name="precipitation",
            argument_value="negative or infinite value",
            valid_values="Non-negative daily precipitation or NaN",
        )
    if values.ndim > 2 and not spatial_time_major and values.shape[1] in (12, 366):
        raise ValueError(
            f"Invalid shape of input array: {values.shape} -- a (time, *cells) block whose first "
            "cell axis is a calendar period length is ambiguous with a (years, periods, *cells) "
            "array; declare it with spatial_time_major=True"
        )
    series = values.reshape(-1) if values.ndim == 2 else values
    result = np.full(series.shape, np.nan, dtype=np.float64)
    if series.size and series.shape[0] >= duration:
        # w_m = sum(n=m..D, 1/n); correlate1d takes oldest-first weights.
        weights = np.cumsum((1.0 / np.arange(1, duration + 1, dtype=np.float64))[::-1])[::-1]
        correlate1d(
            series,
            weights[::-1],
            axis=0,
            origin=(duration - 1) // 2,
            mode="constant",
            output=result,
        )
        result[: duration - 1] = np.nan
    return result.reshape(values.shape)

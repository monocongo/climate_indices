"""Helpers shared by the flood-index NumPy modules."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError

_DAYS_PER_YEAR = 366


def _validated_daily(
    values: npt.ArrayLike, name: str, spatial_time_major: bool, days_per_year: int | None
) -> npt.NDArray[np.float64]:
    """Return a daily input as float64 with masked values as NaN, or raise for an invalid layout or value.

    ``days_per_year`` is the required length of a 2-D ``(years, days)`` array's day axis; ``None`` accepts any length.
    """
    if np.asarray(values).dtype.kind not in "biuf":
        raise InputTypeError(f"{name} must be numeric.", expected_type=float, actual_type=np.asarray(values).dtype.type)
    array = np.asarray(np.ma.asarray(values, dtype=np.float64).filled(np.nan), dtype=np.float64)
    if array.ndim == 0:
        raise DataShapeError(f"{name} must have a time axis.", expected_shape="(time, ...)", actual_shape=array.shape)
    if days_per_year is not None and array.ndim == 2 and array.shape[1] != days_per_year:
        raise DataShapeError(
            f"{name} must have {days_per_year} days per year.",
            expected_shape=f"(years, {days_per_year})",
            actual_shape=array.shape,
        )
    if array.ndim > 2 and not spatial_time_major and array.shape[1] in (12, _DAYS_PER_YEAR):
        raise ValueError(
            f"Invalid shape of input array: {array.shape} -- a (time, *cells) block whose first "
            "cell axis is a calendar period length is ambiguous with a (years, periods, *cells) "
            "array; declare it with spatial_time_major=True"
        )
    if np.any(np.isinf(array)) or np.any(array < 0):
        raise InvalidArgumentError(f"{name} must be non-negative and finite or NaN.", argument_name=name)
    return array


def _validated_pe(pe: npt.ArrayLike, spatial_time_major: bool) -> npt.NDArray[np.float64]:
    """Return PE as float64 with masked values as NaN, or raise for an invalid layout or value."""
    return _validated_daily(pe, "pe", spatial_time_major, _DAYS_PER_YEAR)

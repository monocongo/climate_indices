"""Helpers shared by the flood-index NumPy modules."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError

_DAYS_PER_YEAR = 366


def _validated_pe(pe: npt.ArrayLike, spatial_time_major: bool) -> npt.NDArray[np.float64]:
    """Return PE as float64 with masked values as NaN, or raise for an invalid layout or value."""
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
    return values

"""Run theory: identify spells below or above a threshold in an index time series.

A *run* is a maximal contiguous sequence of time steps on one side of a threshold
(Yevjevich, 1967). For drought monitoring a run below a negative SPI/SPEI
threshold is a drought event; a run above a positive threshold is a wet event.
The same primitive applies to any index series, standardized or not (SPI, SPEI,
EDDI, PDSI, KBDI): see ``docs/algorithm-reference.md``.

Conventions:

- ``direction`` is explicit and is never inferred from the sign of ``threshold``.
  ``direction="below"`` with ``threshold=-1.0`` is the usual drought definition,
  and ``direction="above"`` with ``threshold=1.0`` the usual wet definition, but
  any combination is allowed (for example "above -0.5" for a mild wet spell).
- A NaN value is not in any run and terminates the run it would otherwise
  continue, so a gap splits a run in two.
- A value exactly equal to ``threshold`` is not in a run (comparisons are
  strict).
- ``min_duration`` filters runs *after* identification, and ``interarrival`` is
  then computed between the surviving runs only.
- Masked array elements count as missing, like NaN; non-numeric arrays
  (datetime, string, object, or complex) are rejected with
  :class:`~climate_indices.exceptions.InputTypeError`.

The functions here return run metrics only: spatial aggregation, period slicing,
and DataFrame formatting stay downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.exceptions import (
    DataShapeError,
    DimensionMismatchError,
    InputTypeError,
    InvalidArgumentError,
)

__all__ = ["RunSet", "identify_runs", "identify_runs_xarray"]

Direction = Literal["below", "above"]

_VALID_DIRECTIONS: tuple[str, ...] = ("below", "above")


@dataclass(frozen=True, eq=False)
class RunSet:
    """Metrics for the runs found in a time series, one array entry per run.

    Runs appear in time order. All arrays have the same length, and an index
    without runs is represented by zero-length arrays rather than ``None``.

    Attributes:
        start_index: Index of each run's first time step.
        end_index: Index of each run's last time step (inclusive).
        duration: Number of time steps in each run.
        magnitude: Cumulative deviation from the threshold (the sum of
            ``threshold - value`` for ``direction="below"``, ``value - threshold``
            for ``direction="above"``), always positive.
        intensity: Mean deviation per time step, ``magnitude / duration``.
        peak_value: Most extreme value in each run: the minimum for
            ``direction="below"``, the maximum for ``direction="above"``.
        peak_index: Index of ``peak_value``; the first occurrence when tied.
        interarrival: Number of time steps from each run's start to the next
            run's start, computed after ``min_duration`` filtering; the last
            run has no successor and holds NaN.

    Instances compare by value, treating NaN entries as equal, and are
    unhashable because their fields are arrays.
    """

    start_index: npt.NDArray[np.int64]
    end_index: npt.NDArray[np.int64]
    duration: npt.NDArray[np.int64]
    magnitude: npt.NDArray[np.float64]
    intensity: npt.NDArray[np.float64]
    peak_value: npt.NDArray[np.float64]
    peak_index: npt.NDArray[np.int64]
    interarrival: npt.NDArray[np.float64]

    def __len__(self) -> int:
        """Return the number of runs."""
        return int(self.start_index.size)

    def __eq__(self, other: object) -> bool:
        """Compare every array by value, treating NaN entries as equal."""
        if not isinstance(other, RunSet):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, field.name), getattr(other, field.name), equal_nan=True)
            for field in fields(self)
        )


def _require_numeric(dtype: np.dtype[Any]) -> None:
    """Reject dtypes that would otherwise be coerced into meaningless numbers."""
    if dtype.kind not in "biuf":
        raise InputTypeError(
            "Index values must be numeric: datetime, string, object, and complex arrays are not coerced to float64.",
            expected_type=float,
            actual_type=dtype.type,
        )


def _as_float_series(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce a series to float64, turning masked elements into NaN."""
    _require_numeric(np.asarray(values).dtype)
    return np.asarray(np.ma.asarray(values, dtype=np.float64).filled(np.nan), dtype=np.float64)


def _invalid_threshold(threshold: object) -> InvalidArgumentError:
    """Build the error for a threshold that is not a finite number."""
    return InvalidArgumentError(
        "threshold must be a finite number.",
        argument_name="threshold",
        argument_value=repr(threshold),
        valid_values="a finite int, float, or 0-d numeric array",
    )


def _as_finite_threshold(threshold: object) -> float:
    """Coerce a scalar numeric threshold to float64, rejecting anything else.

    The type check is deliberate: a bare ``float()`` call would silently parse
    numeric strings and round a high-precision ``Decimal``, and comparing a
    float64 series against a threshold that carries more precision than float64
    is a footgun rather than a feature.
    """
    if isinstance(threshold, np.ndarray):
        if threshold.ndim != 0 or threshold.dtype.kind not in "biuf":
            raise _invalid_threshold(threshold)
        raw: object = threshold.item()
    elif isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise _invalid_threshold(threshold)
    else:
        raw = threshold
    try:
        value = float(raw)  # type: ignore[arg-type]  # raw is a scalar int, float, or NumPy scalar here
    except (TypeError, ValueError, OverflowError):
        raise _invalid_threshold(threshold) from None
    if not math.isfinite(value):
        raise _invalid_threshold(threshold)
    return value


def _resolve_options(threshold: float, direction: str, min_duration: int) -> tuple[float, int]:
    """Validate the run-selection options and return the threshold and minimum duration."""
    if direction not in _VALID_DIRECTIONS:
        raise InvalidArgumentError(
            "direction must be 'below' or 'above'.",
            argument_name="direction",
            argument_value=direction,
            valid_values="'below' or 'above'",
        )
    threshold_value = _as_finite_threshold(threshold)
    if isinstance(min_duration, bool) or not isinstance(min_duration, (int, np.integer)) or min_duration < 1:
        raise InvalidArgumentError(
            "min_duration must be a positive integer.",
            argument_name="min_duration",
            argument_value=str(min_duration),
            valid_values="integer >= 1",
        )
    return threshold_value, int(min_duration)


def identify_runs(
    values: npt.ArrayLike,
    threshold: float = -1.0,
    direction: Direction = "below",
    min_duration: int = 1,
) -> RunSet:
    """Identify threshold runs in a one-dimensional index time series.

    Args:
        values: One-dimensional time series of index values (for example SPI or
            SPEI), in time order. Masked elements count as missing, like NaN.
            Non-numeric arrays are rejected.
        threshold: Run threshold. Defaults to -1.0, the conventional
            moderate-drought SPI threshold.
        direction: ``"below"`` to find runs of values below ``threshold``, or
            ``"above"`` for runs above it. Never inferred from the threshold's
            sign.
        min_duration: Minimum run length in time steps to report. Runs shorter
            than this are discarded after identification.

    Returns:
        The runs found, in time order, with their duration, magnitude,
        intensity, peak, and interarrival metrics. An input without runs yields
        a :class:`RunSet` of length zero.

    Raises:
        InvalidArgumentError: If ``direction`` is not ``"below"`` or
            ``"above"``, ``threshold`` is not a finite number (numeric
            strings and ``Decimal`` values are rejected, not parsed or
            rounded), or ``min_duration`` is not a positive integer.
        DataShapeError: If ``values`` is not one-dimensional. For gridded data
            use :func:`identify_runs_xarray`.
        InputTypeError: If ``values`` is not numeric (datetime, string, object,
            or complex).

    Examples:
        >>> import numpy as np
        >>> from climate_indices.runs import identify_runs
        >>> spi = np.array([0.5, -1.2, -1.5, -0.8, 0.3, -1.1, -1.3])
        >>> found = identify_runs(spi, threshold=-1.0)
        >>> len(found)
        2
        >>> found.start_index.tolist(), found.duration.tolist()
        ([1, 5], [2, 2])
    """
    threshold_value, min_duration_value = _resolve_options(threshold, direction, min_duration)

    series = _as_float_series(values)
    if series.ndim != 1:
        raise DataShapeError(
            "values must be a one-dimensional time series; use identify_runs_xarray for gridded data.",
            expected_shape="(time,)",
            actual_shape=series.shape,
        )

    return _identify_runs(series, threshold_value, direction, min_duration_value)


def _identify_runs(
    series: npt.NDArray[np.float64],
    threshold: float,
    direction: Direction,
    min_duration: int,
) -> RunSet:
    """Find runs in a validated one-dimensional float64 series (the apply_ufunc kernel)."""
    series = np.asarray(series, dtype=np.float64)
    condition = series < threshold if direction == "below" else series > threshold
    condition &= ~np.isnan(series)

    # pad so runs touching either end of the series produce a start and an end
    transitions = np.diff(np.concatenate(([False], condition, [False])).astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1

    durations = ends - starts + 1
    long_enough = durations >= min_duration
    starts, ends, durations = starts[long_enough], ends[long_enough], durations[long_enough]

    count = starts.size
    magnitude = np.empty(count, dtype=np.float64)
    intensity = np.empty(count, dtype=np.float64)
    peak_value = np.empty(count, dtype=np.float64)
    peak_index = np.empty(count, dtype=np.int64)

    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        run_values = series[start : end + 1]
        if direction == "below":
            deviations = threshold - run_values
            peak_offset = int(np.argmin(run_values))
        else:
            deviations = run_values - threshold
            peak_offset = int(np.argmax(run_values))
        total = float(deviations.sum())
        magnitude[position] = total
        intensity[position] = total / (end - start + 1)
        peak_index[position] = start + peak_offset
        peak_value[position] = run_values[peak_offset]

    interarrival = np.full(count, np.nan, dtype=np.float64)
    if count > 1:
        interarrival[:-1] = np.diff(starts)

    return RunSet(
        start_index=starts.astype(np.int64),
        end_index=ends.astype(np.int64),
        duration=durations.astype(np.int64),
        magnitude=magnitude,
        intensity=intensity,
        peak_value=peak_value,
        peak_index=peak_index,
        interarrival=interarrival,
    )


def identify_runs_xarray(
    data: xr.DataArray,
    threshold: float = -1.0,
    direction: Direction = "below",
    min_duration: int = 1,
    *,
    time_dim: str = "time",
) -> RunSet | xr.DataArray:
    """Identify threshold runs along the time dimension of a DataArray.

    Applies :func:`identify_runs` to every cell of the input, so a
    ``(time, lat, lon)`` cube yields one :class:`RunSet` per grid cell.

    Args:
        data: Index values with a time dimension.
        threshold: Run threshold, as in :func:`identify_runs`.
        direction: Run direction, as in :func:`identify_runs`.
        min_duration: Minimum run length in time steps, as in
            :func:`identify_runs`.
        time_dim: Name of the time dimension to run along.

    Returns:
        A :class:`RunSet` when ``data`` is one-dimensional (evaluated eagerly).
        Otherwise, an object-dtype ``xr.DataArray`` with the time dimension
        removed, holding one :class:`RunSet` per cell, with the input's
        coordinates and the options recorded in ``attrs``. Dask-backed input
        stays lazy, with the time dimension rechunked whole (whole-series runs
        cannot be computed chunk by chunk) and the cell chunks preserved.
        Per-cell results are ragged by nature, so downstream aggregation is
        expected to reduce them to fixed-shape statistics.

    Raises:
        DimensionMismatchError: If ``data`` has no ``time_dim`` dimension.
        InputTypeError: If ``data`` is not numeric (datetime, string, object,
            or complex).
        InvalidArgumentError: If ``direction``, ``threshold``, or
            ``min_duration`` is invalid.
    """
    if time_dim not in data.dims:
        raise DimensionMismatchError(
            message=(
                f"Dimension '{time_dim}' not found in data. "
                f"Available dimensions: {list(data.dims)}. Use time_dim to specify a custom name."
            ),
            expected_dims=time_dim,
            actual_dims=tuple(data.dims),
            coordinate_name=time_dim,
            reason="missing_dimension",
        )
    threshold_value, min_duration_value = _resolve_options(threshold, direction, min_duration)
    _require_numeric(data.dtype)

    if data.ndim == 1:
        return identify_runs(
            data.values, threshold=threshold_value, direction=direction, min_duration=min_duration_value
        )

    if data.chunks is not None:
        # whole-series runs need the time dimension in one chunk, and pre-chunking it
        # here also stops apply_ufunc from rebalancing the cell chunks
        data = data.chunk({time_dim: -1})

    found = xr.apply_ufunc(
        _identify_runs,
        data,
        input_core_dims=[[time_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[object],
        kwargs={"threshold": threshold_value, "direction": direction, "min_duration": min_duration_value},
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    assert isinstance(found, xr.DataArray)  # single output, so apply_ufunc cannot return a tuple
    found.name = "runs"
    found.attrs = {
        "threshold": threshold_value,
        "direction": direction,
        "min_duration": min_duration_value,
        "long_name": "Run theory events per cell (RunSet objects)",
    }
    return found

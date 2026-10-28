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
- ``max_gap`` pools runs across at most that many non-missing time steps.
  Pooled duration includes the gap; magnitude excludes it. Missing values
  always separate runs, even when pooling is enabled.
- ``min_duration`` filters runs *after* pooling, and ``interarrival`` is
  then computed between the surviving runs only.
- Masked array elements count as missing, like NaN; non-numeric arrays
  (datetime, string, object, or complex) are rejected with
  :class:`~climate_indices.exceptions.InputTypeError`.

``summarize_runs`` reports events and period summaries, assigning each complete
event to its onset period. Spatial aggregation and DataFrame formatting stay
downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices import _native_backend
from climate_indices.exceptions import (
    DataShapeError,
    DimensionMismatchError,
    InputTypeError,
    InvalidArgumentError,
)

# this family's switch on the optional Rust kernels (climate_indices._native_backend);
# None runs the scan and per-run loop below on their pure-Python implementation
_native = _native_backend.EXTENSION

__all__ = ["RunSet", "RunStatistics", "identify_runs", "identify_runs_xarray", "summarize_runs"]

Direction = Literal["below", "above"]

_VALID_DIRECTIONS: tuple[str, ...] = ("below", "above")


@dataclass(frozen=True, eq=False)
class RunSet:
    """Metrics for the runs found in a time series, one array entry per run.

    Runs appear in time order. All arrays have the same length, and an index
    without runs is represented by zero-length arrays rather than ``None``.

    Instances compare by value, treating NaN entries as equal, and are
    unhashable because their fields are arrays.

    :param start_index: index of each run's first time step
    :param end_index: index of each run's last time step (inclusive)
    :param duration: number of time steps from start to end, including pooled gaps
    :param magnitude: cumulative deviation from the threshold, excluding pooled
        gaps (the sum of
        ``threshold - value`` for ``direction="below"``, ``value - threshold``
        for ``direction="above"``), always positive
    :param intensity: mean deviation per time step, ``magnitude / duration``
    :param peak_value: most extreme value in each run: the minimum for
        ``direction="below"``, the maximum for ``direction="above"``
    :param peak_index: index of ``peak_value``; the first occurrence when tied
    :param interarrival: number of time steps from each run's start to the next
        run's start, computed after ``min_duration`` filtering; the last run
        has no successor and holds NaN
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


@dataclass(frozen=True, eq=False)
class RunStatistics:
    """Events and summaries grouped by the period of each event's onset.

    Summary arrays have one entry per sorted period label. An event that crosses
    a period boundary contributes its full duration and magnitude to its onset
    period, so total duration can exceed that period's length.

    :param events: all retained events, in time order
    :param period: sorted unique labels, or ``[0]`` for the whole input
    :param count: number of events starting in each period
    :param total_duration: sum of their elapsed durations, including pooled gaps
    :param max_magnitude: largest event magnitude, or zero when no event starts
    :param valid_count: number of non-NaN, unmasked input steps in each period
    :param frequency: event count divided by valid_count; NaN with no valid steps
    """

    events: RunSet
    period: npt.NDArray[Any]
    count: npt.NDArray[np.int64]
    total_duration: npt.NDArray[np.int64]
    max_magnitude: npt.NDArray[np.float64]
    valid_count: npt.NDArray[np.int64]
    frequency: npt.NDArray[np.float64]


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
    array = values if np.ma.isMaskedArray(values) else np.asarray(values)
    _require_numeric(np.asarray(array).dtype)
    return np.asarray(np.ma.asarray(array, dtype=np.float64).filled(np.nan), dtype=np.float64)


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
        if np.ma.is_masked(threshold) or threshold.ndim != 0 or threshold.dtype.kind not in "iuf":
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


def _resolve_options(threshold: float, direction: str, min_duration: int, max_gap: int) -> tuple[float, int, int]:
    """Validate and normalize the threshold, duration, and pooling options."""
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
    if isinstance(max_gap, bool) or not isinstance(max_gap, (int, np.integer)) or max_gap < 0:
        raise InvalidArgumentError(
            "max_gap must be a non-negative integer.",
            argument_name="max_gap",
            argument_value=str(max_gap),
            valid_values="integer >= 0",
        )
    return threshold_value, int(min_duration), int(max_gap)


def identify_runs(
    values: npt.ArrayLike,
    threshold: float = -1.0,
    direction: Direction = "below",
    min_duration: int = 1,
    *,
    max_gap: int = 0,
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
        min_duration: Minimum elapsed run length in time steps to report.
            Runs shorter than this are discarded after pooling.
        max_gap: Maximum number of non-missing intervening steps to pool.
            Zero disables pooling. NaN and masked values always separate runs.
            Duration includes pooled gaps, magnitude excludes them, and
            intensity divides magnitude by the full elapsed duration.

    Returns:
        The runs found, in time order, with their duration, magnitude,
        intensity, peak, and interarrival metrics. An input without runs yields
        a :class:`RunSet` of length zero.

    Note:
        On a probability-scale series (``output_scale="probability"``) thresholds
        are lower-tail probabilities, identified with ``direction="below"`` as for
        z-scores. The 10%, 5%, and 2.5% levels (0.1, 0.05, 0.025) correspond to
        z-scores of about -1.28, -1.64, and -1.96, which differ from the
        McKee/WMO class boundaries of -1.0, -1.5, and -2.0 (about 0.159, 0.067,
        and 0.023 on this scale).

    Raises:
        InvalidArgumentError: If ``direction`` is not ``"below"`` or
            ``"above"``, ``threshold`` is not a finite number (numeric
            strings and ``Decimal`` values are rejected, not parsed or
            rounded), ``min_duration`` is not a positive integer, or
            ``max_gap`` is not a non-negative integer.
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
    threshold_value, min_duration_value, max_gap_value = _resolve_options(threshold, direction, min_duration, max_gap)

    series = _as_float_series(values)
    if series.ndim != 1:
        raise DataShapeError(
            "values must be a one-dimensional time series; use identify_runs_xarray for gridded data.",
            expected_shape="(time,)",
            actual_shape=series.shape,
        )

    return _identify_runs(series, threshold_value, direction, min_duration_value, max_gap_value)


def _exceedance(values: npt.NDArray[np.float64], threshold: float, direction: Direction) -> npt.NDArray[np.bool_]:
    """Return the steps on the requested side of the threshold, excluding NaN.

    This is the run layer's only comparison, so a caller that shares the block
    scan below cannot drift from :func:`identify_runs`.
    """
    condition = values < threshold if direction == "below" else values > threshold
    return condition & ~np.isnan(values)


def _scan_runs(
    condition: npt.NDArray[np.bool_],
    missing: npt.NDArray[np.bool_],
    *,
    max_gap: int,
    min_duration: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Scan a ``(cells, time)`` exceedance mask for pooled, filtered runs.

    ``missing`` marks the steps that pooling must not cross. Returns each run's
    cell index, onset, and inclusive end, ordered by cell and then onset. This
    is the one run scan behind :func:`identify_runs` and
    :func:`~climate_indices.flood.flood_events`.
    """
    cells, days = condition.shape
    # pad so runs touching either end of a series produce a start and an end
    padded = np.zeros((cells, days + 2), dtype=np.int8)
    padded[:, 1:-1] = condition
    transitions = np.diff(padded, axis=1)
    cell_index, starts = np.nonzero(transitions == 1)
    _, ends = np.nonzero(transitions == -1)
    ends = ends - 1

    if max_gap > 0 and starts.size > 1:
        nan_counts = np.concatenate(
            (np.zeros((cells, 1), dtype=np.int64), np.cumsum(missing, axis=1, dtype=np.int64)), axis=1
        )
        separate = (
            (cell_index[1:] != cell_index[:-1])
            | (starts[1:] - ends[:-1] - 1 > min(max_gap, days))
            | (nan_counts[cell_index[1:], starts[1:]] != nan_counts[cell_index[:-1], ends[:-1] + 1])
        )
        starts = starts[np.concatenate(([True], separate))]
        ends = ends[np.concatenate((separate, [True]))]
        cell_index = cell_index[np.concatenate(([True], separate))]

    durations = ends - starts + 1
    keep = durations >= min_duration
    return cell_index[keep], starts[keep], ends[keep]


def _identify_runs(
    series: npt.NDArray[np.float64],
    threshold: float,
    direction: Direction,
    min_duration: int,
    max_gap: int,
) -> RunSet:
    """Find runs in a validated one-dimensional float64 series (the apply_ufunc kernel)."""
    series = np.asarray(series, dtype=np.float64)
    native = _native_backend.kernels(_native, "identify_runs")
    if native is not None and _native_backend.takes(series):
        # a duration or gap longer than the series is equivalent to the caller's value,
        # and the binding's integer parameters cannot hold an arbitrarily large Python int
        result = native.identify_runs(
            series[:, None],
            threshold,
            direction,
            min(min_duration, series.size + 1),
            min(max_gap, series.size),
        )
        return RunSet(*result[1:])

    condition = _exceedance(series, threshold, direction)
    _, starts, ends = _scan_runs(
        condition[None, :], np.isnan(series)[None, :], max_gap=max_gap, min_duration=min_duration
    )
    durations = ends - starts + 1

    count = starts.size
    magnitude = np.empty(count, dtype=np.float64)
    intensity = np.empty(count, dtype=np.float64)
    peak_value = np.empty(count, dtype=np.float64)
    peak_index = np.empty(count, dtype=np.int64)

    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        run_values = series[start : end + 1]
        active_values = run_values[condition[start : end + 1]] if max_gap else run_values
        if direction == "below":
            deviations = threshold - active_values
            peak_offset = int(np.argmin(run_values))
        else:
            deviations = active_values - threshold
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


def summarize_runs(
    values: npt.ArrayLike,
    *,
    periods: npt.ArrayLike | None = None,
    threshold: float = -1.0,
    direction: Direction = "below",
    min_duration: int = 1,
    max_gap: int = 0,
) -> RunStatistics:
    """Identify runs and summarize them by their onset period.

    Args:
        values: One-dimensional index series, as in :func:`identify_runs`.
            NaN and masked values separate events and do not count as valid
            input steps. Infinities retain the comparisons of ``identify_runs``.
        periods: One label per input step, such as a year or month. Labels must
            be real numbers, strings, datetime64, or timedelta64, without missing
            or infinite values. Missing means NaN or infinity in a float array,
            NaT in a datetime64 or timedelta64 array, or a masked element. A
            string array holds the text ``"nan"`` as an ordinary label, so
            mixed string and NaN input is not rejected. Repeated labels form
            one group. With ``None``,
            report one summary labelled zero, even for an empty input. Explicit
            empty labels produce empty summary arrays.
        threshold: Run threshold, as in :func:`identify_runs`.
        direction: Explicit run direction, as in :func:`identify_runs`.
        min_duration: Minimum elapsed duration after pooling.
        max_gap: Maximum non-missing gap to pool; zero disables pooling.

    Returns:
        Events and summaries in a :class:`RunStatistics`. Each complete event
        belongs to its onset period, even when it extends into later periods.
        Frequency is events per valid input step, not a time fraction, annual
        rate, or return period. With no events, count, total duration, and
        maximum magnitude are zero. Frequency is NaN with no valid input steps.
        Metrics use float64, as in ``identify_runs``.

    Raises:
        DataShapeError: If values is not one-dimensional, or periods does not
            have the same shape as values.
        InputTypeError: If values or period labels have an unsupported dtype.
        InvalidArgumentError: If an event option is invalid, or period labels
            contain masked, missing, or infinite values.
    """
    series = _as_float_series(values)
    events = identify_runs(series, threshold, direction, min_duration, max_gap=max_gap)
    if periods is None:
        period = np.array([0], dtype=np.int64)
        groups = np.zeros(series.size, dtype=np.int64)
    else:
        labels = np.asarray(periods)
        if labels.shape != series.shape:
            raise DataShapeError(
                "periods must have the same one-dimensional shape as values.",
                expected_shape=str(series.shape),
                actual_shape=labels.shape,
            )
        if labels.dtype.kind not in "biufUSmM":
            raise InputTypeError(
                "periods must contain real numeric, string, datetime64, or timedelta64 labels.",
                expected_type=np.ndarray,
                actual_type=labels.dtype.type,
            )
        if (
            np.ma.is_masked(periods)
            or (labels.dtype.kind == "f" and not np.isfinite(labels).all())
            or (labels.dtype.kind in "mM" and np.isnat(labels).any())
        ):
            raise InvalidArgumentError("periods must not contain missing or infinite labels.", argument_name="periods")
        period, groups = np.unique(labels, return_inverse=True)

    event_groups = groups[events.start_index]
    count = np.bincount(event_groups, minlength=period.size).astype(np.int64)
    valid_count = np.bincount(groups[~np.isnan(series)], minlength=period.size).astype(np.int64)
    total_duration = np.zeros(period.size, dtype=np.int64)
    max_magnitude = np.zeros(period.size, dtype=np.float64)
    np.add.at(total_duration, event_groups, events.duration)
    np.maximum.at(max_magnitude, event_groups, events.magnitude)
    frequency = np.full(period.size, np.nan, dtype=np.float64)
    np.divide(count, valid_count, out=frequency, where=valid_count > 0)
    return RunStatistics(events, period, count, total_duration, max_magnitude, valid_count, frequency)


def identify_runs_xarray(
    data: xr.DataArray,
    threshold: float = -1.0,
    direction: Direction = "below",
    min_duration: int = 1,
    *,
    max_gap: int = 0,
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
        max_gap: Maximum non-missing gap to pool, as in :func:`identify_runs`.
        time_dim: Name of the time dimension to run along.

    Returns:
        A :class:`RunSet` when ``data`` is one-dimensional (evaluated eagerly).
        Otherwise, an object-dtype ``xr.DataArray`` with the time dimension
        removed, holding one :class:`RunSet` per cell, with the input's
        coordinates and the options recorded in ``attrs``. Dask-backed input
        stays lazy, with the time dimension rechunked whole (whole-series runs
        cannot be computed chunk by chunk) and the cell chunks preserved, so a
        cell block holds its entire time series in memory at once — roughly the
        whole input for a cube chunked only along time. Per-cell results are
        ragged by nature, so downstream aggregation is expected to reduce them
        to fixed-shape statistics.

    Raises:
        DimensionMismatchError: If ``data`` has no ``time_dim`` dimension.
        InputTypeError: If ``data`` is not numeric (datetime, string, object,
            or complex).
        InvalidArgumentError: If ``direction``, ``threshold``, or
            ``min_duration``, or ``max_gap`` is invalid.
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
    threshold_value, min_duration_value, max_gap_value = _resolve_options(threshold, direction, min_duration, max_gap)
    _require_numeric(data.dtype)

    if data.ndim == 1:
        return identify_runs(
            data.values,
            threshold=threshold_value,
            direction=direction,
            min_duration=min_duration_value,
            max_gap=max_gap_value,
        )

    if data.chunks is not None:
        # whole-series runs need the time dimension in one chunk, and pre-chunking it
        # here also stops apply_ufunc from rebalancing the cell chunks; an empty time
        # axis must go eager because dask's gufunc divides by the core chunk size
        data = data.compute() if data.sizes[time_dim] == 0 else data.chunk({time_dim: -1})

    found = xr.apply_ufunc(
        _identify_runs,
        data,
        input_core_dims=[[time_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[object],
        kwargs={
            "threshold": threshold_value,
            "direction": direction,
            "min_duration": min_duration_value,
            "max_gap": max_gap_value,
        },
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    assert isinstance(found, xr.DataArray)  # single output, so apply_ufunc cannot return a tuple
    found.name = "runs"
    found.attrs = {
        "threshold": threshold_value,
        "direction": direction,
        "min_duration": min_duration_value,
        "max_gap": max_gap_value,
        "long_name": "Run theory events per cell (RunSet objects)",
    }
    return found

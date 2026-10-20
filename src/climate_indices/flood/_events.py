"""Flood-potential events: runs of a daily index above a threshold.

The Flood Index and the other flood indices describe flood potential, not
observed flooding. An event here is a maximal run of consecutive days on which
the index is finite and strictly above ``threshold``; it is a summary of
exceedance, not a claim that flooding occurred.

Event extraction and the per-period summaries share the run scan in
:mod:`climate_indices.runs`, so the comparison convention lives in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.exceptions import (
    CoordinateValidationError,
    InputTypeError,
    InsufficientDataError,
    InvalidArgumentError,
)
from climate_indices.runs import (
    RunSet,
    RunStatistics,
    _exceedance,
    _scan_runs,
    summarize_runs,
)
from climate_indices.validation import (
    validate_dask_chunks,
    validate_time_dimension,
    validate_time_monotonicity,
)
from climate_indices.xarray_adapter import build_output_attrs

# the gridded summary variables, in the order the apply_ufunc kernel returns them
_SUMMARY_FIELDS = ("count", "total_duration", "max_severity", "max_peak", "valid_count", "frequency")

_SUMMARY_METADATA: dict[str, tuple[str, str]] = {
    "count": ("Flood Event Count", "1"),
    "total_duration": ("Flood Event Total Duration", "1"),
    "max_severity": ("Flood Event Maximum Severity", "1"),
    "max_peak": ("Flood Event Maximum Peak", "1"),
    "valid_count": ("Valid Input Step Count", "1"),
    "frequency": ("Flood Event Frequency", "1"),
}


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


@dataclass(frozen=True, eq=False)
class FloodStatistics:
    """Flood-event counts and extrema grouped by the period of each event's onset.

    Summary arrays have one entry per sorted period label. An event that crosses
    a period boundary contributes its full duration, severity, and peak to its
    onset period, matching :func:`~climate_indices.runs.summarize_runs`. The
    events are the run layer's :class:`~climate_indices.runs.RunSet`, so
    ``magnitude`` is the sum of deviations above the threshold.

    A return period is deliberately not provided: the Deo et al. (2015)
    estimator is not recoverable, and the frequency below is an event rate, not
    a recurrence interval.

    :param events: all retained events, in time order
    :param period: sorted unique labels, or ``[0]`` for the whole input
    :param count: number of events starting in each period
    :param total_duration: sum of their durations, in input steps
    :param max_severity: largest event magnitude in each period, or zero when no
        event starts; with the default ``threshold=0`` this is the largest sum of
        raw index values, and with a non-zero threshold it is the largest sum of
        deviations above that threshold
    :param max_peak: largest event peak in each period, or zero when no event starts
    :param valid_count: number of non-NaN, unmasked input steps in each period
    :param frequency: event count divided by valid_count; NaN with no valid steps
    """

    events: RunSet
    period: npt.NDArray[Any]
    count: npt.NDArray[np.int64]
    total_duration: npt.NDArray[np.int64]
    max_severity: npt.NDArray[np.float64]
    max_peak: npt.NDArray[np.float64]
    valid_count: npt.NDArray[np.int64]
    frequency: npt.NDArray[np.float64]


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


def _validate_year_start_month(year_start_month: int) -> None:
    """Reject a year boundary that is not a calendar month, matching ``flood_index``."""
    if (
        isinstance(year_start_month, (bool, np.bool_))
        or not isinstance(year_start_month, (int, np.integer))
        or not 1 <= year_start_month <= 12
    ):
        raise InvalidArgumentError(
            "year_start_month must be an integer from 1 to 12.", argument_name="year_start_month"
        )


def flood_events(
    index: npt.ArrayLike,
    *,
    threshold: float = 0.0,
    min_duration: int = 1,
) -> FloodEvents:
    """Find the runs of a daily flood index that stay above a threshold.

    Args:
        index: Daily index such as :func:`flood_index` output, a 1-D series or
            a time-first Spatial Block ``(time, *cells)``. NaN and masked days
            are never part of an event and end any run.
        threshold: A day counts when ``index > threshold``, so a day exactly
            equal to the threshold is not an event day. The default of zero
            matches the Flood Index's above-average-annual-maximum reading.
        min_duration: Shortest event kept, in days.

    Returns:
        A :class:`FloodEvents` with onset, end, duration, peak, and severity for
        every event, ordered by cell and then onset. Severity is the sum of the
        raw index values on the event's days, not of their deviations from
        ``threshold``.

    Raises:
        InvalidArgumentError: If ``index`` is a scalar, ``threshold`` is not a
            finite number, or ``min_duration`` is not a positive integer.
        InputTypeError: If ``index`` is non-numeric.
    """
    _validate_options(threshold, min_duration)

    values = _validated_index(index)
    days = values.shape[0]
    cell_shape = values.shape[1:]
    # the cell count, not -1, so an empty time axis still has a shape
    block = values.reshape(days, int(np.prod(cell_shape, dtype=np.intp)))

    # time runs along the last axis so one flat array holds every cell's series
    series = np.ascontiguousarray(block.T)
    # an event day is finite as well as above the threshold, so +inf ends a run
    series = np.where(np.isfinite(series), series, np.nan)
    condition = _exceedance(series, threshold, "above")
    cell, onset, end = _scan_runs(condition, np.isnan(series), max_gap=0, min_duration=min_duration)

    flat = np.append(np.where(condition, series, 0.0).ravel(), 0.0)
    offsets = cell * days
    # reduceat reads [onset, end + 1) from the interleaved bounds; the trailing zero
    # keeps an end at the last cell's final day in bounds, and the odd slots are discarded
    if onset.size:
        bounds = np.empty(2 * onset.size, dtype=np.intp)
        bounds[0::2] = offsets + onset
        bounds[1::2] = offsets + end + 1
        severity = np.add.reduceat(flat, bounds)[0::2]
        peak = np.maximum.reduceat(flat, bounds)[0::2]
    else:
        severity = peak = np.empty(0, dtype=np.float64)

    cell_positions = (
        np.stack(np.unravel_index(cell, cell_shape), axis=1).astype(np.intp)
        if cell_shape
        else np.empty((onset.size, 0), dtype=np.intp)
    )
    return FloodEvents(
        cell=cell_positions,
        onset=onset.astype(np.intp),
        end=(end + 1).astype(np.intp),
        duration=(end - onset + 1).astype(np.intp),
        peak=np.asarray(peak, dtype=np.float64),
        severity=np.asarray(severity, dtype=np.float64),
    )


def _period_max_peak(statistics: RunStatistics, labels: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Return the largest event peak per onset period, zero where no event starts."""
    peak = np.full(statistics.period.size, -np.inf, dtype=np.float64)
    if statistics.events.start_index.size:
        groups = np.searchsorted(statistics.period, labels[statistics.events.start_index])
        np.maximum.at(peak, groups, statistics.events.peak_value)
    peak[np.isneginf(peak)] = 0.0
    return peak


def _flood_statistics(
    series: npt.NDArray[np.float64],
    periods: npt.ArrayLike | None,
    *,
    threshold: float,
    min_duration: int,
) -> FloodStatistics:
    """Summarize one validated one-dimensional series over optional period labels."""
    statistics = summarize_runs(
        series, periods=periods, threshold=threshold, direction="above", min_duration=min_duration
    )
    step_labels = np.zeros(series.size, dtype=np.int64) if periods is None else np.asarray(periods)
    return FloodStatistics(
        events=statistics.events,
        period=statistics.period,
        count=statistics.count,
        total_duration=statistics.total_duration,
        max_severity=statistics.max_magnitude,
        max_peak=_period_max_peak(statistics, step_labels),
        valid_count=statistics.valid_count,
        frequency=statistics.frequency,
    )


def _hydrological_year_labels(data: xr.DataArray, time_dim: str, year_start_month: int) -> npt.NDArray[np.int64]:
    """Label each step with the start year of its ``year_start_month`` annual period."""
    if time_dim not in data.coords:
        raise CoordinateValidationError(
            f"Hydrological-year labels need a {time_dim!r} coordinate with datetime values.",
            coordinate_name=time_dim,
            reason="missing_coordinate_labels",
        )
    time_coord = data[time_dim]
    values = time_coord.values
    if values.dtype.kind == "M" and np.isnat(values).any():
        raise CoordinateValidationError(
            f"The {time_dim!r} coordinate must not contain NaT.",
            coordinate_name=time_dim,
            reason="unsupported_datetime_type",
        )
    try:
        years = np.asarray(time_coord.dt.year.values)
        months = np.asarray(time_coord.dt.month.values)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CoordinateValidationError(
            f"The {time_dim!r} coordinate must hold datetime values to derive hydrological years.",
            coordinate_name=time_dim,
            reason="unsupported_datetime_type",
        ) from exc
    return np.asarray(np.where(months >= year_start_month, years, years - 1), dtype=np.int64)


def _summary_block(
    series: np.ndarray,
    period_values: np.ndarray,
    *,
    threshold: float,
    min_duration: int,
) -> tuple[Any, ...]:
    """Summarize one cell's series, returning the summary fields then its run set."""
    statistics = summarize_runs(
        series,
        periods=period_values,
        threshold=threshold,
        direction="above",
        min_duration=min_duration,
    )
    return (
        statistics.count,
        statistics.total_duration,
        statistics.max_magnitude,
        _period_max_peak(statistics, np.asarray(period_values)),
        statistics.valid_count,
        statistics.frequency,
        statistics.events,
    )


def _gridded_flood_summary(
    index: xr.DataArray,
    period_values: npt.NDArray[Any],
    *,
    threshold: float,
    min_duration: int,
    time_dim: str,
) -> xr.Dataset:
    """Report per-cell summaries over an onset-period axis plus each cell's events."""
    period_labels = np.unique(period_values)
    period_dim = "period"
    kernel_outputs = (*_SUMMARY_FIELDS, "events")
    result = xr.apply_ufunc(
        _summary_block,
        index,
        xr.DataArray(period_values, dims=time_dim),
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[[period_dim]] * len(_SUMMARY_FIELDS) + [[]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": {period_dim: period_labels.size}},
        output_dtypes=[np.int64, np.int64, np.float64, np.float64, np.int64, np.float64, object],
        kwargs={"threshold": threshold, "min_duration": min_duration},
    )
    arrays = result if isinstance(result, tuple) else (result,)
    calculation_metadata = {"threshold": threshold, "min_duration": min_duration}
    variables: dict[str, xr.DataArray] = {}
    for name, array in zip(kernel_outputs, arrays, strict=True):
        if name == "events":
            variables[name] = array.assign_attrs(calculation_metadata)
            continue
        long_name, units = _SUMMARY_METADATA[name]
        variables[name] = array.transpose(period_dim, ...).assign_coords({period_dim: period_labels})
        variables[name].attrs = build_output_attrs(
            index,
            cf_metadata={"long_name": long_name, "units": units},
            calculation_metadata=calculation_metadata,
            index_name=long_name,
        )
    return xr.Dataset(variables)


@overload
def flood_event_summary(
    index: xr.DataArray,
    *,
    year_start_month: int,
    threshold: float = 0.0,
    min_duration: int = 1,
    time_dim: str = "time",
) -> FloodStatistics | xr.Dataset: ...


@overload
def flood_event_summary(
    index: npt.ArrayLike,
    *,
    periods: npt.ArrayLike | None = None,
    threshold: float = 0.0,
    min_duration: int = 1,
) -> FloodStatistics: ...


def flood_event_summary(
    index: npt.ArrayLike | xr.DataArray,
    *,
    periods: npt.ArrayLike | None = None,
    threshold: float = 0.0,
    min_duration: int = 1,
    year_start_month: int | None = None,
    time_dim: str = "time",
) -> FloodStatistics | xr.Dataset:
    """Summarize flood-potential events by their onset period.

    Events are the runs of a daily index strictly above ``threshold``, as in
    :func:`flood_events`; an event's full duration, severity, and peak belong to
    the period it starts in. The summary describes flood potential, not
    observed flooding.

    The NumPy path accepts a 1-D series and explicit ``periods``. The DataArray
    path derives hydrological-year labels from the ``year_start_month`` annual
    boundary, matching :func:`flood_index`, and requires a datetime time
    coordinate. A one-dimensional DataArray returns a :class:`FloodStatistics`;
    a DataArray with cell dimensions returns an ``xr.Dataset`` with the six
    summary variables over a ``period`` dimension plus the input's cell
    dimensions, and one ``events`` variable holding each cell's
    :class:`~climate_indices.runs.RunSet`. The time dimension must be a single
    Dask chunk; an empty time axis is rejected on the gridded path.

    A return period ``T`` is not computed. The Deo et al. (2015) estimator is
    not recoverable, so ``frequency`` is the event rate per valid input step,
    not a recurrence interval; ``docs/flood_applications.md`` shows the recipe.

    Args:
        index: One-dimensional index series, or a DataArray with a time
            dimension plus optional cell dimensions. NaN and masked days
            separate events and do not count toward frequency's denominator.
        periods: NumPy only: one label per input step, such as a year or month.
            Missing labels are rejected. With ``None``, report one whole-series
            summary. See :func:`~climate_indices.runs.summarize_runs`.
        threshold: Strict event threshold; equality does not count.
        min_duration: Minimum event duration in input steps.
        year_start_month: Required for DataArray input: the calendar month
            beginning each annual period (1–12), as in :func:`flood_index`.
        time_dim: Name of the time dimension for DataArray input.

    Returns:
        A :class:`FloodStatistics` for NumPy and one-dimensional DataArray
        input, or the ``xr.Dataset`` described above for gridded DataArray
        input. ``max_severity`` is the largest event magnitude (deviations above
        the threshold; raw sums when ``threshold=0``), and ``max_peak`` the
        largest event peak, both zero in a period with no event.

    Raises:
        CoordinateValidationError: If a DataArray has no time dimension or a
            non-datetime, NaT-bearing, or non-monotonic time coordinate, or a
            gridded Dask input has split time chunks.
        DataShapeError: If a NumPy index is not one-dimensional or ``periods``
            does not match its shape.
        InsufficientDataError: If a gridded DataArray has an empty time axis.
        InvalidArgumentError: If options, ``year_start_month``, or period labels
            are invalid, or a NumPy call passes ``year_start_month``.
        InputTypeError: If the index or period labels have an unsupported dtype.
    """
    _validate_options(threshold, min_duration)
    if not isinstance(index, xr.DataArray):
        if year_start_month is not None:
            raise InvalidArgumentError(
                "year_start_month applies only to DataArray input; pass periods for a NumPy series.",
                argument_name="year_start_month",
            )
        return _flood_statistics(_validated_index(index), periods, threshold=threshold, min_duration=min_duration)

    if periods is not None:
        raise InvalidArgumentError(
            "periods applies only to NumPy input; DataArray periods come from year_start_month.",
            argument_name="periods",
        )
    if year_start_month is None:
        raise InvalidArgumentError(
            "year_start_month is required for DataArray input.", argument_name="year_start_month"
        )
    _validate_year_start_month(year_start_month)
    validate_time_dimension(index, time_dim)
    labels = _hydrological_year_labels(index, time_dim, year_start_month)
    validate_time_monotonicity(index[time_dim])
    if index.ndim == 1:
        return _flood_statistics(_validated_index(index.values), labels, threshold=threshold, min_duration=min_duration)
    if index.chunks is not None:
        validate_dask_chunks(index, time_dim)
    if index.sizes[time_dim] == 0:
        raise InsufficientDataError("An empty time axis has no steps to run over.")
    return _gridded_flood_summary(index, labels, threshold=threshold, min_duration=min_duration, time_dim=time_dim)

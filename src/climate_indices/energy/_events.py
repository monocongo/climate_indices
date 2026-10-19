"""Run-theory summaries of climate-driven energy stress, not recorded shortfalls."""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from climate_indices.exceptions import CoordinateValidationError, InsufficientDataError, InvalidArgumentError
from climate_indices.runs import Direction, RunStatistics, _resolve_options, summarize_runs
from climate_indices.validation import validate_dask_chunks, validate_time_dimension, validate_time_monotonicity
from climate_indices.xarray_adapter import build_output_attrs

# the gridded summary variables, in the order the apply_ufunc kernel returns them
_SUMMARY_FIELDS = ("count", "total_duration", "max_magnitude", "valid_count", "frequency")

_SUMMARY_METADATA: dict[str, tuple[str, str]] = {
    "count": ("Energy Drought Event Count", "1"),
    "total_duration": ("Energy Drought Event Total Duration", "1"),
    "max_magnitude": ("Energy Drought Event Maximum Magnitude", "1"),
    "valid_count": ("Valid Input Step Count", "1"),
    "frequency": ("Energy Drought Event Frequency", "1"),
}


def _events_block(
    series: np.ndarray,
    period_values: np.ndarray,
    *,
    threshold: float,
    direction: Direction,
    min_duration: int,
    max_gap: int,
) -> tuple[Any, ...]:
    """Summarize one cell's series, returning the summary fields then its run set."""
    statistics = summarize_runs(
        series,
        periods=period_values,
        threshold=threshold,
        direction=direction,
        min_duration=min_duration,
        max_gap=max_gap,
    )
    return (
        statistics.count,
        statistics.total_duration,
        statistics.max_magnitude,
        statistics.valid_count,
        statistics.frequency,
        statistics.events,
    )


def _period_values(
    periods: Any,
    *,
    time_coord: xr.DataArray | None = None,
    time_dim: str = "time",
) -> Any:
    """Read period labels, rejecting a DataArray whose time coordinate contradicts the input.

    The labels are attached to steps by position, so a silently dropped or reordered
    time coordinate would attribute every event to the wrong period.
    """
    if not isinstance(periods, xr.DataArray):
        return periods
    if time_coord is not None:
        if (
            periods.dims != (time_dim,)
            or time_dim not in periods.coords
            or not np.array_equal(periods[time_dim].values, time_coord.values)
        ):
            raise CoordinateValidationError(
                f"periods must carry the input's {time_dim!r} coordinate labels, in the same order.",
                coordinate_name=time_dim,
                reason="mismatched_coordinates",
            )
    return periods.values


def _validated_labels(periods: Any, size: int) -> np.ndarray:
    """Return gridded period labels as an array, rejecting masked or misaligned ones."""
    if np.ma.is_masked(periods):
        raise InvalidArgumentError("periods must not contain missing labels.", argument_name="periods")
    labels = np.asarray(periods)
    if labels.shape != (size,):
        raise ValueError("periods must hold one label per input time step")
    return labels


def _gridded_events(
    index: xr.DataArray,
    period_values: np.ndarray | None,
    *,
    threshold: float,
    direction: Direction,
    min_duration: int,
    max_gap: int,
    time_dim: str,
) -> xr.Dataset:
    """Report per-cell summaries over an onset-period axis plus each cell's events."""
    period_labels = np.array([0], dtype=np.int64) if period_values is None else np.unique(period_values)
    labels = np.zeros(index.sizes[time_dim], dtype=np.int64) if period_values is None else period_values
    period_dim = "period"
    kernel_outputs = (*_SUMMARY_FIELDS, "events")
    result = xr.apply_ufunc(
        _events_block,
        index,
        xr.DataArray(labels, dims=time_dim),
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[[period_dim]] * len(_SUMMARY_FIELDS) + [[]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": {period_dim: period_labels.size}},
        output_dtypes=[np.int64, np.int64, np.float64, np.int64, np.float64, object],
        kwargs={
            "threshold": threshold,
            "direction": direction,
            "min_duration": min_duration,
            "max_gap": max_gap,
        },
    )
    arrays = result if isinstance(result, tuple) else (result,)
    calculation_metadata = {
        "threshold": threshold,
        "direction": direction,
        "min_duration": min_duration,
        "max_gap": max_gap,
    }
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
def energy_drought_events(  # pyright: ignore[reportOverlappingOverload]
    index: xr.DataArray,
    *,
    threshold: float = -1.0,
    direction: Direction = "below",
    min_duration: int = 1,
    max_gap: int = 0,
    periods: npt.ArrayLike | xr.DataArray | None = None,
    time_dim: str = "time",
) -> RunStatistics | xr.Dataset: ...


@overload
def energy_drought_events(
    index: npt.ArrayLike,
    *,
    threshold: float = -1.0,
    direction: Direction = "below",
    min_duration: int = 1,
    max_gap: int = 0,
    periods: npt.ArrayLike | None = None,
    time_dim: str = "time",
) -> RunStatistics: ...


def energy_drought_events(
    index: npt.ArrayLike | xr.DataArray,
    *,
    threshold: float = -1.0,
    direction: Direction = "below",
    min_duration: int = 1,
    max_gap: int = 0,
    periods: npt.ArrayLike | xr.DataArray | None = None,
    time_dim: str = "time",
) -> RunStatistics | xr.Dataset:
    """Report events and period statistics for a standardized index.

    Events describe the climate-driven component of energy supply and demand,
    not actual generation, load, or recorded supply shortfalls. This function
    uses the generic :func:`~climate_indices.runs.summarize_runs` contract and
    accepts SPI or SPEI as well as standardized energy indices.

    A one-dimensional DataArray returns the same :class:`~climate_indices.runs.RunStatistics`
    the NumPy path returns; it is evaluated eagerly, so a Dask-backed series is
    computed. A DataArray with cell dimensions returns an ``xr.Dataset``: the
    five summary variables over a ``period`` dimension plus the input's cell
    dimensions, and one ``events`` variable holding each cell's
    :class:`~climate_indices.runs.RunSet`. Run events are found per cell, so a
    Dask-backed input must keep the time dimension in a single chunk; cell
    dimensions chunk freely. An empty time axis is rejected on the gridded path
    rather than reported as an empty summary.

    Args:
        index: One-dimensional index series in time order, or a DataArray with a
            time dimension plus optional cell dimensions. NaN and masked values
            separate events and do not count toward frequency's denominator.
        threshold: Strict event threshold; equality does not count. The default
            -1.0 is for a normal-score index, not probability-scale output.
        direction: ``"below"`` for low production such as SREPI. For high demand
            or SRLI stress, explicitly select ``"above"`` and a suitable positive
            threshold. Direction is never inferred from the series or threshold.
        min_duration: Minimum elapsed duration after pooling, in input steps.
        max_gap: Maximum non-missing gap to pool; zero disables pooling. Pooled
            gaps count toward duration but not magnitude. Intensity is magnitude
            divided by elapsed duration.
        periods: Optional label per input step, such as a year or month, as a
            sequence or a DataArray along the time dimension. The complete event
            belongs to its onset period. With None, report one whole-series
            summary. See ``summarize_runs`` for label requirements.
        time_dim: Name of the time dimension for DataArray input (default:
            ``"time"``).

    Returns:
        Generic run metrics and period counts, total durations, maximum
        magnitudes, valid step counts, and frequencies. Frequency is events per
        valid input step, not an annual rate. With no events, count, duration,
        and maximum magnitude are zero; frequency is NaN without valid steps.
        For a DataArray with cell dimensions, the ``xr.Dataset`` described above
        instead, carrying the event options in each variable's attributes.

    Raises:
        CoordinateValidationError: If a DataArray has no declared time dimension,
            a non-monotonic time coordinate, or period labels without matching
            time coordinates. Also raised for a gridded Dask input with split
            time chunks.
        DataShapeError: If the index is not one-dimensional or periods does not
            match its shape.
        InsufficientDataError: If a gridded DataArray has an empty time axis.
        InvalidArgumentError: If a gridded call receives masked period labels.
        InputTypeError: If the index or period labels have an unsupported dtype.
        InvalidArgumentError: If options or period labels are invalid.
        ValueError: If a DataArray period label count does not match its time
            dimension.
    """
    if not isinstance(index, xr.DataArray):
        return summarize_runs(
            index,
            periods=periods,
            threshold=threshold,
            direction=direction,
            min_duration=min_duration,
            max_gap=max_gap,
        )

    threshold, min_duration, max_gap = _resolve_options(threshold, direction, min_duration, max_gap)
    validate_time_dimension(index, time_dim)
    validate_time_monotonicity(index[time_dim])
    labels = _period_values(periods, time_coord=index[time_dim], time_dim=time_dim)
    if index.ndim == 1:
        return summarize_runs(
            np.asarray(index.values),
            periods=labels,
            threshold=threshold,
            direction=direction,
            min_duration=min_duration,
            max_gap=max_gap,
        )

    if index.chunks is not None:
        validate_dask_chunks(index, time_dim)
    if index.sizes[time_dim] == 0:
        raise InsufficientDataError("An empty time axis has no steps to run over.")
    return _gridded_events(
        index,
        None if labels is None else _validated_labels(labels, index.sizes[time_dim]),
        threshold=threshold,
        direction=direction,
        min_duration=min_duration,
        max_gap=max_gap,
        time_dim=time_dim,
    )

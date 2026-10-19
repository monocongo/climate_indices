"""Run-theory summaries of climate-driven energy stress, not recorded shortfalls."""

from __future__ import annotations

import numpy.typing as npt

from climate_indices.runs import Direction, RunStatistics, summarize_runs


def energy_drought_events(
    index: npt.ArrayLike,
    *,
    threshold: float = -1.0,
    direction: Direction = "below",
    min_duration: int = 1,
    max_gap: int = 0,
    periods: npt.ArrayLike | None = None,
) -> RunStatistics:
    """Report events and period statistics for a standardized index.

    Events describe the climate-driven component of energy supply and demand,
    not actual generation, load, or recorded supply shortfalls. This function
    uses the generic :func:`~climate_indices.runs.summarize_runs` contract and
    accepts SPI or SPEI as well as standardized energy indices.

    Args:
        index: One-dimensional index series in time order. NaN and masked values
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
        periods: Optional label per input step, such as a year or month. The
            complete event belongs to its onset period. With None, report one
            whole-series summary. See ``summarize_runs`` for label requirements.

    Returns:
        Generic run metrics and period counts, total durations, maximum
        magnitudes, valid step counts, and frequencies. Frequency is events per
        valid input step, not an annual rate. With no events, count, duration,
        and maximum magnitude are zero; frequency is NaN without valid steps.

    Raises:
        DataShapeError: If the index is not one-dimensional or periods does not
            match its shape.
        InputTypeError: If the index or period labels have an unsupported dtype.
        InvalidArgumentError: If options or period labels are invalid.
    """
    return summarize_runs(
        index,
        periods=periods,
        threshold=threshold,
        direction=direction,
        min_duration=min_duration,
        max_gap=max_gap,
    )

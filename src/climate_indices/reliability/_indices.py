"""Reliability metric accumulators over normalized event records (ADR-0026)."""

from dataclasses import dataclass

from climate_indices.exceptions import InvalidArgumentError
from climate_indices.reliability._events import EventTable, _normalize_events


@dataclass(frozen=True)
class ReliabilityResult:
    """Frozen reliability metrics for one supplied event log and customer count.

    ``saifi`` is sustained customer interruptions per served customer for the
    period represented by the supplied event log and average customer count.
    Partial periods are computed exactly and are never annualized.
    ``customer_interruptions`` is the shared
    ``CI = sum(N_i)`` accumulator: the total customers interrupted by sustained
    events. ``outage_count`` is the number of sustained event rows that
    contributed.
    """

    saifi: float
    customer_interruptions: int
    outage_count: int


def reliability_indices(
    events: EventTable,
    total_customers: float | None = None,
    *,
    exclude_planned: bool = False,
    threshold_minutes: float = 5.0,
) -> ReliabilityResult:
    """Compute SAIFI over sustained interruptions and the shared CI accumulator.

    Args:
        events: A DataFrame, column mapping, or sequence of row mappings,
            validated and classified by the reliability event contract.
        total_customers: The positive average served-customer count over the
            reporting period. It may be fractional. The argument is required;
            an absent or invalid value raises ``InvalidArgumentError``.
        exclude_planned: When True, planned sustained interruptions do not
            contribute. The default False includes them.
        threshold_minutes: Positive sustained-interruption threshold, in
            minutes. Durations strictly above it are sustained; equality is
            momentary.

    Returns:
        A frozen ``ReliabilityResult``. An empty event log returns SAIFI ``0.0``
        with zero customer interruptions and zero outages.

    Raises:
        InvalidArgumentError: ``total_customers`` is absent or invalid,
            ``exclude_planned`` is not a Boolean, or the event record is
            invalid. The exception's ``argument_name`` identifies the field.

    Warns:
        ClimateIndicesWarning: An explicit type contradicts the duration.
        OverlappingEventsWarning: Positive-duration intervals overlap.
    """
    if not isinstance(exclude_planned, bool):
        raise InvalidArgumentError("exclude_planned: must be a Boolean", argument_name="exclude_planned")
    if total_customers is None:
        raise InvalidArgumentError(
            "total_customers: supply the positive average served-customer count",
            argument_name="total_customers",
        )
    frame = _normalize_events(events, total_customers=total_customers, threshold_minutes=threshold_minutes)
    mask = frame["interruption_type"] == "sustained"
    if exclude_planned:
        mask &= ~frame["planned"]
    customer_interruptions = sum(int(count) for count in frame.loc[mask, "customers_interrupted"])
    return ReliabilityResult(
        saifi=customer_interruptions / float(total_customers),
        customer_interruptions=customer_interruptions,
        outage_count=int(mask.sum()),
    )

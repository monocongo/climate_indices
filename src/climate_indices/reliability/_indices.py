"""Reliability metric accumulators over normalized event records (ADR-0026)."""

import math
from dataclasses import dataclass, fields

from climate_indices.exceptions import InvalidArgumentError
from climate_indices.reliability._events import EventTable, _normalize_events

_DURATION_FIELDS = frozenset({"saidi", "customer_minutes_interrupted"})
_UNSET_DURATION = object()


def _comparable(field_name: str, value: object, customer_interruptions: int) -> object:
    """Normalize an unset NaN duration to match the derived-metric semantics.

    With no customer interruptions, CAIDI is undefined whatever the duration, so
    an unset NaN duration compares as 0.0 and a legacy three-field result matches
    an equivalent zero-duration result. With positive interruptions the duration
    determines CAIDI, so an unset NaN duration maps to a marker that stays
    distinct from an explicit 0.0.
    """
    if field_name in _DURATION_FIELDS and isinstance(value, float) and math.isnan(value):
        return 0.0 if customer_interruptions == 0 else _UNSET_DURATION
    return value


@dataclass(frozen=True, eq=False)
class ReliabilityResult:
    """Frozen reliability metrics for one supplied event log and customer count.

    ``saifi`` is sustained customer interruptions per served customer for the
    period represented by the supplied event log and average customer count.
    ``saidi`` is sustained interruption minutes per served customer for the same
    period. Partial periods are computed exactly and are never annualized.
    ``customer_interruptions`` is the shared
    ``CI = sum(N_i)`` accumulator: the total customers interrupted by sustained
    events. ``customer_minutes_interrupted`` is the shared
    ``CMI = sum(r_i * N_i)`` accumulator: total customer-minutes from sustained
    events. ``outage_count`` is the number of sustained event rows that
    contributed.

    ``caidi`` is a read-only property: ``CMI / CI`` minutes per customer
    interruption, independent of the served-customer count. It equals
    ``saidi / saifi`` up to floating-point rounding. With no sustained customer
    interruptions, it returns NaN and ``undefined_metrics`` names ``"caidi"``.
    A sustained interruption with zero duration has CAIDI ``0.0``.
    Both properties derive from the accumulators without adding constructor fields.

    The original three constructor fields retain their order.
    If callers omit ``saidi`` or ``customer_minutes_interrupted``, each omitted
    field defaults to NaN because the original fields cannot determine duration.
    ``reliability_indices()`` always supplies both duration metrics.

    Equality and hashing treat an unset NaN duration as ``0.0`` only when
    ``customer_interruptions`` is zero, where CAIDI is undefined whatever the
    duration. With positive interruptions the NaN duration stays distinct from
    an explicit ``0.0``, so results with different CAIDI never compare equal.
    """

    saifi: float
    customer_interruptions: int
    outage_count: int
    saidi: float = math.nan
    customer_minutes_interrupted: float = math.nan

    @property
    def caidi(self) -> float:
        """Return ``CMI / CI`` minutes per interruption, or NaN when CI is zero."""
        if self.customer_interruptions == 0:
            return math.nan
        return self.customer_minutes_interrupted / self.customer_interruptions

    @property
    def undefined_metrics(self) -> tuple[str, ...]:
        """Return undefined derived metric names, currently only ``caidi``."""
        return ("caidi",) if math.isnan(self.caidi) else ()

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(
            _comparable(field.name, getattr(self, field.name), self.customer_interruptions)
            == _comparable(field.name, getattr(other, field.name), other.customer_interruptions)
            for field in fields(self)
        )

    def __hash__(self) -> int:
        return hash(
            tuple(
                _comparable(field.name, getattr(self, field.name), self.customer_interruptions)
                for field in fields(self)
            )
        )


def reliability_indices(
    events: EventTable,
    total_customers: float | None = None,
    *,
    exclude_planned: bool = False,
    threshold_minutes: float = 5.0,
) -> ReliabilityResult:
    """Compute SAIFI, SAIDI, and CAIDI from shared sustained-interruption accumulators.

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
        A frozen ``ReliabilityResult``. An empty event log returns SAIFI and
        SAIDI ``0.0`` with zero customer interruptions, zero customer-minutes,
        and zero outages. CAIDI is NaN, and ``undefined_metrics`` is
        ``("caidi",)``. Undefined CAIDI does not produce an additional warning.

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
    products = frame.loc[mask, "duration"] * frame.loc[mask, "customers_interrupted"]
    # math.fsum keeps the accumulator order-independent, so shuffled rows agree exactly.
    customer_minutes_interrupted = float(math.fsum(products.tolist()))
    return ReliabilityResult(
        saifi=customer_interruptions / float(total_customers),
        saidi=customer_minutes_interrupted / float(total_customers),
        customer_interruptions=customer_interruptions,
        customer_minutes_interrupted=customer_minutes_interrupted,
        outage_count=int(mask.sum()),
    )

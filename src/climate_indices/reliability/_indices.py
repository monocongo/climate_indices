"""Reliability metric accumulators over normalized event records (ADR-0026)."""

import math
from dataclasses import dataclass, fields

from climate_indices.exceptions import InvalidArgumentError
from climate_indices.reliability._events import EventTable, _normalize_events

_DURATION_FIELDS = frozenset({"saidi", "customer_minutes_interrupted"})
_UNSET_DURATION = object()


def _comparable(field_name: str, value: object, customer_interruptions: int) -> object:
    """Normalize an unset NaN duration to match the derived-metric semantics.

    CAIDI is ``customer_minutes_interrupted / customer_interruptions``. With no
    customer interruptions, CAIDI is undefined whatever the duration, so an unset
    NaN duration compares as 0.0 and a legacy three-field result matches an
    equivalent zero-duration result. With positive interruptions only an unset
    ``customer_minutes_interrupted`` changes CAIDI, so only that field maps to a
    marker distinct from an explicit 0.0. ``saidi`` does not determine CAIDI, so
    its unset NaN still compares as 0.0.
    """
    if field_name in _DURATION_FIELDS and isinstance(value, float) and math.isnan(value):
        if field_name == "customer_minutes_interrupted" and customer_interruptions != 0:
            return _UNSET_DURATION
        return 0.0
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

    ``maifi_e`` is the momentary interruption event frequency:
    ``sum(N_i) / N_T`` over momentary event rows that are not flagged
    ``momentary_event_excluded``, in momentary events per customer. ``maifi`` is
    the momentary interruption frequency: ``sum(device_operations_i * N_i) / N_T``
    over all momentary rows, in device operations per customer. It is ``None``
    when the input has no ``device_operations`` column or a momentary row leaves
    its operation count unknown, because an operation frequency cannot be
    recovered from an event count. Without a ``device_operations`` column,
    ``undefined_metrics`` names ``"maifi"``.

    ``caidi`` is a read-only property: ``CMI / CI`` minutes per customer
    interruption, independent of the served-customer count. It equals
    ``saidi / saifi`` up to floating-point rounding. With no sustained customer
    interruptions, it returns NaN and ``undefined_metrics`` names ``"caidi"``.
    A sustained interruption with zero duration has CAIDI ``0.0``.
    Both properties derive from the accumulators without adding constructor fields.

    The original three constructor fields retain their order, and ``maifi_e``
    and ``maifi`` follow the duration metrics so existing positional calls stay valid.
    If callers omit ``saidi`` or ``customer_minutes_interrupted``, each omitted
    field defaults to NaN because the original fields cannot determine duration.
    ``reliability_indices()`` always supplies both duration metrics.

    Equality and hashing normalize an unset NaN duration. A result with no
    customer interruptions compares as if both durations were ``0.0``, so a legacy
    three-field result matches an equivalent zero-duration result. With positive
    interruptions an unset ``customer_minutes_interrupted`` stays distinct from an
    explicit ``0.0``, because it changes CAIDI. An unset ``saidi`` still compares
    as ``0.0``, because ``saidi`` does not determine CAIDI. Results with different
    CAIDI therefore never compare equal.
    """

    saifi: float
    customer_interruptions: int
    outage_count: int
    saidi: float = math.nan
    customer_minutes_interrupted: float = math.nan
    maifi_e: float = 0.0
    maifi: float | None = None

    @property
    def caidi(self) -> float:
        """Return ``CMI / CI`` minutes per interruption, or NaN when CI is zero."""
        if self.customer_interruptions == 0:
            return math.nan
        return self.customer_minutes_interrupted / self.customer_interruptions

    @property
    def undefined_metrics(self) -> tuple[str, ...]:
        """Return the names of metrics the input cannot determine."""
        names = ["caidi"] if math.isnan(self.caidi) else []
        if self.maifi is None:
            names.append("maifi")
        return tuple(names)

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
    """Compute SAIFI, SAIDI, CAIDI, MAIFI_E, and MAIFI from the event records.

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
        A frozen ``ReliabilityResult``. An empty event log returns SAIFI, SAIDI,
        and ``maifi_e`` ``0.0`` with zero customer interruptions, zero
        customer-minutes, and zero outages. CAIDI is NaN and ``maifi`` is
        ``None`` without a ``device_operations`` column, so ``undefined_metrics``
        is ``("caidi", "maifi")``. A supplied ``device_operations`` column makes
        ``maifi`` defined, including ``0.0`` for a log with no momentary rows.
        Undefined metrics do not produce an additional warning.

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
    momentary = frame["interruption_type"] == "momentary"
    event_mask = momentary & ~frame["momentary_event_excluded"]
    event_customers = sum(int(count) for count in frame.loc[event_mask, "customers_interrupted"])
    maifi = None
    if "device_operations" in frame:
        operations = frame.loc[momentary, "device_operations"]
        if not operations.isna().any():
            contributions = operations * frame.loc[momentary, "customers_interrupted"]
            maifi = float(math.fsum(contributions.tolist())) / float(total_customers)
    return ReliabilityResult(
        saifi=customer_interruptions / float(total_customers),
        saidi=customer_minutes_interrupted / float(total_customers),
        customer_interruptions=customer_interruptions,
        customer_minutes_interrupted=customer_minutes_interrupted,
        outage_count=int(mask.sum()),
        maifi_e=event_customers / float(total_customers),
        maifi=maifi,
    )

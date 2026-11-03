"""Reliability metric accumulators over normalized event records (ADR-0026)."""

import math
from dataclasses import dataclass, fields

from climate_indices.exceptions import InvalidArgumentError
from climate_indices.reliability._events import EventTable, _normalize_events, _period_hours

_DURATION_FIELDS = frozenset({"saidi", "customer_minutes_interrupted"})
_UNSET_DURATION = object()


def _comparable(field_name: str, value: object, customer_interruptions: int, period_hours: float | None) -> object:
    """Normalize an unset NaN duration to match the derived-metric semantics.

    An unset CMI compares as zero only without customer interruptions, when
    CAIDI is undefined either way. An unset SAIDI compares as zero only without
    a reporting period, when ASAI is None either way. Otherwise, an unset
    duration stays distinct from zero because it changes a derived metric.
    """
    if field_name in _DURATION_FIELDS and isinstance(value, float) and math.isnan(value):
        if (field_name == "customer_minutes_interrupted" and customer_interruptions != 0) or (
            field_name == "saidi" and period_hours is not None
        ):
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
    contributed. A step-restored interruption has several rows, so each row
    counts separately. ``momentary_outage_count`` separately tallies momentary
    rows, including zero-duration and zero-customer rows. ``exclude_planned``
    filters sustained rows only, so it does not change this count. It does not
    group rows or apply ``momentary_event_excluded``; that flag affects MAIFI_E only.

    ``period_hours`` records the supplied or derived elapsed reporting hours.
    ``asai`` is a read-only availability fraction, ``1 - (saidi / 60) / period_hours``.
    Without a period, it is None and ``undefined_metrics`` names ``"asai"``.

    A non-positive or non-finite period makes ASAI NaN for a directly constructed
    result. If interruption time exceeds exposure, ASAI is NaN and remains named
    as undefined. A computed unavailable fraction up to one floating-point step
    above 1 instead gives ASAI ``0.0`` to allow for rounding. Larger excesses stay
    undefined. Overlapping events remain valid.

    Bounds do not filter or clip events; callers supply the reporting-period log.
    No Major Event Day filter is applied.

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
    Derived metrics do not add constructor fields. The new fields are
    ``period_hours`` and ``momentary_outage_count``.

    The original three constructor fields retain their order, and ``maifi_e``
    and ``maifi`` follow the duration metrics so existing positional calls stay valid.
    If callers omit ``saidi`` or ``customer_minutes_interrupted``, each omitted
    field defaults to NaN because the original fields cannot determine duration.
    ``reliability_indices()`` always supplies both duration metrics.

    Equality and hashing normalize an unset NaN duration. Without a reporting
    period, a result with no customer interruptions compares as if both durations
    were ``0.0``. A legacy three-field result thus matches an equivalent
    zero-duration result. With positive
    interruptions an unset ``customer_minutes_interrupted`` stays distinct from an
    explicit ``0.0``, because it changes CAIDI. An unset ``saidi`` still compares
    as ``0.0`` only without a reporting period. With a period, an unset ``saidi``
    stays distinct from zero because it changes ASAI. Results with different
    derived metrics therefore never compare equal.
    """

    saifi: float
    customer_interruptions: int
    outage_count: int
    saidi: float = math.nan
    customer_minutes_interrupted: float = math.nan
    maifi_e: float = 0.0
    maifi: float | None = None
    period_hours: float | None = None
    momentary_outage_count: int = 0

    @property
    def asai(self) -> float | None:
        """Return availability, None without hours, or NaN for invalid exposure."""
        if self.period_hours is None:
            return None
        if not math.isfinite(self.period_hours) or self.period_hours <= 0:
            return math.nan
        unavailable = (self.saidi / 60) / self.period_hours
        # Allow one floating-point step above full exposure for arithmetic rounding.
        if 1 < unavailable <= 1 + math.ulp(1.0):
            return 0.0
        return 1 - unavailable if unavailable <= 1 else math.nan

    @property
    def caidi(self) -> float:
        """Return ``CMI / CI`` minutes per interruption, or NaN when CI is zero."""
        if self.customer_interruptions == 0:
            return math.nan
        return self.customer_minutes_interrupted / self.customer_interruptions

    @property
    def undefined_metrics(self) -> tuple[str, ...]:
        """Return undefined derived metric names in CAIDI, MAIFI, ASAI order."""
        names: list[str] = ["caidi"] if math.isnan(self.caidi) else []
        if self.maifi is None:
            names.append("maifi")
        if self.asai is None or math.isnan(self.asai):
            names.append("asai")
        return tuple(names)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(
            _comparable(field.name, getattr(self, field.name), self.customer_interruptions, self.period_hours)
            == _comparable(field.name, getattr(other, field.name), other.customer_interruptions, other.period_hours)
            for field in fields(self)
        )

    def __hash__(self) -> int:
        return hash(
            tuple(
                _comparable(field.name, getattr(self, field.name), self.customer_interruptions, self.period_hours)
                for field in fields(self)
            )
        )


def reliability_indices(
    events: EventTable,
    total_customers: float | None = None,
    *,
    exclude_planned: bool = False,
    threshold_minutes: float = 5.0,
    period_hours: float | None = None,
    period_start: object = None,
    period_end: object = None,
) -> ReliabilityResult:
    """Compute SAIFI, SAIDI, CAIDI, MAIFI_E, MAIFI, and ASAI from the event records.

    Args:
        events: A DataFrame, column mapping, or sequence of row mappings,
            validated and classified by the reliability event contract.
        total_customers: The positive average served-customer count over the
            reporting period. It may be fractional. The argument is required;
            an absent or invalid value raises ``InvalidArgumentError``.
        exclude_planned: When True, planned sustained interruptions do not
            contribute. The momentary count and MAIFI are unaffected. The
            default includes them.
        threshold_minutes: Positive sustained-interruption threshold, in
            minutes. Durations strictly above it are sustained; equality is
            momentary.
        period_hours: Positive elapsed reporting hours for ASAI. No annual
            default is assumed. Without hours or bounds, ASAI is None.
        period_start: Explicit reporting start timestamp. Supply both bounds
            to derive elapsed hours, including leap days and timezone changes.
        period_end: Explicit reporting end timestamp, later than the start.
            If hours and bounds are supplied, they must agree within 1e-9 hours.
            Bounds do not filter or clip events; supply the appropriate log.

    Returns:
        A frozen ``ReliabilityResult``. An empty event log returns SAIFI, SAIDI,
        and ``maifi_e`` ``0.0`` with zero customer interruptions, zero
        customer-minutes, and zero outages. CAIDI is NaN and ``maifi`` is
        ``None`` without a ``device_operations`` column. ASAI is 1.0 with a
        reporting period, otherwise None. If sustained interruption minutes
        exceed the reporting exposure beyond the one-step rounding tolerance,
        ASAI is NaN, not clamped.
        ``undefined_metrics`` names every undefined derived metric in CAIDI,
        MAIFI, ASAI order, without an additional warning. The result records
        elapsed hours and never annualizes partial periods. No Major Event Day
        filter is applied.

    Raises:
        InvalidArgumentError: ``total_customers`` is absent or invalid,
            ``exclude_planned`` is not a Boolean, the reporting period is
            invalid, or the event record is invalid. The exception's
            ``argument_name`` identifies the field.

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
    hours = _period_hours(period_hours=period_hours, period_start=period_start, period_end=period_end)
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
            contributions = (
                int(count) * int(customers)
                for count, customers in zip(operations, frame.loc[momentary, "customers_interrupted"], strict=True)
            )
            maifi = float(math.fsum(contributions)) / float(total_customers)
    return ReliabilityResult(
        saifi=customer_interruptions / float(total_customers),
        saidi=customer_minutes_interrupted / float(total_customers),
        customer_interruptions=customer_interruptions,
        customer_minutes_interrupted=customer_minutes_interrupted,
        outage_count=int(mask.sum()),
        maifi_e=event_customers / float(total_customers),
        maifi=maifi,
        period_hours=hours,
        momentary_outage_count=int(momentary.sum()),
    )

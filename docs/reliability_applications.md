# Climate Indices for Power-System Reliability Applications

`climate_indices.reliability` turns an operational interruption record into the
IEEE Std 1366 distribution reliability indices: SAIFI, SAIDI, CAIDI, MAIFI_E,
MAIFI, and ASAI, with outage counts and the shared customer-interruption
accumulators. The input is an event log plus an average served-customer count,
not a grid or network model.

**These are distribution-reliability indices, not resilience metrics.** A
reliability index measures how often and how long service was interrupted under
past conditions. Resilience is the separate, forward-looking question of how a
system withstands and recovers from a hazard. The EPRI Climate READi
framework's forward-looking metrics — CORi, LORi, and DORi — require fragility
curves and hazard event probabilities that this library does not model. Use
these indices for reliability monitoring and climate-risk diagnostics of
observed interruptions, not as a resilience score or a forecast. This page is a
companion to {doc}`energy_applications`, {doc}`flood_applications`, and
{doc}`wildfire_applications`, which draw the same input-versus-impact
distinction for their families.

This page shows the worked example and how to read the result. For the
equations, units, and the conventions that vary between utilities, see
{doc}`algorithm_refs/reliability`. For the full field contract and every
edge case, see the {ref}`event contract <reliability-event-contract>` in the
API reference.

## The interruption record

Each row names one interruption event. The required fields are:

- `event_id` — a unique string or integer identifier.
- `start_time` — the interruption start timestamp.
- `customers_interrupted` — a nonnegative integer count.
- `duration` (minutes) or `end_time` — at least one complete time source.
  When both are supplied, they must agree within 1e-9 minutes.

Optional fields record context: `interruption_type` (`sustained` or
`momentary`), `planned`, `device_id`, `device_operations`,
`momentary_event_excluded`, `grouped_momentary_event`, `cause`, and
`component_ids`. `reliability_indices()` validates and classifies every row
before it sums.

`classify_interruptions()` applies the sustained/momentary threshold without
computing metrics. `group_momentary_events()` collapses a raw operation log
into one row per momentary event per device and window before the metrics are
computed.

## Worked example: from an event log to the result object

The committed fixture under `tests/fixture/reliability/` holds five synthetic
records: two sustained events, two momentary events, and one zero-duration
event. The served-customer count is 1000 and the reporting period is 24 hours.
The block below loads that log and computes the result object.

```{testcode} reliability-worked-example
from pathlib import Path
import json

import climate_indices
from climate_indices import reliability

repository = Path(climate_indices.__file__).resolve().parents[2]
events = json.loads((repository / "tests/fixture/reliability/events.json").read_text())
result = reliability.reliability_indices(events, total_customers=1000, period_hours=24)

assert result.saifi == 0.3
assert result.saidi == 24.0
assert result.caidi == 80.0
assert result.asai == 1 - (24.0 / 60) / 24
assert result.maifi_e == 0.3
assert result.maifi == 0.75
assert result.customer_interruptions == 300
assert result.customer_minutes_interrupted == 24000.0
assert result.outage_count == 2
assert result.momentary_outage_count == 3
assert result.period_hours == 24.0
assert result.undefined_metrics == ()
```

The two sustained records contribute 300 customer interruptions and 24,000
customer-minutes, so SAIFI is 0.3 interruptions per customer and SAIDI is 24.0
minutes per customer over the day. CAIDI is their ratio, 80.0 minutes per
customer interruption. ASAI is `1 - (24 / 60) / 24`, about 0.9833, the
available fraction of customer-hours. The two momentary rows with customers
contribute customer-weighted totals of 300 momentary events and 750 device
operations, so MAIFI_E is 0.3 events per customer and MAIFI is 0.75 device
operations per customer. The final zero-duration row is momentary and raises
`momentary_outage_count` to three; it contributes nothing to the sustained
metrics.

The same example runs end to end in the
[reliability demonstration notebook][reliability-demo], which the
documentation doctest build executes.

## Reading the result

`ReliabilityResult` is frozen and computes every metric once from the same
accumulators, so the metrics cannot disagree about the record they summed.

- `saifi` is customer interruptions per customer per reporting period.
- `saidi` is interruption minutes per customer per reporting period. Divide by
  60 for hours.
- `caidi` is minutes per customer interruption. It is `None`-free: it is NaN
  when there are no customer interruptions.
- `asai` is the dimensionless available fraction. It is `None` without a
  reporting period and NaN when interruption time exceeds the reporting
  exposure.
- `maifi_e` counts momentary events per customer; `maifi` counts device
  operations per customer and is `None` when operation data is absent.
- `customer_interruptions` and `customer_minutes_interrupted` are the shared
  `CI` and `CMI` accumulators.
- `outage_count` tallies sustained rows and `momentary_outage_count` tallies
  momentary rows. Both count rows, not unique incidents or customers.
- `undefined_metrics` names every metric the input cannot determine, in the
  order `caidi`, `maifi`, `asai`.

Frequencies are per the supplied reporting period, never per year. Each row
contributes its own customer count, so the record must not list the same
customers twice unless they were interrupted twice.

## Partial periods and exposure

ASAI needs a reporting period: supply `period_hours`, or both `period_start`
and `period_end` for leap-aware and daylight-saving-aware elapsed hours. The
library never assumes a default year and never infers the period from event
timestamps.

Partial periods are computed exactly over the supplied period, with no
annualization and no extrapolation. A result for a short period is therefore
**not comparable** with a full-year result, or with a partial-period result of
a different length. The result records `period_hours` so the exposure is
explicit. No Major Event Day filter is applied; filter the event log first if
one is required.

[reliability-demo]: https://github.com/monocongo/climate_indices/blob/main/notebooks/reliability_demo.ipynb

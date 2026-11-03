# Power-System Distribution Reliability Indices

`climate_indices.reliability` computes the IEEE Std 1366 distribution
reliability indices from an operational interruption record and an average
served-customer count. The record is a table of interruption events, not a
grid model: each row carries a start time, either a duration or an end time,
the number of customers interrupted, and optional flags. The result is a
frozen ``ReliabilityResult`` that holds every metric once; see the
{ref}`event contract <reliability-event-contract>` in the API reference.

These are **distribution-reliability indices**, not resilience metrics. A
reliability index measures how often and how long service was interrupted
under past conditions. Resilience is a separate, forward-looking question
about how a system withstands and recovers from a hazard. The EPRI Climate
READi framework's forward-looking metrics — CORi, LORi, and DORi — require
fragility curves and hazard event probabilities that this library does not
model, so they are out of scope. See
[ADR-0026](../adr/0026-reliability-module-api-and-conventions.md), decision 9.

The conformance target is IEEE Std 1366-2022. The definitions below match the
shipped implementation; the conventions that vary between utilities are stated
as explicit caller choices, not inferred. The domain vocabulary for this family
— SAIFI, SAIDI, CAIDI, MAIFI_E, MAIFI, ASAI, CI, CMI, sustained and momentary
interruptions, the reporting period, and the Major Event Day — is defined in
[src/climate_indices/CONTEXT.md](https://github.com/monocongo/climate_indices/blob/main/src/climate_indices/CONTEXT.md#reliability-family).

## Definitions and units

Let the interruption record contain sustained event rows indexed by *i*. For
row *i*:

- *N*{sub}`i` is the number of customers interrupted by the event.
- *r*{sub}`i` is the duration in minutes.
- *N*{sub}`T` is the average served-customer count over the reporting period.
  It is positive and may be fractional.
- *H* is the reporting period in hours.

The two shared accumulators are

$$
CI = \sum_i N_i
\qquad
CMI = \sum_i r_i N_i
$$

where *CI* is customer interruptions and *CMI* is customer-minutes
interrupted. Both sums run over sustained rows only.

| Metric | Equation | Numerator | Denominator | Units |
| --- | --- | --- | --- | --- |
| SAIFI | $CI / N_T$ | Customer interruptions (*CI*) | Served customers (*N*{sub}`T`) | Interruptions per customer per reporting period |
| SAIDI | $CMI / N_T$ | Customer-minutes interrupted (*CMI*) | Served customers (*N*{sub}`T`) | Minutes per customer per reporting period |
| CAIDI | $CMI / CI$ | Customer-minutes interrupted (*CMI*) | Customer interruptions (*CI*) | Minutes per customer interruption |
| MAIFI_E | $\sum_i IDE_i N_i / N_T$ | Momentary interruption events (*IDE*{sub}`i` ∈ {0, 1}) | Served customers (*N*{sub}`T`) | Momentary events per customer per reporting period |
| MAIFI | $\sum_i D_i N_i / N_T$ | Device operations (*D*{sub}`i`) | Served customers (*N*{sub}`T`) | Device operations per customer per reporting period |
| ASAI | $1 - \dfrac{SAIDI / 60}{H}$ | Available customer-hours | Total customer-hours | Dimensionless fraction in [0, 1] |

SAIFI, SAIDI, CAIDI, and the two MAIFI forms are frequencies or durations
**per the supplied reporting period**, never per year. Divide SAIDI by 60 to
obtain `SAIDI_hours`, the unit ASAI uses. CAIDI is the definitional ratio
*CMI* / *CI*, so it is exactly independent of *N*{sub}`T`; the identity
`CAIDI = SAIDI / SAIFI` holds mathematically through the shared accumulators,
but the separately rounded values need not match bit for bit.

ASAI is equivalent to the direct customer-hour form,

$$
ASAI = \frac{N_T \cdot H \cdot 60 - CMI}{N_T \cdot H \cdot 60}
$$

when interruption time does not exceed the reporting exposure. It is a
fraction, never a percentage.

Two row counts are reported alongside the ratios. `outage_count` is the number
of sustained event rows that contributed to *CI* and *CMI*. It counts rows,
not unique physical incidents or customers, so a step-restored interruption
with several restoration-stage rows counts each row separately.
`momentary_outage_count` separately tallies momentary rows, including
zero-duration and zero-customer rows; it does not group rows.

## Sustained and momentary interruptions

Every row is classified as *sustained* or *momentary* by its duration. A
duration strictly above `threshold_minutes` (default 5.0) is sustained; a
duration at or below it is momentary. Equality is momentary. The default
threshold is the IEEE and EIA convention; some regulators use one minute or
another interval, so the threshold is a caller parameter and nothing in the
library infers a jurisdiction.

The two classes feed disjoint metrics:

- Sustained rows contribute to *CI*, *CMI*, SAIFI, SAIDI, CAIDI, ASAI, and
  `outage_count`.
- Momentary rows contribute to `maifi_e` and `maifi` and to
  `momentary_outage_count`. They contribute nothing to SAIFI, SAIDI, or CAIDI.

An explicit `interruption_type` is retained. If it contradicts the duration,
the row keeps its explicit type and the library emits one
`ClimateIndicesWarning`. A row flagged `grouped_momentary_event` is exempt
from that warning, because its duration is an event span rather than an
operation duration.

IEEE MAIFI_E counts momentary interruption *events*. This library never
regroups a caller's rows, so `maifi_e` counts supplied rows as events.
`group_momentary_events()` is the opt-in helper that turns a raw operation log
into one row per event per device and grouping window. MAIFI, the
device-operation form, is reported only when the record carries a
`device_operations` count and no momentary row leaves it unknown; from an
event count alone, a device-operation frequency cannot be recovered.

IEEE MAIFI_E excludes momentary events associated with a sustained
interruption. That association cannot be detected from timing alone, so the
caller sets the optional `momentary_event_excluded` flag. The exclusion moves
an event out of `maifi_e` only; `maifi` still includes its operations, which
is a library convention rather than a verified IEEE requirement.

## Repeat customers and overlapping events

The accumulators are row-additive: each sustained row contributes its own
*N*{sub}`i` to *CI* and its own *r*{sub}`i`*N*{sub}`i` to *CMI*. The library
does not deduplicate customers across rows or events, and it does not merge
step-restoration rows. A record that lists the same customers on two rows
therefore contributes those customers twice. This is the correct behavior for
a record whose rows each name the customers they interrupted, and the caller
must prepare the record accordingly.

Overlapping events are valid input. A step-restored interruption, in which the
same customers are interrupted again while an earlier interruption is still
open, keeps every row. Overlap contributes to the sustained accumulators
without change. Because valid overlaps can accumulate more interruption time
than the reporting exposure, the library does not clamp SAIDI or ASAI to a
physical bound; see *Undefined metrics* below.

## Reporting period and exposure

ASAI needs a reporting period *H*. Supply either `period_hours` or both
`period_start` and `period_end`; the library derives actual elapsed hours from
the bounds. If both forms are supplied, they must agree within 1e-9 hours.
Without either, ASAI is `None` and `undefined_metrics` names `"asai"`. The
library never assumes 8760 hours and never infers the reporting period from
event timestamps.

Elapsed hours are actual hours. A complete 2023 period has 8760 hours and a
complete 2024 period has 8784 hours, so a leap day is counted. Timezone-aware
bounds use actual elapsed clock time, including daylight-saving changes.

Period bounds do not filter or clip events. The caller supplies the event log
for the reporting period. Partial periods are computed exactly over the
supplied period, **with no annualization and no extrapolation**. A
partial-period result is therefore not comparable with a full-year result, nor
with a partial-period result of a different length. The result records the
elapsed `period_hours` so the exposure is explicit.

No Major Event Day filter is applied. The IEEE 1366 recipe,
*T*{sub}`MED` = exp(*α* + 2.5*β*) over a utility-specific five-year daily
SAIDI history, is documented but not implemented; the library holds no such
history. Results with and without major events are not comparable.

## Conventions that vary between utilities

The IEEE definitions are stable, but several reporting choices differ between
utilities and jurisdictions. This library fixes a documented default for each
and exposes it as a parameter where a parameter is possible.

| Convention | Library behavior | How to change it |
| --- | --- | --- |
| Sustained/momentary threshold | 5.0 minutes | `threshold_minutes` on `reliability_indices()`, `classify_interruptions()`, and `group_momentary_events()` |
| Planned outages | Included by default | `exclude_planned=True` removes planned sustained rows; it does not change the momentary metrics |
| MAIFI form | Report `maifi_e` (events) always; report `maifi` (device operations) when operation counts are supplied | Supply `device_operations` to enable `maifi`; use `group_momentary_events()` to form events from a raw operation log |
| Momentary events tied to a sustained interruption | Not excluded unless flagged | Set `momentary_event_excluded=True` on those rows |
| Major Event Day filter | Not applied | Not implemented; filter the event log before calling |
| Partial-period annualization | Never annualized | None; the result always reflects the supplied period |
| Reporting exposure for ASAI | Must be supplied explicitly | `period_hours`, or `period_start` and `period_end` |

## Undefined metrics

Some inputs cannot determine a metric. `ReliabilityResult` returns NaN or
`None` for that metric and names it in `undefined_metrics`, in the order
`caidi`, `maifi`, then `asai`. Undefined metrics do not produce an additional
warning.

- **CAIDI** is undefined when *CI* is zero. An empty log, a momentary-only
  log, and sustained rows with zero interrupted customers all leave CAIDI NaN.
  An explicitly sustained zero-duration event with interrupted customers
  instead gives CAIDI 0.0.
- **MAIFI** is undefined when no `device_operations` column is supplied or a
  momentary row leaves its operation count unknown.
- **ASAI** is undefined without a reporting period, and it is NaN when
  interruption time exceeds the reporting exposure. A computed unavailable
  fraction up to one floating-point step above 1 instead gives ASAI 0.0, to
  allow for rounding. A larger excess is NaN and is never clamped to zero. The
  sustained accumulators and row counts remain unchanged.

## Validation

`tests/test_reliability_indices.py`, `tests/test_reliability_events.py`,
`tests/test_reliability_grouping.py`, and `tests/test_reliability_fixture.py`
check the shipped arithmetic and contracts: the ADR-0026 analytic cases, the
sustained/momentary split, the additive accumulators, threshold and
planned-outage filters, ASAI exposure and rounding, the reporting-period
elapsed hours, and the grouping rules. The committed fixture under
`tests/fixture/reliability/` records its provenance against
`tests/fixture/provenance_schema.json`.

These are **specification-and-regression checks, not external scientific
validation**. The fixture is hand-computed synthetic data, and no event-level
external oracle is available: EIA Form EIA-861 Table 11.1 is aggregated and
can validate relationships, not arithmetic. `VALIDATION.md` records the
reliability evidence tier accordingly.

## References

- IEEE Std 1366-2022, *IEEE Guide for Electric Power Distribution Reliability
  Indices*. Approved 21 September 2022, published 22 November 2022; supersedes
  IEEE Std 1366-2012. <https://standards.ieee.org/ieee/1366/7243/>
- U.S. Energy Information Administration, Form EIA-861, *Annual Electric Power
  Industry Report*, Table 11.1 (reliability).
- EPRI, *Climate READi power-system modelling guide*, Distribution Operations,
  Appendix A. <https://apps.epri.com/climate-readi-modeling-guide/en/dx-ops.html>
- [ADR-0026](../adr/0026-reliability-module-api-and-conventions.md) fixes the
  module API, units, and conventions summarized here.

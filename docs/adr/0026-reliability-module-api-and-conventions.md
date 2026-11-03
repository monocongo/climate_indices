# Reliability APIs are namespaced in the `reliability` package, with metric conventions fixed

## Status

Amended.

Amended by {doc}`0027-drought-namespace-and-module-migration` (#1555):
the drought NumPy module named in decision 1 is now `drought/indices.py`.
Shared `compute.py` and the reliability scope remain unchanged.

Amended by ENERGY2-3 ([#1337](https://github.com/monocongo/climate_indices/issues/1337)):
event normalization, `classify_interruptions()`, the event JSON Schema,
and internal served-customer and reporting-period validation are implemented.
Amended by ENERGY2-4 ([#1338](https://github.com/monocongo/climate_indices/issues/1338)):
`reliability_indices()` and the frozen `ReliabilityResult` compute SAIFI and the
shared `CI = sum(N_i)` accumulator from sustained events, with planned outages
included by default.
Amended by ENERGY2-5 ([#1339](https://github.com/monocongo/climate_indices/issues/1339)):
the same function and result compute SAIDI and the shared
`CMI = sum(r_i N_i)` accumulator over the same sustained rows, and the
`SAIDI_hours` conversion is documented for the later ASAI relationship.
Amended by ENERGY2-6 ([#1340](https://github.com/monocongo/climate_indices/issues/1340)):
`ReliabilityResult.caidi` computes `CMI / CI` from the shared accumulators.
It returns NaN when `CI` is zero and names `caidi` in `undefined_metrics` without an additional warning.
Both are read-only properties, so the result constructor remains unchanged.
CAIDI is the definitional `CMI / CI` ratio and is exactly independent of
`total_customers`; that ratio is tested exactly. Its identity with `SAIDI / SAIFI`
holds mathematically and is tested within four float64 epsilons, not bit-for-bit,
because SAIDI and SAIFI are rounded separately.
Amended by ENERGY2-7 ([#1341](https://github.com/monocongo/climate_indices/issues/1341)):
`reliability_indices()` and `ReliabilityResult` report `maifi_e`, the momentary
interruption event frequency, and `maifi`, the device-operation frequency, and
`group_momentary_events()` groups raw momentary operations into events.
Amended by the [#1341](https://github.com/monocongo/climate_indices/issues/1341)
review: a grouped row keeps the complete sequence interval, and the optional
`grouped_momentary_event` field marks an assembled row so its type survives a
later classification without a contradiction warning.
Amended by ENERGY2-8 ([#1342](https://github.com/monocongo/climate_indices/issues/1342)):
`reliability_indices()` accepts reporting hours or explicit period bounds.
`ReliabilityResult.period_hours` records elapsed hours, and its read-only `asai`
property computes `1 - (SAIDI / 60) / H` from the shared sustained calculation.
Without a period, ASAI is None and `undefined_metrics` names `asai`.
If interruption time exceeds exposure, ASAI is NaN and remains named as undefined.
A computed unavailable fraction up to one floating-point step above 1 instead gives ASAI `0.0` to allow for rounding.
Larger excesses remain undefined; overlapping event rows remain valid, without clamping or deduplication.
The existing `outage_count` tallies sustained rows, including separate restoration stages.
`momentary_outage_count` separately tallies momentary rows; it is independent of
`exclude_planned`, which filters sustained rows.
It does not group rows or apply the MAIFI_E-only `momentary_event_excluded` flag.
Amended by ENERGY2-9 ([#1343](https://github.com/monocongo/climate_indices/issues/1343)):
`simulate_reliability()` implements the seeded Monte Carlo path over supplied
components with series and parallel aggregation, an explicit required seed,
fixed, exponential, and lognormal restoration distributions, and rejection of
`component_topology` with `InvalidArgumentError`. Each trial's system-down
intervals form an event log that `reliability_indices()` consumes. Trials are
sampled in order from one generator, so a shorter run is an exact prefix of a
longer run with the same seed.
Amended by [#1527](https://github.com/monocongo/climate_indices/issues/1527):
`major_event_day_threshold()` fits the Beta Method Major Event Day threshold to
a caller-supplied daily-SAIDI history and returns a frozen
`MajorEventDayThreshold`. `reliability_indices()` removes Major Event Days only
when a threshold is passed as `major_event_threshold`, and the result records
the threshold and the excluded days. The rules are verified against IEEE Std
1366-2012, not the 2022 edition, because no lawful copy of the 2022 text was
obtained; decision 7 records the conventions, the evidence, and that limit.
Adapters remain unimplemented.
ENERGY2-10 through ENERGY2-11
([#1344](https://github.com/monocongo/climate_indices/issues/1344)–[#1345](https://github.com/monocongo/climate_indices/issues/1345))
implement the remaining design. Each decision is amended here as its implementation lands, as
{doc}`0013-flood-module-api-and-naming` was for flood.

The ticket named "ADR-0019", which was free when the epic was written. That
number is now the energy record
({doc}`0019-energy-module-api-and-aggregation`), and ADR-0025 was taken by the
KBDI fast path ({doc}`0025-kbdi-complete-input-fast-path`), so this is ADR-0026.
The module, names, units, and conventions below are unchanged by the
renumbering.

## Context

The parent epic ([#1335](https://github.com/monocongo/climate_indices/issues/1335))
adds power-system reliability indices: SAIFI, SAIDI, CAIDI, MAIFI, ASAI, and
outage counts, computed from an operational outage or interruption record plus a
served-customer count. These are the dependent variables of climate-risk
analysis for electric distribution systems, and the package has no way to
compute them today.

The indices are separated from the energy-meteorology family
([#1157](https://github.com/monocongo/climate_indices/issues/1157)) because they
share neither its inputs — reanalysis and CMIP fields — nor its computation:
standardized anomalies over accumulation windows. Reliability indices are
deterministic arithmetic over an event log. [#1157](https://github.com/monocongo/climate_indices/issues/1157)
excludes "grid, dispatch, market, or capacity-expansion modelling", and index
arithmetic over an outage log is not grid modelling. The one capability that
approaches that boundary — resolving which customers sit downstream of a failed
component — is deferred (decision 9).

The published material that would settle every convention is partly second-hand
or paywalled. IEEE Std 1366-2022 sells the guide that defines the indices; the
definitions and the Major Event Day recipe are quoted here from the IEEE
distribution-reliability working group's own public material and the guide's
public front matter, not from a vendored copy. The Major Event Day rules of
decision 7 come from the IEEE Std 1366-2012 text attached to a public utility
commission filing, because the 2022 text was not available. The EPRI Climate READi
power-system modelling guide is public but states that the industry has not
settled on a resilience metric. Where a convention cannot be pinned from an
accessible source, this record prefers a documented, citable choice over an
unreproducible claim of reproduction.

## Decision

1. **The family lives in `climate_indices.reliability`, a package with a
   public facade over underscore-prefixed implementation modules**, roughly
   `_events.py` (interruption classification and momentary-event grouping),
   `_indices.py` (the metric accumulators), and `_simulate.py` (the seeded Monte
   Carlo path). It follows {doc}`0005-fire-module-api` and
   {doc}`0013-flood-module-api-and-naming`: a family of indices that share a
   physical basis and evolve together gets its own namespaced package rather
   than being spread through the drought-oriented `drought/indices.py` and shared `compute.py`.
   It is imported as `from climate_indices import reliability`. Nothing is
   re-exported from the package root, and no unqualified name is added to
   `typed_public_api.py`. **It is not under `energy`**, because
   [#1157](https://github.com/monocongo/climate_indices/issues/1157) scoped that
   family to meteorological inputs and excludes grid and network modelling. The
   xarray adapter is a later ticket and is beta under
   {doc}`0012-xarray-api-stays-beta-through-3.0.0` until that policy changes; the
   adapter returns a `Dataset`, following the CFFWIS-to-`Dataset` precedent in
   {doc}`0005-fire-module-api`.

2. **The public names are `reliability_indices()`, `ReliabilityResult`,
   `classify_interruptions()`, `group_momentary_events()`,
   `simulate_reliability()`, `major_event_day_threshold()`, and
   `MajorEventDayThreshold`.** One metric entry point returns every metric in a
   frozen result object:

   - `reliability_indices()` takes the event record, `total_customers`, and
     optional reporting-period inputs and returns a `ReliabilityResult`.
     `total_customers` is the positive average served-customer count over the
     period. ASAI requires positive `period_hours` or explicit period start/end
     bounds from which the library derives actual elapsed hours. Without either,
     ASAI is undefined and the result names it; the library never infers the
     reporting period from event timestamps.
     The result records elapsed `period_hours` without annualizing partial periods.
     Bounds do not filter or clip events; callers supply the reporting-period log.
   - `ReliabilityResult` is a frozen dataclass carrying SAIFI, SAIDI, CAIDI,
     MAIFI_E (and MAIFI where the input admits it), ASAI, the outage count, and
     the shared accumulators `CI = Σ N_i` and `CMI = Σ r_i N_i`, so
     `CAIDI = SAIDI / SAIFI` holds mathematically through those shared accumulators.
     CAIDI uses `CMI / CI` directly to preserve independence from `total_customers`.
     The separately rounded floating-point values need not satisfy bit-for-bit equality.
     It also names any metric that is undefined for the input.
   - `classify_interruptions()` applies the sustained/momentary threshold of
     decision 3.
   - `group_momentary_events()` collapses reclosing operations of one device
     within a configurable grouping window (default five minutes), anchored at
     the first operation, into one momentary event. The grouping window is a
     separate setting from the classification threshold of decision 3.
   - `simulate_reliability()` takes supplied component failure rates and repair
     distributions, a series or parallel arrangement, `total_customers`,
     `horizon_hours`, `n_trials`, and a required explicit seed.
     It produces an event log for each trial and calls `reliability_indices()`
     with that trial's log and horizon. It returns one `ReliabilityResult` per
     trial as a `tuple[ReliabilityResult, ...]`, ordered by trial index.
     [#1343](https://github.com/monocongo/climate_indices/issues/1343) implements
     this entry point. It rejects `component_topology` with `InvalidArgumentError`
     because connectivity resolution is deferred.
   - `major_event_day_threshold()` fits the Major Event Day threshold of
     decision 7 to a daily-SAIDI history and returns a frozen
     `MajorEventDayThreshold`. Passing that object to `reliability_indices()` as
     `major_event_threshold` is the only way to filter Major Event Days.

   The event record requires `event_id`, `start_time`, `customers_interrupted`,
   and either an `end_time` timestamp or a `duration` in minutes.
   [#1337](https://github.com/monocongo/climate_indices/issues/1337) implements the
   full schema and validation rules, including the optional flags in decisions
   4 and 5. The {ref}`event contract <reliability-event-contract>`
   documents the shipped classifier and internal validation helpers.

   **Six `saifi()`-style scalar functions are rejected.** Each metric is a ratio
   over the same two accumulators, so six functions would recompute the same
   sums six times and could disagree about the record they summed. A single
   result object computes them once, preserves the mathematical `CAIDI = SAIDI / SAIFI`
   identity subject to floating-point rounding, and lets a degenerate input report
   all of its undefined metrics together.

3. **The sustained/momentary threshold is an explicit parameter, defaulting to 5
   minutes.** IEEE Std 1366-2022 defines a sustained interruption as any
   interruption that lasts more than five minutes, and a momentary interruption
   as the brief loss of power caused by the opening and closing operation of an
   interrupting device. The guide itself notes that some companies or
   jurisdictions use one minute or another interval, so the threshold is a
   caller-supplied parameter, not a constant. The alternative values named by
   the epic are 5 minutes (the IEEE and EIA default), 1 minute (some
   regulators), and other utility-specific intervals. A caller who does not set
   the threshold gets the IEEE default; nothing in the library infers a
   jurisdiction.

4. **MAIFI_E, the event form, is the default MAIFI flavour; the device-operation
   MAIFI is reported only when per-event `device_operations` is supplied.**
   IEEE Std 1366-2022 distinguishes the momentary average interruption frequency
   index, which counts momentary interruptions (each recloser operation), from
   the momentary average interruption event frequency index (MAIFI_E), which
   counts momentary interruption events. `ReliabilityResult` reports `maifi_e`
   by default. The result also reports `maifi` when the event record carries a
   per-event `device_operations` count, and only then: a device-operation
   frequency cannot be recovered from an event count.

   **The IEEE exclusion of momentary events immediately preceding a sustained
   interruption is caller-supplied, not inferred.** MAIFI_E excludes momentary
   events associated with a sustained interruption. The accessible sources for
   that rule are second-hand, so this epic does not attempt to detect the
   exclusion from timing alone. The event schema carries an optional
   `momentary_event_excluded` flag, and `group_momentary_events()` honours it.
   A caller who wants the IEEE exclusion sets the flag; a caller who does not
   gets the unflagged event count, and the result records that the exclusion was
   not applied.

   **`maifi_e` counts supplied rows as events, and `maifi` needs operation
   data.** `reliability_indices()` never regroups a caller's rows, so a raw
   operation log passes through `group_momentary_events()` first, whose output
   is one row per event. `maifi_e` excludes flagged rows; `maifi` uses each
   row's `device_operations` and is undefined without that column. The exclusion
   flag therefore removes an event from `maifi_e` only, as the 2003 subclause
   attaches it to the event form.

   **Grouping is lossy, so the helper rejects unsupported input.** Row totals
   cannot recover the distinct customers of an event, because two operations can
   interrupt the same set or different sets. The helper raises for a group whose
   operations interrupt different customer counts and accepts equal counts under
   a stated same-customer assumption. It also raises for a group that completes
   after its window or mixes exclusion flags, and it never merges different
   `device_id` values. Rows without the column or with null device identifiers
   group by time with other unknown-device rows. This library convention can
   merge distinct unknown devices.

   **A grouped row keeps the complete sequence interval.** Its `start_time` is
   the anchor operation's start and its `end_time` is the latest operation's end,
   so its elapsed `duration` includes the gaps between operations and can exceed
   the classification threshold. Its momentary classification comes from its
   member operations, not from that span, and the helper sets
   `grouped_momentary_event` so a later classification keeps the type without a
   contradiction warning. That field also records the loss: the row's duration is
   an event span, not an operation duration. A caller-supplied row without the
   flag still warns when its explicit type contradicts its duration.

5. **Planned outages are included by default, driven by an explicit `planned`
   flag; `exclude_planned` is the opt-in.** Most U.S. reporting includes planned
   outages, and some utilities and jurisdictions exclude them, so inclusion is
   the default and exclusion is an explicit keyword. The library never infers
   planned status from a cause string or a duration.

6. **Durations use minutes, frequencies apply per reporting period, and ASAI
   is dimensionless.** Durations `r_i` are minutes. SAIDI is minutes per customer
   per period; CAIDI is minutes per customer interruption.
   SAIFI, MAIFI_E, and MAIFI are interruptions or events per customer
   **per the supplied reporting period**,
   not per year. ASAI is a dimensionless fraction in [0, 1] and is never
   reported as a percentage. ASAI uses `1 - (SAIDI / 60) / H`, with minutes
   converted to hours. Without a reporting period, ASAI is None and named in
   `undefined_metrics`. Valid overlapping rows can accumulate more interruption
   time than the reporting exposure. An unavailable fraction up to one
   floating-point step above 1 gives ASAI `0.0` to allow for rounding; a larger
   excess gives ASAI NaN and names it undefined. The sustained accumulators and
   row count remain unchanged, and no larger result is clamped to zero.
   Undefined derived metrics produce no additional warning.
   **Partial periods are computed exactly over the supplied period, with no
   annualization and no extrapolation**, so a
   partial-period index is not comparable to an annual index and the result says
   so. Hours in the period `H` are actual hours: 8760 for a 365-day year and
   8784 for a 366-day year.

7. **Major Event Days are detected from a caller-supplied daily-SAIDI history
   and removed only on request.** The default result is unfiltered. The
   threshold uses the IEEE 1366 Beta Method, `T_MED = exp(α + 2.5β)`.
   [#1527](https://github.com/monocongo/climate_indices/issues/1527) implements
   it as a follow-up outside the original epic.

   **The rules are verified against IEEE Std 1366-2012, not 2022.** The
   verified text is clause 3.5 and its example (printed pages 10–13) of the IEEE
   Std 1366-2012 copy in a public CPUC filing. That filing labels its appendix
   2022, but the attached title and page headers say 2012. No lawful copy of the
   2022 text was obtained. On 2026-11-02 and 2026-11-03, the IEEE SA page, IEEE
   Xplore, and the ANSI store returned HTTP 403, a bot challenge, or no readable
   standard text. No access control was bypassed, and no copy was bought. The
   2022 conformance target of decision 10 is therefore unverified for Major
   Event Days, and the #1527 requirement to verify against 2022 is not met.
   Reverify these rules when a licensed 2022 copy is available.
   [`docs/research/major-event-day-conventions.md`](https://github.com/monocongo/climate_indices/blob/main/docs/research/major-event-day-conventions.md)
   holds the source record and checksums.

   The 2012 clause fixes these rules, and the library follows each one:

   - The history is the daily SAIDI for five sequential years, ending on the
     last day of the last complete reporting period. With fewer than five years,
     all available history is used.
   - Days without interruptions are excluded from the sample, because their
     logarithm is undefined.
   - `α` is the mean and `β` the standard deviation of the natural logarithms
     of the positive daily SAIDI values.
   - The threshold is calculated at the end of a reporting period for use in
     the next period.
   - A Major Event Day has daily SAIDI strictly greater than `T_MED`. Equality
     is not a Major Event Day, and the comparison has no tolerance.
   - An interruption that spans midnight accrues to the day it begins.
   - Daily SAIDI identifies the days, but all indices are calculated with those
     days removed. The clause asks for the activity on them to be analyzed and
     reported separately; the result records the excluded days so a caller can
     report them, and it does not compute a separate Major Event Day metric.

   **`β` uses the sample standard deviation, `n - 1` divisor, as a project
   convention.** The 2012 clause names no divisor. The method author's paper
   (Christie, equation 6) uses `n - 1`, and the clause's worked example prints
   `β = 1.90`, which matches the sample value 1.905 and not the population
   value 1.873. That example prints `T_MED = 66.69`, which is `exp(4.20)` from
   its rounded intermediates; full precision gives 67.10, so the example is not
   a full-precision threshold oracle.

   **`major_event_day_threshold(daily_saidi)` validates the history it is
   given.** It takes a Series or mapping from calendar day to daily SAIDI and
   returns a frozen `MajorEventDayThreshold` with `t_med`, `alpha`, `beta`,
   `history_start`, `history_end`, `positive_days`, and `timezone`, in that
   constructor order. The caller
   selects the interval; the library does not choose a window from a longer
   record. These validation rules are project choices, because the clause does
   not address them. Each raises `InvalidArgumentError` naming `daily_saidi`:

   | History condition | Behavior |
   | --- | --- |
   | Missing value (`None`, NaN, `pd.NA`, masked) or a missing calendar day | Rejected. An unknown day is never treated as a zero day or dropped. |
   | Duplicate day, non-midnight label, or a label that is not a date | Rejected, so each day has exactly one value. |
   | Negative, infinite, Boolean, or string value | Rejected before any logarithm. |
   | More than five years from first to last day | Rejected, because the clause uses five years. |
   | Fewer than five years | Accepted, as the clause allows. `history_start`, `history_end`, and `positive_days` record the reduced support. |
   | Empty, all-zero, or one positive day | Rejected, because the sample standard deviation needs two positive days. |
   | Constant positive values | Accepted; `β = 0` and `T_MED` equals that value up to rounding. |
   | Threshold that overflows float64 | Rejected; no infinite threshold is returned. |

   Two positive days is the arithmetic minimum, not a claim of adequate
   statistical support. The clause also assumes that the logarithms closely
   resemble a Gaussian distribution; the library does not test that
   assumption. A caller can construct a `MajorEventDayThreshold` directly, for
   example from a regulator-published `T_MED`. `reliability_indices()` then
   requires a positive, finite `t_med` and a calendar-date `history_end`.

   **Daily SAIDI uses the target log's own conventions.** With
   `major_event_threshold`, `reliability_indices()` sums `r_i N_i` by start
   day over the sustained rows that SAIFI and SAIDI count, so the
   sustained/momentary threshold and `exclude_planned` apply, and divides by
   `total_customers`. Momentary rows never make a Major Event Day. Units are
   minutes per customer per day, so the history must use the same units,
   customer-count basis, planned-outage policy, and sustained threshold. The
   library cannot check that agreement for a supplied history. Every row that
   starts on a Major Event Day is removed — sustained, momentary, planned, or
   flagged — and every metric, count, and accumulator uses the remaining rows
   through the same path as an unfiltered call.

   **The day boundary comes from the history.** A timezone-naive history labels
   naive calendar days and requires naive event timestamps. A history indexed by
   timezone-aware midnights declares its timezone, and aware event start times
   are converted to it before assignment to days. Mixed awareness raises
   `InvalidArgumentError` naming `start_time`. Each row is assigned by its own
   `start_time`; the library does not link restoration-stage rows. A staged
   restoration accrues to the original start day only if its rows carry that
   start time. Every row must also start after `history_end`, because a
   threshold applies to the period after its history.

   **The result records the filter.** `ReliabilityResult.major_event_threshold`
   holds the threshold that was applied, with its value and history interval,
   or None when no filter was applied. `major_event_days` lists the excluded
   days in ascending order, or is empty. The reporting period is not changed:
   filtered ASAI uses the full `period_hours`, and no metric is rescaled by the
   remaining day count. The 2012 example sums the remaining daily SAIDI values
   without a multiplier. The unchanged ASAI exposure is a project convention,
   because the clause does not address it. The August 2002 working-group draft
   rules that replace zero days and rescale filtered indices are not adopted.
   The library does not refit the threshold after removing large days and does
   not apply the clause 5.3 catastrophic-event treatment.

   **Filtered and unfiltered results are not comparable.** They describe
   different event sets. Two filtered results are comparable only if their
   reporting periods, customer populations, interruption definitions, history
   windows, units, and day boundaries also agree. Partial periods remain
   unannualized, as decision 6 requires.

8. **No new mandatory dependency.** The metric core is NumPy. `pandas` (already
   a hard dependency) handles record ingest, and `scipy.stats` (already a hard
   dependency) is available only to the Monte Carlo sampling path, which
   requires an explicit seed for reproducibility. `networkx` is added only
   behind an optional `climate-indices[reliability]` extra, and only if topology
   resolution is ever pursued (decision 9); `pandapower` and equivalent power-flow
   dependencies are not added. A capability that needs a new dependency belongs
   behind an extra or in the sibling repository the criteria in decision 9
   define.

9. **Scope boundary and sibling-repository criteria.** In scope is
   deterministic index arithmetic over an outage or interruption event log plus
   a served-customer count, sustained/momentary classification, momentary-event
   grouping, and a minimal seeded Monte Carlo path over supplied component data
   with series and parallel aggregation only. **Out of scope:** network topology
   resolution and customer connectivity ("which customers are downstream"),
   AC/DC power flow, protection coordination, recloser-sequence modelling,
   cascading-failure simulation, weather-to-component failure models and
   fragility curves, optimal restoration and crew dispatch, economic loss
   estimation, utility-specific regulatory report formats (EIA-861, RUS
   1730A-119, state PUC filings), real-time SCADA or OMS integration,
   annualization of partial-period results, and any EPRI READi framework metric
   itself (CORi, LORi, DORi), which needs fragility curves and event
   probabilities rather than an outage log.

   A separate repository becomes justified when two or more of these are true:

   1. An implementation requires resolving network topology or customer
      connectivity.
   2. An implementation requires AC or DC power flow, protection coordination,
      or recloser-sequence modelling.
   3. The reliability code exceeds roughly 30 percent of the package's source
      volume.
   4. Reliability-specific dependencies would be forced on all
      `climate_indices` users.
   5. The release cadence needs to diverge.

   Until two hold, an out-of-scope reliability proposal is not rejected on
   merit: open an issue labeled `reliability` recording it as a candidate for
   the eventual split, rather than relitigating the boundary here.
   [CONTRIBUTING.md](https://github.com/monocongo/climate_indices/blob/main/CONTRIBUTING.md#power-system-reliability-scope)
   records the triage response and repeats these criteria.

10. **The conformance target is IEEE Std 1366-2022, and the analytic fixtures
    are recorded here.** IEEE Std 1366-2022 is the current edition (approved 21
    September 2022, published 22 November 2022) and supersedes IEEE Std
    1366-2012, so 2022 is the edition the later tickets reproduce. The index
    definitions are stable across the two editions; the target is named so a
    reader is not left to guess. The hand-computed cases the implementation
    tickets must reproduce are:

    | Case | Input | Expected |
    | --- | --- | --- |
    | SAIFI | `N_T = 1000`, `Σ N_i = 300` | 0.3 interruptions per customer per period |
    | SAIDI | `N_T = 1000`, `Σ r_i N_i = 24 000` customer-minutes | 24.0 minutes per customer per period |
    | CAIDI | `Σ r_i N_i = 24 000`, `Σ N_i = 300` | 80.0 minutes per interruption; equals `SAIDI / SAIFI` |
    | MAIFI_E | `N_T = 1000`, `Σ IDE_i N_i = 300`, where `IDE_i` is 1 for a supplied momentary row without `momentary_event_excluded` set and 0 for any other record; `group_momentary_events()` is the opt-in helper that produces one such row per event from a raw operation log | 0.3 momentary events per customer per period |
    | MAIFI, device-operation form | Same record as MAIFI_E, `N_T = 1000`, `Σ device_operations_i N_i = 750` | 0.75 device operations per customer per period; differs from MAIFI_E 0.3 on the same record |
    | ASAI | `N_T = 1000`, `H = 24 h`, `Σ r_i N_i = 24 000` customer-minutes | 0.983333, from `1 − 24 000 / (1000 · 24 · 60)` |
    | No outage, operation data absent | Empty event record, `N_T = 1000`, `H = 24 h`, no `device_operations` column | SAIFI 0, SAIDI 0, MAIFI_E 0, ASAI 1.0; CAIDI undefined (0/0); MAIFI undefined because operation data is absent. The result names both undefined metrics and emits at most one warning. |
    | No outage, operation data available | Empty event record, `N_T = 1000`, `H = 24 h`, `device_operations` column supplied | SAIFI 0, SAIDI 0, MAIFI_E 0, MAIFI 0, ASAI 1.0; only CAIDI is undefined (0/0). The result names it and emits at most one warning. |
    | Leap year | 366-day period | `H = 8784` hours, against 8760 for 365 days |
    | Major Event Day threshold | Daily SAIDI `[0, e^-1, 1, e, 0]` | Zero days excluded; logarithms `[-1, 0, 1]`, `α = 0`, sample `β = 1`, `T_MED = exp(2.5)` |
    | Strict boundary | One day with daily SAIDI exactly `T_MED = 6.0` | Not a Major Event Day; with `T_MED` one float64 step below 6.0 it is |
    | IEEE 2012 example | Table 2 history, then Table 3 daily SAIDI for January 1994 | `α` prints −0.555 and `β` 1.90; January 28 (237.493) is the only Major Event Day; SAIDI 287.348 unfiltered and 49.855 filtered (printed 287.35 and 49.86) |

    These are analytic checks, not external validation: no event-level oracle is
    available, because EIA Form EIA-861 Table 11.1 is aggregated and can validate
    relationships rather than arithmetic. The validation ticket records that
    honestly in `VALIDATION.md`.

11. **Sources are cited for what they are.** IEEE Std 1366-2022 defines the
    indices and the Major Event Day recipe. The Major Event Day rules
    implemented here are verified against IEEE Std 1366-2012 only (decision 7).
    Christie's paper supports the sample standard deviation; it is not normative
    text. EIA Form EIA-861 Table 11.1 is the
    aggregated national reporting context. The EPRI Climate READi power-system
    modelling guide (Distribution Operations, Appendix A) is the climate-risk
    motivation and states that the industry has not settled on a resilience
    metric. **EPRI product 1008508, "Correlating Power Quality Indices with
    System Reliability Indices", is a March 2005 power-quality report, not a
    Climate READi document.** The product URL supplied with the epic resolves to
    it; it remains a valid corroborating application reference because its
    section 2.2 reproduces the IEEE SAIFI/SAIDI/CAIDI/ASAI/MAIFI equations, but
    it is not cited as a Climate READi source.

## Consequences

The reliability family has a package path, a public surface, a threshold, a
MAIFI flavour, a planned-outage default, a unit contract, and an out-of-scope
list before any implementation ticket starts, so ENERGY2-3 through ENERGY2-11
have no convention left to guess.

`VALIDATION.md` gains a reliability row only when the implementation lands, and
that row is specification-and-regression, not external validation, because no
event-level oracle was found. The Monte Carlo path is unit-and-regression
evidence against the analytic series and parallel results.

The named follow-ups are verification of the Major Event Day rules against a
licensed IEEE Std 1366-2022 copy; network
topology resolution and customer connectivity, gated on the criteria in decision
9; and the xarray adapter and CLI integration, which consume the package rather
than being components of it, as for fire and flood.

This amends {doc}`0001-dual-numpy-xarray-api` only for the reliability family, as
{doc}`0005-fire-module-api` did for fire. No drought, fire, flood, or energy API
changes.

## References

- IEEE Std 1366-2022, *IEEE Guide for Electric Power Distribution Reliability
  Indices*. Approved 21 September 2022, published 22 November 2022; supersedes
  IEEE Std 1366-2012. https://standards.ieee.org/ieee/1366/7243/
- IEEE Std 1366-2012, *IEEE Guide for Electric Power Distribution Reliability
  Indices*, clause 3.5, as attached to a public CPUC filing.
  https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M549/K465/549465153.PDF
- R. D. Christie, *Statistical Classification of Major Reliability Event Days in
  Distribution Systems*, revision 2, 8 August 2001, equation 6.
  https://people.ece.uw.edu/christie/Major%20Events%20Classification02.pdf
- U.S. Energy Information Administration, Form EIA-861, *Annual Electric Power
  Industry Report*, Table 11.1 (reliability).
- EPRI, *Climate READi power-system modelling guide*, Distribution Operations,
  Appendix A. https://apps.epri.com/climate-readi-modeling-guide/en/dx-ops.html
- EPRI, *Correlating Power Quality Indices with System Reliability Indices*,
  Final Report 1008508, March 2005. https://www.epri.com/research/products/000000000001008508
- {doc}`0005-fire-module-api`, {doc}`0012-xarray-api-stays-beta-through-3.0.0`,
  {doc}`0013-flood-module-api-and-naming`,
  {doc}`0014-flood-family-scientific-conventions`,
  {doc}`0019-energy-module-api-and-aggregation`
- [CONTRIBUTING.md](https://github.com/monocongo/climate_indices/blob/main/CONTRIBUTING.md#power-system-reliability-scope)

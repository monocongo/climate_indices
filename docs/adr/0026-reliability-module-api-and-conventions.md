# Reliability APIs are namespaced in the `reliability` package, with metric conventions fixed

## Status

Approved for implementation. ENERGY2-3 ([#1337](https://github.com/monocongo/climate_indices/issues/1337))
implements event normalization, `classify_interruptions()`, the event JSON Schema,
and internal served-customer and reporting-period validation.
Metric calculations, grouping, simulation, and adapters remain unimplemented.
ENERGY2-4 through ENERGY2-11
([#1338](https://github.com/monocongo/climate_indices/issues/1338)–[#1345](https://github.com/monocongo/climate_indices/issues/1345))
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
public front matter, not from a vendored copy. The EPRI Climate READi
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
   than being spread through the drought-oriented `indices.py` and `compute.py`.
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
   `classify_interruptions()`, `group_momentary_events()`, and
   `simulate_reliability()`.** One metric entry point returns every metric in a
   frozen result object:

   - `reliability_indices()` takes the event record, `total_customers`, and
     optional reporting-period inputs and returns a `ReliabilityResult`.
     `total_customers` is the positive average served-customer count over the
     period. ASAI requires positive `period_hours` or explicit period start/end
     bounds from which the library derives actual elapsed hours. Without either,
     ASAI is undefined and the result names it; the library never infers the
     reporting period from event timestamps.
     The result records the supplied period without annualizing partial periods.
   - `ReliabilityResult` is a frozen dataclass carrying SAIFI, SAIDI, CAIDI,
     MAIFI_E (and MAIFI where the input admits it), ASAI, the outage count, and
     the shared accumulators `CI = Σ N_i` and `CMI = Σ r_i N_i`, so
     `CAIDI == SAIDI / SAIFI` holds by construction rather than by convention.
     It also names any metric that is undefined for the input.
   - `classify_interruptions()` applies the sustained/momentary threshold of
     decision 3.
   - `group_momentary_events()` collapses reclosing operations of one device
     within the threshold window into one momentary event, the grouping
     decision 4 needs.
   - `simulate_reliability()` takes supplied component failure rates and repair
     distributions, a series or parallel arrangement, `total_customers`,
     `horizon_hours`, `n_trials`, and a required explicit seed.
     It produces an event log for each trial and calls `reliability_indices()`
     with that trial's log and horizon. It returns one `ReliabilityResult` per
     trial as a `tuple[ReliabilityResult, ...]`, ordered by trial index.
     [#1343](https://github.com/monocongo/climate_indices/issues/1343) implements
     this entry point. It rejects `component_topology` with `InvalidArgumentError`
     because connectivity resolution is deferred.

   The event record requires `event_id`, `start_time`, `customers_interrupted`,
   and either an `end_time` timestamp or a `duration` in minutes.
   [#1337](https://github.com/monocongo/climate_indices/issues/1337) implements the
   full schema and validation rules, including the optional flags in decisions
   4 and 5. The {ref}`event contract <reliability-event-contract>`
   documents the shipped classifier and internal validation helpers.

   **Six `saifi()`-style scalar functions are rejected.** Each metric is a ratio
   over the same two accumulators, so six functions would recompute the same
   sums six times and could disagree about the record they summed. A single
   result object computes them once, makes the `CAIDI == SAIDI / SAIFI` identity
   structural, and lets a degenerate input report all of its undefined metrics
   together.

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
   reported as a percentage. **Partial periods are computed exactly over the
   supplied period, with no annualization and no extrapolation**, so a
   partial-period index is not comparable to an annual index and the result says
   so. Hours in the period `H` are actual hours: 8760 for a 365-day year and
   8784 for a 366-day year.

7. **Major Event Days are not detected.** The IEEE 1366 recipe is documented but
   not implemented: `T_MED = exp(α + 2.5β)`, where `α` is the average of
   `ln(daily SAIDI)` and `β` is the standard deviation of `ln(daily SAIDI)`
   over typically five years of daily SAIDI, excluding days with no
   interruptions; a day whose SAIDI exceeds `T_MED` is a Major Event Day. It
   needs a utility-specific five-year daily-SAIDI history that this package does
   not hold. Results with and without major events are not comparable, and the
   result records that no Major Event Day filter was applied. Detecting and
   filtering Major Event Days is tracked in
   [#1527](https://github.com/monocongo/climate_indices/issues/1527), a separate
   follow-up outside this epic.

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
    | MAIFI_E | `N_T = 1000`, `Σ IDE_i N_i = 300`, where `IDE_i` is 1 for a momentary event that `group_momentary_events()` returns without `momentary_event_excluded` set and 0 for any other record | 0.3 momentary events per customer per period |
    | MAIFI, device-operation form | Same record as MAIFI_E, `N_T = 1000`, `Σ device_operations_i N_i = 750` | 0.75 device operations per customer per period; differs from MAIFI_E 0.3 on the same record |
    | ASAI | `N_T = 1000`, `H = 24 h`, `Σ r_i N_i = 24 000` customer-minutes | 0.983333, from `1 − 24 000 / (1000 · 24 · 60)` |
    | No outage, operation data absent | Empty event record, `N_T = 1000`, `H = 24 h`, no `device_operations` column | SAIFI 0, SAIDI 0, MAIFI_E 0, ASAI 1.0; CAIDI undefined (0/0); MAIFI undefined because operation data is absent. The result names both undefined metrics and emits at most one warning. |
    | No outage, operation data available | Empty event record, `N_T = 1000`, `H = 24 h`, `device_operations` column supplied | SAIFI 0, SAIDI 0, MAIFI_E 0, MAIFI 0, ASAI 1.0; only CAIDI is undefined (0/0). The result names it and emits at most one warning. |
    | Leap year | 366-day period | `H = 8784` hours, against 8760 for 365 days |

    These are analytic checks, not external validation: no event-level oracle is
    available, because EIA Form EIA-861 Table 11.1 is aggregated and can validate
    relationships rather than arithmetic. The validation ticket records that
    honestly in `VALIDATION.md`.

11. **Sources are cited for what they are.** IEEE Std 1366-2022 defines the
    indices and the Major Event Day recipe. EIA Form EIA-861 Table 11.1 is the
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

The named follow-ups are Major Event Day detection and filtering; network
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

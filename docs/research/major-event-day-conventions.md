# Major Event Day conventions: research for #1527

Research date: 2026-11-02. This is an internal research note, not an approved
API or a claim of IEEE conformance. No detection or filtering code accompanies it.

Follow-up, 2026-11-03: S1 was retrieved again with the same SHA-256, and clause
3.5 was read again. A second attempt to read the 2022 text failed. The IEEE SA
and IEEE DRWG pages returned HTTP 403, and IEEE Xplore returned a bot challenge.
#1527 then implemented the 2012 rules. ADR-0026 decision 7 records the adopted
conventions and the resolution of the open questions below. The printed Table 2
threshold is `exp(4.20)`, from the printed `α = -0.555` and `β = 1.90`.

## Relevance and scope

[Issue #1527](https://github.com/monocongo/climate_indices/issues/1527) remains
relevant. [ADR-0026](../adr/0026-reliability-module-api-and-conventions.md),
decision 7, explicitly defers Major Event Day (MED) detection and filtering.
A MED is a day whose daily System Average Interruption Duration Index (SAIDI)
exceeds a threshold derived from that utility's history.

The repository state changed after the initial triage. PR #1526 merged, and
issues #1337 and #1338 closed. The event model and SAIFI metric now exist.
At repository commit `6601566f6d99efe75768bc3dcdb05c8d30f36d50`,
`reliability_indices()` computes SAIFI, customer interruptions, and the sustained
row count. It has no MED input or result metadata.
[PR #1540](https://github.com/monocongo/climate_indices/pull/1540) implements
SAIDI for #1339, but remains open at this check.
Issue #1342, which adds availability and reporting-period support, also remains
open. This research does not duplicate those implementations or close #1527.

## Conclusion and evidence limits

The accessible IEEE Std 1366-2012 text establishes most of the historical
recipe. It does not establish the current project's 2022 conformance target.
A public CPUC filing labels Appendix C as IEEE 1366-2022, but the attached
standard's title and page headers identify **IEEE Std 1366-2012**. The MED
clause is 3.5, on printed pages 10–13, PDF pages 43–46. [S1]

The IEEE Xplore record did not yield readable standard text. The ANSI store
returned HTTP 403. The IEEE SA URL did not yield the 2022 MED clauses through
the available reader. No access controls were bypassed, and no standard was
purchased. A licensed 2022 copy remains necessary before ADR-0026 can record
verified 2022 requirements. [S7]

This note separates three kinds of evidence:

- The **2012 specification** establishes the rules attributed to that edition.
- The method author's paper and IEEE working-group material provide supporting
  evidence, not replacement normative text.
- The proposed validation policy and synthetic checks are project choices and
  specification-level evidence. They are not external numerical validation.

The implementation must not silently interpret an unknown day as a zero day.
The sources inspected do not settle missing-data or degenerate-history behavior.

## Rules verified in IEEE Std 1366-2012

The following statements refer to S1, clause 3.5, unless another location is given.
They must be checked against the 2022 edition before implementation.

| Topic | Verified 2012 rule | Consequence for the proposed contract |
| --- | --- | --- |
| History interval | Use five sequential years ending on the last day of the last complete reporting period. Use all available history when fewer than five years exist. | Do not replace this with a fixed 1,825-value slice. Record dates and leap days. A five-year minimum would be an additional project restriction. |
| Application interval | Calculate the threshold at the end of a reporting period for the subsequent period. | Do not fit the threshold to the period being classified. |
| Zero days | Exclude days without interruptions. The example explicitly excludes its zero-SAIDI day because its logarithm is undefined. | Keep known zeros in the dated history for completeness checks, but not in the logarithmic sample. |
| Transform | Compute the mean α and standard deviation β of the natural logarithms of daily SAIDI. | Do not compute β from raw SAIDI or use base-10 logarithms. |
| Threshold | Equation 29 gives `T_MED = exp(α + 2.5β)`. | Use the same duration units for the history, threshold, and target values. |
| Boundary | A MED has daily SAIDI strictly greater than `T_MED`. | Equality is not a MED. Do not add a tolerance to the classification comparison. |
| Multi-day interruptions | Accrue the entire interruption to its start day. | Do not split its duration at midnight. Require a declared calendar and timezone for day assignment. |
| Metrics after classification | Calculate indices with identified MEDs removed, and analyze/report MED activity separately. | Apply one day mask through the shared metric path, not separate scalar formulas. Preserve the unfiltered result. |
| Customer denominator | The definition of total customers served uses the reporting period's average. A different customer total must be clearly defined. | Record the customer-count basis. Raw customer-minutes are not daily SAIDI. See printed page 4. |
| Applicability | The method assumes the transformed data closely resembles a Gaussian distribution. | A computed threshold alone does not establish the method's suitability. No automatic normality test is selected here. |

The history rule permits shorter records in the 2012 edition. It supplies neither
a numeric minimum positive-day count nor a rule for an empty logarithmic sample.
A partial history and an incomplete history are different cases.
A missing date does not prove that no interruption occurred.

## Standard-deviation convention

The inspected 2012 clause says standard deviation without an explicit divisor.
Richard D. Christie's originating paper gives the sample estimator in equation 6,
on page 11: [S2]

```text
x_i = ln(daily_SAIDI_i), for daily_SAIDI_i > 0
α = sum(x_i) / n
β = sqrt(sum((x_i - α)**2) / (n - 1))
```

This supports **sample standard deviation**, corresponding to NumPy `ddof=1`.
It does not prove that the 2022 standard mandates it.
The paper also discusses alternatives that the final recipe does not adopt;
its other details must not be imported into the project automatically.

The 2012 Table 2 example provides additional, limited evidence. Recomputing logs
from its 30 printed positive daily-SAIDI values gives these results: [S1]

| Calculation from printed SAIDI | Result |
| --- | --- |
| Mean logarithm | `-0.55527230384916` |
| Sample β | `1.9046060997161485` |
| Population β | `1.8725936332370925` |
| Sample-estimator threshold | `67.10395240460595` |
| Population-estimator threshold | `61.94282481943798` |

The standard reports α as `-0.555`, β as `1.90`, and `T_MED` as `66.69`.
Its β agrees with the rounded sample result, not the rounded population result.
However, the printed input and intermediate values do not reproduce the reported
threshold exactly. Recomputing from the separately printed log column also changes
the result. This example is **not a full-precision threshold oracle**.
Do not change a tolerance merely to call it reproduced.

Recommendation: confirm the estimator in the 2022 text and its worked example.
If that text also leaves the divisor unspecified, obtain maintainer approval for
`ddof=1` and document it as an explicit project convention supported by S2.

## Filtering and comparability

The 2012 Table 3 example reports SAIDI `287.35` with all days and `49.86` with
January 28 removed. Summing the printed daily values gives `287.348` and `49.855`,
respectively. Equations 31–32 use the daily sums without a day-count multiplier.
This supports removal without annualizing the remaining observations for SAIDI.
It does **not** settle filtered ASAI's exposure denominator. [S1]

An August 2002 IEEE working-group draft contains two rules that differ from the
published 2012 recipe. It replaces zero days with the smallest positive value,
and scales filtered indices by `D_period / (D_period - D_MED)`.
Those draft rules must not be mistaken for verified 2022 requirements. [S3]

Do not iteratively remove large historical values and refit the threshold.
The 2012 recipe does not prescribe that procedure. Its clause 5.3 discusses
catastrophic-event handling as a utility/regulator decision. The working group's
2025 benchmarking presentation retains catastrophic-day effects in later thresholds
and warns that collection and exclusion differences limit comparisons. [S1, S4]

EIA-861 Schedule 3, Part B requests both including-MED and excluding-MED SAIDI
and SAIFI. Its inspected instructions still name the 2003 and 2012 IEEE editions.
They establish reporting context, not verification of the 2022 algorithm or an
event-level numerical oracle. [S5]

Filtered and unfiltered values describe different event sets. Matching filter
status is necessary but insufficient for comparison. Reporting intervals, customer
populations, outage definitions, history windows, units, and day boundaries must
also agree. The working group's survey records differences in several of these
conventions. [S6] Partial-period indices must not be annualized. ADR-0026 already fixes
that project policy; this research does not change it.

## Proposed validation policy, pending approval

*Pre-implementation snapshot.* ADR-0026 decision 7 adopted the validation policy
and #1527 implemented it; this section records what was proposed before that.
These are conservative library proposals, **not verified IEEE requirements**.
The existing `InvalidArgumentError` should identify the offending input.
No additional dependency is needed.

| Input condition | Proposed behavior |
| --- | --- |
| Missing dates, masked values, `None`, or NaN | Reject the history as incomplete. Never fill unknown days with zero or silently drop them. |
| Duplicate dates or incompatible day boundaries | Reject ambiguous history. Require one daily value per utility and declared day. |
| Negative values or either infinity | Reject invalid SAIDI before applying logarithms. |
| All-zero or empty history | Reject because no logarithmic sample exists. Do not return zero or infinity as a threshold. |
| One positive day | Reject if the approved estimator is `ddof=1`; its standard deviation is undefined. |
| Fewer than five years, with at least two positive days | If 2022 confirms the earlier rule, accept the complete shorter interval and record its reduced history length. Do not claim equivalent statistical support. |
| Constant positive values | The formula gives β = 0. Test the strict boundary separately; constant data does not establish Gaussian suitability. |
| Overflow or non-finite derived threshold | Report an explicit computation error. Do not continue with an infinite threshold. |

A two-positive-day mathematical minimum is not a claim of adequate statistical
history. A stronger minimum would require a separate, justified decision.
No warning threshold or goodness-of-fit cutoff is selected by this research.

## Public API and metadata work still required

*Pre-implementation snapshot.* ADR-0026 decision 7 fixed the public API and the
recorded metadata, and #1527 implemented them; this section records the
questions that were open before that decision.

Do not add a second set of scalar reliability metrics. The implementation should
keep detection separate from opt-in filtering at `reliability_indices()`.
Names and signatures remain unapproved. ADR-0026 must define them after the
remaining source and policy questions are resolved.

A proposed detection result should record the threshold, units, estimator divisor,
positive-day count, zero-day count, history bounds, utility identity, and method
edition. Bounds alone do not prove that the history has no missing days.
Filtering metadata should additionally record the reporting interval, filter status,
excluded dates, calendar/timezone, customer-count basis, and planned-outage policy.
The default must remain unfiltered.

Use one validated event mask for all deterministic metrics and shared accumulators.
Detection uses sustained-event daily SAIDI, but removing a MED applies to all
relevant events on that day, not only the interruption that triggered classification.
The treatment of momentary events must agree with the completed metric contract.

Decisions still required include filtered ASAI exposure hours, interactions with
planned-outage exclusions, and the relationship between history and target customer
populations. Restoration-stage rows also need an unambiguous original interruption
start day when their own timestamps cross midnight. Do not infer these choices
from the short threshold formula.

## Small analytic check for the later tests

This synthetic history's positive values have logarithms `[-1, 0, 1]`, mean zero,
and sample standard deviation one. Zeros do not change the positive-day sample.
With the proposed sample convention, `T_MED = exp(2.5)`.
The population convention instead gives `exp(2.5 * sqrt(2/3))`.
The following standard-library check was executed during this research:

```python
import math
import statistics

history = [0.0, math.exp(-1), 1.0, math.e, 0.0]
logs = [math.log(value) for value in history if value > 0]
assert len(logs) == 3
assert math.isclose(statistics.mean(logs), 0.0, rel_tol=0, abs_tol=1e-15)
assert math.isclose(statistics.stdev(logs), 1.0, rel_tol=0, abs_tol=1e-15)
threshold = math.exp(statistics.mean(logs) + 2.5 * statistics.stdev(logs))
assert math.isclose(threshold, math.exp(2.5), rel_tol=0, abs_tol=1e-14)
assert not (threshold > threshold)
assert math.nextafter(threshold, math.inf) > threshold
assert not (math.nextafter(threshold, 0.0) > threshold)
assert math.isclose(
    math.exp(2.5 * statistics.pstdev(logs)),
    math.exp(2.5 * math.sqrt(2 / 3)), rel_tol=0, abs_tol=1e-14,
)
```

The absolute tolerances cover floating-point log/exp roundoff for this small
synthetic case. The day-classification comparisons deliberately have no tolerance.
These are analytic checks, not tests of a library implementation.
Later regression tests must call the actual public API, exercise the validation
matrix, and compare filtered and unfiltered shared accumulators.

## Implementation prerequisites and next action

1. Obtain lawful access to IEEE Std 1366-2022.
2. Check its MED clause, worked example, applicable annex, and errata against the findings above.
3. Resolve the estimator, validation, calendar, restoration, and filtered-exposure questions in ADR-0026.
4. Wait for the shared duration and reporting-period metric paths to land.
5. Implement and test detection, opt-in filtering, and metadata together under #1527.

These prerequisites were resolved differently than listed. Steps 3–5 were
completed: ADR-0026 decision 7 fixed the conventions, the shared metric paths
landed, and #1527 implemented detection, filtering, and metadata. Steps 1 and 2
were not achieved, so the implementation rests on the 2012 evidence.

This spike supplies source evidence and explicit gaps. It does not satisfy all
acceptance criteria for #1527, approve an API, or change the evidence in `VALIDATION.md`.

## Sources and retrieval record

All sources below were consulted on 2026-11-02. External URLs can change.
No standard PDF or full source text is committed with this note.

- **S1 — IEEE Std 1366-2012 in a public CPUC filing.** Clause 3.5 and example,
  printed pp. 10–13; definitions, p. 4; clause 5.3, pp. 19–20.
  Appendix C's cover says 2022, but the attached title and page headers say 2012.
  PDF pages 43–46 contain the MED text and tables; page 45 was also checked visually.
  <https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M549/K465/549465153.PDF>
  Retrieved PDF SHA-256: `653914289a9e0970ea320bb463a690807edd5e6d684d78c768b0ae9897f04fa0`.
- **S2 — Richard D. Christie, Statistical Classification of Major Reliability
  Event Days in Distribution Systems.** Revision 2, 8 August 2001; p. 11,
  equations 5–6. The author-hosted PDF's filename does not establish its date.
  Equation 6 was checked visually and uses `n - 1`.
  <https://people.ece.uw.edu/christie/Major%20Events%20Classification02.pdf>
  Retrieved PDF SHA-256: `1adde5efedbeb7ebf5efac348c7b5eaf8f350578ed2349cd9c5538ea60e6f07d`.
- **S3 — IEEE DRWG, P1366 Major Event Day Language Draft, August 2002.**
  Step 2 and draft clause 4.4.1. This is a draft, not the adopted 2022 standard.
  <https://cmte.ieee.org/pes-drwg/wp-content/uploads/sites/61/2002-08-P1366MajorEventDayLanguageDraft4.pdf>
- **S4 — IEEE DRWG, 2025 Benchmark Study for 2024 Results.**
  Slide 3, comparability and catastrophic-day exclusions.
  <https://cmte.ieee.org/pes-drwg/wp-content/uploads/sites/61/benchmarking-2025-july-2025-updated.pdf>
- **S5 — EIA-861 Annual Electric Power Industry Report Instructions.**
  Schedule 3, Part B, PDF p. 9, including lines 3–7.
  <https://www.eia.gov/survey/form/eia_861/instructions.pdf>
- **S6 — IEEE DRWG, 2022 Benchmarking Survey.** Slide 8 records different
  calendar-day, rolling-window, start-accrual, and customer-count practices.
  This describes respondents, not requirements of the standard.
  <https://cmte.ieee.org/pes-drwg/wp-content/uploads/sites/61/2022-Benchmarking-Survey.pdf>
- **S7 — 2022 standard access attempts.** These did not provide readable MED
  requirements. Their metadata or search summaries cannot replace the standard.
  <https://standards.ieee.org/ieee/1366/7243/>
  <https://ieeexplore.ieee.org/document/9955492>
  <https://webstore.ansi.org/standards/ieee/ieee13662022>

No jurisdiction-specific variation from the survey is selected as a project default.

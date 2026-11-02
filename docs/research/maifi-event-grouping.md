# Anchor MAIFI_E events at the first operation

Internal research for [#1341][issue-1341], under decision 4 of [ADR-0026][adr-0026].

**Research date:** 2026-11-02

**Scope:** Momentary-event grouping and customer counting at repository commit
`3737bdcbe15ca526b507d19aadd89e63bb5f65fd`. The primary text is IEEE Std 1366-2012 from a
public Indiana docket. No IEEE Std 1366-2022 clause text was read. Each library rule below
is a proposal that no source prescribes.

## Answer

The names are settled: `maifi_e` counts momentary interruption events, and `maifi` counts
device operations ([ADR-0026, decision 4][adr-0026]). #1341 should anchor each event at the
first operation of one device. The window includes exactly five minutes and never chains
from the latest operation. The explanatory notes also describe completion and a hold within
five minutes. The proposed helper therefore checks completion. `reliability_indices()` should count each supplied
momentary row as one event. The 2012 worked example stores whole sequences as rows, and
regrouping them changes MAIFI_E from 7.25 to 6.25. Raw operation logs go through an opt-in
`group_momentary_events()`. Different customer counts in one group raise
`InvalidArgumentError`. Equal counts are accepted under a stated same-customer assumption.
The 2022 wording is not verified.

## Background: operations, sequences, and events

A recloser or breaker opens to clear a fault and then closes. Each open followed by a close
is one momentary interruption, called an operation here. For example, two to four operations followed
by a hold form one momentary interruption event ([IEEE 2012, Clause 2, printed p. 3, PDF p.
14][ieee-2012]). A locked-out device stays open until manual intervention ([IEEE 2012,
printed p. 2, PDF p. 13][ieee-2012]). An interruption that is not part of a momentary event
is sustained ([IEEE 2012, printed p. 4][ieee-2012]). MAIFI weights each operation by its
customers. MAIFI_E weights each event by its customers. Both divide by customers served
([IEEE 2012, 3.4.1–3.4.2, printed p. 9, PDF p. 20][ieee-2012]). In 4.3.1, a breaker, a
recloser, and a sectionalizer serve 2000 customers. The figure shows 1000 upstream customers
with no interruption, 750 momentary customers, and 250 sustained customers beyond the
sectionalizer. The guide prints **SAIFI 0.125, MAIFI 2 × 750 / 2000 = 0.75, and MAIFI_E 1 ×
750 / 2000 = 0.375** ([IEEE 2012, 4.3.1, printed p. 17, PDF p. 28][ieee-2012]). The factor 2
is two recloser shots, and the factor 1 is one event.

## The window opens at the first operation and must close in five minutes

NOTE 1 of the 2012 event definition says the operations "must be completed within" a time of
"five minutes or less". It includes the operations that "occur within five minutes of the
first interruption". NOTE 2 describes a device that operates and "then holds" within five
minutes of the first operation ([IEEE 2012, Clause 2, printed p. 3, PDF p. 14][ieee-2012]).
The 2003 edition uses the same anchor ([IEEE 2003, 3.15, PDF p. 17][ieee-2003-fm]).

**The anchor is the first operation, not sliding adjacency.** Constructed operations at 0,
4, and 8 minutes do not belong to one anchored five-minute sequence. Chaining would merge
them, and no retrieved source supports it. The text does not say what the 8-minute operation becomes,
so the proposal lets it open a new event. **Exactly 5:00 is inside**, because the text says
five minutes or less. **Completion is a second condition**, because two of the three phrases
refer to completion or a hold.

The schema always gives an end, from `end_time` or `start_time` plus `duration` ([event
schema][schema]). A constructed operation from 4:50 to 5:20 joins by start time but fails
completion. So a start-only rule is not full IEEE conformance. The standard states that its notes are "given for
information only and do not contain requirements" ([IEEE 2012, printed p. 2][ieee-2012]).
The proposed library rule, not a source mandate, uses start-time membership inclusive at the
window length. A guard then checks that the last end does not cross anchor plus window. A
late group raises `InvalidArgumentError` instead of silently splitting it or returning a potentially misleading count.
The window is its own parameter, default 5 minutes, because Florida pairs a one-minute
threshold with a five-minute grouping period ([Fla. R 25-6.044(1)(g), (h), (m)][fl]).

The adopted text names one interrupting device. A 2009 draft proposed multi-device
aggregation, and the 2012 text omits it ([DRWG 2009, PDF p. 3][drwg-2009]). So the helper
never merges different `device_id` values. For the missing-device fallback in #1341,
null-device rows group only with each other ([#1341][issue-1341]). This library convention
can merge different unknown devices.

## Table 6 counts two sequences 108 seconds apart as two events

Table 6 records 3 and 4 are breaker 7075 on Aug 2. **Record 3 is at 12:29:02 with 1
operation, and record 4 is at 12:30:50 with 2 operations.** The guide says to "count only the total number
of reclosing sequences". It counts five breaker events, records 1, 3, 4, 7, and 9 ([IEEE
2012, 4.1, printed p. 15, PDF p. 26][ieee-2012]). Equation 44 gives MAIFI = (8 × 2000 + 12
× 750) / 2000 = 12.5. Equation 45 gives **MAIFI_E = (5 × 2000 + 6 × 750) / 2000 = 7.25**
([IEEE 2012, 4.2, printed p. 16, PDF p. 27][ieee-2012]). The 2003 copy agrees ([IEEE 2003,
printed p. 14; Attachment B pp. 22–23][ieee-2003-ky]).

A plain five-minute window merges the two records. MAIFI_E then becomes (4 × 2000 + 6 × 750)
/ 2000 = **6.25**. MAIFI stays 12.5 if each row keeps its own product. Each row is a
completed sequence, not an operation (interpretation). Utilities record momentary data either as sequences or
as individual operations ([APPA 2018 survey, p. 7][appa]). The schema has no field for row
meaning, sequence, or hold ([event schema][schema]).

So `reliability_indices()` should keep caller rows as events and never regroup. In #1341,
rows of 2 and 1 operations, 15 customers each, over 100 served give `maifi` 0.45 and
`maifi_e` 0.3 ([#1341][issue-1341]). If those rows fall within one grouping window, internal regrouping would give `maifi_e` 0.15.
`group_momentary_events()` stays opt-in for raw operation logs, and its output is one valid
input. Without hold markers, the helper cannot tell a second operation from a new sequence.
Its docstring must state this limit. The caller must supply raw operation records for one
customer group, not records that already contain separate sequences.
The helper must not derive `device_operations` from row counts. It sums supplied operation
counts only when all relevant counts are known; otherwise, the operation count remains unknown.
This follows ADR-0026 decision 4, rather than a new IEEE requirement.

## Row totals cannot tell whether 15 + 15 customers is 15 or 30

MAIFI_E needs the distinct customers of each event ([IEEE 2012, 3.1, printed p. 4, PDF p.
15][ieee-2012]). Take two constructed rows of 15 customers in one group. A three-pole
recloser shooting twice reaches the same 15. A single-pole control can trip, reclose, and
lock out each phase independently ([Cheney, Thorne, and Hataway, page not recorded][sel]).
Phase A then phase B can reach two different sets, so 30. `max` gives 15, `sum` gives 30,
and an equal-count check passes both. **No function of row totals recovers customer
identity.** Constructed rows of 1000 and 750 fit 750, 1000, or 1750. In 4.3.1, by
arithmetic, the recloser has 750 + 250 = 1000 downstream customers, but the event count is
750 ([IEEE 2012, printed p. 17][ieee-2012]).

The proposal computes `maifi` per row as `device_operations × customers_interrupted` and
sums before grouping. For `maifi_e`, different counts in a group raise
`InvalidArgumentError`. The message asks for one already-grouped row with the event-level
momentary customer count. Equal counts are accepted, and the docstring states the
same-customer assumption. Exact unions need per-customer identifiers, a later schema change.
ADR-0026 keeps customer connectivity out of scope and adds no dependency ([ADR-0026,
decisions 8 and 9][adr-0026]).

## Exclusions work at two levels, and one case stays unknown

**Event exclusion.** The 2012 MAIFI_E subclause "does not include the events immediately
preceding a sustained interruption". The MAIFI subclause has no such sentence ([IEEE 2012,
3.4.1–3.4.2, printed p. 9][ieee-2012]). The 2003 text names a lockout instead ([IEEE 2003,
4.4.2, printed p. 8, PDF p. 2][ieee-2003-mo]). The `momentary_event_excluded` flag removes
an event from `maifi_e` only. The helper rejects mixed flags in a group. Keeping those
operations in `maifi` is an inference.

**Partial sustained-customer removal.** In 4.3.1, the 250 sustained customers count in
neither momentary form ([IEEE 2012, Equations 48–49, printed p. 17][ieee-2012]). A row flag
cannot split customers inside one sequence. The caller lowers `customers_interrupted` and
records the others as sustained, which changes both forms.

**Same-device lockout.** No retrieved text says whether MAIFI keeps operations before a
lockout of the same device. The library should not infer it.

## Recommended decisions

| Topic | Recommendation | Basis |
| --- | --- | --- |
| Names | `maifi_e` counts events; `maifi` counts operations | ADR-0026 decision 4 |
| Window | Anchor at the first operation on one device; no chaining; start at anchor + window joins; own parameter, default 5 min | 2012 NOTE 1, NOTE 2, p. 3; [Florida][fl] |
| Completion | Reject a group whose last end exceeds the window | Library proposal; completion phrases, 2012 p. 3 |
| `reliability_indices()` | One momentary row is one event; never regroup | Table 6, pp. 15–16 |
| Helper | Opt-in for raw operation logs; window is an upper bound; row meaning stated | [APPA p. 7][appa]; schema |
| Devices | Never merge `device_id` values; null rows only with null rows | DRWG 2009 p. 3; #1341 |
| Customer counts | Unequal: `InvalidArgumentError`, ask for one grouped event-level row. Equal: accept, document the same-customer assumption | 2012 pp. 4, 17; no source for equal counts |
| `maifi` | Per-row product, summed before grouping | 2012 Equation 44, p. 16 |
| Event flag | `maifi_e` only; reject mixed flags | 2012 3.4.2, p. 9 |
| Sustained customers | Caller lowers `customers_interrupted`; both forms change | 2012 4.3.1, p. 17 |

## Intended test examples

Rows are on device D1 with 15 customers and `device_operations=1`, unless stated otherwise.
Rows 1–9 are constructed. Rows 10–12 use published values with a constructed row encoding.

| # | Input | Expected |
| --- | --- | --- |
| 1 | Helper; operations at 0:00 and 4:00, 30 s each | One event, `device_operations` 2 |
| 2 | Operations at 0:00, 4:00, and 8:00 | Two events, {0:00, 4:00} and {8:00} |
| 3 | 0:00, then 5:00 with zero duration | One event; inclusive boundary |
| 4 | 0:00, then 5:00 lasting 30 s | One event by start; completion guard fires |
| 5 | Operations at 0:00 and 5:01 | Two events |
| 6 | 0:00, then 4:50 to 5:20 | Guard fires; no split |
| 7 | D1 at 0:00, D2 at 1:00, null device at 0:30 and 2:00 | Three events: D1, D2, and the two null rows |
| 8 | 15 and 20 customers in one group | `InvalidArgumentError` asking for the event-level count |
| 9 | Mixed `momentary_event_excluded` in one group | `InvalidArgumentError` |
| 10 | `reliability_indices()` on 2012 Table 6 rows; 2000 breaker, 750 recloser customers; `total_customers` 2000 | `maifi` 12.5, `maifi_e` 7.25 |
| 11 | 2012 4.3.1: momentary row of 750 with 2 operations, sustained row of 250; 2000 served | SAIFI 0.125, `maifi` 0.75, `maifi_e` 0.375 |
| 12 | #1341, `reliability_indices()`: rows of 2 and 1 operations, 15 each; 100 served | `maifi` 0.45, `maifi_e` 0.3 |

## Scope and remaining limits

1. These recommendations require approval before implementation. They preserve event boundaries
   and reject unsupported customer-count estimates.
2. The helper cannot reconstruct sequence boundaries or customer identity from the existing schema.
   Null-device grouping is a library convention, not an IEEE definition.
3. Implementation must amend ADR-0026 to distinguish the grouping window from the classification threshold.
   It must also distinguish the event exclusion from the operation metric.
4. The 2022 preview has front matter only ([IEEE 2022 preview, PDF pp. 1–6][ieee-2022]).
   A 2024 panel slide reports no major changes, as a presenter's view ([DRWG 2024, slide 4][drwg-2024]).
   This research does not establish IEEE Std 1366-2022 conformance.

## Sources

[issue-1341]: https://github.com/monocongo/climate_indices/issues/1341
[adr-0026]: ../adr/0026-reliability-module-api-and-conventions.md
[schema]: ../../src/climate_indices/reliability/event_schema.json
[ieee-2012]: https://iurc.portal.in.gov/_entity/sharepointdocumentlocation/def9f7d7-91b6-ec11-983e-001dd80335bf/bb9c6bba-fd52-45ad-8e64-a444aef13c39?file=45647%2C+Public%27s+Exhibit+No.+CX-8.pdf
[ieee-2003-fm]: https://psc.ky.gov/PSCSCF/2006%20cases/2006-00494/Fleming-Mason_Report_032808.pdf
[ieee-2003-ky]: https://psc.ky.gov/pscscf/2006%20cases/2006-00494/KY%20Power_Report_040108.pdf
[ieee-2003-mo]: https://www.efis.psc.mo.gov/Document/Display/77488
[ieee-2022]: https://www.normadoc.com/media/file/b5/e7/25155dce3984aa6a393ece8f97ce.pdf
[drwg-2009]: https://cmte.ieee.org/pes-drwg/wp-content/uploads/sites/61/2009-11-17-Proposed-Changes-to-IEEE-1366.pdf
[drwg-2024]: https://cmte.ieee.org/pes-drwg/wp-content/uploads/sites/61/IEEE-1366-Panel-Presentation.pptx
[fl]: https://www.law.cornell.edu/regulations/florida/Fla-Admin-Code-Ann-R-25-6-044
[appa]: https://www.publicpower.org/system/files/documents/survey_4504402%20%284%29.pdf
[sel]: https://wprcarchives.org/wp-content/uploads/2024/07/Robert-M.-Cheney_TP_DistributionSinglePhase_GH_sek_20080917.pdf

# Compound bootstrap calibration (#1398)

Internal design record for the calibration work of issue
[#1398](https://github.com/monocongo/climate_indices/issues/1398), a follow-up to
[#1371](https://github.com/monocongo/climate_indices/issues/1371) and PR #1397.

**Read order.** The *Predeclared design* and *Scenario groups* sections were written
and committed before any measurement of this issue ran. The *Results* and *Decision*
sections were added afterwards. Every outcome is reported, including failures. No
acceptance band was widened after the run.

## Predeclared design

The measurement extends the harness that produced
`docs/design/compound-bootstrap-characterization.json`. The generative model is
unchanged: two channels of Gaussian moving averages of width `w` months, cut at
month-specific thresholds, so each channel is serially independent at `w = 1` and
persists for `w - 1` months otherwise. Each channel's monthly exceedance
probability follows `c + a * cos(2 * pi * month / 12)`. A same-time correlation
`rho` links the two channels.

- **Replication.** The four scenarios of the original design, the same configuration
  (200 years, 399 resamples, block lengths 3, 6, and 12 years, 128 records, seed
  1371), the same seeds, and the same per-case gates. Replication answers whether the
  failed coverage gate and the excess null rejection reproduce.
- **Trials, bands, and budget.** 128 independent records per scenario. A coverage
  case passes when at least 109 of 128 records cover the truth. An independent-null
  case passes when at most 19 of 128 records reject at nominal 0.05. Both bands come
  from the exact binomial with a per-comparison tail of `0.01 / 864 = 1.16e-5`, the
  value the #1371 design predeclared for its 864 comparisons. The extension groups
  reuse these two bands unchanged, so no case is judged by a wider band. The
  extension does not claim a familywise 0.01 across all extension comparisons; it
  reports every case instead.
- **Wilson intervals.** Every measured proportion carries a Wilson 95% interval.
- **Truths.** Analytic bivariate-normal quadrature supplies each month's truth for
  lag window `(0, 0)`, independently of the counting code. Lagged conditioning has no
  closed form under the moving-average model, so its truth is a large-sample
  simulation: 40,000 years per scenario, direct NumPy counting of anchor-window
  responses, in code separate from the library. The Monte-Carlo standard error is
  below 0.7% per month, far inside the gate band.
- **Missing values.** A missing interval counts as uncovered and a missing p-value
  fails the null gate. The single exception is declared here: a record whose
  `quality_flags` carry a structural guard (`insufficient_years` 1,
  `insufficient_blocks` 2, or `sparse_occurrences` 4) carries no inference by design,
  so it leaves the band and is reported as `structural_exclusions`.
  `insufficient_successes` 8 and `insufficient_expected_count` 16 are not structural
  exclusions; such a month counts as a failure.
- **Band refinement, recorded before the documented run.** The first run of this plan
  showed that a structural guard can fire on a subset of a month's records, which the
  original wording did not anticipate: it treated a month as measurable only when no
  record was excluded, so such months could not be judged at all. The rule now applies
  the *same* exact-binomial tail to the number of records that carry an inference, and
  it reports that count per month as `inferable_records`. With nothing excluded the band
  is unchanged at 109 covered and 19 rejected of 128. Every 200-year case has nothing
  excluded, so the replication group is unaffected. The refinement does not widen an
  allowance: the tail is unchanged, and fewer answered records give a proportionally
  smaller allowance. The Results section states which gate outcomes the refinement
  changes.

### Predeclared decision rules

1. If the 36-month independent case repeats the failed coverage gate or the excess
   pooled null rejection at the original configuration, keep the method unchanged.
2. For each newly measured condition, extend the documented supported scope to it
   only when every predeclared gate of that condition passes. Otherwise record the
   condition as unsupported and name the failing cases.
3. Add a support guard only if a quantity observable from the function's own inputs
   separates the passing from the failing persistent cases without refusing a
   supported case. Otherwise add no guard and state the limits in prose. The
   separation check uses the monthly lag-1 autocorrelation of each channel's
   occurrence series and its exceedance fraction, both computed from the same
   records and reported with the results.

## Scenario groups

| Group | Condition under measurement | Scenarios |
| --- | --- | --- |
| `replication` | The #1371 design | 4 scenarios at 200 years, 399 resamples, lengths 3/6/12 |
| `record_length` | Record length from the 30-year guard floor, including the five-block and ten-block edges | `w = 1` and `w = 6` at 30 years (lengths: library default 4, 6, and 3 years, giving 7, 5, and 10 blocks), 60 years (default 4 and 6 years, giving 15 and 10 blocks), and 100 years (default 5 and 6 years) |
| `persistence` | Persistence between the supported 6 months and the unsupported 36 months | `w = 12, 18, 24, 30` at 200 years, default 6-year blocks |
| `lagged_conditioning` | Conditional lag windows other than `(0, 0)` | `w = 1` and `w = 6` at 200 years with windows `(0, 1)`, `(0, 3)`, and `(1, 3)`, all three block lengths, conditional statistic only |
| `hazard_frequency` | Monthly exceedance probabilities below 0.35 and above 0.55 | Profiles `low` (0.20 to 0.30), `high` (0.70 to 0.80), and `wide` (0.30 to 0.60) at `w = 1` and `w = 6`; profiles `low` and `high` at `w = 36`; profile `low` at 30 years and `w = 1`; all at 6-year blocks |
| `dependent_hazards` | Channels that depend on each other at short persistence | `rho = 0.5` at `w = 1`, 6, and 12 at 6-year blocks (the `w = 36` case is in `replication`) |
| `default_resamples` | The default 1,999 resamples | The four `replication` scenarios at 1,999 resamples and lengths 3/6/12 |

Every scenario runs all three statistics unless its row says otherwise. A `None`
block length requests the library default, `ceil(years ** (1 / 3))` years, and the
report records the length the library used.

## Reproduction

```
uv run scripts/characterize_compound_bootstrap.py --output docs/design/compound-bootstrap-characterization.json
uv run scripts/characterize_compound_bootstrap.py --output docs/design/<partial>.json --group replication
```

Seeds are fixed. A scenario's resample streams derive from the seed, its
predeclared `stream` index, and the trial number, so the report is reproducible
and the groups are mutually independent. Runtime is serial: about three minutes
for the `replication` group and about twenty minutes for the whole plan on the
recording machine (Apple M-series, 10 cores, CPython 3.14).

## Results

The plan ran on 2026-10-19 with CPython 3.14.7, NumPy 2.4.2, and SciPy 1.17.0. It
produced 183 cases. The measured report is
`docs/design/compound-bootstrap-characterization.json`. Runtime was 17 minutes on an
idle machine. The gates are the predeclared bands: at 128 answered records, coverage
passes at 109 or more and null rejection at 19 or fewer per month. **47 cases fail a
predeclared gate.** Every outcome is in the report; this section states the pattern.

| Group | Cases | Passing | Coverage failure | Null failure | Not inferable |
| --- | --- | --- | --- | --- | --- |
| `replication` | 36 | 35 | 1 | 0 | 0 |
| `record_length` | 42 | 12 | 26 | 4 | 0 |
| `persistence` | 12 | 12 | 0 | 0 | 0 |
| `lagged_conditioning` | 18 | 18 | 0 | 0 | 0 |
| `hazard_frequency` | 30 | 15 | 9 | 6 | 0 |
| `dependent_hazards` | 9 | 9 | 0 | 0 | 0 |
| `default_resamples` | 36 | 35 | 1 | 0 | 0 |

### Three failure mechanisms, and only one is miscalibration

The gate flags alone mislead, so the report is read with three questions per month.

1. **Coverage misses: 37 cases.** The interval missed the truth on more records than
   the band allows. This is the calibration signal.
2. **Suppressed p-values: 45 cases.** The library returned NaN instead of a p-value on
   some records, because fewer than 95% of the resamples produced a valid null statistic
   (`invalid_null_resamples`), which a resampled month with a zero denominator causes.
   The predeclared missing-value rule fails the null gate here. This is a support
   condition, not an excess rejection rate. 35 cases show both this and a coverage miss.
3. **Excess rejections: no case.** No independent-hazard (`rho = 0`) case exceeded its
   per-month rejection band anywhere in the plan. Pooled rejection rates over the 24
   independent-hazard 200-year scenarios range from 3.5% to 7.5%, and every per-month
   null gate that failed did so through a suppressed p-value. The #1371 concern about
   null rejection does not reproduce as a per-month gate failure.

### Replication and the 1,999-resample default (AC3)

The `replication` group reproduces the previous report field by field: all 36 cases
match on truth, counts, Wilson intervals, gate flags, and block-length decision
differences. The same single case fails: `likelihood_multiplication_factor` with
12-year blocks in the 36-month independent scenario, February, 106 of 128 covered
against the minimum of 109 (82.8%, Wilson 75.3% to 88.4%).

The 36-month independent scenario keeps its pooled rejection of 7.5% [7.0, 7.9] at 399
resamples. At the default 1,999 resamples the same scenario passes all 9 gate
checks, its pooled rejection falls to 3.5% [3.2, 3.8], and the previously failing
February interval rises above the band. Most of the 399-resample excess is therefore a
resample-count artifact, not a defect of the null test. The 36-month correlated
scenario still fails one coverage check at the default resamples: conditional
probability with 3-year blocks, month 4, 107 of 128 covered.

### Record length (AC4)

| Record | Scenario | Coverage | Pooled rejection | Structural exclusions | Gate outcome |
| --- | --- | --- | --- | --- | --- |
| 30 years | `w = 1` | 78.0% (10584) | 4.8% [4.4, 5.2] | 3240 of 13824 month-records | 9 coverage failures |
| 30 years | `w = 6` | 75.1% (10647) | 3.8% [3.4, 4.1] | 3177 of 13824 | 9 coverage failures |
| 60 years | `w = 1` | 90.1% (9216) | 5.7% [5.3, 6.2] | 0 | 2 coverage, 4 null failures |
| 60 years | `w = 6` | 89.0% (9216) | 5.8% [5.4, 6.3] | 0 | 6 coverage failures |
| 100 years | `w = 1` | 92.3% (9216) | 4.8% [4.4, 5.3] | 0 | pass |
| 100 years | `w = 6` | 92.6% (9216) | 4.1% [3.7, 4.5] | 0 | pass |

Both 100-year scenarios pass all 12 gate checks, at the library default block length
(5 years) and at 6-year blocks. Both 60-year scenarios fail: coverage in months 6 to 8
falls as low as 94 of 128 at `w = 6`, and the low months suppress p-values on up to 10
of 128 records at `w = 1`. The 30-year floor is worse: guards remove 23% of month-records
at `w = 1` and 63% in the low-frequency scenario, and the answered records fail both
gates: months 7 and 8 cover about half of the records that answered, for example 24 of
50 at 6-year blocks, where the gate needs 95%. The five-block edge (30
years, 6-year blocks) and the ten-block edge (30 years at 3-year blocks, 60 years at
6-year blocks) behave the same as the other block lengths at their record length, so
block count from 5 to 20 does not separate passing from failing records.

### Persistence between 6 and 36 months (AC5)

12, 18, 24, and 30 months of persistence all pass every gate at 200 years and 6-year
blocks: coverage 92.4% to 94.3%, pooled rejection 3.5% to 5.6%. The break is therefore
between 30 and 36 months, and 36 months fails on coverage, not on rejection.

### Lagged conditioning (AC5)

All 18 cases pass: windows `(0, 1)`, `(0, 3)`, and `(1, 3)` at `w = 1` and `w = 6`, at
all three block lengths. Coverage ranges from 91.5% to 93.4% and pooled rejection from
4.4% to 6.1%. The 40,000-year simulation reference agrees with quadrature at zero lag,
which the test suite checks.

### Hazard frequency (AC5)

| Profile | Monthly range | `w = 1` | `w = 6` | `w = 36` |
| --- | --- | --- | --- | --- |
| `low` | 0.20 to 0.30 | 3 null failures | 3 null failures | 3 coverage failures |
| `wide` | 0.30 to 0.60 | pass | pass | not run |
| `high` | 0.70 to 0.80 | pass | pass | pass |

The failures concentrate in the lowest months of the `low` profile, where the monthly
probability is 0.20: the library suppresses the p-value on 1 to 7 of 128 records, with 2
to 3 of those at `w = 1`, and at 36-month persistence the LMF interval for month 7
covers 104 of 128 records. Every frequency case ran at 399 resamples. The `wide` profile
also reaches 0.30, and it passes everywhere it ran. The measured supported ranges are
therefore 0.30 to 0.60 and 0.70 to 0.80, not the 0.35 to 0.55 range of the original
seasonal profile alone; 0.60 to 0.70 was not measured. Below 0.30, the library becomes
unable to form a null statistic on enough resamples.

At 30 years with the `low` profile the guards remove 8688 of 13824 month-records (63%)
and the answered records cover the truth on 33.5% of 528 checks, so records that short
are unsupported at any measured frequency.

### Dependent hazards (AC5)

All 9 cases pass at 399 resamples: same-time correlation 0.5 at 1, 6, and 12 months of
persistence, at 6-year blocks. Coverage is 92.6% to 93.9%. `replication` adds a
correlated 36-month case at 399 resamples, which passes coverage, and
`default_resamples` adds it at 1,999 resamples, where it fails one coverage gate. No
correlated case was measured with the wider frequency profiles or the lagged windows.

### Guard separation (decision rule 3)

The predeclared rule allowed a new support guard only if an observable separated the
passing from the failing persistent cases without refusing a supported case. No such
quantity exists in this evidence:

| Scenario | Years | Width | Mean within-month lag-1 autocorrelation | Occurrence fraction | Gate outcome |
| --- | --- | --- | --- | --- | --- |
| `persistence_30_month_independent` | 200 | 30 | 0.398 | 0.452 | pass |
| `frequency_high_36_month_persistence` | 200 | 36 | 0.419 | 0.747 | pass |
| `frequency_low_36_month_persistence` | 200 | 36 | 0.430 | 0.246 | coverage failure |
| `persistent_36_month_independent` | 200 | 36 | 0.453 | 0.440 | coverage failure |
| `record_length_60_years_6_month_overlap` | 60 | 6 | -0.017 | 0.450 | coverage failure |
| `record_length_100_years_independent` | 100 | 1 | -0.012 | 0.451 | pass |

A passing 36-month case sits 0.011 below a failing 36-month case on autocorrelation, and
the two 60-year cases differ from the passing 100-year case on record length only.
Record length alone cannot help either, because the 200-year low-frequency scenario
fails. A fitted rule over persistence, frequency, and record length would be a new
decision surface with its own false refusals, and this evidence cannot predeclare it.
No guard was added.

## Decision (maintainer decision, 2026-10-19)

Recorded for review in this change; the merge confirms it.

1. **Keep the method unchanged.** No rescaling, no new resampling scheme, no new
dependency or kernel. The null test is not implicated: no independent case exceeded a
rejection band, and the 399-resample excess disappears at the default 1,999 resamples.
The remaining failures are interval under-coverage where few effective blocks or few
effective occurrences remain. That is a resolution limit of percentile intervals from
non-circular blocks, and it is the same limit the `few_blocks` flag already reports.
2. **Add no support guard.** The predeclared separation check fails, as the table above
shows. The library keeps returning results with flags and effective resample counts,
and users read the scope below.
3. **Widen the documented supported scope only where every predeclared gate passed.**
Now supported, beyond #1371, grouped by the resample count that measured each condition
because the plan crossed no two of them. At 399 resamples: 200-year records with source
persistence of 12 to 30 months; conditional lag windows `(0, 1)`, `(0, 3)`, and `(1, 3)`
at `w = 1` and `w = 6`; monthly exceedance probabilities of 0.30 to 0.60 at `w = 1` and
`w = 6` and 0.70 to 0.80 at `w = 1`, `w = 6`, and `w = 36`; interval coverage for hazards whose
same-time correlation is 0.5 at 1, 6, and 12 months of persistence; and 100-year records
with the seasonal profile at `w = 1` and `w = 6`. At the default 1,999 resamples:
200-year records with the seasonal profile (monthly exceedance probabilities 0.35 to
0.55) and source persistence of 1, 6, or 36 months, hazards independent. Profile `wide`
at `w = 36` was not run.
4. **Keep the unsupported list, with measured reasons.** 36-month persistence (one
correlated coverage failure at the default resamples, and the February coverage failure
at 399 resamples); monthly frequencies near 0.20 (suppressed null p-values, and coverage
misses when persistence is 36 months); 60-year records (coverage misses to 94 of 128 and
suppressed p-values in months 6 to 9); 30-year records (guards remove most records, and
the answered records fail both gates). Daily inference, field significance,
nonstationarity, and long memory remain outside every claim, as before.

`VALIDATION.md`, the API reference, the `bootstrap_dependence` docstring, and its
manifest `evidence` string carry this scope. Reopen the method question only with a
predeclared replacement characterization, as #1398 permits.

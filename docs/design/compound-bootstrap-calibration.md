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
  fails the null gate. The single exception is predeclared here: a month whose
  `quality_flags` carry a structural guard (`insufficient_years` 1,
  `insufficient_blocks` 2, or `sparse_occurrences` 4) is *not inferable* by design,
  and it is excluded from the gate and reported separately. `insufficient_successes`
  8 and `insufficient_expected_count` 16 are not structural exclusions; such a month
  counts as a failure. A case with fewer than twelve inferable months still reports
  its gate outcome together with the excluded-month count. Structural guards never
  fire at 200 years with marginal probabilities of 0.35 or more, so the replication
  gates and their outcome are unaffected by this rule.

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

_Pending: this section is written after the run._

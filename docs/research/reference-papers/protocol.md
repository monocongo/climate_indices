# Evaluation protocol for the reference-implementation studies (REFIMPL-02)

Issue: [REFIMPL-02](https://github.com/monocongo/climate_indices/issues/1604), part of the [reference-implementation map](https://github.com/monocongo/climate_indices/issues/1602).
Version: **1.0.0** (proposed; it takes effect when the maintainer approves it, see [Status](#status)).
Date: 2026-11-22.

This protocol says how two implementations of the same index are compared, which measures are reported, and which choices are fixed before any study runs.
It runs no comparison and reports no result.
It uses the evidence vocabulary of [`VALIDATION.md`](../../../VALIDATION.md): applying this protocol is not validation, and a result from it upgrades no status there until `VALIDATION.md` says so (REFIMPL-44).

## Status

| Version | Date | State |
| --- | --- | --- |
| 1.0.0 | 2026-11-22 | Proposed. Decisions 1 to 11 below were made by the maintainer in the REFIMPL-02 session on 2026-11-22; the document awaits the maintainer's review and approval of the written text. |

The maintainer owns one of the packages under comparison, so the protocol is written to be defensible against the charge of choosing criteria after the fact.
Every study cites the protocol version it was pre-registered against.

## Decisions

| # | Decision | Section |
| --- | --- | --- |
| 1 | No composite or ranking score. Each criterion is reported separately. | [Criteria](#criteria) |
| 2 | The primary category scale is the six USDM classes of `climate_indices.suitability`; the eleven-class scale of the Earth Engine pilot is a secondary report. One boundary rule applies to both. | [Category scales](#category-scales) |
| 3 | The ladder is L0 to L4 as in the ticket; the inverse-normal transform and the output bound belong to L4. | [Comparison ladder](#comparison-ladder) |
| 4 | A short ADR records that the protocol is versioned and that studies are pre-registered against it. | [ADR](#adr) |
| 5 | Difference classes use the ticket's thresholds, assigned on the maximum absolute difference per series. | [Difference classes](#difference-classes) |
| 6 | The unit of replication is the location (or the synthetic replicate), not the cell-month. Intervals come from a bootstrap over units. | [Units and sample size](#units-and-sample-size) |
| 7 | Statistical quality is measured by moments and a normal-quantile residual against a simulated null band, with no hypothesis test. | [C4](#c4-statistical-quality-of-the-output) |
| 8 | Pre-registration is a committed file plus an annotated tag; results cite the commit. | [Pre-registration](#pre-registration) |
| 9 | Adapter configurations are published before the run and maintainers are invited to review, but a study does not wait for a reply. | [Safeguards](#safeguards) |
| 10 | Every disagreement is entered in the REFIMPL-24 ledger; no result is dropped. | [Safeguards](#safeguards) |
| 11 | Criterion 2 reports signed error against the analytic index, overall and by tail band. | [C2](#c2-accuracy-against-known-truth) |

## Conventions

- **Index scale.** Every difference and threshold is in index units (standard-normal z). Comparisons are made after casting both sides to `float64`. A `float32` output is not re-rounded; it usually falls in the *below reporting precision* class because of its own rounding.
- **Series.** One location, one index, one scale, one configuration, one run. A *pair* is two series on the same time axis.
- **Finite pairs.** A time step enters a comparison only when both values are finite. The share of steps finite in exactly one series is reported as the *mask mismatch* and is never counted as agreement or as disagreement.
- **Missing and infinite values.** NaN, masked, and non-finite outputs are *not finite*. An infinite output is reported as such under C5 and excluded from every other measure.
- **Seeds.** Every random draw uses a seed written in the pre-registration. The generator is `numpy.random.default_rng(seed)`.
- **Percentiles.** Linear interpolation (`numpy.percentile` default, `method="linear"`).

## Criteria

Eight criteria, each reported on its own.
The ticket's proposed starting point is adopted, with the measures below.
No weight, rank, or overall score is computed, and no result may be presented as one.

The protocol fixes the *measures* and, for pair comparisons, the [difference-class](#difference-classes) thresholds.
The threshold a hypothesis is tested against (for example, "median `max|a - b|` over 100 divisions is at most 0.005") is set by each study in its [pre-registration](#pre-registration), before the run; C7 and C8 have no threshold.

### C1 Specification conformance

Does the package reproduce the equations and the published worked examples of the primary literature?

- **Input.** A worked example with printed inputs and printed outputs, with its source and page.
- **Measure.** For each printed output value `v` with `k` printed decimals, the example passes when `|computed - v| <= 0.5 * 10^-k`. Where the source prints rounded intermediate terms, the tolerance is stated in the pre-registration with the reason, and is fixed before the run.
- **Report.** Passes over printed values, and each failing value listed with its computed value and the source's. Not a score across examples.
- **Scope.** The examples are chosen by the study (REFIMPL-28 for Palmer; the REFIMPL-17 to REFIMPL-21 studies for the others).

### C2 Accuracy against known truth

Error against the analytic index on synthetic data drawn from a known distribution (REFIMPL-18, [#1620](https://github.com/monocongo/climate_indices/issues/1620)).

- **Truth.** For an observation `x` drawn from a data-generating distribution with cumulative distribution function `F`, the true index is `z* = Phi^-1(F(x))`, with `Phi^-1` the exact standard-normal quantile function.
- **Zero mass.** For a mixed distribution with probability `q` at zero, `F(x) = q + (1 - q) G(x)` for `x > 0`. The truth at `x = 0` is a convention, not a fact (the scored position is a method axis), so observations equal to zero are *excluded from C2* and their count is reported. Positive observations are scored.
- **Error.** `e = z - z*` for every finite scored observation.
- **Measures per series, then across replicates.** Signed bias `mean(e)`, `RMSE = sqrt(mean(e^2))`, and `max|e|`, each overall and within the bands of `|z*|`: `[0, 1)`, `[1, 2)`, `[2, infinity)`. A band with fewer than 30 observations in a replicate is reported as empty for that replicate.
- **Replicates.** Independent seeded draws. Across replicates report the median and the 2.5th and 97.5th percentiles of each measure (replicates are independent, so no bootstrap is needed).
- **Parameters of the sample.** Distribution family, parameters, record length, calendar grouping, and replicate count are in the pre-registration.
- **Report.** The truth is the index for the *generating* distribution, so a package that fits a different family carries that mismatch in its error. This is a stated limitation, not a defect to hide; the L2 rung separates it.

### C3 Agreement with authoritative products

Agreement with a product or a fixture under a matched configuration, for example NCEI climdiv SPI, nClimDiv PDSI, SPEIbase, NOAA EDDI.

- **Configuration.** The highest ladder rung the package and the product can both reach. The rung is reported with every number; a result at L0 is never described as an implementation difference.
- **Measures per series.** Bias `mean(a - b)` with `a` the package and `b` the product; mean absolute error; RMSE; Pearson `r`; `max|a - b|`; the [difference class](#difference-classes); same-class share and within-one-class share on each [category scale](#category-scales); mask mismatch.
- **Across units.** The median, 5th, and 95th percentile of each per-series measure, with the bootstrap interval of the median ([Units and sample size](#units-and-sample-size)).
- **Product limits.** A product that is a provider's output, with no source available, is reported as unverified for its method axes (`method-matrix.json` carries the status of each axis).

### C4 Statistical quality of the output

Calibration-period mean near 0, standard deviation near 1, and shape near normal, per calendar period.

- **Sample.** For each calendar period (month, week, or day of year, as the package groups it), the index values at the calibration years actually used, finite only. A period with fewer than 10 finite values is reported as *too short* and is not scored.
- **Moments.** Mean, sample standard deviation (`ddof=1`), skewness (Fisher-Pearson, population form `m3 / m2^1.5`), and excess kurtosis (`m4 / m2^2 - 3`).
- **Quantile residual.** Sort the `n` values as `s_(1) <= ... <= s_(n)`. The residual is `max_i |s_(i) - Phi^-1((i - 0.5) / n)|`. Tied values keep their sorted order; no tie adjustment is made.
- **Null band.** For each `n` that occurs, draw 10,000 samples of `n` standard-normal values (seed in the pre-registration), compute the same four statistics and the residual for each, and take the 2.5th and 97.5th percentiles as the band. The band is the reference; no p-value and no hypothesis test is reported.
- **Report.** For each statistic, the share of calendar periods outside the band (about 5% is expected from a calibrated and correct package) and its distribution across periods; and the share of values at an output bound, since clipping distorts every statistic above.
- **Why no test.** About 30 values per period give a normality test little power, and 12 to 366 periods per series would need a multiplicity rule; the band answers the question the criterion asks.
- **Scope.** The band is for a package whose calibration sample is the sample that was scored. For an index fitted on one window and applied to another, C4 is computed on the calibration window only.

### C5 Robustness

Behaviour on zeros, ties, gaps, short or constant records, and unfittable periods.

- **Outcome vocabulary.** Exactly one of:
  1. *correct*: finite output equal to the case's pre-registered expected output within the stated tolerance;
  2. *NaN with warning*: the affected values are NaN and a warning is emitted;
  3. *NaN silent*: the affected values are NaN and nothing is emitted;
  4. *error*: an exception;
  5. *silently wrong*: finite output that differs from the expected output, or any infinity, with no warning.
- **Expected output.** For every case it is written in the pre-registration before the run, with the reason (the specification, or the behaviour the package documents). A package whose documented behaviour is NaN with a warning is scored against *its own* documentation as well as against the specification, and both are reported.
- **Cases.** The case battery is REFIMPL-22's ([#1624](https://github.com/monocongo/climate_indices/issues/1624)): all-zero periods, constant series, tied values, interior and leading gaps, a record shorter than the package's minimum, and a period the fit cannot handle.
- **Report.** The count of each outcome per package, and the case-by-case table. Not a score, and *silently wrong* is listed individually.

### C6 Reproducibility and provenance

Same numbers across platforms, backends, and chunkings; recorded parameters and inputs.

- **Measure.** For each pair of runs (platform, backend, chunking, or thread count), the [difference class](#difference-classes) and `max|a - b|`.
- **Platforms and backends.** Those the study names; the library's NumPy, xarray, and Rust paths are separate runs.
- **Provenance.** Each result records the package version or commit, the dependency lock, the configuration (every argument), the input identifiers with checksums, the platform, and the seed. A run missing any of these is not reportable.

### C7 Performance and scalability

Wall time, peak resident memory, and throughput.

- **Measure.** Wall-clock seconds for the computation only (input read and output write timed separately), peak resident set size, and series per second, at the sizes the study pre-registers.
- **Repetition.** At least 5 runs per cell of the design, the first discarded as warm-up when a cold start is not the object of study. Report the median and the minimum and maximum.
- **Record.** Hardware (processor, core count, memory), operating system, thread settings, and whether the input was in memory or read lazily.
- **No threshold.** Performance is described, not judged against a limit.

### C8 Maintenance and usability signals

Reported descriptively and never scored: date of the latest release, the number of open issues at the pin, licence, how many steps install and run takes on a clean machine, and whether the documented example runs.

## Category scales

Two scales classify index values for the category measures.

- **Primary, six classes.** The U.S. Drought Monitor classes of `climate_indices.suitability.drought_category`: none (above -0.5), D0, D1, D2, D3, D4, with class edges -0.5, -0.8, -1.3, -1.6, -2.0.
- **Secondary, eleven classes.** The Earth Engine pilot's eleven drought-to-wet classes, with edges -2.0, -1.6, -1.3, -0.8, -0.5, 0.5, 0.8, 1.3, 1.6, 2.0.
- **Boundary rule, both scales.** Each class includes its upper edge and excludes its lower edge, so exactly -0.5 is D0, and exactly +0.5 is in the middle class (-0.5, +0.5] of the eleven-class scale. This is the rule of ADR-0028 and `suitability.py`.
- **The pilot differs.** `scripts/earthengine_climate_engine_pilot.py` (`usdm_bin`) includes the lower edge (`np.digitize` with `right=False`), so exactly -0.5 is "none" there. It changes the class only for a value exactly on an edge, which occurs in outputs rounded to two decimals (the NDMC program's). A study that compares with the pilot's published category numbers states which rule it used. This difference is logged in the REFIMPL-24 ledger.
- **Wet side.** The six-class scale puts every wet value in "none", so a change on the wet side is invisible there. The eleven-class report is where wet-side class changes appear.
- **Missing values** are never in any class; they are excluded before classification and counted in the mask mismatch.

## Difference classes

Pre-registered, mutually exclusive, and assigned in the order below to each *pair of series* on its finite pairs.
`d = max |a - b|` over finite pairs.

| Class | Rule |
| --- | --- |
| numerically identical | `d <= 1e-10` |
| equivalent | `1e-10 < d <= 1e-6` |
| below reporting precision | `1e-6 < d <= 0.005` |
| category-preserving | `d > 0.005` and every finite pair falls in the same class of the six-class scale |
| category-changing | `d > 0.005` and at least one finite pair falls in different classes of the six-class scale |

- The magnitude classes dominate: a pair with `d <= 0.005` stays in its magnitude class even if one value sits one rounding step across an edge.
- Report, for every pair, the class, `d`, the share of finite pairs that change class on the six-class scale and on the eleven-class scale, and the share of those changes that start or end in a drought class (D0 to D4).
- A pair with no finite pair in common has no class. It is reported as *no overlap*.
- 0.005 is the half-unit of a two-decimal output, so a package that rounds to two decimals can reach *below reporting precision* at best.
- Series-level classes are summarized across units as counts and shares per class, never as an average class.

## Comparison ladder

The ladder attributes a difference to a cause. At each rung one set of method axes (the twelve of [`method-matrix.json`](method-matrix.json)) is set to the same position in both packages, using each package's documented option.

| Rung | Aligned at this rung (cumulative) | Method axes |
| --- | --- | --- |
| L0 | as shipped: each package's defaults | none |
| L1 | inputs, calibration period, and accumulation window | accumulation, calendar grouping, calibration period |
| L2 | distribution family | distribution |
| L3 | estimator | estimator, location or offset |
| L4 | zero handling, plotting position, clipping, missing-data rules, and the inverse-normal transform | zero handling, plotting position, output bound, missing data and minimum sample, fit failure, inverse normal |

- **Cumulative.** Rung `k` keeps everything aligned at rungs below `k`.
- **Residual.** The difference after L4 is attributed to the implementation only for axes that *could* be aligned. An axis that a package does not expose stays at that package's own position; it is listed as *unalignable*, and the residual is reported as "residual with these axes unaligned", never as an implementation difference.
- **Order.** The effect of a rung is the change in the difference measures from the rung below. The order of the rungs is fixed here and effects are order-dependent, so effects are reported per rung and are not summed or described as shares of a total.
- **Axis positions.** Each aligned configuration cites `method-matrix.json` at a commit and the implementation ids of `lineage.json`. A position marked *unverified* in the matrix is verified in the study before it is used to claim alignment.
- **Inputs at L1.** Aligned inputs means the same array of values, after the same unit conversion, and the same calendar. Where a product is built from different inputs, L1 is not reachable and the study says so.
- **Why the distribution and the window come first.** Laimighofer and Laaha (2022, *Journal of Hydrology* 613, 128385) report, in the abstract, that distribution choice and observation window dominate SPI and SPEI uncertainty while the estimation method matters little. Only the abstract has been read in this project ([`research-notes.md`](research-notes.md)); the claim is a lead, not a verified finding, and the ladder does not depend on it.

## Units and sample size

Cell-months are not independent samples: they are autocorrelated in time (accumulation windows overlap) and correlated across neighbouring locations.

- **Unit of replication.** One *location* (a grid cell, a climate division, a station) for real data; one *replicate series* for synthetic data. Every measure is first computed *per unit*, then summarized across units.
- **What is reported.** The number of units `n_units`, never the number of cell-months. The summary across units is the median, the 5th, and the 95th percentile.
- **Interval.** A 95% percentile bootstrap interval of the across-unit median: resample units with replacement 10,000 times (seed in the pre-registration), take the median each time, and report the 2.5th and 97.5th percentiles.
- **Spatial dependence.** Neighbouring locations are correlated, so the interval understates uncertainty for gridded data. Every report that gives such an interval says so in the sentence that gives the interval; the interval is a measure of unit-to-unit variation, not a confidence statement.
- **Within a unit.** Where uncertainty of a single series' statistic is needed (for example a bias), use a moving-block bootstrap over time with block length `max(12 months, accumulation scale)` expressed in time steps (the block is 12 months of steps for monthly data, 365 days for daily data, and the scale if that is larger), 10,000 resamples.
- **No p-values on cell-months.** Hypotheses are tested by comparing a per-unit statistic (and its interval) with a threshold fixed in the pre-registration, not with a significance test over cell-months.
- **Sample.** The study fixes, before running: the unit population, the selection rule, the seed, and `n_units`, with the reason for that size. A selection rule may not use any value that depends on the outcome (for example, a difference observed in an earlier look). A study that looks at the data to choose units records that look as an amendment.

## Pre-registration

A study's pre-registration is a Markdown file committed before the study runs.
It names:

1. the protocol version;
2. the hypotheses, each with the measure, the direction, and the numerical threshold;
3. the sample (population, rule, seed, `n_units`) and every random seed;
4. the packages and pinned versions (`comparators.lock.yaml`), and each adapter configuration at a commit;
5. the ladder rungs to run and the axes to align at each;
6. exclusions and how they are decided;
7. the analysis code at a commit;
8. what is *exploratory* (anything not in 2).

Procedure:

1. Commit the pre-registration file.
2. Create an annotated tag `prereg/<ticket>-<n>` on that commit, with the date in the message.
3. Run the study. The results cite the tag and its commit hash.
4. Anything changed after the tag is an *amendment*: dated, listed in the result file with the reason, and marked *post hoc* wherever it affects a number. A threshold that is changed after any result has been seen is always an amendment.

The check is `git merge-base --is-ancestor <tag commit> <commit that adds the results>`; a result file without an ancestor pre-registration is not a result under this protocol.
Pushing a tag is a push to a shared resource and needs the maintainer's approval each time.

## Safeguards

- **Adapters.** Every adapter configuration (every argument passed to each package) is committed and tagged before the run. The comparator's maintainers are invited to review their adapter through REFIMPL-06 ([#1608](https://github.com/monocongo/climate_indices/issues/1608)); that issue decides whether and how. A study does not wait for a reply. It reports "no review received" or the review, and any later correction is an amendment.
- **The library is a comparator like the others.** `climate_indices` is run through the same adapter interface, with options set by the same ladder rule. No private argument, no unreleased branch, no code path that a user cannot reach.
- **Interest.** Every report says that the maintainer of `climate_indices` wrote the protocol and ran the studies (REFIMPL-07, [#1609](https://github.com/monocongo/climate_indices/issues/1609), decides the wording).
- **Disagreements.** Every disagreement between two packages that survives L4, every failed pre-registered hypothesis, and every C5 outcome other than *correct* gets an entry in the REFIMPL-24 ledger ([#1626](https://github.com/monocongo/climate_indices/issues/1626)) with a cause class and an owner, before the result is called final. Results that disfavour `climate_indices` are reported with the same prominence and in the same tables as those that favour it.
- **No silent re-runs.** A study is not re-run with changed thresholds, samples, or configurations without an amendment.
- **Passed and failed together.** Failed hypotheses appear in the same table as passed ones.
- **Unverified positions.** A package's behaviour is reported from a run, or from source read at the pin, and the report says which. A claim from documentation alone is marked unverified.

## ADR

This protocol binds future validation work, so [ADR-0029](../../adr/0029-reference-implementation-evaluation-protocol.md) records that decision (not the content of this document).

## What this protocol does not decide

| Open | Owner |
| --- | --- |
| The C5 case battery and expected outputs | REFIMPL-22, [#1624](https://github.com/monocongo/climate_indices/issues/1624) |
| The sizes and hardware of the C7 benchmark | REFIMPL-23, [#1625](https://github.com/monocongo/climate_indices/issues/1625) |
| Whether and how comparator maintainers are invited | REFIMPL-06, [#1608](https://github.com/monocongo/climate_indices/issues/1608) |
| The wording of the conflict-of-interest statement | REFIMPL-07, [#1609](https://github.com/monocongo/climate_indices/issues/1609) |
| Whether the harness lives here or in a sibling repository | REFIMPL-08, [#1610](https://github.com/monocongo/climate_indices/issues/1610) |
| The metric implementations | REFIMPL-16, [#1618](https://github.com/monocongo/climate_indices/issues/1618) |

## Measure reference

For REFIMPL-16 ([#1618](https://github.com/monocongo/climate_indices/issues/1618)).
Every measure is defined above; this table says what each takes and returns.

| Measure | Input | Output | Unit | Defined in |
| --- | --- | --- | --- | --- |
| printed-value pass | computed value, printed value, printed decimals | boolean | none | C1 |
| signed bias, RMSE, max abs error, by band | series, truth series, band edges | three numbers per band | index units | C2 |
| bias, MAE, RMSE, `r`, max abs difference | a pair | five numbers | index units, `r` dimensionless | C3 |
| difference class | a pair | one of five classes, `d` | index units | Difference classes |
| class-change share | a pair, a category scale | share in [0, 1] | none | Category scales |
| mask mismatch | a pair | share in [0, 1] | none | Conventions |
| calendar-period moments | one calendar period's values | mean, sd, skewness, excess kurtosis | index units; skewness and kurtosis dimensionless | C4 |
| quantile residual | one calendar period's values | one number | index units | C4 |
| null band | `n`, seed | two limits per statistic | as the statistic | C4 |
| outside-band share | per-period statistics, band | share in [0, 1] | none | C4 |
| robustness outcome | a case, the run's output, warnings | one of five outcomes | none | C5 |
| across-unit summary | per-unit values | median, 5th, 95th percentile | as the measure | Units and sample size |
| bootstrap interval of a median | per-unit values, seed | two limits | as the measure | Units and sample size |
| block bootstrap interval | one series' statistic, seed | two limits | as the statistic | Units and sample size |
| performance | repeated timings, peak memory | median, minimum, maximum | s, bytes, series per s | C7 |

## Changelog

| Version | Date | Change |
| --- | --- | --- |
| 1.0.0 | 2026-11-22 | First version. Adopts the ticket's starting point with the decisions above; the inverse-normal axis added to L4; one boundary rule for both category scales. |

Versioning: a change to a threshold, a class, or the definition of a measure is a *major* change and does not apply to a study already pre-registered; a new measure or a new rung is *minor*; wording and links are *patch*.

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
| 1.0.0 | 2026-11-22 | Proposed. Decisions 1 to 11 below were made by the maintainer in the REFIMPL-02 session on 2026-11-22; the written text awaits the maintainer's review and approval. |

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

- **Index scale.** Differences and thresholds are in index units. For SPI, SPEI, EDDI and the other normal-scale indices these are standard-normal z. C2, the category scales, and the category-based difference classes apply only to normal-scale indices. For another index (for example PDSI) the study fixes class edges in its pre-registration or reports the category measures as *not applicable*; the magnitude classes still apply.
- **Inputs to a measure.** One series, or a pair, as 1-D float arrays along time, of equal length (a length mismatch raises). The harness loops over units. Integer and boolean input is cast to `float64`; object arrays are rejected. Outputs are unrounded `float64`; rounding is for display only. *Pooled* pairs (several units flattened into one pair) are allowed only to reproduce a published figure, such as the Earth Engine pilot's, and are labelled pooled.
- **Series.** One location, one index, one scale, one configuration, one run. A *pair* is two series on the same time axis.
- **Float width.** Differences are computed in `float64` after casting. Output rounding of a `float32` series is at most about 1.2e-7 for |z| below 4, which is *equivalent* (not *below reporting precision*); its computation error is usually larger. Category codes are assigned to each series at its own width, with the edges cast to that width as `suitability.drought_category` does, and the codes are then compared.
- **Tolerance comparison.** Every comparison with a stated bound is `value <= bound * (1 + 1e-9)`, so that `abs(1.255 - 1.26)`, which is 0.005000000000000115 in `float64`, stays within 0.005.
- **Finite pairs.** A time step enters a comparison only when both values are finite. The share of steps finite in exactly one series is the *mask mismatch*; it is reported beside every pair class and is never counted as agreement or disagreement.
- **Missing and infinite values.** NaN, masked, and non-finite outputs are *not finite*. An infinite output is reported under C5 and excluded from every other measure, except C1, where an infinite computed value is a failure.
- **Undefined values.** A per-unit measure that is undefined (no finite pair, fewer than 2 pairs or zero variance for `r`, a share with a zero denominator, an empty band) is NaN. It is excluded from that measure's summary and bootstrap, and `n_units_defined` is reported beside `n_units`. With fewer than 2 defined units the interval is NaN.
- **Random streams.** Every stochastic measure draws from `numpy.random.Generator(PCG64(SeedSequence([seed, stream, key])))`, with `seed` the pre-registered seed and `(stream, key)`: 1 null band, key `n`; 2 unit bootstrap, key the measure's row number in the [Measure reference](#measure-reference); 3 block bootstrap, key the unit index; 4 synthetic data, key the replicate index. The NumPy version is recorded; streams are not guaranteed across NumPy versions.
- **Percentiles.** Linear interpolation (`numpy.percentile` default, `method="linear"`).

## Criteria

Eight criteria, each reported on its own.
The ticket's proposed starting point is adopted, with the measures below.
No weight, rank, or overall score is computed, and no result may be presented as one.

The protocol fixes the *measures* and, for pair comparisons, the [difference-class](#difference-classes) thresholds.
The threshold a hypothesis is tested against (for example, "median `max|a - b|` over 100 divisions is at most 0.005") is set by each study in its [pre-registration](#pre-registration), before the run; C7 and C8 have no threshold.

### C1 Specification conformance

Does the package reproduce the equations and the published worked examples of the primary literature?

- **Input.** A worked example with printed inputs and printed outputs, with its source and page. Each printed value comes with an integer `k`, the number of printed decimals (trailing zeros count).
- **Measure.** A printed value `v` passes when `|computed - v| <= 0.5 * 10^-k`. A NaN or infinite computed value fails. Where the source prints rounded intermediate terms, the study may supply a different tolerance; it is stated in the pre-registration with the reason and fixed before the run.
- **Report.** Passes over printed values, and each failing value listed with its computed value and the source's. Not a score across examples.
- **Scope.** The examples are chosen by the study (REFIMPL-28 for Palmer; the REFIMPL-17 to REFIMPL-21 studies for the others).

### C2 Accuracy against known truth

Error against the analytic index on synthetic data drawn from a known distribution (REFIMPL-18, [#1620](https://github.com/monocongo/climate_indices/issues/1620)).

- **Truth.** For an observation `x` drawn from a data-generating distribution with cumulative distribution function `F`, the true index is `z* = Phi^-1(F(x))`, computed with `scipy.special.ndtri`. A `z*` that is not finite (`F` equal to 0 or 1 in `float64`) is excluded and counted.
- **Zero mass.** For a mixed distribution with probability `q` at zero, `F(x) = q + (1 - q) G(x)` for `x > 0`, with `G` the cumulative distribution function of the positive component (`G(0) = 0`). The truth at `x = 0` is a convention, not a fact (the scored position is a method axis), so observations equal to zero are *excluded from C2* and their count is reported.
- **Error.** `e = z - z*` for every finite scored observation.
- **Measures per replicate.** Signed bias `mean(e)`, `RMSE = sqrt(mean(e^2))`, and `max|e|`, each overall and within the bands of `|z*|`: `[0, 1)`, `[1, 2)`, `[2, infinity)`. A band with fewer than 30 observations in a replicate is empty for that replicate.
- **Replicates.** Independent seeded draws. Across replicates report the median and the 2.5th and 97.5th percentiles of each measure, taken over the non-empty replicates, with the number of non-empty replicates. This is the one exception to the 5th and 95th percentiles and the bootstrap of [Units and sample size](#units-and-sample-size): replicates are independent, so no bootstrap is needed, and the 2.5th to 97.5th range is a spread, not a confidence interval.
- **Parameters of the sample.** Distribution family, parameters, record length, calendar grouping, and replicate count are in the pre-registration.
- **Limitation.** The truth is the index for the *generating* distribution, so a package that fits a different family carries that mismatch in its error. The L2 rung separates it.

### C3 Agreement with authoritative products

Agreement with a product or a fixture under a matched configuration, for example NCEI climdiv SPI, nClimDiv PDSI, SPEIbase, NOAA EDDI.

- **Configuration.** The highest ladder rung the package and the product can both reach. The rung is reported with every number; a result at L0 is never described as an implementation difference.
- **Measures per series.** With `a` the package and `b` the product: bias `mean(a - b)`, mean absolute error, RMSE, Pearson `r`, `max|a - b|`; the [difference class](#difference-classes); the same-class share and the within-one-class share on each [category scale](#category-scales); the class confusion matrix (counts, rows the product, columns the package); the mask mismatch.
- **Across units.** The median, 5th, and 95th percentile of each per-series measure, with the bootstrap interval of the median ([Units and sample size](#units-and-sample-size)).
- **Product limits.** A product that is a provider's output, with no source available, is reported as unverified for its method axes (`method-matrix.json` carries the status of each axis).

### C4 Statistical quality of the output

Calibration-period mean near 0, standard deviation near 1, and shape near normal, per calendar period.

- **Inputs from the adapter.** With each run the adapter returns `calibration_years` (the years the package actually used) and an integer `period_label` for every time step (month, ISO week, day of year, or the package's own grouping). Both are provenance fields (C6). A package that does not expose its effective window is run with the configured window, and its C4 is marked *unverified*.
- **Sample.** For each period label, the finite index values at the calibration years. A period with fewer than 10 finite values is *too short* and is not scored.
- **Moments.** With `m_k = mean((x - mean(x))^k)`: the mean; the standard deviation with `ddof=1` (this line only); skewness `m3 / m2^1.5`; excess kurtosis `m4 / m2^2 - 3`. A constant period has `m2 = 0`, and its skewness and kurtosis are NaN (the period is not scored for them).
- **Quantile residual.** Sort the `n` values as `s_(1) <= ... <= s_(n)`. The residual is `max_i |s_(i) - Phi^-1((i - 0.5) / n)|`. Tied values keep their sorted order; no tie adjustment is made.
- **Null band.** For each `n` that occurs, draw 10,000 samples of `n` standard-normal values (stream 1), compute the same statistics, and take the 2.5th and 97.5th percentiles. A moment is *outside* when it is below the lower or above the upper limit. The residual is *outside* when it is above its 97.5th percentile only: it is non-negative and a small value is not a defect. No p-value and no hypothesis test is reported.
- **Zero mass in the band.** A zero-handling rule that scores a probability mass `p0` at one value moves the output away from standard normal even when the fit is correct (ADR-0015): `p0` of the values take the rule's zero score and the rest lie above `Phi^-1(p0)`. The null band therefore uses the same rule as the run: `p0` of each null sample takes that zero score, and the remaining draws are `Phi^-1(p0 + (1 - p0) * u)` with `u` uniform on (0, 1). With `p0 = 0` this is the standard-normal band above. Clipping at `output_bound` and a trace rule that scores small values as zero can still move a value out of the band; both are reported with the result (`p0`, `zero_handling`, and the share of values at the zero score).
- **Report.** For each statistic, the share of scored periods outside the band and its distribution across periods. The band comes from independent draws, while the scored sample is the one the package fitted to, which constrains its mean and spread more tightly; the band is therefore conservative, and a share of *at most about* 5% (2.5% for the residual) is expected from a package that is calibrated and correct and configured with the zero rule the band uses. Also report the share of values at the output bound, where a value is at the bound when `abs(z) >=` the package's recorded `output_bound`, since clipping distorts every statistic above.
- **Why no test.** About 30 values per period give a normality test little power, and 12 to 366 periods per series would need a multiplicity rule; the band answers the question the criterion asks.
- **Scope.** C4 is computed on the calibration years only, also for an index fitted on one window and applied to another.

### C5 Robustness

Behaviour on zeros, ties, gaps, short or constant records, and unfittable periods.

- **Case.** `{input, expected, affected_mask, tolerance}`. `expected` is written in the pre-registration before the run, with the reason (the specification, or the behaviour the package documents); it may be NaN. A package whose documented behaviour is NaN with a warning is scored against its own documentation as well as against the specification, and both are reported.
- **Warning.** A `warnings.warn` call or a logging record at level WARNING or above, captured in the calling process.
- **Outcome vocabulary.** Judged on the values at `affected_mask`; exactly one of:
  1. *correct*: finite values equal to `expected` within the tolerance, or NaN where NaN is expected, with no warning required;
  2. *NaN with warning*: NaN where a finite value is expected, and a warning is emitted;
  3. *NaN silent*: NaN where a finite value is expected, and nothing is emitted;
  4. *error*: an exception, recorded as *clear* when the message names the cause and *unclear* when it does not;
  5. *wrong with warning*: finite values that differ from `expected` (or an infinity), and a warning is emitted;
  6. *silently wrong*: finite values that differ from `expected` (or an infinity), and nothing is emitted.
- **Mixed cases.** The warning is a property of the case; the value outcome is a property of each value at `affected_mask`. A case whose values disagree takes the most severe outcome present, in this order: *error*, *silently wrong*, *wrong with warning*, *NaN silent*, *NaN with warning*, *correct*. A silent failure therefore outranks the warned form of the same failure. The case table lists the value outcome of each affected value for a mixed case.
- **Cases.** REFIMPL-11 ([#1613](https://github.com/monocongo/climate_indices/issues/1613)) generates the stress inputs and REFIMPL-22 ([#1624](https://github.com/monocongo/climate_indices/issues/1624)) runs them: all-zero and constant periods, fewer values than the estimator needs, exact ties, missing values inside the accumulation window, a Calibration Period outside or shorter than the record, values beyond the calibration range, and `float32` input, among others.
- **Report.** The count of each outcome per package, and the case-by-case table. Not a score, and each *silently wrong* case is listed individually.

### C6 Reproducibility and provenance

Same numbers across platforms, backends, and chunkings; recorded parameters and inputs.

- **Measure.** For each pair of runs (platform, backend, chunking, or thread count), the [difference class](#difference-classes), `max|a - b|`, and the mask mismatch.
- **Platforms and backends.** Those the study names; the library's NumPy, xarray, and Rust paths are separate runs.
- **Provenance.** Each result records the package version or commit, the dependency lock, the configuration (every argument), the input identifiers with checksums, the platform, the seed, `calibration_years`, `period_label`, and `output_bound`. A run missing any of these is not reportable.

### C7 Performance and scalability

Wall time, peak resident memory, and throughput.

- **Measure.** Wall-clock seconds for the computation only (input read and output write timed separately); peak resident set size in bytes (converting `ru_maxrss` for the platform); `series_per_s = n_series / wall_s` per run, and also cell-months per second where the input is a grid, as REFIMPL-23 ([#1625](https://github.com/monocongo/climate_indices/issues/1625)) reports.
- **Process per memory measurement.** `ru_maxrss` is the highest value over a process's lifetime, so each timed run takes its peak in a fresh child process, from that process's own value. A peak measured in a shared process, or inherited from a warm-up or an earlier comparator, is not reportable.
- **Repetition.** At least 3 timed runs after the warm-up call (a study may require more); report the median, minimum, and maximum. A cold start is timed separately when it is the object of study.
- **Record.** Hardware (processor, core count, memory), operating system, thread settings, and whether the input was in memory or read lazily.
- **No threshold.** Performance is described, not judged against a limit.

### C8 Maintenance and usability signals

Reported descriptively and never scored: date of the latest release, the number of open issues at the pin, licence, how many steps install and run takes on a clean machine, and whether the documented example runs.

## Category scales

Two scales classify normal-scale index values for the category measures.

- **Primary, six classes.** The U.S. Drought Monitor classes of `climate_indices.suitability.drought_category`: none (above -0.5), D0, D1, D2, D3, D4, with class edges -0.5, -0.8, -1.3, -1.6, -2.0.
- **Secondary, eleven classes.** The Earth Engine pilot's eleven drought-to-wet classes, with edges -2.0, -1.6, -1.3, -0.8, -0.5, 0.5, 0.8, 1.3, 1.6, 2.0. The five classes below -0.5 are its drought classes.
- **Boundary rule, both scales.** Each class includes its upper edge and excludes its lower edge, so exactly -0.5 is D0. This is the rule of ADR-0028 and `suitability.py` for the drought-side edges. The rule for the wet-side edges of the eleven-class scale is this protocol's extension, which the library does not implement: exactly +0.5 is in the middle class (-0.5, +0.5].
- **Parameter.** A measure function takes the boundary rule as an argument, `upper_inclusive` (the protocol's) or `lower_inclusive`, so that a published figure computed with the other rule can be reproduced.
- **The pilot differs.** `scripts/earthengine_climate_engine_pilot.py` (`usdm_bin`) includes the lower edge (`np.digitize` with `right=False`), so every edge differs: exactly -0.5 is in the middle (near-normal) class there, and exactly +0.5 is in the class [0.5, 0.8). It changes the class only for a value exactly on an edge, which occurs in outputs rounded to two decimals (the NDMC program's). A study that compares with the pilot's published category numbers states which rule it used. This difference is logged in the REFIMPL-24 ledger.
- **Wet side.** The six-class scale puts every wet value in "none", so a change on the wet side is invisible there. The eleven-class report is where wet-side class changes appear.
- **Class codes.** Classes are ordered integers. *Within one class* is `|code_a - code_b| <= 1`. Missing values are never in any class; they are excluded before classification and counted in the mask mismatch.

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

A pair with no finite pair in common has no class; it is reported as *no overlap*.

A study whose index is not normal-scale (PDSI, for example) replaces the six-class scale in the last two rules with the class edges it fixes in its pre-registration, and the two rules then read against those edges.
When no class edges apply, the category measures are *not applicable* and a pair with `d > 0.005` is *above reporting precision*: that class carries no category content and the report says so, so the pair never lands in a class it does not have.

- The magnitude classes dominate: a pair with `d <= 0.005` stays in its magnitude class even if one value sits one rounding step across an edge.
- Report, for every pair, the class, `d`, the mask mismatch, the share of finite pairs that change class on the six-class scale and on the eleven-class scale (NaN when there is no finite pair), and the share of those changes that start or end in a drought class (D0 to D4 on the six-class scale; the five classes below -0.5 on the eleven-class scale).
- 0.005 is the half-unit of a two-decimal output, so a package that rounds to two decimals can reach *below reporting precision* at best.
- Series-level classes are summarized across units as counts and shares per class, never as an average class.

## Comparison ladder

The ladder attributes a difference to a cause. At each rung one set of method axes (the twelve of [`method-matrix.json`](method-matrix.json), named there by the ids in the last column) is set to the same position in both packages, using each package's documented option.

| Rung | Aligned at this rung (cumulative) | Method axes | Axis ids |
| --- | --- | --- | --- |
| L0 | as shipped: each package's defaults | none | none |
| L1 | inputs, calendar grouping, calibration period, and accumulation window | accumulation, calendar grouping, calibration period | `accumulation`, `calendar_grouping`, `calibration_period` |
| L2 | distribution family | distribution | `distribution` |
| L3 | estimator | estimator, location or offset | `estimator`, `location_offset` |
| L4 | zero handling, plotting position, output bound, missing-data rules, fit failure, and the inverse-normal transform | zero handling, plotting position, output bound, missing data and minimum sample, fit failure, inverse normal | `zero_handling`, `plotting_position`, `output_bound`, `missing_data_minimum_sample`, `fit_failure`, `inverse_normal` |

- **Cumulative.** Rung `k` keeps everything aligned at rungs below `k`.
- **Residual.** The difference after L4 is attributed to the implementation only for axes that *could* be aligned. An axis that a package does not expose stays at that package's own position; it is listed as *unalignable*, and the residual is reported as "residual with these axes unaligned", never as an implementation difference.
- **Attribution.** The effect of rung `k` on a measure `m` is, per unit, `m(L_k) - m(L_{k-1})`, computed on the finite pairs common to every rung of that unit, and then summarized across units. The order of the rungs is fixed here and effects are order-dependent, so effects are reported per rung and are not summed or described as shares of a total.
- **Axis positions.** Each aligned configuration cites `method-matrix.json` at a commit and the implementation ids of `lineage.json`. A position marked *unverified* in the matrix is verified in the study before it is used to claim alignment.
- **Inputs at L1.** Aligned inputs means the same array of values, after the same unit conversion, and the same calendar. Where a product is built from different inputs, L1 is not reachable and the study says so.
- **Why the distribution and the window come first.** Laimighofer and Laaha (2022, *Journal of Hydrology* 613, 128385) report, in the abstract, that distribution choice and observation window dominate SPI and SPEI uncertainty while the estimation method matters little. [`research-notes.md`](research-notes.md) records only the abstract as the source; the claim is a lead, not a verified finding, and the ladder does not depend on it.

## Units and sample size

Cell-months are not independent samples: they are autocorrelated in time (accumulation windows overlap) and correlated across neighbouring locations.

- **Unit of replication.** One *location* (a grid cell, a climate division, a station) for real data; one *replicate series* for synthetic data. Every measure is first computed *per unit*, then summarized across units (C2 is the one exception, see there).
- **What is reported.** `n_units` and `n_units_defined`, never the number of cell-months. The summary across units is the median, the 5th, and the 95th percentile.
- **Interval.** A 95% percentile bootstrap interval of the across-unit median: draw `integers(0, n_units_defined, size=(10000, n_units_defined))` from stream 2, take the median of each resample, and report the 2.5th and 97.5th percentiles.
- **Spatial dependence.** Neighbouring locations are correlated, so the interval understates uncertainty for gridded data. Every report that gives such an interval says so in the sentence that gives the interval; the interval is a measure of unit-to-unit variation, not a confidence statement.
- **Within a unit.** Where uncertainty of a single series' statistic is needed (for example a bias), use a moving-block bootstrap over time (stream 3): blocks of length `L` that overlap, do not wrap, number `ceil(n / L)`, and are cut to `n`; for a pair the same time indices are resampled in both series; 10,000 resamples. `L` is the number of steps in 12 months (12 monthly, 52 weekly, 365 daily), or the accumulation scale in steps if that is larger.
- **No p-values on cell-months.** Hypotheses are tested by comparing a per-unit statistic (and its interval) with a threshold fixed in the pre-registration, not with a significance test over cell-months.
- **Sample.** The study fixes, before running: the unit population, the selection rule, the seed, and `n_units`, with the reason for that size. A selection rule may not use any value that depends on the outcome (for example, a difference observed in an earlier look). A study that looks at the data to choose units records that look as an amendment.

## Pre-registration

A study's pre-registration is a Markdown file committed before the study runs.
It names:

1. the protocol version;
2. the hypotheses, each with the measure, the direction, and the numerical threshold;
3. the sample (population, rule, seed, `n_units`);
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

- **Adapters.** Every adapter configuration (every argument passed to each package) is committed and tagged before the run. Every comparator's maintainers are contacted once (REFIMPL-06, [#1608](https://github.com/monocongo/climate_indices/issues/1608), sent by the maintainer), and that issue decides how and when. A study does not wait for a reply: it reports "no review received" or the review, and any later correction is an amendment.
- **The library is a comparator like the others.** `climate_indices` is run through the same adapter interface, with options set by the same ladder rule. No private argument, no unreleased branch, no code path that a user cannot reach.
- **Interest.** Every report says that the maintainer of `climate_indices` wrote the protocol and ran the studies (REFIMPL-07, [#1609](https://github.com/monocongo/climate_indices/issues/1609), decides the wording).
- **Disagreements.** Every disagreement between two packages that survives L4, every failed pre-registered hypothesis, and every C5 outcome other than *correct* gets an entry in the REFIMPL-24 ledger ([#1626](https://github.com/monocongo/climate_indices/issues/1626)) with a cause class and an owner, before the result is called final. Results that disfavour `climate_indices` are reported with the same prominence and in the same tables as those that favour it.
- **No silent re-runs.** A study is not re-run with changed thresholds, samples, or configurations without an amendment.
- **Passed and failed together.** Failed hypotheses appear in the same table as passed ones.
- **Unverified positions.** A package's behaviour is reported from a run, or from source read at the pin, and the report says which. A claim from documentation alone is marked unverified.

## ADR

This protocol binds future validation work, so [ADR-0029](../../adr/0029-reference-implementation-evaluation-protocol.md) records that decision (not the content of this document).

## Relation to the list in REFIMPL-16

[#1618](https://github.com/monocongo/climate_indices/issues/1618) was written before this protocol. Where they differ, the protocol governs, and the issue text is to be updated once the maintainer approves the protocol.

| #1618 lists | This protocol |
| --- | --- |
| Rank correlation; tail agreement on the lowest and highest deciles; event agreement (onset, duration, severity) | Not adopted in 1.0.0. A study that needs one proposes it as a minor version of the protocol. |
| Category agreement on the eleven-class scale, D0 to D4 thresholds separately | The six-class scale is primary and the eleven-class scale secondary ([Category scales](#category-scales)); the confusion matrix is in C3. |
| A normality-test rejection rate | Replaced by the moments and quantile residual against a null band, with no hypothesis test (decision 7, C4). |
| A cluster bootstrap over locations | Replaced by the bootstrap over units ([Units and sample size](#units-and-sample-size)). |
| Reproduce the pilot's bias, MAE, RMSE, correlation, and class-agreement figures | Pooled pairs and the `lower_inclusive` boundary rule ([Conventions](#conventions), [Category scales](#category-scales)). |

## What this protocol does not decide

| Open | Owner |
| --- | --- |
| The C5 stress inputs and expected outputs | REFIMPL-11, [#1613](https://github.com/monocongo/climate_indices/issues/1613), and REFIMPL-22, [#1624](https://github.com/monocongo/climate_indices/issues/1624) |
| The sizes and hardware of the C7 benchmark | REFIMPL-23, [#1625](https://github.com/monocongo/climate_indices/issues/1625) |
| How and when comparator maintainers are contacted | REFIMPL-06, [#1608](https://github.com/monocongo/climate_indices/issues/1608) |
| The wording of the conflict-of-interest statement | REFIMPL-07, [#1609](https://github.com/monocongo/climate_indices/issues/1609) |
| Whether the harness lives here or in a sibling repository | REFIMPL-08, [#1610](https://github.com/monocongo/climate_indices/issues/1610) |
| The metric implementations | REFIMPL-16, [#1618](https://github.com/monocongo/climate_indices/issues/1618) |

## Measure reference

For REFIMPL-16 ([#1618](https://github.com/monocongo/climate_indices/issues/1618)).
Every measure is defined above; this table says what each takes and returns.
Row numbers are the stream-2 keys.

| # | Measure | Input | Output | Unit | Defined in |
| --- | --- | --- | --- | --- | --- |
| 1 | printed-value pass | computed value, printed value, `k`, optional tolerance | boolean | none | C1 |
| 2 | error, by band | series, truth series, band edges | bias, RMSE, max abs error per band | index units | C2 |
| 3 | replicate summary | per-replicate measures | median, 2.5th and 97.5th percentile, non-empty count | as the measure | C2 |
| 4 | pair agreement | a pair | bias, MAE, RMSE, `r`, max abs difference | index units, `r` dimensionless | C3 |
| 5 | difference class | a pair | one of five classes or *no overlap*, `d` | index units | Difference classes |
| 6 | class-change share | a pair, a category scale, boundary rule | share in [0, 1] or NaN; share of changes touching a drought class | none | Difference classes |
| 7 | same-class and within-one share | a pair, a category scale, boundary rule | two shares in [0, 1] or NaN | none | C3, Category scales |
| 8 | confusion matrix | a pair, a category scale, boundary rule | integer matrix | counts | C3 |
| 9 | mask mismatch | a pair | share in [0, 1] | none | Conventions |
| 10 | calendar-period moments | one period's values | mean, sd, skewness, excess kurtosis | index units; the last two dimensionless | C4 |
| 11 | quantile residual | one period's values | one number | index units | C4 |
| 12 | null band | `n`, seed | two limits per statistic | as the statistic | C4 |
| 13 | outside-band share | per-period statistics, band | share in [0, 1] | none | C4 |
| 14 | output-bound share | a series, `output_bound` | share in [0, 1] | none | C4 |
| 15 | robustness outcome | a case, the run's output, warnings, exception | one of six outcomes; for *error*, clear or unclear | none | C5 |
| 16 | across-unit summary | per-unit values | median, 5th, 95th percentile, `n_units`, `n_units_defined` | as the measure | Units and sample size |
| 17 | bootstrap interval of a median | per-unit values, seed | two limits | as the measure | Units and sample size |
| 18 | block bootstrap interval | a series (or pair), a statistic, block length, seed | two limits | as the statistic | Units and sample size |
| 19 | rung effect | per-unit measures at two rungs | per-unit difference, then summary | as the measure | Comparison ladder |
| 20 | performance | repeated timings, peak memory, `n_series`, cell-months | median, minimum, maximum of seconds, bytes, `series_per_s` | s, bytes, series per s | C7 |

## Changelog

| Version | Date | Change |
| --- | --- | --- |
| 1.0.0 | 2026-11-22 | First version. Adopts the ticket's starting point with the decisions above; the inverse-normal axis added to L4; one boundary rule for both category scales. |

Versioning: a change to a threshold, a class, or the definition of a measure is a *major* change and does not apply to a study already pre-registered; a new measure or a new rung is *minor*; wording and links are *patch*.

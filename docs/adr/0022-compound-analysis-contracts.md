# Compound-analysis contracts

## Status

Amended by #1061: the 3.1.0 promotion in ADR-0012 covers DataArray index APIs,
not compound Dataset routes or eager event catalogs. These routes remain Beta
and continue to emit the public `BetaFeatureWarning`. Their contracts are unchanged.

Amended by CMP-10 (#1374) to record the completed guide and scoped public-study reproduction,
and by [#1398](https://github.com/monocongo/climate_indices/issues/1398) to record the
bootstrap calibration decision.
The accepted API contracts remain unchanged. CMP-09 (#1373) completed the described API implementations.
The maintainer approved these contracts and the bounded milestone scope.
[CMP-01 (#1365)](https://github.com/monocongo/climate_indices/issues/1365) records this decision under
[parent issue #1364](https://github.com/monocongo/climate_indices/issues/1364).

**The NumPy `harmonize()`, `joint_statistics()`, `conditional_probability()`,
`compound_events()`, and `bootstrap_dependence()` routes
([#1367](https://github.com/monocongo/climate_indices/issues/1367),
[#1368](https://github.com/monocongo/climate_indices/issues/1368),
[#1369](https://github.com/monocongo/climate_indices/issues/1369),
[#1370](https://github.com/monocongo/climate_indices/issues/1370),
[#1371](https://github.com/monocongo/climate_indices/issues/1371)) and the beta xarray Dataset routes
for `harmonize()`, `joint_statistics()`, `conditional_probability()`, and `bootstrap_dependence()`
([#1372](https://github.com/monocongo/climate_indices/issues/1372)), and the eager xarray event
catalog ([#1373](https://github.com/monocongo/climate_indices/issues/1373)) are implemented.**
The xarray routes are equivalence-tested against the NumPy core, not independently
validated. Bootstrap evidence is synthetic characterization and regression only.
The maintainer narrowed its supported scope (#1371) to the synthetic cases that passed every
predeclared gate, and then measured further conditions in #1398. The supported scope now covers,
grouped by the resample count that measured each condition because the plan crossed no two of
them: at the default 1,999 resamples, 200-year records with the seasonal monthly probabilities
(0.35 to 0.55), source persistence of 1, 6, or 36 months, and hazards independent of each other;
at 399 resamples, 200-year records with 1 to 30 months of source persistence, conditional lag
windows to `(1, 3)`, monthly probabilities of 0.30 to 0.60 or 0.70 to 0.80, hazards correlating
0.5 at the same time with 1, 6, or 12 months of source persistence, and 100-year records.
36-month persistence at 399 resamples for independent hazards, correlated 36-month persistence at
the default resamples, monthly probabilities near 0.20, and 60-year and 30-year records failed
predeclared gates, so the intervals and p-values are not claimed there.
See `VALIDATION.md` and the amendment below.
The described APIs exist. This status does not claim completed scientific validation;
the [guide](../compound-hazards.md) and nine summer cases in CMP-10 (#1374) establish scoped same-author reference reproduction only.
The implementation responsibilities are explicit:

| Proposed public function | NumPy implementation | Beta xarray implementation | Current state |
| --- | --- | --- | --- |
| `harmonize()` | [#1367](https://github.com/monocongo/climate_indices/issues/1367) | [#1372](https://github.com/monocongo/climate_indices/issues/1372) | NumPy implemented (#1367); beta xarray route implemented (#1372). |
| `joint_statistics()` | [#1368](https://github.com/monocongo/climate_indices/issues/1368) | [#1372](https://github.com/monocongo/climate_indices/issues/1372) | NumPy implemented (#1368); beta xarray route implemented (#1372). |
| `conditional_probability()` | [#1369](https://github.com/monocongo/climate_indices/issues/1369) | [#1372](https://github.com/monocongo/climate_indices/issues/1372) | NumPy implemented (#1369); beta xarray route implemented (#1372). |
| `compound_events()` | [#1370](https://github.com/monocongo/climate_indices/issues/1370) | The separate catalog route below uses this computation. | NumPy implemented (#1370); the eager xarray catalog route is implemented (#1373). |
| `bootstrap_dependence()` | [#1371](https://github.com/monocongo/climate_indices/issues/1371) | [#1372](https://github.com/monocongo/climate_indices/issues/1372) | NumPy implemented (#1371); beta xarray route implemented (#1372). |
| `compound_events_xarray()` | It reuses the NumPy event matcher. | [#1373](https://github.com/monocongo/climate_indices/issues/1373) | Eager numeric catalog implemented (#1373). |

The reference fixture is [#1366](https://github.com/monocongo/climate_indices/issues/1366).
The scoped reproduction, user guide, and evidence classification are recorded in [#1374](https://github.com/monocongo/climate_indices/issues/1374) and `VALIDATION.md`.
The implementation table records the shipped API scope; evidence classification remains separate.

## Context

Existing indices provide fitted cumulative probabilities, run identification,
fit diagnostics, labeled calendars, and explicit fire/flood recurrence state.
Researchers still assemble alignment, joint-frequency, lag, and uncertainty calculations separately.
Those calculations can disagree because their calendars, denominator support, and event definitions differ.

The [milestone plan](https://github.com/monocongo/climate_indices/blob/main/docs/design/compound-module-milestone-1.md)
contains the API sketches, validation design, verified scientific sources, issue specifications, and approved recommendations.
It remains internal because it describes unimplemented work.
This record accepts the contracts that would be difficult to change after release.

## Decision

### 1. The module is namespaced and has two API tiers

Use `climate_indices.compound`, with a public facade over private implementation modules.
Do not add compound names or the module to the package root's `__all__`.
Follow the explicit `runs` import pattern rather than adding unqualified root functions.
The first real function creates the package; an empty scaffold is not a deliverable.

The NumPy core is the stable numerical contract.
The xarray Dataset routes are beta under ADR-0012 and use typed overloads and `BetaFeatureWarning`.
All computation comes from the NumPy core; adapters own labels, metadata, and Dask orchestration.
This adds a family-level placement exception to ADR-0001, as the fire and flood records did.
It does not alter drought, fire, flood, or native-backend behavior.

### 2. Continuous inputs declare probability scale; physical hazards can declare Boolean indicators

Every source declares `input_scale="probability"` or `"indicator"`.
Probability values are finite real numbers in `[0, 1]`, with NaNs and masks representing unknown observations.
Reject infinity, invalid types, out-of-range values, and conflicting source metadata without clipping.
A numerical range check does not prove a valid probability integral transform or a good distribution fit.

Require a threshold and explicit `"above"` or `"below"` direction per probability input.
Use strict comparisons, as `runs.identify_runs()` does. Never infer direction from a name, sign, or index family.
Keep discrete ties and zero-mass placement unchanged.
Use observed marginal frequencies rather than assuming that a threshold equals its exceedance probability.

A Boolean indicator source declares threshold `0.5` and direction `"above"`.
A separate validity mask distinguishes an unknown day from a false indicator.
Its source manifest records the original physical threshold and comparator, including inclusive comparators when applicable.
This permits KBDI, FWI, heat thresholds, and published Boolean hazard series without inventing PIT values.

Reject normal-score and bounded-score declarations in this milestone.
Do not convert z-scores internally; clipped scores cannot recover the fitted probability tails.
Users request `output_scale="probability"` upstream when that route exists.

### 3. Harmonization preserves observed opportunities

Require a declared cadence per source and explicit NumPy dates or real xarray datetime coordinates.
Accept Gregorian `datetime64` calendars only, following ADR-0004's supported calendar names.
Reject unsupported calendars, duplicates, NaT, unordered coordinates, and unlabeled positional time.
Normalize monthly labels to their calendar month; daily labels represent civil days.
Do not regrid, spatially broadcast, or infer missing dates as observations.

Compound analysis counts observed dates and does not fold data into 366-day positional years.
Partial leading and trailing years are allowed because no fitting occurs here.
This is a scoped analysis-layer exception to ADR-0004's January-origin fitting requirement.
It does not change the existing conversion seam.
Restore legacy all-leap NumPy output upstream; synthesized February 29 values must not enter observed counts.

Default alignment is exact. An explicit time-range intersection reports dropped boundaries.
Reindex interior absent dates to invalid positions; never compress unknown gaps.
Require matching spatial coordinates and reject an empty overlap.

To combine daily and monthly hazards, threshold daily inputs first.
Reduce daily occurrence indicators to calendar months with an explicit per-source `any` or `all` rule.
Require every expected day to be valid, including real leap days.
A partial or incomplete month is invalid, even when its observed part contains an exceedance.
Do not average CDF values, interpolate monthly scores to daily values, or call a reduced indicator a PIT value.
For monthly accumulated climate indices, aggregate physical inputs and fit upstream instead.

Reject known conflicting Calibration Periods.
Record and warn about unknown source fit metadata rather than inventing compatibility.
Different source Timescales are allowed and recorded.

### 4. All reported statistics expose their counts and support

`joint_statistics()` handles two or three selected hazards on one complete-case mask.
It returns marginal counts/frequencies, joint count/frequency, the independence product, and LMF.
Frequency means occupied time steps, not independent events per year.
A three-input independence product assumes mutual independence.

LMF is `observed_joint_frequency / product(observed_marginal_frequencies)`.
It cannot be negative. Zero joint frequency gives zero LMF when all marginals are positive.
A zero marginal makes LMF undefined; return NaN and a quality flag, not infinity or a pseudocount estimate.
LMF equal to one does not establish independence of the underlying variables.

`conditional_probability()` is pairwise and names its `given` and `target` sources explicitly.
The target event is any occurrence in an inclusive forward cadence-step window.
An eligible anchor has an observed given value and every requested target step observed.
Count each positive given anchor once, even when its window contains multiple target occurrences.
Incomplete horizons never count, including horizons that already contain a target occurrence.
The comparison baseline is the target-window frequency on those same eligible anchors.
Group by the anchor month. Daily lags mean civil days; monthly lags mean calendar months.
This is time-step conditioning, not event-onset probability or causation.

Default reporting is per calendar month. Explicit pooling recounts all eligible observations;
it does not average monthly LMFs. Day-of-year grouping is deferred.
NaNs remain unknown, and all-missing cells do not fail neighboring cells.
Default `min_samples=30` applies per reported group; shorter support returns counts, NaNs, and quality flags.
Warn about records shorter than 30 observed years and sparse marginal/conditioning occurrences.
These are product guardrails, not universal adequacy guarantees.

### 5. Results remain numeric and serializable

Use frozen result dataclasses for the NumPy route: `AlignedHazards`, `JointResult`,
`ConditionalResult`, `CompoundEventSet`, and `BootstrapResult`.
Copy result arrays, make them read-only, and retain immutable choice metadata.
Do not change the existing `RunSet` class.
Use xarray Datasets for beta statistics results, with numeric counts, flags, probabilities, and named dimensions.

Event matching accepts two `RunSet`s on one complete cadence grid with explicit observation validity and declared cadence.
Accept source choices as separate metadata because `RunSet` does not retain threshold, direction, or minimum-duration declarations.
Record missing source choices as unknown. Never remove an unknown observation's position.
Concurrent matches contain every intersecting pair and report its inclusive intersection.
Sequential matches require `B.start > A.end`; their lag is `B.start - A.end`, so adjacency has lag one.
Match every pair within the requested window and reject bridges through unknown observations.
Retain source-run IDs and boundary-censor flags. Do not combine unrelated source magnitudes into a severity score.

Gridded catalogs use numeric `event` rows and a cell-coordinate lookup table.
They do not contain Python `RunSet` objects or claim an unsupported CF feature type.
They are explicitly eager in milestone 1; reject Dask-backed variables and coordinates without computing them.
The catalog identifies both sources' runs on their shared validity mask, with threshold 0.5,
direction `"above"`, no gap pooling, and the declared `min_duration`.
Source-run IDs are local to each cell's retained runs after duration filtering.
The manifest retains the prepared source declarations; per-cell variables retain support and matching counts.
Three-hazard event matching and distributed ragged catalogs are deferred.

### 6. Monthly inference separates uncertainty from the independence null

Bootstrap inference is opt-in and monthly-only in milestone 1.
Daily descriptive statistics and events remain available, but a daily inference request raises.
A later daily resampling design must settle leap-day and seasonal-phase behavior first.

Use non-circular moving blocks of consecutive complete calendar years.
Default block length is `ceil(Y ** (1/3))` years, with an explicit integer override.
This is a transparent heuristic, not an optimal autocorrelation estimate.
Use the longest contiguous complete inference segment and report it separately from the full descriptive record.
Choose the earliest segment when lengths tie.

For confidence intervals, resample all channels with the same block indices.
For the independence null, resample channels independently while preserving calendar-month phase.
Build conditional response-window indicators on the original chronology before resampling.
Never create response windows across artificial block joins.

Default to 1,999 realizations per distribution and 95% percentile intervals using linear quantiles.
Require an explicit integer seed and local NumPy random generation.
Results must not depend on spatial tile boundaries or worker scheduling.
Record dependency versions; do not promise seeded bitwise identity across NumPy versions.

The joint null statistic is `p_joint - product(p_i)`.
The conditional null statistic is `P(response_window|given) - P(response_window)`.
Default to the one-sided amplification alternative; permit explicit suppression.
Use inclusive tail comparisons and the finite-resample correction `(1 + extreme_count) / (1 + valid_resamples)`.
Do not break ties with random noise or interpret a paired interval as an independence test.

Require 30 complete inference years and five nonoverlapping blocks.
Within each reported month or pooled support, require 10 positive occurrences per marginal channel, or per A and response-window channel.
Require five observed joint/conditional successes for a percentile interval.
Require five expected successes under independence for a null test, using `N*product(p_i)` or `n_A*P(response_window)`.
An observed zero joint count can support suppression inference without receiving a misleading zero-width interval.
Require at least 95% valid realizations per metric.
Warn below 10 nonoverlapping blocks. Return NaN plus flags when the relevant support is inadequate.
Predeclare block-length sensitivity checks. Do not choose the block length by significance.
Validate these defaults through synthetic coverage and null-error characterization before exposing significance.
Grid p-values are local and unadjusted; field significance remains out of scope.

### 7. Execution and audit choices follow existing infrastructure

Require single chunks along `time` and any stacked `hazard` core dimension.
Use `validation.validate_dask_chunks()` and never rechunk or evaluate a grid automatically.
Spatial chunks must remain compatible across inputs.
Use a dedicated shape-reducing adapter; the fitting decorator's calendar folding is inappropriate here.
Validate coordinates and options before graph construction; validate lazy data values inside numerical tasks.
Value-dependent failures can therefore appear when computation starts.
The NumPy support warnings (short record, sparse occurrences, a group below `min_samples`, a zero marginal,
and the bootstrap's limited-support warning)
are value-dependent, so they follow the same rule: in-memory xarray input warns exactly like the NumPy route,
while Dask input leaves those conditions to `quality_flags`, because a warning raised inside a task would
repeat per tile with tile-local counts. Descriptive flags retain `insufficient_support=1` and
`undefined_denominator=2`. They also report `sparse_occurrences=4` for fewer than ten marginal or conditioning
occurrences on nonempty support, and `short_record=8` for fewer than 30 observed calendar years per cell.
The short-record flag uses each cell's eligible steps across the whole record, not each month's group.
These additional bits do not change the counts, estimates, or `min_samples` rule.
Harmonization warnings depend only on dates and declarations, so both backends raise them at the call.
Lifecycle completion records `evaluation="graph_construction"` for a lazy call and `evaluation="eager"` otherwise.
Do not allocate resample-by-time-by-cell arrays.
Measure memory before publishing tile-size guidance.

Register the primary compound probabilities and LMF in the CF registry before their adapters ship.
Use dimensionless units and verified references. Omit unregistered `standard_name`s.
Keep audit/count/catalog variable attributes beside the adapter, following fit diagnostics.
Use the existing output-history builder, structlog lifecycle events, structured exceptions, and applicable warning categories.
Insufficient statistical support is a numeric quality condition, not a distribution-fitting exception.

Record every source declaration, threshold, direction, cadence, aggregation, source Timescale,
Calibration Period, fit/zero/gap policy, missingness rule, lag, support count, inference choice, seed,
version, and known unknown needed to interpret the result.
Use a versioned JSON choice manifest and scalar CF-compatible attributes; do not put source arrays in attributes.

Add no dependency and no compound native kernel.
Earthflux remains a downstream consumer with no import, service, licensing, or configuration dependency from the library.
Its needs are stable names, numeric serialization, CF-style metadata, and auditable counts and choices.

## Amendment: bootstrap calibration scope (#1398)

[Issue #1398](https://github.com/monocongo/climate_indices/issues/1398) measured the
conditions #1371 left open: record lengths from 30 years, persistence between 6 and 36
months, conditional lag windows, hazard frequencies outside 0.35 to 0.55, hazards that
depend on each other at short persistence, and the default 1,999 resamples.
`docs/design/compound-bootstrap-calibration.md` records the predeclared gates, the
results, and the decision. `docs/design/compound-bootstrap-characterization.json` holds
all 183 measured cases.

The decision leaves the design of section 6 unchanged. The evidence:

- No independent-hazard case exceeded its per-month null rejection band. The 7.5%
  pooled rejection that #1371 recorded for the 36-month case at 399 resamples falls to
  3.5% at the default 1,999 resamples, so the null test needs no change.
- The remaining failures are interval under-coverage, which appears where few effective
  blocks or few effective occurrences remain: 30- and 60-year records, 36-month
  persistence, and monthly probabilities near 0.20. Percentile intervals from
  non-circular blocks carry that resolution limit, and `few_blocks` already reports part
  of it.
- No support guard was added. A guard is justified only if a quantity observable from
  the function's own inputs separates the passing from the failing cases without
  refusing a supported case, and no such quantity exists here: a passing 36-month case
  sits 0.011 below a failing one on within-month lag-1 autocorrelation, and the failing
  60-year cases show no persistence to detect. A fitted multi-variable rule would need
  its own predeclared characterization.
- The supported scope widened only where every predeclared gate passed, group by group
  rather than as one configuration. At 399 resamples: 100-year records, persistence to
  30 months, lagged conditioning, monthly probabilities of 0.30 to 0.60 or 0.70 to 0.80,
  and correlated hazards at 1, 6, and 12 months of persistence. At the default 1,999
  resamples: 200-year records with the seasonal probabilities and 1, 6, or 36 months of
  persistence, hazards independent. The plan crossed no two of those conditions.

The guards of section 6, the default of 1,999 realizations, and the requirement to
validate defaults through synthetic characterization stand. A later method change for
36-month persistence or short records would need its own predeclared characterization.

## Alternatives considered

- **Copula-first scope:** Deferred. Empirical support and calendar contracts need evidence before adding multivariate fitting assumptions.
- **Automatic z-score conversion:** Rejected. Clipping and zero handling can make an inverse conversion scientifically false.
- **Probability-only inputs:** Rejected as the sole path. Existing physical fire/heat thresholds can become explicit Boolean indicators without an unsupported fit.
- **Monthly-to-daily interpolation:** Rejected. It invents daily resolution and changes the observation unit.
- **Aggregating probabilities:** Rejected. A sum, mean, or extremum of PIT values is not the fitted monthly CDF.
- **Separate frequency and LMF wrappers:** Rejected. One result keeps counts and denominators together without redundant computation.
- **Datasets in the stable numerical core:** Rejected. The NumPy contract should not depend on the beta adapter surface.
- **Object-valued lazy catalogs:** Deferred. Numeric serialization and bounded variable-length execution require a separate contract.
- **IID resampling or confidence-interval-only significance:** Rejected. Neither supplies the proposed persistence-aware independence test.
- **Daily inference immediately:** Deferred. The first milestone can publish useful descriptive daily analysis without claiming an unresolved daily uncertainty method.
- **New bootstrap/copula dependencies or Rust kernels:** Deferred. Existing dependencies can implement and validate this bounded milestone.

## Consequences

The first implementation is narrower than a general compound-risk framework.
Conservative complete-month and inference-segment policies can exclude useful sparse observations;
outputs expose those exclusions instead of silently relaxing them.
Year blocks preserve within-block dependence but do not guarantee correct inference under long memory or nonstationarity.
Study-specific review remains necessary.

The [parent issue's children](https://github.com/monocongo/climate_indices/issues/1364)
separate contract publication, reference fixtures, harmonization, statistics, events, inference, adapters, and final evidence.
CMP-01 through CMP-10 provide the implementations, public-input reproduction, and evidence classifications.
The reproduction covers three Swedish cells and three summer months, not the full map or its significance mask.
The #1398 amendment records the measured bootstrap calibration scope; a later method
change for 36-month persistence or short records needs its own predeclared
characterization.
Release scheduling and milestone closure remain maintainer decisions.
Synthetic tests, author-method reproduction, and independent external validation remain distinct evidence tiers.

## Scientific source boundary

The verified reproduction target is Vieira Passos, M., Kan, J.-C., Destouni, G., Barquet, K.,
and Kalantari, Z. (2024), *Identifying regional hotspots of heatwaves, droughts, floods, and their co-occurrences*.
*Stochastic Environmental Research and Risk Assessment* 38, 3875–3893.
<https://doi.org/10.1007/s00477-024-02783-3>.

The public inputs are Vieira Passos, M. (2024), *Climate Indices for heat waves, droughts and floods
in Sweden (1922-2021)*, version 2, Figshare, CC BY 4.0.
<https://doi.org/10.6084/m9.figshare.25189118.v2>.

The paper and the pinned author notebook support the empirical LMF reproduction target.
They do not validate this record's new monthly block-bootstrap defaults.
The plan records the verified citations, public file inventory, source-code pin, convention differences,
required Fixture Provenance, and remaining retrieval work.

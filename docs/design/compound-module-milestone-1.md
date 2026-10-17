# Compound module: milestone 1

## Status and implementation gate

The maintainer approved this internal plan and its recommendations.
[Parent issue #1364](https://github.com/monocongo/climate_indices/issues/1364) tracks the milestone.
No compound API or scientific validation exists yet.
Implementation remains blocked until the maintainer merges CMP-01's ADR PR.

The source review used fetched `main` at `a074b1fc5593c5761abd0c49b361654b241cc479`
and the 3.0.0 API contracts. The checkout supplied with the request was older.
The fetched branch contains ADRs through 0021. The accepted contract uses number 0022,
which remained available when CMP-01 preparation started.

**Repository conventions take precedence in two places.**

- This plan lives under `docs/design/`, not a new `docs/plans/` directory.
  `CONTRIBUTING.md` classifies design notes as internal, and `docs/conf.py` excludes this directory.
- Published ADRs currently permit `Accepted`, `Amended`, and `Superseded`, but not `Proposed`.
  The proposal stayed internal during the initial review.
  After approval, CMP-01 promoted it to [ADR-0022](../adr/0022-compound-analysis-contracts.md) with status `Accepted`.
  That record distinguishes accepted contracts from unimplemented functions, as ADR-0019 does.
  No authoring-convention change is required.

No `_bmad-output/` or repository `.claude/skills/` directory was present.
The planning handoffs describe an epic, dependency-ordered child issues, human decision gates,
and oracle-first fixtures. Older handoffs name abandoned ADRs and branches; current source takes precedence.
The available `drive-issue-to-pr` skill was read. Its generic `fix/issue-N` branch suggestion does not
supersede this repository's `docs/<topic>`, `feature/<topic>`, and `test/<topic>` conventions.

The maintainer authorized issue creation and preparation of the CMP-01 documentation PR.
The parent and ten child issues now exist with native parent and dependency links.
No implementation issue may start before the ADR PR merges.
No project board, GitHub milestone, or release target was created. The delivery date remains unassigned.

## Goal and publishable scope

Provide reproducible empirical relationships between two or three index-defined hazards.
Researchers are the first users. Outputs describe index relationships, not observed impacts,
flood occurrence, causal mechanisms, return periods, or operational trigger performance.

| Area | Milestone 1 recommendation | Deferred alternative and reason |
| --- | --- | --- |
| Inputs | Accept declared probability-scale values and explicitly declared Boolean hazard indicators. | Do not accept raw continuous indices or convert z-scores internally. Their distributions and clipping histories differ. |
| Harmonization | Validate Gregorian dates; align equal cadences; aggregate daily **indicators** to calendar months with explicit `any` or `all`. | Do not interpolate monthly values to days or average fitted CDF values. Those operations change the question. |
| Joint statistics | Report joint counts, eligible counts, marginal frequencies, joint frequency, and likelihood multiplication factor (LMF) for two or three hazards. | Do not fit copulas or derive return periods from occupied time steps. |
| Conditional statistics | Report pairwise contemporaneous and forward-window conditional probabilities. | Do not infer causality, estimate all lag combinations automatically, or add multivariate conditioning. |
| Events | Match pairs of `RunSet`s concurrently or sequentially. Provide numeric catalogs for one series and explicitly eager grid subsets. | Do not add three-hazard event matching or lazy, variable-length grid catalogs yet. |
| Inference | Provide paired block-bootstrap intervals and independent-channel block-bootstrap null tests for **monthly** statistics. | Defer daily inference. A daily resampling contract must address leap days, seasonality, and unequal year lengths first. |
| Execution | Provide a stable NumPy core and beta xarray statistics adapters, with explicit whole-time chunks. | Do not add native kernels, a CLI, a pipeline framework, or streaming compound state. |
| Evidence | Require analytic tests, stochastic characterization, and a scoped published-method reproduction before milestone completion. | Do not advertise independent external validation from synthetic tests or a same-author notebook. |

This revises the starting hypothesis in three material ways: monthly-only inference,
pairwise event catalogs, and eager-only gridded catalogs. Daily descriptive statistics remain included.
Boolean inputs make published physical thresholds and existing fire indices usable without pretending
that KBDI, FWI, or a heatwave indicator already has a fitted probability scale.

Later work includes parametric copulas, tail dependence, multivariate standardized indices,
field significance and false-discovery-rate control, daily inference, Rust bootstrap kernels,
ensemble disagreement, and generalized event graphs.

## Existing contracts and integration points

| Existing source | Verified contract | Consequence for compound work |
| --- | --- | --- |
| `runs.py` | `RunSet` has parallel arrays. Ends are inclusive. Comparisons are strict. NaNs split runs. | Reuse `identify_runs`; do not invent another run scanner or sum unrelated magnitudes. |
| `runs.identify_runs_xarray` | One-dimensional input evaluates eagerly. Grids contain object-valued `RunSet`s and can rechunk time automatically. | Do not copy this storage or chunking behavior into serializable compound catalogs. Leave the existing function unchanged. |
| `indices.py`, `compute.py` | SPI, SPEI, and NumPy `standardized_index()` expose fitted probabilities without normal-score clipping. Zero-mass conventions affect ties and endpoints. | Request `output_scale="probability"` upstream. Never reconstruct these values from clipped scores. |
| `fit_diagnostics()` | The NumPy result is `FitDiagnostics`; the root xarray route returns a Dataset. Gamma/Pearson diagnostics do not cover log-logistic SPEI. | Preserve available fit evidence; do not require diagnostics that the source API cannot supply. |
| ADR-0004 | Fitting kernels use positional all-leap years; adapters return Gregorian coordinates. The fitting route requires a January origin. | Count observed Gregorian dates, not synthesized all-leap slots. Compound analysis does not fit or fold years. |
| ADR-0009 | Spatial Block interpretation must be explicit at the NumPy fitting boundary. | New compound arrays use only `(time, *cells)`, explicitly declared for non-1-D inputs. Never reuse legacy `(years, periods)` guessing. |
| ADR-0012 | Xarray remains beta unless a separate decision promotes it. | Emit `BetaFeatureWarning`; promise no patch-level interface breaks on that route. |
| `cf_metadata_registry.py` | The registry is a leaf dependency. Outputs omit unregistered `standard_name`s. Audit-only variables can define attributes beside an adapter. | Register primary compound statistics before their adapters ship. Keep count, validity, and catalog attributes beside the adapter. |
| `exceptions.py` | Shape, argument, coordinate, input-type, alignment, and beta categories already exist. `InsufficientDataError` is a fitting exception. | Reuse applicable errors. Represent insufficient statistical support with NaN plus quality flags, not a misleading fitting error. |
| Fire/flood adapters | Shared validators reject split time chunks. CFFWIS/API carry explicit recurrence state. Returning state can evaluate some outputs. | Consume their completed index arrays and validity policies. Compound analysis adds no recurrence state and never silently evaluates a grid. |
| ADRs 0019 and 0020 | Generic xarray `standardized_index()` wiring remains assigned to #1174 on the reviewed branch. | Do not make compound depend on that unfinished route. Use NumPy standardization or an externally prepared probability array. |

All ADRs 0001–0021 were reviewed. Existing science, calendar conversion, recursive state,
root exports, and optional-native dispatch remain unchanged.

## Recommended decisions and alternatives

### 1. Namespace and API tiers

Use a `climate_indices.compound` package with a public facade and private implementation modules.
Import it explicitly with `import climate_indices.compound as compound`.
Do not add functions or the module to root `__all__`, following `runs` rather than the fire root-module export.
Do not add entries to `typed_public_api.py` or algorithms to drought-oriented `compute.py`.
The accepted ADR will grant this family-level placement exception to ADR-0001.

Start the package with its first working function, not an empty scaffold.
Use `_harmonize.py`, `_statistics.py`, `_events.py`, `_bootstrap.py`, and `_xarray.py`
only as their responsible issues land. Result types stay beside their owning computation.
A flat module would be smaller initially, but the distinct inference and catalog adapters already justify these boundaries.

### 2. Probability values and Boolean indicators

Require an `input_scale` declaration for every input: `"probability"` or `"indicator"`.
For probability inputs, accept real numeric values in `[0, 1]`, NaNs, and masked values.
Reject infinity, complex/object/string/datetime values, and finite out-of-range values.
Do not clip invalid data. Reject conflicting xarray `output_scale` metadata.

Require a finite threshold in `[0, 1]` and an explicit `"above"` or `"below"` direction per probability input.
Use strict `>` and `<`, including at endpoints. Do not randomize ties in a discrete zero mass.
A declared probability input does not prove uniform PIT values or a good fit.
LMF therefore uses **observed** marginals, never `threshold` or `1 - threshold` as marginal probabilities.

An indicator input must have Boolean dtype and a separate validity mask where needed.
Its public threshold declaration is `0.5`, and its direction declaration is `"above"`.
The manifest records the original physical threshold and comparator separately.
This explicit path accepts a published inclusive comparator without changing the strict probability-input contract.

Reject `input_scale="normal"` and `"bounded"` in milestone 1.
A z-score inside `[0, 1]` cannot be detected numerically; the required declaration is a trust boundary.
Document explicit upstream conversion only for a genuinely untruncated standard-normal variable.
`norm.cdf(clipped_spi)` cannot recover the original tail, and ADR-0015's `mean_zero` mode adds another mismatch.
Request upstream probabilities instead.

### 3. Harmonization and calendars

Every NumPy source has an explicit date vector and declared cadence, `"daily"` or `"monthly"`.
Xarray also requires a declared cadence and uses its real time coordinate; dimension-only time is rejected.
Accept only `datetime64` and the Gregorian calendar names already supported by ADR-0004.
Reject cftime, 360-day, noleap, timezone-bearing, duplicate, unordered, and NaT coordinates.
Do not convert between calendar systems.

Monthly dates identify calendar months; normalize month-start/month-end labels to the month's start.
Reject duplicate observations within a month. Daily dates identify individual civil days.
Allow partial first and last years because no positional fitting occurs here.
This is a scoped extension, not a change to the January-origin fitting API.
Restore legacy all-leap NumPy index output upstream before passing observed dates.
Synthetic February 29 values must never count as observations.

`join="exact"` is the default. It requires matching normalized time coverage and matching spatial coordinates.
`join="inner"` explicitly restricts the time range to the common overlap and emits `InputAlignmentWarning` for dropped boundaries.
Within that range, reindex each source to the complete declared cadence and mark absent observations invalid.
Never remove interior missing dates and close the gap. An empty overlap is an error.
Spatial subsetting, broadcasting a point across a grid, and regridding are never implicit.

Daily-to-monthly reduction happens **after thresholding each daily value**.
Require `aggregation[name]="any"` or `"all"` for each reduced input.
Require all expected civil days to be valid, including real February 29 where applicable.
A partial month or one missing day makes that input's month invalid, even if an observed day exceeds the threshold.
Already-monthly sources remain unchanged. Reject irrelevant reducers and monthly-to-daily targets.

`any` asks whether a month contains at least one hazard day; `all` asks whether every day meets the condition.
Neither is a monthly PIT value. The prepared result therefore contains hazard masks, not re-labelled probability values.
Mean/sum/fraction reducers and score interpolation remain out of scope.
If a study needs monthly accumulated climate values, aggregate the physical inputs and refit upstream.

Known, conflicting Calibration Periods are rejected. Missing source fit metadata emits `ClimateIndicesWarning`
and records `unknown`; it does not prevent descriptive analysis or silently claim calibration compatibility.
Different Timescales remain allowed and recorded because their relationship can be the study question.
Compound analysis does not calibrate or refit anything.

### 4. Result objects and event catalogs

Return frozen NumPy dataclasses with copied arrays and immutable metadata mappings.
Make new result arrays read-only; freezing a dataclass alone does not freeze its arrays.
Do not retrofit `RunSet` immutability or introduce a persistence framework.

| Type | Required content |
| --- | --- |
| `AlignedHazards` | The ordered source names, observed time vector, cadence, Boolean exceedance and validity arrays, and harmonization/source manifest. |
| `JointResult` | `n_valid`, `n_joint`, per-source `n_marginal` and marginal frequencies, `joint_probability`, `independent_probability`, `likelihood_multiplication_factor`, quality flags, and metadata. |
| `ConditionalResult` | Eligible anchors, eligible A-positive anchors, successful A-positive anchors, `conditional_probability`, unconditional response-window frequency on the same anchors, quality flags, and metadata. |
| `CompoundEventSet` | Source run IDs, onset/end indices, overlap duration, gap/lag steps, boundary-censor flags, and metadata. Arrays have stable empty dtypes. |
| `BootstrapResult` | The inference-support point estimate, interval bounds, null p-value, effective resample counts, selected block length, inference window, quality flags, and metadata. |

Statistics return an xarray Dataset on the beta route, including for a single location.
Use a `month=1..12` dimension by default, a `hazard` dimension for marginal values,
and the original non-time dimensions for grids. Pooled results omit `month`.
Counts remain int64; undefined estimates remain NaN; quality flags remain numeric.
Do not hide counts or choices inside a single formatted string.

A one-series event result has one row per matched source-run pair.
The xarray catalog has an `event` dimension, source run IDs, numeric cell indices,
and onset/end timestamps. A separate `cell` coordinate table retains spatial labels.
Do not use Python objects or claim a CF discrete-sampling-geometry feature type without a matching convention.
NetCDF/Zarr round-trip tests must preserve numeric values, coordinates, flags, and JSON metadata.

Grid catalogs are explicitly eager. Reject Dask-backed catalog input and instruct the caller to select
and compute a bounded subset first. Do not silently call `.compute()` or `.load()`.
A separate lazy ragged-array design can follow demonstrated demand.

### 5. Estimands, missingness, and lag windows

For selected source hazards `H_i(t)`, use one complete-case eligibility mask:

```text
V(t) = all selected sources are valid at t
N = sum(V)
n_i = sum(V and H_i)
n_joint = sum(V and all H_i)
p_i = n_i / N
p_joint = n_joint / N
p_independent = product(p_i)
LMF = p_joint / p_independent
```

`joint_statistics()` computes all these outputs once. Separate functions for frequency,
joint exceedance, and LMF would repeat the same counts and increase denominator drift.
Frequency means occupied eligible time steps, not the number of independent events per year.
For three inputs, the denominator assumes mutual independence, not only pairwise independence.

`LMF=1` means the selected threshold frequencies match the independence product.
It does not prove independence of the underlying variables. LMF is nonnegative and can exceed one without a finite universal bound.
If any marginal count is zero, retain the counts and joint probability, but return NaN for LMF.
If marginals are positive and the joint count is zero, return `LMF=0`, not NaN or infinity.
Do not add pseudocounts.

For `conditional_probability()`, select exactly two named sources, `given=A` and `target=B`.
For each anchor `t`, define `R(t)` as any B occurrence in the inclusive forward window `[t+lo, t+hi]`.
`(0, 0)` means contemporaneous conditioning; `(1, 3)` means the next one through three cadence steps.
Daily steps are civil days; monthly steps are calendar months, not fixed 30-day intervals.
Reject negative lags and unordered bounds.

An anchor is eligible only if A is observed and **every** B step in the requested window is observed.
Discard incomplete trailing windows, including a window that already contains a B occurrence.
Do not require A to be observed at future times or B outside the requested window.
Return `sum(eligible and A and R) / sum(eligible and A)`.
Count each A-positive time step once, regardless of B multiplicity.
This is a time-step conditional probability, not an event-onset conversion probability.
The unconditional comparison is `P(R)` over those same eligible anchors, not single-step `P(B)`.
Group a lagged statistic by the anchor's month, even when its response window crosses a month or year boundary.

NaNs and masked values remain unknown, never false. Keep time gaps in place.
All-missing cells return zero counts, NaN estimates, and quality flags without failing neighboring cells.
Require `min_samples=30` eligible observations per reported group; allow an explicit positive override for experiments.
Below that limit, return counts and NaN estimates. A zero conditioning count also returns NaN.
Warn about fewer than 30 observed years and fewer than 10 marginal/conditioning occurrences.
These are conservative product policies, not universal scientific adequacy thresholds.
Do not reuse calibration preflight logic as a test of compound statistical support.

### 6. Concurrent and sequential events

`compound_events()` accepts two existing `RunSet`s, one shared time vector, its declared cadence, and one observation-validity mask.
Require a complete cadence grid; unknown observations retain their positions and have validity false.
Validate equal field lengths, sorted nonoverlapping runs, bounds, positive durations, and `duration=end-start+1`.
Reject an asserted run that crosses an invalid observation. Keep source-run IDs after filtering.
Accept source metadata separately because `RunSet` does not retain thresholds, direction, or `min_duration`.
Record unavailable source choices as unknown rather than reconstructing them from run metrics.

For concurrent matching, include every pair whose inclusive intervals overlap.
The reported interval is their intersection, and `overlap_duration` counts its steps.
Do not merge adjacent pair records or discard pairs because another run already matched.

For sequential matching, require `B.start > A.end` and define `lag=B.start-A.end`.
Thus, adjacent runs have lag one. Include every pair within the caller's inclusive lag window.
Require the interval from A's end through B's start to be observed; never infer a cascade across an unknown gap.
Report the interval from A's onset through B's end and the separating lag, not an invented joint duration.

Run boundaries that touch the record boundary or an unknown neighboring step are censored.
Record that fact; do not interpret their observed duration as the complete physical event duration.
Do not combine drought magnitude, fire severity, and heat intensity into an unrequested severity score.
Catalog pair counts are not independent observations and must not feed a binomial significance test.

### 7. Seasonality and block-bootstrap inference

Default to `groupby="month"`. An explicit `groupby="pooled"` reports a mixture over calendar months.
Do not average monthly LMFs to obtain a pooled LMF. Recount the pooled observations.
Do not add day-of-year bins in milestone 1; rare daily threshold events need more support than one annual sample per bin.

Inference is opt-in and monthly-only. Daily descriptive results carry no significance claim.
Requesting bootstrap inference on daily data raises `InvalidArgumentError`, rather than selecting an IID bootstrap.
The published-method reproduction does not validate this new inference method.

Use non-circular moving blocks of consecutive **complete calendar years**, each containing 12 monthly positions.
Default block length is `ceil(Y ** (1/3))` years, where `Y` is the selected inference segment's length.
Accept an explicit integer `block_length_years`.
This is a transparent heuristic, not an estimated optimal length or a guarantee about autocorrelation.

Use the longest contiguous segment of calendar years with complete selected-input observations in each cell.
Break ties by choosing the earliest segment. Do not concatenate separated complete years.
Report this segment and its point estimate separately from the descriptive full-record estimate.
For lagged conditioning, build A and response-window indicators on the original chronology first.
Keep anchors in complete anchor years whose required future responses are also observed.
Resample these derived rows; never create lag responses across artificial block joins.

Sample blocks with replacement from valid starts, concatenate them, and truncate to `Y` whole years.
Keep month positions fixed. For confidence intervals, resample all selected channels with identical block indices.
For the independence null, resample each channel independently with the same block length and calendar phase.
For conditional inference, the two channels are A and the precomputed response-window indicator R.
A three-hazard null tests mutual independence of all three channels.
Do not present a paired confidence interval as an independence test.

Report 95% percentile intervals using `numpy.quantile(method="linear")`.
Default to `n_resamples=1999` separately for intervals and null realizations.
Require an explicit integer `seed`; use a local NumPy generator, not global random state.
Record NumPy and SciPy versions because seeded cross-version bitwise identity is not promised.
Use the same resample-index stream in each cell so spatial tile boundaries cannot change a cell's result.
Different record lengths use deterministic streams derived from the seed and inference-support definition, not worker scheduling.

For joint inference, use excess `T = p_joint - product(p_i)`.
For conditional inference, use `T = P(R|A) - P(R)`.
The default alternative is amplification, `"greater"`; support explicit `"less"` for suppression.
Use inclusive null comparisons and `p=(1+number_at_least_as_extreme)/(B_valid+1)`.
Do not add random noise to break ties, return a zero p-value, or publish a two-sided test without a separate definition.

Require at least 30 complete inference years and five nonoverlapping blocks (`floor(Y/L)`).
Apply occurrence checks separately within each reported month, or across the pooled support when requested.
Require 10 positive occurrences in each marginal channel, or in A and R for conditional inference.
For percentile intervals, require five observed joint/conditional successes; otherwise flag the interval as unestimable.
For null tests, require an expected joint/success count of at least five under independence.
Use `N*product(p_i)` for joint tests and `n_A*P(R)` for conditional tests.
An observed zero joint count can therefore support a suppression test without producing a misleading zero-width interval.
Warn when fewer than 10 nonoverlapping blocks remain.
Require at least 95% valid resamples for each reported metric; otherwise return NaN and a quality flag.
Do not replace undefined LMF or conditional realizations with zero.
Predeclare sensitivity checks at `max(1, floor(L/2))`, `L`, and `2L`, where support permits.
Report conclusions that change with the block length. Do not choose L by the smallest p-value.

These guards do not detect nonstationarity, long memory, or impact relevance.
The inference issue must characterize interval coverage and false-positive rates before exposing significance.
All grid p-values are local and unadjusted. No output says that an entire field is significant.

## Illustrative API sketch

These are proposed signatures and docstring summaries, not executable implementation.
`SourceMetadata` below means a JSON-compatible mapping, not a new public framework.
Each computation validates a manually constructed prepared input too; a dataclass is not proof of validity.

```python
@overload
def harmonize(
    inputs: Mapping[str, npt.ArrayLike], *,
    time: Mapping[str, npt.ArrayLike],
    cadence: Mapping[str, Literal["daily", "monthly"]],
    input_scale: Mapping[str, Literal["probability", "indicator"]],
    thresholds: Mapping[str, float], directions: Mapping[str, Direction],
    target_cadence: Literal["daily", "monthly"],
    aggregation: Mapping[str, Literal["any", "all"]] | None = None,
    validity: Mapping[str, npt.ArrayLike] | None = None,
    join: Literal["exact", "inner"] = "exact",
    source_metadata: Mapping[str, SourceMetadata] | None = None,
    spatial_time_major: bool = False,
) -> AlignedHazards:
    """Threshold declared inputs and align observed hazard opportunities."""

@overload
def harmonize(
    inputs: Mapping[str, xr.DataArray], *,
    cadence: Mapping[str, Literal["daily", "monthly"]],
    input_scale: Mapping[str, Literal["probability", "indicator"]],
    thresholds: Mapping[str, float], directions: Mapping[str, Direction],
    target_cadence: Literal["daily", "monthly"],
    aggregation: Mapping[str, Literal["any", "all"]] | None = None,
    validity: Mapping[str, xr.DataArray] | None = None,
    join: Literal["exact", "inner"] = "exact",
    source_metadata: Mapping[str, SourceMetadata] | None = None,
    time_dim: str = "time",
) -> xr.Dataset:
    """Prepare labeled hazard and validity arrays without evaluating a grid."""

def joint_statistics(
    hazards: AlignedHazards, *, groupby: Literal["month", "pooled"] = "month",
    min_samples: int = 30,
) -> JointResult:
    """Count two or three hazards on one support and return frequencies and LMF."""

def conditional_probability(
    hazards: AlignedHazards, *, given: str, target: str,
    lag_window: tuple[int, int] = (0, 0),
    groupby: Literal["month", "pooled"] = "month", min_samples: int = 30,
) -> ConditionalResult:
    """Estimate a target-window occurrence probability given an anchor hazard."""

def compound_events(
    first: RunSet, second: RunSet, *, time: npt.ArrayLike, valid: npt.ArrayLike,
    cadence: Literal["daily", "monthly"],
    mode: Literal["concurrent", "sequential"],
    lag_window: tuple[int, int] | None = None,
    source_metadata: Mapping[str, SourceMetadata] | None = None,
) -> CompoundEventSet:
    """Match observed source-run pairs without combining their severity measures."""

def bootstrap_dependence(
    hazards: AlignedHazards, *,
    statistic: Literal["joint_probability", "likelihood_multiplication_factor",
                       "conditional_probability"],
    given: str | None = None, target: str | None = None,
    lag_window: tuple[int, int] = (0, 0),
    groupby: Literal["month", "pooled"] = "month",
    block_length_years: int | None = None, n_resamples: int = 1999,
    confidence_level: float = 0.95,
    alternative: Literal["greater", "less"] = "greater", seed: int,
) -> BootstrapResult:
    """Estimate monthly uncertainty and test calendar-conditioned independence."""

def compound_events_xarray(
    hazards: xr.Dataset, *, first: str, second: str,
    mode: Literal["concurrent", "sequential"],
    lag_window: tuple[int, int] | None = None, min_duration: int = 1,
) -> xr.Dataset:
    """Return an eager numeric event catalog for an explicitly computed subset."""
```

The three statistical functions also have overloads accepting the prepared xarray Dataset and returning a Dataset.
Keep their keyword contracts identical. Reject mixed NumPy/xarray input mappings.
Use `assert_type()` cases in `tests/test_type_checking.py` to prove both routes.
The event wrapper uses `identify_runs()` on each prepared occurrence mask with invalid positions set to NaN.
It reports source-mask runs, not original probability magnitudes. Users needing original run metrics retain their `RunSet`s.

The facade re-exports these functions and five result types only.
A separate `likelihood_multiplication_factor()` wrapper, pandas conversion, and a generic bootstrap-callback API are unnecessary.

## Data flow, metadata, errors, and execution

1. Compute source indices and available fit diagnostics upstream.
2. Declare the source representation, threshold, direction, cadence, and source metadata.
3. Use `harmonize()` to produce observed occurrence and validity arrays.
4. Compute descriptive joint or conditional statistics on explicit eligible support.
5. Match runs for an event catalog, or request monthly bootstrap inference.
6. Serialize numeric datasets and their choice manifest downstream.

The manifest records source names and order, representations, original physical comparators,
probability thresholds, directions, source Timescales, requested/effective Calibration Periods,
distributions used, fallback and zero-handling policies, source gap/state/spin-up policy where known,
fit-diagnostic references, source identifiers/checksums where supplied, and explicit unknown fields.
It also records calendar, original/retained dates, cadence, aggregation, dropped boundaries,
missingness policy, grouping, lag units/window, event matching/censoring rules, sample thresholds,
block selector/length, segment selection, confidence method, alternative, seed, resample counts,
package/dependency versions, and quality flags.

Keep large source arrays and diagnostic Datasets out of attributes. Retain their identifiers instead.
Use a versioned JSON string for nested choices and scalar CF-compatible attributes for common fields.
Use `build_output_attrs()` for library version and history, without inheriting unrelated source bounds or standard names.
History timestamps can differ while numeric results remain deterministic.

Proposed registry keys are `compound_joint_probability`, `compound_marginal_probability`,
`compound_independent_probability`, `compound_likelihood_multiplication_factor`,
`compound_conditional_probability`, and `compound_response_window_probability`.
Use `units="1"`, truthful `long_name`, verified references, and no invented `standard_name`.
Probability bounds are `[0, 1]`; LMF has `valid_min=0` and no universal `valid_max`.
Counts, event indices, censor flags, CI bounds, p-values, and quality variables define their attributes beside the adapter.
Use CF-style `flag_masks` and `flag_meanings` for insufficient support, undefined denominator,
missing source metadata, censoring, inadequate bootstrap support, and invalid-resample fractions.

Emit `calculation_started`, `calculation_completed`, and `calculation_failed` through the existing structlog helpers.
Bind the function, source names, shapes, cadence, group, and relevant inference choices.
Cover all-missing early returns and failures with behavior tests. Do not log source arrays or file contents.
Lazy completion means graph construction completed, not all worker computations completed; document this distinction.

Use `InvalidArgumentError` for incompatible choices, `InputTypeError` for invalid representations,
`DataShapeError` for NumPy layouts, and `CoordinateValidationError`/`DimensionMismatchError` for labeled coordinates.
Use existing warning categories where applicable. Use `ClimateIndicesWarning` plus result quality flags for short compound samples.
Do not create a parallel exception hierarchy or misuse `ShortCalibrationWarning` for an analysis window.
Validate options, coordinates, and chunk shapes before graph construction.
For lazy inputs, validate data values inside the numerical task; value-dependent errors can therefore appear when computation starts.

All xarray statistics require one `time` chunk per source and prepared mask.
A stacked `hazard` core dimension must also occupy one chunk.
Validate these dimensions with `validation.validate_dask_chunks()`; never enable automatic rechunking.
For harmonization that changes time length, require the caller to prepare a whole-time chunk before the call.
Spatial chunks must have matching coordinates and compatible boundaries across inputs.
Report incompatible boundaries rather than silently creating a large spatial block.

Use a dedicated shape-reducing `apply_ufunc` adapter rather than the fitting decorator,
which would incorrectly fold years and synthesize observations.
The first adapter may use the established per-cell route; it need not introduce a spatial-kernel optimization.
No bootstrap task allocates `(resample, time, cell)` arrays.
Generate one realization at a time and retain only metric realizations for quantiles.

For `k` sources, `T` times, `C` cells, `B` realizations, and `q` stored metrics,
a conservative planning budget is `(10*k + 24)*T*C + 8*B*q*C` bytes, excluding scheduler/array overhead.
This includes upstream float64 values, Boolean masks, copies, and index workspaces; it is not a measured limit.
For three daily inputs, 100 years, and 1,000 cells, the first term is about 1.97 GB before overhead.
Monthly inference with `T=1200`, `C=1000`, `B=1999`, and `q=1` budgets about 81 MB before overhead.
Measure peak memory in CMP-08 and publish the measured recommendation before advising tile sizes.
Catalog memory additionally grows with the number of matching event pairs, potentially `O(r_A*r_B)`.

## Validation plan and evidence classifications

This plan adds no claim to `VALIDATION.md`. Each implementation PR adds its actual evidence and known gaps.
When that file changes, regenerate `llms.txt` and `llms-full.txt` with the existing script.
The `validation` pytest marker alone never establishes external validation.

| Function or route | Required evidence | Initial classification and promotion rule |
| --- | --- | --- |
| `harmonize` | Hand-counted daily/monthly opportunities; leap years; month labels; partial months; interior gaps; masks; alignment; conflicting metadata. | Regression only; specification-level. Agreement between adapters is not external validation. |
| `joint_statistics` | Exact contingency tables; symmetry under source permutation; bounds; zero marginals; three-source independence; known Gaussian joint probabilities. | Regression only; analytic/specification-level until an external reference is committed. |
| `conditional_probability` | Enumerated windows; `(0,0)` identity with joint/marginal ratios; asymmetry; incomplete horizons; crossing month/year boundaries; missing response steps. | Regression only; analytic/specification-level. A lagged association is not a validated causal cascade. |
| `compound_events` | Enumerated `RunSet` intersections/gaps; inclusive ends; all-pairs matching; invalid bridges; empty catalogs; censored boundaries. | Regression only; specification-level. Retain this status until an independent matching oracle exists. |
| `bootstrap_dependence` | Fixed-seed reproducibility; exact enumeration on tiny block samples; interval coverage and null false-positive experiments; block-length sensitivity. | Synthetic characterization plus regression. A paper using IID resampling cannot upgrade this method's status. |
| Xarray statistics/catalogs | NumPy/eager/Dask equality, transposed/custom dimensions, split-chunk rejection, tile invariance, lazy graph construction, and numeric serialization. | Interface equivalence/regression only. Inherit, but do not enlarge, the numerical core's evidence scope. |
| Published LMF reproduction | Public study inputs and pinned author calculation, with a named subset and explicit convention audit. | Reference-reproduced, not independently validated, only after measured agreement. Mismatched published map magnitudes remain characterization. |

Analytic and synthetic cases must include:

- A balanced Cartesian indicator table with exact independence and `LMF=1`, including three inputs.
- Independent seeded uniform series, with a Monte Carlo error envelope rather than an arbitrary equality tolerance.
- Gaussian-copula samples at zero thresholds, where `P(Z1>0,Z2>0)=1/4+asin(rho)/(2*pi)`.
  Use `rho=0` and `rho=0.5`, giving exact joint probabilities `1/4` and `1/3`.
  Corresponding LMFs are 1 and `4/3` for balanced halves.
- Nonzero Gaussian thresholds checked against an independently evaluated deterministic quadrature, not the implementation's count routine.
- Perfect dependence, mutual exclusion, constants, NaN-only cells, invalid ranges, ties, endpoints, and integer/float32/float64/strided inputs.
- Existing Hypothesis tests for probability bounds, joint symmetry, monotonic threshold behavior, and invariance to source-name permutations.
  Conditional probabilities are deliberately not symmetric.
- Seasonal independent series with changing marginal rates, showing why pooled LMF can differ from month-conditioned LMF.
- Monthly independent persistent series and persistent correlated pairs, including multi-year persistence and overlapping source Timescales.
  Measure interval coverage and null rejection rates over predeclared seeds; record binomial uncertainty for those measured rates.
  Run broader characterization outside the fast suite, but commit its configuration and measured summary.

Use exact integer equality for counts and event indices. Propose `atol=1e-12`, `rtol=0` for analytic frequencies.
Use exact equality for identical seeded resampling indices and masks within a supported dependency environment.
Stochastic acceptance bands must follow sample size and predeclared Monte Carlo error, not the optional-Rust `1e-10` parity tolerance.
No native code changes are planned; ADR-0017 parity applies if acceleration is later added.

### Published-method reproduction: verified target

Use **Vieira Passos, M., Kan, J.-C., Destouni, G., Barquet, K., and Kalantari, Z. (2024)**,
*Identifying regional hotspots of heatwaves, droughts, floods, and their co-occurrences*,
*Stochastic Environmental Research and Risk Assessment* 38, 3875–3893.
DOI: <https://doi.org/10.1007/s00477-024-02783-3>.
The publisher's author metadata, year, DOI, Sections 2.2 and 2.4, and data-availability statement were read.

The public dataset is **Vieira Passos, M. (2024)**,
*Climate Indices for heat waves, droughts and floods in Sweden (1922-2021)*, version 2.
DOI: <https://doi.org/10.6084/m9.figshare.25189118.v2>.
The [Figshare API record](https://api.figshare.com/v2/articles/25189118) confirms CC BY 4.0,
six monthly SPI/SPEI files, two daily heatwave files, and a daily flood-index file.
The archive totals about 9.37 GB; `SPEI12.nc` is about 96 MB and `heatwave_SMHI.nc` about 2.92 GB.
No data files were downloaded for this plan.

The paper links the [author repository](https://github.com/m-passos/hydroclimatic-hazards-hotspots).
The inspected commit is `6c88f255ee3034279dcebe2a2fc40c954dbe9575`.
Its MIT license and `Compound/Compound_patterns.ipynb` were read without executing the notebook.
The notebook's `lmf` calculation supplies the same-author numeric reference method.

CMP-02 must extract three deterministic valid grid cells spanning southern, central, and northern Sweden,
with their actual coordinates and January 1922–December 2021 coverage recorded after inspecting the file headers.
Use the summer months June, July, and August as the first scoped Figure 8 drought–heatwave target.
Do not invent coordinates or reference values now.
Pin the dataset version, file IDs, downloaded checksums, source-code commit, variable names,
cell-selection rule, and final numeric reference arrays.

The source requires an explicit discrepancy audit:

- The notebook pads monthly SPEI to daily dates and then calls linear NaN interpolation.
  This is not the module's recommended monthly aggregation. Reproduce it only in offline fixture preparation.
- The notebook uses inclusive thresholds (`HWI >= 0`, `SPEI <= -1`), unlike the strict probability path.
  Feed explicitly declared Boolean indicators and the original physical comparators to `harmonize()` at the prepared daily cadence.
  This tests the module's empirical relationship layer without claiming to validate upstream PIT fitting.
- The notebook uses IID daily resampling and small random tie noise. Do not copy those significance procedures.
- Its `lmf` returns NaN when no co-occurrences exist. This module returns zero when both marginals are positive.
  Test agreement on nonzero-joint cells and test the intentional zero-case difference separately.
- The paper's prose describes negative LMF values. Its ratio cannot be negative; suppression corresponds to `0 <= LMF < 1`.
- Figure 8 is a map, not a full-precision numeric table. Matching the pinned author method on public inputs earns
  scoped **reference-reproduced** evidence, not proof that the entire map or scientific method is independently validated.

Target exact indicator counts and `atol=1e-12`, `rtol=0` for LMF from identical masks.
Confirm these tolerances after the input/reference audit; never widen them to conceal mismatched conventions.
Store a small redistributable subset and schema-valid `tests/fixture/compound/provenance.json`.
Use `notes` and a companion README for detailed choices because the existing provenance schema rejects unknown keys.
Include the dataset attribution and applicable source-code notice if reference code is adapted.
Normal CI must use committed fixtures without network access or optional external packages.
If the public files cannot produce a matched numeric reference, CMP-02 blocks milestone completion;
characterization alone must not silently satisfy the reproduction requirement.

## Dependencies and earthflux boundary

Add no runtime or test dependency. Use NumPy/SciPy for arithmetic and random sampling,
xarray for labeled adapters, existing Dask execution, structlog, dataclasses, and existing Hypothesis tests.
Do not add `arch`, a copula package, pandas-based output types, or a bootstrap framework.
Existing adapter infrastructure may retain its current pandas dependency; new compound algorithms do not require it.

The library remains BSD-3-Clause and has no import, service call, configuration, or schema owned by earthflux.
Earthflux needs stable names, versioned choice metadata, numeric serializable outputs, counts and quality flags,
source IDs, and honest local-inference labels. Presentation, storage, spatial summaries, and trigger selection stay downstream.
CC BY fixture data and any MIT reference snippets retain their own notices; neither changes the library license.

## Dependency-ordered work breakdown

The CMP identifiers below map to the created GitHub issues.
All implementation is blocked until CMP-01's ADR PR is merged by the maintainer.
No child is marked `ready-for-agent` while a scientific choice or dependency remains unresolved.
Use only existing labels; no `compound` or `epic:compound` label currently exists.

| ID | One-PR slice | Depends on |
| --- | --- | --- |
| [CMP-00 (#1364)](https://github.com/monocongo/climate_indices/issues/1364) | Track the milestone as a parent epic. | All children must complete before closure. |
| [CMP-01 (#1365)](https://github.com/monocongo/climate_indices/issues/1365) | Publish the approved plan and accepted contracts ADR. | The documentation PR requires review and merge. |
| [CMP-02 (#1366)](https://github.com/monocongo/climate_indices/issues/1366) | Prepare the published LMF reference subset and convention audit. | CMP-01. |
| [CMP-03 (#1367)](https://github.com/monocongo/climate_indices/issues/1367) | Implement declared-input harmonization and `AlignedHazards`. | CMP-01. |
| [CMP-04 (#1368)](https://github.com/monocongo/climate_indices/issues/1368) | Implement joint counts, frequencies, LMF, and registry entries. | CMP-03. |
| [CMP-05 (#1369)](https://github.com/monocongo/climate_indices/issues/1369) | Implement conditional and lag-window probabilities. | CMP-04. |
| [CMP-06 (#1370)](https://github.com/monocongo/climate_indices/issues/1370) | Implement concurrent and sequential `RunSet` matching. | CMP-03. |
| [CMP-07 (#1371)](https://github.com/monocongo/climate_indices/issues/1371) | Implement monthly block-bootstrap intervals and null tests. | CMP-04, CMP-05. |
| [CMP-08 (#1372)](https://github.com/monocongo/climate_indices/issues/1372) | Add beta fixed-shape xarray adapters and Dask contracts. | CMP-03, CMP-04, CMP-05, CMP-07. |
| [CMP-09 (#1373)](https://github.com/monocongo/climate_indices/issues/1373) | Add eager numeric xarray event catalogs. | CMP-06, CMP-08. |
| [CMP-10 (#1374)](https://github.com/monocongo/climate_indices/issues/1374) | Complete the published reproduction, user guide, and evidence report. | CMP-02, CMP-04, CMP-05, CMP-07, CMP-08, CMP-09. |

CMP-02 and CMP-03 can proceed independently after the ADR merges.
CMP-06 can proceed beside conditional and inference work.
Do not create a GitHub milestone or project board without separate approval; a parent issue is sufficient initially.

## GitHub issue specifications

The sections below retain the approved issue specifications.
The published bodies also contain scoped contract excerpts, real issue references, and the validation commands.
Each child begins `Part of #1364`, with readable `Blocked by:` lines where applicable.
GitHub records native parent and dependency links, and the parent contains a child checklist.
Issues CMP-02 through CMP-10 carry `status:blocked`; CMP-01 is the documentation work in progress.
The acceptance criteria are the issue-to-PR completion contract, not permission to expand scope.

### CMP-00 — [EPIC] Initial compound-hazard analysis milestone

**GitHub issue:** [#1364](https://github.com/monocongo/climate_indices/issues/1364).

**Labels:** `epic`, `enhancement`, `design`, `ready-for-human`.

**Is your feature request related to a problem? Please describe.**
Researchers repeatedly rebuild threshold alignment, joint frequency, event matching, and dependence inference around existing indices.

**Describe the solution you'd like.**
Deliver the scope and contracts in this plan through CMP-01–CMP-10.
Keep a dependency-ordered child checklist in the parent body.

**Describe alternatives you've considered.**
A copula-first release adds fitting assumptions before the empirical contracts have evidence.
A daily-inference-first release adds unresolved calendar resampling to the first milestone.

**Acceptance criteria:**
- [ ] The maintainer merges the ADR before any implementation issue starts.
- [ ] Every child merges with its required tests and documentation.
- [ ] The public API, CF metadata, serialized results, and beta limitations match the accepted contracts.
- [ ] `VALIDATION.md` records actual evidence, including a scoped published-method reproduction and its known gaps.
- [ ] Daily significance, field significance, and operational prediction are explicitly excluded.

**Dependencies:** Approval to create issues; all child completions for closure.
**Additional context:** The library remains independent of earthflux. The maintainer alone owns merges and release scheduling.

### CMP-01 — Record the compound milestone plan and architecture contracts

**GitHub issue:** [#1365](https://github.com/monocongo/climate_indices/issues/1365).

**Labels:** `documentation`, `design`, `ready-for-human`.

**Is your feature request related to a problem? Please describe.**
Implementation cannot safely choose calendar, denominator, lag, or inference semantics independently.

**Describe the solution you'd like.**
Review this plan and the proposed ADR. Resolve the approval questions below.
After approval, publish the accepted ADR under the next free number and add it to `docs/adr/index.md`.
Keep this detailed plan internal and link it from the ADR through an absolute repository URL.

**Describe alternatives you've considered.**
A published `Proposed` ADR conflicts with current authoring conventions. Keeping the proposal internal avoids an unrelated policy migration.

**Acceptance criteria:**
- [ ] The plan and ADR agree on every public contract and named scope exclusion.
- [ ] The accepted ADR clearly identifies unimplemented APIs and the follow-up issues.
- [ ] The ADR records the scoped ADR-0001 placement and ADR-0004 analysis-calendar exceptions without changing existing code.
- [ ] Published navigation and links pass both Sphinx gates.
- [ ] No implementation, fixture, dependency, CF entry, or unsupported validation claim enters this PR.

**Dependencies:** Maintainer approval of this draft.
**Additional context:** Use `docs/<issue>-compound-contracts`. This is the first issue to drive to a draft PR after approval.

### CMP-02 — Prepare a public drought–heatwave LMF reference fixture

**GitHub issue:** [#1366](https://github.com/monocongo/climate_indices/issues/1366).

**Labels:** `validation`, `type:literature`, `type:testing`, `tests`.

**Is your feature request related to a problem? Please describe.**
Source algebra and internally generated arrays cannot establish a published-method reproduction.

**Describe the solution you'd like.**
Prepare the small Figshare/author-notebook subset described in the validation section.
Audit conventions before generating expected counts and LMF values.

**Describe alternatives you've considered.**
Digitizing the full map loses precision. Recomputing the authors' climate indices would introduce unrelated model and calibration differences.

**Acceptance criteria:**
- [ ] Pin the verified paper citation, dataset version/files, source commit, coordinates, variables, dates, and all processing choices.
- [ ] Record the inclusive thresholds, monthly-to-daily preparation, zero-joint difference, and excluded IID significance method.
- [ ] Commit small inputs, author-method reference counts/LMF, attribution, checksums, README, and schema-valid Fixture Provenance.
- [ ] Add fixture integrity and independent reference self-checks that pass without importing an unimplemented compound module.
- [ ] Add a reviewed preparation script; never execute the downloaded notebook unchanged.
- [ ] Record retrieval or matching failures as blockers; do not replace the target with self-generated compound output.
- [ ] Document the evidence limit without adding a premature `VALIDATION.md` success claim. No CF entry is needed for fixture-only work.

**Dependencies:** CMP-01.
**Additional context:** Use `test/<issue>-compound-reference`. Keep large downloads outside the repository. No normal test uses the network.

### CMP-03 — Add declared-input harmonization and prepared hazard results

**GitHub issue:** [#1367](https://github.com/monocongo/climate_indices/issues/1367).

**Labels:** `enhancement`, `numpy-api`, `tests`.

**Is your feature request related to a problem? Please describe.**
Mixed cadences and missing dates can silently create false joint opportunities.

**Describe the solution you'd like.**
Implement NumPy `harmonize()` and `AlignedHazards`, including explicit declarations, validity, cadence reduction, and source manifests.

**Describe alternatives you've considered.**
Averaging probabilities, interpolating monthly scores, and deleting missing dates change the estimand or close unknown gaps.

**Acceptance criteria:**
- [ ] Validate representations, strict thresholds, explicit directions, Boolean declarations, mappings, layouts, dates, and known calibration conflicts.
- [ ] Cover exact/inner alignment, interior absent dates, real leap days, month labels, partial months, and daily `any`/`all` reduction.
- [ ] Preserve masks and unknown metadata; reject raw z-scores and unsupported calendars without clipping or interpolation.
- [ ] Make result arrays read-only without mutating caller data; add metadata and serialization-oriented schema tests.
- [ ] Add Google-style API docs, error/warning tests, lifecycle tests, and a specification-level `VALIDATION.md` row.
- [ ] Do not add root re-exports, an empty scaffold, adapters, or false CF/PIT labels to aggregated indicators.

**Dependencies:** CMP-01.
**Additional context:** Use `feature/<issue>-compound-harmonization`. Add no dependency. Regenerate derived validation bundles.

### CMP-04 — Add joint frequencies and likelihood multiplication factor

**GitHub issue:** [#1368](https://github.com/monocongo/climate_indices/issues/1368).

**Labels:** `enhancement`, `numpy-api`, `validation`, `tests`.

**Is your feature request related to a problem? Please describe.**
Separate marginal and joint masks produce biased LMF denominators.

**Describe the solution you'd like.**
Implement one `joint_statistics()` count pass and `JointResult` for two or three sources.

**Describe alternatives you've considered.**
Separate frequency and LMF functions duplicate counts. Threshold-implied marginals fail for discrete or poorly fitted PIT series.

**Acceptance criteria:**
- [ ] Return all counts and probabilities on the same eligible support, grouped by month or explicitly pooled.
- [ ] Cover zero marginals, zero joint counts, short support, all-missing cells, and three-source mutual independence.
- [ ] Add exact contingency tables, Gaussian known-answer tests, and Hypothesis symmetry/bounds tests.
- [ ] Add the joint/marginal/independence/LMF registry entries with verified references, truthful units/bounds, and no standard name.
- [ ] Add typed results, lifecycle and error tests, public documentation, and honest `VALIDATION.md` evidence.
- [ ] Regenerate derived bundles; do not claim independence from `LMF=1` alone.

**Dependencies:** CMP-03.
**Additional context:** Use `feature/<issue>-compound-joint-statistics`. The optional-Rust tolerance is not a stochastic test tolerance.

### CMP-05 — Add conditional and forward-window hazard probabilities

**GitHub issue:** [#1369](https://github.com/monocongo/climate_indices/issues/1369).

**Labels:** `enhancement`, `numpy-api`, `validation`, `tests`.

**Is your feature request related to a problem? Please describe.**
Lag analyses often count multiple responses or retain only successful incomplete horizons.

**Describe the solution you'd like.**
Implement pairwise `conditional_probability()` with explicit source roles and inclusive forward windows.

**Describe alternatives you've considered.**
Event-onset probabilities answer a different question. Automatically searching lags adds an uncorrected selection problem.

**Acceptance criteria:**
- [ ] Count each eligible A-positive anchor once and require its entire requested B horizon.
- [ ] Return numerator, denominator, eligible anchors, conditional probability, and matched response-window baseline.
- [ ] Cover zero-lag identity, missing B steps, boundary truncation, daily leap dates, monthly lags, and anchor-month grouping.
- [ ] Reject negative/reversed windows and ambiguous source roles; demonstrate that conditioning is not symmetric.
- [ ] Add conditional/response-window CF registry entries, typed results, lifecycle/error tests, and API/algorithm documentation.
- [ ] Add specification-level `VALIDATION.md` evidence and regenerate bundles. Make no causal or event-onset claim.

**Dependencies:** CMP-04.
**Additional context:** Use `feature/<issue>-compound-conditional`.

### CMP-06 — Match concurrent and sequential RunSet event pairs

**GitHub issue:** [#1370](https://github.com/monocongo/climate_indices/issues/1370).

**Labels:** `enhancement`, `numpy-api`, `tests`.

**Is your feature request related to a problem? Please describe.**
Studies repeatedly rebuild run intersections and lag matching with incompatible endpoint rules.

**Describe the solution you'd like.**
Implement `compound_events()` and `CompoundEventSet` using existing source runs and explicit observation support.

**Describe alternatives you've considered.**
One-to-one matching discards valid relationships. Merging all pairs hides which source events produced each match.

**Acceptance criteria:**
- [ ] Validate manually constructed RunSets and retain their inclusive-end and source-ID conventions.
- [ ] Test all concurrent intersections, sequential gaps, lag boundaries, repeated matches, empty outputs, and invalid bridges.
- [ ] Record left/right censoring and distinguish overlap duration from the sequential enclosing interval.
- [ ] Preserve original inputs; never combine unrelated severity metrics or change `runs.py` behavior.
- [ ] Add stable typed arrays, lifecycle/error tests, event-algorithm documentation, and specification-level `VALIDATION.md` evidence.
- [ ] Document catalog fields and units for the later adapter; no primary index CF key is needed for an event catalog.

**Dependencies:** CMP-03.
**Additional context:** Use `feature/<issue>-compound-events`. Regenerate derived bundles when validation documentation changes.

### CMP-07 — Add monthly block-bootstrap intervals and dependence null tests

**GitHub issue:** [#1371](https://github.com/monocongo/climate_indices/issues/1371).

**Labels:** `enhancement`, `numpy-api`, `validation`, `tests`.

**Is your feature request related to a problem? Please describe.**
IID resampling understates uncertainty under persistence, and paired resampling does not test independence.

**Describe the solution you'd like.**
Implement the monthly moving-year-block contract, including inference-support estimates and explicit seeded sampling.

**Describe alternatives you've considered.**
IID daily resampling and a universal fixed-day block ignore calendar structure. Daily inference remains a separate design task.

**Acceptance criteria:**
- [ ] Separate paired confidence sampling from independent-channel null sampling and define both inference statistics exactly.
- [ ] Preserve month phase and contiguous year blocks; compute lag responses before resampling.
- [ ] Enforce separate interval-success and null-expected-count guards, valid-resample rules, and block-count limits, with numeric quality flags.
- [ ] Test zero-joint suppression without a false zero-width interval or a false zero p-value.
- [ ] Cover reproducibility, tie handling, non-circular boundaries, missing-year segmentation, block-length overrides, and daily rejection.
- [ ] Characterize coverage and null error rates on seasonal/persistent synthetic records, including multi-year dependence and length sensitivity.
- [ ] Commit characterization configurations/results and a justified acceptance band before exposing significance.
- [ ] Document interval/p-value attributes beside the future adapter, local unadjusted inference, lifecycle events, and synthetic-only `VALIDATION.md` status.

**Dependencies:** CMP-04, CMP-05.
**Additional context:** Use `feature/<issue>-compound-bootstrap`. Do not add a bootstrap dependency or native kernel.

### CMP-08 — Add beta xarray statistics adapters with bounded Dask execution

**GitHub issue:** [#1372](https://github.com/monocongo/climate_indices/issues/1372).

**Labels:** `enhancement`, `xarray-api`, `performance`, `tests`.

**Is your feature request related to a problem? Please describe.**
Researchers need labeled grid statistics without hidden materialization, calendar conversion, or denominator drift.

**Describe the solution you'd like.**
Add `harmonize`, joint, conditional, and bootstrap Dataset routes over the NumPy computations.

**Describe alternatives you've considered.**
The fitting decorator has incompatible calendar and output-shape behavior. Separate per-adapter mathematics would create two scientific implementations.

**Acceptance criteria:**
- [ ] Add overloads and `assert_type()` coverage; emit the established beta warning.
- [ ] Preserve coordinates and numeric counts/flags; reject missing coordinates, split core chunks, incompatible grids, and mixed backends.
- [ ] Prove eager/Dask/NumPy equality, custom/transposed dimensions, mask equivalence, and bootstrap tile invariance.
- [ ] Verify graph construction does not evaluate input grids and lifecycle logs describe scheduling honestly.
- [ ] Apply registry metadata and versioned JSON choices; remove inherited physical names/bounds; round-trip a numeric Dataset.
- [ ] Measure peak memory and document chunk guidance. Never allocate a resample-by-time-by-cell cube or rechunk automatically.
- [ ] Document the beta interface and equivalence-only `VALIDATION.md` evidence; regenerate bundles where required.

**Dependencies:** CMP-03, CMP-04, CMP-05, CMP-07.
**Additional context:** Use `feature/<issue>-compound-xarray`. Do not broaden generic standardization or existing fire/flood adapters.

### CMP-09 — Add eager numeric compound-event catalogs for grid subsets

**GitHub issue:** [#1373](https://github.com/monocongo/climate_indices/issues/1373).

**Labels:** `enhancement`, `xarray-api`, `tests`.

**Is your feature request related to a problem? Please describe.**
Object-valued grid RunSets cannot serve as portable event catalogs.

**Describe the solution you'd like.**
Add `compound_events_xarray()` for already computed prepared hazard subsets.

**Describe alternatives you've considered.**
A Dask object array preserves neither portable serialization nor a bounded variable-length output shape.
A distributed ragged-array engine is outside this milestone.

**Acceptance criteria:**
- [ ] Reuse `identify_runs()` and CMP-06 matching, with shared validity and recorded `min_duration`.
- [ ] Return numeric event rows, cell lookup coordinates, timestamps, source-run IDs, lags, and censor flags, including an empty catalog.
- [ ] Reject lazy input without evaluating it; document explicit selection and computation before catalog construction.
- [ ] Add per-variable CF-style metadata without an invented standard name or unsupported CF feature type.
- [ ] Test one-series/grid equivalence and NetCDF/Zarr numeric round trips using installed project facilities.
- [ ] Document eager memory and pair-count risks, beta status, and equivalence-only `VALIDATION.md` evidence.

**Dependencies:** CMP-06, CMP-08.
**Additional context:** Use `feature/<issue>-compound-event-catalog`.

### CMP-10 — Reproduce the public LMF case and publish the compound guide

**GitHub issue:** [#1374](https://github.com/monocongo/climate_indices/issues/1374).

**Labels:** `documentation`, `validation`, `demo`, `tests`.

**Is your feature request related to a problem? Please describe.**
An API release needs an auditable worked study and evidence limits, not only synthetic success cases.

**Describe the solution you'd like.**
Run the prepared public-study subset through the compound API and publish a small reproducible guide and final evidence table.

**Describe alternatives you've considered.**
A full Sweden remapping is expensive and unnecessary for the first contract check.
An author-notebook match cannot be described as independent external validation.

**Acceptance criteria:**
- [ ] Compare source counts exactly and scoped summer LMF at the audited tolerance; record any intentional convention differences.
- [ ] Add offline `validation` tests against the committed reference, with no skip standing in for the reproduction requirement.
- [ ] Record reference-reproduced versus characterization evidence separately in `VALIDATION.md`, with measured errors and known gaps.
- [ ] Publish an API reference and a how-to showing probability inputs, physical Boolean indicators, cadence reduction, lags, counts, and source manifests.
- [ ] Include a seeded monthly-inference example and explain why the daily published reproduction has no new bootstrap-significance claim.
- [ ] Wire published pages into their proper section navigation; keep internal planning excluded; verify all references and CF outputs.
- [ ] Update the core vocabulary for the accepted compound terms, regenerate bundled documents, and run the full required gates.

**Dependencies:** CMP-02, CMP-04, CMP-05, CMP-07, CMP-08, CMP-09.
**Additional context:** Use `docs/<issue>-compound-guide-validation`. Release notes and a release date remain maintainer-owned.

## Gates for future PRs

For this documentation-only planning change, run both published-docs gates.
For later source/test changes, run the repository's full gates before a draft PR:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/ tests/test_type_checking.py
uv run pytest
uv run --extra docs sphinx-build -E -b html -W --keep-going docs docs/_build/html
uv run --extra docs sphinx-build -E -b doctest docs docs/_build/doctest
```

Add `uv run pytest -m validation` when reference tests change.
Run `scripts/generate_llms_txt.py` whenever a bundled source document changes.
Use the release-integrity gate if packaging, workflow, or relevant documentation-publication rules change.
No Rust gate is needed unless later work touches native code or native dispatch.

## Approved recommendations and remaining gates

1. **Scope:** Monthly-only inference and eager-only gridded catalogs are approved for milestone 1. Daily inference requires a later design.
2. **ADR publication:** Promotion of the internal `Proposed` draft to an `Accepted` contracts record is approved for CMP-01.
3. **Input declaration:** The explicit Boolean path for physical thresholds is approved. It avoids an unjustified fit for existing fire and heat indicators.
4. **Conservative support:** The minimum counts, complete-month aggregation, and complete-year inference segment policy are approved. They can exclude sparse real records.
5. **Bootstrap evidence:** The year-block heuristic is approved as the proposed default, conditional on CMP-07's characterization gate. It is not a universal persistence correction.
6. **Reference logistics:** CMP-02 may retrieve the approximately 3 GB drought–heatwave source pair after its blockers resolve. No large download occurred during planning.
7. **Delivery:** The release target remains unassigned pending the contracts and reference audit. This plan does not assume that 3.1.0 is available.

Main scientific risks are sparse joint events, shared seasonal drivers, upstream fit error,
nonstationarity, overlapping Timescales, and treating local p-values or observed lags as stronger evidence than they provide.
Main execution risks are full-history spatial tiles and catalogs with many matching pairs.
No guardrail removes the need for study-specific scientific review.

## Local review record

The plan received a source-contract and scientific-consistency self-review before delivery.
It did not receive an independent external scientific review.
The review corrected these design risks:

- Published `Proposed` ADR status conflicted with repository conventions; the draft stayed internal until the maintainer approved its promotion.
- Aggregated CDF values would have been mislabeled as probabilities; reduction now acts on declared hazard indicators.
- Negative LMF and threshold-implied marginals would be false; the plan uses observed counts and nonnegative ratios.
- Confidence intervals would not test independence; paired and null resampling now have distinct contracts.
- Daily inference and lazy object catalogs lacked defensible calendar/storage contracts; both are explicitly deferred.
- Published inclusive thresholds and IID significance differ from the new defaults; the reference audit records those differences.
- One observed-success guard would suppress valid suppression tests; interval and null-test support now have separate rules.
- Bare RunSets do not retain source choices or cadence; the low-level matcher now accepts those declarations explicitly.

The maintainer approved the recommendations above. The ADR merge and numerical characterization remain mandatory gates, not implementation discretion.

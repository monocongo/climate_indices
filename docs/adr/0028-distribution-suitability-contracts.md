# Distribution-suitability assessment contracts

## Status

Amended: the NumPy comparison core is implemented in
[SUIT-03 (#1584)](https://github.com/monocongo/climate_indices/issues/1584).
GEV for SPEI is implemented in [SUIT-02 (#1583)](https://github.com/monocongo/climate_indices/issues/1583).
The maintainer approved these contracts.
[SUIT-01 (#1582)](https://github.com/monocongo/climate_indices/issues/1582) records this decision under
[epic #1581](https://github.com/monocongo/climate_indices/issues/1581).

**The suitability API is being implemented issue by issue.** This record approves contracts, not scientific validation.
The implementation responsibilities and their current state are explicit:

| Proposed API | Tracking issue | Current state |
| --- | --- | --- |
| `Distribution.gev` for `spei` (Hosking, L-moments) | [#1583](https://github.com/monocongo/climate_indices/issues/1583) | Implemented, with SciPy CDF parity and synthetic-recovery tests; no external fixture ([VALIDATION.md](https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md)). |
| `suitability.compare_distributions()` and `DistributionComparison`, NumPy core | [#1584](https://github.com/monocongo/climate_indices/issues/1584) | Implemented for the gamma, Pearson Type III, log-logistic and GEV candidates (GEV joined with [#1583](https://github.com/monocongo/climate_indices/issues/1583)); the goodness-of-fit field and the GoF and seed provenance attributes arrive with [#1586](https://github.com/monocongo/climate_indices/issues/1586). |
| `suitability.compare_distributions()`, xarray/Dask route | [#1585](https://github.com/monocongo/climate_indices/issues/1585) | Not implemented. |
| `GoodnessOfFit`: Anderson–Darling and parametric bootstrap, optional BH adjustment | [#1586](https://github.com/monocongo/climate_indices/issues/1586) | Not implemented. |
| `suitability.drought_category()` and `suitability.compare_categories()` | [#1587](https://github.com/monocongo/climate_indices/issues/1587) | Not implemented. |
| `suitability.pit_uniformity()` | [#1588](https://github.com/monocongo/climate_indices/issues/1588) | Not implemented. |
| `suitability.event_concordance()` and `suitability.category_agreement()` | [#1590](https://github.com/monocongo/climate_indices/issues/1590) | Not implemented. |
| Workflow reproduction notebook and "choosing a distribution" guide | [#1591](https://github.com/monocongo/climate_indices/issues/1591) | Not implemented. |

Amend this status and implementation table as those issues land.

## Context

Users must choose a distribution per index, region, season and Timescale.
The library reports a K–S statistic and exact p-value per fit (`fit_diagnostics`).
It supports GEV for SPEI and NumPy relative-fit comparisons, but decision-impact and concordance tools remain unimplemented.
The existing p-value is naive: it does not account for estimating parameters from the tested sample.
In the classical Lilliefors setting, naive fitted-parameter K–S p-values are conservative
(as discussed in the SciPy `goodness_of_fit` documentation). This does not establish
conservativeness for the L-moment estimators used here; their naive p-values are
uncalibrated for parameter estimation.

The motivating study is Monish & Rehana (2020), doi:10.1007/s12040-019-1271-x.
It is closed access and has not been read for this record, so none of its figures or method details are quoted here.

## Decision

### 1. Candidate set

- The candidates are the likelihood-based families, and `candidates=None` selects them:
  gamma and Pearson Type III for `spi` and `standardized_index`;
  gamma, Pearson Type III, log-logistic and GEV for `spei`.
  Naming a family outside this set (such as empirical or beta) in `candidates` raises.
- For `standardized_index`, apply `aggregation` and `clip_negatives` as the public function does.
  Gamma and Pearson Type III reject negative input when `clip_negatives=False`;
  signed input requires `clip_negatives=True` to clip it before fitting.
- SPEI gains `Distribution.gev`: Hosking's GEV fitted by L-moments, with `fitting_params` keys
  `loc`, `scale` and `shape`. The shape is Hosking's k, equal to SciPy's `c`.
  Add its entry to `_DISTRIBUTION_DISPLAY_NAMES` in `vocabulary.py` so CLI status logging can use `display_name`.
- GEV is fitted with no zero mass, as for GLO in {doc}`0016-log-logistic-distribution-for-spei`.
- `spi`, `standardized_index` and `fit_diagnostics` reject GEV, because they carry a physical zero mass
  and GEV with a positive shape has support below zero. This follows the ADR-0016 precedent.
- Empirical, beta ({doc}`0024-beta-distribution-for-bounded-series`) and the SPAI families are not comparison candidates.

### 2. Estimators

The comparison uses each family's existing estimator: gamma by Thom's MLE approximation;
Pearson III, GLO and GEV by L-moments.
Log-likelihood is evaluated at those estimates, because those are the parameters the index actually uses, and is labelled as such.
These are not maximum-likelihood estimates in general, so the criteria are penalized likelihoods at fitted estimates,
and a ranking mixes distribution fit with estimator efficiency.
The criteria keep the `aic`, `aicc` and `bic` names, and the documentation, docstrings and attributes carry this caveat.
MLE refitting is not part of this decision and can be added later as an option.

### 3. Criteria and the shared sample

With k free parameters, plus 1 for p0 when a zero mass is fitted (the same for all candidates sharing it):

- AIC = 2k − 2ℓ
- AICc = AIC + 2k(k+1)/(n−k−1), NaN when n ≤ k + 1
- BIC = k·ln n − 2ℓ

Candidates are scored on one shared sample: the same cells, the same steps and one zero definition.
A zero is an exact zero, which is the definition both existing zero-mass estimators fit with.
The 0.0005 Pearson trace rule is a transform convention and does not change the fitted sample.
If trace values are ever reclassified as zeros, the fitted samples and scores change, and the documentation must say so.

### 4. "Best"

`best` is the argmin of the chosen criterion (default `aicc`) over candidates that fitted and have a defined criterion.
Ties go to the earlier candidate in canonical enum order.
The value is −1 when no candidate is eligible, and `delta_to_runner_up` is NaN when fewer than two candidates are eligible.
It is returned whenever it is defined, and no significance threshold is applied.

### 5. Goodness of fit

- K–S is computed with naive p-values, labelled as naive.
- The Anderson–Darling statistic is computed from the probability integral transform.
- An optional parametric bootstrap, which refits with the library's own estimator, gives p-values for both.
- Benjamini–Hochberg adjustment is optional and uses SciPy.
- Warn when Timescale > 12 for monthly data or > 365 days for daily data:
  overlapping aggregation windows across consecutive years invalidate GoF p-values and effective n.
- Identify the GoF p-value of a criterion-selected candidate as a post-selection statistic
  in the result and user-facing documentation; it is not an ordinary confirmatory test.

### 6. Result types and provenance

- The NumPy routes return frozen dataclasses.
  The xarray routes return Datasets with a `candidate` dimension (string labels),
  a `month` or `dayofyear` dimension and the cell dimensions.
- Attributes go through `build_output_attrs` and `provenance_attributes`:
  - `climate_indices_candidates`
  - `climate_indices_fitting_methods` (a JSON object mapping distribution to method)
  - `climate_indices_criterion`
  - `climate_indices_log_likelihood_at = "fitted_estimates"`
  - `scale`
  - the clamped `calibration_year_initial` and `calibration_year_final`
  - `climate_indices_gof_bootstrap_samples`
  - `climate_indices_rng_seed` (decimal string, so seeds above 64 bits round-trip through NetCDF)
  - `climate_indices_version`
- Dimensional fitted parameters (`beta`, `loc`, `scale`) carry the sample's `units` string,
  unchanged for `sum` and `mean`, when it is known. For SPEI, `values` and `pet_mm` must
  carry the same string; otherwise the attribute is absent, never `"1"`. Every other fitted
  parameter is `"1"`. `fit_diagnostics()` follows the same rule (#1650).

### 7. Surface

- The new root module `climate_indices.suitability` is Beta (`BetaFeatureWarning`).
  It has no root re-export: users import it explicitly, as with the compound module.
- xarray/Dask follows {doc}`0001-dual-numpy-xarray-api` and {doc}`0003-dask-time-dimension-single-chunk`:
  time is in a single chunk.
- `drought_category` follows the USDM SPI/SPEI table with six classes: None and D0 to D4.
  Exact edges −0.5, −0.8, −1.3, −1.6 and −2.0 belong to D0, D1, D2, D3 and D4 respectively;
  each interval includes its upper edge and excludes its lower edge. Values above −0.5 are None.
  There is no wet-side classification in this decision; one can be added later.
  `output_scale` is a required argument with no default: only `"normal"` classifies;
  `"probability"` and `"bounded"` raise.
- Event concordance reuses `compound_events(mode="concurrent")` with reference-centric counting:
  - a hit is a reference event overlapped by at least one index event;
  - a miss is a reference event overlapped by none;
  - a false alarm is an index event overlapping no reference event;
  - index events overlapping at least one reference event are counted once, as matched index events, for the denominator of the false-alarm ratio.
  A minimum-overlap setting can be added later if it is needed.

## Consequences

- GEV adds one enum member, which affects exhaustive matches, the CLI choice list and the CF `distribution` attribute.
- Criteria at the library's estimates are not textbook AIC.
  The documentation, docstrings and attributes say that the ranking mixes distribution fit with estimator efficiency.
- Rust parity for GEV and an accelerated comparison loop are deferred, and would fall under {doc}`0017-rust-core-acceleration-backend`.
- The details taken from the motivating study stay unverified until its text is supplied.
  The reproduction ([#1591](https://github.com/monocongo/climate_indices/issues/1591)) is labelled a workflow reproduction, not a replication of its results.

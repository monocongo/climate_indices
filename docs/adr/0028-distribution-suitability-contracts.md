# Distribution-suitability assessment contracts

## Status

Approved for implementation. The maintainer approved these contracts.
[SUIT-01 (#1582)](https://github.com/monocongo/climate_indices/issues/1582) records this decision under
[epic #1581](https://github.com/monocongo/climate_indices/issues/1581).

**No suitability API is implemented yet.** This record approves contracts, not completed code or scientific validation.
Do not start implementation until the maintainer merges the PR containing this record.
The implementation responsibilities are explicit:

| Proposed API | Tracking issue | Current state |
| --- | --- | --- |
| `Distribution.gev` for `spei` (Hosking, L-moments) | [#1583](https://github.com/monocongo/climate_indices/issues/1583) | Not implemented. |
| `suitability.compare_distributions()` and `DistributionComparison`, NumPy core | [#1584](https://github.com/monocongo/climate_indices/issues/1584) | Not implemented. |
| `suitability.compare_distributions()`, xarray/Dask route | [#1585](https://github.com/monocongo/climate_indices/issues/1585) | Not implemented. |
| `GoodnessOfFit`: Anderson–Darling and parametric bootstrap, optional BH adjustment | [#1586](https://github.com/monocongo/climate_indices/issues/1586) | Not implemented. |
| `suitability.drought_category()` and `suitability.compare_categories()` | [#1587](https://github.com/monocongo/climate_indices/issues/1587) | Not implemented. |
| `suitability.pit_uniformity()` | [#1588](https://github.com/monocongo/climate_indices/issues/1588) | Not implemented. |
| `suitability.event_concordance()` and `suitability.category_agreement()` | [#1590](https://github.com/monocongo/climate_indices/issues/1590) | Not implemented. |
| Workflow reproduction notebook and "choosing a distribution" guide | [#1591](https://github.com/monocongo/climate_indices/issues/1591) | Not implemented. |

Amend this status and implementation table as those issues land.

## Context

Users must choose a distribution per index, region, season and Timescale.
The library reports only a K–S statistic per fit (`fit_diagnostics`), has no GEV,
and has no relative-fit, decision-impact or concordance tools.
A K–S test whose parameters were estimated from the same data is conservative
(Lilliefors 1967, as cited by the SciPy `goodness_of_fit` documentation): its p-values are too high,
so poor fits are accepted too easily.

The motivating study is Monish & Rehana (2020), doi:10.1007/s12040-019-1271-x.
It is closed access and has not been read for this record, so none of its figures or method details are quoted here.

## Decision

### 1. Candidate set

- The candidates are the likelihood-based families, and `candidates=None` selects them:
  gamma and Pearson Type III for `spi` and `standardized_index`;
  gamma, Pearson Type III, log-logistic and, once #1583 lands, GEV for `spei`.
  Naming a family outside this set (such as empirical or beta) in `candidates` raises.
- SPEI gains `Distribution.gev`: Hosking's GEV fitted by L-moments, with `fitting_params` keys
  `loc`, `scale` and `shape`. The shape is Hosking's k, equal to SciPy's `c`.
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
  - `climate_indices_rng_seed`
  - `climate_indices_version`

### 7. Surface

- The new root module `climate_indices.suitability` is Beta (`BetaFeatureWarning`).
  It has no root re-export: users import it explicitly, as with the compound module.
- xarray/Dask follows {doc}`0001-dual-numpy-xarray-api` and {doc}`0003-dask-time-dimension-single-chunk`:
  time is in a single chunk.
- `drought_category` follows the USDM SPI/SPEI table with six classes: None and D0 to D4.
  Edges are half-open at −0.5, −0.8, −1.3, −1.6 and −2.0.
  There is no wet-side classification in this decision; one can be added later.
  `output_scale` is a required argument with no default.
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

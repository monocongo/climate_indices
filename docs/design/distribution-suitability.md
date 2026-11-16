# Distribution-suitability assessment tools: design note

Internal planning document (`docs/design/` is excluded from the published docs).
Status: draft for maintainer review. The ADR-0028 text in section B is a draft;
SUIT-01 publishes it under `docs/adr/` as `Approved for implementation` once approved.
Inventory taken against `origin/main` at `aa076954` (2026-11-15).

## Where the repo overrides the task brief

1. **ADR status.** `docs/adr/` is published. CONTRIBUTING.md § *Architecture decision records* allows only `Approved for implementation`, `Accepted`, `Amended` and `Superseded`. There is no `Proposed` status.
   - Precedent: CMP-01 (#1365) kept the proposal inside the internal `docs/design/` plan, then published the ADR after approval.
   - So the ADR is drafted below as **Proposed (draft)** and lives in the design note. SUIT-01 publishes it as `docs/adr/0028-…` with status `Approved for implementation` once the maintainer approves it. 0028 is the next free number: the directory ends at 0027, and no PR or branch touches 0028.
2. **Plan location.** There is no `docs/plans/`. The repo uses the internal `docs/design/` folder (CONTRIBUTING.md lists it as internal). The note goes to `docs/design/distribution-suitability.md`.
3. **"Naive K–S is anti-conservative" is backwards.** The SciPy 1.17 `goodness_of_fit` notes say this, citing Lilliefors (1967): a K–S test whose parameters were fitted to the same data is *conservative*. Its p-values are too high, so it under-rejects and has low power. The practical effect matches what the task brief worries about: naive K–S accepts poor fits too easily. The docs will use the correct term.
4. **#1569 has no hit/miss/false-alarm conventions.** Its metrics are bias, MAE, RMSE, correlation, NRMSE, max-abs, `category_agreement` and `category_within_one`. They live in `scripts/earthengine_climate_engine_pilot.py`, not in the library. Event metrics are therefore new. Category agreement reuses the pilot's two definitions.

## Paper verification

Monish & Rehana (2020) is closed access. Springer serves a bot challenge, IAS returns 403, and OpenAlex, Crossref and Semantic Scholar carry no abstract. This note cites the paper **by DOI only**. Every figure and method detail taken from the task brief is listed under Open Questions as unverified, and none appears in any issue.

Verified sources used:
- The USDM classification page, for SPI/SPEI category ranges.
- The SciPy 1.17 docstrings for `goodness_of_fit`, `genextreme`, `anderson` and `false_discovery_control`.
- In-repo references: Stagge et al. (2015) is already cited in `cf_metadata_registry.py` and `xarray_adapter.py`; ADR-0016 covers GLO and `lmom`.

Stagge (2015) itself could not be fetched (Wiley 403), so the plan makes no new claim about its content.

## Section A. Design

### Goal

Give per-cell, per-calendar-step evidence for choosing a distribution. The evidence covers three things:
- **Relative fit:** information criteria.
- **Absolute fit:** goodness-of-fit tests with honest p-values.
- **Consequence:** whether the drought category changes, and whether events agree with an independent reference.

All of it should run on the NumPy core and on xarray/Dask, and carry provenance that earthflux can register without re-deriving anything.

### Inventory findings

All references are to `origin/main`.

| Area | Finding | Reuse |
|---|---|---|
| `Distribution` (`vocabulary.py:19`) | Members: `pearson`, `gamma`, `loglogistic`, `empirical`, `beta`, `t_location_scale`, `normal`. **No GEV** (grep for `gev\|genextreme` finds nothing). | Add `gev`. |
| Accepted sets | `spi`: gamma, pearson. `spei`: gamma, pearson, loglogistic (no gamma fallback). `standardized_index`: gamma, pearson, empirical, beta. `spai`: t_location_scale, pearson, normal. `fit_diagnostics`: gamma, pearson only. | Likelihood-based comparison candidates (the default `candidates=None`) are gamma/pearson for `spi` and `standardized_index`, and gamma/pearson/loglogistic (plus GEV once SUIT-02 lands) for `spei`. Empirical, beta and SPAI-only families are excluded. |
| Estimators | Gamma: Thom's MLE approximation (`compute.gamma_parameters:1988`; the code comment wrongly says "method of moments"). Pearson III: L-moments (`lmoments.fit`) on the non-zero values, at least 4 non-zero values, falls back to gamma; the 0.0005 trace rule is applied in the transform, not in the fit. Log-logistic: Hosking GLO with ub-pwm L-moments (`lmoments.fit_glo`, ADR-0016); SPEI adds a +1000 offset. Beta: method of moments. | Information criteria are evaluated at these estimates (see OQ-1). |
| `fit_diagnostics` (`compute.py:4096`, `drought/indices.py:850`, `xarray_adapter.py:3402`) | Frozen `FitDiagnostics`, per step and cell: `distribution`, `parameters`, `prob_zero`, `n_valid`, `ks_statistic`, exact `ks_p_value` (`scipy.stats.kstest`), `fell_back_to_gamma`. The xarray route returns a Dataset built with `apply_ufunc`. No AIC, BIC, log-likelihood or AD. | Its block-packing pattern, calibration slicing and `distribution_used` handling. |
| `output_scale` (`vocabulary.py:143`) | `"normal"`, `"probability"` (the CDF, i.e. the PIT) or `"bounded"`. Accepted by spi, spei, spai and standardized_index. | Seasonal PIT diagnostics take `"probability"` output from any index. |
| Empirical | One fixed plotting position, SEI `(1+k)/(n+2)`, needing at least 10 values. Tukey appears only in EDDI and in the #1576 pilot. | Not a candidate in parametric comparison (see out of scope). |
| SPAI (`drought/indices.py:1565`, Beta) | Chanda & Maity (2015). Subtracts the per-calendar-step calibration mean, then fits **one** pooled distribution per cell. No per-step variance scaling. | It is the pooled-fit comparator for the seasonality check. |
| Run theory (`runs.py`) | `identify_runs`, `identify_runs_xarray`, `RunSet`, `summarize_runs`. | Event sets for concordance. |
| Event matching | `compound.compound_events(first, second, *, time, valid, cadence, mode="concurrent")` pairs runs that overlap. | Concordance hit/miss/false-alarm counts are built on it. No new matcher. |
| Category metrics | No classifier in the library. The pilot has `usdm_bin` with 11 classes and `category_agreement`/`category_within_one`. | Lift the definitions into the library; the pilot script stays as it is. |
| CF/provenance | `build_output_attrs` (`xarray_adapter.py:1173`) applies CF attributes, calculation metadata, `climate_indices_version`, then `provenance.provenance_attributes` (a JSON record with sha256 `record_id`). No attribute names the fitting method. Gamma fallback is visible only in `fit_diagnostics`. | New Datasets go through the same path. |
| Rust (ADR-0017/0023) | Kernels exist for gamma, pearson, GLO, lmoments, empirical and runs. None for beta, t or normal. Parity is 1e-10. Rust is optional. | GEV kernel and comparison-loop acceleration are a follow-on, not part of this epic. |
| VALIDATION.md | Free-text Status vocabulary. No row for `fit_diagnostics` or beta. `validation.yaml` holds only command profiles. | Each child adds the row its evidence supports. |
| Docs | `docs/checking-distribution-fit.md` (K–S, PIT histogram, gamma vs Pearson); `docs/choosing-parameters.md`; `notebooks/check_spi_fit.ipynb`. | Grow `checking-distribution-fit.md` into the "choosing a distribution" guide. |
| Data | `scripts/prepare_e2e_inputs.py` builds an nClimGrid low-res sample: precip, pet and wb, 1980–2016, 38×87 grid. NODD terms allow a clearly labelled reduced sample (`docs/research/nclimgrid-acquisition-and-redistribution.md`). | Input for the workflow reproduction. |
| Stale text (separate small fix, not in this epic) | `drought/indices.py:863` and around `:734` say xarray is not wired, but it is. The gamma comment says "method of moments". | Offer as one `fix` issue. |

### Scope

**In scope:**
- GEV for SPEI.
- `compare_distributions` (NumPy and xarray).
- AD and parametric-bootstrap goodness of fit, plus multiple-testing summaries.
- Drought categories and category-change metrics.
- Seasonal PIT uniformity, including SPAI.
- Event concordance and category agreement.
- Provenance attributes.
- Workflow reproduction notebook.
- Guide and VALIDATION rows.

**Out of scope:**
- Rust acceleration (follow-on).
- GEV for `spi`/`standardized_index` (OQ-2).
- A pooled parametric fit for SPI/SPEI (OQ-6).
- Empirical or Tukey as comparison candidates (no likelihood).
- Field significance.
- Automatic per-cell distribution switching inside `spi()`/`spei()`.
- Claims of reproducing the paper's numbers.
- CLI flags.

### API sketch

Signatures only. Everything goes in a new root module `climate_indices/suitability.py`, imported as `import climate_indices.suitability as suitability`, with no root re-export. It ships as Beta with `BetaFeatureWarning`.

```python
Criterion = Literal["aicc", "aic", "bic"]
IndexRoute = Literal["spi", "spei", "standardized_index"]

@dataclass(frozen=True)
class DistributionComparison:
    candidates: tuple[Distribution, ...]
    criterion: Criterion
    n: np.ndarray                 # (step, *cell) shared sample size
    n_parameters: np.ndarray      # (candidate,)
    log_likelihood: np.ndarray    # (candidate, step, *cell), at the library's estimates
    aic: np.ndarray; aicc: np.ndarray; bic: np.ndarray
    best: np.ndarray              # (step, *cell) index into candidates; -1 = none eligible (fitted with a defined criterion)
    delta_to_runner_up: np.ndarray
    fit_failed: np.ndarray        # (candidate, step, *cell) bool
    parameters: Mapping[Distribution, Mapping[str, np.ndarray]]  # reusable as fitting_params
    goodness_of_fit: GoodnessOfFit | None
    metadata: Mapping[str, Any]   # provenance (ADR-0028 §6)

def compare_distributions(
    values, scale, *, index: IndexRoute, candidates: Sequence[Distribution] | None = None,
    pet_mm=None, aggregation: Aggregation = "sum", clip_negatives: bool = True,
    data_start_year=None, calibration_year_initial=None,
    calibration_year_final=None, periodicity=None, criterion: Criterion = "aicc",
    gof: Sequence[Literal["ks", "ad"]] = ("ks", "ad"), n_bootstrap: int = 0,
    bh_adjust: bool = False, alpha: float = 0.05,
    seed: int | None = None, spatial_time_major=False,
    time_dim="time",
) -> DistributionComparison | xr.Dataset

@dataclass(frozen=True)
class GoodnessOfFit:
    ks_statistic: np.ndarray
    ks_p_value_naive: np.ndarray
    ad_statistic: np.ndarray
    ks_p_value_bootstrap: np.ndarray | None
    ad_p_value_bootstrap: np.ndarray | None
    ks_p_value_bh: np.ndarray | None      # BH-adjusted across cells; None unless bh_adjust
    ad_p_value_bh: np.ndarray | None
    rejection_share: np.ndarray | None    # (candidate, step, statistic): share of cells with p < alpha
    n_bootstrap: int

def drought_category(index_values) -> np.ndarray | xr.DataArray   # int8: 0 none, 1..5 = D0..D4, -1 missing
def compare_categories(reference, alternative, *, dims=None) -> CategoryComparison | xr.Dataset
    # confusion (6x6), n_valid, share_changed, share_changed_in_drought

def pit_uniformity(probabilities, *, data_start_year=None, calibration_year_initial=None,
                   calibration_year_final=None, periodicity=None,
                   statistics=("ks", "ad")) -> PitUniformity | xr.Dataset   # per calendar step

def event_concordance(index_runs: RunSet, reference_runs: RunSet, *, time, valid,
                      cadence) -> EventConcordance   # hits, misses, false_alarms, pod, far, csi
def category_agreement(index_categories, reference_categories) -> tuple[float, float]  # exact, within-one
```

**The SPEI route.** When `index="spei"`, the function requires `pet_mm` and applies the same water balance and offset as `spei()`. This guarantees that the `fitting_params` the comparison returns reproduce `spei(..., distribution=d)` exactly.

**The `standardized_index` route.** `aggregation` and `clip_negatives` are applied exactly as `standardized_index()` applies them, before any candidate is fitted, so the compared series is the series the index would standardize. Both options are accepted only with `index="standardized_index"`. A signed series (`clip_negatives=False`) is outside the gamma and Pearson Type III candidates, as it is for `standardized_index()` itself, so that combination raises rather than score a different series.

### Guardrails

- **Shared sample.** Every candidate is scored on the identical calibration sample: same cells, same steps, same zero set. A zero is an exact zero, the definition both existing estimators fit with; the 0.0005 trace rule is a Pearson transform convention and does not change the fitted sample (OQ-3).
- **Undefined AICc.** Where n ≤ k + 1, AICc is undefined and the result is NaN. The minimum-sample rules already in place (Pearson ≥ 4 non-zero values) still apply. A short-record warning fires when n < 30, the repo's 30-year Calibration Period minimum.
- **Default criterion is AICc.** AICc = AIC + 2k(k+1)/(n−k−1). The default is AICc because about 30 values per calendar month is the norm.
- **`best` is recorded with a margin.** The result carries `delta_to_runner_up` and applies no threshold. The guide explains that small deltas do not discriminate.
- **Dependence at Timescale > 12.** The same calendar month in consecutive years shares months once the Timescale exceeds 12, so GoF p-values and effective n are invalid there. Warn when scale > 12 on monthly data. The equivalent rule applies to daily data.
- **Multiple testing.**
  - Report the rejection share against nominal α across cells (`GoodnessOfFit.rejection_share`, at `alpha`).
  - Offer BH adjustment through `scipy.stats.false_discovery_control`, opt-in only (`bh_adjust=True`, which needs a cell axis and fills `ks_p_value_bh`/`ad_p_value_bh`).
  - Document that spatial correlation weakens the BH assumptions. Field significance is out of scope.
- **Post-selection.** The GoF p-value of the AIC-selected candidate is a post-selection statistic, and the docs say so.
- **Bootstrap.** It refits with **the library's own estimator**. `scipy.stats.goodness_of_fit` refits by MLE, so it serves only as a cross-check for MLE-like gamma. The default is `n_bootstrap=0`. The cost (B × candidates × steps × cells) is documented, and the seed is recorded.

### Validation plan

| Item | Evidence | Expected VALIDATION status |
|---|---|---|
| GEV | CDF/PPF parity with `scipy.stats.genextreme` on both bounded branches. Hosking's k equals SciPy's c in sign; the test pins this. L-moment recovery against closed-form population L-moments. Optional R `lmom::pelgev/cdfgev` fixture (OQ-8). | "Cross-implementation comparison, same estimator lineage" with the lmom fixture, otherwise "Specification-level" |
| Log-likelihood and criteria | `sum(logpdf)` checked against SciPy (`gamma`, `pearson3`, `genextreme`; GLO through `fisk` on the k < 0 branch, per ADR-0016) at the same parameters. Round trip: comparison parameters fed as `fitting_params` to `spi`/`spei` give identical output. Synthetic recovery rate reported, not asserted tightly. | Specification-level plus synthetic characterization |
| Bootstrap GoF | Size under H0 within a binomial MC interval of α for every candidate that supports the bootstrap (gamma, Pearson Type III, GLO, GEV), on continuous and rounded samples. Naive K–S shown to under-reject. | Synthetic characterization |
| Categories | Edges match the USDM table ("-0.5 to -0.79" … "-2.0 or less"), including exact boundary values. | Specification-level |
| Concordance | Hand-built `RunSet` fixtures. | Specification-level |
| Reproduction | nClimGrid workflow. | "Workflow reproduction; not reference reproduction" |

### Risks

- The criteria are penalized likelihoods at the library's own estimates (Thom's approximation, L-moments), not textbook AIC/AICc/BIC. A ranking therefore mixes how well a family fits with how efficient its estimator is, and can favour a family because of its estimator (OQ-1).
- n of about 30 per month gives weak discrimination, so maps of "best" may be noise. Mitigation: report the delta, the decision-impact metrics and the GoF share.
- GEV upper-bounded fits put an index value at ±∞ beyond the support. The existing ±3.09 normal-scale clip covers it; the probability scale returns 0 or 1.
- Bootstrap cost on CONUS grids. Default off; Rust follow-on.
- Users may over-read "best" as "correct". The guide, docstrings and provenance carry the caveats.

### Open questions

- **OQ-1.** Evaluate log-likelihood at the library's own estimates (Thom, L-moments) or refit by MLE for the criteria? *Recommend the library estimates*, because those are the parameters the index actually uses. Record `log_likelihood_at = "fitted_estimates"` and each fitting method, and state in the docs and docstrings that the ranking mixes distribution fit with estimator efficiency. The criteria keep the `aic`/`aicc`/`bic` names; the caveat travels with them. An MLE option can come later.
- **OQ-2.** GEV for `standardized_index`/`spi`? They carry a physical zero mass, and GEV with ξ > 0 has support below 0. *Recommend SPEI only*, following the ADR-0016 precedent.
- **OQ-3.** Common zero definition for gamma against Pearson (exact zeros vs below 0.0005)? *Recommend retaining exact zeros as the shared mixture definition*, because Pearson's trace rule is a transform convention; if traces are reclassified, document that the fitted samples and scores change.
- **OQ-4.** Concordance counting with many-to-many overlaps? *Recommend reference-centric counting*: hit = a reference event overlapped by at least one index event; miss = one overlapped by none; false alarm = an index event overlapping no reference event. Expose a minimum-overlap setting later only if needed.
- **OQ-5.** Which independent reference does the reproduction use (USDM maps, a satellite index, soil-moisture percentiles), and on what licence? Until it is resolved, the notebook shows concordance on a labelled demonstration pairing.
- **OQ-6.** Is a pooled (all-months) parametric SPI/SPEI fit needed beyond SPAI? *Recommend no*: SPAI is the pooled comparator.
- **OQ-7.** Explicit `suitability` module import during Beta rather than root re-export? *Recommend explicit import*, as compound does.
- **OQ-8.** Add an R `lmom` GEV fixture (an offline generator, as in #1195)?
- **OQ-9.** Unverified details from the task brief, pending the paper: the candidate set, the K–S level and its per-zone use, AIC per grid point, the 83%/72% (Pearson III at 3/6 months) and 50%/59% (GEV at 12/24 months) figures, the role of SPAI, and the satellite index and its concordance method. Supply a PDF to verify them.
- **OQ-10.** The `drought_category` wet side: USDM defines only None and D0–D4, while the pilot uses 11 symmetric bins. *Recommend the 6 USDM classes*. A wet classification can be added later.

### Work breakdown (dependency order)

```
SUIT-01 ADR ─┬─ SUIT-02 GEV ─────────────────────────┐
             ├─ SUIT-03 compare (NumPy) ─ SUIT-04 xarray ─┤
             │        └─ SUIT-05 GoF ─ SUIT-07 PIT ───────┤
             └─ SUIT-06 categories ─ SUIT-08 concordance ─┴─ SUIT-09 reproduction + guide
```

Follow-on, not part of this epic: a Rust GEV kernel and comparison-loop acceleration under ADR-0017 parity.

---

## Section B. ADR-0028 draft (draft; published by SUIT-01 as `Approved for implementation`)

**Title:** Distribution-suitability assessment contracts.

**Status:** Proposed (draft). It is published as `Approved for implementation` after maintainer approval, and lists the SUIT issues as unimplemented APIs.

**Context.** Users must choose a distribution per index, region, season and Timescale. The library reports only K–S per fit (`fit_diagnostics`), has no GEV, and has no relative-fit, decision-impact or concordance tools. Naive K–S with estimated parameters is conservative (Lilliefors 1967, as cited by SciPy).

**Decision:**
1. **Candidate set.**
   - The candidates are the likelihood-based families, and `candidates=None` selects them: gamma and Pearson Type III for `spi` and `standardized_index`; gamma, Pearson Type III, log-logistic and, once SUIT-02 lands, GEV for `spei`. Naming a family outside this set (such as empirical or beta) in `candidates` raises.
   - SPEI gains `Distribution.gev`: Hosking's GEV fitted by L-moments, with `fitting_params` keys `loc`, `scale` and `shape`. The shape is Hosking's k, equal to SciPy's `c`.
   - All values are fitted with no zero mass, as for GLO.
   - `spi`, `standardized_index` and `fit_diagnostics` reject GEV.
   - Empirical, beta and the SPAI families are not comparison candidates.
2. **Estimators.** The comparison uses each family's existing estimator: gamma by Thom's MLE approximation; Pearson III, GLO and GEV by L-moments. Log-likelihood is evaluated at those estimates and labelled as such. Because these are not maximum-likelihood estimates in general, the criteria are penalized likelihoods at fitted estimates, and a ranking mixes distribution fit with estimator efficiency. MLE refitting is not part of this decision.
3. **Criteria.** With k free parameters, plus 1 for p0 when a zero mass is fitted (the same for all candidates sharing it):
   - AIC = 2k − 2ℓ
   - AICc = AIC + 2k(k+1)/(n−k−1), NaN when n ≤ k + 1
   - BIC = k·ln n − 2ℓ
   - Candidates are scored on one shared sample with one zero definition.
4. **"Best".** `best` is the argmin of the chosen criterion (default `aicc`) over candidates that fitted and have a defined criterion. Ties go to the earlier candidate in canonical enum order. The value is −1 when no candidate is eligible; `delta_to_runner_up` is NaN when fewer than two candidates are eligible. It is returned whenever it is defined, and no significance threshold is applied.
5. **Goodness of fit.**
   - K–S is computed with naive p-values, labelled as naive.
   - The Anderson–Darling statistic is computed from the PIT.
   - An optional parametric bootstrap, which refits with the library's own estimator, gives p-values for both.
   - BH adjustment is optional and uses SciPy.
6. **Result types and provenance.**
   - The NumPy routes return frozen dataclasses. The xarray routes return Datasets with a `candidate` dimension (string labels), a `month`/`dayofyear` dimension and the cell dimensions.
   - Attributes go through `build_output_attrs` and `provenance_attributes`:
     - `climate_indices_candidates`
     - `climate_indices_fitting_methods` (a JSON object mapping distribution to method)
     - `climate_indices_criterion`
     - `climate_indices_log_likelihood_at = "fitted_estimates"`
     - `scale`
     - the clamped `calibration_year_initial`/`calibration_year_final`
     - `climate_indices_gof_bootstrap_samples`
     - `climate_indices_rng_seed`
     - `climate_indices_version`
7. **Surface.**
   - New root module `climate_indices.suitability`, Beta (`BetaFeatureWarning`), with no root re-export.
   - xarray/Dask follows ADR-0001 and ADR-0003 (time in a single chunk).
   - Category edges follow the USDM SPI/SPEI table, half-open at −0.5, −0.8, −1.3, −1.6 and −2.0.
   - Event concordance reuses `compound_events(mode="concurrent")`.

**Consequences.**
- GEV adds one enum member, which affects exhaustive matches, the CLI choice list and the CF `distribution` attribute.
- Criteria at the library's estimates are not textbook AIC; the docs, docstrings and attributes say the ranking mixes distribution fit with estimator efficiency.
- Rust parity for GEV is deferred.

---

## Section C. Issue drafts

The prefix `SUIT-` matches the `CMP-` precedent. The bodies use the Feature_request headings plus a **Scientific core** section, which the `drive-issue-to-pr` triage step consumes.

Shared footer, used in every child:

> The canonical plan is `docs/design/distribution-suitability.md`. The accepted record will be `docs/adr/0028-distribution-suitability-contracts.md` after SUIT-01 merges. Do not start until SUIT-01 has merged and the listed blockers are closed. Add no dependency.

### SUIT-00 — [EPIC] Distribution-suitability assessment tools
**Labels:** `enhancement`, `epic`, `design`, `ready-for-human`

**Is your feature request related to a problem? Please describe.** Choosing gamma, Pearson III, log-logistic or GEV per region, season and Timescale is ad hoc. The library reports only a naive K–S p-value for gamma and Pearson, and nothing tells a user whether the choice changes drought categories or events.

**Describe the solution you'd like.** Deliver the contracts in ADR-0028 through SUIT-01 to SUIT-09:
- [ ] SUIT-01 ADR
- [ ] SUIT-02 GEV
- [ ] SUIT-03 compare (NumPy)
- [ ] SUIT-04 compare (xarray)
- [ ] SUIT-05 GoF
- [ ] SUIT-06 categories
- [ ] SUIT-07 PIT
- [ ] SUIT-08 concordance
- [ ] SUIT-09 reproduction and guide

**Describe alternatives you've considered.**
- Extending `fit_diagnostics` in place would change a Stable return type.
- A Rust-first comparison loop would put speed ahead of contracts that have no evidence yet.

**Acceptance criteria:**
- [ ] The maintainer merges the ADR before implementation starts.
- [ ] Every child merges with its tests, docs and VALIDATION row.
- [ ] Provenance attributes match ADR-0028 §6.
- [ ] The reproduction is labelled a workflow reproduction.

**Additional context:** Motivating study: Monish & Rehana (2020), doi:10.1007/s12040-019-1271-x. Its figures are unverified and not quoted. The earthflux atlas is a separate epic, and the library has no earthflux dependency. Follow-on, not part of this epic: Rust GEV and comparison kernels.

### SUIT-01 — Record the distribution-suitability plan and ADR-0028
**Labels:** `documentation`, `design` · **Blocked by:** none · **Size:** S · **Branch:** `docs/distribution-suitability`

**Problem.** Estimator, criterion, zero-sample and provenance choices must be fixed before the implementation issues can proceed independently.

**Solution.**
- Merge `docs/design/distribution-suitability.md`.
- Resolve OQ-1 to OQ-4, OQ-7 and OQ-10.
- Publish `docs/adr/0028-…` as `Approved for implementation`, with links to the SUIT issues, and add it to `docs/adr/index.md`.

**Acceptance criteria:**
- [ ] The plan and the ADR agree on every contract.
- [ ] The ADR lists unimplemented APIs with issue links.
- [ ] Both Sphinx gates pass.
- [ ] The PR adds no code, fixtures, CF entries or validation claims.

### SUIT-02 — Add GEV (Hosking, L-moments) as an SPEI distribution
**Labels:** `enhancement`, `numpy-api`, `xarray-api`, `validation`, `tests` · **Blocked by:** SUIT-01 · **Size:** M

**Problem.** GEV is a common SPEI candidate in the suitability literature but cannot be fitted here.

**Solution.**
- Add `Distribution.gev`, `lmoments.fit_gev` (Hosking's L-moment estimator using the existing SAMLMR L-moments), `compute.gev_parameters` and `transform_fitted_gev`.
- `spei()` accepts it on NumPy, xarray/Dask and the CLI choice list.
- `spi`, `standardized_index` and `fit_diagnostics` reject it.

**Scientific core:**
- **Expected:** CDF F(x) = exp(−(1 − k(x−ξ)/α)^{1/k}), with the Gumbel limit at k = 0.
- **Edge cases:**
  - k → 0, which needs a series or limit branch; tolerance to be justified in the PR.
  - Values outside the support on either bounded branch: F = 0 or 1, giving index ±∞, then the ±3.09 clip on the normal scale.
  - τ3 outside Hosking's valid range, which is a fit failure (NaN, warning). There is no gamma fallback, matching SPEI.
  - All-NaN cells.
  - n < 3.
- **Precision:** f64. Parity with `scipy.stats.genextreme.cdf` at `rtol=1e-12` (or tighter if measured), with Hosking's k pinned to SciPy's `c`.
- **Reference:** Hosking's L-moment GEV. An `lmom` fixture is optional (OQ-8).

**Acceptance criteria:**
- [ ] CDF/PPF parity tests on both branches and at k ≈ 0.
- [ ] L-moment recovery test on synthetic GEV samples.
- [ ] `fitting_params` round trip: `loc`, `scale`, `shape`.
- [ ] xarray/Dask test with time in a single chunk.
- [ ] The CF `distribution` attribute equals `gev`.
- [ ] `choosing-parameters.md` and the glossary updated.
- [ ] VALIDATION row at the level the evidence supports.
- [ ] llms bundles regenerated.

### SUIT-03 — Add `suitability.compare_distributions` (NumPy core)
**Labels:** `enhancement`, `numpy-api`, `tests` · **Blocked by:** SUIT-01 · **Size:** L

**Problem.** No per-cell relative-fit evidence (log-likelihood, AIC, AICc, BIC, best) exists.

**Solution.**
- Add the `DistributionComparison` dataclass and the NumPy `compare_distributions` described in the plan's API sketch. It covers the `spi`, `spei` and `standardized_index` routes with the default candidate sets of ADR §1, scores every candidate on a shared sample, and returns parameters that can be reused as `fitting_params`.
- GEV appears as a candidate automatically once SUIT-02 has landed.
- Emits `BetaFeatureWarning`.

**Scientific core:**
- **Expected:** ℓ equals the sum of SciPy `logpdf` values at the library's parameters, plus the p0 mixture term.
- **Edge cases:**
  - A zero mass shared across candidates (OQ-3).
  - Pearson fallback is disabled inside the comparison: a failure is recorded, not substituted.
  - n ≤ k + 1, where AICc is NaN.
  - Cells where no candidate is eligible, giving `best = -1`: every fit failed, or every fitted candidate has an undefined criterion (for example gamma-only with n ≤ k + 1).
  - The `n < 30` warning and the `scale > 12` dependence warning.
  - NaN cells.
- **Precision:** criteria are compared to SciPy at `rtol=1e-10`.

**Acceptance criteria:**
- [ ] Log-likelihood checked against SciPy for gamma, `pearson3` and GLO-via-`fisk` (k < 0).
- [ ] Round trip: `spi()`/`spei()` with the returned params are bit-identical to direct calls.
- [ ] Synthetic recovery-rate characterization test.
- [ ] Tie and `best` semantics tested.
- [ ] Metadata dict per ADR §6.
- [ ] Google-style docs.
- [ ] VALIDATION row (specification-level).

### SUIT-04 — xarray/Dask route for `compare_distributions`
**Labels:** `enhancement`, `xarray-api`, `tests` · **Blocked by:** SUIT-03 · **Size:** M

**Solution.**
- Returns a Dataset with dimensions (`candidate`, `month|dayofyear`, cells) and variables for log-likelihood, AIC, AICc, BIC, best (as a string label), delta and the parameters.
- Uses the `fit_diagnostics` `apply_ufunc` pattern, infers calibration years from the time coordinate, and applies CF and provenance through `build_output_attrs`.

**Acceptance criteria:**
- [ ] NumPy and xarray results are equal.
- [ ] Dask spatial chunking produces results identical to the in-memory route; a multi-chunk time axis raises (ADR-0003).
- [ ] Every ADR §6 attribute is present.
- [ ] A Dataset round trip through NetCDF keeps the attributes.
- [ ] Docs example with doctest.

### SUIT-05 — Anderson–Darling and parametric-bootstrap goodness of fit
**Labels:** `enhancement`, `numpy-api`, `xarray-api`, `validation`, `tests` · **Blocked by:** SUIT-04 · **Size:** L (may split into AD and bootstrap)

**Problem.** Naive K–S with estimated parameters is conservative. Its p-values are too high, so it under-rejects (Lilliefors 1967, as cited by SciPy's `goodness_of_fit` docs).

**Solution.**
- Add `GoodnessOfFit` to the comparison: K–S statistic with its naive p-value (labelled), the AD statistic from the PIT, and an optional `n_bootstrap` parametric bootstrap that refits with the library estimator. Each bootstrap sample is drawn from the fitted mixed distribution, including p0.
- Add the rejection-share summary (`rejection_share`, at `alpha`) and opt-in BH adjustment (`bh_adjust=True`, through `scipy.stats.false_discovery_control`, giving `ks_p_value_bh` and `ad_p_value_bh`).
- Take `seed: int | None`. When it is None a seed is drawn from `numpy.random.SeedSequence().entropy`, and the seed actually used is recorded in `climate_indices_rng_seed`. A caller-supplied `Generator` is not accepted: it does not expose its original seed, so the stream could not be reproduced.

**Scientific core:**
- **Edge cases:**
  - Bootstrap refits that fail are counted and excluded, and their share is reported.
  - The p-value is (1 + #≥)/(B_ok + 1), where B_ok is the number of successful refits, so failed draws leave the denominator as well as the statistics. The p-value is NaN when B_ok = 0, and its floor is 1/(B_ok + 1).
  - Exact zeros are excluded from the continuous statistics, as `fit_diagnostics` does today (it filters missing values and exact zeros only). Repeated positive observations are retained with their frequency. Bootstrap draws are rounded to the record's resolution so the observed and simulated samples share the same tie structure, and the p-values are calibrated against that tie-aware null.
- **Determinism:** the same seed gives an identical result.

**Acceptance criteria:**
- [ ] Size under H0 within a binomial MC interval of α for gamma, Pearson Type III, GLO and GEV, including a rounded (tied) record.
- [ ] Test showing naive K–S under-rejecting.
- [ ] Cross-check of AD and bootstrap against `scipy.stats.goodness_of_fit` for gamma (distribution-level agreement only).
- [ ] Cost benchmark noted in `docs/performance.md`.
- [ ] VALIDATION row (synthetic characterization).

### SUIT-06 — Drought categories and category-change metrics
**Labels:** `enhancement`, `numpy-api`, `xarray-api`, `tests` · **Blocked by:** SUIT-01 · **Size:** M

**Problem.** "Which distribution fits best" does not tell managers whether the answer changes.

**Solution.**
- `drought_category()`: USDM SPI/SPEI ranges, giving None and D0 to D4, verified against the USDM classification table.
- `compare_categories()`: a 6×6 confusion matrix, `share_changed`, `share_changed_in_drought` and per-category counts, for NumPy and xarray.

**Scientific core:**
- **Edges:** −0.5 → D0; −0.8 → D1; −1.3 → D2; −1.6 → D3; −2.0 → D4; −0.49 → None.
- **Edge cases:**
  - NaN gives −1 and is excluded from the denominators.
  - Probability-scale input is rejected; a clipped ±3.09 value classifies normally.

**Acceptance criteria:**
- [ ] Boundary-value tests.
- [ ] Hand-computed confusion fixture.
- [ ] xarray reduction over the requested dimensions.
- [ ] Attributes record the edge set and its source.
- [ ] VALIDATION row (specification-level).

### SUIT-07 — Seasonal PIT uniformity diagnostics (incl. SPAI)
**Labels:** `enhancement`, `numpy-api`, `xarray-api`, `tests` · **Blocked by:** SUIT-05 · **Size:** M

**Problem.** Per-calendar-month fitting and SPAI's pooled fit represent seasonality differently. There is no index-agnostic check of which one is calibrated per month.

**Solution.** `pit_uniformity()` takes any `output_scale="probability"` output (spi, spei, spai, standardized_index) and reports per-calendar-step K–S and AD against U(0, 1) over the Calibration Period, reusing the SUIT-05 AD code.

**Scientific core:**
- **Edge cases:**
  - Point masses at p0 are excluded from the continuous statistics, and their share is reported.
  - In-sample PIT is conservative, and the docs say so.
  - Scale > 12 dependence warning.

**Acceptance criteria:**
- [ ] Uniform synthetic input passes at the nominal rate.
- [ ] A deliberately misspecified fit, with a seasonal variance cycle given to SPAI, is detected.
- [ ] NumPy/xarray equality.
- [ ] Docs.
- [ ] VALIDATION row.

### SUIT-08 — Event concordance and category agreement against a reference
**Labels:** `enhancement`, `numpy-api`, `tests` · **Blocked by:** SUIT-06 · **Size:** M

**Solution.**
- `event_concordance()` uses `compound_events(mode="concurrent")` and reference-centric counting (OQ-4) to give hits, misses, false alarms, POD = hits/(hits+misses), FAR = FA/(hits+FA) and CSI = hits/(hits+misses+FA).
- `category_agreement()` returns the exact and within-one shares, with the same definitions as the #1569 pilot's `category_agreement` and `category_within_one`.

**Scientific core:**
- **Edge cases:**
  - Zero events on either side gives NaN ratios, not 0.
  - Runs crossing invalid steps are rejected, as `compound_events` does.
  - Censored runs at the record edges are counted and flagged.

**Acceptance criteria:**
- [ ] Hand-built `RunSet` fixtures cover many-to-many overlaps.
- [ ] Pilot-definition parity test.
- [ ] Docs.
- [ ] VALIDATION row (specification-level).

### SUIT-09 — Workflow reproduction notebook and "choosing a distribution" guide
**Labels:** `documentation`, `type:notebook`, `validation` · **Blocked by:** SUIT-02, SUIT-04, SUIT-05, SUIT-06, SUIT-07, SUIT-08 · **Size:** M

**Solution.**
- A notebook on the prepared nClimGrid low-res sample (1980–2016, calibration 1981–2010) covering SPEI-3/6/12/24 with gamma, Pearson, log-logistic and GEV:
  - shares of best by AICc per scale;
  - GoF rejection share against nominal;
  - category change between the top two candidates;
  - seasonal PIT for SPEI against SPAI;
  - concordance on a labelled demonstration reference (OQ-5).
- Grow `docs/checking-distribution-fit.md` into the "choosing a distribution" guide, with guardrails and caveats, linked from `choosing-parameters.md`.

**Acceptance criteria:**
- [ ] The notebook executes in CI's notebook path (or is documented as manual, matching `check_spi_fit.ipynb`).
- [ ] No claim of reproducing the paper.
- [ ] VALIDATION entry "Workflow reproduction; not reference reproduction".
- [ ] Both Sphinx gates pass.
- [ ] llms bundles regenerated.


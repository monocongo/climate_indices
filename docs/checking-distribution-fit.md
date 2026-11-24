# Choose a distribution and check the fit

SPI and SPEI standardize a value against a fitted distribution. The choice of
family rarely has one answer for a whole domain, a season and a Timescale, and
the library exposes the evidence per calendar step and cell instead of deciding
for you. This guide covers the questions in order: which candidate fits best
relatively, whether any candidate fits at all, whether the choice changes the
drought category a user sees, whether the seasonal probabilities are uniform,
and whether the events agree with an independent reference.

Start with {doc}`choosing-parameters` for the distribution argument itself and
for the calibration period. The executable
[nClimGrid workflow notebook](https://github.com/monocongo/climate_indices/blob/main/notebooks/check_distribution_suitability.ipynb)
runs the comparison, goodness-of-fit, category, seasonal-uniformity and
event-concordance steps on the prepared sample. That is a workflow reproduction
and not a reference reproduction. Prepare the sample once with
`uv run --group dev scripts/prepare_e2e_inputs.py` (about 15 MB on the first
run) and start the notebook kernel in `notebooks/`.

The suitability tools are Beta (ADR-0028) and have no root re-export: import
`climate_indices.suitability` explicitly. A minor release can change their
interface.

## Check a single fit with fit_diagnostics

Use `fit_diagnostics()` with the **same input, Timescale, distribution, and
Calibration Period** as `spi()`. For a gridded xarray input it returns a Dataset
with a `month` (or `dayofyear`) axis and the input's cell axes for most
variables. `distribution_used` has only cell axes; it, `ks_p_value`, `n_valid`,
and `prob_zero` help identify fits worth inspecting. Request `output_scale="probability"` from `spi()` to view its
probability-scale (PIT) output.

```python
from climate_indices import fit_diagnostics, spi
from climate_indices.drought.indices import Distribution

# precip: monthly xarray.DataArray with (time, lat, lon) and units="mm"
options = dict(
    scale=3,
    distribution=Distribution.gamma,
    calibration_year_initial=1981,
    calibration_year_final=2010,
)
fit = fit_diagnostics(precip, **options)
pit = spi(precip, **options, output_scale="probability")
may_need_review = (fit.ks_p_value.sel(month=7) < 0.05) & (fit.n_valid.sel(month=7) > 0)
```

Inspect a map of `may_need_review`, compare gamma and Pearson using identical
inputs, and plot a histogram of finite `pit.sel(time=slice("1981", "2010"))`
values at a representative cell. A roughly uniform histogram is an informal
check of the fitted probabilities, **not** an independent validation result.
Exact zeros (and Pearson trace values below 0.0005) usually create a
probability mass at `p0` with classic zero placement; gamma resets an all-zero
calibration step's `p0` to 0, and Pearson
support-limit masking can override the score. A spike is expected where
`prob_zero` is high.
The KS test excludes zero and missing calibration values; `n_valid` counts the
non-zero values entering that test. Both distributions fit those same non-zero
values, so the p-value assesses the fit's own estimation sample. A `p < 0.05`
flags a fit to investigate, not a calibrated hypothesis test or proof that
Pearson is better.
Check `distribution_used` before attributing a Pearson result to Pearson: a
failed Pearson fit can fall back to gamma for a spatial block. Missing p-values
mean the KS test could not assess the fit.

## Compare candidates with compare_distributions

`suitability.compare_distributions()` scores every likelihood-based candidate on
one shared calibration sample. `index="spi"` and `index="standardized_index"`
compare Pearson Type III and gamma; `index="spei"` also compares the
log-logistic and GEV. Name `candidates` to narrow the set; the result always
follows the canonical enum order, so the input order has no effect. Any other
family raises. The function prepares the series exactly as the named index
does, so `index="spei"` requires `pet_mm` and applies the water balance and the
`+1000` offset.

```python
import climate_indices.suitability as suitability
from climate_indices import compute

comparison = suitability.compare_distributions(
    precip,
    3,
    index="spei",
    pet_mm=pet,
    calibration_year_initial=1981,
    calibration_year_final=2010,
)
comparison["best"]                 # winning candidate name, or "none"
comparison["delta_to_runner_up"]   # AICc gap to the second-best candidate
comparison["fit_failed"]           # True where a family could not be fitted or scored
```

A NumPy call requires the temporal arguments instead of inferring them:

```python
comparison = suitability.compare_distributions(
    precip_values,
    3,
    index="spei",
    pet_mm=pet_values,
    data_start_year=1980,
    periodicity=compute.Periodicity.monthly,
    calibration_year_initial=1981,
    calibration_year_final=2010,
)
```

Read the ranking with these limits:

- **The criteria are not textbook AIC/AICc/BIC.** They are penalized likelihoods
  at the library's own estimates (Thom's closed-form gamma approximation and
  L-moments), not at maximum-likelihood estimates. A ranking mixes how well a
  family fits with how efficient its estimator is, so it can favour a family
  because of its estimator (ADR-0028, decision 2).
- **The default criterion is AICc.** About 30 values per calendar month is the
  common case, and AICc is NaN where `n <= k + 1`. Choose `criterion="aic"` or
  `"bic"` only when you can say why.
- **`best` carries no significance threshold.** `delta_to_runner_up` reports the
  gap, and a small gap does not separate two candidates. Ties go to the earlier
  candidate in the canonical order (`pearson`, `gamma`, `loglogistic`, `gev`),
  and `best` is `-1` on the NumPy route (the string `"none"` on the xarray
  route) when no candidate is eligible.
- **A failed fit is recorded, never replaced.** The Pearson-to-gamma fallback
  that `spi()` applies is disabled here, so read `fit_failed` instead of
  assuming gamma when Pearson fails.
- **Support limits.** A sample value outside a candidate's fitted support gives
  zero likelihood, so that candidate fails and cannot rank. This matters for a
  GEV bounded above. A GEV with shape above 1 has an unbounded density at its
  support end, where likelihood methods are not regular, so treat its criteria
  with caution there.
- **Overlapping windows.** Above a Timescale of 12 months, or 365 days, the same
  calendar step in consecutive years shares window mass. The call warns, and
  the goodness-of-fit p-values are invalid there.
- **Reuse the parameters.** On the NumPy route, `comparison.parameters` is a
  dict that maps each candidate to arguments `fitting_params` accepts, so a
  saved fit reproduces the index exactly instead of being refitted.

## Check the absolute goodness of fit

AICc ranks candidates against each other and says nothing about whether any of
them fits. `gof=("ks", "ad")` (the default) adds the Kolmogorov-Smirnov D with
its **naive** p-value and the Anderson-Darling A^2 of each candidate's
probability integral transform, over the calibration sample:

```python
# xarray route: the fields arrive as Dataset variables.
comparison = suitability.compare_distributions(
    precip,
    3,
    index="spei",
    pet_mm=pet,
    calibration_year_initial=1981,
    calibration_year_final=2010,
    gof=("ks", "ad"),
)
comparison["ks_statistic"]
comparison["ks_p_value_naive"]
comparison["ad_statistic"]

# NumPy route: the fields arrive in a GoodnessOfFit dataclass.
comparison = suitability.compare_distributions(
    precip_values,
    3,
    index="spei",
    pet_mm=pet_values,
    data_start_year=1980,
    periodicity=compute.Periodicity.monthly,
    calibration_year_initial=1981,
    calibration_year_final=2010,
    gof=("ks", "ad"),
)
comparison.goodness_of_fit.ks_statistic
comparison.goodness_of_fit.ks_p_value_naive
comparison.goodness_of_fit.ad_statistic
```

Defaults and limits:

- **The naive K-S p-value is uncalibrated.** The parameters are estimated from
  the sample the test scores, so the null distribution of D does not apply. The
  classical Lilliefors result covers maximum-likelihood estimates, not the
  L-moment and closed-form estimators used here, so the direction of the
  miscalibration is not established (ADR-0028). Anderson-Darling has no naive
  p-value at all. Treat both as screening statistics.
- **The parametric bootstrap refits.** `n_bootstrap > 0` draws samples from the
  fitted mixed distribution, refits every draw with the library's own estimator,
  and reports `(1 + #{statistic* >= statistic}) / (B_ok + 1)` over the refits
  that succeeded. The cost is about `n_bootstrap` times the cost of the fits, so
  bootstrap a region or a sample, not a whole CONUS grid, unless you have the
  budget.
- **Pass the record's rounding unit as `resolution`.** Sums of values rounded to
  a unit are multiples of it. With `resolution=None`, the draws are continuous,
  which is the right null only for an unrounded record; a rounded record is then
  rejected far too often. The unit is never inferred. For a summed window the
  unit is the record's unit; `index="standardized_index"` with
  `aggregation="mean"` uses that unit divided by the Timescale.
- **Failed refits leave the denominator.** Read `bootstrap_failed_share`; a
  candidate whose refits often fail has a p-value from fewer draws.
- **The seed is recorded.** Pass `seed` for a reproducible run; the result
  records the seed used as a decimal string in `climate_indices_rng_seed`, so
  seeds above 64 bits survive a NetCDF round trip.
- **The selected candidate's p-value is post-selection.** A p-value attached to
  the candidate `best` chose is not an ordinary confirmatory test.
- **Across cells.** With a cell axis, `rejection_share` reports the share of
  cells at or below `alpha` for each candidate, step and statistic. It uses the
  bootstrap p-value when one exists and the naive K-S p-value otherwise, so the
  AD column is NaN without a bootstrap. `bh_adjust=True` adds
  Benjamini-Hochberg adjusted p-values across cells; spatial correlation weakens
  the assumptions of that adjustment, and field significance is out of scope.
- **Estimation and ties.** Gamma and Pearson Type III drop exact zeros from the
  statistics, because the zero mass is not part of the continuous distribution
  under test; the log-logistic and GEV keep zeros as ordinary values. Repeated
  values are kept with their frequency. Estimation, ties and failed refits all
  move the rejection rate, so use `rejection_share` to compare candidates, not
  as a calibrated error rate.

## Measure the decision impact

A ranking is abstract. The question a user feels is whether the choice changes
the drought category. `suitability.drought_category()` maps normal-scale
index values to the U.S. Drought Monitor classes (None and D0 to D4), and
`suitability.compare_categories()` counts how often two classifications of the
same values disagree.

```python
from climate_indices import spei
from climate_indices.drought.indices import Distribution

gamma_values = spei(
    precip, pet, scale=3, distribution=Distribution.gamma,
    calibration_year_initial=1981, calibration_year_final=2010,
)
gev_values = spei(
    precip, pet, scale=3, distribution=Distribution.gev,
    calibration_year_initial=1981, calibration_year_final=2010,
)
change = suitability.compare_categories(
    suitability.drought_category(gamma_values, output_scale="normal"),
    suitability.drought_category(gev_values, output_scale="normal"),
    dims="time",
)
change["share_changed"]             # per cell after the time reduction
change["share_changed_in_drought"]
change["confusion"]
```

- `output_scale` is required and only `"normal"` classifies. A `"probability"`
  or `"bounded"` argument raises. The function validates the declared scale, not
  the values, so a probability array declared as normal is silently classified
  and misread.
- Each class includes its upper edge, so -0.5 is D0, -0.8 is D1, -1.3 is D2,
  -1.6 is D3 and -2.0 is D4, while -0.49 is none. The USDM table does not state
  boundary handling, so this rule is the ADR-0028 contract.
- Only the dry side is classified; every wet value is "none". A missing value
  (-1) drops its pair from every count. `share_changed_in_drought` counts a pair
  as in drought when either classification is D0 to D4.
- The confusion matrix and the shares describe how often the two
  classifications disagree. They say nothing about which one is right.

`suitability.category_agreement()` gives the same comparison against a reference
classification as two numbers, the exact and the within-one share. It has no
xarray route, so pass NumPy codes, and a pair with a missing side is dropped.

## Check uniformity by calendar step

Use `suitability.pit_uniformity()` with probability output from a continuous SPI,
SPEI, SPAI, or `standardized_index()` fit. It reports the Kolmogorov–Smirnov
statistic `ks_statistic`, the naive p-value `ks_p_value_naive`, and the
Anderson–Darling statistic `ad_statistic` for each month or day of the 366-day
calendar. For daily xarray input, February 29 includes only observed leap days.
SPAI pools anomalies across the calendar, so months with different spreads can
show nonuniform probabilities even when the pooled fit looks reasonable. SPEI
fits each calendar month separately, and the two diagnostics answer which of the
two routes is calibrated per month.

For the gamma SPI above, use the fitted `prob_zero` and mark exact zero
accumulations. Use the same Calibration Period as the source fit.

```python
from climate_indices import suitability

accumulated = precip.clip(min=0).rolling(time=options["scale"]).sum()
seasonal = suitability.pit_uniformity(
    pit, **options, prob_zero=fit.prob_zero, zero_mask=accumulated == 0
)
july = seasonal.sel(month=7)
```

For a mixed CDF the diagnostic removes the marked point mass and transforms
the remaining probabilities to `(p - p0) / (1 - p0)`. For a Pearson Type III
fit with a zero mass, mark all accumulated values below 0.0005 where
`p0 > 0` in `zero_mask`, and mark the nonzero subset in `trace_mask`.
`zero_share` and `trace_share` count exact zeros and traces separately as
fractions of finite probabilities. `n_valid` counts the probabilities that
enter the statistics. An unmasked mixed-CDF probability below `p0` raises a
mask error. A masked probability must match a supported point-mass placement
at the input precision. For an all-zero gamma calibration step, the diagnostic
accepts a masked probability of 0 with `prob_zero=1`. That step has
`zero_share=1`, `n_valid=0`, and missing test statistics. Pearson Type III also
permits 1 at its upper support limit. Check `distribution_used` from
`fit_diagnostics()` for fallback before specifying the distribution and mask.

Removing Pearson traces truncates the continuous sample at 0.0005. Check the
fitted CDF at that cutoff before interpreting the results. The uniform
reference approximates the retained sample only when that probability is
negligible. An exact probability of 0 or 1 gives an infinite Anderson–Darling
statistic. A p-value from the fit's own
Calibration Period can be conservative and is useful for screening. Above a
Timescale of 12 months or 365 days, overlapping accumulation windows invalidate
the naive p-values and produce a warning.

`suitability.pit_uniformity()` requires the `distribution` that produced the
probabilities, because probability values alone cannot show whether a point mass
is a fitted zero mass. The empirical and beta outputs are discrete or carry
boundary masses at both ends, so they raise instead of being scored as
continuous.

## Compare events and categories against a reference

`suitability.event_concordance()` counts the agreement of an index's events
with a reference event set, both from `climate_indices.runs.identify_runs()` on
the same validity mask:

```python
import numpy as np
from climate_indices.runs import identify_runs

# Mask to the shared valid steps first: NaN separates runs, and a run crossing
# an unknown step is rejected.
index_runs = identify_runs(np.where(valid, spei_values, np.nan), threshold=-1.0, min_duration=2)
reference_runs = identify_runs(
    np.where(valid, reference_values, np.nan), threshold=reference_threshold, min_duration=2
)
report = suitability.event_concordance(
    index_runs, reference_runs, time=time, valid=valid, cadence="monthly"
)
report.hits, report.misses, report.false_alarms, report.pod, report.far, report.csi
```

- The counting is reference-centric (ADR-0028, section 7): a hit is a reference
  event overlapped by at least one index event, a miss is a reference event
  overlapped by none, and a false alarm is an index event overlapping no
  reference event. One index event can overlap several reference events: those
  are several hits and one matched index event.
- `pod`, `far` and `csi` are NaN, never zero, only when their own denominator is
  empty. With reference events but no index events, POD and CSI are zero and
  only FAR is NaN.
- A run that touches the record edge or an unknown step is censored. Censored
  runs are counted and flagged, not corrected for, because their physical onset
  or end is unknown.
- **The reference must be independent.** Agreement with a reference that shares
  the index's input or method is circular. A satellite index, a U.S. Drought
  Monitor class series or a soil-moisture percentile series is the intended kind
  of reference; a percentile of the same water balance is a demonstration, not
  an accuracy score. Report the definition of the reference with the counts.

`suitability.category_agreement()` gives the same idea for classes: the exact
and within-one shares between an index's drought categories and a reference's
categories, over the pairs where both are present.

## Decide, and record what you decided

1. Compare the candidates on one shared calibration sample with
   `compare_distributions()` at the Timescale and Calibration Period you intend
   to publish.
2. Read `best` with `delta_to_runner_up` and `fit_failed`. Keep the candidate
   set small and explain any candidate you add or remove.
3. Add `gof=("ks", "ad")`. Use the bootstrap p-values when the decision is close
   or the record is rounded; there, pass `resolution`.
4. Classify the leading candidates and run `compare_categories()`. Treat a large
   `share_changed` as a reason to prefer the more defensible fit or to publish
   both.
5. Run `pit_uniformity()` per calendar step, with the zero and trace masks the
   fit needs, to see whether any month is systematically nonuniform.
6. If an independent reference exists, run `event_concordance()` and
   `category_agreement()`, and state the reference and its independence.
7. Keep the fitted parameters and the run's metadata, so the choice is
   reproducible and visible in the result.

Every step above answers a different question, and none of them establishes
that a distribution is *correct*. A defensible choice reports the relative
evidence, the absolute evidence, the decision impact and the limits together.

[The executable nClimGrid notebook](https://github.com/monocongo/climate_indices/blob/main/notebooks/check_distribution_suitability.ipynb)
works through this sequence at SPEI-3/6/12/24 with gamma, Pearson, log-logistic
and GEV. It uses the prepared sample, so prepare it once with
`uv run --group dev scripts/prepare_e2e_inputs.py`, then open the notebook with
its kernel in `notebooks/` or set `CLIMATE_INDICES_E2E_DATA` to the
prepared-data directory. The
[fit-audit notebook](https://github.com/monocongo/climate_indices/blob/main/notebooks/check_spi_fit.ipynb)
covers the single-fit diagnostics at SPI-1/3/12. See {doc}`choosing-parameters`
for choosing the distribution and period.

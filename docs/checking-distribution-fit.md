# Check an SPI distribution fit

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

[The executable nClimGrid notebook](https://github.com/monocongo/climate_indices/blob/main/notebooks/check_spi_fit.ipynb)
compares gamma and Pearson at SPI-1/3/12 with PIT histograms and maps of January KS
p-values and zero probability. It uses a small crop of the prepared nClimGrid
sample to keep the exact per-cell KS tests practical. Prepare the pinned sample
once with `uv run --group dev scripts/prepare_e2e_inputs.py` (about 15 MB
on the first run), then open the notebook with its kernel in `notebooks/` (the
notebook's default relative data path assumes that working directory). See
{doc}`choosing-parameters` for choosing the distribution and period.

## Check uniformity by calendar step

Use `suitability.pit_uniformity()` with probability output from a continuous SPI,
SPEI, SPAI, or `standardized_index()` fit. It reports the Kolmogorov–Smirnov
statistic `ks_statistic`, the naive p-value `ks_p_value_naive`, and the
Anderson–Darling statistic `ad_statistic` for each month or day of the 366-day
calendar. For daily xarray input, February 29 includes only observed leap days.
SPAI pools anomalies across the calendar, so months with different spreads can
show nonuniform probabilities even when the pooled fit looks reasonable.

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
mask error. Check `distribution_used` for fallback before specifying the
distribution and mask.

Removing Pearson traces truncates the continuous sample at 0.0005. Check the
fitted CDF at that cutoff before interpreting the results. The uniform
reference approximates the retained sample only when that probability is
negligible. An exact probability of 0 or 1 gives an infinite Anderson–Darling
statistic. A p-value from the fit's own
Calibration Period can be conservative and is useful for screening. Above a
Timescale of 12 months or 365 days, overlapping accumulation windows invalidate
the naive p-values and produce a warning.

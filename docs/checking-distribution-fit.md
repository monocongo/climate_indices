# Check an SPI distribution fit

Use `fit_diagnostics()` with the **same input, Timescale, distribution, and
Calibration Period** as `spi()`. For a gridded xarray input it returns a Dataset
with a `month` (or `dayofyear`) axis and the input's cell axes for most
variables. `distribution_used` has only cell axes; it, `ks_p_value`, `n_valid`,
and `prob_zero` help identify fits worth inspecting. Request `output_scale="probability"` from `spi()` to view its
probability-scale (PIT) output.

```python
from climate_indices import fit_diagnostics, spi
from climate_indices.indices import Distribution

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
non-zero values entering that test. Its p-values use the same calibration record
as the fit (which also includes zeros for Pearson), so `p < 0.05` flags a fit to
investigate, not a calibrated hypothesis test or proof that Pearson is better.
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

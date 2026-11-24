# climate_indices 3.3.0 release notes

Released 2026-11-24. It adds distribution suitability tools, GEV for SPEI,
and the `climate_indices.drought` namespace. See the
[full changelog](https://github.com/monocongo/climate_indices/blob/main/CHANGELOG.md)
and {doc}`the migration guide <deprecations/api-changes>`.

## Refit Pearson Type III SPI when the Calibration Period contains zeros

The Pearson Type III fit now estimates L-moments from nonzero accumulations and
accounts for exact zeros through the Probability of Zero. The previous fit
included zeros in both steps. At calendar steps with zero accumulations during the
Calibration Period, the corrected fit can change both zero and nonzero SPI scores.
Supply newly fitted parameters in place of saved parameters to use the correction.
`standardized_index()` with Pearson Type III follows the same rule.
See [Pearson zero-mass correction](deprecations/api-changes.md#pearson-zero-mass-correction-in-330).

## Compare distributions and drought categories

The Beta `climate_indices.suitability` module compares candidate fits for SPI and
SPEI with NumPy or xarray input. It reports criteria such as AICc at the library's
fitted estimates and includes goodness-of-fit diagnostics with optional parametric
bootstrap p-values. Use `drought_category()` to map normal-scale SPI or SPEI to
U.S. Drought Monitor classes, and `compare_categories()` to measure how often
two classifications differ. The module also offers event and category agreement
metrics. See [the suitability reference](reference.md#climate_indicessuitability).

Use `pit_uniformity()` to check probability-scale outputs from SPAI, SPI, SPEI or
any standardized index. It reports K–S and Anderson–Darling statistics for each
calendar step and a naive K–S p-value over the Calibration Period. The p-value is
in-sample evidence, not a formal test. See
{doc}`checking distribution fit <checking-distribution-fit>`.

SPEI accepts the generalized extreme value distribution with `Distribution.gev`.
The SPEI CLI adds a fourth output per Timescale alongside gamma, Pearson Type III
and log-logistic. Update scripts that expect exactly three SPEI files.

## Migrate drought module imports

Use `from climate_indices import drought` for the typed NumPy and xarray API, or
`from climate_indices.drought import indices, palmer` for the NumPy implementation
modules. The old `climate_indices.indices`, `palmer`, `eto`, `pm_eto` and
`self_calibration` module paths warn starting in 3.3.0 and are scheduled for removal
in 4.0.0. Existing package-root function imports such as
`from climate_indices import spi` remain supported.

## Upgrade Icechunk and check fitted-parameter metadata

Icechunk writes use the Icechunk writer for compatibility with current xarray.
Install the extra with `pip install 'climate_indices[icechunk]>=3.3.0'` on
Python 3.12 or later to get `icechunk>=2.2,<3` and use it with Zarr v3.
See {doc}`writing outputs <writing-outputs>`. The base package adds Linux support
for Python 3.15 and still requires at least Python 3.11.

For xarray inputs, `fit_diagnostics()` now copies the input's `units` to
`beta`, `loc` and `scale` when present. If the input has no `units` attribute,
those parameters have none. Dimensionless parameters such as `alpha`, `skew` and
`prob_zero` retain `units = "1"`.

## Recompute affected Palmer results

The standard Palmer recursion preserves resolved PDSI, PHDI and PMDI values at
the end of the record. Each month still pending receives its own PMDI. The
recursion stops at the last supplied month, before any padding added to complete
the year. Recompute cached PDSI, PHDI and PMDI for records that end partway through
a year or have a deferral run near the end. See
{doc}`the Palmer end-of-record decision <adr/0030-palmer-end-of-record-flush>`.

PMDI also resolves an exact tie between incipient wet and dry spells in favor of
the wet spell when no spell is established. Lower-layer water loss handles zero
Available Water Capacity without a division by zero.

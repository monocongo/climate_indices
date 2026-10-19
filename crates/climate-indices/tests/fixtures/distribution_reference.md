# Standalone distribution regression fixtures

These fixtures compare Rust kernels with the retained pure-Python implementation and SciPy.
They provide regression coverage, not independent scientific validation.
No expected numerical value comes from Rust.
Cargo reads the checked-in text files and does not start Python or run the generator.
The parser uses only the Rust standard library; arrays use the existing `ndarray` dependency.

## Source and regeneration

The source revision is `f4cb938eb622a3ddf8b96c129337e0ebecd4121e` in
[monocongo/climate_indices](https://github.com/monocongo/climate_indices/tree/f4cb938eb622a3ddf8b96c129337e0ebecd4121e).
Each text file also records the revision, runtime versions, units, format, and command.

Run this optional command from the repository root:

```bash
PYTHONPATH=src uv run python scripts/prepare_rust_distribution_fixtures.py
```

The generator blocks `climate_indices._native` in `sys.modules` before library imports.
It asserts that `compute`, `indices`, and `lmoments` resolve to this checkout's source files and that native dispatch is unavailable.
It also refuses tracked source changes relative to the recorded revision.
Expected warnings from deliberately invalid inputs do not prevent generation.
Review fixture changes before accepting a new source revision or runtime version.

## Format and comparison

Each record has a `NAME ROWS COLUMNS` header followed by row-major float64 values.
Vectors have one row. Values use 17 significant digits; `nan`, `inf`, and `-inf` retain their meanings.
Comment lines start with `#`.
Tests require equal array shapes and exact NaN masks.
Finite comparisons use `abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected)`.
Infinite results must have the expected sign; finite substitutes fail.
A test also checks that the comparison helper rejects mask errors and incorrect infinite results.

## Files and reference calculations

### `distribution_fits.txt`

`calibration` has shape `(10, 12)` and represents ten calibration years and twelve independent columns.
The Python gamma call uses monthly periodicity and the full 2000–2009 Calibration Period.
Every parameter and validity vector has shape `(1, 12)`.

The columns contain moderate positive skew, strong positive skew, negative skew, symmetry, zeros with a missing observation,
three observations, a positive constant, all NaNs, all zeros, near symmetry, signed values, and rescaled positive skew.
The constant column produces an infinite gamma alpha and zero beta.
Unusable Pearson and GLO fits retain zero parameters and an exact false validity flag.

- `compute.gamma_parameters` supplies gamma alpha and beta.
- `compute._pearson_parameters_spatial` supplies Pearson zero probabilities, locations, scales, and skews.
  `lmoments.fit_spatial` and the four-nonzero-observation guard supply its validity mask.
- `lmoments.fit_glo_spatial` supplies GLO locations, scales, shapes, and validity masks.

Input values, locations, scales, and gamma beta use synthetic millimeters.
Alpha, skew, GLO shape, zero probabilities, and validity flags are dimensionless.
These short synthetic samples test kernel contracts, not the adequacy of a scientific Calibration Period.

### `distribution_cdfs.txt`

- `gamma_cases` has shape `(24, 3)`, with columns `(x, alpha, beta)`.
  `scipy.stats.gamma.cdf` supplies the CDF; `scipy.stats.gamma.sf` supplies the separate unit-scale survival function.
  Cases include series, continued-fraction, and asymptotic regimes, tails, zeros, negative values, invalid parameters, NaNs, and infinities.
- `gamma_values` and `gamma_probabilities` have shape `(5, 6)`; parameter vectors have shape `(1, 6)`.
  The generator retains `compute.transform_fitted_gamma`'s SciPy CDF and zero-mixture step before output scaling.
  Zero observations bypass an invalid fit, while nonzero observations retain NaNs from that fit.
- `pearson_values` and `pearson_cdf` have shape `(15, 11)`; parameter vectors have shape `(1, 11)`.
  `scipy.stats.pearson3.cdf` supplies probabilities for positive, negative, zero, near-zero, and invalid skews.
  Skews include `1e-4` and `-1e-4`, which pin SciPy's `1.6e-5` normal-transition threshold.
  Probes include both support limits, adjacent values, normal tails, NaNs, infinities, and a zero scale.
- `ks_values` has shape `(7, 7)`; its parameters and statistics have shape `(1, 7)`.
  `compute._pearson_ks_statistics` supplies the Kolmogorov–Smirnov D statistic.
  Samples include ties, zeros, missing values, empty columns, one observation, and an invalid scale.
- `glo_values` and `glo_cdf` have shape `(15, 9)`; parameter vectors have shape `(1, 9)`.
  `compute._loglogistic_fit(..., output_scale="probability")` supplies the GLO probabilities.
  Shapes include both signs, zero, the small-shape threshold, `1e-4`, and NaN.
  Probes include support limits, adjacent values, tails, missing values, and infinities.
- `glo_bare_*` have shape `(1, 7)`. The generator applies the unmasked `cdfglo`
  formula, so a non-positive scale and `|shape| >= 1` yield finite probabilities.
  Rust's `loglogistic_cdf` is compared with these values directly, pinning the
  kernel's documented no-validation contract.

CDF observations, locations, and scales use synthetic millimeters.
The gamma survival-function probe interprets `x` at unit scale and is dimensionless.
Probabilities, skews, shapes, alpha, and KS D are dimensionless.

### `distribution_special.txt`

`igamma_cases` has shape `(23, 2)`, with columns `(a, x)`.
`scipy.special.gammainc` and `gammaincc` supply lower and upper regularized incomplete-gamma values.
The cases cover ordinary and asymptotic regimes, tails, zeros, invalid arguments, NaNs, and infinities.
Their output vectors have shape `(1, 23)`.

`normal_x` and `ndtr` have shape `(1, 18)` and include probes from -40 to 40 and signed infinities.
`quantiles`, `ndtri`, and `norm_ppf` have shape `(1, 19)`.
They include both endpoints, the smallest positive float64, `1e-300`, the predecessor of 1, invalid probabilities, and NaN.
`scipy.special.ndtr`, `ndtri`, and `scipy.stats.norm.ppf` supply the expected results.
All fields are dimensionless.

### `distribution_indices.txt`

- EDDI climatology has shape `(7, 5)` and observations have shape `(6, 5)`.
  The generator retains `indices.eddi`'s NumPy strict-rank and Tukey plotting-position steps.
  Cases include ties, constants, zeros, missing values, short climatologies, and leading-pad counts.
  Out-of-calibration extremes can exceed probability 1 before clipping; the test preserves this kernel behavior.
  `indices._hastings_inverse_normal` supplies 15 inverse-normal probes, including clipping boundaries and NaN.
- PNP values and ratios have shape `(65, 4)`; calibration has shape `(5, 48)` and normals have shape `(1, 48)`.
  Five full calibration years contain twelve calendar steps and four cells, in calendar-step-major order.
  The remaining five observations test partial-year output against the same normals.
  The generator retains `indices.percentage_of_normal`'s NumPy normal calculation and calls that API for the expected ratios.
  Cases include nonpositive normals, sparse and all-NaN calendar steps, constant values, and zero-only cells.
  The returned PNP values are ratios, not ratios multiplied by 100.
- PCI inputs have shape `(1, 365)` or `(1, 366)` and scalar outputs have shape `(1, 1)`.
  `indices.pci` supplies each expected result.
  Cases contain varied rainfall, uniform daily rainfall, zero rainfall, and rainfall across the February/March boundary.

Observations and normals use synthetic millimeters; leading pads are counts.
Probabilities, Hastings results, PNP ratios, and PCI are dimensionless.

## Public-function coverage

`distribution_reference.rs` calls all 21 public numerical functions in these modules:

| Module | Functions |
| --- | --- |
| `gamma` | `gamma_parameters`, `gamma_cdf`, `gamma_sf`, `gamma_probabilities` |
| `pearson` | `pearson_parameters`, `pearson_cdf`, `pearson_cdf_block`, `pearson_ks_statistics` |
| `loglogistic` | `loglogistic_parameters`, `loglogistic_cdf`, `loglogistic_cdf_block` |
| `eddi` | `tukey_probabilities`, `hastings_inverse_normal` |
| `pnp` | `pnp_normals`, `pnp_percentages` |
| `pci` | `pci` |
| `special` | `igam`, `igamc`, `ndtr`, `ndtri`, `norm_ppf` |

The suite also calls both public recurrence functions, `recurrence_inputs` and `run`.
The recurrence test uses explicit expected values for a state increment of one per active day, not Rust-generated fixtures.
It checks both gap policies, a resumed gap, initial poisoning, static-invalid cells, seasonal carry, spin-up, final states, and final gaps.
It runs single-column and multi-column paths, with and without recorded history, and rejects an unknown policy name.
`lmoments` has no public functions.

A deterministic invariant test uses 256 generated positive parameter sets without an additional dependency.
It checks CDF ranges and monotonicity, decreasing survival probabilities, incomplete-gamma complements, normal inverse consistency, and NaN propagation.
Pearson skew and GLO shape take both signs and zero.
These properties supplement the frozen references; they do not replace them.

Run the standalone suite with:

```bash
cargo test -p climate-indices --test distribution_reference
```

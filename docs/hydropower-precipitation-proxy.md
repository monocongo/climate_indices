# Standardize a hydropower precipitation proxy

Use this recipe to standardize a capacity-weighted monthly precipitation series
at Timescales of 1, 3, 6, and 12 months. It uses the existing
{func}`climate_indices.indices.standardized_index` NumPy API, not a separate
hydropower function.

This proxy describes the **climate-driven component of hydro resource**, not
actual hydropower generation or electricity load. Generation also depends on
installed capacity, technology, reservoir operation, curtailment, markets, and
behaviour. The recipe does not model runoff, reservoir inflow, or generation.
See {doc}`energy_applications` for the energy-family scope.

## Prepare the monthly series

Spatial aggregation and installed-capacity weighting are your steps, outside
`climate_indices`. Use tools such as [`xagg`][xagg] or
[`regionmask`][regionmask] to prepare regional masks and aggregation weights.
Neither tool supplies the installed-capacity inventory for this recipe.

Use non-negative monthly precipitation totals in consistent units, such as
mm. For each month, form a weighted mean using the capacity weights and the
precipitation for the areas they represent. Divide the weighted sum by the
sum of the weights. Keep the spatial coverage and weights fixed through time
if the scores must describe climate variation without capacity changes.
Document the capacity inventory, weights, spatial coverage, and missing-data
policy with the resulting series.

The input must be a one-dimensional NumPy array in monthly order, starting
in January of `data_start_year`. Include every calendar month. Represent
missing observations as NaN, not zero, and convert negative missing-data
codes to NaN before calling. Negative values otherwise become zero with a
log warning. Do not remove missing months: that changes the calendar groups
used for Distribution Fitting.

## Compute four Timescales

The example uses **synthetic data**, not observations or C3S output. The
array represents an already capacity-weighted series covering January 1981
through December 2020. Replace it with your prepared monthly series.

```{testcode}
import numpy as np

from climate_indices import compute, indices

rng = np.random.default_rng(1167)
weighted_precipitation = rng.gamma(shape=2.0, scale=40.0, size=40 * 12)
```

Use `aggregation="sum"` to accumulate monthly precipitation totals. Supply
the monthly series, not a series already accumulated over the Timescale.
The Calibration Period below is 1981–2010, inclusive. The function fits each
calendar month's distribution on that period and transforms the full record,
including 2011–2020.

```{testcode}
hydro_proxy = {}
for timescale in (1, 3, 6, 12):
    hydro_proxy[timescale] = indices.standardized_index(
        weighted_precipitation,
        scale=timescale,
        distribution=indices.Distribution.gamma,
        data_start_year=1981,
        calibration_year_initial=1981,
        calibration_year_final=2010,
        periodicity=compute.Periodicity.monthly,
        aggregation="sum",
    )

for timescale, scores in hydro_proxy.items():
    assert scores.shape == weighted_precipitation.shape
    assert np.isnan(scores[: timescale - 1]).all()
    assert np.isfinite(scores[timescale - 1 :]).all()
    assert (np.abs(scores[timescale - 1 :]) <= 3.09).all()
```

The two-parameter Gamma Distribution follows the precipitation convention,
as in {doc}`standardized-hydrologic-indices`. It is an explicit recipe
choice, not the planned empirical standardization for SREPI or SRLI.
Check whether the chosen Distribution fits your series; see
{doc}`checking-distribution-fit` and {doc}`choosing-parameters`.

## Interpret the result

Each dictionary entry contains one unitless score per input month. Negative
scores indicate lower accumulated precipitation relative to the fitted
calendar-month distribution; positive scores indicate higher precipitation.
Normal-score outputs are clipped to [-3.09, 3.09]. They are not generation
estimates, capacity factors, or measurements of energy shortages.

The first `scale - 1` outputs are NaN because their accumulation windows are
incomplete. This gives 0, 2, 5, and 11 leading NaNs for these four Timescales.
A window that contains a missing value also produces NaN. Those windows do
not contribute valid observations to the fit. Use at least 30 years for the
Calibration Period where possible, and inspect missing values and fit quality
before interpreting the scores. See {doc}`data_requirements` for the shared
input and calibration requirements.

## C3S data and validation limits

The Copernicus C3S [global climate and energy indicators][c3s-energy] include
hydropower-related products. This recipe does **not** reproduce the C3S IWP
product: its precipitation accumulation and weighting differ. Check a
product's units, temporal aggregation, spatial weights, and missing-data
conventions before using it. Do not accumulate an already accumulated product
again unless that additional accumulation is intentional.

The executable example checks the API, output shape, incomplete windows,
and score bounds. It does not validate hydro resource or generation against
observations. The existing statistical references apply to the selected
standardization transform, not to a hydropower generation model. A real-data
C3S demonstration remains planned in
[ENERGY-23](https://github.com/monocongo/climate_indices/issues/1179).

[xagg]: https://github.com/ks905383/xagg
[regionmask]: https://regionmask.readthedocs.io/
[c3s-energy]: https://doi.org/10.24381/3bb607bd

# Standardized hydrologic indices (SRI and SSI)

The Standardized Runoff Index (SRI; [Shukla & Wood (2008)][shukla-2008]) and the
Standardized Streamflow Index (SSI; [Vicente-Serrano et al. (2012)][vicente-serrano-2012])
apply the SPI's standardization procedure to a runoff or streamflow series.
`climate_indices` does not model runoff, streamflow, or any other part of the
hydrologic cycle: you supply the series, and `indices.standardized_index()`
standardizes it against the Calibration Period you choose. The index names
describe the input variable, not a separate computation — both recipes below
are thin calls into the generic API, so nothing in the fitting or
standardization is duplicated.

`indices.standardized_index()` is input-agnostic, so it also accepts any other
non-negative monthly or daily series. See {doc}`flood_applications` for what
standardized wetness can and cannot say about flood potential, and
{doc}`algorithms` for the fitting and standardization steps these recipes
reuse.

## SRI from monthly runoff

```python
from climate_indices import compute, indices

# monthly runoff, in any units, non-negative; 1981-01 through 2010-12
runoff = ...  # 1-D numpy array of 360 monthly values

sri = indices.standardized_index(
    runoff,
    scale=3,  # 3-month accumulation
    distribution=indices.Distribution.gamma,  # two-parameter gamma, the SPI convention
    data_start_year=1981,
    calibration_year_initial=1981,
    calibration_year_final=2010,
    periodicity=compute.Periodicity.monthly,
)
```

[Shukla & Wood (2008)][shukla-2008] define the SRI by standardizing runoff the
way the SPI standardizes precipitation, over monthly to seasonal accumulation
periods; `scale` is that accumulation length in months. The two-parameter
gamma fit is the SPI convention, so it is the recipe's default;
`indices.Distribution.pearson` (Pearson Type III) is available when a
three-parameter fit is preferred.

## SSI from monthly streamflow

```python
from climate_indices import compute, indices

# monthly streamflow or river discharge, in any units, non-negative
streamflow = ...  # 1-D numpy array, 1981-01 through 2010-12 for the calibration below

ssi = indices.standardized_index(
    streamflow,
    scale=12,  # 12-month accumulation
    distribution=indices.Distribution.pearson,  # Pearson Type III
    data_start_year=1981,
    calibration_year_initial=1981,
    calibration_year_final=2010,
    periodicity=compute.Periodicity.monthly,
)
```

[Vicente-Serrano et al. (2012)][vicente-serrano-2012] tested six
three-parameter distributions for monthly streamflow — lognormal, Pearson Type
III, log-logistic, general extreme value, generalized Pareto, and Weibull —
with two selection procedures, and found that no single distribution was
optimal across stations, though the commonly used flow-frequency distributions
provided good fits. Of that set this package supports Pearson Type III; the
two-parameter gamma is also available. The remaining distributions, including
log-logistic, are not implemented, and log-logistic support is tracked by
[#106][log-logistic].

## Calibration and accumulation

Both recipes follow the SPI conventions and warnings described in
{doc}`algorithms` and {doc}`choosing-parameters`: use a Calibration Period of
at least 30 years where the record allows, expect a warning for a shorter one,
and choose `scale` from the accumulation window the question needs (1, 3, 6,
and 12 months are the common choices). The NumPy API returns one standardized
value per input time step, unitless and clipped to [-3.09, 3.09] as SPI values
are.

The xarray entry point for `indices.standardized_index()` is not wired yet:
named wrappers and the xarray surface await the flood design decision
([FLOOD-02 #1099][flood-1099]) and the CF metadata registry entries
([FLOOD-06 #1103][flood-1103]).

## Sources

- [Shukla & Wood (2008), *Use of a standardized runoff index for characterizing hydrologic drought*][shukla-2008]
- [Vicente-Serrano et al. (2012), *Accurate computation of a streamflow drought index*][vicente-serrano-2012]

[shukla-2008]: https://doi.org/10.1029/2007GL032487
[vicente-serrano-2012]: https://doi.org/10.1061/(ASCE)HE.1943-5584.0000433
[log-logistic]: https://github.com/monocongo/climate_indices/issues/106
[flood-1099]: https://github.com/monocongo/climate_indices/issues/1099
[flood-1103]: https://github.com/monocongo/climate_indices/issues/1103

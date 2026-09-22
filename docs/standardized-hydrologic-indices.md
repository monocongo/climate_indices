# Standardized hydrologic indices (SRI and SSI)

The Standardized Runoff Index (SRI; [Shukla & Wood (2008)][shukla-2008]) and the
Standardized Streamflow Index (SSI; [Vicente-Serrano et al. (2012)][vicente-serrano-2012])
apply the SPI's standardization procedure to a runoff or streamflow series.
`climate_indices` does not model runoff or streamflow: you supply the series,
and `indices.standardized_index()` standardizes it against the Calibration
Period you choose. The index names describe the input variable, not a separate
computation — both recipes below are thin calls into the generic API, so
nothing in the fitting or standardization is duplicated.

`indices.standardized_index()` is input-agnostic, so it also accepts any other
non-negative monthly or daily series. See {doc}`flood_applications` for what
standardized wetness can and cannot say about flood potential, and
{doc}`algorithm-reference` for the fitting and standardization steps these
recipes reuse.

## SRI from monthly runoff

The recipe assumes `runoff` is a 1-D NumPy array of non-negative monthly
values covering 1981-01 through 2010-12.

```python
from climate_indices import compute, indices

sri = indices.standardized_index(
    runoff,  # 1-D numpy array of non-negative monthly values
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

The recipe assumes `streamflow` is a 1-D NumPy array of non-negative monthly
values covering 1981-01 through 2010-12.

```python
from climate_indices import compute, indices

ssi = indices.standardized_index(
    streamflow,  # 1-D numpy array of non-negative monthly values
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
under two selection procedures, best monthly fit (BMF) and minimum orthogonal
distance (MD). They found that no single distribution was suitable for every
series, because each had limitations, and their procedures instead select a
distribution for each gauging station and month of the year.
`indices.standardized_index()` takes one distribution for the whole call and
reuses it for every calendar period, so this recipe is a single-distribution
approximation of the cited procedure, not a reproduction of it.

Of the six distributions, this package supports Pearson Type III; the
two-parameter gamma is also available. The remaining distributions, including
log-logistic, are not implemented, and log-logistic support is tracked by
[#106][log-logistic].

## Calibration, input, and fitting caveats

Both recipes follow the conventions and warnings in {doc}`data_requirements`
and {doc}`choosing-parameters`. In particular:

- Use a Calibration Period of at least 30 years where the record allows;
  a shorter one emits `ShortCalibrationWarning`, and a window that falls
  outside the input's year coverage is silently replaced by the full record.
- The first `scale - 1` outputs are NaN because no complete accumulation
  window exists before them — 2 leading NaNs for the SRI recipe and 11 for the
  SSI recipe.
- Input values must be non-negative. Negative values, such as a missing-data
  fill value left in the series, are clipped to zero with only a log warning,
  so convert them to NaN before calling.
- Pearson Type III needs at least four non-zero values in every calendar
  period. A failed Pearson fit falls back to gamma with logging rather than an
  exception, so check the fit before relying on `distribution=pearson` for
  short or strongly zero-inflated records; gamma is the documented choice
  there.

The NumPy API returns one standardized value per input time step, unitless and
clipped to [-3.09, 3.09] as SPI values are.

The xarray entry point for `indices.standardized_index()` is not wired yet:
named wrappers await the flood design decision
([FLOOD-02 #1099][flood-1099]), and the xarray surface also needs the CF
metadata registry entries ([FLOOD-06 #1103][flood-1103]).

## Sources

- [Shukla & Wood (2008), *Use of a standardized runoff index for characterizing hydrologic drought*][shukla-2008]
- [Vicente-Serrano et al. (2012), *Accurate computation of a streamflow drought index*][vicente-serrano-2012]

[shukla-2008]: https://doi.org/10.1029/2007GL032487
[vicente-serrano-2012]: https://doi.org/10.1061/(ASCE)HE.1943-5584.0000433
[log-logistic]: https://github.com/monocongo/climate_indices/issues/106
[flood-1099]: https://github.com/monocongo/climate_indices/issues/1099
[flood-1103]: https://github.com/monocongo/climate_indices/issues/1103

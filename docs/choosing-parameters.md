# Choose a distribution and calibration period

Goal: pick a distribution and calibration period deliberately, and read back what
a run was parameterized with. SPI and SPEI are fitted against a distribution over
a calibration period; `eddi`, `percentage_of_normal`, and the Palmer indices are
ranked or fitted over one too, while PET and PCI take no calibration period. The
choices come from the caller, and few of them are validated: no index rejects an
unrepresentative baseline, and SPI and SPEI silently widen a calibration request
that falls outside the record. What each index measures is in {doc}`algorithms`;
the value contract is in {doc}`data_requirements`.

## Prerequisites

- Input that satisfies {doc}`data_requirements`: complete chronological months, the
  366-day daily layout or Gregorian coordinates, declared units, and the missing
  values marked.
- The years the input actually covers, and how much of the calibration window is
  non-missing — the in-memory xarray API raises when the window holds fewer than
  30 effective non-NaN years.
- For a gridded fit, a sense of how zero-inflated the precipitation is.

## Choose the distribution

SPI and SPEI take a `distribution` argument. The command line has no distribution
option and writes both for every requested scale.

- Use `indices.Distribution.gamma` as the starting point: it is the choice
  recommended by McKee et al. (1993), estimates few parameters, and is the right
  fit for strongly zero-inflated precipitation.
- Use `indices.Distribution.pearson` for highly skewed data, or when a gamma fit
  is poor by the Kolmogorov-Smirnov check (p < 0.05). It has more parameters and
  needs more data for a stable estimate.

The fallout of an unstable Pearson fit differs by index: SPI falls back to gamma
for that fit, while SPEI does not fall back at all, so check the Pearson fit
before committing to it there. {doc}`data_requirements` has the error contract
for each path. On gridded input the SPI fallback is per spatial block, so a block
whose fit fails is standardized with gamma while its neighbours use Pearson — see
[ADR-0009](adr/0009-spatial-block-declaration.md).

To standardize new data against a fit computed once — a projection run, or a rerun
that must not refit — compute the parameters and pass them as `fitting_params`
(gamma: `alpha` and `beta`; Pearson: `prob_zero`, `loc`, `scale`, `skew`).
{doc}`workflow-examples` has the fit-and-reuse example and the scale-matching
caveat. Two limits apply. The zero probability is still counted from the values
each call receives, so a reused gamma fit fixes the continuous part of the
distribution, not the zero mass. And per-cell parameter arrays align only on the
NumPy API or on a whole-grid in-memory call: the xarray adapter passes one
`fitting_params` value to every Dask block, so a Dask run should reuse
period-only parameters, or let each block fit.

## Choose the calibration period

- 30 years is the documented minimum. On the NumPy path a shorter period emits
  `ShortCalibrationWarning`, and more than 20% missing values inside the period
  warns about fitting reliability. An in-memory xarray input that contains
  missing values is checked against the same threshold and raises
  `InsufficientDataError` when the window falls short, as it does for a window
  with no overlap at all. A Dask-backed input skips that check, so confirm your
  own coverage there. {doc}`data_requirements` is the full contract.
- The period must fall inside the input's year coverage. Matching it there is the
  caller's responsibility for SPI and SPEI: a request outside the record is
  replaced with the full available record, without an error, and the result
  metadata still records the years you asked for rather than the years used.
  Keep the request inside the coverage if you want the two to agree. `eddi` does
  raise `InvalidArgumentError` for out-of-range bounds, and
  `percentage_of_normal` only partially validates them.
- On the command line, set `--calibration_start_year` and
  `--calibration_end_year`. The xarray API infers both from the `time` coordinate
  when they are omitted, and records inferred years in the result either way;
  pass them explicitly in anything published or compared, so the baseline is
  visible in the call itself and does not move with the coordinate values.
- Prefer a WMO-recommended normal period (for example 1991-2020), avoid periods
  with major gaps, and revisit the baseline when the climate record moves on. The
  selection guidance behind these rules is in {doc}`algorithms`. The
  [notebook](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb)
  works through a full run that passes an explicit 1981-2010 period.

## Verify the choices

Fit a sample of your data and read the warnings and the result metadata back:

```python
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import spi
from climate_indices.indices import Distribution

rng = np.random.default_rng(42)
precip = xr.DataArray(
    rng.gamma(shape=2.0, scale=15.0, size=40 * 12),
    coords={"time": pd.date_range("1981-01-01", periods=40 * 12, freq="MS")},
    dims=["time"],
    attrs={"units": "mm"},
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = spi(
        values=precip,
        scale=3,
        distribution=Distribution.gamma,
        calibration_year_initial=1981,
        calibration_year_final=2010,
    )

print([f"{warning.category.__name__}: {warning.message}" for warning in caught])
print(result.attrs["distribution"], result.attrs["calibration_year_initial"])

# The recorded years are the requested ones: confirm they fall inside the record,
# or the fit used everything while the metadata says otherwise.
years = precip["time"].dt.year
print("coverage:", int(years[0]), int(years[-1]))
```

A `GoodnessOfFitWarning` or `ShortCalibrationWarning` here is a fit report, not a
failure — decide whether the period or distribution should change. Under
`scheduler="processes"` those warnings are raised in the workers, so they appear
on the workers' stderr rather than in `caught` ({doc}`performance` explains the
scheduler choice). The result records `distribution` and the requested
calibration years, alongside `scale` and `climate_indices_version`.

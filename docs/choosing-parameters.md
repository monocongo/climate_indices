# Choose a distribution and calibration period

Goal: pick a distribution and calibration period deliberately, and read back what
a result was standardized against. SPI and SPEI are fitted against a distribution
over a calibration period, and every other index is ranked or fitted over a
calibration period too. Both choices come
from the caller: the implementation does not reject a poor fit or an
unrepresentative baseline, and for SPI and SPEI it silently substitutes the full
record when the requested calibration years fall outside the data. Choose them
deliberately. What each index measures is in {doc}`algorithms`; the contract the
values must satisfy is in {doc}`data_requirements`.

## Prerequisites

- Input that satisfies {doc}`data_requirements`: complete chronological months, the
  366-day daily layout or Gregorian coordinates, declared units, and the missing
  values marked.
- The years the input actually covers, and — for a gridded fit — a sense of how
  zero-inflated the precipitation is.

## Choose the distribution

SPI and SPEI take a `distribution` argument; the other indices have none. The
command line has no distribution option at all, and writes both for every scale.

- Use `indices.Distribution.gamma` as the starting point: it is the choice
  recommended by McKee et al. (1993), estimates few parameters, and is the right
  fit for strongly zero-inflated precipitation.
- Use `indices.Distribution.pearson` for highly skewed data, or when a gamma fit
  is poor by the Kolmogorov-Smirnov check (p < 0.05). It has more parameters and
  needs more data for a stable estimate.

The fallout of an unstable Pearson fit differs by index: SPI falls back to gamma
for that fit, while SPEI raises `InsufficientDataError`. On gridded input that
fallback is per spatial block, so a block whose fit fails is standardized with
gamma while its neighbours use Pearson — see
[ADR-0009](adr/0009-spatial-block-declaration.md).

When the point is to standardize new data against one fixed climatology — a
projection run, or a rerun that must not refit — compute the parameters once and
pass them as `fitting_params` (gamma: `alpha` and `beta`; Pearson: `prob_zero`,
`loc`, `scale`, `skew`). {doc}`workflow-examples` has the fit-and-reuse example,
including the scale-matching caveat.

## Choose the calibration period

- 30 years is the documented minimum. A shorter period emits
  `ShortCalibrationWarning`, and more than 20% missing values inside the period
  warns about fitting reliability.
- The period must fall inside the input's year coverage, and matching it there is
  the caller's responsibility for SPI and SPEI: out-of-range bounds are replaced
  with the full available record, without an error. `eddi` does raise
  `InvalidArgumentError` for out-of-range bounds, and `percentage_of_normal`
  only partially validates them.
- On the command line, set `--calibration_start_year` and `--calibration_end_year`.
  With the xarray API these can be inferred from the `time` coordinate, but pass
  them explicitly for anything published or compared, so the result metadata
  records the baseline instead of leaving it to inference. A Dask-backed input
  skips the effective-years check entirely, so an explicit period matters more
  there, not less.
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
```

A `GoodnessOfFitWarning` or `ShortCalibrationWarning` here is a fit report, not a
failure — decide whether the period or distribution should change. Under
`scheduler="processes"` those warnings are raised in the workers and land in their
stderr rather than in `caught`, which is why {doc}`performance` recommends keeping
the logging visible on a first run. The result records `distribution`,
`calibration_year_initial`, `calibration_year_final`, and `scale`, so a published
dataset states the baseline it was standardized against.

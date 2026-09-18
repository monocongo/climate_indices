# Explanation

Explanation pages describe the indices, the decisions behind them, and the
project itself. They are for understanding, not for carrying out a task. To
learn by doing, start with {doc}`tutorials`; to get a job done, see
{doc}`how-to`.

```{toctree}
:maxdepth: 1

algorithms
wildfire_applications
```

## Indices provided

This project contains Python implementations of various climate index algorithms which provide
a geographical and temporal picture of the severity of precipitation and temperature anomalies
useful for climate monitoring and research.

- [SPI](https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi),
  Standardized Precipitation Index, utilizing both gamma and Pearson Type III distributions
- [SPEI](https://www.researchgate.net/publication/252361460_The_Standardized_Precipitation-Evapotranspiration_Index_SPEI_a_multiscalar_drought_index),
  Standardized Precipitation Evapotranspiration Index, utilizing both gamma and Pearson Type III distributions
- [PET](https://www.ncdc.noaa.gov/monitoring-references/dyk/potential-evapotranspiration),
  Potential Evapotranspiration, utilizing either [Thornthwaite](https://doi.org/10.2307/210739)
  or [Hargreaves](http://dx.doi.org/10.13031/2013.26773) equations
- [PNP](http://www.droughtmanagement.info/percent-of-normal-precipitation/),
  Percentage of Normal Precipitation
- [PCI](https://www.tandfonline.com/doi/abs/10.1111/J.0033-0124.1980.00300.X),
  Precipitation Concentration Index

This Python implementation of the above climate index algorithms is being developed
with the following goals in mind:

- to provide an open source software package to compute a suite of
  climate indices commonly used for climate monitoring, with well
  documented code that is faithful to the relevant literature and
  which produces scientifically verifiable results
- to provide a central, open location for participation and collaboration
  for researchers, developers, and users of climate indices
- to facilitate standardization and consensus on best-of-breed
  climate index algorithms and corresponding compliant implementations in Python
- to provide transparency into the operational code used for climate
  monitoring activities at NCEI/NOAA, and consequent reproducibility
  of published datasets computed from this package
- to incorporate modern software engineering principles and scientific programming
  best practices

## Get involved

Please use, make suggestions, and contribute to this code. Without
diverse participation and community adoption this project will not reach
its potential.

Are you aware of other indices that would be a good addition here? Can
you identify bottlenecks and help optimize performance? Can you suggest new
ways of comparing these implementations against others (or other
criteria) in order to determine best-of-breed? Please fork the code and
have at it, and/or contact us to see if we can help.

- Read our [contributing guidelines](https://github.com/monocongo/climate_indices/blob/main/CONTRIBUTING.md)
- File an [issue](https://github.com/monocongo/climate_indices/issues), or
  submit a [pull request](https://github.com/monocongo/climate_indices/pulls)
- Send us an [email](mailto:monocongo@gmail.com)

## Copyright and licensing

This is a developmental version of code that is originally developed at
NCEI/NOAA, official release version available on
[drought.gov](https://www.drought.gov/drought/python-climate-indices).
This software is under BSD 3-Clause license, copyright James Adams, 2017.
Please read more on our [license](https://github.com/monocongo/climate_indices/blob/main/LICENSE) page.

## Citation

You can cite `climate_indices` in your projects and research papers via the BibTeX
entry below.

```
@misc {climate_indices,
     author = "James Adams",
     title  = "climate_indices, an open source Python library providing reference implementations of commonly used climate indices",
     url    = "https://github.com/monocongo/climate_indices",
     month  = "may",
     year   = "2017--"
}
```

% Hidden until DOCS-8 wires the architecture decision record into the
% four-section navigation; the pages must build now because the xarray
% compatibility matrix links to them as documents. Listed explicitly rather
% than globbed so adding a file under docs/adr/ cannot publish it by
% accident.
```{toctree}
:hidden:

adr/0001-dual-numpy-xarray-api
adr/0002-multiprocessing-cli-dask-xarray
adr/0003-dask-time-dimension-single-chunk
adr/0004-xarray-calendar-semantics
adr/0005-fire-module-api
adr/0006-fire-recursive-state-and-execution
adr/0007-fire-missing-data-policy
adr/0008-pattern-compliance-by-behavior-not-source-greps
adr/0009-spatial-block-declaration
adr/0010-seasonal-carry-is-an-explicit-mask
adr/0011-palmer-spatial-block-and-per-location-scpdsi
```

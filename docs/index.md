[![Build](https://github.com/monocongo/climate_indices/workflows/tests/badge.svg)](https://github.com/monocongo/climate_indices/actions)
[![Coverage](https://coveralls.io/repos/github/monocongo/climate_indices/badge.svg?branch=main)](https://coveralls.io/github/monocongo/climate_indices?branch=main)
[![Quality](https://api.codacy.com/project/badge/Grade/48563cbc37504fc6aa72100370e71f58)](https://www.codacy.com/app/monocongo/climate_indices?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=monocongo/climate_indices&amp;utm_campaign=Badge_Grade)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
![Python | 3.10-3.14](https://img.shields.io/badge/Python-3.10--3.14-blue?logo=python)

# Climate Indices in Python

`climate_indices` provides Python implementations of climate indices for drought
monitoring and research. Start with the section that matches what you need:

```{toctree}
:maxdepth: 1

tutorials
how-to
reference
explanation
```

:::{note}
**Upgrading to 3.0.0?** 3.0.0 ships breaking changes to daily xarray calendar
alignment, the NumPy gridded input shape guard, the PCI February calculation, and
periodicity argument validation. See {doc}`deprecations/api-changes` for what each
one changes and how to adapt, and
[CHANGELOG.md](https://github.com/monocongo/climate_indices/blob/main/CHANGELOG.md)
for the full release history.
:::

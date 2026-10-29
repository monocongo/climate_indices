[![Build](https://github.com/monocongo/climate_indices/workflows/tests/badge.svg)](https://github.com/monocongo/climate_indices/actions)
[![Coverage](https://coveralls.io/repos/github/monocongo/climate_indices/badge.svg?branch=main)](https://coveralls.io/github/monocongo/climate_indices?branch=main)
[![Quality](https://api.codacy.com/project/badge/Grade/48563cbc37504fc6aa72100370e71f58)](https://www.codacy.com/app/monocongo/climate_indices?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=monocongo/climate_indices&amp;utm_campaign=Badge_Grade)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
![Python | 3.11-3.14](https://img.shields.io/badge/Python-3.11--3.14-blue?logo=python)

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
alignment, the NumPy gridded input shape guard, the PCI February calculation,
periodicity argument validation, and the gamma probability of zero. See {doc}`deprecations/api-changes` for what each
one changes and how to adapt, and
[CHANGELOG.md](https://github.com/monocongo/climate_indices/blob/main/CHANGELOG.md)
for the full release history.
:::

## Versions

| Component | Version | Support |
| --- | --- | --- |
| `climate_indices` (Python) | 3.1.0 | Python 3.11–3.14 |
| `climate-indices` (Rust crate) | 0.1.0 | Rust 1.85 or later |

`climate_indices` 3.1.0 is the current development version; 3.0.0 is published on
[PyPI](https://pypi.org/project/climate-indices/).
The Rust crate follows its own `0.x` track, independent of the Python distribution
version; see its
[changelog](https://github.com/monocongo/climate_indices/blob/main/crates/climate-indices/CHANGELOG.md)
for published versions.

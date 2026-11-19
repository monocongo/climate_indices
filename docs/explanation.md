# Explanation

Explanation pages describe the indices, the decisions behind them, and the
project itself. They are for understanding, not for carrying out a task. To
learn by doing, start with {doc}`tutorials`; to get a job done, see
{doc}`how-to`.

```{toctree}
:maxdepth: 1

architecture
project-overview
algorithms
algorithm_refs/edi
algorithm_refs/flood_index
algorithm_refs/eddi
algorithm_refs/palmer
algorithm_refs/pet
algorithm_refs/reliability
wildfire_applications
flood_applications
energy_applications
reliability_applications
floating_point_best_practices
ai-assisted-development
licensing
adr/index
```

## Indices provided

The project implements published climate and hydrologic indices for drought, evapotranspiration,
fire weather, flood potential, energy, power-system reliability, and compound hazards.
{doc}`project-overview` lists every index family with its scope, inputs, and goals.
{doc}`algorithms` explains when to use SPI, SPEI, PET, the Palmer indices, and the additional
precipitation indices, and the algorithm reference pages cite the
published methods for the indices they cover. `VALIDATION.md` in the repository root records the
evidence for each index.

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

The project's primary license is
[BSD-3-Clause](https://github.com/monocongo/climate_indices/blob/main/LICENSE),
copyright James Adams, 2017.
See {doc}`licensing` for the original NOAA public-domain notice and the third-party exceptions.

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

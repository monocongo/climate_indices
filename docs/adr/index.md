# Architecture Decision Records

The decisions that shape `climate_indices` architecture, recorded in the order
they were made. Each record states the decision and the consequences the code
has to live with.

Each record opens with a `## Status` line naming its standing against the
current code: `Accepted`, `Amended` (the decision stands; the record's text was
corrected in place), or `Superseded by ADR-NNNN`. The authoring conventions are in
[CONTRIBUTING.md](https://github.com/monocongo/climate_indices/blob/main/CONTRIBUTING.md).

```{toctree}
:maxdepth: 1

0001-dual-numpy-xarray-api
0002-multiprocessing-cli-dask-xarray
0003-dask-time-dimension-single-chunk
0004-xarray-calendar-semantics
0005-fire-module-api
0006-fire-recursive-state-and-execution
0007-fire-missing-data-policy
0008-pattern-compliance-by-behavior-not-source-greps
0009-spatial-block-declaration
0010-seasonal-carry-is-an-explicit-mask
0011-palmer-spatial-block-and-per-location-scpdsi
0012-xarray-api-stays-beta-through-3.0.0
0013-flood-module-api-and-naming
0014-flood-family-scientific-conventions
```

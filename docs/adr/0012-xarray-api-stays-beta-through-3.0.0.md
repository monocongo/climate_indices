# The xarray DataArray index APIs stay Beta through 3.0.0 and become Stable in 3.1.0

## Status

Amended by #1061: the maintainer confirmed promotion of the DataArray index APIs
for 3.1.0. The compound Dataset routes and eager event catalogs remain Beta.
The original 3.0.0 decision and its reasoning remain below; the 3.1.0 amendment
records the current guarantee.

ADR-0001 added an xarray-native API alongside the NumPy core, and it has been documented
as **Beta** since 2.3.0: computation results match the stable NumPy API, while the
interface surface (parameter inference, metadata attributes, coordinate handling) may
change at minor-version boundaries. Release 3.0.0 changes daily xarray results through
ADR-0004's calendar alignment, the adapter's first user-visible change to computed values.

Promoting in the same release would advertise stability immediately after changing daily
results, and the Stable guarantee already stated in the README is "no breaking changes in
minor versions". The calendar conversion then has no release of soak time: a defect that
forces another result change would have to ship in a minor release that had already
promised stability.

## Original decision (3.0.0)

The xarray DataArray API stays **Beta** for 3.0.0, with the Beta guarantee stated as
such: no breaking changes in patch releases; interface changes may arrive with a minor
version. Nothing promises "no breaking changes in minor versions" for this surface.

Promotion to Stable is targeted for **3.1.0 at the earliest**, after the ADR-0004
calendar change has a release of soak time. Promotion is a separate decision and a
separate change (#1061 removes `BetaFeatureWarning` from the public surface and
updates the README table and Beta API paragraph, `docs/xarray_compatibility.md`,
`docs/xarray_migration.md`, and the `xarray_adapter.py` and `typed_public_api.py` beta
notes). Until then
`BetaFeatureWarning` remains the public warning category for the adapter paths, and
the README API-stability table, the compatibility matrix, and the warning class all
say Beta and nothing else.

## Original consequences (3.0.0)

The README keeps the Beta row for 3.0.0 and names 3.1.0 as the earliest promotion point;
`docs/xarray_compatibility.md` and `docs/xarray_migration.md` carry the same statement,
so a reader of any of them gets the same guarantee. The release-prep PR (#1013) does not
flip the table, and `docs/deprecations/api-changes.md` does not need a fifth breaking
change: holding Beta adds none.

Callers who need an interface they can freeze against continue to use the NumPy API, as
the compatibility matrix's operational guidance already says.

## 3.1.0 amendment: scoped promotion

Release 3.0.0 shipped the ADR-0004 calendar alignment before this promotion.
The maintainer reconfirmed the promotion against 3.1.0's compound-module and
flood-foundation scope in #1061. The index APIs and the new compound analysis
routes have separate stability guarantees.

The DataArray index APIs and public `xarray_adapter` decorator interface become
**Stable in 3.1.0**: no breaking changes in minor versions.
This covers the typed drought/PET/Palmer APIs, generic standardization,
fit diagnostics, and the fire, flood, and energy DataArray entry points.
Index and diagnostic functions that return Datasets are included; the return
container alone does not determine stability.
Numerical algorithms, calendar conversion, validation, metadata contents, and
Dask chunk requirements do not change. Promotion is an interface guarantee,
not a new claim of independent scientific validation.

The compound Dataset routes and eager event catalogs remain **Beta** under
[ADR-0022](0022-compound-analysis-contracts.md).
They continue to emit `BetaFeatureWarning` and promise no breaking changes in
patch releases. They require their own promotion decision.

The original #1061 proposal to remove `BetaFeatureWarning` does not apply.
The class remains public in `climate_indices` and `climate_indices.exceptions`;
existing imports and warning filters continue to work.
Stable index APIs do not emit it. Energy elementwise transforms stop emitting
it; compound warning behavior remains unchanged.
[The API migration notes](../deprecations/api-changes.md) record this
compatibility decision. No public name, signature, or return type is removed.

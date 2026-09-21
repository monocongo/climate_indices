# The xarray DataArray API stays Beta through 3.0.0 and is promoted no earlier than 3.1.0

## Status

Accepted.

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

## Decision

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

## Consequences

The README keeps the Beta row for 3.0.0 and names 3.1.0 as the earliest promotion point;
`docs/xarray_compatibility.md` and `docs/xarray_migration.md` carry the same statement,
so a reader of any of them gets the same guarantee. The release-prep PR (#1013) does not
flip the table, and `docs/deprecations/api-changes.md` does not need a fifth breaking
change: holding Beta adds none.

Callers who need an interface they can freeze against continue to use the NumPy API, as
the compatibility matrix's operational guidance already says.

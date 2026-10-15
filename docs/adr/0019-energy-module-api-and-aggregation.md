# Energy APIs are namespaced in the `energy` package, and the standardize seam gains an aggregation

## Status

Amended: decision 2 also exposes matching aggregation on `indices.fit_diagnostics()`
so its fitting parameters can be reused by the standardized index.

Only decision 2 describes code that exists: the `aggregation` keyword
landed with this record ([#1158](https://github.com/monocongo/climate_indices/issues/1158)).
The `climate_indices.energy` package does not exist yet. Decisions 1 and 3 through 6
are the contract that the remaining energy tickets
([#1157](https://github.com/monocongo/climate_indices/issues/1157)) implement, so a
reader who greps `climate_indices.energy` today finds this record and no module. Each
decision is amended as its implementation lands, as
{doc}`0013-flood-module-api-and-naming` was for flood.

The energy-meteorology family
([#1157](https://github.com/monocongo/climate_indices/issues/1157)) adds demand
proxies (degree days), resource transforms (hub-height wind, capacity factors), and
standardized energy indices (SREPI, a standardized demand index, SRLI) with
run-theory energy-drought events. {doc}`0005-fire-module-api` and
{doc}`0013-flood-module-api-and-naming` settled the same question for fire
and flood, and apply here: a family of indices that share a physical basis and evolve
together gets a namespaced package rather than the drought-oriented `indices.py` and
`compute.py`, whose top-level names it would crowd.

`indices.standardized_index()` cannot carry the standardized energy indices as it
stood. It *summed* the series over `scale`, which is right for accumulated precipitation
but not for a capacity factor. A mean keeps the accumulated value in the series' own
range — a week's mean stays in [0, 1] where a week's sum does not — which both preserves
what the number means (a fraction of capacity) and keeps a support-constrained family
such as beta fittable to it. A rank-based standardization is indifferent to the choice,
since a sum is its mean multiplied by `scale` and the ranks are identical; the
aggregation matters for the value's interpretation and for the families whose support is
the series' range. And the seam clipped negatives to zero, which is wrong for a signed
series such as residual load.

## Decision

1. **Package.** Energy computations live in `climate_indices.energy`, a package with a
   public facade over underscore-prefixed implementation modules, mirroring `fire` and
   `flood`, and are imported as `from climate_indices import energy`. No energy function
   is re-exported from the package root or added to `typed_public_api.py`. Beta xarray
   paths keep the `energy.<name>` route under
   {doc}`0012-xarray-api-stays-beta-through-3.0.0`. The package lands with
   its first real function, not as an empty scaffold.

2. **Aggregation is a keyword of the seam.** `indices.standardized_index()`,
   `indices.fit_diagnostics()`, and `compute.prepare_scaled()` take `aggregation`,
   `"sum"` or `"mean"`, default `"sum"`. Reused fitting parameters must come from
   diagnostics with the same aggregation. A mean is the same windowed sum divided by `scale`, so the leading
   `scale - 1` pad and any window holding a missing value stay NaN exactly as they do
   for a sum, and a `scale` of 1 is the identity under either. The other index surfaces
   pass nothing and get the sum they always got. The pinned-golden parity tests in
   `tests/test_standardized_index.py` pin `standardized_index()` against `spi()` and the
   reference fixtures; every other surface (SPEI, PNP, EDDI, the xarray adapter, and the
   CLI) rests on the unchanged default, and no call site supplies `aggregation`.
   Aggregation and distribution are independent: an accumulated
   non-negative series is `sum` with gamma, a capacity factor is `mean` with beta or
   empirical, and residual load is either with empirical.

3. **Signed and bounded series.** A signed series enters the seam on the empirical
   (rank-based) path only, which has no support to violate and no zero mass.
   A bounded [0, 1] series takes beta or empirical. The parametric families (gamma,
   Pearson Type III, log-logistic) reject a series with negatives when negatives are not
   being clipped, rather than clip it: clipping residual load to zero would corrupt the
   index. `clip_negatives` becomes a keyword of `standardized_index()` with default
   `True`, so existing callers are untouched. That keyword and the rejection land with
   the first consumer that needs them
   ([#1168](https://github.com/monocongo/climate_indices/issues/1168),
   [#1172](https://github.com/monocongo/climate_indices/issues/1172)), because exposing
   `clip_negatives=False` before an empirical path exists would let a signed series
   reach a gamma fit.

4. **Standardization is unit-agnostic.** SREPI, the standardized demand index, and SRLI
   standardize whatever series they are handed; the seam carries no unit contract. That
   is what lets them accept capacity factors, degree days, and demand proxies from any
   source. Every energy docstring and page states that the indices describe the
   climate-driven component of supply and demand, not actual generation or load.

5. **Calendar.** Daily first. Weekly is daily input with `scale=7`, and a 30-day window
   is daily input with `scale=30`; calendar months use monthly periodicity. Neither adds
   a `Periodicity` member. Sub-daily support
   ([#1177](https://github.com/monocongo/climate_indices/issues/1177)) is an additive
   `Periodicity` member, so it changes no signature.

6. **Units.** Temperature is °C, the library's internal convention (`_units.py`
   normalizes K and °F at its boundary). A degree day is a temperature *difference*, so
   K and °C are numerically interchangeable and the output is labelled °C day; the exact
   UDUNITS string belongs to the CF registry entries
   ([#1162](https://github.com/monocongo/climate_indices/issues/1162)). Wind is
   m s⁻¹ and radiation is a mean flux in W m⁻².

The family adds no dependency. A Phase 3 transform that needs one puts it behind a
`climate-indices[energy]` extra, so the core install is unchanged, as the epic's
definition of done requires.

Which physical transforms the package implements at all (degree days, wind profile,
capacity factors) is the scope question of
[#1159](https://github.com/monocongo/climate_indices/issues/1159), which overlaps
`xclim`, `pvlib`, and `windpowerlib`. Nothing here depends on its answer: the package
boundary, the aggregation seam, the support rules, the calendar, and the units hold
whether or not those transforms are kept, so #1159 can only decide implement-versus-defer
and never a name. The function names, argument conventions, and degree-day contract for
those transforms are fixed in the internal energy subsystem design note
(`docs/design/energy-subsystem.md`); only whether they are implemented is provisional on
#1159, and the default constant values are confirmed by the oracle survey
([#1160](https://github.com/monocongo/climate_indices/issues/1160)). That note stays in
the repository rather than this site because it also plans unimplemented indices, the
arrangement {doc}`0005-fire-module-api` chose for fire.

## Consequences

This amends {doc}`0001-dual-numpy-xarray-api` only for the energy family, as
ADR-0005 did for fire and ADR-0013 for flood. `standardized_index()` gains one keyword
whose default reproduces its existing output, which is why this record can land the seam
change without touching SPI or SPEI behavior.

`aggregation` is not exposed on the xarray or CLI surfaces, because `standardized_index()`
is a NumPy entry point only. The energy xarray adapters
([#1174](https://github.com/monocongo/climate_indices/issues/1174)) carry it for the
indices that need it.

A reader who finds decisions 1 and 3 through 6 not yet in code is looking at a contract
awaiting its implementation, not at drift.

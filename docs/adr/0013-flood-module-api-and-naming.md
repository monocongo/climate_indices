# Flood APIs are namespaced in the `flood` package, with `edi` also at the package root

## Status

Amended: `flood.effective_precipitation()` and the `flood` package landed in
#1105; the remaining public names and their xarray adapters landed with FLOOD-09
through FLOOD-13 (#1106–#1110), and the root `climate_indices.edi` route is
live. The API decision is unchanged.

The flood family ([#1098](https://github.com/monocongo/climate_indices/issues/1098))
adds effective precipitation (PE), the Effective Drought Index (EDI), the Flood
Index (I_F), and the Antecedent Precipitation Index (API).

[ADR-0005](./0005-fire-module-api.md) settled the same question for fire weather
and applies directly here. A family of indices that share a physical basis and
evolve together gets its own namespaced package rather than being spread through
the drought-oriented `indices.py` and `compute.py`, whose top-level names it
would crowd. That precedent also answers the family-naming question this issue
was opened to resolve: `climate_indices.fire` holds KBDI, which is physically a
drought/fuel-dryness index, because the *application* names the family. EDI is
the same case. It is a standardized drought index, but it is built on the same
effective-precipitation kernel as I_F, and every artifact of this work — the epic
scope table, the milestones, the board, and the published
`docs/flood_applications.md` — already calls the family flood, so a neutral name
such as `wetness` would invent vocabulary for a family that has a name.

## Decision

Flood computations live in `climate_indices.flood`, a package with a public
facade over underscore-prefixed implementation modules, mirroring
`climate_indices.fire`, and are imported as `from climate_indices import flood`.
The public names are `flood.effective_precipitation()`, `flood.edi()`,
`flood.flood_index()`, and `flood.antecedent_precipitation_index()`.

- **PE is a public index**, not internal plumbing: the epic's scope table lists
  it as a deliverable in its own right, and EDI and I_F are both defined on it.
- **`flood_index()`, never `if_()` or `if()`** — an acronym that collides with
  the language keyword is not a usable public name.
- **`antecedent_precipitation_index()`, never `api()`** — `api` already means
  application programming interface throughout this repository's documentation
  and `typed_public_api.py`, and the collision would be read as one.
- **`edi()`, not `effective_drought_index()`** — it sits beside its sibling
  `eddi` at the package root, and the short form is how users search for it.

Functions are reached through the package, never as unqualified package-root
functions, which is the rule `typed_public_api.py` already states for fire:
family API additions do not go into `typed_public_api.py` and do not re-export
from the package root. **`edi` is the single deliberate exception.** It is a
standardized drought index of the same kind as SPI, SPEI, and EDDI, and a user
looking for it will look for it beside them, so it also rides at
`climate_indices.edi`.

**The root route lands with the xarray adapter (#1108), not with the NumPy
implementation (#1106).** Every index function exported from the package root
today carries both a NumPy and an xarray path, declared as an overload pair in
`typed_public_api.py` (PCI is a hand-written wrapper, but it has both). Adding a
NumPy-only root name would make EDI the first root name without an xarray path,
a precedent worth more than the interim convenience of exposing it early.
`flood.edi()` is available from #1106 for callers and for the #1104 fixture
tests; the root route follows once both paths exist.

Beta xarray paths stay on their `flood.<name>` route, as the fire paths stay on
`fire.<name>`, and are beta under [ADR-0012](./0012-xarray-api-stays-beta-through-3.0.0.md)'s
policy until that policy changes.

Argument names, units, and signatures are recorded in the flood subsystem design
note (`docs/design/flood-subsystem.md`), which stays in the repository rather
than this site because it also plans unimplemented indices — the arrangement
[ADR-0005](./0005-fire-module-api.md) chose for fire.

## Consequences

`climate_indices.edi` did not exist before the xarray adapter landed (#1108);
it is live now. Nothing is added to `indices.py`, so the existing top-level
drought API is unchanged by this family, and a future reader who greps
`climate_indices.edi` finds a route that exists rather than one this record
merely planned.

This amends [ADR-0001](./0001-dual-numpy-xarray-api.md) only for the flood
family, as [ADR-0005](./0005-fire-module-api.md) did for fire. Named wrappers
over `indices.standardized_index()` are outside this decision and stay open
([#1132](https://github.com/monocongo/climate_indices/issues/1132)).

The CLI is not part of this subsystem: as with fire, flood indices reach the CLI
through the existing `climate_indices` CLI only where an xarray adapter and a CF
registry entry exist ([#1115](https://github.com/monocongo/climate_indices/issues/1115)),
so the CLI consumes the package rather than being a component of it.

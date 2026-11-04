# Context Map

## Contexts

- [climate_indices (core library)](./src/climate_indices/CONTEXT.md) — computes standardized drought/moisture indices (SPI, SPEI, PNP, EDDI, Palmer family) from climate observations; the shipped, tested product. The [fire subsystem](./docs/design/fire-subsystem.md) belongs to this context and is implemented; its NumPy routes are stable and its xarray adapters are beta.
- [Climate Index Explorer](./docs/explorer/CONTEXT.md) — a planned interactive tool for exploring precomputed/on-demand index results; currently in research/planning (see `docs/research/interactive-climate-explorer-landscape.md`), no source tree yet

## Application namespaces in the core library

`climate_indices.drought` owns the drought, Palmer, and PET implementations.
Its typed facade exposes the same function objects as the package root.
The generic `standardized_index` and `fit_diagnostics` functions remain there for now.
Shared computation, vocabulary, and statistical services remain at the root.
See [ADR-0027](./docs/adr/0027-drought-namespace-and-module-migration.md) for the module move and temporary aliases.

Application names determine ownership: KBDI stays in `fire`, and EDI stays in `flood`.
EDDI belongs to `drought` and is distinct from EDI.
These namespaces are parts of the core library, not separate bounded contexts.

## Relationships

- **Explorer → core library**: The explorer wraps and drives `climate_indices` — its "Index Capability" concept describes which core-library indices, parameters, and periodicities it exposes to a user. The explorer has no vocabulary or authority over how indices are computed; that belongs entirely to the core library context.
- **Core library → Explorer**: No dependency in this direction. The core library is usable and versioned independently of whether the explorer exists.

## Note

`docs/explorer/CONTEXT.md` is a placeholder location — the explorer has no source code yet. When implementation begins (likely under `src/explorer/` or similar), move its `CONTEXT.md` alongside that code and update this map.

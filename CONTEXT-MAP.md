# Context Map

## Contexts

- [climate_indices (core library)](./src/climate_indices/CONTEXT.md) — computes standardized drought/moisture indices (SPI, SPEI, PNP, EDDI, Palmer family) from climate observations; the shipped, tested product
- [Climate Index Explorer](./docs/explorer/CONTEXT.md) — a planned interactive tool for exploring precomputed/on-demand index results; currently in research/planning (see `docs/research/interactive-climate-explorer-landscape.md`), no source tree yet

## Relationships

- **Explorer → core library**: The explorer wraps and drives `climate_indices` — its "Index Capability" concept describes which core-library indices, parameters, and periodicities it exposes to a user. The explorer has no vocabulary or authority over how indices are computed; that belongs entirely to the core library context.
- **Core library → Explorer**: No dependency in this direction. The core library is usable and versioned independently of whether the explorer exists.

## Note

`docs/explorer/CONTEXT.md` is a placeholder location — the explorer has no source code yet. When implementation begins (likely under `src/explorer/` or similar), move its `CONTEXT.md` alongside that code and update this map.

# Drought APIs use the `drought` package, with temporary aliases for public modules

## Status

Accepted. [#1555](https://github.com/monocongo/climate_indices/issues/1555)
moves the modules and adds the facade described here.
This decision changes import paths, not numerical behavior.

## Context

Fire, flood, energy, and reliability have application-named packages.
The original drought modules remained at the package root.
A matching `drought` package makes their ownership explicit without changing existing root function imports.

The application names the family, not its input variable.
`precip` would exclude the temperature, atmospheric-demand, and water-balance methods that belong beside SPI.
As in {doc}`0013-flood-module-api-and-naming`, KBDI stays in `fire`, and EDI stays in `flood`.
EDDI belongs in `drought`; EDI and EDDI are different indices.

## Decision

### Module ownership

Move these nine modules under `climate_indices.drought`:

| Previous module | Canonical module | Compatibility at the previous path |
| --- | --- | --- |
| `climate_indices.indices` | `climate_indices.drought.indices` | Warning and identity alias until 4.0.0. |
| `climate_indices.palmer` | `climate_indices.drought.palmer` | Warning and identity alias until 4.0.0. |
| `climate_indices.self_calibration` | `climate_indices.drought.self_calibration` | Warning and identity alias until 4.0.0. |
| `climate_indices.eto` | `climate_indices.drought.eto` | Warning and identity alias until 4.0.0. |
| `climate_indices.pm_eto` | `climate_indices.drought.pm_eto` | Warning and identity alias until 4.0.0. |
| `climate_indices._palmer_common` | `climate_indices.drought._palmer_common` | None. The private path is removed. |
| `climate_indices._palmer_duration` | `climate_indices.drought._palmer_duration` | None. The private path is removed. |
| `climate_indices._palmer_pdi` | `climate_indices.drought._palmer_pdi` | None. The private path is removed. |
| `climate_indices._palmer_wells` | `climate_indices.drought._palmer_wells` | None. The private path is removed. |

Shared services such as `compute`, `vocabulary`, `lmoments`, and `utils` stay at the root.
The generic `standardized_index` and `fit_diagnostics` functions remain with the drought API for now.
PET implementations also remain here, although other families use them.
This move does not introduce a separate generic-standardization or evapotranspiration package.

### One typed facade

`climate_indices.drought` exposes the same `typed_public_api` function objects as the root:
`spi`, `spei`, `spai`, `pdsi`, `eddi`, `percentage_of_normal`,
`pet_thornthwaite`, `pet_hargreaves`, `pet_penman_monteith`, `pci`,
`standardized_index`, and `fit_diagnostics`.
It also exposes the same `HumidityInputs` and `RadiationInputs` classes as the root.
These are re-exports, not new wrappers or copies.
For example, `climate_indices.drought.spi is climate_indices.spi` is true.
Existing root function imports remain supported and do not emit a migration warning.

The facade resolves typed functions lazily through module-level `__getattr__`.
This avoids an initialization cycle when the typed API and adapters import drought implementation modules.
The NumPy-only implementation routes remain available under `drought.indices` and `drought.palmer`.
The facade does not add an scPDSI adapter or change any function signature.

### Temporary public-module aliases

The five old public module paths are deprecated in 3.x and removed in 4.0.0.
Each compatibility module emits `ClimateIndicesDeprecationWarning` when imported and installs its canonical module in `sys.modules` under the old name.
Both names therefore resolve to one module object.
A wrapper that copies module attributes would not meet this contract.

Module identity preserves monkeypatch behavior: changing an attribute through an old import changes the canonical module too.
The aliases also let old pickle records resolve public module paths during the deprecation period.
They do not provide aliases for private modules or promise compatibility after 4.0.0.
Shared vocabulary objects retain their existing identity and compatibility module metadata.

## Consequences

Users can migrate imports without changing calculations, array layouts, calibration periods, or backend selection.
Maintained examples and API reference targets use the canonical paths.
The {doc}`migration guide </deprecations/api-changes>` explains warning detection and import changes.

This amends the current module names in ADRs 0001, 0011, 0013, 0019, 0020, and 0026.
Their computation, family-scope, and dual-API decisions remain unchanged.
Historical descriptions of earlier implementations retain their original names.

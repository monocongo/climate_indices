# Fire subsystem

## Decision

Fire-weather and fuel-dryness indices live in one flat, namespaced module:

```text
src/climate_indices/fire.py
from climate_indices import fire
```

`fire.py` is the stable NumPy layer for this family. It is an intentional
family-level exception to the drought-oriented `indices.py`/`compute.py`
placement in [ADR-0001](../adr/0001-dual-numpy-xarray-api.md), recorded in
[ADR-0005](../adr/0005-fire-module-api.md). Fire functions are never
re-exported as unqualified package functions: use
`fire.fosberg_ffwi()`, never `climate_indices.fosberg_ffwi()`.

Keep the module flat until it exceeds roughly 1,500 lines or CFFWIS recurrence
state needs isolated implementation modules. At that point, promote it to a
`fire` package without changing `from climate_indices import fire` or any
public function name. No fire CLI is part of this subsystem.

This family covers meteorological and climatological indices only. It excludes
NFDRS components such as ERC, BI, SC, and IC; fuel models; fire behaviour;
ignition; and occurrence prediction.

## API tiers

- `fire.*` NumPy APIs are stable. They accept scalars or NumPy-compatible
  array-likes, broadcast elementwise inputs where physically meaningful, and
  return NumPy arrays.
- Xarray support is beta and added function by function. It preserves the
  `fire.<name>` public route, uses `xarray_adapter` for validation and metadata,
  and documents its Dask constraints. Recursive indices require one time chunk.
- `typed_public_api.py` remains the unqualified drought/moisture facade. It
  must not gain unqualified fire functions or package-level fire re-exports.
  Fire overloads and xarray dispatch stay with their namespaced facade.

The one multi-output exception is CFFWIS: NumPy returns a named
`CFFWISResult`; xarray returns an `xarray.Dataset` with `ffmc`, `dmc`, `dc`,
`isi`, `bui`, `fwi`, and `dsr` variables. `fwi` is an output variable required
by CFFWIS terminology, not a callable. There is no `fire.fwi()`.

The current `xarray_adapter` finalizes a single `DataArray`, so the CFFWIS
xarray route cannot use it as-is. It requires a multi-output extension that
calls the shared NumPy core once, then rewraps each of the seven variables
with its own `CF_METADATA` entry under the same validation and one-time-chunk
guarantees. Independent per-output adapters are not a substitute: CFFWIS is
one shared computation, not seven.

Stateful functions share one state contract, specified in
[#795](https://github.com/monocongo/climate_indices/issues/795): keyword-only
`initial_<code>` parameters with literature defaults (FFMC 85, DMC 6, DC 15,
KBDI 0), a `spin_up` parameter defaulting to the literature-standard behavior,
and `return_state: bool = False`. With `return_state=True` the final state is
returned alongside the result so a run can be continued without recomputing
the archive. #795 fixes the state object type and wet-spell handling; no index
may invent a different state-return convention.

## Names and input contracts

New configuration is keyword-only. Parameters name their SI units. Core
functions do not infer units. Xarray adapters validate/convert units at their
boundary before calling the NumPy core; conversions never occur in recurrence
kernels.

| Public name | Inputs | Output / accepted alternative |
| --- | --- | --- |
| `fosberg_ffwi(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, cap_at_100=True)` | °C, %, m s⁻¹ | dimensionless FFWI; no alternative units |
| `kbdi(precipitation, maximum_temperature, mean_annual_precipitation, *, units="metric")` | metric: mm day⁻¹, °C, mm year⁻¹ | metric moisture deficit in mm, range 0–200; `units="imperial"` accepts inches day⁻¹, °F, inches year⁻¹ and returns 0–800 hundredths of an inch |
| `ffmc(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, precipitation_mm)` | noon-LST °C, %, 10 m m s⁻¹, 24 h mm | dimensionless Fine Fuel Moisture Code |
| `duff_moisture_code(temperature_celsius, relative_humidity_percent, precipitation_mm, latitude_degrees_north, month)` | noon-LST °C, %, 24 h mm, degrees north, calendar month | dimensionless DMC |
| `drought_code(temperature_celsius, precipitation_mm, latitude_degrees_north, month)` | noon-LST °C, 24 h mm, degrees north, calendar month | dimensionless DC; distinct from package drought indices |
| `initial_spread_index(ffmc, wind_speed_meters_per_second)` | FFMC, 10 m m s⁻¹ | dimensionless ISI |
| `buildup_index(dmc, dc)` | DMC, DC | dimensionless BUI |
| `cffwis_fwi(isi, bui)` | ISI, BUI | dimensionless Canadian Fire Weather Index |
| `daily_severity_rating(cffwis_fwi)` | Canadian FWI | dimensionless DSR |
| `cffwis(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, precipitation_mm, latitude_degrees_north, *, initial_ffmc=85.0, initial_dmc=6.0, initial_dc=15.0, spin_up=None, return_state=False)` | CFFWIS weather inputs above; `initial_*` and `spin_up` follow the shared state contract | `CFFWISResult` plus final state when `return_state=True`; xarray counterpart accepts `tas`, `hurs`, `sfcWind`, `pr`, and optional `lat`, returning `Dataset` |
| `hot_dry_windy(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, height_agl_meters, *, level_axis=-1)` | vertical profiles in °C, %, m s⁻¹, m AGL | hPa m s⁻¹; all levels must identify the lowest 500 m AGL |
| `haines_index(temperature_lower_celsius, temperature_upper_celsius, dewpoint_lower_celsius, *, variant)` | pressure-level °C inputs selected by `variant` | integer 2–6; `variant` is `"low"`, `"mid"`, or `"high"`, never inferred by default |

Only `fosberg_ffwi()` is implemented today; the remaining rows are planned
contracts, not yet callable.

`fosberg_ffwi()` is weather-only and elementwise. KBDI, FFMC, DMC, DC, and
CFFWIS are daily recursive functions; their weather inputs must be ordered in
time. ISI, BUI, Canadian FWI, DSR, HDW, and Haines are derived from concurrent
inputs and do not carry state.

## Validation, metadata, and errors

Use the existing hierarchy: `InvalidArgumentError` for invalid configuration
or incompatible arguments; `DataShapeError` for domain-invalid NumPy shapes;
and `CoordinateValidationError` for invalid xarray dimensions, coordinates, or
chunking. Elementwise invalid observations produce the documented missing
output. Do not add fire-specific exception classes.

Fire outputs have no CF `standard_name`. Xarray metadata comes exclusively from
`CF_METADATA`: each adapter's `long_name`, units, description, and references
come from its registry entry, never hand-written in an adapter. The current
`CFAttributes` schema holds only `long_name`, `units`, and `references` and
has no fire entries; [#798](https://github.com/monocongo/climate_indices/issues/798)
extends the schema with `description` and adds the fire entries, and no fire
adapter ships before that lands.

`drought_code()` names the CFFWIS component only. It neither accepts a climate
calibration period nor means SPI, SPEI, PDSI, or any other drought index.

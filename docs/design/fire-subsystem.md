# Fire subsystem

## Decision

Fire-weather and fuel-dryness indices live in one namespaced package:

```text
src/climate_indices/fire/
    __init__.py   public facade
    _cffwis.py    Canadian Forest Fire Weather Index System
    _fosberg.py   Fosberg FFWI
    _hdw.py       Hot-Dry-Windy Index
    _kbdi.py      Keetch-Byram Drought Index
    _common.py    shared coercion, seed, and gap-policy helpers
    _units.py     CF units-attribute conversion
from climate_indices import fire
```

The package facade is the stable NumPy layer for this family. It is an
intentional family-level exception to the drought-oriented
`indices.py`/`compute.py` placement in
[ADR-0001](../adr/0001-dual-numpy-xarray-api.md), recorded in
[ADR-0005](../adr/0005-fire-module-api.md). Fire functions are never
re-exported as unqualified package functions: use
`fire.fosberg_ffwi()`, never `climate_indices.fosberg_ffwi()`.

ADR-0005 kept the family flat until it passed roughly 1,500 lines or CFFWIS
recurrence state needed isolated implementation modules. The CFFWIS behavior
indices (#804) crossed that line, so the family was promoted to the package
above without changing `from climate_indices import fire` or any public
function name. The implementation modules carry a leading underscore so
`fire.kbdi` and `fire.cffwis` stay bound to the functions rather than the
modules. No fire CLI is part of this subsystem.

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

State initialization, final-state extraction, spin-up, and wet-spell state
follow [ADR-0006](../adr/0006-fire-recursive-state-and-execution.md). No index
may invent a different state-return convention.

KBDI had the same "current adapter can't use it as-is" problem from a
different direction: `kbdi()`'s `units` argument selects between two
registry entries (`kbdi` for metric, `kbdi_imperial` for imperial) at *call*
time, but `xarray_adapter`'s `cf_metadata` parameter binds one fixed
`CFAttributes` dict at *decoration* time (#798). Rather than widen the
generic `@xarray_adapter` decorator, KBDI's adapter (#801) is a manual
function on `fire.kbdi()` itself — following the precedent of
`xarray_adapter.pet_thornthwaite`/`pet_hargreaves`, which also bypass the
decorator — that resolves the registry key per call from `units` and calls
`xr.apply_ufunc` directly. Unlike those PET functions, it does not need
`vectorize=True`: `kbdi()`'s NumPy core already vectorizes over an arbitrary
spatial shape internally, so the adapter dispatches one call per Dask spatial
chunk with the full `time` axis rather than looping per grid cell. The same
per-call resolution need will recur for the CFFWIS extension above.

HDW's adapter (#809) is manual for a different reason: it *reduces* a
dimension. `hot_dry_windy()` collapses a vertical `level` axis to its layer
maximum, and the generic decorator's Dask path only maps `time` to `time` —
it has no way to shrink a core dimension. HDW has a single fixed registry
entry, no per-call resolution, and no state or time semantics at all, so its
adapter is simpler than KBDI's: `xr.apply_ufunc` calls the shared private
kernel `_hot_dry_windy_layer_max` as the kernel, with the caller-named
`level_dim` as the sole core dimension on all four inputs and no output core
dimension. Using the silent kernel rather than the public function keeps one
xarray operation from logging once per Dask block. `level_dim`, like KBDI's
`time_dim`, must be a single Dask chunk; every other dimension, including
`time` if present, is an ordinary passthrough. Inputs are matched with
xarray's exact join, so shared dimensions must carry identical, identically
ordered coordinate labels; unlike KBDI, HDW does not align or reindex them.

## Stateful recurrence contract

KBDI implements the contract today, with `KBDIState` carrying `kbdi`,
`wet_spell_precipitation`, `trailing_gap_days`, and `units`; FFMC, DMC, and DC
implement it with their single code value plus `trailing_gap_days`. The
`cffwis()` orchestrator adds `CFFWISState`, which nests the three single-code
states so each keeps its own gap bookkeeping while a combined call resumes
exactly what the three separate functions would. `CFFWISResult` returns any
requested subset of the seven outputs; a name not selected is `None`.
Single-output APIs take keyword-only `initial_<code>: float or spatial
field | None`, `initial_state`, `return_state=False`, and `spin_up=0`. `None` selects the
literature seed: KBDI 0, FFMC 85, DMC 6, or DC 15. `initial_state` restores the
full named state, including auxiliary values such as KBDI's cumulative
wet-spell precipitation, and cannot be combined with a seed. It is the only
lossless way to append a new observation period.

A stateful call normally returns its index array. With `return_state=True`, it
returns a named `{Index}Result(values, state)`. `CFFWISResult` keeps its named
index outputs and returns its state under the same flag. State types are
algorithm-specific frozen dataclasses, never anonymous tuples or xarray
`Dataset` objects. `spin_up` computes but omits that many leading input days;
there is no universal scientifically valid nonzero default, so callers discard
a study-appropriate transient.

```python
import numpy as np

# illustrative of the shared stateful API; change the arrays and rerun on the next period
history = fire.kbdi(precipitation_1980_2020, temperature_1980_2020, mean_annual_precipitation, return_state=True)
next_year = fire.kbdi(precipitation_2021, temperature_2021, mean_annual_precipitation, initial_state=history.state, return_state=True)
whole = fire.kbdi(precipitation_1980_2021, temperature_1980_2021, mean_annual_precipitation)
np.testing.assert_array_equal(np.concatenate((history.values, next_year.values)), whole)
```

The implementation vectorizes each daily update over spatial cells and loops
only over time. It has a required pure-NumPy baseline; `numba` is not an
optional dependency unless later benchmark evidence justifies its support cost.

The moisture codes evaluate the published equations in the source's
operational units (km/h wind, mm rain, degrees Celsius) and follow the NRCan
reference implementation where it departs from the printed report: FFMC's
moisture-content conversion uses the exact `250 * 59.5 / 101` rather than the
printed `147.2`, in both directions, and DMC's post-rain conversion uses the
reference code's `43.43 * (5.6348 - ln(Wmr - 20))` form of Eq. 15. The
reference lineage is `cffdrs_r` and its Python port `cffdrs_py`; the frozen
vectors in `tests/test_fire_cffwis_moisture.py` pin that port's commit.

The CFFWIS moisture codes select their month- and latitude-dependent tables per
cell. DMC uses five effective-day-length rows: 46 N (`latitude > 30`), 20 N
(`10 < latitude <= 30`), the equator (`-10 < latitude <= 10`), 20 S
(`-30 < latitude <= -10`), and 40 S (`latitude <= -30`). DC uses three
day-length-adjustment rows: north (`latitude > 20`), equator
(`-20 < latitude <= 20`), and south (`latitude <= -20`). A NaN latitude means
the cell has no usable day-length band, so its output stays NaN and its
recurrence never starts — the same treatment as a NaN KBDI climatology.

## Missing data and gaps

Missing days are governed by [ADR-0007](../adr/0007-fire-missing-data-policy.md).
The default `nan_policy="propagate"` never bridges a gap: missing days have
NaN outputs, and the first valid day after an interior gap resumes with a NaN
state, so every later output is NaN. `nan_policy="bridge"` with
`max_gap_days=N` skips interior and trailing runs of at most `N` days with the
state unchanged, and poisons from the first run that exceeds it. The limit is
counted over the continuous series: the state returned by a bridged run
carries `trailing_gap_days` (`None` while no valid day has started the
recurrence), and a resumed call counts its leading missing days against what
remains of `max_gap_days`, so a run spanning an append boundary either stays
bridged or poisons exactly as the one-shot series would. Leading missing days
are unbounded only before the recurrence starts. Interpolation is deliberately
not a policy:
callers fill inputs upstream so the fill stays visible. A day is missing when
any time-varying weather input is NaN or elementwise-invalid for the index
(relative humidity outside [0, 100], negative wind speed); infinity raises
`InvalidArgumentError` instead. Sub-freezing and inactive-index days are valid
observations, and off-season periods must use the seasonal state
carry from #806 rather than NaN. Each stateful implementation must carry the
parametrized gap matrix recorded in ADR-0007.

## Xarray chunking

Future stateful xarray fire adapters validate every time-varying input with
`xarray_adapter._validate_dask_chunks()`. A Dask `time` dimension must be one
chunk, while spatial dimensions may remain chunked. The adapter raises
`CoordinateValidationError` with a rechunk command rather than silently
rechunking and materializing a large history.

## Names and input contracts

New configuration is keyword-only. Parameters name their SI units. Core
functions do not infer units. Xarray adapters validate/convert units at their
boundary before calling the NumPy core; conversions never occur in recurrence
kernels.

| Public name | Inputs | Output / accepted alternative |
| --- | --- | --- |
| `fosberg_ffwi(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, cap_at_100=True)` | °C, %, m s⁻¹ | dimensionless FFWI; no alternative units |
| `kbdi(precipitation, maximum_temperature, mean_annual_precipitation=None, *, units="metric")` | metric: mm day⁻¹, °C, mm year⁻¹; omitting the mean derives it from at least 30 years of record | metric moisture deficit in mm, range 0–203.2 (the exact conversion of 0–800 hundredths of an inch); `units="imperial"` accepts inches day⁻¹, °F, inches year⁻¹ and returns 0–800 hundredths of an inch |
| `ffmc(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, precipitation_mm)` | noon-LST °C, %, 10 m m s⁻¹, 24 h mm | dimensionless Fine Fuel Moisture Code |
| `duff_moisture_code(temperature_celsius, relative_humidity_percent, precipitation_mm, latitude_degrees_north, month)` | noon-LST °C, %, 24 h mm, degrees north, calendar month | dimensionless DMC |
| `drought_code(temperature_celsius, precipitation_mm, latitude_degrees_north, month)` | noon-LST °C, 24 h mm, degrees north, calendar month | dimensionless DC; distinct from package drought indices |
| `initial_spread_index(ffmc, wind_speed_meters_per_second)` | FFMC, 10 m m s⁻¹ | dimensionless ISI |
| `buildup_index(dmc, dc)` | DMC, DC | dimensionless BUI |
| `cffwis_fwi(isi, bui)` | ISI, BUI | dimensionless Canadian Fire Weather Index |
| `daily_severity_rating(cffwis_fwi)` | Canadian FWI | dimensionless DSR |
| `cffwis(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, precipitation_mm, latitude_degrees_north, month, *, initial_ffmc=None, initial_dmc=None, initial_dc=None, initial_state=None, return_state=False, spin_up=0, nan_policy="propagate", max_gap_days=0, outputs=None)` | CFFWIS weather inputs above; `initial_*`, `spin_up`, and `nan_policy`/`max_gap_days` follow the shared stateful contract, and `month` is required by the DMC/DC day-length tables | `CFFWISResult` with the requested subset of the seven named outputs (`None` for names not requested) plus the combined `CFFWISState` when `return_state=True`; xarray counterpart accepts `tas`, `hurs`, `sfcWind`, `pr`, and optional `lat`, returning `Dataset` |
| `hot_dry_windy(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, height_agl_meters, *, level_axis=-1)` | vertical profiles in °C, %, m s⁻¹, m AGL | hPa m s⁻¹; all levels must identify the lowest 500 m AGL |
| `haines_index(temperature_lower_celsius, temperature_upper_celsius, dewpoint_lower_celsius, *, variant)` | pressure-level °C inputs selected by `variant` | integer 2–6; `variant` is `"low"`, `"mid"`, or `"high"`, never inferred by default |

`fosberg_ffwi()`, `hot_dry_windy()`, `kbdi()`, `ffmc()`,
`duff_moisture_code()`, `drought_code()`, `initial_spread_index()`,
`buildup_index()`, `cffwis_fwi()`, `daily_severity_rating()`, and `cffwis()`
are implemented today; the Haines row is a planned contract (#810), not yet
callable. The stateful rows
accept the keyword-only missing-data arguments `nan_policy="propagate"` and
`max_gap_days=0` described above.

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
come from its registry entry, never hand-written in an adapter.
[#798](https://github.com/monocongo/climate_indices/issues/798) extended
`CFAttributes` with `description` and `climate_indices_variant`, and added
entries for the indices implemented today: `kbdi` (metric), `kbdi_imperial`,
`ffwi`, `hdw`, and the seven CFFWIS entries `ffmc`, `dmc`, `dc`, `isi`,
`bui`, `fwi`, and `dsr`. KBDI uses two keys, not one, because
`kbdi()` returns two different unit scales from the same function;
`climate_indices_variant` distinguishes them, and the CFFWIS entries all
carry `cffwis_classic` to distinguish them from a future FWI2025 variant.
The Haines Index (`haines`, #810) has no registry entry yet — its elevation
variant is an open question best resolved against a real implementation, not
guessed ahead of it. FWI is registered as the CFFWIS output name; the
Fosberg index stays `ffwi`. No adapter for an index ships before its registry
entry lands.

`drought_code()` names the CFFWIS component only. It neither accepts a climate
calibration period nor means SPI, SPEI, PDSI, or any other drought index.

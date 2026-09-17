# Fire subsystem

## Decision

Fire-weather and fuel-dryness indices live in one namespaced package:

```text
src/climate_indices/fire/
    __init__.py   public facade
    _cffwis.py    Canadian Forest Fire Weather Index System
    _fosberg.py   Fosberg FFWI
    _haines.py    Haines Index
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
recurrence state needed isolated implementation modules. The CFFWIS moisture
codes (#803) crossed that line; #803's record deferred the promotion to the
remaining CFFWIS work (#804), which landed the package
above without changing `from climate_indices import fire` or any public
function name. The implementation modules carry a leading underscore so
`fire.kbdi` and `fire.cffwis` stay bound to the functions rather than the
modules. No fire CLI is part of this subsystem itself: fire indices are
surfaced through the existing `climate_indices` CLI only where an xarray
adapter and a CF registry entry exist (KBDI as `--index kbdi`, #802), so the
CLI consumes the fire package rather than being a component of this subsystem.

## Scope and boundary

The dividing line is not drought versus wildfire but **meteorological and
climatological indices** versus **operational fire-danger and fire-behaviour
modeling**.

In scope:

- Indices computable from standard meteorological and reanalysis fields
- Deterministic, well-published algorithms with authoritative reference code
- Anything that fits the existing NumPy + xarray + CF metadata + Dask pattern

Out of scope:

- NFDRS Ignition Component, Spread Component, Energy Release Component, and
  Burning Index, which NWCG defines in terms of live and dead fuel moisture
  and fuel models rather than weather alone
- Externally supplied fuel-model catalogs, live fuel state, and operational
  calibration against fuel loads or fire-occurrence records — not the
  weather-driven dead-fuel-moisture recursions (CFFWIS FFMC, DMC, DC) this
  package already computes from weather alone
- Fire-behaviour simulation and rate-of-spread modeling
- Ignition and occurrence prediction
- FWI2025 next-generation reformulations, at least initially

A weather-only index such as the McArthur Forest Fire Danger Index would be an
in-scope proposal. NFDRS Energy Release Component would not: it needs fuel
models and live and dead fuel state this package does not model.

A separate `fire_weather_indices` repository becomes justified when two or more
of these are true:

1. An implementation requires fuel-model or fuel-state data structures
2. An implementation requires operational calibration against fire-occurrence
   records
3. The fire code exceeds roughly 30 percent of the package's source volume
4. Fire-specific dependencies would be forced on all `climate_indices` users
5. The release cadence needs to diverge

Until then, an out-of-scope fire proposal is not rejected on merit: open an
issue in this repository labeled `fire-weather` recording it as a candidate
for the eventual `fire_weather_indices` split, rather than relitigating the
boundary here. [CONTRIBUTING.md](../../CONTRIBUTING.md#fire-weather-scope)
records the triage response.

## API tiers

- `fire.*` NumPy APIs are stable. They accept scalars or NumPy-compatible
  array-likes, broadcast elementwise inputs where physically meaningful, and
  return NumPy arrays.
- Xarray support is beta and added function by function. It preserves the
  `fire.<name>` public route, uses `validation.py` for input-kind, time-axis,
  and chunk checks and `xarray_adapter` for metadata,
  and documents its Dask constraints. Recursive indices require one time chunk.
- `typed_public_api.py` remains the unqualified drought/moisture facade. It
  must not gain unqualified fire functions or package-level fire re-exports.
  Fire overloads and xarray dispatch stay with their namespaced facade.

The one multi-output exception is CFFWIS: NumPy returns a named
`CFFWISResult`; xarray returns an `xarray.Dataset` with `ffmc`, `dmc`, `dc`,
`isi`, `bui`, `fwi`, and `dsr` variables. `fwi` is an output variable required
by CFFWIS terminology, not a callable. There is no `fire.fwi()`.

The CFFWIS xarray route (#807) is a manual multi-output adapter in
`_cffwis.py`, following the KBDI/HDW precedent rather than widening the
generic decorator: one `xr.apply_ufunc` call runs the shared NumPy core once
per Dask spatial block, and each selected output is rewrapped with its own
`CF_METADATA` entry under the same validation and one-time-chunk guarantees.
Independent per-output adapters are not a substitute: CFFWIS is one shared
computation, not seven. The adapter infers `month` from the time coordinate
and latitude from a `lat`/`latitude` coordinate, converts CF `units`
attributes on temperature and precipitation, and warns when a daily time
coordinate is clearly not noon-referenced.

The design table's planned xarray names (`tas`, `hurs`, `sfcWind`, `pr`, `lat`)
are not the shipped signature: one `fire.cffwis()` serves both routes, so the
DataArray path keeps the NumPy parameter names (`temperature_celsius`,
`relative_humidity_percent`, `wind_speed_meters_per_second`,
`precipitation_mm`, `latitude_degrees_north`), with `latitude_degrees_north`
and `month` optional only when they can be inferred from coordinates. A
second set of CF short names would have meant two public spellings for the
same call.

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

Haines' adapter is elementwise, like Fosberg's, but it goes through
`xr.apply_ufunc` — with no core dimension and therefore no chunk constraint
— rather than the generic `@xarray_adapter` decorator, which treats `time` as
the core dimension and would demand a time axis Haines does not have. The
silent kernel `_haines_from_levels` is the apply_ufunc target for the
same observability reason as HDW's, and the registry entry is resolved from
the validated `variant` at call time, the KBDI pattern. Elevation-driven
variant selection stays in the NumPy layer (`haines_index_from_profile()`),
where the pressure axis is explicit.

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
observations, and off-season periods use the seasonal state
carry rather than NaN (see [Seasonal carry and overwintering](#seasonal-carry-and-overwintering)).
Each stateful implementation must carry the
parametrized gap matrix recorded in ADR-0007.

## Seasonal carry and overwintering

The fire season is a caller-supplied policy, not a property of the weather
data: the DC has no calendar, no snow input, and no single threshold that fits
every region. `drought_code()` therefore takes a keyword-only
`in_season` boolean mask, time-first and broadcast against the weather inputs,
and the seasonal carry contract is
[ADR-0010](../adr/0010-seasonal-carry-is-an-explicit-mask.md). `None` treats
every day as in-season, which is the default and leaves the recurrence
unchanged.

Off-season days are neither observations nor missing days. The recurrence
state is frozen, the output emits the carried DC rather than a NaN, and the
mask is the only record of which is which — so NaN keeps its ADR-0007 meaning
of missing or poisoned. An off-season day never counts against
`max_gap_days` and never poisons, and off-season weather has no effect on the
code. A cell whose recurrence has not started yet has no carried value, so it
stays NaN, as any day before its first valid observation does.

Overwintering is the start-up half and is separate from the recurrence:
`overwinter_drought_code()` applies the Lawson and Armitage (2008)
overwintering method to the final autumn DC and the overwinter precipitation
total, and the caller passes its result back as `initial_dc` (or the state's
`dc`) for the next season. Chaining seasons this way is the ADR-0006 append
contract, and overwintering is opt-in: a caller who supplies neither a mask
nor a start-up value gets the same series as before. Only the DC is
overwintered; the FFMC and DMC are assumed to reach saturation over winter.

```python
fall = fire.drought_code(temperature, precipitation, latitude, month,
                         in_season=season_mask, return_state=True)
spring_dc = fire.overwinter_drought_code(fall.state.dc, winter_precipitation_mm)
next_season = fire.drought_code(temperature_next, precipitation_next,
                                latitude, month_next, initial_dc=spring_dc)
```

A single continuous call with `in_season` freezes the DC over the off-season
but does not apply the overwintering equation, so the next season resumes from
the autumn DC. That is the correct carry for a caller who wants the recurrence
uninterrupted across a season boundary; it is not overwintering, and the
spring start-up it produces is the one the overwintering step exists to
replace.

## Xarray chunking

Stateful xarray fire adapters -- KBDI (#801) and CFFWIS (#807) today --
validate every time-varying input with
`climate_indices.validation.validate_dask_chunks()`. A Dask `time` dimension must be one
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
| `drought_code(temperature_celsius, precipitation_mm, latitude_degrees_north, month, *, in_season=None)` | noon-LST °C, 24 h mm, degrees north, calendar month, boolean season mask | dimensionless DC; distinct from package drought indices |
| `overwinter_drought_code(final_fall_dc, overwinter_precipitation, *, carry_over_fraction=0.75, wetting_efficiency=0.75)` | previous season's final DC, overwinter precipitation total in mm | spring start-up DC, constrained to the seed 15 |
| `initial_spread_index(ffmc, wind_speed_meters_per_second)` | FFMC, 10 m m s⁻¹ | dimensionless ISI |
| `buildup_index(dmc, dc)` | DMC, DC | dimensionless BUI |
| `cffwis_fwi(isi, bui)` | ISI, BUI | dimensionless Canadian Fire Weather Index |
| `daily_severity_rating(cffwis_fwi)` | Canadian FWI | dimensionless DSR |
| `cffwis(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, precipitation_mm, latitude_degrees_north=None, month=None, *, initial_ffmc=None, initial_dmc=None, initial_dc=None, initial_state=None, return_state=False, spin_up=0, nan_policy="propagate", max_gap_days=0, outputs=None, time_dim="time")` | CFFWIS weather inputs above; `initial_*`, `spin_up`, and `nan_policy`/`max_gap_days` follow the shared stateful contract, and `month` is required for NumPy input by the DMC/DC day-length tables (inferred from the time coordinate on the xarray route) | `CFFWISResult` with the requested subset of the seven named outputs (`None` for names not requested) plus the combined `CFFWISState` when `return_state=True`; the xarray counterpart (#807)<br>accepts the same weather inputs as DataArrays, with `latitude_degrees_north` and `month` inferable from coordinates, and returns a `Dataset` (or a `CFFWISResult` when `return_state=True`) |
| `hot_dry_windy(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, height_agl_meters, *, level_axis=-1)` | vertical profiles in °C, %, m s⁻¹, m AGL | hPa m s⁻¹; all levels must identify the lowest 500 m AGL |
| `haines_index(temperature_lower_celsius, temperature_upper_celsius, dewpoint_celsius, *, variant, surface_pressure_hpa=None)` | pressure-level °C inputs selected so that each sits at the level its `variant` names (see below), plus optional hPa surface pressure | integer-valued float 2–6; `variant` is `"low"`, `"mid"`, or `"high"`, never inferred by default. Where `surface_pressure_hpa` lies below the variant's lower stability level the cell is NaN rather than scored from a below-ground level |
| `haines_index_from_profile(temperature_celsius, dewpoint_celsius, pressure_hpa, elevation_meters, *, pressure_axis=-1)` | vertical profiles in °C on one pressure axis, strictly decreasing pressures in hPa, and a terrain elevation in m | integer-valued float 2–6; selects the variant per cell from the elevation bands (low below 305 m, mid to 914 m, high above) and interpolates the profiles to the variant's levels in log pressure. The opt-in automatic form of the row above |

Haines level assignments, named by the variants' stability layers (the
moisture term pairs its dewpoint with whichever supplied temperature sits at
the moisture level):

| `variant` | `temperature_lower_celsius` | `temperature_upper_celsius` | `dewpoint_celsius` |
| --- | --- | --- | --- |
| `low` | 950 hPa | 850 hPa | 850 hPa |
| `mid` | 850 hPa | 700 hPa | 850 hPa |
| `high` | 700 hPa | 500 hPa | 700 hPa |

This table replaced the pre-implementation sketch in #810, which named the
third parameter `dewpoint_lower_celsius`. That name was wrong for the `low`
variant, where the moisture level (850 hPa) is the *upper* level of the
stability pair; a caller following it would have passed the 950 hPa dewpoint
and been scored silently. The sketch also could not express the issue's
automatic variant selection -- with three caller-chosen level values there is
nothing for the function to infer the variant from -- so that lives in
`haines_index_from_profile()`.

The variants' stability cut points are (4, 8), (6, 11), and (18, 22) °C and
their moisture cut points (6, 10), (6, 13), and (15, 21) °C, scored as
half-open bins (below the first cut point scores 1, below the second scores 2,
at or above it scores 3) so that non-integer lapse rates and depressions land
where the published integer tables put them. NWS's AWIPS GFE smart-init and
NOAA's LAPS `hainesindex.f` agree on every cut point, and LAPS is also the
precedent for withholding the index when a variant's levels lie below the
surface pressure.

`fosberg_ffwi()`, `hot_dry_windy()`, `kbdi()`, `ffmc()`,
`duff_moisture_code()`, `drought_code()`, `overwinter_drought_code()`,
`initial_spread_index()`, `buildup_index()`, `cffwis_fwi()`,
`daily_severity_rating()`, `cffwis()`, `haines_index()`, and
`haines_index_from_profile()` are implemented today. The stateful rows
accept the keyword-only missing-data arguments `nan_policy="propagate"` and
`max_gap_days=0` described above, and `drought_code()` additionally accepts the
`in_season` mask.

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
`xarray_adapter.build_output_attrs` drops a `standard_name` inherited from
the input attributes when the registry entry defines none, so a source
variable's name (for example `air_temperature`) never misdescribes a computed
fire output.
[#798](https://github.com/monocongo/climate_indices/issues/798) extended
`CFAttributes` with `description` and `climate_indices_variant`, and added
entries for the indices implemented today: `kbdi` (metric), `kbdi_imperial`,
`ffwi`, `hdw`, and the seven CFFWIS entries `ffmc`, `dmc`, `dc`, `isi`,
`bui`, `fwi`, and `dsr`. KBDI uses two keys, not one, because
`kbdi()` returns two different unit scales from the same function;
`climate_indices_variant` distinguishes them, and the CFFWIS entries all
carry `cffwis_classic` to distinguish them from a future FWI2025 variant.
The Haines Index has one entry per elevation variant -- `haines_low`,
`haines_mid`, and `haines_high`, each carrying `low`, `mid`, or `high` as its
`climate_indices_variant` -- because the variant is chosen per call and
decides which pressure layers the output describes. The adapter resolves the
entry from the `variant` argument at call time, as KBDI's does from `units`.
FWI is registered as the CFFWIS output name; the
Fosberg index stays `ffwi`. No adapter for an index ships before its registry
entry lands.

`drought_code()` names the CFFWIS component only. It neither accepts a climate
calibration period nor means SPI, SPEI, PDSI, or any other drought index.

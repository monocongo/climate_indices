# Climate Indices for Energy Applications

Climate indices describe the climate-driven component of energy supply and
demand, not actual generation or load. Generation and load also depend on
installed capacity, technology, curtailment, markets, and behaviour, so a
low-wind week or a cold spell does not establish how much electricity was
produced or consumed. This page is a companion to {doc}`wildfire_applications`
and {doc}`flood_applications`, which make the same distinction for fire and
flood.

**None of the indices described here measures or forecasts actual generation,
demand, or residual load.** Use them for monitoring and risk diagnostics of
the weather and climate conditions that drive the energy system, not as
generation or load estimates. The physical transforms use standard
meteorological inputs: 2 m temperature, near-surface and hub-height wind,
surface solar radiation, and precipitation. The standardized indices use
climate-derived energy proxies that the caller supplies. Wind capacity factors
also require a power curve; the hydropower proxy requires installed-capacity
weights
([energy-family epic #1157][energy-epic]).

## From meteorological drivers to energy droughts

A useful conceptual chain is:

```text
Meteorological drivers           T2m, wind 10/100 m, SSRD, precip           (inputs)
              ↓
Demand proxies                   HDD / CDD / EDD                            (available)
Resource transforms              hub-height wind, wind & PV capacity factors (available)
Hydro proxy                      capacity-weighted n-month precip           (available recipe)
              ↓
Standardized energy indices      SREPI, standardized demand, SRLI (available)
              ↓
Energy-drought events            run-theory onset / duration / magnitude    (available)
              ↓
Calibrated generation & load     hydro generation, electricity demand (MWh) (out of scope)
```

Read each arrow as a transformation of climate information, never as a model
of the energy system. Degree days and capacity factors express how favourable
the weather was for demand or for renewable production; turning them into
megawatt-hours needs installed capacity, plant characteristics, and observed
load, which this family does not carry ([energy-family epic #1157][energy-epic]).

## Energy droughts, SREPI, and SRLI

The energy-meteorology literature has adopted the construction behind SPI and
SPEI — a standardized anomaly over an accumulation window — for *energy
droughts*: sustained periods of low renewable production or high residual
load. [Allen & Otero (2023)][allen-otero-2023] define a Standardised Renewable
Energy Production Index (SREPI) and a Standardised Residual Load Index (SRLI),
with a reference implementation in the [`SEI` R package][sei]. A validated
Python implementation of SREPI, a standardized demand index, and SRLI is available.
The physical transforms make the family usable end to end from raw reanalysis.

Three indices standardize climate-derived energy proxies that the
caller supplies:

- **SREPI** standardizes a climate-derived renewable production proxy or a
  capacity factor, so low values mark weaker-than-usual renewable resource.
- **The standardized demand index** standardizes an energy demand proxy, such
  as energy degree days, so high values mark higher-than-usual
  weather-driven demand.
- **SRLI** standardizes a residual-load proxy — a demand proxy minus a
  renewable production proxy — so high values mark periods when weather-driven
  demand outpaces weather-driven supply. Residual load is signed, which is why
  SRLI admits negative inputs.

Standardization cannot separate climate effects from other influences.
If callers supply observed generation or load, the scores also reflect capacity
changes, curtailment, outages, markets, and behaviour. Such scores are not
weather-only diagnostics.

Standardization is unit-agnostic: it ranks or fits whatever series it is
handed, so capacity factors, degree days, and demand proxies from any source
are acceptable inputs
([ADR-0019](adr/0019-energy-module-api-and-aggregation.md)). The default is
the empirical (rank-based) distribution, which Allen & Otero (2023) recommend
above roughly 100 observations, with beta available for a bounded [0, 1] series
averaged with `aggregation="mean"` when there are too few. Capacity factors are
the common case; a bounded demand proxy qualifies when its source and windowed
values stay within [0, 1]. The beta option is not for accumulated production.
[#1169][energy-1169] added separate probability masses at exact 0 and 1 to the
standardization seam ([ADR-0024](adr/0024-beta-distribution-for-bounded-series.md)):
the continuous beta fit applies only between those endpoints, and a boundary value
scores from its own mass. {func}`climate_indices.energy.srepi` and
{func}`climate_indices.energy.standardized_demand_index` expose it
([#1171][energy-1171]).
SRLI uses the empirical distribution only. That is an API restriction rather
than a mathematical necessity: a rank-based standardization has no support
for a signed series to violate, the parametric families' support-limit and
zero-mass handling is not designed for one, and clipping negatives to zero
would score a different series. {func}`climate_indices.energy.srli` keeps
negatives as given and raises for any other distribution
([#1172][energy-1172]).

### Timescale and aggregation

The accumulation window ("Timescale" in prose; `scale` in code) works as it
does for SPI and SPEI, with one addition. The family is daily first: a weekly
index is daily input with a Timescale of 7, and a 30-day index is daily input
with a Timescale of 30; calendar months use monthly input. Hourly input uses the
`hourly` periodicity and the empirical distribution only, standardized in
hour-of-day x day-of-year strata ([#1177][energy-1177]). A 30-year record leaves about
30 calibration values per stratum, which is too few for a parametric fit.

Daily NumPy inputs to `indices.standardized_index()` must start on January 1
and use the internal 366-slot year layout. The energy NumPy wrappers
inherit this layout. Convert Gregorian arrays with
{func}`climate_indices.utils.transform_to_366day` before standardizing them;
see {doc}`data_requirements` for the conversion requirements. Do not pass an
ordinary Gregorian array directly: seasonal fitting would use the wrong day
slots. Hourly NumPy input follows the same rule in the 8784-slot layout:
`utils.CalendarPlan.from_year_span(year_start, years, len(values), steps_per_day=24).to_all_leap(values)`
converts a Gregorian hourly series. Existing xarray index adapters handle Gregorian time coordinates
separately, including the standardized energy adapters ([#1174][energy-1174]).

Precipitation is *summed* over its window, but a capacity factor is
*averaged*: a week's mean capacity factor stays a fraction of capacity in
[0, 1], where a week's sum does not. `indices.standardized_index()` therefore
takes a keyword-only `aggregation`, `"sum"` (the default) or `"mean"`, which
is available today ([ADR-0019](adr/0019-energy-module-api-and-aggregation.md)).
For rank-based standardization the choice does not change the result, since a
window's sum is its mean multiplied by the Timescale; it matters for what the
accumulated value means and for a bounded family such as beta.

### Events

Run-theory event statistics summarize a standardized index as discrete
energy-drought events: runs beyond a threshold — below it for SREPI, above it
for SRLI or demand — each with an onset, duration, and magnitude (the
cumulative deviation from the threshold).
{func}`climate_indices.energy.energy_drought_events` returns these event
metrics and period summaries ([#1173][energy-1173]). It uses the generic
`runs.summarize_runs()` utility, so the same function accepts SPI and SPEI.
Select direction and threshold explicitly. Direction is never inferred from
an index name or a threshold's sign.

Use `max_gap` to pool events across short observed gaps. Missing and masked
values always separate events. Duration includes pooled gaps, but magnitude
excludes them; intensity is magnitude divided by elapsed duration. The
`min_duration` filter applies after pooling.

Supply one `periods` label per input step for summaries by year, month, or
another declared group. Each complete event belongs to its onset period,
including any duration extending into a later period. The result contains
`events`, `period`, `count`, `total_duration`, `max_magnitude`, `valid_count`,
and `frequency`. Frequency is event count per non-missing input step, not an
annual rate. It is NaN when a period has no valid observations. Without labels,
one summary covers the full input. See {doc}`algorithm-reference` for the full
run and summary contract.

```{testcode}
from climate_indices import energy

result = energy.energy_drought_events(
    [-2, 0, -3], threshold=-1.0, direction="below", max_gap=1,
    periods=[2020, 2020, 2021],
)
assert result.events.duration.tolist() == [3]
assert result.events.intensity.tolist() == [1.0]
assert result.count.tolist() == [1, 0]
assert result.max_magnitude.tolist() == [3.0, 0.0]
```

Allen & Otero (2023) define energy droughts through run theory. These metrics
use the library's strict threshold and threshold-relative magnitude conventions,
not SEI's inclusive comparison and sum of absolute index values. Hand-calculated
tests check the event definitions. The saved SEI SREPI/SRLI series and computed
SPI output check input compatibility, not external validation of event statistics.
Events describe unfavourable weather for the energy system, not recorded supply
shortfalls. This utility accepts standardized arrays, including outputs from
SREPI, the standardized demand index, and SRLI.

## Energy indices

The NumPy degree-day, wind-profile, wind and PV capacity-factor, and event-statistics
functions in `climate_indices.energy` are available, as are SREPI, the standardized
demand index, and SRLI with its residual-load helper.
The NumPy and DataArray APIs are stable from 3.1.0.
[ADR-0019](adr/0019-energy-module-api-and-aggregation.md) fixes the package and scope boundary.
The internal energy subsystem design note records function names, units, and argument contracts. The [energy-family epic #1157][energy-epic] tracks the implementation
order.

- **Heating, cooling, and energy degree days (HDD, CDD, EDD)** — *NumPy API
  available ([#1164][energy-1164]); DataArray input is stable from 3.1.0
  ([#1166][energy-1166])*. Mean-based degree days from daily mean 2 m
  temperature against a base temperature, in °C day; EDD is HDD plus CDD with
  separate heating and cooling bases. CPC's single-base convention ships;
  C3S threshold/reference pairs need a separate contract and are deferred. A caller
  holding Tmin and Tmax computes the daily mean first.
- **Hub-height wind speed** — available as
  {func}`climate_indices.energy.hub_height_wind_speed`
  ([#1165][energy-1165]). Wind extrapolated
  from a finite positive reference height (commonly 10 m or 100 m) to a turbine hub height by a
  power-law or neutral-stability logarithmic profile, in m s⁻¹.
  {func}`climate_indices.energy.wind_shear_exponent` derives per-sample
  shear from two levels: calm pairs yield NaN; positive inverted profiles
  retain negative shear. The default power exponent, 1/7, is a generic
  approximation, not a site-calibrated value. DataArray input is stable from 3.1.0
  ([#1166][energy-1166]): coordinates are preserved, and every dimension may
  carry multiple Dask chunks.
- **Hydropower precipitation proxy** — available as the
  {doc}`hydropower-precipitation-proxy` recipe ([#1167][energy-1167]). It uses
  the existing `indices.standardized_index()` to standardize capacity-weighted
  monthly precipitation at Timescales of 1, 3, 6, and 12 months. It is a proxy
  for hydro resource, not a hydropower generation model.
- **SREPI and the standardized demand index** — available as
  {func}`climate_indices.energy.srepi` and
  {func}`climate_indices.energy.standardized_demand_index`
  ([#1171][energy-1171]), on the empirical distribution option
  ([#1168][energy-1168]) by default, with beta ([#1169][energy-1169]) only for a
  bounded [0, 1] series averaged with `aggregation="mean"`, including the
  endpoint-mass handling described above. Capacity factors are the common case,
  and a bounded demand proxy qualifies on the same terms. SREPI averages each
  window by default, so accumulated production needs `aggregation="sum"`;
  the demand index sums by default. `energy.SREPI_DROUGHT_THRESHOLDS` holds
  Allen & Otero's moderate, severe, and extreme cut-offs (−1.28, −1.64, −1.96).
  DataArray input is stable from 3.1.0 ([#1174][energy-1174]): temporal parameters are
  inferred from the time coordinate, the `srepi` or `standardized_demand_index`
  registry metadata is stamped, and each grid cell is fitted independently to
  match its NumPy series exactly.
- **SRLI and the residual-load helper** — available as
  {func}`climate_indices.energy.srli` and
  {func}`climate_indices.energy.residual_load` ([#1172][energy-1172]).
  `residual_load()` forms demand minus capacity-factor-weighted production from
  installed capacities the caller supplies, in units that match the demand
  series, and leaves the result signed. `srli()` standardizes it on the
  empirical distribution ([#1168][energy-1168]) and never clips a negative.
  High values mark stress. On the default `output_scale="normal"`, Allen &
  Otero (2023) class drought at 1.28, 1.64, and 1.96; the `probability` and
  `bounded` scales need cut-offs of their own. DataArray input is stable from 3.1.0
  ([#1174][energy-1174]) and keeps the empirical distribution: a parametric fit
  does not suit a signed series.
- **Energy-drought event statistics** — available as
  {func}`climate_indices.energy.energy_drought_events` ([#1173][energy-1173]).
  The NumPy utility reports events and period summaries. A 1-D DataArray returns
  the same structure, and a gridded DataArray returns a Dataset of period
  summaries plus one run structure per cell ([#1174][energy-1174]).
- **Wind capacity factor** — available as
  {func}`climate_indices.energy.wind_capacity_factor` ([#1175][energy-1175]).
  It converts hub-height wind through an explicitly selected generic cubic
  curve or a caller-supplied table. It describes the climate-driven resource,
  not actual generation. The NumPy contract and comparison limits appear below.
- **PV capacity factor, temperature-derated** — available as
  {func}`climate_indices.energy.pv_capacity_factor` ([#1176][energy-1176]).
  The fixed Jerez et al. (2015) reference configuration estimates cell temperature
  from surface radiation, 2 m air temperature, and surface (10 m) wind.
  It describes climate-driven solar supply, not actual generation or a plant model.

The degree-day, wind-profile, and wind and PV capacity-factor functions need no new dependencies.
These implementations deliberately overlap `xclim`'s degree-day indicators and
`wind_profile`, `pvlib`, and `windpowerlib`. Those libraries remain documented
alternatives and validation references, never dependencies
([ADR-0019](adr/0019-energy-module-api-and-aggregation.md) decision 7).

### Wind capacity-factor contract

Select the generic curve explicitly with `power_curve="generic"`. Its default
cut-in, rated, and cut-out speeds are 3.5, 13, and 25 m s⁻¹. These defaults and
the cubic formula follow
[`xclim.indices.wind_power_potential`](https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.wind_power_potential),
which cites Tobin et al. (2018). They do not represent a specific turbine.

The factor is zero below cut-in, `(v**3 - cut_in**3)/(rated**3 - cut_in**3)`
from cut-in to rated, and one from rated to cut-out. At cut-out and above,
it is zero. Override the thresholds with `cut_in`, `rated`, and `cut_out`.
They must satisfy `0 <= cut_in < rated < cut_out`.

For a tabulated curve, supply `(speeds, capacity_factors)` as `power_curve`.
Both arrays must be finite, one-dimensional, and equally sized, with at least two knots.
Speeds must be non-negative and strictly increasing. Factors must lie in [0, 1].
Interpolation is linear, includes the endpoint values, and returns zero outside the table.
The table controls shutdown; the generic threshold keywords do not apply to it.
Normalize a manufacturer's power values by rated power before supplying the table.

Optional `air_density` in kg m⁻³ changes the evaluation speed to
`v * (air_density/1.225)**(1/3)` for either curve, including its cut-in and cut-out decisions.
Use this option only for curves referenced to 1.225 kg m⁻³.
Do not apply it to an already density-corrected curve.

Speeds and densities broadcast. Negative or non-finite speeds, non-positive or non-finite
densities, and masked observations yield NaN. The output uses float64 and retains the broadcast shape.
This transform is NumPy-only; it does not dispatch DataArrays or preserve their metadata.
Apply it before time averaging. A curve of mean wind does not equal the mean capacity factor.

```{doctest}
>>> from climate_indices import energy
>>> hub = energy.hub_height_wind_speed([0, 5, 13, 25], 100, 100)
>>> energy.wind_capacity_factor(hub, "generic").round(3)
array([0.   , 0.038, 1.   , 0.   ])
>>> energy.wind_capacity_factor([0, 5, 10, 25], ([0, 10, 25], [0, 1, 0]))
array([0. , 0.5, 1. , 0. ])
```

**Evidence and tolerances.** `tests/test_energy_capacity_factor.py` compares the
cubic calculation against exact rational arithmetic at `rtol=0`, `atol=1e-15`.
That tolerance permits float64 arithmetic error, not model disagreement.
Region boundaries, table knots, and hand-calculated linear examples use exact checks.
These are specification checks, not external numerical validation.

For a future input-matched xclim comparison, the oracle survey recommends
`rtol=0`, `atol=1e-12`, with exact region membership.
C3S uses turbine-specific curves, hub heights, and processed winds.
The survey found no usable matched numerical curve, so no C3S comparison runs here.
No universal model-to-model tolerance is justified. Compare matched-period bias,
MAE/RMSE, and correlation, and record curve, density, height, input, and averaging differences.
Do not treat a capacity factor as observed generation.

### PV reference configuration

For mean irradiance `G` in W m⁻², air temperature `Ta` in °C, and wind speed
`v` in m s⁻¹, Jerez et al. (2015), Methods equations 1–3, specify:

```text
Tcell = 4.3 + 0.943*Ta + 0.028*G - 1.528*v
PR = 1 - 0.005*(Tcell - 25)
PVpot = PR * G/1000
```

The fixed coefficients represent monocrystalline silicon. The reference
cell temperature is 25 °C; air temperature alone does not set that condition.
The function clips the result to [0, 1]. This is project policy: Jerez's
unclipped potential can exceed one. The model excludes panel tilt,
irradiance transposition, optical losses, and inverter losses. Use `pvlib`
for configuration-specific modelling.

The NumPy function accepts broadcast-compatible arrays and returns float64.
Negative radiation or wind, non-finite or masked inputs, and overflowing
results give NaN. Valid zero radiation gives zero. It has no DataArray adapter.

```{testcode}
import numpy as np
from climate_indices import energy

# Convert each ERA5 interval from accumulated energy to mean flux.
interval_seconds = np.array([1800, 3600])
ssrd_j_m2 = np.array([900000, 1800000])
mean_flux = ssrd_j_m2 / interval_seconds
capacity_factor = energy.pv_capacity_factor(mean_flux, 20.0, 2.0)
np.testing.assert_allclose(capacity_factor, [0.47724, 0.47724], rtol=0, atol=1e-12)
```

Align the inputs to the same intervals before calling. Use each accumulation's
actual duration; do not assume 3600 seconds for every product. Do not divide
radiation already expressed in W m⁻². Calculate the nonlinear transform at the
native sampling interval before temporal averaging.

The committed pvlib 0.15.0 fixture checks the matched DC power/temperature
term with supplied cell temperatures at `rtol=0`, `atol=1e-12`. This tolerance
permits float64 roundoff, not configuration differences. Separate hand-calculated
tests check the thermal coefficients. Neither establishes skill against plant
observations. C3S uses a different thermal and AC-generation model, so it is
not an exact numerical reference for this function.

## Preparing inputs

The energy indices are also exposed through the existing command line as
`process_climate_indices --index hdd`, `cdd`, `edd`, `wind_profile`, `srepi`,
`demand_index`, and `srli`, with the same CF metadata and Dask behaviour as the
Python adapters (see {doc}`workflow-examples`). They are not computed by the
`--index all` selection. The capacity factors and the energy-drought event
statistics are not exposed: `energy.wind_capacity_factor()` needs a power curve
and `energy.pv_capacity_factor()` a radiation-temperature-wind triple, and
`energy.energy_drought_events()` reports a per-cell run structure rather than one
gridded index.

The energy interfaces take point series or gridded fields in the units,
location, and spatial support the caller wants to analyse.

`indices.standardized_index()` is the NumPy core: its input block and full-size
working arrays must fit in memory. The adapted
`climate_indices.standardized_index()`, `energy.srepi()`, the standardized demand
index, and `energy.srli()` accept a DataArray and run lazily under Dask, but each
grid cell is standardized in its own call so that every cell matches the NumPy
result exactly, and the time axis must be a single chunk. For a large grid,
aggregate spatially first or process smaller spatial blocks with the full time
series in each block.

- **Units.** Temperature is °C, the library's internal convention; K and °F
  are normalized at the boundary, and because a degree day is a temperature
  difference, K and °C give the same degree days. Wind speed is m s⁻¹ and
  radiation is a *mean* flux in W m⁻². ERA5 `ssrd` is an *accumulated*
  energy in J m⁻², so it must become a mean flux before a PV capacity factor;
  callers divide by each interval's duration in seconds before calling
  `energy.pv_capacity_factor()`.
- **Bias adjustment** of reanalysis or model wind speed is a user-side step.
  Apply a dedicated tool before computing hub-height wind or capacity factors.
- **Spatial aggregation** to countries, bidding zones, or other
  administrative units is a user-side step. Use a tool such as
  [`xagg`][xagg] or [`regionmask`][regionmask] to build the regional series
  first, then standardize it. Weighting by installed capacity, as the
  hydropower proxy does, is part of this step.

### Data sources

The Copernicus Climate Change Service publishes ERA5-derived
[global climate and energy indicators from 1950 to present][c3s-energy] — wind
and PV capacity factors, hydropower proxies, electricity demand, and heating,
cooling, and energy degree days, gridded at 0.25° and aggregated to national
and sub-national units ([DOI 10.24381/3bb607bd][c3s-doi]). Its capacity
factors, degree days, and demand series are ready inputs to the standardized
indices.
ERA5 itself, or CMIP model output, supplies the raw temperature, wind,
radiation, and precipitation for the physical transforms. The
[ERA5 energy demonstration notebook][energy-demo] computes wind-resource and
degree-day proxies, SREPI, standardized demand, an explicitly normalized SRLI
proxy, events, and sampled-cell maps for the January 2017 cold spell. It runs
without credentials on a committed twelve-cell subset during the documentation
doctest build. Its wind-only proxy and illustrative demand/capacity assumptions
are not actual generation or load, or a reproduction of C3S energy indicators.

## What these indices cannot tell you

- **The climate-driven component is not generation or load.** Installed
  capacity, technology, curtailment, markets, and behaviour determine actual
  output and demand. Use climate-derived proxies to interpret the scores as
  weather-only diagnostics.
- **Calibrated energy-system models are out of scope.** Electricity-demand
  models calibrated on measured load, hydropower generation or reservoir-inflow
  models calibrated on observed generation, and grid, dispatch, market, or
  capacity-expansion modelling are outside the family
  ([energy-family epic #1157][energy-epic]). So is plant-level PV or wind
  engineering beyond the generic reference configurations; use `pvlib` or
  `windpowerlib` for that.
- **Validation is limited.** Wind profiles match pinned independent
  implementations (`xclim` power law and `windpowerlib` log law) at
  `atol=1e-12` m s⁻¹ on synthetic inputs. Shear recovery is algebraic
  consistency only. Neither establishes skill against observed hub-height
  wind. The empirical standardization matches R `SEI` 0.2.0 on the attributed
  Bloomfield and Brayshaw (2021) capacity-factor series. [#1180][energy-1180]
  holds the family's full
  [validation inventory][validation], which records the NOAA CPC degree-day
  example, the `SEI`/`xclim`/`windpowerlib`/`pvlib` comparisons, and the
  specification-only items. No C3S product comparison runs, because no matched
  C3S degree-day or capacity-factor input is adopted.

## Sources

- [Allen & Otero (2023), *Standardised indices to monitor energy droughts*][allen-otero-2023]
- [Copernicus C3S, *Global climate and energy indicators from 1950 to present derived from reanalysis*][c3s-doi]
- [Jerez et al. (2015), *The impact of climate change on photovoltaic power generation in Europe*](https://doi.org/10.1038/ncomms10014)
- [pvlib `pvwatts_dc`](https://pvlib-python.readthedocs.io/en/v0.15.0/reference/generated/pvlib.pvsystem.pvwatts_dc.html)
- [`SEI` R package][sei]

[allen-otero-2023]: https://doi.org/10.1016/j.renene.2023.119206
[sei]: https://github.com/noeliaof/SEI
[c3s-energy]: https://cds.climate.copernicus.eu/datasets/sis-energy-global-reanalysis?tab=overview
[c3s-doi]: https://doi.org/10.24381/3bb607bd
[xagg]: https://github.com/ks905383/xagg
[regionmask]: https://regionmask.readthedocs.io/
[validation]: https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md
[energy-epic]: https://github.com/monocongo/climate_indices/issues/1157
[energy-1164]: https://github.com/monocongo/climate_indices/issues/1164
[energy-1165]: https://github.com/monocongo/climate_indices/issues/1165
[energy-1166]: https://github.com/monocongo/climate_indices/issues/1166
[energy-1167]: https://github.com/monocongo/climate_indices/issues/1167
[energy-1168]: https://github.com/monocongo/climate_indices/issues/1168
[energy-1169]: https://github.com/monocongo/climate_indices/issues/1169
[energy-1171]: https://github.com/monocongo/climate_indices/issues/1171
[energy-1174]: https://github.com/monocongo/climate_indices/issues/1174
[energy-1172]: https://github.com/monocongo/climate_indices/issues/1172
[energy-1173]: https://github.com/monocongo/climate_indices/issues/1173
[energy-1175]: https://github.com/monocongo/climate_indices/issues/1175
[energy-1176]: https://github.com/monocongo/climate_indices/issues/1176
[energy-1177]: https://github.com/monocongo/climate_indices/issues/1177
[energy-demo]: https://github.com/monocongo/climate_indices/blob/main/notebooks/energy_demo.ipynb
[energy-1180]: https://github.com/monocongo/climate_indices/issues/1180

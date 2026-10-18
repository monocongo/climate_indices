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
generation or load estimates. The planned physical transforms use standard
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
Resource transforms              hub-height wind (available); wind & PV capacity factor (planned)
Hydro proxy                      capacity-weighted n-month precip           (available recipe)
              ↓
Standardized energy indices      SREPI, standardized demand, SRLI           (planned)
              ↓
Energy-drought events            run-theory onset / duration / magnitude    (planned)
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
Python implementation of those indices is planned as the family's main
contribution; the physical transforms will make it usable end to end from raw
reanalysis.

As planned, three indices standardize climate-derived energy proxies that the
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
above roughly 100 observations, with beta available for bounded [0, 1]
capacity factors averaged with `aggregation="mean"` when there are too few.
The beta option is not for demand proxies or accumulated production.
[#1169][energy-1169] requires separate probability masses at exact 0 and 1;
the continuous beta fit applies only between those endpoints. This endpoint
handling remains planned, not implemented.
SRLI uses the empirical distribution only. That is an API restriction rather
than a mathematical necessity: a rank-based standardization has no support
for a signed series to violate, the parametric families' support-limit and
zero-mass handling is not designed for one, and clipping negatives to zero
would score a different series.

### Timescale and aggregation

The accumulation window ("Timescale" in prose; `scale` in code) works as it
does for SPI and SPEI, with one addition. The family is daily first: a weekly
index is daily input with a Timescale of 7, and a 30-day index is daily input
with a Timescale of 30; calendar months use monthly input. Sub-daily (hourly)
input is planned separately ([#1177][energy-1177]).

Daily NumPy inputs to `indices.standardized_index()` must start on January 1
and use the internal 366-slot year layout. The planned energy NumPy wrappers
inherit this layout. Convert Gregorian arrays with
{func}`climate_indices.utils.transform_to_366day` before standardizing them;
see {doc}`data_requirements` for the conversion requirements. Do not pass an
ordinary Gregorian array directly: seasonal fitting would use the wrong day
slots. Existing xarray index adapters handle Gregorian time coordinates
separately; the planned energy xarray adapters are not available today.

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
cumulative deviation from the threshold). The planned energy function builds
on `runs.identify_runs()`, the run-theory primitive available today, which
takes an explicit `"below"` or `"above"` direction ([#1173][energy-1173]).
Allen & Otero (2023) define energy droughts this way. Events inherit the
caveat at the top of this page: they are runs of unfavourable weather for the
energy system, not recorded supply shortfalls.

## Energy indices

The NumPy degree-day and wind-profile functions in `climate_indices.energy`
are available; the other energy indices below remain planned. The package, its
NumPy and beta xarray tiers, and the scope boundary are fixed by
[ADR-0019](adr/0019-energy-module-api-and-aggregation.md), with function names,
units, and argument contracts recorded in the internal energy subsystem design
note. The [energy-family epic #1157][energy-epic] tracks the implementation
order.

- **Heating, cooling, and energy degree days (HDD, CDD, EDD)** — *NumPy API
  available ([#1164][energy-1164])*. Mean-based degree days from daily mean 2 m
  temperature against a base temperature, in °C day; EDD is HDD plus CDD with
  separate heating and cooling bases. CPC's single-base convention ships;
  C3S threshold/reference pairs need a separate contract and are deferred. A caller
  holding Tmin and Tmax computes the daily mean first.
- **Hub-height wind speed** — available as
  {func}`climate_indices.energy.hub_height_wind_speed`. Wind extrapolated
  from a finite positive reference height (commonly 10 m or 100 m) to a turbine hub height by a
  power-law or neutral-stability logarithmic profile, in m s⁻¹.
  {func}`climate_indices.energy.wind_shear_exponent` derives per-sample
  shear from two levels: calm pairs yield NaN; positive inverted profiles
  retain negative shear. The default power exponent, 1/7, is a generic
  approximation, not a site-calibrated value. xarray/Dask dispatch is not
  available yet.
- **Hydropower precipitation proxy** — available as the
  {doc}`hydropower-precipitation-proxy` recipe ([#1167][energy-1167]). It uses
  the existing `indices.standardized_index()` to standardize capacity-weighted
  monthly precipitation at Timescales of 1, 3, 6, and 12 months. It is a proxy
  for hydro resource, not a hydropower generation model.
- **SREPI and the standardized demand index** — *planned
  ([#1171][energy-1171])*, on the empirical distribution option
  ([#1168][energy-1168]), with beta ([#1169][energy-1169]) only for SREPI
  capacity factors in [0, 1] averaged with `aggregation="mean"`, including the
  planned endpoint-mass handling described above, not for demand proxies.
- **SRLI** — *planned ([#1172][energy-1172])*.
- **Energy-drought event statistics** — *planned ([#1173][energy-1173])*.
- **Wind capacity factor** — *planned ([#1175][energy-1175])*. Hub-height
  wind passed through a required power curve: a generic parametric reference
  curve (cut-in, rated, and cut-out speeds) or a tabulated curve the caller
  supplies, not plant engineering.
- **PV capacity factor, temperature-derated** — *planned
  ([#1176][energy-1176])*. Capacity factor from surface solar radiation,
  reduced as cell temperature — estimated from air temperature, irradiance,
  and wind — rises, after the reference configuration of Jerez et al. (2015).

The degree-day and wind-profile functions are implemented without new
dependencies. The planned capacity factors will also need no new dependencies.
These implementations deliberately overlap `xclim`'s degree-day indicators and
`wind_profile`, `pvlib`, and `windpowerlib`. Those libraries remain documented
alternatives and validation references, never dependencies
([ADR-0019](adr/0019-energy-module-api-and-aggregation.md) decision 7).

## Preparing inputs

The planned energy interfaces will take point series or gridded fields in the
units, location, and spatial support the caller wants to analyse.

The existing `indices.standardized_index()` is NumPy-only, with no xarray or
Dask dispatch. Its input block and full-size working arrays must fit in memory.
For large grids, aggregate spatially first or process smaller spatial blocks
with the full time series in each block. The planned energy xarray interfaces
do not provide a lazy execution path today.

- **Units.** Temperature is °C, the library's internal convention; K and °F
  are normalized at the boundary, and because a degree day is a temperature
  difference, K and °C give the same degree days. Wind speed is m s⁻¹ and
  radiation is a *mean* flux in W m⁻². ERA5 `ssrd` is an *accumulated*
  energy in J m⁻², so it must become a mean flux before a PV capacity factor;
  whether that conversion is a caller's step or an argument is decided by
  [#1176][energy-1176].
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
factors, degree days, and demand series are ready inputs to the planned
standardized indices.
ERA5 itself, or CMIP model output, supplies the raw temperature, wind,
radiation, and precipitation for the physical transforms. An end-to-end
gridded demonstration on these data is planned ([#1179][energy-1179]).

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
  wind. [#1180][energy-1180] owns the family's full
  [validation inventory][validation], including future `SEI` and C3S comparisons
  where those serve as oracles.

## Sources

- [Allen & Otero (2023), *Standardised indices to monitor energy droughts*][allen-otero-2023]
- [Copernicus C3S, *Global climate and energy indicators from 1950 to present derived from reanalysis*][c3s-doi]
- Jerez et al. (2015), the temperature-derated PV reference configuration specified by [#1176][energy-1176]
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
[energy-1167]: https://github.com/monocongo/climate_indices/issues/1167
[energy-1168]: https://github.com/monocongo/climate_indices/issues/1168
[energy-1169]: https://github.com/monocongo/climate_indices/issues/1169
[energy-1171]: https://github.com/monocongo/climate_indices/issues/1171
[energy-1172]: https://github.com/monocongo/climate_indices/issues/1172
[energy-1173]: https://github.com/monocongo/climate_indices/issues/1173
[energy-1175]: https://github.com/monocongo/climate_indices/issues/1175
[energy-1176]: https://github.com/monocongo/climate_indices/issues/1176
[energy-1177]: https://github.com/monocongo/climate_indices/issues/1177
[energy-1179]: https://github.com/monocongo/climate_indices/issues/1179
[energy-1180]: https://github.com/monocongo/climate_indices/issues/1180

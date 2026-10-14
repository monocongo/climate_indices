# Energy subsystem

Internal working note for the energy-meteorology family
([#1157](https://github.com/monocongo/climate_indices/issues/1157)). It plans
unimplemented indices, so it stays in the repository rather than the published site —
the arrangement [ADR-0005](../adr/0005-fire-module-api.md) chose for fire and
[ADR-0013](../adr/0013-flood-module-api-and-naming.md) for flood. The decisions that
hold regardless of scope live in
[ADR-0019](../adr/0019-energy-module-api-and-aggregation.md); this note holds the
per-index naming, units, and argument contracts, plus what is deferred.

## Decision

The public facade is `climate_indices.energy`, imported as
`from climate_indices import energy`, with underscore-prefixed implementation modules
mirroring `fire/` and `flood/`. No energy name is re-exported from the package root.
The API has two tiers, as flood does:

- **NumPy reference functions**, the stable tier, typed and documented.
- **xarray adapters**, beta under [ADR-0012](../adr/0012-xarray-api-stays-beta-through-3.0.0.md),
  added by [#1166](https://github.com/monocongo/climate_indices/issues/1166) and
  [#1174](https://github.com/monocongo/climate_indices/issues/1174).

## Scope and boundary

These indices describe the **climate-driven component** of energy supply and demand,
not actual generation or load, which also depend on installed capacity, technology,
curtailment, markets, and behaviour. Every docstring and doc page says so. The scope
decision itself is [#1159](https://github.com/monocongo/climate_indices/issues/1159),
which reconciles the overlap with `xclim`, `pvlib`, and `windpowerlib`; its outcome can
defer a physical transform but cannot change a name recorded here.

Out of scope by the epic's definition: electricity-demand models calibrated on measured
load data, hydropower generation models calibrated on observed generation, bias
adjustment of wind speed, spatial aggregation to administrative units, and grid,
dispatch, market, or capacity-expansion modelling.

## Standardized indices and the seam

`energy.srepi()`, `energy.standardized_demand_index()`, and `energy.srli()` wrap the
standardization seam of [ADR-0019](../adr/0019-energy-module-api-and-aggregation.md).
Each takes the `indices.standardized_index()` argument shape and fixes its own
`aggregation` and distribution defaults, so a caller supplies a series and a calibration
period and not a statistics choice:

| Function | Series | Aggregation | Distribution |
| --- | --- | --- | --- |
| `energy.srepi()` | renewable production or a capacity factor | `mean` for a capacity factor, `sum` for accumulated production | empirical ([#1168](https://github.com/monocongo/climate_indices/issues/1168)), beta for a bounded series with too few observations ([#1169](https://github.com/monocongo/climate_indices/issues/1169)) |
| `energy.standardized_demand_index()` | energy demand or EDD | `sum` over the accumulation window | empirical, beta for a bounded series |
| `energy.srli()` | demand minus renewable production, signed | `sum` or `mean` | empirical only |

The empirical path is the default because Allen & Otero (2023) recommend it above
roughly 100 observations, and because it is the only family valid for a signed series.
The plotting position, tie handling, and minimum-observation rule are fixed by
[#1168](https://github.com/monocongo/climate_indices/issues/1168) against the convention
the oracle survey confirms.

`energy.energy_drought_events()` produces run-theory onset, duration, magnitude, and
peak statistics on any standardized energy index, built on `runs.identify_runs()`.
The doubled word matches `flood.flood_events()` and keeps a meteorological drought event
distinct from an energy one.

## Physical transforms

Provisional on [#1159](https://github.com/monocongo/climate_indices/issues/1159).

| Function | Inputs | Units | Notes |
| --- | --- | --- | --- |
| `energy.heating_degree_days()` | `temperature` | °C in, °C day out | mean-based; base temperature is a keyword |
| `energy.cooling_degree_days()` | `temperature` | °C in, °C day out | mean-based; base temperature is a keyword |
| `energy.energy_degree_days()` | `temperature` | °C in, °C day out | HDD + CDD, separate heating and cooling base temperatures, per the C3S definition |
| `energy.hub_height_wind_speed()` | `wind_speed`, `reference_height`, `hub_height` | m s⁻¹ | `method="power"` (default) or `"log"`, the latter taking `roughness_length` |
| `energy.wind_capacity_factor()` | `hub_wind_speed`, `power_curve` | dimensionless [0, 1] | generic documented reference curve; not plant engineering |
| `energy.pv_capacity_factor()` | `ssrd`, `temperature`, wind | dimensionless [0, 1] | temperature-derated; mean flux W m⁻² in |

Degree-day conventions:

- **Mean-based only.** A caller holding Tmin/Tmax computes the mean first; no derived-mean
  or min/max keyword.
- **Defaults are the oracle's.** The C3S and Eurostat schemes (15/18, 24/21) are not
  separate APIs — they are non-default base temperatures, so the argument set already
  covers them. The default values are confirmed by the oracle survey
  ([#1160](https://github.com/monocongo/climate_indices/issues/1160)) and pinned by the
  fixtures ([#1163](https://github.com/monocongo/climate_indices/issues/1163)); this note
  deliberately does not invent them.
- **No degree day without a fixture.** [#1164](https://github.com/monocongo/climate_indices/issues/1164)
  proceeds only if #1159 keeps the transforms in scope.

Units: temperature is °C, the library's internal convention, and `_units.py` normalizes
K and °F at its boundary. A degree day is a temperature difference, so K and °C are
numerically interchangeable under it. The exact UDUNITS strings belong to the CF
registry entries ([#1162](https://github.com/monocongo/climate_indices/issues/1162)).

## Deferred

- **Sub-daily (hourly) periodicity.** [#1177](https://github.com/monocongo/climate_indices/issues/1177).
  Forward-compatible as an additive `Periodicity` member; no signature changes.
- **`clip_negatives` on the public seam, and the parametric rejection of a signed series.**
  Land with [#1168](https://github.com/monocongo/climate_indices/issues/1168) and
  [#1172](https://github.com/monocongo/climate_indices/issues/1172), the first consumers
  that make a signed series meaningful. Exposing it earlier would let a signed series
  reach a gamma fit.
- **Accumulated radiation.** ERA5 `ssrd` is accumulated J m⁻² and has to be converted to
  a mean flux before `energy.pv_capacity_factor()`. [#1176](https://github.com/monocongo/climate_indices/issues/1176)
  decides whether the conversion is a caller's step or an argument.
- **CLI integration.** [#1178](https://github.com/monocongo/climate_indices/issues/1178);
  the CLI consumes the package rather than being a component of it, as for fire and flood.

## References

- Allen, S., and Otero, N. (2023). Standardised indices to monitor energy droughts.
  *Renewable Energy*, 217, 119206. https://doi.org/10.1016/j.renene.2023.119206
- Copernicus C3S: Global climate and energy indicators from 1950 to present derived from
  reanalysis. https://doi.org/10.24381/3bb607bd
- `docs/design/flood-subsystem.md`, `docs/design/fire-subsystem.md`
- `docs/research/energy-oracle-survey.md`
  ([#1160](https://github.com/monocongo/climate_indices/issues/1160))

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
decision is [#1159](https://github.com/monocongo/climate_indices/issues/1159), recorded in
[ADR-0019](../adr/0019-energy-module-api-and-aggregation.md) decision 7.
The physical transforms are implemented here without new dependencies.
The degree-day and wind-profile formulas deliberately overlap with `xclim`; `pvlib`
provides PV prior art. The standardized demand index takes degree days as a direct
input. No name recorded here changes with that outcome.

A separate repository becomes justified when two or more of these are true:

1. An implementation requires a calibrated demand or generation model
2. An implementation requires data beyond standard meteorological and reanalysis fields
3. The energy code exceeds roughly 30 percent of the package's source volume
4. Energy-specific dependencies would be forced on all `climate_indices` users
5. The release cadence needs to diverge

Criterion 2 holds because `energy.wind_capacity_factor()` requires a `power_curve`,
not a meteorological or reanalysis field. The other four criteria do not hold, so
no sibling repository is justified
([#1159](https://github.com/monocongo/climate_indices/issues/1159)). Until two do, an
out-of-scope energy proposal is a candidate for that split, not a rejection on merit.
[CONTRIBUTING.md](../../CONTRIBUTING.md#energy-scope) records the triage response.

Out of scope by the epic's definition: electricity-demand models calibrated on measured
load data, hydropower generation models calibrated on observed generation, bias
adjustment of wind speed, spatial aggregation to administrative units, and grid,
dispatch, market, or capacity-expansion modelling.

## Standardized indices and the seam

`energy.srepi()`, `energy.standardized_demand_index()`, and `energy.srli()` wrap the
standardization seam of [ADR-0019](../adr/0019-energy-module-api-and-aggregation.md).
Each takes the `indices.standardized_index()` argument shape with its own
`aggregation` and distribution defaults, so a caller supplies a series and a calibration
period and not a statistics choice, except where the input representation requires an
explicit aggregation:

| Function | Series | Aggregation | Distribution |
| --- | --- | --- | --- |
| `energy.srepi()` | renewable production or a capacity factor | keyword-only `aggregation="mean"` (default) for a capacity factor, `aggregation="sum"` for accumulated production | empirical ([#1168](https://github.com/monocongo/climate_indices/issues/1168)), beta for a bounded series with too few observations ([#1169](https://github.com/monocongo/climate_indices/issues/1169)) |
| `energy.standardized_demand_index()` | energy demand or EDD | `sum` over the accumulation window | empirical, beta for a bounded series |
| `energy.srli()` | demand minus renewable production, signed | `sum` or `mean` | empirical only |

SREPI callers supplying accumulated production must select `aggregation="sum"`;
input meaning is never inferred from magnitudes. Beta is available only for bounded
[0, 1] capacity factors with `aggregation="mean"`, not accumulated production.

The empirical path is the default because Allen & Otero (2023) recommend it above
roughly 100 observations, and because it is the family the energy API admits a signed
series on: a rank-based standardization has no support to violate and no point mass to
place. That is an API restriction, not a mathematical necessity — the parametric
transforms will transform a signed series when given fitting parameters whose support
covers it, but their support-limit and zero-mass handling is not designed for one, and
clipping the negatives instead would score a different series.
The plotting position, tie handling, and minimum-observation rule are fixed by
[#1168](https://github.com/monocongo/climate_indices/issues/1168) against the convention
the oracle survey confirms.

`energy.energy_drought_events()` produces run-theory onset, duration, magnitude, and
peak statistics on any standardized energy index, built on `runs.identify_runs()`.
The doubled word matches `flood.flood_events()` and keeps a meteorological drought event
distinct from an energy one.

## Physical transforms

In scope by [ADR-0019](../adr/0019-energy-module-api-and-aggregation.md) decision 7
([#1159](https://github.com/monocongo/climate_indices/issues/1159)): all four implement
in-house, dependency-free.

| Function | Inputs | Units | Notes |
| --- | --- | --- | --- |
| `energy.heating_degree_days()` | `temperature` | °C in, °C day out | mean-based; base temperature is a keyword |
| `energy.cooling_degree_days()` | `temperature` | °C in, °C day out | mean-based; base temperature is a keyword |
| `energy.energy_degree_days()` | `temperature` | °C in, °C day out | HDD + CDD, separate heating and cooling base temperatures, per the C3S definition |
| `energy.hub_height_wind_speed()` | `wind_speed`, `reference_height`, `hub_height` | m s⁻¹ | `method="power"` (default) or `"log"`, the latter taking `roughness_length` |
| `energy.wind_capacity_factor()` | `hub_wind_speed`, `power_curve` | dimensionless [0, 1] | generic documented reference curve; not plant engineering |
| `energy.pv_capacity_factor()` | `ssrd`, `temperature`, wind | dimensionless [0, 1] | temperature-derated; mean flux W m⁻² in |

Degree-day conventions:

- **Mean-based is the surface that ships first.** A caller holding Tmin/Tmax computes the
  mean first; there is no derived-mean keyword.
- **The convention itself is still the oracle's to reopen,** not just the default values.
  A min/max-based convention would be a *separate function taking both*, not a keyword on
  this one, because a mean-temperature argument cannot recover it: a day with Tmin 10 °C
  and Tmax 25 °C contributes 5 heating degree days from Tmin against a 15 °C base, and
  none at all from its mean of 17.5 °C. That function is additive and leaves the
  mean-based one valid, which is what makes deferring it not a one-way door.
- **Defaults are the oracle's.** The C3S and Eurostat schemes (15/18, 24/21) are not
  separate APIs — they are non-default base temperatures, so the argument set already
  covers them. The default values are confirmed by the oracle survey
  ([#1160](https://github.com/monocongo/climate_indices/issues/1160)) and pinned by the
  fixtures ([#1163](https://github.com/monocongo/climate_indices/issues/1163)); this note
  deliberately does not invent them.
- **No degree day without a fixture.** [#1164](https://github.com/monocongo/climate_indices/issues/1164)
  has the transforms in scope (decision 7) and still lands only against the
  [#1163](https://github.com/monocongo/climate_indices/issues/1163) fixtures.

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

# Energy subsystem

Internal working note for the energy-meteorology family
([#1157](https://github.com/monocongo/climate_indices/issues/1157)). It plans
unimplemented indices, so it stays in the repository rather than the published site —
the arrangement [ADR-0005](../adr/0005-fire-module-api.md) chose for fire and
[ADR-0013](../adr/0013-flood-module-api-and-naming.md) for flood. The decisions that
hold regardless of scope live in
[ADR-0019](../adr/0019-energy-module-api-and-aggregation.md); this note holds the
per-index naming, units, and argument contracts, plus what is deferred.
The NumPy wind-profile functions are implemented by #1165, degree days by #1164,
physical-transform adapters by #1166, event statistics by #1173, and the wind
capacity-factor transform by #1175. The wind capacity-factor DataArray adapter,
other transforms, and the standardized energy xarray tier remain planned.

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

`energy.residual_load()` forms the SRLI input: demand minus the sum, over technologies, of
capacity factor times installed capacity. The caller supplies the capacities in units that
match the demand series, and the result stays signed. `energy.srli()` takes an explicit
`distribution` keyword and raises for any member but the empirical one, so the restriction
is an error message and not a missing argument.

The empirical path is the default because Allen & Otero (2023) recommend it above
roughly 100 observations, and because it is the family the energy API admits a signed
series on: a rank-based standardization has no support to violate and no point mass to
place. That is an API restriction, not a mathematical necessity — the parametric
transforms will transform a signed series when given fitting parameters whose support
covers it, but their support-limit and zero-mass handling is not designed for one, and
clipping the negatives instead would score a different series.
[#1168](https://github.com/monocongo/climate_indices/issues/1168) fixed the plotting
position, tie handling, and minimum-observation rule to the SEI convention the oracle
survey confirms: `p = (1 + k) / (n + 2)` with `k` the count of calibration values at or
below the value (ties at their maximum rank), no extrapolation beyond the calibration
range, and at least ten valid calibration values per calendar step, else missing. The
seam does not fall back to a parametric fit below Allen & Otero's roughly 100
observations; that choice belongs to the wrappers.

`energy.energy_drought_events()` is implemented by #1173. It returns generic
`runs.RunStatistics`: a `RunSet` of events and period summaries, built on
`runs.summarize_runs()` and `runs.identify_runs()`. The NumPy utility accepts any
one-dimensional index series, including SPI, SREPI, and SRLI. Its energy xarray
adapter remains deferred to #1174.

The maintainer approved these event conventions for #1173: strict comparisons;
explicit direction; pooling across at most `max_gap` non-missing steps, never
across NaN or masked values; duration includes pooled gaps but magnitude excludes
them; intensity divides magnitude by elapsed duration; minimum-duration filtering
follows pooling. Each complete event belongs to its onset period. Frequency is
event count per non-NaN, unmasked input step in that period, or NaN with no valid
steps. Periods without events have zero count, duration, and maximum magnitude.
The generic utility retains the existing float64 metrics and infinity comparisons.

The doubled word matches `flood.flood_events()` and keeps a meteorological drought
event distinct from an energy one. The function summarizes climate-driven energy
stress, not actual generation, load, or recorded supply shortfalls.

## Physical transforms

In scope by [ADR-0019](../adr/0019-energy-module-api-and-aggregation.md) decision 7
([#1159](https://github.com/monocongo/climate_indices/issues/1159)): all four implement
in-house, dependency-free.

| Function | Inputs | Units | Notes |
| --- | --- | --- | --- |
| `energy.heating_degree_days()` | `temperature` | °C in, °C day out | mean-based; keyword `base_temperature` |
| `energy.cooling_degree_days()` | `temperature` | °C in, °C day out | mean-based; keyword `base_temperature` |
| `energy.energy_degree_days()` | `temperature` | °C in, °C day out | HDD + CDD; keywords `heating_base_temperature`, `cooling_base_temperature` |
| `energy.hub_height_wind_speed()` | `wind_speed`, `reference_height`, `hub_height` | m s⁻¹ | `method="power"` (default), with `alpha=1/7` by default per #1165, or `"log"`, with `roughness_length` |
| `energy.wind_shear_exponent()` | `wind_speed_lower`, `wind_speed_upper` | dimensionless | per-sample two-level shear; keyword-only `lower_height=10`, `upper_height=100` in metres |
| `energy.wind_capacity_factor()` | `hub_wind_speed`, required `power_curve` | dimensionless [0, 1] | NumPy; explicit `"generic"` cubic or `(speeds, capacity_factors)` table; optional `air_density`; not plant engineering |
| `energy.pv_capacity_factor()` | `ssrd`, `temperature`, wind | dimensionless [0, 1] | temperature-derated; mean flux W m⁻² in |

Wind profiles use scalar finite positive heights in metres; speeds and power-law
exponents broadcast as NumPy arrays. The log law requires a supplied positive
roughness length below both heights and assumes neutral stability and zero
displacement height. Shear is NaN for calm, negative, or non-finite wind pairs;
positive inverted profiles retain negative exponents, as confirmed for #1165.
Independent-implementation comparisons cover the profile arithmetic, not observed-wind
prediction; shear recovery is only algebraic consistency.

Wind capacity factors use xclim's generic 3.5/13/25 m s⁻¹ thresholds unless overridden.
The cubic curve assigns zero at cut-in, one at rated, and zero at cut-out.
Tables use linear interpolation, include both endpoints, and return zero outside their range.
Optional density correction evaluates either curve at `v * (air_density/1.225)**(1/3)`.
The [published contract](../energy_applications.md#wind-capacity-factor-contract)
records units, missing values, averaging order, and evidence limits.
C3S has no matched numerical curve; the tests provide specification evidence only.

Degree-day conventions:

- **Mean-based is the surface that ships first.** A caller holding Tmin/Tmax computes the
  mean first; there is no derived-mean keyword.
- **The convention itself is still the oracle's to reopen,** not just the default values.
  A min/max-based convention would be a *separate function taking both*, not a keyword on
  this one, because a mean-temperature argument cannot recover it: a day with Tmin 10 °C
  and Tmax 25 °C contributes 5 heating degree days from Tmin against a 15 °C base, and
  none at all from its mean of 17.5 °C. That function is additive and leaves the
  mean-based one valid, which is what makes deferring it not a one-way door.
- **CPC first, with explicit bases.** The maintainer selected NOAA CPC's single-base
  convention for [#1163](https://github.com/monocongo/climate_indices/issues/1163):
  `HDD = max(base_temperature - T, 0)` and `CDD = max(T - base_temperature, 0)`.
  The published seven-day CDD example uses 65 °F, exactly `(65 - 32) * 5/9` °C.
  The fixture tests always supply bases; no public default is chosen here.
  Heating and EDD examples are separately labelled specification checks, not
  external numerical validation. See `tests/fixture/energy_cpc/README.md`.
- **C3S remains unresolved.** The C3S and Eurostat schemes (15/18, 24/21) are not
  representable as non-default single bases. The oracle survey
  ([#1160](https://github.com/monocongo/climate_indices/issues/1160)) does not confirm
  default values: each scheme pairs an activation threshold with a separate reference
  temperature, which one base per transform cannot reproduce. No default is established
  until [#1163](https://github.com/monocongo/climate_indices/issues/1163) and
  [#1164](https://github.com/monocongo/climate_indices/issues/1164) reconcile that
  contract and the fixtures pin the values; this note deliberately does not invent them.
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

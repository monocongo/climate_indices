# Energy APIs are namespaced in the `energy` package, and the standardize seam gains an aggregation

## Status

Amended by #1061: energy DataArray entry points are Stable from 3.1.0 under
{doc}`0012-xarray-api-stays-beta-through-3.0.0`. Energy transforms stop emitting
`BetaFeatureWarning`. Earlier Beta descriptions below record their status at introduction.

Amended: decision 2 also exposes matching aggregation on `indices.fit_diagnostics()`
so its fitting parameters can be reused by the standardized index.

Amended again by the energy scope decision
([#1159](https://github.com/monocongo/climate_indices/issues/1159)): decision 7 fixes the
family's scope boundary. The physical transforms stay in this package, implemented
in-house with no new dependency, and calibrated models and preprocessing stay out of
scope. No name or signature recorded here changes. The required wind `power_curve`
meets one split criterion, below the two-criterion threshold.

Amended by the wind-profile implementation
([#1165](https://github.com/monocongo/climate_indices/issues/1165)): decision 1's
`climate_indices.energy` package now exists, exposing `hub_height_wind_speed()`
and `wind_shear_exponent()` with decision 6's m s⁻¹/metre wind contract.
Positive inverted profiles retain negative shear; calm-pair shear is NaN.

Amended again by the degree-day implementation
([#1164](https://github.com/monocongo/climate_indices/issues/1164)): the package also
exposes `heating_degree_days()`, `cooling_degree_days()`, and
`energy_degree_days()`, in-house and dependency-free, with decision 6's degree-day
units.

Amended by the physical-transform adapters
([#1166](https://github.com/monocongo/climate_indices/issues/1166)): degree days,
hub-height wind, and the auxiliary shear diagnostic accept DataArrays through
the beta xarray layer. Their elementwise Dask execution accepts multiple chunks
on every dimension; temperature units are inferred while bases remain in °C.
Standardized energy adapters landed with
[#1174](https://github.com/monocongo/climate_indices/issues/1174): `srepi()`,
`standardized_demand_index()`, and `srli()` accept DataArrays, and `aggregation`
reaches the generic xarray `standardized_index()` entry and the xarray
`fit_diagnostics()` report.

Amended by the event-statistics implementation
([#1173](https://github.com/monocongo/climate_indices/issues/1173)):
`energy.energy_drought_events()` now returns generic run metrics and onset-period
summaries through `runs.summarize_runs()`. Pooling, elapsed duration, and frequency
follow the maintainer-approved contract in the energy subsystem design note.
The energy event xarray adapter landed with
[#1174](https://github.com/monocongo/climate_indices/issues/1174): a 1-D DataArray
returns the same `RunStatistics`, and a gridded one returns a Dataset of period
summaries with one run structure per cell.

Amended by the wind capacity-factor implementation
([#1175](https://github.com/monocongo/climate_indices/issues/1175)):
`energy.wind_capacity_factor()` now implements decision 7's reference transform.
The required `power_curve` selects a generic cubic curve or a normalized table.
Optional density correction applies before curve evaluation. This function is NumPy-only;
its specification checks do not validate actual generation or match a C3S turbine model.

Amended by the PV implementation
([#1176](https://github.com/monocongo/climate_indices/issues/1176)):
`energy.pv_capacity_factor()` implements the Jerez reference configuration on
NumPy inputs. It accepts mean radiation in W m⁻², air temperature in °C, and
surface wind in m s⁻¹. Callers convert accumulated SSRD to mean flux using each
interval's duration in seconds. Clipping to [0, 1] is project policy, not a
Jerez result. The matched pvlib comparison checks only the power/temperature
term; the thermal coefficients have specification-level tests.

Amended by the sub-daily periodicity implementation
([#1177](https://github.com/monocongo/climate_indices/issues/1177)): decision 5's
additive member is `Periodicity.hourly`, one year of 8784 values on the same 366-day
calendar, so a non-leap year carries 24 synthesized February 29 hours
(`utils.CalendarPlan.from_year_span(..., steps_per_day=24)`). Standardization
stratifies by hour of day and day of year — 8784 calendar steps per year — which is
the explicit-group contract the oracle survey requires before an SEI comparison is
meaningful. Hourly supports only `Distribution.empirical`: a 30-year record leaves
about 30 calibration values per step, too few for a parametric fit, while the rank
transform needs no fit, so any other distribution raises `InvalidArgumentError`.
Timestamps are read as clock values with no daylight-saving shift, so a caller
supplies a UTC series. The xarray layer infers an hourly cadence, converts to the
8784-slot layout, and restores the Gregorian hour axis. The CLI keeps its
monthly/daily `--periodicity` choices; an hourly option is a follow-up. The
implementation landed with unit tests rather than an independent hourly oracle,
because the survey records that SEI accepts hourly timestamps without stratifying
them and #1160 landed no hourly fixture.

Amended by the hourly CLI option
([#1406](https://github.com/monocongo/climate_indices/issues/1406)): `--periodicity
hourly` reaches `srepi`, `demand_index`, and `srli`, which already run through the
xarray adapters that validate the 00:00 January 1 start and contiguous hours, plan the
8784-slot layout, and restore the Gregorian hour axis. Every other index rejects
`hourly` during argument validation, before any output is written.

Amended by the pooled seasonal-diurnal climatology
([#1407](https://github.com/monocongo/climate_indices/issues/1407)): the empirical
hourly path gains keyword-only `pool_window_days`, default `0`. A zero window keeps
decision 5's shipped hour x day-of-year strata, so every existing hourly result is
unchanged. A positive window widens each step's calibration sample to the same hour on
that many neighbouring days, wrapping the year boundary; a 30-year step grows from
about 30 values to `30 x (2k + 1)`, which refines the tail resolution of a thin hourly
record that the SEI fixture's 100-value recommendation
(`tests/fixture/energy_sei/warnings.txt`) covers only for monthly data. All of
29 February is held out of every pooled sample, because a non-leap year's 24 hours
there are synthesized rather than observed; a 29 February target is still scored, from
its neighbouring days. The holdout follows each scaled value's accumulation window, so
a `scale` above one also holds out the `scale - 1` steps after 29 February, and a scale
long enough to span a full hourly cycle is rejected. A positive window is rejected for
every periodicity but hourly and every distribution but empirical, and it may not
exceed 182 days, so each calendar day enters a sample at most once. An explicit zero is
accepted on any periodicity and
distribution. The keyword reaches `indices.standardized_index()`, the generic xarray
`standardized_index()`, and `energy.srepi()`,
`energy.standardized_demand_index()`, and `energy.srli()`. Pooled ranking stays in
Python, so the native empirical kernel runs only with the default window.

Decision 2's `aggregation` keyword landed with this record
([#1158](https://github.com/monocongo/climate_indices/issues/1158)) and reached the
xarray surfaces with
[#1174](https://github.com/monocongo/climate_indices/issues/1174). Decisions 3
through 6 landed with their implementations, recorded in the amendments above;
decision 7 records scope, not code. Each decision is amended as its implementation
lands, as {doc}`0013-flood-module-api-and-naming` was for flood.

The energy-meteorology family
([#1157](https://github.com/monocongo/climate_indices/issues/1157)) adds demand
proxies (degree days), resource transforms (hub-height wind, capacity factors), and
standardized energy indices (SREPI, a standardized demand index, SRLI) with
run-theory energy-drought events. {doc}`0005-fire-module-api` and
{doc}`0013-flood-module-api-and-naming` settled the same question for fire
and flood, and apply here: a family of indices that share a physical basis and evolve
together gets a namespaced package rather than the drought-oriented `indices.py` and
`compute.py`, whose top-level names it would crowd.

`indices.standardized_index()` cannot carry the standardized energy indices as it
stood. It *summed* the series over `scale`, which is right for accumulated precipitation
but not for a capacity factor. A mean keeps the accumulated value in the series' own
range — a week's mean stays in [0, 1] where a week's sum does not — which both preserves
what the number means (a fraction of capacity) and keeps a support-constrained family
such as beta fittable to it. A rank-based standardization is indifferent to the choice,
since a sum is its mean multiplied by `scale` and the ranks are identical; the
aggregation matters for the value's interpretation and for the families whose support is
the series' range. And the seam clipped negatives to zero, which is wrong for a signed
series such as residual load.

## Decision

1. **Package.** Energy computations live in `climate_indices.energy`, a package with a
   public facade over underscore-prefixed implementation modules, mirroring `fire` and
   `flood`, and are imported as `from climate_indices import energy`. No energy function
   is re-exported from the package root or added to `typed_public_api.py`. Stable xarray
   paths keep the `energy.<name>` route under
   {doc}`0012-xarray-api-stays-beta-through-3.0.0`. The package lands with
   its first real function, not as an empty scaffold.

2. **Aggregation is a keyword of the seam.** `indices.standardized_index()`,
   `indices.fit_diagnostics()`, and `compute.prepare_scaled()` take `aggregation`,
   `"sum"` or `"mean"`, default `"sum"`. Reused fitting parameters must come from
   diagnostics with the same aggregation. A mean is the same windowed sum divided by `scale`, so the leading
   `scale - 1` pad and any window holding a missing value stay NaN exactly as they do
   for a sum, and a `scale` of 1 is the identity under either. The other index surfaces
   pass nothing and get the sum they always got. The pinned-golden parity tests in
   `tests/test_standardized_index.py` pin `standardized_index()` against `spi()` and the
   reference fixtures; every other surface (SPEI, PNP, EDDI, the xarray adapter, and the
   CLI) rests on the unchanged default, and no call site supplies `aggregation`.
   Aggregation and distribution are independent: an accumulated
   non-negative series is `sum` with gamma, a capacity factor is `mean` with beta or
   empirical, and residual load is either with empirical.

3. **Signed and bounded series.** A signed series enters the seam on the empirical
   (rank-based) path only, which has no support to violate and no zero mass.
   A bounded [0, 1] series takes beta or empirical. The parametric families (gamma,
   Pearson Type III, log-logistic) reject a series with negatives when negatives are not
   being clipped, rather than clip it: clipping residual load to zero would corrupt the
   index. `clip_negatives` becomes a keyword of `standardized_index()` with default
   `True`, so existing callers are untouched. That keyword and the rejection land with
   the first consumer that needs them
   ([#1168](https://github.com/monocongo/climate_indices/issues/1168),
   [#1172](https://github.com/monocongo/climate_indices/issues/1172)), because exposing
   `clip_negatives=False` before an empirical path exists would let a signed series
   reach a gamma fit.

4. **Standardization is unit-agnostic.** SREPI, the standardized demand index, and SRLI
   standardize whatever series they are handed; the seam carries no unit contract. That
   is what lets them accept capacity factors, degree days, and demand proxies from any
   source. Every energy docstring and page states that the indices describe the
   climate-driven component of supply and demand, not actual generation or load.

5. **Calendar.** Daily first. Weekly is daily input with `scale=7`, and a 30-day window
   is daily input with `scale=30`; calendar months use monthly periodicity. Neither adds
   a `Periodicity` member. Sub-daily support
   ([#1177](https://github.com/monocongo/climate_indices/issues/1177)) is an additive
   `Periodicity` member, so it changes no signature; it landed as
   `Periodicity.hourly`, and the amendment above records its contract.

6. **Units.** Temperature is °C, the library's internal convention (`_units.py`
   normalizes K and °F at its boundary). A degree day is a temperature *difference*, so
   K and °C are numerically interchangeable and the output is labelled °C day; the exact
   UDUNITS string belongs to the CF registry entries
   ([#1162](https://github.com/monocongo/climate_indices/issues/1162)). Wind is
   m s⁻¹ and radiation is a mean flux in W m⁻².

7. **Scope boundary.** The physical transforms are implemented in this package rather
   than deferred to `xclim`, `pvlib`, or `windpowerlib`: heating, cooling, and energy
   degree days ([#1164](https://github.com/monocongo/climate_indices/issues/1164)), the
   hub-height wind-speed profile
   ([#1165](https://github.com/monocongo/climate_indices/issues/1165)), the wind
   capacity factor ([#1175](https://github.com/monocongo/climate_indices/issues/1175)),
   and the temperature-derated PV capacity factor
   ([#1176](https://github.com/monocongo/climate_indices/issues/1176)). This reverses
   the earlier position that the transforms would be left to external recipes; the
   duplication of `xclim`'s degree days and `wind_profile`, and the overlap with
   `pvlib`'s prior art for the temperature-derated PV model, is accepted deliberately.
   Each is a published formula — the degree days and the wind profile are closed-form
   arithmetic over standard meteorological fields, and the PV model is the simple
   reference configuration [#1176](https://github.com/monocongo/climate_indices/issues/1176)
   specifies against Jerez et al. (2015); none needs a dependency. The standardized
   demand index ([#1171](https://github.com/monocongo/climate_indices/issues/1171)) takes
   degree days as a direct input, so deferring them would leave the family's own indices
   unreachable from raw reanalysis without an external step. `xclim`, `pvlib`, and
   `windpowerlib` are documented alternatives and validation references, never
   dependencies. Out of scope, as the epic states: electricity-demand models calibrated
   on measured load, hydropower generation models calibrated on observed generation,
   bias adjustment of wind speed, and aggregation to administrative units. The required
   wind `power_curve` meets criterion 2 in the energy subsystem design note.
   The other four criteria do not hold. The two-criterion threshold is not met, so no
   sibling repository is justified.

The family adds no dependency. A transform that needs one would go behind a
`climate-indices[energy]` extra, so the core install is unchanged, as the epic's
definition of done requires; decision 7 keeps every transform in scope dependency-free,
so the extra stays unused.

Which physical transforms the package implements at all was the scope question of
[#1159](https://github.com/monocongo/climate_indices/issues/1159), which overlaps
`xclim`, `pvlib`, and `windpowerlib`. Decision 7 resolves it by keeping all of them in
scope; the package boundary, the aggregation seam, the support rules, the calendar, and
the units never depended on the answer and do not change. The function names, argument
conventions, and degree-day contract for those transforms are fixed in the internal
energy subsystem design note
([docs/design/energy-subsystem.md](https://github.com/monocongo/climate_indices/blob/main/docs/design/energy-subsystem.md)).
That note stays in the repository rather than this site because it also records
deferred work, the arrangement {doc}`0005-fire-module-api` chose for fire. The oracle survey
([#1160](https://github.com/monocongo/climate_indices/issues/1160)) does not confirm
the default constant values: the C3S degree-day schemes pair an activation threshold
with a separate reference temperature, which one base per transform cannot reproduce.
No default is established until
[#1163](https://github.com/monocongo/climate_indices/issues/1163) and
[#1164](https://github.com/monocongo/climate_indices/issues/1164) reconcile that
contract.

## Consequences

This amends {doc}`0001-dual-numpy-xarray-api` only for the energy family, as
ADR-0005 did for fire and ADR-0013 for flood. `standardized_index()` gains one keyword
whose default reproduces its existing output, which is why this record can land the seam
change without touching SPI or SPEI behavior.

`aggregation` is not exposed on the CLI. The CLI's energy indices
(`--index srepi`, `demand_index`, `srli`) use each wrapper's default; the xarray
entry and the energy adapters carry the keyword, and
[#1174](https://github.com/monocongo/climate_indices/issues/1174) landed both.

Every contract in decisions 3 through 6 has landed with its implementation, the
sub-daily `Periodicity` member that decision 5 records as additive
([#1177](https://github.com/monocongo/climate_indices/issues/1177)) among them.
Decision 7 records scope, not code, and does not change as the transforms land.

# Energy oracle survey

Internal research for [#1160](https://github.com/monocongo/climate_indices/issues/1160),
covering Waves 2–4 of [#1157](https://github.com/monocongo/climate_indices/issues/1157).
Research date: 2026-10-16. Scope: the contracts at repository commit
`6715513abc076f741f7fb028c6055179c8015b26`, including
[ADR-0019](../adr/0019-energy-module-api-and-aggregation.md) and the
[energy design](../design/energy-subsystem.md).

These indices describe the **climate-driven component of energy supply and demand**,
not actual generation or load, which also depend on installed capacity, technology,
curtailment, markets, and behaviour.

## Findings and evidence classes

Use SEI for empirical standardization, not beta. Use published degree-day examples
and matched wind-profile implementations before attempting C3S product comparisons.
C3S is a useful energy-data source, but its national degree days, turbine-specific
wind factors, and full-chain PV factors are not interchangeable with the planned
pointwise transforms. Three convention gaps need explicit decisions: degree-day
activation thresholds, beta endpoints/estimator, and hourly calibration groups.

This is a source survey, **not completed validation**. No energy fixtures, numerical
comparisons, or figure digitizations were produced. Tolerances below are proposals.
The classification follows [VALIDATION.md](../../VALIDATION.md):

- **Authoritative reference**: published worked numbers, or provider output with
  matching inputs and conventions. Authority does not establish algorithm independence.
- **Independent-implementation comparison**: another implementation given identical
  inputs/configuration; SEI has this role for empirical indices.
- **Published-figure reproduction**: digitized or reconstructed figures with stated
  read-off uncertainty, not full-precision numerical references.
- **Model-to-model characterization**: differing physical models/configurations;
  agreement describes sensitivity, not numerical correctness.
- **Specification-level / regression only**: formula identities or frozen internal
  results. Neither constitutes external numerical validation.

### Complete Wave 2–4 inventory

These are recommended evidence routes, not claims that fixtures already exist.

| Wave / ticket | Output | Oracle finding and classification |
| --- | --- | --- |
| 2 / #1163–#1164 | HDD, CDD | CPC publishes a worked CDD series; Eurostat and C3S publish scalar examples. Authoritative references for those examples only. National C3S/Eurostat/CPC products are conditional product comparisons, not paired pointwise oracles. |
| 2 / #1164 | EDD | C3S defines population-weighted HDD + CDD. Local `EDD = HDD + CDD` is a specification identity; no matched daily gridded numerical oracle identified. |
| 2 / #1163, #1165 | Power-law wind profile | `xclim.indices.wind_profile` is an independent-implementation candidate. C3S's WS10/WS100-derived shear is consistency evidence only. |
| 2 / #1165 | Log wind profile | `windpowerlib.wind_speed.logarithmic_profile`, with zero displacement, is an independent-implementation candidate; C3S's power-law product does not validate it. |
| 2 / #1166 | Degree-day/wind xarray, CF metadata, Dask | Inherit core references; NumPy/eager/Dask equality and metadata checks are regression/contract evidence, not new scientific oracles. |
| 2 / #1167 | Standardized hydropower precipitation recipe | Existing standardization references validate the chosen statistical transform. No matched generation oracle: C3S IWP is a differently accumulated/weighted precipitation proxy; European hydro outputs use fitted models. |
| 3 / #1168 | Empirical distribution | Pinned SEI is the primary independent implementation, with the plotting position and grouping precautions below. |
| 3 / #1169 | Beta distribution | **No matched oracle identified. SEI does not implement beta.** Separate fitting/endpoint reference required; specification/regression evidence only until then. |
| 3 / #1170–#1171 | SREPI | SEI empirical normal indices on identical production or capacity-factor inputs: independent-implementation comparison. Paper figures are supplementary reproduction only. |
| 3 / #1171 | Standardized demand index | Same SEI transformation on demand or EDD, not validation of a demand-to-load model. Beta remains unsupported by this oracle. |
| 3 / #1170, #1172 | SRLI | SEI empirical transformation of signed, unit-compatible demand minus production: independent-implementation comparison. |
| 3 / #1173 | Energy-drought events | SEI is a qualified cross-check only: equality, magnitude, and intensity differ from `runs.identify_runs()`. No direct oracle for the complete planned event output. |
| 3 / #1174 | Standardized-energy xarray, CF metadata, Dask | Inherit statistical references; execution and metadata parity remain regression/contract checks. |
| 4 / #1175 | Wind capacity factor | `xclim.indices.wind_power_potential` can independently check the matching generic cubic curve. C3S on/offshore factors are model-to-model unless turbine, numerical curve, heights, density policy, and input lineage match. |
| 4 / #1176 | PV capacity factor | Jerez et al. defines the simple model; `pvlib.pvsystem.pvwatts_dc` can check its power/temperature term with supplied cell temperature. C3S and a full pvlib chain are model-to-model, not exact Jerez oracles. |
| 4 / #1177 | Hourly standardized indices | SEI accepts hourly data, but supplies no automatic seasonal-diurnal calendar oracle. Explicit-group statistical comparison is possible after the project selects that contract. |

## Degree days: conventions before fixtures

### C3S and Eurostat are thresholded, not single-base formulas

For daily mean 2 m temperature `T` in °C, the C3S
[demand model, §2.1][c3s-demand] gives:

```text
HDD = 18 - T  if T < 15;  otherwise 0
CDD = T - 21  if T >= 24; otherwise 0
```

Its examples are `T=12 → HDD=6`, `T=16 → HDD=0`, `T=26 → CDD=5`,
`T=22 → CDD=0`. The [Eurostat metadata][eurostat] uses the same temperatures
but **`T <= 15`** for heating, and `T >= 24` for cooling. At exactly 15 °C,
Eurostat therefore gives HDD=3, while the explicit C3S model equation gives 0.
The [C3S PUG][c3s-pug] summary also writes cooling as `T > 24`, conflicting with
`>= 24` in the dedicated model page. Boundary behavior needs provider confirmation
or direct product interrogation; do not conceal it with numerical tolerance.

A single-base formula `max(B - T, 0)` cannot represent a heating activation threshold
of 15 with a reference temperature of 18. Nor can `max(T - B, 0)` represent the
24/21 cooling pair. For example, heating base 18 incorrectly gives HDD=2 at T=16;
base 15 incorrectly gives HDD=3 at T=12.

**Contract gap:** the energy design currently calls these pairs merely non-default
base temperatures. Its one-base-per-transform signature cannot reproduce them.
Before #1163/#1164 fix defaults, explicitly reconcile that statement: either admit
separate activation/reference temperatures and equality rules, or retain a single-base
transform and use a matching single-base reference. The survey does not amend ADRs,
signatures, or implementation tickets, and it does **not** confirm degree-day
defaults: no default is established until this gap is resolved. If C3S remains the intended convention, the
recommended pairs are heating `(threshold=15, reference=18)` and cooling `(24,21)`,
subject to resolving the documented equality disagreement.

### What the C3S product can actually supply

The [PUG Tables 2.1 and 3.1][c3s-pug] list hourly gridded TA, but HDD/CDD/EDD
only as **ADM0 CSV, monthly/seasonal/annual**. Wind and solar factors are hourly grids;
this does not imply the degree days are. No paired daily gridded T/HDD/CDD/EDD
extract was verified.

The demand page computes HDD/CDD at grid level before national population weighting;
EDD is the sum of the weighted components. The [spatial procedure, §2.4][c3s-spatial]
uses special coastal population masks. Applying the nonlinear thresholds to national
mean temperature is not equivalent. The demand page also describes monthly
climatologies, while the PUG advertises time series; inspect actual timestamps and
monthly normalization before treating a file as a daily sum. The
[temporal procedure][c3s-temporal] explicitly sums monthly degree days into seasonal
and annual totals. The catalogue labels degree-day units `°C`; do not assume this
settles the monthly total-versus-mean question.

**For #1163:** begin with published scalar/boundary convention tests. A national C3S
comparison additionally requires the exact gridded daily temperatures, population
weights/masks, aggregation order, monthly normalization, and output vintage.
If those cannot be recovered, record the product as characterization, not a paired
validation fixture. `EDD = HDD + CDD` alone is only a consistency identity.

### NOAA CPC: a genuinely different single-base convention

The [CPC explanation][cpc-explanation] defines daily mean as `(Tmax + Tmin)/2`,
with base 65 °F for both HDD and CDD:

```text
HDD_F = max(65 - mean_F, 0)
CDD_F = max(mean_F - 65, 0)
```

Its published seven-day CDD example has means `[67,65,70,74,78,65,68]` °F,
outputs `[2,0,5,9,13,0,3]`, total **32 °F day**. This is a small authoritative
worked reference for the single-base cooling calculation. Heating has the documented
formula but no corresponding worked series was identified here. Convert a degree-day
*difference* by `5/9`, without the Fahrenheit temperature offset; the corresponding
Celsius base is `(65-32)*5/9`, not a rounded 18 °C.

CPC regional degree days are population weighted, not calculated from regional mean
T. Its [current statistics page][cpc-data] distinguishes the 2010 Census/ACS weighted
archive from legacy 2000-population/1971–2000-normal products and flags archive
corrections. The explanation page describes that older station/regression workflow;
do not attribute it automatically to the newer gridded archive. Retrieval is via the
linked HTTPS, pipe-delimited annual [daily-data directories][cpc-archive]. Freeze the
selected file, region, weights vintage, units, and retrieval date. Raw observations,
forecasts, normals, and seasonal accumulations are different products.

Eurostat republishes JRC monthly station-derived, 25 km gridded estimates aggregated
to NUTS regions; annual values sum months [eurostat]. Those values cannot validate
an ERA5 point transform without matched inputs and spatial support. Retrieve a small
`nrg_chdd` subset through the [Data Browser][eurostat-data], preserving the exported
data code, dimensions, flags, units, geography revision, and request.

## Wind: keep profile consistency separate from power modelling

[C3S preprocessing][c3s-pug] derives speeds from `sqrt(u²+v²)`, corrects the ERA5
10:00 UTC WS10 drop by interpolation, discards WS10 extremes above 60 m s⁻¹, and
applies separate GWA2 mean-ratio adjustments at 10 and 100 m. GWA2 adjustment covers
land/coast to 20 km offshore; the wider domain extends to 300 km. Raw ERA5 and these
processed fields are not interchangeable inputs.

The [C3S power-law page][c3s-profile] defines
`v2 = v1 * (h2/h1)**alpha`, with alpha derived from adjusted 10/100 m winds over
2011–2021, stratified into 24 hours × 12 months per cell. The PUG distinguishes native
bias-adjusted WS100 (`mbc`) from power-law-derived WS100 (`pwl`). Using
`alpha = log(v100/v10)/log(10)` from the same positive pair and recovering v100 is
an algebraic round trip, **not independent validation**. The distributed climatological
alpha need not reproduce every native hourly WS100 observation. Calm winds make the
log-ratio undefined; exclude them from that identity and test calm behavior separately.

For #1165, prefer offline outputs from pinned [xclim wind_profile][xclim-wind]
for the power law with explicit alpha, and [windpowerlib logarithmic_profile][wpl-log]
for `v2 = v1*log(h2/z0)/log(h1/z0)`, explicitly `obstacle_height=0`.
xclim documents only a power-law method; it is not a log-profile oracle. Match heights,
roughness, units, valid-domain policy, and array broadcasting. Neither reference
establishes that a neutral profile predicts real hub-height observations.

The [C3S wind model, Table 2.1][c3s-wind] fixes these reference turbines:

| Turbine | Domain | Rated MW / hub m | Native-WS100 / power-law-WS100 codes |
| --- | --- | --- | --- |
| Vestas V164/8000 | Offshore | 8 / 105 | WP000 / WP010 |
| GE 2.5-103 | Onshore | 2.5 / 100 | WP001 / WP011 |
| Gamesa G132/3300 | Onshore | 3.3 / 84 | WP002; uses WS10 |
| IEA 15MW_240_RWT | Offshore | 15 / 150 | WP003 / WP013 |
| NREL Bespoke_6MW_170 | Onshore | 6 / 135 | WP004 / WP014 |

The model extrapolates to the named hub height and normalizes turbine power by rated
power. Its curves are **plotted**, not supplied as numerical tables on the fetched page;
interpolation and density details are insufficient for an exact reconstruction.
Offshore outputs are gridded only in PUG Table 3.1. No fully matched C3S power-curve
oracle was established. #1175 must recover the numerical curve and its redistribution
terms, cut-in/rated/cut-out equality rules, density correction, wind variant and
aggregation before promoting comparison to input-matched status. Do not digitize the
plot and call it a full-precision reference. For the generic cubic curve, xclim's
[wind_power_potential][xclim-power] documents the regions and density scaling; compare
only with identical parameters and boundary rules. A user-supplied tabulated curve
also needs interpolation/knot tests; a generic cubic curve does not validate it.

## PV: Jerez is not the C3S configuration

[Jerez et al. (2015), Methods, equations 1–3][jerez] specifies, for irradiance G in
W m⁻², air temperature Ta in °C, and surface wind v in m s⁻¹:

```text
Tcell = 4.3 + 0.943*Ta + 0.028*G - 1.528*v
PR = 1 - 0.005*(Tcell - 25)
PVpot = PR * G/1000
```

The equation images and coefficient text were inspected. This is the definitional
source for #1176, not an independently tabulated numerical fixture. Cell temperature
25 °C is the reference; air temperature 25 °C does not guarantee it. The paper
explicitly permits potential above 1. Clipping to [0,1] is therefore a **project policy**
to test separately, not a Jerez result. Its climate inputs are 3-hourly EURO-CORDEX;
its regional figures include installed-capacity assumptions and cannot directly
validate an hourly ERA5 point calculation. Confirm the wind measurement height used
for any fixture, rather than silently interchanging surface and hub-height wind.

A narrow independent comparison is [pvlib pvwatts_dc][pvlib-dc], with the same
externally supplied `Tcell`, `effective_irradiance=G`, `pdc0=1`, `gamma_pdc=-0.005`,
and `temp_ref=25`, without optional irradiance adjustments, losses, or inverter.
This checks the power/temperature term only; it does not independently check the
Chenni/Jerez cell-temperature equation. [pvlib Faiman][pvlib-faiman] is a different
thermal model, not a replacement oracle for that equation.

The [C3S PV model][c3s-pv] is instead a full AC-output/DC-capacity chain: 15-minute
interpolation and hourly re-averaging, irradiance decomposition/transposition,
fixed tilt at 75% of optimum and a documented south-facing orientation, optical
losses, calibrated irradiance-dependent efficiency, Ross thermal modelling with
coefficient -0.0045, and inverter losses. These are structural differences from Jerez,
not floating-point noise. Its private plant metadata and future-tense open-source
statement do not establish a reproducible public generator. Use C3S only for
model-to-model characterization unless the complete configuration becomes available.

ERA5 hourly SSRD in J m⁻² must be divided by the actual interval duration to become
mean W m⁻²; the C3S GHI field is already converted (3600 s for its hourly intervals).
Do not divide it again. Document interval endpoints/time zones and test night,
clipping, invalid inputs, and radiation-unit conversion separately.

## Standardized energy indices: SEI oracle contract

### Pin source, not just package version

Recommend SEI **0.2.0 at originating commit
`4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14`** [sei-description]. The verified
CRAN-mirror 0.2.0 commit is `514b399d19a363af1aadafb6fb07f25f27ce856e` [sei-cran].
Both advertise the same version, but the former fixes grouped calibration selection
from `x_ref[ind]` to `x_ref[ind_ref]` [sei-index]. Source inspection confirmed the
distinctions below; R was not executed.

For uncensored, non-missing calibration size n, [fit_dist][sei-fit] uses:

```text
k(x) = count(x_ref <= x)
p(x) = (1 + k(x))/(n + 2)
z(x) = qnorm(p(x))
```

Ties use their maximum rank, not average rank. There is no interpolation/tail
extrapolation: below the calibration minimum p=1/(n+2); at/above its maximum
p=(n+1)/(n+2). A constant sample consequently does **not** map its observed value
to p=0.5. Zeros/ones are ordinary ties with unbounded defaults. Default minimum is
10 uncensored calibration observations per fit; empirical fitting warns below 100,
but does not automatically fall back to another distribution. #1168 must explicitly
select its minimum-observation policy rather than confusing a recommendation with a
runtime requirement.

[get_pit][sei-pit] treats finite bounds inclusively (`x <= lower`, `x >= upper`) as
censored and fits the interior. Interior probability becomes
`p_lower + (1-p_lower-p_upper)*F_interior(x)`. Two finite bounds require an explicit
numeric pair of endpoint **probabilities**, not automatic normal-scale constants.
For lower-only censoring with mass p0, `"none"` assigns p0, `"prob"` assigns p0/2,
and `"normal"` assigns normal score `-dnorm(qnorm(p0))/p0`. The default normal index
uses `cens="normal"`, unlike `get_pit`'s `"none"` default; record it explicitly.
Boundary evaluations can remain NA when calibration has no
censored observations; upper-only `cens="none"` assigns p_upper, not 1. These source
branches are not recommended physical policies. Validate [0,1] inputs independently:
SEI censoring also accepts observations beyond the stated bounds.

**SEI rejects `dist="beta"`.** Its bounded-data censoring is not a beta estimator
or a canonical zero/one-inflated-beta policy. #1169 needs a separately pinned oracle,
matched estimator, minimum sample count and endpoint policy. #1170's requested
parametric SEI control can use a supported distribution such as gamma on suitable
inputs, but must not be labelled beta validation. Reusing SciPy on both sides would
check wiring, not independently validate SciPy's fit.

### Aggregation, groups, and signed series

[std_index][sei-index] groups before rescaling/rolling aggregation, and grouped
recursion omits `method`, `n_thres`, and `rescale_fun`. Avoid that path: aggregate the
complete series first, then select calibration/evaluation observations and call once
per calendar group, restoring original order. The [rolling helper][sei-aggregate]
includes the current observation and defaults to allowing 10% missing timestamps;
use complete-window preprocessing matching ADR-0019 instead. Calendar-week rescaling
and rolling seven-day windows are not equivalent.

For rank-only fixtures, explicitly use `dist="empirical"`, `index_type="normal"`,
`lower=-Inf`, `upper=Inf`, `cens="none"`, `n_thres=10`, and separate `x_ref`/`x_new`.
Save probabilities with a corresponding `index_type="prob01"` call. Use mean
Aggregation for capacity factors, sums for accumulated production/demand, and the
explicit chosen aggregation for residual load. Save the scaled inputs too: fixed-length
sums and means have identical empirical ranks and can conceal an aggregation bug.

SREPI is low-tail production; standardized demand and SRLI use the high tail for stress.
SRLI inputs are signed `demand - production`; retain negatives. Capacity factors must
first be converted to production using compatible installed capacities and units before
subtraction from demand. SEI validates the statistical transformation, not these
energy-system assumptions or an EDD-to-load relationship [sei-paper].

### Runs and hourly support

[SEI get_drought][sei-runs] uses `<=`/`>=`, magnitude `sum(abs(index))`, and, for
multiple thresholds, an intensity-category count. It can pool gaps or use a separate
termination threshold. The project's [Run contract](../../src/climate_indices/CONTEXT.md)
uses strict comparisons, threshold-relative magnitude, and mean-deviation intensity;
NaN breaks a run. SEI does not directly return all planned onset/peak/summary outputs,
and its loop is not a missing-value-policy oracle.

With identical event membership below a negative or above a positive threshold,
`SEI_magnitude = project_magnitude + abs(threshold)*duration`. That identity is a
qualification, not direct parity. Test threshold equality, duration, missing gaps,
minimum-duration filtering and peaks against explicit contract examples; do not widen
tolerances to reconcile definitions. SEI row counts are not elapsed-time durations.

SEI accepts hourly timestamps but does not automatically stratify by hour of day,
day of year, season, leap days, time zone, or daylight saving. #1177 must first select
that calendar policy. A pooled hourly comparison cannot validate seasonal-diurnal
calibration. The fetched [energy preprint][energy-preprint] presents daily results for
concision, not proof that daily resolution is scientifically preferable. Its published
figures are not the same experiment as this project's calendar-stratified contract.

## Hydropower recipe limits

The [C3S PUG][c3s-pug] distinguishes European, observation-trained random-forest
HRG/HRO/HRI from global installed-capacity-weighted precipitation (IWP).
IWP already uses country-specific n-month precipitation accumulations and is delivered
as national monthly CSV; seasonal/annual IWP uses means [c3s-temporal]. It is not an
unweighted monthly precipitation input, a standardized index, or measured generation.
For #1167, record whether input is raw capacity-weighted monthly precipitation or
already accumulated IWP. Applying `scale=3` to an n-month IWP is another overlapping
aggregation, not automatically a three-month resource index. A C3S series may illustrate
the recipe, but does not independently validate hydropower generation or the recipe's
chosen standardization distribution.

## Retrieval, redistribution, and fixture handoff

| Source | Retrieval and rights finding |
| --- | --- |
| C3S | Dataset `sis-energy-global-reanalysis`, DOI [10.24381/3bb607bd](https://doi.org/10.24381/3bb607bd), version 1.00. The public [catalogue API][c3s-catalogue] explicitly declares **CC-BY-4.0**, corroborating the PUG. Use the CDS download form/API, preserve its generated request, selected wind/technology variant, source filenames, metadata and checksums. Attribution must identify Copernicus/C3S, DOI, version, retrieval date, and subset/conversion changes. Account/terms acceptance is a human retrieval step if required. No dataset downloaded here. |
| Eurostat/JRC | Use a frozen `nrg_chdd` export. [Eurostat reuse policy][eurostat-rights] permits attributed reuse with exceptions, including third-party/source-specific conditions. Since metadata explicitly identifies JRC as producer, confirm that source's terms before committing its numbers; do not blanket-label them CC-BY. Cite data code/URL and access date, identify modifications and required disclaimer. |
| NOAA CPC | Public HTTPS annual archive; NWS [terms][nws-rights] place its information in the public domain unless annotated otherwise. Attribute NOAA/NWS/CPC; identify derived subsets, preserve source notices, imply no endorsement. Check the selected file for exceptions. |
| SEI | [DESCRIPTION][sei-description] says **GPL (>= 2)**, not GPL-3. Generate outputs offline on committed synthetic inputs plus a verified real subset; do not copy package implementation into the library. Software, input, and generated-output rights are separate; [GNU's output guidance][gpl-output] is not a blanket license grant for input data. Maintainer review must approve redistribution. |
| xclim / windpowerlib / pvlib | Prospective offline independent implementations, not new dependencies. Current repository licenses are Apache-2.0 / MIT / BSD-3-Clause respectively [xclim-license], [wpl-license], [pvlib-license]. Pin release, commit and dependency versions before fixture generation; review selected numerical power-curve/data rights separately from software licenses. |
| Jerez article | [Article][jerez] declares CC BY 4.0, with third-party exceptions. Cite equation numbers and coefficients; figures and EURO-CORDEX inputs are not a bundled matched numerical oracle. |

For #1163 and #1170, fixture provenance must validate against
[`tests/fixture/provenance_schema.json`](../../tests/fixture/provenance_schema.json),
which requires source, URL, download date, subset description, SHA-256 checksum,
fixture version and validation tolerance. Also fill the optional `license`,
`citation` and `doi` fields that the rights findings above depend on; the schema has
no rationale field, so put each source's rationale in `notes`. Preserve full original-file and extracted-array checksums,
request/extraction script, units, calendar, missing masks, calibration dates, aggregation
and transformation order. Record packing/rounding and all source variants. The schema
rejects unknown top-level keys; put source commits, R `sessionInfo()`, generator
versions and detailed conventions in `notes` or accompanying metadata. Tests consume
committed outputs offline, not network services or R.

For #1170, use deterministic bounded production, demand and signed residual-load
inputs plus one rights-verified real C3S subset; archive SEI source SHA-256, R version,
dependencies and exact per-group calls. Include ties, constants, all-zero/all-one groups,
NaNs, disjoint calibration/evaluation periods, outside-calibration-range evaluations,
finite-bound evaluations absent from calibration, and group counts 9/10/99/100.
Keep scaled inputs, probabilities and normal indices separate. Obtain maintainer
confirmation of extraction/generation in both fixture PRs. No fixture generated by the
future implementation can serve as its own external oracle.

## Proposed tolerances, not measured agreement

Unless noted, use `rtol=0`; absolute error avoids instability at zero. These starting
values require measurement on generated fixtures, with maximum errors and tail behavior
recorded separately in `measured_stats` before acceptance. No new measured errors
exist in this survey.

| Comparison | Recommendation and rationale |
| --- | --- |
| Published integer CPC example / scalar degree-day examples | Exact equality in original integer units; after °C conversion start `atol=1e-12` °C day for double-precision arithmetic. Equality/activation branches and NaN masks must match exactly. |
| Packed C3S degree-day products | No universal tolerance yet. Recover packing, monthly normalization, inputs and weights first. Away from threshold crossings, bound error by input quantization propagated through sums/weights plus output rounding. A threshold crossing can introduce about 3 °C day discontinuity; it is not roundoff to absorb. |
| Full-precision power/log profile implementations | Start `atol=1e-12` m s⁻¹ on a bounded ordinary-wind test domain. Exact identity-height/calm cases separately. C3S packed fields need a precision-derived bound; the PUG's 16-bit storage alone supplies no universal decimal tolerance. |
| Empirical ranks/counts, groups, missing masks, event membership/durations | Exact equality. Different conventions are failures, not tolerance choices. |
| SEI empirical probabilities / normal indices | Start `atol=1e-15` / `1e-12` respectively for matched counts and groups; measure inverse-normal tails before adopting the latter. Compare unclipped scores and test any project clipping separately. |
| Beta fits/endpoints | No SEI-backed tolerance. Select an independent estimator and endpoint convention first; do not borrow gamma's tolerance. |
| SEI parametric control | Existing zero-handling precedent uses parameter-matched `atol=1e-6` and end-to-end gamma `0.01`; VALIDATION.md reports maximum difference 0.0032 there. Those are prior precipitation results, not energy/beta measurements. Set any new control's bound from its matched estimator and fixtures. |
| Generic wind CF / pvlib matched power term | Start `atol=1e-12` in dimensionless CF for identical full-precision formulas/configurations. Exact curve-region membership and clipping separately. This does not validate arbitrary tabulated curves or the Jerez thermal model. |
| C3S versus generic wind/Jerez PV, or differing pvlib chains | No justified numerical pass/fail tolerance. Report matched-period bias, MAE/RMSE, correlation and configuration differences first; near-zero CF makes relative error misleading. An arbitrary ±0.1 CF gate would hide model mismatch. |
| Figure reproduction | No figure digitized; no numeric uncertainty/tolerance established. Record pixel/read-off uncertainty if this lower-tier evidence is later used. |

## Remaining decisions and access gaps

1. **#1163/#1164:** reconcile one-base signatures with threshold/reference pairs;
   resolve C3S equality inconsistencies and actual monthly normalization. No matched
   daily gridded product is established. Do not promise the fixture requested in #1163
   without those inputs.
2. **#1169/#1170:** select a beta estimator/endpoint oracle; SEI's supported parametric
   control cannot satisfy it. Fix the empirical minimum count explicitly in #1168.
3. **#1173/#1177:** retain the project's Run conventions explicitly, and select the
   hourly calendar before generating grouped reference outputs.
4. **#1175/#1176:** C3S numerical power curves/full PV generator were not retrieved;
   the documentation does not establish exact reproducibility. Prefer narrow matched
   implementation checks and label broader comparisons model-to-model.
5. The earlier research fetch of the final Allen–Otero [paper DOI][energy-final]
   returned HTTP 406 and its publisher page 403. The author preprint and R Journal
   article were accessible. Manual final-paper retrieval is required before asserting
   that its figures/text exactly match the preprint. No dataset licenses were accepted,
   generators executed, or statistical agreement measured during this survey.

## Primary sources

C3S pages were followed from the dataset documentation tab. Live pages describe the
versions fetched on the research date; fixture work must archive/pin its own copies.

[c3s-pug]: https://confluence.ecmwf.int/spaces/CKB/pages/621041351/Global+climate+and+energy+indicators+from+1950+to+present+derived+from+reanalysis+Product+User+Guide+PUG
[c3s-demand]: https://confluence.ecmwf.int/spaces/CKB/pages/621045420/Global+Climate+and+Energy+Indicators+Electricity+and+Energy+Demand+Models
[c3s-profile]: https://confluence.ecmwf.int/spaces/CKB/pages/627852198/Global+Climate+and+Energy+Indicators+Power+Law+for+Wind+Profile+Scaling
[c3s-wind]: https://confluence.ecmwf.int/spaces/CKB/pages/621045498/Global+Climate+and+Energy+Indicators+Wind+Power+Conversion+Model
[c3s-pv]: https://confluence.ecmwf.int/spaces/CKB/pages/621045491/Global+Climate+and+Energy+Indicators+Solar+Photovoltaic+Energy+Conversion+Model
[c3s-spatial]: https://confluence.ecmwf.int/spaces/CKB/pages/627852201/Global+Climate+and+Energy+Indicators+Spatial+Aggregation+Procedure
[c3s-temporal]: https://confluence.ecmwf.int/x/sENsJQ
[c3s-catalogue]: https://cds.climate.copernicus.eu/api/catalogue/v1/collections/sis-energy-global-reanalysis
[eurostat]: https://ec.europa.eu/eurostat/cache/metadata/EN/nrg_chdd_esms.htm
[eurostat-data]: https://ec.europa.eu/eurostat/databrowser/view/nrg_chdd/default/table?lang=en
[eurostat-rights]: https://ec.europa.eu/eurostat/help/copyright-notice
[cpc-explanation]: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days/ddayexp.shtml
[cpc-data]: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days/index.shtml
[cpc-archive]: https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/
[nws-rights]: https://www.weather.gov/disclaimer
[xclim-wind]: https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.wind_profile
[xclim-power]: https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.wind_power_potential
[xclim-license]: https://github.com/Ouranosinc/xclim/blob/main/LICENSE
[wpl-log]: https://windpowerlib.readthedocs.io/en/stable/temp/windpowerlib.wind_speed.logarithmic_profile.html
[wpl-license]: https://github.com/wind-python/windpowerlib/blob/dev/LICENSE
[jerez]: https://www.nature.com/articles/ncomms10014
[pvlib-dc]: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html
[pvlib-faiman]: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html
[pvlib-license]: https://github.com/pvlib/pvlib-python/blob/main/LICENSE
[sei-description]: https://github.com/noeliaof/SEI/blob/4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14/DESCRIPTION
[sei-fit]: https://github.com/noeliaof/SEI/blob/4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14/R/fit_dist.R
[sei-pit]: https://github.com/noeliaof/SEI/blob/4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14/R/get_pit.R
[sei-index]: https://github.com/noeliaof/SEI/blob/4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14/R/std_index.R
[sei-aggregate]: https://github.com/noeliaof/SEI/blob/4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14/R/aggregate_xts.R
[sei-runs]: https://github.com/noeliaof/SEI/blob/4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14/R/get_drought.R
[sei-cran]: https://github.com/cran/SEI/blob/514b399d19a363af1aadafb6fb07f25f27ce856e/R/std_index.R
[sei-paper]: https://journal.r-project.org/articles/RJ-2024-038/
[energy-preprint]: https://eartharxiv.org/repository/object/4752/download/11031/
[energy-final]: https://doi.org/10.1016/j.renene.2023.119206
[gpl-output]: https://www.gnu.org/licenses/gpl-faq.html#GPLOutput

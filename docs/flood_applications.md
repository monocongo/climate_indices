# Climate Indices for Flood and Wet-Extreme Applications

Climate indices describe environmental conditions associated with flood
potential, not floods themselves. Flooding depends on terrain, soils, land
cover, and river routing in addition to how wet the climate has been, so a
positive moisture anomaly does not establish that a flood occurred, will
occur, or where it would occur. This page is a companion to
{doc}`wildfire_applications`, which makes the same distinction for fire.

**None of the indices described here predicts whether, where, or when a flood
will occur.** Use them for hydroclimate context and wet-extreme situational
awareness, not as flood forecasts. Every index in this family describes
wetness conditions relevant to flood potential; the family stops at the
meteorological and climatological boundary, with one documented exception:
input-agnostic standardization that can also accept runoff or streamflow
([flood-family epic #1098][flood-epic]).

## From wet anomalies to flood potential

A useful conceptual chain is:

```text
Antecedent climatic wetness          SPI / SPEI / PHDI wet tail          (available)
              ↓
Accumulated / effective wetness      PE → EDI, I_F, API                  (available)
              ↓
Heavy-precipitation triggers         Rx1day / Rx5day / R95pTOT           (planned)
              ↓
Hydrologic response (boundary)       SRI / SSI via generic standardization (recipes available)
              ↓
Inundation / flood hazard            —                                   (out of scope)
```

Read each arrow as a possible physical pathway, never a deterministic
transition. Wetness accumulates through precipitation, but whether it becomes
a flood depends on how the catchment, soils, and channels respond
([flood-family epic #1098][flood-epic]).

## The wet tail of SPI, SPEI, and PHDI

SPI and SPEI are symmetric and multi-scalar: at any chosen Timescale,
positive values describe wetter-than-normal conditions, and neither was
designed only for drought. PHDI is a single fixed monthly water-balance index,
and its positive side describes an established wet spell. The existing indices
therefore already carry flood-potential information in their upper tail.

SPI and SPEI are dimensionless and standardized to mean 0 and standard
deviation 1, comparable across locations and climate regimes; positive values
are wetter than the Calibration Period norm. The computation and the clipping
of values to [-3.09, 3.09] are documented in {doc}`algorithm-reference`. PHDI,
the Palmer Hydrological Drought Index, is a water-balance severity value for
established ("backed-out") wet and dry spells rather than a
distribution-standardized probability, so its positive side describes an
accumulated surplus rather than a statement about recent rainfall.

The SPI and SPEI wet tail is not an observation of flooding. It is a
standardized probability statement about accumulated moisture relative to the
calibration history, and it is only one input to flood risk
([Seiler et al. (2002)][seiler-2002]; [flood-family epic #1098][flood-epic]).

### Timescale and flood type

For SPI and SPEI, the accumulation window ("Timescale" in prose; `scale` in
code) determines which part of the wetness chain an anomaly reflects. PHDI
has no selectable Timescale. Reading the same accumulation semantics
documented for drought in {doc}`algorithms` on the wet tail:

- **1–3 months**: recent precipitation surplus and near-surface soil
  moisture — antecedent wetness that can precondition a catchment's response.
- **6–12 months**: accumulated wetness and streamflow influence — conditions
  that can precede seasonal riverine flooding when catchments are already
  wet.
- **12–24 months or longer**: long-term hydrological and groundwater
  wetness — relevant to prolonged wet periods rather than a single event.

These mappings are interpretive guidance based on the accumulation window, not
calibrated flood thresholds. No universal SPI or SPEI value marks the onset of
flooding, and a region's flood response can differ from its moisture anomaly.
Seiler et al. (2002) found the wet SPI tail tracked conditions leading up to
major flood events in southern Córdoba, Argentina; that demonstrates the
potential of the approach in one region, not a transferable rule.

## Wet-extreme indices

The NumPy PE kernel, EDI, I_F, and API are available as `flood.effective_precipitation()`,
`flood.edi()`, `flood.flood_index()`, and `flood.antecedent_precipitation_index()`. Module and function names are fixed
by [ADR-0013](adr/0013-flood-module-api-and-naming.md), with argument names,
units, and signatures recorded in the internal flood subsystem design note.
The [flood-family epic #1098][flood-epic] tracks the implementation order.

- **Effective Precipitation (PE)** — *NumPy API available (FLOOD-08
  [#1105][flood-1105])*. The daily accumulated-wetness kernel shared by EDI
  and I_F, after [Byun & Wilhite (1999)][byun-1999]. Its fixed 365-day
  window yields NaN until a full window is available and for any window with
  a missing day. It measures flood potential, not flooding.
- **Effective Drought Index (EDI)** — *NumPy API available (FLOOD-09
  [#1106][flood-1106])*. A daily standardized index derived from effective
  precipitation ([Byun & Wilhite (1999)][byun-1999]); despite the name, it
  shares its kernel with the flood family. It implements the fixed-window form,
  not the paper's variable-duration extension
  ([ADR-0014](adr/0014-flood-family-scientific-conventions.md)); see
  {doc}`algorithm_refs/edi` for the algorithm and validation status.
- **Flood Index (I_F)** — *NumPy API available (FLOOD-10
  [#1107][flood-1107])*. A daily flood-potential index that standardizes effective precipitation
  ([Deo et al. (2015)][deo-2015]). The annual-maximum window follows the
  caller's year boundary, and the kernel question — the abstracts describe an
  exponential form while the implemented kernel is unverified — is recorded in
  [ADR-0014](adr/0014-flood-family-scientific-conventions.md); see
  {doc}`algorithm_refs/flood_index` for the algorithm and validation status.
- **Antecedent Precipitation Index (API)** — *NumPy API available (FLOOD-12
  [#1109][flood-1109])*. A daily recursive wetness measure in mm, after
  Kohler & Linsley (1951): `API_t = k * API_(t-1) + P_t`, where `0 < k < 1` and
  today's precipitation is added after decay. `flood.antecedent_precipitation_index(precipitation, k)`
  accepts daily mm input and defaults to a zero seed. Missing days propagate
  by default; the optional bridge policy and returned state support explicit
  gap handling and append runs. API shows flood potential, not flooding.
- **Generic standardization API** — *NumPy API available (FLOOD-16
  [#1113][flood-1113])*. `indices.standardized_index()` exposes the existing
  distribution-fitting machinery for any non-negative monthly or daily series,
  and {doc}`standardized-hydrologic-indices` shows the SRI and SSI recipes
  (FLOOD-17 [#1114][flood-1114]). Named wrappers are still an open decision
  ([#1132](https://github.com/monocongo/climate_indices/issues/1132)), and the
  xarray entry point also needs the CF metadata registry entries
  ([FLOOD-06 #1103][flood-1103]).

Heavy-precipitation triggers (Rx1day, Rx5day, and R95pTOT per
[Zhang et al. (2011)][zhang-2011]) are in scope, planned for validation against
the ETCCDI reference implementation `climdex.pcic` once its definitions and
percentile conventions are confirmed, with `xclim` an optional cross-check that
never becomes a dependency ([FLOOD-03 #1100][flood-1100]). The WAP/SWAP
wet-anomaly indices are deferred, and snowmelt-dependent indices such as SMRI
are out of scope, as is anything needing a calibrated hydrologic model,
terrain or land-cover data, or routed discharge
([ADR-0014](adr/0014-flood-family-scientific-conventions.md)). None of these
is part of this package yet.

## What these indices cannot tell you

- **Flood potential is not flooding.** Terrain, soils, land cover, and river
  routing determine whether accumulated wetness becomes a flood; these
  indices carry none of that information.
- **They are meteorological and climatological indices.** Flash-flood
  guidance, calibrated rainfall-runoff or routing models, terrain indices
  (TWI, HAND), inundation mapping and remote-sensing flood extent, and
  flood-frequency or return-period analysis of discharge records are outside
  the family's scope ([flood-family epic #1098][flood-epic]).
- **SPI and SPEI are less reliable in arid regions** with many
  zero-precipitation months ({doc}`algorithms` documents the limitation), so
  the wet tail there deserves more caution than in humid climates.
- **They are not yet externally validated.** PE has source-backed algebraic
  checks, not a reproducible numeric paper oracle. [VALIDATION.md][validation]
  has no flood section yet; [FLOOD-20 #1117][flood-1117] will record evidence
  per index. Treat this page as orientation, not evidence of validated skill.

## Sources

- [Byun & Wilhite (1999), *Objective Quantification of Drought Severity and Duration*][byun-1999]
- [Deo et al. (2015), *A Real-time Flood Monitoring Index Based on Daily Effective Precipitation*][deo-2015]
- Kohler & Linsley (1951), *Predicting the runoff from storm rainfall*, U.S. Weather Bureau Research Paper No. 34
- [Seiler, Hayes & Bressan (2002), *Using the Standardized Precipitation Index for Flood Risk Monitoring*][seiler-2002]
- [Zhang et al. (2011), *Indices for monitoring changes in extremes based on daily temperature and precipitation data*][zhang-2011]

[byun-1999]: https://doi.org/10.1175/1520-0442(1999)012%3C2747:OQODSA%3E2.0.CO;2
[deo-2015]: https://doi.org/10.1007/s11269-015-1046-3
[seiler-2002]: https://doi.org/10.1002/joc.799
[zhang-2011]: https://doi.org/10.1002/wcc.147
[validation]: https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md
[flood-epic]: https://github.com/monocongo/climate_indices/issues/1098
[flood-1103]: https://github.com/monocongo/climate_indices/issues/1103
[flood-1100]: https://github.com/monocongo/climate_indices/issues/1100
[flood-1105]: https://github.com/monocongo/climate_indices/issues/1105
[flood-1106]: https://github.com/monocongo/climate_indices/issues/1106
[flood-1107]: https://github.com/monocongo/climate_indices/issues/1107
[flood-1109]: https://github.com/monocongo/climate_indices/issues/1109
[flood-1113]: https://github.com/monocongo/climate_indices/issues/1113
[flood-1114]: https://github.com/monocongo/climate_indices/issues/1114
[flood-1117]: https://github.com/monocongo/climate_indices/issues/1117

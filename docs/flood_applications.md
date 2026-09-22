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
Accumulated / effective wetness      PE → EDI, I_F ; API                 (planned)
              ↓
Heavy-precipitation triggers         Rx1day / Rx5day / R95pTOT           (conditional)
              ↓
Hydrologic response (boundary)       SRI / SSI via generic standardization (planned)
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

## Planned wet-extreme indices

The flood family below is planned, not implemented. Module names, function
names, and arguments will be fixed before implementation in the flood design
decision ([FLOOD-02 #1099][flood-1099]); nothing on this page documents a
callable API. The [flood-family epic #1098][flood-epic] tracks the
implementation order.

- **Effective Precipitation (PE)** — *planned (FLOOD-08 [#1105][flood-1105])*.
  The daily accumulated-wetness kernel shared by EDI and I_F, after
  [Byun & Wilhite (1999)][byun-1999].
- **Effective Drought Index (EDI)** — *planned (FLOOD-09 [#1106][flood-1106])*.
  A daily standardized index derived from effective precipitation
  ([Byun & Wilhite (1999)][byun-1999]); despite the name, it shares its
  kernel with the flood family.
- **Flood Index (I_F)** — *planned (FLOOD-10 [#1107][flood-1107])*. A daily
  flood-potential index that standardizes effective precipitation
  ([Deo et al. (2015)][deo-2015]).
- **Antecedent Precipitation Index (API)** — *planned (FLOOD-12
  [#1109][flood-1109])*. A daily recursive wetness measure with an explicit
  decay constant.
- **Generic standardization API** — *NumPy API available (FLOOD-16
  [#1113][flood-1113])*. `indices.standardized_index()` exposes the existing
  distribution-fitting machinery for any non-negative monthly or daily series,
  so runoff or streamflow can be standardized as SRI or SSI with the wrappers
  planned in FLOOD-17 ([#1114][flood-1114]). The xarray entry point is not wired
  yet.

Heavy-precipitation triggers (Rx1day, Rx5day, and R95pTOT per
[Zhang et al. (2011)][zhang-2011]) are conditional: whether this package
should duplicate existing `xclim` coverage is an open scope decision
([FLOOD-03 #1100][flood-1100]), as are the WAP/SWAP wet-anomaly indices and
snowmelt-dependent indices such as SMRI. Until then, none of these is part of
this package.

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
- **They are not yet validated.** No flood index is implemented yet, so
  [VALIDATION.md][validation] has no flood section at present; the planned
  [FLOOD-20 #1117][flood-1117] adds one as implementations land. Until then,
  treat this page as orientation, not as evidence of validated skill.

## Sources

- [Byun & Wilhite (1999), *Objective Quantification of Drought Severity and Duration*][byun-1999]
- [Deo et al. (2015), *A Real-time Flood Monitoring Index Based on Daily Effective Precipitation*][deo-2015]
- [Seiler, Hayes & Bressan (2002), *Using the Standardized Precipitation Index for Flood Risk Monitoring*][seiler-2002]
- [Zhang et al. (2011), *Indices for monitoring changes in extremes based on daily temperature and precipitation data*][zhang-2011]

[byun-1999]: https://doi.org/10.1175/1520-0442(1999)012%3C2747:OQODSA%3E2.0.CO;2
[deo-2015]: https://doi.org/10.1007/s11269-015-1046-3
[seiler-2002]: https://doi.org/10.1002/joc.799
[zhang-2011]: https://doi.org/10.1002/wcc.147
[validation]: https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md
[flood-epic]: https://github.com/monocongo/climate_indices/issues/1098
[flood-1099]: https://github.com/monocongo/climate_indices/issues/1099
[flood-1100]: https://github.com/monocongo/climate_indices/issues/1100
[flood-1105]: https://github.com/monocongo/climate_indices/issues/1105
[flood-1106]: https://github.com/monocongo/climate_indices/issues/1106
[flood-1107]: https://github.com/monocongo/climate_indices/issues/1107
[flood-1109]: https://github.com/monocongo/climate_indices/issues/1109
[flood-1113]: https://github.com/monocongo/climate_indices/issues/1113
[flood-1114]: https://github.com/monocongo/climate_indices/issues/1114
[flood-1117]: https://github.com/monocongo/climate_indices/issues/1117

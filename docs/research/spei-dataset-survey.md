# Authoritative SPEI Reference Dataset Survey

Resolves: [#771](https://github.com/monocongo/climate_indices/issues/771) (child of the
[external-dataset-validation wayfinder map, #769](https://github.com/monocongo/climate_indices/issues/769))

## Question

Which publicly accessible, authoritative SPEI reference dataset should climate_indices
validate its gamma- and Pearson-III-fitted SPEI output against? The Spanish SPEIbase
(Vicente-Serrano et al., `spei.csic.es` / `digital.csic.es`) is the leading global
gridded candidate — survey it and any realistic alternatives, check
accessibility/licensing, resolve how a global grid maps onto this map's CONUS-default
domain, and recommend one primary acquisition target plus tolerance expectations.

## What climate_indices computes

`climate_indices.indices.spei()` fits precipitation-minus-PET water balance to a gamma
or Pearson Type III distribution per calendar period, using PET supplied by (or
computed via) `eto.eto_thornthwaite` or `eto.eto_hargreaves` — both temperature-only /
temperature-range empirical PET methods. There is no Penman-Monteith PET path in this
library. This is the same PET-method boundary that made the sibling ticket
[#775](https://github.com/monocongo/climate_indices/issues/775) (gridded reference-ET
survey) reject gridMET's ASCE-PM ETo as a numerical validation gate — and it recurs
here, because SPEI's PET input methodology directly shapes the water-balance series
being standardized.

## Candidates investigated

### 1. SPEIbase (Vicente-Serrano, Beguería, et al. — CSIC), current v2.11

- **What it is**: The canonical global gridded SPEI product, the dataset the user
  identified as "the Spanish SPEI." Actively maintained by the same research group that
  defined the SPEI methodology (Vicente-Serrano et al. 2010).
- **Accessibility**: Direct, no-registration bulk NetCDF download from
  `https://spei.csic.es/spei_database_2_11` (and the `spei.csic.es/database.html`
  landing page), plus a mirrored copy on Google Earth Engine
  (`ee.ImageCollection("CSIC/SPEI/2_11")`, free GEE account required — no special
  approval). `digital.csic.es` also hosts archival dataset records with DOIs, though
  the institutional repository page itself returned a bot-protection block during this
  survey (Anubis anti-scraping challenge) — the primary `spei.csic.es` download path is
  unaffected and is the one to use.
- **License**: CC-BY 4.0 (per the Earth Engine catalog entry) / Open Database License
  on the main site — attribution required, redistribution of a derived subset (e.g. a
  committed fixture) is permitted the same way NOAA EDDI and nClimDiv fixtures are
  handled today. Required citation: Beguería, Vicente-Serrano, Reig-Gracia, Latorre
  Garcés (2024), *SPEIbase v.2.10/2.11 [Dataset]*, DIGITAL.CSIC,
  DOI 10.20350/digitalCSIC/16497, plus the methodology papers (Vicente-Serrano et al.
  2010; Beguería et al. 2010, 2014).
- **Resolution/coverage**: Global, 0.5° grid, monthly, January 1901 – current
  (v2.11 extends through late 2024/early 2025; the dataset updates roughly annually).
  Ships all standard SPEI timescales as separate bands/files: SPEI-1 through SPEI-48
  months (i.e., SPEI-1/3/6/12 are all present).
- **Data source**: Derived from CRU TS (currently CRU TS 4.09) monthly precipitation
  and PET fields.
- **PET method — the compatibility problem**: SPEIbase v1.0 (2010) used Thornthwaite
  PET. **Every version from v2.0 onward — including the current v2.11 — uses FAO-56
  Penman-Monteith PET** (an energy-balance method requiring radiation, humidity, and
  wind, computed by CRU as part of CRU TS). The project's own documentation states this
  was a deliberate move away from Thornthwaite because Penman-Monteith is considered
  more physically robust. This is the same PET-formula family (Penman-Monteith)
  that PR #775 found has a known, non-constant, climate-dependent bias against
  climate_indices' Thornthwaite/Hargreaves output — so a v2.11 comparison inherits that
  same confound, now propagated through the standardization step rather than raw PET.

### 2. SPEI Global Drought Monitor (same CSIC group, `spei.csic.es/map/`)

- **What it is**: A companion near-real-time monitoring product from the same group,
  distinct from SPEIbase proper. Included here because it resolves the PET-method
  mismatch that SPEIbase v2.11 has.
- **Accessibility**: Interactive map plus bulk NetCDF download of the full time series;
  only the most recent ~4 weeks require a free login, so the full historical record
  needed for validation is openly downloadable.
- **License**: Open Database License (ODbL 1.0), attribution + share-alike, same family
  of terms as SPEIbase.
- **Resolution/coverage**: Global, coarser 1° grid, monthly, calibration period
  January 1950 – December 2010 (data begins ~1955), same SPEI-1…48 timescale set.
- **Data source**: NOAA NCEP CPC GHCN_CAMS gridded temperature + GPCC "first guess"
  monthly precipitation — different input fields from CRU TS, so a comparison still
  requires acquiring these specific inputs to reproduce matched conditions.
- **PET method**: **Explicitly Thornthwaite** — the project's own documentation states
  this choice was made "due to the lack of real-time data sources for computing more
  robust PET estimations," i.e., real-time operational necessity, not a scientific
  preference for Thornthwaite. This is the same formula family
  (`eto.eto_thornthwaite`) climate_indices implements, which removes the PET-method
  confound that affects SPEIbase v2.11.
- **Caveat**: Coarser resolution (1° vs 0.5°), a fixed and dated calibration window
  (1950–2010) that will not match an arbitrary user-chosen calibration period, and it
  is positioned by its own authors as an operational monitoring tool rather than the
  citable, peer-reviewed climatological reference product that SPEIbase is. It is
  useful as a PET-method-matched secondary check, not as the primary "authoritative
  reference" the map's destination (AMS-citability) is aiming for.

### 3. Other candidates considered and ruled out

- **TerraClimate** (global, 1958–present, ~4 km, monthly water-balance dataset)
  ships PET (Penman-Monteith derived) and precipitation but does **not** publish a
  precomputed SPEI product — using it would mean computing SPEI in-house from its
  fields, which validates nothing external; it would just be climate_indices compared
  against itself using different weather inputs. Not pursued.
- No other actively maintained, publicly documented, peer-reviewed **global gridded
  SPEI** product (precomputed, not just PET/precip components) turned up in this
  search. SPEIbase and its Global Drought Monitor sibling are effectively the field's
  standard; regional SPEI products exist but are not more authoritative than SPEIbase
  for a CONUS-relevant comparison and would not resolve the PET-method question any
  differently.

## CONUS sampling recommendation

Neither SPEIbase (0.5°) nor the Drought Monitor (1°) aligns to NWS climate division
polygons. Follow the same pattern already used for the gridMET PET survey (#775) and
consistent with the map's Notes ("use each dataset's native domain where CONUS doesn't
fit"):

1. Select a small set of CONUS climate divisions spanning an aridity gradient (e.g. one
   humid-East division, one semi-arid Great Plains division, one arid Southwest
   division) rather than attempting full CONUS coverage — this keeps the PET-method
   divergence (which is itself climate-dependent) visible instead of averaged away.
2. Areal-average the grid cells (0.5° for SPEIbase, 1° for the Drought Monitor) falling
   within each division polygon — the same style of aggregation nClimDiv itself
   performs from station data, and the same preprocessing step already identified as
   necessary for gridMET.
3. Compare per-division, per-timescale (SPEI-1/3/6/12) series rather than a single
   pooled statistic, so a regional PET-method bias doesn't get diluted by averaging with
   a region where it's small.

## Recommendation

**Acquire SPEIbase v2.11 as the primary reference target**, since it is the
authoritative, peer-reviewed, most-cited product and the one the map/user specifically
named — but **do not treat it as a numerical validation gate the way NOAA EDDI or
nClimDiv Palmer fixtures are used**. Its FAO-56 Penman-Monteith PET is a materially
different formula family from climate_indices' Thornthwaite/Hargreaves-only
implementation, carrying the same documented, climate-dependent, non-constant bias that
PR #775 found makes tight `atol`/`rtol` acceptance tests indefensible for PET
comparisons. For SPEI specifically, van der Schrier et al. (2011, JGR) is directly on
point: SPEI computed with Thornthwaite vs. Penman-Monteith PET is reported broadly
similar in humid climates but diverges materially in arid/semi-arid regions — i.e.
exactly the CONUS Southwest division that a sampling plan should include, and exactly
where a strict tolerance would misfire in either direction.

**Secondary/complementary acquisition**: the SPEI Global Drought Monitor product, which
uses Thornthwaite PET — the same formula family climate_indices implements. This is
the more PET-method-compatible candidate for anything approaching a tighter numerical
comparison, provided a future ticket also acquires its actual input fields (NOAA
GHCN_CAMS temperature, GPCC "first-guess" precipitation) to run climate_indices'
Thornthwaite path on matched inputs rather than comparing against SPEIbase's CRU-TS-derived,
Penman-Monteith-based series. Frame this as the PET-method-controlled check; frame
SPEIbase v2.11 as the primary, most-authoritative-but-PET-mismatched reference.

## Tolerance caveats for a future acquisition-and-test-writing ticket

1. **SPEIbase v2.11 vs. climate_indices (Thornthwaite or Hargreaves) SPEI**: do not
   adopt an `atol`/`rtol` gate analogous to EDDI (`1e-5`) or Palmer (`5e-5`). Frame any
   resulting test as a plausibility/agreement check — e.g. sign agreement, drought
   category agreement, or correlation coefficient per climate-division/timescale — and
   label it in `VALIDATION.md` explicitly as a plausibility check, not "independently
   validated," mirroring the language discipline the PET survey (#775) recommended.
   Expect closer agreement in humid CONUS divisions and larger, systematic divergence
   in arid/semi-arid divisions, consistent with van der Schrier et al. (2011).
2. **SPEI Global Drought Monitor vs. climate_indices Thornthwaite SPEI, with matched
   GHCN_CAMS/GPCC inputs**: a moderate numerical tolerance is plausibly achievable
   since the PET formula matches, but it is not automatically as tight as EDDI's
   `1e-5` — the Drought Monitor's fixed 1950–2010 calibration window, its own gamma/
   distribution-fitting implementation details (e.g. L-moments vs. MLE parameter
   estimation), and any residual precipitation/temperature preprocessing differences
   all remain unverified confounds. A future ticket should empirically characterize
   achievable agreement (start loose, e.g. `rtol` on the order of 1e-2, and tighten only
   after confirming the fitting methodology matches) rather than assuming EDDI-level
   precision is attainable.
3. Neither candidate uses Hargreaves PET, so climate_indices' Hargreaves-based SPEI
   path has no PET-method-matched external reference from either dataset; any
   comparison there inherits the full PET-formula-mismatch caveat regardless of which
   SPEI product is used.
4. Whichever dataset is acquired, follow `tests/fixture/provenance_schema.json` for
   provenance (source, URL, download date, subset description, checksum, fixture
   version, and an explicit `validation_tolerance` object) — SPEIbase's DOI and
   required citation are given above; the Drought Monitor's citation should be
   confirmed at acquisition time from its current `spei.csic.es/map/` documentation.

## Sources

- [SPEI database — spei.csic.es](https://spei.csic.es/database.html)
- [SPEIbase v2.11 direct download](https://spei.csic.es/spei_database_2_11)
- [SPEIbase v2.11 — Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/CSIC_SPEI_2_11)
- [SPEIbase v2.10 — Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/CSIC_SPEI_2_10)
- [SPEI Global Drought Monitor — spei.csic.es/map](https://spei.csic.es/map/)
- [sbegueria/SPEIbase — R code and README, GitHub](https://github.com/sbegueria/SPEIbase/blob/master/README.md)
- [sbegueria/SPEI — R package for SPEI/PET computation, GitHub](https://github.com/sbegueria/SPEI)
- [SPEIbase: a global 0.5° gridded SPEI data base (NetCDF) — DIGITAL.CSIC](http://digital.csic.es/handle/10261/22449) (institutional record; page blocked an automated fetch during this survey via anti-scraping protection — use the `spei.csic.es` download path instead)
- [Standardized Precipitation Evapotranspiration Index (SPEI) — NCAR Climate Data Guide](https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-evapotranspiration-index-spei)
- van der Schrier, G. et al. (2011). The sensitivity of the PDSI to the Thornthwaite
  and Penman-Monteith parameterizations for potential evapotranspiration. *JGR
  Atmospheres*. https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2010JD015001
- Vicente-Serrano, S. M., Beguería, S., López-Moreno, J. I. (2010). A Multiscalar
  Drought Index Sensitive to Global Warming: The Standardized Precipitation
  Evapotranspiration Index. *Journal of Climate*, 23(7), 1696–1718.
- Beguería, S., Vicente-Serrano, S. M., Reig, F., Latorre, B. (2014). Standardized
  precipitation evapotranspiration index (SPEI) revisited: parameter fitting,
  evapotranspiration models, tools, datasets and drought monitoring. *International
  Journal of Climatology*, 34(10), 3001–3023.
- Prior finding referenced: [`docs/research/pet-gridded-dataset-survey.md`](https://github.com/monocongo/climate_indices/blob/research/pet-gridded-dataset-survey/docs/research/pet-gridded-dataset-survey.md)
  on branch `research/pet-gridded-dataset-survey` (resolves #775) — established the
  Penman-Monteith-vs-Thornthwaite/Hargreaves bias problem this survey applies to SPEI.

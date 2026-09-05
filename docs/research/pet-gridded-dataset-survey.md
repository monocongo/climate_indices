# Gridded Reference-ET Dataset Survey (PET Stretch Validation)

Resolves: [#775](https://github.com/monocongo/climate_indices/issues/775) (child of the
[external-dataset-validation wayfinder map, #769](https://github.com/monocongo/climate_indices/issues/769))

## Question

As a stretch goal beyond the literature worked-example fixtures being extracted in the
sibling ticket [#774](https://github.com/monocongo/climate_indices/issues/774)
(Thornthwaite 1948, Hargreaves & Samani 1985), identify a gridded reference-ET dataset
suitable for comparing climate_indices' `eto.eto_thornthwaite` and `eto.eto_hargreaves`
output over a CONUS sample, and recommend whether acquiring it is worth the effort.

## Candidates investigated

### 1. gridMET (University of Idaho Climatology Lab)

- **Accessibility**: Fully public, no registration or API key. Direct NetCDF file
  download and wget scripts, THREDDS/OPeNDAP catalog
  (`https://thredds.northwestknowledge.net/`), USGS Geo Data Portal (Zarr), and Google
  Earth Engine (`IDAHO_EPSCOR/GRIDMET`). Google Earth Engine access requires a free
  Google account but no special approval.
- **License**: CC0 public domain dedication — "John Abatzoglou has waived all copyright
  and related or neighboring rights to gridMET." No redistribution restriction; a
  derived subset can be committed to the repo as a fixture the same way the NOAA EDDI
  and nClimDiv fixtures are.
- **Resolution**: ~4 km (1/24°) daily grid, CONUS + southern British Columbia,
  1979–present (last ~60 days are provisional).
- **Formula**: Ships a derived variable literally named "Reference evapotranspiration
  (ASCE Penman-Monteith)" — i.e. the ASCE-EWRI standardized Penman-Monteith equation
  (short-crop reference, ETo), computed from gridMET's own temperature, humidity, wind,
  and radiation fields (itself a PRISM/NLDAS blend). Not Thornthwaite, not Hargreaves.
- **CONUS climate-division mapping**: The 4 km grid does not align with NWS climate
  division polygons; a comparison would require areal-averaging grid cells within each
  division boundary (the same kind of aggregation nClimDiv itself already does from
  station data), which is a straightforward but nontrivial preprocessing step, not a
  simple lookup.

### 2. ASCE-EWRI standardized reference ET — other gridded implementations

- **NOAA ETo product (NLDAS-driven)**: Some NOAA/regional products (e.g. via
  ClimateEngine.org) compute ASCE standardized Penman-Monteith ET at 0.125° from NLDAS-2
  forcing, CONUS-wide, 1979–present. Same underlying formula as gridMET's derived
  variable, coarser resolution, and typically fronted by a web tool
  (ClimateEngine.org) rather than a plain bulk-download endpoint — some workflows there
  require a free account. Adds no new formula independence over gridMET and is a worse
  access story, so it does not beat gridMET as a candidate.
- **AgriMet / CIMIS / regional ET networks**: Station-based (CIMIS is California-only,
  AgriMet is Pacific Northwest), not CONUS-gridded, and CIMIS in particular has
  registration/use-agreement friction. Not suitable as a CONUS default-domain
  candidate.
- No other widely-used, publicly-accessible, no-auth, CONUS-gridded reference-ET
  dataset that uses a materially different formula (e.g. Priestley-Taylor,
  pan-evaporation-based) turned up in this search. Every actively maintained gridded
  reference-ET product for CONUS found here is a Penman-Monteith variant, because that
  is what the ASCE-EWRI standard specifies for "reference ET."

## The core problem: formula mismatch, not implementation error

climate_indices' `eto_thornthwaite` and `eto_hargreaves` are temperature-based
(Thornthwaite) and temperature-range/extraterrestrial-radiation-based (Hargreaves-
Samani) empirical PET estimators. gridMET's reference ET — and every other gridded
CONUS product surveyed — is ASCE-EWRI standardized **Penman-Monteith**, an
energy-balance/aerodynamic method requiring radiation, humidity, and wind in addition
to temperature.

Published comparisons (Carpathian region ASR 2020; China Penman-Monteith/Thornthwaite
comparison; global drylands sensitivity study, Adv. Atmos. Sci. 2018; Hargreaves-Samani
vs. ASCE-PM calibration studies across U.S. High Plains and multiple aridity classes)
consistently show:

- Hargreaves-Samani systematically **underestimates** ETo relative to ASCE-PM in
  arid/semi-arid climates (mean bias on the order of −0.3 to −1.7 mm/day reported across
  aridity classes) and can mildly **overestimate** in humid climates (+0.1–0.2 mm/day).
- Thornthwaite-based PET diverges from Penman-Monteith-based PET in a
  climate-dependent, non-constant way; SPEI computed with Thornthwaite vs.
  Penman-Monteith PET is reported to be broadly similar in the CONUS humid East but
  diverges materially in the arid Southwest, especially at short accumulation scales
  (van der Schrier et al. 2011, JGR; AMS 2013 abstract on SPEI Thornthwaite vs.
  Penman-Monteith parameterizations).
- These biases are driven by which meteorological drivers each formula ignores
  (Thornthwaite ignores radiation/humidity/wind entirely; Hargreaves ignores humidity
  and wind), not by any defect in a correct implementation of either formula.

This means a numerical difference between climate_indices' output and a gridMET ETo
comparison is expected and well-documented **even for a bug-free implementation**, and
the expected divergence varies by region (small in humid temperate CONUS, large in the
arid Southwest) and by season. There is no single defensible numerical tolerance — any
`atol`/`rtol` tight enough to catch a real implementation bug would also fail on
correct Thornthwaite/Hargreaves output in the wrong climate zone, and any tolerance
loose enough to pass everywhere would not catch a real regression.

## Recommendation

**Not worth pursuing as a numerical validation gate; worth a documented plausibility
check at most, and only if effort is available after the literature fixtures land.**

Rationale:

- The literature worked-example fixtures from #774 (Thornthwaite 1948, Hargreaves &
  Samani 1985) validate that climate_indices reproduces the *specific formulas as
  published*, against inputs/outputs where the source authors intended agreement to
  near-exact precision. That is genuine independent validation of implementation
  correctness.
- A gridMET comparison, by contrast, is validating implementation-correctness against
  a dataset generated by a *different, more data-hungry method*. Any resulting
  agreement/disagreement is dominated by the known Thornthwaite/Hargreaves-vs-Penman-
  Monteith formula gap described above, not by whether climate_indices' code is right.
  It cannot function as a pass/fail acceptance test the way the NOAA EDDI or nClimDiv
  Palmer fixtures do.
- It could still have narrow value as a **sanity/plausibility check** — e.g. "is
  climate_indices' Thornthwaite output within the published Hargreaves-vs-PM bias
  envelope for a given climate division, in the right direction and rough magnitude" —
  which would catch gross unit errors, sign errors, or order-of-magnitude bugs that
  the literature fixtures (single worked examples) might not exercise across a full
  CONUS seasonal/spatial range. If pursued, it should be framed and labeled in
  `VALIDATION.md` explicitly as a **plausibility check**, not as "Validated" or
  "Independently validated" status — using language distinct from the NOAA EDDI /
  nClimDiv Palmer independent-validation entries, to avoid overstating rigor to a
  future reader (including a future AMS-paper audience, per the map's destination).
- If pursued, gridMET is the clear best candidate if effort is spent at all: CC0
  license (no redistribution barrier, unlike the coarser NLDAS-based alternatives
  which add access friction without a different formula), no-auth direct download,
  daily resolution, full CONUS coverage 1979–present, and an explicit named ETo
  variable already computed with a well-documented formula (ASCE-EWRI PM) — so the
  provenance file is easy to fill out per `tests/fixture/provenance_schema.json`.
  Budget for: (a) selecting a small sample of climate divisions across an
  aridity gradient (e.g. one humid-East, one semi-arid Great Plains, one arid
  Southwest division) so the known bias-by-aridity pattern is visible rather than
  averaged away, (b) areal-averaging the 4 km grid into division polygons, (c)
  documenting expected bias direction/magnitude per division from the cited literature
  as the "tolerance" (there is no single tolerance number — the acceptance criterion
  would be "within the published bias envelope for that climate/method pair," which is
  a qualitative/order-of-magnitude check, not a numerical `atol`).

## Tolerance/framing caveat for any future ticket

Do not adopt an `atol`/`rtol` acceptance test against gridMET ETo the way EDDI/Palmer
fixtures use `atol=1e-5`/`5e-5`. Given the formula mismatch documented above, any
future ticket that acts on this survey should:

1. Label the resulting test/fixture as a plausibility or bias-envelope check, not
   "independent validation," in `VALIDATION.md`.
2. State the acceptance criterion as a qualitative match to published
   Thornthwaite/Hargreaves-vs-Penman-Monteith bias direction and rough magnitude per
   climate class, not a fixed numerical tolerance.
3. Keep it clearly secondary to, and not a substitute for, the #774 literature
   worked-example fixtures, which remain the actual independent-validation evidence
   for PET in `VALIDATION.md`.

## Sources

- [gridMET — Climatology Lab](https://www.climatologylab.org/gridmet.html)
- [GRIDMET on Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET)
- [Reference Evapotranspiration — ClimateEngine.org](https://climateengine.org/datasets/evapotranspiration/reference-evapotranspiration/)
- [The ASCE Standardized Reference Evapotranspiration Equation (ASCE-EWRI, 2005)](https://www.mesonet.org/images/site/ASCE_Evapotranspiration_Formula.pdf)
- van der Schrier, G. et al. (2011). The sensitivity of the PDSI to the Thornthwaite
  and Penman-Monteith parameterizations for potential evapotranspiration. *JGR
  Atmospheres*. https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2010JD015001
- Computation of daily Penman-Monteith reference evapotranspiration in the Carpathian
  Region and comparison with Thornthwaite estimates, *Adv. Sci. Res.*, 2020.
  https://asr.copernicus.org/articles/16/251/2020/
- Sensitivity of potential evapotranspiration estimation to the Thornthwaite and
  Penman-Monteith methods in the study of global drylands, *Adv. Atmos. Sci.*, 2018.
  https://link.springer.com/article/10.1007/s00376-017-6313-1
- Spatio-temporal calibration of Hargreaves-Samani model to estimate reference
  evapotranspiration across U.S. High Plains, *Agronomy Journal*.
  https://acsess.onlinelibrary.wiley.com/doi/abs/10.1002/agj2.20325

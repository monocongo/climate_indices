# Authoritative SPI Reference Dataset Survey

Resolves: [#770](https://github.com/monocongo/climate_indices/issues/770) (child of the
[external-dataset-validation wayfinder map, #769](https://github.com/monocongo/climate_indices/issues/769))

## Question

Which publicly accessible, authoritative SPI reference dataset should climate_indices
validate its gamma- and Pearson-III-fitted SPI output against? Candidates considered:
NOAA CPC's/NCEI's monthly SPI product, WMO SPI technical guidance worked examples, and
other widely cited authoritative sources.

## What climate_indices computes

`climate_indices.indices.spi()` fits monthly (or daily) precipitation accumulations to a
per-calendar-period distribution — either two-parameter **gamma** or **Pearson Type
III** (`indices.Distribution.gamma` / `.pearson3` in `src/climate_indices/indices.py`),
with a `DistributionFallbackStrategy` in `compute.py` that falls back from Pearson III to
gamma when the fit fails (e.g. high zero-precipitation rates). Both fitting paths are
first-class, so a reference dataset that documents which distribution it used is
directly comparable rather than requiring a re-derivation.

## Candidates investigated

### 1. NOAA/NCEI nClimDiv Climate Divisional SPI (`climdiv-sp{01,02,03,06,09,12,24}dv`) — recommended primary

- **What it is**: NCEI's operational monthly SPI product for the 344 CONUS climate
  divisions, part of the legacy Climate Divisional Database (`cirs/climdiv`
  directory), the same data family already used for this repo's committed Palmer
  fixtures (`tests/fixture/palmer/`).
- **Accessibility**: Direct, no-auth bulk download from
  `https://www.ncei.noaa.gov/pub/data/cirs/climdiv/` — plain-text fixed-width files,
  versioned by date (current listing showed an August 2026 refresh, i.e. still actively
  maintained). No account, API key, or request process required.
- **License**: U.S. government work, public domain, free redistribution with NOAA's
  standard no-warranty disclaimer ("users assume responsibility to determine
  usability"). This matches the same NOAA/NCEI terms already documented for nClimGrid in
  `docs/research/nclimgrid-acquisition-and-redistribution.md` — safe for a committed,
  citable fixture.
- **Coverage/domain fit**: Exact match for the map's default CONUS-climate-divisions
  domain, and literally the same spatial units already used for the Palmer/scPDSI
  nClimDiv fixtures — strong cross-index comparability.
- **Computation parameters** (from `drought-readme.txt` at the above URL, dated March
  2014 but still the live documentation for the current files):
  - Timescales: 1, 2, 3, 6, 9, 12, and 24 months (element codes 71–77).
  - Distribution: **Pearson Type III** — corroborated by independent literature
    (Guttman's recommendation for U.S. climate-division data, adopted operationally by
    NCEI) rather than gamma. This lines up with climate_indices' `pearson3` path, not
    its `gamma` path, as the primary comparison target.
  - Stated calibration period: **1931–1990**, fixed, per the README ("All drought data
    are calibrated using the period 1931-1990"), applying to the SPI files alongside
    PDSI/PHDI/PMDI/Z-Index in the same file family.
  - Precision/format: fixed-width `f7.2` (two decimal places), values clipped to the
    range **-4.00 to +4.00**, missing values coded as **-99.99**.
- **Independence caveat — the most important finding of this survey**: this repository
  (`monocongo/climate_indices`) is explicitly named by NOAA/NIDIS as the software behind
  a *different* NCEI drought product — the gridded nClimGrid-monthly SPI/SPEI/PET
  (drought.gov's "Source Code: Climate and Drought Indices in Python (SPI, SPEI, PET)"
  page states the tool is "a developmental/forked version of code that was originally
  developed by [NIDIS] and NOAA's [NCEI]," maintained by this repo's own author). Using
  that gridded nClimGrid-monthly SPI product as a validation "ground truth" would be
  **circular** — it would not test independent correctness, only self-consistency with
  a sibling/ancestor codebase — so it is explicitly **not** recommended as a numerical
  reference (see Candidate 2 below).
  The climate-divisional `climdiv-sp*dv` product, by contrast, is part of NCEI's older
  Climate Divisional Database lineage (README dated 2014, predating this Python
  package's 2017 BSD-license copyright) and is plausibly computed by a separate,
  older Fortran-derived codebase (drought.gov also separately lists "Source Code:
  Drought Indices in Fortran (SPI, PDSI)" as distinct public-release code). This could
  not be fully confirmed from public documentation alone — NOAA does not publish an
  explicit statement that `climdiv-sp*dv` and the Python `climate_indices` package are
  unrelated — so a future acquisition ticket should record this ambiguity explicitly in
  the fixture's `provenance.json` `notes` field rather than claim airtight independence.
- **A documented discrepancy relevant to tolerance-setting**: a peer-reviewed critique,
  Baldwin & Chen (2020, *Journal of Applied Meteorology and Climatology*, "Major Over-
  and Underestimation of Drought Found in NOAA's Climate Divisional SPI Dataset"),
  reports that NCEI's actual production behavior does **not** match the README's stated
  fixed 1931–1990 calibration window — it instead uses a full/expanding period-of-record
  window (approximately 1895 through the latest available year) that is effectively
  recalibrated as new months are added, producing regionally varying bias versus a
  fixed-baseline SPI. This means the README's own documentation cannot be taken at face
  value for reproducing exact values; an acquisition-and-test-writing ticket must
  empirically determine (by reproducing a few divisions both ways) which calibration
  behavior the current files actually reflect before locking any tight tolerance.

### 2. NOAA/NIDIS/NCEI nClimGrid-Monthly Gridded SPI — investigated, not recommended as a numerical gate

- **What it is**: A gridded (not climate-division) SPI/SPEI/PET product, CONUS coverage,
  base period 1895–2014, gamma and Pearson III distributions, timescales 1–72 months,
  freely downloadable NetCDF-4 from `https://www.ncei.noaa.gov/pub/data/nidis/indices/nclimgrid-monthly/`,
  no auth required.
- **Why not recommended**: drought.gov's own documentation states this product is
  computed with "the Python Drought Indices open-source package" and separately
  identifies that package as this repository's own codebase family (see Candidate 1's
  independence caveat). Comparing climate_indices against a product it (or a very close
  relative) produced would not constitute independent validation. It remains useful as
  a secondary sanity/regression check (e.g., "does current climate_indices output still
  match what NOAA currently publishes using this codebase's own lineage"), but should
  not be described as external validation in `VALIDATION.md`.

### 3. WMO Standardized Precipitation Index User Guide (WMO-No. 1090, 2012) — recommended secondary/complementary

- **What it is**: The authoritative technical guidance document for SPI, published by
  the World Meteorological Organization (Svoboda, Hayes, Wood, 2012), used globally as
  the reference methodology description. Includes worked numerical examples of the
  gamma-fitting and standardization procedure.
- **Accessibility**: Freely downloadable PDF, no auth, from
  `https://www.droughtmanagement.info/literature/WMO_standardized_precipitation_index_user_guide_en_2012.pdf`
  (mirrored by the WMO e-Library at `library.wmo.int`). This survey downloaded the PDF
  successfully; full extraction of the specific worked-example table requires a PDF
  text/render toolchain not available in this environment (`poppler-utils` was not
  installed), so a follow-up acquisition ticket should re-fetch and extract the table
  directly rather than rely on this survey's characterization of its exact contents.
- **License**: WMO technical guidance is published for open dissemination to support
  national drought-monitoring capacity building; a future ticket should still confirm
  WMO's specific redistribution terms before committing a verbatim excerpt, and prefer
  citing/reconstructing the worked example's inputs and outputs (numbers derived from
  the guide) over redistributing PDF text if licensing is ambiguous.
- **Coverage/domain fit**: The worked examples use illustrative/generic station data,
  not CONUS climate divisions. This is an acceptable exception under the map's Notes
  ("use each dataset's native domain where CONUS doesn't fit") — this candidate's value
  is as a distribution-fitting-algorithm-level check, the same role Thornthwaite (1948)
  and Hargreaves & Samani (1985) worked examples play for PET validation in this same
  map (per #769's Notes and the existing `docs/algorithm_refs/` pattern).
- **Computation parameters**: gamma distribution fitting per McKee et al. (1993)
  methodology; specific calibration period and timescale in the worked example need
  confirmation at extraction time (see accessibility note above).
- **Tolerance expectation**: potentially tight (`atol` on the order of 1e-3 to 1e-5,
  similar to the existing internal legacy fixtures) since a literature worked example
  typically shows full-precision intermediate values, unlike NCEI's 2-decimal
  climate-divisional output — but this must be confirmed once the actual table is
  extracted.

### 4. Other candidates considered and ruled out

- **WestWide Drought Tracker (WWDT, DRI/WRCC)**: gridded (PRISM-derived), gamma
  distribution, Western US domain only (does not naturally cover full CONUS climate
  divisions) — the map's Notes already anticipate a Western-US-only fit for DRI/WRCC
  products in the context of PDSI supplementation, not SPI. Since NCEI's own
  climate-divisional CONUS product (Candidate 1) is available and better matches the
  domain, WWDT was not pursued further for SPI.
- **NDMC SPI Program (Guttman/McKee original implementation)**: a standalone
  legacy calculation utility distributed by the National Drought Mitigation Center, not
  itself a fixed reference *dataset* with published output values — not a fit for a
  "download reference numbers and compare" fixture pattern.

## Recommendation

**Primary: acquire NOAA/NCEI nClimDiv climate-divisional SPI**
(`climdiv-sp01dv` through `climdiv-sp24dv`, current version, from
`https://www.ncei.noaa.gov/pub/data/cirs/climdiv/`) as the primary external reference,
compared against climate_indices' **Pearson Type III** SPI output (matching NCEI's
documented distribution choice), computed from matching climate-divisional
precipitation input (`climdiv-pcpndv`, same directory) over the same domain already
used for the Palmer fixtures. This maximizes cross-index domain consistency (all of
Palmer, scPDSI, and now SPI validated against the same 344 CONUS climate divisions) and
is unambiguously public-domain, no-auth, freely redistributable.

**Secondary/complementary: extract a WMO SPI User Guide (WMO-No. 1090) worked example**
as a literature-fixture, following the same pattern already used for PET's Thornthwaite
(1948) / Hargreaves (1985) worked examples in `docs/algorithm_refs/`. This provides an
algorithm-level check on the gamma-fitting path (which the CONUS climate-division
product does not exercise, since NCEI's operational product uses Pearson III) and is
untainted by any codebase-lineage concerns, unlike the NCEI-produced candidates.

**Explicitly not recommended for a numerical validation gate**: the NCEI nClimGrid-
monthly gridded SPI product, due to the circularity finding above (drought.gov
documents that product as produced using this repository's own codebase lineage).

## Tolerance and calibration caveats for a future acquisition-and-test-writing ticket

1. **Distribution mismatch risk**: nClimDiv's documented distribution is Pearson III,
   not gamma. Compare against climate_indices' `pearson3` path; do not expect a gamma-
   fitted comparison to agree, and do not treat a gamma/nClimDiv mismatch as a bug.
2. **Calibration-period ambiguity is the central open question.** The README states a
   fixed 1931–1990 window; a peer-reviewed critique (Baldwin & Chen 2020, JAMC) reports
   the actual production behavior uses a full/expanding period-of-record window instead.
   A future ticket must empirically test both interpretations (compute climate_indices
   Pearson III SPI with each calibration window and see which one the current NCEI
   files actually match) before committing to any `atol`/`rtol`. Do not assume the
   README is authoritative for reproduction purposes — verify empirically first.
3. **Quantization floor**: NCEI's published values are rounded to 2 decimal places
   (`f7.2` format) and hard-clipped to [-4.00, +4.00]. Any tolerance tighter than
   roughly `atol=0.005` (half the last reported digit) is not meaningful regardless of
   how exactly the calibration window is matched; expect somewhat looser agreement
   (e.g. `atol` on the order of 0.01–0.05) once the calibration-window question above is
   resolved, not EDDI-level (`1e-5`) or internal-fixture-level (`1e-8`) precision.
4. **Missing-value sentinel**: `-99.99` in the raw files must be masked before any
   numerical comparison, matching the pattern already used for other external fixtures.
5. **Independence/circularity**: record in the fixture's `provenance.json` `notes`
   field that (a) the nClimDiv climate-divisional product's exact codebase lineage
   relative to this repository could not be confirmed as fully independent from public
   documentation alone (though it plausibly predates and differs from the Python
   package family that produces nClimGrid-monthly), and (b) the nClimGrid-monthly
   gridded SPI product was deliberately excluded from this survey's recommendation for
   that reason. This keeps the eventual `VALIDATION.md` entry honest about the strength
   of the independence claim, in the same spirit as the existing Palmer entry's
   disclosure that its fixtures are "climate_indices library reference output."
6. **WMO worked example extraction is incomplete**: this survey could not render the
   WMO PDF's specific worked-example table (missing `poppler-utils` in this
   environment) — a follow-up ticket must re-extract it directly and confirm its
   calibration period, timescale, and precision before treating it as usable fixture
   data.
7. Follow `tests/fixture/provenance_schema.json` for whichever dataset(s) are acquired
   (source, URL, download date, subset description, checksum, fixture version, and an
   explicit `validation_tolerance` object per the caveats above).

## Sources

- [NCEI Climate Divisional Database drought files — cirs/climdiv](https://www.ncei.noaa.gov/pub/data/cirs/climdiv/)
- [NCEI climdiv drought-readme.txt](https://www.ncei.noaa.gov/pub/data/cirs/climdiv/drought-readme.txt)
- [U.S. Climate Divisional Dataset (C00702) — NCEI metadata](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00702/html)
- [Climate Division Datasets (nClimDiv) — Drought.gov](https://www.drought.gov/data-maps-tools/climate-division-datasets-nclimdiv)
- [U.S. Gridded Standardized Precipitation Index (SPI) from nClimGrid-Monthly — Drought.gov](https://www.drought.gov/data-maps-tools/us-gridded-standardized-precipitation-index-spi-nclimgrid-monthly)
- [Source Code: Climate and Drought Indices in Python (SPI, SPEI, PET) — Drought.gov](https://www.drought.gov/data-maps-tools/climate-and-drought-indices-python-spi-spei-pet)
- [Source Code: Drought Indices in Fortran (SPI, PDSI) — Drought.gov](https://www.drought.gov/data-maps-tools/drought-indices-fortran-spi-pdsi)
- [monocongo/climate_indices — GitHub](https://github.com/monocongo/climate_indices)
- Baldwin, M. E., and Chen, F. (2020). Major Over- and Underestimation of Drought Found
  in NOAA's Climate Divisional SPI Dataset. *Journal of Applied Meteorology and
  Climatology*, 59(9), 1587–1601.
  https://repository.library.noaa.gov/view/noaa/60397
- [Standardized Precipitation Index User Guide, WMO-No. 1090 (2012)](https://www.droughtmanagement.info/literature/WMO_standardized_precipitation_index_user_guide_en_2012.pdf)
- [Standardized Precipitation Index User Guide — WMO e-Library](https://library.wmo.int/records/item/39629-standardized-precipitation-index-user-guide)
- Guttman, N. B. (1999). Accepting the Standardized Precipitation Index: A Calculation
  Algorithm. *Journal of the American Water Resources Association*, 35(2), 311–322.
  (cited for the Pearson Type III recommendation for U.S. climate-division data)
- McKee, T. B., Doesken, N. J., and Kleist, J. (1993). The relationship of drought
  frequency and duration to time scales. *Proceedings of the 8th Conference on Applied
  Climatology*, American Meteorological Society, 179–184.
- [WestWide Drought Tracker (WWDT) — Drought.gov](https://www.drought.gov/data-maps-tools/westwide-drought-tracker-wwdt-gridded-monthly-drought-indices-western-us)
- [SPI Program — National Drought Mitigation Center](https://drought.unl.edu/monitoring/SPI/SPIProgram.aspx)
- Prior finding referenced: `docs/research/nclimgrid-acquisition-and-redistribution.md`
  — established the NOAA/NCEI public-domain redistribution terms this survey relies on.

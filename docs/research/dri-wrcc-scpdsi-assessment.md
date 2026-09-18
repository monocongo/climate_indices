# DRI/WRCC scPDSI external-product assessment

**Research date:** 2026-09-18
**Scope:** Whether the Western Regional Climate Center's (WRCC, at the Desert Research Institute) self-calibrated PDSI product can serve as independent external-product validation for `climate_indices.palmer.scpdsi()`. Every claim below is taken from a primary source — the product's own pages and file metadata on `wrcc-archive.dri.edu`, the PRISM terms of use, and the Crossref record for the product's citation (DOI 10.1175/BAMS-D-16-0193.1). The BAMS article itself returned HTTP 403 from this environment, so no claim rests on its contents.

## Answer

WRCC does publish a genuine self-calibrated PDSI: the WestWide Drought Tracker (WWDT) `sc-PDSI`, a 4-km monthly grid over the continental United States for 1895 through September 2025, computed from PRISM monthly temperature and precipitation, STATSGO available-water-capacity, and no snow representation, with self-calibration to 1895-2010. Its files are served from an open, unauthenticated Apache directory listing, so **acquisition is reproducible today**.

It is not usable as-is for implementation validation, for four independent reasons.

1. **The product is frozen and partly broken.** The newest per-month file is September 2025; the files were last modified 2025-10-16, eleven months before this research date. `scpdsi_current_PRISM.nc`, the file the download page offers for current conditions, contains 872,505 NaN values and zero finite values. Every time-series endpoint probed (`/wwdt/time/text/`, `/wwdt/time/regionText/`) returned HTTP 500. Whatever upstream produced this product is no longer running, and WRCC's own site now serves a redirect to `wrcc-archive.dri.edu`.
2. **No data license.** No page on the WWDT site or the WRCC archive states redistribution terms for the gridded outputs; the WRCC site asserts a blanket copyright (© 2016-2026 Western Regional Climate Center). PRISM's inputs are freely reproducible with attribution, but the derived scPDSI output carries no such statement. Committing WWDT-derived fixtures to this repository is redistribution of a product whose terms are unstated.
3. **No published implementation.** WWDT documents the algorithm family (Wells et al. 2004) but publishes no source, no PET formula for the Palmer family, and no AWC processing detail beyond "STATSGO, top 250 cm". A tolerance cannot be defended for differences whose source is unobservable.
4. **The comparison is gridded-to-divisional by construction.** WWDT is a 4-km continental grid; `climate_indices` validation targets are NOAA climate divisions. A defensible comparison needs identical PRISM inputs on the WWDT grid, which is a much larger acquisition than any prior validation fixture in this repository (full record is roughly 5 GB of scPDSI alone, plus comparable PRISM inputs).

A like-for-like comparison is technically constructible — `palmer.scpdsi()` accepts an explicit calibration window (`calibration_year_initial=1895`, `calibration_year_final=2010`) — but it would be plausibility-class agreement, not tolerance-bound independent validation. The repository already characterizes its own `scpdsi()`-versus-nClimDiv comparison as characterization only, with median absolute difference 0.47; the WWDT comparison adds input, snow, AWC, and calibration-window differences on top of that. Recommendation: do not acquire WWDT as a committed fixture set. Confirm product status and redistribution terms with DRI/WRCC first, and treat any later comparison as plausibility/agreement evidence documented as such in `VALIDATION.md`.

## Sourced evidence

### 1. Producer, product, and lineage

- The WWDT About page lists "Self-Calibrated Palmer Drought Severity Index (sc-PDSI)" among its products and states: "sc-PDSI (self-calibrated PDSI) is a locally calibrated version of the PDSI designed to make values of the PDSI more comparable across space (Wells et al., 2004). ... The sc-PDSI ensures that values exceeding +4 occur 2 percent of the time, and likewise for values less than -4." It cites Wells, Goddard & Hayes (2004), *J. Climate* 17, 2335-2351. [about]
- The product's citation is "Abatzoglou, J.T., D.J. McEvoy, and K.T. Redmond, in press, The West Wide Drought Tracker: Drought Monitoring at Fine Spatial Scales, Bulletin of the American Meteorological Society" [about], published as BAMS 98(9), 2017, DOI 10.1175/BAMS-D-16-0193.1 [crossref].
- Every downloaded NetCDF file carries `author: John Abatzoglou - University of Idaho` and `date: 16 October 2025` [netcdf]. WRCC/DRI is the host and legacy producer; the file-level author is now affiliated with the University of Idaho.
- The WRCC archive root states: "WRCC is building a new web site with new and updated tools. Check out https://wrcc.dri.edu/my for the latest." A site-wide banner on every WWDT page reads: "The WestWide Drought Tracker is getting a major upgrade! New and improved maps are located here. We strongly encourage using the new maps application. Continue to use the menu options above for time series and archive features until they are upgraded at a later date." The "here" target redirects to `https://wrcc.app/wwdt` [root, about].
- `https://wrcc.dri.edu/wwdt/` and `https://wwdt.dri.edu/` both redirect to `https://wrcc-archive.dri.edu/wwdt/`, so the archive host is the official current route, not an unaffiliated mirror [redirects].

### 2. Algorithm variant, inputs, and calibration

- Palmer-family inputs are documented as: "Input data for these budget terms consist solely of monthly temperature and precipitation. Fixed soil characteristics are supplied independently by incorporating the available water holding capacity of the top 250 cm of the soil acquired from the State Soil Geographic Data Base (STATSGO). Snow and its effects are not represented." [about]
- The PRISM inputs are "the AN81m and AN81d datasets available from http://www.prism.oregonstate.edu/" [about].
- The calibration window is stated in the NetCDF metadata: title "Monthly gridded data at 4-km (2.5 arc-minute) resolution for Self Calibrated Palmer Drought Severity Index (scPDSI), **calibrated to 1895-2010** for the continental United States", and data description "Self Calibrated Palmer Drought Severity Index (scPDSI), calibrated to 1895-2010" [netcdf]. The window is fixed at the file level, so every post-2010 value is calibrated against 1895-2010 rather than a rolling period.
- The PET formula used for the Palmer family is **not** stated on any WWDT page. The About page's "monthly temperature and precipitation" input statement is consistent with Thornthwaite PET but does not establish it. The time-series page describes scPDSI loosely: "created by replacing constant empirical values- mainly extreme events- with dynamically calculated values that are more spatially explicit" [about, time].
- No source code, notebook, or processing script for the WWDT indices is published on the site; the download page offers only data files and wget scripts [batch].

### 3. Domain, resolution, and temporal coverage

- Grid read from the committed files: 621 latitudes × 1405 longitudes = 872,505 cells, latitude 24.06-49.90, longitude -125.00 to -66.56, projection "GCS WGS 1984" [netcdf].
- Files are labeled monthly but carry a dimension named `day` with one timestep per file (for example `2025-10-15`), a naming quirk any importer must handle [netcdf].
- Per-year-month files run from `scpdsi_1895_1_PRISM.nc` through `scpdsi_2025_9_PRISM.nc`; the directory listing contains 3,155 `.nc`/`.tif` artifacts [listing].
- The latest verified real data is September 2025. `scpdsi_9_PRISM.nc` (all Septembers, 436 MB) and the per-year files were last modified 2025-10-16 05:36-05:37 UTC; `scpdsi_current_PRISM.nc` was last modified 2025-10-16 12:37:05 UTC with `Content-Length: 3499628` [listing, headers].
- `scpdsi_current_PRISM.nc` (timestep 2025-10-15) is entirely missing data: 872,505 NaN cells, 0 finite values. `scpdsi_1895_12_PRISM.nc`, by contrast, holds 476,109 finite values in the range -5.99 to 4.74 [netcdf].
- The time-series application is dead: `https://wrcc-archive.dri.edu/wwdt/time/text/?lat=39.5&lon=-119.8&variable=SCPDSI&start_year=1895&end_year=2025` and its `regionText`/`PDSI` variants all return HTTP 500 ("Page unavailable ... or the requested data may not exist"). The page's own backend contract is `.../time/text/?lat&lon&variable&start_year&end_year`, discovered in `media/js/interface.js` [time, probe].

### 4. Formats, precision, and file layout

| Artifact | Size | Content |
| --- | --- | --- |
| `scpdsi_{year}_{month}_PRISM.nc` | 3.3 MB | One month, float32, CONUS 4-km grid |
| `scpdsi_{month}_PRISM.nc` | 436 MB | All years for one calendar month |
| `scpdsi_current_PRISM.nc` | 3.3 MB | Latest month (currently all-NaN) |
| `scpdsi_REGIONS_PRISM.nc` | 21 MB | 1416 months × 3888 undocumented polygon ids, last modified 2012-07-05 |
| `scpdsi_*.tif` | 1.7 MB | Same data as GeoTIFF |

- Values are stored as float32, `units: Unitless`; the storage precision is not the algorithmic precision [netcdf].
- `scpdsi_REGIONS_PRISM.nc` holds monthly scPDSI for 1895-01 through 2012-12 against 3,888 numeric polygon identifiers with no region names, lookup table, or shapefile on the site. It cannot be mapped to climate divisions and is 14 years stale; treat it as unusable [regions].

### 5. Update, versioning, and current operational state

- The About page states data are "updated with new values at the beginning of each month" [about]. Observed behavior contradicts this by eleven months: all artifact timestamps stop at 2025-10-16, and the current-month file is empty [listing, netcdf].
- There is no versioned release, DOI, checksum manifest, or immutable object naming. Files are overwritten in place under fixed names, and the only revision signal is the Apache `Last-Modified` header (for example `scpdsi_current_PRISM.nc`: `ETag "35666c-64145dfef6a83"`, `Last-Modified: Thu, 16 Oct 2025 12:37:05 GMT`, `Accept-Ranges: bytes`) [headers].
- Near-real-time data are explicitly provisional: "All near-real-time data are considered preliminary and should be used responsibly. All near real-time data should be considered provisional, subject to change, and used accordingly." [time]
- The host is legacy infrastructure: `/wwdt/data/version.php` returns a full `phpinfo()` page for PHP 5.3.29 built 2020-12-04 on Ubuntu 2.6.32, exposing `/jtwrcc/research/WWDT/` server paths [version]. This is an unmaintained deployment, not a managed data service.
- The "new" application at `https://wrcc.app/wwdt` is a single-page app whose bundle posts JSON to `https://wrcc-archive.dri.edu/pass` — the same archive host — and exposes `/login`, `/register`, and `/request-password-reset` routes. Whether its data calls require an account was not established from this environment; a bare request to `/pass` returns `{"status": "ERR", "msg": "Request has no content"}` [app, pass].

### 6. License and redistribution

- No license, terms-of-use, or redistribution statement exists on the WWDT pages or in the NetCDF attributes. The only attribution instruction is `note3: Citation: Westwide Drought Tracker, http://www.wrcc.dri.edu/monitor/WWDT` and the BAMS citation [netcdf, about].
- The WRCC archive site asserts copyright: "© 2016-2026 Western Regional Climate Center (WRCC), 2215 Raggio Parkway, Reno, NV 89512-1095" [citations]. Its product page says data available through its websites "can be downloaded ... at no cost" but makes no grant to reproduce or redistribute [products].
- PRISM's terms, which govern the primary inputs, are permissive: "All data (gridded, polygon, tabular, graphical) retrieved from this website or otherwise provided on the website may be freely reproduced and distributed", with the requirement to "clearly and prominently state, at a minimum, our name, URL, and the date of data access"; PRISM "retains rights to ownership"; data are provided "as is"; and PRISM is "not recommended ... to calculate very long-term trends" [prism-terms]. That permission covers PRISM inputs, not the WWDT output derived from them.
- The time-series disclaimer names "the National Weather Service Cooperative Observer Network (COOP)" as an additional data source, whose terms are not addressed anywhere on the site [time].

### 7. Acquisition routes that work today

- **Open directory listing (recommended for any pilot):** `https://wrcc-archive.dri.edu/wwdt/data/PRISM/scpdsi/` over HTTPS, no authentication, Apache listings, `Accept-Ranges: bytes` so single months can be fetched without downloading the 436 MB series files [listing, headers].
- **Bulk script generation:** `https://wrcc-archive.dri.edu/wwdt/batchdownload.php` builds URL or wget lists from variable/month/timescale selections [batch].
- **Broken:** the map-based time-series extractor (HTTP 500), and the new app's data API is session-oriented and unverified [probe, app].

## Recommended comparison design

If the product is pursued after the operational and licensing questions are resolved, the only defensible design is like-for-like on the WWDT grid:

1. Acquire PRISM monthly precipitation and temperature on the 4-km CONUS grid from `https://prism.oregonstate.edu/downloads` for 1895-present, citing "PRISM Group, Oregon State University, https://prism.oregonstate.edu, accessed <date>" and labeling the derived product as modified.
2. Run `palmer.scpdsi()` with `calibration_year_initial=1895`, `calibration_year_final=2010` to match the WWDT calibration window, using the same STATSGO top-250-cm AWC treatment the WWDT documents and the repository's Thornthwaite PET path.
3. Compare month-by-month, cell-by-cell against `scpdsi_{year}_{month}_PRISM.nc`, excluding the all-NaN `_current` file, with agreement measured over the full 1895-2010 calibration window and a held-out period.

Acceptance criteria must be plausibility-class, not tolerance-bound. Two reference points from this repository bound what is achievable: the Wells-lineage C++ oracle agrees with `scpdsi()` at `atol=5e-5`, while the nClimDiv comparison — a same-family product on the same country — diverges with median absolute difference 0.47. WWDT adds unobservable implementation differences (PET formula, AWC processing, snow omission) plus a gridded-versus-divisional scale mismatch, so any ceiling must be measured on a pilot and then frozen with documented headroom, exactly as `tests/test_nclimdiv_reference.py` does. Suggested assertions for such a pilot: fraction of cell-months agreeing in sign, agreement in NDMC drought categories, and rank correlation by cell. If the pilot cannot separate implementation error from those known differences, the honest outcome is the one the repository already uses for `scpdsi()`-versus-nClimDiv: characterization, not independent validation, stated as such in `VALIDATION.md`.

Cheaper alternatives were considered and rejected:

- `scpdsi_REGIONS_PRISM.nc` (21 MB) is stale (2012) and its polygons carry no names or lookup — unusable.
- Area-weighting the WWDT grid to climate divisions and comparing against the repository's divisional scPDSI conflates PRISM-versus-nClimDiv input differences with implementation differences, so it cannot support any acceptance criterion.

## Unknowns requiring maintainer confirmation

1. **Operational status.** Is WWDT still produced? The data stop in September 2025, the current-month file is empty, and the extractor service is down. Contact WRCC/DRI (or the file author, University of Idaho) before any acquisition work.
2. **Redistribution and release terms.** May WWDT-derived values be committed to this repository or a release asset? No statement was found either way, and the WRCC site claims copyright.
3. **PET formula and AWC processing** used for the Palmer family, needed before a like-for-like comparison can claim to isolate implementation differences.
4. **Future distribution route and authentication.** Whether `wrcc.app` continues WWDT and whether its `/pass` API requires an account, a key, or agreed terms.
5. **Calibration persistence.** Whether 1895-2010 is the calibration window in every file variant (verified in the two files read here, not in all 3,155).

## Sources

[about]: https://wrcc-archive.dri.edu/wwdt/about.php
[time]: https://wrcc-archive.dri.edu/wwdt/time/
[batch]: https://wrcc-archive.dri.edu/wwdt/batchdownload.php
[listing]: https://wrcc-archive.dri.edu/wwdt/data/PRISM/scpdsi/
[netcdf]: `scpdsi_current_PRISM.nc` and `scpdsi_1895_12_PRISM.nc`, downloaded 2026-09-18 from https://wrcc-archive.dri.edu/wwdt/data/PRISM/scpdsi/
[regions]: `scpdsi_REGIONS_PRISM.nc`, downloaded 2026-09-18 from https://wrcc-archive.dri.edu/wwdt/data/PRISM/scpdsi/
[headers]: HTTP HEAD response for https://wrcc-archive.dri.edu/wwdt/data/PRISM/scpdsi/scpdsi_current_PRISM.nc, 2026-09-18
[probe]: HTTP responses for https://wrcc-archive.dri.edu/wwdt/time/text/?lat=39.5&lon=-119.8&variable=SCPDSI&start_year=1895&end_year=2025 and variants, 2026-09-18
[root]: https://wrcc-archive.dri.edu/
[citations]: https://wrcc-archive.dri.edu/About/citations.php
[products]: https://wrcc-archive.dri.edu/About/products.php
[version]: https://wrcc-archive.dri.edu/wwdt/data/version.php
[app]: https://wrcc.app/wwdt and its bundle https://wrcc.app/assets/index-CdI72jUB.js
[pass]: https://wrcc-archive.dri.edu/pass
[crossref]: https://doi.org/10.1175/BAMS-D-16-0193.1 (Crossref metadata fetched 2026-09-18)
[prism-terms]: https://prism.oregonstate.edu/terms/
[downloads]: https://prism.oregonstate.edu/downloads

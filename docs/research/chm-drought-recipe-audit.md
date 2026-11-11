# CHM_Drought V1.0 recipe audit and input pinning

Resolves: [monocongo/earthflux#207](https://github.com/monocongo/earthflux/issues/207),
a child of [monocongo/earthflux#206](https://github.com/monocongo/earthflux/issues/206).

Research date: 2026-11-09. Every access statement below describes what was fetched
on that date. Absent product fields are recorded as absent, not reconstructed from
the published values.

Pinned library: `climate_indices` tag
[`v3.1.0`](https://github.com/monocongo/climate_indices/tree/v3.1.0), the version
Earthflux pins in `worker/requirements.txt`.

## Question

Can this library reproduce the CHM_Drought V1.0 indices from the same public
inputs? Which recipe elements are declared, which match the pinned library, which
differ by design, and which are missing? Which exact input revisions can be
pinned, and does the evidence support a bounded shared-input comparison?

## Answer

**Decision: go with characterization only.**

- **SPI** has a reproducible candidate input: CHM_PRE v4 (figshare v4,
  CC BY 4.0, MD5 recorded below). The CHM_Drought input revision remains unknown.
  A bounded six-cell, 1961-2022 comparison may proceed under the predeclared
  hypothesis grid (section "Predeclared comparison protocol"). The result is
  **characterization, not parity**: input-version and recipe differences remain
  confounded. The product does not declare its Calibration Period, gamma
  estimator, Zero Handling, or clipping.
- **SPEI** is **blocked for a shared-input comparison**. Its PET needs six
  interpolated meteorological fields that the product built in-house from
  restricted CMA station data. Those exact fields were never published. The
  public `CHM_Tmax`/`CHM_Tmin`/`CHM_Tmean`/`CHM_Rh`/`CHM_Ssd`/`CHM_Wind`
  records postdate the product, cover different years, and carry no version or
  checksum that ties them to CHM_Drought V1.0.
- **EDDI** and the **Palmer** family are **not parity targets**. CHM's EDDI uses a
  parametric transform; this library's EDDI uses NOAA PSL empirical ranking.
  CHM's PDSI uses China-adjusted constants. Both differences are by design and
  must not be "fixed" to match one external product.
- **2-week SPI** can use daily periodicity with `scale=14`. It is a secondary
  characterization target, subject to the calendar and sampling checks below.

No comparison should be read as an implementation-parity claim. The product
publishes values, not a reproducible implementation or fitted parameters.

## Product facts

Sources: [Zenodo record 14634774](https://zenodo.org/records/14634774) (DOI
[10.5281/zenodo.14634774](https://doi.org/10.5281/zenodo.14634774), CC BY 4.0),
[metadata PDF](https://zenodo.org/records/14634774/files/Metadata%20for%20CHM_Drought.pdf?download=1),
[ESSD paper](https://doi.org/10.5194/essd-17-837-2025) and
[ESSD supplement](https://essd.copernicus.org/articles/17/837/2025/essd-17-837-2025-supplement.pdf).

- Contents: SPI, SPEI, EDDI at 2 weeks and 1-12 months; PDSI, SC-PDSI, VPD
  monthly. Version V1.0, dated June 2024 in the metadata PDF; the Zenodo record
  was published 2025-01-12.
- Domain: mainland China, 0.1 degrees, 1961-2022.
- Grid read from the NetCDF files: longitude 72.05-135.95 E (640 cells), latitude
  18.05-53.95 N (360 cells), step 0.1 degrees; 744 monthly time steps.
- Inputs named in the paper: CMA daily station data (2419 stations) for Tmax,
  Tmin, Tmean, wind, sunshine duration and relative humidity, interpolated by the
  authors with angular distance-weighted interpolation and correlation decay
  distance; CHM_PRE precipitation; soil available water capacity (AWC) from the
  Global Gridded Surfaces of Selected Soil Characteristics (DAAC `ds_id=1006`).
  CN05.1 and CRU are evaluation datasets only, not inputs.

## Recipe matrix

Status vocabulary, applied against the pinned library:

- **matched** - the product source states the element and it agrees with the
  library route.
- **different** - the product source states the element and it disagrees by
  design.
- **unknown** - the product source does not state the element.

The **Basis** column separates a stated element from one inferred from the
software named by the product. An inferred element is never quietly promoted to
a stated one; it becomes a hypothesis in the comparison protocol.

### SPI

| Element | CHM_Drought V1.0 | climate_indices 3.1.0 | Status | Basis and source |
| --- | --- | --- | --- | --- |
| Input | Precipitation only (paper Table 1); CHM_PRE (NetCDF `Summary`; paper section 2.1) | Precipitation array, monthly or daily periodicity | matched | Stated |
| Accumulation | Window sums at Timescales of 2 weeks and 1-12 months | `sum_to_scale` trailing sum, leading `scale - 1` values NaN | matched for 1-12 months | Stated Timescales |
| Window alignment | Not stated. The metadata PDF reports 12-month values starting January 1962, which is one month later than a trailing sum that ends in the labeled month would start | Trailing sum ending in the labeled month (first 12-month value at 1961-12 for a 1961-01 start) | **unknown** | Not stated; tested as hypothesis A0 or A1 in the protocol |
| Distribution | Gamma (supplement S2, Eq. S1) | `Distribution.gamma` | matched | Stated |
| Gamma estimator | Supplement S2 prints the maximum-likelihood approximation: `A = ln(x_bar) - mean(ln x)`, `shape = (1 + sqrt(1 + 4A/3)) / (4A)`, `scale = x_bar / shape` | `compute.gamma_parameters` applies the identical formula (documented in the library as method of moments) | **unknown** | Sources conflict: the supplement prints the MLE approximation, while the NetCDF `Software` attribute names R `SPEI`, whose gamma path fits unbiased-PWM L-moments (`lmom::pelgam`), not this formula |
| Zero mass | `F(x = 0) = m / n` (supplement Eq. S3) | `prob_zero = zeros / non-missing` over the calibration window | matched | Stated |
| Zero Handling | Not stated | `zero_handling="classic"` scores a zero at `Phi^-1(p0)`; alternatives `center_of_mass` (`p0/2`) and `mean_zero` exist | **unknown** | Not stated. R `SPEI` 1.8.1 (released 2023-03-02) evaluates its zero-mass adjustment before the standardized value is assigned in `R/spei.R`, so that adjustment does not reach the result; the current upstream source applies it after. The effective convention is version-dependent |
| Calibration Period | Not stated anywhere in the paper, supplement, metadata PDF, or NetCDF attributes | Explicit `calibration_year_initial` / `calibration_year_final`, per calendar month; a window the record does not cover is clamped (`policy="clamp"`) | **unknown** | Not stated |
| Per-calendar-month fit | Not stated | Yes: parameters are fitted per calendar month over the calibration years (`gamma_parameters`) | **unknown** | Not stated |
| Normal transform | Standard normal inverse with the published rational approximation coefficients (supplement Eq. S4-S5) | `scipy.stats.norm.ppf` (numerically equivalent) | matched | Stated |
| Clipping | Not stated | Clipped to `[-3.09, 3.09]` on the normal scale | **unknown** | Not stated. R `SPEI` does not clip |
| 2-week Timescale | Published | Daily periodicity with `scale=14` | **unknown** | The library supports a 14-day accumulation; the product's calendar, sampling and window alignment remain unspecified |
| Sign convention | Standardized drought index | Same | matched | Stated |

### SPEI

| Element | CHM_Drought V1.0 | climate_indices 3.1.0 | Status | Basis and source |
| --- | --- | --- | --- | --- |
| Inputs | P and PET (paper Table 1) | Precipitation and PET arrays, same shape | matched | Stated |
| Water balance | `D = P - PET` (paper section 3.2; supplement S3) | `(P - PET) + 1000`, negative precipitation clipped to zero | matched | Stated. For a 3-parameter generalized logistic, a constant offset changes only the fitted location, not the standardized value |
| Distribution | Three-parameter log-logistic (supplement S3, Eq. S6) | `Distribution.loglogistic` (generalized logistic) | matched | Stated |
| Estimator | Not stated | Unbiased-PWM L-moments (`lmoments.fit_glo`) | **unknown** | Not stated. The NetCDF `Software` attribute names R `SPEI`, whose `spei()` default is `distribution = "log-Logistic"`, `fit = "ub-pwm"`; that default is not a declaration of what ran |
| PET formula | FAO-56 Penman, paper Eq. 3: `[0.408 D (Rn - G) + g 900/(T + 273) u2 (es - ea)] / [D + g (1 + 0.34 u2)]` | `pm_eto.pm_eto` implements FAO-56 Eq. 6 in the same form | matched | Stated |
| Radiation pathway | Sunshine duration with Angstrom `as = 0.25`, `bs = 0.5`, `Rs = (as + bs n/N) Ra`; net long-wave with `Rs/Rso` cloud term (supplement S11, Eq. S34-S41) | `solar_radiation_from_sunshine` uses `as = 0.25`, `bs = 0.50`; `net_radiation` implements the same clear-sky and cloud terms | matched | Stated |
| Humidity pathway | `ea = RHmean/100 * e0(T)` (supplement S10, Eq. S32), with `es = (e0(Tmax) + e0(Tmin))/2` in S7 and S10 | `actual_vapor_pressure_from_rhmean` with `e_s = mean_saturation_vapor_pressure(tmin, tmax)` (FAO-56 Eq. 12 and 19) | matched | Stated formula; the `e0(T)` argument is consistent with S10's own `es` definition, but the supplement's notation is loose |
| Temperature in the aerodynamic term | Paper Eq. 3 uses "T, the mean daily air temperature"; the product interpolates a separate Tmean field (paper section 2.1) | `penman_monteith_eto` uses `tmean = (tmin + tmax)/2` | **unknown** | Not stated whether the separate Tmean field or the daily mean of Tmin/Tmax enters Eq. 3 |
| Wind reference height | "Average wind speed" from station data; height not stated | `wind_speed_2m` converts from a declared measurement height, default 2 m | **unknown** | Not stated. CMA station wind is commonly measured at 10 m |
| Elevation input | Psychrometric constant from elevation (supplement S9, Eq. S30); elevation source not stated for the PET grid | `atmospheric_pressure(elevation_m)` from the supplied elevation | **unknown** | Formula stated; the elevation field is not |
| Daily to monthly aggregation | Not stated | The library computes daily ETo; monthly aggregation is the caller's step | **unknown** | Not stated. The SPEIbase evidence shows this convention is material: leap-aware month lengths versus a near-uniform month changes the fit residual by a factor of two to four |
| Calibration Period | Not stated | Explicit window, per calendar month | **unknown** | Not stated |
| Clipping | Not stated | Clipped to `[-3.09, 3.09]` | **unknown** | Not stated |

### EDDI

| Element | CHM_Drought V1.0 | climate_indices 3.1.0 | Status | Basis and source |
| --- | --- | --- | --- | --- |
| Input | ET0 (paper Table 1) | PET array | matched | Stated |
| ET0 formula | ASCE-EWRI standardized reference ET, paper Eq. 4 with `Cn`, `Cd` (Allen et al., 2005); the paper does not print the `Cn` and `Cd` values | `pm_eto` (FAO-56 Eq. 6) | **unknown** | Form stated; constants absent. Only the short-grass daily pair (`Cn = 900`, `Cd = 0.34`) coincides with the FAO-56 form |
| Standardization | Parametric: accumulate ET0, then apply the inverse-normal rational approximation to a probability `P(ET0)` (paper Eq. 5-7; supplement S4) | Non-parametric empirical ranking, Tukey plotting position, Hastings inverse-normal | **different** | Stated. Different by design; not a parity target |
| Distribution for `P(ET0)` | Not stated | Empirical | **unknown** | Not stated |
| Calibration Period | Not stated | Explicit calibration window | **unknown** | Not stated |
| Time axis | The audited EDDI-12 file starts at an offset of 15 days (`time[0] = 15`), unlike the month-start labels of the audited SPI, SPEI and Palmer files | Month-start labels | **unknown** | Observed label offset, not a declared design difference. Whether the labels identify the same monthly window is untested; month alignment needs verification |
| Clipping | Not stated | Clipped to `[-3.09, 3.09]` | **unknown** | Not stated |

### PDSI and SC-PDSI

| Element | CHM_Drought V1.0 | climate_indices 3.1.0 | Status | Basis and source |
| --- | --- | --- | --- | --- |
| Inputs | P, PET, AWC (paper Table 1) | P, PET, AWC arrays | matched | Stated |
| Palmer constant `a` | 16.84, revised for China (paper section 3.4; supplement S5, Eq. S14, citing Zhong et al., 2019) | 17.67 | **different** | Stated; by design |
| Duration factors | `p = 0.755`, `q = 1/1.63`, revised for China (supplement S5, Eq. S16) | `p = 0.897`, `q = 1/3` (standard Palmer) | **different** | Stated; by design |
| AWC source | Global Gridded Surfaces of Selected Soil Characteristics, DAAC `ds_id=1006` (paper section 2.1) | Supplied by the caller | **unknown** | Source stated; the grid version and the cell aggregation are not |
| K-prime formulation | Palmer Eq. 8-9 (paper), supplement Eq. S14-S15 | Standard Palmer formulation, evidenced against nClimDiv | matched | Stated product equations; the library's standard formulation is the one measured in the nClimDiv row of `VALIDATION.md` |
| SC-PDSI calibration | Wells et al. (2004), `fe = 2%`, percentile recalibration, two duration-factor sets; R `scPDSI` package (paper section 3.5; supplement S6) | Wells-lineage self-calibration, three rescaling passes | **different** | The self-calibration family is shared, but the base durations and constants differ by design |
| Calibration Period | Not stated for either Palmer index | Explicit window; SC-PDSI self-calibrates over it | **unknown** | Not stated |

The Palmer family stays outside the comparison. Matching it would require the
China constants and a different scientific definition, which the epic excludes.

## Existing upstream evidence inventory

This audit reuses the committed evidence instead of regenerating it. Tolerances
below are the recorded values in `VALIDATION.md` at `v3.1.0`.

| Evidence | Reference | Tolerance or measured result | What it covers here |
| --- | --- | --- | --- |
| PET Penman-Monteith (FAO-56) | `tests/test_pm_eto.py`, FAO-56 Examples 10-18 | Final ETo `atol = 0.05 mm/day`; radiation intermediates `0.05-0.1 MJ m-2 day-1`; Example 18 ETo = 3.88 mm/day | The equation CHM's SPEI cites; EDDI's ASCE-EWRI form reduces to it for the daily grass reference. Gap: monthly Example 17 is not encoded |
| SPEI log-logistic cross-implementation | `tests/test_spei_loglogistic_reference.py`, R `SPEI` (`distribution = "log-Logistic"`, `fit = "ub-pwm"`) | Series `atol = 1e-6` declared; measured about `1e-12`. Parameters `rtol = 1e-8`, `atol = 1e-6`; measured at most about `1e-10` | The distribution route CHM names. Fixtures, provenance and a version-guarded regeneration script are committed under `tests/fixture/spei_loglogistic/` |
| SPEI log-logistic like-for-like, SPEIbase v2.11 | `tests/test_speibase_like_for_like.py` | Correlation 0.9996-0.99999; mean absolute difference 0.003-0.016; floors at correlation >= 0.97 | The committed measurement of a PET month-length convention effect: leap-aware versus near-uniform months changes the residual two to four times |
| SPI zero handling | `tests/test_zero_handling.py`, `tests/test_zero_handling_reference.py` (R `SEI`, R `SCI`) | Zero Handling constants `atol = 1e-12`; parameter-matched series `atol = 1e-6`; end-to-end `atol = 0.01`; measured end-to-end difference at most 0.0032 | The Zero Handling modes the CHM comparison must vary. The library's gamma is an approximation; R `SEI` uses MLE gamma, and the measured difference is the same order as the CHM estimator question |
| EDDI | `tests/test_noaa_eddi_reference.py`, NOAA PSL | `rtol = 1e-5`, `atol = 1e-5`; maximum observed error `2.43e-6` | Confirms the library's empirical route is the NOAA route. It says nothing about a parametric EDDI |
| PDSI, standard Palmer | `tests/test_nclimdiv_reference.py` | Median absolute difference 0.0127 over 528,384 division-months; 86.2% within 0.05 | Standard-constant evidence only; not the China variant |
| SC-PDSI | `tests/test_scpdsi.py`, Wells-lineage oracle | `atol = 5e-5`, `rtol = 0` | Self-calibration arithmetic, not CHM's China-constant base |

There is no upstream evidence for CHM_Drought itself, and none of the rows above
supports a CHM parity claim. In particular, the R `SPEI` agreement is a
cross-implementation check of a port: `lmoments.fit_glo` is a translation of the
same `lmom` PELGLO routine R `SPEI` calls, so close agreement is expected.

## Input records

### Precipitation (pinnable candidate; product revision unknown)

CHM_PRE, described by [Han et al. (2023)](https://doi.org/10.5194/essd-15-3147-2023)
and named in the SPI-12 and SPEI-12 `Summary` attributes.

| Field | Value |
| --- | --- |
| Record | [10.6084/m9.figshare.21432123.v4](https://doi.org/10.6084/m9.figshare.21432123.v4) |
| Version | 4 |
| Published | 2023-06-18 |
| Licence | CC BY 4.0 |
| Coverage | 1961-2022, daily, 0.1, 0.25 and 0.5 degrees, 18-54 N, 72-136 E |
| 0.1-degree file | `CHM_PRE_0.1dg_19612022.nc.zip`, 1,847,170,334 bytes, MD5 `2444a3296c59e7c7303c7be9161945dc` |
| 0.25-degree file | `CHM_PRE_0.25dg_19612022.nc.zip`, 365,577,616 bytes, MD5 `16e111df0bd34c8ecb01680c15152363` |
| 0.5-degree file | `CHM_PRE_0.5dg_19612022.nc.zip`, 105,588,595 bytes, MD5 `3afc5e548fa792d7db65b344b6cee598` |

Two caveats are recorded rather than resolved. Version 3 (2023-04-05) also covers
1961-2022 with different bytes (`002bd1af9eb7a39235babdb79ddcbc53` for the
0.1-degree file), and CHM_Drought does not declare which of v3 or v4 it read.
Version 4 is the version the ESSD paper cites, so it is the **probable** input,
not a declared one. The paper also links a National Tibetan Plateau Data Center
record (`e5c335d9-cbb9-48a6-ba35-d67dd614bb8c`); that page is a JavaScript
application and its version metadata could not be retrieved without a browser.

The `CHM_PRE` 0.1-degree NetCDF global attributes could not be read by bounded
remote access: the HDF5 metadata needed for an open was not inside a 96 MB
prefix of the 21.2 GB member, so the attempt stopped rather than fetch more. The
pin above rests on the versioned DOI, publication date, licence and MD5, which is
the stronger evidence in any case.

### Meteorological fields (not pinnable)

The paper states that the authors interpolated CMA station data themselves, with
ADW and per-variable correlation decay distance (paper section 2.2; supplement
S1). The interpolated fields used for V1.0 were therefore not a public dataset at
build time. The raw CMA station observations are restricted and are not a
redistribution input.

The public CHM grids that exist today are later, separate releases:

| Variable | Zenodo record | Version | Published | Coverage | Files |
| --- | --- | --- | --- | --- | --- |
| Tmax | [10.5281/zenodo.22097524](https://doi.org/10.5281/zenodo.22097524) | v2 | 2026-08-25 | 1961-2025 | eight zips, 5.18 GB |
| Tmin | [10.5281/zenodo.22092514](https://doi.org/10.5281/zenodo.22092514) | v2 | 2026-08-25 | 1961-2025 | eight zips, 5.27 GB |
| Tmean | [10.5281/zenodo.17472829](https://doi.org/10.5281/zenodo.17472829) | V1.0 | 2025-11-26 | 1961-2024 | seven NetCDF files, 21.55 GB |
| Tmean | [10.5281/zenodo.22097727](https://doi.org/10.5281/zenodo.22097727) | v2 | 2026-08-25 | 1961-2025 | eight zips, 5.14 GB |
| Relative humidity | [10.5281/zenodo.17465143](https://doi.org/10.5281/zenodo.17465143) | V1.0 | 2025-11-26 | 1961-2024 | seven NetCDF files, 21.55 GB |
| Sunshine duration | [10.5281/zenodo.17471082](https://doi.org/10.5281/zenodo.17471082) | V1.0 | 2025-11-26 | 1961-2024 | seven NetCDF files, 21.55 GB |
| Wind speed | [10.5281/zenodo.22093257](https://doi.org/10.5281/zenodo.22093257) | 1.0 | 2026-08-25 | 1961-2025 | twelve zips, 8.23 GB |

All are CC BY 4.0. None predates the CHM_Drought V1.0 build, and none carries a
version, file checksum or processing statement that ties it to CHM_Drought V1.0.
Using them would change the input, so SPEI and EDDI comparisons would become
product-chain characterization of two different products, not a shared-input
comparison.

### Superseded precipitation records

The CHM_PRE V2 records postdate the product and are not inputs:
[10.5281/zenodo.15735374](https://doi.org/10.5281/zenodo.15735374) (V2.1,
2025-07-07, 46.1 GB) and
[10.5281/zenodo.20378336](https://doi.org/10.5281/zenodo.20378336) (V2.1.1,
2026-05-25, 46.8 GB). They are recorded here so a later session does not mistake
them for the V1.0 input.

## NetCDF metadata audit

Method: each Zenodo archive is a deflate ZIP. The audit read only the ZIP central
directory and an 8 MB prefix of one member per family, through HTTP range
requests, and wrote the prefix into a sparse local file at the declared member
size. No full grid was downloaded. Thirteen global attributes (fifteen in EDDI)
and every variable attribute were then read. The metadata PDF MD5 (`ea225708698e6527175de11a7231d889`)
was verified against the downloaded file.

Global attributes present in every audited file (13 in SPI, SPEI, PDSI, SC-PDSI
and VPD; 15 in EDDI):

`Title`, `Resolution`, `Spatial range`, `Version`, `Drought index name`, `Id`,
`Summary`, `Keywords`, `Institution`, `Data producer`, `Software`, `Reference1`,
`Reference2`; EDDI adds `Reference3` and `Index Creators`.

`Software` is the only processing attribute. It names R packages, nothing more:

- SPI and SPEI: "Created in R using the SPEI package".
- EDDI: "Created in R using the SPEI package; and Evapotranspiration package".
- PDSI and SC-PDSI: "Created in R using the scPDSI package".

Global attributes **absent** from every audited file, recorded as absent and not
inferred:

- Calibration Period;
- distribution name and fitted parameters;
- gamma estimator, zero-mass convention, or clipping rule;
- R version, package versions, or a processing date;
- PET aggregation rule, wind measurement height, or elevation source;
- cell-level AWC source (PDSI and SC-PDSI).

Variable attributes are minimal: `_FillValue = NaNf`, `missing_value = NaNf`,
`units = "Value"` (VPD uses `hpa`), and a `long_name` that sometimes names the PET
method (`SPEI-12`: "Standard Precipitation Evaporation Index_Penman"; `EDDI-12`:
"Evaporative Demand Drought Index_Penman").

Grid and calendar, read from the coordinate variables:

- Longitude 72.05 to 135.95 (640), latitude 18.05 to 53.95 (360), step 0.1.
- `time:units = "days since 1961-01-01"`, `calendar = "standard"`, 744 steps,
  `time[0] = 0` for SPI, SPEI, PDSI, SC-PDSI and VPD; `time[0] = 15` for EDDI.
- Index variables are `float32` with shape `(time, lat, lon)`.

The metadata PDF states that a 12-month file "has values starting from January
1962". With a month-start time axis and a trailing 12-month sum, the library's
first valid 12-month value falls in December 1961. The audit could not read the
series values through the bounded prefix, so the leading fill count and the
window alignment are recorded as unresolved and added to the protocol as an
explicit alignment hypothesis.

## Predeclared comparison protocol

This protocol is fixed before any CHM value is read. Every alternative listed here
is tested and reported; none is selected after seeing agreement.

### Scope

| Item | Declared value |
| --- | --- |
| Index | SPI, gamma distribution, monthly steps; daily steps for the 14-day secondary target |
| Timescales | 12 months (primary); 1, 3 and 6 months (secondary); 14-day SPI as a separate secondary target below |
| Period | 1961-01 to 2022-12 (744 monthly steps; daily series for the 14-day target) |
| Input | Candidate CHM_PRE v4, 0.1-degree precipitation: monthly totals for monthly targets, daily values for the 14-day target. Identity with the product input is unconfirmed |
| Cells | Six 0.1-degree cells, declared below |
| Reference | `CHM_SPI-{1,3,6,12}.nc` and `CHM_SPI-2W.nc` from Zenodo record 14634774, all in `SPI.zip` (MD5 `e7251e9e42b1d4976d64ed0e88fc1833`) |
| Library | `climate_indices` 3.1.0, `indices.spi()` |

These exact grid centers span the aridity gradient and lie inside the domain.
Use these centers on both grids, not rounded city coordinates.

| Name | Longitude (E) | Latitude (N) | Setting |
| --- | --- | --- | --- |
| Guangzhou | 113.25 | 23.05 | humid subtropical |
| Shanghai | 121.45 | 31.15 | humid coastal |
| Beijing | 116.35 | 39.85 | semi-humid |
| Lanzhou | 103.75 | 36.05 | semi-arid |
| Urumqi | 87.55 | 43.75 | arid |
| Lhasa | 91.05 | 29.65 | high-altitude |

The fixture build must verify that both coordinate axes contain each declared
center, allowing only coordinate storage precision. It must assert that each
selected cell is non-NaN in the reference file after the Timescale's leading
fill, and that it lies inside the CHM_PRE land mask. If a cell fails, the build
stops; it does not substitute a cell after seeing results.

### Secondary 2-week SPI target

Retain the daily CHM_PRE v4 series for the same six cells and 1961-2022 period.
Use `utils.transform_to_366day` before `indices.spi()` with `scale=14` and
`periodicity=compute.Periodicity.daily`. Use `utils.transform_to_gregorian`
afterward to remove synthetic February 29 outputs. This tests the library's
366-day calendar, not an assumed match to 14 civil days near February 29.

Read the dates and calendar in `CHM_SPI-2W.nc` before reading index values.
Use the published dates as trailing-window ends under A0; this remains a hypothesis.
Compare only matching dates, without interpolation or a fitted time shift.
If dates or calendar are unavailable, record the blocker and stop only this target.

Report C1-C3, E1-E2 and Z1-Z3 with per-calendar-day fits and the same metrics,
using time steps rather than months. Use A0 only; A1 below is a monthly hypothesis.
Keep these results separate from the monthly characterization.

### Predeclared alternatives

| Axis | Alternatives, all reported |
| --- | --- |
| Calibration Period | C1: 1961-2022 (full record, per calendar month). C2: 1961-1990. C3: 1981-2010 |
| Gamma estimator | E1: the library's formula (the supplement's MLE approximation). E2: unbiased-PWM gamma (the R `SPEI` default, `lmom::pelgam`) |
| Zero Handling | Z1: `classic` (`Phi^-1(p0)`). Z2: `center_of_mass` (`Phi^-1(p0/2)`). Z3: no zero-mass adjustment (a zero maps to the lower tail, then clips) |
| Alignment | A0: the date label is the window end (library trailing sum). A1: monthly target only, a one-month shift if the reference's first valid 12-month value is January 1962 |
| Clipping | Library clip at `+/-3.09` is always applied. Months where either absolute score is at least 3.09 are excluded from residual metrics, but their tolerance agreements and disagreements are counted separately |

C1 is only a hypothesis. No source confirms it; it is the R `SPEI` default when
`ref.start` and `ref.end` are left unset. C2 and C3 are the two 30-year normal
periods a China product most plausibly used; they are included to bound the
answer, not because a source names them.

E2 requires an external gamma fit. The fixture build may compute it in R `SPEI`
(a bounded script, version-pinned) or with an L-moment gamma fit for the fixture.
Either way the fit output, script and versions are committed with the fixture.

### Metrics

For each cell, Timescale and hypothesis combination, report:

- eligible months after the declared leading fill, retained months, coverage
  (`retained / eligible`), and months excluded by clipping or missing data;
- counts of clipped finite pairs within and outside `1e-3`, including zero
  accumulations where only the library clips;
- bias, mean absolute difference, root mean square difference, Pearson
  correlation;
- mean sign agreement, and agreement of the seven-class scheme at cuts
  `+/-1.0`, `+/-1.5`, `+/-2.0` (exact class and within one class);
- maximum absolute difference, and the share of months within `1e-3` and within
  `0.01`.

### Evidence rule

The result is characterization in every case. Apply the 99% threshold only to
retained, unclipped finite pairs, using an absolute tolerance of `1e-3`.
Report a passing configuration as "arithmetic agreement on the unclipped subset
under hypothesis H". Name H and give the retained count, eligible count and
coverage. An empty retained subset cannot qualify.

This label does not establish agreement over the eligible series.
Report excluded clipped disagreements separately; they remain disagreements,
including zeros that Z3 clips when the reference gives a different finite score.
Missing pairs remain untested. Neither subset agreement nor clipped agreement
establishes parity: the input revision and recipe remain unconfirmed.

If no configuration reaches that band, the residual pattern is reported as an
unresolved difference. A structured pattern (by calendar month, aridity or
month length) is evidence about a missing convention, not evidence of a library
defect. Any proposed upstream change still needs its own issue, its own
specification and its own evidence, per the epic's dependency chain.

### Fixture and provenance

- Extract only the six cells' series at all declared Timescales from the archived files.
  Store those series, the cell indices, the archive MD5s and the extraction script.
- Archive MD5s for provenance: `SPI.zip` `e7251e9e42b1d4976d64ed0e88fc1833`,
  `SPEI.zip` `3aeffac0415befa8194dcc96deaef547`, `EDDI.zip`
  `dd6f360662a0a0b504a883bbd2a6d613`, `PDSI.zip` `3cbd057267c392b9f8505109d36265cf`,
  `SC-PDSI.zip` `1dcc5b783eef02c2fc958cb922b91280`, `VPD.zip`
  `2e4a423446f2c9923944fada2b89ae6b`.
- Redistribute the extracted series only if the CC BY 4.0 attribution and the
  product DOI travel with them. Do not redistribute the archives or any CMA
  station observation.

## Unresolved questions that block parity

1. Which R `SPEI` version and which `fit`, `ref.start`, `ref.end` and `kernel`
   arguments produced the SPI and SPEI grids?
2. Which CHM_PRE version was read, v3 or v4?
3. Which build of the interpolated meteorological fields produced the PET grids,
   and how were daily ETo values aggregated to monthly PET?
4. Was any clipping applied to the published values?
5. Which temperature enters the aerodynamic term: the interpolated Tmean field,
   or the daily mean of Tmin and Tmax?
6. At which height was wind measured, and was it converted to 2 m?
7. Which elevation field drives the psychrometric constant?
8. For EDDI, which distribution and Calibration Period produced `P(ET0)`?
9. For PDSI and SC-PDSI, which AWC grid version and which calibration window
   were used?

Questions 1, 2 and 4, plus unresolved SPI window alignment, block SPI parity.
Questions 3 and 5-7 additionally block SPEI, not SPI. The product metadata does
not resolve these questions. No author contact was made; that requires a separate
maintainer decision under the epic. A processing-code release or versioned
metadata update could supply the missing declarations.

## What this audit did not do

- It did not download a full CHM_Drought grid, run any index computation, or
  compare any value.
- It did not repeat the R `SPEI` fixtures, the FAO-56 worked examples, or the
  SPEIbase comparison.
- It did not change the library, its validation claims, or any Earthflux code.
- It did not contact the authors or assert anything about their intent.

A later session that finds a CHM_PRE or met-grid version declaration, a processing
script, or a corrected metadata release should revisit the SPI blockers before
the characterization run. A declared Calibration Period would remove the largest
hypothesis axis.

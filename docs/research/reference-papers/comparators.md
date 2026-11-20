# Comparator survey and pinned versions (REFIMPL-03)

**Ticket:** [REFIMPL-03](https://github.com/monocongo/climate_indices/issues/1605), part of [the reference-paper map](https://github.com/monocongo/climate_indices/issues/1602).
**Surveyed:** 2026-11-18, on macOS arm64 (Darwin 27.2.0), CPython 3.12.13, R 4.6.2.
**Companion file:** [`comparators.lock.yaml`](comparators.lock.yaml) holds the exact pins and hashes.

## How to read this

- This is a survey, not a comparison. Every number below is a **smoke value**: it shows that a comparator installed and produced one index value. No evidence class in `VALIDATION.md` fits a one-month smoke value (there, even `characterization` means measured ceilings), it is not evidence for any claim, and no study ticket may cite it. Each study pre-registers its hypotheses, thresholds, and sample before it runs (see the map's Notes).
- Smoke values are at **L0 (the lowest evidence level), as shipped**: the comparator's own defaults, with the arguments each package offers. Where a package has no calibration-period argument, the difference from the others is expected, not a discrepancy.
- Agreement between two packages is evidence only if they are independent. [REFIMPL-04](https://github.com/monocongo/climate_indices/issues/1606) has not resolved lineage, so every cross-package agreement here is "same lineage not excluded".
- "Headless on Linux" was run only for NCL and the NDMC SPI Generator, which ran in Linux containers (see "Programs run in containers"). Everything else ran on macOS arm64. Several CRAN dependencies compile Fortran and C++ (see the environment notes), so [REFIMPL-08](https://github.com/monocongo/climate_indices/issues/1610) must confirm Linux for them.
- "Committing outputs" rows state the project rule (numeric outputs and provenance only, never comparator code). They are not a legal opinion on output licensing.
- Include/exclude entries are proposals for the maintainer to approve.

## Decision summary

| Family | Candidate | Pin | Decision |
| --- | --- | --- | --- |
| SPI, SPEI, PET (Python) | `xclim` | 1.0.0 | **Include** |
| SPI, SPEI (Python) | `spei` (Vonk) | 0.8.3 | **Include** |
| SPI, SPEI (Python) | `standard_precip` | 2.0.0 | **Include** |
| SPI, SPEI (Python) | `precip-index` | git `1c2ef1c3` (no release) | **Exclude** (built upon `climate_indices` per its README; no release or package) |
| SPI, SPEI (R) | `SPEI` | 1.8.1 | **Include** |
| SPI (R) | `SCI` | 1.0-3 | **Include** |
| SPI (R) | `SEI` | 0.2.0 | **Include** |
| SPI (R) | `climatekit` | 0.2.3 | **Include, provisional** (`ck_spi` run; `ck_spei` and `ck_pet` not examined; very new; no calibration-period argument) |
| SPI, SPEI (R) | `standaRdized` | git `1400bf86` (v1.0) | **Exclude** (unmaintained since 2019, daily-only, not on CRAN) |
| SPI, SPEI, EDDI (service) | Climate Engine API `standard_index` | live, accessed 2026-11-18 | **Include** (live product, not an immutable snapshot) |
| SPI, SPEI, EDDI (service) | Earth Engine `GRIDMET/DROUGHT` | live, accessed 2026-11-18 | **Include** (live product, not an immutable snapshot) |
| SPI (product) | NCEI climdiv SPI | existing fixture, 2026-09-12 | **Include** (already a fixture) |
| SPEI (product) | SPEIbase v2.11 | existing fixture, 2026-09-17 | **Include** (already a fixture) |
| SPEI (product) | SPEI Global Drought Monitor | n/a | **Exclude** (web map viewer; no download or API documentation located; Thornthwaite PET) |
| SPI (program) | NDMC SPI Generator 1.7.6.0 | zip sha256 `ce8f37bc`, run under Mono | **Include** (Windows .NET 4.0 program run in a Linux container under Mono; output rounded to two decimals) |
| SPI (product) | nClimGrid SPI | n/a | **Exclude** (circular: this repository's author maintains the gridded nClimGrid SPI tooling) |
| PET (Python) | `pyet` | 1.5.0 | **Include** (no Thornthwaite) |
| PET (Python) | `PyETo` | git `79ddd88d` (no release) | **Include**, with claimed lineage: `climate_indices.eto` is said to derive from it, unverified ([REFIMPL-04](https://github.com/monocongo/climate_indices/issues/1606)) |
| PET (R) | `Evapotranspiration` | 1.16 | **Include** (no Thornthwaite) |
| Other | NCL `dim_spi_n` | conda-forge 6.6.2, linux-64 build `h7a63124_57` | **Include** (linux-64 build in a container under amd64 emulation; NCL is in maintenance) |
| Other | MATLAB SDAT | n/a | **Exclude** (needs MATLAB; original toolbox not obtained) |

"Include" means installed in a throwaway environment and produced one value, as the ticket requires; the two product rows are existing fixtures and were not re-run. Every exclusion has its reason in the family sections below.

## SPI and SPEI: Python

| | `xclim` | `spei` (Vonk) | `standard_precip` |
| --- | --- | --- | --- |
| Repository | [Ouranosinc/xclim](https://github.com/Ouranosinc/xclim) | [martinvonk/spei](https://github.com/martinvonk/spei) | [e-baumer/standard_precip](https://github.com/e-baumer/standard_precip) |
| Pinned release (PyPI upload) | 1.0.0 (2026-11-05) | 0.8.3 (2026-10-28) | 2.0.0 (2026-09-01) |
| Licence | Apache-2.0 | MIT | Apache-2.0 |
| Python | >= 3.11 | >= 3.11 | >= 3.10 |
| Install route | `uv pip install xclim==1.0.0` | `uv pip install spei==0.8.3` | `uv pip install standard-precip==2.0.0` |
| Indices | SPI, SPEI (also standardized streamflow and groundwater indices) | SPI, SPEI, SGI, SSFI | SPI, SPEI |
| Distributions and estimators | `gamma`, `fisk`, `genextreme`, `lognorm` (or an `rv_continuous`); `ML` for all four (default), `APP` for all but `genextreme` (the location parameter must be fixed), `PWM` only with an `lmoments3` distribution object; `fitkwargs` | any SciPy distribution; `fit_freq`, `fit_window`; `fit_method` `MLE` or `MM` in `SI` | `exp, gam, gev, glo, gno, gpa, gum, kap, nor, pe3, wak, wei`; `fit_type` `lmom` or `mle` (`glo` is documented as generalised logistic: `lmoments3` for `lmom`, `scipy.stats.genlogistic` for `mle`, so the two fit types use different families; there is no `fisk` name) |
| Calibration period | `cal_start`, `cal_end` | none in `spi()`/`spei()`; fits the whole record per `fit_window` | `baseline_start`, `baseline_end` (integer years, strings, or Timestamps) |
| Zero handling | `prob_zero_interpolation` (`upper` default), `plotting_position_zero` (`ecdf` default); SPI zero-inflated, SPEI not; clips at ±8.21 | `prob_zero` (True for `spi`, False for `spei`) | not examined |
| PET | `potential_evapotranspiration`: `BR65` (default), `HG85`, `TW48`, `MB05`, `FAO_PM98`, `DA02` | none (takes a water-balance series) | none (takes a water-balance series) |
| Own stated validation | Docstring (the spelling `fitwkargs` is upstream in xclim 1.0.0): results of `climate_indices` "can be reproduced with `method = "APP"` and `fitwkargs = {"floc": 0}`, except for the maximum and minimum values allowed" | JOSS 10(111) 8454; a published notebook compares packages (from the [research notes](research-notes.md); the notebook was not re-read here) | none found in the README or sdist metadata |
| Headless Linux | not run; no platform-specific code expected | not run | not run |
| Committing outputs | numeric outputs and provenance (Apache-2.0) | numeric outputs and provenance (MIT) | numeric outputs and provenance (Apache-2.0) |

### `precip-index`: excluded

[`bennyistanto/precip-index`](https://github.com/bennyistanto/precip-index) (BSD-3-Clause, created 2026-01-20, head `1c2ef1c304196f74cb662fe6439d5e2c444cc0d8`, 4 stars) states in its README: "Built upon the foundation of climate-indices by James Adams". It has no PyPI package, no conda package, and no tagged release (`gh api .../releases` returns an empty list), so it cannot be installed from a pinned source other than a commit. Its README lists additions (multi-distribution support, bidirectional event analysis, parameter persistence); how much code is shared was not examined, so agreement with `climate_indices` would be same-lineage by its own account. This answers the `precip-index` item in the research notes' "Not verified" list for this one package; the other small packages remain with [REFIMPL-04](https://github.com/monocongo/climate_indices/issues/1606).

## SPI and SPEI: R

| | `SPEI` | `SCI` | `SEI` | `climatekit` | `standaRdized` |
| --- | --- | --- | --- | --- | --- |
| Source | [CRAN](https://cran.r-project.org/package=SPEI), [sbegueria/SPEI](https://github.com/sbegueria/SPEI) | [CRAN](https://cran.r-project.org/package=SCI) | [CRAN](https://cran.r-project.org/package=SEI), [noeliaof/SEI](https://github.com/noeliaof/SEI) | [CRAN](https://cran.r-project.org/package=climatekit) | [WillemMaetens/standaRdized](https://github.com/WillemMaetens/standaRdized) |
| Pin | 1.8.1 (CRAN 2023-03-02) | 1.0-3 (2025-05-04) | 0.2.0 (2024-08-27) | 0.2.3 (2026-11-01) | git `1400bf86c6ed06d3818b1cd99eab33d4e32e942c` (v1.0, 2019-04-17) |
| Licence | GPL-2 | GPL (>= 2) | GPL (>= 2) | MIT + file LICENSE | MIT |
| Indices | SPI, SPEI | SPI (generalized `SCI`) | SPI and other standardized indices | `ck_spi`, `ck_spei`, `ck_pet`, `ck_pet_pm` | standardized index, daily basis |
| Estimators | `ub-pwm` (default), `pp-pwm`, `max-lik`; kernel; `ref.start`, `ref.end` | `fitSCI` / `transformSCI`; `p0` zero-mass mixture; optional Weibull plotting-position estimate via `p0.center.mass` (default FALSE; not used in the smoke call) | MLE via `fitdistrplus`; `cens` modes; `lower` censoring | gamma or Pearson III (SPI); log-logistic or GEV (SPEI) | gamma, gamma3, Weibull, GEV, glogis; `mle` or `lmom` |
| Calibration period | `ref.start`, `ref.end` | fit on a subset, transform the full series | `x_ref` | none | `ref.years`, `ref.length` |
| Own stated validation | Beguería et al. (2014) recommend unbiased PWM (per the research notes); SPEIbase v2.11 was built with this exact version (1.8.1), per the SPEI site | R Journal article for `SEI` names `SCI` | R Journal RJ-2024-038 | none found | none found |
| Headless Linux | not run | not run | not run | not run | not run |
| Committing outputs | numeric outputs and provenance only; GPL code stays outside the package | same | same | numeric outputs and provenance (MIT) | numeric outputs and provenance (MIT) |

`SEI` 0.2.0 and `SCI` 1.0-3 are the versions already pinned in `tests/fixture/zero_handling/provenance.json`, which was generated with R 4.6.1. This survey used R 4.6.2.

### `standaRdized`: excluded

It installs and runs (it needs `FAdist` 2.4 and `glogis` 1.0.3, both from CRAN), so this is a judgment, not a failure. Reasons: it is not on CRAN, its last commit is 2019-04-17, it is built around daily series (`agg.length` in days), and `standardized.index` rounds its output to two decimals by default (`digits = 2`). None of that fits a monthly protocol with the pre-registered 1e-6 "equivalent" class (the study tickets register it; see the map's Notes). The maintainer can reverse this; the smoke value below shows it is runnable.

## PET

| | `xclim` | `pyet` | `PyETo` | R `SPEI` | R `Evapotranspiration` | `climatekit` |
| --- | --- | --- | --- | --- | --- | --- |
| Pin | 1.0.0 | 1.5.0 (2026-05-26) | git `79ddd88d727c7ebdad0ce26e220528d2921b66ea` (2022-08-31) | 1.8.1 | 1.16 (2022-01-10) | 0.2.3 |
| Licence | Apache-2.0 | MIT | BSD-3-Clause | GPL-2 | GPL (>= 2) | MIT |
| Install route | PyPI | PyPI | **git commit only**: not on PyPI (404) and not on conda-forge | CRAN | CRAN | CRAN |
| Thornthwaite | `TW48` | **absent** | `thornthwaite` | `thornthwaite` | **absent** | not examined |
| Hargreaves | `HG85` | `hargreaves` | `hargreaves` | `hargreaves` (not run) | `ET.HargreavesSamani` | not examined (`ck_pet` exists) |
| FAO-56 Penman-Monteith | `FAO_PM98` | `pm_fao56` | `fao56_penman_monteith` | `penman` (not run) | `ET.PenmanMonteith` (exists; not run) | not examined (`ck_pet_pm` exists) |
| Stated validation | not examined | README lists a "Benchmarked?" column and benchmark notebooks against KNMI and CoAgMET data; GMD 17, 7083 (v1.3.1, per the research notes) | none found | Beguería et al. (2014) (estimator recommendation, not a PET validation) | none found | none found |
| Committing outputs | numeric outputs and provenance | same | same | same; GPL code stays outside the package | same | same |

Per the research notes, `xclim` changed its Hargreaves method for consistency with recent literature ([pull request 1723](https://github.com/Ouranosinc/xclim/pull/1723)); that was not re-verified here, and it matters for any Hargreaves comparison.

The ticket's claim that `climate_indices.eto` derives from `PyETo` is not re-verified here; it stands as a lead for [REFIMPL-04](https://github.com/monocongo/climate_indices/issues/1606). Until that is resolved, a PET comparison against `PyETo` is "same lineage not excluded".

## Services and products

| | Climate Engine API | Earth Engine `GRIDMET/DROUGHT` | NCEI climdiv SPI | SPEIbase v2.11 |
| --- | --- | --- | --- | --- |
| Kind | live service, `timeseries/standard_index/coordinates` | live catalog collection | static files | static files |
| Version or snapshot | none; accessed 2026-11-18 | none; accessed 2026-11-18 | fixture downloaded 2026-09-12 | fixture downloaded 2026-09-17; built from CRU TS 4.09 with `SPEI` 1.8.1 |
| Distribution | non-parametric requested; log-logistic and gamma also offered (per the research notes) | non-parametric, 1981-2016, windows 14 days to 5 years (per the catalog entry in the research notes) | compared against `climate_indices` Pearson III in the earlier map; NCEI's own estimator was not re-verified here, and its independence from this codebase is plausible but not airtight (per the fixture's provenance) | log-logistic (per the SPEI site), PET from FAO-56 Penman-Monteith |
| Credentials | `CE_API_KEY` from the environment | OAuth (`earthengine authenticate`) plus `EE_PROJECT` | none | none |
| Terms for committing outputs | not verified; the pilot's rule stands: no raw response committed | not verified; the pilot's rule stands: derived values only, no raw response committed (one sample value below) | U.S. Government work, public domain (per the fixture's provenance) | the fixture records CC-BY 4.0 (Earth Engine catalog) **and** ODbL 1.0 (spei.csic.es, share-alike); the SPEI site states ODbL for data and AGPL for its code, where CRAN lists `SPEI` as GPL-2. The committed subset is an adapted database, so confirm the governing terms for it and before refreshing |
| Existing artifacts | `docs/research/earthengine-climate-engine-pilot.md` | same | `tests/fixture/ncei_spi/provenance.json` | `tests/fixture/speibase/provenance.json` |

One exclusion:

- **SPEI Global Drought Monitor.** The SPEI site describes a real-time system that "provides global SPEI maps and data" at 1 degree, and says it uses Thornthwaite PET where SPEIbase uses Penman-Monteith. <https://spei.csic.es/map/> is a JavaScript viewer, and I located no download or API documentation, so there is nothing pinned to run. That it shares the `SPEI` package with SPEIbase is an inference from the same authors, not something the site states. Excluded for now; reopen if the maintainer has a documented data route.

## Programs run in containers

Two comparators do not run natively on the survey machine (macOS arm64), so each has a Dockerfile under [`containers/`](containers/); the NDMC one also has a runner, `smoke.sh`, and the NCL one has a smoke program, `spi_smoke.ncl`, whose input is described below. They are evidence of how the comparators were run, not harness code: [REFIMPL-08](https://github.com/monocongo/climate_indices/issues/1610) decides where the harness environments live and may relocate or replace them. Docker Desktop 29.9.3 (linux/arm64) was used. Windows containers are not available on macOS, so neither container is a Windows environment.

### NDMC SPI Generator

- **What it is.** NDMC's program page (<https://drought.unl.edu/Monitoring/SPI/SPIProgram.aspx>, "released as of 12/7/2018") links `SPIgenerator.zip` (5,071,776 bytes, last modified 2021-11-11, sha256 `ce8f37bc00bf5309cdbedff2e788f698b1f30b2794e17b0e7aaab5c337efb4bf`). It holds a VB.NET console program `SPIGenerator.exe` (reported as version 1.7.6.0 in its exception log), a C# `StandardPrecipitationIndex.dll`, the source of both, sample input, and sample output. Its documentation is `documentation.docx` in the zip.
- **How it ran.** [`containers/ndmc-spi/Dockerfile`](containers/ndmc-spi/Dockerfile): a Debian stage fetches the zip with a pinned checksum, and the `mono:6.12` image (Mono 6.12.0.182, aarch64) runs the shipped 64-bit executable. Debian's own Mono packages lack the `Microsoft.VisualBasic` runtime, which the `mono:6.12` image carries in its GAC. The container runs with `--network none`.
- **Quirks.** The program joins paths with a Windows backslash, so under Mono each output file is created with a name like `o\NAME.csv`; [`smoke.sh`](containers/ndmc-spi/smoke.sh) strips that prefix. The output directory must already exist, or the program writes `exceptions.txt` and exits 0. Output values are rounded to two decimals.
- **Method, from its C# source.** The gamma fit is Thom's closed-form approximation (`gamm = (1 + sqrt(1 + 4A/3)) / (4A)`, `A = ln(mean) - mean(ln x)`) over the positive values, with the zero probability as a mixture; there is no calibration-period argument (the whole record is used), and the incomplete gamma code cites *Numerical Recipes*. This is the same approximation `climate_indices` documents, which is a lineage question for [REFIMPL-04](https://github.com/monocongo/climate_indices/issues/1606), not independence evidence.
- **Check against the zip's own sample.** On the bundled monthly sample (station 250365, scales 1 and 2), 12 of 17 output files match the shipped 2014 sample output exactly. The other 5 differ in the last printed digit(s): in the SPI file, 39 of 1,776 value cells differ, all in the 2-month column, with a maximum difference of 0.02 and the 1-month column identical. The shipped output predates this program build (the DLL source is dated 2018), so version change and Mono-versus-Windows behaviour are not separable here.
- **Not done.** No Windows run; Wine was not tried.

### NCL `dim_spi_n`

- **What it is.** `dim_spi_n(x, nrt, opt, dim)` computes gamma (default) or Pearson III (`opt@spi_type = 3`) SPI per NCL's documentation. NCL's site carries a "Pivot to Python" banner (its date was not checked). The conda-forge package licence is "Apache-2.0 and LicenseRef-Proprietary-Triangle".
- **How it ran.** [`containers/ncl/Dockerfile`](containers/ncl/Dockerfile) installs `ncl=6.6.2=h7a63124_57` (the latest linux-64 build, uploaded 2025-06-03, sha256 `276f9f148e72904e6154ceec17927f97924d9ecf9c7fd45ba84a92224e94578a`) from conda-forge with micromamba. conda-forge ships NCL only for linux-64, osx-64, and osx-arm64, so on Apple silicon the image runs as `linux/amd64` under emulation. The osx-arm64 build (`h5d9bed2_44`, 3,218 files) installed but contains no `ncl` interpreter, so it is not usable. `spi_smoke.ncl` reads `/work/p.txt`, a flattened text copy of the fixture with `-999` for the ten missing months; no committed script writes that file.
- **Method.** The prototype has no calibration-period argument, so I expect the whole record to be used; I did not confirm that from NCL's source. The smoke script set `_FillValue` on the ten missing months.
- **Not done.** No Pearson III or gamma accuracy check against another implementation beyond the smoke value.

## Other

- **MATLAB SDAT** (Farahmand and AghaKouchak 2015): needs a MATLAB licence and the original toolbox was not obtained. The GitHub repo [hydrocodeir/Standardized-Drought-Analysis-Toolbox](https://github.com/hydrocodeir/Standardized-Drought-Analysis-Toolbox) (GPL-3.0) is a browser and Pyodide dashboard that says it "mirrors the original MATLAB SDAT calculation logic". It is a re-implementation, not the original. Excluded; the maintainer may want to reconsider a non-parametric reference for [REFIMPL-20](https://github.com/monocongo/climate_indices/issues/1622).

## Smoke evidence

Local comparator inputs are repository fixtures, so those runs acquire no new data: `tests/fixture/precips_mm_monthly.npy` (monthly precipitation, mm, January 1895 to December 2017, last 10 months NaN), `tests/fixture/temp_celsius.npy`, latitude 25.2292 (from `tests/conftest.py`), and the repo's `pet_thornthwaite.npy` as the PET for SPEI. The Earth Engine and Climate Engine rows are separate provider calls, described below. Reported month: **June 2005**. Calibration 1981-2010 where the package has the argument. Monthly totals are passed unchanged; for xclim the amounts carry the unit label `mm/d` so that no per-day rescaling applies (see the first finding below).

| Comparator | Call | June 2005 |
| --- | --- | --- |
| `climate_indices` (context) | `indices.spi(..., scale=3, gamma)`; Pearson III | 0.343475; 0.272911 |
| `xclim` 1.0.0 | `standardized_precipitation_index(window=3, dist="gamma", method="APP", fitkwargs={"floc": 0})` | 0.343475 |
| `xclim` | same, `method="ML"` (default) | 0.313204 |
| `xclim` | `standardized_precipitation_evapotranspiration_index(window=3, dist="fisk", method="ML")` | 0.47098 |
| `xclim` | `potential_evapotranspiration(method="TW48")`, mm/month | 169.7647 |
| `spei` 0.8.3 | `si.spi(timescale=3, dist=gamma, fit_freq="MS")`, whole-record fit | 0.152579 |
| `spei` | `si.spei(timescale=3, dist=fisk, fit_freq="MS")`, whole-record fit | 0.115064 |
| `standard_precip` 2.0.0 | `SPI().calculate(scale=3, fit_type="lmom", dist_type="gam", baseline 1981-2010)` | 0.342608 |
| `standard_precip` | same, `fit_type="mle"` | 0.343472 |
| `standard_precip` | `SPEI().calculate(dist_type="glo", fit_type="lmom")` | 0.453756 |
| R `SPEI` 1.8.1 | `spi(scale=3, "Gamma", fit="ub-pwm", ref 1981-2010)` | 0.342608 |
| R `SPEI` | same, `fit="max-lik"` | **NA** (see findings) |
| R `SPEI` | `spei(scale=3, "log-Logistic", fit="ub-pwm")` | 0.453500 |
| R `SPEI` | `thornthwaite(Tave, lat)`, mm/month | 169.466548 |
| R `SCI` 1.0-3 | `fitSCI(time.scale=3, "gamma", p0=TRUE)` on 1981-2010, `transformSCI` on the full series | 0.348829 |
| R `SEI` 0.2.0 | `std_index(gamma, "mle", lower=0, cens="none")` on 3-month sums, reference 1981-2010 | 0.343431 |
| R `climatekit` 0.2.3 | `ck_spi(scale=3, "gamma")`, no calibration argument | 0.194254 |
| NDMC SPI Generator 1.7.6.0 (Mono) | `SPIGenerator.exe -i fixture_monthly_wdates.csv -s 1,3 -t monthly`, whole-record fit, two-decimal output | 0.19 (SPI-3), 1.14 (SPI-1) |
| NCL 6.6.2 (linux-64, amd64 emulation) | `dim_spi_n(p, 3, False, 0)`, gamma, no calibration argument | 0.185046 |
| NCL | same, `opt@spi_type = 3` (Pearson III) | 0.121683 |
| `pyet` 1.5.0 | `hamon(tmean, lat)` (the fixture has no Tmin, Tmax, RH, or wind, so Hargreaves and Penman-Monteith could not run), mm/day | 6.985984 (209.58 mm/month) |
| `PyETo` `79ddd88` | `thornthwaite(monthly_t, monthly_mean_dlh, year)`, mm/month | 169.9446 |
| `climate_indices` PET fixture | `pet_thornthwaite.npy` | 169.2625 |
| R `Evapotranspiration` 1.16 | `ET.HargreavesSamani` on the package's `processeddata`, first month (Mar 2001), mm/month | 112.8587 |
| R `standaRdized` `1400bf86` | `standardized.index(agg.length=90, "gamma", "mle")` on the bundled `Ukkel_RR`, 2018-01-24 | 1.33 (rounded by default) |
| Earth Engine `GRIDMET/DROUGHT` | one image (`20200628`), band `spi90d`, point (-84.9, 30.7) | 0.17 |
| Climate Engine API | `standard_index`, `GRIDMET` `pr`, accumulation 90, non-parametric, calibration 1981-2016, point (-84.9, 30.7), last date 2020-07-30 | 0.3886 |

Provider call budget, stated before the run: **one** Earth Engine point sample and **one** Climate Engine request (one coordinate, one variable, one accumulation; 91 seconds). Credentials came from the environment and were never printed. No raw Climate Engine response is committed. The two provider values are at different dates and carry no comparison meaning. Only the `climate_indices` rows and the NDMC bundled-sample comparison above are reproducible from committed artifacts. The Python and R rows came from one-off commands whose scripts were not saved; the NDMC fixture row needs an input CSV that is not committed; the NCL rows need `/work/p.txt`, which no committed script writes. Rerunning the two provider rows spends the same call budget again.

NCL and the NDMC program ran in the containers described above, on the same fixture series (the NDMC input is a station-name line followed by `m/1/yyyy,value` rows with `-99` for the ten missing months). `standaRdized` used its own bundled data because it is daily-basis.

## Findings that change the "known so far" leads

1. **xclim's `climate_indices` equivalence claim matches at one month (0.343475), and the input convention matters.** With monthly totals passed through, xclim `APP` with `floc=0` equals `climate_indices` gamma SPI-3 at 0.343475 (six decimals). My first run fed monthly mean rates (`P / days_in_month`) and gave 0.353969, a difference of 0.0105, because a 3-month window over rates weights months by length differently from a window over totals. The adapter ([REFIMPL-12](https://github.com/monocongo/climate_indices/issues/1614)) must fix and document one convention. This is one month and one series, not a test of the claim.
2. **`pyet` 1.5.0 and R `Evapotranspiration` 1.16 have no Thornthwaite function**, so among the examined packages the Thornthwaite comparison in [REFIMPL-21](https://github.com/monocongo/climate_indices/issues/1623) has four implementations (`climate_indices`, `xclim`, `PyETo`, R `SPEI`); `climatekit` was not examined for it.
3. **`PyETo` is not installable from PyPI or conda-forge**, and its last commit is 2022-08-31. It can be pinned only by commit.
4. **`precip-index` states that it is built upon `climate_indices`** (lineage evidence for [REFIMPL-04](https://github.com/monocongo/climate_indices/issues/1606)).
5. **SPEIbase v2.11 was built with R `SPEI` 1.8.1**, the version pinned here, so the repository's existing SPEIbase fixture and this comparator share a code version.
6. **R `SPEI` (`ub-pwm`) and `standard_precip` (`lmom`) coincide to six decimals (0.342608) on gamma SPI-3**, and `SEI` MLE (0.343431), `standard_precip` MLE (0.343472), and `climate_indices` (0.343475) lie within about 5e-5. The first pair is consistent with one estimator under two names; whether that reflects shared code is a lineage question for REFIMPL-04, not independence evidence.
7. **The NDMC SPI Generator runs on Linux under Mono**, uses the same Thom closed-form gamma approximation that `climate_indices` documents, and prints two decimals. For June 2005 on the fixture (whole-record fit, no calibration argument) it gives 0.19 and NCL gamma gives 0.185046, which round to the same value; that is consistent with one estimator in two programs, and says nothing about independence.
8. **conda-forge's osx-arm64 `ncl` build has no interpreter, but the linux-64 build works**, so the NCL comparator needs `linux/amd64` (emulated on Apple silicon).

## Candidate ledger entries for [REFIMPL-24](https://github.com/monocongo/climate_indices/issues/1626)

Proposed, not yet logged on that ticket, because the ledger's format belongs to it:

| # | Observation | Status |
| --- | --- | --- |
| 1 | R `SPEI` 1.8.1 `spi(fit="max-lik")` and `fit="pp-pwm"` for the gamma distribution return NA coefficients and an all-NA series on a NaN-free 1,466-month series (`ub-pwm` works). Cause not diagnosed; it may be dependency drift on R 4.6.2 (`lmomco` 2.5.8, `lmom` 3.3, `TLMoments` 0.7.5.3), not a package defect | open |
| 2 | Thornthwaite PET for June 2005 differs across packages: 169.2625 (`climate_indices` fixture), 169.4665 (R `SPEI`), 169.7647 (`xclim`), 169.9446 (`PyETo`) mm/month, a single-month spread of 0.68 mm (0.4%). Not aligned for daylength or declination conventions. Feeds [REFIMPL-21](https://github.com/monocongo/climate_indices/issues/1623) | open |
| 3 | `standaRdized` rounds its output to two decimals by default | noted; package excluded |
| 4 | NDMC SPI Generator 1.7.6.0 under Mono reproduces the zip's 2014 sample output except 39 of 1,776 SPI cells (all the 2-month column, maximum difference 0.02). Version change versus Mono-versus-Windows not separable. The program also prints only two decimals, which a 1e-6 "equivalent" class cannot resolve; the C# DLL returns doubles and its source ships in the zip | open |

## Environment notes

- The throwaway environments lived outside the repository. Python: one `uv venv --python 3.12` per package. R: a private library with `R_LIBS_USER`.
- **Local R toolchain fault.** On this machine R's `Makeconf` points `FLIBS` at `.../gcc/current/gcc/aarch64-apple-darwin27/16`, but Homebrew's gcc ships `aarch64-apple-darwin25`. Source builds of packages with Fortran (`lmom`, `deSolve`, `RcppArmadillo`) fail at link time with `ld: library 'emutls_w' not found`. A scratch `R_MAKEVARS_USER` file that overrides `FLIBS` fixed it without changing any global R configuration. A Linux CI image will not have this fault, but anyone reproducing on this class of Mac will.
- Container files live under `docs/research/reference-papers/containers/`. Both Dockerfiles were built and run on this machine; the NDMC one runs with `--network none` because the zip is third-party and untrusted.
- `comparators.lock.yaml` records direct pins and the key resolved dependencies. It is not a full lockfile: [REFIMPL-08](https://github.com/monocongo/climate_indices/issues/1610) owns the harness environments and their locking.
- GitHub archive tarball hashes are recorded for the two git-pinned packages but are not stable guarantees (GitHub can regenerate archives). The commit SHA is the pin.

## Deferred

| Piece | Owner |
| --- | --- |
| Adapters for each included comparator | [REFIMPL-12](https://github.com/monocongo/climate_indices/issues/1614) to [REFIMPL-15](https://github.com/monocongo/climate_indices/issues/1617) |
| Palmer comparators | [REFIMPL-30](https://github.com/monocongo/climate_indices/issues/1632) |
| Implementation lineage and independence | [REFIMPL-04](https://github.com/monocongo/climate_indices/issues/1606) |
| Contacting comparator maintainers and data producers | [REFIMPL-06](https://github.com/monocongo/climate_indices/issues/1608) |
| Linux headless confirmation for the Python and R comparators (NCL and NDMC were run on Linux in containers), harness environments, locking, and where the `containers/` files end up | [REFIMPL-08](https://github.com/monocongo/climate_indices/issues/1610) |
| Provider terms for committing Climate Engine and Earth Engine outputs | [REFIMPL-06](https://github.com/monocongo/climate_indices/issues/1608); its contact table names the Climate Engine team, not Google, so Earth Engine terms need adding |
| Adapters for R `Evapotranspiration`, `climatekit`, NCL, and the NDMC program, which REFIMPL-12 to REFIMPL-15 do not name | maintainer decision, or extend [REFIMPL-13](https://github.com/monocongo/climate_indices/issues/1615) |

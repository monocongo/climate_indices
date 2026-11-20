# Implementation lineage and independence (REFIMPL-04)

**Ticket:** [REFIMPL-04](https://github.com/monocongo/climate_indices/issues/1606), part of [the reference-paper map](https://github.com/monocongo/climate_indices/issues/1602).
**Comparator set and pins:** [`comparators.md`](comparators.md) and [`comparators.lock.yaml`](comparators.lock.yaml) (REFIMPL-03); the Palmer set from REFIMPL-30 ([#1632](https://github.com/monocongo/climate_indices/issues/1632)).
**Machine-readable form:** [`lineage.json`](lineage.json).

## Method

This section was committed before any comparator source was read, so the classes and the evidence standard could not be chosen to fit the findings.

### Classes

Each pair of implementations is classified per index. The classes, strongest first:

| Class | Meaning |
| --- | --- |
| *derived* | One implementation's code for this index was written from, ported from, or forked from the other's code, as stated in its source, documentation, changelog, or commit history. Directed: `a` derived from `b`. |
| *validated against* | One implementation's tests, documentation, or changelog compare its output with the other's, or state that it was changed to reproduce the other, with no shared code. Directed: `a` validated against `b`. |
| *shared numerical routine* | Both call the same code for a step that sets the index value: a distribution-parameter estimator, an L-moment or probability-weighted-moment routine, a plotting-position or empirical-probability routine, or a whole stage such as a PET formula or a Palmer recursion. A common dependency or a common port both count. |
| *independent* | The source of both was read at the pin and none of the above was found. |
| *unknown* | The source of at least one was not available or not read, and its documentation does not settle the question. |

When several relations hold, the pair takes the strongest class, and every relation found is still recorded as evidence. A relation that holds only for one configuration (one distribution, estimator, or method option) is recorded with that configuration, and the pair takes the class in the configurations that the study tickets use.

These do not count as lineage:

- Implementing the same published equation or algorithm. That is what the comparison measures.
- Standard special functions and elementwise arithmetic: the normal quantile, the gamma and incomplete-gamma functions, and the CDF of a distribution once its parameters are fixed, whether from SciPy, Cephes, R's `nmath`, or *Numerical Recipes*. These are recorded where seen but do not change the class.
- Similar or identical output. Agreement is never evidence of lineage.

### Evidence standard

- Every claim cites a file and line at the pinned version (sdist or tarball path, or repository commit), with a short quote. Services and products whose code is not public are cited by documentation URL and access date, and stay *unknown* unless the documentation names the code they run.
- `climate_indices` is cited at the commit this document is written against.
- The leads in the ticket and in [`research-notes.md`](research-notes.md) are verified or refuted, never carried over. A refuted lead is logged in the REFIMPL-24 discrepancy ledger ([#1626](https://github.com/monocongo/climate_indices/issues/1626)).
- Each *unknown* cell becomes a question for REFIMPL-06 ([#1608](https://github.com/monocongo/climate_indices/issues/1608)).

### Implementations and pairs

Pairs are every combination of the implementations listed for an index. PET is split by method because a package can share code for one method and not another. Standard Palmer and scPDSI are separate indices because the library implements them from different recursions.

| Index | Implementations |
| --- | --- |
| SPI | `climate_indices`, `xclim`, `spei`, `standard_precip`, R `SPEI`, R `SCI`, R `SEI`, R `climatekit`, NDMC SPI Generator, NCL `dim_spi_n`, Climate Engine API, Earth Engine `GRIDMET/DROUGHT`, NCEI climdiv SPI |
| SPEI | `climate_indices`, `xclim`, `spei`, `standard_precip`, R `SPEI`, R `SEI`, Climate Engine API, Earth Engine `GRIDMET/DROUGHT`, SPEIbase v2.11 |
| PET, Thornthwaite | `climate_indices`, `xclim` (`TW48`), `PyETo`, R `SPEI` |
| PET, Hargreaves | `climate_indices`, `xclim` (`HG85`), `pyet`, `PyETo`, R `Evapotranspiration`, R `SPEI` |
| PET, FAO-56 Penman-Monteith | `climate_indices`, `xclim` (`FAO_PM98`), `pyet`, `PyETo`, R `Evapotranspiration`, R `SPEI` |
| EDDI and non-parametric | `climate_indices`, NOAA PSL EDDI, Climate Engine API, Earth Engine `GRIDMET/DROUGHT` |
| PDSI, PHDI, PMDI, Z-Index | `climate_indices` (`pdsi`), NCEI `pdi.f`, nClimDiv, `JonKing93/pdsi` |
| scPDSI | `climate_indices` (`scpdsi`), Wells, Goddard and Hayes C++, `Sibada/scPDSI`, `ktripa/scpdsi-tools` |

`precip-index` is excluded from the studies (REFIMPL-03) and is recorded only as lineage evidence for the question of which packages derive from `climate_indices`.

**Amendment after the method commit.** R `SPEI` was added to the three PET rows because REFIMPL-21 ([#1623](https://github.com/monocongo/climate_indices/issues/1623)) lists its `thornthwaite`, `hargreaves`, and `penman` functions. The classes and the evidence standard are unchanged.

**How the method was applied.** NCEI climdiv SPI, nClimDiv, and SPEIbase are products, but their publishers name the code behind them: NCEI publishes `spi.f` and `pdi.f` as the "general use version" of its operational code, and spei.csic.es says SPEIbase v2.11 was computed with R `SPEI` 1.8.1. That code was read in place of the products. The Climate Engine API, Earth Engine `GRIDMET/DROUGHT`, and NOAA PSL EDDI name a method but no code, so their pairs stay *unknown* unless this repository compared against them.

## Result

The eight indices give 168 pairs: 12 *derived*, 20 *validated against*, 30 *shared numerical routine*, 69 *independent*, and 37 *unknown*. Every *unknown* pair involves one of the three providers whose code is not public.

| Index | Pairs | derived | validated against | shared numerical routine | independent | unknown |
| --- | --- | --- | --- | --- | --- | --- |
| SPI | 78 | 2 | 9 | 19 | 27 | 21 |
| SPEI | 36 | 2 | 6 | 11 | 4 | 13 |
| PET, Thornthwaite | 6 | 1 | 0 | 0 | 5 | 0 |
| PET, Hargreaves | 15 | 1 | 1 | 0 | 13 | 0 |
| PET, FAO-56 Penman-Monteith | 15 | 0 | 1 | 0 | 14 | 0 |
| EDDI and non-parametric | 6 | 0 | 3 | 0 | 0 | 3 |
| PDSI, PHDI, PMDI, Z-Index | 6 | 3 | 0 | 0 | 3 | 0 |
| scPDSI | 6 | 3 | 0 | 0 | 3 | 0 |

### Findings

1. **One L-moment code base sits under most parameter fits.** J. R. M. Hosking's Fortran L-moment routines (IBM Research Report RC20525) reach the comparison by four routes: R `lmom`, Hosking's own package, which R `SPEI` calls for its default PWM fits and R `SEI` for `method = "lmoment"`; Python `lmoments3`, a translation of `lmoments.f`, which `standard_precip` uses for `fit_type = "lmom"` and `xclim` for `method = "PWM"`; `climate_indices.lmoments`, a translation of `SAMLMR`, the Pearson III routine, and `PELGLO`; and the upper-case routines inside NCEI's `spi.f` and NCL's Pearson III path. In an L-moment configuration, agreement between any two of these is agreement of one estimator's code. The six-decimal match between R `SPEI` `ub-pwm` and `standard_precip` `lmom` recorded in `comparators.md` is consistent with this, and `standard_precip` also says it is "loosely based on" R `SPEI`.
2. **The two Palmer recursions share an NCDC ancestor.** The Wells, Goddard and Hayes C++ says it is "based mostly on the source code of pdsi.f", a Fortran program from NCDC (`scPDSI/src/scpdsi.cpp`:39-41), and `pdi.f` says it is "a modified version of the program in use at the National Climatic Data Center" (`pdi.f`:13-16). The `pdi.f` and Wells lineages that REFIMPL-29 compared are therefore branches of one NCDC Fortran program, not independent implementations. Their 6.3% disagreement measures the Wells modifications. Neither source says whether Wells's `pdsi.f` and the "old version" in `pdi.f` are the same file (question 4). Within each Palmer row, `climate_indices` is *derived* from its reference: `pdsi` from the `pdi.f` lineage, and `scpdsi` from the Wells C++ through `Sibada/scPDSI`. `JonKing93/pdsi` ("based on a script written by Dave Meko", `README.md`:75) and `ktripa/scpdsi-tools` ("Self-calibration follows Wells et al. (2004); the Palmer water balance and recursion follow Palmer (1965)", `scpdsi_tools.py`:14-15) state no code ancestry. They are the only *independent* Palmer comparators.
3. **NCEI climdiv SPI is not independent of the library or of NCL.** The library's Pearson III path and NCEI's `spi.f` both use Hosking's routines, and a 2018 commit says the library's earlier Pearson code "was originally translated from NCDC global SPI". NCL's Pearson III path (`opt@spi_type = 3`) embeds the NCDC `SPIPE3` code, under Ned Guttman's header, so NCL and the NCEI product are *derived*. This narrows the "plausible but not airtight" independence recorded in `tests/fixture/ncei_spi/provenance.json`: the operational SPI and the library share a parameter-estimation routine.
4. **Thom's approximation is shared as a formula, not as code.** `climate_indices`, `xclim` (`APP`), `climatekit`, the NDMC SPI Generator, and NCL's gamma path all compute Thom's (1958) closed form. None cites another's code, so under the method this is not lineage. The NDMC and NCL gamma paths do share code, but not with the library: their comment blocks are identical, including the misspelling "Abromowitz" and "Thanks to kenny@cs.uiuc.edu", so their inverse-normal and incomplete-gamma routines come from one source that neither names (question 6). `climatekit` says its Thom fit is "consistent with the SPEI R package" (`R/drought.R`:444-446). R `SPEI` fits gamma with `lmom::pelgam` on PWM L-moments (`R/spei.R`:665-690), not with Thom's approximation, so that statement is a documentation claim, not lineage.
5. **`xclim` was tuned to `climate_indices`.** Its tests take gamma `APP` reference values from the library, its SPI docstring says the library's results can be reproduced, and its changelog records that the SPEI offset was changed "to better reproduce results in the reference library". Its log-logistic SPEI reference values come from R `SPEI`. Agreement between `xclim` and either package is *validated against*, not independent.
6. **PET.** The library's `eto` module (Thornthwaite and Hargreaves) is *derived* from `PyETo`. The module credits PyETo, the initial 2017 commit carried the same credit, and the Hargreaves docstring sentence matches PyETo's word for word. `pm_eto` (added 2026) carries no such credit. `pyet` tests its Hargreaves and FAO-56 values against R `Evapotranspiration`. No other PET pair has a stated relation. `xclim`'s PET tests use values with no stated source.
7. **This repository's own comparisons.** The library's tests or research documents compare its output with R `SPEI`, SPEIbase, R `SEI`, R `SCI`, NCEI climdiv SPI, NOAA PSL EDDI, nClimDiv, `Sibada/scPDSI`, and both providers. A comparison does not mean the library was changed to match. It does mean that agreement on those pairs is not independent confirmation.

### Leads checked

| Lead (ticket or research notes) | Verdict |
| --- | --- |
| `xclim` reproduces `climate_indices` with `APP` and `floc = 0`, and changed its SPEI offset to match | Confirmed: *validated against* |
| `climate_indices.lmoments.fit_glo` ports the `lmom` `PELGLO` routine that R `SPEI` uses | Confirmed: *shared numerical routine*, and the library is *validated against* R `SPEI` |
| `climate_indices.eto` derives from `PyETo` | Confirmed for Thornthwaite and Hargreaves; not for `pm_eto` |
| Standard Palmer follows `pdi.f`, scPDSI follows Wells, `Sibada/scPDSI` wraps the Wells C++ | Confirmed, but the Wells C++ itself derives from NCDC Fortran (finding 2) |
| NCEI climdiv SPI's independence is "plausible but not airtight" | Narrowed: it shares Hosking's routines with the library, and the library's pre-2018 Pearson code was translated from NCDC (finding 3) |
| `precip-index` is built on `climate_indices` | Confirmed: its `README.md`, line 42 at `1c2ef1c3`, says "Built upon the foundation of climate-indices" |
| Other included packages derive from `climate_indices` | None found. `xclim` and `spei` compare against it, but no included package states that it derives from it |
| The NDMC program uses the library's Thom approximation (a lineage question in `comparators.md`) | A shared formula, not shared code. The NDMC program shares code with NCL instead (finding 4) |
| R `SPEI` `ub-pwm` and `standard_precip` `lmom` coincide to six decimals | Consistent with shared Hosking-derived routines (finding 1) |
| Provenance of the repository's Thornthwaite PET fixture (`pet_thornthwaite.npy`) | Not resolved. The earliest trace is `5e87424c` (2018-12-11), which moved arrays out of a `tests/fixtures.py` whose content was stripped from history (question 7) |

## Pairs by index

Each table lists the pairs that are not *independent*. An arrow points from the implementation that derives from, or validates against, the other. Citations for every relation are in [Evidence](#evidence) and in `lineage.json`.

### SPI

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / xclim | *validated against* | *validated against* (xclim → climate_indices: test reference values for gamma APP); *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; xclim method = "PWM" with an lmoments3 distribution) |
| climate_indices / spei (Vonk) | *validated against* | *validated against* (spei (Vonk) → climate_indices: documentation only; the comparison code is commented out at v0.8.3) |
| climate_indices / standard_precip | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; standard_precip fit_type = "lmom") |
| climate_indices / R SPEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; R SPEI pp-pwm and ub-pwm fits (default)) |
| climate_indices / R SCI | *validated against* | *validated against* (climate_indices → R SCI: zero placement) |
| climate_indices / R SEI | *validated against* | *validated against* (climate_indices → R SEI: zero placement); *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; R SEI method = "lmoment") |
| climate_indices / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; NCL dim_spi_n Pearson III path (opt@spi_type = 3)) |
| climate_indices / Climate Engine API | *validated against* | *validated against* (climate_indices → Climate Engine API: characterization pilot) |
| climate_indices / Earth Engine GRIDMET/DROUGHT | *validated against* | *validated against* (climate_indices → Earth Engine GRIDMET/DROUGHT: characterization pilot) |
| climate_indices / NCEI climdiv SPI | *validated against* | *validated against* (climate_indices → NCEI climdiv SPI: Pearson III); *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; NCEI climdiv SPI Pearson III (public spi.f)) |
| xclim / spei (Vonk) | *shared numerical routine* | *shared numerical routine* (maximum likelihood: scipy.stats rv_continuous.fit) |
| xclim / standard_precip | *shared numerical routine* | *shared numerical routine* (maximum likelihood: scipy.stats); *shared numerical routine* (Hosking L-moment routines: xclim method = "PWM" with an lmoments3 distribution; standard_precip fit_type = "lmom") |
| xclim / R SPEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: xclim method = "PWM" with an lmoments3 distribution; R SPEI pp-pwm and ub-pwm fits (default)) |
| xclim / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: xclim method = "PWM" with an lmoments3 distribution; R SEI method = "lmoment") |
| xclim / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: xclim method = "PWM" with an lmoments3 distribution; NCL dim_spi_n Pearson III path (opt@spi_type = 3)) |
| xclim / Climate Engine API | *unknown* | code not public |
| xclim / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| xclim / NCEI climdiv SPI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: xclim method = "PWM" with an lmoments3 distribution; NCEI climdiv SPI Pearson III (public spi.f)) |
| spei (Vonk) / standard_precip | *validated against* | *validated against* (spei (Vonk) → standard_precip: documentation notebook); *shared numerical routine* (maximum likelihood: scipy.stats) |
| spei (Vonk) / R SPEI | *validated against* | *validated against* (spei (Vonk) → R SPEI: documentation notebook) |
| spei (Vonk) / Climate Engine API | *unknown* | code not public |
| spei (Vonk) / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| standard_precip / R SPEI | *derived* | *derived* (standard_precip → R SPEI: stated as "loosely based on"); *shared numerical routine* (Hosking L-moment routines: standard_precip fit_type = "lmom"; R SPEI pp-pwm and ub-pwm fits (default)) |
| standard_precip / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: standard_precip fit_type = "lmom"; R SEI method = "lmoment") |
| standard_precip / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: standard_precip fit_type = "lmom"; NCL dim_spi_n Pearson III path (opt@spi_type = 3)) |
| standard_precip / Climate Engine API | *unknown* | code not public |
| standard_precip / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| standard_precip / NCEI climdiv SPI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: standard_precip fit_type = "lmom"; NCEI climdiv SPI Pearson III (public spi.f)) |
| R SPEI / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: R SPEI pp-pwm and ub-pwm fits (default); R SEI method = "lmoment") |
| R SPEI / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: R SPEI pp-pwm and ub-pwm fits (default); NCL dim_spi_n Pearson III path (opt@spi_type = 3)) |
| R SPEI / Climate Engine API | *unknown* | code not public |
| R SPEI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| R SPEI / NCEI climdiv SPI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: R SPEI pp-pwm and ub-pwm fits (default); NCEI climdiv SPI Pearson III (public spi.f)) |
| R SCI / R SEI | *shared numerical routine* | *shared numerical routine* (maximum likelihood: fitdistrplus) |
| R SCI / Climate Engine API | *unknown* | code not public |
| R SCI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| R SEI / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: R SEI method = "lmoment"; NCL dim_spi_n Pearson III path (opt@spi_type = 3)) |
| R SEI / Climate Engine API | *unknown* | code not public |
| R SEI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| R SEI / NCEI climdiv SPI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: R SEI method = "lmoment"; NCEI climdiv SPI Pearson III (public spi.f)) |
| R climatekit / Climate Engine API | *unknown* | code not public |
| R climatekit / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| NDMC SPI Generator / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (gamma path: identical comments, including the misspelling "Abromowitz", show a common source for the inverse normal and incomplete gamma code) |
| NDMC SPI Generator / Climate Engine API | *unknown* | code not public |
| NDMC SPI Generator / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| NCL dim_spi_n / Climate Engine API | *unknown* | code not public |
| NCL dim_spi_n / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| NCL dim_spi_n / NCEI climdiv SPI | *derived* | *derived* (NCL dim_spi_n → NCEI climdiv SPI: Pearson III path (opt@spi_type = 3); via NCEI's public spi.f, which NCEI says is not an exact copy of its operational code); *shared numerical routine* (Hosking L-moment routines: NCL dim_spi_n Pearson III path (opt@spi_type = 3); NCEI climdiv SPI Pearson III (public spi.f)) |
| Climate Engine API / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| Climate Engine API / NCEI climdiv SPI | *unknown* | code not public |
| Earth Engine GRIDMET/DROUGHT / NCEI climdiv SPI | *unknown* | code not public |

The other 27 of 78 pairs are *independent*.

### SPEI

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / xclim | *validated against* | *validated against* (xclim → climate_indices: offset changed to reproduce climate_indices); *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; xclim method = "PWM" with an lmoments3 distribution) |
| climate_indices / standard_precip | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; standard_precip fit_type = "lmom") |
| climate_indices / R SPEI | *validated against* | *validated against* (climate_indices → R SPEI: log-logistic); *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; R SPEI pp-pwm and ub-pwm fits (default)) |
| climate_indices / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; R SEI method = "lmoment") |
| climate_indices / Climate Engine API | *validated against* | *validated against* (climate_indices → Climate Engine API: characterization pilot) |
| climate_indices / Earth Engine GRIDMET/DROUGHT | *validated against* | *validated against* (climate_indices → Earth Engine GRIDMET/DROUGHT: characterization pilot) |
| climate_indices / SPEIbase v2.11 | *validated against* | *validated against* (climate_indices → SPEIbase v2.11: plausibility check); *shared numerical routine* (Hosking L-moment routines: climate_indices Pearson III, log-logistic and GEV fits; SPEIbase v2.11 ub-pwm log-logistic, through R SPEI) |
| xclim / spei (Vonk) | *shared numerical routine* | *shared numerical routine* (maximum likelihood: scipy.stats rv_continuous.fit) |
| xclim / standard_precip | *shared numerical routine* | *shared numerical routine* (maximum likelihood: scipy.stats); *shared numerical routine* (Hosking L-moment routines: xclim method = "PWM" with an lmoments3 distribution; standard_precip fit_type = "lmom") |
| xclim / R SPEI | *validated against* | *validated against* (xclim → R SPEI: fisk ML); *shared numerical routine* (Hosking L-moment routines: xclim method = "PWM" with an lmoments3 distribution; R SPEI pp-pwm and ub-pwm fits (default)) |
| xclim / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: xclim method = "PWM" with an lmoments3 distribution; R SEI method = "lmoment") |
| xclim / Climate Engine API | *unknown* | code not public |
| xclim / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| xclim / SPEIbase v2.11 | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: xclim method = "PWM" with an lmoments3 distribution; SPEIbase v2.11 ub-pwm log-logistic, through R SPEI) |
| spei (Vonk) / standard_precip | *shared numerical routine* | *shared numerical routine* (maximum likelihood: scipy.stats) |
| spei (Vonk) / Climate Engine API | *unknown* | code not public |
| spei (Vonk) / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| standard_precip / R SPEI | *derived* | *derived* (standard_precip → R SPEI: stated as "loosely based on"); *shared numerical routine* (Hosking L-moment routines: standard_precip fit_type = "lmom"; R SPEI pp-pwm and ub-pwm fits (default)) |
| standard_precip / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: standard_precip fit_type = "lmom"; R SEI method = "lmoment") |
| standard_precip / Climate Engine API | *unknown* | code not public |
| standard_precip / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| standard_precip / SPEIbase v2.11 | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: standard_precip fit_type = "lmom"; SPEIbase v2.11 ub-pwm log-logistic, through R SPEI) |
| R SPEI / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: R SPEI pp-pwm and ub-pwm fits (default); R SEI method = "lmoment") |
| R SPEI / Climate Engine API | *unknown* | code not public |
| R SPEI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| R SPEI / SPEIbase v2.11 | *derived* | *derived* (SPEIbase v2.11 → R SPEI: SPEIbase is computed with the package); *shared numerical routine* (Hosking L-moment routines: R SPEI pp-pwm and ub-pwm fits (default); SPEIbase v2.11 ub-pwm log-logistic, through R SPEI) |
| R SEI / Climate Engine API | *unknown* | code not public |
| R SEI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| R SEI / SPEIbase v2.11 | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines: R SEI method = "lmoment"; SPEIbase v2.11 ub-pwm log-logistic, through R SPEI) |
| Climate Engine API / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| Climate Engine API / SPEIbase v2.11 | *unknown* | code not public |
| Earth Engine GRIDMET/DROUGHT / SPEIbase v2.11 | *unknown* | code not public |

The other 4 of 36 pairs are *independent*.

### PET Thornthwaite

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / PyETo | *derived* | *derived* (climate_indices → PyETo) |

The other 5 of 6 pairs are *independent*.

### PET Hargreaves

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / PyETo | *derived* | *derived* (climate_indices → PyETo) |
| pyet / R Evapotranspiration | *validated against* | *validated against* (pyet → R Evapotranspiration) |

The other 13 of 15 pairs are *independent*.

### PET FAO-56 Penman-Monteith

| Pair | Class | Relation found |
| --- | --- | --- |
| pyet / R Evapotranspiration | *validated against* | *validated against* (pyet → R Evapotranspiration) |

The other 14 of 15 pairs are *independent*.

### EDDI and non-parametric

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / NOAA PSL EDDI | *validated against* | *validated against* (climate_indices → NOAA PSL EDDI) |
| climate_indices / Climate Engine API | *validated against* | *validated against* (climate_indices → Climate Engine API: characterization pilot) |
| climate_indices / Earth Engine GRIDMET/DROUGHT | *validated against* | *validated against* (climate_indices → Earth Engine GRIDMET/DROUGHT: characterization pilot) |
| NOAA PSL EDDI / Climate Engine API | *unknown* | code not public |
| NOAA PSL EDDI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |
| Climate Engine API / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public |

No pair is *independent*.

### PDSI, PHDI, PMDI, Z-Index

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / NCEI pdi.f | *derived* | *derived* (climate_indices → NCEI pdi.f) |
| climate_indices / nClimDiv (NCEI) | *derived* | *derived* (climate_indices → nClimDiv (NCEI): through the NCEI Fortran lineage; also validated against nClimDiv) |
| NCEI pdi.f / nClimDiv (NCEI) | *derived* | *derived* (NCEI pdi.f → nClimDiv (NCEI): pdi.f is a modified public version of NCDC's operational program; the operational code is not public) |

The other 3 of 6 pairs are *independent*.

### scPDSI

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / Wells, Goddard and Hayes C++ | *derived* | *derived* (climate_indices → Wells, Goddard and Hayes C++: through Sibada/scPDSI) |
| climate_indices / Sibada/scPDSI | *derived* | *derived* (climate_indices → Sibada/scPDSI: also validated against it (oracle fixture)) |
| Wells, Goddard and Hayes C++ / Sibada/scPDSI | *derived* | *derived* (Sibada/scPDSI → Wells, Goddard and Hayes C++) |

The other 3 of 6 pairs are *independent*.

## Questions for REFIMPL-06

These questions cover every *unknown* pair, plus the residual questions behind some classified ones. They are for REFIMPL-06 ([#1608](https://github.com/monocongo/climate_indices/issues/1608)), except question 7, which is for the maintainer.

1. **Climate Engine API.** Which code computes `standard_index` SPI, SPEI, and EDDI, and does it share code with Earth Engine `GRIDMET/DROUGHT` or NOAA PSL EDDI? It covers 20 *unknown* pairs.
2. **Earth Engine `GRIDMET/DROUGHT`** (producer: University of California Merced). Which code produces it, and which plotting-position formula does it use? The catalog says only that "plotting positions are used". It covers 20 *unknown* pairs, 3 of them shared with question 1.
3. **NOAA PSL EDDI.** Is the operational code available, and do the Climate Engine or `GRIDMET/DROUGHT` EDDI use it? Hobbins et al. (2016) lists J. L. Huntington and C. Morton among its authors (`src/climate_indices/cf_metadata_registry.py`:327). The research notes cite Huntington et al. (2017) for Climate Engine, a reference REFIMPL-05 has yet to verify. It covers the 2 *unknown* pairs between PSL and the providers.
4. **NCEI.** Does the operational climdiv SPI run `spi.f`'s `SPIPE3` code, and does operational nClimDiv run the `pdi.f` lineage? Is the `pdsi.f` that the Wells C++ was based on the same NCDC program as the "old version" in `pdi.f`?
5. **Dave Meko** (via `JonKing93/pdsi`). Was the MATLAB PDSI script written from NCDC Fortran? A yes would change the `JonKing93/pdsi` pairs from *independent* to *derived*.
6. **NDMC and NCAR.** What is the common source of the NDMC SPI Generator's and NCL's gamma code?
7. **Maintainer.** What was the "NCDC global SPI" code that commit `d106d578` replaced, and do the Hosking translations in `lmoments.py`, which predate that commit, come from it? Where did `tests/fixture/pet_thornthwaite.npy` come from?

## Figure specification for REFIMPL-40

For REFIMPL-40 ([#1642](https://github.com/monocongo/climate_indices/issues/1642)):

- **Data.** `lineage.json` only. No edge is drawn by hand.
- **Layout.** One panel per index, or five grouped panels: SPI and SPEI; PET; EDDI; standard Palmer; scPDSI.
- **Nodes.** One per implementation in the index's `implementations` list. Shape shows the kind (package, program, product, service). Implementations with `source_read: false` get a dashed outline.
- **Edges.** One per entry in `pairs[].relations`. *Derived*: solid arrow from `from` to `to`. *Validated against*: dashed arrow from `from` to `to`. *Shared numerical routine*: undirected dotted line, labelled with the routine (Hosking L-moments, SciPy maximum likelihood, `fitdistrplus`, the NDMC and NCL gamma source). *Independent* and *unknown* pairs get no edge.
- **Ancestors.** Optional context nodes, not implementations: Hosking's `lmoments.f` and the NCDC Palmer Fortran, so that the shared ancestry in findings 1 and 2 is visible.

## Limits

- **What "read" means.** At each pin, the search covered origin statements (README, DESCRIPTION, CITATION, changelog, module headers), imports and declared dependencies, every mention of another comparator by name in source, tests, and documentation, and author lists. It was not a line-by-line audit. Copying that leaves no textual trace would be missed. *Independent* means that these searches found no relation.
- **Products.** For NCEI climdiv SPI, nClimDiv, and SPEIbase, the published code stands in for the operational code. NCEI says its public code "is not an exact copy" of what it runs.
- **Palmer pins.** These come from REFIMPL-30 ([#1632](https://github.com/monocongo/climate_indices/issues/1632)), whose pull request ([#1660](https://github.com/monocongo/climate_indices/pull/1660)) is still open. The original Wells C++ is `scPDSI.zip` from `greenleaf.unl.edu` (sha256 `52ceeb42...`; `scpdsi.cpp` dated 2007-05-18). REFIMPL-30's table records it as not obtained.
- **Authorship.** Shared authors were recorded where package metadata shows them. No pair is classified on authorship alone.

## Sources read

| Implementation | Pinned source | Source read |
| --- | --- | --- |
| climate_indices | monocongo/climate_indices@a7235bf7 | yes |
| xclim | xclim 1.0.0 sdist | yes |
| spei (Vonk) | spei 0.8.3 sdist; repository tag v0.8.3 (8276b0c9) | yes |
| standard_precip | standard-precip 2.0.0 sdist | yes |
| R SPEI | SPEI 1.8.1 CRAN tarball | yes |
| R SCI | SCI 1.0-3 CRAN tarball | yes |
| R SEI | SEI 0.2.0 CRAN tarball | yes |
| R climatekit | climatekit 0.2.3 CRAN tarball | yes |
| NDMC SPI Generator | SPIgenerator.zip sha256 ce8f37bc (1.7.6.0) | yes |
| NCL dim_spi_n | NCAR/ncl tag 6.6.2 (ni/src/lib/nfpfort/spid.f, spi3.f) | yes |
| Climate Engine API | live service; code not public | no: code not public |
| Earth Engine GRIDMET/DROUGHT | live collection; code not public | no: code not public |
| NCEI climdiv SPI | product; operational code not public; NCEI's public general-use spi.f (sha256 bd7bb8ab) read in its place | yes |
| SPEIbase v2.11 | product; spei.csic.es names R SPEI 1.8.1 as the code, read in its place | yes |
| PyETo | woodcrafty/PyETo@79ddd88d | yes |
| pyet | pyet 1.5.0 sdist | yes |
| R Evapotranspiration | Evapotranspiration 1.16 CRAN tarball | yes |
| NOAA PSL EDDI | product; code not public | no: code not public |
| NCEI pdi.f | pdi.f sha256 e0c58ccb (2013-04-03) | yes |
| nClimDiv (NCEI) | product; operational code not public; NCEI's public general-use pdi.f read in its place | yes |
| JonKing93/pdsi | JonKing93/pdsi@9a636117 | yes |
| Wells, Goddard and Hayes C++ | greenleaf.unl.edu scPDSI.zip sha256 52ceeb42 (scpdsi.cpp 2007-05-18) | yes |
| Sibada/scPDSI | Sibada/scPDSI@10f8c3b2 | yes |
| ktripa/scpdsi-tools | ktripa/scpdsi-tools@c73b22c8 | yes |

## Evidence

Every citation behind a relation, grouped by the implementation it comes from. `lineage.json` holds the same entries per pair.

**climate_indices**

- monocongo/climate_indices@224ef6bc (2017-06-21), `commit message`: "first draft of function to reproduce the water balance code in pdinew.f"
- monocongo/climate_indices@6dde1a55 (2018-10-03), `commit message`: "initial code for Hargreaves, with previous Thornthwaite folded into same file"
- monocongo/climate_indices@a7235bf7, `docs/research/earthengine-climate-engine-pilot.md`:9-13: "Measure agreement between climate_indices and two provider products that use gridMET inputs."
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/_palmer_pdi.py`:1: "NCEI pdi.f-lineage Palmer spell recursion for the standard PDSI family."
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/eto.py`:5-6: "Derived from original code found in PyETo: https://github.com/woodcrafty/PyETo"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/eto.py`:558: "Based on equation 52 in Allen et al (1998)."
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/indices.py`:277-278: "This matches the NOAA PSL Fortran reference implementation used in the original EDDI code."
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/self_calibration.py`:8: "Every function here is a faithful port of the reference algorithm"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/self_calibration.py`:89: "Ports the reference implementation's kthLargest()"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/lmoments.py`:279: "This is a Python translation of the original Fortran subroutine SAMLMR()"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/lmoments.py`:363-364: "This is the distribution R's SPEI package fits as "log-Logistic" (lmom's PELGLO with unbiased-PWM L-moments), so it is what SPEIbase uses."
- monocongo/climate_indices@a7235bf7, `src/climate_indices/lmoments.py`:381-383: "a Python translation of the PELGLO subroutine from J. R. M. Hosking's ... (IBM Research Report RC20525) as distributed in the R package lmom"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/lmoments.py`:65-70: "based on Fortran code written for inclusion in IBM Research Report RC20525 ... This is a Python translation of the original Fortran subroutine"
- monocongo/climate_indices@a7235bf7, `tests/fixture/ncei_spi/provenance.json`:70: "External ground truth for climate_indices.indices.spi() with Distribution.pearson ... independence is plausible but not airtight"
- monocongo/climate_indices@a7235bf7, `tests/fixture/nclimdiv/provenance.json`:2: "NOAA National Centers for Environmental Information (NCEI)"
- monocongo/climate_indices@a7235bf7, `tests/fixture/noaa-eddi-1month/provenance.json`:2: "NOAA Physical Sciences Laboratory (PSL)"
- monocongo/climate_indices@a7235bf7, `tests/fixture/palmer/provenance.json`:15: "SCPDSI GENERATION ... produced by the Sibada/scPDSI reference C++ (Wells, Goddard & Hayes)"
- monocongo/climate_indices@a7235bf7, `tests/fixture/pet_literature/provenance.json`:3: "https://github.com/woodcrafty/PyETo/blob/master/tests/test_thornthwaite.py, .../tests/test_fao.py"
- monocongo/climate_indices@a7235bf7, `tests/fixture/spei_loglogistic/provenance.json`:2: "R SPEI package (CRAN)"
- monocongo/climate_indices@a7235bf7, `tests/fixture/speibase/provenance.json`:123: "PLAUSIBILITY CHECK ONLY, NOT EXTERNAL NUMERICAL VALIDATION."
- monocongo/climate_indices@a7235bf7, `tests/fixture/zero_handling/provenance.json`:2: "R SEI and SCI packages"
- monocongo/climate_indices@cccf6ff9 (initial commit, 2017-06-13), `thornthwaite.py`:7: "Derived from original code found here: https://github.com/woodcrafty/PyETo"
- monocongo/climate_indices@d106d578 (2018-10-28), `commit message`: "new code for Pearson distribution fitting, replacing previous code that was originally translated from NCDC global SPI"

**xclim**

- xclim 1.0.0 sdist, `CHANGELOG.rst`:1407: "The offset value in standardized_precipitation_evapotranspiration_index is changed to better reproduce results in the reference library monocongo/climate_indices."
- xclim 1.0.0 sdist, `src/xclim/compute/_agro.py`:1109: "The results from climate_indices library can be reproduced with method = "APP" and fitwkargs = {"floc": 0}"
- xclim 1.0.0 sdist, `src/xclim/compute/stats.py`:172-175: "if method == "PWM" and not hasattr(dist, "lmom_fit"): ... Please pass an instance from the lmoments3 package."
- xclim 1.0.0 sdist, `src/xclim/compute/stats.py`:54-56: "if method in ["ML", "MLE"]: ... params = dist.fit(x, *args, method="mle", **kwargs, **fitkwargs)"
- xclim 1.0.0 sdist, `tests/test_indices.py`:351-353: "gamma/APP reference results: Obtained with monocongo/climate_indices library; MS/fisk/ML reference results: Obtained with R package SPEI"

**spei (Vonk)**

- spei 0.8.3 sdist; repository tag v0.8.3 (8276b0c9), `docs/examples/example04_package_comparison.ipynb`:cells 0, 7-8, 15-16: "Current comparisons include: standard_precip (Python), climate_indices (Python), pastas (Python), SPEI (R)"
- spei 0.8.3 sdist; repository tag v0.8.3 (8276b0c9), `docs/examples/example04_package_comparison.ipynb`:cells 12-13: "sr = importr("SPEI") ... spir_res = sr.spi(prec_rsum.values, scale=3)"
- spei 0.8.3 sdist; repository tag v0.8.3 (8276b0c9), `docs/examples/example04_package_comparison.ipynb`:cells 9-11: "Previously there was a significant difference beteween the SPEI and climate_indices package ... (comparison code commented out)"
- spei 0.8.3 sdist; repository tag v0.8.3 (8276b0c9), `src/spei/dist.py`:95: "fit_tuple = dist.fit("

**standard_precip**

- lmoments3 1.0.8 sdist (dependency), `README.rst`:35: "This package contains a Python 3.x implementation of the lmoments.f library created by J. R. M. Hosking."
- standard-precip 2.0.0 sdist, `README.md`:14: "These functions are loosely based on the SPEI package in R"
- standard-precip 2.0.0 sdist, `standard_precip/_distributions.py`:3-6: "L-moment fits go through the external lmoments3 package (Hosking's algorithms ...). MLE fits go through plain scipy.stats distributions"

**R SPEI**

- SPEI 1.8.1 CRAN tarball, `R/spei.R`:665-695: ""ub-pwm" = PWM(x.mon, order = 0:2) ... "log-Logistic" = tryCatch(pelglo(fortran_vec) ... "Gamma" = tryCatch(pelgam(... "PearsonIII" = tryCatch(pelpe3("
- SPEI 1.8.1 CRAN tarball, `R/spi.R`:13: "sol <- spei("
- lmom 3.3 CRAN tarball (dependency), `DESCRIPTION`:Author: "Author: J. R. M. Hosking [aut, cre]"

**R SCI**

- SCI 1.0-3 CRAN tarball, `DESCRIPTION`:22: "Depends: fitdistrplus, lmomco"
- SCI 1.0-3 CRAN tarball, `R/sci.r`:117: "suppressWarnings(do.call(mledist, mledist.par))"

**R SEI**

- SEI 0.2.0 CRAN tarball, `R/fit_dist.R`:267: "fit <- try(fitdistrplus::fitdist(data = data, distr = dist, method = method, ...))"
- SEI 0.2.0 CRAN tarball, `R/fit_dist.R`:291-308: "samp_mom <- lmom::samlmu(data) ... params <- lmom::pelgam(samp_mom)"

**NDMC SPI Generator**

- SPIgenerator.zip sha256 ce8f37bc (1.7.6.0), `code/spi_dll/gammadistributionestimatedprobabilities.cs`:32-34, 186-189: "input prob; return z. See Abromowitz and Stegun _Handbook of Mathematical Functions_, p. 933 ... Reference: Press, Flannery, Teukolsky, and Vetterling, _Numerical Recipes_, pp. 160-163 ... Thanks to kenny@cs.uiuc.edu"

**NCL dim_spi_n**

- NCAR/ncl tag 6.6.2 (ni/src/lib/nfpfort/spid.f, spi3.f), `ni/src/lib/nfpfort/spi3.f`:42, 69-74: "NCDC code ... This program used the program of John Kleist, Colorado State Univ. ... The subroutines and functions in upper case came from Jon Hosking, IBM. Ned Guttman"
- NCAR/ncl tag 6.6.2 (ni/src/lib/nfpfort/spid.f, spi3.f), `ni/src/lib/nfpfort/spid.f`:125, 324-327: "See Abromowitz and Stegun _Handbook of Mathematical Functions_, p. 933 ... Reference: Press, Flannery, Teukolsky, and Vetterling, _Numerical Recipes_, pp. 160-163 ... Thanks to kenny@cs.uiuc.edu"

**NCEI climdiv SPI**

- NCEI public spi.f sha256 bd7bb8ab, `spi.f`:64, 284-288: "written by Ned Guttman in January 1998 ... The subroutines and functions in upper case came from Jon Hosking, IBM."

**SPEIbase v2.11**

- <https://spei.csic.es/database.html> (accessed 2026-11-20): "SPEIbase v2.11: 1) the CRU TS 4.09 dataset ... 2) Using SPEI package version 1.8.1."

**PyETo**

- woodcrafty/PyETo@79ddd88d, `pyeto/fao.py`:328: "Based on equation 52 in Allen et al (1998)."

**pyet**

- pyet 1.5.0 sdist, `tests/test_rpackage.py`:66, 200-215: "lambda0 = 2.45  # Lambda in Danlu R Evapotranspiration package ... def test_fao56(self):"
- pyet 1.5.0 sdist, `tests/test_rpackage.py`:66, 483-490: "lambda0 = 2.45  # Lambda in Danlu R Evapotranspiration package ... r_har = [4.6, 4.3, ...]"

**NCEI pdi.f**

- <https://www.drought.gov/data-maps-tools/drought-indices-fortran-spi-pdsi> (accessed 2026-11-20): "This code is not an exact copy of what is run on NCEI's internal servers, but represents general use version approved for public release."
- pdi.f sha256 e0c58ccb (2013-04-03), `pdi.f`:13-16: "It is a modified version of the program in use at the National Climatic Data Center ... Code taken from the old version is in upper case"

**Sibada/scPDSI**

- Sibada/scPDSI@10f8c3b2, `DESCRIPTION`:16-17: "The function to calculate PDSI is based on the C++ source codes developed by Nathan Wells, Steve Goddard and Michael J. Hayes"
- Sibada/scPDSI@10f8c3b2, `src/pdsi.cpp`:4569: "number llist::kthLargest(int k) {"

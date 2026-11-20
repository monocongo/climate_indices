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
| PDSI, PHDI, PMDI, Z-Index | `climate_indices` (`pdsi`), NCEI `pdi.f`, nClimDiv, `JonKing93/pdsi`, Wells, Goddard and Hayes C++, `Sibada/scPDSI` |
| scPDSI | `climate_indices` (`scpdsi`), Wells, Goddard and Hayes C++, `Sibada/scPDSI`, `ktripa/scpdsi-tools` |

`precip-index` is excluded from the studies (REFIMPL-03) and is recorded only as lineage evidence for the question of which packages derive from `climate_indices`.

## Changes to the method

The Method section above is as committed in `13467e5a`, except for the table rows that amendments 1 and 2 add. Everything below was decided after comparator sources had been read, and is listed so that it can be judged as such.

### Amendments

1. **R `SPEI` added to the three PET rows.** REFIMPL-21 ([#1623](https://github.com/monocongo/climate_indices/issues/1623)) lists its `thornthwaite`, `hargreaves`, and `penman` functions.
2. **The Wells C++ and `Sibada/scPDSI` added to the standard Palmer row.** Both compute standard PDSI as well as scPDSI. Without them, the row cannot hold the shared NCDC ancestry in finding 2, which REFIMPL-31 ([#1633](https://github.com/monocongo/climate_indices/issues/1633)) needs for its comparison across lineages.
3. **"A common port" is read as "ports of one code base".** Two implementations that each translate the same upstream code, such as Hosking's L-moment Fortran, share a numerical routine even if neither calls the other's translation. Every such relation says whether the pair uses the *same dependency* or *separate ports of the same Fortran*, so a reader can apply the narrower reading instead.
4. **Notes.** A statement about another implementation that fits no class, such as "loosely based on", is recorded as a note on the pair and does not change its class.
5. **Program families.** The library's `pdsi`, `pdi.f`, and the Wells C++ each say they carry code from NCDC's Palmer Fortran, under three file names (`pdinew.f`, "the program in use at" NCDC, `pdsi.f`). Their pairs are classed *shared numerical routine*, and each such relation is labelled "same NCDC Palmer program family ..., identity unconfirmed (question 4)". The alternative was to call them *independent* against both sources' own statements.
6. **Strongest class over configurations.** The method gave a pair "the class in the configurations that the study tickets use". It is applied as the strongest class over all configurations, which here gives the same result because the studies use those configurations (see "Configurations" below).
7. **Comparisons with unread products.** The product rule decides what can be said about code that was not read. A comparison that this repository made against a product or service is about the repository's own tests and documents, so the pair is *validated against* even though the product's code was not read.

### How the method was applied

- **Products.** spei.csic.es names the code behind SPEIbase v2.11 (R `SPEI` 1.8.1), so that package was read in its place. NCEI publishes `spi.f` and `pdi.f` but says they are "not an exact copy" of what it runs, so NCEI climdiv SPI and nClimDiv stay unread. Their pairs are *unknown* unless this repository compared against them, and what the public Fortran shows is recorded in notes.
- **Configurations.** A pair takes the strongest class found in any configuration, and each relation names its configuration. The study tickets use the configurations in which the L-moment relations hold. REFIMPL-17 ([#1619](https://github.com/monocongo/climate_indices/issues/1619)) runs the library's Pearson III path, and REFIMPL-19 ([#1621](https://github.com/monocongo/climate_indices/issues/1621)) runs log-logistic, Pearson III, and GEV fits and R `SPEI`'s three fitting methods. A gamma-only SPI comparison is different: there the library's Thom fit shares no code with any L-moment package. Filter on the `configuration` field before treating a gamma agreement as shared code.
- **Python and Rust.** The library's Rust kernels are ports of its own Python code, checked by one parity harness ("One parity registry and harness shared by every ported Rust kernel", `tests/parity_registry.py`:1). Here the two paths are one implementation. Their numerical agreement belongs to REFIMPL-34 ([#1636](https://github.com/monocongo/climate_indices/issues/1636)).
- **Vocabulary.** These classes describe code ancestry. They are not the evidence statuses in `VALIDATION.md`, and none of them changes a status. *Validated against* records that a comparison was made. It does not say the two share code.

## Result

There are 177 pairs over eight indices. Every *unknown* pair involves a provider or product whose code is not public: the Climate Engine API, Earth Engine `GRIDMET/DROUGHT`, NOAA PSL EDDI, NCEI climdiv SPI, or nClimDiv.

| Index | Pairs | derived | validated against | shared numerical routine | independent | unknown |
| --- | --- | --- | --- | --- | --- | --- |
| SPI | 78 | 0 | 9 | 16 | 23 | 30 |
| SPEI | 36 | 0 | 6 | 13 | 4 | 13 |
| PET Thornthwaite | 6 | 1 | 0 | 0 | 5 | 0 |
| PET Hargreaves | 15 | 1 | 1 | 0 | 13 | 0 |
| PET FAO-56 Penman-Monteith | 15 | 0 | 1 | 0 | 14 | 0 |
| EDDI and non-parametric | 6 | 0 | 3 | 0 | 0 | 3 |
| PDSI, PHDI, PMDI, Z-Index | 15 | 1 | 2 | 4 | 4 | 4 |
| scPDSI | 6 | 3 | 0 | 0 | 3 | 0 |
| **All** | **177** | **6** | **22** | **33** | **66** | **50** |

### Findings

1. **One L-moment code base sits under most parameter fits.** J. R. M. Hosking's Fortran L-moment routines (IBM Research Report RC20525) reach the comparison by four routes. R `lmom`, Hosking's own package, is called by R `SPEI` for its default PWM fits and by R `SEI` for `method = "lmoment"`. Python `lmoments3`, a translation of `lmoments.f`, is used by `standard_precip` (`fit_type = "lmom"`) and by `xclim` (`method = "PWM"`). `climate_indices.lmoments` is the library's own translation of `SAMLMR`, the Pearson III routine, and `PELGLO`. NCL's Pearson III path embeds the routines in Fortran. Pairs that share a dependency, such as R `SPEI` and R `SEI`, run the same code. Other pairs run separate ports of the same Fortran, and their agreement is expected rather than independent. The six-decimal match between R `SPEI` `ub-pwm` and `standard_precip` `lmom` in `comparators.md` is consistent with this.
2. **The Palmer recursions share an NCDC ancestor.** The library's standard `pdsi` was translated from NCDC's `pdinew.f` ("new translation of pdinew.f", commit `0b15ec8c`). `pdi.f` is "a modified version of the program in use at the National Climatic Data Center" (`pdi.f`:13-16). The Wells, Goddard and Hayes C++ is "based mostly on the source code of pdsi.f", a Fortran program from NCDC (`scPDSI/src/scpdsi.cpp`:39-41). The `pdi.f` and Wells lineages are therefore branches of one NCDC program family, not independent implementations. The sources do not say whether `pdinew.f`, `pdi.f`, and Wells's `pdsi.f` are the same program (question 4). [REFIMPL-29](https://github.com/monocongo/climate_indices/issues/1631) traced the two recursions' disagreement on identical Z-Index and duration factors (6.3% of months) to six code-level differences between them. By this finding, those are differences between branches, not between independent implementations. The library's `scpdsi` is *derived*: its design says it was "traced directly from the Wells, Goddard & Hayes reference C++ implementation (`Sibada/scPDSI` ...)". Whether that tracing was a clean-room reimplementation is a licensing question for REFIMPL-27 ([#1629](https://github.com/monocongo/climate_indices/issues/1629)). `JonKing93/pdsi` ("based on a script written by Dave Meko", `README.md`:75) and `ktripa/scpdsi-tools` (follows Palmer 1965 and Wells et al. 2004, `scpdsi_tools.py`:14-15) state no code ancestry. They are the only *independent* Palmer comparators.
3. **NCEI climdiv SPI: its operational code is not public, but its public Fortran is not independent.** The library is *validated against* the product. NCEI's public `spi.f` (Ned Guttman, 1998) carries Hosking's routines in upper case, as the library's `lmoments.py` translates them, and NCL's Pearson III path embeds the same NCDC code under the same header. A 2018 commit also says the library's earlier Pearson code was "originally translated from NCDC global SPI". Whether that program is `spi.f` is not stated (question 7), and whether the operational product runs `spi.f` is question 4. Until those are answered, every pair with the product is *unknown* or *validated against*. The "plausible but not airtight" independence recorded in `tests/fixture/ncei_spi/provenance.json` still stands, with this shared ancestry as a reason to doubt it.
4. **Thom's approximation is shared as a formula, and once as code.** `climate_indices`, `xclim` (`APP`), `climatekit`, the NDMC SPI Generator, and NCL's gamma path all compute Thom's (1958) closed form. None cites another's code, so for most pairs this is not lineage. The NDMC and NCL gamma fit routines are the exception. Their comments match apart from capitalisation, including the misspelling "gamma paarameters" and "Bogus data array but do something reasonable", and their estimator lines are the same. So their parameter estimator comes from one source, which neither names (question 6). `climatekit` calls its Thom fit "consistent with the SPEI R package" (`R/drought.R`:444-446). R `SPEI` fits gamma with `lmom::pelgam`, so that statement is recorded as a note, not lineage.
5. **`xclim` was tuned to `climate_indices`.** Its tests take gamma `APP` reference values from the library. Its SPI docstring says the library's results can be reproduced. Its changelog records that the SPEI offset was changed "to better reproduce results in the reference library". Its log-logistic SPEI reference values come from R `SPEI`.
6. **PET.** The library's `eto` module (Thornthwaite and Hargreaves) is *derived* from `PyETo`. The module credits PyETo (`eto.py`:5-6), the initial 2017 commit carried the same credit, and the PET literature fixture names PyETo as the project `eto` "derives its implementation from". `pm_eto` (added 2026) carries no such credit. `pyet` tests its Hargreaves and FAO-56 values against R `Evapotranspiration`. No other PET pair has a stated relation.
7. **This repository's own comparisons.** The library's tests or research documents compare its output with R `SPEI`, SPEIbase, R `SEI`, R `SCI`, NCEI climdiv SPI, NOAA PSL EDDI, nClimDiv, `Sibada/scPDSI`, and both providers. A comparison records that agreement was measured. It does not show that the library was changed to match, or that the two share code, so it does not by itself make a pair dependent. Whether a pair is independent is decided by the other relations recorded for it. The library and R `SCI` have no shared routine, for example, and the SEI maximum-likelihood gamma fit against the library's Thom approximation (`tests/fixture/zero_handling/provenance.json`:17) remains independent-implementation evidence in `VALIDATION.md`.

### Leads checked

| Lead (ticket or research notes) | Verdict |
| --- | --- |
| `xclim` reproduces `climate_indices` with `APP` and `floc = 0`, and changed its SPEI offset to match | Confirmed: *validated against* |
| `climate_indices.lmoments.fit_glo` ports the `lmom` `PELGLO` routine that R `SPEI` uses | Confirmed. The pair is a *shared numerical routine* (separate ports), and the library is *validated against* R `SPEI` |
| `climate_indices.eto` derives from `PyETo` | Confirmed for Thornthwaite and Hargreaves. Not stated for `pm_eto` |
| Standard Palmer follows `pdi.f`, scPDSI follows Wells, `Sibada/scPDSI` wraps the Wells C++ | Confirmed, with two qualifications. The library translated NCDC's `pdinew.f`, and the Wells C++ also descends from NCDC Fortran (finding 2) |
| NCEI climdiv SPI's independence is "plausible but not airtight" | Still open. The public `spi.f` shares Hosking's routines with the library, but the operational code is unread (finding 3) |
| `precip-index` is built on `climate_indices` | Confirmed: its `README.md`, line 42 at `1c2ef1c3`, says "Built upon the foundation of climate-indices" |
| Other included packages derive from `climate_indices` | None found. `xclim` and `spei` compare against it, but none states that it derives from it |
| The NDMC program uses the library's Thom approximation (a lineage question in `comparators.md`) | A shared formula, not shared code. The NDMC program shares its gamma fit code with NCL instead (finding 4) |
| R `SPEI` `ub-pwm` and `standard_precip` `lmom` coincide to six decimals | Consistent with separate ports of Hosking's routines (finding 1) |
| Provenance of the repository's Thornthwaite PET fixture (`pet_thornthwaite.npy`) | Not resolved. The earliest trace is `5e87424c` (2018-12-11), which moved arrays out of a `tests/fixtures.py` whose content was stripped from history (question 7) |

`climatekit` is not in the SPEI row: its `ck_spei` was not examined for REFIMPL-03, and no study ticket lists it. Its `NEWS.md` (lines 12-13 at 0.2.3) says its log-logistic fit now uses the generalised logistic form "as the SPEI package does". That should be classified if `climatekit` SPEI joins a study.

## Pairs by index

Each table lists the pairs that are not *independent*, plus *independent* pairs that carry a note. An arrow points from the implementation that derives from, or validates against, the other. Citations for every relation are in [Evidence](#evidence) and in `lineage.json`.

### SPI

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / xclim | *validated against* | *validated against* (xclim → climate_indices; test reference values for gamma APP)<br>*shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency)) |
| climate_indices / spei (Vonk) | *validated against* | *validated against* (spei (Vonk) → climate_indices; documentation only; the comparison code is commented out at v0.8.3) |
| climate_indices / standard_precip | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); standard_precip fit_type = "lmom" (lmoments3 dependency)) |
| climate_indices / R SPEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency)) |
| climate_indices / R SCI | *validated against* | *validated against* (climate_indices → R SCI; zero placement) |
| climate_indices / R SEI | *validated against* | *validated against* (climate_indices → R SEI; zero placement)<br>*shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); R SEI method = "lmoment" (lmom dependency)) |
| climate_indices / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); NCL dim_spi_n Pearson III path (opt@spi_type = 3) (embedded Fortran (NCDC spi code))) |
| climate_indices / Climate Engine API | *validated against* | *validated against* (climate_indices → Climate Engine API; characterization pilot) |
| climate_indices / Earth Engine GRIDMET/DROUGHT | *validated against* | *validated against* (climate_indices → Earth Engine GRIDMET/DROUGHT; characterization pilot) |
| climate_indices / NCEI climdiv SPI | *validated against* | *validated against* (climate_indices → NCEI climdiv SPI; Pearson III)<br>note: The library's pre-2018 Pearson III code was "originally translated from NCDC global SPI"; NCEI's public spi.f carries Hosking's routines, as the library's lmoments.py does. |
| xclim / spei (Vonk) | *shared numerical routine* | *shared numerical routine* (scipy.stats rv_continuous.fit; maximum likelihood) |
| xclim / standard_precip | *shared numerical routine* | *shared numerical routine* (scipy.stats rv_continuous.fit; maximum likelihood)<br>*shared numerical routine* (Hosking L-moment routines (IBM RC20525); same dependency: xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency); standard_precip fit_type = "lmom" (lmoments3 dependency)) |
| xclim / R SPEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency); R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency)) |
| xclim / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency); R SEI method = "lmoment" (lmom dependency)) |
| xclim / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency); NCL dim_spi_n Pearson III path (opt@spi_type = 3) (embedded Fortran (NCDC spi code))) |
| xclim / Climate Engine API | *unknown* | code not public; question 1 |
| xclim / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| xclim / NCEI climdiv SPI | *unknown* | code not public; question 4 |
| spei (Vonk) / standard_precip | *validated against* | *validated against* (spei (Vonk) → standard_precip; documentation notebook)<br>*shared numerical routine* (scipy.stats rv_continuous.fit; maximum likelihood) |
| spei (Vonk) / R SPEI | *validated against* | *validated against* (spei (Vonk) → R SPEI; documentation notebook; the comparison is textual) |
| spei (Vonk) / Climate Engine API | *unknown* | code not public; question 1 |
| spei (Vonk) / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| spei (Vonk) / NCEI climdiv SPI | *unknown* | code not public; question 4 |
| standard_precip / R SPEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: standard_precip fit_type = "lmom" (lmoments3 dependency); R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency))<br>note: standard_precip says its functions are "loosely based on" R SPEI; that names a design source, not a code port. |
| standard_precip / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: standard_precip fit_type = "lmom" (lmoments3 dependency); R SEI method = "lmoment" (lmom dependency)) |
| standard_precip / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: standard_precip fit_type = "lmom" (lmoments3 dependency); NCL dim_spi_n Pearson III path (opt@spi_type = 3) (embedded Fortran (NCDC spi code))) |
| standard_precip / Climate Engine API | *unknown* | code not public; question 1 |
| standard_precip / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| standard_precip / NCEI climdiv SPI | *unknown* | code not public; question 4 |
| R SPEI / R SCI | *independent* | note: Both depend on lmomco: R SPEI for PWM-to-L-moment conversion and fallback fits, R SCI only for maximum-likelihood starting values, which do not set the value. |
| R SPEI / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); same dependency: R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency); R SEI method = "lmoment" (lmom dependency)) |
| R SPEI / R climatekit | *independent* | note: climatekit calls its Thom fit "consistent with the SPEI R package" without comparing output; R SPEI fits gamma with lmom::pelgam, not Thom's approximation. |
| R SPEI / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency); NCL dim_spi_n Pearson III path (opt@spi_type = 3) (embedded Fortran (NCDC spi code))) |
| R SPEI / Climate Engine API | *unknown* | code not public; question 1 |
| R SPEI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| R SPEI / NCEI climdiv SPI | *unknown* | code not public; question 4 |
| R SCI / R SEI | *shared numerical routine* | *shared numerical routine* (fitdistrplus; maximum likelihood) |
| R SCI / Climate Engine API | *unknown* | code not public; question 1 |
| R SCI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| R SCI / NCEI climdiv SPI | *unknown* | code not public; question 4 |
| R SEI / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: R SEI method = "lmoment" (lmom dependency); NCL dim_spi_n Pearson III path (opt@spi_type = 3) (embedded Fortran (NCDC spi code))) |
| R SEI / Climate Engine API | *unknown* | code not public; question 1 |
| R SEI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| R SEI / NCEI climdiv SPI | *unknown* | code not public; question 4 |
| R climatekit / Climate Engine API | *unknown* | code not public; question 1 |
| R climatekit / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| R climatekit / NCEI climdiv SPI | *unknown* | code not public; question 4 |
| NDMC SPI Generator / NCL dim_spi_n | *shared numerical routine* | *shared numerical routine* (unnamed common source of the gamma fit; gamma path: the gamma fit routines have the same comments (apart from capitalisation), including the misspelling "paarameters", and the same estimator lines) |
| NDMC SPI Generator / Climate Engine API | *unknown* | code not public; question 1 |
| NDMC SPI Generator / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| NDMC SPI Generator / NCEI climdiv SPI | *unknown* | code not public; question 4 |
| NCL dim_spi_n / Climate Engine API | *unknown* | code not public; question 1 |
| NCL dim_spi_n / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| NCL dim_spi_n / NCEI climdiv SPI | *unknown* | code not public; question 4<br>note: NCL's Pearson III path embeds the NCDC SPI code that NCEI publishes as spi.f; whether the operational product runs that code is question 4. |
| Climate Engine API / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 1, 2 |
| Climate Engine API / NCEI climdiv SPI | *unknown* | code not public; question 1, 4 |
| Earth Engine GRIDMET/DROUGHT / NCEI climdiv SPI | *unknown* | code not public; question 2, 4 |

The other 21 of 78 pairs are *independent*.

### SPEI

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / xclim | *validated against* | *validated against* (xclim → climate_indices; offset changed to reproduce climate_indices)<br>*shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency)) |
| climate_indices / standard_precip | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); standard_precip fit_type = "lmom" (lmoments3 dependency)) |
| climate_indices / R SPEI | *validated against* | *validated against* (climate_indices → R SPEI; log-logistic)<br>*shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency)) |
| climate_indices / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); R SEI method = "lmoment" (lmom dependency)) |
| climate_indices / Climate Engine API | *validated against* | *validated against* (climate_indices → Climate Engine API; characterization pilot) |
| climate_indices / Earth Engine GRIDMET/DROUGHT | *validated against* | *validated against* (climate_indices → Earth Engine GRIDMET/DROUGHT; characterization pilot) |
| climate_indices / SPEIbase v2.11 | *validated against* | *validated against* (climate_indices → SPEIbase v2.11; input-matched log-logistic comparison)<br>*shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: climate_indices Pearson III, log-logistic and GEV fits, all from its SAMLMR translation (own translation (lmoments.py)); SPEIbase v2.11 ub-pwm log-logistic (lmom dependency, through R SPEI)) |
| xclim / spei (Vonk) | *shared numerical routine* | *shared numerical routine* (scipy.stats rv_continuous.fit; maximum likelihood) |
| xclim / standard_precip | *shared numerical routine* | *shared numerical routine* (scipy.stats rv_continuous.fit; maximum likelihood)<br>*shared numerical routine* (Hosking L-moment routines (IBM RC20525); same dependency: xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency); standard_precip fit_type = "lmom" (lmoments3 dependency)) |
| xclim / R SPEI | *validated against* | *validated against* (xclim → R SPEI; fisk ML test reference values)<br>*shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency); R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency)) |
| xclim / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency); R SEI method = "lmoment" (lmom dependency)) |
| xclim / Climate Engine API | *unknown* | code not public; question 1 |
| xclim / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| xclim / SPEIbase v2.11 | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: xclim method = "PWM" with an lmoments3 distribution (lmoments3 dependency); SPEIbase v2.11 ub-pwm log-logistic (lmom dependency, through R SPEI)) |
| spei (Vonk) / standard_precip | *shared numerical routine* | *shared numerical routine* (scipy.stats rv_continuous.fit; maximum likelihood) |
| spei (Vonk) / Climate Engine API | *unknown* | code not public; question 1 |
| spei (Vonk) / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| standard_precip / R SPEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: standard_precip fit_type = "lmom" (lmoments3 dependency); R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency))<br>note: standard_precip says its functions are "loosely based on" R SPEI; that names a design source, not a code port. |
| standard_precip / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: standard_precip fit_type = "lmom" (lmoments3 dependency); R SEI method = "lmoment" (lmom dependency)) |
| standard_precip / Climate Engine API | *unknown* | code not public; question 1 |
| standard_precip / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| standard_precip / SPEIbase v2.11 | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); separate ports of the same Fortran: standard_precip fit_type = "lmom" (lmoments3 dependency); SPEIbase v2.11 ub-pwm log-logistic (lmom dependency, through R SPEI)) |
| R SPEI / R SEI | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); same dependency: R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency); R SEI method = "lmoment" (lmom dependency)) |
| R SPEI / Climate Engine API | *unknown* | code not public; question 1 |
| R SPEI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| R SPEI / SPEIbase v2.11 | *shared numerical routine* | *shared numerical routine* (R SPEI 1.8.1 spei() (whole computation); SPEIbase is computed with the package (ub-pwm log-logistic))<br>*shared numerical routine* (Hosking L-moment routines (IBM RC20525); same dependency: R SPEI pp-pwm and ub-pwm fits (default) (lmom dependency); SPEIbase v2.11 ub-pwm log-logistic (lmom dependency, through R SPEI)) |
| R SEI / Climate Engine API | *unknown* | code not public; question 1 |
| R SEI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2 |
| R SEI / SPEIbase v2.11 | *shared numerical routine* | *shared numerical routine* (Hosking L-moment routines (IBM RC20525); same dependency: R SEI method = "lmoment" (lmom dependency); SPEIbase v2.11 ub-pwm log-logistic (lmom dependency, through R SPEI)) |
| Climate Engine API / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 1, 2 |
| Climate Engine API / SPEIbase v2.11 | *unknown* | code not public; question 1 |
| Earth Engine GRIDMET/DROUGHT / SPEIbase v2.11 | *unknown* | code not public; question 2 |

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
| climate_indices / Climate Engine API | *validated against* | *validated against* (climate_indices → Climate Engine API; characterization pilot) |
| climate_indices / Earth Engine GRIDMET/DROUGHT | *validated against* | *validated against* (climate_indices → Earth Engine GRIDMET/DROUGHT; characterization pilot) |
| NOAA PSL EDDI / Climate Engine API | *unknown* | code not public; question 1, 3 |
| NOAA PSL EDDI / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 2, 3 |
| Climate Engine API / Earth Engine GRIDMET/DROUGHT | *unknown* | code not public; question 1, 2 |

No pair is *independent*.

### PDSI, PHDI, PMDI, Z-Index

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / NCEI pdi.f | *shared numerical routine* | *shared numerical routine* (NCDC Palmer Fortran; same NCDC Palmer program family by both sources' own statements, identity unconfirmed (question 4)) |
| climate_indices / nClimDiv (NCEI) | *validated against* | *validated against* (climate_indices → nClimDiv (NCEI)) |
| climate_indices / Wells, Goddard and Hayes C++ | *shared numerical routine* | *shared numerical routine* (NCDC Palmer Fortran; same NCDC Palmer program family by both sources' own statements, identity unconfirmed (question 4)) |
| climate_indices / Sibada/scPDSI | *validated against* | *validated against* (climate_indices → Sibada/scPDSI; the oracle harness run in standard-PDSI mode reproduces the library's Z-Index)<br>*shared numerical routine* (NCDC Palmer Fortran; same NCDC Palmer program family by both sources' own statements, identity unconfirmed (question 4)) |
| NCEI pdi.f / nClimDiv (NCEI) | *unknown* | code not public; question 4<br>note: pdi.f is NCEI's public version of its Palmer program, "not an exact copy" of what it runs; question 4. |
| NCEI pdi.f / Wells, Goddard and Hayes C++ | *shared numerical routine* | *shared numerical routine* (NCDC Palmer Fortran; same NCDC Palmer program family by both sources' own statements, identity unconfirmed (question 4)) |
| NCEI pdi.f / Sibada/scPDSI | *shared numerical routine* | *shared numerical routine* (NCDC Palmer Fortran; same NCDC Palmer program family by both sources' own statements, identity unconfirmed (question 4)) |
| nClimDiv (NCEI) / JonKing93/pdsi | *unknown* | code not public; question 4 |
| nClimDiv (NCEI) / Wells, Goddard and Hayes C++ | *unknown* | code not public; question 4 |
| nClimDiv (NCEI) / Sibada/scPDSI | *unknown* | code not public; question 4 |
| Wells, Goddard and Hayes C++ / Sibada/scPDSI | *derived* | *derived* (Sibada/scPDSI → Wells, Goddard and Hayes C++) |

The other 4 of 15 pairs are *independent*.

### scPDSI

| Pair | Class | Relation found |
| --- | --- | --- |
| climate_indices / Wells, Goddard and Hayes C++ | *derived* | *derived* (climate_indices → Wells, Goddard and Hayes C++; the design names the Wells C++ as distributed in Sibada/scPDSI) |
| climate_indices / Sibada/scPDSI | *derived* | *derived* (climate_indices → Sibada/scPDSI; also validated against it (oracle fixture)) |
| Wells, Goddard and Hayes C++ / Sibada/scPDSI | *derived* | *derived* (Sibada/scPDSI → Wells, Goddard and Hayes C++) |

The other 3 of 6 pairs are *independent*.

## Questions for REFIMPL-06

These questions cover every *unknown* pair, plus the residual questions behind some classified ones. Each *unknown* pair in `lineage.json` lists the questions that cover it. They are for REFIMPL-06 ([#1608](https://github.com/monocongo/climate_indices/issues/1608)), except question 7, which is for the maintainer.

1. **Climate Engine API.** Which code computes `standard_index` SPI, SPEI, and EDDI? Does it share code with Earth Engine `GRIDMET/DROUGHT` or NOAA PSL EDDI? (20 *unknown* pairs.)
2. **Earth Engine `GRIDMET/DROUGHT`** (producer: University of California Merced). Which code produces it, and which plotting-position formula does it use? The catalog says only that "plotting positions are used". (20 *unknown* pairs, 3 shared with question 1.)
3. **NOAA PSL EDDI.** Is the operational code available, and do the Climate Engine or `GRIDMET/DROUGHT` EDDI use it? Hobbins et al. (2016) lists J. L. Huntington and C. Morton among its authors (`src/climate_indices/cf_metadata_registry.py`:327). The research notes cite Huntington et al. (2017) for Climate Engine, a reference REFIMPL-05 has yet to verify. (2 *unknown* pairs, both shared with question 1 or 2.)
4. **NCEI.** Does the operational climdiv SPI run the `SPIPE3` code in the public `spi.f`, and does operational nClimDiv run the `pdi.f` lineage? Are `pdinew.f` (which the library translated in 2017), `pdi.f`, and the `pdsi.f` that the Wells C++ was based on the same NCDC program? (15 *unknown* pairs.)
5. **Dave Meko** (via `JonKing93/pdsi`). Was the MATLAB PDSI script written from NCDC Fortran? A yes would change a `JonKing93/pdsi` pair from *independent* to *shared numerical routine* only where the other implementation is shown to use the same routine. It would not settle which routine operational nClimDiv runs, so the `JonKing93/pdsi` / nClimDiv pair stays *unknown* under question 4.
6. **NDMC and NCAR.** What is the common source of the gamma fit code in the NDMC SPI Generator and NCL?
7. **Maintainer.** Commit `d106d578` (2018) replaced Pearson III code "originally translated from NCDC global SPI". What was that program? The `SAMLMR` and Pearson III translations were already in `compute.py` at the initial commit (2017) and moved to `lmoments.py` in 2019: do they come from it? Where did `tests/fixture/pet_thornthwaite.npy` come from?

## Figure specification for REFIMPL-40

For REFIMPL-40 ([#1642](https://github.com/monocongo/climate_indices/issues/1642)):

- **Data.** `lineage.json` only. No edge is drawn by hand.
- **Layout.** One panel per index, or five grouped panels: SPI and SPEI; PET; EDDI; standard Palmer; scPDSI.
- **Nodes.** One per implementation in the index's `implementations` list, shaped by `kind` (`package`, `program`, `product`, `service`). Implementations with `source_read: false` get a dashed outline.
- **Edges.** One per entry in `pairs[].relations`. *Derived*: solid arrow from `from` to `to`. *Validated against*: dashed arrow from `from` to `to`. *Shared numerical routine*: undirected dotted line labelled with `routine`. *Independent* and *unknown* pairs get no edge. `pairs[].direction` gives the arrow for the pair's own class.
- **Ancestors.** The two entries in `ancestors` (Hosking's L-moment routines and the NCDC Palmer Fortran) are optional context nodes, not implementations, joined to their `descendants`.

## Limits

- **What "read" means.** At each pin, the search covered origin statements (README, DESCRIPTION, CITATION, changelog, module headers), imports and declared dependencies, every mention of another comparator by name in source, tests, and documentation, and author lists. It was not a line-by-line audit. Copying that leaves no textual trace would be missed. *Independent* means that these searches found no relation.
- **Quotes.** Quotes are short attribution lines, cut with "..." and with Markdown formatting dropped. No comparator code is reproduced.
- **Palmer pins.** These come from REFIMPL-30 ([#1632](https://github.com/monocongo/climate_indices/issues/1632)), whose pull request ([#1660](https://github.com/monocongo/climate_indices/pull/1660)) is still open. The original Wells C++ is `scPDSI.zip` from `greenleaf.unl.edu` (sha256 `52ceeb42...`; `scpdsi.cpp` dated 2007-05-18). REFIMPL-30's table records it as not obtained.
- **Authorship.** Shared authors were recorded where package metadata shows them. No pair is classified on authorship alone.

## Sources read

| Implementation | Kind | Pinned source | Source read |
| --- | --- | --- | --- |
| climate_indices | package | monocongo/climate_indices@a7235bf7 | yes |
| xclim | package | xclim 1.0.0 sdist | yes |
| spei (Vonk) | package | spei 0.8.3 sdist; repository tag v0.8.3 (8276b0c9) | yes |
| standard_precip | package | standard-precip 2.0.0 sdist | yes |
| R SPEI | package | SPEI 1.8.1 CRAN tarball | yes |
| R SCI | package | SCI 1.0-3 CRAN tarball | yes |
| R SEI | package | SEI 0.2.0 CRAN tarball | yes |
| R climatekit | package | climatekit 0.2.3 CRAN tarball | yes |
| NDMC SPI Generator | program | SPIgenerator.zip sha256 ce8f37bc (1.7.6.0) | yes |
| NCL dim_spi_n | program | NCAR/ncl tag 6.6.2 (ni/src/lib/nfpfort/spid.f, spi3.f) | yes |
| Climate Engine API | service | live service; code not public | no: code not public |
| Earth Engine GRIDMET/DROUGHT | service | live collection; code not public | no: code not public |
| NCEI climdiv SPI | product | operational code not public; NCEI's public spi.f (sha256 bd7bb8ab) is "not an exact copy" | no: code not public |
| SPEIbase v2.11 | product | spei.csic.es names R SPEI 1.8.1 as the code; that package was read | yes |
| PyETo | package | woodcrafty/PyETo@79ddd88d | yes |
| pyet | package | pyet 1.5.0 sdist | yes |
| R Evapotranspiration | package | Evapotranspiration 1.16 CRAN tarball | yes |
| NOAA PSL EDDI | product | code not public | no: code not public |
| NCEI pdi.f | program | pdi.f sha256 e0c58ccb (2013-04-03) | yes |
| nClimDiv (NCEI) | product | operational code not public; NCEI's public pdi.f is "not an exact copy" | no: code not public |
| JonKing93/pdsi | package | JonKing93/pdsi@9a636117 | yes |
| Wells, Goddard and Hayes C++ | program | greenleaf.unl.edu scPDSI.zip sha256 52ceeb42 (scpdsi.cpp 2007-05-18) | yes |
| Sibada/scPDSI | package | Sibada/scPDSI@10f8c3b2 | yes |
| ktripa/scpdsi-tools | package | ktripa/scpdsi-tools@c73b22c8 | yes |

## Evidence

Every citation behind a relation, note, stated origin, ancestor, or statement in the findings, grouped by the implementation it comes from. `lineage.json` holds the same entries, the last group under `citations`.

**climate_indices**

- monocongo/climate_indices@0b15ec8c (2017-07-28), `commit message`: "updates to function signatures, new translation of pdinew.f, more tests"
- monocongo/climate_indices@224ef6bc (2017-06-21), `commit message`: "first draft of function to reproduce the water balance code in pdinew.f"
- monocongo/climate_indices@5e87424c (2018-12-11), `commit message`: "wrote numpy arrays used for testing into *.npy files, now reading these files for fixture arrays"
- monocongo/climate_indices@6dde1a55 (2018-10-03), `commit message`: "initial code for Hargreaves, with previous Thornthwaite folded into same file"
- monocongo/climate_indices@a7235bf7, `docs/research/earthengine-climate-engine-pilot.md`:9-13: "Measure agreement between climate_indices and two provider products that use gridMET inputs."
- monocongo/climate_indices@a7235bf7, `src/climate_indices/cf_metadata_registry.py`:327: "Hobbins, M. T., Wood, A., McEvoy, D. J., Huntington, J. L., Morton, C., "
- monocongo/climate_indices@a7235bf7, `src/climate_indices/compute.py`:2083: "alphas = (1 + np.sqrt(1 + 4 * a / 3)) / (4 * a)"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/_palmer_pdi.py`:1: "NCEI pdi.f-lineage Palmer spell recursion for the standard PDSI family."
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/eto.py`:5-6: "Derived from original code found in PyETo: https://github.com/woodcrafty/PyETo"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/eto.py`:558: "Based on equation 52 in Allen et al (1998)."
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/pm_eto.py`:1-4: "Penman-Monteith reference evapotranspiration (FAO-56) helper functions. This module implements the component equations from FAO Irrigation and Drainage Paper 56 (Allen et al., 1998)"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/self_calibration.py`:8: "Every function here is a faithful port of the reference algorithm"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/drought/self_calibration.py`:89: "Ports the reference implementation's kthLargest()"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/lmoments.py`:279: "This is a Python translation of the original Fortran subroutine SAMLMR()"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/lmoments.py`:363-364: "This is the distribution R's SPEI package fits as "log-Logistic" (lmom's PELGLO with unbiased-PWM L-moments), so it is what SPEIbase uses."
- monocongo/climate_indices@a7235bf7, `src/climate_indices/lmoments.py`:381-383: "a Python translation of the PELGLO subroutine from J. R. M. Hosking's ... (IBM Research Report RC20525) as distributed in the R package lmom"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/lmoments.py`:601: "lmoments = _estimate_lmoments(timeseries, hint="")"
- monocongo/climate_indices@a7235bf7, `src/climate_indices/lmoments.py`:65-70: "based on Fortran code written for inclusion in IBM Research Report RC20525 ... This is a Python translation of the original Fortran subroutine"
- monocongo/climate_indices@a7235bf7, `tests/fixture/ncei_spi/provenance.json`:70: "External ground truth for climate_indices.indices.spi() with Distribution.pearson ... independence is plausible but not airtight"
- monocongo/climate_indices@a7235bf7, `tests/fixture/noaa-eddi-1month/provenance.json`:2: "NOAA Physical Sciences Laboratory (PSL)"
- monocongo/climate_indices@a7235bf7, `tests/fixture/palmer/provenance.json`:15: "SCPDSI GENERATION ... produced by the Sibada/scPDSI reference C++ (Wells, Goddard & Hayes) ... HARNESS VALIDATION: running the same binary in standard-PDSI mode reproduces the committed zindex.npy for all 344 divisions"
- monocongo/climate_indices@a7235bf7, `tests/fixture/pet_literature/provenance.json`:5: "the upstream project climate_indices.eto derives its implementation from"
- monocongo/climate_indices@a7235bf7, `tests/fixture/spei_loglogistic/provenance.json`:2: "R SPEI package (CRAN)"
- monocongo/climate_indices@a7235bf7, `tests/fixture/speibase/provenance.json`:123: "PLAUSIBILITY CHECK ONLY, NOT EXTERNAL NUMERICAL VALIDATION."
- monocongo/climate_indices@a7235bf7, `tests/fixture/zero_handling/provenance.json`:2: "R SEI and SCI packages"
- monocongo/climate_indices@a7235bf7, `tests/parity_registry.py`:1: "One parity registry and harness shared by every ported Rust kernel."
- monocongo/climate_indices@a7235bf7, `tests/test_nclimdiv_reference.py`:1-5: "Palmer-family comparisons against NOAA nClimDiv. ... against the operational NOAA NCEI nClimDiv reference arrays"
- monocongo/climate_indices@a7235bf7, `tests/test_noaa_eddi_reference.py`:3-4: "Validates the library's EDDI implementation against pre-computed EDDI values from the NOAA Physical Sciences Laboratory (PSL)."
- monocongo/climate_indices@a7235bf7, `tests/test_speibase_like_for_like.py`:1: "Like-for-like SPEI comparison against the CSIC SPEIbase v2.11 grids."
- monocongo/climate_indices@c3b2d165 (2017-09-07), `commit message`: "new version which matches to the original pdinew.f results >99%"
- monocongo/climate_indices@cccf6ff9 (initial commit, 2017-06-13), `thornthwaite.py`:7: "Derived from original code found here: https://github.com/woodcrafty/PyETo"
- monocongo/climate_indices@d0a3a401 (2026-08-08; file removed from main by 2ee0f18b), `docs/superpowers/specs/2026-08-07-scpdsi-epic-design.md`:12: "Traced directly from the Wells, Goddard & Hayes reference C++ implementation (Sibada/scPDSI on GitHub, GPLv3 — used here for algorithm reference only; see "Licensing" below for why no GPL source enters this repository), specifically Rext_PDSI_mon(sc=true) and the functions it calls"
- monocongo/climate_indices@d106d578 (2018-10-28), `commit message`: "new code for Pearson distribution fitting, replacing previous code that was originally translated from NCDC global SPI"
- monocongo/climate_indices@e7b1f461 (2026-04-05), `commit message`: "feat(pm-et): add Penman-Monteith FAO56 implementation and tests"

**xclim**

- xclim 1.0.0 sdist, `CHANGELOG.rst`:1407: "The offset value in standardized_precipitation_evapotranspiration_index is changed to better reproduce results in the reference library monocongo/climate_indices."
- xclim 1.0.0 sdist, `src/xclim/compute/_agro.py`:1109: "The results from climate_indices library can be reproduced with method = "APP" and fitwkargs = {"floc": 0}"
- xclim 1.0.0 sdist, `src/xclim/compute/stats.py`:172-175: "if method == "PWM" and not hasattr(dist, "lmom_fit"): ... Please pass an instance from the lmoments3 package."
- xclim 1.0.0 sdist, `src/xclim/compute/stats.py`:54-56: "if method in ["ML", "MLE"]: ... params = dist.fit(x, *args, method="mle", **kwargs, **fitkwargs)"
- xclim 1.0.0 sdist, `src/xclim/compute/stats.py`:657: "a0 = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A)"
- xclim 1.0.0 sdist, `tests/test_indices.py`:351-353: "gamma/APP reference results: Obtained with monocongo/climate_indices library; MS/fisk/ML reference results: Obtained with R package SPEI"

**spei (Vonk)**

- spei 0.8.3 sdist; repository tag v0.8.3 (8276b0c9), `docs/examples/example04_package_comparison.ipynb`:cells 0, 7-8, 15-16: "Current comparisons include: standard_precip (Python), climate_indices (Python), pastas (Python), SPEI (R)"
- spei 0.8.3 sdist; repository tag v0.8.3 (8276b0c9), `docs/examples/example04_package_comparison.ipynb`:cells 13, 16: "spir_res = sr.spi(prec_rsum.values, scale=3) ... The SPEI R package also has a similar result but seems to vary a bit more."
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

**R climatekit**

- climatekit 0.2.3 CRAN tarball, `NEWS.md`:12-13: "uses the generalised logistic form, which covers any L-skewness between -1 and 1, as the SPEI package does."
- climatekit 0.2.3 CRAN tarball, `R/drought.R`:444-446: "Uses Maximum Likelihood Estimation via Thom's (1958) approximation, as specified by the WMO User Guide (WMO-No. 1090) and consistent with the SPEI R package."
- climatekit 0.2.3 CRAN tarball, `R/drought.R`:466: "alpha <- (1 + sqrt(1 + 4 * A / 3)) / (4 * A)"

**NDMC SPI Generator**

- SPIgenerator.zip sha256 ce8f37bc (1.7.6.0), `code/spi_dll/gammadistributionestimatedprobabilities.cs`:85, 124, 132, 140-142: "gamma - gamma paarameters ... Bogus Data array but do something reasonable ... They were all zeroes. ... Use MLE ... alpha = Math.Log( mn ) - sumlog / nact; gamm = (1.0 + Math.Sqrt( 1.0 + 4.0 * alpha / 3.0 )) / (4.0 * alpha);"

**NCL dim_spi_n**

- NCAR/ncl tag 6.6.2 (ni/src/lib/nfpfort/spid.f, spi3.f), `ni/src/lib/nfpfort/spi3.f`:42, 69-74: "NCDC code ... This program used the program of John Kleist, Colorado State Univ. ... The subroutines and functions in upper case came from Jon Hosking, IBM. Ned Guttman"
- NCAR/ncl tag 6.6.2 (ni/src/lib/nfpfort/spid.f, spi3.f), `ni/src/lib/nfpfort/spid.f`:163, 197, 205, 213, 218-219: "gamma - gamma paarameters ... Bogus data array but do something reasonable ... They were all zeroes. ... Use MLE ... alpha = log(av) - sumlog / nact ... gamm  = (1.0d0 + sqrt(1.0d0 + 4.0d0*alpha/ 3.0d0)) /(4.0d0*alpha)"

**NCEI climdiv SPI**

- NCEI public spi.f sha256 bd7bb8ab, `spi.f`:64, 284-288: "written by Ned Guttman in January 1998 ... The subroutines and functions in upper case came from Jon Hosking, IBM."

**SPEIbase v2.11**

- <https://spei.csic.es/database.html> (accessed 2026-11-20): "SPEIbase v2.11: 1) the CRU TS 4.09 dataset ... 2) Using SPEI package version 1.8.1."
- <https://spei.csic.es/database.html> (accessed 2026-11-20): "Unbiased probability weighted moments (ub-pwm) method has been used for fitting the log-Logistic distribution"

**PyETo**

- woodcrafty/PyETo@79ddd88d, `pyeto/fao.py`:328: "Based on equation 52 in Allen et al (1998)."

**pyet**

- pyet 1.5.0 sdist, `tests/test_rpackage.py`:66, 200-215: "lambda0 = 2.45  # Lambda in Danlu R Evapotranspiration package ... def test_fao56(self):"
- pyet 1.5.0 sdist, `tests/test_rpackage.py`:66, 483-490: "lambda0 = 2.45  # Lambda in Danlu R Evapotranspiration package ... r_har = [4.6, 4.3, ...]"

**NCEI pdi.f**

- <https://www.drought.gov/data-maps-tools/drought-indices-fortran-spi-pdsi> (accessed 2026-11-20): "This code is not an exact copy of what is run on NCEI's internal servers, but represents general use version approved for public release."
- pdi.f sha256 e0c58ccb (2013-04-03), `pdi.f`:13-16: "It is a modified version of the program in use at the National Climatic Data Center ... Code taken from the old version is in upper case"

**JonKing93/pdsi**

- JonKing93/pdsi@9a636117, `README.md`:75: "This PDSI function is based on a script written by Dave Meko."

**Wells, Goddard and Hayes C++**

- greenleaf.unl.edu scPDSI.zip sha256 52ceeb42 (scpdsi.cpp 2007-05-18), `scPDSI/src/scpdsi.cpp`:33-41: "Most recently translated to C++ from FORTRAN ... Based mostly on the source code of pdsi.f, a FORTRAN program that calculates monthly PDSI values. The source code came from NCDC, originally written by Tom Karl, and revised by Roger Winchell, Alan McNab, and Richard Heim."

**Sibada/scPDSI**

- Sibada/scPDSI@10f8c3b2, `DESCRIPTION`:16-17: "The function to calculate PDSI is based on the C++ source codes developed by Nathan Wells, Steve Goddard and Michael J. Hayes"
- Sibada/scPDSI@10f8c3b2, `src/pdsi.cpp`:4569: "number llist::kthLargest(int k) {"

**ktripa/scpdsi-tools**

- ktripa/scpdsi-tools@c73b22c8, `scpdsi_tools.py`:14-15: "Self-calibration follows Wells et al. (2004); the Palmer water balance and recursion follow Palmer (1965)."

**precip-index**

- bennyistanto/precip-index@1c2ef1c3, `README.md`:42: "Built upon the foundation of climate-indices by James Adams"

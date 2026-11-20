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
| PET, Thornthwaite | `climate_indices`, `xclim` (`TW48`), `PyETo` |
| PET, Hargreaves | `climate_indices`, `xclim` (`HG85`), `pyet`, `PyETo`, R `Evapotranspiration` |
| PET, FAO-56 Penman-Monteith | `climate_indices`, `xclim` (`FAO_PM98`), `pyet`, `PyETo`, R `Evapotranspiration` |
| EDDI and non-parametric | `climate_indices`, NOAA PSL EDDI, Climate Engine API, Earth Engine `GRIDMET/DROUGHT` |
| PDSI, PHDI, PMDI, Z-Index | `climate_indices` (`pdsi`), NCEI `pdi.f`, nClimDiv, `JonKing93/pdsi` |
| scPDSI | `climate_indices` (`scpdsi`), Wells, Goddard and Hayes C++, `Sibada/scPDSI`, `ktripa/scpdsi-tools` |

`precip-index` is excluded from the studies (REFIMPL-03) and is recorded only as lineage evidence for the question of which packages derive from `climate_indices`.

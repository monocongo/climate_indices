# Literature review and novelty check

**Ticket:** REFIMPL-05, [#1607](https://github.com/monocongo/climate_indices/issues/1607), part of the
[reference-implementation map](https://github.com/monocongo/climate_indices/issues/1602).
**Date:** 2026-11-22.
**Repository state read:** `monocongo/climate_indices` `main` at `55f3f28c`.

This file answers the ticket's question: what is already published on differences between SPI, SPEI, PET and
Palmer implementations, and what is new in the planned papers. It carries the annotated review, the prior-work
comparison, and one novelty statement per planned paper with the observation that would falsify it.

Two companions carry the evidence:

- [`references.bib`](references.bib) is the bibliography. Every entry was checked against a publisher record
  before it entered the file.
- [`references-verification.json`](references-verification.json) records what each record states. `verify_references.py`
  re-reads the Crossref records, and `tests/test_reference_bibliography.py` compares the two files offline.

## Method

- **Coverage.** All 31 entries are the ticket's seed list plus the two prior cross-package comparisons found
  while searching for it. Nothing was added from memory.
- **Journal articles.** Each DOI was read from the Crossref record (`api.crossref.org/works/{doi}`); 26 entries
  have a DOI. The DOI, year, volume, issue, pages and first author in `references.bib` agree with that record.
- **Works without a DOI.** Five entries: two conference preprints, two agency reports and one piece of software
  documentation. Each was checked against a named page: the NOAA copy of Palmer (1965), the Integrated Drought
  Management Programme scan of McKee et al. (1993), the WMO e-Library record for WMO-No. 1090, independent citing
  works for Heddinghaus and Sabol (1991), and Vonk's comparison notebook itself.
- **What "verified" means here.** The bibliographic fields are verified. A verified entry is not a read source:
  the annotation says what was read in this session and what rests on metadata or on a second-hand statement.
- **Search angles.** The queries asked for prior cross-package comparisons specifically, as apart from
  comparisons of distributions, of indices, or of one index against a product. The searches covered software
  comparison in drought indices, PET package benchmarking, Palmer implementation comparison, and SPI
  implementation differences. Repository material was read too: `research-notes.md`, `comparators.md`,
  `palmer-comparators.md`, `lineage.md`, `protocol.md`, and the earlier map's artifacts.

## Corrections to the seed list and to the repository

Each correction was read from the publisher record. The last column names the ticket that owns the fix.

| Item | Seed list or repository says | The record says | Owner |
| --- | --- | --- | --- |
| NCEI divisional SPI critique | "Baldwin and Chen (2020), *J. Appl. Meteor. Climatol.*" (`sources.yaml`, `tests/fixture/ncei_spi/provenance.json`, `method-matrix.json`, `VALIDATION.md`, `tests/test_ncei_spi_reference.py`) | Spade, de Beurs and Shafer (2020), *J. Appl. Meteor. Climatol.* 59(9), 1469-1480, doi:[10.1175/JAMC-D-19-0272.1](https://doi.org/10.1175/JAMC-D-19-0272.1). No Crossref work with this title has Baldwin or Chen as an author | REFIMPL-44 (VALIDATION.md and README claims); REFIMPL-09 for the matrix basis strings; the fixture note is corrected here |
| R `SEI` article year | "Allen and Otero (2024)" | *The R Journal* 16(4), 102-122, doi:[10.32614/RJ-2024-038](https://doi.org/10.32614/RJ-2024-038); the article is dated 2024-12-31 and Crossref registers the issue on 2025-07-14. BibTeX key `allen2025` uses the registered year | REFIMPL-06 (bibliography hygiene) |
| Rhee and Carbone (2007) title | Cited by author and year only | "A comparison of weekly monitoring methods of the Palmer drought index", *J. Climate* 20(24), 6033-6044, doi:[10.1175/2007JCLI1693.1](https://doi.org/10.1175/2007JCLI1693.1) | none needed; the repository carries no title |
| Ficklin et al. (2015) | "the Penman-Monteith extension of the Jacobi tool" | Two Ficklin 2015 papers exist. The relevant one is *Computers & Geosciences* 85, 136-141, doi:[10.1016/j.cageo.2015.09.013](https://doi.org/10.1016/j.cageo.2015.09.013); the other is an *ERL* drought-trends paper | REFIMPL-06 |
| Carbone et al. (2018) pagination | seed list gives a DOI only | *Int. J. Climatol.* 38(S1); Crossref and OpenAlex carry no page range and no article number, so the entry cites volume and issue only | REFIMPL-06 |

Only the `sources.yaml` entry and its status row in `sources/README.md` are corrected in this ticket; they are
the bibliography's own ledger. The fixture note, `VALIDATION.md`, and the method matrix are left to their owners,
because each is a scientific claim under a different ticket and a review, not a bibliographic field.

## Prior work: what has already compared implementations

The ticket asks for cross-package or cross-software comparisons, as distinct from comparisons of distributions,
of indices, or of a product against its own documentation. Six kinds of prior work exist.

| Work | What it compared | Data and scope | What it concluded | Limit for this project's novelty |
| --- | --- | --- | --- | --- |
| Jacobi et al. (2013), *Water Resour. Res.* | Their MATLAB Palmer tool against the NCDC Fortran | Palmer indices, checked on shared inputs | Discrepancies documented (abstract as recorded in [`research-notes.md`](research-notes.md); full text not readable here, Wiley returned 403) | One tool against one lineage, PDSI only, no pre-registered criteria and no per-axis attribution |
| Vonk's `spei` package comparison notebook (2023, read) | SPI 3-month from `spei`, `standard_precip`, `climate_indices`, R `SPEI`; SGI from `spei` and `pastas` | One station (DEBILT), monthly precipitation, one scale | "Difference is very small between SPEI and the standard_precip package"; the difference from R `SPEI` is attributed to gamma fitting as a hypothesis, and "more research is needed". The `climate_indices` cell is commented out because it could not be installed, and the R call fails in the published build | One station, one index, no reported number for `climate_indices`, no method-axis alignment, not peer reviewed |
| Vremec et al. (2024), *GMD* (read) | Each PET method in `PyEt` against the R `Evapotranspiration` package (Guo et al.) and against published values | Method-by-method benchmarks, daily PET | Agreement is the benchmarking criterion, and each new method must ship benchmark data and tests | PET equations only: no drought index, no distribution or estimator axis, and no effect of PET choice on SPEI or PDSI output |
| Allen and Otero (2025), *The R Journal* (read) | `SEI` positioned against `SCI`, `SPEI`, `standard-precip`, `spei`, `climate-indices`, `standaRdized` and `SPIGA` | Functional description of each package | Existing packages are limited in flexibility and assumptions; `SEI` is more general | Functional comparison, no numeric cross-package comparison and no per-axis attribution |
| Spade et al. (2020), *J. Appl. Meteor. Climatol.* | NCEI's operational divisional SPI against its documented calibration window | One provider product | The product over- and underestimates drought relative to its documentation | A product against its own documentation, not implementation against implementation |
| Guttman (1999), Stagge et al. (2015), Pieper et al. (2020), Laimighofer and Laaha (2022) | Candidate distributions, estimators and uncertainty components inside one standardized-index procedure | Precipitation and water-balance series | The distribution choice and the observation window dominate; the estimator matters less (as [`research-notes.md`](research-notes.md) records for Laimighofer and Laaha) | Distribution and method studies, not implementations: no package is compared |

The Palmer literature adds no implementation comparison beyond Jacobi et al. (2013). Dai (2011) and van der
Schrier et al. (2013) compare index *forms* and produce datasets; Karl (1986) and Alley (1984) analyse the index's
sensitivity and assumptions; Ficklin et al. (2015) substitutes one PET method inside one tool.

## Annotated review

Each annotation gives what the source is for the planned papers and what was read. "Not read" means the
metadata was verified and the content was not opened, so no content claim is made beyond the title.

### Standardized indices

- **McKee et al. (1993).** The paper that introduced the SPI and the classification thresholds. Allen and Otero
  (2025) state that McKee et al. introduced the index and that the thresholds 1, 1.5 and 2 are the common
  moderate, severe and extreme levels. Not read beyond its title page.
- **Guttman (1999).** The SPI calculation algorithm, and the argument for Pearson III. Allen and Otero (2025)
  summarise the argument as the extra flexibility of a third parameter. Not read.
- **Vicente-Serrano et al. (2010).** The SPEI definition: a multiscalar index on the water balance, built to be
  sensitive to global warming. Not read.
- **Beguería et al. (2014).** The SPEI revisited: parameter fitting, evapotranspiration models, tools, datasets
  and drought monitoring. [`research-notes.md`](research-notes.md) records the abstract's recommendation of
  unbiased PWM. Not read in full.
- **Stagge et al. (2015).** Candidate distributions for SPI and SPEI. Allen and Otero (2025) summarise its
  result as the Weibull distribution fitting their monthly accumulations better than the alternatives. There is
  a published comment by Vicente-Serrano and Beguería (2016) and a response (Stagge et al. 2016), so any claim
  drawn from it must say which side it rests on. Not read.
- **Laimighofer and Laaha (2022).** Uncertainty components for SPI and SPEI. The abstract, as recorded in
  [`research-notes.md`](research-notes.md), reports that the distribution choice and the observation window
  dominate while the estimator and the goodness-of-fit test matter little. This is the basis for the protocol's
  [rung order](protocol.md), which is stated there as a lead rather than a verified finding.
- **Carbone et al. (2018).** Uncertainty in SPI estimates: a statistical account, not a package comparison. Not read.
- **Pieper et al. (2020).** A universal candidate distribution function for the SPI, for observations and
  simulations. Not read.
- **Hobbins et al. (2016).** The Evaporative Demand Drought Index, Part I: the definition that EDDI rests on.
  Its Part II (McEvoy et al. 2016, doi:[10.1175/JHM-D-15-0122.1](https://doi.org/10.1175/JHM-D-15-0122.1)) is a
  CONUS assessment, and it is not in this bibliography because the seed list names Part I only. Not read.
- **Farahmand and AghaKouchak (2015).** The non-parametric standardization framework: the basis for comparing
  empirical and parametric SPI, SPEI and EDDI. Not read.
- **WMO-No. 1090 (Svoboda, Hayes and Wood, 2012).** The user guide that fixes the SPI worked examples and the
  recommended practice. The repository already cites it for stage checks and for the Falls City example, which
  `VALIDATION.md` records as not a numerical reference. Read as a library record only.

### Software papers

- **Bourgault et al. (2023), `xclim`.** The software paper for one comparator. `research-notes.md` records that
  the package's own documentation claims SPI agreement with this library under `method="APP"` and
  `fitkwargs={"floc": 0}`, and that its changelog calls this library "the reference library". That claim is a
  lead for REFIMPL-17, not a finding.
- **Vonk (2025), `spei`.** The software paper for another comparator. `research-notes.md` records that its
  notebook compares against `standard_precip`, `climate_indices`, `pastas` and R `SPEI`.
- **Vremec et al. (2024), `PyEt` (read).** The model description for the PET comparator. Section 2 benchmarks
  20 methods against the R `Evapotranspiration` package and against published values, and states that new
  methods must ship benchmark data and tests. This is the closest precedent for the PET criterion of manuscript 1.
- **Allen and Otero (2025), `SEI` (read).** The software paper for a package that computes standardised indices
  in general. It names the other packages, including this one, and compares them functionally. It reports no
  numeric cross-package result.
- **Huntington et al. (2017), Climate Engine.** The software and data-service paper behind one provider
  comparator. Not read.
- **Abatzoglou (2013), gridMET.** The dataset paper behind the `GRIDMET/DROUGHT` provider comparators. Not read.

### Palmer family

- **Palmer (1965).** The primary source. The repository already traces 26 steps of it in
  `docs/algorithm_refs/palmer-provenance.md` (REFIMPL-26, #1628), and that trace, not this entry, is where every
  Palmer claim in the papers must rest.
- **Alley (1984).** The limitations and assumptions of the index: the source for the index's known weaknesses.
- **Karl (1986).** The sensitivity of the PDSI and the Z-Index to the calibration coefficients, including
  potential evapotranspiration. The source for the PET-choice question in manuscript 2.
- **Heddinghaus and Sabol (1991).** The review that the PHDI and PMDI definitions trace to. A conference
  preprint with no DOI and no publisher record; the citation is verified against independent citing works.
- **Wells et al. (2004).** The scPDSI definition. Earlier tickets could not retrieve the full text (HTTP 403) and
  this session did not retry, so every statement about Wells' recursion rests on the port and on published
  descriptions.
- **Jacobi et al. (2013).** The MATLAB tool, checked against the NCDC Fortran. The closest precedent for
  manuscript 2. The publisher page returned 403, so the content claim is carried from `research-notes.md`.
- **Ficklin et al. (2015).** The Penman-Monteith PET method inside a Palmer tool: a precedent for the PET axis
  of manuscript 2, not for its provenance axis.
- **Rhee and Carbone (2007).** Weekly monitoring methods of the Palmer index: the precedent the `GRIDMET/DROUGHT`
  PDSI formulation follows, as `research-notes.md` records from the provider documentation.
- **Dai (2011).** Trends in the various forms of the PDSI: a comparison of index *forms* on one analysis.
- **van der Schrier et al. (2013).** The scPDSI-based global dataset: a product, and the example of a
  well-documented scPDSI product this library does not reproduce.
- **Vose et al. (2014).** nClimDiv: the dataset behind the 344-division PDSI and SPI references. The source for
  the input side of every nClimDiv comparison.
- **Spade et al. (2020).** The NCEI divisional SPI critique, misattributed in this repository (see the
  corrections table). The source for the calibration-window finding in
  `tests/fixture/ncei_spi/provenance.json`.

### Input and index uncertainty

- **Hoffmann et al. (2020).** Uncertainty in drought from index and data selection: PDSI, SPEI and SPI compared
  against soil moisture from two GLDAS generations. This is the closest precedent for a multi-index uncertainty
  statement, and it compares indices, not implementations.

## Novelty statements

The portfolio decision is REFIMPL-01 ([#1603](https://github.com/monocongo/climate_indices/issues/1603)) and is
still open, so the first two statements follow the two manuscripts the [map](https://github.com/monocongo/climate_indices/issues/1602)
names. The third is conditional. Each statement names the nearest prior work and the observation that would
falsify it. None of them claims that no such work exists anywhere; each says what was searched and what was not.

### Manuscript 1: cross-package intercomparison of SPI, SPEI and PET

**Nearest prior work.** Vonk's `spei` comparison notebook is the nearest published comparison of packages, and
Vremec et al. (2024) is the nearest benchmark of PET methods. Jacobi et al. (2013) is the nearest
implementation comparison in the Palmer family.

**Claim.** The planned paper differs from these in four ways. First, it aligns the method axes explicitly, at
pre-registered rungs, instead of comparing shipped defaults. Second, it reports each criterion separately, with
no composite score, so a package can be better on one criterion and worse on another. Third, it attributes the
residual difference per axis and logs every disagreement in a public ledger. Fourth, it reports provenance for
every run, including the calibration years and zero-handling inputs that each package actually used. The
comparison covers many locations rather than one station, and it treats PET in the role it plays inside SPEI and
Palmer rather than as an isolated equation.

**Falsified if.** A published, peer-reviewed work or a maintained public benchmark already does all of the
following on shared inputs: includes this library and at least two of `xclim`, `spei`, `standard_precip`, R
`SPEI` and R `SCI`; aligns distribution, estimator, zero handling and calibration window; reports a per-axis
attribution of the residual difference; and covers more than one location with released provenance. A work that
does three of the four narrows the claim but does not falsify it, and the paper must then cite it and say which
part is new. A single-station notebook, an unmerged pull request, or a comparison of shipped defaults is not a
falsifier.

### Manuscript 2: Palmer family validation, provenance and performance

**Nearest prior work.** Jacobi et al. (2013) documents discrepancies between one MATLAB tool and the NCDC
Fortran. Spade et al. (2020) documents one product's departure from its own documentation. Dai (2011), van der
Schrier et al. (2013) and Vose et al. (2014) compare index forms or supply products.

**Claim.** The planned paper differs in three ways. First, it traces every equation and coefficient of two
established Palmer lineages to the source statements and to the code lines, as a table a reader can re-walk.
Second, it quantifies the divergence between the `pdi.f` and Wells recursions on the operational 344-division
product, with six code-level causes and the share of disagreeing months each explains. Third, it reports the
Rust kernels' parity and performance as part of the same provenance record. No prior work found here combines a
statement-level provenance trace with a measured divergence between the two lineages on a national product.

**Falsified if.** A published work already (a) traces these implementations to source statements at
equation-and-coefficient resolution, and (b) reports the month share and per-cause decomposition of the `pdi.f`
versus Wells disagreement on a comparable domain. A work that does (a) alone, or that compares a single tool
against a single lineage without a cause decomposition, narrows the claim without falsifying it.

### JOSS software paper (conditional)

**Nearest prior work.** The software papers above, in particular Bourgault et al. (2023), Vonk (2025), Vremec et
al. (2024) and Allen and Otero (2025).

**Claim.** The paper would be new only if REFIMPL-01 includes it and if it states a claim those papers do not:
that this library reproduces named authoritative products and two Palmer lineages at stated tolerances, with
the provenance and the divergence record cited. A JOSS paper that only announces the library would add nothing
those papers already provide.

**Falsified if.** REFIMPL-01 decides the portfolio excludes a JOSS paper, or the paper's claims reduce to a
restatement of the four software papers above.

## Not verified, and for the maintainer

- **Publisher-blocked full texts.** Wiley returned 403 for Jacobi et al. (2013) in this session, and for Wells et
  al. (2004) in an earlier ticket. The AMS articles were not opened. Bibliographic fields are verified; content
  claims for the unopened works are carried from `research-notes.md` and are marked as carried.
- **Pagination absent from the record.** Carbone et al. (2018) has no page range in Crossref or OpenAlex.
- **Second-hand content.** The annotations for Guttman (1999), Stagge et al. (2015), McKee et al. (1993) and
  Beguería et al. (2014) rest on statements in Allen and Otero (2025) or in `research-notes.md`. Reading the
  originals would strengthen them and is worth doing before submission.
- **Search limits.** The searches ran on Crossref and general web search. They did not cover subscription
  indexes (Web of Science, Scopus), nor non-English literature, nor the grey literature of package repositories
  beyond the ones named. A pre-submission check should repeat the novelty search there.
- **Portfolio.** REFIMPL-01 ([#1603](https://github.com/monocongo/climate_indices/issues/1603)) owns the final
  portfolio and headline claims; these novelty statements become final only after that decision.

## What this ticket resolves in `research-notes.md`

`research-notes.md` lists twelve works whose bibliographic details were unverified and the authorship of the R
`SEI` article. This ticket verifies all thirteen: see `references.bib` and the corrections table above. The two
`research-notes.md` items that remain open are not bibliographic: which Palmer (1965) tables suffice for
stage-by-stage fixtures (REFIMPL-28, #1630) and the unretrievable Wells et al. (2004) text.

## What this means for the other tickets

- REFIMPL-17 to REFIMPL-23 must cite the prior work in the table above in their pre-registrations, and say in one
  sentence how their claim differs from it.
- REFIMPL-24 ([#1626](https://github.com/monocongo/climate_indices/issues/1626)) inherits no new implementation
  discrepancy from this ticket; the citation misattribution in the corrections table is a documentation defect,
  not an implementation difference.
- REFIMPL-41 and REFIMPL-42 must draw citations from `references.bib` and state the falsifier of each novelty
  claim in their introductions.
- REFIMPL-44 ([#1646](https://github.com/monocongo/climate_indices/issues/1646)) owns the `VALIDATION.md` and
  README wording that still carries the "Baldwin and Chen (2020)" attribution.

# KBDI reference-oracle research

**Research date:** 2026-09-08  
**Scope:** An independent regression oracle for the *original* Keetch–Byram
Drought Index (KBDI) requested in [issue #788][issue]. This is research only:
it does not choose the production API or add a fixture.

## Resolution

No candidate is trustworthy enough to approve **alone** as an end-to-end
regression oracle for the intended original formulation.

The best small, offline anchor is the 30-day **Figure 1 sample record** in
Keetch and Byram (1968), Research Paper SE-38 pp. 11--13. It is a published,
first-party input/output record, covers both single- and multi-day rain events,
and is tiny enough to transcribe into `tests/fixture/` with provenance. It is
not, however, an independent implementation: it exercises the report's
integer lookup tables only, begins from an already-established index of 164,
and uses Table 4 rather than an exact supplied annual-rainfall value. It must
therefore be one leg of a triangulated oracle, not a claim of independent
scientific validation by itself.

The recommended variant is the **corrected original English-unit Equation 18**
(the correction is explicit below), named as such in the eventual fixture
provenance. Do not silently substitute an Australian/Finkele, continuous-mm,
or regional operational KBDI.

## What the primary sources actually specify

### Original operational procedure

The canonical Forest Service record identifies the report as Keetch and Byram,
1968, *A Drought Index for Forest Fire Control*, Research Paper SE-38, 35 pp.
The source PDF used for this review was downloaded from the Forest Service
record on 2026-09-08 (SHA-256
`4a000f5e4da1eb6b414724549b847b0556a1b6d5d56793c5459b18f01c45c03b`).
The landing page and its stable TreeSearch identifier are preferable citations;
the downloaded bytes are recorded only to make this review auditable.

The source's field workflow is unambiguous:

1. Record the preceding 24-hour rain to 0.01 inch and temperature to the
   nearest degree Fahrenheit (0.5 rounds up); use daily maximum or the
   dry-bulb temperature at the basic observation ([SE-38, p. 11][se38-pdf]).
2. Compute **net rain** before the drought factor. An isolated daily rain of
   more than 0.20 inch has 0.20 inch removed; a value at or below 0.20 has no
   effect. During consecutive rainy days with no canopy drying, subtract 0.20
   only once, on the day cumulative rain first exceeds it; all subsequent rain
   in that wet spell is net rain. The first 24-hour period with no measurable
   rain ends the wet spell. The report gives separate snow wording
   ([SE-38, p. 12][se38-pdf]).
3. Subtract net rain, expressed as hundredths of an inch, from yesterday's
   KBDI. Look up the drought factor using that *reduced* index and the rounded
   temperature, then add the factor ([SE-38, pp. 12--13][se38-pdf]).
4. The operational KBDI is an integer 0--800, in hundredths of an inch of
   water deficit. It is not initialized automatically at zero: backtrack to a
   confidently saturated date (for example, snowmelt or 6--8 inches in a
   week), then calculate forward ([SE-38, p. 10][se38-pdf]).

Tables 1--5 are an intentional discretization, not merely presentation.
Their footnote says they were IBM tabulations of Equation 18 at 3-degree
increments from 50 to 108 deg F, with mean annual rainfalls of 15, 25, 35, 50,
and 70 inches; values were rounded to the nearest whole number with 0.5
rounded up ([SE-38, p. 4][se38-pdf]). Table 4 covers a 40--59 inch location
but its tabulated reference rainfall is 50 inches. The source says to use the
next-higher table at a table boundary (examples: 19.50 and 39.50 inches).

This creates a material contract choice:

- **1968 table workflow:** quantized temperature, deficit, annual-rainfall
  category, and integer daily factor. Figure 1 is an exact oracle for this
  workflow.
- **Equation workflow:** evaluate the daily equation at the supplied exact
  state, temperature, and mean annual rainfall. It produces continuous values
  unless a separate rounding contract is chosen. It will not exactly reproduce
  all table cells because the cells represent bins and a reference rainfall.

The report says a computer may use the equation but that the tables are best
for routine work ([SE-38, p. 31][se38-pdf]). Production planning must select
one of those semantics before treating a numerical fixture as authoritative.

### Printed equation and documented corrections

In SE-38 Appendix p. 31, Equation 18 is printed as:

```text
dQ = ((800 - Q) * (0.968 * exp(0.0486 * T) - 0.830)
      / (1 + 10.88 * exp(-0.0441 * R))) * dτ * 10^-3
```

where `dQ` and `Q` are hundredths of an inch, `T` is deg F, `R` is mean annual
rainfall in inches, and `dτ = 1` day. This literal `0.830` is a typographical
error. Alexander (1990), p. 23, records the corrected final numerator
constant as **8.30**, supplies both English and SI forms, and says the original
drought-factor tables based on Equation 18 are nevertheless correct. Its SI
form uses 203.2 mm, deg C, millimetres, `0.0875 * T + 1.5552`, and
`-0.001736 * R` ([Alexander 1990, pp. 23--24][alexander-1990]).

Alexander also identifies a derivational typo in SE-38 Equation 15: its final
constant must be `0.2113`, not `2.113`. Equation 15 is not the daily recurrence
when Equation 18 is used directly, but it matters to anyone reproducing the
appendix derivation. Alexander reports that a 1988 revised reprint corrected
both equations without announcing an erratum. The later formal corrigendum is
Alexander (1992), *Bulletin of the American Meteorological Society* 73(1),
pp. 61--62, DOI [10.1175/1520-0477-73.1.61][alexander-1992].

Consequently, a literal `0.830` implementation is not an "original" oracle:
it conflicts with the source's own correct tables and the documented
corrections. A future fixture must say **"Keetch--Byram (1968), Equation 18
corrected by Alexander (1990/1992)"**, not merely "KBDI".

## Candidate assessment

### 1. SE-38 Figure 1 and Tables 1--5 — best primary-source fixture anchor

- **Provenance and maintenance:** Original US Forest Service publication,
  archived by the Forest Service at the stable TreeSearch record. It is an
  archival specification, not maintained executable software.
- **Formula, order, units, initialization, rain, rounding:** Exactly the
  English-unit table workflow above. Figure 1 gives daily rain, net rain,
  rounded deg-F temperature, reduced previous index, lookup factor, and output
  for Table 4. Its initial index is explicitly 164; it does *not* demonstrate
  saturation initialization. It covers an isolated 0.66-inch event, a two-day
  0.16 + 0.09-inch wet spell, an isolated 0.22-inch event, and a wet spell
  that continues after exceeding the threshold (0.25 then 0.16 inch).
- **Original or later variant:** Original 1968 operational tables. The tables
  embody the corrected intended Equation 18, rather than the literal printed
  `0.830` typo.
- **Licensing/redistribution:** The Forest Service page makes the PDF freely
  available but gives no explicit reuse license. The report credits Forest
  Service staff; works prepared by US Government employees in their official
  duties are not protected by US copyright under 17 USC 105, but the public
  record does not itself make a legal-status declaration. Store a transcription
  of factual numeric values with a full citation and provenance; do not commit
  the scan or represent this research note as legal advice.
- **Reproducibility and edge coverage:** A checked-in transcription is offline
  and fully reproducible as a table lookup. It misses explicit saturation
  initialization, direct continuous-equation arithmetic, values outside Table
  4, the 0/800 clamps, exact 0.20 inch, snow/no-measurable-rain interpretation,
  missing data, and non-daily timestamps.
- **Verdict:** Approve as the first source fixture after an independent
  transcription review, but **not alone**.

### 2. Corrected Equation 18 and Alexander's published correction — required specification leg, not outputs

- **Provenance and maintenance:** SE-38 is the original source; Alexander's
  1990 *Fire Management Notes* article is a published correction, catalogued
  by FRAMES, and the 1992 AMS corrigendum is permanently identified by DOI.
  These are publications, not maintained code.
- **Formula/order/units:** Alexander p. 23 supplies the corrected English and
  SI equations. It confirms rain reduction precedes the factor through its
  definition of `Q` as yesterday's KBDI or the value reduced by daily net rain.
  It does not replace SE-38's field initialization or wet-spell procedure.
- **Initialization/rain/rounding:** The correction does not supply a new
  initialization or rounding rule. The primary table procedure remains the
  source for those details.
- **Original or later variant:** A correction to the original, not a regional
  variant. Do not use it as permission to adopt a later Australian rainfall
  convention.
- **Licensing/redistribution:** Cite rather than redistribute either article;
  no explicit reusable-data license was located for the articles. This does
  not block storing independently calculated numeric outputs with source
  provenance.
- **Reproducibility and edge coverage:** It makes a clean-room equation
  calculation reproducible, but publishes no daily input/output time series
  and does not cover operational edge cases.
- **Verdict:** Mandatory companion specification for a continuous-equation
  fixture; insufficient as an oracle by itself.

### 3. USFS Wildland Fire Assessment System (WFAS) / Drought.gov — official behavior, no reproducible point oracle

Drought.gov identifies WFAS as the USFS provider of current and forecast KBDI
maps. It says the inputs are station latitude, mean annual precipitation,
maximum dry-bulb temperature, and last-24-hour rainfall; it also states the
0.20-inch threshold and the reduce-then-increase order ([Drought.gov][drought-gov]).

- **Provenance/maintenance:** Official operational description and maps, but
  no version-pinned public implementation or documented historical point
  input/output archive was found.
- **Formula/units/init/rain/rounding:** Its documented behavior agrees with
  the original high-level English-unit procedure. It does not publish the
  exact equation variant, rounding, initialization, wet-spell state, or an
  inspectable source-code version.
- **Variant, license, reproducibility, edges:** It is represented as USFS
  KBDI, but the unexposed operational choices mean original-formula fidelity
  cannot be established. Current map images are network-dependent and not a
  byte-stable fixture; no explicit fixture-redistribution license or
  edge-case test vectors were found.
- **Verdict:** Useful operational corroboration only. Do not scrape maps or
  call their values an independent regression oracle.

### 4. Texas A&M Forest Service / Texas Weather Connection (TWC) — first-party published outputs, not reproducible inputs

TWC publishes dated county-summary CSVs, for example its
[2026-09-08 summary][twc-csv] (SHA-256 at retrieval:
`38f68ae42fe98e433fcf1e9215bb6ace3377d1e827facac03938880ebb142957`).
The site says station observations are manually interpolated by Texas A&M
Forest Service experts ([TWC landing page][twc]). Its technical PDF says the
operational calculation uses 4-km NEXRAD rainfall bias-corrected with gauges,
IDW-interpolated NWS maximum temperature, a 30-year PRISM mean annual
precipitation, and county aggregation; it accumulates rain from 08:00 to
08:00 ([TWC measurement guide][twc-guide]).

- **Provenance/maintenance:** Maintained Texas state/university product with
  dated output files and an archive, but no immutable release/version manifest
  or source repository was found. Product updates say the archive was rerun
  from 2015 after a 2023 map-projection change ([TWC update][twc-update]), and
  the FAQ says same-day products may be updated repeatedly.
- **Formula/order/units/init/rain/rounding:** The guide claims the original
  0.20-inch reduce-then-factor procedure and 0--800 English-unit range. It
  does not publish exact Equation-18 constants, wet-spell state handling,
  rounding, initialization, or source code. Its FAQ's 45-deg-F increase
  threshold is not a complete numerical contract.
- **Original or later variant:** It claims the K&B procedure but its gridded,
  radar/gauge-corrected, 08:00 operational product is a regional deployment;
  exact equivalence to SE-38 cannot be established.
- **Licensing/redistribution:** No explicit data license was found on the CSV
  or landing page. Do not commit its output without written reuse terms and a
  provenance record.
- **Reproducibility and edge coverage:** County min/max/average hides the
  underlying 4-km cells. The raw source choices, bias correction, interpolation,
  initial state, and historical revisions prevent reproducing a county value.
  Published outputs reveal no edge-case inputs.
- **Verdict:** A valuable real-world smoke test if TWC supplies a frozen
  point-level input/output bundle, but unsuitable now.

### 5. xclim 0.62.0 — maintained, tested, explicitly a Finkele/FFDI variant

The xclim release tag `v0.62.0` resolves to commit
`6328035cce3661409acae5a11d7f00d45a1bc82f` (2026-08-17). Its
[implementation][xclim-source] is Apache-2.0 and its
[tests][xclim-tests] contain hand-calculated examples.

- **Formula/order/units:** Inputs and output are millimetres and deg C. It
  initializes a 5.0-mm rainfall allowance; each positive-rain day consumes the
  remaining allowance, while a non-positive-rain day resets it. It computes
  effective rain, adds continuous ET minus that rain, then clamps every value
  to `[0, 203.2]` mm. Its constants are the SI/Finkele form
  (`0.0875`, `1.5552`, `-0.00173`), not an exact unit conversion contract for
  SE-38.
- **Initialization/rain/rounding:** Default initial KBDI is zero; there is no
  SE-38 integer table lookup or temperature rounding. The 5.0-mm threshold
  differs from 0.20 inch (5.08 mm) and the "positive" test is a different
  definition of measurable rain.
- **Original or later variant:** The public docstring explicitly says it
  follows Finkele (2006) for Australian FFDI use, with a 203.2-mm cap. Its own
  integration-test comment says that available ClimInd/CEMS definitions differ
  and that it could not find high-quality external KBDI data/code.
- **Licensing/reproducibility/edges:** Excellent pinned-source
  reproducibility, permissive licensing, and tests for zero rain, multi-day
  rain, cap, and initialization. Its expected values are hand calculations,
  not an external data product, and do not cover the SE-38 tables or original
  rain threshold.
- **Verdict:** Reject as an oracle for original KBDI. It can only validate a
  separately named xclim/Finkele compatibility mode.

### 6. ClimInd 0.1-3 — corrected formula cross-check, but different initialization and no KBDI expected-value test

The CRAN source release [ClimInd 0.1-3][climind-tarball] was published
2021-04-10 (tarball SHA-256
`e75dc6ef83256301ef4790e32c145732fbbfb1b05bd7c14cd0f34e1b0068b358`),
under GPL-3.0-or-later. Its KBDI function is
[`R/kbdindex.R`][climind-source] in the source archive.

- **Formula/order/units:** It converts deg C/mm inputs to deg F/inches,
  derives mean annual precipitation from the supplied record, applies a
  0.20-inch wet-spell calculation, subtracts net rain first, then applies the
  corrected continuous `8.30` Equation 18. It returns continuous mm values.
- **Initialization/rain/rounding:** It initializes from a caller-provided
  saturation date or the first week totalling at least 5 mm, drops rows with
  missing temperature or precipitation, begins the recurrence at its second
  retained row, and clamps negative outputs only after the loop. It has no
  upper-800 clamp or SE-38 table/rounding implementation.
- **Original or later variant:** It describes Alexander's corrections, but its
  start-date/rainy-week behavior and missing-row deletion are package choices,
  not the original field contract.
- **Licensing/reproducibility/edges:** A versioned, redistributable GPL source
  release, but GPL source must not be incorporated into this BSD-3-Clause
  project. The release's test tree contains no KBDI expected-value test.
  It can reproduce an equation-only cross-check only after carefully aligning
  start state and avoiding its unmatched behavior.
- **Verdict:** Useful independent formula review tool outside this repository;
  not a regression oracle and not a source to vendor.

### 7. `subond/kbdi-ffdi` v0.2-alpha — pinned source but dormant and untested

The `v0.2-alpha` tag is commit
`f0f05afbfa43ef62dedc92a5ca1f4ce2ca17b4b3` (last pushed 2019-05-05), under
MIT. Its [KBDI source][subond-source] uses a 5.08-mm threshold and corrected
metric `8.30` formula.

- **Formula/order/units:** It accepts raster-shaped deg-C/mm data, converts the
  recurrence in metric units, calculates net rain before ET, and clamps only
  at zero. Mean annual rain can be supplied or is calculated from complete
  calendar years.
- **Initialization/rain/rounding:** Default initial state is zero. It resets a
  wet spell only at exactly zero daily rain, has no SE-38 table rounding or
  upper cap, and has no published numeric regression tests. Its array boolean
  expressions also make it unsuitable to treat as a reviewed multi-cell
  reference implementation.
- **Original or later variant, licensing, reproducibility, edges:** It is an
  unaffiliated raster/FFDI project rather than an official or maintained
  original-KBDI implementation. The pin and MIT license make inspection
  reproducible, but not its scientific authority; no external input/output
  fixtures or complete edge coverage exist.
- **Verdict:** Reject. At most, use once as a qualitative, no-rain SI sanity
  comparison after independently reviewing its output.

## Recommended triangulated oracle procedure

This procedure deliberately has no dependency on future `climate_indices`
production code.

1. **Freeze the source-table anchor.** After two reviewers independently
   transcribe Figure 1, commit only its inputs, expected outputs, and a
   `provenance.json` (source pages/URL, download date, source-PDF hash,
   transcription reviewers, units, and the explicit `table_workflow_1968`
   variant). Do not commit the PDF. The minimum values are below. The fixture
   should provide the previous state `164` rather than pretend the source
   initialized it.
2. **Freeze a separate equation anchor.** Once the production decision is
   explicitly `corrected_equation_18` rather than table lookup, select a
   handful of decimal-safe daily cases covering no rain, exactly 0.20 inch,
   a threshold-crossing wet spell, continuing rain, rain greater than the
   state, and states near 0 and 800. Two reviewers should independently write
   small one-purpose calculators in different languages directly from the
   corrected equation and SE-38 rain state machine. Neither calculator may
   import, copy, or call `climate_indices`; retain their commands and hashed
   stdout in the fixture provenance. Freeze only their agreed outputs and a
   tolerance justified by units/rounding.
3. **Use existing implementations only as a third check.** Run a no-rain,
   interior recurrence through ClimInd outside this repository after aligning
   units and its shifted initialization. xclim may be used only as a looser
   Finkele/SI sanity check. A disagreement is a review trigger, not a reason
   to replace the original fixture with a regional variant.
4. **Keep platform contracts separate.** Missing-value propagation,
   contiguous-day validation, an explicit initial state/default, and
   xarray/CLI behavior are requirements of issue #786, not evidence supplied
   by the 1968 sources. Test those with small project-owned contract tests;
   label them regression coverage, not external validation.

This triangulation gives (a) published original outputs for the table state
machine, (b) independently derived corrected-equation outputs, and (c) a
separate maintained implementation as a limited review signal. It is stronger
than a large mutable operational dataset and remains offline in CI.

### Minimum future Figure 1 fixture

Use Table 4 (its tabulated reference annual rainfall is 50 inches), English
units, and `initial_kbdi = 164` hundredths of an inch. Blank rain cells in the
published form are represented below as `0.00`.

```text
precipitation_in = [
  0, 0, .66, 0, .23, 0, .16, .09, 0, 0, .08, .03, 0, .22, 0,
  .21, 0, 0, 0, .01, 0, 0, 0, 0, 0, 0, 0, 0, .25, .16,
]
temperature_f = [
  79, 75, 70, 76, 79, 84, 65, 66, 83, 70, 67, 65, 76, 69, 65,
  75, 78, 85, 88, 79, 69, 75, 84, 89, 93, 92, 96, 91, 78, 83,
]
expected_kbdi_hundredths_in = [
  174, 182, 142, 151, 159, 173, 177, 176, 190, 196,
  200, 204, 212, 215, 219, 226, 235, 248, 263, 271,
  276, 283, 295, 311, 328, 345, 365, 378, 380, 374,
]
```

For transcription review, Figure 1 additionally gives net-rain values
`[0, 0, .46, 0, .03, 0, 0, .05, 0, 0, 0, 0, 0, .02, 0, .01,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .05, .16]` and drought factors
`[10, 8, 6, 9, 11, 14, 4, 4, 14, 6, 4, 4, 8, 5, 4, 8, 9, 13, 15,
8, 5, 7, 12, 16, 17, 17, 20, 13, 7, 10]`. Those values make the operation
order auditable without deriving expected KBDI from production code.

## Sources

- [Keetch, J. J.; Byram, G. M. (1968), *A Drought Index for Forest Fire
  Control*, Res. Pap. SE-38, Forest Service TreeSearch record][se38-record].
  The relevant printed pages are 4 (table derivation/rounding), 5--9 (tables),
  10--13 (initialization and daily procedure), and 31 (Equation 18).
- [Alexander, M. E. (1990), *Computer Calculation of the Keetch-Byram Drought
  Index—Programmers Beware!*, *Fire Management Notes* 51(4):23--25][alexander-1990].
  The correction and unit equations are on p. 23; the revised-reprint and
  table statements are on p. 24. [FRAMES' catalog record][alexander-frames]
  preserves the bibliographic record and correction summary.
- [Alexander, M. E. (1992), *The Keetch–Byram Drought Index: A Corrigendum*,
  *Bulletin of the American Meteorological Society* 73(1):61--62][alexander-1992].
- [Drought.gov's USFS/WFAS KBDI description][drought-gov].
- [Texas Weather Connection KBDI page][twc], [measurement guide][twc-guide],
  [January 2023 product update][twc-update], and [dated example CSV][twc-csv].
- [xclim `v0.62.0` implementation][xclim-source] and
  [tests][xclim-tests].
- [ClimInd 0.1-3 CRAN source tarball][climind-tarball] and its
  [`kbdindex.R` source at the project's pinned commit][climind-source].
- [`subond/kbdi-ffdi` v0.2-alpha source][subond-source].
- [17 USC 105][usc-105] (government-work copyright rule).

[issue]: https://github.com/monocongo/climate_indices/issues/788
[se38-record]: https://research.fs.usda.gov/treesearch/40
[se38-pdf]: https://research.fs.usda.gov/download/treesearch/40.pdf
[alexander-1990]: https://library.ignfa.gov.in/wp-content/uploads/2026/01/computer-calculation-of-the-keetch-byram-drought-index-programmers-beware-30112021.pdf
[alexander-frames]: https://www.frames.gov/catalog/10946
[alexander-1992]: https://doi.org/10.1175/1520-0477-73.1.61
[drought-gov]: https://www.drought.gov/data-maps-tools/keetch-byram-drought-index
[twc]: https://twc.tamu.edu/kbdi
[twc-guide]: https://twc.tamu.edu/docs/TFS_KBDI_Update.pdf
[twc-update]: https://twc.tamu.edu/docs/KBDI_Product_Updates_January2023.pdf
[twc-csv]: https://twc.tamu.edu/weather_images/summ/summ20260908.csv
[xclim-source]: https://github.com/Ouranosinc/xclim/blob/6328035cce3661409acae5a11d7f00d45a1bc82f/src/xclim/indices/fire/_ffdi.py#L44-L266
[xclim-tests]: https://github.com/Ouranosinc/xclim/blob/6328035cce3661409acae5a11d7f00d45a1bc82f/tests/test_ffdi.py#L15-L61
[climind-tarball]: https://cran.r-project.org/src/contrib/ClimInd_0.1-3.tar.gz
[climind-source]: https://gitlab.com/indecis-eu/indecis/-/blob/e36146e08ceb5ae3080bd87c702d3076cb48118e/R/kbdindex.R#L41-159
[subond-source]: https://github.com/subond/kbdi-ffdi/blob/f0f05afbfa43ef62dedc92a5ca1f4ce2ca17b4b3/kbdiffdi/indices/kbdi.py#L67-L249
[usc-105]: https://www.copyright.gov/title17/92chap1.html#105

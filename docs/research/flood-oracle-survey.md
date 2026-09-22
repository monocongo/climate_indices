# Flood oracle survey: PE, EDI, I_F, and API

Resolves: [#1101](https://github.com/monocongo/climate_indices/issues/1101) (child of the
`v3.1 — Flood subsystem foundation` milestone)

Research date: 2026-09-22. All access statements below describe what was actually
fetched on that date; anything that could not be fetched is marked as requiring manual
retrieval rather than asserted present.

## Question

Which publicly accessible references can validate this library's planned Effective
Precipitation (PE) kernel, Effective Drought Index (EDI), Flood Index (I_F), and
Antecedent Precipitation Index (API)? For each candidate: what it is, what exactly it
would validate, its classification under this survey's ticket-derived vocabulary
(**exact reference**, **independent implementation**, **digitized figure**,
**regression coverage**) — mapped to `VALIDATION.md`'s own tiers below — its
licensing, whether its values are tabulated or only plotted, its access URL/DOI, and its
limitations. Then name one primary oracle per index and record the remaining gaps.

The classification vocabulary is load-bearing. Issue #1101 asks for candidates to be
classified as **exact reference**, **independent implementation**, **digitized
figure**, or **regression source**. Those are the ticket's terms, not `VALIDATION.md`'s
own: `VALIDATION.md` itself says "an exact oracle only for the 1968 integer table
workflow" in the KBDI row and otherwise uses "characterization", "reference-reproduced",
and "independent implementation cross-validation". It never uses "exact reference" at
all, and it never uses "independent implementation" as a standalone label — only inside
that one compound term. This survey uses the ticket's spelling and maps it to the
`VALIDATION.md` tiers as follows: a published
worked example whose numbers are tabulated is an *exact reference*; a
published-but-figure-only series is a *digitized figure* with a stated read-off
uncertainty (the HDW / Srock et al. entry); a second implementation of the same
algorithm is an *independent implementation* (the scPDSI Wells-lineage fixtures); and
fixtures generated from this library's own output are *regression coverage* only.

Two rules keep the tiers honest. A figure-only source is never upgraded to "exact
reference" merely because it was once attached to a paper. And a source that supplies
only the formula, the definitional wording, or the specification - with no reproducible
numbers - is **specification-level**, which is not validation evidence at all. Several
candidates below fall in that last class; the recommendation table labels them as such
instead of promoting them.

Terms used below, defined once: the source papers call effective precipitation **EP**;
this survey and the library's tickets call the same quantity **PE** - the two are
interchangeable throughout. **MEP** is EP's per-calendar-day mean (the climatological
baseline), **DEP** its deviation from MEP, **SEP** the standardized deviation, **APD**
the accumulated precipitation deficit, and **PRN** the precipitation needed for a return
to normal - all defined in full in candidate 1 below. **AWRI** (Available Water
Resources Index) is PE normalized by the harmonic sum `sum 1/N`, from the Byun & Lee
(2002) restatement (candidate 2).

## Answer

- **PE kernel and EDI**: Byun & Wilhite (1999) Table 5 is the only located exact
  reference - tabulated end-to-end output for Hickman, Nebraska, re-verified against the
  AMS table image during this review - but it is a valid oracle only under the paper's
  *variable* summation duration, which #1099 has not yet decided, and only with a
  30-year per-calendar-day MEP baseline behind it (gaps 5 and 6). Byun & Lee (2002) is
  the openly accessible confirmation of the equation algebra.
- **I_F**: Deo et al. (2015) is the definitional source, but its abstract describes an
  exponentially decaying effective precipitation while #1105/#1107 plan the harmonic
  double sum. The two forms are not algebraically equivalent, so this is a real
  convention conflict, and whether the 2015 implementation matched its abstract is
  unverified because the full text is paywalled. The decision should be routed to
  **#1099** - it is not yet on that ticket's checklist, but #1099 owns the other flood-
  module conventions and #1107 can only consume this one, not make it (gap 1).
- **API**: no accessible source tabulates index values. Kohler & Linsley (1951) is not
  digitized online; the located implementations are an independent implementation with
  no published oracle values (a fixed-`k` `ahrapi`) or a different definition (the
  USACE/NRCS weighted method). Unless that changes on manual retrieval, the API row can
  only reach specification-level status, with independent-implementation and internal
  regression coverage only (gap 7).

No primary oracle recommended in this survey rests on a digitized series alone - Table 5's
tabulated values anchor the PE/EDI recommendation; the Figure 1/2 entries below are
supplementary, lower-tier evidence, not primary anchors. Maintainer scientific sign-off on
this survey is the ticket's gate.

## What climate_indices plans to compute

From the dependent flood tickets, so the oracle assessment is against the intended
functions and not a paraphrase:

- **PE kernel** ([#1105](https://github.com/monocongo/climate_indices/issues/1105)):
  `PE_t = sum_{n=1..D} [ (sum_{m=1..n} P_m) / n ]`, `D = 365` by default, evaluated as
  the fixed harmonic filter `PE_t = sum_m w_m P_m`, `w_m = H_D - H_{m-1}`. Leading `D-1`
  days are NaN.
- **EDI** ([#1106](https://github.com/monocongo/climate_indices/issues/1106)): per
  calendar day, `MEP`, `DEP = PE - MEP`, `PRN = DEP / sum_{n=1..D}(1/n)`,
  `EDI = PRN / SD(PRN)`, with the variable-duration extension applied or omitted per
  [#1099](https://github.com/monocongo/climate_indices/issues/1099).
- **I_F** ([#1107](https://github.com/monocongo/climate_indices/issues/1107)):
  `I_F = (PE - mean(PE_max)) / SD(PE_max)`, with the annual-maximum (water year vs
  calendar year) and calibration-window conventions still to be fixed.
- **API** ([#1109](https://github.com/monocongo/climate_indices/issues/1109)):
  `API_t = k * API_{t-1} + P_t`, `0 < k < 1`, with the lag convention documented
  against the alternative `k * (API_{t-1} + P_{t-1})`.

Any oracle must therefore pin down four conventions, not just a formula: the depletion
function, the summation duration, the calendar-day vs annual-maximum normalization, and
the handling of leading windows and missing days.

## Candidates investigated

### 1. Byun & Wilhite (1999), *Journal of Climate* 12, 2747-2756 - originating paper (PE, EDI)

- **What it is**: the paper that introduced effective precipitation (EP), its mean
  (MEP), deviation (DEP), standardized deficit (SEP), accumulated precipitation deficit
  (APD), precipitation needed for a return to normal (PRN), and the EDI. It defines
  three candidate depletion functions and explicitly declines to resolve which is best,
  testing two of them on the High Plains.
- **Full-text access actually obtained**: fetched 2026-09-22 via
  `https://journals.ametsoc.org/doi/pdf/10.1175/1520-0442(1999)012%3C2747:OQODSA%3E2.0.CO;2`
  (the AMS PDF endpoint - not HTML) and
  `https://journals.ametsoc.org/view/journals/clim/12/9/full-i1520-0442-12-9-2747-t05.gif`
  (Table 5). The AMS XML-view path and the plain DOI landing page returned CAPTCHA/403 on
  both the original fetch and on a later re-check, so this access is **not reproducible
  on re-fetch**; treat the original PDF/GIF fetch as unconfirmed rather than as standing
  access. The metadata record is also at
  `https://digitalcommons.unl.edu/droughtfacpub/32` (abstract fetched; PDF blocked by
  Cloudflare). Semantic Scholar independently confirms the AMS PDF as BRONZE open access
  (`isOpenAccess: true`, free to read, no license). DOI:
  [10.1175/1520-0442(1999)012<2747:OQODSA>2.0.CO;2](https://doi.org/10.1175/1520-0442(1999)012%3C2747:OQODSA%3E2.0.CO;2).
- **What it would validate**: the PE kernel itself and the whole
  EP -> MEP -> DEP -> SEP -> PRN -> EDI chain, including the variable-duration summation
  convention.
  - Equation (2) (the form the library plans) is the double sum. The paper's own
    two-day worked identity is exact: `EP_2 = P_1 + (P_1 + P_2)/2`; the alternative
    Equation (3) gives `EP_2 = (2 P_1 + P_2)/3`; Equation (1) is an exponential decay,
    `EP_2 = P_1 exp(-1/2) + P_2 exp(-2/2)`.
  - Table 2 records that MEP is a 30-year mean of EP per calendar day (5-day running
    mean); Table 3 defines dry duration, the summation duration `DS = 365 + dry
    duration`, and drought duration as consecutive `EDI < -1.0`; Table 4 defines
    `EDI_j` as the standardized `PRN_j`.
  - **`DS` is not defined consistently within the paper.** Table 3 reads `DS = 365 +
    dry duration`, while the article's own worked example says "The DS of 5 June is 399
    (365 + 35 - 1)" - a one-day difference that shifts every subsequent value. The two
    readings need to be separated before `DS` is implemented, and Table 3 should be
    re-read during fixture extraction.
  - The paper does not settle the depletion-function choice: "choosing one of the two or
    finding another equation for the best result is beyond the scope of this study". It
    prefers Equation (2) over (3) for "the upper basins of rivers, mountainous areas,
    or sandy areas" and Equation (3) for "lower basins of rivers, areas with good water
    retention, or long-term drought". Any statement that the 1999 paper selected the
    double sum outright is a misreading - which is why the convention question in gap 1
    is a decision, not a lookup.
  - **Table 5 is tabulated numeric output** for the Hickman, Nebraska (USA) example of
    1 January 1995 - 31 December 1996, columns for Equation (2) and Equation (3):
    dry duration 232-493 vs 250-494; drought duration 259-493 vs 366-493; minimum CNS
    (unit and column identity unconfirmed - see the fixture caveats below)
    -299.8 on day 493 vs -256.3 on day 494; minimum APD -214.4 on day 484 vs -217.1 on
    day 484; minimum PRN -70.5 on day 484 vs -173.4 on day 484; minimum EDI -2.5 on
    day 469 vs -1.22 on day 469. The table's column headers are "(2) Fig. 2b" and
    "(3) Fig. 2c". (Transcribed from the Table 5 GIF by this survey and re-read against
    the same GIF during review; re-read once more before the numbers enter a committed
    fixture.)
  - Figures 1-3 are plotted: Figure 1 is the weight-vs-day-pass curve for the three
    equations (caption example: weight 6.4 one day before vs `1/365` at 365 days before
    for Equation (2)); Figure 2 is the daily precipitation and both index panels for
    Hickman; Figure 3 is the 113-station High Plains annual minimum series, which the
    paper cross-checks qualitatively against the recorded 1989 drought.
- **Classification**: **exact reference** for the tabulated Table 5 end-to-end values,
  **digitized figure** for the Figure 2 index-chain panels (2b/2c, `EDI * 100`-scaled per
  their own column headers - not raw EP, which is never separately plotted or tabulated)
  and for Figure 1's weight curve.
- **Licensing**: AMS copyright, no Creative Commons license found on the fetched page.
  Free to read is not the same as free to redistribute or to derive from. The existing
  digitized-figure precedent (HDW / Srock et al.) rests on a CC BY article
  (`atmosphere` 9(7):279); that precedent does **not** transfer here.
- **Limitations**: (i) reproducing Table 5 requires the Hickman 1995-1996 daily
  precipitation series, which is plotted (Figure 2a) but not tabulated, plus the paper's
  missing-data substitution (it began with 193 High Plains stations, discarded those
  missing more than 1% of data, leaving 113, then substituted nearest-station or
  calendar-day averages); (ii) it also requires the climatological baseline the
  anomalies are measured against - a multi-decade record of the same calendar days, for
  MEP and its standard deviation. Two years of Hickman data cannot produce them, and
  `EP` needs more than `DS` days of prior history before its first comparable value, so
  the baseline record and its warm-up are part of the oracle rather than an
  implementation detail (gap 5); (iii) the paper's EDI uses the *variable* `DS` and a
  per-calendar-day 5-day-running-mean MEP and ST(EP) - the library's `D = 365`
  fixed-window form will not reproduce it unless #1099 adopts the extension, and the
  standardization denominator is not pinned to one quantity, which changes the result;
  (iv) the paper's test is design-level, not independently implemented: it is the
  originating group's own arithmetic; (v) permission for a committed digitized
  derivative is unresolved.

### 2. Byun & Lee (2002), *J. Meteor. Soc. Japan* 80(1), 33-44 - originating-group algorithm restatement (PE, AWRI, earlier FI)

- **What it is**: the AWRI paper (same first author) that restates the effective-
  precipitation formulation and applies it to Korean rainy seasons. It is the accessible
  originating-group source for the exact kernel algebra.
- **Access**: open PDF on J-STAGE, fetched 2026-09-22:
  `https://www.jstage.jst.go.jp/article/jmsj/80/1/80_1_33/_pdf`. DOI:
  [10.2151/jmsj.80.33](https://doi.org/10.2151/jmsj.80.33).
- **What it would validate**: the kernel formula and the harmonic normalization,
  verbatim:
  - Eq. (1): `E = sum_{N=1..D} ( sum_{m=1..N} P_m / N )`, where `P_m` is the
    precipitation of `m` days before and `D` the accumulation duration;
  - Eq. (2): `W = E / ( sum_{N=1..D} 1/N )` (the AWRI), with `D = 365` in that study;
  - Eq. (5): the drought index `DI = ( W' - Mean(W') ) / St( W' - Mean(W') )`, which it
    explicitly attributes back to Byun & Wilhite (1999);
  - Eq. (3)-(4): an earlier **Flood Index** `FI = V / St(V)`, `V = W - A_max(W)`, where
    `A_max(W)` is the mean of the yearly maximum of W. It also states that the 1999
    paper uses `D` as a variable while this paper fixes `D = 365`. (This paper's
    second-hand summary of what the 1999 paper concluded about the depletion function
    does not match the 1999 text itself; see candidate 1's "What it would validate"
    discussion of the depletion function above.)
- **Classification**: **specification-level** - it fixes the equation algebra but
  carries no numeric oracle: AWRI/FI series are plotted. The tables are about monsoon
  onset definitions, not index values.
- **Licensing**: J-STAGE free access, copyright the Meteorological Society of Japan; no
  CC license found. Same redistribution caveat as candidate 1.
- **Limitations**: it validates the *formula*, not this library's numeric output; the
  paper's AWRI is divided by `sum 1/N`, which is a different output from raw PE. This
  source also shows that a "flood index" existed in the Byun lineage (2002) before Deo
  et al. (2015), normalized against the annual maximum of AWRI rather than of PE -
  oracle conflation risk.

### 3. Byun & Jung (1998), *J. Korea Water Resour. Assoc.* 31(6), 657-665 - originating-group flood precursor

- **What it is**: "Quantified diagnosis of flood possibility by using effective
  precipitation index" (in Korean with an English abstract). Both Byun & Lee (2002) and
  Moishin et al. (2021) cite it as the effective-precipitation source for flood
  diagnosis. Moishin et al. romanize the authors as "Byeon, H.-R. and Jeong, J.-S.";
  Byun & Lee (2002) list it as Byun and Jung. The same paper under two spellings.
- **Access**: not fetched. No online copy located in this environment.
- **Classification**: **unverified** - not retrieved in this environment; assumed
  specification-level (an originating-group source) pending retrieval, per the "no
  reproducible numbers" rule above.
- **Licensing**: unknown; Korean Water Resources Association.
- **Limitations**: requires manual retrieval; unknown whether values are tabulated.

### 4. Deo, Byun, Adamowski & Kim (2015), *Water Resources Management* 29(11), 4075-4093 - originating paper (I_F)

- **What it is**: the paper that introduces the real-time flood monitoring index I_F for
  Brisbane and Lockyer Valley, with event severity, duration, peak danger and return
  periods.
- **Access**: paywalled. The abstract was fetched 2026-09-22 from
  `https://link.springer.com/article/10.1007/s11269-015-1046-3`; the standard browser
  path and the r.jina.ai render both returned only the reference list, and the Springer
  PDF is gated. Crossref records only the Springer text-and-data-mining license
  (`http://www.springer.com/tdm`); OpenAlex marks it closed (its only repository location
  is a DOI handle with no full text); Semantic Scholar reports CLOSED. DOI:
  [10.1007/s11269-015-1046-3](https://doi.org/10.1007/s11269-015-1046-3).
- **What it would validate**: the I_F definition and the derived event statistics. The
  abstract states I_F is "based on Effective Precipitation (P E)", computed "using
  exponentially-decaying time-reduction function", and "comparing and normalizing the
  P E per day with the means and standard deviations of yearly maximums in the
  hydrological period"; flood start is `I_F >= 0`, severity is the running sum of
  consecutive positive I_F, duration is the count of positive I_F days, and peak danger
  is the maximum I_F.
  - **The abstract itself tabulates event values**: Brisbane January 1974 `I_acc_F = 118`,
    `I_max_F = 4.4`, `D_F = 104 days`, `T = 106.2 years`; December 2010-January 2011
    `I_acc_F = 61.8`, `I_max_F = 2.6`, `D_F = 89 days`, `T = 53 years`; and Lockyer
    Valley December 2010-January 2011 described as most severe with `T = 104.4 years`.
    The superlative is the abstract's own wording and reads as scoped to that valley
    rather than to every event, since the Brisbane 1974 return period it lists is
    longer. All three values are abstract-level until the full text is retrieved.
- **Classification**: **specification-level** for the definition and the abstract-level
  event statistics (not reproducible without the Brisbane/Lockyer record); the daily
  series and any in-paper tables could not be inspected, so their tabulated-vs-plotted
  status is **unverified and requires manual retrieval**.
- **Licensing**: Springer, all rights reserved (TDM only). Digitizing or transcribing
  in-paper figures or tables into a committed fixture is a permission question.
- **Limitations** - the most important convention finding in this survey:
  1. **The 2015 abstract describes an exponentially decaying PE; the kernel the 2015
     paper implements is unverified.** The abstract says the index is computed "using
     exponentially-decaying time-reduction function", and the 2019 follow-up (candidate
     8) repeats that wording. That is the Byun & Wilhite Equation (1) form, which is not
     the harmonic double sum the library plans - the two are not algebraically
     equivalent - so a genuine convention question exists for I_F. The abstract alone
     does not settle what the 2015 implementation did, though: Moishin et al. (2021,
     candidate 5), co-authored by Deo, implements the *double sum* for the same index
     and attributes that PE form to Byun & Jung (1998). The defensible statement is
     therefore "the abstracts describe exponential PE; the implemented kernel is
     unverified until the full text or Byun & Jung (1998) is retrieved". The decision
     belongs to **#1099**, which owns the I_F convention that #1107 consumes.
  2. **Normalization is against yearly maxima "in the hydrological period"** (a water
     year, not plainly a calendar year). #1107 flags this as unresolved, but it is the
     same convention decision as (1) and belongs with #1099.
  3. Reproducing the abstract's event numbers requires the Brisbane/Lockyer daily
     precipitation and the paper's decay constant, neither of which is publicly
     available as a dataset in this environment.
  4. The paper is the originating group's own application; a fixture from it is external
     to this codebase but not independent of the method's inventors.

### 5. Moishin, Deo, Prasad, Raj & Abdulla (2021), *IEEE Access* 9 - same-lineage, openly licensed (PE, AWRI, I_F)

- **What it is**: a ConvLSTM flood-forecasting study for nine Fiji sites that restates
  the I_F computation chain and forecasts it. Deo is a co-author of the 2015 I_F paper;
  this is same-lineage, not an independent group.
- **Access**: open PDF, fetched 2026-09-22:
  `https://ieeexplore.ieee.org/ielx7/6287639/9312710/09378529.pdf`. DOI:
  [10.1109/ACCESS.2021.3065939](https://doi.org/10.1109/ACCESS.2021.3065939).
- **What it would validate**: the algebra, with explicit equation numbers:
  - Eq. (6): `PE_i = sum_{N=1..D} [ sum_{m=1..N} P_m / N ]`, `1 <= m <= 365`;
  - Eq. (7): `AWRI = PE / W`;
  - Eq. (8): `W = sum_{n=1..D} 1/n`;
  - Eq. (9): `I_F = ( PE - mean(E_Pmax) ) / sigma(E_Pmax)`, where the mean and standard
    deviation are taken over the **yearly maximum daily PE** during 1991-2019.
  Its references identify the chain: PE from Byun & Jung (1998), AWRI from Byun & Lee
  (2002), I_F from Deo et al. (2015).
- **Classification**: **independent implementation of the algorithm** (a second worked
  computation of the same formulas, by the method's wider group); its numbers are
  application results for Fiji, not a fixture for the Brisbane/Lockyer values. I_F
  series are plotted; the five tables are model-performance tables, not I_F values.
- **Licensing**: **CC BY 4.0** (verified in the article PDF: "This work is licensed under
  a Creative Commons Attribution 4.0 License"). This is the one source with CC BY terms
  that states the Deo-lineage I_F equations; Chand et al. (2024) and Shen et al. (2025)
  below are also CC BY 4.0, but neither states that chain.
- **Limitations**: (i) it uses the double-sum PE while Deo et al. (2015) describes the
  exponential form, so it cannot arbitrate that mismatch; (ii) the Fiji hourly/daily
  rainfall input is from the Fiji Meteorological Service and is not released with the
  paper, so its plotted I_F series cannot be re-derived; (iii) same-lineage authorship
  means it is not independent validation of the inventors' method.

### 6. Moishin, `flood_monitoring_dss` (GitHub) - same-lineage code (PE, AWRI, FI)

- **What it is**: a Streamlit decision-support app whose `flood_index.py` computes
  `ep` (365-day double-sum), `awri = ep / H_365`, and
  `fi = (ep - mean(calendar-year maxima ep)) / stdev(...)`, then event runs at
  `fi > 0`, `fi >= 1`, `fi >= 1.5`, with duration, severity, AWRI and peak severity.
- **Access**: `https://github.com/memoishin/flood_monitoring_dss`, last push
  2021-04-26, checked 2026-09-22.
- **Classification**: **independent implementation**, but not usable as an oracle - like
  `rrk4910/EDI` below, it has a working implementation of the same lineage but **no
  committed expected values and no tests**, so agreement with it is not scientific
  validation.
- **Licensing**: none (no license file). It cannot be copied or vendored; only
  inspected, or reimplemented from its published algorithm.
- **Limitations**: its annual maxima are calendar-year (`year` from the input date),
  while Deo et al. (2015) says hydrological-period maxima; also unlicensed.

### 7. Deo et al. (2014), "Diagnosis of flood events in Brisbane (Australia) using a flood index based on daily effective precipitation" - conference precursor

- **What it is**: the European Commission / International Conference on Analysis and
  Management of Changing Risks for Natural Hazards paper that precedes the 2015 journal
  article, cited by Moishin et al. (2021) as reference [5].
- **Access**: OpenAlex records a published-version PDF at
  `http://eprints.usq.edu.au/26754/8/Deo_Byun_Adamowski_Kim_AP20_2014_PV.pdf`. The USQ
  eprints host refused connections on 2026-09-22 (and r.jina.ai could not reach it
  either), so the PDF could not be fetched.
- **Classification**: **unverified** - not retrieved. An originating-group published
  version, so plausibly promotable to exact reference once retrieved, but not classified
  as such yet, per the "never upgrade merely by association" rule above.
- **Licensing**: not determinable while the host is unreachable; USQ repositories often
  post publisher versions with unclear reuse terms - check before committing anything.
- **Limitations**: site availability. This is the most promising route to a full,
  same-group statement of the I_F equations and example tables without a Springer
  subscription, if the repository returns.

### 8. Deo et al. (2019), *Theoretical and Applied Climatology* 137, 1201-1215 - follow-up application

- **What it is**: the same I_F applied to Dhaka and Bogra, Bangladesh. The abstract
  fetched 2026-09-22 states I_F is again a daily effective-precipitation index using
  "an exponentially decaying time-reduction function", normalized against "the average
  and standard deviation of yearly maximums, within the considered hydrological
  period", with the same `I_F >= 0` event rules.
- **Access**: paywalled. DOI:
  [10.1007/s00704-018-2657-4](https://doi.org/10.1007/s00704-018-2657-4). OpenAlex
  reports no OA full text; its USQ eprints copy (34937) is on the same unreachable host.
- **Classification**: **specification-level** (definition wording only); numeric tables
  and figures unverified.
- **Licensing**: Springer, all rights reserved.
- **Limitations**: no tabulated I_F values accessible; the rainfall inputs and calibrations
  are not public in this environment.

### 9. Independent implementations: `rrk4910/EDI` (R) and `tidyindex` (R)

- **`rrk4910/EDI`**: an R package (v0.1.0, packaged 2019-12-05) whose `edi.R` reverses
  the series, builds a weighted double-loop sum, and z-scores it with `scale()`. The
  DESCRIPTION declares GPL-3; the repository has no LICENSE file. It is not on CRAN
  (checked against the CRAN package-name index 2026-09-22) and has no tests or expected
  values. **Classification: independent implementation**, but a weak one: its weighting
  loop does not match the Byun & Wilhite Eq. (2) identity as published, and it omits the
  variable-`DS` EDI standardization. Usable only as a cross-check of input/output
  plumbing, not as an oracle.
- **`tidyindex`** (`huizezhang-sherry/tidyindex`, MIT + file LICENSE, `idx_edi()` in
  `R/drought-indexes.R`): multiplies precipitation by `rev(digamma(row_number() + 1) -
  digamma(1))` - the harmonic numbers `H_n`, reversed - sums over a rolling window, and
  z-scores the result. **That is harmonic weighting, but it is not the kernel's filter.**
  The #1105 kernel's coefficient on the observation `m` days back is `H_D - H_{m-1}`;
  tidyindex's coefficients are `H_n` in reverse order. They agree at the most recent lag
  and differ at every other one - at `D = 2`, `[1.5, 1.0]` against the kernel's
  `[1.5, 0.5]`. **Classification: independent implementation of a harmonic-weight
  EDI**, cited as precedent that harmonic differential weighting is an established EDI
  construction, not as an implementation of this kernel. It is a periodic/rolling
  z-score EDI, not the variable-`DS` EDI of the 1999 paper, and it publishes no external
  expected values, so it is not an oracle.
- **`royalosyin/Calculate-Precipitation-based-Agricultural-Drought-Indices-with-Python`**
  (MIT): mentions EDI in its index list but contains no EDI implementation (checked the
  notebook's function definitions 2026-09-22). Excluded.
- **No package oracle exists on the standard channels**: the CRAN package-name index has
  no EDI package (its `drought`, `droughtevents`, `msdrought`, `PowerSDI`, `SCI`, and
  `SPEI` packages are SPI/SPEI/multivariate tools), and the PyPI name index has no EDI
  drought package (`pydrought` 0.2 is SPI-only).

### 10. Adjacent flood-index lineages (do not conflate with Deo et al. 2015)

- **Nosrati, Mohseni Saravi & Shahbazi (2010)**, "Investigation of Flood Event
  Possibility over Iran Using Flood Index", chapter in *Survival and Sustainability*,
  pp. 1355-1361,
  [10.1007/978-3-540-95991-5_127](https://doi.org/10.1007/978-3-540-95991-5_127).
  Closed; abstract fetched 2026-09-22 (Googlebot user-agent) states the FI is computed
  via the **Available Water Resources Index** and compares FI with AWRI - i.e., the
  Byun & Lee (2002) lineage, not Deo et al. (2015). Issue #1107 cites "Nosrati et al.
  (2011)"; Crossref/OpenAlex index no 2011 flood-index paper by Nosrati, so the citation
  needs correction before use. **Classification: independent implementation of a
  different flood-index definition**; not an oracle for Deo-style I_F.
- **Chand, Nguyen-Huy, Deo, Ghimire, Ali & Ghahramani (2024)**, *Water* 16(11), 1560,
  [10.3390/w16111560](https://doi.org/10.3390/w16111560), CC BY 4.0. Defines an hourly
  `SWRI` from a 24-hour water-resources index with weights
  `(H_24 - H_{m-1})/H_24` (`W = H_24 ~ 3.8`; e.g. 0.74 for two hours before), and
  normalizes against **mean monthly maxima**, not annual maxima. **Classification:
  independent implementation of an hourly extension**; it corroborates the harmonic
  weight algebra but uses a different normalization window, so it cannot validate I_F
  as specified.
- **Shen, Yang, Zhang, Chen & Li (2025)**, *Hydrology* 12(5), 104,
  [10.3390/hydrology12050104](https://doi.org/10.3390/hydrology12050104), CC BY 4.0.
  Uses the daily I_F ("daily effective precipitation") as a forecasting target with an
  independent deep-learning application; no reusable fixture data identified.
  **Classification: independent implementation of a downstream application**; not a
  numeric oracle by itself.
- **Earlier Byun-lineage flood index** (candidate 2's Eq. 3-4) normalizes AWRI, not PE.
  Any fixture must state which lineage it encodes.

### 11. Antecedent Precipitation Index candidates (API)

- **Kohler & Linsley (1951), "Predicting the runoff from storm rainfall", U.S. Weather
  Bureau Research Paper No. 34** - the specification the ticket names.
  - **Access**: no digitized copy located. The OpenLibrary work record is
    `https://openlibrary.org/works/OL7350422W` (edition `/books/OL17720246M`, 1951,
    10 pages, sourced from a Columbia University MARC record). Internet Archive has no
    scan; the NOAA Institutional Repository, HathiTrust, and Google Books routes were
    blocked or quota-exhausted on 2026-09-22. **Requires manual retrieval** (library or
    interlibrary loan; the NOAA Central Library likely holds the series).
  - **Classification**: **specification-level**, numeric content
    **unverified**. It is a U.S. Government work and therefore public domain if a copy
    is obtained - unlike candidates 1 and 4, a scan could be redistributed.
  - **Limitations**: whether the report tabulates index values or only figures a
    storm-runoff relation is unknown; the commonly cited recursion
    `API_t = k * API_{t-1} + P_t` could not be verified against the 1951 text itself in
    this environment, only against sources that cite it.
- **`robbub98/ahrapi`** (R, v0.1.0, MIT + file LICENSE, copyright Robert Bartram 2026)
  - **What it is**: an Antecedent Precipitation Analysis package for the July 2021 Ahr
    Valley flood; `R/compute_api.R` implements `API_t = k * API_{t-1} + P_t` with
    `k = 0.95` default, `initial = 0`, and cites "Kohler, M. A., & Linsley, R. K.
    (1951). Predicting the runoff from storm rainfall. U.S. Weather Bureau Research
    Paper, 34."
  - **Access**: `https://github.com/robbub98/ahrapi` (pushed 2026-08-05).
  - **Classification**: **independent implementation**. Its test suite covers file
    reading only (`tests/testthat/test-placeholder.R`, `test-read_radolan.R`) and
    contains no API expected values, so it is not an oracle.
  - **Limitations**: it sets missing precipitation to 0 to preserve recurrence, which
    differs from the fire-domain missing-data contract (`ADR-0007`) this library will
    follow; the vignette's derived values are station-analysis results, not a published
    worked example.
- **`alexgruber/pyapi`**: a 2019 Python class with `gamma = 0.85`, no license, no tests,
  one star. **Not usable** for a committed fixture.
- **USACE Antecedent Precipitation Tool** (`jDeters-USACE/Antecedent-Precipitation-Tool`,
  migrated to `erdc/Antecedent-Precipitation-Tool`, releases through v3.0.9; USACE
  public domain per 17 USC 105, with a joint-work caveat).
  - **What it is**: an operational agency tool, actively maintained. Its README states
    it evaluates antecedent precipitation with "30-day rolling totals and NRCS
    Engineering Field Handbook weighting factors (Combined Method)" citing Sprecher &
    Warne (2000), ERDC/EL TR-WRAP-00-1, and NRCS Engineering Field Handbook Chapter 19
    (1997); its drought determination uses PDSI, not API.
  - **Classification**: **not applicable** to this survey's API definition - it
    implements NRCS monthly weighting factors, not the Kohler & Linsley exponential
    recursion. It is not an oracle for #1109; it is evidence that "antecedent
    precipitation index" is not one definition in agency practice. The two
    cited URLs (`el.erdc.usace.army.mil/elpubs/...` and `info.usda.gov/CED/...`) were
    dead on 2026-09-22 - the ERDC and NRCS documents themselves require manual retrieval.
- **De Moraes et al. (2024)**, *International Journal of Geosciences* 15, 70-86,
  [10.4236/ijg.2024.151006](https://doi.org/10.4236/ijg.2024.151006), CC BY 4.0.
  An open-access application using a **modified** API whose recession coefficient depends
  on air temperature; it cites Kohler & Linsley (1951). **Classification: independent
  implementation of a variant**; it cannot validate a fixed-`k` recursion.

### 12. Operational EDI products - not verified

The ticket's example of an agency publishing EDI operationally could not be confirmed
from primary sources in this environment. The APCC "Global Drought" monitoring page
(`https://www.apcc21.org/monitoring/drought`) and the Korean national drought portal
(`https://www.drought.go.kr/`) are JavaScript shells with no index names in the served
HTML; KMA drought pages are navigation shells; `drought.kma.go.kr` refused connections.
No statement here claims an operational EDI product exists. If one is required, the APCC
and KMA/NDIAC portals need manual retrieval (record the product name, index definition,
basin/period, and license before use).

- **Classification**: **unverified** - no operational EDI product could be confirmed
  from primary sources in this environment.

## Recommendation: one primary oracle per index

Below, *exact reference* means the source carries tabulated, reproducible numbers, and
*specification-level* means it fixes the definition without supplying a usable numeric
oracle. The first two rows are validation evidence only under the variable-`DS`
convention; the last two are not validation evidence as they stand.

| Index | Primary oracle | Classification | Numeric anchor |
| --- | --- | --- | --- |
| PE kernel | Byun & Wilhite (1999), Eq. (2); Byun & Lee (2002) for the open algebra | exact reference, conditional on the variable-`DS` convention (Table 5); digitized figure for the plotted index-chain panels (not raw EP, never separately plotted) | Table 5 (Hickman, NE) for the variable-`DS` chain; Fig. 2b for the Eq. (2) index panel; Fig. 1 for the weight curve |
| EDI | Byun & Wilhite (1999), Eq. (9)/Table 5 | exact reference, conditional on the variable-`DS` convention and a multi-decade MEP baseline | min EDI -2.5 (Eq. 2) and -1.22 (Eq. 3) on day 469, Table 5 |
| I_F | Deo et al. (2015) | specification-level: definition and abstract-level event statistics; implemented kernel unverified; exponential-vs-double-sum conflict | none reproducible; the abstract's `I_acc_F`/`I_max_F`/`D_F`/`T` need the Brisbane/Lockyer record |
| API | Kohler & Linsley (1951) | specification-level; numeric content unverified; no numeric oracle found | none; the analytic `P/(1-k)` limit is regression coverage and the MIT `ahrapi` implementation is an independent implementation, neither an oracle |

For the PE kernel, Byun & Lee (2002) is the openly accessible confirmation of the exact
algebra and should be cited alongside the 1999 paper. For the EDI, the 1999 Table 5 is
the only located external numeric anchor, and it only tests the variable-`DS` form.
For I_F, Moishin et al. (2021, CC BY 4.0) is the only openly licensed same-lineage
statement of the equations, but it encodes the double-sum kernel that the 2015 abstract
does not describe. For API, the survey found **no accessible source with tabulated
values**, so the API row currently cannot reach external-validation status the way SPI,
SPEI, EDDI, and the Palmer family have.

## External validation vs regression coverage

Using `VALIDATION.md` vocabulary explicitly:

- **External scientific validation** could come from Byun & Wilhite (1999) Table 5 for
  PE/EDI (if the variable-duration convention is adopted and the Hickman inputs are
  reconstructed), from Deo et al. (2015) for I_F (if the full text is retrieved and its
  convention is matched), and from Kohler & Linsley (1951) for API (only if that report
  turns out to tabulate values reproducible against a public precipitation record).
- **Digitized figures** would be required for the EDI daily-index series (Byun & Wilhite
  Fig. 2b/2c), for the weight curve (Fig. 1), and for any Deo et al. (2015) series whose
  table cannot be obtained. The HDW / Srock et al. precedent shows the cost and the
  disclosure pattern (marker pixel coordinates, a stated read-off uncertainty, and a
  shape/timing assertion rather than a tight magnitude `atol`); it does not supply a
  licensing answer for AMS or Springer figures.
- **Independent implementations** available for cross-checking, not for oracle duty:
  `tidyindex` (MIT; harmonic kernel), `rrk4910/EDI` (GPL-3; weak), `robbub98/ahrapi`
  (MIT; fixed-`k` API), `alexgruber/pyapi` (unlicensed), and the same-lineage
  `memoishin/flood_monitoring_dss` (unlicensed).
- **Regression coverage** is what remains if the above are not adopted: frozen vectors
  generated from this library's own implementation, the analytic constant-input limit
  `API -> P/(1-k)`, the random-data naive-double-sum equivalence test from #1105, and
  the state round-trip test from #1109. These are valuable contract tests but must not be
  described as external validation in `VALIDATION.md`.

## Remaining gaps and required maintainer sign-off

1. **I_F PE-kernel conflict (highest priority) - decision should be routed to #1099,**
   which is not yet on that ticket's checklist. The Deo et al. (2015) and (2019)
   abstracts describe an exponentially decaying PE; #1105/#1107 plan the harmonic
   double-sum. Either the planned kernel changes, or the divergence is documented and
   signed off. Until the 2015 full text or Byun & Jung (1998) is retrieved, even the
   exponential parameterization (decay constant, summation length, hydrological-period
   definition) is unknown. #1107 consumes this decision; it cannot make it.
2. **Retrieve Deo et al. (2015) full text** through an institutional subscription/ILL
   and record whether I_F values are tabulated or only plotted; then decide between
   transcription and digitization. If figures are used, confirm reuse terms.
3. **Retrieve the 2014 USQ conference PDF** when `eprints.usq.edu.au` returns (or via
   ILL). This is the best chance of a same-group worked example with tables outside the
   Springer paywall.
4. **Byun & Wilhite digitization/permission.** The 1999 AMS article is free to read but
   not CC-licensed. Decide whether transcribing Table 5's twelve numbers and/or
   digitizing Figure 2 for a committed fixture is acceptable, and whether permission is
   needed. Record the decision in the fixture `provenance.json`.
5. **Hickman inputs and baseline.** Table 5 reproduction needs more than the 1995-1996
   daily precipitation for Hickman, Nebraska and the paper's substitution rules: MEP and
   its standard deviation are per-calendar-day statistics of a multi-decade baseline,
   `EP` needs more than `DS` days of prior history, and the 5-day running-mean
   alignment, the dry-day threshold that defines dry duration, and the standardization
   denominator are all unpinned. Confirm the station record source and whether the
   paper's data handling can be matched before promising a tight tolerance; otherwise
   the Table 5 check is qualitative/loose.
6. **Adopt or reject the variable-duration EDI, and fix what `DS` means** (#1099). If
   rejected, Byun & Wilhite Table 5 is not a valid EDI oracle for this library and the
   only remaining EDI evidence is formula-level plus internal regression. If adopted,
   #1099 must also choose between `DS = 365 + dry duration` and the paper's own worked
   example `DS = 365 + 35 - 1`.
7. **Kohler & Linsley (1951) retrieval.** Obtain the 10-page report and record whether
   it contains tabulated index values. If not, the API row must be signed off as
   specification-level, with independent-implementation and regression coverage only, or
   an alternative external product using the same recursion must be found.
8. **Nosrati citation correction.** Issue #1107 cites "Nosrati et al. (2011)"; the
   located work is Nosrati et al. (2010), and its FI uses AWRI, not Deo's PE. Fix the
   reference before it is used to justify a convention.
9. **Operational EDI product.** No APCC/KMA/other operational EDI publication could be
   verified; confirm manually if an operational product is a required evidence tier.
10. **Licensing of same-lineage code.** `memoishin/flood_monitoring_dss` has no license,
    and `rrk4910/EDI` declares GPL-3 only in its DESCRIPTION; neither can be vendored
    into this BSD-licensed repository. Only algorithms may be re-derived from them.

## Fixture and tolerance caveats for #1104

- Follow `tests/fixture/provenance_schema.json` for any committed oracle. Its required
  keys are exactly `source`, `url`, `download_date`, `subset_description`,
  `checksum_sha256`, `fixture_version`, and `validation_tolerance`, and the schema sets
  `additionalProperties: false`, so a misspelled or extra key fails validation.
- Byun & Wilhite Table 5 prints one decimal for APD/PRN (`0.1 mm`) and mixed
  precision for EDI (`-2.5` to one decimal, `-1.22` to two); the day numbers are
  integers. **CNS's unit is unconfirmed and is not grouped with APD/PRN's `0.1 mm`
  here**: independent drought-index literature describes a similarly-named
  accumulated-negative-SEP quantity as a duration (a day count), not millimeters, and
  the transcribed magnitude (-299.8, approaching the length of the whole ~730-day
  Hickman record) is implausible for a precipitation-depth quantity either way.
  Re-confirm CNS's definition and unit against the primary Table 5 image before it
  enters a fixture (folds into gap 4). Any
  comparison tolerance follows each printed value's own confirmed precision (half a unit
  in the last place: `+/-0.05` and `+/-0.005`, not a uniform rule), and the paper's
  5-day-running-mean and variable-`DS` conventions.
- A digitized figure carries a read-off uncertainty that must be stated in the units of
  the plot axis, as the HDW fixture does (`+/-5 hPa m s-1`). Figure 2's index panels are
  scaled (`EDI * 100`), so convert before comparing.
- The Brisbane/Lockyer and Fiji applications cannot be recreated without their input
  rainfall; do not write a test that appears to reproduce them but actually tests only
  this library against itself.
- Because Byun & Wilhite (1999) is the originating group's own arithmetic, passing its
  Table 5 demonstrates reproduction of a published calculation, not independent
  correctness of the method. Say so in `VALIDATION.md`, exactly as the scPDSI entry
  distinguishes independent-implementation cross-validation from external-product
  validation.

## Sources

- Byun, H.-R., and Wilhite, D. A. (1999). Objective Quantification of Drought Severity
  and Duration. *Journal of Climate*, 12(9), 2747-2756.
  https://doi.org/10.1175/1520-0442(1999)012%3C2747:OQODSA%3E2.0.CO;2. Full text and
  table images fetched 2026-09-22 from `journals.ametsoc.org/doi/pdf/...` and
  `journals.ametsoc.org/view/journals/clim/12/9/...`. Metadata record:
  [UNL DigitalCommons](https://digitalcommons.unl.edu/droughtfacpub/32).
- Byun, H.-R., and Lee, D.-K. (2002). Defining Three Rainy Seasons and the Hydrological
  Summer Monsoon in Korea using Available Water Resources Index. *Journal of the
  Meteorological Society of Japan*, 80(1), 33-44.
  https://doi.org/10.2151/jmsj.80.33. PDF:
  https://www.jstage.jst.go.jp/article/jmsj/80/1/80_1_33/_pdf (fetched 2026-09-22).
- Byun, H.-R., and Jung, J.-S. (1998). Quantified diagnosis of flood possibility by
  using effective precipitation index (in Korean with English abstract). *Journal of
  Korea Water Resources Association*, 31(6), 657-665. Cited in Byun & Lee (2002) and
  Moishin et al. (2021); not fetched.
- Deo, R. C., Byun, H.-R., Adamowski, J. F., and Kim, D.-W. (2015). A Real-time Flood
  Monitoring Index Based on Daily Effective Precipitation and its Application to
  Brisbane and Lockyer Valley Flood Events. *Water Resources Management*, 29(11),
  4075-4093. https://doi.org/10.1007/s11269-015-1046-3. Abstract fetched 2026-09-22;
  full text paywalled.
- Deo, R. C., Byun, H.-R., Adamowski, J. F., and Kim, D.-W. (2014). Diagnosis of flood
  events in Brisbane (Australia) using a flood index based on daily effective
  precipitation. *Proceedings of the International Conference on Analysis and Management
  of Changing Risks for Natural Hazards*, European Commission. Published-version PDF
  recorded by OpenAlex at `http://eprints.usq.edu.au/26754/8/Deo_Byun_Adamowski_Kim_AP20_2014_PV.pdf`;
  host unreachable 2026-09-22.
- Deo, R. C., et al. (2019). Quantifying flood events in Bangladesh with a daily-step
  flood monitoring index based on the concept of daily effective precipitation.
  *Theoretical and Applied Climatology*, 137, 1201-1215.
  https://doi.org/10.1007/s00704-018-2657-4. Abstract fetched 2026-09-22.
- Moishin, M., Deo, R. C., Prasad, R., Raj, N., and Abdulla, S. (2021). Designing
  Deep-Based Learning Flood Forecast Model With ConvLSTM Hybrid Algorithm. *IEEE
  Access*, 9. https://doi.org/10.1109/ACCESS.2021.3065939. PDF:
  https://ieeexplore.ieee.org/ielx7/6287639/9312710/09378529.pdf (fetched 2026-09-22;
  CC BY 4.0).
- Moishin, M. `flood_monitoring_dss`. https://github.com/memoishin/flood_monitoring_dss
  (inspected 2026-09-22; no license; no tests).
- Chand, R., Nguyen-Huy, T., Deo, R. C., Ghimire, S., Ali, M., and Ghahramani, A.
  (2024). Copula-Probabilistic Flood Risk Analysis with an Hourly Flood Monitoring
  Index. *Water*, 16(11), 1560. https://doi.org/10.3390/w16111560 (CC BY 4.0).
- Shen, J., Yang, M., Zhang, J., Chen, N., and Li, B. (2025). A New Custom Deep Learning
  Model Coupled with a Flood Index for Multi-Step-Ahead Flood Forecasting. *Hydrology*,
  12(5), 104. https://doi.org/10.3390/hydrology12050104 (CC BY 4.0).
- Nosrati, K., Mohseni Saravi, M., and Shahbazi, A. (2010). Investigation of Flood
  Event Possibility over Iran Using Flood Index. In *Survival and Sustainability*,
  1355-1361. https://doi.org/10.1007/978-3-540-95991-5_127. Abstract fetched
  2026-09-22 (closed access).
- Kumar, R. R., Singh, K. N., Mishra, D. C., and Budhlakoti, N. `EDI` R package v0.1.0
  (GPL-3 per DESCRIPTION). https://github.com/rrk4910/EDI (inspected 2026-09-22; not on
  CRAN).
- Zhang, H. `tidyindex` R package (MIT + file LICENSE).
  https://github.com/huizezhang-sherry/tidyindex (inspected 2026-09-22).
- Kohler, M. A., and Linsley, R. K. (1951). *Predicting the runoff from storm rainfall*.
  U.S. Weather Bureau Research Paper No. 34. OpenLibrary work
  https://openlibrary.org/works/OL7350422W (edition OL17720246M, 10 pages). No online
  copy located; requires manual retrieval.
- Bartram, R. `ahrapi` R package v0.1.0 (MIT + file LICENSE).
  https://github.com/robbub98/ahrapi (inspected 2026-09-22).
- U.S. Army Corps of Engineers. *Antecedent Precipitation Tool*.
  https://github.com/jDeters-USACE/Antecedent-Precipitation-Tool (README; migrated to
  https://github.com/erdc/Antecedent-Precipitation-Tool, releases through v3.0.9). USACE
  public domain with joint-work caveat.
- De Moraes, M. A. E., et al. (2024). Antecedent Precipitation Index to Estimate Soil
  Moisture and Correlate as a Triggering Process in the Occurrence of Landslides.
  *International Journal of Geosciences*, 15, 70-86.
  https://doi.org/10.4236/ijg.2024.151006 (CC BY 4.0).
- `VALIDATION.md` (HDW / Srock et al. digitized-figure entry; KBDI exact-oracle entry;
  external-validation vs regression-coverage language) and
  `tests/fixture/hdw_srock_cedar/provenance.json`.
- `tests/fixture/provenance_schema.json`; `docs/research/dri-wrcc-scpdsi-assessment.md`
  (structure precedent for a repository oracle survey).

Note: issue #1101 and `VALIDATION.md` both cite `docs/research/spi-dataset-survey.md`,
which is not present on `main` - it exists only on the unmerged
`research/spi-dataset-survey` branch. This survey therefore takes
`docs/research/dri-wrcc-scpdsi-assessment.md` as its structural precedent. The dangling
reference in `VALIDATION.md` is pre-existing and outside this ticket's scope.

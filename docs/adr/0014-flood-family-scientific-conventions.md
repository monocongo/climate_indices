# Flood-family scientific conventions, and what stays out of scope

## Status

Accepted. The indices it describes land with FLOOD-08 through FLOOD-13
(#1105–#1110); none exists yet.

The flood oracle survey (#1101, `docs/research/flood-oracle-survey.md`) established
which external oracles for PE, EDI, I_F, and API are actually reproducible, and
which are gated behind paywalled or undigitized material. It left several
conventions open, routing them to the flood design decision
([#1099](https://github.com/monocongo/climate_indices/issues/1099)) rather than
letting an implementation choose silently. This record fixes those conventions
and the family's scope boundary.

The recurring constraint: for this family, the published material that would
settle a question is frequently unobtainable. The Byun & Wilhite (1999) Table 5
values are tabulated, but reproducing them needs the Hickman, Nebraska 1995–1996
daily series (plotted only, never tabulated), the paper's missing-data
substitution rules, and a multi-decade per-calendar-day baseline, and the fetch
that produced the transcription is not reproducible on re-fetch, so the values
would need re-reading before entering a fixture. Deo et al.
(2015) is paywalled, and its abstract alone does not describe the kernel it
implements. Kohler & Linsley (1951) has no digitized copy. Where a convention
cannot be pinned from accessible sources, this record prefers a documented,
citable deviation over an unreproducible claim of reproduction.

## Decision

1. **EDI is the fixed-window form.** The summation duration is `D = 365`, and
   `PRN = DEP / H_D` with `H_D = Σ_{n=1..D} 1/n`, `EDI = PRN / SD(PRN)`. The
   Byun & Wilhite (1999) variable-duration extension — in which the summation
   duration grows with the current dry spell — is **not** implemented. Adopting
   it would convert a fixed linear filter into a recursion whose window depends
   on dry-spell history, and both of its inputs are unresolved: the dry-day
   threshold that defines a dry spell, and the duration definition, which the
   paper contradicts itself on (Table 3 reads `DS = 365 + dry duration`, its own
   worked example says `DS = 365 + 35 − 1`). Its only payoff is Table 5, which is
   unreproducible regardless (see above). Fixing the window at 365 does not
   depart from the originating group: Byun & Lee (2002) restates the same algebra
   with `D = 365` fixed, and notes that the 1999 paper used `D` as a variable.
   Two deviations from the paper are deliberate and belong in the docstrings: no
   5-day running-mean smoothing of the per-calendar-day MEP, and the
   standardization denominator pinned to `SD(PRN)` — the Table 4 reading of EDI
   as the standardized PRN, which Byun & Lee's Eq. (5) coincides with at
   `D = 365` — where the 1999 paper does not pin one quantity.
2. **I_F uses the same PE kernel as EDI**, and that kernel is chosen
   explicitly: the harmonic double sum of Byun & Wilhite Eq. (2), whose weights
   are `w_m = H_D − H_{m−1}`, **selected over Eq. (3)**. The 1999 paper declines
   to rank its three candidate depletion functions, recommending Eq. (2) for
   upper basins, mountainous areas, and sandy areas while recommending Eq. (3)
   for lower basins with good water retention, so the choice is this package's
   decision rather than a reading of the paper. The Deo et al. (2015) and (2019)
   abstracts describe an exponentially decaying effective precipitation
   (Byun & Wilhite's Eq. (1) form) instead, which is not algebraically
   equivalent; **the kernel the 2015 paper implements is unverified** until its
   full text or Byun & Jung (1998) is retrieved. The exponential is **not**
   implemented, because its parameterization — decay constant, summation length,
   definition of "hydrological period" — is not recoverable from the abstracts,
   and it would fork the family's shared kernel. Moishin et al. (2021, CC BY 4.0),
   co-authored with Deo, implements the double sum for the same index and
   attributes that PE form to Byun & Jung (1998), which is the openly licensed
   statement of the algebra to cite until the full text settles the question; no
   kernel-selecting parameter is added before then.
3. **I_F's annual maxima are grouped by a caller-supplied year boundary.**
   `flood_index()` takes a required, keyword-only `year_start_month`; there is no
   default and no water-year detector. The 2015/2019 abstracts say the maxima
   are taken "in the hydrological period", but no start month is recoverable
   from them, and [ADR-0010](./0010-seasonal-carry-is-an-explicit-mask.md)
   already settled the governing principle for this package: the boundary is
   caller policy, and the library does not infer it. Any default would be
   unverifiable against the published convention, since no start month is
   recoverable. Partial
   leading and trailing periods are excluded from the maxima sample rather than
   completed by inference.
4. **The calendar contract is [ADR-0004](./0004-xarray-calendar-semantics.md)
   verbatim.** Flood indices are daily only: 366-day positional layout, every
   year read as a leap year, series beginning in January, with Gregorian
   conversion owned by the xarray seam. No flood-specific calendar convention is
   introduced. PE's window is a rolling sum and therefore calendar-agnostic;
   `year_start_month != 1` groups maxima across the positional year boundary,
   which is the caller's stated convention rather than a conversion the kernel
   performs.
5. **Flood-event helpers are out of scope for the initial implementation.**
   Onset, duration, and severity as runs of `I_F >= 0` — the abstract's flood
   start, with severity the running sum of consecutive positive values and
   duration their count — are the paper's `I_acc_F`/`I_max_F`/`D_F`/`T` event
   statistics, which need the Brisbane and Lockyer Valley rainfall record and
   therefore have no reproducible oracle. (The surveyed same-lineage code uses
   `I_F > 0`.) Unvalidatable code does not belong in a validation-gated
   milestone.
6. **Scope boundary.** Rx1day, Rx5day, and R95pTOT stay in scope: their planned
   oracle is `climdex.pcic` (R), the ETCCDI reference implementation named by
   FLOOD-14 ([#1111](https://github.com/monocongo/climate_indices/issues/1111)),
   pending confirmation of its definitions and percentile conventions against
   Zhang et al. (2011), with `xclim` an optional cross-check that skips when
   absent and never becomes a dependency. WAP and
   SWAP (Lu 2009; Lu et al. 2013) are **deferred**, not rejected: they are a
   separate lineage, no oracle for them was identified, and they would
   introduce a second standardization convention before the first exists. SMRI
   and other snowmelt-dependent indices are **out of scope**, since they need
   snow-pack inputs that daily precipitation and temperature do not provide. A
   sibling repository takes anything requiring a calibrated hydrologic model,
   terrain or land-cover data, or routed discharge.

## Consequences

`VALIDATION.md` must state evidence tiers honestly when the flood section is
written ([#1117](https://github.com/monocongo/climate_indices/issues/1117)):
PE and EDI can be checked against the originating group's own algebra
(Byun & Lee 2002 Eq. (1)/(2)/(5)) and internal regression; API has the analytic
`P/(1 − k)` limit, an independent `ahrapi` comparison, and regression; I_F has
no Deo-algebra check available and stays specification-level until the full text
arrives. `tidyindex` is precedent that harmonic weighting on a rolling window is
an established EDI construction — it does **not** share this kernel's weights
(`H_D − H_{m−1}`, against its reversed `H_n`) and is not an oracle. None of this
is external validation, and only the precipitation-extreme indices can reach
that tier, through `climdex.pcic`.

Deferred work is blocked on retrieval rather than on design: reproducing Table 5
and implementing the variable-duration EDI both wait on the Hickman record and a
settled dry-day threshold; an I_F numeric oracle waits on the Deo et al. (2015)
full text; an API numeric oracle waits on Kohler & Linsley (1951). A reader who
wants the published variable-duration EDI should treat it as an unimplemented
index, not as a bug in `edi()`.

The epic's scope table ([#1098](https://github.com/monocongo/climate_indices/issues/1098))
reflects decision 6, and the family's documented deviation from the published
definition is recorded in the EDI and I_F algorithm reference pages as those
indices land.

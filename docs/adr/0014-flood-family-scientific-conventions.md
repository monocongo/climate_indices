# Flood-family scientific conventions, and what stays out of scope

## Status

Amended: the effective-precipitation kernel landed in #1105; the remaining
indices land with FLOOD-09 through FLOOD-13 (#1106–#1110). The scientific
conventions are unchanged.

Amended again by the flood scope decision
([#1100](https://github.com/monocongo/climate_indices/issues/1100)): decision 6
takes the precipitation-extreme indices out of scope and moves WAP/SWAP from
deferred to not planned. The scientific conventions for the shipped indices are
unchanged.

The flood oracle survey (#1101, `docs/research/flood-oracle-survey.md`) established
which external oracles for PE, EDI, I_F, and API are actually reproducible, and
which are gated behind paywalled or undigitized material. It left several
conventions open, routing them to the flood design decision
([#1099](https://github.com/monocongo/climate_indices/issues/1099)) rather than
letting an implementation choose silently. This record fixes those conventions
and the family's scope boundary.

The recurring constraint: for this family, the published material that would
settle a question is frequently unobtainable. The Byun & Wilhite (1999) Table 5
values are tabulated and have since been read directly from a copy of the
article, but reproducing them still needs the Hickman, Nebraska 1995–1996 daily
series (plotted only, never tabulated), the paper's missing-data substitution
rules, and a multi-decade per-calendar-day baseline; the paper names its data
source as 193 High Plains stations with 37 years (1960–96) of daily
precipitation, reduced to 113 after discarding stations missing more than 1% of
their record. Deo et al. (2015) is paywalled, and its abstract alone does not
describe the kernel it implements. Kohler & Linsley (1951) is partially
retrievable through HathiTrust, which confirms its definitional content but
supplies no tabulated values. Where a convention cannot be pinned from
accessible sources, this record prefers a documented, citable deviation over an
unreproducible claim of reproduction.

## Decision

1. **EDI is the fixed-window form.** The summation duration is `D = 365`, and
   `PRN = DEP / H_D` with `H_D = Σ_{n=1..D} 1/n`. The Byun & Wilhite (1999)
   variable-duration extension — in which the summation duration grows with the
   current dry spell — is **not** implemented. The paper defines both of its
   inputs: dry duration is the period of consecutive negative values of `SEP`
   (Table 3), where `SEP = DEP / ST(EP)` (Eq. 5) over a per-calendar-day
   `ST(EP)`, and the duration is `DS = 365 + dry duration − 1`. That last form is
   the paper's own worked example — 35 days of dry duration on 5 June gives
   `399 = 365 + 35 − 1` — and it disagrees by one with the shorthand in Table 3's
   definition and Table 4's header, where each `j` is "CNS plus i". The paper's
   own real-data maxima follow the shorthand, not the worked example: 245 detected
   dry days give a largest `DS` of 610 and 262 give 627 (`365 + 245 = 610`,
   `365 + 262 = 627`). The worked example remains the operative reading here, and
   the one-day inconsistency is moot for the fixed 365-day window. The reason to
   reject the extension is
   therefore not that its definition is unclear but that it is circular and
   data-hungry: `DS` depends on `SEP`, and `SEP` depends on the multi-decade
   per-calendar-day `MEP` and `ST(EP)` baseline, so the index cannot be computed
   without that baseline. Its only payoff is Table 5, which is unreproducible
   regardless (see above). Fixing the window at 365 does not depart from the
   originating group: Byun & Lee (2002) restates the same algebra with `D = 365`
   fixed, and notes that the 1999 paper used `D` as a variable. The paper also
   defines a 15-day variant (`i` is 365 or 15); only 365 is implemented.
   Two deviations from the paper are deliberate and belong in the docstrings: no
   5-day running-mean smoothing of the per-calendar-day `MEP` and `ST(EP)`, and
   `D` fixed where the paper's `DS` varies. No further deviation is needed in the
   standardization step: with `D` fixed, Eq. (9)'s two forms `PRN / ST(PRN)` and
   `DEP / ST(DEP)` coincide, because the `Σ 1/N` factor is then a constant, and
   `ST(DEP)` equals `ST(EP)`, since `MEP` is a per-calendar-day constant offset.
   The fixed-window EDI is therefore the paper's own `SEP`. The paper does not
   state whether `ST` is a sample or population standard deviation, so that
   choice is recorded in the docstring rather than here.
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
6. **Scope boundary.** Rx1day, Rx5day, and R95pTOT are **out of scope**: the
   `xclim` package already implements the ETCCDI precipitation-extreme
   calculations — `max_1day_precipitation_amount` and
   `max_n_day_precipitation_amount` for Rx1day/Rx5day, and
   `days_over_precip_thresh` for the above-percentile statistic R95pTOT
   measures — and duplicating them here would add a second implementation this
   repository does not own an oracle for, for capability the family's purpose —
   accumulated and effective wetness — does not need. FLOOD-14 and FLOOD-15
   ([#1111](https://github.com/monocongo/climate_indices/issues/1111),
   [#1112](https://github.com/monocongo/climate_indices/issues/1112)) are closed
   as not planned. WAP and SWAP (Lu 2009; Lu et al. 2013) are **not planned**:
   they are a separate lineage, no oracle for them was identified, and they
   would introduce a second standardization convention before the first exists.
   SMRI and other snowmelt-dependent indices are **out of scope**, since they
   need snow-pack inputs that daily precipitation and temperature do not
   provide.

   A separate repository becomes justified when two or more of these are true:

   1. An implementation requires a calibrated rainfall-runoff or routing model
   2. An implementation requires terrain, soil, or land-cover data, or
      remote-sensing flood extent
   3. The flood code exceeds roughly 30 percent of the package's source volume
   4. Flood-specific dependencies would be forced on all `climate_indices` users
   5. The release cadence needs to diverge

   Until then, an out-of-scope flood proposal is not rejected on merit: open an
   issue labeled `flood` recording it as a candidate for the eventual split,
   rather than relitigating the boundary here.
   [CONTRIBUTING.md](https://github.com/monocongo/climate_indices/blob/main/CONTRIBUTING.md#flood-and-wet-extreme-scope)
   records the triage response.

## Consequences

`VALIDATION.md` must state evidence tiers honestly when the flood section is
written ([#1117](https://github.com/monocongo/climate_indices/issues/1117)):
PE and EDI can be checked against the originating group's own algebra
(Byun & Lee 2002 Eq. (1)/(2)/(5)) and internal regression; API has the analytic
`P/(1 − k)` limit, an available-but-unadopted independent `ahrapi`
implementation, and regression, and its
recursion is now cited rather than assumed (Kohler & Linsley 1951 Eq. (3): each
day's index is the previous day's multiplied by `k` with that day's rain added —
the non-lagged form — with typical `k` of 0.85–0.90 and an assumed initial value
converging within several weeks); I_F has
no Deo-algebra check available and stays specification-level until the full text
arrives. `tidyindex` is precedent that harmonic weighting on a rolling window is
an established EDI construction — it does **not** share this kernel's weights
(`H_D − H_{m−1}`, against its reversed `H_n`) and is not an oracle. None of this
is external validation: no reproducible numeric paper oracle has been
identified for PE, EDI, I_F, or API, so the family stops at the source-algebra
and regression tiers.

Deferred work is blocked on retrieval rather than on design: reproducing Table 5
and implementing the variable-duration EDI both wait on the Hickman record, whose
source the paper names, with the dry-day threshold and the `DS` definition now
settled by decision 1; an I_F numeric oracle waits on the Deo et al. (2015) full
text; an API numeric oracle waits on the remaining pages of Kohler & Linsley
(1951), whose retrieved portion tabulates no values. A reader who
wants the published variable-duration EDI should treat it as an unimplemented
index, not as a bug in `edi()`.

The epic's scope table ([#1098](https://github.com/monocongo/climate_indices/issues/1098))
reflects decision 6, and the family's documented deviation from the published
definition is recorded in the EDI and I_F algorithm reference pages as those
indices land.

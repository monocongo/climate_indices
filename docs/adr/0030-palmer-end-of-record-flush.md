# The Palmer end-of-record flush touches only months still pending

## Status

Approved for implementation. The maintainer made both decisions on 2026-11-19, after
[REFIMPL-46 (#1661)](https://github.com/monocongo/climate_indices/issues/1661) compared the
standard Palmer path with NCEI `pdi.f`. Decision 1 is implemented by
[#1665](https://github.com/monocongo/climate_indices/issues/1665). Decision 2 is not yet
implemented and is tracked by [#1666](https://github.com/monocongo/climate_indices/issues/1666).
Promote this record to `Accepted` when #1666 lands.

## Context

When a record ends, a Palmer spell may still be open: some months have a provisional index
and no resolved one. `pdi.f` flushes those months at label 240 (lines 754-765). Both the
Python recursion (`_palmer_pdi._finish_up`) and the Rust kernel (`palmer_pdi::finish_up`)
reproduced that flush statement by statement, and the REFIMPL-46 comparison found two places
where it departs from the literature and from its own evident intent. These are Category C2
in that study's classification: `pdi.f` differs from the literature, so the maintainer
decides.

1. **A stale high-water mark.** `pdi.f` means to reset `k8max` when a spell resolves
   (ASSIGN, lines 850-851). `k8max` is neither an argument of ASSIGN nor in COMMON, so
   line 851 assigns a local variable, and the main program's `k8max` keeps the length of the
   last deferral run. The end-of-record flush then rewrites those slots again, although their
   spell had already resolved. Their PDSI and PHDI return to the provisional value from when
   the month was deferred, and their PMDI becomes the final month's. On division 0101 ending
   December 2002, the PMDI of February to November 2002 (-0.88 to 1.57) became 1.78. Across
   3,680 December record endings (every third division, end years 1991-2022), the stale flush
   changed 1,905 PDSI months (up to 9.55), 137 PHDI months (up to 0.94) and 3,517 PMDI months
   (up to 9.49).
2. **The final month's PMDI.** Months still pending at the end of the record get PDSI = X3
   and PHDI = X3, which is the right provisional value, but their PMDI is the final month's
   (`pdi.f` computes `final_wplm` once). PMDI is defined per month as a probability-weighted
   value of that month's terms (NCEI `divisional-readme.txt` item 4; Heim 2002, p. 1156), and
   `pdi.f` itself computes it per month elsewhere (ASSIGN, lines 846-848; statement 210).

The library had reproduced both on purpose, because agreement with `pdi.f` on full-year
records to 1e-14 was the validation. Reproducing them also meant that appending data to a
record, or truncating it, could change months a spell had long since resolved.

## Decision

1. **The flush rewrites only months still pending.** `_finish_up` and `finish_up` flush the
   `k8` months of the open window, and the high-water mark `k8max` is removed, because nothing
   reads it. A month a spell has resolved keeps its recorded PDSI, PHDI and PMDI whatever
   happens later in the record. This departs from `pdi.f` deliberately, in the direction the
   Fortran's own ASSIGN intended. *Implemented by #1665.*
2. **Each flushed month gets the PMDI of its own state.** The flush computes
   `CASE(ppr, px1, px2, px3)` from that month's own values, not the final month's. *Pending,
   #1666.*
3. **Ending the recursion at the last real month is a separate change**, [#1663](https://github.com/monocongo/climate_indices/issues/1663): a record that ends before December
   stops padding with NaN and flushes at its last real month. It relies on decision 2, because
   the flush then never reads a padded month. It is listed here so the three issues share one
   record, and it changes nothing for December-ending records.

The three changes touch the same lines, so each issue lands in its own pull request and
updates this record's status line.

## Consequences

- **Default output changes for some records.** The months that move are those of the last
  deferral run, which the stale flush rewrote at the end of the record although their spell had
  resolved. That run is usually near the end but need not be: if the spell then runs on without
  deferring, the rewritten months can lie two or more years before the end. A record ending in
  December 2002 and the same record extended to 2022 now agree on every resolved month. A month
  still pending at the end is unchanged by decision 1.
- **Agreement with nClimDiv improves.** Against NCEI nClimDiv over the 344 fixture divisions,
  the maximum PMDI difference falls from 5.5748 to 4.3961, the PDSI mean from 0.0432 to
  0.0428, and the PMDI median from 0.0191 to 0.0190; the PDSI median (0.0127) and the 86.2%
  within 0.05 are unchanged. `tests/test_nclimdiv_reference.py` records the new measurements.
- **`pdsi()` no longer matches `pdi.f` on the months the stale flush rewrote.** Those months
  differ from a compiled `pdi.f` by up to the amounts in the table above. Every other month
  runs the same recursion as before. The provenance record (`palmer-provenance.md`, row 19 and
  D9) states the departure, so nobody restores it as a "fix".
- **The standard-Palmer fixtures were regenerated.** `tests/fixture/palmer/` holds this
  library's own PDSI, PHDI and PMDI output, so it now encodes the departure: only the months that
  moved (PDSI 177 in 51 divisions, PMDI 252 in 80, PHDI 24 in 14) were patched, and the fixture
  provenance records it. On those months the mean absolute difference from nClimDiv fell for PDSI
  (1.29 to 0.045) and PMDI (1.48 to 0.057); PHDI fell on average (0.42 to 0.19) but rose in
  divisions 1108 and 1109.
- **A truncation test replaces the `pdi.f` comparison** for this behavior: truncating division
  0101 at December 2002 must not change any month before it.
- **REFIMPL-29's mechanism F is retired.** The divergence study counted this bug as
  Pdi-vs-Wells class F. With decision 1 the library follows the Wells rule there, so the F
  case no longer diverges, and the pinned study results remain the record of the pinned commit.
- **The Wells recursion is unaffected.** `scpdsi()` never had this flush.

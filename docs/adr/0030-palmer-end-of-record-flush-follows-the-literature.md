# The Palmer end-of-record flush follows the literature, not `pdi.f`

## Status

Approved for implementation. The maintainer made these decisions on 2026-11-19 after
[REFIMPL-46 (#1661)](https://github.com/monocongo/climate_indices/issues/1661) compared the standard Palmer
path with NCEI's public `pdi.f`. One of the three is done; the other two are not yet:

- [#1666](https://github.com/monocongo/climate_indices/issues/1666), the PMDI of the months pending at the end of
  the record: **implemented** with this record.
- [#1665](https://github.com/monocongo/climate_indices/issues/1665), the stale `k8max` that re-flushes months whose
  spell had already resolved: not yet done.
- [#1663](https://github.com/monocongo/climate_indices/issues/1663), a record that ends before December running the
  spell recursion through its NaN padding: not yet done. It is a fix to match `pdi.f`, and it is listed
  here because its flush is the one #1666 and #1665 reshape.

## Context

When the record ends, `pdi.f` flushes the months whose PDSI is still provisional (label 240,
lines 754-765). Three things in that flush depart from the literature and from `pdi.f`'s own
intent:

1. Every flushed month gets the **final month's** PMDI. PMDI is defined per month, as the
   probability-weighted value of that month's own wet and dry terms (NCEI
   `divisional-readme.txt` item 4; Heim 2002, p. 1156), and `pdi.f` computes it that way
   everywhere else (ASSIGN, statement 210). Label 240 also reads month 12 when the record ends
   earlier, a month that was never computed. This is #1666.
2. The flush loops over `k8max`, which `pdi.f` means to reset in ASSIGN but cannot, because
   the assignment on line 851 is to a local variable. Months whose spell had already resolved
   are rewritten with their provisional values. This is #1665.
3. `pdi.f` stops at the last month with data (`nendmo`), where the library runs the recursion
   through the NaN padding of a record that does not end in December. This is #1663.

The library reproduced all three. The first two change default output, so they are recorded
here rather than made quietly.

## Decision

1. **Each month flushed at the end of the record takes the PMDI of its own state.** The flush
   records a month through the same step the recursion uses, so its PMDI is the CASE of that
   month's own `ppr`, `px1`, `px2` and `px3`. (#1666, this change.)
2. **The flush rewrites only months that are still pending.** A month whose spell resolved
   keeps the PDSI, PHDI and PMDI it was given when it resolved. (#1665.)
3. **The recursion ends at the last real month, and flushes there.** The padded months are
   never read. (#1663.)

The library therefore departs from `pdi.f` in these three places, on purpose. X2 not being
updated in the month X1 establishes a spell is a separate `pdi.f` behaviour and stays, because
nClimDiv agrees with it better than with the alternative.

## Consequences

- **Default PMDI changes** for the months pending at the end of every record. Measured on the
  344 fixture divisions (1895-2022), 858 months in 262 divisions change by more than the
  5e-5 comparison tolerance, by up to 5.54. Against NCEI's published nClimDiv PMDI the
  largest absolute difference falls from 5.57 to 1.70 and the median stays at 0.019. PDSI and
  PHDI do not change with #1666.
- A record's PMDI no longer depends on December: the value of a pending month is the same
  whether or not the record ends in it.
- The library no longer matches a `pdi.f` build on these months. The provenance record (D9) names the departure,
  and `pdi.f` remains the comparator for everything else.
- The Python and Rust paths change together and stay bit-identical; the fixtures
  (`tests/fixture/palmer` and the Rust `palmer_reference.txt`) are regenerated from the new code.
- Changing a pending month's PMDI back to the final month's would be a decision to reverse this
  record, not a bug fix.

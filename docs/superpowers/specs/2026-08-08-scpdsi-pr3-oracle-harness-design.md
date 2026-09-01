# Design: scPDSI reference oracle harness (PR3)

**Status**: approved for planning
**Tracking**: [#719](https://github.com/monocongo/climate_indices/issues/719) — part of epic [#716](https://github.com/monocongo/climate_indices/issues/716)
**Parent design**: [`2026-08-07-scpdsi-epic-design.md`](2026-08-07-scpdsi-epic-design.md)

## Problem

PR4 will implement `palmer.scpdsi()` and must be TDD'd against reference values, but no
scPDSI reference data exists in the repository. PR3 produces it: scPDSI, scPHDI, scPMDI and
self-calibrated Z-index for all 344 climate divisions already present in
`tests/fixture/palmer/`.

The reference implementation (`Sibada/scPDSI`) is GPLv3 and this repository is BSD-3-Clause,
so the oracle must be built and run entirely outside the git repository. Only numeric arrays
are committed.

This design covers the oracle harness only. Fixture data is the sole deliverable — no `src/`
changes, and no tests consuming the fixtures (that is PR4's job).

## Inputs

Verified against the working tree, not assumed:

| Property | Value |
|---|---|
| Divisions | 344 (`tests/fixture/palmer/<div>/`, `0101`…) |
| Series per division | `precips.npy`, `pet.npy` — 1536 float64 each |
| Record | 128 years × 12 months, 1895–2022 |
| NaNs in inputs | none, across all 344 divisions |
| AWC | `tests/fixture/palmer_awc.json`, all 344 present, inches, range 4.0–11.0 |
| Calibration | 1931–1990, matching `tests/conftest.py` and `tests/test_palmer.py` |

## Architecture

### License boundary

Two zones, with the repository on the clean side of the line:

```
<scratchpad>/oracle/                 # GPL zone — never committed
    pdsi.cpp  pdsi.h  pdsi_ext.cpp   # reference source
    Rcpp.h  R.h                      # shim headers
    main.cpp                         # CLI: text in, text out
    build.sh
<scratchpad>/drive.py                # BSD zone — npy <-> text, orchestration, comparison
<scratchpad>/work/<div>/{in.txt,std.txt,sc.txt}
```

The repository receives `.npy` arrays and a `provenance.json` edit. No GPL source, headers,
shim code, or build scripts enter it. Nothing in `tests/` references the harness.

`scpdsi.cpp` (the Rcpp entry point) is **not compiled** — `main.cpp` replaces it.

### The shim

`pdsi.h` begins with `#include <Rcpp.h>`, `#include <R.h>`, `using namespace Rcpp;`. Rather
than editing those lines, the harness places its own `Rcpp.h` and `R.h` in the oracle
directory and compiles with `-I.`; angle-bracket includes search `-I` paths, so the shim
shadows the real headers.

The payoff is that `pdsi.cpp` and `pdsi_ext.cpp` stay **byte-identical** to what was
downloaded, which removes any possibility of silently perturbing the algorithm while making
it compile.

Four symbols need replacing, all `std::vector`-backed:

| Symbol | Shim |
|---|---|
| `Rcpp::NumericVector` | `.length()`, `operator[]`, `NumericVector(n)` zero-initialized, copy-assign |
| `Rcpp::NumericMatrix` | `.nrow()`, `operator()(i,j)`, `NumericMatrix(r,c)` zero-initialized |
| `Rf_error(fmt, ...)` | `vsnprintf` → `throw std::runtime_error` |
| `Rf_warning(fmt, ...)` | `vsnprintf` → `fprintf(stderr, ...)` |

`Rcpp::List` is used only by the uncompiled `scpdsi.cpp` and needs no shim.

### Feeding inches exactly

`Rext_init` hardcodes `metric = 1` and `AWC = o_AWC / 25.4`, because its R-facing contract is
millimetre input. The fixture data is already in inches.

Routing inches through the millimetre path — passing `x * 25.4` and letting the oracle divide
it back — is not lossless. Measured over the actual fixture data: **14.0% of the 1,056,768
P/PE values** and **134 of 344 AWC values** are not bit-exact after `(x * 25.4) / 25.4`, at up
to 1.76e-16 relative error (1 ULP).

One ULP is negligible for a smooth computation, but scPDSI is not smooth. It contains order
statistics (`llist::safe_percentile` selects a specific kth element), the 1.25 outlier-ratio
cut in the wet-side duration-factor fit, and tie-breaks such as `if(-x2 > x1 + tolerance)`.
A 1-ULP nudge can flip a discrete branch and produce a visible, unexplainable divergence in
one division. The sc=false validation gate below must mean "the port is correct", not "the
port is correct modulo unit noise".

`metric`, `AWC`, `Ss` and `Su` are all private in `pdsi.h` (declared after the `private:` at
line 188), so neither `main.cpp` nor a subclass can reach them. This requires the **single
edit** to the reference source: adding one public inline accessor, `Rext_use_inches(number
awc_inches)`, and changing no existing line.

The accessor clears the `metric` flag, assigns `AWC` the inch value directly, and then repeats
`Rext_init`'s own surface/underlying soil-layer initialization so the split is identical. The
literal source is not reproduced here — it lives only in the out-of-tree harness — because this
project is BSD-3-Clause and the reference is GPLv3.

`main.cpp` calls `Rext_init(...)` and then `Rext_use_inches(awc)` immediately.

Note that this turns out to be belt-and-braces: `SumAll()` re-derives the soil-layer split from
`AWC` before any water-balance work, and `Rext_PDSI_mon` calls `SumAll()` first, so correctness
depends only on `AWC` holding the exact inch value.

This is safe because `metric` is read in exactly one place on this code path —
`Rext_get_Rvec`'s `if(metric && A[i] != MISSING) A[i] = A[i]/25.4;` in `pdsi_ext.cpp`. The
other five `metric` sites in `pdsi.cpp` (lines 1493, 1561, 1583, 1802, 1878) are all inside
the file-reading routines `GetTemp`/`GetPrecip`/`GetParam`/`CalcThornI`, which
`Rext_PDSI_mon` never enters.

## Output mapping

`Rext_PDSI_mon` fills a 16-column `vals_mat`. The four series come from:

| Output | Source |
|---|---|
| Z-index | `vals_mat(n, 8)` |
| PDSI | `vals_mat(n, 13)` |
| PHDI | `vals_mat(n, 14)` |
| PMDI (WPLM) | `vals_mat(n, 15)` |

Column 8 is rewritten by every `CalcX()` call, and `Calibrate()` ends by calling `CalcX()`.
After the three `Calibrate()` passes, column 8 therefore holds the **final rescaled** Z-index,
which is the self-calibrated Z-index we want. The self-calibrating path has one indexing defect:
`CalcZ` stores the 1-based month from `vals_mat` column 1 in `PeriodList`, and `CalcX` passes it
unchanged to `CalcOneX`. On the standard path, `CalcOrigK` instead converts the month to
0-based before calling `CalcOneX`. Every write to columns 8–12 therefore lands one row high,
and the final period's write targets the row after the matrix's logical end.

The slack row is a storage sentinel, not another input period. The `NumericMatrix` shim keeps
the requested row count separate from its backing capacity: `NumericMatrix(r, c)` allocates
storage for `(r + 1) * c` doubles, while `nrow()` continues to return `r`. `Rext_init` therefore
still creates a logical `nPeriods`-by-16 `vals_mat`, and all calculations and calibration loops
still visit exactly `nPeriods` rows. The final-period writes to columns 8–12 land in allocated
trailing storage without appending synthetic P/PE values or changing calibration.

When writing output, the harness iterates `i` from `0` through `n_values - 1` and emits exactly
`n_values` records. In `sc=true` mode only the Z-index comes from `vals_mat(i + 1, 8)`; standard
Z-index output and columns 13–15 always come from logical row `i`. The sentinel row is never
emitted as an additional record. See the SC-PATH OFF-BY-ONE note in
`tests/fixture/palmer/provenance.json` for the full detail. Columns 13–15 are written by
`Rext_output_X()`, called once at the very end, using its own sequential counter, and are
correct on both paths.

Duration factors come from `Rext_out_params()`: `wetm`, `drym`, `wetb`, `dryb` at indices
0–3. Note the reference orders these `wetm, drym, wetb, dryb`; the committed array is
reordered to `[wetm, wetb, drym, dryb]` to group each side's slope with its intercept.

## Data flow

`drive.py` writes `in.txt`; the binary reads it, runs `Rext_PDSI_mon(sc)`, writes `out.txt`;
`drive.py` parses it back and saves `.npy`.

```
in.txt:   1895 2022 1931 1990 <awc_inches> 1536
          <P[0]> <PE[0]>              # one month per line, %.17g
          ...

out.txt:  DURFACT <wetm> <wetb> <drym> <dryb>
          <Z> <PDSI> <PHDI> <PMDI>    # one month per line, %.17g
          ...
```

`%.17g` round-trips IEEE double exactly, so serialization contributes zero error.

`MISSING` is `-999.00` (`pdsi.h` line 14). The driver maps it to `NaN` and counts every
occurrence. Because the inputs are NaN-free across all 344 divisions, a nonzero count is a
bug signal rather than expected data, and is reported rather than committed.

Invocation is one process per division per mode:

```
./oracle work/<div>/in.txt work/<div>/std.txt --std
./oracle work/<div>/in.txt work/<div>/sc.txt  --sc
```

688 runs total. Per-division processes keep failures localized and make a bad division
reproducible from its `in.txt` alone.

## Validation gate

Before any scPDSI fixture is written, run all 344 divisions in `sc=false` mode and compare
all four standard series against the committed `pdsi.npy`, `phdi.npy`, `pmdi.npy` and
`zindex.npy` at `atol=5e-5, rtol=0` — the tolerance `tests/test_palmer.py` already uses.

**All 344 × 4 comparisons must pass.** If any fail, stop, characterize the failures (which
divisions, which series, max deviation, whether they cluster geographically or by AWC) and
raise it before generating fixtures. A water-balance mismatch would silently corrupt all 344
scPDSI fixtures, and a wrong fixture is worse than a missing one.

**What this gate does and does not prove.** `tests/fixture/palmer/provenance.json` records
the existing arrays as *"climate_indices library reference output"* — they were generated by
this library's own Palmer implementation, not obtained from NOAA. The gate therefore proves
that the oracle's water balance, CAFEC coefficients, K-factors and recursion agree with the
existing port. It is not an independent NOAA validation, and the provenance entry must say so
rather than overclaiming.

The gate is still strong: it exercises every stage the scPDSI path shares with standard PDSI,
leaving only the self-calibration-specific stages (duration-factor fitting, the three
rescaling passes) unvalidated by it. Those are exactly what `scdurfact.npy` gives PR4 a handle
on.

## Deliverables

Per division, in `tests/fixture/palmer/<div>/`:

| File | Contents |
|---|---|
| `scpdsi.npy` | 1536 float64 |
| `scphdi.npy` | 1536 float64 |
| `scpmdi.npy` | 1536 float64 |
| `sczindex.npy` | 1536 float64 |
| `scdurfact.npy` | 4 float64: `[wetm, wetb, drym, dryb]` |

`scdurfact.npy` is beyond the four series named in #719. It costs 344 files of roughly 350
bytes and buys PR4 a stage-level oracle: when `scpdsi()` diverges on a division, PR4 can
assert the duration-factor fit in isolation instead of bisecting an end-to-end mismatch back
through the recursion and three rescaling passes. PR2 established that the wet-side `m` can
legitimately fit negative, which makes this the stage most likely to disagree.

Plus an updated `tests/fixture/palmer/provenance.json`, conforming to
`tests/fixture/provenance_schema.json`. That schema is `additionalProperties: false`, so the
entry stays within its declared keys, using `notes` for anything without a dedicated field. It
must cite Wells, Goddard & Hayes (2004), record the GPLv3 out-of-tree generation
methodology, and set `validation_tolerance` to the `atol=5e-5, rtol=0` the fixtures were
checked at.

## Risks

**`m + b` can be non-positive.** PR2 documented that `extreme_z_sum` can return exactly `0.0`
on the wet side when every rolling sum is filtered out, and that a dry-skewed calibration
window can consequently fit a negative wet `m`. `CalcOneX` divides by `m + b`, so the oracle
can emit `inf` or `nan` rather than failing. The driver checks every output array for
non-finite values and reports the affected divisions instead of committing them.

**Skipped initialization.** `Rext_init` does not call the file-based `initialize()` or
`GetParam()`. The fields those would populate must be unused on the `Rext_PDSI_mon` path. The
sc=false gate is the practical detector — a missed initialization affecting the water balance
would show up as a mismatch there.

**Uncommitted harness.** The GPL constraint means the harness cannot be checked in, so the
fixtures are not regenerable from the repository alone. This is inherent to the licensing
situation and matches how the existing NOAA-derived Palmer fixtures already work. The
provenance entry documents the methodology in enough detail to rebuild the harness from the
upstream source if it is ever needed.

## Out of scope

- Any `src/` change.
- Tests consuming the new fixtures (PR4, [#720](https://github.com/monocongo/climate_indices/issues/720)).
- Weekly-timescale self-calibration, which the reference supports and `climate_indices` does not.
- Revisiting the standard-PDSI fixtures' own provenance (they predate this epic).

## Note on the citation

The epic design doc cites the reference as "Wells, Goddard & Hayes" in one place (line 12) and
"Wells, Goddard & Wilhite" in another (line 59); issue [#719](https://github.com/monocongo/climate_indices/issues/719)
repeats "Wilhite". **Hayes is correct.** The paper is Wells, N., Goddard, S., and Hayes, M. J.,
2004: A Self-Calibrating Palmer Drought Severity Index. *Journal of Climate*, 17(12),
2335–2351. Donald Wilhite is a different drought researcher and not an author. The committed
provenance uses the correct attribution; the epic doc and issue text should be corrected
separately.

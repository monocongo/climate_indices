# Palmer comparator implementations (REFIMPL-30)

Ticket: [REFIMPL-30](https://github.com/monocongo/climate_indices/issues/1632), part of
[the map](https://github.com/monocongo/climate_indices/issues/1602). Out of scope here:
the comparison itself (REFIMPL-31, #1633).

## Status

**Pre-registration.** This section was committed before any comparator was run. The
results sections below are empty until a later commit.

## Question

Which Palmer implementations outside `climate_indices` can run today on common division
inputs, which lineage does each follow, and on what terms?

## Candidates (from the ticket)

NCEI operational `pdi.f`; Wells, Goddard and Hayes C++ (NDMC); `Sibada/scPDSI` (R, GPL-3);
Jacobi et al. (2013) MATLAB tool; Ficklin et al. Penman-Monteith extension
(`dficklin/CAGEO-paper`); `JonKing93/pdsi` (MATLAB, MIT); gridded products (gridMET
pentad PDSI, WestWide Drought Tracker scPDSI, CRU scPDSI, Dai PDSI); `xclim`; other
Python, Julia, or R ports.

Every "known so far" statement in the ticket is a lead to verify, not a finding.

## Hypotheses

- **H1.** `JonKing93/pdsi` runs under GNU Octave, without MATLAB, on division inputs.
- **H2.** The Jacobi tool and the Ficklin extension need MATLAB or heavy porting
  because of their `.fig` launchers; their computational functions may run headless.
- **H3.** Of the runnable implementations, those derived from Jacobi et al. (2013)
  follow the NCEI-like lineage, because that paper checked its tool against the NCDC
  Fortran.
- **H4.** No source for NCEI `pdi.f` or the original NDMC C++ is publicly downloadable
  under a stated licence; both need a request, handled by REFIMPL-06 (#1608).
- **H5.** The gridded products are external products, not runnable implementations.
- **H6.** `xclim` has no Palmer function at the pinned version.
- **H7.** Python and Julia ports that exist derive from `climate_indices` or from a
  comparator above, so none is an independent lineage.

H1 to H7 can each be wrong. A wrong hypothesis is reported as such.

## Lineage rubric (decided from source, before running)

Lineage is read from source, not inferred from outputs. For each implementation record:

1. **Spell logic.** `pdi.f`-style (two tracked indices, `Backtrack` after an established
   spell) or Wells-style (the recursion in `drought/_palmer_wells.py`).
2. **K factor.** Palmer (1965) climate-characteristic form, or Wells duration-factor
   calibration.
3. **PET method.** Thornthwaite, Hamon, Penman-Monteith, or user-supplied.
4. **AWC split.** Surface layer 1 inch and underlying layer remainder, or other.
5. **Calibration handling.** Fixed period or whole record; whether self-calibrating.

Classes: *NCEI-like*, *Wells-like*, *neither*, *undetermined*. A class is asserted only
when items 1 and 2 are both read from source.

## Sample

- **Smoke run:** division 0101, January 1895 to December 2022 (1536 months), inputs
  `tests/fixture/palmer/0101/{precips,pet,temps}.npy` in inches and degrees as stored,
  AWC from `tests/fixture/palmer_awc.json`, calibration 1931-1990.
- **Recipe check:** all 344 divisions, run once to confirm completion.

## Thresholds (fixed before running)

| Claim | Threshold |
| --- | --- |
| Implementation "runs" | At least 99% of the 1536 months finite for division 0101 |
| "Runs under Octave" | Completes with no MATLAB-only call; the exact error is recorded if not |
| Numeric NCEI-like signal | Median absolute difference from `tests/fixture/nclimdiv/zindex.npy` for 0101 of at most 0.05, with the same PET and AWC supplied |
| Numeric Wells-like signal | Standard-mode PDSI agrees with `tests/fixture/palmer/0101/pdsi.npy` while spell months disagree as the 6.3% figure in `tests/fixture/palmer/provenance.json` describes |
| Included in the survey | Runs, licence permits private runs, and the recipe fetches upstream at run time |

The numeric signals corroborate the source reading. They never replace it, and a
mismatch between them is logged on the
[discrepancy ledger](https://github.com/monocongo/climate_indices/issues/1626) rather
than resolved here.

Inputs that a comparator cannot take unchanged (it computes its own PET from
temperature, or expects millimetres) are recorded as an input mismatch. The smoke run
is repeated with the nearest equivalent input where one exists.

## Rules

- Comparator code is never committed. Recipes fetch a pinned upstream at run time.
- Licence is recorded from the upstream repository. A repository with no licence is
  all rights reserved: run privately, never redistribute.
- Outputs stay out of the repository. REFIMPL-31 commits any outputs with provenance
  under `tests/fixture/provenance_schema.json`.
- The include or exclude set is a proposal until the maintainer approves it.

# Palmer comparator implementations (REFIMPL-30)

Ticket: [REFIMPL-30](https://github.com/monocongo/climate_indices/issues/1632), part of
[the map](https://github.com/monocongo/climate_indices/issues/1602). Out of scope here:
the comparison itself (REFIMPL-31, #1633).

## Status

Apart from this paragraph, the sections down to "Rules" are as committed in `2bb81070`, before
any comparator was run. The results below were added afterwards. Where results departed from the
pre-registration, "Deviations" says so.

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

## Results

Observed 2026-11-18 and 2026-11-19. Upstream sources were cloned or downloaded into scratch
directories outside the repository. "Library" means the committed `tests/fixture/palmer`
arrays (the library's own output); "nClimDiv" means `tests/fixture/nclimdiv` (NCEI
operational). Differences are absolute, in index units; "within 0.05" is the share of
division-months.

### Summary

| Implementation | Source and pin | Licence | Runtime | Runs on divisions? | Lineage (from source) | Proposal |
| --- | --- | --- | --- | --- | --- | --- |
| NCEI public `pdi.f` | `pdi.f` 2013-04-03, sha256 `e0c58ccb...`; `soilcnst.asc` 1998-01-12, sha256 `40fc3a55...` | None stated; US Government work, "as is" | Fortran 77; gfortran 16.2 `-std=legacy` | Yes, 344 divisions, 1 s | NCEI, by construction | Include as a check on the library's port; same lineage, not independent |
| `JonKing93/pdsi` | `9a636117577f476a93f206897fb91dc94d4eddc2` | MIT | MATLAB; runs under GNU Octave 9.2.0 | Yes, 344 divisions, 2 min | NCEI-like (K and spell logic read) | **Include**: the NCEI-like comparator for REFIMPL-31; independence from `pdi.f` not established |
| `noaa-nws-cpc/libperl-cpc-palmer` 0.50 | `e694083f101a86e36fa9ba6426dab4d8653e64eb` | CC0 in `LICENSE.md`; `Makefile.PL` says `artistic_2` | Perl 5 | **No**, not as published (three defects below) | NCEI-like K and accountings; PMDI without backtracking | Exclude as published; report defects through REFIMPL-06 |
| `ktripa/scpdsi-tools` | `c73b22c89218236e766fbf2895f2a16ebd8f366e`, 2026-10-21 | MIT | Python, NumPy and pandas | Yes, 344 divisions, 8 s | Wells-like (K and spell logic read) | **Include**: the independent Wells-style comparator, with a large unexplained gap to the Wells oracle |
| `Sibada/scPDSI` | `10f8c3b20465fca0547f2df60e2759d61d3d66b3` | GPL-3.0 | C++ via R | Already used for the scPDSI oracle | Wells | Not independent of the existing fixtures |
| `rpkgs/scPDSI2003` | `8cfa1634fa57bf6241ab77abfd7a9f254567c4d0` | MIT | C++ (CMake) | Not run | Wells: README says only an ET module was changed | Exclude: same C++ lineage as `Sibada/scPDSI` |
| `dficklin/CAGEO-paper` (Ficklin et al.) | `76680f874d7e27e3b866ba46fea0c5587dca9b2c` | None stated (all rights reserved) | MATLAB, `.fig` launcher | Not run | Not determined | Exclude: holds only the amended files, not the Jacobi core |
| Jacobi et al. (2013) tool | Wiley supporting information of doi:10.1002/wrcr.20342; a 2015 news item says to email the author | Unknown | MATLAB | Not obtained | Undetermined (abstract suggests NCEI-like; source not read) | Open: needs the author or the Wiley file |
| Wells, Goddard, Hayes C++ (original) | `src/scpdsi.cpp` v2.0, 5,179 lines, and the manual, from `greenleaf.unl.edu/downloads/scPDSI.zip`; no NDMC download page found | None stated in the zip or the manual | C++ | Obtained outside the repository and read only; not run in this survey | NCEI: its header says it derives from NCDC `pdi.f` and `palmcode.f` (`docs/algorithm_refs/palmer-provenance.md`) | Not independent: it is already the source of the scPDSI oracle fixtures |

Product-only items (`gridMET` pentad PDSI, WestWide Drought Tracker scPDSI, CRU scPDSI, Dai
PDSI) are external products, not runnable implementations; they are not assessed further
here. `research-notes.md` records no Palmer function in the SPI and SPEI modules of `xclim`
1.0.0, which is the version REFIMPL-03 pins; the package as a whole was not searched. GitHub
repository searches ("palmer drought", "pdsi",
"scpdsi", and Julia-language queries) found no Julia Palmer repository. Other ports seen but
not read: `cszang/pdsi` (no licence, wraps the Wells C++), `TreeGuo/Drought_Index` (no
licence, Python), `joergsteinkamp/PDSI_for_LPJ-GUESS` (GPL-3, C header), `alexdum/scpdsi`,
`carlosdobler/global_scpdsi_ww`, `BruceCai98/drought-PDSI-county`.

### NCEI public `pdi.f`

NCEI publishes the Fortran program with a drought.gov page
(`https://drought.gov/data-maps-tools/drought-indices-fortran-spi-pdsi`) that states the code
"is not an exact copy of what is run on NCEI's internal servers, but represents general use
version approved for public release" and is furnished "as is". Files:
`https://www1.ncdc.noaa.gov/pub/data/nidis/software/ncei/palmer/{pdi.f,soilcnst.asc,spi.f}`.
This contradicts H4. The `www.ncei.noaa.gov/pub/data/cirs/software/palmer/pdi_v2.f90` URL that a
web search returned gave 404 and was not used.

Recipe: `containers/palmer-ncei-pdi/run.sh`, then
`containers/palmer-ncei-pdi/smoke_compare.py <repo> <work-dir>/run` for the table below. The
recipe verifies both checksums, edits only the documented parameter statement (`nendyr=2022`,
`nendmo=12`, `iouteyr=2022`) in a scratch copy, and reads the committed temperature and
precipitation inputs, which are nClimDiv divisional values in inches
(`tests/fixture/palmer/provenance.json`) and degrees F (`tests/test_speibase_reference.py`).
`pdi.f` computes its own Thornthwaite PE from the constants in `soilcnst.asc`, so no PET is
supplied.

Input adaptation: rows 1-58 of `soilcnst.asc` belong to another series whose ids collide with
division ids under Fortran blank-null reads (the row `101 0` reads as 1010), so `pdi.f`, which
takes the first matching row, would give division 1010 an AWC of 6.0 where the fixture has
7.0. The recipe uses rows 59 onward (exactly 344 rows) and `prep.py` asserts that every
division's AWC equals `tests/fixture/palmer_awc.json`.

| Index | vs library fixtures: within 0.05 | vs nClimDiv: within 0.05 | vs nClimDiv: months > 1 of 528,384 |
| --- | --- | --- | --- |
| PDSI | 98.59% | 84.81% | 3,863 |
| Z-Index | 100.0% | 87.26% | 2 |
| PHDI | 98.62% | 82.47% | 2,837 |
| PMDI (WPLM) | 98.27% | 76.77% | 522 |

`pdi.f` writes two decimals, so differences of 0.005 are rounding. The 1.4% of PDSI months
more than 0.05 from the library fixtures is unexplained here; REFIMPL-31 owns the cause.

Read this against `docs/algorithm_refs/palmer-provenance.md` D9, which records a different
comparison on the same 344 divisions: the library reproduces a double-precision `pdi.f` build
to within 1.2e-14, and the published single-precision build differs from that build by at most
2.8e-5. This run is not directly comparable, because it patches `pdi.f`'s parameter statement,
restricts `soilcnst.asc` to rows 59 onward, and fixes its own period. Reconciling the 2.8e-5
bound with the 1.4% measured here is REFIMPL-31's.

### `JonKing93/pdsi`

Recipe: `containers/palmer-jonking93/run.sh`, then
`containers/palmer-jonking93/summarize.py <repo> <work-dir>/out` for the table below (Octave
9.2.0, image pinned by digest, amd64 under emulation, no network in the container). H1 holds:
`pdsi()` and the upstream
`soilMoisture` and `zPDSI` run under Octave unmodified.

Lineage, read from source: `zPDSI.m` uses the Palmer 1965 constants (0.897, -2.691, 0.5,
1.5, 0.15) with X1/X2/X3 accountings, `probEnd`, and retroactive assignment of X1/X2 over a
spell (`useX1(t-nump...)`, backtracking); `pdsi.m` computes K' = 1.5 log10(w/D) + 0.5 and
K = 17.67 K' / sum(D K'). The README makes no validation claim against NCEI and does not
state a derivation, so independence from `pdi.f` is not established. One difference to carry
into REFIMPL-31: it starts the soil-moisture model by running ten years on calibration-period
monthly means.

Two runs. "Supplied PE" runs the steps of `pdsi.m` in order with the fixture's `pet.npy`
(the division fixtures store no latitude), on all 344 divisions. "As published" calls
`pdsi()` with its own Thornthwaite PE for division 0101 at the latitude implied by
`soilcnst.asc` (-tan(lat) = -0.6899, about 34.6 degrees). Recorded input mismatch: the
as-published call is made in millimetres and degrees C (`jk_run_all.m:50`, with the conversion
in `prep.py:33`), while the supplied-PE run stays in inches, so the two runs are not
like-for-like in units.

| Check | Result |
| --- | --- |
| Runs (supplied PE, 344 divisions) | 528,384 of 528,384 months finite |
| Runs (as published, 0101) | 1,536 of 1,536 finite; Z differs from the supplied-PE run by a median 0.0118 |
| Pre-registered NCEI-like signal, 0101, Z vs nClimDiv | median 0.0199 (threshold 0.05): met |
| Z vs nClimDiv, 344 divisions | median of per-division medians 0.0278; 68.7% within 0.05; 803 months > 1 |
| PDSI vs nClimDiv, 344 divisions | 0.0327; 63.6% within 0.05; 10,421 months > 1 |
| PMDI (Xm) vs nClimDiv | 0.0463; 53.1% within 0.05; 1,658 months > 1 |
| PDSI vs library fixtures | 0.0286; 69.9% within 0.05 |

For comparison, the library's own PDSI is 98.6% within 0.05 of the nClimDiv fixture for
division 0101, and the earlier validation map records 86.2% across all divisions. The
disagreement with nClimDiv is larger for this comparator than for the library or for
`pdi.f`; the cause is unexplained here.

### `noaa-nws-cpc/libperl-cpc-palmer` 0.50: does not run as published

Source reading shows the Palmer 1965 K formula (`Climatology.pm:191,206`), X1/X2/X3
accountings with the Palmer constants, and a probability-weighted PMDI with no backtracking
(`get_palmer_pmdi`). Three defects were observed in a scratch probe (Perl 5.34) that is not
kept in the repository, so they cannot be re-run from here:

1. Neither module declares `Exporter` or `@EXPORT_OK`, so the documented
   `use CPC::Palmer qw(get_palmer_accountings ...)` leaves the functions undefined. The
   documentation also names `get_palmer`, which does not exist (`update_palmer` does).
2. `get_k_coeff` dies on any non-trivial input: `Climatology.pm:190` computes `($$P + $$L)`
   where `$P` is an array reference, giving `Not a SCALAR reference` (probably `$$P[$i] +
   $$L[$i]` was meant).
3. In `get_palmer_accountings` the early `&_return_palmer_accountings(...)` calls have no
   `return`, so control falls through. With `Z_INDEX=-7`, `X3=3` (a wet spell that should
   end), the expected result is X2=-2.333, PROB_SPELL_END=1; it returns X2=0,
   PROB_SPELL_END=0.

Running it would need a patched copy, which is a different comparator. Excluded as published.

### `ktripa/scpdsi-tools`

Recipe: `containers/palmer-ktripa/run.sh` (clones the pinned commit; needs numpy and pandas,
which the repository environment supplies). It calls the low-level `scpdsi()` with the
committed monthly precipitation and PET, per-division AWC, and the 1931-1990 calibration
window, so inputs match the Wells-oracle fixtures `tests/fixture/palmer/<div>/sc*.npy`.

Lineage, read from source: K has no 17.67 renormalisation (the Wells form), duration factors
are fitted to the calibration period, and the recursion backtracks (`flush`). It is a Python
reimplementation, not a port of the C++. It rescales Z by a damped iteration (at most 8
steps) towards calibration-period 2nd and 98th percentiles of -4 and +4 rather than a fixed
number of passes, which REFIMPL-33 should compare with the library's three passes. It is
one month old, single-author, with a synthetic self-check only.

| Check | Result |
| --- | --- |
| Runs, 344 divisions | 528,384 of 528,384 months finite for all four outputs |
| scPDSI vs Wells oracle | median of per-division medians 0.1331; 0.1% within 5e-5; 24.7% within 0.05; 34,988 months > 1 |
| scZ-Index vs Wells oracle | 0.0549; 0.1% within 5e-5; 48.4% within 0.05; 1,116 months > 1 |
| scPHDI vs Wells oracle | 0.1673; 19.7% within 0.05; 40,380 months > 1 |
| scPMDI vs Wells oracle | 0.1795; 17.0% within 0.05; 15,930 months > 1 |

The library matches the same oracle at `atol=5e-5`, so `scpdsi-tools` does not reproduce the
Wells result on identical inputs. Its scZ-Index is off by a median 0.055, but the compared
value is the final self-calibrated Z, rescaled from percentiles of the recursive PDSI, so
this offset does not place the gap before the recursion or rescaling. Locating it would need
the intermediate departures and the unscaled Z compared separately. The cause is unexplained
here.

### Not run

`rpkgs/scPDSI2003`, `dficklin/CAGEO-paper`, the Jacobi tool, and the NDMC C++: see the summary.

## Deviations from the pre-registration

- **H4 was wrong.** NCEI source is public. The Jacobi zip was not found. The authors' original
  Wells C++ was obtained outside the repository and read only, with no licence stated, as
  `docs/algorithm_refs/palmer-provenance.md` records; the NDMC-hosted copy was not found, so
  that part of H4 is partly wrong.
- **H6 not tested.** The `xclim` statement rests on the module-scoped `research-notes.md`
  entry, not on a search of the whole package.
- **H2 and H3 untested.** The Jacobi core was not obtained, and the Ficklin repository
  holds only amended files.
- **H7 partly wrong.** `JonKing93/pdsi` and `libperl-cpc-palmer` are neither derived from
  `climate_indices` nor from a comparator in the ticket's table.
- **Numeric thresholds.** The "NCEI-like" signal was applied to `JonKing93/pdsi` only. The
  pre-registered "Wells-like" signal was written for standard-mode PDSI and does not apply
  to `scpdsi-tools`, which computes scPDSI; it was compared with the Wells-oracle fixtures
  instead, which was not pre-registered.
- **Order of runs.** `scpdsi-tools` was run after the first commit of results, once the
  maintainer approved running it.
- **Additions not pre-registered:** the `pdi.f` run on all 344 divisions (not only 0101), the
  `soilcnst.asc` row restriction, and latitude taken from `soilcnst.asc`.

## Ledger candidates for REFIMPL-24

| Item | Observation | Proposed class |
| --- | --- | --- |
| `JonKing93/pdsi` vs nClimDiv and library | PDSI 63.6% within 0.05, 10,421 months > 1 | Unexplained |
| `pdi.f` public release vs library fixtures | 1.4% of PDSI months more than 0.05 apart | Unexplained; reconcile with `palmer-provenance.md` D9, which bounds the same single-precision comparison at 2.8e-5, before opening |
| `soilcnst.asc` id collision | Row `101 0` reads as 1010 and is matched first | Defect in comparator (data file); draft upstream note |
| `libperl-cpc-palmer` 0.50 | Three defects above | Defect in comparator; draft upstream report |
| `scpdsi-tools` vs Wells oracle | scPDSI 24.7% within 0.05, 34,988 months > 1; Z-Index median 0.055 off | Unexplained |
| `pdi.f` sync check | `if(iyr.ne.jyr.and.id.ne.jd) STOP` stops only when both differ, probably `.or.`; read from source, not triggered | Defect in comparator (not reproduced) |

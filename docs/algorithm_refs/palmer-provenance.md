# Palmer Provenance Record

This page traces each step of the Palmer computation to the published equation
or reference-code statement the implementation follows, and to the Python and
Rust functions that realize it. It does not change or re-validate the algorithm;
the validation status is in {doc}`palmer` and `VALIDATION.md`.

Line numbers are at commit `46481c48`; function names are the stable handle.
`pdi.f` and `scpdsi.cpp` line numbers refer to the files whose SHA-256 digests
are listed, with URLs, in `docs/research/reference-papers/pdi-f-fidelity.md`.
Source ids are in the manifest
`docs/research/reference-papers/sources/sources.yaml`.

## How to read the table

**Status**

- **V**: the formula was read in a source text opened for this record.
- **R**: taken from reference code that was opened; no primary text for this step was located.
- **V+R**: both.
- **S**: described only in a secondary source or an official description, not a primary paper.
- **U**: no opened source supports the step. Every U row is listed under
  [Unresolved rows](#unresolved-rows).

**Sources opened for this record**

| Short name | What was read | Location |
| --- | --- | --- |
| Palmer 1965 | Palmer, W. C. (1965), *Meteorological Drought*, U.S. Weather Bureau Research Paper No. 45; the NCEI scan, read from page images. Page numbers are the printed ones. | `ncei.noaa.gov/monitoring-content/temp-and-precip/drought/docs/palmer.pdf` |
| Alley 1984 | Alley, W. M. (1984), The Palmer Drought Severity Index: limitations and assumptions, *J. Climate Appl. Meteor.* 23, 1100-1109; scanned copy, read from page images. | local copy |
| Karl 1986 | Karl, T. R. (1986), The sensitivity of the Palmer Drought Severity Index and Palmer's Z-index to their calibration coefficients including potential evapotranspiration, *J. Climate Appl. Meteor.* 25, 77-86; scanned copy, read from page images. | local copy |
| Wells 2004 | Wells, N., Goddard, S., and Hayes, M. J. (2004), A self-calibrating Palmer Drought Severity Index, *J. Climate* 17, 2335-2351; text and page images. | local copy |
| Heim 2002 | Heim, R. R., Jr. (2002), A review of twentieth-century drought indices used in the United States, *Bull. Amer. Meteor. Soc.* 83, 1149-1165. A secondary source, used for the PMDI description only. | local copy |
| Wells C++ (Sibada port) | `src/pdsi.cpp`, `src/pdsi_ext.cpp`, `src/pdsi.h` and `R/scpdsi.R` of `Sibada/scPDSI` 0.1.3 (GPL-3), commit `10f8c3b`. Read only; nothing copied. It is a port of the Wells, Goddard and Hayes C++, not the original. | `github.com/Sibada/scPDSI` |
| NCEI readme | `divisional-readme.txt` of the nClimDiv monthly files: the PDSI, PHDI, PMDI definitions and the 1931-1990 calibration statement. | `ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/divisional-readme.txt` |
| `pdi.f` | NCEI's public Fortran (2013-04-03, 899 lines), read in full and compiled outside the repository for the comparison in D9. drought.gov: "not an exact copy of what is run on NCEI's internal servers, but represents general use version approved for public release", furnished "as is"; no licence is stated. Not vendored. | `ncei.noaa.gov/pub/data/nidis/software/ncei/palmer/pdi.f` |
| Wells C++ (original) | `src/scpdsi.cpp` (v2.0, 5,179 lines) and the manual, from the authors' distribution. Read only. No licence is stated in the zip or the manual. Its header says it derives from NCDC `pdsi.f` and `palmcode.f`. | `greenleaf.unl.edu/downloads/scPDSI.zip` |

**Sources not opened.** Heddinghaus and Sabol (1991), Karl and Knight (1985),
and Jacobi et al. (2013). The NCEI readme cites Heddinghaus and Sabol; that
citation is NCEI's, not a reading of the paper.

## Steps

### 1. Potential values and water balance

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | PE (potential evapotranspiration) | Palmer 1965 p. 7: PE is "Thornthwaite's potential evapotranspiration term"; Alley 1984 p. 1101 and Karl 1986 p. 78 say the same. `pdi.f` computes it internally (294-315): Thornthwaite from the fitted `B`, `H` and `TLA` of `soilcnst.asc`, 0 at or below 32 °F, a separate form at or above 80 °F, and a Julian leap rule (280) that gives February 1900 29 days | Input to the Palmer functions. Thornthwaite: `drought/eto.py:eto_thornthwaite` | `eto.rs:thornthwaite` | The Palmer code accepts any PE series; it does not compute PE. The committed `pet.npy` fixtures equal `pdi.f`'s single-precision PE, Julian February 1900 included (D9). | V+R |
| 2 | Layer capacities | Palmer 1965 p. 8: 1 in. surface layer, remainder underlying (6 = 1 + 5 in Kansas, 10 = 1 + 9 in Iowa). Alley 1984 p. 1101: 25 mm in the surface layer. Wells C++ `CalcActual` (`pdsi.cpp:2959`, `2966`). `pdi.f` 228, 237-240: the input `WCBOT` is the underlying layer; `WCTOP = 1.0`, `WCTOT = WCBOT + WCTOP`; both layers start full | `AWCTOP = 1.0` (`palmer.py:30`); `_get_awc_bot` = `max(awc - 1, 0)` (111) | `awc_bottom` (`palmer.rs:33`) | `awc` is the total, so `awc = WCBOT + 1`. The sources do not address `awc < 1`. The code still starts the surface layer at 1.0. See D1. | V+R |
| 3 | Actual loss, recharge, runoff, ET; layer update | Palmer 1965 p. 7, Eqs. (1) and (2): `Ls = S's or (PE - P)`, whichever is smaller; `Lu = (PE - P - Ls)·S'u/AWC`, `Lu <= S'u`; no runoff until both layers are at capacity. Alley 1984 p. 1101 states the same rules. Wells C++ `CalcActual` (`pdsi.cpp:2945-3016`). `pdi.f` 331-379 | `_calc_recharge` (143), `_calc_water_balances` (348) | `recharge` (61), `water_balance` (215) | Vectorized with `np.where`. UL divides by `awc_bot + AWCTOP = max(awc, 1)` (`palmer.py:202`, `palmer.rs:90`), as `pdi.f` does (`WCTOT`, 372); Palmer 1965 Eq. (2), Alley 1984 Eq. (2) and the Wells C++ `CalcActual` divide by `AWC`. The forms agree for the AWC values in use (`>= 1` in); see D1. | V+R |
| 4 | PR, potential recharge | Palmer 1965 p. 10, Eq. (3): `PR = AWC - S'`. Alley 1984 p. 1101, Eq. (1): `PR = AWC - (Ss + Su)`. Wells C++ `CalcPR` (`pdsi.cpp:2913`). `pdi.f` 290: `PR = WCBOT + WCTOP - SP` | `pr = awc_bot + AWCTOP - sp` (`palmer.py:363`) | `water_balance` (`palmer.rs:276`) | The `pdi.f` form. Equals `AWC - S'` when `awc >= 1`. | V+R |
| 5 | PRO, potential runoff | Palmer 1965 p. 11, Eq. (5): `PRO = AWC - PR = S'`; footnote 1 notes the author would now prefer `PRO = 3P - PR`. Alley 1984 p. 1101, Eq. (3), repeats that remark. Wells C++ `CalcPRO` (`pdsi.cpp:2922`). `pdi.f` 289: `SP = SS + SU` | `sp = ss + su` (`palmer.py:362`) | `water_balance` (`palmer.rs:275`) | None. | V+R |
| 6 | PL, potential loss | Palmer 1965 p. 10, Eq. (4): `PL = PLs + PLu`, `PLs = PE or S's` whichever is smaller, `PLu = (PE - PLs)·S'u/AWC`. Alley 1984 p. 1101, Eq. (2), is the same. Wells C++ `CalcPL` (`pdsi.cpp:2931-2939`) divides by `AWC` and caps at `PRO`. `pdi.f` 319-324 divides by `WCBOT + WCTOP` and caps at `SP` | `_calc_potential_loss` (122): divides by `awc_bot + AWCTOP` and caps at `ss + su` | `potential_loss` (`palmer.rs:40`) | **Follows `pdi.f`; differs from Palmer, Alley and the Wells C++ when `awc < 1`**: the denominator is `max(awc, 1)`, not `awc`. See D1. | V+R |

### 2. CAFEC coefficients, departures, Z-Index

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | α, β, γ, δ | Palmer 1965 p. 11 Eq. (6) `α = ET̄/PE̅`; p. 13 Eqs. (7)-(9) `β = R̄/PR̄`, `γ = RO̅/PRO̅`, `δ = L̄/PL̅`; p. 12 footnote 4: when ET̄ and PE̅ are both zero, α = 1, and α = 0 only when ET̄ = 0 and PE̅ > 0. Alley 1984 p. 1101, Eq. (4); Karl 1986 p. 78, Eqs. (6)-(9). Wells C++ `CalcWBCoef` (`pdsi.cpp:3022-3058`). `pdi.f` 441-481 | `_calc_cafec_coefficients` (401), `_calc_cafec_ratio` (224) | none (Python only) | Sentinels: both sums zero gives 1.0 for α, β, γ and 0.0 for δ; denominator zero with numerator nonzero gives 0.0. Same as `CalcWBCoef` and `pdi.f`. Palmer states the rule for α only. | V+R |
| 8 | CAFEC precipitation P̂ | Palmer 1965 p. 14, Eqs. (10)-(14): `P̂ = α·PE + β·PR + γ·PRO - δ·PL`. Alley 1984 p. 1101, Eq. (5); Karl 1986 p. 78, Eqs. (1)-(5). Wells C++ `Calcd` (`pdsi.cpp:1923`). `pdi.f` 518-519, 568-572 | `_calc_cafec_zindex` (489) | `CafecInputs::cafec` (`palmer_zindex.rs:78`) | The code fixes the summation order so tests can pin it. | V+R |
| 9 | Moisture departure d | Palmer 1965 p. 15, Eq. (15): `d = P - P̂`. Alley 1984 p. 1101, Eq. (5); Karl 1986 p. 78, Eq. (10). Wells C++ `Calcd`. `pdi.f` 520, 573 | `_calc_cafec_zindex` (512) | `raw_zindex` (`palmer_zindex.rs:211`) | None. | V+R |
| 10 | D̄ and K′ | Palmer 1965 p. 24 (Fig. 3) and p. 25 Eq. (26): `K′ = 1.5·log10[((PE+R+RO)/(P+L) + 2.80)/D̄] + 0.50`, D̄ the monthly mean of \|d\|. Wells 2004 p. 2337, Eq. (1), gives the same form. Wells C++ `CalcK` (`pdsi.cpp:2042-2056`). `pdi.f` 488 (the T ratio from the calibration sums, no zero guard), 516-527 | `_calc_zindex_factors` (413), `_calc_k_prime_and_dbar` (425; K′ at 451) | `k_prime_and_dbar` (`palmer_zindex.rs:138`) | The 2.8 is inside the logarithm. **Karl 1986 p. 78, Eq. (13), prints `+ 2.8·D̄⁻¹` outside the logarithm**; the code follows Palmer and Wells. See D8. The T ratio stays in Python. The Sibada R wrapper sets 1.5, 2.8, 0.5 (`R/scpdsi.R:17-19`). A zero D̄ gives K′ = 0.5 in the C++ (`pdsi.cpp:2053`); the Rust returns inf/NaN with no substitution (`palmer_zindex.rs` doc comment). | V+R |
| 11 | K, the 17.67 scaling | Palmer 1965 p. 26, Eq. (27): `K = (17.67 / ΣD̄K′)·K′`; the 17.67 is the mean annual `ΣD̄K′` over the nine areas of Table 9 (p. 25). Wells 2004 p. 2337, Eq. (2), and Alley 1984 p. 1101, Eq. (7); Karl 1986 p. 78, Eq. (12). Wells C++ `CalcOrigK` (`pdsi.cpp:2156`), constant `R/scpdsi.R:21`. `pdi.f` 524-531 | `_calc_kfactors` (454; 17.67 at 474) | none (Python only) | The sum is taken over a contiguous transpose to avoid a 1-ULP strided-reduction difference (`palmer.py:464-472`). | V+R |
| 12 | Z-Index | Palmer 1965 p. 19, Eq. (19): `z = d·k`. Wells 2004 p. 2337, Eq. (3); Alley 1984 p. 1101, Eq. (6); Karl 1986 p. 78, Eq. (11). `pdi.f` 574 | `_calc_raw_zindex` (516) | `raw_zindex` (`palmer_zindex.rs:211`) | None. | V+R |

### 3. Duration factors and the three candidate indices

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 13 | Standard duration factors p = 0.897, q = 1/3 | Palmer 1965 p. 21 Eq. (20) `X = Σz/(0.309t + 2.691)`; p. 22 Eq. (24) `ΔX = z/3 - 0.103·X(i-1)`, Eq. (25) `X = X(i-1) + z/3 - 0.103·X(i-1)`. Wells 2004 p. 2337, Eq. (4): `Xi = 0.897·X(i-1) + (1/3)·Zi`; p. 2342 Eqs. (18)-(20) generalize it | `_palmer_duration.py`: `_PALMER_P = 0.897`, `_PALMER_Q = 1/3` (29-30); `from_defaults` (48) gives m = (1-p)/q, b = p/q | `palmer_pdi.rs:carry` (48) | Palmer's page prints 0.103, 0.309, 2.691 and 1/3; Wells 2004 prints 0.897. Wells Eq. (19) gives `p = 1 - m/(m+b)` and Eq. (20) `q = C/(m+b)`, with `m = -1.236`, `b = -10.764`, `C = -4` for Palmer's values (p. 2342). The Sibada R wrapper sets `PDSI.p = 0.897`, `PDSI.q = 1/3` (`R/scpdsi.R:13-14`). `pdi.f` writes the literals `.897` and `/3.` (650, 658, 673, 729); the code's `b/(m+b)` differs from them only in the last bits (D9). | V+R |
| 14 | X1 (wet candidate), X2 (dry candidate), X3 (established spell) | Palmer 1965 p. 32: X3 is "reserved for... any wet spell or any drought which has become definitely established; i.e., X1 >= 1.00 or X2 <= -1.00". Alley 1984 p. 1103 gives the same ±1.00. **Wells 2004 p. 2337: "A spell becomes established when it reaches the threshold of ±0.5"**, because values between -0.5 and 0.5 are "normal". Wells C++ `ChooseX` (`pdsi.cpp:3581-3649`) and `CalcOneX` (`pdsi.cpp:2400`). `pdi.f` 658-705: `PX1 >= 1.` or `PX2 <= -1.` with `PX3 = 0` | pdi lineage: `_statement_200` (`_palmer_pdi.py:360`, promotes at ±1). Wells lineage: `_candidate_values` (`_palmer_wells.py:74`), `_establish_spell` (212, threshold `_SPELL_THRESHOLD = 0.5`) | pdi: `statement_200` (`palmer_pdi.rs:229`). Wells: `candidate_values` (`palmer_wells.rs:89`), `establish_spell` (253) | **The two lineages use different establishment thresholds, and each matches a published source**: ±1.0 in Palmer and Alley, ±0.5 in Wells. See D3. In the month X1 establishes a spell, `pdi.f` leaves `PX2` unassigned (658-670) and the code reproduces it; see D9. | V+R |
| 15 | Wet/dry duration factors | Wells 2004 p. 2342: "two sets of duration factors, one each for dry periods and wet periods, are required for each location"; Eqs. (18)-(20). Wells C++ `CalcOneX` (`pdsi.cpp:2416-2424`): `X3 >= 0` uses the wet factors | `_select_duration_factors` (`_palmer_pdi.py:191`); x3 == 0 uses the wet factors (`_palmer_pdi.py:194-196`) | `palmer_pdi.rs:select` (35) | Palmer 1965 has one pair of factors, and standard PDSI here uses one pair. The `X3 >= 0` tie-break is in the C++ only; Wells 2004 does not state it. | V+R |

### 4. Spell ending, backtracking, selection

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | Z needed to end a spell, Ze | Palmer 1965 p. 29 Eq. (28) `Ze = -2.691·X(i-1) - 1.50` (drought); p. 30 Eq. (31) `Z = -2.691·X(i-1) + 1.50` (wet). Alley 1984 p. 1103, Eqs. (12) and (13). Wells C++ `CalcOneX` (`pdsi.cpp:2441`): `ZE = (m+b)(wd·0.5 - c·X3)`, which equals `0.5·wd·(m+b) - b·X3`. `pdi.f` 620 (wet) and 630 (drought) | pdi: `_statement_180` (`_palmer_pdi.py:442`; Ze at 458) and `_statement_170` (463; Ze at 479). Wells: `_abatement_transition` (`_palmer_wells.py:118`; Ze at 148) | pdi: `statement_180` (`palmer_pdi.rs:309`), `statement_170` (323). Wells: `abatement_transition` (`palmer_wells.rs:141`) | With p = 0.897 and q = 1/3 the pdi form reproduces ±1.50. The statement numbers 170-220 are `pdi.f`'s own labels (170 at 615, 180 at 625, 190 at 638, 200 at 658, 210 at 725, 220 at 745), each verified against the code's function of that name. Wells 2004 says how backtracking works is "beyond the scope of this study" (p. 2337) and gives no Ze equation. | V+R |
| 17 | Effective wetness/dryness U | Palmer 1965 p. 29 Eq. (29) `Uw = Z + 0.15` (ending a drought); p. 30 Eq. (32) `Ud = Z - 0.15` (ending a wet spell). Alley 1984 p. 1103, Eq. (15) and the unnumbered wet-spell form. Wells C++ `CalcOneX` (`pdsi.cpp:2443`): `newV = Z - wd·(m·0.5) + ...`. `pdi.f` 615, 625 | pdi: `Uw = z + 0.15` (`_palmer_pdi.py:447`), `Ud = z - 0.15` (468). Wells: slope × 0.5 (`_palmer_wells.py:185-187`) | pdi: `statement_180`/`170`. Wells: `palmer_wells.rs:235` | The pdi lineage uses the published 0.15; the Wells lineage uses `0.5·m` exactly (0.1545 for the standard m). Wells 2004 does not describe this change. See D5. | V (pdi) + R (Wells) |
| 18 | Probability a spell has ended, Pe | Palmer 1965 p. 29 Eq. (30): `Pe = ΣU / (Ze + ΣU - U) × 100 = 100V/Q`, with `ΣU = 0` if negative; footnote 7. Alley 1984 p. 1103, Eq. (14); Karl 1986 pp. 81-83 describes Pe. Wells C++ `CalcOneX` (`pdsi.cpp:2442-2462`); `Prob >= 100 - tolerance` ends the spell. `pdi.f` 638-651, with exact `PRO.EQ.100.` and `PPR.GE.100.` | pdi: `_statement_190` (`_palmer_pdi.py:420`; Q 427, Pe 429). Wells: `_abatement_transition` (`_palmer_wells.py:149-157`) | pdi: `statement_190` (`palmer_pdi.rs:287`). Wells: `abatement_transition` (`palmer_wells.rs:141`) | Wells tolerance `_TOLERANCE = 1e-5` (`_palmer_wells.py:16`), as the C++ `tolerance`. | V+R |
| 19 | Backtracking, assignment of X | Palmer 1965 p. 33: while X3 is 0, X takes X1 or X2 according to which reached zero first, applied back through the period. Alley 1984 p. 1104: "(i) assign X = X1 until X1 = 0; (ii) then assign X = X2 until X2 = 0; (iii) repeat steps (i) and (ii) until a month was reached which already had an X value assigned". Wells C++ `Backtrack` (`pdsi.cpp:3562-3578`) and `ChooseX` (3626-3640). `pdi.f` ASSIGN (792-855) and label 240 (754-766) | pdi: `_backtrack_assigned_values` (`_palmer_pdi.py:245`), `_flush_spells` (270), `_assign` (299), `_finish_up` (540). Wells: `_backtrack` (`_palmer_wells.py:80`), `_assign_period` (258) | pdi: `backtrack` (`palmer_pdi.rs:156`), `flush` (178), `assign` (192), `finish_up` (380). Wells: `backtrack` (`palmer_wells.rs:96`), `assign_period` (304) | The end-of-record flush rewrites only the months still pending. `pdi.f` never resets the main program's `k8max` (ASSIGN's `k8max = k8` at 851 sets a local), so it re-flushes months a closed spell had resolved; the code stopped reproducing that in #1665 (ADR-0030) and `pdi.f` still does; see D9. Code uses exact-zero comparisons where the C++ uses `tolerance` in `Backtrack` (`pdsi.cpp:3575`); see {doc}`palmer` "Exact-zero comparisons". | V+R |
| 20 | PDSI selection | Palmer 1965 p. 33 (rule above). Alley 1984 p. 1104 and Wells 2004 p. 2337: with no established spell, the PDSI is X1 or X2, "whichever has the largest absolute value"; with an established spell it is X3. NCEI readme: "Only one value is selected as the PDSI drought index for the month." `pdi.f` CASE (875-880) | `_record_index_values` (`_palmer_pdi.py:222`) | `palmer_pdi.rs:record` (147) | None of the opened papers says which of X1 and X2 wins a tie in absolute value; `pdi.f` CASE gives X1 (879). See D2. `_record_index_values` only stores what backtracking assigned (row 19); the largest-\|X\| comparison is in `_case` (`_palmer_pdi.py:211`) and supplies the near-real-time PMDI. | V |
| 21 | PHDI | Karl 1986 p. 83: the PHDI "does not change sign until Pe equals 100%, and it only differs from the actual value of the PDSI when a series of months are encountered when 0 < Pe < 100%". Heim 2002 p. 1156: the X3 term "has come to be known as the Palmer Hydrological Drought Index". NCEI readme item 2. Wells C++ `getPHDI` (`pdsi.cpp:3982-3993`): PHDI = X3 when X3 is nonzero, else X. `pdi.f` 733-734, 761-763, 810-811, 843-845: the same rule | `_record_index_values` (`_palmer_pdi.py:236`); Wells lineage `_palmer_wells.py:348` | `palmer_pdi.rs:record`; `palmer_wells.rs` (403-423) | No source gives an equation labelled PHDI; the rule is `pdi.f`'s and the C++'s, consistent with Karl's and Heim's descriptions. Karl and Knight (1985), the cited PHDI atlas, was not opened. | V+R |
| 22 | PMDI | Heim 2002 p. 1156: the NWS modification "allows computation of the PDSI operationally by taking the sum of the wet and dry terms after they have been weighted by their probabilities (P and 100% - P, where P = the probability that the spell is over)", citing Heddinghaus and Sabol (1991). NCEI readme item 4: "a weighted average of the wet and dry index terms, using the probability as the weighting factor". Wells C++ `getWPLM` (`pdsi.cpp:3945-3980`). `pdi.f` CASE (861-899) | `_case` (`_palmer_pdi.py:203`); `_pmdi` (`_palmer_wells.py:104`) | `palmer_pdi.rs:case` (64); `palmer_wells.rs:pmdi` (123) | **Tie behaviour matches `pdi.f` and `getWPLM`; tolerance behaviour differs from `getWPLM` only**; see D2. Every month flushed at the end of the record gets the final month's PMDI, in `pdi.f` and here; see D9. Alley 1984 p. 1104 describes an earlier operational rule, from a 1983 personal communication with Heddinghaus: X3 when 0 < Pe <= 50, else the X1 or X2 of opposite sign to X3; that rule is not the weighted average and is not implemented. Neither function cites Heddinghaus and Sabol; the paper was not opened. | S (Heim, NCEI) + R |

### 5. Self-calibrating replacements (Wells lineage)

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 23 | K replacement: K′ from the local climate, then Z rescaled toward targets of -4 and +4 at the 2nd and 98th percentiles of the calibration-period PDSI | Wells 2004 p. 2339, Eqs. (8) and (9): `K = K′(-4.00 / 2nd percentile)` if d < 0, `K = K′(4.00 / 98th percentile)` if d >= 0; "the PDSI must first be calculated using Z = dK′"; f_e = 2%. Wells C++ `Calibrate` (`pdsi.cpp:2242-2322`) | `_calc_scpdsi_k_factors` (`palmer.py:477`), `_calculate_scpdsi_prepared` (980; percentile anchors at 995-997, three passes), `_rescale_scpdsi_zindex` (538; ratios at 551-552), `self_calibration.nan_safe_percentile` (108) | `self_calibration.rs:nan_safe_percentile` (79); not used for rescaling | The percentiles are taken of the calibration-period PDSI, not of Z. The code rescales Z by -4/dry and 4/wet, then reruns the recursion, for a fixed three passes; it does not guarantee final percentiles of exactly -4 and +4. The percentile definition (a truncated order statistic) is the C++ `safe_percentile`; Wells 2004 does not specify one. | V+R |
| 24 | Self-calibrating duration factors | Wells 2004 pp. 2342-2345: separate wet and dry fits by least squares on the accumulated Z over several lengths, Eq. (21) `ΣZ = mt + b`; the dry line is shifted by changing only the y intercept (p. 2343); points are dropped from the long end until the correlation magnitude is at least 0.85, with at least four points (pp. 2344-2345); Eqs. (19)-(20) convert m and b to p and q. Wells C++ `CalcDurFact` (`pdsi.cpp:3650-3757`), `get_Z_sum` (3759-3857), `LeastSquares` (3859): windows 3, 6, 9, 12, 18, 24, 30, 36, 42, 48 months; divided by ±4 | `self_calibration.duration_factors` (404), `extreme_z_sum` (187), `_highest_reasonable` (142), `least_squares_fit` (289); `palmer.py:_scpdsi_duration_factors` (933) | `self_calibration.rs`: `duration_factors` (267), `extreme_z_sum` (116), `highest_reasonable` (88), `least_squares_fit` (162) | The 0.85 threshold and four-point minimum are in the paper. The 1.25 "reasonable" filter on the wet side and the list of monthly window lengths (3 to 48 months) are in the C++ only; the paper's example uses ten points (p. 2345) but its text does not list the lengths or the filter. Weekly window sets are not ported. | V+R |
| 25 | The Wells spell recursion | Wells 2004 p. 2342, Eqs. (18)-(20): `Xt = (1 - m/(m+b))·X(t-1) + C·Zt/(m+b)`, using the m and b of the spell being tracked. Wells C++ `CalcOneX`, `ChooseX`, `Backtrack` (rows 14-19) | `_palmer_wells.py:calculate` (286) | `palmer_wells.rs:calculate` (363) | `dryc = 1 - drym/(drym + wetb)` in the C++ (`pdsi.cpp:3596`; the original has it too, `scpdsi.cpp:3913`) pairs the dry slope with the **wet** intercept, which is not what Eq. (19) gives for the dry spell. The Python reproduces the C++ on purpose (the `DurationFactors.from_fitted` comment at `_palmer_duration.py:102-105`). See D6. | V+R |
| 26 | Rescaling passes | Wells 2004 p. 2339: "the 2d and 98th percentile of PDSI values are used to compute K with (9), and then the PDSI is recomputed"; the step list on p. 2345 has the percentile step (5-6) once. Sibada's Rcpp entry point `Rext_PDSI_mon` calls `Calibrate()` three times (`pdsi_ext.cpp:268-269`); the commented-out `SCMonthlyPDSI` in `pdsi.cpp` (1292-1433) calls it once (1407), as does the original `SCMonthlyPDSI` (`scpdsi.cpp:1825`) | `_calculate_scpdsi_prepared` (`palmer.py:993`, `rescale_passes = 3`) | none (loop is Python) | **The paper and the original C++ apply one rescale; the Sibada port applies three.** The repository follows the port, matching its oracle at `atol = 5e-5`. See D7. | V+R |

## Differences found while tracing

Each is recorded. Dispositions belong to the REFIMPL-24 discrepancy ledger.

**D1. AWC semantics and the PL and UL denominators.** `pdi.f` reads the
underlying-layer capacity `WCBOT` (228) and sets `WCTOP = 1.0` and
`WCTOT = WCBOT + WCTOP` (237-240). PL divides by `WCBOT + WCTOP` (322) and UL by
`WCTOT` (372). The repository's `awc` is the total, so `awc = WCBOT + 1`;
`tests/fixture/palmer_awc.json` equals `WCBOT + 1` for all 344 divisions.
`_calc_potential_loss` divides by `awc_bot + AWCTOP = max(awc, 1)`
(`palmer.py:139`, `palmer.rs:42`), as `pdi.f` does; Palmer 1965 Eq. (4),
Alley 1984 Eq. (2) and the Wells C++ `CalcPL` divide by `AWC`. The two agree for
the AWC values in use (`>= 1` in). UL divides by the same combined capacity
(`palmer.py:202`, `palmer.rs:90`), which `pdi.f` calls `WCTOT` (372), so the
`awc == 0` difference is closed (issue #1664). For `awc < 1` the underlying layer
starts empty and the loss stays zero, so no output value changes; the expression
then differs from Palmer Eq. (2), Alley Eq. (2) and the Wells `CalcActual`, as PL
does. A removed 2023 comment tied an `awc + 1` denominator to NCEI's
pre-2013 code (Jacobi et al. 2013, not opened). The likeliest reading is that
users passed the total capacity as `WCBOT`; that is an inference.

**D2. PMDI near-normal and tolerance rules.** With no established spell
(`x3 == 0`), `pdi.f` CASE returns `X1` unless `ABS(X2) .GT. ABS(X1)` (878-879),
and `getWPLM` returns `x1` unless `-x2 > x1 + tolerance`; both give a tie to `x1`.
`_case` and `_pmdi` (and `case` and `pmdi` in Rust) return `x2` only if
`|x2| > |x1|`, so an exact tie goes to `x1`, as in both references
([#1662](https://github.com/monocongo/climate_indices/issues/1662)). Alley 1984
and Wells 2004 say "largest absolute value" without a tie rule.
`pdi.f` compares the probability exactly with 0 and 100 (883), as the Python
does; `getWPLM` treats probabilities within `tolerance/100` of 0 or 1 as the
established case; the library does not apply that tolerance.

**D3. Spell-establishment threshold.** Palmer 1965 (p. 32) and Alley 1984
(p. 1103) establish a spell at X1 >= 1.00 or X2 <= -1.00, which the pdi lineage
follows. Wells 2004 (p. 2337) states ±0.5 as the threshold for the SC-PDSI,
which the Wells lineage follows. Both are published; this is a method choice.

**D4. Citation error.** `self_calibration.py:3` named the third author of Wells
et al. (2004) "Wilhite"; every other mention says Hayes. Fixed in this PR.

**D5. The 0.15 constant.** Palmer 1965 and Alley 1984 print 0.15 in the
effective-moisture definitions. The Wells C++ uses `0.5·m`, which is 0.1545 for
the standard m. Wells 2004 does not describe the change, so it rests on the
reference code alone. The original C++ has the same `0.5·m` (`scpdsi.cpp:2819`),
so it is not an artefact of the port.

**D6. Wet intercept in the dry coefficient.** The Wells C++ computes the dry
carry coefficient with `wetb`, where Wells 2004 Eq. (19) would use the dry
intercept. The original C++ has it too (`scpdsi.cpp:3913`), so it is not an
artefact of the port. The repository keeps it deliberately to match the oracle. It
is a candidate defect in the comparator lineage, for REFIMPL-06.

**D7. Number of rescaling passes.** Wells 2004 describes one percentile rescale
(pp. 2339, 2345), and the original C++ `SCMonthlyPDSI` calls `Calibrate()` once
(`scpdsi.cpp:1825`). The three passes are the Sibada port's own
(`pdsi_ext.cpp:268-269`), so `rescale_passes = 3` (`palmer.py:993`) follows the
port alone. This is REFIMPL-33's question.

**D8. Karl 1986 prints K′ differently.** Karl 1986 p. 78, Eq. (13), shows
`K′ = 1.5·log10[(PE+R+RO)/(P+L)] + 2.8·D̄⁻¹ + 0.5`, with the 2.8 outside the
logarithm. Palmer 1965 Eq. (26) and Wells 2004 Eq. (1) put it inside, divided by
D̄, and the code follows them. The difference is probably typographic; it is
recorded so nobody "corrects" the code to Karl.

**D9. `pdi.f` fidelity.** A line-by-line and numerical comparison with `pdi.f` is
recorded in REFIMPL-46 ([#1661](https://github.com/monocongo/climate_indices/issues/1661)); the report is
`docs/research/reference-papers/pdi-f-fidelity.md`.

- **Agreement.** Measured by REFIMPL-46 before #1665: on the 344 fixture divisions, the
  code reproduced a double-precision `pdi.f` build to within 1.2e-14 in every
  water-balance term, coefficient and index. Since #1665 that holds for every
  month except those the stale `k8max` flush rewrote, which the library no longer
  rewrites. The published single-precision build differs from that
  build by at most 2.8e-5.
- **Faithful to `pdi.f` but unsupported by any source.** The code reproduces two
  `pdi.f` behaviours that no opened source supports:
  - every month flushed at the end of the record gets the final month's PMDI;
  - X2 is not updated in the month X1 establishes a spell (row 14).

  The Wells manual calls a related form of the X2 behaviour "the bug found in the
  FORTRAN program" (its `-b` flag).
- **Differs from `pdi.f`.** The code differs on records that end before December (the
  NaN padding runs through the recursion). Fix:
  [#1663](https://github.com/monocongo/climate_indices/issues/1663).
  An exact PMDI tie now goes to X1, as in `pdi.f`
  ([#1662](https://github.com/monocongo/climate_indices/issues/1662)).
  The UL denominator at `awc == 0` now matches `pdi.f`
  ([#1664](https://github.com/monocongo/climate_indices/issues/1664)).
- **Maintainer decisions (2026-11-19), recorded in {doc}`ADR-0030 <../adr/0030-palmer-end-of-record-flush>`.**
  The stale `k8max` (row 19), which re-flushed months a closed spell had already resolved,
  is **fixed** ([#1665](https://github.com/monocongo/climate_indices/issues/1665)): the library now flushes only the months still pending, so it
  differs from `pdi.f` on the months the stale flush rewrote. Against nClimDiv the PMDI
  maximum difference fell from 5.5748 to 4.3961. The final-month PMDI of the flushed months
  ([#1666](https://github.com/monocongo/climate_indices/issues/1666)) is still to change. The X2 behaviour stays: nClimDiv agrees with it better than with updating X2.

## History

From `git log --follow` on the Palmer module and its successors.

| Date | Commit | Event |
| --- | --- | --- |
| 2017-06-13 | `cccf6ff9` | Initial commit; the original Palmer port. A "new translation of `pdinew.f`" followed in `0b15ec8c` (2017-07-28); `pdinew.py` was removed in `1957371d` (2018-05-15). |
| 2023-06-30 | `2c2d2e62` | Palmer code removed ([522]). |
| 2023-12-12 | `749ddf09` | Palmer re-added with docstrings naming `pdi.f` statements; these docstrings were dropped on 2026-09-29. |
| 2026-08-08 | `a4e3fc62` | scPDSI self-calibration order statistic (`self_calibration.py`). |
| 2026-08-15 | `f1118052` | Self-calibrating Palmer index (`_palmer_wells.py`). |
| 2026-09-16 | `050f3a2f` | One duration-factor contract and one CAFEC/Z-Index step shared (`_palmer_duration.py`). |
| 2026-09-17 | `4722d189` | n-D block path: the spell recursion vectorized for gridded input (#937). Design records: ADR-0009, ADR-0011. |
| 2026-09-29 | `1c3c9d47` | Standard recursion split into `_palmer_pdi.py` with a Z-series interface; `_palmer_common.py` follows in `9bd8b016`. |
| 2026-10-08 | `fc27e35e`, `257e2ea5`, `140558f4` | Rust ports of the water balance, then the Z-Index, pdi, Wells and self-calibration kernels; Python dispatch and PyO3 bindings (RUST-008). |
| 2026-10-19 | `87be2477` | Crate renamed from `climate-core` to `climate-indices`. |
| 2026-11-03 | `a525f367` | Moved into the `drought` package. |

## Unresolved rows

These are the steps with no located primary source, for REFIMPL-06 (#1608).
Two items listed in the first version of this record are now settled. NCEI `pdi.f`
is publicly available, and its statement labels are verified (rows 14-22). The
original Wells C++ has been opened: D5 and D6 are properties of the original code,
and D7's three passes belong to the port alone.

1. **PMDI (row 22).** The probability-weighted rule rests on Heim (2002),
   NCEI's description, `pdi.f` and the Wells C++. Heddinghaus and Sabol (1991),
   the primary source, was not opened, so D2's tie rule follows the reference
   code, not a paper.
2. **PHDI (row 21).** Described in Karl 1986 and Heim 2002 but no formula is
   given; Karl and Knight (1985), the PHDI atlas, was not opened.
3. **Jacobi et al. (2013)** was not opened (D1).

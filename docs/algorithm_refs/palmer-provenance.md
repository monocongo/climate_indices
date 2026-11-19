# Palmer Provenance Record

This page traces each step of the Palmer computation to the published equation
or reference-code statement the implementation follows, and to the Python and
Rust functions that realize it. It does not change or re-validate the algorithm;
the validation status is in {doc}`palmer` and `VALIDATION.md`.

Line numbers are at commit `46481c48`; function names are the stable handle.

## How to read the table

**Status**

- **V**: the formula was read in a source text opened for this record.
- **R**: taken from reference code that was opened; no primary text for this step was located.
- **V+R**: both.
- **U**: no opened source supports the step. Every U row is listed under
  [Unresolved rows](#unresolved-rows).

**Sources opened for this record**

| Short name | What was read | Location |
| --- | --- | --- |
| Palmer 1965 | Palmer, W. C. (1965), *Meteorological Drought*, U.S. Weather Bureau Research Paper No. 45; the NCEI scan, read from page images. Page numbers are the printed ones. | `ncei.noaa.gov/monitoring-content/temp-and-precip/drought/docs/palmer.pdf` |
| Wells C++ (Sibada port) | `src/pdsi.cpp`, `src/pdsi_ext.cpp`, `src/pdsi.h` of `Sibada/scPDSI` 0.1.3 (GPL-3), commit `10f8c3b`. Read only; nothing copied. It is a port of the Wells, Goddard and Hayes C++, not the original. | `github.com/Sibada/scPDSI` |
| NCEI readme | `divisional-readme.txt` of the nClimDiv monthly files: the PDSI, PHDI, PMDI definitions and the 1931-1990 calibration statement. | `ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/divisional-readme.txt` |

**Sources not opened.** Alley (1984), Karl (1986), Wells, Goddard and Hayes
(2004), Heddinghaus and Sabol (1991), Jacobi et al. (2013), and the NCEI
Fortran `pdi.f`. Where code or docs name them, this page says so and does not
quote them. The NCEI readme cites Heddinghaus and Sabol and Karl; that citation
is NCEI's, not a reading of those papers.

## Steps

### 1. Potential values and water balance

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | PE (potential evapotranspiration) | Palmer 1965 p. 7: PE is "Thornthwaite's potential evapotranspiration term" | Input to the Palmer functions. Thornthwaite: `drought/eto.py:eto_thornthwaite` | `eto.rs:thornthwaite` | The Palmer code accepts any PE series; it does not compute PE. | V |
| 2 | Layer capacities | Palmer 1965 p. 8: 1 in. surface layer, remainder underlying (6 = 1 + 5 in Kansas, 10 = 1 + 9 in Iowa). Wells C++ `CalcActual` (`pdsi.cpp:2959`, `2966`) uses `1.0` and `AWC - 1.0` | `AWCTOP = 1.0` (`palmer.py:30`); `_get_awc_bot` = `max(awc - 1, 0)` (111) | `awc_bottom` (`palmer.rs:33`) | The source does not address `awc < 1`. The code still starts the surface layer at 1.0. | V+R |
| 3 | Actual loss, recharge, runoff, ET; layer update | Palmer 1965 p. 7, Eqs. (1) and (2): `Ls = S's or (PE - P)`, whichever is smaller; `Lu = (PE - P - Ls)·S'u/AWC`, `Lu <= S'u`; no runoff until both layers are at capacity. Wells C++ `CalcActual` (`pdsi.cpp:2945-3016`) | `_calc_recharge` (143), `_calc_water_balances` (348) | `recharge` (61), `water_balance` (215) | Vectorized with `np.where`. UL divides by raw `awc` (`palmer.py:202`, `palmer.rs:90`), as Eq. (2) does. | V+R |
| 4 | PR, potential recharge | Palmer 1965 p. 10, Eq. (3): `PR = AWC - S'`. Wells C++ `CalcPR` (`pdsi.cpp:2913`) | `pr = awc_bot + AWCTOP - sp` (`palmer.py:363`) | `water_balance` (`palmer.rs:276`) | Equals `AWC - S'` when `awc >= 1`. | V+R |
| 5 | PRO, potential runoff | Palmer 1965 p. 11, Eq. (5): `PRO = AWC - PR = S'`; footnote 1 notes the author would now prefer `PRO = 3P - PR`. Wells C++ `CalcPRO` (`pdsi.cpp:2922`) | `sp = ss + su` (`palmer.py:362`) | `water_balance` (`palmer.rs:275`) | None. | V+R |
| 6 | PL, potential loss | Palmer 1965 p. 10, Eq. (4): `PL = PLs + PLu`, `PLs = PE or S's` whichever is smaller, `PLu = (PE - PLs)·S'u/AWC`. Wells C++ `CalcPL` (`pdsi.cpp:2931-2939`) divides by `AWC` and caps at `PRO` | `_calc_potential_loss` (122): divides by `awc_bot + AWCTOP` and caps at `ss + su` | `potential_loss` (`palmer.rs:40`) | **Differs from both sources when `awc < 1`**: the denominator is `max(awc, 1)`, not `awc`. See D1. | V+R |

### 2. CAFEC coefficients, departures, Z-Index

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | α, β, γ, δ | Palmer 1965 p. 11 Eq. (6) `α = ET̄/PE̅`; p. 13 Eqs. (7)-(9) `β = R̄/PR̄`, `γ = RO̅/PRO̅`, `δ = L̄/PL̅`; p. 12 footnote 4: when ET̄ and PE̅ are both zero, α = 1, and α = 0 only when ET̄ = 0 and PE̅ > 0. Wells C++ `CalcWBCoef` (`pdsi.cpp:3022-3058`) | `_calc_cafec_coefficients` (401), `_calc_cafec_ratio` (224) | none (Python only) | Sentinels: both sums zero gives 1.0 for α, β, γ and 0.0 for δ; denominator zero with numerator nonzero gives 0.0. Same as `CalcWBCoef`. Palmer states the rule for α only. | V+R |
| 8 | CAFEC precipitation P̂ | Palmer 1965 p. 14, Eqs. (10)-(14): `P̂ = α·PE + β·PR + γ·PRO - δ·PL`. Wells C++ `Calcd` (`pdsi.cpp:1923`) | `_calc_cafec_zindex` (489) | `CafecInputs::cafec` (`palmer_zindex.rs:78`) | The code fixes the summation order so tests can pin it. | V+R |
| 9 | Moisture departure d | Palmer 1965 p. 15, Eq. (15): `d = P - P̂`. Wells C++ `Calcd` | `_calc_cafec_zindex` (512) | `raw_zindex` (`palmer_zindex.rs:211`) | None. | V+R |
| 10 | D̄ and K′ | Palmer 1965 p. 24 (Fig. 3) and p. 25 Eq. (26): `K′ = 1.5·log10[((PE+R+RO)/(P+L) + 2.80)/D̄] + 0.50`, D̄ the monthly mean of \|d\|. Wells C++ `CalcK` (`pdsi.cpp:2042-2056`) | `_calc_zindex_factors` (413), `_calc_k_prime_and_dbar` (425; K′ at 451) | `k_prime_and_dbar` (`palmer_zindex.rs:138`) | The 2.8 is inside the logarithm, as in Eq. (26). The T ratio stays in Python. The Sibada R wrapper sets the constants 1.5, 2.8, 0.5 (`R/scpdsi.R:17-19`). A zero D̄ gives K′ = 0.5 in the C++ (`pdsi.cpp:2053`); the Rust returns inf/NaN with no substitution (`palmer_zindex.rs` doc comment). | V+R |
| 11 | K, the 17.67 scaling | Palmer 1965 p. 26, Eq. (27): `K = (17.67 / ΣD̄K′)·K′`; the 17.67 is the mean annual `ΣD̄K′` over the nine areas of Table 9 (p. 25). Wells C++ `CalcOrigK` (`pdsi.cpp:2156`, `coe_K2/DKSum`), constant from the R wrapper (`R/scpdsi.R:21`, `PDSI.coe.K2 = 17.67`) | `_calc_kfactors` (454; 17.67 at 474) | none (Python only) | The sum is taken over a contiguous transpose to avoid a 1-ULP strided-reduction difference (`palmer.py:464-472`). | V+R |
| 12 | Z-Index | Palmer 1965 p. 19, Eq. (19): `z = d·k` (called `Z = dK` in later work) | `_calc_raw_zindex` (516) | `raw_zindex` (`palmer_zindex.rs:211`) | None. | V |

### 3. Duration factors and the three candidate indices

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 13 | Standard duration factors p = 0.897, q = 1/3 | Palmer 1965 p. 21 Eq. (20) `X = Σz/(0.309t + 2.691)`; p. 22 Eq. (24) `ΔX = z/3 - 0.103·X(i-1)`, Eq. (25) `X = X(i-1) + z/3 - 0.103·X(i-1)` | `_palmer_duration.py`: `_PALMER_P = 0.897`, `_PALMER_Q = 1/3` (29-30); `from_defaults` (48) gives m = (1-p)/q, b = p/q | `pdi.rs:carry` (48) | The page prints 0.103, 0.309, 2.691 and 1/3. The constant 0.897 is `1 - 0.103`, derived, not printed. The Sibada R wrapper sets `PDSI.p = 0.897`, `PDSI.q = 1/3` (`R/scpdsi.R:13-14`). | V+R |
| 14 | X1 (wet candidate), X2 (dry candidate), X3 (established spell) | Palmer 1965 p. 32: X3 is "reserved for... any wet spell or any drought which has become definitely established; i.e., X1 >= 1.00 or X2 <= -1.00"; both X1 and X2 computed every month (p. 32). Wells C++ `ChooseX` (`pdsi.cpp:3581-3649`) and `CalcOneX` (`pdsi.cpp:2400`) | pdi lineage: `_statement_200` (`_palmer_pdi.py:362`, promotes at ±1). Wells lineage: `_candidate_values` (`_palmer_wells.py:74`), `_establish_spell` (212, threshold `_SPELL_THRESHOLD = 0.5`) | pdi: `statement_200` (`palmer_pdi.rs:231`). Wells: `candidate_values` (`palmer_wells.rs:89`), `establish_spell` (253) | **The two lineages use different establishment thresholds** (±1.0 vs ±0.5). Palmer 1965 gives ±1.00; the Wells C++ uses ±0.5 (`pdsi.cpp:3609`, `3620`). See D3. The pdi lineage's statement-label mapping to NCEI `pdi.f` is not verified (row 16). | V (Palmer) + R (Wells) |
| 15 | Wet/dry factor selection | Wells C++ `CalcOneX` (`pdsi.cpp:2416-2424`): `X3 >= 0` uses the wet factors | `_select_duration_factors` (`_palmer_pdi.py:193`); x3 == 0 uses the wet factors (`_palmer_pdi.py:196-198`) | `pdi.rs:select` (35) | Palmer 1965 has one pair of factors; the wet/dry split is a Wells feature. Standard PDSI here uses one pair. | R |

### 4. Spell ending, backtracking, selection

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | Z needed to end a spell, Ze | Palmer 1965 p. 29 Eq. (28) `Ze = -2.691·X(i-1) - 1.50` (drought); p. 30 Eq. (31) `Z = -2.691·X(i-1) + 1.50` (wet). Wells C++ `CalcOneX` (`pdsi.cpp:2441`): `ZE = (m+b)(wd·0.5 - c·X3)`, which equals `0.5·wd·(m+b) - b·X3` | pdi: `_statement_180` (`_palmer_pdi.py:445`; Ze at 461) and `_statement_170` (466; Ze at 482). Wells: `_abatement_transition` (`_palmer_wells.py:118`; Ze at 148) | pdi: `statement_180` (`palmer_pdi.rs:312`), `statement_170` (326). Wells: `abatement_transition` (`palmer_wells.rs:141`) | With p = 0.897 and q = 1/3 the pdi form reproduces ±1.50. The statement numbers 170-220 are the code's labels for NCEI `pdi.f`; `pdi.f` was not opened. | V+R |
| 17 | Effective wetness/dryness U | Palmer 1965 p. 29 Eq. (29) `Uw = Z + 0.15` (ending a drought); p. 30 Eq. (32) `Ud = Z - 0.15` (ending a wet spell). Wells C++ `CalcOneX` (`pdsi.cpp:2443`): `newV = Z - wd·(m·0.5) + ...` | pdi: `Uw = z + 0.15` (`_palmer_pdi.py:450`), `Ud = z - 0.15` (471). Wells: slope × 0.5 (`_palmer_wells.py:185-187`) | pdi: `statement_180`/`170`. Wells: `palmer_wells.rs:235` | The pdi lineage uses Palmer's rounded 0.15; the Wells lineage uses `0.5·m` exactly (0.1545 for the standard m). See D5. | V+R |
| 18 | Probability a spell has ended, Pe | Palmer 1965 p. 29 Eq. (30): `Pe = ΣU / (Ze + ΣU - U) × 100 = 100V/Q`, with `ΣU = 0` if negative; footnote 7. Wells C++ `CalcOneX` (`pdsi.cpp:2442-2462`); `Prob >= 100 - tolerance` ends the spell | pdi: `_statement_190` (`_palmer_pdi.py:423`; Q 430, Pe 432). Wells: `_abatement_transition` (`_palmer_wells.py:149-157`) | pdi: `statement_190` (`palmer_pdi.rs:290`). Wells: `abatement_transition` (`palmer_wells.rs:141`) | Wells tolerance `_TOLERANCE = 1e-5` (`_palmer_wells.py:16`), as the C++ `tolerance`. | V+R |
| 19 | Backtracking, assignment of X | Palmer 1965 p. 33: while X3 is 0, X takes X1 or X2 according to which reached zero first, applied back through the period. Wells C++ `Backtrack` (`pdsi.cpp:3562-3578`) and `ChooseX` (3626-3640) | pdi: `_backtrack_assigned_values` (`_palmer_pdi.py:247`), `_flush_spells` (272), `_assign` (301), `_finish_up` (543). Wells: `_backtrack` (`_palmer_wells.py:80`), `_assign_period` (258) | pdi: `backtrack` (`palmer_pdi.rs:158`), `flush` (180), `assign` (194), `finish_up` (380). Wells: `backtrack` (`palmer_wells.rs:96`), `assign_period` (304) | The end-of-record flush can rewrite earlier months. Code uses exact-zero comparisons where the C++ uses `tolerance` in `Backtrack` (`pdsi.cpp:3575`); see {doc}`palmer` "Exact-zero comparisons". | V+R |
| 20 | PDSI selection | Palmer 1965 p. 33 (rule above). NCEI readme: "Only one value is selected as the PDSI drought index for the month... on the basis of probabilities." | `_record_index_values` (`_palmer_pdi.py:224`) | `palmer_pdi.rs:record` (149) | None known. | V+R |
| 21 | PHDI | NCEI readme (item 2) describes PHDI as an index of "long-term moisture supply"; it gives no formula. Wells C++ `getPHDI` (`pdsi.cpp:3982-3993`): PHDI = X3 when X3 is nonzero, else X | `_record_index_values` (`_palmer_pdi.py:238`); Wells lineage `_palmer_wells.py:348` | `palmer_pdi.rs:record`; `palmer_wells.rs` (403-423) | No primary text located for PHDI (Karl 1986 not opened). | R |
| 22 | PMDI | NCEI readme (item 4): the modification "incorporates a weighted average of the wet and dry index terms, using the probability as the weighting factor"; PMDI and PDSI agree "when the probability is 100%". Wells C++ `getWPLM` (`pdsi.cpp:3945-3980`) | `_case` (`_palmer_pdi.py:205`); `_pmdi` (`_palmer_wells.py:104`) | `pdi.rs:case` (64); `wells.rs:pmdi` (123) | **Tie and tolerance behaviour differs from `getWPLM`**; see D2. Neither function cites Heddinghaus and Sabol; the paper was not opened. | V+R (NCEI text for the rule, C++ for the formula) |

### 5. Self-calibrating replacements (Wells lineage)

Wells et al. (2004) was not opened, so none of rows 23-26 cites it. Their
source is the Wells C++ port.

| # | Step | Source | Python | Rust | Deviation or note | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 23 | K replacement: K′ from the local climate, then Z rescaled so the 2nd and 98th percentiles map to -4 and +4 | Wells C++ `Calibrate` (`pdsi.cpp:2242-2322`): `dry_ratio = -4/P02`, `wet_ratio = 4/P98`; Z multiplied by the wet or dry ratio by sign | `_calc_scpdsi_k_factors` (`palmer.py:477`), `_rescale_scpdsi_zindex` (538; ratios at 551-552), `self_calibration.nan_safe_percentile` (108) | `self_calibration.rs:nan_safe_percentile` (79); not used for rescaling | The percentile is a truncated order statistic (the C++ `safe_percentile`), not `numpy.percentile`. | R |
| 24 | Self-calibrating duration factors | Wells C++ `CalcDurFact` (`pdsi.cpp:3650-3757`), `get_Z_sum` (3759-3857), `LeastSquares` (3859): windows 3, 6, 9, 12, 18, 24, 30, 36, 42, 48 months; regression of window sums against length; divided by ±4 | `self_calibration.duration_factors` (404), `extreme_z_sum` (187), `_highest_reasonable` (142), `least_squares_fit` (289); `palmer.py:_scpdsi_duration_factors` (933) | `self_calibration.rs`: `duration_factors` (267), `extreme_z_sum` (116), `highest_reasonable` (88), `least_squares_fit` (162) | The dry side takes the most extreme sum; the wet side applies a "reasonable" filter (1.25 tolerance on the 98th percentile). Weekly window sets are not ported. | R |
| 25 | The Wells spell recursion | Wells C++ `CalcOneX`, `ChooseX`, `Backtrack` (rows 14-19) | `_palmer_wells.py:calculate` (286) | `palmer_wells.rs:calculate` (363) | `dryc = 1 - drym/(drym + wetb)` uses the wet intercept in the C++ (`pdsi.cpp:3596`); the Python reproduces it on purpose (`_palmer_duration.py`, `_weighting_fraction` comment). See D6. | R |
| 26 | Rescaling passes | Sibada's Rcpp entry point `Rext_PDSI_mon` calls `Calibrate()` three times (`pdsi_ext.cpp:268-269`). The commented-out `SCMonthlyPDSI` in `pdsi.cpp` (1292-1433) calls it once (1407). | `_calculate_scpdsi_prepared` (`palmer.py:993`, `rescale_passes = 3`) | none (loop is Python) | The pass count is a property of the Sibada port. Whether the original Wells code or the 2004 paper fixes three is open. See D7. | R |

## Differences found while tracing

Each is recorded, not changed. Dispositions belong to the REFIMPL-24
discrepancy ledger.

**D1. PL denominator when `awc < 1`.** Palmer 1965 Eq. (4) and the Wells C++
`CalcPL` divide by `AWC`. `_calc_potential_loss` divides by `awc_bot + AWCTOP =
max(awc, 1)` (`palmer.py:139`, `palmer.rs:42`), while UL divides by raw `awc`.
The two agree when `awc >= 1`. A removed 2023 comment in the repository tied an
`awc + 1` denominator to NCEI's pre-2013 code (Jacobi et al. 2013, not opened);
treat that as a lead.

**D2. PMDI near-normal and tolerance rules.** With no established spell
(`x3 == 0`), `getWPLM` returns `x1` unless `-x2 > x1 + tolerance`, so a tie goes
to `x1`. `_case` and `_pmdi` return `x1` only if `|x1| > |x2|`, so an exact tie
goes to `x2`. `getWPLM` also treats probabilities within `tolerance/100` of 0 or
1 as the established case; the Python compares exactly with 0 and 100. Whether
NCEI `pdi.f` has either rule is unknown.

**D3. Spell-establishment threshold.** Palmer 1965 (p. 32) establishes a spell
at X1 >= 1.00 or X2 <= -1.00, which the pdi lineage follows. The Wells C++
uses 0.5.

**D4. Citation error.** `self_calibration.py:3` names the third author of Wells
et al. (2004) as "Wilhite"; every other mention in the repository says Hayes.

**D5. The 0.15 constant.** Palmer 1965 prints 0.15 in Eqs. (29) and (32). The
Wells C++ uses `0.5·m`, which is 0.1545 for the standard m = 0.309.

**D6. Wet intercept in the dry coefficient.** The Wells C++ computes the dry
carry coefficient with `wetb`. The repository keeps this deliberately.

**D7. Number of rescaling passes.** See row 26. This is REFIMPL-33's question.

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

1. **NCEI `pdi.f` was not opened.** Rows 14-22 for the standard lineage have
   Palmer 1965 for the constants and the Wells C++ for the control flow, but the
   mapping of `_statement_170`...`_statement_220` to `pdi.f` statement labels
   is unverified. Its availability is also unknown.
2. **PHDI (row 21).** No formula text was found; Karl (1986) was not opened.
3. **PMDI (row 22).** The rule rests on NCEI's description and the Wells C++;
   Heddinghaus and Sabol (1991) was not opened. D2 depends on it.
4. **Rows 23-26.** Wells et al. (2004) was not opened; the equations for the
   K replacement, the duration-factor regression and the rescaling passes are
   cited from the Sibada port only. Whether the original Wells C++ matches the
   port is unknown.
5. **Alley (1984)** was not opened. It was a candidate source for the
   water-balance and probability steps; those rows rest on Palmer 1965.
6. **Jacobi et al. (2013)** was not opened (D1).
7. **Karl (1986)** was not opened. The NCEI readme cites it for the 1931-1990
   calibration period; the calibration-window handling is in {doc}`palmer`.

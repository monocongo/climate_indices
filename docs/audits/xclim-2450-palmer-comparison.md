# Palmer indices: climate-indices 3.3.0 compared with xclim PR #2450

| | |
| --- | --- |
| Audit date | 2026-11-25 |
| climate-indices | 3.3.0, tag `v3.3.0` (`e068e6cc`). The published package's Palmer modules are byte-identical to the tag. |
| xclim | [PR #2450](https://github.com/Ouranosinc/xclim/pull/2450), head `91fcf88a76829fd9d41ad848bde4c7faa62645de` (13 commits, last 2026-11-18, open; confirmed current on the audit date) |
| xclim file | `src/xclim/compute/_palmer.py`, SHA-256 `9eb0b0dfe0bb22331c489376cbafe40e583c5c16d26af6e830f209198a4425ae` |
| Reproduce | [`xclim_2450_palmer_compare.py`](xclim_2450_palmer_compare.py) |

**Interest to declare.** This audit lives in the climate-indices repository and
uses it as the comparison base. The measurements against NOAA NCEI do not depend
on climate-indices, and the script reruns all of the full-record figures.

File references below are written `X:line` for xclim `_palmer.py` at the head
above, and as paths under `src/climate_indices/drought/` at `v3.3.0` for
climate-indices. `legacy:line` means `src/climate_indices/palmer.py` at
`2c2d2e62^` (the last version before it was removed on 2023-06-30).

## 1. Executive summary

**At this head, the xclim implementation is not equivalent to PDSI and is not a
valid variant of it. It is divergent and incorrect.** Over 528,384
division-months (344 US climate divisions, 1895–2022), its PDSI has a median
absolute difference of 0.348 from NOAA NCEI, RMSE 6.53, r = 0.33, and values
from −393 to +490. climate-indices on the same inputs gives 0.013, 0.202,
r = 0.996. xclim's scPDSI reaches 1.4×10⁷.

The causes are specific and few. Three changes to xclim (restore one
commented-out assignment, correct potential loss, compute one logarithm in
inches) make its Z, PDSI, PHDI and PMDI reproduce climate-indices' NCEI
statistics, with 100% of months within 0.05 of climate-indices. A fourth (the
zero-denominator CAFEC rule) makes its Z-index identical to floating-point
rounding (maximum difference 5×10⁻¹⁴). So the overall design is sound, and the
standard Palmer path is repairable. The scPDSI path is not repaired by these
changes and needs separate work: its backtracking has no effect at all, and its
calibration sequence and constants differ from the Wells procedure as
climate-indices ports it.

The audit also found **a silent wrong-output bug in climate-indices 3.3.0**: one
missing month inside the calibration period makes `pdsi()` return 0.00 for the
whole record (section 6).

## 2. Comparison table

Verdicts: **match**; **variant** (differs, defensible); **bug** (confirmed,
measured or demonstrated); **gap** (missing); **unverified**.

| Component | climate-indices 3.3.0 | xclim `91fcf88a` | Verdict |
| --- | --- | --- | --- |
| Units | inches throughout; `AWCTOP = 1.0` (`palmer.py`) | no unit handling; surface capacity 25.4 (`X:512-516`) assumes mm, K constants assume inches | **bug** (xclim) |
| Two-layer bucket, surface first | `_calc_recharge`, `palmer.py:143-221` | `X:557-673` | match |
| Initial storage | both layers full, `palmer.py:356-357` | both full, `X:713-716` | match |
| Potential recharge PR = AWC − S | `palmer.py:362-363` | `X:526-529` | match |
| Potential runoff PRO = S | `spsum`, `palmer.py:362,409` | `X:530` | match |
| Potential loss PL | `min(PE, S)`-form, no dependence on P, `palmer.py:122-140` | branches on P ≥ PE, `X:534-552` | **bug** (xclim, inherited) |
| Actual ET, recharge, runoff, loss | `palmer.py:176-221` | `X:557-646` | match (Z identical after the fixes) |
| PET | precomputed input | precomputed input | match |
| AWC | total, inches, scalar or per cell | total, implicit mm, `awhc ≤ 0` → NaN (`X:336`) | variant |
| CAFEC α β γ δ, 12 sets from calibration sums or means | `palmer.py:407-410` | `X:832-847` (means; same ratio) | match |
| CAFEC zero denominator | both zero → 1 (δ → 0); only denominator zero → 0, `palmer.py:224-246` | always 1, including δ, `X:844-847` | **divergence** (changes 58 Z months by > 0.05) |
| CAFEC P̂ and departure d | `palmer.py:506-512` | `X:885-887` | match |
| T ratio | `palmer.py:422` | `X:926` | match |
| D̄ = mean \|d\| over calibration | `palmer.py:439-450` | `X:925` | match |
| K′ = 1.5·log₁₀((T+2.8)/D̄)+0.5 | `palmer.py:451`, D̄ in inches | `X:927`, D̄ in mm | **bug** (xclim, new) |
| K = 17.67·K′/Σ(D̄K′) | `palmer.py:473-474` | `X:928-932` | match (units cancel here) |
| Z = K·d | `palmer.py:512` | `X:960` | match |
| p = 0.897, q = 1/3 | `_palmer_duration.py:29-30,55-57` | `X:1012-1013` | match |
| Ud = Z − 0.15, Uw = Z + 0.15 | `_palmer_pdi.py:448,469` | `X:1406,1450` | match |
| Ze = ∓2.691·X3 ± 1.5 | `_palmer_pdi.py:459,480` | `X:1414,1470` | match |
| Establish a spell at \|X\| ≥ 1 | `_palmer_pdi.py:374,385` | `X:1328,1335` | match |
| Accumulated effective wetness/dryness V | carried each month, `_palmer_pdi.py:335,449,470` | **`# V = PV`**, `X:1168`; V is always 0 | **bug** (xclim, new) |
| Backtracking | `pdi.f` statement port, `_palmer_pdi.py:246-298` | legacy translation, `X:1215-1385` | variant; residual below |
| End-of-record flush | pending months only, `_palmer_pdi.py:541-564` (ADR-0030) | `X:1278-1279` | variant; 9 PDSI months > 0.05 in the final 24 |
| PHDI = X3, else X | `_palmer_pdi.py:237` | `X:1210-1212` | match |
| PMDI probability weighting | `_palmer_pdi.py:204-220` | `X:1196-1207` | match |
| scPDSI K′ (no 17.67) | `palmer.py:477-486` | reuses standard K with 17.67, `X:402-421` | **divergence** |
| scPDSI order | K′ → duration factors → recursion → 3 percentile rescales, `palmer.py:995-1014` | standard PDSI → 1 rescale → duration factors → recursion, `X:1620-1636,1816-1909` | **divergence** (ordering inherited from legacy) |
| scPDSI extreme sums | wet side drops anomalies ≥ 1.25 × P98, `self_calibration.py:142-184` | plain max/min, `X:1693-1702` | **bug** (dropped from legacy) |
| scPDSI regression | trims until r ≥ 0.85; intercept on extreme residual, `self_calibration.py:289-401` | `np.polyfit`, `X:1740-1744` | **bug** (dropped from legacy) |
| scPDSI percentiles | truncated order statistic, `self_calibration.py:108-139` | interpolating `np.percentile`, `X:1661-1664` | variant |
| scPDSI abatement threshold | 0.5·m (≈ 0.15 for Palmer's m), `_palmer_wells.py:187` | constant 0.5, `X:2027,2031` | **bug** (xclim, new) |
| scPDSI reset of V when a spell continues | `_palmer_wells.py:139-145` | V stored unreset, `X:2033-2039,2075` | **bug** (reset dropped from legacy) |
| scPDSI Ze | `_palmer_wells.py:147-148` | `X:2020` (algebraically equal) | match |
| scPDSI establish at 0.5 | `_palmer_wells.py:18,217,227` | `X:2147,2161` | match |
| scPDSI X2 coefficient | uses wet intercept, `_palmer_duration.py:104` | uses dry intercept, `X:2134` | divergence; this audit did not open the Wells C++ source |
| scPDSI backtracking | `_palmer_wells.py:80-101` | **no effect**, `X:2230-2243` | **bug** (inherited) |
| Contraction guard \|c\| < 1 | raises `ConvergenceError`, `_palmer_duration.py:106-117` | none | **bug** (xclim) |
| scPHDI, scPMDI | returned | not returned (`phdi`, `pmdi` are standard) | gap |
| Precision | float64 | float64 outputs | match |
| Snow, frozen soil | none | none (author lists it as future work) | match |
| Vectorization | NumPy over cells; optional Rust; scPDSI one cell at a time | water balance is a Python loop of xarray ops with `concat` (`X:731-790`); recursions use `apply_ufunc(vectorize=True)` | variant; Dask behaviour unverified |

## 3. Critical divergences

### 3.1 V is never accumulated (`X:1168`)

```python
        # V = PV
        Pe = PPe[k]
```

`V` is initialised to 0 (`X:1051`) and never changes, so
`PV = Ud + min(V, 0)` and `Q = Ze + V` see only the current month. The
probability that a spell has ended, `Pe = 100·V/Q`, then ignores all previous
abatement. The legacy code has the live line (`legacy:1369`).

Effect, leaving only this unfixed: PDSI median 0.025, RMSE 0.860, 16.0% of
months off by more than 0.5; category agreement 92.8% against 98.2%.

### 3.2 Potential loss (`X:534-552`)

```python
    potential_loss_surface = xr.where(a >= 0, pet, surface_storage)
    potential_loss_under = xr.where(a >= 0, 0.0, ((pet - potential_loss_surface) * under_storage / awhc))
```

(condensed). Potential loss is a *potential* quantity and does not depend on
precipitation: PLs = min(PE, Ss), PLu = (PE − PLs)·Su/AWC. As written, a wet
month claims PLs = PE even when the surface layer holds less, and a dry month
claims PLs = Ss even when PE is smaller, which makes PLu negative and PL larger
than PE (0.8% of division-months). Inherited from `legacy:363-391`.

Effect, leaving only this unfixed: Z RMSE 0.161 against 0.046, maximum 9.19;
PDSI within 0.05 of NCEI in 55.4% of months against 86.2%.

### 3.3 K′ evaluated in millimetres (`X:927`)

```python
    k_hat = 1.5 * np.log10((t_hat + 2.8) / d_hat) + 0.5
```

`2.8` is calibrated for D̄ in inches. The code needs millimetre inputs
(`X:512-516`), so D̄ is 25.4 times too large for this term. K′ goes negative
when D̄ > about 2.15·(T + 2.8) mm, which **reverses the sign of Z**, and
Σ(D̄K′) can approach zero, which makes K unbounded. Measured: 4.1% of
division/calendar-month K values are negative, 64 of 344 divisions are
affected, minimum K = −17.97.

This is the dominant error. Leaving only this unfixed: Z RMSE 18.7, maximum
1955, r = 0.11. The legacy code worked in inches, so the bug is new with the
change to millimetres. No choice of input unit works: inches break the 25.4
surface capacity.

### 3.4 scPDSI backtracking does nothing (`X:2230-2243`)

```python
    while wet_index_deque and dry_index_deque:
        ...
        pdsi_values[month_index] = num1
```

Every iteration writes the same element, the current month, never the earlier
months being resolved. It also writes into `pdsi_values`, which
`_scpdsi_from_zindex` discards (`X:1638-1644`), and the current month's
`scpdsi` is assigned separately (`X:2071`). Demonstrated: replacing the
function body with `return` leaves xclim's scPDSI **bit-identical** on all
528,384 months. The function is textually identical to `legacy:1743-1778`.

### 3.5 scPDSI calibration

Besides the table rows above, two measured outcomes:

- **Target missed.** Self-calibration should put the calibration-period 2nd and
  98th percentiles near −4 and +4. Median over divisions: climate-indices
  −3.96 / +4.01; xclim −5.19 / +5.02. xclim rescales once, from standard-PDSI
  percentiles, before fitting the duration factors, and never checks the result.
- **Divergence.** In division 0205 the fitted wet factors are m = −0.1076,
  b = 9.0012, so c = 1.0121 and the recurrence grows without bound. It is the
  only division with \|c\| ≥ 1 and the only one whose scPDSI exceeds 20; it
  reaches 1.4×10⁷.

After the four standard-path fixes, xclim scPDSI against climate-indices is
still median 0.849, RMSE 1.65, maximum 12.0, r = 0.850, with category agreement
54.8%.

### 3.6 Missing data, calibration and aggregation

Demonstrated on synthetic 40-year series (not part of the committed script):

| Case | xclim behaviour |
| --- | --- |
| One NaN precipitation month | soil storage becomes NaN and every later output is NaN |
| Daily input with a whole month missing | the monthly total is **0.0** (`X:468-471`; `resample().sum()` skips NaN), a false drought |
| Trailing 9-day partial month | summed and treated as a full month |
| Record starting in July, calibration 1991–2000 | `_calibration_slice` (`X:1769-1775`) selects 1991-07 to 2001-06 for scPDSI, while CAFEC and K use calendar years |
| Calibration start before the record | the slice start is negative and selects 0 months in the case tried; no error |
| 36-month record | no error; scPDSI is entirely NaN |
| Identical years (D̄ = 0) | no error; floating-point residue yields Z up to 270 |

Arid, tropical, all-zero-precipitation and AWC ≤ 1 in cases ran in both
libraries. All-zero precipitation gives PDSI above +90 in both; neither guards
this degenerate climate.

### 3.7 What remains after four fixes

Z is identical. PDSI still differs by more than 10⁻⁶ in 264 months and by more
than 0.05 in 45 (maximum 0.52); PHDI in 298 months (maximum 1.26); PMDI in 329
(maximum 0.49). These sit in the spell state machine. Nine of the PDSI months
over 0.05 are in the final 24 months of the record, consistent with the
different end-of-record flush. **The cause of the others was not established.**

## 4. Measurements

Inputs: `tests/fixture/palmer/` precipitation and PET, `palmer_awc.json`, data
from 1895-01, calibration 1931–1990. xclim received the values × 25.4.
Reference: `tests/fixture/nclimdiv/`. "Within 0.05" and "over 0.5" are shares of
months by absolute difference. "Category" is agreement on the seven bins with
edges −4, −3, −2, 2, 3, 4.

| Index vs NCEI | Median | RMSE | Max | r | Within 0.05 | Over 0.5 | Category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PDSI, climate-indices | 0.013 | 0.202 | 7.63 | 0.996 | 86.2% | 1.1% | 98.2% |
| PDSI, xclim | 0.348 | 6.525 | 483.27 | 0.334 | 10.8% | 39.7% | 74.8% |
| PHDI, climate-indices | 0.016 | 0.135 | 3.85 | 0.999 | 83.8% | 0.8% | 98.1% |
| PHDI, xclim | 0.312 | 6.546 | 483.27 | 0.353 | 10.3% | 34.6% | 75.5% |
| PMDI, climate-indices | 0.019 | 0.075 | 1.70 | 0.999 | 78.4% | 0.3% | 98.2% |
| PMDI, xclim | 0.386 | 6.374 | 483.27 | 0.349 | 8.4% | 41.1% | 76.5% |
| Z, climate-indices | 0.013 | 0.046 | 1.02 | 1.000 | 87.5% | 0.0% | 99.0% |
| Z, xclim | 0.252 | 7.045 | 711.17 | 0.278 | 15.4% | 30.1% | 80.2% |

The climate-indices PDSI row reproduces the figures recorded in `VALIDATION.md`
(median 0.0127, 86.2% within 0.05). NCEI publishes two decimals and the fixture
inputs are not byte-identical to NCEI's, so the practical floor is near 0.013.

Cumulative fixes to xclim, PDSI against NCEI:

| Applied | Median | RMSE | Within 0.05 |
| --- | --- | --- | --- |
| none | 0.348 | 6.525 | 10.8% |
| V | 0.256 | 6.409 | 14.0% |
| V + PL | 0.241 | 17.052 | 15.0% |
| V + PL + K′ | 0.013 | 0.202 | 86.2% |
| V + PL + K′ + CAFEC zero rule | 0.013 | 0.202 | 86.2% |

RMSE rises at the second step because the K′ error is still present and the
changed departures move some Σ(D̄K′) values closer to zero. The fixes interact;
the leave-one-out figures in sections 3.1–3.3 show each one's independent
effect. These edits are diagnostic patches, not a proposed implementation.

## 5. Validation evidence

| | climate-indices 3.3.0 | xclim PR #2450 |
| --- | --- | --- |
| External operational reference | NCEI nClimDiv, 344 divisions, 1895–2022, enforced ceilings | none in the PR |
| scPDSI reference | Wells-lineage C++ oracle fixtures, `atol=5e-5`; externally unvalidated against an operational product | none in the PR |
| Published worked example | Palmer 1965 stage checks; Table 12 X within 0.02 in 32 of 40 months, with the other 8 traced to print inconsistencies | none |
| Claimed but not reproducible | — | comparison with AAFC Visual Basic and FORTRAN code: no code, data, metric or tolerance supplied; **unverified** |
| Climate regimes | CONUS divisions; no non-US external reference | synthetic only |

These climate-indices rows restate `VALIDATION.md`; this audit independently
reran only the nClimDiv comparison.

**The xclim tests cannot run at this head.** `tests/test_palmer.py:7` imports
`palmer_drought_indices`, which no longer exists (`__all__` is at `X:31-38`).
The tests also pass `pet=` and `monthly_inputs=`, and read `ds["z"]`; the
function now takes `evspsblpot`, has no `monthly_inputs`, and returns `Z`. The
changelog entry advertises the same nonexistent function. In content, five of the
six tests check only that output exists, keeps its shape, or is somewhere
finite. The sixth pins one PDSI and one Z value generated by the implementation
itself. Both constants match the current `palmer_indices` output, so **they pin
the defective behaviour** and would need replacing after any fix. On the audit
date the visible checks were pre-commit (failing), documentation, labelling and
licence (passing), with no test job shown; the PR reported merge conflicts
(`mergeable_state=dirty`).

The review, inline and discussion comments (read in full on the audit date, bot
messages aside) concern API conventions, units declarations, naming, linting and
style. aulemahal raised the flux-versus-amount unit problem. None raised the
algorithm errors above.

**One open suggestion would add a bug.** coxipi's proposed compaction of the
CAFEC coefficients computes `beta` as `monthly_bar["RO"] / monthly_bar["PR"]`;
the numerator must be recharge, `R`. The code at this head is correct here
(`X:845`). The suggested block also lacks a comma after the `alpha` entry. Do
not apply it as written.

## 6. Bugs in climate-indices

### 6.1 One missing month inside calibration gives PDSI = 0.00 everywhere

`pdsi()` on division 0101 with a single NaN precipitation value:

| NaN position | Finite Z | PDSI result |
| --- | --- | --- |
| 1950-06 (inside calibration) | 0 of 1536 | **all 1536 months exactly 0.00** |
| 1900-06 (before calibration) | 1535 | finite for the first 65 months, then NaN |
| 2000-06 (after calibration) | 1535 | finite for 1265 months, then NaN |

A NaN PET value inside calibration does the same, and so does the pure-Python
kernel (checked separately from the committed script). The NaN enters the
calibration sums, every K and Z becomes NaN, and the `_py_max`/`_py_min`
clamps, which replicate `pdi.f`'s `max(0, x)` behaviour, turn NaN Z into
X = 0. The comment at `palmer.py:1052-1060` describes this clamping, but the
guard it introduces covers only cells that are missing for the *entire* series,
or fully missing cells within a block. The same path returns all-zero PDSI for a
record of identical years (D̄ = 0). `scpdsi()` raises `ConvergenceError` for
both, which is the safe behaviour. Returning "near normal" for missing data is
worse than xclim's NaN. **Severity: high.**

### 6.2 Minor

`tests/test_palmer.py` still calls the library's own regression arrays
`noaa_pdsi` and so on. `tests/fixture/nclimdiv/provenance.json` says they are
not NCEI values, but the names invite the mistake.

## 7. Provenance, citation and licensing

The author [wrote on the PR](https://github.com/Ouranosinc/xclim/pull/2450):
"Palmer source code from the climate_indices package was also employed as
reference." The files show more than reference use. Comparing each function's
syntax tree (docstrings and type annotations removed, then line similarity of
the regenerated source) with the legacy module:

| Function | Similarity to legacy |
| --- | --- |
| `_find_previous_nonzero`, `_assign_X_backtracking`, `_backtrack_self_calibrated` | 1.00 |
| `_choose_X` | 0.93 |
| `_backtrack` | 0.91 |
| `_compute_X` | 0.89 |
| `_compute_scpdsi` | 0.82 |
| `_between_0s` | 0.81 |
| `_duration_factors` | 0.77 |
| `_pdsi_from_zindex`, `_assign_X` | 0.71 |
| `_self_calibrate` | 0.48 |
| `_wet_spell_abatement`, `_dry_spell_abatement` | 0.29 (same names and control flow) |

The source is the **pre-2023 module that climate-indices removed**, not the
validated 3.3.0 code. The newer water balance, CAFEC and K functions have no
same-named legacy counterpart.

`_palmer.py` contains no mention of climate_indices, James Adams or BSD, and the
PR's `AUTHORS.rst` and changelog additions do not credit it. climate-indices is
BSD-3-Clause ("Copyright 2017, James Adams", also in the file's history at that
time); xclim is Apache-2.0. BSD-3 permits this reuse but requires the copyright
notice, conditions and disclaimer to be retained with redistributed source.
**This should be resolved before merge.** It is an engineering observation, not
legal advice; the copyright holder and the xclim maintainers decide the remedy.
The PR also discloses Microsoft Copilot assistance.

Citations: the xclim file cites Palmer (1965) and Wells et al. (2004) by name
only, with no equation or page numbers. It mentions neither Alley (1984),
Heddinghaus and Sabol (1991, the source usually cited for PMDI), NOAA/NCEI nor
`pdi.f`. The legacy comments citing Palmer's equation numbers did not carry
over. The comment at `X:1736-1739` says ordinary least squares is retained "to
remain consistent with the Palmer/Wells duration-factor derivation"; the
climate-indices Wells port, validated against C++ oracle output, does not use
ordinary least squares. climate-indices traces each stage to equations and
reference-code statements in
[`palmer-provenance.md`](../algorithm_refs/palmer-provenance.md).

## 8. Ratings

Scale: poor, fair, good, strong. These are judgements drawn from the sections
above.

| Criterion | climate-indices 3.3.0 | xclim `91fcf88a` |
| --- | --- | --- |
| Faithfulness to Palmer (1965), standard path | strong | poor (three defects, each repairable) |
| Agreement with NOAA operational output | strong | poor |
| Wells (2004) self-calibration | good (C++ oracle parity; no operational reference) | poor |
| Robustness to edge cases | fair (section 6.1) | poor |
| Validation evidence | good to strong for standard; fair for scPDSI | poor |
| Transparency of method | strong | poor to fair (readable and exposes diagnostics; little citation) |

## 9. Recommendation

**Do not merge PR #2450 at `91fcf88a`.** Do not relabel it as "simplified PDSI"
either: sign-reversed K values and a 10⁷ scPDSI are defects, not
simplifications.

**Confidence: high** for the verdict and the bugs in sections 3.1–3.5, which
rest on code plus measurement. Moderate for the scPDSI differences marked
"divergence" or "variant", whose reference is the climate-indices Wells port
rather than the Wells C++ source. The AAFC comparison and Dask behaviour are
unverified.

### Next steps for the xclim maintainers

1. Consider delegating to climate-indices (BSD-3) as an optional dependency, or
   porting from 3.3.0 with attribution, instead of repairing a translation of
   code that was removed in 2023. 3.3.0 shares none of the four standard-path
   defects measured here, though it has the one in section 6.1.
2. If the current code is kept, fix in this order: K′ units; `V = PV`; potential
   loss; CAFEC zero rule. Use `declare_units`/`convert_units_to`, convert
   fluxes to amounts, and run the arithmetic in one unit system.
3. Treat scPDSI as unfinished. Rebuild it from the Wells C++ source (backtracking,
   K′ without 17.67, calibration order, extreme-sum filter, adaptive regression,
   thresholds, V reset, contraction guard, scPHDI and scPMDI), or leave it out of
   the first merge.
4. Define missing-data behaviour; use `min_count` or completeness checks when
   aggregating; validate calibration bounds and a January start.
5. Replace the tests: fix the import and signature, add an external benchmark,
   and remove the self-generated constants. Correct the changelog name.
6. Add attribution and licence notices.

### Next steps for climate-indices

1. Fix section 6.1: make missing periods stay missing, and make a non-finite K
   raise, as `scpdsi()` does. Check the fix against `pdi.f`'s handling of
   missing data before choosing the behaviour.
2. Rename the `noaa_*` variables.

## 10. Wider picture and a shared benchmark

Users can already get different "PDSI" values from different packages without
any bug: the recursion lineage (NCEI `pdi.f` or Wells), the establishment
threshold (1 or 0.5), PET method, AWC, calibration period and end-of-record
handling all change results. Within climate-indices, the two lineages disagree
on about 6.3% of months, by up to 8 units (`VALIDATION.md`). Output should name
the lineage and calibration period.

**Climate Engine: unverified.** One web search found no public Climate Engine
PDSI source. Its listings concern the gridMET PDSI product, described by the
[Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/GRIDMET_DROUGHT)
as a modified Palmer formula. No code or output was compared.

A shared benchmark could contain:

| Tier | Content | Suggested acceptance |
| --- | --- | --- |
| 1. Operational | nClimDiv, 344 divisions: fixed P, PET, AWC, calibration 1931–1990 | the PDSI, PHDI, PMDI and Z ceilings in `tests/test_nclimdiv_reference.py` |
| 2. Published | Palmer (1965) worked tables | print-rounding bounds, with known misprints listed |
| 3. scPDSI | Wells C++ outputs and duration factors on the same inputs | absolute 5×10⁻⁵; plus 2nd/98th percentiles near ∓4 |
| 4. Contract | NaN gaps, zero denominators, D̄ = 0, arid and tropical series, AWC ≤ 1 in, partial years, bad calibration bounds | missing never becomes a number; invalid input raises |

Tier 1 inputs are US government data. Tiers 1–3 and their provenance records are
under `tests/fixture/`. All external references so far are US or
Palmer-era; a non-US operational reference is still missing.

# Design: self-calibrating PDSI (scPDSI)

**Status**: approved for planning
**Tracking**: [#716](https://github.com/monocongo/climate_indices/issues/716) (epic) — [#717](https://github.com/monocongo/climate_indices/issues/717), [#718](https://github.com/monocongo/climate_indices/issues/718), [#719](https://github.com/monocongo/climate_indices/issues/719), [#720](https://github.com/monocongo/climate_indices/issues/720), [#721](https://github.com/monocongo/climate_indices/issues/721), [#722](https://github.com/monocongo/climate_indices/issues/722)

## Problem

`climate_indices` claims a "Self-calibrated Palmer Drought Severity Index" (scPDSI) output in its docs and CLI, but `palmer.py` contains no self-calibration procedure — `pdsi()` only computes standard PDSI/PHDI/PMDI/Z-Index using Palmer's (1965) fixed national duration-factor and K-factor constants. There is no known definitive open-source Python implementation of scPDSI; a correct one is a genuine contribution, not a routine feature add.

## Reference algorithm

Traced directly from the Wells, Goddard & Hayes reference C++ implementation (`Sibada/scPDSI` on GitHub, GPLv3 — used here for algorithm reference only; see "Licensing" below for why no GPL source enters this repository), specifically `Rext_PDSI_mon(sc=true)` and the functions it calls (`Calcd`, `CalcK`, `CalcZ`, `CalcDurFact`, `get_Z_sum`, `LeastSquares`, `CalcX`/`CalcOneX`, `Calibrate`).

1. **Water balance + CAFEC coefficients** (α, β, γ, δ) — identical to standard `pdsi()`. Fully reusable as-is from `_calc_water_balances`/`_calc_cafec_coefficients`.
2. **Unnormalized K′ factor**: per month, `D[m]` = mean absolute departure (`precip - CAFEC-precip`) over the calibration period only; `k_raw[m] = 1.5·log10((trat[m]+2.8)/D[m]) + 0.5` where `trat` is the existing moisture-demand/supply ratio. This is the *same formula* the current code already computes in `_calc_kfactors` — self-calibration skips the final `ak = 17.67 * k_raw / Σ(D·k_raw)` normalization step that the standard algorithm applies.
3. **Raw Z-index**: `Z_raw[y,m] = (precip[y,m] - CAFEC_precip[y,m]) × k_raw[m]`, computed for **every** period in the record, not just calibration years (unlike step 2's `D`/`trat`, which are calibration-period-only).
4. **Duration factor fitting** (the actual self-calibration, replacing Palmer's fixed 0.897/(1/3)):
   - For wet (`sign=+1`) and dry (`sign=-1`) separately, and for 10 window lengths (3,6,9,12,18,24,30,36,42,48 months): compute the rolling sum of `Z_raw` over that window, restricted to the calibration period, and pick one representative extreme sum per length.
     - Dry side: the single most extreme (most negative) rolling sum, unfiltered.
     - Wet side: the largest rolling sum that is not an outlier — specifically, excluding sums whose ratio to the 98th-percentile rolling sum exceeds 1.25 (a "freak anomaly" filter). This wet/dry asymmetry is a real feature of the reference implementation, not a bug to normalize away — we're porting it faithfully for numeric parity.
   - Fit a line (window length vs. extreme sum) via least squares, but adaptively: if the correlation coefficient is below 0.85, drop the longest-window point and refit, repeating down to a minimum of 4 points.
   - Adjust the intercept so the line passes through the most extreme (sign-weighted) residual point among those used in the final fit.
   - Normalize: `slope /= (sign×4)`, `intercept /= (sign×4)` — this anchors the fitted line to represent a PDSI value of ±4, which is what turns it into a duration-factor slope/intercept pair (`wetm`/`wetb`, `drym`/`dryb`) usable in the same recursion formula as Palmer's fixed constants.
5. **Recursion**: run the existing PDSI state-machine recursion once, fed `Z_raw` and the fitted duration factors, producing an *uncalibrated* scPDSI.
6. **Iterative percentile rescaling** (run exactly 3 times):
   - Restrict the current PDSI values to the calibration-period window.
   - `dry_ratio = -4 / percentile(calib_window, 2%)`, `wet_ratio = 4 / percentile(calib_window, 98%)` — using a specific kth-largest order statistic (`k = int(percentage × n)`, 1-indexed, NaN-safe), not `numpy.percentile`'s interpolation.
   - Rescale every Z value by the appropriate ratio (cumulative across the 3 iterations — each iteration scales the *already-rescaled* Z from the previous one).
   - Rerun the full recursion with the newly rescaled Z.
7. **Outputs**: scPDSI, scPHDI, scPMDI (WPLM), and the final rescaled Z-index, derived from the last recursion pass exactly the way `pdsi()` already derives PHDI/PMDI/Z-Index from X1/X2/X3/Prob today.

### Verified consistency with the existing standard-PDSI code

The reference `CalcOneX` reformulates the recursion in terms of duration-factor slope/intercept (`m`, `b`) and `c = 1 - m/(m+b)`:

```text
newX3 = c·X3 + Z/(m+b)
ZE    = (m+b)·(wd·0.5 - c·X3)
```

Substituting Palmer's original constants (`p=0.897`, `q=1/3` → `m=0.309`, `b=2.691`) gives `Z/(m+b) = Z/3` and `ZE = wd·1.5 - 2.691·X3` — which matches `palmer.py`'s existing hardcoded `_statement_180`/`_statement_170` formulas exactly. This confirms the reference algorithm's recursion is the same one already implemented here, just currently hardcoded to one specific (m, b) pair instead of taking it as a parameter.

## Architecture

- **`palmer.py`**: generalize the recursion functions (`_statement_170`, `_statement_180`, `_statement_190`, `_statement_200`, `_statement_210`) to read duration factors from `data["wetm"]`/`data["wetb"]`/`data["drym"]`/`data["dryb"]` instead of hardcoding `0.897`/`3.0`, defaulting to today's values so `pdsi()` is unaffected. Add the new public `scpdsi()` function here, reusing the water-balance/CAFEC code and the now-generalized recursion.
- **New sibling module** (self-calibration statistics): rolling-window extreme-Z-sum, the adaptive least-squares duration-factor fit, and the kth-largest/percentile order statistic. These are pure functions independent of the `data`-dict machinery in `palmer.py`, so they get their own module to keep `palmer.py` from growing past a reasonable size and to keep cohesive, independently-testable units (per the project's "many small files, high cohesion" convention). Exact module name to be decided during PR2's planning.
- No xarray/CLI/typed_public_api wiring in the core `scpdsi()` PR — `pdsi()` itself has no xarray wrapper yet either (tracked separately, `TODO(v2.5.0)` in `palmer.py`), so `scpdsi()` follows the same numpy-only precedent. CLI wiring is its own sub-PR (#721).

## Fixture / validation strategy

**No R installed on this machine**, and even if there were, `install.packages('scPDSI')` compiles the same underlying C++ via Rcpp — so the plan targets that C++ directly.

**Oracle**: the reference source (`Sibada/scPDSI/src/{pdsi.cpp,pdsi.h,pdsi_ext.cpp}`) has exactly 4 Rcpp-specific dependencies: `Rcpp::NumericVector`, `Rcpp::NumericMatrix`, `Rf_error`, `Rf_warning`. A small shim (plain `std::vector`-backed replacements, exceptions for errors) lets this compile standalone with `clang++` — no R, no Rcpp, no new project dependency.

**Licensing constraint**: `Sibada/scPDSI` is GPLv3; `climate_indices` is BSD-3-Clause. The oracle is built and run **entirely outside this git repository** (session scratchpad), and only its numeric outputs are committed — no GPL source, headers, or derivative code enters the repo. This mirrors how the existing NOAA-derived Palmer fixtures work: external data in, no third-party source distributed.

**Inputs**: existing `tests/fixture/palmer/<div>/precips.npy`+`pet.npy` (344 climate divisions) and `tests/fixture/palmer_awc.json`, with `data_start_year=1895`, `calibration_year_initial=1931`, `calibration_year_final=1990` — matching the conventions already established in `tests/conftest.py` and `tests/test_palmer.py`. Note the oracle's public R-facing entry point (`Rext_init`) assumes millimeter input and internally divides by 25.4 to get inches; our fixture data is already in inches, so the oracle harness replicates `Rext_init`'s internal setup directly in inches rather than reusing that entry point literally.

**Outputs committed**: `scpdsi.npy`, `scphdi.npy`, `scpmdi.npy`, `sczindex.npy` per division, plus a `provenance.json` entry citing Wells, Goddard & Wilhite (2004) and documenting the cross-validation methodology.

**Test tolerance**: start at `ATOL=5e-5`, `RTOL=0` — the same tolerance `test_palmer.py` already uses for standard PDSI against NOAA reference values. Given the self-calibration pipeline involves order statistics and an iterative refinement loop (more floating-point-path-sensitive than the direct standard-PDSI computation), this may need to widen — but only if a specific, explainable divergence is found, not preemptively.

## Epic decomposition

No single PR here is reviewable as one diff — this is genuinely epic-sized. Six independently-scoped sub-issues under [#716](https://github.com/monocongo/climate_indices/issues/716):

| PR | Issue | Scope | Depends on |
|----|-------|-------|------------|
| 1 | [#717](https://github.com/monocongo/climate_indices/issues/717) | Generalize the recursion to accept duration factors (pure refactor, behavior-preserving for `pdsi()`) | — |
| 2 | [#718](https://github.com/monocongo/climate_indices/issues/718) | Self-calibration statistics module (duration-factor fitting, percentile order statistic), TDD'd against hand-built arrays | — |
| 3 | [#719](https://github.com/monocongo/climate_indices/issues/719) | Reference oracle + scPDSI fixture data for all 344 divisions | — |
| 4 | [#720](https://github.com/monocongo/climate_indices/issues/720) | `palmer.scpdsi()` — wires 1+2+3 together, TDD against 3's fixtures | 1, 2, 3 |
| 5 | [#721](https://github.com/monocongo/climate_indices/issues/721) | CLI wiring (`__main__.py` Palmers dispatch) | 4 |
| 6 | [#722](https://github.com/monocongo/climate_indices/issues/722) | Docs correction and epic closeout | 4, 5 |

PRs 1–3 have no dependencies on each other. PR 4 is the integration point. PRs 5–6 close out the CLI/docs surface that was deliberately left honest ("not implemented") in the prior session's dispatch-bug fix.

## Testing approach (applies across PRs 1, 2, 4)

TDD throughout, per project convention (`.claude/rules/testing.md`):
- PR1: existing `tests/test_palmer.py` (1065 tests) is the regression safety net — must pass unchanged before and after the refactor.
- PR2: new unit tests with small, hand-computed expected values (no fixture dependency).
- PR4: new `tests/test_scpdsi.py`, written against PR3's fixtures *before* `scpdsi()` exists (red), following the same pattern as the existing `test_pdsi` in `tests/test_palmer.py`.

## Out of scope

- xarray wrapper for `scpdsi()` (matches `pdsi()`'s current numpy-only state; tracked separately for both under the existing `TODO(v2.5.0)`).
- Unifying the `calibration_year_initial`/`calibration_start_year` naming split (documented as a known, deliberately-undisturbed inconsistency in `src/climate_indices/CONTEXT.md`).
- Weekly-timescale self-calibration (the reference implementation supports it; `climate_indices` is monthly-only for Palmer indices, matching existing `pdsi()` scope).

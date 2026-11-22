# Validation Status

This document records evidence for the library, including the 3.0.0 release artifacts and the 3.1.0 development version.
It separates independent external validation, same-author reference reproduction, synthetic characterization, and regression coverage.
The 3.1.0 additions are the compound-hazard module and the energy family.
Neither has an adopted product-level external oracle.
The 3.2.0 addition is SPAI, whose evidence is regression, specification-level, and an estimator cross-check; it also has no adopted external oracle.
The compound evidence is a same-author published-method reproduction, a synthetic bootstrap characterization, and interface equivalence.
The energy evidence includes an exact published example, independent-implementation comparisons on synthetic and real-series inputs, and specification-level rows.
Their scope limits are in the sections below.

## Test Commands

| Scope | Command | Expected result |
| --- | --- | --- |
| Core suite | `uv run pytest -m "not benchmark and not validation"` | Unit, property, xarray, release guardrail, and regression tests pass. |
| Validation marker suite | `uv run pytest -m validation` | External comparisons and specification checks, including committed NOAA EDDI fixtures, the SPEIbase v2.11 SPEI plausibility floors and input-matched log-logistic comparison, the R `SPEI` log-logistic cross-implementation fixtures, the standard-Palmer nClimDiv comparison, the Palmer (1965) printed worked-example stage checks, the Srock et al. (2018) HDW event-discrimination fixture, the NOAA CPC degree-day worked example and the xclim/windpowerlib wind-profile fixtures (independent implementations on synthetic inputs), the R `SEI` empirical-standardization and energy-wrapper comparison, and the flood-family source-identity and API cross-implementation checks (see "Flood Evidence Classification"), pass; the nine committed compound summer cases reproduce the same-author counts and LMF without independent-validation or significance claims; scPDSI oracle cross-validation passes. |
| Lint | `uv run ruff check src/ tests/` | No lint findings. |
| Format | `uv run ruff format --check src/ tests/` | No formatting changes needed. |
| Notebooks | `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/xarray_getting_started.ipynb notebooks/palmer_indices_xarray.ipynb notebooks/eddi_xarray.ipynb notebooks/icechunk_spi_spei.ipynb` | Requires Python 3.12 or later. All four notebooks execute from a clean checkout; the Icechunk notebook uses the optional extra that the `dev` group installs on Python 3.12 or later. |
| Notebook smoke (Zarr/Dask tutorial, local only) | `bash scripts/smoke_e2e_notebook.sh` | `notebooks/zarr_dask_spi_spei.ipynb` executes end to end from a fresh kernel against the prepared sample inputs. |

## Per-Index Evidence

| Index | Status | Tolerance | Evidence | Known gap |
| --- | --- | --- | --- | --- |
| SPI | Validated (internal) + qualified external comparison (Pearson III) | Gamma `atol=1e-8`, Pearson `atol=1e-5` in xarray equivalence tests; NOAA NCEI climdiv characterization ceilings documented in `tests/test_ncei_spi_reference.py` | Legacy NumPy fixtures and xarray equivalence tests compare gamma and Pearson outputs across scales. `tests/test_ncei_spi_reference.py` characterizes `indices.spi()` (Pearson III, full-period-of-record calibration) against NOAA NCEI's operational climdiv SPI product for all 344 climate divisions and all 7 published timescales; marked `validation`. | `tests/fixture/ncei_spi/provenance.json` records that NCEI's climate-divisional SPI product's independence from this codebase's lineage could not be confirmed as airtight from public documentation alone (the survey exists only on the unmerged branch `research/spi-dataset-survey`, commit `79fec847`; the path is absent from `main` and from `v3.1.0`), and that NCEI's actual calibration behavior is a full/expanding period-of-record window rather than its documented fixed 1931-1990 baseline (Baldwin & Chen 2020). The WMO Falls City output is not a numerical reference. Stage checks use the guide's cited Edwards & McKee report; see below. |
| SPAI (`indices.spai`, `climate_indices.spai`) | Regression only; specification-level, with an independent-implementation cross-check for the t location-scale fit and a closed-form check for the normal fit (no external reference dataset adopted) | t location-scale within `atol=1e-3` of `scipy.stats.t.fit` and a log-likelihood no lower (measured 3e-6 on the synthetic seasonal series); `atol=1e-12` against an independent `scipy` normal computation; the t location-scale reference fit's pooled Calibration Period `abs(mean) <= 0.05` and `abs(sd - 1) <= 0.05` (measured `<= 0.015` and `<= 0.006` over five seeds); NumPy, eager, and Dask paths equal exactly | `tests/test_spai.py` checks the reference fit's pooled calibration moments, monotonicity in precipitation within each calendar period, one linear map of the anomaly under the normal fit, NaN propagation through the accumulation windows, a constant calendar period (a constant score, never infinite), a calendar period the Calibration Period never observes (NaN), a cell whose pooled calibration anomalies have no spread (NaN, with no runtime warning), the first `scale - 1` missing values, the two-year Calibration Period floor, an unchanged fit when data outside the Calibration Period changes, daily periodicity, the Pearson-to-normal fallback warning event, `fitting_params` reuse, the rejected distributions and periodicities, and exact agreement between the NumPy, eager xarray, per-cell, and Dask routes. | The comparison against `scipy` is an estimator cross-check and a closed form, not an independent scientific oracle, and its input is synthetic. No published SPAI values are committed: the article is paywalled, so the anomaly definition rests on the authors' closure and accessible secondary sources, and a body-level discrepancy cannot be excluded; the paper's Indian and Arkansas case studies are not reproduced either. The pooled calibration moments hold for the t location-scale reference fit only: the pooled anomaly is a scale mixture, so the lighter-tailed Pearson and normal fits reach the `+/-3.09` clip on wet-season values (measured pooled `abs(sd - 1)` up to about 0.07 and 0.11 over five seeds). Clipping reduces spread but does not guarantee `sd < 1`. The pooled moments hold over the whole Calibration Period only, not within each calendar period. See "SPAI Evidence Classification" and #1480. |
| SPI zero placement | SEI independent-implementation comparison; SCI estimator cross-check; closed-form and property regression | Zero-placement constants `atol=1e-12`; parameter-matched full series `atol=1e-6`; end-to-end `atol=0.01`; SCI centre-of-mass `atol=0.2` (Weibull-vs-empirical `p0` estimator difference) | `tests/test_zero_handling.py` pins each ADR-0015 mode's normal-scale closed form at `p0 ∈ {0.1, 0.5, 0.9}`, the ideal sample means, and the zero/positive ordering. `tests/test_zero_handling_reference.py` (marked `validation`) compares the modes to committed fixtures generated by R `SEI` 0.2.0 (`std_index(..., lower = 0, cens = "prob" \| "normal")`) and R `SCI` 1.0-3 (`p0.center.mass = TRUE`): the SEI zero constants match exactly, the full series matches when run on SEI's fitted shape/rate, and the end-to-end difference is at most 0.0032 (SEI's maximum-likelihood gamma versus this library's Thom (1958) closed-form approximation). SCI's committed `p0` estimator is independently produced and its documented `(np+1)/(2(n+1))` placement is compared to the exercised mode. The fixture also asserts its SEI shape/rate differ from this library's own fit, so a self-derived fixture cannot pass. Fixture provenance and tolerances are in `tests/fixture/zero_handling/provenance.json`; regenerate with `scripts/prepare_zero_handling_fixtures.py`. | An all-zero calibration step diverges by design: SEI's censored modes assign their constant without a fit, while this library follows ADR-0015 decision 5 and scores the step's zeros as classic (SEI classic gives `+Inf`). SCI estimates `p0` with the Weibull plotting position `np/(n+1)` and places zeros at `(np+1)/(2(n+1))`, so its constants differ from this library's `np/(2n)` by at most 0.16 in z for this fixture and more for sparser steps; `sci_center_of_mass_probability` records the estimator it uses. |
| SPEI | Validated (internal) + input-matched cross-implementation comparison (log-logistic, SPEIbase v2.11); gamma/Thornthwaite plausibility check | Gamma `atol=1e-8`, Pearson `atol=1e-5` in xarray equivalence tests; log-logistic like-for-like floors (correlation >= 0.97) in `tests/fixture/speibase_cru_ts/provenance.json`; gamma plausibility floors in `tests/fixture/speibase/provenance.json` | Legacy NumPy fixtures and xarray equivalence tests compare gamma and Pearson outputs across scales. `tests/test_speibase_like_for_like.py` feeds `indices.spei()` the CRU TS 4.09 precipitation and FAO-56 Penman-Monteith PET that SPEIbase v2.11 was computed from, standardizes each 0.5-degree cell with the log-logistic distribution over 1901-2024, averages the per-cell series inside three CONUS climate divisions, and reproduces the CSIC SPEIbase v2.11 grids (correlation 0.9996-0.99999, mean absolute difference 0.003-0.016 across divisions and timescales). `tests/test_speibase_reference.py` retains the earlier gamma/Thornthwaite comparison against grid-cell-mean SPEIbase as a plausibility check. Both are marked `validation`. | The like-for-like comparison removes the PET-method, distribution-family, and combined precipitation/spatial-support confounds of the plausibility check. `climate_indices.lmoments.fit_glo` is a port of the same `lmom` PELGLO routine R `SPEI` uses, so close agreement is expected, not algorithm independence. The small residual is dominated by a PET day-length convention difference: the test uses leap-aware month lengths (matching public SPEIbase `R/functions.R`), while a near-uniform month reproduces the committed v2.11 grids two to four times more closely. The gamma/Thornthwaite comparison remains plausibility only; see "SPEI Plausibility Classification" below. |
| SPEI log-logistic | Cross-implementation comparison, same estimator lineage (R `SPEI`) | Series `atol=1e-6`; GLO parameters `rtol=1e-8`, `atol=1e-6` | `tests/test_spei_loglogistic_reference.py` (marked `validation`) compares `indices.spei()` (`Distribution.loglogistic`) against the R `SPEI` package (`distribution = "log-Logistic"`, `fit = "ub-pwm"`) for the same three CONUS climate divisions at timescales 1/3/6/12, plus a short highly skewed synthetic series whose constant calendar step is unfittable at scale 1. Both sides fit the identical `(P − PET) + 1000` scaled series. | R `SPEI` fits each calendar step with unbiased-PWM L-moments (`TLMoments::PWM` + `lmom::pelglo`), the estimator `lmoments.fit_glo` implements, so the standardized series agree at ~1e-12 and the parameters to ≤~1e-10; the recorded tolerance absorbs cross-platform wobble only. `SPEI` does not clip to `[-3.09, 3.09]`, so the tests exclude the clipped tail and assert the clip separately. See "SPEI Log-Logistic Cross-Implementation" below. |
| SPEI GEV (`indices.spei()` with `Distribution.gev`) | Specification-level, with independent SciPy cross-checks of the CDF and fitted-transform pipeline; no committed R `lmom::pelgev` fixture | CDF `rtol=1e-12` against `scipy.stats.genextreme.cdf` for shapes -0.4 to 0.4 including exactly 0, with the transform's finite end of support checked; fitted parameters recover a known GEV from 200,000 draws within `abs=0.05` (loc, scale) and `abs=0.02` (shape); fitted-transform pipeline `atol=1e-9`; closed-form L-moments round-trip at `abs=1e-9` | `tests/test_gev.py` covers the shape solver, the Gumbel limit and the series/direct switch in `(1 - Gamma(1 + k)) / k`, the Taylor coefficients against the zeta expansion, the scalar and cell-axis fits, the SciPy CDF comparison, the infinite z of a value beyond the fitted support and its clip in `spei()`, `fitting_params` reproduction, block-versus-per-cell equality, the NumPy and xarray/Dask routes, and rejection on `spi()`, `standardized_index()` and `fit_diagnostics()`. `test_spei_gev_matches_an_independent_scipy_pipeline` independently checks the fitted-transform pipeline using SciPy sample L-moments, a separate shape root solve, `genextreme.cdf` and `norm.ppf`. | No R `lmom::pelgev` fixture is committed (OQ-8 of the distribution-suitability design). The shape solve runs to about `1e-14` and snaps no small `abs(k)` to the Gumbel limit, so any difference from `lmom` `pelgev` near `k = 0` is unmeasured. Pure Python, with no native kernel or parity registration; a Rust GEV kernel is a follow-on under ADR-0017. |
| Standardized index, empirical distribution (`indices.standardized_index()` with `Distribution.empirical`) | Independent-implementation comparison against R `SEI` 0.2.0 (seam fixture); specification-level and regression pins | Seam fixture probabilities `atol=1e-15`, normal scores `atol=1e-12`; closed-form transform pins at `atol=0`; native and Python empirical transforms bit-identical | `tests/test_energy_sei_reference.py::test_standardization_seam_against_sei` (marked `validation`) compares `standardized_index()` with `Distribution.empirical` directly against committed R `SEI` 0.2.0 output in twelve monthly cases; the energy-family rows below exercise the same seam through its thin `standardized_index` wrappers. `tests/test_empirical_distribution.py` pins the `SEI` plotting position `(1 + count(calibration <= x)) / (n + 2)` recomputed per calendar step in the test, the maximum-rank tie rule, the 10-value minimum-calibration rule, one probability shared across the `normal`, `probability`, and `bounded` output conventions, signed-series retention, sum-versus-mean rank identity, and time-major block equivalence; `tests/test_native_parity_empirical.py` asserts the empirical transform is bit-identical across the native and Python paths for every periodicity, grid shape, and output convention; `tests/test_hourly_periodicity.py` covers the hourly empirical-only seam. | The external `SEI` anchor is monthly and specific to that fixture's calibration windows. The hourly pooled-window route has no external comparison. Weekly periodicity is unsupported. The property pins recompute the `SEI` convention in test code rather than compare `SEI` output per property. |
| PET Thornthwaite | Validated | Synthetic-fixture `atol=0.001`; literature worked example `atol=4.0` mm/month | `tests/test_eto.py::test_eto_thornthwaite` covers synthetic regression fixtures; `test_eto_thornthwaite_literature_watson` compares against a Thornthwaite (1948) worked example (Watson & Burnett, 1995) in `tests/fixture/pet_literature/`. See `docs/algorithm_refs/pet.md`. | The source supplies daylight hours, not latitude; coordinate reconstruction is not independently validated. |
| PET Hargreaves | Validated | Synthetic-input sanity checks; literature worked example `atol=0.05` mm/day | `tests/test_eto.py::test_eto_hargreaves_with_fixtures` checks synthetic daily-input output shape and bounds; `test_eto_hargreaves_literature_mehta` compares against a Hargreaves-Samani (1985) worked example (Mehta, 2006) in `tests/fixture/pet_literature/`. See `docs/algorithm_refs/pet.md`. | The source supplies extraterrestrial radiation, not latitude/day of year; coordinate reconstruction is not independently validated. |
| PET Penman-Monteith (FAO-56) | Validated | `atol=0.05` mm/day for the final ETo; `0.05`-`0.1` MJ m-2 day-1 for the printed radiation intermediates | `tests/test_pm_eto.py` checks the Chapter 3 radiation, clear-sky, net-radiation, and wind helpers against FAO-56 worked Examples 10-16, and the full derived-input chain (`penman_monteith_eto`) against the daily grass-reference Example 18 (Uccle, ETo = 3.88 mm/day). `tests/test_pm_eto_xarray.py` checks the xarray path against the NumPy path. See `docs/algorithm_refs/pet.md`. | One daily worked example supplies the end-to-end oracle; the monthly Example 17 is not encoded, and the radiation-helper checks reproduce FAO-56's rounded printed intermediates rather than an independent computation. |
| PNP | Internal regression | Existing fixture tolerances in `tests/test_indices.py` and xarray PNP tests | Percent-of-normal fixture and xarray wrapper tests cover output shape and metadata. | None blocking 3.0.0. |
| PCI | Internal regression | Existing fixture tolerances in `tests/test_indices.py` and xarray PCI tests | Daily rainfall fixture and scalar xarray wrapper tests cover PCI. | None blocking 3.0.0. |
| EDDI | Validated | NOAA reference tests use `rtol=1e-5`, `atol=1e-5` | Committed paired NOAA PSL monthly reference ET/EDDI fixtures for 1-, 3-, and 6-month Timescales, 1979–2023, pass in `tests/test_noaa_eddi_reference.py`; maximum observed error is `2.43e-6`. | This is one fixed latitude/longitude subset, not a CONUS-wide assessment. |
| KBDI | Reference-reproduced, not independently validated | Published Figure 1 series reproduced with `atol=4.0` hundredths of an inch (table-vs-equation discretization, measured maximum 2.91); GHCN station regression `rtol=1e-12`, `atol=1e-9` mm | `tests/test_fire_kbdi_reference.py` reproduces the Keetch & Byram (1968) Figure 1 worked example from a committed fixture, audits the published net-rain wet-spell rule, and runs a complete 30-year GHCN-Daily PRCP/TMAX record for Fresno, California against frozen values from an independent plain-Python calculator; the xclim (Finkele variant) and WFAS cross-checks skip when their optional inputs are absent. Recurrence, gap, and unit contracts are covered by `tests/test_fire_kbdi.py`. | Figure 1 is an exact oracle only for the 1968 integer table workflow, not the corrected continuous Equation 18 the library implements; xclim is a later Australian/FFDI variant, and no reproducible WFAS point oracle was obtainable. |
| CFFWIS (FFMC, DMC, DC, ISI, BUI, FWI, DSR) | Validated against the NRCan reference implementation | All seven components `atol=1e-9` against full-precision `cffdrs_py` reference outputs; xclim sanity check at `atol=1e-9` for DC/DMC/BUI and `atol=0.5` for the FFMC lineage | `tests/test_fire_cffwis_reference.py` runs the 48-day Van Wagner & Pickett (1985) calibration sample committed as `tests/fixture/cffwis_vwp1985/` against reference values generated at `cffdrs_py` commit `0f57fcc`; the xclim cross-check skips when xclim is absent. Recurrence, gap, unit, and seasonal-carry contracts remain covered by `tests/test_fire_cffwis_moisture.py`, `tests/test_fire_cffwis_behavior.py`, and `tests/test_fire_dc_overwintering.py`. | The committed record is one calibration sample at latitude 40 N, so the external check exercises the northern DMC/DC day-length band only; the other latitude bands are covered by frozen regression vectors, and no independent (non-NRCan-lineage) CFFWIS oracle exists. |
| HDW | Event-discrimination reproduced; published magnitudes characterization only | Daily-series correlation floor `r >= 0.80` (measured 0.8679; floor stored in `tests/fixture/hdw_srock_cedar/provenance.json`); digitized reference uncertainty +/-5 hPa m s-1 | `tests/test_fire_hdw_reference.py` computes `fire.hot_dry_windy()` on the CFSR reanalysis profiles in `tests/fixture/hdw_srock_cedar/`, extracted at the grid point of Srock et al. (2018) Figure 4a (33.0 N, 116.5 W; 12 October through 9 November 2003), and confirms the daily maximum falls on 26 October 2003 -- the paper's significant fire-behavior day for the Cedar Fire -- 17% above the next-highest day. The digitized Figure 4a series is committed alongside the profiles, with its marker pixel coordinates for audit. | The paper publishes only figures, so the reference series is digitized (+/-5 hPa m s-1). Its adiabatically adjusted, independently maximized formulation (Section 3) is not the library's per-level VPD x wind product, and the digitized magnitudes are not reproducible from public metadata: the library's daily maximum exceeds them on five of 29 days, supplied by the 0000 UTC analysis on all five and by the 1200 UTC series alone on three, while the library's 1800 UTC values alone correlate at 0.945 with no exceedances. Magnitudes are therefore characterization, not validation; unit and example-value contracts remain covered by `tests/test_fire.py`. |
| PDSI, PHDI, PMDI, Z-Index (standard Palmer) | Qualified independent external-product validation | nClimDiv comparison ceilings documented in `tests/test_nclimdiv_reference.py`; library fixtures `atol=5e-5`, `rtol=0` | `tests/test_nclimdiv_reference.py::test_pdsi_vs_noaa_nclimdiv_qualified_external_validation` compares `pdsi()` against the operational NOAA NCEI nClimDiv arrays for all 344 climate divisions, January 1895 through December 2022 (measured median absolute difference 0.0127; 86.2% of months within 0.05). `tests/test_palmer.py` retains the library-generated standard-Palmer fixtures as regression coverage. Both are marked `validation`. `tests/test_palmer_literature.py` adds stage-by-stage checks against the worked examples printed in Palmer (1965), Tables 1, 2-3, 4, 9 and 12 (`tests/fixture/palmer_literature/`): the water balance, CAFEC coefficients, K normalization, P-hat/d/Z and the X recursion are reproduced within the stated tolerances (0.01 in for the water balance, 0.02 for X, otherwise propagated print rounding) except in the months and cells itemized in that directory's README. | Qualified rather than absolute: NCEI publishes two decimal places, and the committed `precips.npy`/`pet.npy` inputs are not byte-identical to NCEI's operational inputs, so the practical agreement floor is ~0.013 rather than 0.005. See "Palmer Validation Classification" below. |
| scPDSI | Independent implementation cross-validation; nClimDiv comparison is characterization only | Wells-lineage oracle fixtures `atol=5e-5`, `rtol=0` in `tests/test_scpdsi.py`; nClimDiv characterization ceilings in `tests/test_nclimdiv_reference.py` | `tests/test_scpdsi.py` compares the four self-calibrating outputs and fitted duration factors against Wells-lineage reference fixtures for all 344 climate divisions. `tests/test_nclimdiv_reference.py::test_scpdsi_vs_noaa_nclimdiv_characterization` characterizes aggregate agreement against nClimDiv, which applies standard Palmer's fixed national calibration rather than per-division self-calibration. Both are marked `validation`. | No qualified external-product reference yet; assessing DRI/WRCC scPDSI for that role remains open (issue #780). The NCEI Fortran and Wells C++ recursion lineages disagree on roughly 6.3% of months by up to 8 PDSI units, and on those months the NCEI-lineage values sit closer to NOAA operational data 88% of the time (`tests/fixture/palmer/provenance.json`). |
| PE (effective precipitation) | Regression only; specification-level (no external oracle adopted) | Eq. (2) identity `atol=0`; endpoint weights at `pytest.approx` defaults; naive double-sum equivalence `rtol=1e-14`, `atol=1e-13` | `tests/test_flood_reference.py` (marked `validation`) checks the Byun & Wilhite (1999) Eq. (2) two-day identity and the harmonic endpoint weights `w₁ = H_D`, `w_D = 1 / D`, and freezes the public signature; `tests/test_flood_pe.py` checks the harmonic double sum against an independent naive sum, full-window NaN behavior, missing-day recovery, and Spatial Block layouts. | No numeric oracle is committed. The tabulated Byun & Wilhite (1999) Table 5 follows the paper's variable summation duration, not the fixed 365-day contract of ADR-0014, and the identity of the processed Hickman input record remains unconfirmed (issue #1135); see "Flood Evidence Classification" and `tests/fixture/flood/README.md`. |
| EDI | Regression only; specification-level (no external oracle adopted) | Fixed-window algebra checked against closed-form expectations at NumPy defaults; contract tolerances in `tests/test_flood_edi.py` | `tests/test_flood_edi.py` checks Eq. (9)'s fixed-window algebra (the harmonic factor cancels, leaving `DEP / SD(PE)` per calendar day), calibration-period validation, per-calendar-day baselines, and missing-data behavior; `tests/test_flood_reference.py` freezes the public signature. | The published Table 5 end-to-end values need the variable-duration convention this implementation does not use and the Hickman, Nebraska record (issue #1135); no other numeric oracle was located (`docs/research/flood-oracle-survey.md`). |
| I_F (Flood Index) | Regression only; specification-level (no external oracle adopted) | Declared normalization and year boundaries checked against closed-form expectations in `tests/test_flood_index.py` | `tests/test_flood_index.py` checks `I_F = (PE − mean(PE_max)) / SD(PE_max)` against complete annual maxima, `year_start_month` boundaries, partial-period exclusion, and missing maxima; `tests/test_flood_reference.py` freezes the public signature. | The Deo et al. (2015) full text is paywalled and its abstract describes an exponential kernel the library does not implement — ADR-0014 selects the harmonic double sum — so no reproducible I_F values are available. |
| API (Antecedent Precipitation Index) | Independent-implementation comparison on synthetic inputs; specification and regression coverage | `rtol=0`, `atol=1e-14` mm against R `ahrapi`; constant-input limit `abs=1e-15` | `tests/test_flood_reference.py` compares the daily fixed-`k` recurrence against the R vector output and checks the analytic limit. `tests/test_flood_antecedent.py` covers the exact recursion, state round-trip, gap policies, and Spatial Blocks. | The R comparison uses complete input only. There is no published observed-input oracle or flood-prediction evaluation. |
| Flood events and per-period summaries (`flood.flood_events`, `flood.flood_event_summary`) | Regression only; specification-level (no external oracle adopted) | Hand-constructed events exact at `atol=0`; NumPy and per-cell xarray agreement at `rtol=0`; the strict `index > threshold` boundary is frozen | `tests/test_flood_events.py` checks onset, end, duration, peak, and raw-sum severity against a plain day-by-day reference over seeded block input, NaN, masked, infinite, empty, and boundary layouts; `tests/test_flood_event_summary.py` checks per-period counts, durations, maxima, and frequency against closed-form expectations, onset-period assignment, hydrological-year labels, and per-cell agreement between the NumPy and xarray paths. | Return period `T` is not implemented: the Deo et al. (2015) estimator is not recoverable, so `frequency` is an event rate and `docs/flood_applications.md` gives a caller-side ranked-severity recipe. The paper's Brisbane values (`I_F_acc = 118`, `I_F_max = 4.4`, `D_F = 104`, `T = 106.2 years`) are **not** claimed as reproduced; no input record or estimator is available (ADR-0014 decision 5). |
| Compound harmonization (`compound.harmonize`, `AlignedHazards`) | Regression only; specification-level (no external oracle adopted) | Exact Boolean and integer equality; the outputs are masks and counts, so no numeric tolerance applies | `tests/test_compound_harmonize.py` checks hand-counted strict-threshold ties and endpoints, float32 widening, month-label normalization, real leap days and interior absent dates, `exact` and `inner` alignment, and daily `any`/`all` reduction (including the complete-month rule) against an independent calendar loop. It also checks rejection of raw z-scores, out-of-range values, unsupported calendars, and conflicting Calibration Periods without clipping or interpolation, plus read-only results and the serializable manifest. These tests do not use the `validation` marker. | The declared-indicator route participates in the scoped summer reproduction below. This does not validate probability-input fitting, general harmonization, or daily-to-monthly reduction. A declared probability is a trust boundary and is not proven to be a PIT value or a good fit. Agreement of the beta xarray route (#1372) with this function is interface equivalence, not validation. |
| Compound joint statistics (`compound.joint_statistics`, `JointResult`) | Scoped reference-reproduced, not independently validated, for the nine public summer cases below; otherwise analytic/specification-level regression | Exact integer equality for counts and exact equality for the balanced Cartesian tables; `atol=1e-12` with `rtol=0` for closed-form probabilities; the Gaussian cases are predeclared 5-sigma Monte Carlo bands, not tolerances | `tests/test_compound_statistics.py` checks pooled and monthly counts against hand-counted contingency tables, one complete-case support for every marginal with different per-source validity masks (two and three sources), balanced two- and three-source independence (`LMF=1` exactly), the closed-form Gaussian joint probability `1/4 + asin(rho)/(2*pi)` at `rho=0` and `rho=0.5` within its declared band, disjoint hazards (`LMF=0`), a zero marginal (`NaN` plus the `undefined_denominator` flag), short support, sparse occurrences, all-missing grid cells, invalid steps never counting as non-exceedances, and pooled-versus-monthly recounting. Hypothesis tests cover source-permutation symmetry, count nesting, and probability bounds on generated validity masks. It also checks option and manual-`AlignedHazards` validation, read-only results, the JSON manifest, and lifecycle logging. These tests do not use the `validation` marker. | The analytic values come from closed form or hand counting. The separate public-study comparison below reproduces the same-author counts and LMF, not an independent oracle. Agreement of the beta xarray route (#1372) with this function is interface equivalence. `LMF=1` in a balanced design is a property of that design, not evidence of independence in real data. This descriptive function makes no inference about significance; see the separate bootstrap evidence below. Known gaps: the plan's nonzero-threshold quadrature and independent-uniform-envelope checks are not implemented, outside the summer reference scope. The nine public cases do not validate other thresholds, three-hazard real-data relationships, upstream SPEI, or the published significance mask. |
| Compound conditional statistics (`compound.conditional_probability`, `ConditionalResult`) | Regression only; specification-level (no external oracle adopted) | Exact integer equality for counts and exact equality for hand-enumerated windows; `atol=1e-12` with `rtol=0` for closed-form frequencies | `tests/test_compound_statistics.py` checks the `(0, 0)` identity against the joint and marginal counts, hand-enumerated forward windows, one counted success per given-positive anchor despite several target occurrences in its window, discarded incomplete trailing horizons (including a window that already holds a target occurrence), a discarded anchor whose window holds an unknown target step, asymmetry under a role swap, the response-window baseline against the one-step marginal and the joint probability, anchor-month grouping across a year boundary, daily civil-day steps across 29 February, a zero conditioning count (`NaN` plus the `undefined_denominator` flag), short support, all-missing grid cells, and a third source that never enters the selected pair's eligibility. Hypothesis tests cover count nesting and probability bounds on generated validity masks. It also checks option, role, and window validation, read-only results, the JSON manifest, and lifecycle logging. These tests do not use the `validation` marker. | No external oracle is used: the analytic values come from closed form or hand enumeration, so this is specification-level evidence, not external validation. A lagged association is not a validated causal cascade and is not an event-onset conversion probability. Agreement of the beta xarray route (#1372) with this function is interface equivalence, and this descriptive function makes no significance or confidence claim; see the separate bootstrap evidence below. The public LMF reproduction below does not validate conditional response windows. |
| Compound events (`compound.compound_events`, `CompoundEventSet`) | Regression only; specification-level (no external oracle adopted) | Exact index, count, lag, and flag equality; the outputs are indices, counts, lags, and flags (NaN where a lag does not apply), so no numeric tolerance applies | `tests/test_compound_events.py` checks the all-pairs concurrent and sequential enumeration, the four boundary-censor flags, and the rejected-bridge count against an independent brute-force interval scan over 24 seeded random scenarios with random unknown steps, plus hand-counted inclusive-end cases, repeated matches, lag-window boundaries, sequential bridges through an unknown step (rejected and counted), a matched and a rejected candidate in one window, empty catalogs with stable dtypes, validation of manually constructed `RunSet` objects, read-only result arrays, lifecycle logging, and the serializable manifest. These tests do not use the `validation` marker. | No external oracle and no published reference values are committed, and this specification-level status stands until an independent matching oracle exists. The seeded brute-force scan is an internal specification oracle, not external validation. Pair counts are not independent observations, so no significance claim is made or supported here. Agreement with the eager xarray catalog (#1373) is interface equivalence, not validation. |
| Compound monthly bootstrap (`compound.bootstrap_dependence`, `BootstrapResult`) | Synthetic characterization and regression only; screening evidence measured group by group, not as one configuration: at the default 1,999 resamples, 200-year records with the seasonal frequencies (0.35 to 0.55) and 1, 6, or 36 months of persistence with independent hazards; at 399 resamples, 200-year records with 1 to 30 months of persistence, monthly frequencies of 0.30 to 0.60 or 0.70 to 0.80, conditional lag windows to `(1, 3)`, 0.5-correlated hazards at 1, 6, or 12 months of persistence, and 100-year records; 36-month persistence at 399 resamples with independent hazards and at the default resamples with correlated hazards, 60- and 30-year records, and frequencies near 0.20 each failed a predeclared gate | Exact counts and seeded repeats; `atol=1e-12`, `rtol=0` for closed-form frequencies and linear quantiles. Characterization uses predeclared exact-binomial acceptance counts, not floating-point tolerances. | `tests/test_compound_bootstrap.py` covers paired versus independent sampling, tiny-block enumeration, month phase, original-chronology lag responses, missing-year segments, earliest ties, separate interval/null guards, zero-joint suppression, inclusive null ties, valid-resample fractions, immutable outputs, lifecycle events, and cell-tile invariance. `scripts/characterize_compound_bootstrap.py` records 128 independent Gaussian records per scenario over 183 cases: the four original 200-year scenarios, record lengths of 30, 60, and 100 years, persistence of 12 to 36 months, conditional lag windows `(0, 1)`, `(0, 3)`, and `(1, 3)`, frequency profiles from 0.20 to 0.80, correlated hazards, and 399 and 1,999 resamples. A 40,000-year simulation supplies the lagged-conditional truth, and a test checks that reference against quadrature at zero lag. The measured report is `docs/design/compound-bootstrap-characterization.json`. | Supported scope (maintainer decision, 2026-10-19), grouped by the resample count that measured each condition because the plan crossed no two of them: at the default 1,999 resamples, 200-year monthly records with the seasonal frequencies (0.35 to 0.55), source overlap of 1, 6, or 36 months, and independent hazards; at 399 resamples, 200-year monthly records with 1 to 30 months of overlap, lag windows `(0, 0)`, `(0, 1)`, `(0, 3)`, and `(1, 3)`, frequencies of 0.30 to 0.60 or 0.70 to 0.80, and hazards correlated 0.5 at the same time with 1, 6, or 12 months of overlap, plus 100-year records with the seasonal frequencies. Unsupported: 36-month persistence at 399 resamples for independent hazards and at the default resamples for correlated hazards, frequencies near 0.20, 60- and 30-year records, and `wide` at 36-month overlap; each failed a predeclared gate when the guards answered, except `wide` at 36 months, which was not run. No support guard was added, because no observable separates the passing from the failing cases without refusing a supported one. No independent-hazard case exceeded its per-month null rejection band, and the 36-month pooled null rejection falls from 7.5% at 399 resamples to 3.5% (3.2% to 3.8%) at the default resamples. No external validation, daily inference, field significance, or arbitrary-persistence guarantee is claimed. See the Compound Bootstrap Characterization section below. |
| Compound beta xarray routes (`compound.harmonize`, `joint_statistics`, `conditional_probability`, `bootstrap_dependence` on `xarray` input) | Regression only; interface equivalence with the NumPy routes (no external oracle adopted) | Exact equality for every count, flag, Boolean, and NaN position; the estimates come from the same NumPy arithmetic on the same counts, so no numeric tolerance applies | `tests/test_compound_xarray.py` compares every result field of each route with its NumPy counterpart, per month and pooled, for in-memory and Dask input over several spatial chunkings; checks bootstrap results are identical across tile boundaries; proves graph construction runs no task under a scheduler that raises; and checks custom and transposed dimensions, NaN versus explicit validity masks, daily-to-monthly reduction, `join="inner"` with its warning, a single location, and the rejected layouts (split time or hazard chunks, mismatched spatial chunks, mixed backends, missing or shifted coordinates, dimension-only or non-Gregorian time, and options that apply only to NumPy input). It also checks registry attributes, bounds, flag masks, no inherited `standard_name` or bounds, the versioned JSON manifest, h5netcdf and Zarr round trips, the `BetaFeatureWarning` and replayed-warning policy, and lifecycle events. `scripts/measure_compound_xarray_memory.py` records per-tile memory in `docs/design/compound-xarray-memory.json`. These tests do not use the `validation` marker. | Equivalence shows the Dataset routes reproduce the NumPy core, not that either is scientifically validated; the NumPy rows above state that evidence. Value-dependent support warnings are not emitted for Dask input, so read `quality_flags`. Memory figures are `tracemalloc` lower bounds from one machine, one platform, and one daily and one monthly configuration, with a manual peak-RSS spot check for `harmonize` only, and bootstrap tile width is set by compute time, which was not characterized. The route is beta (ADR-0012). |
| HDD, CDD, EDD (`energy.heating_degree_days`, `cooling_degree_days`, `energy_degree_days`) | Exact reference for the published cooling example; specification-level for heating and energy degree days | `rtol=0`, `atol=1e-12` °C day after the `(F - 32) * 5/9` conversion | `tests/test_energy_reference.py` (marked `validation`) reproduces NOAA CPC's published seven-day CDD series (`[67, 65, 70, 74, 78, 65, 68]` °F to `[2, 0, 5, 9, 13, 0, 3]` °F day, total 32 °F day) with the exact `(65 - 32) * 5/9` °C base. Heating, EDD, and missing-day cases are hand-calculated specification checks; EDD = HDD + CDD is an algebraic identity, not an independent oracle. `tests/test_energy_degree_days.py` repeats the CPC example outside the marker and covers input validation, dtypes, missing values, and scalar/empty shapes; `tests/test_energy_xarray.py` covers temperature-unit inference. | No public default base ships: C3S and Eurostat pair an activation threshold with a separate reference temperature, which one base per transform cannot reproduce. The functions take a caller-supplied daily mean temperature, so a caller holding Tmin and Tmax computes `(Tmin + Tmax) / 2` first; no other averaging or weighting is implemented, and no C3S numeric comparison is committed. |
| Hub-height wind-speed profile (`energy.hub_height_wind_speed`, `energy.wind_shear_exponent`) | Independent-implementation comparison on synthetic inputs; not observed-wind validation | `rtol=0`, `atol=1e-12` m s⁻¹; two-level shear recovery `atol=1e-15` as consistency evidence only | `tests/test_energy_reference.py` (marked `validation`) compares the power-law and log-law profiles against pinned `xclim` 0.62.0 `wind_profile` and `windpowerlib` 0.2.2 `logarithmic_profile` outputs for nine synthetic cases, including calm, identity, and missing masks. `tests/test_energy_wind.py` and `tests/test_energy_xarray.py` cover the contracts outside the marker. | The inputs are synthetic and heights are generic, so neither comparison establishes skill against observed hub-height wind. Calm handling and the default one-seventh exponent are project conventions. |
| Wind capacity factor (`energy.wind_capacity_factor`) | Regression only; specification-level (no external numeric curve adopted) | Exact rational arithmetic `rtol=0`, `atol=1e-15` for the cubic curve; exact region boundaries, table knots, and hand-calculated linear examples | `tests/test_energy_capacity_factor.py` checks the `generic` and user-supplied curves against `fractions.Fraction` arithmetic, region membership, cut-in and cut-out, air-density scaling, and masking. An hourly example checks that averaging the capacity factors is not the same as evaluating the curve at mean wind. `docs/research/energy-oracle-survey.md` found no matched numerical C3S curve to adopt. | No turbine-specific curve, observed generation, or model-to-model tolerance is validated. A curve of mean wind is not the mean capacity factor. |
| PV capacity factor (`energy.pv_capacity_factor`) | Independent-implementation comparison for the power/temperature term; specification-level for the thermal model and clipping | `rtol=0`, `atol=1e-12` in dimensionless capacity factor | `tests/test_energy_solar.py` compares nine synthetic cases against `pvlib` 0.15.0 `pvsystem.pvwatts_dc` (pinned commit and SHA-256 in `tests/fixture/energy_solar/`), with the cell temperature supplied through the inverse Jerez thermal equation. The thermal coefficients have hand-calculated tests. | The Jerez thermal model itself, irradiance transposition, optical and inverter losses, and plant output are not validated. Clipping to `[0, 1]` is project policy because the published potential can exceed one. |
| SREPI and standardized demand index (`energy.srepi`, `energy.standardized_demand_index`) | Independent-implementation comparison against R `SEI` 0.2.0; the empirical plotting position, tie rule, and minimum-observation count are fixed to `SEI`'s convention | `scaled_atol=1e-12`, `probability_atol=1e-15`, `normal_atol=1e-12` | `tests/test_energy_sei_reference.py` (marked `validation`) compares the empirical standardization against `SEI` 0.2.0 on the Bloomfield and Brayshaw (2021) German onshore wind capacity-factor series (852 monthly means, 1950-2020, calibration 1981-2010) in two cases and on synthetic capacity, production, demand, and residual arrays in ten cases; it then compares `srepi()` on the capacity, production, and reading cases and `standardized_demand_index()` on the synthetic demand cases. A gamma control is parameter-matched to `SEI`'s MLE values. Fixture provenance, checksums, and session information are in `tests/fixture/energy_sei/`. | The comparison is monthly, so no seasonal-diurnal or hourly calibration is validated against an independent implementation (`#1177` ships with specification tests only); only the two `reading` cases use provider data, so demand, production, and residual coverage is synthetic. The two real-series cases calibrate on 30 values per calendar month, below the 100-value empirical recommendation in `tests/fixture/energy_sei/warnings.txt`, so their probabilities are multiples of 1/32. The gamma control does not validate the library's gamma estimator. No C3S energy-indicator product comparison is committed, the beta distribution has no external reference distribution, and maintainer confirmation of fixture extraction and redistribution remains outstanding. |
| SRLI and residual load (`energy.srli`, `energy.residual_load`) | Independent-implementation comparison against R `SEI` 0.2.0; specification-level for the subtraction | The same `SEI` tolerances as SREPI | `tests/test_energy_sei_reference.py` (marked `validation`) compares `srli()` against `SEI` on the signed series, `tests/test_energy_srli.py` covers the NumPy contracts and negative retention, and `tests/test_energy_standardized_xarray.py` covers the xarray route and the empirical-only distribution restriction. | The committed residual cases are synthetic `demand - production` arrays, so no energy-system demand, capacity, or dispatch model is validated, and no C3S demand comparison is committed. |
| Energy-drought events (`energy.energy_drought_events`) | Regression only; specification-level (no external event catalogue adopted) | Hand-enumerated windows and exact integer counts; NumPy, eager, and Dask summaries compared exactly | `tests/test_energy_events.py` checks run metrics and period summaries on SPI, SREPI, and SRLI series, threshold-relative magnitude, missing steps, period label counts, and gridded Dask summaries that keep the time axis in one chunk. `docs/research/energy-oracle-survey.md` records that `SEI`'s event loop is not a missing-value-policy oracle. | No observed event catalogue validates onset, duration, or severity, and the `SEI` magnitude identity is a qualification, not parity. |
| Distribution comparison (`suitability.compare_distributions`, NumPy and xarray/Dask routes, Beta) | Regression only; specification-level plus synthetic characterization (no external oracle adopted) | Log-likelihood against `scipy.stats` at `rtol=1e-10`; AIC/AICc/BIC formula checks at `rtol=1e-12`; reused `fitting_params` bit-identical to the direct `spi`/`spei`/`standardized_index` call | `tests/test_suitability.py` checks `sum(logpdf)` against `scipy.stats.gamma` and `scipy.stats.pearson3`, the gamma zero-mass term, the generalized-logistic density against `scipy.stats.fisk` for a negative shape and for its reflection for a positive one, the GEV density against its closed form (rather than `scipy.stats.genextreme`) and against a numerical derivative of the transform's distribution function, the GEV log-likelihood against the closed-form sum at the returned parameters, the AIC/AICc/BIC formulas, `best` and tie semantics, the zero-mass parameter count for calendar steps with and without zeros, a recorded Pearson failure that is not replaced by gamma, a sample value outside a fitted support (zero likelihood, which is a failed candidate that cannot rank), a NaN cell in a spatial block, and the round trip through each route, including a spatial block. A seeded gamma-data run records how often gamma is selected. `tests/test_suitability_xarray.py` checks that the xarray route equals the NumPy core variable by variable for 1-D, 2-D and 3-D input and for the SPEI and `standardized_index` routes, that Dask spatial chunks equal the in-memory result, that a time axis split across chunks raises, that every variable carries the ADR-0028 attributes and records the clamped calibration years, that those attributes survive a NetCDF round trip, and that a fully missing series, cell or Dask chunk has no best candidate instead of aborting the grid. | The criteria are penalized likelihoods at the library's own estimates (Thom's (1958) closed-form gamma approximation, L-moments), not maximum-likelihood AIC/AICc/BIC, so a ranking mixes distribution fit with estimator efficiency. About 30 values per calendar step discriminate weakly, and the recovery rate is a characterization, not a guarantee. The xarray route is the NumPy core applied per block, so it inherits these limits and adds no new evidence about the estimators. Goodness of fit is not implemented yet (ADR-0028, #1586). |
| Drought categories (`suitability.drought_category`, `suitability.compare_categories`, Beta) | Regression only; specification-level against the U.S. Drought Monitor SPI/SPEI classification table (no external implementation compared) | Exact integer codes and counts; shares to `approx` default tolerance | `tests/test_suitability_categories.py` checks one literal on every class boundary of the table in float64 and float32 (−0.49 none, −0.5 D0, −0.8 D1, −1.3 D2, −1.6 D3, −2.0 D4) and the adjacent floats either side of each edge, ±3.09, NaN and masked input, rejection of the `probability` and `bounded` scales, a ten-pair hand-computed confusion matrix with two missing pairs (counts, `share_changed`, `share_changed_in_drought`), empty populations, the DataArray route (reduction over all or some dimensions, per-cell results against the NumPy route, a named-dimension reduction over a split time axis, a dimensionless input), the reserved result names, a differing auxiliary coordinate, and an unarrayable input. | The USDM table states ranges with two decimals and does not say how an exact boundary is classed, so the upper-edge-inclusive rule is the ADR-0028 contract, not the table's. Only the dry side is classified. `share_changed_in_drought` counts a pair as in drought when either classification is D0 to D4. The classes say nothing about whether a distribution fits. Agreement of the DataArray route with the NumPy route is interface equivalence, not validation. |
| Event concordance and category agreement (`suitability.event_concordance`, `suitability.category_agreement`, Beta) | Regression only; specification-level against the ADR-0028 section 7 counting rules and the #1569 pilot's two category-agreement definitions (no external implementation compared) | Exact integer counts; ratios to `approx` default tolerance | `tests/test_suitability_concordance.py` checks hand-built `RunSet` fixtures for a single overlap, one index event against two reference events (two hits, one matched index event), two index events against one reference event (one hit, two matched index events), a many-to-many case holding one hit, one miss and one false alarm, a one-step overlap, and every empty-denominator combination: no events on either side leaves all three ratios NaN, reference events without index events give POD and CSI zero with only FAR NaN, and index events without reference events leave POD NaN. It also checks the four censoring counts at the record edges, an unmatched run next to an unknown step, rejection of a run crossing an unknown step, the daily cadence, the recorded counting rule and event counts, and a non-`RunSet` source. A seeded randomized cross-check compares the four counts against a brute-force pair-by-pair interval intersection over 200 disjoint run-set pairs. For `category_agreement` it checks the pilot's two expressions over seeded codes with missing values, a hand-computed case, a missing-side drop, an all-missing input, the `compare_categories` cross-check (`exact` is the complement of `share_changed`), an unsigned-dtype input, a shape mismatch, float codes, an out-of-range code, a masked code, and rejection of a `DataArray` because there is no xarray route. | No independent reference product is compared, and every ratio is a point estimate with no interval. Censoring is counted, not corrected for: a censored run takes part in the counts like any other, so an event extending past the record can be scored as a miss or a false alarm. The metrics say nothing about whether a distribution fits, and a reference that is not independent of the index makes the agreement circular. |

## Reference Independence

This register separates the origin of the inputs from the origin of the expected
outputs. Using the same input is necessary for a numerical comparison; it does
not make the reference output self-generated. Unless stated below, overlap in
people, institutions, or code beyond the named sources is unknown.

| Evidence entry | Independence: inputs | Independence: expected outputs and shared methods |
| --- | --- | --- |
| NCEI SPI | Committed Palmer precipitation arrays come from nClimDiv. Matching to the SPI product's precipitation was spot-checked for division 0101, not established for all divisions. | NOAA NCEI publishes the operational division-level SPI. This project has NOAA/NIDIS roots, and both use Pearson III. The product's precise code lineage and production preprocessing are unknown. This is not the NOAA gridded SPI product that shares the Python code lineage. |
| SPI equation and table checks | The test constructs all monthly inputs. | Edwards & McKee supply equations and printed probability/score pairs. WMO cites their method. Neither tool output nor this library supplies the expected values. |
| SPAI and SPEI GEV estimator checks | The tests construct synthetic inputs and use the same samples on both sides. | SciPy supplies the t and normal reference computations. For GEV, SciPy supplies sample L-moments and the CDF, with a separate shape solve. These share mathematical models and sometimes SciPy primitives, not independent observed outcomes. |
| SPI zero handling | The fixture generator constructs synthetic precipitation and shares it across fits. | R `SEI` and `SCI` supply reference outputs. The parameter-matched case deliberately reuses SEI's Gamma fit; the end-to-end case does not. The modes share published conventions but not all mass estimators. |
| SPEIbase plausibility | NCEI climate-division inputs differ from CSIC's CRU TS 4.09 inputs. Station overlap is unknown. | CSIC publishes the outputs. Their distribution, PET method, spatial averaging, and record end differ from the test path. |
| SPEIbase like-for-like | Both sides use CRU TS 4.09 precipitation and FAO-56 PET; the PET day-length conversion may differ. | CSIC publishes the grids. The library and R `SPEI` share the `lmom` PELGLO estimator lineage. |
| SPEI R log-logistic | Both paths use the same prepared precipitation and PET or synthetic water balance, including the +1000 offset. | R `SPEI` produces the reference. Both paths use unbiased-PWM L-moments and the PELGLO estimator lineage. This is not estimator independence. |
| PET worked examples | Watson & Burnett, Mehta, and FAO-56 supply the inputs. The Thornthwaite and Hargreaves tests reconstruct missing geographic parameters. | Each publication supplies its own worked values. Both sides use the stated PET equations; the publications do not use this package. |
| NOAA EDDI | NOAA PSL supplies reference ET from the paired table. The library does not independently derive it in this test. | NOAA PSL supplies EDDI from the same table. Both follow Hobbins et al. ranking and plotting positions. Exact operational code and shared lineage are unknown. |
| KBDI | Figure 1 supplies its published example. NCEI GHCN-Daily supplies Fresno precipitation and temperature, not KBDI. | Keetch & Byram supply the table-lineage figure. A separate plain-Python calculator supplies the Fresno values using the same recurrence specification. Neither is an operational agency KBDI product. |
| CFFWIS | The published Canadian Forestry Service sample supplies noon weather and starting codes, via `cffdrs_py`. | The NRCan-lineage `cffdrs_py` generates the components, and the stated formula gives DSR. These share the Canadian FWI method and inputs. The xclim check also belongs to that method lineage. |
| HDW | The test extracts NCEP CFSR profiles at the same site and period as the paper; processing differs. | Srock et al. supply the significant fire-behavior date and reference figure. The authors used a different HDW formulation. The figure is not an exact magnitude target. |
| Standard Palmer, nClimDiv | The test uses committed precipitation, PET, and AWC; its precipitation and PET are not byte-identical to the operational inputs. | NOAA NCEI supplies operational indices. The library uses a port of NCEI-lineage `pdi.f`; the exact operational build and processing are unknown. Library-generated fixtures are regression only. |
| scPDSI, Wells and nClimDiv | Wells C++ and this library use the same committed precipitation, PET, AWC, and Calibration Period. The nClimDiv operational inputs are not an exact match. | Wells, Goddard & Hayes lineage C++ generates the scPDSI oracle outside the package. The Python path deliberately follows the Wells recursion, so it is not independent in algorithm. The fixture builder corrects one Z-Index alignment error. NOAA's standard Palmer product is a different calibration method and serves only as characterization. |
| Palmer (1965) stage examples | The publication supplies rounded weather, state, or intermediate values by stage. | Research Paper No. 45 supplies the printed numbers and arithmetic. It predates the software lineages. The standard and Wells recursions differ at spell establishment. |
| Flood API | Bartram's R example supplies synthetic rain; GHCN-Daily is not used as an API reference. | R `ahrapi` generates the comparison values, using the same published recurrence. The other flood indices have no adopted numerical reference. |
| Compound joint statistics | Vieira Passos's Figshare v2 indices and the study's preprocessing supply the declared hazards. | An adaptation of the same authors' notebook gives the expected counts and LMF. It is independent of package execution but not of the study's input or method lineage. |
| NOAA CPC cooling degree days | CPC supplies the seven temperatures and base. The test converts both to Celsius. | CPC supplies worked daily values and the total. It uses the same base-temperature formula, with no library output involved. |
| Energy wind profile and PV term | The tests construct synthetic inputs. The PV test derives an input to reproduce the specified Jerez cell temperature. | `xclim` and `windpowerlib` supply wind profiles; `pvlib` supplies the PVWatts power term. Shared profile and power equations do not validate observed wind, the Jerez thermal model, or observed generation. |
| Empirical standardization, SREPI, demand, and SRLI | Bloomfield & Brayshaw supply the ERA5-derived German wind series. Other cases use synthetic inputs, including demand and residual load. | Allen & Otero's R `SEI` generates expected scores. The library deliberately shares SEI's ranking, tie rule, and minimum count. The Gamma control shares its fitted parameters. |
| Reliability relationships and worked examples | EIA supplies aggregate utility reports, not event logs. The tests construct equivalent logs, use IEEE's example counts with reconstructed timestamps, or construct an EPRI-style scenario. | EIA supplies reporting relationships, IEEE 1366-2012 supplies example values including the Major Event Day case, and the EPRI-style result is derived analytically. None supplies a matched operational event log. |
| GRIDMET cross-provider comparison | Both provider products and the local calculation use gridMET inputs; exact input equivalence and ET crop are unresolved. | Earth Engine publishes the precomputed collection; Climate Engine supplies API values. Producer code and overlap between the two providers are unknown. The calendar windows differ from this package. |

## Reliability Benchmark Relationships

`tests/test_reliability_reference.py` supplies offline `validation` tests for #1345.
The committed panel and scenarios are in `tests/fixture/reliability_benchmarks/`.
Its README records source locations, retrieval checksums, units, licences, and evidence limits.
Its schema-validated provenance records the fixture checksum and fixed tolerances.
Missing fixtures fail rather than skip.
The default suite deselects the module's tests, but collection still imports the module and reads its fixtures.

**Sustained metrics and ASAI: regression coverage, not independent validation; no event-level oracle from an operational outage log.**
The EIA Electric Power Annual [Table 11.1](https://www.eia.gov/electricity/annual/html/epa_11_01.html)
provides 20 national aggregate rows for 2015–2024.
The panel keeps IEEE-method all-events and without-Major-Event-Day values separate.
It checks `CAIDI = SAIDI / SAIFI` for every row with propagated rounding intervals.
The interval half-widths are 0.05 minute for SAIDI and CAIDI and 0.0005 for SAIFI.
These assume uniform column precision with omitted trailing zeros, inferred from the displayed values rather than explicitly guaranteed by EIA.
A one-minute CAIDI corruption must fail the same check for every row.

Equivalent synthetic logs exercise `reliability_indices()` and its shared accumulators for every panel row.
They check finite values, bounds, elapsed reporting hours, and the ASAI relationship.
They do not reconstruct EIA's actual events, customer populations, or Major Event Day filtering.
Arithmetic tolerances are `rtol=0, atol=1e-12`, with `atol=1e-9` for customer-minutes and `atol=1e-15` for ASAI.
The source's 2003/2012 IEEE reporting methods do not establish conformance to the library's 2022 target.

**Momentary metrics: one published worked-example reproduction, not operational validation.**
IEEE Std 1366-2012 section 4.3.1, equations 47–49, supplies SAIFI 0.125, MAIFI 0.75, and MAIFI_E 0.375.
The test reproduces these exactly from the published customer counts and operation count.
The fixture uses constructed timestamps and durations; it does not validate sustained durations or raw-operation grouping.
The public docket source and the precise construction are recorded in the fixture README.
No IEEE document is redistributed.

**EPRI-style scenario: specification-level caller-side post-processing only.**
The [EPRI Climate READi Appendix A](https://apps.epri.com/climate-readi-modeling-guide/en/appendix.html#customer-outage-risk-cori)
relates customer-normalized outage hours to reliability indices.
The constructed scenario gives two outage hours per served customer for one hazard event.
A supplied frequency of 0.25 events per year gives an expected contribution of 0.5 hours per customer per year.
A second case halves the affected population and keeps CAIDI unchanged.
Thus the general relationship uses `frequency * SAIFI * CAIDI / 60`, including the affected-customer fraction.
The library result retains its supplied 24-hour period and does not annualize it.

These checks do not implement READi metrics CORi, LORi, or DORi.
They do not validate fragility curves, hazard probabilities, resilience forecasts, or Monte Carlo results.
The existing analytic event fixture remains specification and regression evidence under
[ADR-0026](docs/adr/0026-reliability-module-api-and-conventions.md).

## SPI With Consecutive Zero Precipitation

PR #585 addressed the gridded report in #582. `tests/test_spi_gridded_zeros.py`
exercises the public xarray API with two neighboring cells, monthly precipitation
in mm from 1990–2020, Timescale 3, and a 1990–2019 Calibration Period. The dry
cell has 27 zero March totals, one missing total, and two identical positive
March totals. Accumulation preserves the zeros and propagates the missing month;
Pearson has too few positive values for L-moments, and Gamma has no positive spread.
The Python Pearson path logs one `distribution_fitting_failures` event without
triggering the whole-block fallback. Unfitted positive March totals remain NaN.
Pearson's zeros keep the classic score in every `zero_handling` mode; Gamma's zeros
use `p0 = 27/29` and the requested mode. The neighboring cell agrees with its
standalone result. This is offline policy and regression coverage, not validation
against the reporter's input or the NDMC executable. No fit policy changed.

## SPI Worked-Example and Equation Checks

[WMO-No. 1090 (2012)](https://www.droughtmanagement.info/literature/WMO_standardized_precipitation_index_user_guide_en_2012.pdf)
section 6 (printed p. 9) describes the fitting and normal transformation without
listing numerical Gamma parameters. Tables 1–2 (pp. 4–5) give SPI classes and
recurrence summaries, not paired precipitation and fit data. Section 8 (pp. 10–13)
shows 32 months of Falls City precipitation and NDMC `SPI_SL_6.exe` output at
Timescales 1, 3, 6, 9, and 12. The output sentinel `-99.00` denotes insufficient
leading months. The [#778 investigation](https://github.com/monocongo/climate_indices/issues/778)
reported median absolute differences of about 0.4–0.6 and maxima above 1.5.
Two or three samples per calendar month and undocumented executable settings
prevent a defensible tolerance. The guide recommends at least 30 years.
No screenshot table is redistributed or used as a numerical oracle.

`tests/test_spi_wmo_worked_examples.py` instead checks separate stages against the
[Edwards & McKee (1997) report that WMO cites](https://mountainscholar.org/bitstream/handle/10217/170176/CLMR_Climatology97-2.pdf):

- Printed p. 19, Eqs. 3.6–3.9: the test uses synthetic 1 and 4 mm monthly totals in
  each calendar month of 2000–2001. The arithmetic mean is 2.5 mm and the geometric
  mean is 2 mm. Substitution gives `A = ln(1.25)`. The test pins shape and scale from
  45-digit Decimal evaluation of the equations. The `1e-13` absolute tolerance allows
  binary64 rounding and cancellation in `A`. These parameters are derived, not printed.
- Printed p. 26, Table 3.1: published pairs `0.1587, 0.5000, 0.8413` map to SPI
  `−1, 0, +1`. Synthetic totals use `x = -ln(1 - p)` mm and supplied unit shape and
  1 mm scale, so `G(x) = p` (Eqs. 3.10–3.12, pp. 19–21). There is no zero mass,
  missing input, or fitted calibration. The probability check allows `1e-15` for
  floating-point error. The score tolerance is `0.00021`, from the four-decimal
  probability rounding interval and the slope of the normal quantile near ±1.
- Figure 3.2 and printed p. 23 give SPI approximately +0.3 for 2 inches of
  January–March precipitation at Fort Collins (1911–1995). The report does not
  tabulate the exact fitted parameters and paired calibration series for that example.
  No station-level fit or full workflow reproduction is claimed.

**Independence:** The tests construct their inputs. Expected values come from the
report's equations and printed normal transformation table, not library output.
WMO and the report share the McKee method; any shared implementation lineage is unknown.

## SPAI Evidence Classification

`indices.spai()` and `climate_indices.spai()` were added in 3.2.0 (#1480) and are Beta on
both their NumPy and DataArray routes.

**Classification: regression and specification level, with an estimator cross-check.**
No independent SPAI reference dataset is committed, and none was located.
The Chanda and Maity (2015) abstract does not name the fitted distribution.
The paper's Indian and Arkansas case studies are not reproduced, and no SPAI values from
any source are pinned.

**Anomaly definition taken from the authors' closure; the article body is unread.** The
implementation computes the anomaly as the calendar-period mean subtraction alone, the
original paper's Eq. (5). The article itself is paywalled: the publisher returns HTTP 403
for the article, discussion, and closure pages, so the definition rests on the authors'
own closure, which reproduces the equation, and on accessible secondary sources. No
accessible passage shows a step the closure omits, but a body-level discrepancy cannot be
excluded, and contacting the authors remains open (#1480).
The closure reproduces that equation as `y_ij = x_ij - x̄_j` and, in its Eq. (2), gives
the anomaly series' standard deviation as `sqrt(Σ(x_ij - x̄_j)² / (n·m - 1))`, measuring
65.3 mm for Gangetic West Bengal against 138.9 mm for the raw series. A divided anomaly
would be dimensionless and near unit scale, so the original SPAI does not divide by the
calendar period's standard deviation. The same closure states the design property this
implementation preserves: "a given anomaly value in both low and high precipitation months
will produce an identical SPAI value", because one distribution is fitted to all the
anomalies "rather than 12 monthwise distributions used in case of SPI".

- Chanda, K., & Maity, R., *Closure to "Meteorological Drought Quantification with
  Standardized Precipitation Anomaly Index..."*, *Journal of Hydrologic Engineering* 21(5),
  <https://doi.org/10.1061/(ASCE)HE.1943-5584.0001369>.
- A peer-reviewed open-access application, *Scientific Reports* 15 (2025),
  <https://doi.org/10.1038/s41598-025-16859-9>, gives the anomaly as `y_ij = x_ij - x̄_j`
  and states that "no normalization by standard deviation (s_j) occurs here", then fits one
  t location-scale distribution to the whole anomaly series by maximum likelihood and
  applies `SPAI = Φ⁻¹[F_t(y_ij | μ, σ, ν)]`.
- A peer-reviewed application, *Earth and Space Science* (2020),
  <https://doi.org/10.1029/2020EA001264>, likewise subtracts the long-term mean for each
  calendar step, then standardizes with an empirical (Weibull plotting-position) CDF.

The t location-scale family is corroborated by the authors themselves: the same closure
records "t-location scale distribution is a suitable parametric distribution that can
describe the anomaly series for Indian precipitation (Chanda and Maity 2016)".

**The same authors' later papers instead divide the anomaly by the calendar period's
standard deviation.** The Hydrological Sciences Journal 2016 paper
(<https://doi.org/10.1080/02626667.2015.1006232>) defines the anomaly by each calendar
step's mean and standard deviation and fits one t location-scale distribution to the whole
anomaly series; their open-access *Meteorological Applications* 2020 paper
(<https://doi.org/10.1002/met.1881>) gives the anomaly as `(x_i,j - x̄_j) / s_j`, then fits
an empirical distribution. Those descriptions are the alternative variant; this library
implements the cited paper's Eq. (5). The difference re-weights the calendar periods, so
values from the two variants are not interchangeable, and no external reference settles
which variant a given application used.

What the committed evidence does show:

- The t location-scale fit's SPAI values agree with `scipy.stats.t.fit`'s on a synthetic
  strongly seasonal series within `atol=1e-3` (measured 3e-6), with a log-likelihood no
  lower than scipy's.
- The normal fit matches an independent `scipy` closed-form computation at `atol=1e-12`,
  and reduces to the pooled anomaly rescaled by its pooled calibration mean and standard
  deviation.
- The t location-scale reference fit's pooled Calibration Period output has
  `|mean| <= 0.05` and `|sd - 1| <= 0.05` (measured `<= 0.015` and `<= 0.006` over five
  seeds). The bound does not extend to the Pearson and normal fits. No fit promises
  unit variance for each calendar period.
- `tests/test_spai.py` checks clipping at both ends on June–September values of the
  synthetic seed-3 monsoon series. It uses Timescale 1 and pools the 600 monthly scores
  in 1961–2010 (`ddof=1`, no missing months; 2011–2020 excluded). Inverting the
  probability output before clipping gives wet-season extrema of about −7.91 and +6.86
  for Pearson, and −5.05 and +6.49 for normal. Clipping lowers the pooled SD from
  1.26660 to 1.0361247952 for Pearson (`|sd - 1| = 0.0361247952`), and from 1.00083
  to 0.8889721338 for normal (`|sd - 1| = 0.1110278662`). The normal uncut sample SD
  is `sqrt(600/599)` because the fit uses the population SD. The `1e-9` SD tolerance
  permits floating-point variation and rounding of the ten-decimal regression pins.
  These are fixture-specific results, not general limits.
- The NumPy, eager xarray, per-cell, and Dask routes agree exactly.

The estimator cross-check is a property of the implementation, not external validation,
and its inputs are synthetic. No value from the article body is pinned, so a convention
the closure and the secondary sources do not show would not be detected here; the
committed tests cannot settle whether the paywalled body adds a step. The publisher and
Academia.edu copies were not accessible, and the author-site PDF request timed out.
The 2025 application links TerraClimate inputs, but no matched numerical SPAI series or
runnable reference was located for its case study or for the original India or Arkansas
cases. Matching inputs alone would not establish the fit, calibration, or clipping.

## Compound Published-Method Reproduction

`tests/test_compound_reference_fixture.py::test_compound_api_reproduces_public_summer_reference`
provides nine offline `validation` cases against `compound.harmonize()` and `compound.joint_statistics()`.
The reference is the same-author deterministic calculation in Vieira Passos et al. (2024),
*Identifying regional hotspots of heatwaves, droughts, floods, and their co-occurrences*,
*Stochastic Environmental Research and Risk Assessment* 38, 3875–3893,
<https://doi.org/10.1007/s00477-024-02783-3>.
The public inputs are Vieira Passos (2024), Figshare version 2,
<https://doi.org/10.6084/m9.figshare.25189118.v2>, under CC BY 4.0.

`tests/fixture/compound/` retains source values, missingness, full coordinates, checksums, file IDs, and the adapted code's MIT notice.
The reference pins author commit `6c88f255ee3034279dcebe2a2fc40c954dbe9575`.
The preparation uses month-end forward filling, followed by linear interpolation of remaining daily gaps, outside the production harmonizer.
The API receives declared Boolean indicators with `HWI >= 0` and `SPEI <= -1`, separate validity, and recorded comparators.
The original source Calibration Period is unknown.

The comparison covers June, July, and August separately over 1922–2021 at the three selected Swedish cells.
The measured run gave exact counts and zero maximum absolute error for joint probability, independence product, and LMF across all nine cases.
Counts require exact equality; floating-point comparisons use `rtol=0`, `atol=1e-12`.
These measured errors do not establish cross-platform bitwise identity.

| Cell | Month | Eligible days | Heat days | Drought days | Joint days | LMF, rounded |
| --- | --- | --- | --- | --- | --- | --- |
| Southern | June | 3,000 | 232 | 566 | 43 | 0.982393079079 |
| Southern | July | 3,100 | 530 | 469 | 68 | 0.848050850867 |
| Southern | August | 3,100 | 303 | 590 | 63 | 1.092465178721 |
| Central | June | 3,000 | 178 | 509 | 26 | 0.860908147723 |
| Central | July | 3,100 | 340 | 494 | 45 | 0.830554894022 |
| Central | August | 3,100 | 123 | 435 | 26 | 1.506401270909 |
| Northern | June | 3,000 | 136 | 743 | 36 | 1.068798986620 |
| Northern | July | 3,100 | 264 | 557 | 48 | 1.011914476906 |
| Northern | August | 3,100 | 66 | 523 | 19 | 1.706356104062 |

**Classification: reference-reproduced, not independently validated.**
The author preparation expands monthly drought values to days; it does not create independently observed daily drought resolution.
Inclusive source thresholds intentionally differ from the library's strict probability comparisons.
The author's zero-joint LMF is undefined, while the library returns zero with positive marginals; every empirical case has positive joint counts.
The checks preserve this intentional difference explicitly.
The nine cases do not reproduce the full Figure 8 map, its significance mask, or upstream SPEI correctness.
The author's IID significance procedure and random tie noise are excluded.
Daily inference remains unsupported, so this comparison makes no new bootstrap-significance claim.

| Route | Final evidence scope for this comparison |
| --- | --- |
| Prepared Boolean indicators and joint statistics | Same-author reference reproduction, restricted to the nine cases above. General harmonization remains specification-level. |
| Conditional statistics and event matching | Analytic/specification-level regression; the LMF reference does not validate these routes. |
| Monthly bootstrap | Synthetic characterization and regression only, with the limits below. Further calibration belongs to #1398. |
| Beta xarray routes and catalogs | Interface equivalence with the NumPy core, not an independent scientific oracle. |

The runnable [compound guide](docs/compound-hazards.md) keeps the reproduction and monthly synthetic inference examples separate.
No fixture download, missing-fixture skip, or optional external package substitutes for this reproduction gate.

## Compound Probability Route and Candidate Event Product

There is **no external oracle** for the declared-probability route or general
compound event catalog. The Swedish summer reproduction above uses declared
indicators and the source's inclusive thresholds, not this route's probability
thresholds. A synthetic integration case in `tests/test_compound_reference_fixture.py`
uses 1–5 June 2000 with strict thresholds `heat > 0.8` and `drought < 0.2`.
Values equal to a threshold do not count as events for that hazard, but those days remain eligible.
Only the day with a missing drought probability is ineligible.
Four eligible days give exact marginal counts 2 and 1, one joint day,
dimensionless joint probability `1/4`, and LMF `2`. These are hand counts,
not real-world event rates, tests of probability calibration, or significance.

The [Dheed database](https://essd.copernicus.org/articles/17/6621/2025/) is an
independent candidate, not an adopted oracle. Section 3.2 reports 26,351 labelled
ERA5 dry–hot events for 1970–2023. Sections 2–3 use local 1% daily rank tails,
30-, 90-, and 180-day dry indicators, at least three consecutive days, and
connected components across longitude, latitude, and time. The library counts
per-cell pairs or dates, not spatially connected events. A point comparison would
need the same daily flags, missing masks, and duration rule. Comparing the
published global event count directly would test different quantities.

## Compound Event Catalog Evidence

`compound.compound_events_xarray()` has regression and interface-equivalence evidence
only. `tests/test_compound_catalog.py` compares each event field, timestamp, censor
flag, support count, retained-run count, pair count, and rejected-bridge count exactly
with the existing NumPy matcher. The cases cover daily and monthly inputs,
concurrent and sequential modes, bounded and unbounded lags, duration filtering,
transposed spatial dimensions, one series, and an irrelevant third hazard.
The tests also cover shared missingness, all-invalid cells, no matches, zero-cell
subsets, rejected malformed inputs, and Dask rejection under a scheduler that fails
if any task runs. It also checks the 1,000,000-row cap, a duplicate hazard label, a
non-finite manifest, and dropped boundary attributes on the cell coordinates. Empty and
populated catalogs preserve values, cell labels, flags,
and versioned JSON metadata through h5netcdf and Zarr storage.

These tests do not use the `validation` marker. They establish no new scientific
validation, external oracle, significance calibration, or causal interpretation.
Pair counts are not independent observations. The beta catalog is explicitly eager;
its memory use and runtime have not been characterized. A selected input can be large
enough to exhaust memory. A match that would exceed the 1,000,000-row catalog cap raises
`InvalidArgumentError` before the rows that would overflow it are allocated. The final
public-study reproduction above concerns frequency counts, not event catalogs.

## Compound Bootstrap Characterization

### The #1371 experiment

The synthetic experiment predeclared a familywise error budget of 0.01 across
864 binomial comparisons. Each coverage case required at least 109 covered
records out of 128. Each independent-null case allowed at most 19 rejections
at nominal alpha 0.05. Missing intervals counted as uncovered; missing p-values
failed the null gate. Wilson 95% intervals accompany all measured proportions.

The default six-year blocks passed these broad screening checks. Coverage
counts ranged from 111 to 125 out of 128 across the default cases. This is not
proof of nominal 95% coverage. The twelve-year LMF interval for February in the
independent 36-month moving-average scenario covered only 106 of 128 truths.
Its estimated coverage was 82.8%, with a Wilson 95% interval of 75.3% to 88.4%.
The characterization therefore failed. The case is retained as failed: it was
not discarded and the acceptance band was not widened.

Block-length changes also changed individual test decisions. The report retains
those counts and every failed case.

The reported run follows a fix to p-value tie handling: the first run compared
nulls to the observed statistic exactly, so floating-point rounding dropped some
ties. The rerun used the same seeds and configuration. Interval coverage is
identical, null rejection counts changed by one in 4 of 36 cases, and no gate
outcome changed.

### #1398 measurements (2026-10-19)

[Issue #1398](https://github.com/monocongo/climate_indices/issues/1398) measured 183
cases: the four #1371 scenarios again, plus record lengths of 30, 60, and 100 years,
persistence of 12, 18, 24, and 30 months, conditional lag windows `(0, 1)`, `(0, 3)`,
and `(1, 3)`, monthly frequencies of 0.20 to 0.30, 0.30 to 0.60, and 0.70 to 0.80,
hazards correlated 0.5 at 1, 6, and 12 months of persistence, and the default 1,999
resamples. The gates, bands, and scenario plan were committed before the run, and the
replication group reproduces the #1371 report field by field. The predeclared design,
results, and decision are in `docs/design/compound-bootstrap-calibration.md`.

47 of the 183 cases fail a predeclared gate. The failures fall into three mechanisms.
37 cases miss the interval coverage band, and 45 cases carry a month where the library
suppressed the p-value because fewer than 95% of resamples were valid; 35 cases do both.
**No independent-hazard case exceeded its per-month rejection band.** The pooled
rejection rate across the 24 independent-hazard 200-year scenarios is 3.5% to 7.5%.
At the default 1,999 resamples the 36-month independent scenario passes all nine gate
checks and its pooled rejection rate is 3.5% (3.2% to 3.8%), so most of the 7.5% excess
seen at 399 resamples was a resample-count artifact. Under-coverage is the remaining
limit, and it appears where few effective blocks or few effective occurrences remain.
Coverage passes at 100 years and fails at 30 and 60 years; it passes for 12, 18, 24,
and 30 months of persistence and fails at 36; it passes for monthly probabilities from
0.30 upward and fails near 0.20.

### Supported scope (maintainer decision, 2026-10-19)

This statement supersedes the narrower 2026-10-18 one. The maintainer kept the method
unchanged, added no support guard because no observable separates the passing from the
failing cases without refusing supported ones, and widened the claim only where every
predeclared gate passed:

- **Supported**, listed group by group because the plan crossed none of these
  conditions with each other:
  - At the default 1,999 resamples: 200-year monthly records with the seasonal
    frequencies (0.35 to 0.55) and source overlap of 1, 6, or 36 months, hazards
    independent of each other. The independent 36-month case passes every gate here.
  - At 399 resamples: 200-year monthly records with source overlap of 1 to 30 months;
    conditional lag windows `(0, 1)`, `(0, 3)`, and `(1, 3)` at 1- and 6-month overlap;
    monthly frequencies of 0.30 to 0.60 at 1- and 6-month overlap and 0.70 to 0.80 at 1-,
    6-, and 36-month overlap; hazards correlated 0.5 at the same time with 1, 6, or 12
    months of overlap, where rejection describes power rather than null calibration; and
    100-year records with the seasonal frequencies at 1- and 6-month overlap.
  - Every listed condition passed every gate in its group. Measured coverage is 91.5% to
    94.8%. This is screening evidence, not proof of nominal 95% coverage.
- **Unsupported.** Sources that persist for 36 months: with independent hazards the
  twelve-year LMF coverage gate fails at 399 resamples and passes at the default
  resamples, and with correlated hazards the default-resample case fails one coverage
  gate (3-year blocks, month 4, 107 of 128). Monthly probabilities near
  0.20, where the null p-value is suppressed on too many resamples and coverage also
  fails at 36-month persistence. Records of 60 years, which miss coverage in months 6
  to 8 (94 to 108 of 128) and suppress p-values in months 6 to 9. Records of 30 years,
  where the guards remove 23% of month-records at typical frequencies and 63% near
  0.20, and the answered records fail both gates. Profile `wide` at 36-month overlap,
  which was not run. Do not read an interval or p-value from these cases as nominal.
- **Never claimed.** No external validation, daily inference, field significance,
  nonstationarity, or long memory. The guards report refusal through `quality_flags`
  and effective resample counts; they do not detect the source structure itself.

The beta xarray adapter is #1372, with equivalence-only evidence recorded in the
compound xarray row above. The [worked guide](docs/compound-hazards.md) preserves these
limits. Improving calibration for 36-month persistence or 60-year records needs a
different method or support contract, and a reopen would need its own predeclared
characterization.

## Backend Parity and Performance

`tests/test_native_index_edge_parity.py` compares monthly SPI, SPEI, and EDDI
at Timescale 1. It uses synthetic near-constant input (rain near 40 mm with
variation of order `1e-3` mm), 8 missing years in a 12-year sample, four complete
years, and one year with insufficient data for each calendar fit or rank.
It checks Gamma for SPI and SPEI, plus Pearson for the near-constant case.
The test requires the recorded calls to the appropriate Rust fitting or ranking
and CDF kernels. The comparison run disables native dispatch. The test checks
identical NaN masks and `rtol=atol=1e-10`, including finite scores in supported
cases and all-NaN output for the one-year case. On macOS arm64 the maximum
observed finite difference was `7.55e-15`. This is backend agreement only.
`test-native` builds the extension and collects these unmarked tests in its
core suite with `CLIMATE_INDICES_REQUIRE_NATIVE=1`. Python-only runs skip them.

The committed [RUST-011 measurements](benchmarks/README.md#rust-kernels-vs-the-python-reference-across-the-parity-registry-rust-011)
used a 35-year synthetic monthly station series and a cell-count sweep on macOS
27.0.1 arm64, Python 3.14.7. They report the best of five after warm-up with
alternating backend order. The 1.06× and 1.78× figures are respectively 32-cell
Gamma SPI and two-cell EDDI results, not a general range for fitting indices.
EDDI ranks observations rather than fitting a distribution. The original
record is `benchmarks/results/rust_vs_python.txt`. On an Apple M5 with macOS
27.2.1 and Python 3.14.7, the same harness gave 1.08× and 1.94× at those two
sizes (`--repeat 5 --skip-threads --skip-dask`), with single-series ratios
of 1.36× for Gamma SPI, 3.18× for Pearson SPI, 1.41× for Gamma SPEI, and
1.51× for EDDI. These are compute timings, not I/O or operational throughput.

Penman–Monteith has Rust kernels for the final equation and derived inputs.
The public path can use them for eligible float64 arrays under the native
floating-point policy. Scalar-only calls, float32 inputs, and the default
NumPy error policy retain the Python path. `tests/test_native_pet_parity.py`
checks dispatch as well as numerical parity. The cited harness measured 0.88×
for its original daily workload and 1.12× on the rerun; neither is a speed guarantee.

## Flood Evidence Classification

The flood family — effective precipitation (PE), the Effective Drought Index
(EDI), the Flood Index (I_F), and the Antecedent Precipitation Index (API) — has
no matched observed-input numerical oracle or measured flood-prediction skill.
PE, EDI, and I_F have specification and regression coverage only. API also has
an independent-implementation comparison on synthetic inputs.

- **PE and EDI**: Byun & Wilhite (1999), section 6a, Figure 2, and Table 5 describe
  the 1995–1996 Hickman example. The paper uses daily precipitation, a variable
  summation duration, and a 30-year calendar-day baseline with five-day smoothing.
  [ADR-0014](docs/adr/0014-flood-family-scientific-conventions.md) selects a fixed
  365-day harmonic window. `tidyindex` and `rrk4910/EDI` also differ in weighting
  or standardization; neither is a matched oracle.
- **Hickman input audit (#1135)**: The [NCEI GHCN-Daily file](https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00253825.dly)
  for `USC00253825` contains all months from 1960–1996. The file retrieved on
  2026-11-22 had SHA-256 `db4bb8219a83d2925eb02d222193305f0157b677d7e293b715e937a52ee345f9`.
  Parsing civil days from `PRCP` (tenths of mm) gives 68 missing (`-9999`) days
  and no nonblank quality flags. The missing days include 2–3 March 1960 and
  1 February 1996. This is a station candidate, not the verified processed input.
  The [HPRCC](https://hprcc.unl.edu/) public site did not yield the paper's frozen
  193-station archive or its 113-station retained subset. The exact station identity,
  nearest-station substitutions within 100 km, calendar-day fills, and exact
  calibration and warm-up inputs remain unconfirmed. No reconstructed series or
  digitized figure is committed; reuse permission for a figure derivative is unresolved.
- **I_F**: The [Deo et al. (2015) abstract](https://doi.org/10.1007/s11269-015-1046-3)
  specifies exponential depletion, not the selected harmonic kernel. The full text
  requires subscription access. Neither matching daily input nor the exact event
  estimator is available. No streamflow or event-skill comparison was undertaken.
- **API**: `tests/fixture/flood/api.csv` matches the R `ahrapi` 0.1.0 vector example
  at `k=0.95` with zero initial state, no spin-up, and complete daily inputs.
  This checks recurrence agreement, not agreement with observed wetness or flood outcomes.
  **Independence:** Bartram supplied the synthetic inputs and the R implementation
  generated the expected outputs. Both implementations use the Kohler and Linsley
  recurrence. Author, institution, or shared code overlap is unknown.
- **Digitized figure**: the Byun & Wilhite Figure 2 index panels are
  supplementary lower-tier evidence the survey recommends only behind the
  tabulated values; no figure has been digitized.
- **Specification and regression checks**: In addition to the API comparison,
  `tests/test_flood_reference.py` (marked `validation`) pins the Byun & Wilhite (1999) Eq. (2) two-day
  identity, the harmonic endpoint weights `w₁ = H_D` and `w_D = 1 / D`, and the
  `P / (1 − k)` constant-input limit, and freezes the public signatures. The
  per-index contract suites, run outside the `validation` marker, cover the
  Eq. (3) API recursion, the fixed-window EDI algebra, annual-maximum
  normalization, missing data, and Spatial Block layouts, and
  `tests/test_flood_xarray.py` checks NumPy/eager/Dask equivalence for the
  DataArray index APIs, stable from 3.1.0.

Even the tabulated Table 5, if it is eventually committed, would show
reproduction of the originating group's own arithmetic rather than independent
correctness of the method — the same distinction the scPDSI entry draws between
independent-implementation cross-validation and external-product validation.
`tests/fixture/flood/README.md` lists the retrieval work each index needs before
a numeric oracle can be committed. Flood indices describe flood potential,
not flooding: they are wetness measures upstream of terrain, soils, and
routing, so even a validated value would not be a flood prediction.

## Energy Evidence Classification

The energy family — heating, cooling, and energy degree days, the hub-height wind
profile, the wind and PV capacity factors, SREPI, the standardized demand
index, SRLI, and the energy-drought event statistics — adopts two classes of
external anchor and no product-level oracle:

- **Exact reference**: NOAA CPC's published seven-day CDD example is transcribed
  in `tests/fixture/energy_cpc/` with its retrieval date and source checksum. It
  is authoritative for that one example, not for the C3S threshold/reference
  scheme or for a default base.
- **Independent implementation**: R `SEI` 0.2.0 pins the empirical
  standardization on a real, attributed ERA5-derived capacity-factor series;
  `xclim` 0.62.0 and `windpowerlib` 0.2.2 pin the wind profile; `pvlib` 0.15.0
  pins the PV power/temperature term. These compare implementations on identical
  inputs, not skill against observations. The empirical seam implements `SEI`'s
  plotting position, tie rule, and minimum count by design, so its close
  agreement is expected rather than an independent check of the method.
- **Not adopted**: the C3S energy-indicator product. Its degree days are
  national monthly CSV, not a matched gridded series, and its capacity factors
  use turbine-specific curves and configurations the library does not reproduce.
  `docs/research/energy-oracle-survey.md` records the missing matched inputs; no
  C3S numeric comparison runs.
- **Not adopted**: beta and the energy-drought event statistics. The `SEI`
  fixture has a finite-bound control on `SEI`'s censored empirical PIT, not a
  beta reference distribution, so the library's beta route has no external
  reference; `SEI`'s event loop uses different missing-value and magnitude
  conventions.
- **Regression only**: the wind capacity factor, the event statistics, and the
  Jerez thermal model and clipping policy. There is **no external oracle** for
  the generic wind curve or the event catalog. The [C3S Wind Power fact sheet](https://confluence.ecmwf.int/pages/viewpage.action?pageId=327655847)
  documents dimensionless onshore and offshore capacity factors on a 0.25° hourly
  ERA5 grid from 1979, with 100 m hubs and distinct representative turbines.
  A matched comparison needs the source hourly wind and exact turbine curves;
  the product is modeled potential, not observed generation.
- A synthetic four-hour check in `tests/test_energy_capacity_factor.py` applies
  the generic curve to `[3.5, 13, 25, NaN]` m/s with no density correction. The
  cut-in, rated, and cut-out points give `[0, 1, 0, NaN]`. The mean of the three
  valid capacity factors is `1/3`; applying the curve to mean wind gives `1`.
  The test uses exact boundary rules and the binary64 result of the fraction.
  This is specification evidence for application order, not a turbine-data comparison.
- **Specification evidence only**: hourly standardization (`#1177`) ranks each
  hour of day x day of year stratum against itself, and its tests check the plotting
  position, the calendar plan, and agreement between the NumPy and xarray paths.
  There is no independent hourly oracle, because `SEI` accepts hourly timestamps
  but does not stratify by hour of day or day of year. The ERA5/C3S gridded
  demonstration remains pending (`#1179`). The pooled seasonal-diurnal option
  (`#1407`) is specification evidence for the same reason: its tests pin the pooled
  plotting position against an explicit window reference, the 29 February holdout,
  the year-boundary wrap, the unchanged default, and the finer tail resolution on a
  synthetic hourly series.

Energy indices describe the climate-driven component of energy supply and
demand, not actual generation, load, or residual load. Even an exactly
reproduced standardized index would not be a generation or demand forecast.
`tests/fixture/energy_cpc/`, `tests/fixture/energy_sei/`, and
`tests/fixture/energy_wind/` record that maintainer confirmation of fixture
transcription or generation and redistribution is still outstanding.

## Reliability Evidence Classification

The reliability family — SAIFI, SAIDI, CAIDI, MAIFI_E, MAIFI, ASAI, and the
outage counts — is **specification-and-regression evidence only**. No
event-level external oracle was adopted, so nothing here is independent
scientific validation.

The reported arithmetic is reproduced against hand-computed analytic cases.
The committed fixture under `tests/fixture/reliability/` is five synthetic
records derived from the ADR-0026 decision-10 table, not downloaded data. Its
`provenance.json` validates against `tests/fixture/provenance_schema.json`,
records the SHA-256 checksum of `events.json`, and sets the validation
tolerance to `rtol=0, atol=0` because the expected values are exact rational
arithmetic. That zero tolerance is honest for the analytic cases only; it is
not a claim that the indices agree with a utility's reports.

No independent implementation and no product oracle are adopted. EIA Form
EIA-861 Table 11.1 is aggregated national reporting, so it can validate
relationships between indices rather than arithmetic on an event log; it is
cited as context, not compared numerically. This is the same distinction
`VALIDATION.md` draws elsewhere between reproducing a source's own arithmetic
and validating a method against independent data.

What the tests check: the ADR-0026 analytic cases, the sustained/momentary
split, the additive `CI` and `CMI` accumulators, the threshold and
planned-outage filters, ASAI exposure and its one-step rounding tolerance,
leap-aware and daylight-saving-aware elapsed hours, the grouping rules, the
field contract, and schema and checksum validation of the fixture. See
`tests/test_reliability_indices.py`, `tests/test_reliability_events.py`,
`tests/test_reliability_grouping.py`, and `tests/test_reliability_fixture.py`.

**Major Event Day detection and filtering: specification-and-regression
evidence, with one published worked-example reproduction; verified against
IEEE Std 1366-2012, not 2022.** `tests/test_reliability_major_events.py`
checks `major_event_day_threshold()` against the hand-computed history
`[0, e^-1, 1, e, 0]`, whose threshold is `exp(2.5)` within `rel=1e-15`.
It checks the strict `>` boundary at `T_MED` and one float64 step below it,
without a tolerance, and the history validation matrix of ADR-0026 decision 7.
It also checks filtered results against an unfiltered call on the manually
filtered log through the shared metric path. It reproduces the IEEE Std
1366-2012 clause 3.5.1 example from the public CPUC copy. Its printed rounded
`alpha` matches the mean of the log-SAIDI values and `beta` matches their
sample standard deviation. January 28 is the only
Major Event Day, and SAIDI is 287.348 with all events and 49.855 without it
(printed 287.35 and 49.86). The printed `T_MED` of 66.69 comes from rounded
intermediates; full precision gives 67.10, so the example is not a
threshold oracle at full precision. The IEEE Std 1366-2022 text was not
obtained, so 2022 conformance is unverified. The sample standard deviation,
the validation rules, and the unchanged ASAI exposure after filtering are
project conventions, not verified IEEE requirements. No utility's Major Event
Day determinations are reproduced, and the Gaussian assumption of the method
is not tested.

Known gaps, stated without overstating coverage: Major Event Day rules are
not verified against the 2022 edition, and the library does not build a
daily-SAIDI history from a multi-year event log; network topology and customer connectivity are not resolved;
and the xarray adapter and CLI integration are unimplemented. The seeded Monte
Carlo path is implemented; `tests/test_reliability_simulation.py` checks it
against a single-component analytic reference for both arrangements and a
two-component series reference, not an external oracle.
Several conventions vary between utilities — the sustained/momentary
threshold, planned-outage inclusion, the MAIFI flavour, the IEEE momentary
pre-sustained exclusion, Major Event Day filtering, and partial-period
annualization — and the library
fixes a documented default for each rather than inferring a jurisdiction. See [the reliability algorithm reference](docs/algorithm_refs/reliability.md) and
[ADR-0026](docs/adr/0026-reliability-module-api-and-conventions.md).

Reliability indices describe how often and how long service was interrupted
under past conditions. They are not resilience metrics, and they are not
forecasts. The EPRI Climate READi forward-looking metrics (CORi, LORi, DORi)
need fragility curves and hazard event probabilities that this library does
not model, so they are out of scope rather than pending.

## EDDI Fixture Policy

The paired NOAA PSL EDDI fixtures are committed and run in CI. They contain
monthly reference ET and 1-, 3-, and 6-month EDDI for latitude 39.75–39.875
and longitude -105.0–-104.875, with the 1979–2023 table baseline.

Required fixture layout:

```text
tests/fixture/noaa-eddi-1month/
tests/fixture/noaa-eddi-3month/
tests/fixture/noaa-eddi-6month/
```

Each directory must contain:

- `provenance.json`
- `metadata.json`
- `pet_input.npy`
- `eddi_reference.npy`

The provenance file follows `tests/fixture/provenance_schema.json`, cites the
NOAA PSL EDDI time-series source, records checksums, and states the calibration
period. The validation tests compare non-NaN reference values and separately
verify NaN placement. `scripts/prepare_noaa_eddi_fixtures.py` refuses an
unexpected table baseline so fixture refreshes are intentional.

## SPEI Plausibility Classification

`indices.spei()` (gamma distribution, Thornthwaite PET) is compared against
grid-cell-mean CSIC SPEIbase v2.11 series in `tests/test_speibase_reference.py`,
for three CONUS climate divisions spanning an aridity gradient (humid Alabama
0101, subhumid Oklahoma 3405, arid southwest Arizona 0205) and timescales 1,
3, 6 and 12 months. The stored reference is the mean of the per-cell SPEI
values, not an index computed from averaged inputs.

This comparison is a **plausibility check, not external numerical
validation**, and is deliberately not enforced with `atol`/`rtol`. Three
confounds separate the two products:

- SPEIbase v2.11 uses FAO-56 Penman-Monteith PET from CRU TS 4.09, while the
  compared series uses Thornthwaite PET computed from the committed nClimDiv
  temperatures; the two PET families diverge with climate aridity (van der
  Schrier et al. 2011).
- SPEIbase standardizes with the log-logistic distribution, while this
  gamma/Thornthwaite comparison still fits gamma. The separate input-matched
  log-logistic comparison below removes this confound.
- The reference is a 0.5-degree grid average inside a climate division polygon
  while the compared series is the division's station-derived areal average,
  from different precipitation inputs (CRU TS vs. nClimDiv).

The tests therefore assert per-division, per-timescale floors on correlation,
sign agreement, and drought-category agreement, derived from the measurements
recorded in `tests/fixture/speibase/provenance.json`;
`scripts/prepare_speibase_fixtures.py` re-measures those values from the arrays
it just built, refusing to publish a refresh that drifts beyond `0.001` of the
recorded expectations, and regenerates the floors from the new measurements;
`test_floors_keep_documented_slack` pins every floor to within a documented
band below its measurement so a floor cannot be widened to hide a regression.
Measured correlation is weakest in the arid Arizona division (0.82-0.86 across
timescales) -- the direction the PET-family mismatch predicts -- and strongest
in subhumid Oklahoma (0.95-0.97), with humid Alabama (0.92-0.94) between them.
With only three divisions the ordering cannot isolate the PET-family confound
from the distribution and precipitation-support confounds, which is precisely
why no tight numerical gate is defensible: it would either fail on that
legitimate climate-dependent bias or have to be widened until it no longer
tested anything.

The compared climate_indices series is calibrated against the full 1901-2022
period of record, matching the committed inputs, while SPEIbase v2.11 is
standardized on its own 1901-2024 record; the fixtures retain the 1901-2022
overlap. The dataset survey behind this comparison lives at
`docs/research/spei-dataset-survey.md` on branch `research/spei-dataset-survey`.

## SPEI Log-Logistic Like-for-Like Comparison

`tests/test_speibase_like_for_like.py` removes the three confounds above by
feeding `indices.spei()` the CRU TS 4.09 monthly precipitation and FAO-56
Penman-Monteith PET that SPEIbase v2.11 was computed from, committed as
per-cell fixtures in `tests/fixture/speibase_cru_ts/`. It standardizes each
0.5-degree cell with the log-logistic (generalized logistic) distribution over
1901-2024 and averages the per-cell series inside the same three climate
divisions, matching SPEIbase's `R/functions.R` (`spei.nc`) per-cell-then-mean
pipeline.

This is a cross-implementation check of a port, not of an algorithmically
independent estimator: `climate_indices.lmoments.fit_glo` is a Python
translation of the same `lmom` PELGLO routine R `SPEI` calls, using the same
unbiased-PWM L-moments, so close agreement is expected.

The agreement is near-exact -- correlation 0.9996-0.99999 and mean absolute
difference 0.003-0.016 across divisions and timescales -- and the tests assert
tight per-series floors (correlation >= 0.97) instead of the loose plausibility
floors. The residual is small and dominated by a PET day-length convention
difference, not by the fit: this test multiplies CRU TS PET (mm/day) by the
leap-aware calendar month lengths, matching SPEIbase's public `R/functions.R`
(`etp * ndays` with `Hmisc::monthDays`), while a near-uniform month length
reproduces the committed SPEIbase v2.11 grids two to four times more closely,
so v2.11's effective generation convention could not be confirmed from public
code. Floors are derived from
`tests/fixture/speibase_cru_ts/provenance.json`, and
`scripts/prepare_speibase_cru_ts_inputs.py` re-measures the agreement from the
arrays it builds, refusing to publish a refresh that drifts beyond `0.001` of
the recorded expectations; `test_floors_keep_documented_slack` pins every floor
to a documented band below its measurement.

The earlier gamma/Thornthwaite comparison is retained because it exercises a
different distribution and PET path against the same reference; it remains a
plausibility check, not numerical validation.

## SPEI Log-Logistic Cross-Implementation

`indices.spei()` with `Distribution.loglogistic` is compared against the R
`SPEI` package (`distribution = "log-Logistic"`, `fit = "ub-pwm"`) in
`tests/test_spei_loglogistic_reference.py`, over the same three CONUS climate
divisions and timescales 1/3/6/12 as the SPEIbase comparison and over a short,
highly skewed synthetic series with one constant calendar month. The committed
R output lives under `tests/fixture/spei_loglogistic/`; `provenance.json`
records the pinned R package versions, the fixture checksum, and the
tolerances, and `scripts/prepare_spei_loglogistic_fixtures.py` refuses a refresh
from a different `SPEI`/`TLMoments`/`lmom` version.

This is a **cross-implementation check of a port**, not a plausibility check and not an algorithmically independent one:
`SPEI` is the reference implementation of the log-logistic SPEI and the
distribution the CSIC SPEIbase is built from (Vicente-Serrano et al., 2010;
Beguería et al., 2014). Both sides receive the identical `(P − PET) + 1000`
water balance -- the offset `indices.spei` applies internally -- so the
comparison isolates the distribution fit from the scaling and offsetting steps,
and `SPEI` fits each calendar step with the same unbiased-PWM L-moments
estimator (`TLMoments::PWM` + `lmom::pelglo`) that `lmoments.fit_glo`
implements. The tolerances (`series_atol = 1e-6`; `parameter_rtol = 1e-8`,
`parameter_atol = 1e-6`) therefore absorb only cross-platform numerical wobble,
not algorithmic differences: measured agreement is at ~1e-12 for the
standardized series and ≤~1e-10 for the fitted parameters. Together the
fixtures span the edge cases the ticket names -- a short sample and extreme
L-skewness (GLO shape to about -0.8) from the synthetic series, a negative
underlying location parameter from the real arid divisions, and a constant
calendar step that is unfittable at scale 1 (R reports NA, this library a
zeroed scale and NaN values; at scale 6 the constant month enters non-constant
multi-month sums and is fittable). `SPEI` does not clip its output to
`[-3.09, 3.09]`, so the comparison covers finite values inside that range and
asserts this library's clip on the out-of-range tail separately (the synthetic
scale-1 fit reports one value of `-inf`).

## Palmer Validation Classification

The Palmer evidence splits by product family, because the two consume different
reference artefacts.

**Standard Palmer (`pdsi()`: PDSI, PHDI, PMDI, Z-Index)** is qualified
independent external-product validation against the operational NOAA NCEI
nClimDiv arrays in `tests/fixture/nclimdiv/`, enforced by
`tests/test_nclimdiv_reference.py::test_pdsi_vs_noaa_nclimdiv_qualified_external_validation`.
Measured over 528,384 division-months, `pdsi()` agrees with nClimDiv to a median
absolute difference of 0.0127, with 86.2% of months within 0.05. The
qualification is deliberate and recorded rather than averaged away: NCEI
publishes two decimal places, and the committed `precips.npy`/`pet.npy` inputs
are not byte-identical to NCEI's operational inputs, so the practical agreement
floor is near 0.013 rather than the 0.005 that published precision alone would
permit. The library-generated standard-Palmer fixtures (`tests/fixture/palmer/`)
remain in the suite as regression coverage only -- their provenance identifies
the source as "climate_indices library reference output" -- and they are no
longer the family's only evidence.

**scPDSI (`scpdsi()`)** has independent implementation cross-validation, not
external-product validation: `tests/test_scpdsi.py` compares the four
self-calibrating outputs and fitted duration factors against the Wells-lineage
C++ oracle fixtures at `atol=5e-5`, `rtol=0`. The nClimDiv comparison in
`tests/test_nclimdiv_reference.py::test_scpdsi_vs_noaa_nclimdiv_characterization`
stays explicitly classified as characterization, because nClimDiv uses standard
Palmer's fixed national K-factors while `scpdsi()` self-calibrates per division.
The two recursion lineages disagree on roughly 6.3% of months by up to 8 PDSI
units (the NCEI-lineage values are closer to NOAA operational data on 88% of
those months; see `tests/fixture/palmer/provenance.json`), so neither series can
serve as the other's tolerance oracle.

**Literature worked examples (Palmer 1965).** `tests/test_palmer_literature.py` checks each
computation stage against numbers printed in Research Paper No. 45, a source older than every
software lineage. One-month water-balance steps from the printed state agree within one print
unit (0.01 in) except three months (1933-08, 1934-05, 1934-07; at most 0.0201 in); CAFEC
coefficients follow from Table 2's rounded means in 91 of 92 testable cases (western Kansas May
beta is 0.0007 outside the rounding interval); and K normalization, P-hat, d and Z agree within
propagated print rounding. Equation 26 for K' is the exception: central Iowa December measures
0.0158 against a bound of 0.0131, and its cause is not established. The paper's printing errors
are itemized in `tests/fixture/palmer_literature/README.md`. The pdi.f-lineage recursion used by
`pdsi()` reproduces Table 12's printed X within 0.02 in 32 of 40 months. The other eight follow
from three arithmetic inconsistencies in the print, and repairing them from the print's own
arithmetic brings every month within 0.02. The Wells recursion
used by `scpdsi()` differs from the printed X in 7 further months where `pdi.f` agrees, all from
its 0.5 establishment threshold where Palmer's example and `pdi.f` use 1. This is stage-level evidence on printed
excerpts, not a full-series validation, and it does not change the classification above.

No further standard-Palmer external reference is required for this validation
contract. scPDSI external-product validation remains open pending the DRI/WRCC
assessment in issue #780.

## scPDSI Calibration-Anchor Measurement

Wells' self-calibration methodology tunes each climate division's duration
and K-prime factors so that, by construction, the calibration-period 2nd and
98th percentiles of the resulting scPDSI series land at exactly -4.0000 and
+4.0000. (`scpdsi()` takes those percentiles of the PDSI output and rescales
the Z-index to match; the Z-index is the quantity adjusted, not the quantity
anchored.) Measuring how closely `scpdsi()` reproduces that anchor across all
344 climate divisions (calibration period 1931-1990) verifies the
self-calibration percentile-scaling step independently of the NOAA nClimDiv
comparison in `tests/test_nclimdiv_reference.py` (see the "Per-Index Evidence"
table above), which instead measures agreement with an external product that
uses different (fixed, national) K-factors.

| Percentile anchor | Median | Mean | Median deviation | Max deviation | Within +/-0.05 | Within +/-0.25 |
| --- | --- | --- | --- | --- | --- | --- |
| 2nd (target -4.0000) | -4.0000 | -4.0142 | 0.0102 | 0.9730 | 74% | 93% |
| 98th (target +4.0000) | +4.0000 | +3.9829 | 0.0130 | 1.5421 | 70% | 91% |

The medians land on the target exactly while the maxima do not, because
`scpdsi()` applies a fixed three rescaling passes rather than iterating to a
fixed point. Divisions whose recursion has not settled within three passes
retain the residual deviations shown above; this is the reference behaviour,
not a defect. These figures are regenerated and bounded by
`test_scpdsi_calibration_anchor_lands_on_target` in
`tests/test_nclimdiv_reference.py`, so they fail a test rather than rotting
silently if the rescaling changes.

This is regression coverage over the committed fixture set, not independent
scientific validation: it confirms the self-calibration percentile-scaling
arithmetic behaves as Wells describes for these 344 divisions, not that the
resulting scPDSI values themselves match an external authoritative source.
See "Palmer Validation Classification" above for the latter.

## Cross-Provider Characterization: Earth Engine GRIDMET Drought

`climate_indices` was compared with the precomputed `GRIDMET/DROUGHT` collection
in the Earth Engine Data Catalog over the Apalachicola-Chattahoochee-Flint basin
bounding box, 1981-2024, on a 1981-2016 calibration window. The full method
alignment, per-scale metrics, difference maps and open questions are in
`docs/research/earthengine-climate-engine-pilot.md`; the numbers are produced by
`scripts/earthengine_climate_engine_pilot.py`.

**Evidence level: characterization comparison, not reference reproduction.**
The reference is a different implementation with a different accumulation
window (fixed trailing days, 5-day cadence, versus calendar months here). Its
SPI and SPEI plotting position is unconfirmed; the catalog endpoints are
consistent with Tukey, not with the SEI position used here. No status in the
"Per-Index Evidence" table above is changed, raised or implied by this section.

The category metrics use the catalog's eleven drought-to-wet classes, including
the D0–D4 thresholds. They are index-category comparisons, not agreement with an
observed USDM map.

| Index | Scale | n | bias | RMSE | r | same class | within one class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPI | 1 mo | 3,855,984 | +0.061 | 0.369 | 0.928 | 0.638 | 0.906 |
| SPI | 3 mo | 3,855,984 | +0.054 | 0.235 | 0.974 | 0.699 | 0.961 |
| SPI | 6 mo | 3,855,984 | +0.048 | 0.194 | 0.984 | 0.730 | 0.975 |
| SPI | 12 mo | 3,855,984 | +0.049 | 0.171 | 0.990 | 0.743 | 0.984 |
| SPEI | 1 mo | 3,848,681 | +0.070 | 0.354 | 0.934 | 0.637 | 0.909 |
| SPEI | 3 mo | 3,848,681 | +0.062 | 0.224 | 0.978 | 0.703 | 0.962 |
| SPEI | 6 mo | 3,848,681 | +0.058 | 0.188 | 0.986 | 0.737 | 0.975 |
| SPEI | 12 mo | 3,855,984 | +0.062 | 0.164 | 0.991 | 0.755 | 0.988 |
| EDDI | 1 mo | 3,848,681 | -0.022 | 0.277 | 0.958 | 0.712 | 0.944 |
| EDDI | 3 mo | 3,848,681 | -0.021 | 0.170 | 0.984 | 0.802 | 0.985 |
| EDDI | 6 mo | 3,855,984 | -0.014 | 0.145 | 0.988 | 0.845 | 0.994 |
| EDDI | 12 mo | 3,848,681 | -0.026 | 0.116 | 0.993 | 0.861 | 0.997 |

One comparison month is missing for SPEI 1, 3 and 6 months and for EDDI 1, 3 and
12 months, so those rows pair 7,303 cells with 527 months rather than 528. The
saved metrics do not record which month.

The plotting-position conventions qualify these numbers.
`Distribution.empirical` uses the SEI position `(k + 1) / (n + 2)`.
For 36 distinct calibration ranks, the exact normal quantiles have a mean of +0.0538.
Tukey's positions have a mean of zero for those ranks.
Their differences depend on rank; +0.0538 is not a constant correction for each observation or the full comparison period.
The catalog endpoints suggest Tukey's convention but do not establish its formula.
The EDDI biases above are -0.014 to -0.026; no SEI correction applies to EDDI.
An alfalfa-reference-ET variant matches less closely, but that result does not establish the producer's reference crop.

The Climate Engine API (`/timeseries/standard_index/coordinates`,
`distribution=nonparametric`, same 1981-2016 Calibration Period) was compared at
three points over the same years.
Correlation is 0.9777 to 0.9945 across indices and timescales; bias is -0.052 to -0.087.
The API requests use 30/90/180/365 trailing days, not calendar-month windows.
The script samples the last daily record at most four days before each month end.
Missing longer intervals remain missing.
These point metrics and the catalog's field metrics have different spatial support.
They cannot establish which reference agrees more closely on a common sample or explain the bias.
The producer's exact rank formula, calendar grouping, reference crop, and endpoint date semantics remain open questions.

A pilot-local Tukey follow-up reduced reference A SPI/SPEI bias by 0.053831–0.053837
within the 1981–2016 Calibration Period, as the rank formulas predict. Over
1981–2024, the reduction was only 0.0222–0.0299, with residual bias +0.0235 to
+0.0400. Bias increased in 2017–2024 under the registered strict-below rank rule.
Reference A EDDI stayed unchanged. The full-period hypothesis failed its pre-registered
criteria, although the calibration-only prediction held. The completed reference B
rerun reproduced its earlier metrics within 1e-10. Over 1981–2024, the SPI/SPEI bias
range was -0.052 to -0.069 with SEI and -0.082 to -0.092 with Tukey.
Over 1981–2016, Tukey changed the bias by exactly -0.053831.
Reference B EDDI stayed unchanged. The report records the period
splits, calendar-month biases, rank rules, and image versions. This is characterization
only; no validation status changes.

## References

- Beguería, S., Vicente-Serrano, S. M., Reig, F., and Latorre, B. (2014).
  Standardized precipitation evapotranspiration index (SPEI) revisited:
  parameter fitting, evapotranspiration models, tools, datasets and drought
  monitoring. International Journal of Climatology, 34(10), 3001-3023.
  https://doi.org/10.1002/joc.3887
- Byun, H.-R., and Lee, D.-K. (2002). Defining three rainy seasons and the
  hydrological summer monsoon in Korea using Available Water Resources Index.
  Journal of the Meteorological Society of Japan, 80(1), 33-44.
  https://doi.org/10.2151/jmsj.80.33
- Byun, H.-R., and Wilhite, D. A. (1999). Objective quantification of drought
  severity and duration. Journal of Climate, 12(9), 2747-2756.
  https://doi.org/10.1175/1520-0442(1999)012%3C2747:OQODSA%3E2.0.CO;2
- Deo, R. C., Byun, H.-R., Adamowski, J. F., and Kim, D.-W. (2015). A real-time
  flood monitoring index based on daily effective precipitation and its
  application to Brisbane and Lockyer Valley flood events. Water Resources
  Management, 29(11), 4075-4093. https://doi.org/10.1007/s11269-015-1046-3
- Kohler, M. A., and Linsley, R. K. (1951). Predicting the runoff from storm
  rainfall. U.S. Weather Bureau Research Paper No. 34.
- Hobbins, M. T., Wood, A., McEvoy, D. J., Huntington, J. L., Morton, C.,
  Anderson, M., and Hain, C. (2016). The Evaporative Demand Drought Index.
  Part I. Journal of Hydrometeorology, 17, 1745-1761.
  https://doi.org/10.1175/JHM-D-15-0121.1
- Palmer, W. C. (1965). Meteorological Drought. U.S. Weather Bureau Research
  Paper No. 45. https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf
- Allen, S., and Otero, N. (2023). Standardised indices to monitor energy
  droughts. *Renewable Energy*, 217, 119206.
  https://doi.org/10.1016/j.renene.2023.119206
- Bloomfield, H., and Brayshaw, D. (2021). ERA5 derived time series of European
  aggregated surface weather variables, wind power, and solar power capacity
  factors: hourly data from 1950-2020. University of Reading.
  https://doi.org/10.17864/1947.000321
- Copernicus Climate Change Service. Global climate and energy indicators from
  1950 to present derived from reanalysis.
  https://doi.org/10.24381/3bb607bd
- Jerez, S., Tobin, I., Vautard, R., Montávez, J. P., López-Romero, J. M.,
  Thais, F., Bartok, B., Christensen, O. B., Colette, A., Déqué, M., Nikulin,
  G., Kotlarski, S., van Meijgaard, E., Teichmann, C., and Wild, M. (2015).
  The impact of climate change on photovoltaic power generation in Europe.
  *Nature Communications*, 6, 10014. https://doi.org/10.1038/ncomms10014
- NOAA Climate Prediction Center. Degree days explained.
  https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days/ddayexp.shtml
- NOAA Physical Sciences Laboratory EDDI archive:
  https://downloads.psl.noaa.gov/Projects/EDDI/
- van der Schrier, G., Jones, P. D., and Briffa, K. R. (2011). The sensitivity
  of the PDSI to the Thornthwaite and Penman-Monteith parameterizations for
  potential evapotranspiration. Journal of Geophysical Research: Atmospheres,
  116, D03106. https://doi.org/10.1029/2010JD015001
- Vicente-Serrano, S. M., Beguería, S., and López-Moreno, J. I. (2010). A
  multiscalar drought index sensitive to global warming: the Standardized
  Precipitation Evapotranspiration Index. Journal of Climate, 23(7),
  1696-1718. https://doi.org/10.1175/2009JCLI2909.1
- Van Wagner, C. E., and Pickett, T. L. (1985). Equations and FORTRAN
  program for the Canadian Forest Fire Weather Index System. Canadian
  Forestry Service, Forest Technical Report 33.
  https://cfs.nrcan.gc.ca/pubwarehouse/pdfs/19973.pdf

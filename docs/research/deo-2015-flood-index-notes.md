# Deo et al. (2015) Flood Index: reading notes

Source: Deo, R. C., Byun, H.-R., Adamowski, J. F., and Kim, D.-W. (2015). A
real-time flood monitoring index based on daily effective precipitation and its
application to Brisbane and Lockyer Valley flood events. *Water Resources
Management*. DOI
[10.1007/s11269-015-1046-3](https://doi.org/10.1007/s11269-015-1046-3). Full text
read from a subscriber copy on 2026-11-23; the PDF is not committed. Equation,
table, and figure numbers below are the paper's.

## What the paper specifies

- **Kernel (Eq. 1a–d).** `P_E,i = Σ_{N=1..D} [ (Σ_{m=1..N} P_m) / N ]`, `D = 365`:
  the harmonic double sum, expanded to `P₁(1 + 1/2 + … + 1/365) + P₂(1/2 + … + 1/365)
  + … + P₃₆₅(1/365)` and approximated as `P₁ + 0.85 P₂ + 0.77 P₃ + … + 4.23×10⁻⁴ P₃₆₅`
  (the approximation is the preceding line divided by `H_365 ≈ 6.48`). This is Byun & Wilhite (1999) Eq. (2) and
  the kernel `flood.effective_precipitation` implements. The software cited is
  Byun's FORTRAN code (`http://atmos.pknu.ac.kr/~intra3/`).
- **"Exponentially decaying".** The abstract and the text after Eq. (3) call this
  weighting an "exponentially-decaying time-reduction function". It is descriptive:
  no exponential, decay constant, or half-life appears in the equations or
  elsewhere in the 19-page article (its appendices and electronic supplement
  were not seen), and Fig. 2a plots the harmonic weights. The abstract's
  wording is therefore **not** a different kernel, and the abstract-versus-body
  conflict recorded in ADR-0014 resolves in favor of the body, which is what the
  library implements.
- **Standardization (Eq. 4).** `I_F = (P_E − mean(P_E^max)) / σ(P_E^max)`, with the
  mean and standard deviation of the *yearly maximum daily* `P_E` over the
  recorded hydrological period (1915–2012). Whether `σ` is the population or
  sample deviation is not stated; the library uses `ddof=0`.
- **Events (Eqs. 5–7).** The body's onset is the first day with `I_F > 0`; the
  abstract's onset phrasing is `I_F >= 0`. Severity is the running sum of positive
  `I_F`; peak danger is the maximum; duration is the day count. The return period
  (Eqs. 8–10) uses a partial-duration series with a mean-rank estimator
  `i / (n + 1)`; the cutoff `I_F_acc ≥ 1` selects "severe" events. The library's
  event helpers keep the strict `I_F > 0` of ADR-0014 decision 5, which follows the
  surveyed same-lineage code, and do not implement a return period.
- **Calendar.** Footnote 1: leap-day rainfall is added to the 1 March value so
  every year has 365 days. The library instead keeps 366 positions per year and
  interpolates the non-leap-year 29 February (ADR-0014 decision 4, #1147). The two
  conventions differ and the paper's values cannot be reproduced exactly without
  adopting the paper's.

## Tabulated values

- Table 2 ranks Brisbane and Lockyer Valley events (onset date, `I_F^acc`,
  `I_F^max`, `D_F`, AWRI, precipitation); the headline Brisbane event is onset
  26 January 1974 with `I_F^acc` 118.1, `I_F^max` 4.4, `D_F` 104 days, and the
  second is 29 December 2010 with 61.3, 2.6, 89.
- Table 3 gives quartiles, minimum, maximum, and standard deviation of daily
  `P`, AWRI, and `I_F` for five events; Table 4 gives the share of days per
  `I_F` band; Fig. 7 tabulates monthly seasonality.

## Why no oracle is committed

Every tabulated value depends on the paper's inputs: Bureau of Meteorology
station 40094 (Harrisville, Brisbane) daily rainfall for 1915–2012, adjusted for
inhomogeneities at the 5, 10, …, 95th percentiles and gap-filled from cumulative
distributions, then run through the paper's 365-day calendar. The adjusted record
is not published with the paper, the unadjusted record would not reproduce
one-decimal table values, and the paper's calendar differs from the library's.
Fig. 2b (the 1974 `I_F` trace) and Figs. 3–4 are plotted, not tabulated.
`I_F` therefore stays specification-level: the kernel is now confirmed against
the primary source, but no numeric values can be reproduced from public inputs.
Obtaining the BoM 40094 record is the first step toward a characterization
comparable to the Hickman one for PE and EDI.

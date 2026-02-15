---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - '_bmad-output/planning-artifacts/prd.md'
  - '_bmad-output/planning-artifacts/architecture.md'
  - '_bmad-output/planning-artifacts/implementation-readiness-report-2026-02-09.md'
workflowType: 'research'
lastStep: 6
research_type: 'Technical Validation'
research_topic: 'EDDI (Evaporative Demand Drought Index) Implementation Validation'
research_goals: 'Validate PR 597 EDDI implementation against Hobbins et al. 2016 methodology, evaluate non-parametric standardization approach, assess PET method impacts, and identify gaps blocking Phase 2 integration'
user_name: 'James'
date: '2026-02-15'
web_research_enabled: true
source_verification: true
---

# EDDI Validation Technical Research

**Date:** 2026-02-15
**Author:** James
**Research Type:** Technical Validation
**Branch:** `feature/issue-414-eddi-clean`
**Status:** Comprehensive validation analysis for Phase 2 planning

---

## Executive Summary

This research validates the EDDI (Evaporative Demand Drought Index) implementation in PR #597 (branch `feature/issue-414-eddi-clean`) against the foundational Hobbins et al. 2016 methodology and identifies critical gaps blocking Phase 2 integration.

### Key Findings

**✅ Algorithm Correctness**
- Implementation correctly uses non-parametric empirical ranking with Tukey plotting positions `(r-0.33)/(N+0.33)`
- Hastings inverse normal approximation (Abramowitz & Stegun 26.2.23) matches NOAA Fortran reference exactly
- Ranking direction is correct: ascending rank for lowest PET (rank=1 for driest conditions, highest EDDI for highest PET)

**❌ Critical Blocking Gap**
- **FR-TEST-004 BLOCKED:** No NOAA reference dataset validation test exists (Architecture v1.1 Pattern 8 requirement)
- No `tests/data/reference/eddi_noaa_reference.nc` dataset
- No provenance documentation for reference data
- **This blocks PR merge and Phase 2 integration**

**⚠️ Methodological Concerns (Mitigated by Parallel Work)**
- **PET Method Sensitivity:** Current implementation accepts any PET input, but Hobbins exclusively uses Penman-Monteith FAO56
- **✅ MITIGATION IN PROGRESS:** Parallel PM FAO56 implementation (see `technical-penman-monteith.md` — Phases 1-4 roadmap complete)
  - Once `eto_penman_monteith()` is available, EDDI users will have the recommended PET method
  - Reduces priority of PET validation code from HIGH to MEDIUM (documentation-only for Phase 2A)
- No validation of whether Tukey plotting positions are optimal (Noguera 2022 recommends parametric log-logistic approach)

**📋 Documentation Gaps**
- No Hobbins et al. 2016 citation anywhere in codebase
- No `docs/algorithms.rst` EDDI entry
- Sign convention not documented (higher PET → higher EDDI → drier conditions)
- No CLI integration (Issue #414)

### Research Structure

This document adapts the standard BMAD 6-step technical research workflow for domain-specific scientific validation:

1. **Hobbins et al. 2016 Methodology Analysis** — What does the foundational paper specify?
2. **Non-Parametric Standardization Deep Dive** — Are Tukey plotting positions the correct choice?
3. **ET Method and EDDI Accuracy** — How does PET method selection impact EDDI accuracy?
4. **PR 597 Gap Analysis** — Systematic comparison against reference methodology
5. **Prioritized Recommendations** — Actionable next steps for Phase 2
6. **References and Sources** — All factual claims verified via web search

---

## 1. Hobbins et al. 2016 Methodology Analysis

### 1.1 Foundational Papers

The EDDI methodology is established in two companion papers published in June 2016:

- **Part I:** [Hobbins et al. (2016) — Linking Drought Evolution to Variations in Evaporative Demand](https://journals.ametsoc.org/view/journals/hydr/17/6/jhm-d-15-0121_1.xml)
  *Authors:* Michael T. Hobbins, Andrew W. Wood, Daniel J. McEvoy, Justin L. Huntington, Charles G. Morton, Martha C. Anderson, Christopher R. Hain
  *DOI:* 10.1175/JHM-D-15-0121.1

- **Part II:** [McEvoy et al. (2016) — CONUS-Wide Assessment Against Common Drought Indicators](https://journals.ametsoc.org/view/journals/hydr/17/6/jhm-d-15-0122_1.xml)
  *Authors:* Daniel J. McEvoy, Justin L. Huntington, Michael T. Hobbins, Andrew Wood, Charles Morton, Martha Anderson, Christopher Hain
  *DOI:* 10.1175/JHM-D-15-0122.1

### 1.2 Official NOAA Operational Specification

The operational NOAA EDDI implementation is documented in:

- **EDDI User Guide v1.0** ([PDF](https://psl.noaa.gov/eddi/pdf/EDDI_UserGuide_v1.0.pdf))
  *Authors:* Jeff Lukas, Mike Hobbins, Imtiaz Rangwala (September 2017)

- **NOAA PSL EDDI Page:** [https://psl.noaa.gov/eddi/](https://psl.noaa.gov/eddi/)
- **NOAA CPC Operational:** [https://www.cpc.ncep.noaa.gov/products/Drought/eddi/](https://www.cpc.ncep.noaa.gov/products/Drought/eddi/)

**Operational Transition (2024):** EDDI has transitioned from NOAA Physical Sciences Laboratory (PSL) to the Climate Prediction Center (CPC) for operational drought monitoring. PSL continues research to enhance EDDI.

### 1.3 Methodology Comparison: Hobbins 2016 vs Our Implementation

| Aspect | Hobbins et al. 2016 Specification | Our Implementation (`feature/issue-414-eddi-clean`) | Match? |
|--------|-----------------------------------|-----------------------------------------------------|--------|
| **E0 Formulation** | Penman-Monteith FAO56 (0.5m reference crop) exclusively | **Any PET input accepted** (no validation) | ⚠️ **Partial** |
| **Input Variables** | Temperature, humidity, wind speed, incoming solar radiation from NLDAS-2 | User-provided PET array (no variable requirements) | ⚠️ **Different** |
| **Temporal Resolution** | Daily (multi-scalar 1-week to 12-month accumulations) | Daily or monthly (via `periodicity` parameter) | ✅ **Match** |
| **Accumulation Scales** | 1-week to 12-month (flash drought to sustained drought) | User-specified `scale` parameter (1-month to N-month) | ✅ **Match** |
| **Standardization Method** | Non-parametric empirical ranking | Non-parametric empirical ranking | ✅ **Match** |
| **Ranking Procedure** | Rank each value against climatology for same calendar period | Correct: `rank = 1 + np.sum(current_value > climatology_valid)` | ✅ **Match** |
| **Rank Direction** | Ascending (lowest PET = rank 1) | Ascending (rank 1 for lowest PET) | ✅ **Match** |
| **Plotting Position** | Not explicitly specified in papers; NOAA Fortran uses Tukey | Tukey: `(rank - 0.33) / (N + 0.33)` | ✅ **Match** |
| **Inverse Normal** | Hastings approximation (Abramowitz & Stegun 26.2.23) | Hastings with exact constants from A&S 26.2.23 | ✅ **Match** |
| **Sign Convention** | Higher E0 → higher EDDI → drier conditions | Higher PET → higher EDDI → drier conditions | ✅ **Match** |
| **Tie Handling** | Not specified | Strict `>` comparison (ties get lower rank) | ⚠️ **Assumed** |
| **Self-Inclusion** | Not specified | Value at calibration year included in its own ranking | ⚠️ **Assumed** |
| **Calibration Period** | 1979-present (rolling) for operational EDDI | User-specified `calibration_year_initial` to `calibration_year_final` | ✅ **Match** |
| **Minimum Climatology** | Not specified | Requires ≥2 valid values (skips period if <2) | ⚠️ **Assumed** |
| **Output Range** | Standardized z-scores (clipped to ±3.09 in code) | Clipped to ±3.09 | ✅ **Match** |

**Assessment Summary:**
- **Core algorithm:** ✅ Matches NOAA reference implementation
- **Critical gap:** ⚠️ No validation that input PET uses appropriate method (PM FAO56)
- **Ambiguities:** Tie handling, self-inclusion, and minimum climatology not specified in papers

### 1.4 Hobbins Specification: E0 Formulation

**From NOAA PSL EDDI Page:**
> "The E0 is calculated using the **Penman-Monteith FAO56 reference evapotranspiration formulation (0.5-m tall reference crop)**, driven by data on temperature, humidity, wind speed, and incoming solar radiation, with these data extracted from the operational North American Land Data Assimilation System (NLDAS-2) dataset."

**Key Point:** Hobbins uses PM FAO56 **exclusively** in the operational implementation. This is not just a recommendation — it's the foundational methodology. Using Thornthwaite or other simplified PET methods may compromise EDDI accuracy (see Section 3).

---

## 2. Non-Parametric Standardization Deep Dive

### 2.1 Plotting Position Formulas: Comparison

Plotting positions convert empirical ranks to probabilities for inverse normal transformation. Different formulas have different statistical properties.

| Formula | Expression | α, β Parameters | Properties | Best For |
|---------|-----------|-----------------|-----------|----------|
| **Tukey (Type 6)** | `(r - 0.33) / (N + 0.33)` | α=0.33, β=0.33 | Approximately median-unbiased for normal distribution | **General-purpose, normal data** |
| **Weibull (Type 1)** | `r / (N + 1)` | α=0, β=0 | Unbiased exceedance probability for all distributions | **Hydrology, water resources** |
| **Hazen (Type 5)** | `(r - 0.5) / N` | α=0.5, β=0.5 | Piecewise linear interpolation of ECDF | **Traditional choice, symmetric** |
| **Gringorten** | `(r - 0.44) / (N + 0.12)` | α=0.44, β=0.12 | Optimized for Gumbel (extreme value) distribution | **Small samples, extreme values** |
| **Cunnane** | `(r - 0.4) / (N + 0.2)` | α=0.4, β=0.4 | Good for normally distributed data | **Default in some software** |

**Sources:**
- [Matplotlib Probscale — Using Different Formulations of Plotting Positions](https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_plot_pos.html)
- [AMS Journal: Plotting Positions in Extreme Value Analysis](https://journals.ametsoc.org/view/journals/apme/45/2/jam2349.1.xml)

### 2.2 Why Tukey for EDDI?

**Tukey's Advantages:**
- **Median-unbiased** for normal distribution — minimizes bias in z-score transformation
- **General-purpose** — performs well across diverse climate conditions (EDDI is CONUS-wide)
- **Consistent with NOAA Fortran** — matches reference implementation exactly

**Comparison to Alternatives:**
- **Weibull** is unbiased for exceedance probability but **not** optimized for normal transformation
- **Gringorten** optimized for extreme value (Gumbel) distributions — EDDI doesn't assume extremes
- **Hazen** is symmetric but less statistically efficient than Tukey

**Verdict:** ✅ Tukey is the correct choice for non-parametric normal transformation

### 2.3 Hastings Inverse Normal Approximation

**Abramowitz & Stegun 26.2.23 Accuracy:**

Our implementation uses the Hastings rational approximation for inverse normal CDF:

```python
_HASTINGS_C0 = 2.515517
_HASTINGS_C1 = 0.802853
_HASTINGS_C2 = 0.010328
_HASTINGS_D1 = 1.432788
_HASTINGS_D2 = 0.189269
_HASTINGS_D3 = 0.001308
```

**Accuracy Characteristics:**
- **Absolute error:** < 4 × 10⁻⁴ (reported in Abramowitz & Stegun)
- **Alternative studies:** Some sources report errors as low as 7.5 × 10⁻⁸
- **Relative error:** Small absolute errors but relative errors significant in tails

**Test Validation:**
- Our test `test_hastings_inverse_normal_accuracy()` validates against `scipy.stats.norm.ppf()` with tolerance `rtol=1e-3, atol=1e-3`
- Test passes across probability range [0.001, 0.999]

**Why Hastings over scipy?**
1. **NOAA compatibility** — Matches Fortran reference implementation exactly
2. **Reproducibility** — Eliminates scipy version dependency for validation
3. **Performance** — Faster than scipy for large arrays (though not benchmarked)

**Sources:**
- [MaplePrimes: An Improved Approximation of the Inverse Normal CDF](https://www.mapleprimes.com/posts/211001-An-Improved-Approximation-Of-The-Inverse)
- [Springer: Approximation of the inverse normal distribution function](https://link.springer.com/article/10.3758/BF03200956)

**Verdict:** ✅ Hastings approximation is appropriate and validated

### 2.4 Parametric vs Non-Parametric: Noguera 2022 Challenge

**Recent Research Finding:**

[Noguera et al. (2022) — Assessment of parametric approaches to calculate the Evaporative Demand Drought Index](https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.7275)
*Published:* International Journal of Climatology

**Key Findings:**
- Evaluated 8 probability distributions for EDDI at 1-, 3-, and 12-month scales
- **Winner:** **Log-logistic distribution** outperformed all others
- **Rejected distributions:** Normal (high percentage of series fail normality tests), Pearson III (higher frequency of extreme values)
- **Advantage:** Parametric approach can model values outside reference climatology period

**Noguera's Recommendation:**
> "The Log-logistic distribution is the best option for generating standardized values over very different climate conditions, and the study recommends adopting a robust parametric approach based on the Log-logistic distribution for drought analysis, as opposed to the **original nonparametric approach**."

**Implication for Our Implementation:**
- Our non-parametric approach matches NOAA operational EDDI (good for validation)
- Noguera 2022 suggests parametric log-logistic may be more accurate
- **Climate Engine** already supports both: non-parametric (default) and log-logistic

**Scope Limitation:**
This research validates the **non-parametric** approach as specified by NOAA. Exploring log-logistic parametric fitting is **out of scope** but should be considered for future enhancement (Phase 3+).

**Sources:**
- [Wiley: Noguera et al. 2022 — Assessment of parametric approaches](https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.7275)
- [GitHub: ivannoguera/EDDI-Log-logistic](https://github.com/ivannoguera/EDDI-Log-logistic)

---

## 3. ET Method and EDDI Accuracy

### 3.1 PET Method Comparison

| PET Method | Input Variables | Complexity | Physical Basis | Appropriate for EDDI? |
|-----------|----------------|-----------|----------------|----------------------|
| **Penman-Monteith FAO56** | Temperature, humidity, wind speed, solar radiation, latitude | High | Fully physics-based (energy balance + aerodynamic) | ✅ **Recommended by Hobbins** |
| **Hargreaves** | Temperature (min/max), solar radiation (extraterrestrial), latitude | Medium | Empirical radiation-based with temperature | ⚠️ **Middle ground** |
| **Thornthwaite** | Temperature only | Low | Temperature-based empirical (pre-1950s) | ❌ **Inappropriate** |

### 3.2 Sensitivity Analysis: PDSI and SPEI Research

**Key Study:** [van der Schrier et al. (2011) — Sensitivity of the PDSI to Thornthwaite and Penman-Monteith parameterizations](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JD015001)

**Findings:**
- **PDSI is insensitive** to PET method choice (Thornthwaite vs PM) because the water balance model normalizes differences
- **SPEI is more sensitive** — PM provides slightly higher correlations to streamflow in water-limited regions
- **Regional variation:** Thornthwaite performs similarly in energy-limited regions, worse in water-limited regions

**Hargreaves Alternative:**
[Springer: Evaluating Sensitivity of Moisture Indices to Six PET Models](https://link.springer.com/article/10.1007/s41748-025-00777-x)
> "Hargreaves and Blaney-Morin-Nigeria emerge as robust alternatives to the data-intensive Penman-Monteith model in data-scarce tropical environments."

**Sources:**
- [AGU: van der Schrier et al. 2011 — PDSI sensitivity study](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JD015001)
- [Springer: PET model comparison for moisture indices](https://link.springer.com/article/10.1007/s41748-025-00777-x)

### 3.3 EDDI-Specific Considerations

**Unlike PDSI/SPEI, EDDI has no water balance normalization.**

EDDI directly ranks PET values — there's no precipitation term to offset systematic biases. This makes EDDI **highly sensitive** to PET method choice:

- **Thornthwaite PET** is systematically biased (temperature-only proxy)
  - Misses wind effects (critical for evaporative demand)
  - Misses humidity effects (vapor pressure deficit)
  - Cannot capture flash drought driven by hot, dry, windy conditions

- **Hargreaves PET** includes radiation proxy but still misses wind/humidity
  - May be acceptable as middle ground in data-limited scenarios
  - Not validated for EDDI (no published studies found)

- **Penman-Monteith FAO56** is the physically-based gold standard
  - Hobbins uses PM FAO56 **exclusively** in operational EDDI
  - Captures energy balance + aerodynamic components

**Verdict:**
- ✅ **PM FAO56 should be required or strongly recommended** for EDDI
- ⚠️ **Hargreaves acceptable with documentation** in data-limited scenarios
- ❌ **Thornthwaite fundamentally inappropriate** — will produce incorrect drought signals

### 3.4 Cross-Reference: Penman-Monteith Implementation

**Parallel Work:** Comprehensive FAO56 Penman-Monteith implementation research completed (see `_bmad-output/planning-artifacts/research/technical-penman-monteith.md`).

**Key Findings from PM Research:**
- Complete FAO56 Equations 1-19 documented and validated
- Hybrid API pattern proposed: `eto_penman_monteith()` public function + private equation helpers
- Validation against FAO56 worked examples (Bangkok tropical, Uccle temperate)
- Phased implementation roadmap: Core equations → Humidity pathways → Documentation

**EDDI Integration Path:**
1. **Phase 2A (EDDI merge):** Document PM as recommended PET method in EDDI docstring
2. **Phase 2B (Post-PM implementation):** Create EDDI + PM usage examples
3. **Phase 2B:** Add `eto_penman_monteith()` to EDDI validation test suite

**Impact on EDDI Recommendation 2:**
The parallel PM implementation work reduces the urgency of PET method validation code in EDDI. Instead of adding runtime warnings for Thornthwaite, the library will provide the correct method (`eto_penman_monteith()`) directly, with documentation guiding users to use it for EDDI.

---

### 3.5 Temporal Resolution: Daily vs Monthly

**From NOAA PSL EDDI:**
> "EDDI is generated **daily** — though with a 5-day lag-time — by analyzing a near-real-time atmospheric dataset. Beyond daily updates, EDDI is **multi-scalar**, meaning the time period can vary to capture drying dynamics that operate at different timescales; EDDI is generated at **1-week through 12-month timescales**."

**Flash Drought Detection:**
[ScienceDirect: Flash drought monitoring using diurnal-provided EDDI](https://www.sciencedirect.com/science/article/pii/S002216942400355X)
> "A particular strength of EDDI is in capturing the precursor signals of water stress at **weekly to monthly timescales**, which makes EDDI a strong tool for preparedness for both flash droughts and ongoing droughts."

**Recent Enhancement:**
Diurnal-provided EDDI (using GNSS atmospheric products) extends flash drought lead time to **37.74 days** vs standard EDDI.

**Our Implementation:**
- Supports both daily and monthly via `periodicity` parameter ✅
- No validation that daily accumulations match NOAA reference ⚠️

**Sources:**
- [NOAA PSL EDDI — Multi-scalar daily generation](https://psl.noaa.gov/eddi/)
- [ScienceDirect: Flash drought monitoring](https://www.sciencedirect.com/science/article/pii/S002216942400355X)
- [NREL: EDDI flash drought early warning](https://www.nrel.colostate.edu/eddi-a-new-drought-index-provides-early-warning-of-flash-droughts/)

---

## 4. PR 597 Gap Analysis

### 4.1 Algorithm Correctness Review

| Component | Implementation Detail | Correctness | Notes |
|-----------|----------------------|-------------|-------|
| **Rank Direction** | `rank = 1 + np.sum(current_value > climatology_valid)` | ✅ Correct | Ascending rank: lowest PET = rank 1 |
| **Plotting Position** | `(rank - 0.33) / (N + 0.33)` | ✅ Correct | Tukey formula, constants as floats `0.33` |
| **Probability Clipping** | `np.clip(p, 1e-10, 1.0 - 1e-10)` | ✅ Correct | Avoids log(0) in Hastings |
| **Inverse Normal** | Hastings constants match A&S 26.2.23 exactly | ✅ Correct | Validated against scipy |
| **Output Clipping** | `np.clip(eddi_values, -3.09, 3.09)` | ✅ Correct | Matches SPI/SPEI range |
| **Tie Handling** | Strict `>` comparison | ⚠️ Assumed | NOAA spec unclear on tie behavior |
| **Self-Inclusion** | Value included in its own climatology ranking | ⚠️ Assumed | Not specified by Hobbins |
| **Constant Precision** | `0.33` vs `1/3` | ⚠️ Minor | Using `0.33` matches Fortran, but `1/3` more precise |

**Critical Finding:** Algorithm correctness is ✅ **validated** with minor ambiguities on tie handling and self-inclusion.

### 4.2 Validation Gaps (FR-TEST-004 BLOCKING)

**Architecture v1.1 Pattern 8 Requirement:**
> "Separate test module: `tests/test_reference_validation.py` for validating against published reference datasets"
> "EDDI validation tolerance: 1e-5 (looser than equivalence tests due to non-parametric ranking FP accumulation)"

**Current Test Coverage:**
- ✅ 15 test functions in `tests/test_eddi.py`
- ✅ Edge cases: all-NaN, negative values, invalid shapes, insufficient climatology
- ✅ Hastings approximation validated against scipy (tolerance 1e-3)
- ✅ Empirical ranking logic validated with synthetic data
- ❌ **NO NOAA reference dataset validation** ← **BLOCKING**

**Missing Components:**

| Component | Status | Blocker? |
|-----------|--------|----------|
| `tests/test_reference_validation.py` module | ❌ Missing | **YES** (Architecture v1.1) |
| `tests/data/reference/eddi_noaa_reference.nc` | ❌ Missing | **YES** (FR-TEST-004) |
| Provenance documentation (source, date, method) | ❌ Missing | **YES** (FR-TEST-004) |
| Reference dataset registry | ❌ Missing | **YES** (Architecture v1.1) |
| Validation test with tolerance 1e-5 | ❌ Missing | **YES** (Architecture v1.1) |

**NOAA Reference Data Availability:**

Archive discovered at: [https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/](https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/)

Contents:
- `data/` directory (likely netCDF files)
- `images/` directory
- `Readme.pdf` (75K, dated 2017-09-20)

**Next Steps:**
1. Download `Readme.pdf` to understand data format and coverage
2. Extract small CONUS subset for validation (e.g., single grid cell, 1980-2020)
3. Create `tests/data/reference/eddi_noaa_reference.nc` with provenance metadata
4. Implement `test_reference_validation.py::test_eddi_noaa_reference()`

**Impact:** This is a **BLOCKING gap** for PR merge. No reference validation = cannot satisfy FR-TEST-004.

### 4.3 Feature Gaps

| Feature | Status | Priority | Epic |
|---------|--------|----------|------|
| **CLI Integration** | ❌ Missing | High | Issue #414 (original motivation) |
| **xarray Adapter** | ❌ Missing | High | Phase 2 (Epic 2) |
| **PM FAO56 PET Support** | ⚠️ Not validated | High | Should be library requirement |
| **EDDI in API Docs** | ❌ Missing | Medium | Phase 2 (Epic 5) |
| **Hobbins Citation** | ❌ Missing | Medium | Documentation gap |
| **Sign Convention Docs** | ❌ Missing | Low | Clarity improvement |

**CLI Integration (Issue #414):**
- Original issue requested EDDI in CLI
- PR #597 implements algorithm but no CLI wrapper
- Scope: Add `process_climate_indices` CLI support for EDDI

**xarray Adapter:**
- PRD v1.1 Epic 2 requires xarray support for all indices
- EDDI needs decorator-based adapter (same pattern as SPI/SPEI)
- Input validation for coordinate names (`time`, `lat`, `lon`)

**PM FAO56 Validation:**
- Current implementation accepts any PET input (no validation)
- Should warn if PET method is not PM FAO56
- Consider adding `pet_method` parameter for documentation

### 4.4 Documentation Gaps

**Severity: Medium**

| Gap | Current State | Recommendation |
|-----|---------------|----------------|
| **No Hobbins citation** | No references in docstring or code comments | Add to `eddi()` docstring: "References: Hobbins et al. (2016) DOI:10.1175/JHM-D-15-0121.1" |
| **No algorithms.rst entry** | EDDI not documented in `docs/algorithms.rst` | Add EDDI section with methodology, plotting positions, inverse normal |
| **Sign convention unclear** | Only mentioned in docstring once | Add explicit note: "Higher PET → higher EDDI → drier conditions" |
| **PET method requirements** | Docstring says "PET values, in any units" | Clarify: "PET values (preferably Penman-Monteith FAO56), in any units" |
| **No examples** | No usage examples in docstring | Add minimal example with monthly PET |

**Example Docstring Addition:**

```python
"""
Computes EDDI (Evaporative Demand Drought Index) using the NOAA PSL
non-parametric empirical ranking approach.

...

Note:
    EDDI is most accurate when using Penman-Monteith FAO56 reference
    evapotranspiration (E0). Using simplified methods like Thornthwaite
    may produce inaccurate drought signals.

References:
    Hobbins, M. T., Wood, A., McEvoy, D. J., et al. (2016). The Evaporative
    Demand Drought Index. Part I: Linking Drought Evolution to Variations
    in Evaporative Demand. J. Hydrometeor., 17(6), 1745-1761.
    https://doi.org/10.1175/JHM-D-15-0121.1

Example:
    >>> import numpy as np
    >>> from climate_indices import indices, compute
    >>> # 10 years of monthly PET data (mm)
    >>> pet = np.random.uniform(50, 200, 10 * 12)
    >>> eddi_6month = indices.eddi(
    ...     pet_values=pet,
    ...     scale=6,
    ...     data_start_year=2000,
    ...     calibration_year_initial=2002,
    ...     calibration_year_final=2007,
    ...     periodicity=compute.Periodicity.monthly,
    ... )
"""
```

### 4.5 Gap Summary Table

| Category | Blocking Gaps | High-Priority Gaps | Medium-Priority Gaps | Low-Priority Gaps |
|----------|---------------|--------------------|--------------------|-------------------|
| **Validation** | • FR-TEST-004 NOAA reference<br>• Pattern 8 test module<br>• Provenance docs | • Tolerance validation (1e-5) | — | • Edge case expansion |
| **Features** | — | • CLI integration (#414)<br>• xarray adapter (Epic 2) | • PM FAO56 docstring<br>• PM integration examples (post-impl) | • pet_method param |
| **Documentation** | — | • Hobbins citation | • algorithms.rst entry<br>• Sign convention | • Usage examples |
| **Code Quality** | — | — | • Constant precision (0.33 vs 1/3) | • Tie handling docs |

**Total Gaps:** 3 blocking, 3 high-priority, 4 medium-priority, 3 low-priority

**Note:** PM FAO56 validation moved from High to Medium priority due to parallel `eto_penman_monteith()` implementation in progress (see Section 3.4).

---

## 5. Prioritized Recommendations and Next Steps

### 5.1 Phase 2 Integration Blockers (Must Do)

#### Recommendation 1: Implement FR-TEST-004 Reference Validation (BLOCKING)

**Priority:** 🔴 **CRITICAL** — Blocks PR merge

**Actions:**
1. Download NOAA PSL EDDI CONUS archive `Readme.pdf` and sample data
2. Extract validation subset:
   - Single grid cell (e.g., 40°N, 105°W — Colorado Front Range)
   - Time period: 1980-2020 (40 years)
   - Scales: 1-month, 6-month, 12-month
3. Create `tests/data/reference/eddi_noaa_reference.nc` with attributes:
   - `source: "NOAA PSL EDDI CONUS archive"`
   - `url: "https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/"`
   - `download_date: "YYYY-MM-DD"`
   - `license: "Public domain (NOAA data)"`
4. Implement `tests/test_reference_validation.py::test_eddi_noaa_reference()`
   - Tolerance: `np.testing.assert_allclose(computed, reference, rtol=1e-5, atol=1e-5)`
5. Document provenance in `tests/data/reference/README.md`

**Acceptance Criteria:**
- ✅ Test passes with tolerance 1e-5
- ✅ Provenance documented
- ✅ FR-TEST-004 satisfied
- ✅ Architecture v1.1 Pattern 8 implemented

**Estimated Effort:** 4-6 hours

---

#### Recommendation 2: Add PET Method Validation (MEDIUM - Deferred to Phase 2B)

**Priority:** 🟡 **MEDIUM** — Can be deferred with PM implementation in progress

**Context:** Parallel implementation of Penman-Monteith FAO56 PET (see `_bmad-output/planning-artifacts/research/technical-penman-monteith.md`) will provide the recommended PET method for EDDI. Once PM is available, EDDI can use it directly.

**Revised Strategy:**
1. **Phase 2A (Merge):** Document PM FAO56 recommendation in EDDI docstring (no validation code)
2. **Phase 2B (Post-PM):** Add `eto_penman_monteith()` integration example
3. **Optional:** Add `pet_method` parameter for documentation purposes only

**Actions (Minimal for Phase 2A):**
1. Update `eddi()` docstring:
   ```python
   """
   Note:
       EDDI is most accurate when using Penman-Monteith FAO56 reference
       evapotranspiration (E0). Using simplified methods like Thornthwaite
       may produce inaccurate drought signals.

   See Also:
       eto_penman_monteith : FAO56 PM reference evapotranspiration (recommended)
       eto_hargreaves : Alternative PET method (middle ground)
   """
   ```

**Acceptance Criteria:**
- ✅ Docstring updated with PET method guidance
- ✅ Cross-reference to `eto_penman_monteith()` when available

**Estimated Effort:** 30 minutes (documentation only)

---

### 5.2 Phase 2 Feature Completion (Should Do)

#### Recommendation 3: CLI Integration (Issue #414)

**Priority:** 🟠 **HIGH** — Original issue motivation

**Actions:**
1. Add `--index eddi` support to `process_climate_indices` CLI
2. Add `--pet_file` parameter for PET input
3. Add `--pet_method` parameter for method documentation
4. Update CLI help text and examples

**Acceptance Criteria:**
- ✅ CLI can compute EDDI from netCDF PET file
- ✅ Help text documents PET method requirements
- ✅ Example in README or docs

**Estimated Effort:** 3-4 hours

---

#### Recommendation 4: xarray Adapter (Epic 2)

**Priority:** 🟠 **HIGH** — PRD v1.1 Epic 2 requirement

**Actions:**
1. Create `@adapt_for_xarray` decorator wrapper for `eddi()`
2. Add CF metadata to registry:
   ```python
   "eddi": {
       "long_name": "Evaporative Demand Drought Index",
       "units": "1",  # dimensionless z-score
       "standard_name": "atmosphere_water_vapor_evaporative_demand_anomaly",
       "references": "Hobbins et al. (2016) doi:10.1175/JHM-D-15-0121.1",
   }
   ```
3. Validate coordinate handling (`time`, `lat`, `lon`)

**Acceptance Criteria:**
- ✅ `eddi(xarray.DataArray)` returns `xarray.DataArray`
- ✅ CF metadata attached
- ✅ Coordinates preserved
- ✅ Test validates xarray path

**Estimated Effort:** 4-6 hours

---

### 5.3 Documentation Improvements (Nice to Have)

#### Recommendation 5: Comprehensive Documentation

**Priority:** 🟡 **MEDIUM**

**Actions:**
1. Add Hobbins et al. 2016 citation to docstring
2. Create `docs/algorithms.rst` EDDI section:
   - Methodology overview
   - Non-parametric vs parametric (Noguera 2022)
   - Tukey plotting positions
   - Hastings inverse normal
   - PET method requirements
3. Add usage examples to docstring
4. Document sign convention explicitly

**Estimated Effort:** 3-4 hours

---

### 5.4 Future Research (Phase 3+)

#### Recommendation 6: Explore Parametric Log-Logistic EDDI

**Priority:** 🟢 **LOW** — Research topic, not immediate need

**Rationale:** Noguera et al. (2022) demonstrated that log-logistic distribution outperforms non-parametric approach and can model values outside reference climatology.

**Actions:**
1. Research Noguera 2022 methodology in depth
2. Prototype log-logistic fitting for EDDI
3. Compare accuracy vs non-parametric on CONUS data
4. If validated, add as optional `distribution=Distribution.loglogistic` parameter

**Estimated Effort:** 16-24 hours (research project)

---

### 5.5 Implementation Roadmap

**Phase 2A (Merge PR #597):**
1. Implement FR-TEST-004 reference validation (Rec 1) — **BLOCKING** (4-6 hours)
2. CLI integration (Rec 3) — **HIGH** (3-4 hours)
3. Update docstring with PM FAO56 recommendation (Rec 2 minimal) — **MEDIUM** (30 min)

**Phase 2B (Epic 2 Integration + PM Integration):**
4. xarray adapter (Rec 4) — **HIGH** (4-6 hours)
5. Integrate with `eto_penman_monteith()` examples (Rec 2 full) — **MEDIUM** (1-2 hours)
6. Documentation (Rec 5) — **MEDIUM** (3-4 hours)

**Phase 3 (Future Enhancement):**
7. Parametric log-logistic exploration (Rec 6) — **LOW** (16-24 hours)

**Estimated Total Effort for Phase 2A Merge:** 8-11 hours (reduced with PM implementation in progress)

---

## 6. References and Sources

### Primary Literature

- Hobbins, M. T., Wood, A., McEvoy, D. J., Huntington, J. L., Morton, C., Anderson, M., & Hain, C. (2016). The Evaporative Demand Drought Index. Part I: Linking Drought Evolution to Variations in Evaporative Demand. *Journal of Hydrometeorology*, 17(6), 1745-1761. [https://doi.org/10.1175/JHM-D-15-0121.1](https://journals.ametsoc.org/view/journals/hydr/17/6/jhm-d-15-0121_1.xml)

- McEvoy, D. J., Huntington, J. L., Hobbins, M. T., Wood, A., Morton, C., Anderson, M., & Hain, C. (2016). The Evaporative Demand Drought Index. Part II: CONUS-Wide Assessment Against Common Drought Indicators. *Journal of Hydrometeorology*, 17(6), 1763-1779. [https://doi.org/10.1175/JHM-D-15-0122.1](https://journals.ametsoc.org/view/journals/hydr/17/6/jhm-d-15-0122_1.xml)

- Noguera, I., Domínguez-Castro, F., & Vicente-Serrano, S. M. (2022). Assessment of parametric approaches to calculate the Evaporative Demand Drought Index. *International Journal of Climatology*, 42(2), 1120-1137. [https://doi.org/10.1002/joc.7275](https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.7275)

### NOAA Operational Documentation

- EDDI User Guide v1.0 (Lukas, J., Hobbins, M., & Rangwala, I., 2017). [https://psl.noaa.gov/eddi/pdf/EDDI_UserGuide_v1.0.pdf](https://psl.noaa.gov/eddi/pdf/EDDI_UserGuide_v1.0.pdf)

- NOAA Physical Sciences Laboratory EDDI Page: [https://psl.noaa.gov/eddi/](https://psl.noaa.gov/eddi/)

- NOAA Climate Prediction Center EDDI (operational): [https://www.cpc.ncep.noaa.gov/products/Drought/eddi/](https://www.cpc.ncep.noaa.gov/products/Drought/eddi/)

- NOAA PSL EDDI CONUS Archive: [https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/](https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/)

### Statistical Methods

- Matplotlib Probscale: Using Different Formulations of Plotting Positions. [https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_plot_pos.html](https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_plot_pos.html)

- Plotting Positions in Extreme Value Analysis (AMS, 2006). [https://journals.ametsoc.org/view/journals/apme/45/2/jam2349.1.xml](https://journals.ametsoc.org/view/journals/apme/45/2/jam2349.1.xml)

- Hastings Approximation References:
  - MaplePrimes: An Improved Approximation of the Inverse Normal CDF. [https://www.mapleprimes.com/posts/211001-An-Improved-Approximation-Of-The-Inverse](https://www.mapleprimes.com/posts/211001-An-Improved-Approximation-Of-The-Inverse)
  - Springer: Approximation of the inverse normal distribution function. [https://link.springer.com/article/10.3758/BF03200956](https://link.springer.com/article/10.3758/BF03200956)

### PET Method Sensitivity

- van der Schrier, G., Barichivich, J., Briffa, K. R., & Jones, P. D. (2011). The sensitivity of the PDSI to the Thornthwaite and Penman-Monteith parameterizations for potential evapotranspiration. *Journal of Geophysical Research: Atmospheres*, 116(D3). [https://doi.org/10.1029/2010JD015001](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JD015001)

- Springer: Evaluating Sensitivity of Moisture Indices to Six PET Models. [https://link.springer.com/article/10.1007/s41748-025-00777-x](https://link.springer.com/article/10.1007/s41748-025-00777-x)

### Flash Drought Detection

- ScienceDirect: Flash drought monitoring using diurnal-provided EDDI. [https://www.sciencedirect.com/science/article/pii/S002216942400355X](https://www.sciencedirect.com/science/article/pii/S002216942400355X)

- NREL: EDDI flash drought early warning. [https://www.nrel.colostate.edu/eddi-a-new-drought-index-provides-early-warning-of-flash-droughts/](https://www.nrel.colostate.edu/eddi-a-new-drought-index-provides-early-warning-of-flash-droughts/)

### Climate Engine

- Climate Engine: Standardized Index Calculation Methods. [https://support.climateengine.org/article/130-standardized-indice-calculation-methods](https://support.climateengine.org/article/130-standardized-indice-calculation-methods)

- Climate Engine: EDDI Support Documentation. [https://support.climateengine.org/article/96-evaporative-demand-drought-index-eddi](https://support.climateengine.org/article/96-evaporative-demand-drought-index-eddi)

### Codebase References

- **Implementation:** `feature/issue-414-eddi-clean:src/climate_indices/indices.py::eddi()`
- **Tests:** `feature/issue-414-eddi-clean:tests/test_eddi.py` (15 test functions)
- **PRD:** `_bmad-output/planning-artifacts/prd.md` (FR-TEST-004, lines 196-205)
- **Architecture:** `_bmad-output/planning-artifacts/architecture.md` (Pattern 8, lines 182-187)
- **Readiness Report:** `_bmad-output/planning-artifacts/implementation-readiness-report-2026-02-09.md` (lines 192-253)

---

## Conclusion

The EDDI implementation in PR #597 (branch `feature/issue-414-eddi-clean`) is **algorithmically correct** and matches the NOAA Fortran reference implementation. The non-parametric empirical ranking approach with Tukey plotting positions and Hastings inverse normal approximation is validated and appropriate.

However, **FR-TEST-004 reference validation is BLOCKING** — no NOAA reference dataset test exists, preventing Phase 2 integration. Additionally, the implementation lacks PET method validation (critical for accuracy), CLI integration (original issue motivation), and comprehensive documentation.

**Immediate Next Steps:**
1. Implement FR-TEST-004 NOAA reference validation (BLOCKING)
2. Add PET method validation warnings (HIGH)
3. CLI integration for Issue #414 (HIGH)

With these gaps addressed, EDDI will be ready for Phase 2 xarray integration and operational use in the climate_indices library.

---

**End of Research Report**

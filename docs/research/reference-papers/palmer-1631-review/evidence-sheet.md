# Palmer spell establishment in one reduced case

Do the published standard Palmer and Wells spell-establishment rules explain why these two PDSI recursions interpret the same five active months differently when the Z-Index and duration factors are held fixed? [Issue #1631](https://github.com/monocongo/climate_indices/issues/1631)

| Index (zero-based) | Z-Index | Standard PDSI (`pdi.f` lineage) | Wells-lineage PDSI |
| --- | ---: | ---: | ---: |
| 0–6 (padding) | `0.0` | `0.0` | `0.0` |
| 7 | `-1.0` | `-0.3333333333333333` | `-0.3333333333333333` |
| **8 (establishment decision)** | `-1.0` | `-0.6323333333333334` | `-0.6323333333333333` |
| **9 (first material output difference)** | `-0.0` | `-0.5672030000000001` | `0.0` |
| 10 | `3.0` | `1.0` | `1.0` |
| 11 | `1.0` | `1.2303333333333335` | `1.2303333333333333` |

Both Python recursions start with zero spell state and use Palmer’s national coefficients, p = 0.897 and q = 1/3. For wet and dry alike, m = (1 − p)/q and b = p/q, computed as `0.30899999999999994` and `2.6910000000000003`. Case E is a reduced sequence from a 13-month seed for division 0102. Its shortest possible length is unknown. The checkout is `380aa01f` (`climate_indices` 3.2.0). The table shows final PDSI at the stored precision. [Committed sequence](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/research/reference-papers/palmer-recursion-divergence-sequences.json#L213-L258)

## What the publications specify

Palmer defines an established spell at X₁ ≥ 1.00 (wet) or X₂ ≤ −1.00 (dry), and describes assigning earlier values using the candidates’ zero crossings. Alley confirms the inclusive thresholds and illustrates retrospective assignment. The printed Palmer inequality signs resolve a strict-inequality transcription in the study’s OCR. [Palmer (1965), printed pp. 32–33, PDF pp. 38–39](https://www.ncei.noaa.gov/monitoring-content/temp-and-precip/drought/docs/palmer.pdf#page=38) · [Alley (1984), pp. 1103–1104](https://doi.org/10.1175/1520-0450%281984%29023%3C1100%3ATPDSIL%3E2.0.CO%3B2)

Wells et al. write that a spell “becomes established when it reaches the threshold of ±0.5.” They describe retrospective revision and leave the detailed triggers and steps of backtracking open. The Wells C++ port and the pinned Python recursion use inclusive comparisons at ±0.5. The C++ backtracking code uses a near-zero tolerance, and Python tests exact zero. [Wells et al. (2004), p. 2337, §2](https://www.pmf.unizg.hr/_download/repository/Wells_2004_sc-PDSI.pdf#page=3) · [Wells C++ port](https://github.com/Sibada/scPDSI/blob/10f8c3b/src/pdsi.cpp#L3562-L3649) · [Provenance, rows 14 and 19](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/algorithm_refs/palmer-provenance.md#L69-L82)

## Proposed interpretation

At index 8 the dry candidate is about −0.6323. It becomes an established drought in the Wells lineage and stays incipient in the standard lineage. Both establish a wet spell at +1.0 at index 10. The Wells recursion ends its drought there and backtracks index 9 to 0.0, where both of its incipient candidates had been zero. The standard recursion retains −0.5672030000000001 at index 9. Changing only the standard implementation’s establishment threshold to 0.5 removes every difference above 10⁻⁹ in this case. The published magnitudes explain the establishment decision. The exact retrospective values come from the traced implementations. [Study #1658](https://github.com/monocongo/climate_indices/pull/1658) · [Provenance D3](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/algorithm_refs/palmer-provenance.md#L124-L127)

**Reply in plain text (60–90 minutes).** Which interpretation do the passages support? What remains ambiguous? Which single check would change your judgment?

[Optional coverage supplement](coverage-assessment.md) · [Numerical replay](optional-replay.md) · [Review outcome template](outcome-note-template.md)

# Palmer Drought Indices

## Scope

The Palmer implementation computes five monthly drought products from
precipitation, PET, and available water capacity:

- Palmer Drought Severity Index (PDSI)
- Palmer Hydrological Drought Index (PHDI)
- Palmer Modified Drought Index (PMDI)
- Palmer Z-Index
- Self-calibrated Palmer Drought Severity Index (scPDSI)

The public NumPy APIs are `climate_indices.palmer.pdsi()` and
`climate_indices.palmer.scpdsi()`. The 3.0.0 xarray notebook demonstrates how
to call the NumPy Palmer routine from labeled xarray inputs and rewrap the
outputs with coordinates and metadata.

## Algorithm Notes

The implementation follows the water-balance structure described by Palmer
(1965):

1. Validate precipitation and PET arrays and available water capacity.
2. Initialize monthly water-balance terms.
3. Compute CAFEC coefficients for the calibration period.
4. Compute K weighting factors.
5. Calculate the Z-Index moisture anomaly.
6. Finish the recursive Palmer index calculations for PDSI, PHDI, and PMDI.

`scpdsi()` shares the water-balance and CAFEC stages, then computes
self-calibrating K-prime factors, fits location-specific duration factors,
uses the Wells recursion, and performs three cumulative Z-index rescaling
passes over the requested calibration period.

### Exact-zero comparisons

Both recursions branch on exact comparisons against zero, and every float
comparison site in `src/climate_indices/palmer.py` carries a rationale comment
plus a `# NOSONAR` marker. Those zeros are state sentinels, not near-equality:
the statements that clear a spell, a candidate, or a calibration sum assign
`0.0` exactly. A tolerance would change the spell state machine and the
committed reference fixtures, so no tolerance is applied to these comparisons.

| Compared value | Exact zero means | Assigned exact `0.0` by |
| --- | --- | --- |
| `px3`, read as `x3` inside `_case` | no established spell: `_case` selects the near-normal (larger-magnitude incipient) value, PHDI falls back to the PDSI value, and an incipient index may be promoted | the spell-end assignments, `_statement_190`'s `ppr >= 100` clamp, and the initial `0.0` |
| `px1` / `px2` | no incipient wet/dry index exists to promote | the `px1_computed > 0` / `px2_computed < 0` clamps, the post-promotion and fizzled-abatement resets, and the initial `0.0` |
| `sx1` / `sx2` | the backtrack trail has no candidate from that index at this step | the initial `0.0`: a deferral overwrites a consumed row and `_assign` resets `k8`, so entries above the current trail window are never read |
| `pro` / `ppr`, at 100 and 0 | the spell certainly ends, or no abatement is underway | `ppr` clamped to `100.0` when `>= 100`, reset to `0.0` when a spell ends or an abatement fizzles |
| CAFEC `numerator` / `denominator` | the month accumulated no water-balance term, so the ratio is undefined | the water-balance accumulators' initial `0.0`: `both_zero` when both sums vanish (`1.0` for alpha/beta/gamma, `0.0` for delta), `0.0` when only the denominator does |
| `k8` | no months are pending a spell flush (integer counter, not a float) | the recursion's own counter |

Exact-zero tests outside these sentinels are guards on a computed divisor, not
sentinels, and raise rather than dividing: `DurationFactors.from_fitted`'s
cross-denominator check and `_palmer_wells._abatement_transition`'s `q` guard
(both `ConvergenceError`), and `DurationFactors.weighting_fraction`'s divisor
check (`ValueError`).

## Validation

`tests/test_palmer.py` compares PDSI, PHDI, PMDI, Z-Index, and CAFEC
coefficients against the committed fixture set for climate divisions. The test
module is marked `validation` and uses `atol=5e-5`, `rtol=0`. That fixture set
is regression coverage only: its provenance states the expected outputs were
generated from this library's own Palmer implementation.

For the standard family (`pdsi()`), qualified independent external-product
validation comes from the NOAA NCEI nClimDiv comparison in
`tests/test_nclimdiv_reference.py`: all 344 climate divisions, January 1895
through December 2022, median absolute difference 0.0127 with 86.2% of months
within 0.05. The comparison is qualified because NCEI publishes two decimal
places and the committed precipitation/PET inputs are not byte-identical to
NCEI's operational inputs, which sets a practical floor near 0.013.

`tests/test_scpdsi.py` compares the four self-calibrating index outputs and
fitted duration factors against committed Wells-lineage reference fixtures for
344 climate divisions at the same tolerance. That comparison is independent
implementation cross-validation. The nClimDiv comparison for `scpdsi()` in
`tests/test_nclimdiv_reference.py` is characterization only, because nClimDiv
applies standard Palmer's fixed national calibration; the NCEI Fortran and
Wells C++ recursion lineages disagree on roughly 6.3% of months by up to 8
PDSI units (see `tests/fixture/palmer/provenance.json`).

## Release Decision

Independent external-product validation for the standard Palmer family is
satisfied by the nClimDiv comparison above. scPDSI has independent
implementation cross-validation (Wells oracle) but no qualified
external-product reference yet; assessing DRI/WRCC scPDSI for that role remains
open (issue #780).

## References

- Palmer, W. C. (1965). Meteorological Drought. U.S. Weather Bureau Research
  Paper No. 45.
  https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf
- Heddinghaus, T. R., and Sabol, P. (1991). A review of the Palmer Drought
  Severity Index and where do we go from here? Preprints, 7th Conference on
  Applied Climatology, American Meteorological Society.
- Wells, N., Goddard, S., and Hayes, M. J. (2004). A self-calibrating Palmer
  Drought Severity Index. Journal of Climate, 17, 2335-2351.

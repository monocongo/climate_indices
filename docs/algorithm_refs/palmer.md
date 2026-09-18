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

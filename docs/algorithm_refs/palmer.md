# Palmer Drought Indices

## Scope

The Palmer implementation computes four monthly drought products from
precipitation, PET, and available water capacity:

- Palmer Drought Severity Index (PDSI)
- Palmer Hydrological Drought Index (PHDI)
- Palmer Modified Drought Index (PMDI)
- Palmer Z-Index

The public NumPy API is `climate_indices.palmer.pdsi()`. The v2.5 xarray
notebook demonstrates how to call the NumPy Palmer routine from labeled xarray
inputs and rewrap the outputs with coordinates and metadata.

## Algorithm Notes

The implementation follows the water-balance structure described by Palmer
(1965):

1. Validate precipitation and PET arrays and available water capacity.
2. Initialize monthly water-balance terms.
3. Compute CAFEC coefficients for the calibration period.
4. Compute K weighting factors.
5. Calculate the Z-Index moisture anomaly.
6. Finish the recursive Palmer index calculations for PDSI, PHDI, and PMDI.

## Validation

`tests/test_palmer.py` compares PDSI, PHDI, PMDI, Z-Index, and CAFEC
coefficients against the committed fixture set for climate divisions. The test
module is marked `validation` and uses `atol=5e-5`, `rtol=0`.

The committed fixtures are treated as regression coverage, not independent
scientific validation. Their provenance states that the expected outputs were
generated from this library's Palmer implementation using NCEI nClimDiv-style
inputs. That fixture set is useful for detecting behavior drift, but it is not
an authoritative external reference.

## Release Decision

For v2.5, independent Palmer reference outputs remain a documented validation
gap. The release checklist should either add an independently sourced Palmer
fixture or explicitly defer that item before issue #627 is closed.

## References

- Palmer, W. C. (1965). Meteorological Drought. U.S. Weather Bureau Research
  Paper No. 45.
  https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf
- Heddinghaus, T. R., and Sabol, P. (1991). A review of the Palmer Drought
  Severity Index and where do we go from here? Preprints, 7th Conference on
  Applied Climatology, American Meteorological Society.

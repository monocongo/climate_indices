# Evaporative Demand Drought Index (EDDI)

## Scope

EDDI is a standardized drought index based on evaporative demand. In this
package, EDDI is computed from PET-like evaporative-demand inputs and follows a
non-parametric ranking workflow:

1. Clip physically invalid negative PET values to zero.
2. Aggregate values over the requested time scale.
3. Rank aggregated values within each calendar period of the calibration
   window.
4. Convert empirical probabilities to standardized normal deviates.
5. Clip output to the package-standard range `[-3.09, 3.09]`.

## Public API

Use `climate_indices.eddi()` for the typed public API. It accepts NumPy arrays
and beta xarray `DataArray` inputs. NumPy callers must provide temporal
parameters explicitly; xarray callers can infer them from the time coordinate.

## Validation

The always-on EDDI tests cover shape handling, missing data behavior, clipping,
empirical ranking properties, and xarray metadata preservation. External NOAA
PSL comparison tests live in `tests/test_noaa_eddi_reference.py` and are marked
`validation`.

The NOAA comparison tests require local fixture directories:

- `tests/fixture/noaa-eddi-1month/`
- `tests/fixture/noaa-eddi-3month/`
- `tests/fixture/noaa-eddi-6month/`

When those fixtures are absent, the tests skip with a message that points to
`scripts/prepare_noaa_eddi_fixtures.py`. This is a documented release gap until
paired PET input and NOAA-computed EDDI outputs are committed or otherwise made
available to CI.

## Tolerance

NOAA reference comparisons use `rtol=1e-5` and `atol=1e-5`. The tolerance is
looser than SPI/SPEI because EDDI is rank based and small differences in
empirical probability handling can move the final z-score slightly.

## References

- Hobbins, M. T., Wood, A., McEvoy, D. J., Huntington, J. L., Morton, C.,
  Anderson, M., and Hain, C. (2016). The Evaporative Demand Drought Index.
  Part I: Linking Drought Evolution to Variations in Evaporative Demand.
  Journal of Hydrometeorology, 17, 1745-1761.
  https://doi.org/10.1175/JHM-D-15-0121.1
- NOAA Physical Sciences Laboratory EDDI resources:
  https://psl.noaa.gov/eddi/
- NOAA PSL EDDI archive:
  https://downloads.psl.noaa.gov/Projects/EDDI/

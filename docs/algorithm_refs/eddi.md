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

The committed NOAA comparison fixtures contain paired monthly reference ET and
EDDI at 1-, 3-, and 6-month Timescales for latitude 39.75–39.875 and longitude
-105.0–-104.875, using the 1979–2023 table baseline:

- `tests/fixture/noaa-eddi-1month/`
- `tests/fixture/noaa-eddi-3month/`
- `tests/fixture/noaa-eddi-6month/`

They are generated from the NOAA PSL EDDI time-series table by
`scripts/prepare_noaa_eddi_fixtures.py` and run in CI.

## Tolerance

NOAA reference comparisons use `rtol=1e-5` and `atol=1e-5`. The committed
fixture comparison's maximum observed error is `2.43e-6`, below both tolerances.

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

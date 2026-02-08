# Reference Dataset Provenance

## Overview

The `.npy` files in this directory serve as scientifically validated reference outputs for
testing SPI (Standardized Precipitation Index) and SPEI (Standardized Precipitation
Evapotranspiration Index) computations. These files originated from the library's initial
development to match NOAA's reference SPI implementation.

## Source Data

- **Origin**: NOAA climate division records
- **Input variables**: Monthly precipitation (mm) and potential evapotranspiration (mm)
- **Temporal coverage**: January 1895 through December 2017
- **Total record**: 123 years, 1476 months
- **Location**: Latitude 25.2292° (used for PET computation)
- **PET method**: Thornthwaite (1948)

## Reference Files

### SPI Directory (`spi/`)

| File | Description |
|------|-------------|
| `precips_mm_monthly.npy` | Input precipitation data (mm), shape (1476,) |
| `spi_01_gamma.npy` | SPI 1-month scale, gamma distribution, full-period calibration |
| `spi_06_gamma.npy` | SPI 6-month scale, gamma distribution, full-period calibration |
| `spi_06_pearson3.npy` | SPI 6-month scale, Pearson Type III, 1981-2010 calibration |

### SPEI Directory (`spei/`)

| File | Description |
|------|-------------|
| `precips_mm_monthly.npy` | Input precipitation data (mm), shape (1476,) — same as SPI input |
| `pet_thornthwaite.npy` | Potential evapotranspiration (mm), Thornthwaite method, shape (1476,) |
| `spei_06_gamma.npy` | SPEI 6-month scale, gamma distribution, full-period calibration |
| `spei_06_pearson3.npy` | SPEI 6-month scale, Pearson Type III, full-period calibration |

## Calibration Periods

- **SPI gamma**: Full period (1895-2017)
- **SPI Pearson Type III**: 1981-2010 (standard WMO climate normal period)
- **SPEI gamma**: Full period (1895-2017)
- **SPEI Pearson Type III**: Full period (1895-2017)

Note: Unlike SPI, SPEI Pearson uses full-period calibration in the reference implementation.

## Methodology References

- **SPI**: McKee, T.B., Doesken, N.J., and Kleist, J. (1993). "The relationship of drought
  frequency and duration to time scales." Proceedings of the 8th Conference on Applied
  Climatology, 17-22.

- **SPEI**: Vicente-Serrano, S.M., Beguería, S., and López-Moreno, J.I. (2010). "A
  multiscalar drought index sensitive to global warming: the standardized precipitation
  evapotranspiration index." Journal of Climate, 23(7), 1696-1718.

- **Thornthwaite PET**: Thornthwaite, C.W. (1948). "An approach toward a rational
  classification of climate." Geographical Review, 38(1), 55-94.

## Validation Tolerance

Test tolerance is set to **1e-5** as specified in functional requirement FR-TEST-004.
This tolerance accounts for:
- Floating-point arithmetic variations across platforms
- Minor differences in numerical optimization convergence (L-moments fitting)
- NumPy version differences in statistical functions

Note: SPEI Pearson Type III may require relaxed tolerance (~1e-3) due to inherent
numerical sensitivity in Pearson L-moments fitting.

## File Format

Binary NumPy array format (`.npy`) is used instead of CSV to:
- Preserve full float64 precision without string conversion artifacts
- Enable direct loading with `np.load()` without parsing overhead
- Avoid platform-specific line ending issues
- Keep files compact (~12KB each vs ~50KB+ as CSV)

## Reproduction

To verify these files match the original `tests/fixture/` sources:

```bash
# check file sizes match
ls -l tests/fixture/{precips_mm_monthly,spi_01_gamma,spi_06_gamma,spi_06_pearson3}.npy
ls -l tests/data/spi/*.npy

ls -l tests/fixture/{precips_mm_monthly,pet_thornthwaite,spei_06_gamma,spei_06_pearson3}.npy
ls -l tests/data/spei/*.npy

# verify binary equality (no output = identical)
cmp tests/fixture/precips_mm_monthly.npy tests/data/spi/precips_mm_monthly.npy
cmp tests/fixture/spi_01_gamma.npy tests/data/spi/spi_01_gamma.npy
cmp tests/fixture/spi_06_gamma.npy tests/data/spi/spi_06_gamma.npy
cmp tests/fixture/spi_06_pearson3.npy tests/data/spi/spi_06_pearson3.npy

cmp tests/fixture/precips_mm_monthly.npy tests/data/spei/precips_mm_monthly.npy
cmp tests/fixture/pet_thornthwaite.npy tests/data/spei/pet_thornthwaite.npy
cmp tests/fixture/spei_06_gamma.npy tests/data/spei/spei_06_gamma.npy
cmp tests/fixture/spei_06_pearson3.npy tests/data/spei/spei_06_pearson3.npy
```

## Usage in Tests

See `tests/test_reference_validation.py` for the validation test suite that uses these
reference datasets to validate both the NumPy computation path (`climate_indices.indices`)
and the xarray computation path (`climate_indices.typed_public_api`).

## Last Updated

2026-02-08 — Files copied from `tests/fixture/` during Story 4.4 implementation

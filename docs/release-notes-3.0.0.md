# climate_indices 3.0.0 release notes

Version 3.0.0 expands the index families, gridded workflows, and scientific
validation. Review the [full changelog](https://github.com/monocongo/climate_indices/blob/main/CHANGELOG.md) and
[API migration guide](deprecations/api-changes.md) before upgrading from 2.4.0.

## Highlights

- **Fire-weather indices:** The new `climate_indices.fire` namespace includes
  KBDI, CFFWIS moisture and behavior indices, Fosberg FFWI, Hot-Dry-Windy, and
  Haines. KBDI, CFFWIS, Hot-Dry-Windy, and Haines have xarray adapters; KBDI is
  available from the CLI.
- **Flood-potential indices:** The new `climate_indices.flood` namespace provides
  effective precipitation, the Effective Drought Index (EDI), the Flood Index,
  and the Antecedent Precipitation Index (API), with NumPy, beta xarray, and CLI
  paths. These describe flood *potential*, not observed flooding. EDI uses
  effective precipitation; it is **not** the Evaporative Demand Drought Index
  (EDDI), which measures evaporative demand (`climate_indices.eddi`). The
  flood-family function is `climate_indices.flood.edi` (also exported as
  `climate_indices.edi`). The project overview, flood guide, and core vocabulary
  now make this distinction explicit (#1259). External numeric oracles for the
  flood family remain a validation gap.
- **Self-calibrated Palmer:** `scpdsi()` adds Wells-style self-calibration and a
  CLI path. Standard `pdsi()` also has a gridded xarray adapter; `scpdsi()` is
  NumPy-only.
- **More statistical options:** SPEI accepts the log-logistic distribution;
  `indices.standardized_index()` generalizes the non-negative series fitting
  pipeline to inputs such as runoff or streamflow. SPI can choose how zeros are
  placed with `zero_handling`. SPI, SPEI, and `standardized_index()` can return
  normal scores, probabilities, or bounded values via `output_scale`. The new
  `fit_diagnostics()` reports per-period fit parameters and goodness of fit.
- **More public tools:** FAO-56 Penman-Monteith PET is available from the package
  root, and `climate_indices.runs` identifies threshold runs. The public
  `climate_indices.validation` namespace collects shared validation checks.
- **Gridded execution:** SPI, SPEI, EDDI, percentage of normal, PET, and PDSI
  process declared time-major `(time, *cells)` blocks. On the documented
  reference grid, measured in-process speedups include 4.4x SPI, 3.6x SPEI,
  53x Thornthwaite PET, 342x EDDI, and 111.3x PDSI. The xarray DataArray API
  remains **Beta** through 3.0.0; the NumPy API is the stable integration
  surface.
- **Evidence and documentation:** `VALIDATION.md` distinguishes external
  scientific checks from regression coverage and known gaps; documentation
  uses MyST Markdown with runnable xarray/Zarr and Dask examples. Python
  3.10–3.14 is supported.

## Upgrade considerations

Six behavior or exception changes need review:

1. Daily xarray SPI, SPEI, EDDI, percentage of normal, and Hargreaves PET
   correct the non-leap-year calendar shift; values may change. Unsupported
   calendars or daily series not beginning January 1 now fail validation.
2. Ambiguous three-or-more-dimensional NumPy grids must declare
   `spatial_time_major=True` or reorder their cell axes. Undeclared gridded
   EDDI and percentage-of-normal inputs now raise `DataShapeError` (a changed
   exception type for percentage of normal).
3. `pci()` corrects February boundaries; recompute values and any thresholds
   calibrated against earlier output.
4. Invalid periodicity raises `PeriodicityError`, not `ValueError`. Catch
   `PeriodicityError` or its parent `InvalidArgumentError`.
5. Gamma fitting now measures the probability of zero on the Calibration
   Period's non-missing observations and honors supplied `prob_zero` values;
   affected SPI, SPEI, or `standardized_index()` results can change.
6. Reversed Calibration Periods now raise `CalibrationPeriodError`;
   `percentage_of_normal()` also rejects windows outside its record. Review
   windows and exception handlers before upgrading.

The legacy `spi` console script and `climate_indices.__spi__` are removed:
use `climate_indices --index spi`. The `--save_params` and `--load_params`
options have no direct CLI replacement; pass fitted parameters to
`indices.spi(fitting_params=...)`. Percentage of normal now prefers
`calibration_year_initial`/`calibration_year_final`; the old keyword names
warn and remain available until 4.0.0.

See the [changelog](https://github.com/monocongo/climate_indices/blob/main/CHANGELOG.md) for exact conditions, affected entry
points, other fixes, and references, and the
[migration guide](deprecations/api-changes.md) for detection and remedies.

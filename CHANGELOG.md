# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-09-18

### Added

- **Wildfire index family (`climate_indices.fire`)**: a public namespace with the
  Keetch-Byram Drought Index (KBDI), the CFFWIS moisture codes (FFMC, DMC, DC) and
  behavior indices (ISI, BUI, `cffwis_fwi`, DSR) including Drought Code overwintering
  and seasonal carry, the Fosberg Fire Weather Index, the Hot-Dry-Windy Index, and the
  Haines Index. KBDI, the CFFWIS orchestrator, Hot-Dry-Windy, and Haines ship xarray
  adapters; the elementwise Fosberg FFWI stays on the NumPy layer, where it has no
  dimension to reduce and so no adapter; every index has a CF metadata registry entry.
  `climate_indices --index kbdi` runs KBDI from the command line.
- **Self-calibrated Palmer Drought Severity Index** (`scpdsi`): duration-factor
  fitting, order-statistic self-calibration, and correlation-adaptive least-squares
  fitting, exposed from the Palmer CLI dispatch (#721).
- **Palmer xarray adapter**: `pdsi()` accepts xarray DataArrays and returns the
  PDSI-family outputs as a Dataset; `scpdsi()` remains NumPy-only (#1016).
- **Validation infrastructure**: `VALIDATION.md` records per-index evidence and known
  gaps, backed by committed external fixtures: NOAA PSL EDDI reference series (maximum
  observed error `2.43e-6`), SPEIbase v2.11 SPEI plausibility floors (#779), NOAA NCEI
  climate-divisional SPI characterization across all 344 divisions, the nClimDiv
  standard-Palmer comparison (median absolute difference 0.0127), Wells-lineage scPDSI
  oracle fixtures (`atol=5e-5`), the NRCan CFFWIS reference (`atol=1e-9`), the Keetch
  & Byram (1968) KBDI Figure 1 record, the Srock et al. (2018) HDW case study, and PET
  literature worked examples. `pytest -m validation` runs the external checks.
- **Public `climate_indices.validation` namespace**: the input-preparation and
  validation checks shared across indices are exported from the package root and
  documented as public API.
- **Declared time-major grid blocks**: a three-or-more-dimensional NumPy array runs as
  a `(time, *cells)` block across SPI, SPEI, EDDI, percentage of normal, PET, PDSI, and
  `compute.prepare_scaled()` when declared with `spatial_time_major=True`, fitting or
  ranking every cell in one pass instead of one call per cell (#941, #942).
- **Zero handling for SPI and `standardized_index()`**: a keyword-only `zero_handling`
  chooses where a zero accumulation is placed within the probability of zero `p0`:
  `"classic"` (the default and the existing score, `Φ⁻¹(p0)`), `"center_of_mass"`
  (`Φ⁻¹(p0 / 2)`, Stagge et al., 2015), or `"mean_zero"` (`−φ(Φ⁻¹(p0)) / p0`, Allen
  and Otero, 2024). It is accepted by `indices.spi()`, `indices.standardized_index()`,
  the package-root `spi()`, `compute.fit_and_standardize()`,
  `compute.transform_fitted_gamma()`, and `compute.transform_fitted_pearson()` for
  series, legacy 2-D, and declared time-major input. A Pearson Type III fit that falls
  back to gamma applies the same mode, a moved zero is still clipped to `[-3.09, 3.09]`,
  and SPEI and EDDI do not take it. The xarray `spi()` rejects an unknown mode when it
  is called, Dask-backed input included. `compute.transform_fitted_gamma()` also gains a
  `probabilities_of_zero` argument, the gamma counterpart of the Pearson transform's,
  and a supplied gamma `alpha` or `beta` that does not broadcast to the values raises
  `ValueError` rather than an `IndexError` from NumPy. CF metadata and the CLI
  flag landed in #1187, and the zero-placement tests and guidance in #1188
  (ADR-0015, #1186).
- **`CalibrationPeriodClampedWarning`**: the gamma, Pearson Type III and log-logistic
  fits (`spi()`, `spei()`, `standardized_index()` and the `compute` fitting and transform
  functions) warn when the record does not cover the requested Calibration Period and
  the fit uses other years. The warning carries `requested_years` and `effective_years`.
  Results are unchanged (#1050).

### Changed

- **The xarray DataArray API stays Beta through 3.0.0** and is promoted no earlier
  than 3.1.0, so the interface may still change in a minor release (ADR-0012).
  Computation results remain identical to the stable NumPy API.
- **Gridded vectorization**: the xarray adapter's per-cell Python loop is replaced by
  one NumPy kernel call per `(time, *cells)` block for SPI, SPEI, EDDI, percentage of
  normal, PET, and PDSI. On the 38x87-cell, 40-year reference grid, serial in-process
  speedups are 4.4x (SPI), 3.6x (SPEI), 53x (Thornthwaite PET), 342x (EDDI), and
  111.3x (gridded PDSI, 135.4 s to 1.2 s); computing the SPI goodness-of-fit K-S
  statistic directly adds ~9.2x on its own. Gamma SPI, EDDI, and percentage of normal
  are bit-for-bit identical to the serial NumPy API, and Thornthwaite PET and the
  Pearson Type III fit are asserted at `atol=1e-12` by
  `tests/test_numerical_equivalence.py`. Reproducible harnesses and committed
  before/after artifacts live under `benchmarks/` (#818, #921-#944, #1017).
- **Python 3.14 support**: `requires-python` is now `>=3.10,<3.15`, and CI covers the
  new version in the test matrix and in wheel smoke tests.
- **Documentation rebuilt on MyST Markdown** with a four-section reader-need
  navigation, replacing the previous reStructuredText sources. The xarray/Zarr
  end-to-end workflow is documented and smoke-executed: canonical calculation path,
  reproducible inputs, persisted results reopened with complete metadata, maps and
  selectable-location time series, and the xarray/Zarr, Palmer, EDDI, and Zarr/Dask
  notebooks.
- **Test suite**: the pattern-compliance source-grep suite is retired, static-data,
  xarray-metadata, and logging suites are table-driven, the CF-metadata contract and
  notebook execution have single owners, and a real CLI end-to-end QA suite replaces
  the mocked Palmer orchestration tests. Palmer oracle sweeps are cached per validation
  session, `assert_type()` checks are enforced in CI, and assertions are no longer
  swallowed by `try/except` (#909-#920, #935, #938, #1031, #1033).
- **CI**: meta checks run outside the version matrix on the minimum supported Python
  (3.10), validation, lint, docs, notebooks, and the security audit each run once on a
  single pinned interpreter instead of validation re-running across every version, and
  the release workflow installs the built wheel on the boundary Pythons.
- **CLI**: Palmer inputs are converted to the inches `palmer.pdsi()` expects, the PET
  stage now runs for temperature-only SPEI, scaled, and Palmer runs, and `--scales` is
  now required for every scaled index (#1002). `"auto"` chunk axes resolve against
  a 100 MB array chunk budget while the input is opened, and a caller-configured
  `array.chunk-size` is honored rather than overwritten (#925).
- **CLI unconsumed arguments**: each index registration declares the arguments it
  consumes, and a flag provided to an index that does not consume it is now rejected
  with an error naming the flag and the index instead of being ignored. KBDI and the
  flood indices' hand-maintained exclusion lists are replaced by that one declaration,
  and the parser is checked against the registrations (#1225).

### Breaking

Six changes alter computed values or exception types without a deprecation period, and
the console script removed below reaches users without one either: its deprecation
warning was added during the 3.0.0 cycle and shipped in no release. Each behavioral
change states what a user sees, how to detect it, and what to change in
`docs/deprecations/api-changes.md`.

- **Daily xarray calendar alignment**: `spi()`, `spei()`, `eddi()`,
  `percentage_of_normal()`, and `xarray_adapter.pet_hargreaves()` may return
  **different, corrected** daily values for any input spanning a non-leap year. The
  adapter previously passed daily Gregorian values straight into the NumPy core's
  366-day-per-year layout, silently shifting every value after February 28. Inputs on
  a supported Gregorian calendar (`standard`, `gregorian`, `proleptic_gregorian`)
  whose daily coordinates begin on January 1 still succeed, now with corrected
  numbers; unsupported calendars and other origins raise `CoordinateValidationError`
  instead of being silently misinterpreted. The NumPy array API is unaffected
  (ADR-0004).
- **NumPy gridded input shape guard**: `indices.spi()`, `indices.spei()`, and
  `compute.prepare_scaled()` read a three-or-more-dimensional NumPy array as a
  time-major `(time, *cells)` block, and reject the one shape that is ambiguous with a
  `(years, periods, *cells)` array unless it is declared. Reorder the cell axes so the
  first one is not a calendar period length, or pass `spatial_time_major=True`, which
  the package-root `spi()` and `spei()` wrappers accept and pass through (ADR-0009).
  `indices.eddi()` and `indices.percentage_of_normal()` reject every undeclared
  three-or-more-dimensional input with `DataShapeError` rather than `ValueError`. For
  `percentage_of_normal()` that changes the exception type: 2.4.0 passed the 3-D array
  to `np.convolve` and failed there with numpy's `ValueError`, so a handler catching
  `ValueError` around it no longer matches. `indices.eddi()` already raised
  `DataShapeError` in 2.4.0, so only its declared-block path is new.
- **PCI February correction**: `pci()` returns different values. The cumulative
  day-of-year boundaries had February written as a month length (28 or 29) instead of
  its cumulative index, which left the February slice empty and let March absorb
  February while starting three days early in a non-leap year and two in a leap year.
  The value moves whenever any of those final three January days (the final two in a
  leap year) is non-zero, or when February and March both carry rainfall, so recompute
  PCI values and recalibrate any downstream thresholds tuned against the previous ones
  (#846).
- **Periodicity validation type**: the periodicity check shared by
  `compute.prepare_scaled()`, `compute.transform_fitted_gamma`,
  `compute.transform_fitted_pearson`, `compute.gamma_parameters()`, and
  `compute.reshape_values()`, along with the index functions, now raises
  `PeriodicityError` — a subclass of `InvalidArgumentError`, and not of `ValueError`.
  Error messages are unchanged, so a handler that catches `ValueError` must be widened
  to catch `PeriodicityError` or its parent `InvalidArgumentError` from
  `climate_indices.exceptions`.
- **Gamma probability of zero from the calibration period**: the gamma transform
  counted zeros over every year of the record, with missing years in the denominator,
  while its shape and scale came from the calibration period. It now divides the
  calibration period's zero count by its non-missing count, as the Pearson Type III fit
  already did. Classic gamma SPI and `standardized_index()`, gamma SPEI, and SPI whose
  Pearson fit falls back to gamma can return different values:
  - at a step with a zero (for SPEI, an exact zero in the offset P − PET series) when
    the calibration period is shorter than the record, or holds a missing value, which
    includes the NaN padding of a record that ends partway through a year and the
    leading `scale − 1` values of a scaled series;
  - everywhere, zeros or not, when a gamma `fitting_params` already carried `prob_zero`
    or `probabilities_of_zero`: gamma ignored the key, which now sets the zero mass
    and must lie in `[0, 1]` and match the values' cells;
  - at the zeros of a step with no calibration data, which are now NaN rather than an
    extreme drought.

  A full-record calibration of complete years, at scale 1, with no missing values and
  no `prob_zero` key is bit-identical, and the NOAA and SPEIbase comparisons are
  unchanged. `fit_diagnostics()` reports the same calibration-period `prob_zero` for
  gamma and returns it in `parameters`, and `compute.transform_fitted_gamma()` reads a
  masked entry as missing (ADR-0015, #1186).
- **Calibration Period windows the record cannot represent**: a reversed window (start
  year after end year) now raises `CalibrationPeriodError`, which is both an
  `InvalidArgumentError` and a `ValueError`, in every index, and `percentage_of_normal()`
  now rejects any window its record does not cover with the same error. The gamma and
  Pearson fits (`spi()`, `spei()`, `standardized_index()`, `fit_diagnostics()`) still
  clamp a window that is not reversed; EDDI and Palmer already rejected both. Before:
  - a reversed window gave all-NaN when it lay inside the record, and fit some other
    rows of the record when it ended before it (#1231);
  - `percentage_of_normal()` gave all-NaN for a reversed window or one that starts after
    the record, and averaged only the years that exist for a window that ends past the
    data but is short enough to pass its length check (#1230);
  - `spi()`, `spei()`, `standardized_index()`, `eddi()` and `percentage_of_normal()`
    returned an all-missing input before checking its window, so a reversed window (and,
    for `eddi()` and `percentage_of_normal()`, one the record does not cover) passed
    unnoticed. The window is now checked first, and a window that passes still returns the
    missing input unchanged. Palmer, `fit_diagnostics()` and the flood indices already
    checked it;
  - a complete `fitting_params` set skipped the window check inside the gamma, Pearson
    and log-logistic transforms, so `spi()`, `spei()` and `standardized_index()`
    transformed values for a reversed window. They now check it before the transform,
    and a window that is not reversed but is not covered still clamps.

  Two more `percentage_of_normal()` changes follow from measuring the window in years
  rather than 12 steps per year: a window longer than a daily record is now rejected (a
  4-year window on a 3-year daily record used to pass), and a window that ends in the
  record's partial final year is now accepted (2000-2010 on 121 monthly values used to
  raise). A window the record covers returns the same numbers, and the `error_type` of
  its logged `calculation_failed` event is now `CalibrationPeriodError`.

### Removed

- **`spi` console script**: removed, along with the `climate_indices.__spi__` module.
  Use `climate_indices --index spi` instead. The deprecation warning was added during
  the 3.0.0 cycle and shipped in no release, so no released version warned before the
  removal; 2.4.0 is the last release that ships the script. The `--save_params` and
  `--load_params` options are retired rather than migrated: fit parameters once with
  `compute.gamma_parameters()` or `compute.pearson_parameters()` and pass them as a
  dict — `{"alpha": ..., "beta": ...}` for gamma — to the `fitting_params` argument of
  `indices.spi()` (#919, #957).
- **`compute.adjust_calibration_years()`**: removed. It was undocumented and used
  only inside `compute`; window resolution moved to `climate_indices._calibration_period`
  (#1214).

### Fixed

- **xarray PET and PCI provenance and alignment**: `pet_hargreaves` and
  `pet_penman_monteith` now reject a `tmin`/`tmax` (and other time-series) pair whose
  non-time coordinates differ with `CoordinateValidationError`, as SPEI already did,
  instead of silently intersecting them and dropping cells; time steps outside the
  shared range are still trimmed with an `InputAlignmentWarning`, and an empty time
  intersection now reports the reason `empty_intersection_after_alignment`. Thornthwaite,
  Hargreaves, and Penman-Monteith output no longer inherits the input's `standard_name`
  (for example `air_temperature`), and `pci` output keeps the input's `history` and
  other attributes, appending its own entry in the shared format (#1218).
- **Palmer**: duration-factor overrides resolve after input validation, masked fitting
  parameters are treated as missing, fitted coefficients are required to be 12-element
  vectors, mismatched cell grids are rejected, and per-cell available water capacity
  pairing is covered by tests (#721, #906, #1016).
- **Fire**: KBDI percentile window alignment, the Haines xarray adapter's warnings and
  dtype guard, and the demo manifest input guard (#810, #811).
- **CLI**: copied chunk sizes are reordered to match the output dimensions and the
  `h5netcdf` engine is pinned when writing chunked NetCDF output, so `--chunksizes`
  writes the requested layout under the minimum-dependency environment instead of
  failing; shared arrays are reset per invocation. The `spi` deprecation migration
  guidance was corrected (#919).
- **CLI `--chunksizes input`**: the option was a silent no-op with the `h5netcdf`
  backend, and a copied chunk size larger than the dimension being written was dropped by
  the writer, so it was ignored again for inputs carrying an unlimited dimension. Copied
  input chunk sizes are now trimmed to the shape actually written, with a warning when
  one is reduced, across the single-output, Palmer, and KBDI writers, and the input file
  list is de-duplicated without reordering, so which variable's chunks are copied no
  longer depends on the per-process string hash seed (#1080, #1081, #1084). The eligibility,
  dimension-match, reorder, and trim rules are documented under `--chunksizes`.
- **CLI daily input**: a daily input whose coordinates start mid-year or contain a gap is
  now rejected with the documented daily contract, where it previously restored the input
  length and wrote silently shifted values. Daily climate-division input is converted with
  the same calendar plan the grid transport uses instead of being written back
  unconverted, and time-free companions such as a division latitude are skipped.
- **`utils` daily calendar plan**: `transform_to_366day` raises `ValueError` rather than
  `DataShapeError` for input longer than its declared span, an empty array is rejected
  instead of being accepted as a zero-length partial year, and `DailyCalendarPlan` and
  `from_year_span` validate their contract on construction rather than clamping a
  negative or over-capacity length.
- **CLI input layout**: the shared-array transport accepts only the time-last orders its
  kernels index — `("division", "time")` for climate divisions, time-last for grids — so
  a time-major `("time", "division")` or `("time", "lat", "lon")` input is rejected
  instead of being standardized along the wrong axis and written out with wrong values
  (#902, #1063). Companion PET and temperature variables are checked against the same
  orders as the precipitation variable, so an input the run cannot read fails before any
  handler runs rather than after earlier variables were copied, and every variable's
  dimensions are confirmed before any of them is copied. The accepted-layout list in that error is
  built from the layout classifier itself, so it names the `("time",)` time-series order
  and cannot drift again, and the companion-dimensions message is well-formed.
- **Typed public API**: the package-root `eddi()` overloads now declare
  `spatial_time_major`, which the runtime already forwarded, so a typed caller can pass
  the keyword the documentation describes.
- **SPI with `Distribution.pearson` on masked or sparse input**: a block with half or more
  of its cells missing (an ocean mask) came back entirely NaN, where the per-cell API
  returned finite values, and a single series that was more than half missing did the
  same. The Pearson-to-gamma fall-back counted input that was already missing as a
  failed fit, and then fitted gamma to the Pearson result instead of the scaled input.
  It now judges only the values the fit lost and refits the scaled input, so block,
  chunked, and per-cell results agree on such grids wherever Pearson fits every valid
  cell (#1118).
- **xarray adapter gridded calibration check**: an eager, in-memory grid was checked
  against a single sampled cell at index `[0, ...]`, so a grid whose first cell was
  masked (an ocean corner) was rejected even though every other cell had enough
  calibration data, and a grid whose time dimension wasn't axis 0 crashed with an
  `IndexError` instead of being checked. The same grid run through Dask skipped the
  check entirely and computed, so eager and Dask disagreed. The whole-grid sample is
  dropped: a gridded input no longer runs this preflight check, matching Dask.
  Neither path enforces a per-cell 30-year non-NaN minimum, so sparse cells can
  still produce finite results. The in-memory single-series (1-D) check is
  unchanged (#979, #1156).
- **Calibration Period past the end of the record**: gamma and Pearson Type III
  fitting (`spi()`, `standardized_index()`, `fit_diagnostics()` and the `compute`
  fitting and transform functions) took a record's last year to be one year later than
  it is. A window running past the data was replaced by that phantom-extended record,
  and a window ending exactly one year past the data was kept as requested; either way
  the phantom year counted toward the 30-year minimum, so `ShortCalibrationWarning`
  never fired for a record shorter than that. Windows now resolve against the true last
  year, so the warning counts real years and fires. Results are unchanged: a window
  ending exactly one year past the data still keeps its start and is cut at the last
  year, and any other window the record does not cover still falls back to the whole
  record.
- **Calibration Period resolution has one owner**: `climate_indices._calibration_period`
  resolves the window for those fits, EDDI and Palmer, and each names whether it clamps
  or rejects a window the record does not cover. EDDI and Palmer still reject such a
  window, now with one shared `CalibrationPeriodError` that is both an
  `InvalidArgumentError` and a `ValueError`, so existing `except` clauses keep working.
  The message wording is shared, EDDI no longer logs a separate error line before
  raising, and the `error_type` of the logged `calculation_failed` event is now
  `CalibrationPeriodError` for both. Reversed windows and percentage of normal's
  windows are settled separately, under Breaking (#1214, #1230, #1231).
- **Fits report the Calibration Period they used**: when the record does not cover the
  requested window, the fit clamps it, but the fit's `distribution_fitting_completed`
  log event and the xarray result's `calibration_year_initial` and
  `calibration_year_final` attributes kept naming the requested years, so the years
  behind the fitted parameters could not be told from the output. They now name the
  years the fit used. The single-series sample-size preflight (`InsufficientDataError`)
  reads those years too: a window the record only partly covered was counted on its
  overlap with the record rather than on the whole record the fit uses, and a window
  with no overlap at all was rejected as containing no data, where the fit clamps it
  (#1050).
- **Pearson-to-gamma fall back**: `spi()` and `standardized_index()` treated any
  argument error as a failed Pearson Type III fit and silently returned gamma values.
  A partial or mis-shaped `fitting_params` set is now rejected before the fit, and
  raises the same `ValueError` in every index regardless of `fallback_to_gamma`, where
  it previously raised only on the SPEI path; only a genuine fit failure falls back.
  The shape check also runs for all-missing input, which previously returned before
  it (#1233).
  `fit_diagnostics()` no longer reports a fall back for an argument error, a `-W error`
  run no longer changes which distribution is fitted (the warning propagates instead),
  and the swap is reported through the public `distribution_fallback` log event rather
  than a private strategy method (#1215).
- **CLI output metadata and writes have one owner**: both CLI backends now write
  through `climate_indices._cli_output`, which sources variable attributes from
  `CF_METADATA`, stamps the library version and a history entry, and writes to a
  temporary file that replaces the target only once the whole file is on disk. The
  shared-array route (SPI, SPEI, percent of normal, Thornthwaite PET, and the Palmer
  outputs) stops hand-writing long names, units and valid ranges, so its long names,
  units and references now match the xarray adapters and the registry: PET units are
  `mm/month` rather than `millimeters`, percent of normal is `Percent of Normal
  Precipitation` rather than `Percentage of Normal Precipitation, N-month`, and the
  Palmer Z-Index uses the `z_index` registry entry. A self-calibrated PDSI registry
  entry (`scpdsi`) is added. Every CLI index now normalizes temperature and
  precipitation units through `climate_indices._units`, so spellings such as `degC`
  are accepted by every index, and the `mm/month` PET output is accepted as input to
  a subsequent monthly run; a per-day rate is rejected for a monthly depth rather
  than used as a monthly total. The `_OUTPUT_SCALE_ATTRS` and `_OUTPUT_SCALE_LABELS`
  maps and the hand-written long-name construction are removed (#1223).
- **`pdsi()` duration-factor override validation**: an override was validated with the
  Wells-lineage coefficients even though the standard PDSI recursion computes only
  `c = b / (m + b)`, so a factor set the recursion could use was rejected when the
  unrelated Wells cross coefficient `dryc = 1 - drym / (drym + wetb)` was not a
  contraction -- for example `{"wetm": 1.0, "wetb": -0.4, "drym": 0.3, "dryb": 1.0}`.
  The override is now validated against the coefficients the standard recursion
  actually uses. The standard PDSI spell recursion also takes a precomputed Z-index
  series and its own `PdiDurationFactors`, like the scPDSI Wells recursion, instead
  of recomputing the Z-index mid-recursion. Default-factor PDSI and scPDSI results are
  unchanged (#1226).
- **A fitted distribution has one owner**: `compute.FittedDistribution` now resolves a
  gamma, Pearson Type III or log-logistic fit from the data or from a caller's
  `fitting_params` in one place -- key normalization, partial-set rejection,
  period-to-cell broadcasting and the gamma probability of zero included -- and the
  transforms, `fit_and_standardize()`, the Pearson-to-gamma fall back and
  `fit_diagnostics()` are compositions over it. The three period-to-cell broadcast
  helpers and the duplicated partial-parameter check are gone, the goodness-of-fit
  warnings and the fit diagnostics report the Kolmogorov-Smirnov statistic through one
  implementation, and `fit_diagnostics()` decides a fall back from the fit outcome
  rather than discarding a standardized-value transform. A gamma fit now resolves its
  probability of zero and its missing-data quality check from the same calibration
  values, so a zero counts as valid there, matching `gamma_parameters()`. Public
  signatures and fitted values are unchanged (#1216).
- **CLI shared-array route accepts either dimension order**: the shared-memory route
  (SPI, SPEI, percent of normal, Thornthwaite PET, and the Palmer outputs) canonicalizes
  every time-carrying input to the time-last order its kernels index, so a CF-typical
  `(time, lat, lon)` grid or `(time, division)` division variable is accepted alongside
  the time-last order, and precipitation in one order can be paired with a companion in
  the other. This supersedes the time-last-only acceptance noted above: a time-major
  input, or a time-major companion beside a time-last precipitation variable, was
  previously rejected with `Invalid dimensions ...` (#1224, #932).

## [2.4.0] - 2026-04-05

### Added

- **Penman-Monteith ETo (FAO-56)**: New module `src/climate_indices/pm_eto.py` implementing
  the full FAO-56 reference evapotranspiration equation with atmospheric, vapor pressure, and
  humidity pathway helpers. Validated against FAO-56 worked examples (±0.05 mm/day tolerance).
- **EDDI public API**: `eddi()` now exported from `climate_indices` with `@overload` signatures
  in `typed_public_api.py` for both NumPy and xarray DataArray inputs (beta xarray path).
- **CF metadata registry expansion**: Added entries for `eddi`, `pdsi`, `phdi`, `pmdi`, and
  `z_index` to `cf_metadata_registry.py` (registry now has 12 entries).

- **NFR-PATTERN-COVERAGE**: 42/42 compliance points achieved — all 7 indices (SPI, SPEI,
  PET Thornthwaite, PET Hargreaves, PNP, PCI, Palmer) satisfy all 6 canonical patterns
  (CF metadata, typed public API overloads, xarray adapter, structlog lifecycle,
  structured exceptions, property-based tests). Validated by `tests/test_pattern_compliance.py`.
- **xarray DataArray API (Beta)**: Native xarray support for `spi()`, `spei()`,
  `pet_thornthwaite()`, and `pet_hargreaves()` — marked as beta/experimental.
  The xarray interface (parameter inference, metadata, coordinate handling) may
  change in future minor releases. Computation results are identical to the
  stable NumPy API. No breaking changes in patch versions.
- **`BetaFeatureWarning`**: New warning class for beta/experimental features
  (subclass of `ClimateIndicesWarning`)
- **`ClimateIndicesDeprecationWarning`**: New warning class for deprecated features with dual
  inheritance from both `ClimateIndicesWarning` and `DeprecationWarning`, enabling
  filterability by either category. Includes context attributes for deprecation version,
  removal version, alternative, and migration URL
- **`emit_deprecation_warning()`**: Helper function for standardized deprecation messages with
  automatic URL construction and consistent formatting
- **Docker Support**: Dockerfile for containerized deployment (#586)
- **`.dockerignore`**: Optimized Docker builds by excluding unnecessary files
- **PyPI Release Guide**: Comprehensive release documentation (`docs/pypi_release_guide.md`, `docs/pypi_release.rst`)
- **Floating Point Best Practices Guide**: Documentation for safe numerical comparisons (`docs/floating_point_best_practices.md`)
- **Test Fixture Management Guide**: Documentation for test data management (`docs/test_fixture_management.md`)
- **Visualization Notebook**: New notebook for precipitation/SPI visualization (`notebooks/visualize_precip_spi.ipynb`)
- **Lock File**: Added `uv.lock` for reproducible dependency resolution
- **Documentation**: Supported Python versions table and deprecation policy in README

### Changed

- **CI/CD**: Enhanced test matrix with Python 3.10-3.13 on Linux and macOS
- **CI/CD**: Added minimum dependency version testing (`--resolution lowest-direct`)
- **CI/CD**: Added ruff and mypy checks as CI lint job
- **CI/CD**: Modernized all GitHub Actions to v4/v5 versions
- **GitHub Actions**: Updated unit tests workflow with improved configuration
- **Documentation Index**: Reorganized Sphinx documentation structure
- **Notebooks**: Improved examples in existing Jupyter notebooks

### Removed

- **`.pypirc`**: Removed from repository (should be user-specific in `~/.pypirc`)

### Notes

- **Palmer xarray wrapper deferred**: `palmer_xarray()` is planned for v2.5.0
  (shipped as 3.0.0) using Pattern C (stack/unpack workaround for xarray Issue #1815,
  see `architecture.md`).
- **NOAA EDDI reference fixtures**: Require manual download; see `tests/fixture/README.md`.
  The `test_noaa_eddi_reference.py` suite skips gracefully when fixtures are absent.

## [2.3.0] - 2026-02-11

### Added

- **xarray DataArray API (Beta)**: Native xarray support for `spi()`, `spei()`,
  `pet_thornthwaite()`, and `pet_hargreaves()` — marked as beta/experimental.
  The xarray interface (parameter inference, metadata, coordinate handling) may
  change in future minor releases. Computation results are identical to the
  stable NumPy API. No breaking changes in patch versions.
- **`BetaFeatureWarning`**: New warning class for beta/experimental features
  (subclass of `ClimateIndicesWarning`)
- **`ClimateIndicesDeprecationWarning`**: New warning class for deprecated features with dual
  inheritance from both `ClimateIndicesWarning` and `DeprecationWarning`, enabling
  filterability by either category. Includes context attributes for deprecation version,
  removal version, alternative, and migration URL
- **`emit_deprecation_warning()`**: Helper function for standardized deprecation messages with
  automatic URL construction and consistent formatting
- **Docker Support**: Dockerfile for containerized deployment (#586)
- **`.dockerignore`**: Optimized Docker builds by excluding unnecessary files
- **PyPI Release Guide**: Comprehensive release documentation (`docs/pypi_release_guide.md`, `docs/pypi_release.rst`)
- **Floating Point Best Practices Guide**: Documentation for safe numerical comparisons (`docs/floating_point_best_practices.md`)
- **Test Fixture Management Guide**: Documentation for test data management (`docs/test_fixture_management.md`)
- **Visualization Notebook**: New notebook for precipitation/SPI visualization (`notebooks/visualize_precip_spi.ipynb`)
- **Lock File**: Added `uv.lock` for reproducible dependency resolution
- **Documentation**: Supported Python versions table and deprecation policy in README

### Changed

- **CI/CD**: Enhanced test matrix with Python 3.10-3.13 on Linux and macOS
- **CI/CD**: Added minimum dependency version testing (`--resolution lowest-direct`)
- **CI/CD**: Added ruff and mypy checks as CI lint job
- **CI/CD**: Modernized all GitHub Actions to v4/v5 versions
- **GitHub Actions**: Updated unit tests workflow with improved configuration
- **Documentation Index**: Reorganized Sphinx documentation structure
- **Notebooks**: Improved examples in existing Jupyter notebooks

### Removed

- **`.pypirc`**: Removed from repository (should be user-specific in `~/.pypirc`)

## [2.2.0] - 2025-08-03

### Added

- **Exception-Based Error Handling**: New robust exception hierarchy for distribution fitting failures
  - `DistributionFittingError` (base class)  
  - `InsufficientDataError` - raised when too few non-zero values for statistical fitting
  - `PearsonFittingError` - raised when L-moments calculation fails
- **Migration Guide**: Comprehensive v2.2.0 migration documentation in README
- **Code Quality Improvements**: Safe floating point comparison guidelines using `numpy.isclose()`
- **Enhanced Test Coverage**: Comprehensive tests for exception handling and fallback behavior
- **Documentation**: 
  - Floating point best practices guide (`docs/floating_point_best_practices.md`)
  - Working examples for safe numerical comparisons
  - Updated build configuration documentation

### Changed

- **Major Dependency Updates**: Updated all packages to latest versions
  - `scipy>=1.15.3` (from 1.14.1) - requires Python 3.10+
  - `dask>=2025.7.0`, `xarray>=2025.6.1`, `h5netcdf>=1.6.3`
  - `pytest>=8.4.1`, `ruff>=0.12.7`, `sphinx>=8.1.3`
- **Build System**: Consolidated and optimized hatch build configuration
  - Reduced package size from 37MB to 207KB (99.4% reduction)
  - Eliminated duplicate exclude lists between sdist and wheel builds
- **Error Handling Architecture**: 
  - Replaced `None` tuple anti-pattern with explicit exceptions
  - Consolidated fallback logic into `DistributionFallbackStrategy` class
  - Improved error messages with detailed context and suggestions
- **Python Version Support**: Dropped Python 3.9, now requires Python 3.10+
- **Floating Point Comparisons**: Replaced direct equality checks with `numpy.isclose()` throughout test suite

### Fixed

- **NumPy 2.0 Compatibility**: 
  - Fixed deprecated `newshape` parameter usage
  - Fixed array-to-scalar conversion warnings
- **Consecutive Zero Precipitation**: Enhanced handling of extensive zero precipitation patterns
- **Test Reliability**: Improved floating point comparison robustness in test assertions
- **Build Exclusions**: Properly exclude development files from distribution packages

### Technical Improvements

- **Code Quality**: Addressed `python:S1244` floating point equality issues
- **Test Architecture**: Enhanced coverage for edge cases and error conditions
- **Logging**: Consistent warning messages for high failure rates in distribution fitting
- **Documentation**: Clear upgrade path for library integrators using internal functions

## [2.1.1] - 2025-01-15

### Added

- **`DistributionFallbackStrategy` class**: Centralized fallback logic for Pearson→Gamma distribution fallbacks
- **Custom exception hierarchy**: `DistributionFittingError`, `InsufficientDataError`, `PearsonFittingError`
- **Comprehensive test coverage**: New test case for distribution fallback strategy consolidation

### Changed

- **Error handling architecture**: Replaced `None` tuple anti-pattern with explicit exception-based error handling
- **Fallback logic**: Consolidated scattered Pearson→Gamma fallback code into single strategy class
- **Logging**: Standardized warning messages for distribution fitting failures

### Fixed

- **Type safety**: Improved error propagation with typed exceptions instead of implicit `None` checks
- **Code maintainability**: Simplified control flow by eliminating complex `None` checking logic

## [2.0.0] - 2023-07-15

### Added

- GitHub Action workflow which performs unit testing on the four supported versions of Python (3.8, 3.9, 3.10, and 3.11)

### Fixed

- L-moments-related errors (#512)
- Various cleanups and formatting indentations

### Changed

- Build and dependency management now using poetry instead of setuptools
- Documentation around installation with examples (#521) 

### Removed

- Palmer indices (these were always half-baked and nobody ever showed any interest in developing them further)
- Numba integration (see [this discussion](https://github.com/monocongo/climate_indices/discussions/502#discussioncomment-6377732)
  for context)
- requirements.txt (dependencies now specified solely in pyproject.toml)
- setup.py (now using poetry as the build tool)

[unreleased]: https://github.com/monocongo/climate_indices/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/monocongo/climate_indices/compare/v2.1.1...v2.2.0
[2.1.1]: https://github.com/monocongo/climate_indices/compare/v2.0.0...v2.1.1
[2.0.0]: https://github.com/monocongo/climate_indices/releases/tag/v2.0.0
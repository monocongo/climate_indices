# Changelog

All notable changes to the `climate-indices` crate are recorded here. The crate
follows its own `0.x` version track, independent of the `climate_indices` Python
distribution, and uses [Semantic Versioning](https://semver.org/).

## 0.1.1 - 2026-10-29

This entry prepares the stable `0.1.1` release; it is not a publication record.
It does not describe an `-rc.N` prerelease for the separate crate rehearsal.
The maintainer must confirm the tag date before publication.
The additions preserve the `0.1.0` public API and its Rust 1.85 minimum.

### Added

- `gamma::gamma_ks_statistics` computes gamma goodness-of-fit D statistics (#1434).
- `empirical::plotting_positions` computes SEI plotting positions for signed series (#1435).
- `runs::identify_runs` identifies threshold runs and computes per-run metrics (#1436).
- `compound::resample_metrics` evaluates paired and independence-null monthly
  bootstrap metrics from caller-supplied resample indices (#1437).
- `fire::initial_spread_index`, `buildup_index`, `fwi`, `daily_severity_rating`,
  and `overwinter_dc` complete the CFFWIS numerical kernels (#1438).
  Fosberg FFWI, HDW, and Haines remain outside the crate (#1439–#1441).
- `aggregation::trailing_sum` and `trailing_mean` provide trailing windows over
  borrowed `(steps, cells)` float64 views, including strided views (#1485).
  Outputs retain the input shape and mark incomplete or missing windows as NaN.
  Independent reductions preserve each window's operation order.
- Criterion cases cover empirical plotting positions for 1 and 32 cells.

### Changed

- Complete-input KBDI uses a table-based drying term and advances eight cells
  at a time (ADR-0025). This path preserves `rtol = atol = 1e-10`,
  not bitwise identity. Other inputs retain the recurrence driver.
- SPI and SPEI examples use the shared aggregation kernel (#1485).
- The README, crate metadata, module overview, and numerical contract describe
  the expanded kernel set (#1442). No kernel changes are part of release preparation.

## 0.1.0 - 2026-10-26

The first stable release. Its code is the same as `0.1.0-rc.1`.

### Added

- Numerical kernels for SPI, SPEI, EDDI, PET, PNP, PCI, the Palmer family,
  fire-weather indices, and flood indices, previously built only into the
  `climate_indices` Python extension.

## 0.1.0-rc.1 - 2026-10-20

A pre-release that rehearses the `0.1.0` release path before the stable version
is spent (see `docs/release-process.md`). Cargo does not select it by default, so
it changes nothing for a user who does not name it.

### Added

- Numerical kernels for SPI, SPEI, EDDI, PET, PNP, PCI, the Palmer family,
  fire-weather indices, and flood indices, previously built only into the
  `climate_indices` Python extension.

## 0.0.0 - 2026-10-08

- Name reservation. Contains no algorithms.

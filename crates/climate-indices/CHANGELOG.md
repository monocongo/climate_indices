# Changelog

All notable changes to the `climate-indices` crate are recorded here. The crate
follows its own `0.x` version track, independent of the `climate_indices` Python
distribution, and uses [Semantic Versioning](https://semver.org/).

Development note (#1485; not included in 0.1.0): `aggregation::trailing_sum` and
`aggregation::trailing_mean` add trailing windows over borrowed `(steps, cells)`
float64 views, including strided views. Each output retains the input shape and
marks incomplete or missing windows as NaN. Each window is reduced independently.
SPI and SPEI examples use the shared kernel. The next release preparation will
assign this change to a versioned entry.

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

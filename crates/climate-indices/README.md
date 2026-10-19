# climate-indices

Pure-Rust numerical kernels for climate and drought indices: the Standardized
Precipitation Index (SPI), the Standardized Precipitation Evapotranspiration Index
(SPEI), the Evaporative Demand Drought Index (EDDI), potential evapotranspiration (PET),
the Palmer family, fire-weather indices, and flood indices.

The crate has no Python, I/O, logging, or metadata dependency. It provides kernels, not
complete index workflows: the caller prepares calendars, selects the calibration period,
validates physical ranges, and applies any aggregation and final index transformation.

## Usage

```toml
[dependencies]
climate-indices = "0.1"
ndarray = "0.17"
```

Array inputs borrow `ndarray` views and array outputs own their data. A daily block has
shape `(days, cells)`, so one series and a grid use the same call:

```rust
use climate_indices::{ClimateError, flood::effective_precipitation};
use ndarray::array;

fn main() -> Result<(), ClimateError> {
    let grid_mm = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
    let effective = effective_precipitation(grid_mm.view(), 2)?;
    assert_eq!(effective.dim(), (3, 2));
    assert_eq!(effective[[1, 1]], 35.0);
    Ok(())
}
```

The crate documentation on docs.rs states the shape, calendar, unit, and error
conventions for each kernel.

## Examples

The crate ships runnable programs that build an index from plain arrays, with no
file input and no orchestration crate:

```bash
cargo run --example spi    # SPI-3 from a monthly precipitation array
cargo run --example spei   # SPEI-6 from precipitation and PET arrays
cargo run --example pet    # Thornthwaite PET from monthly temperatures
```

Each example performs the caller-side steps the kernels document: calendar
preparation, the Timescale accumulation, the selection of the Calibration Period,
the estimate of a discrete zero mass where the distribution has one, and the
standard-normal transform.

## Numerical reference

The kernels originate in the [`climate_indices`](https://github.com/monocongo/climate_indices)
project, whose reference implementations are the numerical parity oracle: the project's
parity suite holds every kernel to `rtol = atol = 1e-10` against them.

## Versioning

The crate follows its own `0.x` version track, independent of the `climate_indices`
distribution. Before 1.0, a minor version can make breaking changes. `ndarray` is
a public dependency, so an incompatible `ndarray` upgrade is a breaking release.

The minimum supported Rust version is 1.85.

## License

BSD-3-Clause. The [`LICENSE`](LICENSE) file includes the third-party notices for the
ported Cephes and NumPy routines.

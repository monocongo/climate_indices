# climate-indices

Pure-Rust numerical kernels cover drought, fire weather, flood, Run Theory,
empirical distributions, compound bootstrap metrics, and trailing-window aggregation.
The drought families include the Standardized Precipitation Index (SPI),
the Standardized Precipitation Evapotranspiration Index (SPEI),
the Evaporative Demand Drought Index (EDDI), potential evapotranspiration (PET),
and the Palmer family.

The crate's only dependency is `ndarray`: no I/O, logging, or metadata layer. It
provides kernels, not
complete index workflows: the caller prepares calendars, selects the calibration period,
validates physical ranges, and applies the final index transformation.
The `aggregation` module provides trailing Timescale sums and means; callers choose
when to apply them and retain all calendar responsibilities.

## Modules

- `aggregation` provides trailing Timescale sums and means.
- `gamma`, `pearson`, `loglogistic`, and `lmoments` provide distribution fits,
  probabilities, and goodness-of-fit statistics.
- `empirical` provides SEI plotting positions for signed climate and energy series.
- `eddi` provides Tukey plotting positions and the Hastings inverse normal.
- `eto` and `pm_eto` provide Thornthwaite, Hargreaves, and Penman-Monteith PET.
- `pnp` and `pci` provide precipitation ratios and concentration indices.
- `palmer`, `palmer_zindex`, `palmer_pdi`, `palmer_wells`, and `self_calibration`
  provide water balance, Z-Index, PDSI, and scPDSI kernels.
- `fire` provides KBDI, CFFWIS moisture codes, ISI, BUI, FWI, DSR, and overwinter Drought Code.
- `flood` provides effective precipitation, EDI, the Flood Index, and the Antecedent Precipitation Index.
- `runs` provides threshold-run identification and per-run metrics.
- `compound` evaluates paired and independence-null bootstrap resample metrics.
  Callers supply the resample indices; the crate generates no random draws.
- `recurrence` provides the shared daily recurrence driver and missing-day policies.
- `special` provides the special functions used by the distribution kernels.

Fosberg FFWI, HDW, and Haines kernels are not included.

## Usage

The following example uses the published `0.1.0` release:

```toml
[dependencies]
climate-indices = "0.1.0"
ndarray = "0.17"
```

The module list above describes this checkout, including additions planned for `0.1.1`.
Until `0.1.1` is published, use a local path dependency for those additions:

```toml
climate-indices = { path = "../climate_indices/crates/climate-indices" }
```

Replace the `climate-indices` dependency under `[dependencies]` with this entry.
Adjust the path relative to your application's `Cargo.toml`.

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

Each example performs the caller-side steps the kernels document. SPI and SPEI
prepare the calendar, accumulate the Timescale window, select the Calibration
Period, estimate a discrete zero mass where the distribution has one, and apply
the standard-normal transform; the PET example prepares temperatures, latitudes,
and leap-year flags for a single kernel call.

## Numerical reference

The kernels originate in the [`climate_indices`](https://github.com/monocongo/climate_indices)
project, whose reference implementations are the numerical parity oracle: the project's
parity suite holds every kernel to `rtol = atol = 1e-10` against them.

## Versioning

The crate follows its own `0.x` version track, independent of the `climate_indices`
distribution. Before 1.0, a minor version can make breaking changes. `ndarray` is
a public dependency, so an incompatible `ndarray` upgrade is a breaking release.

The minimum supported Rust version is 1.85.

## Kernel benchmarks

Run the Criterion suite from the workspace root:

```bash
cargo bench -p climate-indices --bench kernels --locked
# Compile and check every fixture and kernel once, without collecting timings:
cargo bench -p climate-indices --bench kernels --locked -- --test
# Measure one kernel in both cell configurations:
cargo bench -p climate-indices --bench kernels --locked -- gamma_parameters
```

The suite needs only Rust. Criterion is a development dependency; its plotting and
parallel-analysis features are disabled. Version 0.7 supports the crate's Rust 1.85
minimum. Cargo uses the optimized bench profile and the workspace's thin LTO and
single codegen unit. No custom CPU flags are required.

`benches/kernels.rs` generates small deterministic fixtures without files, downloads,
or a random-number dependency. Its base sample is
`((step * 37 + cell * 17 + 11) % 101) / 100.0`. Monthly cases use 40 years
(1980–2019), 12 months, and either 1 or 32 cells. Rain includes zeros; the
log-logistic fit and empirical plotting positions use signed synthetic water-balance values.
Fits and empirical calibration use all 40 years.
Thornthwaite uses positive temperatures, 40° latitude, and Gregorian leap flags.
EDDI ranks only January's PET against its 40-year sample, without leading pads.
Palmer receives rain, PET, and a five-inch available water capacity in inches.
Daily cases use 1,830 consecutive positions and either 1 or 32 cells. FFMC starts
at 85, records all history, and has valid weather with no gaps or seasonal mask.
Effective precipitation uses a 30-day window, whose first 29 outputs are missing.
These synthetic fixtures are measurement inputs, not scientific reference data.

Fixture generation, fit parameters for the probability benchmark, unit conversions,
and output assertions run outside the timed loops. Each timed iteration calls one
kernel with borrowed views and consumes its output through `std::hint::black_box`.
Kernel allocations and output destruction are included. Each recurrence starts
from the same initial state. The smoke command checks fit validity, probability
ranges, empirical plotting positions, PET, water-balance and FFMC outputs, and effective-precipitation padding.

This is a representative subset, not complete workflow coverage: Gamma fit and
probabilities, log-logistic fit, empirical plotting positions, EDDI ranking,
Thornthwaite PET, Palmer water balance, FFMC, and effective precipitation.
It excludes aggregation, runs, compound metrics, other PET, distribution, Palmer,
fire, and flood kernels, final index transformations, I/O, and parallel execution.
It does not measure complete SPI, SPEI, EDDI, or PDSI workflows.

### Representative measurements

Measured on 2026-10-19 with an Apple M5 (10 physical/logical cores), 32 GiB RAM,
macOS 27.0.1 arm64, Rust/Cargo 1.99.0, and Criterion 0.7.0. The suite calls the
kernels serially. The command above used Criterion's defaults: a three-second
warm-up, five-second measurement target, and 100 samples per case. The recorded
run added `-- --noplot`; it used no custom `RUSTFLAGS`.

These measurements predate the empirical benchmark and do not measure that kernel.
Times and throughput below are Criterion's central estimates, not minimum times.
Throughput counts millions of input time-cell values per second (Melem/s), not
output arrays or arithmetic operations. EDDI counts 40 values per cell; the other
monthly kernels count 480. Daily kernels count 1,830, including window padding.

| Kernel | 1 cell, µs/call | 1 cell, Melem/s | 32 cells, µs/call | 32 cells, Melem/s |
| --- | ---: | ---: | ---: | ---: |
| `gamma_parameters` | 1.049 | 457.46 | 76.779 | 200.05 |
| `gamma_probabilities` | 31.092 | 15.44 | 2425.5 | 6.33 |
| `loglogistic_parameters` | 3.929 | 122.17 | 155.45 | 98.81 |
| `thornthwaite` | 9.912 | 48.43 | 238.77 | 64.33 |
| `tukey_probabilities` | 0.464 | 86.27 | 19.538 | 65.52 |
| `water_balance` | 2.588 | 185.49 | 417.12 | 36.82 |
| `ffmc` | 79.187 | 23.11 | 2349.0 | 24.93 |
| `effective_precipitation` | 6.506 | 281.28 | 282.19 | 207.52 |

The [raw output](benches/results/apple-m5.txt) preserves the 95% confidence
intervals and outlier counts. The 32-cell Gamma cases have wide intervals;
repeat measurements before making performance decisions. Criterion also writes
local estimates under `target/criterion/`. These machine-specific results are not
speedup claims or CI thresholds. Re-running reproduces the inputs and protocol,
not identical elapsed times.

## License

The primary license is BSD-3-Clause. The [`LICENSE`](LICENSE) file also retains
the original NOAA public-domain notice and the complete third-party notices and
license texts for Cephes, SciPy/xsf, NumPy, and the Boost.Math Lanczos coefficients.
The Boost.Math coefficients remain under the Boost Software License 1.0.

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
- **Optional Rust backend:** Compiled numerical kernels can accelerate eligible
  calls without changing the public Python API. The retained Python path needs
  no Rust toolchain; gains depend on the workload, and some kernels are slower
  in Rust. See [backend evidence and installation](#optional-rust-backend) below.
- **Evidence and documentation:** `VALIDATION.md` distinguishes external
  scientific checks from regression coverage and known gaps; documentation
  uses MyST Markdown with runnable xarray/Zarr and Dask examples. Python
  3.10–3.14 is supported.

## Optional Rust backend

### What changes, what stays Python

`climate_indices._native` joins the existing Python API as an optional
acceleration backend. `crates/climate-core` contains pure Rust numerics;
`crates/climate-py` supplies the PyO3/NumPy boundary. Ports cover the kernels
behind SPI, SPEI (gamma, Pearson Type III, and log-logistic), the standardized
index, EDDI, PNP, PCI, Thornthwaite, Hargreaves and Penman-Monteith PET,
Palmer/scPDSI, the fire moisture-code and KBDI recurrences, and the flood family.
Validation, Calibration Period resolution, package warnings, xarray/Dask,
CF metadata, provenance, the CLI, and I/O stay Python. Fire behavior indices,
Fosberg FFWI, Hot-Dry-Windy, and Haines remain Python implementations.

The backend introduces no public signature, return-type, or exception change.
Results agree within the parity tolerance below, not necessarily bit for bit.
Python implementations remain available and directly testable. Some prepared
inputs and NumPy/warning configurations retain the Python path even with the
extension installed; importing `_native` does not prove a particular call used
Rust. An unavailable extension (`ImportError`) selects Python; a runtime error
from the extension propagates and is never silently retried. Per-step
`climate_indices.lmoments` failed-fit log records differ on the native path;
failed-fit counts and package warnings remain unchanged.

[ADR-0017](adr/0017-rust-core-acceleration-backend.md) records the architecture;
[the kernel table and dispatch rules](architecture.md#optional-rust-backend)
describe the individual seams and eligibility conditions.

### Numerical evidence

The cross-backend contract is `rtol = atol = 1e-10` with matching NaN positions.
The consolidated registry covers every kernel the extension exposes, with fixed
samples, property-based cases, routing checks, and end-to-end xarray, Dask,
CLI, and diagnostics comparisons. On the committed macOS arm64 fixed-sample
measurement, the largest absolute deviation is **4.547e-13**
(`flood_pe`); the largest relative deviation is **1.868e-13**
(`spi_gamma_spatial_block`). These are measured samples, not error bounds for
all possible inputs or platforms. See the
[per-entry maxima and reproduction command](architecture.md#parity-tolerance).
Backend parity is regression evidence, not independent scientific validation;
[VALIDATION.md](https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md)
records the external-reference evidence and remaining scientific gaps.

### Performance evidence and limits

The merged benchmark harness compares native-enabled dispatch against the
retained Python implementation on identical inputs. Ratios are **Python wall
time / Rust-enabled wall time**: above 1 favors Rust. These are whole index or
helper calls, including Python orchestration and binding/copy costs, not
isolated Rust kernel timings. The native-enabled measurements ignore NumPy
floating-point errors and clear warning filters locally to satisfy the dispatch
guard; they do not describe every caller's default configuration.

Selected fixed-registry results on macOS arm64, Python 3.14.7, best of five
after a warm-up per backend:

| Entry | Python/Rust |
|---|---:|
| Thornthwaite PET (station series) | 219.07 |
| Hargreaves PET | 35.84 |
| Palmer PDSI (station series) | 194.06 |
| scPDSI | 62.45 |
| KBDI | 42.77 |
| Antecedent Precipitation Index | 46.26 |
| Penman-Monteith | 0.88 |
| Penman-Monteith intermediates | 0.45 |
| PCI | 0.97 |
| Fit diagnostics | 1.01 |

Penman-Monteith is slower in Rust on these inputs. PCI and diagnostics are
within 5% of parity; the routine artifact retains only best samples, so those
small differences do not establish a direction. Large station gains are not
grid speedups: the Python spatial paths already vectorize across cells.

The separate Apple M5 CONUS run covers **469,758 land cells × 528 months**
(1981–2024), Python 3.14.7, best of two without warm-up. Single eager-call
ratios are 1.33 (SPI gamma), 1.41 (SPI Pearson Type III), 1.57 (SPEI gamma),
1.70 (EDDI), 1.12 (Thornthwaite), and 2.77 (PDSI). With eight spatial blocks
on one thread, SPI gamma is 0.98 and Thornthwaite 1.03: effectively parity
rather than the station-series gains. Loading and land-cell packing are outside
timing; precipitation zeros are replaced with 0.01 mm, SPEI/PDSI PET is prepared
before timing, and PDSI uses a constant 6-inch AWC, not a measured soil grid.
These are prepared-input compute timings, not raw-grid pipeline timings.

Outer thread pools can help, but eight threads are slower than four for PDSI
and SPI Pearson Type III on that host. Rust kernels remain single-threaded;
Rayon was not adopted. The measurements come from one macOS host per artifact,
with limited repetitions and scheduling noise, not portable speed guarantees.
They are distinct from the earlier NumPy vectorization results in Highlights.
[The benchmark methodology and full tables](https://github.com/monocongo/climate_indices/blob/main/benchmarks/README.md#rust-kernels-vs-the-python-reference-across-the-parity-registry-rust-011)
link the committed routine and CONUS artifacts and reproduction commands.
Isolating binding cost from kernel time, durable timing samples for the AWS
follow-up, and missing daily/full-grid shapes remain work under
[the benchmark ticket](https://github.com/monocongo/climate_indices/issues/1281)
and [AWS follow-up](https://github.com/monocongo/climate_indices/issues/1324);
no unmerged AWS results are used here.

### Installation and platform coverage

For 3.0.0, the release workflow builds five **`cp310-abi3`** binary wheels for
Python 3.10–3.14: Linux x86-64/aarch64 (manylinux_2_28, glibc 2.28 or newer),
macOS arm64/x86-64, and Windows x86-64. A compatible platform wheel takes
precedence over the same-version pure-Python wheel. Other platforms, including
musl/Alpine and older-glibc Linux, use the pure wheel. Both expose the same
public API. Hatchling remains the source-build backend, so an sdist install
also builds pure Python without Rust:

```bash
pip install --no-binary climate_indices climate_indices
```

Check whether the extension is installed:

```bash
python -c "import climate_indices._native"
```

An `ImportError` here means this diagnostic cannot import the optional
extension; ordinary `climate_indices` calls still use Python. Developers who
want it from a checkout run `uv run maturin develop --release` with a Rust
toolchain. Ordinary editable installs are pure Python in a clean checkout, but
`maturin develop` leaves `_native` importable after a later `uv sync`. To return
to Python-only execution, remove the compiled extension and start a new
interpreter; see the
[development guide](development-guide.md#porting-a-kernel-to-rust).
[ADR-0018](adr/0018-optional-rust-packaging.md) covers artifact selection and CI
checks.

## Upgrade considerations

Six behavior or exception changes need review:

1. Daily xarray SPI, SPEI, EDDI, percentage of normal, and Hargreaves PET
   correct the non-leap-year calendar shift; values may change. Unsupported
   calendars, daily series not beginning January 1, or monthly series not
   beginning in January now fail validation.
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

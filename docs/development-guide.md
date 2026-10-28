# Development Guide

## Prerequisites

### Required Software
- **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14
- **uv**: Modern Python package installer ([https://github.com/astral-sh/uv](https://github.com/astral-sh/uv))
- **Git**: Version control
- **Make**: Build automation (optional, for Sphinx docs)

### System Dependencies (for NetCDF support)
```bash
# macOS
brew install hdf5 netcdf

# Ubuntu/Debian
sudo apt-get install libhdf5-dev libnetcdf-dev

# Fedora/RHEL
sudo dnf install hdf5-devel netcdf-devel
```

### Optional: the native Rust extension

Everything in the library runs in pure Python, and `uv sync` installs it that way.
The native extension is an optional accelerator: when it is present, the dispatch
described in [architecture.md](architecture.md#optional-rust-backend) sends prepared
float64 arrays to Rust kernels instead of the Python reference. You do **not** need
Rust to install, import, run, or test `climate_indices`; the native suites
(`tests/test_native_*.py`, covering kernel and PET parity plus the extension smoke
checks) skip without it. A change to a ported kernel is not actually compared
against the Rust path until you build the extension, so build it before claiming
parity. No Rust experience is required: [installing the Rust
toolchain](#install-the-rust-toolchain) below is a one-time, two-command step.

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/monocongo/climate_indices.git
cd climate_indices
```

### 2. Install uv (if not already installed)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### 3. Create Virtual Environment and Install Dependencies
```bash
# Sync all dependencies (core + dev + test)
uv sync --group dev

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### 4. Install in Development Mode
```bash
# Install package in editable mode
uv pip install -e .
```

## Development Workflow

### Running Tests
```bash
# All tests (excludes benchmarks by default)
uv run pytest

# With verbose output
uv run pytest -v

# Specific test file
uv run pytest tests/test_xarray_adapter.py

# Include benchmarks
uv run pytest -m benchmark

# With coverage report
uv run pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Code Quality Checks

#### Linting with Ruff
```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

#### Type Checking with Mypy

Run mypy on Python 3.12, the stack CI's lint job uses. The package-wide `strict`
setting in `pyproject.toml` applies automatically, so no `--strict` flag is
needed.

```bash
# Check the package and the type assertions pytest cannot see
uv run --python 3.12 mypy src/ tests/test_type_checking.py

# Check a specific module
uv run --python 3.12 mypy src/climate_indices/typed_public_api.py
```

The `mypy` commands require a Python 3.11+ dependency stack. The Python 3.10
stack pins numpy to 2.2.x, whose stubs raise findings that no single source tree
can satisfy on both stacks, so the release workflow skips its mypy leg on 3.10.

### Building Documentation

#### Sphinx Documentation
```bash
cd docs
make html
# Open _build/html/index.html
```

#### Clean Build
```bash
cd docs
make clean
make html
```

### Building Distribution Packages
```bash
# Build wheel and sdist
uv run python -m build

# Build a local binary wheel for this platform (needs a Rust toolchain). On Linux
# this is not the published manylinux_2_28 wheel: release.yml builds that inside the
# container named in its matrix, which is what sets the glibc floor (ADR-0018)
uv run maturin build --release --out dist

# Output in dist/
# - climate_indices-X.Y.Z-py3-none-any.whl
# - climate_indices-X.Y.Z.tar.gz
# - climate_indices-X.Y.Z-cp310-abi3-<platform>.whl
```

## Project Structure

```
climate_indices/
├── src/climate_indices/     # Source code
├── tests/                    # Test suite
├── docs/                     # Documentation
├── pyproject.toml            # Build config + dependencies
├── uv.lock                   # Dependency lock file
└── .github/workflows/        # CI/CD pipelines
```

## Development Commands Reference

| Task | Command |
|------|---------|
| **Install dependencies** | `uv sync --group dev` |
| **Run tests** | `uv run pytest` |
| **Run benchmarks** | `uv run pytest -m benchmark` |
| **Coverage report** | `uv run pytest --cov=src --cov-report=html` |
| **Lint code** | `ruff check --fix src/ tests/` |
| **Format code** | `ruff format src/ tests/` |
| **Type check** | `uv run --python 3.12 mypy src/ tests/test_type_checking.py` |
| **Build docs** | `cd docs && make html` |
| **Build package** | `uv run python -m build` |
| **Update deps** | `uv lock` |

## Porting a Kernel to Rust

Adding a Rust kernel means adding a second implementation of a numerical block that
already exists in Python. The Python implementation stays, and it stays the oracle:
the port is only finished when the two agree at `rtol = atol = 1e-10` with identical
NaN positions. The architecture is decided in
[ADR 0017](adr/0017-rust-core-acceleration-backend.md); the crate layout, the
already-ported seams, and the dispatch rules are in
[architecture.md](architecture.md#optional-rust-backend). `crates/climate-indices/src/gamma.rs`
and its dispatch in `src/climate_indices/compute.py` are the reference implementation
of every step below.

### Install the Rust toolchain

Rust is distributed through `rustup`, a toolchain manager that plays the same role
for Rust that `uv` plays for Python. You do not need to pick a version: this
repository's `rust-toolchain.toml` pins the compiler (1.99.0) and the two components
the checks use (`rustfmt`, `clippy`), and `rustup` installs that exact toolchain
automatically the first time you run a Rust command inside the checkout. Install
`rustup` once, then let the checkout pick the version.

```bash
# macOS/Linux (official installer — recommended)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# macOS with Homebrew (alternative). rustup is keg-only, so it is installed but
# not put on PATH: resolve the prefix Homebrew actually used (`/opt/homebrew` on
# Apple Silicon, `/usr/local` on Intel) and add its shim directory to your shell's
# startup file (`~/.zshrc` for zsh, `~/.bash_profile` for bash).
brew install rustup
echo "export PATH=\"$(brew --prefix rustup)/bin:\$PATH\"" >> ~/.zshrc

# Windows (PowerShell)
winget install --id Rustlang.Rustup
```

The official installer places the Rust binaries in `~/.cargo/bin` and adds that
directory to your shell's `PATH`; open a new terminal afterwards (or run
`source "$HOME/.cargo/env"`) so the commands below resolve. If you used Homebrew,
the `echo` line does the same for the Homebrew copy, also taking effect in a new
terminal. This is the step people most often miss: a Homebrew rustup without that
line leaves `cargo` and `rustup` off `PATH`, so every Rust command reports
`command not found` even though the toolchain is installed. On Windows, install the
Microsoft C++ build tools (the Visual Studio installer's "Desktop development with
C++" workload) before building: the MSVC toolchain links against them, and the build
fails at the link step without them.

Verify the install:

```bash
rustup --version
rustup show    # the toolchain in effect for the current directory (1.99.0)
cargo --version
```

`rustup show` lists the toolchains installed and names the one in effect. The first
Rust command you run inside the checkout also downloads the pinned 1.99.0 toolchain
if it is missing; to fetch it explicitly, run
`rustup toolchain install 1.99.0 --component rustfmt --component clippy`.

In this project `cargo` is the Rust build tool and package manager, `rustfmt`
formats Rust code, and `clippy` lints it; `maturin` is the Python bridge that wraps
the compiled crates into an importable extension. `maturin` is already a locked dev
dependency, so `uv sync --group dev` installs it and you run it through `uv`.

Build the extension and confirm it imports, from the repository root:

```bash
uv run maturin develop --release
uv run python -c "import climate_indices._native; print(climate_indices._native.__file__)"
```

`maturin develop` compiles the crates in `crates/`, copies the resulting `_native`
extension into `src/climate_indices/`, and installs the package in editable mode, so
the extension stays importable when `uv sync` later reinstalls the project. The first
build downloads the pinned toolchain and the crates it depends on and can take a few
minutes; later builds are incremental. To return to the pure-Python install, delete
the compiled file and start a new interpreter:

```bash
rm -f src/climate_indices/_native.*.so src/climate_indices/_native.*.pyd
```

The everyday Rust commands, for reference:

| Command | What it does |
|---------|--------------|
| `rustup show` | The toolchain this checkout pins (must be 1.99.0) |
| `rustup update` | Updates installed toolchains — do **not** use this to move the pin; change `channel` in `rust-toolchain.toml` deliberately instead |
| `cargo build` / `cargo test --workspace` | Compiles the crates / runs the crate-level Rust tests; set `PYO3_PYTHON` to the project interpreter, as in [step 8](#8-verify), so they link the right Python |
| `cargo fmt --all -- --check` | Formatting; CI fails on any diff |
| `cargo clippy --workspace --all-targets -- -D warnings` | Lints; CI fails on any warning |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p climate-indices --no-deps` | Crate documentation; CI fails on any rustdoc warning |

### Published-crate consumer smoke test

After each stable crate release and monthly between releases, run the
published-crate consumer smoke test in {doc}`release-process`.
It uses unversioned `cargo add climate-indices` in a fresh project outside the checkout.
It checks the selected stable version, calculates PNP with result assertions, and builds local crate documentation.
Set `EXPECTED_VERSION` to the stable release under test.
This test checks what a Rust user downloads from crates.io; the workspace gates below check repository changes.
It does not replace kernel parity tests, scientific validation, or the docs.rs check.

### Crate quality gates

The published `climate-indices` crate has four required CI gates, one job each, so
each is a required check on the pull request. Run them locally with
`cargo install --locked cargo-deny@0.20.2 cargo-semver-checks@0.51.0` once, then:

| Command | What it proves |
|---------|----------------|
| `cargo package -p climate-indices --locked`, unpack into a directory **outside** this checkout, copy `rust-toolchain.toml` into it, then `cargo clippy --all-targets --all-features --locked -- -D warnings`, `cargo test --all-features --locked`, and `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked` | The crate that a user receives lints, tests, and documents cleanly, on the pinned compiler. Running it from the unpacked `.crate` is what catches a file the crate needs but does not ship. |
| `cargo "+{msrv}" check -p climate-indices --lib --all-features --locked`, where `{msrv}` is the crate's `rust-version` | The library builds on its declared minimum supported Rust version (1.85). `--lib` leaves out `criterion`, which only the benchmarks use. |
| `cargo semver-checks check-release -p climate-indices` | The public API is compatible with the latest version on crates.io. A deliberate break fails here, cited to the offending item. |
| `cargo deny --locked check -W unmaintained -W unsound` | The licenses, advisories, duplicate versions, and sources permitted by `deny.toml` cover everything that reaches a user — the published crate and the `climate-py` bindings the wheel compiles. Unmaintained and unsound advisories are warnings, so an upstream archive notice does not block an unrelated pull request. |

The gates read the crate's own `rust-version` and version, so moving either needs no
workflow change. `cargo semver-checks` compares against the published baseline. The
`0.0.0` placeholder had no API to compare; since `0.1.0` was published, the check has a
real baseline.

CI runs the native legs with `CLIMATE_INDICES_REQUIRE_NATIVE=1`, which turns a
missing extension into a test failure instead of a skip so a broken build cannot
silently drop the parity suite. Set the same variable locally to confirm a change
really exercises the Rust path.

### 1. Trace the Python seam

Read the Python function end to end and name the block that becomes the kernel, then
name everything that stays: validation, calibration-period resolution, data-quality
and goodness-of-fit warnings, the Pearson-to-gamma fallback, logging, zero placement,
support-limit masks, and output scaling. The seam goes where the prepared float64
values are already valid, so the kernel takes no validation decisions and raises no
`climate_indices` exceptions.

Record the semantics the kernel has to reproduce, because these are what the port is
reviewed against: operation order, what NaN means, how zeros are treated, and what a
degenerate column does (all-missing, all-zero, constant, or no positive value).

### 2. Write the kernel documentation block

The `//!` crate/domain header and the `///` block above each public kernel state:

- the Python source it ports, named by function and by the block inside it;
- inputs and outputs, including the array layout (years by columns) and which
  argument is per-column;
- zero semantics, NaN semantics, invalid-input behaviour, short-series and
  degenerate-column behaviour;
- the numerics: the expressions, the summation order, and where SciPy is being
  reproduced.

### 3. Port the kernel

Put the kernel in `crates/climate-indices/src/`, in the module that owns the index
family, and declare that module in `crates/climate-indices/src/lib.rs`
(`pub mod <family>;`) so `climate-py` can reach it by module path. The crate is pure
Rust: no PyO3, no NumPy bindings, no Python exceptions. Where SciPy evaluates a
special function, port the routine SciPy evaluates into `special/` (see the Cephes
ports for `igam`, `igamc`, `ndtri`, `ndtr`, and `lgam`) rather than calling a
general crate — a generic implementation does not hold parity in the tails.
Reproduce the Python arithmetic rather than improving it, and do not add a
fast-math or fused kernel: `[profile.release]` keeps IEEE semantics on purpose.

### 4. Add the binding and the stub

Add the `#[pyfunction]` to `crates/climate-py/src/lib.rs`, converting arrays at the
boundary and returning only `climate-indices` types across it; that crate holds no
algorithm. Register it in the `#[pymodule]` in that file with
`m.add_function(wrap_pyfunction!(<kernel>, m)?)?`: a `#[pyfunction]` that is not
registered is never exposed to Python. Add the matching signature to
`src/climate_indices/_native.pyi`, annotating each array argument with its dtype:
float64 values, bool validity masks, int64 indices. Existing bindings call
`checked_copy`, which rejects unaligned arrays, copies empty and caller-owned
storage before `py.detach`, and wraps the view conversion. Simple elementwise
stages such as PNP percentages and Palmer CAFEC use `checked_view` without releasing
the GIL, avoiding input-sized copies.

### 5. Add dispatch and routing

`climate_indices._native_backend` is the one seam to the extension. It is the only
module that imports `climate_indices._native`, and it owns the kernel-presence check
(`kernels`), the plain, aligned float64 and reporting-policy gate (`plain_float64`,
`policy_allows`, and `takes`, which combines them), and the translation of the
extension's errors (`converging`, `finite_results`). A family module never imports
the extension itself. It copies `_native_backend.EXTENSION` into its own `_native`
attribute, which is that family's switch, and keeps only its eligibility rule and its
marshalling. A new family module also goes into `NATIVE_FAMILIES` in
`tests/conftest.py`; `tests/test_native_backend.py` fails until it does.

The index dispatch helper is named `_native_<kernel>`, lives next to the Python
implementation it ports in `compute.py`, and is reached by its public callers in
`indices.py` as `compute._native_<kernel>`. It returns `None` when a routing
condition fails, and its caller then runs the Python path (as
`_native_gamma_probabilities` and `_native_gamma_parameters` do). The fire and flood
recurrences use a different pattern: `fire/_native.py` and `flood/_native.py` define
functions such as `moisture_code_recurrence` and `api_recurrence`, which return a
callable for the shared daily runner rather than a computed result. Dispatch takes
the Rust path only when every routing condition holds:

- the family's switch is set (`_native is not None`) and every operand is a plain,
aligned float64 `ndarray` whose layout the kernel supports — in `compute.py`, use the
existing `_native_float64` / `_native_lmoment_input` guards rather than writing new
ones;
- prepared arguments the kernel accepts, e.g. parameters that are one per calendar
step (and cell); caller-supplied parameters that vary by year stay in Python;
- NumPy floating-point errors ignored (`np.errstate(all="ignore")`): warnings,
exceptions, callbacks, logging, or printing keep the Python path, because the
kernels do not implement NumPy's reporting policies;
- the extension exports the kernel: probe with
`_native_backend.kernels(_native, "<kernel>")`, so an extension built before the
port falls back to Python. Probe after the other routing checks, just before the
kernel call. `NativeRecorder` records kernel calls, not availability probes.

A kernel is never retried in Python after a Rust runtime error: the exception
propagates, or `_native_backend.converging` or `finite_results` re-raises it as the
error the Python path raises. A missing extension is not an error, it is the
pure-Python install.

### 6. Write the parity tests

Extend the suite that covers the index family — `tests/test_native_parity.py` for the
SPI/EDDI kernels, `tests/test_native_parity_distributions.py` for the distribution
fits, `tests/test_native_parity_fire.py` for the fire recurrences,
`tests/test_native_parity_palmer.py` for the Palmer family — running the same
computation twice through the public or compute-level API: once with `_native`
replaced by the `_Recorder` around the extension, once with `_native` set to `None`,
comparing at `rtol = atol = 1e-10` with matching NaN positions. The compute suites
patch `compute._native`; the fire suite patches `climate_indices.fire._native._native`,
and the Palmer suite patches `palmer._native`. `conftest.disable_native(monkeypatch,
*families)` points the named families, or every family when none is named, at their
Python implementations.
Tests that establish invocation assert the recorded call set, so a run that never
reached Rust fails instead of passing Python against Python; result-only comparisons
discard the call set, and intentional-fallback cases assert an empty one.
`NativeRecorder` records kernel calls, not the availability probes that
`_native_backend.kernels` makes. If an extension lacks a kernel, dispatch falls back
to Python without recording a call to that kernel. A test that expects that call
fails. Rebuild the extension after adding a kernel. Cover the edge cases that the
doc block promises: all-missing and all-zero columns, a constant column, supplied
parameters, and masked or unaligned inputs that must fall back.

A test that targets the Python reference itself uses the `python_backend` fixture
from `tests/conftest.py`, which sets the `_native` attribute of every dispatch module
(`compute`, `eto`, `pm_eto`, `fire`, `flood`, `palmer`, and `compound._bootstrap`) to `None` for the test; a new
dispatch module is added to that fixture in the same change. Loosening a tolerance needs a measured justification in the
ticket (maximum absolute and relative error, where it occurs, which primitive
diverges, and whether reproducing the Python method closes the gap); never edit a
fixture or a reference test to match Rust.

### 7. Regenerate the documentation bundles

If the change touched a file listed in `SUMMARY_FILES` or `FULL_FILES` in
`scripts/generate_llms_txt.py` — this guide and `docs/architecture.md` among them —
regenerate and commit the bundles before verifying, because the verification run
below includes the bundle-sync test:

```bash
uv run scripts/generate_llms_txt.py
uv run pytest tests/test_review_scripts.py
```

### 8. Verify

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p climate-indices --no-deps
PYO3_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" cargo test --workspace
uv run maturin develop --release
uv run python -c "import climate_indices._native"
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run --python 3.12 mypy src/ tests/test_type_checking.py
CLIMATE_INDICES_REQUIRE_NATIVE=1 uv run pytest -n auto   # with the extension built
uv run pytest -m validation -n auto  # external reference suites, per the RUST ticket
rm -f src/climate_indices/_native.*.so src/climate_indices/_native.*.pyd && uv run pytest -n auto   # pure-Python fallback
uv run --extra docs sphinx-build -E -b html -W --keep-going docs docs/_build/html
uv run --extra docs sphinx-build -E -b doctest docs docs/_build/doctest
```

The fallback run matters as much as the native one: nothing user-facing may require
the extension. The native run above sets `CLIMATE_INDICES_REQUIRE_NATIVE=1`, as CI's
native legs do, so a missing or stale extension fails instead of silently skipping
every parity test.

## Coding Standards

### Python Style
- **Line length**: 120 characters
- **Python version**: 3.10 minimum (avoid syntax and stdlib added after 3.10)
- **Type hints**: Required for all functions
- **Docstrings**: Google-style for all public functions
- **Imports**: stdlib → third-party → local (enforced by ruff)

### Import Order Example
```python
# Standard library
from __future__ import annotations
import os
from typing import Optional

# Third-party
import numpy as np
import xarray as xr

# Local
from climate_indices import compute, exceptions
```

### Testing Standards
- **Framework**: pytest with fixtures
- **Coverage target**: >90%
- **Property-based tests**: Use hypothesis for invariants
- **Markers**: Use `@pytest.mark.benchmark` for performance tests

### Icechunk I/O tests

The optional Icechunk integration tests live in `tests/test_icechunk_io.py` and are
marked `icechunk`. They cover round-trip fidelity, SPI and SPEI parity across Icechunk,
NetCDF, and Zarr inputs, session and snapshot semantics, and eager, threaded, and
distributed writes. Install the extra and run just that suite:

```bash
uv sync --extra icechunk   # Python 3.11 or later; `--dev` installs it too
uv run pytest -m icechunk
```

Without the extra (Python 3.10, or a base install) the module skips. Every CI matrix leg
on Python 3.11 and later installs the `dev` group, so those legs run the suite already;
the marker is for selecting it locally.

One test is opt-in in a second way: `test_r2_round_trip` writes to the shared Cloudflare
R2 bucket, so it is marked `validation` and excluded from the default selection. It needs
both an explicit switch and credentials in the environment:

```bash
R2_TEST=1 R2_ACCOUNT_ID=<account> R2_ACCESS_KEY_ID=<key> R2_SECRET_ACCESS_KEY=<secret> \
  uv run pytest -m validation tests/test_icechunk_io.py
```

It creates a fresh `R2_PREFIX/pytest-<id>` repository (default prefix `examples/icechunk`,
default bucket `climate-indices`) and leaves that snapshot for inspection, so it never
touches another object in the bucket. Delete the prefix when it is no longer needed.

### Git Workflow
1. Create feature branch from `main`
2. Make changes with descriptive commits
3. Run tests locally (`uv run pytest`)
4. Push and create pull request
5. CI runs full test matrix
6. Merge after review

## Troubleshooting

### Common Issues

#### NetCDF Import Errors
```bash
# Install system dependencies
brew install hdf5 netcdf  # macOS
sudo apt-get install libhdf5-dev libnetcdf-dev  # Ubuntu
```

#### uv Not Found
```bash
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"  # Add to ~/.bashrc or ~/.zshrc
```

#### Test Failures
```bash
# Clear pytest cache
pytest --cache-clear

# Reinstall dependencies
uv sync --group dev --force
```

---

See [CONTRIBUTING.md](https://github.com/monocongo/climate_indices/blob/main/CONTRIBUTING.md) for detailed contribution guidelines.

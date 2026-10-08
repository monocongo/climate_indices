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
```bash
# Check all source files
mypy src/

# Check specific module
mypy src/climate_indices/typed_public_api.py

# Strict mode (enforced on typed_public_api.py)
mypy --strict src/climate_indices/typed_public_api.py
```

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
| **Type check** | `mypy src/` |
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
[architecture.md](architecture.md#optional-rust-backend). `crates/climate-core/src/gamma.rs`
and its dispatch in `src/climate_indices/compute.py` are the reference implementation
of every step below.

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

Put the kernel in `crates/climate-core/src/`, in the module that owns the index
family, and declare that module in `crates/climate-core/src/lib.rs`
(`pub mod <family>;`) so `climate-py` can reach it by module path. The crate is pure
Rust: no PyO3, no NumPy bindings, no Python exceptions. Where SciPy evaluates a
special function, port the routine SciPy evaluates into `special/` (see the Cephes
ports for `igam`, `ndtri`, `ndtr`, and `lgam`) rather than calling a general crate —
a generic implementation does not hold parity in the tails. Reproduce the Python
arithmetic rather than improving it, and do not add a fast-math or fused kernel:
`[profile.release]` keeps IEEE semantics on purpose.

### 4. Add the binding and the stub

Add the `#[pyfunction]` to `crates/climate-py/src/lib.rs`, converting arrays at the
boundary and returning only `climate-core` types across it; that crate holds no
algorithm. Register it in the `#[pymodule]` in that file with
`m.add_function(wrap_pyfunction!(<kernel>, m)?)?`: a `#[pyfunction]` that is not
registered is never exposed to Python. Add the matching signature to `src/climate_indices/_native.pyi`, where
every array argument is documented as float64. Existing bindings call
`checked_copy`, which rejects unaligned arrays, copies empty and caller-owned
storage before `py.detach`, and wraps the view conversion.

### 5. Add dispatch and routing

The dispatch helper lives in the Python module that owns the computation, next to
its Python implementation, and is named `_native_<kernel>`. Two caller patterns are
in use: a helper returns `None` when a routing condition fails, and its caller runs
the Python path (as `_native_gamma_probabilities` does), or the helper carries no
guard of its own and is called only after the caller's `_native_float64` guard has
passed (as `_native_gamma_parameters` does). Dispatch takes the Rust path only when
every routing condition holds:

- the extension imported (`_native is not None`);
- a plain, aligned float64 `ndarray` whose layout the kernel supports — use the
existing `_native_float64` / `_native_lmoment_input` guards rather than writing new
ones;
- prepared arguments the kernel accepts, e.g. parameters that are one per calendar
step (and cell); caller-supplied parameters that vary by year stay in Python;
- NumPy floating-point errors ignored (`np.errstate(all="ignore")`): warnings,
exceptions, callbacks, logging, or printing keep the Python path, because the
kernels do not implement NumPy's reporting policies.

A kernel is never retried in Python after a Rust runtime error: the exception
propagates. A missing extension is not an error, it is the pure-Python install.

### 6. Write the parity tests

Extend the suite that covers the index family — `tests/test_native_parity.py` for the
SPI/EDDI kernels, `tests/test_native_parity_distributions.py` for the distribution
fits — running the same computation twice through the public or compute-level API:
once with `compute._native` replaced by the `_Recorder` around the extension, once
with `_native` set to `None`, comparing at `rtol = atol = 1e-10` with matching NaN
positions. The recorder asserts the Rust path was actually reached, so the
comparison is never Python against Python. Cover the edge cases the doc block
promises: all-missing and all-zero columns, a constant column, supplied parameters,
and masked or unaligned inputs that must fall back.

A test that targets the Python reference itself uses the `python_backend` fixture
from `tests/conftest.py`, which pins `compute._native` to `None` for the test.
Loosening a tolerance needs a measured justification in the ticket (maximum absolute
and relative error, where it occurs, which primitive diverges, and whether
reproducing the Python method closes the gap); never edit a fixture or a reference
test to match Rust.

### 7. Verify

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
PYO3_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" cargo test --workspace
uv run maturin develop --release
uv run python -c "import climate_indices._native"
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/ tests/test_type_checking.py
uv run pytest -n auto                # with the extension built
uv run pytest -m validation -n auto  # external reference suites, per the RUST ticket
rm -f src/climate_indices/_native.*.so src/climate_indices/_native.*.pyd && uv run pytest -n auto   # pure-Python fallback
uv run --extra docs sphinx-build -E -b html -W --keep-going docs docs/_build/html
uv run --extra docs sphinx-build -E -b doctest docs docs/_build/doctest
```

The fallback run matters as much as the native one: nothing user-facing may require
the extension. CI adds `CLIMATE_INDICES_REQUIRE_NATIVE=1` on its native legs, which
turns a missing extension into a failure instead of a skip.

### 8. Regenerate the documentation bundles

If the change touched a file listed in `SUMMARY_FILES` or `FULL_FILES` in
`scripts/generate_llms_txt.py` — this guide and `docs/architecture.md` among them —
regenerate and commit the bundles alongside it:

```bash
uv run scripts/generate_llms_txt.py
uv run pytest tests/test_review_scripts.py
```

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

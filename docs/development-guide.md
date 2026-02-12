# Development Guide

## Prerequisites

### Required Software
- **Python**: 3.10, 3.11, 3.12, or 3.13
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

# Output in dist/
# - climate_indices-2.2.0-py3-none-any.whl
# - climate_indices-2.2.0.tar.gz
```

## Project Structure

```
climate_indices/
├── src/climate_indices/     # Source code (14 modules)
├── tests/                    # Test suite (26 test files)
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

## Coding Standards

### Python Style
- **Line length**: 120 characters
- **Python version**: 3.10+ (no 3.14+ features)
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
1. Create feature branch from `master`
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

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

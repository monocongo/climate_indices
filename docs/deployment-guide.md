# Deployment Guide

## Docker Deployment

### Building the Container
```bash
docker build -t climate_indices:2.2.0 .
```

### Running the Container
```bash
docker run -v $(pwd)/data:/data climate_indices:2.2.0 \
    --index spi \
    --periodicity monthly \
    --scales 6 \
    --netcdf_precip /data/precip.nc \
    --var_name_precip prcp \
    --calibration_start_year 1981 \
    --calibration_end_year 2010 \
    --output_file_base /data/spi
```

### Dockerfile Overview
```dockerfile
# Multi-stage build
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN uv sync --frozen --no-dev

FROM python:3.11-slim
RUN apt-get update && apt-get install -y libhdf5-dev libnetcdf-dev
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
USER climate  # Non-root user
ENTRYPOINT ["python", "-m", "climate_indices"]
```

## CI/CD Pipelines

### 1. Unit Tests Workflow (`unit-tests-workflow.yml`)

**Trigger**: Push to any branch, pull requests

**Matrix**:
- Python versions: 3.10, 3.11, 3.12, 3.13
- OS: ubuntu-latest

**Steps**:
1. Checkout code
2. Setup Python and uv
3. Install dependencies: `uv sync --group dev`
4. Run tests: `pytest`
5. Upload coverage to Codecov

**Configuration**:
```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12', '3.13']
```

### 2. Release Workflow (`release.yml`)

**Trigger**: Git tag push (vX.Y.Z)

**Steps**:
1. Checkout code
2. Setup Python and build tools
3. Build distribution: `python -m build`
   - Wheel: `climate_indices-X.Y.Z-py3-none-any.whl`
   - Source dist: `climate_indices-X.Y.Z.tar.gz`
4. Publish to PyPI using `PYPI_API_TOKEN`
5. Create GitHub release with artifacts

**Release Process**:
```bash
# 1. Update version in pyproject.toml
# 2. Commit and tag
git add pyproject.toml
git commit -m "Release v2.2.0"
git tag v2.2.0
git push origin master --tags

# 3. GitHub Actions automatically:
#    - Builds packages
#    - Uploads to PyPI
#    - Creates GitHub release
```

### 3. Benchmarks Workflow (`benchmarks.yml`)

**Trigger**: Manual dispatch, scheduled (weekly)

**Steps**:
1. Checkout code
2. Setup Python and uv
3. Install dependencies with performance group: `uv sync --group dev --group performance`
4. Run benchmarks: `pytest -m benchmark --benchmark-json`
5. Compare against baseline
6. Store results as artifact

**Benchmark Results**:
- Stored in GitHub Actions artifacts
- JSON format for historical tracking
- Regression detection via comparison

## PyPI Distribution

### Package Metadata
- **Name**: `climate_indices`
- **Version**: Defined in `pyproject.toml`
- **License**: BSD 3-Clause
- **Python**: >=3.10,<3.14

### Installation
```bash
# From PyPI
pip install climate_indices

# With uv
uv pip install climate_indices

# Specific version
pip install climate_indices==2.2.0
```

### Package Contents
**Included**:
- `src/climate_indices/` - All source code
- `LICENSE` - BSD 3-Clause license
- `README.md` - Package description

**Excluded** (defined in `pyproject.toml`):
- `tests/` - Test suite
- `docs/` - Documentation source
- `notebooks/` - Jupyter notebooks
- `.github/` - CI/CD workflows
- `.venv/` - Virtual environment
- Cache directories

### Build Configuration
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
exclude = ["tests/", "docs/", "notebooks/", ...]
```

## ReadTheDocs Deployment

### Documentation Hosting
- **URL**: [https://climate-indices.readthedocs.io/](https://climate-indices.readthedocs.io/)
- **Source**: `docs/*.rst` (Sphinx)
- **Build**: Automatic on push to `master`
- **Versions**: Tracks git tags

### Build Configuration
```python
# docs/conf.py
project = "climate_indices"
version = "2.2"  # Major.minor
release = "2.2.0"  # Full version

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]
```

## Dependency Management

### Lockfile (uv.lock)
- **Purpose**: Reproducible builds
- **Update**: `uv lock` (run after pyproject.toml changes)
- **Commit**: Yes, commit to repository

### Dependency Groups
```toml
[project.dependencies]
# Core runtime dependencies
scipy>=1.15.3
xarray>=2025.6.1
dask>=2025.7.0
structlog>=24.1.0

[dependency-groups]
dev = [
    "pytest>=8.4.1",
    "ruff>=0.12.7",
    "mypy",
    ...
]
```

## Monitoring and Observability

### Structured Logging
```python
from climate_indices import configure_logging
import logging

configure_logging(
    log_level=logging.INFO,
    log_file="production.log",
    json_logs=True  # For log aggregators
)
```

### Performance Metrics
- Execution time tracking via `@measure_execution_time`
- Memory usage monitoring
- Logged in structured format

### Coverage Tracking
- **Service**: Codecov
- **Target**: >90% line coverage
- **Badge**: Displayed on GitHub README

---

See [architecture.md](./architecture.md) for system design and [development-guide.md](./development-guide.md) for local development setup.

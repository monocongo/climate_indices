# Development Workflow

## Environment Setup

```bash
# Clone and setup
git clone https://github.com/monocongo/climate_indices.git
cd climate_indices

# Create virtual environment and install dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

## Build, Test, and Development Commands

### Essential Commands
```bash
# Lint (Ruff)
ruff check --fix src/ tests/

# Format (Ruff)
ruff format src/ tests/

# Type-check
uv run mypy src/climate_indices

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_indices.py -q

# Run with coverage
uv run pytest --cov=climate_indices --cov-report=term
```

### Definitive Check Command (Run Before Commits)
```bash
ruff check --fix src/ tests/ && ruff format src/ tests/ && uv run mypy src/climate_indices && uv run pytest -q
```

### Pre-commit Hooks
```bash
# First time setup
uv run pre-commit install

# Run all hooks manually
uv run pre-commit run --all-files
```

## CLI Testing

```bash
# SPI computation (serial)
uv run climate_indices --index spi --periodicity monthly --scales 3 \
  --netcdf_precip precip.nc --var_name_precip prcp \
  --calibration_start_year 1981 --calibration_end_year 2010 \
  --output_file_base output/spi

# SPI computation (parallel with Dask)
uv run climate_indices --index spi --periodicity monthly --scales 3 \
  --netcdf_precip precip.nc --var_name_precip prcp \
  --calibration_start_year 1981 --calibration_end_year 2010 \
  --output_file_base output/spi --multiprocessing all
```

## Commit & Pull Request Guidelines

### Commits
- Clear, imperative subject line (e.g., "Add Hargreaves PET calculation")
- Concise body explaining "what" and "why"
- Group logical changes; avoid mixing refactors with behavior changes
- Reference issues when applicable (e.g., "Fixes #123")

### Pull Requests
- Include summary, rationale, and links to issues
- Ensure: lint, type-check, and tests pass locally
- No secrets in diffs
- CI must be green before merge

## Testing Standards

- **Framework**: pytest with comprehensive fixtures
- **Structure**: Tests in top-level `tests/` directory
- **Naming**: Test files as `test_*.py`, functions as `test_*()`
- **Fixtures**: Defined in `tests/conftest.py`
- **Scope**: Session-scoped fixtures for expensive operations
- **Parameterization**: Use `@pytest.mark.parametrize` for multiple test cases
- **Coverage**: Aim for >= 80% on core computational code

## Documentation

```bash
# Build Sphinx docs
cd docs && make html

# Clean docs
cd docs && make clean

# View locally
open docs/_build/html/index.html
```

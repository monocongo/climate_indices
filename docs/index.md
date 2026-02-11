# Climate Indices Documentation Index

**AI Entry Point for `climate_indices` Project**

## Project Quick Reference

| Attribute | Value |
|-----------|-------|
| **Project Name** | climate_indices |
| **Version** | 2.2.0 |
| **Type** | Python Library (Scientific Computing) |
| **Architecture** | Layered Monolith |
| **Python Support** | 3.10, 3.11, 3.12, 3.13 |
| **License** | BSD 3-Clause |
| **Repository** | [github.com/monocongo/climate_indices](https://github.com/monocongo/climate_indices) |
| **Documentation** | [ReadTheDocs](https://climate-indices.readthedocs.io/) |
| **PyPI** | [pypi.org/project/climate-indices](https://pypi.org/project/climate-indices/) |

## Purpose

The **climate_indices** library provides reference implementations of climate indices used for drought monitoring: SPI, SPEI, PET (Thornthwaite & Hargreaves), Palmer Drought Indices, and Percentage of Normal Precipitation. It supports both numpy arrays and xarray DataArrays, with Dask integration for parallel computation.

## Project Classification

- **Repository Structure**: Monolith (single cohesive codebase)
- **Primary Language**: Python 3.10+
- **Build System**: Hatchling (PEP 517) + uv for dependency management
- **Architecture Pattern**: Layered Library (CLI → Public API → Computation → Math → Infrastructure)
- **Entry Points**: 3 CLI commands (`climate_indices`, `process_climate_indices`, `spi`)
- **Test Coverage**: >90% (26 test files with 1005-line fixture system)

## Documentation Structure

### AI-Readable Documentation (BMAD Format)

This documentation set is optimized for AI agents and provides comprehensive technical context:

1. **[project-overview.md](./project-overview.md)** 🔴 CRITICAL START HERE
   - Executive summary and project status
   - Project classification and architecture overview
   - Quick reference (installation, API usage, CLI usage)
   - Design philosophy and use cases
   - AI development guidance and architectural invariants

2. **[architecture.md](./architecture.md)** 🔴 CRITICAL
   - Technical architecture documentation
   - Layered library design pattern
   - Technology stack and dependencies
   - Data flow and computation patterns
   - Testing architecture and CI/CD pipelines
   - Key design decisions and trade-offs

3. **[source-tree-analysis.md](./source-tree-analysis.md)** 🟡 HIGH PRIORITY
   - Annotated directory structure
   - Critical file locations and priorities
   - Module dependency graph
   - Entry points for common development tasks

4. **[component-inventory.md](./component-inventory.md)** 🟡 HIGH PRIORITY
   - Comprehensive catalog of all 14 modules
   - Function-level documentation
   - Dependencies and usage patterns
   - Code examples for each component

5. **[development-guide.md](./development-guide.md)** 🟢 STANDARD
   - Prerequisites and installation
   - Development workflow
   - Testing and code quality commands
   - Coding standards and troubleshooting

6. **[deployment-guide.md](./deployment-guide.md)** 🟢 STANDARD
   - Docker containerization
   - CI/CD pipeline details (3 workflows)
   - PyPI distribution process
   - ReadTheDocs deployment
   - Dependency management with uv

7. **[contribution-guide.md](./contribution-guide.md)** 🟢 STANDARD
   - Development setup for contributors
   - Coding conventions and standards
   - Testing requirements
   - Pull request workflow
   - Code review process

### Sphinx Documentation (Human-Readable, ReadTheDocs)

Comprehensive end-user documentation hosted on ReadTheDocs:

1. **[index.rst](./index.rst)** - Main landing page with badges and navigation
2. **[quickstart.rst](./quickstart.rst)** - Getting started guide
3. **[algorithms.rst](./algorithms.rst)** - Mathematical specifications and methodologies
4. **[reference.rst](./reference.rst)** - API reference (auto-generated from docstrings)
5. **[xarray_migration.rst](./xarray_migration.rst)** - Migration guide from numpy to xarray API
6. **[pypi_release.rst](./pypi_release.rst)** - PyPI release guide for maintainers
7. **[deprecations/index.rst](./deprecations/index.rst)** - Deprecation notices
8. **[troubleshooting.rst](./troubleshooting.rst)** - Common issues and solutions

### Supporting Documentation

- **[README.md](../README.md)** - GitHub landing page with quick start
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Detailed contribution guidelines
- **[CHANGELOG.md](../CHANGELOG.md)** - Version history and breaking changes
- **[LICENSE](../LICENSE)** - BSD 3-Clause license text

## Getting Started

### For Users
```bash
# Install from PyPI
pip install climate_indices

# Or with uv
uv pip install climate_indices
```

### For Developers
```bash
# Clone repository
git clone https://github.com/monocongo/climate_indices.git
cd climate_indices

# Setup development environment
uv sync --group dev
source .venv/bin/activate

# Run tests
uv run pytest
```

### For AI Agents
**Recommended Reading Order**:
1. [project-overview.md](./project-overview.md) - Understand project scope and design
2. [architecture.md](./architecture.md) - Learn system architecture
3. [source-tree-analysis.md](./source-tree-analysis.md) - Navigate codebase
4. [component-inventory.md](./component-inventory.md) - Deep-dive into modules

**For Specific Tasks**:
- **Add new index**: Read `compute.py`, `indices.py`, `xarray_adapter.py` + tests
- **Fix CLI bug**: Start with `__main__.py` or `__spi__.py`
- **Add validation**: Examine `xarray_adapter.py` and `exceptions.py`
- **Performance optimization**: Study `compute.py` and `test_benchmark_*.py` files

## Architecture at a Glance

```
┌────────────────────────────────────────┐
│   CLI Layer (__main__, __spi__)        │
├────────────────────────────────────────┤
│   Public API (typed_public_api,        │
│   xarray_adapter, indices)             │
├────────────────────────────────────────┤
│   Computation (compute, palmer)        │
├────────────────────────────────────────┤
│   Math/Stats (eto, lmoments)           │
├────────────────────────────────────────┤
│   Infrastructure (utils, logging,      │
│   exceptions, performance)             │
└────────────────────────────────────────┘
```

## Key Modules

| Module | Layer | Purpose | Lines | Priority |
|--------|-------|---------|-------|----------|
| `__main__.py` | CLI | Full-featured CLI (all indices) | 1826 | 🔴 CRITICAL |
| `__spi__.py` | CLI | Specialized SPI CLI (param caching) | 1478 | 🟡 HIGH |
| `typed_public_api.py` | API | Strict mypy-compliant API (NEW 2.2.0) | 210 | 🟢 NEW |
| `xarray_adapter.py` | API | Modern xarray interface (EXPANDED 2.2.0) | 1417 | 🔴 CRITICAL |
| `indices.py` | API | Legacy numpy API (STABLE) | 701 | 🟡 HIGH |
| `compute.py` | Computation | Core algorithms | 1127 | 🔴 CRITICAL |
| `palmer.py` | Computation | Palmer drought indices | 806 | 🟡 HIGH |
| `eto.py` | Math | PET methods | 416 | 🟡 HIGH |
| `lmoments.py` | Math | L-moments for fitting | 94 | 🟢 SPECIALTY |
| `exceptions.py` | Infrastructure | Exception hierarchy | 324 | 🔴 CRITICAL |
| `utils.py` | Infrastructure | Utilities | 396 | 🟡 HIGH |
| `logging_config.py` | Infrastructure | Structured logging | 76 | 🟢 STANDARD |
| `performance.py` | Infrastructure | Performance tracking (NEW 2.2.0) | 112 | 🟢 NEW |

## Testing Infrastructure

- **Test Files**: 26 test files organized by functionality
- **Fixtures**: 1005-line `conftest.py` with session-scoped fixtures
- **Categories**: Core functionality, QA (backward compat, property-based), validation, observability, performance benchmarks
- **Coverage**: >90% line coverage
- **Framework**: pytest + hypothesis (property-based) + pytest-benchmark

## CI/CD Pipelines

1. **Unit Tests** (`unit-tests-workflow.yml`): Python 3.10-3.13 matrix on ubuntu-latest
2. **Releases** (`release.yml`): Automated PyPI publishing on git tags
3. **Benchmarks** (`benchmarks.yml`, NEW 2.2.0): Performance regression tracking

## Critical Architectural Invariants

AI agents should be aware of these constraints:

1. **Time Dimension Chunking**: Dask arrays MUST have time as single chunk (`time: -1`)
2. **Calibration Period**: Default minimum 30 years; violations trigger `ShortCalibrationWarning`
3. **Distribution Fitting**: Requires minimum 10 non-zero values
4. **Backward Compatibility**: Legacy numpy API (`indices.py`) must remain stable
5. **Type Safety**: Strict mypy compliance enforced on `typed_public_api.py`

## Recent Changes (v2.2.0)

- ✨ **NEW**: `typed_public_api.py` - Strict mypy-compliant API
- 📈 **EXPANDED**: `xarray_adapter.py` - 400 → 1417 lines (comprehensive xarray support)
- 🧪 **NEW**: Property-based testing with hypothesis
- 🏎️ **NEW**: Benchmark suite with `test_benchmark_*.py` files
- 🔒 **NEW**: Enhanced exception hierarchy with context attributes
- 📊 **NEW**: Performance tracking with `performance.py`
- 🔧 **NEW**: Backward compatibility test suite

## Contributing

See [contribution-guide.md](./contribution-guide.md) for detailed guidelines. Quick summary:

1. Fork repository
2. Create feature branch
3. Follow coding standards (type hints, docstrings, tests)
4. Run tests locally: `uv run pytest`
5. Submit pull request
6. CI runs full test matrix

## License

BSD 3-Clause License. See [LICENSE](../LICENSE) for details.

---

**For AI Assistance**: This index is the starting point. Begin with [project-overview.md](./project-overview.md) for context, then navigate to specific guides based on your task.

**Last Updated**: 2026-02-11 (BMAD workflow execution)
**Documentation Version**: 1.0 (initial BMAD documentation)
**Project Version**: 2.2.0

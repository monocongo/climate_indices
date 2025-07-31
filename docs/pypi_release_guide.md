# PyPI Release Guide for Climate Indices

This document provides comprehensive guidance for releasing new versions of the climate-indices package to PyPI (Python Package Index).

## Overview

The climate-indices package uses modern Python packaging standards with `pyproject.toml` and the `uv` dependency manager. This guide covers the complete process from version management to post-release activities.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Version Management](#version-management)
- [Pre-Release Testing](#pre-release-testing)
- [PyPI Preparation](#pypi-preparation)
- [Release to Test PyPI](#release-to-test-pypi)
- [Release to Production PyPI](#release-to-production-pypi)
- [Post-Release Actions](#post-release-actions)
- [Automated Release](#automated-release-optional)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools
```bash
# Ensure you have the necessary tools installed
uv add --group=dev build twine

# Or using pip
pip install build twine
```

### PyPI Account Setup
1. **Create PyPI Account**: Register at [https://pypi.org/account/register/](https://pypi.org/account/register/)
2. **Enable 2FA**: Required for package uploads
3. **Create API Token**: Go to [Account Settings → API tokens](https://pypi.org/manage/account/token/)
   - Scope: "Entire account" or specific to climate-indices project
   - Store securely - you'll need it for uploads

### Configure Authentication
Create/update `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...  # Your API token

[testpypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...  # Test PyPI token (optional)
```

## Version Management

### Semantic Versioning

Follow [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- **PATCH** (1.0.1): Bug fixes, no breaking changes
- **MINOR** (1.1.0): New features, backward compatible
- **MAJOR** (2.0.0): Breaking changes

### Update Version Number

Edit `pyproject.toml`:
```toml
[project]
name = "climate-indices"
version = "1.x.x"  # Update this line
```

### Update Changelog

Create or update `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [1.x.x] - 2025-07-27

### Added
- DistributionFallbackStrategy for consolidated Pearson→Gamma fallback logic
- Custom exception classes (InsufficientDataError, PearsonFittingError)
- Named constants for magic numbers (MIN_NON_ZERO_VALUES_FOR_PEARSON, etc.)

### Changed
- Replaced None tuple returns with explicit exception handling
- Consolidated fallback logic across compute.py and indices.py
- Enhanced error messages with actionable guidance

### Fixed
- GitHub issue #582: SPI computation failures with extensive zero precipitation
- Improved robustness in dry regions with sparse precipitation data

### Removed
- Deprecated fallback logic patterns
```

## Pre-Release Testing

### Clean Build Environment
```bash
# Remove previous builds
rm -rf dist/ build/ *.egg-info/

# Clean Python cache
find . -type d -name "__pycache__" -delete
find . -name "*.pyc" -delete
```

### Build Package
```bash
# Build source distribution and wheel
uv run python -m build

# Verify build contents
ls -la dist/
tar -tzf dist/climate_indices-*.tar.gz | head -20
unzip -l dist/climate_indices-*.whl
```

### Local Testing
```bash
# Test installation in clean environment
uv venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate

# Install from built wheel
uv pip install dist/climate_indices-*.whl

# Test basic functionality
python -c "
from climate_indices import indices, compute
print('✅ Import successful')

# Test new architecture features
strategy = compute.DistributionFallbackStrategy()
print('✅ New fallback strategy available')

# Test SPI computation
import numpy as np
precip = np.random.exponential(10, 120)  # 10 years monthly
spi = indices.spi(precip, 3, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly)
print(f'✅ SPI computation successful: {len(spi)} values')
"

# Clean up
deactivate
rm -rf test-env
```

### Run Full Test Suite
```bash
# Ensure all tests pass
uv run pytest tests/ -v --cov=climate_indices

# Generate coverage report
uv run pytest --cov=climate_indices --cov-report=html
```

### Package Validation
```bash
# Check package metadata and description
uv run twine check dist/*

# Validate README renders correctly
uv run python -c "
from pathlib import Path
import markdown
content = Path('README.md').read_text()
html = markdown.markdown(content)
print(f'README length: {len(content)} chars, HTML: {len(html)} chars')
"
```

## PyPI Preparation

### Update Project Metadata

Ensure `pyproject.toml` has complete metadata:

```toml
[project]
name = "climate-indices"
version = "1.x.x"
description = "Python library for computing climate indices for drought monitoring and climate research"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "James Adams", email = "monocongo@gmail.com"}
]
keywords = ["climate", "meteorology", "drought", "indices", "SPI", "SPEI", "Palmer"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "xarray>=0.19.0",
    "dask>=2021.6.0",
    "h5netcdf>=0.11.0",
]

[project.urls]
Homepage = "https://github.com/monocongo/climate_indices"
Repository = "https://github.com/monocongo/climate_indices"
Documentation = "https://climate-indices.readthedocs.io/"
"Bug Tracker" = "https://github.com/monocongo/climate_indices/issues"
Changelog = "https://github.com/monocongo/climate_indices/blob/master/CHANGELOG.md"

[project.scripts]
climate_indices = "climate_indices.__main__:main"
```

## Release to Test PyPI

**Always test on Test PyPI first** to catch issues before production release.

### Upload to Test PyPI
```bash
# Upload to Test PyPI
uv run twine upload --repository testpypi dist/*

# Alternative with explicit URL
uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

### Test Installation from Test PyPI
```bash
# Create fresh test environment
uv venv test-pypi-env
source test-pypi-env/bin/activate

# Install from Test PyPI (may need to allow external dependencies)
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    climate-indices

# Test functionality
python -c "
import climate_indices
print('Test PyPI Version:', climate_indices.__version__)

from climate_indices import compute, indices
strategy = compute.DistributionFallbackStrategy()
print('✅ All imports successful from Test PyPI')
"

# Clean up
deactivate
rm -rf test-pypi-env
```

### Verify Test PyPI Page
Visit [https://test.pypi.org/project/climate-indices/](https://test.pypi.org/project/climate-indices/) to verify:
- ✅ Version number is correct
- ✅ Description renders properly
- ✅ Metadata is complete
- ✅ Links work correctly

## Release to Production PyPI

### Final Pre-Flight Check
```bash
# Verify you're releasing the intended version
grep "version" pyproject.toml

# Confirm all tests pass
uv run pytest tests/ -x  # Stop on first failure

# Check git status
git status  # Should be clean
git log --oneline -5  # Review recent commits
```

### Upload to Production PyPI
```bash
# Production upload
uv run twine upload dist/*

# Monitor upload progress
echo "Upload complete! Check https://pypi.org/project/climate-indices/"
```

### Verify Production Release
```bash
# Test installation from production PyPI
uv venv prod-test-env
source prod-test-env/bin/activate

# Install latest version
pip install --upgrade climate-indices

# Verify version and functionality
python -c "
import climate_indices
print('Production Version:', climate_indices.__version__)

# Quick functionality test
from climate_indices import compute, indices
import numpy as np

# Test basic SPI computation
precip = np.random.exponential(10, 60)  # 5 years monthly
spi = indices.spi(precip, 3, indices.Distribution.gamma, 2020, 2020, 2024, compute.Periodicity.monthly)
print(f'✅ SPI computation: {len(spi)} values generated')

# Test new fallback strategy
strategy = compute.DistributionFallbackStrategy()
print('✅ New architecture features available')
"

# Clean up
deactivate
rm -rf prod-test-env
```

## Post-Release Actions

### Git Tag and Release
```bash
# Create annotated tag
git tag -a v1.x.x -m "Release v1.x.x: Architectural improvements for zero precipitation handling"

# Push tag to GitHub
git push origin v1.x.x

# Verify tag
git tag -l | grep v1.x.x
```

### Create GitHub Release

1. Go to [https://github.com/monocongo/climate_indices/releases](https://github.com/monocongo/climate_indices/releases)
2. Click "Create a new release"
3. Select tag `v1.x.x`
4. Title: `v1.x.x: Architectural improvements for zero precipitation handling`
5. Description:

```markdown
## 🏗️ Architectural Improvements

This release consolidates the Pearson→Gamma fallback logic and replaces None tuple anti-patterns with explicit exception handling.

### ✨ New Features
- `DistributionFallbackStrategy` class for centralized fallback logic
- Custom exception classes (`InsufficientDataError`, `PearsonFittingError`)  
- Named constants for configuration thresholds

### 🐛 Bug Fixes
- **GitHub #582**: SPI computation failures with extensive zero precipitation
- Enhanced robustness in dry regions with sparse precipitation data

### 🔧 Developer Experience
- Explicit error handling eliminates None checking anti-patterns
- Centralized fallback logic improves maintainability
- Enhanced error messages provide actionable guidance

### 📋 Full Changelog
[View detailed changes](https://github.com/monocongo/climate_indices/blob/master/CHANGELOG.md)

### 🧪 Testing
- All existing tests pass (backward compatibility maintained)
- New comprehensive test suite for zero precipitation scenarios
- 13/13 tests passing ✅

### 📦 Installation
```bash
pip install --upgrade climate-indices
```

### 🔗 Links
- [PyPI Package](https://pypi.org/project/climate-indices/)
- [Documentation](https://climate-indices.readthedocs.io/)
- [GitHub Repository](https://github.com/monocongo/climate_indices)
```

### Update Documentation
```bash
# Update README.md if needed
# Update API documentation
# Consider creating migration guide for new features

# Rebuild documentation (if using Sphinx)
cd docs/
make clean
make html
```

### Communication
- **GitHub Discussions**: Announce new features
- **Issue Comments**: Notify users who reported related bugs
- **Social Media**: Share release announcement (if applicable)
- **Mailing Lists**: Inform scientific community of improvements

## Automated Release (Optional)

### GitHub Actions Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release to PyPI

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: read
  id-token: write  # For trusted publishing

jobs:
  release:
    runs-on: ubuntu-latest
    environment: release
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build package
      run: python -m build
      
    - name: Check package
      run: twine check dist/*
      
    - name: Upload to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        repository-url: https://pypi.org/legacy/
        # Uses trusted publishing, no token needed
```

### Trusted Publishing Setup

1. Go to [PyPI trusted publisher management](https://pypi.org/manage/account/publishing/)
2. Add publisher:
   - **PyPI Project Name**: `climate-indices`
   - **Owner**: `monocongo`
   - **Repository**: `climate_indices`
   - **Workflow**: `release.yml`
   - **Environment**: `release` (optional)

## Troubleshooting

### Common Issues

#### Authentication Errors
```bash
# Verify token is correct
twine check --repository-url https://test.pypi.org/legacy/ dist/*

# Check ~/.pypirc format
cat ~/.pypirc

# Test authentication
twine upload --repository testpypi dist/* --verbose
```

#### Build Failures
```bash
# Check pyproject.toml syntax
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"

# Verify dependencies
uv run pip-compile pyproject.toml

# Clean build
rm -rf dist/ build/ *.egg-info/
python -m build
```

#### Package Validation Errors
```bash
# Check long description
twine check dist/* --strict

# Validate README rendering
python -m readme_renderer README.md -o /tmp/readme.html
```

#### Version Conflicts
```bash
# Check if version already exists
pip index versions climate-indices

# Update version in pyproject.toml
# Rebuild package
```

### Rollback Procedure

If a release has critical issues:

1. **Cannot delete from PyPI**, but can:
2. **Yank the release**: `twine upload --repository pypi --action yank 1.x.x "Critical bug"`
3. **Release a patch version** immediately with fixes
4. **Update documentation** to warn about the problematic version

### Support Resources

- **PyPI Help**: [https://pypi.org/help/](https://pypi.org/help/)
- **Packaging Guide**: [https://packaging.python.org/](https://packaging.python.org/)
- **Twine Documentation**: [https://twine.readthedocs.io/](https://twine.readthedocs.io/)
- **Build Documentation**: [https://build.pypa.io/](https://build.pypa.io/)

## Release Checklist

Use this checklist for each release:

### Pre-Release
- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite (`uv run pytest`)
- [ ] Build package (`python -m build`)
- [ ] Validate package (`twine check dist/*`)
- [ ] Test local installation
- [ ] Clean git working directory

### Release
- [ ] Upload to Test PyPI
- [ ] Test installation from Test PyPI
- [ ] Upload to Production PyPI
- [ ] Verify production installation
- [ ] Create and push git tag
- [ ] Create GitHub release

### Post-Release
- [ ] Update documentation
- [ ] Announce release
- [ ] Monitor for issues
- [ ] Update project dependencies if needed

---

**Remember**: Always test thoroughly before releasing, and consider the impact on downstream users. When in doubt, release a patch version with fixes rather than trying to modify an existing release.
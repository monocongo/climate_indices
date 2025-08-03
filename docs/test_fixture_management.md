# Test Fixture Management Strategy

## Overview

This document outlines the recommended hybrid approach for managing large test fixtures in the climate_indices project. The current test suite contains 3,803 .npy files across 344 Palmer drought index test directories, totaling ~43MB of test data that should not be included in distribution packages.

## Problem Statement

### Current Issues
- **Repository Size**: 450MB total, with 43MB in test fixtures
- **Distribution Bloat**: Test fixtures were included in PyPI packages (37MB → 199KB after exclusion)
- **Developer Experience**: Large repository cloning times and storage requirements
- **CI Performance**: All fixtures loaded for every test run regardless of necessity

### Test Data Structure
```
tests/fixture/palmer/
├── 0101/          # Climate division 0101
│   ├── temps.npy      # Temperature data (16KB)
│   ├── precips.npy    # Precipitation data (16KB)
│   ├── gammas.npy     # Gamma parameters (4KB)
│   ├── alphas.npy     # Alpha parameters (4KB)
│   ├── betas.npy      # Beta parameters (4KB)
│   ├── pet.npy        # Potential evapotranspiration (16KB)
│   ├── pmdi.npy       # Palmer Modified Drought Index (16KB)
│   ├── zindex.npy     # Z-Index values (16KB)
│   ├── phdi.npy       # Palmer Hydrological Drought Index (16KB)
│   └── pdsi.npy       # Palmer Drought Severity Index (16KB)
├── 0102/          # Climate division 0102
...
└── 9999/          # Climate division 9999 (344 total divisions)
```

## Recommended Hybrid Approach

### Phase 1: Immediate Implementation (Minimal Risk)

#### 1.1 Git LFS for Essential Fixtures
**Goal**: Keep critical test cases in repository with manageable size

**Implementation**:
```bash
# Install Git LFS
git lfs install

# Create .gitattributes
echo "tests/fixture/palmer/essential/*.npy filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
echo "tests/fixture/palmer/smoke_test/*.npy filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
```

**Essential Test Cases Selection Criteria**:
- Geographic diversity (different climate regions)
- Data quality edge cases (missing values, extreme weather)
- Historical significance (drought events, wet periods)
- Computational edge cases (numerical stability tests)

**Recommended Essential Divisions**:
```
tests/fixture/palmer/essential/
├── 0405/  # Arizona - Arid Southwest
├── 2308/  # Louisiana - Humid Southeast  
├── 3205/  # Nevada - Great Basin Desert
├── 1909/  # Iowa - Continental Midwest
├── 4206/  # Pennsylvania - Humid Continental
├── 0506/  # California - Mediterranean
├── 4810/  # Washington - Marine West Coast
├── 3505/  # New Mexico - High Plains
├── 1205/  # Florida - Subtropical
└── 2405/  # Montana - Northern Plains
```

#### 1.2 Move Comprehensive Fixtures to Remote Storage

**AWS S3 Structure**:
```
s3://climate-indices-test-data/
├── fixtures/
│   ├── palmer/
│   │   ├── v1.0/           # Version for reproducibility
│   │   │   ├── divisions.json  # Metadata about available divisions
│   │   │   └── palmer_fixtures_v1.0.tar.gz  # All 334 remaining divisions
│   │   └── latest/         # Symlink to current version
│   └── metadata/
│       └── fixture_index.json  # Index of all available test data
└── checksums/
    └── palmer_fixtures_v1.0.sha256  # Integrity verification
```

#### 1.3 Enhanced Test Configuration

**Environment Variables**:
```python
# tests/conftest.py additions
import os
import pytest
from pathlib import Path

# Test execution modes
COMPREHENSIVE_TESTS = os.getenv("COMPREHENSIVE_TESTS", "false").lower() == "true"
CI_ENVIRONMENT = os.getenv("CI", "false").lower() == "true" 
FIXTURE_CACHE_DIR = Path(os.getenv("FIXTURE_CACHE_DIR", "~/.cache/climate_indices_tests")).expanduser()

@pytest.fixture(scope="session")
def fixture_mode():
    """Determine which fixtures to load based on environment"""
    if COMPREHENSIVE_TESTS or CI_ENVIRONMENT:
        return "comprehensive"
    return "essential"

@pytest.fixture(scope="session")
def palmer_divisions(fixture_mode):
    """Load appropriate Palmer divisions based on test mode"""
    if fixture_mode == "essential":
        return load_essential_divisions()
    else:
        return load_or_download_comprehensive_divisions()
```

### Phase 2: Optimization (Medium Risk)

#### 2.1 Synthetic Test Data Generation

**Purpose**: Reduce dependency on large historical datasets for unit tests

**Implementation**:
```python
# tests/generators.py
import numpy as np
from typing import Dict, List, Tuple

def generate_palmer_test_data(
    division_id: str,
    years: int = 5,
    seed: int = 42,
    climate_type: str = "temperate"
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic but realistic Palmer drought index test data
    
    Args:
        division_id: Climate division identifier
        years: Number of years of data to generate
        seed: Random seed for reproducibility
        climate_type: Climate classification (arid, temperate, humid, etc.)
    
    Returns:
        Dictionary containing all Palmer index arrays
    """
    np.random.seed(seed)
    months = years * 12
    
    # Generate base climate patterns
    precip = generate_precipitation_series(months, climate_type, seed)
    temp = generate_temperature_series(months, climate_type, seed)
    
    # Generate derived values using simplified Palmer calculations
    pet = calculate_pet_thornthwaite(temp, latitude=get_division_latitude(division_id))
    
    return {
        'precips': precip,
        'temps': temp,
        'pet': pet,
        'pdsi': calculate_simplified_pdsi(precip, pet),
        'phdi': calculate_simplified_phdi(precip, pet),
        'pmdi': calculate_simplified_pmdi(precip, pet),
        'zindex': calculate_simplified_zindex(precip, pet),
        'gammas': fit_gamma_parameters(precip),
        'alphas': extract_alpha_parameters(precip),
        'betas': extract_beta_parameters(precip),
    }

@pytest.mark.parametrize("climate_type", ["arid", "temperate", "humid"])
def test_palmer_calculation_synthetic(climate_type):
    """Test Palmer calculations using synthetic data"""
    data = generate_palmer_test_data("synthetic_001", years=10, climate_type=climate_type)
    # Test calculations...
```

#### 2.2 Property-Based Testing Integration

**Purpose**: Test edge cases without storing large fixture files

```python
# tests/test_palmer_properties.py
from hypothesis import given, strategies as st
import numpy as np

@given(
    precipitation=st.lists(
        st.floats(min_value=0.0, max_value=500.0), 
        min_size=120, max_size=1440  # 10-120 years of monthly data
    ),
    temperature=st.lists(
        st.floats(min_value=-40.0, max_value=50.0),
        min_size=120, max_size=1440
    )
)
def test_palmer_indices_properties(precipitation, temperature):
    """Property-based testing for Palmer drought indices"""
    precip_array = np.array(precipitation)
    temp_array = np.array(temperature)
    
    pdsi = calculate_pdsi(precip_array, temp_array)
    
    # Test mathematical properties
    assert np.all(np.isfinite(pdsi[~np.isnan(pdsi)]))  # No infinite values
    assert len(pdsi) == len(precip_array)  # Output length matches input
    # Additional property tests...
```

### Phase 3: Infrastructure (Long-term)

#### 3.1 Test Data CDN Setup

**CloudFront Distribution**:
- S3 bucket as origin
- Global edge locations for faster downloads
- Versioned URLs for cache busting
- Authentication for private test data

#### 3.2 Automated Fixture Management

**GitHub Actions Workflow**:
```yaml
# .github/workflows/update-test-fixtures.yml
name: Update Test Fixtures

on:
  workflow_dispatch:
    inputs:
      fixture_version:
        description: 'New fixture version'
        required: true
        type: string

jobs:
  update-fixtures:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate new fixtures
        run: |
          python scripts/generate_test_fixtures.py --version ${{ inputs.fixture_version }}
          
      - name: Upload to S3
        run: |
          aws s3 cp fixtures/ s3://climate-indices-test-data/fixtures/palmer/${{ inputs.fixture_version }}/ --recursive
          
      - name: Update fixture index
        run: |
          python scripts/update_fixture_index.py --version ${{ inputs.fixture_version }}
```

#### 3.3 Fixture Caching Strategy

**Local Development**:
```python
# tests/fixture_cache.py
class FixtureCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_or_download(self, fixture_key: str, version: str = "latest") -> Path:
        """Get fixture from cache or download if missing/stale"""
        cache_path = self.cache_dir / f"{fixture_key}_{version}.tar.gz"
        
        if not cache_path.exists() or self._is_stale(cache_path):
            self._download_fixture(fixture_key, version, cache_path)
            
        return self._extract_fixture(cache_path)
```

## Implementation Timeline

### Week 1: Setup
- [ ] Configure Git LFS
- [ ] Identify and move essential fixtures (10 divisions)
- [ ] Set up S3 bucket and upload remaining fixtures
- [ ] Update CI configuration

### Week 2: Testing Infrastructure  
- [ ] Implement fixture download logic
- [ ] Add environment variable controls
- [ ] Test comprehensive fixture loading in CI
- [ ] Update documentation

### Week 3: Optimization
- [ ] Begin synthetic data generation implementation
- [ ] Add property-based tests for critical functions
- [ ] Benchmark performance improvements

### Month 2: Advanced Features
- [ ] Implement fixture caching
- [ ] Set up CDN distribution
- [ ] Add automated fixture management
- [ ] Performance optimization

## Testing Strategy

### Test Categories

**1. Unit Tests (Synthetic Data)**
- Fast execution (< 1 second per test)
- No external dependencies
- Property-based testing for edge cases
- Generated data with known characteristics

**2. Integration Tests (Essential Fixtures)**
- Medium execution time (< 10 seconds per test)
- Git LFS fixtures (10 divisions)
- Cross-validation of algorithms
- Regression testing for known results

**3. Comprehensive Tests (Remote Fixtures)**
- Slower execution (< 5 minutes total)
- All 344 divisions
- Full validation against historical data
- Run in CI and before releases

### Pytest Configuration

```python
# pytest.ini
[tool.pytest.ini_options]
markers = [
    "unit: Fast unit tests with synthetic data",
    "integration: Integration tests with essential fixtures", 
    "comprehensive: Full test suite with all fixtures",
    "slow: Tests that take longer than 10 seconds"
]

# Default: unit + integration tests
addopts = "-m 'not comprehensive and not slow'"

# Environment-specific configurations
[tool.pytest.ini_options.env.CI]
addopts = "-m 'not slow'"  # Include comprehensive but exclude very slow tests

[tool.pytest.ini_options.env.NIGHTLY]
addopts = ""  # Run all tests including slow ones
```

## Monitoring and Maintenance

### Metrics to Track
- **Repository size**: Monitor growth over time
- **CI execution time**: Track test performance
- **Cache hit rates**: Fixture download efficiency
- **Test coverage**: Ensure comprehensive testing
- **Fixture staleness**: Data age and update frequency

### Maintenance Tasks
- **Monthly**: Review fixture usage patterns
- **Quarterly**: Update comprehensive fixtures with new data
- **Annually**: Archive old fixture versions
- **As needed**: Add new divisions or test cases

## Migration Guide

### For Developers

**Current Workflow**:
```bash
git clone https://github.com/monocongo/climate_indices.git
cd climate_indices
pytest tests/
```

**New Workflow**:
```bash
# Standard development (essential fixtures only)
git clone https://github.com/monocongo/climate_indices.git
cd climate_indices
git lfs pull  # Download essential fixtures
pytest tests/

# Comprehensive testing (all fixtures)
export COMPREHENSIVE_TESTS=true
pytest tests/  # Will download additional fixtures as needed
```

### For CI/CD Systems

**Update build scripts**:
```yaml
# Before
- name: Run tests
  run: pytest tests/

# After  
- name: Setup Git LFS
  run: git lfs pull

- name: Run comprehensive tests
  env:
    COMPREHENSIVE_TESTS: true
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  run: pytest tests/
```

## Cost Analysis

### Storage Costs (AWS S3)
- **Test fixtures**: ~43MB → $0.001/month
- **CloudFront**: ~$0.05/month (first 1TB free)
- **Data transfer**: Negligible for development team

### Development Benefits
- **Repository size**: 450MB → 50MB (89% reduction)
- **Clone time**: ~2 minutes → ~30 seconds  
- **Storage per developer**: ~450MB → ~50MB
- **CI cache efficiency**: Improved fixture reuse

### Risk Assessment
- **Low risk**: Git LFS integration, fixture caching
- **Medium risk**: S3 dependency, download failures
- **High risk**: Synthetic data accuracy, comprehensive test coverage

## Conclusion

This hybrid approach provides:
- **Immediate benefits**: Reduced repository size and faster cloning
- **Scalability**: Handle growing test data without repository bloat
- **Flexibility**: Different test modes for different scenarios
- **Maintainability**: Clear separation of concerns and automated management

The strategy balances developer experience, CI performance, and test coverage while providing a foundation for long-term growth.
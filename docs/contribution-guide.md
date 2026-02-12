# Contribution Guide

## Getting Started

Thank you for your interest in contributing to `climate_indices`! This guide will help you get started with the contribution process.

### Prerequisites
1. **Python 3.10+** installed
2. **uv** package manager installed
3. **Git** for version control
4. Familiarity with pytest and type hints

## Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/climate_indices.git
cd climate_indices
```

### 2. Set Up Upstream Remote
```bash
git remote add upstream https://github.com/monocongo/climate_indices.git
git fetch upstream
```

### 3. Install Dependencies
```bash
uv sync --group dev
source .venv/bin/activate
```

## Coding Conventions

### Style Guidelines
- **Indentation**: 4 spaces (no tabs)
- **Line length**: 120 characters
- **Whitespace**: After list items, around operators, around method parameters
- **Comments**: Above-line comments only, liberal use encouraged
- **Naming**: `snake_case` for functions/variables, NO `camelCase`
- **Descriptive names**: Avoid `data`, `df2`, `temp` - use meaningful names

### Type Hints
- **Required**: All function arguments and return types MUST be type-hinted
- **Format**:
  ```python
  def compute_index(
      values: np.ndarray,
      scale: int,
      *,  # Keyword-only separator
      distribution: Distribution = Distribution.gamma,
  ) -> np.ndarray:
      """Compute climate index."""
  ```

### Docstrings
- **Format**: Google-style docstrings
- **Required**: All public functions must have docstrings
- **Example**:
  ```python
  def spi(
      values: np.ndarray,
      scale: int,
      distribution: Distribution,
  ) -> np.ndarray:
      """Compute Standardized Precipitation Index.

      Args:
          values: Precipitation time series in mm
          scale: Temporal scale in months (1-72)
          distribution: Fitting distribution (gamma or pearson)

      Returns:
          SPI values as numpy array (same shape as input)

      Raises:
          InsufficientDataError: If fewer than 10 non-zero values
          InvalidArgumentError: If scale is out of range

      Example:
          >>> precip = np.array([...])
          >>> spi_6 = spi(precip, scale=6, distribution=Distribution.gamma)
      """
  ```

### Import Organization
```python
# 1. Standard library
from __future__ import annotations
import os
from typing import Optional

# 2. Third-party packages
import numpy as np
import xarray as xr

# 3. Local modules
from climate_indices import compute, exceptions
```

**Forbidden**: Wildcard imports (`from module import *`)

## Testing Requirements

### Test Coverage
- **Minimum**: >90% line coverage
- **New features**: MUST include tests
- **Bug fixes**: MUST include regression test

### Test Structure
```python
# tests/test_new_feature.py
import pytest
import numpy as np
from climate_indices import new_feature

def test_new_feature_basic():
    """Test basic functionality."""
    result = new_feature(input_data)
    assert result.shape == expected_shape
    np.testing.assert_allclose(result, expected_output)

def test_new_feature_edge_case():
    """Test edge case handling."""
    with pytest.raises(InvalidArgumentError):
        new_feature(invalid_input)
```

### Running Tests Before Submission
```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=term

# Code quality
ruff check --fix src/ tests/
ruff format src/ tests/
mypy src/
```

## Contribution Workflow

### 1. Create Feature Branch
```bash
git checkout master
git pull upstream master
git checkout -b fix-bug-description
# Use descriptive branch names:
# - feature/add-new-index
# - fix/handle-nan-values
# - docs/improve-api-examples
```

### 2. Make Changes
- Follow coding conventions above
- Write tests for new functionality
- Update documentation if needed
- Keep commits focused and atomic

### 3. Commit Messages
```bash
# Good: Clear, concise, imperative mood
git commit -m "Add validation for empty latitude arrays"
git commit -m "Fix NaN handling in temporal scaling"

# Bad: Vague, past tense
git commit -m "Fixed stuff"
git commit -m "Changed things"
```

**Format for larger commits**:
```
Add Hargreaves PET method

- Implement daily PET computation
- Add validation for tmin/tmax inputs
- Include comprehensive test suite
- Update documentation with examples
```

### 4. Pull Upstream Changes Regularly
```bash
git fetch upstream
git merge upstream/master
# Or rebase if preferred:
git rebase upstream/master
```

### 5. Push and Create Pull Request
```bash
git push origin fix-bug-description
```

Then on GitHub:
1. Click "New Pull Request"
2. Provide clear description of changes
3. Reference any related issues (`Fixes #123`)
4. Wait for CI to pass
5. Address review comments

## Pull Request Checklist

Before submitting:
- [ ] Tests pass locally (`uv run pytest`)
- [ ] Code is formatted (`ruff format`)
- [ ] Linting passes (`ruff check --fix`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No unrelated changes included

## Code Review Process

### What Reviewers Check
1. **Functionality**: Does the code work as intended?
2. **Tests**: Are there sufficient tests with good coverage?
3. **Style**: Does it follow project conventions?
4. **Documentation**: Are docstrings and comments clear?
5. **Backward compatibility**: Does it break existing APIs?

### Responding to Reviews
- Address all comments
- Ask questions if feedback is unclear
- Make requested changes in new commits
- Mark conversations as resolved when addressed

## Best Practices

### DO:
✅ Write clear, focused pull requests
✅ Include tests for all new code
✅ Follow the existing code style
✅ Update documentation when needed
✅ Respond promptly to review feedback
✅ Keep commits atomic and well-described

### DON'T:
❌ Mix unrelated changes in one PR
❌ Submit without running tests locally
❌ Ignore linting/formatting errors
❌ Break backward compatibility without discussion
❌ Include commented-out code or debug prints
❌ Use magic numbers without constants

## Specific Contribution Areas

### Adding a New Climate Index
1. Implement core algorithm in `compute.py`
2. Add high-level function to `indices.py` (numpy API)
3. Add wrapper to `xarray_adapter.py` (xarray API)
4. Add tests in `tests/test_compute.py`, `test_indices.py`, `test_xarray_adapter.py`
5. Update documentation in `docs/reference.rst`

### Fixing a Bug
1. Write a failing test that reproduces the bug
2. Fix the bug in the appropriate module
3. Verify the test now passes
4. Add regression test to prevent recurrence

### Improving Documentation
1. Update RST files in `docs/` for Sphinx docs
2. Update docstrings in source code
3. Build docs locally: `cd docs && make html`
4. Check for broken links and formatting issues

## Getting Help

- **Questions**: Open a GitHub issue with `question` label
- **Bugs**: Open a GitHub issue with `bug` label
- **Features**: Open a GitHub issue with `enhancement` label
- **Discussion**: Use GitHub Discussions

## Recognition

Contributors are acknowledged in:
- Git commit history
- GitHub contributor graph
- Release notes (for significant contributions)

---

Thank you for contributing to `climate_indices`! Your efforts help advance open science and climate research.

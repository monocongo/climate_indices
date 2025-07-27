PyPI Release Guide
==================

This document provides comprehensive guidance for releasing new versions of the climate-indices package to PyPI (Python Package Index).

Prerequisites
-------------

Required Tools
^^^^^^^^^^^^^^

Ensure you have the necessary tools installed:

.. code-block:: bash

   # Install build dependencies
   uv add --group=dev build twine

   # Or using pip
   pip install build twine

PyPI Account Setup
^^^^^^^^^^^^^^^^^^

1. **Create PyPI Account**: Register at https://pypi.org/account/register/
2. **Enable 2FA**: Required for package uploads
3. **Create API Token**: Go to Account Settings → API tokens

   - **Scope**: "Entire account" or specific to climate-indices project
   - Store securely - you'll need it for uploads

Configure Authentication
^^^^^^^^^^^^^^^^^^^^^^^^^

Create/update ``~/.pypirc``:

.. code-block:: ini

   [pypi]
   username = __token__
   password = pypi-AgEIcHlwaS5vcmcC...  # Your API token

   [testpypi]
   username = __token__
   password = pypi-AgEIcHlwaS5vcmcC...  # Test PyPI token (optional)

Version Management
------------------

Semantic Versioning
^^^^^^^^^^^^^^^^^^^

Follow `Semantic Versioning <https://semver.org/>`_ (MAJOR.MINOR.PATCH):

- **PATCH** (1.0.1): Bug fixes, no breaking changes
- **MINOR** (1.1.0): New features, backward compatible
- **MAJOR** (2.0.0): Breaking changes

Update Version Number
^^^^^^^^^^^^^^^^^^^^^

Edit ``pyproject.toml``:

.. code-block:: toml

   [project]
   name = "climate-indices"
   version = "1.x.x"  # Update this line

Update Changelog
^^^^^^^^^^^^^^^^

Create or update ``CHANGELOG.md`` following `Keep a Changelog <https://keepachangelog.com/>`_ format:

.. code-block:: markdown

   # Changelog

   ## [1.x.x] - 2025-07-27

   ### Added
   - DistributionFallbackStrategy for consolidated Pearson→Gamma fallback logic
   - Custom exception classes (InsufficientDataError, PearsonFittingError)
   - Named constants for magic numbers

   ### Changed
   - Replaced None tuple returns with explicit exception handling
   - Consolidated fallback logic across compute.py and indices.py

   ### Fixed
   - GitHub issue #582: SPI computation failures with extensive zero precipitation

Pre-Release Testing
-------------------

Clean Build Environment
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Remove previous builds
   rm -rf dist/ build/ *.egg-info/

   # Clean Python cache
   find . -type d -name "__pycache__" -delete
   find . -name "*.pyc" -delete

Build Package
^^^^^^^^^^^^^

.. code-block:: bash

   # Build source distribution and wheel
   uv run python -m build

   # Verify build contents
   ls -la dist/
   tar -tzf dist/climate_indices-*.tar.gz | head -20
   unzip -l dist/climate_indices-*.whl

Local Testing
^^^^^^^^^^^^^

.. code-block:: bash

   # Test installation in clean environment
   uv venv test-env
   source test-env/bin/activate

   # Install from built wheel
   pip install dist/climate_indices-*.whl

   # Test basic functionality
   python -c "
   from climate_indices import indices, compute
   print('✅ Import successful')

   # Test new architecture features
   strategy = compute.DistributionFallbackStrategy()
   print('✅ New fallback strategy available')
   "

   # Clean up
   deactivate
   rm -rf test-env

Run Full Test Suite
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Ensure all tests pass
   uv run pytest tests/ -v --cov=climate_indices

   # Generate coverage report
   uv run pytest --cov=climate_indices --cov-report=html

Package Validation
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Check package metadata and description
   uv run twine check dist/*

Release to Test PyPI
--------------------

**Always test on Test PyPI first** to catch issues before production release.

Upload to Test PyPI
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Upload to Test PyPI
   uv run twine upload --repository testpypi dist/*

   # Alternative with explicit URL
   uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*

Test Installation from Test PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create fresh test environment
   uv venv test-pypi-env
   source test-pypi-env/bin/activate

   # Install from Test PyPI
   pip install --index-url https://test.pypi.org/simple/ \\
       --extra-index-url https://pypi.org/simple/ \\
       climate-indices

   # Test functionality
   python -c "
   import climate_indices
   print('Test PyPI Version:', climate_indices.__version__)
   "

   # Clean up
   deactivate
   rm -rf test-pypi-env

Release to Production PyPI
---------------------------

Final Pre-Flight Check
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Verify you're releasing the intended version
   grep "version" pyproject.toml

   # Confirm all tests pass
   uv run pytest tests/ -x  # Stop on first failure

   # Check git status
   git status  # Should be clean
   git log --oneline -5  # Review recent commits

Upload to Production PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Production upload
   uv run twine upload dist/*

   # Monitor upload progress
   echo "Upload complete! Check https://pypi.org/project/climate-indices/"

Verify Production Release
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Test installation from production PyPI
   uv venv prod-test-env
   source prod-test-env/bin/activate

   # Install latest version
   pip install --upgrade climate-indices

   # Verify version and functionality
   python -c "
   import climate_indices
   print('Production Version:', climate_indices.__version__)

   # Test new architecture features
   from climate_indices import compute
   strategy = compute.DistributionFallbackStrategy()
   print('✅ New architecture features available')
   "

   # Clean up
   deactivate
   rm -rf prod-test-env

Post-Release Actions
--------------------

Git Tag and Release
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create annotated tag
   git tag -a v1.x.x -m "Release v1.x.x: Architectural improvements"

   # Push tag to GitHub
   git push origin v1.x.x

   # Verify tag
   git tag -l | grep v1.x.x

Create GitHub Release
^^^^^^^^^^^^^^^^^^^^^

1. Go to https://github.com/monocongo/climate_indices/releases
2. Click "Create a new release"
3. Select tag ``v1.x.x``
4. Add release notes describing:

   - **New Features**: DistributionFallbackStrategy, custom exceptions
   - **Bug Fixes**: GitHub #582 zero precipitation handling
   - **Developer Experience**: Explicit error handling improvements
   - **Testing**: Comprehensive test coverage
   - **Installation**: ``pip install --upgrade climate-indices``

Update Documentation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Update README.md if needed
   # Update API documentation
   # Rebuild documentation (if using Sphinx)
   cd docs/
   make clean
   make html

Automated Release (Optional)
-----------------------------

GitHub Actions Workflow
^^^^^^^^^^^^^^^^^^^^^^^^

Create ``.github/workflows/release.yml``:

.. code-block:: yaml

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

Trusted Publishing Setup
^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to `PyPI trusted publisher management <https://pypi.org/manage/account/publishing/>`_
2. Add publisher:

   - **PyPI Project Name**: ``climate-indices``
   - **Owner**: ``monocongo``
   - **Repository**: ``climate_indices``
   - **Workflow**: ``release.yml``
   - **Environment**: ``release`` (optional)

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Authentication Errors**

.. code-block:: bash

   # Verify token is correct
   twine check --repository-url https://test.pypi.org/legacy/ dist/*

   # Check ~/.pypirc format
   cat ~/.pypirc

   # Test authentication
   twine upload --repository testpypi dist/* --verbose

**Build Failures**

.. code-block:: bash

   # Check pyproject.toml syntax
   python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"

   # Clean build
   rm -rf dist/ build/ *.egg-info/
   python -m build

**Version Conflicts**

.. code-block:: bash

   # Check if version already exists
   pip index versions climate-indices

   # Update version in pyproject.toml
   # Rebuild package

Rollback Procedure
^^^^^^^^^^^^^^^^^^

If a release has critical issues:

1. **Cannot delete from PyPI**, but can:
2. **Yank the release**: ``twine upload --repository pypi --action yank 1.x.x "Critical bug"``
3. **Release a patch version** immediately with fixes
4. **Update documentation** to warn about the problematic version

Release Checklist
------------------

Pre-Release
^^^^^^^^^^^

- ☐ Update version in ``pyproject.toml``
- ☐ Update ``CHANGELOG.md``
- ☐ Run full test suite (``uv run pytest``)
- ☐ Build package (``python -m build``)
- ☐ Validate package (``twine check dist/*``)
- ☐ Test local installation
- ☐ Clean git working directory

Release
^^^^^^^

- ☐ Upload to Test PyPI
- ☐ Test installation from Test PyPI
- ☐ Upload to Production PyPI
- ☐ Verify production installation
- ☐ Create and push git tag
- ☐ Create GitHub release

Post-Release
^^^^^^^^^^^^

- ☐ Update documentation
- ☐ Announce release
- ☐ Monitor for issues
- ☐ Update project dependencies if needed

Support Resources
-----------------

- **PyPI Help**: https://pypi.org/help/
- **Packaging Guide**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/
- **Build Documentation**: https://build.pypa.io/

.. note::
   Always test thoroughly before releasing, and consider the impact on downstream users. 
   When in doubt, release a patch version with fixes rather than trying to modify an existing release.

.. tip::
   For detailed step-by-step instructions with full code examples, see the complete 
   `PyPI Release Guide (Markdown) <pypi_release_guide.md>`_ in the docs folder.
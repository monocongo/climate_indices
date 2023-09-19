# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] (latest master branch)

### Added

- something we've added, coming soon
- something else we've added

### Fixed

- something we've fixed (#issue_number)
- something else we've fixed (#issue_number)

### Changed

- something we've changed (#issue_number)

### Removed

## [2.0.0] - 2023-07-15

### Added

- GitHub Action workflow which performs unit testing on the four supported versions of Python (3.8, 3.9, 3.10, and 3.11)

### Fixed

- L-moments-related errors (#512)
- Various cleanups and formatting indentations

### Changed

- Build and dependency management now using poetry instead of setuptools
- Documentation around installation with examples (#521) 

### Removed

- Palmer indices (these were always half-baked and nobody ever showed any interest in developing them further)
- Numba integration (see [this discussion](https://github.com/monocongo/climate_indices/discussions/502#discussioncomment-6377732)
  for context)
- requirements.txt (dependencies now specified solely in pyproject.toml)
- setup.py (now using poetry as the build tool)

[unreleased]: https://github.com/monocongo/climate_indices/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/monocongo/climate_indices/releases/tag/v2.0.0
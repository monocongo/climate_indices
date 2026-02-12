Deprecations
============

This section tracks deprecated APIs, removals, and migration guidance.

Deprecation Policy
------------------

``climate_indices`` follows a two-release deprecation cycle:

1. A feature is marked deprecated and emits ``ClimateIndicesDeprecationWarning``.
2. The feature is removed in the announced removal release.

The warning message includes:

- The release where deprecation began.
- The planned removal release.
- The recommended alternative.
- A migration URL for the relevant change.

.. toctree::
   :maxdepth: 1

   api-changes

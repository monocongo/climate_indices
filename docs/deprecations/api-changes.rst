API Changes
===========

This page records feature-level deprecation and migration notes.

``spi`` console script (removed in 3.0.0)
-----------------------------------------

The ``spi`` console script (``climate_indices.__spi__:main``) was deprecated
in 2.4.0 and removed in 3.0.0. The ``climate_indices.__spi__`` module is gone,
so imports of it and invocations of the ``spi`` command raise the usual import
and shell errors rather than ``ClimateIndicesDeprecationWarning``.

Use ``climate_indices --index spi`` instead, with two caveats:

- The ``--save_params`` and ``--load_params`` options, which cache fitted SPI
  distribution parameters in a NetCDF file, are retired along with the script
  rather than migrated to ``climate_indices`` (#957). Fitting parameters remain
  available at the library level: fit the scaled values once with
  ``compute.gamma_parameters()`` or ``compute.pearson_parameters()``, then pass
  the result as the ``fitting_params`` argument of ``indices.spi()``. The SPI
  section of the documentation index shows the gridded workflow. For SPEI, fit
  the series SPEI itself prepares -- precipitation clipped at zero, minus PET,
  plus the 1000 mm offset, then scaled -- and pass those parameters to
  ``indices.spei()``; the SPI workflow's precipitation series fits a different
  distribution and its parameters are accepted without validation.
- For multi-scale runs, ``climate_indices --index spi`` reopens and stages the
  precipitation input once per scale, while the ``spi`` script stages it once
  for all scales. Large multi-scale batches may therefore need more time and
  memory after migrating.

No other API removals are scheduled beyond what is announced in warning
messages.

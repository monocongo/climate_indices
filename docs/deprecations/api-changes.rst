API Changes
===========

This page records feature-level deprecation and migration notes.

``spi`` console script (deprecated in 2.4.0)
--------------------------------------------

The ``spi`` console script (``climate_indices.__spi__:main``) is deprecated
since 2.4.0 and is scheduled for removal in 3.0.0. Invoking it emits
``ClimateIndicesDeprecationWarning``.

Use ``climate_indices --index spi`` instead, with two caveats:

- The ``--save_params`` and ``--load_params`` options, which cache fitted SPI
  distribution parameters in a NetCDF file, are retired along with the script
  rather than migrated to ``climate_indices`` (#957). Fitting parameters remain
  available at the library level: fit the scaled values once with
  ``compute.gamma_parameters()`` or ``compute.pearson_parameters()``, then pass
  the result as the ``fitting_params`` argument of ``indices.spi()`` (or
  ``indices.spei()``). The SPI section of the documentation index shows the
  gridded workflow.
- For multi-scale runs, ``climate_indices --index spi`` reopens and stages the
  precipitation input once per scale, while the ``spi`` script stages it once
  for all scales. Large multi-scale batches may therefore need more time and
  memory after migrating.

No other API removals are scheduled beyond what is announced in warning
messages.

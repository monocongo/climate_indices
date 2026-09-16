API Changes
===========

This page records feature-level deprecation and migration notes.

``spi`` console script (deprecated in 2.4.0)
-------------------------------------------

The ``spi`` console script (``climate_indices.__spi__:main``) is deprecated
since 2.4.0 and is scheduled for removal in 3.0.0. Invoking it emits
``ClimateIndicesDeprecationWarning``.

Use ``climate_indices --index spi`` instead. The ``--save_params`` and
``--load_params`` options, which cache fitted SPI distribution parameters in a
NetCDF file, currently exist only on the ``spi`` script; migrating them to
``climate_indices`` is tracked in #957. The script will not be removed before
that migration lands or the options are explicitly retired.

No other API removals are scheduled beyond what is announced in warning
messages.

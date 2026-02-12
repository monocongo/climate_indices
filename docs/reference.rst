API Reference
=============

.. contents::
    :local:
    :backlinks: none


Public API — Index Functions
-----------------------------

.. note:: The xarray DataArray overloads in ``typed_public_api`` are **beta**.
   NumPy overloads are stable. See :doc:`xarray_migration` for details.

climate_indices.typed_public_api
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.typed_public_api
   :members:


xarray Integration
------------------

.. warning:: **Beta Feature** — The xarray adapter layer is beta.
   See :doc:`xarray_migration` for stability guarantees.

climate_indices.xarray_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.xarray_adapter
   :members:
   :exclude-members: InputType

.. autoclass:: climate_indices.xarray_adapter.InputType
   :members:
   :no-index:


Core Computation Modules
-------------------------

climate_indices.compute
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.compute
   :members:


climate_indices.indices
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.indices
   :members:


climate_indices.eto
~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.eto
   :members:


climate_indices.palmer
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.palmer
   :members:


climate_indices.lmoments
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.lmoments
   :members:


climate_indices.utils
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.utils
   :members:


Error Handling
--------------

climate_indices.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.exceptions
   :members:
   :special-members: __init__


Observability
-------------

climate_indices.logging_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.logging_config
   :members:


climate_indices.performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: climate_indices.performance
   :members:

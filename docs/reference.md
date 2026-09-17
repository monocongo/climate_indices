# API Reference

```{contents}
:backlinks: none
:local: true
```

## Public API — Index Functions

:::{note} The xarray DataArray overloads in `typed_public_api` are **beta**. NumPy overloads are stable. See {doc}`xarray_migration` for details.
:::

### climate_indices.typed_public_api

```{eval-rst}
.. automodule:: climate_indices.typed_public_api
   :members:
```

## xarray Integration

:::{warning} **Beta Feature** — The xarray adapter layer is beta. See {doc}`xarray_migration` for stability guarantees.
:::

The Dask-backed SPI/SPEI workflow is demonstrated end to end in
[notebooks/zarr_dask_spi_spei.ipynb](https://github.com/monocongo/climate_indices/blob/main/notebooks/zarr_dask_spi_spei.ipynb);
{doc}`troubleshooting` covers its setup and failure modes.

### climate_indices.xarray_adapter

```{eval-rst}
.. automodule:: climate_indices.xarray_adapter
   :members:
   :exclude-members: InputType
```

```{eval-rst}
.. autoclass:: climate_indices.xarray_adapter.InputType
   :members:
   :no-index:
```

## Core Computation Modules

### climate_indices.compute

```{eval-rst}
.. automodule:: climate_indices.compute
   :members:
   :exclude-members: DistributionFittingError, InsufficientDataError, PearsonFittingError
```

### climate_indices.indices

```{eval-rst}
.. automodule:: climate_indices.indices
   :members:
```

### climate_indices.eto

```{eval-rst}
.. automodule:: climate_indices.eto
   :members:
```

### climate_indices.palmer

```{eval-rst}
.. automodule:: climate_indices.palmer
   :members:
```

### climate_indices.lmoments

```{eval-rst}
.. automodule:: climate_indices.lmoments
   :members:
```

### climate_indices.utils

```{eval-rst}
.. automodule:: climate_indices.utils
   :members:
```

## Error Handling

### climate_indices.exceptions

```{eval-rst}
.. automodule:: climate_indices.exceptions
   :members:
   :special-members: __init__
```

## Observability

### climate_indices.logging_config

```{eval-rst}
.. automodule:: climate_indices.logging_config
   :members:
```

### climate_indices.performance

```{eval-rst}
.. automodule:: climate_indices.performance
   :members:
```


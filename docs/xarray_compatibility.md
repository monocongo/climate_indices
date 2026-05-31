# xarray Compatibility Matrix

The xarray API is beta in v2.5. Numerical results are expected to match the
stable NumPy API, while parameter inference, metadata, and coordinate behavior
may change in a future minor release.

| Feature | Supported | Coverage |
| --- | --- | --- |
| `DataArray` inputs for SPI | Yes | `tests/test_xarray_equivalence.py` compares 1-D and gridded xarray outputs to NumPy. |
| `DataArray` inputs for SPEI | Yes | `tests/test_xarray_equivalence.py` compares xarray precipitation/PET inputs to NumPy. |
| `DataArray` inputs for EDDI | Yes | EDDI wrapper and metadata tests cover the typed public API. |
| `DataArray` inputs for PET Thornthwaite | Yes | xarray adapter tests cover scalar and spatial latitude handling. |
| `DataArray` inputs for PET Hargreaves | Yes | xarray adapter tests cover aligned daily temperature inputs. |
| `DataArray` inputs for PNP | Yes | PNP wrapper tests cover scale handling and metadata. |
| `DataArray` inputs for PCI | Yes | PCI uses a manual scalar-output wrapper. |
| Palmer direct xarray API | No | Use the NumPy Palmer function with `.values`, then rewrap outputs. See `notebooks/palmer_indices_xarray.ipynb`. |
| Coordinate preservation | Yes | Adapter tests verify time and spatial coordinates are preserved. |
| CF-style metadata | Yes | `CF_METADATA` registry and adapter tests verify `long_name`, `units`, `references`, version, and history attributes. |
| Dask-backed arrays | Yes, constrained | The time dimension must be a single chunk. Adapter tests verify detection and validation. |
| Automatic temporal inference | Yes | Monthly and daily time-coordinate inference is covered by adapter tests. |
| Multi-input alignment | Yes | SPEI aligns precipitation and PET with an inner join and emits a warning when timesteps are dropped. |

## Operational Guidance

- Use NumPy APIs for stable production integrations that cannot absorb beta
  interface changes.
- Use xarray APIs for labeled, gridded workflows where coordinate preservation
  and metadata are more valuable than strict interface stability.
- Keep Dask chunks spatial when possible and leave `time` as one chunk before
  calling index functions.
- Run the notebook CI command before publishing examples:
  `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/xarray_getting_started.ipynb notebooks/palmer_indices_xarray.ipynb notebooks/eddi_xarray.ipynb`.

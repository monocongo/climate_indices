"""One output path for the CLI: attributes from the CF registry, atomic writes.

Both CLI backends -- the shared-memory NumPy route and the xarray route -- write
through ``write_netcdf_atomic`` here. The shared-memory route also builds its
variable attributes with ``build_index_attrs``: they come from
:data:`climate_indices.cf_metadata_registry.CF_METADATA` through the same
``build_output_attrs`` layering the xarray adapters use, so a CLI output carries
the registry's long name, units and references plus the library version and a
history entry. (The xarray route already stamped those attrs in its adapter.)
Writes go beside the target and replace it only once the whole file is on disk,
so a failed computation cannot leave a hollow file where an earlier output was.

Experimental, opt-in packing (``--pack_output``): ``choose_netcdf_encoding``
stores each variable as scaled int16 when its values fit, float32 otherwise,
compressed either way. Without it the output encoding is xarray's default.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import xarray as xr

from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.logging_config import get_logger
from climate_indices.xarray_adapter import build_output_attrs

_logger = get_logger(__name__)

# int16 packing: 32767 steps either side of zero, the minimum reserved as fill
_INT16_MAX = 32767
_INT16_FILL = -32768

# encoding keys inherited from a previously opened file that would override the
# packing chosen here, e.g. an old int16 ``scale_factor`` applied to widened data
_STALE_ENCODING_KEYS = ("dtype", "scale_factor", "add_offset", "_FillValue", "missing_value")


def build_index_attrs(
    source: xr.DataArray,
    cf_key: str,
    *,
    index_name: str,
    calculation_metadata: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build output attributes from ``CF_METADATA`` for a CLI index output.

    :param source: the input the index was computed from, for provenance
    :param cf_key: the ``CF_METADATA`` entry naming the output convention
    :param index_name: display name recorded in the history entry
    :param calculation_metadata: parameters recorded in the history entry
    :param extra: attributes that override the registry entry, e.g. a valid
        range the registry's attribute model does not carry
    :return: the output variable's attributes
    """
    attrs = build_output_attrs(
        source,
        cast("dict[str, str]", CF_METADATA[cf_key]),
        calculation_metadata,
        index_name=index_name,
    )
    # The source's own valid range describes the input, not the computed index;
    # drop it so only an explicit ``extra`` range reaches the output.
    for range_attr in ("valid_min", "valid_max", "valid_range", "actual_range"):
        attrs.pop(range_attr, None)
    if extra:
        attrs.update(extra)
    return attrs


def _chunksizes(variable: xr.Variable) -> tuple[int, ...] | None:
    """
    The on-disk chunk sizes for ``variable``, clamped to its shape.

    A ``chunksizes`` already in the encoding (copied from the input by
    ``--chunksizes input``) wins over the Dask chunks, which keeps the layout
    the user asked for; otherwise the first Dask chunk along each dimension.
    """
    chunks = variable.encoding.get("chunksizes")
    if not chunks and variable.chunks is not None:
        chunks = tuple(dim_chunks[0] for dim_chunks in variable.chunks)
    if not chunks or len(chunks) != variable.ndim or 0 in variable.shape:
        return None
    return tuple(min(chunk, length) for chunk, length in zip(chunks, variable.shape, strict=True))


def choose_netcdf_encoding(
    dataset: xr.Dataset,
    *,
    scale: float = 1e-4,
    complevel: int = 4,
) -> dict[Hashable, dict[str, Any]]:
    """
    Choose the NetCDF encoding of each floating-point data variable.

    A variable whose every finite value satisfies ``abs(value) <= 32767 * scale``
    is packed as int16 with ``scale_factor=scale``; anything else, including a
    variable holding +/-inf, is stored as float32. An all-NaN variable packs as
    int16 (it is only fill values). Every variable gets an explicit encoding,
    and the encoding keys inherited from an opened file that would conflict with
    it are cleared from ``dataset`` so none leak into the write. Dask-backed
    variables are ranged together in one pass.

    :param dataset: the Dataset about to be written; its variables' stale
        encoding keys are removed in place
    :param scale: the int16 packing step
    :param complevel: the zlib compression level
    :return: the ``encoding`` argument for ``Dataset.to_netcdf``
    """
    names = [name for name, variable in dataset.data_vars.items() if np.issubdtype(variable.dtype, np.floating)]
    # NaN -> 0 so an all-NaN variable ranges as zero; +/-inf survives and fails the limit.
    # An empty variable cannot be reduced, so it is left out and ranges as zero too.
    sized = [name for name in names if dataset[name].size]
    largest = abs(dataset[sized]).fillna(0.0).max().compute()
    limit = _INT16_MAX * scale

    encoding: dict[Hashable, dict[str, Any]] = {}
    packed: list[Hashable] = []
    unpacked: list[Hashable] = []
    for name in names:
        variable = dataset.variables[name]
        for key in _STALE_ENCODING_KEYS:
            variable.encoding.pop(key, None)

        if float(largest.get(name, 0.0)) <= limit:
            packed.append(name)
            chosen: dict[str, Any] = {
                "dtype": "int16",
                "scale_factor": scale,
                "add_offset": 0.0,
                "_FillValue": _INT16_FILL,
            }
        else:
            unpacked.append(name)
            chosen = {"dtype": "float32", "_FillValue": np.nan}
        chosen.update(zlib=True, complevel=complevel)
        if (chunksizes := _chunksizes(variable)) is not None:
            chosen["chunksizes"] = chunksizes
        encoding[name] = chosen

    _logger.info("netcdf_packing", int16=packed, float32=unpacked)
    return encoding


def write_netcdf_atomic(
    obj: xr.Dataset | xr.DataArray,
    output_file: str,
    *,
    engine: Literal["netcdf4", "scipy", "h5netcdf"] | None = None,
    pack: bool = False,
) -> None:
    """Write ``obj`` to ``output_file``, replacing it only once fully written.

    :param obj: the Dataset or DataArray to write; a DataArray must be named
    :param output_file: the NetCDF file to write
    :param engine: the NetCDF engine, or None to let xarray choose; packing
        needs an HDF5-backed one (netcdf4 or h5netcdf)
    :param pack: store the variables compressed, as int16 where they fit and
        float32 otherwise (see ``choose_netcdf_encoding``); experimental
    """
    output_path = Path(output_file)
    # A unique temporary per write: two CLI processes writing the same target
    # must not replace or delete each other's in-progress file.
    file_descriptor, temporary_file = tempfile.mkstemp(
        dir=output_path.parent, prefix=f"{output_path.name}.", suffix=".tmp"
    )
    os.close(file_descriptor)
    try:
        if pack:
            dataset = obj.to_dataset() if isinstance(obj, xr.DataArray) else obj
            dataset.to_netcdf(temporary_file, engine=engine, encoding=choose_netcdf_encoding(dataset))
        else:
            obj.to_netcdf(temporary_file, engine=engine)
        Path(temporary_file).replace(output_file)
    except BaseException:
        Path(temporary_file).unlink(missing_ok=True)
        raise

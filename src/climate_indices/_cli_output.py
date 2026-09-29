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
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import xarray as xr

from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.xarray_adapter import build_output_attrs


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


def write_netcdf_atomic(
    obj: xr.Dataset | xr.DataArray,
    output_file: str,
    *,
    engine: Literal["netcdf4", "scipy", "h5netcdf"] | None = None,
) -> None:
    """Write ``obj`` to ``output_file``, replacing it only once fully written.

    :param obj: the Dataset or DataArray to write
    :param output_file: the NetCDF file to write
    :param engine: the NetCDF engine, or None to let xarray choose
    """
    output_path = Path(output_file)
    # A unique temporary per write: two CLI processes writing the same target
    # must not replace or delete each other's in-progress file.
    file_descriptor, temporary_file = tempfile.mkstemp(
        dir=output_path.parent, prefix=f"{output_path.name}.", suffix=".tmp"
    )
    os.close(file_descriptor)
    try:
        obj.to_netcdf(temporary_file, engine=engine)
        Path(temporary_file).replace(output_file)
    except BaseException:
        Path(temporary_file).unlink(missing_ok=True)
        raise

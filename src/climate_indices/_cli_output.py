"""One output path for the CLI: attributes from the CF registry, atomic writes.

Both CLI backends -- the shared-memory NumPy route and the xarray route -- end
here. Attributes come from :data:`climate_indices.cf_metadata_registry.CF_METADATA`
through the same ``build_output_attrs`` layering the xarray adapters use, so a
CLI output carries the registry's long name, units and references plus the
library version and a history entry. Writes go beside the target and replace it
only once the whole file is on disk, so a failed computation cannot leave a
hollow file where an earlier output was.
"""

from __future__ import annotations

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
    temporary_file = f"{output_file}.tmp"
    try:
        obj.to_netcdf(temporary_file, engine=engine)
        Path(temporary_file).replace(output_file)
    except BaseException:
        Path(temporary_file).unlink(missing_ok=True)
        raise

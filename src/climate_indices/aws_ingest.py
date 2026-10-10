"""Available water capacity (AWS/AWC) ingest and harmonization for the Palmer water balance.

The Palmer two-layer soil moisture model takes one total available water capacity
per location, in inches, and splits it into a fixed 25.4 mm (1 in) surface layer
and the remainder as the underlying layer (see ``palmer.AWCTOP``). This module
loads candidate soil-available-water datasets, converts them to a single total in
millimetres of plant-available water, and resamples them onto a climate grid so
the resulting field can be injected into ``climate_indices.pdsi()`` /
``palmer.scpdsi()`` as the ``awc`` argument without touching the algorithm.

Verified sources
----------------
Only the items below were checked against primary documentation while this module
was written; every claim a loader depends on is listed here, and anything that
could not be verified is repeated in the returned ``aws_unverified`` attribute.

``usgs`` -- USGS "Soil properties dataset in the United States, Derived from 2020
gNATSGO database" (Boiko, Kagone & Senay, 2021, DOI 10.5066/P9TI3IS8). The
release metadata (``https://data.usgs.gov/datacatalog/metadata/USGS.5fd7c19cd34e30b9123cb51f.xml``)
states the raster ``awc_gNATSGO_US.tif`` holds a depth-weighted average AWC for
the upper 100 cm, produced from gNATSGO/gSSURGO, originally as cm/cm and
multiplied by 1000 to give **mm per metre**, then reprojected from Albers to
geographic (GCS WGS 1984). One metre of soil therefore makes the published value
numerically equal to total millimetres of available water over 100 cm:
``total_mm = value_mm_per_m * (1000 mm / 1000 mm)``. Native depth: 1000 mm.
The metadata also records that the processing resolution was resampled to
**90 m** (not the ~1 km the source rasters are sometimes described as), and that
missing cells -- including water bodies and built-up areas -- were **filled
upstream** by repeated focal statistics. That upstream fill cannot be
distinguished from real soil in the published product, so this module's own
``filled`` mask only reports holes it filled itself, never the publisher's.

``polaris`` -- POLARIS v1.0 soil properties (Chaney et al., 2019,
DOI 10.1029/2018WR022797), published at ``http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/``
(README verified 2026-10-09). Values are 1 arc-second (~30 m) GeoTIFF tiles split
into 1x1 degree chunks, one file per variable/statistic/depth layer, with
per-statistic files named ``mean``, ``mode``, ``p50``, ``p5`` and ``p95`` and six
depth layers: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm. POLARIS publishes
**no AWC variable**, so it is derived here per layer from the van Genuchten
parameters as ``theta(-33 kPa) - theta(-1500 kPa)``. Per the verified README,
``alpha`` is stored as **log10(kPa^-1)** (a 2019-06-02 erratum corrected it from
log10(cm^-1)) and ``n`` is linear and dimensionless; ``theta_r`` and ``theta_s``
are volumetric fractions (m3/m3). The default statistic is ``p50`` (median), and
the derivation is therefore the available water of **median marginal parameters**,
not the median of available water. The publisher's own client code
(``chaneyn/polaris_api_client``) confirms the same units and layers. License:
CC BY-NC 4.0 (non-commercial), though the plain HTTP tile listing does not repeat
the license text; do not redistribute a derived fixture without checking terms.

``gridmet`` -- The gridMET-derived CONUS PDSI product uses "a static soil water
holding capacity layer (top 1500mm) from STATSGO" as an input
(``https://developers.google.com/earth-engine/datasets/catalog/GRIDMET_DROUGHT``,
drought.gov). That layer is a *processing input to gridMET's PDSI*, not a
published gridMET variable, and ``GRIDMET/DROUGHT`` itself exposes only drought
indices (SPI, SPEI, EDDI, PDSI, Z). **No public download endpoint for that exact
STATSGO storage raster was found.** The gridMET *climate* archive
(``climatologylab.org/gridmet.html``) has no soil/AWC variable either; the
Climatology Lab dataset that does carry soil-water storage is TerraClimate,
whose storage is from Wang-Erlandsson et al. (2016) and is a different product.
The loader here therefore requires a user-supplied raster of the 0-1500 mm
STATSGO storage layer and raises a clear error when it is absent, rather than
silently substituting a different dataset. Native depth: 1500 mm.

Units
-----
Everything this module returns is **millimetres of plant-available water** over
the requested soil column (``aws``), on the climate grid's cell centres, with the
climate mask applied. Volumetric fractions (cm/cm, m3/m3) are dimensionless, so
fraction * thickness_mm gives millimetres directly. The Palmer entry points want
inches; use :func:`aws_mm_to_inches` (25.4 mm per inch) at that boundary.

Missing data
------------
No-data inside land (a cell the climate grid has data for) is filled from the
nearest valid neighbour and recorded in the returned boolean ``filled`` field.
True non-land -- water, or anywhere the climate grid itself is missing -- stays
missing so the soil field matches the climate mask exactly. Values are clipped to
a plausible range (:data:`MIN_AWS_MM` to :data:`MAX_AWS_MM`) and the clipped
counts are logged and recorded in ``attrs``; a cell at exactly 25.4 mm leaves the
Palmer model no underlying-layer capacity.

Depth
-----
``depth_mm="native"`` keeps each dataset's own column. A fixed depth is computed
by layer-thickness-weighted integration, truncating a partly included layer in
proportion to the depth included -- an assumption of uniform water content within
that layer, since none of these products publishes sub-layer horizons. A depth
deeper than a source's native column raises :class:`DepthUnavailableError`;
column totals are never extrapolated.

Caching
-------
Harmonized output is cached as two GeoTIFFs per (source, depth, climate grid) so
that remote reads happen once. Reading rasters and writing the cache need the
optional ``rioxarray``/``rasterio`` dependencies (``pip install
'climate-indices[aws]'``); the arithmetic in this module needs only
numpy/scipy/xarray.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import xarray as xr
from scipy import ndimage

from climate_indices.exceptions import ClimateIndicesError
from climate_indices.logging_config import get_logger

__all__ = [
    "AWS_SOURCES",
    "MAX_AWS_MM",
    "MIN_AWS_MM",
    "NATIVE_DEPTH",
    "POLARIS_LAYERS_MM",
    "SURFACE_LAYER_MM",
    "AwsIngestError",
    "AwsSourceSpec",
    "DepthUnavailableError",
    "EarthEngineUnavailableError",
    "HarmonizedAws",
    "SourceUnavailableError",
    "area_weighted_mean",
    "aws_mm_to_inches",
    "clip_aws_bounds",
    "fill_land_holes",
    "finalize_aws",
    "harmonize_aws",
    "integrate_depth",
    "layer_storage_mm",
    "load_aws",
    "register_source",
    "van_genuchten_available_water",
]

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

#: Palmer's fixed surface-layer capacity, in millimetres (1 inch). The two-layer
#: model breaks below this, so it is also the lower plausibility clip bound.
SURFACE_LAYER_MM = 25.4

#: Lower clip bound. A value at exactly this bound leaves zero underlying-layer
#: capacity; those cells are counted and logged separately.
MIN_AWS_MM = SURFACE_LAYER_MM

#: Upper plausibility clip bound, in millimetres of plant-available water. Well
#: above any of these products' native columns, so it only catches nonsense.
MAX_AWS_MM = 2000.0

#: Sentinel selecting each dataset's own soil column.
NATIVE_DEPTH = "native"

#: Layer coordinate name used by the single-layer (column-total) sources.
_SINGLE_LAYER = "column"

#: POLARIS v1.0 depth layers: (name, top_mm, bottom_mm), verified against the
#: publisher README.
POLARIS_LAYERS_MM: tuple[tuple[str, float, float], ...] = (
    ("0_5", 0.0, 50.0),
    ("5_15", 50.0, 150.0),
    ("15_30", 150.0, 300.0),
    ("30_60", 300.0, 600.0),
    ("60_100", 600.0, 1000.0),
    ("100_200", 1000.0, 2000.0),
)

#: Van Genuchten suction potentials bounding plant-available water, in kPa.
FIELD_CAPACITY_KPA = 33.0
WILTING_POINT_KPA = 1500.0

#: POLARIS statistic directory to read; "p50" is the median.
POLARIS_STATISTIC = "p50"

#: Target-grid latitude rows per POLARIS read. Read bounds come from these rows'
#: cell edges, so each strip is a bounded GDAL window over the 30 m tiles.
POLARIS_STRIP_ROWS = 4

#: Environment variable naming an Earth Engine asset to read POLARIS from. The
#: server-side ``reduceResolution`` path is not implemented; setting this raises
#: a clear error rather than silently reading tiles instead.
POLARIS_EE_ASSET_ENV = "POLARIS_EE_ASSET"

#: Attempts per remote POLARIS tile read. The publisher's plain HTTP server
#: occasionally returns a truncated GeoTIFF strip, which GDAL reports as an
#: ``OSError``; a bounded retry keeps a long ingest from losing all its work.
POLARIS_READ_ATTEMPTS = 3

#: Seconds of linear backoff before each retried tile read.
POLARIS_READ_BACKOFF_S = 2.0

#: Absolute tolerance, in millimetres, for "this depth is within the native column".
_DEPTH_TOLERANCE_MM = 1e-6

Bounds = tuple[float, float, float, float]  # (west, south, east, north) in degrees


class AwsIngestError(ClimateIndicesError):
    """Base exception for available-water-capacity ingest failures."""


class DepthUnavailableError(AwsIngestError):
    """Raised when a requested soil depth exceeds a source's native column."""


class SourceUnavailableError(AwsIngestError):
    """Raised when a source's data cannot be read in this environment."""


class EarthEngineUnavailableError(SourceUnavailableError):
    """Raised when Earth Engine is requested but not usable."""


@dataclass(frozen=True)
class HarmonizedAws:
    """Total available water capacity on a climate grid, plus its fill mask.

    Attributes:
        aws: Total plant-available water in millimetres over the requested depth;
            dimensions are the climate grid's cell dimensions, in the climate
            grid's order, and non-land cells are missing.
        filled: Boolean field, aligned with ``aws``, that is True exactly where
            ``aws`` was missing inside land and was filled from neighbours.
    """

    aws: xr.DataArray
    filled: xr.DataArray

    @property
    def source(self) -> str:
        """Name of the registered source that produced this field."""
        return str(self.aws.attrs["aws_source"])

    @property
    def depth_mm(self) -> float:
        """Soil column depth the total covers, in millimetres."""
        return float(self.aws.attrs["aws_depth_mm"])


# ---------------------------------------------------------------------------
# unit and depth conversion
# ---------------------------------------------------------------------------


def layer_storage_mm(
    fractions: xr.DataArray,
    layers_mm: Sequence[tuple[str, float, float]],
    *,
    layer_dim: str = "layer",
) -> xr.DataArray:
    """Convert per-layer volumetric water fractions to stored water depth in mm.

    A volumetric fraction (cm/cm or m3/m3) is dimensionless, so multiplying by a
    layer's thickness in millimetres gives millimetres of stored water.

    Args:
        fractions: Volumetric available-water fractions with a ``layer_dim``
            dimension whose coordinate matches the layer names in ``layers_mm``.
        layers_mm: Layer description as ``(name, top_mm, bottom_mm)`` triples.
        layer_dim: Name of the layer dimension in ``fractions``.

    Returns:
        Stored water depth in millimetres, with ``layer_dim`` summed away.

    Raises:
        AwsIngestError: If a named layer has non-positive thickness or a layer in
            ``fractions`` is missing from ``layers_mm``.
    """
    thickness = _layer_thicknesses_mm([str(name) for name in fractions[layer_dim].values], layers_mm)
    scaled = fractions * xr.DataArray(thickness, coords={layer_dim: fractions[layer_dim].values}, dims=[layer_dim])
    scaled = scaled.assign_attrs(units="mm", long_name="Stored plant-available water per soil layer")
    return cast(xr.DataArray, scaled)


def van_genuchten_available_water(
    alpha_log10_kpa: npt.ArrayLike,
    n: npt.ArrayLike,
    theta_r: npt.ArrayLike,
    theta_s: npt.ArrayLike,
    *,
    field_capacity_kpa: float = FIELD_CAPACITY_KPA,
    wilting_point_kpa: float = WILTING_POINT_KPA,
) -> npt.NDArray[np.float64]:
    """Available water fraction from van Genuchten parameters.

    Computes ``theta(field_capacity) - theta(wilting_point)`` with

    ``theta(h) = theta_r + (theta_s - theta_r) * (1 + (alpha * |h|) ** n) ** -(1 - 1 / n)``

    for no-residual-capillary formulation (Mualem, 1976; van Genuchten, 1980),
    which is the closed form POLARIS's parameter set supports.

    The arithmetic runs in place over four scratch buffers instead of building a
    chain of full-grid temporaries, because these fields are 30 m rasters and the
    expression's intermediates otherwise cost several times the inputs.

    Args:
        alpha_log10_kpa: Van Genuchten scale parameter as stored by POLARIS,
            ``log10(kPa^-1)``.
        n: Van Genuchten pore-size distribution parameter, dimensionless and
            linear (not log-scaled).
        theta_r: Residual volumetric water content, m3/m3.
        theta_s: Saturated volumetric water content, m3/m3.
        field_capacity_kpa: Suction defining field capacity, in kPa.
        wilting_point_kpa: Suction defining the wilting point, in kPa.

    Returns:
        Available water fraction (m3/m3), dimensionless, as a float array.

    Raises:
        AwsIngestError: If ``n`` is not greater than one, which the closed form
            cannot represent.
    """
    shape = np.broadcast_shapes(*(np.shape(values) for values in (alpha_log10_kpa, n, theta_r, theta_s)))
    shape_array = np.asarray(n, dtype=float)
    if np.any(shape_array <= 1.0):
        raise AwsIngestError(f"van Genuchten n must be greater than 1, got minimum {np.min(shape_array)!r}")
    exponent = 1.0 - (1.0 / shape_array)
    work = np.empty(shape, dtype=float)
    delta = np.empty(shape, dtype=float)
    theta_field_capacity = np.empty(shape, dtype=float)
    theta_wilting_point = np.empty(shape, dtype=float)
    for out, suction in (
        (theta_field_capacity, field_capacity_kpa),
        (theta_wilting_point, wilting_point_kpa),
    ):
        _theta_into(out, work, delta, alpha_log10_kpa, shape_array, exponent, theta_r, theta_s, suction)
    np.subtract(theta_field_capacity, theta_wilting_point, out=theta_field_capacity)
    return theta_field_capacity


def _theta_into(
    out: npt.NDArray[np.float64],
    work: npt.NDArray[np.float64],
    delta: npt.NDArray[np.float64],
    alpha_log10_kpa: npt.ArrayLike,
    n: npt.NDArray[np.float64],
    exponent: npt.NDArray[np.float64],
    theta_r: npt.ArrayLike,
    theta_s: npt.ArrayLike,
    suction_kpa: float,
) -> None:
    """Write ``theta(suction)`` into ``out``, using ``work`` and ``delta`` as scratch."""
    np.power(10.0, alpha_log10_kpa, out=work)
    np.multiply(work, suction_kpa, out=work)
    np.power(work, n, out=work)
    np.add(work, 1.0, out=work)
    np.power(work, -exponent, out=work)
    np.subtract(theta_s, theta_r, out=delta)
    np.multiply(delta, work, out=delta)
    np.add(delta, theta_r, out=out)


def integrate_depth(
    storage_mm: xr.DataArray,
    layers_mm: Sequence[tuple[str, float, float]],
    depth_mm: float,
    *,
    layer_dim: str = "layer",
    native_depth_mm: float | None = None,
) -> xr.DataArray:
    """Integrate per-layer stored water to a total over a soil column depth.

    A layer fully above ``depth_mm`` contributes all of its stored water; a layer
    the depth falls inside contributes in proportion to the fraction of that
    layer included, which assumes uniform water content within the layer.

    Args:
        storage_mm: Per-layer stored water in millimetres, with ``layer_dim``.
        layers_mm: Layer description as ``(name, top_mm, bottom_mm)`` triples.
        depth_mm: Requested column depth from the surface, in millimetres.
        layer_dim: Name of the layer dimension in ``storage_mm``.
        native_depth_mm: The source's own column depth, in millimetres; defaults
            to the deepest bound in ``layers_mm``. Pass it explicitly when
            ``storage_mm`` carries only the layers above ``depth_mm``.

    Returns:
        Total stored water over ``depth_mm``, with ``layer_dim`` summed away.

    Raises:
        AwsIngestError: If ``depth_mm`` is not positive.
        DepthUnavailableError: If ``depth_mm`` exceeds the native column.
    """
    if not np.isfinite(depth_mm) or depth_mm <= 0.0:
        raise AwsIngestError(f"soil depth must be a positive number of millimetres, got {depth_mm!r}")
    source_depth_mm = layers_mm[-1][2] if native_depth_mm is None else native_depth_mm
    if depth_mm > source_depth_mm + _DEPTH_TOLERANCE_MM:
        raise DepthUnavailableError(
            f"requested soil depth of {depth_mm:.1f} mm exceeds the source column of "
            f"{source_depth_mm:.1f} mm; column totals are not extrapolated to deeper soil"
        )
    names = [str(name) for name in storage_mm[layer_dim].values]
    included = xr.DataArray(
        [_included_fraction(name, layers_mm, depth_mm) for name in names],
        coords={layer_dim: storage_mm[layer_dim].values},
        dims=[layer_dim],
    )
    total = (storage_mm * included).sum(dim=layer_dim, skipna=False)
    return cast(
        xr.DataArray,
        total.assign_attrs(units="mm", long_name="Total plant-available water", aws_depth_mm=float(depth_mm)),
    )


def _included_fraction(name: str, layers_mm: Sequence[tuple[str, float, float]], depth_mm: float) -> float:
    """Fraction of a layer's thickness that lies inside the requested column depth.

    The single definition of the partial-layer rule: a layer above the depth counts
    in full, a layer the depth falls inside counts in proportion to the part
    included, and a layer below the depth does not count at all.
    """
    _, top_mm, bottom_mm = _layer_bounds([name], layers_mm)[0]
    included_mm = min(bottom_mm, depth_mm) - top_mm
    return float(np.clip(included_mm / (bottom_mm - top_mm), 0.0, 1.0))


def aws_mm_to_inches(aws_mm: xr.DataArray | npt.ArrayLike) -> xr.DataArray | npt.NDArray[np.float64]:
    """Convert millimetres of available water to the inches the Palmer code takes."""
    if isinstance(aws_mm, xr.DataArray):
        inches = aws_mm / SURFACE_LAYER_MM
        return cast(
            xr.DataArray,
            inches.assign_attrs(units="inches", long_name="Total available water capacity"),
        )
    return np.asarray(aws_mm, dtype=float) / SURFACE_LAYER_MM


def _resolve_depth(depth_mm: float | str, native_depth_mm: float) -> float:
    """Resolve a requested depth, rejecting an unknown string sentinel."""
    if isinstance(depth_mm, str):
        if depth_mm != NATIVE_DEPTH:
            raise AwsIngestError(f"unknown soil depth {depth_mm!r}; expected {NATIVE_DEPTH!r} or millimetres")
        return native_depth_mm
    return float(depth_mm)


def _layer_bounds(
    names: Sequence[str],
    layers_mm: Sequence[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Resolve layer names to their ``(name, top_mm, bottom_mm)`` descriptions."""
    lookup = {name: (name, top_mm, bottom_mm) for name, top_mm, bottom_mm in layers_mm}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise AwsIngestError(f"unknown soil layer(s) {missing!r}; expected one of {sorted(lookup)!r}")
    return [lookup[name] for name in names]


def _layer_thicknesses_mm(
    names: Sequence[str],
    layers_mm: Sequence[tuple[str, float, float]],
) -> list[float]:
    """Layer thicknesses in millimetres, rejecting non-positive thickness."""
    thicknesses: list[float] = []
    for name, top_mm, bottom_mm in _layer_bounds(names, layers_mm):
        thickness_mm = bottom_mm - top_mm
        if thickness_mm <= 0.0:
            raise AwsIngestError(f"soil layer {name!r} has non-positive thickness {thickness_mm!r} mm")
        thicknesses.append(thickness_mm)
    return thicknesses


# ---------------------------------------------------------------------------
# harmonization: area-weighted aggregation onto the climate grid
# ---------------------------------------------------------------------------


def area_weighted_mean(
    source: xr.DataArray,
    lat_values: npt.ArrayLike,
    lon_values: npt.ArrayLike,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray:
    """Average a fine rectilinear field onto a coarser rectilinear grid by area.

    Each target cell receives the area-weighted mean of the source cells it
    overlaps, never a nearest-neighbour sample. Cell edges are derived from cell
    centres (midpoints, with the outer edges extended by half a cell), so the
    source and target cells need not nest. Latitude weights are ``cos(latitude)``
    of the source cell centre, which is what makes high-latitude cells count for
    less; longitude overlap length supplies the other half of the cell area.
    Missing source cells are excluded from both numerator and denominator.

    Args:
        source: Fine field with ``lat_dim`` and ``lon_dim`` dimensions; its
            coordinates may run in either direction.
        lat_values: Target latitude cell centres (any order).
        lon_values: Target longitude cell centres (any order).
        lat_dim: Latitude dimension name.
        lon_dim: Longitude dimension name.

    Returns:
        The area-weighted mean on the target cell centres, missing where no
        source cell overlaps a target cell.

    Raises:
        AwsIngestError: If ``source`` is not two-dimensional, or a coordinate has
            duplicate values, non-finite values, or is not monotonic in one of the
            two directions.
    """
    values = np.asarray(source.values, dtype=float)
    if values.ndim != 2:
        raise AwsIngestError(f"area-weighted averaging needs a 2-D field, got shape {values.shape!r}")
    source_lat = _validated_axis(np.asarray(source[lat_dim].values, dtype=float), lat_dim)
    source_lon = _validated_axis(np.asarray(source[lon_dim].values, dtype=float), lon_dim)
    target_lat = _validated_axis(np.asarray(lat_values, dtype=float), "target latitude")
    target_lon = _validated_axis(np.asarray(lon_values, dtype=float), "target longitude")

    values, source_lat, source_lon = _orient_ascending(values, source_lat, source_lon)
    target_lat_ascending, target_lat_reversed = _ascending(target_lat)
    target_lon_ascending, target_lon_reversed = _ascending(target_lon)

    source_lat_edges = _cell_edges(source_lat)
    source_lon_edges = _cell_edges(source_lon)
    target_lat_edges = _target_edges(target_lat_ascending, source_lat_edges)
    target_lon_edges = _target_edges(target_lon_ascending, source_lon_edges)

    lat_weights = _overlap_weights(source_lat_edges, target_lat_edges, np.cos(np.radians(source_lat)))
    lon_weights = _overlap_weights(source_lon_edges, target_lon_edges, None)

    valid = np.isfinite(values)
    numerator = lat_weights @ np.where(valid, values, 0.0) @ lon_weights.T
    denominator = lat_weights @ valid.astype(float) @ lon_weights.T
    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = np.where(denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), np.nan)
    if target_lat_reversed:
        averaged = averaged[::-1]
    if target_lon_reversed:
        averaged = averaged[:, ::-1]

    return xr.DataArray(
        averaged,
        coords={lat_dim: target_lat, lon_dim: target_lon},
        dims=[lat_dim, lon_dim],
        attrs={**source.attrs, "aws_area_weighted": "true"},
    )


def fill_land_holes(
    values: npt.ArrayLike,
    land_mask: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Fill missing land cells from their nearest valid neighbour.

    Only cells that are land (``land_mask`` True) and missing are filled; missing
    cells outside land -- water, or anywhere the climate grid itself is missing --
    are left missing so the soil field keeps the climate mask. Nearest is measured
    in array-index distance, not great-circle distance, which is adequate at
    climate-grid spacing.

    Args:
        values: Field with missing cells as NaN.
        land_mask: Boolean field, aligned with ``values``, True on land.

    Returns:
        A tuple ``(filled_values, filled_flag)`` where ``filled_flag`` is True
        exactly on the cells that were filled. When no valid cell exists at all,
        the field is returned unchanged and no cell is flagged.
    """
    field = np.asarray(values, dtype=float).copy()
    land = np.asarray(land_mask, dtype=bool)
    if field.shape != land.shape:
        raise AwsIngestError(f"value shape {field.shape!r} does not match land mask shape {land.shape!r}")
    gaps = np.isnan(field) & land
    if not gaps.any() or np.isfinite(field).sum() == 0:
        return field, np.zeros(field.shape, dtype=bool)

    nearest = cast(
        npt.NDArray[np.int32],
        ndimage.distance_transform_edt(np.isnan(field), return_distances=False, return_indices=True),
    )
    filled_values = field[tuple(nearest)]
    filled_values[~np.isfinite(field[tuple(nearest)])] = np.nan
    filled_flag = gaps & np.isfinite(filled_values)
    field[filled_flag] = filled_values[filled_flag]
    field[~land] = np.nan
    return field, filled_flag


def clip_aws_bounds(
    values: npt.ArrayLike,
    *,
    lower_mm: float = MIN_AWS_MM,
    upper_mm: float = MAX_AWS_MM,
) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
    """Clip a field to plausible available-water bounds and report what moved.

    Args:
        values: Field in millimetres, with missing cells as NaN.
        lower_mm: Lower plausibility bound, in millimetres. Defaults to Palmer's
            25.4 mm surface layer, below which the two-layer model has no
            underlying layer to hold water.
        upper_mm: Upper plausibility bound, in millimetres.

    Returns:
        A tuple ``(clipped, report)``. ``report`` holds ``clipped_low``,
        ``clipped_high``, ``at_surface_capacity``, ``min_before_mm`` and
        ``max_before_mm``; the two ``*_before`` entries are NaN when every value
        is missing.
    """
    field = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(field)
    minimum_before = float(field[finite].min()) if finite.any() else float("nan")
    maximum_before = float(field[finite].max()) if finite.any() else float("nan")
    low = finite & (field < lower_mm)
    high = finite & (field > upper_mm)
    if low.any():
        field[low] = lower_mm
    if high.any():
        field[high] = upper_mm
    report = {
        "clipped_low": float(low.sum()),
        "clipped_high": float(high.sum()),
        "at_surface_capacity": float((finite & (field <= lower_mm)).sum()),
        "min_before_mm": minimum_before,
        "max_before_mm": maximum_before,
    }
    return field, report


def harmonize_aws(
    storage_mm: xr.DataArray,
    climate: xr.DataArray,
    *,
    source: str,
    layers_mm: Sequence[tuple[str, float, float]],
    native_depth_mm: float,
    depth_mm: float | str = NATIVE_DEPTH,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    unverified: Sequence[str] = (),
) -> HarmonizedAws:
    """Integrate, aggregate, clip, and mask a per-layer soil field for the Palmer model.

    Args:
        storage_mm: Per-layer stored water in millimetres, with a ``layer``
            dimension and ``lat_dim``/``lon_dim`` coordinates.
        climate: A climate field on the target grid (for example monthly
            precipitation); its non-missing footprint defines land, and its
            ``lat_dim``/``lon_dim`` coordinates define the target cells.
        source: Registered source name, recorded in ``attrs``.
        layers_mm: Layer description as ``(name, top_mm, bottom_mm)`` triples.
        native_depth_mm: The source's own column depth, in millimetres.
        depth_mm: Requested column depth in millimetres, or ``"native"``.
        lat_dim: Latitude dimension name.
        lon_dim: Longitude dimension name.
        unverified: Source facts that could not be verified, recorded in ``attrs``.

    Returns:
        The harmonized total field and its fill mask.

    Raises:
        DepthUnavailableError: If the requested depth exceeds ``native_depth_mm``.
        AwsIngestError: If the climate field's dimensions are unusable.
    """
    resolved_depth_mm = _resolve_depth(depth_mm, native_depth_mm)
    total_mm = integrate_depth(storage_mm, layers_mm, resolved_depth_mm)
    target_lat = climate[lat_dim].values
    target_lon = climate[lon_dim].values
    averaged = area_weighted_mean(total_mm, target_lat, target_lon, lat_dim=lat_dim, lon_dim=lon_dim)
    return finalize_aws(
        averaged,
        climate,
        source=source,
        layers_mm=layers_mm,
        native_depth_mm=native_depth_mm,
        depth_mm=resolved_depth_mm,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        unverified=unverified,
    )


def finalize_aws(
    total_mm: xr.DataArray,
    climate: xr.DataArray,
    *,
    source: str,
    layers_mm: Sequence[tuple[str, float, float]],
    native_depth_mm: float,
    depth_mm: float,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    unverified: Sequence[str] = (),
) -> HarmonizedAws:
    """Clip, fill, and mask an already-aggregated total for the Palmer model.

    This is the second half of :func:`harmonize_aws`, split out for sources that
    aggregate in windows and so arrive already on the climate grid.

    Args:
        total_mm: Total stored water in millimetres on the climate grid.
        climate: A climate field on the target grid; its non-missing footprint
            defines land.
        source: Registered source name, recorded in ``attrs``.
        layers_mm: Layer description as ``(name, top_mm, bottom_mm)`` triples.
        native_depth_mm: The source's own column depth, in millimetres.
        depth_mm: Requested column depth in millimetres.
        lat_dim: Latitude dimension name.
        lon_dim: Longitude dimension name.
        unverified: Source facts that could not be verified, recorded in ``attrs``.

    Returns:
        The finalized total field and its fill mask.
    """
    resolved_depth_mm = float(depth_mm)
    land_mask = _climate_land_mask(climate, lat_dim=lat_dim, lon_dim=lon_dim)
    averaged = total_mm.where(land_mask)
    target_lat = climate[lat_dim].values
    target_lon = climate[lon_dim].values
    clipped, report = clip_aws_bounds(averaged.values)
    if report["clipped_low"]:
        _logger.warning(
            "aws_clipped_low",
            source=source,
            depth_mm=resolved_depth_mm,
            cells=int(report["clipped_low"]),
            lower_mm=MIN_AWS_MM,
        )
    if report["clipped_high"]:
        _logger.warning(
            "aws_clipped_high",
            source=source,
            depth_mm=resolved_depth_mm,
            cells=int(report["clipped_high"]),
            upper_mm=MAX_AWS_MM,
        )
    if report["at_surface_capacity"]:
        _logger.warning(
            "aws_at_surface_capacity",
            source=source,
            depth_mm=resolved_depth_mm,
            cells=int(report["at_surface_capacity"]),
            detail="no underlying-layer capacity for these cells",
        )

    filled_values, filled_flag = fill_land_holes(clipped, land_mask.values)
    attrs: dict[str, Any] = {
        "long_name": "Total available water capacity",
        "units": "mm",
        "aws_source": source,
        "aws_depth_mm": resolved_depth_mm,
        "aws_native_depth_mm": float(native_depth_mm),
        "aws_layers_mm": json.dumps([[name, top, bottom] for name, top, bottom in layers_mm]),
        "aws_area_weighted": "true",
        "aws_filled_cells": int(filled_flag.sum()),
        "aws_clipped_low": int(report["clipped_low"]),
        "aws_clipped_high": int(report["clipped_high"]),
        "aws_cells_at_surface_capacity": int(report["at_surface_capacity"]),
        "aws_min_before_clip_mm": report["min_before_mm"],
        "aws_max_before_clip_mm": report["max_before_mm"],
        "aws_unverified": json.dumps(list(unverified)),
    }
    aws = xr.DataArray(
        filled_values,
        coords={lat_dim: target_lat, lon_dim: target_lon},
        dims=[lat_dim, lon_dim],
        attrs=attrs,
    )
    filled = xr.DataArray(
        filled_flag,
        coords={lat_dim: target_lat, lon_dim: target_lon},
        dims=[lat_dim, lon_dim],
        attrs={"long_name": "Cell filled from neighbours inside land", "units": "1"},
    )
    return HarmonizedAws(aws=aws, filled=filled)


def _climate_land_mask(climate: xr.DataArray, *, lat_dim: str, lon_dim: str) -> xr.DataArray:
    """Boolean field that is True where the climate grid carries data."""
    missing_dims = [dim for dim in (lat_dim, lon_dim) if dim not in climate.dims]
    if missing_dims:
        raise AwsIngestError(f"climate field is missing dimension(s) {missing_dims!r}; it has {climate.dims!r}")
    other_dims = [dim for dim in climate.dims if dim not in (lat_dim, lon_dim)]
    mask = climate.notnull()
    if other_dims:
        mask = mask.any(dim=other_dims)
    return cast(xr.DataArray, mask)


def _validated_axis(values: npt.NDArray[np.float64], name: str) -> npt.NDArray[np.float64]:
    """Reject an unusable coordinate axis."""
    if values.ndim != 1:
        raise AwsIngestError(f"{name} must be one-dimensional, got shape {values.shape!r}")
    if not np.isfinite(values).all():
        raise AwsIngestError(f"{name} contains non-finite cell centres")
    if values.size > 1 and np.unique(values).size != values.size:
        raise AwsIngestError(f"{name} contains duplicate cell centres")
    return values


def _ascending(values: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], bool]:
    """Return ``(ascending values, was reversed)``."""
    if values.size > 1 and values[0] > values[-1]:
        return values[::-1], True
    return values, False


def _orient_ascending(
    values: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    lon: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Flip a field and its axes into ascending latitude and longitude order."""
    lat_ascending, lat_reversed = _ascending(lat)
    lon_ascending, lon_reversed = _ascending(lon)
    if lat_reversed:
        values = values[::-1]
    if lon_reversed:
        values = values[:, ::-1]
    return values, lat_ascending, lon_ascending


def _cell_edges(centres: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cell edges from ascending cell centres, extending the outer half-cells."""
    if centres.size < 2:
        raise AwsIngestError(
            "at least two cell centres are needed to derive cell edges and therefore cell areas; "
            f"got a single centre at {centres!r}"
        )
    midpoints = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate(
        [[centres[0] - (midpoints[0] - centres[0])], midpoints, [centres[-1] + (centres[-1] - midpoints[-1])]]
    )


def _target_edges(centres: npt.NDArray[np.float64], source_edges: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Target cell edges, spanning the source extent for a single-cell axis."""
    if centres.size == 1:
        return np.array([source_edges[0], source_edges[-1]])
    return _cell_edges(centres)


def _overlap_weights(
    source_edges: npt.NDArray[np.float64],
    target_edges: npt.NDArray[np.float64],
    source_weights: npt.NDArray[np.float64] | None,
) -> sp.csr_matrix:
    """Sparse row-normalized overlap matrix from source cells to target cells."""
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for target in range(target_edges.size - 1):
        low, high = target_edges[target], target_edges[target + 1]
        overlap = np.minimum(source_edges[1:], high) - np.maximum(source_edges[:-1], low)
        overlap = np.clip(overlap, 0.0, None)
        selected = np.nonzero(overlap)[0]
        if selected.size == 0:
            continue
        cell_weights = overlap[selected] if source_weights is None else overlap[selected] * source_weights[selected]
        rows.extend([target] * selected.size)
        columns.extend(selected.tolist())
        weights.extend(cell_weights.tolist())
    matrix = sp.csr_matrix(
        (weights, (rows, columns)),
        shape=(target_edges.size - 1, source_edges.size - 1),
    )
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.where(totals > 0.0, totals, 1.0)) @ matrix


# ---------------------------------------------------------------------------
# raster access (optional rioxarray/rasterio dependency)
# ---------------------------------------------------------------------------

_CRS = "EPSG:4326"


def _require_rio() -> Any:
    """Import rioxarray, or explain how to install it."""
    try:
        import rioxarray  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SourceUnavailableError(
            "reading soil rasters and writing the harmonized cache need the optional "
            "geospatial dependencies; install them with: pip install 'climate-indices[aws]'"
        ) from error
    return rioxarray


def _open_raster_window(path_or_url: str, bounds: Bounds | None, *, eager: bool = True) -> xr.DataArray:
    """Open one raster (optionally windowed) as a lat/lon field.

    Reads run through GDAL, so a remote COG/GeoTIFF may be opened over HTTP and
    only the requested window is read instead of downloading the file whole; a
    30 m CONUS mosaic is never materialized in memory. ``eager`` computes the
    window immediately, which bounds memory by that window and releases the
    dataset handle instead of leaving a dask graph holding it open.
    """
    rioxarray = _require_rio()
    field = rioxarray.open_rasterio(
        path_or_url,
        chunks={"x": 2048, "y": 2048},
        lock=False,
    )
    field = field.squeeze("band", drop=True)
    if "x" in field.dims and "y" in field.dims:
        field = field.rename({"x": "lon", "y": "lat"})
    if field.rio.crs is None:
        raise SourceUnavailableError(f"raster {path_or_url!r} has no CRS; cannot place it on the climate grid")
    if str(field.rio.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
        field = field.rio.reproject(_CRS, resampling=_average_resampling())
    # a declared nodata sentinel (POLARIS tiles declare -9999) must never be read as
    # a soil value; missing data is better tracked than silently averaged in
    nodata = field.rio.nodata
    if nodata is not None and np.isfinite(nodata):
        field = field.where(field != nodata)
    # a raster's axes are usually north-to-south and west-to-east; slices below
    # assume ascending order, so flip the descending case
    for dim in ("lat", "lon"):
        if dim in field.dims and field[dim].size > 1 and float(field[dim][0]) > float(field[dim][-1]):
            field = field.isel({dim: slice(None, None, -1)})
    if bounds is not None:
        west, south, east, north = bounds
        # Keep intersecting pixels; area weights trim their partial overlaps.
        half_lat = float(field.lat[1] - field.lat[0]) / 2 if field.sizes["lat"] > 1 else 0.0
        half_lon = float(field.lon[1] - field.lon[0]) / 2 if field.sizes["lon"] > 1 else 0.0
        field = field.sel(lat=slice(south - half_lat, north + half_lat), lon=slice(west - half_lon, east + half_lon))
    field = cast(xr.DataArray, field.rename("values"))
    if eager:
        return cast(xr.DataArray, field.compute())
    return field


def _open_tile_with_retry(tile: Path | str, bounds: Bounds, *, attempts: int = POLARIS_READ_ATTEMPTS) -> xr.DataArray:
    """Read one tile window, retrying a transient transport failure.

    A truncated remote GeoTIFF strip and an unreachable server are both reported
    by GDAL as :class:`OSError`; only the former is worth retrying, but a bounded
    retry of either cannot loop forever.

    Args:
        tile: Local path or HTTP URL of the tile.
        bounds: ``(west, south, east, north)`` window to read.
        attempts: Total read attempts before giving up.

    Returns:
        The requested window as a lat/lon field.

    Raises:
        SourceUnavailableError: If every attempt failed.
    """
    for attempt in range(1, attempts + 1):
        try:
            return _open_raster_window(str(tile), bounds)
        except OSError as error:
            if attempt == attempts:
                raise SourceUnavailableError(f"could not read {tile} after {attempts} attempts: {error}") from error
            _logger.warning("polaris_tile_read_retry", tile=str(tile), attempt=attempt, error=str(error))
            time.sleep(POLARIS_READ_BACKOFF_S * attempt)
    raise AssertionError("unreachable: the loop either returns or raises")


def _average_resampling() -> Any:
    """GDAL's averaging resampler, used for any required CRS transform."""
    try:
        from rasterio.enums import Resampling
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SourceUnavailableError(
            "reprojecting a soil raster needs rasterio; install it with: pip install 'climate-indices[aws]'"
        ) from error
    return Resampling.average


def _write_cache(cache_dir: Path, key: str, result: HarmonizedAws, *, lat_dim: str, lon_dim: str) -> None:
    """Write a harmonized field and its fill mask as two GeoTIFFs."""
    _require_rio()

    cache_dir.mkdir(parents=True, exist_ok=True)
    aws = result.aws.rename({lat_dim: "y", lon_dim: "x"}).rename("aws")
    aws.rio.write_crs(_CRS).rio.to_raster(cache_dir / f"{key}_aws.tif", driver="GTiff")
    filled = result.filled.rename({lat_dim: "y", lon_dim: "x"}).rename("filled")
    filled.astype("uint8").rio.write_crs(_CRS).rio.to_raster(cache_dir / f"{key}_filled.tif", driver="GTiff")


def _read_cache(cache_dir: Path, key: str, *, lat_dim: str, lon_dim: str) -> HarmonizedAws | None:
    """Read a cached harmonized field, or None when it is absent."""
    aws_path = cache_dir / f"{key}_aws.tif"
    filled_path = cache_dir / f"{key}_filled.tif"
    if not (aws_path.exists() and filled_path.exists()):
        return None
    rioxarray = _require_rio()
    aws = rioxarray.open_rasterio(aws_path, lock=False).squeeze("band", drop=True)
    filled = rioxarray.open_rasterio(filled_path, lock=False).squeeze("band", drop=True)
    aws = aws.rename({"x": lon_dim, "y": lat_dim}).compute()
    filled = filled.rename({"x": lon_dim, "y": lat_dim}).compute()
    return HarmonizedAws(aws=aws, filled=filled.astype(bool))


def grid_signature(lat_values: npt.ArrayLike, lon_values: npt.ArrayLike, *, lat_dim: str, lon_dim: str) -> str:
    """Stable hash of a climate grid's cell centres, used as a cache key part."""
    digest = hashlib.sha256()
    digest.update(f"{lat_dim}:{lon_dim}:".encode())
    for values in (lat_values, lon_values):
        axis = np.asarray(values, dtype=np.float64)
        digest.update(np.round(axis, 9).tobytes())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# ingest adapters
# ---------------------------------------------------------------------------

Loader = Callable[..., HarmonizedAws]


@dataclass(frozen=True)
class AwsSourceSpec:
    """One soil-water source: its native column, layers, and reader.

    Attributes:
        name: Source key used by :func:`load_aws`.
        native_depth_mm: Depth of the source's own soil column, in millimetres.
        layers_mm: Layer description as ``(name, top_mm, bottom_mm)`` triples.
        load: Reader taking ``(climate, depth_mm, raw_dir)`` keyword arguments and
            returning a harmonized field.
        note: One-line description of what the source is.
        unverified: Source facts that could not be verified; recorded in ``attrs``.
    """

    name: str
    native_depth_mm: float
    layers_mm: tuple[tuple[str, float, float], ...]
    load: Loader
    note: str
    unverified: tuple[str, ...] = ()


#: Registered sources, keyed by the ``aws_source`` argument of :func:`load_aws`.
AWS_SOURCES: dict[str, AwsSourceSpec] = {}


def register_source(spec: AwsSourceSpec) -> AwsSourceSpec:
    """Register a soil-water source, replacing any existing entry of that name."""
    AWS_SOURCES[spec.name] = spec
    return spec


def _gridmet_aws(
    climate: xr.DataArray,
    depth_mm: float | str,
    raw_dir: Path | None,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> HarmonizedAws:
    """Read the STATSGO 0-1500 mm storage layer used by the gridMET PDSI product."""
    spec = AWS_SOURCES["gridmet"]
    raster = _source_raster_path("gridmet", raw_dir, env_var="GRIDMET_AWC_RASTER")
    if raster is None:
        raise SourceUnavailableError(
            "the gridMET PDSI soil layer is a processing input, not a published gridMET variable: "
            "no public endpoint for the STATSGO 0-1500 mm water-holding-capacity raster was found, and "
            "GRIDMET/DROUGHT exposes only drought indices. Supply the raster (GeoTIFF) with "
            "raw_dir=<dir> named 'gridmet_awc.tif', or set GRIDMET_AWC_RASTER=<path>."
        )
    storage = _open_raster_window(str(raster), None)
    return harmonize_aws(
        storage.assign_coords(layer=[_SINGLE_LAYER]).expand_dims(layer=[_SINGLE_LAYER]),
        climate,
        source=spec.name,
        layers_mm=spec.layers_mm,
        native_depth_mm=spec.native_depth_mm,
        depth_mm=depth_mm,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        unverified=spec.unverified,
    )


def _usgs_aws(
    climate: xr.DataArray,
    depth_mm: float | str,
    raw_dir: Path | None,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> HarmonizedAws:
    """Read the USGS/gNATSGO 0-100 cm AWC raster (``awc_gNATSGO_US.tif``, mm/m)."""
    spec = AWS_SOURCES["usgs"]
    raster = _source_raster_path("usgs", raw_dir, env_var="USGS_AWC_RASTER")
    if raster is None:
        raise SourceUnavailableError(
            "the USGS AWC raster (awc_gNATSGO_US.tif, ~1 GB) is not bundled; download it from "
            "https://doi.org/10.5066/P9TI3IS8, then supply it with raw_dir=<dir> named "
            "'usgs_awc.tif', or set USGS_AWC_RASTER=<path>."
        )
    values = _open_raster_window(str(raster), None)
    # published units are mm of available water per metre of soil over a 100 cm
    # column, so one metre makes the value numerically the column total in mm
    storage = values.assign_coords(layer=[_SINGLE_LAYER]).expand_dims(layer=[_SINGLE_LAYER])
    storage = storage.astype(float).assign_attrs(units="mm", long_name="Total plant-available water")
    return harmonize_aws(
        storage,
        climate,
        source=spec.name,
        layers_mm=spec.layers_mm,
        native_depth_mm=spec.native_depth_mm,
        depth_mm=depth_mm,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        unverified=spec.unverified,
    )


def _polaris_aws(
    climate: xr.DataArray,
    depth_mm: float | str,
    raw_dir: Path | None,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> HarmonizedAws:
    """Derive AWC from POLARIS van Genuchten parameters, layer by layer, in strips.

    The 30 m database is never materialized: each latitude strip of the target
    grid is read from the publisher's tiles as a GDAL window, converted to
    available water one layer at a time, aggregated onto that strip's cells, and
    then discarded. Peak memory is therefore a function of one strip and one
    layer, not of the region or the source raster.
    """
    spec = AWS_SOURCES["polaris"]
    if os.environ.get(POLARIS_EE_ASSET_ENV):
        raise EarthEngineUnavailableError(
            f"{POLARIS_EE_ASSET_ENV} is set, but the Earth Engine reduceResolution path is not "
            "implemented in this module; unset it to read windowed POLARIS tiles, or install and "
            "authenticate earthengine-api and export the asset to GeoTIFF first."
        )
    resolved_depth_mm = _resolve_depth(depth_mm, spec.native_depth_mm)
    # layers entirely below the requested column are never fetched
    needed_layers = [layer for layer in spec.layers_mm if layer[1] < resolved_depth_mm]
    latitudes = np.asarray(climate[lat_dim].values, dtype=float)
    longitudes = np.asarray(climate[lon_dim].values, dtype=float)

    strips: list[xr.DataArray] = []
    for strip_latitudes, bounds in _latitude_strips(latitudes, longitudes, rows=POLARIS_STRIP_ROWS):
        strip_total = None
        for name, top_mm, bottom_mm in needed_layers:
            thickness_mm = bottom_mm - top_mm
            included_fraction = _included_fraction(name, spec.layers_mm, resolved_depth_mm)
            if included_fraction <= 0.0:
                continue
            parameters = {
                parameter: _polaris_parameter(parameter, name, raw_dir, bounds, lat_dim=lat_dim, lon_dim=lon_dim)
                for parameter in ("alpha", "n", "theta_r", "theta_s")
            }
            fraction = xr.apply_ufunc(
                van_genuchten_available_water,
                parameters["alpha"],
                parameters["n"],
                parameters["theta_r"],
                parameters["theta_s"],
                dask="allowed",
                output_dtypes=[float],
            )
            # POLARIS tiles are float32 natively, so keep the fractions (and the
            # storage derived from them) in float32: a soil column is a few hundred
            # millimetres, and float64 only multiplies the resident size
            storage_mm = fraction.astype("float32") * np.float32(thickness_mm * included_fraction)
            del parameters
            strip_total = storage_mm if strip_total is None else strip_total + storage_mm
            del storage_mm
        if strip_total is None:  # pragma: no cover - depth validation prevents this
            raise AwsIngestError(f"no POLARIS layer contributes to a {resolved_depth_mm:.1f} mm column")
        # Full-grid centres preserve cell edges, even for a singleton final strip.
        averaged = area_weighted_mean(strip_total, latitudes, longitudes, lat_dim=lat_dim, lon_dim=lon_dim)
        strips.append(averaged.sel({lat_dim: strip_latitudes}))

    total_mm = xr.concat(strips, dim=lat_dim).sel({lat_dim: latitudes})
    total_mm = total_mm.assign_attrs(
        units="mm",
        long_name="Total plant-available water",
        aws_layers_used=json.dumps([name for name, _, _ in needed_layers]),
    )
    return finalize_aws(
        total_mm,
        climate,
        source=spec.name,
        layers_mm=spec.layers_mm,
        native_depth_mm=spec.native_depth_mm,
        depth_mm=resolved_depth_mm,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        unverified=spec.unverified,
    )


def _latitude_strips(
    latitudes: npt.NDArray[np.float64],
    longitudes: npt.NDArray[np.float64],
    *,
    rows: int,
) -> list[tuple[npt.NDArray[np.float64], Bounds]]:
    """Split the target latitudes into strips of whole rows, with read bounds.

    Each strip carries the target latitudes it contains plus the geographic
    ``(west, south, east, north)`` box to read for them, so a strip's bounds are
    derived from the same cell edges the area-weighted average uses and no target
    cell is split.
    """
    if latitudes.size < 2:
        raise AwsIngestError(
            "at least two latitude cell centres are needed to derive cell edges and therefore "
            f"cell areas; got a single centre at {latitudes[0]!r}"
        )
    ordered = latitudes if latitudes[0] <= latitudes[-1] else latitudes[::-1]
    edges = _cell_edges(ordered)
    lon_edges = _cell_edges(longitudes if longitudes[0] <= longitudes[-1] else longitudes[::-1])
    strips: list[tuple[npt.NDArray[np.float64], Bounds]] = []
    for start in range(0, ordered.size, rows):
        stop = min(start + rows, ordered.size)
        bounds: Bounds = (
            float(lon_edges[0]),
            float(edges[start]),
            float(lon_edges[-1]),
            float(edges[stop]),
        )
        strip_latitudes = ordered[start:stop] if latitudes[0] <= latitudes[-1] else ordered[start:stop][::-1]
        strips.append((strip_latitudes, bounds))
    return strips


#: Environment variables naming a local raster for the single-layer sources.
_RAW_RASTER_NAMES = {"gridmet": "gridmet_awc.tif", "usgs": "usgs_awc.tif"}


def _source_raster_path(source: str, raw_dir: Path | None, *, env_var: str) -> Path | None:
    """Resolve a single-layer source's raster from the environment or ``raw_dir``."""
    configured = os.environ.get(env_var)
    if configured:
        path = Path(configured)
        if not path.exists():
            raise SourceUnavailableError(f"{env_var} points at {path}, which does not exist")
        return path
    if raw_dir is not None:
        candidate = Path(raw_dir) / _RAW_RASTER_NAMES[source]
        if candidate.exists():
            return candidate
    return None


def _polaris_parameter(
    parameter: str,
    layer: str,
    raw_dir: Path | None,
    bounds: Bounds,
    *,
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    """Read one POLARIS parameter for one layer over ``bounds``, mosaicking tiles.

    Tiles are read as GDAL windows and only for the requested bounds, so a CONUS
    30 m mosaic is never materialized.
    """
    tiles = _polaris_tile_paths(parameter, layer, bounds, raw_dir)
    parts = [_open_tile_with_retry(tile, bounds) for tile in tiles]
    if not parts:
        raise SourceUnavailableError(
            f"no POLARIS {parameter}/{layer} tiles resolved for bounds {bounds!r}; expected local tiles under "
            f"raw_dir named {{parameter}}_{{layer}}.tif or reachable publisher tiles"
        )
    if len(parts) == 1:
        field = parts[0]
    else:
        combined = xr.combine_by_coords(parts, combine_attrs="drop_conflicts")
        if not isinstance(combined, xr.Dataset) or "values" not in combined:
            raise AwsIngestError(f"POLARIS {parameter}/{layer} tiles could not be combined into one field")
        field = combined["values"]
    return cast(xr.DataArray, field.rename(parameter))


def _polaris_tile_paths(parameter: str, layer: str, bounds: Bounds, raw_dir: Path | None) -> list[Path | str]:
    """POLARIS tile locations covering ``bounds``, local when a ``raw_dir`` is given.

    The publisher splits the 1 arc-second database into 1x1 degree GeoTIFFs named
    ``lat{south}{north}_lon{west}{east}`` with the whole-degree tile bounds, per
    the verified README and the publisher's own client.

    A ``raw_dir`` is authoritative: it selects local tiles only, never the
    network, so an offline or partial tile set fails loudly instead of quietly
    downloading hundreds of megabytes per layer. Without one, the publisher's
    tiles are read over HTTP.
    """
    west, south, east, north = bounds
    tiles: list[Path | str] = []
    for tile_south in range(int(np.floor(south)), int(np.ceil(north))):
        for tile_west in range(int(np.floor(west)), int(np.ceil(east))):
            name = f"lat{tile_south}{tile_south + 1}_lon{tile_west}{tile_west + 1}.tif"
            if raw_dir is not None:
                local = Path(raw_dir) / parameter / layer / name
                if not local.exists():
                    raise SourceUnavailableError(
                        f"missing local POLARIS tile {local}; a supplied raw_dir is used offline, so "
                        "every tile covering the region must be present for each layer and parameter"
                    )
                tiles.append(local)
            else:
                tiles.append(
                    f"http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/{parameter}/"
                    f"{POLARIS_STATISTIC}/{layer}/{name}"
                )
    return tiles


register_source(
    AwsSourceSpec(
        name="gridmet",
        native_depth_mm=1500.0,
        layers_mm=(("0_1500", 0.0, 1500.0),),
        load=_gridmet_aws,
        note="STATSGO soil water-holding-capacity layer (top 1500 mm) used by the gridMET PDSI product",
        unverified=(
            "no public download endpoint was found for the exact STATSGO 0-1500 mm storage raster "
            "used by the gridMET PDSI processing, so this loader takes a user-supplied raster",
        ),
    )
)

register_source(
    AwsSourceSpec(
        name="usgs",
        native_depth_mm=1000.0,
        layers_mm=(("0_1000", 0.0, 1000.0),),
        load=_usgs_aws,
        note="USGS/gNATSGO depth-weighted AWC for the upper 100 cm (mm per metre)",
        unverified=(
            "the published GeoTIFF's band index, nodata value and dtype could not be read here "
            "(the ScienceBase record is behind a browser challenge), so the raster is read as band 1 "
            "and its declared scale is taken at face value",
        ),
    )
)

register_source(
    AwsSourceSpec(
        name="polaris",
        native_depth_mm=POLARIS_LAYERS_MM[-1][2],
        layers_mm=POLARIS_LAYERS_MM,
        load=_polaris_aws,
        note="POLARIS v1.0 van Genuchten parameters, p50, six layers to 2000 mm, AWC derived per layer",
        unverified=(
            "the derivation uses median marginal parameters (p50) rather than the median of available "
            "water; tile metadata and coverage have not been verified across the full database",
            "the server-side Earth Engine reduceResolution path is not implemented",
        ),
    )
)


def load_aws(
    source: str,
    climate: xr.DataArray,
    *,
    depth_mm: float | str = NATIVE_DEPTH,
    raw_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> HarmonizedAws:
    """Load one source as total available water capacity on the climate grid.

    With ``cache_dir`` set, a harmonized field is read back from GeoTIFFs when the
    (source, depth, climate grid) combination is already cached, so a remote read
    happens once per configuration.

    Args:
        source: Registered source name, one of ``AWS_SOURCES``.
        climate: Climate field on the target grid (for example monthly
            precipitation), whose footprint defines land and whose coordinates
            define the target cells.
        depth_mm: Requested column depth in millimetres, or ``"native"`` for the
            source's own column.
        raw_dir: Directory holding raw source rasters when they are not remote.
        cache_dir: Directory for the harmonized GeoTIFF cache; None disables caching.
        lat_dim: Latitude dimension name.
        lon_dim: Longitude dimension name.

    Returns:
        The harmonized total field and its fill mask.

    Raises:
        AwsIngestError: If ``source`` is not registered.
        SourceUnavailableError: If the source's data or the optional geospatial
            dependencies are unavailable.
        DepthUnavailableError: If the requested depth exceeds the source column.
    """
    spec = AWS_SOURCES.get(source)
    if spec is None:
        known = ", ".join(sorted(AWS_SOURCES))
        raise AwsIngestError(f"unknown aws_source {source!r}; registered sources are {known}")

    resolved_depth_mm = _resolve_depth(depth_mm, spec.native_depth_mm)
    cache_path = None if cache_dir is None else Path(cache_dir)
    signature = grid_signature(climate[lat_dim].values, climate[lon_dim].values, lat_dim=lat_dim, lon_dim=lon_dim)
    key = f"{source}_{resolved_depth_mm:.1f}mm_{signature}"
    if cache_path is not None:
        cached = _read_cache(cache_path, key, lat_dim=lat_dim, lon_dim=lon_dim)
        if cached is not None:
            _logger.info("aws_cache_hit", source=source, depth_mm=resolved_depth_mm, key=key)
            return cached

    _logger.info("aws_ingest_started", source=source, depth_mm=resolved_depth_mm)
    result = spec.load(
        climate,
        depth_mm,
        None if raw_dir is None else Path(raw_dir),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    if cache_path is not None:
        _write_cache(cache_path, key, result, lat_dim=lat_dim, lon_dim=lon_dim)
    _logger.info(
        "aws_ingest_completed",
        source=source,
        depth_mm=resolved_depth_mm,
        filled_cells=int(result.aws.attrs["aws_filled_cells"]),
    )
    return result

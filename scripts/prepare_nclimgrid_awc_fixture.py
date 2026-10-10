#!/usr/bin/env python3
"""Build the reduced nClimGrid-aligned available-water-capacity fixture from POLARIS.

Run from the repository root:

    uv run --extra aws scripts/prepare_nclimgrid_awc_fixture.py

What this produces
    ``tests/fixture/nclimgrid_awc/polaris_awc_1000mm.nc``: total plant-available
    water, in millimetres over a 1000 mm soil column, on a 3 x 3 subset of the
    climate grid used by the pinned nClimGrid monthly example inputs. The subset
    coordinates are literal cell centres of that grid, so the file can be passed
    to the CLI as ``--netcdf_awc`` alongside a precipitation/PET file subset to
    the same nine cells (the CLI compares the coordinates with a tolerance of one
    tenth of the grid spacing).

Source grid
    The pinned reduced nClimGrid sample is fetched from the
    ``monocongo/example_climate_indices`` repository at ``SOURCE_COMMIT`` and its
    SHA-256 is checked, exactly as ``scripts/prepare_e2e_inputs.py`` does. Only
    the coordinate axes and the land footprint of that sample are used: the soil
    values come from POLARIS, never from the climate file.

Soil source
    POLARIS v1.0 van Genuchten parameters (p50, six layers to 2000 mm) read from
    the publisher's 1-degree tiles and converted to available water by
    :mod:`climate_indices.aws_ingest`. The derivation is therefore available
    water for median *marginal* parameters, not the median of available water.
    POLARIS is licensed CC BY-NC 4.0 (non-commercial), so this fixture must not
    be redistributed for commercial use, and it is a reduced, modified work
    rather than an unaltered POLARIS or NCEI product.

Why tiles are downloaded first
    The publisher's plain HTTP server truncates long windowed reads under load,
    which GDAL reports as a read failure partway through a multi-layer ingest.
    The 180 tiles covering this subset are therefore downloaded once, resumably
    and with a tail read verified, into ``--cache-dir/tiles``; the ingest that
    follows reads them locally through :mod:`climate_indices.aws_ingest`'s
    ``raw_dir`` path. A rerun reuses both the tiles and, with ``--cache-dir``,
    the harmonized GeoTIFF cache, so it costs nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import xarray as xr

from climate_indices import aws_ingest

#: Repository root, and the fixture directory this script writes.
_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = _ROOT / "tests" / "fixture" / "nclimgrid_awc"

#: Pinned reduced nClimGrid sample, matching scripts/prepare_e2e_inputs.py.
SOURCE_COMMIT = "ae57c488af832c1ebfdf864c8ed7d16636e2e36f"
SOURCE_URL = f"https://raw.githubusercontent.com/monocongo/example_climate_indices/{SOURCE_COMMIT}/example/input"
CLIMATE_FILE = "nclimgrid_lowres_prcp.nc"
CLIMATE_SHA256 = "31689a564aa56993f5280a84050e32ad310ebc3b3b44c062ae846aba24ca7276"

#: The subset, as index slices into the 38 x 87 example grid. Latitude centres
#: 37.89583206176758, 38.5625, 39.22916793823242 and longitude centres -100.6875,
#: -100.02083587646484, -99.35416412353516 (the Colorado/Kansas border country,
#: the same region the soil benchmark defaults to). Widen the slices to extend the
#: fixture; ingest cost grows with the subset's bounding box, not its cell count.
SUBSET_LAT = slice(20, 23)
SUBSET_LON = slice(36, 39)

#: Soil column depth the fixture covers, in millimetres.
AWC_DEPTH_MM = 1000.0

#: Name of the data variable written, matching the module's own naming.
AWC_VARIABLE = "awc"

#: Fixture layout version recorded by the provenance file.
FIXTURE_VERSION = "1.0.0"

#: Soil parameters the van Genuchten derivation needs, and the layer statistics.
PARAMETERS = ("alpha", "n", "theta_r", "theta_s")

#: Parallel tile downloads, and attempts per tile.
DOWNLOAD_WORKERS = 6
DOWNLOAD_ATTEMPTS = 6


def _sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _climate_sample(cache_dir: Path) -> Path:
    """Download the pinned climate sample into ``cache_dir`` unless already verified."""
    path = cache_dir / CLIMATE_FILE
    if path.exists() and _sha256(path) == CLIMATE_SHA256:
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{SOURCE_URL}/{CLIMATE_FILE}"
    temporary = path.with_name(f".{path.name}.download")
    with urlopen(url, timeout=120) as response, temporary.open("wb") as target:  # noqa: S310 - pinned https source
        shutil.copyfileobj(response, target)
    digest = _sha256(temporary)
    if digest != CLIMATE_SHA256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch for {url}: expected {CLIMATE_SHA256}, got {digest}")
    temporary.replace(path)
    return path


def _subset_climate(cache_dir: Path) -> xr.DataArray:
    """The pinned sample's subset as a (time, lat, lon) field defining the target grid."""
    with xr.open_dataset(_climate_sample(cache_dir), engine="h5netcdf") as dataset:
        climate = dataset["prcp"].isel(lat=SUBSET_LAT, lon=SUBSET_LON).load()
    return cast(xr.DataArray, climate.transpose("time", "lat", "lon"))


def _layers_within_depth() -> list[tuple[str, float, float]]:
    """POLARIS layers contributing to the fixture's soil column."""
    return [layer for layer in aws_ingest.POLARIS_LAYERS_MM if layer[1] < AWC_DEPTH_MM]


def _tile_targets(climate: xr.DataArray, tiles_dir: Path) -> list[tuple[str, Path]]:
    """Publisher URLs and local destinations for every tile the ingest will read."""
    latitudes = np.asarray(climate["lat"].values, dtype=float)
    longitudes = np.asarray(climate["lon"].values, dtype=float)
    targets: dict[Path, str] = {}
    for _strip_latitudes, bounds in aws_ingest._latitude_strips(
        latitudes, longitudes, rows=aws_ingest.POLARIS_STRIP_ROWS
    ):
        for name, _top, _bottom in _layers_within_depth():
            for parameter in PARAMETERS:
                for tile in aws_ingest._polaris_tile_paths(parameter, name, bounds, None):
                    url = str(tile)
                    targets[tiles_dir / parameter / name / Path(url).name] = url
    return sorted((url, path) for path, url in targets.items())


def _remote_size(url: str) -> int | None:
    """Declared length of a remote tile, or None when the server will not say."""
    try:
        with urlopen(Request(url, method="HEAD"), timeout=60) as response:  # noqa: S310 - publisher tiles
            length = response.headers.get("Content-Length")
        return int(length) if length is not None else None
    except (URLError, OSError, ValueError):
        return None


def _tail_readable(path: Path) -> bool:
    """Whether GDAL can decode the last row, which a truncated download cannot."""
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit("reading POLARIS tiles needs the aws extra: pip install 'climate-indices[aws]'") from error
    try:
        with rasterio.open(path) as dataset:
            dataset.read(1, window=Window(0, dataset.height - 1, dataset.width, 1))
        return True
    except Exception:  # noqa: BLE001 - any GDAL failure means "not usable"
        return False


def _fetch_tile(url: str, path: Path) -> str:
    """Download one tile resumably, verifying its declared length and last row."""
    expected = _remote_size(url)
    if path.exists() and (expected is None or path.stat().st_size == expected) and _tail_readable(path):
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if expected is not None and have >= expected:
            if have == expected and _tail_readable(partial):
                partial.replace(path)
                return "downloaded"
            partial.unlink()
            have = 0
        try:
            request = Request(url)
            if have:
                request.add_header("Range", f"bytes={have}-")
            with urlopen(request, timeout=120) as response, partial.open("ab") as target:  # noqa: S310 - publisher tiles
                shutil.copyfileobj(response, target)
        except (URLError, OSError) as error:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise SystemExit(f"could not download {url} after {attempt} attempts: {error}") from error
            time.sleep(2.0 * attempt)
            continue
        size = partial.stat().st_size if partial.exists() else 0
        if (expected is None or size == expected) and _tail_readable(partial):
            partial.replace(path)
            return "downloaded"
        time.sleep(2.0 * attempt)
    raise SystemExit(f"could not download a complete {url}")


def _ensure_tiles(targets: list[tuple[str, Path]]) -> None:
    """Download every missing tile in parallel, reporting a summary."""
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        outcomes = list(pool.map(lambda target: _fetch_tile(*target), targets))
    counts = {outcome: outcomes.count(outcome) for outcome in set(outcomes)}
    print(f"tiles: {len(targets)} needed, {counts}")


def _describe(climate: xr.DataArray, targets: list[tuple[str, Path]]) -> str:
    """Human-readable summary of the target grid and the tiles one pass would read."""
    latitudes = np.asarray(climate["lat"].values, dtype=float)
    longitudes = np.asarray(climate["lon"].values, dtype=float)
    return "\n".join(
        [
            f"grid: {latitudes.size} x {longitudes.size} cells",
            f"  lat {latitudes[0]:.6f} .. {latitudes[-1]:.6f}",
            f"  lon {longitudes[0]:.6f} .. {longitudes[-1]:.6f}",
            f"  {len(targets)} tiles to read, {len(_layers_within_depth())} layers at {AWC_DEPTH_MM:.0f} mm",
        ]
    )


def _build_fixture(climate: xr.DataArray, tiles_dir: Path, cache_dir: Path | None) -> Path:
    """Ingest POLARIS from local tiles onto the subset grid and write the fixture NetCDF."""
    harmonized = aws_ingest.load_aws("polaris", climate, depth_mm=AWC_DEPTH_MM, raw_dir=tiles_dir, cache_dir=cache_dir)
    aws = harmonized.aws
    if bool(np.asarray(harmonized.filled.values, dtype=bool).any()):
        raise SystemExit(
            "the harmonized field needed hole filling; reduce the subset or fix the soil source "
            "before committing a fixture with interpolated cells"
        )
    dataset = xr.Dataset(
        {AWC_VARIABLE: aws.astype("float64")},
        attrs={
            "title": "Total plant-available water capacity (AWC) from POLARIS v1.0, reduced nClimGrid subset",
            "source": f"POLARIS v1.0 p50 van Genuchten parameters, available water derived for 0-{AWC_DEPTH_MM:.0f} mm",
            "nclimgrid_source_commit": SOURCE_COMMIT,
            "nclimgrid_grid": f"cell centres lat[{SUBSET_LAT.start}:{SUBSET_LAT.stop}], "
            f"lon[{SUBSET_LON.start}:{SUBSET_LON.stop}] of the 38x87 example grid",
            "license": "POLARIS v1.0 is CC BY-NC 4.0; reduced and modified, not an unaltered POLARIS product",
        },
    )
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"polaris_awc_{AWC_DEPTH_MM:.0f}mm.nc"
    temporary = path.with_name(f".{path.name}.writing")
    dataset.to_netcdf(temporary, engine="h5netcdf")
    temporary.replace(path)
    return path


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=None, help="directory for the climate sample, tiles, and GeoTIFF cache")
    parser.add_argument("--dry-run", action="store_true", help="print the target grid and exit without reading tiles")
    arguments = parser.parse_args()

    cache_dir = Path(arguments.cache_dir) if arguments.cache_dir else Path(tempfile.gettempdir()) / "awc-fixture-cache"
    climate = _subset_climate(cache_dir / "climate")
    tiles_dir = cache_dir / "tiles"
    targets = _tile_targets(climate, tiles_dir)
    print(_describe(climate, targets))
    if arguments.dry_run:
        return 0

    _ensure_tiles(targets)
    path = _build_fixture(climate, tiles_dir, cache_dir / "harmonized")
    digest = _sha256(path)
    print(f"wrote {path.relative_to(_ROOT)}")
    print(f"checksum_sha256: {digest}")
    print("update tests/fixture/nclimgrid_awc/provenance.json with this digest")
    return 0


if __name__ == "__main__":
    sys.exit(main())

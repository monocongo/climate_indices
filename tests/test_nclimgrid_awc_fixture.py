"""Tests for the reduced nClimGrid-aligned available-water-capacity fixture.

The fixture is ``tests/fixture/nclimgrid_awc/polaris_awc_1000mm.nc``: total
plant-available water in millimetres over a 1000 mm soil column, derived from
POLARIS v1.0 p50 van Genuchten parameters by
``scripts/prepare_nclimgrid_awc_fixture.py``, on nine cells of the pinned
nClimGrid example grid. These tests pin the fixture's grid alignment, value
plausibility, provenance integrity, and usability through the library's Palmer
xarray entry point. They are a development fixture check, not scientific
validation: POLARIS-derived AWC is not an external drought-index oracle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

import climate_indices
from climate_indices import aws_ingest

FIXTURE_DIR = Path(__file__).parent / "fixture" / "nclimgrid_awc"
FIXTURE_PATH = FIXTURE_DIR / "polaris_awc_1000mm.nc"
PROVENANCE_PATH = FIXTURE_DIR / "provenance.json"

#: The exact cell centres the fixture covers, taken from the pinned 38 x 87
#: nClimGrid example grid (latitude indices 20:23, longitude indices 36:39). The
#: sample stores them as float32, so comparisons use the spacing-based tolerance
#: the CLI itself applies rather than exact equality.
EXPECTED_LATITUDES = (37.89583206176758, 38.5625, 39.22916793823242)
EXPECTED_LONGITUDES = (-100.6875, -100.02083587646484, -99.35416412353516)

#: One tenth of the example grid's 2/3 degree spacing, matching the CLI's own
#: coordinate tolerance for matching an AWC file against a climate file.
COORDINATE_ATOL = 0.0666667

#: Soil column depth the fixture covers, in millimetres.
EXPECTED_DEPTH_MM = 1000.0


def _open_fixture() -> xr.Dataset:
    """Open the committed fixture NetCDF."""
    return xr.open_dataset(FIXTURE_PATH, engine="h5netcdf")


def _sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_exposes_awc_on_the_nclimgrid_subset():
    """The variable, dimensions, units, and coordinates match the documented subset."""
    with _open_fixture() as dataset:
        assert "awc" in dataset.variables
        field = dataset["awc"]
        assert field.dims == ("lat", "lon")
        assert field.attrs["units"] == "mm"
        np.testing.assert_allclose(field["lat"].values, EXPECTED_LATITUDES, rtol=0, atol=COORDINATE_ATOL)
        np.testing.assert_allclose(field["lon"].values, EXPECTED_LONGITUDES, rtol=0, atol=COORDINATE_ATOL)
        assert field.shape == (len(EXPECTED_LATITUDES), len(EXPECTED_LONGITUDES))
        assert dataset.attrs["nclimgrid_source_commit"] == "ae57c488af832c1ebfdf864c8ed7d16636e2e36f"


def test_fixture_values_are_plausible_available_water():
    """Values are finite, inside the ingest module's bounds, and not degenerate."""
    with _open_fixture() as dataset:
        values = dataset["awc"].values

    assert np.isfinite(values).all(), "a fixture cell is missing"
    assert float(values.min()) >= aws_ingest.MIN_AWS_MM
    assert float(values.max()) <= aws_ingest.MAX_AWS_MM
    # a column total cannot exceed the column depth it was integrated over
    assert float(values.max()) < EXPECTED_DEPTH_MM
    assert float(values.std()) > 0.0, "a constant field would not exercise per-cell AWC"


def test_fixture_is_accepted_by_the_palmer_xarray_path():
    """The fixture drives a real water balance: AWC in inches over a monthly record."""
    with _open_fixture() as dataset:
        awc_mm = dataset["awc"].load()
    awc_inches = aws_ingest.aws_mm_to_inches(awc_mm)

    months = np.arange(360)
    # months from January, alternating wet winter and dry summer, deterministic
    seasonal = 1.5 + 1.5 * np.cos(2.0 * np.pi * months / 12.0)
    shape = (months.size, *awc_mm.shape)
    precipitation = xr.DataArray(
        np.broadcast_to(seasonal[:, None, None] + 0.5, shape),
        coords={
            "time": xr.date_range("1981-01-01", periods=months.size, freq="MS"),
            "lat": awc_mm["lat"],
            "lon": awc_mm["lon"],
        },
        dims=["time", "lat", "lon"],
        attrs={"units": "inches"},
    )
    pet = xr.DataArray(
        np.broadcast_to((2.0 - 0.5 * np.cos(2.0 * np.pi * months / 12.0))[:, None, None], shape),
        coords=precipitation.coords,
        dims=precipitation.dims,
        attrs={"units": "inches"},
    )

    result = climate_indices.pdsi(precipitation, pet, awc_inches, 1981, 1981, 2010)

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"pdsi", "phdi", "pmdi", "z_index"}
    assert result["pdsi"].dims == ("time", "lat", "lon")
    assert np.isfinite(result["pdsi"].values).all(), "the fixture produced no usable Palmer water balance"


def test_fixture_provenance_matches_the_committed_bytes():
    """The recorded digest, license, and grid description match the fixture on disk."""
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert _sha256(FIXTURE_PATH) == provenance["checksum_sha256"]
    assert provenance["fixture_version"] == "1.0.0"
    assert "1000" in provenance["subset_description"]
    assert "nClimGrid" in provenance["subset_description"]
    # a non-commercial source license must be recorded, not omitted
    assert "BY-NC" in provenance["license"].replace(" ", "")
    assert provenance["url"].startswith("http")

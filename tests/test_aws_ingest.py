"""Unit and depth conversion tests for :mod:`climate_indices.aws_ingest`.

The arithmetic here is checked against hand-computed values so the conversions
and the van Genuchten derivation are pinned independently of the loaders; the
raster-reading path is exercised on local tiles when the optional ``aws`` extra
is installed; no tests read remote data.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

from climate_indices import aws_ingest
from climate_indices.aws_ingest import (
    AWS_SOURCES,
    MAX_AWS_MM,
    MIN_AWS_MM,
    NATIVE_DEPTH,
    POLARIS_LAYERS_MM,
    AwsIngestError,
    AwsSourceSpec,
    DepthUnavailableError,
    HarmonizedAws,
    SourceUnavailableError,
)

# a single layer whose fraction * thickness is easy to check by hand
_ONE_METRE_LAYER = (("0_1000", 0.0, 1000.0),)


def _climate(mask: np.ndarray, *, latitudes: list[float], longitudes: list[float]) -> xr.DataArray:
    """A one-timestep climate field whose data footprint is ``mask``."""
    values = np.where(mask, 1.0, np.nan)
    return xr.DataArray(
        values[np.newaxis, :, :],
        coords={"time": [np.datetime64("2000-01-01")], "lat": latitudes, "lon": longitudes},
        dims=["time", "lat", "lon"],
        attrs={"units": "mm"},
    )


def _layered(layer_values: npt.ArrayLike, *, latitudes: list[float], longitudes: list[float]) -> xr.DataArray:
    """A per-layer field with POLARIS's layer coordinate, shaped (layer, lat, lon)."""
    values = np.asarray(layer_values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(values.size, 1, 1)
    return xr.DataArray(
        values,
        coords={"layer": [name for name, _, _ in POLARIS_LAYERS_MM], "lat": latitudes, "lon": longitudes},
        dims=["layer", "lat", "lon"],
    )


# ---------------------------------------------------------------------------
# unit conversions
# ---------------------------------------------------------------------------


def test_volumetric_fraction_times_thickness_is_millimetres():
    """An AWC of 0.15 cm/cm over 1000 mm of soil is 150 mm of water."""
    fractions = xr.DataArray(
        [[[[0.15]]]],
        coords={"layer": ["0_1000"], "time": [0], "lat": [40.0], "lon": [-100.0]},
        dims=["layer", "time", "lat", "lon"],
    )

    storage = aws_ingest.layer_storage_mm(fractions, _ONE_METRE_LAYER)

    assert float(storage.sel(time=0, lat=40.0, lon=-100.0).item()) == pytest.approx(150.0)
    assert storage.attrs["units"] == "mm"


def test_layer_storage_rejects_unknown_or_empty_layers():
    fractions = xr.DataArray(
        [[[0.1]]],
        coords={"layer": ["0_500"], "lat": [40.0], "lon": [-100.0]},
        dims=["layer", "lat", "lon"],
    )

    with pytest.raises(AwsIngestError, match="unknown soil layer"):
        aws_ingest.layer_storage_mm(fractions, _ONE_METRE_LAYER)


def test_aws_mm_to_inches_uses_palmer_surface_layer_constant():
    """25.4 mm is one inch, which is also Palmer's fixed surface-layer capacity."""
    assert aws_ingest.SURFACE_LAYER_MM == 25.4
    assert float(aws_ingest.aws_mm_to_inches(np.array([25.4, 254.0]))[1]) == pytest.approx(10.0)


def test_van_genuchten_available_water_matches_hand_computed_value():
    """Hand-computed check: alpha=0.02 kPa^-1, n=2, theta_r=0.05, theta_s=0.45."""
    fraction = aws_ingest.van_genuchten_available_water(
        np.log10(0.02),
        2.0,
        0.05,
        0.45,
    )

    # theta(33) = 0.05 + 0.40 * (1 + (0.02 * 33) ** 2) ** -0.5 = 0.3838437626
    # theta(1500) = 0.05 + 0.40 * (1 + (0.02 * 1500) ** 2) ** -0.5 = 0.0633259321
    assert float(np.asarray(fraction)) == pytest.approx(0.32051783053163463, rel=0, abs=1e-15)


def test_van_genuchten_available_water_matches_second_hand_computed_value():
    """A non-integer n exercises the 1 - 1/n exponent of the closed form."""
    fraction = aws_ingest.van_genuchten_available_water(
        np.log10(0.036),
        1.5,
        0.078,
        0.43,
    )

    assert float(np.asarray(fraction)) == pytest.approx(0.21900319706227084, rel=0, abs=1e-15)
    # over a 200 mm layer that is 43.80 mm of available water
    assert float(np.asarray(fraction)) * 200.0 == pytest.approx(43.80063941245417, rel=0, abs=1e-12)


def test_van_genuchten_rejects_a_shape_parameter_at_or_below_one():
    with pytest.raises(AwsIngestError, match="must be greater than 1"):
        aws_ingest.van_genuchten_available_water(np.log10(0.02), 1.0, 0.05, 0.45)


# ---------------------------------------------------------------------------
# depth integration
# ---------------------------------------------------------------------------


def test_integrate_depth_sums_the_whole_column_at_native_depth():
    storage = _layered(
        np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]]).reshape(6, 1, 1),
        latitudes=[40.0],
        longitudes=[-100.0],
    )

    total = aws_ingest.integrate_depth(storage, POLARIS_LAYERS_MM, POLARIS_LAYERS_MM[-1][2])

    assert float(total.sel(lat=40.0, lon=-100.0)) == pytest.approx(210.0)


def test_integrate_depth_weights_partly_included_layers_proportionally():
    """A depth inside a layer takes that fraction of the layer's stored water."""
    storage = _layered(
        np.array([[10.0], [0.0], [0.0], [0.0], [0.0], [80.0]]).reshape(6, 1, 1),
        latitudes=[40.0],
        longitudes=[-100.0],
    )

    # 1200 mm is 200 mm into the 1000-2000 mm layer, i.e. 20% of it
    total = aws_ingest.integrate_depth(storage, POLARIS_LAYERS_MM, 1200.0)

    assert float(total.sel(lat=40.0, lon=-100.0)) == pytest.approx(10.0 + 0.2 * 80.0)


def test_integrate_depth_beyond_the_native_column_raises():
    storage = _layered(
        np.full((6, 1, 1), 10.0),
        latitudes=[40.0],
        longitudes=[-100.0],
    )

    with pytest.raises(DepthUnavailableError, match="exceeds the source column"):
        aws_ingest.integrate_depth(storage, _ONE_METRE_LAYER, 1500.0)


def test_integrate_depth_rejects_a_non_positive_depth():
    storage = _layered(np.full((6, 1, 1), 10.0), latitudes=[40.0], longitudes=[-100.0])

    with pytest.raises(AwsIngestError, match="positive number of millimetres"):
        aws_ingest.integrate_depth(storage, POLARIS_LAYERS_MM, 0.0)


def test_resolve_depth_accepts_the_native_sentinel_and_rejects_other_strings():
    assert aws_ingest._resolve_depth(NATIVE_DEPTH, 1000.0) == pytest.approx(1000.0)
    assert aws_ingest._resolve_depth(750.0, 1000.0) == pytest.approx(750.0)

    with pytest.raises(AwsIngestError, match="unknown soil depth"):
        aws_ingest._resolve_depth("deep", 1000.0)


# ---------------------------------------------------------------------------
# area-weighted aggregation
# ---------------------------------------------------------------------------


def test_area_weighted_mean_averages_nested_cells_by_area():
    """Four source cells over one target cell average by area, with no bias."""
    # rows repeat so the check isolates the nesting/averaging from cos(latitude)
    source = xr.DataArray(
        np.array([[1.0, 2.0], [1.0, 2.0]]),
        coords={"lat": [39.75, 39.25], "lon": [-99.75, -99.25]},
        dims=["lat", "lon"],
    )

    averaged = aws_ingest.area_weighted_mean(source, [39.5], [-99.5])

    assert float(averaged.sel(lat=39.5, lon=-99.5)) == pytest.approx(1.5)


def test_area_weighted_mean_weights_high_latitude_cells_less():
    """Rows inside one target cell count by cos(latitude), not equally."""
    source = xr.DataArray(
        np.array([[1.0, 1.0], [3.0, 3.0]]),
        coords={"lat": [30.5, 60.5], "lon": [-100.0, -99.0]},
        dims=["lat", "lon"],
    )

    averaged = aws_ingest.area_weighted_mean(source, [45.5], [-99.5])
    weights = np.cos(np.radians([30.5, 60.5]))
    expected = float((1.0 * weights[0] + 3.0 * weights[1]) / weights.sum())

    assert float(averaged.item()) == pytest.approx(expected)
    # the 60 N row's larger value is discounted, pulling the mean below the plain 2.0
    assert float(averaged.item()) < 2.0


def test_area_weighted_mean_excludes_missing_source_cells():
    source = xr.DataArray(
        np.array([[np.nan, 2.0], [2.0, 2.0]]),
        coords={"lat": [39.75, 39.25], "lon": [-99.75, -99.25]},
        dims=["lat", "lon"],
    )

    averaged = aws_ingest.area_weighted_mean(source, [39.5], [-99.5])

    # the missing cell is dropped from the average rather than treated as zero
    assert float(averaged.sel(lat=39.5, lon=-99.5)) == pytest.approx(2.0)


def test_area_weighted_mean_marks_target_cells_with_no_overlap_missing():
    source = xr.DataArray(
        np.array([[1.0, 2.0], [1.0, 2.0]]),
        coords={"lat": [40.0, 39.0], "lon": [-100.0, -99.0]},
        dims=["lat", "lon"],
    )

    averaged = aws_ingest.area_weighted_mean(source, [40.0, 39.0], [-80.0, -79.0])

    assert bool(np.isnan(averaged.values).all())


def test_area_weighted_mean_rejects_duplicate_or_non_finite_coordinates():
    values = np.ones((2, 2))
    duplicated = xr.DataArray(values, coords={"lat": [40.0, 40.0], "lon": [-100.0, -99.0]}, dims=["lat", "lon"])
    non_finite = xr.DataArray(values, coords={"lat": [40.0, np.nan], "lon": [-100.0, -99.0]}, dims=["lat", "lon"])

    with pytest.raises(AwsIngestError, match="duplicate cell centres"):
        aws_ingest.area_weighted_mean(duplicated, [40.0], [-100.0])
    with pytest.raises(AwsIngestError, match="non-finite cell centres"):
        aws_ingest.area_weighted_mean(non_finite, [40.0], [-100.0])


# ---------------------------------------------------------------------------
# missing data
# ---------------------------------------------------------------------------


def test_fill_land_holes_fills_inside_land_and_leaves_water_missing():
    values = np.array([[np.nan, 20.0], [np.nan, np.nan]])
    land = np.array([[True, True], [False, True]])

    filled, flag = aws_ingest.fill_land_holes(values, land)

    assert filled[0, 0] == pytest.approx(20.0)
    assert not flag[0, 1]
    assert flag[0, 0]
    assert flag[1, 1]
    assert not flag[1, 0]  # not land: stays missing
    assert bool(np.isnan(filled[1, 0]))


def test_fill_land_holes_is_a_no_op_without_any_valid_value():
    values = np.array([[np.nan, np.nan]])
    land = np.array([[True, True]])

    filled, flag = aws_ingest.fill_land_holes(values, land)

    assert bool(np.isnan(filled).all())
    assert not flag.any()


def test_fill_land_holes_rejects_misaligned_shapes():
    with pytest.raises(AwsIngestError, match="does not match land mask shape"):
        aws_ingest.fill_land_holes(np.ones((2, 2)), np.ones((3, 3), dtype=bool))


def test_clip_aws_bounds_raises_low_values_to_the_surface_layer_and_reports():
    values = np.array([[10.0, 30.0], [5000.0, np.nan]])

    clipped, report = aws_ingest.clip_aws_bounds(values)

    assert clipped[0, 0] == pytest.approx(MIN_AWS_MM)
    assert clipped[1, 0] == pytest.approx(MAX_AWS_MM)
    assert clipped[0, 1] == pytest.approx(30.0)
    assert report["clipped_low"] == 1
    assert report["clipped_high"] == 1
    assert report["at_surface_capacity"] == 1
    assert report["min_before_mm"] == pytest.approx(10.0)
    assert report["max_before_mm"] == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# the harmonization pipeline
# ---------------------------------------------------------------------------


def _synthetic_source_spec() -> AwsSourceSpec:
    """A two-layer source, shaped like POLARIS but synthetic, for pipeline tests."""

    def load(
        climate: xr.DataArray,
        depth_mm: float | str,
        raw_dir,
        *,
        lat_dim: str = "lat",
        lon_dim: str = "lon",
    ) -> HarmonizedAws:
        storage_values = np.full((6, 2, 2), 20.0)
        storage_values[0, :, :] = 100.0
        storage_values[0, 0, 0] = np.nan  # a no-data hole inside land
        storage = _layered(storage_values, latitudes=[40.75, 40.25], longitudes=[-100.75, -100.25])
        return aws_ingest.harmonize_aws(
            storage,
            climate,
            source="synthetic",
            layers_mm=POLARIS_LAYERS_MM,
            native_depth_mm=POLARIS_LAYERS_MM[-1][2],
            depth_mm=depth_mm,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

    return AwsSourceSpec(
        name="synthetic",
        native_depth_mm=POLARIS_LAYERS_MM[-1][2],
        layers_mm=POLARIS_LAYERS_MM,
        load=load,
        note="synthetic two-layer source used by tests",
    )


def test_load_aws_harmonizes_onto_the_climate_grid_and_masks_water():
    aws_ingest.register_source(_synthetic_source_spec())
    try:
        climate = _climate(
            np.array([[True, True], [False, True]]),
            latitudes=[40.75, 40.25],
            longitudes=[-100.75, -100.25],
        )

        result = aws_ingest.load_aws("synthetic", climate, depth_mm=100.0)

        # 100 mm is the 0-50 mm layer plus half of the 50-150 mm layer
        expected_mm = 100.0 + 0.5 * 20.0
        assert result.aws.dims == ("lat", "lon")
        assert result.aws.shape == (2, 2)
        assert float(result.aws.sel(lat=40.75, lon=-100.75)) == pytest.approx(expected_mm)
        assert bool(np.isnan(result.aws.sel(lat=40.25, lon=-100.75)))  # water stays missing
        assert float(result.aws.sel(lat=40.25, lon=-100.25)) == pytest.approx(expected_mm)
        assert result.filled.sum() == 1  # the hole inside land, and not the water cell
        assert result.aws.attrs["aws_source"] == "synthetic"
        assert result.aws.attrs["aws_depth_mm"] == pytest.approx(100.0)
        assert result.aws.attrs["aws_area_weighted"] == "true"
        assert result.aws.attrs["units"] == "mm"
    finally:
        del AWS_SOURCES["synthetic"]


def test_load_aws_fixed_depth_beyond_the_native_column_raises():
    aws_ingest.register_source(_synthetic_source_spec())
    try:
        climate = _climate(np.array([[True]]), latitudes=[40.75], longitudes=[-100.75])

        with pytest.raises(DepthUnavailableError, match="exceeds the source column"):
            aws_ingest.load_aws("synthetic", climate, depth_mm=2500.0)
    finally:
        del AWS_SOURCES["synthetic"]


def test_load_aws_rejects_an_unregistered_source():
    climate = _climate(np.array([[True]]), latitudes=[40.75], longitudes=[-100.75])

    with pytest.raises(AwsIngestError, match="unknown aws_source"):
        aws_ingest.load_aws("nope", climate)


def test_registered_sources_declare_a_depth_and_layers_that_agree():
    assert set(AWS_SOURCES) == {"gridmet", "usgs", "polaris"}
    for spec in AWS_SOURCES.values():
        assert spec.layers_mm[-1][2] == pytest.approx(spec.native_depth_mm)
        assert spec.layers_mm[0][1] == 0.0
        # every source must be able to satisfy the Palmer model's surface layer
        assert spec.native_depth_mm > MIN_AWS_MM
        for _, top_mm, bottom_mm in spec.layers_mm:
            assert bottom_mm > top_mm


def test_uncertain_source_facts_are_declared_rather_than_hidden():
    """Sources whose access or units could not be verified must say so in the registry."""
    assert AWS_SOURCES["gridmet"].unverified
    assert AWS_SOURCES["usgs"].unverified
    assert AWS_SOURCES["polaris"].unverified


def test_single_layer_sources_require_a_user_supplied_raster(monkeypatch):
    """A missing raster fails with the documented message, not a substitute dataset."""
    for name in ("usgs", "gridmet"):
        monkeypatch.delenv(f"{name.upper()}_AWC_RASTER", raising=False)

    climate = _climate(np.array([[True]]), latitudes=[40.75], longitudes=[-100.75])

    with pytest.raises(SourceUnavailableError, match="not bundled"):
        AWS_SOURCES["usgs"].load(climate, NATIVE_DEPTH, None)
    with pytest.raises(SourceUnavailableError, match="no public endpoint"):
        AWS_SOURCES["gridmet"].load(climate, NATIVE_DEPTH, None)


def test_earth_engine_request_fails_clearly(monkeypatch):
    """Requesting the unimplemented Earth Engine path explains itself."""
    monkeypatch.setenv(aws_ingest.POLARIS_EE_ASSET_ENV, "projects/x/assets/polaris")
    climate = _climate(np.array([[True]]), latitudes=[40.75], longitudes=[-100.75])

    with pytest.raises(SourceUnavailableError, match=aws_ingest.POLARIS_EE_ASSET_ENV):
        AWS_SOURCES["polaris"].load(climate, NATIVE_DEPTH, None)


def test_polaris_tile_names_follow_the_publisher_convention():
    tiles = aws_ingest._polaris_tile_paths("alpha", "30_60", (-100.0, 38.0, -99.0, 39.0), None)

    assert [str(tile) for tile in tiles] == [
        "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/alpha/p50/30_60/lat3839_lon-100-99.tif"
    ]


def test_grid_signature_distinguishes_grids_and_is_stable():
    first = aws_ingest.grid_signature([40.0, 41.0], [-100.0], lat_dim="lat", lon_dim="lon")

    assert first == aws_ingest.grid_signature([40.0, 41.0], [-100.0], lat_dim="lat", lon_dim="lon")
    assert first != aws_ingest.grid_signature([41.0, 40.0], [-100.0], lat_dim="lat", lon_dim="lon")


def test_layer_included_fraction_splits_a_partly_covered_layer():
    """The partial-layer rule is defined once: full above, proportional inside, zero below."""
    assert aws_ingest._included_fraction("0_5", POLARIS_LAYERS_MM, 1000.0) == pytest.approx(1.0)
    assert aws_ingest._included_fraction("60_100", POLARIS_LAYERS_MM, 1000.0) == pytest.approx(1.0)
    # 1500 mm reaches 500 of the 1000 mm in the 1000-2000 mm layer
    assert aws_ingest._included_fraction("100_200", POLARIS_LAYERS_MM, 1500.0) == pytest.approx(0.5)
    assert aws_ingest._included_fraction("100_200", POLARIS_LAYERS_MM, 1000.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("depth_mm", "expected_mm"),
    [
        (1000.0, 320.51783053163463),  # fraction x the 1000 mm of layers above 1000 mm
        (300.0, 96.15534915949039),  # 0-5, 5-15 and 15-30 cm only
        (1500.0, 480.77674579745194),  # half of the 1000-2000 mm layer as well
    ],
)
def test_polaris_loader_end_to_end_on_local_tiles(tmp_path, depth_mm, expected_mm):
    """The real POLARIS path on local tiles: naming, strips, scaling, harmonization.

    The tiles carry one constant van Genuchten parameter set (alpha=0.02 kPa^-1,
    n=2, theta_r=0.05, theta_s=0.45) whose available water fraction is
    hand-computed in the unit tests above, so the expected totals are exact: the
    stored water is the fraction times the layer thickness, summed only over the
    layers the requested depth covers.
    """
    rasterio = pytest.importorskip("rasterio", reason="reading POLARIS tiles needs the optional aws extra")
    pytest.importorskip("rioxarray", reason="reading POLARIS tiles needs the optional aws extra")

    parameters = {"alpha": np.log10(0.02), "n": 2.0, "theta_r": 0.05, "theta_s": 0.45}
    # the climate grid sits inside one whole-degree tile, so one tile per layer and
    # parameter is needed; a supplied raw_dir never falls back to the network
    for name, value in parameters.items():
        for layer in ("0_5", "5_15", "15_30", "30_60", "60_100", "100_200"):
            directory = tmp_path / name / layer
            directory.mkdir(parents=True, exist_ok=True)
            values = np.full((36, 36), value, dtype="float32")
            with rasterio.open(
                directory / "lat3839_lon-100-99.tif",
                "w",
                driver="GTiff",
                height=36,
                width=36,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=rasterio.transform.from_bounds(-100.0, 38.0, -99.0, 39.0, 36, 36),
                nodata=-9999.0,
            ) as dataset:
                dataset.write(values, 1)

    latitudes = [38.25, 38.5, 38.75]
    longitudes = [-99.75, -99.5, -99.25]
    land = np.ones((len(latitudes), len(longitudes)), dtype=bool)
    climate = _climate(land, latitudes=latitudes, longitudes=longitudes)

    result = aws_ingest.load_aws("polaris", climate, depth_mm=depth_mm, raw_dir=tmp_path)

    assert float(np.nanmean(result.aws.values)) == pytest.approx(expected_mm, rel=1e-6)
    assert float(np.nanmin(result.aws.values)) == pytest.approx(expected_mm, rel=1e-6)
    assert result.depth_mm == pytest.approx(depth_mm)
    assert result.filled.sum() == 0


@pytest.mark.parametrize("strip_rows", [1, 4, 10])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("irregular", [False, True])
def test_polaris_strips_match_whole_field_aggregation(monkeypatch, strip_rows, descending, irregular):
    """Strip size and climate-axis order must not change cell areas or placement."""
    source_lat = np.linspace(37.75625, 39.24375, 120)
    source_lon = np.linspace(-100.24375, -98.75625, 120)
    theta_s = xr.DataArray(
        0.25 + 0.1 * (source_lat[:, None] - 38.0) + 0.06 * (source_lon[None, :] + 100.0) ** 2,
        coords={"lat": source_lat, "lon": source_lon},
        dims=["lat", "lon"],
    )
    fields = {
        "alpha": xr.full_like(theta_s, np.log10(0.02)),
        "n": xr.full_like(theta_s, 2.0),
        "theta_r": xr.full_like(theta_s, 0.05),
        "theta_s": theta_s,
    }

    def read_parameter(parameter, layer, raw_dir, bounds, **kwargs):
        west, south, east, north = bounds
        return fields[parameter].sel(lat=slice(south, north), lon=slice(west, east))

    monkeypatch.delenv(aws_ingest.POLARIS_EE_ASSET_ENV, raising=False)
    monkeypatch.setattr(aws_ingest, "_polaris_parameter", read_parameter)
    monkeypatch.setattr(aws_ingest, "POLARIS_STRIP_ROWS", strip_rows)
    latitudes = [38.0, 38.25, 38.5, 38.75, 39.0]
    if irregular:
        latitudes = [38.0, 38.2, 38.5, 38.7, 39.0]
    longitudes = [-100.0, -99.75, -99.5, -99.25, -99.0]
    if descending:
        latitudes.reverse()
        longitudes.reverse()
    land = np.ones((5, 5), dtype=bool)
    land[0, 1] = land[4, 3] = False
    climate = _climate(land, latitudes=latitudes, longitudes=longitudes)
    # Closed form with alpha=0.02 and n=2, over a 1000 mm column.
    storage = (theta_s - 0.05) * ((1 + 0.66**2) ** -0.5 - (1 + 30.0**2) ** -0.5) * 1000.0
    expected = aws_ingest.area_weighted_mean(storage, latitudes, longitudes).where(land)

    aggregate = aws_ingest.area_weighted_mean

    def bounded_aggregate(source, target_lat, target_lon, **kwargs):
        assert len(target_lat) <= strip_rows + 2
        return aggregate(source, target_lat, target_lon, **kwargs)

    monkeypatch.setattr(aws_ingest, "area_weighted_mean", bounded_aggregate)
    result = aws_ingest.load_aws("polaris", climate, depth_mm=1000.0)

    np.testing.assert_allclose(result.aws.values, expected.values, rtol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(result.aws.lat.values, latitudes)
    np.testing.assert_array_equal(result.aws.lon.values, longitudes)
    assert result.filled.sum() == 0


def test_geotiff_cache_round_trip_preserves_field_mask_and_metadata(tmp_path):
    """The harmonized cache must survive a write/read cycle, mask and attrs included."""
    pytest.importorskip("rioxarray", reason="the GeoTIFF cache needs the optional aws extra")
    pytest.importorskip("rasterio", reason="the GeoTIFF cache needs the optional aws extra")
    values = np.array([[120.0, 130.0], [np.nan, 150.0]])
    aws = xr.DataArray(
        values,
        coords={"lat": [38.75, 38.25], "lon": [-99.75, -99.25]},
        dims=["lat", "lon"],
        attrs={
            "aws_source": "polaris",
            "aws_depth_mm": 1000.0,
            "aws_native_depth_mm": 2000.0,
            "aws_filled_cells": 1,
            "aws_layers_mm": "[]",
            "aws_unverified": '["example"]',
            "units": "mm",
        },
    )
    filled = xr.DataArray(
        np.array([[False, True], [False, False]]),
        coords={"lat": [38.75, 38.25], "lon": [-99.75, -99.25]},
        dims=["lat", "lon"],
    )

    aws_ingest._write_cache(
        tmp_path, "polaris_1000.0mm_deadbeef", HarmonizedAws(aws=aws, filled=filled), lat_dim="lat", lon_dim="lon"
    )
    cached = aws_ingest._read_cache(tmp_path, "polaris_1000.0mm_deadbeef", lat_dim="lat", lon_dim="lon")

    assert cached is not None
    np.testing.assert_allclose(cached.aws.values, values, equal_nan=True)
    assert cached.filled.values.dtype == bool
    np.testing.assert_array_equal(cached.filled.values, filled.values)
    assert cached.aws.attrs["aws_source"] == "polaris"
    assert isinstance(cached.aws.attrs["aws_depth_mm"], float)
    assert isinstance(cached.aws.attrs["aws_native_depth_mm"], float)
    assert isinstance(cached.aws.attrs["aws_filled_cells"], int)
    assert cached.aws.attrs["aws_layers_mm"] == "[]"
    assert cached.aws.attrs["aws_unverified"] == '["example"]'
    assert cached.depth_mm == pytest.approx(1000.0)
    assert aws_ingest._read_cache(tmp_path, "missing_key", lat_dim="lat", lon_dim="lon") is None


def test_declared_raster_nodata_is_masked_not_read_as_soil(tmp_path):
    """A nodata sentinel such as POLARIS's -9999 must not survive as a value."""
    rasterio = pytest.importorskip("rasterio", reason="reading a raster needs the optional aws extra")
    pytest.importorskip("rioxarray", reason="reading a raster needs the optional aws extra")
    path = tmp_path / "nodata.tif"
    values = np.array([[-1.5, -9999.0], [-2.0, -1.0]], dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_bounds(-100.0, 38.0, -99.0, 39.0, 2, 2),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values, 1)

    field = aws_ingest._open_raster_window(str(path), None)

    # the sentinel is masked, and the sentinel's cell is the only missing one
    assert int(np.isnan(field.values).sum()) == 1
    assert float(np.nanmin(field.values)) == pytest.approx(-2.0)
    assert float(np.nanmax(field.values)) == pytest.approx(-1.0)


def test_polaris_tile_read_retries_a_transient_transport_failure(monkeypatch, tmp_path):
    """A truncated remote read is retried, and a permanent one fails clearly."""
    for name in ("alpha", "n", "theta_r", "theta_s"):
        directory = tmp_path / name / "30_60"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "lat3839_lon-100-99.tif").write_bytes(b"not a real raster")
    monkeypatch.setattr(aws_ingest, "POLARIS_READ_BACKOFF_S", 0.0)
    bounds = (-100.0, 38.0, -99.0, 39.0)
    calls: list[int] = []

    def flaky(tile, bounds, *, eager=True):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("TIFFFillStrip: Read error")
        return xr.DataArray([[1.0]], coords={"lat": [38.5], "lon": [-99.5]}, dims=["lat", "lon"])

    monkeypatch.setattr(aws_ingest, "_open_raster_window", flaky)
    field = aws_ingest._polaris_parameter("alpha", "30_60", tmp_path, bounds)

    assert len(calls) == 3, "a retryable failure must be retried until it succeeds"
    assert field.name == "alpha"

    def always_failing(tile, bounds, *, eager=True):
        raise OSError("connection reset")

    monkeypatch.setattr(aws_ingest, "_open_raster_window", always_failing)
    with pytest.raises(SourceUnavailableError, match="after 3 attempts"):
        aws_ingest._polaris_parameter("n", "30_60", tmp_path, bounds)


def test_raster_window_keeps_pixels_intersecting_bounds(tmp_path):
    """A window can overlap a pixel without containing its centre."""
    rasterio = pytest.importorskip("rasterio", reason="reading a raster needs the optional aws extra")
    pytest.importorskip("rioxarray", reason="reading a raster needs the optional aws extra")
    path = tmp_path / "partial_pixels.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_bounds(-100.0, 38.9, -99.8, 39.1, 2, 2),
    ) as dataset:
        dataset.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), 1)

    field = aws_ingest._open_raster_window(str(path), (-99.97, 38.97, -99.93, 39.03))

    np.testing.assert_array_equal(field.values, [[3.0], [1.0]])


def test_harmonized_result_exposes_its_source_and_depth():
    aws = xr.DataArray(
        np.ones((1, 1)),
        coords={"lat": [40.0], "lon": [-100.0]},
        dims=["lat", "lon"],
        attrs={"aws_source": "polaris", "aws_depth_mm": 1500.0},
    )
    filled = xr.DataArray(
        np.zeros((1, 1), dtype=bool),
        coords={"lat": [40.0], "lon": [-100.0]},
        dims=["lat", "lon"],
    )

    result = HarmonizedAws(aws=aws, filled=filled)

    assert result.source == "polaris"
    assert result.depth_mm == pytest.approx(1500.0)


@pytest.mark.parametrize("dimension", ["lat", "lon", "target latitude", "target longitude"])
def test_aggregation_rejects_non_monotonic_axes(dimension):
    source = xr.DataArray(np.ones((3, 3)), coords={"lat": [0, 1, 2], "lon": [0, 1, 2]}, dims=["lat", "lon"])
    latitudes = longitudes = [0, 1, 2]
    if dimension in source.dims:
        source = source.assign_coords({dimension: [0, 2, 1]})
    elif dimension == "target latitude":
        latitudes = [0, 2, 1]
    else:
        longitudes = [0, 2, 1]
    with pytest.raises(AwsIngestError, match="strictly monotonic"):
        aws_ingest.area_weighted_mean(source, latitudes, longitudes)


@pytest.mark.parametrize("source", ["gridmet", "usgs"])
def test_single_layer_sources_harmonize_supplied_rasters(tmp_path, monkeypatch, source):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    monkeypatch.delenv(f"{source.upper()}_AWC_RASTER", raising=False)
    with rasterio.open(
        tmp_path / f"{source}_awc.tif",
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_bounds(-100, 38, -99, 39, 2, 2),
    ) as dataset:
        dataset.write(np.full((2, 2), 150, dtype="float32"), 1)
    climate = _climate(np.ones((2, 2), dtype=bool), latitudes=[38.75, 38.25], longitudes=[-99.75, -99.25])
    result = aws_ingest.load_aws(source, climate, raw_dir=tmp_path)
    np.testing.assert_allclose(result.aws.values, 150)
    assert result.depth_mm == AWS_SOURCES[source].native_depth_mm


def test_cache_invalidates_land_mask_depth_and_local_raster(tmp_path, monkeypatch):
    pytest.importorskip("rioxarray")
    raw = tmp_path / "raw"
    raw.mkdir()
    raster = raw / "usgs_awc.tif"
    raster.write_text("150")
    monkeypatch.delenv("USGS_AWC_RASTER", raising=False)
    calls = []

    def load(climate, depth, raw_dir, **kwargs):
        calls.append(depth)
        values = xr.full_like(climate.isel(time=0, drop=True), float(raster.read_text()))
        return aws_ingest.finalize_aws(
            values,
            climate,
            source="usgs",
            layers_mm=AWS_SOURCES["usgs"].layers_mm,
            native_depth_mm=1000,
            depth_mm=1000,
        )

    from dataclasses import replace

    monkeypatch.setitem(AWS_SOURCES, "usgs", replace(AWS_SOURCES["usgs"], load=load))
    climate = _climate(np.ones((2, 2), dtype=bool), latitudes=[38.75, 38.25], longitudes=[-99.75, -99.25])
    kwargs = {"raw_dir": raw, "cache_dir": tmp_path / "cache"}
    aws_ingest.load_aws("usgs", climate, **kwargs)
    aws_ingest.load_aws("usgs", climate, **kwargs)
    assert len(calls) == 1
    masked = climate.copy()
    masked.values[0, 0, 1] = np.nan
    result = aws_ingest.load_aws("usgs", masked, **kwargs)
    assert np.isnan(result.aws.values[0, 1])
    assert not result.filled.values[0, 1]
    assert len(calls) == 2
    raster.write_text("175")
    result = aws_ingest.load_aws("usgs", masked, **kwargs)
    assert result.aws.values[0, 0] == 175
    assert len(calls) == 3
    aws_ingest.load_aws("usgs", masked, depth_mm=1000.0, **kwargs)
    assert len(calls) == 4


def test_cache_publication_is_atomic_and_keys_cannot_escape(tmp_path, monkeypatch):
    pytest.importorskip("rioxarray")
    from rioxarray.raster_array import RasterArray

    climate = _climate(np.ones((2, 2), dtype=bool), latitudes=[38.75, 38.25], longitudes=[-99.75, -99.25])
    old = HarmonizedAws(aws=climate.isel(time=0, drop=True), filled=xr.zeros_like(climate.isel(time=0, drop=True)))
    new = HarmonizedAws(aws=old.aws * 2, filled=xr.ones_like(old.filled))
    key = "../escaped"
    kwargs = {"lat_dim": "lat", "lon_dim": "lon"}
    aws_ingest._write_cache(tmp_path, key, old, **kwargs)
    original = RasterArray.to_raster

    def interrupted_write(self, *args, **options):
        original(self, *args, **options)
        cached = aws_ingest._read_cache(tmp_path, key, **kwargs)
        assert cached is not None
        np.testing.assert_array_equal(cached.aws.values, old.aws.values)
        assert not cached.filled.values.any()
        raise OSError("interrupted before publication")

    monkeypatch.setattr(RasterArray, "to_raster", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        aws_ingest._write_cache(tmp_path, key, new, **kwargs)
    assert len(list(tmp_path.iterdir())) == 1
    assert not (tmp_path.parent / "escaped.tif").exists()
    monkeypatch.setattr(RasterArray, "to_raster", original)
    aws_ingest._write_cache(tmp_path, key, new, **kwargs)
    cached = aws_ingest._read_cache(tmp_path, key, **kwargs)
    assert cached is not None
    np.testing.assert_array_equal(cached.aws.values, new.aws.values)
    assert cached.filled.values.all()


def test_raster_reads_apply_transport_limits_through_eager_compute(monkeypatch):
    rasterio = pytest.importorskip("rasterio")
    rioxarray = pytest.importorskip("rioxarray")
    field = xr.DataArray(
        np.ones((1, 2, 2)), coords={"band": [1], "y": [38.75, 38.25], "x": [-99.75, -99.25]}, dims=["band", "y", "x"]
    ).rio.write_crs("EPSG:4326")

    def check_limits():
        options = rasterio.env.getenv()
        assert options["GDAL_HTTP_CONNECTTIMEOUT"] == 10
        assert options["GDAL_HTTP_TIMEOUT"] == 120
        assert options["GDAL_HTTP_LOW_SPEED_TIME"] == 30

    def open_raster(*args, **kwargs):
        check_limits()
        return field

    compute = xr.DataArray.compute

    def checked_compute(self, **kwargs):
        check_limits()
        assert kwargs["scheduler"] == "synchronous"
        return compute(self, **kwargs)

    monkeypatch.setattr(rioxarray, "open_rasterio", open_raster)
    monkeypatch.setattr(xr.DataArray, "compute", checked_compute)
    aws_ingest._open_raster_window("http://example.invalid/tile.tif", None)

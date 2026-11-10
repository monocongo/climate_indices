"""Offline tests for the Earth Engine and Climate Engine pilot glue.

Nothing here needs a network connection or a credential. The tests cover the
pure-Python parts of ``scripts/earthengine_climate_engine_pilot.py``: calendar
bucketing, pentad sampling, grid cropping, the agreement metrics, the empirical
(SEI) standardization convention the comparison depends on, the Climate Engine
request shapes, and the response parser. The Earth Engine reads and the live API
calls are integration territory and are not exercised here.
"""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import norm

from climate_indices import standardized_index
from climate_indices.drought.indices import Distribution
from scripts import earthengine_climate_engine_pilot as pilot

# The catalog's documented SPI/SPEI/EDDI class boundaries, restated from the
# Earth Engine GRIDMET_DROUGHT page so the test does not read the code it checks.
CATALOG_BOUNDARIES = (-2.0, -1.6, -1.3, -0.8, -0.5, 0.5, 0.8, 1.3, 1.6, 2.0)

# The native grid of both GRIDMET collections, as Earth Engine reports it.
NATIVE_GRID = {
    "crs_transform": (1.0 / 24.0, 0, -124.78749996666667, 0, -1.0 / 24.0, 49.42083333333334),
    "shape_2d": (1386, 585),
}


def test_month_ends_covers_every_month_once() -> None:
    months = pilot.month_ends(1980, 2024)
    assert len(months) == 45 * 12
    assert months[0].last_day.astype("datetime64[D]") == np.datetime64("1980-01-31")
    assert months[-1].last_day.astype("datetime64[D]") == np.datetime64("2024-12-31")
    # February length must come from the calendar, not a fixed 28 or 29.
    february = {m.year: m.last_day for m in months if m.month == 2}
    assert february[2000].astype("datetime64[D]") == np.datetime64("2000-02-29")
    assert february[2001].astype("datetime64[D]") == np.datetime64("2001-02-28")
    assert len({m.label for m in months}) == len(months)


def test_last_pentad_index_never_crosses_a_month_boundary() -> None:
    # Pentads every five days from 1980-01-05, so January holds 05, 10, 15, 20, 25, 30.
    epoch_ms = pd.Timestamp("1980-01-01").value // 1_000_000
    stamps = epoch_ms + (np.arange(6) * 5 + 4).astype("int64") * 86_400_000
    months = pilot.month_ends(1980, 1980)[:1]
    assert pilot.last_pentad_index_per_month(stamps, months) == [5], (
        "January must take the 30 January pentad, the last one inside the month"
    )


def test_last_pentad_index_reports_a_month_before_the_first_pentad_and_borrows_backwards() -> None:
    """The sampler never reaches forward; a month with no pentad of its own reuses the previous one.

    Both behaviours are pinned here because the report states which pentad anchor
    it used rather than assuming one pentad per month exists.
    """
    epoch_ms = pd.Timestamp("1980-01-01").value // 1_000_000
    january_pentad = np.array([epoch_ms + 4 * 86_400_000], dtype="int64")
    months = pilot.month_ends(1980, 1980)[:2]
    assert pilot.last_pentad_index_per_month(january_pentad, months) == [0, 0]
    february_pentad = np.array([epoch_ms + 31 * 86_400_000 + 4 * 86_400_000], dtype="int64")
    assert pilot.last_pentad_index_per_month(february_pentad, months) == [-1, 0], (
        "January has no pentad at or before its last day"
    )


def test_usdm_bin_matches_the_documented_catalog_boundaries() -> None:
    values = np.array(list(CATALOG_BOUNDARIES) + [-6.0, 6.0])
    bins = pilot.usdm_bin(values)
    # The catalog reads "-2.0 or less" as extreme drought and "2.0 or more" as
    # extremely wet, so the boundary belongs to the outer class, not the inner one.
    assert bins[0] == 1, "-2.0 is the lower edge of 'extreme drought', not of 'exceptional'"
    assert bins[-2] == 0, "-6.0 is exceptional drought"
    assert bins[-1] == 10, "6.0 is exceptionally wet"
    ordered = pilot.usdm_bin(np.array(sorted(CATALOG_BOUNDARIES)))
    assert np.all(np.diff(ordered.astype(int)) == 1), "each listed boundary starts the next class"
    assert pilot.usdm_bin(np.array([-0.49, 0.0, 0.49]))[0] == pilot.usdm_bin(np.array([0.0]))[0] == 5


def test_comparison_metrics_are_hand_computable() -> None:
    ours = np.array([0.0, 1.0, -1.0])
    theirs = np.array([0.0, 2.0, -2.0])
    metrics = pilot.comparison_metrics(ours, theirs)
    assert metrics["n"] == 3
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert metrics["max_abs_difference"] == pytest.approx(1.0)
    assert metrics["correlation"] == pytest.approx(1.0), "the differenced pairs are perfectly linear here"
    # Categories: (near normal, near normal), (incipient wet, extremely wet),
    # (moderate drought, extreme drought) -> one of three in the same class.
    assert metrics["category_agreement"] == pytest.approx(1.0 / 3.0)
    assert metrics["category_within_one"] == pytest.approx(1.0 / 3.0)


def test_comparison_metrics_drop_missing_pairs_instead_of_counting_them() -> None:
    ours = np.array([0.0, np.nan, 1.0, np.inf])
    theirs = np.array([0.0, 0.0, np.nan, 1.0])
    metrics = pilot.comparison_metrics(ours, theirs)
    assert metrics["n"] == 1, "only the first pair has two finite values"
    assert metrics["bias"] == pytest.approx(0.0)
    assert np.isnan(metrics["correlation"]), "a single pair has no correlation"


def test_comparison_metrics_reject_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        pilot.comparison_metrics(np.zeros(3), np.zeros(4))


def _single_cell(values: np.ndarray, start: str) -> xr.DataArray:
    time = pd.date_range(start, periods=values.size, freq="ME")
    return xr.DataArray(values, dims=("time",), coords={"time": time}, attrs={"units": "mm"}, name="precipitation")


def test_empirical_standardization_uses_the_declared_plotting_position() -> None:
    """The comparison rests on this convention: p = (k + 1) / (n + 2), k = rank from 1.

    The expected values come from ``scipy.stats.norm.ppf`` rather than from this
    library, so a change to the plotting position fails here instead of silently
    shifting every pilot number.
    """
    years = np.arange(1980, 2000)
    values = np.tile(np.arange(1.0, 13.0), years.size).reshape(years.size, 12)
    values[:, 0] = np.arange(1.0, years.size + 1)  # January increases with the year
    data = _single_cell(values.ravel().astype("float64"), "1980-01-31")
    probability = standardized_index(
        data,
        1,
        Distribution.empirical,
        calibration_year_initial=1980,
        calibration_year_final=1999,
        output_scale="probability",
    )
    normal = standardized_index(
        data,
        1,
        Distribution.empirical,
        calibration_year_initial=1980,
        calibration_year_final=1999,
    )
    ranks = np.arange(1, years.size + 1)
    expected = (ranks + 1) / (years.size + 2)
    january_probability = probability.isel(time=probability["time"].dt.month.values == 1)
    assert np.allclose(january_probability.values, expected, atol=1e-12, rtol=0)
    # The normal scale applies a Hastings inverse-normal approximation, so it
    # tracks the exact quantile to about a hundredth in the tail rather than to
    # machine precision. This pins the plotting position, not the approximation.
    january_normal = normal.isel(time=normal["time"].dt.month.values == 1)
    assert np.allclose(january_normal.values, norm.ppf(expected), atol=0.02, rtol=0)


def test_empirical_baseline_length_caps_the_index_magnitude() -> None:
    """A 36-year baseline cannot express a rank beyond p = 37 / 38, about z = 1.94.

    This is a property of rank-based standardization, not a defect, and the pilot
    report states it as a structural limit of the comparison.
    """
    # 1981-2016 is the catalog's documented baseline, so the record starts one
    # year earlier and gives that window exactly 36 calendar steps per month.
    years = np.arange(1980, 2017)
    values = np.tile(np.arange(1.0, 13.0), years.size).reshape(years.size, 12)
    values[:, 0] = np.arange(1.0, years.size + 1)
    data = _single_cell(values.ravel().astype("float64"), "1980-01-31")
    probability = standardized_index(
        data,
        1,
        Distribution.empirical,
        calibration_year_initial=1981,
        calibration_year_final=2016,
        output_scale="probability",
    )
    normal = standardized_index(
        data,
        1,
        Distribution.empirical,
        calibration_year_initial=1981,
        calibration_year_final=2016,
    )
    # Only the calibration years carry a full 36-value sample; a value outside it
    # extrapolates to the open end of the plotting-position range.
    january = (probability["time"].dt.month.values == 1) & (probability["time"].dt.year.values >= 1981)
    probability_january = probability.isel(time=january)
    normal_january = normal.isel(time=january)
    assert float(probability_january.max()) == pytest.approx(37.0 / 38.0, abs=1e-12)
    assert float(probability_january.min()) == pytest.approx(2.0 / 38.0, abs=1e-12)
    assert np.max(np.abs(normal_january.values)) < 2.0, "the rank-based convention cannot leave (-2, 2) here"


def test_spei_path_keeps_a_signed_water_balance() -> None:
    """SPEI standardizes P minus PET, so a dry cell must stay signed.

    With clipping on, every negative balance becomes zero and the whole series
    collapses to one value. The pilot therefore passes ``clip_negatives=False``,
    and this test pins what that flag changes.
    """
    years = np.arange(1980, 2000)
    # Seasonal precipitation with year-to-year noise, so every calendar step has
    # its own ranking and the result is not a flat series.
    generator = np.random.default_rng(20261110)
    seasonal = 50.0 + 40.0 * np.sin(2.0 * np.pi * (np.arange(12) + 0.5) / 12.0)
    precipitation = seasonal[None, :] + generator.normal(0.0, 10.0, size=(years.size, 12))
    pet = np.full_like(precipitation, 400.0)  # every balance is negative
    balance = _single_cell((precipitation - pet).ravel(), "1980-01-31")
    clipped = standardized_index(
        balance, 1, Distribution.empirical, calibration_year_initial=1980, calibration_year_final=1999
    )
    signed = standardized_index(
        balance,
        1,
        Distribution.empirical,
        calibration_year_initial=1980,
        calibration_year_final=1999,
        clip_negatives=False,
    )
    assert np.ptp(clipped.values) == 0.0, "clipping an all-negative balance makes every step identical"
    assert np.all(np.isfinite(signed.values)), "the signed path returns a value for every step"
    assert len(np.unique(np.round(signed.values, 6))) > 12, "the signed path keeps the year-to-year ranking"
    assert float(np.max(signed.values)) > 0.5
    assert float(np.min(signed.values)) < -0.5


def test_crop_grid_to_bbox_keeps_the_native_pixel_edges() -> None:
    cropped = pilot.crop_grid_to_bbox(NATIVE_GRID, pilot.PILOT_BBOX)
    scale_x, _, origin_x, _, scale_y, origin_y = cropped["crs_transform"]
    columns, rows = cropped["shape_2d"]
    lon_min, lat_min, lon_max, lat_max = pilot.PILOT_BBOX

    # The window covers the bounding box ...
    assert origin_x <= lon_min
    assert origin_x + columns * scale_x >= lon_max
    assert origin_y >= lat_max
    assert origin_y + rows * scale_y <= lat_min
    # ... and is the smallest such window, so it does not read the whole CONUS grid.
    assert columns * scale_x - (lon_max - lon_min) < 2 * scale_x
    assert rows * abs(scale_y) - (lat_max - lat_min) < 2 * abs(scale_y)
    assert (columns, rows) == (58, 133)

    # The cropped origin sits on the native lattice, which is what avoids resampling.
    native_origin_x, native_origin_y = NATIVE_GRID["crs_transform"][2], NATIVE_GRID["crs_transform"][5]
    assert (origin_x - native_origin_x) / scale_x == pytest.approx(round((origin_x - native_origin_x) / scale_x))
    assert (native_origin_y - origin_y) / abs(scale_y) == pytest.approx(
        round((native_origin_y - origin_y) / abs(scale_y))
    )


def test_crop_grid_to_bbox_rejects_a_metric_grid() -> None:
    metric = {"crs_transform": (4638.3, 0, 0.0, 0, -4638.3, 0.0), "shape_2d": (10, 10)}
    with pytest.raises(ValueError, match="not in degrees"):
        pilot.crop_grid_to_bbox(metric, pilot.PILOT_BBOX)


def test_grids_match_distinguishes_origin_and_shape() -> None:
    left = pilot.crop_grid_to_bbox(NATIVE_GRID, pilot.PILOT_BBOX)
    assert pilot.grids_match(left, dict(left))
    shifted = dict(left)
    transform = list(left["crs_transform"])
    transform[2] = transform[2] + transform[0]
    shifted["crs_transform"] = tuple(transform)
    assert not pilot.grids_match(left, shifted), "a one-pixel origin shift is a different grid"
    taller = dict(left)
    taller["shape_2d"] = (left["shape_2d"][0], left["shape_2d"][1] + 1)
    assert not pilot.grids_match(left, taller)


def test_check_monthly_totals_catches_unit_mistakes() -> None:
    pilot.check_monthly_totals(np.full((3, 2), 80.0), "precipitation")
    with pytest.raises(RuntimeError, match="negative"):
        pilot.check_monthly_totals(np.array([[80.0, -1.0]]), "precipitation")
    with pytest.raises(RuntimeError, match="domain mean"):
        # metres instead of millimetres: three orders of magnitude low
        pilot.check_monthly_totals(np.full((3, 2), 0.08), "precipitation")
    with pytest.raises(RuntimeError, match="domain mean"):
        # a per-day value that was never summed over the month
        pilot.check_monthly_totals(np.full((3, 2), 2.6), "precipitation")


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


def test_climate_engine_request_keeps_the_key_out_of_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key belongs in the Authorization header only; it must never reach a file or the body."""
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse({"Data": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    secret = "not-a-real-key"
    result = pilot.climate_engine_series(secret, 3, "spi")

    assert captured["url"] == pilot.CE_ENDPOINT
    assert captured["headers"]["Authorization"] == secret
    assert secret not in json.dumps(captured["body"])
    assert secret not in json.dumps(result)
    body = captured["body"]
    assert body["distribution"] == "nonparametric", "the reference documents a non-parametric option"
    assert body["dataset"] == "GRIDMET"
    assert body["accumulation"] == "90", "accumulation is in days for the daily GRIDMET dataset"
    assert body["start_year"] == pilot.CALIBRATION_YEARS[0]
    assert body["end_year"] == pilot.CALIBRATION_YEARS[1]
    assert json.loads(body["coordinates"]) == [[lon, lat] for lon, lat in pilot.CE_POINTS]


def test_climate_engine_native_request_targets_the_native_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.header_items())
        return _FakeResponse({"Data": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    pilot.climate_engine_native("key", "GRIDMET", "pr", "1981-01-01", "2024-12-31")
    assert captured["url"] == pilot.CE_NATIVE_ENDPOINT
    assert captured["body"]["dataset"] == "GRIDMET"
    assert captured["body"]["variable"] == "pr"
    assert captured["body"]["start_date"] == "1981-01-01"
    assert captured["headers"]["Authorization"] == "key"


def test_parse_climate_engine_points_reads_every_block() -> None:
    """The response holds one block per point, labelled by DRI_OBJECTID, in daily records."""
    payload = {
        "Data": [
            {
                "Metadata": {"DRI_OBJECTID": "[-84.9, 30.7]"},
                "Data": [{"Date": "2020-01-01", "spi": 1.25}, {"Date": "2020-01-02", "spi": -9999.0}],
            },
            {
                "Metadata": {"DRI_OBJECTID": "[-84.4, 33.0]"},
                "Data": [{"Date": "2020-01-01", "spi": -0.5}, {"Date": "2020-01-02", "spi": 0.0}],
            },
        ]
    }
    points = pilot.parse_climate_engine_points(payload, "spi")
    assert set(points) == {(-84.9, 30.7), (-84.4, 33.0)}
    first = points[(-84.9, 30.7)]
    assert first.iloc[0] == pytest.approx(1.25)
    assert np.isnan(first.iloc[1]), "the API's -9999 sentinel must not enter a comparison as a real value"
    assert first.index[0] == pd.Timestamp("2020-01-01"), "records keep their own dates"


def test_parse_climate_engine_points_accepts_the_unit_suffixed_native_column() -> None:
    """The native endpoint names its column after the variable and the catalog unit."""
    payload = {
        "Data": [
            {
                "Metadata": {"DRI_OBJECTID": "[-84.9, 30.7]"},
                "Data": [{"Date": "2020-01-01", "pr (mm)": 4.5}, {"Date": "2020-01-02", "pr (mm)": 0.0}],
            }
        ]
    }
    points = pilot.parse_climate_engine_points(payload, "pr")
    assert list(points[(-84.9, 30.7)]) == [4.5, 0.0]


def test_parse_climate_engine_points_rejects_records_without_the_variable() -> None:
    payload = {"Data": [{"Metadata": {"DRI_OBJECTID": "[-84.9, 30.7]"}, "Data": [{"Date": "2020-01-01"}]}]}
    with pytest.raises(RuntimeError, match="none of them 'pr'"):
        pilot.parse_climate_engine_points(payload, "pr")


def test_parse_climate_engine_points_rejects_an_unreadable_label() -> None:
    payload = {"Data": [{"Metadata": {"DRI_OBJECTID": "not-a-coordinate"}, "Data": []}]}
    with pytest.raises(RuntimeError, match="could not read the point label"):
        pilot.parse_climate_engine_points(payload, "spi")


def test_sample_at_month_ends_takes_the_last_record_on_or_before() -> None:
    """Sampling must land on an observation, never on an interpolation."""
    index = pd.to_datetime(["1981-01-05", "1981-01-30", "1981-02-03", "1981-02-27"])
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    months = pilot.month_ends(1981, 1981)[:2]
    assert list(pilot.sample_at_month_ends(series, months)) == [2.0, 4.0]


def test_sample_at_month_ends_returns_missing_before_the_series_starts() -> None:
    series = pd.Series([9.0], index=pd.to_datetime(["1981-03-01"]))
    months = pilot.month_ends(1981, 1981)[:2]
    sampled = pilot.sample_at_month_ends(series, months)
    assert np.isnan(sampled).all(), "a month before the first record has no observation to sample"


def test_sample_at_month_ends_does_not_fill_across_a_missing_month() -> None:
    series = pd.Series([9.0], index=pd.to_datetime(["1981-01-31"]))
    sampled = pilot.sample_at_month_ends(series, pilot.month_ends(1981, 1981)[:2])
    assert sampled[0] == 9.0
    assert np.isnan(sampled[1])


def test_parse_climate_engine_points_rejects_empty_records() -> None:
    payload = {"Data": [{"Metadata": {"DRI_OBJECTID": "[-84.9, 30.7]"}, "Data": []}]}
    with pytest.raises(RuntimeError, match="no records"):
        pilot.parse_climate_engine_points(payload, "spi")


def test_climate_engine_http_error_does_not_echo_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "not-a-real-key"

    def fail(request: Any, timeout: int) -> None:
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", Message(), io.BytesIO(secret.encode()))

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError, match="HTTP 403") as error:
        pilot.climate_engine_series(secret, 1, "spi")
    assert secret not in str(error.value)
    assert error.value.__suppress_context__


def test_compare_climate_engine_selects_dates_not_input_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "COMPARISON_YEARS", (1981, 1981))
    monkeypatch.setattr(pilot, "SCALES", {1: "30d"})
    monkeypatch.setattr(pilot, "CE_POINTS", ((-84.9, 30.7),))
    dates = pd.date_range("1980-01-31", periods=24, freq="ME")
    values = np.arange(24.0)
    cell = xr.DataArray(
        values[:, None, None], dims=("time", "y", "x"), coords={"time": dates, "y": [30.7], "x": [-84.9]}
    )
    ours = {f"{index}_1": cell for index in ("spi", "spei", "eddi")}

    def response(key: str, scale: int, variable: str) -> dict[str, Any]:
        return {
            "request": {},
            "response": {
                "Data": [
                    {
                        "Metadata": {"DRI_OBJECTID": "[-84.9, 30.7]"},
                        "Data": [
                            {"Date": date.strftime("%Y-%m-%d"), variable: value}
                            for date, value in zip(dates[12:], values[12:], strict=True)
                        ],
                    }
                ]
            },
        }

    monkeypatch.setattr(pilot, "climate_engine_series", response)
    metrics, _ = pilot.compare_climate_engine(ours, "not-a-real-key")
    for result in metrics.values():
        assert result["n"] == 12
        assert result["mae"] == 0.0


def test_gridmet_filter_includes_the_final_day(monkeypatch: pytest.MonkeyPatch) -> None:
    class Collection:
        def filterDate(self, start: str, end: str) -> None:
            assert start == "1980-01-01"
            assert end == "2025-01-01"
            raise RuntimeError("date checked")

    class EarthEngine:
        def ImageCollection(self, name: str) -> Collection:
            return Collection()

    with pytest.raises(RuntimeError, match="date checked"):
        pilot.open_monthly_gridmet(EarthEngine(), pilot.PILOT_BBOX, NATIVE_GRID)


def test_scale_mapping_covers_the_catalog_scales() -> None:
    """Each calendar-month scale must name a band that the catalog actually publishes."""
    assert pilot.SCALES == {1: "30d", 3: "90d", 6: "180d", 12: "1y"}
    assert set(pilot.SCALE_DAYS) == set(pilot.SCALES)

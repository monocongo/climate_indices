"""Offline checks for the pre-registered plotting-position experiment (#1571)."""

import gzip
import hashlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import norm

from climate_indices import compute
from climate_indices.drought import indices
from scripts import earthengine_climate_engine_pilot as pilot


def test_tukey_known_ranks_ties_edges_and_missing() -> None:
    calibration = np.arange(1.0, 37.0)[:, None]
    values = np.concatenate(([-1.0], np.arange(1.0, 37.0), [37.0, np.nan]))[:, None]
    probability = pilot.tukey_probabilities(calibration, values)
    ranks = np.concatenate(([1], np.arange(1, 37), [36]))
    expected = (ranks - 1 / 3) / (36 + 1 / 3)
    np.testing.assert_allclose(norm.ppf(probability[:-1, 0]), norm.ppf(expected), atol=1e-14, rtol=0)
    assert np.isnan(probability[-1, 0])
    tied = np.array([0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan])[:, None]
    assert np.isnan(pilot.tukey_probabilities(tied, np.array([[0.0]]))).all()
    tied[-1] = 9.0
    assert pilot.tukey_probabilities(tied, np.array([[0.0]]))[0, 0] == pytest.approx((2 / 3) / (10 + 1 / 3))


def test_tukey_matches_eddi_rank_semantics_with_documented_numerical_differences() -> None:
    calibration = np.arange(36.0)[:, None]
    values = np.array([[0.0], [12.0], [35.0], [np.nan]])
    ours = pilot.tukey_probabilities(calibration, values)
    below = np.count_nonzero(calibration[:, None, :] < values[None, :, :], axis=0)
    eddi_probability = (below + 0.66) / (36 + 0.33)
    eddi_probability[np.isnan(values)] = np.nan
    # Exact thirds versus NOAA's rounded constants differ by at most 0.0002 here.
    np.testing.assert_allclose(ours, eddi_probability, atol=0.0002, rtol=0, equal_nan=True)
    native = compute._native_tukey_probabilities(calibration, values, np.zeros(1))
    if native is not None:
        np.testing.assert_allclose(native, eddi_probability, atol=1e-14, rtol=0, equal_nan=True)
    data = xr.DataArray(
        np.repeat(np.arange(36.0), 12),
        dims="time",
        coords={"time": pd.date_range("1981-01-31", periods=432, freq="ME")},
    )
    result = pilot.tukey_index(data, 1, clip_negatives=False)
    expected = indices.eddi(data.values, 1, 1981, 1981, 2016, compute.Periodicity.monthly)
    # NOAA's rounded constants and Hastings approximation change z by under 0.005.
    np.testing.assert_allclose(result.values, expected, atol=0.005, rtol=0)


def test_compute_indices_sei_is_unchanged_and_eddi_is_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "SCALES", {1: "30d", 3: "90d"})
    random = np.random.default_rng(1571)
    dates = pd.date_range("1980-01-31", periods=540, freq="ME")
    dataset = xr.Dataset(
        {
            name: ("time", random.uniform(0, 100, 540), {"units": "mm"})
            for name in ("precipitation", "reference_et", "alfalfa_et")
        },
        coords={"time": dates},
    )
    original = pilot.compute_indices(dataset)
    explicit = pilot.compute_indices(dataset, "sei")
    tukey = pilot.compute_indices(dataset, "tukey")
    for key in original:
        np.testing.assert_array_equal(original[key].values, explicit[key].values)
        if key.startswith("eddi"):
            np.testing.assert_array_equal(original[key].values, tukey[key].values)
    for scale in pilot.SCALES:
        for index, data in (("spi", dataset.precipitation), ("spei", dataset.precipitation - dataset.reference_et)):
            expected = indices.standardized_index(
                data.values,
                scale,
                indices.Distribution.empirical,
                1980,
                1981,
                2016,
                compute.Periodicity.monthly,
                clip_negatives=index == "spi",
            )
            np.testing.assert_array_equal(original[f"{index}_{scale}"].values, expected)


def test_paired_metrics_use_common_pairs_and_split_dates() -> None:
    dates = pd.date_range("1981-01-31", periods=528, freq="ME")
    reference = xr.DataArray(np.arange(528.0), dims="time", coords={"time": dates})
    sei = reference + 0.05
    tukey = reference.copy()
    tukey[0] = np.nan
    result = pilot.paired_metrics(sei, tukey, reference)
    assert result["1981-2024"]["sei"]["n"] == 527
    assert result["1981-2016"]["sei"]["n"] == 431
    assert result["2017-2024"]["sei"]["n"] == 96
    assert result["1981-2024"]["change"]["bias"] == pytest.approx(-0.05)
    assert len(result["calendar_month_bias"]["tukey"]) == 12
    assert result["calendar_month_bias"]["tukey"] == [0.0] * 12


def test_baseline_control_stops_on_drift() -> None:
    baseline = {"eddi_1": {"n": 12.0, "bias": 0.01, "correlation": 0.99}}
    pilot.check_baseline(baseline, baseline)
    with pytest.raises(RuntimeError, match="baseline drift.*eddi_1.*bias"):
        pilot.check_baseline({"eddi_1": {"n": 12.0, "bias": 0.02, "correlation": 0.99}}, baseline)


def test_expected_reduction_comes_from_formulas() -> None:
    expected = pilot.expected_rank_reduction(36)
    assert expected == pytest.approx(0.053831430857023244, abs=1e-14)


def test_tukey_grid_keeps_months_cells_signs_and_missing_separate() -> None:
    dates = pd.date_range("1980-01-31", periods=540, freq="ME")
    years = np.repeat(np.arange(45.0) - 20, 12)
    values = np.broadcast_to(years[:, None, None], (540, 2, 2)).copy()
    values[1::12, 0, 0] *= -1
    values[:, 1, 1] = np.nan
    data = xr.DataArray(values, dims=("time", "y", "x"), coords={"time": dates})
    result = pilot.tukey_index(data, 1, clip_negatives=False)
    for y, x in ((0, 0), (0, 1), (1, 0)):
        single = pilot.tukey_index(data.isel(y=y, x=x), 1, clip_negatives=False)
        np.testing.assert_array_equal(result.isel(y=y, x=x).values, single.values)
    assert np.isnan(result.isel(y=1, x=1)).all()
    assert result.isel(time=12, y=0, x=0) < 0
    assert result.isel(time=13, y=0, x=0) > 0
    assert np.isnan(pilot.tukey_index(data.isel(y=1, x=1), 3, clip_negatives=False)).all()
    truncated = data.isel(time=slice(1, None))
    with pytest.raises(ValueError, match="complete January-to-December"):
        pilot.tukey_index(truncated, 1, clip_negatives=False)


def test_followup_reads_each_climate_engine_series_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "SCALES", {1: "30d"})
    monkeypatch.setattr(pilot, "CE_POINTS", ((-84.9, 30.7),))
    dates = pd.date_range("1980-01-31", periods=540, freq="ME")
    cell = xr.DataArray(
        np.arange(540.0)[:, None, None], dims=("time", "y", "x"), coords={"time": dates, "y": [30.7], "x": [-84.9]}
    )
    sei = {f"{index}_1": cell for index in ("spi", "spei", "eddi")}
    tukey = {name: array if name.startswith("eddi") else array - 0.05 for name, array in sei.items()}
    calls = []

    def response(key: str, scale: int, variable: str) -> dict[str, Any]:
        calls.append(variable)
        return {
            "request": {"variable": variable},
            "response": {
                "Data": [
                    {
                        "Metadata": {"DRI_OBJECTID": "[-84.9, 30.7]"},
                        "Data": [
                            {"Date": date.strftime("%Y-%m-%d"), variable: float(i)} for i, date in enumerate(dates)
                        ],
                    }
                ]
            },
        }

    monkeypatch.setattr(pilot, "climate_engine_series", response)
    results, baseline = pilot.followup_climate_engine(sei, tukey, "not-a-real-key")
    assert calls == ["spi", "spei", "eddi"]
    assert baseline["spi_1"]["bias"] == 0
    assert results["spi_1"]["1981-2024"]["change"]["bias"] == pytest.approx(-0.05)
    assert results["eddi_1"]["1981-2024"]["change"]["bias"] == 0
    assert "response" not in results["spi_1"]


@pytest.mark.parametrize("skip", [True, False])
def test_followup_preserves_original_results_when_b_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    skip: bool,
) -> None:
    monkeypatch.setattr(pilot, "SCALES", {1: "30d"})
    dates = pd.date_range("1980-01-31", periods=540, freq="ME")
    values = np.random.default_rng(1571).uniform(1.0, 100.0, (540, 2, 2))
    dataset = xr.Dataset(
        {
            name: (("time", "y", "x"), values * (1 if name == "precipitation" else 0.5), {"units": "mm"})
            for name in ("precipitation", "reference_et", "alfalfa_et")
        },
        coords={"time": dates},
    )
    ours = pilot.compute_indices(dataset)
    reference = xr.Dataset(
        {pilot.band_for(name): array.sel(time=slice("1981", "2024")) for name, array in ours.items()}
    )
    original_path = (
        Path(pilot.__file__).resolve().parents[1] / "docs/research/earthengine-pilot-results/pilot-metrics.json"
    )
    original = json.loads(original_path.read_text())
    monkeypatch.setattr(pilot, "check_baseline", lambda current, baseline: None)
    monkeypatch.setenv("CE_API_KEY", "not-a-real-key")

    def timeout(*args: Any) -> None:
        raise TimeoutError("not logged")

    monkeypatch.setattr(pilot, "followup_climate_engine", timeout)
    manifest = {pilot.DAILY_COLLECTION: [{"id": "sample", "system:index": "sample", "system:version": None}]}
    pilot.run_tukey_followup(dataset, reference, {}, manifest, tmp_path, skip)
    saved = json.loads((tmp_path / "pilot-metrics.json").read_text())
    for name in original.keys() - {"tukey_followup_1571"}:
        assert saved[name] == original[name]
    result = saved["tukey_followup_1571"]
    assert result["reference_b"]["status"] == "pending"
    assert result["reference_b"]["http_timeout_seconds"] == pilot.HTTP_TIMEOUT_SECONDS
    if skip:
        assert result["reference_b"]["reason"] == "skipped"
    assert result["image_manifest"]["missing_version"][pilot.DAILY_COLLECTION] == 1
    assert json.loads(gzip.decompress((tmp_path / result["image_manifest"]["file"]).read_bytes())) == manifest
    assert "not-a-real-key" not in json.dumps(saved)


def test_credentials_are_never_forwarded_to_a_redirect_target() -> None:
    handlers = vars(pilot._OPENER)["handlers"]
    assert any(isinstance(handler, pilot._RefuseRedirect) for handler in handlers)
    assert not any(type(handler) is urllib.request.HTTPRedirectHandler for handler in handlers), (
        "a plain redirect handler would follow the target with the Authorization header"
    )
    request = urllib.request.Request("https://example.invalid", headers={"Authorization": "secret"})
    refused = pilot._RefuseRedirect().redirect_request(request, None, 302, "Found", {}, "http://elsewhere")
    assert refused is None


def test_transport_error_retains_reference_a_and_records_the_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pilot, "SCALES", {1: "30d"})
    dates = pd.date_range("1980-01-31", periods=540, freq="ME")
    values = np.random.default_rng(7).uniform(1.0, 100.0, (540, 2, 2))
    dataset = xr.Dataset(
        {
            name: (("time", "y", "x"), values * (1 if name == "precipitation" else 0.5), {"units": "mm"})
            for name in ("precipitation", "reference_et", "alfalfa_et")
        },
        coords={"time": dates},
    )
    ours = pilot.compute_indices(dataset)
    reference = xr.Dataset(
        {pilot.band_for(name): array.sel(time=slice("1981", "2024")) for name, array in ours.items()}
    )
    monkeypatch.setattr(pilot, "check_baseline", lambda current, baseline: None)
    monkeypatch.setenv("CE_API_KEY", "not-a-real-key")

    def refused(*args: Any) -> None:
        raise pilot.ClimateEngineTransportError("Climate Engine API returned HTTP 503")

    monkeypatch.setattr(pilot, "followup_climate_engine", refused)
    pilot.run_tukey_followup(dataset, reference, {}, {}, tmp_path, False)
    result = json.loads((tmp_path / "pilot-metrics.json").read_text())["tukey_followup_1571"]
    assert result["reference_a"], "reference A must survive a provider refusal"
    assert result["reference_b"] == {
        "status": "pending",
        "reason": "Climate Engine API returned HTTP 503",
        "http_timeout_seconds": pilot.HTTP_TIMEOUT_SECONDS,
    }
    assert "not-a-real-key" not in json.dumps(result)


def test_saved_followup_metrics_and_manifest_are_consistent() -> None:
    directory = Path(pilot.__file__).resolve().parents[1] / "docs/research/earthengine-pilot-results"
    payload = json.loads((directory / "pilot-metrics.json").read_text())
    result = payload["tukey_followup_1571"]
    assert result["software"]["git_dirty"] is False
    assert result["reference_b"] == {
        "status": "pending",
        "reason": "skipped",
        "http_timeout_seconds": pilot.HTTP_TIMEOUT_SECONDS,
    }, "the saved B record must be what a --skip-climate-engine rerun writes"
    assert result["baseline_control"] == pilot.CONTROL_TOLERANCE_A
    for name, rows in result["reference_a"].items():
        pilot.check_baseline({name: rows["1981-2024"]["sei"]}, {name: payload["reference_a"][name]})
        assert rows["1981-2024"]["sei"]["n"] == rows["1981-2016"]["sei"]["n"] + rows["2017-2024"]["sei"]["n"]
        assert len(rows["calendar_month_bias"]["tukey"]) == 12
        for period in ("1981-2024", "1981-2016", "2017-2024"):
            for metric, change in rows[period]["change"].items():
                assert change == pytest.approx(rows[period]["tukey"][metric] - rows[period]["sei"][metric], abs=1e-14)
                if name.startswith("eddi"):
                    assert change == 0.0
    assert result["image_manifest"]["missing_version"] == {pilot.DAILY_COLLECTION: 0, pilot.DROUGHT_COLLECTION: 0}
    manifest_bytes = (directory / result["image_manifest"]["file"]).read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == result["image_manifest"]["sha256"]
    # Bound the expansion: a pull request controls both the blob and its recorded hash.
    with gzip.GzipFile(fileobj=io.BytesIO(manifest_bytes)) as handle:
        raw = handle.read(8 * 1024 * 1024)
        assert handle.read(1) == b"", "the manifest must fit the decompression cap"
    manifest = json.loads(raw)
    assert len(manifest[pilot.DAILY_COLLECTION]) == 16437
    assert len(manifest[pilot.DROUGHT_COLLECTION]) == 528
    for rows in manifest.values():
        assert all({"id", "system:index", "system:version"} <= row.keys() for row in rows)

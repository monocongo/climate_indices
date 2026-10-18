"""Offline integrity and author-method checks; the compound API does not exist yet."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from collections import Counter
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixture" / "compound"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_compound_reference_fixture.py"
REFERENCE = json.loads((FIXTURE_DIR / "reference.json").read_text())


def _inputs() -> dict[str, np.ndarray]:
    with np.load(FIXTURE_DIR / "inputs.npz", allow_pickle=False) as inputs:
        return dict(inputs)


@pytest.fixture
def preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    script = runpy.run_path(str(SCRIPT))
    monkeypatch.setitem(script["prepare"].__globals__, "OUTPUT_ROOT", tmp_path.resolve())
    return script


def test_reference_checksum_pins_committed_inputs() -> None:
    reference_bytes = (FIXTURE_DIR / "reference.json").read_bytes()
    provenance = json.loads((FIXTURE_DIR / "provenance.json").read_text())
    assert hashlib.sha256(reference_bytes).hexdigest() == provenance["checksum_sha256"]
    assert hashlib.sha256((FIXTURE_DIR / "inputs.npz").read_bytes()).hexdigest() == REFERENCE["inputs_sha256"]
    assert provenance["validation_tolerance"] == {"count_atol": 0, "lmf_atol": 1e-12, "rtol": 0}


def test_raw_inputs_preserve_source_dates_missingness_and_locations() -> None:
    inputs = _inputs()
    assert set(inputs) == {"monthly_time", "daily_time", "monthly_spei", "daily_hwi"}
    assert inputs["monthly_spei"].shape == (1200, 3)
    assert inputs["daily_hwi"].shape == (36525, 3)
    months = np.arange("1922-01", "2022-01", dtype="datetime64[M]")
    month_ends = (months + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
    np.testing.assert_array_equal(inputs["monthly_time"], month_ends)
    np.testing.assert_array_equal(inputs["daily_time"], np.arange("1922-01-01", "2022-01-01", dtype="datetime64[D]"))
    assert inputs["monthly_spei"].dtype == inputs["daily_hwi"].dtype == np.dtype("float64")
    assert np.isfinite(inputs["daily_hwi"]).all()
    for column, cell in enumerate(REFERENCE["cells"]):
        missing = np.isnan(inputs["monthly_spei"][:, column])
        assert missing.sum() == 11
        assert (
            inputs["monthly_time"][missing].astype("datetime64[D]").astype(str).tolist()
            == cell["monthly_missing_dates"]
        )
    assert [(cell["lat_index"], cell["lon_index"]) for cell in REFERENCE["cells"]] == [(5, 23), (48, 30), (83, 68)]
    np.testing.assert_allclose(
        [(cell["latitude"], cell["longitude"]) for cell in REFERENCE["cells"]],
        [
            (56.03652424242424, 14.046454545454546),
            (62.04727272727273, 14.973636363636365),
            (66.93974242424243, 20.00690909090909),
        ],
        rtol=0,
        atol=1e-12,
    )


def test_reference_records_source_and_method_pins() -> None:
    assert REFERENCE["dataset_doi"] == "10.6084/m9.figshare.25189118.v2"
    assert REFERENCE["paper_doi"] == "10.1007/s00477-024-02783-3"
    assert REFERENCE["author_commit"] == "6c88f255ee3034279dcebe2a2fc40c954dbe9575"
    assert REFERENCE["sources"]["SPEI12.nc"]["file_id"] == 44481290
    assert REFERENCE["sources"]["heatwave_SMHI.nc"]["file_id"] == 44481308
    assert (
        REFERENCE["sources"]["SPEI12.nc"]["sha256"]
        == "bb2b32bc5c6d76ce3257a0b211696128ad6da13ae737650a1376524788e88dce"
    )
    assert (
        REFERENCE["sources"]["heatwave_SMHI.nc"]["sha256"]
        == "c0428087c22cd6586f77af83592185ca513d2f52a809b7173a85960a3497701b"
    )
    assert REFERENCE["preparation"]["heat_comparator"] == "HWI >= 0"
    assert REFERENCE["preparation"]["drought_comparator"] == "SPEI <= -1"
    assert "not reproduced" in REFERENCE["preparation"]["significance"]
    assert [(case["n_heat"], case["n_drought"], case["n_joint"]) for case in REFERENCE["cases"]] == [
        (232, 566, 43),
        (530, 469, 68),
        (303, 590, 63),
        (178, 509, 26),
        (340, 494, 45),
        (123, 435, 26),
        (136, 743, 36),
        (264, 557, 48),
        (66, 523, 19),
    ]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: f"cell{case['cell']}-month{case['month']}")
def test_author_reference_matches_independent_contingency_counts(case: dict[str, Any]) -> None:
    """Use index lookup, interpolation, Counter, and rational arithmetic, not the generator."""
    inputs = _inputs()
    column = case["cell"]
    dates = inputs["daily_time"]
    dates = dates[(dates >= np.datetime64("1922-01-31")) & (dates <= np.datetime64("2021-12-31"))]
    monthly_positions = np.searchsorted(inputs["monthly_time"], dates, side="right") - 1
    drought_values = inputs["monthly_spei"][monthly_positions, column].copy()
    valid = np.isfinite(drought_values)
    assert valid[0]
    assert valid[-1]
    # Only missing padded values interpolate; the finite month-end values stay constant until the next label.
    offsets = np.arange(len(dates))
    drought_values[~valid] = np.interp(offsets[~valid], offsets[valid], drought_values[valid])
    heat_values = inputs["daily_hwi"][np.searchsorted(inputs["daily_time"], dates), column]
    selected = dates.astype("datetime64[M]").astype(np.int64) % 12 + 1 == case["month"]
    table = Counter(zip(heat_values[selected] >= 0, drought_values[selected] <= -1, strict=True))
    n = sum(table.values())
    n_heat = sum(count for (heat, _), count in table.items() if heat)
    n_drought = sum(count for (_, drought), count in table.items() if drought)
    n_joint = table[True, True]
    assert (n, n_heat, n_drought, n_joint) == (case["n_days"], case["n_heat"], case["n_drought"], case["n_joint"])
    assert n == (3000 if case["month"] == 6 else 3100)
    assert case["joint_probability"] == pytest.approx(float(Fraction(n_joint, n)), rel=0, abs=1e-12)
    assert case["independent_probability"] == pytest.approx(
        float(Fraction(n_heat * n_drought, n * n)), rel=0, abs=1e-12
    )
    assert case["lmf"] == pytest.approx(float(Fraction(n_joint * n, n_heat * n_drought)), rel=0, abs=1e-12)


def test_author_comparators_are_inclusive_and_do_not_mutate_inputs(preparation: dict[str, Any]) -> None:
    hwi = np.array([0.0, -0.1, 2.0, -2.0])
    spei = np.array([-1.0, -2.0, 0.0, 1.0])
    heat_copy, drought_copy = hwi.copy(), spei.copy()
    counts = preparation["author_counts"](hwi, spei)
    assert (counts["n_heat"], counts["n_drought"], counts["n_joint"], counts["lmf"]) == (2, 2, 1, 1.0)
    np.testing.assert_array_equal(hwi, heat_copy)
    np.testing.assert_array_equal(spei, drought_copy)


def test_author_zero_joint_result_is_undefined_not_the_future_module_zero(preparation: dict[str, Any]) -> None:
    counts = preparation["author_counts"](np.array([-1.0, 1.0]), np.array([-2.0, 0.0]))
    assert (counts["n_heat"], counts["n_drought"], counts["n_joint"], counts["lmf"]) == (1, 1, 0, None)


@pytest.mark.parametrize(
    "hwi,spei", [([], []), ([1.0], [np.nan]), ([[1.0]], [[1.0]]), ([np.inf], [-2.0]), ([1.0], [-2.0, 0.0])]
)
def test_author_calculator_rejects_unsupported_reference_inputs(
    preparation: dict[str, Any], hwi: Any, spei: Any
) -> None:
    with pytest.raises(ValueError):
        preparation["author_counts"](np.asarray(hwi), np.asarray(spei))


def test_padding_precedes_gap_interpolation(preparation: dict[str, Any]) -> None:
    dates = np.array(["2021-01-31", "2021-02-28", "2021-03-31"], dtype="datetime64[ns]")
    daily = preparation["daily_spei"](dates, np.array([[-2.0], [np.nan], [0.0]]))
    assert daily.sel(time="2021-02-27").item() == -2.0
    assert daily.sel(time="2021-02-28").item() == -1.9375
    # Timestamp-based interpolation can differ from the exact fraction by one rounding step.
    assert daily.sel(time="2021-03-30").item() == pytest.approx(-0.0625, rel=0, abs=1e-15)


@pytest.mark.parametrize("download_date", ["not-a-date", "2026-02-30"])
def test_preparation_rejects_invalid_download_dates(
    preparation: dict[str, Any], tmp_path: Path, download_date: str
) -> None:
    output = tmp_path / "fixture"
    with pytest.raises(ValueError):
        preparation["prepare"](tmp_path / "missing-source", output, download_date)
    assert not output.exists()


def test_preparation_rejects_future_download_dates(preparation: dict[str, Any], tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="Download date cannot be in the future"):
        preparation["prepare"](tmp_path / "missing-source", output, tomorrow)
    assert not output.exists()


def test_preparation_refuses_unpinned_sources(preparation: dict[str, Any], tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    (source / "SPEI12.nc").write_bytes(b"not the public file")
    output = tmp_path / "fixture"
    with pytest.raises(ValueError, match="Unexpected source checksum: SPEI12.nc"):
        preparation["prepare"](source, output, "2026-10-17")
    assert not output.exists()


def test_preparation_never_overwrites_existing_output(preparation: dict[str, Any], tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing fixture")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        preparation["prepare"](tmp_path / "missing-source", output, "2026-10-17")
    assert marker.read_text() == "existing fixture"


@pytest.mark.parametrize("destination", ["../outside", "../allowed-sibling/fixture", "."])
def test_preparation_rejects_output_outside_root(
    preparation: dict[str, Any], tmp_path: Path, destination: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "allowed"
    monkeypatch.setitem(preparation["prepare"].__globals__, "OUTPUT_ROOT", root)
    with pytest.raises(ValueError, match="Output directory must be below"):
        preparation["prepare"](tmp_path / "missing-source", root / destination, "2026-10-17")
    assert not root.exists()


def test_preparation_rejects_absolute_output_outside_root(
    preparation: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "allowed"
    monkeypatch.setitem(preparation["prepare"].__globals__, "OUTPUT_ROOT", root)
    with pytest.raises(ValueError, match="Output directory must be below"):
        preparation["prepare"](tmp_path / "missing-source", tmp_path / "outside", "2026-10-17")
    assert not root.exists()
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("symlink_at_root", [False, True])
def test_preparation_rejects_symlink_escape(
    preparation: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symlink_at_root: bool
) -> None:
    root = tmp_path / "allowed"
    outside = tmp_path / "outside"
    outside.mkdir()
    if symlink_at_root:
        link = root
    else:
        root.mkdir()
        link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setitem(preparation["prepare"].__globals__, "OUTPUT_ROOT", root)
    with pytest.raises(ValueError, match="Output directory must be below"):
        preparation["prepare"](tmp_path / "missing-source", link / "fixture", "2026-10-17")
    assert list(outside.iterdir()) == []


def test_preparation_refuses_dangling_output_symlink(preparation: dict[str, Any], tmp_path: Path) -> None:
    target = tmp_path / "missing"
    output = tmp_path / "fixture"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        preparation["prepare"](tmp_path / "missing-source", output, "2026-10-17")
    assert output.is_symlink()
    assert not target.exists()


@pytest.fixture
def synthetic_sources(preparation: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Small stand-ins for the 3 GB public files, with the hash guard satisfied by name."""
    import pandas as pd
    import xarray as xr

    source = tmp_path / "sources"
    source.mkdir()
    rng = np.random.default_rng(0)
    grid = {"lat": [56.0, 62.0, 67.0], "lon": [14.0, 15.0, 20.0]}
    monthly = pd.date_range("1922-01-31", periods=1200, freq="ME")
    daily = pd.date_range("1922-01-01", "2021-12-31", freq="D")
    for name, variable, time in (("SPEI12.nc", "SPEI", monthly), ("heatwave_SMHI.nc", "heatwave index", daily)):
        values = rng.normal(size=(len(time), 3, 3))
        dataset = xr.Dataset({variable: (("time", "lat", "lon"), values)}, coords={"time": time, **grid})
        dataset.to_netcdf(source / name, engine="h5netcdf")
    pinned = preparation["SOURCES"]
    real_checksum = preparation["checksum"]

    def checksum(path: Path) -> str:
        return str(pinned[path.name]["sha256"]) if path.name in pinned else str(real_checksum(path))

    monkeypatch.setitem(preparation["prepare"].__globals__, "checksum", checksum)
    return source


@pytest.mark.parametrize("compact_date", [False, True])
def test_preparation_publishes_a_complete_fixture_without_leftovers(
    preparation: dict[str, Any], synthetic_sources: Path, tmp_path: Path, compact_date: bool
) -> None:
    if compact_date and sys.version_info < (3, 11):
        pytest.skip("Python 3.10 does not accept compact ISO dates.")
    output = tmp_path / "fixture"
    today = date.today()
    download_date = today.strftime("%Y%m%d") if compact_date else today.isoformat()
    preparation["prepare"](synthetic_sources, output, download_date)
    assert sorted(path.name for path in output.iterdir()) == ["inputs.npz", "provenance.json", "reference.json"]
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["download_date"] == today.isoformat()
    assert hashlib.sha256((output / "reference.json").read_bytes()).hexdigest() == provenance["checksum_sha256"]
    assert len(json.loads((output / "reference.json").read_text())["cases"]) == 9
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fixture", "sources"]


def test_publication_refuses_a_destination_created_during_preparation(
    preparation: dict[str, Any], synthetic_sources: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A destination created after the early check must not be replaced."""
    output = tmp_path / "fixture"
    real_savez = np.savez_compressed

    def late_destination(*args: Any, **kwargs: Any) -> None:
        output.mkdir()
        (output / "keep.txt").write_text("another operator")
        real_savez(*args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", late_destination)
    download_date = date.today().isoformat()
    with pytest.raises(FileExistsError):
        preparation["prepare"](synthetic_sources, output, download_date)
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fixture", "sources"]


def test_failed_write_leaves_neither_output_nor_staging(
    preparation: dict[str, Any], synthetic_sources: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disk_full(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    real_savez = np.savez_compressed
    monkeypatch.setattr(np, "savez_compressed", disk_full)
    output = tmp_path / "fixture"
    with pytest.raises(OSError, match="disk full"):
        preparation["prepare"](synthetic_sources, output, "2026-10-17")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sources"]
    # the next attempt is not blocked by the failed one
    monkeypatch.setattr(np, "savez_compressed", real_savez)
    preparation["prepare"](synthetic_sources, output, "2026-10-17")
    assert output.is_dir()

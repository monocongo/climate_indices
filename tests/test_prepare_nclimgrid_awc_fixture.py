"""Offline regressions for resumable POLARIS fixture downloads."""

import importlib.util
import io
import sys
from pathlib import Path
from urllib.error import URLError

import pytest


@pytest.fixture
def fixture_generator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_nclimgrid_awc_fixture.py"
    spec = importlib.util.spec_from_file_location("prepare_nclimgrid_awc_fixture_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("readable", [False, True])
def test_full_partial_tile_is_validated_before_request(fixture_generator, tmp_path, monkeypatch, readable):
    module = fixture_generator
    target = tmp_path / "tile.tif"
    partial = target.with_suffix(".tif.part")
    partial.write_bytes(b"bad!")
    monkeypatch.setattr(module, "_remote_size", lambda url: 4)
    monkeypatch.setattr(module, "_tail_readable", lambda path: readable or path.read_bytes() == b"good")
    requests = []

    def urlopen(request, **kwargs):
        assert request.get_header("Range") is None
        requests.append(request)
        return io.BytesIO(b"good")

    monkeypatch.setattr(module, "urlopen", urlopen)
    assert module._fetch_tile("http://example.invalid/tile.tif", target) == "downloaded"
    assert target.read_bytes() == (b"bad!" if readable else b"good")
    assert len(requests) == (0 if readable else 1)
    assert not partial.exists()


@pytest.mark.parametrize(
    ("partial_bytes", "expected", "range_header"),
    [(b"go", 4, "bytes=2-"), (b"oversized", 4, None), (b"", None, None)],
)
def test_tile_download_resumes_or_restarts_partial(
    fixture_generator, tmp_path, monkeypatch, partial_bytes, expected, range_header
):
    module = fixture_generator
    target = tmp_path / "tile.tif"
    partial = target.with_suffix(".tif.part")
    partial.write_bytes(partial_bytes)
    monkeypatch.setattr(module, "_remote_size", lambda url: expected)
    monkeypatch.setattr(module, "_tail_readable", lambda path: path.read_bytes() == b"good")

    def urlopen(request, **kwargs):
        assert request.get_header("Range") == range_header
        return io.BytesIO(b"od" if range_header else b"good")

    monkeypatch.setattr(module, "urlopen", urlopen)
    assert module._fetch_tile("http://example.invalid/tile.tif", target) == "downloaded"
    assert target.read_bytes() == b"good"
    assert not partial.exists()


@pytest.mark.parametrize("permanent", [False, True])
def test_tile_download_retries_url_errors(fixture_generator, tmp_path, monkeypatch, permanent):
    module = fixture_generator
    target = tmp_path / "tile.tif"
    monkeypatch.setattr(module, "DOWNLOAD_ATTEMPTS", 2)
    monkeypatch.setattr(module, "_remote_size", lambda url: 4)
    monkeypatch.setattr(module, "_tail_readable", lambda path: True)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    requests = []

    def urlopen(request, **kwargs):
        requests.append(request)
        if permanent or len(requests) == 1:
            raise URLError("connection reset")
        return io.BytesIO(b"good")

    monkeypatch.setattr(module, "urlopen", urlopen)
    if permanent:
        with pytest.raises(SystemExit, match="after 2 attempts"):
            module._fetch_tile("http://example.invalid/tile.tif", target)
        assert not target.exists()
    else:
        assert module._fetch_tile("http://example.invalid/tile.tif", target) == "downloaded"
        assert target.read_bytes() == b"good"
    assert len(requests) == 2


def test_complete_tile_is_reused(fixture_generator, tmp_path, monkeypatch):
    module = fixture_generator
    target = tmp_path / "tile.tif"
    target.write_bytes(b"good")
    monkeypatch.setattr(module, "_remote_size", lambda url: 4)
    monkeypatch.setattr(module, "_tail_readable", lambda path: True)
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: pytest.fail("cached tile must not be downloaded"))
    assert module._fetch_tile("http://example.invalid/tile.tif", target) == "cached"


@pytest.mark.parametrize("dry_run", [False, True])
def test_fixture_main_returns_success_without_building_in_dry_run(fixture_generator, tmp_path, monkeypatch, dry_run):
    module = fixture_generator
    monkeypatch.setattr(
        sys, "argv", [module.__file__, "--cache-dir", str(tmp_path)] + (["--dry-run"] if dry_run else [])
    )
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    monkeypatch.setattr(module, "_subset_climate", lambda cache: None)
    monkeypatch.setattr(module, "_tile_targets", lambda climate, tiles: [])
    monkeypatch.setattr(module, "_describe", lambda climate, targets: "test grid")
    monkeypatch.setattr(module, "_sha256", lambda path: "test digest")
    calls = []
    monkeypatch.setattr(module, "_ensure_tiles", lambda targets: calls.append("tiles"))

    def build(*args):
        calls.append("fixture")
        return tmp_path / "fixture.nc"

    monkeypatch.setattr(module, "_build_fixture", build)
    assert module.main() == 0
    assert calls == ([] if dry_run else ["tiles", "fixture"])

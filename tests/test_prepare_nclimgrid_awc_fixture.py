"""Offline regressions for resumable POLARIS fixture downloads."""

import importlib.util
import io
from pathlib import Path

import pytest


@pytest.mark.parametrize("readable", [False, True])
def test_full_partial_tile_is_validated_before_request(tmp_path, monkeypatch, readable):
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_nclimgrid_awc_fixture.py"
    spec = importlib.util.spec_from_file_location("prepare_nclimgrid_awc_fixture_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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

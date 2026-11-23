"""Offline tests for the Hickman fixture preparation script."""

import hashlib
import io
import json
from http.client import HTTPMessage
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

import pytest

from scripts import prepare_flood_hickman_fixture as prepare


def test_redirect_handler_refuses_other_origins() -> None:
    handler = prepare._ApprovedOriginRedirectHandler()
    original = Request(prepare.URL)
    with io.BytesIO() as response:
        accepted = handler.redirect_request(original, response, 302, "Found", HTTPMessage(), prepare.URL)
        with pytest.raises(URLError, match="refusing redirect"):
            handler.redirect_request(original, response, 302, "Found", HTTPMessage(), "https://example.invalid/")
    assert accepted is not None and accepted.full_url == prepare.URL


def test_publish_preserves_fixture_until_both_csvs_match(tmp_path: Path) -> None:
    target = tmp_path / "hickman_prcp_1960_1996.csv"
    target.write_bytes(b"old\n")
    table = tmp_path / "table5.csv"
    candidate = b"precipitation\n"
    checksum = hashlib.sha256(candidate + b"table\n").hexdigest()
    (tmp_path / "provenance.json").write_text(json.dumps({"checksum_sha256": checksum}))

    with pytest.raises(FileNotFoundError, match=r"table5\.csv"):
        prepare._publish_fixture(tmp_path, candidate)
    assert target.read_bytes() == b"old\n"

    table.write_bytes(b"changed\n")
    with pytest.raises(SystemExit, match="fixture checksum mismatch"):
        prepare._publish_fixture(tmp_path, candidate)
    assert target.read_bytes() == b"old\n"

    table.write_bytes(b"table\n")
    assert prepare._publish_fixture(tmp_path, candidate) == checksum
    assert target.read_bytes() == candidate

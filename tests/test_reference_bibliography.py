"""The reference-paper bibliography matches its recorded publisher verification (REFIMPL-05)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

RESEARCH = Path(__file__).resolve().parent.parent / "docs" / "research" / "reference-papers"
BIB = RESEARCH / "references.bib"
VERIFICATION = RESEARCH / "references-verification.json"
VERIFIER = RESEARCH / "verify_references.py"


@pytest.fixture(scope="module")
def bibliography() -> dict[str, dict[str, str]]:
    spec = importlib.util.spec_from_file_location("verify_references", VERIFIER)
    assert spec is not None
    assert spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = cast(Any, module).parse_bib
    return cast(dict[str, dict[str, str]], parser(BIB.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def verification() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(VERIFICATION.read_text(encoding="utf-8")))


def _pages(value: str) -> str:
    return value.replace("--", "-").replace(" ", "")


def test_every_entry_is_verified(bibliography: dict[str, dict[str, str]], verification: dict[str, Any]) -> None:
    """Each entry has a verification record, and no record is orphaned."""
    records = verification["entries"]
    assert sorted(bibliography) == sorted(records), "bibliography and verification records differ"


def test_entries_carry_the_fields_a_reader_needs(bibliography: dict[str, dict[str, str]]) -> None:
    """An entry states its authors, title and year."""
    for key, entry in bibliography.items():
        for field in ("author", "title", "year"):
            assert entry.get(field), f"{key}: missing {field}"


def test_doi_entries_match_their_crossref_record(
    bibliography: dict[str, dict[str, str]], verification: dict[str, Any]
) -> None:
    """A DOI entry agrees with the Crossref record on DOI, year, volume, number and pages."""
    for key, entry in bibliography.items():
        doi = entry.get("doi")
        record = verification["entries"][key]
        if not doi:
            continue
        assert record["check"] == "crossref", f"{key}: DOI without a Crossref record"
        assert doi.lower() == record["doi"].lower(), f"{key}: DOI differs from the record"
        printed = record.get("published_print") or record.get("issued") or ""
        assert printed.startswith(entry["year"]), f"{key}: year differs from the record ({printed})"
        for bib_field, record_field in (("volume", "volume"), ("number", "issue")):
            recorded = record.get(record_field)
            if entry.get(bib_field) and recorded:
                assert entry[bib_field] == str(recorded), f"{key}: {bib_field} differs from the record"
        recorded_pages = record.get("pages")
        if entry.get("pages") and recorded_pages:
            assert _pages(entry["pages"]) == _pages(str(recorded_pages)), f"{key}: pages differ from the record"


def test_entries_without_a_doi_name_the_record_that_verified_them(
    bibliography: dict[str, dict[str, str]], verification: dict[str, Any]
) -> None:
    """A DOI-less entry was checked against a named publisher, agency or library page."""
    for key, entry in bibliography.items():
        record = verification["entries"][key]
        if entry.get("doi"):
            continue
        assert record["check"] == "record", f"{key}: no DOI and no recorded publisher record"
        assert str(record["record_url"]).startswith("https://"), f"{key}: record_url is not https"
        assert len(str(record["record_states"])) > 40, f"{key}: the record does not say what it states"
        assert record["checked"], f"{key}: no check date"

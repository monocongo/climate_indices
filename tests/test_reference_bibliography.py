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

# The verified set behind REFIMPL-05. Adding or removing a reference means adding or
# removing its verification record too, and updating this tripwire.
EXPECTED_KEYS = frozenset(
    {
        "abatzoglou2013",
        "allen2025",
        "alley1984",
        "begueria2014",
        "bourgault2023",
        "carbone2018",
        "dai2011",
        "farahmand2015",
        "ficklin2015",
        "guttman1999",
        "heddinghaus1991",
        "hobbins2016",
        "hoffmann2020",
        "huntington2017",
        "jacobi2013",
        "karl1986",
        "laimighofer2022",
        "mckee1993",
        "palmer1965",
        "pieper2020",
        "rhee2007",
        "spade2020",
        "stagge2015",
        "vanderschrier2013",
        "vicente-serrano2010",
        "vonk2023comparison",
        "vonk2025",
        "vose2014",
        "vremec2024",
        "wells2004",
        "wmo1090",
    }
)


@pytest.fixture(scope="module")
def verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_references", VERIFIER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bibliography(verifier: ModuleType) -> dict[str, dict[str, str]]:
    parse_bib = cast(Any, verifier).parse_bib
    return cast(dict[str, dict[str, str]], parse_bib(BIB.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def verification() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(VERIFICATION.read_text(encoding="utf-8")))


def test_the_bibliography_is_the_verified_set(bibliography: dict[str, dict[str, str]]) -> None:
    """The bibliography holds exactly the entries this ticket verified."""
    assert frozenset(bibliography) == EXPECTED_KEYS


def test_every_entry_has_a_verification_record(
    bibliography: dict[str, dict[str, str]], verification: dict[str, Any]
) -> None:
    """No entry is unverified, and no record is orphaned."""
    assert sorted(bibliography) == sorted(verification["entries"])


def test_no_entry_disagrees_with_its_record(
    bibliography: dict[str, dict[str, str]], verification: dict[str, Any], verifier: ModuleType
) -> None:
    """Each entry agrees with its Crossref record, or names the page that verified it."""
    check = cast(Any, verifier).mismatches
    problems = [
        problem for key, entry in bibliography.items() for problem in check(key, entry, verification["entries"][key])
    ]
    assert not problems, "\n".join(problems)


def test_a_doi_less_entry_records_what_the_page_states(verification: dict[str, Any]) -> None:
    """A record-checked entry names an https page and says what that page states."""
    records = [record for record in verification["entries"].values() if record["check"] == "record"]
    assert records, "no DOI-less entry is verified against a named page"
    for record in records:
        assert record["record_url"].startswith("https://")
        assert len(record["record_states"]) > 40
        assert record["year"]
        assert record["checked"]


@pytest.mark.parametrize(
    ("key", "field", "value", "expected"),
    [
        ("spade2020", "title", "Unrelated text about something else", "title"),
        ("guttman1999", "journal", "Nature", "journal or booktitle"),
        ("guttman1999", "author", "Guttmann, Nathaniel B.", "first author"),
        ("allen2025", "author", "Allender, Sam and Otero, Noelia", "first author"),
        ("carbone2018", "author", "Carbonell, Gregory J. and Lu, Junyu", "first author"),
    ],
)
def test_a_substantive_change_is_reported(
    bibliography: dict[str, dict[str, str]],
    verification: dict[str, Any],
    verifier: ModuleType,
    key: str,
    field: str,
    value: str,
    expected: str,
) -> None:
    """A wrong title, container or first-author family name no longer certifies."""
    entry = {**bibliography[key], field: value}
    problems = cast(Any, verifier).mismatches(key, entry, verification["entries"][key])
    assert any(expected in problem for problem in problems), problems

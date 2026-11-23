"""The reference-paper bibliography matches its recorded publisher verification (REFIMPL-05)."""

import importlib.util
import io
import json
import urllib.request
from datetime import date
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
    assert frozenset(bibliography) == EXPECTED_KEYS, (
        "adding or removing a reference means updating EXPECTED_KEYS in this test file"
    )


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


def test_the_verification_file_names_its_source(verification: dict[str, Any]) -> None:
    """The record file states where its data came from and when it was checked."""
    assert verification["source"].startswith("Crossref REST API")
    assert date.fromisoformat(verification["issued"])


@pytest.mark.parametrize(
    ("key", "side", "field", "value", "expected"),
    [
        ("spade2020", "entry", "title", "Unrelated text about something else", "title"),
        ("guttman1999", "entry", "journal", "Nature", "journal or booktitle"),
        ("guttman1999", "entry", "author", "Guttmann, Nathaniel B.", "first author"),
        ("allen2025", "entry", "author", "Allender, Sam and Otero, Noelia", "first author"),
        ("carbone2018", "entry", "author", "Carbonell, Gregory J. and Lu, Junyu", "first author"),
        ("guttman1999", "entry", "doi", "", "no doi field"),
        ("guttman1999", "entry", "doi", "10.1111/not-the-doi", "doi"),
        ("guttman1999", "entry", "year", "1998", "year"),
        ("guttman1999", "entry", "volume", "36", "volume"),
        ("guttman1999", "entry", "number", "3", "number"),
        ("guttman1999", "entry", "pages", "311--323", "pages"),
        ("mckee1993", "entry", "pages", "999--999", "pages"),
        ("mckee1993", "entry", "pages", "2", "pages"),
        ("wmo1090", "entry", "number", "77", "number"),
        ("wmo1090", "entry", "number", "90", "number"),
        ("mckee1993", "entry", "author", "", "needs author"),
        ("mckee1993", "entry", "doi", "10.1/foo", "not a Crossref check"),
        ("mckee1993", "record", "record_url", "http://example.invalid/a", "not https"),
        ("mckee1993", "record", "record_states", "too short", "does not say"),
        ("mckee1993", "record", "year", "1994", "year"),
        ("guttman1999", "record", "check", "openalex", "unknown check"),
    ],
)
def test_a_wrong_field_no_longer_certifies(
    bibliography: dict[str, dict[str, str]],
    verification: dict[str, Any],
    verifier: ModuleType,
    key: str,
    side: str,
    field: str,
    value: str,
    expected: str,
) -> None:
    """Every field or record value the check compares is reported when it disagrees."""
    entry = dict(bibliography[key])
    record = dict(verification["entries"][key])
    target = {"entry": entry, "record": record}[side]
    target[field] = value
    problems = cast(Any, verifier).mismatches(key, entry, record)
    assert any(expected in problem for problem in problems), problems


def test_a_cited_year_override_needs_a_note(
    bibliography: dict[str, dict[str, str]], verification: dict[str, Any], verifier: ModuleType
) -> None:
    """A manual cited_year with no explanatory note does not certify."""
    record = {**verification["entries"]["allen2025"]}
    record.pop("note", None)
    problems = cast(Any, verifier).mismatches("allen2025", bibliography["allen2025"], record)
    assert any("cited_year" in problem for problem in problems), problems


@pytest.mark.parametrize(
    "text",
    [
        "@article{a,\n  title = {One},\n@article{b,\n  title = {Two},\n}\n",
        "@article{a,\n  title = {One},\n",
    ],
    ids=["next-entry-starts-first", "end-of-file"],
)
def test_an_unclosed_entry_is_rejected(verifier: ModuleType, text: str) -> None:
    """An entry with no closing brace is a malformed bibliography."""
    parse_bib = cast(Any, verifier).parse_bib
    with pytest.raises(ValueError, match="not closed"):
        parse_bib(text)


def test_an_unbalanced_field_value_is_rejected(verifier: ModuleType) -> None:
    """A field value with unbalanced braces is not valid BibTeX, and _plain would hide it."""
    title = "Major over- and underestimation of drought found in {NOAA's climate divisional {SPI} dataset"
    parse_bib = cast(Any, verifier).parse_bib
    with pytest.raises(ValueError, match="unbalanced braces"):
        parse_bib(f"@article{{spade2020,\n  title   = {{{title}}},\n}}\n")


def test_a_duplicate_field_is_rejected(verifier: ModuleType) -> None:
    """BibTeX rejects a repeated field, so the parser must not keep only the last value."""
    parse_bib = cast(Any, verifier).parse_bib
    with pytest.raises(ValueError, match="duplicate field year"):
        parse_bib("@article{a,\n  year = {1999},\n  Year = {1900},\n}\n")


def test_an_unparsed_line_is_rejected(verifier: ModuleType) -> None:
    """A line that is neither an entry, a field nor a closing brace is malformed."""
    parse_bib = cast(Any, verifier).parse_bib
    with pytest.raises(ValueError, match="unparsed line"):
        parse_bib("@article{a,\n  title = {One},\n  stray text\n}\n")


def test_a_superscript_exponent_still_counts(verifier: ModuleType) -> None:
    """A superscript that follows a word holds content, so it must not be dropped."""
    plain = cast(Any, verifier)._plain
    assert plain("An R<sup>2</sup> model") == plain("An R2 model")
    assert plain("An R<sup>2</sup> model") != plain("An R model")
    assert plain("A title                     <sup>1</sup>") == plain("A title")


def test_a_dotless_i_spelling_matches(verifier: ModuleType) -> None:
    """The dotless-letter accent spelling is the same name as the plain accented one."""
    plain = cast(Any, verifier)._plain
    assert plain(r"Beguer{\'{\i}}a") == plain("Beguería")
    assert plain(r"x\index{y}") == plain("xy")


class _Opener:
    """A stand-in for the module opener that returns one fixed reply."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def open(self, *args: Any, **kwargs: Any) -> io.BytesIO:
        return io.BytesIO(self.payload)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"[]", "failed"),
        (b'{"message": null}', "returned no work record"),
        (b'{"message": {}}', "returned no work record"),
    ],
    ids=["list", "null", "no-doi"],
)
def test_a_malformed_crossref_response_is_a_lookup_failure(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch, payload: bytes, reason: str
) -> None:
    """A response that is not a work record raises the error the retry path handles."""
    monkeypatch.setattr(verifier, "OPENER", _Opener(payload))
    with pytest.raises(RuntimeError, match=reason):
        cast(Any, verifier).fetch("10.1175/JAMC-D-19-0272.1")


def test_a_response_larger_than_the_cap_is_a_lookup_failure(
    verifier: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reply body is read up to a bound, so a huge body cannot be held in memory."""
    payload = b'{"message": {"DOI": "' + b"x" * 200 + b'"}}'
    monkeypatch.setattr(verifier, "OPENER", _Opener(payload))
    monkeypatch.setattr(verifier, "MAX_RESPONSE", 64)
    with pytest.raises(RuntimeError, match="failed"):
        cast(Any, verifier).fetch("10.1175/JAMC-D-19-0272.1")


def test_a_redirect_off_crossref_is_refused(verifier: ModuleType) -> None:
    """An off-endpoint redirect is refused before the request goes to the new host."""
    handler = cast(Any, verifier)._CrossrefRedirect()
    with pytest.raises(OSError, match="redirected a lookup"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://example.invalid/works/10.1/x")


def test_a_redirect_within_crossref_is_followed(verifier: ModuleType) -> None:
    """A redirect that stays on the Crossref endpoint is allowed."""
    handler = cast(Any, verifier)._CrossrefRedirect()
    request = urllib.request.Request("https://api.crossref.org/works/10.1/x")
    moved = handler.redirect_request(request, None, 302, "Found", {}, "https://api.crossref.org/works/10.1/y")
    assert moved is not None

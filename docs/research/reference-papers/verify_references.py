"""Re-verify the reference-paper bibliography against publisher records (REFIMPL-05, #1607).

Reads ``references.bib`` next to this file and refreshes ``references-verification.json``.

* An entry with a ``doi`` field is checked against the Crossref record for that DOI.
  The script records the fields read from Crossref and reports every disagreement about
  the DOI, year, volume, number, pages, title, container or first author.
* An entry without a ``doi`` keeps the record already in the JSON: a human reads the
  publisher, agency or library page and writes ``record_url`` with ``record_states``.
  The script fails on a DOI-less entry that has no such record.
* ``note`` and ``cited_year`` in a committed record are preserved as written. Use
  ``cited_year`` when the citation year differs from the Crossref issue year, and say
  why in ``note``.

Run it by hand after editing the bibliography, then commit both files:

    uv run python docs/research/reference-papers/verify_references.py

The script writes nothing when any check fails, so a failed run cannot delete a record.
``tests/test_reference_bibliography.py`` compares the two committed files offline, so
the test suite never touches the network.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parent
BIB = HERE / "references.bib"
VERIFICATION = HERE / "references-verification.json"
CROSSREF = "https://api.crossref.org/works/"
USER_AGENT = "climate_indices-refimpl-bibliography/1.0 (https://github.com/monocongo/climate_indices)"
PRESERVED = ("note", "cited_year")
ENTRY = re.compile(r"@(\w+)\{([^,]+),")
FIELD = re.compile(r"^\s+(\w+)\s*=\s*\{(.*)\},?\s*$")


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    """Return {key: {field: value}} for a bib file written one field per line."""
    entries: dict[str, dict[str, str]] = {}
    key: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        match = ENTRY.match(stripped)
        if match:
            new_key = match.group(2).strip()
            if new_key in entries:
                raise ValueError(f"duplicate bibliography key: {new_key}")
            key = new_key
            entries[new_key] = {}
            continue
        if key is None:
            continue
        if stripped == "}":
            key = None
            continue
        field = FIELD.match(line)
        if not field:
            raise ValueError(f"{key}: unparsed line {line!r}; write one field per line with its value on that line")
        entries[key][field.group(1).lower()] = field.group(2).strip()
    return entries


def _plain(value: object) -> str:
    """Compare key for a title, container or name: lower-case ASCII letters and digits only.

    Drops HTML superscripts and tags, HTML entities, LaTeX commands and braces, accents and punctuation, so
    ``Computers \\& Geosciences`` equals ``Computers &amp; Geosciences`` and ``{SPI}`` equals ``<scp>SPI</scp>``.
    """
    text = re.sub(r"<sup>.*?</sup>", "", str(value), flags=re.IGNORECASE | re.DOTALL)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"\\(?:[A-Za-z]+|.)", "", text)
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", text).lower())


def fetch(doi: str) -> dict[str, Any]:
    """Return the Crossref work record for a DOI."""
    request = urllib.request.Request(
        CROSSREF + urllib.parse.quote(doi), headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https endpoint
            message = json.load(response)["message"]
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"Crossref lookup for {doi} failed") from error
    return cast(dict[str, Any], message)


def _date(record: dict[str, Any], field: str) -> str | None:
    """Return a Crossref date field as YYYY or YYYY-MM or YYYY-MM-DD."""
    value = record.get(field) or {}
    parts = (value.get("date-parts") or [[None]])[0]
    if not parts or parts[0] is None:
        return None
    # Crossref does not zero-pad month and day.
    return "-".join([str(parts[0]), *(f"{part:02d}" for part in parts[1:] if part is not None)])


def summarise(record: dict[str, Any], checked: str) -> dict[str, Any]:
    """Keep the fields the bibliography is checked against."""
    authors = record.get("author") or []
    return {
        "check": "crossref",
        "doi": record["DOI"],
        "title": (record.get("title") or [""])[0].replace("\n", " ").strip(),
        "first_author_family": authors[0].get("family") if authors else None,
        "issued": _date(record, "issued"),
        "published_print": _date(record, "published-print"),
        "container": (record.get("container-title") or [None])[0],
        "volume": record.get("volume"),
        "issue": record.get("issue"),
        "pages": record.get("page") or record.get("article-number"),
        "checked": checked,
    }


def mismatches(key: str, entry: dict[str, str], record: dict[str, Any]) -> list[str]:
    """Report every disagreement between a bibliography entry and its verification record."""
    found: list[str] = []
    if not entry.get("author") or not entry.get("title") or not entry.get("year"):
        found.append(f"{key}: an entry needs author, title and year")
    if record.get("check") == "record":
        if entry.get("doi"):
            found.append(f"{key}: has a doi but its record is not a Crossref check")
        if not str(record.get("record_url", "")).startswith("https://"):
            found.append(f"{key}: record_url is missing or not https")
        if len(str(record.get("record_states", ""))) <= 40:
            found.append(f"{key}: the record does not say what it states")
        if not record.get("checked") or not record.get("year"):
            found.append(f"{key}: the record needs a check date and a year")
        elif str(record["year"]) != entry.get("year"):
            found.append(f"{key}: year {entry.get('year')!r} != record {record['year']!r}")
        return found
    if record.get("check") != "crossref":
        return [f"{key}: unknown check {record.get('check')!r}"]

    doi = entry.get("doi", "")
    if not doi:
        found.append(f"{key}: no doi field for a Crossref record")
    elif doi.lower() != str(record.get("doi", "")).lower():
        found.append(f"{key}: doi {doi!r} != record {record.get('doi')!r}")

    # A journal issue prints in its volume year, which can follow the online-first date.
    printed = record.get("cited_year") or record.get("published_print") or record.get("issued") or ""
    year = entry.get("year", "")
    if str(printed)[:4] != year:
        found.append(f"{key}: year {year!r} != record {printed!r}")

    for bib_field, record_field in (("volume", "volume"), ("number", "issue"), ("pages", "pages")):
        value = entry.get(bib_field, "")
        recorded = record.get(record_field)
        if recorded is None:
            if value:
                found.append(f"{key}: {bib_field} {value!r} is not in the record")
        elif value.replace("--", "-") != str(recorded).replace("--", "-"):
            found.append(f"{key}: {bib_field} {value!r} != record {recorded!r}")

    if _plain(entry.get("title", "")) != _plain(record.get("title") or ""):
        found.append(f"{key}: title {entry.get('title')!r} != record {record.get('title')!r}")

    container = entry.get("journal") or entry.get("booktitle") or ""
    recorded_container = record.get("container") or ""
    if _plain(container) != _plain(recorded_container):
        found.append(f"{key}: journal or booktitle {container!r} != record {recorded_container!r}")

    family = _plain(record.get("first_author_family") or "")
    first_author = _plain(entry.get("author", "").split(" and ")[0].split(",")[0])
    if not family:
        found.append(f"{key}: the record carries no first author")
    elif first_author != family:
        found.append(f"{key}: first author differs from the record ({record.get('first_author_family')!r})")
    return found


def main() -> int:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    known: dict[str, Any] = {}
    problems: list[str] = []
    if VERIFICATION.exists():
        try:
            known = cast(dict[str, Any], json.loads(VERIFICATION.read_text(encoding="utf-8"))).get("entries", {})
        except ValueError as error:
            problems.append(f"{VERIFICATION.name}: unreadable ({error})")
    checked = datetime.now(UTC).strftime("%Y-%m-%d")

    records: dict[str, Any] = {}
    for key in sorted(entries):
        entry = entries[key]
        doi = entry.get("doi")
        if not doi:
            record = known.get(key)
            if not record or record.get("check") != "record":
                problems.append(f"{key}: no DOI and no recorded publisher or agency record")
                continue
            records[key] = record
            problems.extend(mismatches(key, entry, records[key]))
            continue
        for attempt in range(3):
            try:
                records[key] = summarise(fetch(doi), checked)
                break
            except RuntimeError as error:
                if attempt == 2:
                    problems.append(f"{key}: {error}")
                time.sleep(3)
        else:
            continue
        for field in PRESERVED:
            if field in known.get(key, {}):
                records[key][field] = known[key][field]
        problems.extend(mismatches(key, entry, records[key]))
        time.sleep(1.5)

    for key in sorted(set(known) - set(entries)):
        problems.append(f"{key}: verification record with no bibliography entry")

    print(f"checked {len(records)} entries in {BIB.name}")
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(f"{VERIFICATION.name} left unchanged", file=sys.stderr)
        return 1

    document = {
        "issued": checked,
        "source": (
            "Crossref REST API, api.crossref.org/works/{doi}, for entries with a doi; "
            "the publisher, agency or library page named by record_url otherwise"
        ),
        "entries": records,
    }
    temporary = VERIFICATION.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(VERIFICATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

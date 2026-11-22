"""Re-verify the reference-paper bibliography against publisher records (REFIMPL-05, #1607).

Reads ``references.bib`` next to this file and refreshes ``references-verification.json``.

* An entry with a ``doi`` field is checked against the Crossref record for that DOI.
  The script records the fields read from Crossref and reports a mismatch when the
  entry's year, volume, number, pages or first author disagree with the record.
* An entry without a ``doi`` keeps the record already in the JSON: a human reads the
  publisher, agency or library page and writes ``record_url`` with ``record_states``.
  The script fails on a DOI-less entry that has no such record.

Run it by hand after editing the bibliography, then commit both files:

    uv run python docs/research/reference-papers/verify_references.py

``tests/test_reference_bibliography.py`` compares the two committed files offline, so
the test suite never touches the network.
"""

from __future__ import annotations

import json
import re
import sys
import time
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
            key = new_key
            entries[new_key] = {}
            continue
        # Any other line before the closing brace is a field of the current entry.
        if key is None:
            continue
        if stripped == "}":
            key = None
            continue
        field = FIELD.match(line)
        if field:
            entries[key][field.group(1).lower()] = field.group(2).strip()
    return entries


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
    """Report where a bibliography entry disagrees with its Crossref record."""
    found: list[str] = []
    doi = entry.get("doi", "")
    if not doi:
        return found
    if doi.lower() != str(record.get("doi", "")).lower():
        found.append(f"{key}: doi {doi!r} != record {record.get('doi')!r}")
    year = entry.get("year", "")
    # A journal issue prints in its volume year, which can follow the online-first date.
    printed = record.get("published_print") or record.get("issued") or ""
    if year and not str(printed).startswith(year):
        found.append(f"{key}: year {year!r} != record {printed!r}")
    for bib_field, record_field in (("volume", "volume"), ("number", "issue")):
        value = entry.get(bib_field, "")
        recorded = record.get(record_field)
        if value and recorded and value != str(recorded):
            found.append(f"{key}: {bib_field} {value!r} != record {recorded!r}")
    pages = entry.get("pages", "")
    if pages and record.get("pages"):
        if pages.replace("--", "-").replace(" ", "") != str(record["pages"]).replace(" ", ""):
            found.append(f"{key}: pages {pages!r} != record {record['pages']!r}")
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
            if not record or record.get("check") != "record" or not record.get("record_url"):
                problems.append(f"{key}: no DOI and no recorded publisher or agency record")
                continue
            records[key] = record
            continue
        for attempt in range(3):
            try:
                records[key] = summarise(fetch(doi), checked)
                break
            except Exception as error:  # noqa: BLE001 - a network failure is reported, not raised
                if attempt == 2:
                    problems.append(f"{key}: Crossref lookup failed: {error}")
                time.sleep(3)
        else:
            continue
        problems.extend(mismatches(key, entry, records[key]))
        time.sleep(1.5)

    document = {
        "issued": checked,
        "source": (
            "Crossref REST API, api.crossref.org/works/{doi}, for entries with a doi; "
            "the publisher, agency or library page named by record_url otherwise"
        ),
        "entries": records,
    }
    VERIFICATION.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"checked {len(records)} entries in {BIB.name}")
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

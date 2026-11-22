"""Validate method-matrix.json against method_spec.schema.json and render its tables into method-matrix.md.

    uv run python docs/research/reference-papers/render_method_matrix.py          # rewrite the tables
    uv run python docs/research/reference-papers/render_method_matrix.py --check  # fail if the file is stale

The text between the BEGIN and END markers of method-matrix.md is generated; edit the JSON, not the tables.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]

HERE = Path(__file__).resolve().parent
BEGIN = "<!-- BEGIN GENERATED: {key} -->"
END = "<!-- END GENERATED: {key} -->"
SUMMARY_WIDTH = 110
STATUS = {"verified": "", "unverified": "**unverified.** ", "not_applicable": ""}
RESERVED = {"counts", "aligned"}


def _esc(text: str) -> str:
    return text.replace("<", "&lt;").replace("|", "\\|").replace("\n", " ")


def _cites(cites: list[dict[str, Any]]) -> str:
    return "; ".join(f"`{_esc(c['file'])}:{_esc(c['lines'])}`" for c in cites)


def _row(axis: str, cell: dict[str, Any]) -> str:
    position = STATUS[cell["status"]] + _esc(cell["value"])
    if cell["status"] == "not_applicable":
        position = f"n/a: {position}"
    for option in cell.get("options", []):
        opt = f"{_esc(option['value'])} (`{_esc(option['how'])}`)"
        if option.get("citations"):
            opt += f" [{_cites(option['citations'])}]"
        position += f"<br>Option: {opt}"
    if cell.get("note"):
        position += f"<br>Note: {_esc(cell['note'])}"
    evidence = _cites(cell.get("citations", []))
    if cell.get("basis"):
        evidence = (evidence + "<br>" if evidence else "") + f"Basis: {_esc(cell['basis'])}"
    return f"| {axis.replace('_', ' ')} | {position} | {evidence} |"


def render_summary(doc: dict[str, Any], key: str) -> list[str]:
    """Defaults at a glance: axes as rows, implementations as columns; evidence is in the per-package tables."""
    entries = doc["families"][key]["entries"]
    names = [
        f"{_esc(doc['implementations'][e['implementation']]['name'])}"
        f"<br>`{_esc(e['entry_point'].split(' ')[0].split('(')[0])}`"
        for e in entries
    ]
    out = ["| Axis | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(entries)]
    for axis in doc["axes"]:
        cells = []
        for entry in entries:
            cell = entry["cells"][axis]
            text = (
                cell["value"]
                if len(cell["value"]) <= SUMMARY_WIDTH
                else cell["value"][: SUMMARY_WIDTH - 3].rstrip() + "..."
            )
            mark = " (unverified)" if cell["status"] == "unverified" else ""
            cells.append(_esc(text) + mark)
        out.append(f"| {axis.replace('_', ' ')} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def render_family(doc: dict[str, Any], key: str) -> str:
    family = doc["families"][key]
    out: list[str] = ["#### Defaults at a glance", "", *render_summary(doc, key), "#### Per-package evidence", ""]
    for entry in family["entries"]:
        impl = doc["implementations"][entry["implementation"]]
        out.extend(
            [
                f"##### {impl['name']}: {entry['entry_point']}",
                "",
                f"Read at: {impl['ref']}." + ("" if impl["source_read"] else " **Source not read.**"),
            ]
        )
        if impl.get("note"):
            out.append(f"{impl['note']}")
        out.extend(["", "| Axis | Position (default; options follow) | Evidence |", "| --- | --- | --- |"])
        out.extend(_row(axis, entry["cells"][axis]) for axis in doc["axes"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_aligned(doc: dict[str, Any]) -> str:
    out: list[str] = []
    for cfg in doc["aligned_configurations"]:
        out.extend([f"### `{cfg['id']}`: {cfg['title']}", "", "| Axis | Position |", "| --- | --- |"])
        out.extend(f"| {a.replace('_', ' ')} | {_esc(v)} |" for a, v in cfg["positions"].items())
        out.extend(["", "Reachable in:", ""])
        out.extend(f"- `{i['implementation']}`: {_esc(i['how'])}" for i in cfg["implementations"])
        out.append("")
        if cfg["independent_triples"]:
            triples = "; ".join("{" + ", ".join(f"`{x}`" for x in t) + "}" for t in cfg["independent_triples"])
            out.append(f"Mutually independent triples (lineage.json classes): {triples}.")
        else:
            out.append(f"**No independent triple.** {cfg['absent']}")
        out.extend(["", "Residual differences after alignment:", ""])
        out.extend(f"- {_esc(r)}" for r in cfg["residual_axes"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_counts(doc: dict[str, Any]) -> str:
    out = ["| Family | Entries | Verified | Unverified | Not applicable |", "| --- | --- | --- | --- | --- |"]
    for fam in doc["families"].values():
        counts = {"verified": 0, "unverified": 0, "not_applicable": 0}
        for e in fam["entries"]:
            for c in e["cells"].values():
                counts[c["status"]] += 1
        out.append(
            f"| {_esc(fam['title'])} | {len(fam['entries'])} | {counts['verified']} | {counts['unverified']} | {counts['not_applicable']} |"
        )
    return "\n".join(out) + "\n"


def sections(doc: dict[str, Any]) -> dict[str, str]:
    """Generated sections by marker key. A family key must be usable as a marker and must not shadow a fixed section."""
    out: dict[str, str] = {}
    for key in doc["families"]:
        if key in RESERVED or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise SystemExit(f"family key {key!r} is reserved or cannot be a generated marker key")
        out[key] = render_family(doc, key)
    out["counts"] = render_counts(doc)
    out["aligned"] = render_aligned(doc)
    return out


def check_markers(text: str, keys: list[str]) -> None:
    """Every generated section has one ordered marker pair, none overlaps another, and none is orphaned."""
    found = set(re.findall(r"<!-- BEGIN GENERATED: (\w+) -->", text))
    if found != set(keys):
        raise SystemExit(f"method-matrix.md generates {sorted(found)}, but the document defines {sorted(keys)}")
    spans = []
    for key in keys:
        begin, end = BEGIN.format(key=key), END.format(key=key)
        if text.count(begin) != 1 or text.count(end) != 1 or text.index(end) < text.index(begin):
            raise SystemExit(f"markers for {key!r} must appear once each, BEGIN before END, in method-matrix.md")
        spans.append((text.index(begin), text.index(end), key))
    spans.sort()
    for previous, current in zip(spans, spans[1:], strict=False):
        if current[0] < previous[1]:
            raise SystemExit(f"generated sections {previous[2]!r} and {current[2]!r} overlap in method-matrix.md")


def splice(text: str, key: str, body: str) -> str:
    begin, end = BEGIN.format(key=key), END.format(key=key)
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(end) < text.index(begin):
        raise SystemExit(f"markers for {key!r} must appear once each, BEGIN before END, in method-matrix.md")
    head, _, rest = text.partition(begin)
    _, _, tail = rest.partition(end)
    return f"{head}{begin}\n\n{body}\n{end}{tail}"


def check_references(doc: dict[str, Any]) -> None:
    """Cross-references the schema cannot express."""
    impls = doc["implementations"]
    for key, fam in doc["families"].items():
        for name in (e["implementation"] for e in fam["entries"]):
            if name not in impls:
                raise SystemExit(f"{key}: unknown implementation {name!r}")
    for cfg in doc["aligned_configurations"]:
        _check_alignment(doc, cfg)


def _check_alignment(doc: dict[str, Any], cfg: dict[str, Any]) -> None:
    if cfg["family"] not in doc["families"]:
        raise SystemExit(f"{cfg['id']}: unknown family {cfg['family']!r}")
    members = {i["implementation"] for i in cfg["implementations"]}
    family = {e["implementation"] for e in doc["families"][cfg["family"]]["entries"]}
    if not members <= family:
        raise SystemExit(f"{cfg['id']}: {sorted(members - family)} not an entry of family {cfg['family']!r}")
    for triple in cfg["independent_triples"]:
        if len(set(triple)) != 3 or not set(triple) <= members:
            raise SystemExit(f"{cfg['id']}: triple {triple} is not three distinct members of its implementations")


def _read(path: Path) -> str:
    if path.is_symlink():
        raise SystemExit(f"refusing to read through a symlink: {path}")
    return path.read_text(encoding="utf-8")


def main(argv: list[str]) -> int:
    unknown = sorted(set(argv) - {"--check"})
    if unknown:
        raise SystemExit(f"unknown argument(s): {', '.join(unknown)}; usage: render_method_matrix.py [--check]")
    try:
        schema = json.loads(_read(HERE / "method_spec.schema.json"))
        doc = json.loads(_read(HERE / "method-matrix.json"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read the method matrix: {error}") from error
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(doc, schema)
    check_references(doc)
    path = HERE / "method-matrix.md"
    original = _read(path)
    bodies = sections(doc)
    check_markers(original, list(bodies))
    text = original
    for key, body in bodies.items():
        text = splice(text, key, body)
    check_markers(text, list(bodies))
    if "--check" in argv:
        if text != original:
            print("method-matrix.md is stale; run render_method_matrix.py", file=sys.stderr)
            return 1
        return 0
    if path.is_symlink():
        raise SystemExit(f"refusing to write through a symlink: {path}")
    path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Validate method-matrix.json against method_spec.schema.json and render its tables into method-matrix.md.

    uv run python docs/research/reference-papers/render_method_matrix.py          # rewrite the tables
    uv run python docs/research/reference-papers/render_method_matrix.py --check  # fail if the file is stale

The text between the BEGIN and END markers of method-matrix.md is generated; edit the JSON, not the tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
BEGIN = "<!-- BEGIN GENERATED: {key} -->"
END = "<!-- END GENERATED: {key} -->"
STATUS = {"verified": "", "unverified": "**unverified.** ", "not_applicable": ""}


def _esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _cites(cites: list[dict]) -> str:
    return "; ".join(f"`{c['file']}:{c['lines']}`" for c in cites)


def _row(axis: str, cell: dict) -> str:
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


def render_family(doc: dict, key: str) -> str:
    family = doc["families"][key]
    out: list[str] = []
    for entry in family["entries"]:
        impl = doc["implementations"][entry["implementation"]]
        out.append(f"#### {impl['name']}: {entry['entry_point']}")
        out.append("")
        out.append(f"Read at: {impl['ref']}." + ("" if impl["source_read"] else " **Source not read.**"))
        if impl.get("note"):
            out.append(f"{impl['note']}")
        out.append("")
        out.append("| Axis | Position (default; options follow) | Evidence |")
        out.append("| --- | --- | --- |")
        out.extend(_row(axis, entry["cells"][axis]) for axis in doc["axes"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_aligned(doc: dict) -> str:
    out: list[str] = []
    for cfg in doc["aligned_configurations"]:
        out.append(f"#### `{cfg['id']}`: {cfg['title']}")
        out.append("")
        out.append("| Axis | Position |")
        out.append("| --- | --- |")
        out.extend(f"| {a.replace('_', ' ')} | {_esc(v)} |" for a, v in cfg["positions"].items())
        out.append("")
        out.append("Reachable in:")
        out.append("")
        out.extend(f"- `{i['implementation']}`: {_esc(i['how'])}" for i in cfg["implementations"])
        out.append("")
        if cfg["independent_triples"]:
            triples = "; ".join("{" + ", ".join(f"`{x}`" for x in t) + "}" for t in cfg["independent_triples"])
            out.append(f"Mutually independent triples (lineage.json classes): {triples}.")
        else:
            out.append(f"**No independent triple.** {cfg['absent']}")
        out.append("")
        out.append("Residual differences after alignment:")
        out.append("")
        out.extend(f"- {_esc(r)}" for r in cfg["residual_axes"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_counts(doc: dict) -> str:
    out = ["| Family | Entries | Verified | Unverified | Not applicable |", "| --- | --- | --- | --- | --- |"]
    for fam in doc["families"].values():
        counts = {"verified": 0, "unverified": 0, "not_applicable": 0}
        for e in fam["entries"]:
            for c in e["cells"].values():
                counts[c["status"]] += 1
        out.append(
            f"| {fam['title']} | {len(fam['entries'])} | {counts['verified']} | {counts['unverified']} | {counts['not_applicable']} |"
        )
    return "\n".join(out) + "\n"


def sections(doc: dict) -> dict[str, str]:
    return {
        "counts": render_counts(doc),
        "spi": render_family(doc, "spi"),
        "spei": render_family(doc, "spei"),
        "non_parametric": render_family(doc, "non_parametric"),
        "aligned": render_aligned(doc),
    }


def splice(text: str, key: str, body: str) -> str:
    begin, end = BEGIN.format(key=key), END.format(key=key)
    if begin not in text or end not in text:
        raise SystemExit(f"markers for {key!r} missing from method-matrix.md")
    head, _, rest = text.partition(begin)
    _, _, tail = rest.partition(end)
    return f"{head}{begin}\n\n{body}\n{end}{tail}"


def main(argv: list[str]) -> int:
    schema = json.loads((HERE / "method_spec.schema.json").read_text(encoding="utf-8"))
    doc = json.loads((HERE / "method-matrix.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(doc, schema)
    for fam in doc["families"].values():
        for entry in fam["entries"]:
            if entry["implementation"] not in doc["implementations"]:
                raise SystemExit(f"unknown implementation {entry['implementation']!r}")
    path = HERE / "method-matrix.md"
    original = path.read_text(encoding="utf-8")
    text = original
    for key, body in sections(doc).items():
        text = splice(text, key, body)
    if "--check" in argv:
        if text != original:
            print("method-matrix.md is stale; run render_method_matrix.py", file=sys.stderr)
            return 1
        return 0
    path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""The reference-paper method matrix validates against its schema and its tables are current (REFIMPL-09)."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import jsonschema  # type: ignore[import-untyped]
import pytest

RESEARCH = Path(__file__).resolve().parent.parent / "docs" / "research" / "reference-papers"
RENDERER = RESEARCH / "render_method_matrix.py"


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((RESEARCH / "method_spec.schema.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((RESEARCH / "method-matrix.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_method_matrix", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy(doc: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(doc)))


def _copy_renderer_set(dest: Path) -> None:
    for name in (RENDERER.name, "method_spec.schema.json", "method-matrix.json", "method-matrix.md"):
        (dest / name).write_text((RESEARCH / name).read_text(encoding="utf-8"), encoding="utf-8")


def _first_cell(doc: dict[str, Any], status: str) -> dict[str, Any]:
    return next(
        c for f in doc["families"].values() for e in f["entries"] for c in e["cells"].values() if c["status"] == status
    )


def test_schema_is_valid(schema: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)


def test_matrix_validates_against_schema(schema: dict[str, Any], matrix: dict[str, Any]) -> None:
    jsonschema.validate(matrix, schema)


def test_verified_cell_requires_citation(schema: dict[str, Any], matrix: dict[str, Any]) -> None:
    broken = _copy(matrix)
    del _first_cell(broken, "verified")["citations"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)


def test_unverified_cell_requires_non_empty_basis(schema: dict[str, Any], matrix: dict[str, Any]) -> None:
    for basis in (None, ""):
        broken = _copy(matrix)
        cell = _first_cell(broken, "unverified")
        if basis is None:
            del cell["basis"]
        else:
            cell["basis"] = basis
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)


def test_absent_is_required_only_without_triples(schema: dict[str, Any], matrix: dict[str, Any]) -> None:
    broken = _copy(matrix)
    del next(c for c in broken["aligned_configurations"] if not c["independent_triples"])["absent"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)
    broken = _copy(matrix)
    next(c for c in broken["aligned_configurations"] if c["independent_triples"])["absent"] = "not allowed here"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)


def test_schema_requires_every_axis_and_a_non_empty_alignment(schema: dict[str, Any], matrix: dict[str, Any]) -> None:
    broken = _copy(matrix)
    broken["axes"] = broken["axes"][:-1]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)
    broken = _copy(matrix)
    broken["aligned_configurations"][0]["positions"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)
    broken = _copy(matrix)
    broken["aligned_configurations"][0]["implementations"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)


def test_axes_match_schema_enum(schema: dict[str, Any], matrix: dict[str, Any]) -> None:
    """The renderer uses doc["axes"] as table order; the schema must agree with it and with every entry."""
    assert matrix["axes"] == schema["$defs"]["axis_id"]["enum"]
    assert set(schema["$defs"]["entry"]["properties"]["cells"]["required"]) == set(matrix["axes"])


def test_renderer_rejects_inconsistent_references(renderer: ModuleType, matrix: dict[str, Any]) -> None:
    broken = _copy(matrix)
    broken["families"]["spi"]["entries"][0]["implementation"] = "not_a_real_key"
    with pytest.raises(SystemExit):
        renderer.check_references(broken)

    broken = _copy(matrix)
    broken["aligned_configurations"][0]["family"] = "no_such_family"
    with pytest.raises(SystemExit):
        renderer.check_references(broken)

    broken = _copy(matrix)
    config = next(c for c in broken["aligned_configurations"] if c["family"] == "spei")
    config["implementations"].append({"implementation": "r_sci", "how": "not an entry of this family"})
    with pytest.raises(SystemExit):
        renderer.check_references(broken)

    broken = _copy(matrix)
    config = next(c for c in broken["aligned_configurations"] if c["independent_triples"])
    members = [i["implementation"] for i in config["implementations"]]
    outsider = next(k for k in broken["implementations"] if k not in members)
    for triple in ([members[0], members[0], members[1]], [members[0], members[1], outsider]):
        config["independent_triples"] = [triple]
        with pytest.raises(SystemExit):
            renderer.check_references(broken)


def test_renderer_rejects_reserved_or_unmarkerable_family_keys(renderer: ModuleType, matrix: dict[str, Any]) -> None:
    """A family key must not shadow the counts or aligned section, nor fall outside the marker syntax."""
    for key in ("counts", "aligned", "spi-daily"):
        broken = _copy(matrix)
        broken["families"][key] = broken["families"].pop("spi")
        with pytest.raises(SystemExit):
            renderer.sections(broken)


def test_renderer_checks_references_and_tables_are_current() -> None:
    """The renderer validates cross-references and exits non-zero when method-matrix.md is stale."""
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--check"], capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode == 0, result.stderr


def test_renderer_reports_stale_tables_without_writing(tmp_path: Path) -> None:
    _copy_renderer_set(tmp_path)
    path = tmp_path / "method-matrix.md"
    original = path.read_text(encoding="utf-8")
    stale = original.replace("| SPI | 13 |", "| SPI | 99 |", 1)
    assert stale != original
    path.write_text(stale, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(tmp_path / RENDERER.name), "--check"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 1
    assert path.read_text(encoding="utf-8") == stale


def test_renderer_rewrites_stale_tables(tmp_path: Path) -> None:
    """The write path replaces stale tables with the rendered ones."""
    _copy_renderer_set(tmp_path)
    path = tmp_path / "method-matrix.md"
    original = path.read_text(encoding="utf-8")
    stale = original.replace("| SPI | 13 |", "| SPI | 99 |", 1)
    assert stale != original
    path.write_text(stale, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(tmp_path / RENDERER.name)], capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert path.read_text(encoding="utf-8") == original


def test_renderer_cli_rejects_unknown_arguments(tmp_path: Path) -> None:
    _copy_renderer_set(tmp_path)
    path = tmp_path / "method-matrix.md"
    original = path.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(tmp_path / RENDERER.name), "--chek"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode != 0
    assert "unknown argument" in result.stderr
    assert path.read_text(encoding="utf-8") == original


def test_renderer_cli_rejects_an_invalid_document_without_writing(tmp_path: Path) -> None:
    _copy_renderer_set(tmp_path)
    data = tmp_path / "method-matrix.json"
    doc = json.loads(data.read_text(encoding="utf-8"))
    doc["families"]["spi"]["entries"][0]["implementation"] = "not_a_real_key"
    data.write_text(json.dumps(doc), encoding="utf-8")
    path = tmp_path / "method-matrix.md"
    original = path.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(tmp_path / RENDERER.name)], capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode != 0
    assert "unknown implementation" in result.stderr
    assert path.read_text(encoding="utf-8") == original


def test_renderer_cli_rejects_markers_in_generated_content(tmp_path: Path) -> None:
    """A metadata value carrying a marker cannot be rendered again, so the renderer refuses to write it."""
    _copy_renderer_set(tmp_path)
    data = tmp_path / "method-matrix.json"
    doc = json.loads(data.read_text(encoding="utf-8"))
    doc["implementations"]["climate_indices"]["note"] = "<!-- END GENERATED: spi -->"
    data.write_text(json.dumps(doc), encoding="utf-8")
    path = tmp_path / "method-matrix.md"
    original = path.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(tmp_path / RENDERER.name)], capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode != 0
    assert "markers for 'spi'" in result.stderr
    assert path.read_text(encoding="utf-8") == original


def test_renderer_rejects_symlinked_inputs(tmp_path: Path) -> None:
    _copy_renderer_set(tmp_path)
    path = tmp_path / "method-matrix.json"
    target = tmp_path / "target.json"
    target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    result = subprocess.run(
        [sys.executable, str(tmp_path / RENDERER.name), "--check"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode != 0
    assert "symlink" in result.stderr


def test_escaping_keeps_data_visible(renderer: ModuleType) -> None:
    assert renderer._esc("a | b & <c>") == "a \\| b & &lt;c>"
    assert renderer._cites([{"file": "a|b\nc.py", "lines": "1"}]) == "`a\\|b c.py:1`"


def test_generated_sections_cover_data_families(renderer: ModuleType, matrix: dict[str, Any]) -> None:
    text = (RESEARCH / "method-matrix.md").read_text(encoding="utf-8")
    assert set(re.findall(r"<!-- BEGIN GENERATED: (\w+) -->", text)) == set(renderer.sections(matrix))
    assert "`dim_spi_n`" in text and "`dim_spi_n(p,`" not in text


def test_marker_check_rejects_overlapping_or_orphaned_sections(renderer: ModuleType) -> None:
    pair = {k: (renderer.BEGIN.format(key=k), renderer.END.format(key=k)) for k in "abc"}
    renderer.check_markers(f"{pair['a'][0]}x{pair['a'][1]}{pair['b'][0]}y{pair['b'][1]}", ["a", "b"])
    nested = f"{pair['a'][0]}{pair['b'][0]}y{pair['b'][1]}x{pair['a'][1]}"
    orphan = f"{pair['a'][0]}x{pair['a'][1]}{pair['c'][0]}z{pair['c'][1]}"
    missing = f"{pair['a'][0]}x{pair['a'][1]}"
    for text in (nested, orphan, missing):
        with pytest.raises(SystemExit):
            renderer.check_markers(text, ["a", "b"])


def test_splice_rejects_misordered_or_duplicate_markers(renderer: ModuleType) -> None:
    begin, end = renderer.BEGIN.format(key="k"), renderer.END.format(key="k")
    assert renderer.splice(f"a{begin}old{end}z", "k", "new") == f"a{begin}\n\nnew\n{end}z"
    for text in (f"a{end}b{begin}c", f"{begin}{end}{begin}{end}", "no markers"):
        with pytest.raises(SystemExit):
            renderer.splice(text, "k", "x")

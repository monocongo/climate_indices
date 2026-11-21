"""The reference-paper method matrix validates against its schema and its tables are current (REFIMPL-09)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

RESEARCH = Path(__file__).resolve().parent.parent / "docs" / "research" / "reference-papers"
RENDERER = RESEARCH / "render_method_matrix.py"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads((RESEARCH / "method_spec.schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def matrix() -> dict:
    return json.loads((RESEARCH / "method-matrix.json").read_text(encoding="utf-8"))


def test_schema_is_valid(schema: dict) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)


def test_matrix_validates_against_schema(schema: dict, matrix: dict) -> None:
    jsonschema.validate(matrix, schema)


def test_empty_independent_triples_require_absent(schema: dict, matrix: dict) -> None:
    broken = json.loads(json.dumps(matrix))
    config = next(c for c in broken["aligned_configurations"] if not c["independent_triples"])
    del config["absent"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)


def test_renderer_checks_references_and_tables_are_current() -> None:
    """The renderer validates cross-references and exits non-zero when method-matrix.md is stale."""
    result = subprocess.run([sys.executable, str(RENDERER), "--check"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_splice_rejects_misordered_or_duplicate_markers() -> None:
    spec = importlib.util.spec_from_file_location("render_method_matrix", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    begin, end = module.BEGIN.format(key="k"), module.END.format(key="k")
    assert module.splice(f"a{begin}old{end}z", "k", "new") == f"a{begin}\n\nnew\n{end}z"
    for text in (f"a{end}b{begin}c", f"{begin}{end}{begin}{end}", "no markers"):
        with pytest.raises(SystemExit):
            module.splice(text, "k", "x")

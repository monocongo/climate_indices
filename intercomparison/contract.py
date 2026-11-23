"""Adapter contract of the intercomparison harness (REFIMPL-08, issue 1610).

Standard library only, so every Python adapter can import it from its own environment and a
non-Python adapter can implement the same two JSON documents.

An adapter is a subprocess: ``<adapter> run <request.json> <out_dir>``.

* The *request* names the index, the method positions to apply, and each input series by path
  (relative to the request file) and SHA-256. The adapter verifies the hash before computing.
* The *result* (``result.json`` in ``out_dir``) records the comparator and its version, the
  effective method the comparator actually applied, software versions, wall time, warnings, and
  each output series by path and SHA-256.
* A series is a two-column CSV, ``time,value``. Values use ``repr(float)``, the shortest string
  that round-trips a binary64 exactly; a missing value is ``NaN``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_REQUEST_KEYS = ("schema_version", "study", "unit", "index", "positions", "inputs")
_RESULT_KEYS = (
    "schema_version",
    "comparator",
    "version",
    "effective_method",
    "software",
    "platform",
    "wall_time_s",
    "warnings",
    "outputs",
)


class ContractError(ValueError):
    """A request or result violates the adapter contract."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_series(path: Path, times: Sequence[str], values: Sequence[float]) -> str:
    """Write ``time,value`` rows and return the file's SHA-256."""
    if len(times) != len(values):
        raise ContractError(f"{len(times)} times but {len(values)} values")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("time", "value"))
        writer.writerows((t, repr(float(v))) for t, v in zip(times, values, strict=True))
    return sha256_file(path)


def read_series(path: Path) -> tuple[list[str], list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["time", "value"]:
            raise ContractError(f"{path}: header is not 'time,value'")
        rows = list(reader)
    return [r[0] for r in rows], [float(r[1]) for r in rows]


def read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return loaded


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require(document: Mapping[str, Any], keys: Sequence[str], what: str) -> None:
    missing = [k for k in keys if k not in document]
    if missing:
        raise ContractError(f"{what} is missing {missing}")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"{what} has schema_version {document['schema_version']!r}, expected {SCHEMA_VERSION}")


def _verify(base: Path, entry: Mapping[str, Any], what: str) -> Path:
    path = base / entry["path"]
    if not path.is_file():
        raise ContractError(f"{what}: {path} does not exist")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ContractError(f"{what}: {path} has SHA-256 {actual}, the request or result says {entry['sha256']}")
    return path


def read_request(path: Path) -> dict[str, Any]:
    """Load a request and verify every input's SHA-256 against the file on disk."""
    request = read_json(path)
    _require(request, _REQUEST_KEYS, "request")
    for name, entry in request["inputs"].items():
        _verify(path.parent, entry, f"input {name!r}")
    return request


def read_result(path: Path) -> dict[str, Any]:
    """Load a result and verify every output's SHA-256 against the file on disk."""
    result = read_json(path)
    _require(result, _RESULT_KEYS, "result")
    for name, entry in result["outputs"].items():
        _verify(path.parent, entry, f"output {name!r}")
    return result


def diff_method(requested: Mapping[str, Any], effective: Mapping[str, Any]) -> list[str]:
    """Return one warning per requested position the comparator did not apply as requested.

    A position the comparator does not report counts as a difference: silence is not agreement.
    """
    return [
        f"position {key!r}: requested {want!r}, applied {effective.get(key, '<not reported>')!r}"
        for key, want in requested.items()
        if effective.get(key) != want
    ]


def run_adapter(
    comparator: str,
    compute: Callable[[dict[str, Any], dict[str, tuple[list[str], list[float]]]], dict[str, Any]],
    argv: Sequence[str],
) -> int:
    """Shared command line: ``run <request.json> <out_dir>``.

    ``compute(request, inputs)`` receives the verified request and each input series, and returns
    ``version``, ``effective_method``, ``software`` (versions), optional ``warnings``, and
    ``series``: ``{output name: (times, values)}``.
    """
    if len(argv) != 3 or argv[0] != "run":
        print(f"usage: {comparator} run <request.json> <out_dir>", file=sys.stderr)
        return 2
    request_path, out_dir = Path(argv[1]), Path(argv[2])
    request = read_request(request_path)
    inputs = {name: read_series(request_path.parent / entry["path"]) for name, entry in request["inputs"].items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    computed = compute(request, inputs)
    wall = time.perf_counter() - started
    outputs = {}
    for name, (times, values) in computed["series"].items():
        outputs[name] = {"path": f"{name}.csv", "sha256": write_series(out_dir / f"{name}.csv", times, values)}
    write_json(
        out_dir / "result.json",
        {
            "schema_version": SCHEMA_VERSION,
            "comparator": comparator,
            "version": computed["version"],
            "effective_method": computed["effective_method"],
            "software": computed["software"],
            "platform": platform.platform(),
            "wall_time_s": wall,
            "warnings": list(computed.get("warnings", [])),
            "outputs": outputs,
        },
    )
    return 0

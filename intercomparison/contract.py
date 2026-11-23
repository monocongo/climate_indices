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
  that round-trips a binary64 exactly; a missing value is ``nan`` (``NaN`` is read as well).
* A ``path`` must stay inside the directory of the document that names it.

The ``NOSONAR (S8707)`` marks below are audited: a path read or written here came from the command
line through ``path_arg``, was built by the harness, or is a document path that ``entry_path``
confined to the document's directory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
import warnings
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
_MISSING = object()


class ContractError(ValueError):
    """A request or result violates the adapter contract."""


def path_arg(value: str) -> Path:
    """Resolve a command-line path, refusing a relative one that escapes the working directory.

    Same guard as benchmarks/cli_multiprocessing.py (pythonsecurity:S8707); absolute paths are
    the documented usage and pass through after normalisation.
    """
    resolved = os.path.realpath(value)
    base = os.path.realpath(os.getcwd())
    if not os.path.isabs(value) and resolved != base and not resolved.startswith(base + os.sep):
        raise SystemExit(f"{value!r} is outside the working directory")
    return Path(resolved)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()  # NOSONAR (S8707)


def write_series(path: Path, times: Sequence[str], values: Sequence[float]) -> str:
    """Write ``time,value`` rows and return the file's SHA-256."""
    if len(times) != len(values):
        raise ContractError(f"{len(times)} times but {len(values)} values")
    with path.open("w", newline="", encoding="utf-8") as handle:  # NOSONAR (S8707)
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("time", "value"))
        writer.writerows((t, repr(float(v))) for t, v in zip(times, values, strict=True))
    return sha256_file(path)


def read_series(path: Path, sha256: str | None = None) -> tuple[list[str], list[float]]:
    """Read a series. With ``sha256``, the bytes parsed are the bytes hashed (no verify-then-reopen gap)."""
    data = path.read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ContractError(f"{path}: SHA-256 differs from the {sha256} the document names")
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    if not rows or rows[0] != ["time", "value"]:
        raise ContractError(f"{path}: header is not 'time,value'")
    body = rows[1:]
    if any(len(row) != 2 for row in body):
        raise ContractError(f"{path}: every row must be exactly 'time,value'")
    try:
        return [row[0] for row in body], [float(row[1]) for row in body]
    except ValueError as error:
        raise ContractError(f"{path}: {error}") from error


def read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))  # NOSONAR (S8707)
    if not isinstance(loaded, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return loaded


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    # the paths are the operator's own: normalised by path_arg, or built by the harness
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")  # NOSONAR (S2083, S8707)


def _require(document: Mapping[str, Any], keys: Sequence[str], what: str) -> None:
    missing = [k for k in keys if k not in document]
    if missing:
        raise ContractError(f"{what} is missing {missing}")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"{what} has schema_version {document['schema_version']!r}, expected {SCHEMA_VERSION}")


def _require_mappings(document: Mapping[str, Any], keys: Sequence[str], what: str) -> None:
    for key in keys:
        if not isinstance(document[key], Mapping):
            raise ContractError(f"{what}: {key!r} must be an object")


def _require_metadata(result: Mapping[str, Any]) -> None:
    """The metadata a manifest copies must satisfy the manifest's own schema (manifest.schema.json)."""
    for key in ("comparator", "version", "platform"):
        if not isinstance(result[key], str) or not result[key]:
            raise ContractError(f"result: {key!r} must be a non-empty string")
    wall = result["wall_time_s"]
    if isinstance(wall, bool) or not isinstance(wall, int | float) or not wall >= 0:  # NaN fails `>= 0`
        raise ContractError("result: 'wall_time_s' must be a number of at least 0")
    messages = result["warnings"]
    if not isinstance(messages, list) or not all(isinstance(m, str) for m in messages):
        raise ContractError("result: 'warnings' must be a list of strings")
    if not all(isinstance(v, str) for v in result["software"].values()):
        raise ContractError("result: every 'software' version must be a string")


def entry_path(base: Path, entry: Any, what: str) -> tuple[Path, str]:
    """Resolve ``{"path", "sha256"}`` against ``base``; the path may not leave ``base``."""
    if not isinstance(entry, Mapping) or not all(isinstance(entry.get(k), str) for k in ("path", "sha256")):
        raise ContractError(f"{what}: expected an object with string 'path' and 'sha256'")
    root = base.resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ContractError(f"{what}: {entry['path']!r} is outside {base}")
    return path, entry["sha256"]


def _verify(base: Path, entry: Any, what: str) -> Path:
    path, expected = entry_path(base, entry, what)
    if not path.is_file():
        raise ContractError(f"{what}: {path} does not exist")
    actual = sha256_file(path)
    if actual != expected:
        raise ContractError(f"{what}: {path} has SHA-256 {actual}, the document says {expected}")
    return path


def read_entry_series(base: Path, entry: Any, what: str) -> tuple[list[str], list[float]]:
    """Read the series a ``{"path", "sha256"}`` entry names, hashing the bytes that are parsed."""
    path, expected = entry_path(base, entry, what)
    if not path.is_file():
        raise ContractError(f"{what}: {path} does not exist")
    return read_series(path, expected)


def read_request(path: Path) -> dict[str, Any]:
    """Load a request and verify every input's SHA-256 against the file on disk."""
    request = read_json(path)
    _require(request, _REQUEST_KEYS, "request")
    _require_mappings(request, ("positions", "inputs"), "request")
    for name, entry in request["inputs"].items():
        _verify(path.parent, entry, f"input {name!r}")
    return request


def read_result(path: Path) -> dict[str, Any]:
    """Load a result and verify every output's SHA-256 against the file on disk."""
    result = read_json(path)
    _require(result, _RESULT_KEYS, "result")
    _require_mappings(result, ("effective_method", "software", "outputs"), "result")
    _require_metadata(result)
    for name, entry in result["outputs"].items():
        _verify(path.parent, entry, f"output {name!r}")
    return result


def check_store_index(manifest: Mapping[str, Any]) -> None:
    """Require one array entry per row of the store, with ``store_index`` a permutation of the rows.

    manifest.schema.json cannot say this: duplicate indices would alias one row and an
    out-of-range index has no row, so the metadata could not be mapped to the Zarr store.
    """
    rows = manifest["store"]["shape"][0]
    indices = sorted(entry["store_index"] for entry in manifest["arrays"])
    if indices != list(range(rows)):
        raise ContractError(f"manifest: store_index values {indices} are not the {rows} rows of the store")


def diff_method(requested: Mapping[str, Any], effective: Mapping[str, Any]) -> list[str]:
    """Return one warning per requested position the comparator did not apply as requested.

    A position the comparator does not report counts as a difference: silence is not agreement.
    Types must match too, so ``3``, ``3.0`` and ``True`` are three different values.
    """
    warnings = []
    for key, want in requested.items():
        got = effective.get(key, _MISSING)
        if got is _MISSING or type(got) is not type(want) or got != want:
            shown = "<not reported>" if got is _MISSING else got
            warnings.append(f"position {key!r}: requested {want!r}, applied {shown!r}")
    return warnings


def require_consecutive_months(times: Sequence[str], *, january_start: bool = False) -> None:
    """Raise unless ``times`` are consecutive ``YYYY-MM`` labels (and start in January if asked)."""
    try:
        year, month = int(times[0][:4]), int(times[0][5:7])
    except (IndexError, ValueError) as error:
        raise ContractError(f"times do not start with a YYYY-MM label: {list(times[:1])!r}") from error
    first = year * 12 + month - 1
    for i, label in enumerate(times):
        n = first + i
        if label != f"{n // 12:04d}-{n % 12 + 1:02d}":
            raise ContractError(f"times are not consecutive calendar months: {label!r} at position {i}")
    if january_start and month != 1:
        raise ContractError(f"times start in month {month}, but this comparator takes the first value as January")


def run_adapter(
    comparator: str,
    compute: Callable[[dict[str, Any], dict[str, tuple[list[str], list[float]]]], dict[str, Any]],
    argv: Sequence[str],
) -> int:
    """Shared command line: ``run <request.json> <out_dir>``.

    ``compute(request, inputs)`` receives the verified request and each input series, and returns
    ``version``, ``effective_method``, ``software`` (versions), optional ``warnings``, and
    ``series``: ``{output name: (times, values)}``. Only ``compute`` is timed, so import time is
    not in ``wall_time_s`` as long as the adapter imports its comparator at module level.
    Python warnings raised inside ``compute`` are recorded after the ones it returns.
    """
    if len(argv) != 3 or argv[0] != "run":
        print(f"usage: {comparator} run <request.json> <out_dir>", file=sys.stderr)
        return 2
    request_path, out_dir = path_arg(argv[1]), path_arg(argv[2])
    request = read_request(request_path)
    inputs = {
        name: read_entry_series(request_path.parent, entry, f"input {name!r}")
        for name, entry in request["inputs"].items()
    }
    out_dir.mkdir(parents=True, exist_ok=True)  # NOSONAR (S8707)
    # warnings raised inside compute (fit quality, clamped windows) belong in the result, not only on stderr
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        computed = compute(request, inputs)
        wall = time.perf_counter() - started
    raised = [str(w.message) for w in caught]
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
            "warnings": [*computed.get("warnings", []), *raised],
            "outputs": outputs,
        },
    )
    return 0

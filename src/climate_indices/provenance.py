"""Dataset provenance for computed climate-index outputs.

Every output variable the xarray adapters and the CLI build carries a few
searchable attributes and one strict-JSON record, ``climate_indices_provenance``,
that names the software, the backend, the effective parameters, and each input
(see :doc:`provenance`). The record is per variable, so merging results from
separate runs into one Dataset keeps each variable's own lineage.

This module only builds and reads metadata. It never reads data values, and the
numerical kernels never import it. Git is called at most once per process, the
first time a record is built.
"""

from __future__ import annotations

import copy
import datetime
import enum
import functools
import hashlib
import json
import math
import os
import re
import shutil
import subprocess  # nosec B404 # runs only the resolved git binary, with fixed arguments
from collections.abc import Mapping
from dataclasses import dataclass, fields
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import xarray as xr

from climate_indices.exceptions import InvalidArgumentError

__all__ = [
    "GIT_COMMIT_ATTR",
    "PROVENANCE_ATTR",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_ATTR",
    "SOURCE_ATTR",
    "VERSION_ATTR",
    "read_provenance",
    "software_identity",
    "with_source",
]

SCHEMA = "climate_indices.dataset-provenance"
SCHEMA_VERSION = "1.0.0"

VERSION_ATTR = "climate_indices_version"
GIT_COMMIT_ATTR = "climate_indices_git_commit"
SCHEMA_VERSION_ATTR = "climate_indices_provenance_schema_version"
PROVENANCE_ATTR = "climate_indices_provenance"
#: caller-supplied source identifiers on an input DataArray (see ``with_source``)
SOURCE_ATTR = "climate_indices_source"

# attributes this module owns; an output never inherits them from its input
OWN_ATTRS = (VERSION_ATTR, GIT_COMMIT_ATTR, SCHEMA_VERSION_ATTR, PROVENANCE_ATTR, SOURCE_ATTR)

_DISTRIBUTION = "climate_indices"
_BUILD_INFO = "_build_info.json"
_UNKNOWN = "unknown"

# bounds that keep a record small, strict JSON, and far below HDF5's 64 KiB
# compact-attribute limit
_MAX_STRING = 1024
_MAX_ITEMS = 64
_MAX_INLINE_ARRAY = 16
_MAX_DEPTH = 4
_MAX_UPSTREAM = 16
_MAX_RECORD_BYTES = 32 * 1024

# global attributes worth keeping from an input's parent dataset: identity,
# licensing, and lineage, not descriptive prose
_DATASET_ATTRIBUTES = frozenset(
    {
        "conventions",
        "creator_name",
        "date_created",
        "date_issued",
        "date_modified",
        "doi",
        "history",
        "id",
        "institution",
        "license",
        "naming_authority",
        "processing_level",
        "product_version",
        "project",
        "publisher_name",
        "references",
        "source",
        "source_id",
        "time_coverage_end",
        "time_coverage_start",
        "title",
        "tracking_id",
        "version",
    }
)
_SECRET_NAME = re.compile(r"pass(word|wd)?|secret|token|api_?key|credential|signature|authorization", re.IGNORECASE)
_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_REDACTED = "[redacted]"


@dataclass(frozen=True)
class InputSnapshot:
    """An input description captured before a later step rewrote its metadata.

    The CLI takes one right after opening its inputs, so unit normalization
    cannot replace the source's ``units`` in the record.
    """

    record: dict[str, Any]


def with_source(
    data: xr.DataArray,
    *,
    uri: str | os.PathLike[str] | None = None,
    dataset_id: str | None = None,
    doi: str | None = None,
    version: str | None = None,
    checksum: Mapping[str, str] | None = None,
    dataset_attributes: Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """Return ``data`` labeled with source identifiers for the provenance record.

    Use it for in-memory data, which has no file to name, or to keep the parent
    dataset's global attributes, which a variable selected from a Dataset does
    not carry. The identifiers are stored as the ``climate_indices_source``
    attribute of a shallow copy, so the data stays lazy. Computed outputs record
    them under this input and do not inherit the attribute.

    Args:
        data: The input DataArray.
        uri: Where the data came from. A local path is reduced to its file name; a
            URL loses its user information, query string, and fragment.
        dataset_id: A stable dataset identifier, e.g. an ACDD ``id``.
        doi: The dataset DOI.
        version: The dataset version.
        checksum: An existing checksum with the keys ``algorithm``, ``value``, and
            ``object`` (what it identifies, e.g. ``"file"``). Nothing is computed.
        dataset_attributes: The parent Dataset's ``attrs``; only identity,
            licensing, and lineage keys are kept.

    Returns:
        A shallow copy of ``data`` carrying the identifiers.

    Raises:
        InvalidArgumentError: If ``data`` is not a DataArray or ``checksum`` lacks a key.
    """
    if not isinstance(data, xr.DataArray):
        raise InvalidArgumentError("with_source() labels an xarray.DataArray.", argument_name="data")
    payload: dict[str, Any] = {}
    if uri is not None:
        payload["source"] = _locator(os.fspath(uri))
    identifiers = {"dataset_id": dataset_id, "doi": doi, "version": version}
    if any(value is not None for value in identifiers.values()):
        payload["identifiers"] = _jsonable({key: value for key, value in identifiers.items() if value is not None})
    if checksum is not None:
        missing = {"algorithm", "value", "object"} - set(checksum)
        if missing:
            raise InvalidArgumentError(
                f"checksum needs the keys algorithm, value, and object; missing {sorted(missing)}.",
                argument_name="checksum",
            )
        payload["checksum"] = {key: _jsonable(checksum[key]) for key in ("algorithm", "value", "object")}
    if dataset_attributes is not None:
        payload["dataset_attributes"] = _dataset_attributes(dataset_attributes)
    labeled: xr.DataArray = data.copy(deep=False)
    labeled.attrs = {**data.attrs, SOURCE_ATTR: _canonical(payload)}
    return labeled


def read_provenance(data: xr.DataArray) -> dict[str, Any] | None:
    """Decode the provenance record of one output variable.

    Args:
        data: A computed DataArray, or a variable of a reopened NetCDF or Zarr
            dataset, e.g. ``xr.open_dataset(path)["spi"]``.

    Returns:
        The decoded record, or None when the variable carries none.

    Raises:
        ValueError: If the attribute holds invalid JSON.
    """
    raw = data.attrs.get(PROVENANCE_ATTR)
    if raw is None:
        return None
    try:
        record: dict[str, Any] = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except ValueError as error:
        raise ValueError(f"{PROVENANCE_ATTR} of {data.name!r} is not valid JSON: {error}") from error
    return record


def software_identity() -> dict[str, Any]:
    """Identify the imported climate_indices: version, repository, and source commit.

    Resolved once per process. A source checkout reports its own Git ``HEAD`` and
    tracked-file dirty state; an installed package reports the commit embedded at
    build time, or the one in its ``direct_url.json``. The working directory and
    ``GIT_*`` environment variables never contribute. See :doc:`provenance` for
    the precedence and conflict rules.

    Returns:
        A new dict with ``distribution``, ``version``, ``repository_url``,
        ``git_commit`` (None when not verifiable), ``git_commit_source``,
        ``git_dirty`` (None when unknown), and optional ``notes``/``conflicts``.
    """
    return copy.deepcopy(_cached_software_identity())


@functools.lru_cache(maxsize=1)
def _cached_software_identity() -> dict[str, Any]:
    try:
        distribution: metadata.Distribution | None = metadata.distribution(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        distribution = None
    return resolve_software_identity(Path(__file__).resolve().parent, distribution, shutil.which("git"))


def resolve_software_identity(
    package_dir: Path, distribution: metadata.Distribution | None, git: str | None
) -> dict[str, Any]:
    """Resolve the software identity for an imported package directory (see ``software_identity``)."""
    identity: dict[str, Any] = {
        "distribution": _DISTRIBUTION,
        "version": distribution.version if distribution is not None else _UNKNOWN,
        "repository_url": _repository_url(distribution),
        "git_commit": None,
        "git_commit_source": "unavailable",
        "git_dirty": None,
    }
    notes: list[str] = []
    checkout = _source_checkout(package_dir)
    if checkout is not None:
        # a checkout can move on after a build, so embedded metadata is never trusted here
        state = _git_state(checkout, git) if git is not None else None
        if state is None:
            notes.append("source checkout without a usable git; embedded build metadata is not used for a checkout")
        else:
            identity.update(git_commit=state[0], git_dirty=state[1], git_commit_source="source-checkout")
    else:
        candidates = [
            candidate
            for candidate in (_embedded_build_info(package_dir), _direct_url_commit(distribution, package_dir))
            if candidate is not None
        ]
        if len({candidate["git_commit"] for candidate in candidates}) > 1:
            identity["conflicts"] = candidates
            notes.append("build metadata and direct_url.json name different commits")
        elif candidates:
            identity.update(candidates[0])
        if distribution is not None and not _distribution_matches(distribution, package_dir):
            notes.append("the installed distribution metadata may not describe the imported package")
    if notes:
        identity["notes"] = notes
    return identity


def _source_checkout(package_dir: Path) -> Path | None:
    """The checkout root when the package is imported from ``<root>/src/climate_indices``."""
    root = package_dir.parent.parent
    if package_dir.parent.name == "src" and (root / ".git").exists() and (root / "pyproject.toml").is_file():
        return root
    return None


def _git_state(root: Path, git: str) -> tuple[str, bool] | None:
    """``HEAD`` and tracked-file dirty state of ``root``, or None if Git cannot tell."""
    # GIT_DIR, GIT_WORK_TREE and friends would point git at another repository
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(  # nosec B603 # fixed arguments, no shell
                [git, "--no-optional-locks", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    toplevel = run("rev-parse", "--show-toplevel")
    if toplevel is None or Path(toplevel).resolve() != root.resolve():
        return None
    commit = run("rev-parse", "--verify", "HEAD^{commit}")
    status = run("status", "--porcelain", "--untracked-files=no")
    if commit is None or status is None or not _COMMIT.fullmatch(commit):
        return None
    return commit, bool(status)


def _embedded_build_info(package_dir: Path) -> dict[str, Any] | None:
    """The commit a build hook wrote into the package (``hatch_build.py``)."""
    try:
        info = json.loads((package_dir / _BUILD_INFO).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    commit = info.get("git_commit") if isinstance(info, dict) else None
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        return None
    dirty = info.get("git_dirty")
    return {
        "git_commit": commit,
        "git_commit_source": "build-metadata",
        "git_dirty": dirty if dirty in (True, False) else None,
    }


def _direct_url_commit(distribution: metadata.Distribution | None, package_dir: Path) -> dict[str, Any] | None:
    """The VCS commit pip recorded (PEP 610), when that distribution is the imported one."""
    if distribution is None or not _distribution_matches(distribution, package_dir):
        return None
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
    except ValueError:
        return None
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) and vcs_info.get("vcs") == "git" else None
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        return None
    return {"git_commit": commit, "git_commit_source": "direct-url", "git_dirty": None}


def _distribution_matches(distribution: metadata.Distribution, package_dir: Path) -> bool:
    """Whether the distribution's files hold the package that was imported."""
    located = Path(str(distribution.locate_file("climate_indices/__init__.py")))
    try:
        return located.resolve() == (package_dir / "__init__.py").resolve()
    except OSError:
        return False


def _repository_url(distribution: metadata.Distribution | None) -> str | None:
    """The ``Repository`` project URL from the installed package metadata."""
    if distribution is None:
        return None
    for entry in distribution.metadata.get_all("Project-URL") or []:
        label, _, url = str(entry).partition(",")
        if label.strip().lower() == "repository":
            return url.strip()
    return None


def _backend_identity() -> dict[str, Any]:
    """Whether the optional Rust extension is loaded; kernels are chosen per call."""
    # deferred: compute imports this package's modules, which import this one
    from climate_indices import compute

    native = getattr(compute, "_native", None)
    # a test double may expose any attribute as a callable, so only a real version string counts
    version = None if native is None else getattr(native, "__version__", None)
    return {
        "native_extension": "not-installed" if native is None else "available",
        "native_version": version if isinstance(version, str) else None,
        "kernel_selection": "per call at execution time; not recorded",
    }


def provenance_attributes(
    *,
    index: str | None,
    implementation: str | None,
    interface: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, str]:
    """Build the searchable provenance attributes and the JSON record for one output variable.

    Args:
        index: The index display name, as in the ``history`` entry.
        implementation: The qualified name of the function that computed it, if known.
        interface: ``"python"`` or ``"cli"``.
        parameters: The effective parameters, including defaults and inferred values.
        inputs: Each input by role: a DataArray, a NumPy value, or an ``InputSnapshot``.

    Returns:
        The ``climate_indices_git_commit``, ``climate_indices_provenance_schema_version``,
        and ``climate_indices_provenance`` attributes.
    """
    software = software_identity()
    operation = {"index": index, "interface": interface}
    if implementation is not None:
        operation["implementation"] = implementation
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operation": operation,
        "software": software,
        "backend": _backend_identity(),
        "parameters": _jsonable(dict(parameters)),
        "inputs": [describe_input(role, value) for role, value in inputs.items()],
    }
    record = _bounded(record)
    record["record_id"] = "sha256:" + hashlib.sha256(_canonical(record).encode("ascii")).hexdigest()
    return {
        GIT_COMMIT_ATTR: software["git_commit"] or _UNKNOWN,
        SCHEMA_VERSION_ATTR: SCHEMA_VERSION,
        PROVENANCE_ATTR: _canonical(record),
    }


def snapshot_input(data: xr.DataArray) -> InputSnapshot:
    """Describe an input now, before a later step rewrites its attributes."""
    return InputSnapshot(describe_input("", data))


def describe_input(role: str, data: Any) -> dict[str, Any]:
    """Describe one input from its metadata only; data values are never read."""
    if isinstance(data, InputSnapshot):
        return {**data.record, "role": role}
    if not isinstance(data, xr.DataArray):
        if np.ndim(data) == 0:
            return {"role": role, "type": "scalar", "value": _jsonable(data)}
        return {"role": role, **_array_summary(data)}

    record: dict[str, Any] = {
        "role": role,
        "variable": None if data.name is None else str(data.name),
        "dims": [str(dim) for dim in data.dims],
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "chunked": data.chunks is not None,
        "coordinates": {str(dim): _coordinate_summary(data, dim) for dim in data.dims},
        "attributes": _jsonable({key: value for key, value in data.attrs.items() if key not in OWN_ATTRS}),
    }
    source = data.encoding.get("source")
    if isinstance(source, str | os.PathLike):
        record["source"] = _locator(os.fspath(source))
    caller = _decode(data.attrs.get(SOURCE_ATTR))
    if isinstance(caller, dict):
        record.update(_source_fields(caller))  # only validated source fields can override the opened file's locator
    record["upstream"] = _upstream(data.attrs.get(PROVENANCE_ATTR))
    return record


def _coordinate_summary(data: xr.DataArray, dim: Any) -> dict[str, Any]:
    """Size, first/last label, units, and calendar of one dimension, from its in-memory index."""
    summary: dict[str, Any] = {"size": int(data.sizes[dim])}
    index = data.indexes.get(dim)
    if index is not None and len(index):
        summary["first"] = _jsonable(index[0])
        summary["last"] = _jsonable(index[-1])
    if dim in data.coords:
        coordinate = data.coords[dim]
        for key in ("units", "calendar"):
            value = coordinate.encoding.get(key, coordinate.attrs.get(key))
            if value is None and key == "calendar":
                value = getattr(index, "calendar", None)
            if value is not None:
                summary[key] = _jsonable(value)
    return summary


def _array_summary(data: Any) -> dict[str, Any]:
    """Type, shape, and dtype of an array-like, without converting it."""
    summary: dict[str, Any] = {"type": type(data).__name__}
    shape = getattr(data, "shape", None)
    if shape is None and isinstance(data, list | tuple):
        shape = np.shape(data)
    if shape is not None:
        summary["shape"] = [int(length) for length in shape]
    dtype = getattr(data, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    return summary


def _source_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and sanitize persisted source fields without accepting arbitrary input fields."""
    result: dict[str, Any] = {}
    source = payload.get("source")
    if isinstance(source, dict):
        if source.get("kind") == "url" and isinstance(source.get("url"), str):
            result["source"] = _locator(source["url"])
            if source.get("query_removed") is True:
                result["source"]["query_removed"] = True
        elif source.get("kind") == "local-file" and isinstance(source.get("name"), str):
            result["source"] = _locator(source["name"])
    for field, keys in (
        ("identifiers", ("dataset_id", "doi", "version")),
        ("checksum", ("algorithm", "value", "object")),
    ):
        value = payload.get(field)
        if isinstance(value, dict):
            result[field] = _jsonable({key: value[key] for key in keys if key in value})
    attrs = payload.get("dataset_attributes")
    if isinstance(attrs, dict):
        result["dataset_attributes"] = _dataset_attributes(attrs)
    return result


def _lineage_digest(record: dict[str, Any]) -> dict[str, Any] | None:
    """Accept only the fields a lineage digest describes, normalized as fresh metadata."""
    inputs = record.get("inputs", [])
    if not isinstance(inputs, list) or any(not isinstance(entry, dict) for entry in inputs):
        return None
    if "record_id" in record and not isinstance(record["record_id"], str):
        return None
    digest: dict[str, Any] = {
        key: _jsonable(record[key]) for key in ("record_id", "created", "operation", "parameters") if key in record
    }
    software = record.get("software")
    if isinstance(software, dict):
        digest["software"] = {key: _jsonable(software.get(key)) for key in ("version", "git_commit", "git_dirty")}
    digest["inputs"] = [
        {
            **{key: _jsonable(entry[key]) for key in ("role", "variable") if key in entry},
            **{key: value for key, value in _source_fields(entry).items() if key in ("source", "identifiers")},
        }
        for entry in inputs[:_MAX_ITEMS]
    ]
    return digest


def _upstream(raw: Any) -> list[dict[str, Any]]:
    """Digests of an input's own record and its upstream steps, flattened and deduplicated."""
    if raw is None:
        return []
    record = _decode(raw)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return [{"unparseable": True}]
    digest = _lineage_digest(record)
    if digest is None:
        return [{"unparseable": True}]
    flattened = [digest]
    seen = {digest.get("record_id")}
    for entry in record.get("inputs", []):
        upstream = entry.get("upstream", [])
        if not isinstance(upstream, list):
            flattened.append({"unparseable": True})
            continue
        for step in upstream:
            normalized = _lineage_digest(step) if isinstance(step, dict) else None
            if normalized is None:
                flattened.append({"unparseable": True})
            elif normalized.get("record_id") not in seen:
                seen.add(normalized.get("record_id"))
                flattened.append(normalized)
    if len(flattened) > _MAX_UPSTREAM:
        flattened = [*flattened[:_MAX_UPSTREAM], {"omitted_steps": len(flattened) - _MAX_UPSTREAM}]
    return flattened


def _dataset_attributes(attrs: Mapping[str, Any]) -> Any:
    """The identity, licensing, and lineage keys of a dataset's global attributes."""
    return _jsonable({key: value for key, value in attrs.items() if str(key).lower() in _DATASET_ATTRIBUTES})


def _bounded(record: dict[str, Any]) -> dict[str, Any]:
    """Drop the bulkiest sections, in order, until the record fits ``_MAX_RECORD_BYTES``."""
    # Reserve the exact serialized record_id space before hashing the final reduced record.
    record["record_id"] = "sha256:" + "0" * 64
    truncated: list[str] = []
    for section in ("attributes", "dataset_attributes", "upstream"):
        if len(_canonical(record)) <= _MAX_RECORD_BYTES:
            break
        for entry in record["inputs"]:
            if section in entry:
                entry[section] = {"omitted": "record size limit"}
        truncated.append(f"inputs[].{section}")
        record["truncated"] = truncated
    for section in ("parameters", "inputs", "operation", "software", "backend"):
        if len(_canonical(record)) <= _MAX_RECORD_BYTES:
            break
        omitted = {"omitted": "record size limit"}
        record[section] = [omitted] if section == "inputs" else omitted
        truncated.append(section)
        record["truncated"] = truncated
    del record["record_id"]
    return record


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Normalize a metadata value into bounded, strict-JSON data."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="auto"))
    if isinstance(value, np.timedelta64):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, bytes | bytearray):
        value = bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        return _truncate(_scrub(value))
    if isinstance(value, datetime.date | datetime.time) or type(value).__module__.startswith("cftime"):
        return str(value.isoformat())
    if isinstance(value, os.PathLike):
        return _locator(os.fspath(value))
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _jsonable(value.item(), depth)
    if depth >= _MAX_DEPTH:
        return {"omitted": "nesting depth limit"}
    if isinstance(value, Mapping):
        return _jsonable_mapping(value, depth)
    if isinstance(value, xr.DataArray):
        summary = _array_summary(value)
        if PROVENANCE_ATTR in value.attrs:
            summary["upstream"] = _upstream(value.attrs[PROVENANCE_ATTR])
        return summary
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.size <= _MAX_INLINE_ARRAY:
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, list | tuple) and len(value) <= _MAX_INLINE_ARRAY:
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, np.ndarray | list | tuple) or hasattr(value, "shape"):
        return _array_summary(value)
    # Deferred imports keep the metadata module independent of fire's adapter import cycle.
    from climate_indices.fire import CFFWISState, DCState, DMCState, FFMCState, KBDIState

    if isinstance(value, KBDIState | CFFWISState | FFMCState | DMCState | DCState):
        return _jsonable({field.name: getattr(value, field.name) for field in fields(value)}, depth)
    return {"unsupported_type": type(value).__name__}


def _jsonable_mapping(value: Mapping[Any, Any], depth: int) -> dict[str, Any]:
    """A bounded mapping with string keys; values under secret-like keys are redacted."""
    keys = sorted(value, key=str)
    result = {
        str(key): _REDACTED if _SECRET_NAME.search(str(key)) else _jsonable(value[key], depth + 1)
        for key in keys[:_MAX_ITEMS]
    }
    if len(keys) > _MAX_ITEMS:
        result["omitted_keys"] = len(keys) - _MAX_ITEMS
    return result


def _truncate(text: str) -> str:
    if len(text) <= _MAX_STRING:
        return text
    return f"{text[:_MAX_STRING]}...[truncated {len(text) - _MAX_STRING} characters]"


def _scrub(text: str) -> str:
    """Sanitize URLs and reduce recognizable absolute file paths to their file names."""
    pattern = rf"{_URL.pattern}|(?<![\w:/])(?:[A-Za-z]:[\\/]|/|~[\\/])[^\s\"'<>]+"

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        if "://" in value and not value.lower().startswith("file://"):
            return _sanitize_url(value)[0]
        return str(_locator(value)["name"])

    return re.sub(pattern, replace, text)


def _sanitize_url(url: str) -> tuple[str, bool]:
    """A URL without user information, query, or fragment, and whether a query was dropped."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        netloc = f"{host}:{parts.port}" if parts.port is not None else host
    except ValueError:
        return "[unparseable URL]", False
    cleaned = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    return (f"{cleaned}?{_REDACTED}" if parts.query else cleaned), bool(parts.query)


def _locator(raw: str) -> dict[str, Any]:
    """A privacy-safe source locator: a local file's name, or a sanitized URL."""
    if "://" not in raw:
        return {"kind": "local-file", "name": _truncate(re.split(r"[\\/]", raw.rstrip("\\/"))[-1])}
    if raw.lower().startswith("file://"):
        try:
            path = urlsplit(raw).path
        except ValueError:
            return {"kind": "local-file", "name": "[unparseable URL]"}
        return {"kind": "local-file", "name": _truncate(re.split(r"[\\/]", path.rstrip("/"))[-1])}
    url, query_removed = _sanitize_url(raw)
    locator: dict[str, Any] = {"kind": "url", "url": _truncate(url.removesuffix(f"?{_REDACTED}"))}
    if query_removed:
        locator["query_removed"] = True
    return locator


def _decode(raw: Any) -> Any:
    """Parse a JSON attribute, or None when it is absent or invalid."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _canonical(value: Any) -> str:
    """Strict, ASCII, key-sorted JSON; NetCDF3 and Zarr store it unchanged."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)

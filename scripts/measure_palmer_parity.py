"""Measure Rust-versus-Python parity of ``pdsi()`` and ``scpdsi()`` on the 344 climate divisions.

REFIMPL-34 (#1636). Each division runs twice per index, once with the Rust extension and
once with ``palmer._native`` set to None (the pure-Python reference). Per output the
script records the largest absolute and relative difference, the months whose sign
differs, the NaN-position mismatches, and a SHA-256 of each path's float64 bytes, so
a later ``compare`` run can test bit-for-bit equality across platforms.

    uv run python scripts/measure_palmer_parity.py measure --out parity-<platform>.json
    uv run python scripts/measure_palmer_parity.py compare parity-a.json parity-b.json ...

``measure`` needs the extension (``uv run maturin develop --release``). It fails
if a division does not call the expected Rust kernels. The output path must stay
inside the working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess  # nosec B404 # Only the local rustc runs, with fixed arguments and no shell.
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixture" / "palmer"
START, CALIBRATION = 1895, (1931, 1990)
OUTPUTS = ("pdsi", "phdi", "pmdi", "zindex")
_SHARED_KERNELS = {"palmer_water_balance", "palmer_k_prime", "palmer_raw_zindex"}
_KERNELS = {
    "pdsi": _SHARED_KERNELS | {"palmer_pdi"},
    "scpdsi": _SHARED_KERNELS | {"scpdsi_duration_factors", "palmer_wells"},
}


class _NativeRecorder(ModuleType):
    def __init__(self, extension: ModuleType) -> None:
        super().__init__(extension.__name__)
        self.extension = extension
        self.calls: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self.extension, name)
        if isinstance(attribute, type):
            return attribute

        def record(*args: Any, **kwargs: Any) -> Any:
            result = attribute(*args, **kwargs)
            self.calls.add(name)
            return result

        return record


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def _stats(rust: np.ndarray, python: np.ndarray) -> dict[str, Any]:
    rust, python = np.asarray(rust, dtype=np.float64), np.asarray(python, dtype=np.float64)
    both = np.isfinite(rust) & np.isfinite(python)
    # index before subtracting, so a masked-out infinity never enters the arithmetic
    finite_rust, finite_python = rust[both], python[both]
    difference = np.abs(finite_rust - finite_python)
    nonzero = finite_python != 0
    relative = difference[nonzero] / np.abs(finite_python[nonzero])
    # a finite value against an infinity, or two infinities of opposite sign, is a
    # difference that the finite-only statistics below would report as zero
    infinite = np.isinf(rust) ^ np.isinf(python)
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "max_rel": float(relative.max(initial=0.0)),
        "sign_mismatches": int((np.sign(finite_rust) != np.sign(finite_python)).sum()),
        "nan_mismatches": int((np.isnan(rust) != np.isnan(python)).sum()),
        "infinite_mismatches": int(infinite.sum() + (np.isinf(rust) & np.isinf(python) & (rust != python)).sum()),
        "rust_sha256": _digest(rust),
        "python_sha256": _digest(python),
    }


def _params(rust: dict[str, Any], python: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(python)
    pairs = [(np.asarray(rust[key], dtype=np.float64), np.asarray(python[key], dtype=np.float64)) for key in keys]
    return {
        "keys": keys,
        "max_abs": max((float(np.abs(r - p).max(initial=0.0)) for r, p in pairs), default=0.0),
        "rust_sha256": _digest(np.concatenate([r.ravel() for r, _ in pairs])),
        "python_sha256": _digest(np.concatenate([p.ravel() for _, p in pairs])),
    }


def _division(name: str) -> tuple[str, dict[str, Any]]:
    from climate_indices import _native_backend
    from climate_indices.drought import palmer

    native = _native_backend.EXTENSION
    if native is None:
        raise RuntimeError("the Rust extension is not built")

    directory = FIXTURE / name
    try:
        precips, pet = np.load(directory / "precips.npy"), np.load(directory / "pet.npy")
        awc = float(json.loads((ROOT / "tests" / "fixture" / "palmer_awc.json").read_text(encoding="utf-8"))[name])
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError(f"{name}: cannot load Palmer fixtures: {error}") from error
    result: dict[str, Any] = {}
    for index in (palmer.pdsi, palmer.scpdsi):
        recorder = _NativeRecorder(native)
        previous = palmer._native
        try:
            with np.errstate(all="ignore"):
                palmer._native = recorder
                rust = index(precips, pet, awc, START, *CALIBRATION)
                missing = _KERNELS[index.__name__] - recorder.calls
                if missing:
                    raise RuntimeError(
                        f"{name} {index.__name__}: Rust kernels not called: {', '.join(sorted(missing))}"
                    )
                palmer._native = None
                python = index(precips, pet, awc, START, *CALIBRATION)
        finally:
            palmer._native = previous
        *rust_arrays, rust_params = rust
        *python_arrays, python_params = python
        if rust_params is None or python_params is None:
            raise RuntimeError(f"{name} {index.__name__}: missing fitting parameters")
        entry = {output: _stats(r, p) for output, r, p in zip(OUTPUTS, rust_arrays, python_arrays, strict=True)}
        entry["params"] = _params(rust_params, python_params)
        result[index.__name__] = entry
    return name, result


def _toolchain() -> dict[str, Any]:
    rustc = shutil.which("rustc")

    def run(*args: str) -> str:
        if rustc is None:
            return "unavailable"
        try:
            # The rustc path comes from PATH and all arguments are fixed.
            completed = subprocess.run([rustc, *args], capture_output=True, text=True, check=True)  # nosec B603
            return completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    from climate_indices import _native_backend

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "rustc": run("-vV"),
        "rust_cfg": run("--print", "cfg").splitlines(),
        "extension": getattr(_native_backend.EXTENSION, "__file__", None),
    }


def measure(out: Path, workers: int) -> int:
    from climate_indices import _native_backend

    out = out.resolve()
    if not out.is_relative_to(Path.cwd()):
        print("the output path must stay inside the working directory", file=sys.stderr)
        return 2
    if workers < 1:
        print(f"workers must be at least 1, got {workers}", file=sys.stderr)
        return 2
    if _native_backend.EXTENSION is None:
        print("the Rust extension is not built; run `uv run maturin develop --release`", file=sys.stderr)
        return 2
    names = sorted(path.name for path in FIXTURE.iterdir() if path.is_dir())
    with ProcessPoolExecutor(workers) as pool:
        divisions = dict(pool.map(_division, names))
    summary: dict[str, Any] = {}
    for index in ("pdsi", "scpdsi"):
        for output in (*OUTPUTS, "params"):
            rows = [divisions[name][index][output] for name in names]
            summary[f"{index}.{output}"] = {
                "max_abs": max(row["max_abs"] for row in rows),
                **({} if output == "params" else {"max_rel": max(row["max_rel"] for row in rows)}),
                **(
                    {}
                    if output == "params"
                    else {
                        "sign_mismatches": sum(row["sign_mismatches"] for row in rows),
                        "nan_mismatches": sum(row["nan_mismatches"] for row in rows),
                        "infinite_mismatches": sum(row["infinite_mismatches"] for row in rows),
                    }
                ),
                "python_rust_identical_divisions": sum(row["rust_sha256"] == row["python_sha256"] for row in rows),
            }
    report = {"toolchain": _toolchain(), "divisions_count": len(names), "summary": summary, "divisions": divisions}
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def _mapping(value: Any, path: Path, where: str) -> dict[str, Any]:
    """``value`` as a nonempty mapping, or a report error that names ``where``."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path}: no {where}")
    return value


def _report_digest(stats: Any, key: str, path: Path, where: str) -> str:
    """One report entry's digest for ``key``, or a report error that names ``where``."""
    digest = stats.get(key) if isinstance(stats, dict) else None
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"{path}: invalid digest for {where}")
    return digest


def _report_digests(path: Path, key: str) -> dict[tuple[str, str, str], str]:
    try:
        # the report path is the operator's own CLI argument
        divisions = json.loads(path.read_text(encoding="utf-8"))["divisions"]  # NOSONAR (S8707)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: cannot read divisions: {error}") from error
    digests = {}
    for division, indices in _mapping(divisions, path, "divisions").items():
        for index, outputs in _mapping(indices, path, f"indices for {division}").items():
            for output, stats in _mapping(outputs, path, f"outputs for {division} {index}").items():
                digests[division, index, output] = _report_digest(stats, key, path, f"{division} {index}.{output}")
    return digests


def compare(paths: list[Path], backend: str) -> int:
    """Compare one backend's digests across reports with identical, nonempty coverage."""
    if len(paths) < 2:
        print(f"{backend}: need at least two reports to compare, got {len(paths)}", file=sys.stderr)
        return 2
    key = f"{backend}_sha256"
    first_path, *others = paths
    mismatches: list[str] = []
    try:
        first = _report_digests(first_path, key)
        for path in others:
            actual = _report_digests(path, key)
            for division, index, output in sorted(first.keys() | actual.keys()):
                coordinate = (division, index, output)
                if actual.get(coordinate) != first.get(coordinate):
                    mismatches.append(f"{path} vs {first_path}: {division} {index}.{output}")
    except ValueError as error:
        print(f"{backend}: invalid report: {error}", file=sys.stderr)
        return 2
    for line in mismatches[:50]:
        print(line)
    print(f"{backend}: {len(paths)} reports, {len(mismatches)} differing outputs")
    return 1 if mismatches else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("measure")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--workers", type=int, default=4)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("reports", type=Path, nargs="+")
    cmp_.add_argument("--backend", choices=("rust", "python"), default="rust")
    args = parser.parse_args()
    if args.command == "measure":
        return measure(args.out, args.workers)
    return compare(args.reports, args.backend)


if __name__ == "__main__":
    raise SystemExit(main())

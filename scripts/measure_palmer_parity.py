"""Measure Rust-versus-Python parity of ``pdsi()`` and ``scpdsi()`` on the 344 climate divisions.

REFIMPL-34 (#1636). Each division runs twice per index, once with the Rust extension and
once with ``palmer._native`` set to None (the pure-Python reference). Per output the
script records the largest absolute and relative difference, the months whose sign
differs, the NaN-position mismatches, and a SHA-256 of each path's float64 bytes, so
a later ``compare`` run can test bit-for-bit equality across platforms.

    uv run python scripts/measure_palmer_parity.py measure --out parity-<platform>.json
    uv run python scripts/measure_palmer_parity.py compare parity-a.json parity-b.json ...

``measure`` needs the extension (``uv run maturin develop --release``) and exits
non-zero without it, so a platform that silently fell back to Python cannot pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixture" / "palmer"
START, CALIBRATION = 1895, (1931, 1990)
OUTPUTS = ("pdsi", "phdi", "pmdi", "zindex")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def _stats(rust: np.ndarray, python: np.ndarray) -> dict[str, Any]:
    rust, python = np.asarray(rust, dtype=np.float64), np.asarray(python, dtype=np.float64)
    both = np.isfinite(rust) & np.isfinite(python)
    difference = np.abs(rust - python)[both]
    nonzero = both & (python != 0)
    relative = np.abs(rust - python)[nonzero] / np.abs(python)[nonzero]
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "max_rel": float(relative.max(initial=0.0)),
        "sign_mismatches": int((np.sign(rust)[both] != np.sign(python)[both]).sum()),
        "nan_mismatches": int((np.isnan(rust) != np.isnan(python)).sum()),
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

    directory = FIXTURE / name
    precips, pet = np.load(directory / "precips.npy"), np.load(directory / "pet.npy")
    awc = float(json.loads((ROOT / "tests" / "fixture" / "palmer_awc.json").read_text(encoding="utf-8"))[name])
    # a pool worker handles many divisions, so the extension comes from the backend module,
    # never from ``palmer._native``, which this function leaves switched off between calls
    native = _native_backend.EXTENSION
    assert native is not None, "the Rust extension is not built"
    result: dict[str, Any] = {}
    for index in (palmer.pdsi, palmer.scpdsi):
        runs = []
        try:
            with np.errstate(all="ignore"):
                for backend in (native, None):
                    palmer._native = backend
                    runs.append(index(precips, pet, awc, START, *CALIBRATION))
        finally:
            palmer._native = native
        (rust, python) = runs
        entry = {output: _stats(rust[i], python[i]) for i, output in enumerate(OUTPUTS)}
        entry["params"] = _params(rust[4], python[4])
        result[index.__name__] = entry
    return name, result


def _toolchain() -> dict[str, Any]:
    def run(*command: str) -> str:
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    from climate_indices import _native_backend

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "rustc": run("rustc", "-vV"),
        "rust_cfg": run("rustc", "--print", "cfg").splitlines(),
        "extension": getattr(_native_backend.EXTENSION, "__file__", None),
    }


def measure(out: Path, workers: int) -> int:
    from climate_indices import _native_backend

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
                    }
                ),
                "python_rust_identical_divisions": sum(row["rust_sha256"] == row["python_sha256"] for row in rows),
            }
    report = {"toolchain": _toolchain(), "divisions_count": len(names), "summary": summary, "divisions": divisions}
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def compare(paths: list[Path], backend: str) -> int:
    """Bit-for-bit comparison of one backend's digests across the given reports.

    The first report is the baseline. A report that lacks a division or output the
    baseline has counts as a mismatch, and fewer than two reports cannot compare anything.
    """
    if len(paths) < 2:
        print(f"{backend}: need at least two reports to compare, got {len(paths)}", file=sys.stderr)
        return 2
    reports = {str(path): json.loads(path.read_text(encoding="utf-8")) for path in paths}
    key = f"{backend}_sha256"
    first_name, first = next(iter(reports.items()))
    mismatches: list[str] = []
    for name, report in reports.items():
        for division, indices in first["divisions"].items():
            for index, outputs in indices.items():
                for output, expected in outputs.items():
                    actual = report["divisions"].get(division, {}).get(index, {}).get(output, {}).get(key)
                    if actual != expected[key]:
                        mismatches.append(f"{name} vs {first_name}: {division} {index}.{output}")
    for line in mismatches[:50]:
        print(line)
    print(f"{backend}: {len(reports)} reports, {len(mismatches)} differing outputs")
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

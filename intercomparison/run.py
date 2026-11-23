"""Intercomparison driver: run every adapter on identical inputs and store what each did.

Slice of REFIMPL-08 (issue 1610): gamma SPI-3 for one climate division of tests/fixture/palmer/
through the `climate_indices` and `xclim` adapters, into one Zarr store and one JSON manifest.

    uv run python intercomparison/run.py --division 0101 --study intercomparison/studies/refimpl-08-slice

Heavy runs are manual; CI exercises only the offline tests in tests/test_intercomparison_harness.py.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import re
import shutil
import subprocess  # nosec B404 # runs the fixed adapter commands and git
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS_DIR = ROOT / "intercomparison" / "adapters"
FIXTURE_DIR = ROOT / "tests" / "fixture" / "palmer"

# Artifact names inside a study directory; the writer and the reader must agree.
PRECIP_CSV = "precip.csv"
MANIFEST_JSON = "manifest.json"
STORE_ZARR = "store.zarr"

# One command per adapter; each runs in its own environment.
ADAPTER_COMMANDS: dict[str, list[str]] = {
    "climate_indices": [sys.executable, str(ADAPTERS_DIR / "climate_indices_adapter.py")],
    "xclim": [
        "uv",
        "run",
        "--locked",
        "--project",
        str(ADAPTERS_DIR / "xclim"),
        "python",
        str(ADAPTERS_DIR / "xclim" / "adapter.py"),
    ],
}


def monthly_times(first_year: int, length: int) -> list[str]:
    return [f"{first_year + i // 12}-{i % 12 + 1:02d}" for i in range(length)]


def positions_for(scale: int, first_year: int, last_year: int) -> dict[str, Any]:
    """The aligned configuration `spi-gamma-thom` of method-matrix.json, in the harness vocabulary."""
    return {
        "accumulation": "sum",
        "scale": scale,
        "distribution": "gamma",
        "estimator": "thom_closed_form",
        "zero_handling": "mass_n0_over_n",
        "calendar_grouping": "calendar_month",
        "calibration_period": f"{first_year}/{last_year}",
    }


def zero_accumulations(values: np.ndarray, scale: int) -> int:
    """Count the all-zero ``scale``-month windows: the zeros that zero handling sees.

    An isolated zero month is not one at scale > 1. A window with a missing month is not counted.
    """
    if len(values) < scale:
        return 0
    sums = np.convolve(np.asarray(values, dtype=float), np.ones(scale), mode="valid")
    return int((sums == 0).sum())


def write_request(
    work: Path, *, study: str, unit: str, times: list[str], values: np.ndarray, units: str, positions: dict[str, Any]
) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    digest = contract.write_series(work / PRECIP_CSV, times, values.tolist())
    request = {
        "schema_version": contract.SCHEMA_VERSION,
        "study": study,
        "unit": unit,
        "index": "spi",
        "positions": positions,
        "inputs": {"precip": {"path": PRECIP_CSV, "sha256": digest, "units": units}},
    }
    path = work / "request.json"
    contract.write_json(path, request)
    return path


def run_adapter(name: str, command: list[str], request: Path, out_dir: Path) -> dict[str, Any]:
    """Run one adapter as a subprocess and return its verified result.

    The output directory is emptied first: an adapter that exits 0 without writing must not be
    credited with a previous run's files.
    """
    # any failure but a missing directory must stop the run, or the old files could be credited to this adapter
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(out_dir)
    subprocess.run([*command, "run", str(request), str(out_dir)], check=True)  # nosec B603 # argv is ADAPTER_COMMANDS plus harness-built paths
    if not (out_dir / "result.json").is_file():
        raise contract.ContractError(f"{name}: exited 0 but wrote no result.json")
    result = contract.read_result(out_dir / "result.json")
    if "spi" not in result["outputs"]:
        raise contract.ContractError(f"{name}: the result has no 'spi' output")
    return result


def write_store(
    path: Path, results: dict[str, dict[str, Any]], out_dirs: dict[str, Path], unit: str, times: list[str]
) -> list[str]:
    """One Zarr array ``values`` of shape (implementation, unit, time); returns the implementation order."""
    import zarr

    names = list(results)
    data = np.full((len(names), 1, len(times)), np.nan)
    for i, name in enumerate(names):
        got_times, values = contract.read_entry_series(
            out_dirs[name], results[name]["outputs"]["spi"], f"{name} output 'spi'"
        )
        if got_times != times:
            raise contract.ContractError(f"{name}: output times differ from the input times")
        data[i, 0, :] = values
    group = zarr.open_group(str(path), mode="w")
    group.create_array("values", data=data)
    group.attrs.update(
        {"dims": ["implementation", "unit", "time"], "implementation": names, "unit": [unit], "time": times}
    )
    return names


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()  # nosec B603 B607 # fixed git argv


def harness_state() -> dict[str, Any]:
    """The commit that ran and whether the checkout (untracked files included) differs from it."""
    return {"git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}


def build_manifest(
    *,
    study: str,
    unit: str,
    names: list[str],
    results: dict[str, dict[str, Any]],
    positions: dict[str, Any],
    request: Path,
    source_path: Path,
    source_sha256: str,
    times: list[str],
    units: str,
    zeros: int,
    harness: dict[str, Any],
    created: str | None = None,
) -> dict[str, Any]:
    provenance = contract.read_json(FIXTURE_DIR / "provenance.json")
    entry = contract.read_json(request)["inputs"]["precip"]
    arrays = []
    for i, name in enumerate(names):
        result = results[name]
        warnings = list(result["warnings"]) + [
            f"method: {w}" for w in contract.diff_method(positions, result["effective_method"])
        ]
        arrays.append(
            {
                "implementation": name,
                "comparator": result["comparator"],
                "version": result["version"],
                "effective_method": result["effective_method"],
                "software": result["software"],
                "platform": result["platform"],
                "wall_time_s": result["wall_time_s"],
                "warnings": warnings,
                "output_sha256": result["outputs"]["spi"]["sha256"],
                "store_index": i,
            }
        )
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "study": study,
        "created": created or datetime.date.today().isoformat(),
        "harness": harness,
        "source": f"{source_path.relative_to(ROOT).as_posix()} of this repository",
        # url and download_date are the fixture's own provenance (tests/fixture/palmer/provenance.json)
        "url": provenance["url"],
        "download_date": provenance["download_date"],
        "subset_description": f"Climate division {unit}, monthly precipitation in inches, {times[0]} to {times[-1]}.",
        "checksum_sha256": entry["sha256"],
        "fixture_version": provenance["fixture_version"],
        "notes": (
            "Slice of REFIMPL-08 (issue 1610): shows that the harness runs, not that the comparators agree. "
            "Metrics are REFIMPL-16. url and download_date are those of the fixture's provenance.json. "
            f"The series has {zeros} zero {positions['scale']}-month accumulations, so the zero_handling position is "
            f"{'declared but not exercised' if zeros == 0 else 'exercised'}."
        ),
        "index": "spi",
        "unit": unit,
        "positions": positions,
        "inputs": {
            "precip": {
                "source_path": source_path.relative_to(ROOT).as_posix(),
                "source_sha256": source_sha256,
                "sha256": entry["sha256"],
                "units": units,
                "length": len(times),
                "first": times[0],
                "last": times[-1],
            }
        },
        "store": {
            "path": STORE_ZARR,
            "array": "values",
            "dims": ["implementation", "unit", "time"],
            "shape": [len(names), 1, len(times)],
        },
        "arrays": arrays,
    }


def prepare_study_directory(study_dir: Path) -> Path:
    """Claim an empty directory or reopen a directory with harness output."""
    work = study_dir / "work"
    manifest = study_dir / MANIFEST_JSON
    marker = work / ".intercomparison"
    artifacts = (work, manifest, study_dir / STORE_ZARR, marker, work / PRECIP_CSV, work / "request.json")
    for path in artifacts:
        if path.is_symlink():
            raise contract.ContractError(f"{path}: study artifacts must not be symbolic links")
    study_dir.mkdir(parents=True, exist_ok=True)  # NOSONAR (S8707)
    if manifest.exists() or (any(study_dir.iterdir()) and not marker.is_file()):
        import jsonschema  # type: ignore[import-untyped]

        message = f"{study_dir}: use an empty directory or an existing intercomparison study"
        if not manifest.is_file():
            raise contract.ContractError(message)
        try:
            previous = contract.read_json(manifest)
            schema = contract.read_json(Path(__file__).with_name("manifest.schema.json"))
            jsonschema.validate(previous, schema)
        except (contract.ContractError, jsonschema.ValidationError) as error:
            raise contract.ContractError(message) from error
        if previous["study"] != study_dir.name:
            raise contract.ContractError(message)
    work.mkdir(exist_ok=True)
    marker.touch(exist_ok=True)
    return work


def run_study(study_dir: Path, division: str, scale: int, adapters: dict[str, list[str]]) -> dict[str, Any]:
    harness = harness_state()  # record the checkout before modifying the study
    work = prepare_study_directory(study_dir)
    (study_dir / MANIFEST_JSON).unlink(missing_ok=True)
    if not re.fullmatch(r"\d{4}", division, re.ASCII):  # ASCII keeps \d equivalent to [0-9] and rejects Unicode digits
        raise contract.ContractError("division must be a four-digit climate division ID")
    source = FIXTURE_DIR / division / "precips.npy"
    if not source.resolve().is_relative_to(FIXTURE_DIR.resolve()):
        raise contract.ContractError(f"{source}: outside the Palmer fixture directory")
    data = source.read_bytes()  # one read: the bytes hashed are the bytes that become the run's inputs
    source_sha256 = hashlib.sha256(data).hexdigest()
    values = np.load(io.BytesIO(data))
    if values.ndim != 1 or not values.size:
        raise contract.ContractError(f"{source}: expected a nonempty one-dimensional precipitation series")
    times = monthly_times(1895, len(values))
    positions = positions_for(scale, 1895, int(times[-1][:4]))
    request = write_request(
        work, study=study_dir.name, unit=division, times=times, values=values, units="inch", positions=positions
    )
    out_dirs = {name: work / name for name in adapters}
    results = {name: run_adapter(name, command, request, out_dirs[name]) for name, command in adapters.items()}
    names = write_store(study_dir / STORE_ZARR, results, out_dirs, division, times)
    manifest = build_manifest(
        study=study_dir.name,
        unit=division,
        names=names,
        results=results,
        positions=positions,
        request=request,
        source_path=source,
        source_sha256=source_sha256,
        times=times,
        units="inch",
        zeros=zero_accumulations(values, scale),
        harness=harness,
    )
    contract.check_store_index(manifest)
    contract.write_json(study_dir / MANIFEST_JSON, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--division", default="0101", help="climate division directory under tests/fixture/palmer/")
    parser.add_argument("--study", type=contract.path_arg, required=True, help="study directory (created if absent)")
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args(argv)
    manifest = run_study(args.study, args.division, args.scale, ADAPTER_COMMANDS)
    for array in manifest["arrays"]:
        print(array["implementation"], array["version"], array["output_sha256"][:12], *array["warnings"], sep="  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

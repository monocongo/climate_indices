#!/usr/bin/env python3
"""Generate offline SEI energy references from pinned source and Reading data.

Usage:
    uv run scripts/prepare_energy_sei_fixtures.py <SEI-source.tar.gz> <Reading.nc>

See tests/fixture/energy_sei/README.md for source URLs and extraction details.
R dependencies must already be installed; this script installs only the verified
SEI archive into a temporary library. Neither CI nor library users need R.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404 # fixed R commands on local files; no shell
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixture" / "energy_sei"
R_SCRIPT = ROOT / "scripts" / "energy_sei_reference.R"
SEI_COMMIT = "4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14"
SEI_SHA256 = "49d1f1ee1c68bb0ec758b7b1cff2d29d48daae5dd508fdec7ce5eaabd97501d4"
READING_SHA256 = "3f3726492fa52c0e8d5e1025e3047b09d3ab75a180701f89d4ea78161c6db7e8"
R_TIMEOUT = 3600  # seconds allowed per R command
PAYLOADS = ("inputs.npz", "metadata.json", "reference.npz", "sessionInfo.txt", "warnings.txt")


def _verify_file(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Unexpected source checksum: {path}")


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write named numeric arrays; NumPy before 2.2 reads an allow_pickle keyword as an array."""
    np.savez_compressed(path, **cast("dict[str, Any]", arrays))


def _publish(staging: Path) -> None:
    """Swap in a complete fixture set; restore the previous set if the swap fails."""
    backup = FIXTURE_DIR.with_name(f".{FIXTURE_DIR.name}-backup")
    if backup.exists() and not FIXTURE_DIR.exists():
        os.replace(backup, FIXTURE_DIR)
    try:
        if FIXTURE_DIR.exists():
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(FIXTURE_DIR, backup)
        os.replace(staging, FIXTURE_DIR)
    except BaseException:
        if backup.exists():
            os.replace(backup, FIXTURE_DIR)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _reading_monthly(path: Path) -> np.ndarray:
    """Select DE, preserve all hourly values, and compute complete monthly means."""
    _verify_file(path, READING_SHA256)
    with xr.open_dataset(path) as dataset:
        hours = dataset["time_in_hours_from_first_jan_1950"].values
        np.testing.assert_array_equal(hours, np.arange(622392))
        countries = dataset["NUTS_keys"].values
        locations = np.flatnonzero(countries == "DE")
        if locations.size != 1:
            raise ValueError("Expected exactly one DE country in the Reading source")
        values = dataset["timeseries_data"].isel(NUTS=int(locations[0])).values.astype(np.float64)
    times = np.datetime64("1950-01-01T00") + hours.astype("timedelta64[h]")
    monthly = xr.DataArray(values, dims="time", coords={"time": times}).resample(time="MS").mean(skipna=False)
    result = np.asarray(monthly.values)
    if result.shape != (71 * 12,) or not np.all(np.isfinite(result) & (result >= 0) & (result <= 1)):
        raise ValueError("Expected 852 finite, bounded monthly capacity factors")
    return result


def _synthetic_inputs() -> dict[str, np.ndarray]:
    """Include ties, constants, held-out extremes, gaps, and 9/10/99/100 fits."""
    rng = np.random.default_rng(1170)
    capacity = np.round(rng.uniform(size=(104, 12)), 1)
    capacity[:, 1:4] = [0.0, 1.0, 0.5]
    for month, count in zip(range(4, 8), (9, 10, 99, 100), strict=True):
        capacity[2 + count : 102, month] = np.nan
    capacity[:2, 0] = 0.0
    capacity[102:, 0] = 1.0
    capacity[2:102, 0] = np.clip(capacity[2:102, 0], 0.1, 0.9)
    production = rng.uniform(50.0, 550.0, size=capacity.shape)
    demand = rng.uniform(100.0, 500.0, size=capacity.shape)
    return {
        "capacity": capacity.ravel(),
        "production": production.ravel(),
        "demand": demand.ravel(),
        "residual": (demand - production).ravel(),
    }


def _cases() -> list[dict[str, Any]]:
    cases = []
    for name, function, aggregations in (
        ("capacity", "srepi", ("mean",)),
        ("production", "srepi", ("sum",)),
        ("demand", "standardized_demand_index", ("sum",)),
        ("residual", "srli", ("sum", "mean")),
        ("reading", "srepi", ("mean",)),
    ):
        for aggregation in aggregations:
            for scale in (1, 3):
                cases.append(
                    {
                        "id": f"{name}_{aggregation}_{scale}",
                        "input": name,
                        "function": function,
                        "issue": 1172 if function == "srli" else 1171,
                        "scale": scale,
                        "aggregation": aggregation,
                        "data_start_year": 1950 if name == "reading" else 1900,
                        "calibration_year_initial": 1981 if name == "reading" else 1902,
                        "calibration_year_final": 2010 if name == "reading" else 2001,
                    }
                )
    return cases


def _measure_agreement(
    inputs: dict[str, np.ndarray], cases: list[dict[str, Any]], references: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Measure the library against R outputs without generating reference values."""
    from climate_indices import compute, indices

    measurements: dict[str, dict[str, float]] = {}
    scales: tuple[tuple[compute.OutputScale, float], ...] = (("probability", 1e-15), ("normal", 1e-12))
    for case in cases:
        kwargs = {
            key: case[key]
            for key in ("scale", "data_start_year", "calibration_year_initial", "calibration_year_final", "aggregation")
        }
        measurements[case["id"]] = {}
        for output_scale, tolerance in scales:
            actual = indices.standardized_index(
                inputs[case["input"]].copy(),
                distribution=indices.Distribution.empirical,
                periodicity=compute.Periodicity.monthly,
                output_scale=output_scale,
                clip_negatives=False,
                **kwargs,
            )
            expected = references[f"{case['id']}_{output_scale}"]
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
            measurements[case["id"]][f"{output_scale}_max_abs_error"] = float(np.nanmax(np.abs(actual - expected)))
        # a sum and its mean share empirical ranks, so the scaled values need their own check
        scaled = compute.prepare_scaled(
            inputs[case["input"]].copy(),
            case["scale"],
            compute.Periodicity.monthly,
            aggregation=case["aggregation"],
            clip_negatives=False,
        ).ravel()
        expected_scaled = references[f"{case['id']}_scaled"]
        np.testing.assert_array_equal(np.isnan(scaled), np.isnan(expected_scaled))
        np.testing.assert_allclose(scaled, expected_scaled, rtol=0, atol=1e-12)
        measurements[case["id"]]["scaled_max_abs_error"] = float(np.nanmax(np.abs(scaled - expected_scaled)))
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sei_archive", type=Path)
    parser.add_argument("reading_netcdf", type=Path)
    args = parser.parse_args()
    r, rscript = shutil.which("R"), shutil.which("Rscript")
    if r is None or rscript is None:
        raise FileNotFoundError("R and Rscript are required to regenerate the SEI references")
    cases = _cases()
    env = dict(os.environ)
    if "R_LIBS_USER" not in env and (Path.home() / ".R" / "library").is_dir():
        env["R_LIBS_USER"] = str(Path.home() / ".R" / "library")

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        # verify a private copy so R installs the bytes that passed the checksum
        archive = work / args.sei_archive.name
        shutil.copyfile(args.sei_archive, archive)
        _verify_file(archive, SEI_SHA256)
        inputs = _synthetic_inputs()
        inputs["reading"] = _reading_monthly(args.reading_netcdf)
        library = work / "library"
        library.mkdir()
        subprocess.run(  # nosec B603 # resolved R path; fixed arguments and the verified archive copy
            [r, "CMD", "INSTALL", f"--library={library}", str(archive)],
            env=env,
            check=True,
            timeout=R_TIMEOUT,
        )
        rows = ["case,year,month,value,scale,aggregation,calibration_start,calibration_end"]
        for case in cases:
            for i, value in enumerate(inputs[case["input"]]):
                rows.append(
                    f"{case['id']},{case['data_start_year'] + i // 12},{i % 12 + 1},"
                    f"{value:.17g},{case['scale']},{case['aggregation']},"
                    f"{case['calibration_year_initial']},{case['calibration_year_final']}"
                )
        (work / "input.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        subprocess.run(  # nosec B603 # resolved Rscript path; fixed arguments and temporary files
            [rscript, str(R_SCRIPT), str(library), str(work / "input.csv"), str(work)],
            env=env,
            check=True,
            timeout=R_TIMEOUT,
        )
        references = {}
        for case in cases:
            table = np.genfromtxt(work / f"{case['id']}.csv", delimiter=",", names=True)
            if table.size != inputs[case["input"]].size:
                raise ValueError(f"Incomplete oracle output for {case['id']}")
            for field in ("scaled", "probability", "normal", "count"):
                references[f"{case['id']}_{field}"] = table[field]
        for name in ("gamma_control", "finite_bounds"):
            table = np.genfromtxt(work / f"{name}.csv", delimiter=",", names=True)
            for field in table.dtype.names or ():
                references[f"{name}_{field}"] = table[field]
        measurements = _measure_agreement(inputs, cases, references)
        oracle_files = {name: (work / name).read_bytes() for name in ("sessionInfo.txt", "warnings.txt")}

    metadata = {
        "sei_commit": SEI_COMMIT,
        "sei_archive_sha256": SEI_SHA256,
        "r_script_sha256": hashlib.sha256(R_SCRIPT.read_bytes()).hexdigest(),
        "seed": 1170,
        "python_packages": {"numpy": np.__version__, "xarray": xr.__version__},
        "cases": cases,
        "conventions": {
            "distribution": "empirical",
            "method": "mle",
            "lower": "-Inf",
            "upper": "Inf",
            "cens": "none",
            "n_thres": 10,
            "index_types": ["prob01", "normal"],
            "calendar": "monthly; aggregate complete windows before calendar-month grouping",
            "synthetic_units": {"capacity": "1", "production": "MWh", "demand": "MWh", "residual": "MWh"},
            "residual": "demand - production, without negative clipping",
        },
        "reading": {
            "doi": "10.17864/1947.000321",
            "url": "https://researchdata.reading.ac.uk/321/4/ERA5_data_1950-2020.zip",
            "member": "ERA5_data_1950-2020/wp_onshore/NUTS_0_wp_ons_sim_1_historical_loc_weighted.nc",
            "member_sha256": READING_SHA256,
            "archive_etag": "106c707cabf37e373e28e80bd0769a4c",
            "readme_sha256": "65d4d91746aafea980655652d228f1eb6c7740d2ebbd25dc910780e71cc9efdb",
            "selection": "NUTS_keys == DE; timeseries_data cast to float64 before complete monthly means",
            "time": "622392 consecutive source hours from 1950-01-01; 852 months through 2020-12; no timezone attribute or timezone conversion",
            "units": "dimensionless capacity factor; units interpreted from the provider README",
            "license": "CC-BY-4.0; Copyright 2021 University of Reading; monthly subset derived here",
        },
    }
    # stage the whole set, then publish, so a failed write cannot damage the committed fixtures
    with tempfile.TemporaryDirectory(dir=FIXTURE_DIR.parent, prefix=f".{FIXTURE_DIR.name}-") as temporary:
        staging = Path(temporary) / FIXTURE_DIR.name
        if FIXTURE_DIR.exists():
            shutil.copytree(FIXTURE_DIR, staging)
        else:
            staging.mkdir()
        _save_arrays(staging / "inputs.npz", inputs)
        _save_arrays(staging / "reference.npz", references)
        for name, payload in oracle_files.items():
            (staging / name).write_bytes(payload)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        checksum = hashlib.sha256(b"".join((staging / name).read_bytes() for name in PAYLOADS)).hexdigest()
        provenance = {
            "source": "R SEI 0.2.0 (Allen and Otero); University of Reading (Bloomfield and Brayshaw)",
            "url": f"https://github.com/noeliaof/SEI/tree/{SEI_COMMIT}",
            "download_date": dt.date.today().isoformat(),
            "subset_description": "Synthetic monthly capacity factors, production, demand and signed residual load; DE monthly means from Reading ERA5-derived wind capacity factors, 1950-2020. SEI empirical probabilities/normal indices and a gamma MLE control.",
            "checksum_sha256": checksum,
            "fixture_version": "1.0.0",
            "measured_stats": measurements,
            "validation_tolerance": {
                "rtol": 0.0,
                "scaled_atol": 1e-12,
                "probability_atol": 1e-15,
                "normal_atol": 1e-12,
            },
            "citation": "Allen and Otero (2023), Standardised indices to monitor energy droughts, Renewable Energy 217, 119206; Allen and Otero (2024), Calculating Standardised Indices Using SEI, The R Journal 16(4), 102-122; Bloomfield and Brayshaw (2021), ERA5 derived time series of European aggregated surface weather variables, wind power, and solar power capacity factors, University of Reading.",
            "doi": "10.1016/j.renene.2023.119206; 10.32614/RJ-2024-038; 10.17864/1947.000321",
            "license": "SEI software: GPL (>= 2); no implementation copied. Reading input: CC-BY-4.0, Copyright 2021 University of Reading. Synthetic inputs and generated numerical outputs are fixture data, not SEI source.",
            "notes": "Independent-implementation statistical comparison, not validation of physical power/demand models or actual generation/load. Source checksums, exact conventions and calendar windows are in metadata.json; R versions and dependencies are in sessionInfo.txt. The checksum covers PAYLOADS in scripts/prepare_energy_sei_fixtures.py order. SEI runs once per calendar month after complete-window aggregation. Finite-bound controls document SEI behavior only. Gamma uses SEI MLE, not the library's gamma estimator; it is not beta validation (#1169). Planned energy wrappers remain in #1171/#1172. Maintainer confirmation of extraction, generation and redistribution is required before acceptance.",
        }
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        _publish(staging)
    print(f"Wrote {len(cases)} SEI cases to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()

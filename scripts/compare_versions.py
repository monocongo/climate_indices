#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "h5netcdf>=1.0",
#   "h5py>=3.0",
#   "matplotlib>=3.7",
#   "numpy>=1.24",
#   "pandas>=2.0",
#   "scipy>=1.10",
#   "xarray>=2023.1",
# ]
# ///
"""Visual regression comparison: v2.2.0 (PyPI) vs v2.3.0 (local/git ref).

Installs both versions in isolated uv-managed venvs, runs the climate_indices
CLI for SPI, SPEI, PET, and PNP, then compares outputs and generates plots.

The one known algorithm difference is the SPI/SPEI gamma zero-precipitation
fix: zeros that produced NaN in v2.2.0 now produce a negative SPI value
(drought signal) in v2.3.0.  Everything else should be numerically identical.

Usage examples::

    # Fast fixture-based run (seconds)
    uv run scripts/compare_versions.py --use-fixtures --output-dir /tmp/compare

    # Create the small example NetCDF for the repo
    uv run scripts/compare_versions.py --create-example-netcdf example_data/

    # Full nClimGrid run
    uv run scripts/compare_versions.py \\
        --input-data /path/to/nclimgrid/ \\
        --output-dir comparison_output/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    pass

# ─── constants ────────────────────────────────────────────────────────────────

VERSION_OLD = "2.2.0"
VERSION_NEW = "2.3.0"

# threshold below which differences are considered floating-point noise
IDENTICAL_THRESHOLD = 1e-10

# scales to run for each index type
SPI_SCALES = [1, 3, 6, 12, 24]
SPEI_SCALES = [3, 6, 12]
PNP_SCALES = [3, 6, 12]
DISTRIBUTIONS = ["gamma", "pearson"]

# classification labels
DIFF_IDENTICAL = "IDENTICAL"
DIFF_EXPECTED = "EXPECTED FIX"
DIFF_UNEXPECTED = "UNEXPECTED"

# colors for the summary dashboard (green / amber / red)
COLOR_IDENTICAL = "#2ecc71"
COLOR_EXPECTED = "#f39c12"
COLOR_UNEXPECTED = "#e74c3c"

COLOR_MAP = {
    DIFF_IDENTICAL: COLOR_IDENTICAL,
    DIFF_EXPECTED: COLOR_EXPECTED,
    DIFF_UNEXPECTED: COLOR_UNEXPECTED,
}

# latitude used in the test fixtures (from tests/conftest.py)
FIXTURE_LATITUDE = 25.2292

# data year range for the monthly .npy fixtures (1895-01 through 2017-12)
FIXTURE_DATA_START_YEAR = 1895

# default calibration period matching the fixture conftest
DEFAULT_CALIB_START = 1981
DEFAULT_CALIB_END = 2010


# ─── dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class CompareResult:
    """Statistics from comparing one index output between the two versions."""

    label: str
    index: str
    scale: int | None
    distribution: str | None
    classification: str
    max_abs_diff: float
    rmse: float
    correlation: float
    nan_to_val: int
    val_to_nan: int


# ─── environment management ──────────────────────────────────────────────────


def create_venv(venv_dir: Path, python_version: str = "3.11") -> None:
    """Create an isolated virtual environment using uv.

    Args:
        venv_dir: Target directory for the venv (created if missing).
        python_version: Python version string passed to ``uv venv --python``.
    """
    venv_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "venv", "--python", python_version, str(venv_dir)],
        check=True,
        capture_output=True,
    )


def install_pypi_version(venv_dir: Path, version: str) -> None:
    """Install a pinned version of climate-indices from PyPI into a venv.

    Also installs h5py explicitly because h5netcdf requires it at runtime but
    some uv resolve paths omit the transitive install.

    Args:
        venv_dir: Path to the uv-managed venv.
        version: Exact version string, e.g. ``"2.2.0"``.
    """
    python = str(venv_dir / "bin" / "python")
    subprocess.run(
        ["uv", "pip", "install", f"climate-indices=={version}", "h5py", "--python", python],
        check=True,
        capture_output=True,
    )


def install_local_version(venv_dir: Path, repo_root: Path, git_ref: str = "HEAD") -> None:
    """Install the local checkout of climate-indices into a venv.

    Also installs h5py explicitly (same reason as install_pypi_version).

    Args:
        venv_dir: Path to the uv-managed venv.
        repo_root: Root of the climate_indices git repository.
        git_ref: Git ref to install.  Use ``"HEAD"`` for the current working tree.
    """
    python = str(venv_dir / "bin" / "python")
    if git_ref == "HEAD":
        subprocess.run(
            ["uv", "pip", "install", str(repo_root), "h5py", "--python", python],
            check=True,
            capture_output=True,
        )
    else:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                f"git+file://{repo_root}@{git_ref}",
                "h5py",
                "--python",
                python,
            ],
            check=True,
            capture_output=True,
        )


# ─── NetCDF creation utilities ────────────────────────────────────────────────


def create_fixtures_netcdf(fixtures_dir: Path, precip_nc: Path, temp_nc: Path) -> None:
    """Build single-pixel gridded NetCDFs from the .npy test fixtures.

    Produces two CF-1.8 compliant files with dimensions ``(lat=1, lon=1, time)``.
    Gridded (not timeseries) format is required so that SPEI/PET parallel workers
    can call ``_apply_along_axis_double``, which does not support the timeseries
    input type in either v2.2.0 or v2.3.0.

    Args:
        fixtures_dir: Directory containing the .npy fixture files.
        precip_nc: Output path for the precipitation NetCDF.
        temp_nc: Output path for the temperature NetCDF.
    """
    # fixture arrays are (n_years, 12) — flatten to 1-D monthly time series
    precips = np.load(str(fixtures_dir / "precips_mm_monthly.npy")).flatten()
    temps = np.load(str(fixtures_dir / "temp_celsius.npy")).flatten()

    n_months = min(len(precips), len(temps))
    time = pd.date_range(f"{FIXTURE_DATA_START_YEAR}-01-01", periods=n_months, freq="MS")

    lat = np.array([FIXTURE_LATITUDE])
    lon = np.array([-97.0])

    # reshape to (lat=1, lon=1, time) for InputType.grid
    prcp_3d = precips[:n_months].astype(np.float32)[np.newaxis, np.newaxis, :]
    tavg_3d = temps[:n_months].astype(np.float32)[np.newaxis, np.newaxis, :]

    cf_attrs: dict[str, str] = {"Conventions": "CF-1.8", "history": "Created from climate_indices test fixtures"}

    xr.Dataset(
        {
            "prcp": xr.DataArray(
                prcp_3d,
                coords={"lat": lat, "lon": lon, "time": time},
                dims=["lat", "lon", "time"],
                attrs={"units": "mm", "long_name": "Monthly Precipitation"},
            )
        },
        attrs=cf_attrs,
    ).to_netcdf(str(precip_nc), engine="scipy")

    xr.Dataset(
        {
            "tavg": xr.DataArray(
                tavg_3d,
                coords={"lat": lat, "lon": lon, "time": time},
                dims=["lat", "lon", "time"],
                attrs={"units": "celsius", "long_name": "Monthly Average Temperature"},
            )
        },
        attrs=cf_attrs,
    ).to_netcdf(str(temp_nc), engine="scipy")


def create_example_netcdf(fixtures_dir: Path, output_path: Path) -> None:
    """Create a CF-compliant 3×3 gridded example NetCDF for inclusion in the repo.

    The output has dimensions ``(lat=3, lon=3, time=360)`` covering 1981-2010,
    with variables ``prcp`` (mm), ``tavg`` (°C), and ``awc`` (inches, 2-D).
    Data is synthesised from the .npy fixtures with deterministic RNG perturbations
    so the file remains small (~100-150 KB).

    Args:
        fixtures_dir: Directory containing the .npy fixture files.
        output_path: Destination path for the NetCDF (parent dirs created if needed).
    """
    # fixture arrays are (n_years, 12) — slice by year row, then flatten
    base_precip_2d = np.load(str(fixtures_dir / "precips_mm_monthly.npy"))
    base_temp_2d = np.load(str(fixtures_dir / "temp_celsius.npy"))

    # extract 1981-2010 window (year-row indices from fixture start year 1895)
    start_year_idx = DEFAULT_CALIB_START - FIXTURE_DATA_START_YEAR
    end_year_idx = DEFAULT_CALIB_END - FIXTURE_DATA_START_YEAR + 1
    base_precip = base_precip_2d[start_year_idx:end_year_idx].flatten()
    base_temp = base_temp_2d[start_year_idx:end_year_idx].flatten()
    n_months = len(base_precip)

    time = pd.date_range(f"{DEFAULT_CALIB_START}-01-01", periods=n_months, freq="MS")
    lat = np.array([24.0, 25.0, 26.0])
    lon = np.array([-97.0, -96.0, -95.0])

    rng = np.random.default_rng(42)

    prcp_3d = np.zeros((3, 3, n_months), dtype=np.float32)
    tavg_3d = np.zeros((3, 3, n_months), dtype=np.float32)
    awc_2d = np.zeros((3, 3), dtype=np.float32)

    for i in range(3):
        for j in range(3):
            noise_scale = 1.0 + rng.uniform(-0.05, 0.05)
            prcp_3d[i, j, :] = np.maximum(0.0, base_precip * noise_scale)
            tavg_3d[i, j, :] = base_temp + rng.uniform(-0.5, 0.5)
            awc_2d[i, j] = float(4.5 + rng.uniform(-1.0, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "prcp": xr.DataArray(
                prcp_3d,
                coords={"lat": lat, "lon": lon, "time": time},
                dims=["lat", "lon", "time"],
                attrs={
                    "units": "mm",
                    "long_name": "Monthly Precipitation",
                    "standard_name": "precipitation_amount",
                },
            ),
            "tavg": xr.DataArray(
                tavg_3d,
                coords={"lat": lat, "lon": lon, "time": time},
                dims=["lat", "lon", "time"],
                attrs={
                    "units": "celsius",
                    "long_name": "Monthly Average Temperature",
                    "standard_name": "air_temperature",
                },
            ),
            "awc": xr.DataArray(
                awc_2d,
                coords={"lat": lat, "lon": lon},
                dims=["lat", "lon"],
                attrs={
                    "units": "inches",
                    "long_name": "Available Water Capacity",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "Example climate indices input data (3×3 grid, 1981-2010)",
            "source": "Synthesised from climate_indices test fixtures (seed=42)",
        },
    )

    # scipy backend does not support zlib; use dtype-only encoding.
    # the 3x3x360 dataset is ~36 KB uncompressed, well within the 150 KB target.
    encoding = {
        "prcp": {"dtype": "float32"},
        "tavg": {"dtype": "float32"},
        "awc": {"dtype": "float32"},
    }

    ds.to_netcdf(str(output_path), encoding=encoding, engine="scipy")
    size_kb = output_path.stat().st_size / 1024
    print(f"Created {output_path} ({size_kb:.1f} KB)")
    print(f"  Dimensions: lat={len(lat)}, lon={len(lon)}, time={n_months}")
    print(f"  Period: {DEFAULT_CALIB_START}-01 to {DEFAULT_CALIB_END}-12")
    print(f"  CLI usage:")
    print(
        f"    uv run process_climate_indices --index spi --periodicity monthly"
        f" --scales 6 --netcdf_precip {output_path}"
        f" --var_name_precip prcp --output_file_base /tmp/test"
        f" --calibration_start_year {DEFAULT_CALIB_START}"
        f" --calibration_end_year {DEFAULT_CALIB_END}"
    )


# ─── CLI runner ───────────────────────────────────────────────────────────────


def run_cli(
    venv_dir: Path,
    index: str,
    output_dir: Path,
    calib_start: int,
    calib_end: int,
    scales: list[int] | None = None,
    precip_nc: Path | None = None,
    var_name_precip: str = "prcp",
    temp_nc: Path | None = None,
    var_name_temp: str = "tavg",
    pet_nc: Path | None = None,
    var_name_pet: str = "pet_thornthwaite",
    timeout: int = 600,
) -> bool:
    """Run ``climate_indices`` CLI for a given index inside an isolated venv.

    Output files are written to ``output_dir/output_{var_name}.nc``.

    Args:
        venv_dir: Path to the uv-managed venv containing climate-indices.
        index: Index name as accepted by ``--index`` (``spi``, ``spei``, etc.).
        output_dir: Directory for CLI output NetCDF files.
        calib_start: Calibration period start year.
        calib_end: Calibration period end year.
        scales: Time scales (required for SPI, SPEI, PNP).
        precip_nc: Precipitation NetCDF path (not required for PET).
        var_name_precip: Variable name inside the precip NetCDF.
        temp_nc: Temperature NetCDF path (required for PET, optional for SPEI).
        var_name_temp: Variable name inside the temp NetCDF.
        pet_nc: Pre-computed PET NetCDF path (alternative to temp for SPEI).
        var_name_pet: Variable name inside the PET NetCDF.
        timeout: Subprocess timeout in seconds.

    Returns:
        ``True`` on success, ``False`` on failure or timeout.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = str(output_dir / "output")

    # Use the entry-point script (process_climate_indices) rather than
    # "python -m climate_indices".  On macOS, multiprocessing uses 'spawn'
    # by default: workers import '__main__' but with '-m', that resolves to
    # the built-in bootstrap, not climate_indices.__main__.  The entry-point
    # runs climate_indices.__main__:main, which is importable by workers.
    entry_point = venv_dir / "bin" / "process_climate_indices"
    cmd: list[str] = [
        str(entry_point),
        "--index",
        index,
        "--periodicity",
        "monthly",
        "--output_file_base",
        output_base,
        "--calibration_start_year",
        str(calib_start),
        "--calibration_end_year",
        str(calib_end),
        "--multiprocessing",
        "single",
    ]

    if scales:
        cmd += ["--scales"] + [str(s) for s in scales]

    # precipitation is required for all indices except PET
    if index != "pet" and precip_nc is not None:
        cmd += ["--netcdf_precip", str(precip_nc), "--var_name_precip", var_name_precip]

    if temp_nc is not None:
        cmd += ["--netcdf_temp", str(temp_nc), "--var_name_temp", var_name_temp]

    if pet_nc is not None:
        cmd += ["--netcdf_pet", str(pet_nc), "--var_name_pet", var_name_pet]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        return True
    except subprocess.CalledProcessError as exc:
        stderr_preview = exc.stderr.decode(errors="replace")[:300]
        print(f"  CLI error ({index}): {stderr_preview}", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"  CLI timed out ({index}, timeout={timeout}s)", file=sys.stderr)
        return False


# ─── comparison engine ────────────────────────────────────────────────────────


def find_output_file(output_dir: Path, var_name: str) -> Path | None:
    """Return the output NetCDF path for a variable, or ``None`` if absent.

    Args:
        output_dir: Directory containing CLI output files.
        var_name: Variable name, e.g. ``"spi_gamma_06"``.
    """
    path = output_dir / f"output_{var_name}.nc"
    return path if path.exists() else None


def compute_stats(arr_old: np.ndarray, arr_new: np.ndarray) -> tuple[float, float, float, int, int]:
    """Compute pairwise statistics between two arrays.

    Args:
        arr_old: Array from the old version (any shape, flattened internally).
        arr_new: Array from the new version (same shape as ``arr_old``).

    Returns:
        Tuple of ``(max_abs_diff, rmse, correlation, nan_to_val, val_to_nan)``.
        Statistical values are ``float("nan")`` when fewer than 2 shared valid
        elements exist.
    """
    flat_old = arr_old.flatten().astype(float)
    flat_new = arr_new.flatten().astype(float)

    nan_old = np.isnan(flat_old)
    nan_new = np.isnan(flat_new)

    # NaN transition counts
    nan_to_val = int(np.sum(nan_old & ~nan_new))
    val_to_nan = int(np.sum(~nan_old & nan_new))

    both_valid = ~nan_old & ~nan_new
    if both_valid.sum() < 2:
        return float("nan"), float("nan"), float("nan"), nan_to_val, val_to_nan

    diff = flat_new[both_valid] - flat_old[both_valid]
    max_abs_diff = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    old_valid = flat_old[both_valid]
    new_valid = flat_new[both_valid]
    if np.std(old_valid) < 1e-12 or np.std(new_valid) < 1e-12:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(old_valid, new_valid)[0, 1])

    return max_abs_diff, rmse, correlation, nan_to_val, val_to_nan


def classify_difference(
    index: str,
    max_abs_diff: float,
    nan_to_val: int,
    val_to_nan: int,
    arr_old: np.ndarray,
    arr_new: np.ndarray,
) -> str:
    """Classify the difference between two output arrays into one of three categories.

    Categories:
    - ``IDENTICAL``: max|diff| < 1e-10, no NaN transitions at all.
    - ``EXPECTED FIX``: SPI/SPEI index, only NaN→value transitions (zero-precip fix),
      and all shared valid values match exactly.
    - ``UNEXPECTED``: anything else.

    Args:
        index: Index name (``"spi"``, ``"spei"``, ``"pet"``, ``"pnp"``).
        max_abs_diff: Pre-computed max absolute difference on shared valid pairs.
        nan_to_val: Count of positions that were NaN in old but valid in new.
        val_to_nan: Count of positions that were valid in old but NaN in new.
        arr_old: Raw old array (used for shared-values cross-check).
        arr_new: Raw new array.

    Returns:
        One of ``DIFF_IDENTICAL``, ``DIFF_EXPECTED``, or ``DIFF_UNEXPECTED``.
    """
    if np.isnan(max_abs_diff):
        return DIFF_UNEXPECTED

    if max_abs_diff < IDENTICAL_THRESHOLD and nan_to_val == 0 and val_to_nan == 0:
        return DIFF_IDENTICAL

    # expected fix: SPI/SPEI only, NaN→value (zeros becoming drought signal),
    # no val→NaN regressions, and shared values are numerically identical
    if index in ("spi", "spei") and val_to_nan == 0 and nan_to_val > 0:
        flat_old = arr_old.flatten().astype(float)
        flat_new = arr_new.flatten().astype(float)
        both_valid = ~np.isnan(flat_old) & ~np.isnan(flat_new)
        if both_valid.sum() > 0:
            shared_max_diff = float(np.max(np.abs(flat_new[both_valid] - flat_old[both_valid])))
            if shared_max_diff < IDENTICAL_THRESHOLD:
                return DIFF_EXPECTED

    return DIFF_UNEXPECTED


def compare_outputs(
    old_dir: Path,
    new_dir: Path,
    var_name: str,
    index: str,
    scale: int | None = None,
    distribution: str | None = None,
) -> CompareResult | None:
    """Compare one variable's output NetCDF between the two versions.

    Args:
        old_dir: CLI output directory for v2.2.0.
        new_dir: CLI output directory for v2.3.0.
        var_name: NetCDF variable name, e.g. ``"spi_gamma_06"``.
        index: Index type (``"spi"``, ``"spei"``, ``"pet"``, ``"pnp"``).
        scale: Time scale (months), or ``None`` for un-scaled indices.
        distribution: Distribution name, or ``None``.

    Returns:
        A ``CompareResult``, or ``None`` if either output file is missing.
    """
    old_file = find_output_file(old_dir, var_name)
    new_file = find_output_file(new_dir, var_name)

    if old_file is None or new_file is None:
        missing = []
        if old_file is None:
            missing.append(f"old/{var_name}")
        if new_file is None:
            missing.append(f"new/{var_name}")
        print(f"  Missing output file(s): {', '.join(missing)}")
        return None

    try:
        # no engine override: auto-detect based on file format (NetCDF3 or NetCDF4)
        arr_old = xr.open_dataset(str(old_file))[var_name].values
        arr_new = xr.open_dataset(str(new_file))[var_name].values
    except Exception as exc:
        print(f"  Failed to open {var_name}: {exc}", file=sys.stderr)
        return None

    max_abs_diff, rmse, correlation, nan_to_val, val_to_nan = compute_stats(arr_old, arr_new)
    classification = classify_difference(index, max_abs_diff, nan_to_val, val_to_nan, arr_old, arr_new)

    scale_str = f"-{scale:02d}" if scale is not None else ""
    dist_str = f"/{distribution}" if distribution else ""
    label = f"{index.upper()}{scale_str}{dist_str}"

    return CompareResult(
        label=label,
        index=index,
        scale=scale,
        distribution=distribution,
        classification=classification,
        max_abs_diff=max_abs_diff,
        rmse=rmse,
        correlation=correlation,
        nan_to_val=nan_to_val,
        val_to_nan=val_to_nan,
    )


# ─── plotting ─────────────────────────────────────────────────────────────────


def plot_timeseries_comparison(
    old_dir: Path,
    new_dir: Path,
    var_name: str,
    label: str,
    output_file: Path,
    result: CompareResult,
) -> None:
    """Generate a 2-row comparison figure: overlay time series + difference.

    Row 1 — v2.2.0 (blue solid) vs v2.3.0 (red dashed) on a shared time axis.
    Row 2 — (new − old) difference with y=0 baseline and stats annotation.

    Args:
        old_dir: Output directory for v2.2.0.
        new_dir: Output directory for v2.3.0.
        var_name: Variable name (used to locate output NetCDF files).
        label: Human-readable label for the figure title.
        output_file: Destination PNG path.
        result: Pre-computed comparison statistics.
    """
    old_file = find_output_file(old_dir, var_name)
    new_file = find_output_file(new_dir, var_name)
    if old_file is None or new_file is None:
        return

    try:
        arr_old = xr.open_dataset(str(old_file))[var_name].values.flatten().astype(float)
        arr_new = xr.open_dataset(str(new_file))[var_name].values.flatten().astype(float)
    except Exception as exc:
        print(f"  Plot failed for {var_name}: {exc}", file=sys.stderr)
        return

    time_idx = np.arange(len(arr_old))
    diff = arr_new - arr_old
    diff_color = COLOR_MAP.get(result.classification, "#999999")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"{label}  [{result.classification}]  —  v{VERSION_OLD} vs v{VERSION_NEW}",
        fontsize=12,
        fontweight="bold",
    )
    fig.patch.set_facecolor("#f8f8f8")

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # row 1: overlay
    ax1.plot(time_idx, arr_old, color="steelblue", linewidth=0.7, label=f"v{VERSION_OLD} (old)", alpha=0.9)
    ax1.plot(time_idx, arr_new, color="tomato", linewidth=0.7, linestyle="--", label=f"v{VERSION_NEW} (new)", alpha=0.9)
    ax1.axhline(0, color="gray", linewidth=0.4, linestyle=":")
    ax1.set_ylabel("Index Value", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # row 2: difference
    ax2.plot(time_idx, diff, color=diff_color, linewidth=0.7)
    ax2.fill_between(time_idx, diff, 0, where=~np.isnan(diff), alpha=0.25, color=diff_color)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Δ new − old", fontsize=9)
    ax2.set_xlabel("Month index", fontsize=9)

    corr_str = f"{result.correlation:.4f}" if not np.isnan(result.correlation) else "nan"
    rmse_str = f"{result.rmse:.2e}" if not np.isnan(result.rmse) else "nan"
    stats = (
        f"max|Δ|={result.max_abs_diff:.2e}  RMSE={rmse_str}  r={corr_str}"
        f"  NaN→val={result.nan_to_val}  val→NaN={result.val_to_nan}"
    )
    ax2.set_title(stats, fontsize=8, pad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_file), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(results: list[CompareResult], output_file: Path) -> None:
    """Generate a color-coded summary grid of all comparison results.

    Each row corresponds to one index/scale/distribution combination.
    Background color encodes IDENTICAL (green), EXPECTED (amber), UNEXPECTED (red).

    Args:
        results: List of comparison results, one per index variant.
        output_file: Destination PNG path.
    """
    if not results:
        return

    n = len(results)
    fig_height = max(4.0, n * 0.45 + 2.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    fig.suptitle(
        f"Regression Report: v{VERSION_OLD} (PyPI) vs v{VERSION_NEW} (local)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    headers = ["Index / Scale / Dist", "Classification", "max|Δ|", "RMSE", "r", "NaN→val", "val→NaN"]
    col_widths = [0.26, 0.18, 0.12, 0.12, 0.10, 0.10, 0.10]
    col_x: list[float] = [0.01]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    y_top = 0.92
    row_h = 0.88 / (n + 1)

    # header row
    for header, x in zip(headers, col_x):
        ax.text(x, y_top, header, fontsize=9, fontweight="bold", transform=ax.transAxes, va="top")

    for row_idx, res in enumerate(results):
        y = y_top - (row_idx + 1) * row_h
        bg_color = COLOR_MAP.get(res.classification, "#cccccc")

        rect = plt.Rectangle(
            (0, y - row_h * 0.05),
            1.0,
            row_h * 0.9,
            transform=ax.transAxes,
            facecolor=bg_color,
            alpha=0.20,
            edgecolor="none",
        )
        ax.add_patch(rect)

        max_diff_str = f"{res.max_abs_diff:.2e}" if not np.isnan(res.max_abs_diff) else "nan"
        rmse_str = f"{res.rmse:.2e}" if not np.isnan(res.rmse) else "nan"
        corr_str = f"{res.correlation:.4f}" if not np.isnan(res.correlation) else "nan"
        row_values = [
            res.label,
            res.classification,
            max_diff_str,
            rmse_str,
            corr_str,
            str(res.nan_to_val),
            str(res.val_to_nan),
        ]
        for val, x in zip(row_values, col_x):
            ax.text(x, y + row_h * 0.45, val, fontsize=8, transform=ax.transAxes, va="center")

    counts: dict[str, int] = {DIFF_IDENTICAL: 0, DIFF_EXPECTED: 0, DIFF_UNEXPECTED: 0}
    for res in results:
        counts[res.classification] = counts.get(res.classification, 0) + 1

    summary = (
        f"Total: {n}  |  "
        f"IDENTICAL: {counts[DIFF_IDENTICAL]}  |  "
        f"EXPECTED FIX: {counts[DIFF_EXPECTED]}  |  "
        f"UNEXPECTED: {counts[DIFF_UNEXPECTED]}"
    )
    ax.text(
        0.5,
        0.02,
        summary,
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_file), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard saved: {output_file}")


# ─── console report ───────────────────────────────────────────────────────────


def print_report(results: list[CompareResult]) -> int:
    """Print a formatted console regression report.

    Args:
        results: All comparison results.

    Returns:
        Exit code: 0 if no UNEXPECTED differences, 1 otherwise.
    """
    print()
    print(f"=== Regression Report: v{VERSION_OLD} (PyPI) vs v{VERSION_NEW} (local) ===")

    for res in results:
        max_diff_str = f"{res.max_abs_diff:.2e}" if not np.isnan(res.max_abs_diff) else "     nan"
        corr_str = f"{res.correlation:.3f}" if not np.isnan(res.correlation) else "   nan"
        print(
            f"  {res.label:<22} : {res.classification:<14}"
            f"  max|diff|={max_diff_str}  r={corr_str}  NaN→val={res.nan_to_val}"
        )

    counts: dict[str, int] = {DIFF_IDENTICAL: 0, DIFF_EXPECTED: 0, DIFF_UNEXPECTED: 0}
    for res in results:
        counts[res.classification] = counts.get(res.classification, 0) + 1

    n = len(results)
    print(
        f"\nSUMMARY: {n} checks: "
        f"{counts[DIFF_IDENTICAL]} IDENTICAL, "
        f"{counts[DIFF_EXPECTED]} EXPECTED, "
        f"{counts[DIFF_UNEXPECTED]} UNEXPECTED"
    )
    return 0 if counts[DIFF_UNEXPECTED] == 0 else 1


# ─── orchestration helpers ────────────────────────────────────────────────────


def setup_environments(
    tmpdir: Path,
    repo_root: Path,
    local_ref: str,
    skip_install: bool,
) -> tuple[Path, Path]:
    """Create and populate both virtual environments.

    Args:
        tmpdir: Working directory for venvs and intermediate files.
        repo_root: Root of the climate_indices repository.
        local_ref: Git ref for the new version (``"HEAD"`` = current working tree).
        skip_install: If ``True`` and venvs already exist, skip creation.

    Returns:
        Tuple of ``(venv_old, venv_new)`` paths.
    """
    venv_old = tmpdir / "venv_v220"
    venv_new = tmpdir / "venv_v230"

    if skip_install and venv_old.exists() and venv_new.exists():
        print("  Skipping venv creation (--skip-install).")
        return venv_old, venv_new

    print(f"  Creating venv for v{VERSION_OLD} (PyPI)...")
    create_venv(venv_old)
    install_pypi_version(venv_old, VERSION_OLD)

    print(f"  Creating venv for v{VERSION_NEW} (local ref={local_ref})...")
    create_venv(venv_new)
    install_local_version(venv_new, repo_root, local_ref)

    return venv_old, venv_new


def resolve_input_data(
    args: argparse.Namespace,
    tmpdir: Path,
    repo_root: Path,
) -> tuple[Path, Path]:
    """Resolve input NetCDF files using the configured priority order.

    Priority:
    1. ``--input-data`` directory (user-supplied nClimGrid-style files).
    2. Auto-clone ``monocongo/example_climate_indices`` (depth=1).
    3. ``--use-fixtures`` mode: synthesise NetCDF from ``.npy`` test fixtures.

    Args:
        args: Parsed CLI arguments.
        tmpdir: Temp working directory.
        repo_root: Repository root (for locating fixture files).

    Returns:
        Tuple of ``(precip_nc, temp_nc)`` paths.
    """
    # priority 1: user-supplied directory
    if args.input_data:
        input_dir = Path(args.input_data)
        precip_files = sorted(input_dir.glob("*prcp*.nc")) + sorted(input_dir.glob("*precip*.nc"))
        temp_files = sorted(input_dir.glob("*tavg*.nc")) + sorted(input_dir.glob("*temp*.nc"))
        if precip_files and temp_files:
            print(f"  Using user-supplied data from {input_dir}")
            return precip_files[0], temp_files[0]
        print(f"  Warning: no matching NetCDFs found in {input_dir}, falling back.", file=sys.stderr)

    # priority 2: fixture mode (explicit flag)
    if args.use_fixtures:
        print("  Synthesising input NetCDF from .npy test fixtures...")
        fixtures_dir = repo_root / "tests" / "fixture"
        precip_nc = tmpdir / "fixtures_precip.nc"
        temp_nc = tmpdir / "fixtures_temp.nc"
        create_fixtures_netcdf(fixtures_dir, precip_nc, temp_nc)
        return precip_nc, temp_nc

    # priority 3: auto-clone example_climate_indices
    example_dir = tmpdir / "example_climate_indices"
    if not example_dir.exists():
        print("  Cloning monocongo/example_climate_indices (depth=1)...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/monocongo/example_climate_indices.git",
                    str(example_dir),
                ],
                check=True,
                capture_output=True,
                timeout=90,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"  Clone failed ({exc}), falling back to --use-fixtures mode.", file=sys.stderr)
            fixtures_dir = repo_root / "tests" / "fixture"
            precip_nc = tmpdir / "fixtures_precip.nc"
            temp_nc = tmpdir / "fixtures_temp.nc"
            create_fixtures_netcdf(fixtures_dir, precip_nc, temp_nc)
            return precip_nc, temp_nc

    input_dir = example_dir / "example" / "input"
    # prefer nclimgrid monthly files over other datasets (e.g. daily cmorph)
    precip_files = sorted(input_dir.glob("nclimgrid*prcp*.nc")) or sorted(input_dir.glob("*prcp*.nc"))
    temp_files = sorted(input_dir.glob("nclimgrid*tavg*.nc")) or sorted(input_dir.glob("*tavg*.nc"))
    if precip_files and temp_files:
        print(f"  Using cloned example data from {input_dir}")
        return precip_files[0], temp_files[0]

    # final fallback to fixtures
    print("  Could not locate example data, falling back to fixture mode.", file=sys.stderr)
    fixtures_dir = repo_root / "tests" / "fixture"
    precip_nc = tmpdir / "fixtures_precip.nc"
    temp_nc = tmpdir / "fixtures_temp.nc"
    create_fixtures_netcdf(fixtures_dir, precip_nc, temp_nc)
    return precip_nc, temp_nc


def run_all_indices(
    venv: Path,
    output_dir: Path,
    precip_nc: Path,
    temp_nc: Path,
    calib_start: int,
    calib_end: int,
    indices_to_run: list[str],
) -> dict[str, bool]:
    """Run the full set of requested indices for one venv.

    PET is always computed first when SPEI or PET are requested, because
    SPEI requires a PET input file.

    Args:
        venv: Path to the venv containing the climate-indices version to test.
        output_dir: Directory where CLI output NetCDFs are written.
        precip_nc: Precipitation NetCDF.
        temp_nc: Temperature NetCDF.
        calib_start: Calibration period start year.
        calib_end: Calibration period end year.
        indices_to_run: Subset of ``["spi", "spei", "pet", "pnp"]``.

    Returns:
        Dictionary mapping index name to success flag.
    """
    # clear any stale output files from previous runs to avoid false comparisons
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    # compute PET first (needed both for standalone comparison and as SPEI input)
    pet_needed = any(i in indices_to_run for i in ("spei", "pet"))
    pet_nc: Path | None = None
    if pet_needed:
        ok = run_cli(
            venv,
            "pet",
            output_dir,
            calib_start,
            calib_end,
            temp_nc=temp_nc,
        )
        if "pet" in indices_to_run:
            results["pet"] = ok
        if ok:
            pet_nc = output_dir / "output_pet_thornthwaite.nc"

    if "spi" in indices_to_run:
        ok = run_cli(
            venv,
            "spi",
            output_dir,
            calib_start,
            calib_end,
            scales=SPI_SCALES,
            precip_nc=precip_nc,
        )
        results["spi"] = ok

    if "spei" in indices_to_run:
        if pet_nc is not None and pet_nc.exists():
            ok = run_cli(
                venv,
                "spei",
                output_dir,
                calib_start,
                calib_end,
                scales=SPEI_SCALES,
                precip_nc=precip_nc,
                pet_nc=pet_nc,
            )
        else:
            print("  Skipping SPEI: PET output unavailable.", file=sys.stderr)
            ok = False
        results["spei"] = ok

    if "pnp" in indices_to_run:
        ok = run_cli(
            venv,
            "pnp",
            output_dir,
            calib_start,
            calib_end,
            scales=PNP_SCALES,
            precip_nc=precip_nc,
        )
        results["pnp"] = ok

    return results


def collect_comparisons(
    old_dir: Path,
    new_dir: Path,
    plot_dir: Path,
    indices_to_compare: list[str],
) -> list[CompareResult]:
    """Run all comparisons and save per-index time-series plots.

    Args:
        old_dir: CLI output directory for v2.2.0.
        new_dir: CLI output directory for v2.3.0.
        plot_dir: Directory for individual comparison PNG files.
        indices_to_compare: Subset of index names to compare.

    Returns:
        Flat list of ``CompareResult`` objects.
    """
    all_results: list[CompareResult] = []

    if "spi" in indices_to_compare:
        for scale in SPI_SCALES:
            for dist in DISTRIBUTIONS:
                var_name = f"spi_{dist}_{scale:02d}"
                result = compare_outputs(old_dir, new_dir, var_name, "spi", scale, dist)
                if result:
                    all_results.append(result)
                    plot_timeseries_comparison(
                        old_dir, new_dir, var_name, result.label, plot_dir / f"{var_name}.png", result
                    )

    if "spei" in indices_to_compare:
        for scale in SPEI_SCALES:
            for dist in DISTRIBUTIONS:
                var_name = f"spei_{dist}_{scale:02d}"
                result = compare_outputs(old_dir, new_dir, var_name, "spei", scale, dist)
                if result:
                    all_results.append(result)
                    plot_timeseries_comparison(
                        old_dir, new_dir, var_name, result.label, plot_dir / f"{var_name}.png", result
                    )

    if "pet" in indices_to_compare:
        var_name = "pet_thornthwaite"
        result = compare_outputs(old_dir, new_dir, var_name, "pet", None, None)
        if result:
            all_results.append(result)
            plot_timeseries_comparison(
                old_dir, new_dir, var_name, result.label, plot_dir / f"{var_name}.png", result
            )

    if "pnp" in indices_to_compare:
        for scale in PNP_SCALES:
            var_name = f"pnp_{scale:02d}"
            result = compare_outputs(old_dir, new_dir, var_name, "pnp", scale, None)
            if result:
                all_results.append(result)
                plot_timeseries_comparison(
                    old_dir, new_dir, var_name, result.label, plot_dir / f"{var_name}.png", result
                )

    return all_results


# ─── CLI argument parser ──────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the script's argument parser.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description="Visual regression: v2.2.0 (PyPI) vs v2.3.0 (local) climate indices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Root directory of the climate_indices repository (default: current dir).",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output",
        help="Directory for PNGs and the console report (default: comparison_output/).",
    )
    parser.add_argument(
        "--local-ref",
        default="HEAD",
        help="Git ref for the new version to install locally (default: HEAD).",
    )
    parser.add_argument(
        "--input-data",
        metavar="DIR",
        help="Directory containing nClimGrid-style NetCDFs (*prcp*.nc, *tavg*.nc).",
    )
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        help="Synthesise input NetCDF from .npy test fixtures (fast, single-pixel).",
    )
    parser.add_argument(
        "--create-example-netcdf",
        metavar="OUTPUT_DIR",
        help="Create example_nclimgrid_lowres.nc in OUTPUT_DIR and exit.",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=["spi", "spei", "pet", "pnp"],
        default=["spi", "spei", "pet", "pnp"],
        help="Indices to compare (default: all).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Reuse existing venvs in --tmp-dir (skip uv venv / uv pip install).",
    )
    parser.add_argument(
        "--calibration-start-year",
        type=int,
        default=DEFAULT_CALIB_START,
        help=f"Calibration period start year (default: {DEFAULT_CALIB_START}).",
    )
    parser.add_argument(
        "--calibration-end-year",
        type=int,
        default=DEFAULT_CALIB_END,
        help=f"Calibration period end year (default: {DEFAULT_CALIB_END}).",
    )
    parser.add_argument(
        "--tmp-dir",
        metavar="DIR",
        help="Persistent temp directory for venvs and intermediate outputs. "
        "If omitted a system tempdir is used and cleaned up on exit.",
    )
    return parser


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    """Script entry point.

    Returns:
        Exit code: 0 if all comparisons are IDENTICAL or EXPECTED FIX, 1 otherwise.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # early exit: create example NetCDF and stop
    if args.create_example_netcdf:
        example_out_dir = Path(args.create_example_netcdf)
        example_out_dir.mkdir(parents=True, exist_ok=True)
        fixtures_dir = repo_root / "tests" / "fixture"
        create_example_netcdf(fixtures_dir, example_out_dir / "example_nclimgrid_lowres.nc")
        return 0

    # set up temp working directory
    if args.tmp_dir:
        tmpdir = Path(args.tmp_dir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        cleanup_tmp = False
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="climate_compare_"))
        cleanup_tmp = True

    try:
        print("\n[1/5] Setting up virtual environments...")
        venv_old, venv_new = setup_environments(tmpdir, repo_root, args.local_ref, args.skip_install)

        print("\n[2/5] Resolving input data...")
        precip_nc, temp_nc = resolve_input_data(args, tmpdir, repo_root)
        print(f"  precip: {precip_nc}")
        print(f"  temp:   {temp_nc}")

        calib_start = args.calibration_start_year
        calib_end = args.calibration_end_year

        old_out = tmpdir / "output_v220"
        print(f"\n[3/5] Running v{VERSION_OLD} (PyPI) indices: {args.indices}")
        run_all_indices(venv_old, old_out, precip_nc, temp_nc, calib_start, calib_end, args.indices)

        new_out = tmpdir / "output_v230"
        print(f"\n[4/5] Running v{VERSION_NEW} (local) indices: {args.indices}")
        run_all_indices(venv_new, new_out, precip_nc, temp_nc, calib_start, calib_end, args.indices)

        print("\n[5/5] Comparing outputs and generating plots...")
        plot_dir = output_dir / "plots"
        all_results = collect_comparisons(old_out, new_out, plot_dir, args.indices)

        if all_results:
            plot_summary_dashboard(all_results, output_dir / "summary_dashboard.png")

        exit_code = print_report(all_results)
        print(f"\nAll outputs written to: {output_dir}")
        return exit_code

    finally:
        if cleanup_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

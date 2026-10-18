"""Freeze pure-Python/SciPy distribution regressions for standalone Rust tests.

Run from the repository root with PYTHONPATH=$PWD/src. Cargo never runs this script.
See crates/climate-core/tests/fixtures/distribution_reference.md for the reference scope.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

# Block the extension before any climate_indices import, even if it is installed.
sys.modules["climate_indices._native"] = None  # type: ignore[assignment]
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/climate_indices"
assert str(ROOT / "src") in sys.path, "Set PYTHONPATH=$PWD/src from the repository root."

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import special, stats  # noqa: E402

from climate_indices import compute, indices, lmoments  # noqa: E402

for module in (compute, indices, lmoments):
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == SOURCE / f"{module.__name__.split('.')[-1]}.py"
assert compute._native is None
assert sys.modules.get("climate_indices._native") is None

SOURCE_REVISION = "f4cb938eb622a3ddf8b96c129337e0ebecd4121e"
DESTINATION = ROOT / "crates/climate-core/tests/fixtures"
COMMAND = "PYTHONPATH=src uv run python scripts/prepare_rust_distribution_fixtures.py"


def save(name: str, arrays: dict[str, np.ndarray], units: str) -> None:
    """Write named C-order matrices at round-trip float64 precision."""
    lines = [
        f"# Source revision: {SOURCE_REVISION}",
        f"# Generation command: {COMMAND}",
        "# Classification: Python/SciPy regression, not independent scientific validation. No Rust output is used.",
        f"# Runtime: Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}",
        f"# Units: {units}",
        "# Shapes: NAME ROWS COLUMNS followed by ROWS numeric lines, C order; nan, inf, -inf are explicit.",
        "# Tolerance: rtol=atol=1e-10; exact NaN masks and signed infinities.",
        "# Sources and case descriptions: distribution_reference.md.",
    ]
    for key, values in arrays.items():
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim < 2:
            matrix = matrix.reshape(1, -1)
        assert matrix.ndim == 2 and matrix.size > 0, (key, matrix.shape)
        lines.append(f"{key} {matrix.shape[0]} {matrix.shape[1]}")
        lines.extend(" ".join(format(value, ".17g") for value in row) for row in matrix)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / name).write_text("\n".join(lines) + "\n")


def fitting() -> None:
    """Cover symmetric, both skew signs, zero mass, and invalid sample columns."""
    skewed = np.array([1, 1, 1, 2, 2, 3, 4, 8, 20, 60], dtype=float)
    calibration = np.column_stack(
        [
            [12, 15.5, 9.1, 30.2, 22.4, 18.8, 11.3, 40.7, 25, 14.2],
            skewed,
            100 - skewed,
            np.arange(1, 11),
            [0, -0.0, 0, 3, 5, 4, 9, np.nan, 6, 8],
            [1, 2, 3, *([np.nan] * 7)],
            np.full(10, 2.0),
            np.full(10, np.nan),
            np.zeros(10),
            np.arange(1, 11) + np.array([0.0] * 9 + [0.001]),
            [-9, -6, -4, -3, -2, -1, 1, 2, 3, 7],
            skewed / 5.0,
        ]
    )
    alphas, betas = compute.gamma_parameters(calibration.copy(), 2000, 2000, 2009, compute.Periodicity.monthly)
    p0, locs, scales, skews, failed = compute._pearson_parameters_spatial(calibration)
    _, _, _, valid = lmoments.fit_spatial(calibration)
    valid &= np.count_nonzero(~np.isnan(calibration) & (calibration != 0), axis=0) >= 4
    assert failed == np.count_nonzero(~valid)
    glo_loc, glo_scale, glo_shape, glo_valid = lmoments.fit_glo_spatial(calibration)
    assert valid.tolist() == [True] * 5 + [False] * 4 + [True] * 3
    assert glo_valid.tolist() == valid.tolist()
    assert np.isposinf(alphas[6]) and betas[6] == 0
    assert skews[2] < 0 < skews[1] and skews[3] == 0
    save(
        "distribution_fits.txt",
        {
            "calibration": calibration,
            "gamma_alpha": alphas,
            "gamma_beta": betas,
            "pearson_p0": p0,
            "pearson_loc": locs,
            "pearson_scale": scales,
            "pearson_skew": skews,
            "pearson_valid": valid,
            "glo_loc": glo_loc,
            "glo_scale": glo_scale,
            "glo_shape": glo_shape,
            "glo_valid": glo_valid,
        },
        "calibration/location/scale/beta: mm; alpha/skew/GLO shape/p0: dimensionless; validity: 0/1",
    )


def cdfs() -> None:
    """Freeze SciPy scalar/block CDFs and the Python GLO and zero-mass steps."""
    gamma_cases = np.array(
        [
            [0.2, 0.5, 1],
            [5, 2, 2],
            [1.05, 0.5, 1],
            [55, 50, 1],
            [520, 500, 1],
            [330, 250, 1],
            [240, 180, 1],
            [30, 5, 1],
            [1e-12, 0.01, 0.25],
            [1e4, 2000, 5],
            [0, 2, 3],
            [-1, 2, 3],
            [np.inf, 2, 3],
            [-np.inf, 2, 3],
            [np.nan, 2, 3],
            [1, np.nan, 3],
            [1, 0, 3],
            [1, -1, 3],
            [1, 2, 0],
            [1, 2, -1],
            [1, 2, np.nan],
            [1, np.inf, 1],
            [1, 2, np.inf],
            [np.inf, 2, np.inf],
        ],
        dtype=float,
    )
    x, alpha, beta = gamma_cases.T
    arrays = {
        "gamma_cases": gamma_cases,
        "gamma_cdf": stats.gamma.cdf(x, a=alpha, scale=beta),
        "gamma_sf": stats.gamma.sf(x, a=alpha),
    }
    values = np.array([[0, -0.0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6], [-1] * 6, [np.inf] * 6, [np.nan] * 6])
    alpha = np.array([2, 0.5, np.nan, 0, 2, 3])
    beta = np.array([1, 2, 1, 1, 0, 4])
    p0 = np.array([0, 0.25, 0.5, 0, 1, np.nan])
    # This is compute.transform_fitted_gamma's probability step, before output scaling.
    probabilities = p0 + (1 - p0) * np.where(values == 0, 0, stats.gamma.cdf(values, a=alpha, scale=beta))
    arrays.update(
        gamma_values=values, gamma_alpha=alpha, gamma_beta=beta, gamma_p0=p0, gamma_probabilities=probabilities
    )

    z = np.array([-np.inf, -40, -8, -3, -1, 0, 1, 3, 8, 40, np.inf, np.nan])[:, None]
    # 1e-4 and its mirror pin SciPy's 1.6e-5 normal-transition threshold: their
    # Pearson values differ from the normal CDF by more than the parity tolerance.
    skews = np.array([0.8, -0.8, 0, 1e-6, 1.5, -1.5, np.nan, np.inf, 0.8, 1e-4, -1e-4])
    locs, scales = np.full(11, 3.0), np.array([2.0] * 10 + [0.0])
    values = locs + z * scales
    boundary = locs - 2 * scales / skews
    values = np.vstack([values, boundary - 1e-8, boundary, boundary + 1e-8])
    arrays.update(
        pearson_values=values,
        pearson_loc=locs,
        pearson_scale=scales,
        pearson_skew=skews,
        pearson_cdf=stats.pearson3.cdf(values, skews, loc=locs, scale=scales),
    )
    ks_values = np.column_stack(
        [
            [0, 1, 2, np.nan, 3, 5, 9],
            [1, 1, 2, 2, 3, 4, 8],
            [1, 2, 3, 4, 5, 6, 7],
            np.zeros(7),
            np.full(7, np.nan),
            [2, *([np.nan] * 6)],
            np.arange(1, 8),
        ]
    )
    ks_skew, ks_loc, ks_scale = np.array([0, 0.8, -1.5, 0, 0, 0.8, 0]), np.full(7, 3.0), np.array([2.0] * 6 + [0.0])
    arrays.update(
        ks_values=ks_values,
        ks_skew=ks_skew,
        ks_loc=ks_loc,
        ks_scale=ks_scale,
        pearson_ks=compute._pearson_ks_statistics(ks_values, ks_loc, ks_scale, ks_skew),
    )

    # 1e-4 pins the 1e-6 logistic-shape tolerance: it takes the general branch,
    # where returning the logistic value would exceed the parity tolerance.
    shapes = np.array([-0.75, 0, 0.75, 1e-7, -1e-7, 1e-6, -1e-6, np.nan, 1e-4])
    locs, scales = np.full(9, 3.0), np.full(9, 2.0)
    values = locs + z * scales
    boundary = locs + scales / shapes
    values = np.vstack([values, boundary - 1e-8, boundary, boundary + 1e-8])
    arrays.update(
        glo_values=values,
        glo_loc=locs,
        glo_scale=scales,
        glo_shape=shapes,
        glo_cdf=compute._loglogistic_fit(values, locs, scales, shapes, output_scale="probability"),
    )
    # The unmasked `cdfglo` formula, which `compute._loglogistic_fit` applies before
    # it masks invalid parameters to NaN. Rust's `loglogistic_cdf` must reproduce
    # it for a non-positive scale and for |shape| >= 1, so these probes stay bare.
    # A zero scale needs value != loc: an equal value gives 0/0 = NaN, not the
    # infinite z the contract covers.
    bare_values = np.array([4.0, 4.0, 2.0, 3.0, 5.0, 1.0, 0.0])
    bare_locs = np.full(7, 3.0)
    bare_scales = np.array([0.0, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    bare_shapes = np.array([0.5, 0.5, 1.0, -1.0, 0.5, 0.5, 0.5])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        bare_z = (bare_values - bare_locs) / bare_scales
        bare_y = np.where(
            np.isclose(bare_shapes, 0.0),
            bare_z,
            -np.log(np.maximum(0.0, 1.0 - bare_shapes * bare_z)) / bare_shapes,
        )
        bare = np.clip(1.0 / (1.0 + np.exp(-bare_y)), 0.0, 1.0)
    arrays.update(
        glo_bare_values=bare_values,
        glo_bare_loc=bare_locs,
        glo_bare_scale=bare_scales,
        glo_bare_shape=bare_shapes,
        glo_bare_cdf=bare,
    )
    assert arrays["pearson_cdf"][0, 1] == 0 and arrays["pearson_cdf"][10, 1] == 1
    assert arrays["glo_cdf"][13, 0] == 0 and arrays["glo_cdf"][13, 2] == 1
    # a non-positive scale and |shape| >= 1 are bare-formula cases, not NaN
    assert arrays["glo_bare_cdf"].size == 7 and np.isfinite(arrays["glo_bare_cdf"]).all()
    save(
        "distribution_cdfs.txt",
        arrays,
        "values/location/scale/beta: mm; gamma sf x: unit-scale; all other fields: dimensionless",
    )


def special_functions() -> None:
    """Exercise incomplete-gamma branches and both normal tails with SciPy."""
    cases = np.array(
        [
            [0.5, 0.2],
            [2, 5],
            [0.5, 1.05],
            [50, 55],
            [500, 520],
            [250, 330],
            [180, 240],
            [1e-3, 1e-4],
            [0.01, 0.4],
            [3, 0.8],
            [1000, 1000],
            [1e6, 1e6 + 100],
            [5, 30],
            [1, 0],
            [0, 1],
            [0, 0],
            [-1, 1],
            [1, -1],
            [np.nan, 1],
            [1, np.nan],
            [np.inf, 1],
            [1, np.inf],
            [np.inf, np.inf],
        ],
        dtype=float,
    )
    quantiles = np.array(
        [
            -np.inf,
            -0.1,
            0,
            np.nextafter(0.0, 1.0),
            1e-300,
            1e-30,
            1e-10,
            0.025,
            np.exp(-2),
            0.25,
            0.5,
            0.75,
            0.975,
            1 - 1e-10,
            np.nextafter(1.0, 0.0),
            1,
            1.1,
            np.inf,
            np.nan,
        ]
    )
    x = np.array([-np.inf, -40, -38, -10, -8, -3, -1, -0.1, 0, 0.1, 1, 3, 8, 10, 38, 40, np.inf, np.nan])
    save(
        "distribution_special.txt",
        {
            "igamma_cases": cases,
            "igam": special.gammainc(cases[:, 0], cases[:, 1]),
            "igamc": special.gammaincc(cases[:, 0], cases[:, 1]),
            "normal_x": x,
            "ndtr": special.ndtr(x),
            "quantiles": quantiles,
            "ndtri": special.ndtri(quantiles),
            "norm_ppf": stats.norm.ppf(quantiles),
        },
        "all fields: dimensionless",
    )


def simple_indices() -> None:
    """Freeze EDDI rank steps, PNP normals/ratios, and PCI from Python."""
    climatology = np.column_stack(
        [
            [0, 1, 2, 2, 4, np.nan, 8],
            np.full(7, 2.0),
            np.full(7, np.nan),
            [3, *([np.nan] * 6)],
            [0, 0, 1, 2, 4, 5, 9],
        ]
    )
    values = np.array(
        [
            [0, 2, 1, 3, 0],
            [2, 1, np.nan, 0, 1],
            [3, 3, 0, 4, 2],
            [20, 20, 20, 20, 20],
            [-1, -1, -1, -1, -1],
            [np.nan] * 5,
        ]
    )
    pads = np.array([0, 1, 2, 0, 1])
    # Retain indices.eddi's NumPy rank-count/probability step before clipping.
    counts = np.count_nonzero(~np.isnan(climatology), axis=0)
    below = np.count_nonzero(climatology[:, None, :] < values[None, :, :], axis=0)
    probabilities = (pads + below + 0.66) / (counts + pads + 0.33)
    probabilities = np.where(np.isnan(values) | (counts < 2), np.nan, probabilities)
    hastings_p = np.array(
        [-np.inf, -1, 0, 1e-300, 1e-10, 0.025, 0.25, 0.5, 0.75, 0.975, 1 - 1e-10, 1, 2, np.inf, np.nan]
    )
    arrays = {
        "eddi_climatology": climatology,
        "eddi_values": values,
        "eddi_pads": pads,
        "eddi_probabilities": probabilities,
        "hastings_p": hastings_p,
        "hastings_z": indices._hastings_inverse_normal(hastings_p),
    }

    time = np.arange(65)
    values = np.column_stack([1.0 + (time * 7) % 23, np.zeros(65), np.full(65, np.nan), np.full(65, 8.0)])
    values[time % 12 == 0, 0] = -2
    values[(time % 12 == 2) & (time >= 12), 0] = np.nan
    values[time % 12 == 3, 0] = np.nan
    calibration = values[:60].reshape(5, 48)
    counts = np.count_nonzero(~np.isnan(calibration), axis=0)
    normals = np.nansum(calibration, axis=0) / np.maximum(counts, 1)
    normals = np.where((counts > 0) & (normals > 0), normals, np.nan)
    ratios = indices.percentage_of_normal(
        values.reshape(65, 2, 2),
        1,
        2000,
        2000,
        2004,
        compute.Periodicity.monthly,
        spatial_time_major=True,
    ).reshape(65, 4)
    np.testing.assert_allclose(ratios, values / normals.reshape(12, 4)[time % 12], rtol=0, atol=0, equal_nan=True)
    arrays.update(pnp_values=values, pnp_calibration=calibration, pnp_normals=normals, pnp_ratios=ratios)

    for days in [365, 366]:
        rain = np.array([((day * 13) % 41) / 7.0 for day in range(days)])
        february = np.zeros(days)
        february[58:61] = [1, 2, 4]
        for mode, values in {
            "varied": rain,
            "uniform": np.ones(days),
            "zero": np.zeros(days),
            "february": february,
        }.items():
            key = f"pci_{days}_{mode}"
            arrays[key] = values
            arrays[f"{key}_expected"] = indices.pci(values)
    assert probabilities[3, 0] > 1  # The bare Tukey step does not clip out-of-calibration extremes.
    save(
        "distribution_indices.txt",
        arrays,
        "rain/PNP/EDDI values and climatology/normals: mm; pads: count; ratios/PCI/probabilities/z: dimensionless (PNP is a ratio, not multiplied by 100)",
    )


def main() -> None:
    """Refuse changed source code and generate without native dispatch."""
    subprocess.run(["git", "diff", "--quiet", SOURCE_REVISION, "--", "src/climate_indices"], cwd=ROOT, check=True)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        fitting()
        cdfs()
        special_functions()
        simple_indices()


if __name__ == "__main__":
    main()

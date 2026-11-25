"""Reproduce the numbers in xclim-2450-palmer-comparison.md.

Loads only xclim PR #2450's ``_palmer.py`` (not the fork) so that a released
xclim supplies its single dependency, ``xclim.core.calendar.compare_offsets``::

    uv venv --python 3.12 /tmp/xvenv
    uv pip install --python /tmp/xvenv/bin/python 'climate-indices==3.3.0' 'xclim==1.0.1'
    mkdir -p /tmp/x2450
    gh api 'repos/Ouranosinc/xclim/contents/src/xclim/compute/_palmer.py?ref=91fcf88a76829fd9d41ad848bde4c7faa62645de' \\
        -H 'Accept: application/vnd.github.raw' > /tmp/x2450/_palmer.py
    /tmp/xvenv/bin/python -I docs/audits/xclim_2450_palmer_compare.py . /tmp/x2450/_palmer.py

Review the downloaded file before running it: this script executes it.
"""

import hashlib
import json
import logging
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import xarray as xr

from climate_indices.drought import palmer as ci

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))
warnings.filterwarnings("ignore")

FIXTURES = Path(sys.argv[1]) / "tests" / "fixture"
SOURCE = Path(sys.argv[2]).read_text()
KEYS = ("pdsi", "phdi", "pmdi", "zindex")
MM = 25.4

IDS = json.loads((FIXTURES / "nclimdiv/divisions.json").read_text())
AWC = np.array([json.loads((FIXTURES / "palmer_awc.json").read_text())[d] for d in IDS], float)
PRECIP = np.stack([np.load(FIXTURES / f"palmer/{d}/precips.npy").reshape(-1) for d in IDS], 1)
PET = np.stack([np.load(FIXTURES / f"palmer/{d}/pet.npy").reshape(-1) for d in IDS], 1)
MONTHS = PRECIP.shape[0]
NCEI = {k: np.load(FIXTURES / f"nclimdiv/{k}.npy").T[:MONTHS] for k in KEYS}
TIME = pd.date_range("1895-01-01", periods=MONTHS, freq="MS")

# Each fix is a list of (old, new) source edits; every `old` must occur exactly once.
FIX_V = [("        # V = PV\n", "        V = PV\n")]
FIX_PL = [
    (
        """    potential_loss_surface = xr.where(
        a >= 0,
        pet,
        surface_storage,
    )""",
        "    potential_loss_surface = np.minimum(pet, surface_storage)",
    ),
    (
        """    potential_loss_under = xr.where(
        a >= 0,
        0.0,
        ((pet - potential_loss_surface) * under_storage / awhc),
    )""",
        "    potential_loss_under = (pet - potential_loss_surface) * under_storage / awhc",
    ),
]
FIX_K = [
    (
        "k_hat = 1.5 * np.log10((t_hat + 2.8) / d_hat) + 0.5",
        "k_hat = 1.5 * np.log10((t_hat + 2.8) / (d_hat / 25.4)) + 0.5",
    )
]
FIX_CAFEC = [
    (
        f"{name} = xr.where({den}_bar == 0, 1.0, {num}_bar / {den}_bar)",
        f"{name} = xr.where({den}_bar == 0, xr.where({num}_bar == 0, {zero}, 0.0), {num}_bar / {den}_bar)",
    )
    for name, num, den, zero in (
        ("alpha", "aet", "pet", 1.0),
        ("beta", "r", "pr", 1.0),
        ("gamma", "ro", "pro", 1.0),
        ("delta", "l", "pl", 0.0),
    )
]
NO_BACKTRACK = [
    (
        """    num1 = new_X

    # Walk backward through the stored candidate spell
    # histories following the sign of the current spell.
    while wet_index_deque and dry_index_deque:""",
        """    return
    num1 = 0
    while False:""",
    )
]


def run_xclim(edits=()):
    source = SOURCE
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType("xpalmer")
    sys.modules["xpalmer"] = module
    exec(compile(source, "_palmer.py", "exec"), module.__dict__)  # noqa: S102

    def data(values):
        return xr.DataArray(values * MM, dims=("time", "division"), coords={"time": TIME, "division": IDS})

    awc = xr.DataArray(AWC * MM, dims="division", coords={"division": IDS})
    ds = module.palmer_indices(data(PRECIP), data(PET), awc, calibration_start_year=1931, calibration_end_year=1990)
    names = {"zindex": "Z"}
    out = {k: ds[names.get(k, k)].transpose("time", "division").values for k in (*KEYS, "scpdsi")}
    return out, ds


def run_climate_indices():
    out = {k: np.full((MONTHS, len(IDS)), np.nan) for k in (*KEYS, "scpdsi")}
    for j in range(len(IDS)):
        args = (PRECIP[:, j], PET[:, j], AWC[j], 1895, 1931, 1990)
        for key, values in zip(KEYS, ci.pdsi(*args)[:4], strict=True):
            out[key][:, j] = values
        out["scpdsi"][:, j] = ci.scpdsi(*args)[0]
    return out


def stats(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a[mask] - b[mask])

    def category(values):
        return np.digitize(values, [-4, -3, -2, 2, 3, 4])

    return (
        f"n={mask.sum()} median={np.median(diff):.3f} rmse={np.sqrt(np.mean(diff**2)):.3f} max={diff.max():.6g} "
        f"r={np.corrcoef(a[mask], b[mask])[0, 1]:+.3f} within0.05={100 * np.mean(diff <= 0.05):.1f}% "
        f"over0.5={100 * np.mean(diff > 0.5):.1f}% category={100 * np.mean(category(a[mask]) == category(b[mask])):.1f}% "
        f"months over1e-6={(diff > 1e-6).sum()} over0.05={(diff > 0.05).sum()}"
    )


def table(title, left, right, keys=KEYS):
    print(f"\n== {title}")
    for key in keys:
        print(f"{key:7s} {stats(left[key], right[key])}")


def residual_tail(left, right):
    """Months differing by more than 0.05 in the final 24 months of the record."""
    for key in KEYS:
        print(f"{key:7s} over0.05 in final 24 months: {(np.abs(left[key] - right[key])[-24:] > 0.05).sum()}")


def contraction_probe(scz, scpdsi):
    """Divisions whose fitted xclim scPDSI recurrence does not contract (|c| >= 1)."""
    module = sys.modules["xpalmer"]
    print("\n== xclim scPDSI contraction")
    for j, division in enumerate(IDS):
        factors = {side: module._duration_factors(scz[:, j], 1931, 1990, 1895, side) for side in ("WET", "DRY")}
        coefficients = {side: 1 - m / (m + b) for side, (m, b) in factors.items()}
        if any(abs(c) >= 1 for c in coefficients.values()) or np.nanmax(np.abs(scpdsi[:, j])) > 20:
            print(
                f"division {division}: max|scPDSI|={np.nanmax(np.abs(scpdsi[:, j])):.6g}",
                {side: tuple(round(float(v), 4) for v in pair) for side, pair in factors.items()},
                {side: round(float(c), 4) for side, c in coefficients.items()},
            )


def nan_gap_probe():
    """climate-indices: one missing month in the first division."""
    clean = ci.pdsi(PRECIP[:, 0], PET[:, 0], AWC[0], 1895, 1931, 1990)[0]
    print("\n== climate-indices pdsi(), one NaN precipitation month, division", IDS[0])
    for label, year in (("inside calibration", 1950), ("before calibration", 1900), ("after calibration", 2000)):
        precip = PRECIP[:, 0].copy()
        precip[(year - 1895) * 12 + 5] = np.nan
        pdsi, _, _, zindex, _ = ci.pdsi(precip, PET[:, 0], AWC[0], 1895, 1931, 1990)
        print(
            f"{label} ({year}-06): finite Z={np.isfinite(zindex).sum()}, finite PDSI={np.isfinite(pdsi).sum()}, "
            f"PDSI == 0: {(pdsi == 0).sum()} of {MONTHS} (clean run: {(clean == 0).sum()})"
        )


def main():
    print("xclim _palmer.py sha256:", hashlib.sha256(SOURCE.encode()).hexdigest())
    ours = run_climate_indices()
    raw, ds = run_xclim()
    table("climate-indices vs NCEI", ours, NCEI)
    table("xclim vs NCEI", raw, NCEI)
    table("xclim vs climate-indices", raw, ours, (*KEYS, "scpdsi"))

    for name, out in (("climate-indices", ours), ("xclim", raw)):
        sc = out["scpdsi"]
        window = sc[(1931 - 1895) * 12 : (1991 - 1895) * 12]
        p2, p98 = (np.nanmedian(np.nanpercentile(window, q, axis=0)) for q in (2, 98))
        print(
            f"{name} scPDSI: min={np.nanmin(sc):.6g} max={np.nanmax(sc):.6g} "
            f"median-division calibration P2={p2:.2f} P98={p98:.2f}; "
            f"PDSI min={np.nanmin(out['pdsi']):.6g} max={np.nanmax(out['pdsi']):.6g}"
        )

    k = ds["k"]
    print(f"xclim K<0: {100 * float((k < 0).mean()):.1f}% of division-months-of-year; ", end="")
    print(f"divisions affected: {int((k < 0).any('month').sum())}; min K={float(k.min()):.3f}")
    over = (ds["pl"] > ds["pet"] + 1e-9).mean()
    print(f"xclim potential loss > PET: {100 * float(over):.1f}% of division-months")

    for label, edits in (
        ("V", FIX_V),
        ("V + PL", FIX_V + FIX_PL),
        ("V + PL + K", FIX_V + FIX_PL + FIX_K),
        ("V + PL + K + CAFEC zero rule", FIX_V + FIX_PL + FIX_K + FIX_CAFEC),
    ):
        out, _ = run_xclim(edits)
        table(f"cumulative fix [{label}] vs NCEI", out, NCEI)
        table(f"cumulative fix [{label}] vs climate-indices", out, ours, (*KEYS, "scpdsi"))
    residual_tail(out, ours)

    for label, edits in (("V", FIX_PL + FIX_K), ("PL", FIX_V + FIX_K), ("K", FIX_V + FIX_PL)):
        out, _ = run_xclim(edits)
        table(f"leave out {label} vs NCEI", out, NCEI)

    no_backtrack, _ = run_xclim(NO_BACKTRACK)
    same = np.array_equal(raw["scpdsi"], no_backtrack["scpdsi"], equal_nan=True)
    print("\nxclim scPDSI bit-identical with _backtrack_self_calibrated disabled:", same)
    _, raw_ds = run_xclim()
    contraction_probe(raw_ds["scz"].transpose("time", "division").values, raw["scpdsi"])
    nan_gap_probe()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure the February 29 PE gap behind ADR-0014 decision 4 (issue #1147).

Compares the chained xarray flood calls (``effective_precipitation`` ->
``edi`` / ``flood_index``, which re-interpolate PE on non-leap February 29) with
the NumPy kernels run on the all-leap layout, where PE comes from interpolated
rainfall. Run it to re-check the ADR's figures after a kernel change:

    uv run scripts/probe_flood_feb29_pe.py

The library's info logs interleave with the tables; the tables are the
``=== duration=... ===`` blocks.

Input: synthetic daily rain, 1990-2019, 40 cells, ``default_rng(0)``. Wet-day
probability (0.3 x season, clipped to 0.05-0.9) and the gamma scale (9 mm x
season, shape 0.7) follow ``season = 1 + 0.6 sin(2 pi (doy - 80) / 365.25)``.
The 365- and 30-day windows each run on that rain ("base") and again with rain
on February 26 to March 3 multiplied 25-fold in every year ("storm"), a
contrived upper bound. EDI calibrates on 1990-2019
and I_F on 1991-2018, for ``year_start_month`` 1, 3, and 10. The figures
describe synthetic rain, not observed data.
"""

import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import flood
from climate_indices.utils import DailyCalendarPlan

Y0, Y1 = 1990, 2019
dates = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31", freq="D")
n = len(dates)
plan = DailyCalendarPlan.from_year_span(Y0, Y1 - Y0 + 1, n)
doy = dates.dayofyear.values


def rain(seed: int, cells: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    season = 1 + 0.6 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    p_wet = np.clip(0.3 * season, 0.05, 0.9)
    wet = rng.random((n, cells)) < p_wet[:, None]
    amt = rng.gamma(0.7, 9.0 * season[:, None], size=(n, cells))
    return np.where(wet, amt, 0.0)


def stats(a, b, label):
    d = np.abs(a - b)
    d = d[np.isfinite(d)]
    print(f"  {label:34s} max|d|={d.max():.3e}  mean|d|={d.mean():.3e}  n>1e-9={int((d > 1e-9).sum())}/{d.size}")


storm = ((dates.month == 2) & (dates.day >= 26)) | ((dates.month == 3) & (dates.day <= 3))
for duration, scenario in ((365, "base"), (30, "base"), (30, "storm"), (365, "storm")):
    print(f"\n=== duration={duration} scenario={scenario} ===")
    r = rain(0)  # (time, cells)
    if scenario == "storm":
        r = r.copy()
        r[storm] *= 25  # extreme rain straddling the synthetic Feb 29
    da = xr.DataArray(r, dims=("time", "cell"), coords={"time": dates}, attrs={"units": "mm"})

    # true all-leap NumPy chain: PE from interpolated rainfall, incl. the synthetic Feb 29
    r_al = plan.to_all_leap(r)  # (366*Y, cells)
    pe_al = flood.effective_precipitation(r_al[..., None], duration=duration, spatial_time_major=True)[..., 0]
    # xarray chain: Gregorian PE, which edi()/flood_index() re-synthesize on Feb 29
    pe_x = flood.effective_precipitation(da, duration=duration)
    pe_x_al = plan.to_all_leap(pe_x.values)
    ny = Y1 - Y0 + 1
    # PE at synthetic Feb 29 slots (non-leap years): true vs interpolated
    idx = [(y * 366 + 59) for y in range(ny) if not (Y0 + y) % 4 == 0]  # 1900/2100 not in range
    pe_true29, pe_int29 = pe_al[idx], pe_x_al[idx]
    ok = np.isfinite(pe_true29) & np.isfinite(pe_int29) & (pe_true29 > 0)  # relative gap needs PE > 0
    rel = np.abs(pe_true29 - pe_int29)[ok] / np.abs(pe_true29[ok])
    print(
        f"  PE Feb29 synth: mean PE={np.nanmean(pe_true29):.1f} mm, "
        f"max|d|={np.nanmax(np.abs(pe_true29 - pe_int29)):.3e} mm, max rel={rel.max():.3e}, mean rel={rel.mean():.3e}"
    )
    # sanity: Gregorian PE identical
    stats(plan.to_gregorian(pe_al), pe_x.values, "PE at Gregorian positions (sanity)")

    # EDI
    edi_ref = flood.edi(pe_al[..., None], Y0, Y0, Y1, duration=duration, spatial_time_major=True)[..., 0]
    edi_x = flood.edi(pe_x, duration=duration)
    stats(plan.to_gregorian(edi_ref), edi_x.values, "EDI, all Gregorian days")
    e_ref, e_x = plan.to_gregorian(edi_ref), edi_x.values
    feb29 = (dates.month == 2) & (dates.day == 29)
    stats(e_ref[feb29], e_x[feb29], "EDI, real Feb 29 (leap yrs)")
    stats(e_ref[~feb29], e_x[~feb29], "EDI, all other days")

    # I_F
    for ysm in (1, 3, 10):
        cy0 = Y0 + 1
        fi_ref = flood.flood_index(pe_al[..., None], Y0, cy0, Y1 - 1, year_start_month=ysm, spatial_time_major=True)[
            ..., 0
        ]
        fi_x = flood.flood_index(
            pe_x, calibration_year_initial=cy0, calibration_year_final=Y1 - 1, year_start_month=ysm
        )
        stats(plan.to_gregorian(fi_ref), fi_x.values, f"I_F year_start_month={ysm}")

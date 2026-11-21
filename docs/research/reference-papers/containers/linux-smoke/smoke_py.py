"""Python comparator smoke calls. Usage: smoke_py.py {xclim,spei,standard_precip,pyet,pyeto}.

Prints `label<TAB>June-2005 value` lines. Inputs: /work/inputs.csv (see inputs.py). The calls
and arguments are those listed in the "Smoke evidence" table of comparators.md.
"""
import math
import sys

import numpy as np
import pandas as pd

df = pd.read_csv("/work/inputs.csv", parse_dates=["date"], index_col="date")
df.index.freq = "MS"
JUN05 = pd.Timestamp("2005-06-01")
LAT = 25.2292
CAL = ("1981-01-01", "2010-12-31")


def out(label, value):
    # Full precision: run.sh rounds to the decimals expected.tsv records, so rounding here
    # first could turn a real difference into a match.
    print(f"{label}\t{float(value)!r}")


which = sys.argv[1]

if which == "xclim":
    import xarray as xr
    from xclim import compute as xi

    def da(s, units):
        return xr.DataArray(s.to_numpy(), dims="time", coords={"time": s.index.to_numpy()}, attrs={"units": units})

    from xclim.core.units import convert_units_to

    pr = da(df.precip, "mm/d")  # monthly totals under a rate label: no per-day rescaling
    kw = dict(freq="MS", window=3, cal_start=CAL[0], cal_end=CAL[1])
    out("xclim_spi3_gamma_APP", xi.standardized_precipitation_index(
        pr, dist="gamma", method="APP", fitkwargs={"floc": 0}, **kw).sel(time=JUN05))
    out("xclim_spi3_gamma_ML", xi.standardized_precipitation_index(
        pr, dist="gamma", method="ML", **kw).sel(time=JUN05))
    # PET in mm/month = xclim's mm/d rate times days in the month (not the 30.4375-day hydro month).
    pet = xi.potential_evapotranspiration(
        tas=da(df.tmean, "degC"), lat=xr.DataArray(LAT, attrs={"units": "degrees_north"}), method="TW48")
    pet_mm = convert_units_to(pet, "mm/d", context="hydro") * pet.time.dt.days_in_month
    out("xclim_pet_TW48", pet_mm.sel(time=JUN05))
    wb = da(df.precip, "mm/d") - pet_mm
    wb.attrs["units"] = "mm/d"
    out("xclim_spei3_fisk_ML", xi.standardized_precipitation_evapotranspiration_index(
        wb, dist="fisk", method="ML", **kw).sel(time=JUN05))

elif which == "spei":
    import scipy.stats as sps
    import spei as si

    out("spei_spi3_gamma", si.spi(df.precip, timescale=3, dist=sps.gamma, fit_freq="MS").loc[JUN05])
    out("spei_spei3_fisk", si.spei(df.precip - df.pet, timescale=3, dist=sps.fisk, fit_freq="MS").loc[JUN05])

elif which == "standard_precip":
    from standard_precip.spei import SPEI
    from standard_precip.spi import SPI

    d = df.reset_index()
    for fit in ("lmom", "mle"):
        r = SPI().calculate(d, "date", "precip", scale=3, fit_type=fit, dist_type="gam",
                            baseline_start=1981, baseline_end=2010)
        out(f"standard_precip_spi3_gam_{fit}", r.set_index("date")["precip_scale_3_calculated_index"].loc[JUN05])
    d["wb"] = d.precip - d.pet
    r = SPEI().calculate(d, "date", "wb", scale=3, fit_type="lmom", dist_type="glo",
                              baseline_start=1981, baseline_end=2010)
    out("standard_precip_spei_glo_lmom", r.set_index("date")["wb_scale_3_calculated_index"].loc[JUN05])

elif which == "pyet":
    import pyet

    out("pyet_hamon", pyet.hamon(df.tmean, lat=math.radians(LAT)).loc[JUN05])

elif which == "pyeto":
    sys.path.insert(0, "/opt/PyETo")
    import pyeto

    t = df.tmean["2005"].to_numpy()
    dlh = pyeto.monthly_mean_daylight_hours(pyeto.deg2rad(LAT), 2005)
    out("pyeto_thornthwaite", pyeto.thornthwaite(t, dlh, 2005)[5])

else:
    sys.exit(f"unknown package {which}")

"""Write /work/inputs.csv from the repository fixtures (mounted read-only at /fixture).

Monthly series from January 1895; June 2005 is row index 1325. The last 10 precipitation
months are NaN in the fixture and are written as NA.
"""
import numpy as np
import pandas as pd

p = np.load("/fixture/precips_mm_monthly.npy").ravel()
t = np.load("/fixture/temp_celsius.npy").ravel()
pet = np.load("/fixture/pet_thornthwaite.npy").ravel()
assert p.shape == t.shape == pet.shape == (1476,), (p.shape, t.shape, pet.shape)
idx = pd.date_range("1895-01-01", periods=len(p), freq="MS")
assert idx[1325] == pd.Timestamp("2005-06-01")
pd.DataFrame({"date": idx, "precip": p, "tmean": t, "pet": pet}).to_csv(
    "/work/inputs.csv", index=False, na_rep="NA"
)

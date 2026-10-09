"""Regression checks for the shared parity comparator, without requiring Rust."""

import numpy as np
import pytest
import xarray as xr

from tests import conftest


def test_dataset_parity_compares_variables_at_existing_tolerances() -> None:
    python = xr.Dataset({"fit": ("month", [1.0, np.nan]), "count": ("month", [1, 2]), "distribution": "gamma"})
    rust = python[["count", "distribution", "fit"]].copy(deep=True)
    rust["fit"] += 1e-12
    conftest.assert_native_parity(rust, python)


@pytest.mark.parametrize("mismatch", ["value", "variable", "dimension", "dtype", "nan", "count", "label"])
def test_dataset_parity_rejects_mismatches(mismatch: str) -> None:
    python = xr.Dataset({"fit": ("month", [1.0, np.nan]), "count": ("month", [1, 2]), "distribution": "gamma"})
    rust = python.copy(deep=True)
    if mismatch == "value":
        rust["fit"][0] = 2.0
    elif mismatch == "variable":
        rust = rust.rename({"fit": "other"})
    elif mismatch == "dimension":
        rust = rust.rename({"month": "other"})
    elif mismatch == "dtype":
        rust["fit"] = rust["fit"].astype(np.float32)
    elif mismatch == "nan":
        rust["fit"][1] = 0.0
    elif mismatch == "count":
        rust["count"][0] = 2
    else:
        rust["distribution"] = "other"
    with pytest.raises(AssertionError):
        conftest.assert_native_parity(rust, python)


@pytest.mark.parametrize("as_dataarray", [False, True], ids=["dataset", "dataarray"])
@pytest.mark.parametrize("mismatch", ["value", "label", "name", "dimension", "missing", "extra"])
def test_xarray_parity_rejects_coordinate_mismatches(as_dataarray: bool, mismatch: str) -> None:
    python = xr.DataArray(
        [[1.0, np.nan], [2.0, 3.0]],
        dims=("month", "site"),
        coords={"month": [1, 2], "label": ("month", ["a", "b"])},
    )
    if not as_dataarray:
        python = python.to_dataset(name="fit")
    rust = python.copy(deep=True)
    conftest.assert_native_parity(rust, python)
    if mismatch == "value":
        rust = rust.assign_coords(month=[1, 3])
    elif mismatch == "label":
        rust = rust.assign_coords(label=("month", ["a", "c"]))
    elif mismatch == "name":
        rust = rust.rename({"label": "other"})
    elif mismatch == "dimension":
        rust = rust.assign_coords(label=("site", ["a", "b"]))
    elif mismatch == "missing":
        rust = rust.drop_vars("label")
    else:
        rust = rust.assign_coords(station="A")
    with pytest.raises(AssertionError):
        conftest.assert_native_parity(rust, python)

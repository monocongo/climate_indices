"""CFFWIS validation against the NRCan reference implementation (#805).

The committed ``cffwis_vwp1985`` fixture is the 48-day noon-observation sample
used in the original Canadian Forest Fire Weather Index System calibration
program (Van Wagner and Pickett, 1985), distributed with the Natural Resources
Canada ``cffdrs`` R package and its ``cffdrs_py`` port. Its reference columns
are full-precision daily FFMC, DMC, DC, ISI, BUI, FWI, and DSR from
``cffdrs_py`` commit 0f57fcca2a6a84b69fe8d50f29d947f0be34d5f6, so this is
cross-implementation validation against the reference lineage named in
``docs/design/fire-subsystem.md``, not another frozen copy of this library's
own output. The same record's published one-decimal values agree with the
fixture within the published rounding bound (measured maximum absolute
deviation 0.0497, and 0.0047 for DSR), so the committed tolerance is a CSV
round-trip bound rather than scientific slack.

``tests/test_fire_cffwis_moisture.py`` keeps the 12-day regression vectors for
the moisture codes; this module is the full-record, all-seven-component
reference check.

The xclim cross-check is a sanity test, not an oracle: xclim 0.62.0 evaluates
the printed FFMC moisture-content constant 147.2, while this library follows
the NRCan reference code's exact 250 * 59.5 / 101 (see
``docs/design/fire-subsystem.md``), so FFMC, ISI, FWI, and DSR can differ by
up to the provenance's ``xclim_constant_difference_atol``. DC, DMC, and BUI
carry no such difference and are checked at floating-point precision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from climate_indices import fire

pytestmark = pytest.mark.validation

_FIXTURE_DIR = Path(__file__).parent / "fixture" / "cffwis_vwp1985"
_COMPONENTS = ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr")


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _read_columns() -> dict[str, np.ndarray]:
    with (_FIXTURE_DIR / "reference.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}


def _load_provenance() -> dict[str, Any]:
    with (_FIXTURE_DIR / "provenance.json").open() as handle:
        return json.load(handle)


def _run_cffwis(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = fire.cffwis(
        columns["temperature_celsius"],
        columns["relative_humidity_percent"],
        columns["wind_speed_kmh"] / 3.6,
        columns["precipitation_mm"],
        columns["latitude"][0],
        columns["month"].astype(int),
    )
    return {name: np.asarray(getattr(result, name)).ravel() for name in _COMPONENTS}


def test_fixture_checksum_matches_provenance() -> None:
    """The committed CSV must be the exact artifact the provenance records."""
    checksum = hashlib.sha256((_FIXTURE_DIR / "reference.csv").read_bytes()).hexdigest()
    assert checksum == _load_provenance()["checksum_sha256"]


def test_reference_record_matches_all_seven_components() -> None:
    """Run the 1985 calibration sample through fire.cffwis() and compare every output.

    The measured maximum deviation is 5e-13, the CSV storage round-trip, so a
    failure here means a recurrence, table, or unit regression rather than a
    tolerance being too tight.
    """
    columns = _read_columns()
    values = _run_cffwis(columns)
    atol = _load_provenance()["validation_tolerance"]["reference_atol"]

    for name in _COMPONENTS:
        np.testing.assert_allclose(
            values[name],
            columns[name],
            rtol=0.0,
            atol=atol,
            err_msg=f"{name} deviates from the cffdrs_py reference beyond {atol}",
        )


def test_xclim_sanity_check_agrees_except_for_the_printed_ffmc_constant() -> None:
    """Cross-check against xclim where it is installed, as a sanity test only.

    xclim is not a dependency, so this test skips when it is absent. The DC,
    DMC, and BUI recurrences must agree to floating-point precision; the FFMC
    lineage differs by the printed-versus-exact moisture constant and is
    bounded by the provenance's ``xclim_constant_difference_atol`` instead.
    """
    xclim = pytest.importorskip("xclim")
    xr = pytest.importorskip("xarray")
    pd = pytest.importorskip("pandas")

    columns = _read_columns()
    dates = pd.to_datetime(
        {
            "year": columns["year"].astype(int),
            "month": columns["month"].astype(int),
            "day": columns["day"].astype(int),
        }
    )

    def _on_time(values: np.ndarray, units: str) -> Any:
        return xr.DataArray(values, coords={"time": dates}, dims=["time"], attrs={"units": units})

    # xclim orders its six outputs DC, DMC, FFMC, ISI, BUI, FWI (no DSR).
    reference = xclim.indices.fire.cffwis_indices(
        tas=_on_time(columns["temperature_celsius"], "degC"),
        pr=_on_time(columns["precipitation_mm"], "mm d-1"),
        hurs=_on_time(columns["relative_humidity_percent"], "%"),
        sfcWind=_on_time(columns["wind_speed_kmh"] / 3.6, "m s-1"),
        lat=xr.DataArray(float(columns["latitude"][0]), attrs={"units": "degrees_north"}),
    )
    tolerance = _load_provenance()["validation_tolerance"]
    values = _run_cffwis(columns)

    for name, reference_values in zip(("dc", "dmc", "ffmc", "isi", "bui", "fwi"), reference, strict=True):
        atol = (
            tolerance["reference_atol"] if name in ("dc", "dmc", "bui") else tolerance["xclim_constant_difference_atol"]
        )
        np.testing.assert_allclose(
            values[name],
            np.asarray(reference_values).ravel(),
            rtol=0.0,
            atol=atol,
            err_msg=f"xclim {name}",
        )

    reference_dsr = np.asarray(xclim.indices.fire.daily_severity_rating(reference[5])).ravel()
    np.testing.assert_allclose(
        values["dsr"],
        reference_dsr,
        rtol=0.0,
        atol=tolerance["xclim_constant_difference_atol"],
        err_msg="xclim dsr",
    )

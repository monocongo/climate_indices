"""Parity of the Rust fire kernels with the Python reference implementations.

Each test computes the same result twice through the public ``climate_indices.fire``
API: once with ``climate_indices.fire._native._native`` replaced by a recorder
around the Rust extension, and once with it set to None, which runs the
pure-Python recurrences. The recorder proves the first run reached the Rust
kernels, so the comparison is never Python against Python. The contract is
``rtol = atol = 1e-10`` with matching NaN positions, and — for the recurrence
tests — an identical returned state.

Native dispatch also requires NumPy floating-point errors to be ignored, as in
``compute._native_float64``, so every run here is wrapped in ``np.errstate``.
Skipped when the extension is not built (``uv run maturin develop --release``),
unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1`` is set, as in CI's native legs.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from climate_indices import fire
from climate_indices._recurrence import _raise_non_finite
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.fire import _native as fire_native
from tests import conftest

native = conftest.import_native()

_FIXTURE_DIR = Path(__file__).parent / "fixture" / "cffwis_vwp1985"
_KBDI_FIXTURE = Path(__file__).parent / "fixture" / "kbdi_ghcn" / "fresno_1991_2020.csv"
# the reference record names the DMC column "dmc", the kernel is "duff_moisture_code"
_KERNEL_BY_COMPONENT = {"ffmc": "ffmc", "dmc": "duff_moisture_code", "dc": "drought_code"}


def _rust_and_python(monkeypatch: pytest.MonkeyPatch, run: Callable[[], Any]) -> tuple[Any, Any, set[str]]:
    return conftest.rust_and_python(monkeypatch, fire_native, run)


_assert_parity = conftest.assert_native_parity


def _cffwis_fixture() -> dict[str, np.ndarray]:
    """The NRCan reference record of tests/test_fire_cffwis_reference.py."""
    with (_FIXTURE_DIR / "reference.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}


def _kbdi_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The GHCN daily record of tests/test_fire_kbdi_reference.py."""
    with _KBDI_FIXTURE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    precipitation = np.array([float(row["precipitation_mm"]) for row in rows])
    temperature = np.array([float(row["maximum_temperature_c"]) for row in rows])
    kbdi = np.array([float(row["kbdi_mm"]) for row in rows])
    return precipitation, temperature, kbdi


def _with_gaps(values: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """A copy with NaN on a fraction of its days, which the gap policy reads as missing."""
    gappy = values.copy()
    rng = np.random.default_rng(seed)
    gappy[rng.random(gappy.shape) < fraction] = np.nan
    return gappy


def _cffwis_run(columns: dict[str, np.ndarray], **kwargs: Any) -> Callable[[], Any]:
    return partial(
        fire.cffwis,
        columns["temperature_celsius"],
        columns["relative_humidity_percent"],
        columns["wind_speed_kmh"] / 3.6,
        columns["precipitation_mm"],
        columns["latitude"][0],
        columns["month"].astype(int),
        **kwargs,
    )


# The moisture codes: the NRCan reference record.


@pytest.mark.parametrize("component", ["ffmc", "dmc", "dc"])
def test_moisture_code_reference_record(monkeypatch, component: str) -> None:
    columns = _cffwis_fixture()
    temperature = columns["temperature_celsius"]
    humidity = columns["relative_humidity_percent"]
    precipitation = columns["precipitation_mm"]
    latitude = columns["latitude"][0]
    months = columns["month"].astype(int)
    if component == "ffmc":
        run = partial(
            fire.ffmc,
            temperature,
            humidity,
            columns["wind_speed_kmh"] / 3.6,
            precipitation,
            return_state=True,
        )
    elif component == "dmc":
        run = partial(
            fire.duff_moisture_code, temperature, humidity, precipitation, latitude, months, return_state=True
        )
    else:
        run = partial(fire.drought_code, temperature, precipitation, latitude, months, return_state=True)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {_KERNEL_BY_COMPONENT[component]}
    _assert_parity(rust, python)
    # the fixture's own reference column, at the committed CSV round-trip bound
    np.testing.assert_allclose(np.asarray(rust.values).ravel(), columns[component], atol=1e-9, equal_nan=True)


def test_moisture_codes_over_a_spatial_block(monkeypatch) -> None:
    """Four latitude bands at once, so the day-length tables are indexed per cell."""
    columns = _cffwis_fixture()
    latitude = np.array([55.0, 20.0, 0.0, -40.0])
    temperature = np.repeat(columns["temperature_celsius"][:, None], latitude.size, axis=1)
    precipitation = np.repeat(columns["precipitation_mm"][:, None], latitude.size, axis=1)
    months = np.repeat(columns["month"][:, None].astype(int), latitude.size, axis=1)

    def run() -> tuple[Any, Any]:
        return (
            fire.duff_moisture_code(temperature, 40.0, precipitation, latitude, months, return_state=True),
            fire.drought_code(temperature, precipitation, latitude, months, return_state=True),
        )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"duff_moisture_code", "drought_code"}
    _assert_parity(rust, python)


def test_moisture_codes_with_missing_days(monkeypatch) -> None:
    """Bridged gaps: one short gap resumes the cell, a long one poisons it."""
    columns = _cffwis_fixture()
    temperature = _with_gaps(columns["temperature_celsius"], 0.2, seed=1)
    precipitation = columns["precipitation_mm"]
    latitude = columns["latitude"][0]
    months = columns["month"].astype(int)

    def run() -> tuple[Any, Any]:
        return (
            fire.duff_moisture_code(
                temperature,
                40.0,
                precipitation,
                latitude,
                months,
                nan_policy="bridge",
                max_gap_days=2,
                return_state=True,
            ),
            fire.drought_code(
                temperature,
                precipitation,
                latitude,
                months,
                nan_policy="bridge",
                max_gap_days=2,
                return_state=True,
            ),
        )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"duff_moisture_code", "drought_code"}
    _assert_parity(rust, python)


def test_moisture_codes_with_propagated_gaps(monkeypatch) -> None:
    """The default policy poisons a cell at its first missing day, on both paths."""
    columns = _cffwis_fixture()
    temperature = _with_gaps(columns["temperature_celsius"], 0.2, seed=5)
    precipitation = _with_gaps(columns["precipitation_mm"], 0.1, seed=6)
    kbdi_precipitation, kbdi_temperature, _ = _kbdi_fixture()
    kbdi_precipitation = _with_gaps(kbdi_precipitation[:500], 0.05, seed=7)
    kbdi_temperature = _with_gaps(kbdi_temperature[:500], 0.05, seed=8)

    def run() -> tuple[Any, Any, Any]:
        return (
            fire.ffmc(
                temperature,
                columns["relative_humidity_percent"],
                columns["wind_speed_kmh"] / 3.6,
                precipitation,
                return_state=True,
            ),
            fire.drought_code(
                temperature,
                precipitation,
                columns["latitude"][0],
                columns["month"].astype(int),
                return_state=True,
            ),
            fire.kbdi(kbdi_precipitation, kbdi_temperature, 300.0, return_state=True),
        )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"ffmc", "drought_code", "kbdi"}
    _assert_parity(rust, python)
    # a poisoned cell stops advancing, so the runs are NaN from their first gap on
    assert np.isnan(np.asarray(rust[0].values)).any()
    assert np.isnan(np.asarray(rust[2].values)).any()


def test_multi_season_run_with_overwintering(monkeypatch) -> None:
    """Two seasons chained through the overwinter equation and non-default seeds."""
    columns = _cffwis_fixture()
    days = columns["month"].size
    temperature = np.repeat(columns["temperature_celsius"][:, None], 2, axis=1)
    precipitation = np.repeat(columns["precipitation_mm"][:, None], 2, axis=1)
    latitude = np.array([55.0, 55.0])
    months = np.repeat(columns["month"][:, None].astype(int), 2, axis=1)
    first_season = np.zeros(days, dtype=bool)
    first_season[: days // 2] = True
    second_season = np.zeros(days, dtype=bool)
    second_season[days // 2 + 4 :] = True

    def run() -> tuple[Any, Any, Any]:
        autumn = fire.drought_code(
            temperature,
            precipitation,
            latitude,
            months,
            in_season=first_season,
            return_state=True,
        )
        assert autumn.state is not None
        spring_seed = fire.overwinter_drought_code(np.asarray(autumn.state.dc), np.array([120.0, 400.0]))
        next_season = fire.drought_code(
            temperature,
            precipitation,
            latitude,
            months,
            in_season=second_season,
            initial_dc=spring_seed,
            return_state=True,
        )
        return autumn, next_season, spring_seed

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"drought_code"}
    _assert_parity(rust, python)
    # the fully recharged cell starts the second season at the published seed
    assert rust[2][1] == 15.0


def test_drought_code_seasonal_carry(monkeypatch) -> None:
    """The ADR-0010 shutdown half: off-season days carry the code instead of a gap NaN."""
    columns = _cffwis_fixture()
    in_season = np.zeros(columns["month"].size, dtype=bool)
    in_season[:20] = True
    in_season[38:] = True
    run = partial(
        fire.drought_code,
        columns["temperature_celsius"],
        columns["precipitation_mm"],
        columns["latitude"][0],
        columns["month"].astype(int),
        in_season=in_season,
        return_state=True,
    )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"drought_code"}
    _assert_parity(rust, python)
    # the off-season days carry the code rather than recording a gap
    assert np.isfinite(np.asarray(rust.values)).all()


def test_moisture_code_is_run_with_a_seasonal_mask_kept_by_the_kernel(monkeypatch) -> None:
    """A masked run and an unmasked one differ only where the mask says so."""
    columns = _cffwis_fixture()
    in_season = np.ones(columns["month"].size, dtype=bool)
    in_season[10:20] = False
    arguments = (
        columns["temperature_celsius"],
        columns["precipitation_mm"],
        columns["latitude"][0],
        columns["month"].astype(int),
    )
    rust, python, calls = _rust_and_python(
        monkeypatch, partial(fire.drought_code, *arguments, in_season=in_season, return_state=True)
    )
    assert calls == {"drought_code"}
    _assert_parity(rust, python)
    # within the off-season the code is carried, not advanced
    values = np.asarray(rust.values).ravel()
    assert len(set(values[10:20])) == 1
    assert values[10] == values[9]


def test_moisture_code_resumes_from_a_returned_state(monkeypatch) -> None:
    """A state returned by one run seeds the next, on both paths."""
    columns = _cffwis_fixture()
    arguments = (
        columns["temperature_celsius"],
        columns["precipitation_mm"],
        columns["latitude"][0],
        columns["month"].astype(int),
    )

    def run() -> tuple[Any, Any]:
        first = fire.drought_code(*arguments, return_state=True)
        assert first.state is not None
        second = fire.drought_code(*arguments, initial_state=first.state, return_state=True)
        return first, second

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"drought_code"}
    _assert_parity(rust, python)


def test_moisture_code_resumes_from_a_split_run(monkeypatch) -> None:
    """Two halves chained through the returned state, the second resumed poisoned."""
    columns = _cffwis_fixture()
    half = columns["month"].size // 2
    temperature = columns["temperature_celsius"].copy()
    # the last day of the first half is missing, so the resumed state is poisoned
    temperature[half - 1] = np.nan
    precipitation = columns["precipitation_mm"]
    latitude = columns["latitude"][0]
    months = columns["month"].astype(int)

    def run() -> tuple[Any, Any]:
        first = fire.duff_moisture_code(
            temperature[:half], 40.0, precipitation[:half], latitude, months[:half], return_state=True
        )
        assert first.state is not None
        second = fire.duff_moisture_code(
            temperature[half:],
            40.0,
            precipitation[half:],
            latitude,
            months[half:],
            initial_state=first.state,
            return_state=True,
        )
        return first, second

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"duff_moisture_code"}
    _assert_parity(rust, python)
    # a poisoned state stays poisoned, so the second half never recovers
    assert np.isnan(np.asarray(rust[1].values)).all()


def test_moisture_code_spin_up(monkeypatch) -> None:
    columns = _cffwis_fixture()
    run = partial(
        fire.ffmc,
        columns["temperature_celsius"],
        columns["relative_humidity_percent"],
        columns["wind_speed_kmh"] / 3.6,
        columns["precipitation_mm"],
        spin_up=10,
        return_state=True,
    )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"ffmc"}
    _assert_parity(rust, python)
    assert np.asarray(rust.values).shape[0] == columns["month"].size - 10


# KBDI: the committed GHCN daily record.


def test_kbdi_reference_record(monkeypatch) -> None:
    """The committed record, with the climatology derived inside ``fire.kbdi``."""
    precipitation, temperature, reference = _kbdi_fixture()
    run = partial(fire.kbdi, precipitation, temperature, return_state=True)

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"kbdi"}
    _assert_parity(rust, python)
    # the same tolerance tests/test_fire_kbdi_reference.py holds the Python path to
    tolerance = json.loads((_KBDI_FIXTURE.parent / "provenance.json").read_text())["validation_tolerance"]
    np.testing.assert_allclose(
        np.asarray(rust.values).ravel(),
        reference,
        rtol=tolerance["regression_rtol"],
        atol=tolerance["regression_atol_mm"],
    )


def test_kbdi_with_missing_days_and_spatial_blocks(monkeypatch) -> None:
    precipitation, temperature, _ = _kbdi_fixture()
    precipitation = np.repeat(precipitation[:, None], 3, axis=1)
    temperature = np.repeat(temperature[:, None], 3, axis=1)
    precipitation[:, 1] = _with_gaps(precipitation[:, 1], 0.3, seed=2)
    temperature[:, 2] = _with_gaps(temperature[:, 2], 0.15, seed=3)
    run = partial(
        fire.kbdi,
        precipitation,
        temperature,
        np.array([250.0, 300.0, 350.0]),
        nan_policy="bridge",
        max_gap_days=5,
        return_state=True,
    )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"kbdi"}
    _assert_parity(rust, python)


def test_kbdi_se38_figure1_with_its_published_initial_value(monkeypatch) -> None:
    """The published SE-38 example, seeded with its previous-day KBDI in inches."""
    fixture = Path(__file__).parent / "fixture" / "kbdi_se38_figure1" / "figure1.csv"
    with fixture.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    precipitation = np.array([float(row["precipitation_in"]) for row in rows])
    temperature = np.array([float(row["maximum_temperature_f"]) for row in rows])
    published = np.array([float(row["published_kbdi_hundredths_in"]) for row in rows])
    run = partial(
        fire.kbdi,
        precipitation,
        temperature,
        50.0,
        units="imperial",
        initial_kbdi=164.0,
        return_state=True,
    )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"kbdi"}
    _assert_parity(rust, python)
    # the same tolerance tests/test_fire_kbdi_reference.py holds the Python path to
    tolerance = json.loads((fixture.parent / "provenance.json").read_text())["validation_tolerance"]
    np.testing.assert_allclose(
        np.asarray(rust.values),
        published,
        atol=tolerance["figure1_continuous_equation_atol_hundredths_in"],
    )


def test_kbdi_imperial_units(monkeypatch) -> None:
    """The imperial path converts to metric before the recurrence and back after."""
    precipitation, temperature, _ = _kbdi_fixture()
    run = partial(
        fire.kbdi,
        precipitation[:365] / 25.4,
        temperature[:365] * 9.0 / 5.0 + 32.0,
        12.0,
        units="imperial",
        return_state=True,
    )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"kbdi"}
    _assert_parity(rust, python)


def test_kbdi_resumes_from_a_returned_state(monkeypatch) -> None:
    precipitation, temperature, _ = _kbdi_fixture()
    precipitation = precipitation[:400]
    temperature = temperature[:400]

    def run() -> tuple[Any, Any]:
        first = fire.kbdi(precipitation, temperature, 300.0, return_state=True)
        assert first.state is not None
        second = fire.kbdi(precipitation, temperature, 300.0, initial_state=first.state, return_state=True)
        return first, second

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"kbdi"}
    _assert_parity(rust, python)


# The combined orchestrator: three codes through one shared day loop, and a
# selection that keeps only some of their histories.


def test_cffwis_orchestrator(monkeypatch) -> None:
    rust, python, calls = _rust_and_python(monkeypatch, _cffwis_run(_cffwis_fixture(), return_state=True))
    assert calls == {"ffmc", "duff_moisture_code", "drought_code"}
    _assert_parity(rust, python)


def test_cffwis_orchestrator_with_a_partial_selection(monkeypatch) -> None:
    """A selection keeps only what it reads, so a code can run without recording."""
    run = _cffwis_run(_cffwis_fixture(), outputs=("dc", "isi"), return_state=True)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"ffmc", "duff_moisture_code", "drought_code"}
    # the DMC feeds neither output, so its history is never allocated
    assert rust.dmc is None
    _assert_parity(rust, python)


def test_cffwis_orchestrator_with_missing_days(monkeypatch) -> None:
    columns = _cffwis_fixture()
    columns["temperature_celsius"] = _with_gaps(columns["temperature_celsius"], 0.1, seed=4)
    run = _cffwis_run(columns, nan_policy="bridge", max_gap_days=1)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"ffmc", "duff_moisture_code", "drought_code"}
    _assert_parity(rust, python)


@pytest.mark.usefixtures("python_backend")
def test_the_python_backend_fixture_pins_the_fire_recurrences() -> None:
    """The shared fixture turns the dispatch off where it would otherwise engage."""
    assert fire_native._native is None
    columns = _cffwis_fixture()
    with np.errstate(all="ignore"):
        result = fire.ffmc(
            columns["temperature_celsius"],
            columns["relative_humidity_percent"],
            columns["wind_speed_kmh"] / 3.6,
            columns["precipitation_mm"],
            return_state=True,
        )
    np.testing.assert_allclose(np.asarray(result.values), columns["ffmc"], atol=1e-9, equal_nan=True)


def test_the_non_finite_translation_raises_the_python_error() -> None:
    """The kernel's non-finite signal becomes the error the Python driver raises.

    No caller-supplied input reaches this today (the Python steps reject the same
    values with a finite result), so the mapping is checked directly; the kernel
    side is covered by ``recurrence::tests::a_non_finite_step_result_is_an_error``.
    """
    with pytest.raises(InvalidArgumentError, match="kbdi produced a non-finite value from finite inputs"):
        _raise_non_finite("kbdi", native.NonFiniteResultError("non-finite"))


def test_default_error_policies_keep_the_python_path(monkeypatch) -> None:
    """Native dispatch requires NumPy floating-point errors to be ignored.

    The Rust kernels do not implement NumPy's warning, exception, callback,
    logging, or printing policies, so an install with the usual policies still
    computes in Python even when the extension is built.
    """
    columns = _cffwis_fixture()
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(fire_native, "_native", recorder)
    with np.errstate(divide="warn", over="warn", under="ignore", invalid="warn"):
        result = fire.ffmc(
            columns["temperature_celsius"],
            columns["relative_humidity_percent"],
            columns["wind_speed_kmh"] / 3.6,
            columns["precipitation_mm"],
            return_state=True,
        )
    assert recorder.calls == set()
    assert np.isfinite(np.asarray(result.values)).all()


def test_a_varying_month_per_cell_still_indexes_the_tables_per_cell(monkeypatch) -> None:
    """A month series that differs by cell indexes the day-length tables per cell."""
    columns = _cffwis_fixture()
    latitude = np.array([55.0, -40.0])
    temperature = np.repeat(columns["temperature_celsius"][:, None], latitude.size, axis=1)
    precipitation = np.repeat(columns["precipitation_mm"][:, None], latitude.size, axis=1)
    months = np.repeat(columns["month"][:, None].astype(int), latitude.size, axis=1)
    months[::2, 1] = np.clip(months[::2, 1] + 1, 1, 12)
    run = partial(fire.duff_moisture_code, temperature, 40.0, precipitation, latitude, months, return_state=True)

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"duff_moisture_code"}
    _assert_parity(rust, python)

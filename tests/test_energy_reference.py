"""Oracle-first energy contracts (#1163); absent functions xfail, broken ones fail.

Only CPC's published CDD series is an authoritative numerical example. Heating,
EDD, and missing-value cases are specification checks; wind vectors come from
independent implementations. These describe climate-driven demand/resources,
not actual generation or load. See tests/fixture/energy_cpc/README.md.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixture"


def _load(dataset: str, filename: str = "reference.json") -> dict[str, Any]:
    data = json.loads((FIXTURE_DIR / dataset / filename).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _require_function(name: str, issue: int) -> Callable[..., Any]:
    try:
        module = importlib.import_module("climate_indices.energy")
    except ModuleNotFoundError as error:
        if error.name != "climate_indices.energy":
            raise
        pytest.xfail(f"climate_indices.energy is not implemented; tracked by #{issue}")
    function = getattr(module, name, None)
    if function is None:
        pytest.xfail(f"energy.{name}() is not implemented; tracked by #{issue}")
    assert callable(function)
    return cast(Callable[..., Any], function)


def _assert_reference(actual: Any, expected: Any) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(actual[expected == 0], 0.0)


@pytest.mark.parametrize("dataset", ["energy_cpc", "energy_wind"])
def test_energy_fixture_checksum(dataset: str) -> None:
    """Integrity runs even while implementation-dependent validation is pending."""
    provenance = _load(dataset, "provenance.json")
    payload = (FIXTURE_DIR / dataset / "reference.json").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == provenance["checksum_sha256"]
    assert provenance["validation_tolerance"]["rtol"] == 0.0
    assert provenance["validation_tolerance"]["atol"] == 1e-12


def test_cpc_transcription_matches_published_seven_day_example() -> None:
    example = _load("energy_cpc")["published_cdd"]
    assert example["temperature_f"] == [67, 65, 70, 74, 78, 65, 68]
    assert example["base_f"] == 65
    assert example["cdd_f_day"] == [2, 0, 5, 9, 13, 0, 3]
    assert sum(example["cdd_f_day"]) == example["total_f_day"] == 32


def test_two_level_shear_recovery_is_only_a_consistency_check() -> None:
    data = _load("energy_wind")
    speed = np.asarray(data["wind_speed_m_s"], dtype=np.float64)
    case = data["cases"][0]
    elevated = np.asarray(case["expected_m_s"], dtype=np.float64)
    positive = np.isfinite(speed) & (speed > 0)
    alpha = np.log(elevated[positive] / speed[positive]) / np.log(100 / 10)
    np.testing.assert_allclose(alpha, case["kwargs"]["alpha"], rtol=0.0, atol=1e-15)


def test_unrelated_import_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_import(name: str) -> None:
        raise ModuleNotFoundError("unrelated dependency missing", name="unrelated_dependency")

    monkeypatch.setattr(importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="unrelated dependency missing"):
        _require_function("cooling_degree_days", 1164)


@pytest.mark.parametrize("actual", [[0.0, 2.0, np.nan], [0.0, 1.0, 0.0], [[0.0, 1.0, np.nan]]])
def test_reference_assertion_rejects_wrong_values_masks_and_shapes(actual: Any) -> None:
    with pytest.raises(AssertionError):
        _assert_reference(actual, [0.0, 1.0, np.nan])


@pytest.mark.validation
def test_cooling_degree_days_against_cpc_published_example() -> None:
    function = _require_function("cooling_degree_days", 1164)
    example = _load("energy_cpc")["published_cdd"]
    temperature = (np.asarray(example["temperature_f"], dtype=np.float64) - 32) * (5 / 9)
    actual = function(temperature, base_temperature=(example["base_f"] - 32) * (5 / 9))
    _assert_reference(actual, np.asarray(example["cdd_f_day"]) * (5 / 9))
    assert np.sum(actual) == pytest.approx(example["total_f_day"] * (5 / 9), rel=0.0, abs=1e-12)


@pytest.mark.validation
@pytest.mark.parametrize("case", _load("energy_cpc")["specification_cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("kind", ["heating", "cooling", "energy"])
def test_degree_day_specification_cases(kind: str, case: dict[str, Any]) -> None:
    """Synthetic HDD/EDD are not CPC product validation; bases are always explicit."""
    function = _require_function(f"{kind}_degree_days", 1164)
    temperature = (np.asarray(case["temperature_f"], dtype=np.float64) - 32) * (5 / 9)
    heating_base = (case["heating_base_f"] - 32) * (5 / 9)
    cooling_base = (case["cooling_base_f"] - 32) * (5 / 9)
    if kind == "energy":
        actual = function(temperature, heating_base_temperature=heating_base, cooling_base_temperature=cooling_base)
        expected = case["edd_f_day"]
    else:
        actual = function(temperature, base_temperature=heating_base if kind == "heating" else cooling_base)
        expected = case["hdd_f_day" if kind == "heating" else "cdd_f_day"]
    _assert_reference(actual, np.asarray(expected, dtype=np.float64) * (5 / 9))


@pytest.mark.validation
@pytest.mark.parametrize("case", _load("energy_wind")["cases"])
def test_hub_height_wind_against_independent_implementations(case: dict[str, Any]) -> None:
    function = _require_function("hub_height_wind_speed", 1165)
    speed = np.asarray(_load("energy_wind")["wind_speed_m_s"], dtype=np.float64)
    actual = function(speed, **case["kwargs"])
    _assert_reference(actual, case["expected_m_s"])
    if case["kwargs"]["reference_height"] == case["kwargs"]["hub_height"] or case["kwargs"].get("alpha") == 0:
        np.testing.assert_array_equal(actual, speed)


@pytest.mark.validation
def test_default_power_profile_uses_one_seventh_exponent() -> None:
    function = _require_function("hub_height_wind_speed", 1165)
    data = _load("energy_wind")
    speed = np.asarray(data["wind_speed_m_s"], dtype=np.float64)
    actual = function(speed, reference_height=10.0, hub_height=100.0)
    _assert_reference(actual, data["cases"][0]["expected_m_s"])

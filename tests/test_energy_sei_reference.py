"""SEI statistical references for climate-driven energy supply/demand, not actual load.

Missing energy wrappers (SREPI and the standardized demand index, #1171) xfail; a present but broken wrapper fails.
Integrity, generator, and oracle controls run without R, network access, or those wrappers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from climate_indices import compute, energy, indices
from tests import conftest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixture" / "energy_sei"
METADATA = json.loads((FIXTURE_DIR / "metadata.json").read_text(encoding="utf-8"))
CASES = METADATA["cases"]
PAYLOADS = ("inputs.npz", "metadata.json", "reference.npz", "sessionInfo.txt", "warnings.txt")


@pytest.fixture
def generator() -> Any:
    """The fixture generator, loaded without its optional R and network providers."""
    spec = importlib.util.spec_from_file_location(
        "prepare_energy_sei_fixtures", ROOT / "scripts" / "prepare_energy_sei_fixtures.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(filename: str, key: str) -> np.ndarray:
    with np.load(FIXTURE_DIR / filename, allow_pickle=False) as arrays:
        return np.asarray(arrays[key])


def _assert_reference(actual: Any, expected: np.ndarray, tolerance: float) -> None:
    actual = np.asarray(actual)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance)


def _require_function(name: str, issue: int) -> Callable[..., Any]:
    function = getattr(energy, name, None)
    if function is None:
        pytest.xfail(f"energy.{name}() is not implemented; tracked by #{issue}")
    assert callable(function)
    return cast(Callable[..., Any], function)


def _kwargs(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in ("scale", "data_start_year", "calibration_year_initial", "calibration_year_final", "aggregation")
    }


def test_fixture_integrity_and_source_pin() -> None:
    provenance = json.loads((FIXTURE_DIR / "provenance.json").read_text(encoding="utf-8"))
    checksum = hashlib.sha256(b"".join((FIXTURE_DIR / name).read_bytes() for name in PAYLOADS)).hexdigest()
    assert checksum == provenance["checksum_sha256"]
    assert METADATA["sei_commit"] == "4d4a6d02d0fbfefdb48da63580c6be1bca3d5f14"
    assert METADATA["sei_archive_sha256"] == "49d1f1ee1c68bb0ec758b7b1cff2d29d48daae5dd508fdec7ce5eaabd97501d4"
    script = ROOT / "scripts" / "energy_sei_reference.R"
    assert hashlib.sha256(script.read_bytes()).hexdigest() == METADATA["r_script_sha256"]
    assert provenance["validation_tolerance"] == {
        "rtol": 0.0,
        "scaled_atol": 1e-12,
        "probability_atol": 1e-15,
        "normal_atol": 1e-12,
    }
    assert METADATA["reading"]["member_sha256"] == "3f3726492fa52c0e8d5e1025e3047b09d3ab75a180701f89d4ea78161c6db7e8"
    session = (FIXTURE_DIR / "sessionInfo.txt").read_text(encoding="utf-8")
    assert "SEI_0.2.0" in session
    assert "fitdistrplus_1.2-6" in session


def test_inputs_retain_energy_meaning_and_real_calendar() -> None:
    demand = _load("inputs.npz", "demand")
    production = _load("inputs.npz", "production")
    residual = _load("inputs.npz", "residual")
    np.testing.assert_array_equal(residual, demand - production)
    assert np.any(residual < 0)
    assert np.any(residual > 0)
    reading = _load("inputs.npz", "reading")
    assert reading.shape == (71 * 12,)
    assert np.all(np.isfinite(reading) & (reading >= 0) & (reading <= 1))
    assert METADATA["reading"]["doi"] == "10.17864/1947.000321"


def test_oracle_pins_sample_counts_ties_constants_and_held_out_tails() -> None:
    counts = _load("reference.npz", "capacity_mean_1_count").reshape(-1, 12)
    np.testing.assert_array_equal(counts[0, 4:8], [9, 10, 99, 100])
    probabilities = _load("reference.npz", "capacity_mean_1_probability").reshape(-1, 12)
    assert np.isnan(probabilities[:, 4]).all()
    np.testing.assert_allclose(probabilities[:, 1:4], 101 / 102, rtol=0, atol=1e-15)
    np.testing.assert_allclose(probabilities[:2, 0], 1 / 102, rtol=0, atol=1e-15)
    np.testing.assert_allclose(probabilities[102:, 0], 101 / 102, rtol=0, atol=1e-15)
    capacity = _load("inputs.npz", "capacity").reshape(-1, 12)
    tied = capacity[:, 0] == 0.5
    assert tied.sum() > 1
    assert np.unique(probabilities[tied, 0]).size == 1


def test_complete_window_aggregation_is_distinct_from_empirical_ranks() -> None:
    summed = _load("reference.npz", "residual_sum_3_scaled")
    averaged = _load("reference.npz", "residual_mean_3_scaled")
    np.testing.assert_allclose(summed, averaged * 3, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        _load("reference.npz", "residual_sum_3_probability"),
        _load("reference.npz", "residual_mean_3_probability"),
    )
    capacity = _load("inputs.npz", "capacity")
    expected_missing = np.ones(capacity.shape, dtype=bool)
    expected_missing[2:] = np.isnan(np.lib.stride_tricks.sliding_window_view(capacity, 3)).any(axis=-1)
    np.testing.assert_array_equal(np.isnan(_load("reference.npz", "capacity_mean_3_scaled")), expected_missing)


def test_finite_bound_control_is_not_a_beta_reference() -> None:
    np.testing.assert_array_equal(_load("reference.npz", "finite_bounds_evaluated"), [0, 0.5, 1])
    _assert_reference(_load("reference.npz", "finite_bounds_probability"), np.array([np.nan, 0.5, np.nan]), 1e-15)
    _assert_reference(_load("reference.npz", "finite_bounds_with_endpoints"), np.array([0.1, 0.5, 0.9]), 1e-15)


@pytest.mark.parametrize("actual", [[0.0, 2.0, np.nan], [0.0, 1.0, 0.0], [[0.0, 1.0, np.nan]]])
def test_comparison_rejects_wrong_values_masks_and_shapes(actual: Any) -> None:
    with pytest.raises(AssertionError):
        _assert_reference(actual, np.array([0.0, 1.0, np.nan]), 1e-12)


def test_broken_present_wrapper_does_not_xfail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(energy, "srepi", object(), raising=False)
    with pytest.raises(AssertionError):
        _require_function("srepi", 1171)


def test_generator_rejects_changed_sources_and_pins_synthetic_inputs(generator: Any, tmp_path: Path) -> None:
    source = tmp_path / "changed.tar.gz"
    source.write_bytes(b"not the pinned source")
    with pytest.raises(ValueError, match="Unexpected source checksum"):
        generator._verify_file(source, generator.SEI_SHA256)
    for name, values in generator._synthetic_inputs().items():
        # Generator.uniform scales its range in compiled code that contracts differently per platform
        # (x86-64 against arm64), so the committed inputs reproduce to within an ulp, not bit for bit.
        # The residual (demand - production) carries that operand ulp into near-zero differences, so the
        # tolerance is absolute at the series scale rather than relative to each element.
        expected = _load("inputs.npz", name)
        np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12 * np.nanmax(np.abs(expected)))
    assert generator._cases() == CASES
    assert generator.SEI_SHA256 == METADATA["sei_archive_sha256"]
    assert generator.PAYLOADS == PAYLOADS


def test_generator_pins_the_serialization_runtime(generator: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    assert generator.PYTHON_PACKAGES == METADATA["python_packages"]
    # the pins govern regeneration, not the test run, so check the accepting path at the pinned versions
    monkeypatch.setattr(generator.np, "__version__", generator.PYTHON_PACKAGES["numpy"])
    monkeypatch.setattr(generator.xr, "__version__", generator.PYTHON_PACKAGES["xarray"])
    generator._verify_runtime()
    monkeypatch.setattr(generator.np, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="Regeneration requires Python packages"):
        generator._verify_runtime()


def test_verify_file_accepts_matching_digest_across_chunks(generator: Any, tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    payload = b"x" * (3 << 20)
    source.write_bytes(payload)
    generator._verify_file(source, hashlib.sha256(payload).hexdigest())


def _write_set(directory: Path, payload: bytes) -> None:
    directory.mkdir()
    (directory / "inputs.npz").write_bytes(payload)
    (directory / "provenance.json").write_text(json.dumps({"fixture_version": "1.0.0"}), encoding="utf-8")


def test_save_arrays_writes_only_the_named_members(generator: Any, tmp_path: Path) -> None:
    """NumPy before 2.2 turns an allow_pickle keyword into a member of that name."""
    path = tmp_path / "arrays.npz"
    generator._save_arrays(path, {"alpha": np.arange(3.0), "beta": np.arange(2.0)})
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays.files == ["alpha", "beta"]


@pytest.mark.parametrize("existing", [False, True])
def test_publish_installs_complete_set(
    generator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    target, staging = tmp_path / "fixtures", tmp_path / "staging"
    if existing:
        _write_set(target, b"old")
    _write_set(staging, b"new")
    monkeypatch.setattr(generator, "FIXTURE_DIR", target)
    generator._publish(staging)
    assert (target / "inputs.npz").read_bytes() == b"new"
    assert not staging.exists()
    assert not target.with_name(".fixtures-backup").exists()


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_publish_restores_previous_set(
    generator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    target, staging = tmp_path / "fixtures", tmp_path / "staging"
    _write_set(target, b"old")
    _write_set(staging, b"new")
    conftest.assert_publish_restores_previous_set(generator, staging, target, monkeypatch, error)


def test_publish_recovers_backup_before_failed_retry(
    generator: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "fixtures"
    backup = tmp_path / ".fixtures-backup"
    _write_set(backup, b"old")
    previous = {path.name: path.read_bytes() for path in backup.iterdir()}
    monkeypatch.setattr(generator, "FIXTURE_DIR", target)
    with pytest.raises(FileNotFoundError):
        generator._publish(tmp_path / "missing-staging")
    assert {path.name: path.read_bytes() for path in target.iterdir()} == previous
    assert not backup.exists()


@pytest.mark.validation
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("output_scale", ["probability", "normal"])
def test_standardization_seam_against_sei(case: dict[str, Any], output_scale: compute.OutputScale) -> None:
    values = _load("inputs.npz", case["input"])
    actual = indices.standardized_index(
        values.copy(),
        distribution=indices.Distribution.empirical,
        periodicity=compute.Periodicity.monthly,
        output_scale=output_scale,
        clip_negatives=False,
        **_kwargs(case),
    )
    expected = _load("reference.npz", f"{case['id']}_{output_scale}")
    _assert_reference(actual, expected, 1e-15 if output_scale == "probability" else 1e-12)
    scaled = compute.prepare_scaled(
        values.copy(),
        case["scale"],
        compute.Periodicity.monthly,
        aggregation=case["aggregation"],
        clip_negatives=False,
    )
    _assert_reference(scaled.ravel(), _load("reference.npz", f"{case['id']}_scaled"), 1e-12)


@pytest.mark.validation
def test_gamma_control_with_sei_mle_parameters() -> None:
    values = _load("inputs.npz", "production")
    shape = _load("reference.npz", "gamma_control_shape")[:12]
    rate = _load("reference.npz", "gamma_control_rate")[:12]
    actual = indices.standardized_index(
        values,
        1,
        indices.Distribution.gamma,
        1900,
        1902,
        2001,
        compute.Periodicity.monthly,
        fitting_params={"alpha": shape, "beta": 1 / rate},
    )
    _assert_reference(actual, _load("reference.npz", "gamma_control_normal"), 1e-12)
    fitted_shape, _ = compute.gamma_parameters(values.reshape(-1, 12), 1900, 1902, 2001, compute.Periodicity.monthly)
    assert np.max(np.abs(fitted_shape - shape)) > 1e-4


@pytest.mark.validation
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_planned_energy_wrappers_against_sei(case: dict[str, Any]) -> None:
    function = _require_function(case["function"], case["issue"])
    actual = function(
        _load("inputs.npz", case["input"]),
        distribution=indices.Distribution.empirical,
        periodicity=compute.Periodicity.monthly,
        **_kwargs(case),
    )
    _assert_reference(actual, _load("reference.npz", f"{case['id']}_normal"), 1e-12)

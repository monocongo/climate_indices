"""Dispatch accounting and PM reporting-policy regressions (#1281)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest

from climate_indices import compute, pm_eto
from tests import conftest, parity_registry

sys.path.append(str(Path(__file__).resolve().parents[1] / "benchmarks"))
from benchmarks import profile_dispatch_overhead as harness  # noqa: E402


def test_invalid_rss_worker_output_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conftest.import_native()
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "command_output", lambda *args: "test")
    monkeypatch.setattr(
        harness.subprocess, "check_output", lambda *args, **kwargs: "not JSON" if kwargs.get("text") else b""
    )
    monkeypatch.setattr(sys, "argv", ["profile_dispatch_overhead.py", "--entries", "--sizes", "--rss-size", "1"])
    with pytest.raises(RuntimeError, match="RSS worker returned no valid JSON"):
        harness.main()


def test_rss_probe_requires_native_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pm_eto, "_native_arrays", lambda *args: None)
    with pytest.raises(RuntimeError, match="RSS probe could not prepare native operands"):
        harness.rss_probe(2)


@pytest.mark.parametrize("result", [None, np.zeros(3), np.zeros((2, 1))])
def test_rss_probe_rejects_invalid_result(monkeypatch: pytest.MonkeyPatch, result: object) -> None:
    monkeypatch.setattr(pm_eto, "penman_monteith_eto", lambda *args: result)
    with pytest.raises(RuntimeError, match="RSS probe returned an unexpected result type or shape"):
        harness.rss_probe(2)


def test_main_requires_extension_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness.importlib, "import_module", lambda name: SimpleNamespace(__file__=None))
    monkeypatch.setattr(sys, "argv", ["profile_dispatch_overhead.py", "--entries", "--sizes", "--rss-size", "0"])
    with conftest.preserved_logging_state(), pytest.raises(RuntimeError, match="Native extension has no file path"):
        harness.main()


def test_supplemental_spei_pearson_requires_the_ks_seam() -> None:
    # the series SPEI Pearson path reaches pearson_ks_statistics; the shared
    # _PEARSON_KERNELS set cannot require it because the spatial spi_pearson grid
    # entry does not reach that flat seam
    assert "pearson_ks_statistics" in harness.SUPPLEMENTAL.kernels


@pytest.mark.parametrize("label", ["/tmp/report", "../../escape", "nested/report"])
def test_report_label_must_be_a_plain_name(monkeypatch: pytest.MonkeyPatch, label: str) -> None:
    monkeypatch.setattr(sys, "argv", ["profile_dispatch_overhead.py", "--sizes", "--rss-size", "0", "--label", label])
    with pytest.raises(SystemExit) as exit_info:
        harness.main()
    assert exit_info.value.code == 2


def test_summary_reports_median_and_iqr() -> None:
    assert harness.summary([1.0, 2.0, 3.0, 4.0, 5.0]) == {"median": 3.0, "q1": 2.0, "q3": 4.0, "iqr": 2.0}


def test_prepared_replay_keeps_arguments_without_public_prep() -> None:
    calls: list[np.ndarray] = []
    array = np.ones(3)
    capture = harness.PreparedCalls(SimpleNamespace(kernel=lambda values: calls.append(values)))
    capture.kernel(array)
    capture.replay()
    assert len(calls) == 2
    assert all(value is array for value in calls)


def test_harness_records_raw_samples_and_reaches_kernels() -> None:
    conftest.import_native()
    entry = parity_registry.ENTRIES_BY_NAME["pm_eto_intermediates"]
    values = parity_registry.SAMPLES[entry.family]
    result = harness.measure(entry.name, entry.run(values), 30, values.size, entry.kernels)
    assert result["calls"] == ["pm_eto"]
    assert all(len(samples) == 30 for samples in result["samples_seconds"].values())
    for native, raw, prep in zip(*[result["samples_seconds"][key] for key in ("native", "raw", "prep")], strict=True):
        assert prep == pytest.approx(native - raw)
    assert result["python_over_native"] > 0.0


@pytest.mark.parametrize("name", ["pm_eto_intermediates", "penman_monteith"])
@pytest.mark.parametrize("policy", ["warn", "raise", "warning_error"])
def test_pm_disallowing_policy_keeps_python(
    monkeypatch: pytest.MonkeyPatch, name: str, policy: Literal["warn", "raise", "warning_error"]
) -> None:
    recorder = conftest.NativeRecorder(conftest.import_native())
    entry = parity_registry.ENTRIES_BY_NAME[name]
    monkeypatch.setattr(pm_eto, "_native", recorder)
    run = entry.run(parity_registry.SAMPLES[entry.family])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.resetwarnings()
        if policy == "warning_error":
            warnings.simplefilter("error", RuntimeWarning)
        else:
            np.seterr(invalid=policy)
        run()
    assert not recorder.calls


@pytest.mark.parametrize("name", ["pm_eto", "fao56_eto"])
def test_scalar_array_and_empty_operand_shapes(name: str) -> None:
    conftest.import_native()
    with harness.native_policy():
        for size in (0, 1, 17):
            run, kernel = harness.pet_cases(size)[name if name == "pm_eto" else "fao56_eto_cycling"]
            assert run().shape == (size,)


def test_constant_operands_stay_unexpanded(monkeypatch: pytest.MonkeyPatch) -> None:
    with harness.native_policy():
        arrays = (np.ones(100), np.array(2.0), 0.0)
        prepared = pm_eto._native_arrays(*arrays)
    assert prepared is not None
    shape, operands = prepared
    assert shape == (100,)
    assert np.shares_memory(operands[0], arrays[0])
    assert operands[1:] == (2.0, 0.0)


def test_multi_operand_guard_checks_policy_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute, "_native", SimpleNamespace())
    calls = []
    monkeypatch.setattr(compute, "_native_policy_allows", lambda: calls.append(None) or True)
    assert compute._native_float64s(np.ones(3), np.ones(3), np.ones(3))
    assert len(calls) == 1
    calls.clear()
    assert not compute._native_float64s(np.ones(3), np.ones(3, dtype=np.float32))
    assert not calls


def test_pm_policy_checked_once(monkeypatch: pytest.MonkeyPatch) -> None:
    conftest.import_native()
    original = compute._native_policy_allows
    calls = []

    def policy() -> bool:
        calls.append(None)
        return original()

    monkeypatch.setattr(compute, "_native_policy_allows", policy)
    with harness.native_policy():
        for name in ("pm_eto_intermediates", "penman_monteith"):
            calls.clear()
            entry = parity_registry.ENTRIES_BY_NAME[name]
            entry.run(parity_registry.SAMPLES[entry.family])()
            assert len(calls) == 1

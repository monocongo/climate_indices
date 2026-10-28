"""Check the empirical benchmark's guards, including optimized Python execution."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from climate_indices import compute
from tests import conftest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "empirical_transform.py"


@pytest.mark.parametrize("optimize", [0, 1])
@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("python", None, None),
        ("rust", None, None),
        ("invalid_backend", ValueError, "BACKEND must be"),
        ("invalid_years", ValueError, "CALIBRATION_YEARS must be"),
        ("missing_native", RuntimeError, "build the extension"),
        ("probe_fallback", RuntimeError, "the probe did not call"),
        ("benchmark_fallback", RuntimeError, "the benchmark did not call"),
        ("invalid_shape", RuntimeError, "invalid result"),
        ("nonfinite", RuntimeError, "invalid result"),
    ],
)
def test_empirical_benchmark_guards(monkeypatch, capsys, optimize, case, error, message):
    """Use a small block and controlled results without requiring the extension."""
    values = np.arange(60 * 12, dtype=float).reshape(60, 12, 1)
    monkeypatch.setattr(np.random, "default_rng", lambda seed: SimpleNamespace(gamma=lambda *args, **kwargs: values))
    monkeypatch.setenv("BACKEND", "python" if case == "python" else "rust")
    monkeypatch.setenv("CALIBRATION_YEARS", "21" if case == "invalid_years" else "20")
    if case == "invalid_backend":
        monkeypatch.setenv("BACKEND", "unknown")
    monkeypatch.setattr(compute, "_native", None if case == "missing_native" else SimpleNamespace())

    def transform(data, calibration, output_scale):
        recorder = compute._native
        if isinstance(recorder, conftest.NativeRecorder):
            if case == "benchmark_fallback" and output_scale == "normal":
                recorder.calls.clear()
            elif case != "probe_fallback":
                recorder.calls.add("empirical_transform")
        if output_scale == "probability":
            return (1 + (calibration[:, None] <= data[None]).sum(axis=0)) / (calibration.shape[0] + 2)
        if case == "invalid_shape":
            return data[:1]
        if case == "nonfinite":
            return np.full_like(data, np.nan)
        return data

    monkeypatch.setattr(compute, "_empirical_fit", transform)
    code = compile(SCRIPT.read_text(), str(SCRIPT), "exec", optimize=optimize)
    if error is not None:
        with pytest.raises(error, match=message):
            exec(code, {"__file__": str(SCRIPT)})
        assert capsys.readouterr().out == ""
    else:
        exec(code, {"__file__": str(SCRIPT)})
        assert '"seconds":' in capsys.readouterr().out

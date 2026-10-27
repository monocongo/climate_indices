"""Measure the #1387 empirical block in a fresh process, including binding copies.

Run with ``BACKEND=python`` or ``BACKEND=rust`` and ``CALIBRATION_YEARS=20`` or
``CALIBRATION_YEARS=60``. Build the extension first::

    for years in 20 60; do
      for backend in python rust; do
        BACKEND=$backend CALIBRATION_YEARS=$years uv run benchmarks/empirical_transform.py
      done
    done

The Rust runs are recorded, so a run that silently fell back to Python fails
instead of reporting a Python number under the native label. Peak RSS is the
process high-water mark, so the reported increase bounds what one call adds.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_indices import compute  # noqa: E402  (the repository root has to be importable first)
from tests import conftest  # noqa: E402  (the tests' recorder is reused rather than re-written)

backend = os.environ["BACKEND"]
years = int(os.environ["CALIBRATION_YEARS"])
assert backend in {"python", "rust"} and years in {20, 60}

values = np.random.default_rng(1435).gamma(2.0, size=(60, 12, 65536))
calibration = values[:years]

recorder: conftest.NativeRecorder | None = None
patch = pytest.MonkeyPatch()
native = getattr(compute, "_native", None)
if backend == "python":
    patch.setattr(compute, "_native", None)
else:
    assert native is not None, "build the extension with `uv run maturin develop --release`"
    recorder = conftest.NativeRecorder(native)
    patch.setattr(compute, "_native", recorder)

# Check the exact rank formula on a small column before measuring the full block.
with np.errstate(all="ignore"):
    probe = compute._empirical_fit(values[:, :, :1], calibration[:, :, :1], "probability")
expected = (1 + (calibration[:, None, :, :1] <= values[None, :, :, :1]).sum(axis=0)) / (years + 2)
np.testing.assert_array_equal(probe, expected)
assert recorder is None or recorder.calls == {"empirical_transform"}

rss_unit = 1 if platform.system() == "Darwin" else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
with np.errstate(all="ignore"):
    start = time.perf_counter()
    result = compute._empirical_fit(values, calibration, "normal")
    seconds = time.perf_counter() - start
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
patch.undo()

assert result.shape == values.shape and np.isfinite(result[0]).all()
assert recorder is None or recorder.calls == {"empirical_transform"}
print(
    json.dumps(
        {
            "backend": backend,
            "calibration_years": years,
            "seconds": seconds,
            "input_bytes": values.nbytes,
            "peak_rss_bytes": after,
            "peak_increase_bytes": after - before,
        }
    )
)

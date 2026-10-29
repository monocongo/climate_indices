"""Measure issue #1513 on the unchanged kernels. This script is diagnostic only."""

import logging
import math
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from climate_indices.fire import _native as dispatch
from tests import conftest
from tests.test_fire_cffwis_behavior import (
    _reference_weather,
    _run_chained,
    _run_orchestrator,
    _series,
    _Weather,
)

logging.disable(logging.CRITICAL)
np.show_runtime()
recorder = conftest.NativeRecorder(conftest.import_native())
dispatch._native = recorder
names = ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr")


def cases() -> Iterator[tuple[str, _Weather, dict[str, Any]]]:
    yield "north", _reference_weather(), {}
    yield "south", replace(_reference_weather(), latitude=np.asarray(-35.0)), {}
    for field in ("temperature", "humidity", "wind", "precipitation"):
        for policy, limit in (("propagate", 0), ("bridge", 1), ("bridge", 3)):
            weather = _series(9)
            getattr(weather, field)[3] = np.nan
            yield f"{field}-{policy}-{limit}", weather, {"nan_policy": policy, "max_gap_days": limit}
    for field, value in (("humidity", 101.0), ("humidity", -1.0), ("wind", -1.0)):
        weather = _series(6)
        getattr(weather, field)[2] = value
        yield f"{field}-{value}", weather, {}
    weather = _series(4)
    weather.wind[2] = -1.0
    yield "negative-wind", weather, {}
    weather = _series(7)
    for field in ("temperature", "humidity", "wind", "precipitation", "month"):
        setattr(weather, field, np.tile(getattr(weather, field)[:, None], (1, 2)))
    weather.precipitation[2, 0] = np.nan
    yield "per-cell", weather, {}


for reporting in ("default", "ignore"):
    policy = {"divide": "warn", "over": "warn", "invalid": "warn", "under": "ignore"}
    if reporting == "ignore":
        policy = {"all": "ignore"}
    print("REPORTING:", reporting)
    maxima = {name: [0.0, 0.0] for name in names}
    with np.errstate(**policy):
        for label, weather, options in cases():
            recorder.calls.clear()
            result = _run_orchestrator(weather, **options)
            if label == "north":
                print("orchestrator kernels:", sorted(recorder.calls))
            recorder.calls.clear()
            chained = _run_chained(weather, **options)
            if label == "north":
                print("chained kernels:", sorted(recorder.calls))
                assert isinstance(result.fwi, np.ndarray)
                x = float(result.fwi[5])
                numpy_dsr = float((0.0272 * np.array([x]) ** 1.77)[0])
                print("FWI[5]:", repr(x), "NumPy DSR:", repr(numpy_dsr), "libm DSR:", repr(0.0272 * math.pow(x, 1.77)))
            for name, expected in zip(names, chained, strict=True):
                actual = getattr(result, name)
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
                finite = np.isfinite(expected)
                if not finite.any():
                    continue
                delta = np.abs(actual[finite] - expected[finite])
                relative = delta / np.maximum(np.abs(expected[finite]), np.finfo(float).tiny)
                maxima[name][0] = max(maxima[name][0], float(delta.max()))
                maxima[name][1] = max(maxima[name][1], float(relative.max()))
                if np.any(delta):
                    print(label, name, "max abs:", float(delta.max()), "max rel:", float(relative.max()))
        print("all scenario maxima:", maxima)
        if reporting == "default":
            dispatch._native = None
            for _, weather, options in cases():
                result = _run_orchestrator(weather, **options)
                for name, expected in zip(names, _run_chained(weather, **options), strict=True):
                    np.testing.assert_array_equal(getattr(result, name), expected)
            print("Disabled extension: all 19 scenarios match exactly.")
            dispatch._native = recorder

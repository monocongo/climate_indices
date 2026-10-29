"""Check the hourly benchmark's aggregation guards with optimized Python."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from climate_indices import compute, energy

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "profile_hourly_srepi.py"


@pytest.mark.parametrize("optimize", [0, 1])
@pytest.mark.parametrize("case", ["valid", "series-shape", "series-padding", "grid-shape", "grid-padding"])
def test_hourly_benchmark_aggregation_guards(monkeypatch, optimize, case):
    """Reject invalid series and grid results without timing hourly calculations."""
    monkeypatch.setattr(energy, "srepi", lambda *args, **kwargs: None)

    def aggregate(values, scale):
        name = "series" if values.ndim == 1 else "grid"
        if case == f"{name}-shape":
            return values[:1]
        result = np.zeros_like(values)
        if case != f"{name}-padding":
            result[: scale - 1] = np.nan
        return result

    monkeypatch.setattr(compute, "sum_to_scale", aggregate)
    namespace: dict[str, Any] = {"__name__": __name__}
    code = compile(SCRIPT.read_text(), str(SCRIPT), "exec", optimize=optimize)
    exec(code, namespace)
    if case == "valid":
        measurements = namespace["_run"](1, 2)
        assert [item.name for item in measurements] == ["series", "grid", "monthly-series", "monthly-grid"]
    else:
        message = "unexpected shape" if case.endswith("shape") else "leading NaNs"
        with pytest.raises(RuntimeError, match=message):
            namespace["_run"](1, 2)

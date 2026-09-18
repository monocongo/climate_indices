"""Runnable check for the gridded SPI/SPEI example in docs/performance.md (#931)."""

from __future__ import annotations

import re
from pathlib import Path

import dask
import xarray as xr

_DOC = Path(__file__).resolve().parents[1] / "docs" / "performance.md"


def test_performance_example_stays_lazy_and_parallel(monkeypatch) -> None:
    """The documented example builds lazy results and asks for the processes scheduler."""
    match = re.search(r"```python\n(.*?)\n```", _DOC.read_text(encoding="utf-8"), re.DOTALL)
    assert match is not None, "docs/performance.md has no python example block"

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_compute(*args: object, **kwargs: object) -> tuple[object, ...]:
        calls.append((args, kwargs))
        return args

    monkeypatch.setattr(dask, "compute", fake_compute)
    exec(compile(match.group(1), str(_DOC), "exec"), {"__name__": "__main__"})

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs.get("scheduler") == "processes"
    assert len(args) == 2
    for lazy in args:
        assert isinstance(lazy, xr.DataArray)
        assert lazy.dims == ("time", "lat", "lon")
        assert lazy.shape == (480, 25, 25)
        assert lazy.chunks is not None
        assert lazy.chunksizes["time"] == (480,)

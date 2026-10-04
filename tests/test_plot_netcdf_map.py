"""Smoke test for the standalone NetCDF map utility."""

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

GeoAxes = pytest.importorskip("cartopy.mpl.geoaxes").GeoAxes  # Optional dev dependency.
image = pytest.importorskip("matplotlib.image")


def test_plot_netcdf_map_selects_time_and_defaults_to_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "grid.nc"
    output = tmp_path / "map.png"
    values = np.stack((np.full((10, 10), -2.0), np.full((10, 10), 2.0)))
    xr.Dataset(
        {"spi_03": (("time", "lat", "lon"), values), "rain": (("lat", "lon"), values[0])},
        coords={
            "time": np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]"),
            "lat": np.linspace(30, 40, 10),
            "lon": np.linspace(-120, -100, 10),
        },
    ).to_netcdf(source)
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_netcdf_map.py"
    main = runpy.run_path(str(script))["main"]
    # No Natural Earth downloads in the test; the real-file smoke check covers boundaries.
    monkeypatch.setattr(GeoAxes, "coastlines", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *_args, **_kwargs: None)
    command = [str(script), str(source), "spi_03", "--output", str(output)]

    monkeypatch.setattr(sys, "argv", [*command, "--time", "2020-01"])
    main()
    first = image.imread(output)
    monkeypatch.setattr(sys, "argv", command)
    main()
    last = image.imread(output)

    assert first[..., 0].mean() > first[..., 2].mean()  # Negative SPI is red.
    assert last[..., 2].mean() > last[..., 0].mean()  # Positive SPI is blue.
    monkeypatch.setattr(sys, "argv", [str(script), str(source), "rain", "--output", str(output)])
    main()
    assert output.stat().st_size > 0  # Non-index, time-free variables also work.
    monkeypatch.setattr(sys, "argv", [*command, "--time", "2020"])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "expected a single (lat, lon) grid" in capsys.readouterr().err

"""Smoke test for the standalone NetCDF map utility."""

import runpy
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

GeoAxes = pytest.importorskip("cartopy.mpl.geoaxes").GeoAxes  # Optional dev dependency.
cartopy = pytest.importorskip("cartopy")
Downloader = pytest.importorskip("cartopy.io").Downloader
image = pytest.importorskip("matplotlib.image")


def test_plot_netcdf_map_selects_time_and_defaults_to_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "grid.nc"
    output = tmp_path / "map.png"
    values = np.stack((np.full((10, 10), -2.0), np.full((10, 10), 2.0)))
    xr.Dataset(
        {
            "spi_03": (("time", "lat", "lon"), values),
            "rain": (("lat", "lon"), values[0]),
            "spi_lonlat": (("lon", "lat"), values[0].T),
        },
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
    monkeypatch.setattr(socket, "setdefaulttimeout", lambda _timeout: None)
    command = [str(script), "--input", str(source), "--var", "spi_03", "--output", str(output)]

    monkeypatch.setattr(sys, "argv", [*command, "--time", "2020-01"])
    main()
    first = image.imread(output)
    monkeypatch.setattr(sys, "argv", command)
    main()
    last = image.imread(output)
    monkeypatch.setattr(sys, "argv", [str(script), "--input", str(source), "--var", "spi_lonlat", "--output", str(output)])
    main()
    transposed = image.imread(output)
    nested = tmp_path / "nested" / "map"
    monkeypatch.setattr(sys, "argv", [str(script), "--input", str(source), "--var", "rain", "--output", str(nested)])
    main()

    assert first[..., 0].mean() > first[..., 2].mean()  # Negative SPI is red.
    assert last[..., 2].mean() > last[..., 0].mean()  # Positive SPI is blue.
    assert transposed[..., 0].mean() > transposed[..., 2].mean() + 0.1  # (lon, lat) order still maps lon to x.
    assert nested.read_bytes().startswith(b"\x89PNG")  # Non-index, time-free variables also work.
    assert capsys.readouterr().out.splitlines()[-1] == str(nested)

    link = tmp_path / "link.png"
    link.hardlink_to(source)
    bare = tmp_path / "bare.nc"
    xr.Dataset({"spi_03": (("lat", "lon"), values[0])}).to_netcdf(bare)
    dateline = tmp_path / "dateline.nc"
    xr.Dataset(
        {"spi_03": (("lat", "lon"), np.zeros((2, 4)))}, coords={"lat": [0, 1], "lon": [170, 175, -180, -175]}
    ).to_netcdf(dateline)
    for argv, message in (
        ([*command, "--time", "2020"], "matches 2 time steps"),
        ([str(script), "--input", str(source), "--var", "spi_03", "--output", str(link)], "output must not overwrite input"),
        ([str(script), "--input", str(bare), "--var", "spi_03", "--output", str(output)], "expected a single (lat, lon) grid"),
        ([str(script), "--input", str(dateline), "--var", "spi_03", "--output", str(output)], "longitudes must be monotonic"),
        ([str(script), "--input", str(source), "--var", "spi_03", "--output", str(source / "map.png")], "cannot save map"),
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit, match="2"):
            main()
        assert message in capsys.readouterr().err

    for arguments in (
        ["--var", "spi_03", "--output", str(output)],
        ["--input", str(source), "--output", str(output)],
        [*command, str(source), "spi_03"],
    ):
        monkeypatch.setattr(sys, "argv", [str(script), *arguments])
        with pytest.raises(SystemExit, match="2"):
            main()
    errors = capsys.readouterr().err
    assert "required: --input" in errors
    assert "required: --var" in errors
    assert "unrecognized arguments" in errors


def test_plot_netcdf_map_bounds_boundary_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "grid.nc"
    xr.Dataset({"spi_03": (("lat", "lon"), np.zeros((2, 2)))}, coords={"lat": [0, 1], "lon": [0, 1]}).to_netcdf(source)
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_netcdf_map.py"
    main = runpy.run_path(str(script))["main"]
    timeouts: list[float | None] = []
    monkeypatch.setattr(socket, "setdefaulttimeout", timeouts.append)  # Keep the global timeout out of pytest.
    for key in ("data_dir", "pre_existing_data_dir"):  # No cached Natural Earth data.
        monkeypatch.setitem(cartopy.config, key, tmp_path / key)

    def stalled(_downloader: object, _url: str) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr(Downloader, "_urlopen", stalled)
    monkeypatch.setattr(sys, "argv", [str(script), "--input", str(source), "--var", "spi_03", "--output", str(tmp_path / "map.png")])

    with pytest.raises(SystemExit, match="2"):
        main()
    assert timeouts == [60]
    assert "cannot save map: timed out" in capsys.readouterr().err

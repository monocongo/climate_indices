"""Smoke tests for the standalone NetCDF map utility."""

import runpy
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import fsspec
import numpy as np
import pytest
import xarray as xr

GeoAxes = pytest.importorskip("cartopy.mpl.geoaxes").GeoAxes
cartopy = pytest.importorskip("cartopy")
Downloader = pytest.importorskip("cartopy.io").Downloader
image = pytest.importorskip("matplotlib.image")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_netcdf_map.py"


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
    main = runpy.run_path(str(SCRIPT))["main"]
    monkeypatch.setattr(GeoAxes, "coastlines", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(socket, "setdefaulttimeout", lambda _timeout: None)
    command = [str(SCRIPT), "--input", str(source), "--var", "spi_03", "--output", str(output)]

    monkeypatch.setattr(sys, "argv", [*command, "--time", "2020-01"])
    main()
    first = image.imread(output)
    monkeypatch.setattr(sys, "argv", command)
    main()
    last = image.imread(output)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--input", str(source), "--var", "spi_lonlat", "--output", str(output)]
    )
    main()
    transposed = image.imread(output)
    nested = tmp_path / "nested" / "map"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--input", str(source), "--var", "rain", "--output", str(nested)])
    main()

    assert first[..., 0].mean() > first[..., 2].mean()
    assert last[..., 2].mean() > last[..., 0].mean()
    assert transposed[..., 0].mean() > transposed[..., 2].mean() + 0.1
    assert nested.read_bytes().startswith(b"\x89PNG")
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
        (
            [str(SCRIPT), "--input", str(source), "--var", "spi_03", "--output", str(link)],
            "output must not overwrite input",
        ),
        (
            [str(SCRIPT), "--input", str(bare), "--var", "spi_03", "--output", str(output)],
            "expected a single (lat, lon) grid",
        ),
        (
            [str(SCRIPT), "--input", str(dateline), "--var", "spi_03", "--output", str(output)],
            "longitudes must be monotonic",
        ),
        (
            [str(SCRIPT), "--input", str(source), "--var", "spi_03", "--output", str(source / "map.png")],
            "cannot save map",
        ),
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
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])
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
    main = runpy.run_path(str(SCRIPT))["main"]
    timeouts: list[float | None] = []
    monkeypatch.setattr(socket, "setdefaulttimeout", timeouts.append)
    for key in ("data_dir", "pre_existing_data_dir"):
        monkeypatch.setitem(cartopy.config, key, tmp_path / key)

    def stalled(_downloader: object, _url: str) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr(Downloader, "_urlopen", stalled)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--input", str(source), "--var", "spi_03", "--output", str(tmp_path / "map.png")]
    )

    with pytest.raises(SystemExit, match="2"):
        main()
    assert timeouts == [60]
    assert "cannot save map: timed out" in capsys.readouterr().err


def test_ncei_comparison_plots_matching_month_and_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "grid.nc"
    output = tmp_path / "comparison.png"
    coords = {
        "time": np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]"),
        "lat": np.linspace(30, 40, 10),
        "lon": np.linspace(-120, -100, 10),
    }
    xr.Dataset(
        {
            "spi_03": (("time", "lat", "lon"), np.ones((2, 10, 10)), {"scale": 3}),
            "spei_pearson_03": (("time", "lat", "lon"), np.ones((2, 10, 10)), {"scale": 3}),
            "spi_06": (("time", "lat", "lon"), np.ones((2, 10, 10)), {"scale": 3}),
            "spei_loglogistic_03": (("time", "lat", "lon"), np.ones((2, 10, 10))),
        },
        coords=coords,
    ).to_netcdf(source)
    static = tmp_path / "static.nc"
    xr.Dataset(
        {"spi_03": (("lat", "lon"), np.ones((2, 2)))},
        coords={"time": np.datetime64("2020-01-01", "ns"), "lat": [30, 31], "lon": [-110, -109]},
    ).to_netcdf(static)
    daily = tmp_path / "daily.nc"
    xr.Dataset(
        {"spi_03": (("time", "lat", "lon"), np.ones((2, 2, 2)))},
        coords={
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            "lat": [30, 31],
            "lon": [-110, -109],
        },
    ).to_netcdf(daily)
    main = runpy.run_path(str(SCRIPT))["main"]
    monkeypatch.setattr(GeoAxes, "coastlines", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(socket, "setdefaulttimeout", lambda _timeout: None)
    calls: list[tuple[str, str, int, str]] = []

    def fetch(index: str, distribution: str, scale: int, month: str) -> tuple[str, xr.DataArray]:
        calls.append((index, distribution, scale, month))
        return "https://www.ncei.noaa.gov/reference.nc", xr.DataArray(
            np.full((10, 10), -2.0), dims=("lat", "lon"), coords={"lat": coords["lat"], "lon": coords["lon"]}
        )

    monkeypatch.setitem(main.__globals__, "_ncei_grid", fetch)
    command = [str(SCRIPT), "--input", str(source), "--var", "spi_03", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", [*command, "--compare", "ncei", "--scale", "3", "--time", "2020-01"])
    main()
    assert calls == [("spi", "gamma", 3, "2020-01")]
    assert image.imread(output).shape[1] > 1500
    assert "NCEI/NIDIS nClimGrid" not in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", [*command[:4], "spei_pearson_03", *command[5:], "--compare", "ncei"])
    main()
    assert calls[-1] == ("spei", "pearson", 3, "2020-02")
    output.unlink()
    for arguments, expected in (
        ([*command, "--compare", "wwdt"], "invalid choice"),
        ([*command, "--compare", "ncei", "--scale", "6"], "must match the timescale in --var"),
        ([*command[:4], "spi_06", *command[5:], "--compare", "ncei"], "disagrees with variable scale metadata"),
        (
            [*command[:4], "spei_loglogistic_03", *command[5:], "--compare", "ncei"],
            "does not offer the loglogistic distribution",
        ),
        ([*command, "--scale", "3"], "--scale requires --compare"),
        ([*command[:2], str(static), *command[3:], "--compare", "ncei"], "requires a time-dependent SPI/SPEI"),
        ([*command[:2], str(daily), *command[3:], "--compare", "ncei"], "monthly time steps"),
    ):
        monkeypatch.setattr(sys, "argv", arguments)
        with pytest.raises(SystemExit, match="2"):
            main()
        assert expected in capsys.readouterr().err
        assert not output.exists()

    monkeypatch.setitem(main.__globals__, "_ncei_grid", Mock(side_effect=ValueError("NCEI unavailable")))
    monkeypatch.setattr(sys, "argv", [*command, "--compare", "ncei"])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "NCEI unavailable" in capsys.readouterr().err
    assert not output.exists()


def test_ncei_grid_reads_exact_month_by_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    grid_for = namespace["_ncei_grid"]
    source = tmp_path / "reference.nc"
    xr.Dataset(
        {
            "spi_03": (
                ("time", "lat", "lon"),
                np.array([[[1.0, -999.9]], [[-2.0, 3.0]], [[-999.9, -999.9]]], dtype="f4"),
                {"_FillValue": -999.9},
            )
        },
        coords={
            "time": np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[ns]"),
            "lat": [40.0],
            "lon": [-110.0, -109.0],
        },
    ).to_netcdf(source, engine="h5netcdf")
    requests: list[tuple[str, dict[str, object]]] = []
    original_open = fsspec.open

    def local_open(url: str, **kwargs: object) -> fsspec.core.OpenFile:
        requests.append((url, kwargs))
        return original_open(source, "rb")

    monkeypatch.setattr(fsspec, "open", local_open)
    url, grid = grid_for("spi", "gamma", 3, "2020-02")
    assert url.endswith("/spi-gamma/nclimgrid-spi-gamma-03.nc")
    assert requests[0][1] == {"block_size": 2**20, "timeout": 15, "allow_redirects": False}
    assert grid.shape == (1, 2)
    np.testing.assert_array_equal(grid.values, [[-2.0, 3.0]])
    with pytest.raises(ValueError, match="no valid SPI-3 data for 2020-03"):
        grid_for("spi", "gamma", 3, "2020-03")
    with pytest.raises(ValueError, match="no SPI-3 data for 2020-04"):
        grid_for("spi", "gamma", 3, "2020-04")
    with pytest.raises(ValueError, match="does not offer a 13-month timescale"):
        grid_for("spi", "gamma", 13, "2020-02")
    with pytest.raises(ValueError, match="no complete 72-month accumulation"):
        grid_for("spi", "gamma", 72, "1895-01")
    monkeypatch.setattr(fsspec, "open", Mock(side_effect=FileNotFoundError("offline")))
    with pytest.raises(ValueError, match="NCEI retrieval failed"):
        grid_for("spi", "gamma", 3, "2020-02")


def test_ncei_grid_refuses_redirects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grid_for = runpy.run_path(str(SCRIPT))["_ncei_grid"]
    requested: list[str] = []

    class Redirect(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            requested.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/elsewhere.nc")
            self.end_headers()

        def do_GET(self) -> None:
            self.do_HEAD()

        def log_message(self, *_args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Redirect) as server:
        Thread(target=server.serve_forever, daemon=True).start()
        monkeypatch.setitem(grid_for.__globals__, "NCEI_BASE", f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(ValueError):
            grid_for("spi", "gamma", 3, "2020-01")
        server.shutdown()
    assert requested
    assert all("elsewhere.nc" not in path for path in requested)

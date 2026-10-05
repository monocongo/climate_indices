"""Smoke test for the standalone NetCDF map utility."""

import runpy
import socket
import sys
from email.message import Message
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

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
    monkeypatch.setattr(
        sys, "argv", [str(script), "--input", str(source), "--var", "spi_lonlat", "--output", str(output)]
    )
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
        (
            [str(script), "--input", str(source), "--var", "spi_03", "--output", str(link)],
            "output must not overwrite input",
        ),
        (
            [str(script), "--input", str(bare), "--var", "spi_03", "--output", str(output)],
            "expected a single (lat, lon) grid",
        ),
        (
            [str(script), "--input", str(dateline), "--var", "spi_03", "--output", str(output)],
            "longitudes must be monotonic",
        ),
        (
            [str(script), "--input", str(source), "--var", "spi_03", "--output", str(source / "map.png")],
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
    monkeypatch.setattr(
        sys, "argv", [str(script), "--input", str(source), "--var", "spi_03", "--output", str(tmp_path / "map.png")]
    )

    with pytest.raises(SystemExit, match="2"):
        main()
    assert timeouts == [60]
    assert "cannot save map: timed out" in capsys.readouterr().err


def test_plot_netcdf_map_compares_exact_conus_wwdt_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "grid.nc"
    output = tmp_path / "comparison.png"
    xr.Dataset(
        {
            "spei_03": (("time", "lat", "lon"), np.ones((2, 10, 10)), {"scale": 3}),
            "spi_13": (("time", "lat", "lon"), np.ones((2, 10, 10))),
            "spi_03": (("time", "lat", "lon"), np.ones((2, 10, 10)), {"scale": 3}),
            "spi_06": (("time", "lat", "lon"), np.ones((2, 10, 10)), {"scale": 3}),
        },
        coords={
            "time": np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]"),
            "lat": np.linspace(30, 40, 10),
            "lon": np.linspace(-120, -100, 10),
        },
    ).to_netcdf(source)
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_netcdf_map.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setattr(GeoAxes, "coastlines", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(socket, "setdefaulttimeout", lambda _timeout: None)
    command = [str(script), "--input", str(source), "--var", "spei_03", "--output", str(output)]
    urls: list[str] = []
    png = BytesIO()
    image.imsave(png, np.ones((8, 8, 3)))

    def fetch(url: str, timeout: int) -> BytesIO:
        urls.append(url)
        assert timeout == 15
        response = BytesIO(png.getvalue())
        response.headers = Message()  # type: ignore[attr-defined]
        response.headers["Content-Type"] = "image/png"  # type: ignore[attr-defined]
        return response

    monkeypatch.setitem(main.__globals__, "urlopen", fetch)
    monkeypatch.setattr(sys, "argv", [*command, "--compare", "wwdt", "--scale", "3", "--time", "2020-01"])
    main()
    assert urls == ["https://wrcc-archive.dri.edu/wwdt/images/ARCHIVE/spei3/202001_us_cl.png"]
    assert image.imread(output).shape[1] > 1500

    monkeypatch.setattr(sys, "argv", [*command, "--compare", "wwdt"])
    main()
    assert urls[-1] == "https://wrcc-archive.dri.edu/wwdt/images/ARCHIVE/spei3/202002_us_cl.png"
    monkeypatch.setattr(sys, "argv", [*command[:4], "spi_03", *command[5:], "--compare", "wwdt"])
    main()
    assert urls[-1] == "https://wrcc-archive.dri.edu/wwdt/images/ARCHIVE/spi3/202002_us_cl.png"
    output.unlink()
    monkeypatch.setitem(main.__globals__, "urlopen", Mock(side_effect=HTTPError(urls[-1], 404, "", {}, None)))
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "WWDT image unavailable for SPI-3, 2020-02" in capsys.readouterr().err
    assert not output.exists()

    for arguments, expected in (
        ([*command, "--compare", "noaa"], "no verified SPI/SPEI image endpoint"),
        ([*command, "--compare", "wwdt", "--scale", "6"], "must match the timescale in --var"),
        ([*command, "--compare", "wwdt", "--scale", "13"], "must match the timescale in --var"),
        ([*command[:4], "spi_13", *command[5:], "--compare", "wwdt"], "does not offer a 13-month timescale"),
        ([*command[:4], "spi_06", *command[5:], "--compare", "wwdt"], "disagrees with variable scale metadata"),
        ([*command, "--scale", "3"], "--scale requires --compare"),
    ):
        monkeypatch.setattr(sys, "argv", arguments)
        with pytest.raises(SystemExit, match="2"):
            main()
        assert expected in capsys.readouterr().err
        assert not output.exists()

    monkeypatch.setattr(sys, "argv", [*command, "--compare", "wwdt"])
    monkeypatch.setitem(main.__globals__, "urlopen", Mock(side_effect=URLError("offline")))
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "WWDT request failed" in capsys.readouterr().err
    assert not output.exists()

    def wrong_type(url: str, timeout: int) -> BytesIO:
        response = fetch(url, timeout)
        response.headers.replace_header("Content-Type", "text/html")  # type: ignore[attr-defined]
        return response

    monkeypatch.setitem(main.__globals__, "urlopen", wrong_type)
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "non-PNG content" in capsys.readouterr().err
    assert not output.exists()

    png = BytesIO(b"not a PNG")
    monkeypatch.setitem(main.__globals__, "urlopen", fetch)
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "could not be decoded" in capsys.readouterr().err
    assert not output.exists()

    # The archive has date-coded PNGs even for timescales with no complete accumulation.
    image_for = main.__globals__["_wwdt_image"]
    with pytest.raises(ValueError, match="no complete 72-month accumulation"):
        image_for("spi", 72, "1895-01")

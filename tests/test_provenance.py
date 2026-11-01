"""Dataset provenance: the record every computed output variable carries (#1470)."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from importlib import metadata
from pathlib import Path
from typing import Any

import dask.config
import numpy as np
import pytest
import xarray as xr

import climate_indices
from climate_indices import compute, energy, fire, flood, indices, provenance
from climate_indices.__main__ import main
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.indices import Distribution
from climate_indices.provenance import PROVENANCE_ATTR, read_provenance, resolve_software_identity
from climate_indices.xarray_adapter import build_output_attrs, xarray_adapter

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")


def _monthly(name: str = "prcp", *, seed: int = 0, start: str = "1981-01-01", **attrs: Any) -> xr.DataArray:
    time = xr.date_range(start, periods=480, freq="MS")
    values = np.random.default_rng(seed).gamma(2.0, 30.0, time.size)
    return xr.DataArray(values, dims="time", coords={"time": time}, name=name, attrs={"units": "mm", **attrs})


def _strict(raw: str) -> Any:
    """Parse JSON, rejecting the NaN/Infinity tokens strict JSON does not allow."""

    def reject(token: str) -> None:
        raise AssertionError(f"non-strict JSON token {token}")

    return json.loads(raw, parse_constant=reject)


# --- attachment ----------------------------------------------------------------------------------


def test_spi_output_carries_version_commit_schema_and_record() -> None:
    result = climate_indices.spi(_monthly(), scale=3, distribution=Distribution.gamma)

    identity = provenance.software_identity()
    assert result.attrs["climate_indices_version"] == climate_indices.__version__
    assert result.attrs["climate_indices_git_commit"] == (identity["git_commit"] or "unknown")
    assert result.attrs["climate_indices_provenance_schema_version"] == provenance.SCHEMA_VERSION
    record = _strict(result.attrs[PROVENANCE_ATTR])
    assert record["schema"] == provenance.SCHEMA
    assert record["software"] == identity
    assert record["software"]["version"] == climate_indices.__version__
    assert record["operation"] == {
        "index": "SPI",
        "interface": "python",
        "implementation": "climate_indices.indices.spi",
    }
    assert record["backend"]["kernel_selection"] == "per call at execution time; not recorded"
    assert datetime.datetime.strptime(record["created"], "%Y-%m-%dT%H:%M:%SZ")
    # every attribute is a NetCDF/Zarr-portable scalar
    assert all(isinstance(value, str | int | float) for value in result.attrs.values())


def test_record_holds_resolved_explicit_inferred_and_default_parameters() -> None:
    result = climate_indices.spi(_monthly(), scale=3, distribution=Distribution.pearson)

    parameters = read_provenance(result)["parameters"]  # type: ignore[index]
    assert parameters == {
        "scale": 3,
        "distribution": "pearson",
        "data_start_year": 1981,
        "calibration_year_initial": 1981,
        "calibration_year_final": 2020,
        "periodicity": "monthly",
        "output_scale": "normal",
        "zero_handling": "classic",
        "fitting_params": None,
    }


def test_different_calibration_periods_remain_distinguishable() -> None:
    precipitation = _monthly()
    early = climate_indices.spi(
        precipitation,
        scale=3,
        distribution=Distribution.gamma,
        calibration_year_initial=1981,
        calibration_year_final=2010,
    )
    late = climate_indices.spi(
        precipitation,
        scale=3,
        distribution=Distribution.gamma,
        calibration_year_initial=1991,
        calibration_year_final=2020,
    )

    early_record, late_record = read_provenance(early), read_provenance(late)
    assert early_record is not None
    assert late_record is not None
    assert early_record["parameters"]["calibration_year_initial"] == 1981
    assert late_record["parameters"]["calibration_year_initial"] == 1991
    assert early_record["record_id"] != late_record["record_id"]
    assert not np.allclose(early.values, late.values, equal_nan=True)


def test_a_clamped_calibration_window_records_the_years_the_fit_used() -> None:
    result = climate_indices.spi(
        _monthly(), scale=3, distribution=Distribution.gamma, calibration_year_initial=1950, calibration_year_final=2050
    )

    parameters = read_provenance(result)["parameters"]  # type: ignore[index]
    assert (parameters["calibration_year_initial"], parameters["calibration_year_final"]) == (1981, 2020)


def test_spei_keeps_each_input_and_its_conflicting_metadata_distinct() -> None:
    precipitation = _monthly("prcp", source="gauge network A", institution="Agency A")
    pet = _monthly("pet", seed=1, source="reanalysis B", institution="Agency B", units="mm")

    result = climate_indices.spei(precipitation, pet_mm=pet, scale=6, distribution=Distribution.gamma)

    inputs = {entry["role"]: entry for entry in read_provenance(result)["inputs"]}  # type: ignore[index]
    assert set(inputs) == {"precips_mm", "pet_mm"}
    assert inputs["precips_mm"]["variable"] == "prcp"
    assert inputs["precips_mm"]["attributes"]["source"] == "gauge network A"
    assert inputs["pet_mm"]["variable"] == "pet"
    assert inputs["pet_mm"]["attributes"]["source"] == "reanalysis B"
    # an input's metadata is recorded under it, not merged onto the output
    assert result.attrs.get("institution") == "Agency A"  # the primary's attrs are inherited, as before
    assert "Agency B" not in json.dumps({key: value for key, value in result.attrs.items() if key != PROVENANCE_ATTR})


def test_each_palmer_output_keeps_its_own_index_inputs_and_parameters() -> None:
    precipitation = _monthly("prcp") / 25.4
    pet = _monthly("pet", seed=1) / 40.0

    result = climate_indices.pdsi(precipitation, pet, 5.0, calibration_year_initial=1981, calibration_year_final=2010)

    indices_seen = set()
    for name in ("pdsi", "phdi", "pmdi", "z_index"):
        record = read_provenance(result[name])
        assert record is not None
        indices_seen.add(record["operation"]["index"])
        assert [entry["role"] for entry in record["inputs"]] == ["precips", "pet", "awc"]
        assert record["inputs"][2] == {"role": "awc", "type": "scalar", "value": 5.0}
        assert record["parameters"]["calibration_year_final"] == 2010
    assert indices_seen == {"PDSI", "PHDI", "PMDI", "Z-Index"}


def test_hargreaves_records_both_temperature_inputs_and_latitude() -> None:
    time = xr.date_range("2000-01-01", periods=730, freq="D")
    tmin = xr.DataArray(
        np.full(time.size, 10.0), dims="time", coords={"time": time}, name="tmin", attrs={"units": "degC"}
    )
    tmax = (tmin + 10.0).rename("tmax")

    result = climate_indices.pet_hargreaves(tmin, tmax, latitude=40.0)

    inputs = read_provenance(result)["inputs"]  # type: ignore[index]
    assert [(entry["role"], entry.get("variable")) for entry in inputs] == [
        ("daily_tmin_celsius", "tmin"),
        ("daily_tmax_celsius", "tmax"),
        ("latitude", None),
    ]
    assert inputs[2]["value"] == 40.0


def test_each_fit_diagnostics_output_records_the_values_input() -> None:
    precipitation = _monthly()

    result = climate_indices.fit_diagnostics(precipitation, scale=3, distribution=Distribution.gamma)

    for variable in result.data_vars.values():
        record = read_provenance(variable)
        assert record is not None
        assert [entry["role"] for entry in record["inputs"]] == ["values"]
        assert record["inputs"][0]["variable"] == "prcp"


def test_the_backend_entry_ignores_a_non_string_native_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native parity tests swap the extension for a recorder, whose attributes are callables."""
    monkeypatch.setattr(compute, "_native", types.SimpleNamespace(__version__=lambda: "3.1.0"))

    backend = read_provenance(climate_indices.spi(_monthly(), scale=3, distribution=Distribution.gamma))["backend"]  # type: ignore[index]

    assert backend == {
        "native_extension": "available",
        "native_version": None,
        "kernel_selection": "per call at execution time; not recorded",
    }


def test_numpy_input_returns_a_plain_array_without_metadata() -> None:
    values = _monthly().values
    result = climate_indices.spi(values, 3, Distribution.gamma, 1981, 1981, 2020, compute.Periodicity.monthly)
    assert isinstance(result, np.ndarray)


# --- normalization, bounds, and privacy -----------------------------------------------------------


def test_unusual_input_metadata_serializes_to_strict_bounded_json() -> None:
    nested: dict[str, Any] = {"level": 0}
    cursor = nested
    for depth in range(1, 8):
        cursor["child"] = {"level": depth}
        cursor = cursor["child"]
    attrs = {
        "float32_nan": np.float32("nan"),
        "positive_inf": float("inf"),
        "negative_inf": -np.inf,
        "numpy_int": np.int16(7),
        "numpy_bool": np.bool_(True),
        "bytes": b"caf\xc3\xa9 \xff",
        "date": datetime.date(2020, 1, 2),
        "datetime64": np.datetime64("2020-01-02T03:04:05"),
        "small_array": np.array([1, 2, 3], dtype=np.int32),
        "large_array": np.arange(1000.0),
        "long_text": "x" * 5000,
        "nested": nested,
        "many": {f"key{index:03d}": index for index in range(100)},
        "opaque": object(),
        "complex": 1 + 2j,
    }
    data = xr.DataArray(np.zeros(3), dims="x", name="odd", attrs=attrs)

    output = build_output_attrs(data, index_name="Test")

    recorded = _strict(output[PROVENANCE_ATTR])["inputs"][0]["attributes"]
    assert recorded["float32_nan"] == "NaN"
    assert (recorded["positive_inf"], recorded["negative_inf"]) == ("Infinity", "-Infinity")
    assert recorded["numpy_int"] == 7
    assert recorded["numpy_bool"] is True
    assert recorded["bytes"] == "café \ufffd"
    assert recorded["date"] == "2020-01-02"
    assert recorded["datetime64"] == "2020-01-02T03:04:05"
    assert recorded["small_array"] == [1, 2, 3]
    assert recorded["large_array"] == {"type": "ndarray", "shape": [1000], "dtype": "float64"}
    assert len(recorded["long_text"]) < 1100
    assert recorded["long_text"].endswith("[truncated 3976 characters]")
    assert "nesting depth limit" in json.dumps(recorded["nested"])
    assert recorded["many"]["omitted_keys"] == 36
    assert len(recorded["many"]) == 65
    assert recorded["opaque"] == {"unsupported_type": "object"}
    assert recorded["complex"] == {"unsupported_type": "complex"}


def test_an_oversized_record_drops_input_attributes_first_and_says_so() -> None:
    attrs = {f"comment{index:02d}": "y" * 1000 for index in range(60)}
    data = xr.DataArray(np.zeros(3), dims="x", name="big", attrs=attrs)

    output = build_output_attrs(data, index_name="Test")

    assert len(output[PROVENANCE_ATTR]) <= 32 * 1024
    record = _strict(output[PROVENANCE_ATTR])
    assert record["truncated"] == ["inputs[].attributes"]
    assert record["inputs"][0]["attributes"] == {"omitted": "record size limit"}
    assert record["inputs"][0]["variable"] == "big"


def test_locators_and_text_lose_credentials_tokens_and_local_directories() -> None:
    home = str(Path.home())
    data = xr.DataArray(
        np.zeros(2),
        dims="x",
        name="v",
        attrs={
            "source": "fetched from https://alice:hunter2@data.example.org:8443/x.nc?X-Amz-Signature=abc&token=t#part",
            "comment": f"prepared in {home}/work",
            "api_key": "abc123",
            "Authorization": "Bearer xyz",
        },
    )
    data.encoding["source"] = f"{home}/private/precip.nc"

    serialized = build_output_attrs(data, index_name="Test")[PROVENANCE_ATTR]

    for secret in ("alice", "hunter2", "X-Amz-Signature", "token=t", "abc123", "Bearer", "#part", home):
        assert secret not in serialized
    entry = _strict(serialized)["inputs"][0]
    assert entry["source"] == {"kind": "local-file", "name": "precip.nc"}
    assert entry["attributes"]["source"] == "fetched from https://data.example.org:8443/x.nc?[redacted]"
    assert entry["attributes"]["comment"] == "prepared in work"
    assert entry["attributes"]["api_key"] == entry["attributes"]["Authorization"] == "[redacted]"


def test_text_reduces_a_windows_network_path_to_its_file_name() -> None:
    data = xr.DataArray(
        np.zeros(2),
        dims="x",
        name="v",
        attrs={
            "comment": r"copied from \\fileserver\projects\climate\prcp.nc and \\192.168.1.5\share\q.nc today",
            "note": r"the regex \\d+ or \\S+\\w* and a literal \\n\\t are not paths",
        },
    )

    attributes = _strict(build_output_attrs(data, index_name="Test")[PROVENANCE_ATTR])["inputs"][0]["attributes"]

    assert attributes["comment"] == "copied from prcp.nc and q.nc today"
    assert attributes["note"] == r"the regex \\d+ or \\S+\\w* and a literal \\n\\t are not paths"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (
            "s3://bucket/store.zarr?X-Amz-Credential=k",
            {"kind": "url", "url": "s3://bucket/store.zarr", "query_removed": True},
        ),
        ("https://user:pw@host/p.nc", {"kind": "url", "url": "https://host/p.nc"}),
        (
            "https://user:pw@[2001:db8::1]:8443/p.nc?token=secret#part",
            {"kind": "url", "url": "https://[2001:db8::1]:8443/p.nc", "query_removed": True},
        ),
        ("https://[2001:db8::1]/p.nc", {"kind": "url", "url": "https://[2001:db8::1]/p.nc"}),
        ("file:///Users/alice/data/p.nc", {"kind": "local-file", "name": "p.nc"}),
        ("file://alice:pw@host/private/p.nc?token=secret#part", {"kind": "local-file", "name": "p.nc"}),
        ("file://[broken", {"kind": "local-file", "name": "[unparseable URL]"}),
        ("C:\\Users\\bob\\data\\p.nc", {"kind": "local-file", "name": "p.nc"}),
        ("/home/carol/p.zarr/", {"kind": "local-file", "name": "p.zarr"}),
    ],
)
def test_with_source_records_a_sanitized_locator(uri: str, expected: dict[str, Any]) -> None:
    labeled = provenance.with_source(_monthly(), uri=uri)
    result = climate_indices.spi(labeled, scale=3, distribution=Distribution.gamma)
    assert read_provenance(result)["inputs"][0]["source"] == expected  # type: ignore[index]


def test_with_source_identifiers_checksum_and_dataset_attributes() -> None:
    precipitation = _monthly()
    labeled = provenance.with_source(
        precipitation,
        uri="https://example.org/nclimgrid.nc",
        dataset_id="nclimgrid-monthly",
        doi="10.7289/V5SX6B56",
        version="1.0.0",
        checksum={"algorithm": "sha256", "value": "ab" * 32, "object": "file"},
        dataset_attributes={"title": "nClimGrid", "history": "regridded", "summary": "not kept", "license": "CC0"},
    )

    result = climate_indices.spi(labeled, scale=3, distribution=Distribution.gamma)

    entry = read_provenance(result)["inputs"][0]  # type: ignore[index]
    assert entry["identifiers"] == {"dataset_id": "nclimgrid-monthly", "doi": "10.7289/V5SX6B56", "version": "1.0.0"}
    assert entry["checksum"] == {"algorithm": "sha256", "value": "ab" * 32, "object": "file"}
    assert entry["dataset_attributes"] == {"history": "regridded", "license": "CC0", "title": "nClimGrid"}
    assert provenance.SOURCE_ATTR not in result.attrs
    assert provenance.SOURCE_ATTR not in precipitation.attrs  # the caller's object is not modified


def test_with_source_rejects_an_incomplete_checksum_and_a_non_dataarray() -> None:
    precipitation = _monthly()
    values = np.zeros(3)
    with pytest.raises(InvalidArgumentError, match="object"):
        provenance.with_source(precipitation, checksum={"algorithm": "md5", "value": "0"})
    with pytest.raises(InvalidArgumentError):
        provenance.with_source(values, uri="x.nc")  # type: ignore[arg-type]


# --- history and lineage -------------------------------------------------------------------------


def test_history_is_extended_and_upstream_lineage_is_a_bounded_digest() -> None:
    temperature = _monthly("tavg", history="2020-01-01: regridded").copy(data=np.full(480, 15.0))
    temperature.attrs["units"] = "degC"
    pet = climate_indices.pet_thornthwaite(temperature, latitude=40.0).rename("pet")
    precipitation = _monthly("prcp", history="2020-01-01: quality controlled")

    spei = climate_indices.spei(precipitation, pet_mm=pet, scale=3, distribution=Distribution.gamma)
    again = climate_indices.spi(spei.rename("spei"), scale=3, distribution=Distribution.gamma)

    # the primary input's history is kept and extended; the PET input keeps its own
    assert spei.attrs["history"].splitlines()[0] == "2020-01-01: quality controlled"
    entry = f"SPEI-3 calculated using gamma distribution (climate_indices v{climate_indices.__version__})"
    assert spei.attrs["history"].splitlines()[1].endswith(entry)
    pet_record, spei_record = read_provenance(pet), read_provenance(spei)
    assert pet_record is not None
    assert spei_record is not None
    pet_entry = spei_record["inputs"][1]
    assert pet_entry["attributes"]["history"].startswith("2020-01-01: regridded")
    assert PROVENANCE_ATTR not in pet_entry["attributes"]
    assert [step["record_id"] for step in pet_entry["upstream"]] == [pet_record["record_id"]]
    assert pet_entry["upstream"][0]["operation"]["index"] == "PET Thornthwaite"
    assert pet_entry["upstream"][0]["inputs"][0]["variable"] == "tavg"

    # a further step flattens the chain instead of nesting whole records
    upstream = read_provenance(again)["inputs"][0]["upstream"]  # type: ignore[index]
    assert [step["record_id"] for step in upstream] == [spei_record["record_id"], pet_record["record_id"]]
    assert all("upstream" not in entry for step in upstream for entry in step["inputs"])


def test_reattaching_replaces_the_record_and_never_inherits_input_provenance() -> None:
    first = climate_indices.spi(_monthly(), scale=3, distribution=Distribution.gamma)
    stale = first.copy()
    stale.attrs["climate_indices_git_commit"] = "f" * 40  # e.g. a record made by another build

    attrs = build_output_attrs(stale, index_name="Repeat")

    record = _strict(attrs[PROVENANCE_ATTR])
    assert record["record_id"] != read_provenance(first)["record_id"]  # type: ignore[index]
    assert attrs["climate_indices_git_commit"] == (provenance.software_identity()["git_commit"] or "unknown")
    assert [step["record_id"] for step in record["inputs"][0]["upstream"]] == [read_provenance(first)["record_id"]]  # type: ignore[index]
    assert attrs["history"].count("Repeat calculated") == 1


@pytest.mark.parametrize("container", [xr.DataArray, xr.Dataset])
@pytest.mark.parametrize("raw", [None, '{"schema_version":"1.0.0"}', b'{"schema_version":"1.0.0"}'])
def test_read_provenance_reads_own_attributes(container: Any, raw: Any) -> None:
    data = container(attrs={} if raw is None else {PROVENANCE_ATTR: raw})

    assert read_provenance(data) == (None if raw is None else {"schema_version": "1.0.0"})


@pytest.mark.parametrize("container, label", [(xr.DataArray, "'v'"), (xr.Dataset, "Dataset")])
@pytest.mark.parametrize("raw", ["{not json", 5])
def test_read_provenance_rejects_an_attribute_that_is_not_json(container: Any, label: str, raw: Any) -> None:
    variable = xr.DataArray(np.zeros(1), dims="x", name="v")
    data = variable if container is xr.DataArray else xr.Dataset({"name": variable})
    data.attrs[PROVENANCE_ATTR] = raw

    with pytest.raises(ValueError, match=f"{PROVENANCE_ATTR} of {label} is not valid JSON"):
        read_provenance(data)


def test_an_unparseable_input_record_is_flagged_not_propagated() -> None:
    data = _monthly()
    data.attrs[PROVENANCE_ATTR] = "{not json"
    record = _strict(build_output_attrs(data, index_name="Test")[PROVENANCE_ATTR])
    assert record["inputs"][0]["upstream"] == [{"unparseable": True}]


# --- serialization and laziness ------------------------------------------------------------------


def _write_and_reopen(dataset: xr.Dataset, path: Path, engine: str) -> xr.Dataset:
    if engine == "zarr":
        pytest.importorskip("zarr")
        dataset.to_zarr(path, mode="w", consolidated=False)
        reopened: xr.Dataset = xr.open_zarr(path, consolidated=False)
        return reopened
    dataset.to_netcdf(path, engine=engine)  # type: ignore[call-overload]
    return xr.open_dataset(path, engine=engine)


@pytest.mark.parametrize("engine", ["h5netcdf", "scipy", "zarr"])
def test_netcdf_and_zarr_round_trips_preserve_each_variables_record(tmp_path: Path, engine: str) -> None:
    precipitation = provenance.with_source(_monthly(institution="Ünïcode Agency"), uri="nclimgrid.nc")
    spi_3 = climate_indices.spi(precipitation, scale=3, distribution=Distribution.gamma)
    spi_6 = climate_indices.spi(
        precipitation,
        scale=6,
        distribution=Distribution.gamma,
        calibration_year_initial=1991,
        calibration_year_final=2020,
    )
    combined = xr.Dataset({"spi_3": spi_3, "spi_6": spi_6})

    with _write_and_reopen(combined, tmp_path / f"out.{engine}", engine) as reopened:
        for name, original in (("spi_3", spi_3), ("spi_6", spi_6)):
            attrs = reopened[name].attrs
            assert read_provenance(reopened[name]) == read_provenance(original)
            for key in (
                "climate_indices_version",
                "climate_indices_git_commit",
                "climate_indices_provenance_schema_version",
            ):
                assert attrs[key] == original.attrs[key]
        assert PROVENANCE_ATTR not in reopened.attrs
        assert read_provenance(reopened["spi_6"])["parameters"]["calibration_year_initial"] == 1991  # type: ignore[index]


def test_metadata_collection_keeps_dask_inputs_lazy_and_values_unchanged() -> None:
    eager_input = provenance.with_source(_monthly(), uri="precip.nc")
    lazy_input = eager_input.expand_dims(lat=[10.0, 20.0], lon=[30.0]).chunk({"time": -1, "lat": 1, "lon": 1})
    lazy_input.encoding["source"] = "/data/precip.nc"

    def forbid(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("building the result triggered a Dask computation")

    with dask.config.set(scheduler=forbid):
        lazy = climate_indices.spi(lazy_input, scale=3, distribution=Distribution.gamma)

    assert lazy.chunks == lazy_input.chunks
    assert lazy.dtype == np.float64
    record = read_provenance(lazy)
    assert record is not None
    assert record["inputs"][0]["chunked"] is True
    assert record["inputs"][0]["coordinates"]["lat"] == {"first": 10.0, "last": 20.0, "size": 2}
    eager = climate_indices.spi(eager_input, scale=3, distribution=Distribution.gamma)
    np.testing.assert_array_equal(lazy.isel(lat=0, lon=0).values, eager.values)
    reference = indices.spi(eager_input.values, 3, Distribution.gamma, 1981, 1981, 2020, compute.Periodicity.monthly)
    np.testing.assert_array_equal(eager.values, reference)


# --- the command line ----------------------------------------------------------------------------


@pytest.mark.parametrize(("initial", "final", "effective_final"), [(1981, 2010, 2010), (1950, 2050, 2020)])
def test_cli_output_records_the_input_as_opened_before_unit_normalization(
    tmp_path: Path, initial: int, final: int, effective_final: int
) -> None:
    precip_path = tmp_path / "precip.nc"
    time = xr.date_range("1981-01-01", periods=480, freq="MS")
    xr.Dataset(
        {"precip": ("time", np.random.default_rng(3).gamma(2.0, 1.2, time.size), {"units": "inches"})},
        coords={"time": time},
        attrs={"title": "Fixture", "history": "2026-01-01: created", "comment": "not kept"},
    ).to_netcdf(precip_path)

    main(
        [
            "--index", "spi", "--periodicity", "monthly", "--scales", "3",
            "--calibration_start_year", str(initial), "--calibration_end_year", str(final),
            "--netcdf_precip", str(precip_path), "--var_name_precip", "precip",
            "--output_file_base", str(tmp_path / "out"), "--multiprocessing", "single",
        ]
    )  # fmt: skip

    with xr.open_dataset(tmp_path / "out_spi_gamma_03.nc") as written:
        record = read_provenance(written["spi_gamma_03"])
    assert record is not None
    assert record["operation"]["interface"] == "cli"
    entry = record["inputs"][0]
    assert (entry["role"], entry["variable"]) == ("precipitation", "precip")
    assert entry["attributes"]["units"] == "inches"  # not the normalized "mm"
    assert entry["source"] == {"kind": "local-file", "name": "precip.nc"}
    assert entry["dataset_attributes"] == {"history": "2026-01-01: created", "title": "Fixture"}
    assert str(tmp_path) not in json.dumps(record)
    assert record["parameters"]["calibration_year_initial"] == 1981
    assert record["parameters"]["calibration_year_final"] == effective_final
    assert record["parameters"]["periodicity"] == "monthly"
    assert record["parameters"]["distribution"] == "gamma"


def test_cli_xarray_route_records_the_input_file_and_its_global_attributes(tmp_path: Path) -> None:
    time = xr.date_range("2000-01-01", periods=730, freq="D")
    xr.Dataset(
        {"tas": ("time", 280.0 + 10.0 * np.sin(np.arange(time.size) / 58.0), {"units": "K"})},
        coords={"time": time},
        attrs={"title": "Daily fixture", "institution": "Test", "comment": "not kept"},
    ).to_netcdf(tmp_path / "tas.nc")

    main(
        [
            "--index", "hdd", "--periodicity", "daily", "--heating_base_temperature", "18",
            "--netcdf_temp", str(tmp_path / "tas.nc"), "--var_name_temp", "tas",
            "--output_file_base", str(tmp_path / "out"), "--multiprocessing", "single",
        ]
    )  # fmt: skip

    with xr.open_dataset(tmp_path / "out_heating_degree_days.nc") as written:
        record = read_provenance(written["heating_degree_days"])
        assert provenance.SOURCE_ATTR not in written["heating_degree_days"].attrs
    assert record is not None
    assert record["parameters"] == {"base_temperature": 18.0}
    [entry] = record["inputs"]
    assert entry["source"] == {"kind": "local-file", "name": "tas.nc"}
    assert entry["dataset_attributes"] == {"institution": "Test", "title": "Daily fixture"}


# --- review regressions --------------------------------------------------------------------------


def test_energy_records_name_the_public_function_not_its_kernel() -> None:
    time = xr.date_range("2000-01-01", periods=3, freq="D")
    temperature = xr.DataArray([5.0, 12.0, 20.0], dims="time", coords={"time": time}, attrs={"units": "degC"})
    wind = xr.DataArray([4.0, 5.0, 6.0], dims="time", coords={"time": time}, attrs={"units": "m s-1"})
    outputs = {
        "hub_height_wind_speed": energy.hub_height_wind_speed(wind, reference_height=10, hub_height=80),
        "wind_shear_exponent": energy.wind_shear_exponent(wind, wind * 1.1, lower_height=10, upper_height=80),
        "heating_degree_days": energy.heating_degree_days(temperature, base_temperature=18.0),
    }

    for name, output in outputs.items():
        record = _strict(output.attrs[PROVENANCE_ATTR])
        assert record["operation"]["implementation"] == f"climate_indices.energy.{name}"


def test_flood_records_the_precipitation_units_after_conversion_to_millimeters() -> None:
    # docs/provenance.md names this as the one input family that does not keep its source units
    time = xr.date_range("2000-01-01", periods=60, freq="D")
    precipitation = xr.DataArray(
        np.random.default_rng(0).random(time.size), dims="time", coords={"time": time}, attrs={"units": "inches"}
    )

    record = _strict(flood.effective_precipitation(precipitation).attrs[PROVENANCE_ATTR])

    assert record["inputs"][0]["attributes"]["units"] == "mm"


def test_cli_spei_records_only_parameters_the_computation_used(tmp_path: Path) -> None:
    time = xr.date_range("1981-01-01", periods=480, freq="MS")
    rng = np.random.default_rng(5)
    for name, shape in (("precip", 30.0), ("pet", 10.0)):
        xr.Dataset(
            {
                name: (("division", "time"), rng.gamma(2.0, shape, (1, time.size)), {"units": "mm"}),
                "lat": (("division",), [40.0]),  # the CLI reads a per-division latitude
            },
            coords={"division": [1], "time": time},
        ).to_netcdf(tmp_path / f"{name}.nc")

    main(
        [
            "--index", "spei", "--periodicity", "monthly", "--scales", "3",
            "--calibration_start_year", "1981", "--calibration_end_year", "2010",
            "--netcdf_precip", str(tmp_path / "precip.nc"), "--var_name_precip", "precip",
            "--netcdf_pet", str(tmp_path / "pet.nc"), "--var_name_pet", "pet",
            "--output_file_base", str(tmp_path / "out"), "--multiprocessing", "single",
        ]
    )  # fmt: skip

    with xr.open_dataset(tmp_path / "out_spei_gamma_03.nc") as written:
        record = read_provenance(written["spei_gamma_03"])
    assert record is not None
    assert "zero_handling" not in record["parameters"]  # indices.spei has no such parameter
    assert record["parameters"]["scale"] == 3
    assert [entry["role"] for entry in record["inputs"]] == ["precipitation", "pet"]


def test_log_wind_profile_records_the_effective_roughness_length() -> None:
    wind = xr.DataArray([5.0, 10.0], dims="time", attrs={"units": "m s-1"})
    records = []
    for roughness in (0.01, 0.1):
        output = energy.hub_height_wind_speed(wind, 10.0, 100.0, method="log", roughness_length=roughness)
        records.append(_strict(output.attrs[PROVENANCE_ATTR]))
    assert [record["parameters"]["roughness_length"] for record in records] == [0.01, 0.1]
    assert records[0]["record_id"] != records[1]["record_id"]


def test_positional_scientific_parameters_describe_the_eager_calculation() -> None:
    @xarray_adapter(infer_params=False)
    def gain(values: Any, factor: float = 1.0) -> Any:
        return values * factor

    data = xr.DataArray([1.0, 2.0], dims="time")
    output = gain(data, 2.0)
    assert isinstance(output, xr.DataArray)
    np.testing.assert_array_equal(output.values, [2.0, 4.0])
    assert _strict(output.attrs[PROVENANCE_ATTR])["parameters"] == {"factor": 2.0}


@pytest.mark.parametrize("lazy", [False, True])
def test_cffwis_records_inferred_scalar_latitude_and_initial_state(lazy: bool) -> None:
    time = xr.date_range("2001-07-01T12:00:00", periods=3, freq="D")
    temperature = xr.DataArray([20.0] * 3, dims="time", coords={"time": time, "lat": 46.0}, attrs={"units": "degC"})
    humidity = xr.full_like(temperature, 45.0)
    wind = xr.full_like(temperature, 3.0)
    rain = xr.zeros_like(temperature)
    rain.attrs["units"] = "mm"
    state = fire.CFFWISState(
        ffmc=fire.FFMCState(ffmc=np.array(85.0), trailing_gap_days=np.array(0)),
        dmc=fire.DMCState(dmc=np.array(6.0), trailing_gap_days=None),
        dc=fire.DCState(dc=np.array(15.0), trailing_gap_days=np.array(1)),
    )
    if lazy:
        temperature, humidity, wind, rain = (data.chunk({"time": -1}) for data in (temperature, humidity, wind, rain))

    def forbid(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("provenance triggered a Dask computation")

    with dask.config.set(scheduler=forbid):
        output = fire.cffwis(temperature, humidity, wind, rain, initial_state=state, outputs="fwi")
    assert isinstance(output, xr.Dataset)
    record = _strict(output["fwi"].attrs[PROVENANCE_ATTR])
    inputs = {entry["role"]: entry for entry in record["inputs"]}
    assert inputs["latitude_degrees_north"]["value"] == 46.0
    assert record["parameters"]["initial_state"] == {
        "ffmc": {"ffmc": 85.0, "trailing_gap_days": 0},
        "dmc": {"dmc": 6.0, "trailing_gap_days": None},
        "dc": {"dc": 15.0, "trailing_gap_days": 1},
    }


def test_kbdi_records_state_values_units_and_wet_spell_bookkeeping() -> None:
    state = fire.KBDIState(
        kbdi=np.array([10.0, 20.0]),
        wet_spell_precipitation=np.array([2.0, 3.0]),
        trailing_gap_days=np.array([0, 1]),
        units="imperial",
    )
    attrs = build_output_attrs(_monthly(), parameters={"initial_state": state})
    assert _strict(attrs[PROVENANCE_ATTR])["parameters"]["initial_state"] == {
        "kbdi": [10.0, 20.0],
        "wet_spell_precipitation": [2.0, 3.0],
        "trailing_gap_days": [0, 1],
        "units": "imperial",
    }


@pytest.mark.parametrize("engine", ["h5netcdf", "zarr"])
def test_reopened_source_and_lineage_metadata_are_validated_and_sanitized(tmp_path: Path, engine: str) -> None:
    locator = {"kind": "url", "url": "https://alice:password@host/p.nc?token=secret#part"}
    step = {
        "record_id": "upstream-step",
        "parameters": {"api_key": "hidden-key", "nested": {"token": "hidden-token"}},
        "inputs": [{"role": "original", "source": locator}],
    }
    data = _monthly()
    data.attrs[provenance.SOURCE_ATTR] = json.dumps(
        {
            "source": locator,
            "role": "wrong-role",
            "attributes": {"secret": "injected-attributes"},
            "identifiers": {"dataset_id": {"password": "hidden-id"}},
            "dataset_attributes": {"history": "downloaded https://bob:pw@host/p.nc?key=hidden-query"},
        }
    )
    data.attrs[PROVENANCE_ATTR] = json.dumps(
        {
            **step,
            "schema": provenance.SCHEMA,
            "record_id": "parent",
            "inputs": [{"role": "previous", "source": locator, "upstream": [step]}],
        }
    )
    with _write_and_reopen(xr.Dataset({"prcp": data}), tmp_path / f"input.{engine}", engine) as reopened:
        attrs = build_output_attrs(_monthly(), inputs={"secondary": reopened["prcp"]})
    serialized = attrs[PROVENANCE_ATTR]
    entry = _strict(serialized)["inputs"][0]
    assert entry["role"] == "secondary"
    assert entry["attributes"] == {"units": "mm"}
    assert entry["source"] == {"kind": "url", "url": "https://host/p.nc", "query_removed": True}
    assert [step["record_id"] for step in entry["upstream"]] == ["parent", "upstream-step"]
    for secret in ("alice", "bob", "password@", "pw@", "token=secret", "hidden-", "injected-", "#part"):
        assert secret not in serialized
    assert entry["upstream"][1]["parameters"]["nested"]["token"] == "[redacted]"


@pytest.mark.parametrize(
    "payload",
    [
        {"inputs": None},
        {"inputs": {}},
        {"record_id": []},
        {"record_id": {}},
        {"inputs": [{"upstream": None}]},
        {"inputs": [{"upstream": {}}]},
        {"inputs": [{"upstream": [{"record_id": []}]}]},
    ],
)
def test_structurally_invalid_lineage_is_nonfatal(payload: dict[str, Any]) -> None:
    data = _monthly()
    data.attrs[PROVENANCE_ATTR] = json.dumps({"schema": provenance.SCHEMA, **payload})
    attrs = build_output_attrs(data)
    assert {"unparseable": True} in _strict(attrs[PROVENANCE_ATTR])["inputs"][0]["upstream"]


@pytest.mark.parametrize(
    "path",
    [
        "file:///srv/private/customer-a/pet.nc",
        "file://bob:pw@host/private/pet.nc?token=secret#part",
        "/home/alice/private/pet.nc",
        r"C:\Users\alice\private\pet.nc",
    ],
)
def test_secondary_history_does_not_disclose_absolute_directories(path: str) -> None:
    data = _monthly(history=f"processed {path}")
    attrs = build_output_attrs(_monthly(), inputs={"secondary": data})
    assert _strict(attrs[PROVENANCE_ATTR])["inputs"][0]["attributes"]["history"] == "processed pet.nc"


def test_long_locators_and_unicode_identifiers_cannot_exceed_the_final_budget() -> None:
    data = provenance.with_source(
        _monthly(),
        uri="https://example.org/" + "x" * 70000,
        dataset_id="\U0001f30d" * 1024,
        doi="\U0001f30d" * 1024,
        version="\U0001f30d" * 1024,
    )
    attrs = build_output_attrs(data, inputs={f"input{i}": data for i in range(4)})
    serialized = attrs[PROVENANCE_ATTR]
    assert len(serialized.encode("ascii")) <= 32 * 1024
    record = _strict(serialized)
    assert record["truncated"]
    record_id = record.pop("record_id")
    assert record_id == "sha256:" + hashlib.sha256(provenance._canonical(record).encode("ascii")).hexdigest()


def test_final_size_check_includes_record_id_and_truncation_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    attrs = build_output_attrs(_monthly(), parameters={"comment": "x" * 900})
    record = _strict(attrs[PROVENANCE_ATTR])
    record.pop("record_id")
    budget = len(provenance._canonical(record)) + 10
    monkeypatch.setattr(provenance, "_MAX_RECORD_BYTES", budget)
    attrs = build_output_attrs(_monthly(), parameters={"comment": "x" * 900})
    assert len(attrs[PROVENANCE_ATTR].encode("ascii")) <= budget
    assert "truncated" in _strict(attrs[PROVENANCE_ATTR])


def test_cli_pet_records_division_latitude_as_a_consumed_input(tmp_path: Path) -> None:
    time = xr.date_range("1981-01-01", periods=24, freq="MS")
    xr.Dataset(
        {
            "temp": (("division", "time"), np.full((1, 24), 15.0), {"units": "degC"}),
            "lat": ("division", [40.0], {"units": "degrees_north"}),
        },
        coords={"division": [101], "time": time},
    ).to_netcdf(tmp_path / "temp.nc")
    main(
        [
            "--index",
            "pet",
            "--periodicity",
            "monthly",
            "--netcdf_temp",
            str(tmp_path / "temp.nc"),
            "--var_name_temp",
            "temp",
            "--output_file_base",
            str(tmp_path / "out"),
            "--multiprocessing",
            "single",
        ]
    )
    with xr.open_dataset(tmp_path / "out_pet_thornthwaite.nc") as written:
        record = _strict(written["pet_thornthwaite"].attrs[PROVENANCE_ATTR])
    inputs = {entry["role"]: entry for entry in record["inputs"]}
    assert inputs["latitude"]["variable"] == "lat"
    assert inputs["latitude"]["dims"] == ["division"]
    assert inputs["latitude"]["source"] == {"kind": "local-file", "name": "temp.nc"}


# --- software identity ---------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    assert GIT is not None
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.org",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.org",
    )
    return subprocess.run(
        [GIT, "-C", str(root), *args], capture_output=True, text=True, check=True, env=env
    ).stdout.strip()


def _checkout(root: Path, *, commit: bool = True) -> Path:
    """A minimal climate_indices source checkout; returns its package directory."""
    package = root / "src" / "climate_indices"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text("[project]\nname = 'climate_indices'\n")
    _git(root, "init", "-q")
    if commit:
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "init")
    return package


def _build_info(package: Path, commit: str, dirty: bool = False) -> None:
    (package / "_build_info.json").write_text(json.dumps({"git_commit": commit, "git_dirty": dirty}))


class _FakeDistribution(metadata.Distribution):
    """An installed-distribution stand-in with a chosen version, files, and direct_url.json."""

    def __init__(
        self,
        site: Path,
        *,
        version: str = "9.9.9",
        direct_url: dict[str, Any] | None = None,
        repository: str = "https://github.com/monocongo/climate_indices",
    ) -> None:
        self._site = site
        self._files = {
            "METADATA": f"Metadata-Version: 2.1\nName: climate_indices\nVersion: {version}\n"
            f"Project-URL: Repository, {repository}\n"
        }
        if direct_url is not None:
            self._files["direct_url.json"] = json.dumps(direct_url)

    def read_text(self, filename: str) -> str | None:
        return self._files.get(filename)

    def locate_file(self, path: str | os.PathLike[str]) -> Path:
        return self._site / path


@needs_git
def test_a_source_checkout_reports_head_and_tracked_dirty_state(tmp_path: Path) -> None:
    package = _checkout(tmp_path)
    head = _git(tmp_path, "rev-parse", "HEAD")
    _build_info(package, "a" * 40)  # stale embedded metadata in a checkout is never trusted
    (tmp_path / "untracked.txt").write_text("x")

    clean = resolve_software_identity(package, None, GIT)
    (package / "__init__.py").write_text("# modified\n")
    dirty = resolve_software_identity(package, None, GIT)

    assert (clean["git_commit"], clean["git_commit_source"], clean["git_dirty"]) == (head, "source-checkout", False)
    assert (dirty["git_commit"], dirty["git_dirty"]) == (head, True)
    assert clean["version"] == "unknown"


@needs_git
def test_an_unrelated_repository_cannot_supply_the_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _checkout(tmp_path / "checkout")
    head = _git(tmp_path / "checkout", "rev-parse", "HEAD")
    other = tmp_path / "other"
    _checkout(other)
    monkeypatch.chdir(other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))

    assert resolve_software_identity(package, None, GIT)["git_commit"] == head
    # an installed copy that happens to sit in an unrelated repository is not a checkout
    site = other / "site-packages" / "climate_indices"
    site.mkdir(parents=True)
    assert resolve_software_identity(site, None, GIT)["git_commit"] is None


@needs_git
def test_missing_git_or_an_unborn_head_degrade_to_unknown(tmp_path: Path) -> None:
    package = _checkout(tmp_path / "checkout")
    _build_info(package, "b" * 40)
    unborn = _checkout(tmp_path / "unborn", commit=False)

    without_git = resolve_software_identity(package, None, None)
    no_commit = resolve_software_identity(unborn, None, GIT)

    assert (without_git["git_commit"], without_git["git_commit_source"]) == (None, "unavailable")
    assert "embedded build metadata is not used" in without_git["notes"][0]
    assert no_commit["git_commit"] is None
    assert no_commit["git_dirty"] is None


def test_an_installed_package_reports_its_embedded_build_commit(tmp_path: Path) -> None:
    package = tmp_path / "climate_indices"
    package.mkdir()
    (package / "__init__.py").write_text("")
    _build_info(package, "c" * 40, dirty=True)

    identity = resolve_software_identity(package, _FakeDistribution(tmp_path, version="3.1.0+g1234567"), GIT)

    assert identity == {
        "distribution": "climate_indices",
        "version": "3.1.0+g1234567",
        "repository_url": "https://github.com/monocongo/climate_indices",
        "git_commit": "c" * 40,
        "git_commit_source": "build-metadata",
        "git_dirty": True,
    }


def test_the_repository_url_loses_credentials_and_query(tmp_path: Path) -> None:
    fake = _FakeDistribution(tmp_path, repository="https://alice:hunter2@github.com/o/r?token=t#part")

    identity = resolve_software_identity(tmp_path, fake, None)

    assert identity["repository_url"] == "https://github.com/o/r?[redacted]"


@pytest.mark.parametrize(
    "build_info",
    [None, "not json", json.dumps({"git_commit": "v3.1.0"}), json.dumps({"git_commit": "1234567"})],
)
def test_an_absent_or_invalid_build_commit_is_unknown_never_inferred_from_the_version(
    tmp_path: Path, build_info: str | None
) -> None:
    package = tmp_path / "climate_indices"
    package.mkdir()
    (package / "__init__.py").write_text("")
    if build_info is not None:
        (package / "_build_info.json").write_text(build_info)

    identity = resolve_software_identity(package, _FakeDistribution(tmp_path, version="3.1.0+g1234567"), None)

    assert (identity["git_commit"], identity["git_commit_source"], identity["version"]) == (
        None,
        "unavailable",
        "3.1.0+g1234567",
    )


def test_direct_url_is_used_only_for_the_imported_distribution_and_conflicts_are_unknown(tmp_path: Path) -> None:
    package = tmp_path / "climate_indices"
    package.mkdir()
    (package / "__init__.py").write_text("")
    vcs = {"url": "https://github.com/monocongo/climate_indices", "vcs_info": {"vcs": "git", "commit_id": "d" * 40}}

    matching = resolve_software_identity(package, _FakeDistribution(tmp_path, direct_url=vcs), None)
    elsewhere = resolve_software_identity(package, _FakeDistribution(tmp_path / "other", direct_url=vcs), None)
    _build_info(package, "e" * 40)
    conflicting = resolve_software_identity(package, _FakeDistribution(tmp_path, direct_url=vcs), None)

    assert (matching["git_commit"], matching["git_commit_source"], matching["git_dirty"]) == (
        "d" * 40,
        "direct-url",
        None,
    )
    assert elsewhere["git_commit"] is None
    assert "may not describe the imported package" in elsewhere["notes"][0]
    assert conflicting["git_commit"] is None
    assert {candidate["git_commit"] for candidate in conflicting["conflicts"]} == {"d" * 40, "e" * 40}


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="the fake git is a POSIX shell script")
def test_git_runs_once_per_process_and_never_on_import(tmp_path: Path) -> None:
    marker = tmp_path / "git-calls"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(f'#!/bin/sh\necho "$@" >> "{marker}"\nexec "{GIT}" "$@"\n')
    fake_git.chmod(0o755)
    script = (
        "import pathlib, sys\n"
        "import climate_indices\n"
        "from climate_indices import provenance\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "print(marker.exists())\n"
        "provenance.software_identity(); provenance.software_identity()\n"
        "print(len(marker.read_text().splitlines()) if marker.exists() else 0)\n"
    )
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    output = subprocess.run(
        [sys.executable, "-c", script, str(marker)], capture_output=True, text=True, check=True, env=env, cwd=tmp_path
    ).stdout.split()

    assert output[0] == "False"  # importing climate_indices ran no git command
    from_checkout = provenance.software_identity()["git_commit_source"] == "source-checkout"
    assert int(output[-1]) == (3 if from_checkout else 0)  # one resolution, three git calls

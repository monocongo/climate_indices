"""Offline tests of the intercomparison harness (REFIMPL-08, issue 1610).

The contract, the driver, the result store and the manifest are exercised with the
`climate_indices` adapter and a fake adapter. No comparator environment is built or imported
here: the xclim adapter runs only when the slice is run by hand (intercomparison/README.md).
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import textwrap
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
HARNESS = ROOT / "intercomparison"
sys.path.insert(0, str(HARNESS))  # the adapters import `contract` the same way
import contract  # noqa: E402


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


harness = _load("intercomparison_run", HARNESS / "run.py")
CI_ADAPTER = HARNESS / "adapters" / "climate_indices_adapter.py"
SCHEMA = json.loads((HARNESS / "manifest.schema.json").read_text())
ADAPTERS = {"climate_indices": [sys.executable, str(CI_ADAPTER)]}

# build_manifest records the harness commit, so every test that runs a study needs a checkout
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _validate(manifest: dict[str, Any]) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(manifest, SCHEMA)


def _run_study(path: Path, adapters: dict[str, list[str]] = ADAPTERS) -> dict[str, Any]:
    pytest.importorskip("zarr")
    return harness.run_study(path, "0101", 3, adapters)  # type: ignore[no-any-return]


def _result_dir(tmp_path: Path) -> Path:
    """A valid result directory with one output series."""
    out = tmp_path / "out"
    out.mkdir()
    digest = contract.write_series(out / "spi.csv", ["2000-01"], [1.0])
    contract.write_json(
        out / "result.json",
        {
            "schema_version": 1,
            "comparator": "c",
            "version": "1",
            "effective_method": {"a": 1},
            "software": {},
            "platform": "p",
            "wall_time_s": 0.0,
            "warnings": [],
            "outputs": {"spi": {"path": "spi.csv", "sha256": digest}},
        },
    )
    return out


def test_series_round_trips_binary64_and_nan_exactly(tmp_path: Path) -> None:
    values = [0.1 + 0.2, 1e-310, -0.0, math.nan, 2.0**53 + 2.0, 1 / 3]
    times = [f"2000-{m:02d}" for m in range(1, 7)]
    digest = contract.write_series(tmp_path / "s.csv", times, values)

    got_times, got = contract.read_series(tmp_path / "s.csv", digest)

    assert got_times == times
    assert [math.isnan(v) for v in got] == [math.isnan(v) for v in values]
    # exact: repr(float) is the shortest round-trip string, so no tolerance is needed
    assert [v for v in got if not math.isnan(v)] == [v for v in values if not math.isnan(v)]
    assert math.copysign(1.0, got[2]) == -1.0


def test_series_read_with_the_wrong_hash_is_rejected(tmp_path: Path) -> None:
    contract.write_series(tmp_path / "s.csv", ["2000-01"], [1.0])
    with pytest.raises(contract.ContractError, match="SHA-256"):
        contract.read_series(tmp_path / "s.csv", "0" * 64)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "header"),
        ("t,v\n2000-01,1.0\n", "header"),
        ("time,value\n2000-01\n", "exactly"),
        ("time,value\n\n", "exactly"),
        ("time,value\n2000-01,one\n", "could not convert"),
    ],
)
def test_malformed_series_raises_contract_error(tmp_path: Path, text: str, match: str) -> None:
    (tmp_path / "s.csv").write_text(text)
    with pytest.raises(contract.ContractError, match=match):
        contract.read_series(tmp_path / "s.csv")


def test_request_with_a_tampered_input_is_rejected(tmp_path: Path) -> None:
    values = np.arange(1.0, 25.0)
    request = harness.write_request(
        tmp_path,
        study="t",
        unit="u",
        times=harness.monthly_times(2000, 24),
        values=values,
        units="mm",
        positions=harness.positions_for(3, 2000, 2001),
    )
    contract.read_request(request)  # untouched: accepted

    (tmp_path / "precip.csv").write_text((tmp_path / "precip.csv").read_text().replace("2.0", "2.5", 1))

    with pytest.raises(contract.ContractError, match="SHA-256"):
        contract.read_request(request)


def test_result_with_a_tampered_output_is_rejected(tmp_path: Path) -> None:
    out = _result_dir(tmp_path)
    contract.read_result(out / "result.json")

    contract.write_series(out / "spi.csv", ["2000-01"], [2.0])

    with pytest.raises(contract.ContractError, match="SHA-256"):
        contract.read_result(out / "result.json")


def _escape(result: dict[str, Any]) -> None:
    result["outputs"]["spi"]["path"] = "../elsewhere.csv"


def _absolute(result: dict[str, Any]) -> None:
    result["outputs"]["spi"]["path"] = "/etc/hosts"


def _no_hash(result: dict[str, Any]) -> None:
    del result["outputs"]["spi"]["sha256"]


def _missing_file(result: dict[str, Any]) -> None:
    result["outputs"]["spi"]["path"] = "absent.csv"


def _drop_key(result: dict[str, Any]) -> None:
    del result["wall_time_s"]


def _schema_version(result: dict[str, Any]) -> None:
    result["schema_version"] = 2


def _method_not_an_object(result: dict[str, Any]) -> None:
    result["effective_method"] = "gamma"


def _warnings_a_string(result: dict[str, Any]) -> None:
    result["warnings"] = "fit failed"


def _version_null(result: dict[str, Any]) -> None:
    result["version"] = None


def _negative_wall_time(result: dict[str, Any]) -> None:
    result["wall_time_s"] = -1.0


def _boolean_wall_time(result: dict[str, Any]) -> None:
    result["wall_time_s"] = True


def _software_version_not_a_string(result: dict[str, Any]) -> None:
    result["software"] = {"numpy": 2}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_warnings_a_string, "'warnings' must be a list of strings"),
        (_version_null, "'version' must be a non-empty string"),
        (_negative_wall_time, "'wall_time_s'"),
        (_boolean_wall_time, "'wall_time_s'"),
        (_software_version_not_a_string, "'software' version"),
        (_escape, "outside"),
        (_absolute, "outside"),
        (_no_hash, "string 'path' and 'sha256'"),
        (_missing_file, "does not exist"),
        (_drop_key, "missing"),
        (_schema_version, "schema_version"),
        (_method_not_an_object, "must be an object"),
    ],
)
def test_malformed_result_raises_contract_error(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], match: str
) -> None:
    out = _result_dir(tmp_path)
    (tmp_path / "elsewhere.csv").write_text((out / "spi.csv").read_text())  # same bytes, so only the path is wrong
    result = json.loads((out / "result.json").read_text())
    mutate(result)
    contract.write_json(out / "result.json", result)

    with pytest.raises(contract.ContractError, match=match):
        contract.read_result(out / "result.json")


def test_request_missing_a_required_key_is_rejected(tmp_path: Path) -> None:
    contract.write_json(tmp_path / "request.json", {"schema_version": 1, "study": "t"})
    with pytest.raises(contract.ContractError, match="missing"):
        contract.read_request(tmp_path / "request.json")


def test_diff_method_flags_a_changed_position_and_a_silent_one() -> None:
    requested = {"distribution": "gamma", "estimator": "thom_closed_form", "scale": 3}
    assert contract.diff_method(requested, dict(requested)) == []
    warnings = contract.diff_method(requested, {"distribution": "gamma", "estimator": "mle"})
    assert len(warnings) == 2
    assert "'estimator'" in warnings[0] and "'mle'" in warnings[0]
    assert "<not reported>" in warnings[1]  # silence is not agreement


@pytest.mark.parametrize(("requested", "applied"), [(3, 3.0), (1, True), (3, "3"), (None, None)])
def test_diff_method_distinguishes_values_that_compare_equal_or_are_absent(requested: Any, applied: Any) -> None:
    """3 == 3.0 and 1 == True in Python; a position must match in type too. A requested None is not 'reported'."""
    reported = {"p": applied}
    if requested is None:
        assert contract.diff_method({"p": requested}, {}) != []  # silence is not agreement, even for None
        assert contract.diff_method({"p": requested}, reported) == []  # a reported None matches
    else:
        assert len(contract.diff_method({"p": requested}, reported)) == 1


def test_consecutive_months_are_checked() -> None:
    contract.require_consecutive_months(["1999-11", "1999-12", "2000-01"])
    contract.require_consecutive_months(["2000-01", "2000-02"], january_start=True)
    for bad in (["2000-01", "2000-03"], ["2000-01", "2000-01"], ["2000-12", "2000-01"], [], ["x"]):
        with pytest.raises(contract.ContractError):
            contract.require_consecutive_months(bad)
    with pytest.raises(contract.ContractError, match="first value as January"):
        contract.require_consecutive_months(["2000-03", "2000-04"], january_start=True)


def test_adapter_command_line_rejects_bad_usage() -> None:
    assert contract.run_adapter("x", lambda r, i: {}, ["run", "only-one-argument"]) == 2


def test_warnings_raised_inside_compute_are_recorded_in_the_result(tmp_path: Path) -> None:
    request = harness.write_request(
        tmp_path / "work",
        study="t",
        unit="u",
        times=harness.monthly_times(2000, 24),
        values=np.arange(1.0, 25.0),
        units="mm",
        positions=harness.positions_for(3, 2000, 2001),
    )

    def compute(request: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        warnings.warn("poor gamma fit", UserWarning, stacklevel=1)
        times, values = inputs["precip"]
        return {
            "version": "1",
            "effective_method": {"a": 1},
            "software": {},
            "warnings": ["returned"],
            "series": {"spi": (times, values)},
        }

    out = tmp_path / "out"
    assert contract.run_adapter("c", compute, ["run", str(request), str(out)]) == 0

    assert contract.read_result(out / "result.json")["warnings"] == ["returned", "poor gamma fit"]


@needs_git
def test_slice_runs_end_to_end_and_the_store_matches_the_library(tmp_path: Path) -> None:
    from climate_indices import spi
    from climate_indices.compute import Periodicity
    from climate_indices.vocabulary import Distribution

    manifest = _run_study(tmp_path / "slice")

    _validate(manifest)
    assert manifest == json.loads((tmp_path / "slice" / "manifest.json").read_text())
    (array,) = manifest["arrays"]
    assert array["implementation"] == "climate_indices"
    assert array["warnings"] == []
    assert manifest["inputs"]["precip"]["length"] == 1536
    assert (manifest["inputs"]["precip"]["first"], manifest["inputs"]["precip"]["last"]) == ("1895-01", "2022-12")

    import zarr

    stored = np.asarray(zarr.open_group(str(tmp_path / "slice" / "store.zarr"), mode="r")["values"])
    assert stored.shape == tuple(manifest["store"]["shape"]) == (1, 1, 1536)
    precip = np.load(ROOT / "tests" / "fixture" / "palmer" / "0101" / "precips.npy")
    direct = spi(precip, 3, Distribution.gamma, 1895, 1895, 2022, Periodicity.monthly)
    # plumbing only: bytes through CSV and Zarr must equal a direct call exactly
    np.testing.assert_array_equal(stored[0, 0], direct)
    assert np.isnan(stored[0, 0, :2]).all()  # the first scale-1 months have no 3-month sum


@needs_git
def test_rerun_gives_identical_output_hashes(tmp_path: Path) -> None:
    first = _run_study(tmp_path / "a")
    second = _run_study(tmp_path / "b")
    # the CSV hashes are of float64 text, so identical hashes mean bit-identical series
    assert [a["output_sha256"] for a in first["arrays"]] == [a["output_sha256"] for a in second["arrays"]]


def test_the_climate_indices_adapter_applies_what_it_is_given_and_reports_it(tmp_path: Path) -> None:
    """A request for another distribution is not honoured, and the difference is visible, not silent."""
    positions = harness.positions_for(3, 2000, 2003) | {"distribution": "pearson", "scale": 6}
    request = harness.write_request(
        tmp_path / "work",
        study="t",
        unit="u",
        times=harness.monthly_times(2000, 48),
        values=np.linspace(1.0, 9.0, 48) + np.sin(np.arange(48)),
        units="mm",
        positions=positions,
    )

    result = harness.run_adapter("climate_indices", ADAPTERS["climate_indices"], request, tmp_path / "out")

    assert result["effective_method"]["scale"] == 6  # an argument of the call: followed
    assert result["effective_method"]["distribution"] == "gamma"  # the adapter only runs gamma
    (warning,) = contract.diff_method(positions, result["effective_method"])
    assert "'distribution'" in warning and "'pearson'" in warning and "'gamma'" in warning


def test_the_climate_indices_adapter_refuses_a_series_that_does_not_start_in_january(tmp_path: Path) -> None:
    request = harness.write_request(
        tmp_path / "work",
        study="t",
        unit="u",
        times=harness.monthly_times(2000, 24)[2:],
        values=np.arange(1.0, 23.0),
        units="mm",
        positions=harness.positions_for(3, 2000, 2001),
    )
    with pytest.raises(Exception, match="exit status"):  # CalledProcessError: the adapter exited non-zero
        harness.run_adapter("climate_indices", ADAPTERS["climate_indices"], request, tmp_path / "out")


def test_an_adapter_that_exits_zero_without_writing_is_not_credited_with_an_old_result(tmp_path: Path) -> None:
    request = harness.write_request(
        tmp_path / "work",
        study="t",
        unit="u",
        times=harness.monthly_times(2000, 48),
        values=np.linspace(1.0, 9.0, 48) + np.sin(np.arange(48)),
        units="mm",
        positions=harness.positions_for(3, 2000, 2003),
    )
    out = tmp_path / "out"
    harness.run_adapter("climate_indices", ADAPTERS["climate_indices"], request, out)  # leaves a valid result behind

    with pytest.raises(contract.ContractError, match="wrote no result.json"):
        harness.run_adapter("lazy", [sys.executable, "-c", "pass"], request, out)


def test_an_output_directory_that_cannot_be_emptied_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = harness.write_request(
        tmp_path / "work",
        study="t",
        unit="u",
        times=harness.monthly_times(2000, 48),
        values=np.linspace(1.0, 9.0, 48) + np.sin(np.arange(48)),
        units="mm",
        positions=harness.positions_for(3, 2000, 2003),
    )
    out = tmp_path / "out"
    harness.run_adapter("climate_indices", ADAPTERS["climate_indices"], request, out)  # leaves a valid result behind

    def refuse(path: Path) -> None:
        raise PermissionError(path)

    monkeypatch.setattr(harness.shutil, "rmtree", refuse)
    marker = tmp_path / "ran"
    adapter = [sys.executable, "-c", f"import pathlib; pathlib.Path({str(marker)!r}).touch()"]
    with pytest.raises(PermissionError):
        harness.run_adapter("lazy", adapter, request, out)

    assert not marker.exists()  # the adapter never started, so the old result cannot be credited to it


@needs_git
def test_driver_records_a_warning_when_an_adapter_applied_another_method(tmp_path: Path) -> None:
    fake = tmp_path / "fake_adapter.py"
    fake.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(HARNESS)!r})
            import contract

            def compute(request, inputs):
                times, values = inputs["precip"]
                method = dict(request["positions"], estimator="mle")
                method.pop("zero_handling")
                return {{"version": "0", "effective_method": method, "software": {{}},
                         "warnings": ["own warning"], "series": {{"spi": (times, [0.0] * len(times))}}}}

            raise SystemExit(contract.run_adapter("fake", compute, sys.argv[1:]))
            """
        )
    )

    manifest = _run_study(tmp_path / "s", {"fake": [sys.executable, str(fake)]})

    _validate(manifest)
    warnings = manifest["arrays"][0]["warnings"]
    assert warnings[0] == "own warning"
    assert any("'estimator'" in w and "'mle'" in w for w in warnings)
    assert any("'zero_handling'" in w and "<not reported>" in w for w in warnings)


@needs_git
def test_a_rerun_that_fails_leaves_no_stale_manifest(tmp_path: Path) -> None:
    study = tmp_path / "s"
    _run_study(study)
    assert (study / "manifest.json").is_file()

    with pytest.raises(Exception, match="exit status"):
        harness.run_study(study, "0101", 3, {"broken": [sys.executable, "-c", "raise SystemExit(1)"]})

    assert not (study / "manifest.json").exists()


@needs_git
def test_a_rerun_that_fails_before_any_adapter_runs_leaves_no_stale_manifest(tmp_path: Path) -> None:
    study = tmp_path / "s"
    _run_study(study)
    assert (study / "manifest.json").is_file()

    with pytest.raises(FileNotFoundError):  # an unknown division: np.load fails before any adapter
        harness.run_study(study, "9999", 3, ADAPTERS)

    assert not (study / "manifest.json").exists()


@needs_git
def test_manifest_without_a_required_field_is_rejected(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    manifest = _run_study(tmp_path / "s")
    for field in ("checksum_sha256", "harness", "arrays"):
        broken = {k: v for k, v in manifest.items() if k != field}
        with pytest.raises(jsonschema.ValidationError, match=field):
            _validate(broken)
    broken = json.loads(json.dumps(manifest))
    del broken["arrays"][0]["effective_method"]
    with pytest.raises(jsonschema.ValidationError, match="effective_method"):
        _validate(broken)


def test_committed_slice_manifest_is_valid_and_describes_the_committed_fixture(tmp_path: Path) -> None:
    manifest = json.loads((HARNESS / "studies" / "refimpl-08-slice" / "manifest.json").read_text())
    _validate(manifest)
    assert {a["implementation"] for a in manifest["arrays"]} == {"climate_indices", "xclim"}
    for array in manifest["arrays"]:
        assert array["version"] and array["effective_method"] and array["output_sha256"]
    assert manifest["harness"]["git_dirty"] is False
    by_name = {a["implementation"]: a for a in manifest["arrays"]}
    assert by_name["climate_indices"]["warnings"] == []
    # xclim rolls a mean where the request says sum; the two give the same gamma SPI (ledger P08-A)
    assert by_name["xclim"]["warnings"] == ["method: position 'accumulation': requested 'sum', applied 'mean'"]

    # the fixture file, and the CSV the comparators read, are the ones the manifest hashes
    source = ROOT / manifest["inputs"]["precip"]["source_path"]
    assert contract.sha256_file(source) == manifest["inputs"]["precip"]["source_sha256"]
    times = harness.monthly_times(1895, manifest["inputs"]["precip"]["length"])
    rebuilt = contract.write_series(tmp_path / "precip.csv", times, np.load(source).tolist())
    assert rebuilt == manifest["inputs"]["precip"]["sha256"] == manifest["checksum_sha256"]

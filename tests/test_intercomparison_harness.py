"""Offline tests of the intercomparison harness (REFIMPL-08, issue 1610).

The contract, the driver, the result store and the manifest are exercised with the
`climate_indices` adapter and a fake adapter. No comparator environment is built or imported
here: the xclim adapter runs only when the slice is run by hand (intercomparison/README.md).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import textwrap
from pathlib import Path
from types import ModuleType

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


def _validate(manifest: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(manifest, SCHEMA)


def test_series_round_trips_binary64_and_nan_exactly(tmp_path: Path) -> None:
    values = [0.1 + 0.2, 1e-310, -0.0, math.nan, 2.0**53 + 2.0, 1 / 3]
    times = [f"2000-{m:02d}" for m in range(1, 7)]
    contract.write_series(tmp_path / "s.csv", times, values)

    got_times, got = contract.read_series(tmp_path / "s.csv")

    assert got_times == times
    assert [math.isnan(v) for v in got] == [math.isnan(v) for v in values]
    # exact: repr(float) is the shortest round-trip string, so no tolerance is needed
    assert [v for v in got if not math.isnan(v)] == [v for v in values if not math.isnan(v)]
    assert math.copysign(1.0, got[2]) == -1.0


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
    contract.write_series(tmp_path / "spi.csv", ["2000-01"], [1.0])
    result = {
        "schema_version": 1, "comparator": "c", "version": "1", "effective_method": {"a": 1}, "software": {},
        "platform": "p", "wall_time_s": 0.0, "warnings": [],
        "outputs": {"spi": {"path": "spi.csv", "sha256": contract.sha256_file(tmp_path / "spi.csv")}},
    }  # fmt: skip
    contract.write_json(tmp_path / "result.json", result)
    contract.read_result(tmp_path / "result.json")

    contract.write_series(tmp_path / "spi.csv", ["2000-01"], [2.0])

    with pytest.raises(contract.ContractError, match="SHA-256"):
        contract.read_result(tmp_path / "result.json")


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


def test_adapter_command_line_rejects_bad_usage() -> None:
    assert contract.run_adapter("x", lambda r, i: {}, ["run", "only-one-argument"]) == 2


def test_slice_runs_end_to_end_and_the_store_matches_the_library(tmp_path: Path) -> None:
    from climate_indices import spi
    from climate_indices.compute import Periodicity
    from climate_indices.vocabulary import Distribution

    manifest = harness.run_study(tmp_path / "slice", "0101", 3, ADAPTERS)

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


def test_rerun_gives_identical_output_hashes(tmp_path: Path) -> None:
    first = harness.run_study(tmp_path / "a", "0101", 3, ADAPTERS)
    second = harness.run_study(tmp_path / "b", "0101", 3, ADAPTERS)
    assert first["checksum_sha256"] == second["checksum_sha256"]
    assert [a["output_sha256"] for a in first["arrays"]] == [a["output_sha256"] for a in second["arrays"]]


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

    manifest = harness.run_study(tmp_path / "s", "0101", 3, {"fake": [sys.executable, str(fake)]})

    _validate(manifest)
    warnings = manifest["arrays"][0]["warnings"]
    assert warnings[0] == "own warning"
    assert any("'estimator'" in w and "'mle'" in w for w in warnings)
    assert any("'zero_handling'" in w and "<not reported>" in w for w in warnings)


def test_manifest_without_a_required_field_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    manifest = harness.run_study(tmp_path / "s", "0101", 3, ADAPTERS)
    for field in ("checksum_sha256", "harness", "arrays"):
        broken = {k: v for k, v in manifest.items() if k != field}
        with pytest.raises(Exception, match=field):
            _validate(broken)
    broken = json.loads(json.dumps(manifest))
    del broken["arrays"][0]["effective_method"]
    with pytest.raises(Exception, match="effective_method"):
        _validate(broken)


def test_committed_slice_manifest_is_valid_and_attributes_every_array() -> None:
    manifest = json.loads((HARNESS / "studies" / "refimpl-08-slice" / "manifest.json").read_text())
    _validate(manifest)
    assert {a["implementation"] for a in manifest["arrays"]} == {"climate_indices", "xclim"}
    for array in manifest["arrays"]:
        assert array["version"] and array["effective_method"] and array["output_sha256"]
    assert manifest["harness"]["git_dirty"] is False
    # every committed input hash is that of the committed fixture
    source = ROOT / manifest["inputs"]["precip"]["source_path"]
    assert contract.sha256_file(source) == manifest["inputs"]["precip"]["source_sha256"]

"""Keep Rust numerical contracts and generated accuracy evidence connected."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import native_parity_maxima
from tests import parity_registry

ROOT = Path(__file__).resolve().parents[1]


def test_every_public_function_has_a_numerical_contract() -> None:
    """A new Rust entry point must state its numerical behavior and evidence."""
    functions = 0
    helpers = {("pm_eto.rs", "at"), ("recurrence.rs", "recurrence_inputs"), ("recurrence.rs", "run")}
    for path in (ROOT / "crates/climate-core/src").rglob("*.rs"):
        source = path.read_text(encoding="utf-8")
        public = re.findall(r"^\s*pub fn (\w+)", source, re.MULTILINE)
        documented = re.findall(
            r"((?:^[ \t]*///[^\n]*\n)+)(?:[ \t]*#\[[^\n]+\]\n)*[ \t]*pub fn (\w+)",
            source,
            re.MULTILINE,
        )
        assert [name for _, name in documented] == public, path
        for docs, name in documented:
            functions += 1
            assert "# Numerical contract" in docs, (path, name)
            if (path.name, name) in helpers:
                assert "no independent" in docs, (path, name)
            else:
                assert "Measured accuracy" in docs, (path, name)
                assert any(f"`{entry}`" in docs for entry in parity_registry.ENTRIES_BY_NAME) or any(
                    f"`{entry}`" in docs for entry in native_parity_maxima.SPECIAL_NAMES
                ), (path, name)
    assert functions > 0


def test_accuracy_page_contains_every_measured_entry() -> None:
    """Both documentation systems consume one table covering the full registry."""
    report = (ROOT / "crates/climate-core/NUMERICS.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([a-z0-9_]+)` \|", report, re.MULTILINE)
    assert len(rows) == len(set(rows))
    assert set(rows) == set(parity_registry.ENTRIES_BY_NAME) | native_parity_maxima.SPECIAL_NAMES
    assert 'include_str!("../NUMERICS.md")' in (ROOT / "crates/climate-core/src/lib.rs").read_text(encoding="utf-8")
    assert "{include} ../crates/climate-core/NUMERICS.md" in (ROOT / "docs/rust-numerical-contract.md").read_text(
        encoding="utf-8"
    )


_SCALAR_PROBES = "\n".join(
    [
        "special_igam,1,0,0",
        "special_igamc,1,0,1",
        "special_ndtri,0.5,0,0",
        "special_norm_ppf,0.5,0,0",
        "special_ndtr,0,0,0.5",
    ]
)

# One probe per name, so a focused case does not have to carry the full corpus.
_CORPUS = {
    name: frozenset({(float(a).hex(), float(x).hex())})
    for name, a, x, _ in (line.split(",") for line in _SCALAR_PROBES.splitlines())
}


def _scalar_probe(name: str, a: float, x: float) -> str:
    reference = {
        "special_igam": lambda: native_parity_maxima.special.gammainc(a, x),
        "special_igamc": lambda: native_parity_maxima.special.gammaincc(a, x),
        "special_ndtri": lambda: native_parity_maxima.special.ndtri(a),
        "special_norm_ppf": lambda: native_parity_maxima.special.ndtri(a),
        "special_ndtr": lambda: native_parity_maxima.special.ndtr(a),
    }[name]()
    return f"{name},{a},{x},{reference}"


@pytest.fixture
def scalar_probes() -> str:
    return "\n".join(
        _scalar_probe(name, float.fromhex(a), float.fromhex(x))
        for name, coordinates in native_parity_maxima.PROBE_CORPUS.items()
        for a, x in sorted(coordinates)
    )


def test_special_errors_measures_instead_of_reporting_the_tolerance() -> None:
    output = _SCALAR_PROBES.replace("special_ndtr,0,0,0.5", "special_ndtr,0,0,0.500000000001")
    errors = native_parity_maxima.special_errors(output, corpus=_CORPUS)
    assert errors["special_igam"] == (0.0, 0.0)
    absolute, relative = errors["special_ndtr"]
    assert absolute == pytest.approx(1e-12, rel=1e-4, abs=0)
    assert relative == 2 * absolute


@pytest.mark.parametrize(
    "bad_probe",
    ["special_ndtr,0.25,0,0.6", "special_ndtr,NaN,0,0", "special_ndtr,inf,0,-inf"],
)
def test_special_errors_rejects_value_and_nonfinite_mismatches(bad_probe: str) -> None:
    _, a, x, _ = bad_probe.split(",")
    corpus = _CORPUS | {"special_ndtr": _CORPUS["special_ndtr"] | {(float(a).hex(), float(x).hex())}}
    with pytest.raises(AssertionError):
        native_parity_maxima.special_errors(_SCALAR_PROBES + "\n" + bad_probe, corpus=corpus)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("", "Missing scalar probes"),
        (_SCALAR_PROBES.replace("special_ndtri,0.5,0,0", "special_ndtri,NaN,0,NaN"), "Wrong scalar probe coordinates"),
        (_SCALAR_PROBES + "\nunknown,0,0,0", "Unknown scalar probe"),
        (_SCALAR_PROBES + "\nspecial_ndtr,0,0,0.5", "Duplicate scalar probe"),
        (_SCALAR_PROBES + "\nspecial_ndtr,0e0,0.0,0.5", "Duplicate scalar probe"),
        (_SCALAR_PROBES + "\nspecial_ndtr,0.25,0,0.5987", "Wrong scalar probe count"),
    ],
)
def test_special_errors_rejects_incomplete_evidence(output: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        native_parity_maxima.special_errors(output, corpus=_CORPUS)


def test_default_corpus_rejects_a_truncated_probe_set() -> None:
    """Five probes per function must not pass as the full documented corpus."""
    with pytest.raises(ValueError, match="Wrong scalar probe count: special_igam"):
        native_parity_maxima.special_errors(_SCALAR_PROBES)


def test_default_corpus_accepts_exact_coordinates(scalar_probes: str) -> None:
    assert all(errors == (0.0, 0.0) for errors in native_parity_maxima.special_errors(scalar_probes).values())


@pytest.mark.parametrize(
    ("name", "a", "x"),
    [
        ("special_igam", 1e6, 1e6),
        ("special_igamc", 0.0, 0.0),
        ("special_ndtri", 0.0, 0.0),
        ("special_norm_ppf", 1e-300, 0.0),
        ("special_ndtr", -40.0, 0.0),
        ("special_ndtr", np.inf, 0.0),
        ("special_ndtri", np.nan, 0.0),
    ],
)
def test_default_corpus_rejects_replaced_coordinates(scalar_probes: str, name: str, a: float, x: float) -> None:
    output = scalar_probes.replace(_scalar_probe(name, a, x), _scalar_probe(name, 0.2, x))
    assert output != scalar_probes
    with pytest.raises(ValueError, match=f"Wrong scalar probe coordinates: {name}"):
        native_parity_maxima.special_errors(output)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_compared_pairs_ignores_matching_nonfinite_pairs(value: float) -> None:
    assert parity_registry.compared_pairs(np.array([1.0, value]), np.array([1.0, value])) == 1
    assert parity_registry.compared_pairs(np.array([value]), np.array([value])) == 0


@pytest.mark.parametrize(("left", "right"), [(np.inf, -np.inf), (np.inf, 2.0), (np.nan, np.inf)])
def test_nonfinite_mismatches_remain_failures(left: float, right: float) -> None:
    rust, python = np.array([left]), np.array([right])
    assert parity_registry.compared_pairs(rust, python) == 1
    assert parity_registry.parity_errors(rust, python)[0] == np.inf
    with pytest.raises(AssertionError):
        parity_registry.assert_outcomes_parity(parity_registry.ENTRIES[0], rust, python)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_result_is_not_reported_as_zero_error(value: float) -> None:
    """A vacuous workflow row states that no comparison exists instead of printing zeros."""
    run = parity_registry.ParityRun(
        parity_registry.ENTRIES_BY_NAME["flood_api"],
        np.array([value]),
        np.array([value]),
        frozenset({"antecedent_precipitation_index"}),
        0.0,
        0.0,
    )
    report = native_parity_maxima.contract_report([run], {"special_ndtr": (0.0, 0.0)})
    assert "| `flood_api` | `antecedent_precipitation_index` | no finite comparison | no finite comparison |" in report
    assert "have no finite comparison on this corpus: `flood_api`" in report


def test_report_renders_measured_maxima() -> None:
    run = parity_registry.ParityRun(
        parity_registry.ENTRIES[0], np.array([1.0]), np.array([1.0]), frozenset({"gamma_parameters"}), 2e-14, 4e-14
    )
    report = native_parity_maxima.contract_report([run], {"special_ndtr": (1e-16, 2e-16)})
    assert "| `spi_gamma` | `gamma_parameters` | 2.000000e-14 | 4.000000e-14 |" in report
    assert "| `special_ndtr` | 1.000000e-16 | 2.000000e-16 |" in report
    assert "not an isolated error bound" in report
    assert "Calibration Period includes 1983 through the final record year, 2015" in report
    assert "Only `flood_edi` and `flood_flood_index` prepare an all-leap calendar" in report
    assert "`flood_pe` and `flood_api` retain the 365-day inputs" in report


@pytest.mark.parametrize("cargo", [None, "/trusted/bin/cargo"])
def test_contract_generation_resolves_cargo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cargo: str | None) -> None:
    run = parity_registry.ParityRun(
        parity_registry.ENTRIES[0], np.array([1.0]), np.array([1.0]), frozenset({"gamma_parameters"}), 0.0, 0.0
    )
    monkeypatch.setattr(native_parity_maxima.sys, "argv", ["native_parity_maxima.py", "--write-contract"])
    monkeypatch.setattr(native_parity_maxima.parity_registry, "measure", lambda _: [run])
    which = Mock(return_value=cargo)
    monkeypatch.setattr(native_parity_maxima.shutil, "which", which)
    execute = Mock(return_value=Mock(stdout=_SCALAR_PROBES))
    monkeypatch.setattr(native_parity_maxima.subprocess, "run", execute)
    monkeypatch.setattr(native_parity_maxima, "ROOT", tmp_path)
    monkeypatch.setattr(native_parity_maxima, "special_errors", lambda _: {})
    (tmp_path / "crates/climate-core").mkdir(parents=True)
    if cargo is None:
        with pytest.raises(FileNotFoundError, match="cargo is required"):
            native_parity_maxima.main()
        execute.assert_not_called()
    else:
        assert native_parity_maxima.main() == 0
        execute.assert_called_once_with(
            [cargo, "run", "-p", "climate-core", "--example", "special_accuracy", "--quiet"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        assert (tmp_path / "crates/climate-core/NUMERICS.md").is_file()
    which.assert_called_once_with("cargo")

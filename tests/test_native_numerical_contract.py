"""Keep Rust numerical contracts and generated accuracy evidence connected."""

from __future__ import annotations

import re
from pathlib import Path

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
_CORPUS: dict[str, tuple[int, int]] = dict.fromkeys(native_parity_maxima.SPECIAL_NAMES, (1, 1))

# Two probes for `special_ndtr`, one elsewhere: a case that appends one more ndtr line.
_CORPUS_NDTR_TWO: dict[str, tuple[int, int]] = {
    name: (2 if name == "special_ndtr" else 1, 1) for name in native_parity_maxima.SPECIAL_NAMES
}


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
    with pytest.raises(AssertionError):
        native_parity_maxima.special_errors(_SCALAR_PROBES + "\n" + bad_probe, corpus=_CORPUS_NDTR_TWO)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("", "Missing scalar probes"),
        (_SCALAR_PROBES.replace("special_ndtri,0.5,0,0", "special_ndtri,NaN,0,NaN"), "Too few finite scalar probes"),
        (_SCALAR_PROBES + "\nunknown,0,0,0", "Unknown scalar probe"),
        (_SCALAR_PROBES + "\nspecial_ndtr,0,0,0.5", "Duplicate scalar probe"),
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


def test_compared_pairs_ignores_double_nan_pairs() -> None:
    assert parity_registry.compared_pairs(np.array([1.0, np.nan, np.inf]), np.array([1.0, np.nan, 2.0])) == 2
    assert parity_registry.compared_pairs(np.array([np.nan]), np.array([np.nan])) == 0


def test_all_nan_result_is_not_reported_as_zero_error() -> None:
    """A vacuous workflow row states that no comparison exists instead of printing zeros."""
    run = parity_registry.ParityRun(
        parity_registry.ENTRIES_BY_NAME["flood_api"],
        np.array([np.nan]),
        np.array([np.nan]),
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

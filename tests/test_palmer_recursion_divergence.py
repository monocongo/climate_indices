"""Pin both Palmer recursions on reduced Z-Index sequences that make them disagree (REFIMPL-29, #1631).

Each case is a reduced reproducing sequence for one divergence class, found by
``scripts/find_palmer_minimal_sequences.py`` and described in
``docs/research/reference-papers/palmer-recursion-divergence.md``.  The generator rounds its
sequences after greedy deletion reaches a fixed point, so a case can keep a month that is
removable without losing the class: the sequences are reduced, not proven minimal.  The class name is the set
of code-level differences (E establishment threshold, T termination at |X3| <= 0.5, A abatement
trigger, C candidate carried at establishment, F end-of-record flush) whose omission from
full harmonization leaves the episode disagreeing. This necessity applies only within full
harmonization, not to every subset that removes the disagreement.

The tests characterize behavior: they pin what ``_palmer_pdi.calculate`` (NCEI ``pdi.f``
lineage) and ``_palmer_wells.calculate`` (Wells lineage) return on the same Z-Index series at
Palmer's national duration factors.  They do not say which recursion is right, and they must be
updated, not relaxed, if either recursion is deliberately changed.  #1665 (ADR-0030) did that
for class F: the end-of-record flush now rewrites only the months still pending, so the library
no longer disagrees with Wells on the F case, "E+F" reclassifies as "E", and the "E+T+C" case
loses the five months (15-19) its stale flush had rewritten.  ``_CASES`` keeps
the sequences and the values recorded at the pinned commit (the committed JSON); the two
overrides below say what the current library returns.  Sequences are left-padded
with zeros to whole years, which both recursions treat as no-ops.
"""

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from climate_indices.drought import _palmer_pdi, _palmer_wells
from climate_indices.drought._palmer_duration import DurationFactors
from climate_indices.drought._palmer_pdi import PdiDurationFactors

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ROOT / "docs" / "research" / "reference-papers" / "palmer-recursion-divergence-sequences.json"
needs_checkout_import = pytest.mark.skipif(
    not Path(_palmer_pdi.__file__).resolve().is_relative_to(ROOT / "src"),
    reason="the source pin only accepts a package imported from this checkout",
)


@lru_cache(maxsize=1)
def _load_finder() -> ModuleType:
    """Load the sibling generator script without adding ``scripts/`` to ``sys.path``."""
    spec = importlib.util.spec_from_file_location(
        "find_palmer_minimal_sequences", ROOT / "scripts" / "find_palmer_minimal_sequences.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load scripts/find_palmer_minimal_sequences.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# class -> (active Z sequence, {month index in the padded series: (pdi.f PDSI, Wells PDSI)})
_CASES = {
    "A": (
        [-1.0, -2.2, -0.15, 6.0, 3.0],
        {9: (-0.976003, -0.05)},
    ),
    "A+C": (
        [
            -1.7,
            -2.7,
            -3.0,
            -3.0,
            -4.0,
            -3.0,
            -0.6,
            2.0,
            0.0,
            3.1,
            1.0,
            1.0,
            -2.5,
            -0.2,
            2.0,
            -3.0,
            -1.0,
            -0.15,
            6.0,
            -1.0,
        ],
        {19: (-1.0, -1.057085828), 20: (-1.230333333, -1.281539321), 21: (-1.153609, -0.05)},
    ),
    "C": (
        [-2.0, -4.0, -2.0, -3.0, 1.4, 2.4, 1.0, -0.2, -1.6, -2.0, -2.0, 2.0, 2.2, -3.0, 1.0],
        {22: (-1.0, -1.062988044), 23: (-0.563666667, -0.620166942)},
    ),
    "E": (
        [-1.0, -1.0, -0.0, 3.0, 1.0],
        {9: (-0.567203, 0.0)},
    ),
    "E+A": (
        [2.2459814106903577, -0.3607725973830373],
        {11: (0.0, -0.120257532)},
    ),
    "E+A+C": (
        [
            4.0,
            3.0,
            -1.760804724971848,
            -0.5618104471028332,
            0.43,
            -2.58,
            -2.5934196789941963,
            -0.34,
            -0.2912800684270723,
            -2.6709524502381403,
            -2.9,
            4.02790594538576,
            -3.45,
            -0.48,
            -0.9809322896576268,
            -0.6024294186238482,
            0.6455441156158037,
            -2.8036402656626023,
            -0.28,
            1.71,
            -1.6833150012862512,
            1.44,
            1.75,
            -0.2665657954567833,
            -1.38,
            1.25,
            -2.0,
            -1.0,
            -2.4,
            2.6,
        ],
        {
            25: (-1.70744784, 0.57),
            26: (-2.092685713, -0.561105),
            27: (-1.397139085, 0.48),
            28: (-0.669900426, 1.013893333),
            29: (-0.689755947, 0.820607055),
            30: (-1.078711084, 0.276084528),
            31: (-0.550937176, 0.664314488),
            32: (-1.160857314, -0.727166697),
            33: (-1.374622344, -0.98560186),
            34: (-2.033036242, -1.684084869),
            35: (-0.956966843, -0.643957461),
        },
    ),
    "E+C": (
        [-1.0, -1.0, -3.0, -2.0, -1.0, -1.0, 2.5, 1.5, -1.6, -1.0, -2.0, 1.0, -1.0, 2.9, -0.4, 0.6, -2.3, 0.0],
        {20: (0.733766667, -0.196840161), 21: (0.8581887, 0.2), 22: (0.0, -0.766666667), 23: (0.0, -0.6877)},
    ),
    "E+F": (
        [-2.0, -0.5, 0.0, 1.0],
        {10: (-0.685906, 0.0), 11: (0.0, 0.333333333)},
    ),
    "E+T": (
        [-1.65, 2.0, 1.0, 3.0],
        {9: (0.666666667, 0.173316667), 10: (0.931333333, 0.488798383), 11: (1.835406, 1.43845215)},
    ),
    "E+T+A": (
        [-0.6, -1.0, 1.7, 1.0, 2.3, -1.44, 1.74, -0.5, -3.0, 1.0],
        {
            4: (0.566666667, 0.106744867),
            5: (0.841633333, 0.429083479),
            6: (1.521611767, 1.151554547),
            7: (0.884885755, -0.48),
            8: (1.373742522, 0.58),
        },
    ),
    "E+T+C": (
        [
            -1.0,
            -2.0,
            -1.4,
            -1.1,
            -1.3,
            1.3,
            -2.0,
            0.0,
            -2.0,
            -0.0,
            3.0,
            -0.52,
            -0.8,
            0.8,
            0.0762726169170272,
            -1.3,
            1.4,
            1.0,
            2.0,
        ],
        {
            15: (-0.672041853, 1.0),
            16: (-0.776154875, 0.723666667),
            17: (-0.96287759, 0.382462333),
            18: (-0.597034531, 0.60973538),
            19: (0.0, 0.572356841),
            20: (0.0, -0.500643138),
            21: (0.538490132, 0.017589771),
            22: (0.816358982, 0.349111358),
            23: (1.398940674, 0.979819555),
        },
    ),
    "F": (
        [-2.0, -1.0, -1.3, -2.0, 3.0, 3.0],
        {10: (-0.618839086, 1.0)},
    ),
}


# What the library returns after #1665, where it differs from the pinned-commit record in _CASES.
_LIVE_EXPECTED: dict[str, dict[int, tuple[float, float]]] = {
    "F": {},
    # months 15-19 were stale-flush victims at the pinned commit; only 20-23 still differ
    "E+T+C": {
        20: (0.080070753, -0.500643138),
        21: (0.538490132, 0.017589771),
        22: (0.816358982, 0.349111358),
        23: (1.398940674, 0.979819555),
    },
}
_CLASS_AFTER_1665: dict[str, str | None] = {"F": None, "E+F": "E"}


def _run(active: list[float]) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate([np.zeros(-len(active) % 12), np.asarray(active, dtype=float)])
    wells_factors = DurationFactors.from_defaults()
    pdi_factors = PdiDurationFactors.from_fitted(
        wells_factors.wetm, wells_factors.wetb, wells_factors.drym, wells_factors.dryb
    )
    pdi = _palmer_pdi.calculate(padded.reshape(-1, 12, 1), pdi_factors).pdsi.reshape(-1)
    wells = _palmer_wells.calculate(padded, factors=wells_factors).pdsi
    return pdi, wells


@pytest.mark.parametrize("label", sorted(_CASES))
def test_reduced_sequence_pins_both_recursions(label: str) -> None:
    active, recorded = _CASES[label]
    expected = _LIVE_EXPECTED.get(label, recorded)
    pdi, wells = _run(active)

    differing = np.flatnonzero(np.abs(pdi - wells) > 1e-9)
    assert differing.tolist() == sorted(expected)
    for index, (pdi_value, wells_value) in expected.items():
        assert pdi[index] == pytest.approx(pdi_value, abs=1e-8)
        assert wells[index] == pytest.approx(wells_value, abs=1e-8)


@pytest.mark.parametrize("label", sorted(_CASES))
def test_sequence_still_reproduces_its_class(label: str) -> None:
    """The study's own classifier must still return the class the case is named for."""
    active, _ = _CASES[label]
    assert _load_finder().label_of(active) == _CLASS_AFTER_1665.get(label, label)


def test_flush_fix_resolves_the_f_case_to_the_wells_value() -> None:
    """#1665: the month the stale flush rewrote keeps its resolved value, which Wells also gives."""
    active, recorded = _CASES["F"]
    pdi, wells = _run(active)
    ((index, (stale_value, _)),) = recorded.items()

    assert pdi[index] == pytest.approx(1.0, abs=1e-12)
    assert pdi[index] == pytest.approx(wells[index], abs=1e-12)
    assert stale_value == pytest.approx(-0.618839086, abs=1e-8)  # the pinned-commit (stale) value


def test_generator_rejects_source_drift_before_reading_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    finder = _load_finder()
    output = tmp_path / "sequences.json"
    output.write_text("unchanged")
    monkeypatch.setattr(finder, "RESULTS", tmp_path / "missing-results.json")
    monkeypatch.setattr(finder, "OUT", output)

    def reject_drift() -> None:
        raise SystemExit("source drift")

    monkeypatch.setattr(finder.c, "verify_pinned_sources", reject_drift)
    with pytest.raises(SystemExit, match="source drift"):
        finder.main()
    assert output.read_text() == "unchanged"


@pytest.mark.parametrize("pin", ["different-commit", None])
def test_generator_rejects_results_without_the_study_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pin: str | None
) -> None:
    finder = _load_finder()
    source = tmp_path / "results.json"
    source.write_text(json.dumps({"pinned_commit": pin} if pin is not None else {}))
    output = tmp_path / "sequences.json"
    output.write_text("unchanged")
    monkeypatch.setattr(finder, "RESULTS", source)
    monkeypatch.setattr(finder, "OUT", output)
    monkeypatch.setattr(finder.c, "verify_pinned_sources", lambda: None)

    with pytest.raises(SystemExit, match="study results must use the pinned commit"):
        finder.main()
    assert output.read_text() == "unchanged"


def test_generator_accepts_the_study_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    finder = _load_finder()
    source = tmp_path / "results.json"
    source.write_text(json.dumps({"pinned_commit": finder.c.PINNED_COMMIT, "example_episodes_by_label": {}}))
    output = tmp_path / "sequences.json"
    monkeypatch.setattr(finder, "RESULTS", source)
    monkeypatch.setattr(finder, "OUT", output)
    verify = Mock()
    monkeypatch.setattr(finder.c, "verify_pinned_sources", verify)

    finder.main()
    verify.assert_called_once_with()
    assert json.loads(output.read_text()) == {}


@needs_checkout_import
def test_source_pin_includes_common_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    study = _load_finder().c
    common = "src/climate_indices/drought/_palmer_common.py"

    def compare(args: list[str], **kwargs: object) -> Mock:
        return Mock(returncode=int(args[-1] == common))

    monkeypatch.setattr(study.subprocess, "run", compare)
    with pytest.raises(SystemExit, match="_palmer_common.py differs from the pinned commit"):
        study.verify_pinned_sources()


@pytest.mark.parametrize("name", ["_palmer_pdi", "_palmer_wells", "_palmer_duration", "_palmer_common"])
def test_source_pin_rejects_imports_outside_the_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    study = _load_finder().c
    module = importlib.import_module(f"climate_indices.drought.{name}")
    monkeypatch.setattr(module, "__file__", str(tmp_path / f"{name}.py"))
    monkeypatch.setattr(study.subprocess, "run", Mock(return_value=Mock(returncode=0)))
    with pytest.raises(SystemExit, match="does not resolve to the checked source"):
        study.verify_pinned_sources()


@needs_checkout_import
def test_source_pin_accepts_checkout_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    study = _load_finder().c
    compare = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(study.subprocess, "run", compare)
    study.verify_pinned_sources()
    assert compare.call_count == len(study.STUDIED_SOURCES)


@pytest.mark.parametrize("peaks, expected_index", [([1.0000006, 1.0000008], 2), ([1.0000004, 1.0000002], 0)])
def test_class_maximum_uses_unrounded_values(peaks: list[float], expected_index: int) -> None:
    study = _load_finder().c
    summary = study._descriptive(
        ("test",),
        {"test": np.array([True, False, True])},
        {"test": np.array([peaks[0], 0.0, peaks[1]])},
        {"test": np.zeros(3)},
        {"test": [(0, 1, "E"), (2, 3, "E")]},
        3,
    )
    assert summary["largest"]["E"]["month_index"] == expected_index
    assert summary["largest"]["E"]["abs_diff"] == round(max(peaks), 6)


def test_h3_counts_the_union_of_removed_episode_months(monkeypatch: pytest.MonkeyPatch) -> None:
    study = _load_finder().c
    base = np.array([True] * 6 + [False] + [True] * 6 + [False] + [True] * 6)
    fp, fp2 = base.copy(), base.copy()
    fp[:6] = False
    fp[14:] = False
    fp2[7:13] = False
    fp2[14:] = False
    masks = {"fp": {"test": fp}, "fp2": {"test": fp2}}
    monkeypatch.setattr(study, "disagree_masks", masks.__getitem__)

    summary, removed_months = study._floating_point_summary(("test",), {"test": base})
    assert summary["fp"]["base_months_removed"] == summary["fp2"]["base_months_removed"] == 12
    assert removed_months == 18
    head = {
        "closer_to_nclimdiv": {"pdi": 10000, "wells": 0, "tie": 0},
        "disagree_counts": {str(study.DISAGREE): 10000},
        "disagree_rate": 0.063,
        "max_abs_diff": 8.0,
    }
    h3 = study._hypotheses(head, removed_months, {"episodes_by_label": {}}, 0, 0)["H3"]
    assert h3 == {"floating_point_months_share": 0.0018, "pass": False}


def test_h5_measures_the_lookback_not_the_current_spell() -> None:
    study = _load_finder().c
    counts = study._h5_counts(
        ("test",),
        {"test": np.array([False, False, False, True])},
        {"test": (np.array([2.0, 2.0, 2.0, -2.0]), np.array([2.0, 2.0, 2.0, 0.0]))},
        4,
    )
    assert counts == (1, 1)


@pytest.mark.parametrize("link_parent", [True, False])
def test_results_reject_symlink_escapes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, link_parent: bool) -> None:
    study = _load_finder().c
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "results.json"
    target.write_text("unchanged")
    if link_parent:
        (root / "output").symlink_to(outside, target_is_directory=True)
        out = root / "output" / "results.json"
    else:
        out = root / "results.json"
        out.symlink_to(target)
    monkeypatch.setattr(study, "ROOT", root)

    with pytest.raises(SystemExit, match="outside the checkout|through a symlink"):
        study.write_results(out, {})
    assert target.read_text() == "unchanged"


def test_results_accept_a_parent_symlink_inside_the_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    study = _load_finder().c
    (tmp_path / "output").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "output", target_is_directory=True)
    monkeypatch.setattr(study, "ROOT", tmp_path)

    study.write_results(tmp_path / "link" / "results.json", {"ok": True})
    assert json.loads((tmp_path / "output" / "results.json").read_text()) == {"ok": True}


def test_sequences_match_the_committed_json() -> None:
    """The hand-copied literals must equal the generator's committed output."""
    document = json.loads(SEQUENCES.read_text())
    for label, (active, expected) in _CASES.items():
        record = document[label]
        assert active == pytest.approx(record["z"], abs=1e-12)
        assert sorted(expected)[0] == record["first_differing_index"]
        for index, (pdi_value, wells_value) in expected.items():
            assert pdi_value == pytest.approx(record["pdi_pdsi"][index], abs=1e-6)
            assert wells_value == pytest.approx(record["wells_pdsi"][index], abs=1e-6)

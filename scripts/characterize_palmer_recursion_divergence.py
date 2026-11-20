"""Characterize where the pdi.f and Wells Palmer recursions disagree (REFIMPL-29, #1631).

Read-only with respect to the library: both recursions are called as shipped. The only
substitution is the floating-point isolation runs (P-fp, P-fp2), which patch the
weighting-fraction expression in memory and restore it afterwards.

Hypotheses, thresholds and class definitions are pre-registered in
``docs/research/reference-papers/palmer-recursion-divergence.md``.

Run: ``uv run python scripts/characterize_palmer_recursion_divergence.py``
(about 10 minutes; writes ``palmer-recursion-divergence-results.json``).
``... probe`` reruns the exploratory first-state-divergence probe quoted in deviation D1.
Any other argument is rejected. The full run refuses to write when the studied modules
differ from the pinned commit, and exits 1 when a pre-registered hypothesis fails.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 # Only Git runs, with fixed arguments and no shell.
import sys
import types
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import climate_indices
from climate_indices.drought import _palmer_pdi, _palmer_wells
from climate_indices.drought._palmer_duration import DurationFactors
from climate_indices.drought._palmer_pdi import PdiDurationFactors

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixture" / "palmer"
NCLIMDIV = ROOT / "tests" / "fixture" / "nclimdiv"
START_YEAR = 1895
DISAGREE = 1e-9
THRESHOLDS = (1e-9, 0.01, 0.5, 1.0)
PINNED_COMMIT = "46481c48c79e878eb50ce3398ba062fd3a1ff38c"
STUDIED_SOURCES = (
    "src/climate_indices/drought/_palmer_pdi.py",
    "src/climate_indices/drought/_palmer_wells.py",
    "src/climate_indices/drought/_palmer_duration.py",
    "src/climate_indices/drought/_palmer_common.py",
)


@lru_cache(maxsize=1)
def divisions() -> tuple[str, ...]:
    path = NCLIMDIV / "divisions.json"
    try:
        return tuple(json.loads(path.read_text()))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read the division list at {path}") from error


def load_z(division: str) -> np.ndarray:
    """Load a division's Z series, rejecting a name that resolves outside the fixture tree."""
    if division not in divisions():
        raise ValueError(f"unknown division {division!r}")
    path = (FIXTURE / division / "zindex.npy").resolve()
    if not path.is_relative_to(FIXTURE.resolve()):
        raise ValueError(f"division {division!r} resolves outside the fixture tree")
    array: np.ndarray = np.load(path)
    return array


def verify_pinned_sources() -> None:
    """Refuse to record results when the studied modules differ from the pinned commit."""
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git is unavailable; cannot verify the studied sources against the pin")
    for path in STUDIED_SOURCES:
        module = sys.modules[f"climate_indices.drought.{Path(path).stem}"]
        origin = getattr(module, "__file__", None)
        if origin is None or Path(origin).resolve() != (ROOT / path).resolve():
            raise SystemExit(f"{module.__name__} does not resolve to the checked source {ROOT / path}")
        proc = subprocess.run(  # nosec B603 # The resolved Git path and fixed arguments are trusted local inputs.
            [git, "diff", "--quiet", PINNED_COMMIT, "--", path], cwd=ROOT
        )
        if proc.returncode == 1:
            raise SystemExit(f"{path} differs from the pinned commit {PINNED_COMMIT}; update the pin first")
        if proc.returncode != 0:
            raise SystemExit(f"cannot compare {path} with {PINNED_COMMIT} (git exit {proc.returncode})")


def default_factors() -> tuple[PdiDurationFactors, DurationFactors]:
    """Palmer's national constants, built the way ``pdsi()`` builds them."""
    wells = DurationFactors.from_defaults()
    pdi = PdiDurationFactors.from_fitted(wells.wetm, wells.wetb, wells.drym, wells.dryb)
    return pdi, wells


def fitted_factors(division: str) -> tuple[PdiDurationFactors, DurationFactors]:
    wetm, wetb, drym, dryb = (float(v) for v in np.load(FIXTURE / division / "scdurfact.npy"))
    return (
        PdiDurationFactors.from_fitted(wetm, wetb, drym, dryb),
        DurationFactors.from_fitted(wetm, wetb, drym, dryb),
    )


def run_pdi(z: np.ndarray, factors: PdiDurationFactors) -> _palmer_pdi.PdiResult:
    return _palmer_pdi.calculate(z.reshape(-1, 12, 1), factors)


def run_both(
    z: np.ndarray, pdi_factors: PdiDurationFactors, wells_factors: DurationFactors
) -> tuple[np.ndarray, _palmer_wells.WellsResult, _palmer_pdi.PdiResult]:
    pdi = run_pdi(z, pdi_factors)
    wells = _palmer_wells.calculate(z, factors=wells_factors)
    return pdi.pdsi.reshape(-1), wells, pdi


def episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` runs of True in a 1-D boolean mask."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def headline(config: str) -> dict[str, Any]:
    nclim = np.load(NCLIMDIV / "pdsi.npy")
    divs = divisions()
    if nclim.shape[0] != len(divs):
        raise ValueError(f"nclimdiv pdsi has {nclim.shape[0]} rows but there are {len(divs)} divisions")
    n_months = 0
    counts = dict.fromkeys(THRESHOLDS, 0)
    max_abs = 0.0
    closer_pdi = closer_wells = ties = 0
    per_division: dict[str, float] = {}
    for row, division in enumerate(divs):
        z = load_z(division)
        pdi_f, wells_f = default_factors() if config == "P" else fitted_factors(division)
        pdi_pdsi, wells, _ = run_both(z, pdi_f, wells_f)
        diff = np.abs(pdi_pdsi - wells.pdsi)
        if nclim[row].size != diff.size:
            raise ValueError(f"nclimdiv row {row} has {nclim[row].size} months but {division} has {diff.size}")
        n_months += diff.size
        for t in THRESHOLDS:
            counts[t] += int(np.sum(diff > t))
        max_abs = max(max_abs, float(diff.max()))
        per_division[division] = float(np.mean(diff > DISAGREE))
        bad = diff > DISAGREE
        e_pdi = np.abs(pdi_pdsi[bad] - nclim[row][bad])
        e_wells = np.abs(wells.pdsi[bad] - nclim[row][bad])
        closer_pdi += int(np.sum(e_pdi < e_wells))
        closer_wells += int(np.sum(e_wells < e_pdi))
        ties += int(np.sum(e_pdi == e_wells))
    return {
        "config": config,
        "months": n_months,
        "disagree_counts": {str(t): c for t, c in counts.items()},
        "disagree_rate": counts[DISAGREE] / n_months,
        "max_abs_diff": max_abs,
        "closer_to_nclimdiv": {"pdi": closer_pdi, "wells": closer_wells, "tie": ties},
        "per_division_rate": per_division,
    }


STATE_TOL = 1e-9
_PDI_STATEMENTS = ("_statement_170", "_statement_180", "_statement_190", "_statement_200", "_statement_210")


def trace_pdi(z: np.ndarray, factors: PdiDurationFactors) -> tuple[_palmer_pdi.PdiResult, list[dict[str, Any]]]:
    """Run pdi.f and record, per month, the pre-state, the post-state and the statements entered.

    The statement functions and ``_advance_month`` are wrapped for the duration of the
    call and restored afterwards; the recursion itself is unchanged.
    """
    originals = {name: getattr(_palmer_pdi, name) for name in (*_PDI_STATEMENTS, "_advance_month")}
    entered: list[str] = []
    rows: list[dict[str, Any]] = []

    def make(name: str) -> Callable[..., Any]:
        def wrapper(*args: Any) -> Any:
            active = args[-1]
            if bool(np.any(active)):
                entered.append(name.removeprefix("_statement_"))
            return originals[name](*args)

        return wrapper

    def advance(fac: Any, state: Any, year: int, month: int) -> None:
        entered.clear()
        pre = (float(state.pro[0]), float(state.x3[0]))
        originals["_advance_month"](fac, state, year, month)
        rows.append(
            {
                "pre_pro": pre[0],
                "pre_x3": pre[1],
                "post": (
                    float(state.px1[year, month, 0]),
                    float(state.px2[year, month, 0]),
                    float(state.px3[year, month, 0]),
                    float(state.ppr[year, month, 0]),
                ),
                "branch": tuple(entered),
            }
        )

    try:
        for name in _PDI_STATEMENTS:
            setattr(_palmer_pdi, name, make(name))
        _palmer_pdi._advance_month = advance  # type: ignore[assignment]
        result = run_pdi(z, factors)
    finally:
        for name, fn in originals.items():
            setattr(_palmer_pdi, name, fn)
    return result, rows


def _spell_is_open(x3: float) -> bool:
    """True when ``x3`` marks an established spell.

    ``0.0`` is the library's exact-zero sentinel, not a near-equality: both recursions
    clear a spell by assigning ``0.0`` exactly. The rationale is in
    ``docs/algorithm_refs/palmer.md`` ("Exact-zero comparisons"), so the comparison
    stays exact and Sonar's S1244 rule is suppressed here.
    """
    return x3 != 0.0  # NOSONAR


def trace_wells(z: np.ndarray, factors: DurationFactors) -> tuple[_palmer_wells.WellsResult, list[tuple[str, ...]]]:
    """Run Wells and record, per month, which decision arm ran."""
    orig_continue = _palmer_wells._continue_spell
    orig_abate = _palmer_wells._abatement_transition
    orig_establish = _palmer_wells._establish_spell
    arms: list[str] = []
    log: list[tuple[str, ...]] = []

    def cont(state: Any, z_value: float, fac: Any, transition: Any) -> Any:
        arms.clear()
        if not _spell_is_open(state.x3):
            arms.append("no-spell")
        out = orig_continue(state, z_value, fac, transition)
        if _spell_is_open(state.x3) and not arms:
            arms.append("intensify")
        return out

    def abate(*args: Any, **kwargs: Any) -> Any:
        out = orig_abate(*args, **kwargs)
        if out.selection == 3:
            arm = "abate-fizzle"
        elif out.spell_terminated:
            arm = "abate-end"
        else:
            arm = "abate-cont"
        arms.append(arm)
        return out

    def establish(transition: Any) -> Any:
        out = orig_establish(transition)
        if not _spell_is_open(transition.x3):
            arms.append(f"establish-sel{out.selection}")
        log.append(tuple(arms))
        return out

    try:
        _palmer_wells._continue_spell = cont  # type: ignore[assignment]
        _palmer_wells._abatement_transition = abate
        _palmer_wells._establish_spell = establish
        result = _palmer_wells.calculate(z, factors=factors)
    finally:
        _palmer_wells._continue_spell = orig_continue
        _palmer_wells._abatement_transition = orig_abate
        _palmer_wells._establish_spell = orig_establish
    return result, log


def state_mismatch(pdi_rows: list[dict[str, Any]], wells: _palmer_wells.WellsResult) -> np.ndarray:
    """True where the two machines' post-month (x1, x2, x3, probability) differ."""
    p = np.array([r["post"] for r in pdi_rows])
    w = np.column_stack([wells.x1, wells.x2, wells.x3, wells.probability])
    differs: np.ndarray = np.any(np.abs(p - w) > STATE_TOL, axis=1)
    return differs


def signatures(config: str = "P") -> dict[str, Any]:
    out: dict[str, int] = {}
    n_state_eps = n_out_eps = no_state = 0
    for division in divisions():
        z = load_z(division)
        pdi_f, wells_f = default_factors() if config == "P" else fitted_factors(division)
        pres, prow = trace_pdi(z, pdi_f)
        wres, wlog = trace_wells(z, wells_f)
        smis = state_mismatch(prow, wres)
        omis = np.abs(pres.pdsi.reshape(-1) - wres.pdsi) > DISAGREE
        n_out_eps += len(episodes(omis))
        for s, _ in episodes(smis):
            n_state_eps += 1
            key = f"pre_spell={_spell_is_open(prow[s]['pre_x3'])} pdi={'/'.join(prow[s]['branch'])} wells={'/'.join(wlog[s])}"
            out[key] = out.get(key, 0) + 1
        for s, e in episodes(omis):
            if not smis[max(0, s - 1) : e].any():
                no_state += 1
    return {
        "n_out_episodes": n_out_eps,
        "n_state_episodes": n_state_eps,
        "out_without_state": no_state,
        "signatures": out,
    }


# Each mechanism: (old text, new text, expected occurrences) edits that make pdi.f use the Wells rule.
MECHANISMS: dict[str, list[tuple[str, str, int]]] = {
    "E": [
        ("(state.px1[y, m] >= 1)", "(state.px1[y, m] >= 0.5)", 1),
        ("(state.px2[y, m] <= -1)", "(state.px2[y, m] <= -0.5)", 1),
    ],
    "T": [
        ("(state.x3 >= -0.5) & (state.x3 <= 0.5)", "(state.x3 == 0)", 1),
        ("wet = established & (state.x3 > 0.5)", "wet = established & (state.x3 > 0)", 1),
        ("dry = established & (state.x3 < -0.5)", "dry = established & (state.x3 < 0)", 1),
    ],
    "A": [
        ("ud_new = state.z[y, m] - 0.15", "ud_new = state.z[y, m] - factors.wetm * 0.5", 1),
        ("uw_new = state.z[y, m] + 0.15", "uw_new = state.z[y, m] + factors.drym * 0.5", 1),
        ("wet_intensify = wet & (z >= 0.15)", "wet_intensify = wet & (z >= factors.wetm * 0.5)", 1),
        ("dry_intensify = dry & (z <= -0.15)", "dry_intensify = dry & (z <= -factors.drym * 0.5)", 1),
    ],
    "C": [
        (
            "state.px2[y, m] = np.where(active & ~branch1, px2_new, state.px2[y, m])",
            "state.px2[y, m] = np.where(active, px2_new, state.px2[y, m])",
            1,
        ),
    ],
    "F": [("step = k8 < state.k8max", "step = k8 < state.k8", 1)],
    "Tol": [
        ("_py_min(state.v, 0.0)", "_py_min(state.v + 1e-5, 0.0)", 1),
        ("_py_max(state.v, 0.0)", "_py_max(state.v - 1e-5, 0.0)", 1),
        ("over = ppr_new >= 100", "over = ppr_new >= 100 - 1e-5", 1),
        ("state.pro == 100, state.ze", "state.pro >= 100 - 1e-5, state.ze", 1),
    ],
}
MECH_NAMES = tuple(MECHANISMS)


@lru_cache(maxsize=1)
def _pdi_source() -> str:
    """Read the ``_palmer_pdi`` source once so a mid-run edit cannot mix two versions.

    Refuses a symlink and any file outside the module's own package directory.
    """
    spec = _palmer_pdi.__spec__
    origin = spec.origin if spec is not None else None
    if origin is None or not Path(origin).is_file():
        raise RuntimeError(f"cannot read the _palmer_pdi source at {origin!r}")
    path = Path(origin)
    if path.is_symlink():
        raise RuntimeError(f"refusing to read the _palmer_pdi source through the symlink {path}")
    package = Path(climate_indices.__file__).resolve().parent
    if not path.resolve().is_relative_to(package):
        raise RuntimeError(f"the _palmer_pdi source at {path} is outside the package directory {package}")
    return path.read_text()


def build_pdi_variant(mechs: tuple[str, ...]) -> types.ModuleType:
    """An in-memory copy of ``_palmer_pdi`` with the named mechanisms switched to the Wells rule."""
    source = _pdi_source()
    for mech in mechs:
        for old, new, expected in MECHANISMS[mech]:
            if source.count(old) != expected:
                raise RuntimeError(f"mechanism {mech}: expected {expected} of {old!r}, found {source.count(old)}")
            source = source.replace(old, new)
    name = "_palmer_pdi_variant_" + "_".join(mechs or ("none",))
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    sys.modules[name] = module
    try:
        # pi-lens-ignore: no-compile-call -- the source is the installed module's own text with literal edits
        exec(compile(source, module.__file__, "exec"), module.__dict__)  # nosec B102 - the library's own source, edited by literals
    finally:
        sys.modules.pop(name, None)
    return module


def variant_masks(mechs: tuple[str, ...], wells_cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    module = build_pdi_variant(mechs)
    masks = {}
    for division in divisions():
        z = load_z(division)
        pdi_f, _ = default_factors()
        factors = module.PdiDurationFactors(pdi_f.wetm, pdi_f.wetb, pdi_f.drym, pdi_f.dryb)
        pdsi = module.calculate(z.reshape(-1, 12, 1), factors).pdsi.reshape(-1)
        masks[division] = np.abs(pdsi - wells_cache[division]) > DISAGREE
    return masks


def episode_label(full: np.ndarray, loo: dict[str, np.ndarray], singles: dict[str, np.ndarray], s: int, e: int) -> str:
    """Label of the base episode ``[s, e)`` from the harmonized-variant disagreement masks (D2)."""
    if full[s:e].any():
        return "residual"
    necessary = [m for m in MECH_NAMES if loo[m][s:e].any()]
    if necessary:
        return "+".join(necessary)
    return "overdetermined:" + "+".join(m for m in MECH_NAMES if not singles[m][s:e].any())


def _remaining_months(
    base: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
    singles: dict[str, dict[str, np.ndarray]],
    loo: dict[str, dict[str, np.ndarray]],
) -> dict[str, int]:
    """Disagreeing months left by the base, full and leave-one-out harmonizations."""
    remaining = {
        "none": int(sum(m.sum() for m in base.values())),
        "all": int(sum(m.sum() for m in full.values())),
    }
    for mech in MECH_NAMES:
        remaining[f"only {mech}"] = int(sum(v.sum() for v in singles[mech].values()))
        remaining[f"all but {mech}"] = int(sum(v.sum() for v in loo[mech].values()))
    return remaining


def _label_episodes(
    base: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
    singles: dict[str, dict[str, np.ndarray]],
    loo: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, int], dict[str, int], dict[str, list[tuple[int, int, str]]]]:
    """Count base episodes and months per label and keep each episode's span."""
    labels: dict[str, int] = {}
    months: dict[str, int] = {}
    episode_labels: dict[str, list[tuple[int, int, str]]] = {}
    for division in divisions():
        per_division: list[tuple[int, int, str]] = []
        for s, e in episodes(base[division]):
            label = episode_label(
                full[division],
                {m: loo[m][division] for m in MECH_NAMES},
                {m: singles[m][division] for m in MECH_NAMES},
                s,
                e,
            )
            per_division.append((s, e, label))
            labels[label] = labels.get(label, 0) + 1
            months[label] = months.get(label, 0) + (e - s)
        episode_labels[division] = per_division
    return labels, months, episode_labels


def attribute(wells_cache: dict[str, np.ndarray], base: dict[str, np.ndarray]) -> dict[str, Any]:
    """Leave-one-out attribution of every base episode (deviation D2).

    An episode's necessary set is the mechanisms whose omission from the full
    harmonization leaves the episode's span disagreeing.
    """
    full = variant_masks(MECH_NAMES, wells_cache)
    singles = {m: variant_masks((m,), wells_cache) for m in MECH_NAMES}
    loo = {m: variant_masks(tuple(x for x in MECH_NAMES if x != m), wells_cache) for m in MECH_NAMES}
    labels, months, episode_labels = _label_episodes(base, full, singles, loo)
    return {
        "remaining_disagreeing_months": _remaining_months(base, full, singles, loo),
        "episodes_by_label": dict(sorted(labels.items())),
        "months_by_label": dict(sorted(months.items())),
        "_episode_labels": episode_labels,
    }


PALMER_EDGES = (0.5, 1.0, 2.0, 3.0, 4.0)


def palmer_category(x: np.ndarray) -> np.ndarray:
    """Signed Palmer class: 0 near normal, +-1 incipient ... +-5 extreme (left-closed in magnitude)."""
    categories: np.ndarray = (np.sign(x) * np.digitize(np.abs(x), PALMER_EDGES)).astype(int)
    return categories


def trace_pdi_x3(z: np.ndarray, factors: PdiDurationFactors) -> tuple[np.ndarray, np.ndarray]:
    """pdi.f PDSI and its per-month X3, by snapshotting the state after each ``_advance_month``."""
    original = _palmer_pdi._advance_month
    x3: list[float] = []

    def advance(fac: Any, state: Any, year: int, month: int) -> None:
        original(fac, state, year, month)
        x3.append(float(state.px3[year, month, 0]))

    _palmer_pdi._advance_month = advance  # type: ignore[assignment]
    try:
        result = run_pdi(z, factors)
    finally:
        _palmer_pdi._advance_month = original
    return result.pdsi.reshape(-1), np.array(x3)


def histogram(lengths: list[int]) -> dict[str, int]:
    bins = [
        ("1", 1, 1),
        ("2", 2, 2),
        ("3", 3, 3),
        ("4-6", 4, 6),
        ("7-12", 7, 12),
        ("13-24", 13, 24),
        (">24", 25, 10**9),
    ]
    return {name: sum(lo <= n <= hi for n in lengths) for name, lo, hi in bins}


def _base_masks(
    divs: tuple[str, ...],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    """Run both recursions per division and keep the outputs, disagreement mask and X3 pair."""
    pdi_pdsi: dict[str, np.ndarray] = {}
    wells_pdsi: dict[str, np.ndarray] = {}
    base: dict[str, np.ndarray] = {}
    x3_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for division in divs:
        z = load_z(division)
        pdi_f, wells_f = default_factors()
        p, x3_p = trace_pdi_x3(z, pdi_f)
        w = _palmer_wells.calculate(z, factors=wells_f)
        pdi_pdsi[division], wells_pdsi[division] = p, w.pdsi
        base[division] = np.abs(p - w.pdsi) > DISAGREE
        x3_pairs[division] = (x3_p, w.x3)
    return pdi_pdsi, wells_pdsi, base, x3_pairs


def _floating_point_summary(
    divs: tuple[str, ...], base: dict[str, np.ndarray]
) -> tuple[dict[str, dict[str, int]], int]:
    """Per-variant counts and the months in base episodes removed by either variant."""
    fp: dict[str, dict[str, int]] = {}
    removed_episodes: set[tuple[str, int, int]] = set()
    for variant in ("fp", "fp2"):
        masks = disagree_masks(variant)
        removed = {(d, s, e) for d in divs for s, e in episodes(base[d]) if not masks[d][s:e].any()}
        removed_episodes.update(removed)
        fp[variant] = {
            "disagreeing_months": int(sum(m.sum() for m in masks.values())),
            "months_changed_vs_base": int(sum((masks[d] ^ base[d]).sum() for d in divs)),
            "base_episodes_removed": len(removed),
            "base_months_removed": sum(e - s for _, s, e in removed),
        }
    return fp, sum(e - s for _, s, e in removed_episodes)


def _descriptive(
    divs: tuple[str, ...],
    base: dict[str, np.ndarray],
    pdi_pdsi: dict[str, np.ndarray],
    wells_pdsi: dict[str, np.ndarray],
    episode_labels: dict[str, list[tuple[int, int, str]]],
    n_series: int,
) -> dict[str, Any]:
    """Episode examples, decade and season rates, category changes and largest per label."""
    examples: dict[str, list[list[Any]]] = {}
    for d in divs:
        for s, e, label in episode_labels[d]:
            examples.setdefault(label, []).append([e - s, d, s, e])
    examples = {k: sorted(v)[:5] for k, v in sorted(examples.items())}
    by_decade: dict[str, list[int]] = {}
    by_season = np.zeros(12, dtype=int)
    cat_diff = 0
    largest: dict[str, dict[str, Any]] = {}
    for d in divs:
        idx = np.flatnonzero(base[d])
        for t in idx:
            decade = str((START_YEAR + t // 12) // 10 * 10)
            by_decade.setdefault(decade, [0, 0])[0] += 1
        by_season += np.bincount(idx % 12, minlength=12)
        cat_diff += int(np.sum(palmer_category(pdi_pdsi[d][idx]) != palmer_category(wells_pdsi[d][idx])))
        for s, e, label in episode_labels[d]:
            diff = np.abs(pdi_pdsi[d][s:e] - wells_pdsi[d][s:e])
            k = int(diff.argmax())
            if label not in largest or diff[k] > largest[label]["abs_diff"]:
                largest[label] = {
                    "abs_diff": float(diff[k]),
                    "division": d,
                    "month_index": s + k,
                    "year": START_YEAR + (s + k) // 12,
                    "month": (s + k) % 12 + 1,
                }
    for peak in largest.values():
        peak["abs_diff"] = round(peak["abs_diff"], 6)
    decade_totals: dict[str, int] = {}
    for month in range(n_series):
        key = str((START_YEAR + month // 12) // 10 * 10)
        decade_totals[key] = decade_totals.get(key, 0) + len(divs)
    decade_rate = {k: round(v[0] / decade_totals[k], 6) for k, v in sorted(by_decade.items())}
    return {
        "examples": examples,
        "by_season": by_season,
        "cat_diff": cat_diff,
        "largest": largest,
        "decade_rate": decade_rate,
    }


def _h5_counts(
    divs: tuple[str, ...],
    base: dict[str, np.ndarray],
    x3_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    n_series: int,
) -> tuple[int, int]:
    """Months after three same-sign established months in both machines, and how many disagree."""
    h5_n = h5_bad = 0
    for d in divs:
        x3_p, x3_w = x3_pairs[d]
        for month in range(3, n_series):
            window_p, window_w = x3_p[month - 3 : month], x3_w[month - 3 : month]
            same_sign = np.all(np.sign(window_p) == np.sign(window_w)) and np.all(
                np.sign(window_p) == np.sign(window_p[0])
            )
            if same_sign and np.all(np.abs(window_p) > 1.5) and np.all(np.abs(window_w) > 1.5):
                h5_n += 1
                h5_bad += int(base[d][month])
    return h5_n, h5_bad


def _hypotheses(
    head_p: dict[str, Any],
    floating_point_months: int,
    attribution: dict[str, Any],
    h5_n: int,
    h5_bad: int,
) -> dict[str, dict[str, Any]]:
    """The pre-registered H1 to H5 verdicts."""
    closer = head_p["closer_to_nclimdiv"]
    decided = closer["pdi"] + closer["wells"]
    n_dis = head_p["disagree_counts"][str(DISAGREE)]
    n_unclassified = attribution["episodes_by_label"].get("residual", 0)
    h3_share = floating_point_months / max(n_dis, 1)
    return {
        "H1": {
            "rate": round(head_p["disagree_rate"], 6),
            "max_abs_diff": round(head_p["max_abs_diff"], 6),
            "pass": 0.053 <= head_p["disagree_rate"] <= 0.073 and 7.0 <= head_p["max_abs_diff"] <= 9.0,
        },
        "H2": {"residual_episodes": n_unclassified, "pass": n_unclassified == 0},
        "H3": {
            "floating_point_months_share": round(h3_share, 8),
            "pass": h3_share < 0.001,
        },
        "H4": {
            "pdi_closer_share": round(closer["pdi"] / decided, 6),
            "ties": closer["tie"],
            "pass": closer["pdi"] / decided >= 0.80,
        },
        "H5": {
            "n": h5_n,
            "disagreeing": h5_bad,
            "rate": round(h5_bad / max(h5_n, 1), 6),
            "pass": h5_bad / max(h5_n, 1) < 0.005,
        },
    }


def run_all() -> dict[str, Any]:
    divs = divisions()
    pdi_pdsi, wells_pdsi, base, x3_pairs = _base_masks(divs)
    n_months = sum(m.size for m in base.values())
    n_series = next(iter(base.values())).size

    head_p = headline("P")
    head_f = headline("F")
    per_division_p = head_p.pop("per_division_rate")
    head_f.pop("per_division_rate")

    fp, floating_point_months = _floating_point_summary(divs, base)
    attribution = attribute(wells_pdsi, base)
    episode_labels = attribution.pop("_episode_labels")
    lengths = [e - s for d in divs for s, e in episodes(base[d])]
    descriptive = _descriptive(divs, base, pdi_pdsi, wells_pdsi, episode_labels, n_series)
    h5_n, h5_bad = _h5_counts(divs, base, x3_pairs, n_series)
    n_dis = head_p["disagree_counts"][str(DISAGREE)]
    return {
        "pinned_commit": PINNED_COMMIT,
        "months": n_months,
        "episodes": len(lengths),
        "headline": {"P": head_p, "F": head_f},
        "floating_point": fp,
        "attribution": attribution,
        "largest_by_label": dict(sorted(descriptive["largest"].items())),
        "example_episodes_by_label": descriptive["examples"],
        "episode_length_histogram": histogram(lengths),
        "category_changes": {
            "months": descriptive["cat_diff"],
            "share_of_disagreeing_months": round(descriptive["cat_diff"] / n_dis, 6),
        },
        "disagreement_rate_by_decade": descriptive["decade_rate"],
        "disagreeing_months_by_calendar_month": {str(i + 1): int(v) for i, v in enumerate(descriptive["by_season"])},
        "per_division_rate_P": {k: round(v, 6) for k, v in per_division_p.items()},
        "hypotheses": _hypotheses(head_p, floating_point_months, attribution, h5_n, h5_bad),
    }


def _complement_fraction(m: float, b: float) -> float:
    """The Wells weighting-fraction form, ``1 - m / (m + b)``, for the P-fp run."""
    return 1.0 - m / (m + b)


def division_form(wells: DurationFactors) -> DurationFactors:
    """Wells factors with the pdi.f division form ``b / (m + b)``, for the P-fp2 run."""
    return DurationFactors(
        wetm=wells.wetm,
        wetb=wells.wetb,
        drym=wells.drym,
        dryb=wells.dryb,
        wet_denominator=wells.wet_denominator,
        dry_denominator=wells.dry_denominator,
        wetc=wells.wetb / wells.wet_denominator,
        dryc=wells.wetb / (wells.drym + wells.wetb),
        dry_spell_c=wells.dryb / wells.dry_denominator,
    )


def disagree_masks(variant: str) -> dict[str, np.ndarray]:
    """Per-division disagreement masks on configuration P under a floating-point variant.

    ``base`` changes nothing, ``fp`` gives pdi.f the Wells complement form, ``fp2`` gives
    Wells the pdi.f division form.
    """
    original = _palmer_pdi._weighting_fraction
    try:
        if variant == "fp":
            _palmer_pdi._weighting_fraction = _complement_fraction  # type: ignore[assignment]
        masks = {}
        for division in divisions():
            pdi_f, wells_f = default_factors()
            if variant == "fp2":
                wells_f = division_form(wells_f)
            pdi_pdsi, wells, _ = run_both(load_z(division), pdi_f, wells_f)
            masks[division] = np.abs(pdi_pdsi - wells.pdsi) > DISAGREE
        return masks
    finally:
        _palmer_pdi._weighting_fraction = original


def write_results(out: Path, results: dict[str, Any]) -> None:
    """Write the artifact only through a resolved directory inside the checkout."""
    parent = out.parent.resolve()
    if not parent.is_relative_to(ROOT.resolve()):
        raise SystemExit(f"output directory {parent} is outside the checkout")
    out = parent / out.name
    if out.is_symlink():
        raise SystemExit(f"refusing to write through a symlink: {out}")
    out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["probe"]:
        r = signatures("P")
        print(r["n_out_episodes"], r["n_state_episodes"], r["out_without_state"])
    elif arguments:
        print(f"usage: {Path(__file__).name} [probe]\nunknown arguments: {arguments}", file=sys.stderr)
        raise SystemExit(2)
    else:
        verify_pinned_sources()
        out = ROOT / "docs" / "research" / "reference-papers" / "palmer-recursion-divergence-results.json"
        if out.is_symlink():
            raise SystemExit(f"refusing to write through a symlink: {out}")
        results = run_all()
        write_results(out, results)
        print(f"wrote {out}")
        failed = [name for name, hypothesis in results["hypotheses"].items() if not hypothesis["pass"]]
        if failed:
            print(f"failed hypotheses: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1 if failed else 0)

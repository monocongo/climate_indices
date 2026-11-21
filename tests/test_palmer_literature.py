"""Palmer (1965) printed worked examples against the library, stage by stage (REFIMPL-28, #1630).

The numbers are transcribed from Tables 1, 2, 3, 4, 9 and 12 of Palmer, W. C. (1965),
*Meteorological Drought*, U.S. Weather Bureau Research Paper No. 45, in
``tests/fixture/palmer_literature/``. The tolerances, the hypotheses and the printing
inconsistencies found in the paper are described in that directory's README; they were
committed before any library comparison was run.

Two kinds of test live here:

* *Audits* use the printed numbers alone. They protect the transcription and establish
  which printed cells are inconsistent with their own neighbours; no library code runs.
* *Stage tests* feed the printed inputs of one stage to the library function that owns
  that stage and compare with the printed outputs. A bound is the sum of the half-units of
  every rounded printed quantity that enters the result, weighted by its coefficient.

Where the library does not reproduce the print, the test pins the measured deviation and
names its cause; no tolerance is widened to make a month pass.
"""

import csv
import hashlib
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from climate_indices.drought import _palmer_pdi, _palmer_wells, palmer
from climate_indices.drought._palmer_duration import DurationFactors
from climate_indices.drought._palmer_pdi import PdiDurationFactors

pytestmark = pytest.mark.validation

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixture" / "palmer_literature"
_PROVENANCE = json.loads((FIXTURE / "provenance.json").read_text())
_TOL = _PROVENANCE["validation_tolerance"]

HALF_2DP = _TOL["printed_half_unit_2dp"]  # a value printed to 0.01
HALF_4DP = _TOL["printed_half_unit_4dp"]  # a value printed to 0.0001
AUDIT_ATOL = _TOL["audit_atol_one_print_unit"]
WATER_BALANCE_ATOL = _TOL["water_balance_atol_in"]
X_ATOL = _TOL["x_recursion_atol"]


def _rows(name: str) -> list[dict[str, str]]:
    with (FIXTURE / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _num(text: str) -> float:
    """A printed number; the trace token ``T`` is below the 0.005 print resolution, so 0."""
    return 0.0 if text == "T" else float(text)


def _col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([_num(row[name]) for row in rows])


def _region(rows: list[dict[str, str]], region: str) -> list[dict[str, str]]:
    return [row for row in rows if row["region"] == region and row["month"] != "sum"]


def _ym(row: dict[str, str]) -> tuple[int, int]:
    return int(row["year"]), int(row["month"])


@pytest.fixture
def python_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the Python kernels: the Rust extension is checked against them elsewhere."""
    monkeypatch.setattr(palmer, "_native", None)


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------


def test_fixture_checksum_matches_provenance() -> None:
    """The six CSVs, concatenated in filename-sorted order, are what the provenance records."""
    digest = hashlib.sha256()
    for path in sorted(FIXTURE.glob("table*.csv")):
        digest.update(path.read_bytes())
    assert len(list(FIXTURE.glob("table*.csv"))) == 6
    assert digest.hexdigest() == _PROVENANCE["checksum_sha256"]


# ---------------------------------------------------------------------------
# Audits: printed numbers only
# ---------------------------------------------------------------------------


def test_table1_printed_identities() -> None:
    """Table 1 obeys its own bookkeeping: S = Ss + Su, P = ET + RO + dS, PR = AWC - S'."""
    rows = _rows("table1_water_balance_central_iowa.csv")
    previous_s = 10.0  # both layers full before January 1933
    for row in rows:
        value = {key: _num(text) for key, text in row.items()}
        tag = _ym(row)
        d_s = value["delta_ss_in"] + value["delta_su_in"]
        assert abs(value["ss_in"] + value["su_in"] - value["s_in"]) <= AUDIT_ATOL, tag
        assert abs(value["p_in"] - (value["et_in"] + value["ro_in"] + d_s)) <= AUDIT_ATOL, tag
        assert abs(value["s_in"] - previous_s - d_s) <= AUDIT_ATOL, tag
        assert abs(10.0 - previous_s - value["pr_in"]) <= AUDIT_ATOL, tag
        previous_s = value["s_in"]


_TABLE2_SUMS = {
    "western_kansas": {"et": 18.76, "pe": 29.39, "r": 3.43, "ro": 0.29, "l": 3.44, "p": 19.08},
    "central_iowa": {"et": 25.49, "pe": 27.96, "r": 5.84, "ro": 5.76, "l": 5.58, "p": 31.50},
}


@pytest.mark.parametrize("region", sorted(_TABLE2_SUMS))
def test_table2_columns_add_to_the_printed_sums(region: str) -> None:
    """Every Table 2 column whose annual sum is printed adds to it, and S' = S's + S'u."""
    rows = _region(_rows("table2_long_term_means.csv"), region)
    for name, printed in _TABLE2_SUMS[region].items():
        assert abs(_col(rows, f"{name}_in").sum() - printed) <= HALF_2DP + 1e-9, name
    surface_plus_under = _col(rows, "s_prime_s_in") + _col(rows, "s_prime_u_in")
    np.testing.assert_allclose(surface_plus_under, _col(rows, "s_prime_in"), atol=AUDIT_ATOL)


# The printed row has these two columns transposed; the P-hat, d, z and Z columns are consistent
# with the swapped pair (README, "Printing inconsistencies found").
_TABLE4_TRANSPOSED = {(1934, 4)}


def test_table4_printed_identities() -> None:
    """Table 4: P-hat = col6 + col7 + col8 - col9, d = P - P-hat, z = d k, Z = d K, terms = coefficient x input."""
    coefficients = {int(r["month"]): r for r in _region(_rows("table3_coefficients.csv"), "western_kansas")}
    for row in _rows("table4_moisture_departures_western_kansas.csv"):
        value = {key: float(text) for key, text in row.items()}
        tag = _ym(row)
        coef = {key: float(text) for key, text in coefficients[int(row["month"])].items() if key not in ("region",)}
        gamma_term, delta_term = value["gamma_pro_in"], value["delta_pl_in"]
        if tag in _TABLE4_TRANSPOSED:
            gamma_term, delta_term = delta_term, gamma_term
        phat = value["alpha_pe_in"] + value["beta_pr_in"] + gamma_term - delta_term
        assert abs(phat - value["p_hat_in"]) <= 2 * HALF_2DP + 1e-9, tag
        assert abs(value["p_in"] - value["p_hat_in"] - value["d_in"]) <= AUDIT_ATOL, tag
        # Z = d K and z = d k, each carrying the half-unit of d and of the weight.
        assert (
            abs(value["d_in"] * coef["k_weight"] - value["z_big_in"])
            <= (abs(value["d_in"]) + abs(coef["k_weight"]) + 1) * HALF_2DP
        ), tag
        assert (
            abs(value["d_in"] * coef["k"] - value["z_small_in"]) <= (abs(value["d_in"]) + abs(coef["k"]) + 1) * HALF_2DP
        ), tag
        for term, coefficient, base in (
            ("alpha_pe_in", "alpha", "pe_in"),
            ("beta_pr_in", "beta", "pr_in"),
            ("gamma_pro_in", "gamma", "pro_in"),
            ("delta_pl_in", "delta", "pl_in"),
        ):
            printed_term = value[term]
            if tag in _TABLE4_TRANSPOSED and term == "gamma_pro_in":
                printed_term = value["delta_pl_in"]
            elif tag in _TABLE4_TRANSPOSED and term == "delta_pl_in":
                printed_term = value["gamma_pro_in"]
            assert abs(coef[coefficient] * value[base] - printed_term) <= AUDIT_ATOL, (tag, term)


def test_table4_april_1934_columns_8_and_9_are_transposed_in_the_print() -> None:
    """Pin the printing error: the printed pair only satisfies the coefficients when swapped."""
    row = next(r for r in _rows("table4_moisture_departures_western_kansas.csv") if _ym(r) == (1934, 4))
    coef = next(r for r in _region(_rows("table3_coefficients.csv"), "western_kansas") if r["month"] == "4")
    gamma_expected = float(coef["gamma"]) * float(row["pro_in"])
    delta_expected = float(coef["delta"]) * float(row["pl_in"])
    assert abs(gamma_expected - float(row["gamma_pro_in"])) > 0.1
    assert abs(delta_expected - float(row["delta_pl_in"])) > 0.1
    assert abs(gamma_expected - float(row["delta_pl_in"])) <= AUDIT_ATOL
    assert abs(delta_expected - float(row["gamma_pro_in"])) <= AUDIT_ATOL


# Table 9 prints western Kansas K for April and May transposed relative to Table 3 column 7
# and to K = 17.67 K' / sum(D K').
_TABLE9_TRANSPOSED = {("western_kansas", 4), ("western_kansas", 5)}

# Equation 26 with Table 2's means and Table 9's D-bar reproduces the printed K' within the rounding
# of those inputs in 23 of 24 cases. Central Iowa December prints K' = 1.55; the means give 1.566
# (T = 1.00, D-bar = 0.74), 0.016 against a propagated bound of 0.013, and K = 17.67 K' / sum agrees
# with the printed 1.55. Cause not established; logged in the discrepancy ledger (#1626).
_TABLE9_EQ26_EXCEEDS = {("central_iowa", 12): 0.0158}


@pytest.mark.parametrize("region", ["western_kansas", "central_iowa"])
def test_table9_printed_identities(region: str) -> None:
    """D K', its printed sum, K = 17.67 K' / sum, and Equation 26 for K' from Table 2's means."""
    table9 = _rows("table9_k_factors.csv")
    printed_sum = float(next(r for r in table9 if r["region"] == region and r["month"] == "sum")["dbar_k_prime"])
    means = {int(r["month"]): r for r in _region(_rows("table2_long_term_means.csv"), region)}
    k3 = {int(r["month"]): float(r["k_weight"]) for r in _region(_rows("table3_coefficients.csv"), region)}
    total = 0.0
    for row in _region(table9, region):
        month = int(row["month"])
        dbar, k_prime = float(row["d_bar_in"]), float(row["k_prime"])
        product = dbar * k_prime
        total += product
        if row["dbar_k_prime"]:
            assert abs(product - float(row["dbar_k_prime"])) <= (dbar + k_prime + 1) * HALF_2DP, (region, month)
        k_weight = float(row["k_weight"])
        derived = 17.67 * k_prime / printed_sum
        if (region, month) in _TABLE9_TRANSPOSED:
            assert abs(derived - k_weight) > 0.1, (region, month)
            assert abs(derived - k3[month]) <= (derived / k_prime + derived / printed_sum + 1) * HALF_2DP
        else:
            assert abs(derived - k_weight) <= (derived / k_prime + derived / printed_sum + 1) * HALF_2DP, (
                region,
                month,
            )
            assert k_weight == k3[month], (region, month)
        # Equation 26: K' = 1.5 log10[((PE + R + RO) / (P + L) + 2.80) / D] + 0.50
        mean = {key: _num(text) for key, text in means[month].items() if key not in ("region",)}
        t_ratio = (mean["pe_in"] + mean["r_in"] + mean["ro_in"]) / (mean["p_in"] + mean["l_in"])
        d_t = (3 * HALF_2DP + t_ratio * 2 * HALF_2DP) / (mean["p_in"] + mean["l_in"])
        bound = 1.5 / math.log(10) * (d_t / (t_ratio + 2.8) + HALF_2DP / dbar) + HALF_2DP
        deviation = abs(1.5 * math.log10((t_ratio + 2.8) / dbar) + 0.5 - k_prime)
        if (region, month) in _TABLE9_EQ26_EXCEEDS:
            assert bound < deviation == pytest.approx(_TABLE9_EQ26_EXCEEDS[(region, month)], abs=0.0005)
        else:
            assert deviation <= bound, (region, month)
    assert abs(total - printed_sum) <= HALF_2DP + 1e-9


# Printed inconsistencies in Table 12, each with the arithmetic of its own row or neighbours.
_TABLE12_INCONSISTENT = {
    (1889, 6): "x1",  # X1(May) 0.95 + dX1 0.25 = 1.20, printed 1.10
    (1890, 6): "z_over_3",  # Z = -2.92 but Z/3 = -0.94 (Z = -2.82)
    (1891, 7): "z_over_3",  # Z = 4.22 but Z/3 = 1.74 (Z = 5.22)
}


def _table12_audit_failures() -> dict[tuple[int, int], set[str]]:
    rows = _rows("table12_severity_western_kansas.csv")
    failures: dict[tuple[int, int], set[str]] = {}

    def value(row: dict[str, str], key: str) -> float | None:
        return float(row[key]) if row[key] != "" else None

    for index, row in enumerate(rows):
        tag = _ym(row)
        z, z_over_3 = value(row, "z"), value(row, "z_over_3")
        assert z is not None and z_over_3 is not None
        if abs(z / 3 - z_over_3) > HALF_2DP + 0.0017:
            failures.setdefault(tag, set()).add("z_over_3")
        v, q, pe = value(row, "v"), value(row, "q"), value(row, "pe_pct")
        if q and v is not None and pe is not None and abs(min(100.0, 100 * v / q) - pe) > 0.6:
            failures.setdefault(tag, set()).add("pe_pct")
        if index == 0:
            continue
        previous = rows[index - 1]
        for suffix in ("1", "2", "3"):
            m103, d_x, x = value(row, f"m103_x{suffix}"), value(row, f"dx{suffix}"), value(row, f"x{suffix}")
            x_previous = value(previous, f"x{suffix}")
            if m103 is None or x_previous is None:
                continue
            if abs(-0.103 * x_previous - m103) > AUDIT_ATOL:
                failures.setdefault(tag, set()).add(f"m103_x{suffix}")
            if d_x is not None and abs(z_over_3 + m103 - d_x) > AUDIT_ATOL:
                failures.setdefault(tag, set()).add(f"dx{suffix}")
            if d_x is not None and x is not None and x != 0 and abs(x_previous + d_x - x) > AUDIT_ATOL:
                failures.setdefault(tag, set()).add(f"x{suffix}")
        if not any(row[k] != "" and float(row[k]) == float(row["x"]) for k in ("x1", "x2", "x3")):
            failures.setdefault(tag, set()).add("x_not_a_candidate")
    return failures


def test_table12_printed_recurrences_fail_only_where_the_paper_is_inconsistent() -> None:
    """Pe = 100 V/Q, Z/3, -0.103 X(i-1), dX = Z/3 - 0.103 X(i-1) and X = X(i-1) + dX hold except in three cells."""
    failures = _table12_audit_failures()
    assert failures == {tag: {cell} for tag, cell in _TABLE12_INCONSISTENT.items()}


# ---------------------------------------------------------------------------
# Stage tests
# ---------------------------------------------------------------------------

_WATER_BALANCE_COLUMNS = (
    ("ss", "ss_in"),
    ("su", "su_in"),
    ("r", "r_in"),
    ("ro", "ro_in"),
    ("l", "l_in"),
    ("et", "et_in"),
    ("pl", "pl_in"),
)

# One month of the water balance from the printed start state: months where the library differs
# from the print by more than one print unit even from the printed state, with the measured
# deviation. Print rounding of P, PE, Ss and Su explains 1933-08 but not 1934-05 (0.0201 against
# a propagated bound of 0.016) and only just misses 1934-07; in each, the printed loss is smaller
# than (PE - P) S'u / AWC with the printed values. Logged in the discrepancy ledger (#1626).
_WATER_BALANCE_EXCEEDS = {(1933, 8): 0.012, (1934, 5): 0.0201, (1934, 7): 0.0124}


def _one_month(p: float, pe: float, ss: float, su: float) -> np.ndarray:
    et, loss, recharge, runoff, new_ss, new_su = palmer._calc_recharge(p, pe, ss, su, 10.0)
    potential_loss = palmer._calc_potential_loss(pe, ss, su, 10.0)
    return np.array([float(np.squeeze(x)) for x in (new_ss, new_su, recharge, runoff, loss, et, potential_loss)])


def test_h1_water_balance_one_month_from_the_printed_state(python_path: None) -> None:
    """Table 1 (central Iowa, AWC 1 + 9): each month's balance from the printed start state."""
    rows = _rows("table1_water_balance_central_iowa.csv")
    ss, su = 1.0, 9.0
    exceeded: dict[tuple[int, int], float] = {}
    for row in rows:
        value = {key: float(text) for key, text in row.items()}
        library = _one_month(value["p_in"], value["pe_in"], ss, su)
        printed = np.array([value[column] for _, column in _WATER_BALANCE_COLUMNS])
        deviation = float(np.abs(library - printed).max())
        assert abs(10.0 - (ss + su) - value["pr_in"]) <= AUDIT_ATOL  # PR needs no kernel
        if deviation > WATER_BALANCE_ATOL:
            exceeded[_ym(row)] = round(deviation, 4)
        ss, su = value["ss_in"], value["su_in"]
    assert exceeded == pytest.approx(_WATER_BALANCE_EXCEEDS, abs=0.0006)


def test_h1_water_balance_carrying_the_library_state_stays_within_print_rounding(python_path: None) -> None:
    """Characterization: the library's own unrounded state drifts from the printed one by at most 0.03 in.

    Palmer carried unrounded state through the same recursion; printing P and PE to 0.01 makes the
    two trajectories differ, and a dry spell keeps that difference until the soil refills. Measured
    maximum 0.0286 (S'u, July 1934); this is a ceiling for the run, not a tolerance.
    """
    rows = _rows("table1_water_balance_central_iowa.csv")
    prepared = palmer._initialize_prepared(_col(rows, "p_in"), _col(rows, "pe_in"), 10.0, 1933, 1933, 1935)
    palmer._calc_water_balances(prepared)
    library = {
        "ss_in": prepared.sssdat,
        "su_in": prepared.ssudat,
        "r_in": prepared.rdat,
        "ro_in": prepared.rodat,
        "l_in": prepared.tldat,
        "et_in": prepared.etdat,
        "pl_in": prepared.pldat,
        "pr_in": prepared.prdat,
    }
    worst = max(float(np.abs(values.reshape(-1) - _col(rows, name)).max()) for name, values in library.items())
    assert 0.02 < worst < 0.03
    # PRO is the soil moisture at the start of the month: the previous month's printed S.
    np.testing.assert_allclose(prepared.spdat.reshape(-1)[1:], _col(rows, "s_in")[:-1], atol=0.03)


def _ratio_interval(numerator: float, denominator: float) -> tuple[float, float] | None:
    """Interval for a coefficient built from two means printed to 0.01; None when the mean could be 0."""
    if denominator - HALF_2DP <= 0:
        return None
    return max(numerator - HALF_2DP, 0.0) / (denominator + HALF_2DP), (numerator + HALF_2DP) / (denominator - HALF_2DP)


def test_h2_cafec_coefficients_follow_from_the_printed_means() -> None:
    """Table 3's alpha..delta are the library's ratio of Table 2's means, to within the means' rounding.

    Table 3's footnote says the coefficients came from long-term sums rather than the rounded means,
    so agreement is only to within the rounding of the means: an interval, not a point.
    """
    means = _rows("table2_long_term_means.csv")
    coefficients = _rows("table3_coefficients.csv")
    testable = informative = 0
    outside: list[tuple[str, str, str, float]] = []
    for mean_row, coefficient_row in zip(means, coefficients, strict=True):
        assert (mean_row["region"], mean_row["month"]) == (coefficient_row["region"], coefficient_row["month"])
        mean = {key: _num(text) for key, text in mean_row.items() if key not in ("region",)}
        for name, numerator, denominator, both_zero in (
            ("alpha", mean["et_in"], mean["pe_in"], 1.0),
            ("beta", mean["r_in"], mean["pr_in"], 1.0),
            ("gamma", mean["ro_in"], mean["s_prime_in"], 1.0),
            ("delta", mean["l_in"], mean["pl_in"], 0.0),
        ):
            printed = float(coefficient_row[name])
            point = float(palmer._calc_cafec_ratio(np.array([numerator]), np.array([denominator]), both_zero)[0])
            interval = _ratio_interval(numerator, denominator)
            if interval is None:
                # both means printed as 0: the library's convention (1.0, or 0.0 for delta) is Table 3's
                if denominator == 0 and numerator == 0:
                    assert printed == point, (mean_row["region"], mean_row["month"], name)
                continue
            testable += 1
            low, high = interval
            assert low - HALF_4DP <= point <= high + HALF_4DP
            informative += high - low <= 0.02
            if not low - HALF_4DP <= printed <= high + HALF_4DP:
                outside.append(
                    (
                        mean_row["region"],
                        mean_row["month"],
                        name,
                        round(min(abs(printed - low), abs(printed - high)), 4),
                    )
                )
    # western Kansas May beta: R = 0.38, PR = 3.36 allow at most 0.1148, Table 3 prints 0.1155
    assert [entry[:3] for entry in outside] == [("western_kansas", "5", "beta")]
    assert outside[0][3] == pytest.approx(0.0007, abs=0.0001)
    assert (testable, informative) == (92, 75)


@pytest.mark.parametrize("region", ["western_kansas", "central_iowa"])
def test_h3_k_factors_normalize_to_table_3(region: str, monkeypatch: pytest.MonkeyPatch, python_path: None) -> None:
    """Given Table 9's printed D-bar and K', the library's K = 17.67 K' / sum(D-bar K') is Table 3's K.

    D-bar comes from the library's own calibration record, so it is injected here: this exercises the
    normalization, and Equation 26 for K' is audited from the print above.
    """
    rows = _region(_rows("table9_k_factors.csv"), region)
    dbar = _col(rows, "d_bar_in").reshape(12, 1)
    k_prime = _col(rows, "k_prime").reshape(12, 1)
    monkeypatch.setattr(palmer, "_calc_k_prime_and_dbar", lambda _prepared: (dbar, k_prime))
    prepared = palmer._initialize_prepared(np.zeros(24), np.zeros(24), 10.0, 2000, 2000, 2001)
    palmer._calc_kfactors(prepared)

    sum_bound = float(((dbar + k_prime) * HALF_2DP).sum())
    printed = _col(_region(_rows("table3_coefficients.csv"), region), "k_weight")
    library = prepared.ak.reshape(12)
    bound = library * (HALF_2DP / k_prime.reshape(12) + sum_bound / float((dbar * k_prime).sum())) + HALF_2DP
    assert np.all(np.abs(library - printed) <= bound)


def test_h4_cafec_precipitation_d_and_z_for_western_kansas(python_path: None) -> None:
    """Table 4: from printed PE, PR, PRO, PL, P and Table 3's coefficients and K, the library's P-hat, d and Z."""
    coefficients = _region(_rows("table3_coefficients.csv"), "western_kansas")
    prepared = palmer._initialize_prepared(np.zeros(48), np.zeros(48), 6.0, 1932, 1932, 1935)
    for name in ("alpha", "beta", "gamma", "delta"):
        setattr(prepared, name, _col(coefficients, name).reshape(12, 1))
    prepared.ak = _col(coefficients, "k_weight").reshape(12, 1)
    z = np.full((4, 12, 1), np.nan)
    for row in _rows("table4_moisture_departures_western_kansas.csv"):
        value = {key: float(text) for key, text in row.items()}
        year, month = int(value["year"]) - 1932, int(value["month"]) - 1
        prepared.pet[year, month] = value["pe_in"]
        prepared.prdat[year, month] = value["pr_in"]
        prepared.spdat[year, month] = value["pro_in"]
        prepared.pldat[year, month] = value["pl_in"]
        prepared.precips[year, month] = value["p_in"]
        p_hat = float(np.asarray(palmer._calc_cafec_zindex(prepared, z, year, month)).reshape(-1)[0])
        d_library = value["p_in"] - p_hat
        coef = {key: float(text) for key, text in coefficients[month].items() if key != "region"}
        # Table 4's footnote: col 10 = col 6 + col 7 + col 8 - col 9, so the printed P-hat is the sum of
        # four products each rounded to 0.01. Added to that: rounding of the four inputs (0.01) and of
        # the four coefficients (0.0001), which the library multiplies unrounded.
        inputs = (value["pe_in"], value["pr_in"], value["pro_in"], value["pl_in"])
        weights = (coef["alpha"], coef["beta"], coef["gamma"], coef["delta"])
        carried = sum(w * HALF_2DP for w in weights) + sum(x * HALF_4DP for x in inputs)
        summed_terms = 4 * HALF_2DP
        tag = _ym(row)
        assert abs(p_hat - value["p_hat_in"]) <= carried + summed_terms, tag
        assert abs(d_library - value["d_in"]) <= carried + summed_terms + 2 * HALF_2DP, tag
        z_bound = coef["k_weight"] * (carried + summed_terms + HALF_2DP) + abs(value["d_in"]) * HALF_2DP + HALF_2DP
        assert abs(float(z[year, month, 0]) - value["z_big_in"]) <= z_bound, tag


@lru_cache(maxsize=1)
def _table12_z_series() -> np.ndarray:
    """Z from January 1888: Palmer's stated March 1888 Z = +0.81 (X = +0.27), then Table 12's column 2.

    January and February 1888 (Z = 0) and March (Z = 0.81, the value Palmer states) are modelled, not
    printed: the table starts in April, and the library's recursion begins at zero. Padding to whole
    years after July 1891 is Z = 0 and is never compared.
    """
    z = [0.0, 0.0, 0.81] + [float(r["z"]) for r in _rows("table12_severity_western_kansas.csv")]
    return np.array(z + [0.0] * (48 - len(z)))


def _pdi_x(z: np.ndarray) -> np.ndarray:
    factors = DurationFactors.from_defaults()
    pdi_factors = PdiDurationFactors.from_fitted(factors.wetm, factors.wetb, factors.drym, factors.dryb)
    return _palmer_pdi.calculate(z.reshape(4, 12, 1), pdi_factors).pdsi.reshape(-1)[3:43]


def _wells_x(z: np.ndarray) -> np.ndarray:
    return np.asarray(_palmer_wells.calculate(z, factors=DurationFactors.from_defaults()).pdsi)[3:43]


def _printed_x() -> np.ndarray:
    return _col(_rows("table12_severity_western_kansas.csv"), "x")


def _months(rows: list[dict[str, str]], mask: np.ndarray) -> list[tuple[int, int]]:
    return [_ym(row) for row, flag in zip(rows, mask, strict=True) if flag]


# Months where the library's pdi.f recursion differs from the printed X by more than 0.02, each
# downstream of one of the three printed inconsistencies: June 1889 X1 (carried into X3 until the
# spell ends in August), June 1890 Z (the extra 0.033 in Z/3 decays by 0.897 a month) and July 1891 Z
# (Z/3 prints 1.74, i.e. Z = 5.22, not 4.22).
_H5_EXCEEDS = [(1889, 6), (1889, 7), (1890, 6), (1890, 7), (1890, 8), (1890, 9), (1890, 10), (1891, 7)]


def test_h5_pdi_recursion_reproduces_table_12_where_the_paper_is_consistent() -> None:
    """The pdi.f recursion on Table 12's printed Z gives the printed X in 32 of 40 months, within 0.02."""
    rows = _rows("table12_severity_western_kansas.csv")
    deviation = np.abs(_pdi_x(_table12_z_series()) - _printed_x())
    assert _months(rows, deviation > X_ATOL) == _H5_EXCEEDS


def test_h5_the_eight_months_follow_from_the_three_printed_inconsistencies() -> None:
    """Counterfactual: repair the three cells from the print's own arithmetic and every month matches.

    June 1890 and July 1891 take Z = 3 x the printed Z/3 (the value the printed recursion consumed).
    June 1889 X1 is 0.95 + 0.25 = 1.20, which the library gives (1.208); its printed 1.10 carried
    into X3 for June and July, whose printed-recursion values are 1.20 and 1.196.
    """
    rows = _rows("table12_severity_western_kansas.csv")
    repaired = _table12_z_series().copy()
    for (year, month), row in zip((_ym(r) for r in rows), rows, strict=True):
        if (year, month) in {(1890, 6), (1891, 7)}:
            repaired[3 + rows.index(row)] = 3 * float(row["z_over_3"])
    expected = _printed_x().copy()
    june_1889 = next(i for i, row in enumerate(rows) if _ym(row) == (1889, 6))
    expected[june_1889] = 0.95 + 0.25
    expected[june_1889 + 1] = expected[june_1889] + 0.35 / 3 - 0.103 * expected[june_1889]  # Z/3 = 0.12
    deviation = np.abs(_pdi_x(repaired) - expected)
    assert _months(rows, deviation > X_ATOL) == []


def test_h6_wells_recursion_differs_from_table_12_by_the_establishment_threshold() -> None:
    """Wells et al. establish a spell at |X| >= 0.5; Palmer's printed example, like pdi.f, uses 1.

    Fed the same Z, the Wells recursion differs from the printed X in 7 months where the pdi.f
    recursion agrees with it. Switching only the establishment threshold (mechanism E of
    docs/research/reference-papers/palmer-recursion-divergence.md, #1631) in a copy of the pdi.f
    recursion reproduces the Wells series exactly, so the divergence is that one rule.
    """
    rows = _rows("table12_severity_western_kansas.csv")
    z = _table12_z_series()
    wells_gap = np.abs(_wells_x(z) - _printed_x()) > X_ATOL
    pdi_gap = np.abs(_pdi_x(z) - _printed_x()) > X_ATOL
    assert _months(rows, wells_gap & ~pdi_gap) == [
        (1889, 2),
        (1889, 3),
        (1889, 4),
        (1889, 5),
        (1890, 1),
        (1890, 2),
        (1890, 3),
    ]

    spec = importlib.util.spec_from_file_location(
        "characterize_palmer_recursion_divergence", ROOT / "scripts" / "characterize_palmer_recursion_divergence.py"
    )
    assert spec is not None and spec.loader is not None
    script: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    variant: Any = script.build_pdi_variant(("E",))
    factors = DurationFactors.from_defaults()
    harmonized = variant.calculate(
        z.reshape(4, 12, 1), variant.PdiDurationFactors(factors.wetm, factors.wetb, factors.drym, factors.dryb)
    ).pdsi.reshape(-1)[3:43]
    np.testing.assert_allclose(harmonized, _wells_x(z), atol=1e-9)

# scPDSI PR1: Generalize PDSI Recursion to Accept Duration Factors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize `src/climate_indices/palmer.py`'s internal PDSI recursion so its duration-factor constants (currently hardcoded `0.897`/`3.0`/`2.691`/`1.5`) are read from the `data` dict instead, defaulting to Palmer's (1965) original national values so `palmer.pdsi()`'s output is bit-for-bit-equivalent (within existing floating-point tolerance) to before this change.

**Architecture:** Introduce two named module-level constants (`PALMER_DURATION_P`, `PALMER_DURATION_Q`) and two small helpers (`_default_duration_factors()`, `_duration_factor_c()`), thread four new `data` dict keys (`wetm`, `wetb`, `drym`, `dryb`) through `_initialize_data()`, and rewrite the five recursion functions (`_statement_170`, `_statement_180`, `_statement_190`, `_statement_200`, `_statement_210`) to compute their formulas from those keys instead of literals. Each function's specific hardcoded constant is replaced with the algebraically-equivalent expression in terms of `m`/`b`/`c` — verified in the design doc's "Verified consistency" section.

**Tech Stack:** Python, numpy, pytest (existing `climate_indices` toolchain — no new dependencies).

## Global Constraints

- Pure refactor: `palmer.pdsi()`'s public signature and return values are unchanged. No new public API in this PR (that's PR4, issue #720).
- All existing tests must pass unchanged after this refactor, in particular `tests/test_palmer.py::test_pdsi` at its current `ATOL=5e-5`, `RTOL=0` tolerance against the 344-division NOAA fixture set. Note: this test is marked `@pytest.mark.validation` and is excluded by the project's default `addopts = "-m 'not benchmark and not validation'"` — run it explicitly with `-m validation`.
- Do not touch the `0.15`/`-0.15` "backwater" constants in `_statement_170`/`_statement_180` (the `uw`/`ud` computations) — per the design doc, these are a separate empirical Palmer constant unrelated to the duration-factor slope/intercept, and Wells et al. (2004) does not recalibrate them. Out of scope for this refactor.
- Follow `palmer.py`'s existing docstring convention (Sphinx-style `:param:`/`:return:`/`:rtype:`), not the project's general Google-style default — match the file you're editing.
- Run `uv run ruff check --fix src/ tests/` and `uv run ruff format src/ tests/` after each implementation step; run `uv run mypy src/` before the final commit.
- Design doc: `docs/superpowers/specs/2026-08-07-scpdsi-epic-design.md`. Tracking issue: [#717](https://github.com/monocongo/climate_indices/issues/717).

---

## File Structure

- **Modify**: `src/climate_indices/palmer.py` — add constants/helpers near the top (after the existing `AWCTOP`/`K8_SIZE` constants, ~line 16-17), add default duration-factor keys to `_initialize_data()` (~line 792-814 today), rewrite the formulas in `_statement_170`, `_statement_180`, `_statement_190`, `_statement_200`, `_statement_210` (~lines 415-616 today — exact line numbers will shift as earlier tasks edit the file; use the function names, not stale line numbers, to locate edit points after Task 1).
- **Create**: `tests/test_palmer_duration_factors.py` — new focused unit tests exercising the recursion functions directly with custom duration factors, plus one end-to-end integration test proving the parameterization changes `pdsi()`-equivalent output. All tests in this file call `palmer`'s private (`_`-prefixed) functions directly and are companions to the file's existing NOAA-fixture regression suite (`tests/test_palmer.py`), not replacements for it.

## Task Right-Sizing Note

Tasks 2-5 each touch one recursion function and are independently reviewable (a reviewer could accept "statement_180 is correctly parameterized" while still having questions about "statement_200"). Task 6 is the integration proof + full regression run — it can only be written meaningfully once all five functions are generalized, so it closes the PR.

---

### Task 1: Add duration-factor constants, helpers, and default `data` dict keys

**Files:**
- Modify: `src/climate_indices/palmer.py` (constants block near line 16; `_initialize_data()` near line 738-818)
- Test: `tests/test_palmer_duration_factors.py` (new file)

**Interfaces:**
- Produces: `palmer.PALMER_DURATION_P: float`, `palmer.PALMER_DURATION_Q: float`, `palmer._default_duration_factors() -> tuple[float, float]` (returns `(m, b)`), `palmer._duration_factor_c(m: float, b: float) -> float`. `_initialize_data()`'s returned dict gains four new float keys: `wetm`, `wetb`, `drym`, `dryb`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_palmer_duration_factors.py`:

```python
import numpy as np
import pytest

from climate_indices import palmer


def _blank_data() -> dict:
    """A minimal, structurally-valid `data` dict for exercising the recursion
    functions directly, without needing realistic precip/PET content."""
    return palmer._initialize_data(
        precips=np.zeros(12),
        pet=np.zeros(12),
        awc=1.0,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2000,
    )


def test_initialize_data_sets_default_duration_factors():
    data = _blank_data()

    # standard Palmer PDSI has no wet/dry distinction in its duration factors
    assert data["wetm"] == pytest.approx(data["drym"])
    assert data["wetb"] == pytest.approx(data["dryb"])

    # the implied CAFEC-weighting fraction c = b / (m + b) must reproduce
    # Palmer's published constant (0.897), regardless of how m/b are derived
    c = data["wetb"] / (data["wetm"] + data["wetb"])
    assert c == pytest.approx(0.897)

    # m + b must reproduce Palmer's published 1/q = 3.0
    assert (data["wetm"] + data["wetb"]) == pytest.approx(3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_palmer_duration_factors.py -v`
Expected: FAIL — `KeyError: 'wetm'` (the key doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

In `src/climate_indices/palmer.py`, add after the existing `AWCTOP`/`K8_SIZE` constants (currently lines 16-17):

```python
# Palmer's (1965) fixed national duration-factor parameters. Standard PDSI
# uses these directly; self-calibrating PDSI (scPDSI) fits per-location
# replacements via the same m/b/c relationship (see palmer's scpdsi()).
PALMER_DURATION_P = 0.897
PALMER_DURATION_Q = 1.0 / 3.0


def _default_duration_factors() -> tuple[float, float]:
    """
    Palmer's (1965) fixed national duration-factor slope and intercept,
    derived from the published p and q constants.

    :return a tuple of (m, b), the duration-factor slope and intercept
    :rtype: tuple[float, float]
    """
    m = (1.0 - PALMER_DURATION_P) / PALMER_DURATION_Q
    b = PALMER_DURATION_P / PALMER_DURATION_Q
    return m, b


def _duration_factor_c(m: float, b: float) -> float:
    """
    The CAFEC-style weighting fraction implied by a pair of duration factors.

    :param m: duration-factor slope
    :param b: duration-factor intercept
    :return the weighting fraction c = b / (m + b)
    :rtype: float
    """
    return b / (m + b)
```

In `_initialize_data()`, immediately before the `_validate_fitting_params(data, fitting_params)` call (currently line 816), add:

```python
    # duration factors: default to Palmer's fixed national values. scPDSI
    # (see palmer.scpdsi()) overrides these four keys with per-location
    # fitted values after calling this function.
    default_m, default_b = _default_duration_factors()
    data["wetm"], data["wetb"] = default_m, default_b
    data["drym"], data["dryb"] = default_m, default_b
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_palmer_duration_factors.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
uv run ruff format src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git add src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git commit -m "refactor: add duration-factor constants and default data keys to palmer.py"
```

---

### Task 2: Parameterize `_statement_180` (drought abatement) by `drym`/`dryb`

**Files:**
- Modify: `src/climate_indices/palmer.py`, function `_statement_180` (currently lines 574-593)
- Test: `tests/test_palmer_duration_factors.py`

**Interfaces:**
- Consumes: `data["drym"]`, `data["dryb"]` (from Task 1).
- Produces: no new names; `_statement_180`'s `data["ze"]` assignment now depends on `data["drym"]`/`data["dryb"]` instead of literals.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_palmer_duration_factors.py`:

```python
def test_statement_180_ze_uses_custom_dry_duration_factors():
    data = _blank_data()
    data["drym"], data["dryb"] = 1.0, 2.0  # non-default, to prove they're used

    data["year"], data["month"] = 0, 0
    data["x3"] = -2.0  # an established drought
    data["v"] = 0.0
    # z chosen so that pv = (z + 0.15) + max(v, 0) > 0, falling into the
    # branch that actually computes ze (rather than short-circuiting to
    # _statement_210 for a fizzled abatement)
    data["z"][0, 0] = 0.5

    palmer._statement_180(data)

    m, b = data["drym"], data["dryb"]
    expected_ze = -b * data["x3"] - 0.5 * (m + b)
    assert data["ze"] == pytest.approx(expected_ze)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_palmer_duration_factors.py::test_statement_180_ze_uses_custom_dry_duration_factors -v`
Expected: FAIL — `data["ze"]` computed from the hardcoded `-2.691 * data["x3"] - 1.5` won't match the custom-factor expectation (`-2.0*(-2.0) - 0.5*3.0 = 2.5` expected vs. `-2.691*(-2.0) - 1.5 = 3.882` actual).

- [ ] **Step 3: Write minimal implementation**

In `_statement_180`, replace:

```python
    data["ze"] = -2.691 * data["x3"] - 1.5
```

with:

```python
    m, b = data["drym"], data["dryb"]
    data["ze"] = -b * data["x3"] - 0.5 * (m + b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_palmer_duration_factors.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
uv run ruff format src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git add src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git commit -m "refactor: parameterize statement_180 ze formula by dry duration factors"
```

---

### Task 3: Parameterize `_statement_170` (wet abatement) by `wetm`/`wetb`

**Files:**
- Modify: `src/climate_indices/palmer.py`, function `_statement_170` (currently lines 596-616)
- Test: `tests/test_palmer_duration_factors.py`

**Interfaces:**
- Consumes: `data["wetm"]`, `data["wetb"]` (from Task 1).
- Produces: no new names; `_statement_170`'s `data["ze"]` assignment now depends on `data["wetm"]`/`data["wetb"]` instead of literals.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_palmer_duration_factors.py`:

```python
def test_statement_170_ze_uses_custom_wet_duration_factors():
    data = _blank_data()
    data["wetm"], data["wetb"] = 3.0, 5.0  # non-default, to prove they're used

    data["year"], data["month"] = 0, 0
    data["x3"] = 2.0  # an established wet spell
    data["v"] = 0.0
    # z chosen so that pv = (z - 0.15) + min(v, 0) < 0, falling into the
    # branch that actually computes ze
    data["z"][0, 0] = -0.5

    palmer._statement_170(data)

    m, b = data["wetm"], data["wetb"]
    expected_ze = -b * data["x3"] + 0.5 * (m + b)
    assert data["ze"] == pytest.approx(expected_ze)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_palmer_duration_factors.py::test_statement_170_ze_uses_custom_wet_duration_factors -v`
Expected: FAIL — hardcoded `-2.691 * data["x3"] + 1.5` gives `-2.691*2.0 + 1.5 = -3.882` vs. expected `-5.0*2.0 + 0.5*8.0 = -6.0`.

- [ ] **Step 3: Write minimal implementation**

In `_statement_170`, replace:

```python
    data["ze"] = -2.691 * data["x3"] + 1.5
```

with:

```python
    m, b = data["wetm"], data["wetb"]
    data["ze"] = -b * data["x3"] + 0.5 * (m + b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_palmer_duration_factors.py -v`
Expected: PASS (all three tests so far)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
uv run ruff format src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git add src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git commit -m "refactor: parameterize statement_170 ze formula by wet duration factors"
```

---

### Task 4: Parameterize `_statement_190` and `_statement_210`'s `px3` formula, selected by sign of `x3`

**Files:**
- Modify: `src/climate_indices/palmer.py`, functions `_statement_190` (currently lines 548-571) and `_statement_210` (currently lines 433-466)
- Test: `tests/test_palmer_duration_factors.py`

**Interfaces:**
- Consumes: `data["wetm"]`, `data["wetb"]`, `data["drym"]`, `data["dryb"]`, `data["x3"]` (all from Task 1 / already present).
- Produces: `palmer._select_duration_factors(data: dict[str, Any]) -> tuple[float, float]` — returns `(wetm, wetb)` if `data["x3"] >= 0` else `(drym, dryb)`. Used by both `_statement_190` and `_statement_210`, and later reusable by `scpdsi()` in PR4.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_palmer_duration_factors.py`:

```python
def test_statement_210_px3_selects_dry_factors_when_x3_negative():
    data = _blank_data()
    data["drym"], data["dryb"] = 1.0, 4.0
    data["wetm"], data["wetb"] = 99.0, 99.0  # deliberately different, must NOT be used

    data["year"], data["month"] = 0, 0
    data["x3"] = -1.5  # established drought -> dry factors expected
    data["z"][0, 0] = 2.0

    palmer._statement_210(data)

    m, b = data["drym"], data["dryb"]
    c = palmer._duration_factor_c(m, b)
    expected_px3 = c * data["x3"] + data["z"][0, 0] / (m + b)
    assert data["px3"][0, 0] == pytest.approx(expected_px3)


def test_statement_190_px3_selects_wet_factors_when_x3_positive():
    data = _blank_data()
    data["wetm"], data["wetb"] = 2.0, 6.0
    data["drym"], data["dryb"] = 99.0, 99.0  # deliberately different, must NOT be used

    data["year"], data["month"] = 0, 0
    data["x3"] = 2.0  # established wet spell -> wet factors expected
    data["pro"] = 0.0  # not 100, so q = ze + v
    data["ze"] = 10.0
    data["v"] = 0.0
    data["pv"] = 1.0  # ppr = (pv / q) * 100 = 10 < 100, falls into the px3 branch
    data["z"][0, 0] = 4.0

    palmer._statement_190(data)

    m, b = data["wetm"], data["wetb"]
    c = palmer._duration_factor_c(m, b)
    expected_px3 = c * data["x3"] + data["z"][0, 0] / (m + b)
    assert data["px3"][0, 0] == pytest.approx(expected_px3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_palmer_duration_factors.py -k statement_210_px3 -v`
Run: `uv run pytest tests/test_palmer_duration_factors.py -k statement_190_px3 -v`
Expected: both FAIL — `_select_duration_factors` doesn't exist yet, and the hardcoded `0.897`/`3.0` formula doesn't match the custom-factor expectations.

- [ ] **Step 3: Write minimal implementation**

Add near `_duration_factor_c` (Task 1's location):

```python
def _select_duration_factors(data: dict[str, Any]) -> tuple[float, float]:
    """
    Select the wet or dry duration factors based on the sign of the
    currently-established spell's severity (X3).

    :param data: dictionary of parameters (intialized in pdsi)
    :return a tuple of (m, b) - the duration-factor slope and intercept
    :rtype: tuple[float, float]
    """
    if data["x3"] >= 0:
        return data["wetm"], data["wetb"]
    return data["drym"], data["dryb"]
```

In `_statement_210`, replace:

```python
    data["px3"][year, month] = 0.897 * data["x3"] + data["z"][year, month] / 3.0
```

with:

```python
    m, b = _select_duration_factors(data)
    data["px3"][year, month] = _duration_factor_c(m, b) * data["x3"] + data["z"][year, month] / (m + b)
```

In `_statement_190`, replace:

```python
        data["px3"][year, month] = 0.897 * data["x3"] + data["z"][year, month] / 3.0
```

(the one inside the `else` branch of the `if data["ppr"][year, month] >= 100:` check) with:

```python
        m, b = _select_duration_factors(data)
        data["px3"][year, month] = _duration_factor_c(m, b) * data["x3"] + data["z"][year, month] / (m + b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_palmer_duration_factors.py -v`
Expected: PASS (all five tests so far)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
uv run ruff format src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git add src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git commit -m "refactor: parameterize statement_190/210 px3 formula with duration-factor selection"
```

---

### Task 5: Parameterize `_statement_200`'s `px1` (always wet) and `px2` (always dry) formulas

**Files:**
- Modify: `src/climate_indices/palmer.py`, function `_statement_200` (currently lines 469-546)
- Test: `tests/test_palmer_duration_factors.py`

**Interfaces:**
- Consumes: `data["wetm"]`, `data["wetb"]`, `data["drym"]`, `data["dryb"]`.
- Produces: no new names.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_palmer_duration_factors.py`:

```python
def test_statement_200_px1_always_uses_wet_factors_px2_always_dry():
    data = _blank_data()
    data["wetm"], data["wetb"] = 1.0, 3.0
    data["drym"], data["dryb"] = 2.0, 2.0

    data["year"], data["month"] = 0, 0
    data["x1"], data["x2"] = 1.0, -1.0
    data["z"][0, 0] = 0.4
    # a nonzero px3 prevents the early-return "new spell begins" branches,
    # so both px1 and px2 get computed and asserted on
    data["px3"][0, 0] = 5.0

    palmer._statement_200(data)

    c_wet = palmer._duration_factor_c(data["wetm"], data["wetb"])
    expected_px1 = max(0.0, c_wet * data["x1"] + data["z"][0, 0] / (data["wetm"] + data["wetb"]))
    assert data["px1"][0, 0] == pytest.approx(expected_px1)

    c_dry = palmer._duration_factor_c(data["drym"], data["dryb"])
    expected_px2 = min(0.0, c_dry * data["x2"] + data["z"][0, 0] / (data["drym"] + data["dryb"]))
    assert data["px2"][0, 0] == pytest.approx(expected_px2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_palmer_duration_factors.py::test_statement_200_px1_always_uses_wet_factors_px2_always_dry -v`
Expected: FAIL — hardcoded formula gives `px1 = max(0, 0.897*1.0 + 0.4/3.0) = 1.030` vs. expected `max(0, 0.75*1.0 + 0.4/4.0) = 0.85`.

- [ ] **Step 3: Write minimal implementation**

In `_statement_200`, replace:

```python
    data["px1"][year, month] = max(0, 0.897 * data["x1"] + data["z"][year, month] / 3.0)
```

with:

```python
    wetm, wetb = data["wetm"], data["wetb"]
    data["px1"][year, month] = max(
        0, _duration_factor_c(wetm, wetb) * data["x1"] + data["z"][year, month] / (wetm + wetb)
    )
```

Replace:

```python
    data["px2"][year, month] = min(0.0, 0.897 * data["x2"] + data["z"][year, month] / 3.0)
```

with:

```python
    drym, dryb = data["drym"], data["dryb"]
    data["px2"][year, month] = min(
        0.0, _duration_factor_c(drym, dryb) * data["x2"] + data["z"][year, month] / (drym + dryb)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_palmer_duration_factors.py -v`
Expected: PASS (all six tests so far)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
uv run ruff format src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git add src/climate_indices/palmer.py tests/test_palmer_duration_factors.py
git commit -m "refactor: parameterize statement_200 px1/px2 formulas by wet/dry duration factors"
```

---

### Task 6: Integration proof + full regression

**Files:**
- Test: `tests/test_palmer_duration_factors.py`
- Verify (no modification expected): `tests/test_palmer.py`, `tests/test_main_palmers.py`, full suite

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: nothing new — this task is verification only.

This task proves two things: (1) the duration-factor parameterization actually changes `pdsi()`-equivalent output end-to-end (not just in the five functions tested in isolation), and (2) nothing regressed for the default (standard PDSI) case.

- [ ] **Step 1: Write the integration test**

Add to `tests/test_palmer_duration_factors.py`:

```python
def _run_zindex_pipeline(precips: np.ndarray, pet: np.ndarray, awc: float) -> dict:
    """Runs the same internal pipeline palmer.pdsi() runs, up through
    _calc_kfactors (but not yet _calc_zindex), and returns the data dict.
    Exists so this test can inject custom duration factors between
    initialization and the recursion, without touching palmer.pdsi()'s
    public signature."""
    data = palmer._initialize_data(
        precips=precips,
        pet=pet,
        awc=awc,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2003,
    )
    palmer._calc_water_balances(data)
    palmer._calc_cafec_coefficients(data)
    palmer._calc_zindex_factors(data)
    palmer._calc_kfactors(data)
    return data


def test_custom_duration_factors_change_pdsi_output():
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)

    data_default = _run_zindex_pipeline(precips, pet, awc=5.0)
    palmer._calc_zindex(data_default)
    palmer._finish_up(data_default)

    data_custom = _run_zindex_pipeline(precips, pet, awc=5.0)
    data_custom["wetm"], data_custom["wetb"] = 1.0, 1.0
    data_custom["drym"], data_custom["dryb"] = 1.0, 1.0
    palmer._calc_zindex(data_custom)
    palmer._finish_up(data_custom)

    assert not np.allclose(data_default["pdsi"], data_custom["pdsi"], equal_nan=True)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_palmer_duration_factors.py -v`
Expected: PASS (all seven tests). If it fails, the most likely cause is that one of Tasks 2-5 left a formula still reading the hardcoded literals instead of the `data` dict keys — re-check each `_statement_*` function against Task 1-5's replacements.

- [ ] **Step 3: Run the full existing regression suite**

Run: `uv run pytest tests/test_palmer.py -m validation -v`
Expected: PASS — `test_pdsi`'s NOAA-fixture comparisons must still pass at `ATOL=5e-5`, `RTOL=0` for all 344 climate divisions. (This test is excluded by the default addopts; the `-m validation` flag is required.)

Run: `uv run pytest tests/test_main_palmers.py -v`
Expected: PASS.

Run: `uv run pytest`
Expected: full default suite passes (1061 previously-passing tests, plus the 7 new ones from this file — new total 1068).

Run: `uv run ruff check src/ tests/`
Run: `uv run ruff format --check src/ tests/`
Run: `uv run mypy src/`
Expected: all clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_palmer_duration_factors.py
git commit -m "test: add integration test proving duration-factor parameterization affects pdsi output"
```

- [ ] **Step 5: Update tracking issue**

Comment on [#717](https://github.com/monocongo/climate_indices/issues/717) confirming the refactor is complete and all acceptance criteria are met, then close it. Check the corresponding box in the epic tracking issue [#716](https://github.com/monocongo/climate_indices/issues/716).

```bash
gh issue close 717 --comment "Done: statement_170/180/190/200/210 now read wetm/wetb/drym/dryb from the data dict (defaulting to Palmer's original 0.897/(1/3)-derived values). All existing tests pass unchanged at the existing ATOL=5e-5 tolerance; 7 new unit/integration tests in tests/test_palmer_duration_factors.py verify the parameterization is real and correctly wired. Ready for PR4 (#720) to feed self-calibrated duration factors through this same recursion."
```

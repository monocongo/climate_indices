# scPDSI PR2: Self-Calibration Statistics Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new pure-functions module `src/climate_indices/self_calibration.py` implementing the three statistics pieces of the Wells, Goddard & Wilhite (2004) scPDSI self-calibration procedure — the rolling-window extreme Z-index sum (with its wet/dry asymmetric outlier filter), the correlation-adaptive least-squares duration-factor fit, and the reference implementation's exact kth-smallest/percentile order statistic — plus the small composition function that turns them into a `(slope, intercept)` duration-factor pair.

**Architecture:** One new flat module alongside `palmer.py`, containing only pure functions over numpy arrays. No `data`-dict machinery, no calibration-period bookkeeping (callers slice the Z series to the calibration window before calling), no xarray, no CLI. The import edge is one-directional: `palmer.py` will import `self_calibration` in PR4 (#720); `self_calibration` must never import `palmer`.

**Tech Stack:** Python, numpy, pytest (existing `climate_indices` toolchain — no new dependencies).

## Global Constraints

- **Numeric parity with the reference is the point of this module.** Every function is a faithful port of the reference C++ (`Sibada/scPDSI`, GPLv3, algorithm reference only — no GPL source, headers, or derivative code enters this repository). Where the reference does something that looks like a bug or a quirk, port it faithfully and document it in a comment. Do not "fix" it. Specific quirks that MUST be preserved are called out inline in each task.
- **Floating-point operation order matters.** PR4 (#720) will compare this pipeline's output against a C++ oracle at `ATOL=5e-5`. Where a task specifies an accumulation order (e.g. `running -= old` then `running += new`, rather than `running += new - old`), match it exactly.
- **Docstrings: Google style** (summary, `Args:`, `Returns:`, `Raises:`) per `.claude/rules/architecture.md`. This deliberately differs from `palmer.py`'s legacy Sphinx `:param:` style — this is a new module, so it follows the project's stated convention rather than the neighbouring file's legacy one.
- **Type hints required on every function**, using `X | None` / `tuple[float, float]` / `np.ndarray` (Python 3.10+ syntax).
- **Never raise bare `ValueError`.** Use `climate_indices.exceptions.InvalidArgumentError`, whose signature is `InvalidArgumentError(message, *, argument_name=None, argument_value=None, valid_values=None)` — note the *positional* message parameter, which the example in `.claude/rules/architecture.md` omits.
- **Never import stdlib `logging`.** This module needs no logging at all (pure math, no lifecycle) — do not add a logger.
- **Never compare computed floats with `==`.** Tests use `pytest.approx` / `np.testing.assert_allclose`; `np.isnan()` for NaN assertions.
- Do not modify `src/climate_indices/palmer.py` in this PR. Do not add xarray wrappers, `typed_public_api.py` overloads, or `cf_metadata_registry.py` entries — those belong to PR4/PR5 (#720/#721), and `pdsi()` itself has none yet.
- Run `uv run ruff check --fix src/ tests/` and `uv run ruff format src/ tests/` after each implementation step; run `uv run mypy src/` before the final commit.
- Design doc: `docs/superpowers/specs/2026-08-07-scpdsi-epic-design.md` (§ "Reference algorithm" step 4 and step 6). Tracking issue: [#718](https://github.com/monocongo/climate_indices/issues/718). Epic: [#716](https://github.com/monocongo/climate_indices/issues/716).

---

## File Structure

- **Create**: `src/climate_indices/self_calibration.py` — the whole module. Public surface (`__all__`): `DRY_SIGN`, `DURATION_FACTOR_WINDOW_LENGTHS`, `WET_SIGN`, `duration_factors`, `extreme_z_sum`, `kth_smallest`, `least_squares_fit`, `nan_safe_percentile`. Private helpers: `_validate_sign`, `_highest_reasonable`. Private constants: `_CORRELATION_TOLERANCE`, `_EXTREME_PERCENTILE`, `_MIN_REGRESSION_POINTS`, `_PDSI_ANCHOR`, `_REASONABLE_TOLERANCE`.
- **Create**: `tests/test_self_calibration.py` — all unit tests, grouped in `Test*` classes per public function, matching the `class TestSPICalculation:` convention in `.claude/rules/testing.md`. All expected values in this plan are hand-derived from the reference algorithm and independently confirmed numerically; use them verbatim.

**Module name rationale** (the open decision the design doc deferred to this plan): `self_calibration.py`, following the flat `src/climate_indices/*.py` convention already in use (`eto.py`, `lmoments.py`, `pm_eto.py`). "Self-calibration" is unambiguous in this domain — the SPI/SPEI code uses "calibration period", never "self-calibration" — and the module docstring pins it to Palmer/scPDSI explicitly.

## Task Right-Sizing Note

Tasks 1–4 each deliver one independently reviewable unit of the port: a reviewer can accept "the order statistic matches the reference's truncating 1-indexed semantics" while still questioning "the freak-anomaly filter". Task 4 also carries the composition, because the `/(sign×4)` normalization is only meaningful once the pieces it composes exist. Task 5 is verification and closeout — the architectural import-direction guard plus the full suite.

---

### Task 1: Module scaffold and the reference order statistic

**Files:**
- Create: `src/climate_indices/self_calibration.py`
- Test: `tests/test_self_calibration.py` (new file)

**Interfaces:**
- Produces: `self_calibration.WET_SIGN: int` (= `1`), `self_calibration.DRY_SIGN: int` (= `-1`), `self_calibration.kth_smallest(values: np.ndarray, k: int) -> float`, `self_calibration.nan_safe_percentile(values: np.ndarray, fraction: float) -> float`, `self_calibration._validate_sign(sign: int) -> None`.

**Reference semantics being ported** (read before writing code): the reference's `llist::kthLargest(k)` is misleadingly named — it sorts ascending and returns `A[k-1]`, so it is the **k-th smallest, 1-indexed**, and it returns the MISSING sentinel when `k < 1` or `k > size`. `llist::percentile(p)` computes `k = (int)(p * size)` — integer **truncation**, not rounding, and not numpy's interpolating `np.percentile`. `llist::safe_percentile(p)` first drops MISSING values, then calls `percentile` on what remains. We represent MISSING as `NaN` throughout.

- [ ] **Step 1: Write the failing test**

Create `tests/test_self_calibration.py`:

```python
"""Unit tests for the scPDSI self-calibration statistics module.

Expected values are hand-derived from the Wells, Goddard & Wilhite (2004)
reference algorithm as implemented in the reference C++ (used for algorithm
reference only). Each test's comment shows the derivation.
"""

import numpy as np
import pytest

from climate_indices import self_calibration
from climate_indices.exceptions import InvalidArgumentError


class TestKthSmallest:
    def test_returns_kth_smallest_one_indexed(self):
        values = np.array([5.0, 1.0, 4.0, 2.0, 3.0])

        assert self_calibration.kth_smallest(values, 1) == pytest.approx(1.0)
        assert self_calibration.kth_smallest(values, 3) == pytest.approx(3.0)
        assert self_calibration.kth_smallest(values, 5) == pytest.approx(5.0)

    def test_returns_nan_for_out_of_range_k(self):
        # the reference returns its MISSING sentinel for k < 1 or k > size;
        # we return NaN
        values = np.array([5.0, 1.0, 4.0, 2.0, 3.0])

        assert np.isnan(self_calibration.kth_smallest(values, 0))
        assert np.isnan(self_calibration.kth_smallest(values, 6))

    def test_does_not_mutate_the_caller_s_array(self):
        values = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        original = values.copy()

        self_calibration.kth_smallest(values, 2)

        np.testing.assert_array_equal(values, original)


class TestNanSafePercentile:
    def test_lower_and_upper_percentiles_over_fifty_values(self):
        values = np.arange(1.0, 51.0)

        # k = int(0.02 * 50) = 1 -> smallest value
        assert self_calibration.nan_safe_percentile(values, 0.02) == pytest.approx(1.0)
        # k = int(0.98 * 50) = 49 -> 49th smallest
        assert self_calibration.nan_safe_percentile(values, 0.98) == pytest.approx(49.0)

    def test_index_is_truncated_not_rounded(self):
        values = np.arange(1.0, 11.0)

        # k = int(0.25 * 10) = int(2.5) = 2, so the 2nd smallest -- not the 3rd,
        # and not numpy.percentile's interpolated 3.25
        assert self_calibration.nan_safe_percentile(values, 0.25) == pytest.approx(2.0)

    def test_missing_values_are_dropped_before_ranking(self):
        values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])

        # 5 values survive; k = int(0.6 * 5) = 3 -> 3rd smallest of [1, 2, 3, 4, 5]
        assert self_calibration.nan_safe_percentile(values, 0.6) == pytest.approx(3.0)

    def test_returns_nan_when_every_value_is_missing(self):
        assert np.isnan(self_calibration.nan_safe_percentile(np.array([np.nan, np.nan]), 0.5))

    def test_returns_nan_when_the_computed_index_truncates_to_zero(self):
        values = np.arange(1.0, 6.0)

        # k = int(0.02 * 5) = 0, below the 1-indexed range
        assert np.isnan(self_calibration.nan_safe_percentile(values, 0.02))

    @pytest.mark.parametrize("fraction", [-0.1, 1.5])
    def test_rejects_a_fraction_outside_the_unit_interval(self, fraction):
        with pytest.raises(InvalidArgumentError):
            self_calibration.nan_safe_percentile(np.arange(1.0, 11.0), fraction)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_self_calibration.py -v`
Expected: FAIL — `ImportError: cannot import name 'self_calibration' from 'climate_indices'` (the module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/climate_indices/self_calibration.py`:

```python
"""Statistics for the self-calibrating Palmer Drought Severity Index (scPDSI).

Pure functions implementing the Wells, Goddard & Wilhite (2004) self-calibration
procedure's statistical building blocks: the rolling-window extreme Z-index sum,
the correlation-adaptive least-squares duration-factor fit, and the order
statistic the reference implementation uses for percentile rescaling.

Every function here is a faithful port of the reference algorithm, including
behaviours that differ from what an idiomatic numpy implementation would do
(truncating 1-indexed order statistics rather than interpolating percentiles,
an asymmetric outlier filter applied to the wet side only, and a regression
intercept anchored on the most extreme residual rather than on the mean). Those
divergences are deliberate and are noted at each site; changing them would break
numeric parity with the published algorithm.

These functions are independent of the ``data``-dict machinery in
``climate_indices.palmer`` and take no calibration-period arguments -- callers
slice the Z-index series to the calibration window before calling.
"""

import numpy as np

from climate_indices.exceptions import InvalidArgumentError

__all__ = [
    "DRY_SIGN",
    "WET_SIGN",
    "kth_smallest",
    "nan_safe_percentile",
]

# Sign conventions used throughout: the self-calibration procedure fits wet and
# dry duration factors separately, and several formulas are sign-weighted so
# that "more extreme" means larger for wet spells and smaller for dry ones.
WET_SIGN = 1
DRY_SIGN = -1


def _validate_sign(sign: int) -> None:
    """Reject a spell sign that is neither wet nor dry.

    Args:
        sign: The spell sign to validate.

    Raises:
        InvalidArgumentError: If sign is not WET_SIGN or DRY_SIGN.
    """
    if sign not in (WET_SIGN, DRY_SIGN):
        raise InvalidArgumentError(
            f"invalid spell sign: {sign}",
            argument_name="sign",
            argument_value=str(sign),
            valid_values=f"{WET_SIGN} (wet) or {DRY_SIGN} (dry)",
        )


def kth_smallest(values: np.ndarray, k: int) -> float:
    """Select the kth smallest value, 1-indexed.

    Ports the reference implementation's ``kthLargest()``, which despite its
    name selects in ascending order (k=1 is the minimum, k=size is the maximum)
    and returns its missing-value sentinel for an out-of-range k. No missing
    value handling is done here -- see :func:`nan_safe_percentile` for that.

    Args:
        values: The values to select from.
        k: The 1-indexed rank to select.

    Returns:
        The kth smallest value, or NaN if k is outside [1, len(values)].
    """
    array = np.asarray(values, dtype=float)
    if k < 1 or k > array.size:
        return float("nan")
    # np.partition returns a new array, so the caller's array is left untouched
    return float(np.partition(array, k - 1)[k - 1])


def nan_safe_percentile(values: np.ndarray, fraction: float) -> float:
    """Select the percentile order statistic the reference implementation uses.

    Ports ``safe_percentile()``: drop missing values, then take the
    ``int(fraction * count)``th smallest of what remains. The index is
    *truncated*, and the result is an actual data value -- this is deliberately
    not ``numpy.percentile``, which interpolates between neighbouring values and
    would shift the self-calibration ratios.

    Args:
        values: The values to rank; NaN entries are treated as missing.
        fraction: The percentile as a fraction within [0.0, 1.0].

    Returns:
        The selected value, or NaN if no value qualifies -- either because every
        input was missing, or because the truncated index came out as 0 (which
        happens when the surviving count is small relative to the fraction).

    Raises:
        InvalidArgumentError: If fraction is outside [0.0, 1.0].
    """
    if not 0.0 <= fraction <= 1.0:
        raise InvalidArgumentError(
            f"percentile fraction outside the unit interval: {fraction}",
            argument_name="fraction",
            argument_value=str(fraction),
            valid_values="a float within [0.0, 1.0]",
        )

    array = np.asarray(values, dtype=float)
    present = array[~np.isnan(array)]
    return kth_smallest(present, int(fraction * present.size))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_self_calibration.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/climate_indices/self_calibration.py tests/test_self_calibration.py
uv run ruff format src/climate_indices/self_calibration.py tests/test_self_calibration.py
git add src/climate_indices/self_calibration.py tests/test_self_calibration.py
git commit -m "feat: add scPDSI self-calibration order statistic"
```

---

### Task 2: Rolling-window extreme Z-index sum with the wet-side freak-anomaly filter

**Files:**
- Modify: `src/climate_indices/self_calibration.py`
- Test: `tests/test_self_calibration.py`

**Interfaces:**
- Consumes: `WET_SIGN`, `DRY_SIGN`, `_validate_sign`, `nan_safe_percentile` (Task 1).
- Produces: `self_calibration.extreme_z_sum(z_values: np.ndarray, window_length: int, sign: int) -> float`, `self_calibration._highest_reasonable(sums: list[float], sign: int) -> float`, and private constants `_EXTREME_PERCENTILE: float` (= `0.98`), `_REASONABLE_TOLERANCE: float` (= `1.25`).

**Reference semantics being ported** (`get_Z_sum()`): walk the Z series forward in time maintaining a rolling window of `window_length` **non-missing** values. Missing periods never enter the window — during the initial fill they are retried (a NaN does not consume a window slot), and during the slide they are skipped entirely (the window and running sum are left unchanged, and no new rolling sum is recorded). Then:

- **Dry side** (`sign == DRY_SIGN`): return the single most negative rolling sum, **unfiltered**.
- **Wet side** (`sign == WET_SIGN`): return the largest rolling sum that is not a "freak anomaly", where a freak anomaly is a sum whose ratio to the 98th-percentile rolling sum reaches 1.25 or more. Only strictly-positive sums are candidates, and the floor is `0.0` — if no sum qualifies, the function returns `0.0`, not NaN.

This wet/dry asymmetry is a real feature of the reference implementation, not a bug to normalize away. The reference also computes a 2nd-percentile threshold on the dry side, but never uses it (dry returns the unfiltered extreme), so we skip that dead computation — this is behaviour-preserving and avoids a spurious division.

Two edge cases the reference reaches via its `-999.0` missing sentinel, which we reach explicitly instead:
- **The percentile is unavailable** (fewer than 2 rolling sums, so the truncated index is 0). The reference divides by `-999.0`, which makes the ratio test pass for every positive sum — i.e. no filtering. Reproduce that outcome: when the threshold is NaN, every candidate passes.
- **The percentile is exactly 0.0.** The reference's division yields infinity, which fails the ratio test, so every candidate is rejected and the function returns `0.0`. Reproduce that outcome explicitly rather than dividing by zero.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_self_calibration.py`:

```python
class TestExtremeZSum:
    def test_dry_side_returns_the_most_negative_rolling_sum(self):
        z = np.array([-1.0, -5.0, -2.0, 0.0, -3.0])

        # rolling 2-period sums: -6, -7, -2, -3 -> most negative is -7
        result = self_calibration.extreme_z_sum(z, 2, self_calibration.DRY_SIGN)

        assert result == pytest.approx(-7.0)

    def test_wet_side_discards_a_freak_anomaly(self):
        z = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 97.0])

        # rolling 2-period sums: 2, 3, 4, 5, 100.
        # threshold = 98th percentile = kth smallest with k = int(0.98 * 5) = 4,
        # i.e. 5.0. The 100 sum fails the filter (100 / 5 = 20.0 >= 1.25) and is
        # discarded; 5.0 passes (5 / 5 = 1.0) and is the largest survivor.
        result = self_calibration.extreme_z_sum(z, 2, self_calibration.WET_SIGN)

        assert result == pytest.approx(5.0)

    def test_dry_side_keeps_the_anomaly_the_wet_side_would_discard(self):
        # exact mirror image of the wet case above: rolling sums -2, -3, -4, -5, -100
        z = np.array([-1.0, -1.0, -2.0, -2.0, -3.0, -97.0])

        # the dry side is deliberately unfiltered in the reference algorithm, so
        # it returns the -100 outlier rather than the -5 the wet side would keep
        result = self_calibration.extreme_z_sum(z, 2, self_calibration.DRY_SIGN)

        assert result == pytest.approx(-100.0)

    def test_missing_periods_are_skipped_rather_than_treated_as_zero(self):
        with_missing = np.array([1.0, 1.0, np.nan, 2.0, 2.0, 3.0, 97.0])
        without_missing = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 97.0])

        result = self_calibration.extreme_z_sum(with_missing, 2, self_calibration.WET_SIGN)

        # a NaN mid-slide leaves the window and running sum untouched, so the
        # result is identical to the series with the NaN simply removed
        assert result == pytest.approx(
            self_calibration.extreme_z_sum(without_missing, 2, self_calibration.WET_SIGN)
        )
        assert result == pytest.approx(5.0)

    def test_missing_periods_during_the_initial_fill_are_retried(self):
        z = np.array([1.0, np.nan, 1.0, 2.0, 2.0, 3.0, 97.0])

        # the NaN must not consume one of the window's two slots, so the first
        # window is [1.0, 1.0] and the rolling sums are 2, 3, 4, 5, 100 again
        result = self_calibration.extreme_z_sum(z, 2, self_calibration.WET_SIGN)

        assert result == pytest.approx(5.0)

    def test_rejects_an_invalid_sign(self):
        with pytest.raises(InvalidArgumentError):
            self_calibration.extreme_z_sum(np.arange(10.0), 3, 0)

    def test_rejects_a_nonpositive_window_length(self):
        with pytest.raises(InvalidArgumentError):
            self_calibration.extreme_z_sum(np.arange(10.0), 0, self_calibration.WET_SIGN)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_self_calibration.py -k ExtremeZSum -v`
Expected: FAIL — `AttributeError: module 'climate_indices.self_calibration' has no attribute 'extreme_z_sum'`.

- [ ] **Step 3: Write minimal implementation**

In `src/climate_indices/self_calibration.py`, add `import math` above the `import numpy as np` line, add `"extreme_z_sum"` to `__all__` (keep it alphabetically sorted), add these constants below `DRY_SIGN`:

```python
# A wet-side rolling sum is treated as a "freak anomaly" -- and excluded from
# the duration-factor fit -- once its ratio to the 98th-percentile rolling sum
# reaches this tolerance.
_EXTREME_PERCENTILE = 0.98
_REASONABLE_TOLERANCE = 1.25
```

and append these functions:

```python
def _highest_reasonable(sums: list[float], sign: int) -> float:
    """Select the most extreme rolling sum that is not a freak anomaly.

    Applies the reference implementation's outlier filter: among the sums whose
    own sign matches the spell being fitted, keep only those whose ratio to the
    98th-percentile sum stays below the reasonableness tolerance, and return the
    most extreme survivor.

    Args:
        sums: The rolling sums to filter.
        sign: WET_SIGN or DRY_SIGN, selecting which direction counts as extreme.

    Returns:
        The most extreme surviving sum, or 0.0 if nothing survives.
    """
    threshold = nan_safe_percentile(np.array(sums, dtype=float), _EXTREME_PERCENTILE)

    highest = 0.0
    for value in sums:
        if sign * value <= 0.0:
            continue
        if math.isnan(threshold):
            # Too few sums for a percentile to exist. The reference divides by
            # its missing-value sentinel here, which lets every candidate
            # through, so the filter effectively does not apply.
            is_reasonable = True
        elif threshold == 0.0:
            # The reference's division yields infinity, failing the ratio test
            # for every candidate.
            is_reasonable = False
        else:
            is_reasonable = (value / threshold) < _REASONABLE_TOLERANCE
        if is_reasonable and sign * value > sign * highest:
            highest = value
    return highest


def extreme_z_sum(z_values: np.ndarray, window_length: int, sign: int) -> float:
    """Compute the representative extreme rolling Z-index sum for one window length.

    Ports ``get_Z_sum()``. Slides a window of ``window_length`` non-missing
    Z-index values across the series and picks one representative extreme sum,
    asymmetrically by spell sign: the dry side takes the single most negative
    rolling sum unfiltered, while the wet side discards freak anomalies first
    (see :func:`_highest_reasonable`). The asymmetry is a property of the
    published algorithm, not an oversight.

    Missing periods never enter the window: during the initial fill they are
    retried rather than consuming a slot, and during the slide they leave the
    window and the running sum untouched.

    Callers are responsible for restricting ``z_values`` to the calibration
    period before calling.

    Args:
        z_values: The Z-index series, in chronological order; NaN means missing.
        window_length: The number of non-missing periods in the rolling window.
        sign: WET_SIGN or DRY_SIGN.

    Returns:
        The representative extreme rolling sum for this window length. On the
        wet side this is 0.0 when no rolling sum survives the anomaly filter.

    Raises:
        InvalidArgumentError: If sign is invalid or window_length is not positive.
    """
    _validate_sign(sign)
    if window_length < 1:
        raise InvalidArgumentError(
            f"invalid rolling window length: {window_length}",
            argument_name="window_length",
            argument_value=str(window_length),
            valid_values="a positive integer",
        )

    series = np.asarray(z_values, dtype=float)
    window: list[float] = []
    running = 0.0
    index = 0

    # fill the initial window, retrying past missing periods
    while len(window) < window_length and index < series.size:
        value = float(series[index])
        index += 1
        if not math.isnan(value):
            running += value
            window.append(value)

    extreme = running
    sums = [running]

    # slide the window forward one non-missing period at a time. The subtract-
    # then-add ordering below mirrors the reference's accumulation order and is
    # deliberate: reassociating it as `running += value - window.pop(0)` changes
    # the rounding, which PR4 compares against a C++ oracle at ATOL=5e-5.
    while index < series.size:
        value = float(series[index])
        index += 1
        if not math.isnan(value):
            running -= window.pop(0)
            running += value
            window.append(value)
            sums.append(running)
        if sign * running > sign * extreme:
            extreme = running

    if sign == DRY_SIGN:
        return extreme
    return _highest_reasonable(sums, sign)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_self_calibration.py -v`
Expected: PASS (17 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/climate_indices/self_calibration.py tests/test_self_calibration.py
uv run ruff format src/climate_indices/self_calibration.py tests/test_self_calibration.py
git add src/climate_indices/self_calibration.py tests/test_self_calibration.py
git commit -m "feat: add rolling-window extreme Z-index sum with wet-side anomaly filter"
```

---

### Task 3: Correlation-adaptive least-squares fit

**Files:**
- Modify: `src/climate_indices/self_calibration.py`
- Test: `tests/test_self_calibration.py`

**Interfaces:**
- Consumes: `WET_SIGN`, `DRY_SIGN`, `_validate_sign` (Task 1).
- Produces: `self_calibration.least_squares_fit(x: np.ndarray, y: np.ndarray, sign: int) -> tuple[float, float]` returning `(slope, intercept)`, and private constants `_CORRELATION_TOLERANCE: float` (= `0.85`), `_MIN_REGRESSION_POINTS: int` (= `4`).

**Reference semantics being ported** (`LeastSquares()`): ordinary least squares, then two non-standard steps.

1. **Adaptive trimming.** If the sign-weighted correlation coefficient is below 0.85, drop the *last* point and refit, repeating until the correlation clears the tolerance or only 4 points remain. The reference's comment explains why it drops from the end: "when the correlation is off, it appears better to take the earlier sums rather than the later ones."
2. **Intercept anchoring.** The intercept is *not* `ybar - slope * xbar`. Instead the fitted line is translated to pass through the retained point with the most extreme sign-weighted residual `y[i] - slope * x[i]`, giving `intercept = y[max_i] - slope * x[max_i]`.

The anchoring search starts from `max_residual = 0.0`, `anchor_y = 0.0`, `anchor_x = x[0]`, and only updates on a **strictly greater** sign-weighted residual. Two consequences must be preserved:
- Ties go to the earliest point (so a set of equal residuals anchors on index 0).
- **Deliberately not guarded:** if the retained `y` values are all identical, `ss_y` is 0 (or, because the accumulation is sequential rather than via a pairwise-summation reduction, very slightly negative) and the correlation divides by zero. The reference lets this produce a non-finite correlation and carries on; in Python the constant-`y` case raises `ZeroDivisionError` only when the constant is exactly `0.0` — for any other constant, `ss_y`'s slight negativity means `math.sqrt(ss_y)` raises `ValueError: math domain error` before the division is ever reached. Leave both unguarded here — the degenerate-input guards for this pipeline are tracked on [#720](https://github.com/monocongo/climate_indices/issues/720) (alongside the `m + b == 0` guard PR1's review surfaced), where real fitted data actually flows through and the right behaviour can be decided against the oracle rather than invented, and #720's guard must account for both exception types.
- If no retained point has a sign-weighted residual strictly greater than 0 — which happens when the retained points are perfectly collinear through the origin — the anchor stays at the initial `(x[0], 0.0)`, producing `intercept = -slope * x[0]` rather than 0. This looks wrong and is faithful; do not "fix" it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_self_calibration.py`:

```python
class TestLeastSquaresFit:
    def test_recovers_an_exact_dry_line_without_trimming(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([-11.0, -12.0, -13.0, -14.0, -15.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.DRY_SIGN)

        # correlation is -1, so sign * correlation = 1 >= 0.85 and nothing is
        # trimmed. SSXY / SSX = -10 / 10 = -1. Every residual is -10, ties go to
        # the earliest point, so intercept = y[0] - slope * x[0] = -11 + 1 = -10.
        assert slope == pytest.approx(-1.0)
        assert intercept == pytest.approx(-10.0)

    def test_recovers_an_exact_wet_line_without_trimming(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([11.0, 12.0, 13.0, 14.0, 15.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.WET_SIGN)

        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(10.0)

    def test_intercept_is_anchored_on_the_most_extreme_residual(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([11.0, 12.0, 15.0, 14.0, 15.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.WET_SIGN)

        # SSX = 10, SSXY = 10 -> slope = 1.0. Correlation is 0.8704 >= 0.85, so
        # no trimming. Residuals y - slope*x are 10, 10, 12, 10, 10; index 2 is
        # the most extreme, so intercept = y[2] - slope*x[2] = 15 - 3 = 12.
        # A textbook OLS intercept would be ybar - slope*xbar = 13.4 - 3 = 10.4.
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(12.0)

    def test_trims_trailing_points_until_the_correlation_clears_the_tolerance(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([-11.0, -12.0, -13.0, -14.0, -25.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.DRY_SIGN)

        # Over all five points the correlation is -0.8321, so sign*correlation
        # = 0.8321 < 0.85 and the last point is dropped. The remaining four are
        # perfectly collinear (correlation -1), giving slope = -5/5 = -1 -- not
        # the -3 the untrimmed five-point fit would have produced. Residuals are
        # all -10, so intercept = y[0] - slope*x[0] = -11 + 1 = -10.
        assert slope == pytest.approx(-1.0)
        assert intercept == pytest.approx(-10.0)

    def test_trimming_stops_at_four_retained_points(self):
        x = np.arange(1.0, 11.0)
        y = np.array([-11.0, -12.0, -13.0, -14.0, 50.0, -60.0, 70.0, -80.0, 90.0, -100.0])

        slope, intercept = self_calibration.least_squares_fit(x, y, self_calibration.DRY_SIGN)

        # The trailing six points are pure noise, so the correlation never
        # clears the tolerance and trimming runs to its floor. The four points
        # that survive are the collinear leading ones, giving exactly the fit
        # they define on their own.
        assert slope == pytest.approx(-1.0)
        assert intercept == pytest.approx(-10.0)

    def test_rejects_an_invalid_sign(self):
        with pytest.raises(InvalidArgumentError):
            self_calibration.least_squares_fit(np.arange(5.0), np.arange(5.0), 0)

    def test_rejects_mismatched_input_lengths(self):
        with pytest.raises(InvalidArgumentError):
            self_calibration.least_squares_fit(
                np.arange(5.0), np.arange(4.0), self_calibration.WET_SIGN
            )

    def test_rejects_fewer_points_than_the_trimming_floor(self):
        with pytest.raises(InvalidArgumentError):
            self_calibration.least_squares_fit(
                np.arange(3.0), np.arange(3.0), self_calibration.WET_SIGN
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_self_calibration.py -k LeastSquaresFit -v`
Expected: FAIL — `AttributeError: module 'climate_indices.self_calibration' has no attribute 'least_squares_fit'`.

- [ ] **Step 3: Write minimal implementation**

In `src/climate_indices/self_calibration.py`, add `"least_squares_fit"` to `__all__` (keep it sorted), add these constants alongside the Task 2 constants:

```python
# The duration-factor regression drops trailing points until the sign-weighted
# correlation clears this tolerance, down to a floor of four retained points.
_CORRELATION_TOLERANCE = 0.85
_MIN_REGRESSION_POINTS = 4
```

and append:

```python
def least_squares_fit(x: np.ndarray, y: np.ndarray, sign: int) -> tuple[float, float]:
    """Fit a line by the reference implementation's adaptive least squares.

    Ports ``LeastSquares()``, which differs from textbook OLS in two ways. It
    trims trailing points until the sign-weighted correlation clears 0.85 (down
    to a floor of four retained points), and it anchors the intercept on the
    retained point with the most extreme sign-weighted residual rather than on
    the sample means. Both are deliberate properties of the published
    self-calibration procedure.

    Args:
        x: The independent variable (window lengths, in the duration-factor fit).
        y: The dependent variable (extreme Z-index sums), same length as x.
        sign: WET_SIGN or DRY_SIGN, selecting the direction of "most extreme".

    Returns:
        A tuple of (slope, intercept) for the fitted line.

    Raises:
        InvalidArgumentError: If sign is invalid, the inputs differ in length,
            or fewer than four points were supplied.
    """
    _validate_sign(sign)

    abscissa = np.asarray(x, dtype=float)
    ordinate = np.asarray(y, dtype=float)
    if abscissa.size != ordinate.size:
        raise InvalidArgumentError(
            f"mismatched input lengths: {abscissa.size} and {ordinate.size}",
            argument_name="y",
            argument_value=str(ordinate.size),
            valid_values=f"an array of length {abscissa.size}, matching x",
        )
    if abscissa.size < _MIN_REGRESSION_POINTS:
        raise InvalidArgumentError(
            f"too few points to fit: {abscissa.size}",
            argument_name="x",
            argument_value=str(abscissa.size),
            valid_values=f"at least {_MIN_REGRESSION_POINTS} points",
        )

    # accumulate sequentially rather than via ndarray.sum(), whose pairwise
    # summation would reassociate the additions and perturb the low-order bits
    count = abscissa.size
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    for index in range(count):
        this_x = float(abscissa[index])
        this_y = float(ordinate[index])
        sum_x += this_x
        sum_y += this_y
        sum_x2 += this_x * this_x
        sum_y2 += this_y * this_y
        sum_xy += this_x * this_y

    ss_x = sum_x2 - (sum_x * sum_x) / count
    ss_y = sum_y2 - (sum_y * sum_y) / count
    ss_xy = sum_xy - (sum_x * sum_y) / count
    correlation = ss_xy / (math.sqrt(ss_x) * math.sqrt(ss_y))

    # drop points from the end until the fit correlates well enough, or until
    # only the minimum number of points is left
    last = count - 1
    while sign * correlation < _CORRELATION_TOLERANCE and last > _MIN_REGRESSION_POINTS - 1:
        this_x = float(abscissa[last])
        this_y = float(ordinate[last])
        sum_x -= this_x
        sum_y -= this_y
        sum_x2 -= this_x * this_x
        sum_y2 -= this_y * this_y
        sum_xy -= this_x * this_y

        ss_x = sum_x2 - (sum_x * sum_x) / last
        ss_y = sum_y2 - (sum_y * sum_y) / last
        ss_xy = sum_xy - (sum_x * sum_y) / last
        correlation = ss_xy / (math.sqrt(ss_x) * math.sqrt(ss_y))
        last -= 1

    slope = ss_xy / ss_x

    # translate the line through the retained point whose sign-weighted residual
    # is most extreme. The initial anchor of (x[0], 0.0) is the reference's, and
    # survives whenever no residual is strictly more extreme than zero -- which
    # happens for retained points that are collinear through the origin.
    retained = last + 1
    max_residual = 0.0
    anchor_x = float(abscissa[0])
    anchor_y = 0.0
    for index in range(retained):
        residual = float(ordinate[index]) - slope * float(abscissa[index])
        if sign * residual > sign * max_residual:
            max_residual = residual
            anchor_x = float(abscissa[index])
            anchor_y = float(ordinate[index])

    return float(slope), float(anchor_y - slope * anchor_x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_self_calibration.py -v`
Expected: PASS (25 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/climate_indices/self_calibration.py tests/test_self_calibration.py
uv run ruff format src/climate_indices/self_calibration.py tests/test_self_calibration.py
git add src/climate_indices/self_calibration.py tests/test_self_calibration.py
git commit -m "feat: add correlation-adaptive least-squares fit for duration factors"
```

---

### Task 4: Duration-factor composition and PDSI-anchor normalization

**Files:**
- Modify: `src/climate_indices/self_calibration.py`
- Test: `tests/test_self_calibration.py`

**Interfaces:**
- Consumes: `extreme_z_sum` (Task 2), `least_squares_fit` (Task 3), `_validate_sign` (Task 1).
- Produces: `self_calibration.duration_factors(z_values: np.ndarray, sign: int) -> tuple[float, float]` returning `(m, b)` — the duration-factor slope and intercept in exactly the form `palmer.py`'s `data["wetm"]`/`data["wetb"]`/`data["drym"]`/`data["dryb"]` keys expect (see PR1, #717) — plus `self_calibration.DURATION_FACTOR_WINDOW_LENGTHS: tuple[int, ...]` and the private constant `_PDSI_ANCHOR: float` (= `4.0`).

**Reference semantics being ported** (`CalcDurFact()`): compute one extreme Z sum per window length from the monthly set (3, 6, 9, 12, 18, 24, 30, 36, 42, 48 months), fit the line through those ten points, then divide both slope and intercept by `sign * 4`. The division is what turns a raw regression into duration factors: the fitted line represents a PDSI value of ±4 (the conventional extreme-drought / extreme-wetness threshold), so normalizing by it rescales the line into the `m`/`b` pair the PDSI recursion consumes.

The reference also carries weekly window-length sets; `climate_indices` is monthly-only for Palmer indices (see the design doc's "Out of scope"), so only the monthly set is ported.

**Note on the test below:** the natural-looking end-to-end test — a constant Z series, whose extreme sums are exactly proportional to window length — was deliberately rejected. It produces a perfectly collinear fit through the origin, which lands on the residual-tie edge case described in Task 3 and makes the expected intercept hinge on exact-zero floating-point residuals. The composition test instead pins `duration_factors` against its own independently-tested building blocks, which is what this task actually adds: the window-length set, the call order, and the normalization.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_self_calibration.py`:

```python
class TestDurationFactors:
    def test_window_lengths_match_the_reference_monthly_set(self):
        assert self_calibration.DURATION_FACTOR_WINDOW_LENGTHS == (
            3,
            6,
            9,
            12,
            18,
            24,
            30,
            36,
            42,
            48,
        )

    @pytest.mark.parametrize("sign", [1, -1])
    def test_normalizes_the_raw_fit_by_sign_times_four(self, sign):
        rng = np.random.default_rng(7)
        z = rng.normal(0.0, 1.5, size=600)

        window_lengths = np.array(self_calibration.DURATION_FACTOR_WINDOW_LENGTHS, dtype=float)
        extreme_sums = np.array(
            [
                self_calibration.extreme_z_sum(z, length, sign)
                for length in self_calibration.DURATION_FACTOR_WINDOW_LENGTHS
            ]
        )
        raw_slope, raw_intercept = self_calibration.least_squares_fit(
            window_lengths, extreme_sums, sign
        )

        slope, intercept = self_calibration.duration_factors(z, sign)

        assert slope == pytest.approx(raw_slope / (sign * 4.0))
        assert intercept == pytest.approx(raw_intercept / (sign * 4.0))

        # guard against the assertions above passing vacuously on a zero fit
        assert raw_slope != pytest.approx(0.0)
        assert raw_intercept != pytest.approx(0.0)
        assert slope != pytest.approx(raw_slope)

    def test_dry_factors_are_plausible_for_a_realistic_series(self):
        rng = np.random.default_rng(7)
        z = rng.normal(0.0, 1.5, size=600)

        slope, intercept = self_calibration.duration_factors(z, self_calibration.DRY_SIGN)

        # the PDSI recursion divides by (m + b) and weights by b / (m + b), so a
        # usable dry fit needs a positive sum and a weighting fraction below 1 --
        # Palmer's fixed national values are m = 0.309, b = 2.691 for comparison
        assert slope + intercept > 0.0
        assert 0.0 < intercept / (slope + intercept) < 1.0

    def test_rejects_an_invalid_sign(self):
        with pytest.raises(InvalidArgumentError):
            self_calibration.duration_factors(np.arange(600.0), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_self_calibration.py -k DurationFactors -v`
Expected: FAIL — `AttributeError: module 'climate_indices.self_calibration' has no attribute 'DURATION_FACTOR_WINDOW_LENGTHS'`.

- [ ] **Step 3: Write minimal implementation**

In `src/climate_indices/self_calibration.py`, add `"DURATION_FACTOR_WINDOW_LENGTHS"` and `"duration_factors"` to `__all__` (keep it sorted), add near the other module constants:

```python
# Spell durations, in months, sampled by the duration-factor regression. The
# reference implementation also carries weekly sets; Palmer indices here are
# monthly-only, so only the monthly set is ported.
DURATION_FACTOR_WINDOW_LENGTHS: tuple[int, ...] = (3, 6, 9, 12, 18, 24, 30, 36, 42, 48)

# The fitted duration-factor line represents a PDSI value of +/-4, the
# conventional extreme threshold; normalizing by it rescales the regression into
# the slope/intercept pair the PDSI recursion consumes.
_PDSI_ANCHOR = 4.0
```

and append:

```python
def duration_factors(z_values: np.ndarray, sign: int) -> tuple[float, float]:
    """Fit the wet or dry duration factors for one location's Z-index series.

    Ports ``CalcDurFact()``, the self-calibration step that replaces Palmer's
    (1965) fixed national duration-factor constants with values fitted to the
    location at hand. One representative extreme rolling Z sum is taken per
    spell duration, a line is fitted through those ten points, and the result is
    normalized against a PDSI value of +/-4.

    Callers are responsible for restricting ``z_values`` to the calibration
    period before calling.

    Args:
        z_values: The raw Z-index series for the calibration period, in
            chronological order; NaN means missing.
        sign: WET_SIGN to fit wet-spell factors, DRY_SIGN for dry-spell factors.

    Returns:
        A tuple of (m, b) -- the duration-factor slope and intercept, in the
        form the PDSI recursion consumes as wetm/wetb or drym/dryb.

    Raises:
        InvalidArgumentError: If sign is invalid.
    """
    _validate_sign(sign)

    series = np.asarray(z_values, dtype=float)
    window_lengths = np.array(DURATION_FACTOR_WINDOW_LENGTHS, dtype=float)
    extreme_sums = np.array(
        [extreme_z_sum(series, length, sign) for length in DURATION_FACTOR_WINDOW_LENGTHS]
    )

    slope, intercept = least_squares_fit(window_lengths, extreme_sums, sign)

    anchor = sign * _PDSI_ANCHOR
    return slope / anchor, intercept / anchor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_self_calibration.py -v`
Expected: PASS (30 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/climate_indices/self_calibration.py tests/test_self_calibration.py
uv run ruff format src/climate_indices/self_calibration.py tests/test_self_calibration.py
git add src/climate_indices/self_calibration.py tests/test_self_calibration.py
git commit -m "feat: add scPDSI duration-factor fitting and normalization"
```

---

### Task 5: Import-direction guard and full verification

**Files:**
- Test: `tests/test_self_calibration.py`
- Verify (no modification expected): full test suite, ruff, mypy

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: nothing new — this task is the architectural guard plus verification.

Issue [#718](https://github.com/monocongo/climate_indices/issues/718)'s second acceptance criterion is that the dependency edge runs one way only: `palmer.py` will import this module in PR4, and this module must never import `palmer`. A test enforces it, so a later refactor cannot quietly introduce the cycle.

- [ ] **Step 1: Write the guard test**

Add to `tests/test_self_calibration.py` (and add `import ast` / `from pathlib import Path` to the file's imports):

```python
class TestModuleBoundaries:
    def test_does_not_import_palmer(self):
        """The dependency edge runs palmer -> self_calibration only (issue #718)."""
        source = Path(self_calibration.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported.add(module)
                imported.update(f"{module}.{alias.name}" for alias in node.names)

        offenders = sorted(name for name in imported if "palmer" in name)
        assert not offenders, f"self_calibration must not import palmer: {offenders}"

    def test_public_surface_is_exported(self):
        assert set(self_calibration.__all__) == {
            "DRY_SIGN",
            "DURATION_FACTOR_WINDOW_LENGTHS",
            "WET_SIGN",
            "duration_factors",
            "extreme_z_sum",
            "kth_smallest",
            "least_squares_fit",
            "nan_safe_percentile",
        }
        for name in self_calibration.__all__:
            assert hasattr(self_calibration, name), f"__all__ names a missing attribute: {name}"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_self_calibration.py -v`
Expected: PASS (32 tests). If `test_public_surface_is_exported` fails, an earlier task's `__all__` addition was missed — reconcile `__all__` against the File Structure section's list rather than editing the test.

- [ ] **Step 3: Run the full verification sweep**

```bash
uv run pytest
```
Expected: PASS — 1061 previously-passing tests plus the 32 new ones (new total 1093), 37 deselected.

```bash
uv run pytest tests/test_palmer.py -m validation -v
```
Expected: PASS. This module does not touch `palmer.py`, so the 344-division NOAA fixture comparisons must be entirely unaffected; a failure here means something outside this PR's scope was modified.

```bash
uv run pytest tests/test_pattern_compliance.py -v
```
Expected: PASS. The compliance dashboard tracks the 7 public indices only, so a new private statistics module should not change its results.

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
```
Expected: all clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_self_calibration.py
git commit -m "test: guard the palmer -> self_calibration import direction"
```

- [ ] **Step 5: Update tracking issues**

Push the branch, open the PR against `main`, then comment on and close [#718](https://github.com/monocongo/climate_indices/issues/718) and tick its box in the epic tracking issue [#716](https://github.com/monocongo/climate_indices/issues/716).

```bash
git push -u origin feature/scpdsi-2-selfcal-stats
gh pr create --base main \
  --title "scPDSI PR2: self-calibration statistics module" \
  --body "Part of epic #716, closes #718.

Adds \`src/climate_indices/self_calibration.py\` — the pure, independently-testable statistics pieces of the Wells, Goddard & Wilhite (2004) self-calibration procedure, ported from the reference C++ (algorithm reference only; no GPL source enters this repository):

- \`extreme_z_sum()\` — rolling-window extreme Z-index sum per duration-factor window length, including the wet/dry asymmetric \"reasonable extreme\" filter.
- \`least_squares_fit()\` — correlation-adaptive least squares: trims trailing points until the sign-weighted correlation clears 0.85 (floor of 4 points), then anchors the intercept through the most extreme residual.
- \`kth_smallest()\` / \`nan_safe_percentile()\` — the reference's exact order statistic (1-indexed, truncating \`k = int(fraction * size)\`, NaN-safe), deliberately not \`numpy.percentile\`.
- \`duration_factors()\` — composes the above over the monthly window-length set and normalizes by a PDSI value of ±4, yielding the (m, b) pair PR1 (#717) taught the recursion to consume.

32 unit tests with hand-derived expected values; no fixture or oracle dependency, and no changes to \`palmer.py\`. A guard test enforces the one-way import edge (\`palmer\` → \`self_calibration\`, never the reverse).

Depends on nothing; PR4 (#720) wires this together with #717 and #719."
```

Then, once the PR is open:

```bash
gh issue close 718 --comment "Done in the PR above: src/climate_indices/self_calibration.py provides extreme_z_sum(), least_squares_fit(), kth_smallest()/nan_safe_percentile(), and duration_factors(), each with unit tests against hand-computed expected values and no dependency on palmer.py internals or on PR3's fixtures. The one-way import edge is enforced by a test. Ready for PR4 (#720) to consume."
```

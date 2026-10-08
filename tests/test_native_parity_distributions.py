"""Parity of the Rust Pearson Type III and generalized logistic kernels with Python.

The companion of ``test_native_parity.py`` for the L-moment distributions: SPI with
Pearson Type III, and SPEI with Pearson Type III or the log-logistic (GLO). Each case
runs once through the Rust kernels, proved by the recorder, and once through the
pure-Python reference, at ``rtol = atol = 1e-10`` with matching NaN positions.

Skipped when the extension is not built, unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1``
is set, as in CI's native legs (see ``test_native_parity.py``).
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.special
import scipy.stats
import xarray as xr

from climate_indices import compute, indices, lmoments
from tests import conftest
from tests.test_native_parity import _DATA_START, ATOL, RTOL, _assert_parity, _Recorder, _rust_and_python, _with_zeros

native = conftest.import_native()

_PEARSON = {"pearson_parameters", "pearson_cdf"}
_GLO = {"loglogistic_parameters", "loglogistic_cdf"}
_KERNELS = {indices.Distribution.pearson: _PEARSON, indices.Distribution.loglogistic: _GLO}
_MONTHLY = compute.Periodicity.monthly


def _spi_pearson(values: np.ndarray, scale: int, start: int, end: int, **kwargs):
    periodicity = kwargs.pop("periodicity", _MONTHLY)
    data_start = kwargs.pop("data_start", _DATA_START)
    return lambda: indices.spi(
        values, scale, indices.Distribution.pearson, data_start, start, end, periodicity, **kwargs
    )


def _spei(precips, pet, scale, distribution, **kwargs):
    return lambda: indices.spei(precips, pet, scale, distribution, _MONTHLY, _DATA_START, 1981, 2010, **kwargs)


# --- SPI with Pearson Type III -------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 3, 6, 12])
def test_spi_pearson_monthly_reference_series(monkeypatch, precips_mm_monthly, scale):
    rust, python, calls = _rust_and_python(monkeypatch, _spi_pearson(precips_mm_monthly, scale, 1981, 2010))
    assert calls == _PEARSON
    _assert_parity(rust, python)
    assert np.isnan(rust[: scale - 1]).all()


def test_spi_pearson_matches_the_committed_fixture(
    monkeypatch,
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    spi_6_month_pearson3,
):
    run = _spi_pearson(
        precips_mm_monthly.flatten(),
        6,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        data_start=data_year_start_monthly,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _PEARSON
    _assert_parity(rust, python)
    # the same tolerance tests/test_indices.py holds the Python path to
    np.testing.assert_allclose(rust, spi_6_month_pearson3, atol=0.01, equal_nan=True)


@pytest.mark.parametrize("zero_handling", ["classic", "center_of_mass", "mean_zero"])
@pytest.mark.parametrize("output_scale", ["normal", "probability", "bounded"])
def test_spi_pearson_zeros_zero_handling_and_output_scales(
    monkeypatch, precips_mm_monthly, zero_handling, output_scale
):
    dry = _with_zeros(precips_mm_monthly, 0.25, seed=11)
    run = _spi_pearson(dry, 3, 1981, 2010, zero_handling=zero_handling, output_scale=output_scale)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _PEARSON
    _assert_parity(rust, python)


def test_spi_pearson_trace_values(monkeypatch, precips_mm_monthly):
    """A value below 0.0005 is a trace value when the step has no zero mass."""
    traced = precips_mm_monthly.copy()
    rng = np.random.default_rng(12)
    traced[rng.random(traced.shape) < 0.08] = 0.0003
    rust, python, calls = _rust_and_python(monkeypatch, _spi_pearson(traced, 1, 1981, 2010))
    assert calls == _PEARSON
    _assert_parity(rust, python)


def test_spi_pearson_with_missing_values(monkeypatch, precips_mm_monthly):
    gappy = precips_mm_monthly.copy().flatten()
    gappy[np.random.default_rng(13).random(gappy.size) < 0.15] = np.nan
    gappy[:30] = np.nan
    rust, python, calls = _rust_and_python(monkeypatch, _spi_pearson(gappy, 6, 1981, 2010))
    assert calls == _PEARSON
    _assert_parity(rust, python)


def test_spi_pearson_fewer_than_four_non_zero_values_in_one_month(monkeypatch, precips_mm_monthly):
    """One calendar month with three non-zero calibration values fails its fit on both paths."""
    sparse = precips_mm_monthly.copy()
    sparse[:, 6] = 0.0
    sparse[[90, 95, 100], 6] = [12.0, 30.0, 8.0]
    rust, python, calls = _rust_and_python(monkeypatch, _spi_pearson(sparse, 1, 1981, 2010))
    assert calls == _PEARSON
    _assert_parity(rust, python)


@pytest.mark.parametrize("scale", [1, 30])
def test_spi_pearson_daily_series(
    monkeypatch,
    precips_mm_daily,
    data_year_start_daily,
    calibration_year_start_daily,
    calibration_year_end_daily,
    scale,
):
    run = _spi_pearson(
        precips_mm_daily,
        scale,
        calibration_year_start_daily,
        calibration_year_end_daily,
        periodicity=compute.Periodicity.daily,
        data_start=data_year_start_daily,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _PEARSON
    _assert_parity(rust, python)


@pytest.mark.parametrize("scale", [1, 6, 12])
def test_ncei_precipitation_runs_on_the_rust_kernels(monkeypatch, scale):
    """The NCEI characterization's inputs, which only reach Rust under ``np.errstate(all="ignore")``.

    ``test_ncei_spi_reference.py`` runs under the default NumPy error policy, so it exercises the
    Python path; this runs real division precipitation through both paths and holds the Rust one
    to the same ceiling against NOAA's SPI.
    """
    fixtures = Path(__file__).parent / "fixture"
    divisions = json.loads((fixtures / "nclimdiv" / "divisions.json").read_text(encoding="utf-8"))[:8]
    precips = [np.load(fixtures / "palmer" / division / "precips.npy") for division in divisions]
    run = lambda: np.stack(  # noqa: E731
        [indices.spi(p, scale, indices.Distribution.pearson, 1895, 1895, 2022, _MONTHLY) for p in precips]
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    # a fallback to gamma would add the gamma kernels
    assert calls == _PEARSON
    _assert_parity(rust, python)

    reference = np.load(fixtures / "ncei_spi" / f"sp{scale:02d}.npy")[:8].astype(np.float64)
    provenance = json.loads((fixtures / "ncei_spi" / "provenance.json").read_text(encoding="utf-8"))
    ceiling = provenance["validation_tolerance"][f"sp{scale:02d}_max"]
    both = ~np.isnan(reference) & ~np.isnan(rust)
    assert both.sum() > 0.9 * rust.size
    assert np.abs(rust[both] - reference[both]).max() < ceiling


@pytest.mark.parametrize("fraction", [0.2, 0.6])
def test_pearson_parameters_match_the_python_fit(monkeypatch, precips_mm_monthly, fraction):
    dry = _with_zeros(precips_mm_monthly, fraction, seed=14)
    run = lambda: np.stack(  # noqa: E731
        compute.pearson_parameters(dry, _DATA_START, 1981, 2010, _MONTHLY)
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"pearson_parameters"}
    _assert_parity(rust, python)


def test_pearson_parameters_daily(monkeypatch, precips_mm_daily, data_year_start_daily):
    run = lambda: np.stack(  # noqa: E731
        compute.pearson_parameters(
            precips_mm_daily,
            data_year_start_daily,
            data_year_start_daily,
            data_year_start_daily + 14,
            compute.Periodicity.daily,
        )
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"pearson_parameters"}
    _assert_parity(rust, python)


@pytest.mark.parametrize("dry_months", range(13))
def test_the_gamma_fallback_fires_on_the_same_inputs(monkeypatch, precips_mm_monthly, dry_months):
    """The Pearson fit is lost, and the block refitted with gamma, for the same blocks on both paths.

    The first ``dry_months`` calendar months are dry through the calibration years 1981-2010 (rows
    86-115) and wet elsewhere, so their Pearson fit fails and their wet values are lost. Whether the
    lost fraction crosses the fallback threshold flips between seven and eight such months.
    """
    block = precips_mm_monthly.copy()
    block[86:116, :dry_months] = 0.0

    def run() -> tuple[bool, np.ndarray]:
        standardized, used, _ = compute._fit_pearson_with_fallback(
            block.copy(), indices.Distribution.pearson, _DATA_START, 1981, 2010, _MONTHLY, None, "test"
        )
        return used, standardized

    recorder = _Recorder(native)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monkeypatch.setattr(compute, "_native", recorder)
        rust_used, rust = run()
        monkeypatch.setattr(compute, "_native", None)
        python_used, python = run()
    assert rust_used is python_used
    assert python_used is (dry_months >= 8)
    assert {"pearson_parameters", "pearson_cdf"} <= recorder.calls
    # the gamma refit's transform runs on the Rust kernel (its fit stays Python for a column
    # with no positive value, as in RUST-002)
    assert ("gamma_probabilities" in recorder.calls) is python_used
    _assert_parity(rust, python)


# --- SPEI ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("distribution", "kernels"),
    [(indices.Distribution.pearson, _PEARSON), (indices.Distribution.loglogistic, _GLO)],
    ids=["pearson", "loglogistic"],
)
@pytest.mark.parametrize("scale", [1, 3, 6, 12])
def test_spei_at_several_scales(monkeypatch, precips_mm_monthly, pet_thornthwaite_mm, distribution, kernels, scale):
    run = _spei(precips_mm_monthly, pet_thornthwaite_mm, scale, distribution)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == kernels
    _assert_parity(rust, python)


@pytest.mark.parametrize("distribution", [indices.Distribution.pearson, indices.Distribution.loglogistic])
@pytest.mark.parametrize("output_scale", ["probability", "bounded"])
def test_spei_output_scales(monkeypatch, precips_mm_monthly, pet_thornthwaite_mm, distribution, output_scale):
    run = _spei(precips_mm_monthly, pet_thornthwaite_mm, 3, distribution, output_scale=output_scale)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _KERNELS[distribution]
    _assert_parity(rust, python)


def test_spei_loglogistic_with_zero_precipitation(
    monkeypatch, precips_mm_monthly, pet_thornthwaite_mm, data_year_start_monthly, data_year_end_monthly
):
    """SPEI keeps zeros as ordinary values in its GLO fit, so a dry series is parity-tested too."""
    dry = _with_zeros(precips_mm_monthly, 0.3, seed=16)
    run = lambda: indices.spei(  # noqa: E731
        dry,
        pet_thornthwaite_mm,
        3,
        indices.Distribution.loglogistic,
        _MONTHLY,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _GLO
    _assert_parity(rust, python)


def test_spei_pearson_with_negative_skew(monkeypatch, precips_mm_monthly):
    """PET above precipitation by a gamma amount leaves a negatively skewed water balance."""
    gamma_excess = np.random.default_rng(17).gamma(2.0, 20.0, precips_mm_monthly.shape)
    pet = precips_mm_monthly + gamma_excess
    skews = compute.pearson_parameters(precips_mm_monthly - pet + 1000.0, _DATA_START, 1981, 2010, _MONTHLY)[3]
    assert (skews < 0).all()
    rust, python, calls = _rust_and_python(monkeypatch, _spei(precips_mm_monthly, pet, 1, indices.Distribution.pearson))
    assert calls == _PEARSON
    _assert_parity(rust, python)


# --- the transforms with caller-supplied parameters ----------------------------------


def test_pearson_support_limits_with_negative_and_positive_skew(monkeypatch):
    """Values beyond the lower (positive skew) or upper (negative skew) limit take the sentinels."""
    values = np.tile(np.linspace(-20.0, 150.0, 12), (30, 1))
    skews = np.array([-1.0, -0.4, 1.0, 0.4, 0.0, 2e-5, -2e-5, 1e-5, 1.5, -1.5, 0.05, -0.05])
    run = lambda: compute.transform_fitted_pearson(  # noqa: E731
        values,
        _DATA_START,
        _DATA_START,
        _DATA_START + 29,
        _MONTHLY,
        np.zeros(12),
        np.full(12, 40.0),
        np.full(12, 25.0),
        skews,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"pearson_cdf"}
    _assert_parity(rust, python)
    # both limits are exercised: the sentinels 0.0005 and 0.9995 map to the same z-scores
    assert np.isclose(python, scipy.stats.norm.ppf(0.0005)).any()
    assert np.isclose(python, scipy.stats.norm.ppf(0.9995)).any()


def test_invalid_pearson_parameters_give_the_same_missing_values(monkeypatch):
    values = np.tile(np.linspace(1.0, 100.0, 12), (30, 1))
    run = lambda: compute.transform_fitted_pearson(  # noqa: E731
        values,
        _DATA_START,
        _DATA_START,
        _DATA_START + 29,
        _MONTHLY,
        np.zeros(12),
        np.zeros(12),
        np.zeros(12),
        np.zeros(12),
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"pearson_cdf"}
    _assert_parity(rust, python)


@pytest.mark.parametrize("shape", [-0.6, -0.2, 0.0, 5e-7, 0.3, 0.9, float("nan")])
def test_loglogistic_cdf_beyond_its_support(monkeypatch, shape):
    """A finite value past the support maps to probability 0 or 1; invalid parameters to NaN."""
    values = np.tile(np.linspace(-200.0, 400.0, 12), (30, 1))
    run = lambda: compute.transform_fitted_loglogistic(  # noqa: E731
        values,
        _DATA_START,
        _DATA_START,
        _DATA_START + 29,
        _MONTHLY,
        np.full(12, 50.0),
        np.full(12, 20.0),
        np.full(12, shape),
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"loglogistic_cdf"}
    _assert_parity(rust, python)


def _supplied_glo_parameters(values: np.ndarray) -> list[np.ndarray]:
    with np.errstate(all="ignore"):
        return [np.array(a) for a in compute.loglogistic_parameters(values, _DATA_START, 1981, 2010, _MONTHLY)]


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda a: np.ma.masked_array(a, mask=np.arange(a.size) == 3), id="masked"),
        pytest.param(lambda a: xr.DataArray(a, dims=["period"]), id="dataarray"),
    ],
)
def test_array_subclass_parameters_keep_the_python_cdf(monkeypatch, precips_mm_monthly, wrap):
    """A mask or labels mean something to the NumPy path; a bare buffer would drop them."""
    parameters = [wrap(a) for a in _supplied_glo_parameters(precips_mm_monthly)]

    def run():
        try:
            return compute.transform_fitted_loglogistic(
                precips_mm_monthly, _DATA_START, 1981, 2010, _MONTHLY, *parameters
            )
        except ValueError as error:  # a DataArray fails inside xarray on the Python path
            return type(error)

    recorder = _Recorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(compute, "_native", recorder)
        rust = run()
        monkeypatch.setattr(compute, "_native", None)
        python = run()
    assert "loglogistic_cdf" not in recorder.calls
    if isinstance(python, type):
        assert rust is python
    else:
        _assert_parity(np.asarray(rust), np.asarray(python))


def test_zero_dimensional_values_and_oversized_parameters_keep_the_python_cdf(monkeypatch):
    """Shapes only the private fit takes: both paths must give the same result or error."""
    cases = {
        "scalar": (np.array(5.0), np.array(50.0), np.array(20.0), np.array(0.2)),
        "oversized": (np.ones((20, 12)), np.ones((3, 20, 12)), np.ones((3, 20, 12)), np.full((3, 20, 12), 0.2)),
        "mismatched": (np.ones((20, 12)), np.ones((1, 24)), np.ones((1, 24)), np.full((1, 24), 0.2)),
    }
    for label, arguments in cases.items():
        outcomes = []
        for backend in (native, None):
            monkeypatch.setattr(compute, "_native", backend)
            with np.errstate(all="ignore"):
                try:
                    result = compute._loglogistic_fit(*arguments)
                    outcomes.append(("ok", result.shape))
                except ValueError as error:
                    outcomes.append(("raises", type(error)))
        assert outcomes[0] == outcomes[1], label


# --- spatial blocks ------------------------------------------------------------------


def _spatial_block(precips_mm_monthly: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = precips_mm_monthly.flatten()[:, None, None] * rng.uniform(0.5, 1.5, (1, 3, 4))
    block = _with_zeros(block, 0.1, seed=seed + 1)
    block[:, 0, 0] = np.nan
    block[:, 1, 1] = 0.0
    return block


def test_spi_pearson_spatial_block(monkeypatch, precips_mm_monthly):
    block = _spatial_block(precips_mm_monthly, 20)
    run = _spi_pearson(block, 6, 1981, 2010, spatial_time_major=True)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _PEARSON
    _assert_parity(rust, python)


@pytest.mark.parametrize("distribution", [indices.Distribution.pearson, indices.Distribution.loglogistic])
def test_spei_spatial_block(monkeypatch, precips_mm_monthly, pet_thornthwaite_mm, distribution):
    precips = _spatial_block(precips_mm_monthly, 22)
    pet = pet_thornthwaite_mm.flatten()[:, None, None] * np.ones((1, 3, 4))
    run = lambda: indices.spei(  # noqa: E731
        precips, pet, 3, distribution, _MONTHLY, _DATA_START, 1981, 2010, spatial_time_major=True
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _KERNELS[distribution]
    _assert_parity(rust, python)


def _failing_pearson_block(precips_mm_monthly: np.ndarray) -> np.ndarray:
    """Eleven of twelve calendar months fail: ten dry, one constant (lambda_2 = 0), one healthy."""
    block = precips_mm_monthly.copy()
    block[:, :10] = 0.0
    block[:, 10] = 7.0
    return block


def _failing_loglogistic_block(precips_mm_monthly: np.ndarray) -> np.ndarray:
    """Eleven of twelve calendar months fail: nine missing, one constant, one with two valid values."""
    block = precips_mm_monthly.copy()
    block[:, :9] = np.nan
    block[:, 9] = 7.0
    block[:, 10] = np.nan
    block[[90, 91], 10] = [1.0, 2.0]
    return block


@pytest.mark.parametrize("spatial", [False, True], ids=["monthly-series", "spatial-block"])
@pytest.mark.parametrize(
    ("fit", "make_block", "kernel"),
    [
        (compute.pearson_parameters, _failing_pearson_block, "pearson_parameters"),
        (compute.loglogistic_parameters, _failing_loglogistic_block, "loglogistic_parameters"),
    ],
    ids=["pearson", "loglogistic"],
)
def test_failed_fit_counts_and_warning_events_match(
    monkeypatch, caplog, precips_mm_monthly, fit, make_block, kernel, spatial
):
    """The failed-fit count behind the high-failure-rate warning, and the events it emits, agree."""
    block = make_block(precips_mm_monthly)
    if spatial:
        block = block[:, :, None] * np.ones((1, 1, 3))
    strategy = compute._default_fallback_strategy
    seen: list[tuple[int, int]] = []
    original = strategy.should_warn_high_failure_rate

    def spy(failed: int, total: int) -> bool:
        seen.append((failed, total))
        return original(failed, total)

    monkeypatch.setattr(strategy, "should_warn_high_failure_rate", spy)
    caplog.set_level(logging.WARNING)

    def run() -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[str, str]]]:
        seen.clear()
        caplog.clear()
        parameters = np.stack(fit(block, _DATA_START, 1981, 2010, _MONTHLY))
        events = [
            (record.levelname, str(record.msg.get("event")))
            for record in caplog.records
            if isinstance(record.msg, dict) and record.name != "climate_indices.lmoments"
        ]
        return parameters, list(seen), events

    recorder = _Recorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(compute, "_native", recorder)
        rust, rust_counts, rust_events = run()
        monkeypatch.setattr(compute, "_native", None)
        python, python_counts, python_events = run()
    assert recorder.calls == {kernel}
    expected_total = block[0].size
    assert python_counts == [(11 * (expected_total // 12), expected_total)]
    assert rust_counts == python_counts
    assert rust_events == python_events
    assert any("failure" in event.lower() for _, event in python_events), "the case must trigger the warning"
    _assert_parity(rust, python)


@pytest.mark.parametrize("spatial", [False, True], ids=["monthly-series", "spatial-block"])
@pytest.mark.parametrize(
    ("fit", "kernel"),
    [
        (compute.pearson_parameters, "pearson_parameters"),
        (compute.loglogistic_parameters, "loglogistic_parameters"),
    ],
    ids=["pearson", "loglogistic"],
)
def test_failures_below_the_high_rate_threshold_are_summarized(
    monkeypatch, caplog, precips_mm_monthly, fit, kernel, spatial
):
    """One failed calendar month logs a single summary event on both backends.

    The Rust fit writes none of the per-step ``climate_indices.lmoments`` records,
    so the summary is the only diagnostic for an isolated failure.
    """
    block = precips_mm_monthly.copy()
    block[:, 3] = np.nan
    if spatial:
        block = block[:, :, None] * np.ones((1, 1, 3))
    caplog.set_level(logging.WARNING)

    def summaries() -> list[dict]:
        caplog.clear()
        fit(block, _DATA_START, 1981, 2010, _MONTHLY)
        return [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict) and record.msg.get("event") == "distribution_fitting_failures"
        ]

    recorder = _Recorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(compute, "_native", recorder)
        rust = summaries()
        monkeypatch.setattr(compute, "_native", None)
        python = summaries()
    assert recorder.calls == {kernel}
    cells = block[0].size // 12
    for events in (rust, python):
        assert len(events) == 1
        assert (events[0]["failure_count"], events[0]["total_count"]) == (cells, 12 * cells)


# --- routing -------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf, 1.7e308, 1e101])
def test_unbounded_calibration_values_keep_the_python_fit(monkeypatch, precips_mm_monthly, bad_value):
    """An infinity or an overflowing weight makes the L-moments NaN, where the Python fits differ.

    The guard is conservative: anything past 1e100 stays on Python, overflowing or not.
    """
    values = precips_mm_monthly.copy()
    values[90, 3] = bad_value
    for fit in (compute.pearson_parameters, compute.loglogistic_parameters):
        recorder = _Recorder(native)
        monkeypatch.setattr(compute, "_native", recorder)
        with np.errstate(all="ignore"):
            fit(values, _DATA_START, 1981, 2010, _MONTHLY)
        assert not recorder.calls, fit.__name__


def test_float32_and_masked_blocks_keep_the_python_fit(monkeypatch, precips_mm_monthly):
    recorder = _Recorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    with np.errstate(all="ignore"):
        compute.pearson_parameters(precips_mm_monthly.astype(np.float32), _DATA_START, 1981, 2010, _MONTHLY)
        compute.loglogistic_parameters(np.ma.masked_less(precips_mm_monthly, 1.0), _DATA_START, 1981, 2010, _MONTHLY)
    assert not recorder.calls


def test_numpy_error_policy_keeps_the_python_fit(monkeypatch, precips_mm_monthly):
    recorder = _Recorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compute.pearson_parameters(precips_mm_monthly, _DATA_START, 1981, 2010, _MONTHLY)
        compute.loglogistic_parameters(precips_mm_monthly, _DATA_START, 1981, 2010, _MONTHLY)
    assert not recorder.calls, "the default NumPy error policy is not all-ignore"


def test_climate_indices_warnings_are_unchanged(monkeypatch, precips_mm_monthly):
    """The failure-rate and data-quality warnings come from Python on both paths."""
    gappy = _with_zeros(precips_mm_monthly, 0.9, seed=30).flatten()
    gappy[: 12 * 100] = np.nan

    def warning_types(backend):
        recorder = _Recorder(backend) if backend is not None else None
        monkeypatch.setattr(compute, "_native", recorder)
        with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
            warnings.simplefilter("always")
            _spi_pearson(gappy, 3, 1990, 2017)()
        categories = sorted(
            {
                (w.category.__name__, str(w.message))
                for w in caught
                if issubclass(w.category, Warning) and "climate_indices" in w.category.__module__
            }
        )
        return categories, (recorder.calls if recorder is not None else set())

    rust, rust_calls = warning_types(native)
    python, _ = warning_types(None)
    assert rust == python
    assert rust, "the case is meant to trigger at least one data-quality warning"
    assert rust_calls, "the native arm must actually reach the Rust kernels"


# --- the kernels against their SciPy and Python primitives ---------------------------


def test_pearson_cdf_matches_scipy_across_parameters():
    rng = np.random.default_rng(40)
    count = 30_000
    skews = np.concatenate([rng.uniform(-3.0, 3.0, count), rng.uniform(-3e-5, 3e-5, 500), [0.0, 1.6e-5, -1.6e-5]])
    locs = rng.uniform(-50.0, 50.0, skews.size)
    scales = np.exp(rng.uniform(-2.0, 4.0, skews.size))
    values = locs + scales * rng.normal(0.0, 2.5, skews.size)
    ours = native.pearson_cdf(values[None, :], skews, locs, scales)[0]
    np.testing.assert_allclose(
        ours, scipy.stats.pearson3.cdf(values, skews, loc=locs, scale=scales), rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("skew", [-1.5, -0.4, 0.4, 1.5, 3e-6])
def test_pearson_cdf_tails_keep_their_precision(skew):
    """Tail probabilities to a relative tolerance, and near-1 ones through the index's transform.

    The comparison above uses ``atol=1e-10``, which hides a tail computed as ``1 - igam``;
    the Cephes ports exist because the transformed tails are ill-conditioned.
    """
    z = np.linspace(-14.0, 14.0, 561)
    ours = native.pearson_cdf(z[None, :], np.full(z.size, skew), np.zeros(z.size), np.ones(z.size))[0]
    expected = scipy.stats.pearson3.cdf(z, skew)
    np.testing.assert_allclose(ours, expected, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(scipy.stats.norm.ppf(ours), scipy.stats.norm.ppf(expected), rtol=1e-10, atol=1e-10)


def test_pearson_cdf_edge_arguments_match_scipy():
    values = np.array([0.0, 1.0, np.inf, -np.inf, np.nan, 5.0, 5.0, 5.0, 5.0, 5.0])
    skews = np.array([1.0, 1.0, 1.0, -1.0, 1.0, np.nan, np.inf, 1.0, 1e300, 1e-300])
    locs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0])
    scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    scales[2] = 0.0
    expected = scipy.stats.pearson3.cdf(values, skews, loc=locs, scale=scales)
    ours = native.pearson_cdf(values[None, :], skews, locs, scales)[0]
    np.testing.assert_allclose(ours, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("sample_size", [4, 5, 12, 30, 61])
def test_fits_match_the_python_lmoments_modules(sample_size):
    """The Rust fits against ``lmoments.fit`` and ``fit_glo`` on skewed, symmetric, and tied samples."""
    rng = np.random.default_rng(50 + sample_size)
    columns = [
        rng.gamma(0.8, 30.0, sample_size),
        rng.gamma(30.0, 3.0, sample_size),
        200.0 - rng.gamma(2.0, 20.0, sample_size),
        rng.normal(100.0, 15.0, sample_size),
        np.round(rng.gamma(2.0, 10.0, sample_size)),
    ]
    block = np.stack(columns, axis=1)
    p0, locs, scales, skews, pearson_valid = native.pearson_parameters(block)
    glo_locs, glo_scales, glo_shapes, glo_valid = native.loglogistic_parameters(block)
    for index, column in enumerate(columns):
        try:
            expected = lmoments.fit(column)
            assert pearson_valid[index]
            np.testing.assert_allclose(
                [locs[index], scales[index], skews[index]],
                [expected["loc"], expected["scale"], expected["skew"]],
                rtol=1e-10,
                atol=1e-10,
            )
        except ValueError:
            assert not pearson_valid[index]
        try:
            expected = lmoments.fit_glo(column)
            assert glo_valid[index]
            np.testing.assert_allclose(
                [glo_locs[index], glo_scales[index], glo_shapes[index]],
                [expected["loc"], expected["scale"], expected["shape"]],
                rtol=1e-10,
                atol=1e-10,
            )
        except ValueError:
            assert not glo_valid[index]
    assert (p0 == 0.0).all()


def test_pearson_scale_of_near_symmetric_samples_matches_python():
    """``exp(gammaln(a) - gammaln(a + 0.5))`` cancels ~12 digits for a near-zero L-skewness.

    SciPy's ``gammaln`` rounds to a different last bit with and without fused multiply-adds,
    so the Rust ``lgam`` mirrors the build's contraction (``special::mul_add``); without
    that, ``scale`` differed from the oracle by up to 5e-4 for |tau_3| below about 1.6e-3.
    """
    base = np.arange(1.0, 61.0)
    columns = []
    for bump in np.logspace(-4, 1, 200):
        column = base.copy()
        column[0] += bump
        columns.append(column)
    block = np.stack(columns, axis=1)

    _, scales, skews, valid = lmoments.fit_spatial(block)
    _, rust_scales, rust_skews, rust_valid = native.pearson_parameters(block)[1:]
    assert valid.all() and rust_valid.all()
    assert np.abs(lmoments._estimate_lmoments_spatial(block)[0][2]).min() < 1e-4, "must reach the small-skew range"
    np.testing.assert_allclose(rust_scales, scales, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(rust_skews, skews, rtol=RTOL, atol=ATOL)
    # the single-series fit agrees on a few columns too
    for index in (0, 50, 100, 199):
        np.testing.assert_allclose(rust_scales[index], lmoments.fit(columns[index])["scale"], rtol=RTOL, atol=ATOL)


def test_small_skew_cdf_is_the_normal_cdf_in_both_tails():
    """A zero skew is the normal CDF, i.e. Cephes ``ndtr``, including its far tails."""
    z = np.array([-40.0, -8.0, -1.0, -0.1, 0.0, 0.1, 1.0, 8.0, 40.0])
    ours = native.pearson_cdf(z[None, :], np.zeros(z.size), np.zeros(z.size), np.ones(z.size))[0]
    np.testing.assert_allclose(ours, scipy.special.ndtr(z), rtol=1e-12, atol=0.0)


def test_loglogistic_cdf_matches_the_numpy_formula():
    rng = np.random.default_rng(60)
    count = 20_000
    shapes = np.concatenate([rng.uniform(-0.95, 0.95, count), [0.0, 1e-6, -1e-6, 2e-6, np.nan, 1.5]])
    locs = rng.uniform(-50.0, 50.0, shapes.size)
    scales = np.exp(rng.uniform(-2.0, 4.0, shapes.size))
    values = locs + scales * rng.normal(0.0, 3.0, shapes.size)
    ours = native.loglogistic_cdf(values[None, :], locs, scales, shapes)[0]
    with np.errstate(all="ignore"):
        z = (values - locs) / scales
        y = np.where(np.abs(shapes) <= 1e-6, z, -np.log(np.maximum(0.0, 1.0 - shapes * z)) / shapes)
        expected = np.clip(1.0 / (1.0 + np.exp(-y)), 0.0, 1.0)
    np.testing.assert_allclose(ours, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


# --- the native boundary -------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "arguments"),
    [
        ("pearson_cdf", [np.ones((2, 12)), np.full(12, 0.5), np.zeros(12), np.ones(12)]),
        ("loglogistic_cdf", [np.ones((2, 12)), np.zeros(12), np.ones(12), np.full(12, 0.1)]),
    ],
)
def test_cdf_kernels_reject_a_parameter_of_the_wrong_length(kernel, arguments):
    arguments[2] = np.ones(11)
    with pytest.raises(ValueError, match="11"):
        getattr(native, kernel)(*arguments)


@pytest.mark.parametrize("shape", [(0, 12), (2, 0), (0, 0)])
def test_kernels_accept_empty_blocks(shape):
    empty = np.empty(shape)
    parameters = np.ones(shape[1])
    *pearson, valid = native.pearson_parameters(empty)
    assert all(array.shape == (shape[1],) for array in pearson) and valid.shape == (shape[1],)
    *glo, valid = native.loglogistic_parameters(empty)
    assert all(array.shape == (shape[1],) for array in glo) and valid.shape == (shape[1],)
    assert not valid.any()
    assert native.pearson_cdf(empty, parameters, parameters, parameters).shape == shape
    assert native.loglogistic_cdf(empty, parameters, parameters, parameters).shape == shape


@pytest.mark.parametrize("layout", ["packed", "offset"])
def test_kernels_reject_unaligned_arrays(layout):
    def unaligned_like(original: np.ndarray) -> np.ndarray:
        if layout == "packed":
            array = np.zeros(original.shape, dtype=[("value", "f8"), ("flag", "u1")])["value"]
        else:
            array = np.ndarray(original.shape, dtype=np.float64, buffer=bytearray(original.nbytes + 1), offset=1)
        array[:] = original
        assert not array.flags.aligned
        return array

    block = np.arange(1.0, 25.0).reshape(2, 12)
    parameters = np.full(12, 0.5)
    for kernel in (native.pearson_parameters, native.loglogistic_parameters):
        with pytest.raises(ValueError, match="unaligned float64 array"):
            kernel(unaligned_like(block))
    for kernel in (native.pearson_cdf, native.loglogistic_cdf):
        for position in range(4):
            arguments = [block, parameters, parameters, parameters]
            arguments[position] = unaligned_like(arguments[position])
            with pytest.raises(ValueError, match="unaligned float64 array"):
                kernel(*arguments)

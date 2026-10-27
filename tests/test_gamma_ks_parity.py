"""Gamma KS statistics preserve SciPy's domain and warning decisions."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.special
import scipy.stats

from climate_indices import compute
from climate_indices.exceptions import GoodnessOfFitWarning
from tests import conftest

native = conftest.import_native()


@pytest.mark.parametrize("rows", [0, 1, 40, 1000])
@pytest.mark.parametrize("columns", [12, 366])
def test_statistics_match_scipy_with_missing_and_invalid_values(monkeypatch, rows, columns):
    values = np.random.default_rng(1434).gamma(2.0, 30.0, (rows, columns))
    values.reshape(-1)[::13] = np.nan
    if rows:
        values[:, 0] = np.nan
        values[0, 1:6] = [-1.0, -np.inf, np.inf, -0.0, np.nextafter(0.0, 1.0)]
    alphas = np.full(columns, 2.0)
    betas = np.full(columns, 30.0)
    alphas[6:9] = [0.0, np.inf, np.nan]
    betas[9:12] = [-1.0, np.inf, np.nan]
    # Reversed views exercise the binding's stride handling.
    values, alphas, betas = values[::-1, ::-1], alphas[::-1], betas[::-1]
    with np.errstate(all="ignore"):
        rust = native.gamma_ks_statistics(values, alphas, betas)
        monkeypatch.setattr(compute, "_native", None)
        python = compute._gamma_ks_statistics(values, alphas, betas)
    np.testing.assert_array_equal(np.isnan(rust), np.isnan(python))
    # The existing native parity contract applies to dimensionless D statistics.
    np.testing.assert_allclose(rust, python, rtol=1e-10, atol=1e-10)
    for column in range(columns):
        sample = values[:, column]
        sample = sample[~np.isnan(sample)]
        alpha, beta = alphas[column], betas[column]
        if sample.size and np.isfinite(alpha) and np.isfinite(beta) and alpha > 0 and beta > 0:
            expected = scipy.stats.kstest(sample, lambda x, a=alpha, b=beta: scipy.special.gammainc(a, x / b)).statistic
            np.testing.assert_allclose(rust[column], expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("poor", [False, True])
@pytest.mark.parametrize("columns", [12, 366])
def test_warning_text_categories_and_metadata_match(monkeypatch, poor, columns):
    quantiles = scipy.stats.gamma.ppf((np.arange(40) + 0.5) / 40, 2.0, scale=30.0)
    values = np.tile(quantiles[:, None], (1, columns))
    if poor:
        values[:] = 1000.0
    values[:, 0] = np.nan
    values[0, 1] = -1.0
    alphas, betas = np.full(columns, 2.0), np.full(columns, 30.0)
    alphas[2], betas[3] = np.inf, 0.0
    outcomes = []
    for backend in (native, None):
        recorder = conftest.NativeRecorder(backend) if backend is not None else None
        monkeypatch.setattr(compute, "_native", recorder)
        with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
            warnings.simplefilter("always")
            compute._check_goodness_of_fit_gamma(values, alphas, betas)
        if recorder is not None:
            assert "gamma_ks_statistics" in recorder.calls
        outcomes.append([(item.category, str(item.message), vars(item.message)) for item in caught])
    assert outcomes[0] == outcomes[1]
    if poor:
        assert len(outcomes[0]) == 1
        assert outcomes[0][0][0] is GoodnessOfFitWarning
        assert outcomes[0][0][2]["poor_fit_count"] == columns - 4
    else:
        assert not outcomes[0]


def test_near_threshold_keeps_scipy_decision(monkeypatch):
    values = np.linspace(1.0, 100.0, 40).reshape(-1, 1)
    critical = compute._ks_critical_value(40)
    monkeypatch.setattr(
        compute, "_native", SimpleNamespace(gamma_ks_statistics=lambda *args: np.array([critical - 1e-12]))
    )
    calls = []

    def poor_fit(*args):
        calls.append(None)
        return 0.01

    monkeypatch.setattr(compute, "_ks_poor_fit_p_value", poor_fit)
    with np.errstate(all="ignore"), pytest.warns(GoodnessOfFitWarning, match="step 0 \\(p=0.0100\\)"):
        compute._check_goodness_of_fit_gamma(values, np.full(1, 2.0), np.full(1, 30.0))
    assert len(calls) == 1


def test_clearly_acceptable_statistics_skip_scipy(monkeypatch):
    def fail(*args):
        pytest.fail("An acceptable native D must skip the SciPy check")

    monkeypatch.setattr(compute, "_native", SimpleNamespace(gamma_ks_statistics=lambda *args: np.zeros(1)))
    monkeypatch.setattr(compute, "_ks_poor_fit_p_value", fail)
    with np.errstate(all="ignore"):
        compute._check_goodness_of_fit_gamma(np.ones((40, 1)), np.ones(1), np.ones(1))


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_special_reporting_policy_keeps_scipy(monkeypatch, policy):
    outcomes = []
    for backend in (native, None):
        recorder = conftest.NativeRecorder(backend) if backend is not None else None
        monkeypatch.setattr(compute, "_native", recorder)
        with (
            warnings.catch_warnings(record=True) as caught,
            np.errstate(all="ignore"),
            scipy.special.errstate(domain=policy),
        ):
            warnings.simplefilter("always")
            compute._check_goodness_of_fit_gamma(np.full((40, 1), -1.0), np.ones(1), np.ones(1))
        if recorder is not None:
            assert "gamma_ks_statistics" not in recorder.calls
        outcomes.append([(item.category, str(item.message)) for item in caught])
    assert outcomes[0] == outcomes[1]
    if policy == "warn":
        assert any(category is scipy.special.SpecialFunctionWarning for category, _ in outcomes[0])


@pytest.mark.parametrize("kind", ["float32", "masked", "numpy-policy", "old-extension", "spatial", "parameter-shape"])
def test_ineligible_inputs_keep_python(monkeypatch, kind):
    values = np.ones((40, 1))
    alphas, betas = np.ones(1), np.ones(1)
    recorder = conftest.NativeRecorder(native)
    backend = recorder
    if kind == "float32":
        values = values.astype(np.float32)
    elif kind == "masked":
        values = np.ma.array(values, mask=False)
    elif kind == "old-extension":
        backend = SimpleNamespace()
    elif kind == "spatial":
        values = values[..., None]
        alphas, betas = alphas[:, None], betas[:, None]
    elif kind == "parameter-shape":
        alphas, betas = alphas[:, None], betas[:, None]
    monkeypatch.setattr(compute, "_native", backend)
    with np.errstate(all="warn" if kind == "numpy-policy" else "ignore"):
        compute._check_goodness_of_fit_gamma(values, alphas, betas)
    assert "gamma_ks_statistics" not in recorder.calls


def test_native_failure_is_not_swallowed(monkeypatch):
    def fail(*args):
        raise RuntimeError("native failure")

    monkeypatch.setattr(compute, "_native", SimpleNamespace(gamma_ks_statistics=fail))
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="native failure"):
        compute._check_goodness_of_fit_gamma(np.ones((40, 1)), np.ones(1), np.ones(1))


@pytest.mark.parametrize("parameter", ["alphas", "betas"])
def test_binding_validates_shapes(parameter):
    arguments = {"values": np.ones((3, 2)), "alphas": np.ones(2), "betas": np.ones(2)}
    arguments[parameter] = np.ones(1)
    with pytest.raises(ValueError, match=parameter):
        native.gamma_ks_statistics(**arguments)

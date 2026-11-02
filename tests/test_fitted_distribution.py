"""Tests for the ``FittedDistribution`` seam introduced by issue #1216."""

from unittest import mock

import numpy as np
import pytest
import scipy.stats

from climate_indices import compute, indices
from climate_indices.exceptions import InvalidArgumentError

# The FittedDistribution seam fits one distribution per calendar period, which serves
# SPI, SPEI, standardized_index, and the energy indices. SPAI's t location-scale and
# normal fits are pooled over every calendar period and live in indices.spai() only.
_FITTED_DISTRIBUTIONS = tuple(
    kind
    for kind in indices.Distribution
    if kind not in (indices.Distribution.t_location_scale, indices.Distribution.normal)
)


@pytest.mark.parametrize("distribution", _FITTED_DISTRIBUTIONS)
@pytest.mark.parametrize("output_scale", compute.OUTPUT_SCALES)
def test_transform_owns_the_kernels(
    distribution: indices.Distribution, output_scale: compute.OutputScale, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolved fit never calls a public adapter or resolves its parameters again."""
    values = np.random.default_rng(1417).uniform(0.1, 0.9, (30, 12))
    fitted = compute.FittedDistribution.resolve(values, distribution, 1981, 1981, 2010, compute.Periodicity.monthly)
    expected = compute.fit_and_standardize(
        values, distribution, 1981, 1981, 2010, compute.Periodicity.monthly, output_scale=output_scale
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("A resolved fit must not re-enter a public adapter or resolve again")

    for kind in _FITTED_DISTRIBUTIONS:
        monkeypatch.setattr(compute, f"transform_fitted_{kind.value}", fail)
    monkeypatch.setattr(compute.FittedDistribution, "resolve", fail)
    monkeypatch.setattr(compute.FittedDistribution, "validate_supplied", fail)

    with mock.patch.object(compute._logger, "bind") as bind:
        np.testing.assert_array_equal(fitted.transform(values, output_scale), expected)
    bind.assert_called_once()
    assert bind.return_value.info.call_args_list == [
        mock.call("distribution_transform_started"),
        mock.call("distribution_transform_completed", output_shape=str(values.shape)),
    ]


@pytest.mark.parametrize("distribution", _FITTED_DISTRIBUTIONS)
@pytest.mark.parametrize("supplied", [False, True])
def test_public_adapter_resolves_validates_and_broadcasts_once(
    distribution: indices.Distribution, supplied: bool
) -> None:
    """A public transform resolves one fit and broadcasts it once across cells."""
    values = np.random.default_rng(1417).uniform(0.1, 0.9, (30, 12, 2))
    adapter = getattr(compute, f"transform_fitted_{distribution.value}")
    params = {}
    if supplied:
        fitted = compute.FittedDistribution.resolve(
            values[..., 0], distribution, 1981, 1981, 2010, compute.Periodicity.monthly
        )
        names = {
            "alpha": "alphas",
            "beta": "betas",
            "prob_zero": "probabilities_of_zero",
            "prob_one": "probabilities_of_one",
            "loc": "locs",
            "scale": "scales",
            "skew": "skews",
            "shape": "shapes",
        }
        params = {names[key]: value for key, value in fitted.parameters.items()}
    with (
        mock.patch.object(compute.FittedDistribution, "resolve", wraps=compute.FittedDistribution.resolve) as resolve,
        mock.patch.object(
            compute.FittedDistribution, "validate_supplied", wraps=compute.FittedDistribution.validate_supplied
        ) as validate,
        mock.patch.object(compute, "_broadcast_parameters", wraps=compute._broadcast_parameters) as broadcast,
    ):
        result = adapter(values, 1981, 1981, 2010, compute.Periodicity.monthly, **params)

    assert result.shape == values.shape
    assert resolve.call_count == 1
    assert validate.call_count == 1
    assert broadcast.call_count == 1


@pytest.mark.parametrize("distribution", _FITTED_DISTRIBUTIONS)
def test_all_missing_adapter_keeps_its_return_shape(distribution: indices.Distribution) -> None:
    """Parametric adapters return the original array; empirical retains its folded shape."""
    values = np.full(120, np.nan)
    adapter = getattr(compute, f"transform_fitted_{distribution.value}")
    result = adapter(values, 2000, 2000, 2009, compute.Periodicity.monthly)
    assert np.isnan(result).all()
    if distribution is indices.Distribution.empirical:
        assert result.shape == (10, 12)
    else:
        assert result is values


def test_gamma_missing_adapter_still_skips_supplied_parameter_validation() -> None:
    """Gamma's established all-missing return precedes its parameter validation."""
    values = np.full(120, np.nan)
    assert (
        compute.transform_fitted_gamma(
            values, 2000, 2000, 2009, compute.Periodicity.monthly, probabilities_of_zero=np.full(12, -1.0)
        )
        is values
    )


def test_empirical_fit_keeps_its_calibration_sample() -> None:
    """New observations rank against the original Calibration Period, including ties."""
    calibration = np.tile(np.arange(10.0)[:, None], (1, 12))
    fitted = compute.FittedDistribution.resolve(
        calibration, indices.Distribution.empirical, 2000, 2000, 2009, compute.Periodicity.monthly
    )
    values = np.tile(np.array([-1.0, 0.0, 4.5, 9.0, 10.0, np.nan])[:, None], (1, 12))
    # The SEI plotting position is (1 + count(sample <= value)) / (n + 2).
    expected = np.tile(np.array([1.0, 2.0, 6.0, 11.0, 11.0, np.nan])[:, None] / 12.0, (1, 12))
    np.testing.assert_array_equal(fitted.transform(values, "probability"), expected)
    assert fitted.parameters == {}


@pytest.mark.parametrize("zero_handling", ["classic", "center_of_mass", "mean_zero"])
def test_gamma_fit_reuses_parameters_for_zero_handling(zero_handling: compute.ZeroHandling) -> None:
    """The fit retains its parameters while each transform selects Zero Handling."""
    values = np.tile(np.array([0.0, 1.0, 2.0])[:, None], (1, 12))
    params = {"alpha": np.ones(12), "beta": np.ones(12), "prob_zero": np.full(12, 0.4)}
    fitted = compute.FittedDistribution.resolve(
        values, indices.Distribution.gamma, 2000, 2000, 2002, compute.Periodicity.monthly, params
    )
    probabilities = 0.4 + 0.6 * (1.0 - np.exp(-values))
    expected = scipy.stats.norm.ppf(probabilities)
    if zero_handling == "center_of_mass":
        expected[0] = scipy.stats.norm.ppf(0.2)
    elif zero_handling == "mean_zero":
        expected[0] = -scipy.stats.norm.pdf(scipy.stats.norm.ppf(0.4)) / 0.4
    # The exponential CDF is gamma(alpha=1, beta=1); f64 roundoff needs 1e-14.
    np.testing.assert_allclose(fitted.transform(values, zero_handling=zero_handling), expected, rtol=1e-14, atol=1e-14)
    for key in params:
        np.testing.assert_array_equal(fitted.parameters[key], params[key])


def test_empirical_period_only_sample_aligns_with_the_period_axis() -> None:
    """A reused period-only sample ranks every cell against its own calendar step."""
    calibration = np.tile(np.arange(12.0), (10, 1))
    fitted = compute.FittedDistribution.resolve(
        calibration, indices.Distribution.empirical, 2000, 2000, 2009, compute.Periodicity.monthly
    )
    result = fitted.transform(np.zeros((1, 12, 12)), "probability")
    # January's sample is ten zeros, so k = 10 and p = 11 / 12; every later step's
    # sample starts at 1, so k = 0 and p = 1 / 12. Cells share each step's sample.
    per_step = (1.0 + np.array([10.0] + [0.0] * 11)) / 12.0
    np.testing.assert_array_equal(result, np.broadcast_to(per_step[None, :, None], (1, 12, 12)))


def test_empirical_fit_rejects_mismatched_sample_cells() -> None:
    """A sample fitted on two cells must not be reused on one or three."""
    calibration = np.random.default_rng(1417).uniform(0.0, 1.0, (10, 12, 2))
    fitted = compute.FittedDistribution.resolve(
        calibration, indices.Distribution.empirical, 2000, 2000, 2009, compute.Periodicity.monthly
    )
    for shape in ((1, 12, 1), (1, 12, 3)):
        with pytest.raises(ValueError, match="cell dimensions"):
            fitted.transform(np.zeros(shape), "probability")


@pytest.mark.parametrize(
    "distribution", [indices.Distribution.pearson, indices.Distribution.beta, indices.Distribution.loglogistic]
)
@pytest.mark.parametrize("value", [np.nan, "bad"])
def test_reused_fit_checks_cells_before_missing_detection(distribution: indices.Distribution, value: object) -> None:
    """Resolved parameter shape errors precede all-missing returns and invalid dtypes."""
    values = np.random.default_rng(1417).uniform(0.1, 0.9, (30, 12, 2))
    fitted = compute.FittedDistribution.resolve(values, distribution, 1981, 1981, 2010, compute.Periodicity.monthly)
    with pytest.raises(ValueError, match="cell dimensions|does not broadcast"):
        fitted.transform(np.full((1, 12, 1), value))


def test_beta_support_errors_precede_inferred_boundary_mass_errors() -> None:
    """Invalid observations fail before a missing boundary mass is counted."""
    values = np.tile(np.array([1.0, 1.5])[:, None], (1, 12))
    with pytest.raises(InvalidArgumentError, match="supports only values"):
        compute.transform_fitted_beta(
            values,
            2000,
            2000,
            2001,
            compute.Periodicity.monthly,
            alphas=np.full(12, 2.0),
            betas=np.full(12, 3.0),
            probabilities_of_zero=np.full(12, 0.75),
        )


def test_supplied_beta_parameter_errors_precede_support_errors() -> None:
    """A malformed supplied mass is reported before an out-of-support observation."""
    values = np.tile(np.array([1.0, 1.5])[:, None], (1, 12))
    with pytest.raises(ValueError, match="must lie in"):
        compute.transform_fitted_beta(
            values,
            2000,
            2000,
            2001,
            compute.Periodicity.monthly,
            probabilities_of_zero=np.full(12, -0.1),
        )


def test_empirical_masked_partial_year_is_prepared_once() -> None:
    """Padding must not discard calibration masks or retain a second filled record."""
    values = np.ma.array(np.arange(121.0), mask=np.arange(121) == 0)
    with (
        mock.patch.object(np.ma, "filled", wraps=np.ma.filled) as filled,
        mock.patch.object(compute, "_empirical_fit", wraps=compute._empirical_fit) as kernel,
    ):
        result = compute.fit_and_standardize(
            values, indices.Distribution.empirical, 2000, 2000, 2009, compute.Periodicity.monthly
        )
    assert result.shape == (11, 12)
    # January has nine valid calibration observations, below the empirical minimum.
    assert np.isnan(result[:, 0]).all()
    assert filled.call_count == 1
    prepared, calibration, _ = kernel.call_args.args
    assert np.isnan(prepared[0, 0])
    assert np.shares_memory(prepared, calibration)


@pytest.mark.parametrize(
    "distribution", [kind for kind in _FITTED_DISTRIBUTIONS if kind is not indices.Distribution.empirical]
)
def test_reused_fit_rejects_incompatible_cells(distribution: indices.Distribution) -> None:
    """A two-cell fit must not expand an input that carries only one cell."""
    values = np.random.default_rng(1417).uniform(0.1, 0.9, (30, 12, 2))
    fitted = compute.FittedDistribution.resolve(values, distribution, 1981, 1981, 2010, compute.Periodicity.monthly)
    with pytest.raises(ValueError, match="cell dimensions|does not broadcast"):
        fitted.transform(np.full((1, 12, 1), 0.5))


@pytest.mark.parametrize(
    "distribution", [kind for kind in _FITTED_DISTRIBUTIONS if kind is not indices.Distribution.gamma]
)
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"output_scale": "bad"}, InvalidArgumentError), ({"zero_handling": "bad"}, ValueError)],
)
def test_reused_fit_checks_arguments_before_cells(
    distribution: indices.Distribution, kwargs: dict[str, str], error: type[Exception]
) -> None:
    """An invalid argument fails before a cell mismatch, as in the public adapters."""
    values = np.random.default_rng(1417).uniform(0.1, 0.9, (30, 12, 2))
    fitted = compute.FittedDistribution.resolve(values, distribution, 1981, 1981, 2010, compute.Periodicity.monthly)
    with pytest.raises(error, match=next(iter(kwargs))):
        fitted.transform(np.full((1, 12, 1), 0.5), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("distribution", "params"),
    [
        (indices.Distribution.pearson, {"locs": np.ones(12)}),
        (indices.Distribution.loglogistic, {"locs": np.ones(12)}),
        (indices.Distribution.beta, {"alphas": np.ones(11)}),
    ],
)
def test_adapter_parameter_errors_precede_unsupported_dtype(
    distribution: indices.Distribution, params: dict[str, np.ndarray]
) -> None:
    """Non-gamma adapters report malformed parameters before calling isnan on strings."""
    adapter = getattr(compute, f"transform_fitted_{distribution.value}")
    with pytest.raises(ValueError, match="either none or all|does not broadcast"):
        adapter(np.full(12, "bad"), 2000, 2000, 2009, compute.Periodicity.monthly, **params)


@pytest.mark.parametrize("pipeline", [False, True])
def test_supplied_beta_transform_checks_support_once(pipeline: bool) -> None:
    """A supplied beta fit needs one full-array support scan, not two."""
    values = np.full((30, 12), 0.5)
    parameters = {
        "alpha": np.full(12, 2.0),
        "beta": np.full(12, 3.0),
        "prob_zero": np.full(12, 0.2),
        "prob_one": np.full(12, 0.1),
    }
    with mock.patch.object(compute, "_validate_beta_support", wraps=compute._validate_beta_support) as support:
        if pipeline:
            compute.fit_and_standardize(
                values, indices.Distribution.beta, 1981, 1981, 2010, compute.Periodicity.monthly, parameters
            )
        else:
            compute.transform_fitted_beta(
                values,
                1981,
                1981,
                2010,
                compute.Periodicity.monthly,
                parameters["alpha"],
                parameters["beta"],
                parameters["prob_zero"],
                parameters["prob_one"],
            )
    assert support.call_count == 1


def _monthly_values() -> np.ndarray:
    return np.arange(1.0, 481.0).reshape(40, 12)


def test_resolve_from_data_transforms_like_the_public_transform() -> None:
    """A fit built from data transforms its values like the compatibility adapter."""
    values = _monthly_values()
    fitted = compute.FittedDistribution.resolve(
        values, indices.Distribution.gamma, 1981, 1981, 2010, compute.Periodicity.monthly
    )

    np.testing.assert_array_equal(
        fitted.transform(values),
        compute.transform_fitted_gamma(values, 1981, 1981, 2010, compute.Periodicity.monthly),
    )


def test_transform_and_diagnostics_share_one_seam() -> None:
    """The transform and diagnostics methods compose over the same resolved fit."""
    values = _monthly_values()
    fitted = compute.FittedDistribution.resolve(
        values, indices.Distribution.pearson, 1981, 1981, 2010, compute.Periodicity.monthly
    )

    np.testing.assert_array_equal(
        fitted.transform(values),
        compute.transform_fitted_pearson(values, 1981, 1981, 2010, compute.Periodicity.monthly),
    )
    ks, p_value, n_valid, parameters = fitted.diagnostics(values[:30])
    diagnostics = compute.fit_diagnostics(
        values, indices.Distribution.pearson, 1981, 1981, 2010, compute.Periodicity.monthly
    )
    np.testing.assert_array_equal(ks, diagnostics.ks_statistic)
    np.testing.assert_array_equal(p_value, diagnostics.ks_p_value)
    np.testing.assert_array_equal(n_valid, diagnostics.n_valid)
    assert set(parameters) == set(diagnostics.parameters)
    for name, parameter in parameters.items():
        np.testing.assert_array_equal(parameter, diagnostics.parameters[name])


def test_pearson_fallback_decision_is_shared_by_index_and_diagnostics() -> None:
    """The fall-back decision agrees between ``fit_and_standardize`` and ``fit_diagnostics``.

    A zero-heavy block whose failed Pearson steps still place their sub-0.0005 values
    must not be reported as a gamma fall back when the index path keeps Pearson (#1216).
    """
    rng = np.random.default_rng(7)
    values = rng.gamma(2.0, 2.0, size=(40, 12))
    # seven calendar steps have only two non-zero calibration values, so their Pearson
    # fit fails and every value below 0.0005 is placed rather than lost
    values[:, :7] = 0.0
    values[::20, :7] = 1.0

    diagnostics = compute.fit_diagnostics(
        values,
        indices.Distribution.pearson,
        1981,
        1981,
        2010,
        compute.Periodicity.monthly,
        fallback_to_gamma=True,
    )
    standardized = compute.fit_and_standardize(
        values,
        indices.Distribution.pearson,
        1981,
        1981,
        2010,
        compute.Periodicity.monthly,
        fallback_to_gamma=True,
    )
    # compare against the Pearson transform: equality means no fall back happened
    pearson_only = compute.transform_fitted_pearson(values, 1981, 1981, 2010, compute.Periodicity.monthly)
    stayed_pearson = bool(np.array_equal(standardized, pearson_only, equal_nan=True))

    assert stayed_pearson
    assert diagnostics.fell_back_to_gamma is False


def test_resolve_from_supplied_parameters_does_not_refit(monkeypatch) -> None:
    """A complete supplied parameter set is used as-is rather than fitted again."""
    values = _monthly_values()
    alphas = np.full(12, 4.0)
    betas = np.full(12, 8.0)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("gamma_parameters should not be called")

    monkeypatch.setattr(compute, "gamma_parameters", fail)
    fitted = compute.FittedDistribution.resolve(
        values,
        indices.Distribution.gamma,
        1981,
        1981,
        2010,
        compute.Periodicity.monthly,
        {"alpha": alphas, "beta": betas, "prob_zero": np.zeros(12)},
    )

    np.testing.assert_array_equal(fitted.parameters["alpha"], alphas)
    np.testing.assert_array_equal(fitted.parameters["beta"], betas)


def test_resolve_rejects_a_partial_pearson_set() -> None:
    """A partial Pearson set is one argument error across every surface."""
    values = _monthly_values()

    with pytest.raises(ValueError, match="either none or all"):
        compute.FittedDistribution.resolve(
            values,
            indices.Distribution.pearson,
            1981,
            1981,
            2010,
            compute.Periodicity.monthly,
            {"loc": np.ones(12)},
        )


def test_resolve_rejects_a_mis_shaped_pearson_set() -> None:
    """A Pearson parameter that does not carry the period axis is rejected before any fit."""
    values = _monthly_values()
    parameters = {
        "prob_zero": np.zeros(11),
        "loc": np.ones(11),
        "scale": np.ones(11),
        "skew": np.ones(11),
    }

    with pytest.raises(ValueError, match="must carry the period length"):
        compute.FittedDistribution.resolve(
            values, indices.Distribution.pearson, 1981, 1981, 2010, compute.Periodicity.monthly, parameters
        )

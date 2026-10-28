"""Behavior checks for the shared Pearson fallback and poor-fit reporting policy."""

import logging
import warnings
from unittest import mock

import numpy as np
import pytest

from climate_indices import compute
from climate_indices.exceptions import DistributionFittingError, GoodnessOfFitWarning


@pytest.mark.parametrize("lost_steps", [0, 6, 7])
@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("output_scale", "zero_handling"),
    [("normal", "classic"), ("probability", "center_of_mass"), ("bounded", "mean_zero")],
)
def test_fallback_reports_the_same_distribution_on_both_surfaces(
    lost_steps: int,
    spatial: bool,
    dtype: type[np.float32] | type[np.float64],
    output_scale: compute.OutputScale,
    zero_handling: compute.ZeroHandling,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only newly missing values count, and exactly half does not trigger fallback."""
    values = np.random.default_rng(1418).gamma(2.0, 2.0, (40, 12)).astype(dtype)
    values[:8] = np.nan
    if spatial:
        values = np.stack([values, values], axis=-1)
    params = {
        "prob_zero": np.zeros(12, dtype=dtype),
        "loc": np.ones(12, dtype=dtype),
        "scale": np.ones(12, dtype=dtype),
        "skew": np.zeros(12, dtype=dtype),
    }
    params["scale"][:lost_steps] = 0.0
    args = (compute.Distribution.pearson, 1981, 1981, 2020, compute.Periodicity.monthly, params)
    caplog.set_level(logging.WARNING)

    with np.errstate(all="ignore"):
        result = compute.fit_and_standardize(
            values,
            *args,
            fallback_to_gamma=True,
            fallback_context="policy test",
            output_scale=output_scale,
            zero_handling=zero_handling,
        )
        with mock.patch.object(compute.FittedDistribution, "transform", side_effect=AssertionError("transform")):
            diagnostics = compute.fit_diagnostics(values, *args, fallback_to_gamma=True, fallback_context="policy test")

    fallback = lost_steps > 6
    expected_distribution = compute.Distribution.gamma if fallback else compute.Distribution.pearson
    assert diagnostics.distribution is expected_distribution
    assert diagnostics.fell_back_to_gamma is fallback
    events = [
        record.msg
        for record in caplog.records
        if isinstance(record.msg, dict) and record.msg.get("event") == "distribution_fallback"
    ]
    assert len(events) == (2 if fallback else 0)
    for event in events:
        assert event["from_distribution"] == "pearson"
        assert event["to_distribution"] == "gamma"
        assert event["context"] == "policy test"
    with np.errstate(all="ignore"):
        expected = compute.fit_and_standardize(
            values,
            expected_distribution,
            1981,
            1981,
            2020,
            compute.Periodicity.monthly,
            None if fallback else params,
            output_scale=output_scale,
            zero_handling=zero_handling,
        )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_fit_failure_is_reported_or_propagated(
    diagnostics: bool, fallback: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """A genuine fit failure falls back only when requested, with one public event."""
    values = np.random.default_rng(1418).gamma(2.0, 2.0, (40, 12))
    surface = compute.fit_diagnostics if diagnostics else compute.fit_and_standardize
    args = (values, compute.Distribution.pearson, 1981, 1981, 2020, compute.Periodicity.monthly)
    caplog.set_level(logging.WARNING)
    failure = DistributionFittingError("Pearson fit failed", distribution_name="pearson3")
    with mock.patch.object(compute, "pearson_parameters", side_effect=failure):
        if fallback:
            result = surface(*args, fallback_to_gamma=True)
            if diagnostics:
                assert isinstance(result, compute.FitDiagnostics)
                assert result.distribution is compute.Distribution.gamma
                assert result.fell_back_to_gamma
            else:
                expected = compute.fit_and_standardize(
                    values, compute.Distribution.gamma, 1981, 1981, 2020, compute.Periodicity.monthly
                )
                np.testing.assert_array_equal(result, expected)
        else:
            with pytest.raises(DistributionFittingError) as caught:
                surface(*args)
            assert caught.value is failure
    events = [
        record.msg
        for record in caplog.records
        if isinstance(record.msg, dict) and record.msg.get("event") == "distribution_fallback"
    ]
    assert len(events) == int(fallback)
    if events:
        assert events[0]["reason"] == str(failure)


@pytest.mark.parametrize(("stage", "fallback"), [("transform", False), ("transform", True), ("loss_measurement", True)])
def test_post_fit_errors_propagate_without_fallback(fallback: bool, stage: str) -> None:
    """Only fit resolution errors can trigger Gamma fallback."""
    values = np.random.default_rng(1418).gamma(2.0, 2.0, (40, 12))
    failure = DistributionFittingError("Post-fit failure", distribution_name="pearson3")
    target = compute.FittedDistribution if stage == "transform" else compute
    name = "transform" if stage == "transform" else "_pearson_lost_valid_fraction"
    surface = compute.fit_and_standardize if stage == "transform" else compute.fit_diagnostics
    with (
        mock.patch.object(target, name, side_effect=failure) as operation,
        mock.patch.object(compute.FittedDistribution, "resolve", wraps=compute.FittedDistribution.resolve) as resolve,
        mock.patch.object(compute._default_fallback_strategy, "log_fallback_warning") as log_fallback,
    ):
        with pytest.raises(DistributionFittingError) as caught:
            surface(
                values,
                compute.Distribution.pearson,
                1981,
                1981,
                2020,
                compute.Periodicity.monthly,
                fallback_to_gamma=fallback,
            )
        assert caught.value is failure
        operation.assert_called_once()
    resolve.assert_called_once()
    log_fallback.assert_not_called()


def test_standardization_rejects_a_missing_transform_result() -> None:
    """The transform result check must remain active when Python optimization is enabled."""
    values = np.ones((40, 12))
    with (
        mock.patch.object(compute, "_resolve_pearson_with_fallback", return_value=(None, None)),
        pytest.raises(RuntimeError, match="Pearson standardization returned no transformed values"),
    ):
        compute.fit_and_standardize(values, compute.Distribution.pearson, 1981, 1981, 2020, compute.Periodicity.monthly)


@pytest.mark.parametrize("distribution", [compute.Distribution.gamma, compute.Distribution.pearson])
@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("steps", [1, 5, 6, 12])
def test_poor_fit_warning_aggregates_scalar_and_spatial_results(
    distribution: compute.Distribution, spatial: bool, steps: int
) -> None:
    """Each fit emits one structured warning, independent of the number of poor steps."""
    # Two separated clusters give a poor fit for both continuous distributions.
    values = np.full((60, 12), np.nan)
    values[:, :steps] = np.repeat([0.1, 100.0], 30)[:, None]
    if spatial:
        values = np.stack([values, values], axis=-1)
    fit = compute.gamma_parameters if distribution is compute.Distribution.gamma else compute.pearson_parameters
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", GoodnessOfFitWarning)
        fit(values, 1981, 1981, 2040, compute.Periodicity.monthly)
    fit_warnings = [record.message for record in recorded if record.category is GoodnessOfFitWarning]
    assert len(fit_warnings) == 1
    warning = fit_warnings[0]
    assert isinstance(warning, GoodnessOfFitWarning)
    assert warning.distribution_name == ("gamma" if distribution is compute.Distribution.gamma else "pearson3")
    assert warning.threshold == compute.GOODNESS_OF_FIT_P_VALUE_THRESHOLD
    assert warning.poor_fit_count == steps * (2 if spatial else 1)
    assert warning.total_steps == 12 * (2 if spatial else 1)

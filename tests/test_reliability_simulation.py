"""Contract tests for the seeded Monte Carlo reliability simulation (ENERGY2-9, #1343).

The analytic SAIDI reference uses one initially operational component with
exponential failures and repairs. Its failure-onset intensity is ``rate * P_up(t)``.
For each onset, the expected clipped repair duration contributes only if it exceeds
five minutes. The quadrature therefore measures sustained downtime, not all downtime.
A second reference uses a continuous-time Markov chain for two components in
series, with the same threshold and horizon clipping.
The convergence tolerance is three standard errors of the trial mean. A separate
test checks that the standard error falls at the ``1 / sqrt(n)`` rate.
Trials are sampled in order from one generator, so a shorter run is an exact
prefix of a longer one with the same seed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from scipy import integrate, linalg

from climate_indices import reliability
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.reliability import _simulate

HORIZON_HOURS = 8760.0
TOTAL_CUSTOMERS = 1000


def component(
    component_id: str = "c1",
    failure_rate: float = 0.01,
    repair_time: float = 4.0,
    **changes: object,
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "failure_rate": failure_rate,
        "repair_time": repair_time,
        **changes,
    }


def simulate(
    components: Sequence[Mapping[str, Any]],
    *,
    arrangement: str = "series",
    n_trials: int = 50,
    seed: Any = 11,
    **changes: Any,
) -> tuple[reliability.ReliabilityResult, ...]:
    kwargs: dict[str, Any] = {
        "arrangement": arrangement,
        "total_customers": TOTAL_CUSTOMERS,
        "horizon_hours": HORIZON_HOURS,
        "n_trials": n_trials,
        "seed": seed,
    }
    kwargs.update(changes)
    return reliability.simulate_reliability(components, **kwargs)


def _availability(failure_rate: float, mean_repair_hours: float) -> tuple[float, float]:
    """Return the steady-state availability and the approach rate for one component."""
    repair_rate = 1.0 / mean_repair_hours
    availability = repair_rate / (failure_rate + repair_rate)
    return availability, failure_rate + repair_rate


def _analytic_saidi_minutes(failure_rate: float, mean_repair_hours: float, horizon_hours: float) -> float:
    """Integrate sustained downtime for one component with exponential repairs."""
    threshold_hours = 5.0 / 60.0
    if horizon_hours <= threshold_hours:
        return 0.0
    steady, approach = _availability(failure_rate, mean_repair_hours)

    def integrand(hour: float) -> float:
        up = steady + (1.0 - steady) * math.exp(-approach * hour)
        remaining = horizon_hours - hour
        # E[min(repair, remaining) * I(min(repair, remaining) > threshold)].
        sustained_hours = (threshold_hours + mean_repair_hours) * math.exp(
            -threshold_hours / mean_repair_hours
        ) - mean_repair_hours * math.exp(-remaining / mean_repair_hours)
        return failure_rate * up * sustained_hours

    downtime_hours = float(integrate.quad(integrand, 0.0, horizon_hours - threshold_hours, limit=200)[0])
    return downtime_hours * 60.0


def test_same_seed_reproduces_identical_results() -> None:
    components = [component("c1", 0.01, 4.0), component("c2", 0.005, 2.0)]
    first = simulate(components, n_trials=5, seed=1234)
    second = simulate(components, n_trials=5, seed=1234)
    assert first == second
    assert len(first) == 5
    for left, right in zip(first, second, strict=True):
        assert left.saifi == right.saifi
        assert left.saidi == right.saidi
        assert left.customer_interruptions == right.customer_interruptions


def test_different_seeds_produce_different_results() -> None:
    components = [component("c1", 0.01, 4.0)]
    assert simulate(components, n_trials=5, seed=1) != simulate(components, n_trials=5, seed=2)


def test_seed_is_required() -> None:
    components = [component()]
    with pytest.raises(InvalidArgumentError) as error:
        reliability.simulate_reliability(
            components,
            arrangement="series",
            total_customers=TOTAL_CUSTOMERS,
            horizon_hours=HORIZON_HOURS,
            n_trials=1,
        )
    assert error.value.argument_name == "seed"
    with pytest.raises(InvalidArgumentError) as error:
        simulate(components, n_trials=1, seed=None)
    assert error.value.argument_name == "seed"


def test_public_surface_is_exported_without_extras() -> None:
    assert reliability.simulate_reliability.__name__ == "simulate_reliability"
    assert "simulate_reliability" in reliability.__all__


def test_trials_are_ordered_by_trial_index() -> None:
    runs = simulate([component()], n_trials=3, seed=7)
    assert runs[:2] == simulate([component()], n_trials=2, seed=7)
    assert all(isinstance(result, reliability.ReliabilityResult) for result in runs)


@pytest.mark.parametrize("repair_time", [0.05, 4.0])
@pytest.mark.parametrize("arrangement", ["series", "parallel"])
def test_mean_converges_to_sustained_analytic_expectation(repair_time: float, arrangement: str) -> None:
    components = [component("c1", 0.01, repair_time)]
    n_trials = 2000
    runs = simulate(components, arrangement=arrangement, n_trials=n_trials, seed=20240101)
    saidi = np.array([result.saidi for result in runs], dtype=np.float64)
    analytic = _analytic_saidi_minutes(0.01, repair_time, HORIZON_HOURS)
    # Three standard errors of the trial mean: a ~99.7 percent Monte Carlo band.
    standard_error = float(np.std(saidi, ddof=1) / math.sqrt(n_trials))
    assert abs(float(np.mean(saidi)) - analytic) <= 3.0 * standard_error


def _analytic_series_saidi_minutes(
    failure_rates: tuple[float, float], mean_repair_hours: tuple[float, float], horizon_hours: float
) -> float:
    """Integrate sustained downtime for two independent exponential components."""
    first_rate, second_rate = failure_rates
    first_repair, second_repair = (1.0 / mean for mean in mean_repair_hours)
    # Transient states: first down, second down, both down; both up is absorbing.
    transition = np.array(
        [
            [-(first_repair + second_rate), 0.0, second_rate],
            [0.0, -(second_repair + first_rate), first_rate],
            [second_repair, first_repair, -(first_repair + second_repair)],
        ]
    )
    onset_rate = first_rate + second_rate
    initial = np.array([first_rate, second_rate, 0.0]) / onset_rate
    threshold_hours = 5.0 / 60.0
    at_threshold = linalg.expm(transition * threshold_hours)
    sustained_threshold = threshold_hours * float(initial @ at_threshold @ np.ones(3))
    availability = [_availability(rate, mean) for rate, mean in zip(failure_rates, mean_repair_hours, strict=True)]

    def integrand(hour: float) -> float:
        both_up = math.prod(steady + (1.0 - steady) * math.exp(-approach * hour) for steady, approach in availability)
        remaining = horizon_hours - hour
        # E[min(D, remaining) * I(min(D, remaining) > threshold)] for system repair D.
        tail_integral = linalg.solve(transition, (linalg.expm(transition * remaining) - at_threshold) @ np.ones(3))
        sustained_hours = sustained_threshold + float(initial @ tail_integral)
        return onset_rate * both_up * sustained_hours

    return 60.0 * float(integrate.quad(integrand, 0.0, horizon_hours - threshold_hours, limit=200)[0])


def test_series_analytic_reference_reduces_to_one_failing_component() -> None:
    analytic = _analytic_series_saidi_minutes((0.01, 0.0), (4.0, 2.0), 24.0)
    assert analytic == pytest.approx(_analytic_saidi_minutes(0.01, 4.0, 24.0), rel=1e-12)


def test_two_component_series_mean_converges_to_sustained_analytic_expectation() -> None:
    components = [component("c1", 0.01, 4.0), component("c2", 0.005, 2.0)]
    runs = simulate(components, n_trials=2000, seed=20240101)
    saidi = np.array([result.saidi for result in runs], dtype=np.float64)
    analytic = _analytic_series_saidi_minutes((0.01, 0.005), (4.0, 2.0), HORIZON_HOURS)
    standard_error = float(np.std(saidi, ddof=1) / math.sqrt(saidi.size))
    assert abs(float(np.mean(saidi)) - analytic) <= 3.0 * standard_error


def test_standard_error_falls_with_trial_count() -> None:
    components = [component("c1", 0.01, 4.0), component("c2", 0.005, 2.0)]
    small = np.array([result.saidi for result in simulate(components, n_trials=250, seed=99)])
    large = np.array([result.saidi for result in simulate(components, n_trials=2000, seed=99)])
    small_error = float(np.std(small, ddof=1) / math.sqrt(small.size))
    large_error = float(np.std(large, ddof=1) / math.sqrt(large.size))
    assert large_error < 0.5 * small_error


def test_parallel_system_downtime_does_not_exceed_series() -> None:
    components = [component("c1", 0.01, 4.0), component("c2", 0.005, 2.0)]
    series = simulate(components, arrangement="series", n_trials=200, seed=5150)
    parallel = simulate(components, arrangement="parallel", n_trials=200, seed=5150)
    for series_result, parallel_result in zip(series, parallel, strict=True):
        assert parallel_result.saidi <= series_result.saidi


def test_parallel_can_have_more_interruptions_than_series(monkeypatch: pytest.MonkeyPatch) -> None:
    intervals = {"c1": [(1.0, 10.0)], "c2": [(2.0, 3.0), (4.0, 5.0)]}
    monkeypatch.setattr(_simulate, "_down_intervals", lambda spec, horizon, rng: intervals[spec.component_id])
    components = [component("c1"), component("c2")]
    (series,) = simulate(components, arrangement="series", n_trials=1)
    (parallel,) = simulate(components, arrangement="parallel", n_trials=1)
    assert series.saidi == 540.0
    assert parallel.saidi == 120.0
    assert series.customer_interruptions == TOTAL_CUSTOMERS
    assert parallel.customer_interruptions == 2 * TOTAL_CUSTOMERS


@pytest.mark.parametrize(
    ("intervals", "saidi", "momentary_events"),
    [
        ([(1.0, 1.05)], 0.0, 1.0),
        ([(0.0, 5.0 / 60.0)], 0.0, 1.0),
        ([(1.0, 1.5)], 30.0, 0.0),
    ],
)
def test_simulated_intervals_use_default_classification(
    monkeypatch: pytest.MonkeyPatch,
    intervals: list[tuple[float, float]],
    saidi: float,
    momentary_events: float,
) -> None:
    monkeypatch.setattr(_simulate, "_down_intervals", lambda spec, horizon, rng: intervals)
    (result,) = simulate([component()], n_trials=1)
    assert result.saidi == saidi
    assert result.maifi_e == momentary_events
    assert result.saifi == (1.0 if saidi > 0.0 else 0.0)
    assert result.customer_interruptions == (TOTAL_CUSTOMERS if saidi > 0.0 else 0)


def test_horizon_clipped_repairs_are_momentary() -> None:
    spec = component(failure_rate=1000.0, repair_time=1.0, repair_distribution="fixed")
    (result,) = simulate([spec], n_trials=1, horizon_hours=0.05)
    assert result.outage_count == 0
    assert result.maifi_e == 1.0
    assert result.saidi == 0.0
    assert result.saifi == 0.0
    assert result.customer_interruptions == 0
    assert _analytic_saidi_minutes(1000.0, 1.0, 0.05) == 0.0


def test_fixed_three_minute_repairs_are_momentary() -> None:
    spec = component(failure_rate=1000.0, repair_time=0.05, repair_distribution="fixed")
    (result,) = simulate([spec], n_trials=1, horizon_hours=1.0)
    assert result.outage_count == 0
    assert result.maifi_e > 0.0
    assert result.saidi == 0.0
    assert result.saifi == 0.0
    assert result.customer_interruptions == 0


@pytest.mark.parametrize(
    ("distribution", "changes"),
    [
        ("fixed", {}),
        ("exponential", {}),
        ("lognormal", {"repair_sigma": 0.5}),
    ],
)
def test_repair_distribution_mean_matches_requested_mean(distribution: str, changes: dict[str, Any]) -> None:
    spec = component("c1", 0.005, 24.0, repair_distribution=distribution, **changes)
    runs = simulate([spec], n_trials=400, seed=606)
    caidi = np.array([result.caidi for result in runs], dtype=np.float64)
    # CAIDI is the mean realized restoration time; 24 hours is 1440 minutes.
    assert float(np.mean(caidi)) == pytest.approx(1440.0, rel=0.03)


@pytest.mark.parametrize("failure_rate", [0.0, 0.01])
@pytest.mark.parametrize(("repair_time", "sigma"), [(4.0, 40.0), (4.0, 1e308), (1e-300, 20.0)])
def test_unrepresentable_lognormal_scale_is_rejected(failure_rate: float, repair_time: float, sigma: float) -> None:
    spec = component(
        failure_rate=failure_rate, repair_time=repair_time, repair_distribution="lognormal", repair_sigma=sigma
    )
    with pytest.raises(InvalidArgumentError) as error:
        simulate([spec], n_trials=1)
    assert error.value.argument_name == "repair_sigma"


def test_lognormal_scale_avoids_intermediate_underflow() -> None:
    spec = _simulate._component_spec(
        component(repair_time=1e300, repair_distribution="lognormal", repair_sigma=40.0), "c1"
    )
    distribution = _simulate._repair_distribution(spec)
    assert distribution is not None
    assert distribution.kwds["scale"] == pytest.approx(math.exp(math.log(1e300) - 800.0), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("failure_rate", [0.0, 0.01])
def test_customer_count_above_int64_is_rejected_before_sampling(failure_rate: float) -> None:
    spec = component(failure_rate=failure_rate)
    with pytest.raises(InvalidArgumentError) as error:
        simulate([spec], n_trials=1, total_customers=2**63)
    assert error.value.argument_name == "total_customers"


def test_largest_float64_customer_count_within_int64_is_accepted() -> None:
    customers = int(np.nextafter(float(2**63), 0.0))
    (result,) = simulate([component()], n_trials=1, total_customers=customers)
    assert result.customer_interruptions > 0
    assert result.customer_interruptions % customers == 0


def test_empty_trials_report_defined_zero_metrics() -> None:
    runs = simulate([component("c1", 1e-12, 4.0)], n_trials=4, seed=13)
    assert len(runs) == 4
    for result in runs:
        assert result.saifi == 0.0
        assert result.saidi == 0.0
        assert result.customer_interruptions == 0
        assert math.isnan(result.caidi)
        assert result.undefined_metrics == ("caidi", "maifi")


def test_component_topology_is_rejected_as_deferred() -> None:
    components = [component()]
    with pytest.raises(InvalidArgumentError) as error:
        simulate(components, n_trials=1, component_topology={"c1": ["c2"]})
    assert error.value.argument_name == "component_topology"


@pytest.mark.parametrize(
    ("changes", "argument_name"),
    [
        ({"n_trials": 0}, "n_trials"),
        ({"n_trials": -1}, "n_trials"),
        ({"n_trials": 2.5}, "n_trials"),
        ({"n_trials": True}, "n_trials"),
        ({"horizon_hours": 0.0}, "horizon_hours"),
        ({"horizon_hours": -1.0}, "horizon_hours"),
        ({"total_customers": 0}, "total_customers"),
        ({"total_customers": -10}, "total_customers"),
        ({"total_customers": 100.5}, "total_customers"),
        ({"arrangement": "mesh"}, "arrangement"),
    ],
)
def test_scalar_envelope_validation(changes: dict[str, Any], argument_name: str) -> None:
    components = [component()]
    with pytest.raises(InvalidArgumentError) as error:
        simulate(components, **changes)
    assert error.value.argument_name == argument_name


@pytest.mark.parametrize(
    ("spec", "argument_name"),
    [
        (component("c1", -0.01, 4.0), "failure_rate"),
        (component("c1", math.nan, 4.0), "failure_rate"),
        (component("c1", math.inf, 4.0), "failure_rate"),
        (component("c1", 0.01, -4.0), "repair_time"),
        (component("c1", 0.01, math.nan), "repair_time"),
        (component("c1", 0.01, 4.0, repair_distribution="weibull"), "repair_distribution"),
        (component("c1", 0.01, 4.0, repair_distribution="lognormal"), "repair_sigma"),
        (component("c1", 0.01, 4.0, repair_sigma=0.5), "repair_sigma"),
        (component("c1", 0.01, 4.0, unknown_field=1), "unknown_field"),
        (component("", 0.01, 4.0), "component_id"),
    ],
)
def test_component_spec_validation(spec: dict[str, Any], argument_name: str) -> None:
    with pytest.raises(InvalidArgumentError) as error:
        simulate([spec], n_trials=1)
    assert error.value.argument_name == argument_name


def test_components_must_be_a_non_empty_sequence_of_unique_specs() -> None:
    with pytest.raises(InvalidArgumentError) as error:
        simulate([], n_trials=1)
    assert error.value.argument_name == "components"
    components = [component("c1"), component("c1")]
    with pytest.raises(InvalidArgumentError) as error:
        simulate(components, n_trials=1)
    assert error.value.argument_name == "component_id"


def test_fixed_rate_component_cannot_fail() -> None:
    runs = simulate([component("c1", 0.0, 4.0)], n_trials=3, seed=4)
    assert all(result.outage_count == 0 for result in runs)


@pytest.mark.benchmark
def test_simulation_cost_scales_with_trials_and_components(benchmark: Any) -> None:
    components = [component(f"c{index}", 0.002 * (index + 1), 4.0) for index in range(4)]
    runs = benchmark(
        reliability.simulate_reliability,
        components,
        arrangement="series",
        total_customers=TOTAL_CUSTOMERS,
        horizon_hours=HORIZON_HOURS,
        n_trials=500,
        seed=8080,
    )
    benchmark.extra_info.update(trials=500, components=len(components), results=len(runs))

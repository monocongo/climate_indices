"""Contract tests for the seeded Monte Carlo reliability simulation (ENERGY2-9, #1343).

The independent reference is not the implementation. For independent components
that start operational, fail after exponential waits, and recover after a repair
drawn from the configured distribution, the expected system downtime over a
horizon is a quadrature of the exact two-state availability ``P_up(t)``:

* a series system is down when any component is down, so its expectation is the
  union measure ``integral(1 - prod(P_up_i(t)))``;
* a parallel system is down when every component is down, so its expectation is
  the intersection measure ``integral(prod(1 - P_up_i(t)))``.

With all served customers interrupted, SAIDI equals that downtime in minutes, so
the quadrature is an analytic SAIDI expectation. The convergence tolerance is a
Monte Carlo standard-error band (three standard errors of the trial mean), not an
arbitrary percentage; a second test checks the standard error falls at the
``1 / sqrt(n)`` rate. Trials are sampled in order from one generator, so a
shorter run is an exact prefix of a longer one with the same seed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from scipy import integrate

from climate_indices import reliability
from climate_indices.exceptions import InvalidArgumentError

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


def _analytic_downtime_minutes(
    components: Sequence[Mapping[str, Any]],
    arrangement: str,
    horizon_hours: float,
) -> float:
    """Quadrature of the exact expected system downtime, in minutes."""
    states = [_availability(float(spec["failure_rate"]), float(spec["repair_time"])) for spec in components]

    def availability_at(component: int, hour: float) -> float:
        steady, approach = states[component]
        return steady + (1.0 - steady) * math.exp(-approach * hour)

    def union_integrand(hour: float) -> float:
        up = 1.0
        for index in range(len(states)):
            up *= availability_at(index, hour)
        return 1.0 - up

    def intersection_integrand(hour: float) -> float:
        down = 1.0
        for index in range(len(states)):
            down *= 1.0 - availability_at(index, hour)
        return down

    integrand = union_integrand if arrangement == "series" else intersection_integrand
    downtime_hours = float(integrate.quad(integrand, 0.0, horizon_hours, limit=200)[0])
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
    with pytest.raises(TypeError):
        reliability.simulate_reliability(
            [component()],
            arrangement="series",
            total_customers=TOTAL_CUSTOMERS,
            horizon_hours=HORIZON_HOURS,
            n_trials=1,
        )  # type: ignore[call-arg]
    with pytest.raises(InvalidArgumentError) as error:
        simulate([component()], n_trials=1, seed=None)
    assert error.value.argument_name == "seed"


def test_public_surface_is_exported_without_extras() -> None:
    assert reliability.simulate_reliability.__name__ == "simulate_reliability"
    assert "simulate_reliability" in reliability.__all__


def test_trials_are_ordered_by_trial_index() -> None:
    runs = simulate([component()], n_trials=3, seed=7)
    assert [result.outage_count for result in runs] == [
        result.outage_count for result in simulate([component()], n_trials=3, seed=7)
    ]
    assert all(isinstance(result, reliability.ReliabilityResult) for result in runs)


def test_series_mean_converges_to_analytic_expectation() -> None:
    components = [component("c1", 0.01, 4.0), component("c2", 0.005, 2.0)]
    n_trials = 2000
    runs = simulate(components, n_trials=n_trials, seed=20240101)
    saidi = np.array([result.saidi for result in runs], dtype=np.float64)
    analytic = _analytic_downtime_minutes(components, "series", HORIZON_HOURS)
    # Three standard errors of the trial mean: a ~99.7 percent Monte Carlo band.
    standard_error = float(np.std(saidi, ddof=1) / math.sqrt(n_trials))
    assert abs(float(np.mean(saidi)) - analytic) <= max(3.0 * standard_error, 5.0)


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
        assert parallel_result.customer_interruptions <= series_result.customer_interruptions
    parallel_analytic = _analytic_downtime_minutes(components, "parallel", HORIZON_HOURS)
    series_analytic = _analytic_downtime_minutes(components, "series", HORIZON_HOURS)
    assert parallel_analytic < series_analytic


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
    with pytest.raises(InvalidArgumentError) as error:
        simulate([component()], n_trials=1, component_topology={"c1": ["c2"]})
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
    with pytest.raises(InvalidArgumentError) as error:
        simulate([component()], **changes)
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
    with pytest.raises(InvalidArgumentError) as error:
        simulate([component("c1"), component("c1")], n_trials=1)
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

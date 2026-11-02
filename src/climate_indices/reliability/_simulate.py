"""Seeded Monte Carlo reliability simulation over supplied component data (ADR-0026).

Each component starts operational. Its failure times are exponential with the
supplied constant rate, and each repair draws from the configured restoration
distribution. A trial's system-down intervals are the union of the component
down intervals for a series arrangement and their intersection for a parallel
arrangement. Each trial's intervals become an event log that the validated
deterministic metric path consumes, so the simulation and the metric definitions
cannot drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from climate_indices.exceptions import InvalidArgumentError
from climate_indices.reliability._events import _integer, _number
from climate_indices.reliability._indices import ReliabilityResult, reliability_indices

_ARRANGEMENTS = ("series", "parallel")
_REPAIR_DISTRIBUTIONS = ("fixed", "exponential", "lognormal")
_SPEC_FIELDS = frozenset({"component_id", "failure_rate", "repair_time", "repair_distribution", "repair_sigma"})
_DRAW_BLOCK = 64
_EPOCH = pd.Timestamp("2000-01-01T00:00:00Z")


@dataclass(frozen=True)
class _Component:
    """One validated simulation component."""

    component_id: str
    failure_rate: float
    repair_time: float
    repair_distribution: str
    repair_sigma: float | None


def _invalid(name: str, message: str) -> InvalidArgumentError:
    return InvalidArgumentError(f"{name}: {message}", argument_name=name)


def _component_spec(spec: Mapping[str, Any], component_id: str) -> _Component:
    """Validate one component's failure and restoration parameters."""
    if "failure_rate" not in spec:
        raise _invalid("failure_rate", "is required")
    failure_rate = _number(spec["failure_rate"], "failure_rate")
    if "repair_time" not in spec:
        raise _invalid("repair_time", "is required")
    repair_time = _number(spec["repair_time"], "repair_time", positive=True)
    distribution = spec.get("repair_distribution", "exponential")
    if distribution not in _REPAIR_DISTRIBUTIONS:
        raise _invalid("repair_distribution", "must be fixed, exponential, or lognormal")
    repair_sigma = None
    if distribution == "lognormal":
        if "repair_sigma" not in spec:
            raise _invalid("repair_sigma", "is required for the lognormal distribution")
        repair_sigma = _number(spec["repair_sigma"], "repair_sigma", positive=True)
    elif "repair_sigma" in spec:
        raise _invalid("repair_sigma", "is only valid with the lognormal distribution")
    return _Component(
        component_id=component_id,
        failure_rate=failure_rate,
        repair_time=repair_time,
        repair_distribution=str(distribution),
        repair_sigma=repair_sigma,
    )


def _component_specs(components: object) -> tuple[_Component, ...]:
    """Validate the supplied component mappings into internal specs."""
    if isinstance(components, (str, bytes, Mapping)) or not isinstance(components, Sequence):
        raise _invalid("components", "must be a non-empty sequence of component mappings")
    if len(components) == 0:
        raise _invalid("components", "must contain at least one component")
    specs: list[_Component] = []
    identifiers: set[str] = set()
    for spec in components:
        if not isinstance(spec, Mapping):
            raise _invalid("components", "must contain component mappings")
        unknown = set(spec) - _SPEC_FIELDS
        if unknown:
            raise _invalid(str(min(unknown, key=str)), "is not a simulation component field")
        component_id = spec.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise _invalid("component_id", "must be a non-empty string")
        if component_id in identifiers:
            raise _invalid("component_id", "must be unique")
        identifiers.add(component_id)
        specs.append(_component_spec(spec, component_id))
    return tuple(specs)


def _customer_count(total_customers: object) -> int:
    """Return a whole, positive served-customer count."""
    count = _number(total_customers, "total_customers", positive=True)
    if count != math.floor(count):
        raise _invalid("total_customers", "must be a whole served-customer count")
    return int(count)


def _repair_distribution(spec: _Component) -> Any | None:
    """Return the restoration distribution, or None for a fixed repair time."""
    if spec.repair_distribution == "fixed":
        return None
    if spec.repair_distribution == "exponential":
        return stats.expon(scale=spec.repair_time)
    # ``repair_time`` is the distribution mean, so the log-scale median is mean * exp(-sigma**2 / 2).
    median = spec.repair_time * math.exp(-0.5 * spec.repair_sigma**2)  # type: ignore[operator]
    return stats.lognorm(s=spec.repair_sigma, scale=median)


def _down_intervals(spec: _Component, horizon_hours: float, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Sample one component's down intervals inside the horizon, in hours."""
    if spec.failure_rate <= 0.0:
        return []
    up_time = stats.expon(scale=1.0 / spec.failure_rate)
    repair = _repair_distribution(spec)
    intervals: list[tuple[float, float]] = []
    time = 0.0
    while True:
        waits = np.asarray(up_time.rvs(size=_DRAW_BLOCK, random_state=rng), dtype=np.float64)
        if repair is None:
            repairs = np.full(_DRAW_BLOCK, spec.repair_time, dtype=np.float64)
        else:
            repairs = np.asarray(repair.rvs(size=_DRAW_BLOCK, random_state=rng), dtype=np.float64)
        past_horizon = False
        for wait, repair_hours in zip(waits, repairs, strict=True):
            time += float(wait)
            if time > horizon_hours:
                past_horizon = True
                break
            intervals.append((time, min(time + float(repair_hours), horizon_hours)))
            time += float(repair_hours)
        if past_horizon:
            return intervals


def _union(intervals: list[tuple[float, float]]) -> list[list[float]]:
    """Merge intervals into sorted, disjoint spans, joining touching endpoints."""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _intersection(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    """Return the sorted, disjoint intersection of two merged interval lists."""
    result: list[list[float]] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start < end:
            result.append([start, end])
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


def _system_intervals(
    per_component: list[list[tuple[float, float]]],
    arrangement: str,
) -> list[list[float]]:
    """Combine per-component down intervals into system-down intervals."""
    if arrangement == "series":
        return _union([interval for intervals in per_component for interval in intervals])
    merged = _union(per_component[0])
    for intervals in per_component[1:]:
        merged = _intersection(merged, _union(intervals))
        if not merged:
            break
    return merged


def _event_rows(intervals: list[list[float]], customers: int, trial: int) -> list[dict[str, Any]]:
    """Build one trial's event log from its system-down intervals."""
    return [
        {
            "event_id": f"t{trial}-e{index}",
            "start_time": _EPOCH + pd.to_timedelta(start, unit="h"),
            "duration": (end - start) * 60.0,
            "customers_interrupted": customers,
        }
        for index, (start, end) in enumerate(intervals)
    ]


def simulate_reliability(
    components: Sequence[Mapping[str, Any]],
    *,
    arrangement: str,
    total_customers: float,
    horizon_hours: float,
    n_trials: int,
    seed: int,
    component_topology: Mapping[str, Sequence[str]] | None = None,
) -> tuple[ReliabilityResult, ...]:
    """Simulate series or parallel component failures into reliability metrics.

    Every component starts operational at hour 0. A component fails after an
    exponential wait with its constant ``failure_rate``, then recovers after a
    draw from its restoration distribution. All served customers are interrupted
    whenever the system is down. SAIDI counts only system-down intervals longer
    than five minutes; intervals of five minutes or less contribute to MAIFI_E,
    not SAIDI, SAIFI, or customer interruptions. A series system is down when
    any component is down. A parallel system is down only when every component
    is down. Each trial's system-down
    intervals form an event log that :func:`reliability_indices` consumes, so the
    simulation uses the validated metric path and definition. System-down
    intervals are clipped at the horizon, so an outage that continues beyond it
    contributes only through the horizon. Classification uses the clipped duration.

    Components are mappings with these fields:

    * ``component_id``: non-empty unique string identifier.
    * ``failure_rate``: constant failures per hour, nonnegative.
    * ``repair_time``: positive mean restoration time in hours. The fixed
      distribution uses it as the deterministic duration; the exponential
      distribution uses it as the mean; the lognormal distribution uses it as
      the mean and takes ``repair_sigma`` as its log-scale shape.
    * ``repair_distribution``: optional ``"fixed"``, ``"exponential"`` (default),
      or ``"lognormal"``.
    * ``repair_sigma``: required positive log-scale shape for the lognormal
      distribution and invalid otherwise.

    A zero ``failure_rate`` component never fails. It cannot bring a parallel
    system down.

    Args:
        components: Non-empty sequence of component mappings. See the field
            list above.
        arrangement: ``"series"`` or ``"parallel"``.
        total_customers: Positive whole served-customer count. Every simulated
            event interrupts all of them, so the count must be an integer.
        horizon_hours: Positive simulated horizon in hours. A shorter horizon
            than a year is not annualized.
        n_trials: Positive number of independent trials.
        seed: Nonnegative integer seed. It is required and is the only source of
            randomness; the library holds no module-level generator state.
        component_topology: Deferred connectivity input. Only ``None`` is
            accepted; any supplied value raises. Customer connectivity and
            network topology resolution are out of scope (ADR-0026).

    Returns:
        One :class:`ReliabilityResult` per trial, ordered by trial index. An
        identical seed and identical arguments reproduce the tuple exactly.

    Raises:
        InvalidArgumentError: A component field, envelope argument, or
            ``component_topology`` is invalid. The exception's ``argument_name``
            identifies the field.
    """
    specs = _component_specs(components)
    if component_topology is not None:
        raise _invalid(
            "component_topology",
            "is not implemented: customer connectivity and network topology resolution are deferred (ADR-0026)",
        )
    if not isinstance(arrangement, str) or arrangement not in _ARRANGEMENTS:
        raise _invalid("arrangement", "must be series or parallel")
    customers = _customer_count(total_customers)
    horizon = _number(horizon_hours, "horizon_hours", positive=True)
    trials = _integer(n_trials, "n_trials")
    if trials <= 0:
        raise _invalid("n_trials", "must be a positive integer")
    generator = np.random.default_rng(_integer(seed, "seed"))
    results = []
    for trial in range(trials):
        per_component = [_down_intervals(spec, horizon, generator) for spec in specs]
        intervals = _system_intervals(per_component, arrangement)
        results.append(reliability_indices(_event_rows(intervals, customers, trial), total_customers=float(customers)))
    return tuple(results)

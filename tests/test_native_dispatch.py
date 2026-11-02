"""Check the native dispatch diagnostic against the shared gate and the CLI route."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from benchmarks import native_dispatch as recipe
from climate_indices import compute, indices
from tests import conftest


@pytest.fixture(scope="module", autouse=True)
def native() -> ModuleType:
    return conftest.import_native()


def test_fallback_reason_matches_the_shared_gate() -> None:
    """The recipe's reason must agree with ``takes`` for the layout and policy gates."""
    plain = np.ones(3)
    assert recipe.fallback_reason(plain) == "numpy-policy: divide=warn, over=warn, invalid=warn"
    with np.errstate(all="ignore"):
        assert recipe.fallback_reason(plain) is None
        assert recipe.fallback_reason(np.ma.masked_array(plain)) == "layout: not plain aligned float64"
        assert recipe.fallback_reason(plain.astype(np.float32)) == "layout: not plain aligned float64"


@pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.empirical])
def test_default_policy_keeps_python_and_names_the_reason(
    monkeypatch: pytest.MonkeyPatch, distribution: indices.Distribution
) -> None:
    """A default caller policy reports no kernel, names the gate, and keeps the Python result (#1531)."""
    reference, events = recipe.run_route("numpy", distribution)
    summary = recipe.summarize(events)

    assert summary["kernels"] == []
    assert "numpy-policy: divide=warn, over=warn, invalid=warn" in summary["fallbacks"]

    entry_point = indices.spi if distribution == indices.Distribution.gamma else indices.standardized_index
    values = recipe.sample_grid().values[:, 0, 0].copy()
    monkeypatch.setattr(compute, "_native", None)
    expected = entry_point(
        values,
        scale=3,
        distribution=distribution,
        data_start_year=1981,
        calibration_year_initial=1981,
        calibration_year_final=2010,
        periodicity=compute.Periodicity.monthly,
    )
    np.testing.assert_array_equal(reference, expected)


def test_cli_workers_apply_the_callers_policy() -> None:
    """Under all-ignore, the CLI pool's workers reach the native kernels (#1531)."""
    with np.errstate(all="ignore"):
        _, events = recipe.run_route("cli", indices.Distribution.gamma)
    worker_states = {
        tuple(sorted(state["numpy_policy"].items())) for state in recipe.summarize(events)["worker_states"]
    }
    # the parent and its workers all report the caller's all-ignore policy
    assert all({policy for _, policy in state} == {"ignore"} for state in worker_states)
    assert "gamma_parameters" in recipe.summarize(events)["kernels"]

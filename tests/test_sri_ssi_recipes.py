"""Runnable checks for the SRI and SSI recipes in docs/standardized-hydrologic-indices.md (issue #1114).

The page's examples are extracted and executed here, so a documented argument
that stops matching `indices.standardized_index()` fails in CI alongside the
page. The assertions pin what the recipes must keep doing: SRI shares the SPI
pipeline, and the documented Pearson Type III fit actually takes effect.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import numpy as np

from climate_indices import compute, indices

_DOC = Path(__file__).resolve().parents[1] / "docs" / "standardized-hydrologic-indices.md"
_START_YEAR = 1981
_END_YEAR = 2010
_MONTHS = (_END_YEAR - _START_YEAR + 1) * 12


def _monthly_series(scale: float, exponent: float) -> np.ndarray:
    """Return a deterministic, non-negative monthly series with a seasonal cycle."""
    months = np.arange(_MONTHS, dtype=float)
    seasonal = 1.0 + 0.6 * np.sin(2.0 * np.pi * (months % 12.0) / 12.0)
    return scale * seasonal * np.exp(exponent * np.sin(months / 7.0))


def _documented_recipe(block_index: int, values: np.ndarray, result_name: str) -> np.ndarray:
    """Execute one python block from the recipe page with the series it expects."""
    blocks = re.findall(r"```python\n(.*?)\n```", _DOC.read_text(encoding="utf-8"), re.DOTALL)
    assert len(blocks) == 2, "the recipe page must keep its two python example blocks"

    namespace: dict[str, object] = {"runoff": values, "streamflow": values}
    exec(compile(blocks[block_index], str(_DOC), "exec"), namespace)
    return np.asarray(namespace[result_name])


def test_sri_recipe_standardizes_runoff_with_the_spi_pipeline() -> None:
    """The SRI recipe is the SPI procedure applied to runoff, not a parallel fit."""
    runoff = _monthly_series(scale=50.0, exponent=0.4)

    sri = _documented_recipe(0, runoff, "sri")
    spi = indices.spi(
        runoff,
        3,
        indices.Distribution.gamma,
        _START_YEAR,
        _START_YEAR,
        _END_YEAR,
        compute.Periodicity.monthly,
    )

    np.testing.assert_array_equal(sri, spi)
    assert np.isnan(sri[:2]).all()
    assert np.isfinite(sri[2:]).all()
    assert float(np.nanstd(sri)) > 0.5


def test_ssi_recipe_runs_the_documented_pearson_type_iii_fit() -> None:
    """The SSI recipe's documented Pearson fit takes effect and does not fall back to gamma."""
    streamflow = _monthly_series(scale=20.0, exponent=0.8)

    with mock.patch.object(compute._default_fallback_strategy, "log_fallback_warning") as log_fallback_warning:
        ssi = _documented_recipe(1, streamflow, "ssi")

    assert log_fallback_warning.call_count == 0, "the documented Pearson fit must not fall back to gamma"
    gamma = indices.standardized_index(
        streamflow,
        12,
        indices.Distribution.gamma,
        _START_YEAR,
        _START_YEAR,
        _END_YEAR,
        compute.Periodicity.monthly,
    )
    assert not np.allclose(ssi[11:], gamma[11:]), "the documented Pearson fit must differ from a gamma fit"
    assert np.isnan(ssi[:11]).all()
    assert np.isfinite(ssi[11:]).all()
    assert float(np.nanstd(ssi)) > 0.5

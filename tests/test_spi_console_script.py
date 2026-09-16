"""Deprecation contract for the legacy `spi` console script (issue #919)."""

import sys

import pytest

from climate_indices import __spi__
from climate_indices.exceptions import ClimateIndicesDeprecationWarning


def test_main_emits_deprecation_warning(monkeypatch):
    """Invoking the legacy SPI CLI warns that it is scheduled for removal."""
    monkeypatch.setattr(sys, "argv", ["spi", "--help"])

    with pytest.warns(ClimateIndicesDeprecationWarning) as recorded:
        with pytest.raises(SystemExit) as exit_info:
            __spi__.main()

    assert exit_info.value.code == 0

    # pin every claim the deprecation makes: the lifecycle versions, the
    # replacement, the uncovered parameter-cache options, and the per-scale
    # input reread trade-off (#957)
    message = str(recorded[0].message)
    assert "deprecated since version 2.4.0" in message
    assert "removed in version 3.0.0" in message
    assert "climate_indices --index spi" in message
    assert "save_params" in message
    assert "re-reads the precipitation input once per scale" in message

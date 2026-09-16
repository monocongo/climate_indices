"""Deprecation contract for the legacy `spi` console script (issue #919)."""

import sys

import pytest

from climate_indices import __spi__
from climate_indices.exceptions import ClimateIndicesDeprecationWarning


def test_main_emits_deprecation_warning(monkeypatch):
    """Invoking the legacy SPI CLI warns that it is scheduled for removal."""
    monkeypatch.setattr(sys, "argv", ["spi", "--help"])

    with pytest.warns(
        ClimateIndicesDeprecationWarning,
        match=r"deprecated since version 2\.4\.0.*save_params.*re-reads the precipitation input once per scale",
    ):
        with pytest.raises(SystemExit):
            __spi__.main()

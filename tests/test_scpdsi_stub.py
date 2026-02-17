"""Tests for the scPDSI stub interface.

Validates that the scPDSI function exists with the correct signature,
raises NotImplementedError when called, and is importable from both
indices and typed_public_api modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices import compute, indices
from climate_indices.typed_public_api import scpdsi as scpdsi_typed


class TestScpdsiStub:
    """Tests for scPDSI stub function in indices module."""

    def test_function_exists_in_indices(self) -> None:
        """scpdsi should be importable from the indices module."""
        assert hasattr(indices, "scpdsi")
        assert callable(indices.scpdsi)

    def test_in_module_all(self) -> None:
        """scpdsi should be listed in indices.__all__."""
        assert "scpdsi" in indices.__all__

    def test_raises_not_implemented_error(self) -> None:
        """Calling scpdsi should raise NotImplementedError."""
        precip = np.ones(120)
        pet = np.ones(120)
        with pytest.raises(NotImplementedError, match="scPDSI implementation planned"):
            indices.scpdsi(
                precip_values=precip,
                pet_values=pet,
                awc=150.0,
                data_start_year=2000,
                calibration_year_initial=2000,
                calibration_year_final=2009,
                periodicity=compute.Periodicity.monthly,
            )

    def test_error_message_includes_reference(self) -> None:
        """Error message should cite Wells et al. (2004)."""
        precip = np.ones(12)
        pet = np.ones(12)
        with pytest.raises(NotImplementedError, match="Wells et al.") as exc_info:
            indices.scpdsi(
                precip_values=precip,
                pet_values=pet,
                awc=100.0,
                data_start_year=2000,
                calibration_year_initial=2000,
                calibration_year_final=2000,
                periodicity=compute.Periodicity.monthly,
            )
        assert "doi:" in str(exc_info.value)

    def test_has_docstring(self) -> None:
        """scpdsi should have a docstring with methodology overview."""
        assert indices.scpdsi.__doc__ is not None
        assert "self-calibrating" in indices.scpdsi.__doc__.lower()
        assert "Wells" in indices.scpdsi.__doc__


class TestScpdsiTypedApi:
    """Tests for scPDSI stub in typed_public_api module."""

    def test_function_exists_in_typed_api(self) -> None:
        """scpdsi should be importable from typed_public_api."""
        assert callable(scpdsi_typed)

    def test_typed_api_raises_not_implemented(self) -> None:
        """typed_public_api.scpdsi should also raise NotImplementedError."""
        precip = np.ones(120)
        pet = np.ones(120)
        with pytest.raises(NotImplementedError, match="scPDSI implementation planned"):
            scpdsi_typed(
                precip_values=precip,
                pet_values=pet,
                awc=150.0,
                data_start_year=2000,
                calibration_year_initial=2000,
                calibration_year_final=2009,
                periodicity=compute.Periodicity.monthly,
            )

    def test_typed_api_has_docstring(self) -> None:
        """typed_public_api.scpdsi should have a docstring."""
        assert scpdsi_typed.__doc__ is not None
        assert "Not Yet Implemented" in scpdsi_typed.__doc__

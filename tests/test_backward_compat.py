"""Backward compatibility tests for NumPy computation path.

This module validates that Epic 2's xarray infrastructure (Stories 2.1-2.11) did not
break the existing NumPy-only computation path. While indices.py was enhanced with
validation helpers and structured logging in Epic 1, the core computation logic
remains unchanged.

Test Strategy:
1. Numerical equivalence at 1e-8 tolerance (stricter than existing 0.001/0.01)
2. Signature stability (no new required parameters)
3. No deprecation warnings
4. Return type contract (ndarray, not DataArray)
5. Public API equivalence (typed_public_api.py matches indices.py for NumPy inputs)
6. Error hierarchy documentation (intentional InvalidArgumentError change)
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
import xarray as xr

from climate_indices import indices, spei, spi
from climate_indices.compute import Periodicity
from climate_indices.exceptions import ClimateIndicesError, InvalidArgumentError
from climate_indices.indices import Distribution


class TestNumericalEquivalenceSPI:
    """Test SPI against v1.x reference fixtures at strict 1e-8 tolerance."""

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "data_year_start_monthly",
        "data_year_end_monthly",
        "spi_1_month_gamma",
    )
    def test_spi_1_month_gamma_bitexact(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        spi_1_month_gamma,
    ) -> None:
        """SPI 1-month gamma matches fixture at 1e-8 tolerance."""
        result = indices.spi(
            precips_mm_monthly,
            1,
            Distribution.gamma,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
            Periodicity.monthly,
        )

        np.testing.assert_allclose(
            result,
            spi_1_month_gamma,
            atol=1e-8,
            err_msg="SPI 1-month gamma does not match reference fixture at 1e-8 tolerance",
        )

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "data_year_start_monthly",
        "data_year_end_monthly",
        "spi_6_month_gamma",
    )
    def test_spi_6_month_gamma_bitexact(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        spi_6_month_gamma,
    ) -> None:
        """SPI 6-month gamma matches fixture at 1e-8 tolerance."""
        result = indices.spi(
            precips_mm_monthly.flatten(),
            6,
            Distribution.gamma,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
            Periodicity.monthly,
        )

        np.testing.assert_allclose(
            result,
            spi_6_month_gamma,
            atol=1e-8,
            err_msg="SPI 6-month gamma does not match reference fixture at 1e-8 tolerance",
        )

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "data_year_start_monthly",
        "calibration_year_start_monthly",
        "calibration_year_end_monthly",
        "spi_6_month_pearson3",
    )
    def test_spi_6_month_pearson_bitexact(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        spi_6_month_pearson3,
    ) -> None:
        """SPI 6-month Pearson matches fixture at 1e-8 tolerance."""
        result = indices.spi(
            precips_mm_monthly.flatten(),
            6,
            Distribution.pearson,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            Periodicity.monthly,
        )

        np.testing.assert_allclose(
            result,
            spi_6_month_pearson3,
            atol=1e-8,
            err_msg="SPI 6-month Pearson does not match reference fixture at 1e-8 tolerance",
        )


class TestNumericalEquivalenceSPEI:
    """Test SPEI against v1.x reference fixtures.

    Note: SPEI tolerance is distribution-specific due to numerical differences in
    fitting algorithms:
    - Gamma: 1e-5 tolerance (water balance + gamma fitting)
    - Pearson: 0.001 tolerance (water balance + Pearson Type III fitting)

    Both are stricter than existing test_indices.py (0.01) and validate that
    Epic 2 xarray infrastructure did not change core computation.
    """

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "pet_thornthwaite_mm",
        "data_year_start_monthly",
        "data_year_end_monthly",
        "spei_6_month_gamma",
    )
    def test_spei_6_month_gamma_strict(
        self,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        data_year_end_monthly,
        spei_6_month_gamma,
    ) -> None:
        """SPEI 6-month gamma matches fixture at 1e-5 tolerance (1000x stricter than existing tests)."""
        result = indices.spei(
            precips_mm_monthly,
            pet_thornthwaite_mm,
            6,
            Distribution.gamma,
            Periodicity.monthly,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
        )

        np.testing.assert_allclose(
            result,
            spei_6_month_gamma,
            atol=1e-5,
            err_msg="SPEI 6-month gamma does not match reference fixture at 1e-5 tolerance",
        )

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "pet_thornthwaite_mm",
        "data_year_start_monthly",
        "data_year_end_monthly",
        "spei_6_month_pearson3",
    )
    def test_spei_6_month_pearson_strict(
        self,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        data_year_end_monthly,
        spei_6_month_pearson3,
    ) -> None:
        """SPEI 6-month Pearson matches fixture at 0.001 tolerance (10x stricter than existing tests)."""
        result = indices.spei(
            precips_mm_monthly,
            pet_thornthwaite_mm,
            6,
            Distribution.pearson,
            Periodicity.monthly,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
        )

        np.testing.assert_allclose(
            result,
            spei_6_month_pearson3,
            atol=0.001,
            err_msg="SPEI 6-month Pearson does not match reference fixture at 0.001 tolerance",
        )


class TestPublicAPIEquivalence:
    """Verify public API (typed_public_api.py) produces identical results for NumPy inputs."""

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "data_year_start_monthly",
        "data_year_end_monthly",
        "calibration_year_start_monthly",
        "calibration_year_end_monthly",
        "spi_1_month_gamma",
        "spi_6_month_gamma",
        "spi_6_month_pearson3",
    )
    def test_public_spi_matches_indices_spi(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        spi_1_month_gamma,
        spi_6_month_gamma,
        spi_6_month_pearson3,
    ) -> None:
        """Public climate_indices.spi() matches indices.spi() for NumPy inputs."""
        # test case 1: 1-month gamma
        public_result = spi(
            values=precips_mm_monthly,
            scale=1,
            distribution=Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=data_year_start_monthly,
            calibration_year_final=data_year_end_monthly,
            periodicity=Periodicity.monthly,
        )
        indices_result = indices.spi(
            precips_mm_monthly,
            1,
            Distribution.gamma,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
            Periodicity.monthly,
        )
        np.testing.assert_array_equal(
            public_result,
            indices_result,
            err_msg="Public API SPI 1-month gamma does not match indices.spi (bit-exact)",
        )

        # test case 2: 6-month gamma
        public_result = spi(
            values=precips_mm_monthly.flatten(),
            scale=6,
            distribution=Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=data_year_start_monthly,
            calibration_year_final=data_year_end_monthly,
            periodicity=Periodicity.monthly,
        )
        indices_result = indices.spi(
            precips_mm_monthly.flatten(),
            6,
            Distribution.gamma,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
            Periodicity.monthly,
        )
        np.testing.assert_array_equal(
            public_result,
            indices_result,
            err_msg="Public API SPI 6-month gamma does not match indices.spi (bit-exact)",
        )

        # test case 3: 6-month pearson
        public_result = spi(
            values=precips_mm_monthly.flatten(),
            scale=6,
            distribution=Distribution.pearson,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=Periodicity.monthly,
        )
        indices_result = indices.spi(
            precips_mm_monthly.flatten(),
            6,
            Distribution.pearson,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            Periodicity.monthly,
        )
        np.testing.assert_array_equal(
            public_result,
            indices_result,
            err_msg="Public API SPI 6-month Pearson does not match indices.spi (bit-exact)",
        )

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "pet_thornthwaite_mm",
        "data_year_start_monthly",
        "data_year_end_monthly",
        "spei_6_month_gamma",
        "spei_6_month_pearson3",
    )
    def test_public_spei_matches_indices_spei(
        self,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        data_year_end_monthly,
        spei_6_month_gamma,
        spei_6_month_pearson3,
    ) -> None:
        """Public climate_indices.spei() matches indices.spei() for NumPy inputs."""
        # test case 1: 6-month gamma
        public_result = spei(
            precips_mm=precips_mm_monthly,
            pet_mm=pet_thornthwaite_mm,
            scale=6,
            distribution=Distribution.gamma,
            periodicity=Periodicity.monthly,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=data_year_start_monthly,
            calibration_year_final=data_year_end_monthly,
        )
        indices_result = indices.spei(
            precips_mm_monthly,
            pet_thornthwaite_mm,
            6,
            Distribution.gamma,
            Periodicity.monthly,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
        )
        np.testing.assert_array_equal(
            public_result,
            indices_result,
            err_msg="Public API SPEI 6-month gamma does not match indices.spei (bit-exact)",
        )

        # test case 2: 6-month pearson
        public_result = spei(
            precips_mm=precips_mm_monthly,
            pet_mm=pet_thornthwaite_mm,
            scale=6,
            distribution=Distribution.pearson,
            periodicity=Periodicity.monthly,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=data_year_start_monthly,
            calibration_year_final=data_year_end_monthly,
        )
        indices_result = indices.spei(
            precips_mm_monthly,
            pet_thornthwaite_mm,
            6,
            Distribution.pearson,
            Periodicity.monthly,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
        )
        np.testing.assert_array_equal(
            public_result,
            indices_result,
            err_msg="Public API SPEI 6-month Pearson does not match indices.spei (bit-exact)",
        )


class TestSignatureStability:
    """Verify function signatures have not introduced new required parameters."""

    # v1.x reference signatures (required parameters only, in order)
    REFERENCE_SIGNATURES = {
        "spi": [
            "values",
            "scale",
            "distribution",
            "data_start_year",
            "calibration_year_initial",
            "calibration_year_final",
            "periodicity",
        ],
        "spei": [
            "precips_mm",
            "pet_mm",
            "scale",
            "distribution",
            "periodicity",
            "data_start_year",
            "calibration_year_initial",
            "calibration_year_final",
        ],
        "percentage_of_normal": [
            "values",
            "scale",
            "data_start_year",
            "calibration_start_year",
            "calibration_end_year",
            "periodicity",
        ],
        "pet": [
            "temperature_celsius",
            "latitude_degrees",
            "data_start_year",
        ],
        "pci": ["rainfall_mm"],
    }

    def _get_required_params(self, func) -> list[str]:
        """Extract required parameter names from function signature."""
        sig = inspect.signature(func)
        required = []
        for param_name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        return required

    def test_spi_signature_stability(self) -> None:
        """indices.spi has no new required parameters."""
        current_required = self._get_required_params(indices.spi)
        reference_required = self.REFERENCE_SIGNATURES["spi"]

        assert current_required == reference_required, (
            f"SPI signature changed. Expected required params: {reference_required}, got: {current_required}"
        )

    def test_spei_signature_stability(self) -> None:
        """indices.spei has no new required parameters."""
        current_required = self._get_required_params(indices.spei)
        reference_required = self.REFERENCE_SIGNATURES["spei"]

        assert current_required == reference_required, (
            f"SPEI signature changed. Expected required params: {reference_required}, got: {current_required}"
        )

    def test_percentage_of_normal_signature_stability(self) -> None:
        """indices.percentage_of_normal has no new required parameters."""
        current_required = self._get_required_params(indices.percentage_of_normal)
        reference_required = self.REFERENCE_SIGNATURES["percentage_of_normal"]

        assert current_required == reference_required, (
            f"PNP signature changed. Expected required params: {reference_required}, got: {current_required}"
        )

    def test_pet_signature_stability(self) -> None:
        """indices.pet has no new required parameters."""
        current_required = self._get_required_params(indices.pet)
        reference_required = self.REFERENCE_SIGNATURES["pet"]

        assert current_required == reference_required, (
            f"PET signature changed. Expected required params: {reference_required}, got: {current_required}"
        )

    def test_pci_signature_stability(self) -> None:
        """indices.pci has no new required parameters."""
        current_required = self._get_required_params(indices.pci)
        reference_required = self.REFERENCE_SIGNATURES["pci"]

        assert current_required == reference_required, (
            f"PCI signature changed. Expected required params: {reference_required}, got: {current_required}"
        )


class TestNoDeprecationWarnings:
    """Verify NumPy path does not emit deprecation warnings."""

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "data_year_start_monthly",
        "data_year_end_monthly",
    )
    def test_spi_numpy_no_deprecation_warnings(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
    ) -> None:
        """SPI with NumPy inputs emits no deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            indices.spi(
                precips_mm_monthly.flatten(),
                6,
                Distribution.gamma,
                data_year_start_monthly,
                data_year_start_monthly,
                data_year_end_monthly,
                Periodicity.monthly,
            )

            # filter for deprecation-related warnings
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(
                    warning.category,
                    DeprecationWarning | PendingDeprecationWarning | FutureWarning,
                )
            ]

            assert len(deprecation_warnings) == 0, (
                f"SPI emitted {len(deprecation_warnings)} deprecation warning(s): {[str(dw.message) for dw in deprecation_warnings]}"
            )

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "pet_thornthwaite_mm",
        "data_year_start_monthly",
        "data_year_end_monthly",
    )
    def test_spei_numpy_no_deprecation_warnings(
        self,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        data_year_end_monthly,
    ) -> None:
        """SPEI with NumPy inputs emits no deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            indices.spei(
                precips_mm_monthly,
                pet_thornthwaite_mm,
                6,
                Distribution.gamma,
                Periodicity.monthly,
                data_year_start_monthly,
                data_year_start_monthly,
                data_year_end_monthly,
            )

            # filter for deprecation-related warnings
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(
                    warning.category,
                    DeprecationWarning | PendingDeprecationWarning | FutureWarning,
                )
            ]

            assert len(deprecation_warnings) == 0, (
                f"SPEI emitted {len(deprecation_warnings)} deprecation warning(s): {[str(dw.message) for dw in deprecation_warnings]}"
            )


class TestReturnTypeContract:
    """Verify NumPy inputs return numpy.ndarray (not xarray.DataArray)."""

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "data_year_start_monthly",
        "data_year_end_monthly",
    )
    def test_spi_returns_ndarray(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
    ) -> None:
        """SPI with NumPy input returns numpy.ndarray."""
        result = indices.spi(
            precips_mm_monthly.flatten(),
            6,
            Distribution.gamma,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
            Periodicity.monthly,
        )

        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result)}"
        assert not isinstance(result, xr.DataArray), "Result should not be xarray.DataArray"

    @pytest.mark.usefixtures(
        "precips_mm_monthly",
        "pet_thornthwaite_mm",
        "data_year_start_monthly",
        "data_year_end_monthly",
    )
    def test_spei_returns_ndarray(
        self,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        data_year_end_monthly,
    ) -> None:
        """SPEI with NumPy input returns numpy.ndarray."""
        result = indices.spei(
            precips_mm_monthly,
            pet_thornthwaite_mm,
            6,
            Distribution.gamma,
            Periodicity.monthly,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
        )

        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result)}"
        assert not isinstance(result, xr.DataArray), "Result should not be xarray.DataArray"


class TestErrorHierarchyDocumented:
    """Document intentional ValueError → InvalidArgumentError change from Epic 1."""

    def test_invalid_argument_error_not_valueerror(self) -> None:
        """InvalidArgumentError is not a subclass of ValueError (intentional change)."""
        assert not issubclass(InvalidArgumentError, ValueError), (
            "InvalidArgumentError should NOT inherit from ValueError. "
            "This is an intentional Epic 1 change to improve error hierarchy."
        )

    def test_invalid_argument_error_is_exception(self) -> None:
        """InvalidArgumentError inherits from ClimateIndicesError and Exception."""
        assert issubclass(InvalidArgumentError, ClimateIndicesError), (
            "InvalidArgumentError must inherit from ClimateIndicesError"
        )
        assert issubclass(InvalidArgumentError, Exception), "InvalidArgumentError must inherit from Exception"

    @pytest.mark.usefixtures("precips_mm_monthly", "data_year_start_monthly", "data_year_end_monthly")
    def test_dimension_errors_still_raise_valueerror(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
    ) -> None:
        """Dimension mismatch errors still raise ValueError (unchanged behavior)."""
        # 3-D array should raise ValueError
        three_d_array = np.zeros((4, 4, 8))

        with pytest.raises(ValueError):
            indices.spi(
                three_d_array,
                6,
                Distribution.gamma,
                data_year_start_monthly,
                data_year_start_monthly,
                data_year_end_monthly,
                Periodicity.monthly,
            )

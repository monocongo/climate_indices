"""Tests for input validation in public API functions.

This test module validates the error handling behavior introduced in Story 1.2,
focusing on scale, distribution, and periodicity parameter validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices import ClimateIndicesError, compute, indices
from climate_indices.exceptions import InvalidArgumentError


@pytest.fixture
def valid_precip_data():
    """Provide a minimal valid precipitation array for testing."""
    rng = np.random.default_rng(seed=42)
    return rng.random(12 * 10) * 100


@pytest.fixture
def valid_pet_data():
    """Provide a minimal valid PET array for testing."""
    rng = np.random.default_rng(seed=43)
    return rng.random(12 * 10) * 50


class TestScaleValidation:
    """Test scale parameter validation."""

    def test_scale_below_minimum(self, valid_precip_data):
        """scale=0 raises InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                0,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert exc_info.value.argument_name == "scale"
        assert exc_info.value.argument_value == "0"

    def test_scale_above_maximum(self, valid_precip_data):
        """scale=73 raises InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                73,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert exc_info.value.argument_name == "scale"
        assert exc_info.value.argument_value == "73"

    def test_scale_none(self, valid_precip_data):
        """scale=None raises InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.percentage_of_normal(
                valid_precip_data,
                None,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert exc_info.value.argument_name == "scale"

    def test_scale_at_minimum_boundary(self, valid_precip_data):
        """scale=1 passes (no error)."""
        # should not raise
        indices.spi(
            valid_precip_data,
            1,
            indices.Distribution.gamma,
            2000,
            2000,
            2009,
            compute.Periodicity.monthly,
        )

    def test_scale_at_maximum_boundary(self, valid_precip_data):
        """scale=72 passes (no error)."""
        # should not raise
        indices.spi(
            valid_precip_data,
            72,
            indices.Distribution.gamma,
            2000,
            2000,
            2009,
            compute.Periodicity.monthly,
        )

    def test_scale_error_message_includes_valid_range(self, valid_precip_data):
        """Verify message contains "[1, 72]"."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                0,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert "[1, 72]" in str(exc_info.value)
        assert exc_info.value.valid_values == "[1, 72]"

    def test_scale_error_message_includes_remediation(self, valid_precip_data):
        """Verify message contains common scale examples."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                0,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        message = str(exc_info.value)
        assert "1 (monthly)" in message
        assert "3 (seasonal)" in message
        assert "6 (half-year)" in message
        assert "12 (annual)" in message

    def test_scale_error_attributes(self, valid_precip_data):
        """Verify argument_name="scale" and argument_value set."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                -5,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert exc_info.value.argument_name == "scale"
        assert exc_info.value.argument_value == "-5"
        assert exc_info.value.valid_values is not None


class TestDistributionValidation:
    """Test distribution parameter validation."""

    def test_distribution_none(self, valid_precip_data):
        """None raises InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                6,
                None,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert exc_info.value.argument_name == "distribution"

    def test_distribution_string(self, valid_precip_data):
        """'gamma' string raises InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                6,
                "gamma",
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert exc_info.value.argument_name == "distribution"
        assert exc_info.value.argument_value == "gamma"

    def test_distribution_error_message_includes_valid_values(self, valid_precip_data):
        """Message lists gamma, pearson."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                6,
                None,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        message = str(exc_info.value)
        assert "gamma" in message
        assert "pearson" in message
        assert "gamma, pearson" in exc_info.value.valid_values

    def test_distribution_error_message_includes_remediation(self, valid_precip_data):
        """Message suggests Distribution.gamma."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                6,
                None,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        message = str(exc_info.value)
        assert "Distribution.gamma" in message or "indices.Distribution.gamma" in message
        assert "Distribution.pearson" in message or "indices.Distribution.pearson" in message

    def test_distribution_error_attributes(self, valid_precip_data):
        """Verify argument_name="distribution" set."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spei(
                valid_precip_data,
                valid_precip_data,
                6,
                "invalid",
                compute.Periodicity.monthly,
                2000,
                2000,
                2009,
            )
        assert exc_info.value.argument_name == "distribution"
        assert exc_info.value.argument_value == "invalid"


class TestPeriodicityValidation:
    """Test periodicity parameter validation."""

    def test_periodicity_string(self, valid_precip_data):
        """'monthly' string raises InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                6,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                "monthly",
            )
        assert exc_info.value.argument_name == "periodicity"
        assert exc_info.value.argument_value == "monthly"

    def test_periodicity_none(self, valid_precip_data):
        """None raises InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.percentage_of_normal(
                valid_precip_data,
                6,
                2000,
                2000,
                2009,
                None,
            )
        assert exc_info.value.argument_name == "periodicity"

    def test_periodicity_error_message_includes_valid_values(self, valid_precip_data):
        """Message lists monthly, daily."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                6,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                "invalid",
            )
        message = str(exc_info.value)
        assert "monthly" in message
        assert "daily" in message
        assert "monthly, daily" in exc_info.value.valid_values

    def test_periodicity_error_message_includes_remediation(self, valid_precip_data):
        """Suggests Periodicity.monthly."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                6,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                "monthly",
            )
        message = str(exc_info.value)
        assert "Periodicity.monthly" in message
        assert "Periodicity.daily" in message

    def test_periodicity_error_attributes(self, valid_precip_data):
        """Verify argument_name="periodicity" set."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spei(
                valid_precip_data,
                valid_precip_data,
                6,
                indices.Distribution.gamma,
                "unsupported",
                2000,
                2000,
                2009,
            )
        assert exc_info.value.argument_name == "periodicity"
        assert exc_info.value.argument_value == "unsupported"


class TestValidatorsAreCalledByPublicAPI:
    """Verify that validators are called by each public API function."""

    def test_spi_validates_scale(self, valid_precip_data):
        """spi with scale=0 raises InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                0,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert exc_info.value.argument_name == "scale"

    def test_spi_validates_distribution(self, valid_precip_data):
        """spi with None distribution raises."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                6,
                None,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert exc_info.value.argument_name == "distribution"

    def test_spi_validates_periodicity(self, valid_precip_data):
        """spi with string periodicity raises."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spi(
                valid_precip_data,
                6,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                "monthly",
            )
        assert exc_info.value.argument_name == "periodicity"

    def test_spei_validates_scale(self, valid_precip_data, valid_pet_data):
        """spei with scale=0 raises."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spei(
                valid_precip_data,
                valid_pet_data,
                0,
                indices.Distribution.gamma,
                compute.Periodicity.monthly,
                2000,
                2000,
                2009,
            )
        assert exc_info.value.argument_name == "scale"

    def test_spei_validates_distribution(self, valid_precip_data, valid_pet_data):
        """spei with None distribution raises."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spei(
                valid_precip_data,
                valid_pet_data,
                6,
                None,
                compute.Periodicity.monthly,
                2000,
                2000,
                2009,
            )
        assert exc_info.value.argument_name == "distribution"

    def test_spei_validates_periodicity(self, valid_precip_data, valid_pet_data):
        """spei with string raises."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.spei(
                valid_precip_data,
                valid_pet_data,
                6,
                indices.Distribution.gamma,
                "monthly",
                2000,
                2000,
                2009,
            )
        assert exc_info.value.argument_name == "periodicity"

    def test_percentage_of_normal_validates_scale(self, valid_precip_data):
        """percentage_of_normal with scale=-1 raises."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.percentage_of_normal(
                valid_precip_data,
                -1,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )
        assert exc_info.value.argument_name == "scale"

    def test_percentage_of_normal_validates_periodicity(self, valid_precip_data):
        """percentage_of_normal with string raises."""
        with pytest.raises(InvalidArgumentError) as exc_info:
            indices.percentage_of_normal(
                valid_precip_data,
                6,
                2000,
                2000,
                2009,
                "daily",
            )
        assert exc_info.value.argument_name == "periodicity"

    def test_all_validation_errors_catchable_as_base(self, valid_precip_data):
        """All catchable via ClimateIndicesError."""
        # scale validation
        with pytest.raises(ClimateIndicesError):
            indices.spi(
                valid_precip_data,
                0,
                indices.Distribution.gamma,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )

        # distribution validation
        with pytest.raises(ClimateIndicesError):
            indices.spi(
                valid_precip_data,
                6,
                None,
                2000,
                2000,
                2009,
                compute.Periodicity.monthly,
            )

        # periodicity validation
        with pytest.raises(ClimateIndicesError):
            indices.percentage_of_normal(
                valid_precip_data,
                6,
                2000,
                2000,
                2009,
                "unsupported",
            )

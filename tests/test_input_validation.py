"""Table-driven tests for input validation in the public index APIs.

Each row drives a real public API call (``spi``, ``spei``,
``percentage_of_normal``) through one invalid argument, so the tables are the
single owner of the scale/distribution/periodicity validation contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices import ClimateIndicesError, compute, indices
from climate_indices.exceptions import InvalidArgumentError


@pytest.fixture
def valid_precip_data():
    """Provide a minimal valid precipitation array for testing."""
    return np.random.rand(12 * 10) * 100


@pytest.fixture
def valid_pet_data():
    """Provide a minimal valid PET array for testing."""
    return np.random.rand(12 * 10) * 50


# (call, argument name, argument value) rows. Calls take (precip, pet) so the
# same table drives every public entry point.
INVALID_ARGUMENT_CASES = [
    pytest.param(
        lambda p, pet: indices.spi(p, 0, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly),
        "scale",
        "0",
        id="spi-scale-zero",
    ),
    pytest.param(
        lambda p, pet: indices.spi(p, 73, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly),
        "scale",
        "73",
        id="spi-scale-above-maximum",
    ),
    pytest.param(
        lambda p, pet: indices.spi(p, -5, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly),
        "scale",
        "-5",
        id="spi-scale-negative",
    ),
    pytest.param(
        lambda p, pet: indices.percentage_of_normal(p, None, 2000, 2000, 2009, compute.Periodicity.monthly),
        "scale",
        "None",
        id="percentage-of-normal-scale-none",
    ),
    pytest.param(
        lambda p, pet: indices.percentage_of_normal(p, -1, 2000, 2000, 2009, compute.Periodicity.monthly),
        "scale",
        "-1",
        id="percentage-of-normal-scale-negative",
    ),
    pytest.param(
        lambda p, pet: indices.spei(
            p, pet, 0, indices.Distribution.gamma, compute.Periodicity.monthly, 2000, 2000, 2009
        ),
        "scale",
        "0",
        id="spei-scale-zero",
    ),
    pytest.param(
        lambda p, pet: indices.spi(p, 6, None, 2000, 2000, 2009, compute.Periodicity.monthly),
        "distribution",
        "None",
        id="spi-distribution-none",
    ),
    pytest.param(
        lambda p, pet: indices.spi(p, 6, "gamma", 2000, 2000, 2009, compute.Periodicity.monthly),
        "distribution",
        "gamma",
        id="spi-distribution-string",
    ),
    pytest.param(
        lambda p, pet: indices.spei(p, pet, 6, "invalid", compute.Periodicity.monthly, 2000, 2000, 2009),
        "distribution",
        "invalid",
        id="spei-distribution-invalid",
    ),
    pytest.param(
        lambda p, pet: indices.spei(p, pet, 6, None, compute.Periodicity.monthly, 2000, 2000, 2009),
        "distribution",
        "None",
        id="spei-distribution-none",
    ),
    pytest.param(
        lambda p, pet: indices.spi(p, 6, indices.Distribution.gamma, 2000, 2000, 2009, "monthly"),
        "periodicity",
        "monthly",
        id="spi-periodicity-string",
    ),
    pytest.param(
        lambda p, pet: indices.spi(p, 6, indices.Distribution.gamma, 2000, 2000, 2009, "invalid"),
        "periodicity",
        "invalid",
        id="spi-periodicity-invalid",
    ),
    pytest.param(
        lambda p, pet: indices.percentage_of_normal(p, 6, 2000, 2000, 2009, None),
        "periodicity",
        "None",
        id="percentage-of-normal-periodicity-none",
    ),
    pytest.param(
        lambda p, pet: indices.percentage_of_normal(p, 6, 2000, 2000, 2009, "daily"),
        "periodicity",
        "daily",
        id="percentage-of-normal-periodicity-daily",
    ),
    pytest.param(
        lambda p, pet: indices.spei(p, pet, 6, indices.Distribution.gamma, "monthly", 2000, 2000, 2009),
        "periodicity",
        "monthly",
        id="spei-periodicity-string",
    ),
    pytest.param(
        lambda p, pet: indices.spei(p, pet, 6, indices.Distribution.gamma, "unsupported", 2000, 2000, 2009),
        "periodicity",
        "unsupported",
        id="spei-periodicity-unsupported",
    ),
]

# (call, message fragments, valid_values) rows: the remediation text users act on.
INVALID_ARGUMENT_MESSAGE_CASES = [
    pytest.param(
        lambda p, pet: indices.spi(p, 0, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly),
        ("[1, 72]", "1 (monthly)", "3 (seasonal)", "6 (half-year)", "12 (annual)"),
        "[1, 72]",
        id="scale",
    ),
    pytest.param(
        lambda p, pet: indices.spi(p, 6, None, 2000, 2000, 2009, compute.Periodicity.monthly),
        ("gamma", "pearson", "indices.Distribution.gamma", "indices.Distribution.pearson"),
        "gamma, pearson",
        id="distribution",
    ),
    pytest.param(
        lambda p, pet: indices.spi(p, 6, indices.Distribution.gamma, 2000, 2000, 2009, "monthly"),
        ("monthly", "daily", "compute.Periodicity.monthly", "compute.Periodicity.daily"),
        "monthly, daily",
        id="periodicity",
    ),
]

# (call, argument name) rows proving every validator routes through the base error.
CATCH_ALL_CASES = [
    pytest.param(
        lambda p, pet: indices.spi(p, 0, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly),
        "scale",
        id="scale",
    ),
    pytest.param(
        lambda p, pet: indices.spi(p, 6, None, 2000, 2000, 2009, compute.Periodicity.monthly),
        "distribution",
        id="distribution",
    ),
    pytest.param(
        lambda p, pet: indices.percentage_of_normal(p, 6, 2000, 2000, 2009, "unsupported"),
        "periodicity",
        id="periodicity",
    ),
]


@pytest.mark.parametrize(("call", "argument_name", "argument_value"), INVALID_ARGUMENT_CASES)
def test_invalid_arguments_raise_invalid_argument_error(
    valid_precip_data, valid_pet_data, call, argument_name, argument_value
) -> None:
    """Each invalid argument raises one structured error naming the offending value."""
    with pytest.raises(InvalidArgumentError) as exc_info:
        call(valid_precip_data, valid_pet_data)
    assert exc_info.value.argument_name == argument_name
    assert exc_info.value.argument_value == argument_value


@pytest.mark.parametrize(("call", "expected_fragments", "expected_valid_values"), INVALID_ARGUMENT_MESSAGE_CASES)
def test_invalid_argument_message_names_the_value_and_remediation(
    valid_precip_data, valid_pet_data, call, expected_fragments, expected_valid_values
) -> None:
    """Error messages carry the valid range and the remediation users should apply."""
    with pytest.raises(InvalidArgumentError) as exc_info:
        call(valid_precip_data, valid_pet_data)
    message = str(exc_info.value)
    for fragment in expected_fragments:
        assert fragment in message
    assert exc_info.value.valid_values == expected_valid_values


@pytest.mark.parametrize("scale", [1, 72])
def test_scale_boundary_values_are_accepted(valid_precip_data, scale: int) -> None:
    """The documented scale range is inclusive on both ends."""
    indices.spi(
        valid_precip_data,
        scale,
        indices.Distribution.gamma,
        2000,
        2000,
        2009,
        compute.Periodicity.monthly,
    )


@pytest.mark.parametrize(("call", "argument_name"), CATCH_ALL_CASES)
def test_validation_errors_are_catchable_as_the_base(valid_precip_data, valid_pet_data, call, argument_name) -> None:
    """Callers can catch every validation failure as the library base error."""
    with pytest.raises(ClimateIndicesError):
        call(valid_precip_data, valid_pet_data)

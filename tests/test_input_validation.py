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


# Named invalid public-API calls; the tables below reference these so an API
# signature change is edited in one place.
INVALID_CALLS = {
    "spi-scale-zero": lambda p, pet: indices.spi(
        p, 0, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly
    ),
    "spi-scale-above-maximum": lambda p, pet: indices.spi(
        p, 73, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly
    ),
    "spi-scale-negative": lambda p, pet: indices.spi(
        p, -5, indices.Distribution.gamma, 2000, 2000, 2009, compute.Periodicity.monthly
    ),
    "percentage-of-normal-scale-none": lambda p, pet: indices.percentage_of_normal(
        p, None, 2000, 2000, 2009, compute.Periodicity.monthly
    ),
    "percentage-of-normal-scale-negative": lambda p, pet: indices.percentage_of_normal(
        p, -1, 2000, 2000, 2009, compute.Periodicity.monthly
    ),
    "spei-scale-zero": lambda p, pet: indices.spei(
        p, pet, 0, indices.Distribution.gamma, compute.Periodicity.monthly, 2000, 2000, 2009
    ),
    "spi-distribution-none": lambda p, pet: indices.spi(p, 6, None, 2000, 2000, 2009, compute.Periodicity.monthly),
    "spi-distribution-string": lambda p, pet: indices.spi(p, 6, "gamma", 2000, 2000, 2009, compute.Periodicity.monthly),
    "spei-distribution-invalid": lambda p, pet: indices.spei(
        p, pet, 6, "invalid", compute.Periodicity.monthly, 2000, 2000, 2009
    ),
    "spei-distribution-none": lambda p, pet: indices.spei(
        p, pet, 6, None, compute.Periodicity.monthly, 2000, 2000, 2009
    ),
    "spi-periodicity-monthly": lambda p, pet: indices.spi(
        p, 6, indices.Distribution.gamma, 2000, 2000, 2009, "monthly"
    ),
    "spi-periodicity-invalid": lambda p, pet: indices.spi(
        p, 6, indices.Distribution.gamma, 2000, 2000, 2009, "invalid"
    ),
    "percentage-of-normal-periodicity-none": lambda p, pet: indices.percentage_of_normal(p, 6, 2000, 2000, 2009, None),
    "percentage-of-normal-periodicity-daily": lambda p, pet: indices.percentage_of_normal(
        p, 6, 2000, 2000, 2009, "daily"
    ),
    "percentage-of-normal-periodicity-unsupported": lambda p, pet: indices.percentage_of_normal(
        p, 6, 2000, 2000, 2009, "unsupported"
    ),
    "spei-periodicity-monthly": lambda p, pet: indices.spei(
        p, pet, 6, indices.Distribution.gamma, "monthly", 2000, 2000, 2009
    ),
    "spei-periodicity-unsupported": lambda p, pet: indices.spei(
        p, pet, 6, indices.Distribution.gamma, "unsupported", 2000, 2000, 2009
    ),
}

# (call, argument name, argument value) rows: one row per invalid argument.
INVALID_ARGUMENT_CASES = [
    pytest.param(INVALID_CALLS["spi-scale-zero"], "scale", "0", id="spi-scale-zero"),
    pytest.param(INVALID_CALLS["spi-scale-above-maximum"], "scale", "73", id="spi-scale-above-maximum"),
    pytest.param(INVALID_CALLS["spi-scale-negative"], "scale", "-5", id="spi-scale-negative"),
    pytest.param(
        INVALID_CALLS["percentage-of-normal-scale-none"], "scale", "None", id="percentage-of-normal-scale-none"
    ),
    pytest.param(
        INVALID_CALLS["percentage-of-normal-scale-negative"], "scale", "-1", id="percentage-of-normal-scale-negative"
    ),
    pytest.param(INVALID_CALLS["spei-scale-zero"], "scale", "0", id="spei-scale-zero"),
    pytest.param(INVALID_CALLS["spi-distribution-none"], "distribution", "None", id="spi-distribution-none"),
    pytest.param(INVALID_CALLS["spi-distribution-string"], "distribution", "gamma", id="spi-distribution-string"),
    pytest.param(INVALID_CALLS["spei-distribution-invalid"], "distribution", "invalid", id="spei-distribution-invalid"),
    pytest.param(INVALID_CALLS["spei-distribution-none"], "distribution", "None", id="spei-distribution-none"),
    pytest.param(INVALID_CALLS["spi-periodicity-monthly"], "periodicity", "monthly", id="spi-periodicity-monthly"),
    pytest.param(INVALID_CALLS["spi-periodicity-invalid"], "periodicity", "invalid", id="spi-periodicity-invalid"),
    pytest.param(
        INVALID_CALLS["percentage-of-normal-periodicity-none"],
        "periodicity",
        "None",
        id="percentage-of-normal-periodicity-none",
    ),
    pytest.param(
        INVALID_CALLS["percentage-of-normal-periodicity-daily"],
        "periodicity",
        "daily",
        id="percentage-of-normal-periodicity-daily",
    ),
    pytest.param(INVALID_CALLS["spei-periodicity-monthly"], "periodicity", "monthly", id="spei-periodicity-monthly"),
    pytest.param(
        INVALID_CALLS["spei-periodicity-unsupported"],
        "periodicity",
        "unsupported",
        id="spei-periodicity-unsupported",
    ),
]

# (call, message fragments, valid_values) rows: the remediation text users act on.
INVALID_ARGUMENT_MESSAGE_CASES = [
    pytest.param(
        INVALID_CALLS["spi-scale-zero"],
        ("[1, 72]", "1 (monthly)", "3 (seasonal)", "6 (half-year)", "12 (annual)"),
        "[1, 72]",
        id="scale",
    ),
    pytest.param(
        INVALID_CALLS["spi-distribution-none"],
        ("gamma", "pearson", "indices.Distribution.gamma", "indices.Distribution.pearson"),
        "gamma, pearson",
        id="distribution",
    ),
    pytest.param(
        INVALID_CALLS["spi-periodicity-monthly"],
        ("monthly", "daily", "compute.Periodicity.monthly", "compute.Periodicity.daily"),
        "monthly, daily",
        id="periodicity",
    ),
]

# (call, argument name) rows proving every validator routes through the base error.
CATCH_ALL_CASES = [
    pytest.param(INVALID_CALLS["spi-scale-zero"], "scale", id="scale"),
    pytest.param(INVALID_CALLS["spi-distribution-none"], "distribution", id="distribution"),
    pytest.param(INVALID_CALLS["percentage-of-normal-periodicity-unsupported"], "periodicity", id="periodicity"),
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
    with pytest.raises(ClimateIndicesError) as exc_info:
        call(valid_precip_data, valid_pet_data)
    assert isinstance(exc_info.value, InvalidArgumentError)
    assert exc_info.value.argument_name == argument_name

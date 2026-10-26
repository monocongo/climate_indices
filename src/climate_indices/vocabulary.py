"""Shared domain types and validation, independent of the computation and public APIs."""

from enum import Enum
from typing import Literal, get_args

from climate_indices.exceptions import InvalidArgumentError

__all__ = [
    "Aggregation",
    "Distribution",
    "OutputScale",
    "OUTPUT_SCALES",
    "Periodicity",
    "ZeroHandling",
    "validate_output_scale",
]


class Distribution(Enum):
    """
    Enumeration type for distribution fittings used for SPI and SPEI.
    """

    pearson = "pearson"
    gamma = "gamma"
    #: Generalized logistic (Hosking's GLO), the "log-Logistic" distribution R's
    #: SPEI package and SPEIbase fit; see ADR-0016. Supported by ``spei()`` only.
    loglogistic = "loglogistic"
    #: The empirical (rank-based) distribution: plotting-position probabilities over
    #: the Calibration Period, with no fitted parameters (#1168). Supported by
    #: ``standardized_index()`` only.
    empirical = "empirical"
    #: The beta distribution, the bounded [0, 1] family a capacity factor or another
    #: bounded series is standardized with; see ADR-0024. Supported by
    #: ``standardized_index()`` only.
    beta = "beta"

    @property
    def display_name(self) -> str:
        """Human-readable name for prose and metadata ("log-logistic", not "loglogistic")."""
        return _DISTRIBUTION_DISPLAY_NAMES[self]


#: human-readable distribution names; every member but ``loglogistic`` is its own value
_DISTRIBUTION_DISPLAY_NAMES = {
    Distribution.pearson: "pearson",
    Distribution.gamma: "gamma",
    Distribution.loglogistic: "log-logistic",
    Distribution.empirical: "empirical",
    Distribution.beta: "beta",
}


class Periodicity(Enum):
    """
    Enumeration type for specifying dataset periodicity.

    'monthly' indicates an array of monthly values, assumed to span full years,
    i.e. the first value corresponds to January of the initial year and any
    missing final months of the final year filled with NaN values,
    with size == # of years * 12

    'daily' indicates an array of full years of daily values with 366 days per year,
    as if each year were a leap year and any missing final months of the final
    year filled with NaN values, with array size == (# years * 366)

    'hourly' indicates an array of full years of hourly values with 366 days of 24
    hours per year, i.e. 8784 values per year, on the same 366-day calendar, so a
    non-leap year carries 24 synthesized February 29 values, and any missing final
    days of the final year filled with NaN values, with array size == (# years * 8784)
    """

    monthly = 12
    daily = 366
    hourly = 8784

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(s: str) -> "Periodicity":
        try:
            return Periodicity[s]
        except KeyError as err:
            raise ValueError(f"No periodicity enumeration corresponding to {s}") from err

    def unit(self) -> str:
        if self.name == "monthly":
            unit = "month"
        elif self.name == "daily":
            unit = "day"
        elif self.name == "hourly":
            unit = "hour"
        else:
            raise ValueError(f"No periodicity unit corresponding to {self.name}")

        return unit

    @property
    def period_length(self) -> int:
        """
        The number of time steps comprising one year at this periodicity,
        i.e. 12 for monthly data, 366 for daily data, and 8784 for hourly data.
        """
        return int(self.value)


# the valid number of time steps per year, i.e. the length of the second axis
# of a 2-D (years, periods) input array
_PERIOD_LENGTHS = frozenset(periodicity.period_length for periodicity in Periodicity)

#: Where a zero accumulation is placed within the probability mass at zero, see
#: ADR-0015: "classic" scores it Φ⁻¹(p0), "center_of_mass" Φ⁻¹(p0 / 2), and
#: "mean_zero" −φ(Φ⁻¹(p0)) / p0.
ZeroHandling = Literal["classic", "center_of_mass", "mean_zero"]

_ZERO_HANDLING_MODES: tuple[str, ...] = get_args(ZeroHandling)

#: How each time step's ``scale`` values are combined before the fit, see ADR-0019.
#: "sum" accumulates them, the SPI/SPEI convention; "mean" averages them, which keeps a
#: bounded series such as a capacity factor in its own range. It is chosen independently
#: of the distribution, since the two answer different questions about the same window.
Aggregation = Literal["sum", "mean"]

_AGGREGATIONS: tuple[str, ...] = get_args(Aggregation)

# the output conventions the standardized indices return: the standard-normal
# z-score ("normal", the default), the fitted cumulative probability
# ("probability", the PIT value in [0, 1]), and its signed counterpart
# ("bounded", 2p - 1 in [-1, 1])
OutputScale = Literal["normal", "probability", "bounded"]
OUTPUT_SCALES: tuple[OutputScale, ...] = ("normal", "probability", "bounded")


def validate_output_scale(output_scale: str) -> None:
    """Validate that an output scale is one of the accepted values.

    :param output_scale: the output-scale value to validate
    :raises InvalidArgumentError: if the value is not one of ``OUTPUT_SCALES``
    """
    if output_scale not in OUTPUT_SCALES:
        raise InvalidArgumentError(
            f"Invalid output_scale argument: {output_scale!r}. Supported values: {', '.join(OUTPUT_SCALES)}.",
            argument_name="output_scale",
            argument_value=repr(output_scale),
            valid_values=", ".join(OUTPUT_SCALES),
        )


def _validate_zero_handling(zero_handling: str) -> None:
    """Reject a zero-handling mode other than the three ADR-0015 names."""
    if not isinstance(zero_handling, str) or zero_handling not in _ZERO_HANDLING_MODES:
        raise ValueError(
            f"Invalid zero_handling argument: {zero_handling!r} -- "
            "must be one of 'classic', 'center_of_mass', or 'mean_zero'"
        )


def _validate_aggregation(aggregation: str) -> None:
    """Reject an aggregation other than the two ADR-0019 operations."""
    if not isinstance(aggregation, str) or aggregation not in _AGGREGATIONS:
        raise ValueError(f"Invalid aggregation argument: {aggregation!r} -- must be one of 'sum' or 'mean'")


def _validate_hourly_distribution(periodicity: Periodicity, distribution: Distribution) -> None:
    """Reject an hourly series outside the rank-based empirical transform.

    An hourly calendar step carries only a few dozen calibration values over a normal
    record, which is too few for a parametric fit; the empirical rank transform needs
    no fit and is the only supported hourly standard. See ADR-0019.

    Args:
        periodicity: The periodicity of the series
        distribution: The distribution requested for the standard

    Raises:
        InvalidArgumentError: If the periodicity is hourly and the distribution is not empirical
    """
    if periodicity is Periodicity.hourly and distribution is not Distribution.empirical:
        raise InvalidArgumentError(
            "Hourly periodicity supports only the empirical distribution.",
            argument_name="distribution",
            argument_value=str(distribution),
            valid_values="Distribution.empirical",
        )


# Preserve pickle globals and autodoc paths at the established public re-exports.
Distribution.__module__ = "climate_indices.indices"
Periodicity.__module__ = "climate_indices.compute"
validate_output_scale.__module__ = "climate_indices.compute"

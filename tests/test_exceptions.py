"""Table-driven tests for the custom exception and warning hierarchy.

Each table below is the single owner of one contract: the class hierarchy,
catchability, context attributes, pickling, and the deprecation helper.
"""

from __future__ import annotations

import inspect
import pickle
import warnings

import pytest

from climate_indices import ClimateIndicesError, compute, exceptions

# (class, parent, expected) rows. A parent given as a tuple compares exact
# __bases__; expected False pins the branch separation (warnings are not
# exceptions and vice versa).
HIERARCHY_CASES = [
    # every custom exception derives from the library base
    (ClimateIndicesError, Exception, True),
    (exceptions.DistributionFittingError, ClimateIndicesError, True),
    (exceptions.InsufficientDataError, ClimateIndicesError, True),
    (exceptions.PearsonFittingError, ClimateIndicesError, True),
    (exceptions.DimensionMismatchError, ClimateIndicesError, True),
    (exceptions.CoordinateValidationError, ClimateIndicesError, True),
    (exceptions.InputTypeError, ClimateIndicesError, True),
    (exceptions.InvalidArgumentError, ClimateIndicesError, True),
    (exceptions.ConvergenceError, ClimateIndicesError, True),
    (exceptions.PeriodicityError, ClimateIndicesError, True),
    (exceptions.DataShapeError, ClimateIndicesError, True),
    # fitting-domain subtypes; ConvergenceError covers iterative fitting failures,
    # not general convergence outside the fitting domain
    (exceptions.InsufficientDataError, exceptions.DistributionFittingError, True),
    (exceptions.PearsonFittingError, exceptions.DistributionFittingError, True),
    (exceptions.ConvergenceError, exceptions.DistributionFittingError, True),
    # non-fitting exception types are direct children of the base
    (exceptions.DimensionMismatchError, exceptions.DistributionFittingError, False),
    (exceptions.CoordinateValidationError, exceptions.DistributionFittingError, False),
    (exceptions.InputTypeError, exceptions.DistributionFittingError, False),
    (exceptions.InvalidArgumentError, exceptions.DistributionFittingError, False),
    (exceptions.DataShapeError, exceptions.DistributionFittingError, False),
    (exceptions.DataShapeError, exceptions.DimensionMismatchError, False),
    (exceptions.PeriodicityError, exceptions.InvalidArgumentError, True),
    # custom warnings stay on their own branch, rooted at UserWarning
    (exceptions.ClimateIndicesWarning, UserWarning, True),
    (exceptions.MissingDataWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.ShortCalibrationWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.GoodnessOfFitWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.InputAlignmentWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.BetaFeatureWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.ClimateIndicesDeprecationWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.ClimateIndicesDeprecationWarning, DeprecationWarning, True),
    (exceptions.ClimateIndicesWarning, ClimateIndicesError, False),
    (exceptions.MissingDataWarning, ClimateIndicesError, False),
    (exceptions.ShortCalibrationWarning, ClimateIndicesError, False),
    (exceptions.GoodnessOfFitWarning, ClimateIndicesError, False),
    (exceptions.InputAlignmentWarning, ClimateIndicesError, False),
    (exceptions.BetaFeatureWarning, ClimateIndicesError, False),
    (exceptions.ClimateIndicesDeprecationWarning, ClimateIndicesError, False),
    (ClimateIndicesError, exceptions.ClimateIndicesWarning, False),
    (exceptions.DistributionFittingError, exceptions.ClimateIndicesWarning, False),
    (exceptions.InsufficientDataError, exceptions.ClimateIndicesWarning, False),
    # direct bases pin the warning classes that add no mixin
    (exceptions.MissingDataWarning, (exceptions.ClimateIndicesWarning,), True),
    (exceptions.ShortCalibrationWarning, (exceptions.ClimateIndicesWarning,), True),
]

# (class, base, is_warning) rows: what a user catches each type as.
CATCHABILITY_CASES = [
    (exceptions.DistributionFittingError, ClimateIndicesError, False),
    (exceptions.InsufficientDataError, ClimateIndicesError, False),
    (exceptions.PearsonFittingError, ClimateIndicesError, False),
    (exceptions.DimensionMismatchError, ClimateIndicesError, False),
    (exceptions.CoordinateValidationError, ClimateIndicesError, False),
    (exceptions.InputTypeError, ClimateIndicesError, False),
    (exceptions.InvalidArgumentError, ClimateIndicesError, False),
    (exceptions.ConvergenceError, ClimateIndicesError, False),
    (exceptions.PeriodicityError, ClimateIndicesError, False),
    (exceptions.DataShapeError, ClimateIndicesError, False),
    (exceptions.MissingDataWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.ShortCalibrationWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.GoodnessOfFitWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.InputAlignmentWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.BetaFeatureWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.ClimateIndicesDeprecationWarning, exceptions.ClimateIndicesWarning, True),
    (exceptions.ClimateIndicesDeprecationWarning, DeprecationWarning, True),
]

# (warning class, message) rows emitted when checking base-category filtering.
LIBRARY_WARNINGS = [
    (exceptions.MissingDataWarning, "missing data"),
    (exceptions.ShortCalibrationWarning, "short calibration"),
    (exceptions.GoodnessOfFitWarning, "poor fit"),
    (exceptions.InputAlignmentWarning, "alignment needed"),
    (exceptions.BetaFeatureWarning, "beta feature"),
    (exceptions.ClimateIndicesDeprecationWarning, "deprecated"),
]

# (class, init kwargs, attributes always set on a bare instance, fields copied
# to other fields) rows covering every context attribute contract.
ATTRIBUTE_CASES = [
    (
        exceptions.InsufficientDataError,
        {"non_zero_count": 5, "required_count": 10},
        {},
        {},
    ),
    (
        exceptions.PearsonFittingError,
        {"underlying_error": ValueError("original error")},
        {},
        {},
    ),
    (
        exceptions.DimensionMismatchError,
        {"expected_dims": (10, 20), "actual_dims": (10, 15)},
        {},
        {},
    ),
    (
        exceptions.CoordinateValidationError,
        {"coordinate_name": "time", "reason": "Non-monotonic values"},
        {},
        {},
    ),
    (
        exceptions.InputTypeError,
        {"expected_type": int, "actual_type": str},
        {},
        {},
    ),
    (
        exceptions.InvalidArgumentError,
        {"argument_name": "scale", "argument_value": "0", "valid_values": "[1, 72]"},
        {},
        {},
    ),
    (
        exceptions.DistributionFittingError,
        {
            "distribution_name": "gamma",
            "input_shape": (10, 12),
            "parameters": {"alpha": "0.5", "beta": "1.0"},
            "suggestion": "try pearson3",
            "underlying_error": ValueError("test error"),
        },
        {},
        {},
    ),
    (
        exceptions.ConvergenceError,
        {
            "algorithm": "L-moments",
            "iterations": 50,
            "distribution_name": "gamma",
            "underlying_error": ValueError("numerical overflow"),
        },
        {},
        {},
    ),
    (
        exceptions.PeriodicityError,
        {"periodicity_value": "weekly"},
        {
            "argument_name": "periodicity",
            "argument_value": None,
            "valid_values": "Periodicity.monthly, Periodicity.daily",
        },
        {"periodicity_value": "argument_value"},
    ),
    (
        exceptions.DataShapeError,
        {"expected_shape": "(years, 12)", "actual_shape": (100, 13)},
        {},
        {},
    ),
    (
        exceptions.InputAlignmentWarning,
        {"original_size": 100, "aligned_size": 80, "dropped_count": 20},
        {},
        {},
    ),
    (
        exceptions.MissingDataWarning,
        {"missing_ratio": 0.15, "threshold": 0.20},
        {},
        {},
    ),
    (
        exceptions.ShortCalibrationWarning,
        {"actual_years": 25, "required_years": 30},
        {},
        {},
    ),
    (
        exceptions.GoodnessOfFitWarning,
        {"distribution_name": "gamma", "p_value": 0.03, "threshold": 0.05, "poor_fit_count": 15, "total_steps": 100},
        {},
        {},
    ),
    (
        exceptions.ClimateIndicesDeprecationWarning,
        {
            "deprecated_in": "2.3.0",
            "removal_version": "3.0.0",
            "alternative": "Use new_api instead",
            "migration_url": "https://docs.example.com/migration",
        },
        {},
        {},
    ),
]

# (class, positional args) rows: context attributes must be keyword-only.
KEYWORD_ONLY_CASES = [
    (exceptions.InvalidArgumentError, ("message", "scale")),
    (exceptions.ConvergenceError, ("message", "L-moments")),
    (exceptions.PeriodicityError, ("message", "weekly")),
    (exceptions.DataShapeError, ("message", "(years, 12)")),
    (exceptions.MissingDataWarning, ("message", 0.15)),
    (exceptions.ShortCalibrationWarning, ("message", 25)),
    (exceptions.GoodnessOfFitWarning, ("message", "gamma")),
    (exceptions.InputAlignmentWarning, ("message", 100)),
    (exceptions.ClimateIndicesDeprecationWarning, ("message", "2.3.0")),
]

# (class, init args, init kwargs) rows: Dask pickles exceptions across workers.
PICKLE_CASES = [
    (exceptions.ClimateIndicesError, ("base error",), {}),
    (exceptions.DistributionFittingError, ("fitting failed",), {}),
    (exceptions.InsufficientDataError, ("not enough data",), {"non_zero_count": 5, "required_count": 10}),
    (exceptions.PearsonFittingError, ("pearson failed",), {}),
    (exceptions.DimensionMismatchError, ("dims don't match",), {"expected_dims": (10, 20), "actual_dims": (10, 30)}),
    (exceptions.CoordinateValidationError, ("bad coords",), {"coordinate_name": "time", "reason": "not monotonic"}),
    (exceptions.InputTypeError, ("wrong type",), {"expected_type": type(None), "actual_type": type([])}),
    (
        exceptions.InvalidArgumentError,
        ("bad arg",),
        {"argument_name": "scale", "argument_value": "-1", "valid_values": "positive integers"},
    ),
    (
        exceptions.ConvergenceError,
        ("convergence failed",),
        {"algorithm": "L-moments", "iterations": 100, "distribution_name": "gamma"},
    ),
    (exceptions.PeriodicityError, ("invalid periodicity",), {"periodicity_value": "weekly"}),
    (exceptions.DataShapeError, ("wrong shape",), {"expected_shape": "(years, 12)", "actual_shape": (100, 13)}),
    (exceptions.ClimateIndicesWarning, ("base warning",), {}),
    (exceptions.MissingDataWarning, ("missing data",), {"missing_ratio": 0.15, "threshold": 0.20}),
    (exceptions.ShortCalibrationWarning, ("short calibration",), {"actual_years": 25, "required_years": 30}),
    (
        exceptions.GoodnessOfFitWarning,
        ("poor fit",),
        {"distribution_name": "gamma", "p_value": 0.03, "threshold": 0.05},
    ),
    (
        exceptions.InputAlignmentWarning,
        ("alignment needed",),
        {"original_size": 100, "aligned_size": 80, "dropped_count": 20},
    ),
    (exceptions.BetaFeatureWarning, ("beta feature",), {}),
    (
        exceptions.ClimateIndicesDeprecationWarning,
        ("deprecated feature",),
        {
            "deprecated_in": "2.3.0",
            "removal_version": "3.0.0",
            "alternative": "Use new_feature instead",
            "migration_url": "https://example.com/guide",
        },
    ),
]

# emit_deprecation_warning: (message kwargs, expected message fragments) rows.
EMIT_CASES = [
    pytest.param(
        {},
        (
            "Parameter 'old_param'",
            "deprecated since version 2.3.0",
            "Use 'new_param' instead",
            "removed in version 3.0.0",
            "Migration guide:",
        ),
        id="full-message",
    ),
    pytest.param(
        {"migration_url": None},
        ("https://climate-indices.readthedocs.io/en/stable/deprecations",),
        id="default-url",
    ),
    pytest.param(
        {"migration_url": "api-changes.html"},
        ("https://climate-indices.readthedocs.io/en/stable/deprecations/api-changes.html",),
        id="relative-url",
    ),
    pytest.param(
        {"migration_url": "https://example.com/custom/migration/guide"},
        ("https://example.com/custom/migration/guide",),
        id="absolute-url",
    ),
]

EMIT_BASE_KWARGS = {
    "feature": "Parameter 'old_param'",
    "alternative": "Use 'new_param' instead",
    "deprecated_in": "2.3.0",
    "removal_version": "3.0.0",
    "stacklevel": 2,
}


def _same(actual, expected) -> bool:
    """Identity first: context values include exceptions, where == is not enough."""
    return actual is expected or actual == expected


@pytest.mark.parametrize(
    ("cls", "parent", "expected"),
    HIERARCHY_CASES,
    ids=lambda value: getattr(value, "__name__", None),
)
def test_hierarchy(cls, parent, expected) -> None:
    """Each class sits in exactly the documented place in the hierarchy."""
    matches = cls.__bases__ == parent if isinstance(parent, tuple) else issubclass(cls, parent)
    assert matches is expected


@pytest.mark.parametrize(
    ("cls", "base", "is_warning"),
    CATCHABILITY_CASES,
    ids=lambda value: getattr(value, "__name__", None),
)
def test_every_type_is_catchable_as_its_base(cls, base, is_warning) -> None:
    """Users catch the library base (or stdlib DeprecationWarning) for every library type."""
    if is_warning:
        with pytest.warns(base):
            warnings.warn("test warning", cls, stacklevel=2)
    else:
        with pytest.raises(base):
            raise cls("test error")


def test_filtering_the_base_warning_suppresses_every_library_warning() -> None:
    """One filterwarnings call silences every library warning subtype."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=exceptions.ClimateIndicesWarning)
        for warning_class, message in LIBRARY_WARNINGS:
            warnings.warn(message, warning_class, stacklevel=2)
        assert recorded == []


@pytest.mark.parametrize("category", [exceptions.ClimateIndicesWarning, DeprecationWarning])
@pytest.mark.parametrize("emit", ["warn", "helper"])
def test_deprecation_warning_is_filterable_by_both_bases(category: type, emit: str) -> None:
    """Deprecation warnings, direct or via the helper, respect either base class filter."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=category)
        if emit == "warn":
            warnings.warn("deprecated", exceptions.ClimateIndicesDeprecationWarning, stacklevel=2)
        else:
            exceptions.emit_deprecation_warning(
                feature="test",
                alternative="use other",
                deprecated_in="1.0.0",
                removal_version="2.0.0",
            )
        assert recorded == []


@pytest.mark.parametrize(
    ("cls", "init_kwargs", "defaults", "copies"),
    ATTRIBUTE_CASES,
    ids=lambda value: getattr(value, "__name__", None),
)
def test_context_attributes_are_stored(cls, init_kwargs, defaults, copies) -> None:
    """Context attributes are stored on the instance and echoed by repr."""
    message = "context check"
    context = cls(message, **init_kwargs)
    assert str(context) == message
    assert cls.__name__ in repr(context)
    assert message in repr(context)
    for name, value in init_kwargs.items():
        assert _same(getattr(context, name), value)
    for name, value in defaults.items():
        if value is not None:
            assert getattr(context, name) == value
    for source, target in copies.items():
        assert getattr(context, target) == init_kwargs[source]


@pytest.mark.parametrize(
    ("cls", "init_kwargs", "defaults", "copies"),
    ATTRIBUTE_CASES,
    ids=lambda value: getattr(value, "__name__", None),
)
def test_context_attributes_default_to_none(cls, init_kwargs, defaults, copies) -> None:
    """A bare instance leaves constructor attributes unset, except values the subclass pins."""
    context = cls("context check")
    for name in set(init_kwargs) | set(defaults):
        if name in defaults:
            assert getattr(context, name) == defaults[name]
        else:
            assert getattr(context, name) is None


@pytest.mark.parametrize(
    ("cls", "positional_args"),
    KEYWORD_ONLY_CASES,
    ids=lambda value: getattr(value, "__name__", None),
)
def test_context_attributes_reject_positional_arguments(cls, positional_args) -> None:
    """Context attributes require keywords, so a stray positional argument cannot shift them."""
    with pytest.raises(TypeError, match="positional"):
        cls(*positional_args)


@pytest.mark.parametrize(
    ("cls", "init_args", "init_kwargs"),
    PICKLE_CASES,
    ids=lambda value: getattr(value, "__name__", None),
)
def test_pickle_roundtrip_preserves_type_message_and_context(cls, init_args, init_kwargs) -> None:
    """Dask serializes these across workers, so pickling must preserve their context."""
    original = cls(*init_args, **init_kwargs)
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is type(original)
    assert str(restored) == str(original)
    for name, value in init_kwargs.items():
        assert hasattr(restored, name)
        assert getattr(restored, name) == value


def test_module_all_lists_exactly_the_public_types_and_helper() -> None:
    """__all__ is the public surface: complete, resolvable, and covered by the contract tables."""
    expected_names = {
        "ClimateIndicesError",
        "ConvergenceError",
        "DataShapeError",
        "DistributionFittingError",
        "InsufficientDataError",
        "PearsonFittingError",
        "PeriodicityError",
        "DimensionMismatchError",
        "CoordinateValidationError",
        "InputTypeError",
        "InvalidArgumentError",
        "ClimateIndicesWarning",
        "MissingDataWarning",
        "ShortCalibrationWarning",
        "GoodnessOfFitWarning",
        "InputAlignmentWarning",
        "BetaFeatureWarning",
        "ClimateIndicesDeprecationWarning",
        "emit_deprecation_warning",
    }
    assert set(exceptions.__all__) == expected_names
    for name in exceptions.__all__:
        exported = getattr(exceptions, name)
        assert isinstance(exported, type) or callable(exported), f"{name} is neither a class nor callable"

    # a new public type must join the hierarchy and pickle tables, not slip in untested
    public_types = {
        getattr(exceptions, name) for name in exceptions.__all__ if isinstance(getattr(exceptions, name), type)
    }
    assert public_types <= {row[0] for row in HIERARCHY_CASES}
    assert public_types <= {row[0] for row in PICKLE_CASES}


@pytest.mark.parametrize(("url_kwargs", "expected_fragments"), EMIT_CASES)
def test_emit_deprecation_warning_message(url_kwargs: dict, expected_fragments: tuple[str, ...]) -> None:
    """The helper builds one standardized message and migrates the URL forms users pass."""
    with pytest.warns(exceptions.ClimateIndicesDeprecationWarning) as record:
        exceptions.emit_deprecation_warning(**EMIT_BASE_KWARGS, **url_kwargs)
    assert len(record) == 1
    message = str(record[0].message)
    for fragment in expected_fragments:
        assert fragment in message


def test_emit_deprecation_warning_is_keyword_only() -> None:
    """All helper parameters are keyword-only, so a call site cannot mis-order them."""
    for name, parameter in inspect.signature(exceptions.emit_deprecation_warning).parameters.items():
        assert parameter.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"Parameter '{name}' should be KEYWORD_ONLY, got {parameter.kind.name}"
        )


def test_compute_module_reexports_remain_compatible() -> None:
    """The legacy compute import path stays identical, isinstance-compatible, and catchable."""
    assert compute.DistributionFittingError is exceptions.DistributionFittingError
    assert compute.InsufficientDataError is exceptions.InsufficientDataError
    assert compute.PearsonFittingError is exceptions.PearsonFittingError

    raised = compute.InsufficientDataError("test", non_zero_count=3)
    assert isinstance(raised, exceptions.InsufficientDataError)
    assert isinstance(raised, exceptions.DistributionFittingError)
    assert isinstance(raised, ClimateIndicesError)

    with pytest.raises(exceptions.InsufficientDataError):
        raise compute.InsufficientDataError("test error")
    with pytest.raises(ClimateIndicesError):
        raise compute.PearsonFittingError("test error")

    # pattern used in test_zero_precipitation_fix.py
    try:
        raise compute.InsufficientDataError("Insufficient data", non_zero_count=5, required_count=10)
    except compute.InsufficientDataError as e:
        assert e.non_zero_count == 5
        assert e.required_count == 10

"""Domain vocabulary compatibility and import-boundary regression tests."""

import ast
import inspect
import pickle
from types import ModuleType
from typing import get_args, get_type_hints

import pytest

from climate_indices import compute, indices, vocabulary
from climate_indices.exceptions import InvalidArgumentError


@pytest.mark.parametrize("name", ["Distribution", "Periodicity", "OutputScale", "ZeroHandling", "Aggregation"])
def test_compute_reexports_the_canonical_types(name: str) -> None:
    assert getattr(compute, name) is getattr(vocabulary, name)


def test_indices_reexports_distribution() -> None:
    assert indices.Distribution is vocabulary.Distribution


@pytest.mark.parametrize("name", ["pearson", "gamma", "loglogistic", "empirical", "beta"])
def test_distribution_legacy_pickle_and_metadata(name: str) -> None:
    member = vocabulary.Distribution[name]
    # Protocol-0 payloads captured before the move also pin the serialized module path.
    legacy = f"cclimate_indices.indices\nDistribution\np0\n(V{name}\np1\ntp2\nRp3\n.".encode()
    assert pickle.loads(legacy) is member
    assert pickle.dumps(member, protocol=0) == legacy
    assert pickle.loads(pickle.dumps(member)) is member
    assert member.value == name
    assert member.display_name == ("log-logistic" if name == "loglogistic" else name)
    assert repr(member) == f"<Distribution.{name}: '{name}'>"
    assert type(member).__module__ == "climate_indices.indices"


@pytest.mark.parametrize(
    "name,length,unit", [("monthly", 12, "month"), ("daily", 366, "day"), ("hourly", 8784, "hour")]
)
def test_periodicity_legacy_pickle_and_metadata(name: str, length: int, unit: str) -> None:
    member = vocabulary.Periodicity[name]
    legacy = f"cclimate_indices.compute\nPeriodicity\np0\n(I{length}\ntp1\nRp2\n.".encode()
    assert pickle.loads(legacy) is member
    assert pickle.dumps(member, protocol=0) == legacy
    assert pickle.loads(pickle.dumps(member)) is member
    assert member.period_length == length
    assert member.unit() == unit
    assert str(member) == name
    assert repr(member) == f"<Periodicity.{name}: {length}>"
    assert vocabulary.Periodicity.from_string(name) is member
    assert type(member).__module__ == "climate_indices.compute"


def test_validation_tables_keep_the_same_values() -> None:
    assert get_args(vocabulary.ZeroHandling) == ("classic", "center_of_mass", "mean_zero")
    assert get_args(vocabulary.Aggregation) == ("sum", "mean")
    assert get_args(vocabulary.OutputScale) == ("normal", "probability", "bounded")
    assert compute.OUTPUT_SCALES is vocabulary.OUTPUT_SCALES
    assert compute.validate_output_scale is vocabulary.validate_output_scale
    assert vocabulary.validate_output_scale.__module__ == "climate_indices.compute"
    for value in vocabulary.OUTPUT_SCALES:
        vocabulary.validate_output_scale(value)
    for value in get_args(vocabulary.ZeroHandling):
        vocabulary._validate_zero_handling(value)
    for value in get_args(vocabulary.Aggregation):
        vocabulary._validate_aggregation(value)


def test_validation_errors_keep_the_same_text() -> None:
    with pytest.raises(ValueError, match="^No periodicity enumeration corresponding to weekly$"):
        vocabulary.Periodicity.from_string("weekly")
    with pytest.raises(InvalidArgumentError) as error:
        vocabulary.validate_output_scale("invalid")
    assert (
        str(error.value) == "Invalid output_scale argument: 'invalid'. Supported values: normal, probability, bounded."
    )
    with pytest.raises(ValueError) as zero_error:
        vocabulary._validate_zero_handling("invalid")
    assert str(zero_error.value) == (
        "Invalid zero_handling argument: 'invalid' -- must be one of 'classic', 'center_of_mass', or 'mean_zero'"
    )
    with pytest.raises(ValueError) as aggregation_error:
        vocabulary._validate_aggregation("invalid")
    assert str(aggregation_error.value) == "Invalid aggregation argument: 'invalid' -- must be one of 'sum' or 'mean'"
    with pytest.raises(InvalidArgumentError) as error:
        vocabulary._validate_hourly_distribution(vocabulary.Periodicity.hourly, vocabulary.Distribution.gamma)
    assert str(error.value) == "Hourly periodicity supports only the empirical distribution."


def test_core_annotations_resolve_without_the_facade() -> None:
    hints = get_type_hints(compute.fit_and_standardize)
    assert hints["distribution"] is vocabulary.Distribution
    assert hints["periodicity"] is vocabulary.Periodicity


@pytest.mark.parametrize("module", [compute, vocabulary])
def test_imports_cannot_recreate_the_core_to_facade_cycle(module: ModuleType) -> None:
    # This architecture constraint includes function-local and TYPE_CHECKING imports.
    tree = ast.parse(inspect.getsource(module))
    forbidden = {"indices", "typed_public_api", "xarray_adapter"}
    if module is vocabulary:
        forbidden.add("compute")
    for node in ast.walk(tree):
        if module is vocabulary and isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module in {"enum", "typing", "climate_indices.exceptions"}
        if module is vocabulary and isinstance(node, ast.Import):
            assert {alias.name for alias in node.names} <= {"enum", "typing"}
        if isinstance(node, ast.ImportFrom):
            if node.module == "climate_indices":
                assert not forbidden.intersection(alias.name for alias in node.names)
            else:
                assert node.module not in {f"climate_indices.{name}" for name in forbidden}
            if node.level:
                assert node.module not in forbidden
                assert not forbidden.intersection(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            assert not {f"climate_indices.{name}" for name in forbidden}.intersection(
                alias.name for alias in node.names
            )

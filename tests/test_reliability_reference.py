"""Aggregate relationships and constructed scenarios, not an operational event oracle."""

import calendar
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import jsonschema
import pytest

from climate_indices import reliability
from climate_indices.exceptions import OverlappingEventsWarning

pytestmark = pytest.mark.validation
FIXTURE = Path(__file__).parent / "fixture" / "reliability_benchmarks"


def _load_reference(directory: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((directory / "reference.json").read_text()))


REFERENCE = _load_reference(FIXTURE)
# The recorded tolerances are the single source of truth for the assertions below.
PROVENANCE = cast(dict[str, Any], json.loads((FIXTURE / "provenance.json").read_text()))
TOLERANCE = PROVENANCE["validation_tolerance"]


def test_reference_panel_has_every_declared_year_and_scope() -> None:
    rows = REFERENCE["eia_panel"]
    expected = {(year, scope) for year in range(2015, 2025) for scope in ("all_events", "without_major_event_days")}
    assert len(rows) == len(expected)
    assert {(row["year"], row["scope"]) for row in rows} == expected


def test_provenance_checksum_and_fixed_tolerances() -> None:
    schema = json.loads((FIXTURE.parent / "provenance_schema.json").read_text())
    jsonschema.validate(PROVENANCE, schema)
    assert hashlib.sha256((FIXTURE / "reference.json").read_bytes()).hexdigest() == PROVENANCE["checksum_sha256"]
    assert PROVENANCE["license"]
    assert TOLERANCE == {
        "rtol": 0,
        "saidi_rounding_minutes": 0.05,
        "saifi_rounding": 0.0005,
        "caidi_rounding_minutes": 0.05,
        "arithmetic_atol": 1e-12,
        "accumulator_atol": 1e-9,
        "asai_atol": 1e-15,
        "ieee_atol": 0,
    }


def test_missing_reference_fails_instead_of_skipping(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_reference(tmp_path)


def _assert_eia_relationship(row: dict[str, Any]) -> None:
    saidi, saifi, caidi = row["saidi"], row["saifi"], row["caidi"]
    minutes_in_year = (366 if calendar.isleap(row["year"]) else 365) * 24 * 60
    assert all(math.isfinite(value) for value in (saidi, saifi, caidi))
    assert 0 < saidi <= minutes_in_year
    assert 0 < saifi < minutes_in_year / 5
    assert 5 < caidi <= minutes_in_year
    ds = TOLERANCE["saidi_rounding_minutes"]
    df = TOLERANCE["saifi_rounding"]
    dc = TOLERANCE["caidi_rounding_minutes"]
    ratio = saidi / saifi
    # Propagate half-last-digit rounding through both extremes of the ratio.
    atol = dc + max((saidi + ds) / (saifi - df) - ratio, ratio - (saidi - ds) / (saifi + df))
    assert caidi == pytest.approx(ratio, rel=0, abs=atol)


@pytest.mark.parametrize("row", REFERENCE["eia_panel"], ids=lambda row: f"{row['year']}-{row['scope']}")
def test_eia_aggregate_relationship_and_synthetic_accumulators(row: dict[str, Any]) -> None:
    _assert_eia_relationship(row)
    total_customers = 1000  # A constructed denominator, not EIA's actual customer population.
    ci = round(row["saifi"] * total_customers)
    assert total_customers < ci < 2 * total_customers
    duration = row["saidi"] / row["saifi"]
    events = [
        {
            "event_id": index,
            "start_time": f"{row['year']}-01-0{index + 1}T00:00:00Z",
            "duration": duration,
            "customers_interrupted": customers,
        }
        for index, customers in enumerate((total_customers, ci - total_customers))
    ]
    result = reliability.reliability_indices(
        events,
        total_customers=total_customers,
        period_start=f"{row['year']}-01-01T00:00:00Z",
        period_end=f"{row['year'] + 1}-01-01T00:00:00Z",
    )
    assert result.customer_interruptions == ci
    assert result.customer_minutes_interrupted == pytest.approx(
        row["saidi"] * total_customers, rel=TOLERANCE["rtol"], abs=TOLERANCE["accumulator_atol"]
    )
    assert result.saifi == pytest.approx(row["saifi"], rel=TOLERANCE["rtol"], abs=TOLERANCE["arithmetic_atol"])
    assert result.saidi == pytest.approx(row["saidi"], rel=TOLERANCE["rtol"], abs=TOLERANCE["arithmetic_atol"])
    assert result.caidi == pytest.approx(duration, rel=TOLERANCE["rtol"], abs=TOLERANCE["arithmetic_atol"])
    assert result.caidi == pytest.approx(
        result.saidi / result.saifi, rel=TOLERANCE["rtol"], abs=TOLERANCE["arithmetic_atol"]
    )
    assert result.outage_count == 2
    assert result.momentary_outage_count == 0
    assert result.maifi_e == 0
    assert result.maifi is None
    hours = (366 if calendar.isleap(row["year"]) else 365) * 24
    assert result.period_hours == hours
    assert result.asai == pytest.approx(
        1 - row["saidi"] / (60 * hours), rel=TOLERANCE["rtol"], abs=TOLERANCE["asai_atol"]
    )
    assert result.asai is not None
    assert 0 <= result.asai <= 1


@pytest.mark.parametrize("row", REFERENCE["eia_panel"], ids=lambda row: f"{row['year']}-{row['scope']}")
def test_corrupted_caidi_fails_the_same_relationship_check(row: dict[str, Any]) -> None:
    with pytest.raises(AssertionError):
        _assert_eia_relationship({**row, "caidi": row["caidi"] + 1.0})


@pytest.mark.parametrize("affected_fraction", [1.0, 0.5])
def test_epri_style_customer_normalized_outage_at_stated_frequency(affected_fraction: float) -> None:
    scenario = REFERENCE["epri_style_scenario"]
    events = [
        {**row, "customers_interrupted": int(row["customers_interrupted"] * affected_fraction)}
        for row in scenario["events"]
    ]
    # The rows represent disjoint restoration cohorts during one hazard event.
    with pytest.warns(OverlappingEventsWarning) as caught:
        result = reliability.reliability_indices(events, total_customers=scenario["total_customers"], period_hours=24)
    assert len(caught) == 1
    event_hours = result.customer_minutes_interrupted / (60 * scenario["total_customers"])
    assert event_hours == pytest.approx(
        scenario["customer_outage_hours_per_customer_per_event"] * affected_fraction,
        rel=TOLERANCE["rtol"],
        abs=TOLERANCE["arithmetic_atol"],
    )
    assert event_hours == pytest.approx(result.saidi / 60, rel=TOLERANCE["rtol"], abs=TOLERANCE["arithmetic_atol"])
    assert result.caidi / 60 == pytest.approx(2.0, rel=TOLERANCE["rtol"], abs=TOLERANCE["arithmetic_atol"])
    frequency = scenario["event_frequency_per_year"]
    expected_hours = scenario["expected_customer_outage_hours_per_customer_per_year"] * affected_fraction
    assert frequency * event_hours == pytest.approx(
        expected_hours, rel=TOLERANCE["rtol"], abs=TOLERANCE["arithmetic_atol"]
    )
    assert frequency * result.saifi * result.caidi / 60 == pytest.approx(
        expected_hours, rel=TOLERANCE["rtol"], abs=TOLERANCE["arithmetic_atol"]
    )
    assert result.period_hours == 24  # The caller's risk calculation does not annualize the library result.


def test_ieee_2012_published_momentary_example() -> None:
    example = REFERENCE["ieee_momentary_example"]
    result = reliability.reliability_indices(example["events"], total_customers=example["total_customers"])
    assert result.saifi == pytest.approx(example["expected_saifi"], rel=TOLERANCE["rtol"], abs=TOLERANCE["ieee_atol"])
    assert result.maifi == pytest.approx(example["expected_maifi"], rel=TOLERANCE["rtol"], abs=TOLERANCE["ieee_atol"])
    assert result.maifi_e == pytest.approx(
        example["expected_maifi_e"], rel=TOLERANCE["rtol"], abs=TOLERANCE["ieee_atol"]
    )
    assert result.outage_count == result.momentary_outage_count == 1

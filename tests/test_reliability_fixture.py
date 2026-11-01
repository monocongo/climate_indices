"""Schema and checksum checks for the hand-computed reliability fixture."""

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from climate_indices import reliability
from climate_indices.reliability._events import _normalize_events

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixture/reliability"
SCHEMA = ROOT / "src/climate_indices/reliability/event_schema.json"


def test_analytic_fixture_schema_provenance_and_arrays() -> None:
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    events = json.loads((FIXTURE / "events.json").read_text())
    jsonschema.validate(events, schema)
    provenance = json.loads((FIXTURE / "provenance.json").read_text())
    jsonschema.validate(provenance, json.loads((ROOT / "tests/fixture/provenance_schema.json").read_text()))
    assert hashlib.sha256((FIXTURE / "events.json").read_bytes()).hexdigest() == provenance["checksum_sha256"]
    result = _normalize_events(events, total_customers=1000)
    sustained = result[result["interruption_type"] == "sustained"]
    momentary = result[result["interruption_type"] == "momentary"]
    assert sustained["customers_interrupted"].sum() == 300
    assert (sustained["customers_interrupted"] * sustained["duration"]).sum() == 24000
    assert momentary["customers_interrupted"].sum() == 300
    assert (momentary["customers_interrupted"] * momentary["device_operations"]).sum() == 750
    assert result["duration"].dtype == np.float64
    assert result["customers_interrupted"].dtype == np.int64
    assert result["duration"].tolist()[-1] == 0
    pd_rows = result.to_dict(orient="records")
    again = reliability.classify_interruptions(pd_rows)
    np.testing.assert_array_equal(again["duration"], result["duration"])
    np.testing.assert_array_equal(again["customers_interrupted"], result["customers_interrupted"])


@pytest.mark.parametrize(
    "patch",
    [
        {"duration": -1},
        {"customers_interrupted": 1.5},
        {"event_id": None},
        {"start_time": "not-a-time"},
        {"planned": "false"},
        {"failure_rate": 0.2},
        {"component_ids": "line"},
        {"interruption_type": "brief"},
    ],
)
def test_event_schema_rejects_invalid_structure(patch: dict[str, object]) -> None:
    schema = json.loads(SCHEMA.read_text())
    row = {"event_id": "a", "start_time": "2024-01-01T00:00:00Z", "duration": 0, "customers_interrupted": 0, **patch}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([row], schema)


def test_event_schema_requires_a_duration_source() -> None:
    row = {"event_id": "a", "start_time": "2024-01-01T00:00:00Z", "customers_interrupted": 0}
    schema = json.loads(SCHEMA.read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([row], schema)
    jsonschema.validate([{**row, "end_time": row["start_time"]}], schema)

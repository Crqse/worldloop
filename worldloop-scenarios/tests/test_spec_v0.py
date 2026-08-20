"""Tests for ScenarioSpec v0 (S-01) + JSON Schema validation (S-02)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from worldloop_scenarios.spec import (
    ScenarioSpec,
    ScenarioMeta,
    TimeSpec,
    SpaceSpec,
    EntitiesSpec,
    ActionsSpec,
    TerminationSpec,
)
from worldloop_scenarios.schema_loader import (
    SchemaValidationError,
    load_spec_v0_schema,
    validate_against_schema,
)


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# S-01: ScenarioSpec dataclass
# ---------------------------------------------------------------------------


class TestScenarioSpecDataclass:
    """Tests for the ScenarioSpec dataclass (S-01)."""

    def test_spec_is_frozen(self) -> None:
        """Spec dataclasses MUST be immutable."""
        spec = ScenarioSpec(
            scenario=ScenarioMeta(scenario_id="t", scenario_version="0"),
            time=TimeSpec(max_ticks=10),
            space=SpaceSpec(type="discrete", shape=(5, 5)),
            entities=EntitiesSpec(
                columns=({"name": "energy", "dtype": "float"},),
                initial_count=1,
            ),
            actions=ActionsSpec(
                actions=(
                    {
                        "action_type": "noop",
                        "effects": (
                            {"target": "entity", "field": "energy", "op": "add", "value": 0.0},
                        ),
                    },
                )
            ),
            termination=TerminationSpec(
                stop_conditions=({"kind": "max_ticks"},)
            ),
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.scenario.scenario_id = "mutated"  # type: ignore[misc]

    def test_to_dict_round_trip(self) -> None:
        """to_dict → from_dict round-trip MUST preserve the spec."""
        original = ScenarioSpec(
            scenario=ScenarioMeta(scenario_id="round", scenario_version="0.1", tags=("a", "b")),
            time=TimeSpec(max_ticks=50),
            space=SpaceSpec(type="discrete", shape=(3, 3)),
            entities=EntitiesSpec(
                columns=({"name": "energy", "dtype": "float"},),
                initial_count=2,
            ),
            actions=ActionsSpec(
                actions=(
                    {
                        "action_type": "forage",
                        "effects": (
                            {"target": "entity", "field": "energy", "op": "add", "value": 5.0},
                        ),
                    },
                )
            ),
            termination=TerminationSpec(
                stop_conditions=({"kind": "max_ticks"}, {"kind": "no_alive"})
            ),
        )
        d = original.to_dict()
        restored = ScenarioSpec.from_dict(d)
        assert restored.scenario.scenario_id == "round"
        assert restored.scenario.tags == ("a", "b")
        assert restored.time.max_ticks == 50
        assert restored.space.shape == (3, 3)
        assert restored.entities.initial_count == 2
        assert restored.actions.actions[0]["action_type"] == "forage"
        assert len(restored.termination.stop_conditions) == 2

    def test_world_parameters_hash_stable(self) -> None:
        """Hash MUST be stable across runs (same structure → same hash)."""
        spec = ScenarioSpec(
            scenario=ScenarioMeta(scenario_id="h", scenario_version="0"),
            time=TimeSpec(max_ticks=10),
            space=SpaceSpec(type="discrete", shape=(5, 5)),
            entities=EntitiesSpec(
                columns=({"name": "energy", "dtype": "float"},),
                initial_count=1,
            ),
            actions=ActionsSpec(
                actions=(
                    {
                        "action_type": "noop",
                        "effects": (
                            {"target": "entity", "field": "energy", "op": "add", "value": 0.0},
                        ),
                    },
                )
            ),
            termination=TerminationSpec(
                stop_conditions=({"kind": "max_ticks"},)
            ),
        )
        h1 = spec.world_parameters_hash()
        h2 = spec.world_parameters_hash()
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_world_parameters_hash_excludes_metadata(self) -> None:
        """Hash MUST NOT change when only metadata (author, tags) changes."""
        base = ScenarioSpec(
            scenario=ScenarioMeta(scenario_id="x", scenario_version="0", author="A", tags=("t1",)),
            time=TimeSpec(max_ticks=10),
            space=SpaceSpec(type="discrete", shape=(5, 5)),
            entities=EntitiesSpec(
                columns=({"name": "energy", "dtype": "float"},),
                initial_count=1,
            ),
            actions=ActionsSpec(
                actions=(
                    {
                        "action_type": "noop",
                        "effects": (
                            {"target": "entity", "field": "energy", "op": "add", "value": 0.0},
                        ),
                    },
                )
            ),
            termination=TerminationSpec(
                stop_conditions=({"kind": "max_ticks"},)
            ),
        )
        # Same structure + dynamics, different metadata.
        modified = ScenarioSpec(
            scenario=ScenarioMeta(scenario_id="x", scenario_version="0", author="B", tags=("t2", "t3")),
            time=base.time,
            space=base.space,
            entities=base.entities,
            actions=base.actions,
            termination=base.termination,
        )
        assert base.world_parameters_hash() == modified.world_parameters_hash()

    def test_world_parameters_hash_changes_with_dynamics(self) -> None:
        """Hash MUST change when dynamics (max_ticks) change."""
        s1 = ScenarioSpec(
            scenario=ScenarioMeta(scenario_id="x", scenario_version="0"),
            time=TimeSpec(max_ticks=10),
            space=SpaceSpec(type="discrete", shape=(5, 5)),
            entities=EntitiesSpec(
                columns=({"name": "energy", "dtype": "float"},),
                initial_count=1,
            ),
            actions=ActionsSpec(
                actions=(
                    {
                        "action_type": "noop",
                        "effects": (
                            {"target": "entity", "field": "energy", "op": "add", "value": 0.0},
                        ),
                    },
                )
            ),
            termination=TerminationSpec(
                stop_conditions=({"kind": "max_ticks"},)
            ),
        )
        s2 = ScenarioSpec(
            scenario=s1.scenario,
            time=TimeSpec(max_ticks=20),  # changed
            space=s1.space,
            entities=s1.entities,
            actions=s1.actions,
            termination=s1.termination,
        )
        assert s1.world_parameters_hash() != s2.world_parameters_hash()


# ---------------------------------------------------------------------------
# S-02: JSON Schema validation
# ---------------------------------------------------------------------------


class TestSchemaLoader:
    """Tests for the JSON Schema loader (S-02)."""

    def test_load_schema_returns_dict(self) -> None:
        schema = load_spec_v0_schema()
        assert isinstance(schema, dict)
        assert schema["title"] == "WorldLoop ScenarioSpec v0"
        assert "properties" in schema

    def test_validate_accepts_discrete_grid(self) -> None:
        """discrete_grid.yaml MUST pass schema validation."""
        spec_dict = yaml.safe_load((EXAMPLES_DIR / "discrete_grid.yaml").read_text(encoding="utf-8"))
        validate_against_schema(spec_dict)  # should not raise

    def test_validate_accepts_continuous_field(self) -> None:
        """continuous_field.yaml MUST pass schema validation."""
        spec_dict = yaml.safe_load((EXAMPLES_DIR / "continuous_field.yaml").read_text(encoding="utf-8"))
        validate_against_schema(spec_dict)

    def test_validate_accepts_graph_registry(self) -> None:
        """graph_registry.yaml MUST pass schema validation."""
        spec_dict = yaml.safe_load((EXAMPLES_DIR / "graph_registry.yaml").read_text(encoding="utf-8"))
        validate_against_schema(spec_dict)

    def test_validate_rejects_empty_termination(self) -> None:
        """invalid_missing_termination.yaml MUST fail schema validation."""
        spec_dict = yaml.safe_load(
            (EXAMPLES_DIR / "invalid_missing_termination.yaml").read_text(encoding="utf-8")
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_against_schema(spec_dict)
        assert "stop_conditions" in str(exc_info.value).lower()

    def test_validate_rejects_missing_required_section(self) -> None:
        """A spec missing `actions` MUST fail."""
        bad_spec = {
            "scenario": {"scenario_id": "x", "scenario_version": "0"},
            "time": {"max_ticks": 10},
            "space": {"type": "discrete", "shape": [3, 3]},
            "entities": {"columns": [{"name": "e", "dtype": "float"}], "initial_count": 1},
            "termination": {"stop_conditions": [{"kind": "max_ticks"}]},
        }
        with pytest.raises(SchemaValidationError):
            validate_against_schema(bad_spec)

    def test_validate_rejects_unknown_space_type(self) -> None:
        """Unknown space.type MUST fail."""
        bad_spec = {
            "scenario": {"scenario_id": "x", "scenario_version": "0"},
            "time": {"max_ticks": 10},
            "space": {"type": "hexagonal", "shape": [3, 3]},  # invalid
            "entities": {"columns": [{"name": "e", "dtype": "float"}], "initial_count": 1},
            "actions": {"actions": [{"action_type": "noop", "effects": []}]},
            "termination": {"stop_conditions": [{"kind": "max_ticks"}]},
        }
        with pytest.raises(SchemaValidationError):
            validate_against_schema(bad_spec)

    def test_validate_rejects_discrete_without_shape(self) -> None:
        """discrete space without shape MUST fail (conditional requirement)."""
        bad_spec = {
            "scenario": {"scenario_id": "x", "scenario_version": "0"},
            "time": {"max_ticks": 10},
            "space": {"type": "discrete"},  # missing shape
            "entities": {"columns": [{"name": "e", "dtype": "float"}], "initial_count": 1},
            "actions": {"actions": [{"action_type": "noop", "effects": []}]},
            "termination": {"stop_conditions": [{"kind": "max_ticks"}]},
        }
        with pytest.raises(SchemaValidationError):
            validate_against_schema(bad_spec)

    def test_validate_rejects_additional_properties(self) -> None:
        """Unknown top-level key MUST fail (additionalProperties: false)."""
        bad_spec = {
            "scenario": {"scenario_id": "x", "scenario_version": "0"},
            "time": {"max_ticks": 10},
            "space": {"type": "discrete", "shape": [3, 3]},
            "entities": {"columns": [{"name": "e", "dtype": "float"}], "initial_count": 1},
            "actions": {"actions": [{"action_type": "noop", "effects": []}]},
            "termination": {"stop_conditions": [{"kind": "max_ticks"}]},
            "unknown_section": {},  # additional property
        }
        with pytest.raises(SchemaValidationError):
            validate_against_schema(bad_spec)


# ---------------------------------------------------------------------------
# Round-trip: YAML → dict → ScenarioSpec → dict → schema validation
# ---------------------------------------------------------------------------


class TestSpecRoundTrip:
    """End-to-end round-trip tests: YAML → ScenarioSpec → validation."""

    @pytest.mark.parametrize("example_name", [
        "discrete_grid",
        "continuous_field",
        "graph_registry",
    ])
    def test_yaml_to_spec_to_dict_passes_schema(self, example_name: str) -> None:
        """YAML → ScenarioSpec.from_dict → to_dict → validate_against_schema."""
        yaml_path = EXAMPLES_DIR / f"{example_name}.yaml"
        spec_dict = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        spec = ScenarioSpec.from_dict(spec_dict)
        round_trip_dict = spec.to_dict()
        validate_against_schema(round_trip_dict)  # should not raise

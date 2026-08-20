"""Tests for the static semantic validator (S-03).

Covers all 11 checks from main plan §13.5 with both pass and fail
scenarios, plus the top-level :func:`validate_semantics` aggregator
and the three example YAMLs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worldloop_scenarios.spec import (
    ActionsSpec,
    EntitiesSpec,
    ExogenousSpec,
    RegistriesSpec,
    ScenarioMeta,
    ScenarioSpec,
    SpaceSpec,
    TerminationSpec,
    TimeSpec,
)
from worldloop_scenarios.validator import (
    ValidationIssue,
    ValidationResult,
    check_action_field_references_exist,
    check_boundary_conditions_explicit,
    check_candidate_action_no_direct_write,
    check_effect_targets_are_explicit,
    check_graph_edge_cleanup_on_delete,
    check_L5_no_writeback,
    check_no_guaranteed_infinite_growth,
    check_registry_no_dangling_reference,
    check_resource_signs_consistent,
    check_reward_does_not_read_future,
    check_stochastic_rules_have_seed,
    validate_semantics,
)


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# Fixtures — minimal valid spec builders
# ---------------------------------------------------------------------------


def _base_spec(
    *,
    actions: tuple = (),
    entities_columns: tuple = ({"name": "energy", "dtype": "float"},),
    space_type: str = "discrete",
    space_shape: tuple = (5, 5),
    space_node_ids: tuple = (),
    registries_enabled: bool = False,
    registries_types: tuple = (),
    max_ticks: int = 100,
    deterministic: bool = True,
    exogenous_seed: int | None = None,
    stop_conditions: tuple = ({"kind": "max_ticks"},),
) -> ScenarioSpec:
    """Build a minimal valid spec, parameterized for the tests below."""
    return ScenarioSpec(
        scenario=ScenarioMeta(scenario_id="test", scenario_version="0"),
        time=TimeSpec(max_ticks=max_ticks, deterministic=deterministic),
        space=SpaceSpec(
            type=space_type,
            shape=space_shape if space_type == "discrete" else (),
            node_ids=space_node_ids,
        ),
        entities=EntitiesSpec(columns=entities_columns, initial_count=1),
        actions=ActionsSpec(actions=actions),
        termination=TerminationSpec(stop_conditions=stop_conditions),
        registries=RegistriesSpec(
            registry_types=registries_types,
            enabled=registries_enabled,
        ),
        exogenous=ExogenousSpec(seed=exogenous_seed),
    )


_EFFECT_NOOP = {"target": "entity", "field": "energy", "op": "add", "value": 0.0}


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------


class TestValidateSemanticsAggregator:
    """Tests for the top-level validate_semantics function."""

    def test_valid_spec_returns_no_errors(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "forage",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "add", "value": 5.0},
                    ),
                },
            )
        )
        result = validate_semantics(spec)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_spec_returns_errors(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "forage",
                    "effects": (
                        {"target": "entity", "field": "nonexistent_col", "op": "add", "value": 1.0},
                    ),
                },
            )
        )
        result = validate_semantics(spec)
        assert not result.is_valid
        assert len(result.errors) >= 1

    def test_warnings_do_not_make_spec_invalid(self) -> None:
        """Warnings are informational; only errors invalidate."""
        spec = _base_spec(
            actions=(
                {
                    "action_type": "drain",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "sub", "value": 1.0},
                    ),
                },
            )
        )
        result = validate_semantics(spec)
        # resource_signs_consistent emits a warning for pure-drain fields
        assert len(result.warnings) >= 1
        assert result.is_valid

    def test_example_yamls_pass_validation(self) -> None:
        """All three example YAMLs MUST pass semantic validation."""
        for name in ("discrete_grid", "continuous_field", "graph_registry"):
            spec_dict = yaml.safe_load((EXAMPLES_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
            from worldloop_scenarios.spec import ScenarioSpec as SS

            spec = SS.from_dict(spec_dict)
            result = validate_semantics(spec)
            assert result.is_valid, f"{name} failed: {[i.message for i in result.errors]}"


# ---------------------------------------------------------------------------
# Check 1: action_field_references_exist
# ---------------------------------------------------------------------------


class TestCheck1ActionFieldReferencesExist:
    def test_valid_column_reference_passes(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "forage",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "add", "value": 1.0},
                    ),
                },
            )
        )
        issues = check_action_field_references_exist(spec)
        assert issues == []

    def test_undeclared_column_fails(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "forage",
                    "effects": (
                        {"target": "entity", "field": "stamina", "op": "add", "value": 1.0},
                    ),
                },
            )
        )
        issues = check_action_field_references_exist(spec)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].check_id == "1"
        assert "stamina" in issues[0].message

    def test_non_entity_target_not_checked(self) -> None:
        """Field references on non-entity targets are not checked here."""
        spec = _base_spec(
            actions=(
                {
                    "action_type": "drain",
                    "effects": (
                        {"target": "field", "field": "resource_density", "op": "sub", "value": 0.1},
                    ),
                },
            )
        )
        issues = check_action_field_references_exist(spec)
        assert issues == []


# ---------------------------------------------------------------------------
# Check 2: effect_targets_are_explicit
# ---------------------------------------------------------------------------


class TestCheck2EffectTargetsExplicit:
    def test_valid_effects_pass(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "forage",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "add", "value": 1.0},
                    ),
                },
            )
        )
        issues = check_effect_targets_are_explicit(spec)
        assert issues == []

    def test_invalid_target_fails(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "bad",
                    "effects": (
                        {"target": "universe", "field": "x", "op": "set", "value": 1},
                    ),
                },
            )
        )
        issues = check_effect_targets_are_explicit(spec)
        assert any(i.severity == "error" and i.check_id == "2" for i in issues)

    def test_missing_op_fails(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "bad",
                    "effects": (
                        {"target": "entity", "field": "energy", "value": 1.0},
                    ),
                },
            )
        )
        issues = check_effect_targets_are_explicit(spec)
        assert any("op" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Check 3: resource_signs_consistent
# ---------------------------------------------------------------------------


class TestCheck3ResourceSignsConsistent:
    def test_balanced_field_no_warning(self) -> None:
        """A field with both add and sub does not warn."""
        spec = _base_spec(
            actions=(
                {
                    "action_type": "forage",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "add", "value": 1.0},
                    ),
                },
                {
                    "action_type": "rest",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "sub", "value": 1.0},
                    ),
                },
            )
        )
        issues = check_resource_signs_consistent(spec)
        # energy has both add and sub — no warning
        assert not any("energy" in i.message for i in issues)

    def test_pure_drain_field_warns(self) -> None:
        """A field with only sub warns about pure drain."""
        spec = _base_spec(
            actions=(
                {
                    "action_type": "drain",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "sub", "value": 1.0},
                    ),
                },
            )
        )
        issues = check_resource_signs_consistent(spec)
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert "energy" in issues[0].message


# ---------------------------------------------------------------------------
# Check 4: no_guaranteed_infinite_growth
# ---------------------------------------------------------------------------


class TestCheck4NoGuaranteedInfiniteGrowth:
    def test_bounded_max_ticks_passes(self) -> None:
        spec = _base_spec(max_ticks=100)
        issues = check_no_guaranteed_infinite_growth(spec)
        assert issues == []

    def test_zero_max_ticks_fails(self) -> None:
        spec = _base_spec(max_ticks=0)
        issues = check_no_guaranteed_infinite_growth(spec)
        assert any(i.severity == "error" and i.check_id == "4" for i in issues)

    def test_negative_max_ticks_fails(self) -> None:
        spec = _base_spec(max_ticks=-1)
        issues = check_no_guaranteed_infinite_growth(spec)
        assert any(i.severity == "error" for i in issues)

    def test_huge_max_ticks_without_stop_conditions_warns(self) -> None:
        spec = _base_spec(
            max_ticks=100000,
            stop_conditions=(),  # no stop conditions
        )
        issues = check_no_guaranteed_infinite_growth(spec)
        assert any(i.severity == "warning" for i in issues)


# ---------------------------------------------------------------------------
# Check 5: boundary_conditions_explicit
# ---------------------------------------------------------------------------


class TestCheck5BoundaryConditionsExplicit:
    def test_valid_max_ticks_condition_passes(self) -> None:
        spec = _base_spec(stop_conditions=({"kind": "max_ticks"},))
        issues = check_boundary_conditions_explicit(spec)
        assert issues == []

    def test_threshold_missing_field_fails(self) -> None:
        spec = _base_spec(
            stop_conditions=({"kind": "threshold", "op": "lt", "value": 0.0},),
        )
        issues = check_boundary_conditions_explicit(spec)
        assert any(i.severity == "error" and "field" in i.message for i in issues)

    def test_threshold_missing_op_fails(self) -> None:
        spec = _base_spec(
            stop_conditions=({"kind": "threshold", "field": "energy", "value": 0.0},),
        )
        issues = check_boundary_conditions_explicit(spec)
        assert any("op" in i.message for i in issues)

    def test_threshold_missing_value_fails(self) -> None:
        spec = _base_spec(
            stop_conditions=({"kind": "threshold", "field": "energy", "op": "lt"},),
        )
        issues = check_boundary_conditions_explicit(spec)
        assert any("value" in i.message for i in issues)

    def test_custom_without_predicate_warns(self) -> None:
        spec = _base_spec(
            stop_conditions=({"kind": "custom"},),
        )
        issues = check_boundary_conditions_explicit(spec)
        assert any(i.severity == "warning" and "predicate" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Check 6: stochastic_rules_have_seed
# ---------------------------------------------------------------------------


class TestCheck6StochasticRulesHaveSeed:
    def test_deterministic_no_seed_passes(self) -> None:
        spec = _base_spec(deterministic=True, exogenous_seed=None)
        issues = check_stochastic_rules_have_seed(spec)
        assert issues == []

    def test_stochastic_with_seed_passes(self) -> None:
        spec = _base_spec(deterministic=False, exogenous_seed=42)
        issues = check_stochastic_rules_have_seed(spec)
        assert issues == []

    def test_stochastic_without_seed_warns(self) -> None:
        spec = _base_spec(deterministic=False, exogenous_seed=None)
        issues = check_stochastic_rules_have_seed(spec)
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert "seed" in issues[0].message


# ---------------------------------------------------------------------------
# Check 7: reward_does_not_read_future
# ---------------------------------------------------------------------------


class TestCheck7RewardDoesNotReadFuture:
    def test_reward_from_current_state_passes(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "score",
                    "effects": (
                        {"target": "entity", "field": "reward", "op": "set", "value": "$energy"},
                    ),
                },
            )
        )
        issues = check_reward_does_not_read_future(spec)
        assert issues == []

    def test_reward_reading_future_fails(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "score",
                    "effects": (
                        {"target": "entity", "field": "reward", "op": "set", "value": "$future.energy"},
                    ),
                },
            )
        )
        issues = check_reward_does_not_read_future(spec)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "$future" in issues[0].message or "future" in issues[0].message

    def test_reward_reading_next_tick_fails(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "score",
                    "effects": (
                        {"target": "entity", "field": "reward", "op": "set", "value": "$t+1.energy"},
                    ),
                },
            )
        )
        issues = check_reward_does_not_read_future(spec)
        assert any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# Check 8: candidate_action_no_direct_write
# ---------------------------------------------------------------------------


class TestCheck8CandidateActionNoDirectWrite:
    def test_writing_entity_field_passes(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "forage",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "add", "value": 1.0},
                    ),
                },
            )
        )
        issues = check_candidate_action_no_direct_write(spec)
        assert issues == []

    def test_writing_meta_fails(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "hack",
                    "effects": (
                        {"target": "meta", "field": "tick", "op": "set", "value": 999},
                    ),
                },
            )
        )
        issues = check_candidate_action_no_direct_write(spec)
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_writing_capabilities_field_fails(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "hack",
                    "effects": (
                        {"target": "entity", "field": "capabilities.fields", "op": "set", "value": True},
                    ),
                },
            )
        )
        issues = check_candidate_action_no_direct_write(spec)
        assert any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# Check 9: L5_no_writeback
# ---------------------------------------------------------------------------


class TestCheck9L5NoWriteback:
    def test_non_emergence_target_passes(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "forage",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "add", "value": 1.0},
                    ),
                },
            )
        )
        issues = check_L5_no_writeback(spec)
        assert issues == []

    def test_writing_emergence_target_fails(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "hack",
                    "effects": (
                        {"target": "emergence", "field": "complexity", "op": "set", "value": 0.99},
                    ),
                },
            )
        )
        issues = check_L5_no_writeback(spec)
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_writing_emergent_field_fails(self) -> None:
        spec = _base_spec(
            actions=(
                {
                    "action_type": "hack",
                    "effects": (
                        {"target": "entity", "field": "emergent", "op": "set", "value": True},
                    ),
                },
            )
        )
        issues = check_L5_no_writeback(spec)
        assert any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# Check 10: registry_no_dangling_reference
# ---------------------------------------------------------------------------


class TestCheck10RegistryNoDanglingReference:
    def test_disabled_registries_skipped(self) -> None:
        spec = _base_spec(
            registries_enabled=False,
            actions=(
                {
                    "action_type": "pick",
                    "effects": (),
                    "preconditions": (
                        {"kind": "registry_unowned", "registry_type": "object", "entry_id": "x"},
                    ),
                },
            ),
        )
        issues = check_registry_no_dangling_reference(spec)
        assert issues == []

    def test_declared_registry_type_passes(self) -> None:
        spec = _base_spec(
            registries_enabled=True,
            registries_types=("object", "tool"),
            actions=(
                {
                    "action_type": "pick",
                    "effects": (),
                    "preconditions": (
                        {"kind": "registry_unowned", "registry_type": "object", "entry_id": "x"},
                    ),
                },
            ),
        )
        issues = check_registry_no_dangling_reference(spec)
        assert issues == []

    def test_undeclared_registry_type_fails(self) -> None:
        spec = _base_spec(
            registries_enabled=True,
            registries_types=("object",),
            actions=(
                {
                    "action_type": "pick",
                    "effects": (),
                    "preconditions": (
                        {"kind": "registry_unowned", "registry_type": "artifact", "entry_id": "x"},
                    ),
                },
            ),
        )
        issues = check_registry_no_dangling_reference(spec)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "artifact" in issues[0].message

    def test_variable_registry_type_skipped(self) -> None:
        """Variable references (starting with $) are not checked."""
        spec = _base_spec(
            registries_enabled=True,
            registries_types=("object",),
            actions=(
                {
                    "action_type": "pick",
                    "effects": (),
                    "preconditions": (
                        {"kind": "registry_unowned", "registry_type": "$registry_type", "entry_id": "$entry_id"},
                    ),
                },
            ),
        )
        issues = check_registry_no_dangling_reference(spec)
        assert issues == []


# ---------------------------------------------------------------------------
# Check 11: graph_edge_cleanup_on_delete
# ---------------------------------------------------------------------------


class TestCheck11GraphEdgeCleanupOnDelete:
    def test_non_graph_space_skipped(self) -> None:
        spec = _base_spec(
            space_type="discrete",
            actions=(
                {
                    "action_type": "kill",
                    "effects": (
                        {"target": "entity", "op": "remove"},
                    ),
                },
            ),
        )
        issues = check_graph_edge_cleanup_on_delete(spec)
        assert issues == []

    def test_graph_with_edge_cleanup_passes(self) -> None:
        spec = _base_spec(
            space_type="graph",
            space_node_ids=("n0", "n1"),
            actions=(
                {
                    "action_type": "kill",
                    "effects": (
                        {"target": "entity", "op": "remove"},
                        {"target": "relation", "op": "remove", "field": "edges"},
                    ),
                },
            ),
        )
        issues = check_graph_edge_cleanup_on_delete(spec)
        assert issues == []

    def test_graph_without_edge_cleanup_warns(self) -> None:
        spec = _base_spec(
            space_type="graph",
            space_node_ids=("n0", "n1"),
            actions=(
                {
                    "action_type": "kill",
                    "effects": (
                        {"target": "entity", "op": "remove"},
                    ),
                },
            ),
        )
        issues = check_graph_edge_cleanup_on_delete(spec)
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_graph_without_entity_removal_passes(self) -> None:
        spec = _base_spec(
            space_type="graph",
            space_node_ids=("n0", "n1"),
            actions=(
                {
                    "action_type": "forage",
                    "effects": (
                        {"target": "entity", "field": "energy", "op": "add", "value": 1.0},
                    ),
                },
            ),
        )
        issues = check_graph_edge_cleanup_on_delete(spec)
        assert issues == []


# ---------------------------------------------------------------------------
# ValidationResult / ValidationIssue dataclasses
# ---------------------------------------------------------------------------


class TestValidationResultDataclass:
    def test_is_valid_true_when_no_errors(self) -> None:
        result = ValidationResult()
        result.issues.append(
            ValidationIssue(check_id="x", check_name="x", severity="warning", message="m")
        )
        assert result.is_valid

    def test_is_valid_false_when_error_present(self) -> None:
        result = ValidationResult()
        result.issues.append(
            ValidationIssue(check_id="x", check_name="x", severity="error", message="m")
        )
        assert not result.is_valid

    def test_errors_filters_only_errors(self) -> None:
        result = ValidationResult()
        result.issues.extend([
            ValidationIssue(check_id="1", check_name="a", severity="error", message="e1"),
            ValidationIssue(check_id="2", check_name="b", severity="warning", message="w1"),
            ValidationIssue(check_id="3", check_name="c", severity="error", message="e2"),
        ])
        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_validation_issue_is_frozen(self) -> None:
        issue = ValidationIssue(check_id="1", check_name="x", severity="error", message="m")
        with pytest.raises((AttributeError, TypeError)):
            issue.severity = "warning"  # type: ignore[misc]

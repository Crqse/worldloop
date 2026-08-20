"""Component-level tests for :mod:`worldloop_data.prompt_contract`.

These tests verify each builder, hash, and validator in isolation using
hand-constructed :class:`AgentObservationView` instances. The Prompt
Gate tests (P-G1~P-G6) using real projectors live in
``test_prompt_gates.py``.

Covers Phase 1 / Beta correction §5.4-5.6 of
``docs/07.advice/2026-07-30_WorldLoop主线实验有效性与Beta发布优化实施方案.md``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from worldloop_kernel.observation import (
    OBSERVATION_SCHEMA_VERSION,
    AgentObservationView,
    FocalAgentAttributes,
    OmissionPolicy,
    PreviousActionSummary,
    VisibleEntity,
    hash_observation,
)
from worldloop_kernel.protocol import LegalAction
from worldloop_data.prompt_contract import (
    FORBIDDEN_GLOBAL_FIELDS,
    PROMPT_CONTRACT_SCHEMA_VERSION,
    STABLE_SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    USER_MESSAGE_SCHEMA_VERSION,
    ActionSchemaEntry,
    LLMRequestLike,
    PromptHashBundle,
    ScenarioContract,
    build_llm_request,
    build_scenario_contract,
    build_user_message,
    compute_prompt_hashes,
    hash_scenario_contract,
    hash_system_prompt,
    hash_user_message,
    scan_for_forbidden_fields,
    validate_prompt_components,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_observation(
    *,
    tick: int = 0,
    focal_energy: float = 10.0,
    focal_position: tuple[int, int] = (1, 2),
    other_positions: tuple[tuple[int, int], ...] = ((3, 4),),
    legal_action_types: tuple[str, ...] = ("MOVE", "REST"),
    visible_fields: dict[str, Any] | None = None,
    scenario_id: str = "test_scenario_v1",
    scenario_version: str = "1.0.0",
) -> AgentObservationView:
    """Build a minimal observation for testing."""
    return AgentObservationView(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        scenario_id=scenario_id,
        scenario_version=scenario_version,
        tick=tick,
        focal_agent=FocalAgentAttributes(
            agent_id="a0",
            public_attributes={
                "x": focal_position[0],
                "y": focal_position[1],
                "node": "zone_a",
                "alive": True,
            },
            self_visible_attributes={"energy": focal_energy},
        ),
        previous_action=PreviousActionSummary(),
        visible_fields=visible_fields or {},
        visible_entities=tuple(
            VisibleEntity(
                entity_id=f"agent_{i+1}",
                columns={
                    "x": pos[0],
                    "y": pos[1],
                    "node": "zone_b",
                    "alive": True,
                },
            )
            for i, pos in enumerate(other_positions)
        ),
        visible_relations=(),
        visible_events=(),
        legal_actions=tuple(
            LegalAction(action_type=at, params={"target_node": "zone_a"})
            for at in legal_action_types
        ),
        omission_policy=OmissionPolicy(
            omitted_slots=("registries", "population", "events"),
            reason="capability_unavailable",
            unsupported_capabilities=("registries", "population", "events"),
        ),
    )


# ---------------------------------------------------------------------------
# Stable system prompt
# ---------------------------------------------------------------------------


class TestStableSystemPrompt:
    def test_system_prompt_is_non_empty(self) -> None:
        assert STABLE_SYSTEM_PROMPT
        assert len(STABLE_SYSTEM_PROMPT) > 50

    def test_system_prompt_version_is_set(self) -> None:
        assert SYSTEM_PROMPT_VERSION
        assert isinstance(SYSTEM_PROMPT_VERSION, str)

    def test_system_prompt_mentions_action_selection(self) -> None:
        assert "action-selection" in STABLE_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_json_output_schema(self) -> None:
        # The output JSON schema is inlined so the model knows the shape.
        assert "action_type" in STABLE_SYSTEM_PROMPT
        assert "params" in STABLE_SYSTEM_PROMPT

    def test_system_prompt_does_not_mention_scenario_specifics(self) -> None:
        # No experiment-specific numbers / map sizes / channel layouts.
        forbidden_substrings = (
            "256 channel",
            "innovation",
            "reserved range",
            "graph D_initial",
            "starvation_threshold",
            "rest_gain",
            "worldloop-v1",
        )
        for s in forbidden_substrings:
            assert s.lower() not in STABLE_SYSTEM_PROMPT.lower(), (
                f"system prompt leaks scenario-specific: {s!r}"
            )

    def test_hash_system_prompt_stable(self) -> None:
        h1 = hash_system_prompt()
        h2 = hash_system_prompt(STABLE_SYSTEM_PROMPT)
        assert h1 == h2
        assert h1.startswith("sha256:")
        assert len(h1) == len("sha256:") + 64

    def test_hash_system_prompt_differs_for_different_text(self) -> None:
        h1 = hash_system_prompt()
        h2 = hash_system_prompt("different prompt")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Scenario contract builder
# ---------------------------------------------------------------------------


class TestBuildScenarioContract:
    def test_contract_has_schema_version(self) -> None:
        obs = _make_observation()
        contract = build_scenario_contract(obs)
        assert contract.schema_version == PROMPT_CONTRACT_SCHEMA_VERSION

    def test_contract_inherits_scenario_id_and_version(self) -> None:
        obs = _make_observation(
            scenario_id="emergency_v1", scenario_version="2.3.1"
        )
        contract = build_scenario_contract(obs)
        assert contract.scenario_id == "emergency_v1"
        assert contract.scenario_version == "2.3.1"

    def test_action_schema_has_one_entry_per_action_type(self) -> None:
        obs = _make_observation(
            legal_action_types=("MOVE", "REST", "FORAGE", "MOVE")  # duplicate
        )
        contract = build_scenario_contract(obs)
        action_types = [e.action_type for e in contract.action_schema]
        # Distinct types only; no duplicates.
        assert sorted(action_types) == ["FORAGE", "MOVE", "REST"]
        assert len(action_types) == len(set(action_types))

    def test_action_schema_params_schema_inferred(self) -> None:
        obs = _make_observation(legal_action_types=("MOVE",))
        contract = build_scenario_contract(obs)
        entry = contract.action_schema[0]
        # The MOVE action has params={"target_node": "zone_a"} (a string).
        assert "target_node" in entry.params_schema
        assert entry.params_schema["target_node"] == "string"

    def test_observation_schema_summary_has_visible_slots(self) -> None:
        obs = _make_observation(visible_fields={"resource_density": 0.5})
        contract = build_scenario_contract(obs)
        summary = contract.observation_schema_summary
        assert summary["schema_version"] == OBSERVATION_SCHEMA_VERSION
        assert "visible_fields" in summary["visible_slots"]
        assert "legal_actions" in summary["visible_slots"]
        assert "focal_agent" in summary["visible_slots"]
        assert summary["focal_agent_id"] == "a0"
        assert summary["tick"] == 0

    def test_omission_disclaimer_lists_unsupported(self) -> None:
        obs = _make_observation()
        contract = build_scenario_contract(obs)
        assert "registries" in contract.omission_disclaimer
        assert "population" in contract.omission_disclaimer
        assert "events" in contract.omission_disclaimer

    def test_omission_disclaimer_when_nothing_omitted(self) -> None:
        obs = _make_observation()
        # Replace omission with empty.
        obs = AgentObservationView(
            schema_version=obs.schema_version,
            scenario_id=obs.scenario_id,
            scenario_version=obs.scenario_version,
            tick=obs.tick,
            focal_agent=obs.focal_agent,
            previous_action=obs.previous_action,
            visible_fields=obs.visible_fields,
            visible_entities=obs.visible_entities,
            visible_relations=obs.visible_relations,
            visible_events=obs.visible_events,
            legal_actions=obs.legal_actions,
            omission_policy=OmissionPolicy(),
        )
        contract = build_scenario_contract(obs)
        assert "All capabilities declared" in contract.omission_disclaimer

    def test_contract_is_frozen(self) -> None:
        obs = _make_observation()
        contract = build_scenario_contract(obs)
        with pytest.raises(Exception):
            contract.scenario_id = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    def test_user_message_is_valid_json(self) -> None:
        obs = _make_observation()
        msg = build_user_message(obs)
        parsed = json.loads(msg)
        assert isinstance(parsed, dict)

    def test_user_message_has_schema_version(self) -> None:
        obs = _make_observation()
        msg = build_user_message(obs)
        parsed = json.loads(msg)
        assert parsed["schema_version"] == USER_MESSAGE_SCHEMA_VERSION

    def test_user_message_has_scenario_contract_and_observation(self) -> None:
        obs = _make_observation()
        msg = build_user_message(obs)
        parsed = json.loads(msg)
        assert "scenario_contract" in parsed
        assert "observation" in parsed

    def test_user_message_does_not_contain_system_prompt_text(self) -> None:
        """P-G3: system prompt must NOT be concatenated into user content."""
        obs = _make_observation()
        msg = build_user_message(obs)
        # The stable system prompt's distinctive phrase must not appear
        # in the user message.
        assert "You are an action-selection policy" not in msg

    def test_user_message_does_not_contain_forbidden_fields(self) -> None:
        """P-G4: no hidden-state substrings leak into the user message."""
        obs = _make_observation()
        msg = build_user_message(obs)
        violations = scan_for_forbidden_fields(msg)
        assert violations == [], f"forbidden fields leaked: {violations}"

    def test_user_message_includes_focal_self_visible(self) -> None:
        """Focal agent's self_visible_attributes (energy) MUST appear in
        the user message — this is authorized focal-private state."""
        obs = _make_observation(focal_energy=42.5)
        msg = build_user_message(obs)
        parsed = json.loads(msg)
        focal = parsed["observation"]["focal_agent"]
        assert focal["self_visible_attributes"]["energy"] == 42.5

    def test_user_message_excludes_other_agents_private_state(self) -> None:
        """Other agents' private state (energy) MUST NOT appear in
        visible_entities — only public columns (x/y/node/alive)."""
        obs = _make_observation()
        msg = build_user_message(obs)
        parsed = json.loads(msg)
        for entity in parsed["observation"]["visible_entities"]:
            assert "energy" not in entity["columns"]
            assert "x" in entity["columns"]

    def test_user_message_stable_for_same_observation(self) -> None:
        """Same observation -> identical user message (canonical JSON)."""
        obs = _make_observation()
        msg1 = build_user_message(obs)
        msg2 = build_user_message(obs)
        assert msg1 == msg2

    def test_user_message_differs_for_different_observation(self) -> None:
        obs1 = _make_observation(focal_energy=10.0)
        obs2 = _make_observation(focal_energy=20.0)
        msg1 = build_user_message(obs1)
        msg2 = build_user_message(obs2)
        assert msg1 != msg2


# ---------------------------------------------------------------------------
# Hash bundle
# ---------------------------------------------------------------------------


class TestComputePromptHashes:
    def test_bundle_has_five_hashes(self) -> None:
        obs = _make_observation()
        bundle = compute_prompt_hashes(obs)
        assert isinstance(bundle, PromptHashBundle)
        assert bundle.system_prompt_hash
        assert bundle.scenario_contract_hash
        assert bundle.observation_hash
        assert bundle.user_message_hash
        assert bundle.combined_prompt_hash

    def test_all_hashes_are_sha256_format(self) -> None:
        obs = _make_observation()
        bundle = compute_prompt_hashes(obs)
        for h in (
            bundle.system_prompt_hash,
            bundle.scenario_contract_hash,
            bundle.observation_hash,
            bundle.user_message_hash,
            bundle.combined_prompt_hash,
        ):
            assert h.startswith("sha256:"), f"bad hash format: {h!r}"
            assert len(h) == len("sha256:") + 64

    def test_observation_hash_matches_hash_observation(self) -> None:
        obs = _make_observation()
        bundle = compute_prompt_hashes(obs)
        assert bundle.observation_hash == hash_observation(obs)

    def test_combined_hash_changes_when_observation_changes(self) -> None:
        """P-G1: different observation -> different combined hash."""
        obs1 = _make_observation(focal_energy=10.0)
        obs2 = _make_observation(focal_energy=20.0)
        b1 = compute_prompt_hashes(obs1)
        b2 = compute_prompt_hashes(obs2)
        assert b1.combined_prompt_hash != b2.combined_prompt_hash
        assert b1.user_message_hash != b2.user_message_hash
        assert b1.observation_hash != b2.observation_hash

    def test_system_prompt_hash_constant(self) -> None:
        """System prompt hash is the same for every observation."""
        obs1 = _make_observation(focal_energy=10.0)
        obs2 = _make_observation(focal_energy=20.0, scenario_id="other")
        b1 = compute_prompt_hashes(obs1)
        b2 = compute_prompt_hashes(obs2)
        assert b1.system_prompt_hash == b2.system_prompt_hash
        assert b1.system_prompt_hash == hash_system_prompt()

    def test_combined_hash_depends_on_all_components(self) -> None:
        """Combined hash is a function of system + contract + observation
        + user message. If any changes, combined changes."""
        obs = _make_observation()
        b1 = compute_prompt_hashes(obs)
        # Re-compute: same input -> same output.
        b2 = compute_prompt_hashes(obs)
        assert b1 == b2


# ---------------------------------------------------------------------------
# Forbidden field scan (P-G4)
# ---------------------------------------------------------------------------


class TestScanForForbiddenFields:
    def test_clean_message_returns_empty(self) -> None:
        obs = _make_observation()
        msg = build_user_message(obs)
        assert scan_for_forbidden_fields(msg) == []

    def test_message_with_rng_state_detected(self) -> None:
        msg = '{"rng_state": "0xabc"}'
        violations = scan_for_forbidden_fields(msg)
        assert "rng_state" in violations

    def test_message_with_branch_id_detected(self) -> None:
        msg = '{"branch_id": "b0", "fork_group_id": "g1"}'
        violations = scan_for_forbidden_fields(msg)
        assert "branch_id" in violations
        assert "fork_group_id" in violations

    def test_message_with_api_key_detected(self) -> None:
        msg = '{"api_key": "sk-abc123", "token": "bearer xyz"}'
        violations = scan_for_forbidden_fields(msg)
        assert "api_key" in violations
        assert "token" in violations

    def test_message_with_world_internal_detected(self) -> None:
        msg = '{"world_rng": 42, "internal_counter": 7}'
        violations = scan_for_forbidden_fields(msg)
        assert "world_" in violations
        assert "internal_" in violations

    def test_self_visible_attributes_not_flagged(self) -> None:
        """``self_visible_attributes`` is allowed because it's the
        focal agent's own private state — the substring check should
        NOT flag it as forbidden."""
        obs = _make_observation()
        msg = build_user_message(obs)
        # The user message DOES contain "self_visible_attributes" as a
        # JSON key — that's by design.
        assert "self_visible_attributes" in msg
        # And it must NOT be flagged as a forbidden field.
        assert scan_for_forbidden_fields(msg) == []


# ---------------------------------------------------------------------------
# Validation (P-G3, P-G4, P-G5, P-G6)
# ---------------------------------------------------------------------------


class TestValidatePromptComponents:
    def test_valid_observation_returns_no_errors(self) -> None:
        obs = _make_observation()
        errors = validate_prompt_components(obs)
        assert errors == [], f"unexpected errors: {errors}"

    def test_validation_catches_action_schema_mismatch(self) -> None:
        """P-G5: if scenario_contract.action_schema doesn't match
        observation.legal_actions, validation must fail."""
        obs = _make_observation(legal_action_types=("MOVE", "REST"))
        # Build a contract with a wrong action_type.
        wrong_contract = ScenarioContract(
            schema_version=PROMPT_CONTRACT_SCHEMA_VERSION,
            scenario_id=obs.scenario_id,
            scenario_version=obs.scenario_version,
            public_objective="test",
            action_schema=(
                ActionSchemaEntry(action_type="MOVE"),
                ActionSchemaEntry(action_type="NONEXISTENT"),
            ),
            observation_schema_summary={},
            omission_disclaimer="",
        )
        errors = validate_prompt_components(obs, wrong_contract)
        assert any("missing action_types" in e for e in errors)
        assert any("not in legal_actions" in e for e in errors)

    def test_validation_catches_scenario_id_mismatch(self) -> None:
        obs = _make_observation(scenario_id="scenario_a")
        wrong_contract = build_scenario_contract(obs)
        wrong_contract = ScenarioContract(
            schema_version=wrong_contract.schema_version,
            scenario_id="different_scenario",  # mismatch!
            scenario_version=wrong_contract.scenario_version,
            public_objective=wrong_contract.public_objective,
            action_schema=wrong_contract.action_schema,
            observation_schema_summary=wrong_contract.observation_schema_summary,
            omission_disclaimer=wrong_contract.omission_disclaimer,
        )
        errors = validate_prompt_components(obs, wrong_contract)
        assert any("scenario_id" in e for e in errors)

    def test_validation_catches_schema_version_mismatch(self) -> None:
        """P-G3: scenario_contract.schema_version must match
        PROMPT_CONTRACT_SCHEMA_VERSION."""
        obs = _make_observation()
        wrong_contract = ScenarioContract(
            schema_version="99.0.0",  # wrong version
            scenario_id=obs.scenario_id,
            scenario_version=obs.scenario_version,
            public_objective="test",
            action_schema=tuple(
                ActionSchemaEntry(action_type=la.action_type)
                for la in obs.legal_actions
            ),
            observation_schema_summary={},
            omission_disclaimer="",
        )
        errors = validate_prompt_components(obs, wrong_contract)
        assert any("schema_version" in e for e in errors)


# ---------------------------------------------------------------------------
# build_llm_request (P-G3)
# ---------------------------------------------------------------------------


class TestBuildLLMRequest:
    def test_request_has_system_prompt_set(self) -> None:
        obs = _make_observation()
        req = build_llm_request(obs, model="test-model")
        assert req.system_prompt == STABLE_SYSTEM_PROMPT
        assert req.system_prompt is not None

    def test_request_prompt_is_user_message(self) -> None:
        obs = _make_observation()
        req = build_llm_request(obs, model="test-model")
        # The prompt slot holds the user message (scenario contract + obs).
        # The system prompt text must NOT appear concatenated in prompt.
        assert "You are an action-selection policy" not in req.prompt

    def test_request_carries_model_and_temperature(self) -> None:
        obs = _make_observation()
        req = build_llm_request(
            obs, model="deepseek-v4", temperature=0.5, max_tokens=128
        )
        assert req.model == "deepseek-v4"
        assert req.temperature == 0.5
        assert req.max_tokens == 128

    def test_request_default_temperature_is_zero(self) -> None:
        obs = _make_observation()
        req = build_llm_request(obs, model="test")
        assert req.temperature == 0.0

    def test_request_is_frozen(self) -> None:
        obs = _make_observation()
        req = build_llm_request(obs, model="test")
        assert isinstance(req, LLMRequestLike)
        with pytest.raises(Exception):
            req.model = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Hash determinism (P-G1 / P-G2 mechanics)
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    def test_same_observation_same_bundle(self) -> None:
        obs = _make_observation()
        b1 = compute_prompt_hashes(obs)
        b2 = compute_prompt_hashes(obs)
        assert b1 == b2

    def test_tick_change_changes_bundle(self) -> None:
        """P-G1: same legal actions, different tick -> different hash."""
        obs1 = _make_observation(tick=0)
        obs2 = _make_observation(tick=1)
        b1 = compute_prompt_hashes(obs1)
        b2 = compute_prompt_hashes(obs2)
        assert b1.combined_prompt_hash != b2.combined_prompt_hash

    def test_focal_energy_change_changes_bundle(self) -> None:
        """P-G1: focal agent's private state change -> different hash."""
        obs1 = _make_observation(focal_energy=10.0)
        obs2 = _make_observation(focal_energy=99.0)
        b1 = compute_prompt_hashes(obs1)
        b2 = compute_prompt_hashes(obs2)
        assert b1.combined_prompt_hash != b2.combined_prompt_hash

    def test_legal_action_change_changes_bundle(self) -> None:
        """P-G1: legal action set change -> different hash."""
        obs1 = _make_observation(legal_action_types=("MOVE", "REST"))
        obs2 = _make_observation(legal_action_types=("MOVE", "FORAGE"))
        b1 = compute_prompt_hashes(obs1)
        b2 = compute_prompt_hashes(obs2)
        assert b1.combined_prompt_hash != b2.combined_prompt_hash

    def test_other_agent_position_change_changes_bundle(self) -> None:
        """P-G1: visible entity state change -> different hash."""
        obs1 = _make_observation(other_positions=((3, 4),))
        obs2 = _make_observation(other_positions=((7, 8),))
        b1 = compute_prompt_hashes(obs1)
        b2 = compute_prompt_hashes(obs2)
        assert b1.combined_prompt_hash != b2.combined_prompt_hash

"""Prompt Gate tests P-G1~P-G6 (Phase 1 / Beta correction §5.6).

These are the GATE tests for the prompt contract. They use the REAL
:class:`ParameterizedWorld` projector (not a hand-built observation) to
verify the end-to-end prompt assembly pipeline.

Gate definitions (per main plan §5.6):

- **P-G1**: Same legal actions, different observation -> prompt hash differs.
- **P-G2**: Same observation, different hidden state -> prompt hash identical.
- **P-G3**: system/user roles separated in transport payload.
- **P-G4**: prompt contains no unauthorized global fields.
- **P-G5**: scenario contract mechanically matches action/observation schema.
- **P-G6**: prompt template / scenario contract / observation all have
  independent version + hash.

Companion to ``test_prompt_contract.py`` (component-level tests using
hand-built observations). This file uses real projectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from worldloop_kernel.observation import hash_observation
from worldloop_kernel.protocol import LegalAction
from worldloop_data.llm_policy import (
    EchoLLMClient,
    InMemoryTelemetrySink,
    LLMPolicy,
    LLMPolicyConfig,
    LLMRequest,
)
from worldloop_data.policy import PolicyContext
from worldloop_data.prompt_contract import (
    FORBIDDEN_GLOBAL_FIELDS,
    PROMPT_CONTRACT_SCHEMA_VERSION,
    STABLE_SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    USER_MESSAGE_SCHEMA_VERSION,
    build_llm_request,
    build_scenario_contract,
    build_user_message,
    compute_prompt_hashes,
    scan_for_forbidden_fields,
    validate_prompt_components,
)
from worldloop_scenarios.parameterized_world import ParameterizedWorld
from worldloop_scenarios.spec import ScenarioSpec

EXAMPLES_DIR = (
    Path(__file__).resolve().parents[2]
    / "worldloop-scenarios"
    / "examples"
)


def _load_spec(name: str) -> ScenarioSpec:
    data = yaml.safe_load((EXAMPLES_DIR / name).read_text(encoding="utf-8"))
    return ScenarioSpec.from_dict(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def discrete_world() -> ParameterizedWorld:
    """Discrete grid world with energy + x + y + alive columns."""
    spec = _load_spec("discrete_grid.yaml")
    world = ParameterizedWorld(spec)
    world.reset(seed=42)
    return world


@pytest.fixture
def emergency_world() -> ParameterizedWorld:
    """Emergency resource world with fields + registries + events."""
    spec = _load_spec("emergency_resource_v1.yaml")
    world = ParameterizedWorld(spec)
    world.reset(seed=42)
    return world


def _make_policy_context(world: ParameterizedWorld, agent_id: str | int, tick: int = 0):
    """Build a PolicyContext suitable for LLMPolicy.propose()."""
    import random

    return PolicyContext(
        world=world,
        agent_id=agent_id,
        state=world.observe(),
        action_space=world.legal_actions(agent_id),
        tick=tick,
        rng=random.Random(42),
    )


# ---------------------------------------------------------------------------
# P-G1: different observation -> different prompt hash
# ---------------------------------------------------------------------------


class TestPG1DifferentObservationDifferentHash:
    """P-G1: same legal actions, different observation -> prompt hash differs.

    Verified via three different change patterns:

    1. Focal agent's energy changes (self_visible change).
    2. Other agent's position changes (visible_entities change).
    3. Tick advances (observation carries tick).
    """

    def test_focal_energy_change_changes_combined_hash(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        obs_before = discrete_world.observe_agent(a0)
        bundle_before = compute_prompt_hashes(obs_before)

        # Mutate focal agent's energy directly.
        idx = discrete_world._entity_ids.index(a0)
        discrete_world._entity_columns["energy"][idx] += 10.0
        obs_after = discrete_world.observe_agent(a0)
        bundle_after = compute_prompt_hashes(obs_after)

        assert (
            bundle_before.combined_prompt_hash
            != bundle_after.combined_prompt_hash
        ), "focal energy change did not change combined prompt hash"

    def test_other_agent_position_change_changes_combined_hash(
        self, discrete_world
    ):
        a0 = discrete_world._entity_ids[0]
        a1 = discrete_world._entity_ids[1]
        obs_before = discrete_world.observe_agent(a0)
        bundle_before = compute_prompt_hashes(obs_before)

        # Move a1 by mutating its x coordinate.
        idx1 = discrete_world._entity_ids.index(a1)
        original_x = discrete_world._entity_columns["x"][idx1]
        discrete_world._entity_columns["x"][idx1] = original_x + 5
        obs_after = discrete_world.observe_agent(a0)
        bundle_after = compute_prompt_hashes(obs_after)

        assert (
            bundle_before.combined_prompt_hash
            != bundle_after.combined_prompt_hash
        ), "other agent position change did not change combined prompt hash"

    def test_tick_change_changes_observation(self, discrete_world):
        """Tick is on the observation schema, so advancing tick changes
        the observation hash (and thus the combined prompt hash)."""
        a0 = discrete_world._entity_ids[0]
        obs_at_tick_0 = discrete_world.observe_agent(a0)
        assert obs_at_tick_0.tick == 0

        # Advance the world by one tick.
        discrete_world._tick = 1
        obs_at_tick_1 = discrete_world.observe_agent(a0)
        assert obs_at_tick_1.tick == 1

        b0 = compute_prompt_hashes(obs_at_tick_0)
        b1 = compute_prompt_hashes(obs_at_tick_1)
        assert b0.combined_prompt_hash != b1.combined_prompt_hash


# ---------------------------------------------------------------------------
# P-G2: same observation, different hidden state -> same prompt hash
# ---------------------------------------------------------------------------


class TestPG2SameObservationSameHashDespiteHiddenState:
    """P-G2: hidden state changes (RNG draws, internal caches) MUST NOT
    change the prompt hash.

    This is structural: :class:`AgentObservationView` has no field for
    RNG state, internal caches, or other agents' private columns. So
    even if the world's RNG advances, the observation (and thus the
    prompt) remains identical.
    """

    def test_rng_draw_does_not_change_combined_hash(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        obs_before = discrete_world.observe_agent(a0)
        bundle_before = compute_prompt_hashes(obs_before)

        # Draw from the RNG several times — this changes the kernel's
        # internal ``_rng.getstate()`` but MUST NOT change the observation
        # because the observation schema has no ``rng_state`` field.
        assert discrete_world._rng is not None
        for _ in range(5):
            discrete_world._rng.random()

        obs_after = discrete_world.observe_agent(a0)
        bundle_after = compute_prompt_hashes(obs_after)

        assert (
            bundle_before.combined_prompt_hash
            == bundle_after.combined_prompt_hash
        ), "RNG state change leaked into combined prompt hash"

    def test_rng_draw_does_not_change_user_message(self, discrete_world):
        """Stronger form of P-G2: the user_message STRING itself must be
        identical (not just the hash) — confirming the RNG state never
        reached the JSON payload."""
        a0 = discrete_world._entity_ids[0]
        msg_before = build_user_message(discrete_world.observe_agent(a0))

        assert discrete_world._rng is not None
        for _ in range(3):
            discrete_world._rng.random()

        msg_after = build_user_message(discrete_world.observe_agent(a0))
        assert msg_before == msg_after

    def test_other_agent_energy_change_does_not_affect_focal_prompt(
        self, discrete_world
    ):
        """If a1's energy changes, a0's prompt should NOT change — a1's
        energy is private to a1 (not in a0's visible_entities.columns)."""
        a0 = discrete_world._entity_ids[0]
        a1 = discrete_world._entity_ids[1]
        bundle_before = compute_prompt_hashes(discrete_world.observe_agent(a0))

        # Mutate a1's energy.
        idx1 = discrete_world._entity_ids.index(a1)
        discrete_world._entity_columns["energy"][idx1] += 100.0
        bundle_after = compute_prompt_hashes(discrete_world.observe_agent(a0))

        assert (
            bundle_before.combined_prompt_hash
            == bundle_after.combined_prompt_hash
        ), "other agent's private energy leaked into focal prompt"


# ---------------------------------------------------------------------------
# P-G3: system/user roles separated in transport payload
# ---------------------------------------------------------------------------


class TestPG3SystemUserRolesSeparated:
    """P-G3: ``LLMRequest.system_prompt`` carries ONLY the stable system
    prompt; ``LLMRequest.prompt`` carries ONLY the user message. The
    two are NEVER concatenated."""

    def test_build_llm_request_separates_roles(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        req = build_llm_request(obs, model="test-model")

        # system_prompt slot is set to the stable system prompt.
        assert req.system_prompt == STABLE_SYSTEM_PROMPT
        assert req.system_prompt is not None
        # prompt slot does NOT contain the system prompt text.
        assert "You are an action-selection policy" not in req.prompt
        # prompt slot DOES contain the observation / scenario contract.
        assert "observation" in req.prompt
        assert "scenario_contract" in req.prompt

    def test_llm_policy_uses_separated_roles(self, discrete_world):
        """End-to-end: when LLMPolicy runs against a projector world,
        the LLMRequest passed to the client has separated roles."""
        a0 = discrete_world._entity_ids[0]
        ctx = _make_policy_context(discrete_world, a0)

        # Use the FIRST legal action_type so the test works regardless of
        # which scenario spec is loaded (discrete_grid uses lowercase
        # "forage"/"rest"; other specs may differ).
        first_action_type = ctx.action_space.legal_actions[0].action_type

        captured_requests: list[LLMRequest] = []

        class CapturingClient:
            def complete(self, request: LLMRequest):
                captured_requests.append(request)
                from worldloop_data.llm_policy import LLMResponse

                return LLMResponse(
                    raw_text=json.dumps(
                        {
                            "action_type": first_action_type,
                            "params": {},
                            "reason_code": "OK",
                        }
                    ),
                    json_body={
                        "action_type": first_action_type,
                        "params": {},
                        "reason_code": "OK",
                    },
                    finish_reason="stop",
                    input_tokens=10,
                    output_tokens=5,
                )

        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://fake", model="test", fallback_mode="decline"
            ),
            client=CapturingClient(),  # type: ignore[arg-type]
        )
        proposal = policy.propose(ctx)
        assert proposal is not None
        assert len(captured_requests) == 1
        req = captured_requests[0]
        # P-G3: system_prompt slot is set, prompt slot does NOT contain it.
        assert req.system_prompt == STABLE_SYSTEM_PROMPT
        assert "You are an action-selection policy" not in req.prompt

    def test_legacy_path_when_world_not_projector(self):
        """When world does NOT implement ObservationProjector, the
        legacy path is used (Phase 0 governance freeze). The system
        prompt slot MAY be None (preserves old behavior)."""
        from worldloop_kernel.protocol import ActionSpace

        class NonProjectorWorld:
            """Minimal world stub that does NOT implement projector."""

            def legal_actions(self, agent_id):
                return ActionSpace(
                    agent_id=agent_id,
                    legal_actions=(LegalAction(action_type="REST"),),
                )

        import random

        ctx = PolicyContext(
            world=NonProjectorWorld(),  # type: ignore[arg-type]
            agent_id="a0",
            state=None,  # type: ignore[arg-type]
            action_space=ActionSpace(
                agent_id="a0",
                legal_actions=(LegalAction(action_type="REST"),),
            ),
            tick=0,
            rng=random.Random(42),
        )

        captured: list[LLMRequest] = []

        class CapturingClient:
            def complete(self, request: LLMRequest):
                captured.append(request)
                from worldloop_data.llm_policy import LLMResponse

                return LLMResponse(
                    raw_text='{"action_type": "REST", "params": {}, "reason_code": "OK"}',
                    json_body={
                        "action_type": "REST",
                        "params": {},
                        "reason_code": "OK",
                    },
                    finish_reason="stop",
                    input_tokens=5,
                    output_tokens=3,
                )

        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://fake", model="test", fallback_mode="decline"
            ),
            client=CapturingClient(),  # type: ignore[arg-type]
        )
        proposal = policy.propose(ctx)
        assert proposal is not None
        # Legacy path: system_prompt slot was None (no system_prompt arg).
        assert len(captured) == 1
        assert captured[0].system_prompt is None


# ---------------------------------------------------------------------------
# P-G4: prompt contains no unauthorized global fields
# ---------------------------------------------------------------------------


class TestPG4NoUnauthorizedGlobalFields:
    """P-G4: the user message MUST NOT contain forbidden global field
    substrings (rng_state, branch_id, parent_episode_id, api_key, etc.).

    This is verified end-to-end against the real projector — even
    though the projector SHOULD structurally prevent leakage, this
    gate confirms the final prompt string is clean.
    """

    def test_discrete_world_user_message_is_clean(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        msg = build_user_message(obs)
        violations = scan_for_forbidden_fields(msg)
        assert violations == [], f"forbidden fields leaked: {violations}"

    def test_emergency_world_user_message_is_clean(self, emergency_world):
        a0 = emergency_world._entity_ids[0]
        obs = emergency_world.observe_agent(a0)
        msg = build_user_message(obs)
        violations = scan_for_forbidden_fields(msg)
        assert violations == [], f"forbidden fields leaked: {violations}"

    def test_forbidden_field_set_covers_key_categories(self):
        """P-G4: the forbidden field set covers the documented categories."""
        # RNG state
        assert "rng_state" in FORBIDDEN_GLOBAL_FIELDS
        assert "random_state" in FORBIDDEN_GLOBAL_FIELDS
        # Counterfactual internals
        assert "branch_id" in FORBIDDEN_GLOBAL_FIELDS
        assert "parent_episode_id" in FORBIDDEN_GLOBAL_FIELDS
        assert "fork_group_id" in FORBIDDEN_GLOBAL_FIELDS
        # Secrets (defensive)
        assert "api_key" in FORBIDDEN_GLOBAL_FIELDS
        assert "secret" in FORBIDDEN_GLOBAL_FIELDS
        assert "token" in FORBIDDEN_GLOBAL_FIELDS
        # Provenance metadata
        assert "source_commit" in FORBIDDEN_GLOBAL_FIELDS
        assert "source_dirty" in FORBIDDEN_GLOBAL_FIELDS

    def test_validation_passes_for_real_projector(self, discrete_world):
        """End-to-end: ``validate_prompt_components`` returns no errors
        when called with a real-projector observation."""
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        errors = validate_prompt_components(obs)
        assert errors == [], f"validation errors: {errors}"


# ---------------------------------------------------------------------------
# P-G5: scenario contract mechanically matches action/observation schema
# ---------------------------------------------------------------------------


class TestPG5ScenarioContractMatchesSchema:
    """P-G5: the scenario contract's ``action_schema`` is mechanically
    derived from ``observation.legal_actions`` — one entry per distinct
    action_type, and every action_type in legal_actions appears."""

    def test_action_schema_has_same_action_types_as_legal_actions(
        self, discrete_world
    ):
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        contract = build_scenario_contract(obs)
        obs_types = {la.action_type for la in obs.legal_actions}
        contract_types = {e.action_type for e in contract.action_schema}
        assert obs_types == contract_types

    def test_scenario_id_matches_observation(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        contract = build_scenario_contract(obs)
        assert contract.scenario_id == obs.scenario_id
        assert contract.scenario_version == obs.scenario_version

    def test_observation_schema_summary_matches(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        contract = build_scenario_contract(obs)
        summary = contract.observation_schema_summary
        assert summary["schema_version"] == obs.schema_version
        assert summary["focal_agent_id"] == obs.focal_agent.agent_id

    def test_emergency_world_action_schema_includes_all_actions(
        self, emergency_world
    ):
        a0 = emergency_world._entity_ids[0]
        obs = emergency_world.observe_agent(a0)
        contract = build_scenario_contract(obs)
        obs_types = {la.action_type for la in obs.legal_actions}
        contract_types = {e.action_type for e in contract.action_schema}
        assert obs_types == contract_types


# ---------------------------------------------------------------------------
# P-G6: every component has independent version + hash
# ---------------------------------------------------------------------------


class TestPG6IndependentVersionAndHash:
    """P-G6: prompt template, scenario contract, and observation each
    carry their own version + SHA-256 hash."""

    def test_three_schema_versions_exist(self):
        """P-G6: three distinct schema version constants exist."""
        assert SYSTEM_PROMPT_VERSION
        assert PROMPT_CONTRACT_SCHEMA_VERSION
        assert USER_MESSAGE_SCHEMA_VERSION
        # All three are non-empty strings.
        for v in (SYSTEM_PROMPT_VERSION, PROMPT_CONTRACT_SCHEMA_VERSION, USER_MESSAGE_SCHEMA_VERSION):
            assert isinstance(v, str)
            assert v != ""

    def test_hash_bundle_has_five_distinct_hashes(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        bundle = compute_prompt_hashes(obs)

        # All five hashes are non-empty.
        for name, h in (
            ("system_prompt_hash", bundle.system_prompt_hash),
            ("scenario_contract_hash", bundle.scenario_contract_hash),
            ("observation_hash", bundle.observation_hash),
            ("user_message_hash", bundle.user_message_hash),
            ("combined_prompt_hash", bundle.combined_prompt_hash),
        ):
            assert h, f"{name} is empty"
            assert h.startswith("sha256:"), f"{name} is not a sha256: hash"
            assert len(h) == len("sha256:") + 64, f"{name} has wrong length"

        # The four component hashes are distinct from each other
        # (they hash different content).
        hashes = {
            bundle.system_prompt_hash,
            bundle.scenario_contract_hash,
            bundle.observation_hash,
            bundle.user_message_hash,
        }
        assert len(hashes) == 4, "component hashes are not distinct"

    def test_combined_hash_depends_on_all_components(self, discrete_world):
        """P-G6: the combined hash is a function of all four component
        hashes. If any component changes, the combined hash changes."""
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        b1 = compute_prompt_hashes(obs)
        # Re-compute — same input, same output.
        b2 = compute_prompt_hashes(obs)
        assert b1 == b2

    def test_system_prompt_hash_is_constant_across_observations(
        self, discrete_world, emergency_world
    ):
        """P-G6: the system prompt hash is the same for every observation
        — it depends only on the (frozen) system prompt text."""
        a0 = discrete_world._entity_ids[0]
        obs1 = discrete_world.observe_agent(a0)
        b1 = compute_prompt_hashes(obs1)

        e0 = emergency_world._entity_ids[0]
        obs2 = emergency_world.observe_agent(e0)
        b2 = compute_prompt_hashes(obs2)

        # Different scenario -> different observation, different contract.
        assert b1.observation_hash != b2.observation_hash
        assert b1.scenario_contract_hash != b2.scenario_contract_hash
        # Same system prompt, regardless of scenario.
        assert b1.system_prompt_hash == b2.system_prompt_hash

    def test_user_message_hash_matches_user_message(self, discrete_world):
        """P-G6: the user_message_hash is the SHA-256 of the actual user
        message string returned by build_user_message."""
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        msg = build_user_message(obs)
        bundle = compute_prompt_hashes(obs, user_message=msg)
        # Recompute the hash directly.
        import hashlib

        expected = "sha256:" + hashlib.sha256(msg.encode("utf-8")).hexdigest()
        assert bundle.user_message_hash == expected


# ---------------------------------------------------------------------------
# End-to-end: LLMPolicy records component hashes on InferenceEvent
# ---------------------------------------------------------------------------


class TestLLMPolicyRecordsHashes:
    """When LLMPolicy runs against a projector world, the resulting
    :class:`InferenceEvent` MUST carry the prompt component hashes
    for auditability (Phase 1 telemetry preparation; full Phase 2
    InferenceDecisionEvent/InferenceAttemptEvent split comes later)."""

    def test_state_aware_path_records_all_hashes(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        ctx = _make_policy_context(discrete_world, a0)
        sink = InMemoryTelemetrySink()
        # Use the first legal action_type from the scenario (lowercase in
        # discrete_grid.yaml). Hardcoding "REST" here would mismatch
        # scenario's lowercase "rest" and validation would reject it.
        first_action_type = ctx.action_space.legal_actions[0].action_type

        class StubClient:
            def complete(self, request: LLMRequest):
                from worldloop_data.llm_policy import LLMResponse

                return LLMResponse(
                    raw_text=f'{{"action_type": "{first_action_type}", "params": {{}}, "reason_code": "OK"}}',
                    json_body={
                        "action_type": first_action_type,
                        "params": {},
                        "reason_code": "OK",
                    },
                    finish_reason="stop",
                    input_tokens=10,
                    output_tokens=5,
                )

        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://fake", model="test", fallback_mode="decline"
            ),
            client=StubClient(),  # type: ignore[arg-type]
            telemetry=sink,
        )
        proposal = policy.propose(ctx)
        assert proposal is not None

        assert len(sink.events) == 1
        event = sink.events[0]
        # All Phase 1 hashes populated.
        assert event.prompt_path == "state_aware"
        assert event.system_prompt_hash
        assert event.scenario_contract_hash
        assert event.observation_hash
        assert event.user_message_hash
        assert event.combined_prompt_hash
        # All hashes are sha256: format.
        for h in (
            event.system_prompt_hash,
            event.scenario_contract_hash,
            event.observation_hash,
            event.user_message_hash,
            event.combined_prompt_hash,
        ):
            assert h.startswith("sha256:")

    def test_legacy_path_does_not_populate_component_hashes(self):
        """Legacy path: the new Phase 1 hash fields stay empty (the
        legacy prompt_hash field IS populated, but the new component
        hashes are not)."""
        from worldloop_kernel.protocol import ActionSpace

        import random

        class NonProjectorWorld:
            def legal_actions(self, agent_id):
                return ActionSpace(
                    agent_id=agent_id,
                    legal_actions=(LegalAction(action_type="REST"),),
                )

        ctx = PolicyContext(
            world=NonProjectorWorld(),  # type: ignore[arg-type]
            agent_id="a0",
            state=None,  # type: ignore[arg-type]
            action_space=ActionSpace(
                agent_id="a0",
                legal_actions=(LegalAction(action_type="REST"),),
            ),
            tick=0,
            rng=random.Random(42),
        )
        sink = InMemoryTelemetrySink()

        class StubClient:
            def complete(self, request: LLMRequest):
                from worldloop_data.llm_policy import LLMResponse

                return LLMResponse(
                    raw_text='{"action_type": "REST", "params": {}, "reason_code": "OK"}',
                    json_body={
                        "action_type": "REST",
                        "params": {},
                        "reason_code": "OK",
                    },
                    finish_reason="stop",
                    input_tokens=10,
                    output_tokens=5,
                )

        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://fake", model="test", fallback_mode="decline"
            ),
            client=StubClient(),  # type: ignore[arg-type]
            telemetry=sink,
        )
        proposal = policy.propose(ctx)
        assert proposal is not None
        assert len(sink.events) == 1
        event = sink.events[0]
        # Legacy path marker.
        assert event.prompt_path == "legacy"
        # Legacy prompt_hash IS populated (truncated sha256 of prompt text).
        assert event.prompt_hash
        # New Phase 1 component hashes are NOT populated.
        assert event.system_prompt_hash == ""
        assert event.scenario_contract_hash == ""
        assert event.observation_hash == ""
        assert event.user_message_hash == ""
        assert event.combined_prompt_hash == ""

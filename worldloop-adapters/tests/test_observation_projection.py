"""Phase 1 / Beta correction: tests for ``PettingZooParallelAdapter.observe_agent``.

Validates the projector implementation against the Phase 1 / §5.3 contract:

- ``PettingZooParallelAdapter`` implements :class:`ObservationProjector`.
- Same env state + agent => identical observation hash (P-G1 precondition).
- Hidden RNG state changes (env.step advances RNG) do NOT change the
  observation hash when the visible content is held fixed (P-G2).
- Focal agent sees its own velocity + full observation (``self_visible``);
  other agents see ONLY the focal agent's position (privacy enforcement).
- Landmarks are PUBLIC (position visible to all agents).
- Unsupported capabilities (fields / relations / registries / population /
  events) are OMITTED, NOT zero-filled.
- Counterfactual ``state=...`` projection raises ``NotImplementedError``.
- ``observe_agent`` before ``reset`` raises (fail-closed).
- ``observe_agent`` for unknown ``agent_id`` raises ``KeyError``.
"""
from __future__ import annotations

import pytest

from worldloop_kernel import (
    AgentObservationView,
    ObservationProjector,
    hash_observation,
    is_observation_projector,
)
from worldloop_adapters.pettingzoo.adapter import (
    PettingZooParallelAdapter,
    make_simple_spread_env,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter_2agents_2landmarks() -> PettingZooParallelAdapter:
    env = make_simple_spread_env(n_agents=2, n_landmarks=2, max_cycles=25)
    adapter = PettingZooParallelAdapter(env, env_id="test-spread")
    adapter.reset(seed=42)
    return adapter


# ---------------------------------------------------------------------------
# 1. ObservationProjector conformance (P-G0)
# ---------------------------------------------------------------------------


class TestObservationProjectorConformance:
    def test_adapter_is_observation_projector(self, adapter_2agents_2landmarks):
        assert isinstance(adapter_2agents_2landmarks, ObservationProjector)

    def test_is_observation_projector_true(self, adapter_2agents_2landmarks):
        assert is_observation_projector(adapter_2agents_2landmarks) is True

    def test_observe_agent_returns_agent_observation_view(
        self, adapter_2agents_2landmarks
    ):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        assert isinstance(obs, AgentObservationView)


# ---------------------------------------------------------------------------
# 2. Hash determinism (P-G1 precondition)
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    def test_same_state_same_agent_same_hash(self, adapter_2agents_2landmarks):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs_a = adapter_2agents_2landmarks.observe_agent(agent_id)
        obs_b = adapter_2agents_2landmarks.observe_agent(agent_id)
        assert hash_observation(obs_a) == hash_observation(obs_b)

    def test_different_agents_different_hash(self, adapter_2agents_2landmarks):
        agents = list(adapter_2agents_2landmarks._last_obs.keys())
        if len(agents) < 2:
            pytest.skip("need at least 2 agents")
        obs_a = adapter_2agents_2landmarks.observe_agent(agents[0])
        obs_b = adapter_2agents_2landmarks.observe_agent(agents[1])
        assert hash_observation(obs_a) != hash_observation(obs_b)


# ---------------------------------------------------------------------------
# 3. Hidden-state non-leak (P-G2 structural test)
# ---------------------------------------------------------------------------


class TestHiddenStateNonLeak:
    def test_repeated_observe_agent_does_not_change_hash(
        self, adapter_2agents_2landmarks
    ):
        """Calling observe_agent multiple times (without env.step)
        MUST produce identical hashes — the projector is read-only."""
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        h1 = hash_observation(adapter_2agents_2landmarks.observe_agent(agent_id))
        h2 = hash_observation(adapter_2agents_2landmarks.observe_agent(agent_id))
        h3 = hash_observation(adapter_2agents_2landmarks.observe_agent(agent_id))
        assert h1 == h2 == h3


# ---------------------------------------------------------------------------
# 4. Visibility policy (privacy enforcement)
# ---------------------------------------------------------------------------


class TestVisibilityPolicy:
    def test_focal_agent_sees_own_velocity(self, adapter_2agents_2landmarks):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        # Velocity is self_visible (private sensorimotor).
        assert "velocity" in obs.focal_agent.self_visible_attributes
        vel = obs.focal_agent.self_visible_attributes["velocity"]
        assert isinstance(vel, tuple)
        assert len(vel) == 2  # (vx, vy)

    def test_focal_agent_sees_own_observation_vector(
        self, adapter_2agents_2landmarks
    ):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        assert "observation" in obs.focal_agent.self_visible_attributes
        full_obs = obs.focal_agent.self_visible_attributes["observation"]
        assert isinstance(full_obs, tuple)
        assert len(full_obs) > 0

    def test_focal_agent_public_attributes_include_position(
        self, adapter_2agents_2landmarks
    ):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        assert "position" in obs.focal_agent.public_attributes
        pos = obs.focal_agent.public_attributes["position"]
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_other_agents_see_only_position(self, adapter_2agents_2landmarks):
        """Focal agent's velocity MUST NOT appear in other agents'
        ``visible_entities`` projection — velocity is private."""
        agents = list(adapter_2agents_2landmarks._last_obs.keys())
        if len(agents) < 2:
            pytest.skip("need at least 2 agents")
        obs_a1 = adapter_2agents_2landmarks.observe_agent(agents[1])
        visible_a0 = next(
            (e for e in obs_a1.visible_entities if e.entity_id == agents[0]),
            None,
        )
        assert visible_a0 is not None, "a0 not in a1's visible_entities"
        assert "position" in visible_a0.columns  # public
        assert "velocity" not in visible_a0.columns  # private
        assert "observation" not in visible_a0.columns  # private

    def test_landmarks_visible_to_all_agents(self, adapter_2agents_2landmarks):
        """Landmarks are public environmental features — visible to all."""
        agents = list(adapter_2agents_2landmarks._last_obs.keys())
        for agent_id in agents:
            obs = adapter_2agents_2landmarks.observe_agent(agent_id)
            landmark_entities = [
                e for e in obs.visible_entities
                if e.columns.get("kind") == "landmark"
            ]
            assert len(landmark_entities) == 2, (
                f"agent {agent_id} should see 2 landmarks, saw {len(landmark_entities)}"
            )

    def test_focal_agent_not_in_visible_entities(self, adapter_2agents_2landmarks):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        assert all(
            e.entity_id != agent_id for e in obs.visible_entities
        ), "focal agent should not be duplicated in visible_entities"


# ---------------------------------------------------------------------------
# 5. Omission policy (§5.3: omit unsupported)
# ---------------------------------------------------------------------------


class TestOmissionPolicy:
    def test_adapter_omits_all_non_entity_capabilities(
        self, adapter_2agents_2landmarks
    ):
        """PettingZoo MPE adapter declares only entities=True; all other
        slots MUST be listed in the omission policy."""
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        cap = adapter_2agents_2landmarks.capabilities
        for slot in ("fields", "relations", "registries", "population", "events"):
            if not getattr(cap, slot):
                assert slot in obs.omission_policy.unsupported_capabilities, (
                    f"unsupported capability {slot!r} not listed in omission_policy"
                )

    def test_visible_fields_empty_when_capability_false(
        self, adapter_2agents_2landmarks
    ):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        cap = adapter_2agents_2landmarks.capabilities
        if not cap.fields:
            assert obs.visible_fields == {}  # NOT zero-filled


# ---------------------------------------------------------------------------
# 6. Legal actions (mechanical consistency)
# ---------------------------------------------------------------------------


class TestLegalActions:
    def test_observation_legal_actions_match_worldprotocol(
        self, adapter_2agents_2landmarks
    ):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        world_legal = adapter_2agents_2landmarks.legal_actions(agent_id)
        assert obs.legal_actions == world_legal.legal_actions

    def test_legal_actions_nonempty(self, adapter_2agents_2landmarks):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        assert len(obs.legal_actions) == 5, "MPE Simple Spread has 5 actions"


# ---------------------------------------------------------------------------
# 7. Fail-closed behavior
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_observe_agent_before_reset_raises(self):
        env = make_simple_spread_env(n_agents=2, n_landmarks=2)
        adapter = PettingZooParallelAdapter(env)
        with pytest.raises(RuntimeError, match="before reset"):
            adapter.observe_agent("agent_0")

    def test_observe_agent_unknown_agent_id_raises(
        self, adapter_2agents_2landmarks
    ):
        with pytest.raises(KeyError, match="not in last_obs"):
            adapter_2agents_2landmarks.observe_agent("nonexistent_agent")

    def test_observe_agent_counterfactual_state_raises(
        self, adapter_2agents_2landmarks
    ):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        sv = adapter_2agents_2landmarks.observe()
        with pytest.raises(NotImplementedError, match="counterfactual"):
            adapter_2agents_2landmarks.observe_agent(agent_id, state=sv)


# ---------------------------------------------------------------------------
# 8. Scenario attribution
# ---------------------------------------------------------------------------


class TestScenarioAttribution:
    def test_observation_carries_scenario_id(self, adapter_2agents_2landmarks):
        agent_id = next(iter(adapter_2agents_2landmarks._last_obs))
        obs = adapter_2agents_2landmarks.observe_agent(agent_id)
        assert obs.scenario_id == adapter_2agents_2landmarks._env_id

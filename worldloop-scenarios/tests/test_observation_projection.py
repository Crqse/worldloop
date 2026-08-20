"""Phase 1 / Beta correction: tests for ``ParameterizedWorld.observe_agent``.

Validates the projector implementation against the Phase 1 / §5.3 contract:

- ``ParameterizedWorld`` implements :class:`ObservationProjector` (P-G0).
- Same state + agent + visibility config => identical observation hash
  (P-G1 precondition: deterministic projection).
- Hidden state changes (RNG draws, internal caches) do NOT change the
  observation hash (P-G2 structural test: hidden-state non-leak).
- Focal agent sees its own energy (``self_visible``); other agents do
  NOT see the focal agent's energy (privacy enforcement).
- Unsupported capabilities are OMITTED, NOT zero-filled (§5.3 rule).
- ``omission_policy.unsupported_capabilities`` lists every ``False``
  capability slot.
- Counterfactual ``state=...`` projection raises ``NotImplementedError``.
- ``observe_agent`` before ``reset`` raises (fail-closed).
- ``observe_agent`` for unknown ``agent_id`` raises ``KeyError``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worldloop_kernel import (
    AgentObservationView,
    ObservationProjector,
    hash_observation,
    hash_state,
    is_observation_projector,
)
from worldloop_kernel.action import ActionProposal, ExecutedAction
from worldloop_scenarios.parameterized_world import ParameterizedWorld
from worldloop_scenarios.spec import ScenarioSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_spec(name: str) -> ScenarioSpec:
    data = yaml.safe_load((EXAMPLES_DIR / name).read_text(encoding="utf-8"))
    return ScenarioSpec.from_dict(data)


@pytest.fixture
def discrete_world() -> ParameterizedWorld:
    """Discrete grid world with energy + x + y + alive columns."""
    spec = _load_spec("discrete_grid.yaml")
    world = ParameterizedWorld(spec)
    world.reset(seed=42)
    return world


@pytest.fixture
def graph_world() -> ParameterizedWorld:
    """Graph world with relations + registries enabled."""
    spec = _load_spec("graph_registry.yaml")
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


# ---------------------------------------------------------------------------
# 1. ObservationProjector conformance (P-G0)
# ---------------------------------------------------------------------------


class TestObservationProjectorConformance:
    def test_parameterized_world_is_observation_projector(self, discrete_world):
        assert isinstance(discrete_world, ObservationProjector)

    def test_is_observation_projector_true(self, discrete_world):
        assert is_observation_projector(discrete_world) is True

    def test_observe_agent_returns_agent_observation_view(self, discrete_world):
        agent_id = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(agent_id)
        assert isinstance(obs, AgentObservationView)


# ---------------------------------------------------------------------------
# 2. Hash determinism (P-G1 precondition)
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    def test_same_state_same_agent_same_hash(self, discrete_world):
        agent_id = discrete_world._entity_ids[0]
        obs_a = discrete_world.observe_agent(agent_id)
        obs_b = discrete_world.observe_agent(agent_id)
        assert hash_observation(obs_a) == hash_observation(obs_b)

    def test_different_agents_different_hash(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        a1 = discrete_world._entity_ids[1]
        obs_a = discrete_world.observe_agent(a0)
        obs_b = discrete_world.observe_agent(a1)
        assert hash_observation(obs_a) != hash_observation(obs_b)

    def test_hash_changes_when_focal_energy_changes(self, discrete_world):
        """Self-visible focal state change MUST change the hash (P-G1)."""
        a0 = discrete_world._entity_ids[0]
        obs_before = discrete_world.observe_agent(a0)
        h_before = hash_observation(obs_before)
        # Mutate the focal agent's energy directly via internal state.
        idx = discrete_world._entity_ids.index(a0)
        discrete_world._entity_columns["energy"][idx] += 10.0
        obs_after = discrete_world.observe_agent(a0)
        h_after = hash_observation(obs_after)
        assert h_before != h_after

    def test_hash_changes_when_other_agent_position_changes(self, discrete_world):
        """Visible (public) state of another agent change MUST change the
        focal agent's observation hash — the focal agent sees the other
        agent's public position."""
        a0 = discrete_world._entity_ids[0]
        a1 = discrete_world._entity_ids[1]
        obs_before = discrete_world.observe_agent(a0)
        h_before = hash_observation(obs_before)
        # Move agent a1 to a new position.
        idx1 = discrete_world._entity_ids.index(a1)
        discrete_world._entity_columns["x"][idx1] = (
            discrete_world._entity_columns["x"][idx1] + 1
        )
        obs_after = discrete_world.observe_agent(a0)
        h_after = hash_observation(obs_after)
        assert h_before != h_after


# ---------------------------------------------------------------------------
# 3. Hidden-state non-leak (P-G2 structural test)
# ---------------------------------------------------------------------------


class TestHiddenStateNonLeak:
    def test_rng_state_change_does_not_affect_observation_hash(
        self, discrete_world
    ):
        """Hidden RNG state changes MUST NOT change the observation hash.

        This is the structural P-G2 test: ``AgentObservationView`` has no
        ``rng_state`` field, so RNG draws (which only affect kernel-
        internal ``_rng``) cannot leak into the projection."""
        a0 = discrete_world._entity_ids[0]
        obs_before = discrete_world.observe_agent(a0)
        h_before = hash_observation(obs_before)
        # Draw from the RNG — this changes ``_rng.getstate()`` but MUST
        # NOT change the observation because the observation schema has
        # no field to carry the RNG state.
        assert discrete_world._rng is not None
        discrete_world._rng.random()
        discrete_world._rng.random()
        obs_after = discrete_world.observe_agent(a0)
        h_after = hash_observation(obs_after)
        assert h_before == h_after, (
            "RNG state change leaked into observation hash — schema bug"
        )

    def test_internal_accumulator_change_does_not_alect_observation_hash(
        self, discrete_world
    ):
        """Hidden internal accumulators (e.g., ``_cumulative_births``)
        are NOT in the observation schema; changing them MUST NOT change
        the observation hash."""
        a0 = discrete_world._entity_ids[0]
        obs_before = discrete_world.observe_agent(a0)
        h_before = hash_observation(obs_before)
        discrete_world._cumulative_births += 999
        discrete_world._cumulative_deaths += 888
        obs_after = discrete_world.observe_agent(a0)
        h_after = hash_observation(obs_after)
        assert h_before == h_after


# ---------------------------------------------------------------------------
# 4. Visibility policy (privacy enforcement)
# ---------------------------------------------------------------------------


class TestVisibilityPolicy:
    def test_focal_agent_sees_own_energy(self, discrete_world):
        """Focal agent's energy appears in ``self_visible_attributes``."""
        a0 = discrete_world._entity_ids[0]
        idx = discrete_world._entity_ids.index(a0)
        expected_energy = discrete_world._entity_columns["energy"][idx]
        obs = discrete_world.observe_agent(a0)
        assert "energy" in obs.focal_agent.self_visible_attributes
        assert obs.focal_agent.self_visible_attributes["energy"] == expected_energy

    def test_focal_agent_public_attributes_include_position(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        # x, y, alive are public; energy is self_visible only.
        assert "x" in obs.focal_agent.public_attributes
        assert "y" in obs.focal_agent.public_attributes
        assert "alive" in obs.focal_agent.public_attributes
        assert "energy" not in obs.focal_agent.public_attributes

    def test_other_agents_do_not_see_focal_energy(self, discrete_world):
        """Focal agent's energy MUST NOT appear in any other agent's
        ``visible_entities`` projection — energy is private."""
        a0 = discrete_world._entity_ids[0]
        a1 = discrete_world._entity_ids[1]
        # a1's observation of a0 should not include energy.
        obs_a1 = discrete_world.observe_agent(a1)
        visible_a0 = next(
            (e for e in obs_a1.visible_entities if e.entity_id == a0),
            None,
        )
        assert visible_a0 is not None, "focal agent a0 not in a1's visible_entities"
        assert "energy" not in visible_a0.columns, (
            "energy leaked into other agent's observation — privacy violation"
        )
        # Public columns ARE visible.
        assert "x" in visible_a0.columns
        assert "y" in visible_a0.columns

    def test_focal_agent_not_in_visible_entities(self, discrete_world):
        """Focal agent appears in ``focal_agent``, NOT in ``visible_entities``."""
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        assert all(e.entity_id != a0 for e in obs.visible_entities), (
            "focal agent should not be duplicated in visible_entities"
        )


# ---------------------------------------------------------------------------
# 5. Omission policy (§5.3: omit unsupported, don't补伪零)
# ---------------------------------------------------------------------------


class TestOmissionPolicy:
    def test_discrete_world_omits_unsupported_capabilities(self, discrete_world):
        """Discrete grid world has fields=False, relations=False,
        registries=False (per discrete_grid.yaml). The omission policy
        MUST list these as unsupported."""
        obs = discrete_world.observe_agent(discrete_world._entity_ids[0])
        cap = discrete_world.capabilities
        for slot in ("fields", "relations", "registries", "population", "events"):
            if not getattr(cap, slot):
                assert slot in obs.omission_policy.unsupported_capabilities, (
                    f"unsupported capability {slot!r} not listed in omission_policy"
                )
                assert slot in obs.omission_policy.omitted_slots

    def test_graph_world_omits_fields_when_disabled(self, graph_world):
        """Graph world has fields=False (per graph_registry.yaml)."""
        obs = graph_world.observe_agent(graph_world._entity_ids[0])
        cap = graph_world.capabilities
        if not cap.fields:
            assert "fields" in obs.omission_policy.unsupported_capabilities
            assert obs.visible_fields == {}  # NOT zero-filled

    def test_graph_world_visible_relations_when_enabled(self, graph_world):
        """Graph world has relations=True; visible_relations MUST be
        populated with the actual edges."""
        obs = graph_world.observe_agent(graph_world._entity_ids[0])
        cap = graph_world.capabilities
        if cap.relations:
            assert len(obs.visible_relations) > 0, (
                "relations capability=True but visible_relations is empty"
            )

    def test_emergency_world_visible_fields_when_enabled(self, emergency_world):
        """Emergency resource world has fields=True; visible_fields MUST
        be populated with the actual channel values (NOT zero-filled)."""
        obs = emergency_world.observe_agent(emergency_world._entity_ids[0])
        cap = emergency_world.capabilities
        if cap.fields:
            assert len(obs.visible_fields) > 0, (
                "fields capability=True but visible_fields is empty"
            )


# ---------------------------------------------------------------------------
# 6. Legal actions (mechanical consistency)
# ---------------------------------------------------------------------------


class TestLegalActions:
    def test_observation_legal_actions_match_worldprotocol(self, discrete_world):
        """The ``legal_actions`` in the observation MUST match
        ``world.legal_actions(agent_id)`` — the projector delegates to
        the existing WorldProtocol method."""
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        world_legal = discrete_world.legal_actions(a0)
        assert obs.legal_actions == world_legal.legal_actions

    def test_legal_actions_nonempty(self, discrete_world):
        a0 = discrete_world._entity_ids[0]
        obs = discrete_world.observe_agent(a0)
        assert len(obs.legal_actions) > 0, "discrete_grid should have forage + rest"


# ---------------------------------------------------------------------------
# 7. Fail-closed behavior
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_observe_agent_before_reset_raises(self):
        """observe_agent MUST raise if called before reset()."""
        spec = _load_spec("discrete_grid.yaml")
        world = ParameterizedWorld(spec)
        # No reset() call — internal state is empty.
        with pytest.raises((KeyError, RuntimeError)):
            world.observe_agent("e0")

    def test_observe_agent_unknown_agent_id_raises(self, discrete_world):
        with pytest.raises(KeyError, match="not in spawned"):
            discrete_world.observe_agent("nonexistent_agent")

    def test_observe_agent_counterfactual_state_raises(self, discrete_world):
        """state=... MUST raise NotImplementedError in v0.1."""
        a0 = discrete_world._entity_ids[0]
        from worldloop_kernel import StateView

        # Build a dummy StateView (not actually used).
        sv = discrete_world.observe()
        with pytest.raises(NotImplementedError, match="counterfactual"):
            discrete_world.observe_agent(a0, state=sv)


# ---------------------------------------------------------------------------
# 8. Scenario attribution
# ---------------------------------------------------------------------------


class TestScenarioAttribution:
    def test_observation_carries_scenario_id(self, discrete_world):
        obs = discrete_world.observe_agent(discrete_world._entity_ids[0])
        assert obs.scenario_id == discrete_world._spec.scenario.scenario_id
        assert obs.scenario_version == discrete_world._spec.scenario.scenario_version

    def test_observation_carries_tick(self, discrete_world):
        obs = discrete_world.observe_agent(discrete_world._entity_ids[0])
        assert obs.tick == discrete_world._tick


# ---------------------------------------------------------------------------
# 9. Step integration (observation changes after step)
# ---------------------------------------------------------------------------


class TestStepIntegration:
    def test_observation_changes_after_step(self, discrete_world):
        """After a FORAGE action, the focal agent's energy changes, so
        the observation hash MUST change."""
        a0 = discrete_world._entity_ids[0]
        obs_before = discrete_world.observe_agent(a0)
        h_before = hash_observation(obs_before)
        # Apply a forage action.
        proposal = ActionProposal(
            agent_id=a0,
            action_type="forage",
            params={},
            proposed_at_tick=discrete_world._tick,
            proposer="test",
        )
        executed, _ = discrete_world.validate_action(proposal)
        discrete_world.step(executed)
        obs_after = discrete_world.observe_agent(a0)
        h_after = hash_observation(obs_after)
        assert h_before != h_after, (
            "observation hash should change after a state-changing step"
        )

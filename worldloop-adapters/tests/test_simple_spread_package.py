"""Tests for make_simple_spread_package (B4) — requires pettingzoo + mpe2.

Verifies the Simple Spread factory against the B4 disciplines:
- package protocol shape (pipeline duck typing)
- config parameters flow into world_parameters_hash
- same-seed world determinism (bit-identical state hashes)
- agent lifecycle: alive column, no live agents after truncation
- terminations vs truncations kept distinct in receipts
- Q3-style replay: raw ExecutedAction steps without prior validate_action
"""
from __future__ import annotations

import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("mpe2")

from worldloop_kernel.action import ActionProposal, ExecutedAction
from worldloop_kernel.canonical import hash_state

from worldloop_adapters.scenario_package import (
    ExternalScenarioPackage,
    LifecycleAwareParallelWorld,
    SimpleSpreadConfig,
    make_simple_spread_package,
)


def _first_alive_agent(state) -> str:
    ids = state.entities.ids
    alive = state.entities.columns["alive"]
    for eid, a in zip(ids, alive):
        if a:
            return str(eid)
    raise AssertionError("no alive agent found")


def _step_agent(world, agent_id: str, discrete: int, tick: int):
    proposal = ActionProposal(
        agent_id=agent_id,
        action_type="move",
        params={"discrete_action": discrete},
        proposed_at_tick=tick,
        proposer="test",
    )
    executed, receipt = world.validate_action(proposal)
    assert receipt.success
    return world.step(executed)


# ---------------------------------------------------------------------------
# Package shape + hash
# ---------------------------------------------------------------------------


class TestSimpleSpreadPackageShape:
    def test_returns_external_scenario_package(self):
        pkg = make_simple_spread_package()
        assert isinstance(pkg, ExternalScenarioPackage)

    def test_spec_protocol_shape(self):
        pkg = make_simple_spread_package()
        assert pkg.spec.scenario.scenario_id == pkg.scenario_id
        d = pkg.spec.to_dict()
        assert d["world_parameters"]["env"] == "mpe2/simple_spread_v3"

    def test_config_params_in_hash(self):
        base = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=25))
        more_agents = make_simple_spread_package(
            SimpleSpreadConfig(n_agents=3, max_cycles=25)
        )
        more_cycles = make_simple_spread_package(
            SimpleSpreadConfig(n_agents=2, max_cycles=50)
        )
        assert base.world_parameters_hash != more_agents.world_parameters_hash
        assert base.world_parameters_hash != more_cycles.world_parameters_hash

    def test_same_config_same_hash(self):
        a = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        b = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        assert a.world_parameters_hash == b.world_parameters_hash

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            SimpleSpreadConfig(n_agents=0)
        with pytest.raises(ValueError):
            SimpleSpreadConfig(max_cycles=0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestSimpleSpreadDeterminism:
    def test_same_seed_same_initial_hash(self):
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        w1 = pkg.world_factory(42)
        w2 = pkg.world_factory(42)
        assert hash_state(w1.reset(seed=42)) == hash_state(w2.reset(seed=42))

    def test_same_seed_same_trajectory_hashes(self):
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        hashes: list[list[str]] = []
        for _ in range(2):
            world = pkg.world_factory(42)
            state = world.reset(seed=42)
            run_hashes = []
            for tick in range(4):
                agent = _first_alive_agent(state)
                record = _step_agent(world, agent, (tick % 4) + 1, tick)
                run_hashes.append(record.state_after_hash)
                state = world.observe()
            hashes.append(run_hashes)
        assert hashes[0] == hashes[1]

    def test_different_seed_different_initial_hash(self):
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        w1 = pkg.world_factory(42)
        w2 = pkg.world_factory(43)
        assert hash_state(w1.reset(seed=42)) != hash_state(w2.reset(seed=43))


# ---------------------------------------------------------------------------
# Agent lifecycle: alive column + no proposals after disappearance
# ---------------------------------------------------------------------------


class TestSimpleSpreadLifecycle:
    def test_alive_column_present_and_landmarks_dead(self):
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        world = pkg.world_factory(42)
        state = world.reset(seed=42)
        cols = state.entities.columns
        assert "alive" in cols
        for eid, kind, alive in zip(
            state.entities.ids, cols["kind"], cols["alive"]
        ):
            if kind == "agent":
                assert alive is True, f"agent {eid} should start alive"
            else:
                assert alive is False, f"landmark {eid} must never be alive"

    def test_no_alive_agents_after_truncation(self):
        """After max_cycles, agents disappear → alive all False.

        The rollout orchestrator (`_pick_agent`) filters on the alive
        column, so no further actions are proposed — the B4 honesty rule
        "agent 消失后不再提议动作".
        """
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=3))
        world = pkg.world_factory(42)
        state = world.reset(seed=42)
        last_record = None
        for tick in range(3):
            agent = _first_alive_agent(state)
            last_record = _step_agent(world, agent, 1, tick)
            state = world.observe()
        # max_cycles=3 reached: env truncated every agent.
        alive = state.entities.columns["alive"]
        assert not any(alive), "no agent may be reported alive after truncation"
        # Truncation (not termination) is what fired — flags are distinct.
        receipt = next(iter(last_record.receipts.values()))
        info = receipt.diagnostics["info"]
        assert info["truncation"] is True
        assert info["termination"] is False

    def test_termination_and_truncation_flags_recorded(self):
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        world = pkg.world_factory(42)
        state = world.reset(seed=42)
        record = _step_agent(world, _first_alive_agent(state), 2, 0)
        receipt = next(iter(record.receipts.values()))
        info = receipt.diagnostics["info"]
        assert "termination" in info and "truncation" in info


# ---------------------------------------------------------------------------
# Replay path + provenance
# ---------------------------------------------------------------------------


class TestSimpleSpreadReplayAndProvenance:
    def test_step_accepts_unvalidated_executed_action(self):
        """Q3 replay feeds ExecutedActions rebuilt from records — no
        validate_action round. The world must execute them identically."""
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        world = pkg.world_factory(42)
        state = world.reset(seed=42)
        agent = _first_alive_agent(state)
        record = _step_agent(world, agent, 3, 0)
        recorded_action = record.executed_actions[agent]

        replay_world = pkg.world_factory(42)
        replay_world.reset(seed=42)
        # Rebuild the action as Q3 does (fresh ExecutedAction, no validate).
        rebuilt = ExecutedAction(
            agent_id=recorded_action.agent_id,
            action_type=recorded_action.action_type,
            params=dict(recorded_action.params),
            executed_at_tick=recorded_action.executed_at_tick,
            proposal_hash=recorded_action.proposal_hash,
        )
        replay_record = replay_world.step(rebuilt)
        assert replay_record.state_after_hash == record.state_after_hash

    def test_provenance_contains_seed(self):
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        world = pkg.world_factory(42)
        state = world.reset(seed=42)
        record = _step_agent(world, _first_alive_agent(state), 1, 0)
        assert record.provenance.get("seed") == "42"

    def test_world_is_lifecycle_aware_adapter(self):
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=2, max_cycles=10))
        world = pkg.world_factory(42)
        assert isinstance(world, LifecycleAwareParallelWorld)
        # Honest capability declaration: entities only; no WST/graph slots.
        cap = world.capabilities
        assert cap.entities is True
        assert cap.fields is False and cap.relations is False
        assert cap.exact_restore is True

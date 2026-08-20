"""Phase 5 joint action mode tests for the PettingZoo adapter (§13.5).

Evidence mapping to the Phase 5 gates:
- E-G1: step_joint records executed actions + receipts for ALL active agents.
- E-G2: per-agent reward/termination/truncation reconcile between receipts,
  provenance JSON mirrors, and a bare reference env stepped identically.
- E-G4: checkpoint → immediate restore hash equality per env family
  (allowlist admission evidence for EXACT_RESTORE_VERIFIED_ENV_FAMILIES).
- E-G7: generic (unverified) env families get exact_restore=False.
- E-G8: sequential compatibility mode still works after the joint upgrade.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("mpe2")

from worldloop_kernel import (
    ActionProposal,
    JointAction,
    JointActionError,
    hash_state,
    supports_joint_actions,
)
from worldloop_adapters.pettingzoo import (
    EXACT_RESTORE_VERIFIED_ENV_FAMILIES,
    is_exact_restore_verified,
    make_pettingzoo_capability,
    verify_immediate_restore,
)
from worldloop_adapters.scenario_package import (
    SimpleSpreadConfig,
    SimpleTagConfig,
    make_simple_spread_package,
    make_simple_tag_package,
)


SEED = 123


def _make_world(package_name: str):
    if package_name == "spread":
        pkg = make_simple_spread_package(SimpleSpreadConfig(n_agents=3, max_cycles=25))
    else:
        pkg = make_simple_tag_package(
            SimpleTagConfig(num_good=1, num_adversaries=2, num_obstacles=1, max_cycles=25)
        )
    world = pkg.world_factory(SEED)
    world.reset(seed=SEED)
    return world


def _active_agents(world) -> tuple[str, ...]:
    return tuple(world._current_active_agents())


def _joint_proposal(world, tick: int, actions: dict[str, int]) -> JointAction:
    active = _active_agents(world)
    proposals = {
        agent: ActionProposal(
            agent_id=agent,
            action_type="move",
            params={"discrete_action": actions.get(agent, 0)},
            proposed_at_tick=tick,
            proposer="test",
        )
        for agent in active
        if agent in actions
    }
    return JointAction(
        tick=tick,
        active_agents=active,
        proposals_by_agent=proposals,
        missing_agent_policy="stay",
    )


# ---------------------------------------------------------------------------
# Discovery + validate_joint_action
# ---------------------------------------------------------------------------


class TestValidateJointAction:
    def test_adapter_supports_joint_actions(self):
        world = _make_world("spread")
        assert supports_joint_actions(world)

    def test_full_proposals_become_executed_stage(self):
        world = _make_world("spread")
        active = _active_agents(world)
        joint = _joint_proposal(world, 0, {a: (i % 5) for i, a in enumerate(active)})
        executed_joint, joint_receipt = world.validate_joint_action(joint)
        assert executed_joint.is_executed_stage
        assert set(executed_joint.executed_by_agent) == set(active)
        assert set(joint_receipt.receipts_by_agent) == set(active)
        assert all(r.success for r in joint_receipt.receipts_by_agent.values())

    def test_missing_agent_synthesized_stay(self):
        world = _make_world("spread")
        active = _active_agents(world)
        # Only the first agent proposes; the rest are synthesized.
        joint = _joint_proposal(world, 0, {active[0]: 2})
        executed_joint, joint_receipt = world.validate_joint_action(joint)
        assert set(executed_joint.executed_by_agent) == set(active)
        for agent in active[1:]:
            executed = executed_joint.executed_by_agent[agent]
            assert executed.params["discrete_action"] == 0
            assert joint_receipt.receipts_by_agent[agent].success

    def test_illegal_proposal_substituted_and_rejected(self):
        world = _make_world("spread")
        active = _active_agents(world)
        proposals = {
            active[0]: ActionProposal(
                agent_id=active[0],
                action_type="move",
                params={"discrete_action": 99},  # out of range
                proposed_at_tick=0,
                proposer="test",
            )
        }
        joint = JointAction(
            tick=0,
            active_agents=active,
            proposals_by_agent=proposals,
            missing_agent_policy="stay",
        )
        executed_joint, joint_receipt = world.validate_joint_action(joint)
        rej = joint_receipt.receipts_by_agent[active[0]]
        assert not rej.success
        assert rej.outcome_code == "illegal_action"
        # Substituted executed action is STAY(0).
        assert executed_joint.executed_by_agent[active[0]].params["discrete_action"] == 0

    def test_active_agents_mismatch_raises(self):
        world = _make_world("spread")
        joint = JointAction(
            tick=0,
            active_agents=("ghost_0", "ghost_1"),
            missing_agent_policy="stay",
        )
        with pytest.raises(JointActionError, match="active agents"):
            world.validate_joint_action(joint)


# ---------------------------------------------------------------------------
# step_joint — E-G1 all agents recorded / E-G2 per-agent reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package_name", ["spread", "tag"])
class TestStepJointAllAgents:
    def test_all_active_agents_recorded(self, package_name):
        world = _make_world(package_name)
        active = _active_agents(world)
        joint = _joint_proposal(world, 0, {a: (i % 5) for i, a in enumerate(active)})
        executed_joint, _ = world.validate_joint_action(joint)
        record = world.step_joint(executed_joint)
        # E-G1: every active agent appears in executed_actions AND receipts.
        assert set(map(str, record.executed_actions)) == set(active)
        assert set(map(str, record.receipts)) == set(active)
        assert record.provenance["joint_step"] == "true"
        assert record.provenance["execution_mode"] == "joint"
        assert json.loads(record.provenance["active_agents_before"]) == list(active)

    def test_per_agent_reconciliation_against_bare_env(self, package_name):
        # E-G2: receipts + provenance mirrors must equal a bare reference
        # env stepped with the identical joint action dict.
        world = _make_world(package_name)
        active = _active_agents(world)
        actions = {a: ((i + 1) % 5) for i, a in enumerate(active)}

        # Reference: bare env, same seed, same actions.
        from worldloop_adapters.pettingzoo.adapter import (
            make_simple_spread_env,
            make_simple_tag_env,
        )

        if package_name == "spread":
            ref_env = make_simple_spread_env(n_agents=3, n_landmarks=3, max_cycles=25)
        else:
            ref_env = make_simple_tag_env(
                num_good=1, num_adversaries=2, num_obstacles=1, max_cycles=25
            )
        ref_env.reset(seed=SEED)
        _, ref_rewards, ref_term, ref_trunc, _ = ref_env.step(dict(actions))

        joint = _joint_proposal(world, 0, actions)
        executed_joint, _ = world.validate_joint_action(joint)
        record = world.step_joint(executed_joint)

        rewards_by_agent = json.loads(record.provenance["rewards_by_agent"])
        terms_by_agent = json.loads(record.provenance["terminations_by_agent"])
        truncs_by_agent = json.loads(record.provenance["truncations_by_agent"])
        for agent in active:
            receipt = record.receipts[agent]
            # Receipt diagnostics vs bare env.
            assert receipt.diagnostics["reward"] == pytest.approx(
                float(ref_rewards[agent])
            )
            assert receipt.diagnostics["info"]["termination"] == bool(ref_term[agent])
            assert receipt.diagnostics["info"]["truncation"] == bool(ref_trunc[agent])
            # Provenance mirrors vs receipt.
            assert rewards_by_agent[agent] == pytest.approx(
                receipt.diagnostics["reward"]
            )
            assert terms_by_agent[agent] == receipt.diagnostics["info"]["termination"]
            assert truncs_by_agent[agent] == receipt.diagnostics["info"]["truncation"]
        ref_env.close()

    def test_same_seed_same_actions_same_hash(self, package_name):
        # Determinism of joint mode: two worlds, same seed, same joint
        # action → identical state_after_hash.
        hashes = []
        for _ in range(2):
            world = _make_world(package_name)
            active = _active_agents(world)
            joint = _joint_proposal(world, 0, {a: (i % 5) for i, a in enumerate(active)})
            executed_joint, _ = world.validate_joint_action(joint)
            record = world.step_joint(executed_joint)
            hashes.append(record.state_after_hash)
        assert hashes[0] == hashes[1]


class TestStepJointSemantics:
    def test_joint_differs_from_sequential_focal_stay(self):
        # Joint mode is REAL: all agents moving ≠ focal moving + others STAY.
        world_joint = _make_world("spread")
        active = _active_agents(world_joint)
        joint = _joint_proposal(world_joint, 0, {a: 1 for a in active})
        executed_joint, _ = world_joint.validate_joint_action(joint)
        joint_record = world_joint.step_joint(executed_joint)

        world_seq = _make_world("spread")
        proposal = ActionProposal(
            agent_id=active[0],
            action_type="move",
            params={"discrete_action": 1},
            proposed_at_tick=0,
            proposer="test",
        )
        executed, _ = world_seq.validate_action(proposal)
        seq_record = world_seq.step(executed)

        assert joint_record.state_after_hash != seq_record.state_after_hash
        assert seq_record.provenance["execution_mode"] == "sequential_focal_stay"

    def test_step_joint_requires_executed_stage(self):
        world = _make_world("spread")
        joint = _joint_proposal(world, 0, {})
        with pytest.raises(JointActionError, match="executed-stage"):
            world.step_joint(joint)

    def test_replay_path_cache_miss_reproduces_hash(self):
        # Executed-stage joint action re-submitted to a FRESH world (no
        # validate cache) must reproduce the same state_after_hash.
        world1 = _make_world("spread")
        active = _active_agents(world1)
        joint = _joint_proposal(world1, 0, {a: (i % 5) for i, a in enumerate(active)})
        executed_joint, _ = world1.validate_joint_action(joint)
        record1 = world1.step_joint(executed_joint)

        world2 = _make_world("spread")
        # No validate_joint_action call — forces the re-derive path.
        record2 = world2.step_joint(executed_joint)
        assert record2.state_after_hash == record1.state_after_hash
        assert record2.executed_actions.keys() == record1.executed_actions.keys()

    def test_agent_disappearance_stops_proposals(self):
        # E-G3 (unit-level): run Simple Spread to truncation (max_cycles);
        # after agents disappear, active set is empty and a stale joint
        # action is rejected fail-visible.
        world = _make_world("spread")
        tick = 0
        active = _active_agents(world)
        stale_joint = None
        while _active_agents(world):
            current = _active_agents(world)
            joint = _joint_proposal(world, tick, {a: 1 for a in current})
            executed_joint, _ = world.validate_joint_action(joint)
            if stale_joint is None:
                stale_joint = joint
            world.step_joint(executed_joint)
            tick += 1
            if tick > 30:
                pytest.fail("env did not truncate within 30 ticks")
        # All agents disappeared (truncation at max_cycles=25).
        assert not _active_agents(world)
        # Alive column reports no live agents.
        sv = world.observe()
        alive = sv.entities.columns.get("alive", ())
        assert not any(alive)
        # A stale proposal-stage joint action now fails validation.
        with pytest.raises(JointActionError, match="active agents"):
            world.validate_joint_action(stale_joint)


# ---------------------------------------------------------------------------
# E-G4 / E-G7 — exact-restore allowlist
# ---------------------------------------------------------------------------


class TestExactRestoreAllowlist:
    @pytest.mark.parametrize("package_name", ["spread", "tag"])
    def test_immediate_restore_hash_equal(self, package_name):
        # Allowlist admission evidence: checkpoint → immediate restore →
        # observe hash equality, on a mid-episode state (after 3 joint steps).
        world = _make_world(package_name)
        for tick in range(3):
            active = _active_agents(world)
            joint = _joint_proposal(world, tick, {a: (tick + 1) % 5 for a in active})
            executed_joint, _ = world.validate_joint_action(joint)
            world.step_joint(executed_joint)
        ok, message = verify_immediate_restore(world)
        assert ok, f"immediate restore failed: {message}"

    def test_verified_families_match_allowlist(self):
        assert is_exact_restore_verified("mpe2/simple_spread_v3")
        assert is_exact_restore_verified("mpe2/simple_tag_v3")
        assert set(EXACT_RESTORE_VERIFIED_ENV_FAMILIES) == {
            "mpe2/simple_spread_v3",
            "mpe2/simple_tag_v3",
        }

    def test_generic_family_gets_no_exact_restore_claim(self):
        # E-G7: unverified families must NOT claim exact_restore.
        cap = make_pettingzoo_capability("some/unverified_env_v0")
        assert cap.exact_restore is False
        assert cap.executable_deterministic_replay is False

    def test_verified_family_gets_exact_restore_claim(self):
        cap = make_pettingzoo_capability("mpe2/simple_tag_v3")
        assert cap.exact_restore is True
        assert cap.executable_deterministic_replay is True


# ---------------------------------------------------------------------------
# E-G8 — sequential compatibility mode regression
# ---------------------------------------------------------------------------


class TestSequentialBackwardCompat:
    def test_sequential_step_still_works(self):
        world = _make_world("spread")
        active = _active_agents(world)
        proposal = ActionProposal(
            agent_id=active[0],
            action_type="move",
            params={"discrete_action": 3},
            proposed_at_tick=0,
            proposer="test",
        )
        executed, receipt = world.validate_action(proposal)
        assert receipt.success
        record = world.step(executed)
        assert record.provenance["joint_step"] == "false"
        assert record.provenance["execution_mode"] == "sequential_focal_stay"
        assert set(map(str, record.executed_actions)) == {active[0]}

    def test_sequential_determinism_unchanged(self):
        # Same seed + same sequential action → same hash (twice).
        hashes = []
        for _ in range(2):
            world = _make_world("spread")
            active = _active_agents(world)
            proposal = ActionProposal(
                agent_id=active[0],
                action_type="move",
                params={"discrete_action": 2},
                proposed_at_tick=0,
                proposer="test",
            )
            executed, _ = world.validate_action(proposal)
            record = world.step(executed)
            hashes.append(record.state_after_hash)
        assert hashes[0] == hashes[1]

    def test_joint_hash_stable_between_validate_and_step(self):
        # The pending-joint cache is keyed by the executed joint hash;
        # hashing must be stable across the validate → step boundary.
        world = _make_world("spread")
        active = _active_agents(world)
        joint = _joint_proposal(world, 0, {a: 4 for a in active})
        executed_joint, _ = world.validate_joint_action(joint)
        h1 = hash_state(executed_joint)
        h2 = hash_state(executed_joint)
        assert h1 == h2
        assert h1 in world._pending_joint

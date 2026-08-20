"""Tests for Phase 5 joint action primitives (worldloop_kernel.joint).

Covers:
- JointAction proposal-stage / executed-stage validation rules
- missing_agent_policy semantics (noop / stay / error)
- JointReceipt validation
- JointActionWorld protocol discovery (supports_joint_actions)
- canonical hashing of joint types (determinism + sensitivity)
- backward compatibility: single-agent types and exports untouched
"""

from __future__ import annotations

import pytest

from worldloop_kernel import (
    ActionProposal,
    ActionReceipt,
    ExecutedAction,
    JointAction,
    JointActionError,
    JointActionWorld,
    JointReceipt,
    MISSING_AGENT_ERROR,
    MISSING_AGENT_NOOP,
    MISSING_AGENT_POLICIES,
    MISSING_AGENT_STAY,
    OUTCOME_OK,
    hash_state,
    supports_joint_actions,
)


def _proposal(agent_id: str, tick: int = 0, action_type: str = "MOVE") -> ActionProposal:
    return ActionProposal(
        agent_id=agent_id,
        action_type=action_type,
        params={"direction": 1},
        proposed_at_tick=tick,
        proposer="random",
    )


def _executed(agent_id: str, tick: int = 0, action_type: str = "MOVE") -> ExecutedAction:
    return ExecutedAction(
        agent_id=agent_id,
        action_type=action_type,
        params={"direction": 1},
        executed_at_tick=tick,
        proposal_hash="sha256:deadbeef",
    )


def _receipt() -> ActionReceipt:
    return ActionReceipt(
        executed_action_hash="sha256:cafebabe",
        outcome_code=OUTCOME_OK,
        success=True,
        energy_delta=0.0,
    )


# ---------------------------------------------------------------------------
# JointAction — construction and validation
# ---------------------------------------------------------------------------


class TestJointActionValidation:
    def test_proposal_stage_valid(self):
        ja = JointAction(
            tick=3,
            active_agents=("a0", "a1"),
            proposals_by_agent={"a0": _proposal("a0", 3), "a1": _proposal("a1", 3)},
        )
        assert ja.missing_agent_policy == MISSING_AGENT_NOOP
        assert not ja.is_executed_stage

    def test_executed_stage_valid(self):
        ja = JointAction(
            tick=3,
            active_agents=("a0", "a1"),
            executed_by_agent={"a0": _executed("a0", 3), "a1": _executed("a1", 3)},
        )
        assert ja.is_executed_stage

    def test_negative_tick_rejected(self):
        with pytest.raises(JointActionError, match="tick"):
            JointAction(tick=-1, active_agents=("a0",))

    def test_empty_active_agents_rejected(self):
        with pytest.raises(JointActionError, match="active_agents"):
            JointAction(tick=0, active_agents=())

    def test_duplicate_active_agents_rejected(self):
        with pytest.raises(JointActionError, match="duplicates"):
            JointAction(tick=0, active_agents=("a0", "a0"))

    def test_unknown_missing_agent_policy_rejected(self):
        with pytest.raises(JointActionError, match="missing_agent_policy"):
            JointAction(
                tick=0, active_agents=("a0",), missing_agent_policy="silently_skip"
            )

    def test_proposal_for_inactive_agent_rejected(self):
        with pytest.raises(JointActionError, match="not in active_agents"):
            JointAction(
                tick=0,
                active_agents=("a0",),
                proposals_by_agent={"ghost": _proposal("ghost")},
            )

    def test_proposal_key_agent_id_mismatch_rejected(self):
        with pytest.raises(JointActionError, match="MUST match"):
            JointAction(
                tick=0,
                active_agents=("a0", "a1"),
                proposals_by_agent={"a0": _proposal("a1")},
            )

    def test_partial_proposals_allowed_with_noop_policy(self):
        ja = JointAction(
            tick=0,
            active_agents=("a0", "a1"),
            proposals_by_agent={"a0": _proposal("a0")},
            missing_agent_policy=MISSING_AGENT_NOOP,
        )
        assert set(ja.proposals_by_agent) == {"a0"}

    def test_partial_proposals_allowed_with_stay_policy(self):
        ja = JointAction(
            tick=0,
            active_agents=("a0", "a1"),
            proposals_by_agent={"a0": _proposal("a0")},
            missing_agent_policy=MISSING_AGENT_STAY,
        )
        assert ja.missing_agent_policy == MISSING_AGENT_STAY

    def test_partial_proposals_rejected_with_error_policy(self):
        with pytest.raises(JointActionError, match="missing"):
            JointAction(
                tick=0,
                active_agents=("a0", "a1"),
                proposals_by_agent={"a0": _proposal("a0")},
                missing_agent_policy=MISSING_AGENT_ERROR,
            )

    def test_full_proposals_accepted_with_error_policy(self):
        ja = JointAction(
            tick=0,
            active_agents=("a0", "a1"),
            proposals_by_agent={"a0": _proposal("a0"), "a1": _proposal("a1")},
            missing_agent_policy=MISSING_AGENT_ERROR,
        )
        assert set(ja.proposals_by_agent) == {"a0", "a1"}

    def test_executed_stage_must_cover_all_active(self):
        with pytest.raises(JointActionError, match="cover exactly"):
            JointAction(
                tick=0,
                active_agents=("a0", "a1"),
                executed_by_agent={"a0": _executed("a0")},
            )

    def test_executed_stage_surplus_agent_rejected(self):
        with pytest.raises(JointActionError, match="surplus"):
            JointAction(
                tick=0,
                active_agents=("a0",),
                executed_by_agent={
                    "a0": _executed("a0"),
                    "a1": _executed("a1"),
                },
            )

    def test_executed_key_agent_id_mismatch_rejected(self):
        with pytest.raises(JointActionError, match="MUST match"):
            JointAction(
                tick=0,
                active_agents=("a0", "a1"),
                executed_by_agent={
                    "a0": _executed("a0"),
                    "a1": _executed("a0"),
                },
            )

    def test_replay_style_executed_only_with_error_policy(self):
        # Replay consumers construct executed-stage joint actions with
        # empty proposals; ERROR policy must not reject this shape.
        ja = JointAction(
            tick=5,
            active_agents=("a0",),
            executed_by_agent={"a0": _executed("a0", 5)},
            missing_agent_policy=MISSING_AGENT_ERROR,
        )
        assert ja.is_executed_stage

    def test_int_agent_ids_supported(self):
        ja = JointAction(
            tick=0,
            active_agents=(0, 1),
            proposals_by_agent={
                0: ActionProposal(
                    agent_id=0,
                    action_type="MOVE",
                    params={},
                    proposed_at_tick=0,
                    proposer="random",
                )
            },
        )
        assert 0 in ja.proposals_by_agent

    def test_policies_constant(self):
        assert MISSING_AGENT_POLICIES == ("noop", "stay", "error")


# ---------------------------------------------------------------------------
# JointReceipt
# ---------------------------------------------------------------------------


class TestJointReceipt:
    def test_valid(self):
        jr = JointReceipt(tick=2, receipts_by_agent={"a0": _receipt()})
        assert jr.tick == 2

    def test_negative_tick_rejected(self):
        with pytest.raises(JointActionError, match="tick"):
            JointReceipt(tick=-1, receipts_by_agent={"a0": _receipt()})

    def test_empty_receipts_rejected(self):
        with pytest.raises(JointActionError, match="non-empty"):
            JointReceipt(tick=0, receipts_by_agent={})


# ---------------------------------------------------------------------------
# Protocol discovery
# ---------------------------------------------------------------------------


class _JointCapableWorld:
    def validate_joint_action(self, joint):
        return joint, JointReceipt(tick=joint.tick, receipts_by_agent={"a0": _receipt()})

    def step_joint(self, joint, *, exogenous=None):
        return None


class _SequentialOnlyWorld:
    def validate_action(self, proposal):
        return None

    def step(self, action, *, exogenous=None):
        return None


class TestProtocolDiscovery:
    def test_joint_capable_world_detected(self):
        world = _JointCapableWorld()
        assert supports_joint_actions(world)
        assert isinstance(world, JointActionWorld)

    def test_sequential_only_world_not_detected(self):
        world = _SequentialOnlyWorld()
        assert not supports_joint_actions(world)
        assert not isinstance(world, JointActionWorld)

    def test_non_callable_attributes_not_detected(self):
        class Fake:
            validate_joint_action = "not callable"
            step_joint = 42

        assert not supports_joint_actions(Fake())


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------


class TestJointHashing:
    def test_hash_deterministic(self):
        ja1 = JointAction(
            tick=1,
            active_agents=("a0", "a1"),
            proposals_by_agent={"a0": _proposal("a0", 1), "a1": _proposal("a1", 1)},
        )
        ja2 = JointAction(
            tick=1,
            active_agents=("a0", "a1"),
            # Insertion order differs; canonical mapping encoding sorts keys.
            proposals_by_agent={"a1": _proposal("a1", 1), "a0": _proposal("a0", 1)},
        )
        assert hash_state(ja1) == hash_state(ja2)
        assert hash_state(ja1).startswith("sha256:")

    def test_hash_sensitive_to_content(self):
        base = JointAction(
            tick=1,
            active_agents=("a0",),
            proposals_by_agent={"a0": _proposal("a0", 1, "MOVE")},
        )
        changed = JointAction(
            tick=1,
            active_agents=("a0",),
            proposals_by_agent={"a0": _proposal("a0", 1, "STAY")},
        )
        assert hash_state(base) != hash_state(changed)

    def test_joint_receipt_hashable(self):
        jr = JointReceipt(tick=0, receipts_by_agent={"a0": _receipt()})
        assert hash_state(jr).startswith("sha256:")


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_single_agent_types_unchanged(self):
        # The single-agent flow's types must remain importable and
        # constructible exactly as before Phase 5.
        p = _proposal("a0")
        e = _executed("a0")
        r = _receipt()
        assert p.agent_id == "a0"
        assert e.proposal_hash
        assert r.success

    def test_version_bumped(self):
        import worldloop_kernel

        assert worldloop_kernel.__version__ == "0.1.3"

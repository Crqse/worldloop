"""A-05 Gymnasium adapter conformance tests.

Validates that :class:`GymnasiumAdapter` satisfies the kernel
:class:`WorldProtocol` and M2 Gate §12.7 (a)-(j) for a Gymnasium
single-agent env (CartPole-v1).

Run:
  python -m pytest tests/test_gymnasium_conformance.py -v
"""
from __future__ import annotations

from typing import Any

import pytest

from worldloop_kernel import (
    ActionProposal,
    CapabilityProfile,
    Checkpoint,
    ExecutedAction,
    StateView,
    TransitionRecord,
    WorldProtocol,
    hash_state,
)
from worldloop_kernel.recorder import TransitionRecorder

from worldloop_adapters.gymnasium import (
    GYMNASIUM_WORLD_ID,
    GymnasiumAdapter,
    make_cartpole_env,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adapter(seed: int = 42) -> GymnasiumAdapter:
    env = make_cartpole_env()
    adapter = GymnasiumAdapter(env=env, env_id="cartpole_v1_a05")
    adapter.reset(seed=seed)
    return adapter


def _proposal(*, agent_id: str = "agent_0", discrete: int = 1, tick: int = 0) -> ActionProposal:
    return ActionProposal(
        agent_id=agent_id,
        action_type="step",
        params={"discrete_action": discrete},
        proposed_at_tick=tick,
        proposer="test",
    )


AGENT_ID = "agent_0"


# ---------------------------------------------------------------------------
# M2 Gate (a): protocol conformance
# ---------------------------------------------------------------------------


class TestM2GateAProtocolConformance:
    def test_isinstance_world_protocol(self):
        adapter = _make_adapter()
        assert isinstance(adapter, WorldProtocol)

    def test_legal_actions_closed_discrete(self):
        adapter = _make_adapter()
        aspace = adapter.legal_actions(AGENT_ID)
        assert aspace.is_closed is True
        # CartPole-v1 has Discrete(2) action space.
        assert len(aspace.legal_actions) == 2

    def test_validate_then_step_round_trip(self):
        adapter = _make_adapter()
        proposal = _proposal(discrete=1)
        executed, receipt_placeholder = adapter.validate_action(proposal)
        assert receipt_placeholder.success is True
        record = adapter.step(executed)
        assert record.state_before_hash
        assert record.state_after_hash


# ---------------------------------------------------------------------------
# M2 Gate (b): capability consistency
# M2 Gate (c): no fabricated state
# ---------------------------------------------------------------------------


class TestM2GateBCapabilityConsistency:
    def test_capabilities_returns_profile(self):
        adapter = _make_adapter()
        cap = adapter.capabilities
        assert isinstance(cap, CapabilityProfile)

    def test_capabilities_entities_true(self):
        adapter = _make_adapter()
        assert adapter.capabilities.entities is True

    def test_capabilities_other_slots_false(self):
        adapter = _make_adapter()
        cap = adapter.capabilities
        assert cap.fields is False
        assert cap.relations is False
        assert cap.registries is False
        assert cap.population is False
        assert cap.events is False

    def test_capabilities_exact_restore_true(self):
        adapter = _make_adapter()
        assert adapter.capabilities.exact_restore is True

    def test_observe_no_fabricated_slots(self):
        adapter = _make_adapter()
        sv = adapter.observe()
        assert sv.entities is not None
        assert sv.fields is None
        assert sv.relations is None
        assert sv.registries is None
        assert sv.population is None
        assert sv.events is None

    def test_observe_missing_mask_empty(self):
        adapter = _make_adapter()
        sv = adapter.observe()
        assert sv.missing_mask == {} or all(v is False for v in sv.missing_mask.values())

    def test_observe_single_entity(self):
        """Gymnasium is single-agent → exactly 1 entity."""
        adapter = _make_adapter()
        sv = adapter.observe()
        assert len(sv.entities.ids) == 1
        assert sv.entities.ids[0] == AGENT_ID


# ---------------------------------------------------------------------------
# M2 Gate (g): same seed reset
# ---------------------------------------------------------------------------


class TestM2GateGSameSeedReset:
    def test_same_seed_same_state_hash(self):
        a1 = _make_adapter(seed=42)
        a2 = _make_adapter(seed=42)
        assert hash_state(a1.observe()) == hash_state(a2.observe())

    def test_different_seed_different_state_hash(self):
        a1 = _make_adapter(seed=42)
        a2 = _make_adapter(seed=999)
        assert hash_state(a1.observe()) != hash_state(a2.observe())

    def test_observe_idempotent(self):
        adapter = _make_adapter()
        assert hash_state(adapter.observe()) == hash_state(adapter.observe())


# ---------------------------------------------------------------------------
# M2 Gate (d)/(e)/(f): step counts + reward + termination
# ---------------------------------------------------------------------------


class TestM2GateDEFStepConsistency:
    def test_step_returns_transition_record(self):
        adapter = _make_adapter()
        proposal = _proposal(discrete=1)
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)
        assert isinstance(record, TransitionRecord)

    def test_candidate_executed_receipt_counts_match(self):
        adapter = _make_adapter()
        proposal = _proposal(discrete=0)
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)
        assert len(record.candidate_actions) == len(record.executed_actions) == len(record.receipts)
        assert len(record.candidate_actions) == 1

    def test_reward_matches_direct_env(self):
        """Adapter step reward == direct env.step reward (same seed + action)."""
        # Direct env run.
        env_direct = make_cartpole_env()
        obs_d, _ = env_direct.reset(seed=42)
        # CartPole action 0 = push left.
        _, reward_direct, _, _, _ = env_direct.step(0)

        # Adapter run.
        env_adapter = make_cartpole_env()
        adapter = GymnasiumAdapter(env=env_adapter, env_id="cartpole_direct")
        adapter.reset(seed=42)
        proposal = _proposal(discrete=0)
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)
        receipt = record.receipts[AGENT_ID]
        reward_adapter = float(receipt.diagnostics.get("reward", 0.0))

        assert reward_adapter == pytest.approx(reward_direct, abs=1e-9), (
            f"Reward mismatch: adapter={reward_adapter}, direct={reward_direct}"
        )

    def test_termination_truncation_in_diagnostics_info(self):
        adapter = _make_adapter()
        proposal = _proposal(discrete=1)
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)
        receipt = record.receipts[AGENT_ID]
        info = receipt.diagnostics.get("info", {})
        assert "termination" in info
        assert "truncation" in info
        assert isinstance(info["termination"], bool)
        assert isinstance(info["truncation"], bool)


# ---------------------------------------------------------------------------
# M2 Gate (h): mid-trajectory replay
# ---------------------------------------------------------------------------


class TestM2GateHMidTrajectoryReplay:
    def test_checkpoint_step_restore_tick(self):
        adapter = _make_adapter()
        sv0 = adapter.observe()
        ckpt = adapter.checkpoint()
        assert isinstance(ckpt, Checkpoint)

        # Step to change state.
        proposal = _proposal(discrete=1)
        executed, _ = adapter.validate_action(proposal)
        adapter.step(executed)

        # Restore.
        adapter.restore(ckpt)
        sv_restored = adapter.observe()
        assert sv_restored.meta.tick == sv0.meta.tick

    def test_checkpoint_round_trip_entity_ids(self):
        adapter = _make_adapter()
        sv_before = adapter.observe()
        ids_before = sv_before.entities.ids if sv_before.entities else ()
        ckpt = adapter.checkpoint()

        # Step to change state.
        proposal = _proposal(discrete=1)
        executed, _ = adapter.validate_action(proposal)
        adapter.step(executed)

        adapter.restore(ckpt)
        sv_after = adapter.observe()
        ids_after = sv_after.entities.ids if sv_after.entities else ()
        assert ids_before == ids_after


# ---------------------------------------------------------------------------
# M2 Gate (i): restore downgrade (MPE/Gymnasium all support restore)
# ---------------------------------------------------------------------------


class TestM2GateIRestoreDowngrade:
    def test_capability_exact_restore_true(self):
        adapter = _make_adapter()
        assert adapter.capabilities.exact_restore is True

    def test_restore_after_step_works(self):
        adapter = _make_adapter()
        ckpt = adapter.checkpoint()
        proposal = _proposal(discrete=1)
        executed, _ = adapter.validate_action(proposal)
        adapter.step(executed)
        adapter.restore(ckpt)  # should not raise


# ---------------------------------------------------------------------------
# M2 Gate (j): recorder produces trajectory
# ---------------------------------------------------------------------------


class TestM2GateJRecorder:
    def test_single_env_recorded(self, tmp_path):
        """Gymnasium adapter produces a trajectory via kernel TransitionRecorder."""
        from pathlib import Path

        recorder = TransitionRecorder(
            output_dir=Path(tmp_path),
            world_id=GYMNASIUM_WORLD_ID,
            producer_version="0.1.0",
            validate=True,
        )
        adapter = _make_adapter()
        for tick in range(5):
            proposal = _proposal(discrete=tick % 2, tick=tick)
            executed, _ = adapter.validate_action(proposal)
            record = adapter.step(executed)
            recorder.append(record)
        recorder.close()

        record_files = list(Path(tmp_path).rglob("*.jsonl")) + list(
            Path(tmp_path).rglob("*.json")
        )
        assert len(record_files) > 0, f"Recorder should have written files to {tmp_path}"


# ---------------------------------------------------------------------------
# Cross-adapter: PettingZoo + Gymnasium via same kernel recorder
# ---------------------------------------------------------------------------


class TestCrossAdapterRecorder:
    """Verify that the same kernel recorder can ingest records from both
    PettingZoo (multi-agent) and Gymnasium (single-agent) adapters."""

    def test_two_adapter_families_same_recorder(self, tmp_path):
        from pathlib import Path
        from worldloop_adapters.pettingzoo import (
            PETTINGZOO_WORLD_ID,
            PettingZooParallelAdapter,
            make_simple_spread_env,
        )

        # The recorder's world_id is fixed at construction; we use a
        # generic id here and rely on per-record producer_id for tracking.
        recorder = TransitionRecorder(
            output_dir=Path(tmp_path),
            world_id="worldloop-multi-adapter-test",
            producer_version="0.1.0",
            validate=True,
        )

        # PettingZoo: 2 records.
        pz_env = make_simple_spread_env(n_agents=2, n_landmarks=2, max_cycles=25)
        pz_adapter = PettingZooParallelAdapter(env=pz_env, env_id="spread_cross")
        pz_adapter.reset(seed=42)
        pz_agent = str(next(iter(pz_adapter._last_obs.keys())))
        for tick in range(2):
            proposal = ActionProposal(
                agent_id=pz_agent,
                action_type="move",
                params={"discrete_action": 1},
                proposed_at_tick=tick,
                proposer="test",
            )
            executed, _ = pz_adapter.validate_action(proposal)
            recorder.append(pz_adapter.step(executed))

        # Gymnasium: 2 records.
        gym_env = make_cartpole_env()
        gym_adapter = GymnasiumAdapter(env=gym_env, env_id="cartpole_cross")
        gym_adapter.reset(seed=42)
        for tick in range(2):
            proposal = ActionProposal(
                agent_id="agent_0",
                action_type="step",
                params={"discrete_action": tick % 2},
                proposed_at_tick=tick,
                proposer="test",
            )
            executed, _ = gym_adapter.validate_action(proposal)
            recorder.append(gym_adapter.step(executed))

        recorder.close()
        record_files = list(Path(tmp_path).rglob("*.jsonl")) + list(
            Path(tmp_path).rglob("*.json")
        )
        assert len(record_files) > 0

"""A-06/A-07 OpenEnv adapter conformance tests.

Validates that :class:`OpenEnvWorldAdapter` and
:class:`OpenEnvServerWrapper` satisfy the kernel :class:`WorldProtocol`
and M2 Gate §12.7 (a)-(j) for an OpenEnv-style env (mock client).

Run:
  python -m pytest tests/test_openenv_conformance.py -v
"""
from __future__ import annotations

from pathlib import Path
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

from worldloop_adapters.openenv import (
    OPENENV_WORLD_ID,
    InProcessOpenEnvClient,
    OpenEnvServerWrapper,
    OpenEnvWorldAdapter,
    make_openenv_capability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_client(max_tick: int = 10) -> InProcessOpenEnvClient:
    return InProcessOpenEnvClient(max_tick=max_tick)


def _make_adapter(seed: int = 42, max_tick: int = 10) -> OpenEnvWorldAdapter:
    client = _make_client(max_tick=max_tick)
    adapter = OpenEnvWorldAdapter(client=client, env_id="openenv_mock")
    adapter.reset(seed=seed)
    return adapter


def _proposal(*, agent_id: str = "agent_0", discrete: int = 0, tick: int = 0) -> ActionProposal:
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
        # Mock client has 2 actions (0, 1).
        assert len(aspace.legal_actions) == 2

    def test_legal_actions_returns_discrete_params(self):
        adapter = _make_adapter()
        aspace = adapter.legal_actions(AGENT_ID)
        discretes = sorted(
            la.params.get("discrete_action") for la in aspace.legal_actions
        )
        assert discretes == [0, 1]

    def test_validate_then_step_round_trip(self):
        adapter = _make_adapter()
        proposal = _proposal(discrete=0)
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
        """OpenEnv mock is single-agent → exactly 1 entity."""
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

    def test_different_seed_same_state_hash(self):
        """Mock client ignores seed, so different seeds yield same state.

        This is a known limitation of the mock client; real OpenEnv envs
        should produce different state for different seeds.
        """
        a1 = _make_adapter(seed=42)
        a2 = _make_adapter(seed=999)
        # Mock client returns the same initial state regardless of seed.
        assert hash_state(a1.observe()) == hash_state(a2.observe())

    def test_observe_idempotent(self):
        adapter = _make_adapter()
        assert hash_state(adapter.observe()) == hash_state(adapter.observe())

    def test_reset_clears_pending_actions(self):
        adapter = _make_adapter()
        proposal = _proposal(discrete=0)
        adapter.validate_action(proposal)
        # Reset should clear pending action cache.
        adapter.reset(seed=42)
        # After reset, the previously-validated action is gone; step
        # should reject it.
        executed = ExecutedAction(
            agent_id=proposal.agent_id,
            action_type=proposal.action_type,
            params=proposal.params,
            executed_at_tick=proposal.proposed_at_tick,
            proposal_hash=hash_state(proposal),
        )
        record = adapter.step(executed)
        receipt = record.receipts[AGENT_ID]
        assert receipt.success is False


# ---------------------------------------------------------------------------
# M2 Gate (d)/(e)/(f): step counts + reward + termination
# ---------------------------------------------------------------------------


class TestM2GateDEFStepConsistency:
    def test_step_returns_transition_record(self):
        adapter = _make_adapter()
        proposal = _proposal(discrete=0)
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

    def test_reward_matches_direct_client(self):
        """Adapter step reward == direct client.step reward (same action)."""
        # Direct client run.
        client_direct = _make_client()
        client_direct.reset(seed=42)
        _, reward_direct, _, _, _ = client_direct.step(0)

        # Adapter run.
        client_adapter = _make_client()
        adapter = OpenEnvWorldAdapter(client=client_adapter, env_id="openenv_direct")
        adapter.reset(seed=42)
        proposal = _proposal(discrete=0)
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)
        receipt = record.receipts[AGENT_ID]
        reward_adapter = float(receipt.diagnostics.get("reward", 0.0))

        assert reward_adapter == pytest.approx(reward_direct, abs=1e-9), (
            f"Reward mismatch: adapter={reward_adapter}, direct={reward_direct}"
        )

    def test_reward_action_1_higher_than_action_0(self):
        """Action 1 increments tick by 2 (reward = new tick); action 0 by 1."""
        adapter = _make_adapter()
        # Step with action 0 → tick=1, reward=1.
        p0 = _proposal(discrete=0, tick=0)
        e0, _ = adapter.validate_action(p0)
        r0 = adapter.step(e0)
        reward_0 = float(r0.receipts[AGENT_ID].diagnostics["reward"])

        adapter2 = _make_adapter()
        # Step with action 1 → tick=2, reward=2.
        p1 = _proposal(discrete=1, tick=0)
        e1, _ = adapter2.validate_action(p1)
        r1 = adapter2.step(e1)
        reward_1 = float(r1.receipts[AGENT_ID].diagnostics["reward"])

        assert reward_1 > reward_0
        assert reward_0 == pytest.approx(1.0, abs=1e-9)
        assert reward_1 == pytest.approx(2.0, abs=1e-9)

    def test_termination_truncation_in_diagnostics_info(self):
        adapter = _make_adapter(max_tick=2)
        proposal = _proposal(discrete=0)
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)
        receipt = record.receipts[AGENT_ID]
        info = receipt.diagnostics.get("info", {})
        assert "termination" in info
        assert "truncation" in info
        assert isinstance(info["termination"], bool)
        assert isinstance(info["truncation"], bool)

    def test_termination_true_at_max_tick(self):
        """Mock client terminates when tick >= max_tick."""
        adapter = _make_adapter(max_tick=1)
        # action 0 → tick=1 → terminated (1 >= 1).
        proposal = _proposal(discrete=0)
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)
        receipt = record.receipts[AGENT_ID]
        info = receipt.diagnostics.get("info", {})
        assert info["termination"] is True

    def test_termination_false_below_max_tick(self):
        adapter = _make_adapter(max_tick=10)
        proposal = _proposal(discrete=0)
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)
        receipt = record.receipts[AGENT_ID]
        info = receipt.diagnostics.get("info", {})
        assert info["termination"] is False

    def test_reject_illegal_action(self):
        """Action outside legal_actions is rejected with OUTCOME_ILLEGAL_ACTION."""
        adapter = _make_adapter()
        # Action 99 is not in (0, 1).
        proposal = _proposal(discrete=99)
        executed, receipt = adapter.validate_action(proposal)
        assert receipt.success is False
        # Step should also produce a rejected receipt.
        record = adapter.step(executed)
        step_receipt = record.receipts[AGENT_ID]
        assert step_receipt.success is False


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
        proposal = _proposal(discrete=0)
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

    def test_checkpoint_step_restore_state_hash_matches(self):
        """After restore, observe() hash must match pre-step hash."""
        adapter = _make_adapter()
        sv_before = adapter.observe()
        hash_before = hash_state(sv_before)
        ckpt = adapter.checkpoint()

        # Step twice to change state meaningfully.
        for action in (0, 1):
            proposal = _proposal(discrete=action)
            executed, _ = adapter.validate_action(proposal)
            adapter.step(executed)

        # Hash should differ before restore.
        assert hash_state(adapter.observe()) != hash_before

        adapter.restore(ckpt)
        assert hash_state(adapter.observe()) == hash_before


# ---------------------------------------------------------------------------
# M2 Gate (i): restore downgrade
# ---------------------------------------------------------------------------


class TestM2GateIRestoreDowngrade:
    def test_capability_exact_restore_true(self):
        adapter = _make_adapter()
        assert adapter.capabilities.exact_restore is True

    def test_restore_after_step_works(self):
        adapter = _make_adapter()
        ckpt = adapter.checkpoint()
        proposal = _proposal(discrete=0)
        executed, _ = adapter.validate_action(proposal)
        adapter.step(executed)
        adapter.restore(ckpt)  # should not raise


# ---------------------------------------------------------------------------
# M2 Gate (j): recorder produces trajectory
# ---------------------------------------------------------------------------


class TestM2GateJRecorder:
    def test_single_env_recorded(self, tmp_path):
        """OpenEnv adapter produces a trajectory via kernel TransitionRecorder."""
        recorder = TransitionRecorder(
            output_dir=Path(tmp_path),
            world_id=OPENENV_WORLD_ID,
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
# A-07: OpenEnvServerWrapper (kernel world → OpenEnv server)
# ---------------------------------------------------------------------------


class TestOpenEnvServerWrapper:
    """Verify the reverse direction: a kernel world exposed as an OpenEnv
    server can be driven by an OpenEnv client (or another adapter)."""

    def test_action_space_returns_discrete_ints(self):
        adapter = _make_adapter()
        wrapper = OpenEnvServerWrapper(world=adapter)
        assert wrapper.action_space() == (0, 1)

    def test_reset_returns_state_dict(self):
        adapter = _make_adapter()
        wrapper = OpenEnvServerWrapper(world=adapter)
        state = wrapper.reset(seed=42)
        assert isinstance(state, dict)
        assert "observation" in state
        assert "tick" in state
        assert state["tick"] == 0

    def test_state_returns_observation(self):
        adapter = _make_adapter()
        wrapper = OpenEnvServerWrapper(world=adapter)
        wrapper.reset(seed=42)
        state = wrapper.state()
        assert isinstance(state["observation"], list)
        # Initial mock state after reset: [0.0, 0.0, 0.0, 0.0] (tick=0).
        assert state["observation"] == [0.0, 0.0, 0.0, 0.0]

    def test_step_returns_5_tuple(self):
        adapter = _make_adapter()
        wrapper = OpenEnvServerWrapper(world=adapter)
        wrapper.reset(seed=42)
        result = wrapper.step(0)
        assert len(result) == 5
        state, reward, terminated, truncated, info = result
        assert isinstance(state, dict)
        assert reward == pytest.approx(1.0, abs=1e-9)
        assert terminated is False
        assert truncated is False
        assert info["tick"] == 1

    def test_step_action_1_increments_by_2(self):
        adapter = _make_adapter()
        wrapper = OpenEnvServerWrapper(world=adapter)
        wrapper.reset(seed=42)
        _, reward, _, _, _ = wrapper.step(1)
        # Action 1 → tick=2 → reward=2.
        assert reward == pytest.approx(2.0, abs=1e-9)

    def test_step_illegal_action_returns_rejected_info(self):
        adapter = _make_adapter()
        wrapper = OpenEnvServerWrapper(world=adapter)
        wrapper.reset(seed=42)
        _, reward, terminated, truncated, info = wrapper.step(99)
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert info.get("rejected") is True

    def test_termination_at_max_tick(self):
        adapter = _make_adapter(max_tick=2)
        wrapper = OpenEnvServerWrapper(world=adapter)
        wrapper.reset(seed=42)
        # Step to tick=1.
        wrapper.step(0)
        # Step to tick=2 → terminated.
        _, _, terminated, _, _ = wrapper.step(0)
        assert terminated is True

    def test_round_trip_via_adapter_and_wrapper(self):
        """OpenEnvServerWrapper(kernel world) can be re-wrapped by
        OpenEnvWorldAdapter, and the round trip preserves state semantics."""
        # Inner: a kernel world (adapter around mock client).
        inner_client = _make_client(max_tick=10)
        inner_adapter = OpenEnvWorldAdapter(client=inner_client, env_id="inner")
        # Wrap it as an OpenEnv server.
        wrapper = OpenEnvServerWrapper(world=inner_adapter)
        # Re-wrap the server as an OpenEnv client (the wrapper itself
        # satisfies the OpenEnv client duck type).
        outer_adapter = OpenEnvWorldAdapter(client=wrapper, env_id="outer")
        outer_adapter.reset(seed=42)

        # Drive the outer adapter and check tick progression.
        proposal = _proposal(discrete=0, tick=0)
        executed, _ = outer_adapter.validate_action(proposal)
        record = outer_adapter.step(executed)
        receipt = record.receipts[AGENT_ID]
        # Mock client: action 0 → tick=1 → reward=1.
        assert float(receipt.diagnostics["reward"]) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Cross-adapter: PettingZoo + Gymnasium + OpenEnv via same kernel recorder
# ---------------------------------------------------------------------------


class TestCrossAdapterRecorder:
    """Verify that the same kernel recorder can ingest records from
    PettingZoo, Gymnasium, AND OpenEnv adapters (3 env families)."""

    def test_three_adapter_families_same_recorder(self, tmp_path):
        from worldloop_adapters.gymnasium import (
            GymnasiumAdapter,
            make_cartpole_env,
        )
        from worldloop_adapters.pettingzoo import (
            PettingZooParallelAdapter,
            make_simple_spread_env,
        )

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

        # OpenEnv: 2 records.
        oe_client = _make_client(max_tick=20)
        oe_adapter = OpenEnvWorldAdapter(client=oe_client, env_id="openenv_cross")
        oe_adapter.reset(seed=42)
        for tick in range(2):
            proposal = ActionProposal(
                agent_id="agent_0",
                action_type="step",
                params={"discrete_action": tick % 2},
                proposed_at_tick=tick,
                proposer="test",
            )
            executed, _ = oe_adapter.validate_action(proposal)
            recorder.append(oe_adapter.step(executed))

        recorder.close()
        record_files = list(Path(tmp_path).rglob("*.jsonl")) + list(
            Path(tmp_path).rglob("*.json")
        )
        assert len(record_files) > 0

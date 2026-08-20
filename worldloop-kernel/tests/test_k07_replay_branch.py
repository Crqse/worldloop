"""K-07 explicit tests for checkpoint codec + replay + branch.

Verifies (per main plan §10.4 M0 Gate (e) "exact replay 100% consistent"):
- :func:`compute_checkpoint_checksum`: stable SHA-256, differs on any
  payload / tick / codec / state_view / rng_bundle change.
- :func:`verify_checkpoint_restoration`: success path + four failure
  paths (state mismatch, checksum mismatch, world.restore raises,
  world.observe raises).
- :func:`replay`: bit-identical per-tick hashes on deterministic worlds,
  invariant violations captured (not raised), empty actions just verify
  restoration, capability gating (``exact_restore=False`` →
  ``replay_consistent=False``), ``world.step`` raises → ``ReplayError``.
- :func:`branch`: parent state preserved across branches, branches
  diverge from parent at the right tick, branch errors captured (not
  raised), parent restoration failure → ``ReplayError``.
- Top-level package re-exports the 8 K-07 symbols.

Per lesson L-target-a1b2-02, M0 Gate (e) requires explicit verification.
Per lesson L-target-a1b2-08, invariant/replay semantics must follow
dataclass docstrings strictly.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any, Mapping

import pytest

from tests.test_types import (
    make_capability,
    make_state_meta,
    make_entity_table,
    make_state_view,
)


# ---------------------------------------------------------------------------
# _ToyWorld — minimal WorldProtocol implementation for K-07 tests
# ---------------------------------------------------------------------------
#
# Internal state: tick (int) + score (float).
# - step(action): tick += 1, score += action.params.get("delta", 1.0)
# - observe(): build StateView with single entity "a1" and column score
# - checkpoint(): pickle (tick, score) into opaque_payload
# - restore(cp): unpickle opaque_payload, set tick/score
#
# Deterministic + exact_restore=True → replay must be bit-identical.


class _ToyWorld:
    """Minimal :class:`WorldProtocol` implementation for K-07 tests.

    Implements every method of :class:`WorldProtocol` with the simplest
    possible deterministic behavior. Used by K-07 tests (and reused by
    K-08 toy world tests in expanded form). Not part of the public
    kernel API.
    """

    WORLD_ID = "toy-world-v1"
    WORLD_VERSION = "0.0.1"
    PAYLOAD_CODEC = "pickle+v1"

    def __init__(
        self,
        *,
        capabilities: "Any | None" = None,
        restore_should_fail: bool = False,
        observe_should_fail: bool = False,
        step_should_fail: bool = False,
        observe_state_offset: float = 0.0,
    ) -> None:
        from worldloop_kernel import CapabilityProfile

        self._cap: CapabilityProfile = capabilities or make_capability()
        self._tick: int = 0
        self._score: float = 0.0
        # Failure injection for negative tests.
        self._restore_should_fail = restore_should_fail
        self._observe_should_fail = observe_should_fail
        self._step_should_fail = step_should_fail
        # If non-zero, observe() will return a StateView whose score is
        # offset by this amount — simulates a world whose restore does
        # not actually reproduce the checkpoint's state.
        self._observe_state_offset = observe_state_offset

    # --- WorldProtocol properties ---

    @property
    def capabilities(self):
        return self._cap

    # --- WorldProtocol methods ---

    def reset(self, seed: int, parameters: "Mapping[str, Any] | None" = None):
        self._tick = 0
        self._score = float(parameters.get("initial_score", 0.0)) if parameters else 0.0
        return self._build_state_view()

    def observe(self):
        if self._observe_should_fail:
            raise RuntimeError("observe injected failure")
        return self._build_state_view(offset=self._observe_state_offset)

    def legal_actions(self, agent_id, state=None):
        from worldloop_kernel import ActionSpace, LegalAction

        return ActionSpace(
            agent_id=agent_id,
            legal_actions=(
                LegalAction(
                    action_type="TICK",
                    params={"delta": 1.0},
                    description="advance score by delta (default 1.0)",
                ),
            ),
            is_closed=False,
        )

    def validate_action(self, proposal):
        from worldloop_kernel import ExecutedAction, ActionReceipt, OUTCOME_OK
        from worldloop_kernel.canonical import hash_state

        executed = ExecutedAction(
            agent_id=proposal.agent_id,
            action_type=proposal.action_type,
            params=proposal.params,
            executed_at_tick=proposal.proposed_at_tick,
            proposal_hash=hash_state(proposal),
        )
        receipt = ActionReceipt(
            executed_action_hash=hash_state(executed),
            outcome_code=OUTCOME_OK,
            success=True,
            energy_delta=-1.0,
        )
        return executed, receipt

    def step(self, action, exogenous=None):
        from worldloop_kernel import (
            TransitionRecord,
            PROTOCOL_SCHEMA_VERSION,
            StateDelta,
        )
        from worldloop_kernel.canonical import hash_state
        from worldloop_kernel.diff_apply import diff_state

        if self._step_should_fail:
            raise RuntimeError("step injected failure")

        before = self._build_state_view()
        # Apply the action: tick advances by 1, score += delta (default 1.0).
        delta = float(action.params.get("delta", 1.0))
        self._tick += 1
        self._score += delta
        after = self._build_state_view()

        # Build a receipt for this action (success path).
        from worldloop_kernel import ActionReceipt, OUTCOME_OK
        from worldloop_kernel.canonical import hash_state as _hash

        receipt = ActionReceipt(
            executed_action_hash=_hash(action),
            outcome_code=OUTCOME_OK,
            success=True,
            energy_delta=-1.0,
        )

        return TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id=self.WORLD_ID,
            producer_version=self.WORLD_VERSION,
            tick=before.meta.tick,
            state_before_hash=hash_state(before),
            candidate_actions={},
            executed_actions={action.agent_id: action},
            exogenous_input=exogenous,
            receipts={action.agent_id: receipt},
            state_delta=diff_state(before, after),
            state_after_hash=hash_state(after),
            capability_profile=self._cap,
            provenance={},
        )

    def checkpoint(self):
        from worldloop_kernel import Checkpoint, PROTOCOL_SCHEMA_VERSION
        from worldloop_kernel.replay import compute_checkpoint_checksum

        state_view = self._build_state_view()
        payload = pickle.dumps(
            {
                "tick": self._tick,
                "score": self._score,
            }
        )
        # Build with placeholder checksum, then compute the real one and
        # replace. compute_checkpoint_checksum does NOT include the
        # checksum field in its hash, so this two-step is safe.
        cp_temp = Checkpoint(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            world_id=self.WORLD_ID,
            world_version=self.WORLD_VERSION,
            tick=self._tick,
            state_view=state_view,
            opaque_payload=payload,
            payload_codec=self.PAYLOAD_CODEC,
            capability_profile=self._cap,
            rng_bundle={"main": f"MT19937:{self._tick}:{self._score}"},
            checksum="sha256:placeholder",
        )
        real_checksum = compute_checkpoint_checksum(cp_temp)
        return dataclasses.replace(cp_temp, checksum=real_checksum)

    def restore(self, checkpoint):
        if self._restore_should_fail:
            raise RuntimeError("restore injected failure")
        state = pickle.loads(checkpoint.opaque_payload)
        self._tick = int(state["tick"])
        self._score = float(state["score"])

    # --- Internal helpers ---

    def _build_state_view(self, *, offset: float = 0.0):
        from worldloop_kernel import StateView

        return StateView(
            meta=make_state_meta(tick=self._tick),
            entities=make_entity_table(
                ids=("a1",),
                columns={"score": (self._score + offset,)},
            ),
            capabilities=self._cap,
            missing_mask={},
        )


# ---------------------------------------------------------------------------
# Helpers — build checkpoints / actions for tests
# ---------------------------------------------------------------------------


def _make_executed(*, agent_id: str | int = "a1", tick: int = 0, delta: float = 1.0):
    """Build a minimal valid :class:`ExecutedAction` for tests."""
    from worldloop_kernel import ExecutedAction

    return ExecutedAction(
        agent_id=agent_id,
        action_type="TICK",
        params={"delta": delta},
        executed_at_tick=tick,
        proposal_hash="sha256:proposal-test",
    )


def _make_checkpoint(
    *,
    tick: int = 0,
    payload: bytes = b"\x80\x04test-payload",
    codec: str = "pickle+v1",
    capabilities: "Any | None" = None,
    rng_bundle: "Mapping[str, str] | None" = None,
    score: float = 0.0,
) -> "Any":
    """Build a valid :class:`Checkpoint` with a real checksum."""
    from worldloop_kernel import Checkpoint, PROTOCOL_SCHEMA_VERSION
    from worldloop_kernel.replay import compute_checkpoint_checksum

    cap = capabilities or make_capability()
    state_view = make_state_view(
        capabilities=cap,
        entities=make_entity_table(
            ids=("a1",),
            columns={"score": (score,)},
        ),
        tick=tick,
    )
    cp_temp = Checkpoint(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        world_id="toy-world-v1",
        world_version="0.0.1",
        tick=tick,
        state_view=state_view,
        opaque_payload=payload,
        payload_codec=codec,
        capability_profile=cap,
        rng_bundle=rng_bundle,
        checksum="sha256:placeholder",
    )
    real_checksum = compute_checkpoint_checksum(cp_temp)
    return dataclasses.replace(cp_temp, checksum=real_checksum)


# ---------------------------------------------------------------------------
# TestCheckpointChecksum
# ---------------------------------------------------------------------------


class TestCheckpointChecksum:
    """Verify :func:`compute_checkpoint_checksum` is stable and sensitive."""

    def test_checksum_format(self):
        from worldloop_kernel import compute_checkpoint_checksum

        cp = _make_checkpoint()
        cs = compute_checkpoint_checksum(cp)
        assert cs.startswith("sha256:")
        assert len(cs) == len("sha256:") + 64

    def test_checksum_stable(self):
        from worldloop_kernel import compute_checkpoint_checksum

        cp = _make_checkpoint()
        assert compute_checkpoint_checksum(cp) == compute_checkpoint_checksum(cp)

    def test_checksum_changes_on_payload(self):
        from worldloop_kernel import compute_checkpoint_checksum

        cp_a = _make_checkpoint(payload=b"AAA")
        cp_b = _make_checkpoint(payload=b"BBB")
        assert compute_checkpoint_checksum(cp_a) != compute_checkpoint_checksum(cp_b)

    def test_checksum_changes_on_tick(self):
        from worldloop_kernel import compute_checkpoint_checksum

        cp_a = _make_checkpoint(tick=0)
        cp_b = _make_checkpoint(tick=1)
        assert compute_checkpoint_checksum(cp_a) != compute_checkpoint_checksum(cp_b)

    def test_checksum_changes_on_codec(self):
        from worldloop_kernel import compute_checkpoint_checksum

        cp_a = _make_checkpoint(codec="pickle+v1")
        cp_b = _make_checkpoint(codec="json+v1")
        assert compute_checkpoint_checksum(cp_a) != compute_checkpoint_checksum(cp_b)

    def test_checksum_changes_on_state_view(self):
        from worldloop_kernel import compute_checkpoint_checksum

        cp_a = _make_checkpoint(score=0.0)
        cp_b = _make_checkpoint(score=1.0)
        assert compute_checkpoint_checksum(cp_a) != compute_checkpoint_checksum(cp_b)

    def test_checksum_changes_on_rng_bundle(self):
        from worldloop_kernel import compute_checkpoint_checksum

        cp_a = _make_checkpoint(rng_bundle={"main": "MT19937:0"})
        cp_b = _make_checkpoint(rng_bundle={"main": "MT19937:1"})
        assert compute_checkpoint_checksum(cp_a) != compute_checkpoint_checksum(cp_b)

    def test_checksum_independent_of_stored_checksum(self):
        """compute_checkpoint_checksum must NOT include the ``checksum``
        field — otherwise we'd have a circular dependency."""
        from worldloop_kernel import compute_checkpoint_checksum

        cp = _make_checkpoint()
        cp_other = dataclasses.replace(cp, checksum="sha256:different")
        assert compute_checkpoint_checksum(cp) == compute_checkpoint_checksum(cp_other)

    def test_checksum_rng_bundle_key_order_independent(self):
        """rng_bundle is a Mapping; key iteration order must not affect
        the checksum (sorted internally)."""
        from worldloop_kernel import compute_checkpoint_checksum

        # Build two checkpoints with rng_bundles that have the same
        # content but different insertion order. We need to bypass
        # the frozen field's __post_init__ via dataclasses.replace
        # on the rng_bundle field.
        cp_base = _make_checkpoint(rng_bundle={"a": "1", "b": "2"})
        cp_swap = dataclasses.replace(cp_base, rng_bundle={"b": "2", "a": "1"})
        assert compute_checkpoint_checksum(cp_base) == compute_checkpoint_checksum(cp_swap)

    def test_checksum_no_rng_bundle_vs_empty(self):
        """rng_bundle=None is fine; just doesn't contribute bytes."""
        from worldloop_kernel import compute_checkpoint_checksum

        cp = _make_checkpoint(rng_bundle=None)
        # Should still produce a valid hash.
        cs = compute_checkpoint_checksum(cp)
        assert cs.startswith("sha256:")


# ---------------------------------------------------------------------------
# TestVerifyCheckpointRestoration
# ---------------------------------------------------------------------------


class TestVerifyCheckpointRestoration:
    """Verify :func:`verify_checkpoint_restoration` success and failure paths."""

    def test_restore_success(self):
        from worldloop_kernel import verify_checkpoint_restoration

        world = _ToyWorld()
        # Take a checkpoint at non-trivial tick.
        world._tick = 3
        world._score = 7.5
        cp = world.checkpoint()

        # Move the world away, then verify restoration.
        world._tick = 99
        world._score = -1.0
        ok, msg = verify_checkpoint_restoration(world, cp)
        assert ok is True
        assert msg == ""
        # And the world is now back at the checkpoint's state.
        assert world._tick == 3
        assert world._score == 7.5

    def test_restore_failure_state_mismatch(self):
        """If the world's observe() returns a different state after
        restore, verification must fail."""
        from worldloop_kernel import verify_checkpoint_restoration

        # observe_state_offset makes observe() return a different score.
        world = _ToyWorld(observe_state_offset=100.0)
        world._tick = 0
        world._score = 0.0
        # Build a checkpoint from a *clean* world (no offset) so the
        # stored state_view hash reflects the un-offset score.
        clean_world = _ToyWorld()
        cp = clean_world.checkpoint()

        ok, msg = verify_checkpoint_restoration(world, cp)
        assert ok is False
        assert "state hash mismatch" in msg

    def test_restore_failure_checksum_mismatch(self):
        """If the stored checksum is wrong, verification must fail."""
        from worldloop_kernel import verify_checkpoint_restoration

        world = _ToyWorld()
        cp = world.checkpoint()
        # Corrupt the checksum.
        cp_bad = dataclasses.replace(cp, checksum="sha256:deadbeef")
        ok, msg = verify_checkpoint_restoration(world, cp_bad)
        assert ok is False
        assert "checksum mismatch" in msg

    def test_restore_failure_world_restore_raises(self):
        from worldloop_kernel import verify_checkpoint_restoration

        world = _ToyWorld(restore_should_fail=True)
        cp = _make_checkpoint()
        ok, msg = verify_checkpoint_restoration(world, cp)
        assert ok is False
        assert "world.restore raised" in msg

    def test_restore_skips_checksum_when_empty(self):
        """If ``checkpoint.checksum`` is empty (non-exact-restore world),
        the checksum check is skipped but the state hash check still runs."""
        from worldloop_kernel import verify_checkpoint_restoration

        # Build a non-exact-restore capability.
        cap = make_capability(exact_restore=False, executable_deterministic_replay=False)
        world = _ToyWorld(capabilities=cap)
        world._tick = 0
        world._score = 0.0

        # Build a real checkpoint from the world (so opaque_payload is a
        # valid pickle), then clear the checksum (allowed when
        # exact_restore=False).
        cp_real = world.checkpoint()
        cp = dataclasses.replace(cp_real, checksum="")

        ok, msg = verify_checkpoint_restoration(world, cp)
        assert ok is True
        assert msg == ""


# ---------------------------------------------------------------------------
# TestReplayConsistency
# ---------------------------------------------------------------------------


class TestReplayConsistency:
    """Verify :func:`replay` produces bit-identical hashes and captures
    invariant violations without raising."""

    def test_replay_bit_identical_hashes(self):
        """Two replays from the same checkpoint with the same frozen
        actions MUST produce identical per-tick hashes."""
        from worldloop_kernel import replay

        # Run a baseline to collect per-tick hashes.
        world_baseline = _ToyWorld()
        world_baseline.reset(seed=42)
        cp_start = world_baseline.checkpoint()

        actions = [
            _make_executed(tick=0, delta=1.0),
            _make_executed(tick=1, delta=2.0),
            _make_executed(tick=2, delta=-0.5),
        ]
        # Execute the actions on the baseline to collect expected hashes.
        expected_hashes = []
        for action in actions:
            world_baseline.step(action)
            from worldloop_kernel import hash_state
            expected_hashes.append(hash_state(world_baseline.observe()))

        # Replay on a fresh world from the checkpoint.
        world_replay = _ToyWorld()
        report = replay(world_replay, cp_start, actions)

        assert report.replay_consistent is True
        assert report.restoration_ok is True
        assert report.restoration_message == ""
        assert report.invariant_violations == ()
        assert report.per_tick_hashes == tuple(expected_hashes)
        assert report.final_state_hash == expected_hashes[-1]
        assert report.n_actions == 3

    def test_replay_empty_actions(self):
        """Empty action sequence: replay just verifies restoration."""
        from worldloop_kernel import replay

        world = _ToyWorld()
        world._tick = 5
        world._score = 10.0
        cp = world.checkpoint()

        # Move world away.
        world._tick = 0
        world._score = 0.0

        report = replay(world, cp, [])
        assert report.replay_consistent is True
        assert report.restoration_ok is True
        assert report.per_tick_hashes == ()
        assert report.final_state_hash is None
        assert report.n_actions == 0

    def test_replay_capability_gating_exact_restore_false(self):
        """If ``capabilities.exact_restore=False``, replay_consistent
        MUST be False even if everything else passes."""
        from worldloop_kernel import replay

        cap = make_capability(exact_restore=False, executable_deterministic_replay=False)
        world = _ToyWorld(capabilities=cap)
        world.reset(seed=42)
        cp = world.checkpoint()

        report = replay(world, cp, [_make_executed(tick=0)])
        assert report.cap_exact_restore is False
        assert report.cap_deterministic_replay is False
        assert report.replay_consistent is False

    def test_replay_capability_gating_deterministic_replay_false(self):
        """If ``executable_deterministic_replay=False`` (and thus also
        exact_restore=False per capability rule), replay_consistent is False."""
        from worldloop_kernel import replay

        cap = make_capability(exact_restore=False, executable_deterministic_replay=False)
        world = _ToyWorld(capabilities=cap)
        world.reset(seed=42)
        cp = world.checkpoint()

        report = replay(world, cp, [])
        assert report.cap_deterministic_replay is False
        assert report.replay_consistent is False

    def test_replay_world_step_raises(self):
        """If ``world.step`` raises, ``replay`` MUST raise ``ReplayError``
        (not silently swallow)."""
        from worldloop_kernel import replay, ReplayError

        world = _ToyWorld(step_should_fail=True)
        world.reset(seed=42)
        cp = world.checkpoint()

        with pytest.raises(ReplayError, match="world.step raised"):
            replay(world, cp, [_make_executed(tick=0)])

    def test_replay_world_observe_before_step_raises(self):
        """If ``world.observe`` raises before step, ``replay`` raises."""
        from worldloop_kernel import replay, ReplayError

        world = _ToyWorld(observe_should_fail=True)
        world.reset(seed=42)
        cp = world.checkpoint()

        with pytest.raises(ReplayError, match="world.observe raised"):
            replay(world, cp, [_make_executed(tick=0)])

    def test_replay_report_has_stable_checkpoint_hash(self):
        """The report's checkpoint_hash must equal
        :func:`compute_checkpoint_checksum` of the input checkpoint."""
        from worldloop_kernel import replay, compute_checkpoint_checksum

        world = _ToyWorld()
        world.reset(seed=42)
        cp = world.checkpoint()

        report = replay(world, cp, [])
        assert report.checkpoint_hash == compute_checkpoint_checksum(cp)

    def test_replay_report_has_stable_actions_hash(self):
        """The report's actions_hash must be stable for the same action sequence."""
        from worldloop_kernel import replay

        world_a = _ToyWorld()
        world_a.reset(seed=42)
        cp_a = world_a.checkpoint()

        world_b = _ToyWorld()
        world_b.reset(seed=42)
        cp_b = world_b.checkpoint()

        actions = [_make_executed(tick=0, delta=1.0), _make_executed(tick=1, delta=2.0)]
        report_a = replay(world_a, cp_a, actions)
        report_b = replay(world_b, cp_b, actions)
        assert report_a.actions_hash == report_b.actions_hash

    def test_replay_report_actions_hash_differs_on_different_actions(self):
        from worldloop_kernel import replay

        world_a = _ToyWorld()
        world_a.reset(seed=42)
        cp_a = world_a.checkpoint()

        world_b = _ToyWorld()
        world_b.reset(seed=42)
        cp_b = world_b.checkpoint()

        report_a = replay(world_a, cp_a, [_make_executed(tick=0, delta=1.0)])
        report_b = replay(world_b, cp_b, [_make_executed(tick=0, delta=2.0)])
        assert report_a.actions_hash != report_b.actions_hash

    def test_replay_invariant_violations_recorded_not_raised(self):
        """If a transition invariant fails during replay, the violation
        is recorded in the report — NOT raised as an exception."""
        from worldloop_kernel import replay
        from worldloop_kernel.transition import TransitionRecord
        from worldloop_kernel import PROTOCOL_SCHEMA_VERSION, hash_state
        from worldloop_kernel.diff_apply import diff_state
        from worldloop_kernel import ActionReceipt, OUTCOME_OK, StateDelta

        # Build a world whose step() produces a record with a
        # deliberately wrong state_after_hash. This will fail the
        # hash_round_trip invariant.
        class _BadHashWorld(_ToyWorld):
            def step(self, action, exogenous=None):
                record = super().step(action, exogenous=exogenous)
                # Corrupt the state_after_hash.
                return dataclasses.replace(
                    record, state_after_hash="sha256:deadbeef"
                )

        world = _BadHashWorld()
        world.reset(seed=42)
        cp = world.checkpoint()

        report = replay(world, cp, [_make_executed(tick=0)])
        assert report.replay_consistent is False
        assert len(report.invariant_violations) > 0
        # hash_round_trip should be among the violations.
        assert any("hash_round_trip" in v for v in report.invariant_violations)


# ---------------------------------------------------------------------------
# TestBranchIsolation
# ---------------------------------------------------------------------------


class TestBranchIsolation:
    """Verify :func:`branch` isolates parent state and diverges correctly."""

    def test_branch_empty_alternatives(self):
        from worldloop_kernel import branch

        world = _ToyWorld()
        world.reset(seed=42)
        cp = world.checkpoint()
        results = branch(world, cp, [])
        assert results == []

    def test_branch_parent_state_preserved(self):
        """After branching, the parent world MUST be back at its
        pre-branch state."""
        from worldloop_kernel import branch, hash_state

        world = _ToyWorld()
        world.reset(seed=42)
        # Move the world to a non-trivial state.
        world._tick = 5
        world._score = 17.0
        parent_hash_before = hash_state(world.observe())

        # Fork checkpoint is at tick=0 (different from parent's tick=5).
        cp = _make_checkpoint(tick=0, score=0.0)

        # Run two branches that take different actions.
        results = branch(
            world,
            cp,
            [
                [_make_executed(tick=0, delta=1.0), _make_executed(tick=1, delta=2.0)],
                [_make_executed(tick=0, delta=-1.0)],
            ],
        )

        assert len(results) == 2
        parent_hash_after = hash_state(world.observe())
        assert parent_hash_before == parent_hash_after, (
            "parent state must be unchanged after branching"
        )

    def test_branch_diverges_at_tick(self):
        """Two branches with different actions must diverge at tick 1
        (the first tick after the fork)."""
        from worldloop_kernel import branch

        world = _ToyWorld()
        world.reset(seed=42)
        # Move world to fork point (tick=0, score=0).
        cp = world.checkpoint()

        # First, run a "parent" sequence to get parent_per_tick_hashes.
        parent_actions = [_make_executed(tick=0, delta=1.0)]
        from worldloop_kernel import hash_state
        for action in parent_actions:
            world.step(action)
        parent_hashes = [hash_state(world.observe())]

        # Restore to fork and branch.
        world.restore(cp)
        results = branch(
            world,
            cp,
            [
                [_make_executed(tick=0, delta=1.0)],  # matches parent
                [_make_executed(tick=0, delta=2.0)],  # diverges
            ],
            parent_per_tick_hashes=parent_hashes,
        )

        assert results[0].diverged_at_tick is None, (
            "branch 0 matches parent at every tick"
        )
        assert results[0].final_state_hash == parent_hashes[0]

        assert results[1].diverged_at_tick == 1, (
            "branch 1 diverges at tick 1 (first tick after fork at tick=0)"
        )
        assert results[1].final_state_hash != parent_hashes[0]

    def test_branch_multiple_alternatives_isolated(self):
        """Each branch starts from the same fork checkpoint — their
        per_tick_hashes must reflect only their own actions, not bleed
        from sibling branches."""
        from worldloop_kernel import branch

        world = _ToyWorld()
        world.reset(seed=42)
        cp = world.checkpoint()

        results = branch(
            world,
            cp,
            [
                [_make_executed(tick=0, delta=1.0), _make_executed(tick=1, delta=1.0)],
                [_make_executed(tick=0, delta=5.0), _make_executed(tick=1, delta=5.0)],
                [_make_executed(tick=0, delta=-3.0)],
            ],
        )

        assert len(results) == 3
        # Each branch's final state hash must be distinct (different deltas).
        finals = [r.final_state_hash for r in results]
        assert len(set(finals)) == 3, "branches must produce distinct final states"
        # Each branch's per_tick_hashes length matches its action count.
        assert len(results[0].per_tick_hashes) == 2
        assert len(results[1].per_tick_hashes) == 2
        assert len(results[2].per_tick_hashes) == 1

    def test_branch_error_captured_not_raised(self):
        """If a branch's ``world.step`` raises, the error is captured in
        ``BranchResult.error`` and the next branch still runs."""
        from worldloop_kernel import branch

        # We need a world whose step raises only on the second branch.
        # Simulate by injecting a step failure after the first call.
        class _FlakyWorld(_ToyWorld):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._call_count = 0

            def step(self, action, exogenous=None):
                self._call_count += 1
                if self._call_count == 2:
                    raise RuntimeError("flaky failure on second call")
                return super().step(action, exogenous=exogenous)

        world = _FlakyWorld()
        world.reset(seed=42)
        cp = world.checkpoint()

        results = branch(
            world,
            cp,
            [
                [_make_executed(tick=0, delta=1.0)],  # OK (call 1)
                [_make_executed(tick=0, delta=1.0)],  # raises (call 2)
                [_make_executed(tick=0, delta=1.0)],  # OK (call 3)
            ],
        )

        assert len(results) == 3
        assert results[0].error is None
        assert results[1].error is not None
        assert "flaky failure" in results[1].error
        assert results[1].final_state_hash is None
        assert results[2].error is None
        assert results[2].final_state_hash is not None

    def test_branch_parent_restoration_failure_raises(self):
        """If the final parent restoration fails, ``branch`` MUST raise
        ``ReplayError`` (parent pollution is a hard error)."""
        from worldloop_kernel import branch, ReplayError

        # World whose restore fails after the first call (used for the
        # final parent restoration).
        class _RestoreFailsOnSecondCall(_ToyWorld):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._restore_count = 0

            def restore(self, checkpoint):
                self._restore_count += 1
                # First restore is for branch 0 (OK). Subsequent restores
                # are for branch 1, branch 2, ..., and finally the parent.
                # We make the LAST restore (parent restoration) fail.
                # Since branch() does: restore(fork) per branch, then
                # restore(parent_saved) once at the end, we need to know
                # the total restore count. With 1 branch: 1 fork restore
                # + 1 parent restore = 2. Make the 2nd fail.
                if self._restore_count == 2:
                    raise RuntimeError("parent restoration failed")
                super().restore(checkpoint)

        world = _RestoreFailsOnSecondCall()
        world.reset(seed=42)
        cp = world.checkpoint()

        with pytest.raises(ReplayError, match="failed to restore parent state"):
            branch(world, cp, [[_make_executed(tick=0, delta=1.0)]])

    def test_branch_save_parent_failure_raises(self):
        """If ``world.checkpoint()`` fails before any branch runs,
        ``branch`` MUST raise ``ReplayError``."""
        from worldloop_kernel import branch, ReplayError

        class _CheckpointFails(_ToyWorld):
            def checkpoint(self):
                raise RuntimeError("cannot save parent")

        world = _CheckpointFails()
        world.reset(seed=42)
        cp = _make_checkpoint()

        with pytest.raises(ReplayError, match="failed to save parent state"):
            branch(world, cp, [[_make_executed(tick=0)]])

    def test_branch_fork_restore_failure_recorded(self):
        """If ``world.restore(fork_checkpoint)`` fails for a branch,
        the branch's ``restoration_ok`` is False and ``error`` is set,
        but other branches still run."""
        from worldloop_kernel import branch

        # World whose restore fails only on the first call (branch 0's
        # fork restore), then succeeds for branch 1.
        class _FirstRestoreFails(_ToyWorld):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._restore_count = 0

            def restore(self, checkpoint):
                self._restore_count += 1
                if self._restore_count == 1:
                    raise RuntimeError("first restore failed")
                super().restore(checkpoint)

        world = _FirstRestoreFails()
        world.reset(seed=42)
        # We need parent_saved to be restorable (it's the world's own
        # checkpoint, which doesn't go through the failing restore
        # path... wait, it does). Let's see — branch() does:
        #   parent_saved = world.checkpoint()  # OK, doesn't call restore
        #   for each branch: world.restore(fork)  # call 1, 2, ...
        #   finally: world.restore(parent_saved)  # last call
        # So with 2 branches: restore call 1 (branch 0 fork) fails,
        # call 2 (branch 1 fork) OK, call 3 (parent) OK.
        cp = world.checkpoint()

        results = branch(
            world,
            cp,
            [
                [_make_executed(tick=0, delta=1.0)],  # fork restore fails
                [_make_executed(tick=0, delta=1.0)],  # fork restore OK
            ],
        )

        assert len(results) == 2
        assert results[0].restoration_ok is False
        assert results[0].error is not None
        assert "world.restore(fork) raised" in results[0].error
        assert results[0].final_state_hash is None
        assert results[1].restoration_ok is True
        assert results[1].error is None
        assert results[1].final_state_hash is not None

    def test_branch_branch_ids_are_zero_indexed(self):
        from worldloop_kernel import branch

        world = _ToyWorld()
        world.reset(seed=42)
        cp = world.checkpoint()

        results = branch(
            world,
            cp,
            [
                [_make_executed(tick=0)],
                [_make_executed(tick=0)],
                [_make_executed(tick=0)],
            ],
        )
        assert [r.branch_id for r in results] == ["b0", "b1", "b2"]

    def test_branch_fork_tick_matches_checkpoint(self):
        from worldloop_kernel import branch

        world = _ToyWorld()
        world._tick = 7
        world._score = 3.0
        cp = world.checkpoint()

        results = branch(world, cp, [[_make_executed(tick=7)]])
        assert results[0].fork_tick == 7

    def test_branch_no_parent_hashes_means_no_divergence(self):
        """If parent_per_tick_hashes is None, diverged_at_tick is None
        even if the branch differs from the parent."""
        from worldloop_kernel import branch

        world = _ToyWorld()
        world.reset(seed=42)
        cp = world.checkpoint()

        results = branch(
            world,
            cp,
            [[_make_executed(tick=0, delta=42.0)]],
            parent_per_tick_hashes=None,
        )
        assert results[0].diverged_at_tick is None

    def test_branch_branch_longer_than_parent_diverges(self):
        """If a branch runs longer than the parent and matches at every
        common tick, the first extra tick is treated as divergence."""
        from worldloop_kernel import branch, hash_state

        world = _ToyWorld()
        world.reset(seed=42)
        cp = world.checkpoint()

        # Parent ran 1 tick with delta=1.0; capture its real per-tick hash.
        parent_world = _ToyWorld()
        parent_world.restore(cp)
        parent_world.step(_make_executed(tick=0, delta=1.0))
        parent_hashes = [hash_state(parent_world.observe())]

        # Branch runs 3 ticks with the SAME delta=1.0 for tick 0
        # (matches parent), then continues with extra ticks.
        results = branch(
            world,
            cp,
            [
                [
                    _make_executed(tick=0, delta=1.0),  # matches parent[0]
                    _make_executed(tick=1, delta=1.0),  # extra tick
                    _make_executed(tick=2, delta=1.0),  # extra tick
                ],
            ],
            parent_per_tick_hashes=parent_hashes,
        )
        # i=0: matches parent[0] → no divergence yet.
        # i=1: i >= len(parent_hashes)=1 → divergence at checkpoint.tick + 1 + 1 = 2.
        assert results[0].diverged_at_tick == 2


# ---------------------------------------------------------------------------
# TestReExport — K-07 symbols exported from the top-level package
# ---------------------------------------------------------------------------


class TestReExport:
    """Verify the 8 K-07 symbols are exported from ``worldloop_kernel``."""

    def test_eight_symbols_exported(self):
        import worldloop_kernel as wk

        expected = [
            "CheckpointCodec",
            "ReplayReport",
            "BranchResult",
            "ReplayError",
            "compute_checkpoint_checksum",
            "verify_checkpoint_restoration",
            "replay",
            "branch",
        ]
        for name in expected:
            assert hasattr(wk, name), f"missing top-level export: {name}"
            assert name in wk.__all__, f"missing from __all__: {name}"

    def test_replay_is_callable(self):
        import worldloop_kernel as wk

        assert callable(wk.replay)

    def test_branch_is_callable(self):
        import worldloop_kernel as wk

        assert callable(wk.branch)

    def test_compute_checkpoint_checksum_is_callable(self):
        import worldloop_kernel as wk

        assert callable(wk.compute_checkpoint_checksum)

    def test_verify_checkpoint_restoration_is_callable(self):
        import worldloop_kernel as wk

        assert callable(wk.verify_checkpoint_restoration)

    def test_replay_error_is_exception(self):
        import worldloop_kernel as wk

        assert issubclass(wk.ReplayError, Exception)

    def test_replay_report_is_frozen_dataclass(self):
        import dataclasses
        import worldloop_kernel as wk

        assert dataclasses.is_dataclass(wk.ReplayReport)
        assert getattr(wk.ReplayReport, "__dataclass_params__").frozen is True

    def test_branch_result_is_frozen_dataclass(self):
        import dataclasses
        import worldloop_kernel as wk

        assert dataclasses.is_dataclass(wk.BranchResult)
        assert getattr(wk.BranchResult, "__dataclass_params__").frozen is True

    def test_checkpoint_codec_is_protocol(self):
        """CheckpointCodec is a typing.Protocol (used for documentation
        and external codec registration, not for runtime isinstance checks
        by default)."""
        from typing import Protocol
        import worldloop_kernel as wk

        # Protocol classes have __mro__ that includes typing.Protocol
        # (or they're marked with runtime_checkable).
        assert hasattr(wk.CheckpointCodec, "_is_protocol")
        assert wk.CheckpointCodec._is_protocol is True


# ---------------------------------------------------------------------------
# TestReplayRoundTripProperty — broader round-trip property tests
# ---------------------------------------------------------------------------


class TestReplayRoundTripProperty:
    """Property-style tests: for any deterministic toy world, replay
    MUST reproduce per-tick hashes bit-identically."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 12345])
    def test_replay_reproduces_baseline(self, seed):
        from worldloop_kernel import replay, hash_state

        # Baseline run.
        world_baseline = _ToyWorld()
        world_baseline.reset(seed=seed)
        cp_start = world_baseline.checkpoint()

        actions = [
            _make_executed(tick=i, delta=float(i + 1))
            for i in range(5)
        ]
        expected_hashes = []
        for action in actions:
            world_baseline.step(action)
            expected_hashes.append(hash_state(world_baseline.observe()))

        # Replay on a fresh world.
        world_replay = _ToyWorld()
        report = replay(world_replay, cp_start, actions)

        assert report.replay_consistent is True
        assert report.per_tick_hashes == tuple(expected_hashes)

    @pytest.mark.parametrize("n_actions", [0, 1, 5, 20])
    def test_replay_action_count_matches(self, n_actions):
        from worldloop_kernel import replay

        world = _ToyWorld()
        world.reset(seed=42)
        cp = world.checkpoint()

        actions = [_make_executed(tick=i) for i in range(n_actions)]
        report = replay(world, cp, actions)
        assert report.n_actions == n_actions
        assert len(report.per_tick_hashes) == n_actions

    def test_replay_idempotent(self):
        """Replaying the same checkpoint+actions twice on fresh worlds
        MUST produce identical reports."""
        from worldloop_kernel import replay

        actions = [_make_executed(tick=0, delta=1.0), _make_executed(tick=1, delta=2.0)]

        world_a = _ToyWorld()
        world_a.reset(seed=42)
        cp_a = world_a.checkpoint()
        report_a = replay(world_a, cp_a, actions)

        world_b = _ToyWorld()
        world_b.reset(seed=42)
        cp_b = world_b.checkpoint()
        report_b = replay(world_b, cp_b, actions)

        assert report_a.checkpoint_hash == report_b.checkpoint_hash
        assert report_a.actions_hash == report_b.actions_hash
        assert report_a.per_tick_hashes == report_b.per_tick_hashes
        assert report_a.final_state_hash == report_b.final_state_hash
        assert report_a.replay_consistent == report_b.replay_consistent

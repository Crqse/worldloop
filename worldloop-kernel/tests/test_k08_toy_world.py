"""K-08 explicit tests for the ToyWorld engine and M0 Gate (d) + (e).

Verifies (per main plan §10.4 M0 Gate):
- (d) toy world 1000-step diff/apply 100% consistent: every per-tick
  ``hash_state(apply_delta(before, diff_state(before, after))) ==
  hash_state(after)``.
- (e) toy world 1000-step exact replay 100% consistent: per-tick state
  hashes are bit-identical between the original run and a frozen-action
  replay from the initial checkpoint; counterfactual branch from tick
  500 diverges only on or after tick 500, and the parent world state
  remains identical to a no-branch run.

Also verifies:
- :class:`ToyWorld` satisfies :class:`WorldProtocol` (runtime_checkable).
- Constants and capability profile are exported as documented.
- ``reset`` / ``observe`` / ``legal_actions`` / ``validate_action`` /
  ``step`` / ``checkpoint`` / ``restore`` behave per docstring.
- Energy mechanics: ``move`` net cost 0.5, ``noop`` net cost 0.0,
  energy floor at 0, position wrap-around.
- Top-level package re-exports the 7 K-08 symbols.

Per lesson L-target-a1b2-02, M0 Gate (d) + (e) require explicit
verification at the 1000-step scale, not just unit-scale tests.
Per lesson L-target-a1b2-08, transition semantics (tick / hash
relationships) follow TransitionRecord docstring strictly.
Per lesson L-target-a1b2-10, multi-branch test fixtures target the
intended branch (use real parent_world to generate matching
parent_hashes[0] before testing length-overflow paths).
"""

from __future__ import annotations

from typing import Any

import pytest

from worldloop_kernel import (
    ActionProposal,
    ExecutedAction,
    ActionSpace,
    Checkpoint,
    OUTCOME_ILLEGAL_ACTION,
    OUTCOME_OK,
    StateView,
    ToyWorld,
    TOY_WORLD_ID,
    TOY_WORLD_VERSION,
    TOY_WORLD_PAYLOAD_CODEC,
    DEFAULT_GRID_LENGTH,
    DEFAULT_INITIAL_ENERGY,
    WorldProtocol,
    apply_delta,
    branch,
    hash_state,
    make_toy_capability,
    replay,
)
from worldloop_kernel.engine import (
    MOVE_COST,
    NOOP_COST,
    PASSIVE_RECOVERY,
)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def _proposal(
    *,
    agent_id: str | int = "a1",
    action_type: str = "move",
    direction: int | None = 1,
    tick: int = 0,
) -> ActionProposal:
    """Build a minimal valid ActionProposal for tests."""
    params: dict[str, Any] = {}
    if action_type == "move" and direction is not None:
        params["direction"] = direction
    return ActionProposal(
        agent_id=agent_id,
        action_type=action_type,
        params=params,
        proposed_at_tick=tick,
        proposer="test",
    )


# ---------------------------------------------------------------------------
# TestToyWorldProtocolConformance
# ---------------------------------------------------------------------------


class TestToyWorldProtocolConformance:
    """ToyWorld implements WorldProtocol (runtime_checkable)."""

    def test_isinstance_world_protocol(self):
        world = ToyWorld()
        assert isinstance(world, WorldProtocol)

    def test_all_protocol_methods_present(self):
        world = ToyWorld()
        for method in (
            "reset",
            "observe",
            "legal_actions",
            "validate_action",
            "step",
            "checkpoint",
            "restore",
        ):
            assert callable(getattr(world, method)), f"missing {method}"

    def test_capabilities_property_returns_profile(self):
        from worldloop_kernel import CapabilityProfile

        world = ToyWorld()
        cap = world.capabilities
        assert isinstance(cap, CapabilityProfile)

    def test_capabilities_stable_across_calls(self):
        world = ToyWorld()
        cap1 = world.capabilities
        cap2 = world.capabilities
        assert cap1 is cap2

    def test_exact_restore_and_replay_flags(self):
        world = ToyWorld()
        cap = world.capabilities
        assert cap.exact_restore is True
        assert cap.executable_deterministic_replay is True


# ---------------------------------------------------------------------------
# TestToyWorldConstants
# ---------------------------------------------------------------------------


class TestToyWorldConstants:
    """Module-level constants are exported with documented values."""

    def test_world_id(self):
        assert TOY_WORLD_ID == "worldloop-toy-v1"

    def test_world_version(self):
        assert TOY_WORLD_VERSION == "0.1.0"

    def test_payload_codec(self):
        assert TOY_WORLD_PAYLOAD_CODEC == "pickle+v1"

    def test_default_grid_length(self):
        assert DEFAULT_GRID_LENGTH == 10

    def test_default_initial_energy(self):
        assert DEFAULT_INITIAL_ENERGY == 10.0

    def test_make_toy_capability_entities_only(self):
        cap = make_toy_capability()
        assert cap.entities is True
        assert cap.fields is False
        assert cap.relations is False
        assert cap.registries is False
        assert cap.population is False
        assert cap.events is False
        assert cap.exact_restore is True
        assert cap.executable_deterministic_replay is True
        assert cap.authority == "rule"
        assert cap.ground_truth is True
        assert cap.transition_mode == "deterministic"


# ---------------------------------------------------------------------------
# TestToyWorldInitAndReset
# ---------------------------------------------------------------------------


class TestToyWorldInitAndReset:
    """__init__ and reset enforce value constraints."""

    def test_default_init(self):
        world = ToyWorld()
        assert world.grid_length == DEFAULT_GRID_LENGTH
        assert world.tick == 0
        assert world.position == 0
        assert world.energy == DEFAULT_INITIAL_ENERGY
        assert world.agent_id == "a1"

    def test_custom_init(self):
        world = ToyWorld(grid_length=20, initial_energy=5.0)
        assert world.grid_length == 20
        assert world.energy == 5.0

    def test_invalid_grid_length_raises(self):
        with pytest.raises(ValueError, match="grid_length"):
            ToyWorld(grid_length=0)

    def test_invalid_initial_energy_raises(self):
        with pytest.raises(ValueError, match="initial_energy"):
            ToyWorld(initial_energy=-1.0)

    def test_reset_returns_state_view_tick_zero(self):
        world = ToyWorld()
        sv = world.reset(seed=42)
        assert isinstance(sv, StateView)
        assert sv.meta.tick == 0
        assert world.tick == 0
        assert world.position == 0
        assert world.energy == DEFAULT_INITIAL_ENERGY

    def test_reset_with_parameters_overrides(self):
        world = ToyWorld()
        world.reset(seed=1, parameters={"grid_length": 7, "initial_energy": 3.5})
        assert world.grid_length == 7
        assert world.energy == 3.5

    def test_reset_with_invalid_parameter_raises(self):
        world = ToyWorld()
        with pytest.raises(ValueError):
            world.reset(seed=1, parameters={"grid_length": -1})


# ---------------------------------------------------------------------------
# TestToyWorldObserve
# ---------------------------------------------------------------------------


class TestToyWorldObserve:
    """observe returns the current StateView without mutating state."""

    def test_observe_returns_state_view(self):
        world = ToyWorld()
        world.reset(seed=0)
        sv = world.observe()
        assert isinstance(sv, StateView)

    def test_observe_non_mutating(self):
        world = ToyWorld()
        world.reset(seed=0)
        before = world.observe()
        _ = world.observe()
        after = world.observe()
        assert hash_state(before) == hash_state(after)
        assert world.tick == 0

    def test_state_view_has_position_and_energy(self):
        world = ToyWorld()
        world.reset(seed=0)
        sv = world.observe()
        assert sv.entities is not None
        assert "position" in sv.entities.columns
        assert "energy" in sv.entities.columns
        assert sv.entities.columns["position"] == (0,)
        assert sv.entities.columns["energy"] == (DEFAULT_INITIAL_ENERGY,)


# ---------------------------------------------------------------------------
# TestToyWorldLegalActions
# ---------------------------------------------------------------------------


class TestToyWorldLegalActions:
    """legal_actions returns the documented 3-action space."""

    def test_returns_three_actions(self):
        world = ToyWorld()
        world.reset(seed=0)
        space = world.legal_actions("a1")
        assert isinstance(space, ActionSpace)
        assert len(space.legal_actions) == 3

    def test_action_types_present(self):
        world = ToyWorld()
        world.reset(seed=0)
        space = world.legal_actions("a1")
        types = {(a.action_type, a.params.get("direction")) for a in space.legal_actions}
        assert ("move", 1) in types
        assert ("move", -1) in types
        assert ("noop", None) in types

    def test_action_space_not_closed(self):
        world = ToyWorld()
        world.reset(seed=0)
        space = world.legal_actions("a1")
        assert space.is_closed is False

    def test_agent_id_propagated(self):
        world = ToyWorld()
        world.reset(seed=0)
        space = world.legal_actions("agent-x")
        assert space.agent_id == "agent-x"


# ---------------------------------------------------------------------------
# TestToyWorldValidateAction
# ---------------------------------------------------------------------------


class TestToyWorldValidateAction:
    """validate_action accepts move ±1 and noop, rejects others."""

    def test_accepts_move_plus_one(self):
        world = ToyWorld()
        world.reset(seed=0)
        executed, receipt = world.validate_action(_proposal(direction=1))
        assert executed.action_type == "move"
        assert executed.params == {"direction": 1}
        assert receipt.success is True
        assert receipt.outcome_code == OUTCOME_OK

    def test_accepts_move_minus_one(self):
        world = ToyWorld()
        world.reset(seed=0)
        executed, receipt = world.validate_action(_proposal(direction=-1))
        assert executed.params == {"direction": -1}
        assert receipt.success is True

    def test_accepts_noop(self):
        world = ToyWorld()
        world.reset(seed=0)
        executed, receipt = world.validate_action(
            _proposal(action_type="noop", direction=None)
        )
        assert executed.action_type == "noop"
        assert receipt.success is True

    def test_rejects_unknown_action_type(self):
        world = ToyWorld()
        world.reset(seed=0)
        executed, receipt = world.validate_action(
            _proposal(action_type="teleport", direction=None)
        )
        assert receipt.success is False
        assert receipt.outcome_code == OUTCOME_ILLEGAL_ACTION

    def test_rejects_invalid_direction(self):
        world = ToyWorld()
        world.reset(seed=0)
        executed, receipt = world.validate_action(_proposal(direction=2))
        assert receipt.success is False
        assert receipt.outcome_code == OUTCOME_ILLEGAL_ACTION

    def test_validate_action_does_not_mutate_state(self):
        world = ToyWorld()
        world.reset(seed=0)
        before = world.observe()
        world.validate_action(_proposal(direction=1))
        after = world.observe()
        assert hash_state(before) == hash_state(after)
        assert world.tick == 0
        assert world.position == 0


# ---------------------------------------------------------------------------
# TestToyWorldStep
# ---------------------------------------------------------------------------


class TestToyWorldStep:
    """step applies actions with documented energy mechanics."""

    def test_move_plus_one_advances_position(self):
        world = ToyWorld()
        world.reset(seed=0)
        executed, _ = world.validate_action(_proposal(direction=1))
        record = world.step(executed)
        assert world.position == 1
        assert world.tick == 1
        # Net energy delta: -MOVE_COST + PASSIVE_RECOVERY = -0.5
        assert world.energy == DEFAULT_INITIAL_ENERGY - MOVE_COST + PASSIVE_RECOVERY

    def test_move_minus_one_wraps_around(self):
        world = ToyWorld(grid_length=10)
        world.reset(seed=0)
        # position starts at 0; move -1 wraps to 9
        executed, _ = world.validate_action(_proposal(direction=-1))
        world.step(executed)
        assert world.position == 9

    def test_noop_keeps_position_zero_net_energy(self):
        world = ToyWorld()
        world.reset(seed=0)
        executed, _ = world.validate_action(
            _proposal(action_type="noop", direction=None)
        )
        world.step(executed)
        assert world.position == 0
        # Net energy delta: -NOOP_COST + PASSIVE_RECOVERY = 0.0
        assert world.energy == DEFAULT_INITIAL_ENERGY - NOOP_COST + PASSIVE_RECOVERY

    def test_tick_advances_each_step(self):
        world = ToyWorld()
        world.reset(seed=0)
        for i in range(5):
            executed, _ = world.validate_action(
                _proposal(action_type="noop", direction=None, tick=world.tick)
            )
            world.step(executed)
            assert world.tick == i + 1

    def test_energy_floored_at_zero(self):
        # Set up a world where energy will go negative without the floor.
        world = ToyWorld(initial_energy=0.0)
        world.reset(seed=0)
        # move: -1.0 + 0.5 = -0.5 → floor to 0
        executed, _ = world.validate_action(_proposal(direction=1))
        world.step(executed)
        assert world.energy == 0.0

    def test_step_returns_transition_record(self):
        from worldloop_kernel import TransitionRecord

        world = ToyWorld()
        world.reset(seed=0)
        executed, _ = world.validate_action(_proposal(direction=1))
        record = world.step(executed)
        assert isinstance(record, TransitionRecord)
        assert record.tick == 0  # tick before transition
        assert record.producer_id == TOY_WORLD_ID
        assert record.producer_version == TOY_WORLD_VERSION


# ---------------------------------------------------------------------------
# TestToyWorldCheckpointRestore
# ---------------------------------------------------------------------------


class TestToyWorldCheckpointRestore:
    """checkpoint + restore round-trip preserves state."""

    def test_checkpoint_returns_valid_checkpoint(self):
        world = ToyWorld()
        world.reset(seed=42)
        # Step a few times to mutate state.
        for _ in range(3):
            executed, _ = world.validate_action(_proposal(direction=1, tick=world.tick))
            world.step(executed)
        cp = world.checkpoint()
        assert isinstance(cp, Checkpoint)
        assert cp.checksum.startswith("sha256:")
        assert cp.checksum != "sha256:placeholder"
        assert cp.world_id == TOY_WORLD_ID
        assert cp.world_version == TOY_WORLD_VERSION
        assert cp.tick == 3
        assert cp.payload_codec == TOY_WORLD_PAYLOAD_CODEC

    def test_restore_preserves_state_hash(self):
        world = ToyWorld()
        world.reset(seed=42)
        for _ in range(5):
            executed, _ = world.validate_action(_proposal(direction=1, tick=world.tick))
            world.step(executed)
        cp = world.checkpoint()
        hash_at_cp = hash_state(world.observe())

        # Mutate further, then restore.
        for _ in range(3):
            executed, _ = world.validate_action(_proposal(direction=-1, tick=world.tick))
            world.step(executed)
        assert hash_state(world.observe()) != hash_at_cp

        world.restore(cp)
        assert hash_state(world.observe()) == hash_at_cp
        assert world.tick == 5

    def test_restore_then_step_matches_no_restore_step(self):
        """Two worlds: one restores from checkpoint then steps; the other
        steps without restoring. Both should produce identical state."""
        seed = 7
        # World A: step 5 times, checkpoint, step 2 more.
        world_a = ToyWorld()
        world_a.reset(seed=seed)
        for _ in range(5):
            executed, _ = world_a.validate_action(
                _proposal(direction=1, tick=world_a.tick)
            )
            world_a.step(executed)
        cp = world_a.checkpoint()

        # World B: restore from cp, then step with the same action.
        world_b = ToyWorld()
        world_b.reset(seed=seed)
        world_b.restore(cp)
        # Verify world_b matches world_a at this point.
        assert hash_state(world_b.observe()) == hash_state(world_a.observe())

        # Step both with the same action.
        action_a, _ = world_a.validate_action(
            _proposal(direction=1, tick=world_a.tick)
        )
        action_b, _ = world_b.validate_action(
            _proposal(direction=1, tick=world_b.tick)
        )
        world_a.step(action_a)
        world_b.step(action_b)
        assert hash_state(world_a.observe()) == hash_state(world_b.observe())

    def test_checksum_changes_after_step(self):
        world = ToyWorld()
        world.reset(seed=0)
        cp1 = world.checkpoint()
        executed, _ = world.validate_action(_proposal(direction=1))
        world.step(executed)
        cp2 = world.checkpoint()
        assert cp1.checksum != cp2.checksum
        assert cp1.tick == 0
        assert cp2.tick == 1

    def test_checkpoint_includes_internal_state(self):
        """Verify checkpoint captures grid_length and initial_energy
        (set via reset parameters), not just the defaults."""
        world = ToyWorld()
        world.reset(seed=3, parameters={"grid_length": 15, "initial_energy": 7.0})
        for _ in range(2):
            executed, _ = world.validate_action(_proposal(direction=1))
            world.step(executed)
        cp = world.checkpoint()

        # New world, different defaults, restore should override.
        world2 = ToyWorld()
        world2.reset(seed=0)  # different seed, default grid/energy
        world2.restore(cp)
        assert world2.grid_length == 15
        assert world2.position == 2  # advanced 2 steps
        # Energy after 2 moves: 7.0 - 2*0.5 = 6.0
        assert world2.energy == 6.0


# ---------------------------------------------------------------------------
# TestToyWorld1000StepDiffApply — M0 Gate (d)
# ---------------------------------------------------------------------------


class TestToyWorld1000StepDiffApply:
    """M0 Gate (d): 1000-step diff/apply round-trip 100% consistent.

    For every tick t in [0, 1000):
        hash_state(apply_delta(state_t, diff_state(state_t, state_{t+1})))
        == hash_state(state_{t+1})
    """

    NUM_STEPS = 1000

    def test_1000_step_move_plus_one_round_trip(self):
        world = ToyWorld(grid_length=50, initial_energy=1000.0)
        world.reset(seed=0)
        mismatches: list[int] = []
        for t in range(self.NUM_STEPS):
            before = world.observe()
            executed, _ = world.validate_action(
                _proposal(direction=1, tick=world.tick)
            )
            record = world.step(executed)
            after = world.observe()
            # The state_delta on the record should round-trip.
            rebuilt = apply_delta(before, record.state_delta)
            if hash_state(rebuilt) != hash_state(after):
                mismatches.append(t)
        assert mismatches == [], f"diff/apply mismatch at ticks: {mismatches[:10]}"

    def test_1000_step_mixed_actions_round_trip(self):
        world = ToyWorld(grid_length=50, initial_energy=1000.0)
        world.reset(seed=0)
        mismatches: list[int] = []
        for t in range(self.NUM_STEPS):
            before = world.observe()
            # Cycle: move+1, move-1, noop
            if t % 3 == 0:
                direction = 1
                action_type = "move"
            elif t % 3 == 1:
                direction = -1
                action_type = "move"
            else:
                direction = None
                action_type = "noop"
            executed, _ = world.validate_action(
                _proposal(action_type=action_type, direction=direction, tick=world.tick)
            )
            record = world.step(executed)
            after = world.observe()
            rebuilt = apply_delta(before, record.state_delta)
            if hash_state(rebuilt) != hash_state(after):
                mismatches.append(t)
        assert mismatches == [], f"diff/apply mismatch at ticks: {mismatches[:10]}"

    def test_1000_step_energy_floor_round_trip(self):
        """Energy starts at 0; every move would go negative without the
        floor. diff/apply must still round-trip."""
        world = ToyWorld(grid_length=10, initial_energy=0.0)
        world.reset(seed=0)
        mismatches: list[int] = []
        for t in range(self.NUM_STEPS):
            before = world.observe()
            executed, _ = world.validate_action(
                _proposal(direction=1, tick=world.tick)
            )
            record = world.step(executed)
            after = world.observe()
            rebuilt = apply_delta(before, record.state_delta)
            if hash_state(rebuilt) != hash_state(after):
                mismatches.append(t)
        assert mismatches == [], f"diff/apply mismatch at ticks: {mismatches[:10]}"

    def test_1000_step_state_after_hash_matches_observe(self):
        """TransitionRecord.state_after_hash == hash_state(observe())
        after step. This is the kernel's bit-identity guarantee."""
        world = ToyWorld(grid_length=50, initial_energy=1000.0)
        world.reset(seed=0)
        mismatches: list[int] = []
        for t in range(self.NUM_STEPS):
            executed, _ = world.validate_action(
                _proposal(direction=1, tick=world.tick)
            )
            record = world.step(executed)
            after = world.observe()
            if record.state_after_hash != hash_state(after):
                mismatches.append(t)
        assert mismatches == [], f"state_after_hash mismatch at ticks: {mismatches[:10]}"


# ---------------------------------------------------------------------------
# TestToyWorld1000StepReplay — M0 Gate (e)
# ---------------------------------------------------------------------------


class TestToyWorld1000StepReplay:
    """M0 Gate (e): 1000-step exact replay 100% consistent.

    Captures (checkpoint, executed_actions) from a 1000-step run, then
    replays them from the initial checkpoint and asserts:
    - replay_consistent is True
    - per-tick state hashes are bit-identical
    """

    NUM_STEPS = 1000

    def _run_and_capture(self, *, seed: int, grid_length: int = 50,
                         initial_energy: float = 1000.0,
                         action_pattern: str = "move_plus"):
        """Run the world for NUM_STEPS, capture (checkpoint, actions,
        per_tick_hashes). Returns (checkpoint, actions, per_tick_hashes).

        ``per_tick_hashes[i]`` is the state hash AFTER step i (i.e., the
        state at tick ``i+1`` since the world starts at tick 0). This
        matches ``ReplayReport.per_tick_hashes`` semantics: one hash per
        executed action, in order.
        """
        world = ToyWorld(grid_length=grid_length, initial_energy=initial_energy)
        world.reset(seed=seed)
        checkpoint = world.checkpoint()
        actions: list[ExecutedAction] = []
        per_tick_hashes: list[str] = []
        for t in range(self.NUM_STEPS):
            if action_pattern == "move_plus":
                direction = 1
                action_type = "move"
            elif action_pattern == "mixed":
                cycle = t % 3
                if cycle == 0:
                    direction, action_type = 1, "move"
                elif cycle == 1:
                    direction, action_type = -1, "move"
                else:
                    direction, action_type = None, "noop"
            else:
                raise ValueError(f"unknown pattern: {action_pattern}")
            executed, _ = world.validate_action(
                _proposal(action_type=action_type, direction=direction, tick=world.tick)
            )
            actions.append(executed)
            world.step(executed)
            per_tick_hashes.append(hash_state(world.observe()))
        return checkpoint, actions, per_tick_hashes

    def test_1000_step_move_plus_replay_consistent(self):
        checkpoint, actions, original_hashes = self._run_and_capture(
            seed=0, action_pattern="move_plus"
        )
        # Replay on a fresh world.
        replay_world = ToyWorld(grid_length=50, initial_energy=1000.0)
        report = replay(replay_world, checkpoint, actions)
        assert report.replay_consistent is True, (
            f"replay inconsistent: restoration_ok={report.restoration_ok} "
            f"invariant_violations={report.invariant_violations}"
        )
        assert list(report.per_tick_hashes) == original_hashes, (
            "per-tick hash divergence between original and replay"
        )

    def test_1000_step_mixed_actions_replay_consistent(self):
        checkpoint, actions, original_hashes = self._run_and_capture(
            seed=42, action_pattern="mixed"
        )
        replay_world = ToyWorld(grid_length=50, initial_energy=1000.0)
        report = replay(replay_world, checkpoint, actions)
        assert report.replay_consistent is True
        assert list(report.per_tick_hashes) == original_hashes

    def test_1000_step_replay_bit_identical_across_seeds(self):
        """Replay consistency must hold across multiple seeds."""
        for seed in (0, 1, 7, 42, 123):
            checkpoint, actions, original_hashes = self._run_and_capture(
                seed=seed, action_pattern="move_plus"
            )
            replay_world = ToyWorld(grid_length=50, initial_energy=1000.0)
            report = replay(replay_world, checkpoint, actions)
            assert report.replay_consistent is True, f"seed={seed} inconsistent"
            assert list(report.per_tick_hashes) == original_hashes, (
                f"seed={seed} hash mismatch"
            )

    def test_replay_invariant_violations_empty(self):
        """A clean replay must produce zero invariant violations."""
        checkpoint, actions, _ = self._run_and_capture(seed=0, action_pattern="move_plus")
        replay_world = ToyWorld(grid_length=50, initial_energy=1000.0)
        report = replay(replay_world, checkpoint, actions)
        # invariant_violations is a tuple; assert emptiness via truthiness.
        assert not report.invariant_violations, (
            f"expected no invariant violations, got: {report.invariant_violations}"
        )


# ---------------------------------------------------------------------------
# TestToyWorldCounterfactualBranch
# ---------------------------------------------------------------------------


class TestToyWorldCounterfactualBranch:
    """Counterfactual branch diverges only on or after the branch tick,
    and the parent world state remains identical to a no-branch run."""

    def test_branch_diverges_at_branch_tick(self):
        """Run parent for 500 steps, branch at tick 500, run branch for
        10 more steps with a different action. Divergence should be at
        tick 501 (checkpoint.tick + 1 + 0 = 501)."""
        from worldloop_kernel import branch

        parent = ToyWorld(grid_length=50, initial_energy=1000.0)
        parent.reset(seed=0)
        # Run 500 steps of move+1 to advance to tick 500.
        for _ in range(500):
            executed, _ = parent.validate_action(
                _proposal(direction=1, tick=parent.tick)
            )
            parent.step(executed)
        checkpoint = parent.checkpoint()
        assert checkpoint.tick == 500

        # Parent continues with move+1 for 10 more steps; capture
        # per-tick hashes (one per step, after each step).
        parent_per_tick_hashes: list[str] = []
        for _ in range(10):
            executed, _ = parent.validate_action(
                _proposal(direction=1, tick=parent.tick)
            )
            parent.step(executed)
            parent_per_tick_hashes.append(hash_state(parent.observe()))

        # Restore parent to fork checkpoint before branching.
        parent.restore(checkpoint)

        # Build branch actions on a separate world that mirrors the
        # branch state (also at tick 500). Branch uses move-1 instead
        # of move+1, so the first branch step should diverge.
        branch_seed = ToyWorld(grid_length=50, initial_energy=1000.0)
        branch_seed.reset(seed=0)
        for _ in range(500):
            executed, _ = branch_seed.validate_action(
                _proposal(direction=1, tick=branch_seed.tick)
            )
            branch_seed.step(executed)
        branch_actions: list[ExecutedAction] = []
        for _ in range(10):
            executed, _ = branch_seed.validate_action(
                _proposal(direction=-1, tick=branch_seed.tick)
            )
            branch_actions.append(executed)

        results = branch(
            parent,
            checkpoint,
            [branch_actions],
            parent_per_tick_hashes=parent_per_tick_hashes,
        )
        assert len(results) == 1
        b0 = results[0]
        assert b0.diverged_at_tick == 501, (
            f"expected divergence at tick 501, got {b0.diverged_at_tick}"
        )
        assert b0.error is None

    def test_branch_parent_state_preserved(self):
        """After branch() returns, the parent world's state MUST be
        restored to its pre-branch state. We verify by comparing the
        parent's state hash before and after branch()."""
        from worldloop_kernel import branch

        parent = ToyWorld(grid_length=50, initial_energy=1000.0)
        parent.reset(seed=0)
        for _ in range(500):
            executed, _ = parent.validate_action(
                _proposal(direction=1, tick=parent.tick)
            )
            parent.step(executed)
        checkpoint = parent.checkpoint()

        # Restore parent to checkpoint, capture pre-branch state.
        parent.restore(checkpoint)
        pre_branch_hash = hash_state(parent.observe())

        # Build branch actions on a separate seed world.
        branch_seed = ToyWorld(grid_length=50, initial_energy=1000.0)
        branch_seed.reset(seed=0)
        for _ in range(500):
            executed, _ = branch_seed.validate_action(
                _proposal(direction=1, tick=branch_seed.tick)
            )
            branch_seed.step(executed)
        branch_actions: list[ExecutedAction] = []
        for _ in range(10):
            executed, _ = branch_seed.validate_action(
                _proposal(direction=-1, tick=branch_seed.tick)
            )
            branch_actions.append(executed)

        # Run branch (no parent_per_tick_hashes — we only care about
        # parent state preservation here).
        results = branch(parent, checkpoint, [branch_actions])
        assert len(results) == 1

        # Parent state after branch must equal parent state before branch.
        post_branch_hash = hash_state(parent.observe())
        assert pre_branch_hash == post_branch_hash, (
            "parent state changed after branch() — isolation broken"
        )
        # Parent should be back at the checkpoint's tick.
        assert parent.tick == 500

    def test_branch_with_identical_actions_no_divergence(self):
        """If branch uses the same actions as parent, divergence should
        be None (never diverges)."""
        from worldloop_kernel import branch

        parent = ToyWorld(grid_length=50, initial_energy=1000.0)
        parent.reset(seed=0)
        for _ in range(100):
            executed, _ = parent.validate_action(
                _proposal(direction=1, tick=parent.tick)
            )
            parent.step(executed)
        checkpoint = parent.checkpoint()
        assert checkpoint.tick == 100

        # Capture 10 parent actions + per-tick hashes.
        parent_actions: list[ExecutedAction] = []
        parent_per_tick_hashes: list[str] = []
        for _ in range(10):
            executed, _ = parent.validate_action(
                _proposal(direction=1, tick=parent.tick)
            )
            parent_actions.append(executed)
            parent.step(executed)
            parent_per_tick_hashes.append(hash_state(parent.observe()))

        # Restore to fork before branching.
        parent.restore(checkpoint)

        # Branch with identical actions.
        results = branch(
            parent,
            checkpoint,
            [parent_actions],
            parent_per_tick_hashes=parent_per_tick_hashes,
        )
        assert len(results) == 1
        b0 = results[0]
        assert b0.diverged_at_tick is None, (
            f"identical actions should not diverge, got {b0.diverged_at_tick}"
        )
        # And the branch's per-tick hashes should match the parent's.
        assert list(b0.per_tick_hashes) == parent_per_tick_hashes


# ---------------------------------------------------------------------------
# TestReExport
# ---------------------------------------------------------------------------


class TestReExport:
    """K-08 symbols are re-exported from the top-level package."""

    def test_import_toy_world_from_top(self):
        from worldloop_kernel import ToyWorld as TopToyWorld
        assert TopToyWorld is ToyWorld

    def test_import_make_toy_capability_from_top(self):
        from worldloop_kernel import make_toy_capability as top_fn
        cap = top_fn()
        assert cap.entities is True

    def test_constants_re_exported(self):
        from worldloop_kernel import (
            TOY_WORLD_ID as top_id,
            TOY_WORLD_VERSION as top_ver,
            TOY_WORLD_PAYLOAD_CODEC as top_codec,
            DEFAULT_GRID_LENGTH as top_grid,
            DEFAULT_INITIAL_ENERGY as top_energy,
        )
        assert top_id == TOY_WORLD_ID
        assert top_ver == TOY_WORLD_VERSION
        assert top_codec == TOY_WORLD_PAYLOAD_CODEC
        assert top_grid == DEFAULT_GRID_LENGTH
        assert top_energy == DEFAULT_INITIAL_ENERGY

    def test_all_k08_symbols_in_all(self):
        import worldloop_kernel
        for sym in (
            "ToyWorld",
            "TOY_WORLD_ID",
            "TOY_WORLD_VERSION",
            "TOY_WORLD_PAYLOAD_CODEC",
            "DEFAULT_GRID_LENGTH",
            "DEFAULT_INITIAL_ENERGY",
            "make_toy_capability",
        ):
            assert sym in worldloop_kernel.__all__, f"missing {sym} in __all__"


# ---------------------------------------------------------------------------
# TestNoV1Imports (per lesson L-target-a1b2-05)
# ---------------------------------------------------------------------------


class TestNoV1Imports:
    """engine.py must not import v1 five-layer code or third-party deps."""

    def test_engine_no_v1_imports(self):
        import sys
        # Import engine in a fresh world (it's already imported at top).
        from worldloop_kernel import engine  # noqa: F401
        v1_markers = [
            "current.worldloop",
            "core.L1_",
            "core.L2_",
            "core.L3_",
            "core.L4_",
            "core.L5_",
        ]
        for mod_name in list(sys.modules.keys()):
            for marker in v1_markers:
                if marker in mod_name:
                    pytest.fail(f"v1 import leaked: {mod_name}")

    def test_engine_no_third_party_deps(self):
        import sys
        third_party_markers = ("numpy", "torch", "pandas", "scipy", "sklearn")
        for name in third_party_markers:
            # Allow these to be in sys.modules if they were imported by
            # pytest itself — we only care that engine.py didn't import
            # them. Check by inspecting engine module's direct imports.
            pass
        # Inspect engine's source for direct imports.
        import inspect
        from worldloop_kernel import engine
        src = inspect.getsource(engine)
        for forbidden in ("import numpy", "import torch", "import pandas",
                          "from numpy", "from torch", "from pandas"):
            assert forbidden not in src, f"engine.py contains forbidden import: {forbidden}"

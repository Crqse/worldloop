"""Q7 Gate tests for S-10 Counterfactual Branch Scheduler.

Verifies that :class:`KernelBranchScheduler` produces REAL focal-action
variations (different ``action_type``) rather than replaying the same
baseline actions. This unlocks the Q7 counterfactual gate: when
``branch_count > 0`` and ``held_fixed`` is non-empty, Q7 passes.

Test coverage:
- :class:`NoOpBranchScheduler` always returns ``[]`` (smoke default).
- :class:`KernelBranchScheduler` does not branch at tick 0 (no baseline).
- :class:`KernelBranchScheduler` produces a real alternative action_type
  on :class:`ToyWorld` (baseline ``move`` → alternative ``noop``), with
  non-empty ``held_fixed`` and a branch_summary reporting ``branch_count > 0``.
- :class:`KernelBranchScheduler` skips branching when the focal agent
  has only one distinct legal ``action_type`` (no variation possible).
"""

from __future__ import annotations

import pickle
from typing import Any, Mapping

import pytest

from worldloop_kernel import (
    ActionProposal,
    ActionReceipt,
    ActionSpace,
    CapabilityProfile,
    Checkpoint,
    ExecutedAction,
    LegalAction,
    OUTCOME_OK,
    PROTOCOL_SCHEMA_VERSION,
    StateMeta,
    StateView,
    ToyWorld,
    EntityTable,
)
from worldloop_kernel.canonical import hash_state

from worldloop_data.config import CounterfactualConfig
from worldloop_data.counterfactual import (
    KernelBranchScheduler,
    NoOpBranchScheduler,
)


# ---------------------------------------------------------------------------
# Test 1: NoOpBranchScheduler always returns []
# ---------------------------------------------------------------------------


def test_noop_scheduler_returns_empty():
    """NoOpBranchScheduler.schedule_branches always returns [].

    This is the smoke default — branching disabled. Q7 is skipped when
    branch_count == 0.
    """
    scheduler = NoOpBranchScheduler()

    world = ToyWorld()
    world.reset(seed=42)
    cp = world.checkpoint()

    baseline = ExecutedAction(
        agent_id="a1",
        action_type="move",
        params={"direction": 1},
        executed_at_tick=2,
        proposal_hash="sha256:stub",
    )

    specs = scheduler.schedule_branches(
        checkpoint=cp,
        baseline_actions=[baseline],
        world=world,
        tick=2,
    )

    assert specs == []
    summary = scheduler.branch_summary()
    assert summary["branch_count"] == 0
    assert summary["mode"] == "noop"


# ---------------------------------------------------------------------------
# Test 2: KernelBranchScheduler does not branch at tick 0
# ---------------------------------------------------------------------------


def test_kernel_scheduler_no_branch_at_tick_0():
    """tick=0 returns empty specs.

    At tick 0 the world has just been reset; there is no prior baseline
    to contrast against. The scheduler preserves this rule even when
    branch_every_ticks=1 (which would otherwise fire on every tick).
    """
    scheduler = KernelBranchScheduler(
        config=CounterfactualConfig(
            branch_every_ticks=1,
            branches_per_checkpoint=1,
        )
    )

    world = ToyWorld()
    world.reset(seed=42)
    cp = world.checkpoint()

    baseline = ExecutedAction(
        agent_id="a1",
        action_type="move",
        params={"direction": 1},
        executed_at_tick=0,
        proposal_hash="sha256:stub",
    )

    specs = scheduler.schedule_branches(
        checkpoint=cp,
        baseline_actions=[baseline],
        world=world,
        tick=0,
    )

    assert specs == []


# ---------------------------------------------------------------------------
# Test 3: KernelBranchScheduler produces a real focal-action variation
# ---------------------------------------------------------------------------


def test_kernel_scheduler_real_focal_action_variation():
    """KernelBranchScheduler produces a branch whose alternative action
    has a DIFFERENT ``action_type`` from the baseline.

    On :class:`ToyWorld`, the focal agent ``a1`` has two distinct legal
    action_types: ``move`` and ``noop``. With baseline ``move``, the
    scheduler must produce an alternative ``noop`` branch. The branch
    must record non-empty ``held_fixed`` (Q7 requirement), and after
    execution ``branch_summary().branch_count`` must be > 0.
    """
    scheduler = KernelBranchScheduler(
        config=CounterfactualConfig(
            branch_every_ticks=2,
            branches_per_checkpoint=1,
        )
    )

    world = ToyWorld()
    world.reset(seed=42)
    cp = world.checkpoint()

    # Build a real baseline ExecutedAction via validate_action so the
    # proposal_hash is well-formed.
    baseline_proposal = ActionProposal(
        agent_id="a1",
        action_type="move",
        params={"direction": 1},
        proposed_at_tick=2,
        proposer="test",
    )
    baseline_executed, _ = world.validate_action(baseline_proposal)

    specs = scheduler.schedule_branches(
        checkpoint=cp,
        baseline_actions=[baseline_executed],
        world=world,
        tick=2,
    )

    assert len(specs) >= 1, (
        f"expected >=1 branch spec on ToyWorld, got {len(specs)}"
    )
    spec = specs[0]

    # The alternative action MUST have a different action_type.
    alt_action = spec.alternative_actions[0]
    assert alt_action.action_type != baseline_executed.action_type, (
        f"alternative action_type {alt_action.action_type!r} must differ "
        f"from baseline {baseline_executed.action_type!r}"
    )
    # On ToyWorld, the only alternative to "move" is "noop".
    assert alt_action.action_type == "noop"
    assert alt_action.agent_id == "a1"

    # rationale records the focal-action variation.
    assert "focal-action variation" in spec.rationale
    assert "move" in spec.rationale
    assert "noop" in spec.rationale
    assert "a1" in spec.rationale

    # held_fixed MUST be non-empty (Q7 gate requirement).
    assert spec.held_fixed, "held_fixed must be non-empty for Q7"
    assert "world_state" in spec.held_fixed
    assert "rng_state" in spec.held_fixed
    assert "other_agents" in spec.held_fixed

    # fork_tick and branch_id are well-formed.
    assert spec.fork_tick == 2
    assert spec.branch_id == "b0"

    # Execute the branches via the kernel.branch primitive and verify
    # branch_summary reports branch_count > 0.
    results = scheduler.execute_branches(
        world=world,
        checkpoint=cp,
        specs=specs,
    )
    assert len(results) >= 1, (
        f"expected >=1 branch result, got {len(results)}"
    )

    summary = scheduler.branch_summary()
    assert summary["branch_count"] >= 1, summary
    assert summary["mode"] == "kernel_branch"
    assert summary["held_fixed"], summary["held_fixed"]

    # Every executed branch must have restoration_ok=True (parent world
    # state must NOT be polluted by branching).
    for br in summary["branches"]:
        assert br["restoration_ok"] is True, br


# ---------------------------------------------------------------------------
# Test 4: skip branching when only one distinct legal action_type
# ---------------------------------------------------------------------------


def test_kernel_scheduler_skip_when_only_one_legal_action():
    """When the focal agent has only one distinct legal ``action_type``,
    no branch is produced — the scheduler cannot vary the focal factor.

    Uses a stub world (:class:`_SingleActionStubWorld`) whose
    ``legal_actions`` returns a single ``LegalAction`` with
    ``action_type="only_action"``. Since the baseline action_type
    matches the only available type, there is no alternative, and
    ``schedule_branches`` must return ``[]``.
    """
    scheduler = KernelBranchScheduler(
        config=CounterfactualConfig(
            branch_every_ticks=1,
            branches_per_checkpoint=3,
        )
    )

    world = _SingleActionStubWorld()
    world.reset(seed=42)
    cp = world.checkpoint()

    # Baseline action_type matches the only available action_type.
    baseline_proposal = ActionProposal(
        agent_id="a1",
        action_type="only_action",
        params={},
        proposed_at_tick=2,
        proposer="test",
    )
    baseline_executed, _ = world.validate_action(baseline_proposal)

    specs = scheduler.schedule_branches(
        checkpoint=cp,
        baseline_actions=[baseline_executed],
        world=world,
        tick=2,
    )

    assert specs == [], (
        f"expected no branches when only 1 legal action_type, "
        f"got {len(specs)} specs: {specs}"
    )


# ---------------------------------------------------------------------------
# Stub world for the single-legal-action edge case
# ---------------------------------------------------------------------------


class _SingleActionStubWorld:
    """Minimal stub world exposing exactly one legal ``action_type``.

    Implements just enough of :class:`WorldProtocol` for
    :meth:`KernelBranchScheduler.schedule_branches` to query
    ``legal_actions`` and ``validate_action``. Used only by
    :func:`test_kernel_scheduler_skip_when_only_one_legal_action`.
    """

    def __init__(self) -> None:
        self._seed = 0
        self._cap = CapabilityProfile(
            fields=False,
            entities=True,
            relations=False,
            registries=False,
            population=False,
            events=False,
            exact_restore=True,
            executable_deterministic_replay=True,
            authority="rule",
            ground_truth=True,
            transition_mode="deterministic",
        )

    @property
    def capabilities(self) -> CapabilityProfile:
        return self._cap

    def reset(self, seed: int, parameters: Mapping[str, Any] | None = None) -> StateView:
        self._seed = int(seed)
        return self._build_state_view()

    def observe(self) -> StateView:
        return self._build_state_view()

    def legal_actions(self, agent_id: str | int, state: StateView | None = None) -> ActionSpace:
        return ActionSpace(
            agent_id=agent_id,
            legal_actions=(
                LegalAction(
                    action_type="only_action",
                    params={},
                    description="the only legal action_type on this stub world",
                ),
            ),
            is_closed=True,
        )

    def validate_action(self, proposal: ActionProposal) -> tuple[ExecutedAction, ActionReceipt]:
        executed = ExecutedAction(
            agent_id=proposal.agent_id,
            action_type=proposal.action_type,
            params=dict(proposal.params),
            executed_at_tick=proposal.proposed_at_tick,
            proposal_hash=hash_state(proposal),
        )
        receipt = ActionReceipt(
            executed_action_hash=hash_state(executed),
            outcome_code=OUTCOME_OK,
            success=True,
            energy_delta=0.0,
        )
        return executed, receipt

    def checkpoint(self) -> Checkpoint:
        state_view = self._build_state_view()
        payload = pickle.dumps({"seed": self._seed})
        cp = Checkpoint(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            world_id="single-action-stub",
            world_version="0.1.0",
            tick=0,
            state_view=state_view,
            opaque_payload=payload,
            payload_codec="pickle+v1",
            capability_profile=self._cap,
            rng_bundle={"main": f"seed:{self._seed}"},
            checksum="sha256:placeholder",
        )
        return cp

    def restore(self, checkpoint: Checkpoint) -> None:
        payload = pickle.loads(checkpoint.opaque_payload)
        self._seed = int(payload.get("seed", 0))

    def _build_state_view(self) -> StateView:
        return StateView(
            meta=StateMeta(
                scenario_id="single-action-stub-scenario",
                run_id=f"single-action-stub-run-{self._seed}",
                tick=0,
                config_hash="sha256:stub",
                rng_state_ref=f"seed:{self._seed}",
            ),
            entities=EntityTable(
                schema_id="single-action-stub:entities:v1",
                ids=("a1",),
                columns={"energy": (10.0,)},
            ),
            capabilities=self._cap,
            missing_mask={},
        )

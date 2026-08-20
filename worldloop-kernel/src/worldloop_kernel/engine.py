"""Toy world engine for M0 Gate validation (K-08).

A minimal :class:`WorldProtocol` implementation used to validate the
kernel's diff/apply, replay, and branch capabilities without depending
on the v1 five-layer Native world. Used by M0 Gate §10.4 (toy world
1000 step diff/apply 100% consistent + 100% replay consistent).

Scope (lands in K-08):
- Deterministic 1D grid world with a single agent.
- Two action types: ``move`` (cardinal direction on a 1D grid) and
  ``noop``.
- Energy bookkeeping: each ``move`` costs 1.0 unit; ``noop`` costs 0.5;
  passive recovery +0.5 per tick; energy floor at 0.
- StateView exposes ``entities`` only (``position`` and ``energy``
  columns); all other slots are None. CapabilityProfile reflects this.
- ``exact_restore=True`` and ``executable_deterministic_replay=True``.
- RNG seeded via ``reset(seed)``; the toy world is fully deterministic
  given (seed, action sequence) — there is no exogenous stochasticity.
  The seed is recorded for traceability but does not drive any random
  draw in the current implementation. This keeps the M0 Gate validation
  focused on the kernel's diff/apply/replay machinery rather than on
  RNG-state threading.

The toy world is for kernel self-validation only. It is NOT a research
environment and NOT a benchmark. Any real research uses Native adapter
(M1) or external adapters (M2).

Design rules (per main plan §10.3 C2 test set):
- 1000-step diff/apply round-trip MUST be 100% consistent.
- 1000-step exact replay MUST be 100% consistent (bit-identical state
  hashes per tick).
- Counterfactual branch from tick 500 with alternative action MUST
  diverge only on or after tick 500; parent world state MUST remain
  identical to a no-branch run.

Provenance: extracted from the K-07 ``_ToyWorld`` test helper (which
remains in ``tests/test_k07_replay_branch.py`` for K-07 negative-path
testing with failure-injection hooks); K-08 promotes a clean,
public, no-hooks version to ``worldloop_kernel.engine`` for use by
M0 Gate validation tests and external consumers.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any, Mapping

from worldloop_kernel.action import (
    ActionProposal,
    ExecutedAction,
    ExogenousInput,
    ActionReceipt,
    OUTCOME_OK,
    OUTCOME_ILLEGAL_ACTION,
)
from worldloop_kernel.capability import CapabilityProfile
from worldloop_kernel.canonical import hash_state
from worldloop_kernel.diff_apply import diff_state
from worldloop_kernel.protocol import ActionSpace, LegalAction, WorldProtocol
from worldloop_kernel.replay import compute_checkpoint_checksum
from worldloop_kernel.state import (
    EntityTable,
    StateMeta,
    StateView,
)
from worldloop_kernel.transition import (
    Checkpoint,
    PROTOCOL_SCHEMA_VERSION,
    TransitionRecord,
)

__all__ = [
    "ToyWorld",
    "TOY_WORLD_ID",
    "TOY_WORLD_VERSION",
    "TOY_WORLD_PAYLOAD_CODEC",
    "DEFAULT_GRID_LENGTH",
    "DEFAULT_INITIAL_ENERGY",
    "make_toy_capability",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


TOY_WORLD_ID = "worldloop-toy-v1"
TOY_WORLD_VERSION = "0.1.0"
TOY_WORLD_PAYLOAD_CODEC = "pickle+v1"

#: Default grid length for the 1D toy world.
DEFAULT_GRID_LENGTH = 10
#: Default starting energy for the toy agent.
DEFAULT_INITIAL_ENERGY = 10.0
#: Energy cost of a `move` action (before passive recovery).
MOVE_COST = 1.0
#: Energy cost of a `noop` action (before passive recovery).
NOOP_COST = 0.5
#: Passive energy recovery per tick (added after the action cost).
PASSIVE_RECOVERY = 0.5


def make_toy_capability() -> CapabilityProfile:
    """Return the canonical :class:`CapabilityProfile` for the toy world.

    The toy world exposes ``entities`` only. All other capability slots
    are False. ``exact_restore=True`` and
    ``executable_deterministic_replay=True`` so the kernel can validate
    M0 Gate (d) and (e) on this world.
    """
    return CapabilityProfile(
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


# ---------------------------------------------------------------------------
# ToyWorld
# ---------------------------------------------------------------------------


class ToyWorld:
    """Deterministic 1D-grid single-agent toy world.

    The world owns one agent on a 1D grid of length ``grid_length``.
    The agent has a ``position`` (integer in ``[0, grid_length)``) and
    ``energy`` (float, floor at 0). Each step applies one action and
    advances the tick by 1.

    Action types:
        - ``move``: params ``{"direction": +1 | -1}``. Moves the agent
          one cell in the given direction (wrap-around). Energy cost
          ``MOVE_COST`` (1.0).
        - ``noop``: no movement. Energy cost ``NOOP_COST`` (0.5).

    After the action cost is deducted, ``PASSIVE_RECOVERY`` (0.5) is
    added back. Net cost: ``move`` = 0.5, ``noop`` = 0.0. Energy is
    floored at 0 (never negative).

    The world is fully deterministic given (seed, action sequence).
    ``reset(seed)`` records the seed but does not drive any random draw
    in the current implementation — there is no exogenous stochasticity.
    """

    def __init__(
        self,
        *,
        grid_length: int = DEFAULT_GRID_LENGTH,
        initial_energy: float = DEFAULT_INITIAL_ENERGY,
        capabilities: CapabilityProfile | None = None,
    ) -> None:
        if grid_length < 1:
            raise ValueError(f"grid_length must be >= 1, got {grid_length}")
        if initial_energy < 0:
            raise ValueError(
                f"initial_energy must be >= 0, got {initial_energy}"
            )
        self._grid_length = int(grid_length)
        self._initial_energy = float(initial_energy)
        self._cap = capabilities or make_toy_capability()
        # Mutable internal state.
        self._tick: int = 0
        self._position: int = 0
        self._energy: float = float(initial_energy)
        self._seed: int = 0
        self._agent_id: str = "a1"

    # --- WorldProtocol properties ---

    @property
    def capabilities(self) -> CapabilityProfile:
        return self._cap

    # --- WorldProtocol methods ---

    def reset(
        self,
        seed: int,
        parameters: Mapping[str, Any] | None = None,
    ) -> StateView:
        """Reset the world to its initial state.

        Parameters
        ----------
        seed:
            RNG seed (recorded but not used for randomness — the toy
            world is fully deterministic given the action sequence).
        parameters:
            Optional overrides for ``grid_length`` and
            ``initial_energy``. Changes to ``grid_length`` only take
            effect on ``reset``; subsequent ``step`` calls use the new
            length.
        """
        self._seed = int(seed)
        if parameters:
            if "grid_length" in parameters:
                self._grid_length = int(parameters["grid_length"])
                if self._grid_length < 1:
                    raise ValueError(
                        f"grid_length must be >= 1, got {self._grid_length}"
                    )
            if "initial_energy" in parameters:
                self._initial_energy = float(parameters["initial_energy"])
                if self._initial_energy < 0:
                    raise ValueError(
                        f"initial_energy must be >= 0, got {self._initial_energy}"
                    )
            if "agent_id" in parameters:
                self._agent_id = str(parameters["agent_id"])
        self._tick = 0
        self._position = 0
        self._energy = float(self._initial_energy)
        return self._build_state_view()

    def observe(self) -> StateView:
        return self._build_state_view()

    def legal_actions(
        self,
        agent_id: str | int,
        state: StateView | None = None,
    ) -> ActionSpace:
        return ActionSpace(
            agent_id=agent_id,
            legal_actions=(
                LegalAction(
                    action_type="move",
                    params={"direction": 1},
                    description="move +1 cell (wrap-around)",
                ),
                LegalAction(
                    action_type="move",
                    params={"direction": -1},
                    description="move -1 cell (wrap-around)",
                ),
                LegalAction(
                    action_type="noop",
                    params={},
                    description="no operation (passive recovery only)",
                ),
            ),
            is_closed=False,
        )

    def validate_action(
        self,
        proposal: ActionProposal,
    ) -> tuple[ExecutedAction, ActionReceipt]:
        """Validate a proposal and return (ExecutedAction, ActionReceipt).

        The toy world accepts ``move`` (with ``direction`` in {+1, -1})
        and ``noop``. Unknown action types are rejected with
        ``outcome_code='illegal_action'``.
        """
        if proposal.action_type == "move":
            direction = proposal.params.get("direction", 1)
            if direction not in (1, -1):
                return self._reject(
                    proposal,
                    f"move direction must be +1 or -1, got {direction!r}",
                )
            executed = ExecutedAction(
                agent_id=proposal.agent_id,
                action_type="move",
                params={"direction": int(direction)},
                executed_at_tick=proposal.proposed_at_tick,
                proposal_hash=hash_state(proposal),
            )
        elif proposal.action_type == "noop":
            executed = ExecutedAction(
                agent_id=proposal.agent_id,
                action_type="noop",
                params={},
                executed_at_tick=proposal.proposed_at_tick,
                proposal_hash=hash_state(proposal),
            )
        else:
            return self._reject(
                proposal,
                f"unknown action_type {proposal.action_type!r}",
            )

        receipt = ActionReceipt(
            executed_action_hash=hash_state(executed),
            outcome_code=OUTCOME_OK,
            success=True,
            energy_delta=-MOVE_COST if executed.action_type == "move" else -NOOP_COST,
        )
        return executed, receipt

    def step(
        self,
        action: ExecutedAction,
        exogenous: ExogenousInput | None = None,
    ) -> TransitionRecord:
        before = self._build_state_view()

        # Apply the action.
        if action.action_type == "move":
            direction = int(action.params.get("direction", 1))
            if direction not in (1, -1):
                # Defensive: validate_action should have caught this,
                # but step() may be called directly with a frozen
                # ExecutedAction from replay. We still apply the move
                # with the given direction (clipped to ±1) to keep
                # replay deterministic.
                direction = 1 if direction > 0 else -1
            self._position = (self._position + direction) % self._grid_length
            self._energy -= MOVE_COST
        elif action.action_type == "noop":
            self._energy -= NOOP_COST
        else:
            # Defensive: unknown action_type. We treat it as a no-op
            # but record the actual action_type in the transition.
            pass

        # Passive recovery + tick advance + energy floor.
        self._energy += PASSIVE_RECOVERY
        if self._energy < 0:
            self._energy = 0.0
        self._tick += 1

        after = self._build_state_view()

        # Build receipt (success path — step() trusts that the action
        # was already validated; replay injects frozen ExecutedActions
        # that should have been validated at original execution time).
        receipt = ActionReceipt(
            executed_action_hash=hash_state(action),
            outcome_code=OUTCOME_OK,
            success=True,
            energy_delta=(
                -MOVE_COST + PASSIVE_RECOVERY
                if action.action_type == "move"
                else -NOOP_COST + PASSIVE_RECOVERY
            ),
        )

        return TransitionRecord(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            producer_id=TOY_WORLD_ID,
            producer_version=TOY_WORLD_VERSION,
            tick=before.meta.tick,
            state_before_hash=hash_state(before),
            candidate_actions={},
            executed_actions={action.agent_id: action},
            exogenous_input=exogenous,
            receipts={action.agent_id: receipt},
            state_delta=diff_state(before, after),
            state_after_hash=hash_state(after),
            capability_profile=self._cap,
            provenance={"seed": str(self._seed)},
        )

    def checkpoint(self) -> Checkpoint:
        state_view = self._build_state_view()
        payload = pickle.dumps(
            {
                "tick": self._tick,
                "position": self._position,
                "energy": self._energy,
                "seed": self._seed,
                "grid_length": self._grid_length,
                "initial_energy": self._initial_energy,
                "agent_id": self._agent_id,
            }
        )
        # Two-step construction to avoid the checksum circular dependency
        # (see lesson L-target-a1b2-09): build with placeholder, compute
        # real checksum, replace.
        cp_temp = Checkpoint(
            schema_version=PROTOCOL_SCHEMA_VERSION,
            world_id=TOY_WORLD_ID,
            world_version=TOY_WORLD_VERSION,
            tick=self._tick,
            state_view=state_view,
            opaque_payload=payload,
            payload_codec=TOY_WORLD_PAYLOAD_CODEC,
            capability_profile=self._cap,
            rng_bundle={"main": f"seed:{self._seed}:tick:{self._tick}"},
            checksum="sha256:placeholder",
        )
        real_checksum = compute_checkpoint_checksum(cp_temp)
        return dataclasses.replace(cp_temp, checksum=real_checksum)

    def restore(self, checkpoint: Checkpoint) -> None:
        state = pickle.loads(checkpoint.opaque_payload)
        self._tick = int(state["tick"])
        self._position = int(state["position"])
        self._energy = float(state["energy"])
        self._seed = int(state["seed"])
        self._grid_length = int(state["grid_length"])
        self._initial_energy = float(state["initial_energy"])
        self._agent_id = str(state["agent_id"])

    # --- Public read-only accessors (for tests / inspection) ---

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def position(self) -> int:
        return self._position

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def grid_length(self) -> int:
        return self._grid_length

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # --- Internal helpers ---

    def _reject(
        self,
        proposal: ActionProposal,
        reason: str,
    ) -> tuple[ExecutedAction, ActionReceipt]:
        """Build a rejection (ExecutedAction, ActionReceipt) pair."""
        executed = ExecutedAction(
            agent_id=proposal.agent_id,
            action_type=proposal.action_type,
            params=proposal.params,
            executed_at_tick=proposal.proposed_at_tick,
            proposal_hash=hash_state(proposal),
        )
        receipt = ActionReceipt(
            executed_action_hash=hash_state(executed),
            outcome_code=OUTCOME_ILLEGAL_ACTION,
            success=False,
            energy_delta=0.0,
            diagnostics={"reason": reason},
        )
        return executed, receipt

    def _build_state_view(self) -> StateView:
        return StateView(
            meta=StateMeta(
                scenario_id=f"{TOY_WORLD_ID}-scenario",
                run_id=f"{TOY_WORLD_ID}-run-{self._seed}",
                tick=self._tick,
                config_hash=f"sha256:toy-grid{self._grid_length}-e{self._initial_energy}",
                rng_state_ref=f"seed:{self._seed}:tick:{self._tick}",
            ),
            entities=EntityTable(
                schema_id=f"{TOY_WORLD_ID}-entity-v1",
                ids=(self._agent_id,),
                columns={
                    "position": (self._position,),
                    "energy": (self._energy,),
                },
            ),
            capabilities=self._cap,
            missing_mask={},
        )


# ---------------------------------------------------------------------------
# Ensure ToyWorld satisfies WorldProtocol at import time (defense against
# future drift in either ToyWorld or WorldProtocol).
# ---------------------------------------------------------------------------


# NOTE: We intentionally do NOT assert isinstance(world, WorldProtocol) at
# import time because WorldProtocol is a ``runtime_checkable`` Protocol
# and the check runs on every attribute access — running it once at import
# is enough but slows test collection. Tests in test_k08_toy_world.py
# assert the protocol conformance explicitly.

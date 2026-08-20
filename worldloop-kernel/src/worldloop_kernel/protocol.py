"""World protocol interface (K-04).

Defines the :class:`WorldProtocol` that any world (Native five-layer,
external adapter, learned simulator) implements. The kernel records and
verifies transitions; the world executes them.

Design rules (per ADR §3 and main plan §4.6 / §4.7):
- The protocol is the ONLY surface the kernel uses to drive a world.
- The world is the authority. The kernel never modifies world state
  directly; it only calls ``step`` / ``restore`` and records what
  happened.
- ``step`` returns a :class:`TransitionRecord` — the world is
  responsible for assembling receipts and delta; the kernel only
  validates and persists them.
- The kernel does NOT call LLMs. Policies live outside the kernel and
  feed :class:`ActionProposal` objects through ``validate_action``.

Out of scope (per main plan §4.7):
- LLM calls, prompts, agent memory
- Specific world rules
- Visualization, distributed scheduling
- Training algorithms, domain-specific rewards
- Lifecycle of v1 five-layer internal classes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from worldloop_kernel.action import (
        ActionProposal,
        ExecutedAction,
        ActionReceipt,
        ExogenousInput,
    )
    from worldloop_kernel.capability import CapabilityProfile
    from worldloop_kernel.state import StateView
    from worldloop_kernel.transition import (
        Checkpoint,
        TransitionRecord,
    )

__all__ = [
    "WorldProtocol",
    "ActionSpace",
    "LegalAction",
]


# ---------------------------------------------------------------------------
# Action space — returned by legal_actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegalAction:
    """A single legal action returned by :meth:`WorldProtocol.legal_actions`.

    Attributes
    ----------
    action_type:
        Stable action type string.
    params:
        Default params for this legal action. The policy MAY override
        these when constructing an :class:`ActionProposal`.
    description:
        Optional human-readable description for debugging.
    """

    action_type: str
    params: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class ActionSpace:
    """Action space for a single agent at a single tick.

    Attributes
    ----------
    agent_id:
        ID of the agent this action space belongs to.
    legal_actions:
        Tuple of :class:`LegalAction` the agent may propose. MAY be empty
        if the agent has no legal actions this tick (e.g., dead, stunned,
        or cooldown).
    is_closed:
        ``True`` if the action space is closed (only the listed
        ``action_type`` strings are accepted); ``False`` if the world
        accepts action types not listed (e.g., free-form LLM proposals).
        Closed worlds SHOULD reject unknown action types in
        ``validate_action`` with outcome_code ``"unrecognized_intent"``.
    """

    agent_id: str | int
    legal_actions: tuple[LegalAction, ...] = field(default_factory=tuple)
    is_closed: bool = True


# ---------------------------------------------------------------------------
# WorldProtocol — the ONLY surface the kernel uses to drive a world
# ---------------------------------------------------------------------------


@runtime_checkable
class WorldProtocol(Protocol):
    """Protocol every world implementation must satisfy.

    The kernel treats the world as a black box that exposes this
    protocol. The world owns all state; the kernel only records and
    validates transitions.

    Implementations MAY be classes or modules; the only requirement is
    that they provide every method below with the documented signature.
    """

    @property
    def capabilities(self) -> "CapabilityProfile":
        """Static capability declaration for this world implementation.

        Returns the SAME :class:`CapabilityProfile` for the lifetime of
        the world object. The kernel caches this on
        :class:`StateView` and :class:`Checkpoint`.
        """
        ...

    def reset(
        self,
        seed: int,
        parameters: Mapping[str, Any] | None = None,
    ) -> "StateView":
        """Reset the world to its initial state and return the initial
        :class:`StateView`.

        Parameters
        ----------
        seed:
            RNG seed for deterministic initialization.
        parameters:
            Optional world-specific parameters (e.g., scenario config).
            The kernel treats these as opaque.

        Returns
        -------
        StateView
            The state at tick 0.
        """
        ...

    def observe(self) -> "StateView":
        """Return the current :class:`StateView` without advancing the world.

        Used by policies and consumers that need to inspect the state
        without proposing actions. ``observe`` MUST be deterministic
        between ``step`` calls.
        """
        ...

    def legal_actions(
        self,
        agent_id: str | int,
        state: "StateView | None" = None,
    ) -> ActionSpace:
        """Return the legal action space for ``agent_id`` at the current
        tick (or at ``state`` if provided).

        Parameters
        ----------
        agent_id:
            ID of the agent whose action space is requested.
        state:
            Optional state to query against. If ``None``, the world's
            current state is used. Worlds that do not support
            counterfactual queries MAY raise ``NotImplementedError`` if
            ``state`` is not ``None``.

        Returns
        -------
        ActionSpace
            The action space for the agent at the queried state.
        """
        ...

    def validate_action(
        self,
        proposal: "ActionProposal",
    ) -> tuple["ExecutedAction", "ActionReceipt"]:
        """Validate a candidate :class:`ActionProposal` and return the
        :class:`ExecutedAction` + :class:`ActionReceipt`.

        The world MAY clip, project, or reject the proposal. Even when
        rejected, this method returns a receipt with ``success=False``
        and a non-``"ok"`` outcome code; it does NOT raise.

        The world MUST NOT advance state in this method. ``validate_action``
        produces the action/receipt pair; ``step`` advances the world.
        """
        ...

    def step(
        self,
        action: "ExecutedAction",
        exogenous: "ExogenousInput | None" = None,
    ) -> "TransitionRecord":
        """Apply an :class:`ExecutedAction` to the world and return the
        :class:`TransitionRecord` for the resulting transition.

        Parameters
        ----------
        action:
            The executed action to apply. MUST be the output of a prior
            ``validate_action`` call.
        exogenous:
            Optional tick-scoped exogenous input applied BEFORE the
            action.

        Returns
        -------
        TransitionRecord
            The complete transition record (state_before_hash, receipts,
            state_delta, state_after_hash, ...). The world is
            responsible for assembling all fields.
        """
        ...

    def checkpoint(self) -> "Checkpoint":
        """Snapshot the full restorable world state as a :class:`Checkpoint`.

        The checkpoint MUST include everything the world needs to resume
        exactly: hidden variables, internal caches, RNG state, scheduler
        state. The kernel records the bytes but does NOT interpret them.
        """
        ...

    def restore(self, checkpoint: "Checkpoint") -> None:
        """Restore the world from a :class:`Checkpoint`.

        After ``restore``, ``observe`` MUST return a :class:`StateView`
        whose canonical hash matches the checkpoint's ``state_view``
        hash. The kernel verifies this in K-06 validation.
        """
        ...

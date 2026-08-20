"""Joint action types (Phase 5 — multi-agent same-tick submission).

Defines the protocol-level primitives for submitting a *joint action*:
the set of actions for ALL active agents at the same tick, executed by
the world in a single simultaneous step (e.g., a parallel multi-agent
environment stepping every agent at once), plus the *joint receipt*
that reconciles per-agent outcomes.

Design rules (per main plan §10.2):
- The kernel stays generic: :class:`JointAction` carries NO
  environment-specific fields (no PettingZoo agent naming assumptions,
  no discrete action encodings). Worlds interpret per-agent proposals
  through their own action taxonomy, exactly as with the single-agent
  :class:`~worldloop_kernel.action.ActionProposal` flow.
- Backward compatibility: joint submission is an OPTIONAL extension.
  Worlds that only implement the single-agent
  :class:`~worldloop_kernel.protocol.WorldProtocol` remain fully
  valid; consumers discover joint support via
  :func:`supports_joint_actions`.
- The candidate → executed → receipt separation is preserved per
  agent: ``proposals_by_agent`` holds candidates,
  ``executed_by_agent`` holds the post-legality-check actions, and
  :class:`JointReceipt` holds the authoritative per-agent outcomes.
- Missing-agent policy is explicit. If a proposal set does not cover
  every active agent, the world resolves the gap according to
  ``missing_agent_policy`` (synthesize a no-op, synthesize the world's
  "stay" action, or reject the whole joint action). Silent defaults
  are forbidden — the resolution is recorded in the executed set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from worldloop_kernel.action import (
        ActionProposal,
        ActionReceipt,
        ExecutedAction,
        ExogenousInput,
    )
    from worldloop_kernel.transition import TransitionRecord

__all__ = [
    "JointAction",
    "JointReceipt",
    "JointActionError",
    "JointActionWorld",
    "supports_joint_actions",
    "MISSING_AGENT_NOOP",
    "MISSING_AGENT_STAY",
    "MISSING_AGENT_ERROR",
    "MISSING_AGENT_POLICIES",
]


class JointActionError(ValueError):
    """Raised when a joint action type is internally inconsistent."""


# ---------------------------------------------------------------------------
# Missing-agent policies
# ---------------------------------------------------------------------------

#: Missing agents get a world-synthesized no-op action (no state effect
#: beyond what the world's simultaneous step inherently applies).
MISSING_AGENT_NOOP = "noop"
#: Missing agents get the world's "stay in place" action (worlds whose
#: action space distinguishes an explicit stay/idle primitive).
MISSING_AGENT_STAY = "stay"
#: Missing agents are an error: the joint action MUST cover every
#: active agent or validation rejects the whole submission.
MISSING_AGENT_ERROR = "error"

#: Tuple of all kernel-defined missing-agent policies.
MISSING_AGENT_POLICIES: tuple[str, ...] = (
    MISSING_AGENT_NOOP,
    MISSING_AGENT_STAY,
    MISSING_AGENT_ERROR,
)


# ---------------------------------------------------------------------------
# JointAction — same-tick action set for all active agents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointAction:
    """Same-tick action set covering the world's active agents.

    A joint action progresses through two stages, mirroring the
    single-agent candidate → executed flow:

    1. **Proposal stage** — a scheduler builds a :class:`JointAction`
       with ``proposals_by_agent`` filled and ``executed_by_agent``
       empty, then submits it to ``validate_joint_action``.
    2. **Executed stage** — the world returns a new
       :class:`JointAction` whose ``executed_by_agent`` covers EVERY
       active agent (missing proposals resolved per
       ``missing_agent_policy``). Only executed-stage joint actions
       are accepted by ``step_joint``.

    Replay consumers MAY construct an executed-stage joint action
    directly from a recorded transition (``proposals_by_agent`` empty,
    ``executed_by_agent`` filled) — the executed set is authoritative
    for re-execution.

    Attributes
    ----------
    tick:
        World tick at which the joint action is proposed/executed.
    active_agents:
        Ordered, duplicate-free tuple of the agent ids that are active
        (alive, not terminated/truncated) at ``tick``. The executed
        set MUST cover exactly these agents.
    proposals_by_agent:
        Candidate actions keyed by agent id. Keys MUST be a subset of
        ``active_agents`` and each proposal's ``agent_id`` MUST equal
        its key. May be empty for replay-constructed joint actions.
    executed_by_agent:
        Post-legality-check actions keyed by agent id. Either empty
        (proposal stage) or covering exactly ``active_agents``
        (executed stage). Each executed action's ``agent_id`` MUST
        equal its key.
    missing_agent_policy:
        One of :data:`MISSING_AGENT_POLICIES`. Controls how the world
        resolves active agents without a proposal. With
        :data:`MISSING_AGENT_ERROR`, a proposal-stage joint action
        MUST already cover every active agent.
    """

    tick: int
    active_agents: tuple[str | int, ...]
    proposals_by_agent: Mapping[str | int, "ActionProposal"] = field(
        default_factory=dict
    )
    executed_by_agent: Mapping[str | int, "ExecutedAction"] = field(
        default_factory=dict
    )
    missing_agent_policy: str = MISSING_AGENT_NOOP

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise JointActionError(f"tick must be >= 0, got {self.tick}")
        if not self.active_agents:
            raise JointActionError("active_agents must be non-empty")
        active_set = set(self.active_agents)
        if len(active_set) != len(self.active_agents):
            raise JointActionError(
                f"active_agents contains duplicates: {self.active_agents!r}"
            )
        if self.missing_agent_policy not in MISSING_AGENT_POLICIES:
            raise JointActionError(
                f"missing_agent_policy must be one of "
                f"{MISSING_AGENT_POLICIES!r}, got "
                f"{self.missing_agent_policy!r}"
            )
        # Proposal keys ⊆ active_agents; per-proposal agent_id matches key.
        extra = set(self.proposals_by_agent) - active_set
        if extra:
            raise JointActionError(
                f"proposals_by_agent contains agents not in active_agents: "
                f"{sorted(map(str, extra))!r}"
            )
        for agent_id, proposal in self.proposals_by_agent.items():
            if proposal.agent_id != agent_id:
                raise JointActionError(
                    f"proposal keyed by {agent_id!r} has "
                    f"agent_id={proposal.agent_id!r}; keys MUST match "
                    "the proposal's agent_id"
                )
        # Executed set: either empty (proposal stage) or covering
        # exactly the active agents (executed stage).
        if self.executed_by_agent:
            executed_set = set(self.executed_by_agent)
            if executed_set != active_set:
                missing = sorted(map(str, active_set - executed_set))
                surplus = sorted(map(str, executed_set - active_set))
                raise JointActionError(
                    "executed_by_agent MUST cover exactly active_agents; "
                    f"missing={missing!r}, surplus={surplus!r}"
                )
            for agent_id, executed in self.executed_by_agent.items():
                if executed.agent_id != agent_id:
                    raise JointActionError(
                        f"executed action keyed by {agent_id!r} has "
                        f"agent_id={executed.agent_id!r}; keys MUST match "
                        "the executed action's agent_id"
                    )
        elif self.missing_agent_policy == MISSING_AGENT_ERROR:
            # Proposal stage with ERROR policy: proposals MUST already
            # cover every active agent — there is no synthesis path.
            uncovered = active_set - set(self.proposals_by_agent)
            if uncovered:
                raise JointActionError(
                    f"missing_agent_policy={MISSING_AGENT_ERROR!r} requires "
                    "proposals for every active agent; missing: "
                    f"{sorted(map(str, uncovered))!r}"
                )

    @property
    def is_executed_stage(self) -> bool:
        """``True`` when the executed set is filled (ready for step)."""
        return bool(self.executed_by_agent)


# ---------------------------------------------------------------------------
# JointReceipt — per-agent outcome reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointReceipt:
    """Per-agent authoritative outcomes for one joint submission.

    The joint receipt is the multi-agent analogue of
    :class:`~worldloop_kernel.action.ActionReceipt`: it reconciles the
    outcome of EVERY agent that participated in the joint action —
    success/failure, outcome code, and world-attached diagnostics
    (e.g., per-agent reward / termination / truncation flags live in
    each receipt's ``diagnostics``; the kernel does not interpret
    them).

    Attributes
    ----------
    tick:
        Tick at which the joint action was validated/executed.
    receipts_by_agent:
        Per-agent receipts keyed by agent id. MUST be non-empty. Each
        receipt's ``executed_action_hash`` ties back to that agent's
        executed action.
    """

    tick: int
    receipts_by_agent: Mapping[str | int, "ActionReceipt"]

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise JointActionError(f"tick must be >= 0, got {self.tick}")
        if not self.receipts_by_agent:
            raise JointActionError("receipts_by_agent must be non-empty")


# ---------------------------------------------------------------------------
# JointActionWorld — optional world extension protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class JointActionWorld(Protocol):
    """Optional world extension: same-tick joint action submission.

    Worlds that can execute every active agent's action in a single
    simultaneous step implement this protocol IN ADDITION to
    :class:`~worldloop_kernel.protocol.WorldProtocol`. The single-agent
    ``validate_action``/``step`` path MUST keep working — joint
    submission is additive, never a replacement.

    Consumers discover support via :func:`supports_joint_actions`
    (structural check) and MUST NOT assume joint capability from the
    world's :class:`~worldloop_kernel.capability.CapabilityProfile`
    alone.
    """

    def validate_joint_action(
        self, joint: JointAction
    ) -> tuple[JointAction, JointReceipt]:
        """Validate a proposal-stage joint action.

        Returns an executed-stage :class:`JointAction` (executed set
        covering every active agent, missing proposals resolved per
        ``missing_agent_policy``) plus a :class:`JointReceipt` carrying
        validation-stage receipts. Per-agent rejections MUST surface as
        ``success=False`` receipts — never as exceptions — unless the
        joint action itself is structurally invalid.
        """
        ...

    def step_joint(
        self,
        joint: JointAction,
        *,
        exogenous: "ExogenousInput | None" = None,
    ) -> "TransitionRecord":
        """Execute an executed-stage joint action as ONE world step.

        All agents' actions are applied simultaneously; the returned
        :class:`~worldloop_kernel.transition.TransitionRecord` MUST
        contain executed actions AND receipts for every active agent
        (record-level invariant: ``executed_actions`` keys ==
        ``receipts`` keys).
        """
        ...


def supports_joint_actions(world: Any) -> bool:
    """Return ``True`` if ``world`` structurally implements joint submission.

    Checks for callable ``validate_joint_action`` and ``step_joint``
    attributes. This is the discovery mechanism for the optional
    :class:`JointActionWorld` extension — consumers fall back to the
    single-agent path when this returns ``False``.
    """
    return callable(getattr(world, "validate_joint_action", None)) and callable(
        getattr(world, "step_joint", None)
    )

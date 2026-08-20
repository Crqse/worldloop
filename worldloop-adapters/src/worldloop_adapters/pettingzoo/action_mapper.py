"""Action mapper: PettingZoo action ↔ kernel ActionProposal/ExecutedAction (A-01).

Maps between PettingZoo Parallel API actions (per-agent discrete or Box
actions) and kernel :class:`ActionProposal` / :class:`ExecutedAction` /
:class:`ActionReceipt`.

Mapping summary (per main plan §12.2):
- PettingZoo discrete action (int 0..N-1) → ActionProposal.action_type="move"
  with ``params={"discrete_action": int}``.
- ActionProposal.agent_id (str "agent_0") → PettingZoo agent_id (str).
- ExecutedAction carries the proposal_hash and the resolved discrete action.
- ActionReceipt carries the outcome (success / outcome_code) and reward
  from the env step.

Outcome codes (per kernel §action.py + M2 Gate (e) reward reconciliation):
- ``OUTCOME_OK``: action executed successfully.
- ``OUTCOME_ILLEGAL_ACTION``: action outside the legal action space.
- ``OUTCOME_ILLEGAL_TARGET``: action target invalid (e.g., dead agent).
- ``OUTCOME_UNKNOWN_FAILURE``: env raised an unexpected exception.
"""
from __future__ import annotations

from typing import Any

from worldloop_kernel.action import (
    ActionProposal,
    ActionReceipt,
    ExecutedAction,
    OUTCOME_ILLEGAL_ACTION,
    OUTCOME_OK,
    OUTCOME_UNKNOWN_FAILURE,
)
from worldloop_kernel.canonical import hash_state

__all__ = [
    "proposal_to_pettingzoo_action",
    "build_executed_action",
    "build_receipt",
    "reject_proposal",
    "PETTINGZOO_DEFAULT_ACTION_TYPE",
    "PETTINGZOO_LEGAL_DISCRETE_ACTIONS",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Default action_type for PettingZoo discrete actions. Adapters MAY use
#: env-specific action_type strings (e.g., "move" for MPE, "push" for
#: Simple Tag) by overriding the adapter's ``_action_type`` attribute.
PETTINGZOO_DEFAULT_ACTION_TYPE = "move"

#: Default legal discrete actions for MPE envs (5 actions: STAY/LEFT/RIGHT/DOWN/UP).
PETTINGZOO_LEGAL_DISCRETE_ACTIONS: tuple[int, ...] = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Proposal → PettingZoo action
# ---------------------------------------------------------------------------


def proposal_to_pettingzoo_action(
    proposal: ActionProposal,
    legal_actions: tuple[int, ...] = PETTINGZOO_LEGAL_DISCRETE_ACTIONS,
) -> int | None:
    """Convert an :class:`ActionProposal` to a PettingZoo discrete action int.

    Returns ``None`` if the proposal's discrete_action is outside the
    legal action space (the caller should treat this as a rejection).

    Parameters
    ----------
    proposal:
        The candidate action from the policy.
    legal_actions:
        Tuple of legal discrete action ints for this env.
    """
    discrete = proposal.params.get("discrete_action")
    if discrete is None:
        return None
    try:
        discrete_int = int(discrete)
    except (TypeError, ValueError):
        return None
    if discrete_int not in legal_actions:
        return None
    return discrete_int


# ---------------------------------------------------------------------------
# ExecutedAction + Receipt builders
# ---------------------------------------------------------------------------


def build_executed_action(proposal: ActionProposal) -> ExecutedAction:
    """Build an :class:`ExecutedAction` from a validated :class:`ActionProposal`.

    The executed action carries the ``proposal_hash`` (computed via
    :func:`hash_state`) so that ``step`` can look it up later without
    re-validating. ``executed_at_tick`` is set to the proposal's
    ``proposed_at_tick`` (the world's current tick at proposal time).
    """
    proposal_hash = hash_state(proposal)
    return ExecutedAction(
        agent_id=proposal.agent_id,
        action_type=proposal.action_type,
        params=proposal.params,
        executed_at_tick=proposal.proposed_at_tick,
        proposal_hash=proposal_hash,
    )


def build_receipt(
    executed: ExecutedAction,
    *,
    reward: float = 0.0,
    success: bool = True,
    outcome_code: str = OUTCOME_OK,
    info: dict[str, Any] | None = None,
) -> ActionReceipt:
    """Build an :class:`ActionReceipt` for an executed action.

    The kernel :class:`ActionReceipt` does not have dedicated ``reward``
    or ``info`` fields (the kernel is reward-agnostic). Per ADR §3 and
    M2 Gate (e) reward reconciliation, the env-step reward and info dict
    are surfaced via ``diagnostics`` so consumers can read them as
    ``receipt.diagnostics["reward"]`` and ``receipt.diagnostics["info"]``.

    Parameters
    ----------
    executed:
        The executed action this receipt corresponds to.
    reward:
        The reward from the env step (per-agent). Stored in
        ``diagnostics["reward"]``.
    success:
        Whether the action executed successfully.
    outcome_code:
        Kernel outcome code (``OUTCOME_OK`` / ``OUTCOME_ILLEGAL_ACTION`` / ...).
    info:
        Optional info dict from the env step. Stored in
        ``diagnostics["info"]``.
    """
    diagnostics: dict[str, Any] = {"reward": float(reward)}
    if info:
        diagnostics["info"] = dict(info)
    return ActionReceipt(
        executed_action_hash=hash_state(executed),
        outcome_code=outcome_code,
        success=success,
        energy_delta=0.0,  # PettingZoo envs have no energy concept.
        events=(),
        diagnostics=diagnostics,
    )


def reject_proposal(proposal: ActionProposal, outcome_code: str = OUTCOME_ILLEGAL_ACTION) -> tuple[ExecutedAction, ActionReceipt]:
    """Build an (ExecutedAction, ActionReceipt) pair for a rejected proposal.

    Used when the proposal's discrete_action is outside the legal action
    space or when the agent is dead.
    """
    executed = build_executed_action(proposal)
    receipt = build_receipt(
        executed,
        reward=0.0,
        success=False,
        outcome_code=outcome_code,
        info={"rejected": True},
    )
    return executed, receipt

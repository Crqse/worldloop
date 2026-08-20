"""Action mapper: Gymnasium action ↔ kernel ActionProposal/ExecutedAction (A-05).

Maps between Gymnasium single-agent API actions (discrete int or Box)
and kernel :class:`ActionProposal` / :class:`ExecutedAction` /
:class:`ActionReceipt`.

Mapping (per main plan §12.2):
- Gymnasium discrete action (int 0..N-1) → ActionProposal.action_type="step"
  with ``params={"discrete_action": int}``.
- ActionProposal.agent_id (str "agent_0") → fixed single agent.
- ActionReceipt reward → diagnostics["reward"] (kernel is reward-agnostic).
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
    "proposal_to_gymnasium_action",
    "build_executed_action",
    "build_receipt",
    "reject_proposal",
    "GYMNASIUM_DEFAULT_ACTION_TYPE",
]


GYMNASIUM_DEFAULT_ACTION_TYPE = "step"


def proposal_to_gymnasium_action(
    proposal: ActionProposal,
    legal_actions: tuple[int, ...],
) -> int | None:
    """Convert an :class:`ActionProposal` to a Gymnasium discrete action int.

    Returns ``None`` if the proposal's discrete_action is outside the
    legal action space.
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


def build_executed_action(proposal: ActionProposal) -> ExecutedAction:
    """Build an :class:`ExecutedAction` from a validated :class:`ActionProposal`."""
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

    Reward and info are surfaced via ``diagnostics`` (kernel is
    reward-agnostic; per ADR §3 and M2 Gate (e)).
    """
    diagnostics: dict[str, Any] = {"reward": float(reward)}
    if info:
        diagnostics["info"] = dict(info)
    return ActionReceipt(
        executed_action_hash=hash_state(executed),
        outcome_code=outcome_code,
        success=success,
        energy_delta=0.0,
        events=(),
        diagnostics=diagnostics,
    )


def reject_proposal(
    proposal: ActionProposal, outcome_code: str = OUTCOME_ILLEGAL_ACTION
) -> tuple[ExecutedAction, ActionReceipt]:
    """Build an (ExecutedAction, ActionReceipt) pair for a rejected proposal."""
    executed = build_executed_action(proposal)
    receipt = build_receipt(
        executed,
        reward=0.0,
        success=False,
        outcome_code=outcome_code,
        info={"rejected": True},
    )
    return executed, receipt

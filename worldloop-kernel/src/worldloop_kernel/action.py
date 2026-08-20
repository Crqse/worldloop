"""Action types (K-04).

Defines the candidate → executed → receipt action triple. Candidate
actions are produced by a policy (reflex, LLM, random); the world
validates and executes them, returning an :class:`ActionReceipt`.

Design rules (per ADR §3 and main plan §4.6):
- Candidate and executed actions MUST be separate types. A proposal
  that fails legality check does NOT become an :class:`ExecutedAction`
  with ``success=False``; it becomes a receipt-only record (the world
  may still emit an :class:`ExecutedAction` with ``outcome_code``
  indicating the rejection reason for traceability, but the receipt is
  the authoritative outcome).
- ``outcome_code`` is a stable string enum. The kernel defines a
  minimal taxonomy of kernel-level outcome codes; worlds MAY add their
  own domain-specific codes (e.g., ``"resource_consumed"``).
- :class:`ActionReceipt` is the only place where energy / health deltas
  are declared; the world applies them and the kernel records them.
- Failure paths MUST NOT use exception capture as a no-op and MUST NOT
  fall back to REST. A failed action returns a receipt with
  ``success=False`` and a non-``"ok"`` outcome code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "ActionProposal",
    "ExecutedAction",
    "ExogenousInput",
    "ActionReceipt",
    "ActionError",
    "OutcomeCode",
    "OUTCOME_OK",
    "OUTCOME_DISABLED_BY_ABLATION",
    "OUTCOME_FEATURE_DISABLED",
    "OUTCOME_UNRECOGNIZED_INTENT",
    "OUTCOME_ILLEGAL_TARGET",
    "OUTCOME_ILLEGAL_ACTION",
    "OUTCOME_INSUFFICIENT_ENERGY",
    "OUTCOME_UNKNOWN_FAILURE",
    "KERNEL_OUTCOME_CODES",
]


class ActionError(ValueError):
    """Raised when an action type is internally inconsistent."""


# ---------------------------------------------------------------------------
# Outcome codes — kernel-level taxonomy
# ---------------------------------------------------------------------------

#: General success.
OUTCOME_OK = "ok"
#: Action disabled by ablation (e.g., ``ae_disabled_intents`` hit).
OUTCOME_DISABLED_BY_ABLATION = "disabled_by_ablation"
#: Feature disabled (e.g., ``free_concept_registry_enabled=false``).
OUTCOME_FEATURE_DISABLED = "feature_disabled"
#: LLM candidate could not be parsed into a known ActionIntent.
OUTCOME_UNRECOGNIZED_INTENT = "unrecognized_intent"
#: Action target is illegal (out of range, not owned, ...).
OUTCOME_ILLEGAL_TARGET = "illegal_target"
#: Action itself is illegal at this tick (cooldown, wrong state, ...).
OUTCOME_ILLEGAL_ACTION = "illegal_action"
#: Agent has insufficient energy to execute the action.
OUTCOME_INSUFFICIENT_ENERGY = "insufficient_energy"
#: Action failed for an unknown reason. Worlds SHOULD prefer a more
#: specific code; this is the fallback.
OUTCOME_UNKNOWN_FAILURE = "unknown_failure"

#: Tuple of all kernel-defined outcome codes. Worlds MAY add their own
#: codes (e.g., ``"resource_consumed"`` for FORAGE); the kernel only
#: validates that the code is a non-empty string.
KERNEL_OUTCOME_CODES: tuple[str, ...] = (
    OUTCOME_OK,
    OUTCOME_DISABLED_BY_ABLATION,
    OUTCOME_FEATURE_DISABLED,
    OUTCOME_UNRECOGNIZED_INTENT,
    OUTCOME_ILLEGAL_TARGET,
    OUTCOME_ILLEGAL_ACTION,
    OUTCOME_INSUFFICIENT_ENERGY,
    OUTCOME_UNKNOWN_FAILURE,
)

#: Type alias for outcome codes.
OutcomeCode = str


# ---------------------------------------------------------------------------
# Action proposal — candidate action from a policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionProposal:
    """Candidate action from a policy. Not yet validated.

    The world's ``validate_action`` consumes an :class:`ActionProposal`
    and produces an :class:`ExecutedAction` + :class:`ActionReceipt`.

    Attributes
    ----------
    agent_id:
        ID of the agent proposing the action. ``str`` or ``int`` to
        accommodate both named and indexed agent pools.
    action_type:
        Stable action type string (e.g., ``"FORAGE"``, ``"MOVE"``,
        ``"COMMUNICATE"``). The kernel does NOT enumerate action types;
        the world owns the taxonomy.
    params:
        Action parameters (e.g., target ID, direction, intensity).
        Treated as opaque by the kernel.
    proposed_at_tick:
        Tick at which the proposal was made. MUST be the world's current
        tick at proposal time.
    proposer:
        Origin of the proposal: ``"reflex"``, ``"llm"``, ``"random"``,
        ``"frozen"``, ... Used for replay and ablation grouping.
    """

    agent_id: str | int
    action_type: str
    params: Mapping[str, Any]
    proposed_at_tick: int
    proposer: str

    def __post_init__(self) -> None:
        if not self.action_type:
            raise ActionError("action_type must be a non-empty string")
        if not self.proposer:
            raise ActionError("proposer must be a non-empty string")
        if self.proposed_at_tick < 0:
            raise ActionError(
                f"proposed_at_tick must be >= 0, got {self.proposed_at_tick}"
            )


# ---------------------------------------------------------------------------
# Executed action — action after legality check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutedAction:
    """Action after legality check. May differ from the proposal.

    The world MAY clip, project, or substitute the proposal. The
    ``proposal_hash`` field ties the executed action back to the
    original proposal for traceability.

    Attributes
    ----------
    agent_id:
        ID of the agent executing the action.
    action_type:
        Action type that was actually executed. MAY differ from the
        proposal's ``action_type`` if the world substituted a default
        (e.g., a malformed FORAGE becomes a no-op MOVE).
    params:
        Executed action parameters. MAY differ from the proposal's
        params if the world clipped or projected them.
    executed_at_tick:
        Tick at which the action was executed. MUST be the world's
        current tick at execution time.
    proposal_hash:
        Hash of the original :class:`ActionProposal` (computed via
        :func:`worldloop_kernel.canonical.hash_state` in K-05). Ties
        the executed action back to the proposal for replay and audit.
    """

    agent_id: str | int
    action_type: str
    params: Mapping[str, Any]
    executed_at_tick: int
    proposal_hash: str

    def __post_init__(self) -> None:
        if not self.action_type:
            raise ActionError("action_type must be a non-empty string")
        if not self.proposal_hash:
            raise ActionError("proposal_hash must be a non-empty string")
        if self.executed_at_tick < 0:
            raise ActionError(
                f"executed_at_tick must be >= 0, got {self.executed_at_tick}"
            )


# ---------------------------------------------------------------------------
# Exogenous input — tick-scoped external input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExogenousInput:
    """Tick-scoped external input (resource pulse, hazard spike, ...).

    Exogenous inputs are NOT actions; they are environmental events the
    world injects at the start of a tick before actions are executed.
    The kernel treats them as opaque payloads.

    Attributes
    ----------
    tick:
        Tick at which the exogenous input is applied.
    kind:
        Stable kind string (e.g., ``"resource_pulse"``,
        ``"hazard_spike"``). The kernel does NOT enumerate kinds.
    payload:
        Free-form payload the world interprets.
    """

    tick: int
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ActionError("kind must be a non-empty string")
        if self.tick < 0:
            raise ActionError(f"tick must be >= 0, got {self.tick}")


# ---------------------------------------------------------------------------
# Action receipt — outcome of an executed action
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionReceipt:
    """Outcome of an executed action: success / failure / disabled / ...

    The receipt is the AUTHORITATIVE outcome. Even when an action is
    rejected, the world returns a receipt with ``success=False`` and a
    non-``"ok"`` outcome code; the kernel does NOT infer rejection from
    exceptions.

    Attributes
    ----------
    executed_action_hash:
        Hash of the :class:`ExecutedAction` this receipt corresponds to
        (computed via :func:`worldloop_kernel.canonical.hash_state` in
        K-05). Ties the receipt back to the executed action.
    outcome_code:
        Stable outcome code from :data:`KERNEL_OUTCOME_CODES` or a
        world-defined code. MUST be a non-empty string.
    success:
        ``True`` if the action achieved its intended effect, ``False``
        otherwise. Failure paths MUST set ``success=False`` and a
        non-``"ok"`` outcome code.
    energy_delta:
        Net energy delta applied to the agent by this action. MAY be
        negative (cost), zero (no-op or rejected), or positive (gain).
        Failure paths typically have ``energy_delta <= 0``.
    events:
        Tuple of event kind strings the world surfaces as a result of
        this action (e.g., ``("resource_consumed",)``). The kernel does
        NOT interpret event kinds.
    diagnostics:
        Free-form diagnostic mapping the world attaches (e.g.,
        ``{"target_id": 7, "intensity": 0.8}``). Useful for debugging
        and ablation analysis.
    """

    executed_action_hash: str
    outcome_code: OutcomeCode
    success: bool
    energy_delta: float
    events: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.executed_action_hash:
            raise ActionError(
                "executed_action_hash must be a non-empty string"
            )
        if not self.outcome_code:
            raise ActionError("outcome_code must be a non-empty string")
        # Rule: success=True MUST pair with outcome_code == OUTCOME_OK.
        # Worlds that succeed with a non-"ok" code are misusing the
        # taxonomy. Failure paths are unconstrained (any non-"ok" code).
        if self.success and self.outcome_code != OUTCOME_OK:
            raise ActionError(
                f"success=True MUST pair with outcome_code={OUTCOME_OK!r}, "
                f"got outcome_code={self.outcome_code!r}."
            )
        if not self.success and self.outcome_code == OUTCOME_OK:
            raise ActionError(
                f"success=False MUST NOT pair with outcome_code={OUTCOME_OK!r}; "
                "use a failure-specific code."
            )

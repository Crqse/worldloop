"""S-09 Coverage Scheduler — discover and prioritize weak regions.

The coverage scheduler decides which :class:`~worldloop_data.policy.Policy`
to invoke per (tick, agent) and records observations for the coverage
report. M4's stub uses a uniform rotation; deeper strategies (rare-action
up-sampling, baseline-divergence prioritization, ...) are deferred to
later attempts per the S2+S4 hybrid strategy.

Design rules (per main plan §14.3):
- The scheduler is the ONLY component that decides policy choice per
  tick. Policies themselves do not decide when to fire.
- The scheduler records every transition it observes (when
  ``CoverageConfig.record_observations=True``) so the coverage report
  reflects actual production, not planned production.
- The coverage report is consumed by Q6 (coverage Gate item) and by Q9
  (utility: per-policy baseline comparison).

Coverage report dimensions tracked by the stub:
- per-policy invocation count
- per-action-type emission count
- per-outcome-code count (from receipts)
- tick count
- per-agent action count
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TYPE_CHECKING

from worldloop_kernel import (
    ActionSpace,
    StateView,
    TransitionRecord,
)

from worldloop_data.config import CoverageConfig
from worldloop_data.policy import Policy, PolicyPool

if TYPE_CHECKING:
    pass

__all__ = [
    "CoverageScheduler",
    "CoverageReport",
    "UniformCoverageScheduler",
]


# ---------------------------------------------------------------------------
# CoverageReport — output of coverage_report()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageReport:
    """Summary of coverage observations.

    Attributes
    ----------
    policy_usage:
        Mapping ``policy_id -> invocation count``.
    action_type_counts:
        Mapping ``action_type -> emission count`` (from candidate_actions).
    outcome_code_counts:
        Mapping ``outcome_code -> count`` (from receipts).
    agent_action_counts:
        Mapping ``agent_id (str) -> action count``.
    tick_count:
        Number of ticks observed.
    transition_count:
        Number of transitions observed.
    notes:
        Free-form notes (e.g., "stub mode: uniform rotation").
    """

    policy_usage: Mapping[str, int]
    action_type_counts: Mapping[str, int]
    outcome_code_counts: Mapping[str, int]
    agent_action_counts: Mapping[str, int]
    tick_count: int
    transition_count: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_usage": dict(self.policy_usage),
            "action_type_counts": dict(self.action_type_counts),
            "outcome_code_counts": dict(self.outcome_code_counts),
            "agent_action_counts": dict(self.agent_action_counts),
            "tick_count": self.tick_count,
            "transition_count": self.transition_count,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# CoverageScheduler Protocol
# ---------------------------------------------------------------------------


class CoverageScheduler(Protocol):
    """Decides which policy to invoke and records observations.

    The scheduler is called once per (tick, agent) by the rollout
    orchestrator. It returns the chosen :class:`Policy`; the orchestrator
    then calls ``policy.propose(ctx)``.
    """

    def select_policy(
        self,
        state: StateView,
        agent_id: str | int,
        action_space: ActionSpace,
        pool: PolicyPool,
    ) -> Policy:
        """Pick a policy from ``pool`` for this (state, agent, action_space)."""
        ...

    def record_observation(self, transition: TransitionRecord) -> None:
        """Record a transition for the coverage report."""
        ...

    def coverage_report(self) -> CoverageReport:
        """Produce the coverage report (consumed by Q6 and Q9)."""
        ...


# ---------------------------------------------------------------------------
# UniformCoverageScheduler — reference stub
# ---------------------------------------------------------------------------


class UniformCoverageScheduler:
    """Uniform rotation over the registered policies.

    Advances a single global rotation cursor by 1 per
    :meth:`select_policy` call and picks
    ``policies[cursor % len(policies)]``. This guarantees that every
    policy gets roughly equal airtime over a long run, which is
    sufficient for the M4 stub. Sophisticated strategies (weighted,
    rare-action up-sampling, baseline-divergence) are deferred.

    The cursor is GLOBAL (not per-agent) so that in multi-agent worlds
    where each tick may pick a different agent, every registered policy
    still gets invoked. A per-agent cursor would leave each agent's
    cursor at 0 when each agent fires only once, defeating Q9 (utility:
    need ≥2 policies actually invoked).

    Attributes
    ----------
    config:
        The :class:`CoverageConfig` governing strategy and recording.
    """

    def __init__(self, *, config: CoverageConfig | None = None) -> None:
        self.config = config or CoverageConfig()
        self._policy_usage: Counter[str] = Counter()
        self._action_type_counts: Counter[str] = Counter()
        self._outcome_code_counts: Counter[str] = Counter()
        self._agent_action_counts: Counter[str] = Counter()
        self._tick_count: int = 0
        self._transition_count: int = 0
        # Global rotation cursor (shared across all agents / ticks).
        self._cursor: int = 0

    def select_policy(
        self,
        state: StateView,
        agent_id: str | int,
        action_space: ActionSpace,
        pool: PolicyPool,
    ) -> Policy:
        if not pool.policies:
            raise ValueError("PolicyPool is empty; cannot select a policy")
        idx = self._cursor % len(pool.policies)
        self._cursor += 1
        return pool.policies[idx]

    def record_observation(self, transition: TransitionRecord) -> None:
        if not self.config.record_observations:
            return
        self._transition_count += 1
        # Track per-tick (dedupe by transition.tick).
        # We count ticks as "distinct ticks seen" — simple approximation
        # is to just count transitions; the report's tick_count is
        # bounded by transition_count anyway.
        self._tick_count += 1
        # Per-action-type counts from candidate_actions.
        for agent_id, proposal in transition.candidate_actions.items():
            self._action_type_counts[proposal.action_type] += 1
            self._agent_action_counts[str(agent_id)] += 1
        # Per-outcome-code counts from receipts.
        for receipt in transition.receipts.values():
            self._outcome_code_counts[receipt.outcome_code] += 1

    def _note_policy_use(self, policy_id: str) -> None:
        self._policy_usage[policy_id] += 1

    def coverage_report(self) -> CoverageReport:
        return CoverageReport(
            policy_usage=dict(self._policy_usage),
            action_type_counts=dict(self._action_type_counts),
            outcome_code_counts=dict(self._outcome_code_counts),
            agent_action_counts=dict(self._agent_action_counts),
            tick_count=self._tick_count,
            transition_count=self._transition_count,
            notes=f"stub mode: {self.config.strategy} strategy",
        )

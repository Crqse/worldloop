"""Matched policy-outcome utility evaluation for the Q9 quality gate.

Q9 is deliberately stricter than a policy-coverage check.  A comparison
is valid only when two policies act from the same parent state under the
same exogenous input.  The world is restored from a checkpoint before
each policy action, so the resulting transition records are directly
comparable.

The evaluator is opt-in.  This avoids silently issuing extra calls when
one of the supplied policies is backed by an external LLM.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from worldloop_kernel import ActionProposal, canonical_encode

from worldloop_data.policy import Policy, PolicyContext, PolicyPool

__all__ = [
    "OutcomeUtility",
    "PolicyOutcome",
    "UtilityComparison",
    "UtilityEvaluationReport",
    "evaluate_matched_policy_utility",
]


@dataclass(frozen=True)
class OutcomeUtility:
    """Frozen v1 utility vector extracted from one realized transition."""

    energy_delta: float
    survived: bool
    task_progress: float
    constraint_violations: int
    world_side_effect_penalty: float
    scalar: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy_delta": self.energy_delta,
            "survived": self.survived,
            "task_progress": self.task_progress,
            "constraint_violations": self.constraint_violations,
            "world_side_effect_penalty": self.world_side_effect_penalty,
            "scalar": self.scalar,
        }


@dataclass(frozen=True)
class PolicyOutcome:
    """One policy's realized outcome from a matched parent state."""

    policy_id: str
    seed: int
    tick: int
    focal_agent_id: str
    state_before_hash: str
    exogenous_hash: str
    state_after_hash: str
    action_type: str
    outcome_code: str
    utility: OutcomeUtility

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "seed": self.seed,
            "tick": self.tick,
            "focal_agent_id": self.focal_agent_id,
            "state_before_hash": self.state_before_hash,
            "exogenous_hash": self.exogenous_hash,
            "state_after_hash": self.state_after_hash,
            "action_type": self.action_type,
            "outcome_code": self.outcome_code,
            "utility": self.utility.to_dict(),
        }


@dataclass(frozen=True)
class UtilityComparison:
    """Candidate-minus-baseline comparison under matched conditions."""

    seed: int
    tick: int
    focal_agent_id: str
    baseline_policy_id: str
    candidate_policy_id: str
    state_before_hash: str
    exogenous_hash: str
    baseline_utility: float
    candidate_utility: float
    delta: float
    matched: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "tick": self.tick,
            "focal_agent_id": self.focal_agent_id,
            "baseline_policy_id": self.baseline_policy_id,
            "candidate_policy_id": self.candidate_policy_id,
            "state_before_hash": self.state_before_hash,
            "exogenous_hash": self.exogenous_hash,
            "baseline_utility": self.baseline_utility,
            "candidate_utility": self.candidate_utility,
            "delta": self.delta,
            "matched": self.matched,
        }


@dataclass(frozen=True)
class UtilityEvaluationReport:
    """Machine-readable result consumed by Q9."""

    protocol_version: str
    baseline_policy_id: str
    policy_ids: tuple[str, ...]
    outcomes: tuple[PolicyOutcome, ...]
    comparisons: tuple[UtilityComparison, ...]
    min_improvement: float
    notes: str = ""

    @property
    def valid_comparisons(self) -> tuple[UtilityComparison, ...]:
        return tuple(
            c
            for c in self.comparisons
            if c.matched
            and math.isfinite(c.baseline_utility)
            and math.isfinite(c.candidate_utility)
            and math.isfinite(c.delta)
        )

    @property
    def best_delta(self) -> float | None:
        valid = self.valid_comparisons
        if not valid:
            return None
        return max(c.delta for c in valid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "baseline_policy_id": self.baseline_policy_id,
            "policy_ids": list(self.policy_ids),
            "min_improvement": self.min_improvement,
            "best_delta": self.best_delta,
            "n_outcomes": len(self.outcomes),
            "n_comparisons": len(self.comparisons),
            "n_valid_comparisons": len(self.valid_comparisons),
            "outcomes": [o.to_dict() for o in self.outcomes],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "notes": self.notes,
        }


def _pick_agent(state: Any, cursor: int) -> str | int | None:
    ids = state.entities.ids
    if not ids:
        return None
    columns = state.entities.columns
    if "alive" in columns:
        alive_ids = [
            entity_id
            for entity_id, alive in zip(ids, columns["alive"])
            if alive
        ]
        if not alive_ids:
            return None
        return alive_ids[cursor % len(alive_ids)]
    return ids[cursor % len(ids)]


def _hash_exogenous(exogenous: Any) -> str:
    if exogenous is None:
        return "none"
    return "sha256:" + hashlib.sha256(canonical_encode(exogenous)).hexdigest()


def _extract_utility(record: Any, focal_agent_id: str | int) -> OutcomeUtility:
    receipt = record.receipts[focal_agent_id]
    entity_changes = record.state_delta.entity_changes
    realized_energy_change = 0.0
    if entity_changes is not None:
        for change in entity_changes.changes:
            if (
                str(getattr(change, "entity_id", "")) == str(focal_agent_id)
                and getattr(change, "kind", "") == "update"
                and getattr(change, "column", "") == "energy"
                and isinstance(getattr(change, "before", None), (int, float))
                and isinstance(getattr(change, "after", None), (int, float))
            ):
                realized_energy_change += float(change.after) - float(change.before)
    population = record.state_delta.population_changes
    deaths = tuple(population.changes) if population is not None else ()
    focal_key = str(focal_agent_id)
    focal_died = any(
        getattr(change, "kind", "") == "death"
        and str(getattr(change, "agent_id", "")) == focal_key
        for change in deaths
    )
    non_focal_deaths = sum(
        1
        for change in deaths
        if getattr(change, "kind", "") == "death"
        and str(getattr(change, "agent_id", "")) != focal_key
    )
    non_focal_failures = sum(
        1
        for agent_id, other_receipt in record.receipts.items()
        if str(agent_id) != focal_key and not other_receipt.success
    )
    constraint_violations = 0 if receipt.success else 1
    task_progress = 1.0 if receipt.success else 0.0
    side_effect_penalty = float(non_focal_deaths + non_focal_failures)
    survived = not focal_died
    total_energy_delta = float(receipt.energy_delta) + realized_energy_change
    scalar = (
        total_energy_delta
        + task_progress
        + (0.0 if survived else -5.0)
        - 2.0 * constraint_violations
        - side_effect_penalty
    )
    return OutcomeUtility(
        energy_delta=total_energy_delta,
        survived=survived,
        task_progress=task_progress,
        constraint_violations=constraint_violations,
        world_side_effect_penalty=side_effect_penalty,
        scalar=scalar,
    )


def evaluate_matched_policy_utility(
    *,
    scenario_package: Any,
    policies: Sequence[Policy],
    seeds: Sequence[int],
    horizon: int = 5,
    baseline_policy_id: str = "",
    min_improvement: float = 1e-9,
    policy_pool_config: Any = None,
) -> UtilityEvaluationReport:
    """Evaluate policies from identical checkpoints and exogenous inputs."""

    if len(policies) < 2:
        raise ValueError("Q9 utility evaluation requires at least two policies")
    if horizon <= 0:
        raise ValueError("Q9 utility horizon must be > 0")
    pool = PolicyPool(policies, config=policy_pool_config)
    baseline_id = baseline_policy_id or policies[0].policy_id
    baseline_policy = pool.get_by_id(baseline_id)
    outcomes: list[PolicyOutcome] = []
    comparisons: list[UtilityComparison] = []

    for seed in seeds:
        world = scenario_package.world_factory(int(seed))
        state = world.reset(seed=int(seed))
        pool.begin_episode(int(seed))
        for tick in range(horizon):
            focal_agent_id = _pick_agent(state, tick)
            if focal_agent_id is None:
                break
            action_space = world.legal_actions(agent_id=focal_agent_id)

            # Generate exogenous input exactly once, then checkpoint the
            # advanced exogenous RNG state. Every policy branch receives
            # the same immutable input from that checkpoint.
            generate_exogenous = getattr(world, "generate_exogenous", None)
            exogenous = (
                generate_exogenous(tick)
                if callable(generate_exogenous)
                else None
            )
            matched_checkpoint = world.checkpoint()
            exogenous_hash = _hash_exogenous(exogenous)
            tick_outcomes: dict[str, PolicyOutcome] = {}
            proposals: dict[str, ActionProposal] = {}

            for policy in policies:
                world.restore(matched_checkpoint)
                branch_state = world.observe()
                ctx = PolicyContext(
                    world=world,
                    agent_id=focal_agent_id,
                    state=branch_state,
                    action_space=action_space,
                    tick=tick,
                    rng=pool.rng_for(policy.policy_id),
                )
                proposal = policy.propose(ctx)
                if proposal is None:
                    proposal = ActionProposal(
                        agent_id=focal_agent_id,
                        action_type="noop",
                        params={},
                        proposed_at_tick=tick,
                        proposer="q9_declined",
                    )
                executed, _ = world.validate_action(proposal)
                record = world.step(executed, exogenous=exogenous)
                outcome = PolicyOutcome(
                    policy_id=policy.policy_id,
                    seed=int(seed),
                    tick=tick,
                    focal_agent_id=str(focal_agent_id),
                    state_before_hash=record.state_before_hash,
                    exogenous_hash=exogenous_hash,
                    state_after_hash=record.state_after_hash,
                    action_type=executed.action_type,
                    outcome_code=record.receipts[focal_agent_id].outcome_code,
                    utility=_extract_utility(record, focal_agent_id),
                )
                outcomes.append(outcome)
                tick_outcomes[policy.policy_id] = outcome
                proposals[policy.policy_id] = proposal

            baseline = tick_outcomes[baseline_id]
            for policy in policies:
                if policy.policy_id == baseline_id:
                    continue
                candidate = tick_outcomes[policy.policy_id]
                matched = (
                    baseline.state_before_hash == candidate.state_before_hash
                    and baseline.exogenous_hash == candidate.exogenous_hash
                    and baseline.focal_agent_id == candidate.focal_agent_id
                )
                comparisons.append(
                    UtilityComparison(
                        seed=int(seed),
                        tick=tick,
                        focal_agent_id=str(focal_agent_id),
                        baseline_policy_id=baseline_id,
                        candidate_policy_id=policy.policy_id,
                        state_before_hash=baseline.state_before_hash,
                        exogenous_hash=baseline.exogenous_hash,
                        baseline_utility=baseline.utility.scalar,
                        candidate_utility=candidate.utility.scalar,
                        delta=candidate.utility.scalar - baseline.utility.scalar,
                        matched=matched,
                    )
                )

            # Continue the reference trajectory with the already-evaluated
            # baseline proposal. No policy is invoked a second time.
            world.restore(matched_checkpoint)
            baseline_executed, _ = world.validate_action(proposals[baseline_id])
            world.step(baseline_executed, exogenous=exogenous)
            state = world.observe()

    return UtilityEvaluationReport(
        protocol_version="q9-outcome-utility-v1",
        baseline_policy_id=baseline_policy.policy_id,
        policy_ids=tuple(policy.policy_id for policy in policies),
        outcomes=tuple(outcomes),
        comparisons=tuple(comparisons),
        min_improvement=float(min_improvement),
        notes=(
            "Matched checkpoint evaluation; scalar = energy_delta + "
            "task_progress - 5*focal_death - 2*constraint_violation - "
            "non_focal_side_effect_penalty."
        ),
    )

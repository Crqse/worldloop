"""Single-episode rollout orchestrator.

Drives one world instance from ``reset`` through ``num_ticks`` steps,
delegating per-tick policy choice to a :class:`CoverageScheduler` and
recording every transition via :class:`TransitionRecorder`.

Design rules (per main plan §14.5):
- The orchestrator is the ONLY component that calls ``world.step``. This
  honors the project rule "runtime/facade.py 只做调度和 Bridge 转发，
  不写业务规则" — extended to the data pipeline.
- The orchestrator does NOT decide policy choice; it asks the coverage
  scheduler. The orchestrator does NOT decide branch timing; it asks the
  counterfactual scheduler.
- Per-tick agent selection is round-robin over alive agents (entities
  with ``alive=True`` if the column exists, else all entity ids). This
  is the simplest fair scheduler; richer selection is deferred.
- ``provenance`` on each :class:`TransitionRecord` is augmented with
  ``policy_id`` so Q4 (provenance) and Q9 (utility) can group by policy.
  The world's own provenance (e.g., ``{"seed": "..."}``) is preserved.

Phase 3 additions (per §6.3 + §6.5 of the Beta correction plan):
- **Explicit NOOP on policy decline**: when ``policy.propose(ctx)``
  returns ``None`` (decline), the orchestrator generates an explicit
  ``ActionProposal(action_type="noop", proposer="declined")``, runs it
  through ``world.validate_action`` + ``world.step``, and records the
  transition. This eliminates "hollow" ticks where ``tick_count`` would
  exceed ``transition_count`` and ensures the dataset has one
  transition per tick for deterministic replay.
- **Branch fail-closed grading**: branch scheduler exceptions are
  graded by ``RolloutConfig.run_tier``. ``"evidence"`` raises
  :class:`EvidenceFailClosedError` (the run is invalid for evidence);
  other tiers log a warning and continue (legacy behavior).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldloop_kernel import (
    ActionProposal,
    ActionSpace,
    Checkpoint,
    ExecutedAction,
    JointAction,
    StateView,
    TransitionRecord,
    WorldProtocol,
    supports_joint_actions,
)
from worldloop_kernel.recorder import RecorderManifest, TransitionRecorder

from worldloop_data.config import RolloutConfig
from worldloop_data.coverage import CoverageScheduler, UniformCoverageScheduler
from worldloop_data.counterfactual import (
    CounterfactualBranchScheduler,
    JointKernelBranchScheduler,
    NoOpBranchScheduler,
)
from worldloop_data.policy import Policy, PolicyContext, PolicyPool

__all__ = [
    "RolloutResult",
    "run_rollout",
    "run_joint_rollout",
]

_logger = logging.getLogger(__name__)

# Action type used when a policy declines to propose. The toy world's
# engine recognizes ``"noop"`` as a no-op action (energy cost NOOP_COST);
# other worlds' ``validate_action`` will return ``outcome_code='ok'`` for
# noop if it is in their legal action space, or ``outcome_code=
# 'illegal_action'`` otherwise. In either case the orchestrator records
# the transition and steps the world — the receipt's outcome_code is
# the authoritative record of what happened.
_DECLINED_NOOP_ACTION_TYPE = "noop"
_DECLINED_NOOP_PROPOSER = "declined"


# ---------------------------------------------------------------------------
# RolloutResult — what run_rollout returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutResult:
    """Result of a single-episode rollout.

    Attributes
    ----------
    episode_id:
        Stable episode identifier (e.g., ``"seed42_run0"``).
    seed:
        RNG seed used for ``world.reset``.
    output_dir:
        Directory where the recorder wrote per-tick JSON files and
        ``manifest.json``. ``None`` if ``RolloutConfig.record=False``.
    manifest:
        Recorder manifest snapshot. ``None`` if recording was disabled.
    tick_count:
        Number of ticks actually executed. May be less than
        ``config.num_ticks`` if the world ran out of alive agents.
    transition_count:
        Number of transitions recorded. Equal to ``tick_count`` since
        Phase 3 — every tick produces a transition (explicit NOOP when
        the policy declines), eliminating "hollow" ticks.
    branch_count:
        Number of counterfactual branches generated during this rollout.
    noop_count:
        Number of ticks where the policy declined and the orchestrator
        fell back to an explicit ``noop`` action. Non-zero indicates the
        policy pool could not cover every tick (e.g., LLM retries
        exhausted in EVIDENCE tier before per-decision fail-closed
        triggered, or a scripted policy had no legal action). Should be
        0 for healthy DEV/SMOKE runs with RandomPolicy.
    branch_fail_closed_count:
        Number of branch scheduler exceptions that were caught and
        suppressed (DEV/SMOKE) or would have caused evidence fail-closed
        (EVIDENCE raises before this counter increments). Non-zero in
        DEV/SMOKE indicates branch scheduler instability.
    """

    episode_id: str
    seed: int
    output_dir: Path | None
    manifest: RecorderManifest | None
    tick_count: int
    transition_count: int
    branch_count: int
    noop_count: int = 0
    branch_fail_closed_count: int = 0


# ---------------------------------------------------------------------------
# run_rollout — the orchestrator
# ---------------------------------------------------------------------------


def run_rollout(
    *,
    world: WorldProtocol,
    seed: int,
    episode_id: str,
    output_dir: Path | None,
    policy_pool: PolicyPool,
    coverage: CoverageScheduler | None = None,
    branch_scheduler: CounterfactualBranchScheduler | None = None,
    config: RolloutConfig | None = None,
    producer_id: str = "",
    producer_version: str = "0.1.0",
) -> RolloutResult:
    """Run one episode and return the :class:`RolloutResult`.

    Parameters
    ----------
    world:
        A fresh :class:`WorldProtocol` instance. ``run_rollout`` calls
        ``world.reset(seed)`` and then drives ``world.step`` for up to
        ``config.num_ticks`` ticks.
    seed:
        RNG seed for ``world.reset``.
    episode_id:
        Stable episode identifier. Used as the recorder's directory name
        and recorded into the manifest.
    output_dir:
        Directory for the recorder. If ``config.record=False``, this is
        ignored and recording is disabled (in-memory only).
    policy_pool:
        The :class:`PolicyPool` providing behavior policies.
    coverage:
        Optional :class:`CoverageScheduler`. If ``None``, a
        :class:`UniformCoverageScheduler` is used.
    branch_scheduler:
        Optional :class:`CounterfactualBranchScheduler`. If ``None``, a
        :class:`NoOpBranchScheduler` is used (no branching).
    config:
        Optional :class:`RolloutConfig`. If ``None``, defaults are used.
    producer_id:
        Producer ID for the recorder. Defaults to ``world.capabilities``
        authority label or ``"worldloop-data"``.
    producer_version:
        Producer version for the recorder.
    """
    cfg = config or RolloutConfig()
    cov = coverage or UniformCoverageScheduler()
    branches = branch_scheduler or NoOpBranchScheduler()

    # Resolve producer_id.
    if not producer_id:
        producer_id = getattr(world.capabilities, "authority", "worldloop-data")
        if hasattr(producer_id, "value"):
            producer_id = producer_id.value
        producer_id = str(producer_id)

    # Reset world.
    state = world.reset(seed=seed)

    # Set up recorder.
    recorder: TransitionRecorder | None = None
    rec_output_dir: Path | None = None
    if cfg.record and output_dir is not None:
        rec_output_dir = Path(output_dir).resolve()
        recorder = TransitionRecorder(
            output_dir=rec_output_dir,
            world_id=producer_id,
            producer_version=producer_version,
            validate=True,
        )

    # Per-agent rotation cursor (used when alive set is non-empty).
    agent_cursor = 0
    branch_count = 0
    tick_count = 0
    noop_count = 0
    branch_fail_closed_count = 0

    try:
        for tick in range(cfg.num_ticks):
            # Pick an agent for this tick.
            agent_id = _pick_agent(state, agent_cursor)
            if agent_id is None:
                # No agents left — stop early.
                break
            agent_cursor += 1

            # Get the action space for this agent.
            action_space = world.legal_actions(agent_id=agent_id)

            # Coverage scheduler picks a policy.
            policy = cov.select_policy(
                state=state,
                agent_id=agent_id,
                action_space=action_space,
                pool=policy_pool,
            )

            # Build the per-policy RNG.
            try:
                rng = policy_pool.rng_for(policy.policy_id)
            except KeyError:
                # External policy not in pool — use a per-tick RNG.
                rng = random.Random(seed + tick)

            ctx = PolicyContext(
                world=world,
                agent_id=agent_id,
                state=state,
                action_space=action_space,
                tick=tick,
                rng=rng,
            )

            # Policy proposes (may decline → explicit NOOP fallback).
            # Phase 3 §6.5: declined ticks MUST still produce a transition
            # so the dataset has one transition per tick (deterministic
            # replay, no hollow ticks). The explicit NOOP goes through
            # validate_action + step like any other action — the receipt
            # records outcome_code='ok' for worlds that recognize noop,
            # or outcome_code='illegal_action' for worlds that don't.
            proposal = policy.propose(ctx)
            policy_declined = False
            if proposal is None:
                # Policy declined — synthesize an explicit NOOP proposal
                # so world.validate_action + world.step still fire and a
                # transition is recorded. The proposal_hash ties the
                # noop ExecutedAction back to this synthetic proposal.
                proposal = ActionProposal(
                    agent_id=agent_id,
                    action_type=_DECLINED_NOOP_ACTION_TYPE,
                    params={},
                    proposed_at_tick=tick,
                    proposer=_DECLINED_NOOP_PROPOSER,
                )
                policy_declined = True
                noop_count += 1

            # World validates the proposal.
            executed, receipt = world.validate_action(proposal)

            # Take a checkpoint BEFORE step (for counterfactual branching).
            # Only when the brancher is enabled and might fire this tick.
            checkpoint: Checkpoint | None = None
            baseline_actions: list[ExecutedAction] = []
            try:
                # Defer checkpoint cost unless the brancher might fire.
                # The NoOpBranchScheduler always returns [], so we
                # don't checkpoint when brancher is NoOp.
                if not isinstance(branches, NoOpBranchScheduler):
                    branch_specs = branches.schedule_branches(
                        checkpoint=world.checkpoint(),
                        baseline_actions=[executed],
                        world=world,
                        tick=tick,
                    )
                    if branch_specs:
                        baseline_actions = [executed]
                        checkpoint = world.checkpoint()
                        branch_results = branches.execute_branches(
                            world=world,
                            checkpoint=checkpoint,
                            specs=branch_specs,
                        )
                        branch_count += len(branch_results)
            except Exception as exc:
                # Branch failure grading (Phase 3 §6.3):
                # - EVIDENCE tier: branch scheduler instability invalidates
                #   the run for evidence — raise EvidenceFailClosedError.
                # - DEV/SMOKE/SAFETY_DEMO: log warning and continue (legacy
                #   "swallow and proceed" behavior preserved).
                branch_fail_closed_count += 1
                if cfg.run_tier.lower() == "evidence":
                    from worldloop_data.telemetry import EvidenceFailClosedError

                    raise EvidenceFailClosedError(
                        f"branch scheduler raised at tick {tick} "
                        f"(episode={episode_id}, run_tier=evidence): "
                        f"{exc!r}"
                    ) from exc
                _logger.warning(
                    "branch scheduler raised at tick %s (episode=%s, "
                    "run_tier=%s) — suppressed: %r",
                    tick,
                    episode_id,
                    cfg.run_tier,
                    exc,
                )

            # World steps.
            # Generate exogenous event (if the world supports it) per
            # spec rates. The world remains pure: step only applies what
            # is passed in. This honors M5 §15.5 (e) exogenous dynamics.
            exogenous = None
            gen = getattr(world, "generate_exogenous", None)
            if callable(gen):
                try:
                    exogenous = gen(tick)
                except Exception:
                    exogenous = None
            record = world.step(executed, exogenous=exogenous)

            # Augment candidate_actions with the original ActionProposal.
            # The world's ``step()`` signature only receives the
            # ``ExecutedAction`` (per WorldProtocol), so the original
            # ``ActionProposal`` from ``policy.propose()`` is lost. We
            # restore it here so Q1 (orphan_executed = 0) and Q6
            # (action_type_counts non-empty) have data to consume.
            # Pattern matches provenance augmentation below.
            if not record.candidate_actions:
                record = dataclasses.replace(
                    record,
                    candidate_actions={executed.agent_id: proposal},
                )

            # Augment provenance with policy_id, policy_version, and
            # inference_config (per main plan §14.2: "数据集必须记录
            # policy_id、policy_version 和推理配置"). The world's own
            # provenance keys are preserved.
            augmented_provenance = dict(record.provenance)
            augmented_provenance["policy_id"] = policy.policy_id
            augmented_provenance["policy_version"] = getattr(
                policy, "policy_version", "unknown"
            )
            augmented_provenance["inference_config"] = dict(
                getattr(policy, "inference_config", {})
            )
            augmented_provenance["episode_id"] = episode_id
            # Phase 3: record policy_declined flag so downstream Q6
            # (coverage) can distinguish "noop chosen by policy" from
            # "noop synthesized because policy declined". This is
            # critical for evidence-tier audits where high decline rates
            # may indicate LLM backend issues.
            augmented_provenance["policy_declined"] = policy_declined
            record = dataclasses.replace(record, provenance=augmented_provenance)

            # Record.
            if recorder is not None:
                recorder.append(record)

            # Coverage observation.
            cov.record_observation(record)

            # Track policy usage (the coverage scheduler's own counters
            # don't see this; we expose it via a helper call).
            if hasattr(cov, "_note_policy_use"):
                cov._note_policy_use(policy.policy_id)

            # Observe the new state for the next tick.
            state = world.observe()
            tick_count += 1
    finally:
        if recorder is not None:
            recorder.close()

    manifest = recorder.manifest() if recorder is not None else None

    return RolloutResult(
        episode_id=episode_id,
        seed=seed,
        output_dir=rec_output_dir,
        manifest=manifest,
        tick_count=tick_count,
        transition_count=manifest.record_count if manifest else tick_count,
        branch_count=branch_count,
        noop_count=noop_count,
        branch_fail_closed_count=branch_fail_closed_count,
    )


# ---------------------------------------------------------------------------
# run_joint_rollout — joint action mode orchestrator (Phase 5 §10/§12)
# ---------------------------------------------------------------------------


def run_joint_rollout(
    *,
    world: Any,
    seed: int,
    episode_id: str,
    output_dir: Path | None,
    policy_pool: PolicyPool,
    coverage: CoverageScheduler | None = None,
    branch_scheduler: JointKernelBranchScheduler | None = None,
    config: RolloutConfig | None = None,
    producer_id: str = "",
    producer_version: str = "0.1.0",
) -> RolloutResult:
    """Run one episode in JOINT action mode and return the result.

    Differences from :func:`run_rollout` (sequential focal mode):

    - EVERY active agent proposes each tick; the world executes ONE
      parallel step via ``world.step_joint`` (no focal+STAY filling).
    - The coverage scheduler picks ONE policy per tick which is applied
      to ALL active agents that tick — this keeps the ``policy_id``
      provenance semantics that Q4/Q9 group by (one policy per record).
    - Per-agent decline: agents whose policy declines are simply
      OMITTED from ``proposals_by_agent``; the world's
      ``missing_agent_policy="stay"`` synthesizes an explicit STAY
      proposal/receipt for them (mirrors the sequential explicit-NOOP
      rule — no hollow agents). ``noop_count`` counts declined AGENTS
      (not ticks) in joint mode.
    - The episode stops when the env's active agent set is empty
      (all terminated/truncated) — no proposals are generated for
      vanished agents (E-G3).
    - Branching uses :class:`JointKernelBranchScheduler` (focal agent's
      action varies, non-focal actions mechanically replayed).

    The world must satisfy the kernel ``JointActionWorld`` protocol
    and expose ``active_agents()`` (the PettingZoo adapter does).
    """
    if not supports_joint_actions(world):
        raise TypeError(
            "run_joint_rollout requires a world supporting the kernel "
            "JointActionWorld protocol (validate_joint_action + "
            "step_joint)"
        )
    active_agents_fn = getattr(world, "active_agents", None)
    if not callable(active_agents_fn):
        raise TypeError(
            "run_joint_rollout requires the world to expose "
            "active_agents() so the orchestrator can build "
            "JointAction.active_agents per tick"
        )

    cfg = config or RolloutConfig()
    cov = coverage or UniformCoverageScheduler()
    brancher = branch_scheduler

    if not producer_id:
        producer_id = getattr(world.capabilities, "authority", "worldloop-data")
        if hasattr(producer_id, "value"):
            producer_id = producer_id.value
        producer_id = str(producer_id)

    state = world.reset(seed=seed)

    recorder: TransitionRecorder | None = None
    rec_output_dir: Path | None = None
    if cfg.record and output_dir is not None:
        rec_output_dir = Path(output_dir).resolve()
        recorder = TransitionRecorder(
            output_dir=rec_output_dir,
            world_id=producer_id,
            producer_version=producer_version,
            validate=True,
        )

    branch_count = 0
    tick_count = 0
    noop_count = 0
    branch_fail_closed_count = 0

    try:
        for tick in range(cfg.num_ticks):
            active = [str(a) for a in active_agents_fn()]
            if not active:
                # All agents terminated/truncated — stop; no proposals
                # for vanished agents (E-G3).
                break

            # ONE policy per tick, applied to all active agents.
            lead_space = world.legal_actions(agent_id=active[0])
            policy = cov.select_policy(
                state=state,
                agent_id=active[0],
                action_space=lead_space,
                pool=policy_pool,
            )
            try:
                rng = policy_pool.rng_for(policy.policy_id)
            except KeyError:
                rng = random.Random(seed + tick)

            # Every active agent proposes; declines are omitted and the
            # world synthesizes STAY per missing_agent_policy.
            proposals: dict[str | int, ActionProposal] = {}
            declined_agents: list[str] = []
            for agent_id in active:
                action_space = (
                    lead_space
                    if agent_id == active[0]
                    else world.legal_actions(agent_id=agent_id)
                )
                ctx = PolicyContext(
                    world=world,
                    agent_id=agent_id,
                    state=state,
                    action_space=action_space,
                    tick=tick,
                    rng=rng,
                )
                proposal = policy.propose(ctx)
                if proposal is None:
                    declined_agents.append(agent_id)
                    noop_count += 1
                    continue
                proposals[agent_id] = proposal

            joint = JointAction(
                tick=tick,
                active_agents=tuple(active),
                proposals_by_agent=proposals,
                missing_agent_policy="stay",
            )
            executed_joint, _joint_receipt = world.validate_joint_action(
                joint
            )

            # Counterfactual branching (joint held-fixed semantics).
            if brancher is not None:
                try:
                    branch_specs = brancher.schedule_joint_branches(
                        checkpoint=world.checkpoint(),
                        baseline_joint=executed_joint,
                        world=world,
                        tick=tick,
                    )
                    if branch_specs:
                        checkpoint = world.checkpoint()
                        branch_results = brancher.execute_joint_branches(
                            world=world,
                            checkpoint=checkpoint,
                            specs=branch_specs,
                        )
                        branch_count += len(branch_results)
                except Exception as exc:
                    # Same fail-closed grading as run_rollout (§6.3).
                    branch_fail_closed_count += 1
                    if cfg.run_tier.lower() == "evidence":
                        from worldloop_data.telemetry import (
                            EvidenceFailClosedError,
                        )

                        raise EvidenceFailClosedError(
                            f"joint branch scheduler raised at tick {tick} "
                            f"(episode={episode_id}, run_tier=evidence): "
                            f"{exc!r}"
                        ) from exc
                    _logger.warning(
                        "joint branch scheduler raised at tick %s "
                        "(episode=%s, run_tier=%s) — suppressed: %r",
                        tick,
                        episode_id,
                        cfg.run_tier,
                        exc,
                    )

            # Exogenous hook (same contract as run_rollout).
            exogenous = None
            gen = getattr(world, "generate_exogenous", None)
            if callable(gen):
                try:
                    exogenous = gen(tick)
                except Exception:
                    exogenous = None

            record = world.step_joint(executed_joint, exogenous=exogenous)

            # Provenance augmentation (mirrors run_rollout).
            augmented_provenance = dict(record.provenance)
            augmented_provenance["policy_id"] = policy.policy_id
            augmented_provenance["policy_version"] = getattr(
                policy, "policy_version", "unknown"
            )
            augmented_provenance["inference_config"] = dict(
                getattr(policy, "inference_config", {})
            )
            augmented_provenance["episode_id"] = episode_id
            augmented_provenance["policy_declined"] = bool(declined_agents)
            augmented_provenance["declined_agents"] = json.dumps(
                sorted(declined_agents)
            )
            record = dataclasses.replace(
                record, provenance=augmented_provenance
            )

            if recorder is not None:
                recorder.append(record)

            cov.record_observation(record)
            if hasattr(cov, "_note_policy_use"):
                cov._note_policy_use(policy.policy_id)

            state = world.observe()
            tick_count += 1
    finally:
        if recorder is not None:
            recorder.close()

    manifest = recorder.manifest() if recorder is not None else None

    return RolloutResult(
        episode_id=episode_id,
        seed=seed,
        output_dir=rec_output_dir,
        manifest=manifest,
        tick_count=tick_count,
        transition_count=manifest.record_count if manifest else tick_count,
        branch_count=branch_count,
        noop_count=noop_count,
        branch_fail_closed_count=branch_fail_closed_count,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_agent(state: StateView, cursor: int) -> str | int | None:
    """Pick the next agent to act.

    Selection order:
    1. If the entity table has an ``alive`` column, filter to alive
       agents and pick ``alive[cursor % len(alive)]``.
    2. Otherwise, pick ``ids[cursor % len(ids)]``.
    3. If the entity table is empty, return ``None``.
    """
    ids = state.entities.ids
    if not ids:
        return None
    columns = state.entities.columns
    if "alive" in columns:
        alive_ids = [
            eid for eid, alive in zip(ids, columns["alive"]) if alive
        ]
        if not alive_ids:
            return None
        return alive_ids[cursor % len(alive_ids)]
    return ids[cursor % len(ids)]

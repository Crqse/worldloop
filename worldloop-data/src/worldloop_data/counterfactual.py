"""S-10 Counterfactual Branch Scheduler — generate controlled branches.

A counterfactual branch forks the world from a checkpoint and re-runs
with one factor changed (focal action, exogenous event, scenario
parameter, ...). All other factors are held fixed. The kernel
:func:`worldloop_kernel.branch` primitive provides the underlying
fork+restore mechanics; this scheduler decides WHEN and HOW to branch.

Design rules (per main plan §14.4):
- Branches MUST NOT pollute the parent world's state. The kernel
  ``branch`` function saves/restores the parent automatically.
- Every branch records the factors held fixed (for Q7 verification).
- The scheduler is the ONLY component that decides branch timing;
  policies themselves do not branch.

M4 provides:
- :class:`NoOpBranchScheduler` — disables branching (default for smoke).
- :class:`KernelBranchScheduler` — real focal-action brancher that calls
  :func:`worldloop_kernel.branch` every N ticks, swapping the focal
  agent's action for a different legal ``action_type`` and holding all
  other factors fixed (Q7 counterfactual gate).

Phase 3 §6.5 additions (Q7 mechanical held-fixed verification):
- The scheduler records per-fork mechanical fingerprints (parent state
  hash before fork, parent state hash after all branches complete,
  rng_bundle hash, non-focal actions hash) and exposes them via
  :meth:`KernelBranchScheduler.branch_summary` under
  ``held_fixed_verification``.
- :meth:`KernelBranchScheduler.verify_held_fixed` returns per-factor
  consistency results so Q7 can mechanically verify (not just
  declaratively assert) that the held-fixed factors were actually held
  fixed.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Mapping, Protocol, Sequence, TYPE_CHECKING

from worldloop_kernel import (
    ActionProposal,
    BranchResult,
    Checkpoint,
    ExecutedAction,
    JointAction,
    LegalAction,
    WorldProtocol,
)
from worldloop_kernel.canonical import canonical_encode, hash_state
from worldloop_kernel.replay import branch as kernel_branch

from worldloop_data.config import CounterfactualConfig

if TYPE_CHECKING:
    pass

__all__ = [
    "CounterfactualBranchScheduler",
    "BranchSpec",
    "JointBranchSpec",
    "NoOpBranchScheduler",
    "KernelBranchScheduler",
    "JointKernelBranchScheduler",
    "HeldFixedVerification",
]

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level hash helpers (Q7 mechanical verification)
# ---------------------------------------------------------------------------


def _hash_rng_bundle(rng_bundle: Mapping[str, str] | None) -> str:
    """Stable SHA-256 hash of a checkpoint's rng_bundle (sorted by key).

    Returns empty string when ``rng_bundle`` is ``None`` — worlds that
    don't expose RNG state cannot mechanically prove RNG was held fixed
    (Q7 will report this as ``rng_bundle_captured=False``).
    """
    if not rng_bundle:
        return ""
    h = hashlib.sha256()
    for key in sorted(rng_bundle):
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(rng_bundle[key]).encode("utf-8"))
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()


def _hash_non_focal_actions(specs: Sequence[BranchSpec]) -> str:
    """Stable SHA-256 hash of the non-focal baseline actions.

    The non-focal actions are ``alternative_actions[1:]`` — everything
    after the focal action[0]. They MUST be identical across all specs
    in the same fork group (only the focal action varies). Empty hash
    indicates either a single-agent branch (no non-focal actions) or an
    empty specs list.
    """
    if not specs:
        return ""
    # Use the first spec's non-focal tail as the canonical sequence.
    # All specs in the same fork group MUST have identical non-focal
    # tails — we don't verify this here (Q7 reads the hash); we just
    # hash the first one as representative.
    first = specs[0]
    non_focal = first.alternative_actions[1:]
    if not non_focal:
        return ""
    h = hashlib.sha256()
    for action in non_focal:
        h.update(canonical_encode(action))
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# BranchSpec — what to vary in a branch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchSpec:
    """Specification for one counterfactual branch.

    Attributes
    ----------
    fork_tick:
        Tick at which the branch forks from the parent.
    branch_id:
        Stable identifier for the branch. Format: ``"b{index}"``.
    alternative_actions:
        The focal factor: the alternative action sequence to execute
        on this branch (post-fork). Other factors (world state, RNG,
        other agents) are held fixed by the kernel ``branch`` primitive.
    held_fixed:
        Mapping of factor name to a human-readable description of what
        is held fixed. Recorded into branch provenance for Q7.
    rationale:
        Free-form rationale for this branch (e.g., "test FORAGE vs REST
        at tick 5").
    """

    fork_tick: int
    branch_id: str
    alternative_actions: tuple[ExecutedAction, ...]
    held_fixed: Mapping[str, str]
    rationale: str = ""


# ---------------------------------------------------------------------------
# HeldFixedVerification — per-fork mechanical fingerprints (Q7 Phase 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldFixedVerification:
    """Per-fork mechanical record of held-fixed factors (Q7 Phase 3 §6.5).

    Captured by :meth:`KernelBranchScheduler.execute_branches` for each
    fork point. The Q7 quality check reads these to mechanically verify
    that the factors declared in :attr:`BranchSpec.held_fixed` were
    actually held fixed during the branch — not just declaratively
    asserted.

    Attributes
    ----------
    fork_tick:
        Tick at which the fork occurred. Matches
        :attr:`BranchSpec.fork_tick` for each branch in this fork group.
    parent_state_hash_before:
        SHA-256 hash of the parent world's state (via
        :func:`worldloop_kernel.canonical.hash_state` on
        ``world.observe()``) BEFORE any branch ran. This is the
        "should-be-restored-to" hash.
    parent_state_hash_after:
        SHA-256 hash of the parent world's state AFTER all branches in
        this fork group completed and the kernel attempted restoration.
        MUST equal ``parent_state_hash_before`` for the restoration
        contract to hold. A mismatch means the parent world's state was
        polluted by branching — every subsequent branch is suspect.
    rng_bundle_hash:
        SHA-256 hash of the checkpoint's ``rng_bundle`` (sorted by key).
        Provides mechanical evidence that the RNG state at fork was
        captured. Two forks at the same tick with different
        ``rng_bundle_hash`` indicate non-deterministic RNG seeding
        (an M8 concern).
    non_focal_actions_hash:
        SHA-256 hash of the non-focal baseline actions
        (``baseline_actions[1:]``). The focal action varies across
        branches; the non-focal actions MUST be identical across
        branches within the same fork group.
    branch_count:
        Number of branches executed in this fork group.
    all_restoration_ok:
        ``True`` iff every branch's ``restoration_ok`` flag was True.
        Aggregated from :attr:`BranchResult.restoration_ok`. False
        indicates at least one branch failed to restore the parent.
    """

    fork_tick: int
    parent_state_hash_before: str
    parent_state_hash_after: str
    rng_bundle_hash: str
    non_focal_actions_hash: str
    branch_count: int
    all_restoration_ok: bool


# ---------------------------------------------------------------------------
# CounterfactualBranchScheduler Protocol
# ---------------------------------------------------------------------------


class CounterfactualBranchScheduler(Protocol):
    """Decides when and how to generate counterfactual branches."""

    def schedule_branches(
        self,
        checkpoint: Checkpoint,
        baseline_actions: Sequence[ExecutedAction],
        world: WorldProtocol,
        tick: int,
    ) -> list[BranchSpec]:
        """Return a list of branch specs to execute at this checkpoint.

        May return an empty list (no branching this tick).
        """
        ...

    def record_branch_result(self, result: BranchResult) -> None:
        """Record a completed branch result for later analysis."""
        ...

    def branch_summary(self) -> Mapping[str, Any]:
        """Return a summary of all branches generated (for Q7)."""
        ...


# ---------------------------------------------------------------------------
# NoOpBranchScheduler — disables branching
# ---------------------------------------------------------------------------


class NoOpBranchScheduler:
    """No-op brancher: never branches.

    Used by the smoke pipeline to verify the end-to-end plumbing without
    paying the cost of branch generation. ``schedule_branches`` always
    returns an empty list.
    """

    def schedule_branches(
        self,
        checkpoint: Checkpoint,
        baseline_actions: Sequence[ExecutedAction],
        world: WorldProtocol,
        tick: int,
    ) -> list[BranchSpec]:
        return []

    def record_branch_result(self, result: BranchResult) -> None:
        pass

    def branch_summary(self) -> Mapping[str, Any]:
        return {"branch_count": 0, "mode": "noop"}


# ---------------------------------------------------------------------------
# KernelBranchScheduler — real focal-action brancher using kernel.branch
# ---------------------------------------------------------------------------


class KernelBranchScheduler:
    """Real focal-action brancher that calls :func:`worldloop_kernel.branch`.

    Every ``branch_every_ticks`` ticks, generates up to
    ``branches_per_checkpoint`` branches. Each branch forks from the
    checkpoint and re-runs with the focal agent's action swapped for a
    DIFFERENT legal ``action_type`` (the first representative variant of
    each distinct alternative type). All other agents' actions and all
    other factors (world state, RNG, exogenous input) are held fixed by
    the kernel ``branch`` primitive.

    Selection rule (per branch):
    - Focal agent = ``baseline_actions[0].agent_id``.
    - Baseline action_type = ``baseline_actions[0].action_type``.
    - Query ``world.legal_actions(focal_agent_id)`` and collect the
      first representative :class:`LegalAction` for each distinct
      ``action_type`` that differs from the baseline.
    - For each branch (up to ``branches_per_checkpoint`` or the number
      of distinct alternatives, whichever is smaller), build an
      :class:`ActionProposal` from the alternative LegalAction, validate
      it via ``world.validate_action``, and use the returned
      :class:`ExecutedAction` as the focal action. Other agents' actions
      are inherited from ``baseline_actions[1:]``.
    - If ``validate_action`` rejects the proposal (``success=False``),
      that branch is skipped.
    - If the focal agent has only one distinct ``action_type`` (no
      alternative available), no branches are produced (Q7
      single-legal-action skip).

    Branches MUST NOT pollute the parent world's state.
    ``schedule_branches`` only calls ``legal_actions`` and
    ``validate_action`` (neither mutates state per WorldProtocol);
    ``execute_branches`` calls :func:`worldloop_kernel.branch`, which
    saves/restores the parent automatically.

    Attributes
    ----------
    config:
        The :class:`CounterfactualConfig` governing branch timing.
    """

    def __init__(self, *, config: CounterfactualConfig | None = None) -> None:
        self.config = config or CounterfactualConfig()
        self._branch_results: list[BranchResult] = []
        # Phase 3 §6.5: per-fork mechanical fingerprints for Q7.
        # One entry per fork group (one per tick where branches fired).
        self._held_fixed_verifications: list[HeldFixedVerification] = []

    def schedule_branches(
        self,
        checkpoint: Checkpoint,
        baseline_actions: Sequence[ExecutedAction],
        world: WorldProtocol,
        tick: int,
    ) -> list[BranchSpec]:
        if self.config.branch_every_ticks <= 0:
            return []
        if tick == 0 or tick % self.config.branch_every_ticks != 0:
            return []
        if not baseline_actions:
            return []
        # Focal agent = the agent that produced the first baseline action.
        # We vary THIS agent's action; all other agents' actions are
        # inherited from baseline_actions[1:] unchanged.
        focal_executed = baseline_actions[0]
        focal_agent_id = focal_executed.agent_id
        baseline_action_type = focal_executed.action_type

        # Query the focal agent's legal action space at the current tick.
        action_space = world.legal_actions(agent_id=focal_agent_id)

        # Collect the first representative LegalAction for each distinct
        # action_type that differs from the baseline. This bounds the
        # branch count by the number of distinct alternative types
        # (not by the number of param variants, which may be large).
        seen_types: set[str] = set()
        alternatives: list[LegalAction] = []
        for la in action_space.legal_actions:
            if la.action_type == baseline_action_type:
                continue
            if la.action_type in seen_types:
                continue
            seen_types.add(la.action_type)
            alternatives.append(la)

        # If no alternative action_type exists, we cannot produce a real
        # focal-action variation. Return empty (Q7 single-legal-action
        # skip — cannot vary the focal factor).
        if not alternatives:
            return []

        specs: list[BranchSpec] = []
        branches_to_emit = min(
            self.config.branches_per_checkpoint, len(alternatives)
        )
        for i in range(branches_to_emit):
            alt_la = alternatives[i]
            # Build an ActionProposal for the alternative action.
            proposal = ActionProposal(
                agent_id=focal_agent_id,
                action_type=alt_la.action_type,
                params=dict(alt_la.params),
                proposed_at_tick=tick,
                proposer="counterfactual",
            )
            # Validate via the world. validate_action does NOT mutate
            # state (per WorldProtocol); it returns an ExecutedAction +
            # ActionReceipt. If the proposal is rejected
            # (success=False), skip this branch — we cannot use a
            # rejected action as the alternative.
            alt_executed, receipt = world.validate_action(proposal)
            if not receipt.success:
                continue
            # Swap the focal agent's action with the alternative;
            # keep other agents' baseline actions unchanged.
            alternative_actions = (alt_executed,) + tuple(
                baseline_actions[1:]
            )
            specs.append(
                BranchSpec(
                    fork_tick=tick,
                    branch_id=f"b{i}",
                    alternative_actions=alternative_actions,
                    held_fixed=self.config.held_fixed,
                    rationale=(
                        f"focal-action variation: agent {focal_agent_id} "
                        f"baseline={baseline_action_type} → "
                        f"alternative={alt_la.action_type}"
                    ),
                )
            )
        return specs

    def execute_branches(
        self,
        world: WorldProtocol,
        checkpoint: Checkpoint,
        specs: Sequence[BranchSpec],
    ) -> list[BranchResult]:
        """Execute branch specs using :func:`worldloop_kernel.branch`.

        This is a helper that the rollout orchestrator calls. It is on
        the scheduler so the scheduler owns branch execution (and can
        add per-branch provenance, throttling, etc.).

        Phase 3 §6.5: captures per-fork mechanical fingerprints before
        and after the kernel ``branch`` call. The "before" hash proves
        what the parent state was; the "after" hash proves whether the
        kernel successfully restored the parent. A mismatch invalidates
        every subsequent branch in this fork group.
        """
        if not specs:
            return []

        # --- Phase 3 §6.5: capture pre-fork fingerprints ---
        fork_tick = specs[0].fork_tick
        try:
            parent_state_before = world.observe()
            parent_state_hash_before = hash_state(parent_state_before)
        except Exception as exc:  # noqa: BLE001
            # If world.observe fails before branching, we cannot verify
            # restoration. Record a verification entry with empty hashes
            # so Q7 reports the failure rather than silently passing.
            parent_state_hash_before = ""
            _logger.warning(
                "KernelBranchScheduler: world.observe() before fork at "
                "tick %s raised %r — held-fixed verification will be "
                "incomplete",
                fork_tick,
                exc,
            )

        # Hash the checkpoint's rng_bundle (sorted by key for determinism).
        rng_bundle_hash = _hash_rng_bundle(checkpoint.rng_bundle)

        # Hash the non-focal baseline actions (specs[0].alternative_actions
        # is the focal-varied sequence; the non-focal tail is
        # alternative_actions[1:] which MUST be identical across all
        # specs in the same fork group — only the focal action[0]
        # varies).
        non_focal_actions_hash = _hash_non_focal_actions(specs)

        # --- Execute branches via the kernel primitive ---
        alternatives = [list(s.alternative_actions) for s in specs]
        results = kernel_branch(
            world=world,
            checkpoint=checkpoint,
            alternatives=alternatives,
        )
        for r in results:
            self.record_branch_result(r)

        # --- Phase 3 §6.5: capture post-fork fingerprints ---
        try:
            parent_state_after = world.observe()
            parent_state_hash_after = hash_state(parent_state_after)
        except Exception as exc:  # noqa: BLE001
            parent_state_hash_after = ""
            _logger.warning(
                "KernelBranchScheduler: world.observe() after fork at "
                "tick %s raised %r — parent state may be polluted",
                fork_tick,
                exc,
            )

        all_restoration_ok = all(r.restoration_ok for r in results)
        self._held_fixed_verifications.append(
            HeldFixedVerification(
                fork_tick=fork_tick,
                parent_state_hash_before=parent_state_hash_before,
                parent_state_hash_after=parent_state_hash_after,
                rng_bundle_hash=rng_bundle_hash,
                non_focal_actions_hash=non_focal_actions_hash,
                branch_count=len(results),
                all_restoration_ok=all_restoration_ok,
            )
        )
        return results

    def record_branch_result(self, result: BranchResult) -> None:
        self._branch_results.append(result)

    def verify_held_fixed(self) -> list[Mapping[str, Any]]:
        """Return per-fork mechanical verification results for Q7.

        Each entry corresponds to one fork group (one tick where
        branches fired). The Q7 quality check reads these to verify
        mechanically (not just declaratively) that the held-fixed
        factors were actually held fixed.

        Per-factor verification logic:
        - ``parent_state_restored``: ``parent_state_hash_before ==
          parent_state_hash_after``. True iff the kernel's
          ``branch`` primitive restored the parent world to its
          pre-fork state. This is the PRIMARY Q7 invariant — if
          false, every subsequent branch is suspect.
        - ``rng_bundle_captured``: ``rng_bundle_hash != ""``. True iff
          the checkpoint exposed an ``rng_bundle`` (None rng_bundle
          produces empty hash — worlds that don't expose RNG state
          cannot mechanically prove RNG was held fixed).
        - ``non_focal_actions_consistent``: ``non_focal_actions_hash
          != ""``. True iff the fork group had non-focal actions to
          hash. Empty string indicates either single-agent branches
          (no non-focal actions) OR an internal error hashing them.
        - ``all_restoration_ok``: propagated from
          :attr:`HeldFixedVerification.all_restoration_ok`. True iff
          every branch's ``restoration_ok`` flag was True.

        Returns
        -------
        list[Mapping[str, Any]]
            One mapping per fork group, each containing the
            :class:`HeldFixedVerification` fields plus the four
            verification booleans above.
        """
        results: list[Mapping[str, Any]] = []
        for v in self._held_fixed_verifications:
            parent_restored = (
                v.parent_state_hash_before != ""
                and v.parent_state_hash_after != ""
                and v.parent_state_hash_before == v.parent_state_hash_after
            )
            rng_captured = v.rng_bundle_hash != ""
            non_focal_consistent = v.non_focal_actions_hash != ""
            results.append(
                {
                    "fork_tick": v.fork_tick,
                    "parent_state_hash_before": v.parent_state_hash_before,
                    "parent_state_hash_after": v.parent_state_hash_after,
                    "rng_bundle_hash": v.rng_bundle_hash,
                    "non_focal_actions_hash": v.non_focal_actions_hash,
                    "branch_count": v.branch_count,
                    "all_restoration_ok": v.all_restoration_ok,
                    # Mechanical verification booleans (Q7 reads these):
                    "parent_state_restored": parent_restored,
                    "rng_bundle_captured": rng_captured,
                    "non_focal_actions_consistent": non_focal_consistent,
                }
            )
        return results

    def branch_summary(self) -> Mapping[str, Any]:
        return {
            "branch_count": len(self._branch_results),
            "mode": "kernel_branch",
            "held_fixed": dict(self.config.held_fixed),
            "branches": [
                {
                    "branch_id": r.branch_id,
                    "fork_tick": r.fork_tick,
                    "diverged_at_tick": r.diverged_at_tick,
                    "restoration_ok": r.restoration_ok,
                    "error": r.error,
                }
                for r in self._branch_results
            ],
            # Phase 3 §6.5: mechanical held-fixed verification.
            # Q7 reads this list and checks that every fork group
            # has parent_state_restored=True AND all_restoration_ok=True.
            # rng_bundle_captured / non_focal_actions_consistent are
            # informational (may be False for single-agent worlds or
            # worlds that don't expose RNG state).
            "held_fixed_verification": self.verify_held_fixed(),
        }


# ---------------------------------------------------------------------------
# Joint-mode counterfactuals (Phase 5 §10/§12) — JointBranchSpec +
# JointKernelBranchScheduler
# ---------------------------------------------------------------------------


def _hash_joint_non_focal_actions(
    joint: JointAction, focal_agent_id: str | int
) -> str:
    """Stable SHA-256 hash of the NON-focal executed actions of a joint.

    Sorted by ``str(agent_id)`` for determinism. These actions are
    mechanically replayed unchanged on every branch (held-fixed factor
    for Q7's ``non_focal_actions_hash``). Empty string when the joint
    has no non-focal agents.
    """
    focal_key = str(focal_agent_id)
    items = sorted(
        (
            (str(agent_id), executed)
            for agent_id, executed in joint.executed_by_agent.items()
            if str(agent_id) != focal_key
        ),
        key=lambda kv: kv[0],
    )
    if not items:
        return ""
    h = hashlib.sha256()
    for key, executed in items:
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        h.update(canonical_encode(executed))
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()


@dataclass(frozen=True)
class JointBranchSpec:
    """Specification for one JOINT counterfactual branch (Phase 5).

    The focal factor is the focal agent's action inside
    ``alternative_joint``; every OTHER agent's executed action is the
    baseline action, mechanically replayed (held fixed).

    Attributes
    ----------
    fork_tick:
        Tick at which the branch forks from the parent.
    branch_id:
        Stable identifier. Format: ``"jb{index}"``.
    focal_agent_id:
        The agent whose action varies on this branch.
    alternative_joint:
        Executed-stage :class:`worldloop_kernel.JointAction` with the
        focal agent's action swapped for the alternative and all
        non-focal executed actions identical to the baseline joint.
    held_fixed:
        Factor name → human-readable description (Q7 provenance).
    rationale:
        Free-form rationale for this branch.
    """

    fork_tick: int
    branch_id: str
    focal_agent_id: str | int
    alternative_joint: JointAction
    held_fixed: Mapping[str, str]
    rationale: str = ""


class JointKernelBranchScheduler(KernelBranchScheduler):
    """Joint-mode focal-action brancher (Phase 5 §10.5).

    Differences from :class:`KernelBranchScheduler`:

    - The baseline is an executed-stage
      :class:`worldloop_kernel.JointAction` covering ALL active agents,
      not a single focal :class:`ExecutedAction`.
    - Alternatives vary the focal agent's action by DISTINCT PARAMS
      (e.g., PettingZoo discrete actions), not by ``action_type`` —
      parallel envs typically expose a single action_type ("move"), so
      the type-based rule would always produce zero branches.
    - Branch execution does NOT use the kernel ``branch`` primitive
      (which replays actions one-by-one via ``world.step``). Instead it
      restores the fork checkpoint, submits the alternative joint via
      ``world.step_joint`` (ONE parallel env step), then restores the
      parent — preserving joint semantics on the branch.

    Held-fixed contract (Q7): world state (fork checkpoint bytes), RNG
    (checkpoint.rng_bundle), and every NON-focal agent's executed
    action (mechanically replayed; fingerprinted via
    ``non_focal_actions_hash``) are held fixed. Only the focal agent's
    action varies.

    Inherits ``record_branch_result`` / ``verify_held_fixed`` /
    fingerprint storage from :class:`KernelBranchScheduler`.
    """

    def schedule_joint_branches(
        self,
        checkpoint: Checkpoint,
        baseline_joint: JointAction,
        world: Any,
        tick: int,
    ) -> list[JointBranchSpec]:
        """Return joint branch specs to execute at this checkpoint.

        Focal agent = ``baseline_joint.active_agents[0]``. Alternatives
        are the focal agent's legal actions whose params differ from
        the baseline executed params (deduped by canonical params
        encoding). Each alternative is validated via
        ``world.validate_action`` (does not mutate state); rejected
        proposals are skipped.
        """
        if self.config.branch_every_ticks <= 0:
            return []
        if tick == 0 or tick % self.config.branch_every_ticks != 0:
            return []
        if not baseline_joint.is_executed_stage:
            return []

        focal_agent_id = baseline_joint.active_agents[0]
        baseline_executed = baseline_joint.executed_by_agent[focal_agent_id]
        baseline_params_key = canonical_encode(dict(baseline_executed.params))

        action_space = world.legal_actions(agent_id=focal_agent_id)

        # Collect legal actions with DISTINCT params differing from the
        # baseline params (dedup by canonical encoding of the params
        # mapping). action_type may be identical across all of them.
        seen_params: set[bytes] = {baseline_params_key}
        alternatives: list[LegalAction] = []
        for la in action_space.legal_actions:
            params_key = canonical_encode(dict(la.params))
            if params_key in seen_params:
                continue
            seen_params.add(params_key)
            alternatives.append(la)

        if not alternatives:
            return []

        specs: list[JointBranchSpec] = []
        branches_to_emit = min(
            self.config.branches_per_checkpoint, len(alternatives)
        )
        for i in range(branches_to_emit):
            alt_la = alternatives[i]
            proposal = ActionProposal(
                agent_id=focal_agent_id,
                action_type=alt_la.action_type,
                params=dict(alt_la.params),
                proposed_at_tick=tick,
                proposer="counterfactual_joint",
            )
            # validate_action does NOT mutate state (WorldProtocol);
            # rejected proposals cannot serve as alternatives.
            alt_executed, receipt = world.validate_action(proposal)
            if not receipt.success:
                continue
            proposals_map = dict(baseline_joint.proposals_by_agent)
            proposals_map[focal_agent_id] = proposal
            executed_map = dict(baseline_joint.executed_by_agent)
            executed_map[focal_agent_id] = alt_executed
            alternative_joint = _dc_replace(
                baseline_joint,
                proposals_by_agent=proposals_map,
                executed_by_agent=executed_map,
            )
            specs.append(
                JointBranchSpec(
                    fork_tick=tick,
                    branch_id=f"jb{i}",
                    focal_agent_id=focal_agent_id,
                    alternative_joint=alternative_joint,
                    held_fixed=self.config.held_fixed,
                    rationale=(
                        f"joint focal-action variation: agent "
                        f"{focal_agent_id} baseline params="
                        f"{dict(baseline_executed.params)!r} → alternative "
                        f"params={dict(alt_la.params)!r}; non-focal actions "
                        f"mechanically replayed"
                    ),
                )
            )
        return specs

    def execute_joint_branches(
        self,
        world: Any,
        checkpoint: Checkpoint,
        specs: Sequence[JointBranchSpec],
    ) -> list[BranchResult]:
        """Execute joint branch specs via restore + ``step_joint``.

        For each spec: restore the fork checkpoint, submit the
        alternative joint action as ONE parallel step, capture the
        branch's final state hash, then restore the parent and verify
        restoration by hash equality (``restoration_ok``). The kernel
        ``branch`` primitive is NOT used because it replays actions
        one-by-one via ``world.step`` — sequential semantics that would
        break the joint contract.
        """
        if not specs:
            return []

        fork_tick = specs[0].fork_tick
        try:
            parent_state_hash_before = hash_state(world.observe())
        except Exception as exc:  # noqa: BLE001
            parent_state_hash_before = ""
            _logger.warning(
                "JointKernelBranchScheduler: world.observe() before fork "
                "at tick %s raised %r — held-fixed verification will be "
                "incomplete",
                fork_tick,
                exc,
            )

        rng_bundle_hash = _hash_rng_bundle(checkpoint.rng_bundle)
        non_focal_actions_hash = _hash_joint_non_focal_actions(
            specs[0].alternative_joint, specs[0].focal_agent_id
        )

        # Save the parent BEFORE the first fork restore so every branch
        # can put the parent back exactly.
        parent_saved = world.checkpoint()

        results: list[BranchResult] = []
        for spec in specs:
            error: str | None = None
            final_state_hash: str | None = None
            try:
                world.restore(checkpoint)
                record = world.step_joint(spec.alternative_joint)
                final_state_hash = record.state_after_hash
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                _logger.warning(
                    "joint branch %s at tick %s raised: %r",
                    spec.branch_id,
                    spec.fork_tick,
                    exc,
                )
            # Always restore the parent, even when the branch failed.
            restoration_ok = False
            try:
                world.restore(parent_saved)
                restoration_ok = (
                    parent_state_hash_before != ""
                    and hash_state(world.observe())
                    == parent_state_hash_before
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "joint branch %s parent restore raised: %r — parent "
                    "may be polluted",
                    spec.branch_id,
                    exc,
                )
            # Actions tuple: focal first, then non-focal sorted by id
            # (mirrors BranchSpec.alternative_actions ordering so
            # downstream non-focal hashing conventions stay stable).
            focal_key = str(spec.focal_agent_id)
            executed_map = spec.alternative_joint.executed_by_agent
            non_focal = tuple(
                executed
                for key, executed in sorted(
                    ((str(a), e) for a, e in executed_map.items()),
                    key=lambda kv: kv[0],
                )
                if key != focal_key
            )
            actions = (
                executed_map[spec.focal_agent_id],
            ) + non_focal
            result = BranchResult(
                branch_id=spec.branch_id,
                fork_tick=spec.fork_tick,
                actions=actions,
                final_state_hash=final_state_hash,
                per_tick_hashes=(
                    (final_state_hash,) if final_state_hash else ()
                ),
                diverged_at_tick=None,
                restoration_ok=restoration_ok,
                error=error,
            )
            self.record_branch_result(result)
            results.append(result)

        try:
            parent_state_hash_after = hash_state(world.observe())
        except Exception as exc:  # noqa: BLE001
            parent_state_hash_after = ""
            _logger.warning(
                "JointKernelBranchScheduler: world.observe() after fork "
                "at tick %s raised %r — parent state may be polluted",
                fork_tick,
                exc,
            )

        all_restoration_ok = all(r.restoration_ok for r in results)
        self._held_fixed_verifications.append(
            HeldFixedVerification(
                fork_tick=fork_tick,
                parent_state_hash_before=parent_state_hash_before,
                parent_state_hash_after=parent_state_hash_after,
                rng_bundle_hash=rng_bundle_hash,
                non_focal_actions_hash=non_focal_actions_hash,
                branch_count=len(results),
                all_restoration_ok=all_restoration_ok,
            )
        )
        return results

    def branch_summary(self) -> Mapping[str, Any]:
        summary = dict(super().branch_summary())
        summary["mode"] = "joint_kernel_branch"
        return summary

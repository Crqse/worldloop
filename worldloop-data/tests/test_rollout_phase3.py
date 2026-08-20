"""Phase 3 rollout tests — explicit NOOP + branch fail-closed grading.

Verifies the two Phase 3 §6.3 + §6.5 additions to :func:`run_rollout`:

1. **Explicit NOOP on policy decline** — when ``policy.propose()``
   returns ``None``, the orchestrator synthesizes an
   ``ActionProposal(action_type="noop", proposer="declined")``, runs it
   through ``world.validate_action`` + ``world.step``, and records the
   transition. This eliminates "hollow" ticks where ``tick_count`` would
   exceed ``transition_count`` and ensures the dataset has one
   transition per tick for deterministic replay. The synthetic noop's
   provenance carries ``policy_declined=True`` so downstream Q6 can
   distinguish "policy chose noop" from "orchestrator synthesized noop".

2. **Branch fail-closed grading** — branch scheduler exceptions are
   graded by ``RolloutConfig.run_tier``. ``"evidence"`` raises
   :class:`EvidenceFailClosedError` (the run is invalid for evidence);
   ``"dev"`` / ``"smoke"`` / ``"safety_demo"`` log a warning and continue
   (legacy behavior preserved). ``RolloutResult.branch_fail_closed_count``
   records suppressed failures in non-evidence tiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from worldloop_kernel import (
    ActionProposal,
    BranchResult,
    Checkpoint,
    ExecutedAction,
    WorldProtocol,
)

from worldloop_data.config import RolloutConfig
from worldloop_data.coverage import UniformCoverageScheduler
from worldloop_data.counterfactual import BranchSpec
from worldloop_data.policy import PolicyContext, PolicyPool, RandomPolicy
from worldloop_data.rollout import run_rollout


# Resolve the discrete_grid.yaml example from the sibling scenarios package.
_SCENARIOS_ROOT = Path(__file__).resolve().parents[2] / "worldloop-scenarios"
_DISCRETE_GRID_YAML = _SCENARIOS_ROOT / "examples" / "discrete_grid.yaml"


def _load_discrete_grid_package():
    """Compile discrete_grid.yaml into a ScenarioPackage (test fixture)."""
    if not _DISCRETE_GRID_YAML.exists():
        pytest.skip(
            f"discrete_grid.yaml not found at {_DISCRETE_GRID_YAML}; "
            "worldloop-scenarios must be installed alongside worldloop-data."
        )
    from worldloop_scenarios.compiler import compile_file

    return compile_file(_DISCRETE_GRID_YAML)


# ---------------------------------------------------------------------------
# Test policies
# ---------------------------------------------------------------------------


class _DecliningPolicy:
    """Test policy that always declines (returns None).

    Used to exercise the explicit NOOP path in ``run_rollout``. Has a
    stable ``policy_id`` and ``inference_config`` so provenance checks
    can verify the declining policy's identity is recorded on the
    synthesized noop transition.
    """

    policy_id = "test:declining"
    policy_version = "0.1.0"
    inference_config: Mapping[str, Any] = {"decline_reason": "test_policy"}

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:  # noqa: ARG002
        return None


class _RecordingPolicy:
    """Test policy that emits a noop proposal every tick.

    Used to contrast with ``_DecliningPolicy``: when the POLICY itself
    proposes noop, ``provenance.policy_declined`` MUST be False; when the
    ORCHESTRATOR synthesizes noop (because policy declined),
    ``provenance.policy_declined`` MUST be True.
    """

    policy_id = "test:proposes_noop"
    policy_version = "0.1.0"
    inference_config: Mapping[str, Any] = {"proposed_action": "noop"}

    def propose(self, ctx: PolicyContext) -> ActionProposal | None:
        return ActionProposal(
            agent_id=ctx.agent_id,
            action_type="noop",
            params={},
            proposed_at_tick=ctx.tick,
            proposer="test:proposes_noop",
        )


# ---------------------------------------------------------------------------
# Test branch schedulers
# ---------------------------------------------------------------------------


class _CrashingBranchScheduler:
    """Branch scheduler that always raises on schedule_branches.

    Used to exercise the branch fail-closed grading path in
    ``run_rollout``. The exception is raised AFTER the NoOpBranchScheduler
    isinstance check passes, so the orchestrator's try/except wraps it.
    """

    def __init__(self, *, error_message: str = "test crash") -> None:
        self._error_message = error_message
        self._branch_results: list[BranchResult] = []

    def schedule_branches(
        self,
        checkpoint: Checkpoint,  # noqa: ARG002
        baseline_actions: Sequence[ExecutedAction],  # noqa: ARG002
        world: WorldProtocol,  # noqa: ARG002
        tick: int,  # noqa: ARG002
    ) -> list[BranchSpec]:
        raise RuntimeError(self._error_message)

    def record_branch_result(self, result: BranchResult) -> None:  # noqa: ARG002
        pass

    def branch_summary(self) -> Mapping[str, Any]:
        return {
            "branch_count": 0,
            "mode": "test_crash",
            "held_fixed": {},
        }


# ---------------------------------------------------------------------------
# Tests — explicit NOOP on policy decline
# ---------------------------------------------------------------------------


def test_declining_policy_produces_noop_transition_and_advances_tick(tmp_path):
    """When policy declines, orchestrator synthesizes noop and steps world.

    Verifies:
    - ``tick_count == num_ticks`` (no hollow ticks; the rollout ran to
      completion despite every tick being a decline).
    - ``transition_count == tick_count`` (one transition per tick).
    - ``noop_count == num_ticks`` (every tick was a synthesized noop).
    - Each transition's provenance carries ``policy_declined=True`` and
      the declining policy's ``policy_id``.
    - Each transition's candidate_actions contains the synthetic noop
      ActionProposal with ``proposer="declined"``.
    """
    package = _load_discrete_grid_package()
    world = package.world_factory(42)

    pool = PolicyPool([_DecliningPolicy()])
    cov = UniformCoverageScheduler()

    num_ticks = 5
    result = run_rollout(
        world=world,
        seed=42,
        episode_id="test_decline_noop",
        output_dir=tmp_path / "decline_noop",
        policy_pool=pool,
        coverage=cov,
        config=RolloutConfig(num_ticks=num_ticks, record=True),
    )

    # Every tick produced a transition.
    assert result.tick_count == num_ticks
    assert result.transition_count == num_ticks
    # Every tick was a synthesized noop.
    assert result.noop_count == num_ticks
    # No branch failures in DEV tier.
    assert result.branch_fail_closed_count == 0

    # Inspect recorded transitions.
    rec_dir = Path(result.output_dir)
    transition_files = sorted(rec_dir.glob("t*.json"))
    assert len(transition_files) == num_ticks

    import json

    for f in transition_files:
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        # The synthesized noop action is the executed action.
        executed = r.get("executed_actions", {})
        assert len(executed) == 1
        ex = next(iter(executed.values()))
        assert ex["action_type"] == "noop"
        # The candidate action is the synthetic proposal with
        # proposer="declined".
        candidates = r.get("candidate_actions", {})
        assert len(candidates) == 1
        cand = next(iter(candidates.values()))
        assert cand["proposer"] == "declined"
        assert cand["action_type"] == "noop"
        # Provenance records the declining policy's id + declined flag.
        prov = r.get("provenance", {})
        assert prov.get("policy_id") == "test:declining"
        assert prov.get("policy_declined") is True


def test_policy_proposing_noop_marks_policy_declined_false(tmp_path):
    """When policy itself proposes noop, ``policy_declined`` is False.

    This guards against false-positive ``policy_declined=True`` when the
    policy legitimately chose noop. The orchestrator only synthesizes
    noop when ``policy.propose()`` returns None; when the policy returns
    an explicit noop ActionProposal, the orchestrator uses it as-is and
    records ``policy_declined=False``.
    """
    package = _load_discrete_grid_package()
    world = package.world_factory(42)

    pool = PolicyPool([_RecordingPolicy()])
    cov = UniformCoverageScheduler()

    num_ticks = 3
    result = run_rollout(
        world=world,
        seed=42,
        episode_id="test_proposes_noop",
        output_dir=tmp_path / "proposes_noop",
        policy_pool=pool,
        coverage=cov,
        config=RolloutConfig(num_ticks=num_ticks, record=True),
    )

    assert result.tick_count == num_ticks
    # Noop was proposed by the policy, not synthesized by the orchestrator.
    assert result.noop_count == 0

    import json

    rec_dir = Path(result.output_dir)
    for f in sorted(rec_dir.glob("t*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        prov = r.get("provenance", {})
        assert prov.get("policy_id") == "test:proposes_noop"
        assert prov.get("policy_declined") is False
        # Candidate action's proposer is the policy's, not "declined".
        cand = next(iter(r.get("candidate_actions", {}).values()))
        assert cand["proposer"] == "test:proposes_noop"


# ---------------------------------------------------------------------------
# Tests — branch fail-closed grading
# ---------------------------------------------------------------------------


def test_branch_failure_dev_tier_suppressed_and_counted(tmp_path):
    """DEV tier: branch scheduler exceptions are logged and the rollout continues.

    Verifies:
    - Rollout completes all ticks despite branch scheduler crashing.
    - ``branch_fail_closed_count`` increments per crash.
    - ``branch_count == 0`` (no branches actually completed).
    """
    package = _load_discrete_grid_package()
    world = package.world_factory(42)

    pool = PolicyPool([RandomPolicy()])
    cov = UniformCoverageScheduler()
    branches = _CrashingBranchScheduler(
        error_message="dev-tier test crash"
    )

    num_ticks = 3
    result = run_rollout(
        world=world,
        seed=42,
        episode_id="test_branch_crash_dev",
        output_dir=tmp_path / "branch_crash_dev",
        policy_pool=pool,
        coverage=cov,
        branch_scheduler=branches,
        config=RolloutConfig(num_ticks=num_ticks, record=True, run_tier="dev"),
    )

    # Rollout ran to completion despite branch crashes.
    assert result.tick_count == num_ticks
    # No branches completed (scheduler crashed before producing any).
    assert result.branch_count == 0
    # Every tick's branch call crashed.
    assert result.branch_fail_closed_count == num_ticks


def test_branch_failure_evidence_tier_raises_fail_closed(tmp_path):
    """EVIDENCE tier: branch scheduler exceptions raise EvidenceFailClosedError.

    Verifies Phase 3 §6.3 contract: in evidence tier, branch scheduler
    instability invalidates the run. The orchestrator raises
    :class:`EvidenceFailClosedError` (not the underlying RuntimeError)
    so callers can distinguish "branch scheduler bug" from "evidence
    gate violation".
    """
    from worldloop_data.telemetry import EvidenceFailClosedError

    package = _load_discrete_grid_package()
    world = package.world_factory(42)

    pool = PolicyPool([RandomPolicy()])
    cov = UniformCoverageScheduler()
    branches = _CrashingBranchScheduler(
        error_message="evidence-tier test crash"
    )

    with pytest.raises(EvidenceFailClosedError, match="branch scheduler raised"):
        run_rollout(
            world=world,
            seed=42,
            episode_id="test_branch_crash_evidence",
            output_dir=tmp_path / "branch_crash_evidence",
            policy_pool=pool,
            coverage=cov,
            branch_scheduler=branches,
            config=RolloutConfig(
                num_ticks=3, record=True, run_tier="evidence"
            ),
        )


def test_branch_failure_smoke_tier_suppressed_like_dev(tmp_path):
    """SMOKE tier: branch scheduler exceptions are suppressed (same as DEV).

    SMOKE allows fallback but has fail-closed on mock/missing key at the
    LLM policy level. Branch scheduler instability in SMOKE is still
    treated as a warning (not evidence-invalidating) — branch failures
    are an orchestrator-layer concern, not an LLM-backend concern.
    """
    package = _load_discrete_grid_package()
    world = package.world_factory(42)

    pool = PolicyPool([RandomPolicy()])
    cov = UniformCoverageScheduler()
    branches = _CrashingBranchScheduler(
        error_message="smoke-tier test crash"
    )

    result = run_rollout(
        world=world,
        seed=42,
        episode_id="test_branch_crash_smoke",
        output_dir=tmp_path / "branch_crash_smoke",
        policy_pool=pool,
        coverage=cov,
        branch_scheduler=branches,
        config=RolloutConfig(num_ticks=2, record=True, run_tier="smoke"),
    )

    assert result.tick_count == 2
    assert result.branch_fail_closed_count == 2
    assert result.branch_count == 0


# ---------------------------------------------------------------------------
# Tests — RolloutResult field defaults (back-compat)
# ---------------------------------------------------------------------------


def test_rollout_result_new_fields_default_to_zero(tmp_path):
    """``noop_count`` and ``branch_fail_closed_count`` default to 0.

    Verifies that existing callers that don't inspect the new fields
    aren't broken. The defaults are explicit (not ``None``) so callers
    can safely do arithmetic on them.
    """
    from worldloop_data.rollout import RolloutResult

    r = RolloutResult(
        episode_id="x",
        seed=0,
        output_dir=None,
        manifest=None,
        tick_count=0,
        transition_count=0,
        branch_count=0,
    )
    assert r.noop_count == 0
    assert r.branch_fail_closed_count == 0


def test_default_run_tier_dev_preserves_legacy_behavior(tmp_path):
    """``RolloutConfig()`` default ``run_tier="dev"`` keeps legacy behavior.

    Existing tests that don't pass ``run_tier`` get the legacy "swallow
    branch errors" behavior. This guards against accidental regressions
    where the default would flip to ``"evidence"`` and start raising on
    pre-existing flaky branch schedulers.
    """
    cfg = RolloutConfig()
    assert cfg.run_tier == "dev"

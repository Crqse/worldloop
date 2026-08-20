"""Smoke test: 1×1 end-to-end pipeline on discrete_grid.yaml.

Verifies the seven-step pipeline closes::

    scenario → policy → coverage → counterfactual → export → leakage → quality

This is the smallest possible check that every component is wired
correctly. It does NOT verify Gate-level quality (Q0-Q9 pass rates) —
that is the job of the Gate verification tests added in a later attempt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldloop_data.config import PipelineConfig
from worldloop_data.pipeline import run_pipeline
from worldloop_data.policy import (
    AdversarialPolicy,
    RandomPolicy,
    ScriptedPolicy,
)

# Resolve the discrete_grid.yaml example from the sibling scenarios package.
_SCENARIOS_ROOT = Path(__file__).resolve().parents[2] / "worldloop-scenarios"
_DISCRETE_GRID_YAML = _SCENARIOS_ROOT / "examples" / "discrete_grid.yaml"


def _load_discrete_grid_package():
    """Compile discrete_grid.yaml into a ScenarioPackage."""
    if not _DISCRETE_GRID_YAML.exists():
        pytest.skip(
            f"discrete_grid.yaml not found at {_DISCRETE_GRID_YAML}; "
            "worldloop-scenarios must be installed alongside worldloop-data."
        )
    from worldloop_scenarios.compiler import compile_file

    return compile_file(_DISCRETE_GRID_YAML)


def test_pipeline_smoke_1x1(tmp_path):
    """Run the pipeline with 1 seed × 5 ticks on discrete_grid.yaml.

    Verifies:
    - Pipeline returns a PipelineResult with 1 rollout.
    - At least 1 transition recorded (manifest.record_count >= 1).
    - Quality report file written.
    - Leakage report file written.
    - Coverage report file written.
    - Dataset directory has train split (1 episode → train only).
    """
    package = _load_discrete_grid_package()

    config = PipelineConfig(
        seeds=(42,),
        num_ticks=5,
        output_dir=str(tmp_path / "smoke_dataset"),
        producer_id="worldloop-data-smoke",
        producer_version="0.1.0",
        run_utility_evaluation=True,
        utility_horizon=3,
        utility_baseline_policy_id="adversarial",
    )

    policies = [
        AdversarialPolicy(),
        ScriptedPolicy(preferred_action_type="forage"),
    ]

    result = run_pipeline(
        scenario_package=package,
        policies=policies,
        config=config,
    )

    # --- Rollout produced at least 1 transition ---
    assert len(result.rollouts) == 1
    rollout = result.rollouts[0]
    assert rollout.tick_count >= 1
    assert rollout.manifest is not None
    assert rollout.manifest.record_count >= 1
    assert rollout.manifest.quarantine_count == 0

    # --- Export produced at least one split ---
    assert result.export_result.total_episodes == 1
    assert result.export_result.total_records >= 1
    assert len(result.export_result.splits) >= 1
    # With 1 episode, only the train split should exist.
    split_names = {s.name for s in result.export_result.splits}
    assert "train" in split_names

    # --- Leakage: 0 violations (single seed, single scenario) ---
    assert result.leakage_report.ok
    assert len(result.leakage_report.violations) == 0

    # --- Coverage report has data ---
    assert result.coverage_report.transition_count >= 1
    # Two policies are covered during the rollout. Q9 is evaluated
    # separately from matched checkpoints, not inferred from this count.
    assert len(result.coverage_report.policy_usage) >= 2, (
        f"expected >=2 policies used, got "
        f"{dict(result.coverage_report.policy_usage)}"
    )
    # candidate_actions MUST be populated by the rollout orchestrator
    # (rollout.py augments TransitionRecord with the original proposal).
    # Empty candidate_actions → Q1 orphan_executed > 0 + Q6 action_type_counts empty.
    assert len(result.coverage_report.action_type_counts) >= 1, (
        f"expected >=1 action types, got "
        f"{dict(result.coverage_report.action_type_counts)}"
    )

    # --- Quality report written to disk ---
    q_report_path = result.dataset_dir / "quality_report.json"
    assert q_report_path.exists()
    with open(q_report_path, "r", encoding="utf-8") as f:
        q_data = json.load(f)
    assert "items" in q_data
    assert "overall" in q_data
    assert len(q_data["items"]) == 10  # Q0-Q9

    # --- Leakage + coverage reports written ---
    assert (result.dataset_dir / "leakage_report.json").exists()
    assert (result.dataset_dir / "coverage_report.json").exists()

    # --- Quality report: Q0 (schema) and Q5 (leakage) should pass ---
    q_by_key = {it["key"]: it for it in q_data["items"]}
    assert q_by_key["Q0"]["status"] == "pass", q_by_key["Q0"]
    assert q_by_key["Q1"]["status"] == "pass", q_by_key["Q1"]
    # Q1 evidence MUST NOT contain "orphan executed" — candidate_actions
    # is now populated by rollout.py, so every executed action has a
    # matching candidate. Per ACCEPTANCE "三要素齐全".
    assert "orphan executed" not in q_by_key["Q1"]["evidence"], (
        f"Q1 has orphan executed actions: {q_by_key['Q1']['evidence']}"
    )
    assert q_by_key["Q5"]["status"] == "pass", q_by_key["Q5"]
    # Q4 (provenance) should pass — rollout augments with policy_id.
    assert q_by_key["Q4"]["status"] == "pass", q_by_key["Q4"]
    # Q6 (coverage) should pass — transitions were observed.
    assert q_by_key["Q6"]["status"] == "pass", q_by_key["Q6"]
    # Q6 evidence MUST mention action types — candidate_actions populated
    # means action_type_counts is non-empty, so the coverage report has
    # real action coverage data (not just transition count).
    q6_evidence = q_by_key["Q6"]["evidence"]
    assert "action types" in q6_evidence, q6_evidence
    # Q9 passes only because scripted:forage has a better realized
    # same-state energy outcome than the adversarial(rest) baseline.
    assert q_by_key["Q9"]["status"] == "pass", q_by_key["Q9"]
    assert "matched comparisons" in q_by_key["Q9"]["evidence"]
    assert (result.dataset_dir / "utility_report.json").exists()
    # Q3 (replay) should pass — pipeline constructs a fresh
    # ``world_for_replay`` from the scenario package and the reporter
    # replays the first episode's actions bit-identically. discrete_grid
    # supports ``exact_restore=True`` so Q3 is not skipped.
    assert q_by_key["Q3"]["status"] == "pass", q_by_key["Q3"]
    # Q3 evidence MUST mention "bit-identical" — confirms real replay
    # verification ran (not the old "stub: deferred" skip).
    assert "bit-identical" in q_by_key["Q3"]["evidence"], q_by_key["Q3"]
    # Q7 (counterfactual) should be skipped — smoke uses NoOpBranchScheduler
    # by default (no branches generated). This is the expected baseline;
    # Q7 pass requires a real brancher (see test_rollout_with_kernel_brancher
    # and test_counterfactual.py for Q7 pass coverage).
    assert q_by_key["Q7"]["status"] == "skipped", q_by_key["Q7"]


def test_pipeline_smoke_3seed(tmp_path):
    """Run the pipeline with 3 seeds × 5 ticks for engineering smoke.

    Verifies:
    - 3 rollouts produced.
    - All records valid (Q0 schema pass).
    - Leakage: each seed goes to exactly one split (seed-level check).
    - Splits: with 3 seeds and 0.6/0.2/0.2 ratio, expect train=2, val=1, test=0
      (or train=2, val=0, test=1 depending on rounding).

    Note: single-scenario multi-seed runs disable ``check_scenario`` and
    ``check_world_param`` because both are trivially violated when only
    one scenario is pooled. The Q5-relevant checks for this config are
    ``seed`` (each seed in exactly one split) and ``branch_group``.
    """
    from worldloop_data.config import LeakageConfig
    from worldloop_data.exporter import ExporterConfig
    from worldloop_data.leakage import TrivialLeakageChecker

    package = _load_discrete_grid_package()

    config = PipelineConfig(
        seeds=(42, 43, 44),
        num_ticks=5,
        output_dir=str(tmp_path / "smoke_3seed"),
        producer_id="worldloop-data-smoke-3seed",
        producer_version="0.1.0",
    )

    policies = [RandomPolicy()]

    # Single-scenario run: split by seed, disable scenario/world_param checks.
    exporter = PlainDatasetExporterWithSeedSplit()
    leak_checker = TrivialLeakageChecker(
        config=LeakageConfig(
            check_seed=True,
            check_scenario=False,  # single scenario
            check_world_param=False,  # single world config
            check_branch_group=True,
        )
    )

    result = run_pipeline(
        scenario_package=package,
        policies=policies,
        config=config,
        exporter=exporter,
        leakage_checker=leak_checker,
    )

    # 3 rollouts.
    assert len(result.rollouts) == 3
    for r in result.rollouts:
        assert r.manifest is not None
        assert r.manifest.record_count >= 1

    # Export: 3 episodes total.
    assert result.export_result.total_episodes == 3

    # Leakage: 0 violations (seed is the only meaningful check here).
    assert result.leakage_report.ok, result.leakage_report

    # Quality report.
    q_report_path = result.dataset_dir / "quality_report.json"
    assert q_report_path.exists()
    with open(q_report_path, "r", encoding="utf-8") as f:
        q_data = json.load(f)
    q_by_key = {it["key"]: it for it in q_data["items"]}
    assert q_by_key["Q0"]["status"] == "pass"
    assert q_by_key["Q5"]["status"] == "pass"


class PlainDatasetExporterWithSeedSplit:
    """Wrapper around PlainDatasetExporter that uses seed-based splitting.

    Used in the 3-seed smoke test to align with §14.6 split priority:
    when multiple seeds are present, split by seed (priority 3) rather
    than by episode index (priority 4).
    """

    def __init__(self):
        from worldloop_data.exporter import PlainDatasetExporter
        from worldloop_data.config import ExporterConfig

        self._inner = PlainDatasetExporter(
            config=ExporterConfig(split_strategy="seed")
        )

    def export(self, episodes, dataset_dir):
        return self._inner.export(episodes=episodes, dataset_dir=dataset_dir)


def test_rollout_no_branch_by_default(tmp_path):
    """Verify the default branch scheduler is NoOp (no branches in smoke)."""
    from worldloop_data.counterfactual import NoOpBranchScheduler
    from worldloop_data.rollout import run_rollout
    from worldloop_data.policy import PolicyPool, RandomPolicy
    from worldloop_data.coverage import UniformCoverageScheduler
    from worldloop_data.config import RolloutConfig

    package = _load_discrete_grid_package()
    world = package.world_factory(42)

    pool = PolicyPool([RandomPolicy()])
    cov = UniformCoverageScheduler()

    result = run_rollout(
        world=world,
        seed=42,
        episode_id="test_no_branch",
        output_dir=tmp_path / "no_branch",
        policy_pool=pool,
        coverage=cov,
        config=RolloutConfig(num_ticks=3, record=True),
    )

    assert result.branch_count == 0


def test_rollout_with_kernel_brancher(tmp_path):
    """Verify the kernel branch scheduler can run without crashing.

    Uses branch_every_ticks=2 so a 4-tick rollout triggers at least one
    branch point (tick 2). The branch is a same-action replay (Q3 stub).
    """
    from worldloop_data.config import CounterfactualConfig, RolloutConfig
    from worldloop_data.counterfactual import KernelBranchScheduler
    from worldloop_data.coverage import UniformCoverageScheduler
    from worldloop_data.policy import PolicyPool, RandomPolicy
    from worldloop_data.rollout import run_rollout

    package = _load_discrete_grid_package()
    world = package.world_factory(42)

    pool = PolicyPool([RandomPolicy()])
    cov = UniformCoverageScheduler()
    branches = KernelBranchScheduler(
        config=CounterfactualConfig(
            branch_every_ticks=2,
            branches_per_checkpoint=1,
        )
    )

    result = run_rollout(
        world=world,
        seed=42,
        episode_id="test_branch",
        output_dir=tmp_path / "with_branch",
        policy_pool=pool,
        coverage=cov,
        branch_scheduler=branches,
        config=RolloutConfig(num_ticks=4, record=True),
    )

    # Brancher fired at tick 2 (and possibly tick 0, but tick 0 is
    # excluded by the scheduler). At least 1 branch should have run.
    assert result.branch_count >= 1
    summary = branches.branch_summary()
    assert summary["branch_count"] >= 1
    assert "held_fixed" in summary
    # Q7 contract: held_fixed MUST be a non-empty dict (world_state +
    # rng_state + other_agents at minimum). Empty held_fixed would FAIL
    # Q7 even with branches present.
    assert summary["held_fixed"], (
        f"held_fixed must be non-empty for Q7; got {summary['held_fixed']}"
    )
    # Real focal-action variation: every branch's rationale MUST mention
    # "focal-action variation" (NOT the old "stub: replay baseline
    # actions" stub). This guards against regression to the stub.
    for br in summary["branches"]:
        # branch_summary only exposes branch_id/fork_tick/diverged_at_tick/
        # restoration_ok/error — rationale is on BranchSpec, not BranchResult.
        # We assert structural invariants here; rationale is asserted in
        # test_counterfactual.py::test_kernel_scheduler_real_focal_action_variation.
        assert br["restoration_ok"] is True, (
            f"branch {br['branch_id']} failed restoration (parent world "
            f"polluted): {br}"
        )


def test_no_v1_imports():
    """Verify zero imports from current/worldloop/core/* (v1 five-layer).

    Scans every Python file under src/worldloop_data/ for the forbidden
    import path ``current.worldloop`` or ``worldloop.core``. The package
    must depend only on worldloop_kernel / worldloop_scenarios /
    worldloop_adapters.
    """
    import ast
    import re

    src_root = Path(__file__).resolve().parents[1] / "src" / "worldloop_data"
    assert src_root.exists(), f"src root not found: {src_root}"

    forbidden_patterns = [
        re.compile(r"\bcurrent\.worldloop\b"),
        re.compile(r"\bworldloop\.core\b"),
        re.compile(r"\bfrom\s+current\b"),
        re.compile(r"\bimport\s+current\b"),
    ]

    violations: list[str] = []
    for py_file in src_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        # Quick substring check before AST walk.
        if "current" not in text and "worldloop.core" not in text:
            continue
        tree = ast.parse(text, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for pat in forbidden_patterns:
                        if pat.search(alias.name):
                            violations.append(
                                f"{py_file}:{node.lineno} import {alias.name}"
                            )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for pat in forbidden_patterns:
                    if pat.search(mod):
                        violations.append(
                            f"{py_file}:{node.lineno} from {mod} import ..."
                        )

    assert not violations, (
        "Forbidden v1 imports detected:\n" + "\n".join(violations)
    )

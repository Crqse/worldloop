"""End-to-end pipeline — scenario → published dataset.

Composes :func:`run_rollout` with the six M4 components to turn a
:class:`ScenarioPackage` into a published, leak-checked, quality-scored
dataset::

    scenario → policy pool → coverage-driven run → counterfactual branch
             → dataset export → leakage check → quality report

The pipeline is the thinnest possible orchestration layer: it wires
components together and passes results forward. All business logic
lives in the components; the pipeline only sequences them.

Design rules (per main plan §14.1):
- The pipeline is deterministic given (scenario_package, seeds, config).
  Two runs with the same inputs produce the same dataset (modulo
  wall-clock timestamps in the manifest).
- The pipeline does NOT inspect or modify records; it only forwards
  directories and metadata between components.
- The pipeline returns a :class:`PipelineResult` containing every
  sub-result for downstream inspection (e.g., the quality report).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldloop_kernel import WorldProtocol

from worldloop_data.config import PipelineConfig, RolloutConfig
from worldloop_data.coverage import (
    CoverageReport,
    CoverageScheduler,
    UniformCoverageScheduler,
)
from worldloop_data.counterfactual import (
    CounterfactualBranchScheduler,
    JointKernelBranchScheduler,
    NoOpBranchScheduler,
)
from worldloop_data.exporter import (
    DatasetExporter,
    EpisodeRecords,
    ExportResult,
    PlainDatasetExporter,
)
from worldloop_data.leakage import (
    LeakageChecker,
    LeakageReport,
    TrivialLeakageChecker,
)
from worldloop_data.policy import Policy, PolicyPool
from worldloop_data.quality import (
    QualityReport,
    QualityReporter,
    MinimalQualityReporter,
)
from worldloop_data.rollout import RolloutResult, run_joint_rollout, run_rollout
from worldloop_data.utility import (
    UtilityEvaluationReport,
    evaluate_matched_policy_utility,
)

__all__ = [
    "PipelineResult",
    "run_pipeline",
]


# ---------------------------------------------------------------------------
# PipelineResult — aggregate of every sub-result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """Result of :func:`run_pipeline`.

    Attributes
    ----------
    rollouts:
        Tuple of per-episode :class:`RolloutResult`.
    episodes:
        Tuple of :class:`EpisodeRecords` handed to the exporter.
    export_result:
        The :class:`ExportResult` from the dataset exporter.
    leakage_report:
        The :class:`LeakageReport` from the leakage checker.
    coverage_report:
        The :class:`CoverageReport` from the coverage scheduler.
    quality_report:
        The :class:`QualityReport` from the quality reporter.
    dataset_dir:
        Root directory of the published dataset.
    """

    rollouts: tuple[RolloutResult, ...]
    episodes: tuple[EpisodeRecords, ...]
    export_result: ExportResult
    leakage_report: LeakageReport
    coverage_report: CoverageReport
    quality_report: QualityReport
    utility_report: UtilityEvaluationReport | None
    dataset_dir: Path


# ---------------------------------------------------------------------------
# run_pipeline — the end-to-end orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    scenario_package,
    policies: Sequence[Policy],
    config: PipelineConfig | None = None,
    coverage: CoverageScheduler | None = None,
    branch_scheduler: CounterfactualBranchScheduler | None = None,
    exporter: DatasetExporter | None = None,
    leakage_checker: LeakageChecker | None = None,
    quality_reporter: QualityReporter | None = None,
    policy_pool_config=None,
    joint_mode: bool = False,
) -> PipelineResult:
    """Run the end-to-end M4 pipeline.

    Parameters
    ----------
    scenario_package:
        A :class:`worldloop_scenarios.ScenarioPackage` providing the
        world factory.
    policies:
        Sequence of :class:`Policy` instances for the :class:`PolicyPool`.
        Must be non-empty.
    config:
        Optional :class:`PipelineConfig`. If ``None``, defaults are used.
    coverage, branch_scheduler, exporter, leakage_checker, quality_reporter:
        Optional component overrides. If any is ``None``, the default
        stub implementation is used. This allows callers to plug in
        richer implementations without modifying the pipeline.
    policy_pool_config:
        Optional :class:`PolicyPoolConfig` for the :class:`PolicyPool`.
    joint_mode:
        When ``True``, episodes are driven by :func:`run_joint_rollout`
        (Phase 5 joint action mode: every active agent acts each tick
        via ``world.step_joint``). The world factory must produce
        worlds satisfying the kernel ``JointActionWorld`` protocol, and
        ``branch_scheduler`` (if any) must be a
        :class:`JointKernelBranchScheduler`.
    """
    cfg = config or PipelineConfig()
    if not cfg.output_dir:
        raise ValueError(
            "PipelineConfig.output_dir must be set to a non-empty path"
        )
    if not cfg.seeds:
        raise ValueError("PipelineConfig.seeds must be a non-empty tuple")

    dataset_root = Path(cfg.output_dir).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Per-episode rollouts dir (kept outside the dataset dir so the
    # exporter can copy clean directories into splits).
    rollouts_root = dataset_root / "_episodes"
    rollouts_root.mkdir(parents=True, exist_ok=True)

    # Build the policy pool.
    pool = PolicyPool(policies, config=policy_pool_config)

    # Use a single coverage scheduler across all episodes so the coverage
    # report aggregates over the whole dataset.
    cov = coverage or UniformCoverageScheduler()

    # Brancher: shared across episodes so branch summaries aggregate.
    branches = branch_scheduler or NoOpBranchScheduler()
    if joint_mode and branch_scheduler is not None and not isinstance(
        branch_scheduler, JointKernelBranchScheduler
    ):
        raise TypeError(
            "joint_mode=True requires a JointKernelBranchScheduler (or "
            f"None), got {type(branch_scheduler).__name__}"
        )

    # Per-episode rollout config.
    rollout_config = RolloutConfig(
        num_ticks=cfg.num_ticks,
        record=True,
        producer_id=cfg.producer_id,
        producer_version=cfg.producer_version,
    )

    # ------------------------------------------------------------------
    # Step 1-3: scenario → policy → coverage-driven run (per seed)
    # ------------------------------------------------------------------
    rollouts: list[RolloutResult] = []
    episodes: list[EpisodeRecords] = []

    for idx, seed in enumerate(cfg.seeds):
        episode_id = f"seed{seed}_run{idx}"
        ep_dir = rollouts_root / episode_id

        # Fresh world from the scenario package.
        world = scenario_package.world_factory(seed)

        if joint_mode:
            result = run_joint_rollout(
                world=world,
                seed=seed,
                episode_id=episode_id,
                output_dir=ep_dir,
                policy_pool=pool,
                coverage=cov,
                branch_scheduler=branch_scheduler,
                config=rollout_config,
                producer_id=cfg.producer_id
                or scenario_package.spec.scenario.scenario_id,
                producer_version=cfg.producer_version,
            )
        else:
            result = run_rollout(
                world=world,
                seed=seed,
                episode_id=episode_id,
                output_dir=ep_dir,
                policy_pool=pool,
                coverage=cov,
                branch_scheduler=branches,
                config=rollout_config,
                producer_id=cfg.producer_id
                or scenario_package.spec.scenario.scenario_id,
                producer_version=cfg.producer_version,
            )
        rollouts.append(result)

        episodes.append(
            EpisodeRecords(
                episode_id=episode_id,
                seed=seed,
                scenario_id=scenario_package.spec.scenario.scenario_id,
                world_parameters_hash=scenario_package.world_parameters_hash,
                output_dir=ep_dir,
                branch_group_id=None,
            )
        )

    # ------------------------------------------------------------------
    # Step 4: counterfactual branch summary (already executed in rollout)
    # ------------------------------------------------------------------
    # The branch scheduler has been recording results during rollouts;
    # no additional step is needed here. The summary is consumed by the
    # quality reporter below.

    # ------------------------------------------------------------------
    # Step 5: dataset export
    # ------------------------------------------------------------------
    exp = exporter or PlainDatasetExporter(scenario_package=scenario_package)
    export_result = exp.export(episodes=episodes, dataset_dir=dataset_root)

    # ------------------------------------------------------------------
    # Step 6: leakage check
    # ------------------------------------------------------------------
    leak = leakage_checker or TrivialLeakageChecker()
    leakage_report = leak.check(
        export_result=export_result, episodes=episodes
    )

    # ------------------------------------------------------------------
    # Step 7: quality report
    # ------------------------------------------------------------------
    # Construct a fresh world for Q3 replay verification. The reporter's
    # ``_check_q3_replay`` will ``reset(seed)`` + replay the first
    # episode's actions and verify ``state_after_hash`` bit-identity.
    # Only ``exact_restore`` worlds pass; non-exact-restore worlds are
    # skipped. ``world_factory(seeds[0])`` produces a fresh world whose
    # capabilities match the rollouts' world (same scenario package).
    coverage_report = cov.coverage_report()
    qual = quality_reporter or MinimalQualityReporter()
    world_for_replay = scenario_package.world_factory(cfg.seeds[0])
    utility_report = None
    if cfg.run_utility_evaluation:
        utility_report = evaluate_matched_policy_utility(
            scenario_package=scenario_package,
            policies=policies,
            seeds=cfg.seeds,
            horizon=cfg.utility_horizon,
            baseline_policy_id=cfg.utility_baseline_policy_id,
            min_improvement=cfg.utility_min_improvement,
            policy_pool_config=policy_pool_config,
        )
    quality_report = qual.report(
        export_result=export_result,
        leakage_report=leakage_report,
        coverage_report=coverage_report,
        branch_scheduler=branches,
        world_for_replay=world_for_replay,
        utility_report=utility_report,
    )

    # Write the quality report alongside the dataset.
    quality_report.write(dataset_root / "quality_report.json")
    # Also write the leakage report and coverage report for inspection.
    import json

    with open(dataset_root / "leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(leakage_report.to_dict(), f, indent=2, sort_keys=True)
    with open(dataset_root / "coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(coverage_report.to_dict(), f, indent=2, sort_keys=True)
    if utility_report is not None:
        with open(
            dataset_root / "utility_report.json", "w", encoding="utf-8"
        ) as f:
            json.dump(utility_report.to_dict(), f, indent=2, sort_keys=True)

    return PipelineResult(
        rollouts=tuple(rollouts),
        episodes=tuple(episodes),
        export_result=export_result,
        leakage_report=leakage_report,
        coverage_report=coverage_report,
        quality_report=quality_report,
        utility_report=utility_report,
        dataset_dir=dataset_root,
    )

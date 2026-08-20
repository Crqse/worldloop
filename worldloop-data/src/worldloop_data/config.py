"""Configuration dataclasses for every M4 component.

Each component ships its own ``*Config`` dataclass as the default-value
source (per project rule "各层 *Config dataclass 是默认值源"). Overrides
MUST be explicit and tagged with ``_comment_purpose`` +
``_comment_design_intent`` (per project rule "默认值优先原则").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


__all__ = [
    "PolicyPoolConfig",
    "CoverageConfig",
    "CounterfactualConfig",
    "ExporterConfig",
    "LeakageConfig",
    "QualityConfig",
    "RolloutConfig",
    "PipelineConfig",
]


@dataclass(frozen=True)
class PolicyPoolConfig:
    """Configuration for :class:`PolicyPool` (S-08).

    Attributes
    ----------
    seed:
        Base RNG seed for policy-side stochasticity (e.g., random policy
        choice when multiple policies are registered with equal weight).
    default_proposer:
        Proposer label written into :class:`ActionProposal.proposer` when
        a policy does not override it. Defaults to ``"random"``.
    """

    seed: int = 42
    default_proposer: str = "random"


@dataclass(frozen=True)
class CoverageConfig:
    """Configuration for :class:`CoverageScheduler` (S-09).

    Attributes
    ----------
    strategy:
        Coverage strategy name. ``"uniform"`` rotates over registered
        policies; ``"weighted"`` uses per-policy weights (future work).
    record_observations:
        If True, the scheduler records every transition it sees for the
        coverage report. Set to False for streaming-only mode.
    """

    strategy: str = "uniform"
    record_observations: bool = True


@dataclass(frozen=True)
class CounterfactualConfig:
    """Configuration for :class:`CounterfactualBranchScheduler` (S-10).

    Attributes
    ----------
    branch_every_ticks:
        Generate counterfactual branches every N ticks. ``0`` disables
        branching. Default 0 (NoOp mode) for the stub.
    branches_per_checkpoint:
        Number of branch alternatives to generate per fork point.
    held_fixed:
        Mapping of factor name to a description of what is held fixed
        across branches (e.g., ``{"world_state": "checkpoint bytes"}``).
        Recorded into branch provenance for Q7 verification.
    """

    branch_every_ticks: int = 0
    branches_per_checkpoint: int = 1
    held_fixed: Mapping[str, str] = field(
        default_factory=lambda: {
            "world_state": "checkpoint bytes (opaque_payload)",
            "rng_state": "checkpoint.rng_bundle",
            "other_agents": "frozen at fork tick",
        }
    )


@dataclass(frozen=True)
class ExporterConfig:
    """Configuration for :class:`DatasetExporter` (S-11).

    Attributes
    ----------
    split_strategy:
        Split strategy name. ``"episode"`` splits by episode index;
        ``"scenario"`` / ``"world_param"`` / ``"seed"`` are higher-priority
        strategies (per main plan §14.6). Default ``"episode"`` for the
        stub.
    train_ratio:
        Fraction of episodes assigned to the train split. Ignored when
        ``seed_split_map`` is provided and covers every seed in the
        dataset.
    val_ratio:
        Fraction assigned to validation. Ignored when ``seed_split_map``
        is provided and covers every seed in the dataset.
    test_ratio:
        Fraction assigned to test. ``train + val + test`` MUST equal 1.0.
        Ignored when ``seed_split_map`` is provided and covers every seed
        in the dataset.
    seed_split_map:
        Optional explicit ``{seed: split_name}`` mapping. When provided
        AND ``split_strategy == "seed"``, the exporter assigns episodes
        by this map directly, bypassing the ratio-based round() logic.
        This is the single source of truth for pre-registered splits
        (e.g., 5-seed 3/1/1, 10-seed 8/1/1) and guarantees that
        ``manifest.json`` / ``summary.json`` / ``splits.json`` agree.
        Seeds not in the map fall back to ratio-based assignment.
        ``None`` (default) preserves the legacy ratio behavior.
    write_manifest:
        If True, write a per-split ``manifest.json`` summarizing the split.
    """

    split_strategy: str = "episode"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed_split_map: Mapping[int, str] | None = None
    write_manifest: bool = True

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train+val+test ratios must sum to 1.0, got {total}"
            )
        if self.seed_split_map is not None:
            for split_name in self.seed_split_map.values():
                if split_name not in ("train", "val", "test"):
                    raise ValueError(
                        f"seed_split_map contains unknown split name: "
                        f"{split_name!r} (must be 'train' / 'val' / 'test')"
                    )


@dataclass(frozen=True)
class LeakageConfig:
    """Configuration for :class:`LeakageChecker` (S-12).

    Attributes
    ----------
    check_seed:
        If True, fail if the same seed appears in multiple splits.
    check_scenario:
        If True, fail if the same ``scenario_id`` appears in multiple
        splits (only meaningful when multiple scenarios are pooled).
    check_world_param:
        If True, fail if the same ``world_parameters_hash`` appears in
        multiple splits. Only meaningful when multiple world configs are
        pooled (M5+). Disabled by default for single-scenario runs.
        The Q5 spec lists only ``branch/scenario/seed``; ``world_param``
        is a §14.6 priority-2 split strategy, not a Q5 requirement.
    check_branch_group:
        If True, fail if branches from the same fork point appear in
        different splits (counterfactual leakage).
    """

    check_seed: bool = True
    check_scenario: bool = True
    check_world_param: bool = False
    check_branch_group: bool = True


@dataclass(frozen=True)
class QualityConfig:
    """Configuration for :class:`QualityReporter` (S-13).

    Attributes
    ----------
    run_schema_check:
        Q0 — validate every transition via ``validate_transition``.
    run_traceability_check:
        Q1 — verify candidate/executed/receipt/state refs closure.
    run_diff_apply_check:
        Q2 — verify ``diff_state`` + ``apply_delta`` round-trip on a sample.
    run_replay_check:
        Q3 — verify ``replay`` is bit-identical on ``exact_restore`` worlds.
    run_provenance_check:
        Q4 — verify manifest + per-record policy_id completeness.
    run_quarantine_check:
        Q8 — verify quarantine directory exists with manifest count.
    sample_size:
        Max records to sample for expensive checks (Q2 round-trip, Q3 replay).
    """

    run_schema_check: bool = True
    run_traceability_check: bool = True
    run_diff_apply_check: bool = True
    run_replay_check: bool = True
    run_provenance_check: bool = True
    run_quarantine_check: bool = True
    sample_size: int = 16


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration for :class:`run_rollout` (single episode).

    Attributes
    ----------
    num_ticks:
        Maximum ticks per episode. The rollout also stops on world
        termination conditions defined by the ScenarioSpec.
    record:
        If True, write transitions to ``output_dir`` via
        :class:`TransitionRecorder`. If False, return records in memory only.
    producer_id:
        Producer ID written into the recorder manifest. Defaults to the
        world's ``producer_id``.
    producer_version:
        Producer version written into the manifest.
    run_tier:
        Run tier governing branch fail-closed discipline (per Phase 3
        §6.3 fallback discipline table). One of ``"dev"`` / ``"smoke"``
        / ``"evidence"`` / ``"safety_demo"``. When ``"evidence"``, branch
        scheduler exceptions raise :class:`EvidenceFailClosedError` and
        mark the run as incomplete; otherwise exceptions are logged as
        warnings and the rollout continues (legacy behavior). This field
        is read by :func:`run_rollout` only — it does NOT propagate to
        the LLM policy's per-decision fail-closed (that is set via
        :class:`LLMPolicy` kwargs). Default ``"dev"`` preserves the
        legacy "swallow branch errors" behavior for existing tests.
    """

    num_ticks: int = 100
    record: bool = True
    producer_id: str = ""
    producer_version: str = "0.1.0"
    run_tier: str = "dev"


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for :class:`run_pipeline` (end-to-end).

    Attributes
    ----------
    seeds:
        Tuple of RNG seeds, one per episode. Length determines the number
        of episodes in the dataset. Per the experiment-size discipline,
        M4 smoke uses 1-3 seeds; production scaling is out of scope.
    num_ticks:
        Ticks per episode.
    output_dir:
        Root directory for the published dataset. Subdirectories are
        created per split and per episode.
    producer_id:
        Producer ID for the recorder.
    producer_version:
        Producer version for the recorder.
    run_utility_evaluation:
        Explicitly run matched policy outcome evaluation for Q9. Disabled
        by default so a pipeline containing an external LLM policy never
        issues additional calls without caller authorization.
    utility_horizon:
        Number of matched parent states evaluated per seed.
    utility_baseline_policy_id:
        Policy used as the declared Q9 reference baseline. Empty selects
        the first registered policy.
    utility_min_improvement:
        Minimum candidate-minus-baseline scalar utility required for Q9
        to pass. Ties and smaller differences fail the outcome gate.
    """

    seeds: tuple[int, ...] = (42,)
    num_ticks: int = 100
    output_dir: str = ""
    producer_id: str = ""
    producer_version: str = "0.1.0"
    run_utility_evaluation: bool = False
    utility_horizon: int = 5
    utility_baseline_policy_id: str = ""
    utility_min_improvement: float = 1e-9

"""Phase 3 quality tests — Q7 mechanical held-fixed + Q8 quantity identity.

Verifies the two Phase 3 §6.5 quality-substantiation additions:

1. **Q7 mechanical held-fixed verification** — :class:`KernelBranchScheduler`
   now records per-fork fingerprints (parent_state_hash before/after,
   rng_bundle_hash, non_focal_actions_hash) and exposes them via
   :meth:`branch_summary` under ``held_fixed_verification``. The Q7
   check in :class:`MinimalQualityReporter` reads these and mechanically
   verifies (not just declaratively asserts) that the held-fixed
   factors were actually held fixed.

2. **Q8 quantity identity** — :class:`ExportResult` now carries
   ``produced`` / ``accepted`` / ``quarantined`` /
   ``explicitly_rejected`` / ``dropped`` fields. The Q8 check verifies
   the identity: ``produced == accepted + quarantined +
   explicitly_rejected`` AND ``dropped == 0``.
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

from worldloop_data.config import (
    CounterfactualConfig,
    ExporterConfig,
    QualityConfig,
    RolloutConfig,
)
from worldloop_data.counterfactual import (
    BranchSpec,
    HeldFixedVerification,
    KernelBranchScheduler,
    NoOpBranchScheduler,
)
from worldloop_data.coverage import CoverageReport, UniformCoverageScheduler
from worldloop_data.exporter import (
    EpisodeRecords,
    ExportResult,
    PlainDatasetExporter,
)
from worldloop_data.leakage import LeakageReport
from worldloop_data.policy import PolicyPool, RandomPolicy
from worldloop_data.quality import MinimalQualityReporter, QualityItem
from worldloop_data.rollout import run_rollout


_SCENARIOS_ROOT = Path(__file__).resolve().parents[2] / "worldloop-scenarios"
_DISCRETE_GRID_YAML = _SCENARIOS_ROOT / "examples" / "discrete_grid.yaml"


def _load_discrete_grid_package():
    if not _DISCRETE_GRID_YAML.exists():
        pytest.skip(
            f"discrete_grid.yaml not found at {_DISCRETE_GRID_YAML}; "
            "worldloop-scenarios must be installed alongside worldloop-data."
        )
    from worldloop_scenarios.compiler import compile_file

    return compile_file(_DISCRETE_GRID_YAML)


# ---------------------------------------------------------------------------
# Q7 — KernelBranchScheduler mechanical held-fixed verification
# ---------------------------------------------------------------------------


def test_kernel_brancher_records_held_fixed_verification(tmp_path):
    """KernelBranchScheduler records HeldFixedVerification per fork group.

    After ``execute_branches`` runs, the scheduler's
    ``_held_fixed_verifications`` list contains one entry per fork group
    with parent_state_hash_before/after, rng_bundle_hash, and
    non_focal_actions_hash populated.
    """
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
        episode_id="test_q7_verification",
        output_dir=tmp_path / "q7_verification",
        policy_pool=pool,
        coverage=cov,
        branch_scheduler=branches,
        config=RolloutConfig(num_ticks=4, record=True),
    )

    # At least one fork group fired (tick 2 in a 4-tick rollout).
    assert result.branch_count >= 1
    assert len(branches._held_fixed_verifications) >= 1

    # Every verification entry has the required mechanical fields.
    for v in branches._held_fixed_verifications:
        assert isinstance(v, HeldFixedVerification)
        assert v.fork_tick >= 0
        # Stub world exposes rng_bundle in its checkpoint (engine.py).
        assert v.rng_bundle_hash != ""
        # Single-agent discrete_grid has no non-focal actions.
        assert v.non_focal_actions_hash == ""
        # Parent state restoration MUST hold (kernel's branch primitive
        # is responsible for this).
        assert v.parent_state_hash_before != ""
        assert v.parent_state_hash_after != ""
        assert v.parent_state_hash_before == v.parent_state_hash_after
        assert v.all_restoration_ok is True
        assert v.branch_count >= 1


def test_kernel_brancher_summary_exposes_held_fixed_verification(tmp_path):
    """``branch_summary()`` exposes ``held_fixed_verification`` list.

    The Q7 quality check reads this list to verify mechanically. Each
    entry has the four verification booleans: parent_state_restored,
    rng_bundle_captured, non_focal_actions_consistent, all_restoration_ok.
    """
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

    run_rollout(
        world=world,
        seed=42,
        episode_id="test_q7_summary",
        output_dir=tmp_path / "q7_summary",
        policy_pool=pool,
        coverage=cov,
        branch_scheduler=branches,
        config=RolloutConfig(num_ticks=4, record=True),
    )

    summary = branches.branch_summary()
    assert "held_fixed_verification" in summary
    verifications = summary["held_fixed_verification"]
    assert len(verifications) >= 1

    for v in verifications:
        # Mechanical verification booleans present.
        assert "parent_state_restored" in v
        assert "rng_bundle_captured" in v
        assert "non_focal_actions_consistent" in v
        assert "all_restoration_ok" in v
        # Stub world: parent restored, rng captured, no non-focal
        # actions (single agent).
        assert v["parent_state_restored"] is True
        assert v["rng_bundle_captured"] is True
        assert v["non_focal_actions_consistent"] is False  # single agent
        assert v["all_restoration_ok"] is True


def test_q7_check_passes_with_mechanical_verification(tmp_path):
    """Q7 quality check PASSES when held_fixed_verification is mechanically OK.

    Runs a full rollout with the kernel brancher, then runs the
    MinimalQualityReporter's Q7 check. With parent_state_restored=True
    and all_restoration_ok=True for every fork group, Q7 passes with
    "mechanically verified" in the evidence.
    """
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

    run_rollout(
        world=world,
        seed=42,
        episode_id="test_q7_quality_pass",
        output_dir=tmp_path / "q7_quality_pass",
        policy_pool=pool,
        coverage=cov,
        branch_scheduler=branches,
        config=RolloutConfig(num_ticks=4, record=True),
    )

    reporter = MinimalQualityReporter()
    q7_item = reporter._check_q7_counterfactual(branches)
    assert q7_item.status == "pass"
    assert "mechanically verified" in q7_item.evidence
    assert "parent_state_restored" in q7_item.evidence


def test_q7_check_falls_back_when_no_verification_field():
    """Q7 falls back to declarative when scheduler lacks verification field.

    Schedulers that don't expose ``held_fixed_verification`` (e.g.,
    NoOpBranchScheduler, test stubs) get the legacy "declarative only"
    pass — but only if they have branches (otherwise skipped). Since
    NoOpBranchScheduler has 0 branches, it returns skipped; we use a
    stub scheduler with branches but no verification field.
    """

    class _StubSchedulerWithBranches:
        """Stub scheduler: 1 branch, no held_fixed_verification field."""

        def branch_summary(self) -> Mapping[str, Any]:
            return {
                "branch_count": 1,
                "mode": "stub",
                "held_fixed": {"world_state": "stub"},
                "branches": [
                    {
                        "branch_id": "b0",
                        "fork_tick": 0,
                        "diverged_at_tick": 0,
                        "restoration_ok": True,
                        "error": None,
                    }
                ],
                # NOTE: no "held_fixed_verification" key — simulates
                # a legacy scheduler that hasn't been updated to Phase 3.
            }

    reporter = MinimalQualityReporter()
    q7_item = reporter._check_q7_counterfactual(_StubSchedulerWithBranches())
    assert q7_item.status == "pass"
    assert "declarative only" in q7_item.evidence
    assert "PENDING" in q7_item.evidence


# ---------------------------------------------------------------------------
# Q8 — ExportResult quantity identity
# ---------------------------------------------------------------------------


def test_export_result_quantity_identity_default_zero():
    """ExportResult Q8 fields default to 0 (backward compat).

    Legacy callers that don't populate the new fields still produce a
    valid ExportResult. Q8 will report "produced=0; identity trivially
    satisfied" for these.
    """
    r = ExportResult(
        dataset_dir=Path("/tmp"),
        splits=(),
        total_records=0,
        total_episodes=0,
        split_strategy="episode",
        dataset_manifest_path=Path("/tmp/m.json"),
        splits_path=Path("/tmp/s.json"),
        dataset_card_path=Path("/tmp/c.md"),
        checksums_path=Path("/tmp/cs.json"),
    )
    assert r.produced == 0
    assert r.accepted == 0
    assert r.quarantined == 0
    assert r.explicitly_rejected == 0
    assert r.dropped == 0


def test_exporter_populates_quantity_identity(tmp_path):
    """PlainDatasetExporter populates Q8 quantity identity fields.

    After export, ``produced == accepted + quarantined +
    explicitly_rejected`` and ``dropped == 0``.
    """
    package = _load_discrete_grid_package()
    world = package.world_factory(42)

    pool = PolicyPool([RandomPolicy()])
    cov = UniformCoverageScheduler()

    # Run a rollout to produce records.
    rollout_result = run_rollout(
        world=world,
        seed=42,
        episode_id="test_q8_export",
        output_dir=tmp_path / "q8_records",
        policy_pool=pool,
        coverage=cov,
        config=RolloutConfig(num_ticks=5, record=True),
    )

    # Export.
    exporter = PlainDatasetExporter(config=ExporterConfig(split_strategy="episode"))
    episode_records = EpisodeRecords(
        episode_id=rollout_result.episode_id,
        seed=rollout_result.seed,
        scenario_id="discrete_grid",
        world_parameters_hash="stub",
        output_dir=Path(rollout_result.output_dir),
    )
    export_result = exporter.export(
        episodes=[episode_records],
        dataset_dir=tmp_path / "q8_dataset",
    )

    # Q8 identity: produced == accepted + quarantined + explicitly_rejected
    assert export_result.produced > 0
    assert export_result.accepted == export_result.total_records
    assert export_result.quarantined == 0  # No quarantine in healthy run
    assert export_result.explicitly_rejected == 0
    assert export_result.dropped == 0
    # The identity MUST hold.
    assert (
        export_result.produced
        == export_result.accepted
        + export_result.quarantined
        + export_result.explicitly_rejected
    )


def test_q8_check_passes_with_quantity_identity(tmp_path):
    """Q8 quality check PASSES when quantity identity holds.

    Runs a full export and then the Q8 check. With produced > 0 and
    identity holding, Q8 passes with "quantity identity holds" in evidence.
    """
    package = _load_discrete_grid_package()
    world = package.world_factory(42)

    pool = PolicyPool([RandomPolicy()])
    cov = UniformCoverageScheduler()

    rollout_result = run_rollout(
        world=world,
        seed=42,
        episode_id="test_q8_quality",
        output_dir=tmp_path / "q8_quality_records",
        policy_pool=pool,
        coverage=cov,
        config=RolloutConfig(num_ticks=5, record=True),
    )

    exporter = PlainDatasetExporter(config=ExporterConfig(split_strategy="episode"))
    episode_records = EpisodeRecords(
        episode_id=rollout_result.episode_id,
        seed=rollout_result.seed,
        scenario_id="discrete_grid",
        world_parameters_hash="stub",
        output_dir=Path(rollout_result.output_dir),
    )
    export_result = exporter.export(
        episodes=[episode_records],
        dataset_dir=tmp_path / "q8_quality_dataset",
    )

    reporter = MinimalQualityReporter()
    q8_item = reporter._check_q8_quarantine(export_result)
    assert q8_item.status == "pass"
    assert "quantity identity holds" in q8_item.evidence
    assert f"produced={export_result.produced}" in q8_item.evidence
    assert "dropped=0" in q8_item.evidence


def test_q8_check_fails_when_dropped_nonzero():
    """Q8 quality check FAILS when dropped > 0 (accounting leak).

    Construct an ExportResult with dropped=1 (simulating a leak) and
    verify Q8 fails with "quantity identity violated".
    """
    r = ExportResult(
        dataset_dir=Path("/tmp"),
        splits=(),
        total_records=10,
        total_episodes=1,
        split_strategy="episode",
        dataset_manifest_path=Path("/tmp/m.json"),
        splits_path=Path("/tmp/s.json"),
        dataset_card_path=Path("/tmp/c.md"),
        checksums_path=Path("/tmp/cs.json"),
        # Identity violated: produced=10 but accepted+quarantined+rejected=9
        # → dropped = 10 - 9 - 0 - 0 = 1 (MUST be 0)
        produced=10,
        accepted=9,
        quarantined=0,
        explicitly_rejected=0,
        dropped=1,
    )
    reporter = MinimalQualityReporter()
    q8_item = reporter._check_q8_quarantine(r)
    assert q8_item.status == "fail"
    assert "quantity identity violated" in q8_item.evidence
    assert "produced=10" in q8_item.evidence
    assert "dropped=1" in q8_item.evidence


def test_q8_check_fails_when_produced_does_not_sum():
    """Q8 quality check FAILS when produced != accepted + quarantined + rejected.

    Identity: produced == accepted + quarantined + explicitly_rejected.
    Violated here: 10 != 5 + 3 + 1 = 9.
    """
    r = ExportResult(
        dataset_dir=Path("/tmp"),
        splits=(),
        total_records=5,
        total_episodes=1,
        split_strategy="episode",
        dataset_manifest_path=Path("/tmp/m.json"),
        splits_path=Path("/tmp/s.json"),
        dataset_card_path=Path("/tmp/c.md"),
        checksums_path=Path("/tmp/cs.json"),
        produced=10,
        accepted=5,
        quarantined=3,
        explicitly_rejected=1,
        # dropped will be computed by max(0, 10 - 5 - 3 - 1) = 1
        dropped=1,
    )
    reporter = MinimalQualityReporter()
    q8_item = reporter._check_q8_quarantine(r)
    assert q8_item.status == "fail"
    assert "quantity identity violated" in q8_item.evidence


def test_q8_check_legacy_path_when_produced_zero():
    """Q8 falls back to legacy path when produced == 0.

    Legacy ExportResult (produced=0) gets the "trivially satisfied"
    pass — guards against breaking existing tests that don't populate
    the new fields.
    """
    r = ExportResult(
        dataset_dir=Path("/tmp"),
        splits=(),
        total_records=0,
        total_episodes=0,
        split_strategy="episode",
        dataset_manifest_path=Path("/tmp/m.json"),
        splits_path=Path("/tmp/s.json"),
        dataset_card_path=Path("/tmp/c.md"),
        checksums_path=Path("/tmp/cs.json"),
        # All Q8 fields default to 0.
    )
    reporter = MinimalQualityReporter()
    q8_item = reporter._check_q8_quarantine(r)
    assert q8_item.status == "pass"
    assert "trivially satisfied" in q8_item.evidence

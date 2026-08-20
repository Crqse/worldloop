"""Gate-upgrade tests: mechanical verification for Q0/Q1/Q3/Q4/Q5/Q6.

Each upgraded gate gets BOTH:
- a positive test (legitimate data passes), and
- a negative control (data violating the invariant MUST fail).

The negative controls are the point of this file: a gate that can never
fail is not a gate. Several negatives are constructed so that the
PRE-upgrade check would have passed them, demonstrating the exact hole
each upgrade closes:

- Q0: receipt with ``success=True`` + ``outcome_code="illegal_action"``
  passes every dict-layer invariant (the code IS legal) but violates the
  kernel's receipt pairing rule — only typed reconstruction catches it.
- Q1: a duplicated (episode, tick) record whose hashes form a valid
  chain — the old gate counted its chain link as valid and passed.
- Q3: a tampered initial ``state_before_hash`` — the old gate only
  compared after-hashes and never anchored the reset state.
- Q4: a manifest whose ``total_records`` disagrees with the on-disk
  records — the old gate never opened the manifest.
- Q5: a vacuous upstream report (ok=True, checked_kinds=()) and an
  upstream under-report — the old gate trusted ``ok`` blindly.
- Q6: a coverage report inconsistent with the published dataset — the
  old gate passed ANY report with transition_count > 0.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from worldloop_kernel import (
    ActionProposal,
    PROTOCOL_SCHEMA_VERSION,
    hash_state,
)

from worldloop_data.coverage import CoverageReport
from worldloop_data.exporter import ExportResult, ExportSplit
from worldloop_data.leakage import LeakageReport, LeakageViolation
from worldloop_data.quality import (
    MinimalQualityReporter,
    _record_schema_valid,
)


# ---------------------------------------------------------------------------
# Synthetic record builders (mirrors test_quality.py, kept local so this
# file has no cross-test-module import).
# ---------------------------------------------------------------------------


def _make_proposal_dict(tick: int, agent_id: str = "agent_0") -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "action_type": "forage",
        "params": {"target": 1},
        "proposed_at_tick": tick,
        "proposer": "random",
    }


def _make_executed_dict(tick: int, agent_id: str = "agent_0") -> dict[str, Any]:
    proposal = ActionProposal(
        agent_id=agent_id,
        action_type="forage",
        params={"target": 1},
        proposed_at_tick=tick,
        proposer="random",
    )
    return {
        "agent_id": agent_id,
        "action_type": "forage",
        "params": {"target": 1},
        "executed_at_tick": tick,
        "proposal_hash": hash_state(proposal),
    }


def _make_receipt_dict(outcome_code: str = "ok") -> dict[str, Any]:
    return {
        "executed_action_hash": "sha256:exec0",
        "outcome_code": outcome_code,
        "success": outcome_code == "ok",
        "energy_delta": 1.0 if outcome_code == "ok" else 0.0,
        "events": [],
        "diagnostics": {},
    }


def _make_capability_profile() -> dict[str, Any]:
    return {
        "fields": False,
        "entities": True,
        "relations": False,
        "registries": False,
        "population": False,
        "events": False,
        "exact_restore": True,
        "executable_deterministic_replay": True,
        "authority": "rule",
        "ground_truth": True,
        "transition_mode": "deterministic",
    }


def _make_state_delta(tick: int) -> dict[str, Any]:
    return {
        "field_changes": None,
        "entity_changes": None,
        "relation_changes": None,
        "registry_changes": None,
        "population_changes": None,
        "event_log": None,
        "meta_after": {
            "scenario_id": "test-scenario",
            "run_id": "test-run",
            "tick": tick + 1,
            "config_hash": "sha256:cfg",
            "rng_state_ref": "sha256:rng",
        },
        "missing_mask_after": None,
    }


def _make_clean_record(
    *,
    tick: int = 0,
    state_before: str = "sha256:before0",
    state_after: str = "sha256:after0",
    episode_id: str = "ep1",
    seed: str = "42",
    policy_id: str = "random-v1",
) -> dict[str, Any]:
    agent_id = "agent_0"
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "producer_id": "test-producer",
        "producer_version": "0.1.0",
        "tick": tick,
        "state_before_hash": state_before,
        "state_after_hash": state_after,
        "candidate_actions": {agent_id: _make_proposal_dict(tick, agent_id)},
        "executed_actions": {agent_id: _make_executed_dict(tick, agent_id)},
        "exogenous_input": None,
        "receipts": {agent_id: _make_receipt_dict()},
        "state_delta": _make_state_delta(tick),
        "capability_profile": _make_capability_profile(),
        "provenance": {
            "seed": seed,
            "policy_id": policy_id,
            "policy_version": "0.1.0",
            "inference_config": {},
            "episode_id": episode_id,
        },
    }


def _make_clean_episode(
    *,
    num_ticks: int = 3,
    episode_id: str = "ep1",
    seed: str = "42",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    prev_after: str | None = None
    for t in range(num_ticks):
        records.append(
            _make_clean_record(
                tick=t,
                state_before=prev_after or f"sha256:before{t}",
                state_after=f"sha256:after{t}",
                episode_id=episode_id,
                seed=seed,
            )
        )
        prev_after = f"sha256:after{t}"
    return records


_Q0_REQUIRED_FIELDS = (
    "schema_version",
    "producer_id",
    "producer_version",
    "tick",
    "state_before_hash",
    "state_after_hash",
    "candidate_actions",
    "executed_actions",
    "receipts",
    "state_delta",
    "capability_profile",
)


# ===========================================================================
# Q0 — typed reconstruction + kernel validate_transition
# ===========================================================================


def test_q0_positive_typed_reconstruction_passes():
    """Clean record passes both the dict layer and the kernel layer."""
    reporter = MinimalQualityReporter()
    item = reporter._check_q0_schema([_make_clean_record()])
    assert item.status == "pass", item.evidence
    assert "kernel validate_transition" in item.evidence


def test_q0_negative_receipt_success_outcome_pairing():
    """success=True + non-"ok" LEGAL outcome_code must FAIL.

    This is the exact hole of the pre-upgrade Q0: "illegal_action" IS in
    KERNEL_OUTCOME_CODES, so every dict-layer invariant passes — only the
    kernel ActionReceipt constructor enforces the success/outcome pairing.
    """
    record = _make_clean_record()
    record["receipts"]["agent_0"]["outcome_code"] = "illegal_action"
    # success stays True -> violates the kernel pairing rule.
    assert record["receipts"]["agent_0"]["success"] is True
    # Demonstrate the hole: the dict layer alone accepts this record.
    assert _record_schema_valid(record, _Q0_REQUIRED_FIELDS) is True
    reporter = MinimalQualityReporter()
    item = reporter._check_q0_schema([record])
    assert item.status == "fail", item.evidence
    assert "kernel typed-validation failures: 1" in item.evidence


def test_q0_negative_illegal_entity_change_kind():
    """An entity change with an illegal kind enum must FAIL."""
    record = _make_clean_record()
    record["capability_profile"]["entities"] = True
    record["state_delta"]["entity_changes"] = {
        "schema_id": "toy",
        "changes": [
            {
                "kind": "mutate",  # illegal: must be add/remove/update
                "entity_id": "agent_0",
                "column": "energy",
                "before": 1.0,
                "after": 0.5,
            }
        ],
        "ids_after": None,
    }
    assert _record_schema_valid(record, _Q0_REQUIRED_FIELDS) is True
    reporter = MinimalQualityReporter()
    item = reporter._check_q0_schema([record])
    assert item.status == "fail", item.evidence


# ===========================================================================
# Q1 — tick uniqueness + episode_id presence
# ===========================================================================


def test_q1_positive_clean_episode():
    reporter = MinimalQualityReporter()
    item = reporter._check_q1_traceability(_make_clean_episode(num_ticks=3))
    assert item.status == "pass", item.evidence
    assert "duplicate tick" not in item.evidence
    assert "missing episode_id" not in item.evidence


def test_q1_negative_duplicate_tick_with_valid_chain():
    """Duplicate (episode, tick) whose hashes still chain must FAIL.

    Constructed so every hash-chain link is valid (self-loop hashes) —
    the pre-upgrade Q1 passed this dataset.
    """
    r0 = _make_clean_record(
        tick=0, state_before="sha256:b0", state_after="sha256:a0"
    )
    r1 = _make_clean_record(
        tick=1, state_before="sha256:a0", state_after="sha256:a0"
    )
    r1_dup = copy.deepcopy(r1)
    reporter = MinimalQualityReporter()
    item = reporter._check_q1_traceability([r0, r1, r1_dup])
    assert item.status == "fail", item.evidence
    assert "duplicate tick(s) within episode" in item.evidence
    # The chain itself is fully valid — proving the OLD gate's blind spot.
    assert "2/2 hash-chain links valid" in item.evidence


def test_q1_negative_missing_episode_id():
    record = _make_clean_record()
    del record["provenance"]["episode_id"]
    reporter = MinimalQualityReporter()
    item = reporter._check_q1_traceability([record])
    assert item.status == "fail", item.evidence
    assert "missing episode_id" in item.evidence


# ===========================================================================
# Q3 — every-episode replay + initial-state anchor
# ===========================================================================


def _make_toyworld_episode(
    *,
    seed: int,
    num_ticks: int = 3,
    episode_id: str,
) -> list[dict[str, Any]]:
    """Run ToyWorld for ``num_ticks`` and return record dicts."""
    from worldloop_kernel import ToyWorld

    world = ToyWorld()
    world.reset(seed=seed)
    actions_seq = (
        ("move", {"direction": 1}),
        ("noop", {}),
        ("move", {"direction": -1}),
    )
    records: list[dict[str, Any]] = []
    for tick in range(num_ticks):
        action_type, params = actions_seq[tick % len(actions_seq)]
        proposal = ActionProposal(
            agent_id=world.agent_id,
            action_type=action_type,
            params=params,
            proposed_at_tick=tick,
            proposer="random",
        )
        executed, _receipt = world.validate_action(proposal)
        record = world.step(executed)
        augmented = dict(record.provenance)
        augmented["episode_id"] = episode_id
        record = dataclasses.replace(record, provenance=augmented)
        records.append(asdict(record))
    return records


def test_q3_positive_every_episode_replayed():
    """Two episodes with different seeds are BOTH replayed and anchored."""
    from worldloop_kernel import ToyWorld

    records = _make_toyworld_episode(seed=42, episode_id="seed42_run0")
    records += _make_toyworld_episode(seed=43, episode_id="seed43_run1")
    reporter = MinimalQualityReporter()
    item = reporter._check_q3_replay(records, ToyWorld())
    assert item.status == "pass", item.evidence
    assert "2/2 episode(s)" in item.evidence
    assert "anchor verified" in item.evidence


def test_q3_negative_initial_anchor_tampered():
    """Tampered initial state_before_hash must FAIL via the anchor.

    The pre-upgrade Q3 only compared after-hashes, so this tamper went
    completely undetected.
    """
    from worldloop_kernel import ToyWorld

    records = _make_toyworld_episode(seed=42, episode_id="seed42_run0")
    records[0]["state_before_hash"] = "sha256:TAMPERED_INITIAL"
    reporter = MinimalQualityReporter()
    item = reporter._check_q3_replay(records, ToyWorld())
    assert item.status == "fail", item.evidence
    assert "anchor mismatch" in item.evidence


# ===========================================================================
# Q4 — manifest cross-check
# ===========================================================================


def _make_export_result(
    tmp_path: Path,
    *,
    manifest_total: int,
    split_record_count: int | None = None,
    episode_ids: tuple[str, ...] = ("ep1",),
) -> ExportResult:
    """Write a top-level manifest and return a matching ExportResult."""
    if split_record_count is None:
        split_record_count = manifest_total
    dataset_dir = tmp_path / "dataset"
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"
    payload = {
        "total_records": manifest_total,
        "splits": {
            "train": {
                "episode_count": len(episode_ids),
                "record_count": split_record_count,
            }
        },
        "split_strategy": "episode",
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    split = ExportSplit(
        name="train",
        episode_ids=episode_ids,
        record_count=split_record_count,
        output_dir=train_dir,
        manifest_path=train_dir / "manifest.json",
    )
    return ExportResult(
        dataset_dir=dataset_dir,
        splits=(split,),
        total_records=manifest_total,
        total_episodes=len(episode_ids),
        split_strategy="episode",
        dataset_manifest_path=manifest_path,
        splits_path=dataset_dir / "splits.json",
        dataset_card_path=dataset_dir / "dataset_card.md",
        checksums_path=dataset_dir / "checksums.json",
    )


def test_q4_positive_manifest_crosscheck(tmp_path):
    records = _make_clean_episode(num_ticks=3, episode_id="ep1")
    export_result = _make_export_result(tmp_path, manifest_total=3)
    reporter = MinimalQualityReporter()
    item = reporter._check_q4_provenance(records, export_result=export_result)
    assert item.status == "pass", item.evidence
    assert "manifest cross-check: total_records=3" in item.evidence


def test_q4_negative_manifest_count_mismatch(tmp_path):
    """Manifest claims 5 records but only 3 exist on disk — must FAIL.

    The pre-upgrade Q4 never opened the manifest, so a lying manifest
    passed unchallenged.
    """
    records = _make_clean_episode(num_ticks=3, episode_id="ep1")
    export_result = _make_export_result(tmp_path, manifest_total=5)
    reporter = MinimalQualityReporter()
    item = reporter._check_q4_provenance(records, export_result=export_result)
    assert item.status == "fail", item.evidence
    assert "manifest cross-check failed" in item.evidence
    assert "!= 3 records collected on disk" in item.evidence


def test_q4_negative_undeclared_episode(tmp_path):
    """Records reference an episode_id no split declares — must FAIL."""
    records = _make_clean_episode(num_ticks=3, episode_id="ep_ghost")
    export_result = _make_export_result(
        tmp_path, manifest_total=3, episode_ids=("ep1",)
    )
    reporter = MinimalQualityReporter()
    item = reporter._check_q4_provenance(records, export_result=export_result)
    assert item.status == "fail", item.evidence
    assert "not declared in any split" in item.evidence


def test_q4_negative_inconsistent_seed_within_episode(tmp_path):
    """Same episode_id carrying two different seeds — must FAIL."""
    records = _make_clean_episode(num_ticks=2, episode_id="ep1")
    records[1]["provenance"]["seed"] = "999"
    export_result = _make_export_result(tmp_path, manifest_total=2)
    reporter = MinimalQualityReporter()
    item = reporter._check_q4_provenance(records, export_result=export_result)
    assert item.status == "fail", item.evidence
    assert "inconsistent seed within episode(s)" in item.evidence


# ===========================================================================
# Q5 — anti-vacuous + internal consistency + independent recompute
# ===========================================================================


def _clean_leakage_report(
    checked_kinds: tuple[str, ...] = ("seed", "branch_group"),
) -> LeakageReport:
    return LeakageReport(
        violations=(), by_kind={}, ok=True, checked_kinds=checked_kinds
    )


def test_q5_positive_with_independent_recompute():
    records_by_split = {
        "train": _make_clean_episode(episode_id="ep1", seed="42"),
        "val": _make_clean_episode(episode_id="ep2", seed="43"),
    }
    reporter = MinimalQualityReporter()
    item = reporter._check_q5_leakage(
        _clean_leakage_report(), records_by_split=records_by_split
    )
    assert item.status == "pass", item.evidence
    assert "0 violations (checked:" in item.evidence
    assert "independently recomputed" in item.evidence
    assert "seeds disjoint" in item.evidence


def test_q5_negative_vacuous_report():
    """ok=True with checked_kinds=() means nothing was checked — FAIL.

    The pre-upgrade Q5 trusted ``ok`` blindly and passed this.
    """
    report = LeakageReport(violations=(), by_kind={}, ok=True, checked_kinds=())
    reporter = MinimalQualityReporter()
    item = reporter._check_q5_leakage(report)
    assert item.status == "fail", item.evidence
    assert "vacuous" in item.evidence


def test_q5_negative_internally_inconsistent_report():
    """ok=True while carrying violations is a corrupt report — FAIL."""
    violation = LeakageViolation(
        kind="seed",
        key="42",
        splits=("train", "val"),
        description="seed '42' appears in splits ('train', 'val')",
    )
    report = LeakageReport(
        violations=(violation,),
        by_kind={"seed": 1},
        ok=True,  # inconsistent with violations
        checked_kinds=("seed",),
    )
    reporter = MinimalQualityReporter()
    item = reporter._check_q5_leakage(report)
    assert item.status == "fail", item.evidence
    assert "internally inconsistent" in item.evidence


def test_q5_negative_upstream_underreport_caught_by_recompute():
    """Upstream says ok but the same episode sits in two splits — FAIL."""
    shared = _make_clean_episode(episode_id="ep1", seed="42")
    records_by_split = {
        "train": shared,
        "val": copy.deepcopy(shared),
    }
    reporter = MinimalQualityReporter()
    item = reporter._check_q5_leakage(
        _clean_leakage_report(), records_by_split=records_by_split
    )
    assert item.status == "fail", item.evidence
    assert "independent recompute found episode(s)" in item.evidence


# ===========================================================================
# Q6 — coverage consistency (renamed from "Coverage")
# ===========================================================================


def _make_coverage_report(
    *,
    transition_count: int = 3,
    action_type_counts: dict[str, int] | None = None,
    outcome_code_counts: dict[str, int] | None = None,
    policy_usage: dict[str, int] | None = None,
) -> CoverageReport:
    return CoverageReport(
        policy_usage=(
            policy_usage if policy_usage is not None else {"random-v1": 3}
        ),
        action_type_counts=(
            action_type_counts
            if action_type_counts is not None
            else {"forage": 3}
        ),
        outcome_code_counts=(
            outcome_code_counts
            if outcome_code_counts is not None
            else {"ok": 3}
        ),
        agent_action_counts={"agent_0": 3},
        tick_count=transition_count,
        transition_count=transition_count,
        notes="test",
    )


def test_q6_positive_consistent_with_dataset():
    records = _make_clean_episode(num_ticks=3)
    reporter = MinimalQualityReporter()
    item = reporter._check_q6_coverage(
        _make_coverage_report(), records=records
    )
    assert item.status == "pass", item.evidence
    assert item.name == "Coverage Consistency"
    assert "action types" in item.evidence
    assert "consistent with published dataset" in item.evidence


def test_q6_negative_under_observed():
    """transition_count < published records — coverage missed data, FAIL.

    The pre-upgrade Q6 passed ANY report with transition_count > 0.
    """
    records = _make_clean_episode(num_ticks=3)
    reporter = MinimalQualityReporter()
    item = reporter._check_q6_coverage(
        _make_coverage_report(transition_count=2), records=records
    )
    assert item.status == "fail", item.evidence
    assert "under-observed" in item.evidence


def test_q6_negative_action_type_count_not_dominating():
    """Coverage action_type count below the dataset recount — FAIL."""
    records = _make_clean_episode(num_ticks=3)
    reporter = MinimalQualityReporter()
    item = reporter._check_q6_coverage(
        _make_coverage_report(action_type_counts={"forage": 1}),
        records=records,
    )
    assert item.status == "fail", item.evidence
    assert "action_type 'forage'" in item.evidence


def test_q6_negative_policy_closure_violated():
    """Dataset policy_id never counted by the scheduler — FAIL."""
    records = _make_clean_episode(num_ticks=3)
    reporter = MinimalQualityReporter()
    item = reporter._check_q6_coverage(
        _make_coverage_report(policy_usage={"someone-else": 3}),
        records=records,
    )
    assert item.status == "fail", item.evidence
    assert "never counted by the scheduler" in item.evidence

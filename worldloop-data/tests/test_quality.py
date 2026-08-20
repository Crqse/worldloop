"""Tests for S-13 Quality Reporter — Q0/Q1/Q2/Q4 enhanced checks.

Verifies the four enhanced Gate checks (Q0 Schema, Q1 Traceability,
Q2 Diff/Apply, Q4 Provenance) implemented in attempt 3.

Test strategy (per task spec):
- Integration tests use ``run_pipeline`` to produce a real dataset on
  ``discrete_grid.yaml`` (1 seed × 5 ticks) and assert each Q-item
  passes on the clean dataset.
- Unit tests construct synthetic record dicts (via ``_make_clean_record``
  / ``_make_clean_episode``) and tamper one field to assert the
  corresponding Q-item fails.

Q3/Q5/Q6/Q7/Q8/Q9 are out of scope for this file (Q3 is deferred to
attempt 6; Q5-Q9 have their own coverage in test_smoke.py).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from worldloop_kernel import (
    ActionProposal,
    PROTOCOL_SCHEMA_VERSION,
    hash_state,
)

from worldloop_data.config import PipelineConfig, QualityConfig
from worldloop_data.pipeline import run_pipeline
from worldloop_data.policy import RandomPolicy, ScriptedPolicy
from worldloop_data.quality import MinimalQualityReporter

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


# ---------------------------------------------------------------------------
# Module-scoped fixture: run the pipeline once, share across integration tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_dataset_records(tmp_path_factory):
    """Run the pipeline (1 seed × 5 ticks) and return the records list."""
    package = _load_discrete_grid_package()
    tmp_path = tmp_path_factory.mktemp("quality_dataset")
    config = PipelineConfig(
        seeds=(42,),
        num_ticks=5,
        output_dir=str(tmp_path / "dataset"),
        producer_id="worldloop-data-quality-test",
        producer_version="0.1.0",
    )
    policies = [
        RandomPolicy(),
        ScriptedPolicy(preferred_action_type="forage"),
    ]
    result = run_pipeline(
        scenario_package=package,
        policies=policies,
        config=config,
    )
    reporter = MinimalQualityReporter()
    records = reporter._collect_records(result.export_result)
    assert len(records) >= 1, "pipeline produced no records"
    return records


# ---------------------------------------------------------------------------
# Synthetic record builders — used by unit tests.
# ---------------------------------------------------------------------------


def _make_proposal_dict(tick: int, agent_id: str = "agent_0") -> dict[str, Any]:
    """Build a candidate ActionProposal dict (matches the toy world's shape)."""
    return {
        "agent_id": agent_id,
        "action_type": "forage",
        "params": {"target": 1},
        "proposed_at_tick": tick,
        "proposer": "random",
    }


def _make_executed_dict(tick: int, agent_id: str = "agent_0") -> dict[str, Any]:
    """Build an executed action dict whose proposal_hash matches the candidate."""
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
    """Build a receipt dict with the given outcome_code."""
    return {
        "executed_action_hash": "sha256:exec0",
        "outcome_code": outcome_code,
        "success": outcome_code == "ok",
        "energy_delta": 1.0 if outcome_code == "ok" else 0.0,
        "events": [],
        "diagnostics": {},
    }


def _make_capability_profile(
    *,
    fields: bool = False,
    entities: bool = True,
    relations: bool = False,
    registries: bool = False,
    population: bool = False,
    events: bool = False,
    authority: str = "rule",
    ground_truth: bool = True,
) -> dict[str, Any]:
    """Build a capability_profile dict (toy-world shape by default)."""
    return {
        "fields": fields,
        "entities": entities,
        "relations": relations,
        "registries": registries,
        "population": population,
        "events": events,
        "exact_restore": True,
        "executable_deterministic_replay": True,
        "authority": authority,
        "ground_truth": ground_truth,
        "transition_mode": "deterministic",
    }


def _make_state_delta(
    *,
    tick: int,
    field_changes: Any = None,
    entity_changes: Any = None,
    relation_changes: Any = None,
    registry_changes: Any = None,
    population_changes: Any = None,
    event_log: Any = None,
    meta_after_tick: int | None = None,
) -> dict[str, Any]:
    """Build a state_delta dict.

    By default, all slot changes are None (consistent with a capability
    profile where every slot except ``entities`` is False). ``meta_after``
    is included with ``tick + 1`` to satisfy Q2's meta_after.tick check.
    """
    meta_after = None
    if meta_after_tick is not None:
        meta_after = {
            "scenario_id": "test-scenario",
            "run_id": "test-run",
            "tick": meta_after_tick,
            "config_hash": "sha256:cfg",
            "rng_state_ref": "sha256:rng",
        }
    return {
        "field_changes": field_changes,
        "entity_changes": entity_changes,
        "relation_changes": relation_changes,
        "registry_changes": registry_changes,
        "population_changes": population_changes,
        "event_log": event_log,
        "meta_after": meta_after,
        "missing_mask_after": None,
    }


def _make_clean_record(
    *,
    tick: int = 0,
    state_before: str = "sha256:before0",
    state_after: str = "sha256:after0",
    episode_id: str = "ep1",
    schema_version: str = PROTOCOL_SCHEMA_VERSION,
    outcome_code: str = "ok",
    field_changes: Any = None,
    capability: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a record dict that passes all Q0/Q1/Q2/Q4 checks.

    Tunable parameters let unit tests tamper with one field at a time
    while keeping the rest clean.
    """
    if capability is None:
        capability = _make_capability_profile()
    if provenance is None:
        provenance = {
            "seed": "42",
            "policy_id": "random-v1",
            "policy_version": "0.1.0",
            "inference_config": {},
            "episode_id": episode_id,
        }
    agent_id = "agent_0"
    return {
        "schema_version": schema_version,
        "producer_id": "test-producer",
        "producer_version": "0.1.0",
        "tick": tick,
        "state_before_hash": state_before,
        "state_after_hash": state_after,
        "candidate_actions": {agent_id: _make_proposal_dict(tick, agent_id)},
        "executed_actions": {agent_id: _make_executed_dict(tick, agent_id)},
        "exogenous_input": None,
        "receipts": {agent_id: _make_receipt_dict(outcome_code)},
        "state_delta": _make_state_delta(
            tick=tick,
            field_changes=field_changes,
            meta_after_tick=tick + 1,
        ),
        "capability_profile": capability,
        "provenance": provenance,
    }


def _make_clean_episode(
    *,
    num_ticks: int = 3,
    episode_id: str = "ep1",
) -> list[dict[str, Any]]:
    """Build a list of records forming a valid hash chain in one episode.

    Records are linked: ``state_after_hash`` of tick ``t`` equals
    ``state_before_hash`` of tick ``t+1``.
    """
    records: list[dict[str, Any]] = []
    prev_after: str | None = None
    for t in range(num_ticks):
        state_before = prev_after or f"sha256:before{t}"
        state_after = f"sha256:after{t}"
        records.append(
            _make_clean_record(
                tick=t,
                state_before=state_before,
                state_after=state_after,
                episode_id=episode_id,
            )
        )
        prev_after = state_after
    return records


# ===========================================================================
# Q0 Schema tests (5)
# ===========================================================================


def test_q0_passes_on_clean_dataset(clean_dataset_records):
    """Q0 passes on a real pipeline-produced dataset."""
    reporter = MinimalQualityReporter()
    item = reporter._check_q0_schema(clean_dataset_records)
    assert item.key == "Q0"
    assert item.status == "pass", item.evidence
    assert "7 invariants" in item.evidence


def test_q0_fails_on_missing_field():
    """Q0 fails when a required field is missing."""
    reporter = MinimalQualityReporter()
    records = [_make_clean_record()]
    del records[0]["tick"]
    item = reporter._check_q0_schema(records)
    assert item.status == "fail"
    assert "1/1 records failed" in item.evidence


def test_q0_fails_on_wrong_schema_version():
    """Q0 fails when schema_version does not match PROTOCOL_SCHEMA_VERSION."""
    reporter = MinimalQualityReporter()
    records = [_make_clean_record(schema_version="9.9.9")]
    item = reporter._check_q0_schema(records)
    assert item.status == "fail"


def test_q0_fails_on_exec_receipt_key_mismatch():
    """Q0 fails when executed_actions keys != receipts keys."""
    reporter = MinimalQualityReporter()
    records = [_make_clean_record()]
    # Add an extra executed action with no matching receipt.
    records[0]["executed_actions"]["agent_1"] = _make_executed_dict(
        tick=0, agent_id="agent_1"
    )
    item = reporter._check_q0_schema(records)
    assert item.status == "fail"


def test_q0_fails_on_illegal_outcome_code():
    """Q0 fails when a receipt has an outcome_code not in KERNEL_OUTCOME_CODES."""
    reporter = MinimalQualityReporter()
    records = [_make_clean_record(outcome_code="bogus_outcome")]
    item = reporter._check_q0_schema(records)
    assert item.status == "fail"


# ===========================================================================
# Q1 Traceability tests (3)
# ===========================================================================


def test_q1_passes_on_clean_dataset(clean_dataset_records):
    """Q1 passes on a real pipeline-produced dataset (hash chain valid)."""
    reporter = MinimalQualityReporter()
    item = reporter._check_q1_traceability(clean_dataset_records)
    assert item.key == "Q1"
    assert item.status == "pass", item.evidence
    assert "hash-chain links valid" in item.evidence


def test_q1_fails_on_broken_hash_chain():
    """Q1 fails when state_after_hash != next state_before_hash in an episode."""
    reporter = MinimalQualityReporter()
    records = _make_clean_episode(num_ticks=3)
    # Tamper the middle record's state_after_hash to break the chain link
    # between tick 1 and tick 2.
    records[1]["state_after_hash"] = "sha256:TAMPERED"
    item = reporter._check_q1_traceability(records)
    assert item.status == "fail"
    # One of the two chain links is broken.
    assert "1/2 hash-chain links valid" in item.evidence


def test_q1_passes_single_episode_chain():
    """Q1 passes on a synthetic single-episode hash chain (3 ticks)."""
    reporter = MinimalQualityReporter()
    records = _make_clean_episode(num_ticks=3)
    item = reporter._check_q1_traceability(records)
    assert item.status == "pass", item.evidence
    # 3 records => 2 chain links.
    assert "2/2 hash-chain links valid" in item.evidence
    assert "3/3 exec/receipt keys match" in item.evidence


# ===========================================================================
# Q2 Diff/Apply tests (3)
# ===========================================================================


def test_q2_passes_on_clean_dataset(clean_dataset_records):
    """Q2 passes on a real pipeline-produced dataset (structural check)."""
    reporter = MinimalQualityReporter()
    item = reporter._check_q2_diff_apply(clean_dataset_records)
    assert item.key == "Q2"
    assert item.status == "pass", item.evidence
    assert "structural check" in item.evidence
    assert "Q3 (attempt 6)" in item.evidence


def test_q2_fails_on_delta_capability_mismatch():
    """Q2 fails when capability.fields=False but state_delta.field_changes is non-None."""
    reporter = MinimalQualityReporter()
    # Default capability has fields=False; force field_changes to non-None.
    records = [
        _make_clean_record(
            field_changes=[{"channel": "energy", "before": 1.0, "after": 0.5}],
        )
    ]
    item = reporter._check_q2_diff_apply(records)
    assert item.status == "fail"
    assert "0/1 delta-capability consistent" in item.evidence


def test_q2_skipped_when_disabled():
    """Q2 is skipped when config.run_diff_apply_check=False."""
    reporter = MinimalQualityReporter(
        config=QualityConfig(run_diff_apply_check=False)
    )
    records = [_make_clean_record()]
    item = reporter._check_q2_diff_apply(records)
    assert item.status == "skipped"
    assert "disabled by config" in item.evidence


# ===========================================================================
# Q4 Provenance tests (4)
# ===========================================================================


def test_q4_passes_on_clean_dataset(clean_dataset_records):
    """Q4 passes on a real pipeline-produced dataset."""
    reporter = MinimalQualityReporter()
    item = reporter._check_q4_provenance(clean_dataset_records)
    assert item.key == "Q4"
    assert item.status == "pass", item.evidence
    assert "policy_id + policy_version + inference_config + episode_id + seed" in item.evidence


def test_q4_fails_on_missing_policy_id():
    """Q4 fails when provenance.policy_id is missing."""
    reporter = MinimalQualityReporter()
    records = [_make_clean_record()]
    del records[0]["provenance"]["policy_id"]
    item = reporter._check_q4_provenance(records)
    assert item.status == "fail"


def test_q4_fails_on_missing_policy_version():
    """Q4 fails when provenance.policy_version is missing."""
    reporter = MinimalQualityReporter()
    records = [_make_clean_record()]
    del records[0]["provenance"]["policy_version"]
    item = reporter._check_q4_provenance(records)
    assert item.status == "fail"


def test_q4_fails_on_missing_inference_config():
    """Q4 fails when provenance.inference_config is missing."""
    reporter = MinimalQualityReporter()
    records = [_make_clean_record()]
    del records[0]["provenance"]["inference_config"]
    item = reporter._check_q4_provenance(records)
    assert item.status == "fail"


# ===========================================================================
# Q3 Replay tests (4) — added in attempt 6
# ===========================================================================
#
# Q3 verifies that, given a checkpoint (here: world.reset + frozen action
# sequence) and an ``exact_restore`` world, re-running the actions
# produces bit-identical ``state_after_hash`` values. The tests cover the
# four code paths in ``_check_q3_replay``:
#   - skipped when no world provided
#   - skipped when world.capabilities.exact_restore is False
#   - pass when replay matches (ToyWorld, 5 ticks)
#   - fail when records' state_after_hash is tampered


def _make_q3_replay_records(
    *,
    seed: int = 42,
    num_ticks: int = 5,
    episode_id: str = "seed42_run0",
) -> list[dict[str, Any]]:
    """Run ToyWorld for ``num_ticks`` steps and return record dicts.

    Produces real records with valid ``state_after_hash`` and
    ``provenance.seed``, suitable for Q3 replay verification. The
    provenance is augmented with ``episode_id`` so
    ``_check_q3_replay`` can group records by episode (mirroring what
    ``rollout.run_rollout`` does in production).
    """
    import dataclasses
    from dataclasses import asdict

    from worldloop_kernel import ToyWorld

    world = ToyWorld()
    world.reset(seed=seed)
    # Fixed action sequence (deterministic — ToyWorld is deterministic
    # given seed + actions). Repeats if num_ticks > len(actions_seq).
    actions_seq = (
        ("move", {"direction": 1}),
        ("noop", {}),
        ("move", {"direction": -1}),
        ("move", {"direction": 1}),
        ("noop", {}),
        ("move", {"direction": -1}),
        ("move", {"direction": 1}),
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
        # Augment provenance with episode_id (rollout.py does this in
        # production; we replicate it here so _check_q3_replay can group).
        augmented = dict(record.provenance)
        augmented["episode_id"] = episode_id
        record = dataclasses.replace(record, provenance=augmented)
        records.append(asdict(record))
    return records


def test_q3_skipped_when_no_world():
    """Q3 is skipped when ``world_for_replay`` is None."""
    reporter = MinimalQualityReporter()
    records = _make_clean_episode(num_ticks=3)
    item = reporter._check_q3_replay(records, None)
    assert item.key == "Q3"
    assert item.status == "skipped"
    assert "no world provided" in item.evidence


def test_q3_skipped_when_no_exact_restore():
    """Q3 is skipped when ``world.capabilities.exact_restore`` is False."""
    from worldloop_kernel import ToyWorld
    from worldloop_kernel.capability import CapabilityProfile

    cap = CapabilityProfile(
        fields=False,
        entities=True,
        relations=False,
        registries=False,
        population=False,
        events=False,
        exact_restore=False,  # disable exact_restore
        executable_deterministic_replay=True,
        authority="rule",
        ground_truth=True,
        transition_mode="deterministic",
    )
    world = ToyWorld(capabilities=cap)
    reporter = MinimalQualityReporter()
    records = _make_q3_replay_records(num_ticks=3)
    item = reporter._check_q3_replay(records, world)
    assert item.status == "skipped"
    assert "exact_restore" in item.evidence


def test_q3_pass_when_bit_identical():
    """Q3 passes when replay produces bit-identical ``state_after_hash``.

    Uses ToyWorld (``exact_restore=True``, deterministic) — the same
    world that produced the records. A fresh ToyWorld instance is used
    for replay to verify that ``reset(seed)`` + frozen action sequence
    reproduces the recorded hashes.
    """
    from worldloop_kernel import ToyWorld

    records = _make_q3_replay_records(num_ticks=5)
    # Fresh world for replay (the records were produced by an
    # equivalent ToyWorld with the same seed and action sequence).
    world_for_replay = ToyWorld()
    reporter = MinimalQualityReporter()
    item = reporter._check_q3_replay(records, world_for_replay)
    assert item.key == "Q3"
    assert item.status == "pass", item.evidence
    assert "5/5 records replay bit-identical" in item.evidence
    assert "state_after_hash match" in item.evidence


def test_q3_fail_when_diverged():
    """Q3 fails when records' ``state_after_hash`` is tampered.

    Tampering the 3rd record's hash (index 2, tick=2) means the replay
    will match 4 out of 5 records and diverge on the tampered one.
    """
    from worldloop_kernel import ToyWorld

    records = _make_q3_replay_records(num_ticks=5)
    # Tamper the 3rd record's state_after_hash to a bogus value.
    records[2]["state_after_hash"] = "sha256:TAMPERED_HASH"
    world_for_replay = ToyWorld()
    reporter = MinimalQualityReporter()
    item = reporter._check_q3_replay(records, world_for_replay)
    assert item.status == "fail"
    assert "4/5 records match" in item.evidence
    assert "1 diverged" in item.evidence
    assert "diverged at ticks" in item.evidence

"""R5 evaluator tests for M6 evaluation (audit F-05 / R5).

Audit F-05 found that M6 evaluation code (DataLoader / baselines / runner)
was not tracked in version control and had no专项 tests. The audit R5
section listed 12 specific test areas; this file covers the four areas
NOT already covered by ``test_feature_layout.py`` or
``test_state_materializer.py``:

- R5 #6: split by seed/episode without leakage (TestSplitNoLeakage)
- R5 #7: multi-step target not crossing episode (TestMultiStepTargetNoCrossEpisode)
- R5 #8: ranking only effective when multiple candidates (TestRankingOnlyWhenMultipleCandidates)
- R5 #10: runner output binds protocol hash + source commit (TestRunnerOutputBinding)
- R4: protocol file integrity (TestProtocolIntegrity)

Already covered by other test files (not duplicated here):
- R5 #1 DataLoader doesn't lose state/exogenous → TestF03StateBlocks + TestF03Exogenous
- R5 #2 joint action encoding order stable → TestF03JointAction
- R5 #3 FeatureLayout dynamic slice → TestFeatureLayoutFromDims
- R5 #4 no-action only removes action → TestNoActionInvariance
- R5 #5 shuffled-action only shuffles action → TestShuffledActionInvariance
- R5 #9 4/5/7 actions cross-scenario → TestCrossScenario
- R5 #11 module.__file__ in clean environment → TestModuleImportPath + test_module_import.py
- R5 #12 Windows non-UTF-8 locale stable → TestUtf8Portability
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_PROTOCOL_DIR = _REPO_ROOT / "research" / "evidence" / "m6_v2_protocol"
_PROTOCOL_PATH = _PROTOCOL_DIR / "protocol.json"
_PROTOCOL_SHA_PATH = _PROTOCOL_DIR / "protocol.sha256"


def _make_transition_record(
    tick: int,
    *,
    episode_id: str = "ep_a",
    seed: str = "42",
    energy_delta: float = -1.0,
    candidate_actions: dict[str, Any] | None = None,
    executed_actions: dict[str, Any] | None = None,
    state_before_hash: str = "hash_before",
    state_after_hash: str = "hash_after",
) -> dict[str, Any]:
    """Build a minimal transition record dict for testing."""
    executed = executed_actions or {"e0": {"action_type": "MOVE", "target_node": "base"}}
    candidates = candidate_actions or executed
    return {
        "schema_version": "0.1.0",
        "producer_id": "test-world",
        "producer_version": "0.1.0",
        "tick": tick,
        "state_before_hash": state_before_hash,
        "candidate_actions": candidates,
        "executed_actions": executed,
        "exogenous_input": None,
        "receipts": {aid: {"energy_delta": energy_delta} for aid in executed},
        "state_delta": {"entity_changes": {"changes": []}},
        "state_after_hash": state_after_hash,
        "capability_profile": {
            "fields": True, "entities": True, "relations": True,
            "registries": True, "population": True, "events": False,
            "exact_restore": True, "executable_deterministic_replay": True,
            "authority": "rule", "ground_truth": True,
            "transition_mode": "deterministic",
        },
        "provenance": {"episode_id": episode_id, "seed": seed, "policy_id": "random"},
    }


# ---------------------------------------------------------------------------
# R5 #6: split by seed/episode without leakage
# ---------------------------------------------------------------------------


class TestSplitNoLeakage:
    """R5 #6: split must be by episode (no row splits, no cross-split branch groups).

    The existing ``TrivialLeakageChecker`` enforces this at export time.
    This test class verifies the check holds on synthetic splits where:
    - Same seed appears in both train and test (seed leakage)
    - Same scenario appears in both train and test (scenario leakage)
    - Same branch_group_id appears in both train and test (branch leakage)
    - Same episode_id appears in both train and test (episode leakage)
    """

    def test_seed_leakage_detected(self):
        """Same seed in train and test must be flagged as seed leakage."""
        from worldloop_data.exporter import EpisodeRecords, ExportResult, ExportSplit
        from worldloop_data.leakage import TrivialLeakageChecker

        ep_train = EpisodeRecords(
            episode_id="ep_a", seed=42, scenario_id="scen_a",
            world_parameters_hash="hash_a",
            output_dir=Path("/tmp/ep_a"), branch_group_id=None,
        )
        ep_test = EpisodeRecords(
            episode_id="ep_b", seed=42, scenario_id="scen_b",
            world_parameters_hash="hash_b",
            output_dir=Path("/tmp/ep_b"), branch_group_id=None,
        )
        splits = (
            ExportSplit(name="train", episode_ids=("ep_a",), record_count=1,
                        output_dir=Path("/tmp/train"),
                        manifest_path=Path("/tmp/train/m.json")),
            ExportSplit(name="test", episode_ids=("ep_b",), record_count=1,
                        output_dir=Path("/tmp/test"),
                        manifest_path=Path("/tmp/test/m.json")),
        )
        export = ExportResult(
            dataset_dir=Path("/tmp/ds"), splits=splits,
            total_records=2, total_episodes=2,
            split_strategy="episode",
            dataset_manifest_path=Path("/tmp/ds/m.json"),
            splits_path=Path("/tmp/ds/splits.json"),
            dataset_card_path=Path("/tmp/ds/card.md"),
            checksums_path=Path("/tmp/ds/checksums.json"),
        )
        report = TrivialLeakageChecker().check(export, [ep_train, ep_test])
        assert not report.ok, "seed leakage must be flagged"
        kinds = {v.kind for v in report.violations}
        assert "seed" in kinds, f"expected 'seed' in violations: {kinds}"

    def test_episode_leakage_detected(self):
        """Same episode_id in train and test must be flagged."""
        from worldloop_data.exporter import EpisodeRecords, ExportResult, ExportSplit
        from worldloop_data.leakage import TrivialLeakageChecker

        ep = EpisodeRecords(
            episode_id="ep_a", seed=42, scenario_id="scen_a",
            world_parameters_hash="hash_a",
            output_dir=Path("/tmp/ep_a"), branch_group_id=None,
        )
        splits = (
            ExportSplit(name="train", episode_ids=("ep_a",), record_count=1,
                        output_dir=Path("/tmp/train"),
                        manifest_path=Path("/tmp/train/m.json")),
            ExportSplit(name="test", episode_ids=("ep_a",), record_count=1,
                        output_dir=Path("/tmp/test"),
                        manifest_path=Path("/tmp/test/m.json")),
        )
        export = ExportResult(
            dataset_dir=Path("/tmp/ds"), splits=splits,
            total_records=2, total_episodes=1,
            split_strategy="episode",
            dataset_manifest_path=Path("/tmp/ds/m.json"),
            splits_path=Path("/tmp/ds/splits.json"),
            dataset_card_path=Path("/tmp/ds/card.md"),
            checksums_path=Path("/tmp/ds/checksums.json"),
        )
        report = TrivialLeakageChecker().check(export, [ep])
        assert not report.ok, "episode leakage must be flagged"

    def test_branch_group_leakage_detected(self):
        """Same branch_group_id in train and test must be flagged."""
        from worldloop_data.exporter import EpisodeRecords, ExportResult, ExportSplit
        from worldloop_data.leakage import TrivialLeakageChecker

        ep_train = EpisodeRecords(
            episode_id="ep_a", seed=42, scenario_id="scen_a",
            world_parameters_hash="hash_a",
            output_dir=Path("/tmp/ep_a"), branch_group_id="bg_1",
        )
        ep_test = EpisodeRecords(
            episode_id="ep_b", seed=99, scenario_id="scen_a",
            world_parameters_hash="hash_a",
            output_dir=Path("/tmp/ep_b"), branch_group_id="bg_1",
        )
        splits = (
            ExportSplit(name="train", episode_ids=("ep_a",), record_count=1,
                        output_dir=Path("/tmp/train"),
                        manifest_path=Path("/tmp/train/m.json")),
            ExportSplit(name="test", episode_ids=("ep_b",), record_count=1,
                        output_dir=Path("/tmp/test"),
                        manifest_path=Path("/tmp/test/m.json")),
        )
        export = ExportResult(
            dataset_dir=Path("/tmp/ds"), splits=splits,
            total_records=2, total_episodes=2,
            split_strategy="episode",
            dataset_manifest_path=Path("/tmp/ds/m.json"),
            splits_path=Path("/tmp/ds/splits.json"),
            dataset_card_path=Path("/tmp/ds/card.md"),
            checksums_path=Path("/tmp/ds/checksums.json"),
        )
        report = TrivialLeakageChecker().check(export, [ep_train, ep_test])
        assert not report.ok, "branch group leakage must be flagged"
        kinds = {v.kind for v in report.violations}
        assert "branch_group" in kinds, f"expected 'branch_group': {kinds}"

    def test_no_leakage_when_clean_split(self):
        """Different seed + scenario + branch_group + episode → no leakage."""
        from worldloop_data.exporter import EpisodeRecords, ExportResult, ExportSplit
        from worldloop_data.leakage import TrivialLeakageChecker

        ep_train = EpisodeRecords(
            episode_id="ep_a", seed=42, scenario_id="scen_a",
            world_parameters_hash="hash_a",
            output_dir=Path("/tmp/ep_a"), branch_group_id="bg_1",
        )
        ep_test = EpisodeRecords(
            episode_id="ep_b", seed=99, scenario_id="scen_b",
            world_parameters_hash="hash_b",
            output_dir=Path("/tmp/ep_b"), branch_group_id="bg_2",
        )
        splits = (
            ExportSplit(name="train", episode_ids=("ep_a",), record_count=1,
                        output_dir=Path("/tmp/train"),
                        manifest_path=Path("/tmp/train/m.json")),
            ExportSplit(name="test", episode_ids=("ep_b",), record_count=1,
                        output_dir=Path("/tmp/test"),
                        manifest_path=Path("/tmp/test/m.json")),
        )
        export = ExportResult(
            dataset_dir=Path("/tmp/ds"), splits=splits,
            total_records=2, total_episodes=2,
            split_strategy="episode",
            dataset_manifest_path=Path("/tmp/ds/m.json"),
            splits_path=Path("/tmp/ds/splits.json"),
            dataset_card_path=Path("/tmp/ds/card.md"),
            checksums_path=Path("/tmp/ds/checksums.json"),
        )
        report = TrivialLeakageChecker().check(export, [ep_train, ep_test])
        assert report.ok, f"clean split must not flag leakage: {report.violations}"


# ---------------------------------------------------------------------------
# R5 #7: multi-step target not crossing episode
# ---------------------------------------------------------------------------


class TestMultiStepTargetNoCrossEpisode:
    """R5 #7: ``_attach_multi_step_targets`` must NOT sum across episode boundary.

    Audit R5 #7 requires multi-step target (``multi_step_energy_delta``)
    to not cross episode. The implementation in ``data_loader.py`` groups
    samples by ``episode_id`` before summing the next ``horizon`` ticks;
    samples at the end of an episode accumulate only the remaining ticks.
    """

    def test_multi_step_does_not_cross_episode_boundary(self):
        """End-of-episode sample must accumulate only remaining ticks."""
        from worldloop_data.evaluation.data_loader import (
            TransitionSample, _attach_multi_step_targets,
        )

        # Episode A: 2 samples (ticks 0, 1) with energy_delta 10, 20.
        # Episode B: 1 sample (tick 0) with energy_delta 5.
        # horizon=3, so:
        # - ep_a[0] should sum next 3 within ep_a → only ep_a[1] = 20
        # - ep_a[1] should sum next 3 within ep_a → no more = 0
        # - ep_b[0] should sum next 3 within ep_b → no more = 0
        samples = [
            TransitionSample(
                tick=0, action_type_idx=0, agent_id_idx=0,
                target_node_idx=0, target_agent_idx=-1, has_params=0,
                energy_delta=10.0, position_change_idx=0,
                edge_change_count=0, executed_candidate_rank=0,
                multi_step_energy_delta=0.0,
                episode_id="ep_a", seed="42", split="train",
                policy_id="random", state_before_hash="h_a0",
                state_after_hash="h_a1",
            ),
            TransitionSample(
                tick=1, action_type_idx=0, agent_id_idx=0,
                target_node_idx=0, target_agent_idx=-1, has_params=0,
                energy_delta=20.0, position_change_idx=0,
                edge_change_count=0, executed_candidate_rank=0,
                multi_step_energy_delta=0.0,
                episode_id="ep_a", seed="42", split="train",
                policy_id="random", state_before_hash="h_a1",
                state_after_hash="h_a2",
            ),
            TransitionSample(
                tick=0, action_type_idx=0, agent_id_idx=0,
                target_node_idx=0, target_agent_idx=-1, has_params=0,
                energy_delta=5.0, position_change_idx=0,
                edge_change_count=0, executed_candidate_rank=0,
                multi_step_energy_delta=0.0,
                episode_id="ep_b", seed="99", split="test",
                policy_id="random", state_before_hash="h_b0",
                state_after_hash="h_b1",
            ),
        ]
        _attach_multi_step_targets(samples, horizon=3)

        assert samples[0].multi_step_energy_delta == 20.0, (
            f"ep_a[0] should sum ep_a[1].energy_delta=20, got {samples[0].multi_step_energy_delta}"
        )
        assert samples[1].multi_step_energy_delta == 0.0, (
            f"ep_a[1] is last in episode, should be 0, got {samples[1].multi_step_energy_delta}"
        )
        assert samples[2].multi_step_energy_delta == 0.0, (
            f"ep_b[0] is last in episode, should be 0, got {samples[2].multi_step_energy_delta}"
        )

    def test_multi_step_sums_within_episode_when_long_enough(self):
        """When episode has >=horizon+1 samples, full horizon is summed."""
        from worldloop_data.evaluation.data_loader import (
            TransitionSample, _attach_multi_step_targets,
        )

        # Episode with 4 samples (ticks 0,1,2,3), energy_delta 1,2,3,4.
        # horizon=3:
        # - tick 0: sum(1+1, 1+2, 1+3) = 2+3+4 = 9
        # - tick 1: sum(2+1, 2+2) = 3+4 = 7
        # - tick 2: sum(3+1) = 4
        # - tick 3: no more = 0
        samples = [
            TransitionSample(
                tick=t, action_type_idx=0, agent_id_idx=0,
                target_node_idx=0, target_agent_idx=-1, has_params=0,
                energy_delta=float(t + 1), position_change_idx=0,
                edge_change_count=0, executed_candidate_rank=0,
                multi_step_energy_delta=0.0,
                episode_id="ep_long", seed="42", split="train",
                policy_id="random", state_before_hash=f"h_{t}",
                state_after_hash=f"h_{t+1}",
            )
            for t in range(4)
        ]
        _attach_multi_step_targets(samples, horizon=3)

        assert samples[0].multi_step_energy_delta == 2 + 3 + 4
        assert samples[1].multi_step_energy_delta == 3 + 4
        assert samples[2].multi_step_energy_delta == 4
        assert samples[3].multi_step_energy_delta == 0

    def test_multi_step_respects_horizon_parameter(self):
        """horizon=1 should only look 1 tick ahead."""
        from worldloop_data.evaluation.data_loader import (
            TransitionSample, _attach_multi_step_targets,
        )

        samples = [
            TransitionSample(
                tick=t, action_type_idx=0, agent_id_idx=0,
                target_node_idx=0, target_agent_idx=-1, has_params=0,
                energy_delta=float(t + 1), position_change_idx=0,
                edge_change_count=0, executed_candidate_rank=0,
                multi_step_energy_delta=0.0,
                episode_id="ep_h1", seed="42", split="train",
                policy_id="random", state_before_hash=f"h_{t}",
                state_after_hash=f"h_{t+1}",
            )
            for t in range(3)
        ]
        _attach_multi_step_targets(samples, horizon=1)

        # horizon=1: sample i sums only sample i+1.
        assert samples[0].multi_step_energy_delta == 2.0
        assert samples[1].multi_step_energy_delta == 3.0
        assert samples[2].multi_step_energy_delta == 0.0

    def test_multi_step_handles_empty_sample_list(self):
        """No samples → no error."""
        from worldloop_data.evaluation.data_loader import _attach_multi_step_targets

        _attach_multi_step_targets([], horizon=3)  # must not raise

    def test_multi_step_handles_single_episode_single_sample(self):
        """Single sample in single episode → multi_step = 0."""
        from worldloop_data.evaluation.data_loader import (
            TransitionSample, _attach_multi_step_targets,
        )

        samples = [
            TransitionSample(
                tick=0, action_type_idx=0, agent_id_idx=0,
                target_node_idx=0, target_agent_idx=-1, has_params=0,
                energy_delta=42.0, position_change_idx=0,
                edge_change_count=0, executed_candidate_rank=0,
                multi_step_energy_delta=0.0,
                episode_id="ep_solo", seed="42", split="train",
                policy_id="random", state_before_hash="h_0",
                state_after_hash="h_1",
            ),
        ]
        _attach_multi_step_targets(samples, horizon=3)
        assert samples[0].multi_step_energy_delta == 0.0


# ---------------------------------------------------------------------------
# R5 #8: ranking only effective when multiple candidates
# ---------------------------------------------------------------------------


class TestRankingOnlyWhenMultipleCandidates:
    """R5 #8: ``executed_candidate_rank`` must be 0 when only 1 candidate.

    Audit F-04 fix: candidate ranking is only meaningful when each decision
    point has multiple candidates. ``_extract_executed_candidate_rank``
    returns 0 (no ranking task) when ``len(candidate_actions) <= 1``.
    """

    def test_single_candidate_returns_zero(self):
        """One candidate → rank=0 (no ranking task)."""
        from worldloop_data.evaluation.data_loader import _extract_executed_candidate_rank

        record = {"candidate_actions": {"e0": {"action_type": "MOVE"}}}
        assert _extract_executed_candidate_rank(record, "e0") == 0

    def test_empty_candidates_returns_zero(self):
        """Zero candidates → rank=0."""
        from worldloop_data.evaluation.data_loader import _extract_executed_candidate_rank

        record = {"candidate_actions": {}}
        assert _extract_executed_candidate_rank(record, "e0") == 0

    def test_missing_candidates_key_returns_zero(self):
        """Missing 'candidate_actions' key → rank=0."""
        from worldloop_data.evaluation.data_loader import _extract_executed_candidate_rank

        record = {}
        assert _extract_executed_candidate_rank(record, "e0") == 0

    def test_two_candidates_returns_one_indexed_rank(self):
        """Two candidates → rank is 1-indexed by alphabetical agent_id."""
        from worldloop_data.evaluation.data_loader import _extract_executed_candidate_rank

        record = {
            "candidate_actions": {
                "e1": {"action_type": "MOVE"},
                "e0": {"action_type": "REST"},
            }
        }
        # Sorted agent_ids: ["e0", "e1"]; "e1" is at index 1, so rank=2.
        assert _extract_executed_candidate_rank(record, "e1") == 2
        # "e0" is at index 0, so rank=1.
        assert _extract_executed_candidate_rank(record, "e0") == 1

    def test_three_candidates_preserves_alphabetical_order(self):
        """Three candidates → ranks 1, 2, 3 by alphabetical agent_id."""
        from worldloop_data.evaluation.data_loader import _extract_executed_candidate_rank

        record = {
            "candidate_actions": {
                "e2": {"action_type": "C"},
                "e0": {"action_type": "A"},
                "e1": {"action_type": "B"},
            }
        }
        # Sorted: ["e0", "e1", "e2"]; ranks 1, 2, 3.
        assert _extract_executed_candidate_rank(record, "e0") == 1
        assert _extract_executed_candidate_rank(record, "e1") == 2
        assert _extract_executed_candidate_rank(record, "e2") == 3

    def test_ranking_zero_excluded_from_metric_in_protocol(self):
        """R5 #8 / F-04 fix: protocol.json must specify rank=0 excluded from metric."""
        assert _PROTOCOL_PATH.is_file(), (
            f"protocol.json missing at {_PROTOCOL_PATH}; R4 protocol must exist before R5 #8 test"
        )
        protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
        rank_target = next(
            (t for t in protocol["primary_targets"] if t["name"] == "executed_candidate_rank"),
            None,
        )
        assert rank_target is not None, "protocol must include executed_candidate_rank target"
        # The description must mention "≥2 candidate" or "multiple candidates".
        desc = rank_target["description"].lower()
        assert "2 candidate" in desc or "multiple candidate" in desc, (
            f"protocol must specify rank=0 excluded when <2 candidates: {rank_target['description']}"
        )


# ---------------------------------------------------------------------------
# R5 #10: runner output binds protocol hash + source commit
# ---------------------------------------------------------------------------


class TestRunnerOutputBinding:
    """R5 #10: runner ``summary.json`` must include ``protocol_hash`` + ``source_commit``.

    Audit R5 #10 requires the runner output to bind:
    - ``protocol_hash`` (SHA256 of frozen protocol.json)
    - ``source_commit`` (git HEAD at run time)
    - ``source_commit_dirty`` (whether working tree had uncommitted changes)

    This test verifies the runner module's ``_load_protocol_binding``
    function produces all required fields. A separate integration test
    (running the full runner) would verify they end up in ``summary.json``.
    """

    def test_load_protocol_binding_returns_all_required_fields(self):
        """``_load_protocol_binding`` must return all 5 required fields."""
        # Import the runner module by path (it's a script, not a package).
        import importlib.util
        runner_path = _HERE.parent / "scripts" / "run_m6_evaluation.py"
        spec = importlib.util.spec_from_file_location("run_m6_evaluation", runner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        binding = module._load_protocol_binding()

        required_fields = {
            "protocol_version", "protocol_hash", "protocol_path",
            "source_commit", "source_commit_dirty",
        }
        assert set(binding.keys()) >= required_fields, (
            f"missing fields: {required_fields - set(binding.keys())}"
        )

    def test_protocol_hash_matches_sha256_of_protocol_json(self):
        """``protocol_hash`` must equal SHA256(protocol.json bytes)."""
        import importlib.util
        runner_path = _HERE.parent / "scripts" / "run_m6_evaluation.py"
        spec = importlib.util.spec_from_file_location("run_m6_evaluation_binding", runner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        binding = module._load_protocol_binding()

        # Recompute the expected hash directly from protocol.json.
        protocol_bytes = _PROTOCOL_PATH.read_bytes()
        expected_hash = hashlib.sha256(protocol_bytes).hexdigest()
        assert binding["protocol_hash"] == expected_hash, (
            f"protocol_hash mismatch: binding={binding['protocol_hash']} "
            f"expected={expected_hash}"
        )

    def test_protocol_version_matches_protocol_json(self):
        """``protocol_version`` must match the value in protocol.json."""
        import importlib.util
        runner_path = _HERE.parent / "scripts" / "run_m6_evaluation.py"
        spec = importlib.util.spec_from_file_location("run_m6_evaluation_version", runner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        binding = module._load_protocol_binding()
        protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))

        assert binding["protocol_version"] == protocol["protocol_version"], (
            f"protocol_version mismatch: binding={binding['protocol_version']} "
            f"protocol={protocol['protocol_version']}"
        )

    def test_source_commit_is_not_unknown_when_git_available(self):
        """If git is available, ``source_commit`` should not be UNKNOWN.

        This test is lenient: if git is not installed or the workspace is
        not a git repo, ``source_commit`` may be UNKNOWN or GIT_ERROR.
        We only assert that the field exists and is a string.
        """
        import importlib.util
        runner_path = _HERE.parent / "scripts" / "run_m6_evaluation.py"
        spec = importlib.util.spec_from_file_location("run_m6_evaluation_commit", runner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        binding = module._load_protocol_binding()
        assert isinstance(binding["source_commit"], str)
        assert isinstance(binding["source_commit_dirty"], bool)

    def test_git_output_decode_is_utf8_fail_safe(self):
        """Invalid locale bytes cannot raise a reader-thread exception."""
        import importlib.util

        runner_path = _HERE.parent / "scripts" / "run_m6_evaluation.py"
        spec = importlib.util.spec_from_file_location(
            "run_m6_evaluation_decode", runner_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        decoded = module._decode_git_output(b"commit-\xff")
        assert decoded.startswith("commit-")
        assert "\ufffd" in decoded


# ---------------------------------------------------------------------------
# R4: protocol file integrity
# ---------------------------------------------------------------------------


class TestProtocolIntegrity:
    """R4: M6 v2 protocol files exist and SHA matches.

    Audit F-04 / R4 requires:
    - ``research/evidence/m6_v2_protocol/protocol.md`` — human-readable
    - ``research/evidence/m6_v2_protocol/protocol.json`` — machine-readable
    - ``research/evidence/m6_v2_protocol/protocol.sha256`` — SHA256 of protocol.json
    """

    def test_protocol_directory_exists(self):
        assert _PROTOCOL_DIR.is_dir(), f"protocol directory missing: {_PROTOCOL_DIR}"

    def test_protocol_md_exists(self):
        md_path = _PROTOCOL_DIR / "protocol.md"
        assert md_path.is_file(), f"protocol.md missing: {md_path}"

    def test_protocol_json_exists_and_parses(self):
        assert _PROTOCOL_PATH.is_file(), f"protocol.json missing: {_PROTOCOL_PATH}"
        protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
        assert isinstance(protocol, dict)

    def test_protocol_sha256_file_exists(self):
        assert _PROTOCOL_SHA_PATH.is_file(), (
            f"protocol.sha256 missing: {_PROTOCOL_SHA_PATH}"
        )

    def test_protocol_sha256_matches_actual_file_hash(self):
        """SHA in protocol.sha256 must match SHA256(protocol.json bytes)."""
        sha_text = _PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip()
        # The file may contain "<hash>  protocol.json" or just "<hash>".
        recorded_hash = sha_text.split()[0] if sha_text else ""
        assert recorded_hash, f"protocol.sha256 is empty or malformed: {sha_text!r}"

        actual_hash = hashlib.sha256(_PROTOCOL_PATH.read_bytes()).hexdigest()
        assert recorded_hash == actual_hash, (
            f"protocol.sha256 mismatch: recorded={recorded_hash} actual={actual_hash}"
        )

    def test_protocol_json_has_required_top_level_keys(self):
        """protocol.json must contain the R4 required top-level keys."""
        protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
        required_keys = {
            "protocol_version", "frozen_at", "status",
            "primary_targets", "feature_schema", "split_unit",
            "baseline_parameters", "three_way_comparison_rule",
            "multi_seed_stability_rule", "invalid_run_conditions",
            "amendment_rule", "amendment_history",
        }
        missing = required_keys - set(protocol.keys())
        assert not missing, f"protocol.json missing required keys: {missing}"

    def test_protocol_has_five_primary_targets(self):
        """protocol.json must freeze exactly 5 primary targets (§16.6)."""
        protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
        targets = protocol["primary_targets"]
        assert len(targets) == 5, f"expected 5 primary targets, got {len(targets)}"

        target_names = {t["name"] for t in targets}
        expected_names = {
            "energy_delta", "position_change", "edge_change_count",
            "executed_candidate_rank", "multi_step_energy_delta",
        }
        assert target_names == expected_names, (
            f"primary target names mismatch: {target_names} vs {expected_names}"
        )

    def test_protocol_status_is_frozen(self):
        """protocol.json status must contain 'FROZEN'."""
        protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
        status = protocol["status"].upper()
        assert "FROZEN" in status, (
            f"protocol status must be FROZEN, got: {protocol['status']}"
        )

    def test_protocol_amendment_history_v0_1_0_was_initial_freeze(self):
        """a. v0.1.0 初始冻结时 amendment 为空（历史快照）。

        v0.1.0 的 protocol.json 已被 v0.2.0 覆盖，无法从当前文件直接读取
        v0.1.0 的 amendment_history。改为从 protocol.md 的 Amendment history
        中验证 v0.1.0 被明确标注为 "Initial freeze"——这语义上等价于
        amendment_history 为空（初始冻结 = 无任何前置修订）。
        """
        md_path = _PROTOCOL_DIR / "protocol.md"
        md_text = md_path.read_text(encoding="utf-8")
        # 定位 v0.1.0 条目段落
        marker = "#### v0.1.0 —"
        idx = md_text.find(marker)
        assert idx != -1, "protocol.md Amendment history missing v0.1.0 entry"
        # 截取 v0.1.0 段落（到下一个 #### 或文件末尾）
        next_idx = md_text.find("####", idx + len(marker))
        v010_block = md_text[idx : next_idx if next_idx != -1 else len(md_text)]
        assert "Initial freeze" in v010_block, (
            f"v0.1.0 entry must be marked as 'Initial freeze' (semantic equivalent "
            f"of empty amendment_history), got: {v010_block!r}"
        )

    def test_protocol_amendment_history_v0_2_0_is_complete_and_incremented(self):
        """b. v0.2.0 amendment 非空且字段完整（当前状态）+ c. version 必须递增."""
        protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
        # b. amendment 非空且字段完整
        history = protocol["amendment_history"]
        assert len(history) == 1, (
            f"v0.2.0 protocol must have exactly 1 amendment entry, got {len(history)}"
        )
        amendment = history[0]
        required_amendment_keys = {"version", "date", "changed_by", "change", "reason", "impact"}
        missing_keys = required_amendment_keys - set(amendment.keys())
        assert not missing_keys, f"amendment entry missing keys: {missing_keys}"
        assert amendment["version"] == "m6_v2_v0.2.0", (
            f"amendment version must match current protocol_version, got {amendment['version']}"
        )
        # c. version 必须递增：从 supersedes 解析出 v0.1.0，断言 v0.2.0 > v0.1.0
        supersedes = protocol["supersedes"]
        prev_match = re.search(r"m6_v2_v(\d+\.\d+\.\d+)", supersedes)
        assert prev_match, f"cannot parse previous version from supersedes: {supersedes!r}"
        prev_version = prev_match.group(1)
        cur_version = protocol["protocol_version"].replace("m6_v2_v", "")
        prev_parts = tuple(int(x) for x in prev_version.split("."))
        cur_parts = tuple(int(x) for x in cur_version.split("."))
        assert cur_parts > prev_parts, (
            f"protocol_version must strictly increment: prev={prev_version} cur={cur_version}"
        )
        # d（部分）: amendment.impact 必须显式声明未被静默修改的核心字段
        impact = amendment["impact"]
        preserved_fields = [
            "primary_targets", "feature_schema", "split_unit",
            "baseline_parameters", "three_way_comparison_rule",
            "multi_seed_stability_rule", "invalid_run_conditions",
        ]
        for field in preserved_fields:
            assert field in impact, (
                f"amendment.impact must explicitly mention preserved field '{field}' "
                f"(audit F-04: no silent modification), got impact: {impact!r}"
            )

    def test_protocol_amendment_preserves_core_fields_and_refreshes_hash(self):
        """d. 原 primary targets / feature schema / baseline parameters / verdict rule
        未被静默修改（回归保护）+ e. amendment 后生成新的 protocol hash."""
        protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
        # d. primary_targets 仍是 5 个且名称不变
        targets = protocol["primary_targets"]
        assert len(targets) == 5, f"primary_targets count drift: {len(targets)}"
        expected_names = {
            "energy_delta", "position_change", "edge_change_count",
            "executed_candidate_rank", "multi_step_energy_delta",
        }
        assert {t["name"] for t in targets} == expected_names, (
            f"primary_targets names drift: {[t['name'] for t in targets]}"
        )
        # d. feature_schema 仍禁用硬编码 slice
        forbidden = protocol["feature_schema"].get("hardcoded_slices_forbidden", [])
        assert "1:8" in forbidden and "8:12" in forbidden, (
            f"feature_schema.hardcoded_slices_forbidden drift: {forbidden}"
        )
        # d. baseline_parameters 仍是 7 个且名称不变
        baselines = protocol["baseline_parameters"]
        assert len(baselines) == 7, f"baseline_parameters count drift: {len(baselines)}"
        expected_baselines = {
            "PersistenceBaseline", "MeanDeltaBaseline", "LinearRidgeBaseline",
            "XGBoostBaseline", "NoActionBaseline", "ShuffledActionBaseline",
            "OracleUpperBound",
        }
        assert {b["name"] for b in baselines} == expected_baselines, (
            f"baseline_parameters names drift: {[b['name'] for b in baselines]}"
        )
        # d. verdict rule 字段完整
        tw = protocol["three_way_comparison_rule"]
        assert "pass_condition_per_target" in tw and "tie_margin" in tw and "joint_training_verdict" in tw, (
            f"three_way_comparison_rule missing keys: {list(tw.keys())}"
        )
        ms = protocol["multi_seed_stability_rule"]
        assert "stable_passing_condition" in ms and "final_verdict" in ms, (
            f"multi_seed_stability_rule missing keys: {list(ms.keys())}"
        )
        # e. amendment 后生成新的 protocol hash：当前 hash 不等于 v0.1.0 的历史 hash
        sha_text = _PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip()
        recorded_hash = sha_text.split()[0] if sha_text else ""
        actual_hash = hashlib.sha256(_PROTOCOL_PATH.read_bytes()).hexdigest()
        assert recorded_hash == actual_hash, (
            f"protocol.sha256 mismatch: recorded={recorded_hash} actual={actual_hash}"
        )
        # v0.1.0 的历史 hash 来自 protocol.md 的 Amendment history
        md_text = (_PROTOCOL_DIR / "protocol.md").read_text(encoding="utf-8")
        v010_hash_match = re.search(r"SHA256 \(v0\.1\.0\)[^0-9a-f]*([0-9a-f]{64})", md_text)
        assert v010_hash_match, "cannot extract v0.1.0 SHA256 from protocol.md"
        v010_hash = v010_hash_match.group(1)
        assert actual_hash != v010_hash, (
            f"amendment must refresh protocol hash: cur={actual_hash} == v0.1.0={v010_hash}"
        )

    def test_protocol_forbids_hardcoded_slices(self):
        """protocol.json feature_schema must list forbidden hardcoded slices."""
        protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
        forbidden = protocol["feature_schema"].get("hardcoded_slices_forbidden", [])
        assert "1:8" in forbidden, "protocol must forbid '1:8' hardcoded slice (F-02)"
        assert "8:12" in forbidden, "protocol must forbid '8:12' hardcoded slice (F-02)"

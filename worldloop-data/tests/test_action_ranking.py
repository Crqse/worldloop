"""Tests for worldloop_data.evaluation.action_ranking (M8 Beta B5)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from worldloop_data.evaluation.data_loader import TransitionSample
from worldloop_data.evaluation.action_ranking import (
    attach_energy_outcomes,
    build_ranking_groups,
    evaluate_ranking,
    load_energy_outcomes,
)


def mk(
    episode_id: str,
    seed: str,
    tick: int,
    state_before_hash: str,
) -> TransitionSample:
    return TransitionSample(
        tick=tick,
        action_type_idx=0,
        agent_id_idx=0,
        target_node_idx=0,
        target_agent_idx=-1,
        has_params=0,
        energy_delta=0.0,
        position_change_idx=0,
        edge_change_count=0,
        executed_candidate_rank=0,
        multi_step_energy_delta=0.0,
        episode_id=episode_id,
        seed=seed,
        split="test",
        policy_id="random",
        state_before_hash=state_before_hash,
        state_after_hash="after",
    )


class TestBuildRankingGroups:
    def test_fork_point_grouping(self):
        samples = [
            mk("seed44_run2", "44", 2, "hA"),               # factual
            mk("seed44_run2_cf_t2_b0", "44", 2, "hA"),      # sibling
            mk("seed44_run2_cf_t2_b1", "44", 2, "hA"),      # sibling
            mk("seed44_run2", "44", 3, "hB"),               # lone row
        ]
        groups = build_ranking_groups(samples)
        assert len(groups) == 1
        assert groups[0].key == ("44", 2, "hA")
        assert groups[0].indices == (0, 1, 2)

    def test_singleton_groups_dropped(self):
        samples = [mk("seed44_run2", "44", t, f"h{t}") for t in range(4)]
        assert build_ranking_groups(samples) == []

    def test_min_group_size_one_keeps_singletons(self):
        samples = [mk("seed44_run2", "44", 1, "h1")]
        groups = build_ranking_groups(samples, min_group_size=1)
        assert len(groups) == 1


class TestEvaluateRanking:
    def _two_group_setup(self):
        samples = [
            mk("seed44_run2", "44", 2, "hA"),
            mk("seed44_run2_cf_t2_b0", "44", 2, "hA"),
            mk("seed44_run2", "44", 4, "hB"),
            mk("seed44_run2_cf_t4_b0", "44", 4, "hB"),
        ]
        groups = build_ranking_groups(samples)
        y_true = np.array([2.0, -1.0, 0.0, 3.0])
        return groups, y_true

    def test_oracle_prediction_is_perfect(self):
        groups, y_true = self._two_group_setup()
        m = evaluate_ranking(groups, y_true, y_true.copy())
        assert m.ranking_accuracy == 1.0
        assert m.top1_regret == 0.0
        assert m.energy_mae == 0.0
        assert m.direction_consistency == 1.0
        assert m.n_groups == 2
        assert m.n_candidates == 4

    def test_anti_oracle_is_worse(self):
        """Metric direction: inverting predictions must not score better."""
        groups, y_true = self._two_group_setup()
        oracle = evaluate_ranking(groups, y_true, y_true.copy())
        anti = evaluate_ranking(groups, y_true, -y_true)
        assert anti.ranking_accuracy == 0.0
        assert anti.ranking_accuracy < oracle.ranking_accuracy
        assert anti.top1_regret > oracle.top1_regret
        assert anti.direction_consistency == 0.0

    def test_tie_tolerant_top1(self):
        # Both candidates share the best true outcome — any pick is a hit.
        samples = [
            mk("seed44_run2", "44", 2, "hA"),
            mk("seed44_run2_cf_t2_b0", "44", 2, "hA"),
        ]
        groups = build_ranking_groups(samples)
        y_true = np.array([1.5, 1.5])
        m = evaluate_ranking(groups, y_true, np.array([0.0, 9.0]))
        assert m.ranking_accuracy == 1.0
        assert m.top1_regret == 0.0
        # All pairs tied → direction consistency undefined (NaN kept,
        # 数字诚实: never silently replaced).
        assert np.isnan(m.direction_consistency)

    def test_empty_groups_yield_nan(self):
        m = evaluate_ranking([], np.array([]), np.array([]))
        assert m.n_groups == 0
        assert np.isnan(m.ranking_accuracy)
        assert np.isnan(m.energy_mae)


class TestEnergyOutcomes:
    def _write_jsonl(self, path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_load_energy_outcomes_extracts_agent_energy_change(self, tmp_path):
        records = [
            {
                "tick": 2,
                "provenance": {"episode_id": "seed44_run2"},
                "executed_actions": {"a0": {"action_type": "CONSUME"}},
                "state_delta": {
                    "entity_changes": {
                        "changes": [
                            {
                                "entity_id": "a0",
                                "column": "energy",
                                "kind": "update",
                                "before": 5.0,
                                "after": 7.0,
                            },
                            {
                                "entity_id": "a1",
                                "column": "energy",
                                "kind": "update",
                                "before": 1.0,
                                "after": 0.0,
                            },
                        ]
                    }
                },
            },
            {
                # No energy change for the focal agent → outcome 0.0.
                "tick": 3,
                "provenance": {"episode_id": "seed44_run2"},
                "executed_actions": {"a0": {"action_type": "TRADE"}},
                "state_delta": {"entity_changes": {"changes": []}},
            },
        ]
        path = tmp_path / "transitions.jsonl"
        self._write_jsonl(path, records)
        outcomes = load_energy_outcomes(path)
        assert outcomes[("seed44_run2", 2)] == pytest.approx(2.0)
        assert outcomes[("seed44_run2", 3)] == 0.0

    def test_attach_energy_outcomes_aligns_and_defaults(self):
        samples = [
            mk("seed44_run2", "44", 2, "hA"),
            mk("seed44_run2", "44", 9, "hZ"),  # missing from outcomes
        ]
        outcomes = {("seed44_run2", 2): -1.5}
        arr = attach_energy_outcomes(samples, outcomes)
        assert arr.tolist() == [-1.5, 0.0]

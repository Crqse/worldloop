"""Tests for worldloop_data.evaluation.treatment_comparison (M8 Beta B5).

Includes the required D_matched equal-sample-size guard: |D_matched|
MUST equal |C_transition| (controls for "D is just more rows").
"""
from __future__ import annotations

import numpy as np
import pytest

from worldloop_data.evaluation.data_loader import TransitionSample
from worldloop_data.evaluation.treatment_comparison import (
    NULL_EPSILON,
    TREATMENT_NAMES,
    PairedStats,
    aggregate_model_seed_diffs,
    build_treatments,
    downsample_matched,
    is_counterfactual_sample,
    paired_stats,
    prefix_subset,
    verdict_from_stats,
)


def mk(
    episode_id: str,
    tick: int,
    policy_id: str = "random",
    seed: str = "42",
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
        split="train",
        policy_id=policy_id,
        state_before_hash=f"h{episode_id}_{tick}",
        state_after_hash="after",
    )


def _main_rows(n_conv: int = 8, n_branch: int = 6) -> list[TransitionSample]:
    conv = [mk("seed42_run0", t) for t in range(n_conv)]
    branch = [
        mk(
            f"seed42_run0_cf_t{2 + 2 * (i // 3)}_b{i % 3}",
            2 + 2 * (i // 3),
            policy_id="counterfactual_branch",
        )
        for i in range(n_branch)
    ]
    return conv + branch


class TestIsCounterfactualSample:
    def test_by_policy_id(self):
        assert is_counterfactual_sample(
            mk("weird_episode", 0, policy_id="counterfactual_branch")
        )

    def test_by_episode_suffix(self):
        assert is_counterfactual_sample(mk("seed42_run0_cf_t2_b0", 2))

    def test_conventional_row(self):
        assert not is_counterfactual_sample(mk("seed42_run0", 2))


class TestPrefixSubset:
    def test_deterministic_tick_prefix(self):
        rows = [mk("seed42_run0", t) for t in range(8)]
        a = prefix_subset(rows, 0.25)
        b = prefix_subset(list(reversed(rows)), 0.25)
        assert [s.tick for s in a] == [0, 1]
        # Order-insensitive determinism: same subset regardless of input order.
        assert [s.tick for s in b] == [0, 1]

    def test_fraction_bounds(self):
        with pytest.raises(ValueError):
            prefix_subset([mk("seed42_run0", 0)], 0.0)
        with pytest.raises(ValueError):
            prefix_subset([mk("seed42_run0", 0)], 1.5)


class TestDownsampleMatched:
    def test_deterministic_and_exact_size(self):
        rows = [mk("seed42_run0", t) for t in range(10)]
        a = downsample_matched(rows, 4, seed=7)
        b = downsample_matched(rows, 4, seed=7)
        assert len(a) == 4
        assert [s.tick for s in a] == [s.tick for s in b]

    def test_never_upsamples(self):
        rows = [mk("seed42_run0", t) for t in range(3)]
        with pytest.raises(ValueError, match="downsample"):
            downsample_matched(rows, 5, seed=0)

    def test_n_equal_returns_all(self):
        rows = [mk("seed42_run0", t) for t in range(3)]
        assert downsample_matched(rows, 3, seed=0) == rows


class TestBuildTreatments:
    def test_treatment_names_and_composition(self):
        main = _main_rows(n_conv=8, n_branch=6)
        random_rows = [mk("seed42_run0", t, policy_id="random") for t in range(8)]
        treatments = build_treatments(main, random_rows)
        assert set(treatments) == set(TREATMENT_NAMES)
        # C is branch-free; D = C + branch rows.
        assert all(
            not is_counterfactual_sample(s) for s in treatments["C_transition"]
        )
        assert len(treatments["D_counterfactual"]) == 8 + 6
        assert any(
            is_counterfactual_sample(s)
            for s in treatments["D_counterfactual"]
        )
        assert len(treatments["A_small"]) == 2  # ceil(0.25 * 8)
        assert len(treatments["B_random"]) == 8

    def test_d_matched_has_exactly_c_sample_count(self):
        """REQUIRED M8 guard: |D_matched| == |C| (same-budget control)."""
        main = _main_rows(n_conv=10, n_branch=7)
        random_rows = [mk("seed42_run0", t) for t in range(10)]
        treatments = build_treatments(main, random_rows)
        assert len(treatments["D_matched"]) == len(treatments["C_transition"])
        assert len(treatments["D_matched"]) < len(
            treatments["D_counterfactual"]
        )

    def test_d_matched_is_deterministic(self):
        main = _main_rows()
        random_rows = [mk("seed42_run0", t) for t in range(8)]
        t1 = build_treatments(main, random_rows, match_seed=123)
        t2 = build_treatments(main, random_rows, match_seed=123)
        assert [s.episode_id for s in t1["D_matched"]] == [
            s.episode_id for s in t2["D_matched"]
        ]

    def test_counterfactual_rows_in_b_random_rejected(self):
        main = _main_rows()
        bad_random = [mk("seed42_run0_cf_t2_b0", 2)]
        with pytest.raises(ValueError, match="B_random"):
            build_treatments(main, bad_random)


class TestPairedStats:
    def test_all_positive_diffs(self):
        stats = paired_stats([0.02, 0.03, 0.04, 0.05], seed=0)
        assert stats.mean_diff == pytest.approx(0.035)
        assert stats.ci_low > 0
        assert stats.direction_consistency == 1.0
        assert stats.n_pairs == 4

    def test_nan_diffs_dropped(self):
        stats = paired_stats([0.1, float("nan"), 0.3], seed=0)
        assert stats.n_pairs == 2
        assert stats.mean_diff == pytest.approx(0.2)

    def test_empty_returns_nan(self):
        stats = paired_stats([])
        assert stats.n_pairs == 0
        assert np.isnan(stats.mean_diff)

    def test_single_pair_ci_collapses(self):
        stats = paired_stats([0.1])
        assert stats.ci_low == stats.ci_high == pytest.approx(0.1)

    def test_model_seeds_are_aggregated_within_data_seed(self):
        seed_means, diagnostics = aggregate_model_seed_diffs(
            {
                (600, 0): -0.7,
                (600, 1): -0.6,
                (600, 2): -0.8,
                (601, 0): -0.5,
                (601, 1): -0.5,
                (601, 2): -0.5,
            }
        )
        assert seed_means == pytest.approx({600: -0.7, 601: -0.5})
        assert diagnostics[600]["n_model_seeds"] == 3
        assert diagnostics[600]["std"] > 0
        stats = paired_stats(list(seed_means.values()), seed=0)
        assert stats.n_pairs == 2


class TestVerdicts:
    def _stats(self, mean, lo, hi, n=9):
        return PairedStats(
            mean_diff=mean,
            ci_low=lo,
            ci_high=hi,
            direction_consistency=1.0,
            n_pairs=n,
            diffs=(mean,) * n,
        )

    def test_pass_when_ci_above_zero(self):
        assert verdict_from_stats(self._stats(0.05, 0.01, 0.09)) == "PASS"

    def test_negative_when_ci_below_zero(self):
        assert verdict_from_stats(self._stats(-0.05, -0.09, -0.01)) == "NEGATIVE"

    def test_null_when_ci_spans_zero_and_mean_tiny(self):
        tiny = NULL_EPSILON / 2
        assert verdict_from_stats(self._stats(tiny, -0.02, 0.02)) == "NULL"

    def test_inconclusive_when_ci_spans_zero_and_mean_large(self):
        assert (
            verdict_from_stats(self._stats(0.04, -0.01, 0.09)) == "INCONCLUSIVE"
        )

    def test_lower_is_better_flips_direction(self):
        stats = self._stats(-0.05, -0.09, -0.01)
        assert verdict_from_stats(stats, higher_is_better=False) == "PASS"

    def test_no_pairs_is_inconclusive(self):
        assert verdict_from_stats(paired_stats([])) == "INCONCLUSIVE"

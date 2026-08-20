"""Tests for worldloop_data.evaluation.grouped_split (M8 Beta B5).

Guards required by the M8 task brief:
- random ROW-level splitting is forbidden (strategy rejection + the
  leakage assertion catches a synthetic random row split);
- branch siblings (counterfactual episodes sharing a fork point with
  their parent) never cross splits.
"""
from __future__ import annotations

import random

import pytest

from worldloop_data.evaluation.data_loader import TransitionSample
from worldloop_data.evaluation.grouped_split import (
    GroupKey,
    GroupLeakageError,
    assert_branch_siblings_together,
    assert_no_group_leakage,
    episode_family,
    group_key_for_sample,
    grouped_split,
    is_branch_episode,
)


def mk(
    episode_id: str,
    seed: str,
    tick: int = 0,
    state_before_hash: str = "hash0",
    policy_id: str = "random",
    split: str = "unknown",
) -> TransitionSample:
    """Minimal TransitionSample for split tests."""
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
        split=split,
        policy_id=policy_id,
        state_before_hash=state_before_hash,
        state_after_hash="after0",
    )


class TestEpisodeFamily:
    def test_branch_episode_maps_to_parent(self):
        assert episode_family("seed42_run0_cf_t6_b2") == "seed42_run0"

    def test_plain_episode_maps_to_itself(self):
        assert episode_family("seed42_run0") == "seed42_run0"

    def test_is_branch_episode(self):
        assert is_branch_episode("seed42_run0_cf_t6_b2")
        assert not is_branch_episode("seed42_run0")
        assert not is_branch_episode("seed42_run0_cf_tX_b0")

    def test_group_key_bundles_seed_and_family(self):
        s = mk("seed42_run0_cf_t2_b0", seed="42")
        assert group_key_for_sample(s) == GroupKey(seed="42", family="seed42_run0")


class TestGroupedSplit:
    def test_assigns_by_seed(self):
        samples = [
            mk("seed42_run0", "42"),
            mk("seed42_run0_cf_t2_b0", "42"),
            mk("seed43_run1", "43"),
        ]
        splits = grouped_split(samples, {"42": "train", "43": "test"})
        assert {s.episode_id for s in splits["train"]} == {
            "seed42_run0",
            "seed42_run0_cf_t2_b0",
        }
        assert [s.episode_id for s in splits["test"]] == ["seed43_run1"]

    @pytest.mark.parametrize("strategy", ["random_rows", "row", "random", "shuffle"])
    def test_random_row_splitting_is_forbidden(self, strategy):
        """M8 guard: any row-level strategy is rejected by construction."""
        with pytest.raises(ValueError, match="forbidden|not allowed"):
            grouped_split([mk("seed42_run0", "42")], {"42": "train"}, strategy=strategy)

    def test_missing_seed_raises_instead_of_silent_drop(self):
        with pytest.raises(GroupLeakageError, match="seed_split_map"):
            grouped_split([mk("seed99_run0", "99")], {"42": "train"})


class TestAssertNoGroupLeakage:
    def test_clean_split_passes(self):
        splits = {
            "train": [
                mk("seed42_run0", "42", tick=2, state_before_hash="hA"),
                mk("seed42_run0_cf_t2_b0", "42", tick=2, state_before_hash="hA"),
            ],
            "test": [mk("seed43_run1", "43", tick=2, state_before_hash="hB")],
        }
        assert_no_group_leakage(splits)  # must not raise

    def test_same_seed_in_two_splits_detected(self):
        splits = {
            "train": [mk("seed42_run0", "42")],
            "test": [mk("seed42_run0", "42", tick=1)],
        }
        with pytest.raises(GroupLeakageError, match="seed '42'"):
            assert_no_group_leakage(splits)

    def test_branch_sibling_family_cross_split_detected(self):
        # Branch episode carrying a DIFFERENT seed label placed in
        # another split: the family check must still catch it.
        splits = {
            "train": [mk("seed42_run0", "42")],
            "test": [mk("seed42_run0_cf_t4_b1", "43")],
        }
        with pytest.raises(GroupLeakageError, match="episode family"):
            assert_no_group_leakage(splits)

    def test_fork_point_cross_split_detected(self):
        # Two rows sharing the counterfactual fork point (seed, tick,
        # state_before_hash) in different splits — the fork-point guard.
        splits = {
            "train": [mk("seed42_run0", "42", tick=6, state_before_hash="hF")],
            "test": [
                mk("seed42_run0_cf_t6_b0", "42", tick=6, state_before_hash="hF")
            ],
        }
        with pytest.raises(GroupLeakageError):
            assert_no_group_leakage(splits)

    def test_random_row_split_is_caught(self):
        """M8 guard: a random row-level split scatters seeds across
        splits and MUST be flagged as leakage."""
        samples = [
            mk("seed42_run0", "42", tick=t, state_before_hash=f"h{t}")
            for t in range(10)
        ] + [
            mk("seed43_run1", "43", tick=t, state_before_hash=f"g{t}")
            for t in range(10)
        ]
        rng = random.Random(0)
        splits: dict[str, list[TransitionSample]] = {"train": [], "test": []}
        for s in samples:
            splits["train" if rng.random() < 0.5 else "test"].append(s)
        with pytest.raises(GroupLeakageError):
            assert_no_group_leakage(splits)


class TestBranchSiblingsTogether:
    def test_siblings_with_parent_pass(self):
        splits = {
            "train": [
                mk("seed42_run0", "42"),
                mk("seed42_run0_cf_t2_b0", "42"),
                mk("seed42_run0_cf_t2_b1", "42"),
            ],
            "test": [mk("seed43_run1", "43")],
        }
        assert_branch_siblings_together(splits)  # must not raise

    def test_branch_sibling_crossing_split_detected(self):
        """M8 guard: a branch episode must never leave its parent's split."""
        splits = {
            "train": [mk("seed42_run0", "42")],
            "test": [mk("seed42_run0_cf_t2_b0", "42")],
        }
        with pytest.raises(GroupLeakageError, match="cross splits"):
            assert_branch_siblings_together(splits)

    def test_siblings_without_parent_row_still_guarded(self):
        # Even when the factual row was filtered out, siblings of the
        # same family must stay in one split.
        splits = {
            "train": [mk("seed42_run0_cf_t2_b0", "42")],
            "test": [mk("seed42_run0_cf_t2_b1", "42")],
        }
        with pytest.raises(GroupLeakageError):
            assert_branch_siblings_together(splits)

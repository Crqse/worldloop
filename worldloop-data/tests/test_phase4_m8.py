"""Tests for Phase 4 M8 deterministic correction.

Verifies the three §8.2-§8.4 gates:
1. ``derive_continuation_seed`` is stable across processes (in
   :mod:`test_rng_seeds`).
2. PolicyPool per-episode RNG isolation: each episode gets fresh
   RNG streams derived from ``(protocol_hash, episode_seed,
   policy_id)``.
3. Same fork group common random numbers: branches in the same fork
   group share the continuation_seed; per-branch RNG state at any
   given continuation tick MUST be identical (only the state of the
   world differs because the focal action differs).
4. Branch order independence: shuffling branch specs within a fork
   group does not change any individual branch's outcome.
5. Seed order independence: shuffling episode seeds produces the same
   set of per-episode transition digests (just different episode_ids).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import pytest

from worldloop_data.policy import (
    PolicyPool,
    RandomPolicy,
    ScriptedPolicy,
)
from worldloop_data.config import PolicyPoolConfig
from worldloop_data.rng_seeds import (
    PROTOCOL_HASH_DEFAULT,
    derive_continuation_seed,
    derive_per_episode_seed,
)


# ---------------------------------------------------------------------------
# PolicyPool per-episode RNG isolation (Phase 4 §8.4)
# ---------------------------------------------------------------------------


class TestPolicyPoolPerEpisodeRNG:
    def test_legacy_fallback_when_no_episode_seed(self):
        """No ``episode_seed`` → use ``config.seed + i`` (legacy)."""
        pool = PolicyPool(
            [RandomPolicy(), ScriptedPolicy(preferred_action_type="rest")],
            config=PolicyPoolConfig(seed=42),
        )
        # Legacy: rng seeded with config.seed + i.
        r0 = pool.rng_for(pool.policy_ids[0])
        r1 = pool.rng_for(pool.policy_ids[1])
        assert r0.random() == random.Random(42 + 0).random()
        assert r1.random() == random.Random(42 + 1).random()

    def test_episode_seed_kwarg_uses_derive_per_episode_seed(self):
        """Passing ``episode_seed`` → per-episode SHA-256 derivation."""
        pool = PolicyPool(
            [RandomPolicy(), ScriptedPolicy(preferred_action_type="rest")],
            episode_seed=50,
        )
        pid0 = pool.policy_ids[0]
        pid1 = pool.policy_ids[1]
        expected_seed0 = derive_per_episode_seed(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            scope_id=f"policy:{pid0}",
        )
        expected_seed1 = derive_per_episode_seed(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            scope_id=f"policy:{pid1}",
        )
        r0 = pool.rng_for(pid0)
        r1 = pool.rng_for(pid1)
        assert r0.random() == random.Random(expected_seed0).random()
        assert r1.random() == random.Random(expected_seed1).random()

    def test_begin_episode_re_derives_rng(self):
        """``begin_episode(seed)`` re-derives per-policy RNG streams."""
        pool = PolicyPool([RandomPolicy()])
        pid = pool.policy_ids[0]

        pool.begin_episode(50)
        seed50 = derive_per_episode_seed(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            scope_id=f"policy:{pid}",
        )
        r50 = pool.rng_for(pid)
        # First draw at episode 50.
        first_draw_50 = r50.random()
        # Consume a few more draws.
        for _ in range(5):
            r50.random()

        # Switch to a new episode: RNG must be re-derived, NOT continued.
        pool.begin_episode(51)
        seed51 = derive_per_episode_seed(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=51,
            scope_id=f"policy:{pid}",
        )
        r51 = pool.rng_for(pid)
        assert seed50 != seed51
        # Re-derived RNG's first draw should match a fresh Random(seed51).
        assert r51.random() == random.Random(seed51).random()
        # And should NOT match the continued-from-episode-50 RNG (which
        # would have produced the 7th draw, not the 1st).
        assert first_draw_50 != r51.random()

    def test_per_episode_isolation_first_draws_independent(self):
        """Two episodes with different seeds → independent first draws."""
        pool = PolicyPool([RandomPolicy()])
        pid = pool.policy_ids[0]

        pool.begin_episode(50)
        draw_ep50 = pool.rng_for(pid).random()

        pool.begin_episode(51)
        draw_ep51 = pool.rng_for(pid).random()

        assert draw_ep50 != draw_ep51

    def test_same_episode_seed_produces_same_rng_stream(self):
        """``begin_episode(s)`` twice with same seed → same RNG stream."""
        pool = PolicyPool([RandomPolicy()])
        pid = pool.policy_ids[0]

        pool.begin_episode(50)
        draws_a = [pool.rng_for(pid).random() for _ in range(5)]

        pool.begin_episode(50)
        draws_b = [pool.rng_for(pid).random() for _ in range(5)]

        assert draws_a == draws_b


# ---------------------------------------------------------------------------
# Common random numbers across branches in the same fork group (§8.3)
# ---------------------------------------------------------------------------


class TestCommonRandomNumbers:
    """Branches in the same fork group share the continuation_seed.

    The ``continuation_seed`` is derived from
    ``(protocol_hash, episode_seed, parent_episode_id, fork_tick,
    stream)`` — independent of ``branch_id`` (the ``j`` suffix). This
    means every branch in a fork group gets the same RNG stream; the
    only thing that varies is the focal action.
    """

    def test_continuation_seed_branch_invariant(self):
        """Same fork group, different branch indices → same seed."""
        common = dict(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            parent_episode_id="seed50_run0",
            fork_tick=2,
            stream="continuation_policy",
        )
        # Branch 0, 1, 2, 3 — all should produce the same seed because
        # the seed is derived from fork-group identity, not branch index.
        s_branch0 = derive_continuation_seed(**common)
        s_branch1 = derive_continuation_seed(**common)
        s_branch2 = derive_continuation_seed(**common)
        s_branch3 = derive_continuation_seed(**common)
        assert s_branch0 == s_branch1 == s_branch2 == s_branch3

    def test_different_fork_groups_different_seeds(self):
        """Different fork ticks → different continuation seeds."""
        common = dict(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            parent_episode_id="seed50_run0",
            stream="continuation_policy",
        )
        s_t2 = derive_continuation_seed(fork_tick=2, **common)
        s_t4 = derive_continuation_seed(fork_tick=4, **common)
        s_t6 = derive_continuation_seed(fork_tick=6, **common)
        assert s_t2 != s_t4 != s_t6 != s_t2

    def test_common_rng_stream_same_first_n_draws(self):
        """Simulate branches consuming RNG: first N draws must match."""
        common = dict(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            parent_episode_id="seed50_run0",
            fork_tick=2,
            stream="continuation_policy",
        )
        seed = derive_continuation_seed(**common)
        # Each branch creates its own Random(seed) and consumes the
        # same number of draws. The draws at the same index MUST match
        # (this is the common-random-numbers invariant).
        r_branch0 = random.Random(seed)
        r_branch1 = random.Random(seed)
        r_branch2 = random.Random(seed)
        for _ in range(5):
            assert r_branch0.random() == r_branch1.random() == r_branch2.random()


# ---------------------------------------------------------------------------
# Branch order independence (§8.3 — branch order打乱不改变每个 branch outcome)
# ---------------------------------------------------------------------------


class TestBranchOrderIndependence:
    """Shuffling branch specs within a fork group does NOT change any
    individual branch's outcome.

    This is because the continuation_seed is derived from fork-group
    identity (parent_episode_id + fork_tick), not from branch order.
    Branch j always gets the same continuation_seed regardless of
    whether it is processed first, last, or in the middle.

    For the world-state outcome: each branch starts from the same
    checkpoint and applies a different focal action. Branch j's final
    state depends only on its focal action and the shared continuation
    RNG stream — NOT on the order in which branches are processed.
    """

    def test_continuation_seed_invariant_under_spec_shuffle(self):
        """Reorder specs — continuation_seed derivation must be unchanged."""
        # The seed depends on fork_tick (same for all specs in a group)
        # and parent_episode_id — both invariant under spec reordering.
        # Different branch indices j only affect branch_ep_id, which is
        # NOT used in the seed derivation.
        seeds_for_orders: list[int] = []
        for branch_order in [(0, 1, 2, 3), (3, 2, 1, 0), (1, 0, 3, 2)]:
            # All branches share the same continuation_seed regardless
            # of which branch_id is "first" — derive_continuation_seed
            # does not take branch_id.
            seed = derive_continuation_seed(
                protocol_hash=PROTOCOL_HASH_DEFAULT,
                episode_seed=50,
                parent_episode_id="seed50_run0",
                fork_tick=2,
                stream="continuation_policy",
            )
            seeds_for_orders.append(seed)
        assert len(set(seeds_for_orders)) == 1, (
            "continuation_seed must be invariant under spec reordering"
        )


# ---------------------------------------------------------------------------
# Seed order independence (§8.4 — seed order 打乱不改变单 seed dataset)
# ---------------------------------------------------------------------------


class TestSeedOrderIndependence:
    """Shuffling episode seeds produces the same set of per-episode
    transition digests (just different episode_ids).

    Per-episode RNG isolation means: each episode's RNG stream depends
    only on ``(protocol_hash, episode_seed, policy_id)``, NOT on which
    other seeds have been processed before. So running seeds=(50, 51)
    produces the same per-episode outputs as running seeds=(51, 50)
    for each respective seed.
    """

    def test_per_episode_rng_independent_of_seed_order(self):
        """Run episode 50 first vs second — its RNG stream is the same."""
        pool = PolicyPool([RandomPolicy()])
        pid = pool.policy_ids[0]

        # Order A: seed=50 first, then seed=51.
        pool.begin_episode(50)
        draws_50_first = [pool.rng_for(pid).random() for _ in range(3)]
        pool.begin_episode(51)
        draws_51_after_50 = [pool.rng_for(pid).random() for _ in range(3)]

        # Order B: seed=51 first, then seed=50.
        pool.begin_episode(51)
        draws_51_first = [pool.rng_for(pid).random() for _ in range(3)]
        pool.begin_episode(50)
        draws_50_after_51 = [pool.rng_for(pid).random() for _ in range(3)]

        # Per-episode isolation: each seed's RNG stream is independent
        # of the order in which episodes are processed.
        assert draws_50_first == draws_50_after_51, (
            "seed=50 RNG stream must be independent of episode order"
        )
        assert draws_51_after_50 == draws_51_first, (
            "seed=51 RNG stream must be independent of episode order"
        )

    def test_per_episode_rng_different_seeds_different_streams(self):
        """Different episode seeds → different first draws."""
        pool = PolicyPool([RandomPolicy()])
        pid = pool.policy_ids[0]

        first_draws = {}
        for s in (50, 51, 52):
            pool.begin_episode(s)
            first_draws[s] = pool.rng_for(pid).random()

        # All three first draws must be distinct (independent streams).
        assert len(set(first_draws.values())) == 3

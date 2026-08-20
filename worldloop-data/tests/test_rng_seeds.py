"""Tests for worldloop_data.rng_seeds (Phase 4 §8.2-§8.4).

Verifies that the SHA-256 canonical seed derivation is:
1. Stable across calls within the same process.
2. Stable across separate Python processes (the whole point of
   replacing the built-in ``hash()``).
3. Independent of dict insertion order.
4. Sensitive to every contributing field.
5. Returns ints in the expected range.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

from worldloop_data.rng_seeds import (
    PROTOCOL_HASH_DEFAULT,
    derive_continuation_seed,
    derive_per_episode_seed,
    derive_seed,
)


# ---------------------------------------------------------------------------
# In-process stability + determinism
# ---------------------------------------------------------------------------


class TestDeriveSeedStable:
    def test_same_mapping_same_seed(self):
        m = {"a": 1, "b": "x", "c": (1, 2, 3)}
        assert derive_seed(m) == derive_seed(m)

    def test_dict_insertion_order_independent(self):
        m1 = {"a": 1, "b": 2, "c": 3}
        m2 = {"c": 3, "a": 1, "b": 2}
        m3 = {"b": 2, "c": 3, "a": 1}
        s1 = derive_seed(m1)
        s2 = derive_seed(m2)
        s3 = derive_seed(m3)
        assert s1 == s2 == s3

    def test_different_mappings_different_seeds(self):
        s1 = derive_seed({"a": 1})
        s2 = derive_seed({"a": 2})
        assert s1 != s2

    def test_returns_int_in_64bit_range(self):
        s = derive_seed({"x": 1})
        assert isinstance(s, int)
        assert 0 <= s < 2**64

    def test_empty_mapping_stable_and_nonzero(self):
        s1 = derive_seed({})
        s2 = derive_seed({})
        assert s1 == s2
        # Empty mapping still produces a real hash, not 0.
        assert s1 > 0


class TestContinuationSeed:
    def test_same_fork_group_same_seed(self):
        """Two branches in the same fork group MUST get the same seed."""
        common = dict(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            parent_episode_id="seed50_run0",
            fork_tick=2,
            stream="continuation_policy",
        )
        s1 = derive_continuation_seed(**common)
        s2 = derive_continuation_seed(**common)
        assert s1 == s2

    def test_fork_tick_change_changes_seed(self):
        common = dict(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            parent_episode_id="seed50_run0",
            stream="continuation_policy",
        )
        s_t2 = derive_continuation_seed(fork_tick=2, **common)
        s_t4 = derive_continuation_seed(fork_tick=4, **common)
        assert s_t2 != s_t4

    def test_parent_episode_change_changes_seed(self):
        s_a = derive_continuation_seed(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            parent_episode_id="seed50_run0",
            fork_tick=2,
            stream="continuation_policy",
        )
        s_b = derive_continuation_seed(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=51,
            parent_episode_id="seed51_run1",
            fork_tick=2,
            stream="continuation_policy",
        )
        assert s_a != s_b

    def test_stream_change_changes_seed(self):
        """Same fork group, different RNG streams → different seeds."""
        common = dict(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            parent_episode_id="seed50_run0",
            fork_tick=2,
        )
        s_policy = derive_continuation_seed(stream="continuation_policy", **common)
        s_exog = derive_continuation_seed(stream="exogenous", **common)
        s_sched = derive_continuation_seed(stream="agent_scheduler", **common)
        assert s_policy != s_exog
        assert s_policy != s_sched
        assert s_exog != s_sched

    def test_protocol_hash_change_changes_seed(self):
        common = dict(
            episode_seed=50,
            parent_episode_id="seed50_run0",
            fork_tick=2,
            stream="continuation_policy",
        )
        s_old = derive_continuation_seed(protocol_hash="0.1.0", **common)
        s_new = derive_continuation_seed(protocol_hash="0.2.0", **common)
        assert s_old != s_new


class TestPerEpisodeSeed:
    def test_same_episode_same_scope_same_seed(self):
        s1 = derive_per_episode_seed(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            scope_id="policy:random",
        )
        s2 = derive_per_episode_seed(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            scope_id="policy:random",
        )
        assert s1 == s2

    def test_scope_change_changes_seed(self):
        """Different policies get independent RNG streams."""
        common = dict(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
        )
        s_random = derive_per_episode_seed(scope_id="policy:random", **common)
        s_scripted = derive_per_episode_seed(scope_id="policy:scripted_move", **common)
        s_coverage = derive_per_episode_seed(scope_id="coverage", **common)
        assert s_random != s_scripted
        assert s_random != s_coverage
        assert s_scripted != s_coverage

    def test_episode_seed_change_changes_seed(self):
        """Per-episode isolation: different episodes → different seeds."""
        common = dict(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            scope_id="policy:random",
        )
        s_ep50 = derive_per_episode_seed(episode_seed=50, **common)
        s_ep51 = derive_per_episode_seed(episode_seed=51, **common)
        assert s_ep50 != s_ep51


# ---------------------------------------------------------------------------
# Cross-process stability — the whole point of replacing hash()
# ---------------------------------------------------------------------------


_CROSS_PROCESS_SCRIPT = """
import sys
sys.path.insert(0, {src_path!r})
from worldloop_data.rng_seeds import derive_continuation_seed, derive_per_episode_seed

s_cont = derive_continuation_seed(
    protocol_hash="0.1.0",
    episode_seed=50,
    parent_episode_id="seed50_run0",
    fork_tick=2,
    stream="continuation_policy",
)
s_pol = derive_per_episode_seed(
    protocol_hash="0.1.0",
    episode_seed=50,
    scope_id="policy:random",
)
print(f"{{s_cont}} {{s_pol}}")
"""


class TestCrossProcessStable:
    """Two separate Python processes MUST derive the same seeds.

    This is the M8 §8.2 gate: "fresh process digest identical". The
    built-in ``hash()`` would fail this because PYTHONHASHSEED differs
    per process; SHA-256 canonical encoding does not.
    """

    def test_two_subprocesses_produce_same_seeds(self):
        src_path = str(
            Path(__file__).resolve().parent.parent / "src"
        )
        script = _CROSS_PROCESS_SCRIPT.format(src_path=src_path)
        results = []
        for _ in range(2):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
            )
            results.append(proc.stdout.strip())
        assert results[0] == results[1], (
            "cross-process seed derivation differs — M8 §8.2 gate failed: "
            f"{results[0]!r} != {results[1]!r}"
        )

    def test_seeds_independent_of_pythonhashseed(self):
        """Explicitly set PYTHONHASHSEED to different values; seeds must match."""
        src_path = str(
            Path(__file__).resolve().parent.parent / "src"
        )
        script = _CROSS_PROCESS_SCRIPT.format(src_path=src_path)
        outputs = []
        for hashseed in ("0", "1", "42", "random"):
            env = {**__import__("os").environ, "PYTHONHASHSEED": hashseed}
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            outputs.append(proc.stdout.strip())
        assert all(o == outputs[0] for o in outputs), (
            "seed derivation depends on PYTHONHASHSEED — M8 §8.2 gate failed: "
            f"{outputs!r}"
        )


# ---------------------------------------------------------------------------
# RNG consumption — seed produces a usable random.Random stream
# ---------------------------------------------------------------------------


class TestRNGConsumption:
    def test_seed_produces_stable_random_stream(self):
        """random.Random(derived_seed) gives the same first 5 draws always."""
        seed = derive_continuation_seed(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            parent_episode_id="seed50_run0",
            fork_tick=2,
            stream="continuation_policy",
        )
        r1 = random.Random(seed)
        r2 = random.Random(seed)
        assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]

    def test_different_streams_different_first_draw(self):
        """Different stream labels → different first random draw."""
        common = dict(
            protocol_hash=PROTOCOL_HASH_DEFAULT,
            episode_seed=50,
            parent_episode_id="seed50_run0",
            fork_tick=2,
        )
        s_policy = derive_continuation_seed(stream="continuation_policy", **common)
        s_exog = derive_continuation_seed(stream="exogenous", **common)
        r_policy = random.Random(s_policy)
        r_exog = random.Random(s_exog)
        assert r_policy.random() != r_exog.random()

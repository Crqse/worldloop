"""M8 deterministic RNG seed derivation (Phase 4 §8.2-§8.4).

Replaces the legacy ``random.Random(hash(episode_id) % (2**31))``
pattern with a SHA-256 canonical-bytes derivation that is stable
across Python processes (the built-in ``hash()`` is randomized per
process via ``PYTHONHASHSEED`` since Python 3.3, breaking
cross-process digest equality).

Design rules (per main plan §8.2-§8.4):
- All seed material is canonically encoded via
  :func:`worldloop_kernel.canonical.canonical_encode` to guarantee
  deterministic byte output regardless of dict insertion order or
  platform.
- SHA-256 digest is truncated to 8 bytes (64 bits) and converted to
  a non-negative ``int`` suitable for ``random.Random(seed)``. 64
  bits is ample entropy for RNG seeding and avoids the
  ``2**31 - 1`` ceiling of the legacy ``% (2**31)`` pattern.
- The same ``(protocol_hash, episode_seed, parent_episode_id,
  fork_tick, stream)`` tuple ALWAYS produces the same
  ``continuation_seed`` in any Python process on any platform.
- Per-episode RNG derivation uses
  :func:`derive_per_episode_seed` with a ``scope_id`` (e.g.
  ``"policy:<policy_id>"`` or ``"coverage"``) so different policies
  and the coverage scheduler get independent streams within the
  same episode.

Provenance: Phase 4 §8.2-§8.4 (M8 deterministic correction).
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from worldloop_kernel.canonical import canonical_encode

__all__ = [
    "derive_seed",
    "derive_continuation_seed",
    "derive_per_episode_seed",
    "PROTOCOL_HASH_DEFAULT",
]


#: Default protocol hash used when the caller cannot supply one. This
#: is the kernel's :data:`worldloop_kernel.PROTOCOL_SCHEMA_VERSION`
#: (currently ``"0.1.0"``). Worlds with a stronger identity (e.g.
#: ``world_id + ":" + world_version``) should pass that as
#: ``protocol_hash`` instead.
PROTOCOL_HASH_DEFAULT = "0.1.0"


def derive_seed(material: Mapping[str, Any]) -> int:
    """Derive a stable 64-bit non-negative int seed from a mapping.

    The mapping is canonically encoded (sorted keys, deterministic
    scalar encoding) and SHA-256 hashed; the first 8 bytes of the
    digest become the seed. This is stable across processes and
    platforms — unlike the built-in :func:`hash`, which is randomized
    per process.

    Args:
        material: A mapping of seed-contributing fields. Keys are
            sorted lexicographically by their canonical encoding
            before hashing, so insertion order does not matter.

    Returns:
        A non-negative ``int`` in ``[0, 2**64)`` suitable for
        ``random.Random(seed)``.
    """
    encoded = canonical_encode(dict(material))
    digest = hashlib.sha256(encoded).digest()
    # First 8 bytes → unsigned 64-bit int. Big-endian for stability.
    return int.from_bytes(digest[:8], "big")


def derive_continuation_seed(
    *,
    protocol_hash: str,
    episode_seed: int,
    parent_episode_id: str,
    fork_tick: int,
    stream: str,
) -> int:
    """Derive the continuation RNG seed for a fork group (Phase 4 §8.2).

    The same fork group (same ``parent_episode_id`` + ``fork_tick``)
    MUST produce the same ``continuation_seed`` for all branches in
    that group — only the focal action varies, not the continuation
    RNG stream. This is the "common random numbers" rule from §8.3.

    Args:
        protocol_hash: Protocol identifier (e.g.
            :data:`PROTOCOL_HASH_DEFAULT` or
            ``f"{world_id}:{world_version}"``).
        episode_seed: The parent episode's RNG seed.
        parent_episode_id: The parent episode ID (e.g.
            ``"seed50_run0"``).
        fork_tick: The tick at which the fork occurred.
        stream: Stream label (e.g. ``"continuation_policy"``,
            ``"exogenous"``, ``"agent_scheduler"``).

    Returns:
        A 64-bit non-negative int seed.

    Examples
    --------
    >>> s1 = derive_continuation_seed(
    ...     protocol_hash="0.1.0",
    ...     episode_seed=50,
    ...     parent_episode_id="seed50_run0",
    ...     fork_tick=2,
    ...     stream="continuation_policy",
    ... )
    >>> s2 = derive_continuation_seed(
    ...     protocol_hash="0.1.0",
    ...     episode_seed=50,
    ...     parent_episode_id="seed50_run0",
    ...     fork_tick=2,
    ...     stream="continuation_policy",
    ... )
    >>> s1 == s2  # same fork group → same seed
    True
    """
    return derive_seed(
        {
            "protocol": protocol_hash,
            "episode_seed": int(episode_seed),
            "parent_episode_id": str(parent_episode_id),
            "fork_tick": int(fork_tick),
            "stream": str(stream),
        }
    )


def derive_per_episode_seed(
    *,
    protocol_hash: str,
    episode_seed: int,
    scope_id: str,
) -> int:
    """Derive a per-episode RNG seed for an isolated scope (Phase 4 §8.4).

    Used by :class:`PolicyPool` to derive independent per-policy RNG
    streams that are re-derived for each episode — forbidding the
    legacy pattern of sharing a single mutable RNG across episodes.

    Args:
        protocol_hash: Protocol identifier.
        episode_seed: The episode's RNG seed.
        scope_id: Scope label (e.g. ``"policy:random"``,
            ``"policy:scripted_move"``, ``"coverage"``).

    Returns:
        A 64-bit non-negative int seed.
    """
    return derive_seed(
        {
            "protocol": protocol_hash,
            "episode_seed": int(episode_seed),
            "scope": str(scope_id),
        }
    )

"""Checkpoint mapper: PettingZoo env ↔ kernel Checkpoint (A-01).

Snapshots the full restorable PettingZoo env state as a kernel
:class:`Checkpoint`. The kernel records the bytes but does NOT interpret
them; restoration is delegated back to the adapter via
:func:`restore_checkpoint`.

Design (per main plan §12.2 and §12.7 M2 Gate (d) exact restore):
- PettingZoo MPE envs support mid-trajectory restore by saving the
  unwrapped env's ``__dict__`` (agent positions, velocities, landmark
  positions, step counter, RNG state) and restoring it verbatim.
- The checkpoint includes the kernel :class:`StateView` (for hash
  verification) and an opaque pickle payload (for env restoration).
- After :func:`restore_checkpoint`, ``observe()`` MUST return a
  :class:`StateView` whose canonical hash matches the checkpoint's
  ``state_view`` hash. This is verified by K-06 validation.

Mapping (per §12.7 (h) mid-trajectory replay):
- The opaque payload is ``pickle.dumps(unwrapped_env.__dict__)``.
- The codec is ``"pickle+v1"`` (matches :data:`PETTINGZOO_PAYLOAD_CODEC`).
- The checksum is computed over (state_view hash + opaque payload hash)
  to ensure round-trip consistency.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from typing import Any

from worldloop_kernel.canonical import hash_state
from worldloop_kernel.state import StateView
from worldloop_kernel.transition import Checkpoint

from .capability import PETTINGZOO_PAYLOAD_CODEC

__all__ = ["export_checkpoint", "restore_checkpoint", "compute_checksum"]


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def compute_checksum(state_view: StateView, opaque_payload: bytes) -> str:
    """Compute a stable checksum over (state_view hash + opaque payload).

    The checksum is used by K-06 validation to verify that
    ``restore_checkpoint`` produces a state whose hash matches the
    checkpoint's ``state_view`` hash.
    """
    sv_hash = hash_state(state_view)
    payload_hash = hashlib.sha256(opaque_payload).hexdigest()
    return f"sha256:{sv_hash}:{payload_hash}"


# ---------------------------------------------------------------------------
# Export / restore
# ---------------------------------------------------------------------------


def export_checkpoint(env: Any, state_view: StateView) -> Checkpoint:
    """Snapshot the env as a kernel :class:`Checkpoint`.

    Captures the unwrapped env's ``__dict__`` via ``copy.deepcopy`` +
    ``pickle.dumps`` so that the env can be restored to exactly this
    state later. The :class:`StateView` is included for hash verification.

    Pygame rendering attributes (``game_font``, ``screen``) cannot be
    pickled and are excluded from the snapshot; they are rebuilt by the
    env on next render and do not affect simulation state.

    Parameters
    ----------
    env:
        The PettingZoo Parallel env handle (post-``reset`` or post-``step``).
    state_view:
        The :class:`StateView` corresponding to the env's current state.

    Returns
    -------
    Checkpoint
        Frozen checkpoint with opaque_payload = pickled env __dict__.
    """
    unwrapped = getattr(env, "unwrapped", env)
    # Build a filtered dict excluding non-picklable pygame rendering
    # attributes BEFORE deepcopy. copy.deepcopy invokes __reduce_ex__
    # which raises on Font/Surface objects, so we must exclude them
    # first (not after deepcopy).
    NON_PICKLABLE_KEYS = ("game_font", "screen")
    filtered = {
        k: v
        for k, v in unwrapped.__dict__.items()
        if k not in NON_PICKLABLE_KEYS
    }
    # Deep copy the filtered dict to avoid sharing mutable references
    # with the live env.
    env_dict_copy = copy.deepcopy(filtered)
    opaque_payload = pickle.dumps(env_dict_copy)
    rng_bundle = None
    np_random = env_dict_copy.get("np_random")
    bit_generator = getattr(np_random, "bit_generator", None)
    if bit_generator is not None:
        rng_bundle = {
            "np_random": json.dumps(
                bit_generator.state,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        }

    from .capability import PETTINGZOO_WORLD_ID, PETTINGZOO_WORLD_VERSION
    from worldloop_kernel.transition import PROTOCOL_SCHEMA_VERSION

    return Checkpoint(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        world_id=PETTINGZOO_WORLD_ID,
        world_version=PETTINGZOO_WORLD_VERSION,
        tick=int(state_view.meta.tick),
        state_view=state_view,
        opaque_payload=opaque_payload,
        payload_codec=PETTINGZOO_PAYLOAD_CODEC,
        capability_profile=state_view.capabilities,
        rng_bundle=rng_bundle,
        checksum=compute_checksum(state_view, opaque_payload),
    )


def restore_checkpoint(env: Any, checkpoint: Checkpoint) -> None:
    """Restore the env from a kernel :class:`Checkpoint`.

    Restores the unwrapped env's ``__dict__`` from the checkpoint's
    opaque payload. After this call, the env's observable state MUST
    match the checkpoint's ``state_view`` (verified externally via
    :func:`hash_state`).

    Pygame rendering attributes (``game_font``, ``screen``) are NOT
    restored (they were excluded from the snapshot); the env keeps its
    current rendering attributes, which are rebuilt on next render.

    Parameters
    ----------
    env:
        The PettingZoo Parallel env handle (will be mutated in place).
    checkpoint:
        The :class:`Checkpoint` to restore from.
    """
    if checkpoint.payload_codec != PETTINGZOO_PAYLOAD_CODEC:
        raise ValueError(
            f"unsupported payload codec: {checkpoint.payload_codec!r} "
            f"(expected {PETTINGZOO_PAYLOAD_CODEC!r})"
        )
    env_dict = pickle.loads(checkpoint.opaque_payload)
    unwrapped = getattr(env, "unwrapped", env)
    # Preserve existing pygame rendering attributes (not in snapshot).
    for non_picklable_key in ("game_font", "screen"):
        if non_picklable_key in unwrapped.__dict__:
            env_dict[non_picklable_key] = unwrapped.__dict__[non_picklable_key]
    # Restore __dict__ in place to preserve object identity.
    unwrapped.__dict__.clear()
    unwrapped.__dict__.update(env_dict)

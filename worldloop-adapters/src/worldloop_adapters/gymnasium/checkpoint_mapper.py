"""Checkpoint mapper: Gymnasium env ↔ kernel Checkpoint (A-05).

Snapshots the full restorable Gymnasium env state as a kernel
:class:`Checkpoint`.

Design (per main plan §12.2 and §12.7 M2 Gate (h)):
- Gymnasium envs support mid-trajectory restore by saving the unwrapped
  env's ``__dict__`` (agent state, step counter, RNG state) and restoring
  it verbatim.
- The opaque payload is ``pickle.dumps(unwrapped_env.__dict__)``.
- The codec is ``"pickle+v1"``.
"""
from __future__ import annotations

import copy
import hashlib
import pickle
from typing import Any

from worldloop_kernel.canonical import hash_state
from worldloop_kernel.state import StateView
from worldloop_kernel.transition import Checkpoint, PROTOCOL_SCHEMA_VERSION

from .capability import (
    GYMNASIUM_PAYLOAD_CODEC,
    GYMNASIUM_WORLD_ID,
    GYMNASIUM_WORLD_VERSION,
)

__all__ = ["export_checkpoint", "restore_checkpoint", "compute_checksum"]


def compute_checksum(state_view: StateView, opaque_payload: bytes) -> str:
    """Compute a stable checksum over (state_view hash + opaque payload)."""
    sv_hash = hash_state(state_view)
    payload_hash = hashlib.sha256(opaque_payload).hexdigest()
    return f"sha256:{sv_hash}:{payload_hash}"


def export_checkpoint(env: Any, state_view: StateView) -> Checkpoint:
    """Snapshot the env as a kernel :class:`Checkpoint`.

    Captures the unwrapped env's ``__dict__`` via ``copy.deepcopy`` +
    ``pickle.dumps``.
    """
    unwrapped = getattr(env, "unwrapped", env)
    # Gymnasium envs typically don't have non-picklable rendering attrs
    # (no pygame), but filter defensively in case of viewer/renderer objs.
    NON_PICKLABLE_KEYS = ("viewer", "screen", "renderer")
    filtered = {
        k: v
        for k, v in unwrapped.__dict__.items()
        if k not in NON_PICKLABLE_KEYS
    }
    env_dict_copy = copy.deepcopy(filtered)
    opaque_payload = pickle.dumps(env_dict_copy)

    return Checkpoint(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        world_id=GYMNASIUM_WORLD_ID,
        world_version=GYMNASIUM_WORLD_VERSION,
        tick=int(state_view.meta.tick),
        state_view=state_view,
        opaque_payload=opaque_payload,
        payload_codec=GYMNASIUM_PAYLOAD_CODEC,
        capability_profile=state_view.capabilities,
        checksum=compute_checksum(state_view, opaque_payload),
    )


def restore_checkpoint(env: Any, checkpoint: Checkpoint) -> None:
    """Restore the env from a kernel :class:`Checkpoint`."""
    if checkpoint.payload_codec != GYMNASIUM_PAYLOAD_CODEC:
        raise ValueError(
            f"unsupported payload codec: {checkpoint.payload_codec!r} "
            f"(expected {GYMNASIUM_PAYLOAD_CODEC!r})"
        )
    env_dict = pickle.loads(checkpoint.opaque_payload)
    unwrapped = getattr(env, "unwrapped", env)
    # Preserve existing viewer/renderer if present.
    for non_picklable_key in ("viewer", "screen", "renderer"):
        if non_picklable_key in unwrapped.__dict__:
            env_dict[non_picklable_key] = unwrapped.__dict__[non_picklable_key]
    unwrapped.__dict__.clear()
    unwrapped.__dict__.update(env_dict)

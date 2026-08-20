"""Capability profile for PettingZoo Parallel adapter (A-01).

Declares the :class:`CapabilityProfile` for PettingZoo Parallel
environments wrapped as kernel :class:`WorldProtocol`. The profile is
static for the lifetime of the adapter and is cached on every
:class:`StateView` and :class:`Checkpoint`.

Design (per main plan §12.2 and §12.7 M2 Gate (b)):
- PettingZoo MPE environments (Simple Spread / Simple Tag) expose:
  - agent positions + velocities → ``entities=True``
  - NO WST/ECS field state → ``fields=False``
  - NO WorldGraph edges → ``relations=False``
  - NO ObjectRegistry → ``registries=False``
  - NO birth/death bookkeeping → ``population=False``
  - NO event context → ``events=False``
- PettingZoo envs support exact restore via ``reset(seed)`` +
  deterministic replay; ``exact_restore=True``.
- PettingZoo MPE envs are deterministic given (seed, action sequence);
  ``executable_deterministic_replay=True``.
- Authority is ``rule`` (PettingZoo env is a rule engine, not learned).
- Transitions are deterministic.

Capability reconciliation (per §12.7 (b)):
- The capability profile MUST match the env's actual capability. If a
  PettingZoo env exposes graph-like structure (e.g., a non-MPE env with
  explicit edges), a separate capability profile should be defined
  rather than forcing the MPE profile.
"""
from __future__ import annotations

from typing import Any

from worldloop_kernel.canonical import hash_state
from worldloop_kernel.capability import CapabilityProfile

__all__ = [
    "make_pettingzoo_mpe_capability",
    "make_pettingzoo_capability",
    "is_exact_restore_verified",
    "verify_immediate_restore",
    "EXACT_RESTORE_VERIFIED_ENV_FAMILIES",
    "PETTINGZOO_WORLD_ID",
    "PETTINGZOO_WORLD_VERSION",
    "PETTINGZOO_PAYLOAD_CODEC",
    "PETTINGZOO_SCENARIO_ID_MPE",
    "PETTINGZOO_ENTITY_SCHEMA_ID_MPE",
]

# ---------------------------------------------------------------------------
# World identity constants
# ---------------------------------------------------------------------------

PETTINGZOO_WORLD_ID = "worldloop-pettingzoo-parallel"
PETTINGZOO_WORLD_VERSION = "0.1.0"
PETTINGZOO_PAYLOAD_CODEC = "pickle+v1"

# Schema IDs (stable within a run; embedded in StateView slots).
PETTINGZOO_SCENARIO_ID_MPE = "pettingzoo-mpe-scenario"
PETTINGZOO_ENTITY_SCHEMA_ID_MPE = "pettingzoo-mpe-entity-v1"


def make_pettingzoo_mpe_capability() -> CapabilityProfile:
    """Return the canonical :class:`CapabilityProfile` for PettingZoo MPE envs.

    PettingZoo MPE environments (Simple Spread, Simple Tag) expose only
    the ``entities`` slot (agent positions/velocities + landmark positions).
    Other slots are ``False`` and the corresponding ``missing_mask`` entries
    MUST be ``True`` in :class:`StateView`.

    Exact restore and deterministic replay are both supported (PettingZoo
    MPE is deterministic given seed + action sequence). Authority is
    ``rule`` (MPE is a rule engine, not learned).
    """
    return CapabilityProfile(
        fields=False,
        entities=True,
        relations=False,
        registries=False,
        population=False,
        events=False,
        exact_restore=True,
        executable_deterministic_replay=True,
        authority="rule",
        ground_truth=True,
        transition_mode="deterministic",
    )


# ---------------------------------------------------------------------------
# Exact-restore allowlist (Phase 5 §10.4 capability layering)
# ---------------------------------------------------------------------------

#: Env families whose ``exact_restore`` claim has been MECHANICALLY
#: verified: checkpoint → immediate restore → observe hash equality
#: (see :func:`verify_immediate_restore`). Verification evidence lives
#: in ``tests/test_joint_adapter.py`` (per-family immediate-restore
#: tests) and in the joint pilot's ``restore_probe`` artifacts. Adding
#: an entry WITHOUT such evidence violates the honesty discipline —
#: never pre-register a family speculatively.
EXACT_RESTORE_VERIFIED_ENV_FAMILIES: tuple[str, ...] = (
    "mpe2/simple_spread_v3",
    "mpe2/simple_tag_v3",
)


def is_exact_restore_verified(env_family: str) -> bool:
    """``True`` iff ``env_family`` is in the verified allowlist."""
    return env_family in EXACT_RESTORE_VERIFIED_ENV_FAMILIES


def make_pettingzoo_capability(env_family: str) -> CapabilityProfile:
    """Capability profile layered by the exact-restore allowlist.

    Generic PettingZoo envs (families NOT in
    :data:`EXACT_RESTORE_VERIFIED_ENV_FAMILIES`) get
    ``exact_restore=False`` / ``executable_deterministic_replay=False``
    — the adapter's pickle-based checkpoint MAY work for them, but the
    claim is only made once verified per family. Verified families get
    the full MPE profile.
    """
    verified = is_exact_restore_verified(env_family)
    return CapabilityProfile(
        fields=False,
        entities=True,
        relations=False,
        registries=False,
        population=False,
        events=False,
        exact_restore=verified,
        executable_deterministic_replay=verified,
        authority="rule",
        ground_truth=True,
        transition_mode="deterministic",
    )


def verify_immediate_restore(world: Any) -> tuple[bool, str]:
    """Mechanically verify checkpoint → immediate restore hash equality.

    Takes a checkpoint of the world's CURRENT state, immediately
    restores from it, and compares ``hash_state(world.observe())``
    against ``hash_state(checkpoint.state_view)``. This is the E-G4
    admission test for :data:`EXACT_RESTORE_VERIFIED_ENV_FAMILIES`.

    Note: this probe compares STATE hashes directly instead of calling
    the kernel's ``verify_checkpoint_restoration`` — the adapter uses
    its own checkpoint checksum format
    (``sha256:<state_view_hash>:<payload_hash>``, see
    ``checkpoint_mapper.compute_checksum``), which the kernel's
    checksum recomputation would flag as a false mismatch.

    Returns ``(ok, message)``; ``message`` is empty on success.
    """
    try:
        checkpoint = world.checkpoint()
    except Exception as exc:  # noqa: BLE001 — probe must not raise
        return False, f"world.checkpoint raised: {exc!r}"
    try:
        world.restore(checkpoint)
    except Exception as exc:  # noqa: BLE001
        return False, f"world.restore raised: {exc!r}"
    try:
        observed_hash = hash_state(world.observe())
    except Exception as exc:  # noqa: BLE001
        return False, f"world.observe raised: {exc!r}"
    expected_hash = hash_state(checkpoint.state_view)
    if observed_hash != expected_hash:
        return (
            False,
            f"state hash mismatch after immediate restore: observed "
            f"{observed_hash} != checkpoint.state_view {expected_hash}",
        )
    return True, ""

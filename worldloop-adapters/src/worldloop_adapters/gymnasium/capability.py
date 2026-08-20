"""Capability profile for Gymnasium adapter (A-05).

Declares the :class:`CapabilityProfile` for Gymnasium single-agent
environments wrapped as kernel :class:`WorldProtocol`.

Design (per main plan §12.2 and §12.7 M2 Gate (b)):
- Gymnasium classic control envs (CartPole, MountainCar, ...) expose:
  - agent state (position, velocity, angle, ...) → ``entities=True``
    (single entity, the agent itself)
  - NO WST/ECS field state → ``fields=False``
  - NO WorldGraph edges → ``relations=False``
  - NO ObjectRegistry → ``registries=False``
  - NO birth/death bookkeeping → ``population=False``
  - NO event context → ``events=False``
- Gymnasium envs support exact restore via ``reset(seed)`` + deterministic
  replay; ``exact_restore=True``.
- Gymnasium envs are deterministic given (seed, action sequence);
  ``executable_deterministic_replay=True``.
- Authority is ``rule`` (Gymnasium env is a rule engine).
- Transitions are deterministic.
"""
from __future__ import annotations

from worldloop_kernel.capability import CapabilityProfile

__all__ = [
    "make_gymnasium_discrete_capability",
    "GYMNASIUM_WORLD_ID",
    "GYMNASIUM_WORLD_VERSION",
    "GYMNASIUM_PAYLOAD_CODEC",
    "GYMNASIUM_SCENARIO_ID",
    "GYMNASIUM_ENTITY_SCHEMA_ID",
]

# ---------------------------------------------------------------------------
# World identity constants
# ---------------------------------------------------------------------------

GYMNASIUM_WORLD_ID = "worldloop-gymnasium"
GYMNASIUM_WORLD_VERSION = "0.1.0"
GYMNASIUM_PAYLOAD_CODEC = "pickle+v1"

GYMNASIUM_SCENARIO_ID = "gymnasium-classic-control"
GYMNASIUM_ENTITY_SCHEMA_ID = "gymnasium-entity-v1"


def make_gymnasium_discrete_capability() -> CapabilityProfile:
    """Return the canonical :class:`CapabilityProfile` for Gymnasium discrete envs.

    Gymnasium classic control envs expose only the ``entities`` slot
    (single agent state vector). Other slots are ``False``.
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

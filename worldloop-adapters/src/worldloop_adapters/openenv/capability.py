"""Capability profile for OpenEnv adapter (A-06).

OpenEnv exposes reset/step/state with single-agent semantics similar to
Gymnasium. The capability profile mirrors the Gymnasium profile but uses
OpenEnv-specific world identity.
"""
from __future__ import annotations

from worldloop_kernel.capability import CapabilityProfile

__all__ = [
    "make_openenv_capability",
    "OPENENV_WORLD_ID",
    "OPENENV_WORLD_VERSION",
    "OPENENV_PAYLOAD_CODEC",
    "OPENENV_SCENARIO_ID",
    "OPENENV_ENTITY_SCHEMA_ID",
]

OPENENV_WORLD_ID = "worldloop-openenv"
OPENENV_WORLD_VERSION = "0.1.0"
OPENENV_PAYLOAD_CODEC = "pickle+v1"

OPENENV_SCENARIO_ID = "openenv-scenario"
OPENENV_ENTITY_SCHEMA_ID = "openenv-entity-v1"


def make_openenv_capability() -> CapabilityProfile:
    """Return the canonical :class:`CapabilityProfile` for OpenEnv envs.

    OpenEnv exposes only the ``entities`` slot (single agent state).
    Other slots are ``False``.
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
